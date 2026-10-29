"""Card graph ring detection using WCC and Louvain algorithms as-of timestamp t.

Builds card -- shared element edges (non-generic device_profiles, id_23 proxy
markers + specific device, device + region) over a sliding window (default 30 days).
Applies element cardinality capping (default max 15 cards per element) and
ring size capping (max 30 cards) to eliminate generic mega-clusters.
Identifies rings as components/communities with 3..30 cards and >= 1 card with
prior confirmed fraud or strong risk behaviour.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from assay.detectors import is_generic_device
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import closed_cases, tx_all


@dataclass
class Ring:
    ring_id: str
    cards: list[str]
    shared_elements: list[str]
    element_cardinalities: dict[str, int]
    txn_ids: dict[str, list[int]]
    first_seen: dict[str, str]
    total_amount: float
    members_with_prior_fraud: list[str]
    member_prior_case_ids: dict[str, list[str]]
    algorithm: str
    attestation: Attestation

    def to_dict(self) -> dict:
        return {
            "ring_id": self.ring_id,
            "cards": self.cards,
            "shared_elements": self.shared_elements,
            "element_cardinalities": self.element_cardinalities,
            "txn_ids": self.txn_ids,
            "first_seen": self.first_seen,
            "total_amount": round(self.total_amount, 2),
            "members_with_prior_fraud": self.members_with_prior_fraud,
            "member_prior_case_ids": self.member_prior_case_ids,
            "algorithm": self.algorithm,
            "attestation": self.attestation.to_dict(),
        }


def build_card_graph(
    t: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
    window_days: int = 30,
    cardinality_cap: int = 15,
    extra_edges: Sequence[tuple[str, str]] | None = None,
) -> tuple[nx.Graph, dict[str, list[str]], dict[str, int]]:
    """Build projected card graph as-of t over window_days with element cardinality capping.

    Cards are linked if they share specific non-generic elements (device profiles,
    id_23 proxy + specific device, device + region) whose cardinality in the window
    does not exceed cardinality_cap (default 15).
    """
    if tx is None:
        tx = tx_all()

    t_end = pd.to_datetime(t)
    t_start = t_end - pd.Timedelta(days=window_days)

    tx_win = tx[(tx.t <= t_end) & (tx.t >= t_start)]

    element_cards: dict[str, set[str]] = defaultdict(set)

    # 1. Specific non-generic device profiles (no '?' wildcards, not generic)
    dev_mask = tx_win.device_profile.notna() & (tx_win.device_profile != "")
    if dev_mask.any():
        dev_df = tx_win[dev_mask]
        for card_id, dev in zip(dev_df.card_id, dev_df.device_profile):
            s_dev = str(dev)
            if not is_generic_device(s_dev) and s_dev.count("?") == 0:
                element_cards[f"dev:{s_dev}"].add(str(card_id))

    # 2. id_23 proxy markers combined with a specific non-generic device profile
    if "id_23" in tx_win.columns:
        id23_mask = tx_win.id_23.notna() & (tx_win.id_23 != "")
        if id23_mask.any():
            id23_df = tx_win[id23_mask]
            for card_id, id23, dev in zip(id23_df.card_id, id23_df.id_23, id23_df.device_profile):
                s_id23 = str(id23).strip()
                s_dev = str(dev) if pd.notna(dev) else ""
                if s_id23 and s_id23.lower() != "nan" and not is_generic_device(s_dev) and s_dev.count("?") == 0:
                    element_cards[f"id23:{s_id23}:{s_dev}"].add(str(card_id))

    # 3. Device profile + addr1 billing region
    if "addr1" in tx_win.columns:
        addr_mask = tx_win.addr1.notna() & tx_win.device_profile.notna()
        if addr_mask.any():
            addr_df = tx_win[addr_mask]
            for card_id, dev, addr in zip(addr_df.card_id, addr_df.device_profile, addr_df.addr1):
                s_dev = str(dev)
                if not is_generic_device(s_dev) and s_dev.count("?") == 0:
                    try:
                        i_addr = int(addr)
                        element_cards[f"dev_addr:{s_dev}:{i_addr}"].add(str(card_id))
                    except (ValueError, TypeError):
                        pass

    # 4. Extra edges supplied by caller
    if extra_edges:
        for u, v in extra_edges:
            su, sv = str(u), str(v)
            if su.startswith("C") and sv.startswith("C"):
                element_cards[f"extra:{su}_{sv}"].add(su)
                element_cards[f"extra:{su}_{sv}"].add(sv)
            else:
                element_cards[sv].add(su)

    # Filter elements by cardinality cap
    filtered_element_cards: dict[str, set[str]] = {}
    element_cardinality: dict[str, int] = {}
    for elem, cards in element_cards.items():
        card_count = len(cards)
        element_cardinality[elem] = card_count
        if 2 <= card_count <= cardinality_cap:
            filtered_element_cards[elem] = cards

    G = nx.Graph()
    card_elements: dict[str, set[str]] = defaultdict(set)

    for elem, cards in filtered_element_cards.items():
        card_list = sorted(cards)
        for c in card_list:
            card_elements[c].add(elem)
            G.add_node(c)
        for i in range(len(card_list)):
            for j in range(i + 1, len(card_list)):
                c1, c2 = card_list[i], card_list[j]
                if G.has_edge(c1, c2):
                    G[c1][c2]["shared_elements"].add(elem)
                    G[c1][c2]["weight"] += 1
                else:
                    G.add_edge(c1, c2, shared_elements={elem}, weight=1)

    card_to_elements = {c: sorted(elems) for c, elems in card_elements.items()}
    return G, card_to_elements, element_cardinality


def ring_for(
    card_id: str,
    t: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
    window_days: int = 30,
    cardinality_cap: int = 15,
    max_ring_size: int = 30,
    algo: str = "wcc",
    extra_edges: Sequence[tuple[str, str]] | None = None,
) -> Ring | None:
    """Find the fraud ring for card_id as-of time t.

    Returns Ring dataclass with Attestation if card_id belongs to a component/community
    with 3..max_ring_size cards and >= 1 card with prior confirmed fraud or strong behaviour.
    Returns None if no qualifying ring exists.
    """
    if tx is None:
        tx = tx_all()
    if cc is None:
        cc = closed_cases()

    t_end = pd.to_datetime(t)
    t_start = t_end - pd.Timedelta(days=window_days)

    G, _, element_cardinality = build_card_graph(
        t_end, tx=tx, window_days=window_days, cardinality_cap=cardinality_cap, extra_edges=extra_edges
    )

    if card_id not in G:
        return None

    algo_lower = algo.lower()
    if algo_lower == "wcc":
        ring_cards_set = nx.node_connected_component(G, card_id)
    elif algo_lower == "louvain":
        comp_nodes = nx.node_connected_component(G, card_id)
        if len(comp_nodes) < 3:
            return None
        G_sub = G.subgraph(comp_nodes)
        communities = nx.community.louvain_communities(G_sub, seed=42, weight="weight")
        ring_cards_set = set()
        for comm in communities:
            if card_id in comm:
                ring_cards_set = comm
                break
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Must be 'wcc' or 'louvain'.")

    ring_cards = sorted(list(ring_cards_set))
    if len(ring_cards) < 3 or len(ring_cards) > max_ring_size:
        return None

    # Prior confirmed fraud as-of t
    cc_past = cc[pd.to_datetime(cc.closed_at) <= t_end]
    cc_past_fraud = cc_past[cc_past.outcome == "confirmed_fraud"]
    prior_fraud_all = set(cc_past_fraud.card_id)
    prior_fraud_members = sorted([c for c in ring_cards if c in prior_fraud_all])

    member_prior_case_ids: dict[str, list[str]] = {}
    for c in ring_cards:
        c_cases = cc_past_fraud[cc_past_fraud.card_id == c].case_id.tolist()
        if c_cases:
            member_prior_case_ids[c] = sorted(c_cases)

    # Check strong behaviour if no prior confirmed fraud member
    tx_win = tx[(tx.t <= t_end) & (tx.t >= t_start) & (tx.card_id.isin(ring_cards))]
    has_strong_behavior = len(prior_fraud_members) > 0
    if not has_strong_behavior:
        if len(tx_win) > 0 and (tx_win.risk_score >= 0.70).any():
            has_strong_behavior = True

    if not has_strong_behavior:
        return None

    # Shared elements between members in ring
    shared_elems: set[str] = set()
    for i in range(len(ring_cards)):
        for j in range(i + 1, len(ring_cards)):
            c1, c2 = ring_cards[i], ring_cards[j]
            if G.has_edge(c1, c2):
                shared_elems.update(G[c1][c2]["shared_elements"])
    sorted_elements = sorted(shared_elems)
    elem_cardinalities = {elem: element_cardinality.get(elem, 0) for elem in sorted_elements}

    txn_ids: dict[str, list[int]] = {}
    first_seen: dict[str, str] = {}
    total_amount = 0.0

    for c in ring_cards:
        c_tx = tx_win[tx_win.card_id == c]
        if len(c_tx) > 0:
            txn_ids[c] = c_tx.TransactionID.astype(int).tolist()
            first_seen[c] = c_tx.t.min().strftime("%Y-%m-%d %H:%M:%S")
            total_amount += float(c_tx.amount.sum())
        else:
            txn_ids[c] = []
            first_seen[c] = "N/A"

    ring_id = f"RING-{algo_lower.upper()}-{hashlib.sha256(','.join(ring_cards).encode()).hexdigest()[:8]}"

    all_txn_ids = sorted(tid for tids in txn_ids.values() for tid in tids)
    customer_ids = sorted(tx_win.customer_id.unique().tolist()) if len(tx_win) > 0 else []
    prior_case_ids = sorted([cid for cids in member_prior_case_ids.values() for cid in cids])

    attestation = Attestation(
        what=f"Card {card_id} is in {algo_lower} ring {ring_id} containing {len(ring_cards)} cards (${total_amount:,.2f} vol, {len(prior_fraud_members)} prior fraud members)",
        evidence=Evidence(
            txn_ids=all_txn_ids,
            card_ids=ring_cards,
            customer_ids=customer_ids,
            case_ids=prior_case_ids,
        ),
        query="ring_for",
        params={
            "card_id": card_id,
            "t": t_end.strftime("%Y-%m-%d %H:%M:%S"),
            "window_days": window_days,
            "cardinality_cap": cardinality_cap,
            "max_ring_size": max_ring_size,
            "algo": algo_lower,
        },
    )

    return Ring(
        ring_id=ring_id,
        cards=ring_cards,
        shared_elements=sorted_elements,
        element_cardinalities=elem_cardinalities,
        txn_ids=txn_ids,
        first_seen=first_seen,
        total_amount=round(total_amount, 2),
        members_with_prior_fraud=prior_fraud_members,
        member_prior_case_ids=member_prior_case_ids,
        algorithm=algo_lower,
        attestation=attestation,
    )


def ring_replay(ring: Ring, tx: pd.DataFrame | None = None) -> list[dict]:
    """Time-slider replay showing how the ring forms over time frame-by-frame.

    Every frame is built dynamically from attested transactions.
    Returns list of dicts:
      {"date": "YYYY-MM-DD", "cards_joined": list[str], "shared_elements": list[str], "cumulative_amount": float}
    """
    if tx is None:
        tx = tx_all()

    cards_set = set(ring.cards)
    ring_tx = tx[tx.card_id.isin(cards_set)].sort_values("t").copy()

    if len(ring_tx) == 0:
        return []

    ring_tx["date_str"] = ring_tx.t.dt.strftime("%Y-%m-%d")

    frames: list[dict] = []
    seen_cards: set[str] = set()
    cum_amount = 0.0

    for date_str, group in ring_tx.groupby("date_str", sort=True):
        cum_amount += float(group.amount.sum())
        seen_cards.update(group.card_id.unique())

        sub_tx = tx[(tx.card_id.isin(seen_cards)) & (tx.t <= group.t.max())]

        active_elements: set[str] = set()
        for elem in ring.shared_elements:
            if elem.startswith("dev:"):
                dev_val = elem[4:]
                if (sub_tx.device_profile == dev_val).any():
                    active_elements.add(elem)
            elif elem.startswith("id23:"):
                parts = elem[5:].split(":", 1)
                id23_val = parts[0]
                dev_val = parts[1] if len(parts) > 1 else ""
                match = (sub_tx.id_23 == id23_val) & (sub_tx.device_profile == dev_val)
                if match.any():
                    active_elements.add(elem)
            elif elem.startswith("dev_addr:"):
                parts = elem[9:].rsplit(":", 1)
                dev_val = parts[0]
                try:
                    addr_val = float(parts[1])
                    match = (sub_tx.device_profile == dev_val) & (sub_tx.addr1 == addr_val)
                    if match.any():
                        active_elements.add(elem)
                except (ValueError, IndexError):
                    pass

        frames.append({
            "date": date_str,
            "cards_joined": sorted(list(seen_cards)),
            "shared_elements": sorted(list(active_elements)),
            "cumulative_amount": round(cum_amount, 2),
        })

    return frames


def run_ring_ablation(
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
    window_days: int = 30,
    cardinality_cap: int = 15,
    max_ring_size: int = 30,
    algo: str = "wcc",
    sample_size_legit: int = 5000,
    seed: int = 42,
) -> dict:
    """Backtest ring detection with a proper 2x2 contingency matrix and LR calculation.

    Evaluates:
      Row 1: In ring with prior fraud
      Row 2: Not in ring with prior fraud
      Col 1: Fraud episode (closed confirmed fraud cases)
      Col 2: Legit card-day (sampled Jul-Oct 2016 legitimate card activity)

    Returns Likelihood Ratio LR = P(in ring | fraud) / P(in ring | legit) with 95% CI.
    """
    if tx is None:
        tx = tx_all()
    if cc is None:
        cc = closed_cases()

    # Positives: Confirmed fraud cases
    cc_fraud = cc[cc.outcome == "confirmed_fraud"].copy()
    cc_fraud["opened_dt"] = pd.to_datetime(cc_fraud.opened_at)

    fraud_card_set = set(cc_fraud.card_id)

    # Negatives: Legitimate card-days from Jul-Oct 2016
    tx_jul_oct = tx[tx.month_src == "r1"].copy()
    legit_tx = tx_jul_oct[~tx_jul_oct.card_id.isin(fraud_card_set)]

    rng = np.random.default_rng(seed)
    legit_indices = rng.choice(len(legit_tx), size=min(sample_size_legit, len(legit_tx)), replace=False)
    sample_legit_tx = legit_tx.iloc[legit_indices]

    fraud_samples = [(str(row.card_id), row.opened_dt) for row in cc_fraud.itertuples()]
    legit_samples = [(str(row.card_id), row.t) for row in sample_legit_tx.itertuples()]

    sample_dates = sorted(list({s[1].floor("D") for s in fraud_samples + legit_samples}))

    graphs: dict[pd.Timestamp, tuple[nx.Graph, set[str]]] = {}
    for d in sample_dates:
        t_snap = d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        G, _, _ = build_card_graph(t_snap, tx=tx, window_days=window_days, cardinality_cap=cardinality_cap)
        cc_past = cc[pd.to_datetime(cc.closed_at) <= t_snap]
        prior_fraud = set(cc_past[cc_past.outcome == "confirmed_fraud"].card_id)
        graphs[d] = (G, prior_fraud)

    def is_in_qualifying_ring(card_id: str, t_snap: pd.Timestamp) -> bool:
        d = t_snap.floor("D")
        G, prior_fraud = graphs[d]

        if card_id not in G:
            return False

        if algo.lower() == "wcc":
            nodes = nx.node_connected_component(G, card_id)
        else:
            comp_nodes = nx.node_connected_component(G, card_id)
            if len(comp_nodes) >= 3:
                G_sub = G.subgraph(comp_nodes)
                comms = nx.community.louvain_communities(G_sub, seed=42, weight="weight")
                nodes = set()
                for comm in comms:
                    if card_id in comm:
                        nodes = comm
                        break
            else:
                nodes = set()

        if 3 <= len(nodes) <= max_ring_size:
            return any(c in prior_fraud for c in nodes)
        return False

    n_ring_fraud = sum(1 for cid, t_snap in fraud_samples if is_in_qualifying_ring(cid, t_snap))
    n_no_ring_fraud = len(fraud_samples) - n_ring_fraud

    n_ring_legit = sum(1 for cid, t_snap in legit_samples if is_in_qualifying_ring(cid, t_snap))
    n_no_ring_legit = len(legit_samples) - n_ring_legit

    N_fraud = len(fraud_samples)
    N_legit = len(legit_samples)

    p_in_ring_given_fraud = n_ring_fraud / N_fraud if N_fraud > 0 else 0.0
    p_in_ring_given_legit = n_ring_legit / N_legit if N_legit > 0 else 0.0

    # Laplace smoothing if n_ring_legit is 0
    p_legit_smooth = (n_ring_legit + 0.5) / (N_legit + 1.0)
    p_fraud_smooth = (n_ring_fraud + 0.5) / (N_fraud + 1.0)

    lr = p_fraud_smooth / p_legit_smooth

    # 95% Confidence Interval for LR
    se_ln_lr = math.sqrt(
        (1.0 / (n_ring_fraud + 0.5)) - (1.0 / (N_fraud + 1.0))
        + (1.0 / (n_ring_legit + 0.5)) - (1.0 / (N_legit + 1.0))
    )
    ci_lower = math.exp(math.log(lr) - 1.96 * se_ln_lr)
    ci_upper = math.exp(math.log(lr) + 1.96 * se_ln_lr)

    return {
        "algo": algo,
        "contingency_2x2": {
            "fraud_episodes": {"in_ring": n_ring_fraud, "not_in_ring": n_no_ring_fraud, "total": N_fraud},
            "legit_card_days": {"in_ring": n_ring_legit, "not_in_ring": n_no_ring_legit, "total": N_legit},
        },
        "probabilities": {
            "p_in_ring_given_fraud": round(p_in_ring_given_fraud, 4),
            "p_in_ring_given_legit": round(p_in_ring_given_legit, 4),
        },
        "likelihood_ratio": round(lr, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
    }
