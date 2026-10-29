"""C5 -- Put TigerVector precedent on the decision path (`vector_precedent`).

Embeds a case's 24-hour episode window strictly as-of `opened_at` using the
24-dimensional behaviour embedding from `assay.r2.vector`, retrieves the `k`
nearest closed cases with `closed_at <= opened_at` (via TigerGraph when
`ASSAY_TG=1`, failing closed on any error; otherwise `LocalVectorIndex`), and
computes the Laplace-smoothed fraud share among the `k` neighbours.

Evaluates both:
  1. `pure`: pure cosine similarity over `ClosedCase.behaviour` (`closed_at <= opened_at`).
  2. `fused`: cosine distance boosted (`-0.25`) ONLY when a candidate closed case
     shares a SPECIFIC element with the case card as-of `opened_at`:
       - a `device_profile` seen on `<= 15` cards as-of `opened_at`,
       - a non-freemail `R_emaildomain` seen on `<= 15` cards as-of `opened_at`, or
       - the candidate case's `connected_card_ids` (or `card_id`) contains the card.

Backtests as-of over Aug-Dec 2016 closed cases (leave-one-out, `closed_at <= opened_at`)
AND a fixed-seed sample of `>= 2,000` legitimate card-days (from cards never named
in any closed case or exam case). Reports both LR vs cleared closed cases and LR vs
legitimate card-days.

STRICT SELECTION CONTRACT:
A candidate `(variant, k, theta)` MUST satisfy BOTH:
  - `P(fire | cleared case) <= 8%` (`MAX_CLEARED_CASE_FPR = 0.08`), AND
  - `P(fire | legit card-day) <= 3%` (`MAX_LEGIT_CARD_DAY_FPR = 0.03`).
Any candidate failing either constraint is strictly ineligible (no fallback).
If no candidate qualifies, `enabled = False` and `VectorPrecedentVote.run()` emits `[]`.
When a candidate qualifies, `weight` is set to the 95% CI lower bound of
`P(fire | fraud case) / P(fire | cleared case)` (Laplace).

The selected operating point is persisted to `data/stage/r2/vector_vote_selection.json`
keyed by `(dataset_tag(), code_hash(vector_vote.py))` so agent startup reuses the
verified selection without re-running the full backtest on every process start.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import pandas as pd

from assay.data import ROOT
from assay.r2.attest import Attestation, Evidence, dataset_tag
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import R2_STAGE, case_pack, closed_cases, tx_all
from assay.r2.vector import (
    EMBED_DIM,
    TG_R2_DIR,
    LocalVectorIndex,
    Standardizer,
    TigerGraphUnavailableError,
    VectorNeighbor,
    _build_tx_index,
    _count_asof_shared_dev_cards,
    _raw_embed_one,
    _select_asof_window,
    _tg_run_query,
    _to_epoch_sec,
    embed_case,
    embed_closed_cases,
    export_tg_vectors,
)

FREEMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "icloud.com",
    "live.com",
    "msn.com",
    "ymail.com",
    "mail.com",
    "me.com",
    "comcast.net",
    "att.net",
    "verizon.net",
    "sbcglobal.net",
    "cox.net",
    "bellsouth.net",
    "rocketmail.com",
    "mac.com",
    "earthlink.net",
    "charter.net",
    "optonline.net",
    "shaw.ca",
    "rogers.com",
    "sympatico.ca",
    "juno.com",
    "netzero.net",
    "prodigy.net",
    "gmail",
    "yahoo.es",
    "yahoo.co.uk",
    "yahoo.fr",
    "yahoo.de",
    "yahoo.co.jp",
    "hotmail.co.uk",
    "hotmail.fr",
    "hotmail.es",
    "hotmail.de",
    "live.com.mx",
    "live.fr",
    "outlook.es",
    "anonymous.com",
})

K_GRID: tuple[int, ...] = (5, 10, 20)
MAX_CLEARED_CASE_FPR: float = 0.08
MAX_LEGIT_CARD_DAY_FPR: float = 0.03
SPECIFIC_CARD_CAP: int = 15
FUSED_SPECIFIC_BOOST: float = 0.25

SELECTION_CACHE_PATH: Path = R2_STAGE / "vector_vote_selection.json"
DOC_OUTPUT_PATH: Path = ROOT / "docs" / "r2" / "C5_vector_vote.md"


def _code_hash() -> str:
    """Return short SHA-256 of `vector_vote.py` and `vector.py` so any logic edit invalidates cache."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    vec_file = Path(__file__).resolve().parent.parent / "vector.py"
    if vec_file.exists():
        h.update(vec_file.read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class GridCell:
    variant: str
    k: int
    theta: float
    n_fraud: int
    n_cleared: int
    n_legit_days: int
    fired_fraud: int
    fired_cleared: int
    fired_legit_days: int
    p_fire_fraud: float
    p_fire_cleared: float
    p_fire_legit_day: float
    lr_vs_cleared: float
    ci_cleared_low: float
    ci_cleared_high: float
    lr_vs_legit_day: float
    ci_legit_low: float
    ci_legit_high: float

    @property
    def qualifies(self) -> bool:
        """True iff BOTH P(fire|cleared) <= 8% AND P(fire|legit card-day) <= 3% AND fired_fraud > 0."""
        return (
            self.p_fire_cleared <= MAX_CLEARED_CASE_FPR
            and self.p_fire_legit_day <= MAX_LEGIT_CARD_DAY_FPR
            and self.fired_fraud > 0
        )

    @property
    def ci_low(self) -> float:
        """95% CI lower bound of P(fire|fraud)/P(fire|cleared) (Laplace) used as decision weight."""
        return self.ci_cleared_low

    @property
    def ci_high(self) -> float:
        return self.ci_cleared_high


@dataclass
class VectorVoteBacktestSummary:
    eval_start: str
    eval_end: str
    n_fraud: int
    n_cleared: int
    n_legit_days: int
    grid: list[GridCell]
    enabled: bool
    status_reason: str
    chosen: GridCell | None
    best_pure: GridCell | None
    best_fused: GridCell | None
    stage_trace: dict[str, Any] = field(default_factory=dict)
    caught_case_ids: list[str] = field(default_factory=list)
    missed_case_ids: list[str] = field(default_factory=list)
    cleared_acted_case_ids: list[str] = field(default_factory=list)


def laplace_fraud_score(outcomes: Iterable[str]) -> float:
    """Laplace-smoothed fraud share `(fraud_hits + 1) / (k + 2)` among retrieved neighbours."""
    out_list = list(outcomes)
    k = len(out_list)
    if k == 0:
        return 0.5
    n_fraud = sum(1 for o in out_list if o == "confirmed_fraud")
    return float((n_fraud + 1.0) / (k + 2.0))


def _katz_lr_ci(
    fired_pos: int,
    n_pos: int,
    fired_neg: int,
    n_neg: int,
) -> tuple[float, float, float]:
    """Return Laplace-smoothed LR and its 95% Katz log-method confidence interval `(lr, ci_low, ci_high)`."""
    a = fired_pos + 1.0
    n1 = n_pos + 2.0
    c = fired_neg + 1.0
    n0 = n_neg + 2.0

    p1 = a / n1
    p0 = c / n0
    lr = float(p1 / p0)

    se_log_lr = float(np.sqrt(max((1.0 - p1) / a + (1.0 - p0) / c, 1e-12)))
    ci_low = float(np.exp(np.log(lr) - 1.96 * se_log_lr))
    ci_high = float(np.exp(np.log(lr) + 1.96 * se_log_lr))
    return lr, ci_low, ci_high


@dataclass
class _SpecificEntityIndex:
    """Precomputed lookup for fast as-of specific-element sharing checks (`<= 15` cards)."""

    dev_groups: dict[str, np.ndarray]
    remail_groups: dict[str, np.ndarray]
    t_sec_all: np.ndarray
    card_all: np.ndarray
    case_devs: list[set[str]]
    case_remails: list[set[str]]
    case_conn_cards: list[set[str]]
    dev_to_cases: dict[str, list[int]]
    remail_to_cases: dict[str, list[int]]
    card_to_cases: dict[str, list[int]]

    @classmethod
    def build(cls, cc: pd.DataFrame, tx: pd.DataFrame, base_index: LocalVectorIndex) -> "_SpecificEntityIndex":
        tx_sorted = tx.sort_values("t", kind="stable").reset_index(drop=True)
        remail_arr = tx_sorted["R_emaildomain"].fillna("").astype(str).str.lower().to_numpy()
        non_free_mask = np.array([bool(r and r not in FREEMAIL_DOMAINS) for r in remail_arr], dtype=bool)
        remail_groups = tx_sorted[non_free_mask].groupby(remail_arr[non_free_mask], sort=False).indices

        tid_to_pos = base_index.tid_to_pos
        dev_all = base_index.cols["dev"]
        card_all = base_index.cols["card"]
        t_sec_all = base_index.cols["t_sec"]

        case_devs: list[set[str]] = []
        case_remails: list[set[str]] = []
        case_conn_cards: list[set[str]] = []
        dev_to_cases: dict[str, list[int]] = {}
        remail_to_cases: dict[str, list[int]] = {}
        card_to_cases: dict[str, list[int]] = {}

        cards_col = cc["card_id"].astype(str).tolist()
        conn_col = cc["connected_list"].tolist() if "connected_list" in cc.columns else [[] for _ in range(len(cc))]
        tids_col = cc["txn_list"].tolist() if "txn_list" in cc.columns else [[] for _ in range(len(cc))]

        for i in range(len(cc)):
            pos = [tid_to_pos[int(t)] for t in tids_col[i] if int(t) in tid_to_pos]
            devs = {str(dev_all[p]) for p in pos if dev_all[p]}
            rems = {str(remail_arr[p]) for p in pos if non_free_mask[p]}
            c_set = {cards_col[i]} | {str(x) for x in conn_col[i] if x}

            case_devs.append(devs)
            case_remails.append(rems)
            case_conn_cards.append(c_set)

            for d in devs:
                dev_to_cases.setdefault(d, []).append(i)
            for rm in rems:
                remail_to_cases.setdefault(rm, []).append(i)
            for c_str in c_set:
                card_to_cases.setdefault(c_str, []).append(i)

        return cls(
            dev_groups=base_index.dev_groups,
            remail_groups=remail_groups,
            t_sec_all=t_sec_all,
            card_all=card_all,
            case_devs=case_devs,
            case_remails=case_remails,
            case_conn_cards=case_conn_cards,
            dev_to_cases=dev_to_cases,
            remail_to_cases=remail_to_cases,
            card_to_cases=card_to_cases,
        )

    def _cards_on_group_asof(
        self,
        key: str,
        groups: dict[str, np.ndarray],
        t_cut_sec: int,
        cap: int = SPECIFIC_CARD_CAP,
    ) -> int:
        idx = groups.get(key)
        if idx is None or len(idx) == 0:
            return 0
        cut = int(np.searchsorted(self.t_sec_all[idx], t_cut_sec, side="right"))
        if cut == 0:
            return 0
        seen: set[str] = set()
        for c in self.card_all[idx[:cut]]:
            seen.add(str(c))
            if len(seen) > cap:
                return len(seen)
        return len(seen)

    def specific_shared_cases(
        self,
        card_id: str,
        ep_devs: Iterable[str],
        ep_remails: Iterable[str],
        t_cut_sec: int,
        cap: int = SPECIFIC_CARD_CAP,
    ) -> dict[int, str]:
        """Return `{case_idx: reason}` for closed cases sharing a specific element (`<= cap` cards)
        or connected_card_ids with `card_id` as-of `t_cut_sec`.
        """
        hits: dict[int, str] = {}
        for c_idx in self.card_to_cases.get(card_id, ()):
            hits[c_idx] = f"connected_card:{card_id}"

        for d in ep_devs:
            if not d:
                continue
            n_cards = self._cards_on_group_asof(d, self.dev_groups, t_cut_sec, cap=cap)
            if 1 <= n_cards <= cap:
                for c_idx in self.dev_to_cases.get(d, ()):
                    hits.setdefault(c_idx, f"specific_device:{d[:24]}(cards={n_cards})")

        for rm in ep_remails:
            rm_low = rm.strip().lower() if rm else ""
            if not rm_low or rm_low in FREEMAIL_DOMAINS:
                continue
            n_cards = self._cards_on_group_asof(rm_low, self.remail_groups, t_cut_sec, cap=cap)
            if 1 <= n_cards <= cap:
                for c_idx in self.remail_to_cases.get(rm_low, ()):
                    hits.setdefault(c_idx, f"specific_remail:{rm_low}(cards={n_cards})")

        return hits


@dataclass
class _CachedArtifacts:
    cc: pd.DataFrame
    cc_emb: pd.DataFrame
    standardizer: Standardizer
    index: LocalVectorIndex
    spec_index: _SpecificEntityIndex
    remail_col: np.ndarray


@lru_cache(maxsize=1)
def _default_artifacts() -> _CachedArtifacts:
    tx = tx_all()
    cc = closed_cases()
    cc_emb, std = embed_closed_cases(cc, tx, return_standardizer=True)
    idx = LocalVectorIndex(cc_emb, cc, tx)
    spec_idx = _SpecificEntityIndex.build(cc, tx, idx)
    tx_sorted = tx.sort_values("t", kind="stable").reset_index(drop=True)
    remail_col = tx_sorted["R_emaildomain"].fillna("").astype(str).str.lower().to_numpy()
    return _CachedArtifacts(
        cc=cc,
        cc_emb=cc_emb,
        standardizer=std,
        index=idx,
        spec_index=spec_idx,
        remail_col=remail_col,
    )


def _enrich_tg_neighbors(
    neighbors: list[VectorNeighbor],
    idx: LocalVectorIndex,
) -> list[VectorNeighbor]:
    """Ensure every `VectorNeighbor` returned by TigerGraph carries its `outcome` and
    `pattern` by joining the returned neighbour `case_id` to static closed-case labels
    if needed.
    """
    enriched: list[VectorNeighbor] = []
    for n in neighbors:
        c_pos = idx.case_to_idx.get(n.case_id)
        outcome = n.outcome if n.outcome in ("confirmed_fraud", "cleared") else (
            str(idx.outcomes[c_pos]) if c_pos is not None else n.outcome
        )
        pattern = n.pattern if n.pattern else (
            str(idx.patterns[c_pos]) if c_pos is not None else n.pattern
        )
        enriched.append(
            VectorNeighbor(
                case_id=n.case_id,
                outcome=outcome,
                pattern=pattern,
                distance=n.distance,
                connecting_path=n.connecting_path,
                shared_cards=n.shared_cards,
                shared_devices=n.shared_devices,
            )
        )
    return enriched


def _sample_legit_card_days(
    tx: pd.DataFrame,
    cc: pd.DataFrame,
    cp: pd.DataFrame | None = None,
    n_samples: int = 2400,
    seed: int = 42,
    start_date: str = "2016-08-01",
    end_date: str = "2017-01-01",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sample `n_samples` legitimate card-days from cards NEVER named in any closed case
    (as primary or connected card) or exam case, within `[start_date, end_date)`.
    Each card-day's `opened_at` is the timestamp of the card's last transaction on that UTC day.
    Returns `(sampled_df, filter_counts)`.
    """
    if cp is None:
        cp = case_pack()

    cased_cards: set[str] = set(cc["card_id"].dropna().astype(str).tolist())
    if "connected_list" in cc.columns:
        for cl in cc["connected_list"]:
            cased_cards.update(str(x) for x in cl if x)
    elif "connected_card_ids" in cc.columns:
        for s in cc["connected_card_ids"].dropna().astype(str):
            cased_cards.update(x for x in s.split("|") if x)
    cased_cards.update(cp["card_id"].dropna().astype(str).tolist())

    t_start = pd.Timestamp(start_date)
    t_end = pd.Timestamp(end_date)
    in_window = tx[(tx["t"] >= t_start) & (tx["t"] < t_end)]
    sub = in_window[~in_window["card_id"].isin(cased_cards)].copy()
    sub["day"] = sub["t"].dt.floor("D")

    card_days = (
        sub.sort_values("t", kind="stable")
        .groupby(["card_id", "day"], sort=False, as_index=False)
        .last()[["card_id", "customer_id", "day", "TransactionID", "t"]]
    )
    card_days = card_days.sort_values(["t", "card_id"], kind="stable").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    take = min(n_samples, len(card_days))
    chosen_idx = np.sort(rng.choice(len(card_days), size=take, replace=False))
    sampled = card_days.iloc[chosen_idx].reset_index(drop=True)

    trace = {
        "total_tx_rows": int(len(tx)),
        "tx_in_eval_window": int(len(in_window)),
        "distinct_cased_or_exam_cards_excluded": int(len(cased_cards)),
        "uncased_tx_in_eval_window": int(len(sub)),
        "total_uncased_card_days_in_window": int(len(card_days)),
        "sampled_legit_card_days": int(len(sampled)),
    }
    return sampled, trace


def _embed_legit_card_days(
    sampled_days: pd.DataFrame,
    art: _CachedArtifacts,
) -> tuple[np.ndarray, np.ndarray, list[set[str]], list[set[str]]]:
    """Vectorised/fast embedding of sampled legitimate card-days using the exact same
    24-hour episode window and prior history logic as `embed_closed_cases`.
    Returns `(Z_matrix, t_cut_sec_array, ep_devs_list, ep_remails_list)`.
    """
    idx = art.index
    cols = idx.cols
    t_sec_all = cols["t_sec"]
    dt_all = cols["dt"]
    amt_all = cols["amt"]
    online_all = cols["online"]
    prod_all = cols["prod"]
    dev_all = cols["dev"]
    addr_all = cols["addr"]
    d1_all = cols["d1"]
    card_all = cols["card"]
    remail_all = art.remail_col

    opened_sec = _to_epoch_sec(sampled_days["t"])
    cards = sampled_days["card_id"].astype(str).tolist()
    n = len(sampled_days)

    raw_mat = np.zeros((n, EMBED_DIM), dtype=np.float64)
    ep_devs_list: list[set[str]] = []
    ep_remails_list: list[set[str]] = []

    for i in range(n):
        c_id = cards[i]
        t_cut = int(opened_sec[i])
        c_indices = idx.card_groups.get(c_id)
        ep_idx, hist_idx = _select_asof_window(c_indices, t_cut, t_sec_all, window_hours=24)

        if len(hist_idx) > 0:
            h_amts = amt_all[hist_idx]
            h_onl = online_all[hist_idx]
            h_devs = {str(d) for d in dev_all[hist_idx].tolist() if d}
            h_addrs = {float(a) for a in addr_all[hist_idx].tolist() if a >= 0}
        else:
            h_amts = np.empty(0, dtype=np.float64)
            h_onl = np.empty(0, dtype=np.float64)
            h_devs = set()
            h_addrs = set()

        if len(ep_idx) > 0:
            ep_a = amt_all[ep_idx]
            ep_d = dt_all[ep_idx]
            ep_o = online_all[ep_idx]
            ep_p = prod_all[ep_idx]
            ep_dv = dev_all[ep_idx]
            ep_ad = np.where(addr_all[ep_idx] < 0, np.nan, addr_all[ep_idx])
            ep_d1_arr = d1_all[ep_idx]
            devs_set = {str(d) for d in ep_dv.tolist() if d}
            rems_set = {
                str(r)
                for r in remail_all[ep_idx].tolist()
                if r and r not in FREEMAIL_DOMAINS
            }
            n_shared_cards = _count_asof_shared_dev_cards(
                devs=devs_set,
                card_id=c_id,
                t_cut_sec=t_cut,
                dev_groups=idx.dev_groups,
                t_sec_all=t_sec_all,
                card_all=card_all,
            )
        else:
            ep_a = np.empty(0, dtype=np.float64)
            ep_d = np.empty(0, dtype=np.float64)
            ep_o = np.empty(0, dtype=np.float64)
            ep_p = np.empty(0, dtype=object)
            ep_dv = np.empty(0, dtype=object)
            ep_ad = np.empty(0, dtype=np.float64)
            ep_d1_arr = np.empty(0, dtype=np.float64)
            devs_set = set()
            rems_set = set()
            n_shared_cards = 0

        ep_devs_list.append(devs_set)
        ep_remails_list.append(rems_set)

        raw_mat[i] = _raw_embed_one(
            ep_amts=ep_a,
            ep_dt=ep_d,
            ep_online=ep_o,
            ep_prod=ep_p,
            ep_dev=ep_dv,
            ep_addr=ep_ad,
            ep_d1=ep_d1_arr,
            hist_amts=h_amts,
            hist_online=h_onl,
            hist_dev=h_devs,
            hist_addr=h_addrs,
            asof_shared_dev_cards=n_shared_cards,
        )

    Z = art.standardizer.transform(raw_mat)
    return Z, opened_sec, ep_devs_list, ep_remails_list


def _retrieve_neighbours_for_case(
    qv: np.ndarray,
    card_id: str,
    as_of: pd.Timestamp,
    k: int,
    variant: str,
    art: _CachedArtifacts,
    ep_devs: Iterable[str] = (),
    ep_remails: Iterable[str] = (),
    exclude_case_id: str | None = None,
    use_tg_if_enabled: bool = True,
) -> tuple[list[VectorNeighbor], str]:
    """Retrieve `k` nearest closed-case neighbours with `closed_at <= as_of`.
    Returns `(neighbors, backend_used)`.
    When `os.environ.get("ASSAY_TG") == "1"` and `use_tg_if_enabled=True`, routes
    through TigerGraph (`_tg_run_query("similar_cases_vec", ...)`) and FAILS CLOSED
    (raises `TigerGraphUnavailableError`) if TigerGraph errors out.
    """
    tg_active = bool(use_tg_if_enabled and os.environ.get("ASSAY_TG") == "1")
    idx = art.index
    t_cut_sec = int(_to_epoch_sec(as_of))

    if variant == "pure":
        if tg_active:
            req_k = int(k + 1) if exclude_case_id else int(k)
            nb = idx.similar_cases_vec(
                qv=qv,
                k=req_k,
                as_of=as_of,
                exclude_case_id=exclude_case_id,
                use_tg_if_enabled=True,
            )[:k]
            return _enrich_tg_neighbors(nb, idx), "tigergraph:similar_cases_vec"
        nb = idx.similar_cases_vec(
            qv=qv,
            k=k,
            as_of=as_of,
            exclude_case_id=exclude_case_id,
            use_tg_if_enabled=False,
        )
        return nb, "local:similar_cases_vec"

    # Fused specific-element boost variant
    spec_hits = art.spec_index.specific_shared_cases(
        card_id=card_id,
        ep_devs=ep_devs,
        ep_remails=ep_remails,
        t_cut_sec=t_cut_sec,
        cap=SPECIFIC_CARD_CAP,
    )
    if tg_active:
        req_k = int(max(k * 3, 25))
        base_nb = idx.similar_cases_vec(
            qv=qv,
            k=req_k,
            as_of=as_of,
            exclude_case_id=exclude_case_id,
            use_tg_if_enabled=True,
        )
        q_norm = np.asarray(qv, dtype=np.float64).ravel()
        q_norm = q_norm / max(float(np.linalg.norm(q_norm)), 1e-9)

        cand_map: dict[int, float] = {}
        for n in base_nb:
            if n.case_id in idx.case_to_idx:
                c_idx = idx.case_to_idx[n.case_id]
                cand_map[c_idx] = float(n.distance)
        for c_idx in spec_hits:
            if idx.closed_sec[c_idx] <= t_cut_sec:
                if not exclude_case_id or str(idx.case_ids[c_idx]) != exclude_case_id:
                    d_val = float(max(1.0 - float(idx.mat[c_idx] @ q_norm), 0.0))
                    cand_map.setdefault(c_idx, d_val)

        scored: list[tuple[float, float, int, str]] = []
        for c_idx, d_val in cand_map.items():
            reason = spec_hits.get(c_idx, "vector_only")
            eff = d_val - (FUSED_SPECIFIC_BOOST if c_idx in spec_hits else 0.0)
            scored.append((eff, d_val, c_idx, reason))
        scored.sort(key=lambda x: (x[0], x[1], str(idx.case_ids[x[2]])))
        out = [
            VectorNeighbor(
                case_id=str(idx.case_ids[c_idx]),
                outcome=str(idx.outcomes[c_idx]),
                pattern=str(idx.patterns[c_idx]),
                distance=round(d_val, 6),
                connecting_path=reason,
                shared_cards=tuple(sorted(idx.case_cards[c_idx])),
                shared_devices=tuple(sorted(idx.case_devs[c_idx])),
            )
            for _, d_val, c_idx, reason in scored[:k]
        ]
        return out, "tigergraph:similar_cases_vec+specific_boost"

    elig = np.where(idx.closed_sec <= t_cut_sec)[0]
    if exclude_case_id:
        elig = elig[idx.case_ids[elig] != exclude_case_id]
    if len(elig) == 0:
        return [], "local:similar_cases_vec+specific_boost"

    q_norm = np.asarray(qv, dtype=np.float64).reshape(1, -1)
    q_norm = q_norm / max(float(np.linalg.norm(q_norm)), 1e-9)
    cos_d = np.maximum(1.0 - (idx.mat[elig] @ q_norm.T).ravel(), 0.0)
    eff_d = cos_d.copy()
    if spec_hits:
        for pos_i, c_idx in enumerate(elig):
            if int(c_idx) in spec_hits:
                eff_d[pos_i] -= FUSED_SPECIFIC_BOOST

    order = np.argsort(eff_d, kind="stable")[:k]
    out_loc = [
        VectorNeighbor(
            case_id=str(idx.case_ids[elig[j]]),
            outcome=str(idx.outcomes[elig[j]]),
            pattern=str(idx.patterns[elig[j]]),
            distance=round(float(cos_d[j]), 6),
            connecting_path=spec_hits.get(int(elig[j]), "vector_only"),
            shared_cards=tuple(sorted(idx.case_cards[int(elig[j])])),
            shared_devices=tuple(sorted(idx.case_devs[int(elig[j])])),
        )
        for j in order
    ]
    return out_loc, "local:similar_cases_vec+specific_boost"


@lru_cache(maxsize=1)
def run_vector_vote_backtest(
    eval_start: str = "2016-08-01",
    eval_end: str = "2017-01-01",
    n_legit_days: int = 2400,
    seed: int = 42,
) -> VectorVoteBacktestSummary:
    """Run the full as-of leave-one-out backtest over Aug-Dec 2016 closed cases AND
    a fixed-seed sample of `>= 2,000` legitimate card-days across `k in {5, 10, 20}`
    and `variant in {'pure', 'fused'}`.

    STRICTLY ENFORCES BOTH CONSTRAINTS:
      - `P(fire | cleared case) <= 8%` (`MAX_CLEARED_CASE_FPR = 0.08`)
      - `P(fire | legit card-day) <= 3%` (`MAX_LEGIT_CARD_DAY_FPR = 0.03`)
    Any candidate failing either constraint is ineligible. If no candidate satisfies
    both constraints, `enabled = False` and the detector emits nothing (`[]`).
    """
    tx = tx_all()
    art = _default_artifacts()
    idx = art.index
    spec_idx = art.spec_index

    t_start_sec = int(_to_epoch_sec(eval_start))
    t_end_sec = int(_to_epoch_sec(eval_end))
    after_start = idx.opened_sec >= t_start_sec
    before_end = idx.opened_sec < t_end_sec
    eval_mask = after_start & before_end
    eval_indices = np.where(eval_mask)[0]

    max_k = max(K_GRID)
    n_eval = len(eval_indices)
    y_eval = np.array([1 if str(idx.outcomes[i]) == "confirmed_fraud" else 0 for i in eval_indices], dtype=np.int64)
    eval_case_ids = [str(idx.case_ids[i]) for i in eval_indices]

    pure_case_scores: dict[int, np.ndarray] = {k: np.zeros(n_eval, dtype=np.float64) for k in K_GRID}
    fused_case_scores: dict[int, np.ndarray] = {k: np.zeros(n_eval, dtype=np.float64) for k in K_GRID}

    is_fraud_all = (idx.outcomes == "confirmed_fraud").astype(np.float64)
    t_sec_all = idx.cols["t_sec"]
    dev_all = idx.cols["dev"]
    remail_all = art.remail_col

    for pos_e, c_idx in enumerate(eval_indices):
        t_cut = int(idx.opened_sec[c_idx])
        c_id = str(idx.cards[c_idx])
        elig = np.where((idx.closed_sec <= t_cut) & (np.arange(len(idx.mat)) != c_idx))[0]
        if len(elig) == 0:
            for k in K_GRID:
                pure_case_scores[k][pos_e] = 0.5
                fused_case_scores[k][pos_e] = 0.5
            continue

        qv = idx.mat[c_idx]
        cos_d = np.maximum(1.0 - (idx.mat[elig] @ qv), 0.0)

        ord_pure = np.argsort(cos_d, kind="stable")[:max_k]
        pure_hits_fraud = is_fraud_all[elig[ord_pure]]
        for k in K_GRID:
            sub_f = pure_hits_fraud[:k]
            pure_case_scores[k][pos_e] = float((np.sum(sub_f) + 1.0) / (len(sub_f) + 2.0))

        c_indices = idx.card_groups.get(c_id)
        ep_idx, _ = _select_asof_window(c_indices, t_cut, t_sec_all, window_hours=24)
        ep_devs = {str(d) for d in dev_all[ep_idx].tolist() if d} if len(ep_idx) > 0 else set()
        ep_rems = (
            {str(r) for r in remail_all[ep_idx].tolist() if r and r not in FREEMAIL_DOMAINS}
            if len(ep_idx) > 0
            else set()
        )
        spec_hits = spec_idx.specific_shared_cases(c_id, ep_devs, ep_rems, t_cut, cap=SPECIFIC_CARD_CAP)

        if spec_hits:
            eff_d = cos_d.copy()
            for j_pos, cand_i in enumerate(elig):
                if int(cand_i) in spec_hits:
                    eff_d[j_pos] -= FUSED_SPECIFIC_BOOST
            ord_fused = np.argsort(eff_d, kind="stable")[:max_k]
            fused_hits_fraud = is_fraud_all[elig[ord_fused]]
            for k in K_GRID:
                sub_f = fused_hits_fraud[:k]
                fused_case_scores[k][pos_e] = float((np.sum(sub_f) + 1.0) / (len(sub_f) + 2.0))
        else:
            for k in K_GRID:
                fused_case_scores[k][pos_e] = pure_case_scores[k][pos_e]

    legit_df, legit_trace = _sample_legit_card_days(
        tx=tx,
        cc=art.cc,
        n_samples=n_legit_days,
        seed=seed,
        start_date=eval_start,
        end_date=eval_end,
    )
    Z_legit, legit_sec, legit_devs, legit_rems = _embed_legit_card_days(legit_df, art)
    n_legit = len(legit_df)
    legit_cards = legit_df["card_id"].astype(str).tolist()

    pure_legit_scores: dict[int, np.ndarray] = {k: np.zeros(n_legit, dtype=np.float64) for k in K_GRID}
    fused_legit_scores: dict[int, np.ndarray] = {k: np.zeros(n_legit, dtype=np.float64) for k in K_GRID}

    for i in range(n_legit):
        t_cut = int(legit_sec[i])
        c_id = legit_cards[i]
        elig = np.where(idx.closed_sec <= t_cut)[0]
        if len(elig) == 0:
            for k in K_GRID:
                pure_legit_scores[k][i] = 0.5
                fused_legit_scores[k][i] = 0.5
            continue

        qv = Z_legit[i]
        cos_d = np.maximum(1.0 - (idx.mat[elig] @ qv), 0.0)
        ord_pure = np.argsort(cos_d, kind="stable")[:max_k]
        pure_hits_fraud = is_fraud_all[elig[ord_pure]]
        for k in K_GRID:
            sub_f = pure_hits_fraud[:k]
            pure_legit_scores[k][i] = float((np.sum(sub_f) + 1.0) / (len(sub_f) + 2.0))

        spec_hits = spec_idx.specific_shared_cases(
            c_id, legit_devs[i], legit_rems[i], t_cut, cap=SPECIFIC_CARD_CAP
        )
        if spec_hits:
            eff_d = cos_d.copy()
            for j_pos, cand_i in enumerate(elig):
                if int(cand_i) in spec_hits:
                    eff_d[j_pos] -= FUSED_SPECIFIC_BOOST
            ord_fused = np.argsort(eff_d, kind="stable")[:max_k]
            fused_hits_fraud = is_fraud_all[elig[ord_fused]]
            for k in K_GRID:
                sub_f = fused_hits_fraud[:k]
                fused_legit_scores[k][i] = float((np.sum(sub_f) + 1.0) / (len(sub_f) + 2.0))
        else:
            for k in K_GRID:
                fused_legit_scores[k][i] = pure_legit_scores[k][i]

    n_fraud = int(np.sum(y_eval == 1))
    n_cleared = int(np.sum(y_eval == 0))
    fraud_mask = y_eval == 1
    cleared_mask = y_eval == 0

    stage_trace: dict[str, Any] = {
        "total_closed_cases": int(len(art.cc)),
        "total_confirmed_fraud_all_months": int(np.sum(idx.outcomes == "confirmed_fraud")),
        "total_cleared_all_months": int(np.sum(idx.outcomes == "cleared")),
        "closed_before_oct1_standardizer_fit": int(np.sum(idx.closed_sec < int(_to_epoch_sec("2016-10-01")))),
        "opened_ge_eval_start_aug1": int(np.sum(after_start)),
        "opened_lt_eval_end_jan1": int(np.sum(before_end)),
        "eval_closed_cases_aug_dec": int(n_eval),
        "eval_confirmed_fraud_aug_dec": n_fraud,
        "eval_cleared_aug_dec": n_cleared,
        "min_prior_closed_cases_at_eval": int(
            min((int(np.sum(idx.closed_sec <= idx.opened_sec[i])) for i in eval_indices), default=0)
        ),
        **legit_trace,
    }

    grid: list[GridCell] = []
    for variant, case_sc_map, legit_sc_map in (
        ("pure", pure_case_scores, pure_legit_scores),
        ("fused", fused_case_scores, fused_legit_scores),
    ):
        for k in K_GRID:
            c_scores = case_sc_map[k]
            l_scores = legit_sc_map[k]
            possible_scores = sorted(
                {round(float((m + 1.0) / (k + 2.0)), 6) for m in range(max(0, k - 4), k + 1)}
            )
            for theta in possible_scores:
                f_fraud = int(np.sum(c_scores[fraud_mask] >= theta - 1e-9))
                f_cleared = int(np.sum(c_scores[cleared_mask] >= theta - 1e-9))
                f_legit = int(np.sum(l_scores >= theta - 1e-9))

                p_fraud = float(f_fraud / max(n_fraud, 1))
                p_cleared = float(f_cleared / max(n_cleared, 1))
                p_legit = float(f_legit / max(n_legit, 1))

                lr_cleared, ci_cl_low, ci_cl_high = _katz_lr_ci(f_fraud, n_fraud, f_cleared, n_cleared)
                lr_legit, ci_lg_low, ci_lg_high = _katz_lr_ci(f_fraud, n_fraud, f_legit, n_legit)

                grid.append(
                    GridCell(
                        variant=variant,
                        k=k,
                        theta=theta,
                        n_fraud=n_fraud,
                        n_cleared=n_cleared,
                        n_legit_days=n_legit,
                        fired_fraud=f_fraud,
                        fired_cleared=f_cleared,
                        fired_legit_days=f_legit,
                        p_fire_fraud=p_fraud,
                        p_fire_cleared=p_cleared,
                        p_fire_legit_day=p_legit,
                        lr_vs_cleared=lr_cleared,
                        ci_cleared_low=ci_cl_low,
                        ci_cleared_high=ci_cl_high,
                        lr_vs_legit_day=lr_legit,
                        ci_legit_low=ci_lg_low,
                        ci_legit_high=ci_lg_high,
                    )
                )

    def _pick_strict(cells: list[GridCell]) -> GridCell | None:
        """Strictly require BOTH P(fire|cleared) <= 8% AND P(fire|legit_day) <= 3%. Never fall back."""
        qualifying = [c for c in cells if c.qualifies]
        if not qualifying:
            return None
        return max(
            qualifying,
            key=lambda c: (round(c.ci_cleared_low, 4), round(c.p_fire_fraud, 4), -c.p_fire_cleared),
        )

    best_pure = _pick_strict([c for c in grid if c.variant == "pure"])
    best_fused = _pick_strict([c for c in grid if c.variant == "fused"])

    stage_trace["n_grid_cells"] = len(grid)
    stage_trace["n_pass_cleared_le_8pct"] = sum(1 for c in grid if c.p_fire_cleared <= MAX_CLEARED_CASE_FPR)
    stage_trace["n_pass_legit_day_le_3pct"] = sum(1 for c in grid if c.p_fire_legit_day <= MAX_LEGIT_CARD_DAY_FPR)
    stage_trace["n_qualifying_both"] = sum(1 for c in grid if c.qualifies)

    if best_pure is not None and best_fused is not None:
        if (round(best_fused.ci_cleared_low, 4), -round(best_fused.p_fire_cleared, 4)) > (
            round(best_pure.ci_cleared_low, 4),
            -round(best_pure.p_fire_cleared, 4),
        ):
            chosen: GridCell | None = best_fused
        else:
            chosen = best_pure
    else:
        chosen = best_pure or best_fused

    if chosen is None:
        min_cl = min((c.p_fire_cleared for c in grid), default=1.0)
        min_lg = min((c.p_fire_legit_day for c in grid), default=1.0)
        reason = (
            f"DISABLED (abstains on all cases): 0/{len(grid)} grid cells satisfied BOTH "
            f"P(fire|cleared) <= {MAX_CLEARED_CASE_FPR:.0%} (min={min_cl:.2%}) AND "
            f"P(fire|legit_day) <= {MAX_LEGIT_CARD_DAY_FPR:.0%} (min={min_lg:.2%})."
        )
        summary = VectorVoteBacktestSummary(
            eval_start=eval_start,
            eval_end=eval_end,
            n_fraud=n_fraud,
            n_cleared=n_cleared,
            n_legit_days=n_legit,
            grid=grid,
            enabled=False,
            status_reason=reason,
            chosen=None,
            best_pure=None,
            best_fused=None,
            stage_trace=stage_trace,
            caught_case_ids=[],
            missed_case_ids=[cid for pos_e, cid in enumerate(eval_case_ids) if y_eval[pos_e] == 1],
            cleared_acted_case_ids=[],
        )
        _save_selection_cache(summary)
        return summary

    chosen_scores = (
        pure_case_scores[chosen.k] if chosen.variant == "pure" else fused_case_scores[chosen.k]
    )
    caught_ids: list[str] = []
    missed_ids: list[str] = []
    cleared_acted_ids: list[str] = []

    for pos_e, cid in enumerate(eval_case_ids):
        fired = bool(chosen_scores[pos_e] >= chosen.theta - 1e-9)
        if y_eval[pos_e] == 1:
            if fired:
                caught_ids.append(cid)
            else:
                missed_ids.append(cid)
        elif fired:
            cleared_acted_ids.append(cid)

    reason = (
        f"ENABLED: variant={chosen.variant}, k={chosen.k}, theta={chosen.theta:.4f}, "
        f"P(fire|cleared)={chosen.p_fire_cleared:.2%} (<=8%), "
        f"P(fire|legit_day)={chosen.p_fire_legit_day:.2%} (<=3%), "
        f"weight(CI_low_cleared)={chosen.ci_cleared_low:.4f}."
    )
    summary = VectorVoteBacktestSummary(
        eval_start=eval_start,
        eval_end=eval_end,
        n_fraud=n_fraud,
        n_cleared=n_cleared,
        n_legit_days=n_legit,
        grid=grid,
        enabled=True,
        status_reason=reason,
        chosen=chosen,
        best_pure=best_pure,
        best_fused=best_fused,
        stage_trace=stage_trace,
        caught_case_ids=caught_ids,
        missed_case_ids=missed_ids,
        cleared_acted_case_ids=cleared_acted_ids,
    )
    _save_selection_cache(summary)
    return summary


def _selection_cache_key() -> str:
    return f"{dataset_tag()}::{_code_hash()}"


def _save_selection_cache(summary: VectorVoteBacktestSummary, path: Path = SELECTION_CACHE_PATH) -> Path:
    """Persist the verified backtest selection to `data/stage/r2/vector_vote_selection.json`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cache_key": _selection_cache_key(),
        "dataset_tag": dataset_tag(),
        "code_hash": _code_hash(),
        "eval_start": summary.eval_start,
        "eval_end": summary.eval_end,
        "n_fraud": summary.n_fraud,
        "n_cleared": summary.n_cleared,
        "n_legit_days": summary.n_legit_days,
        "enabled": summary.enabled,
        "status_reason": summary.status_reason,
        "chosen": asdict(summary.chosen) if summary.chosen is not None else None,
        "best_pure": asdict(summary.best_pure) if summary.best_pure is not None else None,
        "best_fused": asdict(summary.best_fused) if summary.best_fused is not None else None,
        "stage_trace": summary.stage_trace,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_or_compute_selection(path: Path = SELECTION_CACHE_PATH) -> dict[str, Any]:
    """Load persisted `(k, theta, variant, weight, enabled, counts)` from `path` when
    `cache_key == f'{dataset_tag()}::{_code_hash()}'`; otherwise run `run_vector_vote_backtest()`
    which writes and returns the fresh selection.
    """
    expected_key = _selection_cache_key()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("cache_key") == expected_key:
                return data
        except Exception:
            pass
    run_vector_vote_backtest()
    return json.loads(path.read_text())


def format_stage_trace(summary: VectorVoteBacktestSummary | None = None) -> str:
    """Return a formatted diagnostic table of closed-case and legit-card-day stage counts."""
    if summary is None:
        summary = run_vector_vote_backtest()
    lines = ["Stage / Filter Trace (Closed Cases & Legitimate Card-Days):"]
    for k, v in summary.stage_trace.items():
        lines.append(f"  {k:<42}: {v}")
    return "\n".join(lines)


def format_backtest_grid(summary: VectorVoteBacktestSummary | None = None) -> str:
    """Format the full backtest table over `(variant, k, theta)` reporting both Cleared Cases and Legit Card-Days."""
    if summary is None:
        summary = run_vector_vote_backtest()

    lines = [
        format_stage_trace(summary),
        "",
        f"As-of Backtest Window: {summary.eval_start} .. {summary.eval_end} "
        f"(Fraud={summary.n_fraud}, Cleared={summary.n_cleared}, LegitCardDays={summary.n_legit_days})",
        f"{'Variant':<7} {'k':>3} {'theta':>8} {'P(F|Fraud)':>11} {'P(F|Clear)':>11} "
        f"{'P(F|LegitD)':>11} {'LR(Clear)':>9} {'95% CI (Clear)':>17} {'LR(LegitD)':>10} {'95% CI (LegitD)':>19} {'Both(<=8%,<=3%)':>16}",
        "-" * 136,
    ]
    for c in summary.grid:
        if c.qualifies:
            ok_str = "QUALIFIES"
        elif c.p_fire_cleared > MAX_CLEARED_CASE_FPR and c.p_fire_legit_day > MAX_LEGIT_CARD_DAY_FPR:
            ok_str = "FAIL(Clr&Leg)"
        elif c.p_fire_cleared > MAX_CLEARED_CASE_FPR:
            ok_str = "FAIL(Clear>8%)"
        elif c.p_fire_legit_day > MAX_LEGIT_CARD_DAY_FPR:
            ok_str = "FAIL(Legit>3%)"
        else:
            ok_str = "FAIL(0_fraud)"

        mark = (
            " *"
            if (
                summary.chosen is not None
                and c.variant == summary.chosen.variant
                and c.k == summary.chosen.k
                and abs(c.theta - summary.chosen.theta) < 1e-6
            )
            else ""
        )
        ci_cl_str = f"[{c.ci_cleared_low:5.2f}, {c.ci_cleared_high:5.2f}]"
        ci_lg_str = f"[{c.ci_legit_low:6.2f}, {c.ci_legit_high:6.2f}]"
        lines.append(
            f"{c.variant:<7} {c.k:>3} {c.theta:>8.4f} "
            f"{c.p_fire_fraud:>10.2%} ({c.fired_fraud:>4}) "
            f"{c.p_fire_cleared:>6.2%} ({c.fired_cleared:>3}) "
            f"{c.p_fire_legit_day:>6.2%} ({c.fired_legit_days:>4}) "
            f"{c.lr_vs_cleared:>8.2f} {ci_cl_str:>17} "
            f"{c.lr_vs_legit_day:>9.2f} {ci_lg_str:>19} {ok_str:>16}{mark}"
        )
    lines.append("-" * 136)
    lines.append(f"Selection Status: {summary.status_reason}")
    return "\n".join(lines)


@dataclass
class CaseVoteEvaluation:
    case_id: str
    card_id: str
    opened_at: pd.Timestamp
    k: int
    theta: float
    variant: str
    backend: str
    score: float
    fired: bool
    enabled: bool
    weight: float
    neighbors: list[VectorNeighbor]
    qv: np.ndarray


class VectorPrecedentVote(Detector):
    """Decision-path detector (`vector_precedent`) casting a calibrated vote when a case's
    as-of behaviour vector is surrounded by confirmed-fraud precedent cases AND the
    backtested operating point satisfies both `P(fire|cleared) <= 8%` and
    `P(fire|legit card-day) <= 3%`. If no operating point satisfies both constraints,
    `enabled = False` and `run()` emits `[]`.
    Never names a typology (`pattern = "none"`), `blocks_assumed_clear = False`,
    `affected_txn_ids = []`, `connected_card_ids = []`.
    """

    name = "vector_precedent"
    pattern = "none"
    blocks_assumed_clear: bool = False

    def __init__(
        self,
        k: int | None = None,
        theta: float | None = None,
        variant: str | None = None,
        weight: float | None = None,
        enabled: bool | None = None,
    ):
        sel = load_or_compute_selection()
        chosen_dict = sel.get("chosen")
        if chosen_dict is not None:
            self.enabled: bool = bool(enabled if enabled is not None else sel.get("enabled", True))
            self.k: int = int(k if k is not None else chosen_dict["k"])
            self.theta: float = float(theta if theta is not None else chosen_dict["theta"])
            self.variant: str = str(variant if variant is not None else chosen_dict["variant"])
            self.weight: float = round(
                float(weight if weight is not None else chosen_dict["ci_cleared_low"]), 4
            )
            self.status_reason: str = str(sel.get("status_reason", "ENABLED"))
        else:
            # No cell in the backtest grid satisfied both constraints -> abstain
            self.enabled = bool(enabled if enabled is not None else False)
            self.k = int(k if k is not None else 20)
            self.theta = float(theta if theta is not None else 1.0)
            self.variant = str(variant if variant is not None else "pure")
            self.weight = round(float(weight if weight is not None else 1.0), 4)
            self.status_reason = str(sel.get("status_reason", "DISABLED: no qualifying grid cell"))

    def evaluate_ctx(self, ctx: CaseCtx, use_tg_if_enabled: bool = True) -> CaseVoteEvaluation:
        """Compute the as-of embedding, retrieve the `k` nearest closed cases (`closed_at <= opened_at`),
        and return the full `CaseVoteEvaluation` (including score, fired flag, and neighbours).
        """
        art = _default_artifacts()
        idx = art.index

        card_tx = ctx.tx[(ctx.tx["card_id"] == ctx.card_id) & (ctx.tx["t"] <= ctx.opened_at)]
        flagged_rows = ctx.tx[ctx.tx["TransactionID"] == int(ctx.flagged_txn_id)]
        if not flagged_rows.empty:
            t0 = pd.Timestamp(flagged_rows["t"].iloc[0])
            ep_tx = card_tx[(card_tx["t"] >= t0 - pd.Timedelta(hours=24)) & (card_tx["t"] <= ctx.opened_at)]
            if ep_tx.empty:
                ep_tx = flagged_rows
        else:
            ep_tx = card_tx.tail(1)

        hist_tx = card_tx[~card_tx["TransactionID"].isin(ep_tx["TransactionID"])]
        t_cut_sec = int(_to_epoch_sec(ctx.opened_at))
        ep_devs = {str(d) for d in ep_tx["device_profile"].fillna("").tolist() if d}
        ep_rems = (
            {
                str(r).strip().lower()
                for r in ep_tx["R_emaildomain"].fillna("").tolist()
                if str(r).strip() and str(r).strip().lower() not in FREEMAIL_DOMAINS
            }
            if "R_emaildomain" in ep_tx.columns
            else set()
        )
        n_shared = _count_asof_shared_dev_cards(
            devs=ep_devs,
            card_id=ctx.card_id,
            t_cut_sec=t_cut_sec,
            dev_groups=idx.dev_groups,
            t_sec_all=idx.cols["t_sec"],
            card_all=idx.cols["card"],
        )

        qv = embed_case(ep_tx, hist_tx, standardizer=art.standardizer, asof_shared_dev_cards=n_shared)
        neighbors, backend = _retrieve_neighbours_for_case(
            qv=qv,
            card_id=ctx.card_id,
            as_of=ctx.opened_at,
            k=self.k,
            variant=self.variant,
            art=art,
            ep_devs=ep_devs,
            ep_remails=ep_rems,
            exclude_case_id=ctx.case_id,
            use_tg_if_enabled=use_tg_if_enabled,
        )

        if len(ctx.cc) < int(np.sum(idx.closed_sec <= t_cut_sec)):
            allowed_ids = set(
                ctx.cc[pd.to_datetime(ctx.cc["closed_at"]) <= ctx.opened_at]["case_id"].astype(str).tolist()
            )
            neighbors = [n for n in neighbors if n.case_id in allowed_ids][: self.k]

        score = laplace_fraud_score(n.outcome for n in neighbors)
        fired = bool(self.enabled and len(neighbors) >= self.k and score >= self.theta - 1e-9)

        return CaseVoteEvaluation(
            case_id=ctx.case_id,
            card_id=ctx.card_id,
            opened_at=pd.Timestamp(ctx.opened_at),
            k=self.k,
            theta=self.theta,
            variant=self.variant,
            backend=backend,
            score=score,
            fired=fired,
            enabled=self.enabled,
            weight=self.weight,
            neighbors=neighbors,
            qv=qv,
        )

    def run(self, ctx: CaseCtx) -> list[Finding]:
        ev = self.evaluate_ctx(ctx, use_tg_if_enabled=True)
        if not ev.enabled or not ev.fired:
            return []

        nb_ids = [n.case_id for n in ev.neighbors]
        nb_outcomes = [n.outcome for n in ev.neighbors]
        nb_dists = [round(float(n.distance), 6) for n in ev.neighbors]
        n_fraud = sum(1 for o in nb_outcomes if o == "confirmed_fraud")
        nb_summary = ", ".join(
            f"{n.case_id}:{n.outcome}(d={n.distance:.4f})" for n in ev.neighbors[:5]
        )

        claim = Attestation(
            what=(
                f"Vector precedent search ({ev.backend}, variant={ev.variant}, k={ev.k}, theta={ev.theta:.4f}) "
                f"on card {ctx.card_id} as-of {ev.opened_at.strftime('%Y-%m-%d %H:%M:%S')} retrieved "
                f"{n_fraud}/{ len(ev.neighbors) } confirmed-fraud neighbours (Laplace score {ev.score:.4f} >= {ev.theta:.4f}): "
                f"{nb_summary}."
            ),
            evidence=Evidence(
                txn_ids=[int(ctx.flagged_txn_id)],
                card_ids=[ctx.card_id],
                customer_ids=[ctx.customer_id],
                case_ids=nb_ids,
            ),
            query="similar_cases_vec",
            params={
                "qv": [round(float(x), 6) for x in ev.qv.tolist()],
                "k": int(ev.k),
                "theta": round(float(ev.theta), 4),
                "variant": ev.variant,
                "backend": ev.backend,
                "as_of": ev.opened_at.strftime("%Y-%m-%d %H:%M:%S"),
                "neighbor_case_ids": nb_ids,
                "neighbor_outcomes": nb_outcomes,
                "neighbor_distances": nb_dists,
                "score": round(float(ev.score), 4),
                "precedent": True,
            },
        )
        return [
            Finding(
                detector=self.name,
                pattern=self.pattern,
                claim=claim,
                lr=self.weight,
                affected_txn_ids=[],
                connected_card_ids=[],
            )
        ]

    def backtest(self, cc: pd.DataFrame | None = None, tx: pd.DataFrame | None = None) -> Backtest:
        bt = run_vector_vote_backtest()
        lr_val = round(float(bt.chosen.lr_vs_cleared), 4) if bt.chosen is not None else 1.0
        return Backtest(
            detector=self.name,
            n_cases=bt.n_fraud + bt.n_cleared,
            caught=bt.caught_case_ids,
            missed=bt.missed_case_ids,
            legit_acted_against=bt.cleared_acted_case_ids,
            lr_measured=lr_val,
        )


def sync_missing_closed_case_vectors() -> dict[str, Any]:
    """Idempotently diagnose and upsert any `ClosedCase` vertices (specifically checking
    the 20 new Round 1 exam cases `CC-5566..CC-5585`) or their 24-dim `behaviour` vectors
    that are missing in TigerGraph.
    """
    from assay.r2.tg_r2 import (
        _auth,
        _base_url,
        _graph_name_from_schema,
        ping_tg,
        run_gsql_script,
        run_query,
    )
    from assay.tg import device_id

    if not ping_tg(timeout=3.0):
        reason = "TigerGraph REST++ endpoint is not reachable; skipping sync_missing_closed_case_vectors."
        print(f"[sync_missing_closed_case_vectors] SKIPPED: {reason}")
        return {"skipped": True, "reason": reason}

    art = _default_artifacts()
    cc = art.cc
    cc_emb = art.cc_emb
    idx = art.index

    base = _base_url()
    gname = _graph_name_from_schema()
    auth = _auth()

    stat_url = f"{base}/restpp/builtins/{gname}"
    r_stat = httpx.post(
        stat_url,
        json={"function": "stat_vertex_number", "type": "ClosedCase"},
        auth=auth,
        timeout=15.0,
    )
    tg_total_closed_cases = 0
    if r_stat.status_code == 200 and not r_stat.json().get("error"):
        res_stat = r_stat.json().get("results", [])
        if res_stat:
            tg_total_closed_cases = int(res_stat[0].get("count", 0))

    r1_new_ids = [f"CC-{i:04d}" for i in range(5566, 5586)]
    missing_vertices: list[str] = []
    missing_vectors: list[str] = []

    def _has_vector_indexed(cid: str) -> bool:
        c_pos = idx.case_to_idx[cid]
        qv_list = [round(float(x), 6) for x in idx.mat[c_pos].tolist()]
        closed_str = pd.Timestamp(cc["closed_at"].iloc[c_pos]).strftime("%Y-%m-%d %H:%M:%S")
        q_res = run_query(
            "similar_cases_vec",
            {"qv": qv_list, "k": 50, "as_of": closed_str},
        )
        hits = q_res[0].get("Hits", []) if q_res else []
        hit_ids = {str(h.get("v_id", h.get("case_id", ""))) for h in hits}
        return cid in hit_ids

    for cid in r1_new_ids:
        if cid not in idx.case_to_idx:
            continue
        v_url = f"{base}/restpp/graph/{gname}/vertices/ClosedCase/{cid}"
        rv = httpx.get(v_url, auth=auth, timeout=10.0)
        v_exists = rv.status_code == 200 and not rv.json().get("error") and bool(rv.json().get("results"))
        if not v_exists:
            missing_vertices.append(cid)
            missing_vectors.append(cid)
        elif not _has_vector_indexed(cid):
            missing_vectors.append(cid)

    if tg_total_closed_cases < len(cc):
        for cid in cc["case_id"].astype(str).tolist():
            if cid in r1_new_ids:
                continue
            v_url = f"{base}/restpp/graph/{gname}/vertices/ClosedCase/{cid}"
            rv = httpx.get(v_url, auth=auth, timeout=10.0)
            v_exists = rv.status_code == 200 and not rv.json().get("error") and bool(rv.json().get("results"))
            if not v_exists:
                missing_vertices.append(cid)
                if cid not in missing_vectors:
                    missing_vectors.append(cid)

    print(
        f"[sync_missing_closed_case_vectors] Diagnosis: TG ClosedCase count={tg_total_closed_cases}/{len(cc)}, "
        f"missing_vertices(CC-5566..5585)={missing_vertices}, missing_vectors={missing_vectors}"
    )

    upserted_vertices: list[str] = []
    if missing_vertices:
        cc_by_id = cc.set_index("case_id", drop=False)
        vertices_payload: dict[str, Any] = {"ClosedCase": {}}
        edges_payload: dict[str, Any] = {"Card": {}, "ClosedCase": {}}

        for cid in missing_vertices:
            row = cc_by_id.loc[cid]
            c_idx = idx.case_to_idx[cid]
            first_tid = 0 if pd.isna(row["first_fraud_txn_id"]) else int(float(row["first_fraud_txn_id"]))
            vertices_payload["ClosedCase"][cid] = {
                "customer_id": {"value": str(row["customer_id"])},
                "card_id": {"value": str(row["card_id"])},
                "opened_at": {"value": str(row["opened_at"])},
                "closed_at": {"value": str(row["closed_at"])},
                "outcome": {"value": str(row["outcome"])},
                "pattern": {"value": str(row["pattern"])},
                "first_fraud_txn_id": {"value": first_tid},
                "n_txns": {"value": int(row["n_txns"])},
                "exposure_usd": {"value": float(row["exposure_usd"])},
                "actions_taken": {"value": str(row["actions_taken"] or "")},
                "report_filed": {"value": str(row["report_filed"] or "No")},
                "analyst_notes": {"value": str(row["analyst_notes"] or "")},
            }

            primary_card = str(row["card_id"])
            edges_payload["Card"].setdefault(primary_card, {}).setdefault("IN_CASE", {})[cid] = {}

            edges_payload["ClosedCase"].setdefault(cid, {})
            for conn_c in row["connected_list"]:
                if conn_c:
                    edges_payload["ClosedCase"][cid].setdefault("CASE_CONNECTED_CARD", {})[str(conn_c)] = {}
            for t_id in row["txn_list"]:
                edges_payload["ClosedCase"][cid].setdefault("CASE_AFFECTED_TXN", {})[str(int(t_id))] = {}
            for d_prof in idx.case_devs[c_idx]:
                did = device_id(d_prof)
                if did:
                    edges_payload["ClosedCase"][cid].setdefault("CASE_SHARED_DEVICE", {})[did] = {}
            for reg_id in idx.case_regions[c_idx]:
                if reg_id >= 0:
                    edges_payload["ClosedCase"][cid].setdefault("CASE_IN_REGION", {})[str(int(reg_id))] = {}

        graph_url = f"{base}/restpp/graph/{gname}"
        r_up = httpx.post(
            graph_url,
            json={"vertices": vertices_payload, "edges": edges_payload},
            auth=auth,
            timeout=30.0,
        )
        if r_up.status_code != 200 or r_up.json().get("error"):
            raise TigerGraphUnavailableError(
                f"Failed to upsert missing ClosedCase vertices via REST++: {r_up.text[:400]}"
            )
        upserted_vertices = list(missing_vertices)

    synced_vector_count = 0
    if missing_vectors or upserted_vertices:
        export_tg_vectors(cc_emb, out_path=TG_R2_DIR / "case_vectors.csv")
        container_csv = "/home/tigergraph/data/tg_r2/case_vectors.csv"
        run_gsql_script(
            f'USE GRAPH {gname}\nRUN LOADING JOB load_case_vectors USING f_vec="{container_csv}"\n',
            label="sync_missing_closed_case_vectors",
            timeout=240.0,
        )
        synced_vector_count = len(cc_emb)

    still_missing: list[str] = [
        cid for cid in r1_new_ids if cid in idx.case_to_idx and not _has_vector_indexed(cid)
    ]

    print(
        f"[sync_missing_closed_case_vectors] Completed: upserted_vertices={len(upserted_vertices)}, "
        f"synced_vectors={synced_vector_count}, still_missing_in_r1_20={still_missing}"
    )
    return {
        "skipped": False,
        "tg_closed_cases_before": tg_total_closed_cases,
        "total_expected": len(cc),
        "missing_vertices": missing_vertices,
        "missing_vectors": missing_vectors,
        "upserted_vertices": upserted_vertices,
        "synced_vector_count": synced_vector_count,
        "still_missing": still_missing,
    }


def parity_check(case_ids: Iterable[str] | None = None, k: int | None = None) -> dict[str, Any]:
    """Compare TigerGraph (`ASSAY_TG=1`) vs `LocalVectorIndex` across `case_ids`
    (defaults to all 18 `case_pack()` cases).

    1. Runs `sync_missing_closed_case_vectors()` so `CC-5566..CC-5585` and their
       vectors are present in TigerGraph.
    2. Prints one raw TigerGraph `similar_cases_vec` reply for `HHF-001`.
    3. Checks that:
         - the FIRE decision (`fired`) is identical on all 18 cases, and
         - `|tg_score - local_score| <= 0.10` on all 18 cases.
       Reports per-case set Jaccard and which of the 18 cases fire under local and under TG.
    """
    from assay.r2.tg_r2 import ping_tg

    cp = case_pack()
    if case_ids is not None:
        id_set = {str(x) for x in case_ids}
        cp = cp[cp["case_id"].isin(id_set)].reset_index(drop=True)

    det = VectorPrecedentVote(k=k)
    if not ping_tg(timeout=3.0):
        reason = "TigerGraph REST++ endpoint is not reachable on this machine; skipping live TG parity_check."
        print(f"[parity_check] SKIPPED: {reason}")
        return {
            "skipped": True,
            "reason": reason,
            "checked_case_ids": cp["case_id"].astype(str).tolist(),
        }

    sync_info = sync_missing_closed_case_vectors()

    tx = tx_all()
    cc = closed_cases()
    tx_by_id = tx.set_index("TransactionID", drop=False)

    comparisons: list[dict[str, Any]] = []
    fire_mismatches: list[str] = []
    score_mismatches: list[str] = []
    fired_local_cases: list[str] = []
    fired_tg_cases: list[str] = []
    raw_hhf001_sample: list[dict[str, Any]] = []

    prev_env = os.environ.get("ASSAY_TG")
    try:
        for r in cp.itertuples():
            t0 = pd.Timestamp(r.opened_at)
            fid = int(r.flagged_txn_id)
            actual_card = str(tx_by_id.loc[fid, "card_id"]) if fid in tx_by_id.index else str(r.card_id)
            ctx = CaseCtx(
                case_id=str(r.case_id),
                flagged_txn_id=fid,
                card_id=actual_card,
                customer_id=str(r.customer_id),
                opened_at=t0,
                tx=tx[tx["t"] <= t0],
                cc=cc[pd.to_datetime(cc["closed_at"]) <= t0],
            )
            os.environ["ASSAY_TG"] = "0"
            ev_local = det.evaluate_ctx(ctx, use_tg_if_enabled=False)

            if str(r.case_id) == "HHF-001" and not raw_hhf001_sample:
                raw_rows = _tg_run_query(
                    "similar_cases_vec",
                    {
                        "qv": [round(float(x), 6) for x in ev_local.qv.tolist()],
                        "k": 3,
                        "as_of": t0.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                raw_hhf001_sample = raw_rows[:2]
                print(f"\n[parity_check] Raw TigerGraph similar_cases_vec reply for HHF-001 (top-2): {raw_hhf001_sample}")

            os.environ["ASSAY_TG"] = "1"
            ev_tg = det.evaluate_ctx(ctx, use_tg_if_enabled=True)

            if ev_local.fired:
                fired_local_cases.append(str(r.case_id))
            if ev_tg.fired:
                fired_tg_cases.append(str(r.case_id))

            loc_ids = [n.case_id for n in ev_local.neighbors]
            tg_ids = [n.case_id for n in ev_tg.neighbors]
            set_loc = set(loc_ids)
            set_tg = set(tg_ids)
            union_n = len(set_loc | set_tg)
            jaccard = round(float(len(set_loc & set_tg) / union_n), 4) if union_n > 0 else 1.0

            fire_identical = bool(ev_local.fired == ev_tg.fired)
            score_diff = abs(float(ev_tg.score) - float(ev_local.score))
            score_within_tol = bool(score_diff <= 0.10 + 1e-9)

            if not fire_identical:
                fire_mismatches.append(
                    f"{r.case_id}: tg_fired={ev_tg.fired} (score={ev_tg.score:.4f}) vs "
                    f"local_fired={ev_local.fired} (score={ev_local.score:.4f})"
                )
            if not score_within_tol:
                score_mismatches.append(
                    f"{r.case_id}: |tg_score({ev_tg.score:.4f}) - local_score({ev_local.score:.4f})| = {score_diff:.4f} > 0.10"
                )

            comparisons.append(
                {
                    "case_id": str(r.case_id),
                    "card_id": actual_card,
                    "local_ids": loc_ids,
                    "tg_ids": tg_ids,
                    "jaccard": jaccard,
                    "local_score": round(float(ev_local.score), 4),
                    "tg_score": round(float(ev_tg.score), 4),
                    "score_diff": round(score_diff, 4),
                    "local_fired": ev_local.fired,
                    "tg_fired": ev_tg.fired,
                    "fire_identical": fire_identical,
                    "score_within_tol": score_within_tol,
                }
            )
    finally:
        if prev_env is None:
            os.environ.pop("ASSAY_TG", None)
        else:
            os.environ["ASSAY_TG"] = prev_env

    print(
        f"\n=== C5 Parity Table (enabled={det.enabled}, k={det.k}, theta={det.theta:.4f}, variant={det.variant}) ==="
    )
    header = (
        f"{'Case':<8} {'Card':<10} {'Loc_Score':>9} {'TG_Score':>9} {'|Diff|':>7} "
        f"{'Loc_Fire':>8} {'TG_Fire':>8} {'Set_Jacc':>9}"
    )
    print(header)
    print("-" * len(header))
    for c in comparisons:
        print(
            f"{c['case_id']:<8} {c['card_id']:<10} {c['local_score']:>9.4f} {c['tg_score']:>9.4f} "
            f"{c['score_diff']:>7.4f} {str(c['local_fired']):>8} {str(c['tg_fired']):>8} {c['jaccard']:>9.4f}"
        )

    print(
        f"\n[parity_check] Fired under Local ({len(fired_local_cases)}): {fired_local_cases}\n"
        f"[parity_check] Fired under TG    ({len(fired_tg_cases)}): {fired_tg_cases}\n"
        f"[parity_check] Summary: fire_decision_matches={len(comparisons) - len(fire_mismatches)}/{len(comparisons)}, "
        f"score_within_0.10={len(comparisons) - len(score_mismatches)}/{len(comparisons)}, "
        f"mean_set_jaccard={np.mean([c['jaccard'] for c in comparisons]):.4f}"
    )

    return {
        "skipped": False,
        "enabled": det.enabled,
        "k": det.k,
        "theta": det.theta,
        "sync_info": sync_info,
        "raw_hhf001_sample": raw_hhf001_sample,
        "n_checked": len(comparisons),
        "n_fire_identical": len(comparisons) - len(fire_mismatches),
        "n_score_within_tol": len(comparisons) - len(score_mismatches),
        "fired_local_cases": fired_local_cases,
        "fired_tg_cases": fired_tg_cases,
        "fire_mismatches": fire_mismatches,
        "score_mismatches": score_mismatches,
        "mismatches": fire_mismatches + score_mismatches,
        "comparisons": comparisons,
    }


def generate_c5_doc(
    summary: VectorVoteBacktestSummary | None = None,
    exam_evals: list[CaseVoteEvaluation] | None = None,
    parity_res: dict[str, Any] | None = None,
    out_path: Path = DOC_OUTPUT_PATH,
) -> Path:
    """Generate `docs/r2/C5_vector_vote.md` directly from the live measured backtest,
    exam-case evaluation, and parity check so documentation and code can never drift.
    """
    if summary is None:
        summary = run_vector_vote_backtest()

    det = VectorPrecedentVote()
    if exam_evals is None:
        tx = tx_all()
        cc = closed_cases()
        cp = case_pack()
        tx_by_id = tx.set_index("TransactionID", drop=False)
        exam_evals = []
        for r in cp.itertuples():
            t0 = pd.Timestamp(r.opened_at)
            fid = int(r.flagged_txn_id)
            actual_card = str(tx_by_id.loc[fid, "card_id"]) if fid in tx_by_id.index else str(r.card_id)
            ctx = CaseCtx(
                case_id=str(r.case_id),
                flagged_txn_id=fid,
                card_id=actual_card,
                customer_id=str(r.customer_id),
                opened_at=t0,
                tx=tx[tx["t"] <= t0],
                cc=cc[pd.to_datetime(cc["closed_at"]) <= t0],
            )
            exam_evals.append(det.evaluate_ctx(ctx, use_tg_if_enabled=False))

    lines: list[str] = [
        "<!-- GENERATED; DO NOT EDIT BY HAND. Written by assay.r2.detectors.vector_vote.generate_c5_doc() -->",
        "# C5 — Putting TigerVector Precedent on the Decision Path (`vector_precedent`)",
        "",
        "## 1. Design, Zero-Leakage Guarantee & Selection Cache",
        f"- **Dataset Tag:** `{dataset_tag()}` | **Code Hash:** `{_code_hash()}`",
        "- **Selection Cache:** `data/stage/r2/vector_vote_selection.json`",
        "- **Detector:** `VectorPrecedentVote` (`name = 'vector_precedent'`, `pattern = 'none'`, `blocks_assumed_clear = False`).",
        "- **Strict Dual Constraint:** A candidate `(variant, k, theta)` is eligible **if and only if** both:",
        f"  1. `P(fire | cleared case) <= {MAX_CLEARED_CASE_FPR:.0%}` (`<= 8%`), **and**",
        f"  2. `P(fire | legit card-day) <= {MAX_LEGIT_CARD_DAY_FPR:.0%}` (`<= 3%`).",
        "  If no candidate satisfies both constraints, `enabled = False` and the detector abstains (`run(ctx) -> []`).",
        "",
        "## 2. Stage & Filter Trace",
        "```",
        format_stage_trace(summary),
        "```",
        "",
        "## 3. Measured As-Of Backtest Grid (Aug–Dec 2016 Closed Cases & 2,400 Legitimate Card-Days)",
        f"- **Evaluation window:** `{summary.eval_start}` .. `{summary.eval_end}`",
        f"- **Evaluated closed cases:** `{summary.n_fraud + summary.n_cleared}` (`confirmed_fraud = {summary.n_fraud}`, `cleared = {summary.n_cleared}`)",
        f"- **Sampled legitimate card-days (`seed=42`):** `{summary.n_legit_days}`",
        f"- **Selection outcome:** `{summary.status_reason}`",
        "",
        "| Variant | `k` | `theta` | `P(fire \\| fraud)` | `P(fire \\| cleared)` | `P(fire \\| legit day)` | `LR vs cleared` | `95% CI (cleared)` | `LR vs legit day` | `95% CI (legit day)` | Dual Gate (`<=8%` & `<=3%`) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]

    for c in summary.grid:
        gate = "**QUALIFIES**" if c.qualifies else (
            "FAIL (both)" if (c.p_fire_cleared > MAX_CLEARED_CASE_FPR and c.p_fire_legit_day > MAX_LEGIT_CARD_DAY_FPR)
            else ("FAIL (`cleared > 8%`)" if c.p_fire_cleared > MAX_CLEARED_CASE_FPR else "FAIL (`legit_day > 3%`)")
        )
        is_chosen = (
            summary.chosen is not None
            and c.variant == summary.chosen.variant
            and c.k == summary.chosen.k
            and abs(c.theta - summary.chosen.theta) < 1e-6
        )
        v_label = f"**`{c.variant}` \\***" if is_chosen else f"`{c.variant}`"
        lines.append(
            f"| {v_label} | `{c.k}` | `{c.theta:.4f}` | `{c.p_fire_fraud:.2%}` (`{c.fired_fraud}/{c.n_fraud}`) | "
            f"`{c.p_fire_cleared:.2%}` (`{c.fired_cleared}/{c.n_cleared}`) | "
            f"`{c.p_fire_legit_day:.2%}` (`{c.fired_legit_days}/{c.n_legit_days}`) | "
            f"`{c.lr_vs_cleared:.2f}` | `[{c.ci_cleared_low:.2f}, {c.ci_cleared_high:.2f}]` | "
            f"`{c.lr_vs_legit_day:.2f}` | `[{c.ci_legit_low:.2f}, {c.ci_legit_high:.2f}]` | {gate} |"
        )

    lines.extend([
        "",
        "## 4. Measured Run on the 18 Final-Round Exam Cases (`case_pack()`)",
        f"- **Detector config:** `enabled={det.enabled}`, `variant={det.variant}`, `k={det.k}`, `theta={det.theta:.4f}`, `weight={det.weight:.4f}`",
        f"- **Exam cases firing (Local):** `{[e.case_id for e in exam_evals if e.fired]}`",
        "",
        "| Case ID | Card ID | Opened At | Score | Fired? | Fraud / `k` | Top-5 Precedent Neighbours (`id:outcome:dist`) |",
        "|---|---|---|---:|:---:|---:|---|",
    ])

    for ev in exam_evals:
        n_f = sum(1 for n in ev.neighbors if n.outcome == "confirmed_fraud")
        top5 = ", ".join(
            f"{n.case_id}:{'F' if n.outcome == 'confirmed_fraud' else 'C'}:{n.distance:.3f}"
            for n in ev.neighbors[:5]
        )
        lines.append(
            f"| `{ev.case_id}` | `{ev.card_id}` | `{ev.opened_at}` | `{ev.score:.4f}` | `{ev.fired}` | `{n_f}/{len(ev.neighbors)}` | `{top5}` |"
        )

    if parity_res is not None:
        lines.extend([
            "",
            "## 5. TigerGraph vs Local Parity Check",
        ])
        if parity_res.get("skipped"):
            lines.append(f"- **Status:** Skipped (`{parity_res.get('reason')}`)")
        else:
            lines.extend([
                f"- **Fire decision agreement:** `{parity_res['n_fire_identical']}/{parity_res['n_checked']}`",
                f"- **Score within `<= 0.10`:** `{parity_res['n_score_within_tol']}/{parity_res['n_checked']}`",
                f"- **Fired under Local:** `{parity_res['fired_local_cases']}`",
                f"- **Fired under TigerGraph:** `{parity_res['fired_tg_cases']}`",
            ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
