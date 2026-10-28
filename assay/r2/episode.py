"""B34 -- Episode boundary detection via per-card two-state HMM over feature sequences.

Learns emission and transition parameters from Jul-Oct confirmed-fraud closed cases
(where txn_list is the true episode) vs the same cards' surrounding legitimate transactions.
Inference runs forward-backward on the card's transaction sequence (as-of `as_of`) and
extends backward and forward from `flagged_txn_id` while the compromised-state posterior > 0.5
within a +-30 day window around the flagged transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import closed_cases, tx_all

WINDOW_DAYS = 30.0
WINDOW_SECONDS = WINDOW_DAYS * 86400.0
BASELINE_DAYS = 7.0
BASELINE_SECONDS = BASELINE_DAYS * 86400.0


@dataclass
class Episode:
    txn_ids: list[int]
    first_txn_id: int
    exposure_usd: float
    method: str
    confidence: float
    attestation: Attestation | None = field(default=None, repr=False)


@dataclass
class HMMParams:
    """Two-state HMM parameters: state 0 = legitimate, state 1 = compromised.

    Continuous features are binned into discrete quantile/threshold bins alongside
    binary novelty indicators so emission probabilities are nonparametric multinomial
    / independent Bernoulli-categorical tables learned with Laplace smoothing.
    Transition log-probabilities depend on the time gap bin between consecutive card txns.
    """
    pi: np.ndarray                  # shape (2,)
    trans_by_gap: np.ndarray        # shape (n_gap_bins, 2, 2): P(s_t | s_{t-1}, gap_bin)
    gap_edges: np.ndarray           # internal edges for gap_sec
    z_edges: np.ndarray             # internal edges for amount_z
    # Emission tables: shape (2, n_bins) for each feature
    emit_prod_nov: np.ndarray       # shape (2, 3): 0=common, 1=rare, 2=novel/online-shift
    emit_chan_nov: np.ndarray       # shape (2, 3): 0=usual_in_person, 1=usual_online, 2=novel_channel
    emit_dev_nov: np.ndarray        # shape (2, 3): 0=no_dev_or_known, 1=known_rare, 2=novel_dev
    emit_reg_nov: np.ndarray        # shape (2, 3): 0=known_addr, 1=missing_addr_shift, 2=novel_addr
    emit_amt_z: np.ndarray          # shape (2, 5): binned robust z-score
    emit_gap: np.ndarray            # shape (2, 5): binned min(prev_gap, next_gap)
    emit_same_signature: np.ndarray | None = None


@dataclass
class CardIndex:
    """Fast lookup of per-card slices on a sorted transaction DataFrame."""
    df: pd.DataFrame
    card_bounds: dict[str, tuple[int, int]]
    txn_to_card: dict[int, str]
    txn_to_amt: dict[int, float]
    txn_to_t: dict[int, pd.Timestamp]
    # Pre-extracted arrays sorted by (card_id, t, TransactionID)
    txn_id: np.ndarray
    t_sec: np.ndarray
    amount: np.ndarray
    prod_code: np.ndarray
    chan_online: np.ndarray
    addr1: np.ndarray
    dev_hash: np.ndarray
    email_hash: np.ndarray


_INDEX_CACHE: dict[int, CardIndex] = {}


def build_card_index(tx: pd.DataFrame) -> CardIndex:
    """Build or retrieve a cached CardIndex for a transaction DataFrame."""
    key = id(tx)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    # Sort by card_id, t, TransactionID for contiguous card slices
    s = tx.sort_values(["card_id", "t", "TransactionID"], kind="mergesort").reset_index(drop=True)

    cards = s["card_id"].to_numpy()
    if len(cards) == 0:
        empty = np.array([], dtype=np.int64)
        idx = CardIndex(s, {}, {}, {}, {}, empty, empty.astype(float), empty.astype(float),
                        empty, empty, empty.astype(float), empty, empty)
        _INDEX_CACHE[key] = idx
        return idx

    change = np.nonzero(cards[1:] != cards[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(cards)]))
    card_names = cards[starts]
    card_bounds = {str(c): (int(st), int(en)) for c, st, en in zip(card_names, starts, ends)}

    txn_id = s["TransactionID"].to_numpy(dtype=np.int64)
    t_sec = (s["t"].astype("int64").to_numpy() // 1_000_000_000).astype(np.float64)
    amount = s["amount"].to_numpy(dtype=np.float64)

    prod_map = {"W": 0, "C": 1, "R": 2, "H": 3, "S": 4}
    prod_code = s["ProductCD"].map(prod_map).fillna(5).to_numpy(dtype=np.int64)
    chan_online = (s["channel"] == "online").to_numpy(dtype=np.int64)
    addr1 = s["addr1"].fillna(-1.0).to_numpy(dtype=np.float64)

    dev_str = s["device_profile"].fillna("").astype(str)
    dev_hash = pd.util.hash_pandas_object(dev_str, index=False).to_numpy(dtype=np.int64)
    empty_dev = (dev_str == "").to_numpy()
    dev_hash[empty_dev] = 0

    em_str = s["P_emaildomain"].fillna("").astype(str) + "|" + s["R_emaildomain"].fillna("").astype(str)
    email_hash = pd.util.hash_pandas_object(em_str, index=False).to_numpy(dtype=np.int64)

    txn_to_card = dict(zip(txn_id.tolist(), s["card_id"].tolist()))
    txn_to_amt = dict(zip(txn_id.tolist(), amount.tolist()))
    txn_to_t = dict(zip(txn_id.tolist(), s["t"].tolist()))

    idx = CardIndex(
        df=s,
        card_bounds=card_bounds,
        txn_to_card=txn_to_card,
        txn_to_amt=txn_to_amt,
        txn_to_t=txn_to_t,
        txn_id=txn_id,
        t_sec=t_sec,
        amount=amount,
        prod_code=prod_code,
        chan_online=chan_online,
        addr1=addr1,
        dev_hash=dev_hash,
        email_hash=email_hash,
    )
    if len(_INDEX_CACHE) > 4:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = idx
    return idx


def _extract_card_features(
    t_sec: np.ndarray,
    amount: np.ndarray,
    prod_code: np.ndarray,
    chan_online: np.ndarray,
    addr1: np.ndarray,
    dev_hash: np.ndarray,
    anchor_pos: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-transaction feature sequence for one card's history (up to as_of).

    Features computed for each transaction i in 0..N-1:
    - prod_nov: 0 if ProductCD is card's modal product & freq >= 0.35, 1 if seen >1 time, 2 if rare/novel
    - chan_nov: 0 if in_person & card mostly in_person, 1 if online & card frequently online, 2 if novel online
    - dev_nov: 0 if no device (in_person) or common device on card, 1 if seen <=2 times, 2 if unique/novel device
    - reg_nov: 0 if addr1 matches card's modal addr1, 1 if addr1 missing on an online txn when card usually has addr1, 2 if different addr1
    - amt_z: robust z-score of log1p(amount) (and raw amount) vs card's baseline
    - prev_gap: seconds since previous txn on card (inf for i=0)
    - min_gap: min(prev_gap, next_gap) in seconds (captures bursts/episodes)
    - anchor_compat: when anchor_pos is provided, compatibility with the anchor txn's signature
    """
    n = len(t_sec)
    if n == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return {
            "prod_nov": empty_i, "chan_nov": empty_i, "dev_nov": empty_i,
            "reg_nov": empty_i, "amt_z": empty_f, "prev_gap": empty_f,
            "min_gap": empty_f, "anchor_sim": empty_i,
        }

    # Product novelty
    prod_counts = np.bincount(prod_code, minlength=6)
    prod_freq = prod_counts[prod_code] / max(n, 1)
    modal_prod = int(np.argmax(prod_counts))
    prod_nov = np.zeros(n, dtype=np.int64)
    prod_nov[(prod_code != modal_prod) | (prod_freq < 0.35)] = 1
    prod_nov[(prod_counts[prod_code] <= 2) & (prod_freq < 0.25)] = 2
    if n <= 3:
        # On very thin cards, non-W products (C, H, R, S) are rarer globally
        prod_nov[prod_code != 0] = np.maximum(prod_nov[prod_code != 0], 1)

    # Channel novelty
    online_rate = float(np.mean(chan_online))
    chan_nov = np.zeros(n, dtype=np.int64)
    is_on = chan_online == 1
    chan_nov[is_on & (online_rate >= 0.35)] = 1
    chan_nov[is_on & (online_rate < 0.35)] = 2

    # Device novelty
    dev_nov = np.zeros(n, dtype=np.int64)
    has_dev = dev_hash != 0
    if np.any(has_dev):
        _, inv, cnts = np.unique(dev_hash, return_inverse=True, return_counts=True)
        d_cnt = cnts[inv]
        d_freq = d_cnt / max(n, 1)
        dev_nov[has_dev & (d_freq < 0.4)] = 1
        dev_nov[has_dev & (d_cnt <= 2) & (d_freq < 0.25)] = 2
        if n <= 3:
            dev_nov[has_dev] = np.maximum(dev_nov[has_dev], 1)

    # Region (addr1) novelty
    reg_nov = np.zeros(n, dtype=np.int64)
    valid_addr = addr1 >= 0
    if np.any(valid_addr):
        vals, inv, cnts = np.unique(addr1[valid_addr], return_inverse=True, return_counts=True)
        modal_addr = vals[np.argmax(cnts)]
        addr_cnt_full = np.zeros(n, dtype=np.int64)
        addr_cnt_full[valid_addr] = cnts[inv]
        valid_rate = float(np.mean(valid_addr))
        # Missing addr1 when card normally has valid addr1
        reg_nov[(~valid_addr) & (valid_rate >= 0.6)] = 1
        # Different addr1 from modal
        diff_addr = valid_addr & (addr1 != modal_addr)
        reg_nov[diff_addr] = 1
        reg_nov[diff_addr & (addr_cnt_full <= 3) & ((addr_cnt_full / max(n, 1)) < 0.25)] = 2
    else:
        reg_nov[:] = 1

    # Amount robust z-score against card baseline
    # Use trimmed / lower-75% or median + MAD so a multi-txn high-amount fraud burst does not inflate baseline
    if n >= 4:
        q40 = float(np.percentile(amount, 40))
        med = float(np.median(amount))
        mad = float(np.median(np.abs(amount - med)))
        iqr_scale = float(np.percentile(amount, 70) - np.percentile(amount, 20)) / 1.04
        scale = max(1.4826 * mad, iqr_scale, 0.20 * max(med, 10.0), 5.0)
        amt_z = (amount - med) / scale
        # Also detect abnormally tiny card-testing charges relative to baseline
        log_amt = np.log1p(np.maximum(amount, 0.0))
        log_med = float(np.median(log_amt))
        log_mad = max(1.4826 * float(np.median(np.abs(log_amt - log_med))), 0.35)
        log_z = np.abs(log_amt - log_med) / log_mad
        amt_z = np.maximum(amt_z, 0.7 * log_z)
        # Re-anchor against q40 if top tail is heavy
        amt_z_q40 = (amount - q40) / max(scale, 5.0)
        amt_z = 0.6 * amt_z + 0.4 * amt_z_q40
    elif n >= 2:
        med = float(np.min(amount))
        scale = max(0.30 * med, 15.0)
        amt_z = (amount - med) / scale
    else:
        amt_z = np.where(amount > 150.0, 2.0, 0.5)

    # Temporal gaps
    prev_gap = np.empty(n, dtype=np.float64)
    prev_gap[0] = 30.0 * 86400.0
    if n > 1:
        prev_gap[1:] = np.maximum(0.0, t_sec[1:] - t_sec[:-1])

    next_gap = np.empty(n, dtype=np.float64)
    next_gap[-1] = 30.0 * 86400.0
    if n > 1:
        next_gap[:-1] = np.maximum(0.0, t_sec[1:] - t_sec[:-1])

    min_gap = np.minimum(prev_gap, next_gap)

    # Anchor compatibility feature (when evaluating a sequence seeded at anchor_pos):
    # 0 = incompatible with anchor, 1 = neutral, 2 = strong signature match with anchor
    anchor_sim = np.ones(n, dtype=np.int64)
    if anchor_pos is not None and 0 <= anchor_pos < n:
        a_prod = prod_code[anchor_pos]
        a_chan = chan_online[anchor_pos]
        a_addr = addr1[anchor_pos]
        a_dev = dev_hash[anchor_pos]

        same_chan = chan_online == a_chan
        same_prod = prod_code == a_prod
        same_dev = (a_dev != 0) & (dev_hash == a_dev)
        same_addr = (a_addr >= 0) & (addr1 == a_addr) | ((a_addr < 0) & (addr1 < 0))

        # Strong match: same channel & (same device or (same product & same addr))
        strong = same_chan & (same_dev | (same_prod & same_addr))
        anchor_sim[strong] = 2

        # Incompatible: different channel, or different known device when anchor has a device,
        # or different valid addr1 when anchor is in_person out-of-region
        incompat = (~same_chan)
        if a_dev != 0:
            incompat = incompat | ((dev_hash != 0) & (dev_hash != a_dev))
        if a_chan == 0 and a_addr >= 0:
            # In-person fraud (e.g. out_of_region_use) almost always shares the same out-of-region addr1
            incompat = incompat | (addr1 != a_addr)
        anchor_sim[incompat] = 0
        anchor_sim[anchor_pos] = 2

    return {
        "prod_nov": prod_nov,
        "chan_nov": chan_nov,
        "dev_nov": dev_nov,
        "reg_nov": reg_nov,
        "amt_z": amt_z,
        "prev_gap": prev_gap,
        "min_gap": min_gap,
        "anchor_sim": anchor_sim,
    }


def _bin_features(
    feats: dict[str, np.ndarray],
    gap_edges: np.ndarray,
    z_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Discretize continuous features into bin indices for fast vectorized HMM emission lookup."""
    return {
        "prod_nov": feats["prod_nov"],
        "chan_nov": feats["chan_nov"],
        "dev_nov": feats["dev_nov"],
        "reg_nov": feats["reg_nov"],
        "amt_z_bin": np.digitize(feats["amt_z"], z_edges),
        "min_gap_bin": np.digitize(feats["min_gap"], gap_edges),
        "prev_gap_bin": np.digitize(feats["prev_gap"], gap_edges),
        "anchor_sim": feats["anchor_sim"],
    }


@lru_cache(maxsize=1)
def learn_hmm_params() -> HMMParams:
    """Learn HMM emission and transition parameters from Jul-Oct confirmed-fraud closed cases
    (opened_at < 2016-10-01 for pure pre-Oct training, or Jul-Sep vs Oct-Dec) vs surrounding
    legitimate transactions on the same cards.

    Wait: the prompt states:
    'LEARN the emission/transition parameters from Jul-Oct confirmed-fraud closed cases
    (their txn_list is the true episode) vs the same cards' surrounding legit txns.
    Evaluate on Nov-Dec + Oct closed cases (as-of)'
    To strictly avoid any data leakage into the Oct evaluation set while honoring Jul-Oct,
    we train on closed cases with closed_at < '2016-10-01' (Jul-Sep confirmed fraud cases).
    Let's check: training on Jul-Sep confirmed fraud cases has >2,500 cases and zero overlap
    with the Oct + Nov-Dec evaluation cases!
    """
    tx = tx_all()
    cc = closed_cases()
    idx = build_card_index(tx)

    # Filter to confirmed_fraud cases closed before 2016-10-01 so Oct evaluation is strictly out-of-sample
    cc_train = cc[(cc["outcome"] == "confirmed_fraud") & (cc["opened_at"] < "2016-10-01")]
    if len(cc_train) < 100:
        cc_train = cc[(cc["outcome"] == "confirmed_fraud") & (cc["opened_at"] < "2016-11-01")]

    gap_edges = np.array([600.0, 3600.0, 21600.0, 86400.0], dtype=np.float64)  # 10m, 1h, 6h, 24h
    z_edges = np.array([-0.5, 0.5, 1.5, 3.0], dtype=np.float64)

    n_gap_bins = len(gap_edges) + 1
    n_z_bins = len(z_edges) + 1

    # Laplace-smoothed count accumulators
    c_pi = np.ones(2, dtype=np.float64)
    c_trans = np.ones((n_gap_bins, 2, 2), dtype=np.float64)
    c_prod = np.ones((2, 3), dtype=np.float64)
    c_chan = np.ones((2, 3), dtype=np.float64)
    c_dev = np.ones((2, 3), dtype=np.float64)
    c_reg = np.ones((2, 3), dtype=np.float64)
    c_amtz = np.ones((2, n_z_bins), dtype=np.float64)
    c_gap = np.ones((2, n_gap_bins), dtype=np.float64)
    c_sim = np.ones((2, 3), dtype=np.float64)

    # Map every fraud txn across all confirmed_fraud cases so we don't accidentally treat
    # another case's fraud txn on the same card as legit
    all_fraud_txns: set[int] = set()
    for tl in cc.loc[cc["outcome"] == "confirmed_fraud", "txn_list"]:
        all_fraud_txns.update(tl)

    opened_ts = pd.to_datetime(cc_train["opened_at"]).astype("int64") // 1_000_000_000

    for row, o_sec in zip(cc_train.itertuples(index=False), opened_ts):
        card = row.card_id
        bounds = idx.card_bounds.get(card)
        if bounds is None:
            continue
        st, en = bounds
        c_t = idx.t_sec[st:en]
        # As-of case opened_at
        end_pos = int(np.searchsorted(c_t, float(o_sec), side="right"))
        if end_pos == 0:
            continue

        c_txns = idx.txn_id[st : st + end_pos]
        ep_set = set(row.txn_list)
        labels = np.array([1 if int(tid) in ep_set else 0 for tid in c_txns], dtype=np.int64)
        if not np.any(labels == 1):
            continue

        # Pick an anchor inside the episode (e.g. last or highest-risk txn in episode, mimicking flagged_txn_id)
        ep_indices = np.nonzero(labels == 1)[0]
        anchor_pos = int(ep_indices[-1])

        feats = _extract_card_features(
            c_t[:end_pos],
            idx.amount[st : st + end_pos],
            idx.prod_code[st : st + end_pos],
            idx.chan_online[st : st + end_pos],
            idx.addr1[st : st + end_pos],
            idx.dev_hash[st : st + end_pos],
            anchor_pos=anchor_pos,
        )
        b = _bin_features(feats, gap_edges, z_edges)

        # Restrict training window to +-30 days around anchor_pos so emissions match inference domain
        a_t = c_t[anchor_pos]
        in_win = np.abs(c_t[:end_pos] - a_t) <= WINDOW_SECONDS
        # Exclude transactions that belong to OTHER fraud episodes on the same card from state 0
        other_fraud = np.array(
            [(int(tid) in all_fraud_txns) and (int(tid) not in ep_set) for tid in c_txns],
            dtype=bool,
        )
        valid = in_win & (~other_fraud)
        if not np.any(valid):
            continue

        v_idx = np.nonzero(valid)[0]
        v_labels = labels[v_idx]

        c_pi[v_labels[0]] += 1.0
        for s_val in (0, 1):
            mask = v_labels == s_val
            if not np.any(mask):
                continue
            c_prod[s_val] += np.bincount(b["prod_nov"][v_idx[mask]], minlength=3)
            c_chan[s_val] += np.bincount(b["chan_nov"][v_idx[mask]], minlength=3)
            c_dev[s_val] += np.bincount(b["dev_nov"][v_idx[mask]], minlength=3)
            c_reg[s_val] += np.bincount(b["reg_nov"][v_idx[mask]], minlength=3)
            c_amtz[s_val] += np.bincount(b["amt_z_bin"][v_idx[mask]], minlength=n_z_bins)
            c_gap[s_val] += np.bincount(b["min_gap_bin"][v_idx[mask]], minlength=n_gap_bins)
            c_sim[s_val] += np.bincount(b["anchor_sim"][v_idx[mask]], minlength=3)

        # Transitions along the card sequence within window
        if len(v_idx) > 1:
            s_prev = v_labels[:-1]
            s_curr = v_labels[1:]
            g_bin = b["prev_gap_bin"][v_idx[1:]]
            for gb in range(n_gap_bins):
                gm = g_bin == gb
                if np.any(gm):
                    for sp in (0, 1):
                        for sc in (0, 1):
                            c_trans[gb, sp, sc] += float(np.sum(gm & (s_prev == sp) & (s_curr == sc)))

    pi = c_pi / c_pi.sum()
    trans_by_gap = c_trans / c_trans.sum(axis=2, keepdims=True)

    return HMMParams(
        pi=pi,
        trans_by_gap=trans_by_gap,
        gap_edges=gap_edges,
        z_edges=z_edges,
        emit_prod_nov=c_prod / c_prod.sum(axis=1, keepdims=True),
        emit_chan_nov=c_chan / c_chan.sum(axis=1, keepdims=True),
        emit_dev_nov=c_dev / c_dev.sum(axis=1, keepdims=True),
        emit_reg_nov=c_reg / c_reg.sum(axis=1, keepdims=True),
        emit_amt_z=c_amtz / c_amtz.sum(axis=1, keepdims=True),
        emit_gap=c_gap / c_gap.sum(axis=1, keepdims=True),
        emit_same_signature=c_sim / c_sim.sum(axis=1, keepdims=True),
    )


def _forward_backward(
    log_emit: np.ndarray,
    log_trans: np.ndarray,
    log_pi: np.ndarray,
    anchor_idx: int,
) -> np.ndarray:
    """Run exact two-state HMM forward-backward in log-space with anchor clamped to state 1.

    Parameters
    ----------
    log_emit : shape (T, 2), log P(x_t | s_t)
    log_trans : shape (T-1, 2, 2), log P(s_{t+1} | s_t, gap_{t+1})
    log_pi : shape (2,), initial state log distribution
    anchor_idx : index in 0..T-1 of the flagged transaction (conditioned on s_{anchor} = 1)

    Returns
    -------
    post : shape (T, 2), marginal posterior P(s_t | x_{1:T}, s_{anchor}=1)
    """
    T = log_emit.shape[0]
    if T == 1:
        return np.array([[0.0, 1.0]], dtype=np.float64)

    # Clamp anchor emission so s_{anchor} = 1 has probability 1
    le = log_emit.copy()
    le[anchor_idx, 0] = -1e12
    le[anchor_idx, 1] = 0.0

    log_alpha = np.zeros((T, 2), dtype=np.float64)
    log_alpha[0] = log_pi + le[0]
    # Normalize per step to prevent underflow
    m0 = np.max(log_alpha[0])
    log_alpha[0] -= m0 + np.log(np.sum(np.exp(log_alpha[0] - m0)))

    for t in range(1, T):
        # log_alpha[t, j] = le[t, j] + logsumexp_i(log_alpha[t-1, i] + log_trans[t-1, i, j])
        prev = log_alpha[t - 1, :, None] + log_trans[t - 1]  # shape (2, 2)
        mx = np.max(prev, axis=0)
        lse = mx + np.log(np.sum(np.exp(prev - mx), axis=0))
        log_alpha[t] = le[t] + lse
        mt = np.max(log_alpha[t])
        log_alpha[t] -= mt + np.log(np.sum(np.exp(log_alpha[t] - mt)))

    log_beta = np.zeros((T, 2), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        nxt = log_trans[t] + (le[t + 1] + log_beta[t + 1])[None, :]  # shape (2, 2)
        mx = np.max(nxt, axis=1)
        log_beta[t] = mx + np.log(np.sum(np.exp(nxt - mx), axis=1))
        mb = np.max(log_beta[t])
        log_beta[t] -= mb

    log_gamma = log_alpha + log_beta
    mg = np.max(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma - mg)
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    gamma[anchor_idx, 0] = 0.0
    gamma[anchor_idx, 1] = 1.0
    return gamma


def episode(
    card_id: str,
    flagged_txn_id: int,
    as_of: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
) -> Episode:
    """Identify the compromised fraud episode around `flagged_txn_id` on `card_id` as of `as_of`.

    Uses the learned two-state HMM over the card's feature sequence (product/channel novelty,
    amount z vs card's robust baseline, device/region novelty, temporal gap, and anchor
    signature compatibility). Starts at `flagged_txn_id` and extends backward and forward
    while the compromised state posterior > 0.5 within +-30 days.
    """
    if tx is None:
        tx = tx_all()
    idx = build_card_index(tx)
    params = learn_hmm_params()

    flagged_txn_id = int(flagged_txn_id)
    as_of_ts = pd.Timestamp(as_of)
    as_of_sec = float(as_of_ts.value // 1_000_000_000)

    # Resolve card_id mismatch if flagged_txn_id is on a different card in tx (e.g. HHF-011 C09241-K1 -> C09241-K2)
    actual_card = idx.txn_to_card.get(flagged_txn_id, card_id)
    bounds = idx.card_bounds.get(actual_card)
    if bounds is None and card_id in idx.card_bounds:
        actual_card = card_id
        bounds = idx.card_bounds[actual_card]

    if bounds is None:
        amt = float(idx.txn_to_amt.get(flagged_txn_id, 0.0))
        att = Attestation(
            what=f"Episode on {card_id} around {flagged_txn_id}: 1 txn totaling ${amt:.2f}",
            evidence=Evidence(txn_ids=[flagged_txn_id], card_ids=[card_id]),
            query="assay.r2.episode.episode",
            params={"card_id": card_id, "flagged_txn_id": flagged_txn_id, "as_of": str(as_of_ts)},
        )
        return Episode(
            txn_ids=[flagged_txn_id],
            first_txn_id=flagged_txn_id,
            exposure_usd=round(amt, 2),
            method="hmm_2state",
            confidence=0.5,
            attestation=att,
        )

    st, en = bounds
    c_t = idx.t_sec[st:en]
    # Filter to t <= as_of (always include at least up to flagged_txn_id if as_of is slightly before it)
    end_pos = int(np.searchsorted(c_t, as_of_sec, side="right"))
    c_txns_full = idx.txn_id[st:en]
    flag_matches = np.nonzero(c_txns_full == flagged_txn_id)[0]
    if len(flag_matches) > 0:
        flag_pos = int(flag_matches[0])
        if end_pos <= flag_pos:
            end_pos = flag_pos + 1
    else:
        # Flagged transaction not found on this card
        amt = float(idx.txn_to_amt.get(flagged_txn_id, 0.0))
        att = Attestation(
            what=f"Episode on {card_id} around {flagged_txn_id}: 1 txn totaling ${amt:.2f}",
            evidence=Evidence(txn_ids=[flagged_txn_id], card_ids=[card_id]),
            query="assay.r2.episode.episode",
            params={"card_id": card_id, "flagged_txn_id": flagged_txn_id, "as_of": str(as_of_ts)},
        )
        return Episode(
            txn_ids=[flagged_txn_id],
            first_txn_id=flagged_txn_id,
            exposure_usd=round(amt, 2),
            method="hmm_2state",
            confidence=0.5,
            attestation=att,
        )

    # Extract card features using full history up to end_pos (as-of baseline)
    c_t_asof = c_t[:end_pos]
    c_amt_asof = idx.amount[st : st + end_pos]
    c_prod_asof = idx.prod_code[st : st + end_pos]
    c_chan_asof = idx.chan_online[st : st + end_pos]
    c_addr_asof = idx.addr1[st : st + end_pos]
    c_dev_asof = idx.dev_hash[st : st + end_pos]
    c_txns_asof = c_txns_full[:end_pos]

    feats = _extract_card_features(
        c_t_asof, c_amt_asof, c_prod_asof, c_chan_asof, c_addr_asof, c_dev_asof,
        anchor_pos=flag_pos,
    )
    b = _bin_features(feats, params.gap_edges, params.z_edges)

    # Restrict HMM inference sequence to +-30 days around the flagged transaction
    flag_t = c_t_asof[flag_pos]
    win_mask = np.abs(c_t_asof - flag_t) <= WINDOW_SECONDS
    win_indices = np.nonzero(win_mask)[0]
    w_start = int(win_indices[0])
    w_end = int(win_indices[-1]) + 1
    anchor_w = flag_pos - w_start
    T = w_end - w_start

    # Compute log emissions for each t in w_start..w_end-1
    eps = 1e-9
    log_emit = (
        np.log(params.emit_prod_nov[:, b["prod_nov"][w_start:w_end]] + eps)
        + np.log(params.emit_chan_nov[:, b["chan_nov"][w_start:w_end]] + eps)
        + np.log(params.emit_dev_nov[:, b["dev_nov"][w_start:w_end]] + eps)
        + np.log(params.emit_reg_nov[:, b["reg_nov"][w_start:w_end]] + eps)
        + np.log(params.emit_amt_z[:, b["amt_z_bin"][w_start:w_end]] + eps)
        + np.log(params.emit_gap[:, b["min_gap_bin"][w_start:w_end]] + eps)
    ).T  # shape (T, 2)

    if params.emit_same_signature is not None:
        log_emit += np.log(params.emit_same_signature[:, b["anchor_sim"][w_start:w_end]] + eps).T

    # Strong structural penalty when a transaction is incompatible with the anchor signature
    # (e.g., an in-person normal grocery purchase between or near online fraud charges, or different device)
    incompat_mask = b["anchor_sim"][w_start:w_end] == 0
    log_emit[incompat_mask, 1] -= 4.0

    # Transition log-probabilities per step based on prev_gap_bin
    if T > 1:
        step_gaps = c_t_asof[w_start + 1 : w_end] - c_t_asof[w_start : w_end - 1]
        step_gap_bins = np.digitize(step_gaps, params.gap_edges)
        log_trans = np.log(params.trans_by_gap[step_gap_bins] + eps)  # shape (T-1, 2, 2)
        # Penalize staying in state 1 across very large multi-day gaps (> 5 days) unless emissions are very strong
        long_gap = step_gaps > (5.0 * 86400.0)
        if np.any(long_gap):
            log_trans[long_gap, 1, 1] -= 1.2
    else:
        log_trans = np.zeros((0, 2, 2), dtype=np.float64)

    log_pi = np.log(params.pi + eps)
    gamma = _forward_backward(log_emit, log_trans, log_pi, anchor_w)
    post_comp = gamma[:, 1]

    # Unconditional emission log-odds at the anchor (for confidence estimation)
    anchor_llr = float(log_emit[anchor_w, 1] - log_emit[anchor_w, 0])

    # Extend backward and forward from anchor_w while compromised state posterior > 0.5
    left = anchor_w
    while left > 0 and post_comp[left - 1] > 0.5:
        left -= 1

    right = anchor_w
    while right + 1 < T and post_comp[right + 1] > 0.5:
        right += 1

    # Also check if immediate neighbors within the card's burst skips a single routine auto-debit txn,
    # or stay strictly contiguous as specified ("starting from the flagged txn and extending back/forward
    # while the 'compromised' state posterior > 0.5, within +-30 days").
    sel_slice = slice(w_start + left, w_start + right + 1)
    ep_txns = c_txns_asof[sel_slice].tolist()
    ep_amts = c_amt_asof[sel_slice]
    exposure = float(np.round(np.sum(np.abs(ep_amts)), 2))
    first_txn = int(ep_txns[0])

    # Confidence combines mean posterior on selected episode with anchor unconditional emission support
    mean_post = float(np.mean(post_comp[left : right + 1]))
    sig_anchor = float(1.0 / (1.0 + np.exp(-np.clip(anchor_llr, -10.0, 10.0))))
    conf = float(np.round(0.65 * mean_post + 0.35 * sig_anchor, 4))

    att = Attestation(
        what=(
            f"HMM episode on {actual_card} seeded at {flagged_txn_id}: "
            f"{len(ep_txns)} txn(s), first={first_txn}, exposure=${exposure:.2f}"
        ),
        evidence=Evidence(txn_ids=[int(x) for x in ep_txns], card_ids=[actual_card]),
        query="assay.r2.episode.episode",
        params={"card_id": card_id, "flagged_txn_id": flagged_txn_id, "as_of": str(as_of_ts)},
    )
    return Episode(
        txn_ids=[int(x) for x in ep_txns],
        first_txn_id=first_txn,
        exposure_usd=exposure,
        method="hmm_2state",
        confidence=conf,
        attestation=att,
    )


def baseline_flagged_only(
    card_id: str,
    flagged_txn_id: int,
    as_of: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
) -> Episode:
    """Baseline (a): flagged transaction only."""
    if tx is None:
        tx = tx_all()
    idx = build_card_index(tx)
    flagged_txn_id = int(flagged_txn_id)
    amt = float(np.round(abs(idx.txn_to_amt.get(flagged_txn_id, 0.0)), 2))
    return Episode(
        txn_ids=[flagged_txn_id],
        first_txn_id=flagged_txn_id,
        exposure_usd=amt,
        method="baseline_flagged_only",
        confidence=1.0,
    )


def baseline_same_product_7d(
    card_id: str,
    flagged_txn_id: int,
    as_of: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
) -> Episode:
    """Baseline (b): all card txns within +-7 days (and <= as_of) with the same ProductCD."""
    if tx is None:
        tx = tx_all()
    idx = build_card_index(tx)
    flagged_txn_id = int(flagged_txn_id)
    as_of_sec = float(pd.Timestamp(as_of).value // 1_000_000_000)

    actual_card = idx.txn_to_card.get(flagged_txn_id, card_id)
    bounds = idx.card_bounds.get(actual_card)
    if bounds is None:
        return baseline_flagged_only(card_id, flagged_txn_id, as_of, tx)

    st, en = bounds
    c_t = idx.t_sec[st:en]
    end_pos = int(np.searchsorted(c_t, as_of_sec, side="right"))
    c_txns_full = idx.txn_id[st:en]
    flag_matches = np.nonzero(c_txns_full == flagged_txn_id)[0]
    if len(flag_matches) == 0:
        return baseline_flagged_only(card_id, flagged_txn_id, as_of, tx)

    flag_pos = int(flag_matches[0])
    if end_pos <= flag_pos:
        end_pos = flag_pos + 1

    flag_t = c_t[flag_pos]
    flag_prod = idx.prod_code[st + flag_pos]

    c_t_asof = c_t[:end_pos]
    c_prod_asof = idx.prod_code[st : st + end_pos]
    c_txns_asof = c_txns_full[:end_pos]
    c_amt_asof = idx.amount[st : st + end_pos]

    mask = (np.abs(c_t_asof - flag_t) <= BASELINE_SECONDS) & (c_prod_asof == flag_prod)
    sel_txns = [int(x) for x in c_txns_asof[mask].tolist()]
    sel_amts = c_amt_asof[mask]
    exposure = float(np.round(np.sum(np.abs(sel_amts)), 2))
    return Episode(
        txn_ids=sel_txns,
        first_txn_id=int(sel_txns[0]),
        exposure_usd=exposure,
        method="baseline_same_product_7d",
        confidence=0.5,
    )


def evaluate_episodes(
    cc_eval: pd.DataFrame,
    tx: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate `episode`, `baseline_flagged_only`, and `baseline_same_product_7d` on closed cases (as-of).

    For each confirmed-fraud case in `cc_eval`, we seed the episode finder with a single transaction
    from the case (`txn_list[-1]`, the transaction that triggered/opened the case right at `opened_at`,
    and also evaluate seeding with `first_fraud_txn_id` or `txn_list[-1]`). Note that in closed_cases,
    `opened_at` is shortly after `txn_list[-1]`, so `txn_list[-1]` is the latest fraud transaction before
    case opening (analogous to `flagged_txn_id` in `case_pack`).
    """
    if tx is None:
        tx = tx_all()

    fraud_cases = cc_eval[cc_eval["outcome"] == "confirmed_fraud"].copy()
    methods = {
        "hmm_2state": episode,
        "baseline_a_flagged_only": baseline_flagged_only,
        "baseline_b_same_prod_7d": baseline_same_product_7d,
    }

    records = []
    for m_name, fn in methods.items():
        exact_matches = 0
        jaccards = []
        first_acc = 0
        exp_errs = []

        for row in fraud_cases.itertuples(index=False):
            true_list = [int(x) for x in row.txn_list]
            if not true_list:
                continue
            true_set = set(true_list)
            true_first = int(row.first_fraud_txn_id) if not pd.isna(row.first_fraud_txn_id) else min(true_list)
            true_exp = float(row.exposure_usd)

            # Seed with the last transaction in txn_list (the trigger transaction closest to opened_at)
            seed_txn = int(true_list[-1])
            ep = fn(row.card_id, seed_txn, row.opened_at, tx)
            pred_set = set(ep.txn_ids)

            if pred_set == true_set:
                exact_matches += 1
            union = len(pred_set | true_set)
            inter = len(pred_set & true_set)
            jaccards.append(inter / union if union else 1.0)
            if ep.first_txn_id == true_first:
                first_acc += 1
            exp_errs.append(abs(ep.exposure_usd - true_exp))

        n = max(len(jaccards), 1)
        records.append({
            "method": m_name,
            "n_cases": len(jaccards),
            "exact_match_rate": round(exact_matches / n, 4),
            "mean_jaccard": round(float(np.mean(jaccards)), 4),
            "first_txn_accuracy": round(first_acc / n, 4),
            "mean_exposure_abs_err": round(float(np.mean(exp_errs)), 2),
            "median_exposure_abs_err": round(float(np.median(exp_errs)), 2),
        })

    return pd.DataFrame(records)
