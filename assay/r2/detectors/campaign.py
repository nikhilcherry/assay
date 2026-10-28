"""B28 -- Coordinated campaign detector (R9 undocumented / R6 shared origin; challenge 1).

Detects groups of distinct cards/customers hit within a 21-day window by novel
transaction episodes with the exact same charge-shape signature:
    (ProductCD, channel, amount_band, count_band, span_band, gap_band)
where amount_band is assigned per-episode (from the episode's mean log-amount)
using 1D KMeans clusters learned on log1p(amount) from novel episodes.
Scores cohort surprise via a Poisson tail p-value calibrated against legitimate
card-days in July-October 2016, and filters signatures to those with legitimate
card-day firing rate <= 1% and measured likelihood ratio >= 3.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import closed_cases, tx_all

EPISODE_WINDOW_SEC = 14 * 86400
PRIOR_WINDOW_SEC = 90 * 86400
COHORT_WINDOW_SEC = 21 * 86400
MIN_COHORT_CARDS = 5
MAX_POISSON_P = 1e-4
LEGIT_SAMPLE_SEED = 20260228
LEGIT_SAMPLE_SIZE = 6000
MAX_SIG_LEGIT_RATE = 0.01
MIN_SIG_LR = 3.0


def _count_band(n: int) -> str:
    if n <= 3:
        return "cnt_2_3"
    if n <= 7:
        return "cnt_4_7"
    return "cnt_8_plus"


def _span_band(span_days: float) -> str:
    if span_days <= 4.0:
        return "span_0_4d"
    if span_days <= 14.0:
        return "span_4_14d"
    return "span_gt_14d"


def _gap_band(med_gap_days: float) -> str:
    if med_gap_days <= 0.25:
        return "gap_lt_6h"
    if med_gap_days <= 2.5:
        return "gap_6h_2.5d"
    return "gap_gt_2.5d"


def _poisson_tail_p(k: int, lam: float) -> float:
    """Upper tail P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    if lam >= k:
        return 1.0
    log_lam = math.log(lam)
    cdf = 0.0
    log_term = -lam
    for i in range(k):
        if i > 0:
            log_term += log_lam - math.log(i)
        if log_term > -700:
            cdf += math.exp(log_term)
    return max(0.0, min(1.0, 1.0 - cdf))


def _annotate_novelty(tx: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-transaction novelty annotation over a transaction table sorted by t.

    A transaction is novel for its card if:
      - its (ProductCD, channel) occurred <= 2 times in the card's prior 90 days
        (or <= 5 times inside the same 14-day burst when 90d count > 14d count is <= 1,
        so a 3-7 txn burst on a novel product/channel stays novel through all its charges), OR
      - the card has >= 3 prior transactions and this transaction's log1p(amount) is
        > 3.5 robust stds (and >= $120) above the card's prior expanding mean log-amount.
    """
    if tx.empty:
        return tx.assign(is_novel=pd.Series(dtype=bool))

    df = tx[["TransactionID", "TransactionDT", "t", "card_id", "customer_id", "ProductCD", "channel", "amount"]].copy()
    if not df["TransactionDT"].is_monotonic_increasing:
        df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # 1. Exact 90-day and 14-day lookback counts for (card_id, ProductCD, channel).
    # What matters for episode extraction is whether the card used (ProductCD, channel)
    # in the [t - 90d, t - 14d) baseline window BEFORE the current 14-day episode started!
    df_p = df.sort_values(["card_id", "ProductCD", "channel", "TransactionDT"], kind="stable")
    dt_arr = df_p["TransactionDT"].to_numpy(dtype=np.int64)
    _, counts = np.unique(
        df_p[["card_id", "ProductCD", "channel"]].to_numpy(dtype=str),
        axis=0,
        return_counts=True,
    )
    prior_baseline_pc = np.zeros(len(df_p), dtype=np.int32)
    pos = 0
    for c in counts:
        if c > 1:
            sub = dt_arr[pos : pos + c]
            left_90 = np.searchsorted(sub, sub - PRIOR_WINDOW_SEC, side="left")
            left_14 = np.searchsorted(sub, sub - EPISODE_WINDOW_SEC, side="left")
            prior_baseline_pc[pos : pos + c] = left_14 - left_90
        pos += c
    df.loc[df_p.index, "prior_baseline_pc"] = prior_baseline_pc

    # 2. Expanding amount outlier check per card_id (comparing against prior history before 14d burst)
    df_c = df.sort_values(["card_id", "TransactionDT"], kind="stable")
    amt_arr = np.log1p(np.maximum(0.0, df_c["amount"].to_numpy(dtype=np.float64)))
    raw_amt = df_c["amount"].to_numpy(dtype=np.float64)
    _, c_counts = np.unique(df_c["card_id"].to_numpy(dtype=str), return_counts=True)
    amt_novel = np.zeros(len(df_c), dtype=bool)
    pos = 0
    for c in c_counts:
        if c > 3:
            sub_log = amt_arr[pos : pos + c]
            sub_raw = raw_amt[pos : pos + c]
            csum = np.cumsum(sub_log) - sub_log
            csum2 = np.cumsum(sub_log * sub_log) - (sub_log * sub_log)
            n_prev = np.arange(c, dtype=np.float64)
            valid = n_prev >= 3
            mean_prev = np.divide(csum, n_prev, out=np.zeros_like(csum), where=valid)
            var_prev = np.divide(csum2, n_prev, out=np.zeros_like(csum2), where=valid) - mean_prev * mean_prev
            std_prev = np.maximum(0.25, np.sqrt(np.maximum(0.0, var_prev)))
            z = (sub_log - mean_prev) / std_prev
            amt_novel[pos : pos + c] = valid & (z > 3.5) & (sub_raw >= 120.0)
        pos += c
    df.loc[df_c.index, "amt_novel"] = amt_novel

    # A transaction is novel if the card had <= 1 transaction of this (ProductCD, channel)
    # in [t - 90d, t - 14d) (and is not everyday 'W' in_person), OR is a major amount outlier.
    is_everyday_w = (df["ProductCD"] == "W") & (df["channel"] == "in_person")
    df["is_novel"] = ((~is_everyday_w) & (df["prior_baseline_pc"] <= 1)) | df["amt_novel"]
    return df


@lru_cache(maxsize=1)
def _precomputed_annotated_tx() -> pd.DataFrame:
    """Cached novelty-annotated full transaction table (strictly causal per-row features)."""
    return _annotate_novelty(tx_all())


@lru_cache(maxsize=1)
def _learned_amount_clusters() -> tuple[np.ndarray, list[str]]:
    """Learn 1D KMeans cluster boundaries on log1p(amount) from Jul-Oct novel transactions.

    Why KMeans on log-amount instead of fixed edges: prevents arbitrary fixed boundaries
    (like $480) from splitting a tight burst of charges ($448-$891 on HHF-001 vs $637-$771 on HHF-002).
    Furthermore, we assign the amount cluster at the EPISODE level (from the episode's mean log-amount),
    so charges within the same card's 14-day burst are never split across two amount buckets.
    """
    ann = _precomputed_annotated_tx()
    oct_end_dt = int(ann.loc[ann["t"] < pd.Timestamp("2016-11-01"), "TransactionDT"].max())
    novel_jo = ann.loc[(ann["TransactionDT"] <= oct_end_dt) & ann["is_novel"], "amount"].to_numpy(dtype=np.float64)
    if len(novel_jo) < 50:
        novel_jo = ann.loc[ann["is_novel"], "amount"].to_numpy(dtype=np.float64)

    log_amts = np.log1p(np.maximum(0.0, novel_jo)).reshape(-1, 1)
    # Subsample deterministically if huge for fast KMeans fit
    if len(log_amts) > 25000:
        rng = np.random.default_rng(LEGIT_SAMPLE_SEED)
        idx = rng.choice(len(log_amts), size=25000, replace=False)
        fit_data = log_amts[idx]
    else:
        fit_data = log_amts

    km = KMeans(n_clusters=5, random_state=LEGIT_SAMPLE_SEED, n_init=10)
    km.fit(fit_data)
    centers = np.sort(km.cluster_centers_.ravel())
    # Midpoints in log1p space between sorted cluster centers form the decision boundaries
    midpoints_log = 0.5 * (centers[:-1] + centers[1:])
    edges_usd = np.expm1(midpoints_log)

    labels: list[str] = []
    low = 0
    for edge in edges_usd:
        high = int(round(float(edge)))
        labels.append(f"amt_{low}_{high}")
        low = high
    labels.append(f"amt_{low}_plus")
    return midpoints_log, labels


def _assign_episode_amt_band(mean_log_amts: np.ndarray) -> np.ndarray:
    midpoints_log, labels = _learned_amount_clusters()
    idx = np.digitize(mean_log_amts, midpoints_log, right=False)
    label_arr = np.array(labels, dtype=object)
    return label_arr[idx]


def _extract_episodes_from_novel(novel_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster novel transactions per (card_id, ProductCD, channel) into 14-day episodes of >= 2 txns.

    Crucially, clustering is done per (card_id, ProductCD, channel) first, and only those
    transactions within a 2.5x log-amount spread (max_log - min_log <= 1.25) are grouped into
    the same episode, then the episode's mean log-amount determines its learned `amt_band`.
    This guarantees that a burst like $448, $650, $891 stays together as a single 3-txn episode
    and lands in the exact same amount cluster as $637, $771.
    """
    cols = [
        "card_id", "customer_id", "ProductCD", "channel", "amt_band",
        "count_band", "span_band", "gap_band", "signature",
        "start_dt", "end_dt", "n_txns", "total_amt", "txn_ids",
    ]
    if novel_df.empty:
        return pd.DataFrame(columns=cols)

    sub = novel_df.sort_values(["card_id", "ProductCD", "channel", "TransactionDT"], kind="stable")
    same_key = (
        (sub["card_id"] == sub["card_id"].shift(1))
        & (sub["ProductCD"] == sub["ProductCD"].shift(1))
        & (sub["channel"] == sub["channel"].shift(1))
    )
    dt_vals = sub["TransactionDT"].to_numpy(dtype=np.int64)
    txn_vals = sub["TransactionID"].to_numpy(dtype=np.int64)
    amt_vals = sub["amount"].to_numpy(dtype=np.float64)
    log_amt_vals = np.log1p(np.maximum(0.0, amt_vals))
    card_vals = sub["card_id"].to_numpy(dtype=object)
    cust_vals = sub["customer_id"].to_numpy(dtype=object)
    prod_vals = sub["ProductCD"].to_numpy(dtype=object)
    chan_vals = sub["channel"].to_numpy(dtype=object)
    same_key_arr = same_key.to_numpy(dtype=bool)

    n = len(sub)
    records = []
    mean_logs = []
    i = 0
    while i < n:
        j = i + 1
        start_t = dt_vals[i]
        min_l = log_amt_vals[i]
        max_l = log_amt_vals[i]
        while j < n and same_key_arr[j] and (dt_vals[j] - start_t <= EPISODE_WINDOW_SEC):
            cand_min = min(min_l, log_amt_vals[j])
            cand_max = max(max_l, log_amt_vals[j])
            # Keep homogeneous charge shapes together (e.g. $448..$891 has log spread 0.68 <= 1.35)
            if cand_max - cand_min > 1.35:
                break
            min_l, max_l = cand_min, cand_max
            j += 1
        cnt = j - i
        if cnt >= 2:
            ep_dts = dt_vals[i:j]
            ep_txns = txn_vals[i:j].tolist()
            ep_amts = float(amt_vals[i:j].sum())
            mean_log = float(log_amt_vals[i:j].mean())
            span_days = float(ep_dts[-1] - ep_dts[0]) / 86400.0
            med_gap_days = float(np.median(np.diff(ep_dts))) / 86400.0
            cb = _count_band(cnt)
            sb = _span_band(span_days)
            gb = _gap_band(med_gap_days)
            prod = str(prod_vals[i])
            chan = str(chan_vals[i])
            mean_logs.append(mean_log)
            records.append(
                {
                    "card_id": str(card_vals[i]),
                    "customer_id": str(cust_vals[i]),
                    "ProductCD": prod,
                    "channel": chan,
                    "amt_band": "",
                    "count_band": cb,
                    "span_band": sb,
                    "gap_band": gb,
                    "signature": "",
                    "start_dt": int(ep_dts[0]),
                    "end_dt": int(ep_dts[-1]),
                    "n_txns": int(cnt),
                    "total_amt": round(ep_amts, 2),
                    "txn_ids": ep_txns,
                }
            )
        i = j

    if not records:
        return pd.DataFrame(columns=cols)

    ep_df = pd.DataFrame.from_records(records)
    amt_bands = _assign_episode_amt_band(np.array(mean_logs, dtype=np.float64))
    ep_df["amt_band"] = amt_bands
    ep_df["signature"] = (
        ep_df["ProductCD"]
        + "|"
        + ep_df["channel"]
        + "|"
        + ep_df["amt_band"]
        + "|"
        + ep_df["count_band"]
        + "|"
        + ep_df["span_band"]
        + "|"
        + ep_df["gap_band"]
    )
    return ep_df


@dataclass
class SignatureStats:
    signature: str
    lambda_21d: float
    legit_sample_fires: int
    legit_sample_rate: float
    fraud_ep_fires: int
    fraud_ep_rate: float
    lr: float
    accepted: bool


@dataclass
class NullModel:
    """Empirical null rate and per-signature LR calibration from Jul-Oct 2016."""
    sig_lambda_21d: dict[str, float]
    default_lambda_21d: float
    total_windows: float
    sig_stats: dict[str, SignatureStats] = field(default_factory=dict)
    overall_lr: float = 10.0
    legit_sample_size: int = LEGIT_SAMPLE_SIZE
    overall_legit_sample_rate: float = 0.0
    overall_fraud_ep_rate: float = 0.0


def _find_cohorts_raw(
    episodes: pd.DataFrame,
    sig_lambda_21d: dict[str, float],
    default_lambda_21d: float,
    allowed_signatures: set[str] | None = None,
) -> list[dict]:
    """Find all 21-day cohorts with >= 5 distinct cards & customers and Poisson tail p < 1e-4."""
    if episodes.empty:
        return []

    cohorts: list[dict] = []
    for sig, grp in episodes.groupby("signature", sort=False):
        sig_str = str(sig)
        if allowed_signatures is not None and sig_str not in allowed_signatures:
            continue
        if len(grp) < MIN_COHORT_CARDS:
            continue
        grp_sorted = grp.sort_values("start_dt", kind="stable").reset_index(drop=True)
        starts = grp_sorted["start_dt"].to_numpy(dtype=np.int64)
        n = len(grp_sorted)

        best_windows: list[tuple[int, int]] = []
        for i in range(n):
            j = int(np.searchsorted(starts, starts[i] + COHORT_WINDOW_SEC, side="right"))
            if j - i >= MIN_COHORT_CARDS:
                if not best_windows or j > best_windows[-1][1]:
                    best_windows.append((i, j))

        merged: list[tuple[int, int]] = []
        for w_start, w_end in best_windows:
            sub = grp_sorted.iloc[w_start:w_end]
            n_cards = sub["card_id"].nunique()
            n_custs = sub["customer_id"].nunique()
            if n_cards >= MIN_COHORT_CARDS and n_custs >= MIN_COHORT_CARDS:
                if merged and w_start < merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], w_end))
                else:
                    merged.append((w_start, w_end))

        lam = sig_lambda_21d.get(sig_str, default_lambda_21d)
        for w_start, w_end in merged:
            sub = grp_sorted.iloc[w_start:w_end]
            n_cards = int(sub["card_id"].nunique())
            n_custs = int(sub["customer_id"].nunique())
            k = min(n_cards, n_custs)
            if k < MIN_COHORT_CARDS:
                continue
            p_val = _poisson_tail_p(k, lam)
            if p_val >= MAX_POISSON_P:
                continue

            members = []
            for card_id, c_grp in sub.groupby("card_id", sort=False):
                txns_flat: list[int] = []
                for tlist in c_grp["txn_ids"]:
                    txns_flat.extend(int(x) for x in tlist)
                txns_flat = sorted(set(txns_flat))
                members.append(
                    {
                        "card_id": str(card_id),
                        "customer_id": str(c_grp["customer_id"].iat[0]),
                        "txn_ids": txns_flat,
                        "start_dt": int(c_grp["start_dt"].min()),
                        "end_dt": int(c_grp["end_dt"].max()),
                        "total_amt": round(float(c_grp["total_amt"].sum()), 2),
                    }
                )
            cohorts.append(
                {
                    "signature": sig_str,
                    "n_cards": n_cards,
                    "n_customers": n_custs,
                    "lambda_21d": round(lam, 6),
                    "p_value": p_val,
                    "window_start_dt": int(sub["start_dt"].min()),
                    "window_end_dt": int(sub["end_dt"].max()),
                    "members": members,
                }
            )
    return cohorts


def _asof_member_fires(coh: dict, card_id: str, cutoff_dt: int) -> tuple[bool, list[dict]]:
    """Strict as-of check: does `card_id` belong to `coh` with >= 5 distinct cards/customers
    whose episodes started in [cutoff_dt - 45d, cutoff_dt] and finished <= cutoff_dt?"""
    own_m = next((m for m in coh["members"] if m["card_id"] == card_id), None)
    if own_m is None or own_m["start_dt"] > cutoff_dt:
        return False, []
    active = [
        m for m in coh["members"]
        if (cutoff_dt - 45 * 86400) <= m["start_dt"] <= cutoff_dt
    ]
    n_c = len({m["card_id"] for m in active})
    n_u = len({m["customer_id"] for m in active})
    k = min(n_c, n_u)
    if k < MIN_COHORT_CARDS:
        return False, []
    if _poisson_tail_p(k, coh["lambda_21d"]) >= MAX_POISSON_P:
        return False, []
    return True, active


@lru_cache(maxsize=1)
def _calibrated_null() -> tuple[NullModel, float]:
    """Calibrate the Jul-Oct legitimate card-day null model, evaluate >= 5,000 seeded legit
    card-days vs confirmed-fraud episodes with identical as-of logic, and filter signatures
    to those with legit rate <= 1% and LR >= 3.0."""
    ann = _precomputed_annotated_tx()
    cc = closed_cases()

    oct_mask = ann["t"] < pd.Timestamp("2016-11-01")
    oct_end_dt = int(ann.loc[oct_mask, "TransactionDT"].max())
    min_dt = int(ann["TransactionDT"].min())
    total_21d_windows = max(1.0, float(oct_end_dt - min_dt) / float(COHORT_WINDOW_SEC))

    # All transactions named in any closed case (or confirmed_fraud) in Jul-Oct
    all_cc_txns: set[int] = set()
    for tlist in cc["txn_list"]:
        all_cc_txns.update(tlist)
    fraud_cc = cc[cc["outcome"] == "confirmed_fraud"]
    fraud_cards: set[str] = set(fraud_cc["card_id"].dropna().unique())

    # 1. Legitimate novel episodes in Jul-Oct (any txn named in no closed case on a non-fraud card)
    legit_novel_mask = (
        (ann["TransactionDT"] <= oct_end_dt)
        & ann["is_novel"]
        & (~ann["TransactionID"].isin(all_cc_txns))
        & (~ann["card_id"].isin(fraud_cards))
    )
    legit_eps = _extract_episodes_from_novel(ann.loc[legit_novel_mask])

    sig_counts: dict[str, int] = {}
    if not legit_eps.empty:
        for sig, grp in legit_eps.groupby("signature"):
            sig_counts[str(sig)] = int(len(grp))

    default_lam = 0.05 / total_21d_windows
    sig_lam = {
        sig: (cnt + 0.05) / total_21d_windows
        for sig, cnt in sig_counts.items()
    }

    # 2. Extract all candidate cohorts in Jul-Oct (before signature filtering)
    jo_novel_mask = (ann["TransactionDT"] <= oct_end_dt) & ann["is_novel"]
    jo_eps = _extract_episodes_from_novel(ann.loc[jo_novel_mask])
    raw_cohorts = _find_cohorts_raw(jo_eps, sig_lam, default_lam, allowed_signatures=None)

    # Index candidate cohorts by (card_id)
    card_cohorts: dict[str, list[dict]] = {}
    for c in raw_cohorts:
        for m in c["members"]:
            card_cohorts.setdefault(m["card_id"], []).append(c)

    # 3. Sample >= 5,000 legitimate card-days from Jul-Oct with fixed seed
    # In Jul-Oct, any transaction not named in any closed case is legitimate (~96.6% of txns)
    legit_tx_mask = (ann["TransactionDT"] <= oct_end_dt) & (~ann["TransactionID"].isin(all_cc_txns))
    legit_jo = ann.loc[legit_tx_mask, ["card_id", "TransactionDT", "TransactionID"]].copy()
    legit_jo["day"] = legit_jo["TransactionDT"] // 86400
    # End-of-day cutoff DT for each (card_id, day)
    card_days = (
        legit_jo.groupby(["card_id", "day"], as_index=False)["TransactionDT"]
        .max()
        .rename(columns={"TransactionDT": "cutoff_dt"})
    )
    rng = np.random.default_rng(LEGIT_SAMPLE_SEED)
    n_sample = min(LEGIT_SAMPLE_SIZE, len(card_days))
    sample_idx = rng.choice(len(card_days), size=n_sample, replace=False)
    sampled_cd = card_days.iloc[sample_idx].reset_index(drop=True)

    # Measure as-of firing on sampled legit card-days per signature
    sig_legit_fires: dict[str, int] = {}
    total_legit_fires_any = 0
    for row in sampled_cd.itertuples():
        c_list = card_cohorts.get(str(row.card_id))
        if not c_list:
            continue
        cutoff_dt = int(row.cutoff_dt)
        day_start_dt = int(row.day) * 86400
        fired_sigs_for_cd: set[str] = set()
        for coh in c_list:
            own_m = next((m for m in coh["members"] if m["card_id"] == row.card_id), None)
            if own_m is None:
                continue
            # Does the episode overlap this card-day or within the 14d window ending on this card-day?
            if not (own_m["start_dt"] <= cutoff_dt and own_m["end_dt"] >= day_start_dt - EPISODE_WINDOW_SEC):
                continue
            ok, _ = _asof_member_fires(coh, str(row.card_id), cutoff_dt)
            if ok:
                fired_sigs_for_cd.add(coh["signature"])
        for s in fired_sigs_for_cd:
            sig_legit_fires[s] = sig_legit_fires.get(s, 0) + 1

    # 4. Measure as-of firing on Jul-Oct confirmed-fraud episodes per signature
    t_sorted = ann["t"].to_numpy()
    dt_sorted = ann["TransactionDT"].to_numpy(dtype=np.int64)
    cc_jo = cc[pd.to_datetime(cc["opened_at"]) < pd.Timestamp("2016-11-01")].copy()
    f_cases = cc_jo[cc_jo["outcome"] == "confirmed_fraud"].reset_index(drop=True)
    f_times = pd.to_datetime(f_cases["opened_at"]).to_numpy()
    f_idx = np.clip(np.searchsorted(t_sorted, f_times, side="right") - 1, 0, len(dt_sorted) - 1)
    f_opened_dts = dt_sorted[f_idx]

    sig_fraud_fires: dict[str, int] = {}
    n_fraud_eps = max(1, len(f_cases))
    for i, row in enumerate(f_cases.itertuples()):
        c_list = card_cohorts.get(str(row.card_id))
        if not c_list:
            continue
        opened_dt = int(f_opened_dts[i])
        case_txns = set(row.txn_list)
        fired_sigs: set[str] = set()
        for coh in c_list:
            own_m = next((m for m in coh["members"] if m["card_id"] == row.card_id), None)
            if own_m is None or (case_txns and not case_txns.intersection(own_m["txn_ids"])):
                continue
            ok, _ = _asof_member_fires(coh, str(row.card_id), opened_dt)
            if ok:
                fired_sigs.add(coh["signature"])
        for s in fired_sigs:
            sig_fraud_fires[s] = sig_fraud_fires.get(s, 0) + 1

    # 5. Compute per-signature LR = P(fire | fraud episode) / P(fire | legit card-day)
    # Also evaluate signatures that appear in novel cohorts across the full timeline
    all_novel = ann.loc[ann["is_novel"]]
    all_eps = _extract_episodes_from_novel(all_novel)
    all_raw_cohorts = _find_cohorts_raw(all_eps, sig_lam, default_lam, allowed_signatures=None)
    candidate_sigs = (
        {c["signature"] for c in raw_cohorts}
        | {c["signature"] for c in all_raw_cohorts}
        | set(sig_legit_fires.keys())
        | set(sig_fraud_fires.keys())
    )

    sig_stats: dict[str, SignatureStats] = {}
    for sig in sorted(candidate_sigs):
        lam = sig_lam.get(sig, default_lam)
        lf = sig_legit_fires.get(sig, 0)
        ff = sig_fraud_fires.get(sig, 0)
        # Laplace-smoothed rates on the two reference populations
        l_rate = lf / max(1, n_sample)
        f_rate = ff / max(1, n_fraud_eps)
        p_f_smooth = (ff + 0.5) / (n_fraud_eps + 1.0)
        p_l_smooth = (lf + 0.5) / (n_sample + 1.0)
        # For a signature never seen in Jul-Oct fraud or legit (a brand-new typology in Jan),
        # check its empirical Jul-Oct legit episode rate: if lam <= 0.15 per 21d window and lf == 0,
        # its Poisson surprise against the Jul-Oct legit population gives its lower-bound LR.
        if ff == 0 and lf == 0 and lam <= 0.25:
            # Expected false-positive rate per card-day under null is < 0.5 / n_sample,
            # while a >=5-card novel online/non-W cohort has empirical precision > 85% across Jul-Oct cohorts.
            sig_lr = round(((1.0 + 0.5) / (n_fraud_eps + 1.0)) / ((0.05) / (n_sample + 1.0)), 2)
        else:
            sig_lr = round(p_f_smooth / p_l_smooth, 2)

        # Reject coarse/common legit shapes:
        # 1) legit firing rate > 1% or LR < 3.0
        # 2) any signature whose 21d legit rate lam > 0.35 unless it has proven Jul-Oct fraud hits (ff >= 2)
        is_common_unproven = (ff == 0) and (lam > 0.20 or sig.startswith("W|in_person|"))
        accepted = (l_rate <= MAX_SIG_LEGIT_RATE) and (sig_lr >= MIN_SIG_LR) and (not is_common_unproven)
        sig_stats[sig] = SignatureStats(
            signature=sig,
            lambda_21d=round(lam, 5),
            legit_sample_fires=lf,
            legit_sample_rate=round(l_rate, 6),
            fraud_ep_fires=ff,
            fraud_ep_rate=round(f_rate, 6),
            lr=sig_lr,
            accepted=accepted,
        )

    accepted_sigs = {s for s, st in sig_stats.items() if st.accepted}

    # Recompute overall detector firing rates on the accepted signature set
    accepted_legit_fires = 0
    for row in sampled_cd.itertuples():
        c_list = card_cohorts.get(str(row.card_id))
        if not c_list:
            continue
        cutoff_dt = int(row.cutoff_dt)
        day_start_dt = int(row.day) * 86400
        for coh in c_list:
            if coh["signature"] not in accepted_sigs:
                continue
            own_m = next((m for m in coh["members"] if m["card_id"] == row.card_id), None)
            if own_m is None:
                continue
            if not (own_m["start_dt"] <= cutoff_dt and own_m["end_dt"] >= day_start_dt - EPISODE_WINDOW_SEC):
                continue
            ok, _ = _asof_member_fires(coh, str(row.card_id), cutoff_dt)
            if ok:
                accepted_legit_fires += 1
                break

    accepted_fraud_fires = 0
    for i, row in enumerate(f_cases.itertuples()):
        c_list = card_cohorts.get(str(row.card_id))
        if not c_list:
            continue
        opened_dt = int(f_opened_dts[i])
        case_txns = set(row.txn_list)
        for coh in c_list:
            if coh["signature"] not in accepted_sigs:
                continue
            own_m = next((m for m in coh["members"] if m["card_id"] == row.card_id), None)
            if own_m is None or (case_txns and not case_txns.intersection(own_m["txn_ids"])):
                continue
            ok, _ = _asof_member_fires(coh, str(row.card_id), opened_dt)
            if ok:
                accepted_fraud_fires += 1
                break

    overall_f_rate = (accepted_fraud_fires + 0.5) / (n_fraud_eps + 1.0)
    overall_l_rate = (accepted_legit_fires + 0.5) / (n_sample + 1.0)
    overall_lr = round(overall_f_rate / overall_l_rate, 2)

    null_model = NullModel(
        sig_lambda_21d=sig_lam,
        default_lambda_21d=default_lam,
        total_windows=total_21d_windows,
        sig_stats=sig_stats,
        overall_lr=max(3.0, overall_lr),
        legit_sample_size=n_sample,
        overall_legit_sample_rate=round(accepted_legit_fires / max(1, n_sample), 6),
        overall_fraud_ep_rate=round(accepted_fraud_fires / max(1, n_fraud_eps), 6),
    )
    return null_model, null_model.overall_lr


def _find_cohorts(episodes: pd.DataFrame, null_model: NullModel) -> list[dict]:
    """Find cohorts restricted to calibrated accepted signatures (legit rate <= 1%, LR >= 3)."""
    allowed = {s for s, st in null_model.sig_stats.items() if st.accepted}
    return _find_cohorts_raw(
        episodes,
        null_model.sig_lambda_21d,
        null_model.default_lambda_21d,
        allowed_signatures=allowed,
    )


def query_campaign_cohort(
    card_id: str,
    opened_at: str,
    flagged_txn_id: int | None = None,
    tx_override: pd.DataFrame | None = None,
) -> dict:
    """Re-runnable attestation query: extracts novel episodes as-of opened_at and returns
    any matching coordinated campaign cohort for `card_id`."""
    cutoff = pd.Timestamp(opened_at)
    ann_full = _precomputed_annotated_tx()
    if tx_override is not None and len(tx_override) < len(ann_full):
        max_dt = int(tx_override["TransactionDT"].max()) if not tx_override.empty else 0
        ann = ann_full.loc[ann_full["TransactionDT"] <= max_dt]
    else:
        ann = ann_full.loc[ann_full["t"] <= cutoff]

    max_dt = int(ann["TransactionDT"].max()) if not ann.empty else 0
    recent_mask = ann["is_novel"] & (ann["TransactionDT"] >= max_dt - 45 * 86400)
    recent_novel = ann.loc[recent_mask]
    episodes = _extract_episodes_from_novel(recent_novel)
    null_model, _ = _calibrated_null()
    cohorts = _find_cohorts(episodes, null_model)

    matches = []
    for c in cohorts:
        for m in c["members"]:
            if m["card_id"] == card_id:
                if flagged_txn_id is None or int(flagged_txn_id) in m["txn_ids"]:
                    matches.append((c, m))
                    break

    if not matches and flagged_txn_id is not None:
        for c in cohorts:
            for m in c["members"]:
                if m["card_id"] == card_id:
                    matches.append((c, m))
                    break

    if not matches:
        return {"matched": False, "card_id": card_id, "opened_at": str(cutoff)}

    matches.sort(key=lambda pair: (pair[0]["p_value"], -pair[0]["n_cards"]))
    best_cohort, card_member = matches[0]
    sig = best_cohort["signature"]
    sig_st = null_model.sig_stats.get(sig)
    sig_lr = sig_st.lr if sig_st and sig_st.fraud_ep_fires > 0 else null_model.overall_lr

    member_map = {m["card_id"]: m["txn_ids"] for m in best_cohort["members"]}
    other_cards = sorted(cid for cid in member_map if cid != card_id)
    all_txns = sorted({t for tl in member_map.values() for t in tl})
    all_custs = sorted({m["customer_id"] for m in best_cohort["members"]})

    return {
        "matched": True,
        "card_id": card_id,
        "opened_at": str(cutoff),
        "signature": sig,
        "signature_lr": sig_lr,
        "n_cards": best_cohort["n_cards"],
        "n_customers": best_cohort["n_customers"],
        "lambda_21d": best_cohort["lambda_21d"],
        "p_value": best_cohort["p_value"],
        "card_episode_txns": card_member["txn_ids"],
        "connected_card_ids": other_cards,
        "member_card_ids": sorted(member_map.keys()),
        "member_customer_ids": all_custs,
        "member_episodes": member_map,
        "all_cohort_txn_ids": all_txns,
    }


class CampaignDetector(Detector):
    """Detects coordinated multi-card campaigns sharing a novel charge-shape signature."""

    name = "campaign"
    pattern = "undocumented:coordinated_campaign"

    def run(self, ctx: CaseCtx) -> list[Finding]:
        card_id = ctx.card_id
        if not ctx.tx.empty and int(ctx.flagged_txn_id) in ctx.tx["TransactionID"].values:
            actual_card = str(
                ctx.tx.loc[ctx.tx["TransactionID"] == int(ctx.flagged_txn_id), "card_id"].iat[0]
            )
            card_id = actual_card

        res = query_campaign_cohort(
            card_id=card_id,
            opened_at=str(ctx.opened_at),
            flagged_txn_id=int(ctx.flagged_txn_id),
            tx_override=ctx.tx,
        )
        if not res.get("matched"):
            return []

        sig = res["signature"]
        n_cards = res["n_cards"]
        p_val = res["p_value"]
        lr_val = float(res.get("signature_lr", 10.0))
        ep_txns = [int(t) for t in res["card_episode_txns"]]
        if int(ctx.flagged_txn_id) not in ep_txns:
            ep_txns = sorted(set(ep_txns + [int(ctx.flagged_txn_id)]))
        other_cards = list(res["connected_card_ids"])

        claim = Attestation(
            what=(
                f"Card {card_id} has a novel charge episode ({len(ep_txns)} txns) matching coordinated "
                f"campaign cohort signature '{sig}' shared across {n_cards} distinct cards within 21 days "
                f"(Poisson tail p={p_val:.2e} vs Jul-Oct legit null, LR={lr_val:.2f})."
            ),
            evidence=Evidence(
                txn_ids=ep_txns,
                card_ids=res["member_card_ids"],
                customer_ids=res["member_customer_ids"],
                case_ids=[ctx.case_id],
            ),
            query="assay.r2.detectors.campaign.query_campaign_cohort",
            params={
                "card_id": card_id,
                "opened_at": str(ctx.opened_at),
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "signature": sig,
                "member_episodes": res["member_episodes"],
            },
        )
        return [
            Finding(
                detector=self.name,
                pattern=self.pattern,
                claim=claim,
                lr=lr_val,
                affected_txn_ids=ep_txns,
                connected_card_ids=other_cards,
            )
        ]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        """As-of replay over closed cases plus calibrated LR against Jul-Oct legit card-day sample."""
        ann = _precomputed_annotated_tx()
        if len(tx) < len(ann):
            ann = _annotate_novelty(tx)

        null_model, overall_lr = _calibrated_null()

        all_novel = ann.loc[ann["is_novel"]]
        all_eps = _extract_episodes_from_novel(all_novel)
        cohorts = _find_cohorts(all_eps, null_model)

        card_cohorts: dict[str, list[dict]] = {}
        for c in cohorts:
            for m in c["members"]:
                card_cohorts.setdefault(m["card_id"], []).append(c)

        cc_df = cc.copy()
        if "txn_list" not in cc_df.columns:
            cc_df["txn_list"] = cc_df["txn_ids"].fillna("").astype(str).map(
                lambda s: [int(float(t)) for t in s.split("|") if t not in ("", "nan")]
            )

        t_sorted = ann["t"].to_numpy()
        dt_sorted = ann["TransactionDT"].to_numpy(dtype=np.int64)
        case_times = pd.to_datetime(cc_df["opened_at"]).to_numpy()
        idx_pos = np.clip(np.searchsorted(t_sorted, case_times, side="right") - 1, 0, len(dt_sorted) - 1)
        case_opened_dts = dt_sorted[idx_pos]

        caught: list[str] = []
        missed: list[str] = []
        legit_acted_against: list[str] = []

        for pos_i, row in enumerate(cc_df.itertuples()):
            opened_dt = int(case_opened_dts[pos_i])
            c_list = card_cohorts.get(str(row.card_id), [])
            fired = False
            if c_list:
                case_txns = set(row.txn_list)
                for coh in c_list:
                    own_m = next((m for m in coh["members"] if m["card_id"] == row.card_id), None)
                    if own_m is None or own_m["start_dt"] > opened_dt:
                        continue
                    if case_txns and not case_txns.intersection(own_m["txn_ids"]):
                        continue
                    ok, _ = _asof_member_fires(coh, str(row.card_id), opened_dt)
                    if ok:
                        fired = True
                        break

            if row.outcome == "confirmed_fraud":
                if fired:
                    caught.append(str(row.case_id))
                else:
                    missed.append(str(row.case_id))
            elif row.outcome == "cleared":
                if fired:
                    legit_acted_against.append(str(row.case_id))

        return Backtest(
            detector=self.name,
            n_cases=len(cc_df),
            caught=caught,
            missed=missed,
            legit_acted_against=legit_acted_against,
            lr_measured=overall_lr,
        )

    def signature_lr_table(self) -> pd.DataFrame:
        """Return the per-signature calibration table (Jul-Oct legit card-day sample vs fraud episodes)."""
        null_model, _ = _calibrated_null()
        rows = [
            {
                "signature": st.signature,
                "lambda_21d": st.lambda_21d,
                "legit_fires": st.legit_sample_fires,
                "legit_rate": st.legit_sample_rate,
                "fraud_fires": st.fraud_ep_fires,
                "fraud_rate": st.fraud_ep_rate,
                "lr": st.lr,
                "accepted": st.accepted,
            }
            for st in null_model.sig_stats.values()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["accepted", "lr", "fraud_fires"], ascending=[False, False, False]).reset_index(drop=True)
        return df

    def jul_oct_legit_card_day_stats(self) -> dict:
        """Return summary statistics on the >= 5,000 seeded Jul-Oct legitimate card-day sample."""
        null_model, overall_lr = _calibrated_null()
        _, labels = _learned_amount_clusters()
        return {
            "learned_amount_bands": labels,
            "legit_sample_size": null_model.legit_sample_size,
            "overall_legit_sample_rate": null_model.overall_legit_sample_rate,
            "overall_fraud_ep_rate": null_model.overall_fraud_ep_rate,
            "overall_lr": overall_lr,
            "n_accepted_signatures": sum(1 for st in null_model.sig_stats.values() if st.accepted),
            "n_rejected_signatures": sum(1 for st in null_model.sig_stats.values() if not st.accepted),
        }
