"""BLOCK C7 -- Holder-level session shift detector ("holder_shift").

Block C4 measured new-device takeover detectors at the card_id level and failed
the keep gate against cleared cases because card_id is an issuer bucket (many
holders). tx_all() has a `uid` column: a holder key (card + billing region +
first-use day). This block re-measures session shift at the holder (`uid`) level:

For the flagged txn's uid, strictly as-of t <= opened_at:
- Did the holder's activity in the last 48h (and separately 7d) move to a
  device_profile AND/OR P_emaildomain the HOLDER (uid) never used before (strictly
  prior to the window), on online charges, with amount >= m x the holder's own
  prior median (m from {2, 3})?
- Requires the holder (uid) to have >= 10 prior transactions before the window
  (else not applicable -- reported explicitly in diagnostics).
- Backtested as-of over ALL 5,585 closed cases (confirmed_fraud vs CLEARED is the
  primary gate comparison because alerts are already selected) and also reports
  legitimate card-days (sampled with fixed seed=42) for information.
- Weight = 95% CI lower bound of P(fire|fraud)/P(fire|cleared), Laplace-smoothed.
- Keep gate: CI_low > 1.5 AND P(fire|cleared) <= 8% (0.08).
  If the detector fails the gate, it emits nothing by default (`run(ctx)` returns
  `[]` unless `emit_when_ungated=True` is passed for inspection) and documents
  the honest negative.
- Never uses TransactionID ranges or any id threshold as a signal. Only
  behavioural columns strictly as-of t <= opened_at.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding

WINDOWS_HOURS: list[int] = [48, 168]  # 48h and 7d (168h)
M_SWEEP: list[float] = [2.0, 3.0]
SHIFT_MODES: list[str] = ["dev_or_email", "dev_only", "email_only", "dev_and_email"]
MIN_PRIOR_TXNS: int = 10
KEEP_GATE_CI_LOW: float = 1.5
KEEP_GATE_MAX_CLEARED_RATE: float = 0.08
EMPTY_PROFILE: str = "? | ? | ? | ?"


def _clean_str_series(s: pd.Series) -> pd.Series:
    """Normalise string column to clean string with '' for missing/placeholder."""
    out = s.fillna("").astype(str).str.strip()
    return out.where(~out.isin(["", "nan", "None", "null", "?"]), "")


def _clean_profile_series(s: pd.Series) -> pd.Series:
    """Normalise device_profile column to '' when empty or all-question-marks."""
    out = s.fillna("").astype(str).str.strip()
    return out.where(~out.isin(["", "nan", "None", EMPTY_PROFILE]), "")


def safe_rate(k: int, n: int) -> float | None:
    """Guard rate calculation: n <= 0 -> None (insufficient data), else clip(k / n, 0.0, 1.0)."""
    if n <= 0:
        return None
    return float(np.clip(k / n, 0.0, 1.0))


def bootstrap_lr_ci(
    n_pos: int,
    k_pos: int,
    n_neg: int,
    k_neg: int,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None, float | None, bool]:
    """Laplace-smoothed LR = P(fire|pos)/P(fire|neg) and 95% bootstrap CI.

    Guards against n_pos <= 0 or n_neg <= 0 or out-of-range k > n:
    - If n_pos <= 0 or n_neg <= 0, returns (None, None, None, True) indicating
      insufficient data (ineligible for the keep gate).
    - Clips k_pos to [0, n_pos] and k_neg to [0, n_neg] before computing rates
      and binomial draws so `rng.binomial` never receives p < 0, p > 1, or NaN.
    """
    if n_pos <= 0 or n_neg <= 0:
        return None, None, None, True

    k_pos_c = int(np.clip(k_pos, 0, n_pos))
    k_neg_c = int(np.clip(k_neg, 0, n_neg))

    p_pos_lap = (k_pos_c + 1.0) / (n_pos + 2.0)
    p_neg_lap = (k_neg_c + 1.0) / (n_neg + 2.0)
    lr = p_pos_lap / p_neg_lap

    p_pos_emp = float(np.clip(k_pos_c / n_pos, 0.0, 1.0))
    p_neg_emp = float(np.clip(k_neg_c / n_neg, 0.0, 1.0))

    rng = np.random.default_rng(seed)
    pos_draws = rng.binomial(n_pos, p_pos_emp, size=n_boot)
    neg_draws = rng.binomial(n_neg, p_neg_emp, size=n_boot)
    boot_p_pos = (pos_draws + 1.0) / (n_pos + 2.0)
    boot_p_neg = (neg_draws + 1.0) / (n_neg + 2.0)
    boot_lrs = boot_p_pos / boot_p_neg

    ci_low = float(np.percentile(boot_lrs, 2.5))
    ci_high = float(np.percentile(boot_lrs, 97.5))
    return round(float(lr), 4), round(ci_low, 4), round(ci_high, 4), False


def _precompute_uid_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Vectorised / group-indexed per-transaction as-of features at the holder (uid) level.

    For each transaction row i on holder `uid`:
    - For window W in {48h, 168h (7d)}:
      - Let s_W be the index of the first transaction on `uid` with timestamp >= t_i - W.
      - Prior history for `uid` before the window is [0, s_W).
      - `prior_cnt_{W}h` = s_W (number of prior txns on this uid strictly before t_i - W).
      - If s_W >= MIN_PRIOR_TXNS (10) and row i is an online charge:
        - `prior_med_{W}h` = median amount of `uid` in [0, s_W).
        - `dev_novel_{W}h` = row i has a non-empty device_profile never seen on `uid` in [0, s_W).
        - `email_novel_{W}h` = row i has a non-empty P_emaildomain never seen on `uid` in [0, s_W).
        - `amt_ratio_{W}h` = row i amount / `prior_med_{W}h`.
    Also computes total prior txns strictly before t_i (`uid_seq`) for applicability reporting.
    Never uses TransactionID as a feature or threshold.
    """
    df = tx[
        [
            "TransactionID",
            "t",
            "card_id",
            "customer_id",
            "uid",
            "amount",
            "channel",
            "DeviceInfo",
            "device_profile",
            "P_emaildomain",
        ]
    ].copy()
    df = df.sort_values("t", kind="stable").reset_index(drop=True)

    df["prof_clean"] = _clean_profile_series(df["device_profile"])
    df["dev_clean"] = _clean_str_series(df["DeviceInfo"])
    df["p_email"] = _clean_str_series(df["P_emaildomain"]).str.lower()
    df["is_online"] = _clean_str_series(df["channel"]).str.lower() == "online"

    # Sequential index within uid
    df["uid_seq"] = df.groupby("uid", sort=False).cumcount()

    # First timestamp each (uid, prof_clean) and (uid, p_email) was ever seen
    first_prof_t = df.groupby(["uid", "prof_clean"], sort=False)["t"].transform("min")
    first_email_t = df.groupby(["uid", "p_email"], sort=False)["t"].transform("min")
    df["first_prof_t"] = first_prof_t
    df["first_email_t"] = first_email_t

    n = len(df)
    t_ns = df["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    amt_arr = df["amount"].to_numpy(dtype=np.float64)
    online_arr = df["is_online"].to_numpy(dtype=bool)
    prof_valid_arr = (df["prof_clean"] != "").to_numpy(dtype=bool)
    email_valid_arr = (df["p_email"] != "").to_numpy(dtype=bool)
    first_prof_ns = df["first_prof_t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    first_email_ns = df["first_email_t"].to_numpy(dtype="datetime64[ns]").astype(np.int64)

    # Allocate standalone writable numpy arrays for 48h and 168h (7d), then assign to df at the end
    prior_cnt_48 = np.zeros(n, dtype=np.int32)
    prior_med_48 = np.full(n, np.nan, dtype=np.float64)
    dev_nov_48 = np.zeros(n, dtype=bool)
    em_nov_48 = np.zeros(n, dtype=bool)
    ratio_48 = np.zeros(n, dtype=np.float64)

    prior_cnt_168 = np.zeros(n, dtype=np.int32)
    prior_med_168 = np.full(n, np.nan, dtype=np.float64)
    dev_nov_168 = np.zeros(n, dtype=bool)
    em_nov_168 = np.zeros(n, dtype=bool)
    ratio_168 = np.zeros(n, dtype=np.float64)

    ns_48h = int(48 * 3600 * 1_000_000_000)
    ns_168h = int(168 * 3600 * 1_000_000_000)

    # Group directly on df (which has RangeIndex 0..n-1) and skip groups with <= MIN_PRIOR_TXNS rows
    uid_indices = df.groupby("uid", sort=False).indices

    for _, raw_idxs in uid_indices.items():
        if len(raw_idxs) <= MIN_PRIOR_TXNS:
            continue
        idxs = np.asarray(raw_idxs, dtype=np.int64)
        m_len = len(idxs)

        u_t = t_ns[idxs]
        u_amt = amt_arr[idxs]
        u_on = online_arr[idxs]
        u_pv = prof_valid_arr[idxs]
        u_ev = email_valid_arr[idxs]
        u_fpt = first_prof_ns[idxs]
        u_fet = first_email_ns[idxs]

        l48 = np.searchsorted(u_t, u_t - ns_48h, side="left")
        l168 = np.searchsorted(u_t, u_t - ns_168h, side="left")

        for i in range(MIN_PRIOR_TXNS, m_len):
            g_idx = int(idxs[i])
            s48 = int(l48[i])
            s168 = int(l168[i])
            prior_cnt_48[g_idx] = s48
            prior_cnt_168[g_idx] = s168

            if not u_on[i]:
                continue

            # 48h window check
            if s48 >= MIN_PRIOR_TXNS:
                win_start_48 = u_t[i] - ns_48h
                dn48 = bool(u_pv[i] and (u_fpt[i] >= win_start_48))
                en48 = bool(u_ev[i] and (u_fet[i] >= win_start_48))
                if dn48 or en48:
                    med48 = float(np.median(u_amt[:s48]))
                    if med48 > 0:
                        prior_med_48[g_idx] = med48
                        dev_nov_48[g_idx] = dn48
                        em_nov_48[g_idx] = en48
                        ratio_48[g_idx] = u_amt[i] / med48

            # 168h (7d) window check
            if s168 >= MIN_PRIOR_TXNS:
                win_start_168 = u_t[i] - ns_168h
                dn168 = bool(u_pv[i] and (u_fpt[i] >= win_start_168))
                en168 = bool(u_ev[i] and (u_fet[i] >= win_start_168))
                if dn168 or en168:
                    med168 = float(np.median(u_amt[:s168]))
                    if med168 > 0:
                        prior_med_168[g_idx] = med168
                        dev_nov_168[g_idx] = dn168
                        em_nov_168[g_idx] = en168
                        ratio_168[g_idx] = u_amt[i] / med168

    df["prior_cnt_48h"] = prior_cnt_48
    df["prior_med_48h"] = prior_med_48
    df["dev_novel_48h"] = dev_nov_48
    df["email_novel_48h"] = em_nov_48
    df["amt_ratio_48h"] = ratio_48

    df["prior_cnt_168h"] = prior_cnt_168
    df["prior_med_168h"] = prior_med_168
    df["dev_novel_168h"] = dev_nov_168
    df["email_novel_168h"] = em_nov_168
    df["amt_ratio_168h"] = ratio_168

    return df


@lru_cache(maxsize=1)
def get_precomputed_uid_tx_all() -> pd.DataFrame:
    """Cached holder-level feature precomputation over frames.tx_all()."""
    from assay.r2.frames import tx_all

    return _precompute_uid_features(tx_all())


def get_precomputed_uid_for_tx(tx: pd.DataFrame) -> pd.DataFrame:
    """Return precomputed uid features for tx (uses cache if tx is full tx_all())."""
    from assay.r2.frames import tx_all

    full = tx_all()
    if len(tx) == len(full) and tx["TransactionID"].iat[0] == full["TransactionID"].iat[0]:
        return get_precomputed_uid_tx_all()
    return _precompute_uid_features(tx)


@lru_cache(maxsize=1)
def _uid_group_cache() -> dict[str, Any]:
    """Fast numpy array lookup structures over get_precomputed_uid_tx_all() for single-case evaluation."""
    feat_df = get_precomputed_uid_tx_all()
    tid_arr = feat_df["TransactionID"].to_numpy(dtype=np.int64)
    tid_to_pos = dict(zip(tid_arr.tolist(), range(len(tid_arr))))
    uid_groups = {
        str(k): np.asarray(v, dtype=np.int64)
        for k, v in feat_df.groupby("uid", sort=False).indices.items()
    }
    card_groups = {
        str(k): np.asarray(v, dtype=np.int64)
        for k, v in feat_df.groupby("card_id", sort=False).indices.items()
    }
    return {
        "feat_df": feat_df,
        "tid_arr": tid_arr,
        "tid_to_pos": tid_to_pos,
        "uid_groups": uid_groups,
        "card_groups": card_groups,
        "t_ns": feat_df["t"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "uid_arr": feat_df["uid"].to_numpy(),
        "amt_arr": feat_df["amount"].to_numpy(dtype=np.float64),
        "online_arr": feat_df["is_online"].to_numpy(dtype=bool),
        "prof_arr": feat_df["prof_clean"].to_numpy(),
        "email_arr": feat_df["p_email"].to_numpy(),
        "48": {
            "prior_cnt": feat_df["prior_cnt_48h"].to_numpy(dtype=np.int32),
            "prior_med": feat_df["prior_med_48h"].to_numpy(dtype=np.float64),
            "dev_novel": feat_df["dev_novel_48h"].to_numpy(dtype=bool),
            "email_novel": feat_df["email_novel_48h"].to_numpy(dtype=bool),
            "amt_ratio": feat_df["amt_ratio_48h"].to_numpy(dtype=np.float64),
        },
        "168": {
            "prior_cnt": feat_df["prior_cnt_168h"].to_numpy(dtype=np.int32),
            "prior_med": feat_df["prior_med_168h"].to_numpy(dtype=np.float64),
            "dev_novel": feat_df["dev_novel_168h"].to_numpy(dtype=bool),
            "email_novel": feat_df["email_novel_168h"].to_numpy(dtype=bool),
            "amt_ratio": feat_df["amt_ratio_168h"].to_numpy(dtype=np.float64),
        },
    }


def evaluate_single_case_uid(
    flagged_txn_id: int,
    card_id: str,
    opened_at: pd.Timestamp,
    window_hours: int = 48,
    m: float = 2.0,
    shift_mode: str = "dev_or_email",
    tx_override: pd.DataFrame | None = None,
    case_id: str = "",
) -> dict[str, Any]:
    """ONE canonical decision function used by BOTH `backtest()` and `run()` / `inspect()`.

    Given a case's `(flagged_txn_id, card_id, opened_at)`:
    1. Resolves the anchor transaction strictly as-of `t <= opened_at` (`flagged_txn_id`
       if present with `t <= opened_at`, else latest transaction on `card_id` with `t <= opened_at`).
    2. Takes that anchor transaction's `uid` (holder key).
    3. Examines the holder's transactions in `[opened_at - window_hours, opened_at]`
       AND the anchor transaction itself (`flagged_txn_id` on that `uid` with `t <= opened_at`).
    4. For each candidate transaction at timestamp `t_cand`, checks the holder's prior
       history strictly before `t_cand - window_hours`:
       - Requires `prior_cnt >= 10` (`MIN_PRIOR_TXNS`).
       - Checks if the candidate transaction is `online`, has `amount >= m * prior_med`,
         and shifted to a novel `device_profile` and/or `P_emaildomain` (per `shift_mode`)
         never seen on `uid` before `t_cand - window_hours`.
    """
    cache = _uid_group_cache()
    tid_to_pos: dict[int, int] = cache["tid_to_pos"]
    t_ns_all: np.ndarray = cache["t_ns"]
    opened_ns = int(pd.Timestamp(opened_at).to_datetime64().astype("datetime64[ns]").astype(np.int64))

    pos = tid_to_pos.get(int(flagged_txn_id))
    anchor_pos: int | None = None
    if pos is not None and t_ns_all[pos] <= opened_ns:
        anchor_pos = pos
    else:
        c_idxs = cache["card_groups"].get(str(card_id))
        if c_idxs is not None and len(c_idxs) > 0:
            c_t = t_ns_all[c_idxs]
            r_idx = int(np.searchsorted(c_t, opened_ns, side="right"))
            if r_idx > 0:
                anchor_pos = int(c_idxs[r_idx - 1])

    if anchor_pos is None:
        return {
            "case_id": case_id,
            "card_id": card_id,
            "flagged_txn_id": flagged_txn_id,
            "uid": None,
            "window_hours": window_hours,
            "m": m,
            "shift_mode": shift_mode,
            "applicable": False,
            "not_applicable_reason": f"flagged_txn_id {flagged_txn_id} not found as-of {opened_at}",
            "fired": False,
            "prior_cnt": 0,
            "total_uid_txns_as_of": 0,
            "prior_med": None,
            "max_online_amt": None,
            "amt_ratio": None,
            "novel_devices": [],
            "novel_emails": [],
            "affected_txn_ids": [],
        }

    uid = str(cache["uid_arr"][anchor_pos])
    u_idxs_all = cache["uid_groups"].get(uid)
    if u_idxs_all is None or len(u_idxs_all) == 0:
        return {
            "case_id": case_id,
            "card_id": card_id,
            "flagged_txn_id": flagged_txn_id,
            "uid": uid,
            "window_hours": window_hours,
            "m": m,
            "shift_mode": shift_mode,
            "applicable": False,
            "not_applicable_reason": f"uid {uid} has 0 transactions as-of {opened_at}",
            "fired": False,
            "prior_cnt": 0,
            "total_uid_txns_as_of": 0,
            "prior_med": None,
            "max_online_amt": None,
            "amt_ratio": None,
            "novel_devices": [],
            "novel_emails": [],
            "affected_txn_ids": [],
        }

    u_t_all = t_ns_all[u_idxs_all]
    r_bound = int(np.searchsorted(u_t_all, opened_ns, side="right"))
    u_idxs = u_idxs_all[:r_bound]
    u_t = u_t_all[:r_bound]
    total_uid_txns = len(u_idxs)

    w_cols = cache[str(window_hours)]
    prior_cnt_arr = w_cols["prior_cnt"][u_idxs]
    prior_med_arr = w_cols["prior_med"][u_idxs]
    dev_nov_arr = w_cols["dev_novel"][u_idxs]
    em_nov_arr = w_cols["email_novel"][u_idxs]
    ratio_arr = w_cols["amt_ratio"][u_idxs]

    u_tids = cache["tid_arr"][u_idxs]
    u_amts = cache["amt_arr"][u_idxs]
    u_online = cache["online_arr"][u_idxs]
    u_profs = cache["prof_arr"][u_idxs]
    u_emails = cache["email_arr"][u_idxs]

    win_ns = int(window_hours * 3600 * 1_000_000_000)
    win_start_ns = opened_ns - win_ns
    anchor_tid = int(cache["tid_arr"][anchor_pos])

    # Candidate mask on this uid: in [opened_at - W, opened_at] OR the anchor txn itself
    cand_mask = (u_t >= win_start_ns) | (u_tids == anchor_tid)
    if not cand_mask.any():
        max_prior_cnt = int(prior_cnt_arr.max()) if total_uid_txns > 0 else 0
    else:
        max_prior_cnt = int(prior_cnt_arr[cand_mask].max())

    if max_prior_cnt < MIN_PRIOR_TXNS:
        # Also compute actual prior count before win_start_ns for clear reporting when total < 10
        n_before_win = int(np.searchsorted(u_t, win_start_ns, side="left"))
        rep_prior = max(max_prior_cnt, n_before_win)
        if rep_prior >= MIN_PRIOR_TXNS and cand_mask.any():
            # Candidate existed before MIN_PRIOR_TXNS only if anchor was early; keep max_prior_cnt
            rep_prior = max_prior_cnt
        p_med_val = round(float(np.median(u_amts[:n_before_win])), 2) if n_before_win > 0 else None
        return {
            "case_id": case_id,
            "card_id": card_id,
            "flagged_txn_id": flagged_txn_id,
            "uid": uid,
            "window_hours": window_hours,
            "m": m,
            "shift_mode": shift_mode,
            "applicable": False,
            "not_applicable_reason": (
                f"not applicable: holder {uid} has {rep_prior} prior txns before "
                f"{window_hours}h window (< {MIN_PRIOR_TXNS} required; {total_uid_txns} total as-of)"
            ),
            "fired": False,
            "prior_cnt": rep_prior,
            "total_uid_txns_as_of": total_uid_txns,
            "prior_med": p_med_val,
            "max_online_amt": None,
            "amt_ratio": None,
            "novel_devices": [],
            "novel_emails": [],
            "affected_txn_ids": [],
        }

    c_prior = prior_cnt_arr[cand_mask]
    c_med = prior_med_arr[cand_mask]
    c_dev = dev_nov_arr[cand_mask]
    c_em = em_nov_arr[cand_mask]
    c_ratio = ratio_arr[cand_mask]
    c_tids = u_tids[cand_mask]
    c_amts = u_amts[cand_mask]
    c_on = u_online[cand_mask]
    c_profs = u_profs[cand_mask]
    c_emails = u_emails[cand_mask]

    if shift_mode == "dev_or_email":
        shift_mask = c_dev | c_em
    elif shift_mode == "dev_only":
        shift_mask = c_dev
    elif shift_mode == "email_only":
        shift_mask = c_em
    elif shift_mode == "dev_and_email":
        shift_mask = c_dev & c_em
    else:
        raise ValueError(f"Unknown shift_mode: {shift_mode}")

    elig_mask = c_prior >= MIN_PRIOR_TXNS
    online_elig_mask = c_on & elig_mask
    fire_mask = online_elig_mask & shift_mask & (c_ratio >= m)

    fired = bool(fire_mask.any())
    affected_ids = [int(x) for x in c_tids[fire_mask].tolist()] if fired else []

    # Reference median for reporting
    valid_meds = c_med[fire_mask] if fired else c_med[online_elig_mask]
    valid_meds = valid_meds[~np.isnan(valid_meds)]
    if len(valid_meds) > 0:
        prior_med_val: float | None = float(valid_meds[-1])
    else:
        n_before_win = int(np.searchsorted(u_t, win_start_ns, side="left"))
        prior_med_val = float(np.median(u_amts[:n_before_win])) if n_before_win > 0 else None

    if online_elig_mask.any():
        max_online_amt: float | None = float(c_amts[online_elig_mask].max())
        max_r = float(c_ratio[online_elig_mask].max())
        max_ratio: float | None = (
            max_r
            if max_r > 0
            else ((max_online_amt / prior_med_val) if (prior_med_val and prior_med_val > 0) else None)
        )
        novel_devs = sorted({str(x) for x in c_profs[online_elig_mask & c_dev] if str(x) != ""})
        novel_ems = sorted({str(x) for x in c_emails[online_elig_mask & c_em] if str(x) != ""})
    else:
        max_online_amt = None
        max_ratio = None
        novel_devs = []
        novel_ems = []

    return {
        "case_id": case_id,
        "card_id": card_id,
        "flagged_txn_id": flagged_txn_id,
        "uid": uid,
        "window_hours": window_hours,
        "m": m,
        "shift_mode": shift_mode,
        "applicable": True,
        "not_applicable_reason": "",
        "fired": fired,
        "prior_cnt": max_prior_cnt,
        "total_uid_txns_as_of": total_uid_txns,
        "prior_med": round(prior_med_val, 2) if prior_med_val is not None else None,
        "max_online_amt": round(max_online_amt, 2) if max_online_amt is not None else None,
        "amt_ratio": round(max_ratio, 2) if max_ratio is not None else None,
        "novel_devices": novel_devs,
        "novel_emails": novel_ems,
        "affected_txn_ids": affected_ids,
    }


def sample_legit_card_days(
    cc: pd.DataFrame,
    feat_df: pd.DataFrame,
    n_sample: int = 3500,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample legitimate card-days from Jul-Dec (cards never named in any closed case)."""
    named_cards = set(cc["card_id"].dropna().unique())
    for c_list in cc.get("connected_list", []):
        if isinstance(c_list, list):
            named_cards.update(c_list)

    jul_dec = feat_df[
        (feat_df["t"] < pd.Timestamp("2017-01-01"))
        & (~feat_df["card_id"].isin(named_cards))
    ].copy()
    jul_dec["day"] = jul_dec["t"].dt.floor("D")

    card_days = jul_dec.drop_duplicates(subset=["card_id", "day"], keep="last")
    actual_n = min(n_sample, len(card_days))
    return card_days.sample(n=actual_n, random_state=seed).sort_values("t").reset_index(drop=True)


def _build_case_anchor_uids(cc: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve each closed case's anchor transaction and holder (`uid`) as-of opened_at.

    Also records `holder_prior_48h`, `holder_prior_168h`, `holder_eligible_48h`, and
    `holder_eligible_168h` using the exact same `evaluate_single_case_uid` logic so
    that `n_fraud_elig` and `n_cleared_elig` are always >= `caught_fraud` and `cleared_fired`.
    """
    cc_df = cc.copy()
    cc_df["opened_dt"] = pd.to_datetime(cc_df["opened_at"])

    tx_by_id = feat_df.set_index("TransactionID")[["uid", "t", "card_id", "uid_seq"]]
    uid_map = tx_by_id["uid"].to_dict()
    t_map = tx_by_id["t"].to_dict()

    by_card = feat_df.groupby("card_id", sort=False)

    anchor_uids: list[str] = []
    anchor_tx_ids: list[int] = []
    prior_48_list: list[int] = []
    prior_168_list: list[int] = []
    elig_48_list: list[bool] = []
    elig_168_list: list[bool] = []

    for row in cc_df.itertuples():
        opened = row.opened_dt
        uid_val = ""
        anchor_tid = -1

        t_list = row.txn_list if isinstance(row.txn_list, list) else []
        valid_tids = [tid for tid in t_list if tid in uid_map and t_map[tid] <= opened]
        if valid_tids:
            ff = row.first_fraud_txn_id
            if pd.notna(ff) and int(ff) in valid_tids:
                anchor_tid = int(ff)
            else:
                anchor_tid = valid_tids[0]
            uid_val = str(uid_map[anchor_tid])
        elif row.card_id in by_card.groups:
            c_rows = by_card.get_group(row.card_id)
            as_of_rows = c_rows[c_rows["t"] <= opened]
            if not as_of_rows.empty:
                last_r = as_of_rows.iloc[-1]
                anchor_tid = int(last_r["TransactionID"])
                uid_val = str(last_r["uid"])

        info48 = evaluate_single_case_uid(
            flagged_txn_id=anchor_tid,
            card_id=str(row.card_id),
            opened_at=opened,
            window_hours=48,
            m=2.0,
            shift_mode="dev_or_email",
            case_id=str(row.case_id),
        )
        info168 = evaluate_single_case_uid(
            flagged_txn_id=anchor_tid,
            card_id=str(row.card_id),
            opened_at=opened,
            window_hours=168,
            m=2.0,
            shift_mode="dev_or_email",
            case_id=str(row.case_id),
        )

        anchor_uids.append(uid_val)
        anchor_tx_ids.append(anchor_tid)
        prior_48_list.append(int(info48["prior_cnt"]))
        prior_168_list.append(int(info168["prior_cnt"]))
        elig_48_list.append(bool(info48["applicable"]))
        elig_168_list.append(bool(info168["applicable"]))

    cc_df["anchor_uid"] = anchor_uids
    cc_df["anchor_txn_id"] = anchor_tx_ids
    cc_df["holder_prior_txns"] = prior_48_list
    cc_df["holder_prior_48h"] = prior_48_list
    cc_df["holder_prior_168h"] = prior_168_list
    cc_df["holder_eligible_ge10"] = elig_48_list
    cc_df["holder_eligible_48h"] = elig_48_list
    cc_df["holder_eligible_168h"] = elig_168_list
    return cc_df


@lru_cache(maxsize=1)
def get_enriched_closed_cases() -> pd.DataFrame:
    """Cached closed cases enriched with holder (`uid`) and eligibility info."""
    from assay.r2.frames import closed_cases

    return _build_case_anchor_uids(closed_cases(), get_precomputed_uid_tx_all())


def _match_cases_uid_level(
    cc_enriched: pd.DataFrame,
    window_hours: int,
    m: float,
    shift_mode: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Evaluate which closed cases fire at the holder (`uid`) level as-of opened_at
    by calling the exact same `evaluate_single_case_uid` function used by `run()`.
    """
    ato_patterns = {"account_takeover", "card_not_present_new_device"}

    caught_all: list[str] = []
    missed_all: list[str] = []
    caught_ato: list[str] = []
    legit_acted: list[str] = []

    for row in cc_enriched.itertuples():
        info = evaluate_single_case_uid(
            flagged_txn_id=int(row.anchor_txn_id),
            card_id=str(row.card_id),
            opened_at=row.opened_dt,
            window_hours=window_hours,
            m=m,
            shift_mode=shift_mode,
            case_id=str(row.case_id),
        )
        fired = bool(info["fired"])

        if row.outcome == "confirmed_fraud":
            if fired:
                caught_all.append(row.case_id)
                if row.pattern in ato_patterns:
                    caught_ato.append(row.case_id)
            else:
                missed_all.append(row.case_id)
        elif row.outcome == "cleared":
            if fired:
                legit_acted.append(row.case_id)

    return caught_all, missed_all, caught_ato, legit_acted


def _count_legit_uid_day_fires(
    legit_sample: pd.DataFrame,
    window_hours: int,
    m: float,
    shift_mode: str,
) -> int:
    """Count how many sampled legitimate card-days fire using `evaluate_single_case_uid`."""
    if legit_sample.empty:
        return 0
    fired_count = 0
    for row in legit_sample.itertuples():
        info = evaluate_single_case_uid(
            flagged_txn_id=int(row.TransactionID),
            card_id=str(row.card_id),
            opened_at=row.t,
            window_hours=window_hours,
            m=m,
            shift_mode=shift_mode,
        )
        if info["fired"]:
            fired_count += 1
    return fired_count


def evaluate_holder_shift_grid(
    cc: pd.DataFrame,
    tx: pd.DataFrame,
    windows_hours: list[int] = WINDOWS_HOURS,
    m_grid: list[float] = M_SWEEP,
    shift_modes: list[str] = SHIFT_MODES,
    n_legit_days: int = 3500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate HolderShift across:
    - window_hours in {48 (48h), 168 (7d)}
    - m in {2.0, 3.0}
    - shift_mode in {"dev_or_email", "dev_only", "email_only", "dev_and_email"}

    Primary comparison: confirmed_fraud vs CLEARED closed cases (both denominator = all
    closed cases AND denominator = eligible closed cases where uid has >= 10 prior txns).
    Also reports legitimate card-day fire rates for information.
    Every rate and CI is guarded against n == 0 or k > n.
    """
    feat_df = get_precomputed_uid_for_tx(tx)
    if len(cc) == 5585 and len(feat_df) == len(get_precomputed_uid_tx_all()):
        cc_enriched = get_enriched_closed_cases()
    else:
        cc_enriched = _build_case_anchor_uids(cc, feat_df)

    legit_days = sample_legit_card_days(cc, feat_df, n_sample=n_legit_days, seed=seed)

    n_fraud = int((cc_enriched["outcome"] == "confirmed_fraud").sum())
    n_cleared = int((cc_enriched["outcome"] == "cleared").sum())
    n_ato = int(
        (
            (cc_enriched["outcome"] == "confirmed_fraud")
            & (cc_enriched["pattern"].isin(["account_takeover", "card_not_present_new_device"]))
        ).sum()
    )
    n_legit = len(legit_days)

    results: list[dict[str, Any]] = []

    for w_h in windows_hours:
        elig_col = f"holder_eligible_{w_h}h"
        n_fraud_elig = int(
            ((cc_enriched["outcome"] == "confirmed_fraud") & cc_enriched[elig_col]).sum()
        )
        n_cleared_elig = int(
            ((cc_enriched["outcome"] == "cleared") & cc_enriched[elig_col]).sum()
        )
        n_legit_elig = int((legit_days[f"prior_cnt_{w_h}h"] >= MIN_PRIOR_TXNS).sum())

        for m in m_grid:
            for mode in shift_modes:
                caught_all, missed_all, caught_ato, legit_acted = _match_cases_uid_level(
                    cc_enriched, window_hours=w_h, m=m, shift_mode=mode
                )
                legit_day_fires = _count_legit_uid_day_fires(
                    legit_days, window_hours=w_h, m=m, shift_mode=mode
                )

                # Primary LR: P(fire|fraud) / P(fire|cleared) over ALL closed cases
                lr_cleared, ci_low_cleared, ci_high_cleared, insuff_all = bootstrap_lr_ci(
                    n_pos=n_fraud,
                    k_pos=len(caught_all),
                    n_neg=n_cleared,
                    k_neg=len(legit_acted),
                    seed=seed,
                )

                # Conditional LR among eligible cases (holder >= 10 prior txns)
                lr_elig, ci_low_elig, ci_high_elig, insuff_elig = bootstrap_lr_ci(
                    n_pos=n_fraud_elig,
                    k_pos=len(caught_all),
                    n_neg=n_cleared_elig,
                    k_neg=len(legit_acted),
                    seed=seed,
                )

                # Informational LR vs legitimate card-days
                lr_legit, ci_low_legit, ci_high_legit, insuff_legit = bootstrap_lr_ci(
                    n_pos=n_fraud,
                    k_pos=len(caught_all),
                    n_neg=n_legit,
                    k_neg=legit_day_fires,
                    seed=seed,
                )

                p_fraud = safe_rate(len(caught_all), n_fraud)
                p_ato = safe_rate(len(caught_ato), n_ato)
                p_cleared = safe_rate(len(legit_acted), n_cleared)
                p_fraud_elig = safe_rate(len(caught_all), n_fraud_elig)
                p_cleared_elig = safe_rate(len(legit_acted), n_cleared_elig)
                p_legit_day = safe_rate(legit_day_fires, n_legit)

                passes_gate = (
                    (not insuff_all)
                    and (ci_low_cleared is not None and ci_low_cleared > KEEP_GATE_CI_LOW)
                    and (p_cleared is not None and p_cleared <= KEEP_GATE_MAX_CLEARED_RATE)
                )

                bt = Backtest(
                    detector="holder_shift",
                    n_cases=len(cc_enriched),
                    caught=caught_all,
                    missed=missed_all,
                    legit_acted_against=legit_acted,
                    lr_measured=float(lr_cleared) if lr_cleared is not None else 1.0,
                )

                results.append(
                    {
                        "detector": "holder_shift",
                        "window_hours": w_h,
                        "window_label": "48h" if w_h == 48 else f"{w_h // 24}d",
                        "m": m,
                        "shift_mode": mode,
                        "n_fraud": n_fraud,
                        "n_fraud_elig_ge10": n_fraud_elig,
                        "caught_fraud": len(caught_all),
                        "p_fire_fraud": round(p_fraud, 5) if p_fraud is not None else None,
                        "p_fire_fraud_elig": round(p_fraud_elig, 5) if p_fraud_elig is not None else None,
                        "n_ato_cnp": n_ato,
                        "caught_ato_cnp": len(caught_ato),
                        "p_fire_ato_cnp": round(p_ato, 5) if p_ato is not None else None,
                        "n_cleared": n_cleared,
                        "n_cleared_elig_ge10": n_cleared_elig,
                        "cleared_fired": len(legit_acted),
                        "p_fire_cleared": round(p_cleared, 5) if p_cleared is not None else None,
                        "p_fire_cleared_elig": round(p_cleared_elig, 5) if p_cleared_elig is not None else None,
                        "cleared_case_ids": legit_acted,
                        "n_legit_days": n_legit,
                        "n_legit_days_elig_ge10": n_legit_elig,
                        "legit_days_fired": legit_day_fires,
                        "p_fire_legit_day": round(p_legit_day, 5) if p_legit_day is not None else None,
                        "lr_vs_cleared": lr_cleared,
                        "ci_low_vs_cleared": ci_low_cleared,
                        "ci_high_vs_cleared": ci_high_cleared,
                        "insufficient_data_all": insuff_all,
                        "lr_vs_cleared_elig": lr_elig,
                        "ci_low_vs_cleared_elig": ci_low_elig,
                        "ci_high_vs_cleared_elig": ci_high_elig,
                        "insufficient_data_elig": insuff_elig,
                        "lr_vs_legit_days": lr_legit,
                        "ci_low_vs_legit_days": ci_low_legit,
                        "ci_high_vs_legit_days": ci_high_legit,
                        "insufficient_data_legit": insuff_legit,
                        "weight_ci_low": ci_low_cleared if ci_low_cleared is not None else 1.0,
                        "passes_keep_gate": passes_gate,
                        "backtest": bt,
                    }
                )

    return results


@lru_cache(maxsize=1)
def _calibrated_holder_shift() -> dict[str, Any]:
    """Compute and cache the full backtest grid and best configuration for HolderShift."""
    from assay.r2.frames import closed_cases, tx_all

    cc = closed_cases()
    tx = tx_all()
    grid = evaluate_holder_shift_grid(cc, tx)

    # Sort primarily by whether it passes the keep gate, then by ci_low_vs_cleared, then caught_fraud
    best = max(
        grid,
        key=lambda r: (
            bool(r["passes_keep_gate"]),
            float(r["ci_low_vs_cleared"] or 0.0),
            int(r["caught_fraud"]),
        ),
    )
    return {
        "best": best,
        "grid": grid,
    }


def inspect_case_holder_shift(
    ctx: CaseCtx,
    window_hours: int = 48,
    m: float = 2.0,
    shift_mode: str = "dev_or_email",
) -> dict[str, Any]:
    """Detailed diagnostic inspection of a single case's holder (`uid`) as-of `ctx.opened_at`.

    Calls the single canonical `evaluate_single_case_uid` function so `inspect()`,
    `run()`, and `backtest()` share the exact same code path.
    """
    return evaluate_single_case_uid(
        flagged_txn_id=int(ctx.flagged_txn_id),
        card_id=str(ctx.card_id),
        opened_at=pd.Timestamp(ctx.opened_at),
        window_hours=window_hours,
        m=m,
        shift_mode=shift_mode,
        tx_override=ctx.tx,
        case_id=str(ctx.case_id),
    )


class HolderShift(Detector):
    """Holder-level session shift detector ("holder_shift", pattern "account_takeover").

    Checks whether the flagged transaction's holder (`uid`) shifted in the last
    `window_hours` (48h or 168h/7d) to a `device_profile` and/or `P_emaildomain`
    never used before by that `uid`, on online charges with amount >= `m` x the
    holder's prior median, requiring >= 10 prior transactions on the `uid`.

    Keep gate:
    - `ci_low_vs_cleared > 1.5` AND `p_fire_cleared <= 0.08`.
    - If `passes_keep_gate` is False (honest negative), `run(ctx)` emits nothing (`[]`)
      unless `emit_when_ungated=True` is explicitly passed for diagnostic inspection.
    """

    name: str = "holder_shift"
    pattern: str = "account_takeover"

    def __init__(
        self,
        window_hours: int | None = None,
        m: float | None = None,
        shift_mode: str | None = None,
        lr_weight: float | None = None,
        emit_when_ungated: bool = False,
    ):
        cal = _calibrated_holder_shift()
        best = cal["best"]

        self.window_hours: int = int(window_hours if window_hours is not None else best["window_hours"])
        self.m: float = float(m if m is not None else best["m"])
        self.shift_mode: str = str(shift_mode if shift_mode is not None else best["shift_mode"])
        self.emit_when_ungated: bool = bool(emit_when_ungated)

        # Look up matching cell in the calibrated grid
        cell = best
        for r in cal["grid"]:
            if (
                r["window_hours"] == self.window_hours
                and r["m"] == self.m
                and r["shift_mode"] == self.shift_mode
            ):
                cell = r
                break

        self.lr_measured: float = float(cell["lr_vs_cleared"] or 1.0)
        self.ci_low: float = float(cell["ci_low_vs_cleared"] or 1.0)
        self.ci_high: float = float(cell["ci_high_vs_cleared"] or 1.0)
        self.p_fire_cleared: float = float(cell["p_fire_cleared"] or 0.0)
        self.passes_keep_gate: bool = bool(cell["passes_keep_gate"])

        if lr_weight is not None:
            self.lr_weight: float = float(lr_weight)
        else:
            self.lr_weight = self.ci_low if self.passes_keep_gate else 1.0

    def inspect(self, ctx: CaseCtx) -> dict[str, Any]:
        """Inspect a case's holder (`uid`) shift features and applicability as-of `ctx.opened_at`."""
        return inspect_case_holder_shift(
            ctx,
            window_hours=self.window_hours,
            m=self.m,
            shift_mode=self.shift_mode,
        )

    def run(self, ctx: CaseCtx) -> list[Finding]:
        """Run HolderShift on a single CaseCtx strictly as-of `ctx.opened_at`.

        Honest negative rule: if the configuration fails the keep gate
        (`CI_low > 1.5` and `P(fire|cleared) <= 8%`), returns `[]` unless
        `self.emit_when_ungated` is True.
        """
        if not self.passes_keep_gate and not self.emit_when_ungated:
            return []

        info = self.inspect(ctx)
        if not info["applicable"] or not info["fired"]:
            return []

        affected_ids = info["affected_txn_ids"]
        uid = info["uid"]
        w_label = "48h" if self.window_hours == 48 else f"{self.window_hours // 24}d"

        att = Attestation(
            what=(
                f"Holder {uid} (card {ctx.card_id}, {info['prior_cnt']} prior txns, "
                f"prior median ${info['prior_med']:.2f}) had {len(affected_ids)} online "
                f"charge(s) in the last {w_label} up to ${info['max_online_amt']:.2f} "
                f"({info['amt_ratio']:.1f}x median) on novel device_profile "
                f"{info['novel_devices']} and/or novel P_emaildomain {info['novel_emails']}."
            ),
            evidence=Evidence(
                txn_ids=affected_ids,
                card_ids=[ctx.card_id],
                customer_ids=[ctx.customer_id],
            ),
            query="holder_shift",
            params={
                "card_id": ctx.card_id,
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "as_of": str(ctx.opened_at),
                "window_hours": self.window_hours,
                "m": self.m,
                "shift_mode": self.shift_mode,
            },
        )

        return [
            Finding(
                detector=self.name,
                pattern=self.pattern,
                claim=att,
                lr=self.lr_weight,
                affected_txn_ids=affected_ids,
                connected_card_ids=[],
            )
        ]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        """Backtest as-of over all closed cases for this configuration."""
        grid = evaluate_holder_shift_grid(
            cc,
            tx,
            windows_hours=[self.window_hours],
            m_grid=[self.m],
            shift_modes=[self.shift_mode],
        )
        return grid[0]["backtest"]


def rerun(params: dict[str, Any]) -> list[int]:
    """Re-runnable attestation query entrypoint for `holder_shift`."""
    from assay.r2.frames import closed_cases, tx_all

    tx = tx_all()
    cc = closed_cases()
    card_id = str(params["card_id"])
    flagged_txn_id = int(params["flagged_txn_id"])
    as_of = pd.Timestamp(params["as_of"])

    cust_rows = tx.loc[tx["card_id"] == card_id, "customer_id"]
    customer_id = str(cust_rows.iat[0]) if not cust_rows.empty else ""

    ctx = CaseCtx(
        case_id=str(params.get("case_id", "RERUN")),
        flagged_txn_id=flagged_txn_id,
        card_id=card_id,
        customer_id=customer_id,
        opened_at=as_of,
        tx=tx[tx["t"] <= as_of],
        cc=cc[pd.to_datetime(cc["closed_at"]) <= as_of],
    )

    det = HolderShift(
        window_hours=int(params.get("window_hours", 48)),
        m=float(params.get("m", 2.0)),
        shift_mode=str(params.get("shift_mode", "dev_or_email")),
        emit_when_ungated=True,
    )
    findings = det.run(ctx)
    return findings[0].affected_txn_ids if findings else []
