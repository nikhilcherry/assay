"""BLOCK C4 -- Account-takeover / new-device detectors.

Exposes two Detector subclasses for pattern "account_takeover":
(a) NewDeviceBurst ("new_device_burst"):
    - Flagged / recent online charge on a DeviceInfo or non-trivial device_profile
      never seen on this card before (as-of prior to the 48h burst window).
    - Part of >= 2 such online charges within 48h on that novel device/profile.
    - Max (or individual) burst charge amount >= m x the card's own prior median
      (m chosen by backtest from {2, 3, 4}).
    - Evaluates optional conditions: id_15 == 'New', novel P_emaildomain or
      'anonymous.com' (measuring each condition's contribution separately).
(b) DeviceChurn ("device_churn"):
    - >= 3 charges within 72h on >= 3 distinct DeviceInfo values and/or >= 3
      distinct email domains (P_emaildomain or R_emaildomain), none of the
      DeviceInfo values being the card's established primary device (>= 2 prior uses).
    - Because card_id is an issuer bucket, normalises by the card's own historical
      device diversity (prior distinct DeviceInfo per 72h active window or overall
      prior device-to-txn ratio) so busy multi-holder buckets do not fire constantly.

All features are computed strictly as-of t <= opened_at using vectorised pandas
operations. TransactionID is never used as a feature or threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding

M_SWEEP: list[float] = [2.0, 3.0, 4.0]
KEEP_GATE_CI_LOW: float = 1.5
EMPTY_PROFILE: str = "? | ? | ? | ?"


def _clean_str_series(s: pd.Series) -> pd.Series:
    """Normalise string column to clean string with '' for missing/placeholder."""
    out = s.fillna("").astype(str).str.strip()
    return out.where(~out.isin(["", "nan", "None", "null", "?"]), "")


def _clean_profile_series(s: pd.Series) -> pd.Series:
    out = s.fillna("").astype(str).str.strip()
    return out.where(~out.isin(["", "nan", "None", EMPTY_PROFILE]), "")


def bootstrap_lr_ci(
    n_pos: int,
    k_pos: int,
    n_neg: int,
    k_neg: int,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Laplace-smoothed LR = P(fire|pos)/P(fire|neg) and 95% analytical/bootstrap CI."""
    p_pos = (k_pos + 1.0) / (n_pos + 2.0)
    p_neg = (k_neg + 1.0) / (n_neg + 2.0)
    lr = p_pos / p_neg

    rng = np.random.default_rng(seed)
    pos_draws = rng.binomial(n_pos, k_pos / max(n_pos, 1), size=n_boot)
    neg_draws = rng.binomial(n_neg, k_neg / max(n_neg, 1), size=n_boot)
    boot_p_pos = (pos_draws + 1.0) / (n_pos + 2.0)
    boot_p_neg = (neg_draws + 1.0) / (n_neg + 2.0)
    boot_lrs = boot_p_pos / boot_p_neg

    ci_low = float(np.percentile(boot_lrs, 2.5))
    ci_high = float(np.percentile(boot_lrs, 97.5))
    return round(float(lr), 4), round(ci_low, 4), round(ci_high, 4)


def _precompute_tx_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-transaction as-of features on a sorted transaction frame.

    Every feature at row i uses ONLY transactions on the same card_id with
    timestamp strictly before row i (or within the trailing window [t_i - W, t_i]).
    Never uses future rows or TransactionID values as signals.
    """
    df = tx[
        [
            "TransactionID",
            "t",
            "card_id",
            "customer_id",
            "amount",
            "channel",
            "DeviceInfo",
            "device_profile",
            "id_15",
            "P_emaildomain",
            "R_emaildomain",
        ]
    ].copy()
    df = df.sort_values("t", kind="stable").reset_index(drop=True)

    df["dev_clean"] = _clean_str_series(df["DeviceInfo"])
    df["prof_clean"] = _clean_profile_series(df["device_profile"])
    # Primary device key: DeviceInfo if present, else non-empty device_profile
    df["dev_key"] = np.where(df["dev_clean"] != "", df["dev_clean"], df["prof_clean"])
    df["p_email"] = _clean_str_series(df["P_emaildomain"]).str.lower()
    df["r_email"] = _clean_str_series(df["R_emaildomain"]).str.lower()
    df["id15_new"] = _clean_str_series(df["id_15"]).str.lower() == "new"
    df["is_online"] = _clean_str_series(df["channel"]).str.lower() == "online"

    # Prior transaction count on card
    df["card_seq"] = df.groupby("card_id", sort=False).cumcount()

    # Occurrence rank of (card_id, dev_key) and (card_id, p_email)
    df["dev_occ"] = df.groupby(["card_id", "dev_key"], sort=False).cumcount()
    df["p_email_occ"] = df.groupby(["card_id", "p_email"], sort=False).cumcount()

    # First timestamp this (card_id, dev_key) was ever seen on this card
    first_dev_t = df.groupby(["card_id", "dev_key"], sort=False)["t"].transform("min")
    df["first_dev_t"] = first_dev_t
    # A device key is novel within a 48h burst if it was first seen on the card
    # within 48h of the current transaction, AND the card had prior history before first_dev_t.
    first_card_t = df.groupby("card_id", sort=False)["t"].transform("min")
    df["card_had_prior_before_dev"] = first_dev_t > first_card_t
    df["dev_novel_48h"] = (
        (df["dev_key"] != "")
        & df["card_had_prior_before_dev"]
        & ((df["t"] - df["first_dev_t"]) <= pd.Timedelta(hours=48))
    )

    # Similarly for P_emaildomain novelty as of first_dev_t (or within 48h)
    first_email_t = df.groupby(["card_id", "p_email"], sort=False)["t"].transform("min")
    df["email_novel_48h"] = (
        (df["p_email"] != "")
        & (first_email_t > first_card_t)
        & ((df["t"] - first_email_t) <= pd.Timedelta(hours=48))
    )
    df["email_anon_or_novel"] = (df["p_email"] == "anonymous.com") | df["email_novel_48h"]

    # Now compute per-card trailing window statistics using numpy arrays per card group
    n = len(df)
    prior_med = np.full(n, np.nan, dtype=np.float64)
    prior_cnt = np.zeros(n, dtype=np.int32)
    burst_online_novel_cnt = np.zeros(n, dtype=np.int32)
    burst_online_novel_max_amt = np.zeros(n, dtype=np.float64)
    burst_id15_new_any = np.zeros(n, dtype=bool)
    burst_email_novel_any = np.zeros(n, dtype=bool)
    burst_anon_any = np.zeros(n, dtype=bool)

    # Device churn 72h features
    win72_cnt = np.zeros(n, dtype=np.int32)
    win72_ndev = np.zeros(n, dtype=np.int32)
    win72_nemail = np.zeros(n, dtype=np.int32)
    win72_all_unestablished_dev = np.zeros(n, dtype=bool)
    prior_distinct_devs = np.zeros(n, dtype=np.int32)
    prior_dev_tx_cnt = np.zeros(n, dtype=np.int32)

    t_ns = df["t"].values.astype("datetime64[ns]").astype(np.int64)
    amt_arr = df["amount"].values.astype(np.float64)
    online_arr = df["is_online"].values
    dev_novel_arr = df["dev_novel_48h"].values
    id15_arr = df["id15_new"].values
    email_nov_arr = df["email_novel_48h"].values
    p_email_arr = df["p_email"].values
    r_email_arr = df["r_email"].values
    dev_key_arr = df["dev_key"].values
    dev_clean_arr = df["dev_clean"].values

    ns_48h = int(48 * 3600 * 1_000_000_000)
    ns_72h = int(72 * 3600 * 1_000_000_000)

    # Factorize strings to integers for fast per-card array operations
    dev_key_codes, _ = pd.factorize(df["dev_key"], sort=False)
    dev_clean_codes, _ = pd.factorize(df["dev_clean"], sort=False)
    p_email_codes, _ = pd.factorize(df["p_email"], sort=False)
    r_email_codes, _ = pd.factorize(df["r_email"], sort=False)
    anon_mask = p_email_arr == "anonymous.com"

    # Empty string codes
    empty_dev_key_code = dev_key_codes[df["dev_key"] == ""][0] if (df["dev_key"] == "").any() else -999
    empty_dev_clean_code = dev_clean_codes[df["dev_clean"] == ""][0] if (df["dev_clean"] == "").any() else -999
    empty_p_code = p_email_codes[df["p_email"] == ""][0] if (df["p_email"] == "").any() else -999
    empty_r_code = r_email_codes[df["r_email"] == ""][0] if (df["r_email"] == "").any() else -999

    # Combined email domain namespace for distinct email count in 72h
    all_emails = pd.concat([df["p_email"], df["r_email"]], ignore_index=True)
    email_codes_all, _ = pd.factorize(all_emails, sort=False)
    p_ec = email_codes_all[:n]
    r_ec = email_codes_all[n:]
    empty_ec = p_ec[df["p_email"] == ""][0] if (df["p_email"] == "").any() else -999

    # Group indices by card_id
    card_indices = df.groupby("card_id", sort=False).indices

    for _, idxs in card_indices.items():
        m_len = len(idxs)
        if m_len <= 1:
            continue

        c_t = t_ns[idxs]
        c_amt = amt_arr[idxs]
        c_online = online_arr[idxs]
        c_dev_nov = dev_novel_arr[idxs]
        c_id15 = id15_arr[idxs]
        c_em_nov = email_nov_arr[idxs]
        c_anon = anon_mask[idxs]
        c_dk = dev_key_codes[idxs]
        c_dc = dev_clean_codes[idxs]
        c_pec = p_ec[idxs]
        c_rec = r_ec[idxs]

        # Left boundaries for 48h and 72h trailing windows [t_i - W, t_i]
        l48 = np.searchsorted(c_t, c_t - ns_48h, side="left")
        l72 = np.searchsorted(c_t, c_t - ns_72h, side="left")
        # Right boundary (inclusive of all rows with same timestamp <= t_i)
        r_bound = np.arange(m_len) + 1

        # Track running prior device counts before l72
        for i in range(1, m_len):
            # 1. NewDeviceBurst features in [l48[i], i]
            s48 = l48[i]
            # Prior history strictly before the 48h window (or before first_dev_t)
            if s48 > 0:
                prior_cnt[idxs[i]] = s48
                # Fast median of prior transactions
                prior_med[idxs[i]] = float(np.median(c_amt[:s48]))

                if c_online[i] and c_dev_nov[i] and c_dk[i] != empty_dev_key_code:
                    # Charges in [s48, i] that are online and on the SAME novel dev_key
                    w_dk = c_dk[s48 : i + 1]
                    w_on = c_online[s48 : i + 1]
                    same_dev_mask = w_on & (w_dk == c_dk[i])
                    cnt_nov = int(same_dev_mask.sum())
                    burst_online_novel_cnt[idxs[i]] = cnt_nov
                    if cnt_nov >= 2:
                        w_amt = c_amt[s48 : i + 1][same_dev_mask]
                        burst_online_novel_max_amt[idxs[i]] = float(w_amt.max())
                        burst_id15_new_any[idxs[i]] = bool(c_id15[s48 : i + 1][same_dev_mask].any())
                        burst_email_novel_any[idxs[i]] = bool(c_em_nov[s48 : i + 1][same_dev_mask].any())
                        burst_anon_any[idxs[i]] = bool(c_anon[s48 : i + 1][same_dev_mask].any())

            # 2. DeviceChurn features in [l72[i], i]
            s72 = l72[i]
            w_len = i - s72 + 1
            win72_cnt[idxs[i]] = w_len
            if w_len >= 3 and s72 > 0:
                # Prior history strictly before the 72h window [0, s72)
                p_dc = c_dc[:s72]
                p_dc_valid = p_dc[p_dc != empty_dev_clean_code]
                prior_dev_tx_cnt[idxs[i]] = len(p_dc_valid)
                if len(p_dc_valid) > 0:
                    u_p_devs, u_p_cnts = np.unique(p_dc_valid, return_counts=True)
                    prior_distinct_devs[idxs[i]] = len(u_p_devs)
                    established_devs = set(u_p_devs[u_p_cnts >= 2].tolist())
                else:
                    established_devs = set()

                w_dc = c_dc[s72 : i + 1]
                w_dc_valid = w_dc[w_dc != empty_dev_clean_code]
                u_w_devs = np.unique(w_dc_valid)
                win72_ndev[idxs[i]] = len(u_w_devs)

                # None of the window's devices is an established device on the card
                if len(u_w_devs) > 0:
                    win72_all_unestablished_dev[idxs[i]] = not any(d in established_devs for d in u_w_devs)
                else:
                    win72_all_unestablished_dev[idxs[i]] = True

                # Distinct email domains (P or R) in the 72h window
                w_pec = c_pec[s72 : i + 1]
                w_rec = c_rec[s72 : i + 1]
                w_emails = np.concatenate([w_pec[w_pec != empty_ec], w_rec[w_rec != empty_ec]])
                win72_nemail[idxs[i]] = len(np.unique(w_emails))

    df["prior_cnt"] = prior_cnt
    df["prior_med"] = np.where(np.isnan(prior_med) | (prior_med <= 0), np.nan, prior_med)
    df["burst_online_novel_cnt"] = burst_online_novel_cnt
    df["burst_online_novel_max_amt"] = burst_online_novel_max_amt
    df["burst_amt_ratio"] = df["burst_online_novel_max_amt"] / df["prior_med"]
    df["burst_id15_new_any"] = burst_id15_new_any
    df["burst_email_novel_any"] = burst_email_novel_any
    df["burst_anon_any"] = burst_anon_any
    df["burst_anon_or_novel_email"] = burst_anon_any | burst_email_novel_any

    df["win72_cnt"] = win72_cnt
    df["win72_ndev"] = win72_ndev
    df["win72_nemail"] = win72_nemail
    df["win72_all_unestablished_dev"] = win72_all_unestablished_dev
    df["prior_distinct_devs"] = prior_distinct_devs
    df["prior_dev_tx_cnt"] = prior_dev_tx_cnt
    # Historical device diversity ratio: prior distinct devices / max(prior device-bearing txns, 1)
    # A single-device or low-diversity card has low prior_distinct_devs (e.g. <= 2) or low churn rate.
    df["prior_dev_diversity"] = np.where(
        df["prior_dev_tx_cnt"] > 0,
        df["prior_distinct_devs"] / df["prior_dev_tx_cnt"],
        0.0,
    )

    return df


@lru_cache(maxsize=1)
def get_precomputed_tx_all() -> pd.DataFrame:
    """Cached precomputation over frames.tx_all()."""
    from assay.r2.frames import tx_all
    return _precompute_tx_features(tx_all())


def get_precomputed_for_tx(tx: pd.DataFrame) -> pd.DataFrame:
    """Return precomputed features for tx (uses cache if tx matches tx_all())."""
    from assay.r2.frames import tx_all
    full = tx_all()
    if len(tx) == len(full) and tx["TransactionID"].iat[0] == full["TransactionID"].iat[0]:
        return get_precomputed_tx_all()
    return _precompute_tx_features(tx)


def sample_legit_card_days(
    cc: pd.DataFrame,
    feat_df: pd.DataFrame,
    n_sample: int = 3500,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample >= 3,000 legitimate card-days from Jul-Dec (cards never named in any closed case)."""
    named_cards = set(cc["card_id"].dropna().unique())
    for c_list in cc.get("connected_list", []):
        if isinstance(c_list, list):
            named_cards.update(c_list)

    jul_dec = feat_df[
        (feat_df["t"] < pd.Timestamp("2017-01-01"))
        & (~feat_df["card_id"].isin(named_cards))
    ].copy()
    jul_dec["day"] = jul_dec["t"].dt.floor("D")

    # Take last transaction of each (card_id, day) as the as-of anchor for that card-day
    card_days = jul_dec.drop_duplicates(subset=["card_id", "day"], keep="last")
    actual_n = min(n_sample, len(card_days))
    return card_days.sample(n=actual_n, random_state=seed).sort_values("t").reset_index(drop=True)


def _match_cases_against_fires(
    cc: pd.DataFrame,
    fired_sub: pd.DataFrame,
    lookback_days: int = 7,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Evaluate which closed cases fire as-of their opened_at timestamp.

    Returns (caught_all_fraud, missed_all_fraud, caught_ato_cnp_new, legit_acted_against).
    """
    cc_df = cc.copy()
    cc_df["opened_dt"] = pd.to_datetime(cc_df["opened_at"])

    by_card = fired_sub.groupby("card_id", sort=False) if not fired_sub.empty else None
    fired_tx_set = set(fired_sub["TransactionID"].tolist()) if not fired_sub.empty else set()

    caught_all: list[str] = []
    missed_all: list[str] = []
    caught_ato: list[str] = []
    legit_acted: list[str] = []

    ato_patterns = {"account_takeover", "card_not_present_new_device"}

    for row in cc_df.itertuples():
        card = row.card_id
        opened = row.opened_dt
        fired = False

        if by_card is not None and card in by_card.groups:
            c_rows = by_card.get_group(card)
            # As-of check: firing transaction must have t <= opened_at and within lookback window
            # OR be one of the case's own listed transactions with t <= opened_at
            in_win = c_rows[
                (c_rows["t"] <= opened)
                & (c_rows["t"] >= opened - pd.Timedelta(days=lookback_days))
            ]
            if not in_win.empty:
                fired = True
            elif isinstance(row.txn_list, list) and row.txn_list:
                in_tx = c_rows[(c_rows["t"] <= opened) & (c_rows["TransactionID"].isin(row.txn_list))]
                if not in_tx.empty:
                    fired = True

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


def _count_legit_card_day_fires(
    legit_sample: pd.DataFrame,
    fired_sub: pd.DataFrame,
    lookback_days: int = 7,
) -> int:
    """Count how many sampled legitimate card-days have a fire within lookback_days as-of end of that day."""
    if fired_sub.empty or legit_sample.empty:
        return 0
    by_card = fired_sub.groupby("card_id", sort=False)
    fired_count = 0
    for row in legit_sample.itertuples():
        if row.card_id in by_card.groups:
            c_rows = by_card.get_group(row.card_id)
            as_of = row.t
            in_win = c_rows[
                (c_rows["t"] <= as_of)
                & (c_rows["t"] >= as_of - pd.Timedelta(days=lookback_days))
            ]
            if not in_win.empty:
                fired_count += 1
    return fired_count


def evaluate_new_device_burst_grid(
    cc: pd.DataFrame,
    tx: pd.DataFrame,
    m_grid: list[float] = M_SWEEP,
    n_legit_days: int = 3500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sweep m in {2, 3, 4} and measure each optional condition's contribution separately:
    - base: online + novel device/profile + >=2 charges in 48h + amount >= m * prior_median
    - + id_15 == 'New'
    - + novel P_emaildomain or anonymous.com
    - + anonymous.com only
    - + id_15 == 'New' AND (novel P_emaildomain or anonymous.com)
    """
    feat_df = get_precomputed_for_tx(tx)
    legit_days = sample_legit_card_days(cc, feat_df, n_sample=n_legit_days, seed=seed)

    n_fraud = int((cc["outcome"] == "confirmed_fraud").sum())
    n_cleared = int((cc["outcome"] == "cleared").sum())
    n_ato = int(
        (
            (cc["outcome"] == "confirmed_fraud")
            & (cc["pattern"].isin(["account_takeover", "card_not_present_new_device"]))
        ).sum()
    )
    n_legit = len(legit_days)

    base_mask = (
        feat_df["is_online"]
        & feat_df["dev_novel_48h"]
        & (feat_df["prior_cnt"] >= 3)
        & (feat_df["burst_online_novel_cnt"] >= 2)
    )

    variants = [
        ("base", pd.Series(True, index=feat_df.index)),
        ("+id_15=='New'", feat_df["burst_id15_new_any"]),
        ("+novel_or_anon_email", feat_df["burst_anon_or_novel_email"]),
        ("+anon_email_only", feat_df["burst_anon_any"]),
        ("+id_15=='New'&novel_or_anon_email", feat_df["burst_id15_new_any"] & feat_df["burst_anon_or_novel_email"]),
    ]

    results: list[dict[str, Any]] = []
    for m in m_grid:
        m_mask = base_mask & (feat_df["burst_amt_ratio"] >= m)
        for var_name, cond_mask in variants:
            full_mask = m_mask & cond_mask
            fired_sub = feat_df[full_mask]

            caught_all, missed_all, caught_ato, legit_acted = _match_cases_against_fires(
                cc, fired_sub, lookback_days=7
            )
            legit_day_fires = _count_legit_card_day_fires(legit_days, fired_sub, lookback_days=7)

            lr_legit, ci_low_legit, ci_high_legit = bootstrap_lr_ci(
                n_pos=n_fraud,
                k_pos=len(caught_all),
                n_neg=n_legit,
                k_neg=legit_day_fires,
                seed=seed,
            )
            lr_cleared, ci_low_cleared, ci_high_cleared = bootstrap_lr_ci(
                n_pos=n_fraud,
                k_pos=len(caught_all),
                n_neg=n_cleared,
                k_neg=len(legit_acted),
                seed=seed,
            )

            p_fraud = len(caught_all) / max(n_fraud, 1)
            p_ato = len(caught_ato) / max(n_ato, 1)
            p_cleared = len(legit_acted) / max(n_cleared, 1)
            p_legit_day = legit_day_fires / max(n_legit, 1)

            bt = Backtest(
                detector="new_device_burst",
                n_cases=len(cc),
                caught=caught_all,
                missed=missed_all,
                legit_acted_against=legit_acted,
                lr_measured=lr_legit,
            )

            results.append(
                {
                    "detector": "new_device_burst",
                    "m": m,
                    "variant": var_name,
                    "n_fraud": n_fraud,
                    "caught_fraud": len(caught_all),
                    "p_fire_fraud": round(p_fraud, 5),
                    "n_ato_cnp": n_ato,
                    "caught_ato_cnp": len(caught_ato),
                    "p_fire_ato_cnp": round(p_ato, 5),
                    "n_cleared": n_cleared,
                    "cleared_fired": len(legit_acted),
                    "p_fire_cleared": round(p_cleared, 5),
                    "cleared_case_ids": legit_acted,
                    "n_legit_days": n_legit,
                    "legit_days_fired": legit_day_fires,
                    "p_fire_legit_day": round(p_legit_day, 5),
                    "lr_vs_legit_days": lr_legit,
                    "ci_low_vs_legit_days": ci_low_legit,
                    "ci_high_vs_legit_days": ci_high_legit,
                    "lr_vs_cleared": lr_cleared,
                    "ci_low_vs_cleared": ci_low_cleared,
                    "ci_high_vs_cleared": ci_high_cleared,
                    "weight_ci_low": ci_low_legit,
                    "passes_keep_gate": ci_low_legit > KEEP_GATE_CI_LOW,
                    "backtest": bt,
                }
            )
    return results


def evaluate_device_churn_grid(
    cc: pd.DataFrame,
    tx: pd.DataFrame,
    n_legit_days: int = 3500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate device_churn across diversity normalisation variants:
    - unnormalised: >=3 txns in 72h, (>=3 distinct DeviceInfo OR >=3 distinct email domains),
      none of them the card's established device.
    - norm_low_prior_devs: adds historical device diversity filter (prior_distinct_devs <= 2)
    - norm_strict_both: >=3 distinct DeviceInfo AND >=3 distinct email domains,
      unestablished devices, prior_distinct_devs <= 2
    - norm_device_only: >=3 distinct DeviceInfo in 72h, unestablished devices,
      prior_distinct_devs <= 1
    """
    feat_df = get_precomputed_for_tx(tx)
    legit_days = sample_legit_card_days(cc, feat_df, n_sample=n_legit_days, seed=seed)

    n_fraud = int((cc["outcome"] == "confirmed_fraud").sum())
    n_cleared = int((cc["outcome"] == "cleared").sum())
    n_ato = int(
        (
            (cc["outcome"] == "confirmed_fraud")
            & (cc["pattern"].isin(["account_takeover", "card_not_present_new_device"]))
        ).sum()
    )
    n_legit = len(legit_days)

    base_72 = (
        (feat_df["win72_cnt"] >= 3)
        & (feat_df["prior_cnt"] >= 3)
        & feat_df["win72_all_unestablished_dev"]
    )

    variants = [
        (
            "unnormalised_dev_or_email>=3",
            base_72 & ((feat_df["win72_ndev"] >= 3) | (feat_df["win72_nemail"] >= 3)),
        ),
        (
            "norm_prior_devs<=2_dev_or_email>=3",
            base_72
            & ((feat_df["win72_ndev"] >= 3) | (feat_df["win72_nemail"] >= 3))
            & (feat_df["prior_distinct_devs"] <= 2),
        ),
        (
            "norm_prior_devs<=1_dev>=3",
            base_72
            & (feat_df["win72_ndev"] >= 3)
            & (feat_df["prior_distinct_devs"] <= 1),
        ),
        (
            "norm_prior_devs<=2_dev>=3_and_email>=3",
            base_72
            & (feat_df["win72_ndev"] >= 3)
            & (feat_df["win72_nemail"] >= 3)
            & (feat_df["prior_distinct_devs"] <= 2),
        ),
    ]

    results: list[dict[str, Any]] = []
    for var_name, mask in variants:
        fired_sub = feat_df[mask]
        caught_all, missed_all, caught_ato, legit_acted = _match_cases_against_fires(
            cc, fired_sub, lookback_days=7
        )
        legit_day_fires = _count_legit_card_day_fires(legit_days, fired_sub, lookback_days=7)

        lr_legit, ci_low_legit, ci_high_legit = bootstrap_lr_ci(
            n_pos=n_fraud,
            k_pos=len(caught_all),
            n_neg=n_legit,
            k_neg=legit_day_fires,
            seed=seed,
        )
        lr_cleared, ci_low_cleared, ci_high_cleared = bootstrap_lr_ci(
            n_pos=n_fraud,
            k_pos=len(caught_all),
            k_neg=len(legit_acted),
            n_neg=n_cleared,
            seed=seed,
        )

        p_fraud = len(caught_all) / max(n_fraud, 1)
        p_ato = len(caught_ato) / max(n_ato, 1)
        p_cleared = len(legit_acted) / max(n_cleared, 1)
        p_legit_day = legit_day_fires / max(n_legit, 1)

        bt = Backtest(
            detector="device_churn",
            n_cases=len(cc),
            caught=caught_all,
            missed=missed_all,
            legit_acted_against=legit_acted,
            lr_measured=lr_legit,
        )

        results.append(
            {
                "detector": "device_churn",
                "variant": var_name,
                "n_fraud": n_fraud,
                "caught_fraud": len(caught_all),
                "p_fire_fraud": round(p_fraud, 5),
                "n_ato_cnp": n_ato,
                "caught_ato_cnp": len(caught_ato),
                "p_fire_ato_cnp": round(p_ato, 5),
                "n_cleared": n_cleared,
                "cleared_fired": len(legit_acted),
                "p_fire_cleared": round(p_cleared, 5),
                "cleared_case_ids": legit_acted,
                "n_legit_days": n_legit,
                "legit_days_fired": legit_day_fires,
                "p_fire_legit_day": round(p_legit_day, 5),
                "lr_vs_legit_days": lr_legit,
                "ci_low_vs_legit_days": ci_low_legit,
                "ci_high_vs_legit_days": ci_high_legit,
                "lr_vs_cleared": lr_cleared,
                "ci_low_vs_cleared": ci_low_cleared,
                "ci_high_vs_cleared": ci_high_cleared,
                "weight_ci_low": ci_low_legit,
                "passes_keep_gate": ci_low_legit > KEEP_GATE_CI_LOW,
                "backtest": bt,
            }
        )
    return results


@lru_cache(maxsize=1)
def _calibrated_defaults() -> dict[str, dict[str, Any]]:
    """Compute and cache the best backtested configuration for both detectors."""
    from assay.r2.frames import closed_cases, tx_all

    cc = closed_cases()
    tx = tx_all()

    ndb_grid = evaluate_new_device_burst_grid(cc, tx)
    # Pick best variant by ci_low_vs_legit_days among base/+novel_or_anon_email/+id_15=='New'
    best_ndb = max(ndb_grid, key=lambda r: (r["ci_low_vs_legit_days"], r["caught_fraud"]))

    dc_grid = evaluate_device_churn_grid(cc, tx)
    # Pick normalised variant with highest ci_low_vs_legit_days
    norm_dc = [r for r in dc_grid if r["variant"].startswith("norm_")]
    best_dc = max(norm_dc, key=lambda r: (r["ci_low_vs_legit_days"], r["caught_fraud"]))

    return {
        "new_device_burst": best_ndb,
        "device_churn": best_dc,
        "ndb_grid": ndb_grid,
        "dc_grid": dc_grid,
    }


class NewDeviceBurst(Detector):
    """(a) "new_device_burst", pattern "account_takeover":
    The flagged/recent txn is online, on a DeviceInfo/device_profile never seen on
    this card before (as-of), part of >= 2 such charges within 48h, with amount >=
    m x the card's own prior median, optionally with a novel/anonymous P_emaildomain
    or id_15 == 'New'.
    """

    name: str = "new_device_burst"
    pattern: str = "account_takeover"

    def __init__(
        self,
        m: float | None = None,
        require_email_novel_or_anon: bool | None = None,
        require_id15_new: bool | None = None,
        lr_weight: float | None = None,
    ):
        cal = _calibrated_defaults()["new_device_burst"]
        self.m: float = float(m if m is not None else cal["m"])
        var = cal["variant"]
        self.require_email_novel_or_anon: bool = (
            require_email_novel_or_anon
            if require_email_novel_or_anon is not None
            else ("email" in var)
        )
        self.require_id15_new: bool = (
            require_id15_new if require_id15_new is not None else ("id_15" in var)
        )
        # Weight = CI lower bound if it passes the keep gate (> 1.5), else 1.0 (honest gate)
        self.ci_low: float = float(cal["ci_low_vs_legit_days"])
        self.lr_measured: float = float(cal["lr_vs_legit_days"])
        self.passes_keep_gate: bool = bool(cal["passes_keep_gate"])
        if lr_weight is not None:
            self.lr_weight = float(lr_weight)
        else:
            self.lr_weight = self.ci_low if self.passes_keep_gate else 1.0

    def run(self, ctx: CaseCtx) -> list[Finding]:
        """Run new_device_burst on a single CaseCtx strictly as-of ctx.opened_at."""
        tx_hist = ctx.tx[ctx.tx["t"] <= ctx.opened_at]
        if tx_hist.empty:
            return []

        card_tx = tx_hist[tx_hist["card_id"] == ctx.card_id].sort_values("t", kind="stable").copy()
        if len(card_tx) < 4:
            return []

        card_tx["dev_clean"] = _clean_str_series(card_tx["DeviceInfo"])
        card_tx["prof_clean"] = _clean_profile_series(card_tx["device_profile"])
        card_tx["dev_key"] = np.where(
            card_tx["dev_clean"] != "", card_tx["dev_clean"], card_tx["prof_clean"]
        )
        card_tx["p_email"] = _clean_str_series(card_tx["P_emaildomain"]).str.lower()
        card_tx["id15_new"] = _clean_str_series(card_tx["id_15"]).str.lower() == "new"
        card_tx["is_online"] = _clean_str_series(card_tx["channel"]).str.lower() == "online"

        # Candidate anchor transactions: recent window (within 7 days of opened_at) or flagged_txn_id
        cutoff = ctx.opened_at - pd.Timedelta(days=7)
        cands = card_tx[
            (card_tx["t"] >= cutoff) | (card_tx["TransactionID"] == ctx.flagged_txn_id)
        ]
        if cands.empty:
            return []

        # Prioritise flagged_txn_id if it is in candidates, otherwise latest candidate
        cands_order = pd.concat(
            [
                cands[cands["TransactionID"] == ctx.flagged_txn_id],
                cands[cands["TransactionID"] != ctx.flagged_txn_id].sort_values("t", ascending=False),
            ]
        )

        for row in cands_order.itertuples():
            if not row.is_online or not row.dev_key:
                continue

            t_anchor = row.t
            win_start = t_anchor - pd.Timedelta(hours=48)
            prior_tx = card_tx[card_tx["t"] < win_start]
            if len(prior_tx) < 3:
                continue

            # Device must never have been seen on this card prior to win_start
            prior_devs = set(prior_tx["dev_key"].unique()) - {""}
            if row.dev_key in prior_devs:
                continue

            # All online charges on this novel device within [win_start, min(t_anchor + 48h, opened_at)]
            win_end = min(t_anchor + pd.Timedelta(hours=48), ctx.opened_at)
            burst_tx = card_tx[
                (card_tx["t"] >= win_start)
                & (card_tx["t"] <= win_end)
                & card_tx["is_online"]
                & (card_tx["dev_key"] == row.dev_key)
            ]
            if len(burst_tx) < 2:
                continue

            prior_med = float(prior_tx["amount"].median())
            if prior_med <= 0:
                continue

            max_amt = float(burst_tx["amount"].max())
            ratio = max_amt / prior_med
            if ratio < self.m:
                continue

            prior_emails = set(prior_tx["p_email"].unique()) - {""}
            burst_emails = set(burst_tx["p_email"].unique()) - {""}
            novel_emails = burst_emails - prior_emails
            has_anon = "anonymous.com" in burst_emails
            has_novel_or_anon_email = bool(novel_emails or has_anon)
            has_id15_new = bool(burst_tx["id15_new"].any())

            if self.require_email_novel_or_anon and not has_novel_or_anon_email:
                continue
            if self.require_id15_new and not has_id15_new:
                continue

            affected_ids = [int(x) for x in burst_tx["TransactionID"].tolist()]
            total_amt = float(burst_tx["amount"].sum())
            email_note = (
                f", novel/anon email ({sorted(novel_emails | ({'anonymous.com'} if has_anon else set()))})"
                if has_novel_or_anon_email
                else ""
            )
            id15_note = ", id_15='New'" if has_id15_new else ""

            att = Attestation(
                what=(
                    f"Card {ctx.card_id} had {len(burst_tx)} online charges (${total_amt:.2f} total, "
                    f"max ${max_amt:.2f} = {ratio:.1f}x prior median ${prior_med:.2f}) within 48h on "
                    f"previously unseen device '{row.dev_key}'{email_note}{id15_note}."
                ),
                evidence=Evidence(
                    txn_ids=affected_ids,
                    card_ids=[ctx.card_id],
                    customer_ids=[ctx.customer_id],
                ),
                query="new_device_burst",
                params={
                    "card_id": ctx.card_id,
                    "as_of": str(ctx.opened_at),
                    "flagged_txn_id": int(ctx.flagged_txn_id),
                    "m": self.m,
                    "require_email_novel_or_anon": self.require_email_novel_or_anon,
                    "require_id15_new": self.require_id15_new,
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

        return []

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        """Backtest as-of over closed cases."""
        grid = evaluate_new_device_burst_grid(cc, tx, m_grid=[self.m])
        for r in grid:
            if self.require_id15_new and self.require_email_novel_or_anon:
                if r["variant"] == "+id_15=='New'&novel_or_anon_email":
                    return r["backtest"]
            elif self.require_email_novel_or_anon:
                if r["variant"] == "+novel_or_anon_email":
                    return r["backtest"]
            elif self.require_id15_new:
                if r["variant"] == "+id_15=='New'":
                    return r["backtest"]
            else:
                if r["variant"] == "base":
                    return r["backtest"]
        return grid[0]["backtest"]


class DeviceChurn(Detector):
    """(b) "device_churn", pattern "account_takeover":
    >= 3 charges within 72h on >= 3 distinct DeviceInfo values and/or >= 3 distinct
    email domains (P or R), none of them the card's established device (>= 2 prior
    uses before the 72h window), normalised by the card's own historical device
    diversity (prior_distinct_devs <= max_prior_devs).
    """

    name: str = "device_churn"
    pattern: str = "account_takeover"

    def __init__(
        self,
        max_prior_devs: int = 2,
        require_both_dev_and_email: bool = False,
        lr_weight: float | None = None,
    ):
        cal = _calibrated_defaults()["device_churn"]
        self.max_prior_devs = int(max_prior_devs)
        self.require_both_dev_and_email = bool(require_both_dev_and_email)
        self.ci_low: float = float(cal["ci_low_vs_legit_days"])
        self.lr_measured: float = float(cal["lr_vs_legit_days"])
        self.passes_keep_gate: bool = bool(cal["passes_keep_gate"])
        if lr_weight is not None:
            self.lr_weight = float(lr_weight)
        else:
            self.lr_weight = self.ci_low if self.passes_keep_gate else 1.0

    def run(self, ctx: CaseCtx) -> list[Finding]:
        """Run device_churn on a single CaseCtx strictly as-of ctx.opened_at."""
        tx_hist = ctx.tx[ctx.tx["t"] <= ctx.opened_at]
        if tx_hist.empty:
            return []

        card_tx = tx_hist[tx_hist["card_id"] == ctx.card_id].sort_values("t", kind="stable").copy()
        if len(card_tx) < 6:
            return []

        card_tx["dev_clean"] = _clean_str_series(card_tx["DeviceInfo"])
        card_tx["p_email"] = _clean_str_series(card_tx["P_emaildomain"]).str.lower()
        card_tx["r_email"] = _clean_str_series(card_tx["R_emaildomain"]).str.lower()

        cutoff = ctx.opened_at - pd.Timedelta(days=7)
        cands = card_tx[
            (card_tx["t"] >= cutoff) | (card_tx["TransactionID"] == ctx.flagged_txn_id)
        ]
        if cands.empty:
            return []

        cands_order = pd.concat(
            [
                cands[cands["TransactionID"] == ctx.flagged_txn_id],
                cands[cands["TransactionID"] != ctx.flagged_txn_id].sort_values("t", ascending=False),
            ]
        )

        for row in cands_order.itertuples():
            t_anchor = row.t
            win_start = t_anchor - pd.Timedelta(hours=72)
            prior_tx = card_tx[card_tx["t"] < win_start]
            if len(prior_tx) < 3:
                continue

            prior_devs_series = prior_tx.loc[prior_tx["dev_clean"] != "", "dev_clean"]
            prior_dev_counts = prior_devs_series.value_counts()
            n_prior_distinct_devs = len(prior_dev_counts)
            if n_prior_distinct_devs > self.max_prior_devs:
                continue

            established_devs = set(prior_dev_counts[prior_dev_counts >= 2].index.tolist())

            win_tx = card_tx[(card_tx["t"] >= win_start) & (card_tx["t"] <= t_anchor)]
            if len(win_tx) < 3:
                continue

            win_devs = sorted(set(win_tx.loc[win_tx["dev_clean"] != "", "dev_clean"].unique()))
            if any(d in established_devs for d in win_devs):
                continue

            win_emails = sorted(
                (set(win_tx["p_email"].unique()) | set(win_tx["r_email"].unique())) - {""}
            )

            cond_dev = len(win_devs) >= 3
            cond_email = len(win_emails) >= 3
            if self.require_both_dev_and_email:
                if not (cond_dev and cond_email):
                    continue
            else:
                if not (cond_dev or cond_email):
                    continue

            affected_ids = [int(x) for x in win_tx["TransactionID"].tolist()]
            total_amt = float(win_tx["amount"].sum())

            att = Attestation(
                what=(
                    f"Card {ctx.card_id} had {len(win_tx)} charges (${total_amt:.2f}) within 72h "
                    f"across {len(win_devs)} distinct unestablished DeviceInfo values ({win_devs}) "
                    f"and {len(win_emails)} distinct email domains ({win_emails}), vs "
                    f"{n_prior_distinct_devs} prior distinct device(s) on this card."
                ),
                evidence=Evidence(
                    txn_ids=affected_ids,
                    card_ids=[ctx.card_id],
                    customer_ids=[ctx.customer_id],
                ),
                query="device_churn",
                params={
                    "card_id": ctx.card_id,
                    "as_of": str(ctx.opened_at),
                    "flagged_txn_id": int(ctx.flagged_txn_id),
                    "max_prior_devs": self.max_prior_devs,
                    "require_both_dev_and_email": self.require_both_dev_and_email,
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

        return []

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        """Backtest as-of over closed cases."""
        grid = evaluate_device_churn_grid(cc, tx)
        target = (
            "norm_prior_devs<=2_dev>=3_and_email>=3"
            if self.require_both_dev_and_email
            else "norm_prior_devs<=2_dev_or_email>=3"
        )
        for r in grid:
            if r["variant"] == target:
                return r["backtest"]
        return grid[0]["backtest"]


def rerun(params: dict[str, Any]) -> list[int]:
    """Re-runnable attestation query entrypoint for both new_device_burst and device_churn."""
    from assay.r2.frames import closed_cases, tx_all

    tx = tx_all()
    cc = closed_cases()
    card_id = str(params["card_id"])
    as_of = pd.Timestamp(params["as_of"])
    flagged_txn_id = int(params.get("flagged_txn_id", 0))

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

    if "m" in params:
        det = NewDeviceBurst(
            m=float(params["m"]),
            require_email_novel_or_anon=bool(params.get("require_email_novel_or_anon", False)),
            require_id15_new=bool(params.get("require_id15_new", False)),
        )
    else:
        det = DeviceChurn(
            max_prior_devs=int(params.get("max_prior_devs", 2)),
            require_both_dev_and_email=bool(params.get("require_both_dev_and_email", False)),
        )

    findings = det.run(ctx)
    return findings[0].affected_txn_ids if findings else []
