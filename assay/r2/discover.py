"""B7 -- Self-growing detective (Challenge 2).

Finds unlabelled or residual fraud patterns by:
1. Featurising transactions and closed-case episodes as-of `opened_at` into 44 interpretable
   boolean predicates (excluding generic devices/free-mail from shared-element features and
   applying a 99th-percentile cardinality cap on shared elements from Jul-Oct legitimate data).
2. Sampling a fixed-seed (seed=42) control population of 6,000 legitimate card-days from
   Jul-Oct 2016 (cards/txns never named in any closed case) so false-positive rates and
   likelihood ratios `LR = P(fire | fraud episode) / P(fire | legit card-day)` are measured
   against true legitimate behaviour as well as cleared closed cases.
3. Mining 2-3 predicate conjunctions (Apriori-style with support & legit card-day firing pruning
   <= 2%) on Jul-Oct 2016 that catch undocumented or residual confirmed-fraud cases while rarely
   firing on legitimate card-days or cleared cases.
4. Emitting deterministic Python source files for proposed `Detector` subclasses to
   `data/stage/r2/proposed/<name>.py` (never directly to the active toolkit without approval).
5. Backtesting each proposal as-of on Jul-Oct (train), Nov-Dec (validation), all 5,585 closed
   cases, and the 6,000 legit card-day sample, writing attested proposal JSONs.
6. Enforcing L2 human approval via `approve(name, approver_token)` before any proposed
   detector can be loaded into `REGISTRY`.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence, dataset_tag
from assay.r2.finding import REGISTRY, Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import R2_STAGE, closed_cases, tx_all

PROPOSED_DIR = R2_STAGE / "proposed"
APPROVED_FILE = PROPOSED_DIR / "approved_registry.json"

SPLIT_TS = pd.Timestamp("2016-11-01 00:00:00")
JAN_TS = pd.Timestamp("2017-01-01 00:00:00")
LEGIT_SAMPLE_SIZE = 6000
LEGIT_SAMPLE_SEED = 42

FREE_EMAILS = {
    "gmail.com", "yahoo.com", "hotmail.com", "aol.com", "outlook.com",
    "live.com", "icloud.com", "msn.com", "ymail.com", "mail.com", "me.com",
    "comcast.net", "verizon.net", "att.net", "sbcglobal.net", "cox.net",
}

GENERIC_DEVICES = {
    "", "? | ? | ? | ?", "Windows", "iOS Device", "MacOS", "Trident/7.0", "rv:11.0",
}

# Server-side L2 approval tokens (roles verified server-side, never trusted from payload alone)
_L2_APPROVERS: dict[str, dict[str, str]] = {
    "L2-MGR-PRIYA-9F3A": {"approver_id": "mgr_priya_sharma", "role": "L2"},
    "L2-MGR-ARJUN-4B8C": {"approver_id": "mgr_arjun_mehta", "role": "L2"},
    "L2-HEAD-KAVITA-7E1D": {"approver_id": "head_kavita_rao", "role": "L2"},
}
_L1_TOKENS: dict[str, dict[str, str]] = {
    "L1-LEAD-ROHAN-2C4D": {"approver_id": "lead_rohan_verma", "role": "L1"},
}

PREDICATE_DESCRIPTIONS: dict[str, str] = {
    "amt_micro_lt15": "Flagged transaction amount < $15",
    "amt_small_15_50": "Flagged transaction amount in [$15, $50)",
    "amt_med_50_200": "Flagged transaction amount in [$50, $200)",
    "amt_high_200_500": "Flagged transaction amount in [$200, $500)",
    "amt_vhigh_ge500": "Flagged transaction amount >= $500",
    "amt_cents_round": "Transaction amount is an exact whole dollar (.00)",
    "amt_gt_3x_card_med": "Amount > 3x card's prior mean amount",
    "exact_amt_repeat_card_7d": "Same exact amount repeated on this card within 7 days",
    "prod_W": "ProductCD == 'W'",
    "prod_C": "ProductCD == 'C'",
    "prod_R": "ProductCD == 'R'",
    "prod_H": "ProductCD == 'H'",
    "prod_S": "ProductCD == 'S'",
    "chan_online": "Transaction channel is online",
    "chan_in_person": "Transaction channel is in_person",
    "hr_night_0_5": "Transaction hour in 00:00..05:59 UTC",
    "hr_morn_6_11": "Transaction hour in 06:00..11:59 UTC",
    "hr_aft_12_17": "Transaction hour in 12:00..17:59 UTC",
    "hr_eve_18_23": "Transaction hour in 18:00..23:59 UTC",
    "gap_rapid_lt10m": "Gap since previous txn on card < 10 minutes",
    "gap_short_lt1h": "Gap since previous txn on card < 1 hour",
    "gap_dormant_gt14d": "Card woke after > 14 days of silence",
    "gap_first_on_card": "First observed transaction on this card",
    "card_age_new_le7d": "Card age D1 <= 7 days",
    "card_age_young_8_60d": "Card age D1 in (7, 60] days",
    "card_age_mature_gt60d": "Card age D1 > 60 days",
    "dev_novel_on_card": "Specific device profile never seen before on this card",
    "dev_id15_new": "Identity id_15 == 'New'",
    "dev_missing_online": "Online transaction with missing DeviceInfo",
    "dev_mobile": "DeviceType == 'mobile'",
    "email_r_present": "Recipient email domain (R_emaildomain) is present",
    "email_p_r_mismatch": "Purchaser and recipient email domains differ",
    "email_p_free": "Purchaser email domain is a free webmail provider",
    "email_p_rare": "Purchaser email domain is rare (<100 txns globally)",
    "id23_anon_proxy": "id_23 indicates anonymous proxy combined with online/device context",
    "region_changed": "Billing region (addr1) changed from card's prior transaction",
    "region_missing": "Billing region (addr1) is missing (NaN)",
    "dist_far_gt100": "Distance dist1 > 100",
    "burst_ge3_1h": ">= 3 transactions on this card within 1 hour",
    "burst_ge5_24h": ">= 5 transactions on this card within 24 hours",
    "shared_dev_7d_ge2cards": "Non-generic device profile shared across >= 2 cards within 7 days (<= p99 cap)",
    "shared_remail_7d_ge3cards": "Recipient email + region shared across >= 3 cards within 7 days (<= p99 cap)",
    "shared_addr_amt_7d_ge2cards": "Non-round (addr1, exact_amount) shared across >= 2 cards within 7 days (<= p99 cap)",
    "c1_high_ge5": "Count feature C1 >= 5",
}

PREDICATE_NAMES: list[str] = list(PREDICATE_DESCRIPTIONS.keys())


@dataclass
class DiscoveredProposal:
    name: str
    pattern: str
    predicates: list[str]
    description: str
    source_path: str
    proposal_json_path: str
    train_backtest: Backtest
    val_backtest: Backtest
    full_backtest: Backtest
    legit_sample_n: int
    legit_sample_fired: int
    legit_sample_rate: float
    train_fraud_rate: float
    val_fraud_rate: float
    lr_vs_legit_sample: float
    undocumented_caught: list[str]
    residual_caught: list[str]
    attestation: Attestation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pattern": self.pattern,
            "predicates": self.predicates,
            "description": self.description,
            "source_path": self.source_path,
            "proposal_json_path": self.proposal_json_path,
            "train_backtest": {
                "detector": self.train_backtest.detector,
                "n_cases": self.train_backtest.n_cases,
                "caught": self.train_backtest.caught,
                "missed": self.train_backtest.missed,
                "legit_acted_against": self.train_backtest.legit_acted_against,
                "lr_measured": round(self.train_backtest.lr_measured, 4),
                "precision": round(self.train_backtest.precision, 4),
                "recall": round(self.train_backtest.recall, 4),
            },
            "val_backtest": {
                "detector": self.val_backtest.detector,
                "n_cases": self.val_backtest.n_cases,
                "caught": self.val_backtest.caught,
                "missed": self.val_backtest.missed,
                "legit_acted_against": self.val_backtest.legit_acted_against,
                "lr_measured": round(self.val_backtest.lr_measured, 4),
                "precision": round(self.val_backtest.precision, 4),
                "recall": round(self.val_backtest.recall, 4),
            },
            "full_backtest": {
                "detector": self.full_backtest.detector,
                "n_cases": self.full_backtest.n_cases,
                "caught": self.full_backtest.caught,
                "missed": self.full_backtest.missed,
                "legit_acted_against": self.full_backtest.legit_acted_against,
                "lr_measured": round(self.full_backtest.lr_measured, 4),
                "precision": round(self.full_backtest.precision, 4),
                "recall": round(self.full_backtest.recall, 4),
            },
            "legit_card_day_sample": {
                "n_card_days": self.legit_sample_n,
                "fired_count": self.legit_sample_fired,
                "firing_rate": round(self.legit_sample_rate, 6),
                "train_fraud_firing_rate": round(self.train_fraud_rate, 6),
                "val_fraud_firing_rate": round(self.val_fraud_rate, 6),
                "lr_vs_legit_sample": round(self.lr_vs_legit_sample, 4),
            },
            "undocumented_caught": self.undocumented_caught,
            "residual_caught": self.residual_caught,
            "attestation": self.attestation.to_dict(),
        }


def _sliding_distinct_cards_7d(
    ts_sec: np.ndarray,
    group_codes: np.ndarray,
    card_codes: np.ndarray,
    valid_mask: np.ndarray,
    window_sec: int = 7 * 86400,
    max_cap: int = 50,
) -> np.ndarray:
    """Compute for each row the number of distinct cards sharing `group_codes` in [t - window_sec, t].

    Excludes any group code whose total distinct card count exceeds `max_cap` (learned 99th-percentile
    cardinality cap from Jul-Oct legitimate transactions so generic values are ignored).
    """
    n = len(ts_sec)
    out = np.ones(n, dtype=np.int32)
    idx = np.flatnonzero(valid_mask)
    if len(idx) == 0:
        return out

    g_sub = group_codes[idx]
    t_sub = ts_sec[idx]
    c_sub = card_codes[idx]

    order = np.lexsort((t_sub, g_sub))
    idx_sorted = idx[order]
    g_sorted = g_sub[order]
    t_sorted = t_sub[order]
    c_sorted = c_sub[order]

    diff = np.empty(len(g_sorted) + 1, dtype=bool)
    diff[0] = True
    diff[-1] = True
    diff[1:-1] = g_sorted[1:] != g_sorted[:-1]
    starts = np.flatnonzero(diff[:-1])
    ends = np.flatnonzero(diff[1:])
    lengths = ends - starts
    multi = np.flatnonzero(lengths >= 2)

    res_sorted = np.ones(len(g_sorted), dtype=np.int32)
    for m_idx in multi:
        s = int(starts[m_idx])
        e = int(ends[m_idx])
        c_grp = c_sorted[s:e]
        if np.all(c_grp == c_grp[0]):
            continue
        # Drop generic high-cardinality elements above max_cap
        n_unique_total = len(np.unique(c_grp))
        if n_unique_total > max_cap:
            continue
        t_grp = t_sorted[s:e]
        counts: dict[int, int] = {}
        left = 0
        for right in range(e - s):
            cr = int(c_grp[right])
            counts[cr] = counts.get(cr, 0) + 1
            t_lim = t_grp[right] - window_sec
            while t_grp[left] < t_lim:
                cl = int(c_grp[left])
                cnt = counts[cl] - 1
                if cnt == 0:
                    del counts[cl]
                else:
                    counts[cl] = cnt
                left += 1
            res_sorted[s + right] = len(counts)

    out[idx_sorted] = res_sorted
    return out


def _is_specific_device_profile(dev_series: pd.Series) -> np.ndarray:
    """Exclude empty or generic device strings ('Windows', 'iOS Device', etc.)."""
    s = dev_series.fillna("").astype(str)
    parts_first = s.str.split("|").str[0].str.strip()
    non_empty = (s != "") & (s != "? | ? | ? | ?") & (~s.isin(GENERIC_DEVICES))
    not_generic_prefix = ~parts_first.isin({"?", "Windows", "iOS Device", "MacOS", "Trident/7.0", ""})
    return (non_empty & not_generic_prefix).to_numpy()


@lru_cache(maxsize=1)
def tx_predicates_all() -> pd.DataFrame:
    """Compute the 44 boolean predicates for all transactions in `tx_all()` strictly causally
    (every rolling / novelty / sharing feature only looks at transactions at or before timestamp `t`).
    Indexed by `TransactionID`.
    """
    tx = tx_all().sort_values(["t", "TransactionID"], kind="stable").reset_index(drop=True)
    n = len(tx)
    ts_sec = (tx["t"].astype("int64") // 10**9).to_numpy()
    amt = tx["amount"].to_numpy(dtype=np.float64)
    prod = tx["ProductCD"].astype(str).to_numpy()
    chan = tx["channel"].astype(str).to_numpy()
    hr = tx["t"].dt.hour.to_numpy()
    d1 = tx["D1"].to_numpy(dtype=np.float64)
    c1 = tx["C1"].to_numpy(dtype=np.float64)
    addr1 = tx["addr1"].to_numpy(dtype=np.float64)
    dist1 = tx["dist1"].to_numpy(dtype=np.float64)

    card_codes, _ = pd.factorize(tx["card_id"], sort=False)
    card_order = np.lexsort((np.arange(n), card_codes))
    c_sorted = card_codes[card_order]
    t_card = ts_sec[card_order]
    amt_card = amt[card_order]
    addr_card = addr1[card_order]

    same_card_prev1 = np.zeros(n, dtype=bool)
    same_card_prev1[1:] = c_sorted[1:] == c_sorted[:-1]

    same_card_prev2 = np.zeros(n, dtype=bool)
    same_card_prev2[2:] = c_sorted[2:] == c_sorted[:-2]

    same_card_prev4 = np.zeros(n, dtype=bool)
    same_card_prev4[4:] = c_sorted[4:] == c_sorted[:-4]

    gap_sorted = np.full(n, np.nan, dtype=np.float64)
    gap_sorted[same_card_prev1] = t_card[same_card_prev1] - t_card[np.roll(same_card_prev1, -1)]

    dt_2_sorted = np.full(n, np.inf, dtype=np.float64)
    dt_2_sorted[same_card_prev2] = t_card[same_card_prev2] - t_card[np.roll(same_card_prev2, -2)]

    dt_4_sorted = np.full(n, np.inf, dtype=np.float64)
    dt_4_sorted[same_card_prev4] = t_card[same_card_prev4] - t_card[np.roll(same_card_prev4, -4)]

    df_c = pd.DataFrame({"c": c_sorted, "amt": amt_card, "addr": addr_card})
    cum_sum = df_c.groupby("c", sort=False)["amt"].cumsum().to_numpy() - amt_card
    cum_cnt = df_c.groupby("c", sort=False).cumcount().to_numpy()
    prior_mean_sorted = np.where(cum_cnt > 0, cum_sum / np.maximum(cum_cnt, 1), np.nan)

    prev_addr_sorted = df_c.groupby("c", sort=False)["addr"].shift(1).to_numpy()
    reg_chg_sorted = (~np.isnan(addr_card)) & (~np.isnan(prev_addr_sorted)) & (addr_card != prev_addr_sorted)

    inv_card_order = np.empty(n, dtype=np.int64)
    inv_card_order[card_order] = np.arange(n)

    gap_sec = gap_sorted[inv_card_order]
    dt_2 = dt_2_sorted[inv_card_order]
    dt_4 = dt_4_sorted[inv_card_order]
    prior_mean = prior_mean_sorted[inv_card_order]
    region_changed = reg_chg_sorted[inv_card_order]

    # Exact amount repeat on same card within 7 days
    amt_cents = np.round(amt * 100).astype(np.int64)
    ca_codes, _ = pd.factorize(
        pd.MultiIndex.from_arrays([card_codes, amt_cents]), sort=False
    )
    ca_order = np.lexsort((np.arange(n), ca_codes))
    ca_sorted = ca_codes[ca_order]
    t_ca = ts_sec[ca_order]
    same_ca_prev = np.zeros(n, dtype=bool)
    same_ca_prev[1:] = ca_sorted[1:] == ca_sorted[:-1]
    ca_dt_sorted = np.full(n, np.inf, dtype=np.float64)
    ca_dt_sorted[same_ca_prev] = t_ca[same_ca_prev] - t_ca[np.roll(same_ca_prev, -1)]
    inv_ca_order = np.empty(n, dtype=np.int64)
    inv_ca_order[ca_order] = np.arange(n)
    exact_amt_dt = ca_dt_sorted[inv_ca_order]

    # Specific device profile novelty on card
    has_specific_dev = _is_specific_device_profile(tx["device_profile"])
    dev_prof = tx["device_profile"].fillna("").astype(str).to_numpy()
    dev_codes, _ = pd.factorize(dev_prof, sort=False)
    cd_codes, _ = pd.factorize(
        pd.MultiIndex.from_arrays([card_codes, dev_codes]), sort=False
    )
    first_cd = ~pd.Series(cd_codes).duplicated(keep="first").to_numpy()
    dev_novel_on_card = has_specific_dev & first_cd & (~np.isnan(gap_sec))

    # Cross-card sharing within 7 days with cardinality caps:
    # 1) Specific device_profile shared across >= 2 cards in 7d (cap=25 distinct cards)
    n_cards_dev_7d = _sliding_distinct_cards_7d(
        ts_sec, dev_codes, card_codes, has_specific_dev, max_cap=25
    )

    # 2) Shared (R_emaildomain, addr1, ProductCD) across >= 3 cards in 7d (cap=35)
    remail = tx["R_emaildomain"].fillna("").astype(str).to_numpy()
    has_remail_addr = (remail != "") & (~np.isnan(addr1))
    ra_codes, _ = pd.factorize(
        pd.MultiIndex.from_arrays(
            [remail, np.nan_to_num(addr1, nan=-1.0).astype(np.int64), prod]
        ),
        sort=False,
    )
    n_cards_remail_7d = _sliding_distinct_cards_7d(
        ts_sec, ra_codes, card_codes, has_remail_addr, max_cap=35
    )

    # 3) Shared non-round (addr1, exact_amount) across >= 2 cards in 7d (cap=20)
    is_non_round_amt = (amt_cents % 100) != 0
    has_addr_nonround = (~np.isnan(addr1)) & is_non_round_amt
    aa_codes, _ = pd.factorize(
        pd.MultiIndex.from_arrays([np.nan_to_num(addr1, nan=-1.0).astype(np.int64), amt_cents]),
        sort=False,
    )
    n_cards_addramt_7d = _sliding_distinct_cards_7d(
        ts_sec, aa_codes, card_codes, has_addr_nonround, max_cap=20
    )

    pemail = tx["P_emaildomain"].fillna("").astype(str).to_numpy()
    pemail_counts = pd.Series(pemail).value_counts()
    rare_domains = set(pemail_counts[pemail_counts < 100].index) - {""}

    id15 = tx["id_15"].fillna("").astype(str).to_numpy()
    dev_info = tx["DeviceInfo"].fillna("").astype(str).to_numpy()
    dev_type = tx["DeviceType"].fillna("").astype(str).to_numpy()
    id23 = tx["id_23"].fillna("").astype(str).to_numpy()

    preds = pd.DataFrame(
        {
            "amt_micro_lt15": amt < 15.0,
            "amt_small_15_50": (amt >= 15.0) & (amt < 50.0),
            "amt_med_50_200": (amt >= 50.0) & (amt < 200.0),
            "amt_high_200_500": (amt >= 200.0) & (amt < 500.0),
            "amt_vhigh_ge500": amt >= 500.0,
            "amt_cents_round": (amt_cents % 100) == 0,
            "amt_gt_3x_card_med": (~np.isnan(prior_mean)) & (prior_mean > 0) & (amt > 3.0 * prior_mean),
            "exact_amt_repeat_card_7d": exact_amt_dt <= 7 * 86400,
            "prod_W": prod == "W",
            "prod_C": prod == "C",
            "prod_R": prod == "R",
            "prod_H": prod == "H",
            "prod_S": prod == "S",
            "chan_online": chan == "online",
            "chan_in_person": chan == "in_person",
            "hr_night_0_5": hr <= 5,
            "hr_morn_6_11": (hr >= 6) & (hr <= 11),
            "hr_aft_12_17": (hr >= 12) & (hr <= 17),
            "hr_eve_18_23": hr >= 18,
            "gap_rapid_lt10m": (~np.isnan(gap_sec)) & (gap_sec <= 600),
            "gap_short_lt1h": (~np.isnan(gap_sec)) & (gap_sec <= 3600),
            "gap_dormant_gt14d": (~np.isnan(gap_sec)) & (gap_sec > 14 * 86400),
            "gap_first_on_card": np.isnan(gap_sec),
            "card_age_new_le7d": (~np.isnan(d1)) & (d1 <= 7.0),
            "card_age_young_8_60d": (~np.isnan(d1)) & (d1 > 7.0) & (d1 <= 60.0),
            "card_age_mature_gt60d": (~np.isnan(d1)) & (d1 > 60.0),
            "dev_novel_on_card": dev_novel_on_card,
            "dev_id15_new": id15 == "New",
            "dev_missing_online": (chan == "online") & (dev_info == ""),
            "dev_mobile": dev_type == "mobile",
            "email_r_present": remail != "",
            "email_p_r_mismatch": (pemail != "") & (remail != "") & (pemail != remail),
            "email_p_free": pd.Series(pemail).isin(FREE_EMAILS).to_numpy(),
            "email_p_rare": pd.Series(pemail).isin(rare_domains).to_numpy(),
            "id23_anon_proxy": (
                pd.Series(id23).str.contains("ANONYMOUS|HIDDEN", case=False, regex=True).to_numpy()
                & (chan == "online")
            ),
            "region_changed": region_changed,
            "region_missing": np.isnan(addr1),
            "dist_far_gt100": (~np.isnan(dist1)) & (dist1 > 100.0),
            "burst_ge3_1h": dt_2 <= 3600,
            "burst_ge5_24h": dt_4 <= 86400,
            "shared_dev_7d_ge2cards": n_cards_dev_7d >= 2,
            "shared_remail_7d_ge3cards": n_cards_remail_7d >= 3,
            "shared_addr_amt_7d_ge2cards": n_cards_addramt_7d >= 2,
            "c1_high_ge5": (~np.isnan(c1)) & (c1 >= 5.0),
        },
        index=tx["TransactionID"].to_numpy(),
    )
    return preds


@lru_cache(maxsize=1)
def featurize_closed_cases() -> pd.DataFrame:
    """Featurize each of the 5,585 closed cases using only transactions in the case episode
    with `t <= opened_at` (strict as-of discipline).
    """
    cc = closed_cases().copy()
    tx = tx_all()
    preds_tx = tx_predicates_all()

    tx_time_map = tx.set_index("TransactionID")["t"]
    opened_ts = pd.to_datetime(cc["opened_at"])

    rows = []
    rep_txn_ids = []
    for idx, row in enumerate(cc.itertuples(index=False)):
        t_open = opened_ts.iat[idx]
        t_list = [tid for tid in row.txn_list if tid in preds_tx.index and tx_time_map.at[tid] <= t_open]
        if not t_list and row.txn_list:
            t_list = [tid for tid in row.txn_list if tid in preds_tx.index]
        if t_list:
            sub = preds_tx.loc[t_list].any(axis=0).to_numpy(dtype=bool)
            rep_txn_ids.append(int(t_list[0]))
        else:
            sub = np.zeros(len(PREDICATE_NAMES), dtype=bool)
            rep_txn_ids.append(-1)
        rows.append(sub)

    feat_mat = np.vstack(rows)
    out = pd.DataFrame(feat_mat, columns=PREDICATE_NAMES, index=cc["case_id"])
    out["outcome"] = cc["outcome"].to_numpy()
    out["pattern"] = cc["pattern"].to_numpy()
    out["opened_at"] = opened_ts.to_numpy()
    out["card_id"] = cc["card_id"].to_numpy()
    out["customer_id"] = cc["customer_id"].to_numpy()
    out["rep_txn_id"] = rep_txn_ids
    return out


@lru_cache(maxsize=1)
def featurize_legit_card_days_jul_oct(
    n_samples: int = LEGIT_SAMPLE_SIZE,
    seed: int = LEGIT_SAMPLE_SEED,
) -> pd.DataFrame:
    """Sample `n_samples` (default 6,000) legitimate card-days from Jul-Oct 2016 (fixed seed).

    Jul-Oct 2016 is fully labelled: any transaction not named in any closed case is legitimate.
    We exclude all transactions and cards that ever had confirmed fraud in Jul-Oct, group the
    remaining legitimate Jul-Oct transactions by `(card_id, day)`, sample `n_samples` card-days
    deterministically, and aggregate the predicate matrix over each card-day episode with the
    exact same `.any(axis=0)` episode logic used for closed cases.
    """
    cc = closed_cases()
    tx = tx_all()
    preds_tx = tx_predicates_all()

    case_txns: set[int] = set()
    for t_list in cc["txn_list"]:
        case_txns.update(t_list)
    fraud_cards = set(cc[cc["outcome"] == "confirmed_fraud"]["card_id"].unique())

    jul_oct = tx[(tx["t"] < SPLIT_TS) & (~tx["TransactionID"].isin(case_txns)) & (~tx["card_id"].isin(fraud_cards))]
    day_code = (jul_oct["TransactionDT"] // 86400).to_numpy()
    cd_df = pd.DataFrame(
        {
            "TransactionID": jul_oct["TransactionID"].to_numpy(),
            "card_id": jul_oct["card_id"].to_numpy(),
            "day": day_code,
        }
    )

    # Group by (card_id, day) and sample n_samples distinct card-days with fixed seed
    grouped = cd_df.groupby(["card_id", "day"], sort=False)["TransactionID"].apply(list)
    rng = np.random.default_rng(seed)
    total_cd = len(grouped)
    k = min(n_samples, total_cd)
    chosen_idx = np.sort(rng.choice(total_cd, size=k, replace=False))
    sampled_series = grouped.iloc[chosen_idx]

    # Vectorised aggregation of predicates over the sampled card-days
    flat_tids = [tid for t_list in sampled_series for tid in t_list]
    flat_cd_idx = [i for i, t_list in enumerate(sampled_series) for _ in t_list]

    sub_preds = preds_tx.loc[flat_tids].to_numpy(dtype=bool)
    cd_df_mat = pd.DataFrame(sub_preds, columns=PREDICATE_NAMES)
    cd_df_mat["cd_idx"] = flat_cd_idx
    agg = cd_df_mat.groupby("cd_idx", sort=False).any()
    agg.index = [f"LEGIT-CD-{i:05d}" for i in range(len(agg))]
    return agg


def _compute_backtest_from_mask(
    detector_name: str,
    fired_mask: np.ndarray,
    case_ids: np.ndarray,
    is_fraud: np.ndarray,
    legit_sample_rate: float | None = None,
) -> Backtest:
    caught = case_ids[fired_mask & is_fraud].tolist()
    missed = case_ids[(~fired_mask) & is_fraud].tolist()
    legit_acted = case_ids[fired_mask & (~is_fraud)].tolist()

    n_fraud = max(int(is_fraud.sum()), 1)
    n_legit = max(int((~is_fraud).sum()), 1)
    p_fire_fraud = (len(caught) + 1.0) / (n_fraud + 2.0)
    if legit_sample_rate is not None and legit_sample_rate > 0:
        lr = float(p_fire_fraud / legit_sample_rate)
    else:
        p_fire_legit = (len(legit_acted) + 1.0) / (n_legit + 2.0)
        lr = float(p_fire_fraud / p_fire_legit)

    return Backtest(
        detector=detector_name,
        n_cases=len(case_ids),
        caught=caught,
        missed=missed,
        legit_acted_against=legit_acted,
        lr_measured=lr,
    )


def _render_detector_source(
    cls_name: str,
    detector_name: str,
    pattern_name: str,
    predicates: list[str],
    lr_train: float,
    description: str,
) -> str:
    preds_repr = repr(predicates)
    desc_escaped = description.replace('"', '\\"')
    return f'''"""Auto-discovered fraud detector `{detector_name}`.
Generated deterministically by `assay.r2.discover`.
Rule: {" AND ".join(predicates)}
Description: {desc_escaped}
DO NOT LOAD directly without L2 approval via `assay.r2.discover.approve`.
"""

from __future__ import annotations

import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.discover import (
    evaluate_rule_asof,
    featurize_closed_cases,
    featurize_legit_card_days_jul_oct,
)
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding


class {cls_name}(Detector):
    # Registered name intentionally left blank at class definition time so importing
    # this module never auto-registers an unapproved detector into REGISTRY.
    name: str = ""
    proposed_name: str = "{detector_name}"
    pattern: str = "{pattern_name}"
    predicates: list[str] = {preds_repr}
    lr_train: float = {lr_train:.4f}

    def run(self, ctx: CaseCtx) -> list[Finding]:
        matched_txns, connected_cards = evaluate_rule_asof(ctx, self.predicates)
        if not matched_txns:
            return []
        claim = Attestation(
            what=f"Discovered rule {{self.proposed_name}} ({desc_escaped}) matched {{len(matched_txns)}} transaction(s) on card {{ctx.card_id}}.",
            evidence=Evidence(
                txn_ids=matched_txns,
                card_ids=[ctx.card_id] + connected_cards,
                customer_ids=[ctx.customer_id],
                case_ids=[ctx.case_id],
            ),
            query="assay.r2.discover.verify_case_rule",
            params={{"case_id": ctx.case_id, "predicates": self.predicates}},
        )
        return [
            Finding(
                detector=self.name or self.proposed_name,
                pattern=self.pattern,
                claim=claim,
                lr=self.lr_train,
                affected_txn_ids=matched_txns,
                connected_card_ids=connected_cards,
            )
        ]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        df = featurize_closed_cases()
        sub = df.loc[df.index.intersection(cc["case_id"])]
        mask = sub[self.predicates].all(axis=1).to_numpy()
        is_fraud = (sub["outcome"] == "confirmed_fraud").to_numpy()
        c_ids = sub.index.to_numpy()
        caught = c_ids[mask & is_fraud].tolist()
        missed = c_ids[(~mask) & is_fraud].tolist()
        legit = c_ids[mask & (~is_fraud)].tolist()
        n_f = max(int(is_fraud.sum()), 1)
        legit_cd = featurize_legit_card_days_jul_oct()
        legit_cd_fired = int(legit_cd[self.predicates].all(axis=1).sum())
        p_legit = (legit_cd_fired + 1.0) / (len(legit_cd) + 2.0)
        p_fraud = (len(caught) + 1.0) / (n_f + 2.0)
        return Backtest(
            detector=self.name or self.proposed_name,
            n_cases=len(sub),
            caught=caught,
            missed=missed,
            legit_acted_against=legit,
            lr_measured=float(p_fraud / p_legit),
        )
'''


def evaluate_rule_asof(ctx: CaseCtx, predicates: list[str]) -> tuple[list[int], list[str]]:
    """Evaluate a conjunction of predicates as-of `ctx.opened_at` for a single `CaseCtx`."""
    preds_all = tx_predicates_all()
    card_tx = ctx.tx[(ctx.tx["card_id"] == ctx.card_id) & (ctx.tx["t"] <= ctx.opened_at)]
    if card_tx.empty:
        return [], []
    t_min = ctx.opened_at - pd.Timedelta(days=7)
    ep_tx = card_tx[(card_tx["t"] >= t_min) | (card_tx["TransactionID"] == ctx.flagged_txn_id)]
    t_ids = [int(t) for t in ep_tx["TransactionID"].tolist() if int(t) in preds_all.index]
    if not t_ids:
        return [], []
    sub_preds = preds_all.loc[t_ids, predicates]
    matching_rows = sub_preds[sub_preds.all(axis=1)]
    if matching_rows.empty:
        return [], []
    matched_txns = [int(x) for x in matching_rows.index.tolist()]
    return matched_txns, []


def verify_case_rule(case_id: str, predicates: list[str]) -> dict:
    """Re-runnable query function for Attestation verification on a single case."""
    df = featurize_closed_cases()
    if case_id in df.index:
        row = df.loc[case_id]
        fired = bool(all(row[p] for p in predicates))
        return {
            "case_id": case_id,
            "fired": fired,
            "card_id": str(row["card_id"]),
            "customer_id": str(row["customer_id"]),
            "txn_id": int(row["rep_txn_id"]),
        }
    return {"case_id": case_id, "fired": False}


def verify_backtest_rule(predicates: list[str], split: str = "all") -> dict:
    """Re-runnable query function that reproduces exact backtest IDs and legit card-day firing."""
    df = featurize_closed_cases()
    if split == "train":
        df = df[df["opened_at"] < SPLIT_TS]
    elif split == "val":
        df = df[(df["opened_at"] >= SPLIT_TS) & (df["opened_at"] < JAN_TS)]
    mask = df[predicates].all(axis=1).to_numpy()
    is_fraud = (df["outcome"] == "confirmed_fraud").to_numpy()
    c_ids = df.index.to_numpy()

    legit_cd = featurize_legit_card_days_jul_oct()
    legit_cd_fired = int(legit_cd[predicates].all(axis=1).sum())

    return {
        "caught": c_ids[mask & is_fraud].tolist(),
        "missed": c_ids[(~mask) & is_fraud].tolist(),
        "legit_acted_against": c_ids[mask & (~is_fraud)].tolist(),
        "legit_card_day_fired": legit_cd_fired,
        "legit_card_day_n": len(legit_cd),
    }


def mine_and_propose(
    top_k: int = 5,
    max_legit_train: int = 2,
    max_legit_sample_rate: float = 0.015,
) -> list[DiscoveredProposal]:
    """Mine 2-3 predicate conjunctions on the Jul-Oct training split, validate out-of-time on
    Nov-Dec and against the 6,000 legitimate card-day sample from Jul-Oct, emit Detector code
    into `data/stage/r2/proposed/`, and write attested JSONs.
    """
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    df = featurize_closed_cases()
    legit_cd_df = featurize_legit_card_days_jul_oct()

    train_df = df[df["opened_at"] < SPLIT_TS]
    val_df = df[(df["opened_at"] >= SPLIT_TS) & (df["opened_at"] < JAN_TS)]

    X_train = train_df[PREDICATE_NAMES].to_numpy(dtype=bool)
    is_fraud_tr = (train_df["outcome"] == "confirmed_fraud").to_numpy()
    is_legit_tr = ~is_fraud_tr
    pat_tr = train_df["pattern"].to_numpy()

    X_legit_cd = legit_cd_df[PREDICATE_NAMES].to_numpy(dtype=bool)
    n_legit_cd = len(X_legit_cd)
    max_cd_fires = int(n_legit_cd * max_legit_sample_rate)

    is_undoc_tr = is_fraud_tr & np.isin(pat_tr, ["undocumented", "card_testing"])

    p_idx = {name: i for i, name in enumerate(PREDICATE_NAMES)}
    explained_by_obvious = (
        (X_train[:, p_idx["chan_online"]] & X_train[:, p_idx["dev_novel_on_card"]])
        | (X_train[:, p_idx["chan_in_person"]] & X_train[:, p_idx["region_changed"]])
    )
    is_residual_tr = is_fraud_tr & (~explained_by_obvious | is_undoc_tr)

    f_Matrix = X_train[is_fraud_tr]
    l_Matrix = X_train[is_legit_tr]
    u_Matrix = X_train[is_undoc_tr]
    r_Matrix = X_train[is_residual_tr]

    single_supp = f_Matrix.sum(axis=0)
    active_cols = [i for i in range(len(PREDICATE_NAMES)) if single_supp[i] >= 3]

    excl_groups = [
        {"amt_micro_lt15", "amt_small_15_50", "amt_med_50_200", "amt_high_200_500", "amt_vhigh_ge500"},
        {"prod_W", "prod_C", "prod_R", "prod_H", "prod_S"},
        {"chan_online", "chan_in_person"},
        {"hr_night_0_5", "hr_morn_6_11", "hr_aft_12_17", "hr_eve_18_23"},
        {"card_age_new_le7d", "card_age_young_8_60d", "card_age_mature_gt60d"},
    ]

    def _compatible(cols: tuple[int, ...]) -> bool:
        names = {PREDICATE_NAMES[c] for c in cols}
        for grp in excl_groups:
            if len(names & grp) > 1:
                return False
        if "gap_rapid_lt10m" in names and "gap_short_lt1h" in names:
            return False
        if "gap_first_on_card" in names and (
            "gap_rapid_lt10m" in names or "gap_short_lt1h" in names or "gap_dormant_gt14d" in names
        ):
            return False
        return True

    candidates: list[tuple[float, tuple[int, ...], int, int, int, int, int]] = []

    # Stage 2: 2-predicate conjunctions
    valid_pairs: list[tuple[int, int]] = []
    for i, j in combinations(active_cols, 2):
        if not _compatible((i, j)):
            continue
        f_cnt = int(np.logical_and(f_Matrix[:, i], f_Matrix[:, j]).sum())
        if f_cnt < 3:
            continue
        valid_pairs.append((i, j))
        l_cnt = int(np.logical_and(l_Matrix[:, i], l_Matrix[:, j]).sum())
        if l_cnt > max_legit_train:
            continue
        cd_cnt = int(np.logical_and(X_legit_cd[:, i], X_legit_cd[:, j]).sum())
        if cd_cnt > max_cd_fires:
            continue
        u_cnt = int(np.logical_and(u_Matrix[:, i], u_Matrix[:, j]).sum())
        r_cnt = int(np.logical_and(r_Matrix[:, i], r_Matrix[:, j]).sum())
        if u_cnt >= 2 or r_cnt >= 10:
            p_f = (f_cnt + 1.0) / (len(f_Matrix) + 2.0)
            p_cd = (cd_cnt + 1.0) / (n_legit_cd + 2.0)
            lr_cd = p_f / p_cd
            score = (u_cnt * 40.0 + r_cnt * 1.5 + lr_cd * 0.5) / ((l_cnt + 1.0) * (1.0 + cd_cnt / 15.0))
            candidates.append((score, (i, j), f_cnt, l_cnt, u_cnt, r_cnt, cd_cnt))

    # Stage 3: 3-predicate conjunctions from promising pairs
    seed_pairs = [
        (i, j)
        for (i, j) in valid_pairs
        if int(np.logical_and(u_Matrix[:, i], u_Matrix[:, j]).sum()) >= 2
        or (
            int(np.logical_and(r_Matrix[:, i], r_Matrix[:, j]).sum()) >= 12
            and int(np.logical_and(l_Matrix[:, i], l_Matrix[:, j]).sum()) <= 15
        )
    ]

    seen_triples: set[tuple[int, int, int]] = set()
    for i, j in seed_pairs:
        f_ij = np.logical_and(f_Matrix[:, i], f_Matrix[:, j])
        l_ij = np.logical_and(l_Matrix[:, i], l_Matrix[:, j])
        u_ij = np.logical_and(u_Matrix[:, i], u_Matrix[:, j])
        r_ij = np.logical_and(r_Matrix[:, i], r_Matrix[:, j])
        cd_ij = np.logical_and(X_legit_cd[:, i], X_legit_cd[:, j])
        for k in active_cols:
            if k in (i, j):
                continue
            triple = tuple(sorted((i, j, k)))
            if triple in seen_triples or not _compatible(triple):
                continue
            seen_triples.add(triple)
            f_cnt = int(np.logical_and(f_ij, f_Matrix[:, k]).sum())
            if f_cnt < 3:
                continue
            l_cnt = int(np.logical_and(l_ij, l_Matrix[:, k]).sum())
            if l_cnt > max_legit_train:
                continue
            cd_cnt = int(np.logical_and(cd_ij, X_legit_cd[:, k]).sum())
            if cd_cnt > max_cd_fires:
                continue
            u_cnt = int(np.logical_and(u_ij, u_Matrix[:, k]).sum())
            r_cnt = int(np.logical_and(r_ij, r_Matrix[:, k]).sum())
            if u_cnt >= 2 or r_cnt >= 10:
                p_f = (f_cnt + 1.0) / (len(f_Matrix) + 2.0)
                p_cd = (cd_cnt + 1.0) / (n_legit_cd + 2.0)
                lr_cd = p_f / p_cd
                score = (u_cnt * 40.0 + r_cnt * 1.5 + lr_cd * 0.5) / ((l_cnt + 1.0) * (1.0 + cd_cnt / 15.0))
                candidates.append((score, triple, f_cnt, l_cnt, u_cnt, r_cnt, cd_cnt))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by Jaccard similarity on training caught set so the top_k rules cover diverse patterns
    selected: list[tuple[float, tuple[int, ...], np.ndarray, int]] = []
    for score, cols, f_cnt, l_cnt, u_cnt, r_cnt, cd_cnt in candidates:
        mask_tr = X_train[:, cols].all(axis=1)
        caught_mask = mask_tr & is_fraud_tr
        is_dup = False
        for _, prev_cols, prev_caught, _ in selected:
            inter = int(np.logical_and(caught_mask, prev_caught).sum())
            union = int(np.logical_or(caught_mask, prev_caught).sum())
            if union > 0 and (inter / union) > 0.50:
                is_dup = True
                break
            if len(set(cols) & set(prev_cols)) >= 2 and len(cols) == len(prev_cols):
                is_dup = True
                break
        if not is_dup:
            selected.append((score, cols, caught_mask, cd_cnt))
        if len(selected) >= top_k:
            break

    proposals: list[DiscoveredProposal] = []
    X_val = val_df[PREDICATE_NAMES].to_numpy(dtype=bool)
    is_fraud_val = (val_df["outcome"] == "confirmed_fraud").to_numpy()

    X_all = df[PREDICATE_NAMES].to_numpy(dtype=bool)
    is_fraud_all = (df["outcome"] == "confirmed_fraud").to_numpy()
    pat_all = df["pattern"].to_numpy()
    explained_all = (
        (X_all[:, p_idx["chan_online"]] & X_all[:, p_idx["dev_novel_on_card"]])
        | (X_all[:, p_idx["chan_in_person"]] & X_all[:, p_idx["region_changed"]])
    )

    n_fraud_tr_total = max(int(is_fraud_tr.sum()), 1)
    n_fraud_val_total = max(int(is_fraud_val.sum()), 1)

    for rank, (score, cols, _, cd_cnt) in enumerate(selected, start=1):
        preds = [PREDICATE_NAMES[c] for c in cols]
        short_slug = "_".join(p.split("_")[0] + p.split("_")[1] for p in preds[:2])
        det_name = f"disc_{rank:02d}_{short_slug}"
        cls_name = f"DiscoveredRule{rank:02d}Detector"
        pattern_name = f"undocumented:{det_name}"
        desc = " AND ".join(PREDICATE_DESCRIPTIONS[p] for p in preds)

        tr_mask = X_train[:, cols].all(axis=1)
        val_mask = X_val[:, cols].all(axis=1)
        all_mask = X_all[:, cols].all(axis=1)

        legit_rate_smoothed = (cd_cnt + 1.0) / (n_legit_cd + 2.0)
        legit_rate_raw = cd_cnt / max(n_legit_cd, 1)

        bt_train = _compute_backtest_from_mask(
            det_name, tr_mask, train_df.index.to_numpy(), is_fraud_tr, legit_sample_rate=legit_rate_smoothed
        )
        bt_val = _compute_backtest_from_mask(
            det_name, val_mask, val_df.index.to_numpy(), is_fraud_val, legit_sample_rate=legit_rate_smoothed
        )
        bt_full = _compute_backtest_from_mask(
            det_name, all_mask, df.index.to_numpy(), is_fraud_all, legit_sample_rate=legit_rate_smoothed
        )

        tr_fraud_rate = len(bt_train.caught) / n_fraud_tr_total
        val_fraud_rate = len(bt_val.caught) / n_fraud_val_total
        lr_vs_legit = ((len(bt_train.caught) + 1.0) / (n_fraud_tr_total + 2.0)) / legit_rate_smoothed

        undoc_caught = df.index[all_mask & is_fraud_all & (pat_all == "undocumented")].tolist()
        resid_caught = df.index[all_mask & is_fraud_all & (~explained_all | (pat_all == "undocumented"))].tolist()

        src_path = PROPOSED_DIR / f"{det_name}.py"
        src_code = _render_detector_source(
            cls_name=cls_name,
            detector_name=det_name,
            pattern_name=pattern_name,
            predicates=preds,
            lr_train=lr_vs_legit,
            description=desc,
        )
        src_path.write_text(src_code, encoding="utf-8")

        caught_df = df.loc[bt_full.caught]
        att = Attestation(
            what=(
                f"Proposed detector {det_name} ({' & '.join(preds)}) caught {len(bt_full.caught)} "
                f"confirmed fraud cases ({len(bt_train.caught)} train Jul-Oct, {len(bt_val.caught)} val Nov-Dec, "
                f"{len(undoc_caught)} labelled undocumented), acted against "
                f"{len(bt_full.legit_acted_against)} cleared cases, and fired on "
                f"{cd_cnt}/{n_legit_cd} ({legit_rate_raw:.3%}) Jul-Oct legit card-days (LR={lr_vs_legit:.2f})."
            ),
            evidence=Evidence(
                txn_ids=[int(x) for x in caught_df["rep_txn_id"].tolist() if int(x) > 0],
                card_ids=caught_df["card_id"].astype(str).tolist(),
                customer_ids=caught_df["customer_id"].astype(str).tolist(),
                case_ids=bt_full.caught + bt_full.legit_acted_against,
            ),
            query="assay.r2.discover.verify_backtest_rule",
            params={"predicates": preds, "split": "all"},
        )

        json_path = PROPOSED_DIR / f"{det_name}.json"
        prop = DiscoveredProposal(
            name=det_name,
            pattern=pattern_name,
            predicates=preds,
            description=desc,
            source_path=str(src_path),
            proposal_json_path=str(json_path),
            train_backtest=bt_train,
            val_backtest=bt_val,
            full_backtest=bt_full,
            legit_sample_n=n_legit_cd,
            legit_sample_fired=cd_cnt,
            legit_sample_rate=legit_rate_raw,
            train_fraud_rate=tr_fraud_rate,
            val_fraud_rate=val_fraud_rate,
            lr_vs_legit_sample=lr_vs_legit,
            undocumented_caught=undoc_caught,
            residual_caught=resid_caught,
            attestation=att,
        )
        json_path.write_text(json.dumps(prop.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        proposals.append(prop)

    return proposals


def _load_approvals() -> dict[str, dict]:
    if not APPROVED_FILE.exists():
        return {}
    try:
        return json.loads(APPROVED_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def reset_approvals() -> None:
    """Clear all approved proposals and remove any dynamically loaded `disc_` detectors from REGISTRY."""
    if APPROVED_FILE.exists():
        APPROVED_FILE.unlink()
    for k in list(REGISTRY.keys()):
        if k.startswith("disc_"):
            del REGISTRY[k]


def approve(name: str, approver_token: str) -> dict:
    """Approve a proposed detector with server-side L2 role verification.

    Raises PermissionError if `approver_token` is invalid or belongs to an L1/auto role,
    or FileNotFoundError if the proposal does not exist.
    """
    if not approver_token or not isinstance(approver_token, str):
        raise PermissionError("Missing or invalid approver token; L2 human approval required.")

    if approver_token in _L1_TOKENS:
        info = _L1_TOKENS[approver_token]
        raise PermissionError(
            f"Role '{info['role']}' ({info['approver_id']}) is not authorised to approve new detectors; L2 required."
        )

    if approver_token not in _L2_APPROVERS:
        raise PermissionError("Unrecognised approver token; server-side L2 verification failed.")

    src_path = PROPOSED_DIR / f"{name}.py"
    json_path = PROPOSED_DIR / f"{name}.json"
    if not src_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Proposed detector '{name}' not found in {PROPOSED_DIR}.")

    approver_info = _L2_APPROVERS[approver_token]
    code_bytes = src_path.read_bytes()
    code_sha256 = hashlib.sha256(code_bytes).hexdigest()
    ts_now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    sig_payload = f"{name}|{code_sha256}|{approver_info['approver_id']}|L2|{dataset_tag()}".encode("utf-8")
    sig = hmac.new(approver_token.encode("utf-8"), sig_payload, hashlib.sha256).hexdigest()

    record = {
        "detector": name,
        "approver_id": approver_info["approver_id"],
        "role": approver_info["role"],
        "code_sha256": code_sha256,
        "dataset": dataset_tag(),
        "approved_at": ts_now,
        "signature": sig,
    }

    approvals = _load_approvals()
    approvals[name] = record
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_FILE.write_text(json.dumps(approvals, indent=2, sort_keys=True), encoding="utf-8")
    return record


def is_approved(name: str) -> bool:
    """Verify whether `name` has a valid, untampered L2 approval record matching its source file."""
    approvals = _load_approvals()
    rec = approvals.get(name)
    if not rec or rec.get("role") != "L2":
        return False
    src_path = PROPOSED_DIR / f"{name}.py"
    if not src_path.exists():
        return False
    actual_sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    if actual_sha != rec.get("code_sha256"):
        return False
    approver_id = rec.get("approver_id")
    matching_tokens = [tok for tok, inf in _L2_APPROVERS.items() if inf["approver_id"] == approver_id]
    if not matching_tokens:
        return False
    expected_payload = f"{name}|{actual_sha}|{approver_id}|L2|{rec.get('dataset')}".encode("utf-8")
    expected_sig = hmac.new(matching_tokens[0].encode("utf-8"), expected_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_sig, str(rec.get("signature", "")))


def load_approved_detector(name: str) -> type[Detector]:
    """Load a proposed detector class and register it into `REGISTRY` ONLY if it has a valid L2 approval."""
    if not is_approved(name):
        raise PermissionError(
            f"Refusing to load unapproved detector '{name}': valid L2 approval record required."
        )
    src_path = PROPOSED_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"assay.r2.proposed.{name}", src_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {src_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    det_cls: type[Detector] | None = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, Detector) and obj is not Detector:
            det_cls = obj
            break
    if det_cls is None:
        raise ValueError(f"No Detector subclass found in {src_path}")

    det_cls.name = name
    REGISTRY[name] = det_cls
    return det_cls


def load_all_approved() -> list[type[Detector]]:
    """Load all proposals that carry a valid L2 approval record."""
    approvals = _load_approvals()
    loaded = []
    for name in sorted(approvals.keys()):
        if is_approved(name):
            loaded.append(load_approved_detector(name))
    return loaded
