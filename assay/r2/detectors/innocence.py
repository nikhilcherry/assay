"""B29 -- Innocence prover: measured EXCULPATORY evidence.

Emits Findings with lr < 1 (pattern "exculpatory:<kind>") backed by an as-of
backtest on July-October 2016 over BOTH:
  (a) Alert population: confirmed_fraud closed cases vs cleared closed cases
  (b) Legit card-day population: confirmed_fraud episodes vs a fixed-seed random
      sample (>= 5,000) of unflagged legitimate Jul-Oct card-days.

Why Round 3 crashed:
In `check_recurring_card`, `same.drop_duplicates(subset=same["t"].dt.date, keep="last")`
passed a Series of `datetime.date` objects to `DataFrame.drop_duplicates(subset=...)`,
which expects column label(s), raising `KeyError: Index([2016-07-02], dtype='object')`.
We now normalise dates to `datetime64[ns]` via `.dt.floor("D")` and deduplicate using
`~same["t"].dt.floor("D").duplicated(keep="last")`.

Why Round 2 had 0/900 cleared hits:
1. In Round 2, `eval_single_case` returned `{}` whenever `has_prior_fraud` was True
   or `len(recent_1h) >= 1` or `channel != "in_person"` or `addr1_share < 0.85` or
   `D1 < 30`. Removing those over-restrictive global blockers restores the clean
   prompt definitions:
   - `recurring`: same ProductCD + amount within 2% at a regular interval (weekly/monthly)
     seen >= 3 times before on this card (Policy R7) -- or cross-card subscription batch
   - `home_region`: addr1 is the card's modal region over >= 10 prior txns
   - `known_device`: device_profile (non-generic) seen on this card >= 30 days before
   - `stable_profile`: amount within the card's own p10-p90 and product mix unchanged
   - `long_tenure_clean`: >= 90 days of history with no confirmed-fraud closed case
2. We print full diagnostics for 5 random cleared cases (`card_id`, `opened_at`,
   number of tx rows found for that card before `opened_at`, and each kind's inputs)
   and report both `LR_alert` (fraud vs cleared) and `LR_legit` (fraud vs legit card-days).

Keep rule (per lead review):
An exculpatory kind is kept for alert scoring iff its alert-population LR upper
95% CI < 0.5 (`ci_high_alert < 0.50`). If no kind passes `ci_high_alert < 0.50`
on the alert population, we report that honestly and keep the legit-population
LRs as documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from assay.detectors import is_generic_device
from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import closed_cases, tx_all

KINDS = (
    "recurring",
    "home_region",
    "known_device",
    "stable_profile",
    "long_tenure_clean",
)

# Alert-population LR upper 95% CI threshold required by lead review
KEEP_CI_HIGH_MAX = 0.50

# Jaccard co-occurrence threshold on legitimate cases for grouping correlated kinds
CORR_THRESHOLD = 0.30

# Fixed random seed and sample size for legitimate Jul-Oct card-days (Lesson 1)
LEGIT_SAMPLE_SEED = 42
LEGIT_SAMPLE_SIZE = 6000


@dataclass(frozen=True)
class KindStats:
    kind: str
    fraud_hits: int
    fraud_total: int
    cleared_hits: int
    cleared_total: int
    legit_hits: int
    legit_total: int
    lr_alert: float
    ci_low_alert: float
    ci_high_alert: float
    lr_legit: float
    ci_low_legit: float
    ci_high_legit: float
    kept: bool

    @property
    def p_fraud(self) -> float:
        return (self.fraud_hits + 0.5) / (self.fraud_total + 1.0)

    @property
    def p_cleared(self) -> float:
        return (self.cleared_hits + 0.5) / (self.cleared_total + 1.0)

    @property
    def p_legit(self) -> float:
        return (self.legit_hits + 0.5) / (self.legit_total + 1.0)

    @property
    def lr(self) -> float:
        return self.lr_alert

    @property
    def ci_low(self) -> float:
        return self.ci_low_alert

    @property
    def ci_high(self) -> float:
        return self.ci_high_alert


def _compute_lr_ci(
    pos_hits: int, pos_total: int, neg_hits: int, neg_total: int
) -> tuple[float, float, float]:
    """Return (lr, ci_low, ci_high) for P(hit|pos) / P(hit|neg) using Haldane-Anscombe
    0.5 continuity correction and normal approximation on log(RR)."""
    a = pos_hits + 0.5
    n1 = pos_total + 1.0
    c = neg_hits + 0.5
    n2 = neg_total + 1.0
    p1 = a / n1
    p2 = c / n2
    lr = float(p1 / p2)
    se = float(np.sqrt(max(1e-12, (1.0 / a) - (1.0 / n1) + (1.0 / c) - (1.0 / n2))))
    ci_low = float(np.exp(np.log(lr) - 1.96 * se))
    ci_high = float(np.exp(np.log(lr) + 1.96 * se))
    return lr, ci_low, ci_high


def _resolve_card_history(
    tx: pd.DataFrame, card_id: str, customer_id: str, before_t: pd.Timestamp
) -> pd.DataFrame:
    """Return prior transactions on `card_id` (falling back to `customer_id` if card_id
    has no records, e.g. HHF-011 C09241-K1 -> C09241-K2) strictly before `before_t`."""
    sub = tx[(tx["card_id"] == card_id) & (tx["t"] < before_t)]
    if sub.empty and customer_id:
        sub = tx[(tx["customer_id"] == customer_id) & (tx["t"] < before_t)]
    return sub


def check_recurring_card(hist: pd.DataFrame, f_row: pd.Series) -> list[int]:
    """Same ProductCD + amount within 2% at a regular interval (weekly 5.5-8.5d or
    monthly 25-35d) seen >= 3 times before on this card (Policy R7 territory)."""
    if len(hist) < 3:
        return []
    amt = float(f_row["amount"])
    prod = f_row["ProductCD"]
    tol = max(0.02 * amt, 0.05)
    same = hist[(hist["ProductCD"] == prod) & ((hist["amount"] - amt).abs() <= tol)].sort_values("t")
    if len(same) < 3:
        return []
    # Deduplicate same-day charges using datetime64[ns] floor('D') Series.duplicated
    days = same["t"].dt.floor("D")
    same_dedup = same[~days.duplicated(keep="last")]
    if len(same_dedup) < 3:
        return []
    times = list(same_dedup["t"]) + [f_row["t"]]
    gaps = np.diff([t.value // 10**9 for t in times]) / 86400.0
    regular_gaps = sum(1 for g in gaps if (5.5 <= g <= 8.5) or (25.0 <= g <= 35.0))
    if regular_gaps >= 2 and ((5.5 <= gaps[-1] <= 8.5) or (25.0 <= gaps[-1] <= 35.0)):
        return [int(x) for x in same_dedup["TransactionID"].tolist()]
    return []


def check_recurring_batch(
    small_prod_tx: pd.DataFrame | None, f_row: pd.Series
) -> tuple[list[int], list[str]]:
    """Cross-card subscription batch: several cards (>= 3 distinct cards) paying the
    same small amount (<= $30, within 1% or $0.02) at the same hour of day on the
    same date."""
    amt = float(f_row["amount"])
    if amt > 30.0 or small_prod_tx is None or small_prod_tx.empty:
        return [], []
    ft = pd.Timestamp(f_row["t"])
    tol = max(0.01 * amt, 0.02)
    f_day = ft.floor("D")
    f_hour = ft.hour
    day_win = small_prod_tx[
        (small_prod_tx["t"] <= ft)
        & (small_prod_tx["t"] >= ft - pd.Timedelta(hours=24))
        & ((small_prod_tx["amount"] - amt).abs() <= tol)
    ]
    if len(day_win) < 3:
        return [], []
    same_slot = day_win[
        (day_win["t"].dt.floor("D") == f_day)
        & (((day_win["t"].dt.hour - f_hour + 12) % 24 - 12).abs() <= 1)
    ]
    if same_slot["card_id"].nunique() >= 3:
        f_id = int(f_row["TransactionID"])
        f_card = str(f_row["card_id"])
        tx_ids = [int(x) for x in same_slot["TransactionID"].tolist() if int(x) != f_id]
        cards = sorted(set(str(c) for c in same_slot["card_id"].tolist() if str(c) != f_card))
        return tx_ids[:15], cards[:15]
    return [], []


def inspect_case_inputs(
    f_row: pd.Series,
    hist: pd.DataFrame,
    has_prior_fraud: bool,
    small_prod_tx: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a structured diagnostic dictionary of all inputs for `f_row` and `hist`
    so tests can inspect why each kind did or did not fire on any case."""
    ft = pd.Timestamp(f_row["t"])
    amt = float(f_row["amount"])
    prod = str(f_row["ProductCD"])
    f_addr = f_row["addr1"]
    f_dev = str(f_row.get("device_profile", "") or "")
    n_hist = len(hist)

    valid_addr = hist["addr1"].dropna() if not hist.empty else pd.Series(dtype=float)
    modal_addr = float(valid_addr.mode().iloc[0]) if len(valid_addr) > 0 else None
    p10 = float(hist["amount"].quantile(0.10)) if n_hist > 0 else None
    p90 = float(hist["amount"].quantile(0.90)) if n_hist > 0 else None
    prod_cnt = int((hist["ProductCD"] == prod).sum()) if n_hist > 0 else 0
    tenure_days = (
        float((ft - hist["t"].iloc[0]).total_seconds() / 86400.0) if n_hist > 0 else 0.0
    )
    old_dev_cnt = (
        int(
            len(
                hist[
                    (hist["device_profile"] == f_dev)
                    & (hist["t"] <= ft - pd.Timedelta(days=30))
                ]
            )
        )
        if (f_dev and not is_generic_device(f_dev) and not hist.empty)
        else 0
    )
    rec_card = check_recurring_card(hist, f_row)
    rec_batch, _ = check_recurring_batch(small_prod_tx, f_row)

    return {
        "flagged_txn_id": int(f_row["TransactionID"]),
        "txn_t": str(ft),
        "n_prior_txns": n_hist,
        "amount": amt,
        "ProductCD": prod,
        "channel": str(f_row["channel"]),
        "addr1": None if pd.isna(f_addr) else int(f_addr),
        "modal_addr1": None if modal_addr is None else int(modal_addr),
        "n_valid_addr1": len(valid_addr),
        "p10_amt": None if p10 is None else round(p10, 2),
        "p90_amt": None if p90 is None else round(p90, 2),
        "prod_prior_count": prod_cnt,
        "device_profile": f_dev,
        "device_is_generic": is_generic_device(f_dev),
        "old_dev_30d_count": old_dev_cnt,
        "tenure_days": round(tenure_days, 1),
        "has_prior_fraud": has_prior_fraud,
        "recurring_card_matches": len(rec_card),
        "recurring_batch_matches": len(rec_batch),
    }


def eval_single_case(
    f_row: pd.Series,
    hist: pd.DataFrame,
    has_prior_fraud: bool,
    small_prod_tx: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Single canonical evaluation function used by calibration, backtest, and `run(ctx)`.

    Implements the 5 prompt-specified exculpatory kinds:
      1. recurring: same ProductCD + amount within 2% at a regular interval (weekly/monthly)
         seen >= 3 times before on this card (Policy R7) OR cross-card subscription batch.
      2. home_region: addr1 is the card's modal region over >= 10 prior txns.
      3. known_device: device_profile (non-generic) seen on this card >= 30 days before.
      4. stable_profile: amount within the card's own p10-p90 and product mix unchanged
         over >= 10 prior txns.
      5. long_tenure_clean: >= 90 days of history with no confirmed-fraud closed case.
    """
    out: dict[str, dict[str, Any]] = {}
    ft = pd.Timestamp(f_row["t"])
    f_id = int(f_row["TransactionID"])
    card_id = str(f_row["card_id"])
    cust_id = str(f_row["customer_id"])
    amt = float(f_row["amount"])
    prod = str(f_row["ProductCD"])
    f_addr = f_row["addr1"]
    n_hist = len(hist)

    # 1. recurring (Policy R7 card-level recurring schedule OR cross-card subscription batch)
    rec_ids = check_recurring_card(hist, f_row)
    batch_ids, batch_cards = ([], [])
    if not rec_ids:
        batch_ids, batch_cards = check_recurring_batch(small_prod_tx, f_row)
    if rec_ids or batch_ids:
        ev_ids = [f_id] + (rec_ids[-10:] if rec_ids else batch_ids[:10])
        mode_desc = (
            f"card recurring schedule ({len(rec_ids)} prior regular charges; Policy R7: "
            f"when customer disputes a recurring pattern charge, recommend CREATE_CASE, "
            f"VERIFY_WITH_CUSTOMER, and WARN_CUSTOMER — do not block)"
            if rec_ids
            else f"cross-card subscription batch ({len(batch_cards) + 1} cards at ${amt:.2f} in same hour)"
        )
        out["recurring"] = {
            "what": f"Transaction {f_id} (${amt:.2f}, {prod}) matches {mode_desc}.",
            "txn_ids": ev_ids,
            "card_ids": [card_id] + batch_cards[:10],
            "customer_ids": [cust_id],
            "case_ids": [],
            "params": {
                "flagged_txn_id": f_id,
                "card_id": card_id,
                "kind": "recurring",
                "policy_rule": "R7",
            },
        }

    # 2. home_region: addr1 is the card's modal region over >= 10 prior txns
    if pd.notna(f_addr) and n_hist >= 10:
        valid_addr = hist["addr1"].dropna()
        if len(valid_addr) >= 10:
            modes = valid_addr.mode()
            if len(modes) > 0 and float(modes.iloc[0]) == float(f_addr):
                modal_txns = hist[hist["addr1"] == f_addr]["TransactionID"].astype(int).tolist()
                out["home_region"] = {
                    "what": (
                        f"Transaction {f_id} is in billing region {int(f_addr)}, which is card {card_id}'s "
                        f"modal region ({len(modal_txns)}/{len(valid_addr)} prior txns)."
                    ),
                    "txn_ids": [f_id] + modal_txns[-8:],
                    "card_ids": [card_id],
                    "customer_ids": [cust_id],
                    "case_ids": [],
                    "params": {
                        "flagged_txn_id": f_id,
                        "card_id": card_id,
                        "kind": "home_region",
                        "addr1": int(f_addr),
                    },
                }

    # 3. known_device: device_profile (non-generic) seen on this card >= 30 days before
    f_dev = str(f_row.get("device_profile", "") or "")
    if f_dev and not is_generic_device(f_dev) and not hist.empty:
        old_dev = hist[
            (hist["device_profile"] == f_dev) & (hist["t"] <= ft - pd.Timedelta(days=30))
        ]
        if not old_dev.empty:
            first_seen = old_dev["t"].min()
            days_ago = int((ft - first_seen).total_seconds() // 86400)
            dev_txns = old_dev["TransactionID"].astype(int).tolist()
            out["known_device"] = {
                "what": (
                    f"Transaction {f_id} uses specific device profile '{f_dev}' first seen on "
                    f"card {card_id} {days_ago} days earlier (>= 30d tenure)."
                ),
                "txn_ids": [f_id] + dev_txns[:5] + dev_txns[-5:],
                "card_ids": [card_id],
                "customer_ids": [cust_id],
                "case_ids": [],
                "params": {
                    "flagged_txn_id": f_id,
                    "card_id": card_id,
                    "kind": "known_device",
                    "device_profile": f_dev,
                },
            }

    # 4. stable_profile: amount within the card's own p10-p90 and product mix unchanged
    if n_hist >= 10:
        amts = hist["amount"].to_numpy()
        p10, p90 = np.quantile(amts, [0.10, 0.90])
        if p10 <= amt <= p90:
            prod_cnt = int((hist["ProductCD"] == prod).sum())
            if prod_cnt >= max(2, int(0.10 * n_hist)):
                sample_ids = hist[hist["ProductCD"] == prod]["TransactionID"].astype(int).tolist()[-8:]
                out["stable_profile"] = {
                    "what": (
                        f"Transaction {f_id} amount ${amt:.2f} lies within card {card_id}'s "
                        f"p10-p90 [${p10:.2f}, ${p90:.2f}] and product '{prod}' matches "
                        f"historical mix ({prod_cnt}/{n_hist} prior txns)."
                    ),
                    "txn_ids": [f_id] + sample_ids,
                    "card_ids": [card_id],
                    "customer_ids": [cust_id],
                    "case_ids": [],
                    "params": {
                        "flagged_txn_id": f_id,
                        "card_id": card_id,
                        "kind": "stable_profile",
                        "p10": round(float(p10), 2),
                        "p90": round(float(p90), 2),
                    },
                }

    # 5. long_tenure_clean: >= 90 days of history with no confirmed-fraud closed case
    if not hist.empty and not has_prior_fraud:
        earliest_t = hist["t"].iloc[0]
        tenure_days = (ft - earliest_t).total_seconds() / 86400.0
        if tenure_days >= 90.0:
            first_id = int(hist.iloc[0]["TransactionID"])
            last_id = int(hist.iloc[-1]["TransactionID"])
            out["long_tenure_clean"] = {
                "what": (
                    f"Card {card_id} has {tenure_days:.1f} days of clean history (>= 90d, "
                    f"{n_hist} prior txns) with zero prior confirmed-fraud closed cases."
                ),
                "txn_ids": [f_id, first_id, last_id],
                "card_ids": [card_id],
                "customer_ids": [cust_id],
                "case_ids": [],
                "params": {
                    "flagged_txn_id": f_id,
                    "card_id": card_id,
                    "kind": "long_tenure_clean",
                    "tenure_days": round(tenure_days, 1),
                },
            }

    return out


def sample_cleared_diagnostics(n: int = 5, seed: int = 42) -> list[dict[str, Any]]:
    """Return detailed diagnostics for `n` random cleared closed cases in Jul-Oct 2016:
    card_id, opened_at, number of tx rows found for that card before opened_at,
    each kind's inputs, and which kinds fired."""
    tx = tx_all()
    cc = closed_cases()
    cc_ts = pd.to_datetime(cc["opened_at"])
    cleared = cc[(cc_ts < pd.Timestamp("2016-11-01")) & (cc["outcome"] == "cleared")].copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cleared), size=min(n, len(cleared)), replace=False)
    sampled = cleared.iloc[np.sort(idx)]

    tx_sorted = tx.sort_values("t", kind="stable")
    tx_by_id = tx_sorted.set_index("TransactionID", drop=False)
    small_tx = tx_sorted[tx_sorted["amount"] <= 30.0]
    small_by_prod: dict[str, pd.DataFrame] = {
        str(k): v for k, v in small_tx.groupby("ProductCD", sort=False)
    }

    cc_work = cc.copy()
    cc_work["closed_ts"] = pd.to_datetime(cc_work["closed_at"])
    fraud_cc = cc_work[cc_work["outcome"] == "confirmed_fraud"]

    out = []
    for r in sampled.itertuples():
        opened_ts = pd.Timestamp(r.opened_at)
        card_id = str(r.card_id)
        cust_id = str(r.customer_id)
        t_list = r.txn_list if isinstance(r.txn_list, list) else []
        f_id = int(t_list[0]) if t_list else int(r.first_fraud_txn_id)
        f_row = tx_by_id.loc[f_id]
        if isinstance(f_row, pd.DataFrame):
            f_row = f_row.iloc[0]

        ft = min(pd.Timestamp(f_row["t"]), opened_ts)
        hist_before_opened = tx_sorted[(tx_sorted["card_id"] == card_id) & (tx_sorted["t"] <= opened_ts)]
        hist = tx_sorted[(tx_sorted["card_id"] == card_id) & (tx_sorted["t"] < ft)]
        prior_fraud = fraud_cc[
            ((fraud_cc["card_id"] == card_id) | (fraud_cc["customer_id"] == cust_id))
            & (fraud_cc["closed_ts"] <= opened_ts)
        ]
        has_prior_fraud = not prior_fraud.empty
        p_df = small_by_prod.get(str(f_row["ProductCD"]))

        inputs = inspect_case_inputs(f_row, hist, has_prior_fraud, small_prod_tx=p_df)
        hits = eval_single_case(f_row, hist, has_prior_fraud, small_prod_tx=p_df)
        out.append(
            {
                "case_id": str(r.case_id),
                "card_id": card_id,
                "customer_id": cust_id,
                "opened_at": str(r.opened_at),
                "tx_rows_before_opened_at": len(hist_before_opened),
                "tx_rows_strictly_before_flagged_txn": len(hist),
                "inputs": inputs,
                "kinds_fired": sorted(hits.keys()),
            }
        )
    return out


def _evaluate_cases_df(cc_sub: pd.DataFrame, tx: pd.DataFrame, all_cc: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all 5 kinds across a set of closed cases (`cc_sub`) strictly as-of
    each case's `opened_at`, calling the exact same `eval_single_case` used by `run()`."""
    tx_sorted = tx.sort_values("t", kind="stable")
    tx_by_id = tx_sorted.set_index("TransactionID", drop=False)

    card_groups: dict[str, pd.DataFrame] = {
        str(k): v for k, v in tx_sorted.groupby("card_id", sort=False)
    }
    cust_groups: dict[str, pd.DataFrame] = {
        str(k): v for k, v in tx_sorted.groupby("customer_id", sort=False)
    }
    small_tx = tx_sorted[tx_sorted["amount"] <= 30.0]
    small_by_prod: dict[str, pd.DataFrame] = {
        str(k): v for k, v in small_tx.groupby("ProductCD", sort=False)
    }

    all_cc_work = all_cc.copy()
    all_cc_work["closed_ts"] = pd.to_datetime(all_cc_work["closed_at"])
    fraud_cc = all_cc_work[all_cc_work["outcome"] == "confirmed_fraud"]
    fraud_by_card: dict[str, list[pd.Timestamp]] = {}
    fraud_by_cust: dict[str, list[pd.Timestamp]] = {}
    for r in fraud_cc.itertuples():
        fraud_by_card.setdefault(str(r.card_id), []).append(r.closed_ts)
        fraud_by_cust.setdefault(str(r.customer_id), []).append(r.closed_ts)

    cc_work = cc_sub.copy()
    cc_work["opened_ts"] = pd.to_datetime(cc_work["opened_at"])

    records = []
    for r in cc_work.itertuples():
        case_id = str(r.case_id)
        outcome = str(r.outcome)
        opened_ts = pd.Timestamp(r.opened_ts)
        card_id = str(r.card_id)
        cust_id = str(r.customer_id)

        t_list = r.txn_list if isinstance(r.txn_list, list) else []
        if pd.notna(r.first_fraud_txn_id):
            f_id = int(r.first_fraud_txn_id)
        elif t_list:
            f_id = int(t_list[0])
        else:
            continue

        if f_id not in tx_by_id.index:
            continue
        f_row = tx_by_id.loc[f_id]
        if isinstance(f_row, pd.DataFrame):
            f_row = f_row.iloc[0]

        ft = min(pd.Timestamp(f_row["t"]), opened_ts)
        c_df = card_groups.get(card_id)
        if c_df is None or c_df.empty:
            c_df = cust_groups.get(cust_id, pd.DataFrame())
        hist = c_df[c_df["t"] < ft] if not c_df.empty else pd.DataFrame()

        c_frauds = [t for t in fraud_by_card.get(card_id, []) if t <= opened_ts]
        u_frauds = [t for t in fraud_by_cust.get(cust_id, []) if t <= opened_ts]
        has_prior_fraud = bool(c_frauds or u_frauds)

        p_df = small_by_prod.get(str(f_row["ProductCD"]))
        hits = eval_single_case(
            f_row,
            hist,
            has_prior_fraud=has_prior_fraud,
            small_prod_tx=p_df,
        )
        records.append(
            {
                "case_id": case_id,
                "outcome": outcome,
                "opened_ts": opened_ts,
                "recurring": "recurring" in hits,
                "home_region": "home_region" in hits,
                "known_device": "known_device" in hits,
                "stable_profile": "stable_profile" in hits,
                "long_tenure_clean": "long_tenure_clean" in hits,
            }
        )
    return pd.DataFrame(records)


def _evaluate_legit_card_days(
    tx: pd.DataFrame, all_cc: pd.DataFrame, sample_size: int = LEGIT_SAMPLE_SIZE, seed: int = LEGIT_SAMPLE_SEED
) -> pd.DataFrame:
    """Sample `sample_size` legitimate card-days from Jul-Oct 2016 (excluding any
    transaction touched by any closed case) and evaluate `eval_single_case`
    as-of that transaction's timestamp."""
    tx_juloct = tx[tx["t"] < pd.Timestamp("2016-11-01")].sort_values("t", kind="stable")

    case_txn_ids: set[int] = set()
    for t_list in all_cc["txn_list"]:
        if isinstance(t_list, list):
            case_txn_ids.update(int(x) for x in t_list)
    for x in all_cc["first_fraud_txn_id"].dropna():
        case_txn_ids.add(int(x))

    legit_tx = tx_juloct[~tx_juloct["TransactionID"].isin(case_txn_ids)].copy()
    legit_tx["day"] = legit_tx["t"].dt.floor("D")
    card_days = legit_tx.drop_duplicates(subset=["card_id", "day"], keep="last")

    rng = np.random.default_rng(seed)
    n_choose = min(sample_size, len(card_days))
    chosen_idx = rng.choice(len(card_days), size=n_choose, replace=False)
    sampled = card_days.iloc[np.sort(chosen_idx)]

    card_groups: dict[str, pd.DataFrame] = {
        str(k): v for k, v in tx_juloct.groupby("card_id", sort=False)
    }
    small_tx = tx_juloct[tx_juloct["amount"] <= 30.0]
    small_by_prod: dict[str, pd.DataFrame] = {
        str(k): v for k, v in small_tx.groupby("ProductCD", sort=False)
    }

    all_cc_work = all_cc.copy()
    all_cc_work["closed_ts"] = pd.to_datetime(all_cc_work["closed_at"])
    fraud_cc = all_cc_work[all_cc_work["outcome"] == "confirmed_fraud"]
    fraud_by_card: dict[str, list[pd.Timestamp]] = {}
    fraud_by_cust: dict[str, list[pd.Timestamp]] = {}
    for r in fraud_cc.itertuples():
        fraud_by_card.setdefault(str(r.card_id), []).append(r.closed_ts)
        fraud_by_cust.setdefault(str(r.customer_id), []).append(r.closed_ts)

    records = []
    for _, f_row in sampled.iterrows():
        ft = pd.Timestamp(f_row["t"])
        card_id = str(f_row["card_id"])
        cust_id = str(f_row["customer_id"])
        c_df = card_groups.get(card_id, pd.DataFrame())
        hist = c_df[c_df["t"] < ft] if not c_df.empty else pd.DataFrame()

        c_frauds = [t for t in fraud_by_card.get(card_id, []) if t <= ft]
        u_frauds = [t for t in fraud_by_cust.get(cust_id, []) if t <= ft]
        has_prior_fraud = bool(c_frauds or u_frauds)

        p_df = small_by_prod.get(str(f_row["ProductCD"]))
        hits = eval_single_case(
            f_row,
            hist,
            has_prior_fraud=has_prior_fraud,
            small_prod_tx=p_df,
        )
        records.append(
            {
                "TransactionID": int(f_row["TransactionID"]),
                "card_id": card_id,
                "t": ft,
                "recurring": "recurring" in hits,
                "home_region": "home_region" in hits,
                "known_device": "known_device" in hits,
                "stable_profile": "stable_profile" in hits,
                "long_tenure_clean": "long_tenure_clean" in hits,
            }
        )
    return pd.DataFrame(records)


@lru_cache(maxsize=1)
def calibrated_innocence_stats() -> tuple[dict[str, KindStats], pd.DataFrame, pd.DataFrame]:
    """Run the Jul-Oct as-of backtest on BOTH:
      (a) closed-case alerts (confirmed_fraud vs cleared) in Jul-Oct 2016
      (b) confirmed_fraud episodes vs 6,000 random legitimate card-days in Jul-Oct 2016.
    Returns (stats_by_kind, cooccurrence_jaccard_df, alert_eval_df).
    """
    tx = tx_all()
    cc = closed_cases()
    cc_ts = pd.to_datetime(cc["opened_at"])
    cc_juloct = cc[cc_ts < pd.Timestamp("2016-11-01")].copy()

    alert_df = _evaluate_cases_df(cc_juloct, tx, cc)
    legit_days_df = _evaluate_legit_card_days(
        tx, cc, sample_size=LEGIT_SAMPLE_SIZE, seed=LEGIT_SAMPLE_SEED
    )

    fraud_df = alert_df[alert_df["outcome"] == "confirmed_fraud"]
    cleared_df = alert_df[alert_df["outcome"] == "cleared"]

    n_fraud = len(fraud_df)
    n_cleared = len(cleared_df)
    n_legit = len(legit_days_df)

    stats: dict[str, KindStats] = {}
    for k in KINDS:
        fh = int(fraud_df[k].sum())
        ch = int(cleared_df[k].sum())
        lh = int(legit_days_df[k].sum())

        lr_alert, ci_l_alert, ci_h_alert = _compute_lr_ci(fh, n_fraud, ch, n_cleared)
        lr_legit, ci_l_legit, ci_h_legit = _compute_lr_ci(fh, n_fraud, lh, n_legit)

        # Lead criterion: an exculpatory kind is kept when its alert-population LR upper CI < 0.5
        kept = bool(ci_h_alert < KEEP_CI_HIGH_MAX)
        stats[k] = KindStats(
            kind=k,
            fraud_hits=fh,
            fraud_total=n_fraud,
            cleared_hits=ch,
            cleared_total=n_cleared,
            legit_hits=lh,
            legit_total=n_legit,
            lr_alert=round(lr_alert, 4),
            ci_low_alert=round(ci_l_alert, 4),
            ci_high_alert=round(ci_h_alert, 4),
            lr_legit=round(lr_legit, 4),
            ci_low_legit=round(ci_l_legit, 4),
            ci_high_legit=round(ci_h_legit, 4),
            kept=kept,
        )

    # Pairwise Jaccard co-occurrence among legitimate card-days + cleared cases
    combined_legit = pd.concat(
        [cleared_df[list(KINDS)], legit_days_df[list(KINDS)]], ignore_index=True
    )
    cooc = pd.DataFrame(index=KINDS, columns=KINDS, dtype=float)
    for k1 in KINDS:
        s1 = combined_legit[k1].to_numpy(dtype=bool)
        for k2 in KINDS:
            s2 = combined_legit[k2].to_numpy(dtype=bool)
            inter = int(np.logical_and(s1, s2).sum())
            union = int(np.logical_or(s1, s2).sum())
            cooc.loc[k1, k2] = round(inter / union, 4) if union > 0 else 0.0

    return stats, cooc, alert_df


def correlated_groups() -> list[set[str]]:
    """Return connected components of kinds whose pairwise Jaccard co-occurrence
    on legit cases is >= CORR_THRESHOLD."""
    _, cooc, _ = calibrated_innocence_stats()
    adj: dict[str, set[str]] = {k: {k} for k in KINDS}
    for i, k1 in enumerate(KINDS):
        for k2 in KINDS[i + 1 :]:
            if float(cooc.loc[k1, k2]) >= CORR_THRESHOLD:
                adj[k1].add(k2)
                adj[k2].add(k1)
    visited: set[str] = set()
    groups: list[set[str]] = []
    for k in KINDS:
        if k not in visited:
            comp: set[str] = set()
            stack = [k]
            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    comp.add(cur)
                    stack.extend(adj[cur] - visited)
            groups.append(comp)
    return groups


def combined_lr(findings: list[Finding]) -> float:
    """Combine exculpatory findings without double-counting correlated kinds.
    Within each correlated group of kinds, only the single strongest finding
    (minimum LR) is multiplied."""
    exculp = [f for f in findings if f.pattern.startswith("exculpatory:") and f.lr < 1.0]
    if not exculp:
        return 1.0

    groups = correlated_groups()
    kind_to_group: dict[str, int] = {}
    for idx, g in enumerate(groups):
        for k in g:
            kind_to_group[k] = idx

    best_per_group: dict[int, float] = {}
    unmapped_lrs: list[float] = []

    for f in exculp:
        kind = f.pattern.split(":", 1)[1] if ":" in f.pattern else f.pattern
        g_idx = kind_to_group.get(kind)
        if g_idx is None:
            unmapped_lrs.append(float(f.lr))
        else:
            if g_idx not in best_per_group or f.lr < best_per_group[g_idx]:
                best_per_group[g_idx] = float(f.lr)

    prod = 1.0
    for lr in best_per_group.values():
        prod *= lr
    for lr in unmapped_lrs:
        prod *= lr
    return float(prod)


class InnocenceEvidence(Detector):
    """Emits zero or more exculpatory Findings (lr < 1) per case.
    Uses `kept` kinds (where alert-population LR upper 95% CI < 0.5).
    Also provides `run_all_kinds(ctx)` so tests and analysts can inspect which
    raw exculpatory signals are present on a case even when `kept` is empty."""

    name = "innocence"
    pattern = "exculpatory"

    def run_all_kinds(self, ctx: CaseCtx) -> list[Finding]:
        """Evaluate and return findings for all kinds that fire on `ctx` (with their
        measured `lr_alert`), regardless of whether `st.kept` is True."""
        stats, _, _ = calibrated_innocence_stats()
        tx_asof = ctx.tx[ctx.tx["t"] <= ctx.opened_at]
        matches = tx_asof[tx_asof["TransactionID"] == int(ctx.flagged_txn_id)]
        if matches.empty:
            return []
        f_row = matches.iloc[0]
        ft = min(pd.Timestamp(f_row["t"]), ctx.opened_at)

        hist = _resolve_card_history(tx_asof, ctx.card_id, ctx.customer_id, ft)
        cc_prior = ctx.cc
        if "closed_at" in cc_prior.columns:
            cc_prior = cc_prior[pd.to_datetime(cc_prior["closed_at"]) <= ctx.opened_at]

        prior_fraud = cc_prior[
            ((cc_prior["card_id"] == ctx.card_id) | (cc_prior["customer_id"] == ctx.customer_id))
            & (cc_prior["outcome"] == "confirmed_fraud")
        ]
        has_prior_fraud = not prior_fraud.empty

        small_prod_tx: pd.DataFrame | None = None
        if float(f_row["amount"]) <= 30.0:
            small_prod_tx = tx_asof[
                (tx_asof["ProductCD"] == f_row["ProductCD"]) & (tx_asof["amount"] <= 30.0)
            ]

        raw_hits = eval_single_case(
            f_row,
            hist,
            has_prior_fraud=has_prior_fraud,
            small_prod_tx=small_prod_tx,
        )
        findings: list[Finding] = []
        for kind in KINDS:
            if kind not in raw_hits:
                continue
            st = stats.get(kind)
            if st is None:
                continue
            info = raw_hits[kind]
            att = Attestation(
                what=info["what"],
                evidence=Evidence(
                    txn_ids=info["txn_ids"],
                    card_ids=info["card_ids"],
                    customer_ids=info["customer_ids"],
                    case_ids=info["case_ids"],
                ),
                query=f"innocence.{kind}",
                params=info["params"],
            )
            findings.append(
                Finding(
                    detector=self.name,
                    pattern=f"exculpatory:{kind}",
                    claim=att,
                    lr=st.lr_alert,
                    affected_txn_ids=[int(ctx.flagged_txn_id)],
                    connected_card_ids=[c for c in info["card_ids"] if c != ctx.card_id],
                )
            )
        return findings

    def run(self, ctx: CaseCtx) -> list[Finding]:
        stats, _, _ = calibrated_innocence_stats()
        all_f = self.run_all_kinds(ctx)
        return [
            f
            for f in all_f
            if stats.get(f.pattern.split(":", 1)[1]) is not None
            and stats[f.pattern.split(":", 1)[1]].kept
        ]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        stats, _, _ = calibrated_innocence_stats()
        kept_kinds = [k for k, st in stats.items() if st.kept]
        df = _evaluate_cases_df(cc, tx, cc)
        if df.empty or not kept_kinds:
            return Backtest(
                detector=self.name,
                n_cases=len(df),
                caught=[],
                missed=df[df["outcome"] == "confirmed_fraud"]["case_id"].tolist()
                if not df.empty
                else [],
                legit_acted_against=[],
                lr_measured=1.0,
            )

        any_kept = df[kept_kinds].any(axis=1)
        fraud_mask = df["outcome"] == "confirmed_fraud"
        legit_mask = df["outcome"] == "cleared"

        caught = df[fraud_mask & any_kept]["case_id"].tolist()
        missed = df[fraud_mask & ~any_kept]["case_id"].tolist()
        legit_hit = df[legit_mask & any_kept]["case_id"].tolist()

        lr, _, _ = _compute_lr_ci(
            len(caught),
            int(fraud_mask.sum()),
            len(legit_hit),
            int(legit_mask.sum()),
        )
        return Backtest(
            detector=self.name,
            n_cases=len(df),
            caught=caught,
            missed=missed,
            legit_acted_against=legit_hit,
            lr_measured=round(lr, 4),
        )
