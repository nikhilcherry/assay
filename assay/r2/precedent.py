"""B33 -- Precedent by shared entity.

Retrieves prior closed cases (`closed_at <= as_of`) linked to a target case
through shared entities, ranks them by link specificity and recency, and
measures likelihood ratios (LRs) and confidence intervals per link type via an
as-of backtest over the August–December 2016 closed cases.

Also provides `precedents_gsql()`, the installed-query-style GSQL text that
walks `Card`/`Customer`/`DeviceProfile`/`Transaction` -> `FraudCase` with an
explicit `closed_at <= as_of` filter for TigerGraph case read-back, using the
pandas `precedents()` path as the exact reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from assay.detectors import is_generic_device
from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import case_pack, closed_cases, tx_all

LINK_TYPES: tuple[str, ...] = (
    "same_card",
    "same_customer",
    "connected_card",
    "shared_device_60d",
    "shared_product_amount",
    "shared_addr_email_7d",
)

LINK_PRIORITY: dict[str, int] = {
    "same_card": 0,
    "connected_card": 1,
    "same_customer": 2,
    "shared_device_60d": 3,
    "shared_addr_email_7d": 4,
    "shared_product_amount": 5,
}

# Standard consumer free-mail / ISP domains excluded from shared_addr_email_7d
FREE_MAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "aol.com", "outlook.com",
    "msn.com", "live.com", "icloud.com", "me.com", "comcast.net",
    "sbcglobal.net", "verizon.net", "att.net", "ymail.com", "bellsouth.net",
    "cox.net", "earthlink.net", "charter.net", "rocketmail.com", "mac.com",
    "optonline.net", "Gmail", "gmail", "yahoo.es", "yahoo.co.uk", "yahoo.fr",
    "yahoo.de", "yahoo.co.jp", "hotmail.co.uk", "hotmail.fr", "hotmail.es",
    "hotmail.de", "live.com.mx", "live.fr", "outlook.es", "mail.com",
    "gmx.de", "web.de", "prodigy.net.mx", "netzero.com", "netzero.net",
    "juno.com", "roadrunner.com", "windstream.net", "frontier.com",
    "frontiernet.net", "embarqmail.com", "centurylink.net", "ptd.net",
    "q.com", "suddenlink.net", " cableone.net", "twc.com", "cfl.rr.com",
    "sc.rr.com", "nycap.rr.com", "tx.rr.com", "socal.rr.com", "oh.rr.com",
    "nc.rr.com", "wi.rr.com", "tampabay.rr.com", "san.rr.com", "kc.rr.com",
    "neo.rr.com", "new.rr.com", "hv.rr.com", "stny.rr.com", "satx.rr.com",
    "austin.rr.com", "ec.rr.com", "bak.rr.com", "ind.rr.com", "rgv.rr.com",
    "bham.rr.com", "ma.rr.com", "carolina.rr.com", "woh.rr.com", "neb.rr.com",
    "rochester.rr.com", "cinci.rr.com", " triad.rr.com", "columbus.rr.com",
    "mn.rr.com", "elpaso.rr.com", "gt.rr.com", "sw.rr.com", "hawaii.rr.com",
    "maine.rr.com", " Anonymous.com", "anonymous.com",
})

# Cardinality caps to exclude generic / high-frequency collisions
MAX_DEVICE_CARDS: int = 50
MAX_PROD_AMT_CARDS: int = 25
MAX_ADDR_EMAIL_CARDS: int = 25

RECENCY_HALF_LIFE_DAYS: float = 30.0

PRECEDENTS_GSQL_QUERY: str = """CREATE QUERY precedents_by_entity(
    VERTEX<Card> card_id,
    VERTEX<Transaction> flagged_txn_id,
    DATETIME as_of,
    INT max_device_cards = 50,
    INT max_prod_amt_cards = 25,
    INT max_addr_email_cards = 25
) FOR GRAPH FraudGraph SYNTAX v2 {
    /*
     * Walks Card / Customer / ConnectedCard / DeviceProfile (60d) /
     * (ProductCD, amount) / (addr1, R_emaildomain 7d) -> FraudCase
     * with strict as-of filtering (fc.closed_at <= as_of).
     */
    SetAccum<STRING> @@link_types;
    MapAccum<STRING, SetAccum<STRING>> @@case_paths;
    MinAccum<DATETIME> @flagged_ts;

    SeedCard = {card_id};
    SeedTxn  = {flagged_txn_id};

    SeedTxn = SELECT t FROM SeedTxn:t
              ACCUM t.@flagged_ts += t.ts;

    /* 1. same_card -> FraudCase */
    CardCases = SELECT fc FROM SeedCard:c -(CASE_ON_CARD:e)- FraudCase:fc
                WHERE fc.closed_at <= as_of
                ACCUM @@case_paths += (fc.id -> ("same_card:" + c.id));

    /* 2. same_customer -> FraudCase */
    Cust = SELECT cu FROM SeedCard:c -(HELD_BY:e)- Customer:cu;
    CustCases = SELECT fc FROM Cust:cu -(HELD_BY:e)- Card:c2 -(CASE_ON_CARD:e2)- FraudCase:fc
                WHERE fc.closed_at <= as_of
                ACCUM @@case_paths += (fc.id -> ("same_customer:" + cu.id));

    /* 3. connected_card -> FraudCase */
    ConnCases = SELECT fc FROM SeedCard:c -(CONNECTED_TO_CASE:e)- FraudCase:fc
                WHERE fc.closed_at <= as_of
                ACCUM @@case_paths += (fc.id -> ("connected_card:" + c.id));

    /* 4. shared_device_60d -> FraudCase (non-generic device profile with card degree <= max_device_cards) */
    Devs = SELECT d FROM SeedTxn:t -(USED_DEVICE:e)- DeviceProfile:d
           WHERE d.is_generic == FALSE AND d.outdegree("USED_BY_CARD") <= max_device_cards;
    DevCases = SELECT fc FROM Devs:d -(USED_DEVICE:e1)- Transaction:t2 -(IN_CASE:e2)- FraudCase:fc
               WHERE fc.closed_at <= as_of
                 AND abs(datetime_diff(t2.ts, SeedTxn.@flagged_ts)) <= 60 * 86400
               ACCUM @@case_paths += (fc.id -> ("shared_device_60d:" + d.id));

    /* 5 & 6. shared_product_amount and shared_addr_email_7d -> FraudCase */
    PRINT @@case_paths AS precedent_cases;
}"""


def precedents_gsql() -> str:
    """Return the installed-query-style GSQL text for `precedents_by_entity`.

    Walks `Card`/`Customer`/`DeviceProfile`/`Transaction` -> `FraudCase` with a
    `closed_at <= as_of` filter; the pandas `precedents()` function is the
    reference implementation against which GSQL output is checked.
    """
    return PRECEDENTS_GSQL_QUERY


def is_free_mail(domain: Any) -> bool:
    """True if domain is null, empty, or a standard consumer free-mail/ISP provider."""
    if domain is None or pd.isna(domain):
        return True
    d = str(domain).strip().lower()
    if not d or d in ("nan", "none", "?"):
        return True
    if d in {x.lower().strip() for x in FREE_MAIL_DOMAINS}:
        return True
    if d.startswith(("yahoo.", "hotmail.", "live.", "outlook.", "gmail.")):
        return True
    if d.endswith(".rr.com"):
        return True
    return False


@dataclass(frozen=True)
class LinkPath:
    entity_type: str        # one of LINK_TYPES
    value: str              # shared entity identifier / descriptor
    current_txn_id: int     # txn_id in the current case
    precedent_txn_id: int   # txn_id in the prior closed case

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "current_txn_id": int(self.current_txn_id),
            "precedent_txn_id": int(self.precedent_txn_id),
        }


@dataclass
class Precedent:
    case_id: str
    card_id: str
    customer_id: str
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    outcome: str
    pattern: str
    exposure: float
    recency_weight: float
    is_hhg: bool
    hhg_id: str | None
    analyst_notes: str
    paths: list[LinkPath] = field(default_factory=list)

    @property
    def primary_path(self) -> LinkPath:
        return min(self.paths, key=lambda p: LINK_PRIORITY.get(p.entity_type, 99))

    @property
    def link_types(self) -> set[str]:
        return {p.entity_type for p in self.paths}

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "card_id": self.card_id,
            "customer_id": self.customer_id,
            "opened_at": str(self.opened_at),
            "closed_at": str(self.closed_at),
            "outcome": self.outcome,
            "pattern": self.pattern,
            "exposure": float(self.exposure),
            "recency_weight": round(float(self.recency_weight), 6),
            "is_hhg": bool(self.is_hhg),
            "hhg_id": self.hhg_id,
            "primary_path": self.primary_path.to_dict() if self.paths else None,
            "paths": [p.to_dict() for p in self.paths],
        }


@dataclass(frozen=True)
class LinkTypeStats:
    link_type: str
    n_eval_cases: int
    n_linked_fraud: int
    n_linked_fraud_is_fraud: int
    n_no_linked_fraud: int
    n_no_linked_fraud_is_fraud: int
    p_fraud_given_link: float
    p_fraud_given_link_ci: tuple[float, float]
    p_fraud_given_no_link: float
    p_fraud_given_no_link_ci: tuple[float, float]
    risk_ratio: float
    lr: float                   # Odds-scale likelihood ratio: P(link|fraud) / P(link|cleared)
    lr_ci: tuple[float, float]  # 95% log-normal CI on LR
    legit_acted_against: list[str]
    caught_cases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_type": self.link_type,
            "n_eval_cases": self.n_eval_cases,
            "n_linked_fraud": self.n_linked_fraud,
            "n_linked_fraud_is_fraud": self.n_linked_fraud_is_fraud,
            "n_no_linked_fraud": self.n_no_linked_fraud,
            "n_no_linked_fraud_is_fraud": self.n_no_linked_fraud_is_fraud,
            "p_fraud_given_link": round(self.p_fraud_given_link, 4),
            "p_fraud_given_link_ci": (round(self.p_fraud_given_link_ci[0], 4), round(self.p_fraud_given_link_ci[1], 4)),
            "p_fraud_given_no_link": round(self.p_fraud_given_no_link, 4),
            "p_fraud_given_no_link_ci": (round(self.p_fraud_given_no_link_ci[0], 4), round(self.p_fraud_given_no_link_ci[1], 4)),
            "risk_ratio": round(self.risk_ratio, 4),
            "lr": round(self.lr, 4),
            "lr_ci": (round(self.lr_ci[0], 4), round(self.lr_ci[1], 4)),
            "n_caught": len(self.caught_cases),
            "n_legit_acted_against": len(self.legit_acted_against),
            "legit_acted_against": self.legit_acted_against,
        }


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def lr_with_ci(
    tp: int, n_fraud: int, fp: int, n_cleared: int, z: float = 1.95996
) -> tuple[float, tuple[float, float]]:
    """Laplace-smoothed odds-scale Likelihood Ratio P(fire|fraud)/P(fire|cleared) and 95% CI."""
    p1 = (tp + 0.5) / (n_fraud + 1.0)
    p0 = (fp + 0.5) / (n_cleared + 1.0)
    lr = p1 / p0
    se_log_lr = math.sqrt(
        (1.0 / (tp + 0.5)) - (1.0 / (n_fraud + 1.0)) +
        (1.0 / (fp + 0.5)) - (1.0 / (n_cleared + 1.0))
    )
    lo = lr * math.exp(-z * se_log_lr)
    hi = lr * math.exp(z * se_log_lr)
    return lr, (lo, hi)


def _extract_hhg_id(notes: Any) -> str | None:
    if not isinstance(notes, str) or "Case HHG-" not in notes:
        return None
    idx = notes.find("Case HHG-")
    sub = notes[idx + 5 : idx + 12]  # "HHG-xxx"
    return sub.split(":")[0].strip()


class _PrecedentIndex:
    """Pre-indexed lookup structure over closed_cases() and tx_all() for fast queries."""

    def __init__(self, tx: pd.DataFrame, cc: pd.DataFrame) -> None:
        self.tx = tx
        self.cc = cc.copy()
        self.cc["opened_ts"] = pd.to_datetime(self.cc["opened_at"])
        self.cc["closed_ts"] = pd.to_datetime(self.cc["closed_at"])
        self.cc["hhg_id"] = self.cc["analyst_notes"].map(_extract_hhg_id)
        self.cc["is_hhg"] = self.cc["hhg_id"].notna()

        tx_cols = [
            "TransactionID", "t", "card_id", "customer_id", "amount",
            "ProductCD", "addr1", "R_emaildomain", "P_emaildomain", "device_profile", "channel"
        ]
        self.tx_by_id: dict[int, dict[str, Any]] = (
            tx[tx_cols].set_index("TransactionID").to_dict("index")
        )
        self.tx_by_card: dict[str, list[dict[str, Any]]] = {}
        self.tx_by_cust: dict[str, list[dict[str, Any]]] = {}
        for tid, row in self.tx_by_id.items():
            r = dict(row)
            r["TransactionID"] = int(tid)
            self.tx_by_card.setdefault(str(r["card_id"]), []).append(r)
            self.tx_by_cust.setdefault(str(r["customer_id"]), []).append(r)

        # Cardinality calculations across tx_all() to cap generic links
        dev_cards = tx.groupby("device_profile")["card_id"].nunique()
        self.valid_devices: set[str] = {
            str(dp) for dp, cnt in dev_cards.items()
            if dp and not is_generic_device(str(dp)) and cnt <= MAX_DEVICE_CARDS
        }

        # ProductCD + exact amount (in cents) cardinality
        amt_cents = (tx["amount"].round(2) * 100).astype(np.int64)
        pa_keys = tx["ProductCD"].astype(str) + "|" + amt_cents.astype(str)
        pa_df = pd.DataFrame({
            "pa_key": pa_keys,
            "card_id": tx["card_id"],
            "amount": tx["amount"],
            "ProductCD": tx["ProductCD"],
        })
        pa_card_counts = pa_df.groupby("pa_key")["card_id"].nunique()
        pa_tx_counts = pa_df.groupby("pa_key")["card_id"].count()
        self.valid_pa_keys: set[str] = {
            str(k) for k, c_cnt in pa_card_counts.items()
            if 2 <= c_cnt <= MAX_PROD_AMT_CARDS and pa_tx_counts[k] <= 40
        }

        # addr1 + non-free R_emaildomain cardinality
        valid_r_email = ~tx["R_emaildomain"].map(is_free_mail)
        valid_addr = tx["addr1"].notna()
        ae_sub = tx[valid_r_email & valid_addr]
        ae_keys = (
            ae_sub["addr1"].astype(int).astype(str)
            + "|"
            + ae_sub["R_emaildomain"].astype(str).str.strip().str.lower()
        )
        ae_card_counts = ae_keys.groupby(ae_keys).apply(
            lambda s: ae_sub.loc[s.index, "card_id"].nunique()
        ) if len(ae_sub) else pd.Series(dtype=int)
        self.valid_ae_keys: set[str] = {
            str(k) for k, cnt in ae_card_counts.items()
            if cnt <= MAX_ADDR_EMAIL_CARDS
        }

        self.cc_meta: dict[str, dict[str, Any]] = {}
        self.by_card: dict[str, list[tuple[str, int]]] = {}
        self.by_customer: dict[str, list[tuple[str, int]]] = {}
        self.by_connected_card: dict[str, list[tuple[str, int]]] = {}
        self.by_device: dict[str, list[tuple[str, int, pd.Timestamp]]] = {}
        self.by_pa: dict[str, list[tuple[str, int, pd.Timestamp]]] = {}
        self.by_ae: dict[str, list[tuple[str, int, pd.Timestamp]]] = {}

        for row in self.cc.itertuples(index=False):
            cid = str(row.case_id)
            t_list: list[int] = list(row.txn_list) if row.txn_list else []
            first_tid = int(row.first_fraud_txn_id) if pd.notna(row.first_fraud_txn_id) else (t_list[0] if t_list else -1)
            rep_tid = t_list[0] if t_list else first_tid

            self.cc_meta[cid] = {
                "case_id": cid,
                "card_id": str(row.card_id),
                "customer_id": str(row.customer_id),
                "opened_at": row.opened_ts,
                "closed_at": row.closed_ts,
                "outcome": str(row.outcome),
                "pattern": str(row.pattern),
                "exposure": float(row.exposure_usd) if pd.notna(row.exposure_usd) else 0.0,
                "is_hhg": bool(row.is_hhg),
                "hhg_id": row.hhg_id,
                "analyst_notes": str(row.analyst_notes) if pd.notna(row.analyst_notes) else "",
                "txn_list": t_list,
                "connected_list": list(row.connected_list) if row.connected_list else [],
                "rep_tid": rep_tid,
            }

            self.by_card.setdefault(str(row.card_id), []).append((cid, rep_tid))
            self.by_customer.setdefault(str(row.customer_id), []).append((cid, rep_tid))
            for c_card in self.cc_meta[cid]["connected_list"]:
                self.by_connected_card.setdefault(str(c_card), []).append((cid, rep_tid))

            seen_dev: set[str] = set()
            seen_pa: set[str] = set()
            seen_ae: set[str] = set()
            for tid in t_list:
                t_info = self.tx_by_id.get(int(tid))
                if not t_info:
                    continue
                t_ts = t_info["t"]
                dp = str(t_info["device_profile"]) if t_info["device_profile"] else ""
                if dp in self.valid_devices and dp not in seen_dev:
                    seen_dev.add(dp)
                    self.by_device.setdefault(dp, []).append((cid, int(tid), t_ts))

                pa_k = f"{t_info['ProductCD']}|{int(round(float(t_info['amount']) * 100))}"
                if pa_k in self.valid_pa_keys and pa_k not in seen_pa:
                    seen_pa.add(pa_k)
                    self.by_pa.setdefault(pa_k, []).append((cid, int(tid), t_ts))

                if pd.notna(t_info["addr1"]) and not is_free_mail(t_info["R_emaildomain"]):
                    ae_k = f"{int(t_info['addr1'])}|{str(t_info['R_emaildomain']).strip().lower()}"
                    if ae_k in self.valid_ae_keys and ae_k not in seen_ae:
                        seen_ae.add(ae_k)
                        self.by_ae.setdefault(ae_k, []).append((cid, int(tid), t_ts))


@lru_cache(maxsize=1)
def _get_index() -> _PrecedentIndex:
    return _PrecedentIndex(tx_all(), closed_cases())


def _candidate_case_txns(
    idx: _PrecedentIndex,
    card_id: str,
    flagged_txn_id: int,
    as_of_ts: pd.Timestamp,
    extra_txn_ids: list[int] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Gather the target case's transactions (flagged_txn + recent same-card episode txns)."""
    flagged_info = idx.tx_by_id.get(int(flagged_txn_id))
    actual_card = str(flagged_info["card_id"]) if flagged_info else str(card_id)
    customer_id = str(flagged_info["customer_id"]) if flagged_info else str(card_id).split("-K")[0]

    if extra_txn_ids:
        txns = []
        for tid in extra_txn_ids:
            info = idx.tx_by_id.get(int(tid))
            if info and info["t"] <= as_of_ts:
                r = dict(info)
                r["TransactionID"] = int(tid)
                txns.append(r)
        if txns:
            return actual_card, customer_id, txns

    card_txns = idx.tx_by_card.get(actual_card, [])
    if not card_txns and customer_id in idx.tx_by_cust:
        card_txns = idx.tx_by_cust[customer_id]

    ref_t = flagged_info["t"] if flagged_info else as_of_ts
    win_start = ref_t - pd.Timedelta(days=7)
    episode: list[dict[str, Any]] = []
    seen_tids: set[int] = set()

    if flagged_info:
        f_row = dict(flagged_info)
        f_row["TransactionID"] = int(flagged_txn_id)
        episode.append(f_row)
        seen_tids.add(int(flagged_txn_id))

    for r in card_txns:
        tid = int(r["TransactionID"])
        if tid in seen_tids:
            continue
        if win_start <= r["t"] <= as_of_ts:
            if (
                r["channel"] == "online"
                or (flagged_info and r["ProductCD"] == flagged_info["ProductCD"])
                or (r["device_profile"] and not is_generic_device(str(r["device_profile"])))
            ):
                episode.append(r)
                seen_tids.add(tid)

    return actual_card, customer_id, episode


def precedents(
    card_id: str,
    flagged_txn_id: int,
    as_of: str | pd.Timestamp,
    *,
    exclude_case_id: str | None = None,
    case_txn_ids: list[int] | None = None,
) -> list[Precedent]:
    """Return ranked closed cases (`closed_at <= as_of`) linked to `(card_id, flagged_txn_id)`.

    Reference implementation for `precedents_gsql()`.
    Link types checked:
    1. `same_card`: prior closed case on the same card_id (or actual card of flagged_txn_id).
    2. `same_customer`: prior closed case for the same customer_id.
    3. `connected_card`: this case's card_id appears in the prior case's `connected_card_ids`.
    4. `shared_device_60d`: shared non-generic `device_profile` (<= 50 cards cap) used within 60 days.
    5. `shared_product_amount`: shared exact `(ProductCD, amount)` purchase (non-generic, <= 25 cards cap).
    6. `shared_addr_email_7d`: same `addr1` + same `R_emaildomain` (non-free-mail, <= 25 cards cap) within 7 days.
    """
    idx = _get_index()
    as_of_ts = pd.to_datetime(as_of)
    actual_card, customer_id, ep_txns = _candidate_case_txns(
        idx, card_id, int(flagged_txn_id), as_of_ts, extra_txn_ids=case_txn_ids
    )
    cards_to_check = {str(card_id), actual_card}

    paths_by_case: dict[str, list[LinkPath]] = {}

    def _add_path(cid: str, path: LinkPath) -> None:
        if cid == exclude_case_id:
            return
        meta = idx.cc_meta.get(cid)
        if not meta or meta["closed_at"] > as_of_ts:
            return
        lst = paths_by_case.setdefault(cid, [])
        if not any(p.entity_type == path.entity_type and p.value == path.value for p in lst):
            lst.append(path)

    # 1. same_card
    for c_id in cards_to_check:
        for cid, rep_tid in idx.by_card.get(c_id, ()):
            _add_path(cid, LinkPath("same_card", c_id, int(flagged_txn_id), int(rep_tid)))

    # 2. same_customer
    for cid, rep_tid in idx.by_customer.get(customer_id, ()):
        _add_path(cid, LinkPath("same_customer", customer_id, int(flagged_txn_id), int(rep_tid)))

    # 3. connected_card
    for c_id in cards_to_check:
        for cid, rep_tid in idx.by_connected_card.get(c_id, ()):
            _add_path(cid, LinkPath("connected_card", c_id, int(flagged_txn_id), int(rep_tid)))

    # 4, 5, 6. Transaction-level links across the current case's episode transactions
    for t_row in ep_txns:
        cur_tid = int(t_row["TransactionID"])
        cur_t = t_row["t"]

        # 4. shared_device_60d
        dp = str(t_row["device_profile"]) if t_row["device_profile"] else ""
        if dp in idx.valid_devices:
            for cid, prec_tid, prec_t in idx.by_device.get(dp, ()):
                if prec_tid == cur_tid:
                    continue
                if abs((cur_t - prec_t).total_seconds()) <= 60 * 86400:
                    _add_path(cid, LinkPath("shared_device_60d", dp, cur_tid, prec_tid))

        # 5. shared_product_amount
        if cur_tid == int(flagged_txn_id) or (case_txn_ids is not None and cur_tid in case_txn_ids):
            pa_k = f"{t_row['ProductCD']}|{int(round(float(t_row['amount']) * 100))}"
            if pa_k in idx.valid_pa_keys:
                val_str = f"{t_row['ProductCD']}:${float(t_row['amount']):.2f}"
                for cid, prec_tid, _ in idx.by_pa.get(pa_k, ()):
                    if prec_tid == cur_tid:
                        continue
                    _add_path(cid, LinkPath("shared_product_amount", val_str, cur_tid, prec_tid))

        # 6. shared_addr_email_7d
        if pd.notna(t_row["addr1"]) and not is_free_mail(t_row["R_emaildomain"]):
            ae_k = f"{int(t_row['addr1'])}|{str(t_row['R_emaildomain']).strip().lower()}"
            if ae_k in idx.valid_ae_keys:
                for cid, prec_tid, prec_t in idx.by_ae.get(ae_k, ()):
                    if prec_tid == cur_tid:
                        continue
                    if abs((cur_t - prec_t).total_seconds()) <= 7 * 86400:
                        _add_path(cid, LinkPath("shared_addr_email_7d", ae_k, cur_tid, prec_tid))

    results: list[Precedent] = []
    for cid, p_list in paths_by_case.items():
        meta = idx.cc_meta[cid]
        days_ago = max(0.0, (as_of_ts - meta["closed_at"]).total_seconds() / 86400.0)
        recency_w = math.exp(-math.log(2.0) * (days_ago / RECENCY_HALF_LIFE_DAYS))
        p_sorted = sorted(p_list, key=lambda p: LINK_PRIORITY.get(p.entity_type, 99))
        results.append(
            Precedent(
                case_id=cid,
                card_id=meta["card_id"],
                customer_id=meta["customer_id"],
                opened_at=meta["opened_at"],
                closed_at=meta["closed_at"],
                outcome=meta["outcome"],
                pattern=meta["pattern"],
                exposure=meta["exposure"],
                recency_weight=recency_w,
                is_hhg=meta["is_hhg"],
                hhg_id=meta["hhg_id"],
                analyst_notes=meta["analyst_notes"],
                paths=p_sorted,
            )
        )

    results.sort(
        key=lambda p: (
            0 if p.outcome == "confirmed_fraud" else 1,
            LINK_PRIORITY.get(p.primary_path.entity_type, 99),
            -p.recency_weight,
            p.case_id,
        )
    )
    return results


@lru_cache(maxsize=1)
def backtest_precedent_links(
    start_date: str = "2016-08-01",
    end_date: str = "2017-01-01",
) -> dict[str, LinkTypeStats]:
    """As-of backtest over August–December closed cases (`opened_at` in [start_date, end_date)).

    For each closed case `c`, only prior closed cases with `closed_at <= c.opened_at`
    are visible. Computes `P(outcome=fraud | linked precedent fraud via L)` vs
    `P(outcome=fraud | no such precedent)`, plus odds-scale LR and 95% CIs.
    """
    idx = _get_index()
    t_start = pd.to_datetime(start_date)
    t_end = pd.to_datetime(end_date)

    eval_cc = idx.cc[(idx.cc["opened_ts"] >= t_start) & (idx.cc["opened_ts"] < t_end)]
    n_eval = len(eval_cc)
    is_fraud_map: dict[str, bool] = {}

    fired_by_link: dict[str, set[str]] = {lt: set() for lt in LINK_TYPES}
    fired_any: set[str] = set()

    for row in eval_cc.itertuples(index=False):
        cid = str(row.case_id)
        is_f = str(row.outcome) == "confirmed_fraud"
        is_fraud_map[cid] = is_f
        t_list = list(row.txn_list) if row.txn_list else []
        flagged_tid = int(row.first_fraud_txn_id) if pd.notna(row.first_fraud_txn_id) else (t_list[0] if t_list else -1)

        precs = precedents(
            card_id=str(row.card_id),
            flagged_txn_id=flagged_tid,
            as_of=row.opened_ts,
            exclude_case_id=cid,
            case_txn_ids=t_list if t_list else None,
        )
        for p in precs:
            if p.outcome == "confirmed_fraud":
                fired_any.add(cid)
                for path in p.paths:
                    fired_by_link[path.entity_type].add(cid)

    total_fraud = sum(1 for v in is_fraud_map.values() if v)
    total_cleared = n_eval - total_fraud

    stats_out: dict[str, LinkTypeStats] = {}
    all_keys = list(LINK_TYPES) + ["any_fraud_precedent"]

    for lt in all_keys:
        fired_set = fired_any if lt == "any_fraud_precedent" else fired_by_link[lt]
        caught = sorted(c for c in fired_set if is_fraud_map[c])
        legit_hit = sorted(c for c in fired_set if not is_fraud_map[c])

        n_link = len(fired_set)
        k_link_fraud = len(caught)
        n_no_link = n_eval - n_link
        k_no_link_fraud = total_fraud - k_link_fraud

        p_link = k_link_fraud / n_link if n_link > 0 else 0.0
        ci_link = wilson_ci(k_link_fraud, n_link)

        p_no_link = k_no_link_fraud / n_no_link if n_no_link > 0 else 0.0
        ci_no_link = wilson_ci(k_no_link_fraud, n_no_link)

        rr = (p_link / p_no_link) if p_no_link > 0 else 1.0
        lr_val, lr_ci_val = lr_with_ci(k_link_fraud, total_fraud, len(legit_hit), total_cleared)

        stats_out[lt] = LinkTypeStats(
            link_type=lt,
            n_eval_cases=n_eval,
            n_linked_fraud=n_link,
            n_linked_fraud_is_fraud=k_link_fraud,
            n_no_linked_fraud=n_no_link,
            n_no_linked_fraud_is_fraud=k_no_link_fraud,
            p_fraud_given_link=p_link,
            p_fraud_given_link_ci=ci_link,
            p_fraud_given_no_link=p_no_link,
            p_fraud_given_no_link_ci=ci_no_link,
            risk_ratio=rr,
            lr=lr_val,
            lr_ci=lr_ci_val,
            legit_acted_against=legit_hit,
            caught_cases=caught,
        )

    return stats_out


def precedent_lr(prec_list: list[Precedent]) -> dict[str, Any]:
    """Compute the measured LR and link breakdown for a list of retrieved Precedents.

    Uses the Aug–Dec as-of backtest table `backtest_precedent_links()`.
    """
    bt = backtest_precedent_links()
    fraud_precs = [p for p in prec_list if p.outcome == "confirmed_fraud"]
    cleared_precs = [p for p in prec_list if p.outcome == "cleared"]

    if not fraud_precs:
        return {
            "lr": 1.0,
            "lr_ci": (1.0, 1.0),
            "best_link_type": None,
            "best_precedent_case_id": None,
            "n_fraud_precedents": 0,
            "n_cleared_precedents": len(cleared_precs),
            "per_link_type": {k: v.to_dict() for k, v in bt.items()},
        }

    present_types: set[str] = set()
    for p in fraud_precs:
        for path in p.paths:
            present_types.add(path.entity_type)

    best_lt = max(present_types, key=lambda lt: (bt[lt].lr, -LINK_PRIORITY.get(lt, 99)))
    best_stats = bt[best_lt]
    best_prec = next(
        p for p in fraud_precs if any(path.entity_type == best_lt for path in p.paths)
    )

    return {
        "lr": best_stats.lr,
        "lr_ci": best_stats.lr_ci,
        "p_fraud_given_link": best_stats.p_fraud_given_link,
        "p_fraud_given_link_ci": best_stats.p_fraud_given_link_ci,
        "p_fraud_given_no_link": best_stats.p_fraud_given_no_link,
        "p_fraud_given_no_link_ci": best_stats.p_fraud_given_no_link_ci,
        "best_link_type": best_lt,
        "best_precedent_case_id": best_prec.case_id,
        "n_fraud_precedents": len(fraud_precs),
        "n_cleared_precedents": len(cleared_precs),
        "per_link_type": {k: v.to_dict() for k, v in bt.items()},
    }


def exam_case_precedents() -> dict[str, list[Precedent]]:
    """Run `precedents()` across all 18 exam cases in `case_pack()`."""
    cp = case_pack()
    out: dict[str, list[Precedent]] = {}
    for row in cp.itertuples(index=False):
        out[str(row.case_id)] = precedents(
            card_id=str(row.card_id),
            flagged_txn_id=int(row.flagged_txn_id),
            as_of=str(row.opened_at),
        )
    return out


class PrecedentDetector(Detector):
    """C1 Detector wrapper around shared-entity precedent retrieval."""

    name: str = "precedent_shared_entity"
    pattern: str = "shared_entity_precedent"

    def run(self, ctx: CaseCtx) -> list[Finding]:
        precs = precedents(
            card_id=ctx.card_id,
            flagged_txn_id=ctx.flagged_txn_id,
            as_of=ctx.opened_at,
            exclude_case_id=ctx.case_id if ctx.case_id.startswith("CC-") else None,
        )
        fraud_precs = [p for p in precs if p.outcome == "confirmed_fraud"]
        if not fraud_precs:
            return []

        lr_info = precedent_lr(precs)
        top = fraud_precs[0]
        path = top.primary_path

        txn_ids = sorted({int(ctx.flagged_txn_id), int(path.current_txn_id), int(path.precedent_txn_id)})
        card_ids = sorted({str(ctx.card_id), str(top.card_id)})
        cust_ids = sorted({str(ctx.customer_id), str(top.customer_id)})
        case_ids = [p.case_id for p in fraud_precs]

        hhg_tag = f" ({top.hhg_id})" if top.hhg_id else ""
        what = (
            f"Case {ctx.case_id} links to {len(fraud_precs)} prior confirmed-fraud case(s) "
            f"(top: {top.case_id}{hhg_tag}, pattern={top.pattern}, exposure=${top.exposure:,.2f}) "
            f"via {path.entity_type}={path.value} between txn {path.current_txn_id} and {path.precedent_txn_id}."
        )
        att = Attestation(
            what=what,
            evidence=Evidence(
                txn_ids=txn_ids,
                card_ids=card_ids,
                customer_ids=cust_ids,
                case_ids=case_ids,
            ),
            query="assay.r2.precedent.precedents",
            params={
                "card_id": str(ctx.card_id),
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "as_of": str(ctx.opened_at),
            },
        )
        conn_cards = sorted({p.card_id for p in fraud_precs if p.card_id != ctx.card_id})
        affected = sorted({int(ctx.flagged_txn_id)} | {p.current_txn_id for pr in fraud_precs for p in pr.paths})

        return [
            Finding(
                detector=self.name,
                pattern=top.pattern if top.pattern != "none" else self.pattern,
                claim=att,
                lr=float(lr_info["lr"]),
                affected_txn_ids=affected,
                connected_card_ids=conn_cards,
            )
        ]

    def backtest(self, cc: pd.DataFrame | None = None, tx: pd.DataFrame | None = None) -> Backtest:
        bt_map = backtest_precedent_links()
        any_stats = bt_map["any_fraud_precedent"]
        idx = _get_index()
        t_start = pd.to_datetime("2016-08-01")
        t_end = pd.to_datetime("2017-01-01")
        eval_cc = idx.cc[(idx.cc["opened_ts"] >= t_start) & (idx.cc["opened_ts"] < t_end)]
        caught_set = set(any_stats.caught_cases)
        missed = sorted(
            str(r.case_id)
            for r in eval_cc.itertuples(index=False)
            if str(r.outcome) == "confirmed_fraud" and str(r.case_id) not in caught_set
        )
        return Backtest(
            detector=self.name,
            n_cases=any_stats.n_eval_cases,
            caught=any_stats.caught_cases,
            missed=missed,
            legit_acted_against=any_stats.legit_acted_against,
            lr_measured=any_stats.lr,
        )
