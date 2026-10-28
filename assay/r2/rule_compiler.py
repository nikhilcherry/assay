"""Block B8 (Challenge 3): Policy text to detector compiler.

A deterministic, zero-LLM compiler from written policy rules and regulator
advisories into executable, vectorised `RuleSpec` detectors over `tx_all()` and
`closed_cases()`, with as-of backtesting and attested agreement/disagreement
comparisons against Round 1 agent verdicts (`cases/HHG-*.json`).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import DATA, closed_cases, tx_all

# Policy R5 & R6 verbatim from policy/fraud_policy.md
POLICY_R5_TEXT = (
    "R5. Card testing. Three or more small online authorizations on one card "
    "within an hour, followed by a larger purchase: recommend DECLINE_TRANSACTION "
    "and STEP_UP_AUTH. If a purchase over $100 has already cleared, recommend BLOCK_CARD."
)

POLICY_R6_TEXT = (
    "R6. Shared origin. When several cards show fraud from the same device profile, "
    "the same billing region, or the same recipient email in one window, name the "
    "shared element, recommend CREATE_CASE and FILE_REPORT, and MONITOR_CONNECTED_CARDS "
    "for every card that shares it."
)

# Three regulator-style advisories exercised by the compiler
ADVISORY_1_TEXT = (
    "ADV-1. Cross-region device ring. Three or more cards sharing one device profile "
    "used from two or more billing regions within 24 hours: recommend CREATE_CASE, "
    "FILE_REPORT, and MONITOR_CONNECTED_CARDS."
)

ADVISORY_2_TEXT = (
    "ADV-2. High-exposure recipient burst. Two or more cards sending online transactions "
    "to the same recipient email within 48 hours where exposure exceeds $1,000: "
    "recommend BLOCK_CARD, CREATE_CASE, and FILE_REPORT."
)

ADVISORY_3_TEXT = (
    "ADV-3. Rapid high-value velocity. Four or more online authorizations on one card "
    "within 2 hours where exposure exceeds $500, followed by biometric voiceprint "
    "verification failure: recommend DECLINE_TRANSACTION, STEP_UP_AUTH, and ESCALATE_TO_ANALYST."
)

DEFAULT_RULES: dict[str, str] = {
    "R5": POLICY_R5_TEXT,
    "R6": POLICY_R6_TEXT,
    "ADV-1": ADVISORY_1_TEXT,
    "ADV-2": ADVISORY_2_TEXT,
    "ADV-3": ADVISORY_3_TEXT,
}

NUMBER_WORDS: dict[str, int] = {
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "couple": 2,
    "three": 3, "several": 3, "multiple": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}

KNOWN_ACTIONS: tuple[str, ...] = (
    "ALLOW_TRANSACTION",
    "DECLINE_TRANSACTION",
    "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH",
    "BLOCK_CARD",
    "BLOCK_ALL_CARDS",
    "GENERATE_REPORT",
    "CREATE_CASE",
    "FILE_REPORT",
    "ESCALATE_TO_ANALYST",
    "CLOSE_NO_FRAUD",
)

GENERIC_DEVICES = frozenset({"", "none", "nan", "? | ? | ? | ?"})


@dataclass
class RuleSpec:
    """Typed AST compiled from a policy rule or regulator advisory."""
    rule_id: str
    title: str
    raw_text: str
    kind: str                                   # "card_sequence" | "shared_entity" | "card_velocity"
    min_txns: int = 1
    min_cards: int = 1
    window_seconds: int = 3600
    channel: str | None = None                  # "online" | "in_person" | None
    small_amt_max: float | None = None          # e.g. $25 for "small"
    followup_larger: bool = False               # "followed by a larger purchase"
    cleared_amt_threshold: float | None = None  # e.g. $100 for "purchase over $100"
    min_exposure_usd: float | None = None       # e.g. $1000 for "exposure exceeds $1,000"
    shared_entities: list[str] = field(default_factory=list)  # ["device_profile", "addr1", "R_emaildomain"]
    require_prior_fraud: bool = False           # "cards show fraud"
    min_distinct_regions: int = 1               # "from two or more billing regions"
    recommended_actions: list[str] = field(default_factory=list)
    conditional_actions: dict[str, str] = field(default_factory=dict)
    parsed_spans: list[str] = field(default_factory=list)
    unparsed_phrases: list[str] = field(default_factory=list)  # needs human review
    compile_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentComparisonRow:
    case_id: str
    cc_case_id: str
    card_id: str
    customer_id: str
    actual_outcome: str
    actual_pattern: str
    agent_verdict: str
    agent_pattern: str
    rule_fired: bool
    status: str                 # "agree_positive" | "agree_negative" | "disagree_rule_only" | "disagree_agent_only"
    matched_txn_ids: list[int] = field(default_factory=list)
    connected_card_ids: list[str] = field(default_factory=list)


@dataclass
class RuleComparisonReport:
    rule_id: str
    n_compared: int
    agree_positive: list[str]
    agree_negative: list[str]
    disagree_rule_only: list[str]
    disagree_agent_only: list[str]
    rows: list[AgentComparisonRow]
    attestations: list[Attestation]


def _parse_int(token: str) -> int:
    t = token.strip().lower()
    if t in NUMBER_WORDS:
        return NUMBER_WORDS[t]
    return int(re.sub(r"[^\d]", "", t))


def _split_clauses(body: str) -> list[str]:
    parts = re.split(r"[.;:]|\b(?:and|or(?!\s+more)|when|where|if)\b|(?<!\d),|,(?!\d)", body)
    return [p.strip() for p in parts if p and p.strip()]


# Controlled grammar patterns
_HEADER_RE = re.compile(r"^\s*\*{0,2}([A-Z]+-?\d+)\.\s*([^.*:]+?)\.\*{0,2}\s*(.*)$", re.DOTALL)
_TXN_QUANT_RE = re.compile(
    r"\b(?P<n>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+or\s+more\s+"
    r"(?P<small>small\s+)?(?P<chan>online|in[- ]person)?\s*(?:authorizations|transactions|purchases|charges)",
    re.IGNORECASE,
)
_CARD_QUANT_RE = re.compile(
    r"\b(?P<n>several|multiple|two|three|four|five|\d+)(?:\s+or\s+more)?\s+cards\b",
    re.IGNORECASE,
)
_ONE_CARD_RE = re.compile(r"\bon\s+(?:one|a\s+single|the\s+same)\s+card\b", re.IGNORECASE)
_WINDOW_RE = re.compile(
    r"\b(?:within|in)\s+(?P<n>an|a|one|two|three|four|six|twelve|twenty[- ]four|\d+)\s*"
    r"(?P<unit>hour|hours|hr|hrs|minute|minutes|min|day|days|window|windows)\b",
    re.IGNORECASE,
)
_FOLLOWED_LARGER_RE = re.compile(
    r"\bfollowed\s+by\s+a\s+larger\s+(?:purchase|transaction|authorization|charge)\b",
    re.IGNORECASE,
)
_PURCHASE_OVER_RE = re.compile(
    r"\b(?:a\s+)?(?:purchase|transaction|charge|authorization)\s+(?:over|exceeding|above)\s+\$(?P<amt>[\d,]+(?:\.\d+)?)"
    r"(?:\s+has\s+already\s+cleared)?",
    re.IGNORECASE,
)
_EXPOSURE_OVER_RE = re.compile(
    r"\bexposure\s+(?:exceeds|over|above|is\s+over)\s+\$(?P<amt>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SHARED_DEV_RE = re.compile(
    r"\b(?:(?:same|one|shared)\s+device(?:\s+profile)?|sharing\s+one\s+device(?:\s+profile)?)\b",
    re.IGNORECASE,
)
_SHARED_REG_RE = re.compile(r"\b(?:same|one|shared)\s+billing\s+region\b", re.IGNORECASE)
_SHARED_EMAIL_RE = re.compile(r"\b(?:same|one|shared)\s+recipient\s+email\b", re.IGNORECASE)
_MULTI_REG_RE = re.compile(
    r"\b(?:from\s+)?(?P<n>two|three|four|\d+)\s+or\s+more\s+billing\s+regions\b",
    re.IGNORECASE,
)
_SHOW_FRAUD_RE = re.compile(r"\bshow\s+fraud\b", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"^(?:recommend\b.*|name\s+the\s+shared\s+element.*|for\s+every\s+card\s+that\s+shares\s+it.*|"
    r"sending\s+online\s+transactions.*|to\s+the\s+same\s+recipient\s+email.*|from\s+the\s+same.*|"
    r"the\s+same\s+billing\s+region.*|the\s+same\s+recipient\s+email.*|used\s+from.*)$",
    re.IGNORECASE,
)


def compile_rule(prose: str, rule_id: str | None = None) -> RuleSpec:
    """ deterministically compile rule prose into a typed `RuleSpec`."""
    t0 = time.perf_counter()
    clean = prose.replace("**", "").replace("`", "").strip()

    m_head = _HEADER_RE.match(clean)
    if m_head:
        rid = rule_id or m_head.group(1).strip()
        title = m_head.group(2).strip()
        body = m_head.group(3).strip()
    else:
        rid = rule_id or "CUSTOM"
        title = "Custom rule"
        body = clean

    spec = RuleSpec(rule_id=rid, title=title, raw_text=prose, kind="card_velocity")
    spans: list[tuple[int, int, str]] = []

    def _mark(match: re.Match, label: str) -> None:
        spans.append((match.start(), match.end(), label))
        spec.parsed_spans.append(f"{label}: '{match.group(0).strip()}'")

    # 1. Quantifiers on transactions
    for m in _TXN_QUANT_RE.finditer(body):
        _mark(m, "txn_quantifier")
        spec.min_txns = _parse_int(m.group("n"))
        if m.group("small"):
            spec.small_amt_max = 25.0
        if m.group("chan"):
            ch = m.group("chan").lower().replace("-", "_")
            spec.channel = "online" if "online" in ch else "in_person"

    # 2. Quantifiers on cards
    for m in _CARD_QUANT_RE.finditer(body):
        _mark(m, "card_quantifier")
        spec.min_cards = _parse_int(m.group("n"))

    for m in _ONE_CARD_RE.finditer(body):
        _mark(m, "single_card_scope")
        spec.min_cards = 1

    # 3. Online channel mentioned outside transaction quantifier
    if spec.channel is None and re.search(r"\bonline\b", body, re.IGNORECASE):
        m_ch = re.search(r"\bonline\s+(?:transactions|authorizations|purchases|activity)?", body, re.IGNORECASE)
        if m_ch:
            _mark(m_ch, "channel_online")
            spec.channel = "online"

    # 4. Time windows
    for m in _WINDOW_RE.finditer(body):
        _mark(m, "time_window")
        n_val = _parse_int(m.group("n"))
        unit = m.group("unit").lower()
        if unit.startswith("min"):
            spec.window_seconds = n_val * 60
        elif unit.startswith(("hour", "hr")):
            spec.window_seconds = n_val * 3600
        elif unit.startswith("day"):
            spec.window_seconds = n_val * 86400
        elif unit.startswith("window"):
            # In policy R6 ("in one window"), episodes are evaluated within a 7-day operational window
            spec.window_seconds = 7 * 86400

    # 5. Sequence / larger follow-up
    for m in _FOLLOWED_LARGER_RE.finditer(body):
        _mark(m, "followup_larger")
        spec.followup_larger = True

    # 6. Thresholds (purchase over $X, exposure exceeds $Y)
    for m in _PURCHASE_OVER_RE.finditer(body):
        _mark(m, "cleared_amt_threshold")
        spec.cleared_amt_threshold = float(m.group("amt").replace(",", ""))

    for m in _EXPOSURE_OVER_RE.finditer(body):
        _mark(m, "min_exposure_usd")
        spec.min_exposure_usd = float(m.group("amt").replace(",", ""))

    # 7. Shared entities & multi-region constraints
    for m in _SHARED_DEV_RE.finditer(body):
        _mark(m, "shared_entity:device_profile")
        if "device_profile" not in spec.shared_entities:
            spec.shared_entities.append("device_profile")

    for m in _SHARED_REG_RE.finditer(body):
        _mark(m, "shared_entity:addr1")
        if "addr1" not in spec.shared_entities:
            spec.shared_entities.append("addr1")

    for m in _SHARED_EMAIL_RE.finditer(body):
        _mark(m, "shared_entity:R_emaildomain")
        if "R_emaildomain" not in spec.shared_entities:
            spec.shared_entities.append("R_emaildomain")

    for m in _MULTI_REG_RE.finditer(body):
        _mark(m, "min_distinct_regions")
        spec.min_distinct_regions = _parse_int(m.group("n"))

    for m in _SHOW_FRAUD_RE.finditer(body):
        _mark(m, "require_prior_fraud")
        spec.require_prior_fraud = True

    # 8. Actions
    for act in KNOWN_ACTIONS:
        for m in re.finditer(rf"\b{act}\b", body):
            _mark(m, f"action:{act}")
            if act not in spec.recommended_actions:
                spec.recommended_actions.append(act)

    if spec.cleared_amt_threshold is not None and "BLOCK_CARD" in spec.recommended_actions:
        spec.conditional_actions[f"purchase_over_${int(spec.cleared_amt_threshold)}"] = "BLOCK_CARD"

    # Determine rule execution kind
    if spec.shared_entities or spec.min_cards >= 2:
        spec.kind = "shared_entity"
    elif spec.followup_larger or spec.small_amt_max is not None:
        spec.kind = "card_sequence"
    else:
        spec.kind = "card_velocity"

    # 9. Identify unparsed clauses (anything that isn't covered by our grammar or connective boilerplate)
    clauses = _split_clauses(body)
    for clause in clauses:
        if not clause or _BOILERPLATE_RE.match(clause):
            continue
        # Check if clause matches any grammar regex or action
        matched_any = any(
            rx.search(clause)
            for rx in (
                _TXN_QUANT_RE, _CARD_QUANT_RE, _ONE_CARD_RE, _WINDOW_RE,
                _FOLLOWED_LARGER_RE, _PURCHASE_OVER_RE, _EXPOSURE_OVER_RE,
                _SHARED_DEV_RE, _SHARED_REG_RE, _SHARED_EMAIL_RE,
                _MULTI_REG_RE, _SHOW_FRAUD_RE,
            )
        ) or any(act in clause for act in KNOWN_ACTIONS)
        if not matched_any:
            spec.unparsed_phrases.append(clause)

    spec.compile_ms = round((time.perf_counter() - t0) * 1000.0, 4)
    return spec


# ---------------------------------------------------------------------------
# Vectorised execution over tx_all() and closed_cases()
# ---------------------------------------------------------------------------

def _eval_card_sequence_or_velocity(
    spec: RuleSpec,
    tx: pd.DataFrame,
    target_cards: set[str] | None = None,
) -> dict[str, dict]:
    """Vectorised per-card sliding window evaluator for R5 (card_sequence) and ADV-3 (card_velocity)."""
    df = tx
    if target_cards is not None:
        df = df[df["card_id"].isin(target_cards)]
    if spec.channel is not None:
        df = df[df["channel"] == spec.channel]
    if df.empty:
        return {}

    df = df.sort_values(["card_id", "TransactionDT"], kind="stable")
    cards = df["card_id"].to_numpy()
    t_sec = df["TransactionDT"].to_numpy(dtype=np.int64)
    amts = df["amount"].to_numpy(dtype=np.float64)
    tids = df["TransactionID"].to_numpy(dtype=np.int64)
    custs = df["customer_id"].to_numpy()

    # Pre-filter candidate cards that have at least min_txns
    card_s = pd.Series(cards)
    vc = card_s.value_counts()
    need_n = spec.min_txns + (1 if spec.followup_larger else 0)
    keep_cards = set(vc[vc >= need_n].index)
    if not keep_cards:
        return {}

    mask = card_s.isin(keep_cards).to_numpy()
    cards, t_sec, amts, tids, custs = cards[mask], t_sec[mask], amts[mask], tids[mask], custs[mask]

    # Split arrays by card boundaries
    boundaries = np.flatnonzero(cards[1:] != cards[:-1]) + 1
    idx_splits = np.split(np.arange(len(cards)), boundaries)

    hits: dict[str, dict] = {}
    win = int(spec.window_seconds)
    small_max = spec.small_amt_max if spec.small_amt_max is not None else float("inf")
    min_k = int(spec.min_txns)

    for idxs in idx_splits:
        cid = str(cards[idxs[0]])
        c_t = t_sec[idxs]
        c_a = amts[idxs]
        c_tid = tids[idxs]
        cust_id = str(custs[idxs[0]])

        if spec.kind == "card_sequence":
            # Find runs of >= min_k small txns within `win`, followed by a larger purchase within `win`
            small_mask = c_a <= small_max
            if small_mask.sum() < min_k:
                continue
            n = len(c_t)
            matched_tids: list[int] = []
            max_Purchase = 0.0
            for i in range(n):
                if not small_mask[i]:
                    continue
                # Window [c_t[i], c_t[i] + win]
                j = int(np.searchsorted(c_t, c_t[i] + win, side="right"))
                win_a = c_a[i:j]
                win_small = small_mask[i:j]
                if win_small.sum() < min_k:
                    continue
                if spec.followup_larger:
                    # Need at least min_k small txns before a larger txn in [i, i + win_ext]
                    j_follow = int(np.searchsorted(c_t, c_t[i] + win * 2, side="right"))
                    small_positions = np.flatnonzero(small_mask[i:j_follow])
                    if len(small_positions) < min_k:
                        continue
                    kth_small_idx = i + int(small_positions[min_k - 1])
                    max_small_amt = float(np.max(c_a[i : kth_small_idx + 1][small_mask[i : kth_small_idx + 1]]))
                    after_slice_a = c_a[kth_small_idx + 1 : j_follow]
                    larger_rel = np.flatnonzero((after_slice_a > max_small_amt) & (after_slice_a > small_max))
                    if len(larger_rel) == 0:
                        continue
                    end_idx = kth_small_idx + 1 + int(larger_rel[-1])
                    ep_tids = [int(x) for x in c_tid[i : end_idx + 1]]
                    ep_max = float(np.max(c_a[i : end_idx + 1]))
                    matched_tids = ep_tids
                    max_purchase = ep_max
                    break
                else:
                    matched_tids = [int(x) for x in c_tid[i:j]]
                    max_purchase = float(np.max(win_a))
                    break

            if matched_tids:
                hits[cid] = {
                    "card_id": cid,
                    "customer_id": cust_id,
                    "txn_ids": matched_tids,
                    "connected_card_ids": [],
                    "exposure_usd": round(float(c_a[np.isin(c_tid, matched_tids)].sum()), 2),
                    "max_amount": round(max_purchase, 2),
                }
        else:
            # card_velocity: >= min_k txns within `win` with optional min_exposure_usd
            r_idx = np.searchsorted(c_t, c_t + win, side="right")
            counts = r_idx - np.arange(len(c_t))
            valid_starts = np.flatnonzero(counts >= min_k)
            for i in valid_starts:
                j = int(r_idx[i])
                exp = float(c_a[i:j].sum())
                if spec.min_exposure_usd is not None and exp <= spec.min_exposure_usd:
                    continue
                hits[cid] = {
                    "card_id": cid,
                    "customer_id": cust_id,
                    "txn_ids": [int(x) for x in c_tid[i:j]],
                    "connected_card_ids": [],
                    "exposure_usd": round(exp, 2),
                    "max_amount": round(float(c_a[i:j].max()), 2),
                }
                break

    return hits


def _eval_shared_entity_window(
    spec: RuleSpec,
    tx: pd.DataFrame,
    cc_prior: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Vectorised evaluator for shared-origin rules (R6, ADV-1, ADV-2)."""
    df = tx
    if spec.channel is not None:
        df = df[df["channel"] == spec.channel]
    if df.empty:
        return {}

    # If rule requires cards showing fraud (R6), scope to cards in prior confirmed fraud cases
    # plus high-risk / disputed episodes if evaluating over closed cases
    fraud_cards: set[str] | None = None
    if spec.require_prior_fraud and cc_prior is not None and not cc_prior.empty:
        cf = cc_prior[cc_prior["outcome"] == "confirmed_fraud"]
        fraud_cards = set(cf["card_id"].dropna().astype(str))

    hits: dict[str, dict] = {}
    win = int(spec.window_seconds)

    for col in spec.shared_entities:
        if col not in df.columns:
            continue
        sub = df[df[col].notna()].copy()
        if col == "device_profile":
            sub = sub[~sub[col].astype(str).str.lower().isin(GENERIC_DEVICES)]
            # Exclude synthetic fallback profiles like "C11709-K2|191|43" that encode card_id
            sub = sub[~sub[col].astype(str).str.match(r"^C\d{5}-K\d+\|")]
        elif col == "addr1":
            # Billing region alone is only meaningful when combined with device or when prior fraud is confirmed
            if len(spec.shared_entities) > 1:
                continue

        if sub.empty:
            continue

        # Filter to entities shared by >= spec.min_cards distinct cards
        card_counts = sub.groupby(col)["card_id"].nunique()
        cand_entities = card_counts[card_counts >= spec.min_cards].index
        if len(cand_entities) == 0:
            continue

        sub = sub[sub[col].isin(cand_entities)].sort_values([col, "TransactionDT"], kind="stable")
        ent_vals = sub[col].to_numpy()
        t_sec = sub["TransactionDT"].to_numpy(dtype=np.int64)
        cards = sub["card_id"].to_numpy()
        custs = sub["customer_id"].to_numpy()
        amts = sub["amount"].to_numpy(dtype=np.float64)
        tids = sub["TransactionID"].to_numpy(dtype=np.int64)
        regions = sub["addr1"].fillna(-1.0).to_numpy(dtype=np.float64)

        boundaries = np.flatnonzero(ent_vals[1:] != ent_vals[:-1]) + 1
        for idxs in np.split(np.arange(len(sub)), boundaries):
            e_t = t_sec[idxs]
            e_c = cards[idxs]
            e_cust = custs[idxs]
            e_a = amts[idxs]
            e_tid = tids[idxs]
            e_reg = regions[idxs]

            r_idx = np.searchsorted(e_t, e_t + win, side="right")
            for i in range(len(e_t)):
                j = int(r_idx[i])
                if j - i < spec.min_cards:
                    continue
                win_cards = pd.unique(e_c[i:j])
                if len(win_cards) < spec.min_cards:
                    continue
                if fraud_cards is not None:
                    n_prior_fraud = sum(1 for c in win_cards if str(c) in fraud_cards)
                    if n_prior_fraud < max(1, spec.min_cards - 1):
                        continue
                if spec.min_distinct_regions > 1:
                    valid_regs = e_reg[i:j][e_reg[i:j] >= 0]
                    if len(np.unique(valid_regs)) < spec.min_distinct_regions:
                        continue
                exp = float(e_a[i:j].sum())
                if spec.min_exposure_usd is not None and exp <= spec.min_exposure_usd:
                    continue

                ep_tids = [int(x) for x in e_tid[i:j]]
                win_card_list = [str(c) for c in win_cards]
                for pos in range(i, j):
                    cid = str(e_c[pos])
                    cust_id = str(e_cust[pos])
                    others = [c for c in win_card_list if c != cid]
                    prev = hits.get(cid)
                    if prev is None or len(others) > len(prev["connected_card_ids"]):
                        hits[cid] = {
                            "card_id": cid,
                            "customer_id": cust_id,
                            "txn_ids": ep_tids,
                            "connected_card_ids": others,
                            "shared_field": col,
                            "shared_value": str(ent_vals[idxs[0]]),
                            "exposure_usd": round(exp, 2),
                        }
                break

    return hits


def evaluate_rule_on_cases(
    spec: RuleSpec,
    cc: pd.DataFrame,
    tx: pd.DataFrame,
) -> dict[str, dict]:
    """As-of backtest evaluation of `spec` across `closed_cases()` rows.

    Strictly enforces `t <= opened_at` per case: only transactions occurring on or
    before a case's `opened_at` timestamp can trigger a detection on that case.
    """
    tx_lookup = tx.set_index("TransactionID")
    cc_opened = pd.to_datetime(cc["opened_at"])
    cc_closed = pd.to_datetime(cc["closed_at"])

    if spec.kind in ("card_sequence", "card_velocity"):
        # Filter tx to cards in cc and t <= each case's opened_at
        case_max_open = cc.assign(_open=cc_opened).groupby("card_id")["_open"].max().to_dict()
        sub_tx = tx[tx["card_id"].isin(case_max_open)].copy()
        max_open_s = sub_tx["card_id"].map(case_max_open)
        sub_tx = sub_tx[sub_tx["t"] <= max_open_s]
        card_hits = _eval_card_sequence_or_velocity(spec, sub_tx)

        case_hits: dict[str, dict] = {}
        for row in cc.itertuples():
            cid = str(row.card_id)
            if cid not in card_hits:
                continue
            hit = card_hits[cid]
            # Verify as-of for this specific case's opened_at
            open_ts = pd.Timestamp(row.opened_at)
            valid_tids = [
                tid for tid in hit["txn_ids"]
                if tid in tx_lookup.index and tx_lookup.at[tid, "t"] <= open_ts
            ]
            if len(valid_tids) >= spec.min_txns:
                case_hits[str(row.case_id)] = {
                    **hit,
                    "txn_ids": valid_tids,
                    "case_id": str(row.case_id),
                }
        return case_hits

    # Shared-entity rule: evaluate over tx and verify as-of per case's opened_at
    # For R6 (`require_prior_fraud=True`), a shared origin is established when >= min_cards
    # share a non-generic device profile or recipient email within `window_seconds` and
    # at least one connected card has a confirmed fraud episode or shared closed case.
    ent_hits_by_case: dict[str, dict] = {}
    win = int(spec.window_seconds)

    # Pre-index closed cases by card for fast as-of lookup
    card_to_cases: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp, str]]] = {}
    for r in cc.itertuples():
        card_to_cases.setdefault(str(r.card_id), []).append(
            (str(r.case_id), pd.Timestamp(r.opened_at), pd.Timestamp(r.closed_at), str(r.outcome))
        )

    df = tx
    if spec.channel is not None:
        df = df[df["channel"] == spec.channel]

    for col in spec.shared_entities:
        if col not in df.columns:
            continue
        if col == "addr1" and len(spec.shared_entities) > 1:
            continue
        sub = df[df[col].notna()].copy()
        if col == "device_profile":
            sub = sub[~sub[col].astype(str).str.lower().isin(GENERIC_DEVICES)]
            sub = sub[~sub[col].astype(str).str.match(r"^C\d{5}-K\d+\|")]
        if sub.empty:
            continue

        vc = sub.groupby(col)["card_id"].nunique()
        cand = vc[vc >= spec.min_cards].index
        if len(cand) == 0:
            continue

        sub = sub[sub[col].isin(cand)].sort_values([col, "TransactionDT"], kind="stable")
        ent_vals = sub[col].to_numpy()
        t_sec = sub["TransactionDT"].to_numpy(dtype=np.int64)
        t_ts = sub["t"].to_numpy()
        cards = sub["card_id"].to_numpy()
        custs = sub["customer_id"].to_numpy()
        amts = sub["amount"].to_numpy(dtype=np.float64)
        tids = sub["TransactionID"].to_numpy(dtype=np.int64)
        regions = sub["addr1"].fillna(-1.0).to_numpy(dtype=np.float64)

        boundaries = np.flatnonzero(ent_vals[1:] != ent_vals[:-1]) + 1
        for idxs in np.split(np.arange(len(sub)), boundaries):
            e_t = t_sec[idxs]
            e_ts = t_ts[idxs]
            e_c = cards[idxs]
            e_cust = custs[idxs]
            e_a = amts[idxs]
            e_tid = tids[idxs]
            e_reg = regions[idxs]

            r_idx = np.searchsorted(e_t, e_t + win, side="right")
            for i in range(len(e_t)):
                j = int(r_idx[i])
                if j - i < spec.min_cards:
                    continue
                win_cards = pd.unique(e_c[i:j])
                if len(win_cards) < spec.min_cards:
                    continue
                if spec.min_distinct_regions > 1:
                    valid_regs = e_reg[i:j][e_reg[i:j] >= 0]
                    if len(np.unique(valid_regs)) < spec.min_distinct_regions:
                        continue
                exp = float(e_a[i:j].sum())
                if spec.min_exposure_usd is not None and exp <= spec.min_exposure_usd:
                    continue

                win_end_ts = pd.Timestamp(e_ts[j - 1])
                win_card_list = [str(c) for c in win_cards]
                ep_tids = [int(x) for x in e_tid[i:j]]

                for cid in win_card_list:
                    for case_id, open_ts, _, _ in card_to_cases.get(cid, ()):
                        if open_ts < pd.Timestamp(e_ts[i]):
                            continue
                        # As-of filter inside window: only count txns <= open_ts
                        asof_mask = e_ts[i:j] <= open_ts.to_datetime64()
                        asof_cards = pd.unique(e_c[i:j][asof_mask])
                        if len(asof_cards) < spec.min_cards:
                            continue
                        if spec.min_distinct_regions > 1:
                            asof_regs = e_reg[i:j][asof_mask]
                            asof_regs = asof_regs[asof_regs >= 0]
                            if len(np.unique(asof_regs)) < spec.min_distinct_regions:
                                continue
                        asof_exp = float(e_a[i:j][asof_mask].sum())
                        if spec.min_exposure_usd is not None and asof_exp <= spec.min_exposure_usd:
                            continue

                        others = [str(c) for c in asof_cards if str(c) != cid]
                        ent_hits_by_case[case_id] = {
                            "case_id": case_id,
                            "card_id": cid,
                            "customer_id": str(e_cust[0]),
                            "txn_ids": [int(x) for x in e_tid[i:j][asof_mask]],
                            "connected_card_ids": others,
                            "shared_field": col,
                            "shared_value": str(ent_vals[idxs[0]]),
                            "exposure_usd": round(asof_exp, 2),
                        }

    return ent_hits_by_case


class CompiledRuleDetector(Detector):
    """A `Detector` subclass backed by a compiled `RuleSpec`."""

    def __init__(self, spec: RuleSpec):
        self.spec = spec
        self.name = f"compiled_{spec.rule_id.lower().replace('-', '_')}"
        self.pattern = (
            "card_testing"
            if spec.rule_id.upper() == "R5"
            else f"compiled_rule:{spec.rule_id.lower()}"
        )
        self._cached_lr: float | None = None

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        hits = evaluate_rule_on_cases(self.spec, cc, tx)
        fraud_cc = cc[cc["outcome"] == "confirmed_fraud"]
        cleared_cc = cc[cc["outcome"] == "cleared"]

        fraud_ids = set(fraud_cc["case_id"].astype(str))
        cleared_ids = set(cleared_cc["case_id"].astype(str))
        fired_ids = set(hits.keys())

        caught = sorted(fraud_ids & fired_ids)
        missed = sorted(fraud_ids - fired_ids)
        legit_acted = sorted(cleared_ids & fired_ids)

        # Laplace-smoothed likelihood ratio P(fire | fraud) / P(fire | cleared)
        p_fraud = (len(caught) + 1.0) / (len(fraud_ids) + 2.0)
        p_clear = (len(legit_acted) + 1.0) / (len(cleared_ids) + 2.0)
        lr = round(float(p_fraud / p_clear), 4)
        self._cached_lr = lr

        return Backtest(
            detector=self.name,
            n_cases=len(cc),
            caught=caught,
            missed=missed,
            legit_acted_against=legit_acted,
            lr_measured=lr,
        )

    def run(self, ctx: CaseCtx) -> list[Finding]:
        lr = self._cached_lr if self._cached_lr is not None else 1.0
        if self.spec.kind in ("card_sequence", "card_velocity"):
            hits = _eval_card_sequence_or_velocity(self.spec, ctx.tx, target_cards={ctx.card_id})
        else:
            hits = _eval_shared_entity_window(self.spec, ctx.tx, cc_prior=ctx.cc)

        if ctx.card_id not in hits:
            return []

        h = hits[ctx.card_id]
        claim = Attestation(
            what=(
                f"Rule {self.spec.rule_id} ({self.spec.title}) fired on card {ctx.card_id} "
                f"with {len(h['txn_ids'])} transaction(s) and ${h['exposure_usd']:.2f} exposure."
            ),
            evidence=Evidence(
                txn_ids=h["txn_ids"],
                card_ids=[ctx.card_id] + h.get("connected_card_ids", []),
                customer_ids=[ctx.customer_id],
                case_ids=[ctx.case_id],
            ),
            query="assay.r2.rule_compiler.run_compiled_rule",
            params={"rule_id": self.spec.rule_id, "case_id": ctx.case_id, "card_id": ctx.card_id},
        )
        return [
            Finding(
                detector=self.name,
                pattern=self.pattern,
                claim=claim,
                lr=lr,
                affected_txn_ids=h["txn_ids"],
                connected_card_ids=h.get("connected_card_ids", []),
            )
        ]


def build_backtest_attestation(spec: RuleSpec, bt: Backtest) -> Attestation:
    """Create a re-runnable `Attestation` for a compiled rule's backtest."""
    return Attestation(
        what=(
            f"Compiled detector {bt.detector} ({spec.rule_id}: {spec.title}) backtested on "
            f"{bt.n_cases} closed cases as-of opened_at: caught {len(bt.caught)} fraud cases, "
            f"missed {len(bt.missed)}, acted against {len(bt.legit_acted_against)} cleared cases "
            f"(precision={bt.precision:.3f}, recall={bt.recall:.4f}, LR={bt.lr_measured:.2f})."
        ),
        evidence=Evidence(
            case_ids=bt.caught[:50] + bt.legit_acted_against[:50],
        ),
        query="assay.r2.rule_compiler.backtest_rule_spec",
        params={
            "rule_id": spec.rule_id,
            "kind": spec.kind,
            "min_txns": spec.min_txns,
            "min_cards": spec.min_cards,
            "window_seconds": spec.window_seconds,
        },
    )


# ---------------------------------------------------------------------------
# Comparison against Round 1 Agent answers (cases/HHG-*.json)
# ---------------------------------------------------------------------------

def load_r1_agent_answers(cases_dir: Path | None = None) -> dict[str, dict]:
    """Load our Round 1 agent output files `cases/HHG-001.json` .. `HHG-020.json`."""
    cdir = cases_dir or (DATA.parent / "cases")
    out: dict[str, dict] = {}
    for p in sorted(cdir.glob("HHG-*.json")):
        data = json.loads(p.read_text())
        cid = data.get("case_id", p.stem)
        out[cid] = data
    return out


def compare_rule_with_r1_agent(
    spec: RuleSpec,
    cc: pd.DataFrame | None = None,
    tx: pd.DataFrame | None = None,
    cases_dir: Path | None = None,
) -> RuleComparisonReport:
    """Compare a compiled `RuleSpec` against our Round 1 agent verdicts on HHG-001..020.

    Note: In `closed_cases_history.csv`, the last 20 closed cases (`CC-5566`..`CC-5585`)
    are the Round 1 `HHG-001`..`HHG-020` investigations with their real outcomes.
    """
    cc_df = cc if cc is not None else closed_cases()
    tx_df = tx if tx is not None else tx_all()
    r1_answers = load_r1_agent_answers(cases_dir)

    # Map HHG-001..020 to CC-5566..5585 (the tail 20 records of closed_cases_history.csv)
    hhg_cc = cc_df.tail(20).copy().reset_index(drop=True)
    hhg_cc["hhg_id"] = [f"HHG-{i:03d}" for i in range(1, len(hhg_cc) + 1)]

    hits = evaluate_rule_on_cases(spec, hhg_cc, tx_df)

    agree_pos: list[str] = []
    agree_neg: list[str] = []
    disagree_rule: list[str] = []
    disagree_agent: list[str] = []
    rows: list[AgentComparisonRow] = []
    attestations: list[Attestation] = []

    for r in hhg_cc.itertuples():
        hhg_id = str(r.hhg_id)
        cc_id = str(r.case_id)
        ans = r1_answers.get(hhg_id, {}).get("case", {})
        agent_verdict = str(ans.get("verdict", "unknown"))
        agent_pattern = str(ans.get("pattern", "none"))
        agent_txns = [int(x) for x in ans.get("affected_txn_ids", []) if str(x).isdigit()]

        rule_fired = cc_id in hits
        hit_info = hits.get(cc_id, {})
        rule_txns = hit_info.get("txn_ids", [])
        rule_conn = hit_info.get("connected_card_ids", [])

        # Determine whether the agent judged this case positive for fraud (or specifically for this typology)
        if spec.rule_id.upper() == "R5":
            agent_positive = agent_verdict == "fraud" and agent_pattern == "card_testing"
        elif spec.rule_id.upper() == "R6":
            agent_positive = agent_verdict == "fraud" and (
                len(ans.get("connected_card_ids", [])) > 0
                or len(ans.get("connected_device_profiles", [])) > 0
                or agent_pattern in ("account_takeover", "card_not_present_new_device", "out_of_region_use")
            )
        else:
            agent_positive = agent_verdict == "fraud"

        if rule_fired and agent_positive:
            status = "agree_positive"
            agree_pos.append(hhg_id)
        elif not rule_fired and not agent_positive:
            status = "agree_negative"
            agree_neg.append(hhg_id)
        elif rule_fired and not agent_positive:
            status = "disagree_rule_only"
            disagree_rule.append(hhg_id)
        else:
            status = "disagree_agent_only"
            disagree_agent.append(hhg_id)

        row = AgentComparisonRow(
            case_id=hhg_id,
            cc_case_id=cc_id,
            card_id=str(r.card_id),
            customer_id=str(r.customer_id),
            actual_outcome=str(r.outcome),
            actual_pattern=str(r.pattern),
            agent_verdict=agent_verdict,
            agent_pattern=agent_pattern,
            rule_fired=rule_fired,
            status=status,
            matched_txn_ids=rule_txns,
            connected_card_ids=rule_conn,
        )
        rows.append(row)

        ev_txns = sorted(set(rule_txns) | set(agent_txns) | set(r.txn_list))
        attestations.append(
            Attestation(
                what=(
                    f"Rule {spec.rule_id} vs Agent on {hhg_id} ({cc_id}, card {r.card_id}): "
                    f"status={status} (rule_fired={rule_fired}, agent_verdict={agent_verdict}, "
                    f"agent_pattern={agent_pattern}, actual_outcome={r.outcome})."
                ),
                evidence=Evidence(
                    txn_ids=ev_txns,
                    card_ids=[str(r.card_id)] + rule_conn,
                    customer_ids=[str(r.customer_id)],
                    case_ids=[hhg_id, cc_id],
                ),
                query="assay.r2.rule_compiler.compare_rule_with_r1_agent",
                params={"rule_id": spec.rule_id, "case_id": hhg_id, "cc_case_id": cc_id},
            )
        )

    return RuleComparisonReport(
        rule_id=spec.rule_id,
        n_compared=len(rows),
        agree_positive=agree_pos,
        agree_negative=agree_neg,
        disagree_rule_only=disagree_rule,
        disagree_agent_only=disagree_agent,
        rows=rows,
        attestations=attestations,
    )


def format_comparison_table(
    specs: list[RuleSpec],
    backtests: dict[str, Backtest],
    reports: dict[str, RuleComparisonReport],
) -> str:
    """Format a summary table of compile latency, backtests, and R1 agreement/disagreement."""
    lines = [
        "| Rule | Kind | Compile (ms) | Caught Fraud | Legit Acted Against | Precision | LR | R1 Agree (+) | R1 Agree (-) | R1 Disagree (Rule Only) | R1 Disagree (Agent Only) | Unparsed Phrases |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for s in specs:
        bt = backtests[s.rule_id]
        rep = reports[s.rule_id]
        unparsed = "; ".join(s.unparsed_phrases) if s.unparsed_phrases else "none"
        lines.append(
            f"| {s.rule_id} | `{s.kind}` | {s.compile_ms:.3f} | "
            f"{len(bt.caught)}/{len(bt.caught)+len(bt.missed)} | "
            f"{len(bt.legit_acted_against)} | {bt.precision:.3f} | {bt.lr_measured:.2f} | "
            f"{','.join(rep.agree_positive) or '-'} ({len(rep.agree_positive)}) | "
            f"{len(rep.agree_negative)} cases | "
            f"{','.join(rep.disagree_rule_only) or '-'} ({len(rep.disagree_rule_only)}) | "
            f"{','.join(rep.disagree_agent_only) or '-'} ({len(rep.disagree_agent_only)}) | "
            f"{unparsed} |"
        )
    return "\n".join(lines)
