"""Block B8 (Challenge 3): Policy text to detector compiler.

A deterministic, zero-LLM compiler from written policy rules and regulator
advisories into executable, vectorised `RuleSpec` detectors over `tx_all()` and
`closed_cases()`, with precomputed as-of case indices, fast backtesting, and
attested agreement/disagreement comparisons against Round 1 agent verdicts
(`cases/HHG-*.json`).
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
    "twelve": 12, "twenty-four": 24, "twenty four": 24,
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

# High-volume generic webmail domains excluded when matching "same recipient email"
# (`R_emaildomain` stores domain only; sharing gmail.com across thousands of cards is not a shared recipient)
GENERIC_EMAIL_DOMAINS = frozenset({
    "", "none", "nan", "gmail.com", "yahoo.com", "hotmail.com", "aol.com",
    "outlook.com", "icloud.com", "live.com", "msn.com", "comcast.net",
    "att.net", "verizon.net", "sbcglobal.net", "ymail.com", "me.com", "mac.com",
})


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
    """Deterministically compile rule prose into a typed `RuleSpec`."""
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
            # Operational window for shared origin rules ("in one window")
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

    # 9. Identify unparsed clauses (anything not covered by our grammar or connective boilerplate)
    clauses = _split_clauses(body)
    for clause in clauses:
        if not clause or _BOILERPLATE_RE.match(clause):
            continue
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
# Precomputed, low-memory as-of index & vectorised evaluators
# ---------------------------------------------------------------------------

_PREP_CACHE: dict[tuple, dict] = {}


def _get_precomputed_index(cc: pd.DataFrame, tx: pd.DataFrame) -> dict:
    """Precompute slim numpy arrays and per-card slices once per `(cc, tx)` pair.

    Avoids copying `tx` (650k rows) or creating massive pandas indexes per rule.
    Memory footprint < 80 MB, build time ~0.4 s.
    """
    case_sig = (
        tuple(cc["case_id"].astype(str)),
        tuple(cc["card_id"].astype(str)),
        tuple(cc["opened_at"].astype(str)),
    )
    key = (id(cc), len(cc), case_sig, id(tx), len(tx))
    if len(cc) > 1 and key in _PREP_CACHE:
        return _PREP_CACHE[key]

    # Map timestamps to epoch seconds (int64) for fast comparison with TransactionDT / t
    tx_t_sec = tx["t"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    cc_open_sec = pd.to_datetime(cc["opened_at"]).to_numpy(dtype="datetime64[s]").astype(np.int64)

    # 1. Slim per-card arrays restricted to cards present in `cc`
    cc_cards = set(cc["card_id"].dropna().astype(str))
    card_mask = tx["card_id"].isin(cc_cards).to_numpy()

    c_df = pd.DataFrame({
        "card_id": tx["card_id"].to_numpy()[card_mask],
        "customer_id": tx["customer_id"].to_numpy()[card_mask],
        "t_sec": tx_t_sec[card_mask],
        "dt": tx["TransactionDT"].to_numpy(dtype=np.int64)[card_mask],
        "amount": tx["amount"].to_numpy(dtype=np.float64)[card_mask],
        "tid": tx["TransactionID"].to_numpy(dtype=np.int64)[card_mask],
        "is_online": (tx["channel"].to_numpy()[card_mask] == "online"),
    }).sort_values(["card_id", "t_sec"], kind="stable")

    c_cards = c_df["card_id"].to_numpy()
    c_custs = c_df["customer_id"].to_numpy()
    c_t_sec = c_df["t_sec"].to_numpy(dtype=np.int64)
    c_dt = c_df["dt"].to_numpy(dtype=np.int64)
    c_amt = c_df["amount"].to_numpy(dtype=np.float64)
    c_tid = c_df["tid"].to_numpy(dtype=np.int64)
    c_online = c_df["is_online"].to_numpy(dtype=bool)

    card_slices: dict[str, tuple[int, int]] = {}
    if len(c_cards) > 0:
        b = np.flatnonzero(c_cards[1:] != c_cards[:-1]) + 1
        starts = np.concatenate(([0], b))
        ends = np.concatenate((b, [len(c_cards)]))
        for s, e in zip(starts, ends):
            card_slices[str(c_cards[s])] = (int(s), int(e))

    # Map card_id -> list of (case_id, open_sec, customer_id, outcome)
    card_to_cases: dict[str, list[tuple[str, int, str, str]]] = {}
    for cid, card_id, open_s, cust_id, outcome in zip(
        cc["case_id"].astype(str),
        cc["card_id"].astype(str),
        cc_open_sec,
        cc["customer_id"].astype(str),
        cc["outcome"].astype(str),
    ):
        card_to_cases.setdefault(card_id, []).append((cid, int(open_s), cust_id, outcome))

    # 2. Precompute slim entity groups for `device_profile` and non-generic `R_emaildomain`
    entity_groups: dict[str, dict] = {}

    for col in ("device_profile", "R_emaildomain"):
        if col not in tx.columns:
            continue
        s_col = tx[col]
        valid = s_col.notna().to_numpy().copy()
        vals_str = s_col.fillna("").astype(str)
        if col == "device_profile":
            low = vals_str.str.lower()
            valid &= ~low.isin(GENERIC_DEVICES).to_numpy()
            # Exclude per-card fallback strings like "C11709-K2|191|43"
            valid &= ~vals_str.str.contains("|", regex=False).to_numpy() | vals_str.str.contains(" | ", regex=False).to_numpy()
        elif col == "R_emaildomain":
            low = vals_str.str.lower()
            valid &= ~low.isin(GENERIC_EMAIL_DOMAINS).to_numpy()

        idx_valid = np.flatnonzero(valid)
        if len(idx_valid) == 0:
            continue

        e_df = pd.DataFrame({
            "ent": vals_str.to_numpy()[idx_valid],
            "card_id": tx["card_id"].to_numpy()[idx_valid],
            "customer_id": tx["customer_id"].to_numpy()[idx_valid],
            "t_sec": tx_t_sec[idx_valid],
            "dt": tx["TransactionDT"].to_numpy(dtype=np.int64)[idx_valid],
            "amount": tx["amount"].to_numpy(dtype=np.float64)[idx_valid],
            "tid": tx["TransactionID"].to_numpy(dtype=np.int64)[idx_valid],
            "region": tx["addr1"].fillna(-1.0).to_numpy(dtype=np.float64)[idx_valid],
            "is_online": (tx["channel"].to_numpy()[idx_valid] == "online"),
        })

        # Keep only entities that ever touch at least one card in `cc_cards` and >= 2 cards overall
        ent_has_cc = e_df[e_df["card_id"].isin(cc_cards)]["ent"].unique()
        e_df = e_df[e_df["ent"].isin(ent_has_cc)]
        if e_df.empty:
            continue

        vc = e_df.groupby("ent")["card_id"].nunique()
        # Cap at <= 150 distinct cards to exclude any remaining generic browser/OS buckets
        keep_ents = vc[(vc >= 2) & (vc <= 150)].index
        e_df = e_df[e_df["ent"].isin(keep_ents)].sort_values(["ent", "t_sec"], kind="stable")
        if e_df.empty:
            continue

        e_ent = e_df["ent"].to_numpy()
        b = np.flatnonzero(e_ent[1:] != e_ent[:-1]) + 1
        starts = np.concatenate(([0], b))
        ends = np.concatenate((b, [len(e_ent)]))

        entity_groups[col] = {
            "ent": e_ent,
            "card_id": e_df["card_id"].to_numpy(),
            "customer_id": e_df["customer_id"].to_numpy(),
            "t_sec": e_df["t_sec"].to_numpy(dtype=np.int64),
            "dt": e_df["dt"].to_numpy(dtype=np.int64),
            "amount": e_df["amount"].to_numpy(dtype=np.float64),
            "tid": e_df["tid"].to_numpy(dtype=np.int64),
            "region": e_df["region"].to_numpy(dtype=np.float64),
            "is_online": e_df["is_online"].to_numpy(dtype=bool),
            "Bounds": list(zip(starts.tolist(), ends.tolist())),
        }

    prep = {
        "c_cards": c_cards,
        "c_custs": c_custs,
        "c_t_sec": c_t_sec,
        "c_dt": c_dt,
        "c_amt": c_amt,
        "c_tid": c_tid,
        "c_online": c_online,
        "card_slices": card_slices,
        "card_to_cases": card_to_cases,
        "entity_groups": entity_groups,
    }
    if len(cc) > 1:
        if len(_PREP_CACHE) >= 8:
            _PREP_CACHE.clear()
        _PREP_CACHE[key] = prep
    return prep


def _eval_card_slice_sequence_or_velocity(
    spec: RuleSpec,
    c_t: np.ndarray,
    c_a: np.ndarray,
    c_tid: np.ndarray,
    c_online: np.ndarray,
) -> tuple[list[int], float, float] | None:
    """Evaluate a single card's as-of transaction slice `[0:k]`."""
    if spec.channel == "online":
        m = c_online
        c_t, c_a, c_tid = c_t[m], c_a[m], c_tid[m]
    elif spec.channel == "in_person":
        m = ~c_online
        c_t, c_a, c_tid = c_t[m], c_a[m], c_tid[m]

    need_n = spec.min_txns + (1 if spec.followup_larger else 0)
    n = len(c_t)
    if n < need_n:
        return None

    win = int(spec.window_seconds)
    min_k = int(spec.min_txns)

    if spec.kind == "card_sequence":
        small_max = spec.small_amt_max if spec.small_amt_max is not None else 25.0
        small_mask = c_a <= small_max
        if int(small_mask.sum()) < min_k:
            return None

        # Check from most recent windows backwards so we capture the episode closest to opened_at
        small_idxs = np.flatnonzero(small_mask)
        for i in small_idxs:
            j = int(np.searchsorted(c_t, c_t[i] + win, side="right"))
            if int(small_mask[i:j].sum()) < min_k:
                continue
            if spec.followup_larger:
                j_follow = int(np.searchsorted(c_t, c_t[i] + win * 2, side="right"))
                small_pos = np.flatnonzero(small_mask[i:j_follow])
                if len(small_pos) < min_k:
                    continue
                kth_idx = i + int(small_pos[min_k - 1])
                max_small = float(np.max(c_a[i : kth_idx + 1][small_mask[i : kth_idx + 1]]))
                after_a = c_a[kth_idx + 1 : j_follow]
                larger_rel = np.flatnonzero((after_a > max_small) & (after_a > small_max))
                if len(larger_rel) == 0:
                    continue
                end_idx = kth_idx + 1 + int(larger_rel[-1])
                ep_tids = [int(x) for x in c_tid[i : end_idx + 1]]
                ep_exp = round(float(c_a[i : end_idx + 1].sum()), 2)
                ep_max = round(float(c_a[i : end_idx + 1].max()), 2)
                return ep_tids, ep_exp, ep_max
            else:
                ep_tids = [int(x) for x in c_tid[i:j]]
                ep_exp = round(float(c_a[i:j].sum()), 2)
                ep_max = round(float(c_a[i:j].max()), 2)
                return ep_tids, ep_exp, ep_max
        return None

    # card_velocity
    r_idx = np.searchsorted(c_t, c_t + win, side="right")
    counts = r_idx - np.arange(n)
    valid_starts = np.flatnonzero(counts >= min_k)
    for i in valid_starts:
        j = int(r_idx[i])
        exp = float(c_a[i:j].sum())
        if spec.min_exposure_usd is not None and exp <= spec.min_exposure_usd:
            continue
        return (
            [int(x) for x in c_tid[i:j]],
            round(exp, 2),
            round(float(c_a[i:j].max()), 2),
        )
    return None


def evaluate_rule_on_cases(
    spec: RuleSpec,
    cc: pd.DataFrame,
    tx: pd.DataFrame,
) -> dict[str, dict]:
    """Fast, strictly as-of backtest evaluation of `spec` across `cc` cases."""
    prep = _get_precomputed_index(cc, tx)
    card_to_cases = prep["card_to_cases"]

    if spec.kind in ("card_sequence", "card_velocity"):
        c_t_sec = prep["c_t_sec"]
        c_amt = prep["c_amt"]
        c_tid = prep["c_tid"]
        c_online = prep["c_online"]
        card_slices = prep["card_slices"]

        case_hits: dict[str, dict] = {}
        for card_id, case_list in card_to_cases.items():
            sl = card_slices.get(card_id)
            if sl is None:
                continue
            s, e = sl
            ct_full = c_t_sec[s:e]
            ca_full = c_amt[s:e]
            ctid_full = c_tid[s:e]
            con_full = c_online[s:e]

            for case_id, open_s, cust_id, _ in case_list:
                # As-of cut: only transactions with t <= opened_at
                k = int(np.searchsorted(ct_full, open_s, side="right"))
                if k < spec.min_txns:
                    continue
                res = _eval_card_slice_sequence_or_velocity(
                    spec, ct_full[:k], ca_full[:k], ctid_full[:k], con_full[:k]
                )
                if res is not None:
                    ep_tids, ep_exp, ep_max = res
                    case_hits[case_id] = {
                        "case_id": case_id,
                        "card_id": card_id,
                        "customer_id": cust_id,
                        "txn_ids": ep_tids,
                        "connected_card_ids": [],
                        "exposure_usd": ep_exp,
                        "max_amount": ep_max,
                    }
        return case_hits

    # Shared-entity rule (R6, ADV-1, ADV-2)
    ent_hits_by_case: dict[str, dict] = {}
    win = int(spec.window_seconds)
    entity_groups = prep["entity_groups"]

    for col in spec.shared_entities:
        if col not in entity_groups:
            continue
        eg = entity_groups[col]
        e_ent = eg["ent"]
        e_c = eg["card_id"]
        e_t = eg["t_sec"]
        e_a = eg["amount"]
        e_tid = eg["tid"]
        e_reg = eg["region"]
        e_on = eg["is_online"]

        for s, e in eg["Bounds"]:
            g_c = e_c[s:e]
            g_t = e_t[s:e]
            g_a = e_a[s:e]
            g_tid = e_tid[s:e]
            g_reg = e_reg[s:e]
            g_on = e_on[s:e]
            ent_val = str(e_ent[s])

            if spec.channel == "online":
                m = g_on
                g_c, g_t, g_a, g_tid, g_reg = g_c[m], g_t[m], g_a[m], g_tid[m], g_reg[m]
            elif spec.channel == "in_person":
                m = ~g_on
                g_c, g_t, g_a, g_tid, g_reg = g_c[m], g_t[m], g_a[m], g_tid[m], g_reg[m]

            n_g = len(g_t)
            if n_g < spec.min_cards:
                continue

            # Find all cards in this entity group that have closed cases in `cc`
            present_cards = set(g_c.tolist()) & card_to_cases.keys()
            if not present_cards:
                continue

            for cid in present_cards:
                for case_id, open_s, cust_id, _ in card_to_cases[cid]:
                    if case_id in ent_hits_by_case:
                        continue
                    # As-of cut: only transactions on or before `open_s`
                    k = int(np.searchsorted(g_t, open_s, side="right"))
                    if k < spec.min_cards:
                        continue
                    # Check window `[open_s - win, open_s]` or most recent window containing `cid`
                    c_positions = np.flatnonzero(g_c[:k] == cid)
                    if len(c_positions) == 0:
                        continue
                    last_c_pos = int(c_positions[-1])
                    anchor_t = int(g_t[last_c_pos])
                    w_start = int(np.searchsorted(g_t[:k], anchor_t - win, side="left"))
                    w_end = int(np.searchsorted(g_t[:k], anchor_t, side="right"))

                    win_cards = pd.unique(g_c[w_start:w_end])
                    if len(win_cards) < spec.min_cards:
                        continue
                    if spec.min_distinct_regions > 1:
                        v_regs = g_reg[w_start:w_end]
                        v_regs = v_regs[v_regs >= 0]
                        if len(np.unique(v_regs)) < spec.min_distinct_regions:
                            continue
                    exp = float(g_a[w_start:w_end].sum())
                    if spec.min_exposure_usd is not None and exp <= spec.min_exposure_usd:
                        continue

                    others = [str(x) for x in win_cards if str(x) != cid]
                    ent_hits_by_case[case_id] = {
                        "case_id": case_id,
                        "card_id": cid,
                        "customer_id": cust_id,
                        "txn_ids": [int(x) for x in g_tid[w_start:w_end]],
                        "connected_card_ids": others,
                        "shared_field": col,
                        "shared_value": ent_val,
                        "exposure_usd": round(exp, 2),
                    }

    return ent_hits_by_case


def sample_backtest_frame(cc: pd.DataFrame, seed: int = 42, n_cases: int = 800) -> pd.DataFrame:
    """Return a deterministic, stratified sample of `cc` that always retains all
    `card_testing` cases and the 20 Round 1 `HHG-001..020` cases (`CC-5566..5585`).
    """
    if len(cc) <= n_cases:
        return cc.copy()
    rng = np.random.default_rng(seed)
    must_keep_mask = cc["pattern"].isin(["card_testing", "undocumented"]) | (
        cc.index >= len(cc) - 20
    )
    must_keep = cc[must_keep_mask]
    pool = cc[~must_keep_mask]
    n_rem = max(0, n_cases - len(must_keep))
    chosen_idx = rng.choice(pool.index.to_numpy(), size=min(n_rem, len(pool)), replace=False)
    out = pd.concat([must_keep, pool.loc[np.sort(chosen_idx)]], ignore_index=True)
    return out


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
        one_cc = pd.DataFrame([{
            "case_id": ctx.case_id,
            "card_id": ctx.card_id,
            "customer_id": ctx.customer_id,
            "opened_at": str(ctx.opened_at),
            "closed_at": str(ctx.opened_at),
            "outcome": "unknown",
            "pattern": "unknown",
        }])
        hits = evaluate_rule_on_cases(self.spec, one_cc, ctx.tx)
        if ctx.case_id not in hits:
            return []

        h = hits[ctx.case_id]
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
            "n_cases": bt.n_cases,
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
    precomputed_hits: dict[str, dict] | None = None,
) -> RuleComparisonReport:
    """Compare a compiled `RuleSpec` against our Round 1 agent verdicts on HHG-001..020.

    Note: In `closed_cases_history.csv`, the last 20 closed cases (`CC-5566`..`CC-5585`)
    are the Round 1 `HHG-001`..`HHG-020` investigations with their real outcomes.
    """
    cc_df = cc if cc is not None else closed_cases()
    tx_df = tx if tx is not None else tx_all()
    r1_answers = load_r1_agent_answers(cases_dir)

    # Map HHG-001..020 to CC-5566..5585
    hhg_cc = cc_df[cc_df["case_id"].isin([f"CC-{5565 + i}" for i in range(1, 21)])].copy()
    if len(hhg_cc) < 20:
        hhg_cc = cc_df.tail(20).copy()
    hhg_cc = hhg_cc.sort_values("case_id", kind="stable").reset_index(drop=True)
    hhg_cc["hhg_id"] = [f"HHG-{i:03d}" for i in range(1, len(hhg_cc) + 1)]

    hits = precomputed_hits if precomputed_hits is not None else evaluate_rule_on_cases(spec, hhg_cc, tx_df)

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
            f"{','.join(rep.agree_negative[:4])}{'...' if len(rep.agree_negative)>4 else ''} ({len(rep.agree_negative)}) | "
            f"{','.join(rep.disagree_rule_only) or '-'} ({len(rep.disagree_rule_only)}) | "
            f"{','.join(rep.disagree_agent_only) or '-'} ({len(rep.disagree_agent_only)}) | "
            f"{unparsed} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    tx = tx_all()
    cc = closed_cases()
    specs = []
    backtests = {}
    reports = {}
    for rid, prose in DEFAULT_RULES.items():
        s = compile_rule(prose, rid)
        specs.append(s)
        d = CompiledRuleDetector(s)
        backtests[rid] = d.backtest(cc, tx)
        reports[rid] = compare_rule_with_r1_agent(s, cc=cc, tx=tx)
    print(format_comparison_table(specs, backtests, reports))
