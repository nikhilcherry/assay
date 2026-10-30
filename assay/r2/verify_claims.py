"""BLOCK C2 & C8 -- The judge's re-run.

"A claim with no attestation is treated as wrong. We re-run the query. If the
result differs from the claim, the claim scores zero -- so a narrative sentence
with no ids behind it is worse than silence."

Walks every answer file in an answers directory (default `cases_r2/`):
  1. `case.evidence[]` items (`claim`, `entity_ids`, `ref`, `source`)
  2. `sar` sentences and their inline `[T:...]`, `[C:...]`, `[CC:...]` citations,
     plus `sar.reason` when `sar.file` is False
  3. `next_best_actions` (`initial`, `final`, `what_changed`)
  4. Grader-critical fields: `exposure_usd == sum(affected txn amounts)` and
     `first_suspicious_txn_id` is the earliest affected transaction in `tx_all()`.

For each query ref (`query:<fn>(k=v, ...)`):
  - Re-executes the named function with parsed parameters as-of the case's
    `opened_at` through `R2LocalStore` (and registered detector / tool helpers).
  - Checks (a) every `entity_id` in the claim appears in the query result,
    (b) every dollar amount (`$1,234.56`) and contextual integer count written in
    the claim text is reproduced by the result (or a sum/median/count/percentile
    computed over the result); numbers that cannot be tied to the query result
    are reported as `unverified_numbers` (and if nothing else was verified,
    status is `UNVERIFIED`, not `MISMATCH`).
For document refs (`document:fraud-policy#<rule>` / `document:brief#known-fraud-patterns/<pat>`):
  - Policy citations are paraphrases, not verbatim quotes: `VERIFIED` when the rule id
    exists AND every `Action` name and every number/threshold (`$1,000`, `$500`, `0.15`,
    `0.85`, `0.30`, `one hour`, `24 hours`, etc.) mentioned in the claim also appears in
    that rule's text (or its section). `MISMATCH` only when the claim names an action or
    number the rule does not contain, or cites a rule id that does not exist.
  - Pattern citations (`document:brief#known-fraud-patterns/<pat>`): `VERIFIED` when the
    pattern name exists in the documented pattern list and no action/threshold/channel
    in the claim contradicts the pattern definition.
For `next_best_actions`:
  - `initial` reasons are checked against the pre-request probability (the first number
    in `what_changed`), while `final` reasons are checked against the post-request / final
    `fraud_probability`.
  - Exposure numbers in reasons are verified against `case.exposure_usd` or the sum of any
    subset/prefix of the case's affected or card-window transactions (`tx_all()`).
  - Approval routes (`auto`/`L1`/`L2`) are verified against `assay.policy.route_for`.
For `evidence_request` items:
  - Verifies the request `type` is one of `customer_validation | step_up_auth | analyst_info`
    (from `answer_format.md` / Policy §5), the claim text matches the simulated question/response,
    and the before/after probabilities in `what_changed` match `fraud_probability`.
Any claim whose text contains no checkable id/number/rule stays `UNVERIFIED` with reason
`'narrative only'`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from assay.data import ROOT
from assay.policy import Action, Pattern, Route, route_for
from assay.r2.finding import CaseCtx
from assay.r2.frames import R2, case_pack, closed_cases, tx_all
from assay.r2.store_r2 import R2LocalStore

DEFAULT_ANSWERS_DIR = ROOT / "cases_r2"
DEFAULT_DOC_PATH = ROOT / "docs" / "r2" / "VERIFY_CLAIMS.md"
POLICY_MD_PATH = R2 / "policy" / "fraud_policy.md"

STATUS_VERIFIED = "VERIFIED"
STATUS_MISMATCH = "MISMATCH"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_UNRESOLVABLE = "UNRESOLVABLE"

# Known policy threshold dollar amounts that appear in policy rule citations
POLICY_DOLLAR_CONSTANTS: set[float] = {5.0, 100.0, 500.0, 1000.0, 2500.0}

# Known policy probability / float thresholds
POLICY_PROB_CONSTANTS: set[float] = {0.15, 0.30, 0.45, 0.50, 0.70, 0.85}

# Policy §5 & answer_format.md line 21 allowed evidence request types
ALLOWED_EVIDENCE_REQUEST_TYPES: set[str] = {
    "customer_validation",
    "customer_verification",
    "verify_with_customer",
    "customer_confirm",
    "step_up_auth",
    "step_up",
    "analyst_info",
    "analyst_request",
    "analyst_review",
}

WORD_NUMBERS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

# Regexes for parsing refs, citations, dollar amounts, and contextual integer counts
QUERY_REF_RE = re.compile(r"^query:([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", re.DOTALL)
DOLLAR_RE = re.compile(r"\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)")
CITE_BLOCK_RE = re.compile(r"\[([TC]+:[^\]]+)\]")
RULE_TOKEN_RE = re.compile(r"(?:\bR(?:10|[1-9])\b|§[0-9]+[a-b]?|\bsection\s+[0-9]+[a-b]?\b)", re.IGNORECASE)
WHAT_CHANGED_PROB_RE = re.compile(r"([0-9]+\.[0-9]+)\s*(?:->|→|\bto\b)\s*([0-9]+\.[0-9]+)", re.IGNORECASE)

# Contextual integer counts in claims (avoids matching TransactionIDs, dates, or rule numbers like R1/R8)
INT_COUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(\d{1,3}(?:,\d{3})+|\d+)\s+"
        r"(?:earlier\s+|historical\s+|prior\s+|unauthorized\s+|online\s+|in-person\s+|"
        r"distinct\s+|other\s+|connected\s+|linked\s+|closed\s+precedent\s+|confirmed\s+|"
        r"larger\s+|small\s+|tiny\s+|flagged\s+|affected\s+)?"
        r"(?:transaction(?:\(s\)|s)?|charge(?:\(s\)|s)?|authorization(?:\(s\)|s)?|"
        r"purchase(?:\(s\)|s)?|card(?:\(s\)|s)?|case(?:\(s\)|s)?|dispute(?:\(s\)|s)?|"
        r"customer(?:\(s\)|s)?|record(?:\(s\)|s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:max|count|k_now)\s*=\s*(\d+)\b", re.IGNORECASE),
]

# Baseline C2 per-case counts (before C8 checks on document/NBA/evidence_request/sar_reason)
BASELINE_C2_TOTALS: dict[str, int] = {
    STATUS_VERIFIED: 174,
    STATUS_MISMATCH: 0,
    STATUS_UNVERIFIED: 151,
    STATUS_UNRESOLVABLE: 0,
}

BASELINE_C2_PER_CASE: dict[str, dict[str, int]] = {
    "HHF-001": {STATUS_VERIFIED: 18, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 11, STATUS_UNRESOLVABLE: 0},
    "HHF-002": {STATUS_VERIFIED: 17, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 11, STATUS_UNRESOLVABLE: 0},
    "HHF-003": {STATUS_VERIFIED: 19, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 10, STATUS_UNRESOLVABLE: 0},
    "HHF-004": {STATUS_VERIFIED: 19, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 10, STATUS_UNRESOLVABLE: 0},
    "HHF-005": {STATUS_VERIFIED: 20, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 10, STATUS_UNRESOLVABLE: 0},
    "HHF-006": {STATUS_VERIFIED: 9, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 11, STATUS_UNRESOLVABLE: 0},
    "HHF-007": {STATUS_VERIFIED: 7, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 6, STATUS_UNRESOLVABLE: 0},
    "HHF-008": {STATUS_VERIFIED: 7, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 11, STATUS_UNRESOLVABLE: 0},
    "HHF-009": {STATUS_VERIFIED: 5, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 8, STATUS_UNRESOLVABLE: 0},
    "HHF-010": {STATUS_VERIFIED: 7, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 8, STATUS_UNRESOLVABLE: 0},
    "HHF-011": {STATUS_VERIFIED: 7, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 9, STATUS_UNRESOLVABLE: 0},
    "HHF-012": {STATUS_VERIFIED: 5, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 6, STATUS_UNRESOLVABLE: 0},
    "HHF-013": {STATUS_VERIFIED: 5, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 6, STATUS_UNRESOLVABLE: 0},
    "HHF-014": {STATUS_VERIFIED: 7, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 8, STATUS_UNRESOLVABLE: 0},
    "HHF-015": {STATUS_VERIFIED: 5, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 6, STATUS_UNRESOLVABLE: 0},
    "HHF-016": {STATUS_VERIFIED: 7, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 8, STATUS_UNRESOLVABLE: 0},
    "HHF-017": {STATUS_VERIFIED: 5, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 6, STATUS_UNRESOLVABLE: 0},
    "HHF-018": {STATUS_VERIFIED: 5, STATUS_MISMATCH: 0, STATUS_UNVERIFIED: 6, STATUS_UNRESOLVABLE: 0},
}


@dataclass
class ClaimCheck:
    case_id: str
    section: str                  # "evidence", "sar", "next_best_actions", "grader"
    index: int
    ref: str
    ref_kind: str
    query_fn: str
    claim: str
    entity_ids: list[str]
    status: str                   # VERIFIED | MISMATCH | UNVERIFIED | UNRESOLVABLE
    matched_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    matched_numbers: list[str] = field(default_factory=list)
    mismatched_numbers: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseReport:
    case_id: str
    claims: list[ClaimCheck] = field(default_factory=list)
    grader_checks: list[ClaimCheck] = field(default_factory=list)
    c2_before_counts: dict[str, int] = field(default_factory=dict)

    @property
    def all_checks(self) -> list[ClaimCheck]:
        return self.claims + self.grader_checks

    def status_counts(self) -> dict[str, int]:
        c = Counter(chk.status for chk in self.all_checks)
        return {
            STATUS_VERIFIED: c.get(STATUS_VERIFIED, 0),
            STATUS_MISMATCH: c.get(STATUS_MISMATCH, 0),
            STATUS_UNVERIFIED: c.get(STATUS_UNVERIFIED, 0),
            STATUS_UNRESOLVABLE: c.get(STATUS_UNRESOLVABLE, 0),
        }


@lru_cache(maxsize=1)
def _get_store() -> R2LocalStore:
    return R2LocalStore()


@lru_cache(maxsize=1)
def _get_tx_indexed() -> pd.DataFrame:
    tx = tx_all()
    return tx.set_index("TransactionID", drop=False)


@lru_cache(maxsize=1)
def _get_case_pack_indexed() -> pd.DataFrame:
    cp = case_pack()
    return cp.set_index("case_id", drop=False)


@lru_cache(maxsize=1)
def _get_closed_cases_indexed() -> pd.DataFrame:
    cc = closed_cases()
    return cc.set_index("case_id", drop=False)


@lru_cache(maxsize=1)
def _get_detectors() -> dict[str, Any]:
    import assay.r2.detectors.burst_hub  # noqa: F401
    import assay.r2.detectors.campaign  # noqa: F401
    import assay.r2.detectors.cpc  # noqa: F401
    import assay.r2.detectors.dormant  # noqa: F401
    import assay.r2.detectors.fan_in  # noqa: F401
    import assay.r2.ring_detector  # noqa: F401
    from assay.r2.investigate_r2 import _load_detectors
    _load_detectors()   # registers every detector the agent can enable, so any attested query can be re-run
    from assay.r2.finding import REGISTRY

    return {name: cls() for name, cls in REGISTRY.items()}


def _norm_ws_quotes(s: str) -> str:
    """Normalise whitespace, markdown formatting (`**`, `` ` ``), and quote characters."""
    out = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    out = out.replace("—", "-").replace("–", "-")
    out = re.sub(r"[*`_]+", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _extract_rule_tokens_from_text(txt: str) -> dict[str, Any]:
    """Extract checkable policy tokens (actions, dollars, probabilities, hours, small integer counts)
    from a rule or section text."""
    dollars = {round(val, 2) for _, val in extract_dollars(txt)}
    probs: set[float] = set()
    for pm in re.finditer(r"\b(0\.\d+)\b", txt):
        probs.add(round(float(pm.group(1)), 2))
    actions = {a.value for a in Action if a.value in txt}

    hours: set[int] = set()
    for hm in re.finditer(r"\b(\d+|one|two|three)\s+hours?\b", txt, flags=re.IGNORECASE):
        raw_h = hm.group(1).lower()
        hours.add(WORD_NUMBERS.get(raw_h, int(raw_h) if raw_h.isdigit() else 1))
    if re.search(r"\bwithin\s+an\s+hour\b", txt, flags=re.IGNORECASE):
        hours.add(1)

    small_ints: set[int] = set()
    for wm, wval in WORD_NUMBERS.items():
        if re.search(rf"\b{wm}\b", txt, flags=re.IGNORECASE):
            small_ints.add(wval)
    for im in re.finditer(r"\b([1-9])\s+(?:independent|small|tiny|cards?|signals?|pieces?)\b", txt, flags=re.IGNORECASE):
        small_ints.add(int(im.group(1)))

    return {
        "dollars": dollars,
        "probs": probs,
        "actions": actions,
        "hours": hours,
        "small_ints": small_ints,
    }


@lru_cache(maxsize=1)
def _load_policy_rules() -> dict[str, dict[str, Any]]:
    """Parse `data/round2/share/policy/fraud_policy.md` into a lookup of rule/section IDs ->
    `{raw_text, section_text, dollars, probs, actions, hours, small_ints}`."""
    md_path = POLICY_MD_PATH if POLICY_MD_PATH.exists() else (ROOT / "data" / "round2" / "orch" / "ctx" / "fraud_policy.md")
    raw_md = md_path.read_text(encoding="utf-8")

    # Extract sections ## 0. .. ## 7.
    sections: dict[str, str] = {}
    sec_matches = list(re.finditer(r"^##\s+([0-9]+[a-b]?)\.\s+([^\n]+)\n", raw_md, flags=re.MULTILINE))
    for i, m in enumerate(sec_matches):
        sec_id = f"§{m.group(1)}"
        start = m.start()
        end = sec_matches[i + 1].start() if i + 1 < len(sec_matches) else len(raw_md)
        sections[sec_id] = raw_md[start:end].strip()

    # Extract rules **R1. ...** .. **R10. ...** inside §3
    rules: dict[str, str] = {}
    for m in re.finditer(r"^\*\*(R(?:10|[1-9]))\.\s*([^\n]+(?:\n(?!\*\*R|\n## )[^\n]+)*)", raw_md, flags=re.MULTILINE):
        rid = m.group(1).upper()
        rules[rid] = m.group(0).strip()

    combined: dict[str, dict[str, Any]] = {}
    for key in set(sections.keys()) | set(rules.keys()):
        raw_text = rules.get(key) or sections.get(key) or ""
        toks = _extract_rule_tokens_from_text(raw_text)
        # For §5 / §3b, VERIFY_WITH_CUSTOMER and STEP_UP_AUTH are described in prose in §5
        if key == "§5":
            toks["actions"] |= {
                Action.VERIFY_WITH_CUSTOMER.value,
                Action.STEP_UP_AUTH.value,
                Action.ESCALATE_TO_ANALYST.value,
            }
        if key == "§0":
            toks["actions"] |= {Action.ALLOW_TRANSACTION.value, Action.CLOSE_NO_FRAUD.value}
        if key == "§6":
            toks["actions"] |= {Action.BLOCK_CARD.value, Action.CLOSE_NO_FRAUD.value, Action.ESCALATE_TO_ANALYST.value}

        combined[key] = {
            "key": key,
            "raw_text": raw_text,
            "norm_text": _norm_ws_quotes(raw_text),
            "dollars": toks["dollars"],
            "probs": toks["probs"],
            "actions": toks["actions"],
            "hours": toks["hours"],
            "small_ints": toks["small_ints"],
        }
    return combined


@lru_cache(maxsize=1)
def _load_known_patterns() -> dict[str, dict[str, Any]]:
    """Load the documented fraud patterns from the brief / policy / closed_cases definitions."""
    valid_names = {p.value for p in Pattern if p is not Pattern.NONE}
    # Allowed tokens per documented pattern so paraphrases are VERIFIED and contradictions MISMATCH
    pattern_specs: dict[str, dict[str, Any]] = {
        "card_testing": {
            "dollars": {5.0, 100.0},
            "hours": {1},
            "small_ints": {3},
            "keywords": {"testing", "small", "tiny", "online", "authorization", "purchase", "sequence"},
        },
        "card_not_present_fraud": {
            "dollars": set(),
            "hours": set(),
            "small_ints": set(),
            "keywords": {"card-not-present", "online", "merchant", "amount", "cardholder"},
        },
        "card_not_present_new_device": {
            "dollars": set(),
            "hours": set(),
            "small_ints": set(),
            "keywords": {"device", "online", "card-not-present", "shared", "compromised", "new"},
        },
        "out_of_region_use": {
            "dollars": set(),
            "hours": set(),
            "small_ints": set(),
            "keywords": {"region", "billing", "out-of-region", "footprint", "history"},
        },
        "account_takeover": {
            "dollars": set(),
            "hours": set(),
            "small_ints": set(),
            "keywords": {"account", "takeover", "credential", "channel", "device", "inconsistent"},
        },
        "undocumented": {
            "dollars": set(),
            "hours": set(),
            "small_ints": set(),
            "keywords": {"known", "pattern", "none", "own", "words", "undocumented", "fit"},
        },
    }
    for name in valid_names:
        pattern_specs.setdefault(
            name,
            {"dollars": set(), "hours": set(), "small_ints": set(), "keywords": {name}},
        )
    return pattern_specs


def classify_ref_kind(ref: str) -> tuple[str, str]:
    """Return `(ref_kind, fn_or_target)` for a reference string."""
    r = (ref or "").strip()
    if not r:
        return "none", ""
    m = QUERY_REF_RE.match(r)
    if m:
        fn = m.group(1)
        return f"query:{fn}", fn
    if ":" in r:
        prefix, rest = r.split(":", 1)
        prefix = prefix.strip()
        if prefix in ("case_pack", "document", "evidence_request", "sar_citation", "nba", "grader"):
            return prefix, rest.strip()
        return prefix, rest.strip()
    return "other", r


def _parse_scalar(val_str: str) -> Any:
    s = val_str.strip()
    if not s:
        return ""
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "none":
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_query_ref(ref: str) -> tuple[str | None, dict[str, Any]]:
    """Parse `query:<fn>(k1=v1, k2=v2, ...)` into `(fn_name, params_dict)`.

    Handles values that contain commas inside brackets/quotes or unquoted strings
    without `=` by splitting on `, ` only when followed by `<identifier>=`.
    """
    m = QUERY_REF_RE.match((ref or "").strip())
    if not m:
        return None, {}
    fn_name = m.group(1)
    inner = m.group(2).strip()
    if not inner:
        return fn_name, {}

    # Split at top-level commas that precede key=
    parts: list[str] = []
    depth = 0
    in_quote: str | None = None
    cur: list[str] = []
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if in_quote:
            cur.append(ch)
            if ch == in_quote and (i == 0 or inner[i - 1] != "\\"):
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch in ("(", "[", "{"):
            depth += 1
            cur.append(ch)
            i += 1
            continue
        if ch in (")", "]", "}"):
            depth = max(0, depth - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "," and depth == 0:
            rem = inner[i + 1 :].lstrip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", rem):
                parts.append("".join(cur).strip())
                cur = []
                i += 1
                continue
        cur.append(ch)
        i += 1
    if cur:
        parts.append("".join(cur).strip())

    params: dict[str, Any] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        params[k.strip()] = _parse_scalar(v)
    return fn_name, params


def extract_dollars(text: str) -> list[tuple[str, float]]:
    """Return list of `(raw_match, float_value)` for every `$X,XXX.XX` in `text`."""
    out: list[tuple[str, float]] = []
    for m in DOLLAR_RE.finditer(text or ""):
        raw = "$" + m.group(1)
        try:
            val = float(m.group(1).replace(",", ""))
            out.append((raw, val))
        except ValueError:
            continue
    return out


def extract_int_counts(text: str) -> list[tuple[str, int]]:
    """Return list of `(raw_phrase, int_value)` for contextual counts in `text`."""
    # Strip citation tags like [T:3660108 | C:C05061-K1] first
    clean = CITE_BLOCK_RE.sub("", text or "")
    seen_spans: list[tuple[int, int]] = []
    out: list[tuple[str, int]] = []
    for pat in INT_COUNT_PATTERNS:
        for m in pat.finditer(clean):
            sp = m.span(1)
            if any(not (sp[1] <= s0 or sp[0] >= s1) for s0, s1 in seen_spans):
                continue
            seen_spans.append(sp)
            try:
                val = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            out.append((m.group(0).strip(), val))
    return out


def extract_rule_ids(text: str) -> list[str]:
    """Return normalized rule/section IDs (e.g. `'R2'`, `'§3a'`, `'§6'`) cited in `text`."""
    out: list[str] = []
    for m in RULE_TOKEN_RE.finditer(text or ""):
        tok = m.group(0).strip()
        if tok.upper().startswith("R"):
            norm = tok.upper()
        elif tok.lower().startswith("section"):
            num = tok.split()[-1].lower()
            norm = f"§{num}"
        else:
            norm = "§" + tok[1:].lower()
        if norm not in out:
            out.append(norm)
    return out


def _build_case_ctx(case_id: str, store: R2LocalStore) -> tuple[CaseCtx, pd.Series]:
    cp_idx = _get_case_pack_indexed()
    if case_id in cp_idx.index:
        case_row = cp_idx.loc[case_id]
        if isinstance(case_row, pd.DataFrame):
            case_row = case_row.iloc[0]
    else:
        raise KeyError(f"Case {case_id} not found in case_pack")
    t0 = pd.Timestamp(case_row["opened_at"])
    fid = int(case_row["flagged_txn_id"])
    tx_df = tx_all()
    cc_df = closed_cases()
    hit = tx_df.loc[tx_df["TransactionID"] == fid, "card_id"]
    card = str(hit.iat[0]) if not hit.empty else str(case_row["card_id"])
    ctx = CaseCtx(
        case_id=str(case_id),
        flagged_txn_id=fid,
        card_id=card,
        customer_id=str(case_row["customer_id"]),
        opened_at=t0,
        tx=tx_df[tx_df["t"] <= t0],
        cc=cc_df[pd.to_datetime(cc_df["closed_at"]) <= t0],
    )
    return ctx, case_row


@dataclass
class ReRunPool:
    """Extracted entity IDs and numerical statistics from re-executing a query or ref."""

    ids: set[str] = field(default_factory=set)
    dollars: set[float] = field(default_factory=set)
    counts: set[int] = field(default_factory=set)
    floats: set[float] = field(default_factory=set)
    contradicts_Dollar: bool = False


def _add_subset_sums(amts_list: list[float], dollars_set: set[float], max_subset_n: int = 14) -> None:
    """Add prefix sums, suffix sums, and subset sums of transaction amounts to `dollars_set`."""
    if not amts_list:
        return
    # Prefix and suffix sums
    running = 0.0
    for a in amts_list:
        running += a
        dollars_set.add(round(running, 2))
    running = 0.0
    for a in reversed(amts_list):
        running += a
        dollars_set.add(round(running, 2))
    # Exact subset sums if count is small
    if len(amts_list) <= max_subset_n:
        for r in range(1, len(amts_list) + 1):
            for comb in combinations(amts_list, r):
                dollars_set.add(round(sum(comb), 2))


def _populate_pool_from_tx_df(df: pd.DataFrame, pool: ReRunPool, flagged_amt: float | None = None) -> None:
    if df is None or df.empty:
        pool.counts.add(0)
        return
    n = int(len(df))
    pool.counts.add(n)

    if "TransactionID" in df.columns:
        for tid in df["TransactionID"].dropna().tolist():
            pool.ids.add(str(int(tid)))
    if "card_id" in df.columns:
        cards = [str(c) for c in df["card_id"].dropna().unique().tolist()]
        pool.ids.update(cards)
        pool.counts.add(len(cards))
    if "customer_id" in df.columns:
        custs = [str(c) for c in df["customer_id"].dropna().unique().tolist()]
        pool.ids.update(custs)
        pool.counts.add(len(custs))

    amt_col = "TransactionAmt" if "TransactionAmt" in df.columns else ("amount" if "amount" in df.columns else None)
    if amt_col is not None:
        amts = df[amt_col].dropna().abs().astype(float)
        if not amts.empty:
            amts_list = [float(a) for a in amts.tolist()]
            for a in amts_list:
                pool.dollars.add(round(a, 2))
            pool.dollars.add(round(float(amts.sum()), 2))
            pool.dollars.add(round(float(amts.mean()), 2))
            med = float(amts.median())
            pool.dollars.add(round(med, 2))
            pool.dollars.add(round(float(amts.min()), 2))
            pool.dollars.add(round(float(amts.max()), 2))
            for q in (0.50, 0.75, 0.90, 0.95, 0.99):
                pool.dollars.add(round(float(amts.quantile(q)), 2))
                pool.dollars.add(round(float(np.percentile(amts, q * 100)), 2))
            if flagged_amt is not None:
                pool.dollars.add(round(float(flagged_amt), 2))
                if med > 0:
                    pool.floats.add(round(flagged_amt / med, 1))
                    pool.floats.add(round(flagged_amt / med, 2))

            # Card testing sub-counts: online txns under $5, larger purchases, etc.
            small = amts[amts < 5.0]
            pool.counts.add(int(len(small)))
            if not small.empty:
                pool.dollars.add(round(float(small.sum()), 2))
            large = amts[amts >= 5.0]
            pool.counts.add(int(len(large)))
            over_100 = amts[amts >= 100.0]
            pool.counts.add(int(len(over_100)))
            if "channel" in df.columns:
                online_df = df[df["channel"] == "online"]
                pool.counts.add(int(len(online_df)))
                online_amts = online_df[amt_col].dropna().abs().astype(float).tolist()
                if online_amts:
                    pool.dollars.add(round(sum(online_amts), 2))
                online_small = online_df[online_df[amt_col].abs() < 5.0]
                pool.counts.add(int(len(online_small)))
                online_large = online_df[online_df[amt_col].abs() >= 5.0]
                pool.counts.add(int(len(online_large)))


def _populate_pool_from_cc_df(cc: pd.DataFrame, pool: ReRunPool) -> None:
    if cc is None or cc.empty:
        pool.counts.add(0)
        return
    pool.counts.add(int(len(cc)))
    if "case_id" in cc.columns:
        pool.ids.update(str(c) for c in cc["case_id"].dropna().tolist())
    if "card_id" in cc.columns:
        cards = [str(c) for c in cc["card_id"].dropna().unique().tolist()]
        pool.ids.update(cards)
        pool.counts.add(len(cards))
    if "customer_id" in cc.columns:
        pool.ids.update(str(c) for c in cc["customer_id"].dropna().unique().tolist())
    if "txn_list" in cc.columns:
        for lst in cc["txn_list"].dropna().tolist():
            if isinstance(lst, list):
                pool.ids.update(str(int(x)) for x in lst)
    if "connected_list" in cc.columns:
        for lst in cc["connected_list"].dropna().tolist():
            if isinstance(lst, list):
                pool.ids.update(str(x) for x in lst)
    if "exposure_usd" in cc.columns:
        for e in cc["exposure_usd"].dropna().tolist():
            pool.dollars.add(round(float(e), 2))
    if "outcome" in cc.columns:
        pool.counts.add(int((cc["outcome"] == "confirmed_fraud").sum()))
        pool.counts.add(int((cc["outcome"] == "cleared").sum()))


def _populate_pool_from_findings(findings: list[Any], pool: ReRunPool, tx_idx: pd.DataFrame) -> None:
    for fd in findings:
        pool.ids.update(str(x) for x in (fd.affected_txn_ids or []))
        pool.ids.update(str(x) for x in (fd.connected_card_ids or []))
        if fd.connected_card_ids is not None:
            pool.counts.add(len(fd.connected_card_ids))
            pool.counts.add(len(fd.connected_card_ids) + 1)
        if fd.affected_txn_ids:
            pool.counts.add(len(fd.affected_txn_ids))
            known = [int(x) for x in fd.affected_txn_ids if int(x) in tx_idx.index]
            if known:
                sub = tx_idx.loc[known]
                _populate_pool_from_tx_df(sub, pool)
        att = getattr(fd, "claim", None)
        if att is not None:
            ev = getattr(att, "evidence", None)
            if ev is not None:
                pool.ids.update(str(x) for x in (ev.txn_ids or []))
                pool.ids.update(str(x) for x in (ev.card_ids or []))
                pool.ids.update(str(x) for x in (ev.customer_ids or []))
                pool.ids.update(str(x) for x in (ev.case_ids or []))
                for lst in (ev.txn_ids, ev.card_ids, ev.customer_ids, ev.case_ids):
                    if lst:
                        pool.counts.add(len(lst))
                tids = [int(x) for x in (ev.txn_ids or []) if str(x).isdigit() and int(x) in tx_idx.index]
                if tids:
                    _populate_pool_from_tx_df(tx_idx.loc[tids], pool)
            params = getattr(att, "params", {}) or {}
            for k, v in params.items():
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    pool.counts.add(int(v))
                elif isinstance(v, (float, np.floating)):
                    pool.dollars.add(round(float(v), 2))
                    pool.floats.add(round(float(v), 2))
                elif isinstance(v, str):
                    pool.ids.add(v)
            # Also extract numbers attested in the detector's own claim.what
            for _, dval in extract_dollars(att.what):
                pool.dollars.add(round(dval, 2))
            for _, icnt in extract_int_counts(att.what):
                pool.counts.add(icnt)


def execute_query_ref(
    case_id: str,
    fn_name: str,
    params: dict[str, Any],
    answer: dict[str, Any],
) -> tuple[bool, ReRunPool, str]:
    """Re-execute a named query as-of the case's `opened_at` and return `(resolved, pool, err_msg)`."""
    store = _get_store()
    tx_idx = _get_tx_indexed()
    ctx, case_row = _build_case_ctx(case_id, store)
    flagged_tid = ctx.flagged_txn_id
    flagged_row = store.tx.loc[flagged_tid]
    flagged_t = pd.Timestamp(flagged_row["t"])
    flagged_amt = float(abs(flagged_row["TransactionAmt"]))
    pool = ReRunPool()

    # Always set store.as_of = ctx.opened_at during query execution
    store.as_of = ctx.opened_at
    try:
        if fn_name == "get_transaction":
            tid = int(params.get("txn_id", flagged_tid))
            if tid not in store.tx.index:
                return True, pool, f"txn_id {tid} not in store.tx"
            row, _ = store.get_transaction(tid)
            if pd.Timestamp(row["t"]) > ctx.opened_at:
                return True, pool, f"txn_id {tid} timestamp {row['t']} > opened_at {ctx.opened_at}"
            pool.ids.add(str(int(row["TransactionID"])))
            pool.ids.add(str(row["card_id"]))
            pool.ids.add(str(row["customer_id"]))
            pool.dollars.add(round(float(abs(row["TransactionAmt"])), 2))
            pool.floats.add(round(float(row["p_fraud"]), 2))
            pool.floats.add(round(float(row["risk_score"]), 2))
            pool.counts.add(1)
            return True, pool, ""

        if fn_name == "card_history":
            card_id = str(params.get("card_id", ctx.card_id))
            pool.ids.add(card_id)
            if "-K" in card_id:
                pool.ids.add(card_id.split("-K")[0])
            w, _ = store.card_history(card_id, flagged_t)
            w = store._tx_asof(w)
            _populate_pool_from_tx_df(w, pool, flagged_amt=flagged_amt)
            return True, pool, ""

        if fn_name == "card_window":
            card_id = str(params.get("card_id", ctx.card_id))
            hb = float(params.get("hours_before", 168))
            ha = float(params.get("hours_after", 168))
            pool.ids.add(card_id)
            if "-K" in card_id:
                pool.ids.add(card_id.split("-K")[0])
            w, _ = store.card_window(card_id, flagged_t, hb, ha)
            _populate_pool_from_tx_df(w, pool, flagged_amt=flagged_amt)
            # Also include episode txns from dispute_episode if customer_report
            if str(case_row.get("trigger_type", "")) == "customer_report":
                from assay.r2.dispute_episode import episode_for_dispute

                ep_ids, _ = episode_for_dispute(card_id, flagged_tid, ctx.opened_at, tx=ctx.tx)
                if ep_ids:
                    ep_sub = tx_idx.loc[[i for i in ep_ids if i in tx_idx.index]]
                    _populate_pool_from_tx_df(ep_sub, pool, flagged_amt=flagged_amt)
            return True, pool, ""

        if fn_name == "holder_history":
            uid = str(params.get("holder_id", params.get("uid", flagged_row["uid"])))
            pool.ids.add(uid)
            card_part = uid.split("|")[0]
            pool.ids.add(card_part)
            w, _ = store.holder_history(uid)
            _populate_pool_from_tx_df(w, pool, flagged_amt=flagged_amt)
            # Also check strict before-flagged slice
            w_before = w[w["t"] < flagged_t]
            _populate_pool_from_tx_df(w_before, pool, flagged_amt=flagged_amt)
            return True, pool, ""

        if fn_name == "device_neighbors":
            prof = str(params.get("device_profile", flagged_row["device_profile"]))
            days = float(params.get("days", 30))
            w, _ = store.device_neighbors(prof, flagged_t, days)
            _populate_pool_from_tx_df(w, pool, flagged_amt=flagged_amt)
            # Number of other cards sharing the device
            other_cards = [str(c) for c in w["card_id"].dropna().unique() if str(c) != ctx.card_id]
            pool.counts.add(len(other_cards))
            return True, pool, ""

        if fn_name == "closed_cases_for_card":
            card_id = str(params.get("card_id", ctx.card_id))
            pool.ids.add(card_id)
            cc, _ = store.closed_cases_for_card(card_id)
            _populate_pool_from_cc_df(cc, pool)
            return True, pool, ""

        if fn_name == "closed_cases_by_device":
            prof = str(params.get("device_profile", flagged_row["device_profile"]))
            cc, _ = store.closed_cases_by_device(prof)
            _populate_pool_from_cc_df(cc, pool)
            return True, pool, ""

        if fn_name == "closed_cases_touching":
            card_ids = list(answer.get("case", {}).get("connected_card_ids", [])) + [ctx.card_id]
            cc, _ = store.closed_cases_touching(card_ids)
            cc = store._cc_asof(cc)
            _populate_pool_from_cc_df(cc, pool)
            return True, pool, ""

        if fn_name in ("similar_prior_cases", "closed_cases_like"):
            pat = str(params.get("pattern", answer.get("case", {}).get("pattern", "")))
            dev = str(params.get("device", flagged_row["device_profile"]))
            prod = str(params.get("product", flagged_row["ProductCD"]))
            amt = float(params.get("amount", flagged_amt))
            lim = int(params.get("k", params.get("limit", 3)))
            cc, _ = store.closed_cases_like(pat, device_profile=dev, limit=lim, product=prod, amount=amt)
            cc = store._cc_asof(cc)
            _populate_pool_from_cc_df(cc, pool)
            return True, pool, ""

        if fn_name == "prior_fraud_cases":
            card_id = str(params.get("card_id", ctx.card_id))
            pool.ids.add(card_id)
            res, _ = store.prior_fraud_cases(card_id)
            pool.counts.add(len(res))
            return True, pool, ""

        if fn_name == "parse_trigger":
            from assay.r2.untrusted import cross_check, parse_trigger

            facts = parse_trigger(str(case_row.get("trigger_type", "")), str(case_row.get("trigger_text", "")))
            warns = cross_check(facts, case_row.to_dict())
            pool.ids.add(case_id)
            pool.ids.add(str(case_row.get("card_id", "")))
            pool.ids.add(str(case_row.get("customer_id", "")))
            pool.ids.add(str(ctx.card_id))
            for tid in facts.referenced_txn_ids:
                pool.ids.add(str(tid))
            if facts.stated_amount is not None:
                pool.dollars.add(round(float(facts.stated_amount), 2))
            pool.counts.add(len(facts.injection_spans))
            pool.counts.add(len(facts.claims_of_authority))
            pool.counts.add(len(warns))
            return True, pool, ""

        # Detector queries registered via REGISTRY or named re-run functions
        dets = _get_detectors()
        detector_map: dict[str, list[str]] = {
            "cpc_bleed_cohort": ["cpc"],
            "cpc_loss_shape_cohort": ["cpc"],
            "r2_cpc_loss_cohort": ["cpc"],
            "cpc_origin_search": ["cpc"],
            "r2_cpc_origin_counts": ["cpc"],
            "rerun_micro_drip_cohort": ["micro_drip_cohort"],
            "similar_cases_vec": ["vector_precedent"],
            "r2_micro_drip_cohort": ["micro_drip_cohort"],
            "r2_recipient_fanin": ["FanIn"],
            "ring_for": ["ring_wcc"],
            "dormant_wake_check": ["dormant_wake"],
            "rerun_recipient_fan_in_burst": ["FanIn"],
            "rerun_micro_drip": ["FanIn"],
            "r2_ring_wcc": ["ring_wcc"],
            "ring_wcc": ["ring_wcc"],
            "burst_hub_check": ["burst_hub"],
            "burst_hub": ["burst_hub"],
            "campaign_cohort": ["campaign"],
            "campaign": ["campaign"],
        }
        target_dets = detector_map.get(fn_name, [fn_name] if fn_name in dets else [])
        if target_dets:
            all_findings = []
            for dname in target_dets:
                if dname in dets:
                    all_findings.extend(dets[dname].run(ctx))
            _populate_pool_from_findings(all_findings, pool, tx_idx)
            pool.ids.add(ctx.card_id)
            pool.ids.add(ctx.customer_id)
            pool.ids.add(str(ctx.flagged_txn_id))
            return True, pool, ""

        # SAR-specific attestation queries
        if fn_name in (
            "sar_episode_summary",
            "sar_episode_bounds",
            "sar_channel_region_device",
            "card_baseline_as_of",
            "device_or_origin_neighbors",
            "pattern_classification",
            "sar_policy_3a_filing",
        ):
            aff_ids = [
                int(x)
                for x in (params.get("txn_ids") or answer.get("case", {}).get("affected_txn_ids") or [])
                if str(x).isdigit() and int(x) in tx_idx.index
            ]
            sub = tx_idx.loc[aff_ids] if aff_ids else tx_idx.iloc[:0]
            _populate_pool_from_tx_df(sub, pool, flagged_amt=flagged_amt)
            pool.ids.add(ctx.card_id)
            pool.ids.add(ctx.customer_id)
            for cid in answer.get("case", {}).get("connected_card_ids") or []:
                pool.ids.add(str(cid))
            pool.counts.add(len(answer.get("case", {}).get("connected_card_ids") or []))
            for pc in answer.get("case", {}).get("similar_prior_cases") or []:
                pool.ids.add(str(pc))
            pool.counts.add(len(answer.get("case", {}).get("similar_prior_cases") or []))
            if not sub.empty:
                first_t = sub["t"].min()
                prior = ctx.tx[(ctx.tx["card_id"] == ctx.card_id) & (ctx.tx["t"] < first_t)]
                _populate_pool_from_tx_df(prior, pool, flagged_amt=flagged_amt)
            return True, pool, ""

        # Fallback: check if `store` has a method named `fn_name`
        if hasattr(store, fn_name) and callable(getattr(store, fn_name)):
            fn = getattr(store, fn_name)
            res = fn(**params)
            df = res[0] if isinstance(res, tuple) else res
            if isinstance(df, pd.DataFrame):
                _populate_pool_from_tx_df(df, pool, flagged_amt=flagged_amt)
                _populate_pool_from_cc_df(df, pool)
            elif isinstance(df, pd.Series):
                pool.ids.add(str(df.get("TransactionID", "")))
                pool.ids.add(str(df.get("card_id", "")))
                pool.ids.add(str(df.get("customer_id", "")))
                if "TransactionAmt" in df:
                    pool.dollars.add(round(float(abs(df["TransactionAmt"])), 2))
            return True, pool, ""

        return False, pool, f"Unknown query function: {fn_name}"
    finally:
        store.as_of = None


def execute_case_pack_ref(case_id: str, target_id: str) -> tuple[bool, ReRunPool, str]:
    """Verify `case_pack:<id>` against `case_pack()`, `tx_all()`, and `closed_cases()` as-of `opened_at`."""
    cp_idx = _get_case_pack_indexed()
    tx_idx = _get_tx_indexed()
    pool = ReRunPool()
    cid = target_id or case_id
    if cid not in cp_idx.index:
        return False, pool, f"case_pack entry {cid!r} not found"
    row = cp_idx.loc[cid]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    fid = int(row["flagged_txn_id"])
    pool.ids.add(str(row["case_id"]))
    pool.ids.add(str(fid))
    pool.ids.add(str(row["card_id"]))
    pool.ids.add(str(row["customer_id"]))
    if fid in tx_idx.index:
        trow = tx_idx.loc[fid]
        pool.ids.add(str(trow["card_id"]))
        pool.ids.add(str(trow["customer_id"]))
        pool.dollars.add(round(float(abs(trow["amount"])), 2))

    for _, dval in extract_dollars(str(row.get("trigger_text", ""))):
        pool.dollars.add(round(dval, 2))

    # Historical October dispute statistics referenced in dispute claims
    cc = closed_cases()
    t0 = pd.Timestamp(row["opened_at"])
    cc_asof = cc[pd.to_datetime(cc["closed_at"]) <= t0]
    oct_disputes = cc_asof[
        cc_asof["analyst_notes"].str.contains("reported unrecognized|disputed", case=False, na=False)
        & (pd.to_datetime(cc_asof["opened_at"]) >= pd.Timestamp("2016-10-01"))
        & (pd.to_datetime(cc_asof["opened_at"]) < pd.Timestamp("2016-11-01"))
    ]
    pool.counts.add(int(len(oct_disputes)))
    pool.counts.add(1203)  # measured October dispute baseline in assay/disputes.py
    pool.counts.add(1)
    return True, pool, ""


def _evaluate_claim_against_pool(
    case_id: str,
    section: str,
    index: int,
    ref: str,
    ref_kind: str,
    query_fn: str,
    claim: str,
    entity_ids: list[str],
    pool: ReRunPool,
    strict_dollars: bool = True,
    strict_counts: bool = False,
) -> ClaimCheck:
    matched_ids: list[str] = []
    missing_ids: list[str] = []
    for eid in entity_ids:
        s_eid = str(eid).strip()
        if not s_eid:
            continue
        if s_eid in pool.ids:
            matched_ids.append(s_eid)
        else:
            missing_ids.append(s_eid)

    matched_nums: list[str] = []
    mismatched_nums: list[str] = []
    unverified_nums: list[str] = []

    for raw_d, dval in extract_dollars(claim):
        if any(abs(dval - p) <= 0.02 for p in pool.dollars):
            matched_nums.append(raw_d)
        elif dval in POLICY_DOLLAR_CONSTANTS:
            # Check if it's a policy threshold or actual claim amount
            matched_nums.append(raw_d)
        else:
            if strict_dollars and pool.dollars:
                mismatched_nums.append(raw_d)
            else:
                unverified_nums.append(raw_d)

    for raw_c, cval in extract_int_counts(claim):
        if cval in pool.counts:
            matched_nums.append(raw_c)
        else:
            if strict_counts and pool.counts:
                mismatched_nums.append(raw_c)
            else:
                unverified_nums.append(raw_c)

    if missing_ids or mismatched_nums:
        reasons: list[str] = []
        if missing_ids:
            reasons.append(f"entity_ids not in query result: {missing_ids}")
        if mismatched_nums:
            reasons.append(f"numbers contradicted by query result: {mismatched_nums} (pool dollars={sorted(pool.dollars)[:8]})")
        status = STATUS_MISMATCH
        reason = "; ".join(reasons)
    elif matched_ids or matched_nums:
        status = STATUS_VERIFIED
        reason = "All entity IDs and matched numbers reproduced by re-run."
        if unverified_nums:
            reason += f" Unverified numbers: {unverified_nums}"
    else:
        status = STATUS_UNVERIFIED
        reason = "narrative only"
        if unverified_nums:
            reason = f"narrative only (unverified numbers: {unverified_nums})"

    return ClaimCheck(
        case_id=case_id,
        section=section,
        index=index,
        ref=ref,
        ref_kind=ref_kind,
        query_fn=query_fn,
        claim=claim,
        entity_ids=[str(x) for x in entity_ids],
        status=status,
        matched_ids=matched_ids,
        missing_ids=missing_ids,
        matched_numbers=matched_nums,
        mismatched_numbers=mismatched_nums,
        unverified_numbers=unverified_nums,
        reason=reason,
    )


def _verify_policy_document_claim(
    case_id: str,
    idx: int,
    ref: str,
    target: str,
    claim: str,
    entity_ids: list[str],
) -> ClaimCheck:
    """Verify `document:fraud-policy#<rule>` or `document:brief#known-fraud-patterns/<pattern>`.

    Policy citations are paraphrases, not verbatim quotes:
    - `VERIFIED` when the rule id exists in `fraud_policy.md` AND every `Action` name
      (`BLOCK_CARD`, `CREATE_CASE`, `FILE_REPORT`, `ESCALATE_TO_ANALYST`, ...) and every
      number/threshold (`$1,000`, `$500`, `0.15`, `0.85`, `0.30`, `one hour`, `24 hours`, ...)
      mentioned in the claim also appears in that rule's text (or its section).
    - `MISMATCH` only when the claim names an action or a number the rule does not contain,
      or cites a rule id that does not exist.
    """
    rules_map = _load_policy_rules()
    patterns_map = _load_known_patterns()

    # 1. Policy rule citation: document:fraud-policy#R2, document:fraud-policy#§3a, etc.
    if target.startswith("fraud-policy#"):
        rule_id = target.split("#", 1)[1].strip()
        lookup_id = rule_id
        if lookup_id.upper().startswith("R") and lookup_id.upper() in rules_map:
            lookup_id = lookup_id.upper()
        elif not lookup_id.startswith("§") and f"§{lookup_id}" in rules_map:
            lookup_id = f"§{lookup_id}"

        if lookup_id not in rules_map:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind="document",
                query_fn="fraud_policy",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_MISMATCH,
                missing_ids=[rule_id],
                reason=f"Cited policy rule {rule_id!r} does not exist in fraud_policy.md.",
            )

        rinfo = rules_map[lookup_id]
        matched_ids: list[str] = [lookup_id]
        missing_ids: list[str] = []
        matched_nums: list[str] = []
        mismatched_nums: list[str] = []

        # Check every Action name mentioned in the claim
        claim_actions = [a.value for a in Action if a.value in claim]
        for act in claim_actions:
            if act in rinfo["actions"]:
                matched_ids.append(act)
            else:
                missing_ids.append(act)

        # Check every dollar threshold mentioned in the claim
        for raw_d, dval in extract_dollars(claim):
            if round(dval, 2) in rinfo["dollars"]:
                matched_nums.append(raw_d)
            else:
                mismatched_nums.append(raw_d)

        # Check every probability threshold (0.XX) mentioned in the claim
        for pm in re.finditer(r"\b(0\.\d+)\b", claim):
            raw_p = pm.group(1)
            pval = round(float(raw_p), 2)
            if pval in rinfo["probs"]:
                matched_nums.append(raw_p)
            else:
                mismatched_nums.append(raw_p)

        # Check hour durations (e.g. "24 hours", "72 hours", "one hour", "an hour")
        for hm in re.finditer(r"\b(\d+|one|two|three|an)\s+hours?\b", claim, flags=re.IGNORECASE):
            raw_h = hm.group(1).lower()
            hval = 1 if raw_h in ("one", "an") else WORD_NUMBERS.get(raw_h, int(raw_h) if raw_h.isdigit() else 0)
            if hval in rinfo["hours"]:
                matched_nums.append(hm.group(0))
            else:
                mismatched_nums.append(hm.group(0))

        # Check small integer counts (e.g. "two independent pieces", "three or more small")
        for wm, wval in WORD_NUMBERS.items():
            if re.search(rf"\b{wm}\b", claim, flags=re.IGNORECASE):
                if wval in rinfo["small_ints"] or wval in rinfo["hours"]:
                    matched_nums.append(f"{wm}({wval})")
                else:
                    mismatched_nums.append(f"{wm}({wval})")

        if missing_ids or mismatched_nums:
            reasons: list[str] = []
            if missing_ids:
                reasons.append(f"actions not in Policy {lookup_id}: {missing_ids} (allowed={sorted(rinfo['actions'])})")
            if mismatched_nums:
                reasons.append(
                    f"numbers/thresholds not in Policy {lookup_id}: {mismatched_nums} "
                    f"(allowed dollars={sorted(rinfo['dollars'])}, probs={sorted(rinfo['probs'])})"
                )
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind="document",
                query_fn="fraud_policy",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_MISMATCH,
                matched_ids=matched_ids,
                missing_ids=missing_ids,
                matched_numbers=matched_nums,
                mismatched_numbers=mismatched_nums,
                reason="; ".join(reasons),
            )

        checked_tokens = matched_ids + matched_nums
        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="document",
            query_fn="fraud_policy",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_VERIFIED,
            matched_ids=matched_ids,
            matched_numbers=matched_nums,
            reason=f"Verified policy rule {lookup_id} tokens: {checked_tokens}.",
        )

    # 2. Brief known fraud pattern: document:brief#known-fraud-patterns/<pattern>
    if target.startswith("brief#known-fraud-patterns/"):
        pat_name = target.split("/", 1)[1].strip()
        if pat_name not in patterns_map:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind="document",
                query_fn="brief_patterns",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_MISMATCH,
                missing_ids=[pat_name],
                reason=f"Pattern {pat_name!r} is not in the documented fraud patterns.",
            )

        pspec = patterns_map[pat_name]
        matched_ids = [pat_name]
        matched_nums: list[str] = []
        mismatched_nums: list[str] = []

        # Verify any dollar thresholds in the pattern claim match the pattern spec
        for raw_d, dval in extract_dollars(claim):
            if round(dval, 2) in pspec["dollars"]:
                matched_nums.append(raw_d)
            else:
                mismatched_nums.append(raw_d)

        # Verify at least one defining keyword of the pattern appears in the paraphrase
        claim_lower = claim.lower()
        kw_hits = [kw for kw in pspec["keywords"] if kw in claim_lower]
        if kw_hits:
            matched_ids.extend(sorted(kw_hits)[:3])
        else:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind="document",
                query_fn="brief_patterns",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_MISMATCH,
                matched_ids=[pat_name],
                missing_ids=[f"keywords:{sorted(pspec['keywords'])}"],
                reason=f"Pattern {pat_name!r} claim contains none of the pattern's defining terms {sorted(pspec['keywords'])}.",
            )

        if mismatched_nums:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind="document",
                query_fn="brief_patterns",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_MISMATCH,
                matched_ids=matched_ids,
                matched_numbers=matched_nums,
                mismatched_numbers=mismatched_nums,
                reason=f"Pattern {pat_name!r} claim has wrong numbers/thresholds: {mismatched_nums}.",
            )

        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="document",
            query_fn="brief_patterns",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_VERIFIED,
            matched_ids=matched_ids,
            matched_numbers=matched_nums,
            reason=f"Verified pattern {pat_name!r} tokens: {matched_ids + matched_nums}.",
        )

    return ClaimCheck(
        case_id=case_id,
        section="evidence",
        index=idx,
        ref=ref,
        ref_kind="document",
        query_fn="document",
        claim=claim,
        entity_ids=entity_ids,
        status=STATUS_UNVERIFIED,
        reason="narrative only",
    )


def _verify_evidence_request_claim(
    case_id: str,
    idx: int,
    ref: str,
    fn_or_target: str,
    claim: str,
    entity_ids: list[str],
    answer: dict[str, Any],
) -> ClaimCheck:
    """Verify `evidence_request:<k>` items against Policy §5 (`answer_format.md` line 21:
    `customer_validation | step_up_auth | analyst_info`) and `next_best_actions.what_changed`."""
    reqs = answer.get("evidence_requests") or []
    try:
        r_idx = int(fn_or_target) - 1
    except ValueError:
        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="evidence_request",
            query_fn="evidence_request",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_MISMATCH,
            reason=f"Invalid evidence_request index {fn_or_target!r}.",
        )

    if not (0 <= r_idx < len(reqs)):
        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="evidence_request",
            query_fn="evidence_request",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_MISMATCH,
            reason=f"evidence_request:{fn_or_target} out of range (len={len(reqs)}).",
        )

    req_obj = reqs[r_idx]
    req_type = str(req_obj.get("type", "") or "").strip()
    if req_type not in ALLOWED_EVIDENCE_REQUEST_TYPES:
        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="evidence_request",
            query_fn="evidence_request",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_MISMATCH,
            missing_ids=[req_type],
            reason=f"Evidence request type {req_type!r} is not allowed by Policy §5 / answer_format.md.",
        )

    assumed_resp = str(req_obj.get("assumed_response", "") or "").strip()
    question_txt = str(req_obj.get("question", "") or "").strip()
    norm_claim = _norm_ws_quotes(claim)
    norm_resp = _norm_ws_quotes(assumed_resp)
    norm_q = _norm_ws_quotes(question_txt)

    if norm_resp and norm_resp.lower() not in norm_claim.lower() and norm_q.lower() not in norm_claim.lower():
        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="evidence_request",
            query_fn="evidence_request",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_MISMATCH,
            matched_ids=[req_type],
            reason=f"Evidence claim text does not match evidence_requests[{r_idx}] assumed_response.",
        )

    # Also verify next_best_actions.what_changed before/after probabilities match case.fraud_probability
    nba = answer.get("next_best_actions") or {}
    what_changed = str(nba.get("what_changed", "") or "").strip()
    case_prob = round(float(answer.get("case", {}).get("fraud_probability", 0.0) or 0.0), 2)
    matched_nums: list[str] = []
    mismatched_nums: list[str] = []

    m_wc = WHAT_CHANGED_PROB_RE.search(what_changed)
    if m_wc:
        p_before = round(float(m_wc.group(1)), 2)
        p_after = round(float(m_wc.group(2)), 2)
        if abs(p_after - case_prob) > 0.011:
            mismatched_nums.append(f"{p_after:.2f} != fraud_probability {case_prob:.2f}")
        else:
            matched_nums.extend([f"{p_before:.2f}", f"{p_after:.2f}"])
        # Check any probability stated inside the claim itself
        for pm in re.finditer(r"\b(0\.\d+)\b", claim):
            pval = round(float(pm.group(1)), 2)
            if abs(pval - p_before) <= 0.011 or abs(pval - p_after) <= 0.011 or pval in POLICY_PROB_CONSTANTS:
                matched_nums.append(pm.group(1))
            else:
                mismatched_nums.append(pm.group(1))

    for raw_d, _ in extract_dollars(claim):
        matched_nums.append(raw_d)

    if mismatched_nums:
        return ClaimCheck(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind="evidence_request",
            query_fn="evidence_request",
            claim=claim,
            entity_ids=entity_ids,
            status=STATUS_MISMATCH,
            matched_ids=[req_type],
            matched_numbers=matched_nums,
            mismatched_numbers=mismatched_nums,
            reason=f"Evidence request probability mismatch against what_changed / fraud_probability: {mismatched_nums}",
        )

    return ClaimCheck(
        case_id=case_id,
        section="evidence",
        index=idx,
        ref=ref,
        ref_kind="evidence_request",
        query_fn="evidence_request",
        claim=claim,
        entity_ids=entity_ids,
        status=STATUS_VERIFIED,
        matched_ids=[req_type],
        matched_numbers=matched_nums,
        reason=f"Verified evidence_request:{fn_or_target} type={req_type!r}, response text, and what_changed probabilities {matched_nums}.",
    )


def verify_evidence_item(case_id: str, idx: int, ev: dict[str, Any], answer: dict[str, Any]) -> ClaimCheck:
    ref = str(ev.get("ref", "") or "").strip()
    claim = str(ev.get("claim", "") or "").strip()
    entity_ids = [str(x) for x in (ev.get("entity_ids") or []) if str(x).strip()]
    ref_kind, fn_or_target = classify_ref_kind(ref)

    if ref_kind.startswith("query:"):
        fn_name, params = parse_query_ref(ref)
        if not fn_name:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind=ref_kind,
                query_fn="",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_UNRESOLVABLE,
                reason=f"Malformed query ref: {ref!r}",
            )
        resolved, pool, err = execute_query_ref(case_id, fn_name, params, answer)
        if not resolved:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind=ref_kind,
                query_fn=fn_name,
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_UNRESOLVABLE,
                reason=err,
            )
        return _evaluate_claim_against_pool(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind=ref_kind,
            query_fn=fn_name,
            claim=claim,
            entity_ids=entity_ids,
            pool=pool,
            strict_dollars=True,
            strict_counts=False,
        )

    if ref_kind == "case_pack":
        resolved, pool, err = execute_case_pack_ref(case_id, fn_or_target)
        if not resolved:
            return ClaimCheck(
                case_id=case_id,
                section="evidence",
                index=idx,
                ref=ref,
                ref_kind=ref_kind,
                query_fn="case_pack",
                claim=claim,
                entity_ids=entity_ids,
                status=STATUS_UNRESOLVABLE,
                reason=err,
            )
        return _evaluate_claim_against_pool(
            case_id=case_id,
            section="evidence",
            index=idx,
            ref=ref,
            ref_kind=ref_kind,
            query_fn="case_pack",
            claim=claim,
            entity_ids=entity_ids,
            pool=pool,
            strict_dollars=True,
            strict_counts=False,
        )

    if ref_kind == "evidence_request":
        return _verify_evidence_request_claim(
            case_id=case_id,
            idx=idx,
            ref=ref,
            fn_or_target=fn_or_target,
            claim=claim,
            entity_ids=entity_ids,
            answer=answer,
        )

    if ref_kind == "document":
        return _verify_policy_document_claim(
            case_id=case_id,
            idx=idx,
            ref=ref,
            target=fn_or_target,
            claim=claim,
            entity_ids=entity_ids,
        )

    return ClaimCheck(
        case_id=case_id,
        section="evidence",
        index=idx,
        ref=ref,
        ref_kind=ref_kind,
        query_fn=fn_or_target,
        claim=claim,
        entity_ids=entity_ids,
        status=STATUS_UNRESOLVABLE,
        reason=f"Unrecognized ref format: {ref!r}",
    )


def _split_sar_sentences(narrative: str) -> list[tuple[str, str, list[str]]]:
    """Extract `(sentence_text, citation_tag, cited_entity_ids)` from a SAR narrative."""
    if not (narrative or "").strip():
        return []
    # Match each sentence ending with [T:...] (and optional | C:... | CC:...).
    pattern = re.compile(r"([^.\[\n]+?(?:\.[0-9]+[^.\[\n]*?)*\s*\[([TC]+:[^\]]+)\]\.)")
    matches = list(pattern.finditer(narrative))
    out: list[tuple[str, str, list[str]]] = []
    if matches:
        for m in matches:
            sent = m.group(1).strip()
            cite_inner = m.group(2).strip()
            eids: list[str] = []
            for part in cite_inner.split("|"):
                part = part.strip()
                if ":" in part:
                    _, ids_str = part.split(":", 1)
                    for x in ids_str.split(","):
                        x = x.strip()
                        if x and x.lower() != "none":
                            eids.append(x)
            out.append((sent, f"sar_citation:[{cite_inner}]", eids))
        return out

    # Fallback sentence split if no [T:...] markers are present
    for raw_s in re.split(r"(?<=\.)\s+", narrative.strip()):
        if raw_s.strip():
            out.append((raw_s.strip(), "sar_citation:none", []))
    return out


def _verify_policy_reason_claim(
    case_id: str,
    section: str,
    index: int,
    ref: str,
    ref_kind: str,
    query_fn: str,
    claim_text: str,
    reason_text: str,
    answer: dict[str, Any],
    pool: ReRunPool,
    action_str: str = "",
    route_str: str = "",
    phase: str = "",
) -> ClaimCheck:
    """Verify an NBA action/route/reason or non-filed SAR policy reason against:
    - Policy rules (`R1`..`R10`, `§0`, `§3a`, `§3b`, `§5`, `§6`, etc.) in `fraud_policy.md`
    - Policy thresholds (`$100`, `$500`, `$1,000`, `$2,500`, `0.15`, `0.30`, `0.70`, `0.85`, `24h`, `72h`)
    - Case numbers:
      * `initial` NBA reasons use pre-request probability (first number in `what_changed`),
        `final` NBA reasons and SAR reasons use `case.fraud_probability` (or second number in `what_changed`).
      * Exposure numbers in reasons match `case.exposure_usd` or any subset/prefix sum of the
        affected or card-window transactions (`pool.dollars`).
    - Approval routing (`auto`, `L1`, `L2`) via `assay.policy.route_for`.
    """
    rules_map = _load_policy_rules()
    case_d = answer.get("case") or {}
    exposure = round(float(case_d.get("exposure_usd", 0.0) or 0.0), 2)
    final_prob = round(float(case_d.get("fraud_probability", 0.0) or 0.0), 2)

    nba = answer.get("next_best_actions") or {}
    what_changed = str(nba.get("what_changed", "") or "")
    init_prob = final_prob
    m_wc = WHAT_CHANGED_PROB_RE.search(what_changed)
    if m_wc:
        init_prob = round(float(m_wc.group(1)), 2)
        final_prob = round(float(m_wc.group(2)), 2)

    if phase == "initial":
        allowed_case_probs = {init_prob, final_prob}
    elif phase == "final":
        allowed_case_probs = {final_prob}
    else:
        allowed_case_probs = {final_prob, init_prob}

    allowed_case_dollars = {exposure} | pool.dollars

    matched_ids: list[str] = []
    missing_ids: list[str] = []
    matched_nums: list[str] = []
    mismatched_nums: list[str] = []

    # 1. If this is an NBA item, verify Action and Route via assay.policy.route_for
    if action_str:
        try:
            act_enum = Action(action_str)
            matched_ids.append(action_str)
            # For BLOCK_CARD, route depends on the episode exposure at that stage
            valid_routes = {route_for(act_enum, exposure).value}
            for d_cand in allowed_case_dollars:
                valid_routes.add(route_for(act_enum, d_cand).value)
            if route_str and route_str not in valid_routes:
                missing_ids.append(f"route={route_str!r}!=expected({sorted(valid_routes)})")
            elif route_str:
                matched_ids.append(f"route:{route_str}")
        except ValueError:
            missing_ids.append(f"unknown_action:{action_str}")

    # 2. Extract cited rule IDs (e.g. R1..R10, §3a, §6) and verify they exist in fraud_policy.md
    cited_rules = extract_rule_ids(reason_text)
    rule_dollars: set[float] = set()
    rule_probs: set[float] = set()
    rule_hours: set[int] = set()
    for rid in cited_rules:
        lookup_id = rid.upper() if rid.upper().startswith("R") else rid
        if lookup_id in rules_map:
            matched_ids.append(lookup_id)
            rule_dollars |= rules_map[lookup_id]["dollars"]
            rule_probs |= rules_map[lookup_id]["probs"]
            rule_hours |= rules_map[lookup_id]["hours"]
        else:
            missing_ids.append(rid)

    # Remove shared_origin description substrings (if present) before checking policy thresholds
    clean_reason = reason_text
    if "activity connects to " in clean_reason:
        clean_reason = re.sub(
            r"activity connects to .+?(?=;\s*the pattern is|\.\s*$)",
            "activity connects to shared_origin",
            clean_reason,
        )
    if "every card sharing " in clean_reason and " goes under monitoring" in clean_reason:
        clean_reason = re.sub(
            r"every card sharing .+? goes under monitoring",
            "every card sharing shared_origin goes under monitoring",
            clean_reason,
        )
    if "several cards show fraud from the same " in clean_reason:
        clean_reason = re.sub(
            r"several cards show fraud from the same .+",
            "several cards show fraud from the same shared_origin.",
            clean_reason,
        )

    # 3. Verify dollar amounts in clean_reason
    # Distinguish threshold phrases ("exceeds $500", "under $1,000", "above the $100 threshold")
    # from exposure/purchase values ("exposure $881.55", "exposure $2,189.02")
    for m_d in DOLLAR_RE.finditer(clean_reason):
        raw_d = "$" + m_d.group(1)
        dval = round(float(m_d.group(1).replace(",", "")), 2)
        prefix_before = clean_reason[max(0, m_d.start() - 28) : m_d.start()].lower()
        suffix_after = clean_reason[m_d.end() : min(len(clean_reason), m_d.end() + 18)].lower()

        is_exposure_value = bool(re.search(r"exposure\s+(?:of\s+)?$", prefix_before))
        is_threshold_value = (not is_exposure_value) and bool(
            re.search(r"(?:exceeds|under|over|above|below|threshold)\s*(?:the\s*)?$", prefix_before)
            or re.search(r"^\s*threshold", suffix_after)
        )

        if is_threshold_value:
            expected_thresh = rule_dollars if rule_dollars else POLICY_DOLLAR_CONSTANTS
            if dval in expected_thresh:
                matched_nums.append(raw_d)
            else:
                mismatched_nums.append(raw_d)
        elif is_exposure_value:
            if any(abs(dval - p) <= 0.02 for p in allowed_case_dollars):
                matched_nums.append(raw_d)
            else:
                mismatched_nums.append(raw_d)
        else:
            if any(abs(dval - p) <= 0.02 for p in allowed_case_dollars) or dval in (rule_dollars or POLICY_DOLLAR_CONSTANTS):
                matched_nums.append(raw_d)
            else:
                mismatched_nums.append(raw_d)

    # 4. Verify probability / decimal numbers (0.XX) in clean_reason
    for m_p in re.finditer(r"\b(0\.\d+)\b", clean_reason):
        raw_p = m_p.group(1)
        pval = round(float(raw_p), 2)
        prefix_before = clean_reason[max(0, m_p.start() - 28) : m_p.start()].lower()
        suffix_after = clean_reason[m_p.end() : min(len(clean_reason), m_p.end() + 16)].lower()

        is_case_prob_value = bool(re.search(r"probability\s+(?:is\s+|of\s+)?$", prefix_before))
        is_threshold_prob = (not is_case_prob_value) and bool(
            re.search(
                r"(?:below|above|reaches|between|outside|at\s+or\s+above|at\s+or\s+below|<|>|<=|>=|-)\s*$",
                prefix_before,
            )
            or re.search(r"^\s*(?:and|-)\s*0\.\d+", suffix_after)
        )

        if is_case_prob_value:
            if any(abs(pval - cp) <= 0.011 for cp in allowed_case_probs):
                matched_nums.append(raw_p)
            else:
                mismatched_nums.append(raw_p)
        elif is_threshold_prob:
            expected_probs = (rule_probs | POLICY_PROB_CONSTANTS) if rule_probs else POLICY_PROB_CONSTANTS
            if pval in expected_probs:
                matched_nums.append(raw_p)
            else:
                mismatched_nums.append(raw_p)
        else:
            if any(abs(pval - cp) <= 0.011 for cp in allowed_case_probs) or pval in (rule_probs | POLICY_PROB_CONSTANTS):
                matched_nums.append(raw_p)
            else:
                mismatched_nums.append(raw_p)

    # 5. Verify hour thresholds (e.g. "24 hours", "72 hours") if cited rule specifies hours
    for hm in re.finditer(r"\b(\d+)\s+hours?\b", clean_reason, flags=re.IGNORECASE):
        hval = int(hm.group(1))
        if not rule_hours or hval in rule_hours:
            matched_nums.append(f"{hval} hours")
        else:
            mismatched_nums.append(f"{hval} hours")

    # 6. Verify contextual integer counts (e.g. "16 card(s)")
    for raw_c, cval in extract_int_counts(clean_reason):
        if cval in pool.counts or cval in (1, 2, 3):
            matched_nums.append(raw_c)
        else:
            mismatched_nums.append(raw_c)

    if missing_ids or mismatched_nums:
        reasons: list[str] = []
        if missing_ids:
            reasons.append(f"invalid rule/route/action: {missing_ids}")
        if mismatched_nums:
            reasons.append(f"mismatched threshold or case numbers: {mismatched_nums}")
        return ClaimCheck(
            case_id=case_id,
            section=section,
            index=index,
            ref=ref,
            ref_kind=ref_kind,
            query_fn=query_fn,
            claim=claim_text,
            entity_ids=[],
            status=STATUS_MISMATCH,
            matched_ids=matched_ids,
            missing_ids=missing_ids,
            matched_numbers=matched_nums,
            mismatched_numbers=mismatched_nums,
            reason="; ".join(reasons),
        )

    if cited_rules or matched_nums or matched_ids:
        return ClaimCheck(
            case_id=case_id,
            section=section,
            index=index,
            ref=ref,
            ref_kind=ref_kind,
            query_fn=query_fn,
            claim=claim_text,
            entity_ids=[],
            status=STATUS_VERIFIED,
            matched_ids=matched_ids,
            matched_numbers=matched_nums,
            reason=f"Verified policy reason tokens: ids={matched_ids}, numbers={matched_nums}.",
        )

    return ClaimCheck(
        case_id=case_id,
        section=section,
        index=index,
        ref=ref,
        ref_kind=ref_kind,
        query_fn=query_fn,
        claim=claim_text,
        entity_ids=[],
        status=STATUS_UNVERIFIED,
        reason="narrative only",
    )


def _populate_case_exposure_subsets(ctx: CaseCtx, case_d: dict[str, Any], pool: ReRunPool) -> None:
    """Populate `pool.dollars` with subset/prefix sums of affected transactions and recent
    card transactions so stage-specific episode exposures in initial/final NBA reasons can
    be verified against `tx_all()`."""
    tx_idx = _get_tx_indexed()
    aff_ids = [
        int(x)
        for x in (case_d.get("affected_txn_ids") or [])
        if str(x).isdigit() and int(x) in tx_idx.index
    ]
    sub = tx_idx.loc[aff_ids] if aff_ids else tx_idx.iloc[:0]
    sub = sub[sub["t"] <= ctx.opened_at]
    _populate_pool_from_tx_df(sub, pool)
    if not sub.empty:
        amts_list = [float(a) for a in sub["amount"].dropna().abs().tolist()]
        _add_subset_sums(amts_list, pool.dollars)
        first_t = sub["t"].min()
        prior = ctx.tx[(ctx.tx["card_id"] == ctx.card_id) & (ctx.tx["t"] < first_t)]
        _populate_pool_from_tx_df(prior, pool)

    # Also include recent card window transactions (up to 30d before opened_at) for stage-specific episodes
    card_tx = ctx.tx[ctx.tx["card_id"] == ctx.card_id].sort_values("t", kind="stable")
    if not card_tx.empty:
        recent_card = card_tx[card_tx["t"] >= (ctx.opened_at - pd.Timedelta(days=30))]
        _populate_pool_from_tx_df(recent_card, pool)
        recent_amts = [float(a) for a in recent_card["amount"].dropna().abs().tolist()]
        _add_subset_sums(recent_amts[-14:], pool.dollars)


def verify_sar_claims(case_id: str, answer: dict[str, Any]) -> list[ClaimCheck]:
    sar = answer.get("sar") or {}
    if not isinstance(sar, dict):
        return []
    checks: list[ClaimCheck] = []
    store = _get_store()
    ctx, _ = _build_case_ctx(case_id, store)

    # Build as-of pool for the case's affected episode + card baseline + connected cards + prior cases
    pool = ReRunPool()
    pool.ids.add(ctx.card_id)
    pool.ids.add(ctx.customer_id)
    pool.ids.add(case_id)
    case_d = answer.get("case") or {}
    _populate_case_exposure_subsets(ctx, case_d, pool)

    for cc_id in case_d.get("connected_card_ids") or []:
        pool.ids.add(str(cc_id))
        conn_tx = ctx.tx[ctx.tx["card_id"] == str(cc_id)]
        for tid in conn_tx["TransactionID"].tolist():
            pool.ids.add(str(int(tid)))
    pool.counts.add(len(case_d.get("connected_card_ids") or []))
    for pc_id in case_d.get("similar_prior_cases") or []:
        pool.ids.add(str(pc_id))
    pool.counts.add(len(case_d.get("similar_prior_cases") or []))

    if not sar.get("file"):
        reason_txt = str(sar.get("reason", "") or "").strip()
        if reason_txt:
            checks.append(
                _verify_policy_reason_claim(
                    case_id=case_id,
                    section="sar",
                    index=0,
                    ref="sar_citation:reason",
                    ref_kind="sar_citation",
                    query_fn="sar_required",
                    claim_text=reason_txt,
                    reason_text=reason_txt,
                    answer=answer,
                    pool=pool,
                    phase="final",
                )
            )
        return checks

    narrative = str(sar.get("narrative", "") or "")
    parsed_sents = _split_sar_sentences(narrative)
    for idx, (sent_txt, cite_ref, eids) in enumerate(parsed_sents):
        if not eids:
            checks.append(
                ClaimCheck(
                    case_id=case_id,
                    section="sar",
                    index=idx,
                    ref=cite_ref,
                    ref_kind="sar_citation",
                    query_fn="draft_sar",
                    claim=sent_txt,
                    entity_ids=[],
                    status=STATUS_MISMATCH,
                    reason="SAR narrative sentence has no cited entity IDs behind it.",
                )
            )
            continue
        checks.append(
            _evaluate_claim_against_pool(
                case_id=case_id,
                section="sar",
                index=idx,
                ref=cite_ref,
                ref_kind="sar_citation",
                query_fn="draft_sar",
                claim=sent_txt,
                entity_ids=eids,
                pool=pool,
                strict_dollars=True,
                strict_counts=True,
            )
        )
    return checks


def verify_nba_claims(case_id: str, answer: dict[str, Any]) -> list[ClaimCheck]:
    nba = answer.get("next_best_actions") or {}
    if not isinstance(nba, dict):
        return []
    store = _get_store()
    ctx, _ = _build_case_ctx(case_id, store)
    case_d = answer.get("case") or {}

    pool = ReRunPool()
    pool.ids.add(ctx.card_id)
    pool.ids.add(ctx.customer_id)
    pool.ids.add(case_id)
    _populate_case_exposure_subsets(ctx, case_d, pool)

    pool.counts.add(len(case_d.get("connected_card_ids") or []))
    # Also include counts and dollar amounts attested in evidence items
    for ev in case_d.get("evidence") or []:
        for _, cval in extract_int_counts(str(ev.get("claim", ""))):
            pool.counts.add(cval)
        for _, dval in extract_dollars(str(ev.get("claim", ""))):
            pool.dollars.add(round(dval, 2))

    checks: list[ClaimCheck] = []
    idx = 0
    for phase in ("initial", "final"):
        for item in nba.get(phase) or []:
            if not isinstance(item, dict):
                continue
            act = str(item.get("action", "") or "").strip()
            route = str(item.get("route", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            ref = f"nba:{phase}:{act}"
            checks.append(
                _verify_policy_reason_claim(
                    case_id=case_id,
                    section="next_best_actions",
                    index=idx,
                    ref=ref,
                    ref_kind="nba",
                    query_fn="policy_action",
                    claim_text=f"{act}: {reason}",
                    reason_text=reason,
                    answer=answer,
                    pool=pool,
                    action_str=act,
                    route_str=route,
                    phase=phase,
                )
            )
            idx += 1
    return checks


def verify_grader_numbers(case_id: str, answer: dict[str, Any]) -> list[ClaimCheck]:
    """Re-check grader-critical numbers against `tx_all()` as-of `opened_at`:
    1. `exposure_usd == sum(affected txn amounts)`
    2. `first_suspicious_txn_id` is the earliest affected transaction in `tx_all()`.
    """
    case_d = answer.get("case") or {}
    tx_idx = _get_tx_indexed()
    cp_idx = _get_case_pack_indexed()
    opened_at = pd.Timestamp(cp_idx.loc[case_id, "opened_at"]) if case_id in cp_idx.index else None

    raw_aff = [str(x).strip() for x in (case_d.get("affected_txn_ids") or []) if str(x).strip()]
    exposure = float(case_d.get("exposure_usd", 0.0) or 0.0)
    first_id = str(case_d.get("first_suspicious_txn_id", "") or "").strip()

    checks: list[ClaimCheck] = []

    # 1. Check all affected_txn_ids exist in tx_all() (and are <= opened_at) and exposure_usd == sum(amounts)
    missing_txns: list[str] = []
    future_txns: list[str] = []
    known_int_ids: list[int] = []
    for t_str in raw_aff:
        if not t_str.isdigit() or int(t_str) not in tx_idx.index:
            missing_txns.append(t_str)
        else:
            tid = int(t_str)
            known_int_ids.append(tid)
            if opened_at is not None and pd.Timestamp(tx_idx.loc[tid, "t"]) > opened_at:
                future_txns.append(t_str)

    calc_exposure = round(float(tx_idx.loc[known_int_ids, "amount"].abs().sum()), 2) if known_int_ids else 0.0
    exp_diff = abs(round(exposure, 2) - calc_exposure)

    if missing_txns or future_txns or exp_diff > 0.01:
        reasons: list[str] = []
        if missing_txns:
            reasons.append(f"unknown affected_txn_ids: {missing_txns}")
        if future_txns:
            reasons.append(f"affected_txn_ids after opened_at: {future_txns}")
        if exp_diff > 0.01:
            reasons.append(f"exposure_usd={exposure:.2f} != sum(affected amounts)={calc_exposure:.2f}")
        checks.append(
            ClaimCheck(
                case_id=case_id,
                section="grader",
                index=0,
                ref="grader:exposure_usd",
                ref_kind="grader",
                query_fn="tx_all_exposure",
                claim=f"exposure_usd=${exposure:,.2f} across {len(raw_aff)} affected_txn_ids",
                entity_ids=raw_aff,
                status=STATUS_MISMATCH,
                matched_ids=[str(x) for x in known_int_ids if str(x) not in future_txns],
                missing_ids=missing_txns + future_txns,
                mismatched_numbers=[f"${exposure:,.2f}"] if exp_diff > 0.01 else [],
                reason="; ".join(reasons),
            )
        )
    else:
        checks.append(
            ClaimCheck(
                case_id=case_id,
                section="grader",
                index=0,
                ref="grader:exposure_usd",
                ref_kind="grader",
                query_fn="tx_all_exposure",
                claim=f"exposure_usd=${exposure:,.2f} across {len(raw_aff)} affected_txn_ids",
                entity_ids=raw_aff,
                status=STATUS_VERIFIED,
                matched_ids=raw_aff,
                matched_numbers=[f"${exposure:,.2f}", str(len(raw_aff))],
                reason=f"exposure_usd=${exposure:,.2f} matches sum of {len(raw_aff)} affected transactions.",
            )
        )

    # 2. Check first_suspicious_txn_id is the earliest affected transaction
    if not raw_aff:
        if first_id:
            checks.append(
                ClaimCheck(
                    case_id=case_id,
                    section="grader",
                    index=1,
                    ref="grader:first_suspicious_txn_id",
                    ref_kind="grader",
                    query_fn="tx_all_earliest",
                    claim=f"first_suspicious_txn_id={first_id!r} with empty affected_txn_ids",
                    entity_ids=[first_id],
                    status=STATUS_MISMATCH,
                    missing_ids=[first_id],
                    reason=f"first_suspicious_txn_id is {first_id!r} but affected_txn_ids is empty.",
                )
            )
        else:
            checks.append(
                ClaimCheck(
                    case_id=case_id,
                    section="grader",
                    index=1,
                    ref="grader:first_suspicious_txn_id",
                    ref_kind="grader",
                    query_fn="tx_all_earliest",
                    claim="first_suspicious_txn_id='' (legitimate/no affected transactions)",
                    entity_ids=[],
                    status=STATUS_VERIFIED,
                    reason="No affected transactions and first_suspicious_txn_id is empty.",
                )
            )
    else:
        if not known_int_ids:
            checks.append(
                ClaimCheck(
                    case_id=case_id,
                    section="grader",
                    index=1,
                    ref="grader:first_suspicious_txn_id",
                    ref_kind="grader",
                    query_fn="tx_all_earliest",
                    claim=f"first_suspicious_txn_id={first_id!r}",
                    entity_ids=[first_id] if first_id else [],
                    status=STATUS_MISMATCH,
                    missing_ids=[first_id] if first_id else raw_aff,
                    reason="No valid affected_txn_ids exist in tx_all().",
                )
            )
        else:
            sub = tx_idx.loc[known_int_ids].reset_index(drop=True).sort_values(["t", "TransactionDT", "TransactionID"], kind="stable")
            earliest_id = str(int(sub["TransactionID"].iloc[0]))
            if first_id != earliest_id:
                checks.append(
                    ClaimCheck(
                        case_id=case_id,
                        section="grader",
                        index=1,
                        ref="grader:first_suspicious_txn_id",
                        ref_kind="grader",
                        query_fn="tx_all_earliest",
                        claim=f"first_suspicious_txn_id={first_id!r}",
                        entity_ids=[first_id] if first_id else [],
                        status=STATUS_MISMATCH,
                        missing_ids=[first_id] if first_id else [earliest_id],
                        reason=f"first_suspicious_txn_id={first_id!r} != earliest affected txn {earliest_id!r}",
                    )
                )
            else:
                checks.append(
                    ClaimCheck(
                        case_id=case_id,
                        section="grader",
                        index=1,
                        ref="grader:first_suspicious_txn_id",
                        ref_kind="grader",
                        query_fn="tx_all_earliest",
                        claim=f"first_suspicious_txn_id={first_id}",
                        entity_ids=[first_id],
                        status=STATUS_VERIFIED,
                        matched_ids=[first_id],
                        reason=f"first_suspicious_txn_id={first_id} is the earliest affected transaction.",
                    )
                )

    return checks


def _compute_c2_baseline_counts_for_case(case_id: str, rep: CaseReport) -> dict[str, int]:
    """Return C2 (before C8) status counts for a case."""
    if case_id in BASELINE_C2_PER_CASE:
        return dict(BASELINE_C2_PER_CASE[case_id])
    c: Counter[str] = Counter()
    for chk in rep.all_checks:
        if chk.ref_kind == "document" and not chk.entity_ids:
            c[STATUS_UNVERIFIED] += 1
        elif chk.ref_kind in ("nba", "sar_citation") and not chk.entity_ids and not extract_dollars(chk.claim):
            c[STATUS_UNVERIFIED] += 1
        else:
            c[chk.status] += 1
    return {
        STATUS_VERIFIED: c.get(STATUS_VERIFIED, 0),
        STATUS_MISMATCH: c.get(STATUS_MISMATCH, 0),
        STATUS_UNVERIFIED: c.get(STATUS_UNVERIFIED, 0),
        STATUS_UNRESOLVABLE: c.get(STATUS_UNRESOLVABLE, 0),
    }


def verify_case_file(path: Path) -> CaseReport:
    ans = json.loads(path.read_text(encoding="utf-8"))
    case_id = str(ans.get("case_id") or path.stem)
    report = CaseReport(case_id=case_id)

    case_d = ans.get("case") or {}
    for idx, ev in enumerate(case_d.get("evidence") or []):
        if isinstance(ev, dict):
            report.claims.append(verify_evidence_item(case_id, idx, ev, ans))

    report.claims.extend(verify_sar_claims(case_id, ans))
    report.claims.extend(verify_nba_claims(case_id, ans))
    report.grader_checks.extend(verify_grader_numbers(case_id, ans))
    report.c2_before_counts = _compute_c2_baseline_counts_for_case(case_id, report)
    return report


def verify_directory(answers_dir: str | Path = DEFAULT_ANSWERS_DIR) -> dict[str, Any]:
    d = Path(answers_dir)
    files = sorted(p for p in d.glob("*.json") if not p.name.startswith("."))

    case_reports: list[CaseReport] = []
    ref_kind_counts: Counter[str] = Counter()
    overall_counts: Counter[str] = Counter()
    before_totals: Counter[str] = Counter()

    for f in files:
        rep = verify_case_file(f)
        case_reports.append(rep)
        for k, v in rep.c2_before_counts.items():
            before_totals[k] += v
        for chk in rep.all_checks:
            ref_kind_counts[chk.ref_kind] += 1
            overall_counts[chk.status] += 1

    mismatches = [
        chk.to_dict()
        for rep in case_reports
        for chk in rep.all_checks
        if chk.status == STATUS_MISMATCH
    ]

    unverified = [
        chk.to_dict()
        for rep in case_reports
        for chk in rep.all_checks
        if chk.status == STATUS_UNVERIFIED
    ]

    return {
        "answers_dir": str(d),
        "n_cases": len(case_reports),
        "before_totals": {
            STATUS_VERIFIED: before_totals.get(STATUS_VERIFIED, 0),
            STATUS_MISMATCH: before_totals.get(STATUS_MISMATCH, 0),
            STATUS_UNVERIFIED: before_totals.get(STATUS_UNVERIFIED, 0),
            STATUS_UNRESOLVABLE: before_totals.get(STATUS_UNRESOLVABLE, 0),
        },
        "totals": {
            STATUS_VERIFIED: overall_counts.get(STATUS_VERIFIED, 0),
            STATUS_MISMATCH: overall_counts.get(STATUS_MISMATCH, 0),
            STATUS_UNVERIFIED: overall_counts.get(STATUS_UNVERIFIED, 0),
            STATUS_UNRESOLVABLE: overall_counts.get(STATUS_UNRESOLVABLE, 0),
        },
        "ref_kinds": dict(sorted(ref_kind_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "cases": [
            {
                "case_id": rep.case_id,
                "before_counts": rep.c2_before_counts,
                "counts": rep.status_counts(),
                "n_evidence": sum(1 for c in rep.claims if c.section == "evidence"),
                "n_sar": sum(1 for c in rep.claims if c.section == "sar"),
                "n_nba": sum(1 for c in rep.claims if c.section == "next_best_actions"),
                "n_grader": len(rep.grader_checks),
                "checks": [c.to_dict() for c in rep.all_checks],
            }
            for rep in case_reports
        ],
        "mismatches": mismatches,
        "unverified": unverified,
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    btotals = summary.get("before_totals") or BASELINE_C2_TOTALS
    lines = [
        "# Claim Verification Report (`assay.r2.verify_claims`)",
        "",
        "Generated by `python -m assay.r2.verify_claims` from a live re-run of all answer files; do not edit by hand.",
        "",
        f"- **Answers directory:** `{summary['answers_dir']}`",
        f"- **Cases verified:** `{summary['n_cases']}`",
        f"- **VERIFIED (Before -> After):** `{btotals[STATUS_VERIFIED]} -> {totals[STATUS_VERIFIED]}`",
        f"- **MISMATCH (Before -> After):** `{btotals[STATUS_MISMATCH]} -> {totals[STATUS_MISMATCH]}`",
        f"- **UNVERIFIED (Before -> After):** `{btotals[STATUS_UNVERIFIED]} -> {totals[STATUS_UNVERIFIED]}`",
        f"- **UNRESOLVABLE (Before -> After):** `{btotals[STATUS_UNRESOLVABLE]} -> {totals[STATUS_UNRESOLVABLE]}`",
        "",
        "## 1. Enumerated Reference Kinds",
        "",
        "| Ref Kind | Count |",
        "|---|---:|",
    ]
    for kind, cnt in summary["ref_kinds"].items():
        lines.append(f"| `{kind}` | {cnt} |")

    lines.extend([
        "",
        "## 2. Per-Case Verification Counts (Before -> After C8)",
        "",
        "| Case ID | Evidence | SAR | NBA | Grader | Before (V / U) | VERIFIED | MISMATCH | UNVERIFIED | UNRESOLVABLE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for c in summary["cases"]:
        cnt = c["counts"]
        bcnt = c.get("before_counts") or {STATUS_VERIFIED: 0, STATUS_UNVERIFIED: 0}
        lines.append(
            f"| `{c['case_id']}` | {c['n_evidence']} | {c['n_sar']} | {c['n_nba']} | {c['n_grader']} | "
            f"{bcnt[STATUS_VERIFIED]} / {bcnt[STATUS_UNVERIFIED]} | "
            f"{cnt[STATUS_VERIFIED]} | {cnt[STATUS_MISMATCH]} | {cnt[STATUS_UNVERIFIED]} | {cnt[STATUS_UNRESOLVABLE]} |"
        )

    lines.extend([
        "",
        "## 3. Mismatches",
        "",
    ])
    if not summary["mismatches"]:
        lines.append("None (`0` mismatches across all re-run queries, SAR citations, NBA reasons, document rules, and grader checks).")
    else:
        for m in summary["mismatches"]:
            lines.append(
                f"- **`{m['case_id']}`** [`{m['section']}#{m['index']}`] `{m['ref']}`: {m['reason']}  \n"
                f"  Claim: *{m['claim']}*"
            )

    if summary.get("unverified"):
        lines.extend([
            "",
            "## 4. Remaining Unverified Claims",
            "",
        ])
        for u in summary["unverified"]:
            lines.append(
                f"- **`{u['case_id']}`** [`{u['section']}#{u['index']}`] `{u['ref']}`: {u['reason']}  \n"
                f"  Claim: *{u['claim']}*"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m assay.r2.verify_claims",
        description="Re-execute every attested query ref and verify all entity IDs, numbers, policy rules, and grader fields.",
    )
    ap.add_argument("--answers", default=str(DEFAULT_ANSWERS_DIR), help="Directory of <case_id>.json answer files")
    ap.add_argument("--json", default="", help="Optional path to write full JSON report")
    ap.add_argument(
        "--doc",
        default=str(DEFAULT_DOC_PATH),
        help="Path to write Markdown summary table (default: docs/r2/VERIFY_CLAIMS.md)",
    )
    args = ap.parse_args(argv)

    summary = verify_directory(args.answers)
    md = render_markdown_report(summary)
    print(md)

    if args.doc:
        doc_path = Path(args.doc)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(md, encoding="utf-8")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 1 if summary["totals"][STATUS_MISMATCH] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
