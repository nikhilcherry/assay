"""B12 -- Untrusted text parser & hostile input defense agent module.

Treats all free text as data, never as instruction.
Quarantines instructions, tag injections, and claims of authority.
Extracts strictly typed TriggerFacts from canonical templates.
Cross-checks facts against ground-truth transaction and case data.
Redacts secrets and PII (PANs, API keys, tokens) before echoing or attesting.
All regexes are guaranteed linear-time (no catastrophic backtracking).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import re
import unicodedata

import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import tx_all


@dataclass
class TriggerFacts:
    customer_denies: bool = False
    referenced_txn_ids: list[int] = field(default_factory=list)
    stated_amount: float | None = None
    stated_score: float | None = None
    stated_channel: str | None = None   # 'online' / 'in_person' only; anything else is ignored
    analyst_ask: str | None = None
    claims_of_authority: list[str] = field(default_factory=list)
    injection_spans: list[tuple[int, int, str]] = field(default_factory=list)


# Secret and PII redaction patterns (linear-time)
SECRET_PATTERNS = [
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bsk-(?:live|test|proj)-[A-Za-z0-9_-]{10,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{10,}\b"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{16}\b"), "[REDACTED_PAN]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]


def redact_secrets(text: str) -> str:
    """Scrub AWS keys, API tokens, Bearer tokens, 16-digit PANs, and SSNs while preserving dataset IDs."""
    if not text:
        return text
    out = text
    for pat, repl in SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


# Authority claims and system override patterns (linear-time, non-backtracking)
AUTHORITY_PATTERNS = [
    re.compile(r"SYSTEM NOTE:[^\n]*", re.IGNORECASE),
    re.compile(r"P\.S\.\s*(?:the\s+)?compliance[^\n]*", re.IGNORECASE),
    re.compile(r"Note from (?:customer|manager|admin|compliance|supervisor|director):[^\n]*", re.IGNORECASE),
    re.compile(r"compliance (?:team\s+)?(?:already\s+)?(?:cleared|approved|confirmed)[^\n]*", re.IGNORECASE),
    re.compile(r"prior instructions are void[^\n]*", re.IGNORECASE),
    re.compile(r"set fraud_probability\s*=\s*[^\s\n;,\]]+", re.IGNORECASE),
    re.compile(r"ignore the fraud policy[^\n]*", re.IGNORECASE),
    re.compile(r"do not file any report[^\n]*", re.IGNORECASE),
    re.compile(r"skip (?:the\s+|SAR\s+)?(?:report|filing)[^\n]*", re.IGNORECASE),
    re.compile(r"do not escalate[^\n]*", re.IGNORECASE),
    re.compile(r"mark the file closed[^\n]*", re.IGNORECASE),
    re.compile(r"classify this case as (?:legitimate|fraud)[^\n]*", re.IGNORECASE),
    re.compile(r"\b(?:CLOSE_NO_FRAUD|BLOCK_CARD|BLOCK_ALL_CARDS|FILE_REPORT|ALLOW_TRANSACTION)\b"),
    re.compile(r"approve it(?:\s+and\s+mark[^\n]*)?", re.IGNORECASE),
    re.compile(r"block card immediately[^\n]*", re.IGNORECASE),
    re.compile(r"--\s*system override[^\n]*", re.IGNORECASE),
]

# Tag injection patterns (bounded/linear)
TAG_PATTERNS = [
    re.compile(r"\[\[[\s\S]*?\]\]"),
    re.compile(r"<!--[\s\S]*?-->"),
    re.compile(r"<\/?(?:system|assistant|prompt|agent|admin|override|note|instruction)[^>]*>", re.IGNORECASE),
]

# Hidden Markdown & Base64 payload patterns (strictly linear without nested quantifiers)
MARKDOWN_PATTERN = re.compile(r"\[[^\]]{0,200}\]\((?:javascript|data|http|https):[^\)]{1,500}\)")
BASE64_CANDIDATE_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b")

# Double negation ("did NOT not make", "never didn't make") where customer actually confirms
DOUBLE_NEGATION_PATTERN = re.compile(
    r"\b(?:did\s+not|didn't|never)\s+(?:not|never|unauthorize)\s+(?:make|made|authorize|recognize)\b",
    re.IGNORECASE,
)

# Denial phrases (including negated confirmations like "cannot confirm", "do not recognize")
DENIAL_PATTERN = re.compile(
    r"\b(?:"
    r"did\s+not\s+(?:make|authorize|recognize|approve)|"
    r"do\s+not\s+(?:recognize|recognise|authorize|authorise|know|recall|remember|make)|"
    r"don't\s+(?:recognize|recognise|authorize|authorise|know|recall|remember|make)|"
    r"didn't\s+(?:make|authorize|authorise|recognize|recognise|approve)|"
    r"cannot\s+(?:confirm|recognize|recognise|verify)|"
    r"can't\s+(?:confirm|recognize|recognise|verify)|"
    r"never\s+(?:made|authorized|authorised|recognized|approved)|"
    r"not\s+(?:made|authorized|authorised|recognized|recognised|my|mine|legitimate)|"
    r"wasn't\s+(?:me|mine|authorized)|"
    r"was\s+not\s+(?:me|mine|authorized)|"
    r"unrecognized|unrecognised|unauthorized|unauthorised|stolen|dispute|fraudulent"
    r")\b",
    re.IGNORECASE,
)

# Confirmation phrases (guarded against negations and conditional phrases like "if it is legitimate")
CONFIRMATION_PATTERN = re.compile(
    r"(?<!not\s)(?<!never\s)(?<!cannot\s)(?<!can't\s)(?<!don't\s)(?<!didn't\s)(?<!no\s)"
    r"(?<!if\s)(?<!whether\s)(?<!if\sit\sis\s)(?<!if\sit's\s)(?<!whether\sit\sis\s)\b(?:"
    r"I\s+confirm|customer\s+confirms|confirms\s+it|it\s+was\s+me|"
    r"I\s+made|I\s+authorized|I\s+authorised|I\s+recognize|I\s+recognise|"
    r"is\s+authorized|is\s+authorised|was\s+authorized|was\s+authorised|"
    r"was\s+legitimate|charge\s+is\s+legitimate|purchase\s+is\s+legitimate|transaction\s+is\s+legitimate"
    r")\b",
    re.IGNORECASE,
)

# Benign non-ASCII punctuation and currency characters (not flagged as hostile homoglyphs)
BENIGN_UNICODE = {
    '\u2018', '\u2019', '\u201a', '\u201b',  # smart single quotes
    '\u201c', '\u201d', '\u201e', '\u201f',  # smart double quotes
    '\u2013', '\u2014',                      # en-dash, em-dash
    '\u2026',                                # ellipsis
    '\u00a3', '\u20ac', '\u20b9', '\u00a5',  # currency symbols (£, €, ₹, ¥)
    '\u00b0', '\u00a9', '\u00ae',            # degree, copyright, registered
}

HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j', 'ѕ': 's',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'S', 'Т': 'T', 'Х': 'X', 'І': 'I',
}

QUOTE_NORM_MAP = {
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
    '\u2013': '-', '\u2014': '-',
}


def find_unicode_anomalies(text: str) -> list[tuple[int, int, str]]:
    """Scan text for zero-width spaces, control chars, and deceptive homoglyphs."""
    spans: list[tuple[int, int, str]] = []
    invisible_chars = {
        '\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufeff',
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u00a0',
        '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
    }
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if ch in invisible_chars or (cat in ('Cf', 'Cc') and ch not in ('\n', '\r', '\t')):
            spans.append((i, i + 1, "zero_width_or_control_char"))
        elif ord(ch) > 127 and ch not in BENIGN_UNICODE:
            spans.append((i, i + 1, "unicode_homoglyph"))
    return spans


def normalize_text(text: str) -> str:
    """Normalize homoglyphs, curly quotes, and remove zero-width formatting for resilient template matching."""
    cleaned = []
    for ch in text:
        if ch in HOMOGLYPH_MAP:
            cleaned.append(HOMOGLYPH_MAP[ch])
        elif ch in QUOTE_NORM_MAP:
            cleaned.append(QUOTE_NORM_MAP[ch])
        elif ch == '\u00a0':
            cleaned.append(' ')
        elif unicodedata.category(ch) in ('Cf', 'Cc') and ch not in ('\n', '\r', '\t'):
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _is_suspicious_base64(token: str) -> bool:
    """Check if a base64-like token actually decodes to printable ASCII instructions/text."""
    if len(token) % 4 != 0:
        return False
    # Ignore pure hex hashes or tokens without base64 padding unless they decode to suspicious keywords
    try:
        raw = base64.b64decode(token, validate=True)
        decoded = raw.decode("utf-8")
    except Exception:
        return False
    if not decoded or not all(ch.isprintable() or ch in "\n\r\t" for ch in decoded):
        return False
    low = decoded.lower()
    suspicious_words = (
        "system", "note", "close", "fraud", "ignore", "policy", "report",
        "override", "approve", "legitimate", "agent", "assistant", "classify", "block", "allow"
    )
    return any(w in low for w in suspicious_words) or ("=" in token and " " in decoded)


def _extract_txn_ids(text: str) -> list[int]:
    """Extract transaction IDs referenced in text without hardcoding 3xxxxxx prefix."""
    explicit = re.findall(
        r"(?:Refers\s+to\s+#?(\d+)|(?:transaction|txn|charge|purchase)\s+#?(\d{5,12}))\b",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return list(dict.fromkeys(int(a or b) for a, b in explicit if (a or b)))
    fallback = re.findall(r"\b(\d{6,10})\b", text)
    if fallback:
        return list(dict.fromkeys(int(x) for x in fallback))
    return []


def _classify_customer_denial(msg: str) -> bool:
    """Classify whether a cleaned customer message denies (True) or confirms (False) a charge."""
    # Check double negation ("I did NOT not make it") -> confirmation
    if DOUBLE_NEGATION_PATTERN.search(msg):
        msg_without_dn = DOUBLE_NEGATION_PATTERN.sub(" I made ", msg)
        if not DENIAL_PATTERN.search(msg_without_dn):
            return False

    denial_matches = [m.group(0).strip().lower() for m in DENIAL_PATTERN.finditer(msg)]
    has_denial = bool(denial_matches)
    has_confirm = bool(CONFIRMATION_PATTERN.search(msg))

    if has_confirm and has_denial and all(m == "dispute" for m in denial_matches):
        return False
    if has_denial:
        return True
    if has_confirm:
        return False
    # Default for customer_report when customer writes in asking to check an unrecognized charge
    return True


def parse_trigger(trigger_type: str, text: str) -> TriggerFacts:
    """Parse trigger text into typed facts according to expected template.

    Quarantines instructions, tag injections, and authority claims.
    Extracts facts ONLY from the template portion of the text after stripping injections.
    Redacts any secrets/PII from recorded claims and analyst_ask.
    """
    facts = TriggerFacts()
    if not text:
        return facts

    # 1. Unicode anomaly check
    facts.injection_spans.extend(find_unicode_anomalies(text))

    # Normalize text for regex parsing
    norm_text = normalize_text(text)

    # Track spans to strip from norm_text before extracting facts
    strip_spans_norm: list[tuple[int, int]] = []

    # 2. Tag injections
    for p in TAG_PATTERNS:
        for m in p.finditer(text):
            facts.injection_spans.append((m.start(), m.end(), "tag_injection"))
            redacted = redact_secrets(m.group(0))
            if redacted not in facts.claims_of_authority:
                facts.claims_of_authority.append(redacted)
        for m in p.finditer(norm_text):
            strip_spans_norm.append((m.start(), m.end()))

    # 3. Markdown injections
    for m in MARKDOWN_PATTERN.finditer(text):
        facts.injection_spans.append((m.start(), m.end(), "markdown_injection"))
        redacted = redact_secrets(m.group(0))
        if redacted not in facts.claims_of_authority:
            facts.claims_of_authority.append(redacted)
    for m in MARKDOWN_PATTERN.finditer(norm_text):
        strip_spans_norm.append((m.start(), m.end()))

    # 4. Base64 injections (verified decodable payload, avoiding false positives on URLs/hashes)
    for m in BASE64_CANDIDATE_PATTERN.finditer(text):
        if _is_suspicious_base64(m.group(0)):
            facts.injection_spans.append((m.start(), m.end(), "base64_payload"))
    for m in BASE64_CANDIDATE_PATTERN.finditer(norm_text):
        if _is_suspicious_base64(m.group(0)):
            strip_spans_norm.append((m.start(), m.end()))

    # 5. Claims of authority
    for p in AUTHORITY_PATTERNS:
        for m in p.finditer(text):
            span_text = redact_secrets(m.group(0))
            facts.injection_spans.append((m.start(), m.end(), "authority_claim"))
            if span_text not in facts.claims_of_authority:
                facts.claims_of_authority.append(span_text)
        for m in p.finditer(norm_text):
            strip_spans_norm.append((m.start(), m.end()))

    # Build sanitized normalized text with all injection spans blanked out (preserving length/indices)
    clean_chars = list(norm_text)
    for s, e in strip_spans_norm:
        for idx in range(max(0, s), min(len(clean_chars), e)):
            if clean_chars[idx] != "\n":
                clean_chars[idx] = " "
    clean_norm = "".join(clean_chars)

    # 6. Strict template parsing based on trigger_type
    if trigger_type == "risk_score":
        m = re.search(
            r"Real-time model scored transaction (?P<txn>\d+)\s*\(\$(?P<amt>[\d,]+(?:\.\d+)?)(?:,\s*(?P<chan>[^)]+))?\)\s*at\s*(?P<score>\d+(?:\.\d+)?)\.\s*Review and decide\.?",
            norm_text,
            re.IGNORECASE,
        )
        if m:
            facts.referenced_txn_ids = [int(m.group("txn"))]
            facts.stated_amount = float(m.group("amt").replace(",", ""))
            facts.stated_score = float(m.group("score"))
            chan = re.sub(r"[\s-]+", "_", (m.group("chan") or "").strip().lower())
            facts.stated_channel = chan if chan in ("online", "in_person") else None
            template_end = m.end()
        else:
            facts.referenced_txn_ids = _extract_txn_ids(clean_norm)
            amt_m = re.search(r"\(\$([\d,]+(?:\.\d+)?)", clean_norm)
            if amt_m:
                facts.stated_amount = float(amt_m.group(1).replace(",", ""))
            score_m = re.search(r"\bat\s+(\d+(?:\.\d+)?)", clean_norm)
            if score_m:
                facts.stated_score = float(score_m.group(1))
            template_end = len(norm_text)

        # Quarantine text outside template
        if template_end < len(norm_text):
            tail = norm_text[template_end:].strip()
            if tail:
                facts.injection_spans.append((template_end, len(text), "quarantined_text"))

    elif trigger_type == "customer_report":
        # Anchor outer quote to "Refers to" or end of clean template to avoid stopping at contractions like "didn't"
        m_quoted = (
            re.search(
                r"Customer\s+(?P<cust>C\d+)\s+message:\s*(?P<q>['\"])(?P<msg>[\s\S]*?)(?P=q)\s*Refers\s+to\s*#?(?P<ref>\d+)\.?",
                clean_norm,
                re.IGNORECASE,
            )
            or re.search(
                r"Customer\s+(?P<cust>C\d+)\s+message:\s*(?P<q>['\"])(?P<msg>[\s\S]*?)\s*Refers\s+to\s*#?(?P<ref>\d+)\.?\s*(?P=q)",
                clean_norm,
                re.IGNORECASE,
            )
            or re.search(
                r"Customer\s+(?P<cust>C\d+)\s+message:\s*(?P<q>['\"])(?P<msg>[\s\S]*)(?P=q)(?P<ref>)",
                clean_norm,
                re.IGNORECASE,
            )
        )
        # Also support unquoted customer messages
        m_unquoted = None
        if not m_quoted:
            m_unquoted = re.search(
                r"Customer\s+(?P<cust>C\d+)\s+message:\s*(?P<msg>[\s\S]*?)(?:\s+Refers\s+to\s*#?(?P<ref>\d+)\.?|\s*$)",
                clean_norm,
                re.IGNORECASE,
            )

        m = m_quoted or m_unquoted
        if m:
            msg = m.group("msg").strip()
            if ( msg.startswith("'") and msg.endswith("'") ) or ( msg.startswith('"') and msg.endswith('"') ):
                msg = msg[1:-1].strip()

            if m.group("ref"):
                facts.referenced_txn_ids = [int(m.group("ref"))]
            else:
                facts.referenced_txn_ids = _extract_txn_ids(clean_norm)

            facts.customer_denies = _classify_customer_denial(msg)

            amt_m = re.search(r"\$([\d,]+(?:\.\d+)?)", msg)
            if amt_m:
                facts.stated_amount = float(amt_m.group(1).replace(",", ""))

            # Determine where the canonical template ended in norm_text
            raw_m = re.search(
                r"Customer\s+C\d+\s+message:\s*[\s\S]*?Refers\s+to\s*\d+\.?",
                norm_text,
                re.IGNORECASE,
            )
            template_end = raw_m.end() if raw_m else m.end()
        else:
            facts.referenced_txn_ids = _extract_txn_ids(clean_norm)
            facts.customer_denies = _classify_customer_denial(clean_norm)
            amt_m = re.search(r"\$([\d,]+(?:\.\d+)?)", clean_norm)
            if amt_m:
                facts.stated_amount = float(amt_m.group(1).replace(",", ""))
            template_end = len(norm_text)

        if template_end < len(norm_text):
            tail = norm_text[template_end:].strip()
            if tail:
                facts.injection_spans.append((template_end, len(text), "quarantined_text"))

    elif trigger_type == "analyst_request":
        # Match multi-line analyst request up to explicit hostile boundary (not bare "Note from branch manager")
        m = re.search(
            r"Analyst\s+request:\s*(?P<ask>[\s\S]*?)(?=$|SYSTEM NOTE:|P\.S\.\s*(?:the\s+)?compliance|Note from (?:customer|manager|admin|compliance|supervisor):|<!--|\[\[|--\s*system override)",
            norm_text,
            re.IGNORECASE,
        )
        if m:
            raw_ask = m.group("ask").strip()
            # Clean any inline tag/authority injections from analyst_ask and redact secrets
            clean_ask = clean_norm[m.start("ask"):m.end("ask")].strip()
            facts.analyst_ask = redact_secrets(clean_ask or raw_ask)
            facts.referenced_txn_ids = _extract_txn_ids(facts.analyst_ask)
            template_end = m.end()
        else:
            facts.analyst_ask = redact_secrets(clean_norm.strip())
            facts.referenced_txn_ids = _extract_txn_ids(clean_norm)
            template_end = len(norm_text)

        if template_end < len(norm_text):
            tail = norm_text[template_end:].strip()
            if tail:
                facts.injection_spans.append((template_end, len(text), "quarantined_text"))

    else:
        # Unknown trigger_type fallback
        facts.referenced_txn_ids = _extract_txn_ids(clean_norm)

    return facts


def cross_check(facts: TriggerFacts, case_row: pd.Series | dict) -> list[str]:
    """Cross-check extracted trigger facts against case and transaction records.

    Returns a list of tampering or mismatch warning strings.
    """
    warnings: list[str] = []

    flagged_id = case_row.get("flagged_txn_id")
    if pd.notna(flagged_id):
        flagged_id_int = int(flagged_id)
        if facts.referenced_txn_ids and flagged_id_int not in facts.referenced_txn_ids:
            warnings.append(
                f"Referenced txn_id mismatch: trigger refers to {facts.referenced_txn_ids}, "
                f"but case flagged_txn_id is {flagged_id_int}"
            )

    tx_df = tx_all()
    tx_match = None
    if pd.notna(flagged_id):
        tx_match = tx_df[tx_df.TransactionID == int(flagged_id)]

    if facts.stated_amount is not None:
        expected_amt = None
        if "amount" in case_row and pd.notna(case_row["amount"]):
            expected_amt = float(case_row["amount"])
        elif tx_match is not None and not tx_match.empty:
            expected_amt = float(tx_match.iloc[0]["amount"])

        if expected_amt is not None and abs(facts.stated_amount - expected_amt) > 0.01:
            warnings.append(
                f"Stated amount mismatch: trigger states ${facts.stated_amount:.2f}, "
                f"but data amount is ${expected_amt:.2f}"
            )

    if facts.stated_score is not None:
        expected_score = None
        if "risk_score" in case_row and pd.notna(case_row["risk_score"]):
            expected_score = float(case_row["risk_score"])
        elif tx_match is not None and not tx_match.empty and pd.notna(tx_match.iloc[0]["risk_score"]):
            expected_score = float(tx_match.iloc[0]["risk_score"])

        if expected_score is not None and abs(facts.stated_score - expected_score) > 0.02:
            warnings.append(
                f"Stated risk_score mismatch: trigger states {facts.stated_score}, "
                f"but data risk_score is {expected_score}"
            )

    return warnings


def attest_trigger_facts(case_id: str, facts: TriggerFacts, warnings: list[str] | None = None) -> Attestation:
    """Produce an attested record of parsed trigger facts and audit findings with secrets redacted."""
    redacted_claims = [redact_secrets(c) for c in facts.claims_of_authority]
    redacted_warnings = [redact_secrets(w) for w in (warnings or [])]
    return Attestation(
        what=f"Parsed untrusted trigger facts for case {case_id}",
        evidence=Evidence(case_ids=[case_id], txn_ids=facts.referenced_txn_ids),
        query="parse_trigger",
        params={
            "customer_denies": facts.customer_denies,
            "referenced_txn_ids": facts.referenced_txn_ids,
            "stated_amount": facts.stated_amount,
            "stated_score": facts.stated_score,
            "claims_of_authority": redacted_claims,
            "injection_spans": facts.injection_spans,
            "warnings": redacted_warnings,
        },
    )
