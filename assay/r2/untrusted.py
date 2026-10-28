"""B12 -- Untrusted text parser & hostile input defense agent module.

Treats all free text as data, never as instruction.
Quarantines instructions, tag injections, and claims of authority.
Extracts strictly typed TriggerFacts from canonical templates.
Cross-checks facts against ground-truth transaction and case data.
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
    analyst_ask: str | None = None
    claims_of_authority: list[str] = field(default_factory=list)
    injection_spans: list[tuple[int, int, str]] = field(default_factory=list)


# Authority claims and system override patterns
AUTHORITY_PATTERNS = [
    re.compile(r"SYSTEM NOTE:.*", re.IGNORECASE),
    re.compile(r"P\.S\.\s*(?:the )?compliance team.*", re.IGNORECASE),
    re.compile(r"Note from customer:.*", re.IGNORECASE),
    re.compile(r"compliance team (?:already )?(?:cleared|approved).*", re.IGNORECASE),
    re.compile(r"prior instructions are void.*", re.IGNORECASE),
    re.compile(r"set fraud_probability=.*", re.IGNORECASE),
    re.compile(r"ignore the fraud policy.*", re.IGNORECASE),
    re.compile(r"do not file any report.*", re.IGNORECASE),
    re.compile(r"skip the report.*", re.IGNORECASE),
    re.compile(r"do not escalate.*", re.IGNORECASE),
    re.compile(r"mark the file closed.*", re.IGNORECASE),
    re.compile(r"classify this case as legitimate.*", re.IGNORECASE),
    re.compile(r"CLOSE_NO_FRAUD", re.IGNORECASE),
    re.compile(r"approve it", re.IGNORECASE),
]

# Tag injection patterns
TAG_PATTERNS = [
    re.compile(r"\[\[.*?\]\]", re.DOTALL),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"<\/?(?:system|assistant|prompt|agent|admin|override|note|instruction)[^>]*>", re.IGNORECASE),
]

# Hidden Markdown & Base64 payload patterns
MARKDOWN_PATTERN = re.compile(r"\[.*?\]\((?:javascript|data|http|https):[^\)]+\)|\[\]\([^\)]+\)")
BASE64_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

# Denial and confirmation phrases for customer messages
DENIAL_PATTERN = re.compile(
    r"\b(?:did not|never|unrecognized|unauthorized|unauthorised|stolen|not my|not mine|didn't|dispute)\b",
    re.IGNORECASE,
)
CONFIRMATION_PATTERN = re.compile(
    r"\b(?:confirm|confirms|it was me|I made|authorized|legitimate|I recognize)\b",
    re.IGNORECASE,
)


def find_unicode_anomalies(text: str) -> list[tuple[int, int, str]]:
    """Scan text for zero-width spaces, control chars, and homoglyphs."""
    spans: list[tuple[int, int, str]] = []
    invisible_chars = {
        '\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufeff',
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u00a0'
    }
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if ch in invisible_chars or (cat in ('Cf', 'Cc') and ch not in ('\n', '\r', '\t')):
            spans.append((i, i + 1, "zero_width_or_control_char"))
        elif ord(ch) > 127:
            spans.append((i, i + 1, "unicode_homoglyph"))
    return spans


def normalize_text(text: str) -> str:
    """Normalize homoglyphs and remove zero-width formatting for resilient template matching."""
    homoglyphs = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
        'Р': 'P', 'С': 'S', 'Т': 'T', 'Х': 'X'
    }
    cleaned = []
    for ch in text:
        if ch in homoglyphs:
            cleaned.append(homoglyphs[ch])
        elif unicodedata.category(ch) in ('Cf', 'Cc') and ch not in ('\n', '\r', '\t'):
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def parse_trigger(trigger_type: str, text: str) -> TriggerFacts:
    """Parse trigger text into typed facts according to expected template.
    
    Quarantines instructions, tag injections, and authority claims.
    Extracts facts ONLY from the template portion of the text.
    """
    facts = TriggerFacts()

    # 1. Unicode anomaly check
    facts.injection_spans.extend(find_unicode_anomalies(text))

    # Normalize text for regex parsing
    norm_text = normalize_text(text)

    # 2. Tag injections
    for p in TAG_PATTERNS:
        for m in p.finditer(text):
            facts.injection_spans.append((m.start(), m.end(), "tag_injection"))
            facts.claims_of_authority.append(m.group(0))

    # 3. Markdown injections
    for m in MARKDOWN_PATTERN.finditer(text):
        facts.injection_spans.append((m.start(), m.end(), "markdown_injection"))
        facts.claims_of_authority.append(m.group(0))

    # 4. Base64 injections
    for m in BASE64_PATTERN.finditer(text):
        facts.injection_spans.append((m.start(), m.end(), "base64_payload"))

    # 5. Claims of authority
    for p in AUTHORITY_PATTERNS:
        for m in p.finditer(text):
            span_text = m.group(0)
            facts.injection_spans.append((m.start(), m.end(), "authority_claim"))
            if span_text not in facts.claims_of_authority:
                facts.claims_of_authority.append(span_text)

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
            template_end = m.end()
        else:
            tx_ids = re.findall(r"\b3\d{6}\b", norm_text)
            if tx_ids:
                facts.referenced_txn_ids = [int(tx_ids[0])]
            amt_m = re.search(r"\(\$([\d,]+(?:\.\d+)?)", norm_text)
            if amt_m:
                facts.stated_amount = float(amt_m.group(1).replace(",", ""))
            score_m = re.search(r"at (\d+(?:\.\d+)?)", norm_text)
            if score_m:
                facts.stated_score = float(score_m.group(1))
            template_end = len(norm_text)

        # Quarantine text outside template
        if template_end < len(text):
            tail = text[template_end:].strip()
            if tail:
                facts.injection_spans.append((template_end, len(text), "quarantined_text"))

    elif trigger_type == "customer_report":
        m = re.search(
            r"Customer (?P<cust>C\d+) message:\s*['\"](?P<msg>.*?)['\"](?:\s*Refers to\s*(?P<ref>\d+)\.?)?",
            norm_text,
            re.IGNORECASE,
        )
        if m:
            msg = m.group("msg")
            if m.group("ref"):
                facts.referenced_txn_ids = [int(m.group("ref"))]
            else:
                tx_ids = re.findall(r"\b3\d{6}\b", norm_text)
                if tx_ids:
                    facts.referenced_txn_ids = [int(tx_ids[0])]

            # Extract facts solely from quoted customer message
            if DENIAL_PATTERN.search(msg):
                facts.customer_denies = True
            elif CONFIRMATION_PATTERN.search(msg):
                facts.customer_denies = False
            else:
                facts.customer_denies = True

            amt_m = re.search(r"\$([\d,]+(?:\.\d+)?)", msg)
            if amt_m:
                facts.stated_amount = float(amt_m.group(1).replace(",", ""))

            template_end = m.end()
        else:
            tx_ids = re.findall(r"\b3\d{6}\b", norm_text)
            if tx_ids:
                facts.referenced_txn_ids = [int(tx_ids[0])]
            facts.customer_denies = True
            template_end = len(norm_text)

        if template_end < len(text):
            tail = text[template_end:].strip()
            if tail:
                facts.injection_spans.append((template_end, len(text), "quarantined_text"))

    elif trigger_type == "analyst_request":
        m = re.search(
            r"Analyst request:\s*(?P<ask>.*?)(?=$|\n|SYSTEM NOTE:|P\.S\.|Note from|<!--|\[\[)",
            norm_text,
            re.IGNORECASE,
        )
        if m:
            facts.analyst_ask = m.group("ask").strip()
            tx_ids = re.findall(r"\b3\d{6}\b", facts.analyst_ask)
            if tx_ids:
                facts.referenced_txn_ids = [int(tx_ids[0])]
            template_end = m.end()
        else:
            facts.analyst_ask = norm_text
            tx_ids = re.findall(r"\b3\d{6}\b", norm_text)
            if tx_ids:
                facts.referenced_txn_ids = [int(tx_ids[0])]
            template_end = len(norm_text)

        if template_end < len(text):
            tail = text[template_end:].strip()
            if tail:
                facts.injection_spans.append((template_end, len(text), "quarantined_text"))

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
    """Produce an attested record of parsed trigger facts and audit findings."""
    return Attestation(
        what=f"Parsed untrusted trigger facts for case {case_id}",
        evidence=Evidence(case_ids=[case_id], txn_ids=facts.referenced_txn_ids),
        query="parse_trigger",
        params={
            "customer_denies": facts.customer_denies,
            "referenced_txn_ids": facts.referenced_txn_ids,
            "stated_amount": facts.stated_amount,
            "stated_score": facts.stated_score,
            "claims_of_authority": facts.claims_of_authority,
            "injection_spans": facts.injection_spans,
            "warnings": warnings or [],
        },
    )
