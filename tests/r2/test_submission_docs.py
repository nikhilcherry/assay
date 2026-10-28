"""Tests for BLOCK N3 submission documents (docs/r2/RESILIENCE.md and docs/r2/INNOVATION.md).

Verifies:
1. Both submission files exist and respect their strict word limits (<= 550 and <= 700 words).
2. Required resilience, boundary, and innovation sections/citations are present.
3. Every number containing a decimal point or '%' in RESILIENCE.md and INNOVATION.md
   appears verbatim in at least one other docs/r2/*.md or docs/**/*.json file.
4. Neither file contains hardcoded secrets or credentials.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOCS_R2 = ROOT / "docs" / "r2"
RESILIENCE_PATH = DOCS_R2 / "RESILIENCE.md"
INNOVATION_PATH = DOCS_R2 / "INNOVATION.md"

# Matches numbers with a decimal point (with optional commas and trailing %/x) or integers with %
DECIMAL_OR_PCT_RE = re.compile(r"\d[\d,]*\.\d+%?|\d[\d,]*%")

SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[0-9A-Za-z]{20,}"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile("-" * 5 + r"BEGIN [A-Z ]*PRIVATE KEY" + "-" * 5),   # built at runtime so this file does not trip scanners
]


def _word_count(text: str) -> int:
    return len(text.split())


def _reference_corpus() -> dict[str, str]:
    """Load all other docs/r2/*.md and docs/**/*.json files as the ground-truth corpus."""
    corpus: dict[str, str] = {}
    for p in sorted(DOCS_R2.glob("*.md")):
        if p.name in ("RESILIENCE.md", "INNOVATION.md"):
            continue
        corpus[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    docs_root = ROOT / "docs"
    for p in sorted(docs_root.rglob("*.json")):
        corpus[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    return corpus


def test_submission_docs_exist_and_word_limits() -> None:
    assert RESILIENCE_PATH.is_file(), f"Missing {RESILIENCE_PATH}"
    assert INNOVATION_PATH.is_file(), f"Missing {INNOVATION_PATH}"

    res_text = RESILIENCE_PATH.read_text(encoding="utf-8")
    inn_text = INNOVATION_PATH.read_text(encoding="utf-8")

    res_words = _word_count(res_text)
    inn_words = _word_count(inn_text)

    print(f"\n[N3 Word Counts] RESILIENCE.md: {res_words}/550 words | INNOVATION.md: {inn_words}/700 words")

    assert 100 <= res_words <= 550, f"RESILIENCE.md word count {res_words} exceeds 550-word limit"
    assert 100 <= inn_words <= 650, f"INNOVATION.md word count {inn_words} exceeds 650-word target (700 max)"


def test_resilience_required_declarations() -> None:
    res_text = RESILIENCE_PATH.read_text(encoding="utf-8")
    res_lower = res_text.lower()

    # Dependencies
    assert "parquet" in res_lower
    assert "tigergraph" in res_lower
    assert "lightgbm" in res_lower

    # Egress & Gemini build-time disclosure
    assert "tokens = 0" in res_lower
    assert "gemini" in res_lower
    assert "saw only code and schema, never customer rows" in res_lower

    # Fail-closed, audit, approvals, idempotency
    assert "fail-closed" in res_lower or "fail closed" in res_lower
    assert "incomplete" in res_lower
    assert "audit" in res_lower and "sha256" in res_lower
    assert "approvals.py" in res_text
    assert "idempotency-key" in res_lower


def test_innovation_citations_and_numbers_match_corpus() -> None:
    res_text = RESILIENCE_PATH.read_text(encoding="utf-8")
    inn_text = INNOVATION_PATH.read_text(encoding="utf-8")
    corpus = _reference_corpus()
    assert corpus, "Expected reference files in docs/r2/*.md or docs/**/*.json"

    # Verify explicit file citations in INNOVATION.md
    for required_cite in (
        "docs/r2/B6_micro_fanin.md",
        "docs/r2/B15_rings.md",
        "docs/r2/B1_model.md",
        "docs/r2/B14_vector.md",
        "docs/r2/B28_campaign.md",
        "docs/r2/B29_innocence.md",
        "docs/r2/B34_episode.md",
        "docs/r2/N7_dispute_episode.md",
    ):
        assert required_cite in inn_text, f"INNOVATION.md must cite {required_cite}"

    print("\n[N3 Number Verification Table]")
    print(f"{'Document':<16} | {'Number Token':<14} | {'Matched Source File'}")
    print("-" * 68)

    for doc_name, text in (("RESILIENCE.md", res_text), ("INNOVATION.md", inn_text)):
        tokens = DECIMAL_OR_PCT_RE.findall(text)
        assert tokens, f"Expected measured numbers in {doc_name}"
        for tok in tokens:
            matched_files = [
                rel_path
                for rel_path, content in corpus.items()
                if tok in content or tok.replace(",", "") in content
            ]
            assert matched_files, (
                f"Number '{tok}' in {doc_name} does not appear in any docs/r2/*.md or docs/**/*.json file"
            )
            print(f"{doc_name:<16} | {tok:<14} | {matched_files[0]}")


def test_no_secrets_in_submission_docs() -> None:
    for path in (RESILIENCE_PATH, INNOVATION_PATH):
        text = path.read_text(encoding="utf-8")
        for pat in SECRET_PATTERNS:
            assert not pat.search(text), f"Secret pattern {pat.pattern} matched in {path.name}"

    try:
        from assay.r2.audit import scan_for_secrets
    except ImportError:
        scan_for_secrets = None

    if scan_for_secrets is not None:
        findings = [
            f for f in scan_for_secrets(ROOT)
            if "RESILIENCE.md" in str(f) or "INNOVATION.md" in str(f)
        ]
        assert not findings, f"Audit secret scanner flagged submission docs: {findings}"
