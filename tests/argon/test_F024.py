"""Reproducer for F024: _extract_txn_ids must retain all referenced transaction IDs."""

from assay.r2.untrusted import _extract_txn_ids, cross_check, parse_trigger


def test_extract_txn_ids_keeps_multiple_explicit_and_fallback():
    text = "Analyst request: Compare prior transaction 3100001 with flagged transaction 3660350"
    assert _extract_txn_ids(text) == [3100001, 3660350]

    fallback_text = "Compare 3100001 and 3660350 and 3100001"
    assert _extract_txn_ids(fallback_text) == [3100001, 3660350]


def test_cross_check_no_false_positive_when_prior_txn_mentioned_first():
    trigger_text = "Analyst request: Compare prior transaction 3100001 with flagged transaction 3660350"
    facts = parse_trigger("analyst_request", trigger_text)
    assert facts.referenced_txn_ids == [3100001, 3660350]

    case_row = {"case_id": "HHF-007", "flagged_txn_id": 3660350}
    warnings = cross_check(facts, case_row)
    assert not any("Referenced txn_id mismatch" in w for w in warnings)
