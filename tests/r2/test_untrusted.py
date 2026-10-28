"""Test suite for B12 untrusted free text parsing, cross-checking, and redteam attacks."""

import json
from pathlib import Path
import pytest

from assay.r2.frames import case_pack
from assay.r2.untrusted import attest_trigger_facts, cross_check, parse_trigger

ATTACKS_PATH = Path("tests/r2/redteam/attacks.json")


def test_18_exam_cases():
    """Verify parse_trigger and cross_check across all 18 exam cases."""
    cp = case_pack()
    assert len(cp) == 18

    print("\n" + "=" * 110)
    print(f"{'Case ID':<8} | {'Trigger Type':<16} | {'Denies':<6} | {'Ref Txns':<12} | {'Amount':<8} | {'Score':<6} | {'Claims':<6} | {'Spans'}")
    print("-" * 110)

    for row in cp.itertuples():
        facts = parse_trigger(row.trigger_type, row.trigger_text)
        warnings = cross_check(facts, row._asdict())
        attestation = attest_trigger_facts(row.case_id, facts, warnings)

        # Cross-check should yield no warnings on canonical exam cases
        assert not warnings, f"Unexpected cross-check warnings on {row.case_id}: {warnings}"

        # Attestation must be valid
        att_dict = attestation.to_dict()
        assert att_dict["how"]["query"] == "parse_trigger"

        # Hostile text cases in exam pack must isolate injection claims/spans
        if row.case_id in ("HHF-004", "HHF-005", "HHF-008", "HHF-010", "HHF-014", "HHF-016"):
            assert len(facts.claims_of_authority) > 0 or len(facts.injection_spans) > 0, \
                f"Expected injection span/claim in hostile case {row.case_id}"

        # Decision facts verification for hostile cases
        if row.case_id in ("HHF-004", "HHF-005"):
            assert facts.customer_denies is True
        elif row.case_id in ("HHF-008", "HHF-010", "HHF-014", "HHF-016"):
            assert facts.customer_denies is False

        amt_str = f"${facts.stated_amount:.2f}" if facts.stated_amount is not None else "None"
        score_str = f"{facts.stated_score:.2f}" if facts.stated_score is not None else "None"
        spans_str = str(facts.injection_spans)

        print(
            f"{row.case_id:<8} | {row.trigger_type:<16} | {str(facts.customer_denies):<6} | "
            f"{str(facts.referenced_txn_ids):<12} | {amt_str:<8} | {score_str:<6} | "
            f"{len(facts.claims_of_authority):<6} | {spans_str}"
        )

    print("=" * 110 + "\n")


def test_redteam_attacks():
    """Verify untrusted parsing invariants against all >=30 redteam attacks."""
    assert ATTACKS_PATH.exists(), f"Attacks file missing at {ATTACKS_PATH}"
    attacks = json.loads(ATTACKS_PATH.read_text())
    assert len(attacks) >= 30, f"Expected >= 30 attacks, found {len(attacks)}"

    cp = case_pack().set_index("case_id")

    for attack in attacks:
        attack_id = attack["id"]
        base_id = attack["base_case_id"]
        mutated_text = attack["mutated_trigger_text"]
        expected = attack["expected_invariant"]

        assert base_id in cp.index, f"Base case {base_id} not found in case pack"
        base_row = cp.loc[base_id]
        trigger_type = base_row["trigger_type"]

        facts = parse_trigger(trigger_type, mutated_text)
        warnings = cross_check(facts, base_row)

        if "customer_denies" in expected:
            assert facts.customer_denies == expected["customer_denies"], \
                f"Attack {attack_id}: expected customer_denies={expected['customer_denies']}, got {facts.customer_denies}"

        if "referenced_txn_ids" in expected:
            assert facts.referenced_txn_ids == expected["referenced_txn_ids"], \
                f"Attack {attack_id}: expected referenced_txn_ids={expected['referenced_txn_ids']}, got {facts.referenced_txn_ids}"

        if "stated_amount" in expected:
            assert facts.stated_amount == expected["stated_amount"], \
                f"Attack {attack_id}: expected stated_amount={expected['stated_amount']}, got {facts.stated_amount}"

        if "stated_score" in expected:
            assert facts.stated_score == expected["stated_score"], \
                f"Attack {attack_id}: expected stated_score={expected['stated_score']}, got {facts.stated_score}"

        if expected.get("expects_warning"):
            assert len(warnings) > 0, f"Attack {attack_id}: expected cross_check warning, got none"
