"""Unit tests for BLOCK B20 Analyst in the loop (challenge 8)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from assay.r2.attest import Attestation
from assay.r2.feedback import (
    FeedbackDecision,
    FeedbackStore,
    compute_finding_signature,
    extract_signature_and_elements,
    get_amount_band,
    run_feedback_demonstration,
)
from assay.r2.finding import Finding


def test_signature_computation() -> None:
    sig1 = compute_finding_signature("det1", "pattern1", "device", "dev123", "50-200")
    sig2 = compute_finding_signature("det1", "pattern1", "device", "dev123", "50-200")
    sig3 = compute_finding_signature("det1", "pattern1", "device", "dev456", "50-200")

    assert sig1 == sig2
    assert sig1 != sig3
    assert len(sig1) == 16


def test_amount_band() -> None:
    assert get_amount_band(10.0) == "<50"
    assert get_amount_band(75.0) == "50-200"
    assert get_amount_band(500.0) == "200-1000"
    assert get_amount_band(2500.0) == ">=1000"


def test_feedback_store_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "feedback.jsonl"
        store = FeedbackStore(jsonl_path)

        dec1 = FeedbackDecision(
            detector="det_test",
            action="reject",
            analyst="analyst_1",
            reason="False positive",
            scope="this_element_value",
            element_type="device_profile",
            element_value="dev_test_1",
        )
        id1 = store.record_decision(dec1)
        assert id1.startswith("DEC-")

        # Reload store from disk
        store2 = FeedbackStore(jsonl_path)
        decisions = store2.list_decisions()
        assert len(decisions) == 1
        assert decisions[0].decision_id == id1
        assert decisions[0].element_value == "dev_test_1"


def test_no_over_generalisation_different_element_values() -> None:
    """Verifies that 3 rejects scoped to one element value (e.g., device_A) suppress device_A

    finding but do NOT suppress the same detector on a different element value (e.g., device_B).
    """
    store = FeedbackStore()
    d1 = store.record_decision(
        FeedbackDecision(
            detector="noisy_toy_detector",
            action="reject",
            scope="this_element_value",
            element_type="device_profile",
            element_value="device_A",
        )
    )
    store.record_decision(
        FeedbackDecision(
            detector="noisy_toy_detector",
            action="reject",
            scope="this_element_value",
            element_type="device_profile",
            element_value="device_A",
        )
    )
    store.record_decision(
        FeedbackDecision(
            detector="noisy_toy_detector",
            action="reject",
            scope="this_element_value",
            element_type="device_profile",
            element_value="device_A",
        )
    )

    att_a = Attestation(
        what="Finding on device_A",
        evidence=None,
        query="test_query",
        params={"element_type": "device_profile", "element_value": "device_A", "amount": 250.0},
    )
    finding_a = Finding(detector="noisy_toy_detector", pattern="p1", claim=att_a, lr=5.0)

    att_b = Attestation(
        what="Finding on device_B",
        evidence=None,
        query="test_query",
        params={"element_type": "device_profile", "element_value": "device_B", "amount": 250.0},
    )
    finding_b = Finding(detector="noisy_toy_detector", pattern="p1", claim=att_b, lr=5.0)

    adjusted = store.apply([finding_a, finding_b], card_id="C00001-K1")
    assert len(adjusted) == 2

    adj_a, adj_b = adjusted[0], adjusted[1]

    # Finding A is suppressed due to 3 rejects (posterior mean <= 0.2)
    assert adj_a.lr == 1.0
    assert d1 in getattr(adj_a, "causal_decision_ids", [])

    # Finding B is NOT suppressed because it has a different element_value (device_B)
    assert adj_b.lr == 5.0
    assert getattr(adj_b, "causal_decision_ids", []) == []


def test_posterior_suppression_rule() -> None:
    """Verifies Beta(1+accepts, 1+rejects) posterior rule:

    Suppression requires rejects >= 3 AND posterior mean <= 0.2.
    """
    store = FeedbackStore()
    # 2 rejects (rejects < 3) -> should NOT suppress
    store.record_decision(
        FeedbackDecision(
            detector="det_test",
            action="reject",
            scope="this_element_value",
            element_type="device_profile",
            element_value="dev_2rejects",
        )
    )
    store.record_decision(
        FeedbackDecision(
            detector="det_test",
            action="reject",
            scope="this_element_value",
            element_type="device_profile",
            element_value="dev_2rejects",
        )
    )

    att = Attestation(
        what="Test finding",
        evidence=None,
        query="test_query",
        params={"element_type": "device_profile", "element_value": "dev_2rejects", "amount": 200.0},
    )
    finding = Finding(detector="det_test", pattern="p1", claim=att, lr=5.0)

    adjusted = store.apply([finding])
    assert adjusted[0].lr == 5.0  # NOT suppressed because rejects < 3


def test_run_feedback_demonstration() -> None:
    res = run_feedback_demonstration(verbose=True)

    assert "wrongly_acted_against_before" in res
    assert "wrongly_acted_against_after" in res
    assert "fraud_caught_before" in res
    assert "fraud_caught_after" in res
    assert "precision_before" in res
    assert "precision_after" in res

    wb = res["wrongly_acted_against_before"]
    wa = res["wrongly_acted_against_after"]
    fb = res["fraud_caught_before"]
    fa = res["fraud_caught_after"]

    # Clear fall in wrongly acted against while preserving caught fraud
    assert wa < wb
    assert fa > 0
    assert res["precision_after"] >= res["precision_before"]
    assert len(res["changed_findings_log"]) >= 3

    att = res["attestation"]
    assert att.query == "assay.r2.feedback.run_feedback_demonstration"
    assert att.params["wrongly_acted_against_after"] == wa
    d_dict = att.to_dict()
    assert d_dict["how"]["query"] == att.query
