"""Tests for B35 -- Devil's advocate + counterfactuals."""

from __future__ import annotations

import json
from pathlib import Path

from assay.r2.advocate import (
    CONFLICT_ESCALATE,
    FORBIDDEN_TIPOFF_WORDS,
    SPOF_SHARE,
    Item,
    advocate,
    counterfactuals,
    evaluate_round1,
    format_round1_table,
    round1_ground_truth,
)
from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Finding


def _make_finding(detector: str, lr: float, txn_id: int = 3478782, card_id: str = "C11891-K1") -> Finding:
    att = Attestation(
        what=f"{detector} fired with lr={lr}",
        evidence=Evidence(txn_ids=[txn_id], card_ids=[card_id]),
        query=detector,
        params={"txn_id": txn_id},
    )
    return Finding(
        detector=detector,
        pattern="card_not_present_fraud",
        claim=att,
        lr=lr,
        affected_txn_ids=[txn_id],
        connected_card_ids=[],
    )


def test_advocate_conflict_and_spof_detection():
    """Verify advocate() builds the opposite case, detects >60% SPOF, and triggers R8 escalation on conflict."""
    answer = {
        "case_id": "TEST-001",
        "case": {
            "status": "open",
            "verdict": "fraud",
            "fraud_probability": 0.78,
            "affected_txn_ids": ["3478782"],
            "connected_card_ids": [],
            "evidence": [],
        },
        "next_best_actions": {"initial": [], "final": []},
    }
    # Strong single finding pushing fraud (lr=12.0), plus conflicting evidence pushing legit (lr=0.30)
    ledger = [
        Item(tag="model", claim="model score 0.450", log_lr=-0.2007, lr=0.818, entity_ids=["3478782"]),
        _make_finding("dispute", lr=12.0, txn_id=3478782),
        _make_finding("recurring", lr=0.30, txn_id=3478782),
    ]

    adv = advocate(answer, ledger)
    assert adv.opposite_verdict == "legitimate"
    assert 0.0 <= adv.conflict_score <= 1.0
    assert adv.spof is not None
    assert adv.spof.tag == "dispute"
    assert adv.spof_share > SPOF_SHARE
    assert adv.recommendation == "escalate"
    assert "R8" in adv.reason
    assert adv.attestation is not None
    att_dict = adv.attestation.to_dict()
    assert att_dict["how"]["query"] == "advocate"
    assert "3478782" in att_dict["evidence"]["txn_ids"]


def test_counterfactuals_minimal_flip_and_no_tipoff():
    """Verify counterfactuals() finds the minimal flip set and generates customer-safe requests without SAR tip-off."""
    answer = {
        "case_id": "TEST-002",
        "case": {
            "status": "closed_fraud",
            "verdict": "fraud",
            "fraud_probability": 0.85,
            "affected_txn_ids": ["3583368"],
            "connected_card_ids": ["C00877-K1"],
            "evidence": [],
        },
        "next_best_actions": {"initial": [], "final": []},
    }
    ledger = [
        Item(tag="model", claim="model score 0.550", log_lr=0.2007, lr=1.222, entity_ids=["3583368"]),
        _make_finding("ring", lr=5.0, txn_id=3583368, card_id="C00877-K1"),
        _make_finding("new_device", lr=1.4, txn_id=3583368, card_id="C11923-K2"),
    ]

    cf = counterfactuals(answer, ledger)
    assert cf.flips_band is True
    # Removing 'ring' (log(5)=1.61) alone drops logit below logit(0.70)=0.847
    assert len(cf.minimal_findings) == 1
    assert cf.minimal_findings[0].tag == "ring"
    assert cf.new_p < 0.70
    assert len(cf.evidence_requests) == 1
    req_text = cf.evidence_requests[0]["request_text"].lower()
    for bad in FORBIDDEN_TIPOFF_WORDS:
        assert bad not in req_text
    assert cf.attestation.to_dict()["how"]["query"] == "counterfactuals"


def test_round1_evaluation_uncertain_vs_confidently_right():
    """Evaluate on all 20 Round 1 cases (cases/HHG-*.json) against known ground truth.

    Verify:
    1. Ground truth matches closed_cases_history.csv (HHG-001,002,004,005,010,012,013,015,017,020 legit; rest fraud).
    2. High conflict_score (>= 0.35) and 'escalate' recommendation coincide with our two uncertain cases
       (HHG-002, HHG-004) and with NONE of the confidently-right cases.
    """
    gt = round1_ground_truth()
    expected_legit = {
        "HHG-001", "HHG-002", "HHG-004", "HHG-005", "HHG-010",
        "HHG-012", "HHG-013", "HHG-015", "HHG-017", "HHG-020",
    }
    actual_legit = {k for k, v in gt.items() if v == "legitimate"}
    assert actual_legit == expected_legit

    rows = evaluate_round1(Path("cases"))
    assert len(rows) == 20

    table_str = format_round1_table(rows)
    print("\n=== B35 Round 1 Devil's Advocate & Counterfactual Evaluation ===")
    print(table_str)

    by_id = {r["case_id"]: r for r in rows}

    # Check our two uncertain cases: HHG-002 and HHG-004
    for u_cid in ("HHG-002", "HHG-004"):
        assert by_id[u_cid]["ours"] == "uncertain"
        assert by_id[u_cid]["conflict_score"] >= CONFLICT_ESCALATE
        assert by_id[u_cid]["recommendation"] == "escalate"

    # Check confidently-right cases: none should have high conflict_score or escalate recommendation
    conf_right = [r for r in rows if r["confidently_right"]]
    assert len(conf_right) == 18
    for r in conf_right:
        assert r["conflict_score"] < CONFLICT_ESCALATE, f"Confidently-right case {r['case_id']} had high conflict: {r}"
        assert r["recommendation"] == "keep", f"Confidently-right case {r['case_id']} recommended escalate: {r}"

    #Also verify no evidence request across all 20 cases has any tip-off wording
    for pth in sorted(Path("cases").glob("HHG-*.json")):
        ans = json.loads(pth.read_text())
        cf = counterfactuals(ans)
        for req in cf.evidence_requests:
            txt = req["request_text"].lower()
            for bad in FORBIDDEN_TIPOFF_WORDS:
                assert bad not in txt
