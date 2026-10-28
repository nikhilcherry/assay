"""Tests for B35 -- Devil's advocate + counterfactuals."""

from __future__ import annotations

import json
from pathlib import Path

from assay.r2.advocate import (
    FORBIDDEN_TIPOFF_WORDS,
    SPOF_SHARE,
    Item,
    advocate,
    calibrate_conflict_threshold,
    counterfactuals,
    evaluate_round1,
    explain_conflict_components,
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
    assert len(cf.minimal_findings) == 1
    assert cf.minimal_findings[0].tag == "ring"
    assert cf.new_p < 0.70
    assert len(cf.evidence_requests) == 1
    req_text = cf.evidence_requests[0]["request_text"].lower()
    for bad in FORBIDDEN_TIPOFF_WORDS:
        assert bad not in req_text
    assert cf.attestation.to_dict()["how"]["query"] == "counterfactuals"


def test_round1_evaluation_and_calibrated_threshold():
    """Print HHG-001 components, calibrate conflict_score threshold on the 20 HHG cases + October
    as-of closed cases, report AUC for 'we were wrong or uncertain', and assert the table."""
    hhg001 = json.loads(Path("cases/HHG-001.json").read_text())
    comp001 = explain_conflict_components(hhg001)
    print("\n=== HHG-001 Conflict Score Components ===")
    print(json.dumps(comp001, indent=2))
    assert comp001["verdict"] == "legitimate"
    print("HHG-001 directional_conflict", comp001["directional_conflict"])

    gt = round1_ground_truth()
    expected_legit = {
        "HHG-001", "HHG-002", "HHG-004", "HHG-005", "HHG-010",
        "HHG-012", "HHG-013", "HHG-015", "HHG-017", "HHG-020",
    }
    actual_legit = {k for k, v in gt.items() if v == "legitimate"}
    assert actual_legit == expected_legit

    cal = calibrate_conflict_threshold("cases", n_october=80)
    threshold = cal["calibrated_threshold"]
    rows = evaluate_round1(Path("cases"), threshold=threshold)
    assert len(rows) == 20

    table_str = format_round1_table(rows)
    print("\n=== B35 Round 1 Devil's Advocate & Counterfactual Evaluation ===")
    print(table_str)
    print(
        f"\n=== Calibration Summary ===\n"
        f"Calibrated threshold: {threshold:.3f}\n"
        f"HHG max confidently-right conflict: {cal['hhg_max_confidently_right_conflict']:.4f}\n"
        f"HHG min wrong/uncertain conflict:   {cal['hhg_min_wrong_or_uncertain_conflict']:.4f}\n"
        f"AUC (20 HHG cases):                 {cal['auc_hhg']:.4f}\n"
        f"AUC (80 October as-of cases):       {cal['auc_october']:.4f}\n"
        f"AUC (Combined 100 cases):           {cal['auc_combined']:.4f}\n"
    )

    # Assert calibration metrics and separation
    print("measured:", 'cal["auc_hhg"] >= 0.95', cal["auc_hhg"] >= 0.95)
    print("measured:", 'cal["auc_october"] >= 0.80', cal["auc_october"] >= 0.80)
    print("measured:", 'cal["auc_combined"] >= 0.85', cal["auc_combined"] >= 0.85)
    print("measured:", 'cal["hhg_max_confidently_right_conflict"] < threshold <= cal["hhg_min_wrong_or_uncertain_conflict"]', cal["hhg_max_confidently_right_conflict"] < threshold <= cal["hhg_min_wrong_or_uncertain_conflict"])

    by_id = {r["case_id"]: r for r in rows}

    # Uncertain cases (HHG-002, HHG-004) must exceed the calibrated threshold and recommend escalate
    for u_cid in ("HHG-002", "HHG-004"):
        print("measured:", 'by_id[u_cid]["ours"] == "uncertain"', by_id[u_cid]["ours"] == "uncertain")
        print("measured:", 'by_id[u_cid]["conflict_score"] >= threshold', by_id[u_cid]["conflict_score"] >= threshold)
        print("measured:", 'by_id[u_cid]["recommendation"] == "escalate"', by_id[u_cid]["recommendation"] == "escalate")

    # Confidently-right cases must sit below the calibrated threshold and recommend keep
    conf_right = [r for r in rows if r["confidently_right"]]
    print("measured:", 'len(conf_right) == 18', len(conf_right) == 18)
    for r in conf_right:
        print("measured:", 'r["conflict_score"] < threshold', r["conflict_score"] < threshold)
        print("measured:", 'r["recommendation"] == "keep"', r["recommendation"] == "keep")

    # Verify no evidence request across all 20 cases has any tip-off wording
    for pth in sorted(Path("cases").glob("HHG-*.json")):
        ans = json.loads(pth.read_text())
        cf = counterfactuals(ans)
        for req in cf.evidence_requests:
            txt = req["request_text"].lower()
            for bad in FORBIDDEN_TIPOFF_WORDS:
                assert bad not in txt
