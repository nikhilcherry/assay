"""Tests for assay.r2.drift (Challenge 12 -- Schema drift survival)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from assay.r2.drift import (
    apply_mapping,
    create_approval_record,
    diff,
    impact,
    propose_mapping,
    run_drift_analysis,
    snapshot,
)


def test_drift_full_pipeline(tmp_path: Path):
    reports = run_drift_analysis()
    assert len(reports) >= 2

    # 1. Closed cases report assertions
    cc_report = next(r for r in reports if r.file_label == "closed_cases_history.csv")
    assert cc_report.old_shape.row_count == 5565
    assert cc_report.new_shape.row_count == 5585

    # Identify 20 new CC IDs
    overlap_change = next(c for c in cc_report.changes if c.kind == "id_overlap")
    assert len(overlap_change.details["new_sample_ids"]) == 20
    assert overlap_change.details["shared_count"] == 5565

    # Shared 5,565 cases diff assertion: 0 changed rows
    changed_rows_change = next(c for c in cc_report.changes if c.kind == "changed_rows")
    assert changed_rows_change.details["changed_count"] == 0

    # 2. RAW Transactions report assertions (397 columns)
    tx_report = next(r for r in reports if r.file_label == "transactions_2017q1.csv")
    assert tx_report.old_shape.row_count == 590742
    assert tx_report.new_shape.row_count == 60196
    assert len(tx_report.old_shape.columns) == 397
    assert len(tx_report.new_shape.columns) == 397

    # Assert semantic breaks are detected
    assert len(tx_report.semantic_breaks) == 3

    # Break 1: Card ID mismatch
    card_mismatch = next(b for b in tx_report.semantic_breaks if b.name == "card_id_mismatch")
    assert card_mismatch.details["case_id"] == "HHF-011"
    assert card_mismatch.details["pack_card_id"] == "C09241-K1"
    assert card_mismatch.details["derived_bank_card_id"] == "C09241-K2"

    # Break 2: risk_score PSI shift
    psi_break = next(b for b in tx_report.semantic_breaks if b.name == "risk_score_psi_shift")
    assert psi_break.details["psi"] >= 0.0

    # Break 3: TransactionDT continuity
    dt_break = next(b for b in tx_report.semantic_breaks if b.name == "transaction_dt_continuity")
    assert dt_break.details["gap_seconds"] == 86
    assert dt_break.details["max_2016_dt"] == 15811134
    assert dt_break.details["min_2017_dt"] == 15811220

    # 3. Gate Refusal assertion: apply_mapping refuses without valid approval
    mapping = propose_mapping(tx_report.changes)
    assert "mapping_hash" in mapping

    approvals_file = tmp_path / "approvals.json"
    approvals_file.write_text(json.dumps({"approvals": []}), encoding="utf-8")

    secret_token = "L2_SECRET_TOKEN_998811"
    with pytest.raises(PermissionError, match="Gate refusal"):
        apply_mapping(mapping, approvals_file, token=secret_token)

    # 4. Gate Success assertion: apply_mapping succeeds with valid approval record signed by L2 token hash
    rec = create_approval_record(mapping, approver="l2_manager_alice", token=secret_token, role="L2")
    approvals_file.write_text(json.dumps({"approvals": [rec]}), encoding="utf-8")

    result = apply_mapping(mapping, approvals_file, token=secret_token)
    assert isinstance(result, dict)
    assert result["status"] == "applied"
    assert result["approver"] == "l2_manager_alice"
    assert result["role"] == "L2"

    # Print summary report
    print("\n" + "=" * 60)
    print("      SCHEMA DRIFT SURVIVAL (CHALLENGE 12) REPORT")
    print("=" * 60)
    for rep in reports:
        print(f"\n--- File: {rep.file_label} ---")
        print(f"Old Shape: {rep.old_shape.row_count} rows, {len(rep.old_shape.columns)} cols, sha256={rep.old_shape.sha256[:12]}")
        print(f"New Shape: {rep.new_shape.row_count} rows, {len(rep.new_shape.columns)} cols, sha256={rep.new_shape.sha256[:12]}")
        print("Detected Changes:")
        for chg in rep.changes:
            print(f"  [{chg.severity.upper()}] {chg.kind}: {chg.description}")
        if rep.semantic_breaks:
            print("Semantic Breaks:")
            for sb in rep.semantic_breaks:
                print(f"  [CRITICAL BREAK] {sb.name}: {sb.description}")
        print(f"Attestation: {rep.attestation.what}")
    print("=" * 60)


if __name__ == "__main__":
    test_drift_full_pipeline(Path(tempfile.mkdtemp()))
