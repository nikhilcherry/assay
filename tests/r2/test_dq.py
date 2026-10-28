"""Tests for Data Quality Detective (assay/r2/dq.py).

Executes all 9 DQ checks, prints the full data quality report, resolves HHF-011,
and builds/displays the investigation taint table for all 18 exam cases.
"""

from __future__ import annotations

from assay.r2.dq import describe_case_taint, resolve_hhf011, run_all_checks, taint
from assay.r2.frames import case_pack, closed_cases, tx_all


def test_data_quality_report_and_taint_table():
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    print("\n" + "=" * 100)
    print("           ASSAY R2 - DATA QUALITY DETECTIVE REPORT")
    print("=" * 100)

    issues = run_all_checks(tx, cc, cp)

    assert len(issues) == 9, f"Expected 9 DQ checks, got {len(issues)}"

    for issue in issues:
        assert issue.id.startswith("DQ-")
        assert issue.affected_count >= 0
        assert issue.attestation is not None
        assert len(issue.attestation.what) > 0

        print(f"\n[{issue.id}] Severity: {issue.severity}")
        print(f"Summary: {issue.what}")
        print(f"Affected Records: {issue.affected_count:,} ({issue.affected_rate_pct:.2f}%)")
        print(f"Why it matters: {issue.why_it_matters_for_fraud}")
        print(f"Attestation Query: {issue.attestation.query}({issue.attestation.params})")
        print(f"Sample Txn IDs (5): {issue.affected_txn_ids[:5]}")

    print("\n" + "=" * 100)
    print("           HHF-011 CARD ID RESOLUTION")
    print("=" * 100)
    hhf011_res = resolve_hhf011(tx, cp)
    for k, v in hhf011_res.items():
        print(f"  {k:<30}: {v}")

    print("\n" + "=" * 140)
    print("           INVESTIGATION TAINT MATRIX & EVIDENCE AVAILABILITY (HHF-001..018)")
    print("=" * 140)
    print(f"{'Case ID':<8} | {'Flagged Txn':<11} | {'Trigger':<15} | {'Tainted DQ Issues':<30} | {'Trust':<17} | {'Unavailable Evidence'}")
    print("-" * 140)

    for row in cp.itertuples():
        desc = describe_case_taint(row, tx, cc)
        t_issues = ", ".join(desc["tainted_issues"]) if desc["tainted_issues"] else "None (Untainted)"
        unavail = " | ".join(desc["unavailable_evidence"]) if desc["unavailable_evidence"] else "All evidence available"
        trigger = str(row.trigger_type)
        print(f"{row.case_id:<8} | {row.flagged_txn_id:<11} | {trigger:<15} | {t_issues:<30} | {desc['trust_verdict']:<17} | {unavail}")

    print("-" * 140)

    # Explicit assertions for known cases
    hhf011_row = cp[cp.case_id == "HHF-011"].iloc[0]
    hhf011_desc = describe_case_taint(hhf011_row, tx, cc)
    assert "DQ-07-CARD-ID-INCONSISTENCY" in hhf011_desc["tainted_issues"], "HHF-011 must be tainted by card ID mismatch"
    assert hhf011_desc["trust_verdict"] == "UNTRUSTWORTHY", "HHF-011 trust verdict must be UNTRUSTWORTHY"
    assert "C09241-K2" in hhf011_desc["investigator_action"], "HHF-011 investigator action must suggest re-mapping to C09241-K2"


if __name__ == "__main__":
    test_data_quality_report_and_taint_table()
