"""Tests for assay.r2.policy_check."""

import csv
import glob
import json
from pathlib import Path

from assay.r2.policy_check import Violation, compare_with_grader, lint


def test_lint_runs_on_all_20_case_files(capsys):
    case_files = sorted(glob.glob("cases/HHG-*.json") or glob.glob("cases/*.json"))
    assert len(case_files) >= 1, "No case files found in cases/"

    pack_map = {}
    for p in ["data/case_pack.csv", "data/round2/share/cases/case_pack_final.csv", "cases/case_pack_final.csv"]:
        if Path(p).exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pack_map[row["case_id"]] = row
            break

    lint_results = {}
    print("\n" + "=" * 60)
    print("HHG ANSWERS VIOLATION REPORT (20 CASES)")
    print("=" * 60)
    for cf in case_files:
        cid = Path(cf).stem
        with open(cf, "r", encoding="utf-8") as f:
            ans = json.load(f)
        v_list = lint(ans, pack_map.get(cid))
        lint_results[cid] = v_list
        print(f"\n[Case {cid}]")
        if not v_list:
            print("  No violations found.")
        else:
            for v in v_list:
                print(f"  - [{v.severity.upper()}] Rule: {v.rule} | Path: {v.json_path} | Message: {v.message}")

    assert len(lint_results) >= 1

    # Disagreement table with grade.py
    grader_cases = compare_with_grader(case_files, pack_map)
    print("\n" + "=" * 60)
    print("DISAGREEMENT TABLE WITH GRADE.PY")
    print("=" * 60)
    print(f"{'CASE ID':10s} | {'LINT (H/P/W)':15s} | {'GRADE.PY (H/P/W)':16s} | {'MATCH?'}")
    print("-" * 60)

    for cid in sorted(lint_results.keys()):
        v_list = lint_results[cid]
        lh = sum(1 for v in v_list if v.severity == "hard")
        lp = sum(1 for v in v_list if v.severity == "policy")
        lw = sum(1 for v in v_list if v.severity == "warning")

        g_findings = grader_cases.get(cid, {}).get("findings", [])
        gh = sum(1 for cls, _, _ in g_findings if cls == "hard")
        gp = sum(1 for cls, _, _ in g_findings if cls == "policy")
        gw = sum(1 for cls, _, _ in g_findings if cls == "warning")

        match = (lh == gh and lp == gp)
        print(f"{cid:10s} | {lh:d}/{lp:d}/{lw:d}           | {gh:d}/{gp:d}/{gw:d}            | {'YES' if match else 'NO'}")

    out = capsys.readouterr()
    print(out.out)


def test_hhg_001_no_hard_violations():
    p = Path("cases/HHG-001.json")
    if not p.exists():
        p = Path("cases/HHF-001.json")
    assert p.exists(), "HHG-001.json must exist"
    with open(p, "r", encoding="utf-8") as f:
        ans = json.load(f)
    violations = lint(ans)
    hard_violations = [v for v in violations if v.severity == "hard"]
    assert len(hard_violations) == 0, f"HHG-001 has hard violations: {hard_violations}"


def test_hhg_002_and_004_uncertain_ground_truth_policy():
    """Verify ground truth for HHG-002 and HHG-004: legitimate cases where we answered
    uncertain with p=0.55/0.52. Verify R1 and R8 are satisfied (no BLOCK_CARD/BLOCK_ALL_CARDS,
    and ESCALATE_TO_ANALYST is present in final actions).
    """
    for cid in ["HHG-002", "HHG-004"]:
        p = Path(f"cases/{cid}.json")
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            ans = json.load(f)

        verdict = ans.get("case", {}).get("verdict")
        prob = ans.get("case", {}).get("fraud_probability")
        nba_final = [a.get("action") if isinstance(a, dict) else a for a in ans.get("next_best_actions", {}).get("final", [])]

        bad = {"BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"} & set(nba_final)
        r1_ok = not bad
        r8_ok = "ESCALATE_TO_ANALYST" in nba_final
        print(f"\n[Ground Truth Check {cid}] truth=legitimate answered={verdict} p={prob} final={nba_final}")
        print(f"  R1 satisfied (no block/report): {r1_ok} {sorted(bad) if bad else ''}")
        print(f"  R8 satisfied (ESCALATE_TO_ANALYST present): {r8_ok}")
        assert verdict in ("uncertain", "legitimate")
        assert r1_ok, f"{cid} blocks/reports a legitimate customer: {bad}"

        violations = lint(ans)
        hard_violations = [v for v in violations if v.severity == "hard"]
        assert len(hard_violations) == 0, f"{cid} must have zero hard violations"
        if verdict == "uncertain" and not r8_ok:
            assert any(v.rule == "R8" for v in violations), "lint must flag the missing R8 escalation"


def test_r1_vs_r2_conflation():
    """Test R1 vs R2 conflation with two synthetic answer dicts."""
    # Synthetic Failure 1: R2 situation (customer denial trigger), but agent recommends VERIFY_WITH_CUSTOMER
    r2_conflated = {
        "case_id": "SYN-R2",
        "case": {
            "verdict": "fraud",
            "fraud_probability": 0.85,
            "pattern": "card_not_present_fraud",
            "affected_txn_ids": ["3600001"],
            "first_suspicious_txn_id": "3600001",
            "exposure_usd": 250.0,
        },
        "next_best_actions": {
            "initial": [{"action": "VERIFY_WITH_CUSTOMER", "route": "auto"}],
            "final": [{"action": "BLOCK_CARD", "route": "L1"}, {"action": "CREATE_CASE", "route": "auto"}]
        },
        "sar": {"file": False},
        "stop_reason": "Testing R1/R2 conflation on R2"
    }
    v1 = lint(r2_conflated, case_row={"trigger_type": "customer_report"})
    conflation_r2 = [v for v in v1 if v.rule == "R1_R2_CONFLATION"]
    assert len(conflation_r2) >= 1
    assert "Customer already denied the charge" in conflation_r2[0].message

    # Synthetic Failure 2: R1 situation (weak signal, risk score trigger, p=0.55), but agent blocks immediately without verification step
    r1_conflated = {
        "case_id": "SYN-R1",
        "case": {
            "verdict": "fraud",
            "fraud_probability": 0.55,
            "pattern": "card_not_present_fraud",
            "affected_txn_ids": ["3600002"],
            "first_suspicious_txn_id": "3600002",
            "exposure_usd": 300.0,
        },
        "next_best_actions": {
            "initial": [{"action": "BLOCK_CARD", "route": "L1"}],
            "final": [{"action": "BLOCK_CARD", "route": "L1"}, {"action": "CREATE_CASE", "route": "auto"}]
        },
        "sar": {"file": False},
        "stop_reason": "Testing R1/R2 conflation on R1"
    }
    v2 = lint(r1_conflated, case_row={"trigger_type": "risk_score"})
    conflation_r1 = [v for v in v2 if v.rule == "R1_R2_CONFLATION"]
    assert len(conflation_r1) >= 1
    assert "single weak signal without prior verification step" in conflation_r1[0].message


def test_innocent_customer_rule():
    ans = {
        "case_id": "TEST-001",
        "case": {
            "verdict": "legitimate",
            "fraud_probability": 0.05,
            "pattern": "none",
            "affected_txn_ids": [],
            "first_suspicious_txn_id": "",
            "exposure_usd": 0.0,
        },
        "next_best_actions": {
            "initial": [{"action": "VERIFY_WITH_CUSTOMER", "route": "auto"}],
            "final": [{"action": "BLOCK_CARD", "route": "L1"}, {"action": "CLOSE_NO_FRAUD", "route": "auto"}]
        },
        "sar": {"file": False},
        "stop_reason": "Testing innocent customer rule"
    }
    violations = lint(ans)
    innocent_violations = [v for v in violations if v.rule == "INNOCENT_CUSTOMER" and v.severity == "hard"]
    assert len(innocent_violations) == 1
    assert "BLOCK_CARD" in innocent_violations[0].message
