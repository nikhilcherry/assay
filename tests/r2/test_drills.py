"""Tests for BLOCK B30 -- Live resilience drill simulator (D1..D9) on 3 exam cases."""

from __future__ import annotations

import json
import resource

import pytest

from assay.data import ROOT
from assay.r2.drills import (
    Scorecard,
    build_slim_agent,
    inspect_decision_path_imports,
    run_drill_battery,
)


@pytest.fixture(scope="module")
def shared_agent():
    """Build the agent ONCE per test session via build_slim_agent() (~4 GB RSS)."""
    store, inv = build_slim_agent()
    return store, inv


@pytest.fixture(scope="module")
def scorecard(shared_agent) -> Scorecard:
    store, inv = shared_agent
    sc = run_drill_battery(case_ids=["HHF-007", "HHF-012", "HHF-003"], store=store, inv=inv)
    doc_path = ROOT / "docs" / "r2" / "B30_drills.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(sc.to_markdown(), encoding="utf-8")

    # Print the required drill table: drill, expected, observed, pass
    print("\n" + sc.format_drill_table())
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nMeasured peak RSS: {peak_kb / (1024 * 1024):.2f} GB ({sc.passed}/{sc.total} PASS)")
    return sc


def test_scorecard_has_all_drills_and_zero_failures(scorecard: Scorecard):
    assert scorecard.cases == ["HHF-007", "HHF-012", "HHF-003"]
    assert scorecard.total == 75  # 25 drill checks per case across 3 cases
    assert scorecard.failed == 0, f"Unexpected drill failures: {[r for r in scorecard.rows if r.outcome != 'PASS']}"
    assert scorecard.passed == scorecard.total

    drill_ids = {r.drill_id for r in scorecard.rows}
    assert drill_ids == {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"}


def test_no_false_saved_status_on_any_failure_drill(scorecard: Scorecard):
    for row in scorecard.rows:
        assert not row.saved_complete_file, f"Drill {row.drill_id} ({row.subcase}) falsely wrote complete <case_id>.json"
        if row.status in ("INCOMPLETE", "REJECTED_INPUT"):
            assert f"{row.case_id}.incomplete.json" in row.files_written
            assert f"{row.case_id}.json" not in row.files_written


def test_every_row_carries_valid_attestation(scorecard: Scorecard):
    for row in scorecard.rows:
        att = row.attestation
        assert isinstance(att, dict)
        assert set(att.keys()) == {"what", "evidence", "how", "when"}
        assert att["how"]["query"] == "assay.r2.drills.run_drill_battery"
        assert att["when"]["dataset"].startswith("r2-2017q1@")


def test_d9_import_inspection_no_network_libs():
    info = inspect_decision_path_imports()
    assert "assay.investigate" in info["checked_modules"]
    assert "assay.r2.investigate_r2" in info["checked_modules"]
    assert info["forbidden_network_or_llm_imports"] == []


def test_scorecard_serialization_roundtrip(scorecard: Scorecard):
    raw_json = scorecard.to_json()
    parsed = json.loads(raw_json)
    assert parsed["total"] == scorecard.total
    assert parsed["passed"] == scorecard.passed
    assert parsed["failed"] == 0
    md = scorecard.to_markdown()
    assert "# Live Resilience Drills Scorecard (B30)" in md
    assert "| drill | expected | observed | pass |" in md
    assert "| D1 |" in md
    assert "| D9 |" in md
