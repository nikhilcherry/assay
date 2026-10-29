"""Tests for BLOCK C9 -- Unknown-fault chaos suite (`assay/r2/chaos.py`).

Follows the TEST RULE strictly:
- Runs the full battery on HHF-007 (and HHF-005 for TigerGraph-path faults if TG is
  reachable at 127.0.0.1:14240, else marks those SKIPPED with reason).
- Writes `docs/r2/C9_chaos.md` with header `generated; do not edit`.
- Asserts ONLY what we control: schema/shape, reproducibility, attestation fields,
  that every scenario produced a row, that no injected scenario is `INVALID` (our
  own harness proof-of-injection), and that the measurement table is written.
- Does NOT assert that all scenarios pass (real agent failures are findings for the lead).
"""

from __future__ import annotations

from pathlib import Path

from assay.data import ROOT
from assay.r2.chaos import (
    DEFAULT_DOC_PATH,
    SCENARIO_NAMES,
    ChaosReport,
    run_chaos_battery,
    write_chaos_doc,
)


def test_chaos_battery_runs_and_writes_doc() -> None:
    report: ChaosReport = run_chaos_battery(base_case_id="HHF-007", tg_case_id="HHF-005")
    doc_path: Path = write_chaos_doc(report, DEFAULT_DOC_PATH)

    md_text = doc_path.read_text(encoding="utf-8")
    print("\n" + md_text)

    # Print full diagnostic notes for any INVALID or FAIL scenario
    for r in report.results:
        if r.status in ("INVALID", "FAIL"):
            print(f"[DIAGNOSTIC {r.status}] {r.scenario}: exit={r.exit_code} notes={r.notes}")

    # 1. Every scenario in SCENARIO_NAMES produced a result row
    assert len(report.results) == len(SCENARIO_NAMES)
    observed_names = [r.scenario for r in report.results]
    assert observed_names == SCENARIO_NAMES

    # 2. Check each row's schema, invariant structure, proof-of-injection, and attestation fields
    for r in report.results:
        assert r.scenario in SCENARIO_NAMES
        assert r.case_id in ("HHF-007", "HHF-005")
        assert r.status in ("PASS", "FAIL", "SKIPPED", "INVALID")
        # Ensure our own fault injection harness fired cleanly (never INVALID)
        assert r.status != "INVALID", f"Harness injection failed for {r.scenario}: {r.notes}"
        assert isinstance(r.files_after, list)
        assert hasattr(r.invariants, "inv_a_exit0_valid_json_only")
        assert hasattr(r.invariants, "inv_b_nonzero_no_fresh_json_and_clear_err")
        assert hasattr(r.invariants, "inv_c_never_only_traceback")
        assert hasattr(r.invariants, "inv_d_concurrent_consistent")
        assert hasattr(r.invariants, "inv_e_never_exit0_missing_or_partial")

        if r.status == "SKIPPED":
            assert r.scenario.startswith("sc_tg_")
            assert not report.tg_reachable
            assert "SKIPPED" in r.notes

        # Attestation contract verification
        att = r.attestation
        assert isinstance(att, dict)
        assert set(att.keys()) == {"what", "evidence", "how", "when"}
        assert r.scenario in att["what"]
        assert att["how"]["query"] == "assay.r2.chaos.run_single_scenario"
        assert att["how"]["params"]["scenario"] == r.scenario
        assert att["how"]["params"]["case_id"] == r.case_id
        assert r.case_id in att["evidence"]["case_ids"]
        assert "ts" in att["when"] and "dataset" in att["when"]

    # 3. Verify generated markdown file exists, has required header, and contains a row per scenario
    assert doc_path == ROOT / "docs" / "r2" / "C9_chaos.md"
    assert doc_path.exists()
    assert "generated; do not edit" in md_text.splitlines()[0]
    for name in SCENARIO_NAMES:
        assert f"| `{name}` |" in md_text
