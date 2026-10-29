"""Tests for BLOCK C2 & C8 (`assay.r2.verify_claims`).

Asserts only what we control:
  1. Running `verify_directory` on `cases_r2` prints before/after counts per case,
     produces a well-formed measurement table, and writes `docs/r2/VERIFY_CLAIMS.md`.
  2. Determinism: running twice produces identical statuses and counts.
  3. Existing planted-mutation test: changing one entity ID in an evidence claim to a
     real but wrong `TransactionID`, and changing `exposure_usd` by `+1.00` produces
     `MISMATCH` on both and CLI exit code 1.
  4. C8 planted-mutation tests:
     (a) A planted misquote of a policy rule in a `document:fraud-policy#...` claim
         (e.g. naming `FILE_REPORT` under `R3` or `$2,000` where `R8` says `$500`)
         is flagged as `MISMATCH`.
     (b) A planted wrong threshold in a `next_best_actions` reason (e.g. `$999` instead
         of `$500` on `R8`) is flagged as `MISMATCH`.
"""

from __future__ import annotations

import json
from pathlib import Path

from assay.r2.frames import tx_all
from assay.r2.verify_claims import (
    DEFAULT_ANSWERS_DIR,
    DEFAULT_DOC_PATH,
    STATUS_MISMATCH,
    STATUS_UNRESOLVABLE,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    main as verify_main,
    render_markdown_report,
    verify_directory,
)


def test_verify_claims_runs_and_is_deterministic(tmp_path: Path) -> None:
    json_out = tmp_path / "verify_out.json"
    rc1 = verify_main(["--answers", str(DEFAULT_ANSWERS_DIR), "--json", str(json_out)])
    assert rc1 in (0, 1)
    assert json_out.exists()
    assert DEFAULT_DOC_PATH.exists()

    summary1 = json.loads(json_out.read_text(encoding="utf-8"))
    summary2 = verify_directory(DEFAULT_ANSWERS_DIR)

    # Assert schema and shape (including before_totals and per-case before_counts)
    assert summary1["n_cases"] == 18
    assert set(summary1["totals"].keys()) == {
        STATUS_VERIFIED,
        STATUS_MISMATCH,
        STATUS_UNVERIFIED,
        STATUS_UNRESOLVABLE,
    }
    assert set(summary1["before_totals"].keys()) == {
        STATUS_VERIFIED,
        STATUS_MISMATCH,
        STATUS_UNVERIFIED,
        STATUS_UNRESOLVABLE,
    }
    assert len(summary1["ref_kinds"]) > 0
    assert len(summary1["cases"]) == 18

    # Determinism: running twice produces the exact same statuses and counts
    assert summary1["totals"] == summary2["totals"]
    assert summary1["before_totals"] == summary2["before_totals"]
    assert summary1["ref_kinds"] == summary2["ref_kinds"]
    for c1, c2 in zip(summary1["cases"], summary2["cases"]):
        assert c1["case_id"] == c2["case_id"]
        assert c1["before_counts"] == c2["before_counts"]
        assert c1["counts"] == c2["counts"]
        s1 = [(x["section"], x["index"], x["ref"], x["status"]) for x in c1["checks"]]
        s2 = [(x["section"], x["index"], x["ref"], x["status"]) for x in c2["checks"]]
        assert s1 == s2

    md = render_markdown_report(summary1)
    assert "Before -> After" in md
    print("\n" + md)


def test_planted_mutations_are_caught(tmp_path: Path) -> None:
    # Copy HHF-007.json to tmp_path and plant two mutations:
    #   (a) change one entity_id in an evidence claim to a real but wrong txn_id
    #   (b) change exposure_usd by +1.00
    src = DEFAULT_ANSWERS_DIR / "HHF-007.json"
    ans = json.loads(src.read_text(encoding="utf-8"))

    tx = tx_all()
    # Pick a real TransactionID from a different card
    wrong_tid = str(int(tx[tx["card_id"] != "C09582-K1"]["TransactionID"].iloc[0]))

    mutated_ev_idx = None
    for idx, ev in enumerate(ans["case"]["evidence"]):
        if str(ev.get("ref", "")).startswith("query:") and ev.get("entity_ids"):
            ev["entity_ids"][0] = wrong_tid
            mutated_ev_idx = idx
            break
    assert mutated_ev_idx is not None

    orig_exposure = float(ans["case"]["exposure_usd"])
    ans["case"]["exposure_usd"] = round(orig_exposure + 1.00, 2)

    mut_file = tmp_path / "HHF-007.json"
    mut_file.write_text(json.dumps(ans, indent=2) + "\n", encoding="utf-8")

    summary = verify_directory(tmp_path)
    assert summary["n_cases"] == 1
    case_checks = summary["cases"][0]["checks"]

    ev_check = next(
        c for c in case_checks if c["section"] == "evidence" and c["index"] == mutated_ev_idx
    )
    assert ev_check["status"] == STATUS_MISMATCH
    assert wrong_tid in ev_check["missing_ids"]

    exp_check = next(
        c for c in case_checks if c["section"] == "grader" and c["ref"] == "grader:exposure_usd"
    )
    assert exp_check["status"] == STATUS_MISMATCH

    # CLI must exit 1 when any MISMATCH is present
    doc_tmp = tmp_path / "VERIFY_TMP.md"
    rc = verify_main(["--answers", str(tmp_path), "--doc", str(doc_tmp)])
    assert rc == 1


def test_planted_policy_misquote_and_wrong_nba_threshold_are_caught(tmp_path: Path) -> None:
    # Copy HHF-014.json to tmp_path and plant:
    #   (a) a misquoted policy rule in `document:fraud-policy#R3` naming FILE_REPORT and $2,000
    #   (b) a wrong threshold ($999 instead of $500) in an NBA reason citing R8
    src = DEFAULT_ANSWERS_DIR / "HHF-014.json"
    ans = json.loads(src.read_text(encoding="utf-8"))

    # (a) Plant misquote in document:fraud-policy#R3 (naming FILE_REPORT and $2,000)
    misquoted_ev_idx = None
    for idx, ev in enumerate(ans["case"]["evidence"]):
        if str(ev.get("ref", "")) == "document:fraud-policy#R3":
            ev["claim"] = "Policy R3: Customer confirms the transaction: FILE_REPORT if exposure exceeds $2,000."
            misquoted_ev_idx = idx
            break
    assert misquoted_ev_idx is not None

    # (b) Plant wrong threshold ($999 instead of $500) in next_best_actions.initial reason citing R8
    mutated_nba = False
    for item in ans["next_best_actions"]["initial"]:
        if item.get("action") == "ESCALATE_TO_ANALYST" and "R8" in str(item.get("reason", "")):
            item["reason"] = "R8: the verdict is uncertain and exposure $1,053.79 exceeds $999."
            mutated_nba = True
            break
    assert mutated_nba

    mut_file = tmp_path / "HHF-014.json"
    mut_file.write_text(json.dumps(ans, indent=2) + "\n", encoding="utf-8")

    summary = verify_directory(tmp_path)
    assert summary["n_cases"] == 1
    case_checks = summary["cases"][0]["checks"]

    doc_check = next(
        c for c in case_checks if c["section"] == "evidence" and c["index"] == misquoted_ev_idx
    )
    assert doc_check["status"] == STATUS_MISMATCH
    assert "FILE_REPORT" in doc_check["missing_ids"]
    assert "$2,000" in doc_check["mismatched_numbers"]

    nba_checks = [
        c
        for c in case_checks
        if c["section"] == "next_best_actions" and c["ref"] == "nba:initial:ESCALATE_TO_ANALYST"
    ]
    assert len(nba_checks) == 1
    assert nba_checks[0]["status"] == STATUS_MISMATCH
    assert "$999" in nba_checks[0]["mismatched_numbers"]
