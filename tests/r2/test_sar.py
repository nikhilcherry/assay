"""Tests for B19 -- Evidence to SAR (challenge 4)."""

from __future__ import annotations

import json
from pathlib import Path

from assay.r2.frames import tx_all
from assay.r2.sar import draft_sar, lint_sar, sar_required

CASES_DIR = Path(__file__).resolve().parents[2] / "cases"


def _load_hhg_cases() -> list[dict]:
    files = sorted(CASES_DIR.glob("HHG-*.json"))
    assert files, f"Expected HHG-*.json files in {CASES_DIR}"
    return [json.loads(f.read_text()) for f in files]


def test_hhg_sar_required_and_lint_clean(capsys):
    tx = tx_all()
    answers = _load_hhg_cases()

    filed_cases: list[str] = []
    legit_cases: list[str] = []
    no_sar_fraud_cases: list[str] = []

    printed_one = False

    for ans in answers:
        cid = ans["case_id"]
        verdict = ans["case"]["verdict"]
        r1_filed = bool(ans.get("sar", {}).get("file"))
        dec = sar_required(ans, tx=tx)
        sar = draft_sar(ans, tx)

        # Policy §3a agreement with Round 1 filed reports
        assert dec.required == r1_filed, (
            f"{cid}: sar_required={dec.required} ({dec.reason_text}) != R1 sar.file={r1_filed}"
        )

        if verdict == "legitimate":
            legit_cases.append(cid)
            assert not dec.required, f"Legitimate case {cid} must never require a SAR"
            assert not sar.required
            assert sar.sentences == []
            assert sar.to_answer_sar()["file"] is False
            assert lint_sar(sar, tx) == []
        elif dec.required:
            filed_cases.append(cid)
            errs = lint_sar(sar, tx)
            assert errs == [], f"{cid} failed SAR lint: {errs}"
            md = sar.to_markdown()
            html_doc = sar.to_html()
            assert f"# Suspicious Activity Report (SAR) — {cid}" in md
            assert "<!DOCTYPE html>" in html_doc
            assert "Evidence Graph" in html_doc
            assert len(sar.evidence_graph["nodes"]) >= 3
            assert len(sar.evidence_graph["edges"]) >= 2

            if cid == "HHG-006" and not printed_one:
                print("\n" + "=" * 72)
                print(md)
                print("=" * 72)
                printed_one = True
        else:
            no_sar_fraud_cases.append(cid)
            assert not sar.required
            assert lint_sar(sar, tx) == []

    print(
        f"\nHHG summary: total={len(answers)}, sar_filed={len(filed_cases)} ({filed_cases}), "
        f"legit_no_sar={len(legit_cases)} ({legit_cases}), "
        f"other_no_sar={len(no_sar_fraud_cases)} ({no_sar_fraud_cases})"
    )
    assert "HHG-006" in filed_cases
    assert "HHG-014" in filed_cases
    assert len(legit_cases) > 0


def test_sar_lint_catches_bad_numbers_and_missing_citations():
    tx = tx_all()
    ans = json.loads((CASES_DIR / "HHG-006.json").read_text())
    sar = draft_sar(ans, tx)
    assert lint_sar(sar, tx) == []

    # Mutate a dollar figure in sentence 0 so it no longer matches tx
    orig_text = sar.sentences[0].text
    sar.sentences[0].text = orig_text.replace("$1,906.07", "$9,999.99")
    errs = lint_sar(sar, tx)
    assert any("$9,999.99" in e for e in errs), f"Expected dollar mismatch error, got {errs}"

    # Remove citation marker and add speculation word without probability
    sar.sentences[0].text = "The cardholder likely made an unauthorized transaction."
    errs2 = lint_sar(sar, tx)
    assert any("missing [T:...]" in e for e in errs2)
