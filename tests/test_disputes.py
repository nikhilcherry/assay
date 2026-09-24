"""A cardholder's denial, as measured in docs/DISPUTE_REPORT.json, and the rule
that the agent never invents a disputing customer changing their mind."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assay import disputes  # noqa: E402


def test_history_says_every_dispute_was_fraud():
    r = json.loads(disputes.REPORT.read_text())
    assert r["disputes"] == r["disputes_confirmed_fraud"] > 1000
    assert r["lr_history"] > 1000


def test_a_denial_always_raises_the_odds():
    for p in (0.001, 0.003, 0.01, 0.1, 0.5, 0.7, 0.9):
        post, _ = disputes.posterior(p)
        assert post > p


def test_reading_rises_with_the_model_score():
    ps = [disputes.posterior(p)[0] for p in (0.002, 0.008, 0.015, 0.03, 0.07, 0.15, 0.3, 0.5, 0.7)]
    assert ps == sorted(ps)


def test_a_near_zero_score_is_not_enough_to_convict():
    # under 0.005 the score is 14x as common on legitimate transactions: the case stays open
    assert 0.4 < disputes.posterior(0.003)[0] < 0.6


FILES = sorted((ROOT / "cases").glob("HHG-*.json")) + sorted((ROOT / "monitor").glob("MON-*.json"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_a_disputed_charge_is_only_cleared_on_a_recurring_match(path):
    a = json.loads(path.read_text())
    claims = [e["claim"] for e in a["case"]["evidence"]]
    disputed = any("as not theirs" in c for c in claims)
    if disputed and a["case"]["verdict"] == "legitimate":
        assert any("recurring" in c for c in claims), "cleared a denial without R7"


def test_detector_ratios_are_the_measured_ones():
    from assay import detectors as D
    r = json.loads((ROOT / "docs" / "LR_REPORT.json").read_text())["detectors"]
    assert D.LR_STRUCTURING == r["structuring"]["lr"]
    assert D.LR_CARD_TESTING == r["card_testing"]["lr"]
    assert D.LR_MEMORY == r["memory"]["lr"]
    # measured on top of the model, a detector adds far less than its raw lift
    for k in ("structuring", "card_testing", "memory"):
        assert r[k]["lr"] < r[k]["lr_marginal"]
