"""The submitted answer files, checked against the brief's Answer Format and
against the policy's own routing. Needs no dataset: it reads cases/ only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assay.validate import validate  # noqa: E402

FILES = sorted((ROOT / "cases").glob("HHG-*.json"))


def test_twenty_answer_files():
    assert [f.stem for f in FILES] == [f"HHG-{i:03d}" for i in range(1, 21)]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_answer_is_valid(path):
    a = json.loads(path.read_text())
    assert a["case_id"] == path.stem
    assert validate(a) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_exposure_is_zero_exactly_when_legitimate(path):
    c = json.loads(path.read_text())["case"]
    assert (c["verdict"] == "legitimate") == (c["exposure_usd"] == 0)


def test_not_everything_is_blocked():
    """The brief: half the cases are legitimate; an agent that blocks everything scores badly."""
    blocked = sum(any(x["action"] in ("BLOCK_CARD", "BLOCK_ALL_CARDS")
                      for x in json.loads(f.read_text())["next_best_actions"]["final"]) for f in FILES)
    assert 0 < blocked <= 12


def test_block_all_cards_never_recommended_without_r10():
    for f in FILES:
        for x in json.loads(f.read_text())["next_best_actions"]["final"]:
            assert x["action"] != "BLOCK_ALL_CARDS" or x["reason"].startswith("R10")
