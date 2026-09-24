"""The replay traces in site/data: they must say exactly what the answer files
say, and their probability ledgers must add up. No dataset needed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"
TRACES = sorted(p for p in DATA.glob("*-*.json"))


def committed(case_id: str) -> dict:
    folder = "cases" if case_id.startswith("HHG") else "monitor"
    return json.loads((ROOT / folder / f"{case_id}.json").read_text())


def test_there_is_a_trace_for_every_answer_file():
    ids = {p.stem for p in TRACES}
    assert {p.stem for p in (ROOT / "cases").glob("HHG-*.json")} <= ids
    assert {p.stem for p in (ROOT / "monitor").glob("MON-*.json")} <= ids


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_trace_matches_the_answer_file(path):
    t = json.loads(path.read_text())
    a, b = t["answer"], committed(t["case_id"])
    for k in ("verdict", "fraud_probability", "pattern", "exposure_usd", "connected_card_ids", "affected_txn_ids"):
        assert a["case"][k] == b["case"][k], k
    assert [x["action"] for x in a["next_best_actions"]["final"]] == \
           [x["action"] for x in b["next_best_actions"]["final"]]


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_ledger_adds_up_in_log_odds(path):
    t = json.loads(path.read_text())
    odds = t["odds"]
    assert odds[0]["tag"] == "model"
    lo = math.log(odds[0]["p"] / (1 - odds[0]["p"]))
    for o in odds[1:]:
        if o["tag"] in ("gate", "evidence"):
            break
        lo += math.log(o["lr"])
    gate = next(o for o in odds if o["tag"] == "gate")
    assert abs(1 / (1 + math.exp(-lo)) - gate["p"]) < 2e-3
    assert round(odds[-1]["p"], 2) == t["answer"]["case"]["fraud_probability"]


@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_every_graph_claim_cites_a_query_the_agent_ran(path):
    t = json.loads(path.read_text())
    ran = {s["ref"] for s in t["steps"]}
    for e in t["answer"]["case"]["evidence"]:
        if e["ref"].startswith("query:"):
            assert e["ref"] in ran, e["ref"]


def test_the_ring_traversal_reaches_all_28_cards():
    t = json.loads((DATA / "HHG-014.json").read_text())
    step = next(s for s in t["steps"] if s["query"] == "device_neighbors")
    assert step["n"] == len(step["txns"]) == 60
    assert len({x["card"] for x in step["txns"]}) == 28
