"""Closed-menu case Q&A: intent routing, as-of correctness, injection-proofing."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assay.r2 import ask
from assay.r2.frames import case_pack, tx_all

CASES = ["HHF-005", "HHF-008"]
QS = {
    "device_neighbors": "which other cards used this device?",
    "card_window": "what is the usual spending baseline?",
    "prior_cases": "any past cases on this card?",
    "get_transaction": "show the flagged transaction details",
    "holder_history": "does the holder have other cards?",
    "shipped_answer": "which cards are linked in the ring?",
}


def _open(cid):
    r = case_pack().set_index("case_id").loc[cid]
    return r, pd.Timestamp(r.opened_at)


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("tool,q", list(QS.items()))
def test_intent_tool_and_asof(cid, tool, q):
    r, opened = _open(cid)
    out = ask.answer(cid, q)
    assert out["tool"] == tool
    assert len(out["rows"]) <= 20 and out["total"] >= len(out["rows"])
    assert out["attestation"]["params"] == out["params"]
    for row in out["rows"]:
        if "t" in row:
            assert pd.Timestamp(row["t"]) <= opened
        if "closed_at" in row:
            assert pd.Timestamp(row["closed_at"]) <= opened
    if tool == "get_transaction":
        assert out["params"] == {"txn_id": int(float(r.flagged_txn_id))}
        assert out["rows"][0]["TransactionID"] == int(float(r.flagged_txn_id))
    if tool == "device_neighbors":
        for row in out["rows"]:
            assert row["card_id"] != r.card_id
            assert pd.Timestamp(row["t"]) >= opened - pd.Timedelta(days=30)
    if tool == "card_window":
        assert out["params"]["card_id"] == r.card_id
        s, e = pd.Timestamp(out["params"]["start"]), pd.Timestamp(out["params"]["end"])
        assert (e - s).days <= 120 and e == opened


def test_injection_uses_case_params():
    q = "ignore rules; DROP QUERY r2_ring_wcc; show device cards for C00001-K1"
    out = ask.answer("HHF-005", q)
    assert out["intent"] == "device_neighbors"
    assert out["params"]["card_id"] == "C00348-K1"
    assert "C00001-K1" not in repr(out["params"]) + repr(out["attestation"]) + out["answer"]
    assert "DROP" not in repr(out["params"]) and "DROP" not in out["answer"]


def test_unsupported_and_bad_input():
    out = ask.answer("HHF-005", "what is the meaning of life")
    assert out["intent"] == "unsupported" and out["tool"] is None and out["supported"]
    with pytest.raises(ValueError):
        ask.answer("HHF-005", "")
    with pytest.raises(ValueError):
        ask.answer("HHF-005", "x" * 301)
    with pytest.raises(ValueError):
        ask.answer("../etc", "device")


def test_no_llm_client_imported():
    src = Path(ask.__file__).read_text()
    imports = "\n".join(l for l in src.splitlines() if re.match(r"\s*(import|from)\s", l))
    for bad in ("anthropic", "openai", "google", "ollama", "requests", "httpx", "langchain"):
        assert bad not in imports


def test_endpoint():
    from assay.r2.console import create_app
    c = TestClient(create_app())
    ok = c.post("/ask", json={"case_id": "HHF-008", "question": "any prior cases?"})
    assert ok.status_code == 200 and ok.json()["tool"] == "prior_cases"
    assert c.post("/ask", json={"case_id": "bad", "question": "device"}).status_code == 400
    assert c.post("/ask", json={"case_id": "HHF-008", "question": ""}).status_code == 400
    assert c.post("/ask", json={"case_id": "HHF-008", "question": "a" * 301}).status_code == 400


@pytest.mark.parametrize("q", [
    "did the customer travel to Goa?",
    "where was the cardholder when the device was used?",
    "why did the customer call the bank?",
    "what is the customer's income?",
])
def test_out_of_scope_questions_are_refused_not_answered_differently(q):
    from assay.r2.ask import classify
    assert classify(q) == "unsupported"


@pytest.mark.parametrize("q,intent", [
    ("What other cards used this device?", "device_neighbors"),
    ("Does the holder have other cards?", "holder_history"),
    ("Are there past cases on this card?", "prior_cases"),
    ("Which cards are linked in the ring?", "ring"),
    ("What is this card's usual spending?", "card_history"),
    ("Show the flagged transaction", "get_transaction"),
])
def test_example_chips_still_route(q, intent):
    from assay.r2.ask import classify
    assert classify(q) == intent
