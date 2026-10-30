"""Challenge 13 on graph schemas: parser, mapper, scorer, compiler, approval.
Offline: no LLM, no TigerGraph. Handout-bound checks skip without the data."""

import json
from pathlib import Path

import pytest

from assay.r2.graphmap import proposal as P
from assay.r2.graphmap.compile import Unresolved, compile_fanin, where
from assay.r2.graphmap.gsql import ground_truth, parse, parse_text, strip_comments
from assay.r2.graphmap.mapper import map_schema
from assay.r2.graphmap.score import grade
from assay.r2.graphmap.world import PLANTED, Lowerer, build

HANDOUT = Path(__file__).resolve().parents[2] / "data/round2/graphmap_handout"

MINI = """# Expected mapping (ground truth):
#   Person        exact   Client where kind = "PERSON"
#   Transaction   exact   Posting (not Ledger)
#   INITIATED     exact   reverse_Posting_Debit
CREATE GRAPH Bank_Mini ()
USE GRAPH Bank_Mini
CREATE SCHEMA_CHANGE JOB j FOR GRAPH Bank_Mini {
  ADD VERTEX Client (PRIMARY_ID clientId STRING, legalNameHash STRING, birthDate DATETIME, isCustomer BOOL) WITH primary_id_as_attribute="true";
  ADD VERTEX Acct (PRIMARY_ID acctNo STRING, openingDate DATETIME, status STRING, overdraftLimitAmount DOUBLE) WITH primary_id_as_attribute="true";
  ADD VERTEX Posting (PRIMARY_ID ref STRING, amount DOUBLE, postingDateTime DATETIME) WITH primary_id_as_attribute="true";
  ADD VERTEX Ledger (PRIMARY_ID id STRING, amount DOUBLE, bookingDate DATETIME) WITH primary_id_as_attribute="true";
  ADD VERTEX PaymentOrder (PRIMARY_ID id STRING, orderAmount DOUBLE, orderDateTime DATETIME) WITH primary_id_as_attribute="true";
  ADD DIRECTED EDGE holder (FROM Client, TO Acct) WITH REVERSE_EDGE="reverse_holder";
  ADD DIRECTED EDGE Posting_Debit (FROM Posting, TO Acct) WITH REVERSE_EDGE="reverse_Posting_Debit";
  ADD DIRECTED EDGE Posting_Credit (FROM Posting, TO Acct) WITH REVERSE_EDGE="reverse_Posting_Credit";
  ADD DIRECTED EDGE Posting_Creditor (FROM Posting, TO Client) WITH REVERSE_EDGE="reverse_Posting_Creditor";
  ADD DIRECTED EDGE Ledger_Reference (FROM Ledger, TO Posting) WITH REVERSE_EDGE="reverse_Ledger_Reference";
  ADD DIRECTED EDGE executes (FROM Posting, TO PaymentOrder) WITH REVERSE_EDGE="reverse_executes";
  ADD DIRECTED EDGE orderDebits (FROM PaymentOrder, TO Acct) WITH REVERSE_EDGE="reverse_orderDebits";
}
"""


def test_comments_never_reach_the_mapper():
    with_gt = map_schema(parse_text(MINI)).to_json()
    without = map_schema(parse_text(strip_comments(MINI))).to_json()
    assert with_gt == without
    assert "Expected" not in strip_comments(MINI)


def test_ground_truth_is_read_for_the_scorer_only(tmp_path):
    p = tmp_path / "mini.gsql"
    p.write_text(MINI)
    gt = ground_truth(p)
    assert gt["Transaction"]["status"] == "exact" and "Posting" in gt["Transaction"]["rest"]


def test_transaction_is_the_movement_not_ledger_order_or_limit():
    m = map_schema(parse_text(MINI))
    assert m.vertices("Transaction") == ["Posting"]
    why = {r["candidate"]: r["why"] for r in m.choices["Transaction"].rejected}
    assert "Ledger" in why and "PaymentOrder" in why
    assert "Acct" not in m.vertices("Transaction")      # overdraftLimitAmount is not a payment amount
    assert m.vertices("Account") == ["Acct"]


def test_edges_walk_backwards_and_pick_the_debit_side():
    m = map_schema(parse_text(MINI))
    path = m.choices["INITIATED"].sources[0]["path"]
    assert [s["gsql"] for s in path] == ["reverse_Posting_Debit"]
    assert [s["gsql"] for s in m.choices["PAYS_TO_ACCOUNT"].sources[0]["path"]] == ["Posting_Credit"]
    # INITIATED is account-level debit: never routed through the order stage
    assert all("orderDebits" not in s["edge"] for src in m.choices["INITIATED"].sources for s in src["path"])


def test_shared_party_type_becomes_a_filter():
    m = map_schema(parse_text(MINI))
    assert m.choices["Person"].sources[0]["filter"] == "isCustomer = true"
    ben = m.choices["Beneficiary"]
    assert ben.status == "approximate" and ben.sources[0]["filter"] == "isCustomer = false"
    assert m.choices["PAYS"].status == "approximate"     # the payee is a role


def test_scorer_counts_agreement(tmp_path):
    p = tmp_path / "mini.gsql"
    p.write_text(MINI)
    s = parse(p)
    g = grade(map_schema(s), ground_truth(p), s)
    assert g["graded"] == 3 and g["right"] == 3


def test_compiler_refuses_unconfirmed_filters_and_reverses_steps():
    s = parse_text(MINI)
    m = map_schema(s)
    q, _ = compile_fanin(s, m)
    assert "-(Posting_Creditor:e)->" not in q and "reverse_Posting_Creditor" in q   # PAYS walked payee -> txn
    assert "s.isCustomer == FALSE" in q
    with pytest.raises(Unresolved):
        where(s, "Client", "kind = <counterparty value>", "s")


def test_lowering_plants_one_ring_and_one_decoy():
    w = build()
    pays = [(t, b) for t, b in w.e["PAYS"] if b == PLANTED["mule"]]
    payer = {t: a for a, t in w.e["INITIATED"]}
    assert {payer[t] for t, _ in pays} == PLANTED["mule_payers"]
    assert len({payer[t] for t, b in w.e["PAYS"] if b == PLANTED["decoy"]}) == 4
    s = parse_text(MINI)
    low = Lowerer(s, map_schema(s))
    low.lower(w)
    assert low.vertices["Posting"] and low.vertices["Client"]


def test_approval_needs_a_role_and_unapproved_cannot_run(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DIR", tmp_path)
    p = tmp_path / "mini.gsql"
    p.write_text(MINI)
    doc = P.propose(p, use_llm=False)
    with pytest.raises(PermissionError):
        P.approved_mapping(doc["proposal_id"])
    with pytest.raises(PermissionError):
        P.approve(doc["proposal_id"], "L1_analyst", "someone")
    P.approve(doc["proposal_id"], "L2_manager", "lead", take_rule=["OWNS"])
    m = P.approved_mapping(doc["proposal_id"])
    assert m.vertices("Transaction") == ["Posting"]
    assert json.loads((tmp_path / f"{doc['proposal_id']}.json").read_text())["approved"]["role"] == "L2_manager"


@pytest.mark.skipif(not HANDOUT.exists(), reason="organiser handout not present")
@pytest.mark.parametrize("name,graded", [("bank1_iso20022_payments", 25), ("bank2_bian_service_domains", 32),
                                         ("bank3_core_banking_t24", 25)])
def test_handout_rules_mapping_matches_ground_truth(name, graded):
    f = HANDOUT / f"{name}.gsql"
    s = parse(f)
    g = grade(map_schema(s), ground_truth(f), s)
    assert g["graded"] == graded and g["right"] == graded   # dev set: the rules were written against it
