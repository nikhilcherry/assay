"""Live tests for N2 -- TigerGraph as the Round 2 source of truth.

Tests:
1. Clean skip if TigerGraph is unreachable.
2. Idempotent `install()` + query smoke tests + `push_vectors()`.
3. `write_case()` and `read_cases()` round-trip verification (vertex attributes + edges).
4. `parity()` across all 18 exam cases (asserting well-formed responses, as-of
   correctness, attestation structure, and printing the full measurement & timing table).
5. Fail-closed verification (`TigerGraphUnavailableError` raised when TG is unreachable).
"""

from __future__ import annotations

import pandas as pd
import pytest

from assay.r2 import tg_r2
from assay.r2.frames import case_pack
from assay.r2.vector import TigerGraphUnavailableError


@pytest.fixture(scope="module", autouse=True)
def _require_live_tg():
    if not tg_r2.ping_tg():
        pytest.skip("TigerGraph CE 4.2.5 is not reachable at localhost:14240; skipping live TG tests.")


def test_install_and_push_vectors():
    inst = tg_r2.install()
    assert inst["vector_attribute_present"] is True
    assert inst["vector_mode"] in ("native_vector", "list_double_in_graph")
    for qname in tg_r2.REQUIRED_QUERIES:
        assert qname in inst["installed_queries"]
        if qname != "r2_upsert_case":
            assert qname in inst["smoke_test_ms"]
            assert inst["smoke_test_ms"][qname] >= 0.0

    # Verify idempotency: second install() call succeeds cleanly and quickly
    inst2 = tg_r2.install()
    assert inst2["vector_attribute_present"] is True

    pushed = tg_r2.push_vectors()
    assert pushed["n_vectors_pushed"] == 5585
    assert pushed["sample_top1_case_id"].startswith("CC-")
    print("\n[N2] install() mode:", inst["vector_mode"], "smoke_test_ms:", inst["smoke_test_ms"])
    print("[N2] push_vectors():", pushed)


def test_write_and_read_case_roundtrip():
    test_case_id = "HHF-TEST-N2"
    test_card_id = "C05061-K1"
    test_txn_id = 3660108
    test_connected_card = "C06220-K1"
    test_prior_case = "CC-0101"

    tg_r2.delete_fraud_case(test_case_id)
    try:
        sample_answer = {
            "case_id": test_case_id,
            "card_id": test_card_id,
            "stop_reason": "Fraud probability above 0.85 with two independent signals",
            "case": {
                "status": "escalated",
                "verdict": "fraud",
                "fraud_probability": 0.91,
                "pattern": "shared_origin_ring",
                "pattern_description": "Coordinated shared-origin activity across linked cards",
                "exposure_usd": 412.50,
                "summary": "N2 round-trip verification case on C05061-K1",
                "affected_txn_ids": [test_txn_id],
                "connected_card_ids": [test_connected_card],
                "connected_device_profiles": [],
                "similar_prior_cases": [test_prior_case],
                "written_at": "2017-01-15 02:00:00",
            },
            "sar": {"file": True},
            "next_best_actions": {
                "final": [
                    {"action": "BLOCK_CARD", "route": "L1"},
                    {"action": "CREATE_CASE", "route": "auto"},
                    {"action": "FILE_REPORT", "route": "L2"},
                ]
            },
        }

        w_res = tg_r2.write_case(sample_answer, card_id=test_card_id)
        assert w_res["graph_case_id"] == f"ASSAY-{test_case_id}"
        assert w_res["elapsed_ms"] >= 0.0

        # Read back from primary card
        r_res = tg_r2.read_cases(test_card_id)
        assert r_res["card_id"] == test_card_id
        matching = [fc for fc in r_res["fraud_cases"] if fc["case_id"] == f"ASSAY-{test_case_id}"]
        assert len(matching) == 1
        fc = matching[0]
        assert fc["alert_id"] == test_case_id
        assert fc["verdict"] == "fraud"
        assert abs(fc["fraud_probability"] - 0.91) < 1e-4
        assert fc["pattern"] == "shared_origin_ring"
        assert abs(fc["exposure_usd"] - 412.50) < 1e-2
        assert fc["sar_filed"] is True
        assert fc["final_actions"] == "BLOCK_CARD|CREATE_CASE|FILE_REPORT"
        assert test_txn_id in fc["cited_txns"]
        assert test_connected_card in fc["linked_cards"]
        assert test_prior_case in fc["drew_on_cases"]

        # Also verify read-back from the connected card via LINKED_IN_CASE reverse edge
        r_conn = tg_r2.read_cases(test_connected_card)
        conn_matching = [x for x in r_conn["fraud_cases"] if x["case_id"] == f"ASSAY-{test_case_id}"]
        assert len(conn_matching) == 1
        print(
            f"\n[N2] write_case/read_cases round-trip OK: write={w_res['elapsed_ms']}ms, read={r_res['elapsed_ms']}ms"
        )
    finally:
        tg_r2.delete_fraud_case(test_case_id)


def test_parity_and_as_of_on_18_exam_cases():
    rep = tg_r2.parity(k=5)
    assert rep["n_cases"] == 18
    assert len(rep["rows"]) == 18

    cp = case_pack().set_index("case_id")
    for row in rep["rows"]:
        opened_at = pd.Timestamp(cp.loc[row.case_id, "opened_at"])
        assert len(row.tg_vec_ids) > 0
        assert len(row.tg_ring_members) >= 1
        assert row.card_id in row.tg_ring_members
        assert row.vec_ms >= 0.0
        assert row.fused_ms >= 0.0
        assert row.ring_ms >= 0.0

        # Verify as-of correctness and attestation schema on vector_precedent and ring_wcc
        vp = tg_r2.vector_precedent(cp.loc[row.case_id].to_dict() | {"case_id": row.case_id}, k=5)
        for h in vp["vec_hits"] + vp["fused_hits"]:
            assert pd.Timestamp(h["closed_at"]) <= opened_at, (
                f"Future closed case leaked in {row.case_id}: {h['case_id']} closed_at {h['closed_at']} > {opened_at}"
            )
        att_dict = vp["attestation"].to_dict()
        assert set(att_dict.keys()) == {"what", "evidence", "how", "when"}
        assert att_dict["how"]["query"] in ("similar_cases_fused", "similar_cases_vec")

    print(
        f"\n[N2] Parity summary: vec_top1_agree={rep['vec_top1_agreement']}/18, "
        f"mean_vec_ov@5={rep['mean_vec_overlap_at_k']}, "
        f"mean_fused_ov@5={rep['mean_fused_overlap_at_k']}, "
        f"ring_exact_agree={rep['ring_exact_agreement']}/18, "
        f"mean_ring_jaccard={rep['mean_ring_jaccard']}"
    )
    print(
        f"[N2] Latencies (ms): vec mean={rep['mean_vec_ms']} (p95={rep['p95_vec_ms']}), "
        f"fused mean={rep['mean_fused_ms']} (p95={rep['p95_fused_ms']}), "
        f"ring_wcc mean={rep['mean_ring_ms']} (p95={rep['p95_ring_ms']})"
    )


def test_fail_closed_when_tg_unreachable():
    with pytest.raises(TigerGraphUnavailableError) as exc_info:
        tg_r2.require_tg(host="http://127.0.0.1", port=19999, timeout=0.5)
    assert "Fail-closed" in str(exc_info.value)

    with pytest.raises(TigerGraphUnavailableError):
        tg_r2.run_query("r2_read_cases", {"card": "C05061-K1"}, host="http://127.0.0.1", port=19999, timeout=0.5)
