"""Regression test for F016: TigerGraph vector search response key mismatch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from assay.r2 import vector


def _make_mini_index() -> vector.LocalVectorIndex:
    e0 = np.zeros(vector.EMBED_DIM, dtype=np.float64)
    e0[0] = 1.0
    e1 = np.zeros(vector.EMBED_DIM, dtype=np.float64)
    e1[1] = 1.0

    cc = pd.DataFrame(
        {
            "case_id": ["CC-001", "CC-002"],
            "card_id": ["C0001-K1", "C0002-K1"],
            "customer_id": ["C0001", "C0002"],
            "opened_at": ["2016-08-01 00:00:00", "2016-08-02 00:00:00"],
            "closed_at": ["2016-08-05 00:00:00", "2016-08-06 00:00:00"],
            "outcome": ["confirmed_fraud", "cleared"],
            "pattern": ["card_testing", "none"],
            "txn_list": [[101], [102]],
            "connected_list": [["C0099-K1"], []],
        }
    )
    cc_emb = cc[["case_id", "card_id", "customer_id", "opened_at", "closed_at", "outcome", "pattern"]].copy()
    cc_emb["embedding"] = [e0, e1]

    tx = pd.DataFrame(
        {
            "TransactionID": [101, 102],
            "t": pd.to_datetime(["2016-08-01 00:00:00", "2016-08-02 00:00:00"]),
            "TransactionDT": [1000.0, 2000.0],
            "amount": [15.0, 40.0],
            "channel": ["online", "online"],
            "ProductCD": ["C", "W"],
            "device_profile": ["devA", "devB"],
            "addr1": [100.0, 200.0],
            "D1": [10.0, 20.0],
            "P_emaildomain": ["a.com", "b.com"],
            "card_id": ["C0001-K1", "C0002-K1"],
            "risk_score": [0.9, 0.1],
        }
    )
    return vector.LocalVectorIndex(cc_emb, cc, tx)


def test_tg_vector_response_parses_aliased_keys_and_computes_distance(monkeypatch):
    idx = _make_mini_index()
    monkeypatch.setenv("ASSAY_TG", "1")

    fake_rows = [
        {
            "v_id": "CC-001",
            "attributes": {
                "case_id": "CC-001",
                "outcome": "confirmed_fraud",
                "pattern": "card_testing",
                "connecting_path": "DeviceProfile(D1)->ClosedCase(CC-001)",
                "shared_cards": ["C0001-K1", "C0099-K1"],
                "shared_devices": ["devA"],
            },
        },
        {
            "v_id": "CC-002",
            "attributes": {
                "case_id": "CC-002",
                "outcome": "cleared",
                "pattern": "none",
                "connecting_path": "BillingRegion(200)->ClosedCase(CC-002)",
                "shared_cards": ["C0002-K1"],
                "shared_devices": ["devB"],
            },
        },
    ]
    monkeypatch.setattr(vector, "_tg_run_query", lambda qname, params: fake_rows)

    qv = np.zeros(vector.EMBED_DIM, dtype=np.float64)
    qv[0] = 1.0  # identical to CC-001 (distance 0.0), orthogonal to CC-002 (distance 1.0)

    vec_hits = idx.similar_cases_vec(qv, k=2, as_of=pd.Timestamp("2016-09-01"))
    assert len(vec_hits) == 2
    assert vec_hits[0].shared_cards == ("C0001-K1", "C0099-K1")
    assert vec_hits[0].shared_devices == ("devA",)
    assert vec_hits[0].distance == 0.0
    assert vec_hits[1].distance == 1.0

    fused_hits = idx.similar_cases_fused("C0001-K1", qv, k=2, as_of=pd.Timestamp("2016-09-01"))
    assert len(fused_hits) == 2
    assert fused_hits[0].shared_cards == ("C0001-K1", "C0099-K1")
    assert fused_hits[0].shared_devices == ("devA",)
    assert fused_hits[0].distance == 0.0
    assert fused_hits[1].distance == 1.0

    # Distance-weighted k-NN should weight CC-001 (d=0.0, w=20) much higher than CC-002 (d=1.0, w=1/1.05)
    p_fraud = vector._knn_fraud_prob(fused_hits)
    assert p_fraud > 0.85
