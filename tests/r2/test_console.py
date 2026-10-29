"""Tests for BLOCK B26 -- Live demo console (`assay/r2/console.py`)."""

from __future__ import annotations

import time
from fastapi.testclient import TestClient

from assay.r2.console import DEMO_L1_TOKEN, DEMO_L2_TOKEN, create_app


def test_console_end_to_end():
    t0 = time.time()
    app = create_app()
    startup_s = time.time() - t0
    client = TestClient(app)

    # 1. Self-contained HTML page loads with no CDN links
    r_html = client.get("/")
    assert r_html.status_code == 200
    html = r_html.text
    assert "ASSAY R2" in html
    assert "treated as data" in html
    assert "awaiting approval" in html
    assert "https://cdn" not in html.lower()

    # 2. /cases lists all 18 cases and highlights prompt injections
    r_cases = client.get("/cases")
    assert r_cases.status_code == 200
    cases = r_cases.json()
    assert len(cases) == 18

    injected_ids = [c["case_id"] for c in cases if c["has_injection"]]
    assert set(injected_ids) == {
        "HHF-004",
        "HHF-005",
        "HHF-008",
        "HHF-010",
        "HHF-014",
        "HHF-016",
    }
    for c in cases:
        if c["has_injection"]:
            q_segs = [s for s in c["segments"] if s.get("quarantined")]
            assert len(q_segs) >= 1
            assert all(s.get("label") == "treated as data" for s in q_segs)

    # 3. /run/HHF-007 runs the real R2Investigator over R2LocalStore
    r_run = client.post("/run/HHF-007")
    assert r_run.status_code == 200
    data = r_run.json()
    assert data["ok"] is True
    assert data["fail_closed"] is False
    assert data["case_id"] == "HHF-007"
    assert data["verdict"] in ("fraud", "legitimate", "uncertain")
    assert isinstance(data["calls"], list) and len(data["calls"]) > 0
    assert all("row_count" in call and "params" in call for call in data["calls"])
    assert isinstance(data["ledger"], list) and len(data["ledger"]) > 0
    assert all("lr" in step and "probability_after" in step for step in data["ledger"])
    assert data["ledger"][-1]["probability_after"] == data["fraud_probability"]
    assert "visual" in data and isinstance(data["visual"], dict)
    assert len(data["visual"]["pulse_txns"]) >= 5
    assert data["visual"]["topology"]["hub_type"] == "card_testing"

    # 4. Approvals flow mounted under /approvals for L1/L2 actions
    gated_actions = [a for a in data["actions"] if a["approval_route"] in ("L1", "L2")]
    assert len(gated_actions) >= 1
    for ga in gated_actions:
        assert ga["status"] == "awaiting approval"
        pid = ga["proposal_id"]
        route = ga["approval_route"]

        # If L2, verify L1 token is rejected (403) before L2 token succeeds
        if route == "L2":
            r_deny = client.post(
                f"/approvals/proposals/{pid}/decision",
                headers={
                    "Authorization": f"Bearer {DEMO_L1_TOKEN}",
                    "Idempotency-Key": f"test-deny-{pid}",
                },
                json={"decision": "approve", "note": "wrong role test"},
            )
            assert r_deny.status_code == 403

        token = DEMO_L2_TOKEN if route == "L2" else DEMO_L1_TOKEN
        r_app = client.post(
            f"/approvals/proposals/{pid}/decision",
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": f"test-ok-{pid}",
            },
            json={"decision": "approve", "note": "approved in test"},
        )
        assert r_app.status_code == 200
        decided = r_app.json()
        assert decided["status"] == "approved"

    # Verify hash-chained ledger integrity via mounted /approvals/ledger/verify
    r_ver = client.get("/approvals/ledger/verify")
    assert r_ver.status_code == 200
    assert r_ver.json()["valid"] is True

    # 5. 'Break it' toggle: sets ASSAY_FAULT (including default card_history:down@1) and demonstrates fail-closed live
    r_fault = client.post("/fault", json={"fault": "card_history:down@1"})
    assert r_fault.status_code == 200
    assert r_fault.json()["active"] is True

    r_broken = client.post("/run/HHF-007")
    assert r_broken.status_code == 200
    b_data = r_broken.json()
    assert b_data["ok"] is False
    assert b_data["fail_closed"] is True
    assert b_data["answer"] is None
    assert "FAIL_CLOSED" in b_data["error"]
    assert len(b_data["calls"]) == 3

    # Clear fault and confirm recovery
    client.post("/fault", json={"fault": ""})
    r_recover = client.post("/run/HHF-007")
    assert r_recover.status_code == 200
    assert r_recover.json()["ok"] is True

    print(
        f"[B26 metrics] startup_s={startup_s:.2f}s cases={len(cases)} "
        f"injected={len(injected_ids)} HHF-007_verdict={data['verdict']} "
        f"HHF-007_p={data['fraud_probability']} HHF-007_exp=${data['exposure_usd']:.2f} "
        f"HHF-007_calls={len(data['calls'])} HHF-007_findings={len(data['ledger'])} "
        f"HHF-007_elapsed_ms={data['elapsed_ms']}ms"
    )


def test_verify_endpoint_reruns_claims():
    client = TestClient(create_app())
    r = client.get("/verify/HHF-007")
    assert r.status_code == 200
    d = r.json()
    assert {"VERIFIED", "MISMATCH", "UNVERIFIED", "UNRESOLVABLE"} <= set(d["counts"])
    assert d["counts"]["MISMATCH"] == 0
    assert d["total"] == len(d["claims"]) > 0
    assert d["elapsed_ms"] >= 0
    assert {"section", "index", "ref", "status", "reason", "claim"} <= set(d["claims"][0])
    assert client.get("/verify/BAD").status_code == 400
