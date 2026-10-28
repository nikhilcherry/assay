"""Tests for Block B10 -- Server-side approvals, audit ledger & drill hardening."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assay.r2.approvals import cli, create_app, route_for

AGENT_TOKEN = "agent-token-123"
L1_TOKEN_ALICE = "l1-token-alice"
L1_TOKEN_BOB = "l1-token-bob"
L2_TOKEN_CAROL = "l2-token-carol"


def hash_tok(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@pytest.fixture
def test_setup(tmp_path: Path):
    approvers_path = tmp_path / "approvers.json"
    ledger_path = tmp_path / "ledger.jsonl"

    approvers_data = [
        {"name": "agent_bot", "role": "agent", "token_sha256": hash_tok(AGENT_TOKEN)},
        {"name": "alice_l1", "role": "L1", "token_sha256": hash_tok(L1_TOKEN_ALICE)},
        {"name": "bob_l1", "role": "L1", "token_sha256": hash_tok(L1_TOKEN_BOB)},
        {"name": "carol_l2", "role": "L2", "token_sha256": hash_tok(L2_TOKEN_CAROL)},
    ]
    approvers_path.write_text(json.dumps(approvers_data), encoding="utf-8")

    app = create_app(ledger_path=ledger_path, approvers_path=approvers_path)
    client = TestClient(app)

    return client, ledger_path, approvers_path


def test_route_for_policy_mapping():
    assert route_for("ALLOW_TRANSACTION", 5000.0) == "auto"
    assert route_for("VERIFY_WITH_CUSTOMER", 10000.0) == "auto"
    assert route_for("DECLINE_TRANSACTION", 10.0) == "L1"
    assert route_for("BLOCK_CARD", 2500.00) == "L1"
    assert route_for("BLOCK_CARD", 2500.01) == "L2"
    assert route_for("BLOCK_ALL_CARDS", 50.0) == "L2"
    assert route_for("FILE_REPORT", 10.0) == "L2"


def test_missing_approvers_file_fails_closed(tmp_path: Path):
    missing_path = tmp_path / "non_existent_approvers.json"
    with pytest.raises(FileNotFoundError):
        create_app(ledger_path=tmp_path / "ledger.jsonl", approvers_path=missing_path)


def test_auto_route_executes_without_human(test_setup):
    client, ledger_path, _ = test_setup
    res = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={
            "case_id": "HHF-001",
            "action": "VERIFY_WITH_CUSTOMER",
            "exposure_usd": 150.0,
            "evidence_ref": "ref_1",
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["route"] == "auto"
    assert data["status"] == "auto_executed"
    assert data["decision"] == "approve"

    v_res = client.get("/ledger/verify")
    assert v_res.status_code == 200
    assert v_res.json()["valid"] is True


def test_exposure_boundary_2500_and_wrong_role_403(test_setup):
    client, _, _ = test_setup

    # 2500.00 is L1 route
    p1 = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-002", "action": "BLOCK_CARD", "exposure_usd": 2500.00},
    ).json()
    assert p1["route"] == "L1"

    # L1 user (Alice) approves L1 item
    d1 = client.post(
        f"/proposals/{p1['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "key-dec-1"},
        json={"decision": "approve", "note": "L1 ok"},
    )
    assert d1.status_code == 200
    assert d1.json()["status"] == "approved"

    # 2500.01 is L2 route
    p2 = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-003", "action": "BLOCK_CARD", "exposure_usd": 2500.01},
    ).json()
    assert p2["route"] == "L2"

    # L1 user tries to decide L2 item -> 403
    d2 = client.post(
        f"/proposals/{p2['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "key-dec-2"},
        json={"decision": "approve"},
    )
    assert d2.status_code == 403

    # L2 user decides L2 item -> 200
    d3 = client.post(
        f"/proposals/{p2['id']}/decision",
        headers={"Authorization": f"Bearer {L2_TOKEN_CAROL}", "Idempotency-Key": "key-dec-3"},
        json={"decision": "approve", "note": "L2 ok"},
    )
    assert d3.status_code == 200
    assert d3.json()["status"] == "approved"


def test_body_role_ignored(test_setup):
    client, _, _ = test_setup

    p = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-004", "action": "FILE_REPORT", "exposure_usd": 100.0},
    ).json()
    assert p["route"] == "L2"

    # L1 user sends body claiming role: L2 -> server ignores body role and rejects with 403
    res = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "key-body-role"},
        json={"decision": "approve", "role": "L2", "user": "carol_l2"},
    )
    assert res.status_code == 403


def test_agent_cannot_approve(test_setup):
    client, _, _ = test_setup

    p = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}"},
        json={"case_id": "HHF-005", "action": "DECLINE_TRANSACTION", "exposure_usd": 50.0},
    ).json()

    res = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}", "Idempotency-Key": "key-agent-app"},
        json={"decision": "approve"},
    )
    assert res.status_code == 403


def test_self_approval_forbidden(test_setup):
    client, _, _ = test_setup

    p = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}"},
        json={"case_id": "HHF-006", "action": "DECLINE_TRANSACTION", "exposure_usd": 50.0},
    ).json()

    # Alice attempts to approve her own proposal -> 403
    d1 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "key-self-1"},
        json={"decision": "approve"},
    )
    assert d1.status_code == 403

    # Bob approves Alice's proposal -> 200
    d2 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_BOB}", "Idempotency-Key": "key-self-2"},
        json={"decision": "approve"},
    )
    assert d2.status_code == 200


def test_double_decision_409(test_setup):
    client, _, _ = test_setup

    p = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-007", "action": "DECLINE_TRANSACTION", "exposure_usd": 50.0},
    ).json()

    d1 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "key-double-1"},
        json={"decision": "approve"},
    )
    assert d1.status_code == 200

    d2 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L2_TOKEN_CAROL}", "Idempotency-Key": "key-double-2"},
        json={"decision": "reject"},
    )
    assert d2.status_code == 409


def test_idempotency_key_replay_and_missing_header(test_setup):
    client, _, _ = test_setup

    p = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-008", "action": "DECLINE_TRANSACTION", "exposure_usd": 50.0},
    ).json()

    # Missing Idempotency-Key -> 400
    res_no_key = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}"},
        json={"decision": "approve"},
    )
    assert res_no_key.status_code == 400

    # First request -> 200
    body = {"decision": "approve", "note": "idempotency test"}
    res1 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "idem-key-100"},
        json=body,
    )
    assert res1.status_code == 200

    # Same Idempotency-Key + Same body -> returns original result
    res2 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "idem-key-100"},
        json=body,
    )
    assert res2.status_code == 200
    assert res2.json() == res1.json()

    # Same Idempotency-Key + Different body -> 409
    diff_body = {"decision": "reject", "note": "changed mind"}
    res3 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "idem-key-100"},
        json=diff_body,
    )
    assert res3.status_code == 409


def test_list_proposals_by_case_id(test_setup):
    client, _, _ = test_setup

    client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-010", "action": "VERIFY_WITH_CUSTOMER", "exposure_usd": 100.0},
    )
    client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-010", "action": "DECLINE_TRANSACTION", "exposure_usd": 100.0},
    )
    client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-011", "action": "VERIFY_WITH_CUSTOMER", "exposure_usd": 200.0},
    )

    # Filter by case_id
    res_filter = client.get("/proposals?case_id=HHF-010")
    assert res_filter.status_code == 200
    items = res_filter.json()
    assert len(items) == 2
    assert all(i["case_id"] == "HHF-010" for i in items)

    # List all
    res_all = client.get("/proposals")
    assert res_all.status_code == 200
    assert len(res_all.json()) >= 3


def test_rate_limit_wrong_role_attempts(test_setup):
    client, _, _ = test_setup

    p = client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-012", "action": "FILE_REPORT", "exposure_usd": 5000.0},
    ).json()

    # Alice (L1) makes 5 wrong role attempts on L2 proposal
    for i in range(5):
        res = client.post(
            f"/proposals/{p['id']}/decision",
            headers={
                "Authorization": f"Bearer {L1_TOKEN_ALICE}",
                "Idempotency-Key": f"rl-key-{i}",
            },
            json={"decision": "approve"},
        )
        assert res.status_code == 403

    # 6th wrong role attempt from Alice hits rate limit -> 429
    res6 = client.post(
        f"/proposals/{p['id']}/decision",
        headers={"Authorization": f"Bearer {L1_TOKEN_ALICE}", "Idempotency-Key": "rl-key-5"},
        json={"decision": "approve"},
    )
    assert res6.status_code == 429


def test_tampered_ledger_detected(test_setup):
    client, ledger_path, _ = test_setup

    client.post(
        "/proposals",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
        json={"case_id": "HHF-008", "action": "VERIFY_WITH_CUSTOMER", "exposure_usd": 10.0},
    )

    v1 = client.get("/ledger/verify").json()
    assert v1["valid"] is True

    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    tampered_entry = json.loads(lines[0])
    tampered_entry["data"]["case_id"] = "TAMPERED-CASE"
    lines[0] = json.dumps(tampered_entry)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    v2 = client.get("/ledger/verify").json()
    assert v2["valid"] is False
    assert "Hash mismatch" in v2["error"]


def test_cli_make_token(tmp_path: Path, monkeypatch, capsys):
    approvers_file = tmp_path / "cli_approvers.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approvals",
            "make-token",
            "--name",
            "dave_l2",
            "--role",
            "L2",
            "--approvers",
            str(approvers_file),
        ],
    )
    cli()

    captured = capsys.readouterr().out
    assert "TOKEN:" in captured
    assert "dave_l2" in captured

    # Verify file content stores only sha256
    with open(approvers_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["name"] == "dave_l2"
    assert data[0]["role"] == "L2"
    assert "token_sha256" in data[0]
    assert "TOKEN:" not in data[0]
