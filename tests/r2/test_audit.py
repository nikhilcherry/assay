"""Tests for BLOCK B11 -- Audit trail + data protection."""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from assay.r2.audit import (
    AuditLog,
    egress_register,
    redact,
    replay,
    scan_for_secrets,
)
from assay.r2.frames import case_pack


def test_audit_log_verify_clean(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    log = AuditLog(log_file)

    log.log_run_start("HHF-001", {"analyst": "unit_test"})
    log.log_tool_call("HHF-001", "get_card_history", {"card_id": "C05061-K1"}, result=[{"id": 1}], ms=12.5)
    log.log_finding("HHF-001", {"pattern": "shared_origin", "lr": 5.2})
    log.log_decision("HHF-001", {"action": "BLOCK_CARD"}, delta={"prev": "VERIFY", "new": "BLOCK"})
    log.log_run_end("HHF-001", "success")

    ok, line_num = log.verify()
    assert ok is True
    assert line_num == -1


def test_audit_log_detect_edit(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    log = AuditLog(log_file)

    log.log_run_start("HHF-001")
    log.log_tool_call("HHF-001", "query_tx", {"card_id": "C01001-K1"}, result=[{"tx": 101}])
    log.log_decision("HHF-001", {"action": "CLOSE_NO_FRAUD"})

    # Tamper with event at index 1
    lines = log_file.read_text(encoding="utf-8").splitlines()
    event_1 = json.loads(lines[1])
    event_1["params"]["card_id"] = "C01001-K9_TAMPERED"
    lines[1] = json.dumps(event_1)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad_line = log.verify()
    assert ok is False
    assert bad_line == 1


def test_audit_log_detect_delete(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    log = AuditLog(log_file)

    log.log_run_start("HHF-001")
    log.log_tool_call("HHF-001", "tool_1", {})
    log.log_tool_call("HHF-001", "tool_2", {})
    log.log_run_end("HHF-001")

    # Delete line 1 (tool_1)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    del lines[1]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad_line = log.verify()
    assert ok is False
    assert bad_line == 1


def test_audit_log_detect_reorder(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    log = AuditLog(log_file)

    log.log_run_start("HHF-001")
    log.log_tool_call("HHF-001", "tool_1", {})
    log.log_tool_call("HHF-001", "tool_2", {})
    log.log_run_end("HHF-001")

    # Swap line 1 and line 2
    lines = log_file.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad_line = log.verify()
    assert ok is False
    assert bad_line == 1


def test_redaction():
    fake_api_key = "AIza" + "Sy" + "D" * 33
    fake_aq_token = "AQ." + "Ab8" + "1" * 14
    fake_pwd = "supersecretpassword123"
    fake_bearer = "Bearer " + "eyJ" + "1" * 30
    fake_phone = "+91 " + "9876543210"
    fake_aadhaar = "2345 " + "6789 0123"
    fake_auth_header = "Authorization: " + "SecretKey12345"

    payload = {
        "api_key": fake_api_key,
        "auth_token": fake_aq_token,
        "user_password": fake_pwd,
        "email": "investigator.alice@bankdomain.com",
        "card_id": "C05061-K1",
        "customer_id": "C05061",
        "flagged_txn_id": 3660108,
        "free_text": "A" * 250,
        "headers": {"Authorization": fake_bearer},
        "indian_phone": fake_phone,
        "aadhaar": fake_aadhaar,
        "auth_line": fake_auth_header,
    }

    redacted = redact(payload)

    assert redacted["api_key"] == "[REDACTED_SECRET]"
    assert redacted["user_password"] == "[REDACTED_SECRET]"
    assert redacted["email"] == "[REDACTED_EMAIL]"
    assert redacted["card_id"] == "C05061-K1"
    assert redacted["customer_id"] == "C05061"
    assert redacted["flagged_txn_id"] == 3660108
    assert "TRUNCATED" in redacted["free_text"]
    assert len(redacted["free_text"]) < 250
    assert "[REDACTED_TOKEN]" in redacted["headers"]["Authorization"]
    assert "[REDACTED_PHONE]" in redacted["indian_phone"]
    assert "[REDACTED_AADHAAR]" in redacted["aadhaar"]
    assert redacted["auth_line"] == "Authorization: [REDACTED_AUTH]"


def test_redact_preserves_case_pack_ids():
    cp = case_pack()

    for row in cp.to_dict(orient="records"):
        redacted = redact(row)
        assert redacted["case_id"] == row["case_id"]
        assert redacted["flagged_txn_id"] == row["flagged_txn_id"]
        assert redacted["card_id"] == row["card_id"]
        assert redacted["customer_id"] == row["customer_id"]


def test_replay_order(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    log = AuditLog(log_file)

    log.log_run_start("HHF-001")
    log.log_tool_call("HHF-001", "query_tx", {"card_id": "C05061-K1"}, result=[123], row_count=1, ms=10.0)
    log.log_finding("HHF-001", {"pattern": "shared_origin"})
    log.log_decision("HHF-001", {"action": "BLOCK_CARD"}, delta={"step": 1})
    log.log_run_end("HHF-001")

    # Log events for another case interleaved
    log.log_run_start("HHF-002")
    log.log_tool_call("HHF-002", "query_tx", {"card_id": "C06220-K1"}, result=[], row_count=0, ms=5.0)

    steps_1 = replay(log_file, "HHF-001")
    assert len(steps_1) == 5
    assert steps_1[0]["event_type"] == "run_start"
    assert steps_1[1]["tool"] == "query_tx"
    assert steps_1[1]["row_count"] == 1
    assert "result_digest" in steps_1[1]
    assert steps_1[2]["event_type"] == "finding"
    assert steps_1[3]["decision"]["action"] == "BLOCK_CARD"
    assert steps_1[3]["delta"] == {"step": 1}

    steps_2 = replay(log_file, "HHF-002")
    assert len(steps_2) == 2
    assert steps_2[1]["tool"] == "query_tx"


def test_secret_scan_repo(capsys):
    repo_root = Path(__file__).parents[2]
    stats: dict = {}
    findings = scan_for_secrets(repo_root, stats)
    captured = capsys.readouterr()
    assert "Scanned" in captured.out
    assert stats["files_scanned"] > 0
    with capsys.disabled():
        print(f"\n[secret scan] scanned {stats['files_scanned']} files (tracked + untracked-not-ignored, data/ skipped); findings={len(findings)}")
    assert findings == [], f"Secrets found in tracked/untracked files: {findings}"


def test_egress_register():
    reg = egress_register()
    assert isinstance(reg, list)
    assert len(reg) >= 1
    assert reg[0]["decision_path"] is False


def test_cli_verify_and_replay(tmp_path: Path):
    log_file = tmp_path / "cli_audit.jsonl"
    log = AuditLog(log_file)

    log.log_run_start("HHF-001")
    log.log_tool_call("HHF-001", "query_tx", {"card_id": "C05061-K1"}, result=[{"tx": 1}], row_count=1, ms=8.0)
    log.log_decision("HHF-001", {"action": "BLOCK_CARD"}, delta={"from": "NONE", "to": "BLOCK"})
    log.log_run_end("HHF-001")

    # Run verify via subprocess
    res_v = subprocess.run(
        [sys.executable, "-m", "assay.r2.audit", "verify", str(log_file)],
        capture_output=True,
        text=True,
    )
    assert res_v.returncode == 0
    assert "is VALID" in res_v.stdout

    # Run replay via subprocess
    res_r = subprocess.run(
        [sys.executable, "-m", "assay.r2.audit", "replay", str(log_file), "HHF-001"],
        capture_output=True,
        text=True,
    )
    assert res_r.returncode == 0
    assert "STEP" in res_r.stdout
    assert "query_tx" in res_r.stdout
    assert "BLOCK_CARD" in res_r.stdout


def test_redact_masks_pii_but_not_ids():
    # Indian phone numbers
    for ph in ["+91 98765 43210", "+91-9876543210", "9876543210", "6123456789"]:
        out = redact(f"call {ph} now")
        assert "[REDACTED_PHONE]" in out, (ph, out)
        assert "98765" not in out and "6123" not in out
    # Aadhaar-like 12-digit groups
    for ad in ["2345 6789 0123", "2345-6789-0123", "234567890123"]:
        assert "[REDACTED_AADHAAR]" in redact(f"id {ad}"), ad
    # Anything after Authorization:
    out = redact("Authorization: " + "Basic " + "x" * 20 + " trailing")
    assert out == "Authorization: [REDACTED_AUTH]"

    # Id shapes must never be altered, as bare strings or in free text
    txn, card, cust = "3660153", "C09582-K1", "C09582"
    for s in [txn, card, cust, f"txn {txn} on card {card} of customer {cust}", "3583227", "C08106-K1"]:
        assert redact(s) == s, s
    rec = redact({"txn_id": 3660153, "card_id": card, "customer_id": cust, "note": f"{txn} {card} {cust}"})
    assert rec == {"txn_id": 3660153, "card_id": card, "customer_id": cust, "note": f"{txn} {card} {cust}"}
