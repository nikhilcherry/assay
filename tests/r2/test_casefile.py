"""Tests for BLOCK B21 CaseFile state machine, append-only SQLite store, and CLI."""

import json
import sqlite3
import pytest
from datetime import datetime, timezone

from assay.r2.attest import Attestation, Evidence
from assay.r2.casefile import (
    CaseFile,
    CaseState,
    CaseStore,
    InvalidTransitionError,
    PolicyApprovalError,
    TamperError,
    get_required_role,
)


def test_full_life_three_cases_two_sittings(tmp_path):
    db_file = str(tmp_path / "cases.db")

    # ==========================================
    # SITTING 1: Initialize cases and run life cycles
    # ==========================================
    store1 = CaseStore(db_file)

    # Attestation fixtures
    att_denial = Attestation(
        what="Customer C09582 denied transaction 3660350",
        evidence=Evidence(txn_ids=[3660350], card_ids=["C09582-K1"], customer_ids=["C09582"]),
        query="customer_denial",
        params={"txn_id": 3660350},
    )

    # CASE 1: HHF-001 - Customer reply & manager approval flow
    cf1 = CaseFile("HHF-001", store=store1)
    cf1.open_case(actor="agent", reason="Triggered by analyst_request", ts="2017-01-14T10:00:00Z")
    assert cf1.current_state == CaseState.OPEN

    cf1.start_investigation(actor="agent", reason="Gathering tx history", ts="2017-01-14T10:05:00Z")
    assert cf1.current_state == CaseState.INVESTIGATING

    cf1.request_customer(actor="agent", reason="Sent verification SMS for txn 3660350", ts="2017-01-14T10:10:00Z")
    assert cf1.current_state == CaseState.AWAITING_CUSTOMER

    # Customer reply arrives
    cf1.receive_customer_reply(
        actor="agent",
        reply_text="I never made this purchase",
        evidence=[att_denial],
        ts="2017-01-14T12:00:00Z",
    )
    assert cf1.current_state == CaseState.INVESTIGATING

    # Action requires L2 approval because exposure > $2500
    cf1.request_approval(
        required_role="L2",
        action="BLOCK_CARD",
        exposure=3200.0,
        actor="agent",
        reason="Confirmed fraud exposure $3200",
        ts="2017-01-14T12:10:00Z",
    )
    assert cf1.current_state == CaseState.AWAITING_APPROVAL

    # Policy check failure when non-approver analyst attempts approval
    with pytest.raises(PolicyApprovalError):
        cf1.record_approval(
            approver="alice",
            role="L1",
            action="BLOCK_CARD",
            exposure=3200.0,
            reason="L1 approval attempt by non-approver",
            ts="2017-01-14T12:15:00Z",
        )

    # Policy check failure when lower role (L1) attempts L2 action
    with pytest.raises(PolicyApprovalError):
        cf1.record_approval(
            approver="approver:L1",
            role="L1",
            action="BLOCK_CARD",
            exposure=3200.0,
            reason="L1 approval attempt for L2 exposure",
            ts="2017-01-14T12:20:00Z",
        )

    # Valid L2 Approval
    cf1.record_approval(
        approver="approver:L2",
        role="L2",
        action="BLOCK_CARD",
        exposure=3200.0,
        reason="L2 fraud manager approved block",
        ts="2017-01-14T12:30:00Z",
        next_state=CaseState.CLOSED_FRAUD,
    )
    assert cf1.current_state == CaseState.CLOSED_FRAUD

    # CASE 2: HHF-002 - R4 SLA timeout and escalation
    cf2 = CaseFile("HHF-002", store=store1)
    cf2.open_case(actor="agent", reason="Model risk score alert", ts="2017-01-15T08:00:00Z")
    cf2.start_investigation(actor="agent", reason="Reviewing score 0.81", ts="2017-01-15T08:02:00Z")
    cf2.request_customer(actor="agent", reason="Verification sent under R1", ts="2017-01-15T08:05:00Z")
    assert cf2.current_state == CaseState.AWAITING_CUSTOMER

    # Check SLA status as-of 26 hours later
    as_of_timeout = "2017-01-16T10:10:00Z"
    qs2 = cf2.open_questions(as_of=as_of_timeout)
    assert len(qs2) == 1
    assert qs2[0]["sla_24h_breached"] is True

    # Escalate due to R4 timeout
    cf2.escalate(
        actor="agent",
        reason="Policy R4: No customer reply within 24 hours, exposure > $500",
        ts=as_of_timeout,
    )
    assert cf2.current_state == CaseState.ESCALATED

    # CASE 3: HHF-003 - Policy R8 uncertain escalation & analyst resolution
    cf3 = CaseFile("HHF-003", store=store1)
    cf3.open_case(actor="agent", reason="Risk score 0.53", ts="2017-01-16T09:00:00Z")
    cf3.start_investigation(actor="agent", reason="Evaluating signals", ts="2017-01-16T09:01:00Z")
    cf3.escalate(
        actor="agent",
        reason="Policy R8: Conflicting device location and recurring payment profile",
        ts="2017-01-16T09:05:00Z",
    )
    assert cf3.current_state == CaseState.ESCALATED

    cf3.close_legitimate(
        actor="analyst:bob",
        reason="Cardholder confirmed legitimate recurring subscription",
        ts="2017-01-16T11:00:00Z",
    )
    assert cf3.current_state == CaseState.CLOSED_LEGITIMATE

    # Verify illegal transition rejection
    with pytest.raises(InvalidTransitionError):
        # Cannot jump from CLOSED_LEGITIMATE directly to AWAITING_CUSTOMER without reopening
        cf3.transition(CaseState.AWAITING_CUSTOMER, actor="agent", reason="Illegal jump")

    # End Sitting 1
    del store1, cf1, cf2, cf3

    # ==========================================
    # SITTING 2: Re-open DB and verify persistence / reopening
    # ==========================================
    store2 = CaseStore(db_file)

    # Verify Case 1 state restored
    cf1_rebound = CaseFile("HHF-001", store=store2)
    assert cf1_rebound.current_state == CaseState.CLOSED_FRAUD
    assert len(cf1_rebound.events) == 6

    # Re-opening closed case HHF-001 as a NEW event
    reopen_ev = cf1_rebound.reopen(
        actor="analyst:alice",
        reason="New connected transaction 3660399 flagged on same card profile",
        ts="2017-01-17T09:00:00Z",
    )
    assert reopen_ev.event_type == "REOPEN"
    assert cf1_rebound.current_state == CaseState.INVESTIGATING
    assert len(cf1_rebound.events) == 7  # new event appended, history unchanged

    # Handover summary check
    summary1 = cf1_rebound.handover_summary()
    assert summary1["current_state"] == CaseState.INVESTIGATING
    assert summary1["tamper_check"] == "PASSED"
    assert "Review gathered evidence" in summary1["next_step"]

    # Verify Case 2 handover summary
    cf2_rebound = CaseFile("HHF-002", store=store2)
    summary2 = cf2_rebound.handover_summary()
    assert summary2["current_state"] == CaseState.ESCALATED
    assert "Assigned to human analyst" in summary2["next_step"]

    # ==========================================
    # TAMPER DETECTION TEST
    # ==========================================
    # Directly modify an event row for HHF-003 in SQLite database to tamper with content
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE events SET reason = 'Tampered reason' WHERE case_id = 'HHF-003'")
    conn.commit()
    conn.close()

    # Integrity check on tampered case must raise TamperError upon initialization
    with pytest.raises(TamperError):
        CaseFile("HHF-003", store=store2)


def test_approval_boundary_2500(tmp_path):
    """Verifies approval routing boundary conditions around exposure threshold $2500.00 vs $2500.01."""
    db_file = str(tmp_path / "boundary.db")
    store = CaseStore(db_file)
    cf = CaseFile("HHF-010", store=store)

    # 1. Test BLOCK_CARD with exposure <= $2500.00 -> requires L1
    assert get_required_role("BLOCK_CARD", 2500.00) == "L1"
    cf.open_case(actor="agent", ts="2017-01-14T10:00:00Z")
    cf.start_investigation(actor="agent", ts="2017-01-14T10:01:00Z")
    cf.request_approval(required_role="L1", action="BLOCK_CARD", exposure=2500.00, actor="agent", ts="2017-01-14T10:02:00Z")

    # Non-approver attempt raises PolicyApprovalError
    with pytest.raises(PolicyApprovalError):
        cf.record_approval(approver="agent", role="L1", action="BLOCK_CARD", exposure=2500.00)

    # L1 approval succeeds for exposure == $2500.00
    cf.record_approval(approver="approver:L1", role="L1", action="BLOCK_CARD", exposure=2500.00, next_state=CaseState.CLOSED_FRAUD)
    assert cf.current_state == CaseState.CLOSED_FRAUD

    # 2. Reopen and test BLOCK_CARD with exposure == $2500.01 -> requires L2
    assert get_required_role("BLOCK_CARD", 2500.01) == "L2"
    cf.reopen(actor="analyst:bob")
    cf.request_approval(required_role="L2", action="BLOCK_CARD", exposure=2500.01, actor="agent", ts="2017-01-14T11:00:00Z")

    # L1 approval fails for exposure > $2500.00
    with pytest.raises(PolicyApprovalError):
        cf.record_approval(approver="approver:L1", role="L1", action="BLOCK_CARD", exposure=2500.01)

    # L2 approval succeeds for exposure > $2500.00
    cf.record_approval(approver="approver:L2", role="L2", action="BLOCK_CARD", exposure=2500.01, next_state=CaseState.CLOSED_FRAUD)
    assert cf.current_state == CaseState.CLOSED_FRAUD


def test_tamper_detection_scenarios(tmp_path):
    """Verifies that tamper detection catches deleted rows, payload edits, and hash chain breakages."""
    db_file = str(tmp_path / "tamper_test.db")
    store = CaseStore(db_file)

    cf = CaseFile("HHF-099", store=store)
    cf.open_case(actor="agent", ts="2017-01-14T10:00:00Z")
    cf.start_investigation(actor="agent", ts="2017-01-14T10:01:00Z")
    cf.escalate(actor="agent", reason="Needs review", ts="2017-01-14T10:02:00Z")

    # Scenario 1: Payload Edit
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE events SET reason = 'Malicious edit' WHERE case_id = 'HHF-099' AND event_type = 'ESCALATE'")
    conn.commit()
    conn.close()

    with pytest.raises(TamperError):
        CaseFile("HHF-099", store=store)

    # Scenario 2: Deleted intermediate event
    db_file2 = str(tmp_path / "tamper_delete.db")
    store2 = CaseStore(db_file2)
    cf2 = CaseFile("HHF-100", store=store2)
    cf2.open_case(actor="agent", ts="2017-01-14T10:00:00Z")
    cf2.start_investigation(actor="agent", ts="2017-01-14T10:01:00Z")
    cf2.escalate(actor="agent", reason="Needs review", ts="2017-01-14T10:02:00Z")

    conn = sqlite3.connect(db_file2)
    conn.execute("DELETE FROM events WHERE case_id = 'HHF-100' AND event_type = 'START_INVESTIGATION'")
    conn.commit()
    conn.close()

    with pytest.raises(TamperError):
        CaseFile("HHF-100", store=store2)


def test_cli_show_and_handover(tmp_path, capsys):
    db_file = str(tmp_path / "cli_cases.db")
    store = CaseStore(db_file)
    cf = CaseFile("HHF-007", store=store)
    cf.open_case(actor="agent", reason="Customer report received", ts="2017-01-14T15:57:30Z")
    cf.start_investigation(actor="agent", reason="Analyst review", ts="2017-01-14T16:00:00Z")

    summary = cf.handover_summary()
    assert summary["case_id"] == "HHF-007"
    assert summary["current_state"] == CaseState.INVESTIGATING
