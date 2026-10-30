"""BLOCK B21 -- Case management (Challenge 14).

State Machine:
  OPEN -> INVESTIGATING -> AWAITING_CUSTOMER | AWAITING_APPROVAL | ESCALATED -> CLOSED_FRAUD | CLOSED_LEGITIMATE

Append-only event log persisted per case in SQLite with SHA-256 hash chaining.
Re-opening a closed case appends a REOPEN event (never edits historical records).
Derived views include current state, open questions (SLA tracking under Policy R4),
decisions audit trail, and cold handover summary.
Policy approval requirements (L1 vs L2, $2500 threshold) are strictly enforced.
Tamper integrity is verified automatically on case loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from assay.r2.attest import Attestation, Evidence


class InvalidTransitionError(ValueError):
    """Raised when an illegal state transition is attempted in the case state machine."""
    pass


class TamperError(ValueError):
    """Raised when event log tamper or hash chain mismatch is detected."""
    pass


class PolicyApprovalError(ValueError):
    """Raised when an action requires an approval role that the actor does not possess."""
    pass


class CaseState:
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    AWAITING_CUSTOMER = "AWAITING_CUSTOMER"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    ESCALATED = "ESCALATED"
    CLOSED_FRAUD = "CLOSED_FRAUD"
    CLOSED_LEGITIMATE = "CLOSED_LEGITIMATE"


# Allowed state transitions according to policy & state machine workflow
VALID_TRANSITIONS: dict[str, set[str]] = {
    CaseState.OPEN: {
        CaseState.OPEN,
        CaseState.INVESTIGATING,
        CaseState.AWAITING_CUSTOMER,
        CaseState.AWAITING_APPROVAL,
        CaseState.ESCALATED,
        CaseState.CLOSED_FRAUD,
        CaseState.CLOSED_LEGITIMATE,
    },
    CaseState.INVESTIGATING: {
        CaseState.INVESTIGATING,
        CaseState.AWAITING_CUSTOMER,
        CaseState.AWAITING_APPROVAL,
        CaseState.ESCALATED,
        CaseState.CLOSED_FRAUD,
        CaseState.CLOSED_LEGITIMATE,
    },
    CaseState.AWAITING_CUSTOMER: {
        CaseState.AWAITING_CUSTOMER,
        CaseState.INVESTIGATING,
        CaseState.AWAITING_APPROVAL,
        CaseState.ESCALATED,
        CaseState.CLOSED_FRAUD,
        CaseState.CLOSED_LEGITIMATE,
    },
    CaseState.AWAITING_APPROVAL: {
        CaseState.AWAITING_APPROVAL,
        CaseState.INVESTIGATING,
        CaseState.AWAITING_CUSTOMER,
        CaseState.ESCALATED,
        CaseState.CLOSED_FRAUD,
        CaseState.CLOSED_LEGITIMATE,
    },
    CaseState.ESCALATED: {
        CaseState.ESCALATED,
        CaseState.INVESTIGATING,
        CaseState.AWAITING_CUSTOMER,
        CaseState.AWAITING_APPROVAL,
        CaseState.CLOSED_FRAUD,
        CaseState.CLOSED_LEGITIMATE,
    },
    CaseState.CLOSED_FRAUD: {
        CaseState.CLOSED_FRAUD,
        CaseState.OPEN,
        CaseState.INVESTIGATING,
    },
    CaseState.CLOSED_LEGITIMATE: {
        CaseState.CLOSED_LEGITIMATE,
        CaseState.OPEN,
        CaseState.INVESTIGATING,
    },
}


def get_required_role(action: Optional[str] = None, exposure: Optional[float] = None) -> Optional[str]:
    """Returns the required approval role ('L1' or 'L2') per Fraud Operations Policy Section 2."""
    if not action:
        return None
    act = action.upper()
    if act in ("BLOCK_ALL_CARDS", "FILE_REPORT"):
        return "L2"
    if act == "DECLINE_TRANSACTION":
        return "L1"
    if act == "BLOCK_CARD":
        if exposure is not None:
            return "L2" if exposure > 2500.00 else "L1"
        return "L1"
    return None


def _normalize_evidence(evidence: Optional[list]) -> list[dict]:
    """Converts evidence items (Attestation objects, dicts, or strings) into list of dicts."""
    if not evidence:
        return []
    res = []
    for item in evidence:
        if isinstance(item, Attestation):
            res.append(item.to_dict())
        elif hasattr(item, "to_dict") and callable(getattr(item, "to_dict")):
            res.append(item.to_dict())
        elif isinstance(item, dict):
            res.append(item)
        elif isinstance(item, str):
            res.append({"what": item})
        else:
            res.append({"item": str(item)})
    return res


def _parse_ts(ts_val: Any) -> datetime:
    """Parses ISO string or datetime object into a UTC datetime."""
    if isinstance(ts_val, datetime):
        dt = ts_val
    else:
        ts_str = str(ts_val).strip().replace("Z", "+00:00")
        if "T" not in ts_str and " " in ts_str:
            ts_str = ts_str.replace(" ", "T")
        try:
            dt = datetime.fromisoformat(ts_str)
        except ValueError:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class Event:
    id: int
    case_id: str
    event_type: str
    actor: str
    from_state: str
    to_state: str
    evidence_refs: list[dict]
    reason: str
    ts: str
    prev_hash: str
    hash: str


class CaseStore:
    """SQLite append-only event log with hash-chain integrity verification."""

    def __init__(self, db_path: str = "cases.db"):
        self.db_path = db_path
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if str(self.db_path) == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:")
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if str(self.db_path) != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    evidence_refs TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_case_id ON events(case_id);")
            conn.commit()

    def get_events(self, case_id: str) -> list[Event]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE case_id = ? ORDER BY id ASC", (case_id,)
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

    def get_last_event(self, case_id: str) -> Optional[Event]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE case_id = ? ORDER BY id DESC LIMIT 1", (case_id,)
            ).fetchone()
            return self._row_to_event(row) if row else None

    def append_event(
        self,
        case_id: str,
        event_type: str,
        actor: str,
        from_state: str,
        to_state: str,
        evidence_refs: list[dict],
        reason: str,
        ts: Optional[str] = None,
    ) -> Event:
        if ts is None:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        last = self.get_last_event(case_id)
        prev_hash = last.hash if last else "0" * 64
        evidence_json = json.dumps(evidence_refs, sort_keys=True)

        payload_dict = {
            "case_id": case_id,
            "event_type": event_type,
            "actor": actor,
            "from_state": from_state,
            "to_state": to_state,
            "evidence_refs": evidence_json,
            "reason": reason,
            "ts": ts,
            "prev_hash": prev_hash,
        }
        payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
        event_hash = hashlib.sha256(payload_bytes).hexdigest()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (case_id, event_type, actor, from_state, to_state, evidence_refs, reason, ts, prev_hash, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (case_id, event_type, actor, from_state, to_state, evidence_json, reason, ts, prev_hash, event_hash),
            )
            event_id = cursor.lastrowid
            conn.commit()

        return Event(
            id=event_id,
            case_id=case_id,
            event_type=event_type,
            actor=actor,
            from_state=from_state,
            to_state=to_state,
            evidence_refs=evidence_refs,
            reason=reason,
            ts=ts,
            prev_hash=prev_hash,
            hash=event_hash,
        )

    def verify_integrity(self, case_id: Optional[str] = None) -> bool:
        """Verifies hash chain integrity across stored events."""
        with self._get_conn() as conn:
            if case_id:
                rows = conn.execute("SELECT * FROM events WHERE case_id = ? ORDER BY id ASC", (case_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM events ORDER BY id ASC").fetchall()

        chains: dict[str, str] = {}
        for row in rows:
            c_id = row["case_id"]
            expected_prev = chains.get(c_id, "0" * 64)
            if row["prev_hash"] != expected_prev:
                raise TamperError(
                    f"Tamper detected in case {c_id} at event id {row['id']}: "
                    f"prev_hash mismatch ({row['prev_hash']} != expected {expected_prev})"
                )

            payload_dict = {
                "case_id": c_id,
                "event_type": row["event_type"],
                "actor": row["actor"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "evidence_refs": row["evidence_refs"],
                "reason": row["reason"],
                "ts": row["ts"],
                "prev_hash": row["prev_hash"],
            }
            recomputed = hashlib.sha256(json.dumps(payload_dict, sort_keys=True).encode("utf-8")).hexdigest()
            if row["hash"] != recomputed:
                raise TamperError(
                    f"Tamper detected in case {c_id} at event id {row['id']}: "
                    f"hash mismatch ({row['hash']} != recomputed {recomputed})"
                )

            chains[c_id] = row["hash"]

        return True

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        evidence = json.loads(row["evidence_refs"]) if row["evidence_refs"] else []
        return Event(
            id=row["id"],
            case_id=row["case_id"],
            event_type=row["event_type"],
            actor=row["actor"],
            from_state=row["from_state"],
            to_state=row["to_state"],
            evidence_refs=evidence,
            reason=row["reason"],
            ts=row["ts"],
            prev_hash=row["prev_hash"],
            hash=row["hash"],
        )


class CaseFile:
    """Case state machine and event log interface."""

    def __init__(self, case_id: str, store: Optional[CaseStore] = None, db_path: str = "cases.db", auto_verify: bool = True):
        self.case_id = case_id
        self.store = store if store is not None else CaseStore(db_path)
        if auto_verify and self.store.get_events(self.case_id):
            self.verify_integrity()

    @property
    def events(self) -> list[Event]:
        return self.store.get_events(self.case_id)

    @property
    def current_state(self) -> str:
        last = self.store.get_last_event(self.case_id)
        return last.to_state if last else CaseState.OPEN

    def verify_integrity(self) -> bool:
        return self.store.verify_integrity(self.case_id)

    def open_case(
        self,
        actor: str = "agent",
        reason: str = "Case opened",
        evidence: Optional[list] = None,
        ts: Optional[str] = None,
    ) -> Event:
        last = self.store.get_last_event(self.case_id)
        if last is not None:
            if last.to_state not in (CaseState.CLOSED_FRAUD, CaseState.CLOSED_LEGITIMATE):
                return last
            return self.reopen(actor=actor, reason=reason, evidence=evidence, ts=ts)

        return self.store.append_event(
            case_id=self.case_id,
            event_type="OPEN_CASE",
            actor=actor,
            from_state=CaseState.OPEN,
            to_state=CaseState.OPEN,
            evidence_refs=_normalize_evidence(evidence),
            reason=reason,
            ts=ts,
        )

    def transition(
        self,
        to_state: str,
        actor: str = "agent",
        reason: str = "",
        evidence: Optional[list] = None,
        event_type: str = "TRANSITION",
        required_role: Optional[str] = None,
        action: Optional[str] = None,
        exposure: Optional[float] = None,
        ts: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> Event:
        curr = self.current_state

        if to_state not in VALID_TRANSITIONS.get(curr, set()):
            raise InvalidTransitionError(
                f"Illegal state transition for case {self.case_id}: cannot transition from '{curr}' to '{to_state}'"
            )

        effective_req = get_required_role(action, exposure) or required_role

        if curr == CaseState.AWAITING_APPROVAL:
            pending_req = None
            for ev in reversed(self.events):
                if ev.event_type == "REQUEST_APPROVAL":
                    if "L2" in ev.reason:
                        pending_req = "L2"
                    elif "L1" in ev.reason:
                        pending_req = "L1"
                    break
                if ev.from_state != CaseState.AWAITING_APPROVAL and ev.to_state != CaseState.AWAITING_APPROVAL:
                    break
            if "L2" in (effective_req, pending_req):
                effective_req = "L2"
            elif not effective_req and pending_req:
                effective_req = pending_req

        if effective_req:
            actor_role = None
            if isinstance(details, dict) and details.get("role"):
                actor_role = str(details["role"])
            if actor.startswith("approver:"):
                suffix = actor.split(":", 1)[1]
                if suffix in ("L1", "L2") or not actor_role:
                    actor_role = suffix
            elif actor in ("L1", "L2") and not actor_role:
                actor_role = actor

            valid = False
            if actor_role == "L2":
                valid = True
            elif actor_role == "L1" and effective_req == "L1":
                valid = True

            if not valid:
                raise PolicyApprovalError(
                    f"Action requires approval role '{effective_req}', but actor is '{actor}' (role: {actor_role})"
                )

        return self.store.append_event(
            case_id=self.case_id,
            event_type=event_type,
            actor=actor,
            from_state=curr,
            to_state=to_state,
            evidence_refs=_normalize_evidence(evidence),
            reason=reason,
            ts=ts,
        )

    # Convenience transition methods
    def start_investigation(
        self, actor: str = "agent", reason: str = "Investigation started", evidence: Optional[list] = None, ts: Optional[str] = None
    ) -> Event:
        return self.transition(CaseState.INVESTIGATING, actor=actor, reason=reason, evidence=evidence, event_type="START_INVESTIGATION", ts=ts)

    def request_customer(
        self, actor: str = "agent", reason: str = "Verification requested from customer", evidence: Optional[list] = None, ts: Optional[str] = None
    ) -> Event:
        return self.transition(CaseState.AWAITING_CUSTOMER, actor=actor, reason=reason, evidence=evidence, event_type="REQUEST_CUSTOMER", ts=ts)

    def receive_customer_reply(
        self, actor: str = "agent", reply_text: str = "", evidence: Optional[list] = None, ts: Optional[str] = None, next_state: str = CaseState.INVESTIGATING
    ) -> Event:
        reason = f"Customer reply received: {reply_text}" if reply_text else "Customer reply received"
        return self.transition(next_state, actor=actor, reason=reason, evidence=evidence, event_type="RECEIVE_CUSTOMER_REPLY", ts=ts)

    def request_approval(
        self, required_role: str, action: str, actor: str = "agent", reason: str = "", evidence: Optional[list] = None, exposure: Optional[float] = None, ts: Optional[str] = None
    ) -> Event:
        policy_req = get_required_role(action, exposure) or required_role
        full_reason = f"Awaiting {policy_req} approval for action: {action}. {reason}".strip()
        return self.transition(CaseState.AWAITING_APPROVAL, actor=actor, reason=full_reason, evidence=evidence, event_type="REQUEST_APPROVAL", ts=ts)

    def record_approval(
        self, approver: str, role: str, action: str, reason: str = "Approved", evidence: Optional[list] = None, ts: Optional[str] = None, next_state: str = CaseState.INVESTIGATING, exposure: Optional[float] = None
    ) -> Event:
        app_clean = approver.strip()
        is_agent = app_clean.lower() in ("agent", "system", "") or app_clean.lower().startswith("agent:")
        actor = app_clean if app_clean.startswith("approver:") else (f"analyst:{app_clean}" if not app_clean.startswith("analyst:") else app_clean)
        policy_req = get_required_role(action, exposure)
        full_reason = f"Action '{action}' approved by {role}: {reason}"
        details = {} if is_agent else {"role": role, "approver": app_clean, "action": action}
        return self.transition(
            next_state,
            actor=actor,
            reason=full_reason,
            evidence=evidence,
            event_type="RECORD_APPROVAL",
            required_role=policy_req,
            action=action,
            exposure=exposure,
            ts=ts,
            details=details,
        )

    def escalate(
        self, actor: str = "agent", reason: str = "Escalated to human analyst", evidence: Optional[list] = None, ts: Optional[str] = None
    ) -> Event:
        return self.transition(CaseState.ESCALATED, actor=actor, reason=reason, evidence=evidence, event_type="ESCALATE", ts=ts)

    def close_fraud(
        self, actor: str = "agent", reason: str = "Confirmed fraud", evidence: Optional[list] = None, required_role: Optional[str] = None, action: Optional[str] = None, exposure: Optional[float] = None, ts: Optional[str] = None
    ) -> Event:
        return self.transition(CaseState.CLOSED_FRAUD, actor=actor, reason=reason, evidence=evidence, event_type="CLOSE_FRAUD", required_role=required_role, action=action, exposure=exposure, ts=ts)

    def close_legitimate(
        self, actor: str = "agent", reason: str = "Cleared as legitimate", evidence: Optional[list] = None, ts: Optional[str] = None
    ) -> Event:
        return self.transition(CaseState.CLOSED_LEGITIMATE, actor=actor, reason=reason, evidence=evidence, event_type="CLOSE_LEGITIMATE", ts=ts)

    def reopen(
        self, actor: str = "agent", reason: str = "Case reopened", evidence: Optional[list] = None, target_state: str = CaseState.INVESTIGATING, ts: Optional[str] = None
    ) -> Event:
        curr = self.current_state
        if curr not in (CaseState.CLOSED_FRAUD, CaseState.CLOSED_LEGITIMATE):
            raise InvalidTransitionError(f"Cannot reopen case {self.case_id} because it is in state '{curr}', not closed.")
        return self.store.append_event(
            case_id=self.case_id,
            event_type="REOPEN",
            actor=actor,
            from_state=curr,
            to_state=target_state,
            evidence_refs=_normalize_evidence(evidence),
            reason=reason,
            ts=ts or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # Derived views
    def open_questions(self, as_of: Optional[Any] = None) -> list[dict]:
        """Lists what the case is waiting on, since when, and SLA status per Policy R4 (24h)."""
        curr = self.current_state
        last = self.store.get_last_event(self.case_id)
        if not last or curr in (CaseState.CLOSED_FRAUD, CaseState.CLOSED_LEGITIMATE):
            return []

        ref_dt = _parse_ts(as_of) if as_of else datetime.now(timezone.utc)
        since_dt = _parse_ts(last.ts)
        elapsed_hours = (ref_dt - since_dt).total_seconds() / 3600.0
        sla_breached = elapsed_hours > 24.0

        if curr == CaseState.AWAITING_CUSTOMER:
            return [{
                "question": "Waiting on customer reply to transaction verification request.",
                "waiting_on": "customer",
                "since": last.ts,
                "elapsed_hours": round(elapsed_hours, 2),
                "sla_hours": 24.0,
                "sla_24h_breached": sla_breached,
            }]
        elif curr == CaseState.AWAITING_APPROVAL:
            return [{
                "question": f"Waiting on manager approval: {last.reason}",
                "waiting_on": "approver",
                "since": last.ts,
                "elapsed_hours": round(elapsed_hours, 2),
                "sla_hours": 24.0,
                "sla_24h_breached": sla_breached,
            }]
        elif curr == CaseState.ESCALATED:
            return [{
                "question": f"Escalated to human analyst: {last.reason}",
                "waiting_on": "analyst",
                "since": last.ts,
                "elapsed_hours": round(elapsed_hours, 2),
                "sla_hours": 24.0,
                "sla_24h_breached": sla_breached,
            }]
        elif curr in (CaseState.OPEN, CaseState.INVESTIGATING):
            return [{
                "question": "Active investigation: gathering evidence and running fraud detectors.",
                "waiting_on": "agent/analyst",
                "since": last.ts,
                "elapsed_hours": round(elapsed_hours, 2),
                "sla_hours": 24.0,
                "sla_24h_breached": False,
            }]
        return []

    @property
    def decisions(self) -> list[dict]:
        """Chronological audit trail of decisions made."""
        res = []
        for ev in self.events:
            res.append({
                "event_type": ev.event_type,
                "actor": ev.actor,
                "from_state": ev.from_state,
                "to_state": ev.to_state,
                "reason": ev.reason,
                "ts": ev.ts,
                "evidence_refs": ev.evidence_refs,
            })
        return res

    def handover_summary(self, as_of: Optional[Any] = None) -> dict:
        """Cold takeover summary detailing case status, decisions, open questions, and next step."""
        self.verify_integrity()
        curr = self.current_state
        last = self.store.get_last_event(self.case_id)
        qs = self.open_questions(as_of=as_of)

        next_step = ""
        if curr == CaseState.OPEN:
            next_step = "Initialize investigation and run detectors on customer transaction history."
        elif curr == CaseState.INVESTIGATING:
            next_step = "Review gathered evidence, calculate fraud exposure, and determine policy action."
        elif curr == CaseState.AWAITING_CUSTOMER:
            if qs and qs[0].get("sla_24h_breached"):
                next_step = "SLA BREACHED (>24h no reply). Per Policy R4: MONITOR_CARD, DECLINE_TRANSACTION for pending authorizations, and escalate if exposure > $500."
            else:
                next_step = "Awaiting customer reply to verification request (within 24h SLA window)."
        elif curr == CaseState.AWAITING_APPROVAL:
            next_step = "Review pending high-impact action and recorded evidence, then record L1/L2 approval or rejection."
        elif curr == CaseState.ESCALATED:
            next_step = "Assigned to human analyst for manual investigation and final verdict under Policy R8."
        elif curr == CaseState.CLOSED_FRAUD:
            next_step = "Case closed as confirmed fraud. No further action needed unless new evidence causes reopen."
        elif curr == CaseState.CLOSED_LEGITIMATE:
            next_step = "Case closed as legitimate. No further action needed unless new evidence causes reopen."

        all_evidence = []
        for ev in self.events:
            all_evidence.extend(ev.evidence_refs)

        return {
            "case_id": self.case_id,
            "current_state": curr,
            "last_updated": last.ts if last else None,
            "open_questions": qs,
            "decisions": self.decisions,
            "evidence_summary_count": len(all_evidence),
            "next_step": next_step,
            "tamper_check": "PASSED",
        }


def main() -> None:
    """CLI for assay.r2.casefile: python -m assay.r2.casefile show HHF-007 / handover HHF-007."""
    parser = argparse.ArgumentParser(description="Assay CaseFile CLI")
    parser.add_argument("command", choices=["show", "handover"], help="Command to execute")
    parser.add_argument("case_id", help="Case ID (e.g., HHF-007)")
    parser.add_argument("--db", default="cases.db", help="Path to SQLite database")

    args = parser.parse_args()

    store = CaseStore(args.db)
    cf = CaseFile(args.case_id, store=store)

    if args.command == "show":
        print(f"=== CASE {cf.case_id} ===")
        print(f"Current State: {cf.current_state}")
        print(f"Events Log ({len(cf.events)} events):")
        for ev in cf.events:
            print(f"  [{ev.ts}] {ev.event_type} | Actor: {ev.actor} | {ev.from_state} -> {ev.to_state}")
            print(f"    Reason: {ev.reason}")
            if ev.evidence_refs:
                print(f"    Evidence ({len(ev.evidence_refs)} refs): {ev.evidence_refs}")
    elif args.command == "handover":
        summary = cf.handover_summary()
        print(f"=== HANDOVER SUMMARY: {summary['case_id']} ===")
        print(f"Current State: {summary['current_state']}")
        print(f"Last Updated: {summary['last_updated']}")
        print(f"Next Step: {summary['next_step']}")
        if summary["open_questions"]:
            print("Open Questions:")
            for q in summary["open_questions"]:
                print(f"  - {q['question']} (Waiting on: {q['waiting_on']}, Since: {q['since']}, SLA Breached: {q['sla_24h_breached']})")
        print(f"Decisions Audit Trail ({len(summary['decisions'])} events):")
        for d in summary["decisions"]:
            print(f"  - [{d['ts']}] {d['actor']} ({d['event_type']}): {d['reason']}")


if __name__ == "__main__":
    main()
