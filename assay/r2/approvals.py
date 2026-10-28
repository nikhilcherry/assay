"""BLOCK B10 -- Server-side approvals & append-only audit ledger.

Enforces role-based access control (L1, L2, agent), policy approval routes,
tamper-evident audit logging (SHA256 hash chain), body tamper logging, constant-time
token validation, idempotency key replay handling, wrong-role rate limiting (5/min),
and fail-closed approver configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, status

logger = logging.getLogger("assay.approvals")

AUTO_ACTIONS = {
    "ALLOW_TRANSACTION",
    "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH",
    "GENERATE_REPORT",
    "CREATE_CASE",
    "ESCALATE_TO_ANALYST",
    "CLOSE_NO_FRAUD",
}

FORBIDDEN_BODY_FIELDS = {"role", "user", "username", "user_id", "token"}


def route_for(action: str, exposure_usd: float) -> str:
    """Computes approval route server-side per Fraud Operations Policy Section 2."""
    if action in AUTO_ACTIONS:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if float(exposure_usd) <= 2500.0 else "L2"
    if action in ("BLOCK_ALL_CARDS", "FILE_REPORT"):
        return "L2"
    raise ValueError(f"Unknown action: {action}")


class Ledger:
    """Append-only SHA256 hash-chained JSONL audit ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _get_last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last_line = ""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if not last_line:
            return "0" * 64
        try:
            entry = json.loads(last_line)
            return entry.get("hash", "0" * 64)
        except Exception:
            return "0" * 64

    def append(self, entry_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        prev_hash = self._get_last_hash()
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "entry_type": entry_type,
            "timestamp": ts,
            "data": data,
            "prev_hash": prev_hash,
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        entry_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        entry = dict(payload)
        entry["hash"] = entry_hash

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

        return entry

    def verify(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"valid": True, "entries": 0, "error": None}

        lines = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line.strip())

        prev_hash = "0" * 64
        for idx, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except Exception as e:
                return {"valid": False, "entries": idx, "error": f"Invalid JSON line {idx}: {e}"}

            stored_hash = entry.get("hash")
            stored_prev = entry.get("prev_hash")

            if stored_prev != prev_hash:
                return {
                    "valid": False,
                    "entries": idx,
                    "error": f"prev_hash mismatch line {idx}: expected {prev_hash}, got {stored_prev}",
                }

            payload = {
                "entry_type": entry.get("entry_type"),
                "timestamp": entry.get("timestamp"),
                "data": entry.get("data"),
                "prev_hash": stored_prev,
            }
            canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            calc_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

            if calc_hash != stored_hash:
                return {
                    "valid": False,
                    "entries": idx,
                    "error": f"Hash mismatch line {idx}: computed {calc_hash}, stored {stored_hash}",
                }

            prev_hash = stored_hash

        return {"valid": True, "entries": len(lines), "error": None}


def load_approvers(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Loads approvers configuration once at startup. Fails closed if file is missing."""
    if path is None:
        env_path = os.environ.get("ASSAY_APPROVERS")
        if env_path:
            path = Path(env_path)
    if path is None:
        raise FileNotFoundError(
            "Approvers file path not specified and ASSAY_APPROVERS env var is not set"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Approvers file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Approvers file must contain a JSON list")
        return data


def create_app(
    ledger_path: str | Path | None = None, approvers_path: str | Path | None = None
) -> FastAPI:
    if ledger_path is None:
        ledger_path = os.environ.get("ASSAY_LEDGER", "ledger.jsonl")

    # Load approvers at startup; fails closed if missing
    approvers = load_approvers(approvers_path)

    app = FastAPI(title="Assay Approvals API")

    ledger = Ledger(ledger_path)
    proposals: Dict[str, Dict[str, Any]] = {}

    if ledger.path.exists():
        with open(ledger.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    e_type = entry.get("entry_type")
                    data = entry.get("data", {})
                    if e_type == "proposal_created":
                        proposals[data["id"]] = dict(data)
                    elif e_type == "proposal_decided":
                        pid = data.get("proposal_id")
                        if pid in proposals:
                            proposals[pid]["status"] = data.get("status")
                            proposals[pid]["decision"] = data.get("decision")
                            proposals[pid]["decided_by"] = data.get("decided_by")
                            proposals[pid]["note"] = data.get("note")
                            proposals[pid]["decided_at"] = entry.get("timestamp")
                except Exception:
                    pass

    app.state.ledger = ledger
    app.state.approvers = approvers
    app.state.proposals = proposals
    app.state.idempotency_store = {}
    app.state.wrong_role_attempts = {}

    def get_current_user(authorization: Optional[str]) -> Dict[str, Any]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )
        token = authorization.split("Bearer ", 1)[1].strip()
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        for user in app.state.approvers:
            user_hash = user.get("token_sha256", "")
            if hmac.compare_digest(token_hash, user_hash):
                return user

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    @app.post("/proposals", status_code=status.HTTP_201_CREATED)
    async def create_proposal(request: Request, authorization: Optional[str] = Header(None)):
        user = get_current_user(authorization)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        found_forbidden = FORBIDDEN_BODY_FIELDS.intersection(body.keys())
        if found_forbidden:
            logger.warning(
                f"Tamper attempt detected: forbidden auth fields in proposal body: {sorted(found_forbidden)}"
            )

        case_id = body.get("case_id")
        action = body.get("action")
        exposure_usd = body.get("exposure_usd")
        evidence_ref = body.get("evidence_ref", "")

        if not case_id or not action or exposure_usd is None:
            raise HTTPException(
                status_code=400, detail="Missing required fields: case_id, action, exposure_usd"
            )

        try:
            route = route_for(action, float(exposure_usd))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        if route == "auto":
            proposal = {
                "id": proposal_id,
                "case_id": case_id,
                "action": action,
                "exposure_usd": float(exposure_usd),
                "evidence_ref": evidence_ref,
                "route": route,
                "status": "auto_executed",
                "created_by": user["name"],
                "created_at": now,
                "decided_by": "system",
                "decision": "approve",
                "note": "Auto-executed server-side per policy route auto",
                "decided_at": now,
            }
        else:
            proposal = {
                "id": proposal_id,
                "case_id": case_id,
                "action": action,
                "exposure_usd": float(exposure_usd),
                "evidence_ref": evidence_ref,
                "route": route,
                "status": "pending",
                "created_by": user["name"],
                "created_at": now,
                "decided_by": None,
                "decision": None,
                "note": None,
                "decided_at": None,
            }

        app.state.proposals[proposal_id] = proposal
        app.state.ledger.append("proposal_created", proposal)

        return proposal

    @app.post("/proposals/{proposal_id}/decision")
    async def make_decision(
        proposal_id: str,
        request: Request,
        authorization: Optional[str] = Header(None),
        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    ):
        user = get_current_user(authorization)

        if not idempotency_key:
            raise HTTPException(status_code=400, detail="Missing Idempotency-Key header")

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        # Replay / Idempotency check
        if idempotency_key in app.state.idempotency_store:
            stored_body, stored_response = app.state.idempotency_store[idempotency_key]
            if stored_body == body:
                return stored_response
            else:
                raise HTTPException(
                    status_code=409, detail="Idempotency key reused with different request body"
                )

        found_forbidden = FORBIDDEN_BODY_FIELDS.intersection(body.keys())
        if found_forbidden:
            logger.warning(
                f"Tamper attempt detected: forbidden auth fields in decision body: {sorted(found_forbidden)}"
            )

        if proposal_id not in app.state.proposals:
            raise HTTPException(status_code=404, detail="Proposal not found")

        proposal = app.state.proposals[proposal_id]

        if proposal["status"] != "pending":
            raise HTTPException(status_code=409, detail="Proposal already decided")

        # Rate limiting check per token for wrong-role attempts (5 per minute)
        token_hash = user.get("token_sha256", "")
        now_t = time.time()
        recent_attempts = [
            t for t in app.state.wrong_role_attempts.get(token_hash, []) if now_t - t < 60
        ]
        app.state.wrong_role_attempts[token_hash] = recent_attempts

        if len(recent_attempts) >= 5:
            logger.warning(
                f"SECURITY EVENT: Rate limit enforced for authorization failures by user {user.get('name')}"
            )
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded for authorization failures"
            )

        decider_role = user.get("role")
        decider_name = user.get("name")

        role_error = None
        if decider_role == "agent":
            role_error = "Agent identity cannot approve decisions"
        elif decider_name == proposal["created_by"]:
            role_error = "Self-approval is forbidden"
        else:
            route = proposal["route"]
            if route == "L2" and decider_role != "L2":
                role_error = "L1 role cannot approve L2 route proposals"
            elif route == "L1" and decider_role not in ("L1", "L2"):
                role_error = "Role not authorized for L1 route proposals"

        if role_error:
            recent_attempts.append(now_t)
            logger.warning(
                f"SECURITY EVENT: Unauthorized approval attempt by {decider_name} (role={decider_role}) for proposal {proposal_id} (route={proposal.get('route')}): {role_error}"
            )
            raise HTTPException(status_code=403, detail=role_error)

        decision = body.get("decision")
        if decision not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="Decision must be 'approve' or 'reject'")

        note = body.get("note", "")
        now_iso = datetime.now(timezone.utc).isoformat()

        proposal["status"] = "approved" if decision == "approve" else "rejected"
        proposal["decision"] = decision
        proposal["decided_by"] = decider_name
        proposal["note"] = note
        proposal["decided_at"] = now_iso

        decision_data = {
            "proposal_id": proposal_id,
            "decision": decision,
            "decided_by": decider_name,
            "decided_by_role": decider_role,
            "note": note,
            "status": proposal["status"],
        }
        app.state.ledger.append("proposal_decided", decision_data)

        app.state.idempotency_store[idempotency_key] = (body, proposal)
        return proposal

    @app.get("/proposals")
    async def list_proposals(case_id: Optional[str] = None):
        items = list(app.state.proposals.values())
        if case_id is not None:
            items = [p for p in items if p.get("case_id") == case_id]
        return items

    @app.get("/proposals/{proposal_id}")
    async def get_proposal(proposal_id: str):
        if proposal_id not in app.state.proposals:
            raise HTTPException(status_code=404, detail="Proposal not found")
        return app.state.proposals[proposal_id]

    @app.get("/ledger/verify")
    async def verify_ledger():
        return app.state.ledger.verify()

    return app


def cli():
    parser = argparse.ArgumentParser(
        prog="assay.r2.approvals", description="Assay Approvals CLI & Server"
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI approvals server")
    serve_parser.add_argument("--port", type=int, default=8741, help="Port to bind (default: 8741)")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")

    token_parser = subparsers.add_parser("make-token", help="Generate approver token")
    token_parser.add_argument("--name", required=True, help="Approver name")
    token_parser.add_argument(
        "--role", required=True, choices=["L1", "L2", "agent"], help="Approver role"
    )
    token_parser.add_argument("--approvers", help="Path to approvers.json")

    args = parser.parse_args()

    if args.command == "serve":
        try:
            import uvicorn

            uvicorn.run(
                "assay.r2.approvals:create_app", factory=True, host=args.host, port=args.port
            )
        except ImportError:
            print(
                "uvicorn is not installed. Run with an ASGI server or install uvicorn.",
                file=sys.stderr,
            )
    elif args.command == "make-token":
        approvers_file = args.approvers or os.environ.get("ASSAY_APPROVERS", "approvers.json")
        p = Path(approvers_file)
        approvers_data = []
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                try:
                    approvers_data = json.load(f)
                except Exception:
                    approvers_data = []
        raw_token = secrets.token_urlsafe(32)
        token_sha256 = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        approvers_data.append({"name": args.name, "role": args.role, "token_sha256": token_sha256})
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(approvers_data, f, indent=2)
        print(f"Approver token generated for name='{args.name}', role='{args.role}'")
        print(f"TOKEN: {raw_token}")
        print("Save this token now. Only its SHA256 digest has been saved to disk.")
        print(f"Saved token_sha256 to: {p.resolve()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
