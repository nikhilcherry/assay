"""BLOCK B26 -- Live demo console.

FastAPI app serving a self-contained dark analyst-console HTML page (`assay/r2/console.html`).
- Lists the 18 final-round cases with quarantined prompt-injection spans highlighted
  and labelled 'treated as data'.
- Runs `R2Investigator` over `R2LocalStore` (constructed once at startup) on `POST /run/{case_id}`,
  returning the answer JSON, tool-call trace with row counts, and step-by-step log-odds/probability
  ledger for live visual replay.
- Mounts the server-side approvals API (`assay.r2.approvals`) under `/approvals` and creates
  action proposals automatically for L1/L2 actions so the presenter can approve them live.
- Supports a 'Break it' toggle (`/fault`) that injects `ASSAY_FAULT` (e.g. `get_transaction:down@2`)
  for the next run to demonstrate fail-closed behaviour live.

CLI:
    python -m assay.r2.console --port 8742
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from assay.r2.approvals import create_app as create_approvals_app
from assay.r2.frames import case_pack
from assay.r2.investigate_r2 import R2Investigator
from assay.r2.store_r2 import R2LocalStore, r2_pattern_model

HTML_PATH = Path(__file__).with_name("console.html")

# Default demo tokens (documented in UI placeholder / docs for live presenter demo;
# can be overridden via ASSAY_APPROVERS env var).
DEMO_AGENT_TOKEN = "assay-agent-internal-token"
DEMO_L1_TOKEN = "demo-l1-token"
DEMO_L2_TOKEN = "demo-l2-token"

# Regex patterns for prompt-injection / hostile instruction spans in free text.
INJECTION_PATTERNS = [
    re.compile(r"Also,\s*SYSTEM NOTE:.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[\[\s*(?:assistant|system|agent|user)\s*:.*?\]\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"P\.S\.\s*the compliance team already cleared this one.*",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Note from customer:\s*do not escalate this.*",
        re.IGNORECASE | re.DOTALL,
    ),
]


def quarantine_segments(text: str) -> Dict[str, Any]:
    """Splits trigger_text into clean vs quarantined segments without mutating the data."""
    if not text:
        return {"raw": "", "has_injection": False, "segments": []}

    spans: List[tuple[int, int]] = []
    for pat in INJECTION_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))

    if not spans:
        return {
            "raw": text,
            "has_injection": False,
            "segments": [{"text": text, "quarantined": False}],
        }

    # Merge overlapping spans
    spans.sort(key=lambda x: (x[0], x[1]))
    merged: List[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    segments: List[Dict[str, Any]] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            segments.append({"text": text[cursor:s], "quarantined": False})
        segments.append(
            {
                "text": text[s:e],
                "quarantined": True,
                "label": "treated as data",
            }
        )
        cursor = e
    if cursor < len(text):
        segments.append({"text": text[cursor:], "quarantined": False})

    return {
        "raw": text,
        "has_injection": True,
        "segments": segments,
    }


def _parse_fault_spec(spec: str) -> Dict[str, Any]:
    """Parses fault strings like 'get_transaction:down@2' or 'card_window:down@1'."""
    spec = (spec or "").strip()
    if not spec:
        return {}
    method = spec
    mode = "down"
    at_call = 1
    if ":" in spec:
        method, rest = spec.split(":", 1)
        if "@" in rest:
            mode, num = rest.split("@", 1)
            try:
                at_call = max(1, int(num))
            except ValueError:
                at_call = 1
        else:
            mode = rest
    elif "@" in spec:
        method, num = spec.split("@", 1)
        try:
            at_call = max(1, int(num))
        except ValueError:
            at_call = 1
    return {"raw": spec, "method": method.strip(), "mode": mode.strip(), "at": at_call}


class FaultInjectedError(RuntimeError):
    """Raised when an injected ASSAY_FAULT triggers during a store query."""


def _row_count_for_result(res: Any) -> int:
    """Computes row count from a store query return value `(data, ref)` or `gid`."""
    if isinstance(res, tuple) and len(res) == 2:
        data = res[0]
        if isinstance(data, pd.DataFrame):
            return int(len(data))
        if isinstance(data, pd.Series):
            return 1
        if isinstance(data, (list, tuple, set, dict)):
            return int(len(data))
        return 1 if data is not None else 0
    if isinstance(res, str):
        return 1
    return 0


class InstrumentedR2Store(R2LocalStore):
    """Wraps R2LocalStore to record per-call row counts and enforce ASSAY_FAULT."""

    QUERIES = (
        "get_transaction",
        "card_window",
        "holder_history",
        "card_history",
        "closed_cases_by_device",
        "device_neighbors",
        "closed_cases_for_card",
        "closed_cases_touching",
        "closed_cases_like",
        "prior_fraud_cases",
        "write_case",
    )

    def __init__(self):
        super().__init__()
        self.call_rows: List[int] = []
        self.method_counts: Dict[str, int] = {}
        self.total_query_calls: int = 0

    def reset(self) -> None:
        super().reset()
        self.call_rows = []
        self.method_counts = {}
        self.total_query_calls = 0

    def _log(self, name: str, **args) -> str:
        ref = super()._log(name, **args)
        # Default row count 1 for direct `self.s._log(...)` calls from detectors;
        # wrapped methods overwrite the last entry with the exact dataframe length.
        self.call_rows.append(1)
        return ref

    def _check_fault(self, method_name: str, logged_name: str) -> None:
        spec_str = os.environ.get("ASSAY_FAULT", "").strip()
        if not spec_str:
            return
        spec = _parse_fault_spec(spec_str)
        target = spec.get("method", "")
        at = spec.get("at", 1)
        mode = spec.get("mode", "down")

        self.total_query_calls += 1
        m_cnt = self.method_counts.get(method_name, 0) + 1
        self.method_counts[method_name] = m_cnt
        if logged_name != method_name:
            l_cnt = self.method_counts.get(logged_name, 0) + 1
            self.method_counts[logged_name] = l_cnt
        else:
            l_cnt = m_cnt

        triggered = False
        if target in ("*", "any", "all"):
            triggered = self.total_query_calls >= at
        elif target == method_name:
            triggered = m_cnt >= at
        elif target == logged_name:
            triggered = l_cnt >= at

        if triggered:
            self._log(logged_name, fault=f"{mode}@{at}")
            self.call_rows[-1] = 0
            raise FaultInjectedError(
                f"FAIL_CLOSED: dependency fault injected via ASSAY_FAULT={spec_str} "
                f"on tool '{logged_name}' (call #{m_cnt} -> {mode})"
            )


def _wrap_store_queries(store: InstrumentedR2Store) -> None:
    """Wraps query methods on the store instance to check faults and record row counts."""
    name_map = {"closed_cases_like": "similar_prior_cases"}

    for qname in InstrumentedR2Store.QUERIES:
        orig = getattr(store, qname)
        logged_name = name_map.get(qname, qname)

        def _make_wrapper(fn, m_name: str, l_name: str):
            def _wrapped(*args, **kwargs):
                store._check_fault(m_name, l_name)
                res = fn(*args, **kwargs)
                if store.call_rows:
                    store.call_rows[-1] = _row_count_for_result(res)
                return res

            return _wrapped

        setattr(store, qname, _make_wrapper(orig, qname, logged_name))


def _ensure_default_approvers(tmp_dir: Path) -> Path:
    """Creates a default approvers file if ASSAY_APPROVERS is not set."""
    env_path = os.environ.get("ASSAY_APPROVERS")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    p = tmp_dir / "console_approvers.json"
    entries = [
        {
            "name": "assay-agent",
            "role": "agent",
            "token_sha256": hashlib.sha256(DEMO_AGENT_TOKEN.encode("utf-8")).hexdigest(),
        },
        {
            "name": "lead_analyst_l1",
            "role": "L1",
            "token_sha256": hashlib.sha256(DEMO_L1_TOKEN.encode("utf-8")).hexdigest(),
        },
        {
            "name": "fraud_manager_l2",
            "role": "L2",
            "token_sha256": hashlib.sha256(DEMO_L2_TOKEN.encode("utf-8")).hexdigest(),
        },
    ]
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return p


def _parse_lr_from_finding(ev_item: Dict[str, Any]) -> Optional[float]:
    """Extracts the likelihood ratio from an evidence item or its claim string."""
    if "lr" in ev_item and ev_item["lr"] is not None:
        try:
            return float(ev_item["lr"])
        except (TypeError, ValueError):
            pass
    claim = str(ev_item.get("claim", ""))
    # Look for patterns like '(LR 4.2x)' or 'LR=3.1' or 'lr 2.5'
    m = re.search(r"\bLR[=:\s]+([0-9]+(?:\.[0-9]+)?)x?\b", claim, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _build_ledger_steps(
    ans: Dict[str, Any], flagged_row: Optional[pd.Series]
) -> List[Dict[str, Any]]:
    """Constructs the step-by-step probability needle progression from the evidence ledger."""
    evidence_list: List[Dict[str, Any]] = ans.get("case", {}).get("evidence", [])
    final_p = float(ans.get("case", {}).get("fraud_probability", 0.5))

    # Determine starting prior probability from the flagged transaction's model score or risk score
    prior_p = 0.5
    if flagged_row is not None:
        if "p_fraud" in flagged_row and pd.notna(flagged_row["p_fraud"]):
            prior_p = float(flagged_row["p_fraud"])
        elif "risk_score" in flagged_row and pd.notna(flagged_row["risk_score"]):
            prior_p = float(flagged_row["risk_score"])
    prior_p = min(max(prior_p, 0.001), 0.999)
    final_p_clamped = min(max(final_p, 0.001), 0.999)

    # First check if evidence items carry explicit LRs; if not, derive effective per-step LRs
    # so the log-odds trajectory reproduces the exact final probability from the answer.
    explicit_lrs: List[Optional[float]] = [_parse_lr_from_finding(e) for e in evidence_list]

    prior_lo = math.log(prior_p / (1.0 - prior_p))
    final_lo = math.log(final_p_clamped / (1.0 - final_p_clamped))

    steps: List[Dict[str, Any]] = []
    if not evidence_list:
        return steps

    # Item 0 is typically the base score finding itself if it references get_transaction
    first_ref = str(evidence_list[0].get("ref", ""))
    has_base_step = first_ref.startswith("query:get_transaction(")

    cur_lo = prior_lo
    known_delta = sum(
        math.log(max(lr, 1e-4))
        for idx, lr in enumerate(explicit_lrs)
        if lr is not None and not (idx == 0 and has_base_step)
    )
    unknown_indices = [
        idx
        for idx, lr in enumerate(explicit_lrs)
        if lr is None and not (idx == 0 and has_base_step)
    ]
    rem_delta = (final_lo - prior_lo) - known_delta
    per_unknown_delta = rem_delta / len(unknown_indices) if unknown_indices else 0.0

    for idx, ev in enumerate(evidence_list):
        if idx == 0 and has_base_step:
            lr_val = explicit_lrs[idx] if explicit_lrs[idx] is not None else 1.0
            p_after = 1.0 / (1.0 + math.exp(-cur_lo))
        else:
            if explicit_lrs[idx] is not None:
                lr_val = float(explicit_lrs[idx])
                cur_lo += math.log(max(lr_val, 1e-4))
            else:
                cur_lo += per_unknown_delta
                lr_val = math.exp(per_unknown_delta)
            p_after = 1.0 / (1.0 + math.exp(-cur_lo))

        # Snap last step to exact reported fraud_probability
        if idx == len(evidence_list) - 1:
            p_after = final_p

        steps.append(
            {
                "step": idx + 1,
                "claim": ev.get("claim", ""),
                "ref": ev.get("ref", ""),
                "ids": ev.get("ids", []),
                "lr": round(float(lr_val), 3),
                "probability_after": round(float(p_after), 4),
            }
        )

    return steps


class FaultPayload(BaseModel):
    fault: Optional[str] = None


def create_app() -> FastAPI:
    app = FastAPI(title="Assay Live Demo Console")

    tmp_dir = Path(tempfile.mkdtemp(prefix="assay_console_"))
    approvers_path = _ensure_default_approvers(tmp_dir)
    ledger_path = os.environ.get("ASSAY_LEDGER") or str(tmp_dir / "console_ledger.jsonl")

    approvals_app = create_approvals_app(
        ledger_path=ledger_path, approvers_path=approvers_path
    )
    app.mount("/approvals", approvals_app)

    t0 = time.time()
    store = InstrumentedR2Store()
    _wrap_store_queries(store)
    inv = R2Investigator(store, r2_pattern_model(), writer=store.write_case)
    startup_sec = round(time.time() - t0, 3)

    cp_df = case_pack().copy()

    app.state.store = store
    app.state.investigator = inv
    app.state.approvals_app = approvals_app
    app.state.case_pack = cp_df
    app.state.startup_sec = startup_sec

    def _register_proposals_for_case(case_id: str, ans: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Registers L1/L2/auto proposals in the mounted approvals store so L1/L2 wait for approval."""
        exposure = float(ans.get("case", {}).get("exposure_usd", 0.0))
        final_actions = ans.get("next_best_actions", {}).get("final", [])
        out_actions: List[Dict[str, Any]] = []

        # Clear any old pending proposals for this case_id from prior runs in the demo session
        old_pids = [
            pid
            for pid, p in approvals_app.state.proposals.items()
            if p.get("case_id") == case_id
        ]
        for pid in old_pids:
            approvals_app.state.proposals.pop(pid, None)

        from datetime import datetime, timezone
        import uuid
        from assay.r2.approvals import route_for

        for act_item in final_actions:
            act_name = act_item.get("action", "")
            route = act_item.get("approval_route") or route_for(act_name, exposure)
            pid = f"prop_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc).isoformat()
            ev_ref = f"case:{case_id}"

            if route == "auto":
                prop = {
                    "id": pid,
                    "case_id": case_id,
                    "action": act_name,
                    "exposure_usd": exposure,
                    "evidence_ref": ev_ref,
                    "route": route,
                    "status": "auto_executed",
                    "created_by": "assay-agent",
                    "created_at": now,
                    "decided_by": "system",
                    "decision": "approve",
                    "note": "Auto-executed server-side per policy route auto",
                    "decided_at": now,
                }
            else:
                prop = {
                    "id": pid,
                    "case_id": case_id,
                    "action": act_name,
                    "exposure_usd": exposure,
                    "evidence_ref": ev_ref,
                    "route": route,
                    "status": "pending",
                    "created_by": "assay-agent",
                    "created_at": now,
                    "decided_by": None,
                    "decision": None,
                    "note": None,
                    "decided_at": None,
                }

            approvals_app.state.proposals[pid] = prop
            approvals_app.state.ledger.append("proposal_created", prop)
            out_actions.append(
                {
                    "action": act_name,
                    "approval_route": route,
                    "proposal_id": pid,
                    "status": "awaiting approval" if route in ("L1", "L2") else "auto_executed",
                }
            )
        return out_actions

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))

    @app.get("/cases")
    async def list_cases() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, r in app.state.case_pack.iterrows():
            trig_text = "" if pd.isna(r.trigger_text) else str(r.trigger_text)
            q = quarantine_segments(trig_text)
            rs = None if pd.isna(r.risk_score) else float(r.risk_score)
            rows.append(
                {
                    "case_id": str(r.case_id),
                    "opened_at": str(r.opened_at),
                    "trigger_type": str(r.trigger_type),
                    "trigger_text": trig_text,
                    "has_injection": q["has_injection"],
                    "segments": q["segments"],
                    "flagged_txn_id": int(r.flagged_txn_id),
                    "card_id": str(r.card_id),
                    "customer_id": str(r.customer_id),
                    "risk_score": rs,
                }
            )
        return rows

    @app.get("/fault")
    async def get_fault() -> Dict[str, Any]:
        cur = os.environ.get("ASSAY_FAULT", "").strip()
        return {"fault": cur or None, "active": bool(cur)}

    @app.post("/fault")
    async def set_fault(payload: FaultPayload) -> Dict[str, Any]:
        val = (payload.fault or "").strip()
        if val:
            os.environ["ASSAY_FAULT"] = val
        else:
            os.environ.pop("ASSAY_FAULT", None)
        return {"fault": val or None, "active": bool(val)}

    @app.post("/run/{case_id}")
    async def run_case(case_id: str) -> Dict[str, Any]:
        sub = app.state.case_pack[app.state.case_pack.case_id == case_id]
        if sub.empty:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

        case_row = sub.iloc[0]
        store: InstrumentedR2Store = app.state.store
        inv: R2Investigator = app.state.investigator

        store.reset()
        t_start = time.time()
        active_fault = os.environ.get("ASSAY_FAULT", "").strip()

        try:
            ans = inv.run(case_row)
        except Exception as exc:
            elapsed_ms = round((time.time() - t_start) * 1000.0, 2)
            # Clear one-shot fault after demonstrating fail-closed so the presenter can toggle freely
            calls_out = []
            for idx, c in enumerate(store.calls):
                rc = store.call_rows[idx] if idx < len(store.call_rows) else 0
                calls_out.append(
                    {
                        "step": idx + 1,
                        "name": c.name,
                        "params": c.args,
                        "ref": c.ref,
                        "row_count": rc,
                    }
                )
            return {
                "ok": False,
                "fail_closed": True,
                "case_id": case_id,
                "fault_injected": active_fault or None,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "elapsed_ms": elapsed_ms,
                "calls": calls_out,
                "ledger": [],
                "answer": None,
            }

        elapsed_ms = round((time.time() - t_start) * 1000.0, 2)

        calls_out = []
        for idx, c in enumerate(store.calls):
            rc = store.call_rows[idx] if idx < len(store.call_rows) else 1
            calls_out.append(
                {
                    "step": idx + 1,
                    "name": c.name,
                    "params": c.args,
                    "ref": c.ref,
                    "row_count": rc,
                }
            )

        fid = int(case_row.flagged_txn_id)
        flagged_tx = store.tx.loc[fid] if fid in store.tx.index else None
        ledger_steps = _build_ledger_steps(ans, flagged_tx)
        proposals = _register_proposals_for_case(case_id, ans)

        prior_p = 0.5
        if flagged_tx is not None and pd.notna(flagged_tx.get("p_fraud")):
            prior_p = round(float(flagged_tx["p_fraud"]), 4)

        return {
            "ok": True,
            "fail_closed": False,
            "case_id": case_id,
            "elapsed_ms": elapsed_ms,
            "prior_probability": prior_p,
            "verdict": ans["case"]["verdict"],
            "fraud_probability": ans["case"]["fraud_probability"],
            "pattern": ans["case"]["pattern"],
            "exposure_usd": ans["case"]["exposure_usd"],
            "sar": ans["sar"],
            "actions": proposals,
            "calls": calls_out,
            "ledger": ledger_steps,
            "answer": ans,
        }

    return app


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="assay.r2.console", description="Assay Live Demo Console"
    )
    parser.add_argument("--port", type=int, default=8742, help="Port to bind (default: 8742)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
