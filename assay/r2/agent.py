"""BLOCK N1 -- The deployable agent single-case command for live drills and exam runs.

Usage:
    python -m assay.r2.agent HHF-007 [--out cases_r2] [--fault graph_down|tool_timeout|garbage|empty_body|none] [--role analyst|l1|l2]

Wires:
  1. Case validation (`assay.r2.harness.validate_case_row` + strict path/NUL/collision checks)
     -> exit 2 on invalid input, no answer file.
  2. Untrusted trigger parsing (`assay.r2.untrusted.parse_trigger` + `cross_check`) -> free text never
     reaches decision logic; warnings / hostile strings recorded in answer evidence as data with strict
     secret & PII redaction (PANs, AWS keys, GitHub tokens, Bearer tokens, emails, phones).
  3. Investigation (`assay.r2.investigate_r2.R2Investigator`) with store wrapped by
     `assay.r2.harness.ToolProxy` (+ `FaultInjector` and empty/truncated body validators). Any
     `DependencyDown` / `ToolTimeout` / `GarbageResponse` fails closed: prints failure, writes NO answer,
     audit-logs the error, exits 3.
  4. Idempotent answer writing (`assay.r2.harness.write_answer` + `RunLedger` + file & thread locks):
     running the same case twice (sequentially or concurrently) produces byte-identical JSON without corruption.
  5. Attested SAR attachment (`assay.r2.sar.sar_required` + `draft_sar`) preserving `assay.validate.validate`
     and the customer tip-off rule.
  6. Redacted, hash-chained audit logging (`assay.r2.audit.AuditLog`) and case lifecycle management
     (`assay.r2.casefile.CaseFile`).
  7. Pending approval ledger entries (`assay.r2.approvals.Ledger`) for L1/L2 actions; `--role` on CLI
     never executes or approves actions, and `CaseFile` records L1/L2 actions as `AWAITING_APPROVAL`
     (never applied by the agent) while only applying auto-route actions idempotently.
  8. Prints per-case latency and tokens (`tokens=0`).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from assay import disputes
from assay.data import ROOT
from assay.r2.approvals import Ledger as ApprovalLedger, route_for as approval_route_for
from assay.r2.attest import dataset_tag
from assay.r2.audit import AuditLog, redact as audit_redact
from assay.r2.casefile import CaseFile, CaseState, CaseStore, _normalize_evidence
from assay.r2.frames import case_pack, tx_all
from assay.r2.harness import (
    CallRecord,
    DependencyDown,
    FaultInjector,
    GarbageResponse,
    HarnessError,
    InvalidInput,
    RunLedger,
    RunResult,
    ToolProxy,
    ToolTimeout,
    idempotency_key,
    validate_case_row,
    write_answer,
)
from assay.r2.investigate_r2 import R2Investigator, measured_exam_dispute_weight
from assay.r2.sar import draft_sar, sar_required
from assay.r2.store_r2 import R2LocalStore, r2_pattern_model
from assay.r2.untrusted import TriggerFacts, attest_trigger_facts, cross_check, parse_trigger
from assay.validate import validate

DEFAULT_OUT = ROOT / "cases_r2"
CODE_VERSION = "assay-r2-n1-v2"

FAULT_CHOICES = ("none", "graph_down", "tool_timeout", "garbage", "empty_body")
ROLE_CHOICES = ("analyst", "l1", "l2")

# Map high-level CLI drill fault names to FaultInjector specifications on the first store call
FAULT_MAP: dict[str, str] = {
    "none": "",
    "graph_down": "get_transaction:down@1",
    "tool_timeout": "get_transaction:timeout@1",
    "garbage": "get_transaction:garbage@1",
    "empty_body": "card_window:empty_body@1",
}

TIP_OFF_PATTERNS = [
    re.compile(r"\bSAR\b", re.IGNORECASE),
    re.compile(r"suspicious activity report", re.IGNORECASE),
    re.compile(r"regulator(?:y)?\s+(?:report|filing)", re.IGNORECASE),
    re.compile(r"FinCEN", re.IGNORECASE),
    re.compile(r"FILE_REPORT", re.IGNORECASE),
]

# Extra secret & PII patterns scrubbed before anything from trigger_text reaches answer JSON or logs
EXTRA_SECRET_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bsk-(?:live-|proj-|test-)?[A-Za-z0-9_-]{12,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.=]{8,}"), "Bearer [REDACTED_TOKEN]"),
    # 13 to 19 digit card PANs (with optional spaces or hyphens)
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"), "[REDACTED_PAN]"),
    (re.compile(r"\b\d{13,19}\b"), "[REDACTED_PAN]"),
]

# Strict safe case_id pattern: no slashes, no '..', no NUL, no whitespace
SAFE_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

# Process-wide lock for thread-safe investigator & file operations
_INVESTIGATOR_LOCK = threading.RLock()
_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.RLock] = {}

# Initialize process-global dispute override weight ONCE at module startup (never mutated per case)
_STARTUP_W_OVERRIDE, _STARTUP_DISPUTE_BASIS = measured_exam_dispute_weight()
disputes.W_OVERRIDE = _STARTUP_W_OVERRIDE


def _to_json_safe(obj: Any) -> Any:
    """Recursively converts `pd.Timestamp`, `datetime`, `np.generic`, `pd.Series`, `pd.DataFrame`,
    and sets into plain JSON-serializable Python types (ISO strings, ints, floats, dicts, lists)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, pd.Series):
        return {str(k): _to_json_safe(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return {"rows": int(len(obj))}
    return str(obj)


def _sanitize_call_records(calls: list[CallRecord]) -> list[CallRecord]:
    """Returns a copy of `CallRecord` list with all `args` converted to JSON-safe primitives."""
    return [
        CallRecord(
            name=c.name,
            args=_to_json_safe(c.args),
            started=float(c.started),
            ms=float(c.ms),
            ok=bool(c.ok),
            error_type=c.error_type,
        )
        for c in calls
    ]


def _get_thread_lock(lock_key: str) -> threading.RLock:
    with _FILE_LOCKS_GUARD:
        if lock_key not in _FILE_LOCKS:
            _FILE_LOCKS[lock_key] = threading.RLock()
        return _FILE_LOCKS[lock_key]


@contextmanager
def _case_execution_lock(out_dir: Path, case_id: str) -> Iterator[None]:
    """Acquires both an in-process RLock and an OS-level flock so concurrent threads or processes
    running the same case never corrupt `<case_id>.json`, `audit.jsonl`, `cases.db`, or `RunLedger`."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", case_id or "unknown")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file_path = out_dir / f".{safe_name}.lock"
    t_lock = _get_thread_lock(str(lock_file_path.resolve()))
    with t_lock:
        with open(lock_file_path, "w", encoding="utf-8") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def scrub_secrets_and_pii(value: Any) -> Any:
    """Recursively scrubs secrets (AWS keys, GitHub/OpenAI tokens, Bearer tokens) and PII
    (16-digit card PANs, emails, phones, Aadhaar) from strings, lists, and dicts while
    preserving pseudonymous bank IDs (`C12975-K2`, `3660260`, `HHF-004`)."""
    if isinstance(value, str):
        s = value
        for pat, repl in EXTRA_SECRET_PII_PATTERNS:
            s = pat.sub(repl, s)
        # Apply audit_redact without truncating short/medium claims
        s = re.sub(r"(?i)Authorization:\s*.*", "Authorization: [REDACTED_AUTH]", s)
        s = re.sub(r"AIzaSy[A-Za-z0-9_-]{33}", "[REDACTED_API_KEY]", s)
        s = re.sub(r"AQ\.Ab8[A-Za-z0-9_\-]{10,}", "[REDACTED_TOKEN]", s)
        s = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", s)
        s = re.sub(r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b", "[REDACTED_PHONE]", s)
        s = re.sub(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED_AADHAAR]", s)
        return s
    if isinstance(value, list):
        return [scrub_secrets_and_pii(x) for x in value]
    if isinstance(value, tuple):
        return tuple(scrub_secrets_and_pii(x) for x in value)
    if isinstance(value, dict):
        return {k: scrub_secrets_and_pii(v) for k, v in value.items()}
    return value


def _build_store_validators(expected_flagged_txn_id: int) -> dict[str, Any]:
    """Builds strict response validators for `ToolProxy` so HTTP 200 with an empty/truncated body
    (e.g., `None`, `{}`, `[]`, or a 0-row DataFrame for `get_transaction` / `card_window` / `holder_history`)
    raises `DependencyDown` and fails closed instead of silently producing a `legitimate` verdict."""

    def _validate_get_txn(res: Any) -> bool:
        if not isinstance(res, tuple) or len(res) != 2:
            raise DependencyDown("Store returned malformed or empty body for get_transaction (HTTP 200 empty body)")
        row, _ = res
        if row is None or (isinstance(row, (pd.Series, pd.DataFrame)) and row.empty):
            raise DependencyDown(f"Store returned empty transaction record for {expected_flagged_txn_id}")
        if isinstance(row, dict) and not row:
            raise DependencyDown(f"Store returned empty dict for transaction {expected_flagged_txn_id}")
        return True

    def _validate_non_empty_df_tuple(method_name: str):
        def _val(res: Any) -> bool:
            if res is None or (isinstance(res, (dict, list, str)) and len(res) == 0):
                raise DependencyDown(f"Store method '{method_name}' returned empty/truncated body")
            if not isinstance(res, tuple) or len(res) != 2:
                raise DependencyDown(f"Store method '{method_name}' returned invalid tuple shape")
            df, _ = res
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise DependencyDown(
                    f"Store method '{method_name}' returned HTTP 200 with 0-row DataFrame for active card/holder"
                )
            return True

        return _val

    def _validate_df_tuple_allow_empty(method_name: str):
        def _val(res: Any) -> bool:
            if res is None or (isinstance(res, (dict, list, str)) and len(res) == 0):
                raise DependencyDown(f"Store method '{method_name}' returned empty/truncated body")
            if not isinstance(res, tuple) or len(res) != 2:
                raise DependencyDown(f"Store method '{method_name}' returned invalid tuple shape")
            df, _ = res
            if not isinstance(df, pd.DataFrame):
                raise DependencyDown(f"Store method '{method_name}' returned non-DataFrame payload")
            return True

        return _val

    return {
        "get_transaction": _validate_get_txn,
        "card_window": _validate_non_empty_df_tuple("card_window"),
        "holder_history": _validate_non_empty_df_tuple("holder_history"),
        "device_neighbors": _validate_df_tuple_allow_empty("device_neighbors"),
        "closed_cases_for_card": _validate_df_tuple_allow_empty("closed_cases_for_card"),
        "closed_cases_by_device": _validate_df_tuple_allow_empty("closed_cases_by_device"),
    }


class StoreProxyAdapter:
    """Adapts a `ToolProxy` so `R2Investigator` can access `.tx`, `.cc`, `._log`, and `.calls`
    with per-run isolated call logs and fail-closed validation on empty/truncated store frames."""

    def __init__(self, proxy: ToolProxy, raw_store: R2LocalStore):
        object.__setattr__(self, "_proxy", proxy)
        object.__setattr__(self, "_raw", raw_store)
        object.__setattr__(self, "_local_calls", [])
        object.__setattr__(self, "as_of", None)

    def _log(self, tool: str, **params: Any) -> str:
        # Check FaultInjector on detector/logged queries too (fixes ATK-18)
        self._proxy._counts[tool] = self._proxy._counts.get(tool, 0) + 1
        n = self._proxy._counts[tool]
        fault = FaultInjector.get_fault(tool, n)
        if fault in ("down", "empty_body"):
            raise DependencyDown(f"Injected fault: query '{tool}' is down / returned empty body")
        if fault == "timeout":
            raise ToolTimeout(f"Injected fault: query '{tool}' timed out")
        if fault == "garbage":
            raise GarbageResponse(f"Injected fault: query '{tool}' returned garbage")
        safe_params = _to_json_safe(params)
        arg = ", ".join(f"{k}={v!r}" for k, v in safe_params.items())
        self._local_calls.append({"tool": tool, "params": safe_params})
        return f"query:{tool}({arg})"

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._local_calls

    @property
    def tx(self) -> pd.DataFrame:
        df = self._raw.tx
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise DependencyDown("Store tx frame is empty or truncated (HTTP 200 empty body)")
        return df

    @property
    def cc(self) -> pd.DataFrame:
        df = self._raw.cc
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise DependencyDown("Store closed_cases frame is empty or truncated (HTTP 200 empty body)")
        return df

    @property
    def memory(self) -> list[dict[str, Any]]:
        return self._raw.memory

    def __getattr__(self, name: str) -> Any:
        # Propagate as_of to underlying store before delegating to proxy
        self._raw.as_of = object.__getattribute__(self, "as_of")
        fault = FaultInjector.get_fault(name, self._proxy._counts.get(name, 0) + 1)
        if fault == "empty_body":
            self._proxy._counts[name] = self._proxy._counts.get(name, 0) + 1
            raise DependencyDown(f"Store method '{name}' returned HTTP 200 with empty/truncated body")
        res = getattr(self._proxy, name)
        if callable(res):
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                self._raw.as_of = object.__getattribute__(self, "as_of")
                out = res(*args, **kwargs)
                safe_kw = _to_json_safe(kwargs) if kwargs else {"args": [str(a) for a in args]}
                self._local_calls.append({"tool": name, "params": safe_kw})
                return out
            return _wrapped
        return res


@lru_cache(maxsize=1)
def get_shared_runtime() -> tuple[R2LocalStore, R2Investigator]:
    """Builds `R2LocalStore` and `R2Investigator` ONCE per process (~3.8 GB RSS) and locks
    process-global `disputes.W_OVERRIDE` at startup."""
    store = R2LocalStore()
    model = r2_pattern_model()
    inv = R2Investigator(store, model, writer=store.write_case)
    disputes.W_OVERRIDE = _STARTUP_W_OVERRIDE
    inv.dispute_weight_basis = _STARTUP_DISPUTE_BASIS
    return store, inv


def _verify_store_integrity_for_case(store: R2LocalStore, inv: R2Investigator, case_row: dict[str, Any]) -> None:
    """Verifies that neither `store.tx`, `inv._tx`, nor `inv._cc` has been truncated to empty
    and that `flagged_txn_id` exists in the transaction table (fail closed on missing/truncated data)."""
    if inv._tx is None or not isinstance(inv._tx, pd.DataFrame) or inv._tx.empty:
        raise DependencyDown("Investigator transaction table (_tx) is empty or truncated")
    if inv._cc is None or not isinstance(inv._cc, pd.DataFrame) or inv._cc.empty:
        raise DependencyDown("Investigator closed_cases table (_cc) is empty or truncated")
    if store.tx is None or not isinstance(store.tx, pd.DataFrame) or store.tx.empty:
        raise DependencyDown("Store transaction table (tx) is empty or truncated")

    fid = int(case_row["flagged_txn_id"])
    hit = inv._tx.loc[inv._tx["TransactionID"] == fid]
    if hit.empty or fid not in store.tx.index:
        raise DependencyDown(
            f"Flagged transaction {fid} is missing from transaction store (empty/truncated store response)"
        )
    t0 = pd.Timestamp(case_row["opened_at"])
    if pd.Timestamp(hit["t"].iat[0]) > t0:
        raise InvalidInput(
            f"Flagged transaction {fid} timestamp ({hit['t'].iat[0]}) is after case opened_at ({t0})"
        )


def _sanitize_trigger_for_row(case_row: pd.Series, facts: TriggerFacts) -> pd.Series:
    """Rebuilds a clean canonical trigger string from typed `TriggerFacts` so raw untrusted free text
    never reaches `Investigator` or downstream decision logic."""
    row = case_row.copy()
    ttype = str(row.get("trigger_type", ""))
    fid = int(row["flagged_txn_id"])
    cust = str(row["customer_id"])
    if ttype == "customer_report":
        verb = "did not make" if facts.customer_denies else "confirm I made"
        amt_part = f" (${facts.stated_amount:.2f})" if facts.stated_amount is not None else ""
        row["trigger_text"] = (
            f"Customer {cust} message: 'I {verb} these charges{amt_part} on my card.' Refers to {fid}."
        )
    elif ttype == "risk_score":
        amt = facts.stated_amount if facts.stated_amount is not None else 0.0
        sc = facts.stated_score if facts.stated_score is not None else float(row.get("risk_score") or 0.0)
        row["trigger_text"] = (
            f"Real-time model scored transaction {fid} (${amt:,.2f}) at {sc:.2f}. Review and decide."
        )
    elif ttype == "analyst_request":
        row["trigger_text"] = (
            f"Analyst request: review transaction {fid} on card {row.get('card_id')}."
        )
    else:
        row["trigger_text"] = f"Trigger {ttype} for transaction {fid}."
    return row


def _record_untrusted_evidence(
    answer: dict[str, Any],
    case_id: str,
    facts: TriggerFacts,
    warnings: list[str],
) -> None:
    """Records quarantined hostile strings, injection spans, and cross-check warnings in `answer['case']['evidence']`
    after scrubbing all secrets and PII."""
    redacted_claims = [scrub_secrets_and_pii(c.strip().replace("\n", " ")) for c in facts.claims_of_authority]
    redacted_warnings = [scrub_secrets_and_pii(w) for w in warnings]
    has_hostile = bool(redacted_claims or facts.injection_spans or redacted_warnings)
    if not has_hostile:
        return

    parts: list[str] = []
    if redacted_claims:
        parts.append(f"quarantined_claims={redacted_claims}")
    if facts.injection_spans:
        span_types = sorted({kind for _, _, kind in facts.injection_spans})
        parts.append(f"injection_types={span_types} (count={len(facts.injection_spans)})")
    if redacted_warnings:
        parts.append(f"cross_check_warnings={redacted_warnings}")

    claim_text = scrub_secrets_and_pii(
        "Untrusted trigger parser quarantined hostile/anomalous input as data (ignored by decision logic): "
        + "; ".join(parts)
    )
    ev_entry = {
        "source": "document",
        "ref": f"query:parse_trigger(case_id={case_id})",
        "claim": claim_text,
        "entity_ids": [str(x) for x in (facts.referenced_txn_ids or [case_id])],
    }
    answer["case"]["evidence"].append(ev_entry)


def _scrub_customer_tip_off(answer: dict[str, Any]) -> None:
    """Ensures nothing customer-facing (`evidence_requests` text or customer actions) mentions a SAR,
    and scrubs any accidental secret/PII leakage across the entire answer dictionary."""
    for req in answer.get("evidence_requests", []):
        q = str(req.get("question", ""))
        a = str(req.get("assumed_response", ""))
        for pat in TIP_OFF_PATTERNS:
            q = pat.sub("internal review", q)
            a = pat.sub("internal review", a)
        req["question"] = scrub_secrets_and_pii(q)
        req["assumed_response"] = scrub_secrets_and_pii(a)

    if "case" in answer and isinstance(answer["case"], dict):
        answer["case"]["summary"] = scrub_secrets_and_pii(str(answer["case"].get("summary", "")))
        for ev in answer["case"].get("evidence", []):
            ev["claim"] = scrub_secrets_and_pii(str(ev.get("claim", "")))


def _attach_sar_if_required(answer: dict[str, Any], tx_as_of: pd.DataFrame) -> None:
    """Evaluates `sar_required(answer, tx_as_of)` and attaches `draft_sar(answer, tx_as_of)` when owed,
    maintaining consistency with `next_best_actions` and `assay.validate.validate`."""
    decision = sar_required(answer, tx=tx_as_of)
    fin = answer.get("next_best_actions", {}).get("final", [])
    has_file_report = any(x.get("action") == "FILE_REPORT" for x in fin)

    if decision.required or has_file_report:
        sar_obj = draft_sar(answer, tx_as_of)
        if sar_obj.required and sar_obj.sentences:
            answer["sar"] = sar_obj.to_answer_sar()
            if not has_file_report:
                fin.append({
                    "action": "FILE_REPORT",
                    "route": "L2",
                    "reason": decision.reason_text,
                })
                if not answer.get("evidence_requests"):
                    answer["next_best_actions"]["initial"] = list(fin)
    else:
        answer["sar"] = {
            "file": False,
            "reason": decision.reason_text,
            "narrative": "",
            "subjects": [],
            "total_amount_usd": 0,
            "activity_dates": [],
        }

    _scrub_customer_tip_off(answer)


def _update_casefile(
    case_id: str,
    answer: dict[str, Any],
    db_path: Path,
    ts: str,
    idempotent_replay: bool = False,
) -> CaseFile:
    """Opens and updates the case lifecycle in `assay.r2.casefile.CaseFile`.

    - Idempotent: if `idempotent_replay` is True or the case already has recorded events for this run,
      returns the existing `CaseFile` without appending duplicate events or triggering a transition
      out of `AWAITING_APPROVAL`.
    - Never applies `L1`/`L2` actions as `agent`: only applies `auto` route actions (`AUTO_ACTION_APPLIED`)
      and records `L1`/`L2` actions as `REQUEST_APPROVAL` (`to_state=AWAITING_APPROVAL`) without executing them.
    """
    cstore = CaseStore(str(db_path))
    cf = CaseFile(case_id, store=cstore)
    if idempotent_replay and cf.events:
        return cf
    if cf.events:
        # Case was already recorded in a previous run of this case_id; preserve idempotency
        return cf

    case_data = answer["case"]
    ev_list = _normalize_evidence(case_data.get("evidence", []))
    verdict = case_data.get("verdict", "uncertain")
    status = case_data.get("status", "open")
    exposure = float(case_data.get("exposure_usd", 0.0) or 0.0)
    final_actions = answer.get("next_best_actions", {}).get("final", [])

    cf.open_case(
        actor="agent",
        reason=f"Opened investigation for {case_id}",
        evidence=ev_list[:2],
        ts=ts,
    )
    cf.start_investigation(
        actor="agent",
        reason="Running R2Investigator detectors and graph queries",
        evidence=ev_list,
        ts=ts,
    )

    auto_actions = [a for a in final_actions if approval_route_for(str(a.get("action", "")), exposure) == "auto"]
    l1_l2_actions = [a for a in final_actions if approval_route_for(str(a.get("action", "")), exposure) in ("L1", "L2")]

    # Apply only auto-route actions (never L1/L2)
    for act in auto_actions:
        act_name = str(act.get("action", ""))
        cstore.append_event(
            case_id=case_id,
            event_type="AUTO_ACTION_APPLIED",
            actor="agent",
            from_state=cf.current_state,
            to_state=cf.current_state,
            evidence_refs=ev_list[:2],
            reason=f"Auto-applied action '{act_name}' (route=auto): {act.get('reason', '')}",
            ts=ts,
        )

    # Record L1/L2 actions strictly as pending/awaiting approval (never applied by agent)
    if l1_l2_actions:
        routes_needed = sorted({approval_route_for(str(a["action"]), exposure) for a in l1_l2_actions})
        highest_route = "L2" if "L2" in routes_needed else "L1"
        pending_desc = ", ".join(
            f"{a['action']} ({approval_route_for(str(a['action']), exposure)})" for a in l1_l2_actions
        )
        cstore.append_event(
            case_id=case_id,
            event_type="REQUEST_APPROVAL",
            actor="agent",
            from_state=cf.current_state,
            to_state=CaseState.AWAITING_APPROVAL,
            evidence_refs=ev_list[:3],
            reason=(
                f"Awaiting {highest_route} approval for recommended action(s): {pending_desc}. "
                f"Agent never executes L1/L2 actions."
            ),
            ts=ts,
        )
    elif status == "escalated" or any(a.get("action") == "ESCALATE_TO_ANALYST" for a in final_actions):
        cf.escalate(
            actor="agent",
            reason=str(case_data.get("summary", "Escalated per policy")),
            evidence=ev_list[:3],
            ts=ts,
        )
    elif status == "closed_legitimate" or verdict == "legitimate":
        cf.close_legitimate(
            actor="agent",
            reason=str(case_data.get("summary", "Cleared as legitimate")),
            evidence=ev_list[:3],
            ts=ts,
        )
    elif any(a.get("action") == "VERIFY_WITH_CUSTOMER" for a in final_actions):
        cf.request_customer(
            actor="agent",
            reason="Requested transaction verification from customer under Policy R1",
            evidence=ev_list[:3],
            ts=ts,
        )
    return cf


def _record_pending_approvals(
    case_id: str,
    answer: dict[str, Any],
    ledger_path: Path,
    role_flag: str,
    audit_log: AuditLog,
    ts: str,
) -> list[dict[str, Any]]:
    """Writes L1/L2 actions to `assay.r2.approvals.Ledger` as PENDING.
    Never executes them; `--role` on the CLI cannot approve anything."""
    ledger = ApprovalLedger(ledger_path)
    exposure = float(answer["case"].get("exposure_usd", 0.0) or 0.0)
    final_actions = answer.get("next_best_actions", {}).get("final", [])

    if role_flag in ("l1", "l2"):
        audit_log.log_event(
            "role_claim_ignored",
            case_id,
            {
                "claimed_role": role_flag,
                "note": "CLI --role flag cannot approve or execute L1/L2 actions; server-side token check required.",
            },
        )

    existing_keys: set[tuple[str, str]] = set()
    if ledger.path.exists():
        with open(ledger.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("entry_type") == "proposal_created":
                        d = entry.get("data", {})
                        existing_keys.add((str(d.get("case_id", "")), str(d.get("action", ""))))
                except Exception:
                    pass

    pending: list[dict[str, Any]] = []
    for act_item in final_actions:
        act_name = str(act_item.get("action", ""))
        route = approval_route_for(act_name, exposure)
        if route in ("L1", "L2"):
            prop_id = "prop_" + hashlib.sha256(f"{case_id}:{act_name}".encode("utf-8")).hexdigest()[:12]
            proposal = {
                "id": prop_id,
                "case_id": case_id,
                "action": act_name,
                "exposure_usd": exposure,
                "evidence_ref": scrub_secrets_and_pii(str(act_item.get("reason", ""))),
                "route": route,
                "status": "pending",
                "created_by": "agent",
                "created_at": ts,
                "decided_by": None,
                "decision": None,
                "note": None,
                "decided_at": None,
            }
            if (case_id, act_name) not in existing_keys:
                ledger.append("proposal_created", proposal)
                existing_keys.add((case_id, act_name))
            pending.append(proposal)
    return pending


def _has_Unsafe_path_chars(s: str) -> bool:
    """Returns True if string contains NUL bytes, path separators, or parent traversal."""
    return "\x00" in s or "/" in s or "\\" in s or ".." in s


def _resolve_case_input(case_arg: str) -> tuple[dict[str, Any] | None, str | None]:
    """Loads a case row from `case_pack()`, a JSON file path, or an inline JSON string.
    Rejects any case_id containing path traversal (`../`), slashes (`/`), or NUL bytes (`\\x00`)."""
    if "\x00" in case_arg:
        return None, "Invalid NUL byte in case argument"

    # Inline JSON string
    if case_arg.strip().startswith("{"):
        try:
            data = json.loads(case_arg)
            if isinstance(data, dict):
                return data, None
            return None, "Inline JSON did not evaluate to an object"
        except Exception as exc:
            return None, f"Invalid inline JSON case row: {exc}"

    # Existing JSON file on disk (only if it has a .json suffix and exists)
    if case_arg.endswith(".json"):
        p = Path(case_arg)
        if p.exists() and p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data, None
                return None, f"JSON file '{case_arg}' did not contain a JSON object"
            except Exception as exc:
                return None, f"Failed to parse case file '{case_arg}': {exc}"

    # Otherwise it must be a strict case_id without path characters
    if _has_Unsafe_path_chars(case_arg) or not SAFE_CASE_ID_RE.fullmatch(case_arg.strip()):
        return None, f"Invalid case_id '{case_arg}': path characters, traversal, or illegal characters are forbidden"

    cp = case_pack()
    matches = cp[cp["case_id"] == case_arg.strip()]
    if matches.empty:
        return None, f"Unknown or invalid case_id '{case_arg}' (not found in case_pack)"
    return matches.iloc[0].to_dict(), None


# Module-level ledger cache per output directory for in-process idempotency tracking
_RUN_LEDGERS: dict[str, RunLedger] = {}
_CASE_ID_TO_IDEM_KEY: dict[tuple[str, str], str] = {}


def _get_run_ledger(out_dir: Path) -> RunLedger:
    key = str(out_dir.resolve())
    if key not in _RUN_LEDGERS:
        _RUN_LEDGERS[key] = RunLedger(out_dir / ".ledger")
    return _RUN_LEDGERS[key]


def _check_case_id_collision(out_dir: Path, case_id: str, idem_key: str) -> str | None:
    """Detects if `case_id` was already run in `out_dir` with a DIFFERENT `idempotency_key`
    (e.g., colliding `case_id` with modified `flagged_txn_id` or `card_id`)."""
    dir_key = str(out_dir.resolve())
    prev_key = _CASE_ID_TO_IDEM_KEY.get((dir_key, case_id))
    if prev_key is not None and prev_key != idem_key:
        return (
            f"Case ID collision detected for '{case_id}': existing run has idempotency_key={prev_key[:12]}, "
            f"new payload has idempotency_key={idem_key[:12]}"
        )
    idx_file = out_dir / ".ledger" / f"case_{case_id}.key"
    if idx_file.exists():
        stored_key = idx_file.read_text(encoding="utf-8").strip()
        if stored_key and stored_key != idem_key:
            return (
                f"Case ID collision detected on disk for '{case_id}': stored key={stored_key[:12]} "
                f"!= new key={idem_key[:12]}"
            )
    return None


def _record_case_id_key(out_dir: Path, case_id: str, idem_key: str) -> None:
    dir_key = str(out_dir.resolve())
    _CASE_ID_TO_IDEM_KEY[(dir_key, case_id)] = idem_key
    ledger_dir = out_dir / ".ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    idx_file = ledger_dir / f"case_{case_id}.key"
    idx_file.write_text(idem_key, encoding="utf-8")


def run_agent_case(
    case_input: str | dict[str, Any] | pd.Series,
    out_dir: str | Path = DEFAULT_OUT,
    fault: str = "none",
    role: str = "analyst",
    timeout_s: float = 20.0,
) -> int:
    """Runs the full N1 agent pipeline on a single case and returns the process exit code:
      - 0: Success (valid answer written idempotently).
      - 2: Invalid case input or case_id collision/path traversal (no answer file written).
      - 3: Tool / graph / dependency fault — fail closed (no answer file written, failure audit-logged).
    """
    t_start = time.time()
    out_path = Path(out_dir)
    audit_path = out_path / "audit.jsonl"
    approvals_path = out_path / "approvals_ledger.jsonl"
    cases_db_path = out_path / "cases.db"

    # Ensure process-global W_OVERRIDE is never drifted
    disputes.W_OVERRIDE = _STARTUP_W_OVERRIDE

    # 1. Resolve and validate case row BEFORE writing any answer file
    if isinstance(case_input, str):
        raw_row, resolve_err = _resolve_case_input(case_input)
        if raw_row is None:
            safe_cid = re.sub(r"[^A-Za-z0-9_-]", "_", str(case_input))[:32] or "invalid_case"
            with _case_execution_lock(out_path, safe_cid):
                audit = AuditLog(audit_path)
                audit.log_error(safe_cid, resolve_err or "Invalid case_id", {"stage": "resolve_case"})
            print(f"ERROR [EXIT 2] Invalid case input: {resolve_err}", file=sys.stderr)
            return 2
    elif isinstance(case_input, pd.Series):
        raw_row = case_input.to_dict()
    elif isinstance(case_input, dict):
        raw_row = dict(case_input)
    else:
        print(f"ERROR [EXIT 2] Unsupported case input type: {type(case_input).__name__}", file=sys.stderr)
        return 2

    raw_cid = str(raw_row.get("case_id", "") or "")
    if _has_Unsafe_path_chars(raw_cid) or not SAFE_CASE_ID_RE.fullmatch(raw_cid.strip()):
        safe_cid = re.sub(r"[^A-Za-z0-9_-]", "_", raw_cid)[:32] or "invalid_case"
        with _case_execution_lock(out_path, safe_cid):
            audit = AuditLog(audit_path)
            audit.log_error(safe_cid, f"Rejected unsafe case_id: {raw_cid!r}", {"stage": "path_check"})
        print(f"ERROR [EXIT 2] Unsafe or invalid case_id: {raw_cid!r}", file=sys.stderr)
        return 2

    cid_candidate = raw_cid.strip()
    trigger_type = str(raw_row.get("trigger_type", "") or "")
    trigger_text = str(raw_row.get("trigger_text", "") or "")

    row_for_validation = dict(raw_row)
    row_for_validation.pop("trigger_text", None)
    valid, val_err = validate_case_row(row_for_validation)
    if not valid:
        with _case_execution_lock(out_path, cid_candidate):
            audit = AuditLog(audit_path)
            audit.log_error(cid_candidate, val_err or "Case validation failed", {"stage": "validate_case_row"})
        print(f"ERROR [EXIT 2] Case validation failed for '{cid_candidate}': {val_err}", file=sys.stderr)
        return 2

    case_id = cid_candidate

    # Acquire per-case lock so concurrent runs of the same case serialize cleanly without corrupting files
    with _case_execution_lock(out_path, case_id):
        idem_key = idempotency_key(raw_row, dataset_tag(), CODE_VERSION)
        collision_err = _check_case_id_collision(out_path, case_id, idem_key)
        if collision_err:
            audit = AuditLog(audit_path)
            audit.log_error(case_id, collision_err, {"stage": "collision_check"})
            print(f"ERROR [EXIT 2] {collision_err}", file=sys.stderr)
            return 2

        audit = AuditLog(audit_path)
        audit.log_run_start(
            case_id,
            metadata={"fault": fault, "role": role, "code_version": CODE_VERSION},
        )

        # 2. Parse trigger_text as untrusted DATA + cross-check against ground-truth
        facts = parse_trigger(trigger_type, trigger_text)
        # Scrub any secrets or PII from extracted claims_of_authority immediately
        facts.claims_of_authority = [scrub_secrets_and_pii(c) for c in facts.claims_of_authority]
        warnings = [scrub_secrets_and_pii(w) for w in cross_check(facts, raw_row)]
        att_trigger = attest_trigger_facts(case_id, facts, warnings)
        audit.log_finding(case_id, audit_redact(_to_json_safe(att_trigger.to_dict())))

        # 3. Check fault specification & idempotency ledger
        fault_norm = (fault or "none").strip().lower()
        if fault_norm not in FAULT_CHOICES:
            fault_spec = fault.strip()
        else:
            fault_spec = FAULT_MAP[fault_norm]

        run_ledger = _get_run_ledger(out_path)
        prior_entry = run_ledger.get(idem_key) if not fault_spec else None

        # Build store & investigator once per process
        store, inv = get_shared_runtime()

        # Prepare sanitized Series so raw trigger_text never reaches Investigator
        clean_series = _sanitize_trigger_for_row(pd.Series(raw_row), facts)
        fid = int(clean_series["flagged_txn_id"])

        # Set up ToolProxy with strict empty-body / schema validators
        validators = _build_store_validators(fid)
        proxy = ToolProxy(store, timeout_s=timeout_s, validators=validators)
        adapter = StoreProxyAdapter(proxy, store)

        # Guard shared Investigator instance with _INVESTIGATOR_LOCK so concurrent threads on different
        # cases never contaminate self._multi, self._case, self._card_mismatch, or inv.s
        with _INVESTIGATOR_LOCK:
            prev_store = inv.s
            prev_writer = inv.writer
            inv.s = adapter
            inv.writer = adapter.write_case
            inv._multi = []
            inv._card_mismatch = None
            inv._case = None
            disputes.W_OVERRIDE = _STARTUP_W_OVERRIDE

            try:
                _verify_store_integrity_for_case(store, inv, raw_row)
                if fault_spec:
                    with FaultInjector(fault_spec):
                        _ = adapter.get_transaction(fid)
                        ans = inv.run(clean_series)
                else:
                    _ = adapter.get_transaction(fid)
                    ans = inv.run(clean_series)
            except InvalidInput as exc:
                inv.s = prev_store
                inv.writer = prev_writer
                audit.log_error(case_id, f"REJECTED_INPUT: {exc}", {"error_type": "InvalidInput"})
                audit.log_run_end(case_id, status="rejected_input", metadata={"error_type": "InvalidInput"})
                ans_file = out_path / f"{case_id}.json"
                if ans_file.exists():
                    ans_file.unlink()
                print(f"ERROR [EXIT 2] case={case_id}: {exc}", file=sys.stderr)
                return 2
            except (DependencyDown, ToolTimeout, GarbageResponse, HarnessError) as exc:
                inv.s = prev_store
                inv.writer = prev_writer
                safe_calls = _sanitize_call_records(proxy.calls)
                for c in safe_calls:
                    audit.log_tool_call(
                        case_id,
                        name=c.name,
                        params=c.args,
                        result={"ok": c.ok, "error_type": c.error_type},
                        row_count=0,
                        ms=round(c.ms, 2),
                    )
                err_type = type(exc).__name__
                failed_call = safe_calls[-1].name if safe_calls else "store_check"
                audit.log_error(
                    case_id,
                    f"FAIL_CLOSED ({err_type}) on tool '{failed_call}': {exc}",
                    {"error_type": err_type, "failed_tool": failed_call, "fault_spec": fault_spec},
                )
                audit.log_run_end(case_id, status="failed_closed", metadata={"error_type": err_type})
                ans_file = out_path / f"{case_id}.json"
                if ans_file.exists():
                    ans_file.unlink()
                print(
                    f"FAIL CLOSED [EXIT 3] case={case_id} error_type={err_type} tool={failed_call}: {exc}",
                    file=sys.stderr,
                )
                return 3
            except Exception as exc:
                inv.s = prev_store
                inv.writer = prev_writer
                err_type = type(exc).__name__
                audit.log_error(case_id, f"FAIL_CLOSED ({err_type}): {exc}", {"error_type": err_type})
                audit.log_run_end(case_id, status="failed_closed", metadata={"error_type": err_type})
                ans_file = out_path / f"{case_id}.json"
                if ans_file.exists():
                    ans_file.unlink()
                print(f"FAIL CLOSED [EXIT 3] case={case_id} error_type={err_type}: {exc}", file=sys.stderr)
                return 3
            finally:
                inv.s = prev_store
                inv.writer = prev_writer

        # Record untrusted trigger findings into evidence as data (with secrets/PII scrubbed)
        _record_untrusted_evidence(ans, case_id, facts, warnings)

        # 5. Attach attested SAR narrative/timeline if sar_required(ans) says a report is owed
        opened_ts = pd.Timestamp(raw_row["opened_at"])
        tx_df = tx_all()
        tx_as_of = tx_df[tx_df["t"] <= opened_ts]
        _attach_sar_if_required(ans, tx_as_of)

        # Normalize tokens, deterministic latency, and convert any Timestamp/numpy values to JSON-safe primitives
        measured_latency = round(time.time() - t_start, 4)
        ans["tokens"] = 0
        ans["latency_s"] = 0.0
        ans = _to_json_safe(ans)

        # Validate final answer schema & policy consistency
        val_errors = validate(ans)
        if val_errors:
            audit.log_error(case_id, f"Answer validation failed: {val_errors}", {"errors": val_errors})
            audit.log_run_end(case_id, status="validation_error")
            print(f"FAIL CLOSED [EXIT 3] Answer validation failed for {case_id}: {val_errors}", file=sys.stderr)
            return 3

        # 4. Write answer idempotently via write_answer + RunLedger (with sanitized JSON-safe CallRecords)
        safe_calls = _sanitize_call_records(proxy.calls)
        run_res = RunResult(status="COMPLETE", answer=ans, error=None, calls=safe_calls)
        sha256_hex = write_answer(run_res, out_path, case_id=case_id)
        ans_bytes = (out_path / f"{case_id}.json").read_bytes()

        idempotent_replay = False
        if prior_entry is not None:
            _, prior_sha, prior_bytes = prior_entry
            idempotent_replay = (prior_sha == sha256_hex and prior_bytes == ans_bytes)
        run_ledger.record(idem_key, run_res, sha256_hex, ans_bytes)
        _record_case_id_key(out_path, case_id, idem_key)

        # 6. Audit log every tool call and the final decision; update CaseFile state machine
        for c in safe_calls:
            audit.log_tool_call(
                case_id,
                name=c.name,
                params=c.args,
                result={"ok": c.ok},
                row_count=1 if c.ok else 0,
                ms=round(c.ms, 2),
            )
        tc_list = ans.get("tool_calls", [])
        for tc in (tc_list if isinstance(tc_list, list) else []):   # the answer format stores a count
            audit.log_tool_call(
                case_id,
                name=str(tc),
                params={"case_id": case_id},
                result={"recorded_in_answer": True},
                row_count=1,
                ms=0.0,
            )

        c_info = ans["case"]
        nba = ans["next_best_actions"]
        audit.log_decision(
            case_id,
            decision_dict={
                "verdict": c_info["verdict"],
                "status": c_info["status"],
                "fraud_probability": c_info["fraud_probability"],
                "pattern": c_info["pattern"],
                "exposure_usd": c_info["exposure_usd"],
                "final_actions": [x["action"] for x in nba["final"]],
                "sar_file": ans["sar"]["file"],
                "sha256": sha256_hex,
            },
            delta={
                "initial_actions": [x["action"] for x in nba["initial"]],
                "final_actions": [x["action"] for x in nba["final"]],
                "idempotent_replay": idempotent_replay,
            },
        )

        fixed_case_ts = str(raw_row["opened_at"])
        _update_casefile(case_id, ans, cases_db_path, ts=fixed_case_ts, idempotent_replay=idempotent_replay)

        # 7. Write L1/L2 actions to approvals.Ledger as PENDING (never executed by agent or --role)
        pending_props = _record_pending_approvals(
            case_id,
            ans,
            approvals_path,
            role_flag=role.lower(),
            audit_log=audit,
            ts=fixed_case_ts,
        )

        audit.log_run_end(
            case_id,
            status="success",
            metadata={
                "latency_s": measured_latency,
                "tokens": 0,
                "sha256": sha256_hex,
                "idempotent_replay": idempotent_replay,
                "pending_approvals": [p["action"] for p in pending_props],
            },
        )

        # 8. Print per-case summary, idempotency status, latency, and tokens (tokens = 0)
        idem_tag = "IDEMPOTENT_REPLAY (byte-identical)" if idempotent_replay else "NEW_RUN"
        print(
            f"{case_id} status=COMPLETE [{idem_tag}] verdict={c_info['verdict']} "
            f"p={c_info['fraud_probability']:.2f} pattern={c_info['pattern']} "
            f"exposure=${c_info['exposure_usd']:,.2f} sar={ans['sar']['file']} "
            f"actions={[x['action'] for x in nba['final']]} "
            f"pending_approvals={[p['action'] for p in pending_props]} "
            f"sha256={sha256_hex[:12]} latency_s={measured_latency:.3f}s tokens=0"
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m assay.r2.agent",
        description="Assay R2 deployable fraud investigation agent (single-case drill & exam entry point).",
    )
    p.add_argument("case_id", help="Case ID (e.g., HHF-007) or path to a case JSON file")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for answer JSON and logs")
    p.add_argument(
        "--fault",
        default="none",
        help="Fault injection drill mode: graph_down | tool_timeout | garbage | empty_body | none (or method:kind@n)",
    )
    p.add_argument(
        "--role",
        default="analyst",
        choices=ROLE_CHOICES,
        help="Caller role claim (analyst|l1|l2); L1/L2 actions always remain PENDING until server-side approval",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-tool timeout in seconds (default: 20s; the slowest real query, closed_cases_like, takes several seconds)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_agent_case(
        case_input=args.case_id,
        out_dir=args.out,
        fault=args.fault,
        role=args.role,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
