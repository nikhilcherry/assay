"""Restricted MCP tool server over stdio implemented directly with JSON-RPC 2.0.

Provides 5 read-only graph query tools and 1 write tool (write_case) with strict
input validation, JSON Schema input validation, SQL/GSQL injection protection,
date range limits (<= 120 days), row count truncation (<= 500 rows), approval checks
for sensitive actions (BLOCK/FILE_REPORT), detailed per-call logging, and a single
clean DataFrame -> JSON conversion helper handling NaN/NaT/numpy types.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from assay.r2.frames import closed_cases, tx_all
from assay.store import Store

INJECTION_RE = re.compile(
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE|RUN\s+QUERY|INTERPRET|HAVING|ACCUM|POST-ACCUM)\b|--|/\*|\*/|;\s*\w|OR\s+1\s*=\s*1|AND\s+1\s*=\s*1)",
    re.IGNORECASE,
)

CARD_ID_RE = re.compile(r"^C\d{5}-K\d+$")


def _clean_val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v.isoformat()
    if isinstance(v, np.ndarray):
        return [_clean_val(x) for x in v.tolist()]
    if isinstance(v, (list, tuple, set)):
        return [_clean_val(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean_val(val) for k, val in v.items()}
    if isinstance(v, pd.Series):
        return _clean_data(v.to_dict())
    if isinstance(v, pd.DataFrame):
        return _clean_data(v)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (str, bool)):
        return v
    return str(v)


def _clean_data(obj: Any) -> Any:
    """Universal DataFrame/Series/Dict/List -> JSON object conversion helper.

    Ensures NaN/NaT/numpy arrays/Series/scalars are safely converted without
    triggering numpy array truthiness ambiguity errors.
    """
    if isinstance(obj, pd.DataFrame):
        recs = obj.to_dict("records")
        return [{str(k): _clean_val(v) for k, v in r.items()} for r in recs]
    if isinstance(obj, pd.Series):
        return {str(k): _clean_val(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, dict):
        return {str(k): _clean_val(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_clean_val(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [_clean_val(x) for x in obj]
    return _clean_val(obj)


def _check_injection(val: Any) -> None:
    if isinstance(val, str):
        if INJECTION_RE.search(val):
            raise ValueError(f"Security violation: SQL/GSQL injection pattern detected in input: '{val}'")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_injection(k)
            _check_injection(v)
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            _check_injection(item)


class R2Store(Store):
    """Store implementation using Round 2 frames (tx_all and closed_cases)."""

    def __init__(self):
        super().__init__()
        self.memory: list[dict] = []
        self._tx = tx_all()
        self._cc = closed_cases()

    @property
    def tx(self) -> pd.DataFrame:
        return self._tx

    @property
    def cc(self) -> pd.DataFrame:
        return self._cc

    def get_transaction(self, txn_id: int) -> Tuple[pd.Series, str]:
        ref = self._log("get_transaction", txn_id=txn_id)
        sub = self._tx[self._tx.TransactionID == txn_id]
        if sub.empty:
            raise ValueError(f"Transaction ID {txn_id} not found")
        return sub.iloc[0], ref

    def card_window(self, card_id: str, start: Any, end: Any) -> Tuple[pd.DataFrame, str]:
        ref = self._log("card_window", card_id=card_id, start=str(start), end=str(end))
        s_dt = pd.to_datetime(start)
        e_dt = pd.to_datetime(end)
        w = self._tx[(self._tx.card_id == card_id) & (self._tx.t >= s_dt) & (self._tx.t <= e_dt)]
        return w, ref

    def holder_history(self, card_id: str) -> Tuple[pd.DataFrame, str]:
        ref = self._log("holder_history", card_id=card_id)
        sub_card = self._tx[self._tx.card_id == card_id]
        if sub_card.empty:
            return self._tx.iloc[:0], ref
        cust_id = sub_card.iloc[0]["customer_id"]
        return self._tx[self._tx.customer_id == cust_id], ref

    def device_neighbors(self, card_id: str) -> Tuple[pd.DataFrame, str]:
        ref = self._log("device_neighbors", card_id=card_id)
        sub_card = self._tx[self._tx.card_id == card_id]
        if sub_card.empty:
            return self._tx.iloc[:0], ref
        dev_profiles = [p for p in sub_card["device_profile"].dropna().unique() if str(p).strip()]
        if not dev_profiles:
            return self._tx.iloc[:0], ref
        return self._tx[self._tx.device_profile.isin(dev_profiles)], ref

    def prior_cases(self, card_id: str) -> Tuple[pd.DataFrame, str]:
        ref = self._log("prior_cases", card_id=card_id)
        m1 = self._cc["card_id"] == card_id
        if "connected_list" in self._cc.columns:
            m2 = self._cc["connected_list"].map(
                lambda l: card_id in l if isinstance(l, (list, set, tuple, np.ndarray)) and len(l) > 0 else False
            )
            res = self._cc[m1 | m2].drop_duplicates(subset=["case_id"])
        else:
            res = self._cc[m1]
        return res, ref

    def write_case(self, answer: dict, card_id: str = "") -> str:
        case_id = answer.get("case_id") or answer.get("id") or "UNKNOWN"
        gid = f"ASSAY-{case_id}"
        c = answer.get("case", {})
        conn = set(c.get("connected_card_ids", [])) if isinstance(c, dict) else set()
        self.memory.append({
            "v_id": gid,
            "card_id": card_id,
            "connected": conn,
            "attributes": {
                "verdict": c.get("verdict") if isinstance(c, dict) else "",
                "pattern": c.get("pattern") if isinstance(c, dict) else "",
                "alert_id": case_id,
            },
        })
        self._log("write_case", graph_case_id=gid)
        return gid


@dataclass
class CallLogEntry:
    tool: str
    params: dict
    row_count: int
    ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MCPServer:
    """JSON-RPC 2.0 MCP Server over stdio without external dependencies."""

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(self, store: Optional[Any] = None):
        self.store = store if store is not None else R2Store()
        self.append_only_store: list[dict] = []
        self.call_log: list[CallLogEntry] = []

    @property
    def tx(self) -> pd.DataFrame:
        if hasattr(self.store, "tx"):
            return getattr(self.store, "tx")
        return tx_all()

    @property
    def cc(self) -> pd.DataFrame:
        if hasattr(self.store, "cc"):
            return getattr(self.store, "cc")
        return closed_cases()

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "get_transaction",
                "description": "Fetch details for a transaction by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "txn_id": {
                            "type": ["integer", "string"],
                            "description": "Transaction ID (integer or digit string)",
                        }
                    },
                    "required": ["txn_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "card_window",
                "description": "Fetch transactions for a card within a date window (max 120 days)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "card_id": {
                            "type": "string",
                            "description": "Card ID (e.g. C09582-K1)",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start timestamp (ISO format)",
                        },
                        "end": {
                            "type": "string",
                            "description": "End timestamp (ISO format)",
                        },
                    },
                    "required": ["card_id", "start", "end"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "holder_history",
                "description": "Fetch transaction history for the cardholder",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "card_id": {
                            "type": "string",
                            "description": "Card ID (e.g. C09582-K1)",
                        }
                    },
                    "required": ["card_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "device_neighbors",
                "description": "Find transactions sharing device profiles with the specified card",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "card_id": {
                            "type": "string",
                            "description": "Card ID (e.g. C09582-K1)",
                        }
                    },
                    "required": ["card_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "prior_cases",
                "description": "Fetch prior closed cases involving the specified card",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "card_id": {
                            "type": "string",
                            "description": "Card ID (e.g. C09582-K1)",
                        }
                    },
                    "required": ["card_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "write_case",
                "description": "Write an investigated case object to the append-only store",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "case_json": {
                            "type": "object",
                            "description": "Case object following answer format",
                        }
                    },
                    "required": ["case_json"],
                    "additionalProperties": False,
                },
            },
        ]

    def _validate_args(self, tool_name: str, args: dict) -> None:
        schemas = {t["name"]: t["inputSchema"] for t in self.get_tool_schemas()}
        if tool_name not in schemas:
            raise ValueError(f"Unknown tool: '{tool_name}'")

        schema = schemas[tool_name]
        allowed = set(schema["properties"].keys())
        unknown = set(args.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown parameter(s) for tool '{tool_name}': {sorted(list(unknown))}")

        for req in schema.get("required", []):
            if req not in args:
                raise ValueError(f"Missing required parameter '{req}' for tool '{tool_name}'")

        _check_injection(args)

        if "card_id" in args:
            card_id = str(args["card_id"])
            if not CARD_ID_RE.match(card_id):
                raise ValueError(f"Invalid card_id format: '{card_id}'. Must match pattern ^C\\d{{5}}-K\\d+$")

        if "txn_id" in args:
            txn_id = args["txn_id"]
            if isinstance(txn_id, str):
                if not txn_id.isdigit():
                    raise ValueError(f"Invalid txn_id format: '{txn_id}'. Must be an integer or digit string")
            elif not isinstance(txn_id, (int, np.integer)):
                raise ValueError(f"Invalid txn_id type: {type(txn_id).__name__}")

    def call_tool(self, tool_name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        self._validate_args(tool_name, args)

        if tool_name == "get_transaction":
            res = self._exec_get_transaction(args)
        elif tool_name == "card_window":
            res = self._exec_card_window(args)
        elif tool_name == "holder_history":
            res = self._exec_holder_history(args)
        elif tool_name == "device_neighbors":
            res = self._exec_device_neighbors(args)
        elif tool_name == "prior_cases":
            res = self._exec_prior_cases(args)
        elif tool_name == "write_case":
            res = self._exec_write_case(args)
        else:
            raise ValueError(f"Unknown tool: '{tool_name}'")

        ms = round((time.perf_counter() - t0) * 1000, 2)
        row_count = res.get("total_rows", 1) if isinstance(res, dict) else 1
        self.call_log.append(CallLogEntry(tool=tool_name, params=args, row_count=row_count, ms=ms))
        return res

    def _exec_get_transaction(self, args: dict) -> dict:
        txn_id = int(args["txn_id"])
        if hasattr(self.store, "get_transaction"):
            try:
                row_or_df, ref = self.store.get_transaction(txn_id)
                rec = _clean_data(row_or_df)
                return {"transaction": rec, "total_rows": 1, "truncated": False}
            except Exception as e:
                if "not found" in str(e).lower():
                    raise ValueError(str(e))
        sub = self.tx[self.tx.TransactionID == txn_id]
        if sub.empty:
            raise ValueError(f"Transaction ID {txn_id} not found")
        rec = _clean_data(sub.iloc[0])
        return {"transaction": rec, "total_rows": 1, "truncated": False}

    def _exec_card_window(self, args: dict) -> dict:
        card_id = str(args["card_id"])
        start_dt = pd.to_datetime(args["start"])
        end_dt = pd.to_datetime(args["end"])

        if end_dt < start_dt:
            raise ValueError("Start date must be before or equal to end date")

        diff_days = (end_dt - start_dt).total_seconds() / 86400.0
        if diff_days > 120.0:
            raise ValueError(f"Date range of {diff_days:.1f} days exceeds maximum limit of 120 days")

        if hasattr(self.store, "card_window"):
            try:
                df, ref = self.store.card_window(card_id, args["start"], args["end"])
            except TypeError:
                df = self.tx[(self.tx.card_id == card_id) & (self.tx.t >= start_dt) & (self.tx.t <= end_dt)]
        else:
            df = self.tx[(self.tx.card_id == card_id) & (self.tx.t >= start_dt) & (self.tx.t <= end_dt)]

        recs = _clean_data(df)
        if not isinstance(recs, list):
            recs = [recs] if recs else []

        total = len(recs)
        truncated = total > 500
        if truncated:
            recs = recs[:500]

        return {
            "card_id": card_id,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "transactions": recs,
            "total_rows": total,
            "returned_rows": len(recs),
            "truncated": truncated,
        }

    def _exec_holder_history(self, args: dict) -> dict:
        card_id = str(args["card_id"])
        if hasattr(self.store, "holder_history"):
            try:
                df, ref = self.store.holder_history(card_id)
            except Exception:
                sub_card = self.tx[self.tx.card_id == card_id]
                if sub_card.empty:
                    df = self.tx.iloc[:0]
                else:
                    cust_id = sub_card.iloc[0]["customer_id"]
                    df = self.tx[self.tx.customer_id == cust_id]
        else:
            sub_card = self.tx[self.tx.card_id == card_id]
            if sub_card.empty:
                df = self.tx.iloc[:0]
            else:
                cust_id = sub_card.iloc[0]["customer_id"]
                df = self.tx[self.tx.customer_id == cust_id]

        recs = _clean_data(df)
        if not isinstance(recs, list):
            recs = [recs] if recs else []

        total = len(recs)
        truncated = total > 500
        if truncated:
            recs = recs[:500]

        cust_id = recs[0].get("customer_id") if recs else None
        return {
            "card_id": card_id,
            "customer_id": cust_id,
            "transactions": recs,
            "total_rows": total,
            "returned_rows": len(recs),
            "truncated": truncated,
        }

    def _exec_device_neighbors(self, args: dict) -> dict:
        card_id = str(args["card_id"])
        if hasattr(self.store, "device_neighbors"):
            try:
                df, ref = self.store.device_neighbors(card_id)
            except Exception:
                sub_card = self.tx[self.tx.card_id == card_id]
                dev_profiles = [p for p in sub_card["device_profile"].dropna().unique() if str(p).strip()] if not sub_card.empty else []
                df = self.tx[self.tx.device_profile.isin(dev_profiles)] if dev_profiles else self.tx.iloc[:0]
        else:
            sub_card = self.tx[self.tx.card_id == card_id]
            dev_profiles = [p for p in sub_card["device_profile"].dropna().unique() if str(p).strip()] if not sub_card.empty else []
            df = self.tx[self.tx.device_profile.isin(dev_profiles)] if dev_profiles else self.tx.iloc[:0]

        recs = _clean_data(df)
        if not isinstance(recs, list):
            recs = [recs] if recs else []

        total = len(recs)
        truncated = total > 500
        if truncated:
            recs = recs[:500]

        dev_profiles = list({r["device_profile"] for r in recs if r.get("device_profile")})
        return {
            "card_id": card_id,
            "device_profiles": dev_profiles,
            "neighbors": recs,
            "total_rows": total,
            "returned_rows": len(recs),
            "truncated": truncated,
        }

    def _exec_prior_cases(self, args: dict) -> dict:
        card_id = str(args["card_id"])
        if hasattr(self.store, "prior_cases"):
            try:
                df, ref = self.store.prior_cases(card_id)
            except Exception:
                cc = self.cc
                m1 = cc["card_id"] == card_id
                if "connected_list" in cc.columns:
                    m2 = cc["connected_list"].map(
                        lambda l: card_id in l if isinstance(l, (list, set, tuple, np.ndarray)) and len(l) > 0 else False
                    )
                    df = cc[m1 | m2].drop_duplicates(subset=["case_id"])
                else:
                    df = cc[m1]
        else:
            cc = self.cc
            m1 = cc["card_id"] == card_id
            if "connected_list" in cc.columns:
                m2 = cc["connected_list"].map(
                    lambda l: card_id in l if isinstance(l, (list, set, tuple, np.ndarray)) and len(l) > 0 else False
                )
                df = cc[m1 | m2].drop_duplicates(subset=["case_id"])
            else:
                df = cc[m1]

        recs = _clean_data(df)
        if not isinstance(recs, list):
            recs = [recs] if recs else []

        total = len(recs)
        truncated = total > 500
        if truncated:
            recs = recs[:500]

        return {
            "card_id": card_id,
            "cases": recs,
            "total_rows": total,
            "returned_rows": len(recs),
            "truncated": truncated,
        }

    def _exec_write_case(self, args: dict) -> dict:
        case_json = args["case_json"]
        if not isinstance(case_json, dict):
            raise ValueError("case_json must be an object")

        if "case_id" not in case_json and "id" not in case_json:
            raise ValueError("Invalid case_json format: missing required key 'case_id'")
        if "case" not in case_json:
            raise ValueError("Invalid case_json format: missing required key 'case'")

        case_id = str(case_json.get("case_id") or case_json.get("id"))

        json_str = json.dumps(case_json)
        requires_approval = False
        for act in ["BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT", "BLOCK"]:
            if act in json_str:
                requires_approval = True
                break

        if requires_approval:
            approval_keys = [
                "approval_id",
                "approval_ids",
                "approvals",
                "approved_by",
                "l1_approval_id",
                "l2_approval_id",
                "approval",
            ]
            found_appr = False
            for k in approval_keys:
                v = case_json.get(k) or (
                    case_json.get("case", {}).get(k) if isinstance(case_json.get("case"), dict) else None
                )
                if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and len(v) > 0):
                    found_appr = True
                    break

            if not found_appr:
                raise ValueError("Refused write_case: BLOCK/FILE_REPORT action requires an approval ID")

        self.append_only_store.append(case_json)
        if hasattr(self.store, "write_case"):
            try:
                card_id = case_json.get("card_id") or case_json.get("case", {}).get("card_id", "")
                self.store.write_case(case_json, card_id=card_id)
            except Exception:
                pass

        return {
            "status": "written",
            "case_id": case_id,
            "store_index": len(self.append_only_store) - 1,
            "total_rows": 1,
        }

    def handle_request(self, req: dict) -> Optional[dict]:
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            return self.make_error(req.get("id") if isinstance(req, dict) else None, -32600, "Invalid Request")

        req_id = req.get("id")
        method = req.get("method")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "assay-mcp", "version": "1.0.0"},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.get_tool_schemas()}}

        if method == "tools/call":
            params = req.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                res = self.call_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(res, default=str),
                            }
                        ]
                    },
                }
            except ValueError as e:
                return self.make_error(req_id, -32602, str(e))
            except Exception as e:
                return self.make_error(req_id, -32603, f"Internal error: {str(e)}")

        return self.make_error(req_id, -32601, f"Method not found: {method}")

    @staticmethod
    def make_error(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def run_stdio_server(store: Optional[Any] = None) -> None:
    server = MCPServer(store=store)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Content-Length:"):
            length = int(line.split(":")[1].strip())
            sys.stdin.readline()
            payload = sys.stdin.read(length)
        else:
            payload = line

        try:
            req = json.loads(payload)
        except Exception as e:
            resp = MCPServer.make_error(None, -32700, f"Parse error: {str(e)}")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = server.handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
