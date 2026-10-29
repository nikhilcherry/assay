"""Fail-closed execution harness for card-fraud investigation drills (BLOCK B9).

Implements ToolProxy, FaultInjector, run_case, write_answer, idempotency_key, and RunLedger.
Ensures typed errors on dependency failures, canonical JSON byte-identical output across runs,
and strict rejection of invalid or hostile inputs with no silent fallback.
Includes a CLI drill runner (`python -m assay.r2.harness drill`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# --- Exception Hierarchy ---------------------------------------------------


class HarnessError(Exception):
    """Base exception for harness failures."""


class DependencyDown(HarnessError):
    """Raised when an underlying tool or graph dependency is down."""


class ToolTimeout(HarnessError):
    """Raised when a tool call exceeds its allocated execution timeout."""


class GarbageResponse(HarnessError):
    """Raised when a tool response fails shape or schema validation."""


class InvalidInput(HarnessError):
    """Raised when input case data contains invalid, hostile, or malformed data."""


# --- Data Structures -------------------------------------------------------


@dataclass
class CallRecord:
    name: str
    args: dict
    started: float
    ms: float
    ok: bool
    error_type: str | None = None


@dataclass
class RunResult:
    status: str  # COMPLETE | INCOMPLETE | REJECTED_INPUT
    answer: dict | None
    error: str | None
    calls: list[CallRecord]
    error_type: str | None = None
    failed_call: CallRecord | None = None


# --- Fault Injector --------------------------------------------------------


class FaultInjector:
    """Context manager and env var parser for fault injection drills.

    Fault specification format: "method:kind@n"
    where kind in {"down", "timeout", "garbage", "slow"} and n is the 1-indexed call number.
    Multiple specifications can be comma-separated.
    """

    _override_faults: dict[tuple[str, int], str] | None = None

    def __init__(self, fault_spec: str):
        self.fault_spec = fault_spec
        self._prev: dict[tuple[str, int], str] | None = None

    def __enter__(self):
        self._prev = FaultInjector._override_faults
        FaultInjector._override_faults = FaultInjector.parse_spec(self.fault_spec)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        FaultInjector._override_faults = self._prev

    @classmethod
    def parse_spec(cls, spec: str) -> dict[tuple[str, int], str]:
        faults: dict[tuple[str, int], str] = {}
        if not spec:
            return faults
        for token in re.split(r"[,;]", spec):
            token = token.strip()
            if not token or ":" not in token:
                continue
            method, rest = token.split(":", 1)
            if "@" in rest:
                kind, n_str = rest.split("@", 1)
                try:
                    n = int(n_str)
                except ValueError:
                    n = 1
            else:
                kind = rest
                n = 1
            faults[(method.strip(), n)] = kind.strip()
        return faults

    @classmethod
    def get_fault(cls, method_name: str, call_number: int) -> str | None:
        if cls._override_faults is not None:
            if (method_name, call_number) in cls._override_faults:
                return cls._override_faults[(method_name, call_number)]
        env_spec = os.environ.get("ASSAY_FAULT", "")
        if env_spec:
            parsed = cls.parse_spec(env_spec)
            if (method_name, call_number) in parsed:
                return parsed[(method_name, call_number)]
        return None


# --- Tool Proxy ------------------------------------------------------------


class ToolProxy:
    """Wraps every public method of a Store object with timeout, validation, and call tracking."""

    def __init__(
        self,
        store: Any,
        timeout_s: float = 5.0,
        validators: dict[str, Callable[[Any], bool]] | None = None,
    ):
        self.store = store
        self.timeout_s = float(timeout_s)
        self.validators = validators or {}
        self.calls: list[CallRecord] = []
        self._counts: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"ToolProxy has no attribute '{name}'")
        attr = getattr(self.store, name)
        if not callable(attr):
            return attr

        def _wrapper(*args, **kwargs):
            return self._call_method(name, attr, *args, **kwargs)

        return _wrapper

    def _call_method(self, name: str, method_fn: Callable, *args, **kwargs) -> Any:
        self._counts[name] = self._counts.get(name, 0) + 1
        n = self._counts[name]
        fault = FaultInjector.get_fault(name, n)

        # Bind arguments for audit call recording
        try:
            sig = inspect.signature(method_fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_args = dict(bound.arguments)
        except Exception:
            bound_args = kwargs | {f"arg_{i}": str(v) for i, v in enumerate(args)}

        started = time.time()

        def _exec():
            if fault == "down":
                raise DependencyDown(f"Injected fault: store method '{name}' is down")
            if fault == "slow":
                time.sleep(min(self.timeout_s * 0.5, 0.5))
            if fault == "timeout":
                time.sleep(self.timeout_s + 1.0)
            if fault == "garbage":
                return "__GARBAGE_RESPONSE__"
            return method_fn(*args, **kwargs)

        err_type = None
        try:
            if fault == "down":
                raise DependencyDown(f"Injected fault: store method '{name}' is down")

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_exec)
                try:
                    res = future.result(timeout=self.timeout_s)
                except concurrent.futures.TimeoutError as exc:
                    raise ToolTimeout(f"Method '{name}' timed out after {self.timeout_s}s") from exc
                except HarnessError:
                    raise
                except Exception as exc:
                    raise DependencyDown(f"Method '{name}' failed with exception: {exc}") from exc

            # Validate return value shape
            if name in self.validators:
                val_fn = self.validators[name]
                try:
                    valid = bool(val_fn(res))
                except Exception as exc:
                    raise GarbageResponse(f"Validator for '{name}' raised: {exc}") from exc
                if not valid:
                    raise GarbageResponse(f"Method '{name}' return value failed validation")
            else:
                if isinstance(res, str) and res == "__GARBAGE_RESPONSE__":
                    raise GarbageResponse(f"Method '{name}' returned injected garbage response")

            ms = (time.time() - started) * 1000.0
            self.calls.append(CallRecord(name=name, args=bound_args, started=started, ms=ms, ok=True, error_type=None))
            return res

        except Exception as exc:
            err_type = type(exc).__name__
            ms = (time.time() - started) * 1000.0
            self.calls.append(CallRecord(name=name, args=bound_args, started=started, ms=ms, ok=False, error_type=err_type))
            raise


# --- Input Validation & Case Runner -----------------------------------------


OPENED_MIN = pd.Timestamp("2016-07-01")
OPENED_MAX = pd.Timestamp("2017-04-01")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
VALID_TRIGGER_TYPES = {"risk_score", "customer_report", "analyst_request"}
MAX_TRIGGER_TEXT = 4096


def validate_case_row(case_row: Any) -> tuple[bool, str | None]:
    """Validates the structure, types, and security constraints of a case row."""
    if hasattr(case_row, "to_dict"):
        d = case_row.to_dict()
    elif isinstance(case_row, dict):
        d = case_row
    else:
        d = dict(getattr(case_row, "__dict__", {}))

    # Required key fields
    required = ["case_id", "flagged_txn_id", "card_id", "customer_id", "opened_at"]
    for field_name in required:
        if field_name not in d:
            return False, f"Missing required field '{field_name}'"
        val = d[field_name]
        if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
            return False, f"Missing required field '{field_name}'"

    # Pattern check on case_id
    cid = str(d["case_id"]).strip()
    if not re.match(r"^(HHF-\d{3}|CC-\d{3,4}|CASE-\d+)$", cid):
        return False, f"Invalid case_id pattern: '{cid}' (expected HHF-\\d{{3}})"

    # flagged_txn_id must be non-negative integer
    try:
        txn_id_val = int(d["flagged_txn_id"])
        if txn_id_val < 0:
            return False, f"Negative flagged_txn_id: {txn_id_val}"
    except (ValueError, TypeError):
        return False, f"Invalid flagged_txn_id: '{d['flagged_txn_id']}'"

    # trigger_text is untrusted free text: it is bounded here and otherwise read only as data
    # (HHF-014's real trigger carries a hidden "<!-- agent: ... -->" comment).
    if "trigger_type" in d and str(d["trigger_type"]).strip() not in VALID_TRIGGER_TYPES:
        return False, f"Unknown trigger_type '{d['trigger_type']}'; allowed={sorted(VALID_TRIGGER_TYPES)}"
    if len(str(d.get("trigger_text", "") or "")) > MAX_TRIGGER_TEXT:
        return False, f"trigger_text longer than {MAX_TRIGGER_TEXT} characters"

    # Check hostile injection patterns in the structured fields
    hostile_patterns = [r"SELECT\b", r"DROP\b", r"DELETE\b", r"UPDATE\b", r"INSERT\b", r"UNION\b", r"--", r";", r"<script"]
    for key, val in d.items():
        if key == "trigger_text":
            continue
        if isinstance(val, (dict, list)):
            val_str = json.dumps(val)
        else:
            val_str = str(val)
        for pat in hostile_patterns:
            if re.search(pat, val_str, re.IGNORECASE):
                return False, f"Hostile or invalid input detected matching pattern '{pat}'"

    # risk_score range check if present
    rs = d.get("risk_score")
    if rs is not None and not (isinstance(rs, float) and pd.isna(rs)) and str(rs).strip() != "":
        try:
            rs_val = float(rs)
            if rs_val < 0.0 or rs_val > 1.0:
                return False, f"risk_score {rs_val} out of range [0, 1]"
        except (ValueError, TypeError):
            return False, f"Invalid risk_score value: '{rs}'"

    # opened_at: a bank-local timestamp inside the data's period (red team: a "...Z" date passed the
    # parse check and crashed mid-run on a tz-aware comparison)
    opened_at = str(d["opened_at"]).strip()
    try:
        ts = pd.Timestamp(opened_at)
    except Exception:
        return False, f"Unparseable opened_at timestamp: '{opened_at}'"
    if ts.tzinfo is not None:
        return False, f"opened_at must be a local timestamp without a time zone: '{opened_at}'"
    if not (OPENED_MIN <= ts < OPENED_MAX):
        return False, f"opened_at {opened_at} is outside the data period [{OPENED_MIN.date()}, {OPENED_MAX.date()})"

    # Card and customer ids follow the bank's format, and the card belongs to the customer
    # (a different K-number of the same customer is a data-quality case, investigated as DQ-07)
    card, cust = str(d["card_id"]).strip(), str(d["customer_id"]).strip()
    if not re.fullmatch(r"C\d{5}-K\d{1,2}", card) or not re.fullmatch(r"C\d{5}", cust):
        return False, f"Malformed card_id/customer_id: '{card}' / '{cust}'"
    if card.rsplit("-K", 1)[0] != cust:
        return False, f"card_id {card} does not belong to customer {cust}"

    # Control characters have no place in a case row (tabs and newlines in free text are fine)
    for key, val in d.items():
        if isinstance(val, str) and CONTROL_CHARS.search(val):
            return False, f"Control characters in field '{key}'"

    # Check for negative amounts if present
    amt = d.get("amount")
    if amt is not None and not (isinstance(amt, float) and pd.isna(amt)) and str(amt).strip() != "":
        try:
            if float(amt) < 0.0:
                return False, f"Negative transaction amount: {amt}"
        except (ValueError, TypeError):
            return False, f"Invalid transaction amount: '{amt}'"

    return True, None


def run_case(
    case_row: Any,
    investigate_fn: Callable[[Any, ToolProxy], dict],
    store: Any,
    timeout_s: float = 5.0,
    validators: dict[str, Callable[[Any], bool]] | None = None,
) -> RunResult:
    """Executes a case investigation inside the fail-closed harness."""
    valid, err_msg = validate_case_row(case_row)
    if not valid:
        return RunResult(status="REJECTED_INPUT", answer=None, error=err_msg, calls=[], error_type="InvalidInput")

    proxy = ToolProxy(store, timeout_s=timeout_s, validators=validators)
    try:
        answer = investigate_fn(case_row, proxy)
        if not isinstance(answer, dict):
            return RunResult(
                status="INCOMPLETE",
                answer=None,
                error="investigate_fn returned non-dict answer",
                calls=proxy.calls,
                error_type="GarbageResponse",
                failed_call=proxy.calls[-1] if proxy.calls else None,
            )
        return RunResult(status="COMPLETE", answer=answer, error=None, calls=proxy.calls)
    except HarnessError as exc:
        failed = proxy.calls[-1] if proxy.calls else None
        return RunResult(
            status="INCOMPLETE",
            answer=None,
            error=str(exc),
            calls=proxy.calls,
            error_type=type(exc).__name__,
            failed_call=failed,
        )
    except Exception as exc:
        failed = proxy.calls[-1] if proxy.calls else None
        return RunResult(
            status="INCOMPLETE",
            answer=None,
            error=f"{type(exc).__name__}: {str(exc)}",
            calls=proxy.calls,
            error_type=type(exc).__name__,
            failed_call=failed,
        )


# --- Serialization & Output Writing -----------------------------------------


def canonicalize(obj: Any) -> Any:
    """Recursively formats dicts, lists, and floats into canonical form for byte-equality."""
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: canonicalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        c_list = [canonicalize(x) for x in obj]
        if all(isinstance(x, (int, float, str)) for x in c_list):
            return sorted(c_list, key=lambda x: str(x))
        return c_list
    return obj


def write_answer(result: RunResult, out_dir: str | Path, case_id: str | None = None) -> str:
    """Writes the investigation answer or incomplete marker to disk atomically."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cid = case_id
    if cid is None and result.answer and isinstance(result.answer, dict):
        cid = result.answer.get("case_id")
    if cid is None:
        cid = "unknown_case"

    if result.status == "COMPLETE" and result.answer is not None:
        c_ans = canonicalize(result.answer)
        json_str = json.dumps(c_ans, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        data_bytes = json_str.encode("utf-8")
        target_file = out_path / f"{cid}.json"
        tmp_file = out_path / f"{cid}.json.tmp.{uuid.uuid4().hex}"
        try:
            tmp_file.write_bytes(data_bytes)
            os.replace(tmp_file, target_file)
        finally:
            tmp_file.unlink(missing_ok=True)
        return hashlib.sha256(data_bytes).hexdigest()
    else:
        # Status is INCOMPLETE or REJECTED_INPUT
        payload = {
            "case_id": cid,
            "status": result.status,
            "error": str(result.error) if result.error else "Unknown error",
            "error_type": result.error_type,
        }
        json_str = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        data_bytes = json_str.encode("utf-8")
        target_file = out_path / f"{cid}.incomplete.json"
        tmp_file = out_path / f"{cid}.incomplete.json.tmp.{uuid.uuid4().hex}"
        try:
            tmp_file.write_bytes(data_bytes)
            os.replace(tmp_file, target_file)
        finally:
            tmp_file.unlink(missing_ok=True)
        return hashlib.sha256(data_bytes).hexdigest()


# --- Idempotency & Run Ledger -----------------------------------------------


def idempotency_key(case_row: Any, dataset_tag: str, code_version: str) -> str:
    """Computes a deterministic hash key for a case execution configuration."""
    if hasattr(case_row, "to_dict"):
        d = case_row.to_dict()
    elif isinstance(case_row, dict):
        d = case_row
    else:
        d = dict(getattr(case_row, "__dict__", {}))

    cid = str(d.get("case_id", ""))
    opened_at = str(d.get("opened_at", ""))
    flagged_txn_id = str(d.get("flagged_txn_id", ""))
    card_id = str(d.get("card_id", ""))
    customer_id = str(d.get("customer_id", ""))

    raw = f"{cid}:{opened_at}:{flagged_txn_id}:{card_id}:{customer_id}:{dataset_tag}:{code_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RunLedger:
    """Ledger recording completed run results and content hashes by idempotency key."""

    def __init__(self, ledger_dir: str | Path | None = None):
        self.ledger_dir = Path(ledger_dir) if ledger_dir else None
        if self.ledger_dir:
            self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, tuple[RunResult, str, bytes]] = {}

    def get(self, key: str) -> tuple[RunResult, str, bytes] | None:
        if key in self._memory:
            return self._memory[key]
        if self.ledger_dir:
            f = self.ledger_dir / f"{key}.json"
            if f.exists():
                data = json.loads(f.read_text(encoding="utf-8"))
                calls = [CallRecord(**c) for c in data.get("calls", [])]
                res = RunResult(
                    status=data["status"],
                    answer=data["answer"],
                    error=data["error"],
                    calls=calls,
                    error_type=data.get("error_type"),
                )
                sha = data["sha256"]
                content = data["content"].encode("utf-8")
                self._memory[key] = (res, sha, content)
                return res, sha, content
        return None

    def record(self, key: str, result: RunResult, sha256: str, content: bytes) -> None:
        self._memory[key] = (result, sha256, content)
        if self.ledger_dir:
            f = self.ledger_dir / f"{key}.json"
            tmp = self.ledger_dir / f"{key}.json.tmp.{uuid.uuid4().hex}"
            payload = {
                "key": key,
                "status": result.status,
                "answer": result.answer,
                "error": result.error,
                "error_type": result.error_type,
                "calls": [asdict(c) for c in result.calls],
                "sha256": sha256,
                "content": content.decode("utf-8"),
            }
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, f)


# --- CLI Drill Runner -------------------------------------------------------


def _resolve_store() -> Any:
    try:
        from assay.r2.store_r2 import R2LocalStore
        return R2LocalStore()
    except Exception:
        from assay.store import LocalStore
        return LocalStore()


def _resolve_investigate(target: str) -> Callable[[Any, Any], dict]:
    mod_name, attr_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr_name)

    if inspect.isclass(obj):
        try:
            from assay.r2.store_r2 import r2_pattern_model
            model = r2_pattern_model()
            inst = obj(model=model)
        except Exception:
            inst = obj()
        if hasattr(inst, "investigate"):
            return lambda case_row, store: inst.investigate(case_row, store)
        return lambda case_row, store: inst(case_row, store)

    return obj


def _resolve_case_row(case_arg: str) -> dict:
    if Path(case_arg).exists():
        text = Path(case_arg).read_text(encoding="utf-8")
        return json.loads(text)

    try:
        from assay.r2.frames import case_pack
        cp = case_pack()
        hit = cp[cp.case_id == case_arg]
        if len(hit) > 0:
            return hit.iloc[0].to_dict()
    except Exception:
        pass

    try:
        return json.loads(case_arg)
    except Exception:
        raise ValueError(f"Could not resolve case '{case_arg}' as ID, file path, or JSON.")


def main():
    parser = argparse.ArgumentParser(description="Assay R2 Fail-closed Harness Drill Runner")
    subparsers = parser.add_subparsers(dest="command")

    drill_p = subparsers.add_parser("drill", help="Run a drill with fault injection")
    drill_p.add_argument("--case", required=True, help="Case ID (e.g. HHF-007) or JSON file path")
    drill_p.add_argument("--fault", default="", help="Fault spec (e.g. get_transaction:down@1)")
    drill_p.add_argument(
        "--investigate",
        default="assay.investigate:Investigator",
        help="Callable path 'module:function' or 'module:Class'",
    )
    drill_p.add_argument("--timeout", type=float, default=5.0, help="Tool timeout in seconds")

    args = parser.parse_args()

    if args.command == "drill":
        case_row = _resolve_case_row(args.case)
        store = _resolve_store()
        investigate_fn = _resolve_investigate(args.investigate)

        if args.fault:
            with FaultInjector(args.fault):
                res = run_case(case_row, investigate_fn, store, timeout_s=args.timeout)
        else:
            res = run_case(case_row, investigate_fn, store, timeout_s=args.timeout)

        res_dict = {
            "status": res.status,
            "answer": res.answer,
            "error": res.error,
            "error_type": res.error_type,
            "failed_call": asdict(res.failed_call) if res.failed_call else None,
            "calls": [asdict(c) for c in res.calls],
        }
        print(json.dumps(canonicalize(res_dict), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
