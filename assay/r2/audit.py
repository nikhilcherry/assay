"""BLOCK B11 -- Audit trail & Data protection module for card-fraud investigations.

Provides:
  - AuditLog: Append-only, hash-chained JSONL audit trail recording tool calls,
    findings, decisions, errors, and run lifecycles. Supports tamper detection via verify().
  - replay: Reconstruction of agent steps and decision deltas for regulator/analyst audit.
  - redact: Redaction engine masking API keys, bearer tokens, authorization headers,
    Indian phone numbers, Aadhaar numbers, email addresses, and long free text while
    preserving operational pseudonymous bank IDs.
  - egress_register: Statement of external service dependencies on the decision path.
  - scan_for_secrets: Repository scanner checking tracked and untracked files for secrets.
  - CLI interface (`python -m assay.r2.audit verify` and `replay`).

Pseudonymous Bank Identifier Preservation Note:
Pseudonymous operational identifiers (card_id, customer_id, TransactionID/txn_id,
case_id, uid, flagged_txn_id) are explicitly preserved during redaction because
audit log verification, graph traversal, and regulatory replay depend on deterministic
cross-referencing across investigation steps. These identifiers contain no personal
identifiable information (PII) or secrets.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_KEY_KEYWORDS = {"key", "token", "secret", "password"}

BANK_ID_KEYS = {
    "card_id",
    "customer_id",
    "txn_id",
    "transactionid",
    "flagged_txn_id",
    "case_id",
    "uid",
    "connected_card_ids",
    "txn_ids",
    "first_suspicious_txn_id",
    "first_fraud_txn_id",
}


def _is_secret_key(key: str) -> bool:
    """Checks whether a dictionary key indicates a secret field."""
    k_lower = key.lower()
    if k_lower in BANK_ID_KEYS:
        return False
    return any(kw in k_lower for kw in SECRET_KEY_KEYWORDS)


def _redact_str(s: str) -> str:
    """Masks secret patterns, bearer tokens, headers, phones, Aadhaar, emails, and truncates long text."""
    # 1. Anything after Authorization:
    s = re.sub(r"(?i)Authorization:\s*.*", "Authorization: [REDACTED_AUTH]", s)

    # 2. API keys & bearer tokens
    s = re.sub(r"AIzaSy[A-Za-z0-9_-]{33}", "[REDACTED_API_KEY]", s)
    s = re.sub(r"AQ\.Ab8[A-Za-z0-9_\-]{10,}", "[REDACTED_TOKEN]", s)
    s = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.=]{10,}", "Bearer [REDACTED_TOKEN]", s)
    s = re.sub(r"(?:sk|ghp|xoxb|xoxp|secret)-[A-Za-z0-9_-]{16,}", "[REDACTED_KEY]", s)

    # 3. Email addresses
    s = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", s)

    # 4. Indian phone numbers (+91 optional, 10-digits starting with 6-9)
    s = re.sub(r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b", "[REDACTED_PHONE]", s)

    # 5. Aadhaar-like 12-digit numbers (starting with 2-9)
    s = re.sub(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED_AADHAAR]", s)

    # 6. Free text length over 200 characters (truncate + sha256 digest)
    if len(s) > 200:
        digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
        s = s[:200] + f"... [TRUNCATED len={len(s)} sha256={digest}]"

    return s


def redact(obj: Any) -> Any:
    """Redacts secrets, emails, phones, Aadhaar, headers, and long text from data structures.

    Keeps pseudonymous bank ids (card_id, customer_id, txn_id, case_id, uid)
    intact as required for audit trail continuity and decision replay.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in BANK_ID_KEYS:
                new_dict[k] = v
            elif isinstance(k, str) and _is_secret_key(k):
                new_dict[k] = "[REDACTED_SECRET]"
            else:
                new_dict[k] = redact(v)
        return new_dict
    elif isinstance(obj, list):
        return [redact(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(redact(item) for item in obj)
    elif isinstance(obj, str):
        return _redact_str(obj)
    else:
        return obj


def compute_line_hash(record: dict) -> str:
    """Computes SHA256 hex digest of a log record over canonical JSON, excluding 'hash'."""
    clean = {k: v for k, v in record.items() if k != "hash"}
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only, hash-chained JSONL audit log for fraud agent investigations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev_hash = "0" * 64

        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    try:
                        last_rec = json.loads(lines[-1])
                        self.prev_hash = last_rec.get("hash", "0" * 64)
                    except Exception:
                        pass

    def log_event(self, event_type: str, case_id: str, payload: dict) -> dict:
        """Appends a hash-chained, redacted event record to the log file."""
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        redacted_payload = redact(payload)

        event_dict = {
            "event_type": event_type,
            "case_id": case_id,
            "ts": ts,
            "prev_hash": self.prev_hash,
            **redacted_payload,
        }

        h = compute_line_hash(event_dict)
        event_dict["hash"] = h

        canonical_line = json.dumps(event_dict, sort_keys=True, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical_line + "\n")

        self.prev_hash = h
        return event_dict

    def log_run_start(self, case_id: str, metadata: dict | None = None) -> dict:
        return self.log_event("run_start", case_id, {"metadata": metadata or {}})

    def log_tool_call(
        self,
        case_id: str,
        name: str,
        params: dict,
        result: Any = None,
        row_count: int = 0,
        ms: float = 0.0,
    ) -> dict:
        redacted_res = redact(result)
        res_canonical = json.dumps(redacted_res, sort_keys=True, ensure_ascii=False)
        res_digest = hashlib.sha256(res_canonical.encode("utf-8")).hexdigest()

        return self.log_event(
            "tool_call",
            case_id,
            {
                "name": name,
                "params": params,
                "result_digest": res_digest,
                "row_count": row_count,
                "ms": ms,
            },
        )

    def log_finding(self, case_id: str, finding_dict: dict) -> dict:
        return self.log_event("finding", case_id, {"finding": finding_dict})

    def log_decision(
        self, case_id: str, decision_dict: dict, delta: dict | None = None
    ) -> dict:
        return self.log_event(
            "decision",
            case_id,
            {"decision": decision_dict, "delta": delta or {}},
        )

    def log_error(self, case_id: str, error: str, details: dict | None = None) -> dict:
        return self.log_event("error", case_id, {"error": error, "details": details or {}})

    def log_run_end(
        self, case_id: str, status: str = "success", metadata: dict | None = None
    ) -> dict:
        return self.log_event(
            "run_end",
            case_id,
            {"status": status, "metadata": metadata or {}},
        )

    def verify(self) -> tuple[bool, int]:
        """Verifies integrity of the hash-chained JSONL log file.

        Returns (True, -1) if unbroken and valid.
        Returns (False, first_bad_line_index) if modified, reordered, deleted, or invalid.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True, -1

        expected_prev_hash = "0" * 64

        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except Exception:
                    return False, idx

                if (
                    "prev_hash" not in record
                    or "hash" not in record
                    or "event_type" not in record
                ):
                    return False, idx

                if record["prev_hash"] != expected_prev_hash:
                    return False, idx

                calc_hash = compute_line_hash(record)
                if calc_hash != record["hash"]:
                    return False, idx

                expected_prev_hash = record["hash"]

        return True, -1


def replay(path: str | Path, case_id: str) -> list[dict]:
    """Replays audit trail events for case_id in chronological order for regulatory audit.

    Returns ordered list of step dicts detailing agent inputs, tool execution,
    digest hashes, findings, and decision deltas.
    """
    path = Path(path)
    if not path.exists():
        return []

    steps = []
    step_num = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if rec.get("case_id") != case_id:
                continue

            evt = rec.get("event_type")
            step = {
                "step": step_num,
                "event_type": evt,
                "ts": rec.get("ts"),
            }

            if evt == "tool_call":
                step.update(
                    {
                        "tool": rec.get("name"),
                        "params": rec.get("params"),
                        "result_digest": rec.get("result_digest"),
                        "row_count": rec.get("row_count"),
                        "ms": rec.get("ms"),
                    }
                )
            elif evt == "finding":
                step["finding"] = rec.get("finding")
            elif evt == "decision":
                step["decision"] = rec.get("decision")
                step["delta"] = rec.get("delta")
            elif evt == "error":
                step["error"] = rec.get("error")
                step["details"] = rec.get("details")
            elif evt in ("run_start", "run_end"):
                step["metadata"] = rec.get("metadata")
                step["status"] = rec.get("status")

            steps.append(step)
            step_num += 1

    return steps


def egress_register() -> list[dict]:
    """Returns declared list of external services used on the decision path."""
    return [
        {
            "service": "none",
            "decision_path": False,
            "description": "All transaction investigation, graph traversal, and fraud evaluation run strictly locally within the bank system boundary.",
        }
    ]


def scan_for_secrets(root: str | Path, stats: dict | None = None) -> list[dict]:
    """Scans tracked and untracked-but-not-ignored git files under root for secret patterns, excluding data/."""
    root = Path(root).resolve()
    files_set = set()
    try:
        res1 = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        for f in res1.stdout.splitlines():
            if f.strip():
                files_set.add(root / f.strip())

        res2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        for f in res2.stdout.splitlines():
            if f.strip():
                files_set.add(root / f.strip())

        files = sorted(list(files_set))
    except Exception:
        files = sorted([p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts])

    # Ignore the top-level (gitignored) data/ directory
    files = [f for f in files if f.relative_to(root).parts[0] != "data"]

    if stats is not None:
        stats["files_scanned"] = len(files)
    print(f"Scanned {len(files)} files for secrets.")

    patterns = [
        ("AIza_api_key", re.compile(r"AIzaSy[A-Za-z0-9_-]{33}")),
        ("openai_anthropic_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
        ("github_token", re.compile(r"ghp_[A-Za-z0-9]{36}")),
        ("private_key", re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|PRIVATE) KEY-----")),
        ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
        (
            "hardcoded_secret_assign",
            re.compile(
                r"(?i)(api_key|secret_key|auth_token|password)\s*=\s*['\"](?!\[REDACTED)[A-Za-z0-9_\-]{12,}['\"]"
            ),
        ),
    ]

    ignored_exts = {
        ".parquet",
        ".pyc",
        ".png",
        ".jpg",
        ".jpeg",
        ".zip",
        ".tar",
        ".gz",
        ".db",
    }

    findings = []
    for filepath in files:
        if filepath.suffix.lower() in ignored_exts or not filepath.exists():
            continue

        try:
            rel_path = str(filepath.relative_to(root))
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(content.splitlines(), start=1):
                if "re.compile" in line or "pattern" in line or "regex" in line:
                    continue
                for pname, pat in patterns:
                    match = pat.search(line)
                    if match:
                        findings.append(
                            {
                                "file": rel_path,
                                "line_no": line_no,
                                "pattern": pname,
                                "match": line.strip()[:100],
                            }
                        )
        except Exception:
            continue

    return findings


def main():
    """CLI runner for verify and replay commands."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m assay.r2.audit verify <log_path>")
        print("  python -m assay.r2.audit replay <log_path> <case_id>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "verify":
        if len(sys.argv) < 3:
            print("Error: missing log path argument")
            sys.exit(1)
        log_path = sys.argv[2]
        log = AuditLog(log_path)
        ok, bad_line = log.verify()
        if ok:
            print(f"Audit log {log_path} is VALID.")
            sys.exit(0)
        else:
            print(f"Audit log {log_path} is INVALID at line {bad_line}.")
            sys.exit(1)

    elif cmd == "replay":
        if len(sys.argv) < 4:
            print("Error: missing log path or case_id argument")
            sys.exit(1)
        log_path = sys.argv[2]
        case_id = sys.argv[3]
        steps = replay(log_path, case_id)

        print(
            f"{'STEP':<5} {'EVENT/TOOL':<22} {'PARAMS':<30} {'ROWS':<6} {'DIGEST':<14} {'DELTA':<20}"
        )
        print("-" * 100)
        for s in steps:
            step_no = s.get("step", 0)
            evt_tool = s.get("tool") or s.get("event_type", "")
            if s.get("event_type") == "decision":
                act = (s.get("decision") or {}).get("action")
                if act:
                    evt_tool = f"decision:{act}"
            params_str = str(s.get("params", "")) if s.get("params") is not None else ""
            if len(params_str) > 28:
                params_str = params_str[:25] + "..."
            rows_str = str(s.get("row_count")) if s.get("row_count") is not None else "-"
            digest_str = (
                (s.get("result_digest") or "")[:12] if s.get("result_digest") else "-"
            )
            delta_str = str(s.get("delta")) if s.get("delta") else "-"
            if len(delta_str) > 18:
                delta_str = delta_str[:15] + "..."
            print(
                f"{step_no:<5} {evt_tool:<22} {params_str:<30} {rows_str:<6} {digest_str:<14} {delta_str:<20}"
            )


if __name__ == "__main__":
    main()
