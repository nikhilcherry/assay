"""BLOCK C9 -- Unknown-fault chaos suite (Stage 2 live resilience drills).

Runs the real single-case command `python -m assay.r2.agent <CASE> --out <tmpdir>`
as a SUBPROCESS under faults injected WITHOUT editing agent code:
  1. Environment variables honoured by the codebase (`ASSAY_FAULT`, `ASSAY_TG`,
     `ASSAY_TG_HOST`, `ASSAY_TG_PORT`).
  2. A `sitecustomize.py` placed on `PYTHONPATH` in a temporary directory ONLY for
     `sc_*` scenarios (`os.replace` raising `ENOSPC`, `time.time` jumping backwards,
     slow store calls exceeding `--timeout`, or nth TigerGraph REST HTTP query
     raising / returning `{}` / `[]` / malformed JSON via `urllib.request.urlopen`).
     Every `sitecustomize` fault writes a proof-of-injection marker file when it
     fires; if the marker is missing after the run, the scenario is reported as
     `INVALID` (harness did not fire) with full stderr captured for diagnosis.
  3. Filesystem conditions (read-only `--out` dir, `--out` dir is a regular file,
     pre-existing stale `<case>.json` or leftover `.tmp` file before a run).
  4. Process conditions (`SIGTERM` / `SIGKILL` mid-run via a timer against the
     clean agent without `sitecustomize`, two concurrent subprocesses running the
     same case into the same `--out` dir).
  5. Hostile / weird input conditions (homoglyph / zero-width case id, path
     traversal in case id, CRLF in fields, 1 MB `trigger_text`, `NaN`/`Infinity`
     `risk_score`, duplicate conflicting case runs / rows).

For each scenario records:
  - exit code
  - stdout / stderr first meaningful line
  - which files exist afterwards
  - whether invariants (a)..(e) hold
  - an `assay.r2.attest.Attestation` with exact ids and a re-runnable query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from assay.data import ROOT
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import case_pack
from assay.validate import validate as validate_answer_schema

DEFAULT_DOC_PATH = ROOT / "docs" / "r2" / "C9_chaos.md"
SUBPROCESS_TIMEOUT_S = 90.0


def is_tg_reachable(host: str = "127.0.0.1", port: int = 14240, timeout_s: float = 1.0) -> bool:
    """Checks whether TigerGraph RESTPP is reachable at `host:port`."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first_meaningful_line(text: str) -> str:
    """Returns the first non-empty line, preferring agent status/error banners over pandas warnings."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in lines:
        if any(
            tag in ln
            for tag in (
                "FAIL CLOSED",
                "ERROR [EXIT",
                "status=COMPLETE",
                "Error in sitecustomize",
                "Traceback (most recent call last)",
            )
        ):
            return ln[:220]
    for ln in lines:
        if "PerformanceWarning" not in ln and "frame.insert" not in ln and not ln.startswith("[CPC]"):
            return ln[:220]
    return lines[0][:220]


def _only_traceback(stdout: str, stderr: str) -> bool:
    """Returns True if output is non-empty and consists solely of an unhandled Python traceback
    without a structured one-line error summary (`FAIL CLOSED` or `ERROR [EXIT`)."""
    combined = ((stdout or "") + "\n" + (stderr or "")).strip()
    if not combined:
        return False
    has_tb = "Traceback (most recent call last):" in combined or "Error in sitecustomize" in combined
    if not has_tb:
        return False
    has_clean_banner = ("FAIL CLOSED" in combined) or ("ERROR [EXIT" in combined)
    return not has_clean_banner


def _list_out_files(out_dir: Path) -> list[str]:
    if not out_dir.exists() or not out_dir.is_dir():
        return []
    items: list[str] = []
    for p in sorted(out_dir.iterdir(), key=lambda x: x.name):
        if p.name == ".chaos_fired.marker":
            continue
        if p.is_dir():
            sub = sorted(c.name for c in p.iterdir())
            items.append(f"{p.name}/({len(sub)})")
        else:
            items.append(p.name)
    return items


@dataclass
class InvariantCheck:
    inv_a_exit0_valid_json_only: bool
    inv_b_nonzero_no_fresh_json_and_clear_err: bool
    inv_c_never_only_traceback: bool
    inv_d_concurrent_consistent: bool
    inv_e_never_exit0_missing_or_partial: bool

    @property
    def all_hold(self) -> bool:
        return (
            self.inv_a_exit0_valid_json_only
            and self.inv_b_nonzero_no_fresh_json_and_clear_err
            and self.inv_c_never_only_traceback
            and self.inv_d_concurrent_consistent
            and self.inv_e_never_exit0_missing_or_partial
        )

    def format_short(self) -> str:
        def _flag(b: bool) -> str:
            return "ok" if b else "FAIL"

        return (
            f"a={_flag(self.inv_a_exit0_valid_json_only)} "
            f"b={_flag(self.inv_b_nonzero_no_fresh_json_and_clear_err)} "
            f"c={_flag(self.inv_c_never_only_traceback)} "
            f"d={_flag(self.inv_d_concurrent_consistent)} "
            f"e={_flag(self.inv_e_never_exit0_missing_or_partial)}"
        )


@dataclass
class ScenarioResult:
    scenario: str
    category: str
    case_id: str
    exit_code: int | str
    first_line: str
    files_after: list[str]
    invariants: InvariantCheck
    status: str  # PASS | FAIL | SKIPPED | INVALID
    notes: str = ""
    elapsed_s: float = 0.0
    attestation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosReport:
    cases_tested: list[str]
    tg_reachable: bool
    results: list[ScenarioResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "SKIPPED")

    @property
    def invalid(self) -> int:
        return sum(1 for r in self.results if r.status == "INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_tested": self.cases_tested,
            "tg_reachable": self.tg_reachable,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "invalid": self.invalid,
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "<!-- generated; do not edit -->",
            "# C9 — Unknown-Fault Chaos Suite Results",
            "",
            f"- **Cases tested:** `{', '.join(self.cases_tested)}`",
            f"- **TigerGraph reachable at 127.0.0.1:14240:** `{self.tg_reachable}`",
            f"- **Total scenarios:** {self.total} ({self.passed} PASS, {self.failed} FAIL, {self.skipped} SKIPPED, {self.invalid} INVALID)",
            "",
            "## Invariant Legend",
            "- **(a)** `exit == 0` => complete valid `<case>.json` and no leftover `.tmp` or `.incomplete.json` artifacts.",
            "- **(b)** `exit != 0` => NO `<case>.json` (or pre-existing stale `<case>.json` removed / never updated as fresh) and a clear one-line error.",
            "- **(c)** Never a Python traceback as the only output.",
            "- **(d)** Two concurrent runs of the same case => identical `<case>.json` bytes or one clean refusal.",
            "- **(e)** Never `exit == 0` with a missing, empty, or partial `<case>.json`.",
            "",
            "## Chaos Battery Table",
            "",
            "| scenario | category | case | exit | invariant results | files after | first line | PASS/FAIL |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in self.results:
            inv_str = r.status if r.status in ("SKIPPED", "INVALID") else r.invariants.format_short()
            files_str = ", ".join(r.files_after) if r.files_after else "(none)"
            if len(files_str) > 65:
                files_str = files_str[:62] + "..."
            fl = (r.first_line or r.notes or "-").replace("|", "\\|").replace("\n", " ")
            if len(fl) > 95:
                fl = fl[:92] + "..."
            lines.append(
                f"| `{r.scenario}` | {r.category} | `{r.case_id}` | `{r.exit_code}` | `{inv_str}` | `{files_str}` | {fl} | **{r.status}** |"
            )

        failures = [r for r in self.results if r.status in ("FAIL", "INVALID")]
        if failures:
            lines.extend(["", "## Diagnostic Findings (Failed / Invalid Scenarios)", ""])
            for f in failures:
                lines.append(
                    f"- **`{f.scenario}`** (`{f.case_id}`, status `{f.status}`, exit `{f.exit_code}`): "
                    f"invariants `{f.invariants.format_short()}` — first_line=`{f.first_line}` notes=`{f.notes}`"
                )
        lines.append("")
        return "\n".join(lines)


def _evaluate_invariants(
    exit_code: int,
    stdout: str,
    stderr: str,
    out_dir: Path,
    expected_case_id: str,
    stale_before: tuple[float, str] | None = None,
    killed_by_sigkill: bool = False,
    concurrent_check: bool = True,
) -> tuple[InvariantCheck, str]:
    """Evaluates invariants (a)..(e) on a completed subprocess run."""
    notes: list[str] = []
    ans_path = out_dir / f"{expected_case_id}.json" if out_dir.is_dir() else None
    ans_exists = bool(ans_path and ans_path.exists() and ans_path.is_file())

    ans_valid = False
    if ans_exists and ans_path is not None:
        try:
            raw_bytes = ans_path.read_bytes()
            if len(raw_bytes) > 0:
                parsed = json.loads(raw_bytes.decode("utf-8"))
                val_errs = validate_answer_schema(parsed)
                if isinstance(parsed, dict) and parsed.get("case_id") == expected_case_id and not val_errs:
                    ans_valid = True
                else:
                    notes.append(f"schema_errors={val_errs}")
        except Exception as exc:
            notes.append(f"json_parse_err={exc}")

    leftover_bad: list[str] = []
    if out_dir.exists() and out_dir.is_dir():
        for p in out_dir.iterdir():
            name = p.name
            if name == ".chaos_fired.marker":
                continue
            if ".tmp." in name or name.endswith(".incomplete.json"):
                leftover_bad.append(name)
            elif name.endswith(".json") and name != f"{expected_case_id}.json":
                leftover_bad.append(name)

    if exit_code == 0:
        inv_a = ans_exists and ans_valid and len(leftover_bad) == 0
        if not inv_a:
            notes.append(f"inv_a: exists={ans_exists} valid={ans_valid} leftover={leftover_bad}")
    else:
        inv_a = True

    if exit_code == 0:
        inv_e = ans_exists and ans_valid
    else:
        inv_e = True

    if exit_code != 0:
        first_line = _first_meaningful_line(stderr) or _first_meaningful_line(stdout)
        has_clear_err = (
            bool(first_line)
            and not first_line.startswith("Traceback")
            and "Error in sitecustomize" not in first_line
        )
        # An OS SIGKILL (-9 / 137) kills the process immediately in kernel space so it cannot print
        if killed_by_sigkill and exit_code in (-9, 137):
            has_clear_err = True
        if stale_before is not None:
            if ans_exists and ans_path is not None:
                after_mtime = ans_path.stat().st_mtime
                after_sha = _sha256_bytes(ans_path.read_bytes())
                notes.append(
                    f"stale_file_remained(mtime_changed={after_mtime != stale_before[0]}, sha_changed={after_sha != stale_before[1]})"
                )
                inv_b = False
            else:
                inv_b = has_clear_err
        else:
            inv_b = (not ans_exists) and has_clear_err
            if ans_exists:
                notes.append("inv_b: <case>.json exists on non-zero exit")
            if not has_clear_err:
                notes.append("inv_b: missing clear one-line error")
    else:
        inv_b = True

    inv_c = not _only_traceback(stdout, stderr)
    if not inv_c:
        notes.append("inv_c: traceback was the only output")

    inv_d = concurrent_check

    return (
        InvariantCheck(
            inv_a_exit0_valid_json_only=inv_a,
            inv_b_nonzero_no_fresh_json_and_clear_err=inv_b,
            inv_c_never_only_traceback=inv_c,
            inv_d_concurrent_consistent=inv_d,
            inv_e_never_exit0_missing_or_partial=inv_e,
        ),
        "; ".join(notes),
    )


def _make_attestation(
    scenario: str,
    case_id: str,
    exit_code: int | str,
    status: str,
    txn_id: int | None = None,
    card_id: str | None = None,
    customer_id: str | None = None,
) -> dict[str, Any]:
    ev = Evidence(
        txn_ids=[int(txn_id)] if txn_id else [],
        card_ids=[str(card_id)] if card_id else [],
        customer_ids=[str(customer_id)] if customer_id else [],
        case_ids=[str(case_id)] if case_id else [],
    )
    return Attestation(
        what=f"Chaos scenario '{scenario}' on {case_id} exited with {exit_code} and invariant check={status}.",
        evidence=ev,
        query="assay.r2.chaos.run_single_scenario",
        params={"scenario": scenario, "case_id": case_id},
    ).to_dict()


def _run_agent_subproc(
    case_arg: str,
    out_dir: Path,
    extra_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
    sitecustomize_code: str | None = None,
    signal_spec: tuple[int, float] | None = None,
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str, str, float]:
    """Runs `python -m assay.r2.agent <case_arg> --out <out_dir>` in a child subprocess.

    Only places a `sitecustomize.py` directory on `PYTHONPATH` when `sitecustomize_code` is provided.
    All other scenarios run the clean agent with no `sitecustomize` on `PYTHONPATH`.
    """
    env = os.environ.copy()
    env.setdefault("ASSAY_TG", "0")
    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)

    sc_temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if sitecustomize_code is not None:
        sc_temp_dir = tempfile.TemporaryDirectory(prefix="assay_sc_")
        sc_path = Path(sc_temp_dir.name) / "sitecustomize.py"
        sc_path.write_text(sitecustomize_code, encoding="utf-8")
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{sc_temp_dir.name}:{ROOT}" + (f":{existing_pp}" if existing_pp else "")
    else:
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{ROOT}" + (f":{existing_pp}" if existing_pp else "")

    cmd = [sys.executable, "-m", "assay.r2.agent", case_arg, "--out", str(out_dir)]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        timer: threading.Timer | None = None
        if signal_spec is not None:
            sig_num, delay_s = signal_spec

            def _send_sig() -> None:
                if proc.poll() is None:
                    try:
                        proc.send_signal(sig_num)
                    except OSError:
                        pass

            timer = threading.Timer(delay_s, _send_sig)
            timer.daemon = True
            timer.start()

        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            rc = -999
        finally:
            if timer is not None:
                timer.cancel()
    finally:
        if sc_temp_dir is not None:
            sc_temp_dir.cleanup()

    elapsed = round(time.time() - t0, 3)
    return rc, stdout or "", stderr or "", elapsed


def _build_tg_sitecustomize(mode: str, marker_path: Path) -> str:
    """Builds a minimal, stdlib-only `sitecustomize.py` that intercepts `urllib.request.urlopen`
    and `http.client.HTTPConnection.request` calls to TigerGraph port 14240 (or `/query/`, `/restpp/`, `/graph/`).

    Does NOT touch `builtins.__import__` (which broke pandas/numpy C-extension imports in Round 2
    when `__import__("sys")` or `getattr` ran during half-loaded modules).
    Writes `marker_path` when the fault fires.
    """
    marker_str = str(marker_path)
    return f"""
import urllib.error
import urllib.request

_MARKER = {marker_str!r}
_MODE = {mode!r}
_CALL_COUNT = 0

def _mark_fired(detail: str) -> None:
    try:
        with open(_MARKER, "a", encoding="utf-8") as f:
            f.write(detail + "\\n")
    except Exception:
        pass

_orig_urlopen = urllib.request.urlopen

class _FakeTGResponse:
    def __init__(self, body_bytes: bytes, status: int = 200):
        self._body = body_bytes
        self.status = status
        self.code = status
        self.headers = {{"Content-Type": "application/json"}}
    def read(self, *args, **kwargs) -> bytes:
        return self._body
    def getcode(self) -> int:
        return self.status
    def info(self):
        return self.headers
    def close(self) -> None:
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False

def _patched_urlopen(req, *args, **kwargs):
    global _CALL_COUNT
    url = getattr(req, "full_url", str(req))
    if (":14240" in url) or ("/query/" in url) or ("/restpp/" in url) or ("/graph/" in url):
        _CALL_COUNT += 1
        if _MODE == "raises" and _CALL_COUNT >= 1:
            _mark_fired("urlopen_raises@" + str(_CALL_COUNT) + ":" + url)
            raise urllib.error.URLError("Injected TigerGraph query failure on call #" + str(_CALL_COUNT))
        if _MODE == "empty_dict" and _CALL_COUNT >= 1:
            _mark_fired("urlopen_empty_dict@" + str(_CALL_COUNT) + ":" + url)
            return _FakeTGResponse(b"{{}}")
        if _MODE == "empty_list" and _CALL_COUNT >= 1:
            _mark_fired("urlopen_empty_list@" + str(_CALL_COUNT) + ":" + url)
            return _FakeTGResponse(b"[]")
        if _MODE == "malformed_json" and _CALL_COUNT >= 1:
            _mark_fired("urlopen_malformed_json@" + str(_CALL_COUNT) + ":" + url)
            return _FakeTGResponse(b"{{malformed_json: [unclosed")
    return _orig_urlopen(req, *args, **kwargs)

urllib.request.urlopen = _patched_urlopen
"""


# --- Scenario Definitions ---------------------------------------------------


SCENARIO_NAMES: list[str] = [
    # 1. Baseline & Env-var faults
    "env_clean_baseline",
    "env_assay_fault_midrun_shared_device_down",
    "env_assay_fault_get_txn_garbage",
    "env_assay_tg_unreachable_port",
    # 2. sitecustomize.py import-time monkeypatch faults
    "sc_os_replace_enospc",
    "sc_time_jump_backwards",
    "sc_slow_call_past_timeout",
    "sc_tg_nth_query_raises",
    "sc_tg_nth_query_empty_dict",
    "sc_tg_nth_query_empty_list",
    "sc_tg_nth_query_malformed_json",
    # 3. Filesystem conditions
    "fs_readonly_out_dir",
    "fs_out_dir_is_regular_file",
    "fs_stale_case_json_before_fault",
    "fs_stale_tmp_file_before_clean_run",
    # 4. Process conditions
    "proc_sigterm_mid_run",
    "proc_sigkill_mid_run",
    "proc_concurrent_same_case_same_dir",
    # 5. Hostile / weird input conditions
    "inp_homoglyph_zero_width_case_id",
    "inp_path_traversal_case_id",
    "inp_crlf_in_structured_fields",
    "inp_1mb_trigger_text",
    "inp_nan_risk_score",
    "inp_inf_risk_score",
    "inp_duplicate_case_rows_in_pack",
]


def run_single_scenario(
    scenario: str,
    base_case_id: str = "HHF-007",
    tg_case_id: str = "HHF-005",
    tg_up: bool | None = None,
) -> ScenarioResult:
    """Runs one named chaos scenario in a fresh temporary directory and returns its `ScenarioResult`."""
    cp = case_pack().set_index("case_id", drop=False)
    base_row = cp.loc[base_case_id].to_dict()
    base_fid = int(base_row["flagged_txn_id"])
    base_card = str(base_row["card_id"])
    base_cust = str(base_row["customer_id"])

    if tg_up is None:
        tg_up = is_tg_reachable()

    # --- TigerGraph-path scenarios (run on HHF-005 if TG is reachable, else SKIPPED) ---
    if scenario.startswith("sc_tg_"):
        if not tg_up:
            reason = "SKIPPED: TigerGraph not reachable at 127.0.0.1:14240"
            inv = InvariantCheck(True, True, True, True, True)
            return ScenarioResult(
                scenario=scenario,
                category="sitecustomize_tg",
                case_id=tg_case_id,
                exit_code="SKIP",
                first_line=reason,
                files_after=[],
                invariants=inv,
                status="SKIPPED",
                notes=reason,
                elapsed_s=0.0,
                attestation=_make_attestation(scenario, tg_case_id, "SKIP", "SKIPPED"),
            )

        tg_row = cp.loc[tg_case_id].to_dict()
        tg_fid = int(tg_row["flagged_txn_id"])
        tg_card = str(tg_row["card_id"])
        tg_cust = str(tg_row["customer_id"])

        mode_map = {
            "sc_tg_nth_query_raises": "raises",
            "sc_tg_nth_query_empty_dict": "empty_dict",
            "sc_tg_nth_query_empty_list": "empty_list",
            "sc_tg_nth_query_malformed_json": "malformed_json",
        }
        if scenario not in mode_map:
            raise ValueError(f"Unknown TG scenario: {scenario}")

        with tempfile.TemporaryDirectory(prefix="chaos_tg_") as td:
            out_dir = Path(td)
            marker_file = out_dir / ".chaos_fired.marker"
            sc_code = _build_tg_sitecustomize(mode_map[scenario], marker_file)

            rc, out, err, elapsed = _run_agent_subproc(
                tg_case_id,
                out_dir,
                env_overrides={"ASSAY_TG": "1"},
                sitecustomize_code=sc_code,
            )
            fired = marker_file.exists() and len(marker_file.read_text(encoding="utf-8").strip()) > 0
            fired_detail = marker_file.read_text(encoding="utf-8").strip().splitlines()[0] if fired else "not_fired"
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, tg_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)

            if "Error in sitecustomize" in (err or "") or not fired:
                status = "INVALID"
                notes = f"proof-of-injection marker missing ({fired_detail}); full_stderr={err!r}"
            else:
                notes = (f"fired={fired_detail}; " + notes).strip("; ")
                if rc == 0:
                    inv.inv_b_nonzero_no_fresh_json_and_clear_err = False
                    notes = (notes + "; expected non-zero fail-closed exit on TG fault, got 0").strip("; ")
                status = "PASS" if (inv.all_hold and rc != 0) else "FAIL"

            return ScenarioResult(
                scenario=scenario,
                category="sitecustomize_tg",
                case_id=tg_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, tg_case_id, rc, status, tg_fid, tg_card, tg_cust),
            )

    # --- 1. Environment variable scenarios ---
    if scenario == "env_clean_baseline":
        with tempfile.TemporaryDirectory(prefix="chaos_base_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(base_case_id, out_dir)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(out) or _first_meaningful_line(err)
            status = "PASS" if (rc == 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="env_fault",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "env_assay_fault_midrun_shared_device_down":
        with tempfile.TemporaryDirectory(prefix="chaos_env1_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                env_overrides={"ASSAY_FAULT": "holder_history:down@1"},
            )
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            if rc == 0:
                status = "INVALID"
                notes = "ASSAY_FAULT=holder_history:down@1 did not fire"
            else:
                status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="env_fault",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "env_assay_fault_get_txn_garbage":
        with tempfile.TemporaryDirectory(prefix="chaos_env2_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                env_overrides={"ASSAY_FAULT": "get_transaction:garbage@1"},
            )
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="env_fault",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "env_assay_tg_unreachable_port":
        with tempfile.TemporaryDirectory(prefix="chaos_env_tg_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                env_overrides={
                    "ASSAY_TG": "1",
                    "ASSAY_TG_HOST": "127.0.0.1",
                    "ASSAY_TG_PORT": "1",
                },
            )
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="env_fault",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    # --- 2. sitecustomize.py monkeypatch scenarios (ONLY these use sitecustomize_code) ---
    if scenario == "sc_os_replace_enospc":
        with tempfile.TemporaryDirectory(prefix="chaos_enospc_") as td:
            out_dir = Path(td)
            marker_file = out_dir / ".chaos_fired.marker"
            sc_code = f"""
import errno, os
_orig_replace = os.replace
def _enospc_replace(src, dst, *args, **kwargs):
    if str(dst).endswith("{base_case_id}.json"):
        with open({str(marker_file)!r}, "w", encoding="utf-8") as mf:
            mf.write("os.replace_enospc\\n")
        raise OSError(errno.ENOSPC, "No space left on device (injected os.replace ENOSPC)")
    return _orig_replace(src, dst, *args, **kwargs)
os.replace = _enospc_replace
"""
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                sitecustomize_code=sc_code,
            )
            fired = marker_file.exists()
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            if not fired:
                status = "INVALID"
                notes = f"proof-of-injection marker missing for os.replace; full_stderr={err!r}"
            else:
                status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="sitecustomize",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "sc_time_jump_backwards":
        with tempfile.TemporaryDirectory(prefix="chaos_clock_") as td:
            out_dir = Path(td)
            marker_file = out_dir / ".chaos_fired.marker"
            sc_code = f"""
import time
_orig_time = time.time
_calls = 0
def _backward_time():
    global _calls
    _calls += 1
    if _calls == 1:
        try:
            with open({str(marker_file)!r}, "w", encoding="utf-8") as mf:
                mf.write("time_jump_backwards\\n")
        except Exception:
            pass
    return _orig_time() - (_calls * 60.0)
time.time = _backward_time
"""
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                sitecustomize_code=sc_code,
            )
            fired = marker_file.exists()
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(out) or _first_meaningful_line(err)
            if not fired:
                status = "INVALID"
                notes = f"proof-of-injection marker missing for time.time; full_stderr={err!r}"
            else:
                status = "PASS" if inv.all_hold else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="sitecustomize",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "sc_slow_call_past_timeout":
        # Instead of hooking builtins.__import__, patch concurrent.futures.ThreadPoolExecutor.submit
        # (stdlib, always available in sitecustomize) so when ToolProxy submits `_exec` we sleep past timeout_s!
        with tempfile.TemporaryDirectory(prefix="chaos_slow_") as td:
            out_dir = Path(td)
            marker_file = out_dir / ".chaos_fired.marker"
            sc_code = f"""
import concurrent.futures
import time

_orig_submit = concurrent.futures.ThreadPoolExecutor.submit
_count = 0

def _slow_submit(self, fn, *args, **kwargs):
    global _count
    _count += 1
    if _count >= 2:
        def _delayed():
            try:
                with open({str(marker_file)!r}, "w", encoding="utf-8") as mf:
                    mf.write("slow_submit@" + str(_count) + "\\n")
            except Exception:
                pass
            time.sleep(1.5)
            return fn(*args, **kwargs)
        return _orig_submit(self, _delayed)
    return _orig_submit(self, fn, *args, **kwargs)

concurrent.futures.ThreadPoolExecutor.submit = _slow_submit
"""
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                extra_args=["--timeout", "0.25"],
                sitecustomize_code=sc_code,
            )
            fired = marker_file.exists()
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            if not fired:
                status = "INVALID"
                notes = f"proof-of-injection marker missing for slow submit; full_stderr={err!r}"
            else:
                status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="sitecustomize",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    # --- 3. Filesystem condition scenarios (NO sitecustomize) ---
    if scenario == "fs_readonly_out_dir":
        with tempfile.TemporaryDirectory(prefix="chaos_ro_") as td:
            ro_dir = Path(td) / "readonly_out"
            ro_dir.mkdir(parents=True, exist_ok=True)
            ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
            try:
                rc, out, err, elapsed = _run_agent_subproc(base_case_id, ro_dir)
                files_after = _list_out_files(ro_dir)
                inv, notes = _evaluate_invariants(rc, out, err, ro_dir, base_case_id)
            finally:
                ro_dir.chmod(stat.S_IRWXU)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="filesystem",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "fs_out_dir_is_regular_file":
        with tempfile.TemporaryDirectory(prefix="chaos_fileout_") as td:
            file_as_dir = Path(td) / "out_is_a_file"
            file_as_dir.write_text("not a directory\n", encoding="utf-8")
            rc, out, err, elapsed = _run_agent_subproc(base_case_id, file_as_dir)
            files_after = _list_out_files(file_as_dir)
            inv, notes = _evaluate_invariants(rc, out, err, file_as_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="filesystem",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "fs_stale_case_json_before_fault":
        with tempfile.TemporaryDirectory(prefix="chaos_stale_") as td:
            out_dir = Path(td)
            stale_file = out_dir / f"{base_case_id}.json"
            stale_bytes = b'{"case_id": "HHF-007", "stale": true}\n'
            stale_file.write_bytes(stale_bytes)
            old_ts = time.time() - 3600.0
            os.utime(stale_file, (old_ts, old_ts))
            stale_before = (stale_file.stat().st_mtime, _sha256_bytes(stale_bytes))

            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                extra_args=["--fault", "graph_down"],
            )
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(
                rc, out, err, out_dir, base_case_id, stale_before=stale_before
            )
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="filesystem",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "fs_stale_tmp_file_before_clean_run":
        with tempfile.TemporaryDirectory(prefix="chaos_tmpfile_") as td:
            out_dir = Path(td)
            leftover_tmp = out_dir / f"{base_case_id}.json.tmp.deadbeef00"
            leftover_tmp.write_text("{partial_from_killed_run", encoding="utf-8")

            rc, out, err, elapsed = _run_agent_subproc(base_case_id, out_dir)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(out) or _first_meaningful_line(err)
            status = "PASS" if inv.all_hold else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="filesystem",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    # --- 4. Process condition scenarios (CLEAN agent: NO sitecustomize on PYTHONPATH) ---
    if scenario == "proc_sigterm_mid_run":
        # Clean agent takes ~4-5s to build runtime + investigate; send SIGTERM at 1.5s
        with tempfile.TemporaryDirectory(prefix="chaos_sigterm_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                signal_spec=(signal.SIGTERM, 1.5),
            )
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            # Lead's SIGTERM handler turns SIGTERM into exit 3 with "FAIL CLOSED [EXIT 3] ... stopped by SIGTERM"
            status = "PASS" if (rc == 3 and "FAIL CLOSED [EXIT 3]" in fl and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="process",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "proc_sigkill_mid_run":
        # Clean agent: send SIGKILL at 1.5s; expected returncode -9 or 137 and no <case>.json
        with tempfile.TemporaryDirectory(prefix="chaos_sigkill_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(
                base_case_id,
                out_dir,
                signal_spec=(signal.SIGKILL, 1.5),
            )
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(
                rc, out, err, out_dir, base_case_id, killed_by_sigkill=True
            )
            fl = _first_meaningful_line(err) or _first_meaningful_line(out) or f"killed by SIGKILL (rc={rc})"
            status = "PASS" if (rc in (-9, 137) and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="process",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "proc_concurrent_same_case_same_dir":
        with tempfile.TemporaryDirectory(prefix="chaos_conc_") as td, tempfile.TemporaryDirectory(prefix="chaos_ref_") as td_ref:
            out_dir = Path(td)
            ref_dir = Path(td_ref)

            rc_ref, _, _, _ = _run_agent_subproc(base_case_id, ref_dir)
            ref_bytes = (ref_dir / f"{base_case_id}.json").read_bytes() if (ref_dir / f"{base_case_id}.json").exists() else b""

            env = os.environ.copy()
            env["ASSAY_TG"] = "0"
            env["PYTHONPATH"] = str(ROOT)
            cmd = [sys.executable, "-m", "assay.r2.agent", base_case_id, "--out", str(out_dir)]

            t0 = time.time()
            p1 = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p2 = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out1, err1 = p1.communicate(timeout=SUBPROCESS_TIMEOUT_S * 1.5)
            out2, err2 = p2.communicate(timeout=SUBPROCESS_TIMEOUT_S * 1.5)
            elapsed = round(time.time() - t0, 3)

            rc1, rc2 = p1.returncode, p2.returncode
            files_after = _list_out_files(out_dir)
            ans_path = out_dir / f"{base_case_id}.json"
            final_bytes = ans_path.read_bytes() if ans_path.exists() else b""

            both_ok = (rc1 == 0 and rc2 == 0)
            one_ok_one_clean_refusal = (
                (rc1 == 0 and rc2 in (2, 3) and not _only_traceback(out2, err2))
                or (rc2 == 0 and rc1 in (2, 3) and not _only_traceback(out1, err1))
            )
            bytes_match_ref = len(final_bytes) > 0 and final_bytes == ref_bytes
            concurrent_ok = (both_ok or one_ok_one_clean_refusal) and bytes_match_ref

            combined_out = (out1 or "") + "\n" + (out2 or "")
            combined_err = (err1 or "") + "\n" + (err2 or "")
            inv, notes = _evaluate_invariants(
                0 if (rc1 == 0 or rc2 == 0) else rc1,
                combined_out,
                combined_err,
                out_dir,
                base_case_id,
                concurrent_check=concurrent_ok,
            )
            if not bytes_match_ref:
                notes = (notes + f"; concurrent bytes ({len(final_bytes)}) != ref ({len(ref_bytes)})").strip("; ")
            fl = f"rc1={rc1}, rc2={rc2} | " + (_first_meaningful_line(out1) or _first_meaningful_line(err1))
            status = "PASS" if inv.all_hold else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="process",
                case_id=base_case_id,
                exit_code=f"{rc1}/{rc2}",
                first_line=fl[:220],
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, f"{rc1}/{rc2}", status, base_fid, base_card, base_cust),
            )

    # --- 5. Hostile / weird input scenarios (NO sitecustomize on PYTHONPATH) ---
    if scenario == "inp_homoglyph_zero_width_case_id":
        weird_cid = "\u041d\u041dF-\u200b007"
        with tempfile.TemporaryDirectory(prefix="chaos_homo_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(weird_cid, out_dir)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="input",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "inp_path_traversal_case_id":
        trav_cid = "../../etc/HHF-007"
        with tempfile.TemporaryDirectory(prefix="chaos_trav_") as td:
            out_dir = Path(td)
            rc, out, err, elapsed = _run_agent_subproc(trav_cid, out_dir)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="input",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "inp_crlf_in_structured_fields":
        mut_row = dict(base_row)
        mut_row["card_id"] = f"{base_card}\r\nX-Injected: true"
        mut_row["customer_id"] = f"{base_cust}\r\n"
        with tempfile.TemporaryDirectory(prefix="chaos_crlf_") as td:
            out_dir = Path(td)
            case_file = Path(td) / "input_crlf.json"
            case_file.write_text(json.dumps(mut_row), encoding="utf-8")
            rc, out, err, elapsed = _run_agent_subproc(str(case_file), out_dir)
            case_file.unlink(missing_ok=True)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="input",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "inp_1mb_trigger_text":
        mut_row = dict(base_row)
        mut_row["trigger_text"] = "A" * (1024 * 1024)
        with tempfile.TemporaryDirectory(prefix="chaos_1mb_") as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir()
            case_file = Path(td) / "case_1mb.json"
            case_file.write_text(json.dumps(mut_row), encoding="utf-8")
            rc, out, err, elapsed = _run_agent_subproc(str(case_file), out_dir)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="input",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario in ("inp_nan_risk_score", "inp_inf_risk_score"):
        bad_val = "NaN" if scenario == "inp_nan_risk_score" else "Infinity"
        mut_row = dict(base_row)
        mut_row["trigger_type"] = "risk_score"
        mut_row["risk_score"] = bad_val
        with tempfile.TemporaryDirectory(prefix="chaos_naninf_") as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir()
            case_file = Path(td) / "case_naninf.json"
            case_file.write_text(json.dumps(mut_row), encoding="utf-8")
            rc, out, err, elapsed = _run_agent_subproc(str(case_file), out_dir)
            files_after = _list_out_files(out_dir)
            inv, notes = _evaluate_invariants(rc, out, err, out_dir, base_case_id)
            fl = _first_meaningful_line(err) or _first_meaningful_line(out)
            status = "PASS" if (rc != 0 and inv.all_hold) else "FAIL"
            return ScenarioResult(
                scenario=scenario,
                category="input",
                case_id=base_case_id,
                exit_code=rc,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc, status, base_fid, base_card, base_cust),
            )

    if scenario == "inp_duplicate_case_rows_in_pack":
        # Clean agent (no sitecustomize): run HHF-007 once to establish its idempotency key in .ledger,
        # then run a second case row with the SAME case_id ("HHF-007") but a conflicting flagged_txn_id/card_id
        # into the same out_dir. The second run must refuse with exit 2 (case_id collision) and must NOT
        # overwrite the valid HHF-007.json from the first run with the conflicting row's answer.
        with tempfile.TemporaryDirectory(prefix="chaos_duprow_") as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir()
            rc1, _, _, _ = _run_agent_subproc(base_case_id, out_dir)
            ans_path = out_dir / f"{base_case_id}.json"
            before_mtime = ans_path.stat().st_mtime if ans_path.exists() else 0.0
            before_sha = _sha256_bytes(ans_path.read_bytes()) if ans_path.exists() else ""

            dup_row = dict(base_row)
            dup_row["flagged_txn_id"] = 3660108
            dup_row["card_id"] = "C05061-K1"
            dup_row["customer_id"] = "C05061"
            dup_file = Path(td) / "dup_case.json"
            dup_file.write_text(json.dumps(dup_row), encoding="utf-8")

            rc2, out2, err2, elapsed = _run_agent_subproc(str(dup_file), out_dir)
            files_after = _list_out_files(out_dir)
            fl = _first_meaningful_line(err2) or _first_meaningful_line(out2)

            # Check that the conflicting duplicate row was rejected (rc2 == 2) with a clear one-line error,
            # no traceback, and that the first run's HHF-007.json was untouched (same sha & mtime).
            after_sha = _sha256_bytes(ans_path.read_bytes()) if ans_path.exists() else ""
            after_mtime = ans_path.stat().st_mtime if ans_path.exists() else 0.0
            untouched = (rc1 == 0) and (after_sha == before_sha) and (after_mtime == before_mtime)
            clear_err = (rc2 == 2) and ("Case ID collision" in fl or "ERROR [EXIT 2]" in fl)
            no_tb = not _only_traceback(out2, err2)

            inv = InvariantCheck(
                inv_a_exit0_valid_json_only=True,
                inv_b_nonzero_no_fresh_json_and_clear_err=clear_err and untouched,
                inv_c_never_only_traceback=no_tb,
                inv_d_concurrent_consistent=True,
                inv_e_never_exit0_missing_or_partial=True,
            )
            status = "PASS" if inv.all_hold else "FAIL"
            notes = f"rc1={rc1}, rc2={rc2}, prior_json_untouched={untouched}"
            return ScenarioResult(
                scenario=scenario,
                category="input",
                case_id=base_case_id,
                exit_code=rc2,
                first_line=fl,
                files_after=files_after,
                invariants=inv,
                status=status,
                notes=notes,
                elapsed_s=elapsed,
                attestation=_make_attestation(scenario, base_case_id, rc2, status, base_fid, base_card, base_cust),
            )

    raise ValueError(f"Unknown scenario: {scenario}")


def run_chaos_battery(
    base_case_id: str = "HHF-007",
    tg_case_id: str = "HHF-005",
    scenarios: list[str] | None = None,
) -> ChaosReport:
    """Runs the full C9 unknown-fault chaos battery sequentially (one subprocess at a time,
    except the concurrency scenario) and returns a `ChaosReport`."""
    target_scenarios = scenarios if scenarios is not None else list(SCENARIO_NAMES)
    tg_up = is_tg_reachable()
    results: list[ScenarioResult] = []

    for name in target_scenarios:
        res = run_single_scenario(
            scenario=name,
            base_case_id=base_case_id,
            tg_case_id=tg_case_id,
            tg_up=tg_up,
        )
        results.append(res)

    cases_tested = [base_case_id]
    if tg_up and any(s.startswith("sc_tg_") for s in target_scenarios):
        cases_tested.append(tg_case_id)

    return ChaosReport(
        cases_tested=cases_tested,
        tg_reachable=tg_up,
        results=results,
    )


def write_chaos_doc(report: ChaosReport, doc_path: str | Path = DEFAULT_DOC_PATH) -> Path:
    """Writes `docs/r2/C9_chaos.md` with header `<!-- generated; do not edit -->`."""
    p = Path(doc_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.to_markdown(), encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BLOCK C9 unknown-fault chaos battery.")
    parser.add_argument("--case", default="HHF-007", help="Primary case ID (default: HHF-007)")
    parser.add_argument("--tg-case", default="HHF-005", help="TigerGraph fault case ID (default: HHF-005)")
    parser.add_argument("--out-doc", default=str(DEFAULT_DOC_PATH), help="Path to output markdown doc")
    args = parser.parse_args(argv)

    report = run_chaos_battery(base_case_id=args.case, tg_case_id=args.tg_case)
    out_p = write_chaos_doc(report, args.out_doc)
    print(report.to_markdown())
    print(
        f"Chaos suite complete: {report.passed} PASS, {report.failed} FAIL, "
        f"{report.skipped} SKIPPED, {report.invalid} INVALID -> {out_p}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
