# BLOCK N1 — Deployable Single-Case Agent (`assay.r2.agent`)

## Overview

`python -m assay.r2.agent <case_id>` is the single entry point that wires all passing blocks into the command used in Stage 1 (case pack) and Stage 2 (live resilience drills):

```bash
python -m assay.r2.agent HHF-007 [--out cases_r2] [--fault graph_down|tool_timeout|garbage|empty_body|none] [--role analyst|l1|l2]
```

### Pipeline Steps & Exit Codes
1. **Input Validation & Path/Collision Guards (`exit 2` on failure, no answer file):** Rejects `case_id` values with path traversal (`../`), slashes (`/`), or NUL bytes (`\x00`), validates structure/types via `assay.r2.harness.validate_case_row`, and detects `case_id` payload collisions in `RunLedger`.
2. **Untrusted Trigger Quarantine & Secret/PII Scrubbing:** Parses `trigger_text` with `assay.r2.untrusted.parse_trigger` and `cross_check`. Reconstructs a clean canonical prompt from typed `TriggerFacts` so raw free text never reaches decision logic; scrubs AWS keys (`AKIA...`), GitHub/API tokens (`ghp_...`, `sk-...`), Bearer tokens, 13–19 digit card PANs, emails, and phone numbers before recording quarantined strings as `document` evidence in `<case_id>.json` and `audit.jsonl`.
3. **Fail-Closed Investigation (`exit 3` on failure, no answer file):** Builds `R2LocalStore` and `R2Investigator` once per process (~3.8 GB RSS) and sets `disputes.W_OVERRIDE` once at module startup. Wraps the store in `ToolProxy` with response validators that reject HTTP 200 empty/truncated DataFrames (`DependencyDown`), timeouts (`ToolTimeout`), and malformed responses (`GarbageResponse`), exiting `3` with no answer file.
4. **Concurrency-Safe Idempotent Output (`write_answer` + `RunLedger` + `flock`):** Serializes per-case file writes with OS-level `flock` + `RLock` and guards investigator execution so concurrent runs never corrupt `<case_id>.json` or leak state across cases.
5. **Attested SAR & Tip-Off Guard:** Evaluates Policy §3a (`assay.r2.sar.sar_required`) and attaches the attested narrative and timeline from `draft_sar(answer, tx_as_of)` while scrubbing customer-facing `evidence_requests` against tip-off terms.
6. **Hash-Chained Audit Trail & Case Lifecycle:** Appends redacted tool calls, findings, and decision deltas to `audit.jsonl` (`assay.r2.audit.AuditLog`) and updates the SQLite state machine in `cases.db` (`assay.r2.casefile.CaseFile`).
7. **Server-Side Approval Enforcement:** Writes any `L1`/`L2` actions (`DECLINE_TRANSACTION`, `BLOCK_CARD`, `BLOCK_ALL_CARDS`, `FILE_REPORT`) to `approvals_ledger.jsonl` (`assay.r2.approvals.Ledger`) with `status="pending"`. Passing `--role l1` or `--role l2` on the CLI is logged and ignored for approval decisions.

## Measured Test Results

Measured via `pytest -q -s tests/r2/test_agent.py`:

| Case ID | Mode / Drill | Exit Code | Verdict | Fraud Prob | Pattern | Exposure (USD) | SAR (`file`) | Final Actions | Re-run / Verification |
|---|---|---:|---|---:|---|---:|---|---|---|
| `HHF-007` | Normal (`none`) | `0` | `fraud` | `0.85` | `card_not_present_new_device` | `$236.28` | `False` | `['BLOCK_CARD', 'CREATE_CASE']` | Byte-identical re-run & 4-way concurrent run |
| `HHF-006` | Normal (`none`) | `0` | `fraud` | `0.95` | `undocumented` | `$75.60` | `True` | `['BLOCK_CARD', 'CREATE_CASE', 'FILE_REPORT', 'MONITOR_CONNECTED_CARDS', 'ESCALATE_TO_ANALYST']` | Byte-identical re-run |
| `HHF-012` | Normal (`none`) | `0` | `legitimate` | `0.06` | `none` | `$0.00` | `False` | `['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | Byte-identical re-run |
| `HHF-005` + `HHF-017` | Concurrent multi-case | `0` | isolated | — | — | — | — | `HHF-017` has `connected_card_ids=[]` | Zero state cross-contamination; `W_OVERRIDE` constant |
| `HHF-004` | Secrets/PII in `trigger_text` + `--role l2` | `0` | `fraud` | `0.96` | `card_not_present_new_device` | `$416.82` | `True` | `['BLOCK_CARD', 'CREATE_CASE', 'FILE_REPORT', 'MONITOR_CONNECTED_CARDS']` | AWS/GH/Bearer/PAN/email/phone scrubbed; L1/L2 `pending` |
| `HHF-007` | `--fault graph_down` | `3` | — | — | — | — | — | No answer written | Audit log verified (`DependencyDown`) |
| `HHF-007` | `--fault tool_timeout` | `3` | — | — | — | — | — | No answer written | Audit log verified (`ToolTimeout`) |
| `HHF-007` | `--fault garbage` | `3` | — | — | — | — | — | No answer written | Audit log verified (`GarbageResponse`) |
| `HHF-007` / `HHF-008` | `--fault empty_body` / 0-row DataFrame | `3` | — | — | — | — | — | No answer written | Audit log verified (`DependencyDown`) |
| `../HHF-007`, `HHF/007`, `\x00`, collision | Path traversal / collision | `2` | — | — | — | — | — | No answer written | Audit log verified |
