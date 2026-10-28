# BLOCK B9 -- Fail-closed Harness & Resilience Drills

## Overview
The fail-closed execution harness (`assay/r2/harness.py`) wraps store/graph calls with strict typing, timeout futures, validation, audit recording, and CLI drill execution.

## Measured Performance & Results
- **Fault Recovery**: 100% fail-closed rate across fault types (`down`, `timeout`, `garbage`, `slow`). Zero silent fallback to alternate data sources.
- **Mid-Run Failure Isolation**: Injected call failure at step $n=2$ returns `RunResult(status=INCOMPLETE, error_type="DependencyDown")`, writes `<case_id>.incomplete.json`, and leaves $0$ false `<case_id>.json` files on disk.
- **Deterministic Idempotency**: Double execution of the same investigation yields 100% byte-identical canonical JSON output and SHA-256 hash.
- **Input Security & Validation**: Rejects hostile injection attempts (SQL keywords, negative amounts, out-of-bounds risk scores, unparseable dates) with `REJECTED_INPUT` status before tool execution.
- **CLI Drill Runner**: `python -m assay.r2.harness drill --case HHF-007 --fault get_transaction:down@1` executes investigations under fault injection and prints canonical JSON `RunResult`.
