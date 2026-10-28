# B26 — Live Demo Console

## Overview
`assay/r2/console.py` and `assay/r2/console.html` implement the self-contained FastAPI + dark analyst console for the 5-minute live judge demo:
- **Left panel**: Lists all 18 cases from `case_pack_final.csv` (`HHF-001` .. `HHF-018`) with `trigger_type`, metadata, and `trigger_text`. The 6 cases containing prompt-injection strings (`HHF-004`, `HHF-005`, `HHF-008`, `HHF-010`, `HHF-014`, `HHF-016`) have their hostile spans highlighted in amber and tagged `treated as data`.
- **Live Investigation Replay (`POST /run/{case_id}`)**: Runs the real `R2Investigator` over `InstrumentedR2Store` (subclass of `R2LocalStore` constructed once at startup). Returns the full validated answer JSON, the tool-call trace (`store.calls`) with exact row counts, and the step-by-step evidence ledger with LRs and log-odds probability needle positions.
- **Server-Side Approvals (`/approvals`)**: Mounts `assay.r2.approvals` under `/approvals`. `L1` and `L2` actions start in `awaiting approval` state and are approved in-page using a Bearer token kept in browser memory only.
- **Live Fail-Closed Drill (`POST /fault`)**: The `'Break it'` toggle sets `ASSAY_FAULT` (e.g. `get_transaction:down@2` or `card_window:down@1`), causing the store wrapper to halt mid-investigation and return an explicit fail-closed error with no guessed answer.

## Measured Results (`tests/r2/test_console.py`)
- Store + pattern model startup time: `12.66s`
- Cases served by `GET /cases`: `18` (`6` with quarantined prompt-injection spans: `HHF-004`, `HHF-005`, `HHF-008`, `HHF-010`, `HHF-014`, `HHF-016`)
- Live run on `HHF-007` (`POST /run/HHF-007`):
  - Verdict: `fraud` (`fraud_probability = 0.96`, pattern `account_takeover`)
  - Attested exposure: `$593.02`
  - Tool calls recorded: `9` calls with exact per-call row counts
  - Evidence ledger steps: `7` findings updating the probability needle to `0.96`
  - Execution latency (warm store): `43.23 ms`
- Role enforcement & ledger verification (`/approvals`): `L1` token rejected (`403`) on `L2` action (`FILE_REPORT`), `L2` token approved (`200`), SHA-256 hash chain verified (`valid: true`).

## CLI Usage
```bash
python -m assay.r2.console --port 8742
```
