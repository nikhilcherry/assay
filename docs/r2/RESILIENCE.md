# Assay Resilience & Boundary Statement

## 1. Runtime Dependencies
Assay runs as a deterministic Python 3 investigation engine (`assay/r2/investigate_r2.py`) under Fraud Operations Policy v1.0. Runtime dependencies are strictly local:
- **Staged Local Frames (`assay/r2/frames.py`)**: `data/stage/r2/tx_all.parquet` (650,938 transactions, July 2016–January 2017), `closed_cases_history.csv` (5,585 closed cases), and `cases/case_pack_final.csv` (18 exam cases).
- **TigerGraph 4.2 CE (`docs/r2/B14_vector.md`, `docs/r2/B15_rings.md`)**: Backing graph and 24-dimension `COSINE` vector index (`ClosedCase.behaviour`) queried via GSQL (`similar_cases_fused`, `tg_wcc`, `tg_louvain`) when `ASSAY_TG=1`.
- **Local LightGBM Scorer (`docs/r2/B1_model.md`)**: Isotonic-calibrated `model_r2` artifact and pre-scored local parquet cache (`scores_b1.parquet`, `scores_r2.parquet`).
- **Python Libraries**: `pandas`, `numpy`, `networkx`, `scikit-learn`, `lightgbm`, `fastapi`, `httpx`, `pytest`.

## 2. Boundary & Data Egress (`tokens = 0`)
- **Decision-Time Egress — NOTHING Leaves the Bank (`docs/r2/B11_audit.md`)**: External service dependencies on the decision path equal **0** (`100%` local execution, LLM calls = 0, tokens = 0). No customer, card, or transaction data leaves the boundary at runtime.
- **Build-Time Model Disclosure**: Build-time Gemini use for writing code saw only code and schema, never customer rows.

## 3. Fail-Closed Behaviour on Dependency Failure
Assay wraps every store, model, and graph call in `assay/r2/harness.py` (`docs/r2/B9_harness.md`, `docs/r2/B14_vector.md`):
- **Store / Graph / Model Faults (`down`, `timeout`, `garbage`, `slow`)**: Tested across drills with a `100%` fail-closed rate and zero silent fallback. When `ASSAY_TG=1` and TigerGraph 4.2 is unreachable, `LocalVectorIndex` raises `TigerGraphUnavailableError`.
- **Mid-Run Failure Isolation**: If any dependency fails mid-investigation, the harness aborts with a non-zero exit code, returns `RunResult(status=INCOMPLETE, error_type="DependencyDown")`, writes `<case_id>.incomplete.json`, and leaves **0** false `<case_id>.json` answer files on disk.
- **Hostile Input Defence**: All `trigger_text` strings are treated strictly as data, never instructions. Malformed or hostile tool inputs (SQL keywords, negative amounts, out-of-bounds risk scores, unparseable dates) return `REJECTED_INPUT` prior to execution.
- **Missing Approver Config (`docs/r2/B10_approvals.md`)**: If `ASSAY_APPROVERS` is absent, the approval server fails closed at startup (`FileNotFoundError`).

## 4. Redacted Hash-Chained Audit Log
Every tool call, parameter, row count, digest, and decision delta is recorded in an append-only SHA256 hash-chained JSONL ledger (`assay/r2/audit.py`, `docs/r2/B11_audit.md`):
- **Redaction (`redact`)**: Masks API keys, Bearer tokens, Authorization headers, secret keys, phone numbers, Aadhaar numbers, and emails, and truncates strings > 200 chars with a SHA256 suffix while preserving bank IDs (`case_id`, `flagged_txn_id`, `card_id`, `customer_id`, `uid`) `100%` intact.
- **Verification & Replay**: `python -m assay.r2.audit verify <log_path>` catches any modified, deleted, or reordered line; `python -m assay.r2.audit replay <log_path> <case_id>` reconstructs the exact step trace.

## 5. Server-Side Approvals & Idempotent Reruns
- **Role Enforcement (`assay/r2/approvals.py`, `docs/r2/B10_approvals.md`)**: `auto` actions execute immediately; `L1` (`DECLINE_TRANSACTION`, `BLOCK_CARD` <= `$2,500.00`) and `L2` (`BLOCK_CARD` >= `$2,500.01`, `BLOCK_ALL_CARDS`, `FILE_REPORT`) verify SHA256 Bearer token digests via `hmac.compare_digest`. Body-role spoofing, agent tokens, and self-approvals return `HTTP 403 Forbidden`; 5 wrong-role attempts/min trigger `HTTP 429 Too Many Requests` on the 6th attempt.
- **Idempotency (`docs/r2/B9_harness.md`, `docs/r2/B10_approvals.md`)**: Reruns produce `100%` byte-identical JSON and SHA256 digests. Decisions require `Idempotency-Key` (`HTTP 400` if missing; cached response on match; `HTTP 409 Conflict` on mismatch or double decision).
