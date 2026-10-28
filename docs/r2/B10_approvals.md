# Block B10 Results -- Server-Side Approvals & Audit Ledger

## Build Overview
- Server-side approval engine built with `fastapi` and `pydantic`.
- Role authentication via constant-time SHA256 hashed Bearer token lookup (`hmac.compare_digest`) against server-side approvers file (`ASSAY_APPROVERS`).
- Fail-closed initialization: server refuses to start if approvers file is missing.
- Approval routing (`route_for`) implemented server-side following Fraud Operations Policy Section 2.
- Body role tamper detection implemented with warning logging.
- SHA256 hash-chained append-only JSONL ledger (`Ledger`) storing only token hashes.
- Idempotency key replay protection (`Idempotency-Key` header required for decisions): exact body match returns cached decision, differing body yields `HTTP 409 Conflict`.
- Rate-limiting: wrong-role authorization failures capped at 5 per minute per token; 6th attempt yields `HTTP 429 Too Many Requests` with security event logging.
- CLI subcommands: `make-token` (prints raw token once, appends only SHA256 digest) and `serve`.

## Test Output & Measured Results
- `route_for` policy mapping:
  - `ALLOW_TRANSACTION`, `VERIFY_WITH_CUSTOMER` -> `auto`
  - `DECLINE_TRANSACTION` -> `L1`
  - `BLOCK_CARD` at $2,500.00 -> `L1`
  - `BLOCK_CARD` at $2,500.01 -> `L2`
  - `BLOCK_ALL_CARDS`, `FILE_REPORT` -> `L2`
- FastAPI test suite outcomes (`pytest tests/r2/test_approvals.py`):
  - Missing approvers file -> server initialization fails closed (`FileNotFoundError`).
  - Missing `Idempotency-Key` header on decision -> returned `HTTP 400 Bad Request`.
  - Same Idempotency-Key + same body -> returned original cached result.
  - Same Idempotency-Key + different body -> returned `HTTP 409 Conflict`.
  - Wrong role decision attempt on L2 item -> returned `HTTP 403 Forbidden`.
  - Body role spoofing (`"role": "L2"`) in body -> ignored, logged tamper attempt, returned `HTTP 403 Forbidden`.
  - Agent decision attempt -> returned `HTTP 403 Forbidden`.
  - Self-approval attempt -> returned `HTTP 403 Forbidden`.
  - Double decision on already decided proposal -> returned `HTTP 409 Conflict`.
  - Rate limiting (5 wrong-role attempts / min) -> 6th attempt returned `HTTP 429 Too Many Requests`.
  - Auto route proposals -> status `auto_executed` immediately without human intervention.
  - `GET /proposals?case_id=HHF-010` -> filtered listing returned matching proposals.
  - Ledger verification (`/ledger/verify`):
    - Valid ledger -> `{"valid": True}`.
    - Tampered entry payload -> detected hash mismatch, returned `{"valid": False}`.
  - CLI `make-token` -> printed raw token once and saved only SHA256 digest to JSON disk config.
