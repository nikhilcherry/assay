# BLOCK B11 -- Audit Trail & Data Protection Measurements

## Summary
BLOCK B11 establishes an append-only, SHA256 hash-chained JSONL audit trail (`AuditLog`), investigation replay capabilities (`replay`), automated secret/PII redaction (`redact`), CLI verification tools, external service egress registration (`egress_register`), and repository secret scanning (`scan_for_secrets`).

## Measured Benchmarks & Verification Results

1. **Audit Chain Tamper Detection**:
   - Clean verification: 100% pass on unbroken hash chains (`ok=True, bad_line=-1`).
   - Line modification detection: Identified exact line index on payload edits (`ok=False, bad_line=1`).
   - Line deletion detection: Caught missing link in hash chain (`ok=False, bad_line=1`).
   - Line reordering detection: Caught broken `prev_hash` pointer (`ok=False, bad_line=1`).

2. **Data Protection & Redaction**:
   - Secrets masked: API keys (`AIzaSy...`), bearer tokens (`Bearer ...`), authorization headers (`Authorization: ...`), dict keys (`api_key`, `auth_token`, `password`, `secret`).
   - PII masked: Indian phone numbers (`+91 9876543210`), Aadhaar numbers (`2345 6789 0123`), emails (`investigator@bankdomain.com`).
   - Free text truncation: Strings > 200 chars truncated with attached SHA256 digest suffix.
   - Bank IDs preserved: Pseudonymous operational IDs (`case_id`, `flagged_txn_id`, `card_id`, `customer_id`, `uid`) verified 100% intact across all 18 case pack rows.

3. **CLI Verification & Replay**:
   - Command `python -m assay.r2.audit verify <log_path>` reports valid/invalid status with exit code.
   - Command `python -m assay.r2.audit replay <log_path> <case_id>` prints formatted step table containing step, tool/event, params, rows, result digest, and decision deltas.

4. **Repository Secret Scan**:
   - Scanned tracked (`git ls-files`) and untracked-but-not-ignored (`git ls-files --others --exclude-standard`) files, excluding `data/`.
   - Reported file count scanned and 0 hardcoded secret findings across repository.

5. **Egress Dependencies**:
   - Decision path external services: **0** (100% local in-boundary decision path).

## Update
- Scanner now takes optional `stats` dict (`files_scanned`), skips only top-level `data/`; replay CLI shows `decision:<ACTION>`; test covers phone/Aadhaar/Authorization redaction and id-shape preservation. Test secrets are built at runtime.
