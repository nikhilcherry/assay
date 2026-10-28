# BLOCK B21 -- Casefile Management & Audit Trail

## Overview
Subsystem `assay.r2.casefile` implements Challenge 14 (Case Management). It provides a full lifecycle state machine and append-only SQLite store with SHA-256 hash chaining to ensure immutable, verifiable audit trails for card fraud investigations.

## Key Capabilities
- **State Machine**: Transitions through `OPEN`, `INVESTIGATING`, `AWAITING_CUSTOMER`, `AWAITING_APPROVAL`, `ESCALATED`, `CLOSED_FRAUD`, and `CLOSED_LEGITIMATE`. Illegal transitions raise `InvalidTransitionError`.
- **Policy Approval Routing**: Enforces policy Section 2 server-side rules:
  - `DECLINE_TRANSACTION` and `BLOCK_CARD` (exposure $\le \$2,500.00$) require `L1` (team lead) approval.
  - `BLOCK_CARD` (exposure $> \$2,500.00$), `BLOCK_ALL_CARDS`, and `FILE_REPORT` require `L2` (fraud manager) approval.
  - Attempts by unauthorized actors or lower roles raise `PolicyApprovalError`.
- **Automatic Tamper Verification**: Auto-verifies event hash chain (`prev_hash` integrity and canonical SHA-256 payload checksums) whenever a case is loaded (`__init__`). Detects payload edits, deleted rows, and reordered events by raising `TamperError`.
- **SLA & Open Questions**: Tracks R4 24h customer response SLAs and open investigation questions.
- **Cold Handover View**: Provides `handover_summary()` allowing an analyst to pick up any open or closed case cold.
- **CLI**: Executable via `python -m assay.r2.casefile show <case_id>` and `handover <case_id>`.

## Measured Results
- `test_full_life_three_cases_two_sittings`: Evaluated 3 cases across multiple DB sittings; verified reopen, customer reply, SLA breach, escalation, role checks, and direct SQL tamper detection.
- `test_approval_boundary_2500`: Verified exact boundary behaviour at $\$2,500.00$ (`L1` allowed) vs $\$2,500.01$ (`L1` rejected, `L2` required).
- `test_tamper_detection_scenarios`: Verified detection of payload modifications and intermediate row deletion.
