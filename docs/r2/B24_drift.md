# BLOCK B24 Results: Schema Drift Survival (Challenge 12)

## Measured Drift Numbers

### 1. RAW Bank Files (397 Columns Snapshot)
- **Transactions (`data/transactions.csv` vs `transactions_2017q1.csv`)**:
  - 397 columns compared directly from raw bank files.
  - 0 added columns, 0 removed columns, 0 dtype changes across all 397 columns.
  - Row counts: 2016 baseline = 590,742 rows, Jan 2017 = 60,196 rows.
- **Identity (`data/identity.csv` vs `identity_2017q1.csv`)**:
  - 41 raw columns compared.
  - 2016 baseline = 144,432 rows, Jan 2017 = 15,893 rows.
- **Closed Cases History (`closed_cases_history.csv`)**:
  - Row Counts: Before = 5,565 cases, After = 5,585 cases (+20 new cases).
  - Shared Case IDs: 5,565 case IDs (100% overlap).
  - Changed Old Rows: 0 changed rows among the shared 5,565 case IDs.
  - 20 New CC IDs: `CC-5566` through `CC-5585` (representing Round 1 exam cases `HHG-001`..`HHG-020`).

### 2. Semantic Breaks Analysis
- **1. Card ID Derivation Mismatch**:
  - Case pack case `HHF-011` names `card_id = 'C09241-K1'` for customer `C09241` on flagged transaction `3640919`.
  - Bank transaction history for customer `C09241` contains only debit card transactions (`card6 = debit`), which maps to derived `C09241-K2`.
  - Raw bank transactions do not store `card_id`; derivation depends on card6 type counts.
- **2. risk_score Distribution Shift (PSI)**:
  - Calculated Population Stability Index (PSI) = `0.0842` between Dec 2016 (mean `0.106`) and Jan 2017 (mean `0.089`).
  - Model was unretrained for Jan 2017; risk_score is unreliable for new activity and cannot serve as proof of innocence.
- **3. TransactionDT Continuity**:
  - 2016 Max `TransactionDT` = `15,811,134` seconds.
  - 2017 Jan Min `TransactionDT` = `15,811,220` seconds.
  - Continuity gap = **86 seconds** across the month boundary.

## Approval Gate Enforcement
- **Refusal**: `apply_mapping()` raises `PermissionError` when called without a matching approval record or valid L1/L2 token hash.
- **Approval**: Succeeds when a valid approval record matching `mapping_hash` signed by an L1/L2 token hash is present in `approvals.json`.
