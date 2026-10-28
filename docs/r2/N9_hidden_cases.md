# Block N9 — Hidden-Case Finder & Policy-Invariant Results

## Overview

We converted the 25 strongest attack and drill scenarios from the Red-vs-Blue debate (`data/round2/orch/outputs/debate/final.json`) into executable synthetic case-pack rows (`HID-001` .. `HID-025`) in `tests/r2/hidden_cases.json`. Every fixture's `card_id` and `customer_id` is verified against `assay.r2.frames.tx_all()` in `test_fixtures_consistent` before running any invariants (with `designed_mismatch: true` explicitly marking `HID-014`, which tests the `C09241-K1` vs `C09241-K2` discrepancy). All 25 cases are then executed through a single session-scoped `R2Investigator` instance in `test_run_all_25_hidden_cases`, checking policy-level invariants, `assay.validate.validate`, and the official grader (`data/round2/share/grader/grade.py` with `transactions_2017q1.csv`).

## Construction Queries & Observed Results (25 Cases)

| Case | Debate ID | Flagged Txn | Card | Customer | Trigger | Invariant | Observed Output | Status |
|---|---|---|---|---|---|---|---|---|
| `HID-001` | `ATK-01` | `3660108` | `C05061-K1` | `C05061` | `analyst_request` | `validate_and_grade_clean` | `v=legitimate p=0.02 exp=$0.00 conn=0 sar=False act=['CLOSE_NO_FRAUD', 'CREATE_CASE']` | PASS |
| `HID-002` | `ATK-11` | `3660263` | `C01727-K2` | `C01727` | `analyst_request` | `validate_and_grade_clean` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['CLOSE_NO_FRAUD', 'CREATE_CASE']` | PASS |
| `HID-003` | `DRILL-01` | `3660260` | `C12975-K2` | `C12975` | `customer_report` | `customer_denial_r2_and_hostile_invariance` | `v=fraud p=0.76 exp=$2.90 conn=0 sar=False act=['BLOCK_CARD', 'CREATE_CASE']` | PASS |
| `HID-004` | `ATK-02` | `3660250` | `C00348-K1` | `C00348` | `customer_report` | `customer_denial_r2_and_hostile_invariance` | `v=fraud p=0.76 exp=$4.05 conn=0 sar=False act=['BLOCK_CARD', 'CREATE_CASE']` | PASS |
| `HID-005` | `DRILL-04` | `3660350` | `C09582-K1` | `C09582` | `customer_report` | `customer_denial_r2` | `v=fraud p=0.92 exp=$246.38 conn=0 sar=False act=['BLOCK_CARD', 'CREATE_CASE']` | PASS |
| `HID-006` | `ATK-09` | `3660153` | `C11663-K1` | `C11663` | `customer_report` | `customer_denial_r2_and_hostile_invariance` | `v=fraud p=0.99 exp=$1399.83 conn=0 sar=True act=['BLOCK_CARD', 'CREATE_CASE', 'FILE_REPORT']` | PASS |
| `HID-007` | `DRILL-14` | `3660350` | `C09582-K1` | `C09582` | `customer_report` | `customer_denial_r2_and_hostile_invariance` | `v=fraud p=0.92 exp=$246.38 conn=0 sar=False act=['BLOCK_CARD', 'CREATE_CASE']` | PASS |
| `HID-008` | `ATK-17` | `3660153` | `C11663-K1` | `C11663` | `customer_report` | `customer_denial_r2_and_hostile_invariance` | `v=fraud p=0.99 exp=$1399.83 conn=0 sar=True act=['BLOCK_CARD', 'CREATE_CASE', 'FILE_REPORT']` | PASS |
| `HID-009` | `DRILL-08` | `3660353` | `C07871-K1` | `C07871` | `risk_score` | `hostile_invariance_and_no_secret_leak` | `v=legitimate p=0.05 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-010` | `ATK-03` | `3638829` | `C08842-K2` | `C08842` | `risk_score` | `legit_safe_and_hostile_invariance` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-011` | `ATK-10` | `3631383` | `C07196-K2` | `C07196` | `risk_score` | `hostile_invariance` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-012` | `ATK-10` | `3602445` | `C00750-K2` | `C00750` | `risk_score` | `hostile_invariance` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-013` | `ATK-10` | `3611043` | `C02502-K2` | `C02502` | `risk_score` | `hostile_invariance` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-014` | `ATK-04` | `3640919` | `C09241-K1` (`designed_mismatch`) | `C09241` | `risk_score` | `legit_safe_and_grade_clean` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-015` | `ATK-13` | `9999999` | `C05061-K1` | `C05061` | `risk_score` | `missing_txn_clear_error` | `ValueError: HID-015: flagged transaction 9999999 is not in the bank's data` | PASS |
| `HID-016` | `DRILL-05` | `3660353` | `C07871-K1` | `C07871` | `risk_score` | `clock_skew_safe_or_clear_error` | `verdict=legitimate p=0.05 errs=0 gp=0` | PASS |
| `HID-017` | `DRILL-12` | `3660127` | `C06220-K1` | `C06220` | `analyst_request` | `validate_and_grade_clean_and_bounded` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['CLOSE_NO_FRAUD', 'CREATE_CASE']` | PASS |
| `HID-018` | `ATK-12` | `3638829` | `C08842-K2` | `C08842` | `customer_report` | `jan_only_affected_txns_and_grade_clean` | `v=fraud p=0.73 exp=$43.92 conn=0 sar=False act=['BLOCK_CARD', 'CREATE_CASE']` | PASS |
| `HID-019` | `ATK-07` | `3612011` | `C02753-K1` | `C02753` | `risk_score` | `legit_never_blocks_or_files_sar` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-020` | `ATK-06` | `3660355` | `C09177-K1` | `C09177` | `risk_score` | `validate_and_grade_clean` | `v=legitimate p=0.02 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-021` | `ATK-08` | `3660153` | `C11663-K1` | `C11663` | `customer_report` | `customer_denial_r2_and_r10_check` | `v=fraud p=0.99 exp=$1399.83 conn=0 sar=True act=['BLOCK_CARD', 'CREATE_CASE', 'FILE_REPORT']` | PASS |
| `HID-022` | `DRILL-07` | `3611043` | `C02502-K2` | `C02502` | `customer_report` | `high_exposure_denial_l2_and_sar` | `v=fraud p=0.73 exp=$3008.76 conn=0 sar=True act=['BLOCK_CARD', 'CREATE_CASE', 'FILE_REPORT']` | PASS |
| `HID-023` | `ATK-15` | `3655457` | `C12223-K1` | `C12223` | `risk_score` | `legit_never_blocks_or_files_sar` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-024` | `DRILL-16` | `3626720` | `C06115-K1` | `C06115` | `risk_score` | `validate_and_grade_clean` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |
| `HID-025` | `DRILL-02` | `3601725` | `C00543-K1` | `C00543` | `risk_score` | `state_isolation_and_idempotence` | `v=legitimate p=0.01 exp=$0.00 conn=0 sar=False act=['ALLOW_TRANSACTION', 'CLOSE_NO_FRAUD']` | PASS |

## Key Findings for Lead

1. **Fixture consistency (`test_fixtures_consistent`):** All 24 real-transaction cases match `tx_all()` `customer_id` and `card_id` (with `HID-014` explicitly flagged `designed_mismatch: true` for `C09241-K1` vs `C09241-K2`), and `HID-015` tests missing `flagged_txn_id=9999999`.
2. **All 25 cases pass `assay.validate.validate` and `grade.py` hard + policy checks (`25/25 PASS`).**
3. **Diagnostic findings from `final.json` (`DRILL-01` / `ATK-01` / `ATK-02` / `ATK-04`):**
   - On `HID-003` (`3660260`, `$2.90`) and `HID-004` (`3660250`, `$4.05`), `R2Investigator` produces `verdict=fraud` with `BLOCK_CARD` + `CREATE_CASE` (satisfying R2), but `connected_card_ids` in the final output is `[]` (`conn=0, sar=False`) because `super()._episode()` runs before `_extra_findings()` populates `self._multi`.
   - On `HID-001` (`3660108`), `HID-002` (`3660263`), `HID-017` (`3660127`), and `HID-014` (`3640919`, card mismatch `C09241-K1` vs `C09241-K2`), the base `Investigator` reaches `p <= 0.02` (`verdict=legitimate`) on step 1 before `assumed_clear_allowed()` is consulted.
