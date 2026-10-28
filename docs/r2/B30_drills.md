# Live Resilience Drills Scorecard (B30)

- **Cases tested:** HHF-007, HHF-012, HHF-003
- **Total drill assertions:** 69 (23 per case across D1–D9)
- **Passed:** 69
- **Failed:** 0

## Diagnostic Findings in Harness / Approvals / Investigator

- `harness.validate_case_row` does not validate `trigger_type` against `{'risk_score','customer_report','analyst_request'}`; caught in investigator wrapper.
- `harness.validate_case_row` does not enforce a maximum length on `trigger_text` (100k char string accepted by `validate_case_row`); caught in investigator wrapper.
- `harness.write_answer` leaves a temporary `.tmp.<uuid>` file on disk if `os.replace` raises mid-write, though target `<case_id>.json` is never created and no 'saved' status is returned.

## Summary Drill Table (drill, expected, observed, pass)

| drill | expected | observed | pass |
|---|---|---|---|
| `D1:HHF-007:call_1(reset:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-007:call_3(holder_history:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-007:call_last(closed_cases_like:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D2:HHF-007:get_transaction:timeout@1` | INCOMPLETE / ToolTimeout / no <case>.json | INCOMPLETE / ToolTimeout / saved=False | **PASS** |
| `D3:HHF-007:injected_garbage_string` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-007:wrong_columns` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-007:nan_amounts` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-007:negative_ids` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D4:HHF-007:sql_injection` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-007:negative_risk_score` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-007:unknown_trigger_type` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | INCOMPLETE / ValueError / saved=False | **PASS** |
| `D4:HHF-007:huge_trigger_text_100k` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | INCOMPLETE / ValueError / saved=False | **PASS** |
| `D4:HHF-007:nonexistent_flagged_txn` | INCOMPLETE / KeyError / no <case>.json | INCOMPLETE / KeyError / saved=False | **PASS** |
| `D4:HHF-007:unicode_prompt_injection` | COMPLETE / verdict=fraud (prompt ignored) | COMPLETE / verdict=fraud | **PASS** |
| `D4:HHF-007:card_id_mismatch_resolved` | COMPLETE / resolved to txn truth / verdict=fraud | COMPLETE / verdict=fraud | **PASS** |
| `D5:HHF-007:wrong_role_L1_on_L2` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-007:agent_token_approval` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-007:role_in_request_body` | HTTP 403 + body tamper logged + status=pending | HTTP 403 + tamper_logged=True + status=pending | **PASS** |
| `D5:HHF-007:replayed_decision_idempotency` | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | **PASS** |
| `D6:HHF-007:same_case_twice_identical_bytes` | COMPLETE/COMPLETE & identical canonical JSON bytes | COMPLETE/COMPLETE & sha256_match=True | **PASS** |
| `D7:HHF-007:exception_after_tmp_write` | OSError raised / no <case>.json / no 'saved' status | OSError / final_exists=False | **PASS** |
| `D8:HHF-007:hash_chain_tamper_detection` | verify() valid=False with SHA256 Hash mismatch | valid=False (Hash mismatch line 0...) | **PASS** |
| `D9:HHF-007:no_external_llm_on_decision_path` | 0 network/LLM imports in investigate modules | forbidden_imports=0 | **PASS** |
| `D9:HHF-007:local_lgbm_missing_score_fail_closed` | INCOMPLETE / DependencyDown (MissingScore) / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |

*(Full 69-row table across HHF-007, HHF-012, and HHF-003 generated automatically when running `pytest tests/r2/test_drills.py` or `python -m assay.r2.drills`.)*
