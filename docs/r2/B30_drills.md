# Live Resilience Drills Scorecard (B30)

- **Cases tested:** HHF-007, HHF-012, HHF-003
- **Total drill assertions:** 75
- **Passed:** 75
- **Failed:** 0

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
| `D4:HHF-007:sql_injection_card_id` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-007:sql_in_trigger_text_is_data` | COMPLETE / verdict=fraud (text never executed) | COMPLETE / verdict=fraud | **PASS** |
| `D4:HHF-007:negative_risk_score` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-007:unknown_trigger_type` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-007:huge_trigger_text_100k` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-007:nonexistent_flagged_txn` | INCOMPLETE / KeyError / no <case>.json | INCOMPLETE / KeyError / saved=False | **PASS** |
| `D4:HHF-007:unicode_prompt_injection` | COMPLETE / verdict=fraud (prompt ignored) | COMPLETE / verdict=fraud | **PASS** |
| `D4:HHF-007:card_id_mismatch_resolved` | COMPLETE / resolved to txn truth / verdict=fraud (DQ-07) | COMPLETE / verdict=fraud | **PASS** |
| `D5:HHF-007:wrong_role_L1_on_L2` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-007:agent_token_approval` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-007:role_in_request_body` | HTTP 403 + body tamper logged + status=pending | HTTP 403 + tamper_logged=True + status=pending | **PASS** |
| `D5:HHF-007:replayed_decision_idempotency` | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | **PASS** |
| `D6:HHF-007:same_case_twice_identical_bytes` | COMPLETE/COMPLETE & identical canonical JSON bytes | COMPLETE/COMPLETE & sha256_match=True (4407 bytes) | **PASS** |
| `D7:HHF-007:exception_after_tmp_write` | OSError raised / no <case>.json / no 'saved' status | OSError / final_exists=False | **PASS** |
| `D8:HHF-007:hash_chain_tamper_detection` | verify() valid=False with SHA256 Hash mismatch | valid=False (Hash mismatch line 0: computed 59fdcfcec...) | **PASS** |
| `D9:HHF-007:no_external_llm_on_decision_path` | 0 network/LLM imports in investigate modules | forbidden_imports=0 | **PASS** |
| `D9:HHF-007:local_lgbm_missing_score_fail_closed` | INCOMPLETE / DependencyDown (MissingScore) / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-012:call_1(reset:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-012:call_3(holder_history:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-012:call_last(closed_cases_like:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D2:HHF-012:get_transaction:timeout@1` | INCOMPLETE / ToolTimeout / no <case>.json | INCOMPLETE / ToolTimeout / saved=False | **PASS** |
| `D3:HHF-012:injected_garbage_string` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-012:wrong_columns` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-012:nan_amounts` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-012:negative_ids` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D4:HHF-012:sql_injection_card_id` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-012:sql_in_trigger_text_is_data` | COMPLETE / verdict=legitimate (text never executed) | COMPLETE / verdict=legitimate | **PASS** |
| `D4:HHF-012:negative_risk_score` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-012:unknown_trigger_type` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-012:huge_trigger_text_100k` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-012:nonexistent_flagged_txn` | INCOMPLETE / KeyError / no <case>.json | INCOMPLETE / KeyError / saved=False | **PASS** |
| `D4:HHF-012:unicode_prompt_injection` | COMPLETE / verdict=legitimate (prompt ignored) | COMPLETE / verdict=legitimate | **PASS** |
| `D4:HHF-012:card_id_mismatch_resolved` | COMPLETE / resolved to txn truth / verdict=uncertain (DQ-07) | COMPLETE / verdict=uncertain | **PASS** |
| `D5:HHF-012:wrong_role_L1_on_L2` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-012:agent_token_approval` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-012:role_in_request_body` | HTTP 403 + body tamper logged + status=pending | HTTP 403 + tamper_logged=True + status=pending | **PASS** |
| `D5:HHF-012:replayed_decision_idempotency` | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | **PASS** |
| `D6:HHF-012:same_case_twice_identical_bytes` | COMPLETE/COMPLETE & identical canonical JSON bytes | COMPLETE/COMPLETE & sha256_match=True (3457 bytes) | **PASS** |
| `D7:HHF-012:exception_after_tmp_write` | OSError raised / no <case>.json / no 'saved' status | OSError / final_exists=False | **PASS** |
| `D8:HHF-012:hash_chain_tamper_detection` | verify() valid=False with SHA256 Hash mismatch | valid=False (Hash mismatch line 0: computed a06fe4ee7...) | **PASS** |
| `D9:HHF-012:no_external_llm_on_decision_path` | 0 network/LLM imports in investigate modules | forbidden_imports=0 | **PASS** |
| `D9:HHF-012:local_lgbm_missing_score_fail_closed` | INCOMPLETE / DependencyDown (MissingScore) / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-003:call_1(reset:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-003:call_3(holder_history:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D1:HHF-003:call_last(closed_cases_like:down@1)` | INCOMPLETE / DependencyDown / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |
| `D2:HHF-003:get_transaction:timeout@1` | INCOMPLETE / ToolTimeout / no <case>.json | INCOMPLETE / ToolTimeout / saved=False | **PASS** |
| `D3:HHF-003:injected_garbage_string` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-003:wrong_columns` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-003:nan_amounts` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D3:HHF-003:negative_ids` | INCOMPLETE / GarbageResponse / no <case>.json | INCOMPLETE / GarbageResponse / saved=False | **PASS** |
| `D4:HHF-003:sql_injection_card_id` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-003:sql_in_trigger_text_is_data` | COMPLETE / verdict=fraud (text never executed) | COMPLETE / verdict=fraud | **PASS** |
| `D4:HHF-003:negative_risk_score` | REJECTED_INPUT / InvalidInput / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-003:unknown_trigger_type` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-003:huge_trigger_text_100k` | REJECTED_INPUT\|INCOMPLETE / ValueError / no <case>.json | REJECTED_INPUT / InvalidInput / saved=False | **PASS** |
| `D4:HHF-003:nonexistent_flagged_txn` | INCOMPLETE / KeyError / no <case>.json | INCOMPLETE / KeyError / saved=False | **PASS** |
| `D4:HHF-003:unicode_prompt_injection` | COMPLETE / verdict=fraud (prompt ignored) | COMPLETE / verdict=fraud | **PASS** |
| `D4:HHF-003:card_id_mismatch_resolved` | COMPLETE / resolved to txn truth / verdict=fraud (DQ-07) | COMPLETE / verdict=fraud | **PASS** |
| `D5:HHF-003:wrong_role_L1_on_L2` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-003:agent_token_approval` | HTTP 403 / proposal status=pending | HTTP 403 / status=pending | **PASS** |
| `D5:HHF-003:role_in_request_body` | HTTP 403 + body tamper logged + status=pending | HTTP 403 + tamper_logged=True + status=pending | **PASS** |
| `D5:HHF-003:replayed_decision_idempotency` | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | same_key=200, mutated=409, new_key=409, ledger_decisions=1 | **PASS** |
| `D6:HHF-003:same_case_twice_identical_bytes` | COMPLETE/COMPLETE & identical canonical JSON bytes | COMPLETE/COMPLETE & sha256_match=True (11590 bytes) | **PASS** |
| `D7:HHF-003:exception_after_tmp_write` | OSError raised / no <case>.json / no 'saved' status | OSError / final_exists=False | **PASS** |
| `D8:HHF-003:hash_chain_tamper_detection` | verify() valid=False with SHA256 Hash mismatch | valid=False (Hash mismatch line 0: computed 38d2a9d8d...) | **PASS** |
| `D9:HHF-003:no_external_llm_on_decision_path` | 0 network/LLM imports in investigate modules | forbidden_imports=0 | **PASS** |
| `D9:HHF-003:local_lgbm_missing_score_fail_closed` | INCOMPLETE / DependencyDown (MissingScore) / no <case>.json | INCOMPLETE / DependencyDown / saved=False | **PASS** |

## Detailed Drill Evidence Table

| Drill | Category | Case | Subcase | Result | Status | Error Type | Files Written | False 'Saved'? | Error / Evidence |
|---|---|---|---|---|---|---|---|---|---|
| D1 | store_down | HHF-007 | call_1(reset:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Injected fault: store method 'reset' is down |
| D1 | store_down | HHF-007 | call_3(holder_history:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Injected fault: store method 'holder_history' is down |
| D1 | store_down | HHF-007 | call_last(closed_cases_like:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Injected fault: store method 'closed_cases_like' is down |
| D2 | store_timeout | HHF-007 | get_transaction:timeout@1 | **PASS** | `INCOMPLETE` | `ToolTimeout` | `HHF-007.incomplete.json` | No | Method 'get_transaction' timed out after 0.15s |
| D3 | garbage_return | HHF-007 | injected_garbage_string | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-007.incomplete.json` | No | Method 'get_transaction' return value failed validation |
| D3 | garbage_return | HHF-007 | wrong_columns | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-007.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D3 | garbage_return | HHF-007 | nan_amounts | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-007.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D3 | garbage_return | HHF-007 | negative_ids | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-007.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D4 | hostile_input | HHF-007 | sql_injection_card_id | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-007.incomplete.json` | No | Hostile or invalid input detected matching pattern 'DROP\b' |
| D4 | hostile_input | HHF-007 | sql_in_trigger_text_is_data | **PASS** | `COMPLETE` | `None` | `HHF-007.json` | No | verdict=fraud (clean baseline=fraud) |
| D4 | hostile_input | HHF-007 | negative_risk_score | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-007.incomplete.json` | No | risk_score -0.45 is not a number in [0, 1] |
| D4 | hostile_input | HHF-007 | unknown_trigger_type | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-007.incomplete.json` | No | Unknown trigger_type 'martian_telepathy'; allowed=['analyst_request', 'customer_report', 'ri... |
| D4 | hostile_input | HHF-007 | huge_trigger_text_100k | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-007.incomplete.json` | No | trigger_text longer than 4096 characters |
| D4 | hostile_input | HHF-007 | nonexistent_flagged_txn | **PASS** | `INCOMPLETE` | `KeyError` | `HHF-007.incomplete.json` | No | KeyError: 'Flagged transaction 999999999 does not exist in store' |
| D4 | hostile_input | HHF-007 | unicode_prompt_injection | **PASS** | `COMPLETE` | `None` | `HHF-007.json` | No | verdict=fraud (matches clean baseline=fraud) |
| D4 | hostile_input | HHF-007 | card_id_mismatch_resolved | **PASS** | `COMPLETE` | `None` | `HHF-007.json` | No | resolved to txn truth card=C09582-K1; verdict=fraud; cleared=False |
| D5 | approval_security | HHF-007 | wrong_role_L1_on_L2 | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals |
| D5 | approval_security | HHF-007 | agent_token_approval | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | Agent identity cannot approve decisions |
| D5 | approval_security | HHF-007 | role_in_request_body | **PASS** | `HTTP_403` | `BodyTamperLogged_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals (tamper_logged=True) |
| D5 | approval_security | HHF-007 | replayed_decision_idempotency | **PASS** | `HTTP_200_idempotent_409_conflict` | `ReplayProtected` | `ledger.jsonl` | No | same_key=200, mutated_body=409, new_key_after_decide=409, ledger_decisions=1 |
| D6 | idempotent_rerun | HHF-007 | same_case_twice_identical_bytes | **PASS** | `COMPLETE/COMPLETE` | `None` | `HHF-007.json` | No | sha256=a4174acc26695d20... (4407 bytes identical across both runs) |
| D7 | atomic_write_kill | HHF-007 | exception_after_tmp_write | **PASS** | `ABORTED_WRITE` | `OSError` | `(none)` | No | Simulated kill/power-loss after tmp write before os.replace \| final_json_exists=False |
| D8 | ledger_tamper | HHF-007 | hash_chain_tamper_detection | **PASS** | `TAMPER_DETECTED` | `LedgerIntegrityError` | `ledger.jsonl` | No | Hash mismatch line 0: computed 59fdcfcec58755804bff29b58c24c0af8e6d5561e5e0cc2c90f1e6ce778a9... |
| D9 | model_provider_failure | HHF-007 | no_external_llm_on_decision_path | **PASS** | `VERIFIED_NO_EXTERNAL_MODEL` | `None` | `(none)` | No | No external LLM/network libs imported by ['assay.investigate', 'assay.r2.investigate_r2']; i... |
| D9 | model_provider_failure | HHF-007 | local_lgbm_missing_score_fail_closed | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Method 'get_transaction' failed with exception: txn 3660350 has no model score (run B1: assa... |
| D1 | store_down | HHF-012 | call_1(reset:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-012.incomplete.json` | No | Injected fault: store method 'reset' is down |
| D1 | store_down | HHF-012 | call_3(holder_history:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-012.incomplete.json` | No | Injected fault: store method 'holder_history' is down |
| D1 | store_down | HHF-012 | call_last(closed_cases_like:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-012.incomplete.json` | No | Injected fault: store method 'closed_cases_like' is down |
| D2 | store_timeout | HHF-012 | get_transaction:timeout@1 | **PASS** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | Method 'get_transaction' timed out after 0.15s |
| D3 | garbage_return | HHF-012 | injected_garbage_string | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-012.incomplete.json` | No | Method 'get_transaction' return value failed validation |
| D3 | garbage_return | HHF-012 | wrong_columns | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-012.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D3 | garbage_return | HHF-012 | nan_amounts | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-012.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D3 | garbage_return | HHF-012 | negative_ids | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-012.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D4 | hostile_input | HHF-012 | sql_injection_card_id | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-012.incomplete.json` | No | Hostile or invalid input detected matching pattern 'DROP\b' |
| D4 | hostile_input | HHF-012 | sql_in_trigger_text_is_data | **PASS** | `COMPLETE` | `None` | `HHF-012.json` | No | verdict=legitimate (clean baseline=legitimate) |
| D4 | hostile_input | HHF-012 | negative_risk_score | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-012.incomplete.json` | No | risk_score -0.45 is not a number in [0, 1] |
| D4 | hostile_input | HHF-012 | unknown_trigger_type | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-012.incomplete.json` | No | Unknown trigger_type 'martian_telepathy'; allowed=['analyst_request', 'customer_report', 'ri... |
| D4 | hostile_input | HHF-012 | huge_trigger_text_100k | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-012.incomplete.json` | No | trigger_text longer than 4096 characters |
| D4 | hostile_input | HHF-012 | nonexistent_flagged_txn | **PASS** | `INCOMPLETE` | `KeyError` | `HHF-012.incomplete.json` | No | KeyError: 'Flagged transaction 999999999 does not exist in store' |
| D4 | hostile_input | HHF-012 | unicode_prompt_injection | **PASS** | `COMPLETE` | `None` | `HHF-012.json` | No | verdict=legitimate (matches clean baseline=legitimate) |
| D4 | hostile_input | HHF-012 | card_id_mismatch_resolved | **PASS** | `COMPLETE` | `None` | `HHF-012.json` | No | resolved to txn truth card=C12223-K1; verdict=uncertain; cleared=False |
| D5 | approval_security | HHF-012 | wrong_role_L1_on_L2 | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals |
| D5 | approval_security | HHF-012 | agent_token_approval | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | Agent identity cannot approve decisions |
| D5 | approval_security | HHF-012 | role_in_request_body | **PASS** | `HTTP_403` | `BodyTamperLogged_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals (tamper_logged=True) |
| D5 | approval_security | HHF-012 | replayed_decision_idempotency | **PASS** | `HTTP_200_idempotent_409_conflict` | `ReplayProtected` | `ledger.jsonl` | No | same_key=200, mutated_body=409, new_key_after_decide=409, ledger_decisions=1 |
| D6 | idempotent_rerun | HHF-012 | same_case_twice_identical_bytes | **PASS** | `COMPLETE/COMPLETE` | `None` | `HHF-012.json` | No | sha256=02b52273960c5f4c... (3457 bytes identical across both runs) |
| D7 | atomic_write_kill | HHF-012 | exception_after_tmp_write | **PASS** | `ABORTED_WRITE` | `OSError` | `(none)` | No | Simulated kill/power-loss after tmp write before os.replace \| final_json_exists=False |
| D8 | ledger_tamper | HHF-012 | hash_chain_tamper_detection | **PASS** | `TAMPER_DETECTED` | `LedgerIntegrityError` | `ledger.jsonl` | No | Hash mismatch line 0: computed a06fe4ee7f9ddae7d3b6ee9ce87da8a6a9c0380e5a8fe9073a21ea34adaee... |
| D9 | model_provider_failure | HHF-012 | no_external_llm_on_decision_path | **PASS** | `VERIFIED_NO_EXTERNAL_MODEL` | `None` | `(none)` | No | No external LLM/network libs imported by ['assay.investigate', 'assay.r2.investigate_r2']; i... |
| D9 | model_provider_failure | HHF-012 | local_lgbm_missing_score_fail_closed | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-012.incomplete.json` | No | Method 'get_transaction' failed with exception: txn 3655457 has no model score (run B1: assa... |
| D1 | store_down | HHF-003 | call_1(reset:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-003.incomplete.json` | No | Injected fault: store method 'reset' is down |
| D1 | store_down | HHF-003 | call_3(holder_history:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-003.incomplete.json` | No | Injected fault: store method 'holder_history' is down |
| D1 | store_down | HHF-003 | call_last(closed_cases_like:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-003.incomplete.json` | No | Injected fault: store method 'closed_cases_like' is down |
| D2 | store_timeout | HHF-003 | get_transaction:timeout@1 | **PASS** | `INCOMPLETE` | `ToolTimeout` | `HHF-003.incomplete.json` | No | Method 'get_transaction' timed out after 0.15s |
| D3 | garbage_return | HHF-003 | injected_garbage_string | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-003.incomplete.json` | No | Method 'get_transaction' return value failed validation |
| D3 | garbage_return | HHF-003 | wrong_columns | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-003.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D3 | garbage_return | HHF-003 | nan_amounts | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-003.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D3 | garbage_return | HHF-003 | negative_ids | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-003.incomplete.json` | No | Method 'holder_history' return value failed validation |
| D4 | hostile_input | HHF-003 | sql_injection_card_id | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-003.incomplete.json` | No | Hostile or invalid input detected matching pattern 'DROP\b' |
| D4 | hostile_input | HHF-003 | sql_in_trigger_text_is_data | **PASS** | `COMPLETE` | `None` | `HHF-003.json` | No | verdict=fraud (clean baseline=fraud) |
| D4 | hostile_input | HHF-003 | negative_risk_score | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-003.incomplete.json` | No | risk_score -0.45 is not a number in [0, 1] |
| D4 | hostile_input | HHF-003 | unknown_trigger_type | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-003.incomplete.json` | No | Unknown trigger_type 'martian_telepathy'; allowed=['analyst_request', 'customer_report', 'ri... |
| D4 | hostile_input | HHF-003 | huge_trigger_text_100k | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-003.incomplete.json` | No | trigger_text longer than 4096 characters |
| D4 | hostile_input | HHF-003 | nonexistent_flagged_txn | **PASS** | `INCOMPLETE` | `KeyError` | `HHF-003.incomplete.json` | No | KeyError: 'Flagged transaction 999999999 does not exist in store' |
| D4 | hostile_input | HHF-003 | unicode_prompt_injection | **PASS** | `COMPLETE` | `None` | `HHF-003.json` | No | verdict=fraud (matches clean baseline=fraud) |
| D4 | hostile_input | HHF-003 | card_id_mismatch_resolved | **PASS** | `COMPLETE` | `None` | `HHF-003.json` | No | resolved to txn truth card=C11663-K1; verdict=fraud; cleared=False |
| D5 | approval_security | HHF-003 | wrong_role_L1_on_L2 | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals |
| D5 | approval_security | HHF-003 | agent_token_approval | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | Agent identity cannot approve decisions |
| D5 | approval_security | HHF-003 | role_in_request_body | **PASS** | `HTTP_403` | `BodyTamperLogged_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals (tamper_logged=True) |
| D5 | approval_security | HHF-003 | replayed_decision_idempotency | **PASS** | `HTTP_200_idempotent_409_conflict` | `ReplayProtected` | `ledger.jsonl` | No | same_key=200, mutated_body=409, new_key_after_decide=409, ledger_decisions=1 |
| D6 | idempotent_rerun | HHF-003 | same_case_twice_identical_bytes | **PASS** | `COMPLETE/COMPLETE` | `None` | `HHF-003.json` | No | sha256=e169fa6880eed031... (11590 bytes identical across both runs) |
| D7 | atomic_write_kill | HHF-003 | exception_after_tmp_write | **PASS** | `ABORTED_WRITE` | `OSError` | `(none)` | No | Simulated kill/power-loss after tmp write before os.replace \| final_json_exists=False |
| D8 | ledger_tamper | HHF-003 | hash_chain_tamper_detection | **PASS** | `TAMPER_DETECTED` | `LedgerIntegrityError` | `ledger.jsonl` | No | Hash mismatch line 0: computed 38d2a9d8d89f48458095869f6d23ad95053fb83671105d1d2fe059f2ef6c2... |
| D9 | model_provider_failure | HHF-003 | no_external_llm_on_decision_path | **PASS** | `VERIFIED_NO_EXTERNAL_MODEL` | `None` | `(none)` | No | No external LLM/network libs imported by ['assay.investigate', 'assay.r2.investigate_r2']; i... |
| D9 | model_provider_failure | HHF-003 | local_lgbm_missing_score_fail_closed | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-003.incomplete.json` | No | Method 'get_transaction' failed with exception: txn 3660153 has no model score (run B1: assa... |
