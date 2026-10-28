# Live Resilience Drills Scorecard (B30)

- **Cases tested:** HHF-007, HHF-012
- **Total drill assertions:** 46
- **Passed:** 33
- **Failed:** 13

## Diagnostic Findings in Harness / Approvals / Investigator

- harness.validate_case_row does not validate `trigger_type` against {'risk_score','customer_report','analyst_request'}; caught in investigator wrapper.
- harness.validate_case_row does not enforce a maximum length on `trigger_text` (100k char string accepted by validate_case_row); caught in investigator wrapper.
- harness.write_answer leaves temporary file (HHF-007.json.tmp.712e9250711549f5821485d50894086e) on disk if os.replace raises mid-write, though target `<case_id>.json` is never created and no 'saved' status is returned.
- harness.write_answer leaves temporary file (HHF-012.incomplete.json.tmp.047df11d14a54e7399168601a3dc3574) on disk if os.replace raises mid-write, though target `<case_id>.json` is never created and no 'saved' status is returned.

## Drill Results Table

| Drill | Category | Case | Subcase | Result | Status | Error Type | Files Written | False 'Saved'? | Error / Evidence |
|---|---|---|---|---|---|---|---|---|---|
| D1 | store_down | HHF-007 | call_1(reset:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Injected fault: store method 'reset' is down |
| D1 | store_down | HHF-007 | call_3(holder_history:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Injected fault: store method 'holder_history' is down |
| D1 | store_down | HHF-007 | call_last(closed_cases_like:down@1) | **PASS** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Injected fault: store method 'closed_cases_like' is down |
| D2 | store_timeout | HHF-007 | card_txns:timeout@1 | **PASS** | `INCOMPLETE` | `ToolTimeout` | `HHF-007.incomplete.json` | No | Method 'closed_cases_like' timed out after 0.15s |
| D3 | garbage_return | HHF-007 | injected_garbage_string | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-007.incomplete.json` | No | Method 'get_transaction' return value failed validation |
| D3 | garbage_return | HHF-007 | wrong_columns | **FAIL** | `COMPLETE` | `` | `HHF-007.json` | YES (BUG) | - |
| D3 | garbage_return | HHF-007 | nan_amounts | **FAIL** | `COMPLETE` | `` | `HHF-007.json` | YES (BUG) | - |
| D3 | garbage_return | HHF-007 | negative_ids | **FAIL** | `COMPLETE` | `` | `HHF-007.json` | YES (BUG) | - |
| D4 | hostile_input | HHF-007 | sql_injection | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-007.incomplete.json` | No | Hostile or invalid input detected matching pattern 'DROP\b' |
| D4 | hostile_input | HHF-007 | negative_risk_score | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-007.incomplete.json` | No | risk_score -0.45 out of range [0, 1] |
| D4 | hostile_input | HHF-007 | unknown_trigger_type | **PASS** | `INCOMPLETE` | `ValueError` | `HHF-007.incomplete.json` | No | ValueError: Unknown trigger_type 'martian_telepathy'; allowed=['analyst_request', 'customer_... |
| D4 | hostile_input | HHF-007 | huge_trigger_text_100k | **PASS** | `INCOMPLETE` | `ValueError` | `HHF-007.incomplete.json` | No | ValueError: Hostile oversized trigger_text (100000 chars > 4096 max) |
| D4 | hostile_input | HHF-007 | nonexistent_flagged_txn | **PASS** | `INCOMPLETE` | `KeyError` | `HHF-007.incomplete.json` | No | KeyError: 'Flagged transaction 999999999 does not exist in store' |
| D4 | hostile_input | HHF-007 | unicode_prompt_injection | **PASS** | `COMPLETE` | `None` | `HHF-007.json` | No | verdict=fraud (matches clean baseline) |
| D4 | hostile_input | HHF-007 | card_id_mismatch_resolved | **PASS** | `COMPLETE` | `None` | `HHF-007.json` | No | resolved to txn truth card=C09582-K1; verdict=fraud |
| D5 | approval_security | HHF-007 | wrong_role_L1_on_L2 | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals |
| D5 | approval_security | HHF-007 | agent_token_approval | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | Agent identity cannot approve decisions |
| D5 | approval_security | HHF-007 | role_in_request_body | **PASS** | `HTTP_403` | `BodyTamperLogged_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals (tamper_logged=True) |
| D5 | approval_security | HHF-007 | replayed_decision_idempotency | **PASS** | `HTTP_200_idempotent_409_conflict` | `ReplayProtected` | `ledger.jsonl` | No | same_key=200, mutated_body=409, new_key_after_decide=409, ledger_decisions=1 |
| D6 | idempotent_rerun | HHF-007 | same_case_twice_identical_bytes | **FAIL** | `COMPLETE/COMPLETE` | `None` | `HHF-007.json` | No | sha256=86b29b1f034e5d29... (4405 bytes identical across both runs) |
| D7 | atomic_write_kill | HHF-007 | exception_after_tmp_write | **PASS** | `ABORTED_WRITE` | `OSError` | `(none)` | No | Simulated kill/power-loss after tmp write before os.replace \| final_json_exists=False |
| D8 | ledger_tamper | HHF-007 | hash_chain_tamper_detection | **PASS** | `TAMPER_DETECTED` | `LedgerIntegrityError` | `ledger.jsonl` | No | Hash mismatch line 0: computed 1444321641e28e40066a090486fc61176c49f57a9fb30c5e79db10a978ae1... |
| D9 | model_provider_failure | HHF-007 | no_external_llm_on_decision_path | **PASS** | `VERIFIED_NO_EXTERNAL_MODEL` | `None` | `(none)` | No | No external LLM/network libs imported by ['assay.investigate', 'assay.r2.investigate_r2']; i... |
| D9 | model_provider_failure | HHF-007 | local_lgbm_missing_score_fail_closed | **FAIL** | `INCOMPLETE` | `DependencyDown` | `HHF-007.incomplete.json` | No | Method 'get_transaction' failed with exception: txn 3660350 has no model score (run B1: assa... |
| D1 | store_down | HHF-012 | baseline_discovery | **FAIL** | `INCOMPLETE` | `ToolTimeout` | `(none)` | No | Method 'closed_cases_like' timed out after 10.0s |
| D2 | store_timeout | HHF-012 | card_txns:timeout@1 | **PASS** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | Method 'closed_cases_like' timed out after 0.15s |
| D3 | garbage_return | HHF-012 | injected_garbage_string | **PASS** | `INCOMPLETE` | `GarbageResponse` | `HHF-012.incomplete.json` | No | Method 'get_transaction' return value failed validation |
| D3 | garbage_return | HHF-012 | wrong_columns | **FAIL** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | Method 'closed_cases_like' timed out after 5.0s |
| D3 | garbage_return | HHF-012 | nan_amounts | **FAIL** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | Method 'closed_cases_like' timed out after 5.0s |
| D3 | garbage_return | HHF-012 | negative_ids | **FAIL** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | Method 'closed_cases_like' timed out after 5.0s |
| D4 | hostile_input | HHF-012 | sql_injection | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-012.incomplete.json` | No | Hostile or invalid input detected matching pattern 'DROP\b' |
| D4 | hostile_input | HHF-012 | negative_risk_score | **PASS** | `REJECTED_INPUT` | `InvalidInput` | `HHF-012.incomplete.json` | No | risk_score -0.45 out of range [0, 1] |
| D4 | hostile_input | HHF-012 | unknown_trigger_type | **PASS** | `INCOMPLETE` | `ValueError` | `HHF-012.incomplete.json` | No | ValueError: Unknown trigger_type 'martian_telepathy'; allowed=['analyst_request', 'customer_... |
| D4 | hostile_input | HHF-012 | huge_trigger_text_100k | **PASS** | `INCOMPLETE` | `ValueError` | `HHF-012.incomplete.json` | No | ValueError: Hostile oversized trigger_text (100000 chars > 4096 max) |
| D4 | hostile_input | HHF-012 | nonexistent_flagged_txn | **PASS** | `INCOMPLETE` | `KeyError` | `HHF-012.incomplete.json` | No | KeyError: 'Flagged transaction 999999999 does not exist in store' |
| D4 | hostile_input | HHF-012 | unicode_prompt_injection | **FAIL** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | verdict=None (matches clean baseline) |
| D4 | hostile_input | HHF-012 | card_id_mismatch_resolved | **FAIL** | `INCOMPLETE` | `ToolTimeout` | `HHF-012.incomplete.json` | No | resolved to txn truth card=C12223-K1; verdict=None |
| D5 | approval_security | HHF-012 | wrong_role_L1_on_L2 | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals |
| D5 | approval_security | HHF-012 | agent_token_approval | **PASS** | `HTTP_403` | `HTTPException_403` | `ledger.jsonl` | No | Agent identity cannot approve decisions |
| D5 | approval_security | HHF-012 | role_in_request_body | **PASS** | `HTTP_403` | `BodyTamperLogged_403` | `ledger.jsonl` | No | L1 role cannot approve L2 route proposals (tamper_logged=True) |
| D5 | approval_security | HHF-012 | replayed_decision_idempotency | **PASS** | `HTTP_200_idempotent_409_conflict` | `ReplayProtected` | `ledger.jsonl` | No | same_key=200, mutated_body=409, new_key_after_decide=409, ledger_decisions=1 |
| D6 | idempotent_rerun | HHF-012 | same_case_twice_identical_bytes | **FAIL** | `INCOMPLETE/INCOMPLETE` | `None` | `HHF-012.json` | No | sha256=9ac2f03cb66dc440... (0 bytes identical across both runs) |
| D7 | atomic_write_kill | HHF-012 | exception_after_tmp_write | **PASS** | `ABORTED_WRITE` | `OSError` | `HHF-012.incomplete.json.tmp.047df11d14a54e7399168601a3dc3574` | No | Simulated kill/power-loss after tmp write before os.replace \| final_json_exists=False |
| D8 | ledger_tamper | HHF-012 | hash_chain_tamper_detection | **PASS** | `TAMPER_DETECTED` | `LedgerIntegrityError` | `ledger.jsonl` | No | Hash mismatch line 0: computed 6ea269884bbbcd1bcf76c9c8849fa446fcab2b2b2f5423549e9b9e271d188... |
| D9 | model_provider_failure | HHF-012 | no_external_llm_on_decision_path | **PASS** | `VERIFIED_NO_EXTERNAL_MODEL` | `None` | `(none)` | No | No external LLM/network libs imported by ['assay.investigate', 'assay.r2.investigate_r2']; i... |
| D9 | model_provider_failure | HHF-012 | local_lgbm_missing_score_fail_closed | **FAIL** | `INCOMPLETE` | `DependencyDown` | `HHF-012.incomplete.json` | No | Method 'get_transaction' failed with exception: txn 3655457 has no model score (run B1: assa... |
