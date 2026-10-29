<!-- generated; do not edit -->
# C9 — Unknown-Fault Chaos Suite Results

- **Cases tested:** `HHF-007, HHF-005`
- **TigerGraph reachable at 127.0.0.1:14240:** `True`
- **Total scenarios:** 25 (21 PASS, 0 FAIL, 0 SKIPPED, 4 INVALID)

## Invariant Legend
- **(a)** `exit == 0` => complete valid `<case>.json` and no leftover `.tmp` or `.incomplete.json` artifacts.
- **(b)** `exit != 0` => NO `<case>.json` (or pre-existing stale `<case>.json` removed / never updated as fresh) and a clear one-line error.
- **(c)** Never a Python traceback as the only output.
- **(d)** Two concurrent runs of the same case => identical `<case>.json` bytes or one clean refusal.
- **(e)** Never `exit == 0` with a missing, empty, or partial `<case>.json`.

## Chaos Battery Table

| scenario | category | case | exit | invariant results | files after | first line | PASS/FAIL |
|---|---|---|---|---|---|---|---|
| `env_clean_baseline` | env_fault | `HHF-007` | `0` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(2), HHF-007.json, approvals_ledger.jso...` | HHF-007 status=COMPLETE [NEW_RUN] verdict=fraud p=0.91 pattern=card_testing exposure=$246.38... | **PASS** |
| `env_assay_fault_midrun_shared_device_down` | env_fault | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=DependencyDown tool=holder_history: Injected fa... | **PASS** |
| `env_assay_fault_get_txn_garbage` | env_fault | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=GarbageResponse tool=get_transaction: Validator... | **PASS** |
| `env_assay_tg_unreachable_port` | env_fault | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=TigerGraphUnavailableError: Fail-closed: TigerG... | **PASS** |
| `sc_os_replace_enospc` | sitecustomize | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=OSError: required data/dependency unavailable o... | **PASS** |
| `sc_time_jump_backwards` | sitecustomize | `HHF-007` | `0` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(2), HHF-007.json, approvals_ledger.jso...` | HHF-007 status=COMPLETE [NEW_RUN] verdict=fraud p=0.91 pattern=card_testing exposure=$246.38... | **PASS** |
| `sc_slow_call_past_timeout` | sitecustomize | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=ToolTimeout tool=reset: Method 'reset' timed ou... | **PASS** |
| `sc_tg_nth_query_raises` | sitecustomize_tg | `HHF-005` | `0` | `INVALID` | `.HHF-005.lock, .ledger/(2), HHF-005.json, approvals_ledger.jso...` | jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id) | **INVALID** |
| `sc_tg_nth_query_empty_dict` | sitecustomize_tg | `HHF-005` | `0` | `INVALID` | `.HHF-005.lock, .ledger/(2), HHF-005.json, approvals_ledger.jso...` | jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id) | **INVALID** |
| `sc_tg_nth_query_empty_list` | sitecustomize_tg | `HHF-005` | `0` | `INVALID` | `.HHF-005.lock, .ledger/(2), HHF-005.json, approvals_ledger.jso...` | jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id) | **INVALID** |
| `sc_tg_nth_query_malformed_json` | sitecustomize_tg | `HHF-005` | `0` | `INVALID` | `.HHF-005.lock, .ledger/(2), HHF-005.json, approvals_ledger.jso...` | jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id) | **INVALID** |
| `fs_readonly_out_dir` | filesystem | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `(none)` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=PermissionError: required data/dependency unava... | **PASS** |
| `fs_out_dir_is_regular_file` | filesystem | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `(none)` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=FileExistsError: required data/dependency unava... | **PASS** |
| `fs_stale_case_json_before_fault` | filesystem | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 error_type=DependencyDown tool=get_transaction: Injected f... | **PASS** |
| `fs_stale_tmp_file_before_clean_run` | filesystem | `HHF-007` | `0` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(2), HHF-007.json, approvals_ledger.jso...` | HHF-007 status=COMPLETE [NEW_RUN] verdict=fraud p=0.91 pattern=card_testing exposure=$246.38... | **PASS** |
| `proc_sigterm_mid_run` | process | `HHF-007` | `3` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | FAIL CLOSED [EXIT 3] case=HHF-007 stopped by SIGTERM before completing; no answer written | **PASS** |
| `proc_sigkill_mid_run` | process | `HHF-007` | `-9` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(0), audit.jsonl` | killed by SIGKILL (rc=-9) | **PASS** |
| `proc_concurrent_same_case_same_dir` | process | `HHF-007` | `0/0` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(2), HHF-007.json, approvals_ledger.jso...` | rc1=0, rc2=0 \| HHF-007 status=COMPLETE [IDEMPOTENT_REPLAY (byte-identical)] verdict=fraud p... | **PASS** |
| `inp_homoglyph_zero_width_case_id` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.invalid_1341785536.lock, audit.jsonl` | ERROR [EXIT 2] Invalid case input: Invalid case_id 'ННF-​007': path characters, traversal, o... | **PASS** |
| `inp_path_traversal_case_id` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.invalid_c5f3886dfa.lock, audit.jsonl` | ERROR [EXIT 2] Invalid case input: Invalid case_id '../../etc/HHF-007': path characters, tra... | **PASS** |
| `inp_crlf_in_structured_fields` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, audit.jsonl` | ERROR [EXIT 2] Case validation failed for 'HHF-007': Malformed card_id/customer_id: 'C09582-K1 | **PASS** |
| `inp_1mb_trigger_text` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, audit.jsonl` | ERROR [EXIT 2] Case validation failed for 'HHF-007': trigger_text longer than 4096 characters | **PASS** |
| `inp_nan_risk_score` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, audit.jsonl` | ERROR [EXIT 2] Case validation failed for 'HHF-007': risk_score 'NaN' out of range: not a nu... | **PASS** |
| `inp_inf_risk_score` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, audit.jsonl` | ERROR [EXIT 2] Case validation failed for 'HHF-007': risk_score 'Infinity' out of range: not... | **PASS** |
| `inp_duplicate_case_rows_in_pack` | input | `HHF-007` | `2` | `a=ok b=ok c=ok d=ok e=ok` | `.HHF-007.lock, .ledger/(2), HHF-007.json, approvals_ledger.jso...` | ERROR [EXIT 2] Case ID collision detected on disk for 'HHF-007': stored key=ba877cae2d2a != ... | **PASS** |

## Diagnostic Findings (Failed / Invalid Scenarios)

- **`sc_tg_nth_query_raises`** (`HHF-005`, status `INVALID`, exit `0`): invariants `a=ok b=ok c=ok d=ok e=ok` — first_line=`jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)` notes=`proof-of-injection marker missing (not_fired); full_stderr='/home/nikhil/Projects/assay/assay/r2/store_r2.py:91: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n  jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)\n'`
- **`sc_tg_nth_query_empty_dict`** (`HHF-005`, status `INVALID`, exit `0`): invariants `a=ok b=ok c=ok d=ok e=ok` — first_line=`jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)` notes=`proof-of-injection marker missing (not_fired); full_stderr='/home/nikhil/Projects/assay/assay/r2/store_r2.py:91: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n  jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)\n'`
- **`sc_tg_nth_query_empty_list`** (`HHF-005`, status `INVALID`, exit `0`): invariants `a=ok b=ok c=ok d=ok e=ok` — first_line=`jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)` notes=`proof-of-injection marker missing (not_fired); full_stderr='/home/nikhil/Projects/assay/assay/r2/store_r2.py:91: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n  jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)\n'`
- **`sc_tg_nth_query_malformed_json`** (`HHF-005`, status `INVALID`, exit `0`): invariants `a=ok b=ok c=ok d=ok e=ok` — first_line=`jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)` notes=`proof-of-injection marker missing (not_fired); full_stderr='/home/nikhil/Projects/assay/assay/r2/store_r2.py:91: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n  jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)\n'`
