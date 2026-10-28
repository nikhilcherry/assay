# Block B25 — Someone Else's Schema (Challenge 13)

## Overview
`assay/r2/schema_map.py` separates the 32 canonical `tx_all()` columns into:
- **`SOURCE_COLS` (27 columns)**: `TransactionID`, `ts`, `card_id`, `customer_id`, `card4`, `card6`, `amount`, `ProductCD`, `channel`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `R_emaildomain`, `risk_score`, `DeviceType`, `DeviceInfo`, `id_30`, `id_31`, `id_33`, `id_15`, `id_23`, `M4`, `M5`, `M6`, `D1`, `C1`. Matched via distributional/regex fingerprints, composite synthesis (`concat_card_id`, `concat_date_time`, `onehot_channel`, `split_pipe`), and Hungarian assignment (`scipy.optimize.linear_sum_assignment`). Any pipe-composed column (`any_pipe > 0.01`) is blocked from directly matching `DeviceInfo`, `id_30`, `id_31`, or `id_33`.
- **`DERIVED_COLS` (5 columns)**: `TransactionDT`, `t`, `month_src`, `device_profile`, `uid`. Recomputed after source mapping by the shared pipeline function `recompute_derived_columns()` using the exact `assay.r2.frames.stage()` code path (`normalize_timestamp_series(ts)` for tz-naive UTC `t`, `assay.store.device_profile` on raw `(DeviceInfo, id_30, id_31, id_33)` for `device_profile`, and `TransactionDT` either mapped from the variant, joined back through `TransactionID` on `frames.tx_all()`, or marked honestly as `"not recoverable from this variant"` when neither exists).

## Red-vs-Blue Debate Attack Defences
1. **ATK-08 (`amount` in cents & `risk_score` on `0..100` scale)**: `_score_candidate()` detects `100x` scaling on both `amount` (`cents_to_usd`) and `risk_score` (`pct_to_unit`), emitting `/ 100.0` so amounts are in USD and `risk_score` is in `[0.0, 1.0]`.
2. **ATK-09 (Missing Identity Table + Split `card_id` + Split `ts`)**: Synthesizes `card_id` via `concat_card_id(cust, seq)` and `ts` via `concat_date_time(date, time)`. When the identity table is omitted entirely, marks the 7 identity columns and `device_profile` in `mapping.not_recoverable`, populates the missing identity columns with `None`, and sets `device_profile = ""` (empty string) so null devices never link unrelated cards into a false ring.
3. **ATK-10 (Timezone-Aware ISO Timestamps `'Z'` & `'+05:30'`)**: `_score_candidate()` detects ISO-8601 strings with timezone offsets (`tz_iso_to_ts`), and `normalize_timestamp_series()` converts them to tz-naive UTC `datetime64[ns]` so `self._tx.t <= t0` never raises a tz-comparison `TypeError` or drops the flagged transaction due to IST/UTC offset drift.
4. **All-Null & Extra Junk Columns**: Columns with `null_rate >= 0.999` are isolated in `mapping.all_null_foreign` and given `0.0` score against all canonical columns so they never match sparse fields (`id_23`, `id_33`, etc.).

## Measured Results (20,000-row sample of `tx_all()`, seed=42)
| Variant | Transform / Attack Summary | `SOURCE_COLS` Recovered | `DERIVED_COLS` Exact Match | Gaps / Decoys |
|---|---|---|---|---|
| `variant_1_renamed_cents_epoch` | Opaque renamed + shuffled columns, amounts in cents (`×100`), `ts` in epoch seconds, 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 decoys ignored |
| `variant_2_split_card_datetime_onehot_packed_dev` | `card_id` split, `ts` split into `date` + `time`, `channel` one-hot, raw device columns unpacked from `device_profile`, 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 decoys ignored |
| `variant_3_separate_identity_table` | Identity columns in second table (`fk_tx_ref`), amounts in cents, direct `TransactionDT` column, 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 decoys ignored |
| `variant_4_full_combined_stress` | Split tables + split `card_id` + one-hot `channel` + cents amounts + epoch-ms timestamps + 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 decoys ignored |
| `ATK-08` | Cents `amount` + `0..100` `risk_score` + 2 all-null columns + 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 2 all-null + 4 decoys |
| `ATK-09` | Missing identity table + split `card_id` + split `date`/`time` + 1 all-null + 4 decoys | 20 / 20 present (100.0%) | 4 / 4 (`device_profile=""`) | 8 honest gaps reported |
| `ATK-10` | Mixed `'Z'` and `'+05:30'` timezone-aware ISO timestamps + 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%, tz-naive UTC) | 4 decoys ignored |
