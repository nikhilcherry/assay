# Block B25 — Someone Else's Schema (Challenge 13)

## Overview
`assay/r2/schema_map.py` separates the 32 canonical `tx_all()` columns into:
- **`SOURCE_COLS` (27 columns)**: `TransactionID`, `ts`, `card_id`, `customer_id`, `card4`, `card6`, `amount`, `ProductCD`, `channel`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `R_emaildomain`, `risk_score`, `DeviceType`, `DeviceInfo`, `id_30`, `id_31`, `id_33`, `id_15`, `id_23`, `M4`, `M5`, `M6`, `D1`, `C1`. Matched via distributional/regex fingerprints, composite synthesis (`concat_card_id`, `concat_date_time`, `onehot_channel`, `split_pipe`), and Hungarian assignment (`scipy.optimize.linear_sum_assignment`).
- **`DERIVED_COLS` (5 columns)**: `TransactionDT`, `t`, `month_src`, `device_profile`, `uid`. Recomputed after source mapping by the shared pipeline function `recompute_derived_columns()`.

## Learned `TransactionDT` Origin (`learn_dt_origin` & `canonical_dt_origin`)
No origin timestamp is hardcoded:
- When a foreign variant carries a seconds-offset column alongside a timestamp, `learn_dt_origin(ts, offset)` computes `origin = median(ts - offset)` over sample rows and asserts `spread == 0.0` seconds.
- When a foreign variant carries only a timestamp (ISO, date+time split, or epoch s/ms), `canonical_dt_origin()` learns the canonical origin from `frames.tx_all()` the exact same way (`median(t - TransactionDT)` = `2016-07-02 00:00:00`, spread `0.0s`).

## Safety & Governance Controls
1. **Confidence Floor (`confidence_floor=0.55`)**: Any source column candidate scoring below the confidence floor is refused (`mapping.refused_below_floor`) and its foreign column is listed in `mapping.unmapped_foreign` rather than guessed.
2. **Proposal File & Role-Checked Approval Gate (`approve(proposal_id, approver_role)`)**: `infer_mapping()` writes a JSON proposal (`SMP-<hash>.json`) containing every choice, its confidence, learned `dt_origin` + `dt_origin_spread_sec`, sample `TransactionID` justifications, and `Attestation`. `apply(..., require_proposal_approval=True)` blocks execution until an authorized `L1` or `L2` role approves the proposal.

## Measured Results (20,000-row sample of `tx_all()`, seed=42)
| Variant | Transform Summary | Learned `dt_origin` (spread) | `SOURCE_COLS` Recovered | `DERIVED_COLS` Exact Match | Decoy Columns Ignored |
|---|---|---|---|---|---|
| `variant_1_renamed_cents_epoch` | Opaque renamed + shuffled columns, amounts in cents (`×100`), `ts` in epoch seconds, 4 decoys | `2016-07-02T00:00:00` (`0.0s`) | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
| `variant_2_split_card_datetime_onehot_packed_dev` | `card_id` split into customer + suffix, `ts` split into `date` + `time` columns, `channel` one-hot encoded, raw device columns unpacked from `device_profile`, 4 decoys | `2016-07-02T00:00:00` (`0.0s`) | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
| `variant_3_separate_identity_table` | Identity columns in second table with renamed FK (`fk_tx_ref`), amounts in cents, custom seconds-offset column with shifted origin, 4 decoys | `2016-06-01T00:00:00` (`0.0s`) | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
| `variant_4_full_combined_stress` | Split tables + split `card_id` + one-hot `channel` + cents amounts + epoch-ms timestamps + 4 decoys | `2016-07-02T00:00:00` (`0.0s`) | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
