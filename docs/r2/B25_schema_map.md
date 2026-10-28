# Block B25 — Someone Else's Schema (Challenge 13)

## Overview
`assay/r2/schema_map.py` separates the 32 canonical `tx_all()` columns into:
- **`SOURCE_COLS` (27 columns)**: `TransactionID`, `ts`, `card_id`, `customer_id`, `card4`, `card6`, `amount`, `ProductCD`, `channel`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `R_emaildomain`, `risk_score`, `DeviceType`, `DeviceInfo`, `id_30`, `id_31`, `id_33`, `id_15`, `id_23`, `M4`, `M5`, `M6`, `D1`, `C1`. Matched via distributional/regex fingerprints, composite synthesis (`concat_card_id`, `concat_date_time`, `onehot_channel`, `split_pipe`), and Hungarian assignment (`scipy.optimize.linear_sum_assignment`). Any pipe-composed column (`any_pipe > 0.01`) is blocked from directly matching `DeviceInfo`, `id_30`, `id_31`, or `id_33`.
- **`DERIVED_COLS` (5 columns)**: `TransactionDT`, `t`, `month_src`, `device_profile`, `uid`. Recomputed after source mapping by the shared pipeline function `recompute_derived_columns()` using the exact `assay.r2.frames.stage()` code path (`pd.to_datetime(ts)` for `t`, `assay.store.device_profile` on raw `(DeviceInfo, id_30, id_31, id_33)` for `device_profile`, and `TransactionDT` either mapped from the variant, joined back through `TransactionID` on `frames.tx_all()`, or marked honestly as `"not recoverable from this variant"` when neither exists).

## Safety & Governance Controls
1. **Confidence Floor (`confidence_floor=0.55`)**: Any source column candidate scoring below the confidence floor is refused (`mapping.refused_below_floor`) and its foreign column is listed in `mapping.unmapped_foreign` rather than guessed.
2. **Honest Unrecoverable Gaps (`NOT_RECOVERABLE = "not recoverable from this variant"`)**: Because `ts` and `TransactionDT` do not share a constant offset in the bank's data, if a variant carries only a timestamp and `TransactionID` cannot be joined back to `tx_all()`, `TransactionDT` and `uid` are marked `"not recoverable from this variant"` in the proposal (`mapping.not_recoverable`) rather than fabricated.
3. **Proposal File & Role-Checked Approval Gate (`approve(proposal_id, approver_role)`)**: `infer_mapping()` writes a JSON proposal (`SMP-<hash>.json`) containing every choice, its confidence, sample `TransactionID` justifications, and `Attestation`. `apply(..., require_proposal_approval=True)` blocks execution until an authorized `L1` or `L2` role approves the proposal.

## Measured Results (20,000-row sample of `tx_all()`, seed=42)
| Variant | Transform Summary | `SOURCE_COLS` Recovered | `DERIVED_COLS` Exact Match | Decoy Columns Ignored |
|---|---|---|---|---|
| `variant_1_renamed_cents_epoch` | Opaque renamed + shuffled columns, amounts in cents (`×100`), `ts` in epoch seconds, 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
| `variant_2_split_card_datetime_onehot_packed_dev` | `card_id` split into customer + suffix, `ts` split into `date` + `time` columns, `channel` one-hot encoded, raw device columns unpacked from `device_profile`, 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
| `variant_3_separate_identity_table` | Identity columns in second table with renamed FK (`fk_tx_ref`), amounts in cents, direct `TransactionDT` column, 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
| `variant_4_full_combined_stress` | Split tables + split `card_id` + one-hot `channel` + cents amounts + epoch-ms timestamps + 4 decoys | 27 / 27 (100.0%) | 5 / 5 (100.0%) | 4 / 4 |
