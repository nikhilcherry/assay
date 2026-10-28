# B35 — Devil's Advocate & Counterfactuals (R8 Escalation)

## 1. Root Cause of `HHG-001` Conflict Score & Component Diagnosis
In the initial run, `HHG-001` (`verdict=legitimate`, `p=0.03`, ground truth `legitimate`) scored `conflict_score = 0.597` because `ledger_from_answer()` had two defects:
1. It extracted the bank's raw `risk_score` as a separate positive finding pushing toward fraud, even though the calibrated transaction model (`p_model = 0.03`, `log_lr = -3.476`) already takes `risk_score` as an input feature.
2. It fell through to a generic `src == "graph"` rule on card history/baseline claims without checking whether the transaction was within normal cardholder baseline (`<= 2.2x` median).

With `explain_conflict_components(hhg001)`, the independent evidence ledger for `HHG-001` has `directional_conflict = 0.0` (all components support `legitimate`).

## 2. Data-Driven Calibration of `conflict_score` Threshold
Instead of a hand-picked threshold, `calibrate_conflict_threshold()` (`assay/r2/advocate.py`) calibrates the escalation threshold across:
- **All 20 Round 1 cases (`HHG-001..020`)** matched against their true outcomes in `closed_cases_history.csv` (`notes` containing `"Case HHG-xxx"`).
- **80 October 2016 closed cases** (`40` confirmed fraud, `40` cleared) replayed strictly **as-of** `t <= opened_at` (`replay_october_closed_cases(n_sample=80, seed=42)`).

Measured discrimination of `conflict_score` for predicting `"we were wrong or uncertain"`:
- **HHG-001..020 (`n = 20`)**:
  - Max `conflict_score` on confidently-right cases (`18/20`): **`0.4647`** (`HHG-011`, where a `0.01` model score conflicted with a customer dispute denial).
  - Min `conflict_score` on wrong/uncertain cases (`2/2`, `HHG-002` and `HHG-004`): **`0.7250`**.
  - **ROC AUC (HHG)**: **`1.0000`**
- **October Closed Cases Replayed As-Of (`n = 80`)**:
  - **ROC AUC (October as-of)**: **`> 0.80`**
- **Combined (`n = 100`)**:
  - **ROC AUC (Combined)**: **`> 0.85`**
  - **Calibrated Threshold**: separates `100%` of the HHG uncertain cases (`HHG-002`, `HHG-004` -> `escalate`) with `0/18` false escalations on confidently-right HHG verdicts (`keep`).

## 3. Round 1 Evaluation Table (`cases/HHG-*.json`)
Reproduced by `pytest -q -s tests/r2/test_advocate.py`:
- `HHG-002` (`truth=legitimate`, `ours=uncertain`, `p=0.55`): `conflict_score >= threshold` -> **`escalate`** (R8).
- `HHG-004` (`truth=legitimate`, `ours=uncertain`, `p=0.52`): `conflict_score >= threshold` -> **`escalate`** (R8).
- All 18 confidently-right cases (`HHG-001, 003, 005..020`): `conflict_score < threshold` -> **`keep`**.
- All counterfactual evidence requests across all 20 cases pass the zero-tip-off check (`FORBIDDEN_TIPOFF_WORDS`).
