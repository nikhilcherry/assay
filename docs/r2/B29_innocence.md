# B29 — Innocence Prover: Measured Exculpatory Evidence

## Overview
`InnocenceEvidence` (`assay/r2/detectors/innocence.py`) evaluates the five exculpatory kinds (`recurring`, `home_region`, `known_device`, `stable_profile`, `long_tenure_clean`) using a single canonical evaluation path (`eval_single_case()`) across `calibrated_innocence_stats()`, `backtest()`, and `run(ctx)`.

## Why Round 2 Had 0/900 Cleared Hits & What Diagnostics Showed
In Round 2, `eval_single_case()` returned `{}` whenever a card had any prior closed case or any other transaction within 1 hour, and required `channel == "in_person"`, `addr1` share `>= 85%`, and `D1 >= 30` simultaneously. Restoring the clean prompt-specified definitions and running `sample_cleared_diagnostics(n=5, seed=42)` on 5 random cleared Jul–Oct cases explains the exact closed-case distribution:
- **Cleared cases sit on much thinner card histories than confirmed-fraud episodes**: As shown in the Round 1 output (`5,540` Jul–Oct closed cases: `4,640` `confirmed_fraud` vs `900` `cleared`), `cleared` cases fire on `recurring` (`~66/900 = 7.3%`), `home_region` (`~70/900 = 7.8%`), `stable_profile` (`~213/900 = 23.7%`), and `long_tenure_clean` (`36/900 = 4.0%`).
- **Why `LR_alert` (`P(kind | confirmed_fraud) / P(kind | cleared)`) is `>= 0.50` on all 5 kinds**: Because `card_id`s are issuer buckets (Shared Lesson 5), `confirmed_fraud` episodes are heavily concentrated on high-volume cards that easily satisfy `>= 10` prior transactions (`home_region` `23.0%` on fraud vs `7.8%` on cleared; `stable_profile` `53.4%` on fraud vs `23.7%` on cleared). Even `long_tenure_clean` (`>= 90` days of clean history with zero prior confirmed fraud) achieves `LR_alert = 0.7420` (`95% CI [0.5190, 1.0607]`), which does not pass the strict `ci_high_alert < 0.50` keep gate.
- **Honest Keep Gate**: Under `KEEP_CI_HIGH_MAX = 0.50` on the alert population, `kept = []` (`InnocenceEvidence.run(ctx)` emits no false-confidence exculpatory LRs into the alert ledger), while `InnocenceEvidence.run_all_kinds(ctx)` exposes all raw exculpatory findings (including Policy **R7** recurring pattern matches) for analyst inspection and rule routing.

## Policy Rule R7 Citation (`recurring`)
Under Fraud Operations Policy v1.0 **Rule R7 (Disputed but legitimate)**:
> *When the customer disputes a charge that matches their own recurring pattern (same merchant, same amount, monthly), recommend `CREATE_CASE`, `VERIFY_WITH_CUSTOMER`, and `WARN_CUSTOMER`. Do not block.*

Every `exculpatory:recurring` finding cites `R7` in `claim.what` and `claim.params["policy_rule"] = "R7"`.
