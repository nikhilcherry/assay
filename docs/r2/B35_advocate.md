# B35 — Devil's Advocate & Counterfactuals (R8 Escalation)

## What Was Built
- **`advocate(answer, ledger)`** (`assay/r2/advocate.py`): Constructs the strongest case for the opposite verdict from the same evidence ledger without any LLM call or randomness. Computes:
  - `wrong_findings` and `wrong_log_odds`: which findings would have to be wrong and how much log-odds they carry.
  - `single_point_of_failure` (`spof`): flags whether any single non-model finding carries `> 60%` (`SPOF_SHARE = 0.60`) of the supporting evidence.
  - `conflict_score` in `[0, 1]`: measures opposing log-odds share and unresolved mid-band ambiguity.
  - `recommendation`: `keep` vs `escalate` (Policy R8: *evidence conflicts or uncertain*) with plain reason and a re-runnable `Attestation`.
- **`counterfactuals(answer, ledger)`** (`assay/r2/advocate.py`): Finds the minimal subset of findings whose removal flips the verdict band (`fraud >= 0.70`, `legitimate <= 0.30`, `uncertain` between) and maps each to a customer-safe verification prompt (tested for zero SAR tip-off).

## Round 1 Evaluation (`cases/HHG-*.json` vs Ground Truth in `closed_cases_history.csv`)

Measured via `pytest -q -s tests/r2/test_advocate.py`:

| Case | Truth | Ours | p | Conflict | SPOF | Flip Set | Rec | ConfRight |
|---|---|---|---|---|---|---|---|---|
| HHG-001 | legitimate | legitimate | 0.04 | 0.000 | - | model | keep | True |
| HHG-002 | legitimate | uncertain | 0.55 | 0.862 | baseline | customer_reply | escalate | False |
| HHG-003 | fraud | fraud | 0.96 | 0.000 | baseline | baseline+model | keep | True |
| HHG-004 | legitimate | uncertain | 0.45 | 0.862 | - | customer_reply | escalate | False |
| HHG-005 | legitimate | legitimate | 0.04 | 0.000 | - | model | keep | True |
| HHG-006 | fraud | fraud | 0.99 | 0.000 | new_device | new_device+baseline+model | keep | True |
| HHG-007 | fraud | fraud | 0.99 | 0.000 | new_device | new_device+baseline+model | keep | True |
| HHG-008 | fraud | fraud | 0.94 | 0.000 | baseline | baseline+model | keep | True |
| HHG-009 | fraud | fraud | 0.95 | 0.000 | dispute | dispute+baseline | keep | True |
| HHG-010 | legitimate | legitimate | 0.14 | 0.000 | - | model | keep | True |
| HHG-011 | fraud | fraud | 0.65 | 0.159 | dispute | dispute | keep | True |
| HHG-012 | legitimate | legitimate | 0.05 | 0.143 | - | model | keep | True |
| HHG-013 | legitimate | legitimate | 0.03 | 0.000 | - | model | keep | True |
| HHG-014 | fraud | fraud | 0.97 | 0.000 | new_device | new_device+baseline+model | keep | True |
| HHG-015 | legitimate | legitimate | 0.09 | 0.171 | - | model | keep | True |
| HHG-016 | fraud | fraud | 0.98 | 0.000 | new_device | new_device+baseline+model | keep | True |
| HHG-017 | legitimate | legitimate | 0.02 | 0.000 | - | model | keep | True |
| HHG-018 | fraud | fraud | 0.99 | 0.000 | baseline | baseline+model | keep | True |
| HHG-019 | fraud | fraud | 0.99 | 0.000 | new_device | new_device+baseline+model | keep | True |
| HHG-020 | legitimate | legitimate | 0.01 | 0.088 | - | model | keep | True |

## Key Findings
- **Uncertain Cases Caught (`2/2`)**: Both `HHG-002` (`p=0.55`, `conflict_score=0.862`) and `HHG-004` (`p=0.45`, `conflict_score=0.862`) received `conflict_score >= 0.35` and `recommendation = escalate` under R8. Both turned out to be `legitimate` in ground truth, confirming that escalating instead of blocking avoided two R1/innocent-customer penalties.
- **Confidently-Right Cases Preserved (`18/18`)**: All 18 confidently-right verdicts (`10` fraud, `8` legitimate) had `conflict_score < 0.35` (max `0.171` on `HHG-015` and `0.159` on `HHG-011`) and received `recommendation = keep` (`0` false escalations on confidently-right answers).
