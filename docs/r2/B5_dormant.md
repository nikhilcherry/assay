# Block B5 Results -- Dormant Card Wake-Up Detector

## Overview
Rebuilt `DormantWake(Detector)` in `assay/r2/detectors/dormant.py` implementing pattern `undocumented:dormant_card_wake`.
Addressed false positive root cause: removed `holder` UID silence drift artifact and enforced dormancy strictly per `card_id` using prior transaction history on that card.

## Unbiased Parameter Sweep & Evaluation (Jul-Oct 2016 Baseline)
Evaluated grid sweep $N \in \{14, 21, 30, 45, 60\}$ days silence, $k \in \{2, 3, 4, 5\}$ txns burst, and $m \in \{2.0, 3.0, 5.0, 10.0\}$ amount multiplier.
Evaluated against 4,675 confirmed fraud cases and a sample of 5,000 legitimate card-days from Jul-Oct 2016 (seed=42).

- **Constraint**: Legitimate fire rate $\le 2.0\%$
- **Chosen Optimal Setting**: $N = 14$, $k = 2$, $m = 5.0$
- **Confirmed Fraud Caught**: 184 / 4,675 (3.94%)
- **Legitimate Sample Fired**: 34 / 5,000 (0.68%)
- **Laplace-Smoothed Likelihood Ratio**: $LR = 5.65$
- **95% Bootstrap CI**: $[3.92, 8.15]$ (1,000 iterations, fixed seed 42)

## Attestation & Reproducibility
- Attestation query: `"dormant_wake"`
- Parameters: `{"card_id": ..., "as_of": ..., "N": 14, "k": 2, "m": 5.0}`
- Re-run function `rerun(params)` in module reproduces exact evidence transaction IDs. Verified in pytest suite.

## Exam Cases Evaluation (18 Cases)
Ran chosen setting against the 18 final-round cases in `case_pack_final.csv`.
Fired on exactly **1 case**: `HHF-003` (Card `C11663-K1`).
Card-level silence on all other exam cases is $< 5$ days, confirming those cards were active and not dormant.

### HHF-003 Episode Analysis (Card C11663-K1)
- **Wake-up Trigger**: 21.8 days of card silence followed by a 48h burst of 2 online transactions (`3660147`, `3660148`).
- **January Transaction Log**:
  - `3660147`: $181.73 (ProductCD `W`, in_person)
  - `3660148`: $357.82 (ProductCD `W`, in_person)
  - `3660153`: $281.33 (ProductCD `W`, in_person, flagged transaction)
- **Calculated Exposure USD**: $820.88 across all 3 episode transactions.
