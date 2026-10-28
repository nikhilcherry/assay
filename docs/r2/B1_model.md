# B1 -- January 2017 Transaction Fraud Model (`model_r2`) & Drift Analysis

## 1. Monthly Label Completeness Analysis

| Month | Cases Opened | Fraud Cases | Cleared Cases | Total Txns | Named Txns | Share Named (%) | Confirmed Fraud Txns | Confirmed Fraud Rate (%) | Complete Labels? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2016-07 | 1,515 | 1,000 | 515 | 130,295 | 3,761 | 2.8865% | 3,245 | 2.4905% | `True` (fully_labelled) |
| 2016-08 | 1,341 | 1,211 | 130 | 93,786 | 3,878 | 4.1349% | 3,748 | 3.9963% | `True` (fully_labelled) |
| 2016-09 | 1,337 | 1,226 | 111 | 92,903 | 3,891 | 4.1882% | 3,779 | 4.0677% | `True` (fully_labelled) |
| 2016-10 | 1,347 | 1,203 | 144 | 100,420 | 3,425 | 3.4107% | 3,283 | 3.2693% | `True` (fully_labelled) |
| 2016-11 | 31 | 28 | 3 | 84,805 | 21 | 0.0248% | 21 | 0.0248% | `False` (partially_labelled) |
| 2016-12 | 14 | 7 | 7 | 88,533 | 101 | 0.1141% | 101 | 0.1141% | `False` (partially_labelled) |
| 2017-01 | 0 | 0 | 0 | 60,196 | 0 | 0.0000% | 0 | 0.0000% | `False` (unlabelled_exam) |

**Training Set Decision:**
- **July–October 2016 (`2016-07` .. `2016-10`) are the only completely labelled months:** 2.88% to 4.19% of all transactions each month are named in closed cases, and the confirmed fraud rate is 3.37% overall (2.49% in Jul, 4.00% in Aug, 4.07% in Sep, 3.27% in Oct), matching the underlying population fraud rate.
- **November–December 2016 (`2016-11` .. `2016-12`) are incomplete:** Only 45 closed cases exist by `opened_at` (31 in Nov, 14 in Dec), covering just 0.0248% of Nov transactions and 0.1141% of Dec transactions. Unnamed Nov–Dec transactions are uninvestigated, not confirmed legitimate.
- **Decision:** We train strictly on the completely labelled months (**July–October 2016**; Jul–Sep for out-of-time evaluation on October 2016, and all four fully labelled months Jul–Oct for the production scorer).

## 2. Out-of-Time Evaluation on Latest Fully-Labelled Month (October 2016)

- **Holdout Split:** Train `2016-07 .. 2016-09` (`n=316,984`), Evaluate OOT on `2016-10` (`n=100,420`, fraud rate `3.2690%`).
- **Expected Calibration Error (ECE, 10 bins):** `0.00069` (requirement: `< 0.05`).

| Model Variant (October 2016 OOT) | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| `model_r2` (`without_risk_score` - production) | **0.9206** | **0.4936** | **0.0215** |
| `model_r2` (`with_risk_score` ablation) | 0.9641 | 0.6349 | 0.0161 |
| Bank `risk_score` baseline (same October rows) | 0.8660 | 0.2521 | 0.0544 |

### 10-Bin Calibration Table (October 2016 2-Fold Cross-Fitted Isotonic)

| Probability Bin | Count | Mean Predicted `p` | Observed Fraud Rate | Absolute Gap |
|---|---:|---:|---:|---:|
| `(-0.001, 0.1]` | 93,527 | 0.01126 | 0.01133 | 0.00008 |
| `(0.1, 0.2]` | 3,601 | 0.14369 | 0.14746 | 0.00377 |
| `(0.2, 0.3]` | 790 | 0.24334 | 0.23797 | 0.00536 |
| `(0.3, 0.4]` | 342 | 0.34160 | 0.30409 | 0.03750 |
| `(0.4, 0.5]` | 367 | 0.44844 | 0.46594 | 0.01750 |
| `(0.5, 0.6]` | 167 | 0.56123 | 0.56886 | 0.00763 |
| `(0.6, 0.7]` | 1,028 | 0.65537 | 0.64591 | 0.00946 |
| `(0.7, 0.8]` | 269 | 0.74533 | 0.75836 | 0.01304 |
| `(0.8, 0.9]` | 315 | 0.83189 | 0.80317 | 0.02871 |
| `(0.9, 1.0]` | 14 | 0.99900 | 0.92857 | 0.07043 |

## 3. Feature Drift Report (`Jul-Oct` Training vs `Jan 2017`, Top-25 Features by Gain)

- **Flagged Top-25 Features (`PSI > 0.25`):** `['D1n', 'uid_amt_std', 'card_id_n', 'uid_n']` (`4/25` features).
- **January In-Distribution Rate (`score(jan_ids).in_distribution.mean()`):** `1.0670%` (Training Jul-Oct In-Distribution Rate: `0.9090%`).
- **How `in_distribution` is defined and used:** A transaction has `in_distribution = False` when it falls into a severely shifted bucket (`PSI contrib > 0.05` and `>1.5x` density shift) of any top-25 feature with `PSI > 0.25`, or exceeds the 99th-percentile training Mahalanobis distance across the top numeric features (`d2 > 85.04`). Downstream investigators use `in_distribution = False` to pull the model's likelihood ratio toward `1.0` (down-weighting model confidence on drifted/unseen activity), never to flip a verdict.

| Rank | Feature | Type | Gain | Oct Single AUC | Jul-Oct vs Jan PSI | Flagged (`PSI > 0.25`) | Train Null % | Jan Null % |
|---:|---|---|---:|---:|---:|---|---:|---:|
| 1 | `V257` | `numeric` | 37,583.3 | 0.6647 | **0.0027** | `False` | 75.70% | 77.30% |
| 2 | `D1n` | `numeric` | 33,344.5 | 0.6186 | **2.2558** | `True` | 0.07% | 0.04% |
| 3 | `uid_amt_mean` | `numeric` | 26,268.1 | 0.5355 | **0.0662** | `False` | 11.52% | 8.29% |
| 4 | `C14` | `numeric` | 25,100.7 | 0.6138 | **0.0035** | `False` | 0.00% | 0.00% |
| 5 | `V258` | `numeric` | 24,731.2 | 0.6656 | **0.0027** | `False` | 75.70% | 77.30% |
| 6 | `uid_amt_std` | `numeric` | 24,610.3 | 0.5644 | **0.7825** | `True` | 32.19% | 73.77% |
| 7 | `DeviceInfo` | `categorical` | 21,421.0 | 0.6213 | **0.0042** | `False` | 77.98% | 77.54% |
| 8 | `V294` | `numeric` | 20,485.1 | 0.5996 | **0.0015** | `False` | 0.00% | 0.05% |
| 9 | `V243` | `numeric` | 19,725.1 | 0.6626 | **0.0031** | `False` | 75.70% | 77.30% |
| 10 | `C13` | `numeric` | 19,370.5 | 0.6408 | **0.0069** | `False` | 0.00% | 0.00% |
| 11 | `card_id_amt_mean` | `numeric` | 18,216.8 | 0.5288 | **0.0330** | `False` | 0.00% | 0.00% |
| 12 | `card_id_n` | `numeric` | 18,042.3 | 0.5306 | **0.7376** | `True` | 0.00% | 0.00% |
| 13 | `card_id_amt_std` | `numeric` | 17,913.4 | 0.5386 | **0.1168** | `False` | 0.67% | 0.00% |
| 14 | `uid_n` | `numeric` | 17,811.5 | 0.6002 | **1.7469** | `True` | 11.52% | 8.29% |
| 15 | `card1` | `numeric` | 15,539.7 | 0.5029 | **0.0311** | `False` | 0.00% | 0.00% |
| 16 | `card2` | `numeric` | 15,247.9 | 0.5066 | **0.0396** | `False` | 1.57% | 1.60% |
| 17 | `addr1_freq` | `numeric` | 14,844.4 | 0.6537 | **0.0203** | `False` | 11.46% | 8.26% |
| 18 | `C1` | `numeric` | 14,440.0 | 0.5736 | **0.0009** | `False` | 0.00% | 0.00% |
| 19 | `C11` | `numeric` | 13,541.4 | 0.5561 | **0.0015** | `False` | 0.00% | 0.00% |
| 20 | `D2` | `numeric` | 13,327.4 | 0.6174 | **0.0116** | `False` | 49.85% | 47.83% |
| 21 | `addr1` | `numeric` | 13,144.1 | 0.6080 | **0.0173** | `False` | 11.46% | 8.26% |
| 22 | `C8` | `numeric` | 12,569.5 | 0.6749 | **0.0000** | `False` | 0.00% | 0.00% |
| 23 | `R_emaildomain` | `categorical` | 12,364.7 | 0.6449 | **0.0109** | `False` | 75.15% | 79.04% |
| 24 | `TransactionAmt` | `numeric` | 11,730.1 | 0.5154 | **0.0498** | `False` | 0.00% | 0.00% |
| 25 | `V187` | `numeric` | 11,165.3 | 0.6735 | **0.0048** | `False` | 74.31% | 77.07% |

## 4. Score PSI (Dec -> Jan) & Comparison vs Round-1 Model (`scores_r2.parquet`)

- **B1 `p_fraud` (`scores_b1.parquet`) Dec -> Jan PSI:** `0.00334`
- **Bank `risk_score` Dec -> Jan PSI:** `0.10162`
- **Round-1 baseline (`scores_r2.parquet`) Jan mean `p_fraud`:** `0.01961` vs **B1 (`scores_b1.parquet`) Jan mean `p_fraud`:** `0.02988`
- **Jan `p_fraud` Pearson correlation (B1 vs R1):** `0.86118` (Spearman: `0.72727`)

## 5. Final Case Pack (`HHF-001` .. `HHF-018`) Flagged Transaction Scores

| Case ID | Trigger Type | Flagged Txn ID | Card ID | Bank `risk_score` | R1 `p_fraud` | B1 `p_fraud` (`p`) | `in_distribution` | Mahalanobis `d2` |
|---|---|---:|---|---:|---:|---:|---|---:|
| HHF-001 | `analyst_request` | 3660108 | `C05061-K1` | 0.0800 | 0.1892 | **0.1845** | `False` | 8.73 |
| HHF-002 | `analyst_request` | 3660127 | `C06220-K1` | 0.0700 | 0.0714 | **0.1162** | `False` | 11.02 |
| HHF-003 | `customer_report` | 3660153 | `C11663-K1` | 0.0900 | 0.0010 | **0.0044** | `False` | 27.17 |
| HHF-004 | `customer_report` | 3660260 | `C12975-K2` | 0.0400 | 0.0010 | **0.0044** | `False` | 30.84 |
| HHF-005 | `customer_report` | 3660250 | `C00348-K1` | 0.0400 | 0.0010 | **0.0050** | `False` | 19.01 |
| HHF-006 | `analyst_request` | 3660263 | `C01727-K2` | 0.1500 | 0.1517 | **0.4549** | `False` | 7.45 |
| HHF-007 | `customer_report` | 3660350 | `C09582-K1` | 0.1900 | 0.0071 | **0.0060** | `False` | 13.05 |
| HHF-008 | `risk_score` | 3660353 | `C07871-K1` | 0.1100 | 0.0057 | **0.0249** | `False` | 35.38 |
| HHF-009 | `risk_score` | 3660355 | `C09177-K1` | 0.1000 | 0.0102 | **0.0498** | `False` | 7.11 |
| HHF-010 | `risk_score` | 3631383 | `C07196-K2` | 0.0700 | 0.0330 | **0.0189** | `False` | 20.35 |
| HHF-011 | `risk_score` | 3640919 | `C09241-K1` | 0.3100 | 0.0102 | **0.0062** | `False` | 26.44 |
| HHF-012 | `risk_score` | 3655457 | `C12223-K1` | 0.0500 | 0.0010 | **0.0077** | `False` | 8.92 |
| HHF-013 | `risk_score` | 3626720 | `C06115-K1` | 0.1100 | 0.0528 | **0.0636** | `False` | 12.50 |
| HHF-014 | `risk_score` | 3602445 | `C00750-K2` | 0.0600 | 0.0010 | **0.0044** | `False` | 14.96 |
| HHF-015 | `risk_score` | 3601725 | `C00543-K1` | 0.0700 | 0.0010 | **0.0033** | `False` | 23.10 |
| HHF-016 | `risk_score` | 3611043 | `C02502-K2` | 0.1300 | 0.0036 | **0.0033** | `False` | 21.82 |
| HHF-017 | `risk_score` | 3638829 | `C08842-K2` | 0.0100 | 0.0010 | **0.0019** | `False` | 9.25 |
| HHF-018 | `risk_score` | 3612011 | `C02753-K1` | 0.0300 | 0.0010 | **0.0010** | `False` | 15.29 |
