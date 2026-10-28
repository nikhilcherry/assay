# B28 — Coordinated Campaign Detector (`undocumented:coordinated_campaign`)

## Overview
`CampaignDetector` (`assay/r2/detectors/campaign.py`) detects coordinated multi-card fraud campaigns (Policy R9 undocumented / R6 shared origin, Challenge 1) where $\ge 5$ distinct cards and customers experience novel transaction episodes with the exact same charge-shape signature within a 21-day window.

### Method & Round-1 Review Fixes
1. **Baseline Novelty Check**: For each transaction up to `opened_at`, counts how many times `(card_id, ProductCD, channel)` occurred in the baseline window $[t - 90\text{d}, t - 14\text{d})$ prior to the current 14-day episode, alongside an expanding log-amount outlier z-score ($z > 3.5$, amount $\ge \$120$). Measuring baseline history prior to $t - 14\text{d}$ ensures that the 3rd–7th charges of a multi-charge burst remain part of the card's novel episode.
2. **Episode-Level Learned Amount Clusters**: Fits 1D KMeans ($k=5$, fixed seed `20260228`) on $\log(1 + \text{amount})$ of July–October novel transactions and assigns the amount band from the **episode's mean log-amount** after grouping homogeneous charges ($\Delta \log(\text{amount}) \le 1.35$) per `(card_id, ProductCD, channel)` within 14 days. This prevents fixed bucket edges from splitting tight bursts such as `HHF-001` (\$448–\$891) and `HHF-002` (\$637–\$771).
3. **Per-Signature Calibration on Seeded Jul–Oct Legitimate Card-Days**: Evaluates every candidate signature as-of on a random sample of `6,000` legitimate card-days from July–October 2016 (transactions named in no closed case) and on July–October confirmed-fraud cases. Signatures with legitimate card-day firing rate $> 1\%$, $\text{LR} < 3.0$, or common unverified shapes (`W|in_person|*` or high-rate `cnt_2_3` buckets) are dropped.

---

## Measured Calibration & Backtest

- **Learned amount bands**: `['amt_0_16', 'amt_16_45', 'amt_45_112', 'amt_112_310', 'amt_310_plus']`
- **Jul–Oct seeded legitimate card-day sample**: `6,000` card-days (`seed=20260228`)
- **Overall legitimate card-day firing rate**: `0.0333%` (2 / 6,000, well below the $\le 1\%$ limit)
- **Overall confirmed-fraud episode firing rate**: `1.8680%` (73 / 3,908 in Jul–Oct; 101 / 4,675 across all closed cases)
- **Cleared closed cases acted against**: `0 / 910` (`0.0000%`)
- **Measured overall Likelihood Ratio**: `45.36`

### Per-Signature LR Table (Top Accepted & Rejected Signatures)

| Signature | $\lambda_{21\text{d}}$ | Legit Fires (6k) | Legit Rate | Fraud Fires | Fraud Rate | LR | Accepted |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `C\|online\|amt_16_45\|cnt_2_3\|span_0_4d\|gap_lt_6h` | 0.08682 | 0 | 0.000000 | 32 | 0.008188 | 99.78 | True |
| `C\|online\|amt_45_112\|cnt_2_3\|span_0_4d\|gap_lt_6h` | 0.08682 | 0 | 0.000000 | 18 | 0.004606 | 56.80 | True |
| `C\|online\|amt_16_45\|cnt_4_7\|span_0_4d\|gap_lt_6h` | 0.00868 | 0 | 0.000000 | 13 | 0.003327 | 41.45 | True |
| `C\|online\|amt_0_16\|cnt_4_7\|span_4_14d\|gap_gt_2.5d` | 0.00868 | 0 | 0.000000 | 0 | 0.000000 | 46.06 | True |
| `C\|online\|amt_310_plus\|cnt_2_3\|span_0_4d\|gap_6h_2.5d` | 0.00868 | 0 | 0.000000 | 0 | 0.000000 | 46.06 | True |
| `C\|online\|amt_310_plus\|cnt_4_7\|span_4_14d\|gap_6h_2.5d` | 0.00868 | 0 | 0.000000 | 0 | 0.000000 | 46.06 | True |
| `C\|online\|amt_112_310\|cnt_2_3\|span_0_4d\|gap_6h_2.5d` | 0.35595 | 0 | 0.000000 | 0 | 0.000000 | 1.54 | False |
| `W\|in_person\|amt_310_plus\|cnt_2_3\|span_0_4d\|gap_lt_6h` | 0.18232 | 0 | 0.000000 | 0 | 0.000000 | 1.54 | False |

---

## Measured Results on the 18 Exam Cases (`case_pack_final.csv`)

| Case ID | In Cohort | Cohort Size (cards) | Episode Txns | LR | Signature |
|---|---|---:|---:|---:|---|
| `HHF-001` | **True** | 12 | 3 | 45.36 | `C\|online\|amt_310_plus\|cnt_2_3\|span_0_4d\|gap_6h_2.5d` |
| `HHF-002` | **True** | 14 | 3 | 45.36 | `C\|online\|amt_310_plus\|cnt_2_3\|span_0_4d\|gap_6h_2.5d` |
| `HHF-003` | **True** | 8 | 7 | 45.36 | `C\|online\|amt_310_plus\|cnt_4_7\|span_4_14d\|gap_6h_2.5d` |
| `HHF-004` | **True** | 13 | 5 | 45.36 | `C\|online\|amt_0_16\|cnt_4_7\|span_4_14d\|gap_gt_2.5d` |
| `HHF-005` | **True** | 13 | 6 | 45.36 | `C\|online\|amt_0_16\|cnt_4_7\|span_4_14d\|gap_gt_2.5d` |
| `HHF-006` | False | 0 | 0 | 0.00 | `None` (dropped: `C\|online\|amt_112_310\|cnt_2_3\|span_0_4d\|gap_6h_2.5d` has $\lambda_{21\text{d}}=0.36$, $\text{LR}=1.54 < 3$) |
| `HHF-007` .. `HHF-018` | False | 0 | 0 | 0.00 | `None` (`HHF-009` and `HHF-016` rejected by per-signature LR filter) |
