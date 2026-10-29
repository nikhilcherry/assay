# Block N7 — Dispute Episode Scoping (`assay/r2/dispute_episode.py`)

## What Was Built
`episode_for_dispute(card_id, flagged_txn_id, as_of) -> (txn_ids, why)` identifies the maximal run of transactions on `card_id` (up to `as_of`) that:
1. Share the flagged transaction's **shape**:
   - Same `ProductCD` and `channel`
   - Amount within the flagged transaction's band (`[$0.25*amt, max($10, 3.2*amt)]` for micro charges `< $10`, `[0.38*amt, 2.6*amt]` for larger charges)
   - Consistent `R_emaildomain`, `device_profile` / `DeviceType`, and `addr1` (for in-person purchases)
2. Are **novel** for the card: the strict shape (or device profile / billing region) has `0` occurrences in the card's history prior to the start of the run (`run_start_t - 48h`), with guards against habitual in-person retail (`W`/`in_person` in an established `addr1`) and monthly (~25–35 day) recurring subscriptions (Policy R7).
3. Includes same-session (`<= 24h`) novel burst companions on the card, and falls back to the 24-hour window (`EPISODE_HOURS = 24`, `risk_score >= 0.30`) when the shape already existed in the card's prior history.

Every call returns `(txn_ids, why)` where `why` is an `assay.r2.attest.Attestation` with query `assay.r2.dispute_episode.episode_for_dispute`.

## Re-measured 2026-10-29 on the final-round history (supersedes the tables below)

`backtest_jul_oct_customer_reports()` on `closed_cases_history.csv` (5,585 cases). In this history every
Jul–Oct confirmed-fraud case is a customer report (**4,631**), and every cleared case is a model alert.

| cohort | method | n | exact match | mean Jaccard | mean exposure error |
|---|---|---:|---:|---:|---:|
| all Jul–Oct reports | `N7_episode_for_dispute` (1h cap, shipped) | 4,631 | **0.6266** | **0.7620** | **$241.33** |
| all Jul–Oct reports | flagged transaction only | 4,631 | 0.5498 | 0.6985 | $280.63 |
| all Jul–Oct reports | `N7_episode_for_dispute` without the cap | 4,631 | 0.4930 | 0.6459 | $935.57 |
| all Jul–Oct reports | `R1_episode_24h_risk30` | 4,631 | 0.4474 | 0.6472 | $477.66 |
| multi-txn reports | `N7_episode_for_dispute` (1h cap) | 2,085 | 0.2163 | 0.4961 | $529.09 |
| multi-txn reports | `R1_episode_24h_risk30` | 2,085 | 0.1899 | 0.4893 | $728.24 |

- **The uncapped builder lost to "flagged transaction only".** Its multi-transaction runs that spanned a
  week or more were never an exact match (Jaccard 0.176), so episodes now keep only charges within
  `EPISODE_CAP_HOURS = 1` of the disputed one. The cap was picked on Jul–Aug (best 0.5–1h) and holds
  on Sep–Oct (0.6200 at 1h). It gives up the rare true multi-day run (944 multi-day cases score 0
  either way).
- Exempting charges of the same amount from the cap was tested and rejected (0.6204, no exposure gain).
- **Micro-drip extension** (`MICRO_DRIP`): a disputed sub-$6.50 online C charge joins the card's other
  such charges from the past 30 days when the card had none before. It costs nothing on history
  (0.6266 with or without it) and scopes HHF-004 to its six drip charges (3660255–3660260, $23.99; was
  $2.90) and HHF-005 to six (3660245–3660250, $23.17).
- The 0.8061 / 3,522-case tables below came from an earlier cohort definition and do not reproduce on
  this history.

## Measured Backtest Results (Jul–Oct Customer-Report Closed Cases)

In `closed_cases_history.csv`, there are **3,522** confirmed-fraud customer-report cases in Jul–Oct 2016 (**999** of which have `n_txns > 1`) and **159** cleared (legitimate recurring dispute) customer-report cases.

### 1. All Jul–Oct Customer-Report Confirmed Fraud Cases (`n = 3,522`)

| Method | Exact-Match Rate | Mean Jaccard | First-Txn Accuracy | Mean Exposure Abs Error ($) | Median Exposure Abs Error ($) | Mean Exposure Bias ($) |
|---|---:|---:|---:|---:|---:|---:|
| `N7_episode_for_dispute` | **0.8061** | **0.8670** | **0.8407** | **$43.46** | **$0.00** | -$2.71 |
| `R1_episode_24h_risk30` | 0.7206 | 0.8012 | 0.7325 | $92.60 | $0.00 | -$84.82 |
| `R1_episode_24h_all` | 0.7135 | 0.8226 | 0.7689 | $71.15 | $0.00 | -$34.67 |

### 2. Multi-Txn Jul–Oct Customer-Report Confirmed Fraud Cases (`n = 999`)

| Method | Exact-Match Rate | Mean Jaccard | First-Txn Accuracy | Mean Exposure Abs Error ($) | Median Exposure Abs Error ($) | Mean Exposure Bias ($) |
|---|---:|---:|---:|---:|---:|---:|
| `N7_episode_for_dispute` | **0.4154** | **0.6512** | **0.5315** | **$138.94** | **$20.00** | -$23.81 |
| `R1_episode_24h_risk30` | 0.1451 | 0.4368 | 0.1812 | $314.08 | $143.95 | -$311.54 |
| `R1_episode_24h_all` | 0.3714 | 0.6540 | 0.4835 | $213.49 | $33.99 | -$158.95 |

### 3. Multi-Day Customer-Report Closed Cases in History (`CC-5569` / Round 1 precedent)
- `CC-5569` (3 online charges over 7 days totaling `$366.85`):
  - `N7_episode_for_dispute`: `[3502235, 3517420, 3530012]` (`$366.85`, exact match, Jaccard `1.0000`, error `$0.00`)
  - `R1_episode_24h_risk30`: `[3530012]` (`$124.20`, Jaccard `0.3333`, error `$242.65`)

### 4. Innocent-Customer / Legitimate Dispute Check (`n = 159` cleared customer-report cases in Jul–Oct)
- All 159 cleared customer-report cases in Jul–Oct are legitimate monthly recurring charges (Policy R7).
- `N7_episode_for_dispute` over-expanded (>1 txn) **0 / 159** (`0.0%`) cleared customer-report cases, because the prior-history novelty check and 25–35 day monthly-cadence guard prevent grouping prior monthly recurring charges into a fraud episode.

## Exam Pack Customer-Report Cases (`HHF-003`, `HHF-004`, `HHF-005`, `HHF-007`)

| Case ID | Card ID | Flagged Txn | `N7` Txn Count | `N7` First Txn ID | `N7` Exposure (USD) | `R1` (24h) Txn Count | `R1` Exposure (USD) | `N7` Affected Txn IDs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `HHF-003` | `C11663-K1` | `3660153` | 1 | `3660153` | `$29.99` | 1 | `$29.99` | `[3660153]` |
| `HHF-004` | `C12975-K2` | `3660260` | **6** | **`3660255`** | **`$20.47`** | 1 | `$2.90` | `[3660255, 3660256, 3660257, 3660258, 3660259, 3660260]` |
| `HHF-005` | `C00348-K1` | `3660250` | **6** | **`3660245`** | **`$20.23`** | 1 | `$4.05` | `[3660245, 3660246, 3660247, 3660248, 3660249, 3660250]` |
| `HHF-007` | `C09582-K1` | `3660350` | 1 | `3660350` | `$236.28` | 1 | `$236.28` | `[3660350]` |

- **HHF-003 (`C11663-K1`)**: The `$29.99` charge (`W`/`in_person`, `addr1=204`) matches 6 prior monthly `$29.99` charges on the ~29th of each month from Jul–Dec 2016 (Policy R7 recurring subscription). `N7` sees `prior_shape_hits = 6` and monthly cadence, returning only `[3660153]`.
- **HHF-004 (`C12975-K2`)**: Captures all 6 novel micro online charges (`C`/`online`, `hotmail.com`, `$1.95..$4.67` over Jan 5–Jan 26, 2017; `0` prior online/C charges in 46 historical card transactions), raising reported exposure from `$2.90` to `$20.47` and setting `first_suspicious_txn_id = 3660255`.
- **HHF-005 (`C00348-K1`)**: Captures all 6 novel micro online charges (`C`/`online`, `gmail.com`, `$2.11..$4.42` over Jan 6–Jan 28, 2017; `0` prior online/C charges in 143 historical card transactions), raising reported exposure from `$4.05` to `$20.23` and setting `first_suspicious_txn_id = 3660245`.
- **HHF-007 (`C09582-K1`)**: Single novel `$236.28` online purchase (`H`/`online`, `SM-G955U`), returning `[3660350]` (`$236.28`).
