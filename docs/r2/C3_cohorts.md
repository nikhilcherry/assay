# Block C3 Results -- Cohort Detectors (`MicroDripCohort` & `FanIn`)

All numbers below are measured directly by `pytest tests/r2/test_cohorts.py` over `assay.r2.frames` (`tx_all()`, `closed_cases()`, `case_pack()`). Neither detector reads `TransactionID` ranges or thresholds as a feature.

---

## 1. `MicroDripCohort` (`assay/r2/detectors/micro_charge.py`)

- **Detector identifiers**: `name = "micro_drip_cohort"`, `pattern = "undocumented:sub_threshold_micro_drip"` (`MicroCharge` is preserved intact).
- **Per-card new micro-drip definition (as-of $t \le \text{opened\_at}$)**:
  1. Online (`channel == 'online'`), `ProductCD == 'C'`, `amount < $6.50`.
  2. $\ge 4$ such micro charges inside a 30-day window `[t_start, min(t_start + 30d, opened_at)]`.
  3. Span between first and last micro charge in the window is $\ge 7.0$ days (and explicitly $> 1$ hour, excluding pure card-testing bursts under Rule R5).
  4. **Zero** such micro charges in the 60 days prior to `t_start` (`[t_start - 60d, t_start)`).
- **Monthly distribution of cards qualifying for a new micro-drip (`monthly_new_drip_distribution`)**:

| Month | By Start Month | By Qualification Month | Sample Qualifying Cards |
|---|---:|---:|---|
| `2016-07` | 2 | 1 | `C12412-K1` |
| `2016-08` | 4 | 4 | `C00253-K1, C01564-K1, C03060-K1, C10637-K1` |
| `2016-09` | 1 | 2 | `C08135-K2, C08935-K1` |
| `2016-10` | 2 | 2 | `C07960-K2, C11859-K1` |
| `2016-11` | 5 | 2 | `C11548-K1, C11653-K1` |
| `2016-12` | 18 | 11 | `C01598-K1, C01715-K1, C04646-K1, C05543-K1, C06040-K1, C06431-K2...` |
| `2017-01` | 2 | 19 | `C00348-K1, C01029-K1, C01105-K1, C01357-K1, C03097-K1, C04513-K1...` |

- **Choice of $K = 6$ other cards ($\ge 7$ total cards in overlapping 30-day window)**:
  - Across Aug–Nov 2016, the monthly count of new-drip cards is **1 to 5 by start month** (and **2 to 4 by qualification month**), and the maximum number of active overlapping new-drip cards in any daily 30-day window across Oct–Nov 2016 is **5 cards**.
  - Setting $K = 6$ other cards ($\ge 7$ total cards) yields **0 false fires** across all 61 daily windows in Oct–Nov 2016 while capturing the December–January coordinated campaign (11 cards qualified in Dec, 19 in Jan).
- **As-of Backtest & LR Measurement (`LR_RULE`)**:
  - Evaluated over **2,621** Jul–Dec confirmed-fraud closed cases (with $\ge 90\text{d}$ history), **524** cleared closed cases, **5,000** never-cased legitimate card-days (`seed=42`), and **90** daily 30-day windows (`2016-10-03` to `2016-12-31`):
  - **Per-card new micro-drip alone**: fires on `5 / 2,621` confirmed-fraud cases (`0.19%`), `1 / 524` cleared cases (`0.19%`), and `5 / 5,000` never-cased legitimate card-days (`0.10%`) — confirming per-card micro-drips are a weak signal.
  - **Cohort detector ($K \ge 6$ other cards)**: fires on **0 / 2,621** confirmed-fraud cases, **0 / 524** cleared cases, and **0 / 5,000** legitimate card-days (`lr_laplace = 1.9069`, 95% CI `[0.1192, 30.4996]`). In daily window scans, **0 / 59** windows in Oct–Nov fired; **18 / 31** windows in Dec had $\ge 7$ active cards as the unlabelled Dec–Jan campaign began.
  - Because the cohort form fired 0 times on historical closed fraud cases, `lr` is set to **`1.0`** (never an invented ratio), carrying `connected_card_ids`, `affected_txn_ids`, and all cohort transaction IDs in the attestation for Policy §3a / R9.

---

## 2. `FanIn` (`assay/r2/detectors/fan_in.py`)

- **Removed unmeasured `lr = 3.0` assumption**: `FanIn` never emits `lr = 3.0` or `_lr_is_assumption`.
- **Tightened Recipient Cohort Definition**:
  1. The flagged transaction **itself** (`TransactionID == flagged_txn_id`) must pay the recipient (`channel == 'online'`, `ProductCD == 'C'`, non-freemail `R_emaildomain`, and `amount >= $100.0`).
  2. Cohort cards are cards whose **first** charge to that `R_emaildomain` falls in `[opened_at - 21d, opened_at]` with `amount >= $100.0`.
- **Justification of the `$100.0` Amount Floor from Historical Data (`analyze_amount_floor_distribution`)**:

| Jul–Dec Online `ProductCD=='C'` Non-Freemail Band | Txns (Jul–Dec) | Cards (Jul–Dec) | Confirmed Fraud Txns | Cleared / Uncased Txns | Fraud Rate |
|---|---:|---:|---:|---:|---:|
| `< $30` | 817 | 428 | 49 | 768 | `6.00%` |
| `$30 - $70` (biller band) | 864 | 540 | 57 | 807 | `6.60%` |
| `$70 - $100` | 314 | 238 | 21 | 293 | `6.69%` |
| `>= $100` (floor) | 389 | 266 | 42 | 347 | `10.80%` |

  - In Jul–Dec history, `83.9%` (`1,995 / 2,384`) of online `ProductCD == 'C'` non-freemail charges are under `$100` (routine small billers/telecoms with `93.6%` cleared/uncased share), whereas charges `>= $100` have nearly double the confirmed-fraud rate (`10.80%` vs `6.37%`).
  - Applied to `sc.rr.com` in January 2017, `min_amount = 100.0` cleanly separates:
    - **`< $100` legitimate biller group**: 8 txns across **4 cards** (`C00043-K1, C01564-K2, C08312-K1, C09628-K1`), amounts `$32.63 – $66.77`.
    - **`>= $100` fan-in cohort**: 38 txns across **13 cards** (`12` first-time cards in the 21d window `[2017-01-07, 2017-01-28]` plus `C06115-K1` whose first charge was `2017-01-05`), amounts `$103.52 – $382.38`.
- **As-of Backtest & LR Measurement (`LR_RULE`)**:
  - Over **2,621** Jul–Dec confirmed-fraud cases, **524** cleared cases, **5,000** never-cased legitimate card-days (`seed=42`), and **90** daily 21d windows:
  - Fired **0 / 2,621** times on confirmed fraud, **0 / 524** times on cleared cases, **0 / 5,000** times on legitimate card-days, and **0 / 90** historical daily windows (`lr_laplace = 1.9069`, 95% CI `[0.1192, 30.4996]`).
  - Therefore `FanIn` sets `lr = 1.0` (never `3.0`), attaching `connected_card_ids` and attested cohort transaction IDs for Policy §3a / R6.

---

## 3. Exam Pack Results (`HHF-001 .. HHF-018`)

| Case | `micro_drip_cohort` | `FanIn` | Cohort Size | Connected Cards (`connected_card_ids`) | Affected Txns (`affected_txn_ids`) |
|---|---|---|---:|---|---|
| **HHF-004** (`C12975-K2`) | **FIRES** (`lr=1.00`) | False | **18** (`17` other) | `17` cards (`C00348-K1, C01029-K1, C01105-K1, C01357-K1, C03097-K1, ...`) | `[3660255..3660260]` (6 txns, `93` cohort txns attested) |
| **HHF-005** (`C00348-K1`) | **FIRES** (`lr=1.00`) | False | **18** (`17` other) | `17` cards (`C01029-K1, C01105-K1, C01357-K1, C03097-K1, C04513-K1, ...`) | `[3660245..3660250]` (6 txns, `93` cohort txns attested) |
| **HHF-006** (`C01727-K2`) | False | **FIRES** (`lr=1.00`, `kind='evidence'`) | **12** (`11` other) | `11` cards (`C00534-K2, C01357-K1, C01507-K1, C01594-K1, C02167-K2, ...`) | `[3660261, 3660262, 3660263]` (3 txns, `29` cohort txns attested) |
| **HHF-013** (`C06115-K1`) | False | **Context only** (`lr=1.00`, `kind='context'`) | **12** | `12` cards (context note only; flagged txn `3626720` is in-person `W` `$409.30`) | `[]` |
| **All other 14 cases** | False | False | 0 | `[]` | `[]` |
