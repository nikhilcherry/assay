# Block C4 Results -- Account-Takeover / New-Device Detectors (`new_device_burst` & `device_churn`)

## Overview
Built `assay/r2/detectors/new_device.py` exposing two `Detector` subclasses targeting pattern `"account_takeover"`:
1. **`NewDeviceBurst` (`"new_device_burst"`)**: Flags online charges on a `DeviceInfo` / `device_profile` never seen on the card prior to the 48h burst window, with $\ge 2$ online charges on that novel device within 48h and max charge amount $\ge m \times$ the card's prior median ($m \in \{2, 3, 4\}$). Separately measures the contribution of `id_15 == 'New'` and novel/`anonymous.com` `P_emaildomain`.
2. **`DeviceChurn` (`"device_churn"`)**: Flags $\ge 3$ charges within 72h across $\ge 3$ distinct `DeviceInfo` values and/or $\ge 3$ distinct email domains (`P_emaildomain` or `R_emaildomain`), where none of the `DeviceInfo` values is an established device on the card ($\ge 2$ prior uses), normalised by the card's prior device diversity (`prior_distinct_devs <= 2`).

All features are computed strictly as-of $t \le \text{opened\_at}$ without using `TransactionID` ranges or thresholds.

---

## Backtest 1: `new_device_burst` (5,585 Closed Cases + 3,500 Legitimate Card-Days, Seed=42)

Populations:
- **Confirmed Fraud Cases**: 4,675 total (of which 2,284 are `account_takeover` or `card_not_present_new_device`)
- **Cleared Cases**: 910
- **Legitimate Card-Days**: 3,500 sampled from Jul–Dec 2016 on cards never named in any closed case (`seed=42`)

| $m$ | Variant | $P(\text{fire}\mid\text{fraud})$ | $P(\text{fire}\mid\text{ATO/CNP-ND})$ | $P(\text{fire}\mid\text{cleared})$ | $P(\text{fire}\mid\text{legit day})$ | $\text{LR}_{\text{legit}}$ (95% CI) | $\text{LR}_{\text{cleared}}$ | Passes Gate ($\text{CI}_{\text{low}} > 1.5$)? |
|---:|---|---:|---:|---:|---:|---|---:|:---:|
| 2 | `base` | 83 / 4,675 (1.775%) | 56 / 2,284 (2.452%) | 9 / 910 (0.989%) | 68 / 3,500 (1.943%) | 0.9126 [0.6657, 1.2530] | 1.6373 | False |
| 2 | `+id_15=='New'` | 41 / 4,675 (0.877%) | 31 / 2,284 (1.357%) | 6 / 910 (0.659%) | 32 / 3,500 (0.914%) | 0.9522 [0.6003, 1.4847] | 1.1704 | False |
| 2 | `+novel_or_anon_email` | 30 / 4,675 (0.642%) | 20 / 2,284 (0.876%) | 1 / 910 (0.110%) | 14 / 3,500 (0.400%) | 1.5464 [0.8559, 2.9679] | 2.0191 | False |
| 2 | `+anon_email_only` | 15 / 4,675 (0.321%) | 11 / 2,284 (0.482%) | 0 / 910 (0.000%) | 10 / 3,500 (0.286%) | 1.0897 [0.5021, 2.5028] | 3.1204 | False |
| 2 | `+id_15=='New'&novel_or_anon_email` | 15 / 4,675 (0.321%) | 11 / 2,284 (0.482%) | 0 / 910 (0.000%) | 9 / 3,500 (0.257%) | 1.1887 [0.5425, 2.7905] | 3.1204 | False |
| 3 | `base` | 48 / 4,675 (1.027%) | 33 / 2,284 (1.445%) | 4 / 910 (0.440%) | 31 / 3,500 (0.886%) | 1.1486 [0.7335, 1.7873] | 1.5920 | False |
| 3 | `+id_15=='New'` | 20 / 4,675 (0.428%) | 15 / 2,284 (0.657%) | 3 / 910 (0.330%) | 17 / 3,500 (0.486%) | 0.8819 [0.4706, 1.6626] | 0.8191 | False |
| 3 | `+novel_or_anon_email` | 15 / 4,675 (0.321%) | 10 / 2,284 (0.438%) | 0 / 910 (0.000%) | 7 / 3,500 (0.200%) | 1.4529 [0.6400, 3.6323] | 3.1204 | False |
| 3 | `+anon_email_only` | 8 / 4,675 (0.171%) | 6 / 2,284 (0.263%) | 0 / 910 (0.000%) | 5 / 3,500 (0.143%) | 1.0776 [0.3910, 3.2328] | 1.7552 | False |
| 3 | `+id_15=='New'&novel_or_anon_email` | 8 / 4,675 (0.171%) | 6 / 2,284 (0.263%) | 0 / 910 (0.000%) | 5 / 3,500 (0.143%) | 1.0776 [0.3910, 3.2328] | 1.7552 | False |
| 4 | `base` | 30 / 4,675 (0.642%) | 21 / 2,284 (0.919%) | 3 / 910 (0.330%) | 20 / 3,500 (0.571%) | 1.0970 [0.6305, 1.9467] | 1.2092 | False |
| 4 | `+id_15=='New'` | 14 / 4,675 (0.299%) | 11 / 2,284 (0.482%) | 2 / 910 (0.220%) | 11 / 3,500 (0.314%) | 0.8920 [0.4150, 1.9502] | 0.7313 | False |
| 4 | `+novel_or_anon_email` | 12 / 4,675 (0.257%) | 9 / 2,284 (0.394%) | 0 / 910 (0.000%) | 5 / 3,500 (0.143%) | 1.5086 [0.5881, 4.4141] | 2.5353 | False |
| 4 | `+anon_email_only` | 6 / 4,675 (0.128%) | 5 / 2,284 (0.219%) | 0 / 910 (0.000%) | 4 / 3,500 (0.114%) | 0.9468 [0.3056, 3.1886] | 1.3652 | False |
| 4 | `+id_15=='New'&novel_or_anon_email` | 6 / 4,675 (0.128%) | 5 / 2,284 (0.219%) | 0 / 910 (0.000%) | 4 / 3,500 (0.114%) | 0.9468 [0.3056, 3.1886] | 1.3652 | False |

### Cleared Cases Fired On (`new_device_burst`)
- **`m=2, base` (9 cleared cases)**: `['CC-1546', 'CC-1662', 'CC-3423', 'CC-4041', 'CC-4224', 'CC-4266', 'CC-5208', 'CC-5569', 'CC-5583']`
- **`m=2, +novel_or_anon_email` (1 cleared case)**: `['CC-5569']`
- **`m=3, base` (4 cleared cases)**: `['CC-1546', 'CC-4041', 'CC-4266', 'CC-5583']`
- **`m=3, +novel_or_anon_email` (0 cleared cases)**: `[]`
- **`m=4, +novel_or_anon_email` (0 cleared cases)**: `[]`

### Measured Contribution of Each Condition
- **`id_15 == 'New'`**: Reduces recall by ~50% while leaving $\text{LR}_{\text{legit}} \approx 0.88 - 0.95$; does **not** improve the likelihood ratio over `base`.
- **Novel `P_emaildomain` or `anonymous.com` (`+novel_or_anon_email`)**: Eliminates cleared-case false positives at $m \ge 3$ (0/910 cleared cases fired, $\text{LR}_{\text{cleared}} = 3.12$) and lifts $\text{LR}_{\text{legit}}$ from $0.91$ to $1.55$ ($m=2$) and $1.51$ ($m=4$).
- **Keep Gate Result**: Because `card_id` is an issuer bucket where new underlying cardholders frequently appear with a new device and email, even the best variant (`m=2, +novel_or_anon_email`, $\text{LR}_{\text{legit}} = 1.5464$, 95% CI $[0.8559, 2.9679]$) has $\text{CI}_{\text{low}} = 0.8559 \le 1.5$. Thus **`new_device_burst` fails the keep gate ($\text{CI}_{\text{low}} \le 1.5$)** and defaults to `lr=1.0` (context weight).

---

## Backtest 2: `device_churn` (5,585 Closed Cases + 3,500 Legitimate Card-Days, Seed=42)

| Variant | $P(\text{fire}\mid\text{fraud})$ | $P(\text{fire}\mid\text{ATO/CNP-ND})$ | $P(\text{fire}\mid\text{cleared})$ | $P(\text{fire}\mid\text{legit day})$ | $\text{LR}_{\text{legit}}$ (95% CI) | $\text{LR}_{\text{cleared}}$ | Passes Gate ($\text{CI}_{\text{low}} > 1.5$)? |
|---|---:|---:|---:|---:|---|---:|:---:|
| `unnormalised_dev_or_email>=3` | 647 / 4,675 (13.840%) | 287 / 2,284 (12.566%) | 85 / 910 (9.341%) | 525 / 3,500 (15.000%) | 0.9225 [0.8313, 1.0298] | 1.4547 | False |
| `norm_prior_devs<=2_dev_or_email>=3` | 538 / 4,675 (11.508%) | 238 / 2,284 (10.420%) | 66 / 910 (7.253%) | 440 / 3,500 (12.571%) | 0.9154 [0.8127, 1.0282] | 1.5446 | False |
| `norm_prior_devs<=1_dev>=3` | 15 / 4,675 (0.321%) | 7 / 2,284 (0.306%) | 3 / 910 (0.330%) | 15 / 3,500 (0.429%) | 0.7484 [0.3692, 1.5086] | 0.6241 | False |
| `norm_prior_devs<=2_dev>=3_and_email>=3` | 14 / 4,675 (0.299%) | 9 / 2,284 (0.394%) | 1 / 910 (0.110%) | 7 / 3,500 (0.200%) | 1.3991 [0.5998, 3.6000] | 1.4627 | False |

### Cleared Cases Fired On (`device_churn`)
- **`unnormalised_dev_or_email>=3` (85 cleared cases)**: Fires heavily on busy multi-holder card buckets (`['CC-0001', 'CC-0050', 'CC-0126', ...]`).
- **`norm_prior_devs<=1_dev>=3` (3 cleared cases)**: `['CC-3063', 'CC-4191', 'CC-4666']`
- **`norm_prior_devs<=2_dev>=3_and_email>=3` (1 cleared case)**: `['CC-4666']`

### Keep Gate Result (`device_churn`)
Normalising by historical device diversity (`prior_distinct_devs <= 2`) and requiring both $\ge 3$ distinct unestablished `DeviceInfo` values and $\ge 3$ distinct email domains in 72h reduces cleared-case fires from 85/910 to 1/910 (`CC-4666`) and raises $\text{LR}_{\text{legit}}$ from $0.9225$ to $1.3991$ (95% CI $[0.5998, 3.6000]$). Because $\text{CI}_{\text{low}} = 0.5998 \le 1.5$, **`device_churn` also fails the keep gate** and carries `lr=1.0`.

---

## Exam Pack Evaluation (18 Cases in `case_pack_final.csv`)

1. **`new_device_burst` (calibrated: `m=2.0, +novel_or_anon_email`, `lr=1.0`, `ci_low=0.8559`)**:
   - **Fires on `HHF-009` (`C09177-K1`)**:
     - `affected_txn_ids`: `[3660354, 3660355]`
     - Evidence: 2 online charges (`$489.62` total, max `$261.61` = `4.4x` prior median `$59.95`) within 48h on previously unseen device `'LG-H831 Build/MRA58K'`, novel/anon email (`['anonymous.com']`), `id_15='New'`.
     - Note: Also fires under `m=3.0` and `m=4.0` with `+novel_or_anon_email` (where 0/910 cleared cases fire historically).
   - **All other 17 exam cases**: `FIRED=False`.

2. **`device_churn` (calibrated: `max_prior_devs=2, require_both_dev_and_email=True`, `lr=1.0`, `ci_low=0.5998`)**:
   - **Fires on `HHF-008` (`C07871-K1`)**:
     - `affected_txn_ids`: `[3660351, 3660352, 3660353]`
     - Evidence: 3 charges (`$443.19` total) within 72h across 3 distinct unestablished `DeviceInfo` values (`['Trident/7.0', 'Windows', 'iOS Device']`) and 3 distinct email domains (`['anonymous.com', 'gmail.com', 'outlook.com']`), vs 1 prior distinct device on card `C07871-K1`.
   - **All other 17 exam cases**: `FIRED=False`.
