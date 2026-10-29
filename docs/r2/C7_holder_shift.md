# Block C7 Results -- Holder-Level Session Shift (`holder_shift`)

## Overview & Single-Code-Path Design
Block C4 (`docs/r2/C4_new_device.md`) evaluated new-device takeover rules at the `card_id` level and found that `card_id` is a shared issuer bucket. Block C7 (`assay/r2/detectors/holder_shift.py`) re-measures session shift at the **holder (`uid` = `card_id | addr1 | first_use_day`)** level strictly as-of $t \le \text{opened\_at}$:
- Both `backtest()`, `inspect()`, and `run()` call the **exact same function** (`evaluate_single_case_uid`), resolving the case's anchor transaction as-of `opened_at`, taking its `uid`, and checking candidate transactions on that `uid` in `[opened_at - W, opened_at]` (plus the anchor transaction itself).
- Requires the holder (`uid`) to have $\ge 10$ prior transactions strictly before the lookback window (`48h` or `7d` / `168h`); otherwise reports **not applicable**.
- Checks whether the holder (`uid`) had online charge(s) in the window on a `device_profile` and/or `P_emaildomain` never used before by that `uid`, with charge amount $\ge m \times$ the holder's own prior median ($m \in \{2, 3\}$).
- **Keep Gate**: 95% CI lower bound of $\text{LR}_{\text{cleared}} = P(\text{fire}\mid\text{fraud}) / P(\text{fire}\mid\text{cleared}) > 1.5$ (Laplace-smoothed) **AND** $P(\text{fire}\mid\text{cleared}) \le 8\%$. Every rate and bootstrap CI is guarded against $n = 0$ (`insufficient data`, ineligible for the gate) and clipped to $[0, 1]$ after verifying $n > 0$.

---

## Applicability on Closed Cases & Exam Cases (`uid` $\ge 10$ Prior Transactions)
Why `uid` applicability is very sparse on fraud and exam alerts:
- In `frames.py`, `uid` is defined as `card_id | addr1 | (day - D1)`.
- When an account-takeover or new-device attack occurs (or when a card is used with a different billing `addr1` or a reset `D1`), the flagged transaction lands on a **newly created `uid`** with only `0–2` prior transactions, even if the underlying `card_id` had 10–60 prior transactions.
- Consequently, across the 5,585 closed cases in `closed_cases_history.csv`:
  - **Confirmed Fraud Cases**: only **`236 / 4,675` (`5.05%`)** have a resolved holder (`uid`) with $\ge 10$ prior transactions as-of `opened_at`.
  - **Cleared Cases**: only **`8 / 910` (`0.88%`)** have a resolved holder (`uid`) with $\ge 10$ prior transactions as-of `opened_at` (the denominator mismatch in the earlier run was caused by `holder_eligible_ge10` checking only the single anchor txn's `uid_seq` while a later txn in the 48h window reached `prior_cnt >= 10`; unifying both onto `evaluate_single_case_uid` ensures `n_elig` and `fired` use the exact same eligibility definition).
- **Across all 18 exam cases (`case_pack_final.csv`)**: **0 / 18 cases** have a flagged `uid` with $\ge 10$ prior transactions (`total_uid_txns_as_of` is `1` or `2` on every single exam case).

---

## Backtest Measurement & Gate Decision
- The full 16-cell backtest grid (`window` $\in \{48\text{h}, 7\text{d}\}$, $m \in \{2.0, 3.0\}$, `shift_mode` $\in \{\text{dev\_or\_email}, \text{dev\_only}, \text{email\_only}, \text{dev\_and\_email}\}$) is printed by `pytest -q -s tests/r2/test_holder_shift.py` with exact counts `(caught / n_fraud)`, `(caught / n_fraud_elig)`, `(cleared_fired / n_cleared)`, `(cleared_fired / n_cleared_elig)`, `(legit_days_fired / 3500)`, Laplace-smoothed $\text{LR}_{\text{cleared}}$ [95% CI], and $\text{LR}_{\text{elig}}$ [95% CI].
- Because only $\sim 5\%$ of fraud cases and $< 2\%$ of cleared cases have $\ge 10$ prior transactions on the flagged transaction's `uid`, and because cleared cases that do have $\ge 10$ prior transactions on their `uid` also fire when a repeat holder uses a new device or email on a larger online purchase, `HolderShift` fails to provide a useful exam-pack signal (and is `not applicable` on all 18 exam cases).

---

## Exam Pack Evaluation (18 Cases in `case_pack_final.csv`)

On all 18 exam cases (`HHF-001` through `HHF-018`), the flagged transaction's `uid` has only `0` or `1` prior transaction (`1` or `2` total transactions as-of `opened_at`), which is below the required `MIN_PRIOR_TXNS = 10`. Therefore:
- **`applicable = False`** (`not applicable: holder <uid> has 0..1 prior txns before window (< 10 required)`) on **all 18 exam cases**, including all sensitive cases (`HHF-009, HHF-010, HHF-012..018`).
- **`fired = False`** on **all 18 exam cases** (both gated and ungated).

| Case ID | Sensitive? | Card ID | Flagged `uid` | Prior Txns (`48h` / `7d`) | Total `uid` Txns As-Of | Applicable ($\ge 10$)? | `48h` / `7d` Fire |
|---|:---:|---|---|---:|---:|:---:|:---:|
| `HHF-001` | | `C05061-K1` | `C05061-K1\|204\|196` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-002` | | `C06220-K1` | `C06220-K1\|264\|201` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-003` | | `C11663-K1` | `C11663-K1\|325\|213` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-004` | | `C12975-K2` | `C12975-K2\|299\|207` | `0 / 0` | `2` | No (`<10`) | N/A (`False`) |
| `HHF-005` | | `C00348-K1` | `C00348-K1\|126\|209` | `0 / 0` | `2` | No (`<10`) | N/A (`False`) |
| `HHF-006` | | `C01727-K2` | `C01727-K2\|231\|210` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-007` | | `C09582-K1` | `C09582-K1\|476\|-101` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-008` | | `C07871-K1` | `C07871-K1\|87\|203` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-009` | **Yes** | `C09177-K1` | `C09177-K1\|337\|209` | `0 / 0` | `2` | No (`<10`) | N/A (`False`) |
| `HHF-010` | **Yes** | `C07196-K2` | `C07196-K2\|205\|190` | `0 / 0` | `2` | No (`<10`) | N/A (`False`) |
| `HHF-011` | | `C09241-K2` | `C09241-K2\|494\|-43` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-012` | **Yes** | `C12223-K1` | `C12223-K1\|476\|191` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-013` | **Yes** | `C06115-K1` | `C06115-K1\|203\|187` | `1 / 1` | `2` | No (`<10`) | N/A (`False`) |
| `HHF-014` | **Yes** | `C00750-K2` | `C00750-K2\|315\|151` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-015` | **Yes** | `C00543-K1` | `C00543-K1\|485\|-125` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-016` | **Yes** | `C02502-K2` | `C02502-K2\|123\|183` | `0 / 0` | `2` | No (`<10`) | N/A (`False`) |
| `HHF-017` | **Yes** | `C08842-K2` | `C08842-K2\|299\|183` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
| `HHF-018` | **Yes** | `C02753-K1` | `C02753-K1\|299\|183` | `0 / 0` | `1` | No (`<10`) | N/A (`False`) |
