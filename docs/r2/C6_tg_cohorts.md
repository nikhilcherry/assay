# Block C6 — Cohort Detectors on TigerGraph (`r2_micro_drip_cohort` & `r2_recipient_fanin`)

## 1. What Was Built

- **`graph/queries/r2_cohorts.gsql`**: Two GSQL V2 queries on graph `assay` (TigerGraph CE 4.2):
  1. `r2_micro_drip_cohort(flagged_txn_id, opened_at, max_amt=6.50, min_charges=4, min_span_days=7.0, window_days=30, quiet_days=60, min_other_cards=6)`:
     Mirrors `assay.r2.detectors.micro_charge.cohort_members`. Finds all `Card` vertices with online `ProductCD == "C"` transactions `< $6.50` at or before `opened_at`, orders each card's micro-charges chronologically, checks the 60-day quiet period prior to each candidate window start, verifies `>= 4` charges within `[t0, min(t0 + 30d, opened_at)]` spanning `>= 7` days (and `> 1` hour), checks overlap with `[opened_at - 30d, opened_at]`, and verifies that `flagged_txn_id` belongs to its card's episode and at least `min_other_cards` (6) other cards overlap.
  2. `r2_recipient_fanin(flagged_txn_id, opened_at, freemail, window_days=21, min_amount=100.0, min_k_now=5, burst_mult=3.0, max_p=0.0001)`:
     Mirrors `assay.r2.detectors.fan_in.cohort_members`. Verifies that `flagged_txn_id` itself is an online `ProductCD == "C"` charge with `amount >= $100` to a non-freemail `r_email` (`SET<STRING> freemail` parameter plus standard freemail prefix rules), finds all cards whose first-ever online `ProductCD == "C"` charge to that `r_email` falls in `[opened_at - 21d, opened_at]` and has `amount >= $100` in the window, computes the causal 21-day baseline mean/max prior to `min(opened_at - 21d, 2017-01-01)`, and evaluates the Poisson tail burst criteria.
- **`assay/r2/tg_cohorts.py`**:
  - `install(conn)`: runs `CREATE OR REPLACE QUERY` + `INSTALL QUERY r2_micro_drip_cohort, r2_recipient_fanin`, verifies `/restpp/endpoints/assay?dynamic=true`, and immediately smoke-runs both queries over REST++.
  - `micro_drip_cohort_tg(...)` and `recipient_fanin_tg(...)`: return `(member_card_ids, per_card_txn_ids, params)` matching the Python functions and fail closed (`TigerGraphUnavailableError`) on any connection, HTTP, GSQL, or schema/key error.
  - `parity(case_ids)`: compares TigerGraph vs local Python across all 18 cases in `case_pack()`.

## 2. Schema Alignment Note (`R_emaildomain`)

In `graph/schema.gsql` and `graph/load_r2.gsql`:
- `EmailDomain` vertices and `PURCHASER_EMAIL` edges only index the purchaser email domain (`p_email`).
- The recipient email domain (`R_emaildomain`) is stored directly on the `Txn` vertex as the `r_email` (`STRING`) attribute (`$"r_email"` in `load_r2.gsql`).
- Accordingly, `r2_recipient_fanin` traverses `Card:c -(MADE>)- Txn:t` and filters on `t.r_email` directly, achieving exact parity with `frames.tx_all().R_emaildomain` without altering the schema or reloading data.

## 3. Measured Parity Results on the 18 Exam Cases (`case_pack()`)

All 36 comparisons (18 cases × 2 cohort detectors) produced **100% identical** member card sets and per-card transaction ID lists between local Python and live TigerGraph GSQL (`mismatches = 0`).

| Case | Detector | Local Members | TG Members | Total Cohort Txns (Local / TG) | Identical? | TG Latency (ms) |
|---|---|---:|---:|---:|:---:|---:|
| HHF-001 | `micro_drip_cohort` | 14 | 14 | 75 / 75 | True | 87.3 |
| HHF-002 | `micro_drip_cohort` | 15 | 15 | 85 / 85 | True | 75.5 |
| HHF-004 | `recipient_fanin` | 12 | 12 | 25 / 25 | True | 164.0 |
| HHF-005 | `recipient_fanin` | 12 | 12 | 26 / 26 | True | 168.4 |
| HHF-006 | `micro_drip_cohort` | 16 | 16 | 93 / 93 | True | 78.7 |
| All other 31 (case, detector) pairs | — | 0 | 0 | 0 / 0 | True | 9.3 – 94.0 |

### Fired Cohort Details (Identical in Local Python and TigerGraph)

- **HHF-001 (`micro_drip_cohort`, as-of `2017-01-15 01:25:19`)**: 14 cards (`C00059-K1`, `C00253-K1`, `C01598-K1`, `C04521-K1`, `C05061-K1`, `C05182-K1`, `C06110-K1`, `C06220-K1`, `C06262-K2`, `C07902-K1`, `C08713-K1`, `C11044-K2`, `C11218-K1`, `C13462-K1`), 75 cohort transactions.
- **HHF-002 (`micro_drip_cohort`, as-of `2017-01-19 23:18:58`)**: 15 cards (`C00253-K1`, `C01598-K1`, `C01727-K2`, `C04521-K1`, `C05061-K1`, `C05182-K1`, `C06110-K1`, `C06220-K1`, `C06262-K2`, `C07902-K1`, `C08713-K1`, `C10256-K1`, `C11044-K2`, `C11218-K1`, `C13462-K1`), 85 cohort transactions.
- **HHF-006 (`micro_drip_cohort`, as-of `2017-01-28 10:59:42`)**: 16 cards (`C01598-K1`, `C01727-K2`, `C04521-K1`, `C05061-K1`, `C05182-K1`, `C06220-K1`, `C06262-K2`, `C07902-K1`, `C08713-K1`, `C10256-K1`, `C10528-K1`, `C11044-K2`, `C11218-K1`, `C11622-K2`, `C12582-K1`, `C13462-K1`), 93 cohort transactions.
- **HHF-004 (`recipient_fanin` on `sc.rr.com`, as-of `2017-01-26 21:09:37`)**: 12 cards (`C00348-K1`, `C01079-K1`, `C01841-K1`, `C02909-K1`, `C04850-K2`, `C05919-K2`, `C06115-K1`, `C07419-K1`, `C08335-K1`, `C10105-K1`, `C11402-K1`, `C12975-K2`), 25 cohort transactions.
- **HHF-005 (`recipient_fanin` on `sc.rr.com`, as-of `2017-01-28 23:45:16`)**: 12 cards (`C00348-K1`, `C01079-K1`, `C01841-K1`, `C02909-K1`, `C04850-K2`, `C05919-K2`, `C06115-K1`, `C07419-K1`, `C08335-K1`, `C10105-K1`, `C11402-K1`, `C12975-K2`), 26 cohort transactions.
