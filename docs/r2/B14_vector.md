# B14 — TigerVector: Vector Search Fused with Graph Traversal

## 1. Zero-Leakage Audit of the 24-Dim Behaviour Embedding
Every embedding (`assay/r2/vector.py`) is computed strictly from `tx_all()` rows on the card with `t <= opened_at` (the 24-hour episode window ending at the card's latest transaction at or before `opened_at`, compared against prior card history strictly before that window).

No post-investigation fields from `closed_cases` (`outcome`, `actions_taken`, `exposure_usd`, `connected_card_ids`, `txn_ids`, `first_fraud_txn_id`, `n_txns`, `analyst_notes`, `closed_at`, `pattern`, `report_filed`) are ever read when constructing an embedding. `test_embeddings_and_ablation` verifies that overwriting all post-hoc columns in `closed_cases` leaves every embedding bit-for-bit identical.

| # | Feature Name | Source Column(s) (as-of `t <= opened_at`) |
|---:|---|---|
| 0 | `ep_log_amt_mean` | `tx_all.amount` (24h episode window `t <= opened_at`) |
| 1 | `ep_log_amt_max` | `tx_all.amount` (24h episode window `t <= opened_at`) |
| 2 | `ep_log_amt_std` | `tx_all.amount` (24h episode window `t <= opened_at`) |
| 3 | `ep_log_amt_sum` | `tx_all.amount` (24h episode window `t <= opened_at`) |
| 4 | `vs_card_amt_ratio` | `tx_all.amount` (episode vs card prior `t < window_start`) |
| 5 | `vs_card_amt_z` | `tx_all.amount` (episode vs card prior `t < window_start`) |
| 6 | `card_log_hist_cnt` | `tx_all.TransactionID` (count of card prior txns `t < window_start`) |
| 7 | `card_prior_online_rate` | `tx_all.channel` (online rate on card prior txns `t < window_start`) |
| 8 | `ep_log_win_cnt` | `tx_all.TransactionID` (count of card txns in 24h window `t <= opened_at`) |
| 9 | `ep_log_span_sec` | `tx_all.TransactionDT` (time span in 24h window `t <= opened_at`) |
| 10 | `ep_log_min_gap_sec` | `tx_all.TransactionDT` (min inter-txn gap in 24h window `t <= opened_at`) |
| 11 | `ep_small_auth_frac` | `tx_all.amount` (fraction `< $25` in 24h window `t <= opened_at`) |
| 12 | `ep_online_frac` | `tx_all.channel` (fraction `== 'online'` in 24h window `t <= opened_at`) |
| 13 | `prod_W_frac` | `tx_all.ProductCD` (fraction `== 'W'` in 24h window `t <= opened_at`) |
| 14 | `prod_C_frac` | `tx_all.ProductCD` (fraction `== 'C'` in 24h window `t <= opened_at`) |
| 15 | `prod_R_frac` | `tx_all.ProductCD` (fraction `== 'R'` in 24h window `t <= opened_at`) |
| 16 | `prod_H_frac` | `tx_all.ProductCD` (fraction `== 'H'` in 24h window `t <= opened_at`) |
| 17 | `prod_S_frac` | `tx_all.ProductCD` (fraction `== 'S'` in 24h window `t <= opened_at`) |
| 18 | `device_present_frac` | `tx_all.device_profile` (non-empty rate in 24h window `t <= opened_at`) |
| 19 | `device_novel_frac` | `tx_all.device_profile` (unseen on card prior to `window_start`) |
| 20 | `region_novel_frac` | `tx_all.addr1` (billing region unseen on card prior to `window_start`) |
| 21 | `hour_sin` | `tx_all.TransactionDT` (sin of UTC hour-of-day in 24h window `t <= opened_at`) |
| 22 | `card_age_log_d1` | `tx_all.D1` (card age days at episode `t <= opened_at`) |
| 23 | `asof_shared_dev_cards_log` | `tx_all.device_profile` + `tx_all.card_id` (other cards on device with `t <= opened_at`) |

## 2. TigerGraph 4.2 Vector Schema, Queries & Fail-Closed Behaviour
- `graph/r2_vector_schema.gsql`: Adds `VECTOR ATTRIBUTE behaviour(DIMENSION=24, METRIC="COSINE")` to `ClosedCase` and defines `load_case_vectors` from `/home/tigergraph/data/tg_r2/case_vectors.csv`.
- `graph/queries/r2_vector.gsql`:
  - `similar_cases_vec(LIST<FLOAT> qv, INT k, DATETIME as_of)`: filters `ClosedCase` to `closed_at <= as_of`, runs `vectorSearch({ClosedCase.behaviour}, qv, k, filter=Eligible)`, and traverses 1–2 hops from the hits to shared `Card` and `DeviceProfile` vertices.
  - `similar_cases_fused(VERTEX<Card> c, LIST<FLOAT> qv, INT k, INT hops, DATETIME as_of)`: traverses 2 hops from `Card` (`t <= as_of`) through `DeviceProfile`, `BillingRegion`, and `EmailDomain` to `ClosedCase` (`closed_at <= as_of`), runs `vectorSearch({ClosedCase.behaviour}, qv, k, filter=CandidateSet)`, and expands 1–2 hops from the hits to shared cards and devices.
- **Fail-closed:** When `ASSAY_TG=1`, `LocalVectorIndex` calls TigerGraph directly and raises `TigerGraphUnavailableError` if TigerGraph is unreachable (never falling back silently).

## 3. Held-Out As-Of Ablation on October–December 2016 Closed Cases
Re-measured 2026-10-29 with `run_ablation(k=5)`. November–December hold only 45 closed cases, so the
window is October–December: **1,392** cases, **1,238** `confirmed_fraud` (88.94% majority baseline) and
**154** `cleared`. Each case's index holds only cases closed before it opened (at least 4,052 prior
cases available). An earlier version of this section reported "1,547 Nov–Dec cases"; that count and its
figures were wrong and are withdrawn.

| Method | Accuracy | Balanced Acc | Brier | Cleared cases acted against (p >= 0.70) |
|---|---|---|---|---|
| Bank `risk_score` (max in 24h episode) | 0.5589 | 0.3171 | 0.3033 | — |
| Vector-only k-NN (k=5) | 0.8951 | 0.6511 | 0.0863 | 75 / 154 |
| Traversal-only 2-hop | 0.8757 | 0.5179 | 0.1101 | 125 / 154 |
| Fused 2-hop + vector | 0.8843 | 0.6336 | 0.0896 | 78 / 154 |

- Vector precedent beats the bank score on calibration (Brier 0.0863 vs 0.3033), but its accuracy is
  barely above always saying fraud (0.8951 vs 0.8894).
- **Measured negative: graph traversal hurts.** Fused and vector-only disagree on 59 cases, and fused is
  right on 22 of them (37.29%). Traversal-only acts against 125 cleared cases vs 75 for vector-only.
- Neither is on the decision path: precedent never enters the ledger, and the answer's
  `similar_prior_cases` comes from the rule-based `closed_cases_like` lookup.
