# C10: Common-Point-of-Compromise (CPC) Detector on TigerGraph (`r2_cpc.gsql`)

## What Was Built
We moved the graph-shaped parts of `assay/r2/detectors/cpc.py` onto TigerGraph (`graph/queries/r2_cpc.gsql`) with Python wrappers and parity verification in `assay/r2/tg_cpc.py`:

1. **`r2_cpc_loss_cohort` (`graph/queries/r2_cpc.gsql`)**:
   - Starting from `flagged_txn_id` as-of `opened_at` (`t.ts <= opened_at`), inspects the flagged card's $\pm 3\text{d}$ online `ProductCD == 'C'` charges to determine the loss shape (`min_amt`, `max_amt`, `min_count=2`, `max_count=5`).
   - Traverses `Card -(MADE>)- Txn` in `[-15d, min(t0 + 10d, opened_at)]`, ordering per-card charges via a `HeapAccum<BurstTxn>` (`TYPEDEF TUPLE<INT ts_sec, INT tid, STRING dev>`) to find a $\le 4\text{d}$ burst window.
   - Enforces the exact history and novelty filters as-of each card's burst start (`@prior_cnt >= 5`, no prior `ProductCD == 'C'` charge $\ge \$250$, no specific `device_profile` in the burst seen on that card prior to the burst start), and prunes any non-flagged cards sharing a specific device across the cohort.
   - Returns `has_shape`, `member_cards`, `first_loss_sec`, and `burst_txns`.

2. **`r2_cpc_origin_counts` (`graph/queries/r2_cpc.gsql`)**:
   - Given a set of cohort `card_ids`, their `card_loss_epochs` (`<card_id>:<loss_epoch_sec>`), `opened_at`, and `freemail`, traverses `Card -(MADE>)- Txn` in each card's 60-day lookback `[loss_sec - 60d, loss_sec)` and extracts all 6 candidate origin key families used by `cpc.py:extract_features_sub`:
     1. `ProductCD_amount:<product>_<amount_2dp>`
     2. `ProductCD_addr1:<product>_<addr1>` (`addr1 >= 0`)
     3. `R_emaildomain:<r_email>` (non-empty, non-freemail)
     4. `P_emaildomain_ProductCD:<p_email>_<product>` (non-empty, non-freemail)
     5. `device_profile:<device_profile>` (specific device)
     6. `merchant_like:<product>_<amount_2dp>_<dist1>` (`dist1 >= 0.0`)
   - Computes per-origin distinct cohort card counts (`origin_card_count`), per-origin cohort transaction and `(card_id, txn_id)` pairs, full-graph distinct card cardinality (`origin_total_cardinality`), and as-of distinct card cardinality (`origin_asof_cardinality`).

3. **Schema Coverage & Cardinality Note**:
   - Every attribute required by the 6 origin feature families in `cpc.py` exists on the `Txn` vertex in `graph/schema.gsql` and `graph/load_r2.gsql` (`product`, `amount`, `addr1`, `r_email`, `p_email`, `device_profile`, `dist1`). Zero candidate origin families had to be dropped.
   - In `cpc.py:get_precomputed_index`, `idx.global_card_counts` indexes full-dataset distinct card counts for 3 families (`ProductCD_amount`, `ProductCD_addr1`, `R_emaildomain`) and falls back to `k_aff + k_ctrl` for the other 3. `compute_origin_lifts_tg` uses `origin_total_cardinality` from TigerGraph for those 3 indexed families and `k_aff + k_ctrl` for the remaining 3, and iterates candidate keys in sorted lexicographical order (matching `aff_feats.groupby('feature', sort=True)` in `cpc.py`) so `compute_origin_lifts_tg` is 100% row-for-row identical to `cpc.py`.

## Measured Parity (`tests/r2/test_tg_cpc.py`)
Running `pytest -q -s -x -p no:cacheprovider tests/r2/test_tg_cpc.py` against the live `assay` graph on TigerGraph CE 4.2:

| Case | Card | Local Cohort Cards | TG Cohort Cards | Local Candidate Origins (60d) | TG Candidate Origins (60d) | Members Identical | Burst Txns Identical | Origin Counts Identical | Cardinalities Identical | Lifts Identical |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| `HHF-001` | `C05061-K1` | 3 | 3 | 1024 | 1024 | `True` | `True` | `True` | `True` | `True` |
| `HHF-002` | `C06220-K1` | 12 | 12 | 2456 | 2456 | `True` | `True` | `True` | `True` | `True` |
| `HHF-003` | `C11663-K1` | 11 | 11 | 1772 | 1772 | `True` | `True` | `True` | `True` | `True` |
| `HHF-006` | `C01727-K2` | 0 | 0 | 0 | 0 | `True` | `True` | `True` | `True` | `True` |

- **HHF-001** (`3660108`, `C05061-K1`): 3 cohort cards (`['C05061-K1', 'C07212-K2', 'C11543-K2']`); 1,024 candidate origins and lift table matched 100% (`96.2ms` cohort, `344.5ms` origin counts).
- **HHF-002** (`3660127`, `C06220-K1`): 12 cohort cards (`['C01727-K2', 'C02167-K2', 'C02342-K1', 'C05061-K1', 'C05770-K1', 'C06220-K1', 'C06962-K2', 'C07212-K2', 'C07925-K1', 'C10553-K1', 'C11309-K1', 'C11543-K2']`); 2,456 candidate origins and lift table matched 100% (`120.1ms` cohort, `484.9ms` origin counts).
- **HHF-003** (`3660153`, `C11663-K1`): 11 cohort cards (`['C00450-K1', 'C01727-K2', 'C03080-K2', 'C04668-K1', 'C05770-K1', 'C06220-K1', 'C06962-K2', 'C07997-K1', 'C08943-K2', 'C11309-K1', 'C11663-K1']`); 1,772 candidate origins and lift table matched 100% (`105.5ms` cohort, `405.8ms` origin counts).
- **HHF-006** (`3660263`, `C01727-K2`): 0 cohort cards (`[]`); 0 candidate origins matched 100% (`57.9ms` cohort, `16.1ms` origin counts).
