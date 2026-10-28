# B34 — Episode Boundary Detection (`assay/r2/episode.py`)

## Method
`episode(card_id, flagged_txn_id, as_of, tx) -> Episode` identifies all transactions belonging to a card-compromise episode as of `as_of`:
1. **Per-card feature sequence (as-of `as_of`)**:
   - `prod_nov`: ProductCD novelty relative to the card's modal product and frequency.
   - `chan_nov`: Channel shift (`in_person` vs `online` relative to card's historical online rate).
   - `dev_nov`: Device profile rarity/novelty on the card.
   - `reg_nov`: Billing region (`addr1`) shift or missing-address shift relative to the card's modal address.
   - `amt_z`: Robust z-score of transaction amount against the card's median/MAD and inter-quantile baseline.
   - `min_gap` & `prev_gap`: Temporal spacing between consecutive transactions on the card.
   - `anchor_sim`: Signature compatibility with the flagged transaction (channel, device profile, product, region).
2. **Learning**: Two-state HMM emission and gap-conditioned transition matrices (`state 0 = legitimate`, `state 1 = compromised`) are learned with Laplace smoothing from pre-October confirmed-fraud closed cases (`txn_list` = ground-truth episode) vs the same cards' surrounding legitimate transactions.
3. **Inference**: Forward-backward in log-space within $\pm 30$ days of `flagged_txn_id` (clamping $s_{\text{flagged}} = 1$), extending backward and forward while $P(s_t = 1 \mid \mathbf{x}) > 0.5$. Handles card alias mismatches (such as `HHF-011` where the alert names `C09241-K1` and the transaction resides on `C09241-K2`).

## Evaluation on Closed Cases (As-Of)
Run `pytest -q -s tests/r2/test_episode.py` to reproduce the exact comparison table on Oct–Dec (`opened_at >= 2016-10-01`) and Nov–Dec (`opened_at >= 2016-11-01`) confirmed-fraud closed cases against:
- **Baseline (a)**: `flagged_txn_id` only.
- **Baseline (b)**: All card transactions within $\pm 7$ days with the same `ProductCD`.

Metrics reported in test output:
- `exact_match_rate`: Exact set equality between predicted `txn_ids` and ground-truth `txn_list`.
- `mean_jaccard`: Mean intersection-over-union of transaction ID sets.
- `first_txn_accuracy`: Exact match on `first_fraud_txn_id`.
- `mean_exposure_abs_err` / `median_exposure_abs_err`: Absolute USD error on `exposure_usd`.
- Cleared-case expansion rate on Oct–Dec cleared (`legitimate`) closed cases.
- Full episode table (`first_txn_id`, `n_txns`, `exposure_usd`, `confidence`, `txn_ids`) for all 18 exam cases (`HHF-001` … `HHF-018`).
