# B4: Common Point of Compromise (CPC) & Coordinated Loss-Shape Detector

## Method
The CPC detector (`assay/r2/detectors/cpc.py`) investigates cases where several cards experience sudden losses with no shared device (such as `HHF-001` and `HHF-002`), relying strictly on behavioral transaction data (never lowering thresholds based on untrusted `trigger_text`):

1. **Precomputed O(log N) Index (`get_precomputed_index`)**:
   - Once per process (~2s, <100 MB), we index `tx_all()` timestamps, card indices, 5th-transaction timestamps, first `ProductCD=='C' >= $250` timestamps, specific device first-seen timestamps, and monthly card activity decile pools.
2. **Tight Loss-Shape Cohort (`fast_find_cohort_by_tid`)**:
   - Using binary search (`np.searchsorted`) over the precomputed arrays, we find cards within `[-15d, opened_at]` showing 2–5 online `ProductCD == 'C'` charges between $250 and $2,200 within $\le 4$ days on a novel specific device for that card, with $\ge 5$ prior transactions and no prior `C` charges $\ge \$250$, and prune any shared device across the cohort.
3. **Size-Stratified Backtest & Confidence Intervals (`get_stratified_lr_table`)**:
   - We backtest the cohort detector over confirmed-fraud `C`-burst episodes vs 5,000 legitimate card-days from Jul–Oct 2016, stratifying by as-of cohort size bucket (`3`, `4-5`, `6+`).
   - For each size bucket, we compute the Laplace-smoothed likelihood ratio ($\text{LR}$) and its 95% confidence interval (Katz log-normal method). Cohort size 3 is accepted only when its measured 95% CI lower bound exceeds $2.0$, and any resulting finding carries its size-stratified LR.
4. **Activity-Matched Controls & Origin Search (`fast_matched_controls`, `compute_origin_lifts`)**:
   - Each cohort card is matched against 20 control cards from the same monthly transaction-count decile and primary `ProductCD`.
   - Over the 60 days preceding each card's first loss (plus full-history verification for `(ProductCD='R', amount=99.93)` and `(ProductCD='R', amount=100.09)`), we compute matched lift and one-sided Fisher exact $p$-value, requiring matched lift $\ge 5$, Fisher $p < 0.001$, and bank-wide cardinality $\le 99\text{th}$ percentile of legitimate Jul–Oct values.
5. **Honest R9 Fallback**:
   - When no candidate origin passes the activity-matched control test, the detector reports an honest negative on the prior origin and emits the coordinated loss-shape cohort finding (`undocumented:coordinated_cnp_burst_no_shared_device`) with the size-stratified LR and all cohort cards and burst transactions attested.

## RESULTS
(To be filled by lead from test output)
