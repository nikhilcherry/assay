# B4: Common Point of Compromise (CPC) & Coordinated Loss-Shape Detector

## Method
The CPC detector (`assay/r2/detectors/cpc.py`) investigates cases where several cards experience sudden losses with no shared device (such as `HHF-001` and `HHF-002`), relying strictly on behavioral transaction data (never lowering thresholds based on untrusted `trigger_text`):

1. **Tight Loss-Shape Cohort (`find_affected_cohort`)**:
   - Starting from the flagged transaction as of `opened_at`, we extract the episode's burst shape: 2–5 online `ProductCD == 'C'` charges between $250 and $2,200 within $\le 4$ days on a device profile never previously used by that card, on an established card ($\ge 5$ prior transactions) with no prior `C` charges $\ge \$250$, and with no device shared across the cohort.
   - Because `HHF-001` opens earlier (`2017-01-15 01:25:19`) than `HHF-002` (`2017-01-19 23:18:58`), its as-of cohort is smaller (3 cards vs 6+ cards).
2. **Size-Stratified Backtest & Confidence Intervals (`get_stratified_lr_table`)**:
   - We backtest the cohort detector over confirmed-fraud `C`-burst episodes vs 5,000 legitimate card-days from Jul–Oct 2016, stratifying by as-of cohort size bucket (`3`, `4-5`, `6+`).
   - For each size bucket, we compute the Laplace-smoothed likelihood ratio ($\text{LR}$) and its 95% confidence interval (Katz log-normal method). Cohort size 3 is accepted only when its measured 95% CI lower bound exceeds $2.0$, and any resulting finding carries its size-stratified LR.
3. **Activity-Matched Controls & Origin Search (`get_matched_controls`, `compute_origin_lifts`)**:
   - Each cohort card is matched against 20 control cards from the same 30-day transaction-count decile and primary `ProductCD` (seeded RNG, excluding cohort cards).
   - Over the 60 days preceding each card's first loss (plus full-history verification for `(ProductCD='R', amount=99.93)` and `(ProductCD='R', amount=100.09)`), we compute matched lift and one-sided Fisher exact $p$-value, requiring matched lift $\ge 5$, Fisher $p < 0.001$, and bank-wide cardinality $\le 99\text{th}$ percentile of legitimate Jul–Oct values.
4. **Honest R9 Fallback**:
   - When no candidate origin passes the activity-matched control test, the detector reports an honest negative on the prior origin and emits the coordinated loss-shape cohort finding (`undocumented:coordinated_cnp_burst_no_shared_device`) with the size-stratified LR and all cohort cards and burst transactions attested.

## RESULTS
(To be filled by lead from test output)
