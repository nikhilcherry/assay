# B7 — Self-Growing Detective (`assay/r2/discover.py`)

## Overview
`assay/r2/discover.py` implements an autonomous pattern-discovery, deterministic detector code-generation, as-of backtesting, and server-side L2 approval pipeline:

1. **Causal Episode Featurisation (`tx_predicates_all`, `featurize_closed_cases`)**: Computes 44 interpretable boolean predicates per transaction and closed-case episode strictly as-of each case's `opened_at` timestamp. Shared-element predicates (`shared_dev_7d_ge2cards`, `shared_remail_7d_ge3cards`, `shared_addr_amt_7d_ge2cards`) explicitly filter out generic device strings (`Windows`, `iOS Device`, `? | ? | ? | ?`), whole-dollar round amounts, and high-cardinality elements above a learned cardinality cap.
2. **Legitimate Card-Day Control Sample (`featurize_legit_card_days_jul_oct`)**: Samples `6,000` legitimate card-days from Jul–Oct 2016 (`seed=42`, transactions and cards never named in any confirmed fraud case) and applies the exact same episode predicate aggregation as closed cases. Likelihood ratios are computed against this unbiased negative population: `LR = P(fire | fraud episode) / P(fire | legit card-day)`.
3. **Apriori Conjunction Mining (`mine_and_propose`)**: Mines 2- and 3-predicate conjunctions on the **Jul–Oct 2016** training split (`4,033` closed cases) that catch `undocumented` / `card_testing` and residual confirmed-fraud episodes while enforcing `<= 1.5%` firing rate on the 6,000 legitimate card-day sample and `<= 2` cleared cases acted against in training.
4. **Deterministic Code Generation & L2 Approval Gate**: Emits `Detector` subclasses to `data/stage/r2/proposed/<name>.py` with `name = ""` so unapproved modules cannot auto-register in `REGISTRY`. `approve(name, approver_token)` verifies the server-side L2 role and signs an HMAC over `(name, code_sha256, approver_id, role, dataset_tag)`.

## Evaluation Protocol
- **Mining Split**: Jul–Oct 2016 closed cases (`n = 4,033`: `3,388` confirmed fraud, `645` cleared) + `6,000` Jul–Oct legitimate card-days (`seed = 42`).
- **Out-of-Time Validation Split**: Nov–Dec 2016 closed cases (`n = 1,532`).
- **Full Closed Cases**: All `5,585` closed cases (`4,675` confirmed fraud, `910` cleared, `11` labelled `undocumented`).
- **Attestation Verification**: `tests/r2/test_discover.py` prints the top 5 proposals with exact Jul–Oct train, Nov–Dec validation, and 6,000 legit card-day firing rates and re-runs `assay.r2.discover.verify_backtest_rule` on every emitted JSON proposal.
