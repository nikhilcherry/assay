# B18 -- Containment Planner (Challenge 7) Results

## Summary
The Containment Planner (`assay/r2/containment.py`) evaluates a card-fraud ring to determine the optimal set of card actions that maximizes net loss reduction while penalizing actions against legitimate customers.

## Key Components Measured
1. **Exposure Model**:
   - Computes `already_lost` (sum of suspicious amounts so far) and `forward_exposure` empirically derived from confirmed fraud episodes in `closed_cases()` (median remaining loss multiplier given $k$ suspicious txns seen so far: $M(1)=1.00$, $M(2)=0.35$, $M(\ge 3)=0.15$).
2. **Evidence Strength Model**:
   - Derived per card in $[0.05, 1.0]$ based on prior confirmed fraud cases (+0.35), loss shape similarity (+0.35), and device link specificity (+0.30).
3. **Innocence Model**:
   - Fitted on Jul-Oct empirical baseline ($P(\text{innocent}) = 0.916$ for cards with $\ge 30\text{d}$ stable transaction history, high-cardinality link, and no post-ring loss shape match; $0.985$ for cleared closed case history).
4. **Planner & Trade-off Frontier**:
   - Uses exact enumeration for rings with $\le 18$ cards and greedy marginal-gain proof optimization for $> 18$ cards.
   - Outputs a frontier table ($k$ blocks vs exposure cut vs expected innocents touched vs net gain = cut - penalty $\times$ expected innocents).
   - Routes actions per policy ladder (`BLOCK_CARD` L1 $\le \$2,500$, L2 above; `BLOCK_ALL_CARDS` under R10; `MONITOR_CARD` / `STEP_UP_AUTH` / `VERIFY_WITH_CUSTOMER` / `MONITOR_CONNECTED_CARDS` otherwise).
   - Produces re-runnable `Attestation` objects for every plan line and frontier step.

## Measured Results on Real Rings from `tx_all()`
- **Ring 1** (Shared Device Profile: `ALE-L23 Build/HuaweiALE-L23 | ? | chrome 63.0 for android | ?`):
  - Total cards in ring: 3 (`C01784-K1`, `C00294-K1`, `C00294-K2`)
  - Total exposure cut: $399.92
  - Expected innocents touched: 0.00
  - Net gain ($1,000 penalty/innocent): $399.92
  - Cleared-only history customers blocked: 0
- **Ring 2** (Shared Device Profile: `SM-G532M Build/MMB29T | ? | Samsung/Android | ?`):
  - Total cards in ring: 3 (`C01389-K1`, `C00768-K1`, `C00768-K2`)
  - Total exposure cut: $1,249.80
  - Expected innocents touched: 0.00
  - Net gain ($1,000 penalty/innocent): $1,249.80
  - Cleared-only history customers blocked: 0

## Verification
- Innocence model correctly assigns probability $P(\text{innocent})$ with measured Jul-Oct empirical basis ($0.916$ for clean history + generic link, $0.985$ for cleared cases).
- Forward exposure measured from empirical medians in `closed_cases()` rather than uncalibrated heuristics.
- Per-card evidence strength dynamically calculated from card history, loss shape match, and link specificity.
- Zero policy breaches: No customer with only cleared history was assigned `BLOCK_CARD` or `BLOCK_ALL_CARDS`.
- All tests pass in < 4 seconds with 100% attestation coverage.
