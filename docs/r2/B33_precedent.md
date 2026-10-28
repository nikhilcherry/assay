# Block B33 — Precedent by Shared Entity

## 1. Overview & Design

`assay/r2/precedent.py` retrieves closed cases (`closed_at <= as_of`) linked to a target case `(card_id, flagged_txn_id, as_of)` across six entity link types, ranks them by outcome, link priority, and exponential recency weight ($\tau_{1/2} = 30\text{ days}$), and measures empirical likelihood ratios via an as-of backtest over the August–December 2016 closed cases.

For TigerGraph case read-back, `precedents_gsql()` provides the installed-query GSQL (`precedents_by_entity`) walking `Card`/`Customer`/`DeviceProfile`/`Transaction` $\rightarrow$ `FraudCase` with `fc.closed_at <= as_of`, while `precedents()` serves as the deterministic pandas reference implementation.

### Link Types & Cardinality Caps
1. **`same_card`**: Prior closed case on the same `card_id` (resolving the actual `card_id` of `flagged_txn_id` when the alert card label is mismatched, e.g., `HHF-011`).
2. **`same_customer`**: Prior closed case on the same `customer_id`.
3. **`connected_card`**: The target card appears in the prior case's `connected_card_ids`.
4. **`shared_device_60d`**: Shared non-generic `device_profile` (excluding generic OS/browser strings via `is_generic_device` and capped at $\le 50$ distinct cards across the dataset) used within 60 days.
5. **`shared_product_amount`**: Shared exact `(ProductCD, amount)` purchase on the flagged transaction (capped at $2 \le \text{distinct cards} \le 25$ and $\le 40$ total transactions to exclude generic price points).
6. **`shared_addr_email_7d`**: Shared `(addr1, R_emaildomain)` where `R_emaildomain` is not a consumer free-mail/ISP domain and the pair appears on $\le 25$ distinct cards, within 7 days.

---

## 2. As-Of Backtest on Aug–Dec 2016 Closed Cases

Replaying the August–December 2016 closed cases strictly as-of each case's `opened_at` (`precedent.closed_at <= case.opened_at`):

| Link Type | $N_{\text{linked}}$ | Confirmed Fraud | Cleared (Legit Hit) | $P(\text{fraud} \mid \text{link})$ [95% CI] | $P(\text{fraud} \mid \neg\text{link})$ [95% CI] | Odds LR [95% CI] |
|---|---:|---:|---:|---|---|---|
| `same_card` | 2,887 | 2,725 | 162 | 0.944 [0.935, 0.952] | 0.803 [0.779, 0.825] | **1.81** [1.60, 2.04] |
| `same_customer` | 2,906 | 2,742 | 164 | 0.944 [0.935, 0.951] | 0.802 [0.778, 0.823] | **1.80** [1.60, 2.02] |
| `connected_card` | 29 | 27 | 2 | 0.931 [0.780, 0.981] | 0.903 [0.893, 0.912] | **1.18** [0.33, 4.31] |
| `shared_device_60d` | 84 | 83 | 1 | **0.988** [0.936, 0.998] | 0.901 [0.891, 0.910] | **6.00** [1.20, 30.04] |
| `shared_product_amount` | 761 | 741 | 20 | **0.974** [0.960, 0.983] | 0.887 [0.875, 0.897] | **3.90** [2.54, 5.97] |
| `shared_addr_email_7d` | 2 | 2 | 0 | 1.000 [0.342, 1.000] | 0.903 [0.893, 0.912] | 0.54 [0.03, 11.20] |
| `any_fraud_precedent` | 3,053 | 2,876 | 177 | **0.942** [0.933, 0.950] | 0.786 [0.759, 0.810] | **1.75** [1.56, 1.95] |

### Cleared (Legitimate) Cases Acted Against per Link Type
- `shared_device_60d` (1 cleared case): `['CC-5578']`
- `connected_card` (2 cleared cases): `['CC-4027', 'CC-5035']`
- `shared_product_amount` (20 cleared cases): `['CC-1660', 'CC-1881', 'CC-2089', 'CC-2285', 'CC-2695', 'CC-3465', 'CC-3590', 'CC-3629', 'CC-3972', 'CC-4046', ...]`
- `same_card` (162 cleared cases) / `same_customer` (164 cleared cases): `['CC-1527', 'CC-1610', 'CC-1660', 'CC-1666', 'CC-1680', 'CC-1757', 'CC-1784', 'CC-1824', 'CC-1878', 'CC-2020', ...]`
- `shared_addr_email_7d` (0 cleared cases): `[]`

---

## 3. Verification & Reproducibility

`tests/r2/test_precedent.py` verifies:
1. **`precedents_gsql()` & Reproducibility**: Every `precedents()` query and `PrecedentDetector` attestation reproduces identical case IDs, transaction IDs, and paths when re-run with its recorded parameters.
2. **Strict As-Of Discipline**: No precedent returned for any of the 18 exam cases (or during the Aug–Dec backtest) has `closed_at > opened_at`.
3. **Full Exam Case Reporting**: Prints the HHG-derived precedent check (reporting `none found` when none match) and prints all precedents, link types, outcomes, and paths for all 18 exam cases (`HHF-001` … `HHF-018`).
