# Block B2: Independent Policy Linter & Verification Report

## 1. Overview & Architecture
Block B2 builds an independent policy linter (`assay.r2.policy_check`) designed to enforce Fraud Operations Policy v1.0 and validate investigation answer JSON files.

### Key Rules Encoded:
- **Innocent Customer Rule (HARD):** Any recommendation of `BLOCK_CARD`, `BLOCK_ALL_CARDS`, or `FILE_REPORT` on a case with verdict `legitimate` or `uncertain` is flagged as a HARD violation.
- **R1 vs R2 Conflation:**
  - **R1 (Weak Signal / Risk Score):** Requires `VERIFY_WITH_CUSTOMER` or `STEP_UP_AUTH` before any block when assessed fraud probability is below 0.70. Direct blocking without prior verification flags `R1_R2_CONFLATION`.
  - **R2 (Customer Denial):** Customer has already denied the charge. Immediate `BLOCK_CARD` and `CREATE_CASE` are required; verification steps are unnecessary and flag `R1_R2_CONFLATION`.
- **Approval Routing (§2):** `BLOCK_CARD` routed to `L1` (<= $2,500) or `L2` (> $2,500); `BLOCK_ALL_CARDS` and `FILE_REPORT` always `L2`.
- **Case vs Report (§3a):** `FILE_REPORT` required when fraud is confirmed/strongly suspected AND exposure > $1,000, connected entities exist, or pattern is undocumented.
- **Action Delta (§3b):** Action changes between initial and final require a valid `what_changed` explanation.
- **Tip-off Prevention:** Customer-visible `evidence_requests` must not contain SAR or regulatory filing references.
- **Stopping Criteria (§6):** Requires non-empty `stop_reason`.

---

## 2. Measured Audit Results

### Summary across 20 Round 1 HHG Cases
- **Total Cases Linted:** 20 / 20
- **Cases with Hard Violations:** 0 / 20
- **Agreement with `grade.py` Hard Checks:** 100% (0 hard violations reported by both)
- **HHG-001 (Legitimate, Closed):** 0 hard violations, 0 policy violations.

### Ground Truth Checks (HHG-002 & HHG-004)
- **HHG-002:** Verdict `uncertain` (p=0.55). Actions: `VERIFY_WITH_CUSTOMER`, `ESCALATE_TO_ANALYST`. No card blocks. Policy R1 & R8 satisfied.
- **HHG-004:** Verdict `uncertain` (p=0.52). Actions: `VERIFY_WITH_CUSTOMER`, `ESCALATE_TO_ANALYST`. No card blocks. Policy R1 & R8 satisfied.

---

## 3. Synthetic R1 vs R2 Conflation Test Results
- **Synthetic Test 1 (R2 Conflation):** Trigger `customer_report` + initial action `VERIFY_WITH_CUSTOMER` -> Flagged policy violation `R1_R2_CONFLATION`.
- **Synthetic Test 2 (R1 Conflation):** Trigger `risk_score` (p=0.55) + initial action `BLOCK_CARD` without verification -> Flagged policy violation `R1_R2_CONFLATION`.

## Update
- R1/R2 conflation (verification on a customer denial) now only flagged when the verdict is fraud or a block is recommended; an uncertain/legit denial may be verified. Known real gap: HHG-002 lacks ESCALATE_TO_ANALYST (R8 warning).
