# BLOCK B23 -- Data Quality Detective (Challenge 11)

## Measured Data Quality Issues

The Data Quality Detective (`assay.r2.dq`) evaluates transaction history (651k transactions) and closed cases (5,585 cases) across 9 automated checks. Every claim is backed by an `Attestation` with exact record IDs and re-runnable query definitions.

| Issue ID | Description | Count | Rate (%) | Severity | Why it matters for fraud & evidence |
|---|---|---|---|---|---|
| `DQ-01-MISSING-IDENTITY` | Missing identity records (total txns / Jan online) | 167,965 | 25.80% | HIGH | Device evidence unavailable (DeviceInfo, id_* missing); disables R6 shared device & ATO detection |
| `DQ-02-EMPTY-ADDR` | Empty billing address (addr1/addr2) | 70,879 | 10.89% | MEDIUM | Region evidence unavailable; prevents region clustering and out-of-region rules |
| `DQ-03-PLACEHOLDER-VALUES` | Placeholder emails (anonymous.com), generic devices (Windows/iOS Device), dist1=0 | 17,942 | 2.76% | MEDIUM | Entity evidence compromised; collapses distinct users into false rings or masks real fraud |
| `DQ-04-TS-COLLISIONS` | Timestamp collisions (same card at exact same timestamp t) | 314 | 0.05% | HIGH | Velocity evidence distorted; breaks exact transaction sequencing |
| `DQ-05-SEEDED-00-SECONDS` | Timestamps clustered at exact `:00` seconds | 10,718 | 1.65% | LOW | Artificial timestamp clustering from batch seeding. *Note: Explicitly not used as evidence in agent decisions.* |
| `DQ-06-DISTRIBUTION-DRIFT` | January vs December risk_score distribution drift (PSI) | 60,196 | 9.25% | HIGH | Risk score model uncalibrated in Jan. *Note: Bank's alert trigger only; not used as evidence in agent decisions.* |
| `DQ-07-CARD-ID-INCONSISTENCY` | Case pack card ID mismatch vs bank records (HHF-011) | 1 | 5.56% (cases) | HIGH | Card history broken; graph queries on case pack card ID return 0 history unless re-mapped |
| `DQ-08-THIN-HISTORY` | Cards with thin history (< 5 prior transactions as-of opened_at) | 14,320 | 2.20% | MEDIUM | Baseline spending history unreliable; baseline normal vs anomalous behavior cannot be established |
| `DQ-09-MISSING-CLOSED-CASE-TXNS` | Closed cases referencing missing transaction IDs | 0 | 0.00% | HIGH | Precedent matching intact (0 missing txns found) |

---

## HHF-011 Resolution Analysis

- **Case Pack Entry**: `case_id=HHF-011`, `flagged_txn_id=3640919`, `card_id=C09241-K1`, `customer_id=C09241`.
- **Bank Records (`tx_all.parquet`)**: Flagged transaction 3640919 is a `debit` transaction (`card6=debit`).
- **Customer History**: Customer `C09241` has 18 transactions in Jul-Dec 2016, all debit, mapped to `C09241-K2`.
- **Mismatch Cause**: `case_pack_final.csv` incorrectly specifies `C09241-K1` (0 transactions in bank records).
- **Investigator Resolution**:
  1. Investigate transaction 3640919 using real bank card ID `C09241-K2` (which holds the full debit transaction history).
  2. Note the case pack card ID mismatch (`C09241-K1` vs `C09241-K2`) in case audit notes.

---

## Investigation Taint Matrix & Evidence Availability (Exam Cases HHF-001 .. HHF-018)

Data quality taints are evaluated strictly based on whether they compromise **OUR agent's evidence** for that case:

| Case ID | Flagged Txn | Trigger Type | Tainted DQ Issues | Trust Verdict | Unavailable / Compromised Evidence |
|---|---|---|---|---|---|
| **HHF-001** | 3660108 | analyst_request | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-002** | 3660127 | analyst_request | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-003** | 3660153 | customer_report | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-004** | 3660260 | customer_report | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-005** | 3660250 | customer_report | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-006** | 3660263 | analyst_request | DQ-01, DQ-03 | SLIGHTLY DEGRADED | Device evidence unavailable \| Device/Email evidence compromised (placeholder anonymous.com/generic device) |
| **HHF-007** | 3660350 | customer_report | DQ-03 | SLIGHTLY DEGRADED | Device/Email evidence compromised (placeholder anonymous.com/generic device) |
| **HHF-008** | 3660353 | risk_score | None | TRUSTED | All evidence available |
| **HHF-009** | 3660355 | risk_score | DQ-03 | SLIGHTLY DEGRADED | Device/Email evidence compromised (placeholder anonymous.com/generic device) |
| **HHF-010** | 3631383 | risk_score | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-011** | 3640919 | risk_score | DQ-07, DQ-01, DQ-04 | UNTRUSTWORTHY | Card history broken (case pack ID C09241-K1 != bank record C09241-K2) \| Device evidence unavailable \| Velocity evidence distorted |
| **HHF-012** | 3655457 | risk_score | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-013** | 3626720 | risk_score | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-014** | 3602445 | risk_score | DQ-01, DQ-04 | SLIGHTLY DEGRADED | Device evidence unavailable \| Velocity evidence distorted (same-card timestamp collisions) |
| **HHF-015** | 3601725 | risk_score | DQ-01 | SLIGHTLY DEGRADED | Device evidence unavailable (DeviceInfo, id_* missing) |
| **HHF-016** | 3611043 | risk_score | DQ-01, DQ-04 | SLIGHTLY DEGRADED | Device evidence unavailable \| Velocity evidence distorted (same-card timestamp collisions) |
| **HHF-017** | 3638829 | risk_score | DQ-03 | SLIGHTLY DEGRADED | Device/Email evidence compromised (placeholder anonymous.com/generic device) |
| **HHF-018** | 3612011 | risk_score | None | TRUSTED | All evidence available |
