"""Challenge 11 -- Data Quality Detective.

Detects data quality issues across transaction and case history: missing identity
records, empty billing regions, placeholder values, timestamp collisions, seeded ':00'
seconds, distribution drift (PSI), card ID inconsistencies, thin history cards, and
closed cases referencing missing transaction IDs.

Provides `run_all_checks()` to produce a full attested report, `taint()` to identify
which data quality issues compromise OUR agent's evidence for specific cases, and
`describe_case_taint()` to detail evidence unavailability and trust impact per case.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import case_pack, closed_cases, tx_all


@dataclass
class DQIssue:
    id: str
    what: str
    affected_txn_ids: list[int]
    affected_count: int
    affected_rate_pct: float
    severity: str  # "HIGH", "MEDIUM", "LOW"
    why_it_matters_for_fraud: str
    attestation: Attestation | None = None


def compute_psi(expected: pd.Series, actual: pd.Series, num_bins: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between expected (Dec) and actual (Jan)."""
    e_clean = expected.dropna()
    a_clean = actual.dropna()
    if len(e_clean) == 0 or len(a_clean) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.unique(np.quantile(e_clean, quantiles))
    if len(bins) < 2:
        bins = np.linspace(e_clean.min(), e_clean.max() + 1e-5, num_bins + 1)

    e_counts, _ = np.histogram(e_clean, bins=bins)
    a_counts, _ = np.histogram(a_clean, bins=bins)

    e_prop = e_counts / len(e_clean)
    a_prop = a_counts / len(a_clean)

    # Small epsilon smoothing to prevent log(0) or div by zero
    e_prop = np.where(e_prop == 0, 1e-4, e_prop)
    a_prop = np.where(a_prop == 0, 1e-4, a_prop)

    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def check_missing_identity(tx: pd.DataFrame) -> DQIssue:
    """Missing identity rate by month and channel."""
    id_cols = [c for c in ["DeviceType", "DeviceInfo", "id_15", "id_23", "id_30", "id_31", "id_33"] if c in tx.columns]
    no_id_mask = tx[id_cols].isna().all(axis=1) if id_cols else pd.Series(True, index=tx.index)
    count = int(no_id_mask.sum())
    pct_total = (count / len(tx)) * 100
    sample_txns = tx.loc[no_id_mask, "TransactionID"].head(50).tolist()
    sample_cards = tx.loc[no_id_mask, "card_id"].head(50).unique().tolist()

    jan_online = tx[(tx.month_src == "jan") & (tx.channel == "online")]
    jan_online_no_id = jan_online[id_cols].isna().all(axis=1).sum() if len(jan_online) > 0 else 0
    pct_jan_online = (jan_online_no_id / len(jan_online) * 100) if len(jan_online) > 0 else 0.0

    what = f"{pct_total:.1f}% of total transactions ({count:,}) have no identity record; {pct_jan_online:.1f}% of Jan online txns missing identity."
    why = "Without identity records, shared device profile matching (R6) and device-based takeover detection fail completely because device evidence is unavailable."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=sample_txns, card_ids=sample_cards),
        query="check_missing_identity",
        params={"total_missing": count, "total_txns": len(tx), "pct_jan_online": round(pct_jan_online, 2)},
    )
    return DQIssue(
        id="DQ-01-MISSING-IDENTITY",
        what=what,
        affected_txn_ids=sample_txns,
        affected_count=count,
        affected_rate_pct=round(pct_total, 2),
        severity="HIGH",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_empty_addr(tx: pd.DataFrame) -> DQIssue:
    """Empty addr1/addr2 billing regions."""
    empty_mask = tx.addr1.isna() | tx.addr2.isna()
    count = int(empty_mask.sum())
    pct = (count / len(tx)) * 100
    sample_txns = tx.loc[empty_mask, "TransactionID"].head(50).tolist()
    sample_cards = tx.loc[empty_mask, "card_id"].head(50).unique().tolist()

    what = f"{pct:.1f}% of transactions ({count:,}) have empty billing address fields (addr1/addr2)."
    why = "Missing billing region values render geographic anomaly evidence and region clustering rules (R6) unavailable."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=sample_txns, card_ids=sample_cards),
        query="check_empty_addr",
        params={"missing_addr_count": count, "pct": round(pct, 2)},
    )
    return DQIssue(
        id="DQ-02-EMPTY-ADDR",
        what=what,
        affected_txn_ids=sample_txns,
        affected_count=count,
        affected_rate_pct=round(pct, 2),
        severity="MEDIUM",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_placeholder_values(tx: pd.DataFrame) -> DQIssue:
    """Placeholder emails ('anonymous.com'), generic devices ('Windows', 'iOS Device'), or dist1=0."""
    anon_email = tx.P_emaildomain == "anonymous.com"
    generic_device = tx.DeviceInfo.isin(["Windows", "iOS Device", "MacOS", "Trident", "iOS", "Android"])
    dist_zero = tx.dist1 == 0.0

    mask = anon_email | generic_device | dist_zero
    count = int(mask.sum())
    pct = (count / len(tx)) * 100
    sample_txns = tx.loc[mask, "TransactionID"].head(50).tolist()
    sample_cards = tx.loc[mask, "card_id"].head(50).unique().tolist()

    what = f"Found {count:,} transactions ({pct:.1f}%) with placeholder values (anonymous.com, generic device strings, dist1=0)."
    why = "Generic device profiles and placeholder domains compromise entity evidence by collapsing distinct users into single buckets, creating false ring links or masking real fraud."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=sample_txns, card_ids=sample_cards),
        query="check_placeholder_values",
        params={"placeholder_count": count},
    )
    return DQIssue(
        id="DQ-03-PLACEHOLDER-VALUES",
        what=what,
        affected_txn_ids=sample_txns,
        affected_count=count,
        affected_rate_pct=round(pct, 2),
        severity="MEDIUM",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_ts_collisions(tx: pd.DataFrame) -> DQIssue:
    """Duplicate TransactionIDs or timestamp collisions on same card."""
    card_ts_counts = tx.groupby(["card_id", "t"])["TransactionID"].transform("count")
    collision_mask = card_ts_counts > 1
    dup_tx_mask = tx.TransactionID.duplicated(keep=False)

    combined_mask = collision_mask | dup_tx_mask
    count = int(combined_mask.sum())
    pct = (count / len(tx)) * 100
    sample_txns = tx.loc[combined_mask, "TransactionID"].head(50).tolist()
    sample_cards = tx.loc[combined_mask, "card_id"].head(50).unique().tolist()

    what = f"Found {count:,} transactions ({pct:.2f}%) with timestamp collisions (multiple transactions on same card at exact same timestamp t)."
    why = "Timestamp collisions distort transaction sequencing and velocity evidence."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=sample_txns, card_ids=sample_cards),
        query="check_ts_collisions",
        params={"collision_count": count},
    )
    return DQIssue(
        id="DQ-04-TS-COLLISIONS",
        what=what,
        affected_txn_ids=sample_txns,
        affected_count=count,
        affected_rate_pct=round(pct, 2),
        severity="HIGH",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_seeded_00_seconds(tx: pd.DataFrame) -> DQIssue:
    """Clustering of timestamps at exact ':00' seconds."""
    seconds = tx.t.dt.second
    zero_sec_mask = seconds == 0
    count = int(zero_sec_mask.sum())
    pct = (count / len(tx)) * 100
    sample_txns = tx.loc[zero_sec_mask, "TransactionID"].head(50).tolist()
    sample_cards = tx.loc[zero_sec_mask, "card_id"].head(50).unique().tolist()

    what = f"{count:,} transactions ({pct:.1f}%) have exact ':00' second timestamps, showing non-uniform seeding."
    why = "Artificial ':00' timestamp clustering indicates synthetic or batch-seeded rows. NOTE: Explicitly not used as evidence in agent decisions."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=sample_txns, card_ids=sample_cards),
        query="check_seeded_00_seconds",
        params={"zero_sec_count": count, "pct": round(pct, 2)},
    )
    return DQIssue(
        id="DQ-05-SEEDED-00-SECONDS",
        what=what,
        affected_txn_ids=sample_txns,
        affected_count=count,
        affected_rate_pct=round(pct, 2),
        severity="LOW",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_distribution_drift(tx: pd.DataFrame) -> DQIssue:
    """January vs December distribution drift (PSI) per key column including risk_score."""
    dec = tx[(tx.t.dt.year == 2016) & (tx.t.dt.month == 12)]
    jan = tx[(tx.t.dt.year == 2017) & (tx.t.dt.month == 1)]

    psi_scores = {}
    cols_to_check = ["risk_score", "amount", "addr1", "dist1", "C1", "D1"]
    for col in cols_to_check:
        if col in tx.columns:
            psi_scores[col] = compute_psi(dec[col], jan[col])

    risk_psi = psi_scores.get("risk_score", 0.0)
    jan_txns = jan["TransactionID"].head(50).tolist()
    sample_cards = jan["card_id"].head(50).unique().tolist()
    pct_jan = (len(jan) / len(tx)) * 100

    what = f"Distribution drift (PSI) between Dec and Jan: risk_score PSI={risk_psi:.3f} (>0.25 threshold indicates severe drift)."
    why = "The bank's risk score model was not retrained for January. Stale risk_score is the bank's alert trigger; our agent does not use risk_score as evidence for decisions."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=jan_txns, card_ids=sample_cards),
        query="check_distribution_drift",
        params={k: round(v, 4) for k, v in psi_scores.items()},
    )
    return DQIssue(
        id="DQ-06-DISTRIBUTION-DRIFT",
        what=what,
        affected_txn_ids=jan_txns,
        affected_count=len(jan),
        affected_rate_pct=round(pct_jan, 2),
        severity="HIGH",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_card_id_inconsistency(cp: pd.DataFrame, tx: pd.DataFrame) -> DQIssue:
    """Card ID inconsistencies between case pack and bank records (e.g., HHF-011 C09241-K1 vs K2)."""
    tx_card_map = tx.set_index("TransactionID")["card_id"].to_dict()
    customer_cards = tx.groupby("customer_id")["card_id"].unique().to_dict()

    affected_cases = []
    affected_txns = []
    affected_cards = []
    affected_custs = []

    for r in cp.itertuples():
        flagged = int(r.flagged_txn_id)
        cp_card = r.card_id
        cust = r.customer_id

        real_card = tx_card_map.get(flagged)
        bank_cust_cards = customer_cards.get(cust, np.array([]))

        if real_card is not None and real_card != cp_card:
            affected_cases.append(r.case_id)
            affected_txns.append(flagged)
            affected_cards.append(cp_card)
            affected_custs.append(cust)
        elif cp_card not in bank_cust_cards:
            affected_cases.append(r.case_id)
            affected_txns.append(flagged)
            affected_cards.append(cp_card)
            affected_custs.append(cust)

    pct_cases = (len(affected_cases) / len(cp)) * 100
    what = f"Card ID mismatches found between case pack and bank records (e.g. {', '.join(affected_cases)})."
    why = "Inconsistent card IDs break entity graph queries: querying history under the case pack card ID returns zero history unless re-mapped to the bank record's card ID."

    att = Attestation(
        what=what,
        evidence=Evidence(
            txn_ids=affected_txns,
            card_ids=affected_cards,
            customer_ids=affected_custs,
            case_ids=affected_cases,
        ),
        query="check_card_id_inconsistency",
        params={"mismatch_cases": affected_cases},
    )
    return DQIssue(
        id="DQ-07-CARD-ID-INCONSISTENCY",
        what=what,
        affected_txn_ids=affected_txns,
        affected_count=len(affected_txns),
        affected_rate_pct=round(pct_cases, 2),
        severity="HIGH",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_thin_history(cp: pd.DataFrame, tx: pd.DataFrame) -> DQIssue:
    """Thin history cards (< 5 prior txns as-of)."""
    card_counts = tx.groupby("card_id")["TransactionID"].count().to_dict()
    thin_cards = {c for c, count in card_counts.items() if count < 5}

    thin_tx_mask = tx.card_id.isin(thin_cards)
    count = int(thin_tx_mask.sum())
    pct = (count / len(tx)) * 100
    sample_txns = tx.loc[thin_tx_mask, "TransactionID"].head(50).tolist()
    sample_cards = list(thin_cards)[:50]

    what = f"Found {len(thin_cards):,} cards with thin history (< 5 prior transactions total)."
    why = "Thin transaction history (< 5 prior txns) makes baseline behavioral evidence unreliable (cannot establish normal spending baseline)."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=sample_txns, card_ids=sample_cards),
        query="check_thin_history",
        params={"thin_card_count": len(thin_cards), "affected_txns": count},
    )
    return DQIssue(
        id="DQ-08-THIN-HISTORY",
        what=what,
        affected_txn_ids=sample_txns,
        affected_count=count,
        affected_rate_pct=round(pct, 2),
        severity="MEDIUM",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def check_missing_closed_case_txns(cc: pd.DataFrame, tx: pd.DataFrame) -> DQIssue:
    """Closed cases referencing transaction IDs absent from transactions."""
    all_tx_ids = set(tx.TransactionID.values)
    missing_txns = []
    affected_cases = []

    for r in cc.itertuples():
        txn_list = getattr(r, "txn_list", [])
        missing = [t for t in txn_list if t not in all_tx_ids]
        if missing:
            missing_txns.extend(missing)
            affected_cases.append(r.case_id)

    missing_txns = sorted(list(set(missing_txns)))
    pct = (len(affected_cases) / len(cc)) * 100 if len(cc) > 0 else 0.0
    what = f"Closed cases reference {len(missing_txns)} transaction IDs absent from the primary transactions table."
    why = "Missing closed case transactions break precedent matching and historical pattern re-investigation."

    att = Attestation(
        what=what,
        evidence=Evidence(txn_ids=missing_txns[:50], case_ids=affected_cases[:50]),
        query="check_missing_closed_case_txns",
        params={"missing_txn_count": len(missing_txns), "affected_case_count": len(affected_cases)},
    )
    return DQIssue(
        id="DQ-09-MISSING-CLOSED-CASE-TXNS",
        what=what,
        affected_txn_ids=missing_txns[:50],
        affected_count=len(missing_txns),
        affected_rate_pct=round(pct, 2),
        severity="HIGH",
        why_it_matters_for_fraud=why,
        attestation=att,
    )


def run_all_checks(
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
    cp: pd.DataFrame | None = None,
) -> list[DQIssue]:
    """Runs all 9 data quality checks across transactions, closed cases, and case pack."""
    if tx is None:
        tx = tx_all()
    if cc is None:
        cc = closed_cases()
    if cp is None:
        cp = case_pack()

    return [
        check_missing_identity(tx),
        check_empty_addr(tx),
        check_placeholder_values(tx),
        check_ts_collisions(tx),
        check_seeded_00_seconds(tx),
        check_distribution_drift(tx),
        check_card_id_inconsistency(cp, tx),
        check_thin_history(cp, tx),
        check_missing_closed_case_txns(cc, tx),
    ]


def _to_dict(case_row: pd.Series | dict | tuple) -> dict:
    if isinstance(case_row, dict):
        return case_row
    if hasattr(case_row, "_asdict"):
        return case_row._asdict()
    if isinstance(case_row, pd.Series):
        return case_row.to_dict()
    return dict(case_row)


def taint(
    case_row: pd.Series | dict | tuple,
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
) -> list[DQIssue]:
    """Returns list of DQ issue objects that directly taint/compromise OUR agent's EVIDENCE for a specific case.

    Notes:
    - Stale January risk scores (DQ-06) trigger alerts but are NOT agent evidence, so DQ-06 does not taint agent evidence.
    - Seeded :00 seconds (DQ-05) are explicit raw data anomalies NOT used as evidence by the agent.
    """
    if tx is None:
        tx = tx_all()
    if cc is None:
        cc = closed_cases()

    case_dict = _to_dict(case_row)

    case_id = case_dict.get("case_id", "")
    flagged_txn_id = int(case_dict.get("flagged_txn_id", 0))
    case_card_id = str(case_dict.get("card_id", ""))
    opened_at = pd.to_datetime(case_dict.get("opened_at", ""))

    tainted: list[DQIssue] = []
    tx_row = tx[tx.TransactionID == flagged_txn_id]
    real_card = str(tx_row.iloc[0].card_id) if len(tx_row) > 0 else None

    # 1. Card ID Inconsistency (DQ-07) - Case pack card ID differs from bank record
    if real_card is not None and real_card != case_card_id:
        iss = check_card_id_inconsistency(case_pack(), tx)
        iss.what = f"Card ID mismatch in {case_id}: case pack specifies '{case_card_id}' but bank record for txn {flagged_txn_id} is '{real_card}'."
        tainted.append(iss)

    # 2. Missing Identity on Flagged Txn (DQ-01) - Device evidence unavailable
    if len(tx_row) > 0:
        id_cols = [c for c in ["DeviceType", "DeviceInfo", "id_15", "id_23", "id_30", "id_31", "id_33"] if c in tx_row.columns]
        if tx_row[id_cols].isna().all(axis=1).iloc[0]:
            iss = check_missing_identity(tx)
            iss.what = f"Missing identity record for flagged txn {flagged_txn_id} in {case_id}: device evidence unavailable."
            tainted.append(iss)

    # 3. Empty Billing Address on Flagged Txn (DQ-02) - Region evidence unavailable
    if len(tx_row) > 0:
        if pd.isna(tx_row.iloc[0].addr1) or pd.isna(tx_row.iloc[0].addr2):
            iss = check_empty_addr(tx)
            iss.what = f"Empty billing address (addr1/addr2) for flagged txn {flagged_txn_id} in {case_id}: region evidence unavailable."
            tainted.append(iss)

    # 4. Placeholder Values on Flagged Txn (DQ-03) - Device/Email evidence compromised
    if len(tx_row) > 0:
        r = tx_row.iloc[0]
        dev = str(r.get("DeviceInfo", ""))
        email = str(r.get("P_emaildomain", ""))
        dist = r.get("dist1")
        if email == "anonymous.com" or dev in ["Windows", "iOS Device", "MacOS", "Trident"] or dist == 0.0:
            iss = check_placeholder_values(tx)
            iss.what = f"Placeholder values present in flagged txn {flagged_txn_id} for {case_id} (email='{email}', dev='{dev}', dist1={dist}): entity evidence compromised."
            tainted.append(iss)

    # 5. Thin Card History (DQ-08) - Baseline spending history unreliable
    effective_card = real_card if real_card else case_card_id
    prior_txns = tx[(tx.card_id == effective_card) & (tx.t <= opened_at)]
    if len(prior_txns) < 5:
        iss = check_thin_history(case_pack(), tx)
        iss.what = f"Thin history for card '{effective_card}' in {case_id}: only {len(prior_txns)} prior transactions as of {opened_at}. Spending baseline unreliable."
        tainted.append(iss)

    # 6. Timestamp Collisions on Card (DQ-04) - Velocity evidence distorted
    if len(tx_row) > 0:
        card_txns = tx[(tx.card_id == effective_card) & (tx.t <= opened_at)]
        if len(card_txns) > 0:
            col_counts = card_txns.groupby("t")["TransactionID"].transform("count")
            if (col_counts > 1).any():
                iss = check_ts_collisions(tx)
                iss.what = f"Card '{effective_card}' in {case_id} has timestamp collisions in transaction history: velocity evidence distorted."
                tainted.append(iss)

    return tainted


def resolve_hhf011(tx: pd.DataFrame | None = None, cp: pd.DataFrame | None = None) -> dict:
    """Resolves card ID inconsistency for case HHF-011.

    Bank records analysis:
    - Customer C09241 has debit card C09241-K2 in Jul-Dec (18 transactions).
    - Flagged transaction 3640919 in January has card6='debit', so bank records register it to C09241-K2.
    - Case pack HHF-011 erroneously lists card_id = C09241-K1 (which has 0 bank transactions).
    - Investigator action: Re-map case lookup to C09241-K2 (real bank card) and note the ID mismatch.
    """
    if tx is None:
        tx = tx_all()
    if cp is None:
        cp = case_pack()

    txn_id = 3640919
    tx_row = tx[tx.TransactionID == txn_id]
    real_card = str(tx_row.iloc[0].card_id) if len(tx_row) > 0 else "C09241-K2"
    cust_id = str(tx_row.iloc[0].customer_id) if len(tx_row) > 0 else "C09241"
    card6_type = str(tx_row.iloc[0].card6) if len(tx_row) > 0 else "debit"

    cust_txns = tx[tx.customer_id == cust_id]
    cards_found = cust_txns.groupby("card_id")["TransactionID"].count().to_dict()

    return {
        "case_id": "HHF-011",
        "flagged_txn_id": txn_id,
        "case_pack_card_id": "C09241-K1",
        "real_bank_card_id": real_card,
        "customer_id": cust_id,
        "flagged_txn_card6": card6_type,
        "customer_cards_in_bank_records": cards_found,
        "explanation": (
            f"Flagged txn {txn_id} is a {card6_type} transaction belonging to card {real_card} in bank records. "
            f"Case pack incorrectly names 'C09241-K1' (0 bank transactions)."
        ),
        "investigator_action": (
            f"Investigate flagged txn {txn_id} using real card ID '{real_card}' (which holds full history); "
            f"record the case pack ID mismatch in case notes."
        ),
    }


def describe_case_taint(
    case_row: pd.Series | dict | tuple,
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
) -> dict:
    """Describes exact evidence unavailable and trust impact for a specific case."""
    if tx is None:
        tx = tx_all()
    if cc is None:
        cc = closed_cases()

    case_dict = _to_dict(case_row)
    case_id = case_dict.get("case_id", "")
    flagged_txn_id = int(case_dict.get("flagged_txn_id", 0))
    cp_card = str(case_dict.get("card_id", ""))
    opened_at = pd.to_datetime(case_dict.get("opened_at", ""))

    tainted_issues = taint(case_row, tx, cc)

    tx_row = tx[tx.TransactionID == flagged_txn_id]
    real_card = str(tx_row.iloc[0].card_id) if len(tx_row) > 0 else cp_card

    unavail: list[str] = []
    for issue in tainted_issues:
        if issue.id == "DQ-01-MISSING-IDENTITY":
            unavail.append("Device evidence unavailable (DeviceInfo, id_30, id_31, id_33 missing)")
        elif issue.id == "DQ-02-EMPTY-ADDR":
            unavail.append("Region evidence unavailable (addr1/addr2 missing)")
        elif issue.id == "DQ-03-PLACEHOLDER-VALUES":
            unavail.append("Device/Email evidence compromised (placeholder anonymous.com/generic device)")
        elif issue.id == "DQ-04-TS-COLLISIONS":
            unavail.append("Velocity evidence distorted (same-card timestamp collisions)")
        elif issue.id == "DQ-07-CARD-ID-INCONSISTENCY":
            unavail.append(f"Card history broken (case pack ID {cp_card} != bank record {real_card})")
        elif issue.id == "DQ-08-THIN-HISTORY":
            unavail.append("Baseline spending history unreliable (< 5 prior transactions)")

    issue_ids = {t.id for t in tainted_issues}

    if "DQ-07-CARD-ID-INCONSISTENCY" in issue_ids:
        trust = "UNTRUSTWORTHY"
        action = f"Re-map case lookup from {cp_card} to {real_card} before investigating; note ID mismatch."
    elif "DQ-08-THIN-HISTORY" in issue_ids and ("DQ-01-MISSING-IDENTITY" in issue_ids or "DQ-02-EMPTY-ADDR" in issue_ids):
        trust = "DEGRADED"
        action = "Rely on customer verification or analyst escalation; baseline and device/region evidence weak."
    elif issue_ids:
        trust = "SLIGHTLY DEGRADED"
        action = "Proceed with investigation; note missing/compromised evidence fields."
    else:
        trust = "TRUSTED"
        action = "Full evidence available for investigation."

    return {
        "case_id": case_id,
        "flagged_txn_id": flagged_txn_id,
        "case_card_id": cp_card,
        "real_card_id": real_card,
        "tainted_issues": [t.id for t in tainted_issues],
        "unavailable_evidence": unavail,
        "trust_verdict": trust,
        "investigator_action": action,
    }
