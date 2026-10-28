"""BLOCK B20 -- Analyst in the loop (challenge 8): Accept and reject on findings,
and behaviour changes from analyst decisions without retraining a model.

Attested: when behaviour changes, record which analyst decisions caused it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import CaseCtx, Detector, Finding
from assay.r2.frames import closed_cases, tx_all


def get_amount_band(amount: float) -> str:
    """Classifies transaction amount into standard amount bands."""
    if amount < 50.0:
        return "<50"
    elif amount < 200.0:
        return "50-200"
    elif amount < 1000.0:
        return "200-1000"
    else:
        return ">=1000"


def compute_finding_signature(
    detector: str,
    pattern: str,
    element_type: str = "",
    element_value: str = "",
    amount_band: str = "",
) -> str:
    """Computes a stable deterministic hash of (detector, pattern, element_type+value, amount_band)."""
    raw = f"{detector}:{pattern}:{element_type}:{element_value}:{amount_band}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_signature_and_elements(
    finding: Finding, card_id: str = ""
) -> tuple[str, str, str, str]:
    """Extracts (finding_signature, element_type, element_value, amount_band) from a Finding."""
    params = finding.claim.params if finding.claim else {}
    et = str(params.get("element_type", ""))
    ev = str(params.get("element_value", ""))
    ab = str(params.get("amount_band", ""))
    if not ab and "amount" in params:
        try:
            ab = get_amount_band(float(params["amount"]))
        except (ValueError, TypeError):
            ab = "unknown"
    if not et and card_id:
        et = "card_id"
        ev = card_id
    sig = compute_finding_signature(finding.detector, finding.pattern, et, ev, ab)
    return sig, et, ev, ab


@dataclass
class FeedbackDecision:
    """An analyst decision recorded in FeedbackStore."""

    decision_id: str = ""
    finding_signature: str = ""
    detector: str = ""
    action: str = "reject"  # "accept" | "reject"
    analyst: str = "analyst_l1"
    reason: str = ""
    scope: str = "this_element_value"  # "this_card" | "this_element_value" | "detector_wide"
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    card_id: str = ""
    element_type: str = ""
    element_value: str = ""
    amount_band: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    from_dict = classmethod(lambda cls, d: cls(**d))


class FeedbackStore:
    """Append-only store for analyst decisions (JSONL on disk + in-memory index)."""

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.filepath = Path(filepath) if filepath else None
        self._decisions: list[FeedbackDecision] = []
        if self.filepath and self.filepath.exists():
            self._load()

    def _load(self) -> None:
        if not self.filepath:
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._decisions.append(FeedbackDecision.from_dict(json.loads(line)))

    def record_decision(self, decision: FeedbackDecision) -> str:
        """Appends an analyst decision to store and returns its decision_id."""
        if not decision.decision_id:
            h = hashlib.md5(
                f"{decision.ts}:{decision.finding_signature}:{decision.element_value}:{len(self._decisions)}".encode(
                    "utf-8"
                )
            ).hexdigest()[:8]
            decision.decision_id = f"DEC-{len(self._decisions) + 1:04d}-{h}"
        self._decisions.append(decision)
        if self.filepath:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
        return decision.decision_id

    def list_decisions(self) -> list[FeedbackDecision]:
        return list(self._decisions)

    def _matches(
        self,
        decision: FeedbackDecision,
        finding: Finding,
        sig: str,
        et: str,
        ev: str,
        card_id: str,
    ) -> bool:
        if decision.detector != finding.detector:
            return False
        if decision.scope == "detector_wide":
            return True
        if decision.scope == "this_element_value":
            if decision.finding_signature and decision.finding_signature == sig:
                return True
            if (
                decision.element_type
                and decision.element_value
                and decision.element_type == et
                and decision.element_value == ev
            ):
                return True
            return False
        if decision.scope == "this_card":
            target_cards = set(finding.connected_card_ids)
            if card_id:
                target_cards.add(card_id)
            if finding.claim and finding.claim.evidence:
                target_cards.update(finding.claim.evidence.card_ids)
            if decision.card_id and decision.card_id in target_cards:
                return True
            return False
        return False

    def apply(self, findings: list[Finding], card_id: str = "") -> list[Finding]:
        """Adjusts findings based on analyst decisions.

        Suppression only applies where Beta(1+accepts, 1+rejects) posterior mean of fraud <= 0.2
        AND rejects >= 3. Otherwise leaves the finding alone.
        """
        adjusted: list[Finding] = []
        for f in (copy.deepcopy(x) for x in findings):
            sig, et, ev, ab = extract_signature_and_elements(f, card_id)
            matching = [
                d for d in self._decisions if self._matches(d, f, sig, et, ev, card_id)
            ]
            if not matching:
                setattr(f, "causal_decision_ids", [])
                if f.claim:
                    f.claim.params["causal_decision_ids"] = []
                adjusted.append(f)
                continue

            decision_ids = [d.decision_id for d in matching]
            n_accepts = sum(1 for d in matching if d.action == "accept")
            n_rejects = sum(1 for d in matching if d.action == "reject")

            posterior_mean = (1.0 + n_accepts) / (2.0 + n_accepts + n_rejects)

            # Suppress finding if element value is verified mostly legit (rejects >= 3 and posterior_mean <= 0.2)
            if n_rejects >= 3 and posterior_mean <= 0.2:
                f.lr = 1.0
                if f.claim:
                    f.claim.params["causal_decision_ids"] = decision_ids
                    f.claim.params["adjusted_lr"] = 1.0
                setattr(f, "causal_decision_ids", decision_ids)
            else:
                setattr(f, "causal_decision_ids", [])
                if f.claim:
                    f.claim.params["causal_decision_ids"] = []

            adjusted.append(f)
        return adjusted


class NoisyToyDetector(Detector):
    """A deliberately noisy toy detector ('high amount + device profile') used for

    demonstrating analyst feedback rounds.
    """

    name = "noisy_toy_detector"
    pattern = "undocumented:noisy_new_device_high_amount"

    def run(self, ctx: CaseCtx) -> list[Finding]:
        if ctx.tx is None or ctx.tx.empty:
            return []

        card_tx = ctx.tx[ctx.tx.card_id == ctx.card_id]
        if card_tx.empty:
            return []

        matching = card_tx[
            (card_tx.amount > 150.0) & (card_tx.device_profile.fillna("") != "")
        ]
        if matching.empty:
            matching = card_tx[card_tx.amount > 150.0]
        if matching.empty:
            return []

        row = matching.iloc[-1]
        txn_id = int(row.TransactionID)
        dev = (
            str(row.device_profile)
            if pd.notna(row.device_profile) and str(row.device_profile) != ""
            else "unknown_device"
        )
        amt = float(row.amount)

        att = Attestation(
            what=f"Noisy toy detector fired on txn {txn_id} (amount=${amt:.2f}, device={dev})",
            evidence=Evidence(
                txn_ids=[txn_id],
                card_ids=[ctx.card_id],
                customer_ids=[ctx.customer_id],
                case_ids=[ctx.case_id],
            ),
            query="assay.r2.feedback.NoisyToyDetector.run",
            params={
                "detector": self.name,
                "pattern": self.pattern,
                "element_type": "device_profile",
                "element_value": dev,
                "amount": amt,
                "amount_band": get_amount_band(amt),
            },
        )
        finding = Finding(
            detector=self.name,
            pattern=self.pattern,
            claim=att,
            lr=5.0,
            affected_txn_ids=[txn_id],
            connected_card_ids=[ctx.card_id],
        )
        return [finding]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Any:
        return None


def run_feedback_demonstration(
    store_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs a 1-round analyst feedback demonstration:

    1. Analyst reviews the TOP 100 highest-scoring alerts from Jul-Aug (top by amount/score).
    2. Simulates analyst feedback: accept on confirmed_fraud, reject on cleared/legit.
    3. Posterior suppression rule: Beta(1+accepts, 1+rejects) mean <= 0.2 AND rejects >= 3.
    4. Evaluates Sep-Oct 2016 cases + negative card-days population before & after feedback.
    5. Reports wrongly acted against before/after, fraud caught before/after, precision change,
       and details 3 explicit decision IDs and the exact Sep-Oct findings they changed.
    """
    tx = tx_all()
    cc = closed_cases().copy()
    cc["opened_t"] = pd.to_datetime(cc.opened_at)

    jul_aug_cases = cc[
        (cc.opened_t >= "2016-07-01") & (cc.opened_t <= "2016-08-31 23:59:59")
    ].copy()
    sep_oct_cases = cc[
        (cc.opened_t >= "2016-09-01") & (cc.opened_t <= "2016-10-31 23:59:59")
    ].copy()

    all_case_cards = set(cc.card_id.unique())
    legit_cards = set(tx.card_id.unique()) - all_case_cards

    tx_legit = tx[tx.card_id.isin(legit_cards) & (tx.month_src == "r1")].copy()
    tx_legit["date"] = tx_legit.t.dt.date

    tx_legit_candidates = tx_legit[tx_legit.amount > 150.0]

    jul_aug_legit_days = tx_legit_candidates[
        (tx_legit_candidates.t >= "2016-07-01") & (tx_legit_candidates.t <= "2016-08-31 23:59:59")
    ].groupby(["card_id", "date"])["t"].max().reset_index()

    sep_oct_legit_days = tx_legit_candidates[
        (tx_legit_candidates.t >= "2016-09-01") & (tx_legit_candidates.t <= "2016-10-31 23:59:59")
    ].groupby(["card_id", "date"])["t"].max().reset_index()

    detector = NoisyToyDetector()
    card_tx_map = {cid: df for cid, df in tx.groupby("card_id")}

    # Gather all alerts generated in Jul-Aug
    jul_aug_alerts = []

    for row in jul_aug_cases.itertuples():
        opened_at = row.opened_t
        card_df = card_tx_map.get(row.card_id)
        sub_tx = card_df[card_df.t <= opened_at] if card_df is not None else pd.DataFrame()

        ctx = CaseCtx(
            case_id=row.case_id,
            flagged_txn_id=row.txn_list[0] if row.txn_list else 0,
            card_id=row.card_id,
            customer_id=row.customer_id,
            opened_at=opened_at,
            tx=sub_tx,
            cc=cc[pd.to_datetime(cc.closed_at) <= opened_at],
        )
        findings = detector.run(ctx)
        for f in findings:
            amt = float(f.claim.params.get("amount", 0.0))
            jul_aug_alerts.append(
                {
                    "case_id": row.case_id,
                    "card_id": row.card_id,
                    "opened_at": opened_at,
                    "outcome": row.outcome,
                    "finding": f,
                    "amount": amt,
                }
            )

    for row in jul_aug_legit_days.itertuples():
        opened_at = row.t
        card_df = card_tx_map.get(row.card_id)
        sub_tx = card_df[card_df.t <= opened_at] if card_df is not None else pd.DataFrame()
        cust_id = sub_tx.customer_id.iloc[0] if not sub_tx.empty else "UNKNOWN"

        ctx = CaseCtx(
            case_id=f"LEGIT-DAY-{row.card_id}-{row.date}",
            flagged_txn_id=0,
            card_id=row.card_id,
            customer_id=cust_id,
            opened_at=opened_at,
            tx=sub_tx,
            cc=cc[pd.to_datetime(cc.closed_at) <= opened_at],
        )
        findings = detector.run(ctx)
        for f in findings:
            amt = float(f.claim.params.get("amount", 0.0))
            jul_aug_alerts.append(
                {
                    "case_id": f"LEGIT-DAY-{row.card_id}-{row.date}",
                    "card_id": row.card_id,
                    "opened_at": opened_at,
                    "outcome": "cleared",
                    "finding": f,
                    "amount": amt,
                }
            )

    # Sort alerts by amount descending and select the TOP 100 alerts for the 1 analyst feedback round
    jul_aug_alerts.sort(key=lambda x: x["amount"], reverse=True)
    top_100_alerts = jul_aug_alerts[:100]

    store = FeedbackStore(store_path)

    for alert in top_100_alerts:
        f = alert["finding"]
        sig, et, ev, ab = extract_signature_and_elements(f, alert["card_id"])
        action = "accept" if alert["outcome"] == "confirmed_fraud" else "reject"
        reason = (
            "Confirmed fraud in top 100 alert"
            if action == "accept"
            else "Cleared/legit customer in top 100 alert"
        )
        dec = FeedbackDecision(
            finding_signature=sig,
            detector=detector.name,
            action=action,
            analyst="analyst_l1",
            reason=reason,
            scope="this_element_value",
            ts=alert["opened_at"].isoformat(),
            card_id=alert["card_id"],
            element_type=et,
            element_value=ev,
            amount_band=ab,
        )
        store.record_decision(dec)

    # Evaluate on Sep-Oct cases and negative card-days population
    wrongly_acted_before = []
    fraud_caught_before = []

    wrongly_acted_after = []
    fraud_caught_after = []

    changed_findings_log = []

    for row in sep_oct_cases.itertuples():
        opened_at = row.opened_t
        card_df = card_tx_map.get(row.card_id)
        sub_tx = card_df[card_df.t <= opened_at] if card_df is not None else pd.DataFrame()

        ctx = CaseCtx(
            case_id=row.case_id,
            flagged_txn_id=row.txn_list[0] if row.txn_list else 0,
            card_id=row.card_id,
            customer_id=row.customer_id,
            opened_at=opened_at,
            tx=sub_tx,
            cc=cc[pd.to_datetime(cc.closed_at) <= opened_at],
        )
        findings_raw = detector.run(ctx)

        active_before = [f for f in findings_raw if f.lr > 1.0]
        if active_before:
            if row.outcome == "cleared":
                wrongly_acted_before.append(row.case_id)
            elif row.outcome == "confirmed_fraud":
                fraud_caught_before.append(row.case_id)

        findings_after = store.apply(findings_raw, card_id=row.card_id)
        active_after = [f for f in findings_after if f.lr > 1.0]

        for f_b, f_a in zip(findings_raw, findings_after):
            cids = getattr(f_a, "causal_decision_ids", [])
            if cids and f_b.lr > 1.0 and f_a.lr == 1.0:
                params = f_a.claim.params if f_a.claim else {}
                changed_findings_log.append(
                    {
                        "case_id": row.case_id,
                        "outcome": row.outcome,
                        "element_value": params.get("element_value", ""),
                        "before_lr": f_b.lr,
                        "after_lr": f_a.lr,
                        "causal_decision_ids": cids,
                    }
                )

        if active_after:
            if row.outcome == "cleared":
                wrongly_acted_after.append(row.case_id)
            elif row.outcome == "confirmed_fraud":
                fraud_caught_after.append(row.case_id)

    for row in sep_oct_legit_days.itertuples():
        opened_at = row.t
        card_df = card_tx_map.get(row.card_id)
        sub_tx = card_df[card_df.t <= opened_at] if card_df is not None else pd.DataFrame()
        cust_id = sub_tx.customer_id.iloc[0] if not sub_tx.empty else "UNKNOWN"

        case_key = f"LEGIT-DAY-{row.card_id}-{row.date}"
        ctx = CaseCtx(
            case_id=case_key,
            flagged_txn_id=0,
            card_id=row.card_id,
            customer_id=cust_id,
            opened_at=opened_at,
            tx=sub_tx,
            cc=cc[pd.to_datetime(cc.closed_at) <= opened_at],
        )
        findings_raw = detector.run(ctx)
        active_before = [f for f in findings_raw if f.lr > 1.0]
        if active_before:
            wrongly_acted_before.append(case_key)

        findings_after = store.apply(findings_raw, card_id=row.card_id)
        active_after = [f for f in findings_after if f.lr > 1.0]

        for f_b, f_a in zip(findings_raw, findings_after):
            cids = getattr(f_a, "causal_decision_ids", [])
            if cids and f_b.lr > 1.0 and f_a.lr == 1.0:
                params = f_a.claim.params if f_a.claim else {}
                changed_findings_log.append(
                    {
                        "case_id": case_key,
                        "outcome": "legit_card_day",
                        "element_value": params.get("element_value", ""),
                        "before_lr": f_b.lr,
                        "after_lr": f_a.lr,
                        "causal_decision_ids": cids,
                    }
                )

        if active_after:
            wrongly_acted_after.append(case_key)

    n_wb = len(wrongly_acted_before)
    n_wa = len(wrongly_acted_after)
    n_fb = len(fraud_caught_before)
    n_fa = len(fraud_caught_after)

    prec_before = n_fb / (n_fb + n_wb) if (n_fb + n_wb) else 0.0
    prec_after = n_fa / (n_fa + n_wa) if (n_fa + n_wa) else 0.0

    pct_change_wa = ((n_wa - n_wb) / n_wb * 100) if n_wb else 0.0
    pct_change_fc = ((n_fa - n_fb) / n_fb * 100) if n_fb else 0.0
    pct_change_prec = ((prec_after - prec_before) / prec_before * 100) if prec_before else 0.0

    if verbose:
        print("=" * 85)
        print("ANALYST IN THE LOOP (CHALLENGE 8) REALISTIC 1-ROUND FEEDBACK EVALUATION (SEP-OCT)")
        print("=" * 85)
        print(f"{'Metric':<35} | {'Before Feedback':<15} | {'After Feedback':<15} | {'% Change':<10}")
        print("-" * 85)
        print(f"{'Wrongly Acted Against (Legit)':<35} | {n_wb:<15} | {n_wa:<15} | {pct_change_wa:+.1f}%")
        print(f"{'Fraud Caught (Confirmed Fraud)':<35} | {n_fb:<15} | {n_fa:<15} | {pct_change_fc:+.1f}%")
        print(f"{'Precision (Fraud / Total Acted)':<35} | {prec_before:<15.4f} | {prec_after:<15.4f} | {pct_change_prec:+.1f}%")
        print("-" * 85)
        print(f"Top Alerts Reviewed by Analyst (Jul-Aug): {len(top_100_alerts)}")
        print(f"Feedback Decisions Logged: {len(store.list_decisions())}")
        print("-" * 85)
        print("EXACT SEP-OCT FINDINGS CHANGED BY ANALYST DECISIONS (SAMPLE 3):")
        for i, log in enumerate(changed_findings_log[:3], 1):
            print(f"  {i}. Case ID: {log['case_id']}")
            print(f"     Outcome: {log['outcome']}")
            print(f"     Element Value: '{log['element_value']}'")
            print(f"     LR Change: {log['before_lr']} -> {log['after_lr']}")
            print(f"     Causal Decision IDs: {log['causal_decision_ids']}")
        print("=" * 85)

    sample_evidence = wrongly_acted_before[:5] + fraud_caught_before[:5]

    demo_attestation = Attestation(
        what=f"Analyst feedback on top 100 alerts reduced wrongly-acted-against customers from {n_wb} to {n_wa} ({pct_change_wa:.1f}%) while preserving fraud caught ({n_fb} to {n_fa}, {pct_change_fc:.1f}%), improving precision from {prec_before:.4f} to {prec_after:.4f}",
        evidence=Evidence(
            case_ids=sample_evidence,
            card_ids=[sep_oct_cases.iloc[0].card_id] if not sep_oct_cases.empty else [],
        ),
        query="assay.r2.feedback.run_feedback_demonstration",
        params={
            "wrongly_acted_against_before": n_wb,
            "wrongly_acted_against_after": n_wa,
            "fraud_caught_before": n_fb,
            "fraud_caught_after": n_fa,
            "precision_before": prec_before,
            "precision_after": prec_after,
            "changed_findings_count": len(changed_findings_log),
            "sample_changed_findings": changed_findings_log[:3],
        },
    )

    return {
        "wrongly_acted_against_before": n_wb,
        "wrongly_acted_against_after": n_wa,
        "fraud_caught_before": n_fb,
        "fraud_caught_after": n_fa,
        "precision_before": prec_before,
        "precision_after": prec_after,
        "changed_findings_log": changed_findings_log,
        "attestation": demo_attestation,
    }
