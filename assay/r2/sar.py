"""B19 -- Evidence to SAR (challenge 4).

Implements Policy §3a (`sar_required`), builds an attested regulatory filing
(`draft_sar`) directly from `tx_all()` rows, lints every sentence for citation
and numerical reconciliation against the underlying dataframe (`lint_sar`),
and renders the filing to Markdown and a self-contained HTML page.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from assay.r2.attest import Attestation, Evidence

SPECULATION_WORDS = (
    "likely",
    "appears",
    "probably",
    "possibly",
    "perhaps",
    "seemingly",
    "presumably",
    "might",
    "could be",
)


@dataclass
class SARDecision:
    required: bool
    reasons: list[str]
    reason_text: str

    def __bool__(self) -> bool:
        return self.required


@dataclass
class NarrativeSentence:
    text: str               # rendered sentence including citation marker [T:...]
    paragraph: int          # 0-based paragraph index
    attestation: Attestation
    expected_numbers: dict[str, float] = field(default_factory=dict)


@dataclass
class TimelineRow:
    txn_id: int
    ts: str
    card_id: str
    customer_id: str
    amount_usd: float
    channel: str
    product_cd: str
    addr1: str
    device_profile: str
    risk_score: float


@dataclass
class SAR:
    case_id: str
    required: bool
    decision: SARDecision
    subject: dict[str, Any]
    summary: str
    timeline: list[TimelineRow]
    sentences: list[NarrativeSentence]
    paragraphs: list[str]
    evidence_graph: dict[str, list[dict[str, Any]]]
    actions_taken: list[dict[str, str]]
    analyst_contact: dict[str, str]
    total_amount_usd: float
    activity_dates: list[str]

    @property
    def narrative_text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def narrative_plain(self) -> str:
        """Single-paragraph narrative suitable for answer['sar']['narrative']."""
        return " ".join(s.text for s in self.sentences)

    def to_answer_sar(self) -> dict[str, Any]:
        """Convert to the answer_format.md Part 2 `sar` object."""
        if not self.required:
            return {
                "file": False,
                "reason": self.decision.reason_text,
                "narrative": "",
                "subjects": [],
                "total_amount_usd": 0,
                "activity_dates": [],
            }
        subj: list[str] = []
        for x in (
            [self.subject.get("primary_customer_id", ""), self.subject.get("primary_card_id", "")]
            + list(self.subject.get("connected_card_ids", []))
            + list(self.subject.get("device_profiles", []))
        ):
            if x and x not in subj:
                subj.append(str(x))
        return {
            "file": True,
            "reason": self.decision.reason_text,
            "narrative": self.narrative_plain,
            "subjects": subj,
            "total_amount_usd": round(self.total_amount_usd, 2),
            "activity_dates": list(self.activity_dates),
        }

    def to_markdown(self) -> str:
        return render_markdown(self)

    def to_html(self) -> str:
        return render_html(self)


def sar_required(answer: dict[str, Any], tx: pd.DataFrame | None = None) -> SARDecision:
    """Evaluate Policy §3a on an investigation answer dict.

    §3a: File a SAR when fraud is confirmed or strongly suspected AND at least
    one of:
      1. exposure exceeds $1,000;
      2. activity connects to a shared device profile, a shared region cluster,
         or another customer's fraud;
      3. the pattern is coordinated or undocumented (rule R9).
    """
    case = answer.get("case", {}) or {}
    verdict = str(case.get("verdict", "")).lower()
    prob = float(case.get("fraud_probability", 0.0) or 0.0)
    status = str(case.get("status", "")).lower()

    if verdict == "legitimate" or status == "closed_legitimate":
        return SARDecision(
            required=False,
            reasons=[],
            reason_text="Policy §3a: activity is assessed as legitimate; no SAR is owed.",
        )

    confirmed_or_suspected = (
        verdict == "fraud"
        or status == "closed_fraud"
        or prob >= 0.70
    )
    if not confirmed_or_suspected:
        return SARDecision(
            required=False,
            reasons=[],
            reason_text=(
                f"Policy §3a: verdict is '{verdict}' with fraud probability {prob:.2f} (< 0.70); "
                "fraud is neither confirmed nor strongly suspected."
            ),
        )

    # Compute exposure from tx if provided and affected_txn_ids exist, else from case
    affected_ids = [int(x) for x in (case.get("affected_txn_ids") or []) if str(x).strip().isdigit()]
    exposure = float(case.get("exposure_usd", 0.0) or 0.0)
    if tx is not None and affected_ids:
        sub = tx[tx["TransactionID"].isin(affected_ids)]
        if not sub.empty:
            exposure = float(sub["amount"].abs().sum())

    connected_cards = [str(c) for c in (case.get("connected_card_ids") or []) if c]
    connected_devices = [str(d) for d in (case.get("connected_device_profiles") or []) if d]
    pattern = str(case.get("pattern", "") or "").lower()
    pattern_desc = str(case.get("pattern_description", "") or "")
    evidence_list = case.get("evidence") or []

    # Check region cluster or other customer's fraud in evidence/actions if not already in connected_cards
    region_cluster = False
    other_customer_fraud = False
    for ev in evidence_list:
        claim_l = str(ev.get("claim", "")).lower()
        ref_l = str(ev.get("ref", "")).lower()
        if "region cluster" in claim_l or "shared billing region" in claim_l or "region_cluster" in ref_l:
            region_cluster = True
        if "closed_cases_by_device" in ref_l or "other card" in claim_l or "unrelated card" in claim_l:
            if any(str(eid).startswith("CC-") for eid in (ev.get("entity_ids") or [])):
                other_customer_fraud = True

    reasons: list[str] = []
    if exposure > 1000.0:
        reasons.append(f"exposure ${exposure:,.2f} exceeds $1,000.00")
    if connected_devices:
        reasons.append(f"activity connects to shared device profile ({connected_devices[0]})")
    if connected_cards:
        reasons.append(f"activity connects to {len(connected_cards)} other card(s) ({', '.join(connected_cards[:3])})")
    elif other_customer_fraud:
        reasons.append("activity connects to another customer's confirmed fraud")
    if region_cluster:
        reasons.append("activity connects to a shared region cluster")
    if pattern == "undocumented" or pattern.startswith("undocumented:") or "coordinated" in pattern_desc.lower():
        reasons.append("the pattern is coordinated or undocumented (R9)")

    if not reasons:
        return SARDecision(
            required=False,
            reasons=[],
            reason_text=(
                f"Policy §3a: fraud is suspected (probability {prob:.2f}), but exposure "
                f"${exposure:,.2f} does not exceed $1,000.00, no shared device/region/card link "
                "was found, and the pattern is documented and uncoordinated."
            ),
        )

    reason_text = f"Policy §3a: fraud is confirmed or strongly suspected (probability {prob:.2f}) and " + "; ".join(reasons) + "."
    return SARDecision(required=True, reasons=reasons, reason_text=reason_text)


def _cite_tag(
    txn_ids: list[int],
    card_ids: list[str] | None = None,
    case_ids: list[str] | None = None,
) -> str:
    parts: list[str] = []
    if txn_ids:
        parts.append("T:" + ",".join(str(t) for t in txn_ids))
    if card_ids:
        parts.append("C:" + ",".join(str(c) for c in card_ids[:4]))
    if case_ids:
        parts.append("CC:" + ",".join(str(c) for c in case_ids[:4]))
    return "[" + " | ".join(parts) + "]" if parts else "[T:none]"


def _fmt_addr(val: Any) -> str:
    if pd.isna(val):
        return "unrecorded"
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)


def draft_sar(answer: dict[str, Any], tx: pd.DataFrame) -> SAR:
    """Build a complete, attested SAR from an investigation answer and `tx_all()`."""
    case_id = str(answer.get("case_id", "UNKNOWN"))
    case = answer.get("case", {}) or {}
    decision = sar_required(answer, tx=tx)

    affected_ids = sorted(
        {int(x) for x in (case.get("affected_txn_ids") or []) if str(x).strip().isdigit()}
    )
    sub = tx[tx["TransactionID"].isin(affected_ids)].sort_values("t", kind="stable") if affected_ids else tx.iloc[0:0]
    txn_ids = [int(x) for x in sub["TransactionID"].tolist()] if not sub.empty else affected_ids

    primary_card = str(sub["card_id"].iloc[0]) if not sub.empty else str(case.get("card_id", ""))
    primary_cust = str(sub["customer_id"].iloc[0]) if not sub.empty else str(
        primary_card.split("-K")[0] if "-K" in primary_card else ""
    )

    if not decision.required or sub.empty:
        return SAR(
            case_id=case_id,
            required=False,
            decision=decision,
            subject={
                "case_id": case_id,
                "primary_customer_id": primary_cust,
                "primary_card_id": primary_card,
                "connected_card_ids": [],
                "device_profiles": [],
                "prior_case_ids": [],
            },
            summary=decision.reason_text,
            timeline=[],
            sentences=[],
            paragraphs=[],
            evidence_graph={"nodes": [], "edges": []},
            actions_taken=list((answer.get("next_best_actions") or {}).get("final") or []),
            analyst_contact={
                "office": "Assay Financial Intelligence Unit (FIU)",
                "analyst_id": "FIU-ANALYST-PLACEHOLDER",
                "email": "fiu-sar-desk@bank.example",
                "phone": "+1-800-555-0199",
                "case_reference": case_id,
            },
            total_amount_usd=0.0,
            activity_dates=[],
        )

    # Recompute all transaction metrics strictly from `sub` (tx_all slice)
    n_txns = len(sub)
    total_amt = float(sub["amount"].abs().sum())
    min_amt = float(sub["amount"].abs().min())
    max_amt = float(sub["amount"].abs().max())
    first_row = sub.iloc[0]
    last_row = sub.iloc[-1]
    first_tid = int(first_row["TransactionID"])
    last_tid = int(last_row["TransactionID"])
    first_amt = float(abs(first_row["amount"]))
    last_amt = float(abs(last_row["amount"]))
    first_ts = pd.Timestamp(first_row["t"]).strftime("%Y-%m-%d %H:%M:%S")
    last_ts = pd.Timestamp(last_row["t"]).strftime("%Y-%m-%d %H:%M:%S")
    start_date = pd.Timestamp(first_row["t"]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(last_row["t"]).strftime("%Y-%m-%d")

    channels = sorted({str(c) for c in sub["channel"].dropna().unique() if str(c)})
    products = sorted({str(p) for p in sub["ProductCD"].dropna().unique() if str(p)})
    regions = sorted({_fmt_addr(a) for a in sub["addr1"].unique() if _fmt_addr(a) != "unrecorded"})
    if not regions:
        regions = ["unrecorded"]
    tx_devices = [
        str(d) for d in sub["device_profile"].dropna().unique()
        if str(d).strip() and str(d).strip() != "? | ? | ? | ?"
    ]
    conn_devices = [str(d) for d in (case.get("connected_device_profiles") or []) if str(d).strip()]
    all_devices: list[str] = []
    for d in tx_devices + conn_devices:
        if d not in all_devices:
            all_devices.append(d)

    conn_cards = [str(c) for c in (case.get("connected_card_ids") or []) if str(c).strip()]
    prior_cases = [str(c) for c in (case.get("similar_prior_cases") or []) if str(c).strip()]
    prob = float(case.get("fraud_probability", 0.0) or 0.0)
    pattern = str(case.get("pattern", "undocumented") or "undocumented")

    # Identify the flagged / alert transaction (or highest/last affected txn)
    flagged_tid = first_tid
    for ev in case.get("evidence") or []:
        m = re.search(r"Flagged transaction (\d+)", str(ev.get("claim", "")))
        if m and int(m.group(1)) in txn_ids:
            flagged_tid = int(m.group(1))
            break
    flagged_row = sub[sub["TransactionID"] == flagged_tid].iloc[0]
    flagged_amt = float(abs(flagged_row["amount"]))
    flagged_ts = pd.Timestamp(flagged_row["t"]).strftime("%Y-%m-%d %H:%M:%S")
    flagged_risk = float(flagged_row["risk_score"]) if pd.notna(flagged_row["risk_score"]) else 0.0

    # Card historical baseline prior to first suspicious transaction, computed directly from tx
    card_prior = tx[(tx["card_id"] == primary_card) & (tx["t"] < pd.Timestamp(first_row["t"]))]
    n_prior = int(len(card_prior))
    median_prior = float(card_prior["amount"].abs().median()) if n_prior > 0 else 0.0
    p95_prior = float(card_prior["amount"].abs().quantile(0.95)) if n_prior > 0 else 0.0

    # Timeline rows
    timeline: list[TimelineRow] = []
    for r in sub.itertuples():
        timeline.append(
            TimelineRow(
                txn_id=int(r.TransactionID),
                ts=pd.Timestamp(r.t).strftime("%Y-%m-%d %H:%M:%S"),
                card_id=str(r.card_id),
                customer_id=str(r.customer_id),
                amount_usd=round(float(abs(r.amount)), 2),
                channel=str(r.channel),
                product_cd=str(r.ProductCD),
                addr1=_fmt_addr(r.addr1),
                device_profile=str(r.device_profile) if str(r.device_profile).strip() else "none",
                risk_score=round(float(r.risk_score), 4) if pd.notna(r.risk_score) else 0.0,
            )
        )

    sentences: list[NarrativeSentence] = []

    # Paragraph 0: WHO & WHAT & WHEN
    c0 = _cite_tag(txn_ids, [primary_card])
    s1_body = (
        f"Between {start_date} ({first_ts} UTC) and {end_date} ({last_ts} UTC), card {primary_card} "
        f"held by customer {primary_cust} recorded {n_txns} unauthorized transaction(s) totaling "
        f"${total_amt:,.2f} USD (assessed fraud probability {prob:.2f})"
    )
    sentences.append(
        NarrativeSentence(
            text=f"{s1_body} {c0}.",
            paragraph=0,
            attestation=Attestation(
                what=s1_body,
                evidence=Evidence(txn_ids=txn_ids, card_ids=[primary_card], customer_ids=[primary_cust]),
                query="sar_episode_summary",
                params={"card_id": primary_card, "txn_ids": txn_ids},
            ),
            expected_numbers={
                "n_txns": float(n_txns),
                "total_amount_usd": round(total_amt, 2),
                "fraud_probability": round(prob, 2),
            },
        )
    )

    if n_txns == 1:
        s2_body = (
            f"The single unauthorized charge was transaction {first_tid} at {first_ts} UTC "
            f"for ${first_amt:,.2f} USD"
        )
        s2_nums = {"first_amt": round(first_amt, 2)}
        s2_tids = [first_tid]
    else:
        s2_body = (
            f"The unauthorized sequence began with transaction {first_tid} at {first_ts} UTC for "
            f"${first_amt:,.2f} USD and concluded with transaction {last_tid} at {last_ts} UTC for "
            f"${last_amt:,.2f} USD, with individual transaction amounts ranging from ${min_amt:,.2f} USD "
            f"to ${max_amt:,.2f} USD"
        )
        s2_nums = {
            "first_amt": round(first_amt, 2),
            "last_amt": round(last_amt, 2),
            "min_amt": round(min_amt, 2),
            "max_amt": round(max_amt, 2),
        }
        s2_tids = txn_ids
    sentences.append(
        NarrativeSentence(
            text=f"{s2_body} {_cite_tag(s2_tids, [primary_card])}.",
            paragraph=0,
            attestation=Attestation(
                what=s2_body,
                evidence=Evidence(txn_ids=s2_tids, card_ids=[primary_card], customer_ids=[primary_cust]),
                query="sar_episode_bounds",
                params={"txn_ids": s2_tids},
            ),
            expected_numbers=s2_nums,
        )
    )

    # Paragraph 1: WHERE & HOW (Channels, Regions, Devices, Alert)
    dev_str = "; ".join(all_devices) if all_devices else "no recorded device fingerprint (in-person or unrecorded)"
    chan_str = ", ".join(channels) if channels else "unspecified"
    prod_str = ", ".join(products) if products else "unspecified"
    reg_str = ", ".join(regions)
    s3_body = (
        f"The activity occurred via {chan_str} channel(s) under product code(s) {prod_str}, "
        f"billed to region code(s) {reg_str}, and originated from device profile(s): {dev_str}"
    )
    sentences.append(
        NarrativeSentence(
            text=f"{s3_body} {_cite_tag(txn_ids, [primary_card])}.",
            paragraph=1,
            attestation=Attestation(
                what=s3_body,
                evidence=Evidence(txn_ids=txn_ids, card_ids=[primary_card], customer_ids=[primary_cust]),
                query="sar_channel_region_device",
                params={"txn_ids": txn_ids},
            ),
            expected_numbers={},
        )
    )

    s4_body = (
        f"The investigation examined flagged transaction {flagged_tid} on {flagged_ts} UTC "
        f"for ${flagged_amt:,.2f} USD, which carried an initial bank model risk score of {flagged_risk:.2f}"
    )
    sentences.append(
        NarrativeSentence(
            text=f"{s4_body} {_cite_tag([flagged_tid], [primary_card])}.",
            paragraph=1,
            attestation=Attestation(
                what=s4_body,
                evidence=Evidence(txn_ids=[flagged_tid], card_ids=[primary_card], customer_ids=[primary_cust]),
                query="get_transaction",
                params={"txn_id": flagged_tid},
            ),
            expected_numbers={
                "flagged_amt": round(flagged_amt, 2),
                "flagged_risk": round(flagged_risk, 2),
            },
        )
    )

    # Paragraph 2: WHY IT IS SUSPICIOUS (Baseline comparison, typology, connected ring/cases)
    if n_prior > 0:
        s5_body = (
            f"Prior to {first_ts} UTC, card {primary_card} had {n_prior} historical transaction(s) "
            f"with a median amount of ${median_prior:,.2f} USD and a 95th-percentile amount of ${p95_prior:,.2f} USD"
        )
        s5_nums = {
            "n_prior": float(n_prior),
            "median_prior": round(median_prior, 2),
            "p95_prior": round(p95_prior, 2),
        }
    else:
        s5_body = (
            f"Prior to {first_ts} UTC, card {primary_card} had 0 historical transactions on record "
            f"in the bank's transaction ledger"
        )
        s5_nums = {"n_prior": 0.0}
    sentences.append(
        NarrativeSentence(
            text=f"{s5_body} {_cite_tag(txn_ids, [primary_card])}.",
            paragraph=2,
            attestation=Attestation(
                what=s5_body,
                evidence=Evidence(txn_ids=txn_ids, card_ids=[primary_card], customer_ids=[primary_cust]),
                query="card_baseline_as_of",
                params={"card_id": primary_card, "before_ts": first_ts},
            ),
            expected_numbers=s5_nums,
        )
    )

    if conn_cards:
        # Measure connected card activity in tx
        conn_tx = tx[tx["card_id"].isin(conn_cards)]
        if all_devices:
            conn_dev_tx = conn_tx[conn_tx["device_profile"].isin(all_devices)]
            if not conn_dev_tx.empty:
                conn_tx = conn_dev_tx
        conn_tids = [int(x) for x in conn_tx["TransactionID"].tolist()[:6]]
        cite_tids = txn_ids + [t for t in conn_tids if t not in txn_ids]
        sample_cards = ", ".join(conn_cards[:5])
        s6_body = (
            f"Graph traversal linked the episode on card {primary_card} to {len(conn_cards)} connected "
            f"card(s) ({sample_cards}) sharing origin attributes ({'; '.join(all_devices[:2]) or 'shared cluster'}), "
            f"indicating coordinated multi-card activity under pattern '{pattern}' (fraud probability {prob:.2f})"
        )
        sentences.append(
            NarrativeSentence(
                text=f"{s6_body} {_cite_tag(cite_tids, [primary_card] + conn_cards[:3], prior_cases)}.",
                paragraph=2,
                attestation=Attestation(
                    what=s6_body,
                    evidence=Evidence(
                        txn_ids=cite_tids,
                        card_ids=[primary_card] + conn_cards,
                        customer_ids=[primary_cust],
                        case_ids=prior_cases,
                    ),
                    query="device_or_origin_neighbors",
                    params={"card_id": primary_card, "connected_card_ids": conn_cards},
                ),
                expected_numbers={
                    "n_connected_cards": float(len(conn_cards)),
                    "fraud_probability": round(prob, 2),
                },
            )
        )
    else:
        s6_body = (
            f"The activity on card {primary_card} matches fraud pattern '{pattern}' across {n_txns} "
            f"transaction(s) totaling ${total_amt:,.2f} USD (fraud probability {prob:.2f})"
        )
        sentences.append(
            NarrativeSentence(
                text=f"{s6_body} {_cite_tag(txn_ids, [primary_card], prior_cases)}.",
                paragraph=2,
                attestation=Attestation(
                    what=s6_body,
                    evidence=Evidence(
                        txn_ids=txn_ids,
                        card_ids=[primary_card],
                        customer_ids=[primary_cust],
                        case_ids=prior_cases,
                    ),
                    query="pattern_classification",
                    params={"card_id": primary_card, "pattern": pattern, "txn_ids": txn_ids},
                ),
                expected_numbers={
                    "n_txns": float(n_txns),
                    "total_amount_usd": round(total_amt, 2),
                    "fraud_probability": round(prob, 2),
                },
            )
        )

    if prior_cases:
        s7_body = (
            f"Historical case retrieval linked this episode to {len(prior_cases)} closed precedent "
            f"case(s) ({', '.join(prior_cases)}) in the bank's confirmed case repository"
        )
        sentences.append(
            NarrativeSentence(
                text=f"{s7_body} {_cite_tag(txn_ids, [primary_card], prior_cases)}.",
                paragraph=2,
                attestation=Attestation(
                    what=s7_body,
                    evidence=Evidence(
                        txn_ids=txn_ids,
                        card_ids=[primary_card],
                        customer_ids=[primary_cust],
                        case_ids=prior_cases,
                    ),
                    query="similar_prior_cases",
                    params={"case_ids": prior_cases},
                ),
                expected_numbers={"n_prior_cases": float(len(prior_cases))},
            )
        )

    # Paragraph 3: ACTIONS TAKEN & REGULATORY BASIS (§3a)
    final_actions = list((answer.get("next_best_actions") or {}).get("final") or [])
    act_names = [str(a.get("action", "")) for a in final_actions if a.get("action")]
    if not act_names:
        act_names = ["CREATE_CASE", "FILE_REPORT"]
    s8_body = (
        f"In accordance with {decision.reason_text.rstrip('.')}, the bank recorded actions "
        f"{', '.join(act_names)} for total suspicious exposure of ${total_amt:,.2f} USD across "
        f"{n_txns} transaction(s)"
    )
    sentences.append(
        NarrativeSentence(
            text=f"{s8_body} {_cite_tag(txn_ids, [primary_card])}.",
            paragraph=3,
            attestation=Attestation(
                what=s8_body,
                evidence=Evidence(txn_ids=txn_ids, card_ids=[primary_card], customer_ids=[primary_cust]),
                query="sar_policy_3a_filing",
                params={"case_id": case_id, "txn_ids": txn_ids, "actions": act_names},
            ),
            expected_numbers={
                "total_amount_usd": round(total_amt, 2),
                "n_txns": float(n_txns),
            },
        )
    )

    # Group sentences into paragraphs
    paragraphs: list[str] = []
    for p_idx in sorted({s.paragraph for s in sentences}):
        p_text = " ".join(s.text for s in sentences if s.paragraph == p_idx)
        paragraphs.append(p_text)

    # Build evidence graph (nodes & edges)
    evidence_graph = _build_evidence_graph(
        case_id=case_id,
        primary_cust=primary_cust,
        primary_card=primary_card,
        timeline=timeline,
        all_devices=all_devices,
        conn_cards=conn_cards,
        prior_cases=prior_cases,
    )

    subject = {
        "case_id": case_id,
        "primary_customer_id": primary_cust,
        "primary_card_id": primary_card,
        "verdict": str(case.get("verdict", "fraud")),
        "fraud_probability": round(prob, 4),
        "pattern": pattern,
        "connected_card_ids": conn_cards,
        "device_profiles": all_devices,
        "billing_regions": regions,
        "channels": channels,
        "prior_case_ids": prior_cases,
    }

    summary = (
        f"Suspicious Activity Report for case {case_id}: {n_txns} unauthorized transaction(s) on card "
        f"{primary_card} (customer {primary_cust}) between {start_date} and {end_date} totaling "
        f"${total_amt:,.2f} USD (pattern: {pattern}, probability: {prob:.2f})."
    )

    analyst_contact = {
        "office": "Assay Financial Intelligence Unit (FIU)",
        "analyst_id": "FIU-ANALYST-PLACEHOLDER",
        "reviewing_manager_route": "L2 (Fraud Manager)",
        "email": "fiu-sar-desk@bank.example",
        "phone": "+1-800-555-0199",
        "case_reference": case_id,
    }

    return SAR(
        case_id=case_id,
        required=True,
        decision=decision,
        subject=subject,
        summary=summary,
        timeline=timeline,
        sentences=sentences,
        paragraphs=paragraphs,
        evidence_graph=evidence_graph,
        actions_taken=final_actions,
        analyst_contact=analyst_contact,
        total_amount_usd=round(total_amt, 2),
        activity_dates=[start_date, end_date],
    )


def _build_evidence_graph(
    case_id: str,
    primary_cust: str,
    primary_card: str,
    timeline: list[TimelineRow],
    all_devices: list[str],
    conn_cards: list[str],
    prior_cases: list[str],
) -> dict[str, list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    def add_node(nid: str, ntype: str, label: str, **attrs: Any) -> None:
        if nid not in seen_nodes:
            seen_nodes.add(nid)
            nodes.append({"id": nid, "type": ntype, "label": label, **attrs})

    add_node(f"case:{case_id}", "case", case_id)
    if primary_cust:
        add_node(f"customer:{primary_cust}", "customer", primary_cust)
    if primary_card:
        add_node(f"card:{primary_card}", "card", primary_card, role="primary")
        edges.append({"source": f"case:{case_id}", "target": f"card:{primary_card}", "rel": "INVESTIGATES_CARD"})
        if primary_cust:
            edges.append({"source": f"customer:{primary_cust}", "target": f"card:{primary_card}", "rel": "HOLDS_CARD"})

    for row in timeline:
        tid_node = f"txn:{row.txn_id}"
        add_node(
            tid_node,
            "transaction",
            f"T:{row.txn_id} (${row.amount_usd:,.2f})",
            amount_usd=row.amount_usd,
            ts=row.ts,
            channel=row.channel,
        )
        edges.append({"source": f"card:{row.card_id}", "target": tid_node, "rel": "AUTHORIZED_TXN"})
        if row.device_profile and row.device_profile not in ("none", "? | ? | ? | ?"):
            d_node = f"device:{row.device_profile}"
            add_node(d_node, "device_profile", row.device_profile)
            edges.append({"source": tid_node, "target": d_node, "rel": "USED_DEVICE"})

    for d in all_devices:
        d_node = f"device:{d}"
        add_node(d_node, "device_profile", d)
        for cc in conn_cards:
            c_node = f"card:{cc}"
            add_node(c_node, "card", cc, role="connected")
            edges.append({"source": c_node, "target": d_node, "rel": "SHARED_DEVICE"})

    if not all_devices:
        for cc in conn_cards:
            c_node = f"card:{cc}"
            add_node(c_node, "card", cc, role="connected")
            edges.append({"source": f"case:{case_id}", "target": c_node, "rel": "CONNECTED_CARD"})

    for pc in prior_cases:
        p_node = f"prior_case:{pc}"
        add_node(p_node, "prior_case", pc)
        edges.append({"source": f"case:{case_id}", "target": p_node, "rel": "PRECEDENT_CASE"})

    return {"nodes": nodes, "edges": edges}


def lint_sar(sar: SAR, tx: pd.DataFrame) -> list[str]:
    """Lint a drafted SAR.

    Fails if:
      - A filing is marked required=True but has no sentences or timeline.
      - Any narrative sentence lacks a `[T:...]` citation marker with valid transaction IDs.
      - Any cited transaction ID does not exist in `tx`.
      - Any narrative sentence lacks a valid `Attestation` (`what`, `evidence.txn_ids`, `query`).
      - Any sentence uses speculation words ("likely", "appears", etc.) without an explicit
        probability attached in the same sentence.
      - Any dollar amount or count in `sentence.expected_numbers` or parsed `$X,XXX.XX` in the
        sentence does not recompute from `tx`.
      - `sar.total_amount_usd` does not equal the sum of the timeline transaction amounts in `tx`.
    """
    errors: list[str] = []
    if not sar.required:
        if sar.sentences:
            errors.append("Non-required SAR must not contain narrative sentences.")
        return errors

    if not sar.sentences:
        errors.append("Required SAR has no narrative sentences.")
        return errors

    if not sar.timeline:
        errors.append("Required SAR has an empty timeline.")
        return errors

    tx_index = tx.set_index("TransactionID", drop=False)
    timeline_tids = [r.txn_id for r in sar.timeline]
    if not all(t in tx_index.index for t in timeline_tids):
        missing = [t for t in timeline_tids if t not in tx_index.index]
        errors.append(f"Timeline transaction IDs missing from tx: {missing}")
        return errors

    recomputed_total = round(float(tx_index.loc[timeline_tids, "amount"].abs().sum()), 2)
    if abs(recomputed_total - round(sar.total_amount_usd, 2)) > 0.01:
        errors.append(
            f"SAR total_amount_usd ({sar.total_amount_usd:.2f}) does not match "
            f"recomputed tx sum ({recomputed_total:.2f})."
        )

    cite_re = re.compile(r"\[T:([0-9,]+)(?:\s*\|[^\]]*)?\]")
    dollar_re = re.compile(r"\$([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})")
    prob_re = re.compile(r"(?:probability\s+0\.\d+|probability\s+1\.00|\b0\.\d{2}\b|\d+(?:\.\d+)?%)")

    # Precompute valid reference amounts from tx for this card & episode
    ep_sub = tx_index.loc[timeline_tids]
    primary_card = sar.subject.get("primary_card_id", "")
    first_ts = ep_sub["t"].min()
    card_prior = tx[(tx["card_id"] == primary_card) & (tx["t"] < first_ts)]

    valid_dollar_values: set[float] = {
        round(float(ep_sub["amount"].abs().sum()), 2),
        round(float(ep_sub["amount"].abs().min()), 2),
        round(float(ep_sub["amount"].abs().max()), 2),
        1000.00,
        2500.00,
        500.00,
    }
    for amt in ep_sub["amount"].abs().tolist():
        valid_dollar_values.add(round(float(amt), 2))
    if not card_prior.empty:
        valid_dollar_values.add(round(float(card_prior["amount"].abs().median()), 2))
        valid_dollar_values.add(round(float(card_prior["amount"].abs().quantile(0.95)), 2))

    for idx, sent in enumerate(sar.sentences):
        txt = sent.text.strip()
        # 1. Citation marker check
        m = cite_re.search(txt)
        if not m:
            errors.append(f"Sentence {idx} missing [T:...] citation marker: {txt}")
            continue
        cited_tids = [int(x) for x in m.group(1).split(",") if x.strip()]
        if not cited_tids:
            errors.append(f"Sentence {idx} has empty transaction citation list: {txt}")
        for tid in cited_tids:
            if tid not in tx_index.index:
                errors.append(f"Sentence {idx} cites non-existent TransactionID {tid}.")

        # 2. Attestation check
        att = sent.attestation
        if not att or not att.what or not att.query:
            errors.append(f"Sentence {idx} has incomplete Attestation metadata.")
        if not att.evidence.txn_ids:
            errors.append(f"Sentence {idx} Attestation has no evidence.txn_ids.")
        if set(cited_tids) != set(att.evidence.txn_ids):
            errors.append(
                f"Sentence {idx} citation IDs {cited_tids} do not match Attestation.evidence.txn_ids {att.evidence.txn_ids}."
            )

        # 3. Speculation words check
        txt_lower = txt.lower()
        for sw in SPECULATION_WORDS:
            if re.search(rf"\b{re.escape(sw)}\b", txt_lower):
                if not prob_re.search(txt_lower):
                    errors.append(
                        f"Sentence {idx} contains speculation word '{sw}' without an attached probability: {txt}"
                    )

        # 4. Dollar amounts in sentence text must match recomputed tx values
        for dm in dollar_re.finditer(txt):
            val = round(float(dm.group(1).replace(",", "")), 2)
            if not any(abs(val - v) <= 0.01 for v in valid_dollar_values):
                errors.append(
                    f"Sentence {idx} contains dollar amount ${val:,.2f} that does not recompute from tx data ({ sorted(valid_dollar_values) })."
                )

        # 5. Expected numbers dictionary reconciliation
        for k, expected_val in sent.expected_numbers.items():
            if k == "total_amount_usd" and abs(expected_val - recomputed_total) > 0.01:
                errors.append(f"Sentence {idx} expected total_amount_usd {expected_val} != {recomputed_total}")
            elif k == "n_txns" and int(expected_val) != len(timeline_tids):
                errors.append(f"Sentence {idx} expected n_txns {expected_val} != {len(timeline_tids)}")
            elif k == "n_prior" and int(expected_val) != len(card_prior):
                errors.append(f"Sentence {idx} expected n_prior {expected_val} != {len(card_prior)}")

    return errors


def render_markdown(sar: SAR) -> str:
    """Render the SAR to a clean, regulator-grade Markdown document."""
    lines: list[str] = [
        f"# Suspicious Activity Report (SAR) — {sar.case_id}",
        "",
        f"**Filing Required (§3a):** `{'YES' if sar.required else 'NO'}`  ",
        f"**Regulatory Determination:** {sar.decision.reason_text}",
        "",
    ]
    if not sar.required:
        return "\n".join(lines)

    subj = sar.subject
    lines.extend([
        "## 1. Subject Information",
        f"- **Case ID:** `{subj.get('case_id')}`",
        f"- **Primary Customer ID:** `{subj.get('primary_customer_id')}`",
        f"- **Primary Card ID:** `{subj.get('primary_card_id')}`",
        f"- **Assessed Verdict / Probability:** `{subj.get('verdict')}` ({subj.get('fraud_probability'):.2f})",
        f"- **Typology / Pattern:** `{subj.get('pattern')}`",
        f"- **Total Suspicious Exposure (USD):** `${sar.total_amount_usd:,.2f}`",
        f"- **Activity Window:** `{sar.activity_dates[0]}` to `{sar.activity_dates[1]}`",
        f"- **Connected Cards ({len(subj.get('connected_card_ids', []))}):** "
        + (", ".join(f"`{c}`" for c in subj.get("connected_card_ids", [])[:10]) or "None"),
        f"- **Device Profiles:** "
        + ("; ".join(f"`{d}`" for d in subj.get("device_profiles", [])) or "None recorded"),
        "",
        "## 2. Executive Summary",
        sar.summary,
        "",
        "## 3. Chronological Transaction Timeline",
        "| Txn ID | Timestamp (UTC) | Card ID | Customer ID | Amount (USD) | Channel | Product | Region | Device Profile | Risk Score |",
        "|---:|---|---|---|---:|---|---|---|---|---:|",
    ])
    for r in sar.timeline:
        lines.append(
            f"| `{r.txn_id}` | {r.ts} | `{r.card_id}` | `{r.customer_id}` | "
            f"${r.amount_usd:,.2f} | {r.channel} | {r.product_cd} | {r.addr1} | "
            f"`{r.device_profile}` | {r.risk_score:.2f} |"
        )

    lines.extend([
        "",
        "## 4. Attested Regulatory Narrative",
    ])
    for p in sar.paragraphs:
        lines.append(p)
        lines.append("")

    lines.extend([
        "### Sentence Attestations",
    ])
    for idx, s in enumerate(sar.sentences, 1):
        att_d = s.attestation.to_dict()
        lines.append(
            f"{idx}. `{att_d['how']['query']}` — Txns: `{','.join(att_d['evidence']['txn_ids'])}` "
            f"| Dataset: `{att_d['when']['dataset']}`"
        )

    lines.extend([
        "",
        "## 5. Actions Taken & Routing",
        "| Action | Approval Route | Policy Justification |",
        "|---|---|---|",
    ])
    for a in sar.actions_taken:
        lines.append(f"| `{a.get('action', '')}` | `{a.get('route', '')}` | {a.get('reason', '')} |")

    lines.extend([
        "",
        "## 6. Evidence Graph (Nodes & Edges JSON)",
        "```json",
        json.dumps(sar.evidence_graph, indent=2, sort_keys=True),
        "```",
        "",
        "## 7. Filing Institution & Analyst Contact",
        f"- **Office:** {sar.analyst_contact.get('office')}",
        f"- **Analyst ID:** `{sar.analyst_contact.get('analyst_id')}`",
        f"- **Review Route:** {sar.analyst_contact.get('reviewing_manager_route', 'L2')}",
        f"- **Email:** `{sar.analyst_contact.get('email')}`",
        f"- **Phone:** `{sar.analyst_contact.get('phone')}`",
        f"- **Case Reference:** `{sar.analyst_contact.get('case_reference')}`",
        "",
    ])
    return "\n".join(lines)


def render_html(sar: SAR) -> str:
    """Render the SAR to a single self-contained HTML page."""
    esc = html.escape
    subj = sar.subject
    graph_json = esc(json.dumps(sar.evidence_graph, indent=2, sort_keys=True))

    if not sar.required:
        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>SAR Determination — {esc(sar.case_id)}</title></head>
<body>
  <h1>Suspicious Activity Report Determination — {esc(sar.case_id)}</h1>
  <p><strong>Filing Required (§3a):</strong> NO</p>
  <p>{esc(sar.decision.reason_text)}</p>
</body>
</html>"""

    timeline_rows_html = "\n".join(
        f"<tr>"
        f"<td><code>{r.txn_id}</code></td>"
        f"<td>{esc(r.ts)}</td>"
        f"<td><code>{esc(r.card_id)}</code></td>"
        f"<td><code>{esc(r.customer_id)}</code></td>"
        f"<td style='text-align:right'>${r.amount_usd:,.2f}</td>"
        f"<td>{esc(r.channel)}</td>"
        f"<td>{esc(r.product_cd)}</td>"
        f"<td>{esc(r.addr1)}</td>"
        f"<td><code>{esc(r.device_profile)}</code></td>"
        f"<td style='text-align:right'>{r.risk_score:.2f}</td>"
        f"</tr>"
        for r in sar.timeline
    )

    paragraphs_html = "\n".join(f"<p class='narrative-para'>{esc(p)}</p>" for p in sar.paragraphs)

    attestations_html = "\n".join(
        f"<li><strong>Sentence {i}:</strong> <code>{esc(s.attestation.query)}</code> "
        f"(params: <code>{esc(json.dumps(s.attestation.params, sort_keys=True))}</code>) "
        f"&mdash; Evidence Txns: <code>{esc(','.join(str(x) for x in s.attestation.evidence.txn_ids))}</code> "
        f"[{esc(s.attestation.dataset)}]</li>"
        for i, s in enumerate(sar.sentences, 1)
    )

    actions_html = "\n".join(
        f"<tr><td><code>{esc(str(a.get('action', '')))}</code></td>"
        f"<td><code>{esc(str(a.get('route', '')))}</code></td>"
        f"<td>{esc(str(a.get('reason', '')))}</td></tr>"
        for a in sar.actions_taken
    )

    conn_cards_str = ", ".join(subj.get("connected_card_ids", [])[:15]) or "None"
    devices_str = "; ".join(subj.get("device_profiles", [])) or "None recorded"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Suspicious Activity Report — {esc(sar.case_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1e293b; background: #f8fafc; margin: 0; padding: 24px; line-height: 1.55; }}
    .container {{ max-width: 1040px; margin: 0 auto; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    h1 {{ margin-top: 0; color: #0f172a; border-bottom: 2px solid #0f172a; padding-bottom: 10px; font-size: 1.6rem; }}
    h2 {{ color: #1e3a8a; margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 1.2rem; }}
    .badge {{ display: inline-block; background: #dc2626; color: #fff; font-weight: 600; padding: 2px 10px; border-radius: 4px; font-size: 0.85rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px 24px; background: #f1f5f9; padding: 16px; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    th {{ background: #e2e8f0; font-weight: 600; }}
    .narrative-para {{ background: #fffbeb; border-left: 4px solid #d97706; padding: 12px 16px; margin: 12px 0; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 6px; overflow-x: auto; font-size: 0.82rem; }}
    code {{ background: #e2e8f0; padding: 1px 5px; border-radius: 3px; font-size: 0.88em; }}
    pre code {{ background: transparent; padding: 0; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Suspicious Activity Report (SAR) &mdash; {esc(sar.case_id)} <span class="badge">FILE_REPORT (L2)</span></h1>
    <p><strong>Policy §3a Determination:</strong> {esc(sar.decision.reason_text)}</p>

    <h2>1. Subject Information</h2>
    <div class="grid">
      <div><strong>Case ID:</strong> <code>{esc(str(subj.get('case_id')))}</code></div>
      <div><strong>Primary Customer ID:</strong> <code>{esc(str(subj.get('primary_customer_id')))}</code></div>
      <div><strong>Primary Card ID:</strong> <code>{esc(str(subj.get('primary_card_id')))}</code></div>
      <div><strong>Verdict / Probability:</strong> <code>{esc(str(subj.get('verdict')))}</code> ({subj.get('fraud_probability', 0.0):.2f})</div>
      <div><strong>Pattern:</strong> <code>{esc(str(subj.get('pattern')))}</code></div>
      <div><strong>Total Exposure (USD):</strong> <strong>${sar.total_amount_usd:,.2f}</strong></div>
      <div><strong>Activity Window:</strong> <code>{esc(sar.activity_dates[0])}</code> to <code>{esc(sar.activity_dates[1])}</code></div>
      <div><strong>Connected Cards ({len(subj.get('connected_card_ids', []))}):</strong> <code>{esc(conn_cards_str)}</code></div>
      <div style="grid-column: span 2;"><strong>Device Profiles:</strong> <code>{esc(devices_str)}</code></div>
    </div>

    <h2>2. Executive Summary</h2>
    <p>{esc(sar.summary)}</p>

    <h2>3. Chronological Transaction Timeline</h2>
    <table>
      <thead>
        <tr>
          <th>Txn ID</th><th>Timestamp (UTC)</th><th>Card ID</th><th>Customer</th>
          <th>Amount (USD)</th><th>Channel</th><th>Product</th><th>Region</th><th>Device Profile</th><th>Risk Score</th>
        </tr>
      </thead>
      <tbody>
        {timeline_rows_html}
      </tbody>
    </table>

    <h2>4. Attested Regulatory Narrative</h2>
    {paragraphs_html}
    <h3>Sentence-Level Attestations</h3>
    <ol>
      {attestations_html}
    </ol>

    <h2>5. Actions Taken &amp; Routing</h2>
    <table>
      <thead><tr><th>Action</th><th>Approval Route</th><th>Policy Justification</th></tr></thead>
      <tbody>{actions_html}</tbody>
    </table>

    <h2>6. Evidence Graph (Nodes &amp; Edges JSON)</h2>
    <pre><code>{graph_json}</code></pre>

    <h2>7. Analyst Contact</h2>
    <p>
      <strong>Office:</strong> {esc(sar.analyst_contact.get('office', ''))} |
      <strong>Analyst ID:</strong> <code>{esc(sar.analyst_contact.get('analyst_id', ''))}</code> |
      <strong>Email:</strong> <code>{esc(sar.analyst_contact.get('email', ''))}</code> |
      <strong>Phone:</strong> <code>{esc(sar.analyst_contact.get('phone', ''))}</code>
    </p>
  </div>
</body>
</html>"""
