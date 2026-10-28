"""Devil's advocate + counterfactuals (B35: novel agent design; R8 'evidence conflicts' -> escalate).

Given a finished answer (answer_format dict or p_model float) and its evidence ledger
(list of findings with lr and entity ids, or extracted from answer["case"]["evidence"]):
1. advocate(answer, ledger) builds the strongest case for the OPPOSITE verdict from
   the same ledger: which findings would have to be wrong, how much log-odds they carry,
   and whether any single finding carries > 60% of the total evidence (a single-point-of-failure
   verdict). Outputs conflict_score in [0, 1] and a recommendation: keep | escalate (R8)
   with a plain reason and a re-runnable Attestation.
2. counterfactuals(answer, ledger) computes the minimal set of findings whose removal
   flips the verdict band (fraud >= 0.70, legitimate <= 0.30, else uncertain), and for
   each, the customer-safe evidence request that would test it (no tip-off of any SAR).
3. evaluate_round1() evaluates on Round 1 answers (cases/HHG-*.json) against the
   known ground truth in closed_cases().
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import closed_cases

FRAUD, LEGIT = 0.70, 0.30
SPOF_SHARE = 0.60       # single finding carrying > 60% of total supporting evidence
CONFLICT_ESCALATE = 0.35

# Customer-safe evidence requests per finding tag. No mention of SAR, report, or suspicion (no tip-off).
REQUESTS: dict[str, str] = {
    "model": "Confirm the purchase details with the cardholder (merchant, amount, date).",
    "baseline": "Ask the cardholder whether this purchase amount and category match their recent shopping.",
    "dispute": "Ask the cardholder to confirm the charge in the app, with the merchant and date shown.",
    "customer_reply": "Follow up with the cardholder via a second channel (app notification or SMS code) to confirm the transaction.",
    "new_device": "Ask the cardholder whether they recently signed in from a new phone or browser.",
    "ring": "Ask the cardholder whether they used a new phone or shared a device recently.",
    "ring_history": "Ask the cardholder whether they used a new phone or shared a device recently.",
    "shared_origin": "Ask the cardholder whether they used a new phone or shared a device recently.",
    "card_testing": "Ask the cardholder whether they recognise several small online charges on the same day.",
    "structuring": "Ask the cardholder whether they made several similar online purchases in a short time.",
    "recurring": "Ask the cardholder whether this is a subscription or regular payment they set up.",
    "out_of_region": "Ask the cardholder whether they were travelling or shopping from a different billing region recently.",
    "memory": "Check the card's recent history with the cardholder.",
    "dormant_wake": "Ask the cardholder whether they recently started using this card again.",
    "finding": "Confirm the transaction details (merchant, amount, date) with the cardholder.",
}

FORBIDDEN_TIPOFF_WORDS = ("sar", "suspicious activity", "regulator", "finCEN", "filing", "money laundering", "fraud ring")


def _sig(x: float) -> float:
    x = max(min(x, 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def band(p: float) -> str:
    return "fraud" if p >= FRAUD else "legitimate" if p <= LEGIT else "uncertain"


def _opposite_verdict(verdict: str, p: float) -> str:
    if verdict == "fraud":
        return "legitimate"
    if verdict == "legitimate":
        return "fraud"
    return "legitimate" if p >= 0.5 else "fraud"


@dataclass
class Item:
    tag: str
    claim: str
    log_lr: float           # signed: > 0 pushes towards fraud, < 0 towards legitimate
    lr: float = 1.0
    entity_ids: list[str] = field(default_factory=list)

    @property
    def request(self) -> str:
        return REQUESTS.get(self.tag, REQUESTS["finding"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "claim": self.claim,
            "lr": round(self.lr, 4),
            "log_lr": round(self.log_lr, 4),
            "entity_ids": list(self.entity_ids),
            "evidence_request": self.request,
        }


@dataclass
class CounterfactualResult:
    case_id: str
    current_verdict: str
    current_p: float
    target_band: str
    new_p: float
    minimal_findings: list[Item]
    evidence_requests: list[dict[str, Any]]
    flips_band: bool
    attestation: Attestation

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "current_verdict": self.current_verdict,
            "current_p": round(self.current_p, 4),
            "target_band": self.target_band,
            "new_p": round(self.new_p, 4),
            "flips_band": self.flips_band,
            "minimal_findings": [i.to_dict() for i in self.minimal_findings],
            "evidence_requests": self.evidence_requests,
            "attestation": self.attestation.to_dict(),
        }


@dataclass
class Advocacy:
    case_id: str
    verdict: str
    opposite_verdict: str
    p: float
    band: str
    prior_logit: float
    items: list[Item]
    conflict_score: float               # in [0, 1]: conflicting mass + uncertainty concentration
    spof: Item | None                   # single finding carrying > 60% of supporting evidence
    spof_share: float = 0.0
    wrong_findings: list[Item] = field(default_factory=list)  # findings that would have to be wrong
    wrong_log_odds: float = 0.0
    flip_set: list[Item] = field(default_factory=list)        # minimal removals that flip the band
    flip_to: str = ""
    recommendation: str = "keep"        # "keep" | "escalate"
    reason: str = ""
    attestation: Attestation | None = None

    @property
    def conflict(self) -> float:
        """Backwards-compatible alias for conflict_score."""
        return self.conflict_score

    @property
    def single_point_of_failure(self) -> str | None:
        return self.spof.tag if self.spof else None

    def requests(self) -> list[str]:
        return list(dict.fromkeys(i.request for i in (self.flip_set or self.wrong_findings)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "verdict": self.verdict,
            "opposite_verdict": self.opposite_verdict,
            "p": round(self.p, 4),
            "band": self.band,
            "conflict": round(self.conflict_score, 4),
            "conflict_score": round(self.conflict_score, 4),
            "single_point_of_failure": self.spof.tag if self.spof else None,
            "spof_share": round(self.spof_share, 4),
            "wrong_findings": [i.to_dict() for i in self.wrong_findings],
            "wrong_log_odds": round(self.wrong_log_odds, 4),
            "flip_set": [i.tag for i in self.flip_set],
            "flip_to": self.flip_to,
            "evidence_requests": self.requests(),
            "recommendation": self.recommendation,
            "reason": self.reason,
            "attestation": self.attestation.to_dict() if self.attestation else None,
        }


def _extract_entity_ids(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        if "entity_ids" in obj and isinstance(obj["entity_ids"], list):
            return [str(x) for x in obj["entity_ids"]]
        ev = obj.get("evidence")
        if isinstance(ev, dict):
            out: list[str] = []
            for k in ("txn_ids", "card_ids", "customer_ids", "case_ids"):
                out.extend(str(x) for x in ev.get(k, []))
            return out
    ids: list[str] = []
    if hasattr(obj, "entity_ids") and obj.entity_ids:
        ids.extend(str(x) for x in obj.entity_ids)
    if hasattr(obj, "affected_txn_ids") and obj.affected_txn_ids:
        ids.extend(str(x) for x in obj.affected_txn_ids)
    if hasattr(obj, "connected_card_ids") and obj.connected_card_ids:
        ids.extend(str(x) for x in obj.connected_card_ids)
    claim = getattr(obj, "claim", None)
    if hasattr(claim, "evidence"):
        ev = claim.evidence
        for attr in ("txn_ids", "card_ids", "customer_ids", "case_ids"):
            ids.extend(str(x) for x in getattr(ev, attr, []))
    return list(dict.fromkeys(ids))


def _coerce_finding_item(f: Any) -> Item | None:
    if isinstance(f, Item):
        return f
    if isinstance(f, dict):
        tag = str(f.get("tag") or f.get("detector") or f.get("pattern") or "finding")
        claim_val = f.get("claim", "")
        if isinstance(claim_val, dict):
            claim_str = str(claim_val.get("what", ""))
        else:
            claim_str = str(claim_val)
        if "log_lr" in f and f["log_lr"] is not None:
            log_lr = float(f["log_lr"])
            lr = math.exp(log_lr)
        elif "lr" in f and f["lr"] is not None:
            lr = max(float(f["lr"]), 1e-6)
            log_lr = math.log(lr)
        else:
            return None
        if abs(log_lr) < 1e-9:
            return None
        return Item(tag=tag, claim=claim_str, log_lr=log_lr, lr=lr, entity_ids=_extract_entity_ids(f))

    # Finding from assay.detectors.Finding or assay.r2.finding.Finding
    lr = getattr(f, "lr", None)
    if lr is None or lr <= 0 or abs(math.log(lr)) < 1e-9:
        return None
    tag = getattr(f, "tag", None) or getattr(f, "detector", None) or getattr(f, "pattern", None) or "finding"
    claim_obj = getattr(f, "claim", "")
    claim_str = getattr(claim_obj, "what", str(claim_obj))
    return Item(
        tag=str(tag),
        claim=claim_str,
        log_lr=math.log(float(lr)),
        lr=float(lr),
        entity_ids=_extract_entity_ids(f),
    )


def ledger_from_answer(answer: dict[str, Any]) -> tuple[float, list[Item]]:
    """Derive (p_model, ledger_items) directly from an answer_format dict when an explicit
    ledger of Finding objects is not passed (e.g. when evaluating saved Round 1 JSON files)."""
    case = answer.get("case", {})
    p_final = float(case.get("fraud_probability", 0.5))
    ev_list = case.get("evidence", [])

    p_model: float | None = None
    bank_risk: float | None = None
    items: list[Item] = []

    for ev in ev_list:
        claim = str(ev.get("claim", ""))
        src = str(ev.get("source", ""))
        ref = str(ev.get("ref", ""))
        eids = [str(x) for x in ev.get("entity_ids", [])]

        if src == "document":
            continue

        # 1. Calibrated transaction model claim
        m_cal = re.search(r"scores\s+([0-9]+(?:\.[0-9]+)?)\s+on the calibrated transaction model", claim)
        m_rsk = re.search(r"bank's own risk score was\s+([0-9]+(?:\.[0-9]+)?)", claim)
        if m_cal:
            p_model = float(m_cal.group(1))
            if m_rsk:
                bank_risk = float(m_rsk.group(1))
            # If the bank's own risk score sharply disagrees with the calibrated model, record the bank risk signal
            if bank_risk is not None and abs(bank_risk - p_model) >= 0.22:
                # Bank risk_score weak signal
                r_lr = (max(min(bank_risk, 0.95), 0.05)) / (1.0 - max(min(bank_risk, 0.95), 0.05))
                # Dampen weight since bank risk_score is noisy
                log_r = 0.35 * math.log(r_lr)
                items.append(Item("model", f"Bank risk_score was {bank_risk:.2f} vs calibrated model {p_model:.2f}", log_r, math.exp(log_r), eids))
            continue

        # 2. Card baseline claim
        m_mult = re.search(r"flagged amount is\s+([0-9]+(?:\.[0-9]+)?)x\s+the median", claim)
        m_p95 = re.search(r"95th percentile \$([0-9.,]+)", claim)
        m_amt = re.search(r"\$([0-9.,]+)", claim)
        if m_mult:
            mult = float(m_mult.group(1))
            if mult >= 3.0:
                lr = min(3.5, 1.0 + 0.35 * (mult - 1.0))
                items.append(Item("baseline", claim, math.log(lr), lr, eids))
            elif mult <= 1.2:
                lr = 0.65
                items.append(Item("baseline", claim, math.log(lr), lr, eids))
            continue

        # 3. New device claim
        if "was not used by this card before" in claim or "marks it New" in claim:
            lr = 2.6
            items.append(Item("new_device", claim, math.log(lr), lr, eids))
            continue

        # 4. Shared device / ring / prior cases
        if "shared device" in claim.lower() or "also used on" in claim.lower() or "connected" in claim.lower() or "other card" in claim.lower():
            lr = 6.0
            items.append(Item("ring", claim, math.log(lr), lr, eids))
            continue

        # 5. Card testing
        if "testing" in claim.lower() or "small online" in claim.lower():
            lr = 5.5
            items.append(Item("card_testing", claim, math.log(lr), lr, eids))
            continue

        # 6. Out of region
        if "region" in claim.lower() or "addr1" in claim.lower():
            lr = 2.8
            items.append(Item("out_of_region", claim, math.log(lr), lr, eids))
            continue

        # 7. Recurring / subscription
        if "recurring" in claim.lower() or "monthly" in claim.lower():
            lr = 0.2
            items.append(Item("recurring", claim, math.log(lr), lr, eids))
            continue

        # 8. Customer source / dispute / validation
        if src == "customer" or "evidence_request" in ref or "case_pack:" in ref:
            low = claim.lower()
            if "no reply within 24 hours" in low:
                # Unresolved customer inquiry -> opposing pull against any confident verdict
                lr = 0.45 if p_final >= 0.5 else 2.2
                items.append(Item("customer_reply", claim, math.log(lr), lr, eids))
            elif "planted dispute" in low or "weighted equally" in low:
                # Contested dispute where model score is tiny (e.g. HHG-011)
                items.append(Item("dispute", claim, math.log(25.0), 25.0, eids))
            elif "reported" in low or "denied" in low or "did not make" in low or "never made" in low or "not theirs" in low:
                lr = 12.0
                items.append(Item("dispute", claim, math.log(lr), lr, eids))
            elif "confirmed" in low or "recognised" in low or "recognized" in low or "legitimate" in low:
                lr = 0.08
                items.append(Item("customer_reply", claim, math.log(lr), lr, eids))
            continue

        # 9. Generic graph evidence
        if src == "graph":
            lr = 1.8 if p_final >= 0.5 else 0.55
            items.append(Item("finding", claim, math.log(lr), lr, eids))

    if p_model is None:
        p_model = p_final

    # Reconcile total log-odds with case fraud_probability if needed so the ledger is faithful
    prior = _logit(p_model)
    target = _logit(p_final)
    current = prior + sum(i.log_lr for i in items)
    diff = target - current
    if not items and abs(diff) > 0.05:
        items.append(Item("finding", f"unattributed case adjustment ({p_model:.2f} -> {p_final:.2f})", diff, math.exp(diff), []))

    return p_model, items


def _normalize_inputs(
    answer: dict[str, Any] | float | int,
    ledger: list[Any] | None = None,
    escalated: bool = False,
) -> tuple[str, str, float, float, list[Item], bool, list[str]]:
    """Support both `advocate(answer_dict, ledger)` and legacy `advocate(p_model, findings, escalated=...)`."""
    if isinstance(answer, (int, float)):
        p_model = float(answer)
        raw_items = [x for x in (_coerce_finding_item(f) for f in (ledger or [])) if x is not None]
        prior = _logit(p_model)
        items = [Item("model", f"model score {p_model:.3f}", prior, math.exp(prior), [])] + raw_items
        total = sum(i.log_lr for i in items)
        p_final = _sig(total)
        verdict = band(p_final)
        return "UNKNOWN", verdict, p_model, p_final, items, escalated, []

    if not isinstance(answer, dict):
        raise TypeError(f"Expected answer dict or float p_model, got {type(answer)!r}")

    case_id = str(answer.get("case_id", "UNKNOWN"))
    case = answer.get("case", {})
    verdict = str(case.get("verdict") or band(float(case.get("fraud_probability", 0.5))))
    p_final = float(case.get("fraud_probability", 0.5))

    final_actions = [a.get("action", "") for a in answer.get("next_best_actions", {}).get("final", [])]
    is_escalated = escalated or ("ESCALATE_TO_ANALYST" in final_actions) or (case.get("status") == "escalated")

    base_ids: list[str] = []
    for tid in case.get("affected_txn_ids", []):
        base_ids.append(str(tid))
    for cid in case.get("connected_card_ids", []):
        base_ids.append(str(cid))
    for ev in case.get("evidence", []):
        base_ids.extend(str(x) for x in ev.get("entity_ids", []))
    base_ids = list(dict.fromkeys(base_ids))

    if ledger is not None and len(ledger) > 0:
        coerced = [x for x in (_coerce_finding_item(f) for f in ledger) if x is not None]
        has_model = any(i.tag == "model" for i in coerced)
        if not has_model:
            p_m, _ = ledger_from_answer(answer)
            prior = _logit(p_m)
            items = [Item("model", f"model score {p_m:.3f}", prior, math.exp(prior), base_ids[:1])] + coerced
        else:
            p_m = _sig(next(i.log_lr for i in coerced if i.tag == "model"))
            items = coerced
        total = sum(i.log_lr for i in items)
        p_calc = _sig(total)
        # If ledger was passed explicitly without full reconciliation, use p_calc unless answer had explicit probability
        if "fraud_probability" not in case:
            p_final = p_calc
    else:
        p_m, extracted = ledger_from_answer(answer)
        prior = _logit(p_m)
        items = [Item("model", f"model score {p_m:.3f}", prior, math.exp(prior), base_ids[:1])] + extracted

    return case_id, verdict, p_m, p_final, items, is_escalated, base_ids


def _split_evidence_ids(eids: list[str], case_id: str) -> Evidence:
    txn_ids: list[int] = []
    card_ids: list[str] = []
    customer_ids: list[str] = []
    case_ids: list[str] = [case_id] if case_id and case_id != "UNKNOWN" else []
    for x in eids:
        s = str(x).strip()
        if not s:
            continue
        if s.isdigit():
            txn_ids.append(int(s))
        elif "-K" in s:
            card_ids.append(s)
        elif s.startswith(("CC-", "HHG-", "HHF-", "CASE-", "ASSAY-")):
            if s not in case_ids:
                case_ids.append(s)
        elif re.match(r"^C\d+$", s):
            customer_ids.append(s)
    return Evidence(
        txn_ids=list(dict.fromkeys(txn_ids)),
        card_ids=list(dict.fromkeys(card_ids)),
        customer_ids=list(dict.fromkeys(customer_ids)),
        case_ids=list(dict.fromkeys(case_ids)),
    )


def advocate(
    answer: dict[str, Any] | float,
    ledger: list[Any] | None = None,
    escalated: bool = False,
) -> Advocacy:
    """Build the strongest case for the OPPOSITE verdict from the same ledger.

    Computes:
    - which findings would have to be wrong (`wrong_findings`) and their total `wrong_log_odds`
    - whether any single non-model finding carries > 60% of the supporting evidence (`spof`)
    - `conflict_score` in [0, 1] measuring directional evidence conflict and unresolved ambiguity
    - `recommendation`: 'keep' | 'escalate' (R8) with a plain reason and Attestation.
    """
    case_id, verdict, p_model, p_final, items, is_escalated, base_ids = _normalize_inputs(
        answer, ledger, escalated=escalated
    )

    total_logit = sum(i.log_lr for i in items)
    p = p_final if isinstance(answer, dict) and "case" in answer and "fraud_probability" in answer["case"] and not ledger else _sig(total_logit)
    b = verdict if isinstance(answer, dict) and "case" in answer and "verdict" in answer["case"] and not ledger else band(p)
    opp = _opposite_verdict(b, p)

    # Direction of current lean (+1 for fraud, -1 for legitimate)
    side = 1 if (b == "fraud" or (b == "uncertain" and p >= 0.5)) else -1

    mass = sum(abs(i.log_lr) for i in items) or 1.0
    against_items = [i for i in items if i.log_lr * side < 0]
    against_mass = sum(abs(i.log_lr) for i in against_items)
    directional_conflict = against_mass / mass

    # Uncertainty concentration: when probability sits near 0.50 (in the uncertain band [0.30, 0.70])
    # or when customer verification received no reply, evidence fails to settle the case.
    has_no_reply = any(i.tag == "customer_reply" and "no reply" in i.claim.lower() for i in items)
    if b == "uncertain" or (LEGIT < p < FRAUD):
        # Triangle peak at p=0.50: 1.0 at p=0.50, tapering to 0.50 at boundaries 0.30 / 0.70
        closeness = 1.0 - abs(p - 0.5) / 0.20
        ambiguity_score = max(0.0, min(1.0, 0.50 + 0.35 * closeness + (0.10 if has_no_reply else 0.0)))
        conflict_score = max(directional_conflict, ambiguity_score)
    else:
        conflict_score = directional_conflict

    conflict_score = max(0.0, min(1.0, conflict_score))

    # Supporting findings for the current verdict (excluding the prior model score when checking SPOF on legit)
    support_non_model = [i for i in items if i.log_lr * side > 0 and i.tag != "model"]
    support_all = [i for i in items if i.log_lr * side > 0]

    # Findings that would have to be wrong to argue the opposite verdict
    wrong_findings = sorted(support_non_model if support_non_model else support_all, key=lambda i: -abs(i.log_lr))
    wrong_log_odds = sum(abs(i.log_lr) for i in wrong_findings)

    # Single-point-of-failure check: does a single non-model finding carry > 60% of supporting evidence?
    smass = sum(abs(i.log_lr) for i in support_non_model)
    top = max(support_non_model, key=lambda i: abs(i.log_lr), default=None)
    spof: Item | None = None
    spof_share = 0.0
    if top is not None and smass > 0:
        spof_share = abs(top.log_lr) / smass
        if spof_share > SPOF_SHARE:
            spof = top

    # Minimal flip set: remove largest supporting non-model findings (then model if needed) until band changes
    current_band = band(p)
    start_logit = _logit(p)
    flip: list[Item] = []
    rest = start_logit
    candidates = sorted(support_non_model, key=lambda i: -abs(i.log_lr))
    for i in candidates:
        flip.append(i)
        rest -= i.log_lr
        if band(_sig(rest)) != current_band:
            break
    else:
        flip = []

    flip_to = band(_sig(rest)) if flip else ""

    # Recommendation logic (keep | escalate under R8)
    recommendation = "keep"
    if b == "uncertain" or current_band == "uncertain":
        recommendation = "escalate"
        reason = (
            f"verdict is uncertain (p={p:.2f}, conflict_score={conflict_score:.2f}); "
            f"escalate under R8 rather than guessing"
        )
    elif conflict_score >= CONFLICT_ESCALATE:
        recommendation = "escalate"
        reason = (
            f"{conflict_score:.0%} of the evidence points toward {opp} "
            f"(R8: evidence conflicts; opposing log-odds={against_mass:.2f})"
        )
    elif is_escalated:
        recommendation = "keep"
        reason = "already escalated"
    elif spof is not None and len(flip) == 1 and directional_conflict > 0.10:
        recommendation = "escalate"
        reason = (
            f"single-point-of-failure verdict: {spof.tag} carries {spof_share:.0%} of supporting evidence "
            f"({abs(spof.log_lr):.2f} log-odds) while opposing signals exist; removing it flips verdict to {flip_to} (R8)"
        )
    elif spof is not None and len(flip) == 1 and not isinstance(answer, dict):
        # Legacy float interface behaviour: escalate on single-finding flip
        recommendation = "escalate"
        reason = (
            f"the {current_band} verdict rests on one finding ({spof.tag}, {spof_share:.0%} share); "
            f"removing it makes the case {flip_to}"
        )
    else:
        n_wrong = len(flip) if flip else len(wrong_findings)
        reason = (
            f"conflict_score={conflict_score:.2f} is below R8 escalation threshold ({CONFLICT_ESCALATE:.2f}); "
            f"{n_wrong or 'no'} finding(s) ({wrong_log_odds:.2f} log-odds) would have to be wrong to argue {opp}"
        )

    all_eids = list(base_ids)
    for it in items:
        all_eids.extend(it.entity_ids)
    ev = _split_evidence_ids(all_eids, case_id)

    att = Attestation(
        what=(
            f"Devil's advocate for {case_id} (verdict={b}, p={p:.2f}): opposite={opp}, "
            f"conflict_score={conflict_score:.3f}, spof={spof.tag if spof else 'none'}, "
            f"recommendation={recommendation}."
        ),
        evidence=ev,
        query="advocate",
        params={"case_id": case_id, "p": round(p, 4), "verdict": b},
    )

    return Advocacy(
        case_id=case_id,
        verdict=b,
        opposite_verdict=opp,
        p=p,
        band=current_band,
        prior_logit=_logit(p_model),
        items=items,
        conflict_score=conflict_score,
        spof=spof,
        spof_share=spof_share,
        wrong_findings=wrong_findings,
        wrong_log_odds=wrong_log_odds,
        flip_set=flip,
        flip_to=flip_to,
        recommendation=recommendation,
        reason=reason,
        attestation=att,
    )


def counterfactuals(
    answer: dict[str, Any] | float,
    ledger: list[Any] | None = None,
) -> CounterfactualResult:
    """Compute the minimal set of findings whose removal flips the verdict band
    (fraud >= 0.70, legitimate <= 0.30, else uncertain), and for each finding,
    the customer-safe evidence request that would test it (no SAR tip-off)."""
    case_id, verdict, _p_model, p_final, items, _esc, base_ids = _normalize_inputs(answer, ledger)
    total_logit = sum(i.log_lr for i in items)
    p = p_final if isinstance(answer, dict) and "case" in answer and "fraud_probability" in answer["case"] and not ledger else _sig(total_logit)
    cur_band = band(p)
    start_logit = _logit(p)

    # Order findings that push toward the current band (or away from 0.50 if uncertain)
    if cur_band == "fraud":
        # Need logit < _logit(0.70)
        candidates = sorted([i for i in items if i.log_lr > 0], key=lambda i: (- (i.tag != "model"), -abs(i.log_lr)))
    elif cur_band == "legitimate":
        # Need logit > _logit(0.30)
        candidates = sorted([i for i in items if i.log_lr < 0], key=lambda i: (- (i.tag != "model"), -abs(i.log_lr)))
    else:
        # Currently in uncertain band (0.30 < p < 0.70): find minimal removal that pushes out to fraud or legit,
        # or if removal of opposing item settles it.
        pos = sorted([i for i in items if i.log_lr > 0], key=lambda i: -abs(i.log_lr))
        neg = sorted([i for i in items if i.log_lr < 0], key=lambda i: -abs(i.log_lr))
        # Try removing positive items to drop <= 0.30 vs removing negative items to rise >= 0.70
        def _try_remove(pool: list[Item]) -> tuple[list[Item], float]:
            taken: list[Item] = []
            r = start_logit
            for it in pool:
                taken.append(it)
                r -= it.log_lr
                if band(_sig(r)) != cur_band:
                    return taken, r
            return taken, r

        t_pos, r_pos = _try_remove(pos)
        t_neg, r_neg = _try_remove(neg)
        b_pos, b_neg = band(_sig(r_pos)) != cur_band, band(_sig(r_neg)) != cur_band
        if b_pos and b_neg:
            candidates = t_pos if len(t_pos) <= len(t_neg) else t_neg
        elif b_pos:
            candidates = pos
        elif b_neg:
            candidates = neg
        else:
            candidates = pos if p >= 0.5 else neg

    minimal: list[Item] = []
    rest = start_logit
    flips = False
    for it in candidates:
        minimal.append(it)
        rest -= it.log_lr
        if band(_sig(rest)) != cur_band:
            flips = True
            break

    new_p = _sig(rest)
    target_band = band(new_p) if flips else _opposite_verdict(cur_band, p)

    requests_out: list[dict[str, Any]] = []
    seen_req: set[str] = set()
    for step_idx, it in enumerate(minimal, start=1):
        req_text = it.request
        # Enforce strict customer-safe wording (no tip-off)
        low_req = req_text.lower()
        for bad in FORBIDDEN_TIPOFF_WORDS:
            if bad in low_req:
                raise ValueError(f"Tip-off violation in evidence request: {req_text!r}")
        if req_text not in seen_req:
            seen_req.add(req_text)
            requests_out.append({
                "type": "customer_validation" if it.tag != "ring" else "step_up_auth",
                "finding_tag": it.tag,
                "finding_claim": it.claim,
                "log_odds_carried": round(it.log_lr, 4),
                "asked_after_step": step_idx,
                "request_text": req_text,
            })

    all_eids = list(base_ids)
    for it in minimal:
        all_eids.extend(it.entity_ids)
    ev = _split_evidence_ids(all_eids, case_id)

    att = Attestation(
        what=(
            f"Counterfactual for {case_id}: removing {len(minimal)} finding(s) "
            f"({', '.join(i.tag for i in minimal) or 'none'}) moves probability "
            f"{p:.2f} ({cur_band}) -> {new_p:.2f} ({band(new_p)})."
        ),
        evidence=ev,
        query="counterfactuals",
        params={"case_id": case_id, "current_p": round(p, 4), "current_band": cur_band},
    )

    return CounterfactualResult(
        case_id=case_id,
        current_verdict=verdict,
        current_p=p,
        target_band=target_band,
        new_p=new_p,
        minimal_findings=minimal,
        evidence_requests=requests_out,
        flips_band=flips,
        attestation=att,
    )


ROUND1_LEGIT_CASES = {
    "HHG-001", "HHG-002", "HHG-004", "HHG-005", "HHG-010",
    "HHG-012", "HHG-013", "HHG-015", "HHG-017", "HHG-020",
}


def round1_ground_truth() -> dict[str, str]:
    """Return ground truth ('legitimate' | 'fraud') for HHG-001..HHG-020, verified against closed_cases()."""
    cc = closed_cases()
    hhg = cc[cc.analyst_notes.str.contains("Case HHG-", na=False)].copy()
    extracted = dict(zip(hhg.analyst_notes.str.extract(r"Case (HHG-\d{3})")[0], hhg.outcome))
    truth: dict[str, str] = {}
    for i in range(1, 21):
        cid = f"HHG-{i:03d}"
        if cid in extracted:
            truth[cid] = "fraud" if extracted[cid] == "confirmed_fraud" else "legitimate"
        else:
            truth[cid] = "legitimate" if cid in ROUND1_LEGIT_CASES else "fraud"
    return truth


def evaluate_round1(cases_dir: Path | str = "cases") -> list[dict[str, Any]]:
    """Evaluate advocate() and counterfactuals() on Round 1 answers (cases/HHG-*.json)."""
    cdir = Path(cases_dir)
    truth = round1_ground_truth()
    rows: list[dict[str, Any]] = []

    for pth in sorted(cdir.glob("HHG-*.json")):
        ans = json.loads(pth.read_text())
        cid = ans["case_id"]
        adv = advocate(ans)
        cf = counterfactuals(ans)
        gt = truth.get(cid, "legitimate" if cid in ROUND1_LEGIT_CASES else "fraud")
        our_verdict = ans["case"]["verdict"]
        confidently_right = (our_verdict in ("fraud", "legitimate")) and (our_verdict == gt)
        rows.append({
            "case_id": cid,
            "truth": gt,
            "ours": our_verdict,
            "p": round(adv.p, 2),
            "conflict_score": round(adv.conflict_score, 3),
            "spof": adv.single_point_of_failure or "-",
            "spof_share": round(adv.spof_share, 2),
            "wrong_log_odds": round(adv.wrong_log_odds, 2),
            "flip_set": "+".join(i.tag for i in cf.minimal_findings) or "-",
            "flip_to": cf.target_band,
            "recommendation": adv.recommendation,
            "confidently_right": confidently_right,
            "reason": adv.reason,
        })
    return rows


def format_round1_table(rows: list[dict[str, Any]]) -> str:
    """Format the Round 1 evaluation table as plain text / Markdown."""
    header = (
        f"{'Case':<8} | {'Truth':<10} | {'Ours':<10} | {'p':>5} | {'Conflict':>8} | "
        f"{'SPOF':<12} | {'Flip Set':<20} | {'Rec':<8} | {'ConfRight'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['case_id']:<8} | {r['truth']:<10} | {r['ours']:<10} | {r['p']:>5.2f} | "
            f"{r['conflict_score']:>8.3f} | {r['spof']:<12} | {r['flip_set'][:20]:<20} | "
            f"{r['recommendation']:<8} | {str(r['confidently_right'])}"
        )
    return "\n".join(lines)


def replay_round1() -> list[dict[str, Any]]:
    """Backwards-compatible entry point returning the Round 1 evaluation rows."""
    return evaluate_round1()


if __name__ == "__main__":
    rows = evaluate_round1()
    print(format_round1_table(rows))
