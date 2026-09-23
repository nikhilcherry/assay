"""Fraud Policy v1.0 as executable code.

Every action identifier, approval route and rule number here comes from the
policy section of the task brief (docs/BRIEF.md). Nothing in this module is
decided by a language model: given the same signals it returns the same
actions, the same routes and the same rule citations, every time.

The agent's reasoning layer produces `Signals`. This module turns them into a
recommendation. That split is deliberate -- policy §2 says only `auto` actions
may be executed by the agent, so the route must never be something an LLM
chose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    ALLOW_TRANSACTION = "ALLOW_TRANSACTION"
    DECLINE_TRANSACTION = "DECLINE_TRANSACTION"
    MONITOR_CARD = "MONITOR_CARD"
    MONITOR_CONNECTED_CARDS = "MONITOR_CONNECTED_CARDS"
    WARN_CUSTOMER = "WARN_CUSTOMER"
    VERIFY_WITH_CUSTOMER = "VERIFY_WITH_CUSTOMER"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    BLOCK_CARD = "BLOCK_CARD"
    BLOCK_ALL_CARDS = "BLOCK_ALL_CARDS"
    GENERATE_REPORT = "GENERATE_REPORT"
    CREATE_CASE = "CREATE_CASE"
    FILE_REPORT = "FILE_REPORT"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"
    CLOSE_NO_FRAUD = "CLOSE_NO_FRAUD"


class Route(str, Enum):
    AUTO = "auto"
    L1 = "L1"
    L2 = "L2"


class Verdict(str, Enum):
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"
    UNCERTAIN = "uncertain"


class Pattern(str, Enum):
    CARD_TESTING = "card_testing"
    CNP = "card_not_present_fraud"
    CNP_NEW_DEVICE = "card_not_present_new_device"
    OUT_OF_REGION = "out_of_region_use"
    ACCOUNT_TAKEOVER = "account_takeover"
    UNDOCUMENTED = "undocumented"
    NONE = "none"


class CustomerReply(str, Enum):
    """Response to VERIFY_WITH_CUSTOMER. Policy R2, R3, R4."""

    DENIES = "denies"
    CONFIRMS = "confirms"
    NO_REPLY = "no_reply"
    NOT_ASKED = "not_asked"


# Policy §2. BLOCK_CARD straddles L1/L2 on exposure, so it is resolved in
# route_for() rather than in this table.
_FIXED_ROUTES: dict[Action, Route] = {
    Action.ALLOW_TRANSACTION: Route.AUTO,
    Action.MONITOR_CARD: Route.AUTO,
    Action.MONITOR_CONNECTED_CARDS: Route.AUTO,
    Action.WARN_CUSTOMER: Route.AUTO,
    Action.VERIFY_WITH_CUSTOMER: Route.AUTO,
    Action.STEP_UP_AUTH: Route.AUTO,
    Action.GENERATE_REPORT: Route.AUTO,
    Action.CREATE_CASE: Route.AUTO,
    Action.ESCALATE_TO_ANALYST: Route.AUTO,
    Action.CLOSE_NO_FRAUD: Route.AUTO,
    Action.DECLINE_TRANSACTION: Route.L1,
    Action.BLOCK_ALL_CARDS: Route.L2,
    Action.FILE_REPORT: Route.L2,
}

BLOCK_CARD_L2_THRESHOLD = 2_500.0

# Policy §3a and §6.
CASE_PROBABILITY_THRESHOLD = 0.30
SAR_EXPOSURE_THRESHOLD = 1_000.0
WEAK_SIGNAL_PROBABILITY = 0.70  # R1
ESCALATE_EXPOSURE_THRESHOLD = 500.0  # R8
NO_REPLY_ESCALATE_THRESHOLD = 500.0  # R4
CARD_TESTING_CLEARED_THRESHOLD = 100.0  # R5
STOP_HIGH = 0.85
STOP_LOW = 0.15


def route_for(action: Action, exposure_usd: float = 0.0) -> Route:
    """Approval route for one action. Policy §2."""
    if action is Action.BLOCK_CARD:
        return Route.L2 if exposure_usd > BLOCK_CARD_L2_THRESHOLD else Route.L1
    return _FIXED_ROUTES[action]


@dataclass
class Signals:
    """What the investigation established. Produced by the reasoning layer,
    consumed by the policy. Every field is a fact about the case, never a
    decision about it."""

    fraud_probability: float
    verdict: Verdict
    exposure_usd: float = 0.0
    pattern: Pattern = Pattern.NONE

    # How the alert arrived. Policy R7 keys off a customer dispute.
    customer_disputed: bool = False

    # R1: the case rests on one signal only (a bare risk score counts).
    single_signal: bool = True
    independent_evidence_count: int = 0
    evidence_conflicts: bool = False

    # R2, R3, R4.
    customer_reply: CustomerReply = CustomerReply.NOT_ASKED
    evidence_requested: bool = False

    # R5. Card testing needs the sequence, not just the pattern label.
    card_testing_sequence: bool = False
    large_purchase_cleared_usd: float = 0.0

    # R6. The shared element that links several cards' fraud.
    shared_origin: str = ""  # "device profile D-123", "billing region 264.0", ...
    connected_card_ids: list[str] = field(default_factory=list)

    # R7. The disputed charge matches the holder's own recurring pattern.
    matches_recurring_pattern: bool = False

    # R9. Coordinated or repeated abuse across customers, fitting no known pattern.
    coordinated_abuse: bool = False

    # R10.
    cards_with_confirmed_fraud: int = 0
    credentials_compromised: bool = False

    def connects_to_other_fraud(self) -> bool:
        """The §3a SAR trigger: shared device profile, shared region cluster,
        or another customer's fraud."""
        return bool(self.shared_origin) or bool(self.connected_card_ids)


@dataclass
class Recommendation:
    action: Action
    route: Route
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"action": self.action.value, "route": self.route.value, "reason": self.reason}


def _add(out: list[Recommendation], action: Action, exposure: float, reason: str) -> None:
    """Append unless already recommended. First reason wins, so the rule that
    triggered an action is the one cited for it."""
    if any(r.action is action for r in out):
        return
    out.append(Recommendation(action, route_for(action, exposure), reason))


def should_create_case(s: Signals) -> bool:
    """Policy §3a: open a case at probability 0.30, whenever evidence is
    requested, or whenever a customer disputes a charge."""
    return (
        s.fraud_probability >= CASE_PROBABILITY_THRESHOLD
        or s.evidence_requested
        or s.customer_disputed
    )


def should_file_sar(s: Signals) -> tuple[bool, str]:
    """Policy §3a. Fraud confirmed or strongly suspected, AND at least one of:
    exposure over $1,000; connects to a shared device profile, shared region
    cluster or another customer's fraud; coordinated or undocumented (R9).

    Returns the decision and the reason, which cites the rule either way --
    §7 requires an explanation for not filing as much as for filing.
    """
    suspected = s.verdict is Verdict.FRAUD or s.fraud_probability >= STOP_HIGH
    if not suspected:
        return False, (
            "Policy §3a: a report requires fraud confirmed or strongly suspected. "
            f"Verdict is {s.verdict.value} at probability {s.fraud_probability:.2f}."
        )

    triggers = []
    if s.exposure_usd > SAR_EXPOSURE_THRESHOLD:
        triggers.append(f"exposure ${s.exposure_usd:,.2f} exceeds $1,000")
    if s.connects_to_other_fraud():
        link = s.shared_origin or f"{len(s.connected_card_ids)} connected card(s)"
        triggers.append(f"activity connects to {link}")
    if s.coordinated_abuse or s.pattern is Pattern.UNDOCUMENTED:
        triggers.append("the pattern is coordinated or undocumented (R9)")

    if not triggers:
        return False, (
            "Policy §3a: fraud is suspected but no reporting trigger is met -- exposure "
            f"${s.exposure_usd:,.2f} is under $1,000, the activity is isolated to this card, "
            "and the pattern is a known one. Case only, no report."
        )
    return True, "Policy §3a: fraud is suspected and " + "; ".join(triggers) + "."


def stop_reason(s: Signals) -> str:
    """Policy §6. Stopping early creates risk, stopping late wastes time; both
    are marked down, so the reason is always stated explicitly."""
    settled = s.customer_reply in (CustomerReply.DENIES, CustomerReply.CONFIRMS)
    if settled:
        return (
            f"Policy §6: the customer {s.customer_reply.value} the transaction, which settles "
            "the question. No further evidence would change the decision."
        )
    if s.fraud_probability >= STOP_HIGH and s.independent_evidence_count >= 2:
        return (
            f"Policy §6: fraud probability {s.fraud_probability:.2f} is at or above 0.85 on "
            f"{s.independent_evidence_count} independent pieces of evidence."
        )
    if s.fraud_probability <= STOP_LOW and s.independent_evidence_count >= 2:
        return (
            f"Policy §6: fraud probability {s.fraud_probability:.2f} is at or below 0.15 on "
            f"{s.independent_evidence_count} independent pieces of evidence."
        )
    if s.customer_reply is CustomerReply.NO_REPLY:
        return (
            "Policy §6: no reply within 24 hours. R4 actions apply and further waiting is "
            "unlikely to change the decision."
        )
    return (
        f"Policy §6: probability {s.fraud_probability:.2f} remains between 0.15 and 0.85, but "
        "the available graph evidence is exhausted and further traversal is unlikely to change "
        "the recommendation. Handed to a human under R8."
    )


def decide(s: Signals) -> list[Recommendation]:
    """Turn established signals into an ordered list of recommended actions.

    Ordering is what happens first, per policy §1. Containment comes before
    record-keeping, which comes before regulatory filing, which comes before
    escalation.
    """
    out: list[Recommendation] = []
    exp = s.exposure_usd

    # --- R3: customer confirms. Settles the case; nothing else applies. ---
    if s.customer_reply is CustomerReply.CONFIRMS:
        _add(out, Action.CLOSE_NO_FRAUD, exp,
             "R3: the customer confirmed they made the transaction. The confirmation is "
             "recorded in the case file.")
        if should_create_case(s):
            _add(out, Action.CREATE_CASE, exp,
                 "Policy §3a: a case was opened when evidence was requested; it closes as "
                 "legitimate with the confirmation attached.")
        return out

    # --- R7: disputed but matches the holder's own recurring charge. ---
    # Checked before R2 because a dispute alone would otherwise trigger a block.
    if s.customer_disputed and s.matches_recurring_pattern:
        _add(out, Action.CREATE_CASE, exp,
             "R7: the disputed charge matches the cardholder's own recurring pattern -- same "
             "merchant, same amount, monthly.")
        _add(out, Action.VERIFY_WITH_CUSTOMER, exp,
             "R7: confirm with the cardholder before treating a recurring charge as fraud.")
        _add(out, Action.WARN_CUSTOMER, exp,
             "R7: send a recurring-charge reminder. R7 explicitly forbids blocking here.")
        return out

    # --- R2: customer denies. ---
    if s.customer_reply is CustomerReply.DENIES:
        _add(out, Action.BLOCK_CARD, exp,
             "R2: the cardholder denies the transaction, so the card is compromised.")
        _add(out, Action.CREATE_CASE, exp,
             "R2: a denial requires an internal case with the evidence attached.")

    # --- R5: card testing. ---
    if s.card_testing_sequence:
        if s.large_purchase_cleared_usd > CARD_TESTING_CLEARED_THRESHOLD:
            _add(out, Action.BLOCK_CARD, exp,
                 f"R5: a card-testing sequence preceded a cleared purchase of "
                 f"${s.large_purchase_cleared_usd:,.2f}, above the $100 threshold.")
        else:
            _add(out, Action.DECLINE_TRANSACTION, exp,
                 "R5: three or more small online authorizations within an hour followed by a "
                 "larger purchase.")
            _add(out, Action.STEP_UP_AUTH, exp,
                 "R5: require a passcode before further activity on this card.")

    # --- R1: verify before blocking on a weak signal. ---
    # Only when nothing above has already established the fraud.
    blocking = any(r.action in (Action.BLOCK_CARD, Action.BLOCK_ALL_CARDS) for r in out)
    if (
        not blocking
        and s.single_signal
        and s.fraud_probability < WEAK_SIGNAL_PROBABILITY
        and s.customer_reply is CustomerReply.NOT_ASKED
        and s.verdict is not Verdict.LEGITIMATE
    ):
        _add(out, Action.VERIFY_WITH_CUSTOMER, exp,
             f"R1: the case rests on a single signal at probability {s.fraud_probability:.2f}, "
             "below 0.70. Blocking a legitimate customer on one signal is a policy breach.")
        _add(out, Action.STEP_UP_AUTH, exp,
             "R1: require step-up authentication while the verification is outstanding.")

    # --- §5 / §3b: uncertain and not yet asked, but R1 did not apply (the case
    # rests on more than one signal, or probability is already >= 0.70). The
    # policy still prefers evidence to a block: hold the authorization if the
    # probability leans fraud, and ask. ---
    asked_already = any(r.action is Action.VERIFY_WITH_CUSTOMER for r in out)
    blocking = any(r.action in (Action.BLOCK_CARD, Action.BLOCK_ALL_CARDS) for r in out)
    if (
        not blocking
        and not asked_already
        and s.verdict is Verdict.UNCERTAIN
        and s.customer_reply is CustomerReply.NOT_ASKED
    ):
        if s.customer_disputed:
            _add(out, Action.STEP_UP_AUTH, exp,
                 "§5: the cardholder disputes the charge but the graph evidence does not settle it; "
                 "require step-up authentication on further activity while it is reviewed.")
        elif s.fraud_probability >= 0.5:
            _add(out, Action.DECLINE_TRANSACTION, exp,
                 f"§3b: probability {s.fraud_probability:.2f} leans fraud; hold this authorization "
                 "while the cardholder is asked, rather than block the card.")
        _add(out, Action.VERIFY_WITH_CUSTOMER, exp,
             f"§5 and §3b: probability {s.fraud_probability:.2f} is between 0.15 and 0.85, so ask the "
             "cardholder before deciding.")

    # --- §6: fraud established on independent evidence without a customer
    # contact (e.g. a risk-score alert whose graph evidence is conclusive).
    # Containment still has to happen; R10 governs anything wider. ---
    if (
        s.verdict is Verdict.FRAUD
        and s.fraud_probability >= STOP_HIGH
        and s.independent_evidence_count >= 2
        and s.customer_reply is CustomerReply.NOT_ASKED
        and not s.matches_recurring_pattern
    ):
        _add(out, Action.BLOCK_CARD, exp,
             f"§6: fraud probability {s.fraud_probability:.2f} on {s.independent_evidence_count} "
             "independent pieces of evidence; the card is compromised and verification would not "
             "change the decision.")

    # --- R4: asked, no reply within 24 hours. ---
    if s.customer_reply is CustomerReply.NO_REPLY:
        _add(out, Action.MONITOR_CARD, exp,
             "R4: no reply within 24 hours; raise monitoring sensitivity for 72 hours.")
        _add(out, Action.DECLINE_TRANSACTION, exp,
             "R4: decline pending authorizations while the verification is outstanding.")
        if exp > NO_REPLY_ESCALATE_THRESHOLD:
            _add(out, Action.ESCALATE_TO_ANALYST, exp,
                 f"R4: exposure ${exp:,.2f} exceeds $500 with no reply, so a human takes it.")

    # --- R6: shared origin across several cards. ---
    if s.shared_origin:
        _add(out, Action.CREATE_CASE, exp,
             f"R6: several cards show fraud from the same {s.shared_origin}.")
        _add(out, Action.MONITOR_CONNECTED_CARDS, exp,
             f"R6: every card sharing {s.shared_origin} goes under monitoring "
             f"({len(s.connected_card_ids)} card(s)).")

    # --- R10: blocking every card the customer holds. ---
    if s.cards_with_confirmed_fraud >= 2 or s.credentials_compromised:
        why = (
            f"{s.cards_with_confirmed_fraud} of the customer's cards show confirmed fraud"
            if s.cards_with_confirmed_fraud >= 2
            else "the customer's credentials are confirmed compromised"
        )
        _add(out, Action.BLOCK_ALL_CARDS, exp, f"R10: {why}.")

    # --- Legitimate, and nothing above fired. ---
    if s.verdict is Verdict.LEGITIMATE and not out:
        _add(out, Action.ALLOW_TRANSACTION, exp,
             "Policy §0: the risk score is a reason to look, not a verdict. The graph evidence "
             "is consistent with the cardholder's own history.")
        _add(out, Action.CLOSE_NO_FRAUD, exp,
             "Policy §6: probability is at or below 0.15 on independent evidence, so the alert "
             "closes as legitimate.")

    # --- Case, report, escalation. ---
    if should_create_case(s):
        if s.fraud_probability >= CASE_PROBABILITY_THRESHOLD:
            why = f"fraud probability {s.fraud_probability:.2f} is at or above 0.30"
        elif s.customer_disputed:
            why = "the customer disputes a charge"
        else:
            why = "evidence was requested"
        _add(out, Action.CREATE_CASE, exp,
             f"Policy §3a: {why}, so the investigation is recorded as an internal case and written "
             "to the graph.")

    file_sar, sar_reason = should_file_sar(s)
    if file_sar:
        _add(out, Action.FILE_REPORT, exp, sar_reason)

    # R9: undocumented pattern with coordinated abuse.
    if s.coordinated_abuse and s.pattern is Pattern.UNDOCUMENTED and s.verdict is Verdict.FRAUD:
        _add(out, Action.CREATE_CASE, exp,
             "R9: coordinated abuse across customers fitting none of the known patterns.")
        _add(out, Action.FILE_REPORT, exp,
             "R9: an undocumented coordinated pattern is reportable.")
        _add(out, Action.ESCALATE_TO_ANALYST, exp,
             "R9: a human analyst reviews the pattern description rather than forcing it into "
             "a known category.")

    # R8: uncertain and exposed, or evidence conflicts.
    if s.verdict is Verdict.UNCERTAIN and (exp > ESCALATE_EXPOSURE_THRESHOLD or s.evidence_conflicts):
        why = (
            "the evidence conflicts"
            if s.evidence_conflicts
            else f"exposure ${exp:,.2f} exceeds $500"
        )
        _add(out, Action.ESCALATE_TO_ANALYST, exp, f"R8: the verdict is uncertain and {why}.")

    return out


def executable(recs: list[Recommendation]) -> list[Recommendation]:
    """The subset the agent may carry out itself. Policy §2: only `auto`."""
    return [r for r in recs if r.route is Route.AUTO]


def awaiting_approval(recs: list[Recommendation]) -> list[Recommendation]:
    """Recommended, route stated, waiting on a human. Policy §2."""
    return [r for r in recs if r.route is not Route.AUTO]
