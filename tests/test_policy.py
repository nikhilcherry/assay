"""One test per policy rule, plus the negative case for each.

The negatives matter more than the positives here: the brief says half the
cases are legitimate and an agent that blocks everything scores badly, so most
of these assert that a rule does *not* fire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assay.policy import (  # noqa: E402
    Action,
    CustomerReply,
    Pattern,
    Route,
    Signals,
    Verdict,
    awaiting_approval,
    decide,
    executable,
    route_for,
    should_create_case,
    should_file_sar,
    stop_reason,
)


def actions(s: Signals) -> set[Action]:
    return {r.action for r in decide(s)}


def rec(s: Signals, action: Action):
    return next(r for r in decide(s) if r.action is action)


# --------------------------------------------------------------------------
# §2 Approval routing
# --------------------------------------------------------------------------

def test_auto_actions_route_auto():
    for a in (
        Action.ALLOW_TRANSACTION, Action.MONITOR_CARD, Action.MONITOR_CONNECTED_CARDS,
        Action.WARN_CUSTOMER, Action.VERIFY_WITH_CUSTOMER, Action.STEP_UP_AUTH,
        Action.GENERATE_REPORT, Action.CREATE_CASE, Action.ESCALATE_TO_ANALYST,
        Action.CLOSE_NO_FRAUD,
    ):
        assert route_for(a) is Route.AUTO, a


def test_decline_is_l1():
    assert route_for(Action.DECLINE_TRANSACTION) is Route.L1


def test_block_card_is_l1_at_or_below_2500():
    assert route_for(Action.BLOCK_CARD, 2_500.0) is Route.L1
    assert route_for(Action.BLOCK_CARD, 100.0) is Route.L1


def test_block_card_is_l2_above_2500():
    assert route_for(Action.BLOCK_CARD, 2_500.01) is Route.L2


def test_block_all_and_file_report_always_l2():
    assert route_for(Action.BLOCK_ALL_CARDS, 1.0) is Route.L2
    assert route_for(Action.FILE_REPORT, 1.0) is Route.L2


def test_agent_may_only_execute_auto():
    s = Signals(fraud_probability=0.95, verdict=Verdict.FRAUD, exposure_usd=5_000.0,
                customer_reply=CustomerReply.DENIES)
    recs = decide(s)
    assert all(r.route is Route.AUTO for r in executable(recs))
    assert all(r.route is not Route.AUTO for r in awaiting_approval(recs))
    assert Action.BLOCK_CARD in {r.action for r in awaiting_approval(recs)}


# --------------------------------------------------------------------------
# R1 Verify before you block on a weak signal
# --------------------------------------------------------------------------

def test_r1_weak_single_signal_verifies_instead_of_blocking():
    s = Signals(fraud_probability=0.45, verdict=Verdict.UNCERTAIN, single_signal=True)
    a = actions(s)
    assert Action.VERIFY_WITH_CUSTOMER in a
    assert Action.BLOCK_CARD not in a
    assert "R1" in rec(s, Action.VERIFY_WITH_CUSTOMER).reason


def test_r1_does_not_fire_above_070():
    s = Signals(fraud_probability=0.80, verdict=Verdict.FRAUD, single_signal=True,
                independent_evidence_count=2)
    assert Action.VERIFY_WITH_CUSTOMER not in actions(s)


def test_r1_does_not_fire_on_multiple_signals():
    # Verification is still requested, but under §5/§3b rather than R1, and
    # the authorization is held because the probability leans fraud.
    s = Signals(fraud_probability=0.55, verdict=Verdict.UNCERTAIN, single_signal=False)
    assert not rec(s, Action.VERIFY_WITH_CUSTOMER).reason.startswith("R1")
    assert Action.DECLINE_TRANSACTION in actions(s)
    assert Action.STEP_UP_AUTH not in actions(s)


def test_uncertain_below_half_asks_without_declining():
    s = Signals(fraud_probability=0.35, verdict=Verdict.UNCERTAIN, single_signal=False)
    assert Action.VERIFY_WITH_CUSTOMER in actions(s)
    assert Action.DECLINE_TRANSACTION not in actions(s)


def test_established_fraud_without_contact_is_contained():
    s = Signals(fraud_probability=0.93, verdict=Verdict.FRAUD, independent_evidence_count=3,
                single_signal=False, exposure_usd=300.0)
    r = rec(s, Action.BLOCK_CARD)
    assert r.route is Route.L1 and r.reason.startswith("§6")


def test_high_probability_on_one_piece_of_evidence_is_not_blocked():
    s = Signals(fraud_probability=0.93, verdict=Verdict.FRAUD, independent_evidence_count=1)
    assert Action.BLOCK_CARD not in actions(s)


def test_r1_does_not_fire_once_the_customer_has_answered():
    s = Signals(fraud_probability=0.45, verdict=Verdict.UNCERTAIN, single_signal=True,
                customer_reply=CustomerReply.DENIES)
    assert Action.BLOCK_CARD in actions(s)


# --------------------------------------------------------------------------
# R2 Customer denies
# --------------------------------------------------------------------------

def test_r2_denial_blocks_and_creates_case():
    s = Signals(fraud_probability=0.90, verdict=Verdict.FRAUD, exposure_usd=200.0,
                customer_reply=CustomerReply.DENIES)
    a = actions(s)
    assert Action.BLOCK_CARD in a and Action.CREATE_CASE in a
    assert "R2" in rec(s, Action.BLOCK_CARD).reason


def test_r2_adds_report_above_1000():
    s = Signals(fraud_probability=0.90, verdict=Verdict.FRAUD, exposure_usd=1_500.0,
                customer_reply=CustomerReply.DENIES)
    assert Action.FILE_REPORT in actions(s)


def test_r2_adds_report_when_connected_even_if_cheap():
    s = Signals(fraud_probability=0.90, verdict=Verdict.FRAUD, exposure_usd=80.0,
                customer_reply=CustomerReply.DENIES,
                shared_origin="device profile D-77", connected_card_ids=["C1-K1", "C2-K1"])
    assert Action.FILE_REPORT in actions(s)


def test_r2_no_report_when_small_and_isolated():
    s = Signals(fraud_probability=0.90, verdict=Verdict.FRAUD, exposure_usd=49.0,
                customer_reply=CustomerReply.DENIES, pattern=Pattern.CNP)
    assert Action.FILE_REPORT not in actions(s)


# --------------------------------------------------------------------------
# R3 Customer confirms
# --------------------------------------------------------------------------

def test_r3_confirmation_closes_no_fraud():
    s = Signals(fraud_probability=0.40, verdict=Verdict.LEGITIMATE,
                customer_reply=CustomerReply.CONFIRMS, evidence_requested=True)
    a = actions(s)
    assert Action.CLOSE_NO_FRAUD in a
    assert Action.BLOCK_CARD not in a and Action.FILE_REPORT not in a


# --------------------------------------------------------------------------
# R4 No reply within 24 hours
# --------------------------------------------------------------------------

def test_r4_no_reply_monitors_and_declines():
    s = Signals(fraud_probability=0.50, verdict=Verdict.UNCERTAIN, exposure_usd=200.0,
                customer_reply=CustomerReply.NO_REPLY, evidence_requested=True)
    a = actions(s)
    assert Action.MONITOR_CARD in a and Action.DECLINE_TRANSACTION in a
    assert Action.ESCALATE_TO_ANALYST not in a


def test_r4_escalates_above_500():
    s = Signals(fraud_probability=0.50, verdict=Verdict.UNCERTAIN, exposure_usd=900.0,
                customer_reply=CustomerReply.NO_REPLY, evidence_requested=True)
    assert Action.ESCALATE_TO_ANALYST in actions(s)


# --------------------------------------------------------------------------
# R5 Card testing
# --------------------------------------------------------------------------

def test_r5_sequence_declines_and_steps_up():
    s = Signals(fraud_probability=0.75, verdict=Verdict.FRAUD, exposure_usd=60.0,
                pattern=Pattern.CARD_TESTING, card_testing_sequence=True,
                independent_evidence_count=3)
    a = actions(s)
    assert Action.DECLINE_TRANSACTION in a and Action.STEP_UP_AUTH in a
    assert Action.BLOCK_CARD not in a


def test_r5_blocks_once_a_purchase_over_100_cleared():
    s = Signals(fraud_probability=0.90, verdict=Verdict.FRAUD, exposure_usd=450.0,
                pattern=Pattern.CARD_TESTING, card_testing_sequence=True,
                large_purchase_cleared_usd=250.0, independent_evidence_count=3)
    a = actions(s)
    assert Action.BLOCK_CARD in a
    assert "R5" in rec(s, Action.BLOCK_CARD).reason


def test_r5_does_not_fire_without_the_sequence():
    s = Signals(fraud_probability=0.75, verdict=Verdict.FRAUD, exposure_usd=60.0,
                pattern=Pattern.CNP, independent_evidence_count=2)
    assert Action.DECLINE_TRANSACTION not in actions(s)


# --------------------------------------------------------------------------
# R6 Shared origin
# --------------------------------------------------------------------------

def test_r6_shared_origin_creates_case_reports_and_monitors_connected():
    s = Signals(fraud_probability=0.88, verdict=Verdict.FRAUD, exposure_usd=300.0,
                shared_origin="device profile D-99",
                connected_card_ids=["C1-K1", "C2-K1", "C3-K2"],
                independent_evidence_count=3)
    a = actions(s)
    assert {Action.CREATE_CASE, Action.FILE_REPORT, Action.MONITOR_CONNECTED_CARDS} <= a
    assert "D-99" in rec(s, Action.MONITOR_CONNECTED_CARDS).reason


def test_r6_does_not_fire_on_an_isolated_card():
    s = Signals(fraud_probability=0.88, verdict=Verdict.FRAUD, exposure_usd=300.0,
                independent_evidence_count=2)
    assert Action.MONITOR_CONNECTED_CARDS not in actions(s)


# --------------------------------------------------------------------------
# R7 Disputed but legitimate
# --------------------------------------------------------------------------

def test_r7_recurring_charge_is_never_blocked():
    s = Signals(fraud_probability=0.20, verdict=Verdict.LEGITIMATE, exposure_usd=0.0,
                customer_disputed=True, matches_recurring_pattern=True)
    a = actions(s)
    assert {Action.CREATE_CASE, Action.VERIFY_WITH_CUSTOMER, Action.WARN_CUSTOMER} <= a
    assert Action.BLOCK_CARD not in a and Action.DECLINE_TRANSACTION not in a


def test_r7_beats_r2_when_both_could_apply():
    # A dispute that matches the holder's own recurring charge must not block,
    # even though a dispute alone would.
    s = Signals(fraud_probability=0.35, verdict=Verdict.UNCERTAIN,
                customer_disputed=True, matches_recurring_pattern=True,
                customer_reply=CustomerReply.DENIES)
    assert Action.BLOCK_CARD not in actions(s)


# --------------------------------------------------------------------------
# R8 Escalate when uncertain and exposed
# --------------------------------------------------------------------------

def test_r8_escalates_uncertain_above_500():
    s = Signals(fraud_probability=0.50, verdict=Verdict.UNCERTAIN, exposure_usd=800.0,
                single_signal=False)
    assert Action.ESCALATE_TO_ANALYST in actions(s)


def test_r8_escalates_on_conflicting_evidence_at_any_exposure():
    s = Signals(fraud_probability=0.50, verdict=Verdict.UNCERTAIN, exposure_usd=50.0,
                single_signal=False, evidence_conflicts=True)
    assert Action.ESCALATE_TO_ANALYST in actions(s)


def test_r8_does_not_escalate_small_uncertain_cases():
    s = Signals(fraud_probability=0.50, verdict=Verdict.UNCERTAIN, exposure_usd=120.0,
                single_signal=False)
    assert Action.ESCALATE_TO_ANALYST not in actions(s)


# --------------------------------------------------------------------------
# R9 Undocumented patterns
# --------------------------------------------------------------------------

def test_r9_undocumented_coordinated_abuse():
    s = Signals(fraud_probability=0.80, verdict=Verdict.FRAUD, exposure_usd=400.0,
                pattern=Pattern.UNDOCUMENTED, coordinated_abuse=True,
                independent_evidence_count=3)
    a = actions(s)
    assert {Action.CREATE_CASE, Action.FILE_REPORT, Action.ESCALATE_TO_ANALYST} <= a


def test_r9_report_fires_below_the_1000_threshold():
    # R9 is its own SAR trigger; exposure is irrelevant to it.
    s = Signals(fraud_probability=0.80, verdict=Verdict.FRAUD, exposure_usd=75.0,
                pattern=Pattern.UNDOCUMENTED, coordinated_abuse=True,
                independent_evidence_count=2)
    assert Action.FILE_REPORT in actions(s)


# --------------------------------------------------------------------------
# R10 Never BLOCK_ALL_CARDS
# --------------------------------------------------------------------------

def test_r10_blocks_all_on_two_confirmed_cards():
    s = Signals(fraud_probability=0.95, verdict=Verdict.FRAUD, exposure_usd=3_000.0,
                cards_with_confirmed_fraud=2, independent_evidence_count=3)
    assert Action.BLOCK_ALL_CARDS in actions(s)


def test_r10_blocks_all_on_confirmed_credential_compromise():
    s = Signals(fraud_probability=0.95, verdict=Verdict.FRAUD, exposure_usd=900.0,
                pattern=Pattern.ACCOUNT_TAKEOVER, credentials_compromised=True,
                independent_evidence_count=3)
    assert Action.BLOCK_ALL_CARDS in actions(s)


def test_r10_refuses_on_a_single_compromised_card():
    s = Signals(fraud_probability=0.95, verdict=Verdict.FRAUD, exposure_usd=3_000.0,
                cards_with_confirmed_fraud=1, customer_reply=CustomerReply.DENIES,
                independent_evidence_count=3)
    a = actions(s)
    assert Action.BLOCK_ALL_CARDS not in a
    assert Action.BLOCK_CARD in a


# --------------------------------------------------------------------------
# §3a A case is not a report
# --------------------------------------------------------------------------

def test_case_opens_at_030():
    assert should_create_case(Signals(fraud_probability=0.30, verdict=Verdict.UNCERTAIN))
    assert not should_create_case(Signals(fraud_probability=0.29, verdict=Verdict.LEGITIMATE))


def test_case_opens_whenever_evidence_is_requested():
    assert should_create_case(
        Signals(fraud_probability=0.10, verdict=Verdict.UNCERTAIN, evidence_requested=True))


def test_case_opens_on_a_dispute():
    assert should_create_case(
        Signals(fraud_probability=0.05, verdict=Verdict.LEGITIMATE, customer_disputed=True))


def test_no_report_without_suspected_fraud():
    file_it, why = should_file_sar(
        Signals(fraud_probability=0.20, verdict=Verdict.LEGITIMATE, exposure_usd=9_000.0))
    assert not file_it and "§3a" in why


def test_report_reason_is_given_even_when_declining():
    file_it, why = should_file_sar(
        Signals(fraud_probability=0.90, verdict=Verdict.FRAUD, exposure_usd=200.0,
                pattern=Pattern.CNP))
    assert not file_it
    assert "under $1,000" in why  # §7: explain the non-filing too


def test_sar_and_file_report_always_agree():
    # answer.sar.file must match whether FILE_REPORT is in the final actions.
    for s in (
        Signals(fraud_probability=0.95, verdict=Verdict.FRAUD, exposure_usd=5_000.0),
        Signals(fraud_probability=0.95, verdict=Verdict.FRAUD, exposure_usd=50.0,
                pattern=Pattern.CNP),
        Signals(fraud_probability=0.10, verdict=Verdict.LEGITIMATE),
        Signals(fraud_probability=0.80, verdict=Verdict.FRAUD, pattern=Pattern.UNDOCUMENTED,
                coordinated_abuse=True),
    ):
        file_it, _ = should_file_sar(s)
        assert file_it == (Action.FILE_REPORT in actions(s))


# --------------------------------------------------------------------------
# §6 Stopping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p,n,expect", [
    (0.90, 2, "at or above 0.85"),
    (0.10, 2, "at or below 0.15"),
    (0.90, 1, "exhausted"),   # high probability on one signal is not enough to stop
    (0.50, 5, "exhausted"),
])
def test_stop_reason_states_which_condition_was_met(p, n, expect):
    s = Signals(fraud_probability=p, verdict=Verdict.UNCERTAIN, independent_evidence_count=n)
    assert expect in stop_reason(s)


def test_a_reply_settles_the_question():
    s = Signals(fraud_probability=0.50, verdict=Verdict.FRAUD,
                customer_reply=CustomerReply.DENIES, independent_evidence_count=1)
    assert "settles the question" in stop_reason(s)


# --------------------------------------------------------------------------
# §7 Explaining, and the legitimate path
# --------------------------------------------------------------------------

def test_every_recommendation_cites_something():
    s = Signals(fraud_probability=0.88, verdict=Verdict.FRAUD, exposure_usd=1_800.0,
                shared_origin="billing region 264.0", connected_card_ids=["C1-K1"],
                customer_reply=CustomerReply.DENIES, independent_evidence_count=3)
    for r in decide(s):
        assert r.reason, r.action
        assert any(tok in r.reason for tok in ("R1", "R2", "R3", "R4", "R5", "R6",
                                               "R7", "R8", "R9", "R10", "§"))


def test_a_legitimate_case_allows_and_closes():
    s = Signals(fraud_probability=0.04, verdict=Verdict.LEGITIMATE,
                independent_evidence_count=3)
    a = actions(s)
    assert a == {Action.ALLOW_TRANSACTION, Action.CLOSE_NO_FRAUD}


def test_a_legitimate_case_never_files_or_blocks():
    s = Signals(fraud_probability=0.04, verdict=Verdict.LEGITIMATE, exposure_usd=0.0,
                independent_evidence_count=3)
    a = actions(s)
    assert not ({Action.FILE_REPORT, Action.BLOCK_CARD, Action.BLOCK_ALL_CARDS,
                 Action.DECLINE_TRANSACTION} & a)
