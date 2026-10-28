"""Tests for Containment Planner (Challenge 7 - Block B18)."""

from __future__ import annotations

import pandas as pd
import pytest

from assay.r2.containment import (
    compute_card_exposure,
    compute_card_innocence,
    compute_evidence_strength,
    plan,
)
from assay.r2.frames import closed_cases, tx_all


def _find_real_rings(min_cards: int = 3, max_cards: int = 18, top_n: int = 2) -> list[tuple[str, pd.DataFrame]]:
    """Builds real rings from tx_all() grouped by shared non-generic device_profile

    containing closed-case fraud, capping cards-per-profile to exclude generic device profiles.
    """
    tx = tx_all()
    cc = closed_cases()

    fraud_cards = set(cc[cc["outcome"] == "confirmed_fraud"]["card_id"].dropna().tolist())

    dp_tx = tx[
        tx["device_profile"].notna()
        & (tx["device_profile"] != "")
        & (~tx["device_profile"].str.startswith("?"))
        & (~tx["device_profile"].str.startswith("na"))
    ].copy()

    profile_cards = dp_tx.groupby("device_profile")["card_id"].apply(lambda s: set(s.dropna()))

    valid_profiles = []
    for dp, cards in profile_cards.items():
        if min_cards <= len(cards) <= max_cards and len(cards & fraud_cards) >= 1:
            valid_profiles.append((dp, len(cards)))

    valid_profiles.sort(key=lambda x: x[1], reverse=True)

    rings = []
    for dp, _ in valid_profiles[:top_n]:
        ring_tx = dp_tx[dp_tx["device_profile"] == dp].copy()
        rings.append((dp, ring_tx))

    return rings


def test_containment_real_rings() -> None:
    """Builds 2 real rings from tx_all(), plans them, prints the frontiers,

    names each customer touched and their innocence evidence, and asserts no
    BLOCK on any card whose customer has only cleared history.
    """
    cc = closed_cases()

    cust_outcomes = cc.groupby("customer_id")["outcome"].apply(lambda s: set(s.dropna()))
    cleared_only_custs = set(
        cust_outcomes[cust_outcomes.apply(lambda s: "cleared" in s and "confirmed_fraud" not in s)].index
    )

    rings = _find_real_rings(min_cards=3, max_cards=18, top_n=2)
    assert len(rings) == 2, f"Expected 2 real rings from tx_all(), found {len(rings)}"

    for idx, (dp, ring_tx) in enumerate(rings, 1):
        ring_id = f"RING-REAL-{idx}"

        res = plan(
            ring=ring_tx,
            penalty_per_innocent=1000.0,
            closed_cases_df=cc,
            ring_id=ring_id,
        )

        print(f"\n==================== {ring_id} (Device: {dp[:45]}...) ====================")
        print(f"Total cards in ring: {res.total_cards}")
        print(f"Recommended blocks: {res.recommended_blocks}")
        print(f"Total exposure cut: ${res.total_exposure_cut:.2f}")
        print(f"Expected innocents touched: {res.expected_innocents_touched:.2f}")
        print(f"Net gain: ${res.net_gain:.2f}")

        print("\n--- Frontier Table ---")
        print(f"{'k':<3} | {'Cards Blocked':<35} | {'Exp Cut ($)':<12} | {'Exp Innocents':<13} | {'Net Gain ($)':<12}")
        print("-" * 85)
        for f in res.frontier:
            c_str = ", ".join(f.cards_blocked) if f.cards_blocked else "None"
            if len(c_str) > 35:
                c_str = c_str[:32] + "..."
            print(f"{f.k_blocks:<3} | {c_str:<35} | {f.exposure_cut:<12.2f} | {f.expected_innocents_touched:<13.2f} | {f.net_gain:<12.2f}")

        print("\n--- Card & Customer Innocence Evaluation & Plan Actions ---")
        for act in res.actions:
            ev = act.attestation.evidence
            print(f"Customer: {act.customer_id:<8} | Card: {act.card_id:<10} | Action: {act.action:<22} ({act.route:<4})")
            print(f"  P(Innocent): {act.innocence_prob:.3f} (Innocent={act.is_innocent}) | Ev Strength: {act.evidence_strength:.2f}")
            print(f"  Innocence Reasons: {', '.join(act.innocence_reasons)}")
            print(f"  Exposure Cut: ${act.exposure_cut:.2f} | Already Lost: ${act.already_lost:.2f} | Forward Exp: ${act.forward_exposure:.2f}")
            print(f"  Evidence IDs -> Cards: {ev.card_ids}, Customers: {ev.customer_ids}, Txns: {ev.txn_ids[:5]}, Cases: {ev.case_ids}")
            print(f"  Justification: {act.justification}\n")

        assert res.total_cards > 0
        assert len(res.frontier) > 0
        assert len(res.actions) == res.total_cards

        for act in res.actions:
            if act.customer_id in cleared_only_custs:
                assert act.action not in (
                    "BLOCK_CARD",
                    "BLOCK_ALL_CARDS",
                ), f"Policy breach: Card {act.card_id} (Customer {act.customer_id}) has ONLY cleared history but was assigned {act.action}!"

        for act in res.actions:
            assert act.attestation.what != ""
            assert act.attestation.query == "assay.r2.containment.plan_action"
            assert "card_id" in act.attestation.params


def test_exposure_and_innocence_models() -> None:
    """Tests exposure calculation and innocence model logic."""
    tx = tx_all()
    cc = closed_cases()

    sample_card = tx["card_id"].iloc[0]
    ctxns = tx[tx["card_id"] == sample_card]
    cust_id = ctxns["customer_id"].iloc[0]

    ev_str = compute_evidence_strength(sample_card, ctxns, closed_cases_df=cc)
    assert 0.05 <= ev_str <= 1.0

    exp = compute_card_exposure(ctxns, evidence_strength=ev_str, closed_cases_df=cc)
    assert exp.already_lost >= 0.0
    assert exp.forward_exposure >= 0.0

    inn = compute_card_innocence(
        card_id=sample_card,
        customer_id=cust_id,
        card_txns=ctxns,
        closed_cases_df=cc,
    )
    assert isinstance(inn.is_innocent, bool)
    assert 0.0 <= inn.innocence_prob <= 1.0


def test_policy_route_assignment() -> None:
    """Tests policy route assignment (L1 <= $2500, L2 > $2500)."""
    ring_data = pd.DataFrame(
        [
            {"TransactionID": 101, "card_id": "C001-K1", "customer_id": "C001", "amount": 1200.0, "risk_score": 0.8, "t": pd.Timestamp("2017-01-01 10:00:00")},
            {"TransactionID": 102, "card_id": "C001-K1", "customer_id": "C001", "amount": 1000.0, "risk_score": 0.9, "t": pd.Timestamp("2017-01-01 11:00:00")},
            {"TransactionID": 103, "card_id": "C002-K1", "customer_id": "C002", "amount": 5000.0, "risk_score": 0.95, "t": pd.Timestamp("2017-01-01 10:00:00")},
            {"TransactionID": 104, "card_id": "C002-K1", "customer_id": "C002", "amount": 4000.0, "risk_score": 0.95, "t": pd.Timestamp("2017-01-01 11:00:00")},
        ]
    )

    ev_strengths = {"C001-K1": 0.9, "C002-K1": 0.9}

    res = plan(
        ring=ring_data,
        penalty_per_innocent=500.0,
        evidence_strengths=ev_strengths,
        closed_cases_df=pd.DataFrame(),
    )

    for act in res.actions:
        if act.action == "BLOCK_CARD":
            if act.exposure_cut <= 2500.0:
                assert act.route == "L1"
            else:
                assert act.route == "L2"
