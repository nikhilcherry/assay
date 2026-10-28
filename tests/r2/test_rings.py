"""Test suite for assay.r2.rings graph ring detection, ring replay, and 2x2 ablation backtest."""

from __future__ import annotations

import pytest

from assay.r2.frames import case_pack, closed_cases, tx_all
from assay.r2.rings import Ring, ring_for, ring_replay, run_ring_ablation


def test_ring_for_exam_cards():
    cp = case_pack()
    tx = tx_all()
    cc = closed_cases()

    print("\n" + "=" * 80)
    print("EXAM CARDS RING DETECTION RESULTS (18 CARDS)")
    print("=" * 80)

    rings_found = 0
    for row in cp.itertuples():
        case_id = row.case_id
        card_id = row.card_id
        opened_at = row.opened_at

        ring_wcc = ring_for(card_id, opened_at, tx=tx, cc=cc, algo="wcc")
        ring_louvain = ring_for(card_id, opened_at, tx=tx, cc=cc, algo="louvain")

        if ring_wcc:
            rings_found += 1
            print(f"Case {case_id} | Card {card_id} | Opened: {opened_at}")
            print(f"  WCC Ring ID         : {ring_wcc.ring_id}")
            print(f"  Ring Size (Cards)   : {len(ring_wcc.cards)}")
            print(f"  Member Cards        : {ring_wcc.cards}")
            print(f"  First Seen Dates    :")
            for c, fdt in ring_wcc.first_seen.items():
                print(f"    - {c}: {fdt}")
            print(f"  Shared Elements & Cardinalities:")
            for elem, card_cnt in ring_wcc.element_cardinalities.items():
                print(f"    - {elem} (cardinality: {card_cnt})")
            print(f"  Prior Fraud Members & Closed Case IDs:")
            for c, cids in ring_wcc.member_prior_case_ids.items():
                print(f"    - {c}: {cids}")
            print(f"  Total Amount        : ${ring_wcc.total_amount:,.2f}")
            if ring_louvain:
                print(f"  Louvain Ring ID     : {ring_louvain.ring_id} ({len(ring_louvain.cards)} cards)")
            print(f"  Attestation         : {ring_wcc.attestation.what}")

            # Challenge 5: Ring Replay
            replay_frames = ring_replay(ring_wcc, tx=tx)
            print(f"\n  --- RING REPLAY ({len(replay_frames)} frames) ---")
            for frame in replay_frames:
                print(
                    f"    [{frame['date']}] Cards Joined: {len(frame['cards_joined'])} {frame['cards_joined']} | "
                    f"Elements Active: {len(frame['shared_elements'])} | "
                    f"Cum Amount: ${frame['cumulative_amount']:,.2f}"
                )
            print("-" * 80)
        else:
            print(f"Case {case_id} | Card {card_id} | Opened: {opened_at} -> No qualifying ring")

    print(f"\nTotal Exam Cards with Qualifying Rings: {rings_found} / {len(cp)}")
    assert len(cp) == 18


def test_ring_ablation_backtest():
    tx = tx_all()
    cc = closed_cases()

    res_wcc = run_ring_ablation(tx=tx, cc=cc, algo="wcc")
    res_louvain = run_ring_ablation(tx=tx, cc=cc, algo="louvain")

    print("\n" + "=" * 80)
    print("ABLATION BACKTEST 2x2 RESULTS (4,675 FRAUD vs 5,000 LEGIT CARD-DAYS)")
    print("=" * 80)

    print("\n--- Weakly Connected Components (WCC) ---")
    print(f"Contingency 2x2 Matrix:")
    print(f"  In Ring  | Fraud Episodes: {res_wcc['contingency_2x2']['fraud_episodes']['in_ring']:4d} | Legit Card-Days: {res_wcc['contingency_2x2']['legit_card_days']['in_ring']:4d}")
    print(f"  Not Ring | Fraud Episodes: {res_wcc['contingency_2x2']['fraud_episodes']['not_in_ring']:4d} | Legit Card-Days: {res_wcc['contingency_2x2']['legit_card_days']['not_in_ring']:4d}")
    print(f"  Total    | Fraud Episodes: {res_wcc['contingency_2x2']['fraud_episodes']['total']:4d} | Legit Card-Days: {res_wcc['contingency_2x2']['legit_card_days']['total']:4d}")
    print(f"P(In Ring | Fraud)           : {res_wcc['probabilities']['p_in_ring_given_fraud']:.4f} ({res_wcc['probabilities']['p_in_ring_given_fraud']:.2%})")
    print(f"P(In Ring | Legit)           : {res_wcc['probabilities']['p_in_ring_given_legit']:.4f} ({res_wcc['probabilities']['p_in_ring_given_legit']:.2%})")
    print(f"MEASURED LIKELIHOOD RATIO    : {res_wcc['likelihood_ratio']:.4f}x")
    print(f"95% CONFIDENCE INTERVAL      : [{res_wcc['ci_95'][0]:.4f}x, {res_wcc['ci_95'][1]:.4f}x]")

    print("\n--- Louvain Community Detection ---")
    print(f"Contingency 2x2 Matrix:")
    print(f"  In Ring  | Fraud Episodes: {res_louvain['contingency_2x2']['fraud_episodes']['in_ring']:4d} | Legit Card-Days: {res_louvain['contingency_2x2']['legit_card_days']['in_ring']:4d}")
    print(f"  Not Ring | Fraud Episodes: {res_louvain['contingency_2x2']['fraud_episodes']['not_in_ring']:4d} | Legit Card-Days: {res_louvain['contingency_2x2']['legit_card_days']['not_in_ring']:4d}")
    print(f"  Total    | Fraud Episodes: {res_louvain['contingency_2x2']['fraud_episodes']['total']:4d} | Legit Card-Days: {res_louvain['contingency_2x2']['legit_card_days']['total']:4d}")
    print(f"P(In Ring | Fraud)           : {res_louvain['probabilities']['p_in_ring_given_fraud']:.4f} ({res_louvain['probabilities']['p_in_ring_given_fraud']:.2%})")
    print(f"P(In Ring | Legit)           : {res_louvain['probabilities']['p_in_ring_given_legit']:.4f} ({res_louvain['probabilities']['p_in_ring_given_legit']:.2%})")
    print(f"MEASURED LIKELIHOOD RATIO    : {res_louvain['likelihood_ratio']:.4f}x")
    print(f"95% CONFIDENCE INTERVAL      : [{res_louvain['ci_95'][0]:.4f}x, {res_louvain['ci_95'][1]:.4f}x]")
    print("=" * 80)

    assert res_wcc["likelihood_ratio"] > 1.0


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
