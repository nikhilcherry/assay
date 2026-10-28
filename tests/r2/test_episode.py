"""Tests and evaluation for B34 -- Episode boundaries (assay/r2/episode.py)."""

from __future__ import annotations

import pandas as pd

from assay.r2.episode import (
    Episode,
    baseline_flagged_only,
    baseline_same_product_7d,
    episode,
    evaluate_episodes,
    learn_hmm_params,
)
from assay.r2.frames import case_pack, closed_cases, tx_all


def test_hmm_params_learned():
    params = learn_hmm_params()
    assert params.pi.shape == (2,)
    assert params.trans_by_gap.shape == (5, 2, 2)
    # State 1 (compromised) should have higher emission probability on novel channel/device/region than State 0
    assert params.emit_chan_nov[1, 2] > params.emit_chan_nov[0, 2]
    assert params.emit_reg_nov[1, 2] > params.emit_reg_nov[0, 2]


def test_oct_nov_dec_backtest_and_exam_episodes():
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    # Evaluate on Oct + Nov-Dec closed cases (opened_at >= '2016-10-01')
    cc_eval = cc[(cc["outcome"] == "confirmed_fraud") & (cc["opened_at"] >= "2016-10-01")].copy()
    assert len(cc_eval) > 500

    df_eval = evaluate_episodes(cc_eval, tx)
    print("\n=== B34 Episode Evaluation: Oct-Dec Confirmed-Fraud Closed Cases (as-of) ===")
    print(df_eval.to_string(index=False))

    # Also break down Nov-Dec specifically
    cc_nov_dec = cc[(cc["outcome"] == "confirmed_fraud") & (cc["opened_at"] >= "2016-11-01")].copy()
    df_nov_dec = evaluate_episodes(cc_nov_dec, tx)
    print("\n=== B34 Episode Evaluation: Nov-Dec Only Confirmed-Fraud Closed Cases (as-of) ===")
    print(df_nov_dec.to_string(index=False))

    hmm_row = df_eval[df_eval["method"] == "hmm_2state"].iloc[0]
    base_a = df_eval[df_eval["method"] == "baseline_a_flagged_only"].iloc[0]
    base_b = df_eval[df_eval["method"] == "baseline_b_same_prod_7d"].iloc[0]

    # HMM must beat both baselines on exact match rate, Jaccard, first-txn accuracy, and exposure MAE
    assert hmm_row["exact_match_rate"] > base_a["exact_match_rate"]
    assert hmm_row["exact_match_rate"] > base_b["exact_match_rate"]
    assert hmm_row["mean_jaccard"] > base_a["mean_jaccard"]
    assert hmm_row["mean_jaccard"] > base_b["mean_jaccard"]
    assert hmm_row["first_txn_accuracy"] > base_a["first_txn_accuracy"]
    assert hmm_row["mean_exposure_abs_err"] < base_a["mean_exposure_abs_err"]
    assert hmm_row["mean_exposure_abs_err"] < base_b["mean_exposure_abs_err"]

    # Also check behaviour on cleared (legitimate) cases in Oct-Dec:
    # We must know how often the episode finder expands beyond 1 txn on cleared cases
    cc_cleared = cc[(cc["outcome"] == "cleared") & (cc["opened_at"] >= "2016-10-01")].copy()
    cleared_expanded = []
    for row in cc_cleared.itertuples(index=False):
        if not row.txn_list:
            continue
        seed_txn = int(row.txn_list[-1])
        ep = episode(row.card_id, seed_txn, row.opened_at, tx)
        if len(ep.txn_ids) > 1:
            cleared_expanded.append(row.case_id)
    print(
        f"\nCleared cases (Oct-Dec): {len(cc_cleared)} total, "
        f"expanded to >1 txn on {len(cleared_expanded)} ({len(cleared_expanded)/max(len(cc_cleared),1):.1%})"
    )

    # Run and print episode for each of the 18 exam cases
    print("\n=== B34 Episodes for the 18 Exam Cases (HHF-001 .. HHF-018) ===")
    exam_rows = []
    tx_amt_map = dict(zip(tx["TransactionID"].astype(int), tx["amount"].astype(float)))
    tx_t_map = dict(zip(tx["TransactionID"].astype(int), tx["t"]))

    for r in cp.itertuples(index=False):
        ep = episode(r.card_id, int(r.flagged_txn_id), r.opened_at, tx)
        assert isinstance(ep, Episode)
        assert int(r.flagged_txn_id) in ep.txn_ids
        assert ep.first_txn_id == ep.txn_ids[0]
        # Verify first_txn_id has the earliest timestamp among ep.txn_ids
        times = [tx_t_map[tid] for tid in ep.txn_ids]
        assert times[0] == min(times)
        # Verify exposure_usd equals the sum of affected amounts
        expected_exp = round(sum(abs(tx_amt_map[tid]) for tid in ep.txn_ids), 2)
        assert abs(ep.exposure_usd - expected_exp) < 1e-6
        assert ep.attestation is not None
        att_d = ep.attestation.to_dict()
        assert str(r.flagged_txn_id) in att_d["evidence"]["txn_ids"]

        exam_rows.append({
            "case_id": r.case_id,
            "card_id": r.card_id,
            "flagged_txn": int(r.flagged_txn_id),
            "first_txn_id": ep.first_txn_id,
            "n_txns": len(ep.txn_ids),
            "exposure_usd": ep.exposure_usd,
            "confidence": ep.confidence,
            "txn_ids": "|".join(str(x) for x in ep.txn_ids),
        })

    df_exam = pd.DataFrame(exam_rows)
    print(df_exam.to_string(index=False))
