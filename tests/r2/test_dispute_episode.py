"""Tests for Block N7 (`assay.r2.dispute_episode`).

Follows the repo TEST RULE:
- Assert only what we control: schema/shape, reproducibility, as-of correctness
  (no peeking at future transactions), attestation fields & JSON serialization,
  and that the backtest & exam measurement tables are produced.
- Print measured backtest metrics and HHF-003, HHF-004, HHF-005, HHF-007 outputs.
"""

from __future__ import annotations

import json
import pandas as pd

from assay.r2.attest import Attestation
from assay.r2.dispute_episode import (
    backtest_jul_oct_customer_reports,
    episode_for_dispute,
    get_dispute_index,
    inspect_exam_customer_reports,
    round1_episode_rule,
)
from assay.r2.frames import case_pack, tx_all


def test_episode_for_dispute_contract_and_attestation():
    """Verify return types, Attestation contract fields, and deterministic re-run."""
    tx = tx_all()
    cp = case_pack()
    row = cp[cp["case_id"] == "HHF-004"].iloc[0]

    txn_ids_1, why_1 = episode_for_dispute(row.card_id, int(row.flagged_txn_id), row.opened_at, tx)
    txn_ids_2, why_2 = episode_for_dispute(row.card_id, int(row.flagged_txn_id), row.opened_at, tx)

    assert isinstance(txn_ids_1, list)
    assert len(txn_ids_1) >= 1
    assert all(isinstance(x, int) for x in txn_ids_1)
    assert int(row.flagged_txn_id) in txn_ids_1
    assert txn_ids_1 == txn_ids_2

    assert isinstance(why_1, Attestation)
    d = why_1.to_dict()
    assert set(d.keys()) == {"what", "evidence", "how", "when"}
    assert d["how"]["query"] == "assay.r2.dispute_episode.episode_for_dispute"
    assert d["how"]["params"]["card_id"] == str(row.card_id)
    assert int(d["how"]["params"]["flagged_txn_id"]) == int(row.flagged_txn_id)
    assert d["evidence"]["txn_ids"] == [str(x) for x in txn_ids_1]
    assert d["evidence"]["card_ids"] == [str(row.card_id)]
    assert d["what"] == why_2.what

    # Verify valid JSON serialization
    parsed = json.loads(why_1.to_json())
    assert parsed["how"]["query"] == "assay.r2.dispute_episode.episode_for_dispute"


def test_as_of_strictness_no_future_leakage():
    """Verify `episode_for_dispute` never includes transactions after `as_of`
    and produces identical results when future transactions are stripped from `tx`."""
    tx = tx_all()
    idx = get_dispute_index(tx)
    t_map = dict(zip(idx.txn_id.tolist(), idx.df["t"].tolist()))

    cp = case_pack()
    for cid in ("HHF-003", "HHF-004", "HHF-005", "HHF-007"):
        row = cp[cp["case_id"] == cid].iloc[0]
        as_of_ts = pd.Timestamp(row.opened_at)
        flagged_id = int(row.flagged_txn_id)

        ids_full, _ = episode_for_dispute(row.card_id, flagged_id, as_of_ts, tx)
        assert all(pd.Timestamp(t_map[tid]) <= as_of_ts for tid in ids_full)

        # Truncate tx to t <= flagged_txn.t (earlier than opened_at) and check that
        # an earlier cutoff strictly excludes any transaction after that cutoff
        flag_t = pd.Timestamp(t_map[flagged_id])
        tx_cut = tx[tx["t"] <= flag_t].copy()
        ids_cut_on_full, _ = episode_for_dispute(row.card_id, flagged_id, flag_t, tx)
        ids_cut_on_trunc, _ = episode_for_dispute(row.card_id, flagged_id, flag_t, tx_cut)

        assert ids_cut_on_full == ids_cut_on_trunc
        assert all(pd.Timestamp(t_map[tid]) <= flag_t for tid in ids_cut_on_full)


def test_backtest_and_print_exam_results():
    """Run the Jul-Oct customer-report backtest and print the comparison table + exam cases."""
    bt_df, bt_meta = backtest_jul_oct_customer_reports()

    required_cols = {
        "cohort",
        "method",
        "n_cases",
        "exact_match_rate",
        "mean_jaccard",
        "first_txn_accuracy",
        "mean_exposure_abs_err",
        "median_exposure_abs_err",
    }
    assert required_cols.issubset(set(bt_df.columns))
    assert len(bt_df) >= 6

    exam_df = inspect_exam_customer_reports(("HHF-003", "HHF-004", "HHF-005", "HHF-007"))
    assert list(exam_df["case_id"]) == ["HHF-003", "HHF-004", "HHF-005", "HHF-007"]

    # Ensure round1_episode_rule also runs cleanly
    for row in exam_df.itertuples(index=False):
        r1_ids = round1_episode_rule(row.card_id, int(row.flagged_txn_id), row.opened_at)
        assert isinstance(r1_ids, list) and len(r1_ids) >= 1

    print("\n" + "=" * 88)
    print("N7 BACKTEST: Jul-Oct Closed Cases Triggered by Customer Report")
    print("=" * 88)
    print(bt_df.to_string(index=False))
    print(
        f"\nCleared (legitimate recurring dispute) cases in Jul-Oct: {bt_meta['n_jul_oct_cust_cleared']} "
        f"| Over-expanded (>1 txn) by N7: {len(bt_meta['cleared_overexpanded_case_ids'])} "
        f"({bt_meta['cleared_overexpanded_case_ids']})"
    )
    if bt_meta["multi_day_details"]:
        print("\nMulti-day customer-report closed cases in full history:")
        for d in bt_meta["multi_day_details"]:
            print(
                f"  {d['case_id']} [{d['method']}]: pred={d['pred_txns']} (${d['pred_exp']:.2f}) "
                f"vs true={d['true_txns']} (${d['true_exp']:.2f})"
            )

    print("\n" + "=" * 88)
    print("N7 EXAM PACK OUTPUTS: HHF-003, HHF-004, HHF-005, HHF-007")
    print("=" * 88)
    for row in exam_df.itertuples(index=False):
        print(
            f"{row.case_id} ({row.card_id}, flagged={row.flagged_txn_id}, opened={row.opened_at}):\n"
            f"  N7 episode_for_dispute : {row.n7_n_txns} txn(s), first={row.n7_first_txn_id}, "
            f"exposure=${row.n7_exposure_usd:.2f}, txns={row.n7_txn_ids}\n"
            f"  R1 24h window rule     : {row.r1_n_txns} txn(s), first={row.r1_first_txn_id}, "
            f"exposure=${row.r1_exposure_usd:.2f}, txns={row.r1_txn_ids}\n"
            f"  Attestation            : {row.what}"
        )
    print("=" * 88)
