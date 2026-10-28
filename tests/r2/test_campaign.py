"""Tests for B28 -- CampaignDetector (undocumented:coordinated_campaign)."""

from __future__ import annotations

import pandas as pd

from assay.r2.detectors.campaign import CampaignDetector, query_campaign_cohort
from assay.r2.finding import CaseCtx, REGISTRY
from assay.r2.frames import case_pack, closed_cases, tx_all


def test_registry_and_exam_cases():
    assert "campaign" in REGISTRY
    det = CampaignDetector()
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    cc_ts = pd.to_datetime(cc["closed_at"])

    print("\n=== B28 CampaignDetector: 18 Exam Cases ===")
    print(f"{'Case':<8} {'InCohort':<9} {'Size':>4} {'Txns':>4} {'LR':>7}  {'Signature'}")
    for row in cp.itertuples():
        opened = pd.Timestamp(row.opened_at)
        ctx = CaseCtx(
            case_id=str(row.case_id),
            flagged_txn_id=int(row.flagged_txn_id),
            card_id=str(row.card_id),
            customer_id=str(row.customer_id),
            opened_at=opened,
            tx=tx.loc[tx["t"] <= opened],
            cc=cc.loc[cc_ts <= opened],
        )
        findings = det.run(ctx)
        if findings:
            f = findings[0]
            sig = f.claim.params["signature"]
            cohort_size = len(f.claim.evidence.card_ids)
            print(
                f"{row.case_id:<8} {'True':<9} {cohort_size:4d} {len(f.affected_txn_ids):4d} "
                f"{f.lr:7.2f}  {sig}"
            )
            # Re-run attestation query to verify exact reproducibility
            re_run = query_campaign_cohort(
                card_id=f.claim.params["card_id"],
                opened_at=f.claim.params["opened_at"],
                flagged_txn_id=f.claim.params["flagged_txn_id"],
            )
            assert re_run["matched"] is True
            assert re_run["signature"] == sig
            assert re_run["member_episodes"] == f.claim.params["member_episodes"]
            assert int(row.flagged_txn_id) in f.affected_txn_ids
            assert len(f.connected_card_ids) >= 4
        else:
            print(f"{row.case_id:<8} {'False':<9} {0:4d} {0:4d} {0.0:7.2f}  None")


def test_backtest_and_signature_lr_table():
    det = CampaignDetector()
    tx = tx_all()
    cc = closed_cases()

    bt = det.backtest(cc, tx)
    jo_stats = det.jul_oct_legit_card_day_stats()
    sig_table = det.signature_lr_table()

    n_cleared = int((cc["outcome"] == "cleared").sum())
    cleared_fp_rate = len(bt.legit_acted_against) / max(1, n_cleared)

    print("\n=== B28 Per-Signature LR Table (Jul-Oct Calibrated) ===")
    if not sig_table.empty:
        print(sig_table.to_string(index=False))

    print("\n=== B28 Backtest Summary ===")
    print(f"  Total closed cases:            {bt.n_cases}")
    print(f"  Confirmed fraud caught:        {len(bt.caught)} / {len(bt.caught) + len(bt.missed)} ({bt.recall:.4%})")
    print(f"  Cleared cases fired on:        {len(bt.legit_acted_against)} / {n_cleared} ({cleared_fp_rate:.4%})")
    print(f"  Legit cases acted against:     {bt.legit_acted_against}")
    print(f"  Measured LR (vs legit sample): {bt.lr_measured}")

    print("\n=== B28 Jul-Oct Seeded Legit Card-Day Sample Stats ===")
    for k, v in jo_stats.items():
        print(f"  {k}: {v}")

    # Every accepted signature must have legit firing rate <= 1% and LR >= 3.0
    accepted = sig_table[sig_table["accepted"]]
    assert not accepted.empty
    assert (accepted["legit_rate"] <= 0.01).all()
    assert (accepted["lr"] >= 3.0).all()
    assert jo_stats["overall_legit_sample_rate"] <= 0.01
