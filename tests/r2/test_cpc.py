import pandas as pd

from assay.r2.frames import case_pack, tx_all, closed_cases
from assay.r2.finding import CaseCtx
from assay.r2.detectors.cpc import CommonPointOfCompromise


def test_cpc_stratified_backtest_and_exam_cases():
    cp = case_pack()
    tx = tx_all()
    cc = closed_cases()

    detector = CommonPointOfCompromise()

    # 1. Run backtest and verify size-stratified LR table is produced and reproducible
    bt = detector.backtest(cc, tx)
    strat_table = detector.get_stratified_lr_table()

    assert set(strat_table.keys()) == {"3", "4-5", "6+"}, "Must produce stratified LR table for cohort sizes 3, 4-5, 6+"
    print("\nSize-Stratified Cohort Backtest LR Table (confirmed-fraud C-burst episodes vs 5,000 legit Jul-Oct card-days):")
    for bucket in ("3", "4-5", "6+"):
        s = strat_table[bucket]
        print(
            f"  Cohort size {bucket:>3}: fraud_hits={s['k_pos']}/{s['n_pos']} ({s['p_pos']:.2%}), "
            f"legit_hits={s['k_neg']}/{s['n_neg']} ({s['p_neg']:.2%}), "
            f"LR={s['lr']:.2f}, 95% CI=[{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
        )
        assert s["lr"] > 0.0 and s["ci_low"] <= s["lr"] <= s["ci_high"]

    n_pos = len(bt.caught) + len(bt.missed)
    n_neg = bt.n_cases - n_pos
    fire_rate_fraud = len(bt.caught) / max(n_pos, 1)
    fire_rate_legit = len(bt.legit_acted_against) / max(n_neg, 1)

    print("\nOverall Backtest Summary:")
    print(f"  Fraud Episodes Tested: {n_pos} | Caught: {len(bt.caught)} ({fire_rate_fraud:.2%})")
    print(f"  Legit Card-Days Tested: {n_neg} | False Accusations: {len(bt.legit_acted_against)} ({fire_rate_legit:.2%})")
    print(f"  Precision: {bt.precision:.2%}")
    print(f"  Recall: {bt.recall:.2%}")
    print(f"  Overall LR Measured: {bt.lr_measured:.2f}")

    assert fire_rate_legit <= 0.02, f"False positive rate on legit sample too high: {fire_rate_legit:.2%}"

    # 2. Run on exam cases HHF-001, HHF-002, HHF-006
    exam_cases = cp[cp.case_id.isin(['HHF-001', 'HHF-002', 'HHF-006'])]
    tx_ids = set(tx.TransactionID)
    size_3_passes = strat_table["3"]["ci_low"] > 2.0

    for row in exam_cases.itertuples():
        opened_at = pd.to_datetime(row.opened_at)
        ctx = CaseCtx(
            case_id=row.case_id,
            flagged_txn_id=int(row.flagged_txn_id),
            card_id=row.card_id,
            customer_id=row.customer_id,
            opened_at=opened_at,
            tx=tx[tx.t <= opened_at],
            cc=cc[pd.to_datetime(cc.closed_at) <= opened_at]
        )

        findings = detector.run(ctx)
        diag = CommonPointOfCompromise.last_diagnostics.get(row.case_id, {})

        if row.case_id == 'HHF-001':
            if size_3_passes:
                assert len(findings) > 0, "Expected finding for HHF-001 when size-3 LR 95% CI lower bound > 2"
                f = findings[0]
                print(f"\nCase {row.case_id} Final Finding (size-3 LR passed):")
                print(f"  Pattern: {f.pattern}")
                print(f"  Claim: {f.claim.what}")
                print(f"  Connected Cards ({len(f.connected_card_ids)}): {f.connected_card_ids}")
                print(f"  Affected Txns: {f.affected_txn_ids}")
                print(f"  LR (size-3 stratified): {f.lr:.2f}")
                assert abs(f.lr - strat_table["3"]["lr"]) < 1e-6
                assert all(tid in tx_ids for tid in f.affected_txn_ids)
                assert all(int(tid) in tx_ids for tid in f.claim.evidence.txn_ids)
            else:
                assert len(findings) == 0
                assert diag.get("status") == "cohort_too_small"
                print(f"\nCase {row.case_id} Honest Record: {diag}")

        elif row.case_id == 'HHF-002':
            assert len(findings) > 0, "Expected finding for HHF-002"
            f = findings[0]
            print(f"\nCase {row.case_id} Final Finding:")
            print(f"  Pattern: {f.pattern}")
            print(f"  Claim: {f.claim.what}")
            print(f"  Connected Cards ({len(f.connected_card_ids)}): {f.connected_card_ids}")
            print(f"  Affected Txns: {f.affected_txn_ids}")
            print(f"  LR (stratified): {f.lr:.2f}")
            assert all(tid in tx_ids for tid in f.affected_txn_ids)
            assert all(int(tid) in tx_ids for tid in f.claim.evidence.txn_ids)

        else:
            print(f"\nCase {row.case_id} (informational print only): {len(findings)} findings | diag={diag.get('status')}")
            if findings:
                f = findings[0]
                print(f"  Pattern: {f.pattern}")
                print(f"  Claim: {f.claim.what}")
                assert all(tid in tx_ids for tid in f.affected_txn_ids)
