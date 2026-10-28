"""Tests for Dormant Card Wake-Up Detector (Block B5).
Runs parameter sweep over 5,585 closed cases and 5,000 sampled Jul-Oct legit txns.
Selects optimal N, k, m, computes 95% Bootstrap CI for LR, tests rerun reproducibility,
evaluates on the 18 exam cases, and lists all January transactions for C11663-K1 (HHF-003).
"""

from __future__ import annotations

import pandas as pd
import pytest

from assay.r2.frames import tx_all, closed_cases, case_pack
from assay.r2.finding import CaseCtx
from assay.r2.detectors.dormant import DormantWake, sweep_backtest, rerun, N_SWEEP, K_SWEEP, M_SWEEP


def test_dormant_wake_sweep_and_case_pack():
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    print("\n--- B5 DORMANT CARD WAKE-UP DETECTOR SWEEP ---")
    results = sweep_backtest(cc, tx, N_grid=N_SWEEP, k_grid=K_SWEEP, m_grid=M_SWEEP, seed=42)

    # Print sweep table
    print(f"{'N':>4} | {'k':>3} | {'m':>5} | {'Caught':>6} | {'Fraud%':>7} | {'LegitFired':>10} | {'Legit%':>7} | {'Score':>6} | {'LR (95% CI)':>22}")
    print("-" * 92)
    for r in results:
        ci_str = f"{r['lr_measured']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{r['N']:4d} | {r['k']:3d} | {r['m']:5.1f} | {r['caught']:6d} | {r['fraud_rate']*100:6.2f}% | {r['legit_sample_fired']:10d} | {r['legit_rate']*100:6.2f}% | {r['score']:6d} | {ci_str:>22}")

    # Select optimal setting under legit_rate <= 2% constraint
    valid_res = [r for r in results if r["legit_rate"] <= 0.02]
    if valid_res:
        best_res = max(valid_res, key=lambda x: (x["score"], x["lr_measured"], x["caught"]))
    else:
        best_res = min(results, key=lambda x: x["legit_rate"])

    N_opt, k_opt, m_opt = best_res["N"], best_res["k"], best_res["m"]
    print(f"\nCHOSEN OPTIMAL SETTING: N={N_opt}, k={k_opt}, m={m_opt}")
    print(f"Counts: caught={best_res['caught']} / 4,675 fraud ({best_res['fraud_rate']*100:.2f}%), "
          f"legit_sample_fired={best_res['legit_sample_fired']} / 5,000 legit ({best_res['legit_rate']*100:.2f}%)")
    print(f"Measured LR = {best_res['lr_measured']} (95% CI: [{best_res['ci_low']}, {best_res['ci_high']}])")

    # Instantiate detector with chosen setting
    detector = DormantWake(N=N_opt, k=k_opt, m=m_opt, lr_measured=best_res["lr_measured"])

    # Test on 18 exam cases from case_pack
    print("\n--- EVALUATION ON 18 EXAM CASES (case_pack_final.csv) ---")
    tx_by_id = tx.set_index("TransactionID")
    fired_cases = []

    for row in cp.itertuples():
        flagged_id = int(row.flagged_txn_id)
        if flagged_id in tx_by_id.index:
            actual_card = tx_by_id.at[flagged_id, "card_id"]
            actual_cust = tx_by_id.at[flagged_id, "customer_id"]
        else:
            actual_card = row.card_id
            actual_cust = row.customer_id

        opened_at = pd.to_datetime(row.opened_at)
        tx_as_of = tx[tx["t"] <= opened_at]
        cc_as_of = cc[pd.to_datetime(cc["closed_at"]) <= opened_at]

        ctx = CaseCtx(
            case_id=row.case_id,
            flagged_txn_id=flagged_id,
            card_id=actual_card,
            customer_id=actual_cust,
            opened_at=opened_at,
            tx=tx_as_of,
            cc=cc_as_of
        )

        findings = detector.run(ctx)

        if findings:
            fired_cases.append((row.case_id, findings))
            for f in findings:
                print(f"Case {row.case_id} FIRED: {f.claim.what}")
                print(f"  Pattern: {f.pattern}, LR: {f.lr}")
                print(f"  Evidence txn_ids: {f.affected_txn_ids}")

                # Assert query name and params structure
                assert f.claim.query == "dormant_wake"
                assert "card_id" in f.claim.params
                assert "as_of" in f.claim.params

                # Test rerun(params) reproducibility
                reproduced_ids = rerun(f.claim.params)
                assert reproduced_ids == f.affected_txn_ids, f"rerun failed for case {row.case_id}: {reproduced_ids} != {f.affected_txn_ids}"

                for tid in f.affected_txn_ids:
                    assert tid in tx_by_id.index, f"TransactionID {tid} from evidence not found in tx_all!"
        else:
            print(f"Case {row.case_id} DID NOT FIRE")

    print(f"\nTotal exam cases fired: {len(fired_cases)} / {len(cp)}")
    assert best_res["legit_rate"] <= 0.02, "Chosen setting must satisfy legit fire rate <= 2%"

    # Inspect HHF-003 January activity on C11663-K1
    print("\n--- HHF-003 / CARD C11663-K1 JANUARY TRANSACTION ANALYSIS ---")
    c11663_jan = tx[(tx["card_id"] == "C11663-K1") & (tx["t"] >= "2017-01-01")].sort_values("t")
    print(f"Total January transactions on C11663-K1: {len(c11663_jan)}")
    print(f"{'TxnID':>10} | {'Timestamp':>19} | {'Amount ($)':>10} | {'Product':>7} | {'Channel':>9} | {'Email Domain':>15}")
    print("-" * 82)
    for r in c11663_jan.itertuples():
        p_domain = str(r.P_emaildomain) if hasattr(r, "P_emaildomain") and pd.notna(r.P_emaildomain) else "N/A"
        print(f"{r.TransactionID:10d} | {str(r.t):19s} | {r.amount:10.2f} | {r.ProductCD:7s} | {r.channel:9s} | {p_domain:>15s}")

    # Compute exposure_usd for HHF-003
    hhf3_finding = next(f for cid, fs in fired_cases if cid == "HHF-003" for f in fs)
    burst_txids = set(hhf3_finding.affected_txn_ids)
    
    # Check for same-shape later txns (ProductCD='W', online/C11663-K1)
    first_burst_t = c11663_jan[c11663_jan["TransactionID"].isin(burst_txids)]["t"].min()
    episode_txns = c11663_jan[(c11663_jan["t"] >= first_burst_t)]
    exposure_usd = float(episode_txns["amount"].sum())
    print(f"\nHHF-003 Episode Exposure USD: ${exposure_usd:.2f} (across {len(episode_txns)} transactions)")
    assert exposure_usd >= sum(c11663_jan[c11663_jan["TransactionID"].isin(burst_txids)]["amount"])


if __name__ == "__main__":
    test_dormant_wake_sweep_and_case_pack()
