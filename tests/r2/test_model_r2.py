"""Tests for Block B1 -- Retrained transaction fraud model for January 2017 & Drift / OOD API."""

from __future__ import annotations

import pandas as pd

from assay.r2.frames import case_pack, tx_all
from assay.r2.model_r2 import (
    DRIFT_REPORT_PATH,
    REPORT_R2_JSON,
    REPORT_R2_MD,
    SCORES_B1_PATH,
    attest_txn_score,
    drift_report,
    score,
    score_txn,
    train_r2,
)


def test_train_and_score_january():
    report = train_r2(rounds=400, num_threads=8)

    assert SCORES_B1_PATH.exists(), "scores_b1.parquet was not written"
    assert DRIFT_REPORT_PATH.exists(), "drift_b1_report.json was not written"
    assert REPORT_R2_JSON.exists(), "model_b1_report.json was not written"
    assert REPORT_R2_MD.exists(), "docs/r2/B1_model.md was not written"

    scores = pd.read_parquet(SCORES_B1_PATH)
    assert list(scores.columns) == ["TransactionID", "p_fraud"]
    assert not scores["p_fraud"].isna().any(), "NaN probabilities found in scores_b1.parquet"
    assert (scores["p_fraud"] >= 0.0).all() and (scores["p_fraud"] <= 1.0).all(), "p_fraud out of [0, 1]"

    # 1. Verify label completeness analysis: Jul-Oct complete, Nov-Dec partial (31 Nov, 14 Dec cases)
    m_rates = {r["month"]: r for r in report["monthly_label_rates"]}
    for m in ("2016-07", "2016-08", "2016-09", "2016-10"):
        assert m_rates[m]["complete_labels"] is True
        assert m_rates[m]["fraud_rate"] > 0.02
    for m in ("2016-11", "2016-12"):
        assert m_rates[m]["complete_labels"] is False
        assert m_rates[m]["named_rate"] < 0.005
    assert m_rates["2016-11"]["n_cases_opened"] == 31
    assert m_rates["2016-12"]["n_cases_opened"] == 14

    # 2. Verify OOT evaluation on latest fully-labelled month (October 2016):
    # OOT AUC > bank risk_score AUC, and 10-bin calibration ECE < 0.05
    oct_holdout = report["october_holdout"]
    oct_auc = oct_holdout["without_risk_score"]["auc"]
    bank_auc = oct_holdout["bank_risk_score_only"]["auc"]
    ece = oct_holdout["ece_10bin"]
    assert 0.90 < oct_auc < 0.995, f"Suspicious October holdout AUC: {oct_auc}"
    assert oct_auc > bank_auc, f"OOT AUC {oct_auc} did not beat bank risk_score AUC {bank_auc}"
    assert ece < 0.05, f"Calibration ECE {ece} >= 0.05"
    assert len(report["calibration_10bins"]) == 10

    # Verify no single feature has leak-level AUC (> 0.99) on October holdout
    print("\n--- TOP 25 FEATURES BY GAIN, OCT SINGLE AUC & JAN PSI ---")
    dr = drift_report()
    assert len(dr["features"]) == 25
    for item in dr["features"]:
        print(
            f"  {item['rank']:2d}. {item['feature']:<24s} gain={item['gain']:10.1f}  "
            f"oct_single_auc={item['oct_single_auc']:.4f}  psi={item['psi']:.4f}  flagged={item['flagged']}"
        )
        assert item["oct_single_auc"] < 0.99, f"Feature {item['feature']} leaked with AUC {item['oct_single_auc']}"
        assert (item["psi"] > 0.25) == item["flagged"]

    # 3. Verify `score(txn_ids) -> DataFrame[txn_id, p, in_distribution: bool]` API
    tx = tx_all()
    jan_tids = tx.loc[tx["month_src"] == "jan", "TransactionID"].astype(int).tolist()
    jan_scored_df = score(jan_tids)
    assert list(jan_scored_df.columns) == ["txn_id", "p", "in_distribution"]
    assert len(jan_scored_df) == len(jan_tids)
    assert jan_scored_df["p"].between(0.0, 1.0).all()
    assert jan_scored_df["in_distribution"].dtype == bool

    # 4. Verify all 18 exam case-pack transactions are scored and printed (no arbitrary threshold on them)
    cp = case_pack()
    assert len(cp) == 18
    cp_tids = cp["flagged_txn_id"].astype(int).tolist()
    cp_df = score(cp_tids)
    assert len(cp_df) == 18
    assert not cp_df["p"].isna().any()

    for r in cp.itertuples():
        tid = int(r.flagged_txn_id)
        p = score_txn(tid)
        assert 0.0 <= p <= 1.0
        att = attest_txn_score(tid, card_id=str(r.card_id), customer_id=str(r.customer_id))
        d = att.to_dict()
        assert str(tid) in d["evidence"]["txn_ids"]
        assert d["how"]["query"] == "assay.r2.model_r2.score"

    # Print full summary for lead inspection
    print("\n--- 1. MONTHLY CLOSED CASES & LABEL COMPLETENESS ---")
    for r in report["monthly_label_rates"]:
        print(
            f"  {r['month']}: cases_opened={r['n_cases_opened']:4d} "
            f"(fraud={r['n_cases_fraud']:4d}, cleared={r['n_cases_cleared']:3d}) | "
            f"txns={r['n_txns']:7d}, named={r['n_named_txns']:4d} ({r['named_rate']:.4%}), "
            f"fraud_txns={r['n_fraud_txns']:4d} ({r['fraud_rate']:.4%}) | complete={r['complete_labels']}"
        )

    print("\n--- 2. OCTOBER 2016 OOT HOLDOUT & 10-BIN CALIBRATION ---")
    print("Oct Holdout (without risk_score):", oct_holdout["without_risk_score"])
    print("Oct Holdout (with risk_score):   ", oct_holdout["with_risk_score"])
    print("Oct Holdout (bank risk_score):   ", oct_holdout["bank_risk_score_only"])
    print(f"Oct 10-Bin ECE: {ece:.5f}")
    for b in report["calibration_10bins"]:
        print(f"  {b['bin']:<18s} n={b['n']:6d}  pred={b['mean_predicted']:.5f}  obs={b['observed_rate']:.5f}  gap={b['abs_gap']:.5f}")

    print("\n--- 3. DRIFT REPORT SUMMARY ---")
    print("Flagged top-25 features (PSI > 0.25):", dr["flagged_features"])
    print(
        f"In-distribution rate: train(Jul-Oct)={dr['train_in_distribution_rate']:.4%}, "
        f"jan={dr['jan_in_distribution_rate']:.4%}"
    )
    print("Jan Comparison (B1 vs R1 model):     ", report["r1_vs_b1_january_comparison"])

    print("\n--- 4. CASE PACK 18 FLAGGED TXN SCORES ---")
    for item in report["case_pack_scores"]:
        print(
            f"  {item['case_id']} ({item['trigger_type']:<15s}): txn={item['flagged_txn_id']} "
            f"bank_rs={item['bank_risk_score']:.2f} | R1_p={item['r1_p_fraud']} | "
            f"B1_p={item['p_fraud']:.4f} (raw={item['p_raw']:.4f}) | "
            f"in_dist={str(item['in_distribution']):<5s} (d2={item['mahalanobis_d2']:.2f})"
        )
