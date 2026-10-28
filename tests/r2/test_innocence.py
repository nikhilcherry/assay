"""Tests and measurements for B29 -- InnocenceEvidence detector."""

from __future__ import annotations

import pandas as pd

from assay.r2.detectors.innocence import (
    KEEP_CI_HIGH_MAX,
    KINDS,
    InnocenceEvidence,
    calibrated_innocence_stats,
    combined_lr,
    correlated_groups,
    sample_cleared_diagnostics,
)
from assay.r2.finding import REGISTRY, CaseCtx
from assay.r2.frames import case_pack, closed_cases, tx_all


def test_cleared_cases_diagnostics_and_calibration():
    assert "innocence" in REGISTRY

    print("\n=== DIAGNOSTICS FOR 5 RANDOM CLEARED CLOSED CASES (JUL-OCT 2016) ===")
    diags = sample_cleared_diagnostics(n=5, seed=42)
    assert len(diags) == 5
    for d in diags:
        print(
            f"Case {d['case_id']} | card={d['card_id']} | opened_at={d['opened_at']} | "
            f"tx_before_opened={d['tx_rows_before_opened_at']} | "
            f"tx_strictly_before_flagged={d['tx_rows_strictly_before_flagged_txn']} | "
            f"kinds_fired={d['kinds_fired']}"
        )
        print(f"    inputs={d['inputs']}")

    stats, cooc, alert_df = calibrated_innocence_stats()

    n_fraud = int((alert_df["outcome"] == "confirmed_fraud").sum())
    n_cleared = int((alert_df["outcome"] == "cleared").sum())
    n_legit = stats[KINDS[0]].legit_total
    assert n_fraud > 0 and n_cleared > 0 and n_legit >= 5000

    print("\n=== B29 JUL-OCT AS-OF BACKTEST: (A) ALERTS (FRAUD VS CLEARED) & (B) FRAUD VS LEGIT CARD-DAYS ===")
    print(f"Populations: confirmed_fraud={n_fraud}, cleared_alerts={n_cleared}, legit_card_days={n_legit}")
    header = (
        f"{'kind':<18} {'fraud_hit':>11} {'cleared_hit':>11} {'legit_hit':>11} "
        f"{'P(k|F)':>7} {'P(k|C)':>7} {'P(k|L)':>7} "
        f"{'LR_alert':>9} {'95% CI (alert)':>18} {'LR_legit':>9} {'95% CI (legit)':>18} {'kept':>6}"
    )
    print(header)
    total_cleared_hits = 0
    for k in KINDS:
        st = stats[k]
        total_cleared_hits += st.cleared_hits
        f_str = f"{st.fraud_hits}/{st.fraud_total}"
        c_str = f"{st.cleared_hits}/{st.cleared_total}"
        l_str = f"{st.legit_hits}/{st.legit_total}"
        ci_a = f"[{st.ci_low_alert:.4f}, {st.ci_high_alert:.4f}]"
        ci_l = f"[{st.ci_low_legit:.4f}, {st.ci_high_legit:.4f}]"
        print(
            f"{k:<18} {f_str:>11} {c_str:>11} {l_str:>11} "
            f"{st.p_fraud:>7.4f} {st.p_cleared:>7.4f} {st.p_legit:>7.4f} "
            f"{st.lr_alert:>9.4f} {ci_a:>18} {st.lr_legit:>9.4f} {ci_l:>18} {str(st.kept):>6}"
        )
        if st.kept:
            assert st.ci_high_alert < KEEP_CI_HIGH_MAX
        else:
            assert st.ci_high_alert >= KEEP_CI_HIGH_MAX

    # Verify cleared cases have non-zero hits now that the Round 2 over-filtering bug is fixed
    assert total_cleared_hits > 0, "Cleared cases must have non-zero hits across baseline kinds"

    kept_kinds = [k for k in KINDS if stats[k].kept]
    print(f"\nKept exculpatory kinds (alert LR upper 95% CI < {KEEP_CI_HIGH_MAX}): {kept_kinds}")

    print("\n=== PAIRWISE JACCARD CO-OCCURRENCE AMONG LEGITIMATE CASES ===")
    print(cooc.to_string())
    groups = correlated_groups()
    print(f"\nCorrelated groups: {[sorted(g) for g in groups]}")


def test_combined_lr_and_exam_cases():
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()
    det = InnocenceEvidence()

    print("\n=== 18 EXAM CASES: EXCULPATORY FINDINGS (KEPT vs ALL RAW KINDS) ===")
    for r in cp.itertuples():
        opened_at = pd.Timestamp(r.opened_at)
        tx_asof = tx[tx["t"] <= opened_at]
        cc_asof = cc[pd.to_datetime(cc["closed_at"]) <= opened_at]
        ctx = CaseCtx(
            case_id=str(r.case_id),
            flagged_txn_id=int(r.flagged_txn_id),
            card_id=str(r.card_id),
            customer_id=str(r.customer_id),
            opened_at=opened_at,
            tx=tx_asof,
            cc=cc_asof,
        )
        kept_findings = det.run(ctx)
        all_findings = det.run_all_kinds(ctx)

        for f in all_findings:
            assert f.pattern.startswith("exculpatory:")
            d = f.claim.to_dict()
            assert int(r.flagged_txn_id) in [int(x) for x in d["evidence"]["txn_ids"]]
            assert d["how"]["query"].startswith("innocence.")
            if f.pattern == "exculpatory:recurring":
                assert "R7" in f.claim.what or d["how"]["params"].get("policy_rule") == "R7"

        comb_kept = combined_lr(kept_findings)
        comb_all = combined_lr(all_findings)
        assert comb_kept <= 1.0
        assert comb_all <= 1.0

        kept_str = [f"{f.pattern.split(':')[1]}(lr={f.lr:.4f})" for f in kept_findings]
        all_str = [f"{f.pattern.split(':')[1]}(lr={f.lr:.4f})" for f in all_findings]
        print(
            f"{r.case_id} | card={r.card_id:<10} | txn={r.flagged_txn_id} | "
            f"kept={kept_str} (comb_lr={comb_kept:.4f}) | raw_kinds={all_str} (raw_comb_lr={comb_all:.4f})"
        )
        for f in all_findings:
            print(f"    -> {f.pattern}: {f.claim.what}")


def test_detector_backtest_method():
    tx = tx_all()
    cc = closed_cases()
    cc_juloct = cc[pd.to_datetime(cc["opened_at"]) < pd.Timestamp("2016-11-01")].copy()
    det = InnocenceEvidence()
    bt = det.backtest(cc_juloct, tx)
    print(
        f"\nOverall Detector Backtest (any kept exculpatory kind firing on Jul-Oct closed cases): "
        f"n_cases={bt.n_cases}, fraud_hit={len(bt.caught)}, "
        f"cleared_hit={len(bt.legit_acted_against)}, lr_measured={bt.lr_measured:.4f}"
    )
    assert bt.n_cases == len(cc_juloct)
