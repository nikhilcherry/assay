"""Test suite for Block B6:
- FanIn ('undocumented:recipient_fan_in'): R_emaildomain burst only (no device_profile browser
  rollouts), 72h episode linkage check (context note if outside episode).
- MicroCharge ('undocumented:slow_drip_micro_charges'): 30-day cohort burst test on novel
  sub-$6.50 ProductCD 'C' slow drips."""

from __future__ import annotations

import pandas as pd
import pytest

from assay.r2.detectors.fan_in import FanIn, rerun_recipient_fan_in_burst
from assay.r2.detectors.micro_charge import MicroCharge, rerun_slow_drip_cohort_burst
from assay.r2.finding import CaseCtx, REGISTRY
from assay.r2.frames import case_pack, closed_cases, tx_all


def test_registry_registration():
    assert "MicroCharge" in REGISTRY
    assert "FanIn" in REGISTRY


def test_micro_charge_backtest():
    tx = tx_all()
    cc = closed_cases()
    mc = MicroCharge()
    bt = mc.backtest(cc, tx, sample_size=5000, seed=42)

    print("\n=== MicroCharge (slow_drip_micro_charges Cohort Burst) Backtest Table ===")
    print(f"Total Evaluated (Confirmed Fraud + Legit Sample): {bt.n_cases}")
    print(f"Confirmed Fraud Caught (Jul-Dec): {len(bt.caught)} / {len(bt.caught) + len(bt.missed)} ({mc._p_fraud:.4%})")
    print(f"Legit Sample Acted Against (Jul-Oct): {len(bt.legit_acted_against)} / 5000 ({mc._p_legit:.4%})")
    print(f"Empirical Laplace LR: {bt.lr_measured:.4f} (95% CI: [{mc._lr_ci[0]:.4f}, {mc._lr_ci[1]:.4f}])")
    print(f"Active Finding LR: {mc._measured_lr:.2f} (is_assumption={mc._lr_is_assumption})")
    print("Historical 30-day Window Novel Drip Counts (Jul-Dec):")
    for w_start, w_end, cnt in mc._hist_window_counts:
        print(f"  Window {w_start} .. {w_end}: {cnt} novel slow-drip cards")

    assert mc._p_legit <= 0.01


def test_fan_in_backtest():
    tx = tx_all()
    cc = closed_cases()
    fi = FanIn()
    bt = fi.backtest(cc, tx, sample_size=5000, seed=42)

    print("\n=== FanIn (recipient_fan_in R_emaildomain Burst) Backtest Table ===")
    print(f"Total Evaluated (Confirmed Fraud + Legit Sample): {bt.n_cases}")
    print(f"Confirmed Fraud Caught (Jul-Dec): {len(bt.caught)} / {len(bt.caught) + len(bt.missed)} ({fi._p_fraud:.4%})")
    print(f"Caught Case IDs: {bt.caught}")
    print(f"Legit Sample Acted Against (Jul-Oct): {len(bt.legit_acted_against)} / 5000 ({fi._p_legit:.4%})")
    print(f"Empirical Laplace LR: {bt.lr_measured:.4f} (95% CI: [{fi._lr_ci[0]:.4f}, {fi._lr_ci[1]:.4f}])")
    print(f"Active Finding LR: {getattr(fi, '_measured_lr', 1.0):.2f} (measured, never assumed)")

    assert fi._p_legit <= 0.02


def test_exam_cases_and_reproducibility():
    tx = tx_all()
    cp = case_pack()
    cc = closed_cases()

    mc = MicroCharge()
    fi = FanIn()

    mc.backtest(cc, tx, sample_size=5000, seed=42)
    fi.backtest(cc, tx, sample_size=5000, seed=42)

    print("\n=== ALL 18 EXAM CASES DIAGNOSTIC TABLE ===")
    header = (
        f"{'Case':<8} | {'FI Kind':<9} | {'FI Linking Value':<24} | {'k_now':<5} | "
        f"{'Base(m/M)':<10} | {'FI p-val':<9} || {'MC Fired':<8} | {'MC chg/span':<11} | "
        f"{'Cohort k':<8} | {'MC Base(m/M)':<12} | {'MC p-val':<9}"
    )
    print(header)
    print("-" * len(header))

    exam_out = {}

    for row in cp.itertuples():
        case_id = str(row.case_id)
        flagged_id = int(row.flagged_txn_id)
        opened_at = pd.to_datetime(row.opened_at)
        card_id = str(row.card_id)
        cust_id = str(row.customer_id)

        tx_asof = tx[tx.t <= opened_at]
        cc_asof = cc[pd.to_datetime(cc.closed_at) <= opened_at]

        ctx = CaseCtx(
            case_id=case_id,
            flagged_txn_id=flagged_id,
            card_id=card_id,
            customer_id=cust_id,
            opened_at=opened_at,
            tx=tx_asof,
            cc=cc_asof,
        )

        fi_info = fi.inspect_case(ctx)
        mc_info = mc.inspect_case(ctx)

        fi_f = fi.run(ctx)
        mc_f = mc.run(ctx)

        exam_out[case_id] = {
            "fi_info": fi_info,
            "mc_info": mc_info,
            "fi_findings": fi_f,
            "mc_findings": mc_f,
        }

        fi_link = fi_info["linking_value"][:24]
        fi_base = f"{fi_info['baseline_mean']:.2f}/{fi_info['baseline_max']}"
        fi_p = f"{fi_info['p_value']:.1e}" if fi_info["k_now"] > 0 else "1.0e+00"

        mc_cs = f"{mc_info['card_n_charges']}/{mc_info['card_span_days']:.1f}d"
        mc_base = f"{mc_info['baseline_mean']:.2f}/{mc_info['baseline_max']}"
        mc_p = f"{mc_info['p_value']:.1e}" if mc_info["k_now"] > 0 else "1.0e+00"

        print(
            f"{case_id:<8} | {fi_info['kind']:<9} | {fi_link:<24} | {fi_info['k_now']:<5} | "
            f"{fi_base:<10} | {fi_p:<9} || {str(bool(mc_f)):<8} | {mc_cs:<11} | "
            f"{mc_info['k_now']:<8} | {mc_base:<12} | {mc_p:<9}"
        )

    print("\n--- Detailed Findings & Context Notes on Exam Cases ---")
    for cid, data in exam_out.items():
        for f in data["fi_findings"]:
            kind = getattr(f, "kind", "evidence")
            print(f"[FanIn:{kind.upper()}] {cid} (LR={f.lr}): {f.claim.what}")
            # Reproducibility check via registered query function
            p = f.claim.params
            re_info = rerun_recipient_fan_in_burst(
                tx, p["card_id"], p["flagged_txn_id"], p["opened_at"], p["window_days"]
            )
            assert re_info["k_now"] == p["k_now"]
            assert re_info["kind"] == p["kind"]

        for f in data["mc_findings"]:
            print(f"[MicroCharge:EVIDENCE] {cid} (LR={f.lr}): {f.claim.what}")
            p = f.claim.params
            re_info = rerun_slow_drip_cohort_burst(
                tx,
                p["card_id"],
                p["flagged_txn_id"],
                p["opened_at"],
                p["max_amt"],
                p["min_charges"],
                p["min_span_days"],
                p["window_days"],
            )
            assert re_info["k_now"] == p["k_now"]
            assert re_info["fired"] is True

    # Assertions required by lead review:
    # 1. HHF-006 fires FanIn as primary evidence
    assert exam_out["HHF-006"]["fi_info"]["kind"] == "evidence"
    assert len(exam_out["HHF-006"]["fi_findings"]) == 1
    # the cohort form never fires on history, so it weighs nothing and only carries its linked cards (C3)
    assert exam_out["HHF-006"]["fi_findings"][0].lr == 1.0
    assert len(exam_out["HHF-006"]["fi_findings"][0].connected_card_ids) >= 1

    # 2. HHF-013 yields only a context note (lr=1.0, kind='context', affected_txn_ids=[])
    assert exam_out["HHF-013"]["fi_info"]["kind"] == "context"
    assert len(exam_out["HHF-013"]["fi_findings"]) == 1
    assert getattr(exam_out["HHF-013"]["fi_findings"][0], "kind", None) == "context"
    assert exam_out["HHF-013"]["fi_findings"][0].lr == 1.0
    assert exam_out["HHF-013"]["fi_findings"][0].affected_txn_ids == []

    # 3. HHF-014 and HHF-016 no longer fire on device_profile bursts
    assert exam_out["HHF-014"]["fi_findings"] == []
    assert exam_out["HHF-016"]["fi_findings"] == []

    # 4. Measured negative: Jul-Dec already sees 4-24 novel slow-drip starts per 30-day
    #    window, so January's 9-13 is not a burst. MicroCharge must stay silent, not invent one.
    assert exam_out["HHF-004"]["mc_findings"] == []
    assert exam_out["HHF-005"]["mc_findings"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
