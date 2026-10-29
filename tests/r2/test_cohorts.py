"""Tests for Block C3: Cohort detectors MicroDripCohort (HHF-004/005) and FanIn (HHF-006/013).

Asserts only what we control:
- Reproducibility / determinism (two runs identical)
- As-of correctness (no transaction with t > opened_at in any finding)
- Every evidence ID exists in tx_all() / closed_cases() / case_pack()
- Schema / shape and Attestation fields
- Pure function `cohort_members(tx_asof, flagged_txn_id, opened_at)` contract
- Neither detector uses TransactionID ranges/thresholds as a feature
- Prints measured tables over all 18 exam cases, monthly distributions, amount floor analysis, and backtests.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pandas as pd

import assay.r2.detectors.fan_in as fan_in_mod
import assay.r2.detectors.micro_charge as micro_mod
from assay.r2.detectors.fan_in import (
    FanIn,
    analyze_amount_floor_distribution,
    cohort_members as fan_in_cohort_members,
    rerun_recipient_fan_in_burst,
)
from assay.r2.detectors.micro_charge import (
    MicroCharge,
    MicroDripCohort,
    cohort_members as micro_cohort_members,
    monthly_new_drip_distribution,
    rerun_micro_drip_cohort,
)
from assay.r2.finding import REGISTRY, CaseCtx
from assay.r2.frames import case_pack, closed_cases, tx_all


def _build_ctx(row: pd.Series, tx: pd.DataFrame, cc: pd.DataFrame) -> CaseCtx:
    t0 = pd.Timestamp(row.opened_at)
    fid = int(row.flagged_txn_id)
    hit = tx.loc[tx.TransactionID == fid, "card_id"]
    card = str(hit.iat[0]) if not hit.empty else str(row.card_id)
    return CaseCtx(
        case_id=str(row.case_id),
        flagged_txn_id=fid,
        card_id=card,
        customer_id=str(row.customer_id),
        opened_at=t0,
        tx=tx[tx.t <= t0],
        cc=cc[pd.to_datetime(cc.closed_at) <= t0],
    )


def test_no_transaction_id_leakage_in_source():
    """Verify neither detector module uses TransactionID comparisons/ranges (e.g. >= 3660000) as a signal."""
    forbidden_patterns = [
        r"TransactionID\s*>",
        r"TransactionID\s*<",
        r"TransactionID\s*>=",
        r"TransactionID\s*<=",
        r">=\s*36",
        r">\s*36",
        r"3660000",
    ]
    for mod in (micro_mod, fan_in_mod):
        src = inspect.getsource(mod)
        for pat in forbidden_patterns:
            matches = re.findall(pat, src)
            assert not matches, f"Forbidden TransactionID threshold pattern {pat!r} found in {mod.__name__}"


def test_cohorts_backtest_and_exam_pack():
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    all_txn_ids = set(tx.TransactionID.astype(int).tolist())
    txn_time_lookup = dict(zip(tx.TransactionID.astype(int), tx.t))

    # 1. Print monthly new micro-drip distribution (justifying K=6 other cards)
    dist = monthly_new_drip_distribution(tx)
    print("\n=== MONTHLY NEW MICRO-DRIP DISTRIBUTION (Jul 2016 - Jan 2017) ===")
    print(f"{'Month':<10} | {'By Start Month':>14} | {'By Qual Month':>13} | {'Qualifying Cards'}")
    print("-" * 80)
    for m, d in dist.items():
        assert isinstance(d["by_start_month"], int)
        assert isinstance(d["by_qual_month"], int)
        cards_preview = ", ".join(d["cards_by_qual"][:6]) + ("..." if len(d["cards_by_qual"]) > 6 else "")
        print(f"{m:<10} | {d['by_start_month']:>14} | {d['by_qual_month']:>13} | {cards_preview}")

    # 2. Print FanIn amount floor distribution analysis
    floor_info = analyze_amount_floor_distribution(tx, cc)
    print("\n=== FAN-IN AMOUNT FLOOR ANALYSIS (Jul-Dec Historical Online ProductCD=='C' Non-Freemail) ===")
    print(f"{'Amount Band':<24} | {'Txns':>8} | {'Cards':>7} | {'Fraud Txns':>10} | {'Cleared/Uncased':>15} | {'Fraud Rate':>10}")
    print("-" * 88)
    for b in floor_info["jul_dec_bands"]:
        print(
            f"{b['band']:<24} | {b['n_txns_jul_dec']:>8} | {b['n_cards_jul_dec']:>7} | "
            f"{b['confirmed_fraud_txns']:>10} | {b['cleared_or_uncased_txns']:>15} | {b['fraud_rate']:>10.4f}"
        )
    print("Jan 2017 sc.rr.com < $100 :", floor_info["jan_sc_rr_lt_100"])
    print("Jan 2017 sc.rr.com >= $100:", floor_info["jan_sc_rr_ge_100"])

    # 3. Run backtests for MicroDripCohort and FanIn (and verify legacy MicroCharge still works)
    det_drip = MicroDripCohort()
    det_fan = FanIn()
    det_legacy = MicroCharge()

    assert REGISTRY.get("micro_drip_cohort") is MicroDripCohort
    assert REGISTRY.get("FanIn") is FanIn
    assert REGISTRY.get("MicroCharge") is MicroCharge

    bt_drip = det_drip.backtest(cc, tx)
    bt_fan = det_fan.backtest(cc, tx)
    bt_legacy = det_legacy.backtest(cc, tx)

    print("\n=== BACKTEST SUMMARY TABLE (Jul-Dec Closed Cases + 5,000 Never-Cased Legit Card-Days) ===")
    for name, bt, stats in [
        ("micro_drip_cohort", bt_drip, det_drip.backtest_stats),
        ("FanIn", bt_fan, det_fan.backtest_stats),
    ]:
        assert isinstance(bt.n_cases, int) and bt.n_cases > 0
        assert isinstance(bt.lr_measured, float)
        print(f"Detector: {name}")
        print(f"  Backtest cases evaluated : {bt.n_cases}")
        print(f"  Caught fraud cases       : {len(bt.caught)} ({bt.caught[:5]})")
        print(f"  Cleared cases acted on   : {len(bt.legit_acted_against)} ({bt.legit_acted_against[:5]})")
        print(f"  Detailed stats           : {stats}")

    # Verify FanIn never uses lr=3.0 assumption
    assert not getattr(det_fan, "_lr_is_assumption", False)
    assert det_fan.lr != 3.0

    # 4. Run both detectors over all 18 case_pack() cases twice (check determinism, as-of correctness, evidence IDs)
    print("\n=== 18 EXAM CASES: MicroDripCohort & FanIn RESULTS ===")
    header = (
        f"{'Case':<8} | {'Detector':<18} | {'Fired?':<10} | {'LR':>6} | "
        f"{'Cohort':>6} | {'AffTxns':>7} | {'Connected Cards'}"
    )
    print(header)
    print("-" * len(header))

    for row in cp.itertuples():
        s_row = pd.Series(row._asdict())
        ctx = _build_ctx(s_row, tx, cc)
        t_open = ctx.opened_at

        # Also test future-contaminated tx frame vs as-of tx frame to verify internal as-of filtering
        ctx_full = CaseCtx(
            case_id=ctx.case_id,
            flagged_txn_id=ctx.flagged_txn_id,
            card_id=ctx.card_id,
            customer_id=ctx.customer_id,
            opened_at=t_open,
            tx=tx,
            cc=ctx.cc,
        )

        for det, pure_fn, rerun_fn in [
            (det_drip, micro_cohort_members, rerun_micro_drip_cohort),
            (det_fan, fan_in_cohort_members, rerun_recipient_fan_in_burst),
        ]:
            res1 = det.run(ctx)
            res2 = det.run(ctx)
            res_full = det.run(ctx_full)

            # Determinism & as-of invariance
            assert [f.claim.to_dict()["what"] for f in res1] == [f.claim.to_dict()["what"] for f in res2]
            assert [f.claim.to_dict()["evidence"] for f in res1] == [f.claim.to_dict()["evidence"] for f in res2]
            assert [f.claim.to_dict()["evidence"] for f in res1] == [f.claim.to_dict()["evidence"] for f in res_full]
            assert [f.connected_card_ids for f in res1] == [f.connected_card_ids for f in res_full]
            assert [f.affected_txn_ids for f in res1] == [f.affected_txn_ids for f in res_full]

            # Pure function cohort_members contract check
            m_cards, per_card_tx, params = pure_fn(ctx.tx, ctx.flagged_txn_id, t_open)
            assert isinstance(m_cards, list)
            assert isinstance(per_card_tx, dict)
            assert isinstance(params, dict)
            for c_id, t_ids in per_card_tx.items():
                assert c_id in m_cards
                for tid in t_ids:
                    assert int(tid) in all_txn_ids
                    assert txn_time_lookup[int(tid)] <= t_open, f"Future txn {tid} in cohort_members for {ctx.case_id}"

            if not res1:
                print(f"{ctx.case_id:<8} | {det.name:<18} | {'False':<10} | {'-':>6} | {0:>6} | {0:>7} | []")
                continue

            for fd in res1:
                kind = getattr(fd, "kind", "evidence")
                fired_str = "True" if kind == "evidence" else f"ctx({kind})"
                coh_size = len(fd.connected_card_ids) + (1 if fd.connected_card_ids or fd.affected_txn_ids else 0)
                conn_preview = (
                    ", ".join(fd.connected_card_ids[:5])
                    + (f" (+{len(fd.connected_card_ids) - 5} more)" if len(fd.connected_card_ids) > 5 else "")
                )
                print(
                    f"{ctx.case_id:<8} | {det.name:<18} | {fired_str:<10} | {fd.lr:>6.2f} | "
                    f"{coh_size:>6} | {len(fd.affected_txn_ids):>7} | [{conn_preview}]"
                )

                # As-of correctness and existence in tx_all()
                assert fd.lr != 3.0, "Assumed LR=3.0 is forbidden"
                for tid in fd.affected_txn_ids:
                    assert int(tid) in all_txn_ids, f"Missing affected_txn_id {tid}"
                    assert txn_time_lookup[int(tid)] <= t_open, f"Future affected_txn_id {tid} > {t_open}"

                for tid in fd.claim.evidence.txn_ids:
                    assert int(tid) in all_txn_ids, f"Missing evidence txn_id {tid}"
                    assert txn_time_lookup[int(tid)] <= t_open, f"Future evidence txn_id {tid} > {t_open}"

                # Attestation schema check
                att_dict = fd.claim.to_dict()
                assert set(att_dict.keys()) == {"what", "evidence", "how", "when"}
                assert att_dict["how"]["query"] in ("rerun_micro_drip_cohort", "rerun_recipient_fan_in_burst")

                # Re-runnable query check
                q_params = fd.claim.params
                rerun_out = rerun_fn(
                    tx,
                    card_id=q_params["card_id"],
                    flagged_txn_id=q_params["flagged_txn_id"],
                    opened_at=q_params["opened_at"],
                )
                assert isinstance(rerun_out, dict)

    # Confirm legacy MicroCharge still runs without error on HHF-004
    ctx_004 = _build_ctx(cp[cp.case_id == "HHF-004"].iloc[0], tx, cc)
    legacy_out = det_legacy.run(ctx_004)
    assert isinstance(legacy_out, list)
