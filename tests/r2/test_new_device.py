"""Tests and backtest measurements for BLOCK C4 (NewDeviceBurst and DeviceChurn).

Follows the strict TEST RULE:
- Asserts reproducibility, as-of correctness (no future data), schema/shape,
  attestation fields, determinism, no TransactionID-as-feature, and that the
  measurement tables are produced.
- Prints measured backtests over all 5,585 closed cases and >= 3,000 legit
  card-days, and prints exam pack results across all 18 case_pack() cases
  without asserting which exam cases fire or hoping for specific LRs.
"""

from __future__ import annotations

import inspect
import pandas as pd
import pytest

from assay.r2.detectors import new_device as nd_mod
from assay.r2.detectors.new_device import (
    DeviceChurn,
    NewDeviceBurst,
    _calibrated_defaults,
    evaluate_device_churn_grid,
    evaluate_new_device_burst_grid,
    rerun,
    sample_legit_card_days,
)
from assay.r2.finding import REGISTRY, Backtest, CaseCtx, Finding
from assay.r2.frames import case_pack, closed_cases, tx_all


def _build_ctx_from_case_row(row, tx: pd.DataFrame, cc: pd.DataFrame) -> CaseCtx:
    t0 = pd.Timestamp(row.opened_at)
    fid = int(row.flagged_txn_id)
    hit = tx.loc[tx["TransactionID"] == fid, "card_id"]
    card = str(hit.iat[0]) if not hit.empty else str(row.card_id)
    return CaseCtx(
        case_id=str(row.case_id),
        flagged_txn_id=fid,
        card_id=card,
        customer_id=str(row.customer_id),
        opened_at=t0,
        tx=tx[tx["t"] <= t0],
        cc=cc[pd.to_datetime(cc["closed_at"]) <= t0],
    )


def test_registry_and_no_txid_leakage():
    """Verify both detectors are registered and do not use TransactionID ranges/thresholds."""
    assert "new_device_burst" in REGISTRY
    assert "device_churn" in REGISTRY
    assert REGISTRY["new_device_burst"] is NewDeviceBurst
    assert REGISTRY["device_churn"] is DeviceChurn

    src = inspect.getsource(nd_mod)
    # Must never filter on organizer-inserted TransactionID >= 3660000 or similar
    assert "3660" not in src
    assert "TransactionID >" not in src
    assert "TransactionID >=" not in src
    assert "TransactionID <" not in src


def test_backtest_measurements_and_ablation_tables():
    """Run full backtest sweeps over all 5,585 closed cases and >= 3,000 legit card-days."""
    tx = tx_all()
    cc = closed_cases()
    assert len(cc) == 5585

    cal = _calibrated_defaults()
    ndb_grid = cal["ndb_grid"]
    dc_grid = cal["dc_grid"]

    assert len(ndb_grid) == 15  # 3 m values x 5 condition variants
    assert len(dc_grid) == 4    # 4 normalisation variants

    print("\n=== C4 BACKTEST TABLE 1: new_device_burst (5,585 closed cases + 3,500 legit card-days) ===")
    header = (
        f"{'m':>3} | {'variant':<34} | {'fraud':>10} | {'ato_cnp':>10} | "
        f"{'cleared':>9} | {'legit_day':>10} | {'LR_legit (95% CI)':<24} | {'LR_clr':>7} | {'keep?'}"
    )
    print(header)
    print("-" * len(header))
    for r in ndb_grid:
        assert r["n_fraud"] == 4675
        assert r["n_cleared"] == 910
        assert r["n_legit_days"] >= 3000
        assert isinstance(r["backtest"], Backtest)
        ci_str = f"{r['lr_vs_legit_days']:.2f} [{r['ci_low_vs_legit_days']:.2f}, {r['ci_high_vs_legit_days']:.2f}]"
        print(
            f"{r['m']:>3.0f} | {r['variant']:<34} | "
            f"{r['caught_fraud']:>4}/{r['n_fraud']:<5} | "
            f"{r['caught_ato_cnp']:>4}/{r['n_ato_cnp']:<5} | "
            f"{r['cleared_fired']:>3}/{r['n_cleared']:<5} | "
            f"{r['legit_days_fired']:>4}/{r['n_legit_days']:<5} | "
            f"{ci_str:<24} | {r['lr_vs_cleared']:>7.2f} | {r['passes_keep_gate']}"
        )
        print(f"    cleared_case_ids ({len(r['cleared_case_ids'])}): {r['cleared_case_ids']}")

    print("\n=== C4 BACKTEST TABLE 2: device_churn (5,585 closed cases + 3,500 legit card-days) ===")
    header2 = (
        f"{'variant':<38} | {'fraud':>10} | {'ato_cnp':>10} | "
        f"{'cleared':>9} | {'legit_day':>10} | {'LR_legit (95% CI)':<24} | {'LR_clr':>7} | {'keep?'}"
    )
    print(header2)
    print("-" * len(header2))
    for r in dc_grid:
        assert r["n_fraud"] == 4675
        assert r["n_cleared"] == 910
        assert r["n_legit_days"] >= 3000
        assert isinstance(r["backtest"], Backtest)
        ci_str = f"{r['lr_vs_legit_days']:.2f} [{r['ci_low_vs_legit_days']:.2f}, {r['ci_high_vs_legit_days']:.2f}]"
        print(
            f"{r['variant']:<38} | "
            f"{r['caught_fraud']:>4}/{r['n_fraud']:<5} | "
            f"{r['caught_ato_cnp']:>4}/{r['n_ato_cnp']:<5} | "
            f"{r['cleared_fired']:>3}/{r['n_cleared']:<5} | "
            f"{r['legit_days_fired']:>4}/{r['n_legit_days']:<5} | "
            f"{ci_str:<24} | {r['lr_vs_cleared']:>7.2f} | {r['passes_keep_gate']}"
        )
        print(f"    cleared_case_ids ({len(r['cleared_case_ids'])}): {r['cleared_case_ids']}")

    best_ndb = cal["new_device_burst"]
    best_dc = cal["device_churn"]
    print(
        f"\nSelected new_device_burst: m={best_ndb['m']}, variant={best_ndb['variant']}, "
        f"LR_legit={best_ndb['lr_vs_legit_days']} CI=[{best_ndb['ci_low_vs_legit_days']}, {best_ndb['ci_high_vs_legit_days']}], "
        f"passes_keep_gate={best_ndb['passes_keep_gate']}"
    )
    print(
        f"Selected device_churn: variant={best_dc['variant']}, "
        f"LR_legit={best_dc['lr_vs_legit_days']} CI=[{best_dc['ci_low_vs_legit_days']}, {best_dc['ci_high_vs_legit_days']}], "
        f"passes_keep_gate={best_dc['passes_keep_gate']}"
    )


def test_exam_pack_run_and_attestation_reproducibility():
    """Run both detectors over the 18 case_pack() cases; verify schema, determinism, and as-of rules."""
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()
    assert len(cp) == 18

    tx_id_set = set(tx["TransactionID"].tolist())
    card_id_set = set(tx["card_id"].tolist())
    cust_id_set = set(tx["customer_id"].tolist())

    # Test both calibrated detectors and explicit base configurations
    detectors = [
        ("new_device_burst (calibrated)", NewDeviceBurst()),
        ("new_device_burst (m=3, base)", NewDeviceBurst(m=3.0, require_email_novel_or_anon=False, require_id15_new=False)),
        ("new_device_burst (m=3, +email)", NewDeviceBurst(m=3.0, require_email_novel_or_anon=True, require_id15_new=False)),
        ("device_churn (calibrated)", DeviceChurn()),
    ]

    print("\n=== C4 EXAM PACK EVALUATION (18 cases) ===")
    for label, det in detectors:
        print(f"\n--- Detector: {label} (lr_weight={det.lr_weight}, ci_low={det.ci_low}, keep={det.passes_keep_gate}) ---")
        for row in cp.itertuples():
            ctx = _build_ctx_from_case_row(row, tx, cc)
            f1 = det.run(ctx)
            f2 = det.run(ctx)

            # Determinism check
            assert len(f1) == len(f2)
            if f1:
                assert f1[0].affected_txn_ids == f2[0].affected_txn_ids
                assert f1[0].lr == f2[0].lr
                assert f1[0].claim.what == f2[0].claim.what

                fd = f1[0]
                assert isinstance(fd, Finding)
                assert fd.detector in ("new_device_burst", "device_churn")
                assert fd.pattern == "account_takeover"
                assert len(fd.affected_txn_ids) >= 2

                # Every attested ID must exist in tx_all and have t <= opened_at
                ev = fd.claim.evidence
                assert set(ev.txn_ids).issubset(tx_id_set)
                assert set(ev.card_ids).issubset(card_id_set)
                assert set(ev.customer_ids).issubset(cust_id_set)

                sub_tx = tx[tx["TransactionID"].isin(ev.txn_ids)]
                assert (sub_tx["t"] <= ctx.opened_at).all(), "As-of violation: future txn in evidence!"

                # Re-runnable attestation check
                re_ids = rerun(fd.claim.params)
                assert re_ids == fd.affected_txn_ids, f"Rerun mismatch on {row.case_id}"

                # As-of invariance: passing full tx (with future rows) to a modified ctx must not change output
                ctx_with_future = CaseCtx(
                    case_id=ctx.case_id,
                    flagged_txn_id=ctx.flagged_txn_id,
                    card_id=ctx.card_id,
                    customer_id=ctx.customer_id,
                    opened_at=ctx.opened_at,
                    tx=tx,  # includes rows after opened_at
                    cc=cc,
                )
                f_future = det.run(ctx_with_future)
                assert len(f_future) == 1
                assert f_future[0].affected_txn_ids == fd.affected_txn_ids

                print(
                    f"  {row.case_id} ({ctx.card_id}): FIRED=True | lr={fd.lr:.4f} | "
                    f"txns={fd.affected_txn_ids} | what={fd.claim.what}"
                )
            else:
                print(f"  {row.case_id} ({ctx.card_id}): FIRED=False")
