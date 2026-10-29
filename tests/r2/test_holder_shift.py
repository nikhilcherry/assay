"""Tests and backtest measurements for BLOCK C7 -- HolderShift (`holder_shift`)."""

from __future__ import annotations

import inspect

import pandas as pd

from assay.r2.detectors import holder_shift as hs
from assay.r2.detectors.holder_shift import (
    KEEP_GATE_CI_LOW,
    KEEP_GATE_MAX_CLEARED_RATE,
    MIN_PRIOR_TXNS,
    HolderShift,
    _calibrated_holder_shift,
    bootstrap_lr_ci,
    get_enriched_closed_cases,
    inspect_case_holder_shift,
    rerun,
    safe_rate,
)
from assay.r2.finding import REGISTRY, Backtest, CaseCtx
from assay.r2.frames import case_pack, closed_cases, tx_all

SENSITIVE_EXAM_CASES = {
    "HHF-009",
    "HHF-010",
    "HHF-012",
    "HHF-013",
    "HHF-014",
    "HHF-015",
    "HHF-016",
    "HHF-017",
    "HHF-018",
}


def _make_exam_ctx(row: pd.Series, tx: pd.DataFrame, cc: pd.DataFrame) -> CaseCtx:
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


def _fmt_rate(k: int, n: int, r: float | None) -> str:
    if r is None or n <= 0:
        return f"{k}/{n} (insufficient)"
    return f"{k}/{n} ({r*100:.2f}%)"


def _fmt_lr(lr: float | None, lo: float | None, hi: float | None, insuff: bool) -> str:
    if insuff or lr is None or lo is None or hi is None:
        return "insufficient data"
    return f"{lr:.4f} [{lo:.4f}, {hi:.4f}]"


def test_safe_rate_and_bootstrap_guards() -> None:
    """Verify rate and CI helpers guard n == 0 and out-of-range inputs cleanly."""
    assert safe_rate(0, 0) is None
    assert safe_rate(5, 0) is None
    assert safe_rate(3, 10) == 0.3

    lr, lo, hi, insuff = bootstrap_lr_ci(0, 0, 10, 1)
    assert insuff is True and lr is None and lo is None and hi is None

    lr2, lo2, hi2, insuff2 = bootstrap_lr_ci(10, 2, 0, 0)
    assert insuff2 is True and lr2 is None and lo2 is None and hi2 is None

    # Out-of-range k > n is safely clipped without raising ValueError
    lr3, lo3, hi3, insuff3 = bootstrap_lr_ci(10, 12, 8, 9)
    assert insuff3 is False and lr3 is not None and lo3 is not None and hi3 is not None


def test_registry_and_backtest_table() -> None:
    """Verify HolderShift registers in REGISTRY and prints full measured backtest table."""
    assert "holder_shift" in REGISTRY
    assert REGISTRY["holder_shift"] is HolderShift

    cc_enr = get_enriched_closed_cases()
    n_fraud = int((cc_enr["outcome"] == "confirmed_fraud").sum())
    n_cleared = int((cc_enr["outcome"] == "cleared").sum())
    n_fraud_elig_48 = int(
        ((cc_enr["outcome"] == "confirmed_fraud") & cc_enr["holder_eligible_48h"]).sum()
    )
    n_cleared_elig_48 = int(
        ((cc_enr["outcome"] == "cleared") & cc_enr["holder_eligible_48h"]).sum()
    )
    n_fraud_elig_168 = int(
        ((cc_enr["outcome"] == "confirmed_fraud") & cc_enr["holder_eligible_168h"]).sum()
    )
    n_cleared_elig_168 = int(
        ((cc_enr["outcome"] == "cleared") & cc_enr["holder_eligible_168h"]).sum()
    )

    print("\n" + "=" * 125)
    print(f"C7 HOLDER_SHIFT APPLICABILITY ON CLOSED CASES (uid >= {MIN_PRIOR_TXNS} prior txns before window):")
    print(
        f"  48h Window -- Confirmed Fraud: {n_fraud_elig_48} / {n_fraud} ({n_fraud_elig_48 / n_fraud:.2%}) | "
        f"Cleared Cases: {n_cleared_elig_48} / {n_cleared} ({n_cleared_elig_48 / n_cleared:.2%})"
    )
    print(
        f"  7d Window  -- Confirmed Fraud: {n_fraud_elig_168} / {n_fraud} ({n_fraud_elig_168 / n_fraud:.2%}) | "
        f"Cleared Cases: {n_cleared_elig_168} / {n_cleared} ({n_cleared_elig_168 / n_cleared:.2%})"
    )
    print("=" * 125)

    cal = _calibrated_holder_shift()
    grid = cal["grid"]
    best = cal["best"]

    assert len(grid) == 16  # 2 windows x 2 m x 4 shift_modes

    header = (
        f"{'Win':<4} | {'m':<3} | {'Shift Mode':<13} | {'Fraud All':<15} | {'Fraud Elig':<15} | "
        f"{'Cleared All':<15} | {'Cleared Elig':<15} | {'LegitDay':<15} | "
        f"{'LR_cleared [95% CI]':<25} | {'LR_elig [95% CI]':<25} | {'Gate'}"
    )
    print("\nC7 HOLDER_SHIFT BACKTEST GRID (5,585 Closed Cases + 3,500 Legit Card-Days):")
    print(header)
    print("-" * len(header))

    for r in grid:
        fa_str = _fmt_rate(r["caught_fraud"], r["n_fraud"], r["p_fire_fraud"])
        fe_str = _fmt_rate(r["caught_fraud"], r["n_fraud_elig_ge10"], r["p_fire_fraud_elig"])
        ca_str = _fmt_rate(r["cleared_fired"], r["n_cleared"], r["p_fire_cleared"])
        ce_str = _fmt_rate(r["cleared_fired"], r["n_cleared_elig_ge10"], r["p_fire_cleared_elig"])
        ld_str = _fmt_rate(r["legit_days_fired"], r["n_legit_days"], r["p_fire_legit_day"])
        lrc_str = _fmt_lr(
            r["lr_vs_cleared"],
            r["ci_low_vs_cleared"],
            r["ci_high_vs_cleared"],
            r["insufficient_data_all"],
        )
        lre_str = _fmt_lr(
            r["lr_vs_cleared_elig"],
            r["ci_low_vs_cleared_elig"],
            r["ci_high_vs_cleared_elig"],
            r["insufficient_data_elig"],
        )
        print(
            f"{r['window_label']:<4} | {r['m']:<3.1f} | {r['shift_mode']:<13} | {fa_str:<15} | {fe_str:<15} | "
            f"{ca_str:<15} | {ce_str:<15} | {ld_str:<15} | {lrc_str:<25} | {lre_str:<25} | {r['passes_keep_gate']}"
        )
        if r["cleared_case_ids"]:
            print(f"     -> Cleared cases fired ({len(r['cleared_case_ids'])}): {r['cleared_case_ids']}")

    print("-" * len(header))
    print(
        f"BEST CELL: win={best['window_label']}, m={best['m']}, mode={best['shift_mode']} -> "
        f"caught={best['caught_fraud']}/{best['n_fraud']} (elig={best['caught_fraud']}/{best['n_fraud_elig_ge10']}), "
        f"cleared={best['cleared_fired']}/{best['n_cleared']} (elig={best['cleared_fired']}/{best['n_cleared_elig_ge10']}), "
        f"LR_cleared={best['lr_vs_cleared']} [{best['ci_low_vs_cleared']}, {best['ci_high_vs_cleared']}], "
        f"passes_keep_gate={best['passes_keep_gate']}"
    )

    det = HolderShift()
    bt = det.backtest(closed_cases(), tx_all())
    assert isinstance(bt, Backtest)
    assert bt.detector == "holder_shift"
    assert bt.n_cases == 5585
    assert len(bt.caught) + len(bt.missed) == n_fraud

    if not det.passes_keep_gate:
        assert det.lr_weight == 1.0
    else:
        assert det.lr_weight > KEEP_GATE_CI_LOW
        assert det.p_fire_cleared <= KEEP_GATE_MAX_CLEARED_RATE


def test_exam_pack_fires_and_sensitive_case_details() -> None:
    """Print which of the 18 case_pack() cases fire (or are not applicable) across 48h and 7d,
    and print full details for HHF-009, 010, 012-018."""
    cp = case_pack()
    tx = tx_all()
    cc = closed_cases()

    det_default = HolderShift()

    print("\n" + "=" * 125)
    print(
        f"EXAM PACK EVALUATION (18 Cases) -- Default HolderShift (passes_keep_gate={det_default.passes_keep_gate}, lr_weight={det_default.lr_weight}):"
    )
    print("=" * 125)

    for row in cp.itertuples():
        ctx = _make_exam_ctx(row, tx, cc)
        findings_gated = det_default.run(ctx)
        if not det_default.passes_keep_gate:
            assert findings_gated == [], "Ungated detector must emit nothing when it fails the keep gate"

        info_48_m2 = inspect_case_holder_shift(ctx, window_hours=48, m=2.0, shift_mode="dev_or_email")
        info_48_m3 = inspect_case_holder_shift(ctx, window_hours=48, m=3.0, shift_mode="dev_or_email")
        info_7d_m2 = inspect_case_holder_shift(ctx, window_hours=168, m=2.0, shift_mode="dev_or_email")
        info_7d_m3 = inspect_case_holder_shift(ctx, window_hours=168, m=3.0, shift_mode="dev_or_email")

        status_48 = (
            f"FIRED(m2={info_48_m2['fired']},m3={info_48_m3['fired']})"
            if info_48_m2["applicable"]
            else f"N/A (prior={info_48_m2['prior_cnt']}<10, total_uid={info_48_m2['total_uid_txns_as_of']})"
        )
        status_7d = (
            f"FIRED(m2={info_7d_m2['fired']},m3={info_7d_m3['fired']})"
            if info_7d_m2["applicable"]
            else f"N/A (prior={info_7d_m2['prior_cnt']}<10, total_uid={info_7d_m2['total_uid_txns_as_of']})"
        )

        marker = "[SENSITIVE]" if row.case_id in SENSITIVE_EXAM_CASES else "           "
        print(
            f"{row.case_id} {marker} | card={ctx.card_id:<10} | uid={str(info_48_m2['uid']):<18} | "
            f"48h: {status_48:<34} | 7d: {status_7d:<34} | gated_emitted={len(findings_gated)}"
        )
        if info_48_m2["applicable"] or info_7d_m2["applicable"] or (row.case_id in SENSITIVE_EXAM_CASES):
            print(
                f"   -> details: prior_48h={info_48_m2['prior_cnt']}, prior_7d={info_7d_m2['prior_cnt']}, "
                f"med_48h={info_48_m2['prior_med']}, max_on_48h={info_48_m2['max_online_amt']}, "
                f"ratio_48h={info_48_m2['amt_ratio']}, nov_dev_48h={info_48_m2['novel_devices']}, "
                f"nov_em_48h={info_48_m2['novel_emails']}, txns_48h_m2={info_48_m2['affected_txn_ids']}"
            )
            if info_7d_m2["applicable"]:
                print(
                    f"   -> 7d details: prior_7d={info_7d_m2['prior_cnt']}, med_7d={info_7d_m2['prior_med']}, "
                    f"max_on_7d={info_7d_m2['max_online_amt']}, ratio_7d={info_7d_m2['amt_ratio']}, "
                    f"nov_dev_7d={info_7d_m2['novel_devices']}, nov_em_7d={info_7d_m2['novel_emails']}, "
                    f"txns_7d_m2={info_7d_m2['affected_txn_ids']}"
                )


def test_as_of_correctness_and_attestation_rerun() -> None:
    """Verify strict as-of behaviour (future data after opened_at never changes output),
    that backtest and run() agree on the exact same case, and that any emitted attestation
    reproduces via rerun(params)."""
    tx = tx_all()
    cc_enr = get_enriched_closed_cases()
    cp = case_pack()

    det_forced = HolderShift(window_hours=48, m=2.0, shift_mode="dev_or_email", emit_when_ungated=True)

    cal = _calibrated_holder_shift()
    cell_48_m2 = [
        r
        for r in cal["grid"]
        if r["window_hours"] == 48 and r["m"] == 2.0 and r["shift_mode"] == "dev_or_email"
    ][0]
    caught_ids = cell_48_m2["backtest"].caught
    assert len(caught_ids) > 0, "Expected at least one historical closed case caught to test attestation rerun"

    sample_case_id = caught_ids[0]
    c_row = cc_enr[cc_enr["case_id"] == sample_case_id].iloc[0]
    t0 = pd.Timestamp(c_row.opened_at)
    fid = int(c_row.anchor_txn_id)

    ctx_as_of = CaseCtx(
        case_id=str(c_row.case_id),
        flagged_txn_id=fid,
        card_id=str(c_row.card_id),
        customer_id=str(c_row.customer_id),
        opened_at=t0,
        tx=tx[tx["t"] <= t0],
        cc=cc_enr[pd.to_datetime(cc_enr["closed_at"]) <= t0],
    )

    ctx_with_future = CaseCtx(
        case_id=str(c_row.case_id),
        flagged_txn_id=fid,
        card_id=str(c_row.card_id),
        customer_id=str(c_row.customer_id),
        opened_at=t0,
        tx=tx,
        cc=cc_enr,
    )

    res_as_of = det_forced.run(ctx_as_of)
    res_future = det_forced.run(ctx_with_future)
    print(
        f"\n[AS-OF & BACKTEST AGREEMENT] Picked caught case={sample_case_id} "
        f"(card={c_row.card_id}, anchor_txn_id={fid}, uid={c_row.anchor_uid}): "
        f"backtest_caught=True, run(as_of)_fired={len(res_as_of) == 1}, "
        f"run(with_future)_fired={len(res_future) == 1}, "
        f"affected_txn_ids={res_as_of[0].affected_txn_ids if res_as_of else []}"
    )
    assert len(res_as_of) == len(res_future) == 1
    assert res_as_of[0].affected_txn_ids == res_future[0].affected_txn_ids

    # Check attestation schema and rerun reproducibility
    f0 = res_as_of[0]
    d = f0.claim.to_dict()
    assert set(d.keys()) == {"what", "evidence", "how", "when"}
    assert d["how"]["query"] == "holder_shift"
    re_ids = rerun(d["how"]["params"])
    assert re_ids == f0.affected_txn_ids

    # Verify determinism across all 18 exam cases
    for row in cp.itertuples():
        ctx = _make_exam_ctx(row, tx, cc_enr)
        i1 = det_forced.inspect(ctx)
        i2 = det_forced.inspect(ctx)
        assert i1 == i2


def test_no_transaction_id_as_feature() -> None:
    """Ensure TransactionID is never used as a threshold or numerical feature."""
    src = inspect.getsource(hs)
    forbidden = ["TransactionID >", "TransactionID <", "TransactionID >=", "TransactionID <=", "3600000"]
    for tok in forbidden:
        assert tok not in src, f"Forbidden TransactionID threshold {tok!r} found in holder_shift.py"
