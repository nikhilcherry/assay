"""Tests and measurements for Block B33 -- Precedent by shared entity."""

from __future__ import annotations

import pandas as pd

from assay.r2.finding import CaseCtx
from assay.r2.frames import case_pack, closed_cases, tx_all
from assay.r2.precedent import (
    LINK_TYPES,
    PrecedentDetector,
    backtest_precedent_links,
    exam_case_precedents,
    precedent_lr,
    precedents,
    precedents_gsql,
)


def test_precedents_gsql_and_reference_parity() -> None:
    gsql = precedents_gsql()
    assert "CREATE QUERY precedents_by_entity" in gsql
    assert "FraudCase" in gsql
    assert "fc.closed_at <= as_of" in gsql
    assert "CASE_ON_CARD" in gsql
    assert "USED_DEVICE" in gsql

    # Verify reproducibility of the pandas reference query when re-run with identical params
    cp = case_pack()
    for row in cp.itertuples(index=False):
        r1 = precedents(str(row.card_id), int(row.flagged_txn_id), str(row.opened_at))
        r2 = precedents(str(row.card_id), int(row.flagged_txn_id), str(row.opened_at))
        assert [p.to_dict() for p in r1] == [p.to_dict() for p in r2]


def test_as_of_strictly_enforced() -> None:
    # Using an early timestamp before any case closed should return 0 precedents
    early = precedents("C05061-K1", 3660108, "2016-07-01 00:00:00")
    assert early == []

    # Verify that for every exam case, no precedent uses a case closed after the exam case opened
    cp = case_pack()
    for row in cp.itertuples(index=False):
        as_of = pd.to_datetime(row.opened_at)
        precs = precedents(str(row.card_id), int(row.flagged_txn_id), as_of)
        for p in precs:
            assert p.closed_at <= as_of, f"{p.case_id} closed_at {p.closed_at} > {as_of}"
            assert 0.0 < p.recency_weight <= 1.0
            assert len(p.paths) >= 1
            for path in p.paths:
                assert path.entity_type in LINK_TYPES


def test_backtest_and_precedent_lr() -> None:
    stats = backtest_precedent_links()
    print("\n=== BACKTEST: AUG-DEC CLOSED CASES AS-OF PER LINK TYPE ===")
    header = (
        f"{'Link Type':<24} | {'N_link':>6} | {'Fraud':>5} | {'Cleared':>7} | "
        f"{'P(F|link) [95% CI]':<23} | {'P(F|~link) [95% CI]':<23} | {'LR [95% CI]':<22}"
    )
    print(header)
    print("-" * len(header))
    for lt in list(LINK_TYPES) + ["any_fraud_precedent"]:
        s = stats[lt]
        assert s.n_eval_cases > 3000
        assert 0.0 <= s.p_fraud_given_link <= 1.0
        assert 0.0 <= s.p_fraud_given_no_link <= 1.0
        assert s.lr > 0.0
        assert s.lr_ci[0] <= s.lr <= s.lr_ci[1]
        n_clr = s.n_linked_fraud - s.n_linked_fraud_is_fraud
        pf_l = f"{s.p_fraud_given_link:.3f} [{s.p_fraud_given_link_ci[0]:.3f},{s.p_fraud_given_link_ci[1]:.3f}]"
        pf_nl = f"{s.p_fraud_given_no_link:.3f} [{s.p_fraud_given_no_link_ci[0]:.3f},{s.p_fraud_given_no_link_ci[1]:.3f}]"
        lr_s = f"{s.lr:.2f} [{s.lr_ci[0]:.2f},{s.lr_ci[1]:.2f}]"
        print(f"{lt:<24} | {s.n_linked_fraud:>6} | {s.n_linked_fraud_is_fraud:>5} | {n_clr:>7} | {pf_l:<23} | {pf_nl:<23} | {lr_s:<22}")
        if s.legit_acted_against:
            print(f"  -> Cleared (legit) cases acted against ({len(s.legit_acted_against)}): {s.legit_acted_against[:10]}"
                  + (" ..." if len(s.legit_acted_against) > 10 else ""))


def test_exam_cases_and_hhg_precedents() -> None:
    cp = case_pack()
    all_precs = exam_case_precedents()
    assert len(all_precs) == 18

    print("\n=== EXAM CASES WITH HHG-DERIVED PRECEDENTS (analyst_notes contains 'Case HHG-') ===")
    hhg_exam_hits: dict[str, list[tuple[str, str, str, str]]] = {}
    for cid, p_list in all_precs.items():
        hhg_precs = [p for p in p_list if p.is_hhg]
        if hhg_precs:
            hhg_exam_hits[cid] = [
                (p.case_id, p.hhg_id or "", p.outcome, f"{p.primary_path.entity_type}={p.primary_path.value}")
                for p in hhg_precs
            ]
            for p in hhg_precs:
                paths_str = "; ".join(
                    f"{path.entity_type}={path.value} (txns {path.current_txn_id}<->{path.precedent_txn_id})"
                    for path in p.paths
                )
                print(
                    f"  {cid} -> {p.case_id} ({p.hhg_id}) | outcome={p.outcome} | "
                    f"pattern={p.pattern} | exposure=${p.exposure:.2f} | recency={p.recency_weight:.3f} | {paths_str}"
                )
    if not hhg_exam_hits:
        print("  none found")

    print("\n=== ALL PRECEDENTS AND LINK PATHS FOR ALL 18 EXAM CASES ===")
    for row in cp.itertuples(index=False):
        cid = str(row.case_id)
        p_list = all_precs[cid]
        lr_res = precedent_lr(p_list)
        print(
            f"\n[{cid}] card={row.card_id} cust={row.customer_id} flagged_txn={row.flagged_txn_id} "
            f"opened_at={row.opened_at} | total_precedents={len(p_list)} "
            f"(fraud={lr_res['n_fraud_precedents']}, cleared={lr_res['n_cleared_precedents']}) "
            f"| measured_LR={lr_res['lr']:.2f} (best_link={lr_res['best_link_type']})"
        )
        if not p_list:
            print("  none found")
        for p in p_list:
            hhg_lbl = f" [{p.hhg_id}]" if p.is_hhg else ""
            paths_fmt = " ; ".join(
                f"{pt.entity_type}({pt.value}: {pt.current_txn_id}<->{pt.precedent_txn_id})"
                for pt in p.paths
            )
            print(
                f"  - {p.case_id}{hhg_lbl}: outcome={p.outcome}, pattern={p.pattern}, "
                f"exposure=${p.exposure:.2f}, closed={p.closed_at}, recency={p.recency_weight:.3f} "
                f"| paths: {paths_fmt}"
            )


def test_detector_interface_and_attestation() -> None:
    det = PrecedentDetector()
    bt = det.backtest()
    assert bt.n_cases > 3000
    assert bt.lr_measured > 0.0

    cp = case_pack()
    tx = tx_all()
    cc = closed_cases()
    cc["closed_ts"] = pd.to_datetime(cc["closed_at"])

    # Run detector on all exam cases and verify attestation reproducibility
    fired_count = 0
    for row in cp.itertuples(index=False):
        opened = pd.to_datetime(row.opened_at)
        ctx = CaseCtx(
            case_id=str(row.case_id),
            flagged_txn_id=int(row.flagged_txn_id),
            card_id=str(row.card_id),
            customer_id=str(row.customer_id),
            opened_at=opened,
            tx=tx[tx["t"] <= opened],
            cc=cc[cc["closed_ts"] <= opened],
        )
        findings = det.run(ctx)
        if findings:
            fired_count += 1
            f = findings[0]
            d = f.claim.to_dict()
            assert d["how"]["query"] == "assay.r2.precedent.precedents"
            params = d["how"]["params"]
            rerun = precedents(params["card_id"], int(params["flagged_txn_id"]), params["as_of"])
            rerun_fraud_ids = [p.case_id for p in rerun if p.outcome == "confirmed_fraud"]
            assert d["evidence"]["case_ids"] == [str(x) for x in rerun_fraud_ids]
            assert int(row.flagged_txn_id) in [int(x) for x in d["evidence"]["txn_ids"]]
    assert fired_count >= 1
