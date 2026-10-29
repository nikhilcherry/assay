"""Tests for Block C5 -- `VectorPrecedentVote` (`assay/r2/detectors/vector_vote.py`).

Tests assert only what we control:
  1. Reproducibility & determinism across repeated runs.
  2. As-of correctness: no retrieved neighbour ever has `closed_at > opened_at`, and
     no case ever retrieves itself.
  3. Schema/shape and attestation contract fields (`what`, `evidence`, `how`, `when`),
     `pattern == "none"`, `blocks_assumed_clear is False`, `precedent=True` in params,
     `affected_txn_ids == []`, `connected_card_ids == []`, and all attested IDs exist
     in `closed_cases()` / `tx_all()`.
  4. Backtest measurement table is produced and printed over Aug-Dec closed cases and
     `>= 2,000` legitimate card-days, with `P(fire|cleared) <= 8%` and `weight` equal to
     the 95% CI lower bound of `P(fire|fraud)/P(fire|cleared)` (Laplace).
  5. Fail-closed behaviour when `ASSAY_TG=1` and TigerGraph errors or is unreachable.
  6. `parity_check` syncs missing closed-case vectors (`CC-5566..CC-5585`), prints a raw
     TigerGraph reply for `HHF-001`, prints which of the 18 exam cases fire under local
     and under TG, asserts that the fire decision is identical on all 18 cases and
     `|tg_score - local_score| <= 0.10`, and skips cleanly with a printed reason when TG
     is down.
"""

from __future__ import annotations

import pandas as pd
import pytest

from assay.r2.detectors.vector_vote import (
    MAX_CLEARED_CASE_FPR,
    VectorPrecedentVote,
    format_backtest_grid,
    parity_check,
    run_vector_vote_backtest,
)
from assay.r2.finding import CaseCtx, REGISTRY
from assay.r2.frames import case_pack, closed_cases, tx_all
from assay.r2.vector import TigerGraphUnavailableError


def _make_ctx(row, tx: pd.DataFrame, cc: pd.DataFrame, tx_by_id: pd.DataFrame) -> CaseCtx:
    t0 = pd.Timestamp(row.opened_at)
    fid = int(row.flagged_txn_id)
    actual_card = str(tx_by_id.loc[fid, "card_id"]) if fid in tx_by_id.index else str(row.card_id)
    return CaseCtx(
        case_id=str(row.case_id),
        flagged_txn_id=fid,
        card_id=actual_card,
        customer_id=str(row.customer_id),
        opened_at=t0,
        tx=tx[tx["t"] <= t0],
        cc=cc[pd.to_datetime(cc["closed_at"]) <= t0],
    )


def test_backtest_grid_and_selection():
    summary = run_vector_vote_backtest()
    table_str = format_backtest_grid(summary)
    print("\n=== C5 VectorPrecedentVote Backtest Grid (Aug-Dec 2016 Closed Cases + Legit Card-Days) ===")
    print(table_str)

    assert summary.n_fraud > 0
    assert summary.n_cleared > 0
    assert summary.n_legit_days >= 2000
    assert len(summary.grid) > 0

    chosen = summary.chosen
    assert chosen.k in (5, 10, 20)
    assert chosen.variant in ("pure", "fused")
    assert chosen.p_fire_cleared <= MAX_CLEARED_CASE_FPR
    assert chosen.ci_cleared_low > 0.0

    det = VectorPrecedentVote()
    assert REGISTRY.get("vector_precedent") is VectorPrecedentVote
    assert det.name == "vector_precedent"
    assert det.pattern == "none"
    assert VectorPrecedentVote.blocks_assumed_clear is False
    assert det.blocks_assumed_clear is False
    assert det.k == chosen.k
    assert abs(det.theta - chosen.theta) < 1e-6
    assert abs(det.weight - round(chosen.ci_cleared_low, 4)) < 1e-6

    bt = det.backtest()
    assert bt.detector == "vector_precedent"
    assert bt.n_cases == summary.n_fraud + summary.n_cleared
    assert len(bt.caught) + len(bt.missed) == summary.n_fraud
    print(
        f"\n[Backtest Summary] caught={len(bt.caught)}/{summary.n_fraud} ({bt.recall:.2%}), "
        f"cleared_acted_against={len(bt.legit_acted_against)}/{summary.n_cleared} "
        f"({len(bt.legit_acted_against) / summary.n_cleared:.2%}), "
        f"precision={bt.precision:.2%}, lr_vs_cleared={bt.lr_measured:.2f}, "
        f"weight(ci_low_cleared)={det.weight:.2f}"
    )
    print(f"Cleared cases acted against ({len(bt.legit_acted_against)}): {bt.legit_acted_against}")


def test_exam_cases_asof_determinism_and_attestations(monkeypatch):
    monkeypatch.setenv("ASSAY_TG", "0")
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()
    tx_by_id = tx.set_index("TransactionID", drop=False)
    cc_by_id = cc.set_index("case_id", drop=False)
    cc_closed_ts = pd.to_datetime(cc_by_id["closed_at"])

    det = VectorPrecedentVote()

    print(
        f"\n=== C5 VectorPrecedentVote on 18 Exam Cases "
        f"(variant={det.variant}, k={det.k}, theta={det.theta:.4f}, weight={det.weight:.2f}) ==="
    )
    header = (
        f"{'Case':<8} {'Card':<10} {'Opened_At':<19} {'Score':>7} {'Fired?':<6} "
        f"{'Fraud/k':>7}  {'Top-5 Neighbours (id:outcome:dist)'}"
    )
    print(header)
    print("-" * len(header))

    fired_local: list[str] = []
    for r in cp.itertuples():
        ctx = _make_ctx(r, tx, cc, tx_by_id)
        ev1 = det.evaluate_ctx(ctx, use_tg_if_enabled=False)
        ev2 = det.evaluate_ctx(ctx, use_tg_if_enabled=False)
        findings = det.run(ctx)

        # Determinism check
        assert [n.case_id for n in ev1.neighbors] == [n.case_id for n in ev2.neighbors]
        assert ev1.score == ev2.score
        assert ev1.fired == ev2.fired

        # Shape & existence & strict as-of check on every retrieved neighbour
        assert len(ev1.neighbors) == det.k
        for nb in ev1.neighbors:
            assert nb.case_id in cc_by_id.index, f"Unknown neighbour case_id {nb.case_id}"
            assert nb.case_id != ctx.case_id, "Case must never retrieve itself"
            assert nb.outcome in ("confirmed_fraud", "cleared")
            nb_closed = pd.Timestamp(cc_closed_ts.loc[nb.case_id])
            assert nb_closed <= ctx.opened_at, (
                f"Future leakage: neighbour {nb.case_id} closed at {nb_closed} > opened_at {ctx.opened_at}"
            )

        n_fraud_nb = sum(1 for n in ev1.neighbors if n.outcome == "confirmed_fraud")
        nb_str = ", ".join(
            f"{n.case_id}:{'F' if n.outcome == 'confirmed_fraud' else 'C'}:{n.distance:.3f}"
            for n in ev1.neighbors[:5]
        )
        print(
            f"{ctx.case_id:<8} {ctx.card_id:<10} {str(ctx.opened_at):<19} "
            f"{ev1.score:>7.4f} {str(ev1.fired):<6} {f'{n_fraud_nb}/{len(ev1.neighbors)}':>7}  {nb_str}"
        )

        if ev1.fired:
            fired_local.append(ctx.case_id)
            assert len(findings) == 1
            fd = findings[0]
            assert fd.detector == "vector_precedent"
            assert fd.pattern == "none"
            assert fd.lr == det.weight
            assert fd.affected_txn_ids == []
            assert fd.connected_card_ids == []

            ad = fd.claim.to_dict()
            assert set(ad.keys()) == {"what", "evidence", "how", "when"}
            assert ad["how"]["query"] == "similar_cases_vec"
            params = ad["how"]["params"]
            assert params["k"] == det.k
            assert params["theta"] == round(det.theta, 4)
            assert params["backend"] == ev1.backend
            assert params["precedent"] is True
            assert params["neighbor_case_ids"] == [n.case_id for n in ev1.neighbors]
            assert params["neighbor_outcomes"] == [n.outcome for n in ev1.neighbors]
            assert len(params["neighbor_distances"]) == det.k
            assert ad["evidence"]["case_ids"] == [n.case_id for n in ev1.neighbors]
            for tid_s in ad["evidence"]["txn_ids"]:
                assert int(tid_s) in tx_by_id.index
        else:
            assert findings == []

    print(f"\n[Exam Pack Local Run] Fired cases ({len(fired_local)}/18): {fired_local}")


def test_asof_leave_one_out_on_closed_cases_sample(monkeypatch):
    monkeypatch.setenv("ASSAY_TG", "0")
    tx = tx_all()
    cc = closed_cases()
    cc_by_id = cc.set_index("case_id", drop=False)
    cc_closed_ts = pd.to_datetime(cc_by_id["closed_at"])

    det = VectorPrecedentVote()
    aug_dec = cc[pd.to_datetime(cc["opened_at"]) >= pd.Timestamp("2016-08-01")].iloc[:25]

    for r in aug_dec.itertuples():
        t0 = pd.Timestamp(r.opened_at)
        fid = int(r.txn_list[0]) if r.txn_list else 0
        ctx = CaseCtx(
            case_id=str(r.case_id),
            flagged_txn_id=fid,
            card_id=str(r.card_id),
            customer_id=str(r.customer_id),
            opened_at=t0,
            tx=tx[tx["t"] <= t0],
            cc=cc[pd.to_datetime(cc["closed_at"]) <= t0],
        )
        ev = det.evaluate_ctx(ctx, use_tg_if_enabled=False)
        for nb in ev.neighbors:
            assert nb.case_id != ctx.case_id, f"Closed case {ctx.case_id} retrieved itself!"
            assert pd.Timestamp(cc_closed_ts.loc[nb.case_id]) <= t0


def test_fail_closed_when_tg_enabled_and_unreachable(monkeypatch):
    monkeypatch.setenv("ASSAY_TG", "1")
    monkeypatch.setenv("ASSAY_TG_HOST", "http://127.0.0.1")
    monkeypatch.setenv("ASSAY_TG_PORT", "1")

    import assay.r2.vector as vec_mod

    def _boom(qname, params):
        raise TigerGraphUnavailableError(f"Simulated TG outage on {qname}")

    monkeypatch.setattr(vec_mod, "_tg_run_query", _boom)

    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()
    tx_by_id = tx.set_index("TransactionID", drop=False)
    ctx = _make_ctx(next(cp.itertuples()), tx, cc, tx_by_id)

    det = VectorPrecedentVote()
    with pytest.raises(TigerGraphUnavailableError):
        det.run(ctx)


def test_parity_check():
    res = parity_check()
    assert "skipped" in res
    if not res["skipped"]:
        assert res["n_checked"] == 18
        assert res["sync_info"]["still_missing"] == []
        assert res["fire_mismatches"] == [], f"TG vs Local fire decision mismatch: {res['fire_mismatches']}"
        assert res["score_mismatches"] == [], f"TG vs Local score diff > 0.10: {res['score_mismatches']}"
