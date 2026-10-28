from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from assay.data import ROOT
from assay.r2.finding import CaseCtx
from assay.r2.frames import case_pack, closed_cases, tx_all
from assay.r2.vector import (
    EMBED_DIM,
    FEATURE_NAMES,
    FEATURE_SPEC,
    FORBIDDEN_CASE_COLS,
    FusedVectorDetector,
    LocalVectorIndex,
    TigerGraphUnavailableError,
    embed_case,
    embed_closed_cases,
    embed_exam_cases,
    export_tg_vectors,
    format_feature_table,
    run_ablation,
)


def test_no_leakage_in_feature_spec():
    print("\n[B14 Feature Audit: 24 Features & Exact Source Columns]\n" + format_feature_table())
    assert len(FEATURE_SPEC) == EMBED_DIM == 24
    assert len(FEATURE_NAMES) == EMBED_DIM == 24

    for fname, src in FEATURE_SPEC:
        lower_name = fname.lower()
        lower_src = src.lower()
        for forbidden in FORBIDDEN_CASE_COLS:
            assert forbidden.lower() not in lower_name, f"Forbidden token {forbidden} in feature {fname}"
            assert f"cc.{forbidden.lower()}" not in lower_src, f"Forbidden column {forbidden} in source {src}"
            assert f"closed_cases.{forbidden.lower()}" not in lower_src
        assert "tx_all." in src


def test_embeddings_and_ablation():
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    cc_emb, std = embed_closed_cases(cc, tx, return_standardizer=True)
    assert len(cc_emb) == len(cc) == 5585

    mat = np.vstack(cc_emb["embedding"].to_list())
    assert mat.shape == (5585, EMBED_DIM)
    assert not np.isnan(mat).any()
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

    # Verify that mutating post-hoc closed-case columns (outcome, exposure_usd, txn_list, connected_list)
    # produces the EXACT same embedding (zero post-hoc leakage).
    cc_sub = cc.iloc[:25].copy()
    emb_orig = embed_closed_cases(cc_sub, tx, return_standardizer=False)
    cc_mut = cc_sub.copy()
    cc_mut["outcome"] = "cleared"
    cc_mut["exposure_usd"] = 999999.0
    cc_mut["n_txns"] = 999
    cc_mut["txn_list"] = [[] for _ in range(len(cc_mut))]
    cc_mut["connected_list"] = [["C99999-K9"] * 10 for _ in range(len(cc_mut))]
    cc_mut["pattern"] = "none"
    emb_mut = embed_closed_cases(cc_mut, tx, return_standardizer=False)
    for i in range(len(cc_sub)):
        assert np.allclose(emb_orig["embedding"].iloc[i], emb_mut["embedding"].iloc[i], atol=1e-9)

    # Determinism check on single case embedding
    sample_row = cc.iloc[100]
    opened_ts = pd.Timestamp(sample_row.opened_at)
    sample_asof = tx[(tx["card_id"] == sample_row.card_id) & (tx["t"] <= opened_ts)]
    t_latest = sample_asof["t"].max()
    ep_tx = sample_asof[sample_asof["t"] >= t_latest - pd.Timedelta(hours=24)]
    hist_tx = sample_asof[sample_asof["t"] < t_latest - pd.Timedelta(hours=24)]
    v1 = embed_case(ep_tx, hist_tx, standardizer=std, asof_shared_dev_cards=0)
    v2 = embed_case(ep_tx, hist_tx, standardizer=std, asof_shared_dev_cards=0)
    assert np.allclose(v1, v2, atol=1e-9)

    # Exam pack embeddings (all 18 HHF cases)
    cp_emb = embed_exam_cases(cp, tx, standardizer=std)
    assert len(cp_emb) == 18
    cp_mat = np.vstack(cp_emb["embedding"].to_list())
    assert cp_mat.shape == (18, EMBED_DIM)
    assert not np.isnan(cp_mat).any()

    # Export CSV for TigerGraph vector loading
    csv_path = export_tg_vectors(cc_emb)
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        first_line = f.readline().strip()
    assert header == "case_id|behaviour"
    parts = first_line.split("|")
    assert parts[0].startswith("CC-")
    assert len(parts[1].split(",")) == EMBED_DIM

    # Ablation over ALL held-out Nov-Dec closed cases (as-of each case's opened_at)
    rep = run_ablation(cc=cc, tx=tx, k=5)
    print("\n[B14 Filter Step Trace]", rep.filter_trace)
    print(
        f"\n[B14 Ablation Oct-Dec (as-of opened_at)] n_eval={rep.n_eval} "
        f"(fraud={rep.n_fraud}, cleared={rep.n_cleared}, majority_baseline={rep.majority_baseline_acc:.4f})\n"
        f"  {'Method':<18} | {'Acc':>7} | {'BalAcc':>7} | {'Brier':>7} | {'LegitActed(p>=0.7)':>18}\n"
        f"  {'-'*66}\n"
        f"  {'RiskScore Base':<18} | {rep.acc_base:>7.4f} | {rep.bal_acc_base:>7.4f} | {rep.brier_base:>7.4f} | {'-':>18}\n"
        f"  {'Vector-only':<18} | {rep.acc_vec:>7.4f} | {rep.bal_acc_vec:>7.4f} | {rep.brier_vec:>7.4f} | {len(rep.legit_acted_vec):>18}\n"
        f"  {'Traversal-only':<18} | {rep.acc_trav:>7.4f} | {rep.bal_acc_trav:>7.4f} | {rep.brier_trav:>7.4f} | {len(rep.legit_acted_trav):>18}\n"
        f"  {'Fused (2-hop+vec)':<18} | {rep.acc_fused:>7.4f} | {rep.bal_acc_fused:>7.4f} | {rep.brier_fused:>7.4f} | {len(rep.legit_acted_fused):>18}\n"
        f"  Precedent set changed (Fused vs Vector-only): {rep.changed_precedent_count}/{rep.n_eval} ({rep.changed_precedent_frac:.2%})\n"
        f"  Verdict flips vs RiskScore Base:\n"
        f"    Vector-only flipped {rep.vec_flips_vs_base}/{rep.n_eval} verdicts; correct on {rep.vec_flips_vs_base_correct}/{rep.vec_flips_vs_base} ({rep.vec_flips_vs_base_acc:.2%})\n"
        f"    Fused       flipped {rep.fused_flips_vs_base}/{rep.n_eval} verdicts; correct on {rep.fused_flips_vs_base_correct}/{rep.fused_flips_vs_base} ({rep.fused_flips_vs_base_acc:.2%})\n"
        f"  Verdict flips Fused vs Vector-only: {rep.fused_flips_vs_vec}/{rep.n_eval}; Fused right on {rep.fused_flips_vs_vec_correct}/{rep.fused_flips_vs_vec} ({rep.fused_flips_vs_vec_acc:.2%})"
    )

    assert rep.n_eval >= 500
    assert rep.changed_precedent_count > 100

    # Verify detector + attestation on exam case HHF-001
    det = FusedVectorDetector(k=5)
    r0 = cp.iloc[0]
    opened_0 = pd.Timestamp(r0.opened_at)
    ctx0 = CaseCtx(
        case_id=str(r0.case_id),
        flagged_txn_id=int(r0.flagged_txn_id),
        card_id=str(r0.card_id),
        customer_id=str(r0.customer_id),
        opened_at=opened_0,
        tx=tx[tx["t"] <= opened_0],
        cc=cc[pd.to_datetime(cc["closed_at"]) <= opened_0],
    )
    findings = det.run(ctx0)
    if findings:
        f0 = findings[0]
        d0 = f0.claim.to_dict()
        assert d0["how"]["query"] == "similar_cases_fused"
        assert len(d0["evidence"]["case_ids"]) > 0


def test_fail_closed_when_tg_unreachable(monkeypatch):
    tx = tx_all()
    cc = closed_cases().iloc[:50].copy()
    cc_emb = embed_closed_cases(cc, tx)
    idx = LocalVectorIndex(cc_emb, cc, tx)

    monkeypatch.setenv("ASSAY_TG", "1")
    monkeypatch.setenv("ASSAY_TG_HOST", "http://127.0.0.1:59999")
    with pytest.raises(TigerGraphUnavailableError):
        idx.similar_cases_vec(cc_emb["embedding"].iloc[0], k=3)
    with pytest.raises(TigerGraphUnavailableError):
        idx.similar_cases_fused(str(cc_emb["card_id"].iloc[0]), cc_emb["embedding"].iloc[0], k=3)


def test_gsql_files_syntax():
    schema_gsql = (ROOT / "graph" / "r2_vector_schema.gsql").read_text(encoding="utf-8")
    assert 'ALTER VERTEX ClosedCase ADD VECTOR ATTRIBUTE behaviour(DIMENSION=24, METRIC="COSINE")' in schema_gsql
    assert "/home/tigergraph/data/tg_r2/case_vectors.csv" in schema_gsql

    query_gsql = (ROOT / "graph" / "queries" / "r2_vector.gsql").read_text(encoding="utf-8")
    assert "CREATE OR REPLACE QUERY similar_cases_vec" in query_gsql
    assert "vectorSearch({ClosedCase.behaviour}, qv, k, filter=Eligible)" in query_gsql
    assert "WHERE cc.closed_at <= as_of" in query_gsql
    assert "CREATE OR REPLACE QUERY similar_cases_fused" in query_gsql
    assert "filter=CandidateSet" in query_gsql


@pytest.mark.skipif(os.environ.get("ASSAY_TG") != "1", reason="ASSAY_TG!=1; skipping live TigerGraph vector test")
def test_tigergraph_live_parity():
    from assay.tg import TigerGraphStore, gsql_file

    tx = tx_all()
    cc = closed_cases()
    cc_emb = embed_closed_cases(cc, tx)
    export_tg_vectors(cc_emb)

    gsql_file(ROOT / "graph" / "r2_vector_schema.gsql")
    gsql_file(ROOT / "graph" / "queries" / "r2_vector.gsql")

    store = TigerGraphStore()
    idx = LocalVectorIndex(cc_emb, cc, tx)
    qv = cc_emb["embedding"].iloc[0].tolist()
    local_res = idx.similar_cases_vec(cc_emb["embedding"].iloc[0], k=5, use_tg_if_enabled=False)
    tg_res = store._q("similar_cases_vec", {"qv": qv, "k": 5, "as_of": "2099-01-01 00:00:00"})
    tg_ids = {r["v_id"] for r in tg_res}
    local_ids = {r.case_id for r in local_res}
    assert len(tg_ids & local_ids) >= 3
