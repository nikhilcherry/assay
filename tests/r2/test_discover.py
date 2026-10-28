"""Tests for B7 -- Self-growing detective (`assay/r2/discover.py`)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from assay.r2.discover import (
    PREDICATE_NAMES,
    PROPOSED_DIR,
    approve,
    featurize_closed_cases,
    featurize_legit_card_days_jul_oct,
    is_approved,
    load_all_approved,
    load_approved_detector,
    mine_and_propose,
    reset_approvals,
    verify_backtest_rule,
)
from assay.r2.finding import REGISTRY, CaseCtx
from assay.r2.frames import closed_cases, tx_all


def test_discover_mine_validate_and_approval_gate():
    reset_approvals()

    # 1. Featurize all 5,585 closed cases and 6,000 Jul-Oct legitimate card-days
    feats = featurize_closed_cases()
    assert len(feats) == 5585
    assert len(PREDICATE_NAMES) >= 40

    legit_cd = featurize_legit_card_days_jul_oct()
    assert len(legit_cd) >= 5000

    # 2. Mine top 5 conjunctions on Jul-Oct and validate out-of-time on Nov-Dec + legit card-days
    proposals = mine_and_propose(top_k=5, max_legit_train=2, max_legit_sample_rate=0.015)
    assert len(proposals) == 5

    print("\n=== TOP 5 DISCOVERED DETECTOR PROPOSALS (Jul-Oct Train | Nov-Dec Val | 6,000 Legit Card-Days) ===")
    for i, p in enumerate(proposals, 1):
        tr = p.train_backtest
        vl = p.val_backtest
        fl = p.full_backtest
        print(
            f"[{i}] {p.name}: {' AND '.join(p.predicates)}\n"
            f"    Desc: {p.description}\n"
            f"    Train (Jul-Oct, n={tr.n_cases}): caught={len(tr.caught)} ({p.train_fraud_rate:.2%}), "
            f"cleared_acted={len(tr.legit_acted_against)}\n"
            f"    Val   (Nov-Dec, n={vl.n_cases}): caught={len(vl.caught)} ({p.val_fraud_rate:.2%}), "
            f"cleared_acted={len(vl.legit_acted_against)}\n"
            f"    Legit Card-Days (Jul-Oct, n={p.legit_sample_n}): fired={p.legit_sample_fired} "
            f"({p.legit_sample_rate:.3%}), LR(fraud/legit_cd)={p.lr_vs_legit_sample:.2f}\n"
            f"    Full  (5585 cc): caught={len(fl.caught)} (undoc={len(p.undocumented_caught)}, "
            f"residual={len(p.residual_caught)}), cleared_acted={len(fl.legit_acted_against)}\n"
            f"    Undocumented IDs caught: {p.undocumented_caught}\n"
            f"    Cleared acted against IDs: {fl.legit_acted_against}"
        )
        # Verify firing rate on the legit card-day sample is <= 2%
        assert p.legit_sample_rate <= 0.02, f"Proposal {p.name} legit card-day firing rate > 2%"

    # 3. Verify files are written to data/stage/r2/proposed/ and NOT in active toolkit
    toolkit_dir = Path("assay/r2/detectors")
    for p in proposals:
        src_path = Path(p.source_path)
        json_path = Path(p.proposal_json_path)
        assert src_path.exists() and src_path.parent == PROPOSED_DIR
        assert json_path.exists() and json_path.parent == PROPOSED_DIR
        if toolkit_dir.exists():
            assert not (toolkit_dir / src_path.name).exists()

        # Verify JSON contents and attestation reproducibility
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["name"] == p.name
        reran = verify_backtest_rule(**payload["attestation"]["how"]["params"])
        assert reran["caught"] == payload["full_backtest"]["caught"]
        assert reran["legit_acted_against"] == payload["full_backtest"]["legit_acted_against"]
        assert reran["missed"] == payload["full_backtest"]["missed"]
        assert reran["legit_card_day_fired"] == payload["legit_card_day_sample"]["fired_count"]

    # At least one discovered rule must catch undocumented cases and generalize to Nov-Dec validation
    total_undoc = sum(len(p.undocumented_caught) for p in proposals)
    total_val_caught = sum(len(p.val_backtest.caught) for p in proposals)
    assert total_undoc > 0, "Expected discovered rules to catch at least one undocumented fraud case"
    assert total_val_caught > 0, "Expected discovered rules to catch fraud cases in Nov-Dec validation window"

    # 4. Assert unapproved proposals are NOT in REGISTRY and cannot be loaded
    top_prop = proposals[0]
    assert top_prop.name not in REGISTRY
    assert not is_approved(top_prop.name)

    with pytest.raises(PermissionError, match="unapproved"):
        load_approved_detector(top_prop.name)

    # Even if someone imports the generated file directly, its `name` is empty so REGISTRY is untouched
    spec = importlib.util.spec_from_file_location("temp_unapproved", top_prop.source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert top_prop.name not in REGISTRY

    # 5. Assert L1 token or invalid token is rejected by server-side check
    with pytest.raises(PermissionError, match="L2"):
        approve(top_prop.name, "L1-LEAD-ROHAN-2C4D")

    with pytest.raises(PermissionError, match="Unrecognised"):
        approve(top_prop.name, "FAKE-TOKEN-123")

    assert not is_approved(top_prop.name)
    assert load_all_approved() == []

    # 6. Approve with valid L2 token and verify loading + execution on a closed case
    rec = approve(top_prop.name, "L2-MGR-PRIYA-9F3A")
    assert rec["role"] == "L2"
    assert rec["detector"] == top_prop.name
    assert is_approved(top_prop.name)

    det_cls = load_approved_detector(top_prop.name)
    assert top_prop.name in REGISTRY
    assert REGISTRY[top_prop.name] is det_cls

    det = det_cls()
    cc = closed_cases()
    tx = tx_all()
    bt = det.backtest(cc, tx)
    assert bt.caught == top_prop.full_backtest.caught
    assert bt.legit_acted_against == top_prop.full_backtest.legit_acted_against

    # Run `det.run(ctx)` as-of on the first caught case
    target_case_id = top_prop.full_backtest.caught[0]
    c_row = cc[cc["case_id"] == target_case_id].iloc[0]
    t_open = pd.Timestamp(c_row["opened_at"])
    ctx = CaseCtx(
        case_id=str(c_row["case_id"]),
        flagged_txn_id=int(c_row["txn_list"][0]),
        card_id=str(c_row["card_id"]),
        customer_id=str(c_row["customer_id"]),
        opened_at=t_open,
        tx=tx[tx["t"] <= t_open],
        cc=cc[pd.to_datetime(cc["closed_at"]) <= t_open],
    )
    findings = det.run(ctx)
    assert len(findings) == 1
    assert findings[0].detector == top_prop.name
    assert findings[0].pattern == top_prop.pattern
    assert findings[0].affected_txn_ids

    # Clean up approval state at end of test
    reset_approvals()
    assert top_prop.name not in REGISTRY
