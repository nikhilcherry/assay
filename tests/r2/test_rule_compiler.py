"""Tests for Block B8 (Challenge 3): Policy text to detector compiler."""

from __future__ import annotations

import time

from assay.r2.frames import closed_cases, tx_all
from assay.r2.rule_compiler import (
    ADVISORY_1_TEXT,
    ADVISORY_2_TEXT,
    ADVISORY_3_TEXT,
    DEFAULT_RULES,
    POLICY_R5_TEXT,
    POLICY_R6_TEXT,
    CompiledRuleDetector,
    build_backtest_attestation,
    compare_rule_with_r1_agent,
    compile_rule,
    format_comparison_table,
)


def test_compile_all_five_rules_fast_and_deterministic():
    t0 = time.perf_counter()
    specs = [compile_rule(text, rid) for rid, text in DEFAULT_RULES.items()]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # All 5 rules compile in well under 50ms total
    assert elapsed_ms < 50.0
    by_id = {s.rule_id: s for s in specs}

    # Check R5 AST
    r5 = by_id["R5"]
    assert r5.kind == "card_sequence"
    assert r5.min_txns == 3
    assert r5.channel == "online"
    assert r5.small_amt_max == 25.0
    assert r5.window_seconds == 3600
    assert r5.followup_larger is True
    assert r5.cleared_amt_threshold == 100.0
    assert set(r5.recommended_actions) == {"DECLINE_TRANSACTION", "STEP_UP_AUTH", "BLOCK_CARD"}
    assert r5.unparsed_phrases == []

    # Check R6 AST
    r6 = by_id["R6"]
    assert r6.kind == "shared_entity"
    assert r6.min_cards == 3
    assert set(r6.shared_entities) == {"device_profile", "addr1", "R_emaildomain"}
    assert r6.require_prior_fraud is True
    assert set(r6.recommended_actions) == {"CREATE_CASE", "FILE_REPORT", "MONITOR_CONNECTED_CARDS"}
    assert r6.unparsed_phrases == []

    # Check ADV-1 AST
    adv1 = by_id["ADV-1"]
    assert adv1.kind == "shared_entity"
    assert adv1.min_cards == 3
    assert adv1.shared_entities == ["device_profile"]
    assert adv1.min_distinct_regions == 2
    assert adv1.window_seconds == 86400
    assert adv1.unparsed_phrases == []

    # Check ADV-2 AST
    adv2 = by_id["ADV-2"]
    assert adv2.kind == "shared_entity"
    assert adv2.min_cards == 2
    assert adv2.channel == "online"
    assert adv2.shared_entities == ["R_emaildomain"]
    assert adv2.window_seconds == 48 * 3600
    assert adv2.min_exposure_usd == 1000.0
    assert adv2.unparsed_phrases == []

    # Check ADV-3 AST including explicit unparsed phrase ("needs human")
    adv3 = by_id["ADV-3"]
    assert adv3.kind == "card_velocity"
    assert adv3.min_txns == 4
    assert adv3.channel == "online"
    assert adv3.window_seconds == 7200
    assert adv3.min_exposure_usd == 500.0
    assert len(adv3.unparsed_phrases) == 1
    assert "biometric voiceprint verification failure" in adv3.unparsed_phrases[0]


def test_backtest_and_compare_all_five_rules():
    tx = tx_all()
    cc = closed_cases()

    specs = []
    backtests = {}
    reports = {}

    for rid, prose in DEFAULT_RULES.items():
        spec = compile_rule(prose, rid)
        specs.append(spec)

        det = CompiledRuleDetector(spec)
        bt = det.backtest(cc, tx)
        backtests[rid] = bt

        bt_att = build_backtest_attestation(spec, bt)
        att_dict = bt_att.to_dict()
        assert att_dict["how"]["query"] == "assay.r2.rule_compiler.backtest_rule_spec"
        assert bt.n_cases == len(cc)
        assert len(bt.caught) + len(bt.missed) == (cc["outcome"] == "confirmed_fraud").sum()

        rep = compare_rule_with_r1_agent(spec, cc=cc, tx=tx)
        reports[rid] = rep
        assert rep.n_compared == 20
        assert (
            len(rep.agree_positive)
            + len(rep.agree_negative)
            + len(rep.disagree_rule_only)
            + len(rep.disagree_agent_only)
            == 20
        )
        assert len(rep.attestations) == 20

    # R5 must catch card_testing cases with high precision and match HHG-006
    bt_r5 = backtests["R5"]
    assert len(bt_r5.caught) >= 10
    assert bt_r5.lr_measured > 5.0
    assert "HHG-006" in reports["R5"].agree_positive

    # R6 must catch shared-origin cases with high precision
    bt_r6 = backtests["R6"]
    assert len(bt_r6.caught) >= 50
    assert bt_r6.lr_measured > 5.0

    table = format_comparison_table(specs, backtests, reports)
    print("\n=== B8 RULE COMPILER SUMMARY TABLE ===")
    print(table)
    for rid, rep in reports.items():
        bt = backtests[rid]
        print(
            f"\n[{rid}] compile_ms={specs[list(DEFAULT_RULES).index(rid)].compile_ms:.3f} "
            f"caught={len(bt.caught)} legit_acted={len(bt.legit_acted_against)} "
            f"precision={bt.precision:.3f} recall={bt.recall:.4f} LR={bt.lr_measured:.2f}"
        )
        print(f"  caught sample (<=10): {bt.caught[:10]}")
        print(f"  legit_acted_against:  {bt.legit_acted_against}")
        print(f"  R1 agree_positive:     {rep.agree_positive}")
        print(f"  R1 agree_negative:     {rep.agree_negative}")
        print(f"  R1 disagree_rule_only: {rep.disagree_rule_only}")
        print(f"  R1 disagree_agent_only:{rep.disagree_agent_only}")
