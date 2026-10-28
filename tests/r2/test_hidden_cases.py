"""BLOCK N9 -- Hidden-case finder & policy-invariant harness.

Loads the 25 executable hidden cases derived from data/round2/orch/outputs/debate/final.json,
validates fixture consistency in `test_fixtures_consistent` first (deriving/checking card_id
and customer_id against tx_all() except when `designed_mismatch` is True), then runs all 25
cases through a single session-scoped R2Investigator instance, prints the full PASS/FAIL
table, and checks policy-level invariants + assay.validate.validate + official grader.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import time

import pandas as pd
import pytest

from assay.data import ROOT
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import R2, closed_cases, tx_all
from assay.r2.investigate_r2 import R2Investigator
from assay.r2.store_r2 import R2LocalStore, r2_pattern_model
from assay.validate import validate
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).parent))
from fake_secrets import FAKE, fill  # noqa: E402

HIDDEN_JSON = ROOT / "tests" / "r2" / "hidden_cases.json"
GRADER_PY = R2 / "grader" / "grade.py"


def _load_grader():
    spec = importlib.util.spec_from_file_location("r2_official_grader", GRADER_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def grader_ctx():
    mod = _load_grader()
    tx_map: dict[str, dict] = {}
    with open(R2 / "transactions_2017q1.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            tx_map[r["TransactionID"]] = {
                "amt": float(r["TransactionAmt"]),
                "dt": int(r["TransactionDT"]),
            }
    return mod, tx_map


@pytest.fixture(scope="session")
def investigator():
    store = R2LocalStore()
    return R2Investigator(store, r2_pattern_model(), writer=store.write_case)


@pytest.fixture(scope="session")
def hidden_cases() -> list[dict]:
    data = fill(json.loads(HIDDEN_JSON.read_text()))
    assert len(data) == 25, f"Expected 25 hidden cases, got {len(data)}"
    return data


def _series_from_row(row: dict, tx_idx: pd.DataFrame | None = None, designed_mismatch: bool = False) -> pd.Series:
    d = dict(row)
    fid = int(d["flagged_txn_id"])
    if tx_idx is not None and fid in tx_idx.index and not designed_mismatch:
        tx_row = tx_idx.loc[fid]
        d["customer_id"] = str(tx_row["customer_id"])
        d["card_id"] = str(tx_row["card_id"])
    if d.get("risk_score") is None:
        d["risk_score"] = float("nan")
    return pd.Series(d)


def _check_invariant(
    hc: dict,
    inv: R2Investigator,
    grader_mod,
    jan_tx_map: dict[str, dict],
    jan_txn_int_set: set[int],
    tx_idx: pd.DataFrame,
) -> tuple[bool, str, dict]:
    """Run one hidden case and evaluate its policy-level invariant.

    Returns (passed, observed_summary, detail_dict).
    """
    cid = hc["case_id"]
    inv_name = hc["invariant"]
    designed_mismatch = bool(hc.get("designed_mismatch", False))
    row = hc["row"]
    s = _series_from_row(row, tx_idx=tx_idx, designed_mismatch=designed_mismatch)

    # Special case: missing transaction fault
    if inv_name == "missing_txn_clear_error":
        try:
            ans = inv.run(s)
            return False, f"returned verdict={ans['case']['verdict']} instead of error", {"error": None}
        except ValueError as exc:
            msg = str(exc)
            ok = str(row["flagged_txn_id"]) in msg and cid in msg
            return ok, f"ValueError: {msg[:70]}", {"error": msg}
        except Exception as exc:
            return False, f"unexpected {type(exc).__name__}: {exc}", {"error": str(exc)}

    # Special case: clock skew (opened_at < flagged_txn.t)
    if inv_name == "clock_skew_safe_or_clear_error":
        try:
            ans = inv.run(s)
            errs = validate(ans, known_txns=jan_txn_int_set)
            g_findings = grader_mod.grade_case(cid, ans, {k: str(v or "") for k, v in s.to_dict().items()}, jan_tx_map)
            hard_pol = [f for f in g_findings if f[0] in ("hard", "policy")]
            ok = not errs and not hard_pol
            c = ans["case"]
            obs = f"verdict={c['verdict']} p={c['fraud_probability']:.2f} errs={len(errs)} gp={len(hard_pol)}"
            return ok, obs, {"ans": ans, "validate_errs": errs, "grader": hard_pol}
        except ValueError as exc:
            return True, f"ValueError(clean): {str(exc)[:60]}", {"error": str(exc)}
        except Exception as exc:
            return False, f"unhandled {type(exc).__name__}: {exc}", {"error": str(exc)}

    t0 = time.time()
    ans = inv.run(s)
    elapsed = time.time() - t0
    c = ans["case"]
    fin_actions = [x["action"] for x in ans["next_best_actions"]["final"]]
    ini_actions = [x["action"] for x in ans["next_best_actions"]["initial"]]
    routes = {x["action"]: x["route"] for x in ans["next_best_actions"]["final"]}

    val_errs = validate(ans, known_txns=jan_txn_int_set)
    meta = {k: ("" if pd.isna(v) else str(v)) for k, v in s.to_dict().items()}
    g_findings = grader_mod.grade_case(cid, ans, meta, jan_tx_map)
    hard_pol = [f for f in g_findings if f[0] in ("hard", "policy")]

    reasons: list[str] = []
    if val_errs:
        reasons.append(f"validate:{val_errs}")
    if hard_pol:
        reasons.append(f"grader:{[(x[0], x[1]) for x in hard_pol]}")

    # Universal policy invariants checked on every completed answer
    if c["verdict"] == "legitimate":
        bad_legit = {"BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"} & set(fin_actions)
        if bad_legit:
            reasons.append(f"legit_blocked:{sorted(bad_legit)}")
    if c["verdict"] == "uncertain" and c["exposure_usd"] > 500:
        if "ESCALATE_TO_ANALYST" not in fin_actions:
            reasons.append("r8_missing_escalate")
    if c["verdict"] == "fraud" and c["connected_card_ids"]:
        if "FILE_REPORT" not in fin_actions or not ans["sar"]["file"]:
            reasons.append("3a_connected_missing_sar")

    # Specific invariant checks
    if "customer_denial_r2" in inv_name:
        if not ({"BLOCK_CARD", "CREATE_CASE"} <= set(fin_actions)):
            reasons.append(f"r2_missing_block_or_case:{fin_actions}")

    if "bounded" in inv_name:
        if elapsed > 30.0:
            reasons.append(f"slow:{elapsed:.1f}s")

    if "jan_only_affected_txns" in inv_name:
        non_jan = [t for t in c["affected_txn_ids"] if int(t) not in jan_txn_int_set]
        if non_jan:
            reasons.append(f"non_jan_txns:{non_jan}")

    if "legit_never_blocks_or_files_sar" in inv_name or "legit_safe" in inv_name:
        bad = {"BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"} & set(fin_actions)
        if bad:
            reasons.append(f"harmed_legit:{sorted(bad)}")

    if "r10_check" in inv_name:
        if "BLOCK_ALL_CARDS" in fin_actions:
            cc = closed_cases()
            prior_cards = set(
                cc[(cc.customer_id == str(s["customer_id"])) & (cc.outcome == "confirmed_fraud")].card_id
            ) | {str(s["card_id"])}
            if len(prior_cards) < 2:
                reasons.append("r10_unjustified_block_all")

    if "high_exposure_denial_l2_and_sar" in inv_name:
        if "BLOCK_CARD" not in fin_actions or "CREATE_CASE" not in fin_actions or "FILE_REPORT" not in fin_actions:
            reasons.append(f"missing_r2_3a_actions:{fin_actions}")
        if c["exposure_usd"] > 2500 and routes.get("BLOCK_CARD") != "L2":
            reasons.append(f"block_route:{routes.get('BLOCK_CARD')}!=L2")
        if routes.get("FILE_REPORT") != "L2":
            reasons.append(f"sar_route:{routes.get('FILE_REPORT')}!=L2")

    if "no_secret_leak" in inv_name:
        dumped = json.dumps(ans)
        for secret in (FAKE["AWS_KEY"], FAKE["SK_KEY"], FAKE["CARD"]):
            if secret in dumped:
                reasons.append(f"leaked_secret:{secret}")

    if "hostile_invariance" in inv_name and hc.get("neutral_trigger_text"):
        neutral_row = dict(row)
        neutral_row["trigger_text"] = hc["neutral_trigger_text"]
        ans_neutral = inv.run(_series_from_row(neutral_row, tx_idx=tx_idx, designed_mismatch=designed_mismatch))
        cn = ans_neutral["case"]
        fin_neutral = [x["action"] for x in ans_neutral["next_best_actions"]["final"]]
        if (
            c["verdict"] != cn["verdict"]
            or fin_actions != fin_neutral
            or abs(c["fraud_probability"] - cn["fraud_probability"]) > 1e-6
        ):
            reasons.append(
                f"hostile_diff:({c['verdict']},{c['fraud_probability']:.2f},{fin_actions})"
                f"!=({cn['verdict']},{cn['fraud_probability']:.2f},{fin_neutral})"
            )

    if inv_name == "state_isolation_and_idempotence":
        ans2 = inv.run(s)
        c2 = ans2["case"]
        fin2 = [x["action"] for x in ans2["next_best_actions"]["final"]]
        if (
            c["verdict"] != c2["verdict"]
            or abs(c["fraud_probability"] - c2["fraud_probability"]) > 1e-6
            or c["connected_card_ids"] != c2["connected_card_ids"]
            or fin_actions != fin2
        ):
            reasons.append("non_idempotent_run")

    ok = len(reasons) == 0
    obs = (
        f"v={c['verdict']} p={c['fraud_probability']:.2f} exp=${c['exposure_usd']:.2f} "
        f"conn={len(c['connected_card_ids'])} sar={ans['sar']['file']} act={fin_actions}"
    )
    if reasons:
        obs += " | FAIL: " + "; ".join(reasons)
    return ok, obs, {
        "verdict": c["verdict"],
        "fraud_probability": c["fraud_probability"],
        "exposure_usd": c["exposure_usd"],
        "connected_card_ids": c["connected_card_ids"],
        "sar_file": ans["sar"]["file"],
        "initial_actions": ini_actions,
        "final_actions": fin_actions,
        "validate_errs": val_errs,
        "grader_hard_policy": hard_pol,
        "reasons": reasons,
        "latency_s": round(elapsed, 3),
    }


def test_fixtures_consistent(hidden_cases: list[dict]):
    """Validate fixture construction against tx_all() before running any investigator invariants."""
    df = tx_all()
    tx_idx = df.set_index("TransactionID")
    print("\n=== N9 HIDDEN CASE CONSTRUCTION QUERIES ===")
    for hc in hidden_cases:
        cid = hc["case_id"]
        q = hc["construction_query"]
        row = hc["row"]
        fid = int(row["flagged_txn_id"])
        designed_mismatch = bool(hc.get("designed_mismatch", False))
        print(f"{cid} ({hc['debate_id']}): {q} [designed_mismatch={designed_mismatch}]")
        if hc["invariant"] == "missing_txn_clear_error":
            assert fid not in tx_idx.index, f"{cid} expects missing txn {fid}"
            continue
        assert fid in tx_idx.index, f"{cid} flagged_txn_id {fid} not found in tx_all()"
        tx_row = tx_idx.loc[fid]
        assert str(tx_row["customer_id"]) == str(row["customer_id"]), (
            f"{cid} customer_id mismatch {tx_row['customer_id']!r} != {row['customer_id']!r}"
        )
        if designed_mismatch:
            assert str(tx_row["card_id"]) != str(row["card_id"]), (
                f"{cid} marked designed_mismatch=True but card_id matches {tx_row['card_id']!r}"
            )
        else:
            assert str(tx_row["card_id"]) == str(row["card_id"]), (
                f"{cid} card_id mismatch {tx_row['card_id']!r} != {row['card_id']!r}"
            )


def test_run_all_25_hidden_cases(hidden_cases: list[dict], investigator: R2Investigator, grader_ctx):
    """Execute all 25 hidden cases on the shared R2Investigator, print full PASS/FAIL table, and attest."""
    grader_mod, jan_tx_map = grader_ctx
    jan_txn_int_set = {int(k) for k in jan_tx_map}
    tx_idx = tx_all().set_index("TransactionID")

    results = []
    passed_cids = []
    failed_cids = []
    used_txns = []
    used_cards = []
    used_custs = []

    print("\n=== N9 HIDDEN CASES EXECUTION TABLE ===")
    header = f"{'CASE':<8} | {'DEBATE':<8} | {'INVARIANT':<38} | {'STATUS':<6} | {'OBSERVED'}"
    print(header)
    print("-" * len(header))

    for hc in hidden_cases:
        cid = hc["case_id"]
        row = hc["row"]
        if hc["invariant"] != "missing_txn_clear_error":
            used_txns.append(int(row["flagged_txn_id"]))
        used_cards.append(str(row["card_id"]))
        used_custs.append(str(row["customer_id"]))

        ok, obs, detail = _check_invariant(hc, investigator, grader_mod, jan_tx_map, jan_txn_int_set, tx_idx)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed_cids.append(cid)
        else:
            failed_cids.append((cid, hc["scenario"], hc["invariant"], obs, detail))
        results.append((cid, hc["debate_id"], hc["invariant"], status, obs, detail))
        print(f"{cid:<8} | {hc['debate_id']:<8} | {hc['invariant']:<38} | {status:<6} | {obs}")

    att = Attestation(
        what=f"Executed 25 hidden cases on R2Investigator: {len(passed_cids)} passed, {len(failed_cids)} failed policy invariants.",
        evidence=Evidence(
            txn_ids=sorted(set(used_txns)),
            card_ids=sorted(set(used_cards)),
            customer_ids=sorted(set(used_custs)),
            case_ids=[hc["case_id"] for hc in hidden_cases],
        ),
        query="tests.r2.test_hidden_cases.run_all_25",
        params={"n_cases": len(hidden_cases), "passed": len(passed_cids), "failed": len(failed_cids)},
    )
    att_dict = att.to_dict()
    assert set(att_dict.keys()) == {"what", "evidence", "how", "when"}
    assert len(results) == 25
    print(f"\nATTESTATION: {att.to_json()}")
    assert not failed_cids, f"{len(failed_cids)} hidden case invariant(s) violated: {[x[0] for x in failed_cids]}"
