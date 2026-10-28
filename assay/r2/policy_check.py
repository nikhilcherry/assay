"""Independent policy linter for answer files.
Checks Fraud Operations Policy v1.0:
- Approval routes (§2)
- Case vs Report (§3a)
- Action evolution / delta tracking (§3b)
- Stopping criteria (§6)
- Tip-off prevention
- Probability/verdict consistency
- Policy Rules R1 to R10
- R1 vs R2 conflation
- Innocent Customer Rule (hard violation if BLOCK_CARD, BLOCK_ALL_CARDS, or FILE_REPORT
  is recommended for a verdict of legitimate or uncertain)
"""

from __future__ import annotations

import csv
import glob
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Violation:
    rule: str
    severity: str  # "hard", "policy", "warning"
    message: str
    json_path: str


ACTIONS = {
    "ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS",
    "GENERATE_REPORT", "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD"
}

AUTO_ACTIONS = {
    "ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "GENERATE_REPORT", "CREATE_CASE",
    "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD"
}

VERDICTS = {"fraud", "legitimate", "uncertain"}

TIPOFF_RE = re.compile(
    r"\b(sar|suspicious activity report|file[d]? a report|regulator|fincen|reported to the authorit|law enforcement)\b",
    re.I
)


def dig(d: Any, *names: str) -> tuple[Any, bool]:
    """Helper to extract named key from dictionary or one level deep."""
    if not isinstance(d, dict):
        return None, False
    for n in names:
        if n in d:
            return d[n], True
    for v in d.values():
        if isinstance(v, dict):
            for n in names:
                if n in v:
                    return v[n], True
    return None, False


def as_list(v: Any) -> list[str]:
    """Normalize action lists whether string, dict, or list of objects."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for x in (v if isinstance(v, list) else [v]):
        if isinstance(x, dict):
            x = x.get("action") or x.get("name") or x.get("type") or ""
        if isinstance(x, str) and x:
            out.append(x)
    return out


def norm_ids(v: Any) -> list[str]:
    """Normalize ID list from string, float, int, list, or dict."""
    out = []
    if v is None:
        return out
    if isinstance(v, (str, int, float)):
        v = [v]
    if isinstance(v, dict):
        v = list(v.values())
    for x in (v if isinstance(v, list) else []):
        if isinstance(x, dict):
            x = x.get("id") or x.get("txn_id") or x.get("transaction_id") or ""
        if isinstance(x, float) and float(x).is_integer():
            x = int(x)
        s = str(x).strip()
        if s and s.lower() not in ("none", "null", "nan"):
            out.append(s)
    return out


def lint(ans: dict[str, Any], case_row: Any = None) -> list[Violation]:
    """Lint an answer dictionary against Fraud Operations Policy v1.0 and grader rules."""
    violations: list[Violation] = []

    def add_hard(rule: str, msg: str, path: str) -> None:
        violations.append(Violation(rule=rule, severity="hard", message=msg, json_path=path))

    def add_policy(rule: str, msg: str, path: str) -> None:
        violations.append(Violation(rule=rule, severity="policy", message=msg, json_path=path))

    def add_warning(rule: str, msg: str, path: str) -> None:
        violations.append(Violation(rule=rule, severity="warning", message=msg, json_path=path))

    if not isinstance(ans, dict):
        add_hard("HARD", "Answer file is not a JSON object", "$")
        return violations

    case_obj, _ = dig(ans, "case", "investigation", "assessment")
    case_obj = case_obj if isinstance(case_obj, dict) else ans
    nba_obj, _ = dig(ans, "next_best_actions", "next_best_action", "actions")
    sar_obj, _ = dig(ans, "sar", "SAR", "suspicious_activity_report")

    crow_dict = {}
    if case_row is not None:
        if isinstance(case_row, dict):
            crow_dict = case_row
        elif hasattr(case_row, "to_dict"):
            crow_dict = case_row.to_dict()

    cid = ans.get("case_id")
    expected_cid = crow_dict.get("case_id")
    if expected_cid and str(cid or "").upper() != str(expected_cid).upper():
        add_hard("HARD", f"case_id mismatch: got {cid!r}, expected {expected_cid!r}", "case_id")

    verdict_raw, has_v = dig(case_obj, "verdict", "final_verdict", "decision", "disposition")
    verdict = str(verdict_raw or "").strip().lower()
    if not has_v:
        add_hard("HARD", "No verdict field", "case.verdict")
    elif verdict not in VERDICTS:
        add_hard("HARD", f"Invalid verdict {verdict_raw!r}, expected one of {sorted(VERDICTS)}", "case.verdict")

    prob_raw, has_p = dig(case_obj, "fraud_probability", "probability", "p_fraud")
    try:
        prob = float(prob_raw) if prob_raw is not None else None
    except (TypeError, ValueError):
        prob = None

    if not has_p or prob is None:
        add_hard("HARD", "Missing numeric fraud_probability", "case.fraud_probability")
    elif not (0.0 <= prob <= 1.0):
        add_hard("HARD", f"fraud_probability {prob} outside [0, 1]", "case.fraud_probability")

    pattern_raw, has_pat = dig(case_obj, "pattern", "fraud_pattern")
    pat_str = str(pattern_raw or "").strip()
    if not has_pat or not pat_str:
        add_hard("HARD", "Missing pattern field", "case.pattern")

    exp_raw, has_exp = dig(case_obj, "exposure_usd", "exposure")
    try:
        exposure = float(exp_raw) if exp_raw is not None else None
    except (TypeError, ValueError):
        exposure = None

    if not has_exp or exposure is None:
        add_hard("HARD", "Missing numeric exposure_usd", "case.exposure_usd")
    elif exposure < 0:
        add_hard("HARD", f"Negative exposure_usd {exposure}", "case.exposure_usd")

    aff_txns = norm_ids(dig(case_obj, "affected_txn_ids", "affected_transaction_ids", "affected_transactions")[0])
    first_txns = norm_ids(dig(case_obj, "first_suspicious_txn_id", "first_fraud_txn_id")[0])

    raw_final = dig(nba_obj, "final", "after_evidence", "final_actions")[0] if isinstance(nba_obj, dict) else nba_obj
    fin_actions = [a.strip().upper() for a in as_list(raw_final)]

    raw_initial = dig(nba_obj, "initial", "before_evidence", "initial_actions")[0] if isinstance(nba_obj, dict) else None
    ini_actions = [a.strip().upper() for a in as_list(raw_initial)]

    routes = {}
    if isinstance(raw_final, list):
        for x in raw_final:
            if isinstance(x, dict):
                an = str(x.get("action") or x.get("name") or "").strip().upper()
                rt = x.get("route") or x.get("approval_route") or x.get("approval")
                if an and rt:
                    routes[an] = str(rt).strip().upper()

    if not fin_actions:
        add_hard("HARD", "next_best_actions.final is empty or absent", "next_best_actions.final")

    for a in set(fin_actions + ini_actions) - ACTIONS:
        add_hard("HARD", f"Unknown action {a!r} in next_best_actions", "next_best_actions")

    # Innocent Customer Rule (HARD)
    if verdict in ("legitimate", "uncertain"):
        disallowed = {"BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"}
        found_disallowed = set(fin_actions + ini_actions) & disallowed
        if found_disallowed:
            add_hard(
                "INNOCENT_CUSTOMER",
                f"Innocent customer rule breach: action(s) {sorted(found_disallowed)} not permitted on verdict '{verdict}'",
                "next_best_actions"
            )

    if verdict == "legitimate":
        if aff_txns:
            add_hard("HARD", "Verdict is legitimate but affected_txn_ids is non-empty", "case.affected_txn_ids")
        if exposure is not None and exposure > 0:
            add_hard("HARD", f"Verdict is legitimate but exposure_usd {exposure} > 0", "case.exposure_usd")

    if verdict == "fraud" and not aff_txns:
        add_hard("HARD", "Verdict is fraud but affected_txn_ids is empty", "case.affected_txn_ids")

    if first_txns and aff_txns and first_txns[0] not in aff_txns:
        add_hard("HARD", f"first_suspicious_txn_id {first_txns[0]} is not in affected_txn_ids", "case.first_suspicious_txn_id")

    if prob is not None and verdict == "fraud" and prob < 0.5:
        add_warning("VERDICT_PROB_MISMATCH", f"Verdict is fraud but fraud_probability is low ({prob})", "case.fraud_probability")

    if prob is not None and verdict == "legitimate" and prob > 0.5:
        add_warning("VERDICT_PROB_MISMATCH", f"Verdict is legitimate but fraud_probability is high ({prob})", "case.fraud_probability")

    # Approval routes (§2)
    blocks = {"BLOCK_CARD", "BLOCK_ALL_CARDS"} & set(fin_actions)
    if blocks and exposure is not None:
        for b in sorted(blocks):
            route = routes.get(b) or str(dig(ans, "approval_route", "expected_route_for_block")[0] or "").upper() or None
            want_route = "L2" if (b == "BLOCK_ALL_CARDS" or exposure > 2500) else "L1"
            if route and route != want_route:
                add_policy("ROUTE", f"{b} at exposure ${exposure:.2f} needs route {want_route}, got {route}", "next_best_actions.final")
            elif not route:
                add_warning("ROUTE", f"{b} recommended without stated approval route (expected {want_route})", "next_best_actions.final")

    for a, rt in routes.items():
        if a in AUTO_ACTIONS and rt not in ("AUTO", "", None):
            add_warning("ROUTE", f"{a} is an auto action but is routed {rt}", "next_best_actions.final")

    if "FILE_REPORT" in fin_actions:
        rt = routes.get("FILE_REPORT")
        if rt and rt != "L2":
            add_policy("ROUTE", f"FILE_REPORT requires route L2, got {rt}", "next_best_actions.final")

    # R1 vs R2 Conflation Checks
    trig_type = crow_dict.get("trigger_type")
    is_denial = trig_type == "customer_report"

    # R2 Situation: Customer denied transaction (R2)
    if is_denial:
        # R2 requires immediate block and case without verification step
        verification_in_actions = bool({"VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH"} & set(ini_actions + fin_actions))
        if verification_in_actions and (verdict == "fraud" or blocks):
            add_policy(
                "R1_R2_CONFLATION",
                "Conflating R1 with R2: Customer already denied the charge (R2); verification step (VERIFY_WITH_CUSTOMER/STEP_UP_AUTH) is unnecessary",
                "next_best_actions"
            )
        if verdict == "fraud":
            if "BLOCK_CARD" not in fin_actions:
                add_policy("R2", "R2 breach: customer denial requires BLOCK_CARD", "next_best_actions.final")
            if "CREATE_CASE" not in fin_actions:
                add_policy("R2", "R2 breach: customer denial requires CREATE_CASE", "next_best_actions.final")

    # R1 Situation: Weak signal (p < 0.70, no denial)
    if prob is not None and blocks and prob < 0.70 and not is_denial:
        prior_verify = bool({"VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH"} & set(ini_actions))
        ev_reqs = ans.get("evidence_requests") or []
        if isinstance(ev_reqs, list):
            for req in ev_reqs:
                if isinstance(req, dict) and req.get("type") in ("customer_validation", "step_up_auth"):
                    prior_verify = True
        if not prior_verify:
            add_policy(
                "R1_R2_CONFLATION",
                f"Conflating R1 with R2: Block recommended at probability {prob:.2f} on a single weak signal without prior verification step (R1)",
                "next_best_actions"
            )

    # R8: Escalate when uncertain and exposed
    if verdict == "uncertain":
        if exposure is not None and exposure > 500:
            if "ESCALATE_TO_ANALYST" not in fin_actions:
                add_policy("R8", f"R8 breach: uncertain verdict with exposure ${exposure:.2f} requires ESCALATE_TO_ANALYST", "next_best_actions.final")
        elif "ESCALATE_TO_ANALYST" not in fin_actions:
            add_warning("R8", "Uncertain verdict recommended without ESCALATE_TO_ANALYST", "next_best_actions.final")

    # R9: Undocumented patterns
    if pat_str.lower() in ("undocumented", "unknown", "novel", "other"):
        missing_r9 = {"CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST"} - set(fin_actions)
        if missing_r9:
            add_policy("R9", f"R9 breach: undocumented pattern requires {sorted(missing_r9)}", "next_best_actions.final")
        pat_desc = str(dig(case_obj, "pattern_description", "description")[0] or "").strip()
        if not pat_desc:
            add_policy("R9", "R9 breach: pattern_description must be provided for undocumented pattern", "case.pattern_description")

    # R10: BLOCK_ALL_CARDS requirement
    if "BLOCK_ALL_CARDS" in fin_actions:
        add_policy("R10", "R10 breach: BLOCK_ALL_CARDS requires confirmed compromise across 2+ cards or credentials", "next_best_actions.final")

    # §3a Case vs Report
    conn_cards = norm_ids(dig(case_obj, "connected_card_ids", "connected_cards", "linked_cards")[0])
    conn_devs = norm_ids(dig(case_obj, "connected_device_profiles", "connected_devices")[0])
    in_final = "FILE_REPORT" in fin_actions

    if verdict == "fraud":
        if "CREATE_CASE" not in fin_actions:
            add_policy("3a", "§3a breach: verdict is fraud but CREATE_CASE is missing", "next_best_actions.final")
        need_report = (exposure is not None and exposure > 1000) or bool(conn_cards) or bool(conn_devs) or (pat_str.lower() in ("undocumented", "unknown", "novel"))
        if need_report and not in_final:
            add_policy("3a", "§3a breach: exposure > $1,000, connected entities, or undocumented pattern requires FILE_REPORT", "next_best_actions.final")
        elif not need_report and in_final and (exposure is not None and exposure <= 1000) and not conn_cards and not conn_devs:
            add_warning("3a", "FILE_REPORT recommended although §3a does not appear to require it", "next_best_actions.final")

    sar_file = None
    if isinstance(sar_obj, dict):
        sf, _ = dig(sar_obj, "file", "required", "filed", "should_file")
        if isinstance(sf, bool):
            sar_file = sf
        elif isinstance(sf, str):
            sar_file = sf.strip().lower() in ("yes", "true", "required", "file")
    elif isinstance(sar_obj, bool):
        sar_file = sar_obj

    if sar_file is not None and sar_file != in_final:
        add_policy("3a", f"sar.file={sar_file} but FILE_REPORT {'is' if in_final else 'is not'} in final actions", "sar.file")

    if sar_file:
        narr = str(dig(sar_obj if isinstance(sar_obj, dict) else {}, "narrative", "text")[0] or "").strip()
        if not narr:
            add_policy("3a", "SAR filing required but narrative is empty", "sar.narrative")
        elif len(narr.split()) < 40:
            add_warning("3a", f"SAR narrative is thin ({len(narr.split())} words, expected >= 40)", "sar.narrative")
        subj = norm_ids(dig(sar_obj if isinstance(sar_obj, dict) else {}, "subjects")[0])
        if not subj:
            add_policy("3a", "SAR filing required but subjects list is empty", "sar.subjects")

    # §3b Action delta
    if ini_actions and fin_actions:
        if set(ini_actions) != set(fin_actions):
            changed, _ = dig(nba_obj if isinstance(nba_obj, dict) else {}, "what_changed", "change", "delta")
            ch_str = str(changed or "").strip().lower()
            if not ch_str or ch_str in ("nothing", "none"):
                add_policy("3b", "§3b breach: final actions differ from initial but what_changed is empty or 'nothing'", "next_best_actions.what_changed")

    # Tip-off (§6 / Guidelines)
    ev_reqs, _ = dig(ans, "evidence_requests", "evidence_request")
    if ev_reqs is not None:
        blob = json.dumps(ev_reqs)
        if TIPOFF_RE.search(blob):
            add_policy("TIPOFF", "Tip-off risk: evidence_requests text mentions SAR/report/regulator; customer-visible text must never reveal this", "evidence_requests")

    # §6 Stopping
    stop_reason, _ = dig(ans, "stop_reason", "reason_for_stopping")
    if not stop_reason or not str(stop_reason).strip():
        add_policy("6", "§6 breach: stop_reason field is missing or empty", "stop_reason")

    return violations


def compare_with_grader(case_files: list[str], pack_map: dict[str, dict]) -> dict[str, Any]:
    """Run answer files against grade.py via subprocess and return disagreement details."""
    grader_script = None
    for p in ["data/round2/share/grader/grade.py", "grader/grade.py"]:
        if Path(p).exists():
            grader_script = Path(p)
            break

    if not grader_script:
        return {"error": "grade.py not found"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cases_dir = tmp_path / "teams" / "assay" / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        for cf in case_files:
            (cases_dir / Path(cf).name).write_bytes(Path(cf).read_bytes())

        pack_path = tmp_path / "case_pack.csv"
        with open(pack_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "opened_at", "trigger_type", "trigger_text", "flagged_txn_id", "card_id", "customer_id", "risk_score"])
            for cf in case_files:
                cid = Path(cf).stem
                meta = pack_map.get(cid, {})
                writer.writerow([
                    cid,
                    meta.get("opened_at", "2017-01-01 00:00:00"),
                    meta.get("trigger_type", "analyst_request"),
                    meta.get("trigger_text", ""),
                    meta.get("flagged_txn_id", "1000"),
                    meta.get("card_id", "C00001-K1"),
                    meta.get("customer_id", "C00001"),
                    meta.get("risk_score", "0.5")
                ])

        out_json = tmp_path / "report.json"
        cmd = [
            sys.executable, str(grader_script),
            "--teams-dir", str(tmp_path / "teams"),
            "--case-pack", str(pack_path),
            "--out", str(out_json)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)

        if not out_json.exists():
            return {"error": f"grader failed: {res.stderr}"}

        with open(out_json, "r", encoding="utf-8") as f:
            report = json.load(f)

        return report.get("teams", {}).get("assay", {}).get("cases", {})


def main() -> None:
    case_files = sorted(glob.glob("cases/HHG-*.json") or glob.glob("cases/*.json"))
    if not case_files:
        print("No case files found in cases/")
        return

    print(f"==================================================")
    print(f"LINTING {len(case_files)} CASE FILES")
    print(f"==================================================")

    pack_map = {}
    for p in ["data/case_pack.csv", "data/round2/share/cases/case_pack_final.csv", "cases/case_pack_final.csv"]:
        if Path(p).exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pack_map[row["case_id"]] = row
            break

    lint_results: dict[str, list[Violation]] = {}
    for cf in case_files:
        cid = Path(cf).stem
        with open(cf, "r", encoding="utf-8") as f:
            ans = json.load(f)
        crow = pack_map.get(cid)
        v_list = lint(ans, crow)
        lint_results[cid] = v_list

        print(f"\n--- {cid} ---")
        if not v_list:
            print("  Clean (no violations)")
        else:
            for v in v_list:
                print(f"  [{v.severity.upper():7s}] Rule: {v.rule:16s} Path: {v.json_path:25s} Msg: {v.message}")

    grader_cases = compare_with_grader(case_files, pack_map)
    print("\n==================================================")
    print("DISAGREEMENT TABLE (policy_check.py vs grade.py)")
    print("==================================================")
    print(f"{'CASE_ID':10s} | {'LINT HARD':10s} | {'LINT POL':10s} | {'GRD HARD':10s} | {'GRD POL':10s} | {'DISAGREEMENT'}")
    print("-" * 75)

    disagreements_found = 0
    for cid in sorted(lint_results.keys()):
        v_list = lint_results[cid]
        l_hard = sum(1 for v in v_list if v.severity == "hard")
        l_pol = sum(1 for v in v_list if v.severity == "policy")

        g_findings = grader_cases.get(cid, {}).get("findings", [])
        g_hard = sum(1 for cls, _, _ in g_findings if cls == "hard")
        g_pol = sum(1 for cls, _, _ in g_findings if cls == "policy")

        disagree = (l_hard != g_hard) or (l_pol != g_pol)
        if disagree:
            disagreements_found += 1
            status = "YES"
        else:
            status = "NO"

        print(f"{cid:10s} | {l_hard:10d} | {l_pol:10d} | {g_hard:10d} | {g_pol:10d} | {status}")

    print(f"\nTotal cases checked: {len(lint_results)}, Disagreements: {disagreements_found}")


if __name__ == "__main__":
    main()
