"""Schema and consistency checks for an answer file, from the brief's Answer
Format section. Run on every file before it is written."""

from __future__ import annotations

from assay.policy import Action, Route, route_for

STATUS = {"open", "closed_fraud", "closed_legitimate", "escalated"}
VERDICT = {"fraud", "legitimate", "uncertain"}
PATTERN = {"card_testing", "card_not_present_fraud", "card_not_present_new_device", "out_of_region_use",
           "account_takeover", "undocumented", "none"}
REQ = {"customer_validation", "step_up_auth", "analyst_info"}
SOURCES = {"graph", "document", "customer", "external"}


def validate(a: dict, known_txns: set | None = None) -> list[str]:
    e = []
    c = a["case"]
    for k in ("case_id", "case", "evidence_requests", "next_best_actions", "sar", "stop_reason", "tool_calls",
              "tokens", "latency_s"):
        if k not in a:
            e.append(f"missing {k}")
    if c["status"] not in STATUS: e.append("status")
    if c["verdict"] not in VERDICT: e.append("verdict")
    if c["pattern"] not in PATTERN: e.append("pattern")
    if not 0 <= c["fraud_probability"] <= 1: e.append("probability")
    if c["pattern"] == "undocumented" and not c["pattern_description"]: e.append("pattern_description")
    if c["verdict"] == "legitimate" and (c["affected_txn_ids"] or c["exposure_usd"] or a["sar"]["file"]):
        e.append("legitimate case carries fraud fields")
    for ev in c["evidence"]:
        if ev["source"] not in SOURCES: e.append(f"evidence source {ev['source']}")
    for r in a["evidence_requests"]:
        if r["type"] not in REQ: e.append("request type")
    fin = a["next_best_actions"]["final"]
    for part in ("initial", "final"):
        for x in a["next_best_actions"][part]:
            act = Action(x["action"])
            if Route(x["route"]) is not route_for(act, c["exposure_usd"]):
                e.append(f"route {x['action']}")
    has_report = any(x["action"] == "FILE_REPORT" for x in fin)
    if has_report != a["sar"]["file"]: e.append("sar.file disagrees with FILE_REPORT")
    if a["sar"]["file"] and not a["sar"]["narrative"]: e.append("sar narrative")
    if not a["sar"]["file"] and (a["sar"]["subjects"] or a["sar"]["activity_dates"]): e.append("sar empty fields")
    if not a["evidence_requests"] and a["next_best_actions"]["initial"] != fin: e.append("final != initial")
    if round(sum(0 for _ in []), 2): pass
    if known_txns is not None:
        for t in c["affected_txn_ids"]:
            if int(t) not in known_txns: e.append(f"unknown txn {t}")
    return e
