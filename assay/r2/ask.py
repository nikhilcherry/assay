"""Closed-menu Q&A about the open case. No LLM, and free text never becomes a query.

The question text is only matched against fixed keyword rules to pick ONE intent from
a closed list. Every query parameter comes from the case row (case_pack) and the bank
record (tx_all); each goes through the same validators as the restricted MCP tool layer
(assay.r2.mcp_server: card id / txn id formats, 120-day cap, 500-row cap). All reads are
as-of the case's opened_at.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from assay.r2.frames import case_pack, tx_all
from assay.r2.mcp_server import MCPServer, R2Store, _clean_data

MAX_Q = 300
SHOW = 20
ROW_CAP = 500
ANSWERS_DIR = Path(__file__).resolve().parents[2] / "cases_r2"
CASE_ID_RE = re.compile(r"^HHF-\d{3}$")

SUPPORTED = [
    "What other cards used this device?",
    "What is this card's usual spending?",
    "Are there past cases on this card?",
    "Show the flagged transaction.",
    "Does the holder have other cards?",
    "Which cards are linked in the ring?",
]

# Ordered: first matching rule wins. Keywords only choose the intent.
# Topics the bank's data cannot answer. Checked first, so "did the customer travel?" is refused
# instead of being answered with a different question that happens to share a keyword.
_OUT_OF_SCOPE = re.compile(
    r"\b(travel\w*|trips?|abroad|vacation|holiday|locations?|where (is|was|did)|called|phoned|phone calls?|"
    r"employer|income|salary|address(es)?|relatives?|family|intent|motive|why)\b", re.I)

_RULES = [
    ("device_neighbors", re.compile(r"\b(devices?|shared device|same device|handset|same phone|this phone)\b", re.I)),
    ("holder_history", re.compile(r"\b(holder|cardholder|same person|owner)\b|\bother cards (of|for|on) (this|the) "
                                  r"(customer|holder|person)\b|\bcustomer'?s other cards\b", re.I)),
    ("prior_cases", re.compile(r"\b(past cases?|prior cases?|previous cases?|precedents?|closed cases?|history of cases)\b", re.I)),
    ("ring", re.compile(r"\b(linked|link|connected|ring|cohort|network|cluster)\b", re.I)),
    ("card_history", re.compile(r"\b(history|usual|baseline|median|typical|spending|spend|normal|average)\b", re.I)),
    ("get_transaction", re.compile(r"\b(transaction|txn|flagged|details?)\b", re.I)),
]

_server = None


def _srv() -> MCPServer:
    global _server
    if _server is None:
        _server = MCPServer(R2Store())
    return _server


def classify(question: str) -> str:
    q = (question or "")[:MAX_Q]
    if _OUT_OF_SCOPE.search(q):
        return "unsupported"
    for intent, rx in _RULES:
        if rx.search(q):
            return intent
    return "unsupported"


def _s(v: Any, n: int = 60) -> Any:
    if isinstance(v, str):
        return v[:n]
    return v


_TX_COLS = ["TransactionID", "t", "card_id", "amount", "ProductCD", "channel", "risk_score", "device_profile"]


def _tx_rows(df: pd.DataFrame) -> List[dict]:
    cols = [c for c in _TX_COLS if c in df.columns]
    out = []
    for r in _clean_data(df[cols]):
        out.append({k: _s(v) for k, v in r.items()})
    return out


def _fmt(x: float) -> str:
    return f"${x:,.2f}"


def _case_row(case_id: str) -> pd.Series:
    cp = case_pack()
    sub = cp[cp.case_id == case_id]
    if sub.empty:
        raise LookupError(f"Case {case_id} not found")
    return sub.iloc[0]


def answer(case_id: str, question: str) -> Dict[str, Any]:
    if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
        raise ValueError("case_id must match HHF-NNN")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is empty")
    if len(question) > MAX_Q:
        raise ValueError(f"question longer than {MAX_Q} characters")

    intent = classify(question)
    if intent == "unsupported":
        return {"intent": "unsupported", "tool": None, "params": {}, "rows": [], "total": 0,
                "answer": "I can only answer a fixed set of questions about this case.",
                "supported": SUPPORTED, "attestation": None}

    row = _case_row(case_id)
    card = str(row["card_id"])
    txn = int(float(row["flagged_txn_id"]))
    opened = pd.Timestamp(row["opened_at"])
    srv, store = _srv(), _srv().store
    tx = tx_all()

    def done(tool, params, rows, total, text, query=None):
        return {"intent": intent, "tool": tool, "params": params, "rows": rows[:SHOW], "total": total,
                "answer": text, "attestation": {"query": query or tool, "params": params}}

    if intent == "get_transaction":
        params = {"txn_id": txn}
        srv._validate_args("get_transaction", params)
        sub = tx[(tx.TransactionID == txn) & (tx.t <= opened)]
        rows = _tx_rows(sub)
        text = (f"Flagged transaction {txn} is {_fmt(float(sub.iloc[0].amount))} on {card} at {rows[0]['t']}."
                if len(sub) else f"Transaction {txn} is not visible as of {opened}.")
        return done("get_transaction", params, rows, len(sub), text)

    if intent == "card_history":
        params = {"card_id": card, "start": str(opened - pd.Timedelta(days=120)), "end": str(opened)}
        res = srv.call_tool("card_window", dict(params))
        recs = [r for r in res["transactions"] if pd.Timestamp(r["t"]) <= opened]
        am = pd.Series([r["amount"] for r in recs], dtype=float)
        recs.sort(key=lambda r: r["t"])
        rows = [{k: _s(r.get(k)) for k in _TX_COLS} for r in recs[-10:]]
        if len(am):
            text = (f"{card} has {len(am)} transactions in the 120 days to opening: median {_fmt(am.median())}, "
                    f"95th percentile {_fmt(am.quantile(0.95))}; last {len(rows)} shown.")
        else:
            text = f"{card} has no transactions in the 120 days before opening."
        return done("card_window", params, rows, len(recs), text)

    if intent == "device_neighbors":
        params = {"card_id": card}
        srv._validate_args("device_neighbors", params)
        flagged = tx[tx.TransactionID == txn]
        prof = flagged.iloc[0].device_profile if len(flagged) else None
        if prof is None or pd.isna(prof) or not str(prof).strip():
            return done("device_neighbors", params, [], 0, "The flagged transaction has no device profile, so no device neighbours can be found.")
        df, _ = store.device_neighbors(card)
        df = df[(df.device_profile == prof) & (df.card_id != card) & (df.t <= opened)
                & (df.t >= opened - pd.Timedelta(days=30))].sort_values("t").tail(ROW_CAP)
        cards = sorted(set(df.card_id))
        params2 = dict(params, as_of=str(opened), days=30, device_profile=str(prof)[:60])
        text = (f"{len(cards)} other card(s) used the flagged device profile in the 30 days to opening "
                f"({len(df)} transactions)." if len(df) else "No other cards used the flagged device in the 30 days to opening.")
        return done("device_neighbors", params2, _tx_rows(df), len(df), text)

    if intent == "holder_history":
        params = {"card_id": card}
        srv._validate_args("holder_history", params)
        df, _ = store.holder_history(card)
        df = df[df.t <= opened].sort_values("t").tail(ROW_CAP)
        cards = sorted(set(df.card_id))
        text = f"The holder has {len(df)} transactions across {len(cards)} card(s) up to opening: {', '.join(cards[:6])}."
        return done("holder_history", dict(params, as_of=str(opened)), _tx_rows(df.tail(SHOW)), len(df), text)

    if intent == "prior_cases":
        params = {"card_id": card}
        srv._validate_args("prior_cases", params)
        df, _ = store.prior_cases(card)
        df = df[pd.to_datetime(df.closed_at) <= opened]
        rows = [{k: _s(r.get(k), 80) for k in ("case_id", "card_id", "closed_at", "outcome", "pattern", "n_txns")}
                for r in _clean_data(df)]
        text = (f"{len(rows)} case(s) closed before opening touch {card}; outcomes: "
                + (", ".join(sorted({str(r.get('outcome')) for r in rows})) or "none") + ".") if rows else \
            f"No cases closed before opening touch {card}."
        return done("prior_cases", dict(params, as_of=str(opened)), rows, len(rows), text)

    # ring: read-only from the shipped answer file
    p = ANSWERS_DIR / f"{case_id}.json"
    params = {"case_id": case_id}
    if not p.is_file():
        return done("shipped_answer", params, [], 0, "No shipped answer file for this case.")
    case = json.loads(p.read_text()).get("case", {})
    conn = [str(c) for c in case.get("connected_card_ids", []) if CARD_RE.fullmatch(str(c))]
    refs = [str(e.get("ref", ""))[:120] for e in case.get("evidence", []) if isinstance(e, dict)]
    rows = [{"connected_card": c} for c in conn] + [{"evidence_ref": r} for r in refs]
    text = f"The shipped answer links {len(conn)} card(s) and cites {len(refs)} evidence reference(s)."
    return done("shipped_answer", params, rows, len(rows), text)


CARD_RE = re.compile(r"^C\d{5}-K\d+$")
