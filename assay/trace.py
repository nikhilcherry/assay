"""Export each investigation as a replayable trace for the web replay (site/).

    python -m assay.trace            # the 20 exam cases + the 60 monitor alerts -> site/data/

A trace is what actually happened inside one run, in order:

  steps     every graph query the agent issued, with the vertices it returned
            (so the replay can draw the graph growing exactly as the agent saw it)
  odds      the probability after each finding that moved it: the calibrated
            model score, then each likelihood ratio in log-odds, then the §6 gate
            and the simulated evidence reply
  answer    the answer file itself (verdict, actions, SAR), unchanged

Nothing here is re-derived for display. The traces come from re-running the
same Investigator on the reference store, and main() fails if any verdict,
probability, pattern, exposure or connected card differs from the committed
answer files.
"""

from __future__ import annotations

import json
import math
import sys

import pandas as pd

from assay.data import ROOT, case_pack
from assay.investigate import Investigator
from assay.patterns import PatternModel
from assay.store import LocalStore

OUT = ROOT / "site" / "data"
MAX_TXNS_PER_STEP = 60

QUERIES = ["get_transaction", "card_window", "holder_history", "card_history", "closed_cases_by_device",
           "device_neighbors", "closed_cases_for_card", "closed_cases_touching", "closed_cases_like",
           "prior_fraud_cases"]


class TracingStore(LocalStore):
    """LocalStore that also remembers what each query returned."""

    def __init__(self):
        super().__init__()
        self.results: list = []
        for name in QUERIES:
            setattr(self, name, self._wrap(getattr(self, name)))

    def reset(self) -> None:
        super().reset()
        self.results = []

    def _wrap(self, fn):
        def run(*a, **kw):
            n = len(self.calls)
            out = fn(*a, **kw)
            self.results.append((n, out[0]))
            return out
        return run


def _txn(r) -> dict:
    return {"id": str(int(r.TransactionID)), "t": str(r.t)[:16], "amt": round(float(r.TransactionAmt), 2),
            "product": r.ProductCD, "channel": r.channel, "card": r.card_id,
            "region": None if pd.isna(r.addr1) else int(r.addr1),
            "device": r.device_profile or None, "new": r.id_15 == "New",
            "proxy": r.id_23 if isinstance(r.id_23, str) else None,
            "risk": round(float(r.risk_score), 3), "p": round(float(r.p_fraud), 3)}


def _cc(r) -> dict:
    return {"id": r.case_id, "card": r.card_id, "outcome": r.outcome, "pattern": r.pattern,
            "opened": str(r.opened_at)[:10], "txns": [str(t) for t in r.txn_list][:20],
            "connected": list(r.connected_list)[:30], "exposure": round(float(r.exposure_usd), 2)}


def _summarise(result, flagged_t, cap: int = MAX_TXNS_PER_STEP) -> dict:
    """What a query returned, reduced to the vertices the replay draws."""
    if isinstance(result, pd.Series):
        return {"txns": [_txn(result)], "n": 1}
    if isinstance(result, list):
        return {"memory": [{"id": m["v_id"], "verdict": m["attributes"].get("verdict"),
                            "pattern": m["attributes"].get("pattern")} for m in result], "n": len(result)}
    if isinstance(result, pd.DataFrame):
        if "case_id" in result.columns:
            return {"closed_cases": [_cc(r) for r in result.itertuples()], "n": len(result)}
        if "TransactionID" in result.columns:
            d = result.copy()
            n = len(d)
            # Keep what an investigator looks at: nearest the alert in time, and the highest scored.
            d["_gap"] = (d.t - flagged_t).abs()
            d = pd.concat([d.nsmallest(cap // 2, "_gap"),
                           d.nlargest(cap // 2, "p_fraud")]).drop_duplicates("TransactionID")
            return {"txns": [_txn(r) for r in d.sort_values("t").itertuples()], "n": n}
    return {}


def _odds(inv: Investigator, ans: dict) -> list[dict]:
    """The probability path: model prior, each likelihood ratio, then the evidence reply."""
    last = inv.last
    p = last["p_model"]
    path = [{"label": "calibrated model", "lr": None, "p": round(p, 4), "tag": "model"}]
    for fnd in last["findings"]:
        if fnd.lr == 1.0 or not fnd.claim:
            continue
        p = 1 / (1 + math.exp(-(math.log(p / (1 - p)) + math.log(fnd.lr))))
        path.append({"label": fnd.tag, "lr": round(fnd.lr, 3), "p": round(p, 4), "tag": fnd.tag,
                     "claim": fnd.claim})
    assert abs(p - last["p1"]) < 1e-6, (p, last["p1"])
    path.append({"label": "§6 gate", "lr": None, "p": round(last["p1"], 4), "tag": "gate", "band": last["band"]})
    if ans["evidence_requests"]:
        r = ans["evidence_requests"][-1]
        p1, p2 = last["p1"], last["p2"]
        lr = (p2 / (1 - p2)) / (p1 / (1 - p1)) if 0 < p2 < 1 and 0 < p1 < 1 else None
        path.append({"label": r["type"].replace("_", " "), "lr": None if lr is None else round(lr, 3),
                     "p": round(p2, 4), "tag": "evidence", "claim": r["assumed_response"]})
    return path


def trace(inv: Investigator, store: TracingStore, case: pd.Series) -> dict:
    ans = inv.run(case)
    f = inv.last["flagged"]
    by_call: dict = {}
    for n, res in store.results:
        by_call.setdefault(n, res)
    steps = []
    for i, c in enumerate(store.calls):
        args = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in c.args.items()}
        s = {"i": i, "query": c.name, "args": args, "ref": c.ref}
        if i in by_call:
            # A device traversal is the ring itself: keep every card it reached.
            s.update(_summarise(by_call[i], f.t, 400 if c.name == "device_neighbors" else MAX_TXNS_PER_STEP))
        steps.append(s)
    ep = inv.last["episode"]
    return {
        "case_id": ans["case_id"],
        "trigger": case.get("trigger_type", ""),
        "trigger_text": case.get("trigger_text", ""),
        "opened_at": str(case.get("opened_at", "")),
        "flagged": _txn(f),
        "holder": f.uid,
        "p_model": round(inv.last["p_model"], 4),
        "bank_risk": round(float(f.risk_score), 3),
        "pro": inv.last["pro"], "con": inv.last["con"],
        "odds": _odds(inv, ans),
        "steps": steps,
        "episode": {"txns": [str(int(t)) for t in ep.txns.TransactionID], "cards": ep.connected_cards,
                    "devices": ep.connected_devices},
        "answer": ans,
    }


def _check(t: dict, committed: dict) -> list[str]:
    a, b = t["answer"]["case"], committed["case"]
    return [k for k in ("verdict", "fraud_probability", "pattern", "exposure_usd", "connected_card_ids")
            if a[k] != b[k]]


def main(argv=None) -> int:
    store = TracingStore()
    inv = Investigator(store, PatternModel(), writer=store.write_case)
    OUT.mkdir(parents=True, exist_ok=True)
    index, bad = [], 0

    def emit(case, committed_path, kind):
        nonlocal bad
        t = trace(inv, store, case)
        if committed_path.exists():
            diff = _check(t, json.loads(committed_path.read_text()))
            if diff:
                print(f"{t['case_id']}: differs from {committed_path.name} on {diff}", file=sys.stderr)
                bad += 1
        t["kind"] = kind
        (OUT / f"{t['case_id']}.json").write_text(json.dumps(t, separators=(",", ":"), default=str))
        c = t["answer"]["case"]
        index.append({"id": t["case_id"], "kind": kind, "verdict": c["verdict"], "p": c["fraud_probability"],
                      "pattern": c["pattern"], "exposure": c["exposure_usd"], "card": t["flagged"]["card"],
                      "amt": t["flagged"]["amt"], "bank_risk": t["bank_risk"], "t": t["flagged"]["t"],
                      "connected": len(c["connected_card_ids"]), "sar": t["answer"]["sar"]["file"],
                      "trigger": t["trigger"], "source": case.get("source", "")})
        print(f"{t['case_id']} {c['verdict']:<10} p={c['fraud_probability']:.2f} steps={len(t['steps'])}")

    # Same order as assay.run then assay.monitor, so case memory matches the committed files.
    for _, case in case_pack().sort_values("opened_at").iterrows():
        emit(case, ROOT / "cases" / f"{case.case_id}.json", "exam")
    alerts_csv = ROOT / "monitor" / "alerts.csv"
    if alerts_csv.exists():
        for _, a in pd.read_csv(alerts_csv).sort_values("opened_at").iterrows():
            emit(a, ROOT / "monitor" / f"{a.case_id}.json", "monitor")
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
