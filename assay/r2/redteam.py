"""Red-team harness: compile attacker scenarios into an overlay dataset, run the agent blind, score it.

Attackers (data/round2/orch/redteam.py, Argon personas) only ever see the policy, our public
write-ups and aggregate statistics. A scenario names placeholder cards by kind ("dormant",
"regular", ...); this module binds each to a real card here, clones that card's own January row
for every planted event (so the model scores it like any real row) and overrides only the fields
the attacker controls.

    python -m assay.r2.redteam build  <scenarios_dir> <overlay_checkout>   # writes data/ overlay + truth.json
    python -m assay.r2.redteam score  <overlay_checkout>                   # after the agent has run
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from assay.data import DATA, ROOT
from assay.r2.frames import R2, case_pack, tx_all

FIRST_ID = 3_700_000
JAN0 = pd.Timestamp("2017-01-01")
JAN_END = pd.Timestamp("2017-01-31 23:59:59")


# ----------------------------------------------------------------- binding
def _card_stats(tx: pd.DataFrame) -> pd.DataFrame:
    g = tx.groupby("card_id")
    return pd.DataFrame({"n": g.size(), "first": g.t.min(), "last": g.t.max()})


def _exam_cards() -> set[str]:
    cp = case_pack()
    out = set(cp.card_id)
    t = tx_all().set_index("TransactionID").card_id
    out |= {t.get(int(x)) for x in cp.flagged_txn_id}
    for p in sorted((ROOT / "cases_r2").glob("HHF-*.json")):
        out |= set(json.loads(p.read_text())["case"]["connected_card_ids"])
    return {c for c in out if c}


def _pick(pool: list[str], seed: str, used: set[str]) -> str | None:
    cand = [c for c in pool if c not in used]
    if not cand:
        return None
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return cand[h % len(cand)]


def bind_cards(sc: dict, tx: pd.DataFrame, jan_cards: set[str], used: set[str], opened: pd.Timestamp) -> dict[str, str]:
    """placeholder label -> real card id, chosen as-of the scenario's first planted event."""
    first = opened - pd.Timedelta(hours=max(float(e.get("hours_before_open", 0)) for e in sc["events"]))
    past = tx[tx.t < first]
    st = _card_stats(past)
    st = st[st.index.isin(jan_cards)]
    recent = st["last"] >= first - pd.Timedelta(days=7)
    pools = {
        "regular": st.index[(st.n >= 40) & (st.n < 300) & recent],
        "dormant": st.index[(st.n >= 20) & (st.n < 300) & (st["last"] < first - pd.Timedelta(days=30))],
        # the sparsest cards available (only ~20 January cards have <= 5 past txns)
        "thin": st.sort_values("n").index[:80],
        "heavy": st.index[(st.n >= 300) & recent],
    }
    out = {}
    for c in sc["cards"]:
        kind = c.get("profile", "regular")
        pool = sorted(pools.get(kind, pools["regular"]))
        card = _pick(pool, f"{sc['scenario_id']}:{c['label']}", used)
        if card is None:
            raise ValueError(f"no free {kind} card for {sc['scenario_id']}:{c['label']}")
        used.add(card)
        out[c["label"]] = card
    return out


# ----------------------------------------------------------------- rows
DEV_COLS = ["DeviceInfo", "id_30", "id_31", "id_33"]


def _event_rows(sc, cards, raw_by_card, idt, next_id, opened):
    rows, ids, id_rows = [], {}, []
    for e in sc["events"]:
        card = cards[e["card"]]
        tmpl = raw_by_card[card]
        r = tmpl.copy()
        t = opened - pd.Timedelta(hours=float(e.get("hours_before_open", 0)))
        t = min(max(t, JAN0), JAN_END)
        dt_s = (t - pd.Timestamp(tmpl.ts)).total_seconds()
        r["TransactionID"] = next_id
        r["ts"] = t.strftime("%Y-%m-%d %H:%M:%S")
        r["TransactionDT"] = int(tmpl.TransactionDT + dt_s)
        if pd.notna(tmpl.D1):   # D1 counts days since the account opened: keep the holder key stable
            r["D1"] = float(tmpl.D1 + (r["TransactionDT"] // 86400 - tmpl.TransactionDT // 86400))
        r["TransactionAmt"] = round(float(e["amount"]), 2)
        r["ProductCD"] = e.get("product", tmpl.ProductCD)
        r["channel"] = e.get("channel", tmpl.channel)
        r["risk_score"] = float(e.get("risk_score", tmpl.risk_score))
        a = e.get("addr1", "home")
        r["addr1"] = tmpl.addr1 if a == "home" else float(a)
        for col, key in (("P_emaildomain", "p_email"), ("R_emaildomain", "r_email")):
            v = e.get(key, "home" if key == "p_email" else "none")
            r[col] = tmpl[col] if v == "home" else (np.nan if v in ("none", "", None) else v)
        rows.append(r)
        dev = e.get("device", "home")
        home_id = idt.get(int(tmpl.TransactionID))
        if dev == "home" and home_id is not None:
            ir = home_id.copy(); ir["TransactionID"] = next_id; id_rows.append(ir)
        elif dev not in ("home", "none", "", None):
            parts = [p.strip() for p in str(dev).split("|")] + [""] * 4
            ir = pd.Series({"TransactionID": next_id, **dict(zip(DEV_COLS, parts[:4]))})
            ir["DeviceType"] = "mobile" if any(k in parts[1] for k in ("Android", "iOS")) else "desktop"
            id_rows.append(ir)
        ids[e["id"]] = next_id
        next_id += 1
    return rows, id_rows, ids, next_id


def build(scen_dir: Path, overlay: Path) -> dict:
    scen = []
    for p in sorted(scen_dir.glob("*.json")):
        if p.name.endswith(".results.json"):
            continue
        for sc in json.loads(p.read_text()):
            sc["_persona"] = p.stem
            scen.append(sc)
    jan = pd.read_csv(R2 / "transactions_2017q1.csv", low_memory=False)
    ident = pd.read_csv(R2 / "identity_2017q1.csv", low_memory=False)
    tx = tx_all()
    jan["card_id"] = jan.TransactionID.map(tx.set_index("TransactionID").card_id)
    jan = jan[jan.TransactionID < 3_660_000]          # organic rows only as templates
    raw_by_card = {c: g.sort_values("TransactionDT").iloc[-1] for c, g in jan.groupby("card_id")}
    idt = {int(r.TransactionID): r for _, r in ident.iterrows()}
    used = _exam_cards()
    next_id, all_rows, all_id, cases, truth, skipped = FIRST_ID, [], [], [], {}, []
    for n, sc in enumerate(scen, start=1):
        cid = f"CASE-{1000 + n}"
        try:
            c = sc["case"]
            opened = pd.Timestamp(2017, 1, int(c.get("day", 20)), 21, 0, 0)
            flag = next(e for e in sc["events"] if e["id"] == c["flagged"])
            cards = bind_cards(sc, tx, set(raw_by_card), used, opened)
            rows, id_rows, ids, next_id = _event_rows(sc, cards, raw_by_card, idt, next_id, opened)
        except Exception as exc:
            skipped.append({"scenario": sc.get("scenario_id"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        all_rows += rows; all_id += id_rows
        fcard = cards[flag["card"]]
        row = {"case_id": cid, "opened_at": opened.strftime("%Y-%m-%d %H:%M:%S"),
               "trigger_type": c.get("trigger_type", "risk_score"), "trigger_text": c.get("trigger_text", ""),
               "flagged_txn_id": ids[flag["id"]], "card_id": fcard, "customer_id": fcard.rsplit("-K", 1)[0],
               "risk_score": flag.get("risk_score", "") if c.get("trigger_type") == "risk_score" else ""}
        row.update({k: v for k, v in (sc.get("raw_case_overrides") or {}).items() if k in row})
        cases.append(row)
        t = sc.get("truth", {})
        truth[cid] = {"scenario": sc["scenario_id"], "persona": sc["_persona"], "idea": sc.get("idea", ""),
                      "verdict": t.get("verdict"), "pattern": t.get("pattern"), "expect": t.get("expect", "answer"),
                      "affected": sorted(ids[x] for x in t.get("affected", []) if x in ids),
                      "cards": cards, "why": t.get("why_it_should_fool_assay", "")}
    _write_overlay(overlay, jan, ident, pd.DataFrame(all_rows), pd.DataFrame(all_id), pd.DataFrame(cases))
    (overlay / "redteam_truth.json").write_text(json.dumps(truth, indent=1, default=str))
    (overlay / "redteam_skipped.json").write_text(json.dumps(skipped, indent=1))
    return {"cases": len(cases), "rows": len(all_rows), "skipped": skipped}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink() if not dst.is_dir() or dst.is_symlink() else None
    if not dst.exists():
        dst.symlink_to(src)


def _write_overlay(overlay: Path, jan, ident, rows, id_rows, cases) -> None:
    d = overlay / "data"
    for p in DATA.iterdir():
        if p.name not in ("round2", "stage"):
            _link(p, d / p.name)
    for p in (DATA / "stage").iterdir():
        if p.name != "r2":
            _link(p, d / "stage" / p.name)
    for p in (DATA / "stage" / "r2").iterdir():
        if p.name not in ("tx_all.parquet", "scores_r2.parquet"):
            _link(p, d / "stage" / "r2" / p.name)
    share = d / "round2" / "share"
    for p in R2.iterdir():
        if p.name not in ("transactions_2017q1.csv", "identity_2017q1.csv", "cases"):
            _link(p, share / p.name)
    (share / "cases").mkdir(parents=True, exist_ok=True)
    _link(R2 / "cases" / "answer_format.md", share / "cases" / "answer_format.md")
    raw = pd.read_csv(R2 / "transactions_2017q1.csv", low_memory=False)
    pd.concat([raw, rows[raw.columns]], ignore_index=True).to_csv(share / "transactions_2017q1.csv", index=False)
    pd.concat([ident, id_rows.reindex(columns=ident.columns)], ignore_index=True).to_csv(share / "identity_2017q1.csv", index=False)
    cases.to_csv(share / "cases" / "case_pack_final.csv", index=False)


# ----------------------------------------------------------------- scoring
def score(overlay: Path) -> dict:
    truth = json.loads((overlay / "redteam_truth.json").read_text())
    status = json.loads((overlay / "redteam_status.json").read_text())   # case -> exit code
    rows = []
    for cid, t in truth.items():
        code = status.get(cid)
        p = overlay / "cases_r2" / f"{cid}.json"
        ans = json.loads(p.read_text()) if code == 0 and p.exists() else None
        c = ans["case"] if ans else {}
        acts = {a["action"] for a in (ans or {}).get("next_best_actions", {}).get("final", [])}
        hard = bool(acts & {"BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"})   # the brief: block or report a legit customer = 0
        got = set(int(x) for x in c.get("affected_txn_ids", []))
        want = set(t["affected"])
        if t["expect"] == "rejected":
            outcome = "REJECTED_OK" if code == 2 else ("CRASH" if code not in (0, 2) else "ANSWERED_BAD_INPUT")
        elif ans is None:
            outcome = "WRONGLY_REJECTED" if code == 2 else "CRASH"
        elif t["verdict"] == "fraud":
            outcome = {"fraud": "CAUGHT", "uncertain": "HEDGED", "legitimate": "MISSED"}[c["verdict"]]
            if outcome == "CAUGHT" and want and len(got & want) < len(want):
                outcome = "CAUGHT_UNDERCOUNTED"
        elif t["verdict"] == "legitimate":
            outcome = "FALSE_ALARM" if (c["verdict"] == "fraud" or hard) else (
                "CLEARED_OK" if c["verdict"] == "legitimate" else "HEDGED_LEGIT")
        else:
            outcome = "UNCERTAIN_OK" if c["verdict"] == "uncertain" else f"UNCERTAIN_TRUTH_{c['verdict'].upper()}"
        rows.append({"case": cid, "persona": t["persona"], "scenario": t["scenario"], "truth": t["verdict"],
                     "expect": t["expect"], "exit": code, "verdict": c.get("verdict"), "p": c.get("fraud_probability"),
                     "pattern": c.get("pattern"), "truth_pattern": t["pattern"],
                     "affected_recall": round(len(got & want) / len(want), 2) if want else None,
                     "exposure": c.get("exposure_usd"), "actions": sorted(acts), "outcome": outcome,
                     "idea": t["idea"]})
    df = pd.DataFrame(rows)
    (overlay / "redteam_results.json").write_text(df.to_json(orient="records", indent=1))
    return {"by_outcome": df.outcome.value_counts().to_dict(),
            "by_persona": {p: g.outcome.value_counts().to_dict() for p, g in df.groupby("persona")}}


def main(argv: list[str]) -> int:
    if argv[0] == "build":
        print(json.dumps(build(Path(argv[1]), Path(argv[2])), indent=1, default=str))
    elif argv[0] == "score":
        print(json.dumps(score(Path(argv[1])), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
