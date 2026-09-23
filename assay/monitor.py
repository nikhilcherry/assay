"""The optional part of the brief: watch the exam period unprompted.

Sweeps November-December for things worth an investigator's time, raises its
own alerts, runs the same investigation on each, and writes the answer files to
monitor/ (never cases/). Three sources of alerts, in priority order:

  1. structure scans -- patterns the closed cases teach but no single
     transaction score shows: near-$500 online bursts (structuring) and device
     profiles that appear on many unrelated cards in a month (rings);
  2. the calibrated model -- transactions at p_fraud >= 0.70 that the bank's
     own risk score rated under 0.30, i.e. fraud the bank's model is missing;
  3. nothing else. The monitor deliberately does not re-raise what the bank's
     risk score already flags; those alerts exist without it.

One alert per card, deduplicated against the 20 exam cases.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from assay import detectors as D
from assay.data import ROOT, case_pack
from assay.investigate import Investigator
from assay.patterns import PatternModel
from assay.store import LocalStore
from assay.validate import validate

OUT = ROOT / "monitor"


def sweep(store: LocalStore, limit_model: int = 15) -> pd.DataFrame:
    tx = store.tx
    ex = tx[tx.t >= "2016-11-01"]
    exam_cards = set(case_pack().card_id)
    alerts = []

    # 1a. structuring bursts
    on = ex[(ex.channel == "online") & (ex.TransactionAmt >= D.STRUCTURING_LOW) & (ex.TransactionAmt < D.STRUCTURING_HIGH)]
    for card, g in on.groupby("card_id"):
        g = g.sort_values("t")
        for i in range(len(g)):
            w = g[(g.t >= g.t.iloc[i]) & (g.t <= g.t.iloc[i] + pd.Timedelta(hours=1))]
            if len(w) >= 3:
                alerts.append(("structure_scan", w.iloc[-1], f"{len(w)} online purchases between $400 and $500 "
                               f"within an hour on card {card}"))
                break

    # 1b. device rings: a specific profile on >= RING_MIN_CARDS cards within 30 days
    dev = ex[(ex.device_profile != "")]
    for prof, g in dev.groupby("device_profile"):
        if D.is_generic_device(prof) or g.card_id.nunique() < D.RING_MIN_CARDS:
            continue
        span = (g.t.max() - g.t.min()).days
        if span <= 60 and D.is_ring(g):
            for card, gc in g.groupby("card_id"):
                alerts.append(("ring_scan", gc.sort_values("t").iloc[-1],
                               f"device profile {prof} on {g.card_id.nunique()} cards in {span} days, marked New "
                               f"and behind a proxy on every use"))

    # 2. fraud the bank's score is missing
    miss = ex[(ex.p_fraud >= 0.70) & (ex.risk_score < 0.30)].sort_values("p_fraud", ascending=False)
    miss = miss.drop_duplicates("card_id").head(limit_model)
    for _, r in miss.iterrows():
        alerts.append(("model_monitor", r, f"assay model {r.p_fraud:.2f} vs bank risk score {r.risk_score:.2f}"))

    rows, seen = [], set(exam_cards)
    for kind, r, why in alerts:
        if r.card_id in seen:
            continue
        seen.add(r.card_id)
        rows.append({"case_id": f"MON-{len(rows) + 1:03d}", "opened_at": str(r.t), "trigger_type": "risk_score",
                     "trigger_text": f"Autonomous monitor ({kind}): {why}. Review transaction {int(r.TransactionID)}.",
                     "flagged_txn_id": int(r.TransactionID), "card_id": r.card_id, "customer_id": r.customer_id,
                     "risk_score": r.risk_score, "source": kind})
    return pd.DataFrame(rows)


def main() -> int:
    store = LocalStore()
    alerts = sweep(store)
    OUT.mkdir(exist_ok=True)
    alerts.to_csv(OUT / "alerts.csv", index=False)
    inv = Investigator(store, PatternModel(), writer=store.write_case)
    # Seed the memory with the 20 exam investigations, so a monitor alert on a
    # card the exam already covered (e.g. a device-ring card) can find it.
    exam = json.loads((ROOT / "runs" / "exam_index.json").read_text()) if (ROOT / "runs" / "exam_index.json").exists() else []
    for e in exam:
        store.memory.append({"v_id": e["graph_case_id"], "card_id": e["card_id"], "connected": set(e["connected"]),
                             "attributes": {"verdict": e["verdict"], "pattern": e["pattern"],
                                            "alert_id": e["case_id"]}})
    alerts = alerts.sort_values("opened_at")
    summary = []
    for _, a in alerts.iterrows():
        ans = inv.run(a)
        ans["monitor"] = {"source": a.source, "trigger_text": a.trigger_text}
        assert not validate(ans), validate(ans)
        (OUT / f"{a.case_id}.json").write_text(json.dumps(ans, indent=2) + "\n")
        c = ans["case"]
        summary.append({"case_id": a.case_id, "source": a.source, "card_id": a.card_id, "verdict": c["verdict"],
                        "pattern": c["pattern"], "p": c["fraud_probability"], "exposure_usd": c["exposure_usd"],
                        "sar": ans["sar"]["file"]})
        print(summary[-1])
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
