"""Does the agent get it right? Run it on October's closed cases, which it has never seen.

    python -m assay.crossfit        # month-held-out scores first
    python -m assay.evaluate        # -> docs/EVAL_REPORT.json

The bank closed 1,347 cases in October: 144 alerts its analysts cleared and
1,203 cardholder disputes it confirmed as fraud. Each is replayed through the
same Investigator, with three guards against peeking:

  * every July-October transaction carries a score from a model that never saw
    its month (assay/crossfit.py), not the production model's in-sample score;
  * each investigation only sees closed cases closed before its own alert opened;
  * fraud cases are investigated as a bank alert on the first fraud transaction,
    with the customer's complaint removed: every dispute in the history was
    fraud, so letting the dispute in would grade the agent on its own prior.

What is measured is the investigation: model score, the holder's history, the
detectors, the graph around the card, and the §6 gate.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from assay.crossfit import OUT as OOF_PATH
from assay.data import ROOT, closed_cases
from assay.investigate import Investigator
from assay.patterns import PatternModel
from assay.store import LocalStore

REPORT = ROOT / "docs" / "EVAL_REPORT.json"


class AsOfStore(LocalStore):
    """The reference store, with held-out scores and a closed-case history that ends at the alert."""

    def __init__(self):
        super().__init__()
        oof = pd.read_parquet(OOF_PATH).set_index("TransactionID").p_oof
        self.tx["p_fraud"] = self.tx.TransactionID.map(oof).fillna(self.tx.p_fraud)
        self.by_card = {k: v for k, v in self.tx.groupby("card_id")}
        self.by_uid = {k: v for k, v in self.tx.groupby("uid")}
        dp = self.tx[self.tx.device_profile != ""]
        self.by_device = {k: v for k, v in dp.groupby("device_profile")}
        self.all_cc = self.cc.copy()
        self.all_cc["_closed"] = pd.to_datetime(self.all_cc.closed_at)

    def as_of(self, when) -> None:
        self.cc = self.all_cc[self.all_cc._closed < pd.Timestamp(when)].drop(columns="_closed")
        self.memory = []


def alerts(cc: pd.DataFrame) -> pd.DataFrame:
    oct_ = cc[cc.opened_at.astype(str).str[:7] == "2016-10"]
    rows = []
    for r in oct_.itertuples():
        fid = r.first_fraud_txn_id if pd.notna(r.first_fraud_txn_id) else r.txn_list[0]
        rows.append({"case_id": r.case_id, "opened_at": str(r.opened_at), "trigger_type": "risk_score",
                     "trigger_text": f"Review transaction {int(fid)}.", "flagged_txn_id": int(fid),
                     "card_id": r.card_id, "customer_id": r.customer_id,
                     "truth": "fraud" if r.outcome == "confirmed_fraud" else "legitimate"})
    return pd.DataFrame(rows)


def main() -> int:
    store = AsOfStore()
    inv = Investigator(store, PatternModel())
    cases = alerts(closed_cases()).sort_values("opened_at")
    out = []
    for i, c in enumerate(cases.itertuples(index=False)):
        store.as_of(c.opened_at)
        a = inv.run(pd.Series(c._asdict()))
        out.append({"case_id": c.case_id, "truth": c.truth, "verdict": a["case"]["verdict"],
                    "p": a["case"]["fraud_probability"], "p_model": round(float(inv.last["p_model"]), 4),
                    "bank": round(float(inv.last["flagged"].risk_score), 4)})
        if i % 100 == 0:
            print(i, len(cases), flush=True)
    df = pd.DataFrame(out)
    y = (df.truth == "fraud").astype(int)

    def side(d):
        return {"n": int(len(d)), **{v: int((d.verdict == v).sum()) for v in ("fraud", "legitimate", "uncertain")}}

    decided = df[df.verdict != "uncertain"]
    rep = {
        "month": "2016-10, never seen by the scores used",
        "cases": int(len(df)),
        "cleared_by_the_bank": side(df[df.truth == "legitimate"]),
        "confirmed_fraud_without_the_customer_complaint": side(df[df.truth == "fraud"]),
        "accuracy_when_decided": round(float((decided.verdict == decided.truth).mean()), 4),
        "held_for_a_human": round(float((df.verdict == "uncertain").mean()), 4),
        "brier_agent": round(float(np.mean((df.p - y) ** 2)), 4),
        "brier_model_alone": round(float(np.mean((df.p_model - y) ** 2)), 4),
        "brier_bank_risk_score": round(float(np.mean((df.bank - y) ** 2)), 4),
        "note": "the case mix is the bank's (89% fraud), not the brief's 50/50, so read the two sides separately",
    }
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    df.to_csv(ROOT / "runs" / "eval_october.csv", index=False)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
