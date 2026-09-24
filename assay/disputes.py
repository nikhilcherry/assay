"""What a cardholder's "I never made this purchase" is worth, measured.

    python -m assay.disputes      # -> docs/DISPUTE_REPORT.json

Two facts from the bank's own October, the month the model never trained on,
pull in opposite directions:

  1. Every one of the 1,203 disputes the bank investigated was confirmed fraud,
     including the 15% its model (and ours) scored under 0.05. In the history a
     low model score does not clear a dispute.
  2. The brief says half the exam cases are legitimate, so the case pack may hold
     planted disputes on legitimate transactions, which the history cannot show.
     For those, the only honest discriminator is how the model score is
     distributed on disputed fraud versus on legitimate transactions, and that
     is measured: a score under 0.005 is 14 times more common on a legitimate
     transaction; a score of 0.6-0.8 is 88 times more common on disputed fraud.

The posterior for a disputed transaction is the average of the two readings,
weighted equally because nothing in the data says which world the case pack is
drawn from. The weight is the one assumption here, and it is stated as one.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from assay.data import ROOT, STAGE, closed_cases

REPORT = ROOT / "docs" / "DISPUTE_REPORT.json"
BINS = [0, .005, .01, .02, .05, .1, .2, .4, .6, .8, 1.0001]
W_HISTORY = 0.5     # assumption: the case pack is as likely to hold planted disputes as real ones


def measure() -> dict:
    cc = closed_cases()
    sc = pd.read_parquet(STAGE / "scores.parquet").set_index("TransactionID")
    oct_ = cc[cc.opened_at.astype(str).str[:7] == "2016-10"]
    disputed = oct_.analyst_notes.str.contains("reported unrecognized", na=False)
    d, a = oct_[disputed], oct_[~disputed]
    p_disp = d.first_fraud_txn_id.astype(int).map(sc.p_oot).dropna().clip(1e-3, 1 - 1e-3)
    fraud_ids = set(sum(cc[cc.outcome == "confirmed_fraud"].txn_list.tolist(), []))
    oot = sc[sc.p_oot.notna()]
    p_legit = oot[~oot.index.isin(fraud_ids)].p_oot

    # Smallest likelihood ratio consistent with 0 cleared out of n (rule of three, 95%).
    n = len(p_disp)
    floor = 1 - 3 / n
    logit = np.log(p_disp / (1 - p_disp))
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if (1 / (1 + np.exp(-(logit + math.log(mid))))).mean() < floor:
            lo = mid
        else:
            hi = mid
    hd = np.histogram(p_disp, BINS)[0] / len(p_disp)
    hl = np.histogram(p_legit, BINS)[0] / len(p_legit)
    return {
        "month": "2016-10 (out of time)",
        "disputes": int(len(d)), "disputes_confirmed_fraud": int((d.outcome == "confirmed_fraud").sum()),
        "model_alerts": int(len(a)), "model_alerts_cleared": int((a.outcome == "cleared").sum()),
        "disputed_scored_under_0.05": round(float((p_disp < .05).mean()), 3),
        "lr_history": round(hi),
        "lr_history_note": "smallest LR under which the model + dispute reproduces a >= 1 - 3/n fraud rate on "
                           f"{n} scored October disputes",
        "score_bins": [{"from": a_, "to": min(b_, 1.0), "disputed_fraud": round(float(x), 4),
                        "legitimate": round(float(y), 4), "lr": round(float(x / max(y, 1e-9)), 3)}
                       for a_, b_, x, y in zip(BINS, BINS[1:], hd, hl)],
        "w_history": W_HISTORY,
    }


@lru_cache(maxsize=1)
def _report() -> dict:
    return json.loads(REPORT.read_text())


def posterior(p_model: float) -> tuple[float, dict]:
    """P(fraud | disputed, model score), and the two readings it averages."""
    r = _report()
    p = min(max(p_model, 1e-4), 1 - 1e-4)
    p_hist = 1 / (1 + math.exp(-(math.log(p / (1 - p)) + math.log(r["lr_history"]))))
    b = next(x for x in r["score_bins"] if x["from"] <= p_model < x["to"] or x["to"] == 1.0)
    p_planted = b["lr"] / (1 + b["lr"])
    w = r["w_history"]
    return w * p_hist + (1 - w) * p_planted, {"p_history": p_hist, "p_planted": p_planted, "lr_score": b["lr"]}


if __name__ == "__main__":
    rep = measure()
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))
