"""Measure every detector's likelihood ratio on top of the model.

    python -m assay.crossfit        # month-held-out scores first
    python -m assay.measure         # -> docs/LR_REPORT.json

A detector's likelihood ratio is applied on top of the model score, so the
number that matters is not "how much more often is this fraud than average"
(the marginal lift, which double-counts whatever the model already sees) but
the LR that makes model-plus-finding reproduce the fraud rate actually observed
on the detector's hits. Both are reported; `lr` is the second, fitted on
month-held-out scores so the model has never seen the month it is scoring, with
a bootstrap 95% interval over the detector's hits.

Labels are the bank's closed cases, which name 3.4% of July-October as fraud:
the whole fraud rate of the underlying data, so an unnamed transaction is
legitimate to a good approximation.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from assay import detectors as D
from assay.crossfit import OUT as OOF_PATH
from assay.data import ROOT, closed_cases
from assay.store import LocalStore

REPORT = ROOT / "docs" / "LR_REPORT.json"
BOOT = 400


def fit_lr(p, y) -> float:
    """The LR under which mean(sigmoid(logit(p) + log LR)) equals the observed rate."""
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    target = float(np.mean(y))
    if target <= 0:
        return 0.0
    lg = np.log(p / (1 - p))
    lo, hi = 1e-4, 1e6
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(lg + math.log(mid))))).mean() < target else (lo, mid)
    return hi


def summarise(units: pd.DataFrame, base: float, unit: str, note: str, seed: int = 7) -> dict:
    """units: one row per detector hit, with p (the score the agent would combine with) and y."""
    n, k = len(units), int(units.y.sum())
    out = {"unit": unit, "n": n, "fraud": k, "note": note}
    if not n:
        return out
    rate = k / n
    rng = np.random.default_rng(seed)
    boots = [fit_lr(units.p.values[i], units.y.values[i])
             for i in (rng.integers(0, n, n) for _ in range(BOOT))]
    out.update(rate=round(rate, 4),
               lr_marginal=round((rate / (1 - rate)) / (base / (1 - base)), 2) if rate < 1 else None,
               lr=round(fit_lr(units.p, units.y), 2),
               ci95=[round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
               mean_model_p=round(float(units.p.mean()), 4))
    return out


def structuring_units(tx):
    on = tx[(tx.channel == "online") & (tx.TransactionAmt >= D.STRUCTURING_LOW)
            & (tx.TransactionAmt < D.STRUCTURING_HIGH)].sort_values("t")
    rows = []
    for _, g in on.groupby("card_id"):
        used = set()
        for i in range(len(g)):
            if g.index[i] in used:
                continue
            w = g[(g.t >= g.t.iloc[i]) & (g.t <= g.t.iloc[i] + pd.Timedelta(hours=1))]
            if len(w) >= 3:
                rows.append({"p": w.p.max(), "y": bool(w.y.any())})   # the agent scores the burst at its max
                used |= set(w.index)
    return pd.DataFrame(rows, columns=["p", "y"])


def card_testing_units(tx):
    rows = []
    for _, g in tx[tx.channel == "online"].groupby("card_id"):
        if (g.TransactionAmt < D.SMALL_AUTH_USD).sum() < 3:
            continue
        g = g.sort_values("t")
        small = g[g.TransactionAmt < D.SMALL_AUTH_USD]
        done = set()
        for i in range(len(small) - 2):
            run = small.iloc[i:]
            run = run[run.t <= small.iloc[i].t + pd.Timedelta(hours=1)]
            if len(run) >= 3 and run.index[0] not in done:
                done |= set(run.index)
                after = g[(g.t > run.t.max()) & (g.t <= run.t.max() + pd.Timedelta(hours=24))
                          & (g.TransactionAmt >= 20)].head(3)
                if len(after):
                    rows.append({"p": after.p.max(), "y": bool(after.y.any())})
    return pd.DataFrame(rows, columns=["p", "y"])


def ring_units(tx, store):
    rows = []
    dp = tx[tx.device_profile != ""]
    for prof, g in dp.groupby("device_profile"):
        if D.is_generic_device(prof) or g.card_id.nunique() < D.RING_MIN_CARDS:
            continue
        full = store.by_device[prof]
        for r in g.itertuples():
            nb = full[(full.t >= r.t - pd.Timedelta(days=30)) & (full.t <= r.t + pd.Timedelta(days=30))]
            if D.is_ring(nb):
                rows.append({"p": r.p, "y": bool(r.y), "profile": prof, "card": r.card_id})
    return pd.DataFrame(rows, columns=["p", "y", "profile", "card"])


def memory_units(tx, cc):
    """A card the bank has already confirmed fraud on, more than a day before this transaction."""
    first = cc[cc.outcome == "confirmed_fraud"].assign(o=lambda d: pd.to_datetime(d.opened_at)).groupby("card_id").o.min()
    prior = tx.card_id.map(first)
    m = tx[prior.notna() & (tx.t > prior + pd.Timedelta(days=1))]
    return m[["p", "y"]].reset_index(drop=True)


def main() -> int:
    store = LocalStore()
    cc = closed_cases()
    fraud = set(sum(cc[cc.outcome == "confirmed_fraud"].txn_list.tolist(), []))
    oof = pd.read_parquet(OOF_PATH).set_index("TransactionID").p_oof
    tx = store.tx[(store.tx.t >= "2016-07-01") & (store.tx.t < "2016-11-01")].copy()
    tx["p"] = tx.TransactionID.map(oof)
    tx["y"] = tx.TransactionID.isin(fraud)
    tx = tx[tx.p.notna()]
    base = float(tx.y.mean())

    ring = ring_units(tx, store)
    rep = {
        "basis": "July-October 2016; each month scored by a model trained on the other three (assay/crossfit.py); "
                 "labels are the bank's closed cases",
        "base_rate": round(base, 4),
        "detectors": {
            "structuring": summarise(structuring_units(tx), base, "burst of >=3 online $400-$500 on a card within an hour",
                                     "scored at the burst's strongest transaction, as the agent does"),
            "card_testing": summarise(card_testing_units(tx), base, "run of >=3 online < $5 within an hour, then a purchase >= $20",
                                      "scored at the strongest follow-on purchase"),
            "ring": summarise(ring, base, "transaction from a device profile that is a ring within +-30 days",
                              f"{ring.profile.nunique() if len(ring) else 0} qualifying profile(s), "
                              f"{ring.card.nunique() if len(ring) else 0} cards: effectively one cluster, so the "
                              "interval understates the uncertainty"),
            "memory": summarise(memory_units(tx, cc), base, "transaction on a card with confirmed fraud opened > 1 day earlier",
                                "what the agent's case memory asserts when it finds an earlier fraud verdict on the card"),
        },
    }
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
