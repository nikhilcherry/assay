"""Which known pattern a fraud episode is, learned from the closed cases.

The five documented patterns are labels on 4,640 confirmed closed cases. A
small multiclass model on the first fraudulent transaction of each case learns
what separates them in this data (channel, product code, device novelty,
region novelty, match flags). Card testing and the undocumented patterns are
handled by explicit rules in the investigator, because they are defined by a
sequence across transactions rather than by any one transaction.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from assay.data import STAGE, closed_cases, load_transactions

PATTERNS = ["card_not_present_fraud", "card_not_present_new_device", "out_of_region_use", "account_takeover"]
CATS = ["ProductCD", "channel", "id_15", "id_23", "M4", "M5", "M6", "DeviceType", "card4", "card6", "P_emaildomain"]
NUM = ["TransactionAmt", "dist1", "D1", "C1", "C13", "C14", "risk_score", "region_new", "device_new",
       "card_n_prior", "card_regions_prior", "amt_vs_card_median", "hour"]
PATH = STAGE / "pattern_model.txt"


def novelty(tx: pd.DataFrame) -> pd.DataFrame:
    """Per transaction: has this card been seen in this billing region / on
    this device before? Computed causally (only earlier transactions)."""
    tx = tx.sort_values("ts").copy()
    tx["hour"] = pd.to_datetime(tx.ts).dt.hour
    tx["card_n_prior"] = tx.groupby("card_id").cumcount()
    reg = tx.card_id + "|" + tx.addr1.astype(str)
    tx["region_new"] = (~reg.duplicated()).astype(int).where(tx.addr1.notna(), -1)
    dev = tx.card_id + "|" + tx.DeviceInfo.astype(str)
    tx["device_new"] = (~dev.duplicated()).astype(int).where(tx.DeviceInfo.notna(), -1)
    first_reg = ~reg.duplicated() & tx.addr1.notna()
    tx["card_regions_prior"] = first_reg.groupby(tx.card_id).cumsum() - first_reg
    med = tx.groupby("card_id").TransactionAmt.transform(lambda s: s.expanding().median().shift(1))
    tx["amt_vs_card_median"] = tx.TransactionAmt / med
    return tx


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CATS + NUM].copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    return X


def train() -> dict:
    tx = novelty(load_transactions())
    cc = closed_cases()
    cf = cc[cc.pattern.isin(PATTERNS)][["first_fraud_txn_id", "pattern"]].dropna()
    cf["TransactionID"] = cf.first_fraud_txn_id.astype(int)
    d = cf.merge(tx, on="TransactionID")
    y = d.pattern.map({p: i for i, p in enumerate(PATTERNS)}).values
    X = _frame(d)
    params = dict(objective="multiclass", num_class=len(PATTERNS), learning_rate=0.05, num_leaves=31,
                  min_child_samples=20, verbose=-1, seed=7)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(d))
    pred = np.zeros((len(d), len(PATTERNS)))
    for k in range(5):
        m = lgb.train(params, lgb.Dataset(X[fold != k], y[fold != k]), 300)
        pred[fold == k] = m.predict(X[fold == k])
    acc = float((pred.argmax(1) == y).mean())
    final = lgb.train(params, lgb.Dataset(X, y), 300)
    final.save_model(str(PATH))
    conf = pd.crosstab(pd.Series(y).map(dict(enumerate(PATTERNS))),
                       pd.Series(pred.argmax(1)).map(dict(enumerate(PATTERNS))))
    return {"n": int(len(d)), "cv_accuracy": acc, "confusion": conf.to_dict()}


class PatternModel:
    def __init__(self):
        self.m = lgb.Booster(model_file=str(PATH))
        self.tx = novelty(load_transactions()).set_index("TransactionID")

    def predict(self, txn_id: int) -> dict[str, float]:
        X = _frame(self.tx.loc[[txn_id]].reset_index())
        p = self.m.predict(X)[0]
        return {k: float(v) for k, v in zip(PATTERNS, p)}

    def row(self, txn_id: int) -> pd.Series:
        return self.tx.loc[txn_id]


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, default=str))
