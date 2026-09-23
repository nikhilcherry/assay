"""Transaction-level fraud model, trained only on the bank's closed cases.

The closed-case file labels 14,055 transactions as confirmed fraud between
July and October -- 3.4% of the period's 417,404 transactions, which is the
full fraud rate of the underlying IEEE-CIS data. So a July-October transaction
that no closed case names is, to a very good approximation, legitimate, and
the period is a complete labelled training set.

This module turns that into the one number the investigation leans on: a
probability that a transaction belongs to a fraud episode, validated on a
held-out month and calibrated with isotonic regression so that 0.3 means
roughly three in ten.

Nothing here reads the original Kaggle labels. The only supervision is
closed_cases_history.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from assay.data import STAGE, closed_case_labels, load_transactions

CAT_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "channel",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_15", "id_23", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
    "id_12", "id_16", "id_28", "id_29", "id_34", "id_35", "id_36", "id_37", "id_38",
]
DROP = {"TransactionID", "ts", "customer_id", "card_id", "label", "uid", "TransactionDT", "day", "month", "c6", "k"}

MODEL_PATH = STAGE / "model.txt"
ISO_PATH = STAGE / "isotonic.json"
SCORES_PATH = STAGE / "scores.parquet"
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "MODEL_REPORT.json"


def features(tx: pd.DataFrame) -> pd.DataFrame:
    """Raw Vesta columns plus holder-level aggregates.

    `customer_id` is an issuer bucket (card1), not a person. The holder key
    below -- card, billing region, and the account-open day implied by D1 --
    is what makes "is this amount normal for them" meaningful.
    """
    df = tx.copy()
    df["day"] = df["TransactionDT"] // 86400
    df["D1n"] = df["day"] - df["D1"]
    df["uid"] = df["card_id"] + "_" + df["addr1"].astype(str) + "_" + df["D1n"].astype(str)
    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    df["amt_cents"] = (df["TransactionAmt"] * 100 % 100).round()
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    for key in ("uid", "card_id"):
        g = df.groupby(key)["TransactionAmt"]
        df[f"{key}_n"] = g.transform("size")
        df[f"{key}_amt_mean"] = g.transform("mean")
        df[f"{key}_amt_std"] = g.transform("std")
        df[f"{key}_amt_ratio"] = df["TransactionAmt"] / df[f"{key}_amt_mean"]
    for col in ("P_emaildomain", "addr1", "DeviceInfo", "id_33", "dist1"):
        df[f"{col}_uid_nuniq"] = df.groupby("uid")[col].transform("nunique")
    for col in ("uid", "card_id", "addr1", "P_emaildomain", "DeviceInfo"):
        df[f"{col}_freq"] = df[col].map(df[col].value_counts())
    for c in CAT_COLS:
        if c in df:
            df[c] = df[c].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP]


def _fit(X, y, rounds=600):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_cat_to_onehot=8, cat_smooth=20, verbose=-1, num_threads=12, seed=7)
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)


def train() -> dict:
    tx = load_transactions()
    labels = closed_case_labels()
    df = features(tx)
    df["label"] = df["TransactionID"].map(labels).fillna(0).astype(int)
    df["month"] = df["ts"].str[:7]
    cols = feature_columns(df)
    labelled = df[df["month"] <= "2016-10"]

    # 1. Honest estimate: train Jul-Sep, score October, never seen.
    tr, va = labelled[labelled.month <= "2016-09"], labelled[labelled.month == "2016-10"]
    m = _fit(tr[cols], tr.label)
    p_va = m.predict(va[cols])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_va, va.label)

    # Risk-score baseline on the same month, for the report.
    report = {
        "train": "2016-07..2016-09", "validate": "2016-10",
        "n_train": int(len(tr)), "n_validate": int(len(va)), "validate_fraud_rate": float(va.label.mean()),
        "auc_model": float(roc_auc_score(va.label, p_va)),
        "auc_bank_risk_score": float(roc_auc_score(va.label, va.risk_score)),
        "ap_model": float(average_precision_score(va.label, p_va)),
        "ap_bank_risk_score": float(average_precision_score(va.label, va.risk_score)),
    }
    # Calibration measured by 2-fold cross-fitting the isotonic map on October,
    # so the reliability table is not the isotonic fit grading itself.
    half = np.arange(len(va)) % 2 == 0
    cal = np.empty(len(va))
    for a, b in ((half, ~half), (~half, half)):
        cal[b] = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_va[a], va.label.values[a]).predict(p_va[b])
    report["brier_calibrated"] = float(brier_score_loss(va.label, cal))
    bins = pd.cut(cal, [0, .05, .15, .3, .5, .7, .85, 1.0], include_lowest=True)
    rel = pd.DataFrame({"p": cal, "y": va.label.values, "bin": bins}).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    report["reliability"] = [
        {"bin": str(i), **{k: (int(v) if k == "n" else round(float(v), 4)) for k, v in r.items()}}
        for i, r in rel.iterrows()]

    # 2. Production model: all four labelled months, same calibration map.
    final = _fit(labelled[cols], labelled.label)
    final.save_model(str(MODEL_PATH))
    ISO_PATH.write_text(json.dumps({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}))

    raw = final.predict(df[cols])
    # In-sample months get the out-of-time model's view where possible would be
    # nicer; for Jul-Oct we keep the production score, which is only used for
    # context (the exam cases are all Nov-Dec, fully out of sample).
    out = pd.DataFrame({"TransactionID": df.TransactionID, "uid": df.uid, "p_raw": raw,
                        "p_fraud": iso.predict(raw)})
    # Out-of-time view of October (model never saw it), used to validate the
    # episode builder against the bank's own October cases without leakage.
    oot = pd.Series(cal, index=va.TransactionID.values)
    out["p_oot"] = out.TransactionID.map(oot)
    out.to_parquet(SCORES_PATH, index=False)
    imp = pd.Series(final.feature_importance("gain"), index=cols).sort_values(ascending=False)
    report["top_features_by_gain"] = [str(c) for c in imp.index[:25]]
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
