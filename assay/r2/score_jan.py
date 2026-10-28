"""Baseline January scores from the Round 1 model, unchanged.

The Round 1 LightGBM (trained Jul-Oct on closed-case labels, OOT October AUC
0.964) is applied to January with the same features, computed over the full
Jul-Jan frame (holder aggregates need the history). No retraining: this is the
honest fallback until B1's retrain is validated. Writes
data/stage/r2/scores_r2.parquet [TransactionID, p_raw, p_fraud, model].
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from assay.data import load_transactions
from assay.model import ISO_PATH, MODEL_PATH, features
from assay.r2.frames import R2_STAGE, _jan, tx_all


def main() -> None:
    old = load_transactions()
    jan = _jan()
    ids = tx_all().set_index("TransactionID").card_id
    jan["card_id"] = jan.TransactionID.map(ids)
    df = features(pd.concat([old, jan[[c for c in old.columns if c in jan.columns]]], ignore_index=True))
    m = lgb.Booster(model_file=str(MODEL_PATH))
    cols = m.feature_name()
    j = df[df.TransactionID.isin(set(jan.TransactionID))]
    raw = m.predict(j[cols])
    iso = json.loads(ISO_PATH.read_text())
    p = np.interp(raw, iso["x"], iso["y"])
    out = pd.DataFrame({"TransactionID": j.TransactionID.values, "p_raw": raw, "p_fraud": p, "model": "r1_lgbm_unchanged"})
    out.to_parquet(R2_STAGE / "scores_r2.parquet", index=False)
    print(f"scored {len(out):,} jan txns; mean p={p.mean():.4f}; share p>0.5 = {(p > 0.5).mean():.4f}")


if __name__ == "__main__":
    main()
