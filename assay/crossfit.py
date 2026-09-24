"""Month-held-out scores for July-October, so likelihood ratios can be measured.

    python -m assay.crossfit        # -> data/stage/scores_oof.parquet   (~20 min)

The production model is trained on all four labelled months, so its scores on
those months have seen the answers: a detector's lift measured against them
collapses toward 1. The agent applies likelihood ratios to exam-period scores
the model has never seen, so the fair baseline is a score from a model that
never saw that month either. Each month here is scored by a model trained on the
other three, then mapped through the production isotonic calibration.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from assay.data import STAGE, closed_case_labels, load_transactions
from assay.model import ISO_PATH, _fit, feature_columns, features

OUT = STAGE / "scores_oof.parquet"
MONTHS = ["2016-07", "2016-08", "2016-09", "2016-10"]


def main() -> int:
    df = features(load_transactions())
    df["label"] = df["TransactionID"].map(closed_case_labels()).fillna(0).astype(int)
    df["month"] = df["ts"].str[:7]
    cols = feature_columns(df)
    iso = json.loads(ISO_PATH.read_text())
    parts = []
    for m in MONTHS:
        tr = df[df.month.isin([x for x in MONTHS if x != m])]
        te = df[df.month == m]
        raw = _fit(tr[cols], tr.label).predict(te[cols])
        parts.append(pd.DataFrame({"TransactionID": te.TransactionID.values, "month": m,
                                   "p_oof": np.interp(raw, iso["x"], iso["y"])}))
        print(m, len(te), f"mean p_oof {parts[-1].p_oof.mean():.4f} vs label rate {te.label.mean():.4f}", flush=True)
    pd.concat(parts).to_parquet(OUT, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
