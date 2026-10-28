"""LocalStore over the full July 2016 - January 2017 picture.

Same interface and query names as assay.store.LocalStore, so the Round 1
Investigator runs unchanged; only the data underneath grows. p_fraud is the
Round 1 model's score for Jul-Dec and B1's retrained score for January
(data/stage/r2/scores_r2.parquet). A January txn with no score is a hard
error, never a silent zero (fail closed).
"""

from __future__ import annotations

import pandas as pd

from assay.data import STAGE
from assay.r2.frames import R2_STAGE, closed_cases, tx_all
from assay.store import TXN_COLS, LazyGroups, LocalStore, Store


class MissingScore(RuntimeError):
    pass


class R2LocalStore(LocalStore):
    def __init__(self):
        Store.__init__(self)
        self.memory = []
        tx = tx_all().rename(columns={"amount": "TransactionAmt"})
        tx = tx[[c for c in TXN_COLS if c in tx.columns] + ["t", "uid", "device_profile", "month_src"]].copy()
        old = pd.read_parquet(STAGE / "scores.parquet")[["TransactionID", "p_fraud"]]
        p = R2_STAGE / "scores_r2.parquet"
        new = pd.read_parquet(p)[["TransactionID", "p_fraud"]] if p.exists() else old.iloc[:0]
        sc = pd.concat([old[~old.TransactionID.isin(new.TransactionID)], new])
        tx = tx.merge(sc, on="TransactionID", how="left")
        self.tx = tx.set_index("TransactionID", drop=False).sort_values("t", kind="stable")
        self.by_card = LazyGroups(self.tx, "card_id")
        self.by_uid = LazyGroups(self.tx, "uid")
        self.by_device = LazyGroups(self.tx[self.tx.device_profile != ""], "device_profile")
        self.cc = closed_cases()

    # -- as-of view ---------------------------------------------------------------
    # An investigation opened at T sees only transactions up to T and cases closed by T.
    # Round 1's windows also looked forward from the flagged charge; in January that would
    # read activity that had not happened yet when the case was opened.
    as_of: pd.Timestamp | None = None

    def _tx_asof(self, d: pd.DataFrame) -> pd.DataFrame:
        return d if self.as_of is None else d[d.t <= self.as_of]

    def _cc_asof(self, cc: pd.DataFrame) -> pd.DataFrame:
        return cc if self.as_of is None else cc[pd.to_datetime(cc.closed_at) <= self.as_of]

    def card_window(self, card_id, t0, hours_before, hours_after):
        w, ref = super().card_window(card_id, t0, hours_before, hours_after)
        return self._tx_asof(w), ref

    def holder_history(self, uid):
        w, ref = super().holder_history(uid)
        return self._tx_asof(w), ref

    def device_neighbors(self, profile, t0, days):
        w, ref = super().device_neighbors(profile, t0, days)
        return self._tx_asof(w), ref

    def closed_cases_for_card(self, card_id):
        cc, ref = super().closed_cases_for_card(card_id)
        return self._cc_asof(cc), ref

    def closed_cases_by_device(self, profile):
        cc, ref = super().closed_cases_by_device(profile)
        return self._cc_asof(cc), ref

    def get_transaction(self, txn_id: int):
        row, ref = super().get_transaction(txn_id)
        if pd.isna(row.p_fraud):
            raise MissingScore(f"txn {txn_id} has no model score (run B1: assay.r2.model_r2) -- refusing to guess")
        return row, ref


def r2_pattern_model():
    """Round 1 pattern classifier with January rows added to its novelty frame."""
    import lightgbm as lgb

    from assay.data import load_transactions
    from assay.patterns import PATH, PatternModel, novelty
    from assay.r2.frames import _jan

    pm = PatternModel.__new__(PatternModel)
    pm.m = lgb.Booster(model_file=str(PATH))
    old = load_transactions()
    jan = _jan()
    jan["card_id"] = jan.TransactionID.map(tx_all().set_index("TransactionID").card_id)
    # Only the columns the classifier and its novelty features read: all 397 cost ~8 GB.
    from assay.patterns import CATS, NUM
    keep = ["TransactionID", "ts", "card_id", "addr1", "DeviceInfo"] + [c for c in CATS + NUM if c in old.columns]
    keep = list(dict.fromkeys(keep))
    old = old[keep]
    pm.tx = novelty(pd.concat([old, jan[[c for c in keep if c in jan.columns]]], ignore_index=True)).set_index("TransactionID")
    return pm
