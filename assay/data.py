"""Loading and holder resolution.

The raw CSVs are staged once to parquet (`python -m assay.data`). Card IDs are
not a column in transactions.csv; they are derived here and checked against
every card ID the closed cases and the case pack name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAGE = DATA / "stage"

IDENTITY_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_01", "id_02", "id_05", "id_06",
                 "id_12", "id_13", "id_15", "id_16", "id_17", "id_19", "id_20", "id_23", "id_28",
                 "id_29", "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]


def stage() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    tx = pd.read_csv(DATA / "transactions.csv", low_memory=False)
    idt = pd.read_csv(DATA / "identity.csv", low_memory=False)
    tx = tx.merge(idt[IDENTITY_COLS], on="TransactionID", how="left")
    tx["card_id"] = derive_card_ids(tx)
    tx.to_parquet(STAGE / "tx.parquet", index=False)


def derive_card_ids(tx: pd.DataFrame) -> pd.Series:
    """`customer_id` is card1. Within a customer, a card is a distinct card6
    (credit / debit / missing), numbered K1, K2... from the least-used type to
    the most-used. Found empirically: this reproduces 99.2% of the 14,975 card
    IDs named in closed cases (127 misses, all on customers where two card
    types are close in volume) and all 20 case-pack card IDs. Where a closed
    case names a card for a transaction, that label wins over the rule.
    """
    c6 = tx["card6"].fillna("NA")
    counts = (pd.DataFrame({"customer_id": tx.customer_id, "c6": c6}).value_counts()
              .reset_index(name="n").sort_values(["customer_id", "n", "c6"]))
    counts["k"] = counts.groupby("customer_id").cumcount() + 1
    key = pd.DataFrame({"customer_id": tx.customer_id, "c6": c6})
    k = key.merge(counts, on=["customer_id", "c6"], how="left")["k"].values
    derived = tx.customer_id + "-K" + pd.Series(k, index=tx.index).astype(str)

    # Ground truth from the bank's own records overrides the rule.
    cc = closed_cases()
    named = {}
    for r in cc.itertuples():
        for t in r.txn_list:
            named[t] = r.card_id
    # Propagate a named card to every transaction of that customer and card type.
    fix = {}
    tid_to_row = pd.Series(range(len(tx)), index=tx.TransactionID)
    for t, card in named.items():
        if t in tid_to_row.index:
            i = tid_to_row[t]
            fix[(tx.customer_id.iat[i], c6.iat[i])] = card
    if fix:
        mapped = pd.Series([fix.get(kv) for kv in zip(tx.customer_id, c6)], index=tx.index)
        derived = mapped.fillna(derived)
    return derived


@lru_cache(maxsize=1)
def closed_cases() -> pd.DataFrame:
    cc = pd.read_csv(DATA / "closed_cases_history.csv")
    cc["txn_list"] = cc.txn_ids.astype(str).map(
        lambda s: [int(float(t)) for t in s.split("|") if t not in ("", "nan")])
    cc["connected_list"] = cc.connected_card_ids.fillna("").map(lambda s: [c for c in s.split("|") if c])
    return cc


def closed_case_labels() -> pd.Series:
    """TransactionID -> 1 (in a confirmed-fraud case) / 0 (in a cleared case)."""
    cc = closed_cases()
    lab = {}
    for r in cc.itertuples():
        for t in r.txn_list:
            if r.outcome == "confirmed_fraud":
                lab[t] = 1
            else:
                lab.setdefault(t, 0)
    return pd.Series(lab)


@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    return pd.read_parquet(STAGE / "tx.parquet")


def holder_ids(tx: pd.DataFrame) -> pd.Series:
    """Holder = card + billing region + account-open day (transaction day minus
    D1). Missing parts are kept as 'na' so every transaction has a holder."""
    day = tx["TransactionDT"] // 86400
    d1n = (day - tx["D1"]).map(lambda x: "na" if pd.isna(x) else str(int(x)))
    reg = tx["addr1"].map(lambda x: "na" if pd.isna(x) else str(int(x)))
    return tx["card_id"].astype(str) + "|" + reg + "|" + d1n


def case_pack() -> pd.DataFrame:
    return pd.read_csv(DATA / "case_pack.csv")


if __name__ == "__main__":
    stage()
    print("staged", STAGE / "tx.parquet")
