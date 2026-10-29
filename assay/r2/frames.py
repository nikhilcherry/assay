"""One slim frame of every transaction, July 2016 to January 2017, plus the
final-round closed cases and case pack. All final-round blocks read from here.

Card ids for January reuse the Round 1 (customer, card6) -> card mapping so
that a card keeps its id across the month boundary; only combinations never
seen before Jan fall back to the ranking rule in data.derive_card_ids.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from assay.data import DATA, IDENTITY_COLS, STAGE, load_transactions
from assay.store import device_profile

R2 = DATA / "round2" / "share"
R2_STAGE = DATA / "stage" / "r2"

COLS = ["TransactionID", "TransactionDT", "ts", "card_id", "customer_id", "card4", "card6",
        "TransactionAmt", "ProductCD", "channel", "addr1", "addr2", "dist1", "P_emaildomain",
        "R_emaildomain", "risk_score", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
        "id_15", "id_23", "M4", "M5", "M6", "D1", "C1"]


def _jan() -> pd.DataFrame:
    tx = pd.read_csv(R2 / "transactions_2017q1.csv", low_memory=False)
    idt = pd.read_csv(R2 / "identity_2017q1.csv", low_memory=False)
    return tx.merge(idt[[c for c in IDENTITY_COLS if c in idt.columns]], on="TransactionID", how="left")


def _jan_card_ids(old: pd.DataFrame, jan: pd.DataFrame) -> pd.Series:
    m = (old.assign(c6=old.card6.fillna("NA")).groupby(["customer_id", "c6"]).card_id
         .agg(lambda s: s.mode().iat[0]).to_dict())
    c6 = jan.card6.fillna("NA")
    out = pd.Series([m.get(k) for k in zip(jan.customer_id, c6)], index=jan.index, dtype=object)
    miss = out.isna()
    if miss.any():   # new (customer, card type): next free K number for that customer
        used = old.groupby("customer_id").card_id.agg(lambda s: {c.rsplit("-K", 1)[1] for c in s})
        nxt = {}
        for i in out[miss].index:
            cust, k6 = jan.customer_id.at[i], c6.at[i]
            if (cust, k6) not in nxt:
                taken = {int(x) for x in used.get(cust, set())} | {int(v.rsplit("-K", 1)[1]) for (c, _), v in nxt.items() if c == cust}
                nxt[(cust, k6)] = f"{cust}-K{max(taken, default=0) + 1}"
            out.at[i] = nxt[(cust, k6)]
    return out


def stage() -> None:
    R2_STAGE.mkdir(parents=True, exist_ok=True)
    old = load_transactions()
    jan = _jan()
    jan["card_id"] = _jan_card_ids(old, jan)
    both = pd.concat([old[COLS].assign(month_src="r1"), jan[COLS].assign(month_src="jan")], ignore_index=True)
    both["t"] = pd.to_datetime(both.ts)
    both["device_profile"] = [device_profile(r) for r in both[["DeviceInfo", "id_30", "id_31", "id_33"]].to_dict("records")]
    day = both.TransactionDT // 86400
    both["uid"] = (both.card_id.astype(str) + "|" + both.addr1.map(lambda x: "na" if pd.isna(x) else str(int(x)))
                   + "|" + (day - both.D1).map(lambda x: "na" if pd.isna(x) else str(int(x))))
    both = both.rename(columns={"TransactionAmt": "amount"}).sort_values("t", kind="stable")
    both.to_parquet(R2_STAGE / "tx_all.parquet", index=False)


@lru_cache(maxsize=1)
def tx_all() -> pd.DataFrame:
    """Columns: TransactionID(int) t(datetime) card_id customer_id uid amount ProductCD channel
    addr1 addr2 dist1 P_emaildomain R_emaildomain risk_score device_profile id_15 id_23 DeviceType
    card4 card6 M4 M5 M6 D1 C1 month_src('r1'|'jan')."""
    return pd.read_parquet(R2_STAGE / "tx_all.parquet")


@lru_cache(maxsize=1)
def closed_cases() -> pd.DataFrame:
    """The 5,585 final-round closed cases. Adds txn_list (list[int]) and connected_list (list[str])."""
    cc = pd.read_csv(R2 / "closed_cases_history.csv")
    cc["txn_list"] = cc.txn_ids.fillna("").astype(str).map(
        lambda s: [int(float(t)) for t in s.split("|") if t not in ("", "nan")])
    cc["connected_list"] = cc.connected_card_ids.fillna("").map(lambda s: [c for c in s.split("|") if c])
    return cc


def case_pack() -> pd.DataFrame:
    return pd.read_csv(R2 / "cases" / "case_pack_final.csv")


if __name__ == "__main__":
    stage()
    t = tx_all(); cp = case_pack()
    print(t.shape, t.t.min(), t.t.max(), (t.month_src == "jan").sum())
    j = t.set_index("TransactionID")
    # A case row is untrusted input: an unparseable flagged id is reported here, not a crash.
    fid = pd.to_numeric(cp.flagged_txn_id, errors="coerce")
    miss = []
    for r, f in zip(cp.itertuples(), fid):
        real = j.at[int(f), "card_id"] if pd.notna(f) and int(f) in j.index else None
        if real != r.card_id:
            miss.append((r.case_id, r.card_id, real or f"txn missing ({r.flagged_txn_id!r})"))
    print(f"case pack card ids resolved: {len(cp) - len(miss)}/{len(cp)}")
    for m in miss:
        print("  miss", *m)
