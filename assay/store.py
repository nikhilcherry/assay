"""The investigation's view of the data: a small set of named graph queries.

Every method here corresponds one-to-one to an installed GSQL query in
`graph/queries/` and is logged as a tool call, so each evidence item in an
answer file can cite the query that produced it (`ref`).

Two implementations share this interface:

* `LocalStore` answers from the staged parquet files. It is the reference
  implementation, used by the tests and when no database is running.
* `TigerGraphStore` (assay/tg.py) answers the same questions from TigerGraph
  via installed queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from assay.data import STAGE, closed_cases, holder_ids, load_transactions

TXN_COLS = ["TransactionID", "ts", "card_id", "customer_id", "TransactionAmt", "ProductCD", "channel",
            "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain", "risk_score", "DeviceType",
            "DeviceInfo", "id_30", "id_31", "id_33", "id_15", "id_23", "M4", "M5", "M6", "D1", "C1",
            "TransactionDT"]


def device_profile(row) -> str:
    """The brief's DeviceProfile: DeviceInfo + OS + browser + screen."""
    parts = [row.get("DeviceInfo"), row.get("id_30"), row.get("id_31"), row.get("id_33")]
    if all(pd.isna(p) for p in parts):
        return ""
    return " | ".join("?" if pd.isna(p) else str(p) for p in parts)


@dataclass
class Call:
    name: str
    args: dict

    @property
    def ref(self) -> str:
        a = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"query:{self.name}({a})"


@dataclass
class Store:
    calls: list[Call] = field(default_factory=list)

    def _log(self, name: str, **args) -> str:
        c = Call(name, args)
        self.calls.append(c)
        return c.ref

    def reset(self) -> None:
        self.calls = []


class LocalStore(Store):
    """Reference implementation over the staged parquet files."""

    def __init__(self):
        super().__init__()
        self.memory: list[dict] = []   # FraudCase vertices this process has written
        tx = load_transactions()[TXN_COLS].copy()
        sc = pd.read_parquet(STAGE / "scores.parquet")
        tx = tx.merge(sc[["TransactionID", "p_fraud"]], on="TransactionID", how="left")
        tx["uid"] = holder_ids(tx)
        tx["t"] = pd.to_datetime(tx["ts"])
        tx["device_profile"] = [
            device_profile(r) for r in tx[["DeviceInfo", "id_30", "id_31", "id_33"]].to_dict("records")]
        self.tx = tx.set_index("TransactionID", drop=False).sort_values("t")
        self.by_card = {k: v for k, v in self.tx.groupby("card_id")}
        self.by_uid = {k: v for k, v in self.tx.groupby("uid")}
        dp = self.tx[self.tx.device_profile != ""]
        self.by_device = {k: v for k, v in dp.groupby("device_profile")}
        self.cc = closed_cases()

    # -- queries -------------------------------------------------------------
    def get_transaction(self, txn_id: int):
        ref = self._log("get_transaction", txn_id=txn_id)
        return self.tx.loc[txn_id], ref

    def card_window(self, card_id: str, t0, hours_before: float, hours_after: float):
        ref = self._log("card_window", card_id=card_id, hours_before=hours_before, hours_after=hours_after)
        d = self.by_card.get(card_id, self.tx.iloc[:0])
        w = d[(d.t >= t0 - pd.Timedelta(hours=hours_before)) & (d.t <= t0 + pd.Timedelta(hours=hours_after))]
        return w, ref

    def holder_history(self, uid: str):
        ref = self._log("holder_history", holder_id=uid)
        return self.by_uid.get(uid, self.tx.iloc[:0]), ref

    def card_history(self, card_id: str, before):
        ref = self._log("card_history", card_id=card_id)
        d = self.by_card.get(card_id, self.tx.iloc[:0])
        return d[d.t < before], ref

    def closed_cases_by_device(self, profile: str):
        """Closed cases whose transactions used this device profile."""
        ref = self._log("closed_cases_by_device", device_profile=profile)
        d = self.by_device.get(profile, self.tx.iloc[:0])
        ids = set(d.TransactionID)
        return self.cc[self.cc.txn_list.map(lambda l: bool(ids.intersection(l)))], ref

    def device_neighbors(self, profile: str, t0, days: float):
        ref = self._log("device_neighbors", device_profile=profile, days=days)
        d = self.by_device.get(profile, self.tx.iloc[:0])
        w = d[(d.t >= t0 - pd.Timedelta(days=days)) & (d.t <= t0 + pd.Timedelta(days=days))]
        return w, ref

    def closed_cases_for_card(self, card_id: str):
        ref = self._log("closed_cases_for_card", card_id=card_id)
        return self.cc[self.cc.card_id == card_id], ref

    def closed_cases_touching(self, card_ids: list[str]):
        ref = self._log("closed_cases_touching", n_cards=len(card_ids))
        s = set(card_ids)
        cc = self.cc
        return cc[cc.card_id.isin(s) | cc.connected_list.map(lambda l: bool(s.intersection(l)))], ref

    def closed_cases_like(self, pattern: str, device_profile: str = "", limit: int = 3, product: str = "",
                          amount: float = 0.0):
        """Closed cases of the same typology, ranked by how closely their
        transactions resemble this one: a named device model first, then the
        same product code, then nearest amount."""
        dev = device_profile.split(" | ")[0].split(" Build")[0] if device_profile else ""
        ref = self._log("similar_prior_cases", pattern=pattern, device=dev or "-", product=product or "-",
                        amount=round(amount, 2), k=limit)
        cc = self.cc[self.cc.pattern == pattern]
        if dev and dev not in ("Windows", "MacOS", "iOS Device", "Trident/7.0"):
            hit = cc[cc.analyst_notes.str.contains(dev, regex=False, na=False)]
            if len(hit):
                cc = hit
        rows = []
        for r in cc.itertuples():
            t = self.tx.loc[[i for i in r.txn_list if i in self.tx.index]]
            if not len(t):
                continue
            same = t[t.ProductCD == product] if product else t
            d = (same.TransactionAmt - amount).abs().min() if len(same) else 1e9
            rows.append((0 if len(same) else 1, d, r.case_id))
        ids = [c for _, _, c in sorted(rows)[:limit]]
        return self.cc.set_index("case_id").loc[ids].reset_index(), ref

    # -- case memory ---------------------------------------------------------
    def prior_fraud_cases(self, card_id: str):
        ref = self._log("prior_fraud_cases", card_id=card_id)
        return [m for m in self.memory if card_id == m["card_id"] or card_id in m["connected"]], ref

    def write_case(self, answer: dict, card_id: str = "") -> str:
        gid = f"ASSAY-{answer['case_id']}"
        c = answer["case"]
        self.memory.append({"v_id": gid, "card_id": card_id, "connected": set(c["connected_card_ids"]),
                            "attributes": {"verdict": c["verdict"], "pattern": c["pattern"],
                                           "alert_id": answer["case_id"]}})
        self._log("write_case", graph_case_id=gid)
        return gid
