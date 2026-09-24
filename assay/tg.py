"""TigerGraph: load the data, install the queries, and serve investigations.

    python -m assay.tg prepare   # write data/tg/*.csv from the staged parquet
    python -m assay.tg setup     # schema + loading job + queries, via gsql in the container
    python -m assay.tg check     # row counts per vertex type

`TigerGraphStore` has the same interface as `LocalStore`: every method is one
installed query, called over REST++, and its result becomes the same frame the
detectors already consume. `write_case` upserts the finished investigation as
a FraudCase vertex with edges to the card, the transactions it cites, the
devices it implicates and the closed cases it drew on -- the case memory the
next investigation retrieves through `prior_fraud_cases`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

from assay.data import DATA, ROOT, STAGE, closed_cases, holder_ids, load_transactions
from assay.store import Store, device_profile

TG_DIR = DATA / "tg"
CONTAINER = os.environ.get("ASSAY_TG_CONTAINER", "assay-tg")
HOST = os.environ.get("ASSAY_TG_HOST", "http://127.0.0.1")
GRAPH = "assay"


def device_id(profile: str) -> str:
    return "D" + hashlib.sha1(profile.encode()).hexdigest()[:10] if profile else ""


# --------------------------------------------------------------- prepare
def prepare() -> None:
    TG_DIR.mkdir(parents=True, exist_ok=True)
    tx = load_transactions()
    sc = pd.read_parquet(STAGE / "scores.parquet")[["TransactionID", "p_fraud"]]
    tx = tx.merge(sc, on="TransactionID")
    tx["uid"] = holder_ids(tx)
    prof = [device_profile(r) for r in tx[["DeviceInfo", "id_30", "id_31", "id_33"]].to_dict("records")]
    out = pd.DataFrame({
        "txn_id": tx.TransactionID.astype(str), "ts": tx.ts, "amount": tx.TransactionAmt,
        "product": tx.ProductCD, "channel": tx.channel,
        "addr1": tx.addr1.fillna(-1).astype(int), "dist1": tx.dist1.fillna(-1),
        "p_email": tx.P_emaildomain.fillna(""), "r_email": tx.R_emaildomain.fillna(""),
        "risk_score": tx.risk_score, "p_fraud": tx.p_fraud.round(5),
        "id_15": tx.id_15.fillna(""), "id_23": tx.id_23.fillna(""), "device_type": tx.DeviceType.fillna(""),
        "device_info": tx.DeviceInfo.fillna(""), "os": tx.id_30.fillna(""), "browser": tx.id_31.fillna(""),
        "screen": tx.id_33.fillna(""), "device_profile": prof,
        "m4": tx.M4.fillna(""), "m5": tx.M5.fillna(""), "m6": tx.M6.fillna(""),
        "d1": tx.D1.fillna(-1), "c1": tx.C1.fillna(-1),
        "card_id": tx.card_id, "customer_id": tx.customer_id, "holder_id": tx.uid,
    })
    out["device_id"] = [device_id(p) for p in prof]
    out.to_csv(TG_DIR / "txn.csv", index=False)

    card = tx.groupby("card_id").agg(customer_id=("customer_id", "first"), card_type=("card6", "first"),
                                    network=("card4", "first")).reset_index().fillna("")
    card.to_csv(TG_DIR / "card.csv", index=False)

    d = out[out.device_id != ""].groupby(["device_id", "device_profile"]).agg(
        n_cards=("card_id", "nunique"), n_txns=("txn_id", "size")).reset_index()
    d.rename(columns={"device_profile": "profile"}).to_csv(TG_DIR / "device.csv", index=False)

    s = tx.sort_values(["card_id", "ts"])[["card_id", "TransactionID", "TransactionDT"]]
    nxt = pd.DataFrame({"src": s.TransactionID.astype(str).values[:-1], "dst": s.TransactionID.astype(str).values[1:],
                        "gap_seconds": (s.TransactionDT.values[1:] - s.TransactionDT.values[:-1]).astype(int),
                        "same": s.card_id.values[:-1] == s.card_id.values[1:]})
    nxt[nxt.same].drop(columns="same").to_csv(TG_DIR / "next.csv", index=False)

    cc = closed_cases()
    c = cc.drop(columns=["txn_list", "connected_list"]).copy()
    c["first_fraud_txn_id"] = c.first_fraud_txn_id.map(lambda x: "" if pd.isna(x) else str(int(x)))
    c = c.fillna("")
    c.to_csv(TG_DIR / "closed_case.csv", index=False)
    pd.DataFrame([(r.case_id, str(t)) for r in cc.itertuples() for t in r.txn_list],
                 columns=["case_id", "txn_id"]).to_csv(TG_DIR / "closed_case_txn.csv", index=False)
    pd.DataFrame([(r.case_id, k) for r in cc.itertuples() for k in r.connected_list],
                 columns=["case_id", "card_id"]).to_csv(TG_DIR / "closed_case_connected.csv", index=False)
    print("prepared", sorted(p.name for p in TG_DIR.glob("*.csv")))


# ------------------------------------------------------------------ gsql
def gsql_file(path) -> str:
    dst = f"/tmp/{path.name}"
    subprocess.run(["docker", "--context", "default", "cp", str(path), f"{CONTAINER}:{dst}"], check=True)
    r = subprocess.run(["docker", "--context", "default", "exec", "-u", "tigergraph", CONTAINER, "bash", "-c",
                        f"/home/tigergraph/tigergraph/app/cmd/gsql {dst}"], capture_output=True, text=True)
    return r.stdout + r.stderr


def setup() -> None:
    for f in ("schema.gsql", "load.gsql", "queries/investigation.gsql"):
        print(f"== {f}")
        print(gsql_file(ROOT / "graph" / f)[-3000:])


# ----------------------------------------------------------------- store
class TigerGraphStore(Store):
    """Same questions as LocalStore, answered by installed GSQL queries."""

    TXN_MAP = {"txn_id": "TransactionID", "amount": "TransactionAmt", "product": "ProductCD",
               "p_email": "P_emaildomain", "r_email": "R_emaildomain", "device_type": "DeviceType",
               "device_info": "DeviceInfo", "os": "id_30", "browser": "id_31", "screen": "id_33",
               "m4": "M4", "m5": "M5", "m6": "M6", "d1": "D1", "c1": "C1", "holder_id": "uid"}

    def __init__(self):
        super().__init__()
        import pyTigerGraph as tg
        self.conn = tg.TigerGraphConnection(host=HOST, graphname=GRAPH, username=os.environ.get("ASSAY_TG_USER", "tigergraph"),
                                            password=os.environ.get("ASSAY_TG_PASSWORD", "tigergraph"),
                                            restppPort="14240", gsPort="14240")
        try:
            self.conn.getToken(self.conn.createSecret())
        except Exception:
            pass
        self.cc = closed_cases()

    # -- plumbing ----------------------------------------------------------
    def _q(self, name: str, params: dict):
        res = self.conn.runInstalledQuery(name, params=params, timeout=120000)
        return res[0][next(iter(res[0]))]

    def _txns(self, rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(self.TXN_MAP.values()) + ["t", "device_profile", "p_fraud",
                                                                        "card_id", "channel", "addr1"])
        df = pd.DataFrame([r["attributes"] for r in rows]).rename(columns=self.TXN_MAP)
        df["TransactionID"] = df.TransactionID.astype(int)
        df["t"] = pd.to_datetime(df.ts)
        df["addr1"] = df.addr1.where(df.addr1 >= 0)
        for c in ("id_15", "id_23", "DeviceInfo", "id_30", "id_31", "id_33", "P_emaildomain", "R_emaildomain"):
            df[c] = df[c].replace("", None)
        df = df.set_index("TransactionID", drop=False).sort_values("t")
        return df

    def _cases(self, rows) -> pd.DataFrame:
        ids = {r["v_id"] for r in rows}
        return self.cc[self.cc.case_id.isin(ids)]

    @staticmethod
    def _ts(t) -> str:
        return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    # -- the queries -------------------------------------------------------
    def get_transaction(self, txn_id: int):
        ref = self._log("get_transaction", txn_id=txn_id)
        return self._txns(self._q("get_transaction", {"txn": (str(txn_id),)})).iloc[0], ref

    def card_window(self, card_id, t0, hours_before, hours_after):
        ref = self._log("card_window", card_id=card_id, hours_before=hours_before, hours_after=hours_after)
        return self._txns(self._q("card_window", {"card": (card_id,), "t0": self._ts(t0),
                                                  "hours_before": int(hours_before),
                                                  "hours_after": int(hours_after)})), ref

    def card_history(self, card_id, before):
        ref = self._log("card_history", card_id=card_id)
        return self._txns(self._q("card_history", {"card": (card_id,), "before_ts": self._ts(before)})), ref

    def holder_history(self, uid):
        ref = self._log("holder_history", holder_id=uid)
        return self._txns(self._q("holder_history", {"holder": (uid,)})), ref

    def device_neighbors(self, profile, t0, days):
        ref = self._log("device_neighbors", device_profile=profile, days=days)
        return self._txns(self._q("device_neighbors", {"device": (device_id(profile),), "t0": self._ts(t0),
                                                       "days": int(days)})), ref

    def closed_cases_for_card(self, card_id):
        ref = self._log("closed_cases_for_card", card_id=card_id)
        return self._cases(self._q("closed_cases_for_card", {"card": (card_id,)})), ref

    def closed_cases_by_device(self, profile):
        ref = self._log("closed_cases_by_device", device_profile=profile)
        return self._cases(self._q("closed_cases_by_device", {"device": (device_id(profile),)})), ref

    def closed_cases_touching(self, card_ids):
        ref = self._log("closed_cases_touching", n_cards=len(card_ids))
        return self._cases(self._q("closed_cases_touching", {"cards": [(c,) for c in card_ids]})), ref

    def closed_cases_like(self, pattern, device_profile="", limit=3, product="", amount=0.0):
        dev = device_profile.split(" | ")[0].split(" Build")[0] if device_profile else ""
        if dev in ("Windows", "MacOS", "iOS Device", "Trident/7.0"):
            dev = ""
        ref = self._log("similar_prior_cases", pattern=pattern, device=dev or "-", product=product or "-",
                        amount=round(amount, 2), k=limit)
        rows = self._q("similar_prior_cases", {"pattern": pattern, "device": dev, "product": product,
                                               "amount": float(amount), "k": limit})
        ids = [r["v_id"] for r in rows]
        return self.cc.set_index("case_id").loc[ids].reset_index(), ref

    def prior_fraud_cases(self, card_id):
        ref = self._log("prior_fraud_cases", card_id=card_id)
        return [{"v_id": r["v_id"], "attributes": r["attributes"]}
                for r in self._q("prior_fraud_cases", {"card": (card_id,)})], ref

    # -- write-back: the case memory ----------------------------------------
    def reset_memory(self) -> int:
        """Forget earlier runs' FraudCase vertices (and their edges), so a run's case
        memory is causal: an investigation only finds cases this run closed before it.
        The reference store starts empty for the same reason."""
        return self.conn.delVertices("FraudCase")

    def write_case(self, answer: dict, card_id: str = "") -> str:
        c = answer["case"]
        gid = f"ASSAY-{answer['case_id']}"
        final = [x["action"] for x in answer["next_best_actions"]["final"]]
        self.conn.upsertVertex("FraudCase", gid, {
            "alert_id": answer["case_id"], "status": c["status"], "verdict": c["verdict"],
            "fraud_probability": c["fraud_probability"], "pattern": c["pattern"],
            "pattern_description": c["pattern_description"], "exposure_usd": c["exposure_usd"],
            "summary": c["summary"], "stop_reason": answer["stop_reason"], "sar_filed": answer["sar"]["file"],
            "final_actions": "|".join(final), "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        edges = [("ABOUT_CARD", "Card", [card_id])] if card_id else []
        edges += [("CITES", "Txn", c["affected_txn_ids"]), ("LINKS_CARD", "Card", c["connected_card_ids"]),
                  ("IMPLICATES", "DeviceProfile", [device_id(p) for p in c["connected_device_profiles"]]),
                  ("DREW_ON", "ClosedCase", c["similar_prior_cases"])]
        for etype, ttype, targets in edges:
            if targets:
                self.conn.upsertEdges("FraudCase", etype, ttype, [(gid, t, {}) for t in targets])
        self._log("write_case", graph_case_id=gid)
        return gid


def check() -> None:
    s = TigerGraphStore()
    for v in ("Customer", "Card", "Holder", "Txn", "DeviceProfile", "EmailDomain", "BillingRegion", "ClosedCase",
              "FraudCase"):
        print(v, s.conn.getVertexCount(v))


if __name__ == "__main__":
    {"prepare": prepare, "setup": setup, "check": check}[sys.argv[1]]()
