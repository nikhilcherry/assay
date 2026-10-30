"""A small synthetic canonical world, written into a bank's *native* schema.

SYNTHETIC: no customer data. 12 customers, 14 accounts, legitimate payees,
and two planted shapes an investigator must tell apart:
  * B07 -- a mule: five accounts pay it ~$490 each within 50 minutes, into one
    account (A14), three of them from the same device / IP (D13 / I13);
  * B01 -- a utility: four accounts pay it too, each from their own device.
Fan-in alone cannot separate them; shared device evidence can, and a bank
whose schema has no device cannot (the answer must say so).

`lower()` writes canonical facts through a mapping (the *writer* mapping,
standing in for how the bank stores data); intermediate records on multi-hop
paths (BIAN orders, authentication assessments, relationship plans) are
created once per anchor and shared, as a bank would."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from assay.r2.graphmap.gsql import Schema
from assay.r2.graphmap.mapper import Mapping


@dataclass
class World:
    v: dict[str, list[str]] = field(default_factory=dict)            # canonical type -> ids
    e: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # canonical edge -> (from, to)
    attrs: dict[str, dict[str, object]] = field(default_factory=dict)  # id -> canonical attr -> value


def build(seed: int = 7) -> World:
    r = random.Random(seed)
    w = World()
    w.v["Person"] = [f"P{i:02d}" for i in range(1, 13)]
    w.v["Account"] = [f"A{i:02d}" for i in range(1, 15)]
    # B01 utility, B07 mule, B10..B22 one personal payee per account (rent, family, ...)
    w.v["Beneficiary"] = ["B01", "B07"] + [f"B{i}" for i in range(10, 23)]
    w.v["Device"] = [f"D{i:02d}" for i in range(1, 14)]
    w.v["IPAddress"] = [f"I{i:02d}" for i in range(1, 14)]
    owner = {f"A{i:02d}": f"P{min(i, 12):02d}" for i in range(1, 15)}
    w.e["OWNS"] = [(p, a) for a, p in owner.items()]
    dev = {p: f"D{int(p[1:]):02d}" for p in w.v["Person"]}
    ben_acct = {"B01": "A13", "B07": "A14"} | {f"B{i}": f"A{(i - 9) % 13 + 1:02d}" for i in range(10, 23)}
    w.v["Transaction"] = []
    w.e.update({k: [] for k in ("INITIATED", "PAYS", "PAYS_TO_ACCOUNT", "FROM_DEVICE", "FROM_IP")})
    t0 = datetime(2017, 1, 15, 9, 0)

    def txn(acct, ben, amount, when, device):
        tid = f"T{len(w.v['Transaction']) + 1:03d}"
        w.v["Transaction"].append(tid)
        w.e["INITIATED"].append((acct, tid))
        w.e["PAYS"].append((tid, ben))
        w.e["PAYS_TO_ACCOUNT"].append((tid, ben_acct[ben]))
        w.e["FROM_DEVICE"].append((tid, device))
        w.e["FROM_IP"].append((tid, "I" + device[1:]))
        w.attrs[tid] = {"Transaction.amount": round(amount, 2), "Transaction.timestamp": when}

    accts = [a for a in w.v["Account"] if a != "A14"]
    for i, a in enumerate(accts):                       # ordinary payments to each account's own payee
        for _ in range(r.randint(1, 3)):
            txn(a, f"B{10 + i}", r.uniform(12, 300), t0 + timedelta(hours=r.uniform(0, 300)), dev[owner[a]])
    for a in ("A01", "A03", "A04", "A06"):               # the utility: fan-in, own devices
        txn(a, "B01", r.uniform(40, 90), t0 + timedelta(hours=r.uniform(0, 300)), dev[owner[a]])
    for i, a in enumerate(("A02", "A05", "A07", "A09", "A11")):   # the mule ring
        d = "D13" if i < 3 else dev[owner[a]]
        txn(a, "B07", 480 + r.uniform(0, 19), t0 + timedelta(days=10, minutes=10 * i), d)
    for p in w.v["Person"]:
        w.attrs[p] = {}
    return w


PLANTED = {"mule": "B07", "decoy": "B01", "mule_payers": {"A02", "A05", "A07", "A09", "A11"},
           "mule_into": "A14", "shared_device": "D13"}


# ------------------------------------------------------------ lowering
class Lowerer:
    def __init__(self, schema: Schema, writer: Mapping):
        self.s, self.m = schema, writer
        self.vertices: dict[str, dict[str, dict]] = {}     # type -> native id -> attrs
        self.edges: list[tuple[str, str, str, str, str]] = []  # (from type, from id, edge, to type, to id)
        self.native: dict[str, tuple[str, object]] = {}     # canonical id -> (type, native id)
        self.canon: dict[tuple[str, str], str] = {}         # (type, str(native id)) -> canonical id
        self.memo: dict[tuple[str, str], object] = {}

    def type_for(self, el: str) -> tuple[str, str | None]:
        for s in self.m.choices[el].sources:
            if s.get("vertex") and not s.get("attr") and s.get("status") != "approximate":
                return s["vertex"], s.get("filter")
        raise ValueError(f"writer mapping has no vertex for {el}")

    def nid(self, vtype: str, key: str):
        pid = next(a for a in self.s.vertices[vtype].attrs if a.primary)
        if pid.type in ("INT", "UINT"):
            return 100000 + (sum(ord(c) * 31 ** i for i, c in enumerate(key)) % 899999)
        return key

    def add(self, vtype: str, key: str, attrs: dict | None = None, canonical: str | None = None):
        n = self.nid(vtype, key)
        self.vertices.setdefault(vtype, {}).setdefault(n, {}).update(attrs or {})
        if canonical:
            self.native[canonical] = (vtype, n)
            self.canon[(vtype, str(n))] = canonical
        return n

    def _set_filter(self, vtype: str, flt: str | None, attrs: dict, role: str):
        if not flt:
            return
        name = flt.split()[0]
        a = self.s.vertices[vtype].attr(name)
        if a is None:
            return
        if a.type == "BOOL":
            attrs[name] = "true" in flt.lower() and "!=" not in flt
        elif "<" in flt or " in " in flt.lower():   # bank-side truth for the synthetic world
            attrs[name] = {"Person": "Individual", "Beneficiary": "Counterparty"}[role]
        else:
            attrs[name] = flt.split("=", 1)[1].strip().strip("'\"")

    def _value(self, vtype: str, attr: str, value):
        a = self.s.vertices[vtype].attr(attr)
        if isinstance(value, datetime):
            if a.type == "DATETIME":
                return value.strftime("%Y-%m-%d %H:%M:%S")
            return value.strftime("%y%m%d%H%M")      # core-banking audit stamp convention
        return value

    def lower(self, w: World):
        for el in ("Person", "Account", "Beneficiary", "Transaction", "Device", "IPAddress"):
            try:
                vtype, flt = self.type_for(el)
            except ValueError:
                continue
            for cid in w.v.get(el, []):
                attrs: dict = {}
                self._set_filter(vtype, flt, attrs, el)
                for k, val in w.attrs.get(cid, {}).items():
                    for s in self.m.choices[k].sources:
                        if s.get("vertex") == vtype and s.get("attr") and not s.get("via"):
                            attrs[s["attr"]] = self._value(vtype, s["attr"], val)
                            break
                self.add(vtype, cid, attrs, canonical=cid)
        for el, pairs in w.e.items():
            ch = self.m.choices[el]
            for a, b in pairs:
                if a not in self.native or b not in self.native:
                    continue
                ta, tb = self.native[a][0], self.native[b][0]
                src = next((s for s in ch.sources if s.get("path") and s["path"][0]["from"] == ta
                            and s["path"][-1]["to"] == tb), None)
                if src is None:
                    continue
                self._walk(src, a, b)

    def _walk(self, src: dict, a: str, b: str):
        path = src["path"]
        anchor = a if self.native[a][0] in self._txn_types() else b if self.native[b][0] in self._txn_types() else a
        prev_t, prev_id = self.native[a]
        for i, st in enumerate(path):
            if i == len(path) - 1:
                nt, nid = self.native[b]
            else:
                nt = st["to"]
                key = (anchor, nt)
                if key not in self.memo:
                    self.memo[key] = self.add(nt, f"{nt[:6]}_{anchor}")
                nid = self.memo[key]
            if st["forward"]:
                self.edges.append((prev_t, prev_id, st["edge"], nt, nid))
            else:
                self.edges.append((nt, nid, st["edge"], prev_t, prev_id))
            prev_t, prev_id = nt, nid

    def _txn_types(self) -> set[str]:
        return {s["vertex"] for s in self.m.choices["Transaction"].sources if s.get("vertex")}
