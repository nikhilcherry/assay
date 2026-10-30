"""Approved mapping -> GSQL. One canonical investigation, compiled per bank.

Every canonical edge becomes a tag walk: seed one end with itself, walk the
mapped bank path step by step (reverse_* where the path runs against the
bank's edge), and union the seeds into a SetAccum on the other end. Hop
count, intermediate types and direction all come from the mapping, so the
same investigation runs on a 1-hop ISO 20022 edge and a 3-hop BIAN
execution -> order -> authentication -> device path.

The investigation is mule fan-in: for every payee, which accounts paid it,
how much, into which account, from which devices / IPs. A filter the mapping
has not resolved (a '<...>' placeholder awaiting sample records) refuses to
compile: an unconfirmed discriminator would silently change the answer."""

from __future__ import annotations

import re

from assay.r2.graphmap.gsql import Schema
from assay.r2.graphmap.mapper import Mapping

# which end of each canonical edge gets tagged, for a transaction-centred query
TAG = {"PAYS": "from", "INITIATED": "to", "PAYS_TO_ACCOUNT": "from", "FROM_DEVICE": "from", "FROM_IP": "from"}


class Unresolved(ValueError):
    pass


def _literal(schema: Schema, vtype: str, attr: str, raw: str) -> str:
    a = schema.vertices[vtype].attr(attr)
    raw = raw.strip().strip("'\"")
    if a.type == "BOOL":
        return "TRUE" if raw.lower() in ("true", "1", "y", "yes") else "FALSE"
    if a.type in ("INT", "UINT", "DOUBLE", "FLOAT"):
        return str(float(raw) if a.type in ("DOUBLE", "FLOAT") else int(raw))
    return '"' + raw.replace('"', '') + '"'


def where(schema: Schema, vtype: str, flt: str | None, alias: str) -> str:
    """'IsCstmr = false' -> 's.IsCstmr == FALSE'; 'partyType in (A, B)' -> OR chain.
    Only filters on attributes `vtype` actually has; others are not this end's."""
    if not flt:
        return ""
    if "<" in flt:
        raise Unresolved(f"filter not confirmed from sample records: {flt}")
    m = re.match(r"^\s*(\w+)\s*(=|==|!=|<>)\s*(.+?)\s*$", flt)
    n = re.match(r"^\s*(\w+)\s+(?:in|IN)\s*\((.+)\)\s*$", flt)
    if n and schema.vertices[vtype].attr(n.group(1)):
        vals = [_literal(schema, vtype, n.group(1), x) for x in n.group(2).split(",")]
        return "(" + " OR ".join(f"{alias}.{n.group(1)} == {v}" for v in vals) + ")"
    if m and schema.vertices[vtype].attr(m.group(1)):
        op = "!=" if m.group(2) in ("!=", "<>") else "=="
        return f"{alias}.{m.group(1)} {op} {_literal(schema, vtype, m.group(1), m.group(3))}"
    return ""


def _back(schema: Schema, step: dict) -> str:
    """GSQL name for walking `step` the other way."""
    e = schema.edges[step["edge"]]
    if not e.directed:
        return e.name
    return (e.reverse or "reverse_" + e.name) if step["forward"] else e.name


def _arrow(schema: Schema, gsql_name: str) -> str:
    base = next((e for e in schema.edges.values() if gsql_name in (e.name, e.reverse)), None)
    return "->" if base is None or base.directed else "-"


class Compiler:
    def __init__(self, schema: Schema, mapping: Mapping):
        self.s, self.m = schema, mapping
        self.n = 0
        self.lines: list[str] = []
        self.notes: list[str] = []

    def var(self) -> str:
        self.n += 1
        return f"S{self.n}"

    def tag_walk(self, el: str):
        """Fill @<el> on the TAG end of every mapped path of canonical edge `el`."""
        ch = self.m.choices[el]
        if not ch.sources or not any(s.get("path") for s in ch.sources):
            self.notes.append(f"{el}: not mapped as a walkable path in this bank ({ch.status}); evidence unavailable")
            return False
        for src in ch.sources:
            path = src.get("path")
            if not path:
                continue
            if TAG[el] == "from":   # seed the far end, walk back to the start
                steps = [(st["to"], _back(self.s, st), st["from"]) for st in reversed(path)]
            else:
                steps = [(st["from"], st["gsql"], st["to"]) for st in path]
            seed_t = steps[0][0]
            flt = src.get("filter")
            v = self.var()
            w = where(self.s, seed_t, flt, "s")
            self.lines.append(f"  {v} = {{{seed_t}.*}};")
            self.lines.append(f"  {v} = SELECT s FROM {v}:s{' WHERE ' + w if w else ''} ACCUM s.@{el} += s;")
            for i, (a, g, b) in enumerate(steps):
                nv = self.var()
                last = i == len(steps) - 1
                wt = where(self.s, b, flt, "t") if last else ""
                self.lines.append(f"  {nv} = SELECT t FROM {v}:s -({g}:e){_arrow(self.s, g)} {b}:t"
                                  f"{' WHERE ' + wt if wt else ''} ACCUM t.@{el} += s.@{el};")
                v = nv
        return True

    def fanin(self) -> str:
        g = self.s.graph
        have = {el: self.tag_walk(el) for el in ("PAYS", "INITIATED", "PAYS_TO_ACCOUNT", "FROM_DEVICE", "FROM_IP")}
        if not have["PAYS"] or not have["INITIATED"]:
            raise Unresolved("fan-in needs PAYS and INITIATED mapped as paths")
        amount = {s["vertex"]: s["attr"] for s in self.m.choices["Transaction.amount"].sources
                  if s.get("attr") and not s.get("via")}
        body = []
        for s in self.m.choices["Transaction"].sources:
            t = s["vertex"]
            if s.get("attr") or s.get("status") == "approximate":
                continue
            amt = f"t.{amount[t]}" if t in amount else "0.0"
            if t not in amount:
                self.notes.append(f"{t}: no amount mapped; counted with amount 0")
            v = self.var()
            body.append(f"  {v} = {{{t}.*}};")
            acc = [f"@@payers += (b -> t.@INITIATED)", f"@@amount += (b -> {amt})", "@@n += (b -> 1)"]
            if have["PAYS_TO_ACCOUNT"]:
                acc.append("@@into += (b -> t.@PAYS_TO_ACCOUNT)")
            if have["FROM_DEVICE"]:
                acc.append("@@devices += (b -> t.@FROM_DEVICE)")
            if have["FROM_IP"]:
                acc.append("@@ips += (b -> t.@FROM_IP)")
            body.append(f"  {v} = SELECT t FROM {v}:t WHERE t.@PAYS.size() > 0 ACCUM FOREACH b IN t.@PAYS DO "
                        + ", ".join(acc) + " END;")
        head = [f"CREATE QUERY canon_fanin() FOR GRAPH {g} {{",
                "  // generated by assay.r2.graphmap.compile from the approved mapping; do not edit",
                "  SetAccum<VERTEX> @PAYS, @INITIATED, @PAYS_TO_ACCOUNT, @FROM_DEVICE, @FROM_IP;",
                "  MapAccum<VERTEX, SetAccum<VERTEX>> @@payers, @@into, @@devices, @@ips;",
                "  MapAccum<VERTEX, SumAccum<DOUBLE>> @@amount;",
                "  MapAccum<VERTEX, SumAccum<INT>> @@n;"]
        tail = ["  PRINT @@payers, @@amount, @@n, @@into, @@devices, @@ips;", "}"]
        return "\n".join(head + self.lines + body + tail) + "\n"


def compile_fanin(schema: Schema, mapping: Mapping) -> tuple[str, list[str]]:
    c = Compiler(schema, mapping)
    q = c.fanin()
    return q, c.notes
