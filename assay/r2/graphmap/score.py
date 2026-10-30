"""Grade a Mapping against the organiser's '# Expected mapping' comments.

Two numbers per element:
  * right  -- we named the same schema objects (or both said unmapped);
  * status -- we also agree on exact / approximate / unmapped.
Elements the ground truth does not mention are not graded."""

from __future__ import annotations

import re

from assay.r2.graphmap.gsql import Schema
from assay.r2.graphmap.mapper import Mapping


def _ids(rest: str, schema: Schema) -> dict[str, set[str]]:
    """Schema identifiers named in a ground-truth line, minus negations and
    'where/when' conditions (a filter, not part of the mapped object)."""
    rest = re.sub(r"\(not [^)]*\)", "", rest)
    names_v, names_e = set(schema.vertices), set(schema.edges)
    names_a = {a.name for v in schema.vertices.values() for a in v.attrs}
    out = {"v": set(), "e": set(), "a": set()}
    for alt in rest.split(";"):
        core = re.split(r"\b(?:where|when)\b", alt)[0]
        for tok in re.findall(r"[A-Za-z_][\w.]*", core):
            for part in tok.split("."):
                base = part[len("reverse_"):] if part.startswith("reverse_") else part
                if base in names_e:
                    out["e"].add(base)
                elif part in names_v:
                    out["v"].add(part)
                elif part not in names_v:
                    # ground truth may shorten a flattened name: BICFI for FinInstnId_BICFI
                    out["a"].update(n for n in names_a if n == part or n.endswith("_" + part))
    return out


def grade(m: Mapping, gt: dict[str, dict], schema: Schema) -> dict:
    rows = []
    for el, truth in gt.items():
        ch = m.choices.get(el)
        if ch is None:
            continue
        ids = _ids(truth["rest"], schema)
        ours_v = {s["vertex"] for s in ch.sources if s.get("vertex") and not s.get("attr")}
        ours_e = {st["edge"] for s in ch.sources for st in s.get("path", [])}
        ours_a = {s["attr"] for s in ch.sources if s.get("attr")}
        path_v = {st[k] for s in ch.sources for st in s.get("path", []) for k in ("from", "to")}
        if truth["status"] == "unmapped" or ch.status == "unmapped":
            right = truth["status"] == ch.status
        elif ch.kind == "vertex":
            # an attribute-level mapping (Device = PROTOCOL.TERMINAL_ID) compares attributes
            right = (ours_a == ids["a"] and bool(ours_a)) if ours_a else ours_v == ids["v"]
        elif ch.kind == "edge" and ours_a and not ours_e:   # attribute-level link ("no edge")
            right = ours_a <= ids["a"]
        elif ch.kind == "edge":
            arrow = "->" in truth["rest"]
            e_ok = ids["e"] <= ours_e and (arrow or ours_e <= ids["e"])
            right = e_ok and ids["v"] <= path_v | ours_v and bool(ours_e)
        elif ch.kind == "attr":
            right = bool(ours_a) and ours_a <= ids["a"]
        else:
            named = ours_a | ours_e
            right = bool(named) and named <= ids["a"] | ids["e"]
        rows.append({"element": el, "right": right, "status_match": right and truth["status"] == ch.status,
                     "truth": truth["status"] + " " + truth["rest"], "ours": ch.status + " " + ", ".join(sorted(ch.names()))})
    n = len(rows)
    return {"graph": m.graph, "graded": n, "right": sum(r["right"] for r in rows),
            "status_match": sum(r["status_match"] for r in rows), "rows": rows}
