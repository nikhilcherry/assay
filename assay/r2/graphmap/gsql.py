"""Read a bank's GSQL schema into plain vertex / edge records.

The mapper only ever sees `parse()`: every comment is stripped first, so the
organiser's "Expected mapping (ground truth)" block cannot leak into a choice.
`ground_truth()` reads that block separately, for the scorer alone."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Attr:
    name: str
    type: str
    primary: bool = False


@dataclass
class Vertex:
    name: str
    attrs: list[Attr]

    def attr(self, name: str) -> Attr | None:
        return next((a for a in self.attrs if a.name == name), None)


@dataclass
class Edge:
    name: str
    pairs: list[tuple[str, str]]
    directed: bool
    reverse: str = ""
    attrs: list[Attr] = field(default_factory=list)


@dataclass
class Schema:
    graph: str
    vertices: dict[str, Vertex]
    edges: dict[str, Edge]
    source: str = ""

    def neighbours(self, v: str):
        """Every one-hop step out of vertex type `v`: (edge, forward?, other end).
        forward=False means the step walks the edge against its direction, which
        in GSQL is the reverse edge (or the same name when undirected)."""
        for e in self.edges.values():
            for a, b in e.pairs:
                if a == v:
                    yield e, True, b
                if b == v and (a != v or not e.directed):
                    yield e, False, a


_COMMENT = re.compile(r"/\*.*?\*/|(?:#|//)[^\n]*", re.S)
_VERTEX = re.compile(r"(?:ADD|CREATE)\s+VERTEX\s+(\w+)\s*\((.*?)\)\s*(?:WITH[^;\n]*)?;?", re.S | re.I)
_EDGE = re.compile(r"(?:ADD|CREATE)\s+(DIRECTED|UNDIRECTED)\s+EDGE\s+(\w+)\s*\((.*?)\)\s*(WITH[^;\n]*)?;?", re.S | re.I)
_GRAPH = re.compile(r"CREATE\s+GRAPH\s+(\w+)", re.I)


def strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def _attrs(body: str) -> list[Attr]:
    out = []
    for part in body.split(","):
        words = part.split()
        if not words:
            continue
        if words[0].upper() == "PRIMARY_ID" and len(words) >= 3:
            out.append(Attr(words[1], words[2].upper(), True))
        elif len(words) >= 2 and words[0].upper() not in ("FROM", "TO"):
            out.append(Attr(words[0], words[1].upper()))
    return out


def parse_text(text: str, source: str = "") -> Schema:
    body = strip_comments(text)
    g = _GRAPH.search(body)
    vertices = {m.group(1): Vertex(m.group(1), _attrs(m.group(2))) for m in _VERTEX.finditer(body)}
    edges = {}
    for m in _EDGE.finditer(body):
        kind, name, inner, with_ = m.groups()
        pairs = [(a, b) for a, b in re.findall(r"FROM\s+(\w+)\s*,\s*TO\s+(\w+)", inner, re.I)]
        # edge attributes follow the last FROM/TO pair
        tail = re.split(r"TO\s+\w+", inner, flags=re.I)[-1]
        rev = re.search(r'REVERSE_EDGE\s*=\s*"(\w+)"', with_ or "")
        edges[name] = Edge(name, pairs, kind.upper() == "DIRECTED", rev.group(1) if rev else "", _attrs(tail))
    return Schema(g.group(1) if g else Path(source).stem, vertices, edges, source)


def parse(path: str | Path) -> Schema:
    return parse_text(Path(path).read_text(), str(path))


# ------------------------------------------------------------ ground truth
_GT_LINE = re.compile(r"^#\s{2,}(?P<names>[A-Za-z_][\w.]*(?:,\s*[A-Za-z_][\w.]*)*)\s+(?P<status>exact|approximate|unmapped)\b\s*(?P<rest>.*)$")


def ground_truth(path: str | Path) -> dict[str, dict]:
    """{canonical element: {status, rest}} from the '# Expected mapping' block.
    Scorer-only; never passed to the mapper."""
    out: dict[str, dict] = {}
    on, last = False, None
    for line in Path(path).read_text().splitlines():
        if "Expected mapping" in line:
            on = True
            continue
        if not on:
            continue
        if not line.startswith("#"):
            break
        m = _GT_LINE.match(line)
        if m:
            last = [n.strip() for n in m.group("names").split(",")]
            for n in last:
                out[n] = {"status": m.group("status"), "rest": m.group("rest").strip()}
        elif last and line.strip("# ").strip():
            for n in last:  # wrapped continuation line
                out[n]["rest"] += " " + line.strip("# ").strip()
    return out
