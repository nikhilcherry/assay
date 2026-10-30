"""LLM proposer for graph-schema mapping, held to the schema by code.

The model sees the bank schema with every comment stripped (never the
organiser's ground truth), the canonical model, and an output contract. What
comes back is a proposal, not an answer: `validate()` resolves every vertex,
edge and attribute name against the parsed schema, works out each step's
direction and its GSQL name (reverse_* when walked backwards) itself, and
drops anything that does not exist or does not connect. A dropped element is
reported, never silently repaired.

Egress: the schema text only (type, edge and attribute names). No rows."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

from assay.r2.graphmap import canonical
from assay.r2.graphmap.gsql import Schema, strip_comments
from assay.r2.graphmap.mapper import Choice, Mapping

MODEL = os.environ.get("ASSAY_GRAPHMAP_MODEL", "gemini-3.5-flash")
PROMPT_VERSION = "v1"   # v2 ("be complete") measured no better on the dev schemas (146 vs 147/175); kept v1
KEY_FILE = Path.home() / ".config/assay/gemini.env"

SYSTEM = """You map a bank's TigerGraph schema onto a canonical fraud-investigation model.
You get the bank's GSQL (comments removed) and must answer for EVERY canonical element.

Canonical vertices: """ + ", ".join(canonical.VERTICES) + """
Canonical edges (canonical direction): """ + "; ".join(f"{k} ({a}->{b})" for k, (a, b) in canonical.EDGES.items()) + """
Canonical attributes: """ + ", ".join(canonical.ATTRS) + """
Derived: account_under_investigation (how to tell an account is under investigation).

Rules a careful banking data engineer follows:
- Transaction = the actual movement of money (execution / posting header / authorisation), not the order or
  consent that requests it, not a batch or message header, not a return, not a ledger/GL/statement posting
  that merely records it, not a balance snapshot, not a standing-order template.
- A role or subtype is a filter on a shared type (e.g. parties where type = counterparty).
- A concept that only exists as an attribute maps to that attribute (vertex + attr) with status approximate.
- Edges are paths of bank edges; walk an edge backwards when the bank points it the other way.
- status: exact = same meaning, directly; approximate = needs a filter, a multi-hop path through another
  entity, a code/format conversion, or is a near-meaning; unmapped = nothing in the schema carries it.
- Use only names that appear in the schema. Prefer the fewest, most specific sources.

Answer with ONE JSON object, no prose, keyed by canonical element:
{"Person": {"status": "exact", "sources": [{"vertex": "X", "filter": "attr = value or null"}], "why": "..."},
 "Device": {"status": "approximate", "sources": [{"vertex": "LoginLog", "attr": "TERMINAL_ID"}], "why": "..."},
 "OWNS": {"status": "exact", "sources": [{"steps": [{"edge": "E1", "from": "Person-side type", "to": "next type"}, ...], "filter": null}], "why": "..."},
 "Transaction.amount": {"status": "exact", "sources": [{"vertex": "T", "attr": "AMT", "via": [steps or empty], "transform": "e.g. divide by 100 or null"}], "why": "..."},
 "account_under_investigation": {"status": "...", "sources": [{"vertex": "...", "attr": "...", "filter": "..."} or {"steps": [...], "filter": "..."}], "why": "..."},
 ... every element ...}
Steps list bank vertex types in the order walked, from the canonical source end to the canonical target end."""


def _key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("no GEMINI_API_KEY")


def ask(schema_text: str, model: str = MODEL, timeout: float = 300) -> dict:
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": strip_comments(schema_text)}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32768, "responseMimeType": "application/json"}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_key()}"
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            parts = d["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return json.loads(re.sub(r"^```\w*\n|\n```$", "", text.strip()))
        except Exception as e:   # transient venue Wi-Fi / 5xx / truncated JSON
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"LLM mapping failed: {last}")


# ------------------------------------------------------------ validation
def _resolve_steps(schema: Schema, steps: list[dict]) -> tuple[list[dict], str]:
    out = []
    for i, st in enumerate(steps or []):
        name = str(st.get("edge", ""))
        name = name[len("reverse_"):] if name.startswith("reverse_") and name not in schema.edges else name
        e = schema.edges.get(name) or next((x for x in schema.edges.values() if x.reverse == name), None)
        if e is None:
            return [], f"edge {name!r} not in schema"
        a, b = st.get("from"), st.get("to")
        prev = out[-1]["to"] if out else None
        if prev and a != prev:
            a = prev
        fwd = [(x, y) for x, y in e.pairs if x == a and (b is None or y == b)]
        bwd = [(x, y) for x, y in e.pairs if y == a and (b is None or x == b)]
        if fwd:
            x, y = fwd[0]
            out.append({"edge": e.name, "forward": True, "gsql": e.name, "from": x, "to": y})
        elif bwd:
            x, y = bwd[0]
            g = e.name if not e.directed else (e.reverse or "reverse_" + e.name)
            out.append({"edge": e.name, "forward": False, "gsql": g, "from": y, "to": x})
        else:
            return [], f"step {i}: {e.name} does not join {a} and {b}"
    return out, ""


def validate(schema: Schema, raw: dict) -> tuple[Mapping, list[str]]:
    problems, choices = [], {}
    for el in canonical.ELEMENTS:
        kind = "vertex" if el in canonical.VERTICES else "edge" if el in canonical.EDGES else "attr" if "." in el else "derived"
        r = raw.get(el) or {}
        ch = Choice(el, kind, rule="LLM proposal, schema-checked", why=[str(r.get("why", ""))[:400]])
        status = r.get("status", "unmapped")
        for s in (r.get("sources") or []) if status != "unmapped" else []:
            if not isinstance(s, dict):
                continue
            src: dict = {}
            if s.get("steps"):
                path, err = _resolve_steps(schema, s["steps"])
                if err:
                    problems.append(f"{el}: {err}")
                    continue
                src["path"] = path
            v = s.get("vertex")
            if v:
                if v not in schema.vertices:
                    problems.append(f"{el}: vertex {v!r} not in schema")
                    continue
                src["vertex"] = v
                a = s.get("attr")
                if a:
                    if not schema.vertices[v].attr(a):
                        problems.append(f"{el}: {v}.{a} not in schema")
                        continue
                    src["attr"] = a
            if s.get("via"):
                via, err = _resolve_steps(schema, s["via"])
                if err:
                    problems.append(f"{el} via: {err}")
                else:
                    src["via"] = via
            for k in ("filter", "transform"):
                if s.get(k):
                    src[k] = str(s[k])
            if kind == "edge" and "path" not in src and "attr" not in src:
                problems.append(f"{el}: edge source without steps")
                continue
            if src:
                src["status"] = status
                ch.sources.append(src)
        ch.status = status if ch.sources else "unmapped"
        if status != "unmapped" and not ch.sources:
            ch.why.append("every proposed source failed the schema check")
        choices[el] = ch
    return Mapping(schema.graph, choices), problems


def map_schema_llm(schema: Schema, cache: Path | None = None, model: str = MODEL) -> tuple[Mapping, list[str]]:
    raw = None
    if cache and cache.exists():
        raw = json.loads(cache.read_text())
    if raw is None:
        raw = ask(Path(schema.source).read_text(), model=model)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, indent=1))
    return validate(schema, raw)


REVIEW = """You are reviewing a schema mapping before a human approves it. You get the bank schema, a first
mapping proposal (JSON), and a structural second opinion from a rule-based mapper (its sources, the rule that
fired, why, and the look-alike candidates it rejected). The rule mapper reads names poorly (opaque codes defeat
it) but is good at structure: it finds every type that carries an amount and a time, every account product a
payment touches, every owner edge, and it rejects batches, returns, ledger postings and order stages.
For each element decide: keep the first proposal, take the rule mapper's, or combine them. Combine when the
rule mapper found an additional genuine source of the same concept that the first proposal missed. Ignore
rule-mapper sources that only resemble the concept. Output the final mapping in exactly the same JSON contract
as the first proposal, every element present."""


def _compact(m: Mapping) -> dict:
    out = {}
    for el, ch in m.choices.items():
        srcs = []
        for s in ch.sources:
            d = {k: s[k] for k in ("vertex", "attr", "filter") if s.get(k)}
            if s.get("path"):
                d["steps"] = [{"edge": st["edge"], "from": st["from"], "to": st["to"]} for st in s["path"]]
            if s.get("via"):
                d["via"] = [{"edge": st["edge"], "from": st["from"], "to": st["to"]} for st in s["via"]]
            srcs.append(d)
        out[el] = {"status": ch.status, "rule": ch.rule, "sources": srcs, "why": ch.why[:3],
                   "rejected": ch.rejected[:3]}
    return out


def review(schema_text: str, first: dict, rules: Mapping, model: str = MODEL, timeout: float = 300) -> dict:
    user = (strip_comments(schema_text) + "\n\n=== FIRST PROPOSAL ===\n" + json.dumps(first)
            + "\n\n=== RULE MAPPER ===\n" + json.dumps(_compact(rules)))
    body = {"systemInstruction": {"parts": [{"text": SYSTEM + "\n\n" + REVIEW}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32768, "responseMimeType": "application/json"}}
    return _post_json(model, body, timeout)


def _post_json(model: str, body: dict, timeout: float) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_key()}"
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            parts = d["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return json.loads(re.sub(r"^```\w*\n|\n```$", "", text.strip()))
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def map_schema_reviewed(schema: Schema, rules: Mapping, cache_dir: Path | None = None,
                        model: str = MODEL) -> tuple[Mapping, list[str]]:
    """Proposal -> review against the rule mapper's structural evidence -> schema check."""
    stem = Path(schema.source).stem
    first_c = cache_dir / f"{stem}.json" if cache_dir else None
    rev_c = cache_dir / f"{stem}.review.json" if cache_dir else None
    text = Path(schema.source).read_text()
    first = json.loads(first_c.read_text()) if first_c and first_c.exists() else ask(text, model=model)
    if first_c and not first_c.exists():
        first_c.parent.mkdir(parents=True, exist_ok=True)
        first_c.write_text(json.dumps(first, indent=1))
    if rev_c and rev_c.exists():
        final = json.loads(rev_c.read_text())
    else:
        final = review(text, first, rules, model=model)
        if rev_c:
            rev_c.write_text(json.dumps(final, indent=1))
    return validate(schema, final)
