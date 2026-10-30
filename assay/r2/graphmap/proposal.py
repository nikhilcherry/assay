"""Mapping proposal -> human approval -> the mapping the investigation runs on.

  python -m assay.r2.graphmap.proposal propose data/round2/graphmap_handout/bank2_bian_service_domains.gsql
  python -m assay.r2.graphmap.proposal approve <proposal_id> --role L2_manager [--take-rule OWNS,Account]

The proposal is the LLM mapping after review against the rule mapper's
structural evidence, schema-checked. Each element says whether the two
mappers agree (measured: agreement was right 140/145 times across the
handout + first blind set) and what still needs sample records. Disagreements
are what a human reads first. Approval records who, when, with which role,
and every element they changed; an unapproved proposal cannot be used."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from assay.r2.graphmap import canonical
from assay.r2.graphmap.gsql import parse
from assay.r2.graphmap.mapper import Choice, Mapping, map_schema

DIR = Path("data/round2/graphmap_proposals")
ROLES = {"L1_lead", "L2_manager", "team_lead", "fraud_manager"}   # L1 analysts may not approve a re-map


def _same(a: Choice, b: Choice) -> bool:
    return (a.status == "unmapped") == (b.status == "unmapped") and a.names() == b.names()


def propose(path: Path, use_llm: bool = True, samples: dict | None = None) -> dict:
    schema = parse(path)
    rules = map_schema(schema, samples=samples)
    final, problems, source = rules, [], "rules"
    if use_llm:
        from assay.r2.graphmap.llm import map_schema_reviewed
        final, problems = map_schema_reviewed(schema, rules, cache_dir=Path("data/round2/graphmap_llm/gemini-3.5-flash_v1"))
        source = "llm+review"
    pid = hashlib.sha256((path.read_text() + json.dumps(final.to_json(), sort_keys=True)).encode()).hexdigest()[:12]
    elements = {}
    for el in canonical.ELEMENTS:
        f, r = final.choices[el], rules.choices[el]
        elements[el] = {"proposed": asdict(f), "rule_mapper": asdict(r), "agree": _same(f, r),
                        "pending": sorted(set(f.pending + r.pending))}
    doc = {"proposal_id": pid, "schema": str(path), "graph": schema.graph, "source": source,
           "created_at": datetime.now(timezone.utc).isoformat(), "schema_check_drops": problems,
           "elements": elements, "approved": None}
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"{pid}.json").write_text(json.dumps(doc, indent=1))
    (DIR / f"{pid}.md").write_text(render(doc))
    return doc


def approve(pid: str, role: str, approver: str, take_rule: list[str] = (), note: str = "") -> dict:
    if role not in ROLES:
        raise PermissionError(f"role {role!r} cannot approve a schema re-map; allowed: {sorted(ROLES)}")
    p = DIR / f"{pid}.json"
    doc = json.loads(p.read_text())
    bad = [e for e in take_rule if e not in doc["elements"]]
    if bad:
        raise ValueError(f"unknown elements {bad}")
    doc["approved"] = {"by": approver, "role": role, "at": datetime.now(timezone.utc).isoformat(),
                       "took_rule_mapper_for": list(take_rule), "note": note,
                       "unresolved_disagreements": [e for e, v in doc["elements"].items()
                                                    if not v["agree"] and e not in take_rule]}
    p.write_text(json.dumps(doc, indent=1))
    (DIR / f"{pid}.md").write_text(render(doc))
    return doc


def approved_mapping(pid: str) -> Mapping:
    doc = json.loads((DIR / f"{pid}.json").read_text())
    if not doc.get("approved"):
        raise PermissionError(f"proposal {pid} is not approved; run `approve` first")
    take = set(doc["approved"]["took_rule_mapper_for"])
    ch = {}
    for el, v in doc["elements"].items():
        d = v["rule_mapper"] if el in take else v["proposed"]
        ch[el] = Choice(**d)
    return Mapping(doc["graph"], ch)


def render(doc: dict) -> str:
    els = doc["elements"]
    dis = [e for e, v in els.items() if not v["agree"]]
    out = [f"# Schema mapping proposal {doc['proposal_id']} -- {doc['graph']}", "",
           f"Schema: `{doc['schema']}` · proposer: {doc['source']} · created {doc['created_at'][:19]}Z", ""]
    a = doc.get("approved")
    out += [f"**Approved** by {a['by']} ({a['role']}) at {a['at'][:19]}Z; took the rule mapper for "
            f"{', '.join(a['took_rule_mapper_for']) or 'nothing'}." if a else "**NOT APPROVED** -- the investigation will refuse to run on it.", ""]
    out += [f"{len(els) - len(dis)}/{len(els)} elements: both mappers agree. Read these {len(dis)} first: "
            + ", ".join(dis), ""]
    for e, v in els.items():
        p = v["proposed"]
        names = sorted({x for s in p["sources"] for x in ([s.get("vertex"), s.get("attr")] + [st["gsql"] for st in s.get("path", [])]) if x})
        out.append(f"## {e} -- {p['status']}{'' if v['agree'] else '  ⚠ mappers disagree'}")
        out.append(f"- proposed: {', '.join(names) or '—'}" + "".join(f"  [filter {s['filter']}]" for s in p["sources"] if s.get("filter")))
        if p["why"] and p["why"][0]:
            out.append(f"- why: {p['why'][0]}")
        if not v["agree"]:
            r = v["rule_mapper"]
            rn = sorted({x for s in r["sources"] for x in ([s.get("vertex"), s.get("attr")] + [st["gsql"] for st in s.get("path", [])]) if x})
            out.append(f"- rule mapper: {r['status']} {', '.join(rn) or '—'} ({'; '.join(r['why'][:2])})")
            for rej in r["rejected"][:2]:
                out.append(f"  - rejected {rej['candidate']}: {rej['why']}")
        for pend in v["pending"]:
            out.append(f"- needs sample records: {pend}")
        out.append("")
    if doc["schema_check_drops"]:
        out += ["## Dropped by the schema check (names the model invented)", ""] + [f"- {d}" for d in doc["schema_check_drops"]]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("schema")
    p.add_argument("--no-llm", action="store_true")
    q = sub.add_parser("approve")
    q.add_argument("proposal_id")
    q.add_argument("--role", required=True)
    q.add_argument("--by", default="analyst")
    q.add_argument("--take-rule", default="")
    q.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "propose":
        d = propose(Path(a.schema), use_llm=not a.no_llm)
        dis = [e for e, v in d["elements"].items() if not v["agree"]]
        print(f"proposal {d['proposal_id']} ({d['graph']}): {len(dis)} disagreements -> {DIR / (d['proposal_id'] + '.md')}")
    else:
        d = approve(a.proposal_id, a.role, a.by, [x for x in a.take_rule.split(",") if x], a.note)
        print(f"approved {a.proposal_id} by {a.by} ({a.role}); rule mapper taken for {d['approved']['took_rule_mapper_for']}")


if __name__ == "__main__":
    main()
