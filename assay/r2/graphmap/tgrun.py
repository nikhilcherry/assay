"""Run the canonical fan-in investigation on a bank graph inside TigerGraph.

  python -m assay.r2.graphmap.tgrun data/round2/graphmap_handout/bank2_bian_service_domains.gsql

1. CREATE the bank's graph from its own GSQL file, untouched.
2. Write the synthetic world into it through the *writer* mapping
   (`--writer rules`, the mapping that matches the organiser's ground truth on
   every graded element -- a stand-in for how the bank stores data).
3. Read sample records back from TigerGraph and re-map with them, so pending
   choices (a discriminator's values) are settled by records, not guesses.
4. Compile canon_fanin from the *approved* mapping, install, run, and turn
   the bank's own ids back into canonical ones.
Only graphs named Bank_* are ever dropped or created."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from assay.r2.graphmap.compile import compile_fanin
from assay.r2.graphmap.gsql import parse
from assay.r2.graphmap.mapper import Mapping, map_schema
from assay.r2.graphmap.world import PLANTED, Lowerer, build

CONTAINER = "assay-tg"
HOST = os.environ.get("ASSAY_TG_HOST", "http://127.0.0.1")


def gsql(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".gsql", delete=False) as f:
        f.write(text)
    os.chmod(f.name, 0o644)   # the container's tigergraph user must read it
    dst = f"/tmp/{Path(f.name).name}"
    subprocess.run(["docker", "--context", "default", "cp", f.name, f"{CONTAINER}:{dst}"], check=True)
    r = subprocess.run(["docker", "--context", "default", "exec", "-u", "tigergraph", CONTAINER, "bash", "-c",
                        f"/home/tigergraph/tigergraph/app/cmd/gsql {dst}"], capture_output=True, text=True)
    os.unlink(f.name)
    return r.stdout + r.stderr


def conn(graph: str):
    import pyTigerGraph as tg
    c = tg.TigerGraphConnection(host=HOST, graphname=graph, username=os.environ.get("ASSAY_TG_USER", "tigergraph"),
                                password=os.environ.get("ASSAY_TG_PASSWORD", "tigergraph"),
                                restppPort="14240", gsPort="14240")
    try:
        c.getToken(c.createSecret())
    except Exception:
        pass
    return c


# GSQL keywords that a bank may use as an attribute name; TigerGraph 4.2 rejects them
# (the handout's T24 CUSTOMER.TARGET fails with a syntax error as shipped)
RESERVED = ("TARGET", "SOURCE")


def load_compat(text: str) -> tuple[str, list[str]]:
    import re
    renamed = []
    for w in RESERVED:
        pat = re.compile(rf"(?<=[(,])(\s*){w}(\s+(?:INT|UINT|STRING|DOUBLE|FLOAT|BOOL|DATETIME)\b)")
        if pat.search(text):
            text = pat.sub(rf"\g<1>{w}_\g<2>", text)
            renamed.append(f"attribute {w} -> {w}_ (GSQL reserved word; not used by any mapping)")
    return text, renamed


def create(path: Path, graph: str) -> tuple[str, list[str]]:
    assert graph.startswith("Bank_"), graph
    out = gsql(f"DROP GRAPH {graph} CASCADE\n")
    text, renamed = load_compat(path.read_text())
    return out + gsql(text), renamed


def load(c, low: Lowerer) -> dict:
    n = {"vertices": 0, "edges": 0}
    for vtype, rows in low.vertices.items():
        n["vertices"] += c.upsertVertices(vtype, [(k, v) for k, v in rows.items()])
    by: dict[tuple, list] = {}
    for ft, fid, e, tt, tid in low.edges:
        by.setdefault((ft, e, tt), []).append((fid, tid))
    for (ft, e, tt), pairs in by.items():
        n["edges"] += c.upsertEdges(ft, e, tt, pairs)
    return n


def samples(c, schema, k: int = 25) -> dict[str, list[dict]]:
    out = {}
    for v in schema.vertices:
        try:
            out[v] = [r["attributes"] for r in c.getVertices(v, limit=k)]
        except Exception:
            out[v] = []
    return out


def interpret(res: list[dict], low: Lowerer, notes: list[str] = ()) -> dict:
    r = {}
    for part in res:
        r.update(part)
    canon = {str(n): cid for (_, n), cid in low.canon.items()}
    c = lambda x: canon.get(str(x), str(x))
    out = {}
    for ben, payers in r.get("@@payers", {}).items():
        payers = sorted(c(p) for p in payers)
        # an edge that did not compile is "no evidence", not "no devices"
        no_dev = any(n.startswith("FROM_DEVICE:") for n in notes)
        no_ip = any(n.startswith("FROM_IP:") for n in notes)
        devs = None if no_dev else sorted(c(d) for d in r.get("@@devices", {}).get(ben, []))
        ips = None if no_ip else sorted(c(d) for d in r.get("@@ips", {}).get(ben, []))
        out[c(ben)] = {"payers": payers, "n_payments": r["@@n"][ben], "amount": round(r["@@amount"][ben], 2),
                       "into": sorted(c(a) for a in r.get("@@into", {}).get(ben, [])),
                       "devices": devs, "ips": ips}
    return out


def verdict(row: dict) -> str:
    if len(row["payers"]) < 3:
        return "no fan-in"
    if row["devices"] is None and row["ips"] is None:
        return "fan-in; UNCERTAIN -- this bank's schema has no device or IP on payments, so a mule ring and a popular payee look the same"
    shared = row["devices"] is not None and len(row["devices"]) < row["n_payments"]
    return "fan-in + shared device: MULE RING" if shared else "fan-in, every payer on their own device: popular payee"


def run(path: Path, approved: Mapping | None = None, writer: Mapping | None = None, log=print) -> dict:
    schema = parse(path)
    writer = writer or map_schema(schema)
    log(f"== {schema.graph}: create graph from {path.name}")
    out, renamed = create(path, schema.graph)
    for n in renamed:
        log("   load-compat: " + n)
    if "could not be created" in out or "Syntax error" in out:
        raise RuntimeError(out[-2000:])
    low = Lowerer(schema, writer)
    low.lower(build())
    c = conn(schema.graph)
    log(f"   loaded {load(c, low)} (synthetic world through the writer mapping)")
    smp = samples(c, schema)
    approved = approved or map_schema(schema, samples=smp)
    q, notes = compile_fanin(schema, approved)
    for n in notes:
        log("   note: " + n)
    inst = gsql(f"USE GRAPH {schema.graph}\nDROP QUERY canon_fanin\n{q}\nINSTALL QUERY canon_fanin\n")
    if "Query installation finished" not in inst and "installed successfully" not in inst.lower():
        raise RuntimeError("install failed:\n" + inst[-3000:])
    res = c.runInstalledQuery("canon_fanin", timeout=120000)
    rows = interpret(res, low, notes)
    for ben in sorted(rows, key=lambda b: -len(rows[b]["payers"])):
        if len(rows[ben]["payers"]) >= 3:
            log(f"   {ben}: {len(rows[ben]['payers'])} payers {rows[ben]['payers']} ${rows[ben]['amount']} into "
                f"{rows[ben]['into']} devices {rows[ben]['devices']} -> {verdict(rows[ben])}")
    return {"graph": schema.graph, "query": q, "notes": notes, "load_compat": renamed, "rows": rows,
            "verdicts": {b: verdict(r) for b, r in rows.items()},
            "planted_ok": rows.get(PLANTED["mule"], {}).get("payers") and
                          set(rows[PLANTED["mule"]]["payers"]) == PLANTED["mule_payers"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schemas", nargs="+")
    ap.add_argument("--proposal", nargs="*", default=[], help="approved proposal id per schema, same order")
    ap.add_argument("--dry-run-unapproved", action="store_true",
                    help="run the proposal's mapping without approval; output is labelled DRY RUN")
    ap.add_argument("--out", default="data/round2/graphmap_tg")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    from assay.r2.graphmap import proposal as P
    for i, p in enumerate(a.schemas):
        approved, label = None, "rules (no proposal)"
        if i < len(a.proposal):
            pid = a.proposal[i]
            if a.dry_run_unapproved:
                doc = json.loads((P.DIR / f"{pid}.json").read_text())
                from assay.r2.graphmap.mapper import Choice
                approved = Mapping(doc["graph"], {e: Choice(**v["proposed"]) for e, v in doc["elements"].items()})
                label = f"DRY RUN: proposal {pid}, NOT approved"
            else:
                approved, label = P.approved_mapping(pid), f"approved proposal {pid}"
        print(f"## {Path(p).name}: investigation mapping = {label}")
        r = run(Path(p), approved=approved)
        r["mapping"] = label
        Path(a.out, Path(p).stem + ".json").write_text(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
