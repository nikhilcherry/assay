"""Run the agent on the case pack and write one answer file per case.

    python -m assay.run                 # all 20, TigerGraph backend if reachable
    python -m assay.run --local         # reference backend (parquet), no database
    python -m assay.run HHG-006 HHG-014 # a subset
"""

from __future__ import annotations

import argparse
import json
import sys

from assay.data import ROOT, case_pack
from assay.investigate import Investigator
from assay.patterns import PatternModel
from assay.validate import validate

OUT = ROOT / "cases"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--local", action="store_true", help="use the parquet reference store, not TigerGraph")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    if a.local:
        from assay.store import LocalStore
        store = LocalStore()
        writer = store.write_case
    else:
        from assay.tg import TigerGraphStore
        store = TigerGraphStore()
        writer = store.write_case
    inv = Investigator(store, PatternModel(), writer=writer)

    # Chronological, so case memory is causal: an investigation can only find
    # cases that were closed before its alert was opened.
    cp = case_pack().sort_values("opened_at")
    if a.cases:
        cp = cp[cp.case_id.isin(a.cases)]
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    index = []
    for _, case in cp.iterrows():
        ans = inv.run(case)
        index.append({"case_id": case.case_id, "card_id": case.card_id,
                      "graph_case_id": ans["case"]["graph_case_id"], "verdict": ans["case"]["verdict"],
                      "pattern": ans["case"]["pattern"], "connected": ans["case"]["connected_card_ids"]})
        errs = validate(ans)
        bad += bool(errs)
        (out / f"{case.case_id}.json").write_text(json.dumps(ans, indent=2) + "\n")
        c = ans["case"]
        print(f"{case.case_id} {c['verdict']:<10} p={c['fraud_probability']:.2f} {c['pattern']:<28} "
              f"exp=${c['exposure_usd']:>9,.2f} n={len(c['affected_txn_ids']):>2} conn={len(c['connected_card_ids']):>2} "
              f"sar={ans['sar']['file']!s:<5} final={[x['action'] for x in ans['next_best_actions']['final']]}"
              + (f"  INVALID: {errs}" if errs else ""))
    if not a.cases:
        (ROOT / "runs").mkdir(exist_ok=True)
        (ROOT / "runs" / "exam_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
