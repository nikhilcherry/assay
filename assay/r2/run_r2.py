"""Run the agent on the final-round case pack.

    python -m assay.r2.run_r2                  # all 18 -> cases_r2/
    python -m assay.r2.run_r2 HHF-007          # one case (the drills' single-case command)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from assay.data import ROOT
from assay.r2.investigate_r2 import R2Investigator
from assay.r2.frames import case_pack
from assay.r2.store_r2 import R2LocalStore, r2_pattern_model
from assay.validate import validate

OUT = ROOT / "cases_r2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    store = R2LocalStore()
    inv = R2Investigator(store, r2_pattern_model(), writer=store.write_case)
    cp = case_pack().sort_values("opened_at")
    if a.cases:
        cp = cp[cp.case_id.isin(a.cases)]
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for _, case in cp.iterrows():
        t = time.time()
        ans = inv.run(case)
        errs = validate(ans)
        bad += bool(errs)
        (out / f"{case.case_id}.json").write_text(json.dumps(ans, indent=2) + "\n")
        c = ans["case"]
        print(f"{case.case_id} {case.trigger_type:<15} {c['verdict']:<10} p={c['fraud_probability']:.2f} {c['pattern']:<28} "
              f"exp=${c['exposure_usd']:>9,.2f} conn={len(c['connected_card_ids']):>2} sar={ans['sar']['file']!s:<5} "
              f"final={[x['action'] for x in ans['next_best_actions']['final']]} {time.time() - t:.1f}s"
              + (f"  INVALID: {errs}" if errs else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
