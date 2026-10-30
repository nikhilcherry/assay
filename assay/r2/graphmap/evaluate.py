"""Score the three mappers on a set of graph schemas with ground-truth comments.

  python -m assay.r2.graphmap.evaluate 'data/round2/graphmap_handout/*.gsql'
  python -m assay.r2.graphmap.evaluate 'data/round2/graphmap_blind2/*.gsql'   # held out

LLM answers are cached per schema (data/round2/graphmap_llm/), so a re-run
reproduces the recorded numbers without new model calls."""

from __future__ import annotations

import concurrent.futures as cf
import glob
import sys
from pathlib import Path

from assay.r2.graphmap.gsql import ground_truth, parse
from assay.r2.graphmap.llm import PROMPT_VERSION, map_schema_llm, map_schema_reviewed
from assay.r2.graphmap.mapper import map_schema
from assay.r2.graphmap.score import grade

CACHE = Path(f"data/round2/graphmap_llm/gemini-3.5-flash_{PROMPT_VERSION}")


def one(f: str):
    s = parse(f)
    gt = ground_truth(f)
    rules = map_schema(s)
    llm, _ = map_schema_llm(s, cache=CACHE / f"{Path(f).stem}.json")
    rev, drops = map_schema_reviewed(s, rules, cache_dir=CACHE)
    g = [grade(m, gt, s) for m in (rules, llm, rev)]
    return Path(f).stem, [x["right"] for x in g], g[0]["graded"], len(drops)


def main():
    for pat in sys.argv[1:]:
        tot = [0, 0, 0, 0]
        with cf.ThreadPoolExecutor(8) as ex:
            for name, (a, b, c), n, d in ex.map(one, sorted(glob.glob(pat))):
                print(f"  {name:30s} rules {a:2d}  llm {b:2d}  llm+review {c:2d}  of {n}   schema-check drops {d}")
                tot = [tot[0] + a, tot[1] + b, tot[2] + c, tot[3] + n]
        print(f"{pat}: rules {tot[0]}  llm {tot[1]}  llm+review {tot[2]}  of {tot[3]}")


if __name__ == "__main__":
    main()
