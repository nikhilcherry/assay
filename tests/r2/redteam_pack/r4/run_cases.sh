#!/bin/bash
# usage: runcases.sh <overlay>
cd "$1"; export PYTHONPATH=$PWD; unset ASSAY_TG
P=/home/nikhil/Projects/assay/.venv/bin/python
mkdir -p cases_r2 logs
ids=$($P -c "import pandas as pd;print(' '.join(pd.read_csv('data/round2/share/cases/case_pack_final.csv').case_id))")
echo "{" > /dev/null
: > status.tsv
for c in $ids; do
  $P -m assay.r2.agent $c --out cases_r2 > logs/$c.log 2>&1; echo -e "$c\t$?" >> status.tsv
done
$P - <<PY
import json
d={l.split()[0]:int(l.split()[1]) for l in open('status.tsv')}
json.dump(d,open('redteam_status.json','w'),indent=1)
PY
