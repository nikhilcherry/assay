# assay -- final-round (R2) reproduction. See docs/r2/REPRO.md.
PY ?= .venv/bin/python
CASE ?= HHF-007
export PYTHONPATH := .

R1_FOLDER := 1YDJUW1fiE7Jx8R9KqknC4IcsED9zll2A
R2_FOLDER := 1qmpJfwnAV9LJh_LXxNG4PTj_INcQRKCA
SHARE := data/round2/share

.PHONY: setup data stage case all grade test test-offline tg-up tg-load

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Prints the exact gdown commands for the two Drive folders; nothing is fetched automatically.
data:
	@echo "# Round 1 data -> data/"
	@echo ".venv/bin/gdown --folder $(R1_FOLDER) -O data"
	@echo "# Final-round data -> data/round2/ (lands in data/round2/share/)"
	@echo ".venv/bin/gdown --folder $(R2_FOLDER) -O data/round2"

stage:
	$(PY) -m assay.data && $(PY) -m assay.model && $(PY) -m assay.r2.frames && $(PY) -m assay.r2.score_jan

# One case, the command the drills use (set ASSAY_TG=1 to use TigerGraph for rings and write-back)
case:
	$(PY) -m assay.r2.agent $(CASE) --out cases_r2

# The shipped answers: each case in its own fresh process, exactly as `make case` runs it
all:
	for c in $$($(PY) -c "import pandas as p; print(' '.join(p.read_csv('$(SHARE)/cases/case_pack_final.csv').case_id))"); do \
		$(PY) -m assay.r2.agent $$c --out cases_r2 || exit 1; done

grade:
	rm -rf $${TMPDIR:-/tmp}/assay-grade && mkdir -p $${TMPDIR:-/tmp}/assay-grade/teams/assay/cases
	cp cases_r2/*.json $${TMPDIR:-/tmp}/assay-grade/teams/assay/cases/
	$(PY) $(SHARE)/grader/grade.py --teams-dir $${TMPDIR:-/tmp}/assay-grade/teams \
		--case-pack $(SHARE)/cases/case_pack_final.csv \
		--transactions $(SHARE)/transactions_2017q1.csv --out report.json

test:
	$(PY) -m pytest tests -q

# Only tests that need neither data nor a database.
test-offline:
	env -u ASSAY_TG $(PY) -m pytest -q tests/r2/test_repro.py

tg-up:
	docker start assay-tg 2>/dev/null || docker run -d --name assay-tg --ulimit nofile=1000000:1000000 -p 14240:14240 \
		-v "$$PWD/data:/home/tigergraph/data:ro" -t tigergraph/community:4.2.5

tg-load:
	$(PY) -m assay.tg prepare
	$(PY) -m assay.tg setup
	$(PY) -m assay.r2.tg_load prepare
	$(PY) -m assay.r2.tg_load load
