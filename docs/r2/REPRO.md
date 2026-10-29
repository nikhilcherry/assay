# Reproducing the Final-Round Run from a Clean Clone

Data is not tracked in git. Python 3.12+ is required (`python3 -m venv .venv`); Docker is only needed for the containerized run or the TigerGraph Community Edition 4.2.5 graph path. A clean clone runs in local mode (no `ASSAY_TG`) and decides identically to the graph mode that produced `cases_r2/` (`docs/r2/C6_tg_cohorts.md`).

## 1. From-Zero Setup and Execution

```bash
git clone <repo> assay && cd assay

# 1. Create virtualenv and install pinned requirements (~60–90 s)
make setup

# 2. Download organiser Google Drive datasets into data/ and data/round2/
.venv/bin/pip install gdown
make data                  # prints the two exact gdown commands below:
.venv/bin/gdown --folder 1YDJUW1fiE7Jx8R9KqknC4IcsED9zll2A -O data
.venv/bin/gdown --folder 1qmpJfwnAV9LJh_LXxNG4PTj_INcQRKCA -O data/round2

# 3. Stage Parquet frames and train/score models (~2–4 min total)
make stage
#   - python -m assay.data         (~35 s) -> data/stage/*.parquet
#   - python -m assay.model        (~90 s) -> data/stage/model.txt + isotonic.json
#   - python -m assay.patterns     (~20 s) -> data/stage/pattern_model.txt
#   - python -m assay.r2.frames    (~25 s) -> data/stage/r2/tx_all.parquet (650,938 rows)
#   - python -m assay.r2.score_jan (~40 s) -> data/stage/r2/scores_r2.parquet (60,196 rows)

# 4. Run a single case (the drills command, ~20 s cold)
make case CASE=HHF-007

# 5. Run all 18 final-round cases (~6 min across 18 fresh processes)
make all

# 6. Run the official organiser grader (~1 s)
make grade
```

### Expected Outputs & Timings

| Step | Command | Expected Wall Time | Output Artifact / Verification |
|---|---|---|---|
| Setup | `make setup` | ~60–90 s | `.venv/bin/python` with all packages in `requirements.txt` |
| Data | `make data` | <0.1 s (prints cmds) | `data/*.csv` (Round 1) and `data/round2/share/*` (Final Round) |
| Stage | `make stage` | ~2–4 min | `data/stage/r2/tx_all.parquet` (`(650938, 32)`), `17/18` case pack cards resolved |
| Single case | `make case CASE=HHF-007` | ~20 s cold | `cases_r2/HHF-007.json` (passes `assay.validate` with 0 errors) |
| Full pack | `make all` | ~6 min (18 fresh processes) | `cases_r2/HHF-001.json` .. `cases_r2/HHF-018.json` (exit code 0) |
| Grader | `make grade` | ~1 s | `report.json` (`hard: 0`, `policy: 0` across all 18 cases) |

## 2. Single-Case Drill Command

During live drills, run any single case from the root directory with one command:

```bash
make case CASE=HHF-007
# or directly without make:
PYTHONPATH=. .venv/bin/python -m assay.r2.agent HHF-007 --out cases_r2
```

It prints a one-line summary (`HHF-007 status=COMPLETE ...`) and writes `cases_r2/HHF-007.json`.
Exit codes: 0 answer written, 2 invalid input, 3 a tool, the graph or a startup data file failed
(`FAIL CLOSED`, no answer file). With `ASSAY_TG=1` any graph error or malformed graph reply is also exit 3.

## 3. Docker (Code-Only Image, Read-Only Data Mount)

The `Dockerfile` uses `python:3.12-slim`, runs as non-root user `assay` (UID 1000), copies code only (`.dockerignore` excludes `data/`, `.venv/`, `cases_r2/`), and sets `ENTRYPOINT ["make"]`. Once `data/` is staged on the host, mount `/app/data` read-only:

```bash
docker build -t assay .
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/cases_r2:/app/cases_r2" \
  assay case CASE=HHF-007
```

## 4. TigerGraph CE 4.2.5 Path (how `cases_r2/` is produced)

```bash
make tg-up      # Starts container `assay-tg` (tigergraph/community:4.2.5) on port 14240
                # with $PWD/data mounted read-only at /home/tigergraph/data:ro
make tg-load    # Runs assay.tg prepare && assay.tg setup && assay.r2.tg_load prepare && assay.r2.tg_load load

# Install the final-round queries (ring, cohorts, vectors, case write-back); idempotent
PYTHONPATH=. .venv/bin/python -c "from assay.r2 import tg_r2, tg_cohorts; tg_r2.install(); tg_cohorts.install()"

# Run every case with TigerGraph deciding the ring and both cohorts
ASSAY_TG=1 make all
ASSAY_TG=1 .venv/bin/python -m assay.r2.console --port 8742   # the demo console
```

Under `ASSAY_TG=1` the graph decides `r2_ring_wcc`, `r2_micro_drip_cohort`, and `r2_recipient_fanin`;
`similar_cases_vec` is queried but abstains (`docs/r2/C5_vector_vote.md`). Each case is written back as a `FraudCase`. If the graph is down or answers
wrongly, or the write-back fails ("case NOT saved to the TigerGraph graph"), the run exits 3 and writes
no answer.

### Checks that regenerate the generated docs

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/r2/test_drills.py   # live drills: 84 checks (docs/r2/B30_drills.md)
PYTHONPATH=. .venv/bin/python -m assay.r2.verify_claims               # 174 claims re-run, 0 mismatches (VERIFY_CLAIMS.md)
PYTHONPATH=. .venv/bin/python -m assay.r2.factsheet                   # FACTS.md: 6 fraud, 3 uncertain, 9 legitimate
PYTHONPATH=. .venv/bin/python -m assay.r2.ablation                    # ABLATION.md
```

## 5. Test Suite: Offline vs Data-Dependent Tests

Run the full test suite when staged data is present:

```bash
make test       # runs: .venv/bin/python -m pytest tests -q
```

Run clean-clone checks without `data/`, network, or Docker (used by `.github/workflows/r2.yml` on push):

```bash
make test-offline
# or: PYTHONPATH=. .venv/bin/python -m pytest -q tests/r2/test_repro.py
```

### Which Tests Need `data/` vs Run Without Data

| Test File / Target | Needs `data/`? | Needs TigerGraph (`ASSAY_TG=1`)? | Notes |
|---|---|---|---|
| `tests/r2/test_repro.py` | **No** | **No** | Parses `Makefile`, checks module imports, AST-scans third-party imports against `requirements.txt`, scans `assay/` and `tests/` for secret-shaped strings, verifies `Dockerfile` and `.github/workflows/r2.yml`. |
| `tests/r2/test_vector.py` | **No** (skips) | **Yes** (skips unless `ASSAY_TG=1`) | Live TigerGraph vector/traversal test; skips cleanly when `ASSAY_TG` is unset or `data/` is absent. |
| All other `tests/r2/test_*.py` & `tests/test_*.py` | **Yes** | **No** | Exercised against `assay.r2.frames.tx_all()`, `closed_cases()`, and `case_pack()` (`data/stage/r2/tx_all.parquet` and `data/round2/share/`). Any test that checks `R2.exists()` skips cleanly when `data/` is absent. |
