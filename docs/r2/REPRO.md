# Reproducing the Final-Round Run from a Clean Clone

Data is not tracked in git. Python 3.12+ is required (`python3 -m venv .venv`); Docker is only needed for the containerized run or the optional TigerGraph Community Edition 4.2.5 graph path.

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
#   - python -m assay.model        (~90 s) -> data/stage/pattern_lgb.pkl
#   - python -m assay.r2.frames    (~25 s) -> data/stage/r2/tx_all.parquet (650,938 rows)
#   - python -m assay.r2.score_jan (~40 s) -> data/stage/r2/jan_scores.parquet (60,196 rows)

# 4. Run a single case (the drills command, ~1.5 s)
make case CASE=HHF-007

# 5. Run all 18 final-round cases (~15–25 s)
make all

# 6. Run the official organiser grader (~1 s)
make grade
```

### Expected Outputs & Timings

| Step | Command | Expected Wall Time | Output Artifact / Verification |
|---|---|---|---|
| Setup | `make setup` | ~60–90 s | `.venv/bin/python` with all packages in `requirements.txt` |
| Data | `make data` | <0.1 s (prints cmds) | `data/*.csv` (Round 1) and `data/round2/share/*` (Final Round) |
| Stage | `make stage` | ~2–4 min | `data/stage/r2/tx_all.parquet` (`(650938, 32)`), `18/18` case pack cards resolved |
| Single case | `make case CASE=HHF-007` | ~1–2 s | `cases_r2/HHF-007.json` (passes `assay.validate` with 0 errors) |
| Full pack | `make all` | ~15–25 s | `cases_r2/HHF-001.json` .. `cases_r2/HHF-018.json` (exit code 0) |
| Grader | `make grade` | ~1 s | `report.json` (`hard: 0`, `policy: 0` across all 18 cases) |

## 2. Single-Case Drill Command

During live drills, run any single case from the root directory with one command:

```bash
make case CASE=HHF-007
# or directly without make:
PYTHONPATH=. .venv/bin/python -m assay.r2.run_r2 HHF-007
```

It prints a one-line summary (`HHF-007 customer_report ...`) and writes `cases_r2/HHF-007.json`.

## 3. Docker (Code-Only Image, Read-Only Data Mount)

The `Dockerfile` uses `python:3.12-slim`, runs as non-root user `assay` (UID 1000), copies code only (`.dockerignore` excludes `data/`, `.venv/`, `cases_r2/`), and sets `ENTRYPOINT ["make"]`. Once `data/` is staged on the host, mount `/app/data` read-only:

```bash
docker build -t assay .
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/cases_r2:/app/cases_r2" \
  assay case CASE=HHF-007
```

## 4. TigerGraph CE 4.2.5 Path (Optional)

```bash
make tg-up      # Starts container `assay-tg` (tigergraph/community:4.2.5) on port 14240
                # with $PWD/data mounted read-only at /home/tigergraph/data:ro
make tg-load    # Runs assay.tg prepare && assay.tg setup && assay.r2.tg_load prepare && assay.r2.tg_load load
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
