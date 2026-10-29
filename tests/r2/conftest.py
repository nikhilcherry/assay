"""Final-round tests read the organiser's data (gitignored, fetched with `make data`).

Without it, only the tests that need no data are collected, so CI on a fresh clone checks
those and skips the rest instead of failing at import.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "round2" / "share" / "closed_cases_history.csv"

OFFLINE = {
    "test_approvals.py",
    "test_harness.py",
    "test_policy_check.py",
    "test_repro.py",
    "test_submission_docs.py",
}

collect_ignore = []
if not DATA.exists():
    collect_ignore = sorted(p.name for p in Path(__file__).parent.glob("test_*.py") if p.name not in OFFLINE)
