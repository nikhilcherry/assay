"""Reproducers written by the Argon swarm; each one failed before its fix.

They read the organiser data, so a fresh clone without it skips them."""
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "round2" / "share" / "closed_cases_history.csv"
collect_ignore_glob = [] if DATA.exists() else ["test_*.py"]
