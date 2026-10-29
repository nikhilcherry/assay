from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def test_repro_doc_matches_makefile_and_cold_timing():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    repro = (ROOT / "docs" / "r2" / "REPRO.md").read_text(encoding="utf-8")

    # 1. Every python -m <module> in Makefile `stage:` is documented in REPRO.md
    stage_match = re.search(r"^stage:\n\t(.+)$", makefile, flags=re.MULTILINE)
    assert stage_match is not None
    stage_modules = re.findall(r"-m\s+(assay\.[a-zA-Z0-9_.]+)", stage_match.group(1))
    assert "assay.patterns" in stage_modules
    for mod in stage_modules:
        assert f"python -m {mod}" in repro, f"Missing {mod} in docs/r2/REPRO.md"

    # 2. Single-case CLI without make uses hardened assay.r2.agent entrypoint, not run_r2
    assert "assay.r2.run_r2" not in repro
    assert "python -m assay.r2.agent HHF-007 --out cases_r2" in repro

    # 3. Single-case timing reflects ~20 s cold load rather than ~1.5 s / ~1-2 s
    assert "~1.5 s" not in repro
    assert "~1–2 s" not in repro
    assert "~20 s cold" in repro
