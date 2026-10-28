"""Clean-clone reproducibility checks. Pass without data, network or Docker."""
from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    "setup",
    "data",
    "stage",
    "case",
    "all",
    "grade",
    "test",
    "test-offline",
    "tg-up",
    "tg-load",
]
PIP_NAMES = {
    "sklearn": "scikit-learn",
    "pyTigerGraph": "pyTigerGraph",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "starlette": "fastapi",
    "pydantic": "fastapi",
}


def _makefile() -> str:
    return (ROOT / "Makefile").read_text()


def test_makefile_targets_exist() -> None:
    mf = _makefile()
    targets = set(re.findall(r"^([A-Za-z0-9_-]+):", mf, re.M))
    missing = [t for t in TARGETS if t not in targets]
    assert not missing, f"Missing Makefile targets: {missing}"


def test_makefile_required_commands() -> None:
    mf = _makefile()
    assert "python3 -m venv .venv" in mf
    assert "pip install -r requirements.txt" in mf
    assert "1YDJUW1fiE7Jx8R9KqknC4IcsED9zll2A" in mf
    assert "1qmpJfwnAV9LJh_LXxNG4PTj_INcQRKCA" in mf
    assert "assay.data" in mf
    assert "assay.r2.frames" in mf
    assert "assay.r2.score_jan" in mf
    assert "assay.r2.run_r2" in mf
    assert "grade.py" in mf
    assert "pytest" in mf
    assert "tigergraph/community:4.2.5" in mf
    assert "assay.tg" in mf
    assert "assay.r2.tg_load" in mf


def test_make_data_prints_gdown_commands() -> None:
    proc = subprocess.run(
        ["make", "-s", "data"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "gdown --folder 1YDJUW1fiE7Jx8R9KqknC4IcsED9zll2A -O data" in out
    assert "gdown --folder 1qmpJfwnAV9LJh_LXxNG4PTj_INcQRKCA -O data/round2" in out


def test_makefile_modules_importable() -> None:
    mods = set(re.findall(r"-m\s+((?:assay)[\w.]*)", _makefile()))
    assert mods >= {
        "assay.data",
        "assay.r2.frames",
        "assay.r2.score_jan",
        "assay.r2.run_r2",
        "assay.tg",
        "assay.r2.tg_load",
    }
    for m in sorted(mods):
        spec = importlib.util.find_spec(m)
        assert spec is not None, f"Module referenced in Makefile is not importable: {m}"


def _requirements() -> set[str]:
    names: set[str] = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            pkg = re.split(r"[<>=!~\[; ]", line)[0].lower().replace("_", "-")
            names.add(pkg)
            # Every requirement in requirements.txt must be pinned with a version bound
            assert re.search(r"[><=~!]=?", line), f"Unpinned requirement: {line}"
    return names


def _imports(path: Path):
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return  # reported by test_all_sources_parse
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            yield n.module.split(".")[0]


def test_third_party_imports_in_requirements() -> None:
    reqs = _requirements()
    local_roots = {
        p.stem for p in (ROOT / "assay").glob("*.py")
    } | {
        p.name for p in (ROOT / "assay").iterdir() if p.is_dir() and (p / "__init__.py").exists()
    }
    missing: dict[str, set[str]] = {}
    for p in (ROOT / "assay").rglob("*.py"):
        for top in _imports(p):
            if (
                top == "assay"
                or top in local_roots
                or top in sys.stdlib_module_names
                or top == "__future__"
            ):
                continue
            pip = PIP_NAMES.get(top, top).lower().replace("_", "-")
            if pip not in reqs:
                missing.setdefault(pip, set()).add(str(p.relative_to(ROOT)))
    assert not missing, {k: sorted(v) for k, v in missing.items()}


def test_no_secret_shaped_strings() -> None:
    pats = [
        re.compile("AI" + "za[0-9A-Za-z_-]{30,}"),
        re.compile("AQ" + r"\.[A-Za-z0-9_-]{20,}"),
        re.compile("sk" + "-[A-Za-z0-9]{20,}"),
        re.compile("gh" + "p_[A-Za-z0-9]{30,}"),
        re.compile("AKIA" + "[0-9A-Z]{16}"),
        re.compile("-" * 5 + "BEGIN " + ".*PRIVATE" + " KEY" + "-" * 5),
    ]
    hits = []
    for base in ("assay", "tests"):
        for p in (ROOT / base).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix in {".pyc", ".parquet", ".png", ".pkl"}:
                continue
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for pat in pats:
                if pat.search(text):
                    hits.append(f"{p.relative_to(ROOT)}: {pat.pattern[:8]}")
    assert not hits, hits


def test_all_sources_parse() -> None:
    bad = []
    for base in ("assay", "tests"):
        for p in (ROOT / base).rglob("*.py"):
            try:
                ast.parse(p.read_text(), filename=str(p))
            except SyntaxError as e:
                bad.append(f"{p.relative_to(ROOT)}:{e.lineno}")
    assert not bad, bad


def test_dockerfile_and_workflow_and_repro_doc() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.12-slim" in dockerfile
    assert "USER assay" in dockerfile
    assert 'ENTRYPOINT ["make"]' in dockerfile

    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert "data/" in dockerignore

    workflow = (ROOT / ".github" / "workflows" / "r2.yml").read_text()
    assert "3.12" in workflow
    assert "tests/r2/test_repro.py" in workflow

    repro_doc = (ROOT / "docs" / "r2" / "REPRO.md").read_text()
    assert "make setup" in repro_doc
    assert "make stage" in repro_doc
    assert "make case CASE=HHF-007" in repro_doc
    assert "test_repro.py" in repro_doc
