"""BLOCK C1 -- Leave-one-out ablation of the shipped agent.

"An idea scores zero unless you show a case or a measurement where it changes the
outcome. An ablation helps."

Usage:
    python -m assay.r2.ablation          # prints matrix and writes docs/r2/ABLATION.md
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

import pandas as pd

from assay import disputes
from assay.data import ROOT
from assay.r2 import agent as r2_agent
from assay.r2 import dispute_episode
from assay.r2 import investigate_r2
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import case_pack, tx_all
from assay.r2.store_r2 import R2LocalStore, case_writer, r2_pattern_model

OUT_DOC = ROOT / "docs" / "r2" / "ABLATION.md"
HARD_ACTIONS = frozenset({"BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"})


@dataclass(frozen=True)
class CaseOutcome:
    """Shipped outcome fields tracked per case across ablation runs."""

    case_id: str
    verdict: str
    fraud_probability: float
    pattern: str
    exposure_usd: float
    connected_card_ids_count: int
    sar_filed: bool
    final_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "verdict": self.verdict,
            "fraud_probability": self.fraud_probability,
            "pattern": self.pattern,
            "exposure_usd": self.exposure_usd,
            "connected_card_ids_count": self.connected_card_ids_count,
            "sar_filed": self.sar_filed,
            "final_actions": list(self.final_actions),
        }


@dataclass(frozen=True)
class ConfigSpec:
    """Specification for an ablation configuration."""

    name: str
    category: str
    description: str
    ablatable: bool = True
    not_ablatable_reason: str = ""


@lru_cache(maxsize=1)
def _base_store_and_model() -> tuple[R2LocalStore, Any]:
    """Build R2LocalStore and r2_pattern_model once per process to keep runtime fast."""
    return R2LocalStore(), r2_pattern_model()


def clear_leaking_caches() -> None:
    """Clear any module-level caches that could leak state across ablation configs."""
    if hasattr(dispute_episode.backtest_jul_oct_customer_reports, "cache_clear"):
        dispute_episode.backtest_jul_oct_customer_reports.cache_clear()
    if hasattr(r2_agent.get_shared_runtime, "cache_clear"):
        r2_agent.get_shared_runtime.cache_clear()
    if hasattr(disputes._report, "cache_clear"):
        disputes._report.cache_clear()


@contextmanager
def _noop_patch() -> Iterator[None]:
    yield


def _make_detector_patch(det_name: str) -> Callable[[], Iterator[None]]:
    @contextmanager
    def _patch() -> Iterator[None]:
        orig_enabled = list(investigate_r2.ENABLED)
        try:
            investigate_r2.ENABLED = [d for d in orig_enabled if d != det_name]
            yield
        finally:
            investigate_r2.ENABLED = orig_enabled

    return _patch


def _make_all_r2_detectors_patch() -> Callable[[], Iterator[None]]:
    @contextmanager
    def _patch() -> Iterator[None]:
        orig_enabled = list(investigate_r2.ENABLED)
        try:
            investigate_r2.ENABLED = []
            yield
        finally:
            investigate_r2.ENABLED = orig_enabled

    return _patch


@contextmanager
def _patch_jan_drift_guard() -> Iterator[None]:
    """Switch off the January-drift guard (`model_trusted(f): pd.Timestamp(f.t) < MODEL_TRAINED_UNTIL`).

    We patch `R2Investigator.model_trusted` to return `True` so January scores are trusted
    to clear cases just like Jul-Dec scores, while keeping `MODEL_TRAINED_UNTIL` intact for
    January affected_txn filtering (or also restoring `MODEL_TRAINED_UNTIL` in finally).
    """
    orig_model_trusted = investigate_r2.R2Investigator.model_trusted
    orig_trained_until = investigate_r2.MODEL_TRAINED_UNTIL
    try:
        investigate_r2.R2Investigator.model_trusted = lambda self, f: True  # type: ignore[method-assign]
        yield
    finally:
        investigate_r2.R2Investigator.model_trusted = orig_model_trusted  # type: ignore[method-assign]
        investigate_r2.MODEL_TRAINED_UNTIL = orig_trained_until


@contextmanager
def _patch_card_mismatch_guard() -> Iterator[None]:
    """Switch off the card-mismatch guard (HHF-011, DQ-07) in `R2Investigator._extra_findings`."""
    orig_extra_findings = investigate_r2.R2Investigator._extra_findings

    def _patched_extra_findings(self: investigate_r2.R2Investigator, case: Any, f: Any):
        out, scenario = orig_extra_findings(self, case, f)
        self._card_mismatch = None
        return out, scenario

    try:
        investigate_r2.R2Investigator._extra_findings = _patched_extra_findings  # type: ignore[method-assign]
        yield
    finally:
        investigate_r2.R2Investigator._extra_findings = orig_extra_findings  # type: ignore[method-assign]


@contextmanager
def _patch_dispute_episode_builder() -> Iterator[None]:
    """Switch off the N7 dispute episode builder so customer reports use the Round 1 episode window."""
    orig_ep_fn = dispute_episode.episode_for_dispute

    def _disabled_episode_for_dispute(
        card_id: str,
        flagged_txn_id: int,
        as_of: pd.Timestamp | str,
        tx: pd.DataFrame | None = None,
    ) -> tuple[list[int], Attestation]:
        fid = int(flagged_txn_id)
        return [fid], Attestation(
            what=f"Ablated dispute episode builder for {fid}",
            evidence=Evidence(txn_ids=[fid], card_ids=[str(card_id)]),
            query="assay.r2.dispute_episode.episode_for_dispute",
            params={"card_id": str(card_id), "flagged_txn_id": fid, "as_of": str(as_of)},
        )

    try:
        dispute_episode.episode_for_dispute = _disabled_episode_for_dispute
        yield
    finally:
        dispute_episode.episode_for_dispute = orig_ep_fn


@contextmanager
def _patch_episode_cap_hours() -> Iterator[None]:
    """Switch off the 1h cap (`EPISODE_CAP_HOURS`) on `dispute_episode.episode_for_dispute`."""
    orig_cap = dispute_episode.EPISODE_CAP_HOURS
    try:
        dispute_episode.EPISODE_CAP_HOURS = 24.0 * 365.0
        yield
    finally:
        dispute_episode.EPISODE_CAP_HOURS = orig_cap


@contextmanager
def _patch_dispute_weight_override() -> Iterator[None]:
    """Switch off `disputes.W_OVERRIDE` so disputes revert to Round 1's 50/50 assumption (`W_HISTORY = 0.5`)."""
    orig_fn = investigate_r2.measured_exam_dispute_weight
    orig_w = disputes.W_OVERRIDE
    orig_startup_w = r2_agent._STARTUP_W_OVERRIDE
    try:
        investigate_r2.measured_exam_dispute_weight = lambda: (None, {"mode": "r1_default_0.5"})  # type: ignore[assignment]
        disputes.W_OVERRIDE = None
        r2_agent._STARTUP_W_OVERRIDE = None  # type: ignore[assignment]
        yield
    finally:
        investigate_r2.measured_exam_dispute_weight = orig_fn
        disputes.W_OVERRIDE = orig_w
        r2_agent._STARTUP_W_OVERRIDE = orig_startup_w


@contextmanager
def _patch_assumed_clear_guard() -> Iterator[None]:
    """Switch off `R2Investigator.assumed_clear_allowed` so simulated customer confirmation is always allowed."""
    orig_fn = investigate_r2.R2Investigator.assumed_clear_allowed
    try:
        investigate_r2.R2Investigator.assumed_clear_allowed = lambda self, findings: None  # type: ignore[method-assign]
        yield
    finally:
        investigate_r2.R2Investigator.assumed_clear_allowed = orig_fn  # type: ignore[method-assign]


CONFIG_SPECS: list[ConfigSpec] = [
    ConfigSpec(
        name="full",
        category="baseline",
        description="Full shipped R2 agent (all 6 enabled detectors and all guards active).",
    ),
    *(
        ConfigSpec(
            name=f"no_{det}",
            category="detector",
            description=f"Disable `{det}` from `investigate_r2.ENABLED`.",
        )
        for det in investigate_r2.ENABLED
    ),
    ConfigSpec(
        name="no_r2_detectors",
        category="detector_group",
        description="Disable all 6 R2 detectors (`investigate_r2.ENABLED = []`).",
    ),
    ConfigSpec(
        name="no_jan_drift_guard",
        category="guard",
        description="Disable January-drift guard (`R2Investigator.model_trusted` returns True for Jan; see `MODEL_TRAINED_UNTIL`).",
    ),
    ConfigSpec(
        name="no_card_mismatch_guard",
        category="guard",
        description="Disable DQ-07 card-mismatch guard on HHF-011 (`_card_mismatch = None`).",
    ),
    ConfigSpec(
        name="no_dispute_episode",
        category="guard",
        description="Disable N7 dispute episode builder (`dispute_episode.episode_for_dispute` falls back to Round 1 window).",
    ),
    ConfigSpec(
        name="no_episode_cap_hours",
        category="guard",
        description="Disable 1h episode cap (`dispute_episode.EPISODE_CAP_HOURS` set to uncapped 8760h).",
    ),
    ConfigSpec(
        name="no_w_override",
        category="guard",
        description="Disable measured exam dispute weight (`disputes.W_OVERRIDE = None`, reverting to Round 1 `W_HISTORY = 0.5`).",
    ),
    ConfigSpec(
        name="no_assumed_clear_guard",
        category="guard",
        description="Disable `R2Investigator.assumed_clear_allowed` guard (allows simulated customer confirmation even when detectors/mismatch fire).",
    ),
]

NOT_ABLATABLE_GUARDS: list[ConfigSpec] = [
    ConfigSpec(
        name="uncertain_r9_undocumented_strip",
        category="guard",
        description="Inline `if c['verdict'] == 'uncertain':` block inside `R2Investigator.run` that clears `pattern='undocumented'` and prepends `MONITOR_CONNECTED_CARDS`.",
        ablatable=False,
        not_ablatable_reason=(
            "Embedded inline inside `R2Investigator.run` after `super().run(case)` with no module attribute or helper hook; "
            "switching it off would require rewriting `R2Investigator.run`."
        ),
    ),
    ConfigSpec(
        name="card_id_resolution_in_ctx",
        category="guard",
        description="Inline resolution of `card = self._tx.loc[self._tx.TransactionID == fid, 'card_id'].iat[0]` inside `R2Investigator._ctx`.",
        ablatable=False,
        not_ablatable_reason=(
            "Hardcoded local lookup inside `R2Investigator._ctx` and `Investigator.run` (`self.s.get_transaction(fid)`); "
            "cannot be toggled via a module attribute without replacing `_ctx` and `Investigator.run`."
        ),
    ),
]

_PATCHERS: dict[str, Callable[[], Iterator[None]]] = {
    "full": _noop_patch,
    **{f"no_{det}": _make_detector_patch(det) for det in investigate_r2.ENABLED},
    "no_r2_detectors": _make_all_r2_detectors_patch(),
    "no_jan_drift_guard": _patch_jan_drift_guard,
    "no_card_mismatch_guard": _patch_card_mismatch_guard,
    "no_dispute_episode": _patch_dispute_episode_builder,
    "no_episode_cap_hours": _patch_episode_cap_hours,
    "no_w_override": _patch_dispute_weight_override,
    "no_assumed_clear_guard": _patch_assumed_clear_guard,
}

# Public registry of ablatable config names -> context manager factory
CONFIGS: dict[str, Callable[[], Iterator[None]]] = dict(_PATCHERS)


@contextmanager
def apply_config(name: str) -> Iterator[None]:
    """Context manager that applies the monkeypatch for `name` and restores all state in `finally`."""
    if name not in CONFIGS:
        raise KeyError(f"Unknown ablation config: {name!r}. Valid configs: {list(CONFIGS)}")
    clear_leaking_caches()
    cm_factory = CONFIGS[name]
    with cm_factory():
        try:
            yield
        finally:
            clear_leaking_caches()


def _run_single_case_shipped_entry(
    inv: investigate_r2.R2Investigator,
    store: R2LocalStore,
    raw_row: pd.Series,
    tx_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run one case through the exact entry logic used by the shipped agent (`agent.py` / `run_r2.py`)."""
    store.memory.clear()
    raw_dict = raw_row.to_dict()
    case_id = str(raw_dict["case_id"])
    trigger_type = str(raw_dict.get("trigger_type", "") or "")
    trigger_text = str(raw_dict.get("trigger_text", "") or "")

    facts = r2_agent.parse_trigger(trigger_type, trigger_text)
    facts.claims_of_authority = [r2_agent.scrub_secrets_and_pii(c) for c in facts.claims_of_authority]
    warnings = [r2_agent.scrub_secrets_and_pii(w) for w in r2_agent.cross_check(facts, raw_dict)]

    clean_series = r2_agent._sanitize_trigger_for_row(pd.Series(raw_dict), facts)
    inv._multi = []
    inv._card_mismatch = None
    inv._case = None

    ans = inv.run(clean_series)
    r2_agent._record_untrusted_evidence(ans, case_id, facts, warnings)

    opened_ts = pd.Timestamp(raw_dict["opened_at"])
    tx_as_of = tx_df[tx_df["t"] <= opened_ts]
    r2_agent._attach_sar_if_required(ans, tx_as_of)
    return r2_agent._to_json_safe(ans)


def run_config(name: str) -> dict[str, CaseOutcome]:
    """Run configuration `name` over all 18 `case_pack()` rows.

    Builds a fresh `R2Investigator` inside the config's context manager (since detectors and
    `disputes.W_OVERRIDE` are loaded at construction) while reusing the cached immutable
    `R2LocalStore` and `r2_pattern_model` frames for speed.
    """
    store, model = _base_store_and_model()
    tx_df = tx_all()
    cp = case_pack().sort_values("case_id")

    outcomes: dict[str, CaseOutcome] = {}
    with apply_config(name):
        inv = investigate_r2.R2Investigator(store, model, writer=case_writer(store))
        for _, row in cp.iterrows():
            ans = _run_single_case_shipped_entry(inv, store, row, tx_df)
            c = ans["case"]
            nba_final = tuple(str(x["action"]) for x in ans["next_best_actions"]["final"])
            outcomes[str(row.case_id)] = CaseOutcome(
                case_id=str(row.case_id),
                verdict=str(c["verdict"]),
                fraud_probability=round(float(c["fraud_probability"]), 2),
                pattern=str(c["pattern"]),
                exposure_usd=round(float(c["exposure_usd"]), 2),
                connected_card_ids_count=len(c.get("connected_card_ids", [])),
                sar_filed=bool(ans.get("sar", {}).get("file", False)),
                final_actions=nba_final,
            )
    return outcomes


@dataclass
class DiffReport:
    """Comparison between the `full` configuration and an ablated configuration `other`."""

    config_name: str
    per_case_changes: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    exposure_left_fraud_usd: float = 0.0
    exposure_delta_on_full_fraud_usd: float = 0.0
    cases_leaving_fraud: list[str] = field(default_factory=list)
    new_hard_action_cases: list[str] = field(default_factory=list)
    dropped_hard_action_cases: list[str] = field(default_factory=list)

    @property
    def n_changed_cases(self) -> int:
        return len(self.per_case_changes)

    @property
    def new_hard_actions_count(self) -> int:
        return len(self.new_hard_action_cases)


def diff(
    full: dict[str, CaseOutcome],
    other: dict[str, CaseOutcome],
    config_name: str = "other",
) -> DiffReport:
    """Compute per-case field changes, money view, and innocent-risk view between `full` and `other`."""
    per_case: dict[str, dict[str, tuple[Any, Any]]] = {}
    exposure_left_fraud = 0.0
    exposure_delta_on_full_fraud = 0.0
    cases_leaving_fraud: list[str] = []
    new_hard_cases: list[str] = []
    dropped_hard_cases: list[str] = []

    fields = (
        "verdict",
        "fraud_probability",
        "pattern",
        "exposure_usd",
        "connected_card_ids_count",
        "sar_filed",
        "final_actions",
    )

    for cid in sorted(full.keys()):
        f_out = full[cid]
        o_out = other[cid]
        changes: dict[str, tuple[Any, Any]] = {}
        for fld in fields:
            v_f = getattr(f_out, fld)
            v_o = getattr(o_out, fld)
            if v_f != v_o:
                changes[fld] = (v_f, v_o)
        if changes:
            per_case[cid] = changes

        # Money view: sum of exposure on cases whose verdict leaves 'fraud'
        if f_out.verdict == "fraud" and o_out.verdict != "fraud":
            exposure_left_fraud += f_out.exposure_usd
            exposure_delta_on_full_fraud += f_out.exposure_usd
            cases_leaving_fraud.append(cid)
        elif f_out.verdict == "fraud" and o_out.verdict == "fraud" and f_out.exposure_usd > o_out.exposure_usd:
            exposure_delta_on_full_fraud += (f_out.exposure_usd - o_out.exposure_usd)

        # Innocent-risk view: cases that newly get BLOCK_CARD / BLOCK_ALL_CARDS / FILE_REPORT
        f_hard = set(f_out.final_actions) & HARD_ACTIONS
        o_hard = set(o_out.final_actions) & HARD_ACTIONS
        if o_hard - f_hard:
            new_hard_cases.append(cid)
        if f_hard - o_hard:
            dropped_hard_cases.append(cid)

    return DiffReport(
        config_name=config_name,
        per_case_changes=per_case,
        exposure_left_fraud_usd=round(exposure_left_fraud, 2),
        exposure_delta_on_full_fraud_usd=round(exposure_delta_on_full_fraud, 2),
        cases_leaving_fraud=cases_leaving_fraud,
        new_hard_action_cases=new_hard_cases,
        dropped_hard_action_cases=dropped_hard_cases,
    )


def _format_cell_change(f_out: CaseOutcome, o_out: CaseOutcome, changes: dict[str, tuple[Any, Any]]) -> str:
    """Compact summary of how a case changed from `full` to `other` for the markdown matrix."""
    if not changes:
        return "—"
    parts: list[str] = []
    if "verdict" in changes or "fraud_probability" in changes:
        parts.append(f"{f_out.verdict}({f_out.fraud_probability:.2f})->{o_out.verdict}({o_out.fraud_probability:.2f})")
    if "pattern" in changes:
        parts.append(f"pat:{f_out.pattern}->{o_out.pattern}")
    if "exposure_usd" in changes:
        parts.append(f"exp:${f_out.exposure_usd:,.2f}->${o_out.exposure_usd:,.2f}")
    if "connected_card_ids_count" in changes:
        parts.append(f"conn:{f_out.connected_card_ids_count}->{o_out.connected_card_ids_count}")
    if "sar_filed" in changes:
        parts.append(f"SAR:{f_out.sar_filed}->{o_out.sar_filed}")
    if "final_actions" in changes and "verdict" not in changes and "sar_filed" not in changes:
        parts.append(f"act:{list(f_out.final_actions)}->{list(o_out.final_actions)}")
    return "; ".join(parts)


def run_all_ablations() -> tuple[dict[str, dict[str, CaseOutcome]], dict[str, DiffReport]]:
    """Run `full` and every ablated configuration in `CONFIGS`."""
    all_runs: dict[str, dict[str, CaseOutcome]] = {}
    full_res = run_config("full")
    all_runs["full"] = full_res

    diffs: dict[str, DiffReport] = {"full": diff(full_res, full_res, config_name="full")}
    for name in CONFIGS:
        if name == "full":
            continue
        res = run_config(name)
        all_runs[name] = res
        diffs[name] = diff(full_res, res, config_name=name)
    return all_runs, diffs


def format_markdown_report(
    all_runs: dict[str, dict[str, CaseOutcome]],
    diffs: dict[str, DiffReport],
) -> str:
    """Build the complete `docs/r2/ABLATION.md` report from measured ablation results."""
    full_res = all_runs["full"]
    changed_case_ids = sorted({cid for d in diffs.values() for cid in d.per_case_changes})

    lines: list[str] = [
        "# Leave-One-Out Ablation of the Shipped R2 Agent",
        "",
        "Generated by `python -m assay.r2.ablation`; do not edit.",
        "",
        "## 1. Summary & Methodology",
        "",
        "Each row switches off one detector (`investigate_r2.ENABLED`) or one guard via a temporary",
        "monkeypatch context manager, constructs a fresh `R2Investigator`, clears module caches, and",
        "re-runs all 18 exam cases (`case_pack()`) through the shipped agent pipeline.",
        "",
        "- **Money view (`$ exposure no longer acted on`)**: sum of `exposure_usd` on cases whose verdict",
        "  leaves `fraud` (plus total exposure reduction on `fraud` cases when episode scoping changes).",
        "- **Innocent-risk view (`new hard actions`)**: count of cases that newly receive `BLOCK_CARD`,",
        "  `BLOCK_ALL_CARDS`, or `FILE_REPORT` compared to `full`.",
        "",
        "## 2. Ablation Matrix (Cases That Change Across Configs)",
        "",
    ]

    headers = (
        ["Config", "Category"]
        + changed_case_ids
        + ["$ exposure no longer acted on", "$ fraud exposure delta", "new hard actions"]
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    spec_map = {s.name: s for s in CONFIG_SPECS}
    for cfg_name in CONFIGS:
        spec = spec_map[cfg_name]
        d = diffs[cfg_name]
        other_res = all_runs[cfg_name]
        row_cells = [f"`{cfg_name}`", spec.category]
        for cid in changed_case_ids:
            ch = d.per_case_changes.get(cid, {})
            row_cells.append(_format_cell_change(full_res[cid], other_res[cid], ch))
        leave_str = f"${d.exposure_left_fraud_usd:,.2f}"
        if d.cases_leaving_fraud:
            leave_str += f" ({', '.join(d.cases_leaving_fraud)})"
        delta_str = f"${d.exposure_delta_on_full_fraud_usd:,.2f}"
        hard_str = str(d.new_hard_actions_count)
        if d.new_hard_action_cases:
            hard_str += f" ({', '.join(d.new_hard_action_cases)})"
        row_cells.extend([leave_str, delta_str, hard_str])
        lines.append("| " + " | ".join(row_cells) + " |")

    lines += [
        "",
        "## 3. Detailed Per-Config Field Changes",
        "",
        "| Config | Description | Cases Changed | Exact Field Diffs (full -> ablated) |",
        "|---|---|---|---|",
    ]
    for cfg_name in CONFIGS:
        spec = spec_map[cfg_name]
        d = diffs[cfg_name]
        if not d.per_case_changes:
            lines.append(f"| `{cfg_name}` | {spec.description} | 0 | none |")
            continue
        case_summaries: list[str] = []
        for cid, ch_dict in sorted(d.per_case_changes.items()):
            f_str = ", ".join(f"{k}: `{v[0]}` -> `{v[1]}`" for k, v in ch_dict.items())
            case_summaries.append(f"**{cid}** ({f_str})")
        lines.append(
            f"| `{cfg_name}` | {spec.description} | {d.n_changed_cases} | {' ; '.join(case_summaries)} |"
        )

    lines += [
        "",
        "## 4. Baseline (`full`) Outcomes on All 18 Exam Cases",
        "",
        "| Case | Verdict | p | Pattern | Exposure (USD) | Connected Cards | SAR Filed | Final Actions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cid, out in sorted(full_res.items()):
        lines.append(
            f"| {cid} | {out.verdict} | {out.fraud_probability:.2f} | {out.pattern} | "
            f"${out.exposure_usd:,.2f} | {out.connected_card_ids_count} | {out.sar_filed} | "
            f"`{list(out.final_actions)}` |"
        )

    lines += [
        "",
        "## 5. Guards Not Ablatable Without Editing Source",
        "",
        "| Guard | Description | Why Not Ablatable via Attribute Patch |",
        "|---|---|---|",
    ]
    for g in NOT_ABLATABLE_GUARDS:
        lines.append(f"| `{g.name}` | {g.description} | {g.not_ablatable_reason} |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    _ = argv
    all_runs, diffs = run_all_ablations()
    md = format_markdown_report(all_runs, diffs)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
