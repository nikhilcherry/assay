"""Challenge 13 -- Someone else's schema: fingerprint-based column and table
matching with Hungarian assignment, explicit separation of SOURCE vs DERIVED
canonical columns (recomputed via the exact `assay.r2.frames.stage()` code path
with `assay.store.device_profile` and `TransactionDT` resolved from the variant
or joined back through `TransactionID` on `frames.tx_all()`, or marked honestly
as `not recoverable from this variant`), structural reshape recovery (splits,
merges, date+time splits, one-hot channels, cents/epoch transforms, separate
identity tables), proposal-file persistence, role-enforced human approval
(`approve(proposal_id, approver_role)`), confidence floor refusal, and attested
sample-record justifications."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import R2_STAGE, tx_all
from assay.store import device_profile

# Source columns stored by a bank / handed out in a variant (must be mapped from the variant)
SOURCE_COLS: list[str] = [
    "TransactionID", "ts", "card_id", "customer_id", "card4", "card6",
    "amount", "ProductCD", "channel", "addr1", "addr2", "dist1", "P_emaildomain",
    "R_emaildomain", "risk_score", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
    "id_15", "id_23", "M4", "M5", "M6", "D1", "C1",
]

# Derived canonical columns built by frames.stage()
DERIVED_COLS: list[str] = [
    "TransactionDT", "t", "month_src", "device_profile", "uid",
]

CANONICAL_COLS: list[str] = [
    "TransactionID", "TransactionDT", "ts", "card_id", "customer_id", "card4", "card6",
    "amount", "ProductCD", "channel", "addr1", "addr2", "dist1", "P_emaildomain",
    "R_emaildomain", "risk_score", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
    "id_15", "id_23", "M4", "M5", "M6", "D1", "C1", "month_src", "t",
    "device_profile", "uid",
]

NOT_RECOVERABLE: str = "not recoverable from this variant"

CONFIDENCE_FLOOR: float = 0.55
APPROVAL_THRESHOLD: float = 0.72
VALID_APPROVER_ROLES: frozenset[str] = frozenset({
    "L1", "L2", "L1_analyst", "L1_lead", "L2_manager", "team_lead", "fraud_manager",
})

PROPOSALS_DIR: Path = R2_STAGE / "schema_proposals"

_RE_CARD_ID = re.compile(r"^C\d{4,6}-K\d+$")
_RE_CUST_ID = re.compile(r"^C\d{4,6}$")
_RE_CARD_SUFFIX = re.compile(r"^-?K\d+$")
_RE_RES = re.compile(r"^\d{3,4}x\d{3,4}$")
_RE_UID = re.compile(r"^C\d{4,6}-K\d+\|(?:na|-?\d+)\|(?:na|-?\d+)$")
_RE_ISO_DT = re.compile(r"^201[67]-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
_RE_DATE_ONLY = re.compile(r"^201[67]-\d{2}-\d{2}$")
_RE_TIME_ONLY = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_OS_PREFIXES = ("android", "ios ", "windows", "mac os", "linux", "func")
_BROWSER_TOKENS = ("chrome", "safari", "firefox", "ie ", "edge", "samsung", "opera", "webview", "android browser")


@lru_cache(maxsize=1)
def _canonical_tx_lookup() -> pd.DataFrame:
    """Cached lookup of TransactionID -> (TransactionDT, month_src) from frames.tx_all()."""
    tx = tx_all()
    return tx.set_index("TransactionID")[["TransactionDT", "month_src"]]


@dataclass
class ColumnProfile:
    """Distributional and syntactic fingerprint of a single column or candidate."""
    name: str
    null_rate: float
    n_unique: int
    cardinality_ratio: float
    is_numeric: bool
    is_bool_or_01: bool
    is_int_like: bool
    is_monotonic: bool
    p0: float
    p05: float
    p50: float
    p95: float
    p100: float
    mean: float
    frac_2dec: float
    top_values: set[str]
    all_small_values: set[str]
    regex_rates: dict[str, float]


@dataclass
class MappingChoice:
    """Single canonical column mapping with expression, confidence, sample ids, and attestation."""
    canonical_col: str
    expr: str
    confidence: float
    sample_ids: list[int]
    attestation: Attestation
    is_derived: bool = False
    needs_approval: bool = False
    approved: bool = False
    notes: str = ""

    def __iter__(self) -> Iterator[Any]:
        """Unpacks as (their_expr, confidence, justification_sample_record_ids)."""
        yield self.expr
        yield self.confidence
        yield self.sample_ids

    def __getitem__(self, idx: int) -> Any:
        return (self.expr, self.confidence, self.sample_ids)[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_col": self.canonical_col,
            "expr": self.expr,
            "confidence": self.confidence,
            "sample_ids": self.sample_ids,
            "attestation": self.attestation.to_dict(),
            "is_derived": self.is_derived,
            "needs_approval": self.needs_approval,
            "approved": self.approved,
            "notes": self.notes,
        }


@dataclass
class SchemaMapping(Mapping[str, MappingChoice]):
    """Mapping from canonical column names to MappingChoice objects, persisted as a proposal."""
    choices: dict[str, MappingChoice] = field(default_factory=dict)
    join_info: dict[str, Any] = field(default_factory=dict)
    unmapped_foreign: list[str] = field(default_factory=list)
    refused_below_floor: dict[str, tuple[str, float]] = field(default_factory=dict)
    not_recoverable: dict[str, str] = field(default_factory=dict)
    proposal_id: str = ""
    proposal_path: str = ""
    proposal_approved: bool = False
    approved_by_role: str = ""
    approved_at: str = ""

    def __getitem__(self, key: str) -> MappingChoice:
        return self.choices[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.choices)

    def __len__(self) -> int:
        return len(self.choices)

    @property
    def source_choices(self) -> dict[str, MappingChoice]:
        return {k: v for k, v in self.choices.items() if k in SOURCE_COLS}

    @property
    def derived_choices(self) -> dict[str, MappingChoice]:
        return {k: v for k, v in self.choices.items() if k in DERIVED_COLS}

    def flagged_for_approval(self) -> list[MappingChoice]:
        return [c for c in self.choices.values() if c.needs_approval and not c.approved]

    def approve(self, canonical_cols: list[str] | None = None, approver: str = "L1_analyst") -> None:
        role = approver.strip()
        if role not in VALID_APPROVER_ROLES:
            raise PermissionError(
                f"Invalid approver role '{role}'. Must be one of {sorted(VALID_APPROVER_ROLES)}."
            )
        targets = canonical_cols if canonical_cols is not None else list(self.choices.keys())
        now_s = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for col in targets:
            if col in self.choices:
                self.choices[col].approved = True
                self.choices[col].notes = f"Approved by {role} at {now_s}"
        if canonical_cols is None or not self.flagged_for_approval():
            self.proposal_approved = True
            self.approved_by_role = role
            self.approved_at = now_s
        if self.proposal_path:
            self.save_proposal(Path(self.proposal_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_approved": self.proposal_approved,
            "approved_by_role": self.approved_by_role,
            "approved_at": self.approved_at,
            "join_info": self.join_info,
            "unmapped_foreign": self.unmapped_foreign,
            "not_recoverable": self.not_recoverable,
            "refused_below_floor": {
                k: {"candidate": v[0], "confidence": v[1]} for k, v in self.refused_below_floor.items()
            },
            "choices": {k: c.to_dict() for k, c in self.choices.items()},
        }

    def save_proposal(self, path: Path | None = None) -> Path:
        if path is None:
            PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
            path = PROPOSALS_DIR / f"{self.proposal_id}.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        self.proposal_path = str(path)
        return path


@dataclass
class ValidationReport:
    """Summary of applying a SchemaMapping to foreign data."""
    proposal_id: str
    recovered_cols: list[str]
    derived_cols: list[str]
    not_recoverable_cols: dict[str, str]
    missing_cols: list[str]
    flagged_low_confidence: list[str]
    skipped_unapproved: list[str]
    unmapped_foreign_cols: list[str]
    refused_below_floor: dict[str, tuple[str, float]]
    null_rate_deltas: dict[str, float]
    domain_warnings: dict[str, str]
    n_rows: int


_PROPOSAL_REGISTRY: dict[str, SchemaMapping] = {}
_REF_PROFILES_CACHE: dict[str, ColumnProfile] | None = None


def approve(
    proposal_id: str | SchemaMapping,
    approver_role: str = "L1",
    canonical_cols: list[str] | None = None,
) -> SchemaMapping:
    """Server-side human approval gate for a schema mapping proposal.

    Verifies that `approver_role` is an authorized L1/L2 role, marks the proposal
    and its choices as approved, updates the proposal JSON file on disk, and returns
    the updated SchemaMapping.
    """
    role = approver_role.strip()
    if role not in VALID_APPROVER_ROLES:
        raise PermissionError(
            f"Unauthorized approver_role '{role}'. Allowed roles: {sorted(VALID_APPROVER_ROLES)}."
        )

    if isinstance(proposal_id, SchemaMapping):
        mapping = proposal_id
        mapping.approve(canonical_cols=canonical_cols, approver=role)
        if mapping.proposal_id:
            _PROPOSAL_REGISTRY[mapping.proposal_id] = mapping
        return mapping

    pid = str(proposal_id)
    if pid in _PROPOSAL_REGISTRY:
        mapping = _PROPOSAL_REGISTRY[pid]
        mapping.approve(canonical_cols=canonical_cols, approver=role)
        return mapping

    prop_path = PROPOSALS_DIR / f"{pid}.json"
    if not prop_path.exists():
        raise KeyError(f"Unknown schema mapping proposal_id: {pid}")
    data = json.loads(prop_path.read_text())
    now_s = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["proposal_approved"] = True
    data["approved_by_role"] = role
    data["approved_at"] = now_s
    for c_key, c_val in data.get("choices", {}).items():
        if canonical_cols is None or c_key in canonical_cols:
            c_val["approved"] = True
            c_val["notes"] = f"Approved by {role} at {now_s}"
    prop_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    raise KeyError(f"Proposal {pid} updated on disk, but not in memory registry.")


def recompute_derived_columns(
    df: pd.DataFrame,
    allow_txid_lookup: bool = True,
) -> pd.DataFrame:
    """Shared pipeline function that computes the 5 DERIVED canonical columns
    (`t`, `TransactionDT`, `month_src`, `device_profile`, `uid`) from a DataFrame
    of mapped SOURCE columns using the exact code path from `assay.r2.frames.stage()`:

    - `both["t"] = pd.to_datetime(both.ts)`
    - `both["device_profile"] = [device_profile(r) for r in both[["DeviceInfo", "id_30", "id_31", "id_33"]].to_dict("records")]`
    - `TransactionDT`: if not already present in `df`, joins back through `TransactionID`
      against `frames.tx_all()` when `allow_txid_lookup=True` and IDs match; if impossible,
      leaves `TransactionDT` (and `uid` which depends on `TransactionDT // 86400`) absent
      so the caller marks them as `not recoverable from this variant`.
    - `day = both.TransactionDT // 86400`
    - `both["uid"] = (both.card_id.astype(str) + "|" + both.addr1.map(...) + "|" + (day - both.D1).map(...))`
    """
    out = df.copy()

    # 1. Exact frames.stage() derivation for `t`: both["t"] = pd.to_datetime(both.ts)
    if "ts" in out.columns:
        out["t"] = pd.to_datetime(out["ts"])

    # 2. Resolve `TransactionDT` and `month_src`
    if "TransactionDT" not in out.columns and allow_txid_lookup and "TransactionID" in out.columns:
        lookup = _canonical_tx_lookup()
        tids = pd.Series(out["TransactionID"].to_numpy(), index=out.index)
        matched_dt = tids.map(lookup["TransactionDT"])
        if float(matched_dt.notna().mean()) >= 0.95:
            out["TransactionDT"] = matched_dt.astype("int64")
            if "month_src" not in out.columns:
                out["month_src"] = tids.map(lookup["month_src"]).astype(object)

    if "month_src" not in out.columns and "t" in out.columns:
        out["month_src"] = pd.Series(
            np.where(out["t"] >= pd.Timestamp("2017-01-01"), "jan", "r1"),
            index=out.index,
            dtype=object,
        )

    # 3. Exact frames.stage() derivation for `device_profile` using imported assay.store.device_profile
    # Sanitize raw identity components first: if any component accidentally contains pipe-delimited
    # strings (e.g., from a pre-composed field), extract only its corresponding raw component slot
    # so device_profile is never composed twice.
    dev_cols = ["DeviceInfo", "id_30", "id_31", "id_33"]
    if all(c in out.columns for c in dev_cols):
        clean_dev = pd.DataFrame(index=out.index)
        for slot_idx, c in enumerate(dev_cols):
            col_s = out[c]
            has_pipe = col_s.dropna().astype(str).str.contains(r"\|", regex=True)
            if len(has_pipe) > 0 and bool(has_pipe.any()):
                parts = col_s.fillna("").astype(str).str.split("|", expand=True)
                idx_to_take = min(slot_idx, parts.shape[1] - 1)
                clean_s = parts[idx_to_take].str.strip().replace({"?": None, "": None, "nan": None, "None": None})
                clean_dev[c] = clean_s
                out[c] = clean_s
            else:
                clean_dev[c] = col_s.replace({"?": None, "": None, "nan": None, "None": None})
        out["device_profile"] = [
            device_profile(r) for r in clean_dev[dev_cols].to_dict("records")
        ]

    # 4. Exact frames.stage() derivation for `uid`
    if all(c in out.columns for c in ("card_id", "addr1", "TransactionDT", "D1")) and out["TransactionDT"].notna().all():
        day = out["TransactionDT"].astype("int64") // 86400
        out["uid"] = (
            out["card_id"].astype(str)
            + "|"
            + out["addr1"].map(lambda x: "na" if pd.isna(x) else str(int(x)))
            + "|"
            + (day - out["D1"]).map(lambda x: "na" if pd.isna(x) else str(int(x)))
        )

    return out


def _compute_profile(s: pd.Series, name: str, max_sample: int = 10000) -> ColumnProfile:
    n_total = len(s)
    if n_total == 0:
        return ColumnProfile(
            name=name, null_rate=1.0, n_unique=0, cardinality_ratio=0.0,
            is_numeric=False, is_bool_or_01=False, is_int_like=False, is_monotonic=False,
            p0=0.0, p05=0.0, p50=0.0, p95=0.0, p100=0.0, mean=0.0, frac_2dec=0.0,
            top_values=set(), all_small_values=set(), regex_rates={},
        )

    non_null = s.dropna()
    if pd.api.types.is_object_dtype(non_null.dtype) or pd.api.types.is_string_dtype(non_null.dtype):
        str_vals = non_null.astype(str)
        non_empty = str_vals[~str_vals.isin(["", "nan", "None", "null", "NA"])]
        null_rate = 1.0 - (len(non_empty) / n_total)
        non_null_clean = non_empty
    else:
        null_rate = 1.0 - (len(non_null) / n_total)
        non_null_clean = non_null

    n_clean = len(non_null_clean)
    if n_clean == 0:
        return ColumnProfile(
            name=name, null_rate=1.0, n_unique=0, cardinality_ratio=0.0,
            is_numeric=False, is_bool_or_01=False, is_int_like=False, is_monotonic=False,
            p0=0.0, p05=0.0, p50=0.0, p95=0.0, p100=0.0, mean=0.0, frac_2dec=0.0,
            top_values=set(), all_small_values=set(), regex_rates={},
        )

    if n_clean > max_sample:
        sample = non_null_clean.iloc[np.linspace(0, n_clean - 1, max_sample, dtype=int)]
    else:
        sample = non_null_clean

    n_unique = int(sample.nunique())
    card_ratio = n_unique / len(sample)

    is_dt = pd.api.types.is_datetime64_any_dtype(s.dtype)
    is_bool = pd.api.types.is_bool_dtype(s.dtype)
    is_num = pd.api.types.is_numeric_dtype(s.dtype) and not is_bool

    num_sample = pd.Series(dtype=float)
    if is_num:
        num_sample = sample.astype(float)
    elif not is_dt and not is_bool:
        coerced = pd.to_numeric(sample, errors="coerce").dropna()
        if len(coerced) >= 0.95 * len(sample):
            is_num = True
            num_sample = coerced.astype(float)

    is_bool_or_01 = is_bool
    is_int_like = False
    is_monotonic = False
    p0 = p05 = p50 = p95 = p100 = mean = frac_2dec = 0.0

    if is_num and len(num_sample) > 0:
        vals = num_sample.to_numpy()
        p0, p05, p50, p95, p100 = [float(x) for x in np.percentile(vals, [0, 5, 50, 95, 100])]
        mean = float(np.mean(vals))
        uniq_v = set(np.unique(vals[:1000]))
        if uniq_v <= {0.0, 1.0} and len(uniq_v) >= 1:
            is_bool_or_01 = True
        rem = np.abs(vals - np.round(vals))
        is_int_like = bool(np.mean(rem < 1e-6) > 0.99)
        if not is_int_like:
            rem100 = np.abs(vals * 100 - np.round(vals * 100))
            frac_2dec = float(np.mean((rem >= 1e-6) & (rem100 < 1e-4)))
        if len(vals) > 10:
            diffs = np.diff(vals)
            is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs > 0))

    str_sample = sample.astype(str)
    vc = str_sample.value_counts()
    top_values = set(vc.head(10).index)
    all_small_values = set(vc.index) if n_unique <= 20 else set()

    sl = str_sample.str.lower()
    pipe_counts = str_sample.str.count(r"\|") if not is_num else pd.Series(0, index=str_sample.index)
    regex_rates: dict[str, float] = {
        "is_dt_dtype": 1.0 if is_dt else 0.0,
        "iso_dt": float(str_sample.str.match(_RE_ISO_DT).mean()) if not is_num else 0.0,
        "date_only": float(str_sample.str.match(_RE_DATE_ONLY).mean()) if (not is_num and not is_dt) else 0.0,
        "time_only": float(str_sample.str.match(_RE_TIME_ONLY).mean()) if (not is_num and not is_dt) else 0.0,
        "card_id": float(str_sample.str.match(_RE_CARD_ID).mean()) if not is_num else 0.0,
        "cust_id": float(str_sample.str.match(_RE_CUST_ID).mean()) if not is_num else 0.0,
        "card_suffix": float(str_sample.str.match(_RE_CARD_SUFFIX).mean()) if not is_num else 0.0,
        "email": float(str_sample.str.contains(r"\.(?:com|net|org|edu|co\.uk|de|fr|es|jp|mx)$", regex=True).mean()) if not is_num else 0.0,
        "res": float(str_sample.str.match(_RE_RES).mean()) if not is_num else 0.0,
        "os": float(sl.str.startswith(_OS_PREFIXES).mean()) if not is_num else 0.0,
        "browser": float(sl.apply(lambda x: any(b in x for b in _BROWSER_TOKENS)).mean()) if not is_num else 0.0,
        "ip_proxy": float(sl.str.startswith("ip_proxy:").mean()) if not is_num else 0.0,
        "dev_prof": float((pipe_counts == 3).mean()) if not is_num else 0.0,
        "any_pipe": float((pipe_counts >= 1).mean()) if not is_num else 0.0,
        "uid": float(str_sample.str.match(_RE_UID).mean()) if not is_num else 0.0,
    }
    if not is_num and not is_dt:
        dev_info_hits = str_sample.str.contains(
            r"Build/|Trident/|rv:|Windows|Mac OS|iOS Device|SM-|Moto|HUAWEI|LGE", regex=True
        )
        regex_rates["dev_info"] = float(dev_info_hits.mean())
    else:
        regex_rates["dev_info"] = 0.0

    return ColumnProfile(
        name=name, null_rate=null_rate, n_unique=n_unique, cardinality_ratio=card_ratio,
        is_numeric=is_num, is_bool_or_01=is_bool_or_01, is_int_like=is_int_like,
        is_monotonic=is_monotonic, p0=p0, p05=p05, p50=p50, p95=p95, p100=p100,
        mean=mean, frac_2dec=frac_2dec, top_values=top_values,
        all_small_values=all_small_values, regex_rates=regex_rates,
    )


def reference_profiles() -> dict[str, ColumnProfile]:
    """Compute and cache canonical column profiles from frames.tx_all()."""
    global _REF_PROFILES_CACHE
    if _REF_PROFILES_CACHE is None:
        ref_df = tx_all()
        idx = np.linspace(0, len(ref_df) - 1, 25000, dtype=int)
        sub = ref_df.iloc[idx]
        _REF_PROFILES_CACHE = {col: _compute_profile(sub[col], col) for col in CANONICAL_COLS if col in sub.columns}
    return _REF_PROFILES_CACHE


def _null_sim(r_null: float, f_null: float, tol: float = 0.25) -> float:
    diff = abs(r_null - f_null)
    return max(0.0, 1.0 - diff / tol)


def _is_transaction_dt_profile(fp: ColumnProfile) -> bool:
    """True if `fp` matches the distribution of `TransactionDT` (and NOT `TransactionID` or cents `amount`).
    In `tx_all()`, `TransactionID` is strictly in `[3_000_000 .. 3_670_000]`, whereas
    `TransactionDT` spans `[86_400 .. 18_490_000]` (`p100 >= 5_000_000`, `p50 >= 5_000_000`)."""
    return (
        fp.is_numeric
        and fp.is_int_like
        and fp.null_rate <= 0.01
        and 0.0 <= fp.p0 <= 2_500_000
        and fp.p50 >= 4_000_000
        and 8_000_000 <= fp.p100 <= 25_000_000
        and fp.cardinality_ratio > 0.5
    )


def _score_candidate(
    canon: str,
    fp: ColumnProfile,
    rp: ColumnProfile,
) -> tuple[float, str | None]:
    """Score a foreign ColumnProfile against a canonical SOURCE column profile.
    Returns (similarity_score_in_0_1, optional_transform_tag)."""
    ns = _null_sim(rp.null_rate, fp.null_rate, tol=0.25)

    # Never map a pipe-composed column (like device_profile or uid) directly onto
    # any raw identity component (DeviceInfo, id_30, id_31, id_33)
    if canon in ("DeviceInfo", "id_30", "id_31", "id_33") and fp.regex_rates.get("any_pipe", 0.0) > 0.01:
        return 0.0, None

    if canon == "TransactionID":
        if not fp.is_numeric or not fp.is_int_like or fp.null_rate > 0.01:
            return 0.0, None
        if 2_900_000 <= fp.p0 <= 3_700_000 and 3_000_000 <= fp.p100 <= 3_800_000 and fp.cardinality_ratio > 0.98:
            return 0.98, None
        return 0.0, None

    if canon == "ts":
        if fp.null_rate > 0.01:
            return 0.0, None
        if fp.regex_rates.get("iso_dt", 0.0) > 0.90 and fp.regex_rates.get("is_dt_dtype", 0.0) < 0.5:
            return 0.97, None
        if fp.regex_rates.get("is_dt_dtype", 0.0) > 0.5:
            return 0.95, "dt_to_ts"
        if fp.is_numeric and 1.46e9 <= fp.p0 <= 1.49e9 and 1.46e9 <= fp.p100 <= 1.49e9:
            return 0.92, "epoch_to_ts"
        if fp.is_numeric and 1.46e12 <= fp.p0 <= 1.49e12 and 1.46e12 <= fp.p100 <= 1.49e12:
            return 0.91, "epoch_ms_to_ts"
        return 0.0, None

    if canon == "card_id":
        r = fp.regex_rates.get("card_id", 0.0)
        return (0.98 * r, None) if r > 0.8 else (0.0, None)

    if canon == "customer_id":
        r = fp.regex_rates.get("cust_id", 0.0)
        return (0.98 * r, None) if r > 0.8 else (0.0, None)

    if canon == "ProductCD":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        if fp.all_small_values <= {"W", "C", "R", "H", "S"} and len(fp.all_small_values & {"W", "C", "R", "H", "S"}) >= 3:
            return 0.98 * ns, None
        return 0.0, None

    if canon == "channel":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        low_vals = {v.lower() for v in fp.all_small_values}
        if low_vals == {"in_person", "online"}:
            return 0.98 * ns, None
        return 0.0, None

    if canon == "card4":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        low_vals = {v.lower() for v in fp.all_small_values}
        target = {"visa", "mastercard", "american express", "discover"}
        if low_vals <= target and len(low_vals & {"visa", "mastercard"}) == 2:
            return 0.98 * ns, None
        return 0.0, None

    if canon == "card6":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        low_vals = {v.lower() for v in fp.all_small_values}
        target = {"debit", "credit", "debit or credit", "charge card"}
        if low_vals <= target and len(low_vals & {"debit", "credit"}) == 2:
            return 0.98 * ns, None
        return 0.0, None

    if canon == "DeviceType":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        low_vals = {v.lower() for v in fp.all_small_values}
        if low_vals <= {"desktop", "mobile"} and len(low_vals) >= 1:
            return 0.97 * _null_sim(rp.null_rate, fp.null_rate, 0.35), None
        return 0.0, None

    if canon == "id_15":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        if fp.all_small_values <= {"New", "Found", "Unknown"} and len(fp.all_small_values & {"New", "Found"}) >= 1:
            return 0.97 * _null_sim(rp.null_rate, fp.null_rate, 0.35), None
        return 0.0, None

    if canon == "id_23":
        r = fp.regex_rates.get("ip_proxy", 0.0)
        return (0.97 * r, None) if r > 0.5 else (0.0, None)

    if canon == "M4":
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        if fp.all_small_values <= {"M0", "M1", "M2"} and len(fp.all_small_values) >= 2:
            return 0.97 * _null_sim(rp.null_rate, fp.null_rate, 0.30), None
        return 0.0, None

    if canon in ("M5", "M6"):
        if fp.is_numeric or not fp.all_small_values:
            return 0.0, None
        if fp.all_small_values <= {"T", "F"} and len(fp.all_small_values) == 2:
            return 0.95 * _null_sim(rp.null_rate, fp.null_rate, 0.30), None
        return 0.0, None

    if canon in ("P_emaildomain", "R_emaildomain"):
        r = fp.regex_rates.get("email", 0.0)
        if r < 0.5:
            return 0.0, None
        return 0.96 * r * _null_sim(rp.null_rate, fp.null_rate, 0.40), None

    if canon == "id_30":
        r = fp.regex_rates.get("os", 0.0)
        return (0.96 * r, None) if r > 0.4 else (0.0, None)

    if canon == "id_31":
        r = fp.regex_rates.get("browser", 0.0)
        return (0.96 * r, None) if r > 0.4 else (0.0, None)

    if canon == "id_33":
        r = fp.regex_rates.get("res", 0.0)
        return (0.97 * r, None) if r > 0.5 else (0.0, None)

    if canon == "DeviceInfo":
        if fp.is_numeric or fp.regex_rates.get("any_pipe", 0.0) > 0.01:
            return 0.0, None
        r = fp.regex_rates.get("dev_info", 0.0)
        if r > 0.25 and fp.null_rate > 0.40:
            return 0.92 * _null_sim(rp.null_rate, fp.null_rate, 0.35), None
        return 0.0, None

    if canon == "risk_score":
        if not fp.is_numeric or fp.is_int_like or fp.null_rate > 0.05:
            return 0.0, None
        if 0.0 <= fp.p0 <= 0.05 and 0.50 <= fp.p100 <= 1.0001 and 0.03 <= fp.p50 <= 0.25:
            return 0.96, None
        return 0.0, None

    if canon == "amount":
        if not fp.is_numeric or fp.null_rate > 0.05 or fp.p0 <= 0:
            return 0.0, None
        # Direct dollars: float with cents decimals, median ~50-90, p95 ~300-600
        if not fp.is_int_like and fp.frac_2dec > 0.30 and 20.0 <= fp.p50 <= 150.0 and 150.0 <= fp.p95 <= 1200.0:
            return 0.96, None
        # Cents: integer-like, median ~5000-9000, p95 ~30000-60000
        if fp.is_int_like and 2000.0 <= fp.p50 <= 15000.0 and 15000.0 <= fp.p95 <= 120000.0 and fp.p100 < 2_500_000:
            return 0.94, "cents_to_usd"
        return 0.0, None

    if canon == "addr1":
        if not fp.is_numeric or not fp.is_int_like:
            return 0.0, None
        if 0.03 <= fp.null_rate <= 0.25 and 90.0 <= fp.p0 <= 120.0 and 480.0 <= fp.p100 <= 600.0 and 250.0 <= fp.p50 <= 340.0:
            return 0.93 * _null_sim(rp.null_rate, fp.null_rate, 0.15), None
        return 0.0, None

    if canon == "addr2":
        if not fp.is_numeric or not fp.is_int_like:
            return 0.0, None
        if 0.03 <= fp.null_rate <= 0.25 and 85.0 <= fp.p50 <= 89.0 and 85.0 <= fp.p05 <= 89.0 and fp.p100 <= 120.0:
            return 0.95 * _null_sim(rp.null_rate, fp.null_rate, 0.15), None
        return 0.0, None

    if canon == "dist1":
        if not fp.is_numeric:
            return 0.0, None
        if 0.40 <= fp.null_rate <= 0.80 and fp.p0 == 0.0 and 3.0 <= fp.p50 <= 25.0 and fp.p100 >= 500.0:
            return 0.92 * _null_sim(rp.null_rate, fp.null_rate, 0.25), None
        return 0.0, None

    if canon == "D1":
        if not fp.is_numeric or not fp.is_int_like:
            return 0.0, None
        if fp.null_rate <= 0.05 and fp.p0 == 0.0 and 0.0 <= fp.p50 <= 30.0 and 500.0 <= fp.p100 <= 750.0:
            return 0.93, None
        return 0.0, None

    if canon == "C1":
        if not fp.is_numeric or not fp.is_int_like:
            return 0.0, None
        if fp.null_rate <= 0.02 and fp.p0 == 0.0 and 1.0 <= fp.p50 <= 3.0 and fp.p100 >= 800.0:
            return 0.93, None
        return 0.0, None

    return 0.0, None


def _join_foreign_tables(
    foreign: pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """If multiple tables are provided, find the main transaction table and left-join
    auxiliary (e.g. identity) tables by high-overlap key columns."""
    if isinstance(foreign, pd.DataFrame):
        col_origins = {c: "main" for c in foreign.columns}
        return foreign.copy(), col_origins, {"mode": "single_table"}

    if isinstance(foreign, list):
        tables = {f"table_{i}": df for i, df in enumerate(foreign)}
    else:
        tables = dict(foreign)

    main_name = max(tables, key=lambda k: len(tables[k]))
    main_df = tables[main_name].copy()
    col_origins = {c: main_name for c in main_df.columns}
    join_info: dict[str, Any] = {"mode": "multi_table", "main_table": main_name, "joins": []}

    main_keys: dict[str, set[Any]] = {}
    for c in main_df.columns:
        s = main_df[c].dropna()
        if len(s) >= 0.95 * len(main_df) and s.nunique() >= 0.90 * len(s):
            main_keys[c] = set(s.iloc[:25000])

    for tname, tdf in tables.items():
        if tname == main_name:
            continue
        best_pair: tuple[str, str] | None = None
        best_overlap = 0.0
        for tc in tdf.columns:
            ts = tdf[tc].dropna()
            if len(ts) == 0 or ts.nunique() < 0.80 * len(ts):
                continue
            t_set = set(ts.iloc[:25000])
            for mc, m_set in main_keys.items():
                if main_df[mc].dtype.kind != tdf[tc].dtype.kind:
                    continue
                ov = len(t_set & m_set) / len(t_set)
                if ov > best_overlap:
                    best_overlap = ov
                    best_pair = (mc, tc)

        if best_pair is not None and best_overlap >= 0.50:
            mc, tc = best_pair
            dedup_tdf = tdf.drop_duplicates(subset=[tc])
            rename_map = {}
            for c in dedup_tdf.columns:
                if c == tc:
                    continue
                new_c = c if c not in main_df.columns else f"{tname}__{c}"
                rename_map[c] = new_c
                col_origins[new_c] = tname
            right_sub = dedup_tdf[[tc] + [c for c in dedup_tdf.columns if c != tc]].rename(columns=rename_map)
            main_df = main_df.merge(right_sub, left_on=mc, right_on=tc, how="left")
            if tc != mc and tc in main_df.columns:
                main_df = main_df.drop(columns=[tc])
            join_info["joins"].append({
                "right_table": tname,
                "left_key": mc,
                "right_key": tc,
                "overlap": round(best_overlap, 4),
            })

    return main_df, col_origins, join_info


@dataclass
class _SyntheticCandidate:
    expr: str
    series: pd.Series
    profile: ColumnProfile
    source_cols: list[str]
    target_hint: str | None = None
    bonus_conf: float = 0.0


def _discover_composite_candidates(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
) -> list[_SyntheticCandidate]:
    """Detect split/merged columns (card_id split, date+time split, device_profile packed,
    uid packed) and one-hot encodings in foreign df."""
    cands: list[_SyntheticCandidate] = []

    # 1. Split card_id: a customer_id column (C\d{5}) + a card suffix column (K\d or -K\d)
    cust_cols = [c for c, p in profiles.items() if p.regex_rates.get("cust_id", 0.0) > 0.8]
    sfx_cols = [c for c, p in profiles.items() if p.regex_rates.get("card_suffix", 0.0) > 0.8]
    for cc in cust_cols:
        for sc in sfx_cols:
            sfx_clean = df[sc].astype(str).str.lstrip("-")
            combined = df[cc].astype(str) + "-" + sfx_clean
            expr = f"concat_card_id({cc},{sc})"
            cands.append(_SyntheticCandidate(
                expr=expr,
                series=combined,
                profile=_compute_profile(combined, expr),
                source_cols=[cc, sc],
                target_hint="card_id",
                bonus_conf=0.97,
            ))

    # 2. Split card_id into customer_id when card_id is present but customer_id is not
    card_cols = [c for c, p in profiles.items() if p.regex_rates.get("card_id", 0.0) > 0.8]
    if card_cols and not cust_cols:
        for kc in card_cols:
            extracted_cust = df[kc].astype(str).str.split("-").str[0]
            expr = f"split_card_cust({kc})"
            cands.append(_SyntheticCandidate(
                expr=expr,
                series=extracted_cust,
                profile=_compute_profile(extracted_cust, expr),
                source_cols=[kc],
                target_hint="customer_id",
                bonus_conf=0.96,
            ))

    # 3. Date + Time split into two columns (e.g. "2016-08-14" and "16:47:09")
    date_cols = [c for c, p in profiles.items() if p.regex_rates.get("date_only", 0.0) > 0.8]
    time_cols = [c for c, p in profiles.items() if p.regex_rates.get("time_only", 0.0) > 0.8]
    for dc in date_cols:
        for tc in time_cols:
            ts_combined = df[dc].astype(str).str.strip() + " " + df[tc].astype(str).str.strip()
            expr_ts = f"concat_date_time({dc},{tc})"
            prof_ts = _compute_profile(ts_combined, expr_ts)
            cands.append(_SyntheticCandidate(
                expr=expr_ts,
                series=ts_combined,
                profile=prof_ts,
                source_cols=[dc, tc],
                target_hint="ts",
                bonus_conf=0.96,
            ))

    # 4. Unpack device_profile ("DeviceInfo | id_30 | id_31 | id_33") into constituent SOURCE parts
    dev_prof_cols = [c for c, p in profiles.items() if p.regex_rates.get("dev_prof", 0.0) > 0.10]
    for dpc in dev_prof_cols:
        s_str = df[dpc].fillna("").astype(str)
        parts = s_str.str.split("|", expand=True)
        if parts.shape[1] >= 4:
            part_targets = [(0, "DeviceInfo"), (1, "id_30"), (2, "id_31"), (3, "id_33")]
            for idx, tcol in part_targets:
                part_s = parts[idx].str.strip().replace({"?": None, "": None, "nan": None, "None": None})
                expr = f"split_pipe({dpc},{idx})"
                cands.append(_SyntheticCandidate(
                    expr=expr,
                    series=part_s,
                    profile=_compute_profile(part_s, expr),
                    source_cols=[dpc],
                    target_hint=tcol,
                ))

    # 5. Unpack uid ("card_id|addr1|day_minus_D1") if card_id is missing
    uid_cols = [c for c, p in profiles.items() if p.regex_rates.get("uid", 0.0) > 0.80]
    for uc in uid_cols:
        parts = df[uc].fillna("").astype(str).str.split("|", expand=True)
        if parts.shape[1] >= 3 and not card_cols and not (cust_cols and sfx_cols):
            c_s = parts[0].str.strip()
            expr = f"split_uid_card({uc})"
            cands.append(_SyntheticCandidate(
                expr=expr,
                series=c_s,
                profile=_compute_profile(c_s, expr),
                source_cols=[uc],
                target_hint="card_id",
                bonus_conf=0.94,
            ))

    # 6. One-hot channel columns: pair of binary {0,1}/bool columns summing to 1 with ~0.74 / ~0.26 means
    bool_cols = [
        c for c, p in profiles.items()
        if p.is_bool_or_01 and p.null_rate < 0.01 and 0.05 <= p.mean <= 0.95
    ]
    for i in range(len(bool_cols)):
        for j in range(i + 1, len(bool_cols)):
            c1, c2 = bool_cols[i], bool_cols[j]
            v1 = df[c1].astype(float)
            v2 = df[c2].astype(float)
            if np.mean(np.abs((v1 + v2) - 1.0) < 1e-6) > 0.99:
                in_p_col, onl_col = (c1, c2) if v1.mean() >= v2.mean() else (c2, c1)
                ch_series = pd.Series(np.where(df[onl_col].astype(bool), "online", "in_person"), index=df.index)
                expr = f"onehot_channel(in_person={in_p_col},online={onl_col})"
                cands.append(_SyntheticCandidate(
                    expr=expr,
                    series=ch_series,
                    profile=_compute_profile(ch_series, expr),
                    source_cols=[in_p_col, onl_col],
                    target_hint="channel",
                    bonus_conf=0.96,
                ))

    return cands


def _sample_justification_ids(
    df: pd.DataFrame,
    series: pd.Series,
    txn_id_series: pd.Series | None,
    k: int = 5,
) -> list[int]:
    """Select up to k deterministic non-null record IDs that exemplify a column's values."""
    valid_mask = series.notna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        valid_mask = valid_mask & (~series.astype(str).isin(["", "nan", "None", "null"]))
    valid_indices = df.index[valid_mask][:k]
    if len(valid_indices) == 0:
        valid_indices = df.index[:k]
    if txn_id_series is not None:
        return [int(txn_id_series.loc[i]) for i in valid_indices]
    return [int(i) for i in valid_indices]


def _identify_redundant_derived_foreign_cols(
    raw_profiles: dict[str, ColumnProfile],
    used_raw_cols: set[str],
) -> set[str]:
    """Identify foreign columns that are pre-existing copies of our DERIVED_COLS
    so they are not misreported as unmapped foreign decoys."""
    redundant: set[str] = set()
    for col, fp in raw_profiles.items():
        if col in used_raw_cols:
            continue
        if fp.regex_rates.get("uid", 0.0) > 0.80:
            redundant.add(col)
        elif fp.regex_rates.get("dev_prof", 0.0) > 0.10:
            redundant.add(col)
        elif not fp.is_numeric and fp.all_small_values and fp.all_small_values <= {"r1", "jan"}:
            redundant.add(col)
        elif fp.regex_rates.get("is_dt_dtype", 0.0) > 0.5 or fp.regex_rates.get("iso_dt", 0.0) > 0.90:
            redundant.add(col)
        elif _is_transaction_dt_profile(fp):
            redundant.add(col)
    return redundant


def infer_mapping(
    foreign: pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame],
    approval_threshold: float = APPROVAL_THRESHOLD,
    confidence_floor: float = CONFIDENCE_FLOOR,
    proposal_dir: Path | None = None,
    allow_txid_lookup: bool = True,
) -> SchemaMapping:
    """Infer mapping from foreign DataFrame(s) to our canonical schema.

    1. Matches the 27 SOURCE_COLS against foreign columns (and composite candidates)
       via distributional/regex fingerprints and Hungarian assignment.
    2. Refuses to guess any SOURCE column whose best match is below `confidence_floor`.
    3. Schedules the 5 DERIVED_COLS (`t`, `TransactionDT`, `month_src`, `device_profile`,
       `uid`) using the exact `assay.r2.frames.stage()` code path:
       - If the variant carries a `TransactionDT` column, maps it directly.
       - Otherwise, if `TransactionID` is mapped and `allow_txid_lookup=True`, joins
         `TransactionDT` back through `TransactionID` against `frames.tx_all()`.
       - If neither is possible (e.g., unknown/anonymized TransactionIDs and no
         TransactionDT column), marks `TransactionDT` (and `uid`) in the proposal as
         `"not recoverable from this variant"` (honest gap) rather than guessing.
    4. Persists a proposal JSON file to disk and returns a `SchemaMapping` with
       attested sample record IDs on every choice.
    """
    joined_df, col_origins, join_info = _join_foreign_tables(foreign)
    ref_profs = reference_profiles()

    raw_profiles: dict[str, ColumnProfile] = {
        col: _compute_profile(joined_df[col], col) for col in joined_df.columns
    }
    synth_cands = _discover_composite_candidates(joined_df, raw_profiles)

    cand_names: list[str] = list(joined_df.columns) + [sc.expr for sc in synth_cands]
    cand_profiles: dict[str, ColumnProfile] = dict(raw_profiles)
    cand_series: dict[str, pd.Series] = {c: joined_df[c] for c in joined_df.columns}
    cand_sources: dict[str, list[str]] = {c: [c] for c in joined_df.columns}
    for sc in synth_cands:
        cand_profiles[sc.expr] = sc.profile
        cand_series[sc.expr] = sc.series
        cand_sources[sc.expr] = sc.source_cols

    # Match strictly over SOURCE_COLS first
    source_list = list(SOURCE_COLS)
    n_source = len(source_list)
    n_cand = len(cand_names)

    sim_matrix = np.zeros((n_source, n_cand), dtype=float)
    transform_matrix: dict[tuple[int, int], str | None] = {}

    for i, c_col in enumerate(source_list):
        rp = ref_profs[c_col]
        for j, f_name in enumerate(cand_names):
            fp = cand_profiles[f_name]
            score, tfm = _score_candidate(c_col, fp, rp)
            synth_match = next((sc for sc in synth_cands if sc.expr == f_name), None)
            if synth_match is not None:
                if synth_match.target_hint is not None and synth_match.target_hint != c_col:
                    score = 0.0
                elif synth_match.bonus_conf > 0 and score > 0:
                    score = max(score, synth_match.bonus_conf)
                if f_name.startswith("split_pipe("):
                    score *= 0.91
            sim_matrix[i, j] = score
            transform_matrix[(i, j)] = tfm

    # Hungarian matching on cost = 1.0 - similarity over SOURCE_COLS
    row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix)

    prelim_assigned: dict[str, tuple[str, float, str | None, bool]] = {}
    refused_below_floor: dict[str, tuple[str, float]] = {}
    not_recoverable: dict[str, str] = {}
    used_raw_cols: set[str] = set()

    matched_pairs = sorted(zip(row_ind, col_ind), key=lambda pair: sim_matrix[pair[0], pair[1]], reverse=True)
    for r_idx, c_idx in matched_pairs:
        score = float(sim_matrix[r_idx, c_idx])
        c_col = source_list[r_idx]
        f_name = cand_names[c_idx]
        if score <= 0.0:
            continue
        if score < confidence_floor:
            refused_below_floor[c_col] = (f_name, round(score, 4))
            continue
        tfm = transform_matrix[(r_idx, c_idx)]
        prelim_assigned[c_col] = (f_name, score, tfm, False)
        for src in cand_sources[f_name]:
            used_raw_cols.add(src)

    # Fallback if customer_id was not a separate column in foreign data but card_id was mapped
    if "customer_id" not in prelim_assigned and "card_id" in prelim_assigned:
        c_expr = prelim_assigned["card_id"][0]
        if 0.95 >= confidence_floor:
            prelim_assigned["customer_id"] = (f"split_card_cust({c_expr})", 0.95, None, False)

    # Locate TransactionID series for sample record justification IDs and TransactionDT join-back
    txn_id_series: pd.Series | None = None
    if "TransactionID" in prelim_assigned:
        tid_col = prelim_assigned["TransactionID"][0]
        if tid_col in cand_series:
            txn_id_series = cand_series[tid_col]

    # Register the 5 DERIVED_COLS (`t`, `TransactionDT`, `month_src`, `device_profile`, `uid`)
    if "ts" in prelim_assigned:
        ts_conf = prelim_assigned["ts"][1]
        prelim_assigned["t"] = ("pd.to_datetime(ts)", round(ts_conf, 4), None, True)
        prelim_assigned["month_src"] = ("frames.stage.month_src(t)", round(ts_conf, 4), None, True)

    # Check if the variant directly carries a TransactionDT column (not yet used by SOURCE_COLS)
    direct_dt_cols = [
        col for col, fp in raw_profiles.items()
        if col not in used_raw_cols and _is_transaction_dt_profile(fp)
    ]
    txdt_recoverable = False
    if direct_dt_cols:
        dt_col = direct_dt_cols[0]
        used_raw_cols.add(dt_col)
        prelim_assigned["TransactionDT"] = (dt_col, 0.96, None, True)
        txdt_recoverable = True
    elif allow_txid_lookup and txn_id_series is not None:
        lookup = _canonical_tx_lookup()
        hit_rate = float(txn_id_series.iloc[:1000].isin(lookup.index).mean())
        if hit_rate >= 0.95:
            tid_col = prelim_assigned["TransactionID"][0]
            prelim_assigned["TransactionDT"] = (
                f"join_tx_all_TransactionDT({tid_col})",
                round(hit_rate * prelim_assigned["TransactionID"][1], 4),
                None,
                True,
            )
            txdt_recoverable = True

    if not txdt_recoverable:
        not_recoverable["TransactionDT"] = NOT_RECOVERABLE
        prelim_assigned["TransactionDT"] = (NOT_RECOVERABLE, 0.0, None, True)

    dev_parts = ["DeviceInfo", "id_30", "id_31", "id_33"]
    if all(p in prelim_assigned for p in dev_parts):
        dev_conf = min(prelim_assigned[p][1] for p in dev_parts)
        prelim_assigned["device_profile"] = (
            "assay.store.device_profile(DeviceInfo,id_30,id_31,id_33)",
            round(dev_conf, 4),
            None,
            True,
        )

    uid_src_parts = ["card_id", "addr1", "D1"]
    if txdt_recoverable and all(p in prelim_assigned for p in uid_src_parts):
        uid_conf = min(
            [prelim_assigned[p][1] for p in uid_src_parts] + [prelim_assigned["TransactionDT"][1]]
        )
        prelim_assigned["uid"] = (
            "frames.stage.uid(card_id,addr1,TransactionDT,D1)",
            round(uid_conf, 4),
            None,
            True,
        )
    else:
        not_recoverable["uid"] = NOT_RECOVERABLE
        prelim_assigned["uid"] = (NOT_RECOVERABLE, 0.0, None, True)

    choices: dict[str, MappingChoice] = {}
    for c_col in CANONICAL_COLS:
        if c_col not in prelim_assigned:
            continue
        f_name, conf, tfm, is_der = prelim_assigned[c_col]
        if tfm == "cents_to_usd":
            full_expr = f"{f_name} / 100.0"
        elif tfm in ("epoch_to_ts", "epoch_ms_to_ts", "dt_to_ts"):
            full_expr = f"{tfm}({f_name})"
        else:
            full_expr = f_name

        s_for_ids = cand_series.get(f_name, joined_df.iloc[:, 0])
        sample_ids = _sample_justification_ids(joined_df, s_for_ids, txn_id_series, k=5)
        is_unrec = full_expr == NOT_RECOVERABLE
        needs_app = (conf < approval_threshold) and not is_unrec

        kind_label = "derived" if is_der else "source"
        att = Attestation(
            what=(
                f"Column '{c_col}' is {NOT_RECOVERABLE}."
                if is_unrec
                else f"Mapped {kind_label} column '{c_col}' <- '{full_expr}' with confidence {conf:.2f}."
            ),
            evidence=Evidence(txn_ids=sample_ids),
            query="assay.r2.schema_map.infer_mapping",
            params={
                "canonical_col": c_col,
                "their_expr": full_expr,
                "confidence": round(conf, 4),
                "is_derived": is_der,
            },
        )
        choices[c_col] = MappingChoice(
            canonical_col=c_col,
            expr=full_expr,
            confidence=round(conf, 4),
            sample_ids=sample_ids,
            attestation=att,
            is_derived=is_der,
            needs_approval=needs_app,
            approved=(not needs_app) and (not is_unrec),
            notes=NOT_RECOVERABLE if is_unrec else "",
        )

    redundant_derived = _identify_redundant_derived_foreign_cols(raw_profiles, used_raw_cols)
    unmapped = [c for c in joined_df.columns if c not in used_raw_cols and c not in redundant_derived]

    sig_str = "|".join(f"{k}:{v.expr}:{v.confidence}" for k, v in sorted(choices.items()))
    prop_id = "SMP-" + hashlib.sha256(sig_str.encode()).hexdigest()[:10]

    mapping = SchemaMapping(
        choices=choices,
        join_info=join_info,
        unmapped_foreign=unmapped,
        refused_below_floor=refused_below_floor,
        not_recoverable=not_recoverable,
        proposal_id=prop_id,
        proposal_approved=not any(c.needs_approval for c in choices.values()),
    )
    out_dir = proposal_dir if proposal_dir is not None else PROPOSALS_DIR
    mapping.save_proposal(out_dir / f"{prop_id}.json")
    _PROPOSAL_REGISTRY[prop_id] = mapping
    return mapping


def _eval_source_expr(
    expr: str,
    df: pd.DataFrame,
    out_so_far: dict[str, pd.Series],
) -> pd.Series:
    """Evaluate a SOURCE column mapping expression against the joined foreign DataFrame."""
    if expr in df.columns:
        return df[expr].copy()

    if expr.endswith(" / 100.0"):
        col = expr[: -len(" / 100.0")]
        return df[col].astype(float) / 100.0

    m = re.match(r"^(\w+)\((.+)\)$", expr)
    if not m:
        raise ValueError(f"Unrecognized mapping expression: {expr}")
    fn, arg_str = m.group(1), m.group(2)

    if fn == "concat_card_id":
        c1, c2 = [x.strip() for x in arg_str.split(",", 1)]
        return df[c1].astype(str) + "-" + df[c2].astype(str).str.lstrip("-")

    if fn == "concat_date_time":
        c1, c2 = [x.strip() for x in arg_str.split(",", 1)]
        return df[c1].astype(str).str.strip() + " " + df[c2].astype(str).str.strip()

    if fn == "split_card_cust":
        if arg_str in df.columns:
            base = df[arg_str].astype(str)
        elif "card_id" in out_so_far:
            base = out_so_far["card_id"].astype(str)
        else:
            base = _eval_source_expr(arg_str, df, out_so_far).astype(str)
        return base.str.split("-").str[0]

    if fn == "split_pipe":
        col, idx_s = [x.strip() for x in arg_str.rsplit(",", 1)]
        idx = int(idx_s)
        parts = df[col].fillna("").astype(str).str.split("|", expand=True)
        return parts[idx].str.strip().replace({"?": None, "": None, "nan": None, "None": None})

    if fn == "split_uid_card":
        return df[arg_str].fillna("").astype(str).str.split("|").str[0]

    if fn == "onehot_channel":
        kv = dict(item.split("=", 1) for item in arg_str.split(","))
        onl_col = kv["online"]
        return pd.Series(np.where(df[onl_col].astype(bool), "online", "in_person"), index=df.index)

    if fn == "dt_to_ts":
        s = df[arg_str] if arg_str in df.columns else _eval_source_expr(arg_str, df, out_so_far)
        return pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M:%S")

    if fn in ("epoch_to_ts", "epoch_ms_to_ts"):
        unit = "ms" if "ms" in fn else "s"
        dt = pd.to_datetime(df[arg_str].astype("int64"), unit=unit).dt.tz_localize(None)
        return dt.dt.strftime("%Y-%m-%d %H:%M:%S")

    raise ValueError(f"Unsupported source mapping function: {fn}")


def apply(
    mapping: SchemaMapping,
    foreign: pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame],
    strict_approval: bool = True,
    require_proposal_approval: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Apply an inferred SchemaMapping to foreign data:
    1. Evaluates all approved SOURCE_COLS expressions.
    2. Recomputes all recoverable DERIVED_COLS (`t`, `TransactionDT`, `month_src`,
       `device_profile`, `uid`) using `recompute_derived_columns()` (which shares
       the exact `frames.stage()` code path and `assay.store.device_profile`).
    """
    if require_proposal_approval and not mapping.proposal_approved:
        raise PermissionError(
            f"Proposal '{mapping.proposal_id}' requires human approval via "
            f"approve('{mapping.proposal_id}', approver_role) before apply()."
        )

    joined_df, _, _ = _join_foreign_tables(foreign)
    ref_profs = reference_profiles()

    out_cols: dict[str, pd.Series] = {}
    recovered: list[str] = []
    derived: list[str] = []
    flagged: list[str] = []
    skipped: list[str] = []
    null_deltas: dict[str, float] = {}
    domain_warnings: dict[str, str] = {}

    # 1. Evaluate SOURCE_COLS (card_id before customer_id in case customer_id is split from card_id)
    source_order = [c for c in SOURCE_COLS if c != "customer_id"] + ["customer_id"]
    for c_col in source_order:
        if c_col not in mapping.choices:
            continue
        choice = mapping.choices[c_col]
        if choice.expr == NOT_RECOVERABLE:
            continue
        if choice.needs_approval:
            flagged.append(c_col)
            if strict_approval and not choice.approved:
                skipped.append(c_col)
                continue

        s = _eval_source_expr(choice.expr, joined_df, out_cols)
        out_cols[c_col] = s
        recovered.append(c_col)

        f_null = float(s.isna().mean())
        null_deltas[c_col] = round(f_null - ref_profs[c_col].null_rate, 4)
        if abs(null_deltas[c_col]) > 0.25:
            domain_warnings[c_col] = f"Null rate delta {null_deltas[c_col]:+.2f} vs canonical"

    # 2. If TransactionDT was directly mapped from a foreign column in the variant, copy it in first
    allow_lookup = False
    if "TransactionDT" in mapping.choices:
        dt_choice = mapping.choices["TransactionDT"]
        if dt_choice.expr in joined_df.columns:
            if not (dt_choice.needs_approval and strict_approval and not dt_choice.approved):
                out_cols["TransactionDT"] = joined_df[dt_choice.expr].astype("int64")
        elif dt_choice.expr.startswith("join_tx_all_TransactionDT("):
            allow_lookup = True

    # 3. Recompute DERIVED_COLS from the recovered SOURCE_COLS using shared recompute_derived_columns()
    source_df = pd.DataFrame(out_cols, index=joined_df.index)
    recomputed_df = recompute_derived_columns(source_df, allow_txid_lookup=allow_lookup)
    # Sync sanitized raw identity columns back into out_cols
    for id_c in ("DeviceInfo", "id_30", "id_31", "id_33"):
        if id_c in recomputed_df.columns:
            out_cols[id_c] = recomputed_df[id_c]

    for d_col in DERIVED_COLS:
        if d_col not in mapping.choices:
            continue
        choice = mapping.choices[d_col]
        if choice.expr == NOT_RECOVERABLE:
            continue
        if choice.needs_approval:
            flagged.append(d_col)
            if strict_approval and not choice.approved:
                skipped.append(d_col)
                continue
        if d_col in recomputed_df.columns:
            out_cols[d_col] = recomputed_df[d_col]
            derived.append(d_col)
            f_null = float(recomputed_df[d_col].isna().mean())
            null_deltas[d_col] = round(f_null - ref_profs[d_col].null_rate, 4)

    canonical_df = pd.DataFrame({c: out_cols[c] for c in CANONICAL_COLS if c in out_cols}, index=joined_df.index)
    missing = [c for c in CANONICAL_COLS if c not in canonical_df.columns]

    report = ValidationReport(
        proposal_id=mapping.proposal_id,
        recovered_cols=recovered,
        derived_cols=derived,
        not_recoverable_cols=dict(mapping.not_recoverable),
        missing_cols=missing,
        flagged_low_confidence=flagged,
        skipped_unapproved=skipped,
        unmapped_foreign_cols=list(mapping.unmapped_foreign),
        refused_below_floor=dict(mapping.refused_below_floor),
        null_rate_deltas=null_deltas,
        domain_warnings=domain_warnings,
        n_rows=len(canonical_df),
    )
    return canonical_df, report
