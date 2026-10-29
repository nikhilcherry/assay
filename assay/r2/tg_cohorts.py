"""C6 -- TigerGraph GSQL cohort detectors (`r2_micro_drip_cohort` and `r2_recipient_fanin`).

Mirrors the pure functions in:
  - `assay.r2.detectors.micro_charge.cohort_members`
  - `assay.r2.detectors.fan_in.cohort_members`
on the live TigerGraph instance (`assay` graph, CE 4.2, GSQL V2).

Schema note:
  In `graph/schema.gsql` and `graph/load_r2.gsql`, `R_emaildomain` is stored directly on
  the `Txn` vertex as attribute `r_email: STRING` (whereas `EmailDomain` vertices only
  index `p_email` via `PURCHASER_EMAIL`). `r2_recipient_fanin` therefore queries `t.r_email`
  directly on `Txn` vertices traversed from `Card -(MADE>)- Txn`.

Fail-closed contract:
  Every query installation, smoke run, or query invocation raises `TigerGraphUnavailableError`
  if TigerGraph is unreachable, returns a non-200 HTTP code, returns a GSQL error, or returns
  a malformed payload missing required keys.
"""

from __future__ import annotations

import re

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from assay.data import ROOT
from assay.r2.detectors.fan_in import (
    FREEMAIL_PREFIXES,
    cohort_members as fan_in_cohort_local,
    freemail_domains_by_base_rate,
)
from assay.r2.detectors.micro_charge import cohort_members as micro_drip_cohort_local
from assay.r2.frames import case_pack, tx_all
from assay.r2.tg_r2 import (
    _graph_name_from_schema,
    _installed_query_names,
    _strip_drops_and_installs,
    require_tg,
    run_gsql_script,
    run_query,
)
from assay.r2.vector import TigerGraphUnavailableError

COHORT_QUERIES: tuple[str, ...] = (
    "r2_micro_drip_cohort",
    "r2_recipient_fanin",
)

DEFAULT_FREEMAIL_SET: frozenset[str] = frozenset(
    {
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "aol.com",
        "anonymous.com",
        "icloud.com",
        "live.com",
        "msn.com",
        "me.com",
        "ymail.com",
        "mail.com",
        "gmx.de",
        "gmx.com",
        "protonmail.com",
        "rocketmail.com",
        *FREEMAIL_PREFIXES,
    }
)


def _extract_row(results: list[dict[str, Any]], query_name: str, required_keys: tuple[str, ...]) -> dict[str, Any]:
    """Merge GSQL PRINT result blocks and validate required keys (fail-closed on malformed reply)."""
    if not isinstance(results, list) or not results:
        raise TigerGraphUnavailableError(
            f"Fail-closed: TigerGraph query '{query_name}' returned empty/malformed results: {results!r}"
        )
    merged: dict[str, Any] = {}
    for item in results:
        if not isinstance(item, dict):
            raise TigerGraphUnavailableError(
                f"Fail-closed: TigerGraph query '{query_name}' returned non-dict element: {item!r}"
            )
        merged.update(item)
    missing = [k for k in required_keys if k not in merged]
    if missing:
        raise TigerGraphUnavailableError(
            f"Fail-closed: TigerGraph query '{query_name}' reply missing keys {missing}; got {list(merged.keys())}"
        )
    return merged


def smoke_test_cohort_queries(
    *,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, float]:
    """Smoke-run both cohort GSQL queries over REST++ and fail loudly if either errors or
    returns a malformed payload (guards against the `INSTALL QUERY` silent non-install gotcha).
    """
    require_tg(host=host, port=port)
    timings_ms: dict[str, float] = {}

    # 1. Smoke-run r2_micro_drip_cohort on HHF-001
    t0 = time.perf_counter()
    res_md = run_query(
        "r2_micro_drip_cohort",
        {
            "flagged_txn_id": "3660108",
            "opened_at": "2017-01-15 01:25:19",
            "max_amt": 6.50,
            "min_charges": 4,
            "min_span_days": 7.0,
            "window_days": 30,
            "quiet_days": 60,
            "min_other_cards": 6,
        },
        host=host,
        port=port,
    )
    timings_ms["r2_micro_drip_cohort"] = round((time.perf_counter() - t0) * 1000.0, 2)
    row_md = _extract_row(
        res_md,
        "r2_micro_drip_cohort",
        ("fired", "member_cards", "per_card_txns", "n_cohort_cards", "n_other_cards"),
    )
    print(
        f"[SMOKE] r2_micro_drip_cohort ({timings_ms['r2_micro_drip_cohort']}ms) -> "
        f"fired={row_md['fired']}, n_cards={len(row_md['member_cards'])}"
    )

    # 2. Smoke-run r2_recipient_fanin on HHF-004
    t1 = time.perf_counter()
    res_fi = run_query(
        "r2_recipient_fanin",
        {
            "flagged_txn_id": "3660260",
            "opened_at": "2017-01-26 21:09:37",
            "freemail": sorted(DEFAULT_FREEMAIL_SET),
            "window_days": 21,
            "min_amount": 100.0,
            "min_k_now": 5,
            "burst_mult": 3.0,
            "max_p": 1e-4,
        },
        host=host,
        port=port,
    )
    timings_ms["r2_recipient_fanin"] = round((time.perf_counter() - t1) * 1000.0, 2)
    row_fi = _extract_row(
        res_fi,
        "r2_recipient_fanin",
        ("fired", "member_cards", "per_card_txns", "k_now", "R_emaildomain"),
    )
    print(
        f"[SMOKE] r2_recipient_fanin ({timings_ms['r2_recipient_fanin']}ms) -> "
        f"fired={row_fi['fired']}, dom={row_fi['R_emaildomain']!r}, n_cards={len(row_fi['member_cards'])}"
    )

    return timings_ms


def install(conn: Any = None, *, force: bool = False) -> dict[str, Any]:
    """Create and install `r2_micro_drip_cohort` and `r2_recipient_fanin` from
    `graph/queries/r2_cohorts.gsql`, then smoke-run each query once over REST++ and fail
    loudly if either is not installed or errors.

    `conn` may be None, a `(host, port)` tuple, or a dict with optional `host`/`port` keys.
    Never drops or alters existing schema or data.
    """
    host: str | None = None
    port: int | None = None
    if isinstance(conn, tuple) and len(conn) == 2:
        host, port = str(conn[0]), int(conn[1])
    elif isinstance(conn, dict):
        host = conn.get("host")
        port = int(conn["port"]) if conn.get("port") is not None else None

    require_tg(host=host, port=port)
    gname = _graph_name_from_schema()

    gsql_path = ROOT / "graph" / "queries" / "r2_cohorts.gsql"
    gsql_text = _strip_drops_and_installs(gsql_path.read_text())

    # Always apply CREATE OR REPLACE QUERY so the latest query definitions are active
    create_out = run_gsql_script(gsql_text, label="create_r2_cohorts")
    # GSQL warnings mention "an error margin", so match real failures, not the word "error"
    created = all(re.search(rf"Successfully created queries: \[[^\]]*\b{q}\b", create_out) for q in COHORT_QUERIES)
    if not created or re.search(r"syntax error|semantic error|saved as draft", create_out, re.I):
        raise TigerGraphUnavailableError(f"Fail-closed: Failed to CREATE r2_cohorts.gsql:\n{create_out}")

    installed_before = _installed_query_names(gname)
    to_install = list(COHORT_QUERIES) if force else [q for q in COHORT_QUERIES if q not in installed_before]
    # Even if already listed in endpoints, if we just ran CREATE OR REPLACE QUERY we must INSTALL QUERY
    # so the compiled REST++ binary reflects the current GSQL definition.
    q_csv = ", ".join(COHORT_QUERIES)
    inst_out = run_gsql_script(
        f"USE GRAPH {gname}\nINSTALL QUERY {q_csv}\n",
        label=f"install_r2_cohorts({q_csv})",
        timeout=480.0,
    )
    if re.search(r"installation failed|syntax error|semantic error|is DRAFT|exited with code [1-9]", inst_out, re.I):
        raise TigerGraphUnavailableError(f"Fail-closed: INSTALL QUERY {q_csv} failed:\n{inst_out}")

    installed_after = _installed_query_names(gname)
    still_missing = [q for q in COHORT_QUERIES if q not in installed_after]
    if still_missing:
        raise TigerGraphUnavailableError(
            f"Fail-closed: Queries missing from REST++ endpoints after INSTALL: {still_missing}"
        )

    # Mandatory smoke-run of each query to guard against silent non-install
    timings = smoke_test_cohort_queries(host=host, port=port)
    return {
        "graph": gname,
        "installed_queries": list(COHORT_QUERIES),
        "previously_missing": to_install,
        "smoke_test_ms": timings,
    }


def micro_drip_cohort_tg(
    flagged_txn_id: int,
    opened_at: pd.Timestamp | str,
    max_amt: float = 6.50,
    min_charges: int = 4,
    min_span_days: float = 7.0,
    window_days: int = 30,
    quiet_days: int = 60,
    min_other_cards: int = 6,
    *,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> tuple[list[str], dict[str, list[int]], dict[str, Any]]:
    """Run `r2_micro_drip_cohort` on TigerGraph and return `(member_card_ids, per_card_txn_ids, params)`
    matching `assay.r2.detectors.micro_charge.cohort_members`.
    Fails closed (`TigerGraphUnavailableError`) on any TG error or malformed reply.
    """
    require_tg(host=host, port=port)
    t_open = pd.Timestamp(opened_at)
    as_of_str = t_open.strftime("%Y-%m-%d %H:%M:%S")

    params: dict[str, Any] = {
        "flagged_txn_id": int(flagged_txn_id),
        "opened_at": str(t_open),
        "max_amt": float(max_amt),
        "min_charges": int(min_charges),
        "min_span_days": float(min_span_days),
        "window_days": int(window_days),
        "quiet_days": int(quiet_days),
        "min_other_cards": int(min_other_cards),
    }

    res = run_query(
        "r2_micro_drip_cohort",
        {
            "flagged_txn_id": str(int(flagged_txn_id)),
            "opened_at": as_of_str,
            "max_amt": float(max_amt),
            "min_charges": int(min_charges),
            "min_span_days": float(min_span_days),
            "window_days": int(window_days),
            "quiet_days": int(quiet_days),
            "min_other_cards": int(min_other_cards),
        },
        host=host,
        port=port,
        timeout=timeout,
    )
    row = _extract_row(
        res,
        "r2_micro_drip_cohort",
        ("fired", "flagged_card", "n_cohort_cards", "n_other_cards", "member_cards", "per_card_txns"),
    )

    raw_members = row.get("member_cards")
    raw_txns = row.get("per_card_txns")
    if not isinstance(raw_members, list) or not isinstance(raw_txns, dict):
        raise TigerGraphUnavailableError(
            f"Fail-closed: malformed types in r2_micro_drip_cohort reply: {row!r}"
        )

    n_cohort = int(row.get("n_cohort_cards", 0))
    n_other = int(row.get("n_other_cards", 0))
    fired = bool(row.get("fired", False))

    if n_cohort > 0:
        params["n_cohort_cards"] = n_cohort
        params["n_other_cards"] = n_other
        params["fired"] = fired

    if not fired:
        return [], {}, params

    member_cards = sorted(str(c) for c in raw_members if c)
    per_card_txns: dict[str, list[int]] = {}
    for cid in member_cards:
        tlist = raw_txns.get(cid, [])
        if not isinstance(tlist, list):
            raise TigerGraphUnavailableError(
                f"Fail-closed: per_card_txns[{cid!r}] is not a list: {tlist!r}"
            )
        per_card_txns[cid] = sorted(int(x) for x in tlist)

    return member_cards, per_card_txns, params


def recipient_fanin_tg(
    flagged_txn_id: int,
    opened_at: pd.Timestamp | str,
    window_days: int = 21,
    min_amount: float = 100.0,
    min_k_now: int = 5,
    burst_mult: float = 3.0,
    max_p: float = 1e-4,
    freemail: set[str] | list[str] | None = None,
    tx_asof: pd.DataFrame | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> tuple[list[str], dict[str, list[int]], dict[str, Any]]:
    """Run `r2_recipient_fanin` on TigerGraph and return `(member_card_ids, per_card_txn_ids, params)`
    matching `assay.r2.detectors.fan_in.cohort_members`.
    Passes the freemail exclusion list as a `SET<STRING>` parameter.
    Fails closed (`TigerGraphUnavailableError`) on any TG error or malformed reply.
    """
    require_tg(host=host, port=port)
    t_open = pd.Timestamp(opened_at)
    as_of_str = t_open.strftime("%Y-%m-%d %H:%M:%S")

    params: dict[str, Any] = {
        "flagged_txn_id": int(flagged_txn_id),
        "opened_at": str(t_open),
        "window_days": int(window_days),
        "min_amount": float(min_amount),
        "min_k_now": int(min_k_now),
        "burst_mult": float(burst_mult),
        "max_p": float(max_p),
    }

    if freemail is not None:
        freemail_set = {str(d).strip().lower() for d in freemail if str(d).strip()}
    elif tx_asof is not None and not tx_asof.empty:
        df_cut = tx_asof[tx_asof.t <= t_open]
        freemail_set = set(DEFAULT_FREEMAIL_SET) | freemail_domains_by_base_rate(df_cut)
    else:
        freemail_set = set(DEFAULT_FREEMAIL_SET)

    res = run_query(
        "r2_recipient_fanin",
        {
            "flagged_txn_id": str(int(flagged_txn_id)),
            "opened_at": as_of_str,
            "freemail": sorted(freemail_set),
            "window_days": int(window_days),
            "min_amount": float(min_amount),
            "min_k_now": int(min_k_now),
            "burst_mult": float(burst_mult),
            "max_p": float(max_p),
        },
        host=host,
        port=port,
        timeout=timeout,
    )
    row = _extract_row(
        res,
        "r2_recipient_fanin",
        (
            "fired",
            "flagged_qualifies",
            "flagged_card_in_first_time_cohort",
            "flagged_card",
            "R_emaildomain",
            "k_now",
            "baseline_mean",
            "baseline_max",
            "p_value",
            "member_cards",
            "per_card_txns",
        ),
    )

    raw_members = row.get("member_cards")
    raw_txns = row.get("per_card_txns")
    if not isinstance(raw_members, list) or not isinstance(raw_txns, dict):
        raise TigerGraphUnavailableError(
            f"Fail-closed: malformed types in r2_recipient_fanin reply: {row!r}"
        )

    flagged_card = str(row.get("flagged_card", ""))
    if not flagged_card:
        return [], {}, params

    if not bool(row.get("flagged_qualifies", False)):
        params["flagged_qualifies"] = False
        return [], {}, params

    if not bool(row.get("flagged_card_in_first_time_cohort", False)):
        params["flagged_card_in_first_time_cohort"] = False
        return [], {}, params

    k_now = int(row.get("k_now", 0))
    base_mean = float(row.get("baseline_mean", 0.0))
    base_max = int(row.get("baseline_max", 0))
    p_val = float(row.get("p_value", 1.0))
    fired = bool(row.get("fired", False))

    params.update(
        {
            "R_emaildomain": str(row.get("R_emaildomain", "")),
            "k_now": k_now,
            "baseline_mean": round(base_mean, 4),
            "baseline_max": base_max,
            "p_value": float(f"{p_val:.4e}"),
            "fired": fired,
        }
    )
    if not fired:
        return [], {}, params

    member_cards = sorted(str(c) for c in raw_members if c)
    per_card_txns: dict[str, list[int]] = {}
    for cid in member_cards:
        tlist = raw_txns.get(cid, [])
        if not isinstance(tlist, list):
            raise TigerGraphUnavailableError(
                f"Fail-closed: per_card_txns[{cid!r}] is not a list: {tlist!r}"
            )
        per_card_txns[cid] = sorted(int(x) for x in tlist)

    return member_cards, per_card_txns, params


@dataclass
class CohortParityRow:
    case_id: str
    card_id: str
    flagged_txn_id: int
    opened_at: str
    detector: str
    local_members: list[str]
    tg_members: list[str]
    local_txns: dict[str, list[int]]
    tg_txns: dict[str, list[int]]
    members_identical: bool
    txns_identical: bool
    identical: bool
    tg_ms: float


def parity(case_ids: list[str] | None = None) -> dict[str, Any]:
    """Compare TigerGraph GSQL vs local Python `cohort_members` for both `MicroDripCohort`
    and `FanIn` across `case_pack()` cases (or the subset in `case_ids`).
    Prints a formatted comparison table (case, detector, local members, tg members, identical?)
    and returns structured parity measurements.
    """
    require_tg()
    tx = tx_all()
    cp = case_pack()
    if case_ids is not None:
        wanted = {str(cid) for cid in case_ids}
        cp = cp[cp.case_id.isin(wanted)].copy()

    rows: list[CohortParityRow] = []
    mismatches: list[str] = []

    for r in cp.itertuples():
        cid = str(r.case_id)
        card_id = str(r.card_id)
        ftid = int(r.flagged_txn_id)
        t_open = pd.Timestamp(r.opened_at)
        tx_asof = tx[tx.t <= t_open]

        # 1. MicroDripCohort: Local vs TG
        loc_md_members, loc_md_txns, _ = micro_drip_cohort_local(tx_asof, ftid, t_open)
        t0 = time.perf_counter()
        tg_md_members, tg_md_txns, _ = micro_drip_cohort_tg(ftid, t_open)
        md_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        md_mem_eq = loc_md_members == tg_md_members
        md_txn_eq = loc_md_txns == tg_md_txns
        md_ident = md_mem_eq and md_txn_eq
        if not md_ident:
            mismatches.append(
                f"{cid} [micro_drip]: local_members={loc_md_members} vs tg_members={tg_md_members}; "
                f"local_txns={loc_md_txns} vs tg_txns={tg_md_txns}"
            )
        rows.append(
            CohortParityRow(
                case_id=cid,
                card_id=card_id,
                flagged_txn_id=ftid,
                opened_at=str(t_open),
                detector="micro_drip_cohort",
                local_members=loc_md_members,
                tg_members=tg_md_members,
                local_txns=loc_md_txns,
                tg_txns=tg_md_txns,
                members_identical=md_mem_eq,
                txns_identical=md_txn_eq,
                identical=md_ident,
                tg_ms=md_ms,
            )
        )

        # 2. Recipient FanIn: Local vs TG
        freemail_set = set(DEFAULT_FREEMAIL_SET) | freemail_domains_by_base_rate(tx_asof)
        loc_fi_members, loc_fi_txns, _ = fan_in_cohort_local(tx_asof, ftid, t_open)
        t1 = time.perf_counter()
        tg_fi_members, tg_fi_txns, _ = recipient_fanin_tg(
            ftid, t_open, freemail=freemail_set
        )
        fi_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        fi_mem_eq = loc_fi_members == tg_fi_members
        fi_txn_eq = loc_fi_txns == tg_fi_txns
        fi_ident = fi_mem_eq and fi_txn_eq
        if not fi_ident:
            mismatches.append(
                f"{cid} [recipient_fanin]: local_members={loc_fi_members} vs tg_members={tg_fi_members}; "
                f"local_txns={loc_fi_txns} vs tg_txns={tg_fi_txns}"
            )
        rows.append(
            CohortParityRow(
                case_id=cid,
                card_id=card_id,
                flagged_txn_id=ftid,
                opened_at=str(t_open),
                detector="recipient_fanin",
                local_members=loc_fi_members,
                tg_members=tg_fi_members,
                local_txns=loc_fi_txns,
                tg_txns=tg_fi_txns,
                members_identical=fi_mem_eq,
                txns_identical=fi_txn_eq,
                identical=fi_ident,
                tg_ms=fi_ms,
            )
        )

    header = (
        f"{'Case':<8} {'Detector':<19} {'Local Members':<28} "
        f"{'TG Members':<28} {'TxnsEq?':<8} {'Identical?':<10} {'TG_ms':>7}"
    )
    print("\n=== C6 TigerGraph vs Local Cohort Parity ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        loc_s = (
            f"{len(row.local_members)}: {','.join(row.local_members[:2])}"
            + ("..." if len(row.local_members) > 2 else "")
            if row.local_members
            else "0: []"
        )
        tg_s = (
            f"{len(row.tg_members)}: {','.join(row.tg_members[:2])}"
            + ("..." if len(row.tg_members) > 2 else "")
            if row.tg_members
            else "0: []"
        )
        print(
            f"{row.case_id:<8} {row.detector:<19} {loc_s:<28} "
            f"{tg_s:<28} {str(row.txns_identical):<8} {str(row.identical):<10} {row.tg_ms:>7.1f}"
        )

    if mismatches:
        print("\nCohort parity mismatches:")
        for m in mismatches:
            print("  ", m)

    fired_rows = [r for r in rows if r.local_members or r.tg_members]
    return {
        "n_cases": len(cp),
        "n_comparisons": len(rows),
        "n_identical": sum(1 for r in rows if r.identical),
        "n_fired_either": len(fired_rows),
        "n_fired_identical": sum(1 for r in fired_rows if r.identical),
        "rows": rows,
        "mismatches": mismatches,
    }
