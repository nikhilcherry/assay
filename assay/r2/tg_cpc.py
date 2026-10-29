"""C10 -- TigerGraph GSQL Common-Point-of-Compromise (CPC) queries
(`r2_cpc_loss_cohort` and `r2_cpc_origin_counts`).

Mirrors the graph-shaped helpers in `assay.r2.detectors.cpc`:
  - `fast_find_cohort_by_tid` -> `cpc_loss_cohort_tg` (and drop-in `find_cohort_by_tid_tg`)
  - `extract_features_sub` / `compute_origin_lifts` -> `cpc_origin_counts_tg` (and drop-in
    `compute_origin_lifts_tg`)

Schema & attribute support note:
  In `graph/schema.gsql` and `graph/load_r2.gsql`, every attribute required by the 6
  candidate origin feature families in `cpc.py:extract_features_sub` is loaded onto the
  `Txn` vertex:
    - `product` (`ProductCD`), `amount` (`amount`), `addr1` (`INT`, `-1` when NaN),
      `r_email` (`R_emaildomain`), `p_email` (`P_emaildomain`),
      `device_profile` (`device_profile`), and `dist1` (`DOUBLE`, `-1.0` when NaN).
  Therefore all 6 feature families (`ProductCD_amount`, `ProductCD_addr1`, `R_emaildomain`,
  `P_emaildomain_ProductCD`, `device_profile`, `merchant_like`) are supported directly in
  GSQL with zero unsupported keys.

  Cardinality nuance documented:
    In `cpc.py:get_precomputed_index`, `idx.global_card_counts` precomputes bank-wide
    distinct card counts for 3 of the 6 families (`ProductCD_amount`, `R_emaildomain`,
    `ProductCD_addr1`) and falls back to `k_aff + k_ctrl` in `compute_origin_lifts` for the
    other 3 families (`P_emaildomain_ProductCD`, `device_profile`, `merchant_like`).
    `r2_cpc_origin_counts` computes the true bank-wide distinct card count across `Txn`
    vertices for all 6 families (`origin_total_cardinality`, plus `origin_asof_cardinality`
    restricted to `t.ts <= opened_at`). In `compute_origin_lifts_tg`, we use the graph's
    `origin_total_cardinality` for the 3 indexed families and `k_aff + k_ctrl` for the
    remaining 3, and iterate candidate keys in sorted lexicographical order (matching
    `aff_feats.groupby('feature')` in `cpc.py`) so `compute_origin_lifts_tg` is 100%
    row-for-row identical to `cpc.py`.

Fail-closed contract:
  Every query installation, smoke run, or query invocation raises `TigerGraphUnavailableError`
  if TigerGraph is unreachable, returns a non-200 HTTP code, returns a GSQL error, or returns
  a malformed payload missing required keys.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from assay.data import ROOT
from assay.r2.detectors.cpc import (
    FREE_MAIL,
    PrecomputedIndex,
    compute_origin_lifts as compute_origin_lifts_local,
    extract_features_sub,
    fast_find_cohort_by_tid,
    fast_matched_controls,
    get_precomputed_index,
    hypergeom_sf,
)
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

CPC_QUERIES: tuple[str, ...] = (
    "r2_cpc_loss_cohort",
    "r2_cpc_origin_counts",
)

SUPPORTED_ORIGIN_FAMILIES: tuple[str, ...] = (
    "ProductCD_amount",
    "ProductCD_addr1",
    "R_emaildomain",
    "P_emaildomain_ProductCD",
    "device_profile",
    "merchant_like",
)

INDEXED_CARDINALITY_FAMILIES: frozenset[str] = frozenset(
    {
        "ProductCD_amount",
        "ProductCD_addr1",
        "R_emaildomain",
    }
)


def _extract_row(
    results: list[dict[str, Any]],
    query_name: str,
    required_keys: tuple[str, ...],
) -> dict[str, Any]:
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


def smoke_test_cpc_queries(
    *,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, float]:
    """Smoke-run both CPC GSQL queries over REST++ and fail loudly if either errors or
    returns a malformed payload (guards against the `INSTALL QUERY` silent non-install gotcha).
    """
    require_tg(host=host, port=port)
    timings_ms: dict[str, float] = {}

    # 1. Smoke-run r2_cpc_loss_cohort on HHF-001
    t0 = time.perf_counter()
    res_lc = run_query(
        "r2_cpc_loss_cohort",
        {
            "flagged_txn_id": "3660108",
            "opened_at": "2017-01-15 01:25:19",
        },
        host=host,
        port=port,
    )
    timings_ms["r2_cpc_loss_cohort"] = round((time.perf_counter() - t0) * 1000.0, 2)
    row_lc = _extract_row(
        res_lc,
        "r2_cpc_loss_cohort",
        (
            "has_shape",
            "flagged_card",
            "ProductCD",
            "channel",
            "min_amt",
            "max_amt",
            "min_count",
            "max_count",
            "t0_sec",
            "member_cards",
            "first_loss_sec",
            "burst_txns",
        ),
    )
    member_cards = sorted(str(c) for c in row_lc["member_cards"])
    first_loss_sec = {str(k): int(v) for k, v in row_lc["first_loss_sec"].items()}
    print(
        f"[SMOKE] r2_cpc_loss_cohort ({timings_ms['r2_cpc_loss_cohort']}ms) -> "
        f"has_shape={row_lc['has_shape']}, n_cards={len(member_cards)}"
    )

    # 2. Smoke-run r2_cpc_origin_counts on the HHF-001 cohort
    card_loss_epochs = [f"{cid}:{first_loss_sec[cid]}" for cid in member_cards if cid in first_loss_sec]
    t1 = time.perf_counter()
    res_oc = run_query(
        "r2_cpc_origin_counts",
        {
            "card_ids": member_cards if member_cards else ["C05061-K1"],
            "card_loss_epochs": card_loss_epochs if card_loss_epochs else ["C05061-K1:1484208000"],
            "opened_at": "2017-01-15 01:25:19",
            "freemail": sorted(FREE_MAIL),
            "lookback_days": 60,
        },
        host=host,
        port=port,
    )
    timings_ms["r2_cpc_origin_counts"] = round((time.perf_counter() - t1) * 1000.0, 2)
    row_oc = _extract_row(
        res_oc,
        "r2_cpc_origin_counts",
        (
            "origin_card_count",
            "origin_total_cardinality",
            "origin_asof_cardinality",
            "origin_cards",
            "origin_txns",
            "origin_card_txns",
        ),
    )
    print(
        f"[SMOKE] r2_cpc_origin_counts ({timings_ms['r2_cpc_origin_counts']}ms) -> "
        f"n_origins={len(row_oc['origin_card_count'])}"
    )

    return timings_ms


def install(conn: Any = None, *, force: bool = False) -> dict[str, Any]:
    """Create and install `r2_cpc_loss_cohort` and `r2_cpc_origin_counts` from
    `graph/queries/r2_cpc.gsql`, then smoke-run each query once over REST++ and fail
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

    gsql_path = ROOT / "graph" / "queries" / "r2_cpc.gsql"
    gsql_text = _strip_drops_and_installs(gsql_path.read_text())

    create_out = run_gsql_script(gsql_text, label="create_r2_cpc")
    created = all(re.search(rf"Successfully created queries: \[[^\]]*\b{q}\b", create_out) for q in CPC_QUERIES)
    if not created or re.search(r"syntax error|semantic error|saved as draft", create_out, re.I):
        raise TigerGraphUnavailableError(f"Fail-closed: Failed to CREATE r2_cpc.gsql:\n{create_out}")

    installed_before = _installed_query_names(gname)
    to_install = list(CPC_QUERIES) if force else [q for q in CPC_QUERIES if q not in installed_before]

    q_csv = ", ".join(CPC_QUERIES)
    inst_out = run_gsql_script(
        f"USE GRAPH {gname}\nINSTALL QUERY {q_csv}\n",
        label=f"install_r2_cpc({q_csv})",
        timeout=480.0,
    )
    if re.search(r"installation failed|syntax error|semantic error|is DRAFT|exited with code [1-9]", inst_out, re.I):
        raise TigerGraphUnavailableError(f"Fail-closed: INSTALL QUERY {q_csv} failed:\n{inst_out}")

    installed_after = _installed_query_names(gname)
    still_missing = [q for q in CPC_QUERIES if q not in installed_after]
    if still_missing:
        raise TigerGraphUnavailableError(
            f"Fail-closed: Queries missing from REST++ endpoints after INSTALL: {still_missing}"
        )

    timings = smoke_test_cpc_queries(host=host, port=port)
    return {
        "graph": gname,
        "installed_queries": list(CPC_QUERIES),
        "previously_missing": to_install,
        "smoke_test_ms": timings,
    }


def cpc_loss_cohort_tg(
    flagged_txn_id: int,
    opened_at: pd.Timestamp | str | int,
    *,
    idx: PrecomputedIndex | None = None,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, Any] | None, list[str], dict[str, int], dict[str, list[int]]]:
    """Run `r2_cpc_loss_cohort` on TigerGraph and return
    `(shape, cohort, first_loss_ns, burst_txns)` with the exact same structure as
    `assay.r2.detectors.cpc.fast_find_cohort_by_tid`.
    """
    require_tg(host=host, port=port)
    t_open = pd.Timestamp(opened_at)
    as_of_str = t_open.strftime("%Y-%m-%d %H:%M:%S")

    res = run_query(
        "r2_cpc_loss_cohort",
        {
            "flagged_txn_id": str(int(flagged_txn_id)),
            "opened_at": as_of_str,
        },
        host=host,
        port=port,
        timeout=timeout,
    )
    row = _extract_row(
        res,
        "r2_cpc_loss_cohort",
        (
            "has_shape",
            "flagged_card",
            "ProductCD",
            "channel",
            "min_amt",
            "max_amt",
            "min_count",
            "max_count",
            "t0_sec",
            "member_cards",
            "first_loss_sec",
            "burst_txns",
        ),
    )

    if not bool(row.get("has_shape", False)):
        return None, [], {}, {}

    flagged_card = str(row.get("flagged_card", ""))
    t0_ns = int(row.get("t0_sec", 0)) * 1_000_000_000
    flagged_c_idx = int(idx.card_to_idx.get(flagged_card, -1)) if idx is not None else -1

    shape: dict[str, Any] = {
        "mode": "c_online_burst",
        "ProductCD": str(row.get("ProductCD", "C")),
        "channel": str(row.get("channel", "online")),
        "min_amt": float(row.get("min_amt", 250.0)),
        "max_amt": float(row.get("max_amt", 2200.0)),
        "min_count": int(row.get("min_count", 2)),
        "max_count": int(row.get("max_count", 5)),
        "prior_cap": 250.0,
        "t0_ns": t0_ns,
        "flagged_card": flagged_card,
        "flagged_c_idx": flagged_c_idx,
    }

    raw_members = row.get("member_cards")
    raw_loss_sec = row.get("first_loss_sec")
    raw_burst = row.get("burst_txns")
    if not isinstance(raw_members, list) or not isinstance(raw_loss_sec, dict) or not isinstance(raw_burst, dict):
        raise TigerGraphUnavailableError(
            f"Fail-closed: malformed types in r2_cpc_loss_cohort reply: {row!r}"
        )

    cohort = sorted(str(c) for c in raw_members if c)
    first_loss_ns: dict[str, int] = {}
    burst_txns: dict[str, list[int]] = {}
    for cid in cohort:
        if cid not in raw_loss_sec or cid not in raw_burst:
            raise TigerGraphUnavailableError(
                f"Fail-closed: cohort card {cid!r} missing from first_loss_sec or burst_txns"
            )
        first_loss_ns[cid] = int(raw_loss_sec[cid]) * 1_000_000_000
        tlist = raw_burst[cid]
        if not isinstance(tlist, list):
            raise TigerGraphUnavailableError(
                f"Fail-closed: burst_txns[{cid!r}] is not a list: {tlist!r}"
            )
        burst_txns[cid] = sorted(int(x) for x in tlist)

    return shape, cohort, first_loss_ns, burst_txns


def find_cohort_by_tid_tg(
    idx: PrecomputedIndex,
    flagged_txn_id: int,
    opened_at_ns: int,
    *,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, Any] | None, list[str], dict[str, int], dict[str, list[int]]]:
    """Drop-in replacement for `cpc.fast_find_cohort_by_tid(idx, flagged_txn_id, opened_at_ns)`."""
    return cpc_loss_cohort_tg(
        flagged_txn_id=flagged_txn_id,
        opened_at=pd.Timestamp(int(opened_at_ns)),
        idx=idx,
        host=host,
        port=port,
        timeout=timeout,
    )


def cpc_origin_counts_tg(
    first_loss_ns: dict[str, int],
    opened_at: pd.Timestamp | str | int,
    lookback_days: int = 60,
    freemail: set[str] | list[str] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Run `r2_cpc_origin_counts` on TigerGraph for a mapping `{card_id: first_loss_ns}`
    and return:
      - `origin_card_count`: `{feature_key: int}` (how many of the input cards touched it in 60d lookback)
      - `origin_total_cardinality`: `{feature_key: int}` (full-graph distinct cards touching key)
      - `origin_asof_cardinality`: `{feature_key: int}` (as-of `opened_at` distinct cards touching key)
      - `origin_cards`: `{feature_key: sorted list[card_id]}`
      - `origin_txns`: `{feature_key: sorted list[txn_id]}`
      - `features_df`: `pd.DataFrame(columns=['card_id', 'TransactionID', 'feature'])` matching `extract_features_sub`
    """
    require_tg(host=host, port=port)
    if not first_loss_ns:
        return {
            "origin_card_count": {},
            "origin_total_cardinality": {},
            "origin_asof_cardinality": {},
            "origin_cards": {},
            "origin_txns": {},
            "features_df": pd.DataFrame(columns=["card_id", "TransactionID", "feature"]),
        }

    t_open = pd.Timestamp(opened_at)
    as_of_str = t_open.strftime("%Y-%m-%d %H:%M:%S")
    fm_list = sorted(freemail if freemail is not None else FREE_MAIL)

    card_ids = sorted(str(c) for c in first_loss_ns.keys())
    card_loss_epochs = [
        f"{cid}:{int(first_loss_ns[cid]) // 1_000_000_000}" for cid in card_ids
    ]

    res = run_query(
        "r2_cpc_origin_counts",
        {
            "card_ids": card_ids,
            "card_loss_epochs": card_loss_epochs,
            "opened_at": as_of_str,
            "freemail": fm_list,
            "lookback_days": int(lookback_days),
        },
        host=host,
        port=port,
        timeout=timeout,
    )
    row = _extract_row(
        res,
        "r2_cpc_origin_counts",
        (
            "origin_card_count",
            "origin_total_cardinality",
            "origin_asof_cardinality",
            "origin_cards",
            "origin_txns",
            "origin_card_txns",
        ),
    )

    raw_cnt = row.get("origin_card_count")
    raw_tot_card = row.get("origin_total_cardinality")
    raw_asof_card = row.get("origin_asof_cardinality")
    raw_cards = row.get("origin_cards")
    raw_txns = row.get("origin_txns")
    raw_card_txns = row.get("origin_card_txns")
    if (
        not isinstance(raw_cnt, dict)
        or not isinstance(raw_tot_card, dict)
        or not isinstance(raw_asof_card, dict)
        or not isinstance(raw_cards, dict)
        or not isinstance(raw_txns, dict)
        or not isinstance(raw_card_txns, dict)
    ):
        raise TigerGraphUnavailableError(
            f"Fail-closed: malformed types in r2_cpc_origin_counts reply: {row!r}"
        )

    # Sort dictionary keys deterministically to match pandas groupby('feature', sort=True)
    origin_card_count = {str(k): int(raw_cnt[k]) for k in sorted(raw_cnt.keys())}
    origin_total_cardinality = {str(k): int(raw_tot_card[k]) for k in sorted(raw_tot_card.keys())}
    origin_asof_cardinality = {str(k): int(raw_asof_card[k]) for k in sorted(raw_asof_card.keys())}
    origin_cards = {str(k): sorted(str(c) for c in raw_cards[k]) for k in sorted(raw_cards.keys())}
    origin_txns = {str(k): sorted(int(t) for t in raw_txns[k]) for k in sorted(raw_txns.keys())}

    feat_rows: list[tuple[str, int, str]] = []
    for feat in sorted(raw_card_txns.keys()):
        ct_list = raw_card_txns[feat]
        feat_s = str(feat)
        if not isinstance(ct_list, list):
            raise TigerGraphUnavailableError(
                f"Fail-closed: origin_card_txns[{feat_s!r}] is not a list: {ct_list!r}"
            )
        for item in ct_list:
            cid_s, tid_s = str(item).rsplit(":", 1)
            feat_rows.append((cid_s, int(tid_s), feat_s))

    feat_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    features_df = (
        pd.DataFrame(feat_rows, columns=["card_id", "TransactionID", "feature"])
        if feat_rows
        else pd.DataFrame(columns=["card_id", "TransactionID", "feature"])
    )

    return {
        "origin_card_count": origin_card_count,
        "origin_total_cardinality": origin_total_cardinality,
        "origin_asof_cardinality": origin_asof_cardinality,
        "origin_cards": origin_cards,
        "origin_txns": origin_txns,
        "features_df": features_df,
    }


def compute_origin_lifts_tg(
    idx: PrecomputedIndex,
    affected_cards: list[str],
    first_loss_ns: dict[str, int],
    controls: dict[str, list[str]],
    as_of_tx: pd.DataFrame | None = None,
    lookback_days: int = 60,
    *,
    opened_at: pd.Timestamp | str | int | None = None,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop-in replacement for `cpc.compute_origin_lifts` backed by `r2_cpc_origin_counts`
    on TigerGraph for the cohort origin counts and cardinalities, while keeping the
    matched-control lift and one-sided Fisher exact p-value calculation in Python.

    Returns `(res_df, aff_feats)` with the exact schema and sorting of
    `cpc.compute_origin_lifts`.
    """
    sorted_affected = sorted(str(c) for c in affected_cards)
    if opened_at is not None:
        t_open = pd.Timestamp(opened_at)
    elif as_of_tx is not None and not as_of_tx.empty:
        t_open = pd.Timestamp(as_of_tx["t"].max())
    elif first_loss_ns:
        t_open = pd.Timestamp(max(first_loss_ns.values()))
    else:
        t_open = pd.Timestamp("2017-01-31 23:59:59")

    tg_aff = cpc_origin_counts_tg(
        first_loss_ns={c: int(first_loss_ns[c]) for c in sorted_affected if c in first_loss_ns},
        opened_at=t_open,
        lookback_days=lookback_days,
        host=host,
        port=port,
        timeout=timeout,
    )
    aff_feats = tg_aff["features_df"]
    aff_counts: dict[str, int] = tg_aff["origin_card_count"]
    tot_card: dict[str, int] = tg_aff["origin_total_cardinality"]

    if not aff_counts:
        return pd.DataFrame(), aff_feats

    all_ctrl_cards = {ctrl_c for c_list in controls.values() for ctrl_c in c_list}
    ctrl_counts: dict[str, int] = {}

    if as_of_tx is not None and not as_of_tx.empty and all_ctrl_cards:
        sub_ctrl_tx = as_of_tx[as_of_tx.card_id.isin(all_ctrl_cards)]
        if not sub_ctrl_tx.empty:
            feats_ctrl_all = extract_features_sub(sub_ctrl_tx)
            tid_to_t = sub_ctrl_tx.set_index("TransactionID")["t"]
            feats_ctrl_all["t"] = feats_ctrl_all["TransactionID"].map(tid_to_t)
            ctrl_frames = []
            for c in sorted_affected:
                t_loss = pd.Timestamp(first_loss_ns[c])
                w_start = t_loss - pd.Timedelta(days=lookback_days)
                c_ctrls = controls.get(c, [])
                if c_ctrls:
                    cf = feats_ctrl_all[
                        (feats_ctrl_all.card_id.isin(c_ctrls))
                        & (feats_ctrl_all.t >= w_start)
                        & (feats_ctrl_all.t < t_loss)
                    ]
                    if not cf.empty:
                        ctrl_frames.append(cf[["card_id", "TransactionID", "feature"]])
            if ctrl_frames:
                ctrl_df = pd.concat(ctrl_frames, ignore_index=True)
                ctrl_counts = ctrl_df.groupby("feature")["card_id"].nunique().to_dict()

    N_aff = len(sorted_affected)
    N_ctrl = max(len(all_ctrl_cards), 1)

    rows = []
    # Iterate in sorted feature order so stable sort on ['pval', 'lift'] breaks ties identically to pandas groupby('feature')
    for feat in sorted(aff_counts.keys()):
        k_aff = int(aff_counts[feat])
        if k_aff < 2:
            continue
        k_ctrl = int(ctrl_counts.get(feat, 0))
        share_aff = k_aff / N_aff
        share_ctrl = max(k_ctrl, 0.5) / N_ctrl
        lift = share_aff / share_ctrl
        pval = hypergeom_sf(int(k_aff), N_aff + N_ctrl, int(k_aff) + k_ctrl, N_aff)
        fam = feat.split(":", 1)[0]
        if fam in INDEXED_CARDINALITY_FAMILIES:
            card_glob = int(tot_card.get(feat, k_aff + k_ctrl))
        else:
            card_glob = int(k_aff + k_ctrl)
        rows.append(
            {
                "feature": feat,
                "k_aff": int(k_aff),
                "k_ctrl": k_ctrl,
                "lift": float(lift),
                "pval": float(pval),
                "cardinality": card_glob,
            }
        )

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(["pval", "lift"], ascending=[True, False]).reset_index(drop=True)
    return res_df, aff_feats


def local_origin_counts(
    idx: PrecomputedIndex,
    affected_cards: list[str],
    first_loss_ns: dict[str, int],
    as_of_tx: pd.DataFrame,
    lookback_days: int = 60,
) -> tuple[dict[str, int], dict[str, int], pd.DataFrame]:
    """Compute local Python origin counts (`k_aff` per candidate origin key) and
    indexed global cardinalities for direct parity comparison against `cpc_origin_counts_tg`.
    """
    if not affected_cards:
        return {}, {}, pd.DataFrame(columns=["card_id", "TransactionID", "feature"])

    sub_tx = as_of_tx[as_of_tx.card_id.isin(set(affected_cards))]
    if sub_tx.empty:
        return {}, {}, pd.DataFrame(columns=["card_id", "TransactionID", "feature"])

    feats_all = extract_features_sub(sub_tx)
    tid_to_t = sub_tx.set_index("TransactionID")["t"]
    feats_all["t"] = feats_all["TransactionID"].map(tid_to_t)

    aff_frames = []
    for c in affected_cards:
        t_loss = pd.Timestamp(first_loss_ns[c])
        w_start = t_loss - pd.Timedelta(days=lookback_days)
        c_f = feats_all[(feats_all.card_id == c) & (feats_all.t >= w_start) & (feats_all.t < t_loss)]
        if not c_f.empty:
            aff_frames.append(c_f[["card_id", "TransactionID", "feature"]])

    aff_feats = (
        pd.concat(aff_frames, ignore_index=True)
        if aff_frames
        else pd.DataFrame(columns=["card_id", "TransactionID", "feature"])
    )
    if aff_feats.empty:
        return {}, {}, aff_feats

    counts = {str(k): int(v) for k, v in aff_feats.groupby("feature")["card_id"].nunique().items()}
    cardinalities = {
        str(k): int(idx.global_card_counts[k])
        for k in counts
        if k in idx.global_card_counts
    }
    return counts, cardinalities, aff_feats


def _compare_lift_tables(
    loc_df: pd.DataFrame,
    tg_df: pd.DataFrame,
    tol: float = 1e-12,
) -> tuple[bool, str | None]:
    """Compare local vs TigerGraph lift DataFrames and return `(identical, first_diff_str)`."""
    if len(loc_df) != len(tg_df):
        return False, f"row_count: local={len(loc_df)} vs tg={len(tg_df)}"
    if loc_df.empty and tg_df.empty:
        return True, None

    for i in range(len(loc_df)):
        lr = loc_df.iloc[i]
        tr = tg_df.iloc[i]
        if (
            str(lr.feature) != str(tr.feature)
            or int(lr.k_aff) != int(tr.k_aff)
            or int(lr.k_ctrl) != int(tr.k_ctrl)
            or int(lr.cardinality) != int(tr.cardinality)
            or not math.isclose(float(lr.lift), float(tr.lift), rel_tol=tol, abs_tol=tol)
            or not math.isclose(float(lr.pval), float(tr.pval), rel_tol=tol, abs_tol=tol)
        ):
            diff_msg = (
                f"row[{i}] local=(feat={lr.feature}, k_aff={lr.k_aff}, k_ctrl={lr.k_ctrl}, "
                f"lift={lr.lift:.6g}, p={lr.pval:.6g}, card={lr.cardinality}) vs "
                f"tg=(feat={tr.feature}, k_aff={tr.k_aff}, k_ctrl={tr.k_ctrl}, "
                f"lift={tr.lift:.6g}, p={tr.pval:.6g}, card={tr.cardinality})"
            )
            return False, diff_msg
    return True, None


@dataclass
class CpcParityRow:
    case_id: str
    card_id: str
    flagged_txn_id: int
    opened_at: str
    local_members: list[str]
    tg_members: list[str]
    local_burst_txns: dict[str, list[int]]
    tg_burst_txns: dict[str, list[int]]
    local_first_loss_ns: dict[str, int]
    tg_first_loss_ns: dict[str, int]
    local_origin_counts: dict[str, int]
    tg_origin_counts: dict[str, int]
    local_indexed_cardinality: dict[str, int]
    tg_indexed_cardinality: dict[str, int]
    members_identical: bool
    burst_txns_identical: bool
    origin_counts_identical: bool
    cardinality_identical: bool
    lifts_identical: bool
    first_lift_diff: str | None
    identical: bool
    cohort_ms: float
    origin_ms: float


def parity(case_ids: list[str] | None = None) -> dict[str, Any]:
    """Compare TigerGraph GSQL (`r2_cpc_loss_cohort` & `r2_cpc_origin_counts`) vs local
    Python (`fast_find_cohort_by_tid` & `compute_origin_lifts`) across `case_ids`
    (defaults to `['HHF-001', 'HHF-002', 'HHF-003', 'HHF-006']`).
    Prints a detailed comparison table and returns structured parity measurements.
    """
    require_tg()
    idx = get_precomputed_index()
    tx = tx_all()
    cp = case_pack()
    if case_ids is None:
        case_ids = ["HHF-001", "HHF-002", "HHF-003", "HHF-006"]
    wanted = {str(cid) for cid in case_ids}
    cp = cp[cp.case_id.isin(wanted)].copy()

    rows: list[CpcParityRow] = []
    mismatches: list[str] = []

    for r in cp.itertuples():
        cid = str(r.case_id)
        card_id = str(r.card_id)
        ftid = int(r.flagged_txn_id)
        t_open = pd.Timestamp(r.opened_at)
        opened_ns = int(t_open.value)
        tx_asof = tx[tx.t <= t_open]

        # 1. Loss cohort: Local vs TG
        loc_shape, loc_members, loc_loss_ns, loc_burst = fast_find_cohort_by_tid(idx, ftid, opened_ns)
        t0 = time.perf_counter()
        tg_shape, tg_members, tg_loss_ns, tg_burst = cpc_loss_cohort_tg(ftid, t_open, idx=idx)
        cohort_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        loc_burst_sorted = {k: sorted(v) for k, v in loc_burst.items()}
        tg_burst_sorted = {k: sorted(v) for k, v in tg_burst.items()}

        # 2. Origin counts (60d lookback): Local vs TG
        loc_oc, loc_card, _ = local_origin_counts(idx, loc_members, loc_loss_ns, tx_asof, lookback_days=60)
        t1 = time.perf_counter()
        tg_oc_res = cpc_origin_counts_tg(tg_loss_ns, t_open, lookback_days=60)
        origin_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        tg_oc = tg_oc_res["origin_card_count"]
        tg_tot_card = tg_oc_res["origin_total_cardinality"]
        tg_indexed_card = {k: tg_tot_card[k] for k in loc_card if k in tg_tot_card}

        # 3. Full lift table comparison (`compute_origin_lifts` vs `compute_origin_lifts_tg`)
        lifts_eq = True
        first_lift_diff: str | None = None
        if loc_shape is not None and tg_shape is not None and loc_members:
            loc_controls = fast_matched_controls(idx, loc_members, loc_shape["t0_ns"], rng_seed=42)
            tg_controls = fast_matched_controls(idx, tg_members, tg_shape["t0_ns"], rng_seed=42)
            loc_lifts_df, _ = compute_origin_lifts_local(
                idx, loc_members, loc_loss_ns, loc_controls, tx_asof, lookback_days=60
            )
            tg_lifts_df, _ = compute_origin_lifts_tg(
                idx,
                tg_members,
                tg_loss_ns,
                tg_controls,
                tx_asof,
                lookback_days=60,
                opened_at=t_open,
            )
            lifts_eq, first_lift_diff = _compare_lift_tables(loc_lifts_df, tg_lifts_df, tol=1e-12)

        mem_eq = loc_members == tg_members
        burst_eq = loc_burst_sorted == tg_burst_sorted and loc_loss_ns == tg_loss_ns
        oc_eq = loc_oc == tg_oc
        card_eq = loc_card == tg_indexed_card
        ident = mem_eq and burst_eq and oc_eq and card_eq and lifts_eq

        if not ident:
            mismatches.append(
                f"{cid}: mem_eq={mem_eq}, burst_eq={burst_eq}, oc_eq={oc_eq}, "
                f"card_eq={card_eq}, lifts_eq={lifts_eq}, first_lift_diff={first_lift_diff}"
            )

        rows.append(
            CpcParityRow(
                case_id=cid,
                card_id=card_id,
                flagged_txn_id=ftid,
                opened_at=str(t_open),
                local_members=loc_members,
                tg_members=tg_members,
                local_burst_txns=loc_burst_sorted,
                tg_burst_txns=tg_burst_sorted,
                local_first_loss_ns=loc_loss_ns,
                tg_first_loss_ns=tg_loss_ns,
                local_origin_counts=loc_oc,
                tg_origin_counts=tg_oc,
                local_indexed_cardinality=loc_card,
                tg_indexed_cardinality=tg_indexed_card,
                members_identical=mem_eq,
                burst_txns_identical=burst_eq,
                origin_counts_identical=oc_eq,
                cardinality_identical=card_eq,
                lifts_identical=lifts_eq,
                first_lift_diff=first_lift_diff,
                identical=ident,
                cohort_ms=cohort_ms,
                origin_ms=origin_ms,
            )
        )

    header = (
        f"{'Case':<8} {'Card':<10} {'LocCards':>8} {'TGCards':>8} "
        f"{'LocOrigins':>10} {'TGOrigins':>10} {'MemEq':<6} {'OrigEq':<6} "
        f"{'CardEq':<6} {'LiftEq':<6} {'CohortMs':>8} {'OrigMs':>8}"
    )
    print("\n=== C10 TigerGraph vs Local CPC Parity ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.case_id:<8} {row.card_id:<10} {len(row.local_members):>8} {len(row.tg_members):>8} "
            f"{len(row.local_origin_counts):>10} {len(row.tg_origin_counts):>10} "
            f"{str(row.members_identical):<6} {str(row.origin_counts_identical):<6} "
            f"{str(row.cardinality_identical):<6} {str(row.lifts_identical):<6} "
            f"{row.cohort_ms:>8.1f} {row.origin_ms:>8.1f}"
        )
        print(f"  [{row.case_id}] Local cohort members ({len(row.local_members)}): {row.local_members}")
        print(f"  [{row.case_id}] TG    cohort members ({len(row.tg_members)}): {row.tg_members}")
        top_loc = sorted(row.local_origin_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        top_tg = sorted(row.tg_origin_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        print(f"  [{row.case_id}] Local top-5 origin counts (of {len(row.local_origin_counts)}): {top_loc}")
        print(f"  [{row.case_id}] TG    top-5 origin counts (of {len(row.tg_origin_counts)}): {top_tg}")
        if row.first_lift_diff is not None:
            print(f"  [{row.case_id}] First differing lift row: {row.first_lift_diff}")

    if mismatches:
        print("\nCPC parity mismatches:")
        for m in mismatches:
            print("  ", m)

    return {
        "n_cases": len(rows),
        "n_identical": sum(1 for r in rows if r.identical),
        "supported_origin_families": list(SUPPORTED_ORIGIN_FAMILIES),
        "unsupported_origin_families": [],
        "rows": rows,
        "mismatches": mismatches,
    }
