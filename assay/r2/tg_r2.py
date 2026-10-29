"""N2 -- TigerGraph as the live source of truth for Round 2.

Key implementation details (Round 3):
- `graph/r2_vector_schema.gsql` (applied) added `ClosedCase.behaviour(DIMENSION=24, METRIC="COSINE")`
  and loading job `load_case_vectors`.
- `graph/queries/r2_vector.gsql` (applied) uses `FOR GRAPH assay SYNTAX v2` and
  `vectorSearch({ClosedCase.behaviour}, qv, k, {candidate_set: ...})`.
- `graph/queries/r2_ring_wcc.gsql` and `graph/queries/r2_case_io.gsql` also use
  `FOR GRAPH assay SYNTAX v2` so v2 edge pattern syntax `-(EDGE>)-` compiles cleanly.
- `install()` checks the REST++ endpoint list (`/restpp/endpoints/assay?dynamic=true`)
  and `LS` first, only creates/installs queries that are missing, and smoke-tests every
  installed query over REST++.
- `push_vectors()` writes `data/tg_r2/case_vectors.csv`, `docker cp`s it into
  `/home/tigergraph/data/tg_r2/case_vectors.csv`, and runs
  `RUN LOADING JOB load_case_vectors USING f_vec="/home/tigergraph/data/tg_r2/case_vectors.csv"`.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from assay.data import DATA, ROOT
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import case_pack, closed_cases, tx_all
from assay.r2.rings import ring_for
from assay.r2.vector import (
    EMBED_DIM,
    LocalVectorIndex,
    TigerGraphUnavailableError,
    embed_case,
    embed_closed_cases,
    export_tg_vectors,
)
from assay.tg import CONTAINER, GRAPH, HOST, device_id

TG_PORT = int(os.environ.get("ASSAY_TG_PORT", "14240"))
TG_USER = os.environ.get("ASSAY_TG_USER", "tigergraph")
TG_PASS = os.environ.get("ASSAY_TG_PASSWORD", "tigergraph")

REQUIRED_QUERIES: tuple[str, ...] = (
    "similar_cases_vec",
    "similar_cases_fused",
    "r2_ring_wcc",
    "r2_upsert_case",
    "r2_read_cases",
)


def _base_url(host: str | None = None, port: int | None = None) -> str:
    h = (host or os.environ.get("ASSAY_TG_HOST", HOST)).rstrip("/")
    p = port or int(os.environ.get("ASSAY_TG_PORT", str(TG_PORT)))
    if h.count(":") >= 2:
        return h
    return f"{h}:{p}"


def _auth() -> tuple[str, str]:
    user = os.environ.get("ASSAY_TG_USER", TG_USER)
    pw = os.environ.get("ASSAY_TG_PASSWORD", TG_PASS)
    return (user, pw)


def _graph_name_from_schema() -> str:
    schema_txt = (ROOT / "graph" / "schema.gsql").read_text()
    m = re.search(r"CREATE\s+GRAPH\s+(\w+)", schema_txt)
    return m.group(1) if m else GRAPH


def ping_tg(host: str | None = None, port: int | None = None, timeout: float = 5.0) -> bool:
    """Return True if TigerGraph REST++ is reachable and responsive, False otherwise."""
    try:
        url = f"{_base_url(host, port)}/restpp/echo"
        r = httpx.get(url, auth=_auth(), timeout=timeout)
        return r.status_code == 200 and not r.json().get("error", False)
    except Exception:
        return False


def require_tg(host: str | None = None, port: int | None = None, timeout: float = 5.0) -> None:
    """Fail closed with TigerGraphUnavailableError if TigerGraph is not reachable."""
    url = f"{_base_url(host, port)}/restpp/echo"
    try:
        r = httpx.get(url, auth=_auth(), timeout=timeout)
        if r.status_code != 200:
            raise TigerGraphUnavailableError(
                f"TigerGraph at {url} returned HTTP {r.status_code}: {r.text[:200]}"
            )
    except TigerGraphUnavailableError:
        raise
    except Exception as exc:
        raise TigerGraphUnavailableError(
            f"Fail-closed: TigerGraph is unreachable at {url} ({type(exc).__name__}: {exc})"
        ) from exc


def run_gsql_script(script_content: str, label: str = "gsql", timeout: float = 420.0) -> str:
    """Execute GSQL directly via `docker --context default exec -u tigergraph assay-tg
    /home/tigergraph/tigergraph/app/cmd/gsql <script_content>` and print the complete raw output.
    """
    require_tg(timeout=10.0)
    try:
        proc = subprocess.run(
            [
                "docker",
                "--context",
                "default",
                "exec",
                "-u",
                "tigergraph",
                CONTAINER,
                "/home/tigergraph/tigergraph/app/cmd/gsql",
                script_content,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        print(f"\n--- [GSQL:{label}] raw output ---\n{out.strip()}\n--- [END GSQL:{label}] ---")
        return out
    except Exception as exc:
        raise TigerGraphUnavailableError(f"Fail-closed: GSQL script '{label}' failed: {exc}") from exc


def _installed_query_names(gname: str) -> set[str]:
    """Return the set of currently installed REST++ query names on `gname`."""
    url_ep = f"{_base_url()}/restpp/endpoints/{gname}?dynamic=true"
    resp = httpx.get(url_ep, auth=_auth(), timeout=15.0)
    if resp.status_code != 200:
        return set()
    data = resp.json()
    installed: set[str] = set()
    prefix = f"/query/{gname}/"
    for key in data:
        if prefix in key:
            qname = key.split(prefix, 1)[1].split()[0].strip()
            installed.add(qname)
    return installed


def run_query(
    query_name: str,
    params: dict[str, Any] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Run an installed GSQL query via REST++ `/restpp/query/<graph>/<query_name>`.
    Fails closed (`TigerGraphUnavailableError`) on any connection, HTTP, or GSQL error.
    """
    base = _base_url(host, port)
    gname = _graph_name_from_schema()
    url = f"{base}/restpp/query/{gname}/{query_name}"
    qparams: list[tuple[str, str]] = []
    if params:
        for k, v in params.items():
            if isinstance(v, (list, tuple, set)):
                if not v:
                    # REST++ reads an absent SET parameter as NULL; send one "" that the query skips
                    qparams.append((k, ""))
                for item in v:
                    if isinstance(item, tuple) and len(item) == 1:
                        qparams.append((k, str(float(item[0]) if isinstance(item[0], np.floating) else item[0])))
                    elif isinstance(item, (float, np.floating)):
                        qparams.append((k, f"{float(item):.6f}"))
                    else:
                        qparams.append((k, str(item)))
            elif isinstance(v, bool):
                qparams.append((k, "true" if v else "false"))
            elif isinstance(v, (float, np.floating)):
                qparams.append((k, f"{float(v):.6f}"))
            else:
                qparams.append((k, str(v)))

    try:
        resp = httpx.get(
            url,
            params=qparams,
            auth=_auth(),
            headers={"GSQL-TIMEOUT": str(int(timeout * 1000))},
            timeout=timeout,
        )
    except Exception as exc:
        raise TigerGraphUnavailableError(
            f"Fail-closed: TigerGraph REST++ query '{query_name}' at {url} failed: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise TigerGraphUnavailableError(
            f"Fail-closed: TigerGraph query '{query_name}' HTTP {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    if data.get("error"):
        raise TigerGraphUnavailableError(
            f"Fail-closed: TigerGraph query '{query_name}' error {data.get('code')}: {data.get('message')}"
        )

    results = data.get("results", [])
    if not isinstance(results, list):
        return []
    return results


def _ensure_january_loaded(gname: str) -> dict[str, int]:
    """Ensure January 2017 transactions and the 5,585 closed cases are loaded into TigerGraph.
    Upsert-only; never drops or deletes Round 1 vertices.
    """
    require_tg()
    url = f"{_base_url()}/restpp/builtins/{gname}"
    resp = httpx.post(
        url,
        json={"function": "stat_vertex_number", "type": "*"},
        auth=_auth(),
        timeout=15.0,
    )
    if resp.status_code != 200 or resp.json().get("error"):
        raise TigerGraphUnavailableError(f"Failed to read vertex counts: {resp.text[:200]}")

    counts = {r["v_type"]: int(r["count"]) for r in resp.json().get("results", [])}
    if counts.get("Txn", 0) < 650000 or counts.get("ClosedCase", 0) < 5585:
        from assay.r2 import tg_load

        tg_load.prepare()
        load_sql = (ROOT / "graph" / "load_r2.gsql").read_text()
        clean_lines = [ln for ln in load_sql.splitlines() if not ln.strip().upper().startswith("DROP")]
        run_gsql_script("\n".join(clean_lines), label="load_assay_r2")
        resp2 = httpx.post(
            url,
            json={"function": "stat_vertex_number", "type": "*"},
            auth=_auth(),
            timeout=15.0,
        )
        counts = {r["v_type"]: int(r["count"]) for r in resp2.json().get("results", [])}
    return counts


def _strip_drops_and_installs(gsql_text: str) -> str:
    """Keep GSQL script strictly additive (no DROP commands) and strip trailing
    INSTALL lines so we can batch-install missing queries in a single statement.
    """
    lines: list[str] = []
    for ln in gsql_text.splitlines():
        up = ln.strip().upper()
        if up.startswith("DROP ") or up.startswith("INSTALL QUERY"):
            continue
        lines.append(ln)
    return "\n".join(lines)


def smoke_test_queries() -> dict[str, float]:
    """Run every required R2 query over REST++ with minimal params, print raw responses,
    and return latency in milliseconds.
    Raises TigerGraphUnavailableError with full response body if any query 404s or fails.
    """
    require_tg()
    timings_ms: dict[str, float] = {}
    zero_qv = [round(float(1.0 / np.sqrt(EMBED_DIM)), 6)] * EMBED_DIM

    # 1. similar_cases_vec
    t0 = time.perf_counter()
    res_vec = run_query(
        "similar_cases_vec",
        {"qv": zero_qv, "k": 3, "as_of": "2017-01-15 00:00:00"},
    )
    timings_ms["similar_cases_vec"] = round((time.perf_counter() - t0) * 1000.0, 2)
    print(
        f"[SMOKE] similar_cases_vec ({timings_ms['similar_cases_vec']}ms) -> "
        f"keys={list(res_vec[0].keys()) if res_vec else []}"
    )
    if not res_vec or "Hits" not in res_vec[0]:
        raise TigerGraphUnavailableError(f"Smoke test failed for similar_cases_vec: {res_vec}")

    # 2. similar_cases_fused
    t0 = time.perf_counter()
    res_fused = run_query(
        "similar_cases_fused",
        {"c": "C05061-K1", "qv": zero_qv, "k": 3, "hops": 2, "as_of": "2017-01-15 00:00:00"},
    )
    timings_ms["similar_cases_fused"] = round((time.perf_counter() - t0) * 1000.0, 2)
    print(
        f"[SMOKE] similar_cases_fused ({timings_ms['similar_cases_fused']}ms) -> "
        f"keys={list(res_fused[0].keys()) if res_fused else []}"
    )
    if not res_fused or "R" not in res_fused[0]:
        raise TigerGraphUnavailableError(f"Smoke test failed for similar_cases_fused: {res_fused}")

    # 3. r2_ring_wcc
    t0 = time.perf_counter()
    res_ring = run_query(
        "r2_ring_wcc",
        {"seed_card": "C05061-K1", "as_of": "2017-01-15 01:25:19", "window_days": 30, "cap": 15},
    )
    timings_ms["r2_ring_wcc"] = round((time.perf_counter() - t0) * 1000.0, 2)
    print(f"[SMOKE] r2_ring_wcc ({timings_ms['r2_ring_wcc']}ms) -> {res_ring[0] if res_ring else {}}")
    if not res_ring or "members" not in res_ring[0]:
        raise TigerGraphUnavailableError(f"Smoke test failed for r2_ring_wcc: {res_ring}")

    # 4. r2_read_cases
    t0 = time.perf_counter()
    res_read = run_query("r2_read_cases", {"card": "C05061-K1"})
    timings_ms["r2_read_cases"] = round((time.perf_counter() - t0) * 1000.0, 2)
    print(f"[SMOKE] r2_read_cases ({timings_ms['r2_read_cases']}ms) -> keys={[list(x.keys()) for x in res_read]}")
    if not res_read or "FraudCases" not in res_read[0]:
        raise TigerGraphUnavailableError(f"Smoke test failed for r2_read_cases: {res_read}")

    return timings_ms


_VECTOR_MODE: str = "native_vector"


def install() -> dict[str, Any]:
    """Idempotently verify schema and install any missing R2 queries:
    1. Inspect `LS`, `SHOW QUERY *`, and `/restpp/endpoints/assay?dynamic=true` first.
    2. Do not re-run schema changes or queries that are already installed.
    3. Only CREATE + INSTALL what is missing (`r2_ring_wcc`, `r2_upsert_case`, `r2_read_cases`,
       or vector queries if not yet installed), printing every GSQL response.
    4. Smoke-test every required query over REST++.
    """
    require_tg()
    gname = _graph_name_from_schema()

    # Step 1: Check currently installed endpoints & schema
    installed_before = _installed_query_names(gname)
    print(f"\n[INSTALL] Currently installed REST++ queries on '{gname}': {sorted(installed_before)}")

    ls_out = run_gsql_script(f"USE GRAPH {gname}\nLS\nSHOW QUERY *\n", label="step1_ls_and_show_query")
    counts = _ensure_january_loaded(gname)

    # Step 2: Check vector attribute `behaviour` and loading job `load_case_vectors`
    if "behaviour" in ls_out:
        print("[INSTALL] Vector attribute `ClosedCase.behaviour` already present; skipping schema change.")
    else:
        vec_schema_sql = (ROOT / "graph" / "r2_vector_schema.gsql").read_text()
        run_gsql_script(vec_schema_sql, label="step2_vector_schema_change")

    if "load_case_vectors" not in ls_out:
        load_vec_sql = f"""USE GRAPH {gname}
CREATE LOADING JOB load_case_vectors FOR GRAPH {gname} {{
  DEFINE FILENAME f_vec = "/home/tigergraph/data/tg_r2/case_vectors.csv";
  LOAD f_vec TO VECTOR ATTRIBUTE behaviour ON VERTEX ClosedCase VALUES ($0, SPLIT($1, ",")) USING SEPARATOR="|", HEADER="true", EOL="\\n";
}}
"""
        run_gsql_script(load_vec_sql, label="step2_create_load_case_vectors")

    # Step 3: Determine which required queries are not yet installed
    missing_queries = [q for q in REQUIRED_QUERIES if q not in installed_before]
    print(f"[INSTALL] Missing queries to CREATE + INSTALL: {missing_queries}")

    if missing_queries:
        if any(q in missing_queries for q in ("similar_cases_vec", "similar_cases_fused")):
            vec_sql = _strip_drops_and_installs((ROOT / "graph" / "queries" / "r2_vector.gsql").read_text())
            vec_out = run_gsql_script(vec_sql, label="step3_create_r2_vector")
            if "Successfully created queries" not in vec_out:
                raise TigerGraphUnavailableError(f"Failed to CREATE r2_vector.gsql queries:\n{vec_out}")

        if "r2_ring_wcc" in missing_queries:
            ring_sql = _strip_drops_and_installs((ROOT / "graph" / "queries" / "r2_ring_wcc.gsql").read_text())
            ring_out = run_gsql_script(ring_sql, label="step3_create_r2_ring_wcc")
            if "Successfully created queries" not in ring_out:
                raise TigerGraphUnavailableError(f"Failed to CREATE r2_ring_wcc.gsql:\n{ring_out}")

        if any(q in missing_queries for q in ("r2_upsert_case", "r2_read_cases")):
            case_io_sql = _strip_drops_and_installs((ROOT / "graph" / "queries" / "r2_case_io.gsql").read_text())
            case_io_out = run_gsql_script(case_io_sql, label="step3_create_r2_case_io")
            if "Successfully created queries" not in case_io_out:
                raise TigerGraphUnavailableError(f"Failed to CREATE r2_case_io.gsql:\n{case_io_out}")

        # Install all missing queries in one INSTALL QUERY statement
        q_csv = ", ".join(missing_queries)
        inst_out = run_gsql_script(
            f"USE GRAPH {gname}\nINSTALL QUERY {q_csv}\n",
            label=f"step3_install_missing_queries({q_csv})",
            timeout=480.0,
        )
        if "error" in inst_out.lower() or "failed" in inst_out.lower():
            raise TigerGraphUnavailableError(f"INSTALL QUERY {q_csv} failed:\n{inst_out}")

    installed_after = _installed_query_names(gname)
    print(f"[INSTALL] Installed REST++ queries after install: {sorted(installed_after)}")
    still_missing = [q for q in REQUIRED_QUERIES if q not in installed_after]
    if still_missing:
        raise TigerGraphUnavailableError(
            f"Queries still missing from REST++ endpoints after INSTALL: {still_missing}"
        )

    # Step 4: Smoke-test all installed queries via REST++
    timings = smoke_test_queries()
    return {
        "graph": gname,
        "vertex_counts": counts,
        "vector_attribute_present": True,
        "vector_mode": _VECTOR_MODE,
        "installed_queries": list(REQUIRED_QUERIES),
        "smoke_test_ms": timings,
    }


def push_vectors(cc_emb: pd.DataFrame | None = None) -> dict[str, Any]:
    """Export 24-dim causal case embeddings for all 5,585 closed cases to
    `data/tg_r2/case_vectors.csv`, copy it into the container at
    `/home/tigergraph/data/tg_r2/case_vectors.csv`, and run `load_case_vectors`.
    """
    require_tg()
    gname = _graph_name_from_schema()
    t0 = time.perf_counter()
    if cc_emb is None:
        cc_emb = embed_closed_cases()
    csv_path = export_tg_vectors(cc_emb)

    container_csv = "/home/tigergraph/data/tg_r2/case_vectors.csv"
    # The container mounts the repo's data/ read-only at /home/tigergraph/data (see the
    # assay-tg run command), so a CSV written under data/tg_r2 is already visible to the loader.
    if not str(csv_path.resolve()).startswith(str((ROOT / "data").resolve())):
        raise TigerGraphUnavailableError(
            f"Fail-closed: {csv_path} is outside data/, which is the only directory mounted into {CONTAINER}.")

    out = run_gsql_script(
        f'USE GRAPH {gname}\nRUN LOADING JOB load_case_vectors USING f_vec="{container_csv}"\n',
        label="push_vectors_run_load_case_vectors",
        timeout=240.0,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)

    # Verify via similar_cases_vec that vector search returns populated hits
    sample_vec = [round(float(x), 6) for x in cc_emb["embedding"].iloc[0].tolist()]
    res = run_query(
        "similar_cases_vec",
        {"qv": sample_vec, "k": 5, "as_of": "2017-02-01 00:00:00"},
    )
    hits = res[0].get("Hits", []) if res else []
    print(f"[PUSH_VECTORS] sample query returned {len(hits)} hits: {[h.get('v_id') for h in hits]}")
    if not hits:
        raise TigerGraphUnavailableError(
            f"push_vectors verification failed: similar_cases_vec returned 0 hits. Log: {out[-500:]}"
        )

    return {
        "n_vectors_pushed": len(cc_emb),
        "csv_path": str(csv_path),
        "elapsed_sec": elapsed_s,
        "sample_top1_case_id": str(hits[0].get("v_id", "")),
        "vector_mode": _VECTOR_MODE,
    }


def _resolve_case_card_and_embedding(
    case_row: pd.Series | dict[str, Any] | Any,
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
) -> tuple[str, str, str, pd.Timestamp, np.ndarray, list[int]]:
    """Resolve actual card_id (handling HHF-011 C09241-K1 -> C09241-K2), opened_at,
    24-dim query embedding `qv`, and episode transaction IDs as-of `opened_at`.
    """
    if tx is None:
        tx = tx_all()
    if cc is None:
        cc = closed_cases()

    if isinstance(case_row, dict):
        case_id = str(case_row["case_id"])
        card_id_raw = str(case_row["card_id"])
        customer_id = str(case_row.get("customer_id", card_id_raw.split("-")[0]))
        opened_at = pd.Timestamp(case_row["opened_at"])
        flagged_txn_id = int(case_row.get("flagged_txn_id", 0))
    else:
        case_id = str((getattr(case_row, "case_id") if hasattr(case_row, "case_id") else case_row["case_id"]))
        card_id_raw = str((getattr(case_row, "card_id") if hasattr(case_row, "card_id") else case_row["card_id"]))
        customer_id = str(getattr(case_row, "customer_id", card_id_raw.split("-")[0]))
        opened_at = pd.Timestamp((getattr(case_row, "opened_at") if hasattr(case_row, "opened_at") else case_row["opened_at"]))
        flagged_txn_id = int(getattr(case_row, "flagged_txn_id", 0) or 0)

    _, std = embed_closed_cases(cc, tx, return_standardizer=True)
    tx_by_id = tx.set_index("TransactionID", drop=False)

    if flagged_txn_id and flagged_txn_id in tx_by_id.index:
        actual_card = str(tx_by_id.loc[flagged_txn_id, "card_id"])
    else:
        actual_card = card_id_raw

    asof_tx = tx[(tx["card_id"] == actual_card) & (tx["t"] <= opened_at)]
    if flagged_txn_id and flagged_txn_id in tx_by_id.index:
        t0 = pd.Timestamp(tx_by_id.loc[flagged_txn_id, "t"])
        win_start = t0 - pd.Timedelta(hours=24)
        ep_tx = asof_tx[(asof_tx["t"] >= win_start) & (asof_tx["t"] <= opened_at)]
        if ep_tx.empty:
            ep_tx = tx_by_id.loc[[flagged_txn_id]]
    else:
        ep_tx = asof_tx.tail(1)

    hist_tx = asof_tx[~asof_tx["TransactionID"].isin(ep_tx["TransactionID"])]
    devs = {str(d) for d in ep_tx["device_profile"].fillna("").tolist() if d}
    if devs:
        dev_tx = tx[(tx["device_profile"].isin(devs)) & (tx["t"] <= opened_at) & (tx["card_id"] != actual_card)]
        n_shared = int(dev_tx["card_id"].nunique())
    else:
        n_shared = 0

    qv = embed_case(ep_tx, hist_tx, standardizer=std, asof_shared_dev_cards=n_shared)
    ep_tids = ep_tx["TransactionID"].astype(int).tolist()
    return case_id, actual_card, customer_id, opened_at, qv, ep_tids


def vector_precedent(
    case_row: pd.Series | dict[str, Any] | Any,
    k: int = 5,
    tx: pd.DataFrame | None = None,
    cc: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run TigerVector top-k as-of (`similar_cases_vec`) and fused 2-hop traversal +
    vector search (`similar_cases_fused`) on TigerGraph for `case_row`.
    Fails closed (`TigerGraphUnavailableError`) if TigerGraph is down.
    """
    require_tg()
    case_id, actual_card, customer_id, opened_at, qv, ep_tids = _resolve_case_card_and_embedding(
        case_row, tx=tx, cc=cc
    )
    as_of_str = opened_at.strftime("%Y-%m-%d %H:%M:%S")
    qv_list = [round(float(x), 6) for x in qv.tolist()]

    t0 = time.perf_counter()
    vec_res = run_query(
        "similar_cases_vec",
        {"qv": qv_list, "k": int(k), "as_of": as_of_str},
    )
    vec_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    t1 = time.perf_counter()
    fused_res = run_query(
        "similar_cases_fused",
        {"c": actual_card, "qv": qv_list, "k": int(k), "hops": 2, "as_of": as_of_str},
    )
    fused_ms = round((time.perf_counter() - t1) * 1000.0, 2)

    raw_vec_hits = vec_res[0].get("Hits", []) if vec_res else []
    raw_fused_hits = fused_res[0].get("R", []) if fused_res else []

    def _parse_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for r in rows:
            vid = str(r.get("v_id", r.get("case_id", "")))
            if vid == case_id:
                continue
            attrs = r.get("attributes", r)
            parsed.append(
                {
                    "case_id": vid,
                    "outcome": str(attrs.get("outcome", attrs.get("Hits.outcome", attrs.get("R.outcome", "")))),
                    "pattern": str(attrs.get("pattern", attrs.get("Hits.pattern", attrs.get("R.pattern", "")))),
                    "exposure_usd": float(
                        attrs.get("exposure_usd", attrs.get("Hits.exposure_usd", attrs.get("R.exposure_usd", 0.0)))
                    ),
                    "card_id": str(attrs.get("card_id", attrs.get("Hits.card_id", attrs.get("R.card_id", "")))),
                    "closed_at": str(
                        attrs.get("closed_at", attrs.get("Hits.closed_at", attrs.get("R.closed_at", "")))
                    ),
                    "connecting_path": str(attrs.get("connecting_path", "")),
                    "shared_cards": sorted(str(x) for x in attrs.get("shared_cards", []) if x),
                    "shared_devices": sorted(str(x) for x in attrs.get("shared_devices", []) if x),
                }
            )
        return parsed[:k]

    vec_hits = _parse_hits(raw_vec_hits)
    fused_hits = _parse_hits(raw_fused_hits)
    effective_hits = fused_hits if fused_hits else vec_hits

    shared_cards = sorted({c for h in effective_hits for c in h["shared_cards"] if c != actual_card})
    prec_ids = [h["case_id"] for h in effective_hits]

    attestation = Attestation(
        what=(
            f"TigerVector precedent search for {case_id} (card {actual_card}) as-of {as_of_str} "
            f"returned {len(vec_hits)} global vector hits ({', '.join(h['case_id'] for h in vec_hits[:3])}) "
            f"and {len(fused_hits)} 2-hop fused hits ({', '.join(h['case_id'] for h in fused_hits[:3])})."
        ),
        evidence=Evidence(
            txn_ids=ep_tids,
            card_ids=[actual_card] + shared_cards[:5],
            customer_ids=[customer_id],
            case_ids=prec_ids,
        ),
        query="similar_cases_fused" if fused_hits else "similar_cases_vec",
        params={"c": actual_card, "qv": [round(float(x), 4) for x in qv_list], "k": int(k), "as_of": as_of_str},
    )

    return {
        "case_id": case_id,
        "card_id": actual_card,
        "as_of": as_of_str,
        "qv": qv_list,
        "vec_hits": vec_hits,
        "fused_hits": fused_hits,
        "effective_hits": effective_hits,
        "vec_ms": vec_ms,
        "fused_ms": fused_ms,
        "attestation": attestation,
    }


def ring_wcc(
    card: str,
    t: pd.Timestamp | str,
    window_days: int = 30,
    cap: int = 15,
    max_ring_size: int = 30,
) -> dict[str, Any]:
    """Run installed `r2_ring_wcc` query on TigerGraph as-of timestamp `t` and return
    the connected component members, shared elements, prior fraud members, and attestation.
    Fails closed (`TigerGraphUnavailableError`) if TigerGraph is unreachable.
    """
    require_tg()
    as_of_str = pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    res = run_query(
        "r2_ring_wcc",
        {
            "seed_card": str(card),
            "as_of": as_of_str,
            "window_days": int(window_days),
            "cap": int(cap),
        },
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    row: dict[str, Any] = {}
    for item in res:
        if isinstance(item, dict):
            row.update(item)

    members = sorted(str(x) for x in row.get("members", []) if x)
    shared_elements = sorted(str(x) for x in row.get("shared_elements", []) if x)
    txn_ids = sorted(int(x) for x in row.get("txn_ids", []) if str(x).isdigit())
    total_amount = round(float(row.get("total_amount", 0.0)), 2)
    prior_fraud_cards = sorted(str(x) for x in row.get("prior_fraud_cards", []) if x)
    prior_case_ids = sorted(str(x) for x in row.get("prior_case_ids", []) if x)

    max_risk = float(row.get("max_risk", 0.0) or 0.0)
    # Same ring rule as assay.r2.rings.ring_for: 3..max_ring_size members, and a member with
    # prior confirmed fraud or a bank score >= 0.70 in the window.
    qualifies = (3 <= len(members) <= max_ring_size) and (len(prior_fraud_cards) > 0 or max_risk >= 0.70)
    ring_id = f"RING-WCC-{hashlib.sha256(','.join(members).encode()).hexdigest()[:8]}"

    attestation = Attestation(
        what=(
            f"Card {card} is in wcc ring {ring_id} containing {len(members)} cards "
            f"(${total_amount:,.2f} vol, {len(prior_fraud_cards)} prior fraud members)"
        ),
        evidence=Evidence(
            txn_ids=txn_ids,
            card_ids=members,
            customer_ids=sorted({c.split("-")[0] for c in members}),
            case_ids=prior_case_ids,
        ),
        query="r2_ring_wcc",
        params={
            "seed_card": str(card),
            "as_of": as_of_str,
            "window_days": int(window_days),
            "cap": int(cap),
        },
    )

    return {
        "seed_card": str(card),
        "as_of": as_of_str,
        "members": members,
        "qualifies_as_ring": qualifies,
        "ring_id": ring_id,
        "max_risk": max_risk,
        "shared_elements": shared_elements,
        "txn_ids": txn_ids,
        "total_amount": total_amount,
        "prior_fraud_cards": prior_fraud_cards,
        "prior_case_ids": prior_case_ids,
        "elapsed_ms": elapsed_ms,
        "attestation": attestation,
    }


def write_case(answer: dict[str, Any], card_id: str = "") -> dict[str, Any]:
    """Upsert an investigated case vertex (`FraudCase`) and its edges (`ABOUT_CARD`,
    `CITES`, `LINKS_CARD`, `IMPLICATES`, `DREW_ON`) into TigerGraph via `r2_upsert_case`.
    Fails closed (`TigerGraphUnavailableError`) if TigerGraph is down.
    """
    require_tg()
    alert_id = str(answer.get("case_id", "UNKNOWN"))
    gid = alert_id if alert_id.startswith("ASSAY-") else f"ASSAY-{alert_id}"
    c = answer.get("case", answer)

    resolved_card = str(card_id or c.get("card_id") or answer.get("card_id") or "")
    status = str(c.get("status", "closed"))
    verdict = str(c.get("verdict", "uncertain"))
    fraud_prob = float(c.get("fraud_probability", 0.5))
    pattern = str(c.get("pattern", "none"))
    pattern_desc = str(c.get("pattern_description", ""))
    exposure_usd = float(c.get("exposure_usd", 0.0))
    summary = str(c.get("summary", ""))
    stop_reason = str(answer.get("stop_reason", c.get("stop_reason", "completed")))

    sar_block = answer.get("sar", {})
    sar_filed = bool(sar_block.get("file", False)) if isinstance(sar_block, dict) else bool(c.get("sar_filed", False))

    nba = answer.get("next_best_actions", {})
    if isinstance(nba, dict) and "final" in nba:
        final_actions = "|".join(
            str(x["action"]) if isinstance(x, dict) else str(x) for x in nba["final"]
        )
    else:
        final_actions = str(c.get("final_actions", ""))

    written_at = str(c.get("written_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

    txn_ids = [str(int(t)) for t in c.get("affected_txn_ids", []) if str(t) != ""]
    connected_cards = [str(k) for k in c.get("connected_card_ids", []) if k]
    dev_profiles = [str(p) for p in c.get("connected_device_profiles", []) if p]
    dev_ids = [p if p.startswith("D") and len(p) == 11 else device_id(p) for p in dev_profiles]
    dev_ids = [d for d in dev_ids if d]
    prior_cases = [str(p) for p in c.get("similar_prior_cases", []) if p]

    t0 = time.perf_counter()
    res = run_query(
        "r2_upsert_case",
        {
            "case_id": gid,
            "alert_id": alert_id,
            "card_id": resolved_card,
            "status": status,
            "verdict": verdict,
            "fraud_probability": fraud_prob,
            "pattern": pattern,
            "pattern_description": pattern_desc,
            "exposure_usd": exposure_usd,
            "summary": summary,
            "stop_reason": stop_reason,
            "sar_filed": sar_filed,
            "final_actions": final_actions,
            "written_at": written_at,
            "txn_ids": txn_ids,
            "connected_cards": connected_cards,
            "device_ids": dev_ids,
            "prior_cases": prior_cases,
        },
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    written_id = str(res[0].get("written_case_id", gid)) if res else gid

    return {
        "graph_case_id": written_id,
        "alert_id": alert_id,
        "card_id": resolved_card,
        "elapsed_ms": elapsed_ms,
    }


def read_cases(card: str) -> dict[str, Any]:
    """Read back both agent-investigated `FraudCase` records and historical `ClosedCase`
    records linked to `card` via installed GSQL query `r2_read_cases`.
    Fails closed (`TigerGraphUnavailableError`) if TigerGraph is down.
    """
    require_tg()
    t0 = time.perf_counter()
    res = run_query("r2_read_cases", {"card": str(card)})
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    fraud_raw: list[dict[str, Any]] = []
    closed_raw: list[dict[str, Any]] = []
    for item in res:
        if "FraudCases" in item:
            fraud_raw = item["FraudCases"]
        if "ClosedCases" in item:
            closed_raw = item["ClosedCases"]

    fraud_cases: list[dict[str, Any]] = []
    for r in fraud_raw:
        vid = str(r.get("v_id", r.get("case_id", "")))
        attrs = r.get("attributes", r)
        fraud_cases.append(
            {
                "case_id": vid,
                "alert_id": str(attrs.get("FraudCases.alert_id", attrs.get("alert_id", ""))),
                "status": str(attrs.get("FraudCases.status", attrs.get("status", ""))),
                "verdict": str(attrs.get("FraudCases.verdict", attrs.get("verdict", ""))),
                "fraud_probability": float(
                    attrs.get("FraudCases.fraud_probability", attrs.get("fraud_probability", 0.0))
                ),
                "pattern": str(attrs.get("FraudCases.pattern", attrs.get("pattern", ""))),
                "pattern_description": str(
                    attrs.get("FraudCases.pattern_description", attrs.get("pattern_description", ""))
                ),
                "exposure_usd": float(attrs.get("FraudCases.exposure_usd", attrs.get("exposure_usd", 0.0))),
                "summary": str(attrs.get("FraudCases.summary", attrs.get("summary", ""))),
                "stop_reason": str(attrs.get("FraudCases.stop_reason", attrs.get("stop_reason", ""))),
                "sar_filed": bool(attrs.get("FraudCases.sar_filed", attrs.get("sar_filed", False))),
                "final_actions": str(attrs.get("FraudCases.final_actions", attrs.get("final_actions", ""))),
                "written_at": str(attrs.get("FraudCases.written_at", attrs.get("written_at", ""))),
                "cited_txns": sorted(int(x) for x in attrs.get("cited_txns", []) if str(x).isdigit()),
                "linked_cards": sorted(str(x) for x in attrs.get("linked_cards", []) if x),
                "implicated_devices": sorted(str(x) for x in attrs.get("implicated_devices", []) if x),
                "drew_on_cases": sorted(str(x) for x in attrs.get("drew_on_cases", []) if x),
            }
        )

    closed_list: list[dict[str, Any]] = []
    for r in closed_raw:
        vid = str(r.get("v_id", r.get("case_id", "")))
        attrs = r.get("attributes", r)
        closed_list.append(
            {
                "case_id": vid,
                "card_id": str(attrs.get("card_id", "")),
                "outcome": str(attrs.get("outcome", "")),
                "pattern": str(attrs.get("pattern", "")),
                "exposure_usd": float(attrs.get("exposure_usd", 0.0)),
                "closed_at": str(attrs.get("closed_at", "")),
            }
        )

    return {
        "card_id": str(card),
        "fraud_cases": sorted(fraud_cases, key=lambda x: x["case_id"]),
        "closed_cases": sorted(closed_list, key=lambda x: x["case_id"]),
        "elapsed_ms": elapsed_ms,
    }


def delete_fraud_case(case_id: str) -> None:
    """Delete a test FraudCase vertex by primary ID via REST++ so tests leave clean state."""
    require_tg()
    gname = _graph_name_from_schema()
    gid = case_id if case_id.startswith("ASSAY-") else f"ASSAY-{case_id}"
    url = f"{_base_url()}/restpp/graph/{gname}/vertices/FraudCase/{gid}"
    httpx.delete(url, auth=_auth(), timeout=15.0)


@dataclass
class ParityCaseRow:
    case_id: str
    card_id: str
    opened_at: str
    tg_vec_ids: list[str]
    local_vec_ids: list[str]
    vec_overlap_at_k: int
    vec_top1_match: bool
    tg_fused_ids: list[str]
    local_fused_ids: list[str]
    fused_overlap_at_k: int
    tg_ring_members: list[str]
    local_ring_members: list[str]
    ring_exact_match: bool
    ring_jaccard: float
    vec_ms: float
    fused_ms: float
    ring_ms: float


def parity(k: int = 5) -> dict[str, Any]:
    """Compare live TigerGraph results (`vector_precedent`, `ring_wcc`) against the
    local pandas/networkx implementations (`LocalVectorIndex`, `ring_for`) across all
    18 exam cases in `case_pack()`. Prints a formatted comparison table and any
    mismatches with exact IDs.
    """
    require_tg()
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    cc_emb = embed_closed_cases(cc, tx)
    local_idx = LocalVectorIndex(cc_emb, cc, tx)

    rows: list[ParityCaseRow] = []
    vec_mismatches: list[str] = []
    ring_mismatches: list[str] = []

    for r in cp.itertuples():
        # 1. Live TigerGraph vector precedent
        tg_vp = vector_precedent(r, k=k, tx=tx, cc=cc)
        actual_card = tg_vp["card_id"]
        opened_at = pd.Timestamp(r.opened_at)
        qv = np.array(tg_vp["qv"], dtype=np.float64)

        # 2. Local vector index precedent
        loc_vec = local_idx.similar_cases_vec(
            qv, k=k, as_of=opened_at, exclude_case_id=r.case_id, use_tg_if_enabled=False
        )
        loc_fused = local_idx.similar_cases_fused(
            card_id=actual_card,
            qv=qv,
            k=k,
            hops=2,
            as_of=opened_at,
            exclude_case_id=r.case_id,
            use_tg_if_enabled=False,
        )

        tg_vec_ids = [h["case_id"] for h in tg_vp["vec_hits"]]
        loc_vec_ids = [h.case_id for h in loc_vec]
        vec_overlap = len(set(tg_vec_ids) & set(loc_vec_ids))
        vec_top1_match = bool(tg_vec_ids and loc_vec_ids and tg_vec_ids[0] == loc_vec_ids[0])
        if set(tg_vec_ids) != set(loc_vec_ids):
            vec_mismatches.append(
                f"{r.case_id} ({actual_card}): TG_vec={tg_vec_ids} vs Local_vec={loc_vec_ids}"
            )

        tg_fused_ids = [h["case_id"] for h in tg_vp["effective_hits"]]
        loc_fused_ids = [h.case_id for h in loc_fused]
        fused_overlap = len(set(tg_fused_ids) & set(loc_fused_ids))

        # 3. Live TigerGraph ring_wcc vs local ring_for
        tg_rw = ring_wcc(actual_card, opened_at, window_days=30, cap=15)
        loc_ring = ring_for(actual_card, opened_at, tx=tx, cc=cc, window_days=30, cardinality_cap=15, algo="wcc")

        # Compare the decision: the qualifying ring, or just the card when there is none
        tg_members = tg_rw["members"] if tg_rw["qualifies_as_ring"] else [actual_card]
        loc_members = loc_ring.cards if loc_ring is not None else [actual_card]

        set_tg = set(tg_members)
        set_loc = set(loc_members)
        inter = len(set_tg & set_loc)
        union = len(set_tg | set_loc)
        jacc = round(inter / union, 4) if union > 0 else 1.0
        exact_ring = set_tg == set_loc
        if not exact_ring:
            ring_mismatches.append(
                f"{r.case_id} ({actual_card}): TG_ring={sorted(set_tg)} vs Local_ring={sorted(set_loc)}"
            )

        rows.append(
            ParityCaseRow(
                case_id=str(r.case_id),
                card_id=actual_card,
                opened_at=str(r.opened_at),
                tg_vec_ids=tg_vec_ids,
                local_vec_ids=loc_vec_ids,
                vec_overlap_at_k=vec_overlap,
                vec_top1_match=vec_top1_match,
                tg_fused_ids=tg_fused_ids,
                local_fused_ids=loc_fused_ids,
                fused_overlap_at_k=fused_overlap,
                tg_ring_members=tg_members,
                local_ring_members=loc_members,
                ring_exact_match=exact_ring,
                ring_jaccard=jacc,
                vec_ms=tg_vp["vec_ms"],
                fused_ms=tg_vp["fused_ms"],
                ring_ms=tg_rw["elapsed_ms"],
            )
        )

    # Print formatted parity table
    header = (
        f"{'Case':<8} {'Card':<10} {'TG_Vec_Top1':<10} {'Loc_Vec_Top1':<12} "
        f"{'Vec_Ov@5':<8} {'Fused_Ov@5':<10} {'TG_Ring#':<8} {'Loc_Ring#':<9} "
        f"{'Ring_Jacc':<9} {'Vec_ms':>7} {'Fused_ms':>8} {'Ring_ms':>7}"
    )
    print("\n=== N2 TigerGraph vs Local Parity (18 Exam Cases) ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        tv1 = row.tg_vec_ids[0] if row.tg_vec_ids else "-"
        lv1 = row.local_vec_ids[0] if row.local_vec_ids else "-"
        print(
            f"{row.case_id:<8} {row.card_id:<10} {tv1:<10} {lv1:<12} "
            f"{row.vec_overlap_at_k:<8} {row.fused_overlap_at_k:<10} "
            f"{len(row.tg_ring_members):<8} {len(row.local_ring_members):<9} "
            f"{row.ring_jaccard:<9.2f} {row.vec_ms:>7.1f} {row.fused_ms:>8.1f} {row.ring_ms:>7.1f}"
        )

    if vec_mismatches:
        print("\nVector top-k differences (HNSW approximate vs exact brute-force):")
        for m in vec_mismatches:
            print("  ", m)
    if ring_mismatches:
        print("\nRing WCC differences (qualifying ring, TG vs local):")
        for m in ring_mismatches:
            print("  ", m)

    summary = {
        "n_cases": len(rows),
        "vec_top1_agreement": sum(1 for r in rows if r.vec_top1_match),
        "mean_vec_overlap_at_k": round(float(np.mean([r.vec_overlap_at_k for r in rows])), 2),
        "mean_fused_overlap_at_k": round(float(np.mean([r.fused_overlap_at_k for r in rows])), 2),
        "ring_exact_agreement": sum(1 for r in rows if r.ring_exact_match),
        "mean_ring_jaccard": round(float(np.mean([r.ring_jaccard for r in rows])), 3),
        "mean_vec_ms": round(float(np.mean([r.vec_ms for r in rows])), 2),
        "mean_fused_ms": round(float(np.mean([r.fused_ms for r in rows])), 2),
        "mean_ring_ms": round(float(np.mean([r.ring_ms for r in rows])), 2),
        "p95_vec_ms": round(float(np.percentile([r.vec_ms for r in rows], 95)), 2),
        "p95_fused_ms": round(float(np.percentile([r.fused_ms for r in rows], 95)), 2),
        "p95_ring_ms": round(float(np.percentile([r.ring_ms for r in rows], 95)), 2),
        "rows": rows,
        "vec_mismatches": vec_mismatches,
        "ring_mismatches": ring_mismatches,
    }
    return summary
