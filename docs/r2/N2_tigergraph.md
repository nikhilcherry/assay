# Block N2 — TigerGraph as the Live Source of Truth

## Round 3 Fixes Applied

1. **`SYNTAX v2` in `r2_ring_wcc.gsql` and `r2_case_io.gsql`**:
   - Added `FOR GRAPH assay SYNTAX v2` to `r2_ring_wcc`, `r2_upsert_case`, and `r2_read_cases` so multi-hop edge patterns `-(EDGE>)-` compile cleanly alongside `r2_vector.gsql`.
2. **Idempotent `install()` without redundant work**:
   - `install()` inspects `/restpp/endpoints/assay?dynamic=true` and `USE GRAPH assay; LS; SHOW QUERY *` first.
   - Because `ClosedCase.behaviour` (`DIMENSION=24, METRIC="COSINE"`), `load_case_vectors`, `similar_cases_vec`, and `similar_cases_fused` are already installed on `assay`, `install()` skips them and only runs `CREATE OR REPLACE QUERY` + `INSTALL QUERY` for any query missing from `/restpp/endpoints/assay?dynamic=true`, printing every GSQL response.
3. **Vector Loading (`push_vectors()`)**:
   - Writes `data/tg_r2/case_vectors.csv`, `docker cp`s it into `assay-tg:/home/tigergraph/data/tg_r2/case_vectors.csv`, runs `RUN LOADING JOB load_case_vectors USING f_vec="/home/tigergraph/data/tg_r2/case_vectors.csv"`, and verifies non-empty top-$k$ hits from `similar_cases_vec`.

## Final-Round State: the Graph Decides

Under `ASSAY_TG=1` (how `cases_r2/` is produced) four decisions come from installed GSQL, and a fifth query runs but abstains:

| decision | query | file |
|---|---|---|
| common point of compromise: loss-shape cohort and origin counts (HHF-001–003; parity in `docs/r2/C10_tg_cpc.md`) | `r2_cpc_loss_cohort`, `r2_cpc_origin_counts` | `graph/queries/r2_cpc.gsql` |
| ring membership | `r2_ring_wcc` | `graph/queries/r2_ring_wcc.gsql` |
| micro-drip cohort | `r2_micro_drip_cohort` | `graph/queries/r2_cohorts.gsql` |
| recipient fan-in | `r2_recipient_fanin` | `graph/queries/r2_cohorts.gsql` |
| precedent query (abstains: no setting passes both false-alarm limits, 84% of closed cases are fraud; `docs/r2/C5_vector_vote.md`) | `similar_cases_vec` (TigerVector) | `graph/queries/r2_vector.gsql` |

Each case is also written back as a `FraudCase` (`r2_upsert_case`).

- **Fail closed**: any connection, HTTP, GSQL or schema/key error, and any malformed reply, raises
  `TigerGraphUnavailableError`. The run exits 3 with no answer file and no fallback to local code. If the
  write-back fails or returns no graph id, the run says "case NOT saved to the TigerGraph graph" and exits 3
  (drill D10, `docs/r2/B30_drills.md`).
- **Local mode** (no `ASSAY_TG`, the default on a clean clone) uses the pandas implementations and
  decides identically: same cohort members and transaction ids on all 18 cases (36 comparisons, 0
  mismatches, `docs/r2/C6_tg_cohorts.md`), same ring decisions (`tg_r2.parity()`).
