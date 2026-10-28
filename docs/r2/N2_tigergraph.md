# Block N2 — TigerGraph as the Live Source of Truth

## Round 3 Fixes Applied

1. **`SYNTAX v2` in `r2_ring_wcc.gsql` and `r2_case_io.gsql`**:
   - Added `FOR GRAPH assay SYNTAX v2` to `r2_ring_wcc`, `r2_upsert_case`, and `r2_read_cases` so multi-hop edge patterns `-(EDGE>)-` compile cleanly alongside `r2_vector.gsql`.
2. **Idempotent `install()` without redundant work**:
   - `install()` inspects `/restpp/endpoints/assay?dynamic=true` and `USE GRAPH assay; LS; SHOW QUERY *` first.
   - Because `ClosedCase.behaviour` (`DIMENSION=24, METRIC="COSINE"`), `load_case_vectors`, `similar_cases_vec`, and `similar_cases_fused` are already installed on `assay`, `install()` skips them and only runs `CREATE OR REPLACE QUERY` + `INSTALL QUERY` for any query missing from `/restpp/endpoints/assay?dynamic=true`, printing every GSQL response.
3. **Vector Loading (`push_vectors()`)**:
   - Writes `data/tg_r2/case_vectors.csv`, `docker cp`s it into `assay-tg:/home/tigergraph/data/tg_r2/case_vectors.csv`, runs `RUN LOADING JOB load_case_vectors USING f_vec="/home/tigergraph/data/tg_r2/case_vectors.csv"`, and verifies non-empty top-$k$ hits from `similar_cases_vec`.
