# B15: Graph Algorithm Ring Detection & Ablation Results

## TigerGraph GDS Equivalence (GSQL Queries)

When running on TigerGraph, the NetworkX WCC and Louvain algorithm calls map directly to the following GDS GSQL queries:

```gsql
// 1. TigerGraph Weakly Connected Components (tg_wcc)
// Runs on the bipartite graph of Cards and non-generic shared DeviceProfiles / ProxyMarkers
CREATE QUERY tg_wcc(SET<STRING> v_types, SET<STRING> e_types, INT max_iter, STRING attr_name) FOR GRAPH CardFraudGraph {
    # Runs TigerGraph GDS WCC algorithm to cluster cards connected via specific non-generic elements
    RUN QUERY tg_wcc_internal(v_types, e_types, max_iter, attr_name);
}

// 2. TigerGraph Louvain Community Detection (tg_louvain)
// Runs on the projected Card-Card graph weighted by number of shared elements
CREATE QUERY tg_louvain(SET<STRING> v_types, SET<STRING> e_types, STRING weight_attr, INT max_levels, INT max_iter, STRING attr_name) FOR GRAPH CardFraudGraph {
    # Runs TigerGraph GDS Louvain algorithm to partition card graph into dense fraud communities
    RUN QUERY tg_louvain_internal(v_types, e_types, weight_attr, max_levels, max_iter, attr_name);
}
```

## Graph Capping and Element Filtering Policy

To eliminate generic mega-clusters (e.g. hundreds of cards connected via generic browser/OS strings or widely shared proxies):
1. **Element Filtering**: Bare `id_23` proxy markers are never used alone. They require a specific, fully specified non-generic device profile without `?` wildcards.
2. **Cardinality Capping**: Elements shared by > 15 distinct cards in a 30-day window are dropped.
3. **Ring Size Capping**: Components/communities larger than 30 cards are classified as generic clusters and excluded.

## Measured 2x2 Ablation Results

Evaluated over 4,675 confirmed fraud case openings and 5,000 sampled legitimate card-days from Jul-Oct 2016:

### Contingency 2x2 Matrix (WCC)

| Population | Fraud Episodes (Confirmed Fraud) | Legit Card-Days (Jul-Oct 2016) | Total |
|---|---|---|---|
| **In Ring w/ Prior Fraud** | 123 | 15 | 138 |
| **Not in Ring w/ Prior Fraud** | 4,552 | 4,985 | 9,537 |
| **Total** | 4,675 | 5,000 | 9,675 |

- **P(In Ring \| Fraud)**: 123 / 4,675 = **0.0263 (2.63%)**
- **P(In Ring \| Legit)**: 15 / 5,000 = **0.0030 (0.30%)**
- **MEASURED LIKELIHOOD RATIO (LR)**: **8.6946x**
- **95% CONFIDENCE INTERVAL**: **[5.0934x, 14.8419x]**

## Exam Case HHF-005 Ring Replay

For exam case HHF-005 (`C00348-K1`), `ring_for` identifies a 12-card ring `RING-WCC-c4e13e20`:
- **Member Cards**: `['C00348-K1', 'C00779-K1', 'C01063-K2', 'C01983-K1', 'C02014-K2', 'C05039-K1', 'C06947-K2', 'C07033-K1', 'C08072-K1', 'C11249-K1', 'C12656-K1', 'C13067-K1']`
- **Prior Fraud Members & Closed Case IDs**:
  - `C00779-K1`: `['CC-0780']`
  - `C01063-K2`: `['CC-1064']`
  - `C01983-K1`: `['CC-1984']`
  - `C05039-K1`: `['CC-5040']`
  - `C06947-K2`: `['CC-5566']`
  - `C08072-K1`: `['CC-5573']`
  - `C11249-K1`: `['CC-5582']`
- **Total Ring Exposure Volume**: $29,568.99

`ring_replay(ring_wcc)` dynamically generates 24 attested time frames showing the step-by-step accumulation of cards, shared elements, and volume.
