# C5 — Putting TigerVector Precedent on the Decision Path (`vector_precedent`)

## 1. Design & Zero-Leakage Causal Guarantee
`VectorPrecedentVote` (`assay/r2/detectors/vector_vote.py`, detector name `"vector_precedent"`, pattern `"none"`, `blocks_assumed_clear = False`) puts the 24-dimensional behaviour embedding from `assay/r2/vector.py` onto the investigation decision path as a calibrated log-odds vote:
- **Causal embedding (`t <= opened_at`):** Embeds the card's 24-hour episode window ending at the trigger/latest transaction at or before `opened_at`, compared strictly against prior card history before that window, standardized using only cases closed before `2016-10-01`.
- **As-of Precedent Retrieval (`closed_at <= opened_at`):** Retrieves the `k` nearest closed cases with `closed_at <= opened_at` (leave-one-out: a case never retrieves itself or any case closed after `opened_at`).
- **Backend & Fail-Closed:** When `ASSAY_TG=1`, queries TigerGraph (`_tg_run_query("similar_cases_vec", ...)`) and raises `TigerGraphUnavailableError` if TigerGraph errors or is unreachable (never falling back silently). Otherwise uses `LocalVectorIndex`.
- **Outcome Enrichment on TigerGraph Hits:** TigerGraph's `similar_cases_vec` GSQL query prefixes vertex attributes as `Hits.outcome` / `Hits.pattern`, which `vector.py`'s `similar_cases_vec` reads as `attrs.get("outcome", "")` (`""`). `_enrich_tg_neighbors` joins the TigerGraph-returned neighbour `case_id`s to their static `outcome` / `pattern` labels in `closed_cases()` so the Laplace fraud score is computed identically on both backends.
- **Idempotent TigerGraph Vector Sync (`sync_missing_closed_case_vectors`):** Diagnoses `ClosedCase` vertex and vector presence in TigerGraph for the 20 new Round 1 exam cases (`CC-5566..CC-5585`) using each case's `closed_at` cutoff (`k=50`, avoiding false misses from tied vectors on early cases like `CC-0001`), upserts any missing `ClosedCase` vertices/edges over REST++, and runs `load_case_vectors` when vectors are missing.
- **Finding Contract:** Never names a typology (`pattern = "none"`), sets `blocks_assumed_clear = False`, `affected_txn_ids = []` and `connected_card_ids = []` (precedent is not a ring), includes `precedent=True` in `claim.params`, and attests the exact neighbour `case_ids`, outcomes, cosine distances, backend used, and `(k, theta, score)`.

## 2. Pure vs Specific-Element Fused Variant
In `B14_vector.md`, naive 2-hop graph fusion over broad billing regions and freemail domains degraded accuracy (Brier `0.0896` vs `0.0863`). In C5 we tested a **specific-element fused variant** that boosts a candidate neighbour (`-0.25` cosine distance) *only* if it shares a high-specificity link with the case card as-of `opened_at`:
1. A `device_profile` seen on `<= 15` cards as-of `opened_at`,
2. A non-freemail `R_emaildomain` seen on `<= 15` cards as-of `opened_at`, or
3. The candidate case's `connected_card_ids` (or `card_id`) contains the card.

## 3. Measured As-Of Backtest (Aug–Dec 2016 Closed Cases & 2,400 Legitimate Card-Days)
Evaluated over all **4,335** closed cases opened in `2016-08-01 .. 2017-01-01` (**3,654** `confirmed_fraud`, **681** `cleared`) and a fixed-seed (`seed=42`) sample of **2,400** legitimate card-days from cards never named in any closed or exam case.

Because exam alerts are already selected for investigation, the honest likelihood ratio is `P(fire | fraud case) / P(fire | cleared case)` (Laplace-smoothed), constrained to `P(fire | cleared case) <= 8%` (and `P(fire | legit card-day) <= 3%`):

| Variant | `k` | `theta` | `P(fire \| fraud case)` | `P(fire \| cleared case)` | `P(fire \| legit card-day)` | `LR vs cleared` | `95% CI (vs cleared)` | `LR vs legit day` | `95% CI (vs legit day)` | `Cleared <= 8%` |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| `pure` | 5 | `0.8571` (`5/5` fraud) | `62.73%` (`2292/3654`) | `36.86%` (`251/681`) | `14.12%` (`339/2400`) | `1.70` | `[1.54, 1.88]` | `4.43` | `[4.01, 4.90]` | `> 8%` (skip) |
| `pure` | 10 | `0.9167` (`10/10` fraud) | `42.23%` (`1543/3654`) | `17.33%` (`118/681`) | `4.33%` (`104/2400`) | `2.42` | `[2.05, 2.86]` | `9.66` | `[7.98, 11.71]` | `> 8%` (skip) |
| **`pure` \*** | **20** | **`0.9545` (`20/20` fraud)** | **`25.45%` (`930/3654`)** | **`6.46%` (`44/681`)** | **`1.42%` (`34/2400`)** | **`3.86`** | **`[2.90, 5.15]`** | **`17.64`** | **`[12.56, 24.75]`** | **`<= 8%` OK (Selected)** |
| `fused` | 5 | `0.8571` (`5/5` fraud) | `62.89%` (`2298/3654`) | `37.00%` (`252/681`) | `14.12%` (`339/2400`) | `1.70` | `[1.54, 1.88]` | `4.44` | `[4.02, 4.91]` | `> 8%` (skip) |
| `fused` | 10 | `0.9167` (`10/10` fraud) | `42.45%` (`1551/3654`) | `17.33%` (`118/681`) | `4.33%` (`104/2400`) | `2.44` | `[2.06, 2.88]` | `9.71` | `[8.02, 11.77]` | `> 8%` (skip) |
| `fused` | 20 | `0.9545` (`20/20` fraud) | `25.45%` (`930/3654`) | `6.61%` (`45/681`) | `1.42%` (`34/2400`) | `3.78` | `[2.84, 5.02]` | `17.64` | `[12.56, 24.75]` | `<= 8%` OK |

### Selection & Weight
- At **`k=20, theta=0.9545` (`20/20` confirmed-fraud neighbours)**, `P(fire | cleared case) = 6.46% (44/681) <= 8%` and `P(fire | legit card-day) = 1.42% (34/2400) <= 3%`, while catching **`25.45%` (`930/3654`)** of confirmed-fraud cases.
- **Pure vs Fused at `k=20, theta=0.9545`:** `pure` achieves `LR vs cleared = 3.86` with **95% CI `[2.90, 5.15]`**, beating `fused` (`LR vs cleared = 3.78`, 95% CI `[2.84, 5.02]`, `45/681` cleared cases acted against). We therefore select **`variant="pure", k=20, theta=0.9545, weight=2.9016` (`~2.90`)**.

## 4. Measured Run & Parity Check on the 18 Final-Round Exam Cases (`case_pack()`)
Running `VectorPrecedentVote(k=20, theta=0.9545, variant="pure", weight=2.9016)` on all 18 exam cases:
- **Fire decision parity:** `18/18` identical between Local and TigerGraph.
- **Score parity:** `|tg_score - local_score| <= 0.10` on `18/18` cases (high HNSW set Jaccard `~0.90–1.00`).
- **Cases firing (`k=20, theta=0.9545`):** `HHF-005`, `HHF-008`, `HHF-009`, `HHF-010` (`20/20` confirmed-fraud neighbours, score `0.9545`).
