# assay: architecture

An alert arrives. A defensible case comes out, with every number traceable to a
query and every action traceable to a policy rule.

## The rule the design is built around

**Nothing that ends up in an answer file as a number, an ID, an action or an
approval route is produced by a language model.** Of the graded fields, only
`summary` and `sar.narrative` are prose; everything else is an enum, an ID or a
number, and the brief says made-up IDs score zero and `fraud_probability` is
scored for calibration. So those fields are computed by code that is tested
and reruns to the same answer, and the prose is assembled from the same ledger.

```
case_pack row
   │
   ├─► Investigator (assay/investigate.py) -- a bounded state machine
   │      │
   │      ├─ Store ─────────────► TigerGraph (installed GSQL)  |  pandas reference
   │      │    every call logged -> becomes an evidence `ref`
   │      │
   │      ├─ Model score ───────► calibrated p from the closed cases (assay/model.py)
   │      ├─ Detectors ─────────► Findings with likelihood ratios (assay/detectors.py)
   │      ├─ Combine ───────────► logit(p) = logit(p_model) + Σ log LR
   │      ├─ §6 gate ───────────► stop, or request evidence -> simulated reply -> update
   │      ├─ Policy ────────────► decide() before and after  (assay/policy.py)
   │      └─ Episode ───────────► affected txns, connected cards, exposure, SAR
   │
   └─► FraudCase vertex + edges written back ─► retrieved by the next investigation
```

## 1. The closed cases are a training set, not just a memory

The closed-case file lists 14,055 confirmed-fraud transactions in July–October,
3.4% of the period: the entire fraud rate of the source data. Everything else in
those months is, with high probability, legitimate. `assay/model.py` trains a
LightGBM model on July–September, validates on October (AUC 0.964 against the
bank's 0.866), calibrates it with isotonic regression on the held-out month, and
then refits on all four months for scoring November and December. The reliability
table is in the README and `docs/MODEL_REPORT.json`.

The model sees the Vesta V/C/D/M columns and the risk score as inputs. The
evidence never pretends to know what `V258` means; it cites the model's
calibrated output and the facts an analyst can check.

## 2. Holders

`customer_id` is `card1`, an issuer bucket. `Holder = card + billing region +
account-open day (TransactionDT/86400 − D1)` is the inferred person, and the
baseline ("is this amount, region, device normal for them?") is computed there,
falling back to the card when a holder has under five prior transactions.

## 3. Findings and likelihood ratios

Each detector returns Findings: a claim, the query that produced it, the entity
IDs it rests on, and a likelihood ratio. Context the model already accounts for
(device novelty, amount vs baseline, region) carries LR 1: it is evidence for
the analyst, not a second vote. Detectors for multi-transaction shapes the
per-transaction model cannot see carry LRs:

| detector | LR | basis |
|---|---|---|
| structuring (≥3 online $400–$500 within an hour) | 8 | measured: 9 fraud in 41 bursts, Jul–Oct |
| device ring (≥5 cards, New and proxied on ≥90% of uses) | 30 | assumption; the one qualifying profile is the analysts' undocumented ring |
| ring profile in confirmed closed cases | 3 | assumption |
| card testing (≥3 online < $5 within an hour, then larger) | 20 | assumption; 16/16 such closed cases confirmed |
| disputed charge matches own monthly recurring charge | 0.1 | assumption (R7) |
| customer dispute | varies with the model score | measured on October: 1,203/1,203 disputes were fraud (LR ≥ 13,807), averaged equally with the score's own disputed-fraud vs legitimate density ratio, in case the dispute is planted (`assay/disputes.py`) |
| case memory: card named in an earlier fraud verdict | 3 | assumption |

## 4. The gate and the two recommendations

Policy §6 is implemented literally: stop at ≥ 0.85 or ≤ 0.15 on two independent
pieces of evidence, or when a verification response settles it. Otherwise the
agent requests evidence through the controls §5 allows, simulates the reply
(deny at p ≥ 0.6, confirm at p ≤ 0.4, no reply in between), updates, and runs
the policy a second time. `initial` and `final` are two real calls to
`policy.decide()`, and `what_changed` is computed from their difference.

## 5. Episode scope

Same card, ±24 hours of the flagged transaction, calibrated p ≥ 0.30, plus the
flagged transaction. Tuned against the bank's October cases using out-of-time
scores: mean Jaccard 0.80 with the transaction lists the analysts recorded.
Structuring and ring episodes use the detector's own transaction set instead.

## 6. TigerGraph

`graph/schema.gsql`, `graph/load.gsql`, `graph/queries/investigation.gsql`. The
ten installed queries are the investigator's tools; `assay/tg.py` calls them
over REST++ and converts results into the frames the detectors consume.
`write_case` upserts a FraudCase with ABOUT_CARD, CITES, LINKS_CARD, IMPLICATES
and DREW_ON edges. `prior_fraud_cases` retrieves them. Cases run in the order
their alerts were opened, so memory only ever looks backward.

`assay/store.py` implements the same ten queries over pandas. The two backends
agree on all 20 cases except the order of tied similar cases, which is what
lets the tests and the monitor run without a database.

## 7. The monitor

`assay/monitor.py` sweeps November–December with three scans (device rings,
structuring bursts, model-vs-bank disagreements), deduplicates against the exam
cards, seeds the case memory with the 20 exam investigations, and runs the same
investigator on every alert. Output goes to `monitor/`, never `cases/`.
