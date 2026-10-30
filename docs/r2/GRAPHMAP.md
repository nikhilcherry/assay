# Someone else's schema, as a graph (Challenge 13, organiser handout 2026-10-30)

The organiser handed out three bank graph schemas: ISO 20022 payments, BIAN service domains and Temenos T24
core banking. Each comes with a ground-truth mapping onto one canonical investigation model in its comments.
`assay/r2/graphmap/` maps a bank's GSQL onto that model, gets a human to approve the mapping, and runs one
canonical investigation on TigerGraph against the bank's own schema.

## Pipeline

1. **Parse.** `gsql.parse` strips every comment first, so the ground truth never reaches a mapper.
   `gsql.ground_truth` reads those comments for the scorer only; a test pins this.
2. **Rule mapper** (`mapper.py`). Structure decides first:
   - Transaction = the record that carries an amount and a time, and is not a batch or header, not a return,
     not a ledger record that points back at a payment, and not an order that another record executes.
   - Account = what transactions debit and credit.
   - Person = the record with person attributes that reaches an account within two hops. A shared party type
     is split by its discriminator attribute.
   - Edges = the shortest schema path, scored by the edge's own words (debtor vs creditor; "ultimate" is
     penalised). The path is walked as `reverse_*` when the bank's edge points the other way.

   Every choice records the rule that fired, why, the rejected look-alikes and what still needs sample records.
3. **LLM proposer** (`llm.py`, gemini-3.5-flash, temperature 0). It sees the stripped schema only. A review
   round then shows it the rule mapper's structural evidence. `validate()` resolves every name and works out
   step direction and GSQL names in code; invented names are dropped and reported. Egress: schema text only.
4. **Proposal → approval** (`proposal.py`). Each element is marked "mappers agree" or "disagree". Approval
   needs a lead or manager role and records who, when and every element taken from the rule mapper instead.
   An unapproved proposal cannot drive the investigation.
5. **Compile + run** (`compile.py`, `tgrun.py`). Each canonical edge becomes a GSQL tag walk along its
   mapped path. A discriminator not yet confirmed from records refuses to compile.

## Measured (`python -m assay.r2.graphmap.evaluate '<glob>'`, cached, reproducible)

| Set | Rules | LLM | LLM + review |
|---|---|---|---|
| Handout, 3 banks, 82 graded elements (**dev**: the rules were written against it) | 82 | 72 | 75 |
| Blind set 1: Finacle, Open Banking UK, ISO 8583, FIBO, Mambu (175) (dev after first look) | 79 | 147 | 148 |
| **Blind set 2: FLEXCUBE, SWIFT MT, NACHA ACH, Vault, SAP FS (175), scored once, held out** | 92 | 147 | **149 (85%)** |

- The rule mapper's 82/82 on the handout is fit, not skill. Blind, it scores about 50% because opaque codes
  (Finacle `GAM`, `DTH`) defeat name cues.
- Where the two mappers agree they were right 140 of 145 times on the dev sets. That agreement flag is what
  the human reads first.
- The blind schemas and their ground truth were written by gemini-4-argon, in the handout's format, before
  any scoring. The mapper is a different model of the same family; treat that as a possible home advantage.
- The posting-name fix (below) came after the held-out run. Only the rules column was recomputed after it.

## Investigation on TigerGraph (`python -m assay.r2.graphmap.tgrun <schemas> [--proposal ids]`)

**Synthetic world** (no customer data):
- 12 customers and 14 accounts.
- Mule B07: 5 accounts pay about $490 each within 50 minutes into A14, and 3 of them come from device D13.
- Decoy B01: a utility that 4 accounts pay, each from their own device.

**Separation of writer and reader.** The data is written into each bank's native graph through the *rule*
mapping, which matches the organiser's ground truth on every graded element. The fan-in query is compiled from
the *proposal*, so a wrong proposal shows up as a wrong answer.

| Bank (organiser's file, loaded as-is*) | B07 | B01 |
|---|---|---|
| ISO 20022 | 5 payers → A14, $2,459.93, **uncertain**: no device or IP on payments | uncertain |
| T24 | same 5 payers, same amount, **uncertain** for the same reason | uncertain |
| BIAN | same, devices D09/D11/D13 → **mule ring** (path: execution → order → authentication → device, 3 hops) | each payer on their own device → **popular payee** |

The same result holds with the LLM proposals (dry run, unapproved, labelled as such). BIAN's
`partyType` customer/counterparty split was settled from sample records read back from TigerGraph.

\* T24's `CUSTOMER.TARGET` is a GSQL reserved word, so the file fails to parse on TigerGraph 4.2.5 as shipped.
`tgrun.load_compat` renames that one attribute to `TARGET_` and logs it; no mapping uses it.

## Known limits

- An attribute-only device or IP (T24 `PROTOCOL.TERMINAL_ID`) is mapped but not joined: the fan-in reports
  "evidence unavailable" instead of joining on values.
- A string date's format (T24 `YYMMDDHHMM`) is flagged as needing sample records; the compiler does not
  parse it.
- Found by a unit test: the rules rejected any type named "posting", but a core's posting can be the movement
  itself. Fixed: the structural reference-edge rule catches ledger postings instead.
