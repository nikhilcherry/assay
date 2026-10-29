# Red team: 10 Argon attackers, 2 rounds, 80 scenarios

## How it worked
Ten `gemini-4-argon` personas played people trying to beat Assay: a micro-charge skimmer, a ring
leader, a dormant-card buyer, an account-takeover artist, a card tester, a friendly-fraud disputer, a
prompt injector, a data saboteur, a breached-merchant campaigner, and a designer of *legitimate*
customers who look like fraud. Each got the bank's policy, our INNOVATION and RESILIENCE write-ups
and aggregate statistics (`tests/r2/redteam_pack/aggregate_stats.txt`). **No customer row was ever
sent.** Each persona wrote 4 scenarios per round: planted transactions on placeholder cards, one case,
and the truth.

`assay/r2/redteam.py` binds each placeholder to a real card of the requested kind (never an exam
card) and clones that card's own January row for every planted event. The real model therefore
scores it like any transaction. The harness builds an overlay dataset, runs every case **blind**
through the single-case command, and scores the answer against the attacker's truth. In round 2
each persona saw its round-1 results and attacked again. Scenarios and results:
`tests/r2/redteam_pack/`.

## Scoreboard (40 scenarios per round)
| | R1 first run | R1 after fixes | R2 first run | R2 after fixes |
|---|---:|---:|---:|---:|
| fraud caught | 6 | 6 | 3 | 3 |
| fraud caught, episode under-counted | 5 | 5 | 6 | 6 |
| fraud hedged (uncertain, escalated) | 0 | 1 | 2 | 5 |
| **fraud missed** | **15** | **14** | **13** | **10** |
| legit cleared | 3 | 1 | 6 | 5 |
| legit hedged (uncertain) | 0 | 3 | 1 | 2 |
| legit blocked, customer denied it (R2 requires the block) | 4 | 4 | 4 | 4 |
| **legit blocked, no denial** | **4** | **3** | **2** | **2** |
| bad input rejected / answered / crashed | 2 / 1 / 0 | 2 / 1 / 0 | 0 / 1 / 1 | 2 / 0 / 0 |

(One round-2 saboteur case never reached the agent: the CSV cut its text at the NUL byte.)

## What it found, and what changed
1. **Hostile case rows got through validation.** A `...Z` timestamp from 1999 crashed mid-run, and a
   malformed card ID (`C00000-Z9`) was investigated. `validate_case_row` now requires a local
   timestamp inside the data period, bank-format card and customer IDs with the card belonging to
   the customer, and no control characters. A bad flagged ID also crashed `make stage`, which is fixed.
2. **The agent cleared cases its own detectors had flagged.** A dormant card woke up (×5.65), but the
   prior was so low that a *simulated* "customer confirms" closed the case. Now an assumed reply may
   only settle a case where no measured detector fired; otherwise the case is uncertain and goes to
   an analyst. Exam answers are unchanged, and 4 frauds went from missed to hedged. The cost: 3 legit
   lookalikes went from cleared to hedged (analyst review, not a block).
3. **Hub caps.** Six round-1 rings hid by putting 16+ cards on one element, above the 15-card cap.
   The new `burst_hub` detector (an element on 3+ cards within 72h and on at most one card in the 60
   days before) measures LR 10.68, CI [4.45, 25.64], and is weighed at 4.45. It fires on no exam case.
   It rescued no red-team case: most attackers shared a generic `Windows | chrome` profile, which no
   rule can tell apart from ordinary traffic.
4. **Multi-hour takeovers.** Episodes may now extend to 48h on a device the card first used within
   48h. Exact match 0.6266 → 0.6372 on 4,631 historical reports. The red team's takeovers reused the
   victim's own device, so they stay under-counted.

## What we did not change, and why
- **Most remaining misses come from simulated replies.** At p ≤ 0.4 the brief-allowed simulation has
  the customer confirm, so a fraud we score low becomes "confirmed, legitimate". In production the
  reply is real. This is our weakest point, and the red team found it in every persona.
- **Friendly fraud** (a customer denying their own purchase) is blocked because R2 says a denial
  settles the case.
- **No prompt injection changed a verdict.** The injector's round-2 cases were rerun with a neutral
  trigger in place of fake IVR transcripts, fake card-mismatch tags and spoofed R3 links, and all 4
  verdicts came out the same. Its 3 round-2 "wins" came from stealthy fraud underneath the text, not
  from the text.

## Rounds 3 and 4 (run blind on commit d2662ff, local mode)
Round 3's scenarios were written against the earlier write-ups. Round 4's personas read the current ones
(cohort links, record conflict, "link, not vote"). Each round compiled 40 scenarios into an overlay
dataset and ran the single-case command blind. Scenarios, truth and results: `tests/r2/redteam_pack/r3`
and `r4`.

| | R3 | R4 |
|---|---:|---:|
| fraud caught | 1 | 1 |
| fraud caught, episode under-counted | 8 | 7 |
| fraud hedged (uncertain, escalated) | 7 | 12 |
| **fraud missed** | **8** | **3** |
| legit cleared | 2 | 3 |
| legit hedged (uncertain) | 3 | 6 |
| legit blocked, customer denied it (R2 requires the block) | 3 | 4 |
| **legit blocked, no denial** | **3** | **0** |
| bad input rejected / answered / crashed | 4 / 0 / 0 | 3 / 0 / 0 |
| fraud case rejected because the attacker's own case row was invalid | 1 | 1 |

What changed after them: R3's CASE-1040 came back `uncertain` and still carried `BLOCK_CARD`, because
R5 blocked on a card-testing shape whatever the verdict. Now R5 blocks only when fraud is established.
Otherwise it declines and steps up authentication (R1: verify before you block). The round 4 misses were
slow multi-day takeovers on heavy cards and a 6-card ring that rotated user agents and split charges.
They are what we still miss.
