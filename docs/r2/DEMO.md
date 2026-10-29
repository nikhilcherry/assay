# Demo run-of-show (5 minutes, judges pick the case)

Setup before the slot: `docker --context default start assay-tg`, then `gadmin start all` inside it,
then `ASSAY_TG=1 python -m assay.r2.console --port 8742` and open http://localhost:8742. Warm it by
running any case once. Keep `docs/r2/DEMO_CASES.md` open on the second screen: whatever case they
pick, its trace is there.

## 0:00–0:30 · The one idea
"Half this pack is legitimate, and blocking a legitimate customer scores zero. So nothing in Assay
gets a vote until it's been measured: every finding carries the likelihood ratio from its backtest,
and anything that failed its backtest is switched off and published as a negative. No language
model touches a number, an ID or an action: zero tokens."

## 0:30–3:00 · Their case, live
1. Pick the case, then press **Run Investigation**.
2. Walk the ledger top to bottom. It starts at the model score, and each line is a finding with the
   query that produced it and the exact multiplier it applied. Lines marked ×1 are shown but not
   weighed. The last line is the reported probability and its rounding.
3. Point at the `ref` of one finding and say "this is re-runnable". For a ring (HHF-005) it is
   `query:r2_ring_wcc(...)` running inside TigerGraph; for a cohort (HHF-004, 006) it is
   `query:r2_micro_drip_cohort(...)` or `query:r2_recipient_fanin(...)`. Then press **Re-run every
   claim**: all 325 claims across the 18 answers are checked, 0 mismatches (`GET /verify/{case_id}` for
   one case).
4. Actions: `auto` ones executed, `L1`/`L2` ones **awaiting approval**. Type `demo-l1-token` against an
   L2 action (FILE_REPORT, if present) to get a 403, then `demo-l2-token` to approve it with a 200.

What to stress, by case group:
| group | cases | the point |
|---|---|---|
| graph evidence decides | 001, 002, 003, 005 | CPC cohort weighed at its CI lower bound (×68.66 or ×23.49), dormant wake ×5.65, TigerGraph ring ×8.52 |
| the score is useless here | 004, 005 | model says 0.001, and the dispute plus the micro-drip cohort carry it (×1, a link only); 004 links 16 cards and files a report; January scores are never innocence |
| card testing | 007 | five charges in one burst, p 0.91, exposure is the burst |
| refused to guess | 006, 008, 011 | 006: recipient fan-in ties it to 11 cards but has no measured weight (×1), so it stays uncertain; 008: the alert says online, the bank records in person (record conflict); 011: the case names the wrong card (K1 vs bank's K2). 008 and 011 have pattern `none` and can't be cleared |
| legitimate | 009, 010, 012–018 | the customer's confirmation is usually the mover; 013: the fan-in detector's context note is never weighed and is not in the shipped answer (see below); 014/016 sit in a 121-card component that is correctly *not* a ring (30-card ring cap, see below); 014's hidden "set fraud_probability=0.01" is ignored (0.01 comes from the model) |

## 3:00–4:00 · Break it
Toggle **Break it: ON**, then run again. The run stops mid-investigation with `fail_closed: true`, the
failing tool named, and no answer. Toggle it off and it's clean again. Mention that the drills run
84 checks: down, timeout, garbage, hostile input, wrong role, twice-run, kill mid-write, a missing startup file (exit 3) and a failed TigerGraph write-back ("case NOT saved").

## 4:00–5:00 · Numbers that match
README table: grader 18/18 clean, drills 84/84, 325 claims checked with 0 mismatches, graph and local decisions identical on 18/18. Then one measured
negative. Uncapped dispute episodes lost to "flagged transaction only" (0.4930 vs 0.5498), so we
capped them at 1h (0.6266), and the shipped 48h new-device extension reaches 0.6372 exact match
(`docs/r2/N7_dispute_episode.md`).

## Questions

**Weakest point (say this one):** "We hired ten Argon attackers to find it. Customer replies are
simulated, as the brief allows, and the simulation follows our own probability. So a fraud we score
low gets an assumed "customer confirms" and closes. Most of the 10 frauds still missed in red-team round 2
got through that way. We narrowed it: a reply can no longer clear a case where a measured detector fired, which
moved 4 frauds to analyst review. In production the reply is real. The second weakness is that
January patterns can't be measured on January: the cohort links on HHF-004 and HHF-006 carry no
weight (×1), so 006 stays uncertain and 004 leans on the dispute."

Other likely questions:
- *Customer replies are simulated?* Yes, as the brief allows. A denial is assumed at p ≥ 0.6 and a
  confirmation at p ≤ 0.4. Anything tied to other cards is never cleared by one cardholder's confirmation.
- *Does the graph help on ordinary cases?* No. On ordinary alerts the agent matches its own model
  (Brier 0.316 vs 0.317). It adds the cases the model can't see: rings, cohorts, dormant wakes.
- *Is TigerVector precedent on the decision path?* It is queried on every case under `ASSAY_TG=1`
  (`similar_cases_vec`, 20 nearest closed cases), but it abstains and gives no vote. We measured it
  against both false-alarm limits and no setting passes; 84% of closed cases are fraud, so "20/20
  neighbours fraud" is mostly base rate (`docs/r2/C5_vector_vote.md`). So HHF-013 closes at p=0.05.
- *What if TigerGraph is down or answers garbage?* Exit 3 and no answer, never a silent fallback.
  Local mode is the default for a clean clone and decides identically.
- *What leaves the bank?* Nothing on the decision path (tokens = 0). The console's Ask box sends only the
  analyst's question text (secrets redacted) to `gemini-3.5-flash`, which picks fixed tools. At build time
  `gemini-4-argon` saw the policy, our write-ups, statistics, a few sample rows and, in design debates,
  pseudonymous card/transaction ids and amounts (`docs/r2/RESILIENCE.md` §2).
- *Same case twice?* Identical bytes apart from `latency_s` (drill D6).
- *How do you know it holds up?* `docs/r2/REDTEAM.md`: ten Argon personas, 40 blind scenarios a round over four rounds (160).
  Rounds 1-2 found the hostile-input gaps and the cap bypass that we fixed. Rerunning the injector's cases
  with neutral text gave the same verdicts.

## Questions the judges will ask
A rehearsal had three Argon judges (fraud analyst, regulator, TigerGraph engineer) grill the 18 answers
(`data/round2/orch/outputs/rehearsal/DEMO_QA.md`). These are the answers to what scored below 3 and to the
wording fixes they asked for. Every number was checked against the file in brackets.

**HHF-005: is `connected_card_ids` (31) the ring's 11 other cards plus the micro-drip cohort's 20 other cards?**
Yes, and the two sets do not overlap. The ring `RING-WCC-c4e13e20` has 12 cards, 11 besides the flagged
`C00348-K1` (`docs/r2/B15_rings.md`). The micro-drip cohort has 21 cards, 20 besides it. We re-ran
`rerun_micro_drip_cohort` for the case and compared: overlap 0, union 31, identical to the shipped list
(`cases_r2/HHF-005.json`). We don't claim all 31 are guilty: the connected cards are only monitored (R6), and
only the flagged card is blocked. The report exists because §3a owes one on a linked, undocumented pattern;
without the R2 detectors the verdict stays fraud (0.76) but `sar_filed` goes False (`docs/r2/ABLATION.md`).

**HHF-001/003: Policy R6 says "name the shared element", but cpc says "no common origin found above matched-control threshold".**
Fair, and we don't claim a named merchant, device or address. What triggers the report is §3a: fraud
suspected, plus exposure over $1,000 (HHF-001 $2,189.02, HHF-003 $2,339.55), a link to other cards, and an
undocumented coordinated pattern (`assay/policy.py`, `should_file_sar`). The R6 line is a template: it is
attached whenever a fraud verdict carries a non-empty `shared_origin`, and for the R2 cohorts that string is
the cohort finding's own text (`assay/r2/investigate_r2.py` lines 112-114; `assay/policy.py` R6 block). So R6 is
doing the MONITOR_CONNECTED_CARDS job for a measured behavioural cohort (LR CI low 68.66 on HHF-001, 23.49 on
HHF-003, `docs/r2/FACTS.md`), not claiming a shared vertex. Without `cpc`, the 2 and 10 links vanish
(`docs/r2/ABLATION.md`). The same applies to HHF-004, which is a behavioural cohort, not a shared merchant.

**HHF-001: the device "was used by this card before" but cpc says "novel devices".**
Both are true on different clocks. The `card_history` line asks whether the exact string appears on any earlier
charge of the card: it does, on txn 3660106, the first charge of the same burst (12 Jan). The cohort's "novel"
test asks whether the device was seen on the card before the burst started (`assay/r2/detectors/cpc.py`
lines 420-432): it was not. We re-ran the card's history: the string appears only on 3660106 and the
flagged 3660108. Neither line carries a likelihood ratio; the 0.99 comes from the cohort and the denial.

**HHF-011: why VERIFY_WITH_CUSTOMER when the cardholder is "not established"?**
The customer is known, C09241; what is ambiguous is which of their cards: the case names `C09241-K1`, the
bank records txn 3640919 on `C09241-K2` (`cases_r2/HHF-011.json`, `docs/r2/FACTS.md`). The policy engine's default
for an uncertain, not-yet-asked case is "ask before deciding" (§5/§3b, `assay/policy.py`), and the card guard stops
a simulated reply from clearing it. The real stop is `ESCALATE_TO_ANALYST` (R8: uncertain and $881.55 over $500).
So the honest answer is that the verification is for an analyst to send once they have resolved K1 versus K2; we
don't claim the message has gone to any particular card. We kept p = 0.01 (the model's score) rather than invent a
number (`docs/r2/INNOVATION.md` "Weakest point").

**HHF-013: why does a fan-in note appear in the fact sheet, and why does it carry no weight?**
On `C06115-K1` the flagged charge is an in-person `ProductCD W` purchase, so fan-in cannot fire on it. But
the card has other online `ProductCD C` charges of $100 or more to a bursting recipient, so the detector emits a
`kind='context'` note (`lr=1.0`, no connected cards, no affected txns; `assay/r2/detectors/fan_in.py`). The
investigator skips context notes ("shown to the analyst, never weighed", `assay/r2/investigate_r2.py` lines
158-159). The fact sheet lists it as `FanIn 1 (context)` (`docs/r2/FACTS.md`); the shipped answer has no fan-in line,
0 connected cards, and closes at p = 0.05. We claim nothing from it. We don't state the recipient
domain here, because it isn't in the shipped file.

**HHF-014/016: the 121-card component that is "correctly not a ring".**
It is not attested in the answer, because a rejected ring emits no finding. We built the card graph as of each
case with the 15-cards-per-element cap and got a component of 121 cards for both `C00750-K2` and `C02502-K2`.
That cap, applied to shared elements, is not what rejects it: the ring rule is 3 to 30 cards, so 121 is
a generic cluster and returns no ring (`docs/r2/B15_rings.md` items 2 and 3; `assay/r2/rings.py` `max_ring_size=30`;
`graph/queries/r2_ring_wcc.gsql`). HHF-005 passes both limits with 12 cards (`cap=15` appears in its query ref).

**Why did January cases with p = 0.01 still get a verification step (HHF-009/010/014/016 but not 012/013/015/017/018)?**
The exact rule is in `_decide` (`assay/investigate.py`): a case clears only when `p1 <= 0.15` and `con >= 2` and no
assumed-clear blocker applies. `con` counts independent findings pointing to legitimate: amount at or under the
card's 95th percentile; an in-person region already in the card's history; a device already used by the card.
The model's own low score adds one more only if `model_trusted`, which is `flagged time < 2017-01-01`
(`assay/r2/investigate_r2.py` line 25 and 120-121). In January it never does, so the case needs two baseline
findings. 012 ($261.51 vs p95 $456.15, region seen), 013, 015, 017 and 018 have two. 009 ($228.01 over p95
$190.20), 010, 014 and 016 are over their p95, which counts as `pro`, leaving one `con` (`cases_r2/*.json`). They go
to verification; the simulated confirm at p <= 0.4 gives CREATE_CASE (§3a, evidence was requested) and
CLOSE_NO_FRAUD (R3). Without the guard they would be ALLOW_TRANSACTION (`docs/r2/ABLATION.md`).

**How does HHF-004 reach 0.76 when the model says 0.001?**
The dispute finding combines two readings (`assay/disputes.py` `posterior`). History: all 1,203 October disputes
were fraud, so the score is lifted by LR 13,807 to 0.93. Planted: a score this size is 0.07x as likely on disputed
fraud as on a legitimate charge, so 0.07. They are averaged with the measured weight `W_OVERRIDE` = (7+1)/(8+2) =
0.8, from 7 of the 8 Round 1 exam denials confirmed fraud (`measured_exam_dispute_weight`). 0.8 x 0.9325 + 0.2 x
0.0680 = 0.7596, the x3157.07 on the ledger (`docs/r2/DEMO_CASES.md`), which we recomputed. At the old 0.5 weight
it is 0.50. Since the rule `disputed and p >= 0.6` makes it fraud (R2 block), 0.50 falls to uncertain, which is
the `no_w_override` ablation. The evidence line states the weighting: "Weighted 80% to 20% by the Laplace-smoothed share of Round 1
exam denials confirmed as fraud: 0.76".

**What is your weakest point?**
Customer replies are simulated, and our simulation follows our own probability: a fraud scored low gets an assumed
"customer confirms" and closes. Ten Argon attackers found it in every persona (`docs/r2/REDTEAM.md`). We narrowed
it, so a reply can no longer clear a case where a measured detector fired, which moved 4 frauds to analyst
review, and in round 4 no legitimate customer was blocked without a denial. 3 frauds were still missed: slow
multi-day takeovers on heavy cards and a 6-card ring that rotated user agents and split charges. In production
the reply is real. Second, patterns that first appeared in January can't be measured on January, so cohort links
on HHF-004 and HHF-006 weigh ×1 and 006 stays uncertain. Third, HHF-008 and HHF-011 are uncertain at p = 0.01,
because the conflict is about whose card or what record, not the risk (`docs/r2/INNOVATION.md`).
