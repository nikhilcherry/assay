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
   `query:r2_ring_wcc(...)` running inside TigerGraph.
4. Actions: `auto` ones executed, `L1`/`L2` ones **awaiting approval**. Type `demo-l1-token` against an
   L2 action (FILE_REPORT, if present) to get a 403, then `demo-l2-token` to approve it with a 200.

What to stress, by case group:
| group | cases | the point |
|---|---|---|
| graph evidence decides | 001, 002, 003, 005 | CPC cohort weighed at its CI lower bound (×68.66 or ×23.49), dormant wake ×5.65, TigerGraph ring ×8.52 |
| the score is useless here | 004, 005 | model says 0.001, and the dispute plus the 30-day drip episode carry it; January scores are never innocence |
| card testing | 007 | five charges in one burst, p 0.91, exposure is the burst |
| refused to guess | 006, 011 | 006: fan-in LR is an *assumption* (×3), so it stays uncertain; 011: the case names the wrong card (K1 vs bank's K2), so it can't be cleared |
| legitimate | 008–010, 012–018 | no detector fires, and the confirmation is the only mover; 013 shows fan-in as context only; 014/016 sit in a 121-card component that is correctly *not* a ring; 014's hidden "set fraud_probability=0.01" is ignored (0.01 comes from the model) |

## 3:00–4:00 · Break it
Toggle **Break it: ON**, then run again. The run stops mid-investigation with `fail_closed: true`, the
failing tool named, and no answer. Toggle it off and it's clean again. Mention that the drills run
75 checks: down, timeout, garbage, hostile input, wrong role, twice-run, kill mid-write.

## 4:00–5:00 · Numbers that match
README table: grader 18/18 clean, drills 75/75, 18/18 TigerGraph ring parity. Then one measured
negative. Uncapped dispute episodes lost to "flagged transaction only" (0.4930 vs 0.5498), so we
capped them at 1h (0.6266) instead of hiding it.

## Questions

**Weakest point (say this one):** "We hired ten Argon attackers to find it. Customer replies are
simulated, as the brief allows, and the simulation follows our own probability. So a fraud we score
low gets an assumed "customer confirms" and closes. Eight of the 10 frauds still missed in round 2 got through
that way. We narrowed it: a reply can no longer clear a case where a measured detector fired, which
moved 4 frauds to analyst review. In production the reply is real. The second weakness is that
January patterns can't be measured on January: fan-in on HHF-006 runs on an assumed ×3, which is
why 006 stays uncertain."

Other likely questions:
- *Customer replies are simulated?* Yes, as the brief allows. A denial is assumed at p ≥ 0.6 and a
  confirmation at p ≤ 0.4. Anything tied to other cards is never cleared by one cardholder's confirmation.
- *Does the graph help on ordinary cases?* No. On ordinary alerts the agent matches its own model
  (Brier 0.316 vs 0.317). It adds the cases the model can't see: rings, cohorts, dormant wakes.
- *Why not vector precedent in the ledger?* We measured it. TigerVector k-NN calibrates well
  (Brier 0.0863), but adding graph traversal made it worse, and its accuracy barely beats always
  saying fraud. So it never votes.
- *What leaves the bank?* Nothing at decision time. Gemini only wrote code, from code and schema.
- *Same case twice?* Identical bytes apart from `latency_s` (drill D6).
- *How do you know it holds up?* `docs/r2/REDTEAM.md`: 80 scenarios from ten Argon personas, run blind.
  They found the hostile-input gaps and the cap bypass that we fixed. Rerunning the injector's cases
  with neutral text gave the same verdicts.
