# Innovation Claim: Evidence That Knows Its Own Weight

Sources: `docs/r2/FACTS.md` (what fired where, from a live run), `docs/r2/ABLATION.md` (what each part
changes) and `docs/r2/VERIFY_CLAIMS.md` (does every claim reproduce).

## 1. The idea
Half the exam is legitimate, and blocking a legitimate customer scores zero. So no signal in Assay votes
until it has been measured, and each gets exactly the vote its measurement earns:
- **Measured weight**: an as-of backtest against the bank's labelled closed cases. When the sample is small,
  the weight is the 95% CI lower bound.
- **Link, not vote**: a pattern that never happened before January has no measurable likelihood ratio. It
  still links the case to other cards for §3a, but weighs 1.0. Rarity never becomes a weight.
- **Escalate, never decide**: untrusted text and conflicting records only send a case to a person.

Every finding carries a re-runnable attestation. `verify_claims` checks all 325 of them: 0
mismatches.

## 2. On the exam
| signal | cases | weight |
|---|---|---|
| Common point of compromise | HHF-001, 002, 003 | 68.66 / 23.49 / 23.49 (CI low) |
| Dormant card wakes | HHF-003 | 5.65 |
| Capped graph ring (`docs/r2/B15_rings.md`) | HHF-005: 12 cards | 8.52 |
| New micro-drip cohort | HHF-004, 005 | link only |
| Recipient fan-in (`docs/r2/B6_micro_fanin.md`) | HHF-006: 12 first-time payers | link only |

The result is 6 fraud, 3 uncertain and 9 legitimate, matching the brief's nine legitimate cases. The grader
reports 18/18 clean. Under `ASSAY_TG=1` the graph decides the ring and both cohorts through installed GSQL,
and any graph error fails closed. Graph and local code agree on all 18 cases (`docs/r2/C6_tg_cohorts.md`).

## 3. Where it changes the outcome (ablation)
- Without `cpc`, HHF-001 and HHF-002 flip from fraud to legitimate.
- Without `micro_drip_cohort`, HHF-004 loses its 16 linked cards and the report §3a owes on a $23.99 loss.
- Without `ring_wcc`, HHF-005 drops from 0.96 to 0.76.
- Without the card-mismatch guard, HHF-011 is cleared on a cardholder nobody has identified.
- Without the record-conflict guard, HHF-008 is closed, although the alert says "online" and the bank's
  record says in person.
- Without the measured dispute weight, HHF-004 falls from fraud to uncertain.

## 4. Measured negatives
- **TigerVector precedent** (`docs/r2/B14_vector.md`, `C5_vector_vote.md`): no setting keeps false fires
  under 8% of cleared cases and 3% of legitimate card-days at once. 84% of closed cases are fraud, so
  "the nearest 20 were fraud" mostly restates the base rate. The query runs and abstains. An earlier
  selection weighed it ×4.09 until we found a fallback that dropped one constraint. We fixed the code and
  gave up the vote.
- **New-device takeover** (`C4_new_device.md`): it fails the gate against cleared cases, because
  `card_id` is an issuer bucket. At the holder level (`C7_holder_shift.md`) it can't apply: every exam
  holder has at most one prior charge. HHF-009 is verified and closed.
- **Others that lost to simpler baselines**: per-card micro-charges and the fan-in's old assumed 3.0
  (`docs/r2/B6_micro_fanin.md`), "innocence" signals (`docs/r2/B29_innocence.md`), charge-shape
  campaigns (`docs/r2/B28_campaign.md`), an HMM episode segmenter (`docs/r2/B34_episode.md`), and an
  uncapped dispute episode (`docs/r2/N7_dispute_episode.md`).
- **January drift** (`docs/r2/B1_model.md`): only 1.0670% of January rows are in-distribution, so a
  January score can raise suspicion but never clear a case.

## 5. Tested by adversaries
Argon personas wrote 160 blind attack scenarios over four rounds (`docs/r2/REDTEAM.md`,
`tests/r2/redteam_pack/`). In round 4, no legitimate customer was blocked without a denial, and 3 of
the frauds were missed. Eight Argon critics argued four rounds over what to build. We overruled them where
the numbers did: no takeover vote, no precedent vote, and no uncertain probabilities forced into a
calibrated-looking band.

**Weakest point.** HHF-008 and HHF-011 are uncertain at p = 0.01. The conflict concerns whose card it is
or what was recorded, not how risky the charge is, so we escalate rather than invent a number.
