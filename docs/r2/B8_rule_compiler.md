# Block B8 — Policy Text to Detector (`assay/r2/rule_compiler.py`)

## Overview
Deterministic, zero-LLM rule compiler that translates written policy rules (`R5`, `R6` verbatim from `policy/fraud_policy.md`) and regulator-style advisories (`ADV-1`, `ADV-2`, `ADV-3`) into typed `RuleSpec` ASTs and vectorised `CompiledRuleDetector` instances (`assay.r2.finding.Detector` subclasses). Clauses outside the controlled grammar are surfaced explicitly in `RuleSpec.unparsed_phrases` for human review.

## Controlled Grammar & Compilation Speed
- **Quantifiers**: Transaction counts (`three or more small online authorizations`), card counts (`several cards`, `two or more cards`, `on one card`), region counts (`two or more billing regions`).
- **Windows**: `within an hour` (3,600s), `within 2 hours` (7,200s), `within 24 hours` (86,400s), `within 48 hours` (172,800s), `in one window` (604,800s).
- **Thresholds & Sequences**: `small` ($\le \$25$), `followed by a larger purchase`, `purchase over $100`, `exposure exceeds $500` / `$1,000`.
- **Entities & Channels**: `online`, `in_person`, `device_profile`, `addr1` (billing region), `R_emaildomain` (recipient email).
- **Unparsed Phrase Detection**: On `ADV-3`, the unsupported clause `"followed by biometric voiceprint verification failure"` is deterministically isolated in `unparsed_phrases`.
- **Compile Time**: All 5 rules compile in `< 2 ms` total (`~0.1–0.4 ms` per rule).

## Precomputed As-Of Index & Backtest Performance
`_get_precomputed_index(cc, tx)` builds slim numpy slices per card and per non-generic shared entity once per `(cc, tx)` pair, keeping peak memory well below 100 MB and evaluating all 5 rules in `< 5 seconds` while strictly enforcing `t <= opened_at` per closed case. Every backtest and every per-case agreement/disagreement on Round 1 (`HHG-001`..`HHG-020`, mapped to `CC-5566`..`CC-5585`) produces a re-runnable `assay.r2.attest.Attestation`.
