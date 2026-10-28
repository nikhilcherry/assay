# B12 — Hostile Free Text & Prompt Injection Defence (Challenge 10)

## 1. Architecture (`assay/r2/untrusted.py`)

All free text (`trigger_text`, `analyst_notes`) is treated strictly as untrusted data, never as instruction:
1. **Linear-Time Sanitization & Quarantine**: All regexes are linear-time ($O(N)$). Tag injections (`[[...]]`, `<!--...-->`, `<system>...</system>`), markdown links, verified base64 payloads, and claims of authority (`SYSTEM NOTE:`, `P.S. compliance...`, `Note from customer:`) are recorded in `injection_spans` and `claims_of_authority`, and stripped from the normalized buffer before fact extraction.
2. **Template-Bound Fact Extraction**: Only typed `TriggerFacts` (`customer_denies`, `referenced_txn_ids`, `stated_amount`, `stated_score`, `analyst_ask`) flow into downstream investigation logic.
3. **Resilient Quote, Contraction, and Negation Handling**: Outer quote matching is anchored to `Refers to` or end-of-message (supporting contractions like `don't`/`didn't`, smart quotes `‘...’`/`“...”`, unquoted messages, and multi-line text). Negated confirmations (`cannot confirm`, `do not recognize`) and double negations (`did NOT not make`) are accurately classified.
4. **Cross-Checking (`cross_check`)**: Extracted `referenced_txn_ids` (supporting arbitrary integer IDs), `stated_amount`, and `stated_score` are verified against ground-truth case and transaction records. Mismatches produce explicit tampering warnings.
5. **Secret & PII Redaction (`redact_secrets`)**: AWS keys (`AKIA...`), GitHub/API tokens (`ghp_...`, `sk-live-...`), Bearer tokens, 16-digit card PANs, and SSNs are scrubbed before being stored in `claims_of_authority` or `Attestation` records.

## 2. Measured Results on the 18 Exam Cases

| Case ID | Trigger Type | `customer_denies` | `referenced_txn_ids` | `stated_amount` | `stated_score` | Authority Claims | Injection Spans |
|---|---|---|---|---|---|---|---|
| HHF-001 | `analyst_request` | `False` | `[3660108]` | `None` | `None` | 0 | `[]` |
| HHF-002 | `analyst_request` | `False` | `[3660127]` | `None` | `None` | 0 | `[]` |
| HHF-003 | `customer_report` | `True` | `[3660153]` | `None` | `None` | 0 | `[]` |
| HHF-004 | `customer_report` | `True` | `[3660260]` | `None` | `None` | 3 | `[(114, 202, 'authority_claim'), ...]` |
| HHF-005 | `customer_report` | `True` | `[3660250]` | `None` | `None` | 3 | `[(114, 202, 'authority_claim'), ...]` |
| HHF-006 | `analyst_request` | `False` | `[3660263]` | `None` | `None` | 0 | `[]` |
| HHF-007 | `customer_report` | `True` | `[3660350]` | `$236.28` | `None` | 0 | `[]` |
| HHF-008 | `risk_score` | `False` | `[3660353]` | `$165.65` | `0.81` | 3 | `[(89, 166, 'tag_injection'), ...]` |
| HHF-009 | `risk_score` | `False` | `[3660355]` | `$228.01` | `0.80` | 0 | `[]` |
| HHF-010 | `risk_score` | `False` | `[3631383]` | `$1336.84` | `0.66` | 3 | `[(83, 161, 'authority_claim'), ...]` |
| HHF-011 | `risk_score` | `False` | `[3640919]` | `$881.55` | `0.71` | 0 | `[]` |
| HHF-012 | `risk_score` | `False` | `[3655457]` | `$261.51` | `0.53` | 0 | `[]` |
| HHF-013 | `risk_score` | `False` | `[3626720]` | `$409.30` | `0.69` | 0 | `[]` |
| HHF-014 | `risk_score` | `False` | `[3602445]` | `$1053.79` | `0.57` | 3 | `[(83, 152, 'tag_injection'), ...]` |
| HHF-015 | `risk_score` | `False` | `[3601725]` | `$112.75` | `0.71` | 0 | `[]` |
| HHF-016 | `risk_score` | `False` | `[3611043]` | `$3008.76` | `0.79` | 4 | `[(83, 166, 'authority_claim'), ...]` |
| HHF-017 | `risk_score` | `False` | `[3638829]` | `$43.92` | `0.55` | 0 | `[]` |
| HHF-018 | `risk_score` | `False` | `[3612011]` | `$34.22` | `0.80` | 0 | `[]` |

## 3. Red-Team Benchmark (`tests/r2/redteam/attacks.json`)

- **45 reproducible red-team attacks** tested in `tests/r2/test_untrusted.py`:
  - Unicode Cyrillic homoglyphs, zero-width spaces, smart quotes (`‘...’`, `“...”`), and unquoted messages.
  - Contractions (`don't`, `didn't`, `it's`) inside single-quoted customer reports.
  - Negation traps (`cannot confirm`, `do not recognize`) and double negations (`did NOT not make`).
  - Multi-line customer reports and multi-line analyst requests, plus benign `"Note from branch manager"` phrasing.
  - Inline `[[assistant: ...]]` tag injections embedded inside customer quote blocks.
  - Non-`3xxxxxx` transaction IDs (`4012345`), mismatched transaction IDs, amounts, and risk scores.
  - Secrets/PII (`AKIA...`, `sk-live-...`, `ghp_...`, 16-digit PANs) scrubbed before attestation.
  - 200 KB adversarial input benchmarked at `< 0.5 s`.
