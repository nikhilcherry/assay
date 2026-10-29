# Demo cheat sheet: all 18 cases

Generated from the live console (`ASSAY_TG=1 python -m assay.r2.console --port 8742`). Each block is what the trace shows when the case is run; LRs are the ones the ledger applied. `link (x1)` is a detector that tied the case to other cards but carries no measured weight; `escalated` is a guard that sends the case to an analyst. The TigerVector precedent query runs on every case but abstains (no vote), so it never appears here.

## HHF-001: fraud p=0.99, undocumented, $2,189.02
- Trigger: analyst_request, txn 3660108, card C05061-K1. Model starts at 0.1892.
- x68.6622 -> 0.9412: Coordinated loss-shape cohort across 3 cards (bucket '3', measured LR=714.57, 95% CI low=68.66: 2-3 online ProductCD C charges on novel devices with no shared device); no common origin found above matched-control threshold (best c
- x8 -> 0.9923: Evidence request answered: Cardholder states they did not make this transaction and still has the card.
- Reported fraud_probability 0.99 (ledger 0.9923, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: BLOCK_CARD (L1, awaiting approval), CREATE_CASE (auto, auto_executed), MONITOR_CONNECTED_CARDS (auto, auto_executed), FILE_REPORT (L2, awaiting approval), ESCALATE_TO_ANALYST (auto, auto_executed)
- 10 tool calls in 3525 ms; affected 3 txn(s), 2 connected card(s), SAR=True.

## HHF-002: fraud p=0.94, undocumented, $2,166.71
- Trigger: analyst_request, txn 3660127, card C06220-K1. Model starts at 0.0714.
- x23.4945 -> 0.6438: Affected loss-shape cohort (12 cards, bucket '6+', measured LR=357.29) shares common origin ProductCD_addr1:W_122 within 60 days before loss (matched lift=14.3x, Fisher p=6.51e-05, cardinality=142).
- x8 -> 0.9353: Evidence request answered: Cardholder states they did not make this transaction and still has the card.
- Reported fraud_probability 0.94 (ledger 0.9353, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: BLOCK_CARD (L1, awaiting approval), CREATE_CASE (auto, auto_executed), MONITOR_CONNECTED_CARDS (auto, auto_executed), FILE_REPORT (L2, awaiting approval), ESCALATE_TO_ANALYST (auto, auto_executed)
- 9 tool calls in 3291 ms; affected 3 txn(s), 11 connected card(s), SAR=True.

## HHF-003: fraud p=0.99, undocumented, $2,339.55
- Trigger: customer_report, txn 3660153, card C11663-K1. Model starts at 0.0010.
- x3157.07 -> 0.7596: Cardholder C11663 reported the $1,800.00 charge as not theirs (case pack trigger, 2017-01-31 20:31:28). Measured on October: all 1,203 disputes the bank investigated were fraud, 15% of them scored under 0.05, so a low score does n
- x5.653 -> 0.9470: Card C11663-K1 woke up after 21.8d card silence with burst of 2 txns (max $357.82, total $539.55) in 48h.
- x23.4945 -> 0.9976: Coordinated loss-shape cohort across 11 cards (bucket '6+', measured LR=357.29, 95% CI low=23.49: 2-3 online ProductCD C charges on novel devices with no shared device); no common origin found above matched-control threshold (best
- Reported fraud_probability 0.99 (ledger 0.9976, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: BLOCK_CARD (L1, awaiting approval), CREATE_CASE (auto, auto_executed), MONITOR_CONNECTED_CARDS (auto, auto_executed), FILE_REPORT (L2, awaiting approval), ESCALATE_TO_ANALYST (auto, auto_executed)
- 10 tool calls in 3464 ms; affected 3 txn(s), 10 connected card(s), SAR=True.

## HHF-004: fraud p=0.76, undocumented, $23.99
- Trigger: customer_report, txn 3660260, card C12975-K2. Model starts at 0.0010.
- x3157.07 -> 0.7596: Cardholder C12975 reported the $2.90 charge as not theirs (case pack trigger, 2017-01-26 21:09:37). Measured on October: all 1,203 disputes the bank investigated were fraud, 15% of them scored under 0.05, so a low score does not c
- link (x1) [micro_drip_cohort]: Coordinated sub-threshold micro-drip cohort on card C12975-K2: 6 online ProductCD=='C' charges < $6.50 over >= 7d with 0 prior in 60d; 16 other card(s) (17 total, threshold K>=6 from Aug-Nov base rate 1-6/mo) have overlapping 30d 
- Reported fraud_probability 0.76 (ledger 0.7596, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: BLOCK_CARD (L1, awaiting approval), CREATE_CASE (auto, auto_executed), MONITOR_CONNECTED_CARDS (auto, auto_executed), FILE_REPORT (L2, awaiting approval), ESCALATE_TO_ANALYST (auto, auto_executed)
- 9 tool calls in 1213 ms; affected 6 txn(s), 16 connected card(s), SAR=True.

## HHF-005: fraud p=0.96, undocumented, $23.17
- Trigger: customer_report, txn 3660250, card C00348-K1. Model starts at 0.0010.
- x3157.07 -> 0.7596: Cardholder C00348 reported the $4.05 charge as not theirs (case pack trigger, 2017-01-28 23:45:16). Measured on October: all 1,203 disputes the bank investigated were fraud, 15% of them scored under 0.05, so a low score does not c
- x8.52 -> 0.9642: Card C00348-K1 is in wcc ring RING-WCC-c4e13e20 containing 12 cards ($29,568.99 vol, 7 prior fraud members)
- link (x1) [micro_drip_cohort]: Coordinated sub-threshold micro-drip cohort on card C00348-K1: 6 online ProductCD=='C' charges < $6.50 over >= 7d with 0 prior in 60d; 20 other card(s) (21 total, threshold K>=6 from Aug-Nov base rate 1-6/mo) have overlapping 30d 
- Reported fraud_probability 0.96 (ledger 0.9642, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: BLOCK_CARD (L1, awaiting approval), CREATE_CASE (auto, auto_executed), MONITOR_CONNECTED_CARDS (auto, auto_executed), FILE_REPORT (L2, awaiting approval), ESCALATE_TO_ANALYST (auto, auto_executed)
- 10 tool calls in 1135 ms; affected 6 txn(s), 31 connected card(s), SAR=True.

## HHF-006: uncertain p=0.15, none, $967.68
- Trigger: analyst_request, txn 3660263, card C01727-K2. Model starts at 0.1517.
- link (x1) [FanIn]: Recipient fan-in cohort on R_emaildomain=sc.rr.com (online ProductCD=='C', amount >= $100): 12 distinct cards used it for the first time in 21d before 2017-01-28 10:59:42 (Jul-Dec baseline mean=0.00, max=0, Poisson p=0.00e+00); fl
- escalated (x1), for analyst review: FanIn tie this card to 11 other card(s) (C00534-K2, C01357-K1, C01507-K1, C01594-K1, C02167-K2, C03060-K1). One cardholder's confirmation cannot clear activity that spans other cards.
- Reported fraud_probability 0.15 (ledger 0.1517, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: MONITOR_CONNECTED_CARDS (auto, auto_executed), VERIFY_WITH_CUSTOMER (auto, auto_executed), STEP_UP_AUTH (auto, auto_executed), CREATE_CASE (auto, auto_executed), ESCALATE_TO_ANALYST (auto, auto_executed)
- 9 tool calls in 1766 ms; affected 3 txn(s), 11 connected card(s), SAR=False.

## HHF-007: fraud p=0.91, card_testing, $246.38
- Trigger: customer_report, txn 3660350, card C09582-K1. Model starts at 0.0071.
- x2.07 -> 0.0146: Card testing: 4 online authorizations under $5 within an hour ($1.53, $3.46, $3.33, $1.78), followed by 1 larger purchase(s) up to $236.28.
- x706.966 -> 0.9129: Cardholder C09582 reported the $236.28 charge as not theirs (case pack trigger, 2017-01-14 15:57:30). Measured on October: all 1,203 disputes the bank investigated were fraud, 15% of them scored under 0.05, so a low score does not
- Reported fraud_probability 0.91 (ledger 0.9129, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: BLOCK_CARD (L1, awaiting approval), CREATE_CASE (auto, auto_executed)
- 8 tool calls in 1232 ms; affected 5 txn(s), 0 connected card(s), SAR=False.

## HHF-008: uncertain p=0.01, none, $165.65
- Trigger: risk_score, txn 3660353, card C07871-K1. Model starts at 0.0057.
- escalated (x1), for analyst review: The alert describes transaction 3660353 as online but the bank records it as in-person, so the alert and the record disagree on how the card was used.
- Reported fraud_probability 0.01 (ledger 0.0057, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: VERIFY_WITH_CUSTOMER (auto, auto_executed), CREATE_CASE (auto, auto_executed), ESCALATE_TO_ANALYST (auto, auto_executed)
- 8 tool calls in 1294 ms; affected 1 txn(s), 0 connected card(s), SAR=False.

## HHF-009: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3660355, card C09177-K1. Model starts at 0.0102.
- x0.0667 -> 0.0007: Evidence request answered: Cardholder confirms they made the transaction.
- Reported fraud_probability 0.01 (ledger 0.0007, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: CREATE_CASE (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1460 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-010: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3631383, card C07196-K2. Model starts at 0.0330.
- x0.0667 -> 0.0023: Evidence request answered: Cardholder confirms they made the transaction.
- Reported fraud_probability 0.01 (ledger 0.0023, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: CREATE_CASE (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1900 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-011: uncertain p=0.01, none, $881.55
- Trigger: risk_score, txn 3640919, card C09241-K1. Model starts at 0.0102.
- escalated (x1), for analyst review: The case names card C09241-K1 but the bank records transaction 3640919 on C09241-K2, so the cardholder who would be asked is not established.
- Reported fraud_probability 0.01 (ledger 0.0102, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: VERIFY_WITH_CUSTOMER (auto, auto_executed), CREATE_CASE (auto, auto_executed), ESCALATE_TO_ANALYST (auto, auto_executed)
- 8 tool calls in 1318 ms; affected 1 txn(s), 0 connected card(s), SAR=False.

## HHF-012: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3655457, card C12223-K1. Model starts at 0.0010.
- No weighed evidence beyond the model score.
- Reported fraud_probability 0.01 (ledger 0.0010, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: ALLOW_TRANSACTION (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1395 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-013: legitimate p=0.05, none, $0.00
- Trigger: risk_score, txn 3626720, card C06115-K1. Model starts at 0.0528.
- No weighed evidence beyond the model score.
- Reported fraud_probability 0.05 (ledger 0.0528, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: ALLOW_TRANSACTION (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1141 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-014: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3602445, card C00750-K2. Model starts at 0.0010.
- x0.0667 -> 0.0001: Evidence request answered: Cardholder confirms they made the transaction.
- Reported fraud_probability 0.01 (ledger 0.0001, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: CREATE_CASE (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1255 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-015: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3601725, card C00543-K1. Model starts at 0.0010.
- No weighed evidence beyond the model score.
- Reported fraud_probability 0.01 (ledger 0.0010, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: ALLOW_TRANSACTION (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 924 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-016: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3611043, card C02502-K2. Model starts at 0.0036.
- x0.0667 -> 0.0002: Evidence request answered: Cardholder confirms they made the transaction.
- Reported fraud_probability 0.01 (ledger 0.0002, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: CREATE_CASE (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1223 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-017: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3638829, card C08842-K2. Model starts at 0.0010.
- No weighed evidence beyond the model score.
- Reported fraud_probability 0.01 (ledger 0.0010, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: ALLOW_TRANSACTION (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 989 ms; affected 0 txn(s), 0 connected card(s), SAR=False.

## HHF-018: legitimate p=0.01, none, $0.00
- Trigger: risk_score, txn 3612011, card C02753-K1. Model starts at 0.0010.
- No weighed evidence beyond the model score.
- Reported fraud_probability 0.01 (ledger 0.0010, rounded to 2 places and clipped to [0.01, 0.99]).
- Actions: ALLOW_TRANSACTION (auto, auto_executed), CLOSE_NO_FRAUD (auto, auto_executed)
- 8 tool calls in 1192 ms; affected 0 txn(s), 0 connected card(s), SAR=False.
