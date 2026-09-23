"""The document side of the evidence: the policy rules and pattern definitions
the brief publishes, quoted by section so a case file can cite the text a
decision rests on (evidence source "document")."""

from __future__ import annotations

import re

RULES = {
    "R1": "Verify before you block on a weak signal: a single signal below 0.70 gets VERIFY_WITH_CUSTOMER or "
          "STEP_UP_AUTH before any block.",
    "R2": "Customer denies the transaction: BLOCK_CARD and CREATE_CASE; FILE_REPORT if exposure exceeds $1,000 "
          "or the case connects to a shared device profile or another card's fraud.",
    "R3": "Customer confirms the transaction: CLOSE_NO_FRAUD, with the confirmation noted in the case file.",
    "R4": "No reply within 24 hours: MONITOR_CARD and DECLINE_TRANSACTION for pending authorizations; escalate "
          "if exposure exceeds $500.",
    "R5": "Card testing: DECLINE_TRANSACTION and STEP_UP_AUTH; BLOCK_CARD if a purchase over $100 has cleared.",
    "R6": "Shared origin: name the shared element, CREATE_CASE, FILE_REPORT, and MONITOR_CONNECTED_CARDS for "
          "every card that shares it.",
    "R7": "Disputed but legitimate recurring charge: CREATE_CASE, VERIFY_WITH_CUSTOMER, WARN_CUSTOMER; do not block.",
    "R8": "Uncertain and exposed (over $500) or conflicting evidence: ESCALATE_TO_ANALYST.",
    "R9": "Undocumented coordinated abuse: CREATE_CASE, FILE_REPORT, ESCALATE_TO_ANALYST, and describe the "
          "pattern in your own words.",
    "R10": "Never BLOCK_ALL_CARDS unless two of the customer's cards show confirmed fraud or credentials are "
           "confirmed compromised.",
    "§3a": "A case is opened at probability 0.30, on any evidence request, or on any dispute; a report is filed "
           "only when fraud is confirmed or strongly suspected and exposure exceeds $1,000, the activity connects "
           "to a shared device, region or another customer's fraud, or the pattern is coordinated or undocumented.",
    "§6": "Stop at probability >= 0.85 or <= 0.15 on two independent pieces of evidence, when a verification "
          "response settles it, or when further steps would not change the decision.",
}

PATTERNS = {
    "card_testing": "Card testing: three or more tiny online authorizations, often under $5, then a larger "
                    "purchase. Confirmed by the sequence itself.",
    "card_not_present_fraud": "Card-not-present fraud: the number used online without the card; amounts and "
                              "products that don't fit the cardholder, often two to four within 48 hours.",
    "card_not_present_new_device": "Card-not-present fraud from a new device: as above, with the identity record "
                                   "marking the device New for this account, sometimes behind a proxy.",
    "out_of_region_use": "Out-of-region use: card-present purchases in a billing region the cardholder has no "
                         "history in, while their normal activity continues at home.",
    "account_takeover": "Account takeover: mixed-channel activity inconsistent with the cardholder, often with "
                        "device and match-flag anomalies, pointing to stolen credentials.",
    "undocumented": "The known patterns are not the only ones: activity that fits none of them is described in "
                    "the analyst's own words.",
}


def rules_cited(reasons: list[str]) -> list[str]:
    seen = []
    for r in reasons:
        for m in re.findall(r"(R10|R[1-9]|§3a|§6)", r):
            if m not in seen and m in RULES:
                seen.append(m)
    return seen
