# Block B20 -- Analyst in the Loop (Challenge 8) Results

## Summary of Measured Results

The feedback system and demonstration were evaluated using a realistic 1-round analyst review flow over closed cases and the shared negative card-days population (legitimate cards from Jul-Oct not named in any case):
- **Feedback Training Window**: July 1, 2016 – August 31, 2016 (Top 100 highest-scoring alerts reviewed by analyst)
- **Evaluation Window**: September 1, 2016 – October 31, 2016

## Performance Metrics (Before vs. After 1 Analyst Feedback Round)

```
====================================================================================
ANALYST IN THE LOOP (CHALLENGE 8) REALISTIC 1-ROUND FEEDBACK EVALUATION (SEP-OCT)
====================================================================================
Metric                              | Before Feedback | After Feedback  | % Change  
------------------------------------------------------------------------------------
Wrongly Acted Against (Legit)       | 8133            | 134             | -98.4%
Fraud Caught (Confirmed Fraud)      | 2184            | 2184            | +0.0%
Precision (Fraud / Total Acted)     | 0.2117          | 0.9422          | +345.1%
------------------------------------------------------------------------------------
Top Alerts Reviewed by Analyst (Jul-Aug): 100
Feedback Decisions Logged: 100
------------------------------------------------------------------------------------
EXACT SEP-OCT FINDINGS CHANGED BY ANALYST DECISIONS (SAMPLE 3):
  1. Case ID: CC-3788
     Outcome: cleared
     Element Value: 'unknown_device'
     LR Change: 5.0 -> 1.0
     Causal Decision IDs: ['DEC-0001-48f8f863', 'DEC-0002-488c7271', 'DEC-0003-9d07b3c9']
  2. Case ID: CC-3790
     Outcome: cleared
     Element Value: 'unknown_device'
     LR Change: 5.0 -> 1.0
     Causal Decision IDs: ['DEC-0001-48f8f863', 'DEC-0002-488c7271', 'DEC-0003-9d07b3c9']
  3. Case ID: CC-3791
     Outcome: cleared
     Element Value: 'unknown_device'
     LR Change: 5.0 -> 1.0
     Causal Decision IDs: ['DEC-0001-48f8f863', 'DEC-0002-488c7271', 'DEC-0003-9d07b3c9']
====================================================================================
```

## Key Findings

1. **Clear Drop in Wrongly-Acted-Against Customers**: False actions against legitimate customers fell from **8,133 down to 134** (**-98.4%**) after 1 feedback round on top 100 alerts.
2. **Zero Loss in Fraud Retention**: Confirmed fraud caught remained unchanged at **2,184 / 2,184** (**100.0% retained**) because posterior suppression only triggers when element values are proven predominantly legitimate (`Beta posterior mean <= 0.2 AND rejects >= 3`).
3. **Massive Precision Gain**: Investigation precision surged from **21.17% to 94.22%** (+345.1%).
4. **Attested Decision Lineage**: Every changed finding records the exact list of analyst decision IDs that caused its LR adjustment.
