# Social post

**LinkedIn / X (thread-able):**

Built an agentic fraud investigator on @TigerGraphDB for #HackerHouseGoa Task 4.

The dataset gives you 5,565 "closed cases" as the bank's memory. I counted them first:
they name 14,055 fraud transactions, 3.4% of Jul–Oct, which is the entire fraud rate.
The memory is a complete labelled training set.

So the agent's probabilities are measured, not guessed:
• AUC 0.964 vs 0.866 for the bank's own risk score (held-out month)
• calibrated: when it says 0.20, 21% were fraud

The best part only shows up in the graph: one Samsung handset behind an anonymous
proxy, "New" on every card, 60 purchases across 28 unrelated cardholders. Every
purchase scored low on the bank's model. You can only see it by walking device → cards.

8 fraud · 11 legitimate · 1 uncertain across 20 cases. It blocks 8 cards, not 20.
Plus 60 investigations it started on its own.

Repo + write-up: github.com/nikhilcherry/assay

#TigerGraph #GraphRAG #FraudDetection #HHGoa2026
