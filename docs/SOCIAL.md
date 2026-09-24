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

10 fraud · 8 legitimate · 2 held for a human across 20 cases. It blocks 10 cards, not 20.
Plus 60 investigations it started on its own.

Watch it think: every investigation replayed query by query, with the graph growing
and the probability needle moving on each piece of evidence:
nikhilcherry.github.io/assay

Repo + write-up: github.com/nikhilcherry/assay

#TigerGraph #GraphRAG #FraudDetection #HHGoa2026
