# The bank already had the answer key. It just filed it under "memory".

*Building assay, an agentic fraud investigator on TigerGraph, for Hacker House Goa 2026.*

The brief for this task hands you six months of card transactions (590,742 of
them, from the IEEE-CIS dataset), strips out the fraud label, and replaces it with
a risk score from the bank's own model. Then it gives you twenty alerts from
November and December and asks an agent to investigate each one: what kind of
fraud is this, how far does it go, what should the bank do, and who has to approve it.

It also gives you something it describes modestly: 5,565 *closed cases*, the
bank's finished investigations from July to October. "Your labelled history and
your agent's starting memory," the brief calls it. Most people will load it into
a vector store and retrieve similar cases. The first thing I did was count.

## 14,055

The closed cases name 14,055 transactions as confirmed fraud. Between July and
October there are 417,404 transactions in total. That is 3.4%, which is
the fraud rate of the IEEE-CIS data itself.

So the closed cases aren't a sample of the fraud. They are, near enough, *all*
of it. Any July–October transaction that no closed case mentions is
legitimate with high probability. The bank's memory is a complete labelled
training set, hiding in plain sight.

That changes what the agent can be. Instead of a pile of heuristics ("new device
+30 points, out of region +25"), you can train a model on exactly the bank's own
conclusions and *measure* it. Trained on July to September and scored on October,
which it never saw:

- the bank's risk score: AUC **0.866**, average precision 0.25
- assay's model: AUC **0.964**, average precision **0.64**

And because the brief says `fraud_probability` is scored for calibration, the
model is calibrated on that held-out month. When assay says 0.20, about one in
five of those transactions was fraud (0.201 predicted, 0.212 observed). When it
says 0.77, it's 0.768.

No original Kaggle labels were touched. The only supervision is the bank's own case file.

## A probability is not a verdict

A calibrated score is where an investigation starts, not where it ends. The
agent is a small state machine around it:

1. Pull the flagged transaction and the cardholder's history from TigerGraph.
2. Build a baseline. This needs care, because `customer_id` in this dataset is
   an issuer bucket, not a person. One card in the case pack has 10,332 transactions. So
   assay resolves a **holder**: card, billing region, and the account-open day
   implied by the `D1` column. That's the level at which "is this normal for
   them?" means anything.
3. Run detectors for the things one transaction's score can't see: bursts,
   sequences, and shared devices.
4. Combine: the model's log-odds plus the likelihood ratio of each finding.
5. Apply policy §6 literally: stop at ≥ 0.85 or ≤ 0.15 on two independent pieces
   of evidence, or ask for more.
6. Run the fraud policy (encoded as code with 50 tests) *twice*: once before the
   evidence comes back, once after. That's how `initial` and `final` actions
   come out genuinely different, instead of as two copies of one list.

Twenty cases in, the result was 8 fraud, 11 legitimate, 1 uncertain. Eight
alerts where the bank's model said 0.52 to 0.90 closed as legitimate. The brief
warns that "an agent that blocks everything scores badly". This one blocks eight cards.

## The fraud you can only see in the graph

Case HHG-014 was an analyst's hunch: "several cards this month show purchases
from the same unusual device profile." The flagged transaction was $74.96 with a
bank risk score of 0.05. On its own it looks like nothing.

Starting from that transaction, the graph query goes to its device profile and
then to every other card that used it. The same Samsung SM-G935F handset,
Android 7.0, Chrome 62, 1920x1080, behind an anonymous proxy and marked *New*
every single time, placed 60 purchases on **28 unrelated cards** in three weeks.
Every one of them scored low. No individual card looks compromised. The fraud
exists only as a shape.

I wanted to know whether "one device, many cards" was a real signal or just
popular phones, so I measured it on July–October. 205 specific device profiles
appear on five or more cards. Require "New on ≥90% of uses" and 13 remain;
their transactions are 1.5% fraud, which is just popular phones. Add "behind a
proxy on ≥90% of uses" and **exactly one** remains, the same handset. It's in
four closed cases from August and September that the bank's analysts could not
match to any known typology.

So assay files it as *undocumented*: device-ring fraud. That means a case, a
suspicious activity report, and monitoring on all 27 connected cards (policy R6
and R9).

## Measuring my own assumptions, and finding one was wrong

Case HHG-006 was four online purchases in 30 minutes, each just under $500.
The closed cases had five examples of exactly that shape, all confirmed fraud.
My first instinct was to treat the shape as damning and give it a likelihood
ratio of 40.

Then I measured it. Across July–October there are 41 bursts like that. **Nine**
are fraud. Precision 0.22 against a base rate of 0.034 works out to a likelihood
ratio of about 8, not 40. The other 32 bursts were ordinary people buying
ordinary expensive things. The detector now uses 8, and the monitor, which
sweeps the exam period for these bursts on its own, cleared 7 of the 19 it found
as legitimate. HHG-006 is still fraud at 0.98, because its strongest transaction
already scores high and the customer disputed it. Now the number behind that is
one I can defend.

Every likelihood ratio I couldn't measure is listed in the README as an assumption.

## Case memory, in the graph

Each finished investigation is written back to TigerGraph as a `FraudCase`
vertex, with edges to its card, the transactions it cites, the devices it
implicates and the closed cases it drew on. Alerts are investigated in the order
they were opened, so an investigation can only find cases closed before it.
That keeps the memory causal instead of peeking at the future.

The agent's cases are a different vertex type from the bank's closed cases. The
bank's cases are the training labels, and the agent writing into its own
calibration set is the kind of leak that looks fine until it doesn't.

## Going looking

The brief offers an optional extra: let the agent watch the exam period on its
own. assay's monitor runs three scans (device rings, near-$500 bursts, and
transactions where its model and the bank's disagree sharply). It raised 60
alerts nobody asked for and investigated each with the same agent. 27 ring
cards, 9 structuring episodes, and 14 frauds the bank's model had rated under 0.30.

## What I'd do with another week

Replace the simulated customer replies with a proper model of reply behaviour,
measure the likelihood ratios that are still assumptions, and put the SAR
narratives through a regulator-style checklist. The code for all of it is at
the link below: 101 tests, and everything reproduces from the four CSVs.

*Repo: github.com/nikhilcherry/assay*
