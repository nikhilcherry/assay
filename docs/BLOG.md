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

Twenty cases in, the result was 10 fraud, 8 legitimate, 2 held for a human.
Eight alerts where the bank's model said 0.52 to 0.90 closed as legitimate. The
brief warns that "an agent that blocks everything scores badly". This one blocks ten cards.

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
ratio of about 8, not 40. The other bursts were ordinary people buying
ordinary expensive things.

Then I measured it again, properly, and 8 was still wrong. That 8 is how much
more often a burst is fraud than an average transaction, but the agent applies
it *on top of* the model score, and the model already sees most of what makes
those bursts suspicious. The right number is the one that makes model-plus-finding
match what actually happened, measured on scores from a model that never saw
that month: ×1.64. Card testing went from an assumed ×20 to ×2.07, case memory
from ×3 to ×1.11. The monitor, which sweeps the exam period for bursts on its own,
now clears 15 of the 19 it finds. HHG-006 is still fraud at 0.99, because its strongest transaction
already scores high and the customer disputed it. Now the number behind that is
one I can defend.

Every likelihood ratio I couldn't measure is listed in the README as an assumption.

The second one I got wrong was bigger, and I found it on deadline day. The agent
treated a cardholder's "I never made this purchase" as tripling the odds. So a
disputed charge the model scored at 0.01 came out at 0.03, and the agent
simulated the customer recognising the purchase and withdrawing the dispute. It
cleared three disputes that way. Then I counted again. In October, the month the
model never saw, the bank investigated 1,203 disputes, and **every one** was
fraud, including the 15% the model scored under 0.05. A low score does not clear
a denial, and the customer had already denied it: inventing a retraction
contradicts the case itself.

But the brief also says half the exam is legitimate, so some disputes may be
planted on legitimate transactions, which the history can't show. For those the
model score *is* the evidence, and its two distributions are measurable: under
0.005 it is 14 times more common on a legitimate transaction than on disputed
fraud. So the agent now averages the two readings, weighted equally, and says
that the equal weight is the assumption. Two of the three disputes became fraud.
The third, at 0.52, goes to a human, which is what the policy says to do with
evidence that doesn't settle the question.

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
cards, 4 structuring episodes (of 19 it looked at), and 14 frauds the bank's model had rated under 0.30.

## Is it right?

I replayed every case the bank closed in October through the agent, blind:
scores from a model that never saw October, a closed-case history that ends when
each alert opens, and the customer's complaint removed from the fraud cases. The
bank's own score, at 0.5, catches 48.5% of the confirmed fraud and flags 100% of
the alerts its analysts went on to clear, because those alerts *are* its
flags. assay catches 59% and flags 4.9%.

And one result I didn't love: on those ordinary cases, the agent's
probabilities are no better than its model's. The graph evidence rarely fires on
an average case. Its value is the case the model can't see at all, like a ring
of 28 cards the model scored 0.06. I'd rather say that than hide it.

## Watching it think

An agent's answer file is the end of the story, but judges and analysts want to
see the middle. So every investigation is also exported as a trace: each graph
query in the order the agent ran it, the vertices it returned, and the
probability after each piece of evidence. The replay at
**nikhilcherry.github.io/assay** draws it. The graph grows query by query, and a
needle on a log-odds scale moves with each likelihood ratio. For HHG-014 the
reading goes from 0.06 at the model score, to 0.65 when the device ring appears
(×30), to 0.85 when the bank's own closed cases on that handset turn up (×3).
The brass tick for the bank's score stays at 0.05 the whole time.

Nothing in the replay is animated for effect. The traces come from rerunning the
same agent, and the export refuses to write if a single verdict, probability or
connected card differs from the submitted answer files. The tests check that
every ledger adds up in log-odds and that every graph claim cites a query the
agent actually ran.

## What I'd do with another week

Replace the simulated customer replies with a proper model of reply behaviour,
measure the likelihood ratios that are still assumptions, and put the SAR
narratives through a regulator-style checklist. The code for all of it is at
the link below: 427 tests, and everything reproduces from the four CSVs.

*Repo: github.com/nikhilcherry/assay · Replay: nikhilcherry.github.io/assay ·
Demo video: github.com/nikhilcherry/assay/releases/tag/replay-v1*
