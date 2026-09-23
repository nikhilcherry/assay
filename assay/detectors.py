"""Detectors: each looks at one aspect of a case and returns Findings.

A Finding is a claim with provenance -- the query that produced it and the
IDs it rests on -- which is exactly the shape of an `evidence[]` entry in the
answer file. Findings that carry a likelihood ratio move the probability;
findings with lr=1 are context an analyst needs but that the model score
already accounts for (device novelty, amount, product code: all model inputs),
so counting them again would double-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMALL_AUTH_USD = 5.0          # brief: card testing is "often under $5"
STRUCTURING_LOW, STRUCTURING_HIGH = 400.0, 500.0   # "each just under $500" (closed cases CC-3748 et al.)
RING_MIN_CARDS = 5            # a device profile on this many cards in the window is a ring, not a household
RING_MIN_NEW = 0.9            # ...marked New on (nearly) every use,
RING_MIN_PROXY = 0.9          # ...and behind a proxy on (nearly) every use.
# Measured on July-October: of 205 specific device profiles seen on >= 5 cards,
# exactly one also meets both share thresholds (the SM-G935F profile the bank's
# analysts filed as undocumented in CC-2649 et al.). Without the proxy clause,
# 13 profiles qualify and their transactions are 1.5% fraud -- popular handsets.

# Structuring: 41 near-$500 bursts on one card within an hour in July-October,
# 9 of them confirmed fraud. Precision 0.22 against a 0.034 base rate is a
# likelihood ratio of ~8, and it is applied to the burst's strongest model
# score rather than to the flagged transaction alone.
LR_STRUCTURING = 8.0


def is_ring(nb: pd.DataFrame) -> bool:
    return (nb.card_id.nunique() >= RING_MIN_CARDS and (nb.id_15 == "New").mean() >= RING_MIN_NEW
            and nb.id_23.notna().mean() >= RING_MIN_PROXY)


@dataclass
class Finding:
    claim: str
    ref: str
    entity_ids: list = field(default_factory=list)
    source: str = "graph"
    lr: float = 1.0           # likelihood ratio applied to the fraud odds
    independent: bool = True  # counts toward policy §6's "two independent pieces"
    tag: str = ""

    def as_evidence(self) -> dict:
        return {"claim": self.claim, "source": self.source, "ref": self.ref,
                "entity_ids": [str(e) for e in self.entity_ids]}


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def structuring(card_win: pd.DataFrame, t0) -> pd.DataFrame | None:
    """Several online purchases just under $500 within an hour of each other."""
    w = card_win[(card_win.channel == "online") & (card_win.TransactionAmt >= STRUCTURING_LOW)
                 & (card_win.TransactionAmt < STRUCTURING_HIGH)]
    w = w[(w.t - t0).abs() <= pd.Timedelta(hours=1)]
    return w if len(w) >= 3 else None


def card_testing(card_win: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """>=3 small online authorizations within an hour, then a larger purchase (R5)."""
    on = card_win[card_win.channel == "online"].sort_values("t")
    small = on[on.TransactionAmt < SMALL_AUTH_USD]
    if len(small) < 3:
        return None
    for i in range(len(small) - 2):
        run = small.iloc[i:]
        run = run[run.t <= small.iloc[i].t + pd.Timedelta(hours=1)]
        if len(run) >= 3:
            after = on[(on.t > run.t.max()) & (on.t <= run.t.max() + pd.Timedelta(hours=24))
                       & (on.TransactionAmt >= 20)]
            if len(after):
                return run, after.head(3)
    return None


def recurring(history: pd.DataFrame, flagged) -> pd.DataFrame | None:
    """Same amount, same product code on this holder, roughly monthly (R7)."""
    amt = flagged.TransactionAmt
    same = history[(history.TransactionAmt - amt).abs() <= max(0.02, 0.005 * amt)]
    same = same[(same.ProductCD == flagged.ProductCD) & (same.TransactionID != flagged.TransactionID)]
    before = same[same.t < flagged.t].sort_values("t")
    if len(before) < 2:
        return None
    gaps = before.t.diff().dropna().dt.days.tolist() + [(flagged.t - before.t.iloc[-1]).days]
    monthly = [g for g in gaps if 25 <= g <= 35]
    return before if len(monthly) >= 2 else None


def is_generic_device(profile: str) -> bool:
    """'Windows | ? | chrome 66.0 | ?' is half the internet. Only a specific
    handset model with a full profile can link people."""
    if not profile:
        return True
    info = profile.split(" | ")[0]
    return info in ("?", "Windows", "MacOS", "iOS Device", "Trident/7.0", "rv:11.0", "Linux") or profile.count("?") >= 2


def holder_baseline(hist: pd.DataFrame, before) -> dict:
    h = hist[hist.t < before]
    if not len(h):
        return {"n": 0}
    return {
        "n": int(len(h)),
        "median": float(h.TransactionAmt.median()),
        "p95": float(h.TransactionAmt.quantile(0.95)),
        "regions": set(h.addr1.dropna().astype(int).tolist()),
        "products": set(h.ProductCD.dropna()),
        "devices": set(h.device_profile[h.device_profile != ""]),
        "online_share": float((h.channel == "online").mean()),
        "first": h.t.min(), "last": h.t.max(),
    }


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def combine(prior: float, findings: list[Finding]) -> float:
    return sigmoid(logit(prior) + sum(np.log(f.lr) for f in findings if f.lr != 1.0))
