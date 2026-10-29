"""Burst hub: a specific device element that appears from nowhere on several cards within 72 hours.

The ring detector drops elements seen on more than 15 cards, because issuer buckets share devices
with hundreds of cards for months. The red team (docs/r2/REDTEAM.md) used that cap to hide rings:
put 16+ stolen cards on one fresh device and the edge disappears. A burst hub is the opposite
of a bucket: the element (device, id_23 + device, or device + billing region, the ring's own keys)
had at most one card in the 60 days before the window and 3+ cards inside it. No upper cap.

Backtest (4,640 Jul-Oct confirmed-fraud openings vs 5,000 legitimate Jul-Oct card-days, as-of):
fires on 1.16% of fraud vs 0.10% of legitimate, LR 10.68, 95% CI [4.45, 25.64]; Jul-Aug 9.85,
Sep-Oct 19.95. The ledger uses the CI lower bound.
"""

from __future__ import annotations

import pandas as pd

from assay.detectors import is_generic_device
from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import CaseCtx, Detector, Finding

WINDOW_H = 72
PRIOR_DAYS = 60
MIN_CARDS = 3
MAX_PRIOR_CARDS = 1
LR_BURST_HUB = 4.45


def _elements(d: pd.DataFrame) -> pd.Series:
    """Element keys per row, as in assay.r2.rings.build_card_graph (specific devices only)."""
    ok = (d.device_profile != "") & ~d.device_profile.map(is_generic_device) & (d.device_profile.str.count(r"\?") == 0)
    d = d[ok]
    keys = [("dev:" + d.device_profile).rename("e").to_frame().assign(card=d.card_id, t=d.t)]
    has23 = d.id_23.notna() & (d.id_23.astype(str) != "") & (d.id_23.astype(str) != "nan")
    keys.append(("id23:" + d.id_23[has23].astype(str) + ":" + d.device_profile[has23]).rename("e").to_frame()
                .assign(card=d.card_id[has23], t=d.t[has23]))
    hasad = d.addr1.notna()
    keys.append(("dev_addr:" + d.device_profile[hasad] + ":" + d.addr1[hasad].astype(int).astype(str)).rename("e")
                .to_frame().assign(card=d.card_id[hasad], t=d.t[hasad]))
    return pd.concat(keys)


class BurstHub(Detector):
    name = "burst_hub"
    pattern = "undocumented:device_burst_hub"

    def run(self, ctx: CaseCtx) -> list[Finding]:
        lo = ctx.opened_at - pd.Timedelta(hours=WINDOW_H)
        plo = lo - pd.Timedelta(days=PRIOR_DAYS)
        mine = ctx.tx[(ctx.tx.card_id == ctx.card_id) & (ctx.tx.t >= lo)]
        if mine.empty:
            return []
        own = _elements(mine)
        if own.empty:
            return []
        devs = set(mine.device_profile)
        near = ctx.tx[(ctx.tx.t >= plo) & ctx.tx.device_profile.isin(devs)]
        el = _elements(near)
        best = None
        for e in sorted(set(own.e)):
            rows = el[el.e == e]
            recent = sorted(set(rows.card[rows.t >= lo]))
            prior = set(rows.card[rows.t < lo])
            if len(recent) >= MIN_CARDS and len(prior) <= MAX_PRIOR_CARDS and (best is None or len(recent) > len(best[1])):
                best = (e, recent, prior)
        if best is None:
            return []
        e, recent, prior = best
        others = [c for c in recent if c != ctx.card_id]
        txns = sorted(int(x) for x in near.TransactionID[near.card_id.isin(recent) & (near.t >= lo)
                                                          & near.device_profile.isin(devs)])
        claim = Attestation(
            what=(f"Burst hub: element {e} appeared on {len(recent)} cards in the {WINDOW_H}h before the case "
                  f"opened and on {len(prior)} card(s) in the {PRIOR_DAYS} days before that; this card is one of them "
                  f"(measured LR 10.68, weighed at its 95% CI lower bound {LR_BURST_HUB})."),
            evidence=Evidence(txn_ids=txns[:200], card_ids=recent,
                              customer_ids=sorted({c.rsplit("-K", 1)[0] for c in recent}), case_ids=[ctx.case_id]),
            query="burst_hub",
            params={"card_id": ctx.card_id, "as_of": str(ctx.opened_at), "window_h": WINDOW_H,
                    "prior_days": PRIOR_DAYS, "min_cards": MIN_CARDS},
        )
        # membership is evidence about the card, not a list of its losses: no affected txns
        return [Finding(detector=self.name, pattern=self.pattern, claim=claim, lr=LR_BURST_HUB,
                        affected_txn_ids=[], connected_card_ids=others)]
