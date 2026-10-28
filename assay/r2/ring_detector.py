"""Ring membership (assay.r2.rings) as an ordinary detector, so it lands in the ledger.

The graph is cards joined by shared non-generic elements in the 30 days before the
case opened; the ring is the flagged card's weakly connected component, kept only if
3..30 cards and at least one member already carries confirmed fraud. Its LR is the
measured backtest (tests/r2/test_rings.py, run_ring_ablation(algo="wcc")): 4,675
confirmed-fraud openings vs 5,000 legit Jul-Oct card-days -> x8.52.
"""

from __future__ import annotations

import pandas as pd

from assay.r2.finding import CaseCtx, Detector, Finding
from assay.r2.rings import ring_for

LR_RING_WCC = 8.52


class RingDetector(Detector):
    name = "ring_wcc"
    pattern = "undocumented:shared_element_ring"

    def run(self, ctx: CaseCtx) -> list[Finding]:
        # Card ids are issuer buckets; a bucket busier than 99% of cards shares an element with
        # nearly everything, so its component says nothing about this cardholder.
        recent = ctx.tx[ctx.tx.t > ctx.opened_at - pd.Timedelta(days=30)].groupby("card_id").size()
        if recent.get(ctx.card_id, 0) > recent.quantile(0.99):
            return []
        ring = ring_for(ctx.card_id, ctx.opened_at, tx=ctx.tx, cc=ctx.cc, algo="wcc")
        if ring is None:
            return []
        others = [c for c in ring.cards if c != ctx.card_id]
        # membership is evidence about the card, not a list of its losses: no affected txns
        return [Finding(detector=self.name, pattern=self.pattern, claim=ring.attestation, lr=LR_RING_WCC,
                        affected_txn_ids=[], connected_card_ids=others)]
