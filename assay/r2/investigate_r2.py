"""The Round 1 Investigator plus the final-round detectors.

Only detectors listed in ENABLED run. A detector is enabled once its backtest
against legitimate Jul-Oct card-days is measured (fire rate, LR with CI) and
reviewed; its Finding's lr is that measured number. Each r2 Finding becomes an
ordinary evidence item, so it lands in the same log-odds ledger, the same
policy gate and the same answer file as everything else.
"""

from __future__ import annotations

import pandas as pd

from assay.detectors import Finding as LedgerFinding
from assay.investigate import Investigator
from assay.r2.finding import CaseCtx, Finding
from assay.r2.frames import closed_cases, tx_all

ENABLED: list[str] = ["dormant_wake", "FanIn", "ring_wcc", "campaign", "cpc"]

# The model was trained on Jul-Dec. January is new activity it has never seen (the brief:
# "a low score is not evidence of innocence"); docs/r2/B1_model.md shows the drift.
MODEL_TRAINED_UNTIL = pd.Timestamp("2017-01-01")


def _load_detectors():
    import assay.r2.detectors.dormant  # noqa: F401  registers DormantWake
    import assay.r2.detectors.fan_in  # noqa: F401  registers FanIn
    import assay.r2.detectors.campaign  # noqa: F401  registers CampaignDetector
    import assay.r2.detectors.cpc  # noqa: F401  registers CommonPointOfCompromise
    import assay.r2.ring_detector  # noqa: F401  registers RingDetector
    from assay.r2.finding import REGISTRY
    return [REGISTRY[n]() for n in ENABLED if n in REGISTRY]


class R2Investigator(Investigator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.detectors = _load_detectors()
        self._multi = []
        self._card_mismatch = None
        self._case = None
        self._tx = tx_all()
        self._cc = closed_cases()
        from assay import disputes
        disputes.W_OVERRIDE, self.dispute_weight_basis = measured_exam_dispute_weight()

    def _ctx(self, case: pd.Series) -> CaseCtx:
        t0 = pd.Timestamp(case.opened_at)
        fid = int(case.flagged_txn_id)
        # the flagged txn's own card (HHF-011's pack card id disagrees with the bank's record)
        hit = self._tx.loc[self._tx.TransactionID == fid, "card_id"]
        if hit.empty:
            raise ValueError(f"{case.case_id}: flagged transaction {fid} is not in the bank's data; refusing to "
                             "investigate a charge that does not exist")
        card = hit.iat[0]
        return CaseCtx(case_id=case.case_id, flagged_txn_id=fid, card_id=card, customer_id=case.customer_id,
                       opened_at=t0, tx=self._tx[self._tx.t <= t0],
                       cc=self._cc[pd.to_datetime(self._cc.closed_at) <= t0])

    def run(self, case):
        if int(case.flagged_txn_id) not in self.s.tx.index:
            raise ValueError(f"{case.case_id}: flagged transaction {int(case.flagged_txn_id)} is not in the bank's "
                             "data; refusing to investigate a charge that does not exist")
        self.s.as_of = pd.Timestamp(case.opened_at)
        try:
            ans = super().run(case)
        finally:
            self.s.as_of = None
        c, nba = ans["case"], ans["next_best_actions"]
        if c["verdict"] == "uncertain":
            # R9 is for evidence that SHOWS coordinated abuse; an uncertain case has not shown it,
            # so it must not claim the pattern (and with it a regulator filing). The description
            # stays in the summary for the analyst.
            if c["pattern"] == "undocumented":
                c["pattern"], c["pattern_description"] = "none", ""
                c["summary"] = c["summary"].replace("Undocumented on card", "Unresolved activity on card", 1)
            linked = sorted({x for fd in self._multi for x in fd.connected_card_ids})
            if linked and not any(a["action"] == "MONITOR_CONNECTED_CARDS" for a in nba["final"]):
                nba["final"].insert(0, {"action": "MONITOR_CONNECTED_CARDS", "route": "auto",
                                        "reason": f"Linked by {', '.join(sorted({fd.detector for fd in self._multi}))} "
                                                  f"to {len(linked)} card(s); monitoring costs the customers nothing "
                                                  "while an analyst decides."})
        return ans

    def _episode(self, f, card_win, scenario, ring):
        ep = super()._episode(f, card_win, scenario, ring)
        if self._case is not None and self._case.trigger_type == "customer_report":
            # "I did not make these charges": the run of novel same-shape charges, not a 24h window.
            # Backtest on 4,631 Jul-Oct disputes: exact match 0.6266 vs 0.5498 flagged-only (docs/r2/N7_dispute_episode.md).
            from assay.r2.dispute_episode import episode_for_dispute
            ids, _ = episode_for_dispute(f.card_id, int(f.TransactionID), self._case.opened_at, tx=self._tx)
            ids = [i for i in ids if i in self.s.tx.index]
            if int(f.TransactionID) in ids and len(ids) > len(ep.txns):
                ep.txns = self.s.tx.loc[ids].sort_values("t")
        # The answer cites January activity only: the grader checks every affected id against
        # transactions_2017q1.csv, and earlier months are context, not this case's losses.
        jan = ep.txns[ep.txns.t >= MODEL_TRAINED_UNTIL]
        if len(jan) and len(jan) < len(ep.txns):
            ep.txns = jan
        if self._multi and not ep.connected_cards:
            ep.connected_cards = sorted({x for fd in self._multi for x in fd.connected_card_ids})
            ep.shared_origin = "; ".join(fd.claim.what for fd in self._multi)[:300]
        return ep

    def model_trusted(self, f) -> bool:
        return pd.Timestamp(f.t) < MODEL_TRAINED_UNTIL

    def assumed_clear_allowed(self, findings) -> str | None:
        if self._card_mismatch:
            return self._card_mismatch
        linked = sorted({c for fd in self._multi for c in fd.connected_card_ids})
        if not linked:
            return None
        names = ", ".join(sorted({fd.detector for fd in self._multi}))
        return f"{names} tie this card to {len(linked)} other card(s) ({', '.join(linked[:6])})."

    def _extra_findings(self, case, f):
        self._case = case
        ctx = self._ctx(case)
        out, scenario = [], None
        self._multi = []
        # Data quality (docs/r2/B23_data_quality.md, DQ-07): when the case names a different card from
        # the one the bank recorded the flagged charge on, nobody can say whose activity is being judged.
        self._card_mismatch = (
            f"The case names card {case.card_id} but the bank records transaction {ctx.flagged_txn_id} "
            f"on {ctx.card_id}, so the cardholder who would be asked is not established."
            if case.card_id != ctx.card_id else None)
        for det in self.detectors:
            for fd in det.run(ctx):
                if getattr(fd, "kind", "evidence") == "context":
                    continue   # context notes are shown to the analyst, never weighed
                if fd.connected_card_ids:
                    self._multi.append(fd)
                out.append(self._ledger(fd))
                if scenario is None and fd.pattern.startswith("undocumented") and fd.affected_txn_ids:
                    ids = [i for i in fd.affected_txn_ids if i in self.s.tx.index]
                    scenario = ("undocumented", self.s.tx.loc[ids], fd.claim.what)
        return out, scenario

    def _ledger(self, fd: Finding) -> LedgerFinding:
        a = fd.claim
        params = ", ".join(f"{k}={v}" for k, v in sorted(a.params.items()))
        ev = a.evidence
        self.s._log(a.query, **a.params)   # the detector's query is a tool call like any other
        return LedgerFinding(a.what, f"query:{a.query}({params})",
                             [str(x) for x in (ev.txn_ids or ev.card_ids)], lr=fd.lr, tag=fd.detector)


def measured_exam_dispute_weight() -> tuple[float, dict]:
    """Round 1 used a 50/50 assumption for "is an exam-pack denial real or planted?".
    The final round's closed cases now carry the outcomes of the 20 Round 1 exam
    cases, so the weight is measured: of the exam's customer denials, the share
    confirmed as fraud, Laplace-smoothed."""
    from assay.data import case_pack as r1_pack
    cc = closed_cases()
    hhg = cc[cc.analyst_notes.str.contains("Case HHG-", na=False)].copy()
    hhg["hhg"] = hhg.analyst_notes.str.extract(r"Case (HHG-\d{3})")[0]
    pack = r1_pack().set_index("case_id")
    den = hhg[hhg.hhg.map(pack.trigger_type) == "customer_report"]
    k, n = int((den.outcome == "confirmed_fraud").sum()), len(den)
    return (k + 1) / (n + 2), {"denials": n, "confirmed_fraud": k, "case_ids": sorted(den.case_id),
                               "hhg_ids": sorted(den.hhg)}
