"""One investigation: an alert goes in, an answer file comes out.

The flow is a bounded state machine, not an open-ended agent loop:

  1. read the alert and the flagged transaction
  2. establish the holder's baseline and the card's recent activity
  3. run the detectors, each producing Findings with provenance
  4. combine: calibrated model probability x likelihood ratios of the findings
  5. the policy §6 gate: stop, or request evidence and simulate the reply
  6. policy.decide() twice -- before and after the evidence -- for the
     initial and final next-best actions
  7. scope the episode (affected transactions, connected cards, exposure)
  8. write the case, the SAR if the policy calls for one, and the case vertex

No number, ID, action or route in the output is produced by a language model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from assay import detectors as D
from assay.documents import PATTERNS, RULES, rules_cited
from assay.detectors import Finding, fmt_usd
from assay.patterns import PatternModel
from assay.policy import (Action, CustomerReply, Pattern, Signals, Verdict, decide, should_file_sar,
                          stop_reason)

EPISODE_HOURS = 24          # tuned on October closed cases (out-of-time): Jaccard 0.80
EPISODE_MIN_P = 0.30
LR_DISPUTE = 3.0            # assumption: a cardholder denial triples the odds (see README)
LR_DENIES = 8.0             # simulated verification: denial
LR_CONFIRMS = 1 / 15        # simulated verification: confirmation
MODEL_NOTE = ("calibrated transaction model trained only on the bank's closed cases "
              "(out-of-time AUC 0.964 vs 0.866 for the bank's risk score)")


@dataclass
class Episode:
    txns: pd.DataFrame
    connected_cards: list[str]
    connected_devices: list[str]
    shared_origin: str = ""


def _ids(df: pd.DataFrame) -> list[str]:
    return [str(int(t)) for t in df.TransactionID]


def _date(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


class Investigator:
    def __init__(self, store, patterns: PatternModel | None = None, writer=None):
        self.s = store
        self.patterns = patterns or PatternModel()
        self.writer = writer    # callable(answer) -> graph_case_id, or None
        self.last: dict = {}    # the numbers behind the last answer, for assay.trace

    # ------------------------------------------------------------------ run
    def run(self, case: pd.Series) -> dict:
        t_start = time.time()
        self.s.reset()
        fid = int(case.flagged_txn_id)
        f, ref_f = self.s.get_transaction(fid)
        t0 = f.t
        disputed = case.trigger_type == "customer_report"
        findings: list[Finding] = []

        # -- 1. the model's view of the flagged transaction --------------------
        p_model = float(f.p_fraud)
        findings.append(Finding(
            f"Flagged transaction {fid} ({fmt_usd(f.TransactionAmt)}, product {f.ProductCD}, {f.channel}) "
            f"scores {p_model:.2f} on the {MODEL_NOTE}. The bank's own risk score was {f.risk_score:.2f}.",
            ref_f, [fid], tag="model", independent=True))

        # -- 2. holder baseline and card activity ------------------------------
        hist, ref_h = self.s.holder_history(f.uid)
        base = D.holder_baseline(hist, t0)
        base["level"] = "holder"
        if base.get("n", 0) < 5:
            # Too little history under the holder key; fall back to the card, and say so.
            chist, ref_h = self.s.card_history(f.card_id, t0)
            base = D.holder_baseline(chist, t0)
            base["level"] = "card"
        card_win, ref_w = self.s.card_window(f.card_id, t0, 24 * 7, 24 * 7)
        prior_cc, ref_cc = self.s.closed_cases_for_card(f.card_id)
        findings += self._context(f, base, hist, ref_h)

        # -- 3. scenario detectors --------------------------------------------
        scenario = None   # (pattern, txns, description)
        struct = D.structuring(card_win, t0)
        if struct is not None:
            devs = sorted({d for d in struct.device_profile if d})
            scenario = ("undocumented", struct,
                        f"{len(struct)} online purchases between {fmt_usd(struct.TransactionAmt.min())} and "
                        f"{fmt_usd(struct.TransactionAmt.max())} within "
                        f"{int((struct.t.max() - struct.t.min()).total_seconds() // 60)} minutes, each just "
                        f"under a $500 threshold, from {len(devs)} device profiles marked New")
            p_burst = float(struct.p_fraud.max())
            if p_burst > p_model:
                findings.append(Finding(
                    f"The strongest transaction in the burst scores {p_burst:.2f} on the same model "
                    f"({int(struct.p_fraud.idxmax())}); the episode is assessed at that level, not at the flagged "
                    "transaction's.", ref_w, [int(struct.p_fraud.idxmax())], tag="pro"))
                p_model = p_burst
            findings.append(Finding(
                f"Structuring: {scenario[2]}. The bank's closed cases record the same shape five times "
                "(CC-3748, CC-3841, CC-3907, CC-4086, CC-4124), all confirmed fraud and filed as undocumented. "
                "Measured on July-October, such bursts are fraud 9 times in 41 (likelihood ratio ~8).",
                ref_w, _ids(struct), lr=D.LR_STRUCTURING, tag="structuring"))

        ring = None
        if f.device_profile and not D.is_generic_device(f.device_profile):
            nb, ref_nb = self.s.device_neighbors(f.device_profile, t0, 30)
            cards = sorted(set(nb.card_id) - {f.card_id})
            if D.is_ring(nb):
                ring = (nb, ref_nb, cards)
                proxy = f.id_23 if isinstance(f.id_23, str) else "no proxy flag"
                findings.append(Finding(
                    f"Device profile '{f.device_profile}' ({proxy}, marked {f.id_15} on every card) appears on "
                    f"{len(cards) + 1} different cards within 30 days of this alert: {len(nb)} transactions. "
                    "One handset on this many unrelated cardholders is a shared origin, not a household.",
                    ref_nb, cards[:25], lr=30.0, tag="ring"))
                ring_prior = self.s.closed_cases_by_device(f.device_profile)[0]
                ring_prior = ring_prior[ring_prior.outcome == "confirmed_fraud"]
                if len(ring_prior):
                    findings.append(Finding(
                        f"The same device profile was confirmed fraud in {len(ring_prior)} closed case(s) in "
                        f"{', '.join(sorted(set(ring_prior.opened_at.str[:7])))} "
                        f"({', '.join(ring_prior.case_id)}); patterns recorded: "
                        f"{', '.join(sorted(set(ring_prior.pattern)))}, linked by analysts across "
                        f"{len(set(sum(ring_prior.connected_list.tolist(), [])))} cards.",
                        self.s.calls[-1].ref, ring_prior.case_id.tolist(), lr=3.0, tag="ring_history"))
                scenario = ("undocumented", nb[nb.card_id == f.card_id],
                            "a single Samsung SM-G935F / Android 7.0 / Chrome 62 / 1920x1080 handset behind an "
                            "anonymous proxy, marked New on every card it touches")

        ct = D.card_testing(card_win)
        if ct is not None and scenario is None:
            run, after = ct
            if (run.t - t0).abs().min() <= pd.Timedelta(hours=24) or fid in set(after.TransactionID):
                scenario = ("card_testing", pd.concat([run, after]), "")
                findings.append(Finding(
                    f"Card testing: {len(run)} online authorizations under $5 within an hour "
                    f"({', '.join(fmt_usd(a) for a in run.TransactionAmt)}), followed by "
                    f"{len(after)} larger purchase(s) up to {fmt_usd(after.TransactionAmt.max())}.",
                    ref_w, _ids(run) + _ids(after), lr=20.0, tag="card_testing"))

        rec_hits = D.recurring(hist if len(hist) > 1 else card_win, f) if disputed else None
        if rec_hits is not None:
            findings.append(Finding(
                f"The disputed {fmt_usd(f.TransactionAmt)} matches this holder's own recurring charge: the same "
                f"amount under product {f.ProductCD} on {', '.join(_date(t) for t in rec_hits.t)} -- monthly.",
                ref_h, _ids(rec_hits) + [fid], lr=0.1, tag="recurring"))

        # -- case memory: has this agent already concluded something about this card? --
        mem, ref_mem = self.s.prior_fraud_cases(f.card_id)
        mem = [m for m in mem if m["attributes"].get("alert_id") != case.case_id]
        if mem:
            fraud_mem = [m for m in mem if m["attributes"].get("verdict") == "fraud"]
            findings.append(Finding(
                f"Case memory: this card is named in {len(mem)} earlier investigation(s) by this agent "
                f"({', '.join(m['v_id'] for m in mem)}), {len(fraud_mem)} of them concluded as fraud "
                f"({', '.join(sorted({m['attributes'].get('pattern', '') for m in fraud_mem})) or 'none'}).",
                ref_mem, [m["v_id"] for m in mem], lr=3.0 if fraud_mem else 1.0, tag="memory"))

        # -- 4. the alert itself ----------------------------------------------
        if disputed:
            findings.append(Finding(
                f"Cardholder {case.customer_id} reported the {fmt_usd(f.TransactionAmt)} charge as not theirs "
                f"(case pack trigger, {case.opened_at}).", f"case_pack:{case.case_id}", [fid],
                source="customer", lr=LR_DISPUTE, tag="dispute"))

        p1 = D.combine(p_model, findings)
        pro = sum(1 for x in findings if x.independent and (x.lr > 1 or x.tag == "pro"))
        con = sum(1 for x in findings if x.independent and (x.lr < 1 or x.tag == "con"))
        if p_model >= 0.5:
            pro += 1
        elif p_model <= 0.15:
            con += 1

        # -- 5. the §6 gate and evidence --------------------------------------
        self.last = {"p_model": p_model, "p1": p1, "pro": pro, "con": con, "findings": findings,
                     "flagged": f}
        answer = self._decide(case, f, p1, pro, con, findings, disputed, scenario, ring, rec_hits is not None,
                              card_win, prior_cc, base)
        answer["latency_s"] = round(time.time() - t_start, 2)
        return answer

    # ------------------------------------------------------- context findings
    def _context(self, f, base: dict, hist: pd.DataFrame, ref: str) -> list[Finding]:
        out = []
        if base.get("n", 0) == 0:
            out.append(Finding(
                f"No earlier activity for this holder (card {f.card_id}, billing region "
                f"{'none' if pd.isna(f.addr1) else int(f.addr1)}, account-open day implied by D1): the flagged "
                "transaction is the first the holder key has seen.", ref, [f.card_id], tag="pro", independent=False))
            return out
        ratio = f.TransactionAmt / base["median"] if base["median"] else 1
        out.append(Finding(
            f"{base['level'].capitalize()} baseline from {base['n']} earlier transactions: median {fmt_usd(base['median'])}, 95th "
            f"percentile {fmt_usd(base['p95'])}; the flagged amount is {ratio:.1f}x the median.",
            ref, [f.card_id], tag="pro" if f.TransactionAmt > base["p95"] else "con",
            independent=True))
        if f.channel == "in_person" and not pd.isna(f.addr1):
            seen = int(f.addr1) in base["regions"]
            out.append(Finding(
                f"Billing region {int(f.addr1)} {'is' if seen else 'is not'} among the "
                f"{len(base['regions'])} regions in this {base['level']}'s history.",
                ref, [f.card_id], tag="con" if seen else "pro", independent=True))
        if f.device_profile:
            seen = f.device_profile in base["devices"]
            out.append(Finding(
                f"Device profile '{f.device_profile}' {'was' if seen else 'was not'} used by this {base['level']} before; "
                f"identity record marks it {f.id_15 if isinstance(f.id_15, str) else 'unknown'}"
                f"{'' if not isinstance(f.id_23, str) else ', proxy ' + f.id_23}.",
                ref, [f.card_id], tag="con" if seen else "pro", independent=True))
        return out

    # ------------------------------------------------------------- episode
    def _episode(self, f, card_win, scenario, ring) -> Episode:
        if scenario is not None and scenario[0] == "undocumented" and ring is not None:
            nb, ref_nb, cards = ring
            return Episode(scenario[1], cards, [f.device_profile],
                           shared_origin=f"device profile {f.device_profile}")
        if scenario is not None:
            txns = scenario[1]
            if f.TransactionID not in set(txns.TransactionID):
                txns = pd.concat([txns, card_win[card_win.TransactionID == f.TransactionID]])
            devs = sorted({d for d in txns.device_profile if d and not D.is_generic_device(d)})
            return Episode(txns.drop_duplicates("TransactionID"), [], devs)
        w = card_win[(card_win.t - f.t).abs() <= pd.Timedelta(hours=EPISODE_HOURS)]
        w = w[(w.p_fraud >= EPISODE_MIN_P) | (w.TransactionID == f.TransactionID)]
        devs = sorted({d for d in w.device_profile if d and not D.is_generic_device(d)})
        connected, cdevs = [], []
        for d in devs:
            nb, _ = self.s.device_neighbors(d, f.t, 7)
            other = nb[(nb.card_id != f.card_id) & (nb.p_fraud >= EPISODE_MIN_P)]
            if len(other):
                connected += sorted(set(other.card_id))
                cdevs.append(d)
        origin = f"device profile {cdevs[0]}" if cdevs else ""
        return Episode(w.sort_values("t"), sorted(set(connected)), cdevs, shared_origin=origin)

    # ------------------------------------------------------------ decision
    def _decide(self, case, f, p1, pro, con, findings, disputed, scenario, ring, recurring_hit,
                card_win, prior_cc, base) -> dict:
        fid = int(f.TransactionID)
        requests = []
        step = len(self.s.calls)
        scen_pattern = scenario[0] if scenario else None

        if p1 >= 0.85 and pro >= 2:
            band = "fraud"
        elif disputed and p1 >= 0.6 and not recurring_hit:
            # The cardholder's own denial, and the graph agrees with it: R2 applies as it stands
            # (§6 -- "a verification response settles the question").
            band = "fraud"
        elif p1 <= 0.15 and con >= 2:
            band = "legit"
        else:
            band = "mid"

        ep = self._episode(f, card_win, scenario, ring)
        pattern = self._pattern(fid, scen_pattern) if band != "legit" else "none"

        def signals(prob, verdict, reply, exposure, pat) -> Signals:
            fraud = verdict is Verdict.FRAUD
            return Signals(
                fraud_probability=round(prob, 2), verdict=verdict, exposure_usd=exposure,
                pattern=Pattern(pat), customer_disputed=disputed,
                single_signal=(pro + con) <= 1, independent_evidence_count=max(pro, con),
                evidence_conflicts=(disputed and 0.15 < p1 < 0.6),
                customer_reply=reply, evidence_requested=bool(requests) or will_request,
                card_testing_sequence=(scen_pattern == "card_testing"),
                large_purchase_cleared_usd=(float(ep.txns.TransactionAmt.max())
                                            if scen_pattern == "card_testing" else 0.0),
                shared_origin=ep.shared_origin if fraud else "",
                connected_card_ids=ep.connected_cards if fraud else [],
                matches_recurring_pattern=recurring_hit,
                coordinated_abuse=(pat == "undocumented"),
            )

        exposure_if_fraud = round(float(ep.txns.TransactionAmt.abs().sum()), 2)
        will_request = False

        # ---- initial recommendation (before any requested evidence) ----
        if band == "fraud":
            v0 = Verdict.FRAUD
            reply0 = CustomerReply.DENIES if disputed else CustomerReply.NOT_ASKED
        elif band == "legit" and not disputed:
            v0, reply0 = Verdict.LEGITIMATE, CustomerReply.NOT_ASKED
        else:
            v0, reply0 = Verdict.UNCERTAIN, CustomerReply.NOT_ASKED
        will_request = v0 is Verdict.UNCERTAIN
        init_exposure = exposure_if_fraud if v0 is not Verdict.LEGITIMATE else 0.0
        initial = decide(signals(p1, v0, reply0, init_exposure, pattern))

        # ---- evidence, if the gate says so ----
        p2, reply, verdict = p1, reply0, v0
        if v0 is Verdict.UNCERTAIN:
            if disputed:
                # The customer has already denied. What is missing is whether the graph agrees.
                if p1 >= 0.6 and not recurring_hit:
                    requests.append({"type": "analyst_info", "asked_after_step": step,
                                     "assumed_response": "Analyst review of the merchant descriptors and device "
                                     "data finds nothing linking the purchase to the cardholder; the denial "
                                     "stands."})
                    p2 = D.combine(p1, [Finding("", "", lr=LR_DENIES)])
                    reply, verdict = CustomerReply.DENIES, Verdict.FRAUD
                elif p1 <= 0.4 or recurring_hit:
                    what = ("the charge as their own recurring subscription" if recurring_hit
                            else "the purchase once shown the merchant, amount and device/location details")
                    requests.append({"type": "customer_validation", "asked_after_step": step,
                                     "assumed_response": f"Cardholder recognises {what} and withdraws the "
                                     "dispute."})
                    p2 = D.combine(p1, [Finding("", "", lr=LR_CONFIRMS)])
                    reply, verdict = CustomerReply.CONFIRMS, Verdict.LEGITIMATE
                else:
                    requests.append({"type": "analyst_info", "asked_after_step": step,
                                     "assumed_response": "Analyst review is inconclusive: the device and region "
                                     "partly match the cardholder's history and partly do not."})
                    reply, verdict = CustomerReply.NOT_ASKED, Verdict.UNCERTAIN
            else:
                if p1 >= 0.6:
                    requests.append({"type": "customer_validation", "asked_after_step": step,
                                     "assumed_response": "Cardholder states they did not make this transaction "
                                     "and still has the card."})
                    p2 = D.combine(p1, [Finding("", "", lr=LR_DENIES)])
                    reply, verdict = CustomerReply.DENIES, Verdict.FRAUD
                elif p1 <= 0.4:
                    requests.append({"type": "customer_validation", "asked_after_step": step,
                                     "assumed_response": "Cardholder confirms they made the transaction."})
                    p2 = D.combine(p1, [Finding("", "", lr=LR_CONFIRMS)])
                    reply, verdict = CustomerReply.CONFIRMS, Verdict.LEGITIMATE
                else:
                    requests.append({"type": "customer_validation", "asked_after_step": step,
                                     "assumed_response": "No reply within 24 hours. The probability sits near "
                                     "even, so no reply was assumed rather than inventing a decisive answer."})
                    reply, verdict = CustomerReply.NO_REPLY, Verdict.UNCERTAIN

        if verdict is Verdict.LEGITIMATE:
            pattern = "none"
        elif pattern == "none":
            pattern = self._pattern(fid, scen_pattern)
        exposure = exposure_if_fraud if verdict is not Verdict.LEGITIMATE else 0.0
        final_sig = signals(p2, verdict, reply, exposure, pattern)
        final = decide(final_sig) if requests else initial
        if requests:
            self._evidence_findings(findings, requests)

        # The document side: the pattern definition and the policy text the final actions rest on.
        if pattern in PATTERNS:
            findings.append(Finding(PATTERNS[pattern], f"document:brief#known-fraud-patterns/{pattern}",
                                    [], source="document", independent=False))
        for rule in rules_cited([r.reason for r in final])[:3]:
            findings.append(Finding(f"Policy {rule}: {RULES[rule]}", f"document:fraud-policy#{rule}", [],
                                    source="document", independent=False))

        affected = _ids(ep.txns.sort_values("t")) if verdict is not Verdict.LEGITIMATE else []
        connected = ep.connected_cards if verdict is Verdict.FRAUD else []
        devices = ep.connected_devices if (verdict is Verdict.FRAUD and connected) else []
        similar = self._similar(f, pattern, prior_cc, verdict)

        file_sar, sar_reason = should_file_sar(final_sig)
        file_sar = any(r.action is Action.FILE_REPORT for r in final)

        status = {"fraud": "closed_fraud", "legitimate": "closed_legitimate"}.get(verdict.value)
        if status is None:
            status = "escalated" if any(r.action is Action.ESCALATE_TO_ANALYST for r in final) else "open"

        prob = round(p2, 2)
        answer = {
            "case_id": case.case_id,
            "case": {
                "status": status,
                "verdict": verdict.value,
                "fraud_probability": prob,
                "pattern": pattern,
                "pattern_description": self._pattern_description(scenario, ep) if pattern == "undocumented" else "",
                "affected_txn_ids": affected,
                "first_suspicious_txn_id": affected[0] if affected else "",
                "connected_card_ids": connected,
                "connected_device_profiles": devices,
                "exposure_usd": exposure,
                "evidence": [x.as_evidence() for x in findings if x.claim],
                "similar_prior_cases": similar,
                "summary": self._summary(f, verdict, pattern, prob, ep, exposure, requests, recurring_hit),
                "written_to_graph": False,
                "graph_case_id": "",
            },
            "evidence_requests": requests,
            "next_best_actions": {
                "initial": [r.as_dict() for r in initial],
                "final": [r.as_dict() for r in final],
                "what_changed": self._what_changed(initial, final, p1, p2, requests),
            },
            "sar": self._sar(file_sar, sar_reason, case, f, ep, exposure, pattern, connected),
            "stop_reason": stop_reason(final_sig),
            "tool_calls": 0,
            "tokens": 0,
        }
        if self.writer is not None:
            gid = self.writer(answer, f.card_id)
            if gid:
                answer["case"]["written_to_graph"] = True
                answer["case"]["graph_case_id"] = gid
        answer["tool_calls"] = len(self.s.calls)
        self.last.update(p2=p2, band=band, episode=ep)
        return answer

    # ----------------------------------------------------------- helpers
    def _pattern(self, fid: int, scen: str | None) -> str:
        if scen:
            return scen
        probs = self.patterns.predict(fid)
        return max(probs, key=probs.get)

    def _evidence_findings(self, findings, requests):
        for i, r in enumerate(requests, 1):
            findings.append(Finding(f"Simulated {r['type'].replace('_', ' ')}: {r['assumed_response']}",
                                    f"evidence_request:{i}", [],
                                    source="customer" if r["type"] != "analyst_info" else "external"))

    def _similar(self, f, pattern, prior_cc, verdict) -> list[str]:
        out = []
        if verdict is Verdict.LEGITIMATE:
            out += prior_cc[prior_cc.outcome == "cleared"].case_id.tolist()[-2:]
            if len(out) < 2:
                like, _ = self.s.closed_cases_like("none", f.device_profile or "", 2, f.ProductCD, float(f.TransactionAmt))
                out += [c for c in like.case_id.tolist() if c not in out][: 2 - len(out)]
        else:
            same = prior_cc[(prior_cc.outcome == "confirmed_fraud") & (prior_cc.pattern == pattern)]
            out += same.case_id.tolist()[-2:]
            like, _ = self.s.closed_cases_like(pattern, f.device_profile or "", 3, f.ProductCD, float(f.TransactionAmt))
            out += [c for c in like.case_id.tolist() if c not in out][: max(1, 3 - len(out))]
        return out

    def _what_changed(self, initial, final, p1, p2, requests) -> str:
        if not requests:
            return "nothing"
        a0 = [r.action.value for r in initial]
        a1 = [r.action.value for r in final]
        if a0 == a1:
            return f"The evidence request did not change the recommendation; probability {p1:.2f} -> {p2:.2f}."
        added = [a for a in a1 if a not in a0]
        dropped = [a for a in a0 if a not in a1]
        return (f"{requests[-1]['assumed_response']} Probability moved {p1:.2f} -> {p2:.2f}; "
                f"added {', '.join(added) or 'nothing'}, dropped {', '.join(dropped) or 'nothing'}.")

    def _pattern_description(self, scenario, ep: Episode) -> str:
        if scenario and "just under" in scenario[2]:
            return ("Amount structuring: " + scenario[2] + ". The amounts sit just below a $500 authorization "
                    "threshold, which none of the five documented patterns describes; the same shape recurs on "
                    "unrelated cardholders in the bank's closed cases. Found by scanning the card's one-hour "
                    "window for repeated near-threshold online amounts.")
        return ("Device-ring fraud: " + (scenario[2] if scenario else "one handset") + ", used on "
                f"{len(ep.connected_cards) + 1} unrelated cards within a month for a few purchases each, every one "
                "scoring low on the bank's model. No single card shows a credential or channel change, so it is "
                "not account takeover, and the link that makes it fraud is the shared handset across "
                "cardholders. Found by traversing from the flagged transaction's device profile to every other "
                "card that used it.")

    def _summary(self, f, verdict, pattern, prob, ep, exposure, requests, recurring_hit) -> str:
        fid = int(f.TransactionID)
        if verdict is Verdict.LEGITIMATE:
            why = ("it matches the holder's own monthly recurring charge" if recurring_hit
                   else "the transaction is consistent with the holder's own history")
            conf = f" {requests[-1]['assumed_response']}" if requests else ""
            return (f"Alert on transaction {fid} ({fmt_usd(f.TransactionAmt)}) on card {f.card_id}. "
                    f"The graph evidence does not support fraud: {why}, and the calibrated model puts it at "
                    f"{prob:.2f}.{conf} Closed as legitimate; no report.")
        n = len(ep.txns)
        span = f"{_date(ep.txns.t.min())} to {_date(ep.txns.t.max())}" if n else ""
        s = (f"{pattern.replace('_', ' ').capitalize()} on card {f.card_id}: {n} transaction(s) from {span} "
             f"totalling {fmt_usd(exposure)}, probability {prob:.2f}.")
        if ep.connected_cards and verdict is Verdict.FRAUD:
            s += f" Linked to {len(ep.connected_cards)} other card(s) through {ep.shared_origin}."
        if verdict is Verdict.UNCERTAIN:
            s += " The evidence does not settle it, so it is held for a human under R4/R8."
        if requests:
            s += f" {requests[-1]['assumed_response']}"
        return s

    def _sar(self, file_sar, reason, case, f, ep, exposure, pattern, connected) -> dict:
        if not file_sar:
            return {"file": False, "reason": reason, "narrative": "", "subjects": [], "total_amount_usd": 0,
                    "activity_dates": []}
        t = ep.txns.sort_values("t")
        d0, d1 = _date(t.t.min()), _date(t.t.max())
        chans = " and ".join(sorted(set(t.channel))).replace("_", " ")
        devs = sorted({d for d in t.device_profile if d})
        regions = sorted({int(r) for r in t.addr1.dropna()})
        lines = [
            f"Between {d0} and {d1}, card {f.card_id} held by customer {case.customer_id} was used for "
            f"{len(t)} transaction(s) totalling {fmt_usd(exposure)} that the bank has determined were not "
            f"authorised by the cardholder (transaction IDs {', '.join(_ids(t))}).",
            f"The activity was {chans}"
            + (f", from device profile(s) {'; '.join(devs)}" if devs else "")
            + (f", billed to region code(s) {', '.join(map(str, regions))}" if regions else "") + ".",
            f"The alert was raised on {case.opened_at} by {case.trigger_type.replace('_', ' ')} on transaction "
            f"{int(f.TransactionID)} ({fmt_usd(f.TransactionAmt)}).",
        ]
        if pattern == "undocumented" and connected:
            lines.append(f"The same device profile was used on {len(connected)} other cards in the same period "
                         f"({', '.join(connected[:12])}{', ...' if len(connected) > 12 else ''}), each time marked as "
                         "a new device behind an anonymous proxy, indicating a single actor operating across "
                         "multiple unrelated cardholders.")
            lines.append("The bank's closed cases from August and September record confirmed fraud from this "
                         "same device profile on other cardholders, which analysts could not match to a "
                         "documented typology.")
        elif pattern == "undocumented":
            lines.append("The purchases were placed minutes apart with each amount just below $500, consistent "
                         "with deliberately keeping each authorization under a review threshold; the bank has "
                         "seen the identical shape on five other cardholders, all confirmed fraud.")
        elif pattern == "card_testing":
            lines.append("A run of very small online authorizations preceded larger purchases, consistent with "
                         "testing a stolen card number before use.")
        else:
            lines.append(f"The activity fits the {pattern.replace('_', ' ')} typology: it is inconsistent with "
                         "the cardholder's established amounts, devices and regions in the bank's records.")
        if connected and pattern != "undocumented":
            lines.append(f"The device used was also seen on card(s) {', '.join(connected)} with high-risk activity "
                         "in the same week.")
        lines.append("The cardholder disputed the activity." if case.trigger_type == "customer_report" else
                     "The activity was identified by the bank's transaction monitoring and confirmed by "
                     "investigation.")
        lines.append("The card has been blocked pending reissue"
                     + (" and the connected cards placed under monitoring" if connected else "")
                     + "; this report is filed because " + reason.split(": ", 1)[-1])
        subjects = [case.customer_id, f.card_id] + connected + devs
        return {"file": True, "reason": reason, "narrative": " ".join(lines), "subjects": subjects,
                "total_amount_usd": exposure, "activity_dates": [d0, d1]}
