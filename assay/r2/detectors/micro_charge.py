"""MicroCharge and MicroDripCohort detectors for undocumented typology:
coordinated slow-drip micro-charge campaign cohort burst.

- Keeps existing `MicroCharge` ('undocumented:slow_drip_micro_charges') working.
- Adds `MicroDripCohort` (name='micro_drip_cohort', pattern='undocumented:sub_threshold_micro_drip'):
  A card has a NEW micro-drip when, as-of t <= opened_at, it has >= 4 online ProductCD=='C'
  charges each < $6.50 spread over >= 7 days inside a 30-day window and NO such micro charges
  in the 60 days before the window, excluding pure card-testing bursts (all charges within 1h,
  inherently excluded by span >= 7d plus an explicit > 1h span check).
  Fires when the case's flagged txn belongs to its card's new micro-drip AND >= K other cards
  (K=6, meaning >= 7 total cohort cards, justified from the Aug-Nov monthly base rate of 1-6
  new-drip cards per month) started a new micro-drip whose window overlaps [opened_at-30d, opened_at].
  Connected cards = the other cohort cards (as-of), affected_txn_ids = the flagged card's drip txns,
  attestation evidence.txn_ids = every cohort card's drip txn ids.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding


def poisson_tail_p(k: int, lam: float) -> float:
    """Computes P(X >= k) for X ~ Poisson(max(lam, 0.05))."""
    rate = max(float(lam), 0.05)
    if k <= 0:
        return 1.0
    term = math.exp(-rate)
    cdf = term
    for i in range(1, k):
        term *= rate / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def _find_card_new_drips(
    micro_df: pd.DataFrame,
    t_open: pd.Timestamp,
    window_days: int = 30,
    quiet_days: int = 60,
    min_charges: int = 4,
    min_span_days: float = 7.0,
) -> dict[str, dict]:
    """Finds all cards in `micro_df` (already filtered to online, ProductCD=='C', amount < max_amt,
    and t <= t_open) that have a valid new micro-drip episode whose 30d window [t_w0, t_w_end]
    overlaps [t_open - 30d, t_open].

    A card qualifies if there is a start charge at t_w0 such that:
      1. Zero micro charges on that card in [t_w0 - 60d, t_w0).
      2. Within [t_w0, min(t_w0 + 30d, t_open)], there are >= 4 micro charges.
      3. The span of those charges is >= 7 days (and > 1 hour, excluding card-testing bursts).
      4. The episode window [t_w0, t_w_end] overlaps [t_open - 30d, t_open], i.e., t_w_end >= t_open - 30d.
    """
    if micro_df.empty:
        return {}

    win_ns = int(window_days * 86400 * 1e9)
    quiet_ns = int(quiet_days * 86400 * 1e9)
    min_span_ns = int(min_span_days * 86400 * 1e9)
    one_hour_ns = int(3600 * 1e9)
    t_open_ns = pd.Timestamp(t_open).value
    ov_start_ns = t_open_ns - win_ns

    # Fast groupby using numpy arrays on sorted dataframe
    df_s = micro_df.sort_values(["card_id", "t"], kind="mergesort")
    cards = df_s["card_id"].to_numpy()
    t_ns = df_s["t"].to_numpy(dtype="datetime64[ns]").view("int64")
    txids = df_s["TransactionID"].to_numpy(dtype="int64")

    _, idx_start, counts = np.unique(cards, return_index=True, return_counts=True)
    qual_mask = counts >= min_charges
    qual_starts = idx_start[qual_mask]
    qual_counts = counts[qual_mask]

    out: dict[str, dict] = {}
    for s, c in zip(qual_starts, qual_counts):
        c_t = t_ns[s : s + c]
        c_tx = txids[s : s + c]
        cid = str(cards[s])

        # Evaluate candidate window start indices i where no prior charge exists in [c_t[i] - 60d, c_t[i])
        best_ep = None
        for i in range(c - min_charges + 1):
            t0 = c_t[i]
            if i > 0 and (t0 - c_t[i - 1]) < quiet_ns:
                continue
            t_lim = min(t0 + win_ns, t_open_ns)
            # Find all charges in [t0, t_lim]
            j = int(np.searchsorted(c_t, t_lim, side="right"))
            n_in = j - i
            if n_in < min_charges:
                continue
            t_last = c_t[j - 1]
            span_ns = t_last - t0
            if span_ns < min_span_ns or span_ns <= one_hour_ns:
                continue
            # Window of this new micro-drip is [t0, min(t0 + 30d, t_open_ns)]; also check last charge time
            win_end_ns = min(t0 + win_ns, t_open_ns)
            if win_end_ns >= ov_start_ns and t0 <= t_open_ns:
                ep_txns = [int(x) for x in c_tx[i:j]]
                best_ep = {
                    "card_id": cid,
                    "t_start": pd.Timestamp(t0),
                    "t_end": pd.Timestamp(t_last),
                    "win_end": pd.Timestamp(win_end_ns),
                    "n_charges": n_in,
                    "span_days": float(span_ns / (86400.0 * 1e9)),
                    "txn_ids": ep_txns,
                }
        if best_ep is not None:
            out[cid] = best_ep
    return out


def cohort_members(
    tx_asof: pd.DataFrame,
    flagged_txn_id: int,
    opened_at: pd.Timestamp | str,
    max_amt: float = 6.50,
    min_charges: int = 4,
    min_span_days: float = 7.0,
    window_days: int = 30,
    quiet_days: int = 60,
    min_other_cards: int = 6,
) -> tuple[list[str], dict[str, list[int]], dict]:
    """Pure function for GSQL mirroring: computes the micro-drip cohort as-of `opened_at`.

    Returns:
      (member_card_ids, per_card_txn_ids, params_used)
    If the flagged transaction does not belong to its card's new micro-drip or if fewer
    than `min_other_cards` other cards overlap [opened_at - 30d, opened_at], returns
    still the raw cohort discovered alongside `qualified=False` in params_used, or if
    flagged_txn_id is not in a new micro-drip on its card, returns ([], {}, params_used).
    """
    t_open = pd.Timestamp(opened_at)
    params = {
        "flagged_txn_id": int(flagged_txn_id),
        "opened_at": str(t_open),
        "max_amt": float(max_amt),
        "min_charges": int(min_charges),
        "min_span_days": float(min_span_days),
        "window_days": int(window_days),
        "quiet_days": int(quiet_days),
        "min_other_cards": int(min_other_cards),
    }
    if tx_asof.empty:
        return [], {}, params

    df = tx_asof[tx_asof.t <= t_open]
    if df.empty:
        return [], {}, params

    f_row = df[df.TransactionID == int(flagged_txn_id)]
    if f_row.empty:
        return [], {}, params
    flagged_card = str(f_row.card_id.iloc[0])

    micro_df = df[
        (df.channel == "online")
        & (df.ProductCD == "C")
        & (df.amount < max_amt)
    ]
    if micro_df.empty:
        return [], {}, params

    drips = _find_card_new_drips(
        micro_df,
        t_open=t_open,
        window_days=window_days,
        quiet_days=quiet_days,
        min_charges=min_charges,
        min_span_days=min_span_days,
    )
    if flagged_card not in drips:
        return [], {}, params

    flagged_ep = drips[flagged_card]
    if int(flagged_txn_id) not in flagged_ep["txn_ids"]:
        return [], {}, params

    member_cards = sorted(drips.keys())
    per_card_txns = {c: sorted(drips[c]["txn_ids"]) for c in member_cards}
    n_other = len(member_cards) - 1
    params["n_cohort_cards"] = len(member_cards)
    params["n_other_cards"] = n_other
    params["fired"] = bool(n_other >= min_other_cards)
    if n_other < min_other_cards:
        return [], {}, params
    return member_cards, per_card_txns, params


def monthly_new_drip_distribution(
    tx: pd.DataFrame,
    max_amt: float = 6.50,
    min_charges: int = 4,
    min_span_days: float = 7.0,
    window_days: int = 30,
    quiet_days: int = 60,
) -> dict[str, dict]:
    """Computes the monthly count of cards starting a new micro-drip across Jul 2016 - Jan 2017.
    Used to justify K (min_other_cards = 6, i.e. >= 7 total cards) from the Aug-Nov monthly base rate.
    """
    micro_df = tx[
        (tx.channel == "online")
        & (tx.ProductCD == "C")
        & (tx.amount < max_amt)
    ].sort_values(["card_id", "t"], kind="mergesort")

    win_ns = int(window_days * 86400 * 1e9)
    quiet_ns = int(quiet_days * 86400 * 1e9)
    min_span_ns = int(min_span_days * 86400 * 1e9)
    one_hour_ns = int(3600 * 1e9)

    cards = micro_df["card_id"].to_numpy()
    t_ns = micro_df["t"].to_numpy(dtype="datetime64[ns]").view("int64")
    _, idx_start, counts = np.unique(cards, return_index=True, return_counts=True)

    qual_starts = idx_start[counts >= min_charges]
    qual_counts = counts[counts >= min_charges]

    episodes = []
    for s, c in zip(qual_starts, qual_counts):
        c_t = t_ns[s : s + c]
        cid = str(cards[s])
        i = 0
        while i <= c - min_charges:
            t0 = c_t[i]
            if i > 0 and (t0 - c_t[i - 1]) < quiet_ns:
                i += 1
                continue
            t_lim = t0 + win_ns
            j = int(np.searchsorted(c_t, t_lim, side="right"))
            if (j - i) >= min_charges:
                span_ns = c_t[j - 1] - t0
                if span_ns >= min_span_ns and span_ns > one_hour_ns:
                    t_qual = pd.Timestamp(c_t[i + min_charges - 1])
                    # Determine the first charge >= i+min_charges-1 where span >= 7d
                    for k_idx in range(i + min_charges - 1, j):
                        if (c_t[k_idx] - t0) >= min_span_ns:
                            t_qual = pd.Timestamp(c_t[k_idx])
                            break
                    episodes.append(
                        {
                            "card_id": cid,
                            "t_start": pd.Timestamp(t0),
                            "t_qual": t_qual,
                            "start_month": pd.Timestamp(t0).strftime("%Y-%m"),
                            "qual_month": t_qual.strftime("%Y-%m"),
                        }
                    )
                    i = j
                    continue
            i += 1

    ep_df = pd.DataFrame(episodes)
    months = ["2016-07", "2016-08", "2016-09", "2016-10", "2016-11", "2016-12", "2017-01"]
    dist: dict[str, dict] = {}
    for m in months:
        if ep_df.empty:
            dist[m] = {"by_start_month": 0, "by_qual_month": 0, "cards_by_qual": []}
        else:
            s_cnt = int((ep_df.start_month == m).sum())
            q_sub = ep_df[ep_df.qual_month == m]
            q_cnt = int(len(q_sub))
            dist[m] = {
                "by_start_month": s_cnt,
                "by_qual_month": q_cnt,
                "cards_by_qual": sorted(q_sub.card_id.unique().tolist()),
            }
    return dist


def rerun_micro_drip_cohort(
    tx: pd.DataFrame,
    card_id: str,
    flagged_txn_id: int,
    opened_at: str,
    max_amt: float = 6.50,
    min_charges: int = 4,
    min_span_days: float = 7.0,
    window_days: int = 30,
    quiet_days: int = 60,
    min_other_cards: int = 6,
) -> dict:
    """Registered re-runnable query function for MicroDripCohort attestations."""
    t_open = pd.Timestamp(opened_at)
    tx_asof = tx[tx.t <= t_open]
    members, per_card_txns, params = cohort_members(
        tx_asof,
        flagged_txn_id=int(flagged_txn_id),
        opened_at=t_open,
        max_amt=max_amt,
        min_charges=min_charges,
        min_span_days=min_span_days,
        window_days=window_days,
        quiet_days=quiet_days,
        min_other_cards=min_other_cards,
    )
    all_txns = sorted({tid for tids in per_card_txns.values() for tid in tids})
    return {
        "fired": bool(len(members) >= min_other_cards + 1),
        "card_id": str(card_id),
        "member_card_ids": members,
        "connected_card_ids": [c for c in members if c != str(card_id)],
        "affected_txn_ids": per_card_txns.get(str(card_id), []),
        "cohort_txn_ids": all_txns,
        "params": params,
    }


class MicroDripCohort(Detector):
    """(a) MicroDripCohort in assay/r2/detectors/micro_charge.py:
    name = 'micro_drip_cohort', pattern = 'undocumented:sub_threshold_micro_drip'.

    K = 6 other cards (>= 7 total cards in the overlapping 30-day cohort window), chosen
    from the Aug-Nov monthly base rate of 1-6 new-drip cards per month (0 historical 30d
    windows in Aug-Nov exceed 6 cards; Dec rises to 11 as the campaign begins, Jan reaches 19).

    Historical backtest over Jul-Dec closed cases and 5,000 legitimate card-days (cards never
    named in a case, sampled with seed=42):
    - Cohort burst (K>=6 other cards) fires 0 times on Jul-Dec closed fraud cases and 0 times
      on legitimate card-days (Laplace LR = 1.07, 95% CI [0.07, 17.12]).
    - Per LR_RULE, because the cohort form never fired on Jul-Dec closed fraud cases, `lr` is
      set to `1.0` (never an invented ratio), carrying `connected_card_ids`, `affected_txn_ids`,
      and the full cohort attestation for Policy §3a / R9. Any non-measured weight is placed only
      in `lr_proposal` with its stated assumption.
    """

    name = "micro_drip_cohort"
    pattern = "undocumented:sub_threshold_micro_drip"
    enabled: bool = True

    def __init__(
        self,
        max_amt: float = 6.50,
        min_charges: int = 4,
        min_span_days: float = 7.0,
        window_days: int = 30,
        quiet_days: int = 60,
        min_other_cards: int = 6,
        enabled: bool = True,
    ):
        self.max_amt = max_amt
        self.min_charges = min_charges
        self.min_span_days = min_span_days
        self.window_days = window_days
        self.quiet_days = quiet_days
        self.min_other_cards = min_other_cards
        self.enabled = enabled
        # Measured LR from backtest; when 0 historical fraud fires, lr=1.0 per LR_RULE
        self.lr: float = 1.0
        self.lr_ci: tuple[float, float] = (0.07, 17.12)
        self.lr_laplace: float = 1.0
        self.lr_proposal: dict = {
            "proposed_lr": None,
            "assumption": (
                "No unmeasured LR is applied. Cohort burst (K>=6 other cards) fired 0 times on "
                "Jul-Dec closed fraud cases and 0 times on 5,000 never-cased legitimate card-days; "
                "leaving lr=1.0 and attaching connected_card_ids + attested cohort txn_ids for Policy §3a."
            ),
        }
        self.backtest_stats: dict = {}

    def run(self, ctx: CaseCtx) -> list[Finding]:
        if not self.enabled or ctx.tx.empty:
            return []

        t_open = pd.Timestamp(ctx.opened_at)
        members, per_card_txns, params = cohort_members(
            ctx.tx,
            flagged_txn_id=int(ctx.flagged_txn_id),
            opened_at=t_open,
            max_amt=self.max_amt,
            min_charges=self.min_charges,
            min_span_days=self.min_span_days,
            window_days=self.window_days,
            quiet_days=self.quiet_days,
            min_other_cards=self.min_other_cards,
        )
        if not members:
            return []

        other_cards = [c for c in members if c != str(ctx.card_id)]
        if len(other_cards) < self.min_other_cards:
            return []

        own_txns = per_card_txns.get(str(ctx.card_id), [])
        all_cohort_txns = sorted({tid for tids in per_card_txns.values() for tid in tids})
        cust_ids = sorted({c.rsplit("-K", 1)[0] for c in members})

        claim = Attestation(
            what=(
                f"Coordinated sub-threshold micro-drip cohort on card {ctx.card_id}: "
                f"{len(own_txns)} online ProductCD=='C' charges < ${self.max_amt:.2f} over >= {self.min_span_days:.0f}d "
                f"with 0 prior in {self.quiet_days}d; {len(other_cards)} other card(s) ({len(members)} total, "
                f"threshold K>={self.min_other_cards} from Aug-Nov base rate 1-6/mo) have overlapping 30d new micro-drips "
                f"as of {t_open} (total cohort txns={len(all_cohort_txns)}, measured lr={self.lr:.2f})."
            ),
            evidence=Evidence(
                txn_ids=all_cohort_txns,
                card_ids=members,
                customer_ids=cust_ids,
                case_ids=[ctx.case_id],
            ),
            query="rerun_micro_drip_cohort",
            params={
                "card_id": str(ctx.card_id),
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "opened_at": str(t_open),
                "max_amt": self.max_amt,
                "min_charges": self.min_charges,
                "min_span_days": self.min_span_days,
                "window_days": self.window_days,
                "quiet_days": self.quiet_days,
                "min_other_cards": self.min_other_cards,
            },
        )

        fd = Finding(
            detector=self.name,
            pattern=self.pattern,
            claim=claim,
            lr=self.lr,
            affected_txn_ids=own_txns,
            connected_card_ids=other_cards,
        )
        fd.lr_proposal = self.lr_proposal  # type: ignore[attr-defined]
        fd.per_card_txn_ids = per_card_txns  # type: ignore[attr-defined]
        return [fd]

    def backtest(
        self, cc: pd.DataFrame, tx: pd.DataFrame, sample_size: int = 5000, seed: int = 42
    ) -> Backtest:
        """As-of backtest over Jul-Dec closed cases (confirmed_fraud and cleared) plus
        5,000 legitimate card-days sampled with a fixed seed from cards never named in a case.
        Also scans historical 30d windows across Jul-Dec to report fire counts.
        """
        tx_sorted = tx.sort_values("t", kind="mergesort") if not tx.t.is_monotonic_increasing else tx
        micro_all = tx_sorted[
            (tx_sorted.channel == "online")
            & (tx_sorted.ProductCD == "C")
            & (tx_sorted.amount < self.max_amt)
        ].sort_values(["card_id", "t"], kind="mergesort")

        win_ns = int(self.window_days * 86400 * 1e9)
        quiet_ns = int(self.quiet_days * 86400 * 1e9)
        min_span_ns = int(self.min_span_days * 86400 * 1e9)
        one_hour_ns = int(3600 * 1e9)

        # Precompute per-card micro charges for fast as-of evaluation
        cards_arr = micro_all["card_id"].to_numpy()
        t_arr = micro_all["t"].to_numpy(dtype="datetime64[ns]").view("int64")
        tx_arr = micro_all["TransactionID"].to_numpy(dtype="int64")
        _, idx_start, counts = np.unique(cards_arr, return_index=True, return_counts=True)

        qual_starts = idx_start[counts >= self.min_charges]
        qual_counts = counts[counts >= self.min_charges]

        # Precompute for each card all candidate new-drip starts (t0_ns, t_qual_ns, t_end_30d_ns, txn_times_ns, txn_ids)
        card_candidates: dict[str, list[tuple[int, int, np.ndarray, np.ndarray]]] = {}
        all_episodes: list[tuple[str, int, int, int]] = []  # (cid, t0_ns, t_qual_ns, t0_plus_30d_ns)

        for s, c in zip(qual_starts, qual_counts):
            c_t = t_arr[s : s + c]
            c_tx = tx_arr[s : s + c]
            cid = str(cards_arr[s])
            cands = []
            for i in range(c - self.min_charges + 1):
                t0 = int(c_t[i])
                if i > 0 and (t0 - int(c_t[i - 1])) < quiet_ns:
                    continue
                t_lim = t0 + win_ns
                j = int(np.searchsorted(c_t, t_lim, side="right"))
                if (j - i) < self.min_charges:
                    continue
                # Earliest index k >= i + min_charges - 1 where span >= min_span_ns and > 1h
                t_qual = -1
                for k_idx in range(i + self.min_charges - 1, j):
                    sp = int(c_t[k_idx]) - t0
                    if sp >= min_span_ns and sp > one_hour_ns:
                        t_qual = int(c_t[k_idx])
                        break
                if t_qual != -1:
                    cands.append((t0, t_qual, c_t[i:j], c_tx[i:j]))
                    all_episodes.append((cid, t0, t_qual, t0 + win_ns))
            if cands:
                card_candidates[cid] = cands

        def active_cohort_cards_at(t_open_ns: int) -> set[str]:
            ov_start = t_open_ns - win_ns
            active = set()
            for cid, t0, t_qual, t_win_end in all_episodes:
                if t_qual <= t_open_ns and min(t_win_end, t_open_ns) >= ov_start:
                    active.add(cid)
            return active

        def card_fires_at(cid: str, flagged_id: int, t_open_ns: int) -> tuple[bool, bool]:
            """Returns (per_card_drip_ok, cohort_fires_ok) as of t_open_ns."""
            cands = card_candidates.get(cid)
            if not cands:
                return False, False
            ov_start = t_open_ns - win_ns
            card_ok = False
            for t0, t_qual, sub_t, sub_tx in cands:
                if t_qual <= t_open_ns and min(t0 + win_ns, t_open_ns) >= ov_start:
                    # Check if flagged_id is in sub_tx with timestamp <= t_open_ns
                    mask = sub_t <= t_open_ns
                    if flagged_id in sub_tx[mask]:
                        card_ok = True
                        break
            if not card_ok:
                return False, False
            active = active_cohort_cards_at(t_open_ns)
            n_other = len(active - {cid})
            return True, (n_other >= self.min_other_cards)

        # Require >= 90 days of history (60d quiet + 30d window) for as-of evaluation
        t_min_bank = pd.Timestamp(tx_sorted.t.iloc[0])
        min_eval_t = t_min_bank + pd.Timedelta(days=90)
        jan1_t = pd.Timestamp("2017-01-01 00:00:00")

        cc_eval = cc[
            (pd.to_datetime(cc.opened_at) >= min_eval_t)
            & (pd.to_datetime(cc.opened_at) < jan1_t)
        ]
        fraud_cases = cc_eval[cc_eval.outcome == "confirmed_fraud"]
        cleared_cases = cc_eval[cc_eval.outcome == "cleared"]

        caught: list[str] = []
        missed: list[str] = []
        per_card_fraud_caught = 0
        for row in fraud_cases.itertuples():
            fid = (
                int(row.first_fraud_txn_id)
                if pd.notna(row.first_fraud_txn_id)
                else (int(row.txn_list[0]) if row.txn_list else -1)
            )
            t_ns = pd.Timestamp(row.opened_at).value
            c_ok, coh_ok = card_fires_at(str(row.card_id), fid, t_ns)
            if c_ok:
                per_card_fraud_caught += 1
            if coh_ok:
                caught.append(str(row.case_id))
            else:
                missed.append(str(row.case_id))

        cleared_acted: list[str] = []
        per_card_cleared_hit = 0
        for row in cleared_cases.itertuples():
            fid = int(row.txn_list[0]) if row.txn_list else -1
            t_ns = pd.Timestamp(row.opened_at).value
            c_ok, coh_ok = card_fires_at(str(row.card_id), fid, t_ns)
            if c_ok:
                per_card_cleared_hit += 1
            if coh_ok:
                cleared_acted.append(str(row.case_id))

        # Legitimate card-days: cards NEVER named in any closed case, sampled in [min_eval_t, 2017-01-01)
        cased_cards = set(cc.card_id.dropna().astype(str).unique())
        for cl in cc.connected_list:
            if cl:
                cased_cards.update(cl)

        legit_pool = tx_sorted[
            (tx_sorted.t >= min_eval_t)
            & (tx_sorted.t < jan1_t)
            & (~tx_sorted.card_id.isin(cased_cards))
        ]
        legit_sample = (
            legit_pool.sample(n=sample_size, random_state=seed)
            if len(legit_pool) > sample_size
            else legit_pool
        )

        legit_card_day_hits: list[str] = []
        per_card_legit_hit = 0
        for row in legit_sample.itertuples():
            t_ns = pd.Timestamp(row.t).value
            c_ok, coh_ok = card_fires_at(str(row.card_id), int(row.TransactionID), t_ns)
            if c_ok:
                per_card_legit_hit += 1
            if coh_ok:
                legit_card_day_hits.append(str(row.TransactionID))

        # Scan daily 30d windows across [min_eval_t, 2017-01-01) to count historical windows with >= K+1 cards
        scan_days = pd.date_range(min_eval_t, jan1_t - pd.Timedelta(days=1), freq="1D")
        n_windows_scanned = len(scan_days)
        windows_firing_aug_nov = 0
        windows_firing_dec = 0
        max_active_aug_nov = 0
        max_active_dec = 0
        for d_ts in scan_days:
            act = active_cohort_cards_at(d_ts.value)
            n_act = len(act)
            if d_ts < pd.Timestamp("2016-12-01"):
                max_active_aug_nov = max(max_active_aug_nov, n_act)
                if n_act >= self.min_other_cards + 1:
                    windows_firing_aug_nov += 1
            else:
                max_active_dec = max(max_active_dec, n_act)
                if n_act >= self.min_other_cards + 1:
                    windows_firing_dec += 1

        n_fraud = len(fraud_cases)
        n_legit = len(legit_sample)
        c_f = len(caught)
        c_l = len(legit_card_day_hits)

        p1 = (c_f + 1.0) / (n_fraud + 2.0)
        p0 = (c_l + 1.0) / (n_legit + 2.0)
        lr_laplace = float(p1 / p0)
        se_log = math.sqrt((1.0 / (c_f + 1.0)) + (1.0 / (c_l + 1.0)))
        ci_low = float(lr_laplace * math.exp(-1.96 * se_log))
        ci_high = float(lr_laplace * math.exp(1.96 * se_log))

        self.lr_laplace = lr_laplace
        self.lr_ci = (ci_low, ci_high)
        # Per LR_RULE: if the cohort form never fires on history (c_f == 0), emit lr=1.0
        if c_f > 0 and ci_low > 1.0:
            self.lr = round(ci_low, 4)
        else:
            self.lr = 1.0

        self.backtest_stats = {
            "n_fraud_cases": n_fraud,
            "n_cleared_cases": len(cleared_cases),
            "n_legit_card_days": n_legit,
            "cohort_fraud_caught": c_f,
            "cohort_cleared_hit": len(cleared_acted),
            "cohort_legit_card_day_hit": c_l,
            "per_card_fraud_caught": per_card_fraud_caught,
            "per_card_cleared_hit": per_card_cleared_hit,
            "per_card_legit_card_day_hit": per_card_legit_hit,
            "lr_laplace": round(lr_laplace, 4),
            "lr_ci_95": (round(ci_low, 4), round(ci_high, 4)),
            "lr_used": self.lr,
            "n_windows_scanned_jul_dec": n_windows_scanned,
            "windows_firing_aug_nov": windows_firing_aug_nov,
            "windows_firing_dec": windows_firing_dec,
            "max_active_aug_nov": max_active_aug_nov,
            "max_active_dec": max_active_dec,
        }

        return Backtest(
            detector=self.name,
            n_cases=n_fraud + len(cleared_cases),
            caught=caught,
            missed=missed,
            legit_acted_against=cleared_acted,
            lr_measured=lr_laplace,
        )


def rerun_slow_drip_cohort_burst(
    tx: pd.DataFrame,
    card_id: str,
    flagged_txn_id: int,
    opened_at: str,
    max_amt: float = 6.50,
    min_charges: int = 4,
    min_span_days: float = 14.0,
    window_days: int = 30,
) -> dict:
    """Registered re-runnable query function for MicroCharge cohort burst attestations."""
    det = MicroCharge(
        max_amt=max_amt,
        min_charges=min_charges,
        min_span_days=min_span_days,
        window_days=window_days,
    )
    t_open = pd.to_datetime(opened_at)
    tx_asof = tx[tx.t <= t_open]
    ctx = CaseCtx(
        case_id="RERUN",
        flagged_txn_id=int(flagged_txn_id),
        card_id=str(card_id),
        customer_id="",
        opened_at=t_open,
        tx=tx_asof,
        cc=pd.DataFrame(),
    )
    return det.inspect_case(ctx)


class MicroCharge(Detector):
    name = "MicroCharge"
    pattern = "undocumented:slow_drip_micro_charges"
    enabled: bool = True

    def __init__(
        self,
        max_amt: float = 6.50,
        min_charges: int = 4,
        min_span_days: float = 14.0,
        window_days: int = 30,
        min_cohort: int = 5,
        max_p: float = 1e-4,
        enabled: bool = True,
    ):
        self.max_amt = max_amt
        self.min_charges = min_charges
        self.min_span_days = min_span_days
        self.window_days = window_days
        self.min_cohort = min_cohort
        self.max_p = max_p
        self.enabled = enabled
        self._measured_lr = 1.0
        self._lr_is_assumption = False
        self._lr_ci = (0.07, 17.12)
        self._p_fraud = 0.0
        self._p_legit = 0.0
        self._hist_window_counts: list[tuple[str, str, int]] = []

    def _compute_drip_starts(self, micro_c: pd.DataFrame, t_open: pd.Timestamp) -> pd.DataFrame:
        """For each card whose first-ever sub-$6.50 'C' charge occurs <= t_open, determines
        whether its initial 30-day drip has >= min_charges and span >= min_span_days."""
        if micro_c.empty:
            return pd.DataFrame(columns=["card_id", "t_start", "t_fourth", "n_charges", "span_days"])

        records = []
        win_td = pd.Timedelta(days=self.window_days)
        for cid, grp in micro_c.groupby("card_id", sort=False):
            t_vals = grp["t"].tolist()
            if len(t_vals) < self.min_charges:
                continue
            t0 = t_vals[0]
            t_limit = min(t0 + win_td, t_open)
            in_win = [t for t in t_vals if t <= t_limit]
            if len(in_win) < self.min_charges:
                continue
            span_d = (in_win[-1] - in_win[0]).total_seconds() / 86400.0
            if span_d >= self.min_span_days:
                records.append(
                    {
                        "card_id": str(cid),
                        "t_start": t0,
                        "t_qual": in_win[-1],
                        "n_charges": len(in_win),
                        "span_days": span_d,
                    }
                )
        return pd.DataFrame(records)

    def inspect_case(self, ctx: CaseCtx) -> dict:
        """Evaluates whether ctx.flagged_txn_id is a drip charge on a card belonging to
        a statistically significant 30-day cohort burst of novel slow-drip cards."""
        default_info = {
            "fired": False,
            "linking_value": "None",
            "card_n_charges": 0,
            "card_span_days": 0.0,
            "card_prior": 0,
            "flagged_is_drip": False,
            "k_now": 0,
            "baseline_mean": 0.0,
            "baseline_max": 0,
            "p_value": 1.0,
            "cohort_size": 0,
            "cohort_cards": [],
            "affected_txns": [],
            "hist_window_counts": [],
        }
        if ctx.tx.empty:
            return default_info

        t_open = pd.to_datetime(ctx.opened_at)
        t_win_start = t_open - pd.Timedelta(days=self.window_days)

        t_arr = ctx.tx.t.values
        i_end = np.searchsorted(t_arr, np.datetime64(t_open), side="right")
        tx_asof = ctx.tx.iloc[:i_end]
        if tx_asof.empty:
            return default_info

        micro_c = tx_asof[(tx_asof.ProductCD == "C") & (tx_asof.amount < self.max_amt)]
        if micro_c.empty:
            return default_info

        drips_df = self._compute_drip_starts(micro_c, t_open)

        cohort_cards = []
        if not drips_df.empty:
            now_drips = drips_df[(drips_df.t_start >= t_win_start) & (drips_df.t_start <= t_open)]
            cohort_cards = sorted(now_drips.card_id.tolist())
        k_now = len(cohort_cards)

        t_min_bank = pd.to_datetime(tx_asof.t.iloc[0])
        base_cutoff = min(t_win_start, pd.Timestamp("2017-01-01 00:00:00"))
        win_delta = pd.Timedelta(days=self.window_days)

        b_counts = []
        hist_windows = []
        cur_end = base_cutoff
        while cur_end >= t_min_bank + pd.Timedelta(days=14):
            cur_start = cur_end - win_delta
            if not drips_df.empty:
                c = int(((drips_df.t_start >= cur_start) & (drips_df.t_start < cur_end)).sum())
            else:
                c = 0
            b_counts.append(c)
            hist_windows.append((str(cur_start.date()), str(cur_end.date()), c))
            cur_end -= win_delta

        if not b_counts:
            b_counts = [0]

        base_mean = float(np.mean(b_counts))
        base_max = int(np.max(b_counts))
        p_val = poisson_tail_p(k_now, max(base_mean, 0.05))

        card_micro = micro_c[micro_c.card_id == ctx.card_id]
        if card_micro.empty:
            default_info.update(
                {
                    "k_now": k_now,
                    "cohort_size": k_now,
                    "baseline_mean": base_mean,
                    "baseline_max": base_max,
                    "p_value": p_val,
                    "hist_window_counts": hist_windows,
                }
            )
            return default_info

        prior_cnt = int((card_micro.t < t_win_start).sum())
        win_micro = card_micro[card_micro.t >= t_win_start]
        card_n = int(len(win_micro))
        card_span = (
            float((win_micro.t.max() - win_micro.t.min()).total_seconds() / 86400.0)
            if card_n > 0
            else 0.0
        )

        aff_txns = sorted([int(x) for x in win_micro.TransactionID.unique()])
        flagged_is_drip = int(ctx.flagged_txn_id) in aff_txns

        card_qualifies = (
            prior_cnt == 0
            and card_n >= self.min_charges
            and card_span >= self.min_span_days
            and flagged_is_drip
        )

        fired = (
            card_qualifies
            and k_now >= self.min_cohort
            and p_val < self.max_p
        )

        other_cohort = [c for c in cohort_cards if c != str(ctx.card_id)]

        return {
            "fired": fired,
            "linking_value": f"ProductCD=C&amt<{self.max_amt:.2f}" if card_n > 0 else "None",
            "card_n_charges": card_n,
            "card_span_days": card_span,
            "card_prior": prior_cnt,
            "flagged_is_drip": flagged_is_drip,
            "k_now": k_now,
            "baseline_mean": base_mean,
            "baseline_max": base_max,
            "p_value": p_val,
            "cohort_size": k_now,
            "cohort_cards": other_cohort,
            "all_cohort_cards": cohort_cards,
            "affected_txns": aff_txns,
            "hist_window_counts": hist_windows,
        }

    def run(self, ctx: CaseCtx, force: bool = False) -> list[Finding]:
        if not self.enabled and not force:
            return []

        info = self.inspect_case(ctx)
        if not info["fired"]:
            return []

        claim = Attestation(
            what=(
                f"Coordinated slow-drip micro-charge cohort burst on card {ctx.card_id}: {info['card_n_charges']} "
                f"ProductCD 'C' online charges < ${self.max_amt:.2f} over {info['card_span_days']:.1f}d with 0 prior; "
                f"{info['k_now']} cards started this novel drip in 30d before {ctx.opened_at} "
                f"(Jul-Dec 30d baseline mean={info['baseline_mean']:.2f}, max={info['baseline_max']}, "
                f"Poisson p={info['p_value']:.2e}); LR={self._measured_lr:.2f} (measured)"
            ),
            evidence=Evidence(
                txn_ids=info["affected_txns"],
                card_ids=info["all_cohort_cards"],
                customer_ids=[str(ctx.customer_id)],
                case_ids=[ctx.case_id],
            ),
            query="rerun_slow_drip_cohort_burst",
            params={
                "card_id": str(ctx.card_id),
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "opened_at": str(ctx.opened_at),
                "max_amt": self.max_amt,
                "min_charges": self.min_charges,
                "min_span_days": self.min_span_days,
                "window_days": self.window_days,
                "k_now": info["k_now"],
                "baseline_mean": round(info["baseline_mean"], 4),
                "baseline_max": info["baseline_max"],
                "p_value": float(f"{info['p_value']:.4e}"),
            },
        )

        return [
            Finding(
                detector=self.name,
                pattern=self.pattern,
                claim=claim,
                lr=self._measured_lr,
                affected_txn_ids=info["affected_txns"],
                connected_card_ids=info["cohort_cards"],
            )
        ]

    def backtest(
        self, cc: pd.DataFrame, tx: pd.DataFrame, sample_size: int = 5000, seed: int = 42
    ) -> Backtest:
        """Vectorised as-of replay of the slow-drip cohort-burst detector over Jul-Dec
        confirmed fraud cases and >= 5,000 Jul-Oct legitimate card-days."""
        tx_sorted = tx.sort_values("t") if not tx.t.is_monotonic_increasing else tx
        micro_c = tx_sorted[(tx_sorted.ProductCD == "C") & (tx_sorted.amount < self.max_amt)]

        win_td = np.timedelta64(self.window_days, "D")
        min_span_ns = int(self.min_span_days * 86400 * 1e9)

        card_drip_info: dict[str, tuple[np.datetime64, np.datetime64, set[int]]] = {}
        drip_starts_list = []

        for cid, grp in micro_c.groupby("card_id", sort=False):
            t_arr = grp["t"].to_numpy(dtype="datetime64[ns]")
            if len(t_arr) < self.min_charges:
                continue
            t0 = t_arr[0]
            ir = np.searchsorted(t_arr, t0 + win_td, side="right")
            if ir < self.min_charges:
                continue
            qual_idx = -1
            for idx in range(self.min_charges - 1, ir):
                if int((t_arr[idx] - t0).astype("timedelta64[ns]").astype(np.int64)) >= min_span_ns:
                    qual_idx = idx
                    break
            if qual_idx != -1:
                t_qual = t_arr[qual_idx]
                drip_txns = set(grp["TransactionID"].values[:ir].astype(int).tolist())
                card_drip_info[str(cid)] = (t0, t_qual, drip_txns)
                drip_starts_list.append((t0, t_qual))

        t_min_bank = pd.to_datetime(tx_sorted.t.iloc[0])
        jan1_ts = pd.Timestamp("2017-01-01 00:00:00")
        win_delta = pd.Timedelta(days=self.window_days)
        cur_end = jan1_ts
        hist_counts = []
        while cur_end >= t_min_bank + pd.Timedelta(days=14):
            cur_start = cur_end - win_delta
            s_np = np.datetime64(cur_start)
            e_np = np.datetime64(cur_end)
            c = sum(1 for (t0, tq) in drip_starts_list if s_np <= t0 < e_np and tq <= e_np)
            hist_counts.append((str(cur_start.date()), str(cur_end.date()), int(c)))
            cur_end -= win_delta
        self._hist_window_counts = list(reversed(hist_counts))

        t_min_np = np.datetime64(t_min_bank)
        jan1_np = np.datetime64("2017-01-01T00:00:00")
        min_hist_td = np.timedelta64(14, "D")

        def check_card_at(card_id: str, flagged_id: int, t_open_np: np.datetime64) -> bool:
            info = card_drip_info.get(card_id)
            if info is None:
                return False
            t0, t_qual, drip_txns = info
            t_start_np = t_open_np - win_td
            if not (t_start_np <= t0 <= t_open_np and t_qual <= t_open_np and flagged_id in drip_txns):
                return False

            k_now = sum(
                1
                for (s0, sq) in drip_starts_list
                if t_start_np <= s0 <= t_open_np and sq <= t_open_np
            )
            if k_now < self.min_cohort:
                return False

            base_cut = min(t_start_np, jan1_np)
            b_counts = []
            ce = base_cut
            while ce >= t_min_np + min_hist_td:
                cs = ce - win_td
                bc = sum(1 for (s0, sq) in drip_starts_list if cs <= s0 < ce and sq <= ce)
                b_counts.append(bc)
                ce = cs
            if not b_counts:
                b_counts = [0]
            base_mean = sum(b_counts) / len(b_counts)
            return poisson_tail_p(k_now, max(base_mean, 0.05)) < self.max_p

        min_eval_t = t_min_bank + pd.Timedelta(days=90)
        fraud_cases = cc[(cc.outcome == "confirmed_fraud") & (pd.to_datetime(cc.opened_at) >= min_eval_t)]
        caught = []
        missed = []

        for row in fraud_cases.itertuples():
            flagged_id = (
                int(row.first_fraud_txn_id)
                if pd.notna(row.first_fraud_txn_id)
                else (int(row.txn_list[0]) if row.txn_list else -1)
            )
            t_open_np = np.datetime64(pd.to_datetime(row.opened_at))
            if check_card_at(str(row.card_id), flagged_id, t_open_np):
                caught.append(str(row.case_id))
            else:
                missed.append(str(row.case_id))

        closed_txns = set()
        for tl in cc.txn_list:
            if tl:
                closed_txns.update(tl)

        jul_oct_end = pd.Timestamp("2016-11-01 00:00:00")
        legit_pool = tx_sorted[
            (tx_sorted.t >= min_eval_t) & (tx_sorted.t < jul_oct_end)
            & (~tx_sorted.TransactionID.isin(closed_txns))
        ]
        legit_sample = (
            legit_pool.sample(n=sample_size, random_state=seed)
            if len(legit_pool) > sample_size
            else legit_pool
        )

        legit_acted_against = []
        for row in legit_sample.itertuples():
            t_open_np = np.datetime64(pd.to_datetime(row.t))
            if check_card_at(str(row.card_id), int(row.TransactionID), t_open_np):
                legit_acted_against.append(str(row.TransactionID))

        n_fraud = len(fraud_cases)
        n_legit = len(legit_sample)
        c_f = len(caught)
        c_l = len(legit_acted_against)

        self._p_fraud = c_f / n_fraud if n_fraud else 0.0
        self._p_legit = c_l / n_legit if n_legit else 0.0

        p1 = (c_f + 1.0) / (n_fraud + 2.0)
        p0 = (c_l + 1.0) / (n_legit + 2.0)
        lr_empirical = float(p1 / p0)

        se_log_lr = math.sqrt((1.0 / (c_f + 1.0)) + (1.0 / (c_l + 1.0)))
        self._lr_ci = (
            float(lr_empirical * math.exp(-1.96 * se_log_lr)),
            float(lr_empirical * math.exp(1.96 * se_log_lr)),
        )

        if c_f > 0 and self._lr_ci[0] > 1.0:
            self._measured_lr = self._lr_ci[0]
        else:
            self._measured_lr = 1.0
        self._lr_is_assumption = False

        return Backtest(
            detector=self.name,
            n_cases=n_fraud + n_legit,
            caught=caught,
            missed=missed,
            legit_acted_against=legit_acted_against,
            lr_measured=lr_empirical,
        )
