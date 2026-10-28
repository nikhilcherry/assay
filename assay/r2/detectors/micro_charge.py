"""MicroCharge detector for undocumented typology: coordinated slow-drip micro-charge
campaign cohort burst ('undocumented:slow_drip_micro_charges').
Per-card slow drips alone are not a fraud signal historically (LR 0.63, legit 1.62%).
What is new in January is the COHORT BURST: 11-17 cards starting a novel slow drip
(>= 4 ProductCD 'C' online charges < $6.50, >= 14d span, ZERO prior sub-$6.50 'C' charges)
within a 30-day window.
- Counts cards whose novel slow drip STARTS in the 30-day window before opened_at (k_now).
- Baseline = mean and max of the same count over every 30-day window across Jul-Dec.
- Fires when k_now >= 5, Poisson tail p < 1e-4, AND the flagged txn is one of the drip charges."""

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
        self._measured_lr = 3.0
        self._lr_is_assumption = True
        self._lr_ci = (0.05, 59.0)
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
            # Charges within [t0, min(t0 + 30d, t_open)]
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

        # Compute all cards in the bank (as-of t_open) that have a valid novel slow-drip starting at t_start
        drips_df = self._compute_drip_starts(micro_c, t_open)

        # Cohort of cards whose novel slow-drip STARTED in [t_win_start, t_open]
        cohort_cards = []
        if not drips_df.empty:
            now_drips = drips_df[(drips_df.t_start >= t_win_start) & (drips_df.t_start <= t_open)]
            cohort_cards = sorted(now_drips.card_id.tolist())
        k_now = len(cohort_cards)

        # Causal 30-day baseline windows across Jul-Dec before t_win_start
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

        # Check flagged card's own sub-$6.50 'C' charges
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

        lr_note = (
            f"LR={self._measured_lr:.2f} [ASSUMPTION: fallback LR=3.0 used because Jul-Dec had 0 cohort bursts]"
            if self._lr_is_assumption
            else f"LR={self._measured_lr:.2f} (measured)"
        )

        claim = Attestation(
            what=(
                f"Coordinated slow-drip micro-charge cohort burst on card {ctx.card_id}: {info['card_n_charges']} "
                f"ProductCD 'C' online charges < ${self.max_amt:.2f} over {info['card_span_days']:.1f}d with 0 prior; "
                f"{info['k_now']} cards started this novel drip in 30d before {ctx.opened_at} "
                f"(Jul-Dec 30d baseline mean={info['baseline_mean']:.2f}, max={info['baseline_max']}, "
                f"Poisson p={info['p_value']:.2e}); {lr_note}"
            ),
            evidence=Evidence(
                txn_ids=info["affected_txns"],
                card_ids=info["all_cohort_cards"],
                customer_ids=[str(ctx.customer_id)],
                case_ids=[ctx.case_id],
            ),
            query="rerun_slow_drip_cohort_burst",
            params={
                "lr_assumption": self._lr_is_assumption,
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

        # Precompute card drip start & qualification times across the dataset
        win_td = np.timedelta64(self.window_days, "D")
        min_span_ns = int(self.min_span_days * 86400 * 1e9)

        card_drip_info: dict[str, tuple[np.datetime64, np.datetime64, set[int]]] = {}
        drip_starts_list = []

        for cid, grp in micro_c.groupby("card_id", sort=False):
            t_arr = grp["t"].values
            if len(t_arr) < self.min_charges:
                continue
            t0 = t_arr[0]
            ir = np.searchsorted(t_arr, t0 + win_td, side="right")
            if ir < self.min_charges:
                continue
            # Find the earliest index >= min_charges-1 where span >= min_span_days
            qual_idx = -1
            for idx in range(self.min_charges - 1, ir):
                if int(t_arr[idx] - t0) >= min_span_ns:
                    qual_idx = idx
                    break
            if qual_idx != -1:
                t_qual = t_arr[qual_idx]
                drip_txns = set(grp["TransactionID"].values[:ir].astype(int).tolist())
                card_drip_info[str(cid)] = (t0, t_qual, drip_txns)
                drip_starts_list.append((t0, t_qual))

        # Record Jul-Dec 30-day window counts for documentation and test output
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
            # Card's novel drip must have started in [t_open - 30d, t_open], qualified by t_open,
            # and flagged_id must be one of the drip charges
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

        # "No prior small charges" is trivially true at the start of the data; evaluate
        # only cases with >= 90 days of history behind them.
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

        if c_f < 3:
            self._measured_lr = 3.0
            self._lr_is_assumption = True
        else:
            self._measured_lr = lr_empirical
            self._lr_is_assumption = False

        return Backtest(
            detector=self.name,
            n_cases=n_fraud + n_legit,
            caught=caught,
            missed=missed,
            legit_acted_against=legit_acted_against,
            lr_measured=lr_empirical,
        )
