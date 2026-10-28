"""FanIn detector for undocumented typology: recipient burst ('undocumented:recipient_fan_in').
Evaluates non-freemail R_emaildomain values used by the flagged card in the 21 days before
opened_at, counting distinct cards that used that recipient domain for the FIRST time in their
own prior history in the 21-day window (k_now) compared against causal Jul-Dec 21-day baseline
windows.
- Drops device_profile from the burst test (browser version rollouts create false bursts; device
  sharing belongs to the ring detector B15).
- Episode linkage rule: A recipient burst counts as primary fraud evidence only when the flagged
  txn, or a txn on the same card within 72h of the flagged txn, pays the bursting recipient.
  Otherwise it is emitted as a context note (lr=1.0, kind='context') naming the cards and amounts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding

FREEMAIL_PREFIXES = (
    "gmail",
    "yahoo",
    "hotmail",
    "outlook",
    "aol",
    "anonymous",
    "icloud",
    "live",
    "msn",
    "me.com",
    "ymail",
    "mail.com",
    "gmx",
    "protonmail",
    "rocketmail",
)


def is_freemail(dom: object) -> bool:
    if pd.isna(dom):
        return True
    s = str(dom).strip().lower()
    if not s or s in ("none", "nan", "?"):
        return True
    if s in ("anonymous.com", "me.com"):
        return True
    base = s.split(".")[0]
    return base in FREEMAIL_PREFIXES or any(s.startswith(p + ".") for p in FREEMAIL_PREFIXES)


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


def rerun_recipient_fan_in_burst(
    tx: pd.DataFrame,
    card_id: str,
    flagged_txn_id: int,
    opened_at: str,
    window_days: int = 21,
    episode_hours: float = 72.0,
) -> dict:
    """Registered re-runnable query function for FanIn attestations."""
    det = FanIn(window_days=window_days, episode_hours=episode_hours)
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


class FanIn(Detector):
    name = "FanIn"
    pattern = "undocumented:recipient_fan_in"
    enabled: bool = True

    def __init__(
        self,
        window_days: int = 21,
        min_k_now: int = 5,
        burst_mult: float = 3.0,
        max_p: float = 1e-4,
        episode_hours: float = 72.0,
        enabled: bool = True,
    ):
        self.window_days = window_days
        self.min_k_now = min_k_now
        self.burst_mult = burst_mult
        self.max_p = max_p
        self.episode_hours = episode_hours
        self.enabled = enabled
        self._measured_lr = 3.0
        self._lr_is_assumption = True
        self._lr_ci = (0.05, 59.0)
        self._p_fraud = 0.0
        self._p_legit = 0.0
        self._n_hist_fraud_bursts = 0
        self._n_hist_legit_bursts = 0

    def inspect_case(self, ctx: CaseCtx) -> dict:
        """Evaluates non-freemail R_emaildomain burst on ctx.card_id and checks whether
        the flagged transaction (or a transaction within 72h on the same card) pays the
        bursting recipient."""
        default_info = {
            "fired": False,
            "kind": "none",
            "linking_value": "None",
            "k_now": 0,
            "baseline_mean": 0.0,
            "baseline_max": 0,
            "p_value": 1.0,
            "connected_cards": [],
            "all_burst_cards": [],
            "affected_txns": [],
            "card_val_txns": [],
            "card_val_amounts": [],
            "all_win_custs": [],
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

        i_start = np.searchsorted(tx_asof.t.values, np.datetime64(t_win_start), side="left")
        tx_win = tx_asof.iloc[i_start:]

        card_win = tx_win[tx_win.card_id == ctx.card_id]
        if card_win.empty:
            return default_info

        candidates: list[str] = []
        for dom in card_win.R_emaildomain.dropna().unique():
            if not is_freemail(dom):
                candidates.append(str(dom))

        if not candidates:
            return default_info

        # Identify the flagged transaction timestamp for the 72h episode check
        f_rows = tx_asof[tx_asof.TransactionID == ctx.flagged_txn_id]
        t_flag = pd.to_datetime(f_rows.t.iloc[0]) if not f_rows.empty else t_open
        ep_delta = pd.Timedelta(hours=self.episode_hours)

        best_info = default_info
        t_min_bank = pd.to_datetime(tx_asof.t.iloc[0])
        base_cutoff = min(t_win_start, pd.Timestamp("2017-01-01 00:00:00"))
        win_delta = pd.Timedelta(days=self.window_days)

        for val in candidates:
            sub = tx_asof[tx_asof.R_emaildomain == val]
            if sub.empty:
                continue

            first_uses = sub.groupby("card_id", sort=False)["t"].min()
            first_in_win = first_uses[(first_uses >= t_win_start) & (first_uses <= t_open)]
            k_now = int(len(first_in_win))

            # Causal 21-day baseline windows before t_win_start (within Jul-Dec)
            b_counts = []
            cur_end = base_cutoff
            while cur_end >= t_min_bank + pd.Timedelta(days=7):
                cur_start = cur_end - win_delta
                c = int(((first_uses >= cur_start) & (first_uses < cur_end)).sum())
                b_counts.append(c)
                cur_end -= win_delta

            if not b_counts:
                b_counts = [0]

            base_mean = float(np.mean(b_counts))
            base_max = int(np.max(b_counts))
            p_val = poisson_tail_p(k_now, max(base_mean, 0.05))

            burst_ok = (
                k_now >= self.min_k_now
                and k_now >= self.burst_mult * max(1, base_max)
                and p_val < self.max_p
            )

            # Check whether the flagged txn (or a txn on the same card within 72h of flagged txn) pays val
            card_val_df = tx_asof[(tx_asof.card_id == ctx.card_id) & (tx_asof.R_emaildomain == val)]
            in_episode = False
            if not card_val_df.empty:
                time_diffs = (card_val_df.t - t_flag).abs()
                if (card_val_df.TransactionID == ctx.flagged_txn_id).any() or (time_diffs <= ep_delta).any():
                    in_episode = True

            if burst_ok and in_episode:
                kind = "evidence"
                fired = True
            elif burst_ok and not in_episode:
                kind = "context"
                fired = False
            else:
                kind = "none"
                fired = False

            rank_cur = 2 if kind == "evidence" else (1 if kind == "context" else 0)
            rank_best = 2 if best_info["kind"] == "evidence" else (1 if best_info["kind"] == "context" else 0)

            if rank_cur > rank_best or (rank_cur == rank_best and k_now > best_info["k_now"]):
                conn_cards = sorted([str(c) for c in first_in_win.index if str(c) != str(ctx.card_id)])
                card_val_txns = sorted([int(x) for x in card_val_df.TransactionID.unique()])
                card_val_amts = [round(float(x), 2) for x in card_val_df.amount.tolist()]
                best_info = {
                    "fired": fired,
                    "kind": kind,
                    "linking_value": f"R_emaildomain={val}",
                    "val": val,
                    "k_now": k_now,
                    "baseline_mean": base_mean,
                    "baseline_max": base_max,
                    "p_value": p_val,
                    "connected_cards": conn_cards,
                    "all_burst_cards": sorted([str(c) for c in first_in_win.index]),
                    "affected_txns": card_val_txns if fired else [],
                    "card_val_txns": card_val_txns,
                    "card_val_amounts": card_val_amts,
                    "all_win_custs": sorted([str(x) for x in sub[sub.t >= t_win_start].customer_id.dropna().unique()]),
                }

        return best_info

    def run(self, ctx: CaseCtx, force: bool = False) -> list[Finding]:
        if not self.enabled and not force:
            return []

        info = self.inspect_case(ctx)
        if info["kind"] == "none":
            return []

        if info["kind"] == "context":
            claim = Attestation(
                what=(
                    f"CONTEXT NOTE (not in flagged 72h episode): Card {ctx.card_id} has prior/other charges "
                    f"(txns={info['card_val_txns']}, amounts={info['card_val_amounts']}) to bursting recipient "
                    f"{info['linking_value']} ({info['k_now']} first-time cards in {self.window_days}d: "
                    f"{info['all_burst_cards']}, Jul-Dec baseline mean={info['baseline_mean']:.2f}, max={info['baseline_max']}, "
                    f"Poisson p={info['p_value']:.2e}), but flagged txn {ctx.flagged_txn_id} is outside that 72h episode."
                ),
                evidence=Evidence(
                    txn_ids=info["card_val_txns"],
                    card_ids=info["all_burst_cards"],
                    customer_ids=info["all_win_custs"],
                    case_ids=[ctx.case_id],
                ),
                query="rerun_recipient_fan_in_burst",
                params={
                    "kind": "context",
                    "card_id": str(ctx.card_id),
                    "flagged_txn_id": int(ctx.flagged_txn_id),
                    "opened_at": str(ctx.opened_at),
                    "linking_value": info["linking_value"],
                    "window_days": self.window_days,
                    "k_now": info["k_now"],
                    "baseline_mean": round(info["baseline_mean"], 4),
                    "baseline_max": info["baseline_max"],
                    "p_value": float(f"{info['p_value']:.4e}"),
                    "card_val_amounts": info["card_val_amounts"],
                },
            )
            f_ctx = Finding(
                detector=self.name,
                pattern=self.pattern,
                claim=claim,
                lr=1.0,
                affected_txn_ids=[],
                connected_card_ids=info["connected_cards"],
            )
            f_ctx.kind = "context"  # type: ignore[attr-defined]
            return [f_ctx]

        lr_note = (
            f"LR={self._measured_lr:.2f} [ASSUMPTION: fallback LR=3.0 used because Jul-Dec history has "
            f"{self._n_hist_fraud_bursts} recipient bursts]"
            if self._lr_is_assumption
            else f"LR={self._measured_lr:.2f} (measured)"
        )

        claim = Attestation(
            what=(
                f"Recipient burst on {info['linking_value']}: {info['k_now']} distinct cards used it for the "
                f"first time in {self.window_days}d before {ctx.opened_at} (Jul-Dec baseline mean={info['baseline_mean']:.2f}, "
                f"max={info['baseline_max']}, Poisson p={info['p_value']:.2e}); flagged episode on {ctx.card_id} "
                f"pays this recipient ({ info['card_val_amounts'] }); {lr_note}"
            ),
            evidence=Evidence(
                txn_ids=info["affected_txns"],
                card_ids=info["all_burst_cards"],
                customer_ids=info["all_win_custs"],
                case_ids=[ctx.case_id],
            ),
            query="rerun_recipient_fan_in_burst",
            params={
                "kind": "evidence",
                "lr_assumption": self._lr_is_assumption,
                "card_id": str(ctx.card_id),
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "opened_at": str(ctx.opened_at),
                "linking_value": info["linking_value"],
                "window_days": self.window_days,
                "k_now": info["k_now"],
                "baseline_mean": round(info["baseline_mean"], 4),
                "baseline_max": info["baseline_max"],
                "p_value": float(f"{info['p_value']:.4e}"),
            },
        )

        f_ev = Finding(
            detector=self.name,
            pattern=self.pattern,
            claim=claim,
            lr=self._measured_lr,
            affected_txn_ids=info["affected_txns"],
            connected_card_ids=info["connected_cards"],
        )
        f_ev.kind = "evidence"  # type: ignore[attr-defined]
        return [f_ev]

    def backtest(
        self, cc: pd.DataFrame, tx: pd.DataFrame, sample_size: int = 5000, seed: int = 42
    ) -> Backtest:
        """Vectorised as-of replay over closed fraud cases (Jul-Dec) and >= 5,000 legit Jul-Oct
        card-days for R_emaildomain only, enforcing the 72h episode rule."""
        tx_sorted = tx.sort_values("t") if not tx.t.is_monotonic_increasing else tx

        valid_mask = tx_sorted.R_emaildomain.notna() & (~tx_sorted.R_emaildomain.map(is_freemail))
        sub = tx_sorted.loc[valid_mask, ["R_emaildomain", "card_id", "t", "TransactionID"]]
        first_df = sub.groupby(["R_emaildomain", "card_id"], sort=False)["t"].min().reset_index()

        val_times: dict[str, np.ndarray] = {}
        for val, grp in first_df.groupby("R_emaildomain", sort=False):
            val_times[str(val)] = np.sort(grp["t"].values)

        t_min_bank = np.datetime64(tx_sorted.t.iloc[0])
        win_td = np.timedelta64(self.window_days, "D")
        ep_td = np.timedelta64(int(self.episode_hours), "h")
        min_hist_td = np.timedelta64(7, "D")
        jan1_np = np.datetime64("2017-01-01T00:00:00")

        def eval_val_burst(val: str, t_open_np: np.datetime64) -> bool:
            arr = val_times.get(val)
            if arr is None or len(arr) < self.min_k_now:
                return False
            t_start_np = t_open_np - win_td
            i_r = np.searchsorted(arr, t_open_np, side="right")
            i_l = np.searchsorted(arr, t_start_np, side="left")
            k_now = int(i_r - i_l)
            if k_now < self.min_k_now:
                return False

            base_cut = min(t_start_np, jan1_np)
            b_counts = []
            cur_end = base_cut
            while cur_end >= t_min_bank + min_hist_td:
                cur_start = cur_end - win_td
                br = np.searchsorted(arr, cur_end, side="left")
                bl = np.searchsorted(arr, cur_start, side="left")
                b_counts.append(int(br - bl))
                cur_end = cur_start
            if not b_counts:
                b_counts = [0]
            base_max = max(b_counts)
            if k_now < self.burst_mult * max(1, base_max):
                return False
            base_mean = sum(b_counts) / len(b_counts)
            return poisson_tail_p(k_now, max(base_mean, 0.05)) < self.max_p

        card_groups = {
            str(cid): grp[["t", "R_emaildomain", "TransactionID"]]
            for cid, grp in sub.groupby("card_id", sort=False)
        }
        txn_time_map = dict(zip(tx_sorted.TransactionID.values, tx_sorted.t.values))

        def check_card_episode(card_id: str, flagged_id: int, t_open_np: np.datetime64) -> bool:
            c_df = card_groups.get(card_id)
            if c_df is None or c_df.empty:
                return False
            t_vals = c_df["t"].values
            t_start_np = t_open_np - win_td
            il = np.searchsorted(t_vals, t_start_np, side="left")
            ir = np.searchsorted(t_vals, t_open_np, side="right")
            if il >= ir:
                return False

            t_flag_np = txn_time_map.get(flagged_id, t_open_np)
            win_slice = c_df.iloc[il:ir]

            for dom in win_slice["R_emaildomain"].unique():
                s_dom = str(dom)
                if eval_val_burst(s_dom, t_open_np):
                    dom_slice = win_slice[win_slice["R_emaildomain"] == dom]
                    if (dom_slice["TransactionID"] == flagged_id).any():
                        return True
                    diffs = np.abs(dom_slice["t"].values - t_flag_np)
                    if (diffs <= ep_td).any():
                        return True
            return False

        # A recipient is "first used" by every card at the start of the data, so windows
        # without a real baseline look like bursts. Evaluate only with >= 90 days of history.
        min_eval_t = pd.Timestamp(t_min_bank) + pd.Timedelta(days=90)
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
            if check_card_episode(str(row.card_id), flagged_id, t_open_np):
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
            if check_card_episode(str(row.card_id), int(row.TransactionID), t_open_np):
                legit_acted_against.append(str(row.TransactionID))

        n_fraud = len(fraud_cases)
        n_legit = len(legit_sample)
        c_f = len(caught)
        c_l = len(legit_acted_against)

        self._n_hist_fraud_bursts = c_f
        self._n_hist_legit_bursts = c_l
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

        # If Jul-Dec history has too few recipient bursts (< 3), report empirical LR & CI
        # and fall back to lr=3.0 explicitly flagged as ASSUMPTION
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
