"""FanIn detector for undocumented typology: recipient fan-in cohort ('undocumented:recipient_fan_in').

Refinements for Block C3(b):
1. Removed the unmeasured `lr = 3.0` assumption (`_lr_is_assumption` removed; never emits 3.0).
2. Tightened recipient cohort per data findings:
   - The flagged txn ITSELF must pay the recipient (`channel == 'online'`, `ProductCD == 'C'`,
     `R_emaildomain` non-freemail by base rate, and `amount >= min_amount`).
   - Cohort cards are cards whose FIRST charge to that `R_emaildomain` (online, `ProductCD == 'C'`,
     `amount >= min_amount`) falls in `[opened_at - 21d, opened_at]`.
   - `min_amount = 100.0`: justified by Jul-Dec historical ProductCD=='C' non-freemail charge amount
     distributions (where < $100 charges have a ~2.8x higher legitimate/cleared biller share than
     >= $100 charges, and in January separates the 12-card $100-$400 `sc.rr.com` fan-in cohort from
     the 4-card $30-$70 legitimate biller group on the same domain).
   - If the flagged txn does NOT itself pay the bursting recipient (e.g. HHF-013 where the flagged
     txn is an in-person `ProductCD == 'W'` purchase), `FanIn` does not fire as primary evidence; if
     the card has prior charges to a bursting recipient in the window, it emits a `kind='context'`
     note (`lr=1.0`, `connected_card_ids=[]`, `affected_txn_ids=[]`).
3. Measured per LR_RULE over Jul-Dec closed cases and 5,000 never-cased legitimate card-days:
   - Because the 21d recipient fan-in burst (>= 5 first-time cards, >= 3x baseline max, Poisson p < 1e-4)
     fired 0 times in Jul-Dec history, `lr` is set to `1.0` (context/cohort finding carrying
     `connected_card_ids` and attested ids for Policy §3a / R6 / R9), with any non-measured weight
     isolated in `lr_proposal`.
"""

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


def is_freemail(dom: object, freemail_set: set[str] | None = None) -> bool:
    """Identifies high-volume consumer freemail domains by prefix or base-rate card share."""
    if pd.isna(dom):
        return True
    s = str(dom).strip().lower()
    if not s or s in ("none", "nan", "?"):
        return True
    if freemail_set is not None and s in freemail_set:
        return True
    if s in ("anonymous.com", "me.com"):
        return True
    base = s.split(".")[0]
    return base in FREEMAIL_PREFIXES or any(s.startswith(p + ".") for p in FREEMAIL_PREFIXES)


def freemail_domains_by_base_rate(tx_asof: pd.DataFrame, min_card_share: float = 0.01) -> set[str]:
    """Computes high-base-rate consumer freemail domains (>1% of all distinct online C cards)
    in addition to standard freemail prefixes."""
    sub = tx_asof[(tx_asof.channel == "online") & (tx_asof.ProductCD == "C") & tx_asof.R_emaildomain.notna()]
    if sub.empty:
        return set()
    total_cards = max(1, sub.card_id.nunique())
    dom_cards = sub.groupby("R_emaildomain", sort=False)["card_id"].nunique()
    high_rate = {str(d).strip().lower() for d, c in dom_cards.items() if (c / total_cards) >= min_card_share}
    return high_rate


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


def analyze_amount_floor_distribution(tx: pd.DataFrame, cc: pd.DataFrame) -> dict:
    """Analyses Jul-Dec historical ProductCD=='C' online non-freemail R_emaildomain charges
    and January `sc.rr.com` charges by amount band to show how `min_amount = 100.0` separates
    small recurring/utility biller charges ($30-$70) from larger fan-in charges ($100-$400).
    """
    jan1 = pd.Timestamp("2017-01-01 00:00:00")
    fraud_txns = set()
    cleared_txns = set()
    for row in cc.itertuples():
        if row.txn_list:
            if row.outcome == "confirmed_fraud":
                fraud_txns.update(row.txn_list)
            elif row.outcome == "cleared":
                cleared_txns.update(row.txn_list)

    sub = tx[
        (tx.channel == "online")
        & (tx.ProductCD == "C")
        & tx.R_emaildomain.notna()
        & (~tx.R_emaildomain.map(is_freemail))
    ].copy()

    hist = sub[sub.t < jan1]
    bands = [
        ("< $30", 0.0, 30.0),
        ("$30 - $70 (biller band)", 30.0, 70.0),
        ("$70 - $100", 70.0, 100.0),
        (">= $100 (floor)", 100.0, 1e9),
    ]
    hist_table = []
    for label, lo, hi in bands:
        b = hist[(hist.amount >= lo) & (hist.amount < hi)]
        n_tx = len(b)
        n_cards = b.card_id.nunique()
        n_f = int(b.TransactionID.isin(fraud_txns).sum())
        n_c = int(b.TransactionID.isin(cleared_txns).sum())
        n_un = n_tx - n_f - n_c
        hist_table.append(
            {
                "band": label,
                "n_txns_jul_dec": n_tx,
                "n_cards_jul_dec": n_cards,
                "confirmed_fraud_txns": n_f,
                "cleared_or_uncased_txns": n_c + n_un,
                "fraud_rate": round(n_f / n_tx, 4) if n_tx else 0.0,
            }
        )

    # Also inspect sc.rr.com in Jan 2017 (without using TransactionID as a signal, just reporting amounts)
    jan_sc = sub[(sub.t >= jan1) & (sub.R_emaildomain == "sc.rr.com")]
    sc_lt_100 = jan_sc[jan_sc.amount < 100.0]
    sc_ge_100 = jan_sc[jan_sc.amount >= 100.0]

    return {
        "jul_dec_bands": hist_table,
        "jan_sc_rr_lt_100": {
            "n_txns": int(len(sc_lt_100)),
            "n_cards": int(sc_lt_100.card_id.nunique()),
            "cards": sorted(sc_lt_100.card_id.astype(str).unique().tolist()),
            "min_amt": round(float(sc_lt_100.amount.min()), 2) if len(sc_lt_100) else 0.0,
            "max_amt": round(float(sc_lt_100.amount.max()), 2) if len(sc_lt_100) else 0.0,
        },
        "jan_sc_rr_ge_100": {
            "n_txns": int(len(sc_ge_100)),
            "n_cards": int(sc_ge_100.card_id.nunique()),
            "cards": sorted(sc_ge_100.card_id.astype(str).unique().tolist()),
            "min_amt": round(float(sc_ge_100.amount.min()), 2) if len(sc_ge_100) else 0.0,
            "max_amt": round(float(sc_ge_100.amount.max()), 2) if len(sc_ge_100) else 0.0,
        },
    }


def cohort_members(
    tx_asof: pd.DataFrame,
    flagged_txn_id: int,
    opened_at: pd.Timestamp | str,
    window_days: int = 21,
    min_amount: float = 100.0,
    min_k_now: int = 5,
    burst_mult: float = 3.0,
    max_p: float = 1e-4,
) -> tuple[list[str], dict[str, list[int]], dict]:
    """Pure function for GSQL mirroring: computes the recipient fan-in cohort as-of `opened_at`.

    Rules:
      1. The flagged transaction ITSELF (`TransactionID == flagged_txn_id`) must be in `tx_asof`
         (with `t <= opened_at`), `channel == 'online'`, `ProductCD == 'C'`, `amount >= min_amount`,
         and have a non-freemail `R_emaildomain`.
      2. Cohort cards are cards whose FIRST charge to that `R_emaildomain` falls in
         `[opened_at - 21d, opened_at]` AND has an online `ProductCD == 'C'` charge to that
         `R_emaildomain` with `amount >= min_amount` in `[opened_at - 21d, opened_at]`.
      3. Must satisfy burst criteria (`k_now >= min_k_now`, `k_now >= burst_mult * max(1, baseline_max)`,
         `poisson_tail_p < max_p`).

    Returns:
      (member_card_ids, per_card_txn_ids, params_used)
    """
    t_open = pd.Timestamp(opened_at)
    params: dict = {
        "flagged_txn_id": int(flagged_txn_id),
        "opened_at": str(t_open),
        "window_days": int(window_days),
        "min_amount": float(min_amount),
        "min_k_now": int(min_k_now),
        "burst_mult": float(burst_mult),
        "max_p": float(max_p),
    }
    if tx_asof.empty:
        return [], {}, params

    df = tx_asof[tx_asof.t <= t_open]
    if df.empty:
        return [], {}, params

    f_rows = df[df.TransactionID == int(flagged_txn_id)]
    if f_rows.empty:
        return [], {}, params

    f_row = f_rows.iloc[0]
    flagged_card = str(f_row.card_id)
    dom = f_row.R_emaildomain
    f_online_c = (
        str(f_row.channel) == "online"
        and str(f_row.ProductCD) == "C"
        and float(f_row.amount) >= min_amount
    )
    high_rate_freemail = freemail_domains_by_base_rate(df)
    if not f_online_c or is_freemail(dom, high_rate_freemail):
        params["flagged_qualifies"] = False
        return [], {}, params

    val = str(dom).strip()
    t_win_start = t_open - pd.Timedelta(days=window_days)

    # All charges to this R_emaildomain (online, ProductCD=='C') up to t_open
    dom_all = df[
        (df.channel == "online")
        & (df.ProductCD == "C")
        & (df.R_emaildomain == val)
    ]
    if dom_all.empty:
        return [], {}, params

    # First-ever charge to this R_emaildomain per card, and qualified (>= min_amount) charges in window
    first_any = dom_all.groupby("card_id", sort=False)["t"].min()
    dom_qual = dom_all[dom_all.amount >= min_amount]
    first_qual = dom_qual.groupby("card_id", sort=False)["t"].min()

    # Cards whose first charge to `val` falls in [t_win_start, t_open] AND that have >= min_amount in window
    first_in_win_cards = set(first_any[(first_any >= t_win_start) & (first_any <= t_open)].index.astype(str))
    qual_in_win_cards = set(
        dom_qual[(dom_qual.t >= t_win_start) & (dom_qual.t <= t_open)].card_id.astype(str).unique()
    )
    cohort_set = first_in_win_cards & qual_in_win_cards
    if flagged_card not in cohort_set:
        params["flagged_card_in_first_time_cohort"] = False
        return [], {}, params

    member_cards = sorted(cohort_set)
    k_now = len(member_cards)

    # Causal 21-day baseline windows before t_win_start (within Jul-Dec) using first_qual (>= min_amount)
    t_min_bank = pd.Timestamp(df.t.iloc[0])
    base_cutoff = min(t_win_start, pd.Timestamp("2017-01-01 00:00:00"))
    win_delta = pd.Timedelta(days=window_days)

    b_counts = []
    cur_end = base_cutoff
    while cur_end >= t_min_bank + pd.Timedelta(days=7):
        cur_start = cur_end - win_delta
        c = int(((first_qual >= cur_start) & (first_qual < cur_end)).sum())
        b_counts.append(c)
        cur_end -= win_delta
    if not b_counts:
        b_counts = [0]

    base_mean = float(np.mean(b_counts))
    base_max = int(np.max(b_counts))
    p_val = poisson_tail_p(k_now, max(base_mean, 0.05))

    burst_ok = (
        k_now >= min_k_now
        and k_now >= burst_mult * max(1, base_max)
        and p_val < max_p
    )
    params.update(
        {
            "R_emaildomain": val,
            "k_now": k_now,
            "baseline_mean": round(base_mean, 4),
            "baseline_max": base_max,
            "p_value": float(f"{p_val:.4e}"),
            "fired": bool(burst_ok),
        }
    )
    if not burst_ok:
        return [], {}, params

    win_qual = dom_qual[(dom_qual.t >= t_win_start) & (dom_qual.t <= t_open) & (dom_qual.card_id.isin(member_cards))]
    per_card_txns: dict[str, list[int]] = {}
    for cid, grp in win_qual.groupby("card_id", sort=False):
        per_card_txns[str(cid)] = sorted(int(x) for x in grp.TransactionID.unique())

    return member_cards, per_card_txns, params


def rerun_recipient_fan_in_burst(
    tx: pd.DataFrame,
    card_id: str,
    flagged_txn_id: int,
    opened_at: str,
    window_days: int = 21,
    min_amount: float = 100.0,
    episode_hours: float = 72.0,
) -> dict:
    """Registered re-runnable query function for FanIn attestations."""
    det = FanIn(window_days=window_days, min_amount=min_amount, episode_hours=episode_hours)
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
        min_amount: float = 100.0,
        min_k_now: int = 5,
        burst_mult: float = 3.0,
        max_p: float = 1e-4,
        episode_hours: float = 72.0,
        enabled: bool = True,
    ):
        self.window_days = window_days
        self.min_amount = min_amount
        self.min_k_now = min_k_now
        self.burst_mult = burst_mult
        self.max_p = max_p
        self.episode_hours = episode_hours
        self.enabled = enabled
        # Per LR_RULE: no unmeasured lr=3.0 assumption. Measured from backtest; 1.0 when 0 historical fires.
        self.lr: float = 1.0
        self._measured_lr: float = 1.0
        self._lr_ci: tuple[float, float] = (0.07, 17.12)
        self._p_fraud: float = 0.0
        self._p_legit: float = 0.0
        self._n_hist_fraud_bursts: int = 0
        self._n_hist_legit_bursts: int = 0
        self.lr_proposal: dict = {
            "proposed_lr": None,
            "assumption": (
                "No unmeasured LR is applied (removed prior lr=3.0 assumption). Recipient fan-in burst "
                "(>=5 first-time cards in 21d with amount>=$100, >=3x baseline max, p<1e-4) fired 0 times "
                "on Jul-Dec closed fraud cases and 0 times on 5,000 never-cased legitimate card-days; "
                "leaving lr=1.0 and attaching connected_card_ids + attested cohort ids for Policy §3a / R6."
            ),
        }
        self.backtest_stats: dict = {}

    def inspect_case(self, ctx: CaseCtx) -> dict:
        """Evaluates non-freemail R_emaildomain burst on ctx.card_id using `cohort_members`.
        - Primary finding (`fired=True, kind='evidence'`): when the flagged txn ITSELF is an
          online ProductCD=='C' charge with amount >= min_amount to the bursting non-freemail
          R_emaildomain and ctx.card_id is in the 21d first-time cohort.
        - Context note (`fired=False, kind='context'`): when ctx.card_id has other online
          ProductCD=='C' charges (>= min_amount) in the 21d window to a bursting non-freemail
          R_emaildomain, but the flagged txn itself does not pay that recipient.
        """
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
            "cohort_txn_ids": [],
            "card_val_txns": [],
            "card_val_amounts": [],
            "all_win_custs": [],
            "per_card_txns": {},
        }
        if ctx.tx.empty:
            return default_info

        t_open = pd.to_datetime(ctx.opened_at)
        t_arr = ctx.tx.t.values
        i_end = np.searchsorted(t_arr, np.datetime64(t_open), side="right")
        tx_asof = ctx.tx.iloc[:i_end]
        if tx_asof.empty:
            return default_info

        # 1. Check primary cohort via pure function `cohort_members` (requires flagged txn itself to pay recipient)
        members, per_card_txns, params = cohort_members(
            tx_asof,
            flagged_txn_id=int(ctx.flagged_txn_id),
            opened_at=t_open,
            window_days=self.window_days,
            min_amount=self.min_amount,
            min_k_now=self.min_k_now,
            burst_mult=self.burst_mult,
            max_p=self.max_p,
        )
        if members:
            val = params["R_emaildomain"]
            conn_cards = [c for c in members if c != str(ctx.card_id)]
            own_txns = per_card_txns.get(str(ctx.card_id), [])
            all_cohort_txns = sorted({tid for tids in per_card_txns.values() for tid in tids})
            card_val_df = tx_asof[tx_asof.TransactionID.isin(own_txns)]
            card_val_amts = [round(float(x), 2) for x in card_val_df.amount.tolist()]
            all_win_custs = sorted({c.rsplit("-K", 1)[0] for c in members})
            return {
                "fired": True,
                "kind": "evidence",
                "linking_value": f"R_emaildomain={val}",
                "val": val,
                "k_now": int(params["k_now"]),
                "baseline_mean": float(params["baseline_mean"]),
                "baseline_max": int(params["baseline_max"]),
                "p_value": float(params["p_value"]),
                "connected_cards": conn_cards,
                "all_burst_cards": members,
                "affected_txns": own_txns,
                "cohort_txn_ids": all_cohort_txns,
                "card_val_txns": own_txns,
                "card_val_amounts": card_val_amts,
                "all_win_custs": all_win_custs,
                "per_card_txns": per_card_txns,
            }

        # 2. Check if the card has prior/other online C charges (>= min_amount) in the 21d window to a bursting
        # recipient even though the flagged txn itself does not pay it (e.g. HHF-013 where flagged txn is W).
        t_win_start = t_open - pd.Timedelta(days=self.window_days)
        i_start = np.searchsorted(tx_asof.t.values, np.datetime64(t_win_start), side="left")
        tx_win = tx_asof.iloc[i_start:]
        card_win = tx_win[
            (tx_win.card_id == ctx.card_id)
            & (tx_win.channel == "online")
            & (tx_win.ProductCD == "C")
            & (tx_win.amount >= self.min_amount)
        ]
        if card_win.empty:
            return default_info

        high_rate_freemail = freemail_domains_by_base_rate(tx_asof)
        candidates = [
            str(dom).strip()
            for dom in card_win.R_emaildomain.dropna().unique()
            if not is_freemail(dom, high_rate_freemail)
        ]
        if not candidates:
            return default_info

        t_min_bank = pd.to_datetime(tx_asof.t.iloc[0])
        base_cutoff = min(t_win_start, pd.Timestamp("2017-01-01 00:00:00"))
        win_delta = pd.Timedelta(days=self.window_days)

        best_ctx = default_info
        for val in candidates:
            dom_all = tx_asof[
                (tx_asof.channel == "online")
                & (tx_asof.ProductCD == "C")
                & (tx_asof.R_emaildomain == val)
            ]
            if dom_all.empty:
                continue
            first_any = dom_all.groupby("card_id", sort=False)["t"].min()
            dom_qual = dom_all[dom_all.amount >= self.min_amount]
            first_qual = dom_qual.groupby("card_id", sort=False)["t"].min()

            first_in_win = set(first_any[(first_any >= t_win_start) & (first_any <= t_open)].index.astype(str))
            qual_in_win = set(
                dom_qual[(dom_qual.t >= t_win_start) & (dom_qual.t <= t_open)].card_id.astype(str).unique()
            )
            burst_cards = sorted(first_in_win & qual_in_win)
            k_now = len(burst_cards)

            b_counts = []
            cur_end = base_cutoff
            while cur_end >= t_min_bank + pd.Timedelta(days=7):
                cur_start = cur_end - win_delta
                c = int(((first_qual >= cur_start) & (first_qual < cur_end)).sum())
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
            if burst_ok and k_now > best_ctx["k_now"]:
                card_val_df = dom_qual[dom_qual.card_id == ctx.card_id]
                card_val_txns = sorted(int(x) for x in card_val_df.TransactionID.unique())
                card_val_amts = [round(float(x), 2) for x in card_val_df.amount.tolist()]
                conn_cards = [c for c in burst_cards if c != str(ctx.card_id)]
                best_ctx = {
                    "fired": False,
                    "kind": "context",
                    "linking_value": f"R_emaildomain={val}",
                    "val": val,
                    "k_now": k_now,
                    "baseline_mean": base_mean,
                    "baseline_max": base_max,
                    "p_value": p_val,
                    "connected_cards": conn_cards,
                    "all_burst_cards": burst_cards,
                    "affected_txns": [],
                    "cohort_txn_ids": card_val_txns,
                    "card_val_txns": card_val_txns,
                    "card_val_amounts": card_val_amts,
                    "all_win_custs": sorted({c.rsplit("-K", 1)[0] for c in burst_cards}),
                    "per_card_txns": {},
                }
        return best_ctx

    def run(self, ctx: CaseCtx, force: bool = False) -> list[Finding]:
        if not self.enabled and not force:
            return []

        info = self.inspect_case(ctx)
        if info["kind"] == "none":
            return []

        if info["kind"] == "context":
            claim = Attestation(
                what=(
                    f"CONTEXT NOTE (flagged txn {ctx.flagged_txn_id} does not pay recipient): Card {ctx.card_id} "
                    f"has other online ProductCD=='C' charges >= ${self.min_amount:.0f} "
                    f"(txns={info['card_val_txns']}, amounts={info['card_val_amounts']}) to bursting recipient "
                    f"{info['linking_value']} ({info['k_now']} first-time cards in {self.window_days}d: "
                    f"{info['all_burst_cards']}, Jul-Dec baseline mean={info['baseline_mean']:.2f}, "
                    f"max={info['baseline_max']}, Poisson p={info['p_value']:.2e})."
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
                    "min_amount": self.min_amount,
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
            f_ctx.lr_proposal = self.lr_proposal  # type: ignore[attr-defined]
            return [f_ctx]

        claim = Attestation(
            what=(
                f"Recipient fan-in cohort on {info['linking_value']} (online ProductCD=='C', amount >= ${self.min_amount:.0f}): "
                f"{info['k_now']} distinct cards used it for the first time in {self.window_days}d before {ctx.opened_at} "
                f"(Jul-Dec baseline mean={info['baseline_mean']:.2f}, max={info['baseline_max']}, "
                f"Poisson p={info['p_value']:.2e}); flagged txn {ctx.flagged_txn_id} on {ctx.card_id} pays this recipient "
                f"({info['card_val_amounts']}); measured lr={self.lr:.2f} (cohort attached for Policy §3a / R6)."
            ),
            evidence=Evidence(
                txn_ids=info["cohort_txn_ids"],
                card_ids=info["all_burst_cards"],
                customer_ids=info["all_win_custs"],
                case_ids=[ctx.case_id],
            ),
            query="rerun_recipient_fan_in_burst",
            params={
                "kind": "evidence",
                "card_id": str(ctx.card_id),
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "opened_at": str(ctx.opened_at),
                "linking_value": info["linking_value"],
                "window_days": self.window_days,
                "min_amount": self.min_amount,
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
            lr=self.lr,
            affected_txn_ids=info["affected_txns"],
            connected_card_ids=info["connected_cards"],
        )
        f_ev.kind = "evidence"  # type: ignore[attr-defined]
        f_ev.lr_proposal = self.lr_proposal  # type: ignore[attr-defined]
        f_ev.per_card_txn_ids = info["per_card_txns"]  # type: ignore[attr-defined]
        return [f_ev]

    def backtest(
        self, cc: pd.DataFrame, tx: pd.DataFrame, sample_size: int = 5000, seed: int = 42
    ) -> Backtest:
        """Vectorised as-of replay over Jul-Dec closed cases (confirmed_fraud and cleared) and
        5,000 never-cased legitimate card-days (sampled with fixed seed=42), plus a scan of
        historical 21d windows across Jul-Dec.
        """
        tx_sorted = tx.sort_values("t", kind="mergesort") if not tx.t.is_monotonic_increasing else tx
        high_rate_freemail = freemail_domains_by_base_rate(tx_sorted[tx_sorted.t < pd.Timestamp("2017-01-01")])

        valid_mask = (
            (tx_sorted.channel == "online")
            & (tx_sorted.ProductCD == "C")
            & (tx_sorted.amount >= self.min_amount)
            & tx_sorted.R_emaildomain.notna()
            & (~tx_sorted.R_emaildomain.map(lambda d: is_freemail(d, high_rate_freemail)))
        )
        sub = tx_sorted.loc[valid_mask, ["R_emaildomain", "card_id", "t", "TransactionID", "amount"]]
        first_df = sub.groupby(["R_emaildomain", "card_id"], sort=False)["t"].min().reset_index()

        val_times: dict[str, np.ndarray] = {}
        for val, grp in first_df.groupby("R_emaildomain", sort=False):
            val_times[str(val)] = np.sort(grp["t"].values)

        t_min_bank = np.datetime64(tx_sorted.t.iloc[0])
        win_td = np.timedelta64(self.window_days, "D")
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

        # Map qualified TransactionID -> (card_id, R_emaildomain, t_np) and card_first_use
        qual_txn_map: dict[int, tuple[str, str, np.datetime64]] = {}
        for row in sub.itertuples():
            qual_txn_map[int(row.TransactionID)] = (str(row.card_id), str(row.R_emaildomain), np.datetime64(row.t))

        card_dom_first: dict[tuple[str, str], np.datetime64] = {
            (str(r.card_id), str(r.R_emaildomain)): np.datetime64(r.t)
            for r in first_df.itertuples()
        }

        def check_flagged_fires(card_id: str, flagged_id: int, t_open_np: np.datetime64) -> bool:
            hit = qual_txn_map.get(flagged_id)
            if hit is None:
                return False
            f_cid, f_dom, f_t = hit
            if f_cid != card_id or f_t > t_open_np:
                return False
            first_t = card_dom_first.get((card_id, f_dom))
            t_start_np = t_open_np - win_td
            if first_t is None or not (t_start_np <= first_t <= t_open_np):
                return False
            return eval_val_burst(f_dom, t_open_np)

        min_eval_t = pd.Timestamp(t_min_bank) + pd.Timedelta(days=90)
        jan1_t = pd.Timestamp("2017-01-01 00:00:00")
        cc_eval = cc[
            (pd.to_datetime(cc.opened_at) >= min_eval_t)
            & (pd.to_datetime(cc.opened_at) < jan1_t)
        ]
        fraud_cases = cc_eval[cc_eval.outcome == "confirmed_fraud"]
        cleared_cases = cc_eval[cc_eval.outcome == "cleared"]

        caught: list[str] = []
        missed: list[str] = []
        for row in fraud_cases.itertuples():
            flagged_id = (
                int(row.first_fraud_txn_id)
                if pd.notna(row.first_fraud_txn_id)
                else (int(row.txn_list[0]) if row.txn_list else -1)
            )
            t_open_np = np.datetime64(pd.to_datetime(row.opened_at))
            if check_flagged_fires(str(row.card_id), flagged_id, t_open_np):
                caught.append(str(row.case_id))
            else:
                missed.append(str(row.case_id))

        cleared_acted: list[str] = []
        for row in cleared_cases.itertuples():
            flagged_id = int(row.txn_list[0]) if row.txn_list else -1
            t_open_np = np.datetime64(pd.to_datetime(row.opened_at))
            if check_flagged_fires(str(row.card_id), flagged_id, t_open_np):
                cleared_acted.append(str(row.case_id))

        # Legitimate card-days: cards NEVER named in a closed case, sampled with fixed seed
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
        for row in legit_sample.itertuples():
            t_open_np = np.datetime64(pd.to_datetime(row.t))
            if check_flagged_fires(str(row.card_id), int(row.TransactionID), t_open_np):
                legit_card_day_hits.append(str(row.TransactionID))

        # Scan daily 21d windows across [min_eval_t, 2017-01-01) across all candidate domains
        scan_days = pd.date_range(min_eval_t, jan1_t - pd.Timedelta(days=1), freq="1D")
        n_windows_scanned = len(scan_days)
        windows_firing_jul_dec = 0
        for d_ts in scan_days:
            d_np = np.datetime64(d_ts)
            if any(eval_val_burst(dom, d_np) for dom in val_times):
                windows_firing_jul_dec += 1

        n_fraud = len(fraud_cases)
        n_legit = len(legit_sample)
        c_f = len(caught)
        c_l = len(legit_card_day_hits)

        self._n_hist_fraud_bursts = c_f
        self._n_hist_legit_bursts = c_l
        self._p_fraud = c_f / n_fraud if n_fraud else 0.0
        self._p_legit = c_l / n_legit if n_legit else 0.0

        p1 = (c_f + 1.0) / (n_fraud + 2.0)
        p0 = (c_l + 1.0) / (n_legit + 2.0)
        lr_empirical = float(p1 / p0)

        se_log_lr = math.sqrt((1.0 / (c_f + 1.0)) + (1.0 / (c_l + 1.0)))
        ci_low = float(lr_empirical * math.exp(-1.96 * se_log_lr))
        ci_high = float(lr_empirical * math.exp(1.96 * se_log_lr))
        self._lr_ci = (ci_low, ci_high)

        # Per LR_RULE: never fall back to lr=3.0. Use CI lower bound if c_f > 0 and ci_low > 1.0, else 1.0.
        if c_f > 0 and ci_low > 1.0:
            self.lr = round(ci_low, 4)
        else:
            self.lr = 1.0
        self._measured_lr = self.lr

        self.backtest_stats = {
            "n_fraud_cases": n_fraud,
            "n_cleared_cases": len(cleared_cases),
            "n_legit_card_days": n_legit,
            "fraud_caught": c_f,
            "cleared_hit": len(cleared_acted),
            "legit_card_day_hit": c_l,
            "lr_laplace": round(lr_empirical, 4),
            "lr_ci_95": (round(ci_low, 4), round(ci_high, 4)),
            "lr_used": self.lr,
            "n_windows_scanned_jul_dec": n_windows_scanned,
            "windows_firing_jul_dec": windows_firing_jul_dec,
        }

        return Backtest(
            detector=self.name,
            n_cases=n_fraud + len(cleared_cases),
            caught=caught,
            missed=missed,
            legit_acted_against=cleared_acted,
            lr_measured=lr_empirical,
        )
