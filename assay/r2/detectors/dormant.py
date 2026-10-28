"""BLOCK B5 -- Dormant card wake-up detector (challenge 1).

Pattern: a card silent for >= N days (N chosen by backtest from {14,21,30,45,60})
then a burst (>= k txns or amount >= m x its own historical median within 48h).
Build class DormantWake(Detector) in assay/r2/detectors/dormant.py,
pattern "undocumented:dormant_card_wake".

Silence is measured strictly on card_id using prior transaction history on that card.
Backtest uses 5,000 sampled legitimate card-days from Jul-Oct 2016 for unbiased LR.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np

from assay.r2.finding import Detector, Finding, Backtest, CaseCtx
from assay.r2.attest import Attestation, Evidence


N_SWEEP = [14, 21, 30, 45, 60]
K_SWEEP = [2, 3, 4, 5]
M_SWEEP = [2.0, 3.0, 5.0, 10.0]


class DormantWake(Detector):
    name: str = "dormant_wake"
    pattern: str = "undocumented:dormant_card_wake"

    def __init__(self, N: int = 14, k: int = 2, m: float = 5.0, lr_measured: float = 5.653):
        self.N = int(N)
        self.k = int(k)
        self.m = float(m)
        self.lr_measured = float(lr_measured)

    def run(self, ctx: CaseCtx) -> list[Finding]:
        """Run dormant wake detection on a single case context as-of opened_at."""
        tx_hist = ctx.tx[ctx.tx["t"] <= ctx.opened_at]
        if tx_hist.empty:
            return []

        # Filter strictly for the case card_id
        card_tx = tx_hist[tx_hist["card_id"] == ctx.card_id].sort_values("t")
        if card_tx.empty:
            return []

        # Find recent transactions within 14 days before opened_at
        cutoff = ctx.opened_at - pd.Timedelta(days=14)
        recent_tx = card_tx[card_tx["t"] >= cutoff]
        if recent_tx.empty:
            recent_tx = card_tx[card_tx["TransactionID"] == ctx.flagged_txn_id]
        if recent_tx.empty:
            return []

        findings = []

        for idx, row in recent_tx.iterrows():
            t0 = row["t"]

            # Calculate silence gap on CARD ONLY using prior history before t0
            card_hist = card_tx[card_tx["t"] < t0]
            if card_hist.empty:
                continue

            silence_days = (t0 - card_hist["t"].max()).total_seconds() / 86400.0

            if silence_days < self.N:
                continue

            # 48h burst starting at t0 for card_id
            burst_window = card_tx[(card_tx["t"] >= t0) & (card_tx["t"] <= t0 + pd.Timedelta(hours=48))]
            burst_k = len(burst_window)

            hist_median = float(card_hist["amount"].median())
            if hist_median <= 0:
                hist_median = 1.0

            max_burst_amt = float(burst_window["amount"].max())
            burst_m = max_burst_amt / hist_median

            if burst_k >= self.k or burst_m >= self.m:
                affected_ids = [int(x) for x in burst_window["TransactionID"].tolist()]
                total_amt = float(burst_window["amount"].sum())

                att = Attestation(
                    what=(f"Card {ctx.card_id} woke up after {silence_days:.1f}d card silence "
                          f"with burst of {burst_k} txns (max ${max_burst_amt:.2f}, total ${total_amt:.2f}) in 48h."),
                    evidence=Evidence(
                        txn_ids=affected_ids,
                        card_ids=[ctx.card_id],
                        customer_ids=[ctx.customer_id]
                    ),
                    query="dormant_wake",
                    params={
                        "card_id": ctx.card_id,
                        "as_of": str(ctx.opened_at),
                        "N": self.N,
                        "k": self.k,
                        "m": self.m
                    }
                )

                findings.append(Finding(
                    detector=self.name,
                    pattern=self.pattern,
                    claim=att,
                    lr=self.lr_measured,
                    affected_txn_ids=affected_ids,
                    connected_card_ids=[]
                ))
                break  # Fire once per case context

        return findings

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        """Backtest over closed cases as-of opened_at."""
        sweep_res = sweep_backtest(cc, tx, N_grid=[self.N], k_grid=[self.k], m_grid=[self.m])
        res = sweep_res[0]
        return res["backtest"]


def rerun(params: dict) -> list[int]:
    """Reproduce exact evidence transaction IDs from attestation parameters."""
    from assay.r2.frames import tx_all
    tx = tx_all()
    card_id = params["card_id"]
    as_of = pd.to_datetime(params["as_of"])
    N = int(params.get("N", 14))
    k = int(params.get("k", 2))
    m = float(params.get("m", 5.0))

    tx_hist = tx[tx["t"] <= as_of]
    card_tx = tx_hist[tx_hist["card_id"] == card_id].sort_values("t")
    if card_tx.empty:
        return []

    cutoff = as_of - pd.Timedelta(days=14)
    recent_tx = card_tx[card_tx["t"] >= cutoff]
    if recent_tx.empty:
        return []

    for idx, row in recent_tx.iterrows():
        t0 = row["t"]
        card_hist = card_tx[card_tx["t"] < t0]
        if card_hist.empty:
            continue
        silence_days = (t0 - card_hist["t"].max()).total_seconds() / 86400.0
        if silence_days < N:
            continue
        burst_window = card_tx[(card_tx["t"] >= t0) & (card_tx["t"] <= t0 + pd.Timedelta(hours=48))]
        burst_k = len(burst_window)
        hist_median = float(card_hist["amount"].median())
        if hist_median <= 0:
            hist_median = 1.0
        burst_m = float(burst_window["amount"].max()) / hist_median
        if burst_k >= k or burst_m >= m:
            return [int(x) for x in burst_window["TransactionID"].tolist()]
    return []


def bootstrap_lr_ci(
    n_fraud: int,
    caught: int,
    n_legit: int,
    legit_fired: int,
    n_boot: int = 1000,
    seed: int = 42
) -> tuple[float, float, float]:
    """Compute 95% bootstrap confidence interval for Laplace-smoothed LR."""
    rng = np.random.default_rng(seed)
    f_arr = np.zeros(n_fraud, dtype=int)
    f_arr[:caught] = 1
    l_arr = np.zeros(n_legit, dtype=int)
    l_arr[:legit_fired] = 1

    boot_lrs = []
    for _ in range(n_boot):
        f_s = rng.choice(f_arr, size=n_fraud, replace=True)
        l_s = rng.choice(l_arr, size=n_legit, replace=True)
        p_f = (f_s.sum() + 1.0) / (n_fraud + 2.0)
        p_l = (l_s.sum() + 1.0) / (n_legit + 2.0)
        boot_lrs.append(p_f / p_l)

    p_f0 = (caught + 1.0) / (n_fraud + 2.0)
    p_l0 = (legit_fired + 1.0) / (n_legit + 2.0)
    lr0 = p_f0 / p_l0

    ci_low = float(np.percentile(boot_lrs, 2.5))
    ci_high = float(np.percentile(boot_lrs, 97.5))
    return round(lr0, 3), round(ci_low, 3), round(ci_high, 3)


def sweep_backtest(
    cc: pd.DataFrame,
    tx: pd.DataFrame,
    N_grid: list[int] = N_SWEEP,
    k_grid: list[int] = K_SWEEP,
    m_grid: list[float] = M_SWEEP,
    seed: int = 42
) -> list[dict]:
    """Sweeps over N, k, m grid using card_id silence and unbiased legit sampling."""
    df = tx[["TransactionID", "t", "card_id", "amount"]].copy().sort_values("t")

    # Card silence calculation
    df["prev_t"] = df.groupby("card_id")["t"].shift(1)
    df["silence_days"] = (df["t"] - df["prev_t"]).dt.total_seconds() / 86400.0

    min_N = min(N_grid)
    cand_df = df[df["silence_days"] >= min_N].copy()

    card_groups = df.groupby("card_id")
    cands_info = []

    for row in cand_df.itertuples():
        card_df = card_groups.get_group(row.card_id)
        t0 = row.t

        card_hist = card_df[card_df["t"] < t0]
        if card_hist.empty:
            continue

        burst = card_df[(card_df["t"] >= t0) & (card_df["t"] <= t0 + pd.Timedelta(hours=48))]
        b_k = len(burst)

        hist_med = float(card_hist["amount"].median())
        if hist_med <= 0:
            hist_med = 1.0
        b_m = float(burst["amount"].max()) / hist_med

        cands_info.append({
            "TransactionID": row.TransactionID,
            "card_id": row.card_id,
            "t": t0,
            "silence_days": row.silence_days,
            "burst_k": b_k,
            "burst_m": b_m,
        })

    cand_feat_df = pd.DataFrame(cands_info) if cands_info else pd.DataFrame(columns=["TransactionID", "card_id", "t", "silence_days", "burst_k", "burst_m"])

    # Prepare legitimate sampling from Jul-Oct 2016
    cc_df = cc.copy()
    cc_df["opened_at_dt"] = pd.to_datetime(cc_df["opened_at"])

    all_fraud_txns = set()
    for t_list in cc_df["txn_list"]:
        if isinstance(t_list, list):
            all_fraud_txns.update(t_list)

    legit_jul_oct = tx[(tx["t"] < "2016-11-01") & (~tx["TransactionID"].isin(all_fraud_txns))]
    n_legit_sample = min(5000, len(legit_jul_oct))
    sampled_legit = legit_jul_oct.sample(n=n_legit_sample, random_state=seed)
    sampled_legit_txids = set(sampled_legit["TransactionID"].tolist())

    cc_fraud = cc_df[cc_df["outcome"] == "confirmed_fraud"]
    cc_cleared = cc_df[cc_df["outcome"] == "cleared"]

    results = []

    for N in N_grid:
        for k in k_grid:
            for m in m_grid:
                if cand_feat_df.empty:
                    trig_df = pd.DataFrame()
                else:
                    cond_s = cand_feat_df["silence_days"] >= N
                    cond_b = (cand_feat_df["burst_k"] >= k) | (cand_feat_df["burst_m"] >= m)
                    trig_df = cand_feat_df[cond_s & cond_b]

                firing_txids = set(trig_df["TransactionID"].tolist()) if not trig_df.empty else set()
                trig_by_card = trig_df.groupby("card_id") if not trig_df.empty else None

                # Measure legit fire rate on sampled legit txns
                legit_sample_fired = len(sampled_legit_txids.intersection(firing_txids))
                legit_rate = legit_sample_fired / float(n_legit_sample)

                # Evaluate closed fraud cases
                caught_cases = []
                missed_cases = []

                for c in cc_fraud.itertuples():
                    card = c.card_id
                    opened_at = c.opened_at_dt
                    fired = False

                    if trig_by_card is not None and card in trig_by_card.groups:
                        c_trigs = trig_by_card.get_group(card)
                        in_win = c_trigs[(c_trigs["t"] <= opened_at) & (c_trigs["t"] >= opened_at - pd.Timedelta(days=14))]
                        if not in_win.empty:
                            fired = True
                        elif hasattr(c, "txn_list") and isinstance(c.txn_list, list):
                            if not c_trigs[c_trigs["TransactionID"].isin(c.txn_list)].empty:
                                fired = True

                    if fired:
                        caught_cases.append(c.case_id)
                    else:
                        missed_cases.append(c.case_id)

                # Evaluate cleared cases in cc for Backtest object
                legit_acted = []
                for c in cc_cleared.itertuples():
                    card = c.card_id
                    opened_at = c.opened_at_dt
                    fired = False

                    if trig_by_card is not None and card in trig_by_card.groups:
                        c_trigs = trig_by_card.get_group(card)
                        in_win = c_trigs[(c_trigs["t"] <= opened_at) & (c_trigs["t"] >= opened_at - pd.Timedelta(days=14))]
                        if not in_win.empty:
                            fired = True
                        elif hasattr(c, "txn_list") and isinstance(c.txn_list, list):
                            if not c_trigs[c_trigs["TransactionID"].isin(c.txn_list)].empty:
                                fired = True

                    if fired:
                        legit_acted.append(c.case_id)

                fraud_rate = len(caught_cases) / float(len(cc_fraud))

                # Unbiased Laplace-smoothed Likelihood Ratio & 95% Bootstrap CI
                lr0, ci_low, ci_high = bootstrap_lr_ci(
                    n_fraud=len(cc_fraud),
                    caught=len(caught_cases),
                    n_legit=n_legit_sample,
                    legit_fired=legit_sample_fired,
                    seed=seed
                )

                score = len(caught_cases) - 3 * legit_sample_fired

                bt = Backtest(
                    detector="dormant_wake",
                    n_cases=len(cc_df),
                    caught=caught_cases,
                    missed=missed_cases,
                    legit_acted_against=legit_acted,
                    lr_measured=round(lr0, 4)
                )

                results.append({
                    "N": N,
                    "k": k,
                    "m": m,
                    "caught": len(caught_cases),
                    "fraud_rate": fraud_rate,
                    "legit_sample_fired": legit_sample_fired,
                    "legit_rate": legit_rate,
                    "score": score,
                    "lr_measured": round(lr0, 4),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "backtest": bt
                })

    return results
