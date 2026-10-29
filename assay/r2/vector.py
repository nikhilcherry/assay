"""B14 -- TigerVector: behaviour embeddings per closed case and exam case,
fused with 2-hop graph traversal over shared DeviceProfile / BillingRegion /
EmailDomain.

Every embedding is 24-dimensional, deterministic, strictly causal (as-of each
case's opened_at), and z-standardized against historical cases closed before
the evaluation window.

NO LEAKAGE GUARANTEE:
Embeddings are computed ONLY from transactions and identity rows on the card
with `t <= opened_at` (a 24-hour window ending at the trigger/latest transaction
on or before `opened_at`, compared against prior card history strictly before
that window). No post-investigation fields (`outcome`, `actions_taken`,
`exposure_usd`, `connected_card_ids`, `txn_ids`, `first_fraud_txn_id`, `n_txns`,
`analyst_notes`, `closed_at`, `pattern`, `report_filed`) are ever read when
constructing an embedding or defining the episode window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from assay.data import DATA
from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding
from assay.r2.frames import case_pack, closed_cases, tx_all

EMBED_DIM = 24

# Closed-case columns filled AFTER a case opens; forbidden as embedding inputs
FORBIDDEN_CASE_COLS: frozenset[str] = frozenset({
    "outcome",
    "actions_taken",
    "exposure_usd",
    "connected_card_ids",
    "connected_list",
    "txn_ids",
    "txn_list",
    "first_fraud_txn_id",
    "n_txns",
    "analyst_notes",
    "closed_at",
    "pattern",
    "report_filed",
})

# Exact mapping of each of the 24 features -> source column(s) in tx_all()
FEATURE_SPEC: tuple[tuple[str, str], ...] = (
    # Episode amount statistics (4)
    ("ep_log_amt_mean", "tx_all.amount (24h episode window t <= opened_at)"),
    ("ep_log_amt_max", "tx_all.amount (24h episode window t <= opened_at)"),
    ("ep_log_amt_std", "tx_all.amount (24h episode window t <= opened_at)"),
    ("ep_log_amt_sum", "tx_all.amount (24h episode window t <= opened_at)"),
    # Episode vs card's own historical baseline before episode (4)
    ("vs_card_amt_ratio", "tx_all.amount (episode vs card prior t < window_start)"),
    ("vs_card_amt_z", "tx_all.amount (episode vs card prior t < window_start)"),
    ("card_log_hist_cnt", "tx_all.TransactionID (count of card prior txns t < window_start)"),
    ("card_prior_online_rate", "tx_all.channel (online rate on card prior txns t < window_start)"),
    # Inter-txn timing & burst size in episode window (4)
    ("ep_log_win_cnt", "tx_all.TransactionID (count of card txns in 24h window t <= opened_at)"),
    ("ep_log_span_sec", "tx_all.TransactionDT (time span in 24h window t <= opened_at)"),
    ("ep_log_min_gap_sec", "tx_all.TransactionDT (min inter-txn gap in 24h window t <= opened_at)"),
    ("ep_small_auth_frac", "tx_all.amount (fraction < $25 in 24h window t <= opened_at)"),
    # Channel & ProductCD mix in episode window (6)
    ("ep_online_frac", "tx_all.channel (fraction == 'online' in 24h window t <= opened_at)"),
    ("prod_W_frac", "tx_all.ProductCD (fraction == 'W' in 24h window t <= opened_at)"),
    ("prod_C_frac", "tx_all.ProductCD (fraction == 'C' in 24h window t <= opened_at)"),
    ("prod_R_frac", "tx_all.ProductCD (fraction == 'R' in 24h window t <= opened_at)"),
    ("prod_H_frac", "tx_all.ProductCD (fraction == 'H' in 24h window t <= opened_at)"),
    ("prod_S_frac", "tx_all.ProductCD (fraction == 'S' in 24h window t <= opened_at)"),
    # Device, region, email novelty + hour-of-day + D1 + as-of shared-device card count (6)
    ("device_present_frac", "tx_all.device_profile (non-empty rate in 24h window t <= opened_at)"),
    ("device_novel_frac", "tx_all.device_profile (unseen on card prior to window_start)"),
    ("region_novel_frac", "tx_all.addr1 (billing region unseen on card prior to window_start)"),
    ("hour_sin", "tx_all.TransactionDT (sin of UTC hour-of-day in 24h window t <= opened_at)"),
    ("card_age_log_d1", "tx_all.D1 (card age days at episode t <= opened_at)"),
    ("asof_shared_dev_cards_log", "tx_all.device_profile + tx_all.card_id (other cards on device with t <= opened_at)"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURE_SPEC)
FEATURE_SOURCES: dict[str, str] = dict(FEATURE_SPEC)
assert len(FEATURE_NAMES) == EMBED_DIM

TG_R2_DIR = DATA / "tg_r2"


def format_feature_table() -> str:
    """Return a printable table of the 24 embedding features and their exact source columns."""
    lines = [f"{'#':>2}  {'Feature Name':<28}  {'Source Column(s) (as-of t <= opened_at)'}"]
    lines.append("-" * 92)
    for idx, (fname, src) in enumerate(FEATURE_SPEC):
        lines.append(f"{idx:>2}  {fname:<28}  {src}")
    return "\n".join(lines)


def _to_epoch_sec(series_or_ts) -> np.ndarray | int:
    """Convert any datetime Series, Index, or Timestamp to integer UTC epoch seconds.
    Immune to pandas 2.x/3.x datetime64 resolution differences ([s], [ms], [us], [ns]).
    """
    if isinstance(series_or_ts, (pd.Timestamp, str)):
        ts = pd.Timestamp(series_or_ts)
        return int(ts.timestamp())
    dt = pd.to_datetime(series_or_ts)
    # Convert via datetime64[s] -> int64 epoch seconds
    if isinstance(dt, pd.Series):
        return dt.to_numpy(dtype="datetime64[s]").astype(np.int64)
    return np.asarray(dt, dtype="datetime64[s]").astype(np.int64)


@dataclass(frozen=True)
class VectorNeighbor:
    case_id: str
    outcome: str
    pattern: str
    distance: float
    connecting_path: str = ""
    shared_cards: tuple[str, ...] = ()
    shared_devices: tuple[str, ...] = ()


@dataclass
class AblationRow:
    case_id: str
    actual_fraud: int
    p_base: float
    p_vec: float
    p_trav: float
    p_fused: float
    vec_ids: list[str] = field(default_factory=list)
    trav_ids: list[str] = field(default_factory=list)
    fused_ids: list[str] = field(default_factory=list)
    changed_precedent: bool = False
    vec_flipped_base: bool = False
    vec_flip_correct: bool = False
    fused_flipped_base: bool = False
    fused_flip_correct: bool = False
    fused_flipped_vec: bool = False
    fused_vs_vec_flip_correct: bool = False


@dataclass
class AblationReport:
    n_eval: int
    n_fraud: int
    n_cleared: int
    k: int
    majority_baseline_acc: float
    acc_base: float
    acc_vec: float
    acc_trav: float
    acc_fused: float
    bal_acc_base: float
    bal_acc_vec: float
    bal_acc_trav: float
    bal_acc_fused: float
    brier_base: float
    brier_vec: float
    brier_trav: float
    brier_fused: float
    changed_precedent_count: int
    changed_precedent_frac: float
    vec_flips_vs_base: int
    vec_flips_vs_base_correct: int
    vec_flips_vs_base_acc: float
    fused_flips_vs_base: int
    fused_flips_vs_base_correct: int
    fused_flips_vs_base_acc: float
    fused_flips_vs_vec: int
    fused_flips_vs_vec_correct: int
    fused_flips_vs_vec_acc: float
    filter_trace: dict[str, int] = field(default_factory=dict)
    legit_acted_vec: list[str] = field(default_factory=list)
    legit_acted_trav: list[str] = field(default_factory=list)
    legit_acted_fused: list[str] = field(default_factory=list)
    rows: list[AblationRow] = field(default_factory=list)


def _raw_embed_one(
    ep_amts: np.ndarray,
    ep_dt: np.ndarray,
    ep_online: np.ndarray,
    ep_prod: np.ndarray,
    ep_dev: np.ndarray,
    ep_addr: np.ndarray,
    ep_d1: np.ndarray,
    hist_amts: np.ndarray,
    hist_online: np.ndarray,
    hist_dev: set[str],
    hist_addr: set[float],
    asof_shared_dev_cards: int,
) -> np.ndarray:
    """Compute unstandardized 24-dim feature vector for one case episode using
    ONLY transaction/identity attributes with t <= opened_at.
    """
    v = np.zeros(EMBED_DIM, dtype=np.float64)
    n_ep = len(ep_amts)
    if n_ep == 0:
        return v

    log_amts = np.log1p(np.maximum(ep_amts, 0.0))
    amt_mean = float(np.mean(ep_amts))
    amt_max = float(np.max(ep_amts))
    v[0] = float(np.mean(log_amts))
    v[1] = float(np.max(log_amts))
    v[2] = float(np.std(log_amts)) if n_ep > 1 else 0.0
    v[3] = float(np.log1p(np.sum(np.maximum(ep_amts, 0.0))))

    n_hist = len(hist_amts)
    if n_hist > 0:
        h_mean = float(np.mean(hist_amts))
        h_std = float(np.std(hist_amts)) if n_hist > 1 else max(h_mean * 0.5, 10.0)
        v[4] = float(np.log1p(amt_mean / max(h_mean, 1.0)))
        v[5] = float(np.clip((amt_max - h_mean) / max(h_std, 5.0), -5.0, 15.0))
        v[6] = float(np.log1p(n_hist))
        v[7] = float(np.mean(hist_online))
    else:
        v[4] = float(np.log1p(amt_mean / 50.0))
        v[5] = 0.0
        v[6] = 0.0
        v[7] = float(np.mean(ep_online))

    v[8] = float(np.log1p(n_ep))
    if n_ep > 1:
        s_dt = np.sort(ep_dt)
        gaps = np.maximum(np.diff(s_dt), 0)
        span = float(s_dt[-1] - s_dt[0])
        min_gap = float(np.min(gaps))
    else:
        span = 0.0
        min_gap = 86400.0
    v[9] = float(np.log1p(span))
    v[10] = float(np.log1p(min_gap))
    v[11] = float(np.mean(ep_amts < 25.0))

    v[12] = float(np.mean(ep_online))
    v[13] = float(np.mean(ep_prod == "W"))
    v[14] = float(np.mean(ep_prod == "C"))
    v[15] = float(np.mean(ep_prod == "R"))
    v[16] = float(np.mean(ep_prod == "H"))
    v[17] = float(np.mean(ep_prod == "S"))

    dev_nonempty = [d for d in ep_dev if d]
    v[18] = float(len(dev_nonempty) / n_ep)
    if dev_nonempty:
        novel_d = sum(1 for d in dev_nonempty if d not in hist_dev)
        v[19] = float(novel_d / len(dev_nonempty))
    else:
        v[19] = 0.0

    valid_addr = [float(a) for a in ep_addr if not np.isnan(a) and a >= 0]
    if valid_addr:
        if hist_addr:
            novel_a = sum(1 for a in valid_addr if a not in hist_addr)
            v[20] = float(novel_a / len(valid_addr))
        else:
            v[20] = 0.5
    else:
        v[20] = 1.0  # missing billing region (typical in online/C product)

    hours = ((ep_dt % 86400) / 3600.0) * (2.0 * np.pi / 24.0)
    v[21] = float(np.mean(np.sin(hours)))

    valid_d1 = ep_d1[~np.isnan(ep_d1) & (ep_d1 >= 0)]
    d1_val = float(np.mean(valid_d1)) if len(valid_d1) > 0 else 0.0
    v[22] = float(np.log1p(max(d1_val, 0.0)))
    v[23] = float(np.log1p(max(asof_shared_dev_cards, 0)))
    return v


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        return cls(mean=mu, scale=sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.scale
        Z = np.nan_to_num(Z, nan=0.0, posinf=5.0, neginf=-5.0)
        norms = np.linalg.norm(Z, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return np.round(Z / norms, 6)


def embed_case(
    case_txns: pd.DataFrame,
    card_history: pd.DataFrame,
    standardizer: Standardizer | None = None,
    asof_shared_dev_cards: int = 0,
) -> np.ndarray:
    """Compute the 24-dim standardized behaviour embedding for one case episode
    given its 24h episode transactions (`t <= opened_at`) and prior card history
    (strictly before the episode window).
    """
    if case_txns.empty:
        raw = np.zeros(EMBED_DIM, dtype=np.float64)
        return standardizer.transform(raw) if standardizer is not None else raw

    amt_col = "amount" if "amount" in case_txns.columns else "TransactionAmt"
    ep_amts = case_txns[amt_col].to_numpy(dtype=np.float64, copy=False)
    if "TransactionDT" in case_txns.columns:
        ep_dt = case_txns["TransactionDT"].to_numpy(dtype=np.float64, copy=False)
    else:
        ep_dt = _to_epoch_sec(case_txns["ts"]).astype(np.float64)
    ep_online = (case_txns["channel"].to_numpy() == "online").astype(np.float64)
    ep_prod = case_txns["ProductCD"].to_numpy()
    ep_dev = (
        case_txns["device_profile"].fillna("").to_numpy()
        if "device_profile" in case_txns.columns
        else np.repeat("", len(case_txns))
    )
    ep_addr = (
        case_txns["addr1"].to_numpy(dtype=np.float64, copy=False)
        if "addr1" in case_txns.columns
        else np.full(len(case_txns), np.nan)
    )
    ep_d1 = (
        case_txns["D1"].to_numpy(dtype=np.float64, copy=False)
        if "D1" in case_txns.columns
        else np.zeros(len(case_txns))
    )

    ep_ids = set(case_txns["TransactionID"].tolist()) if "TransactionID" in case_txns.columns else set()
    if not card_history.empty:
        if ep_ids and "TransactionID" in card_history.columns:
            hist = card_history[~card_history["TransactionID"].isin(ep_ids)]
        else:
            hist = card_history
    else:
        hist = card_history

    if not hist.empty:
        h_amt_col = "amount" if "amount" in hist.columns else "TransactionAmt"
        hist_amts = hist[h_amt_col].to_numpy(dtype=np.float64, copy=False)
        hist_online = (hist["channel"].to_numpy() == "online").astype(np.float64)
        hist_dev = (
            {d for d in hist["device_profile"].fillna("").tolist() if d}
            if "device_profile" in hist.columns
            else set()
        )
        hist_addr = (
            {float(a) for a in hist["addr1"].dropna().tolist() if a >= 0}
            if "addr1" in hist.columns
            else set()
        )
    else:
        hist_amts = np.empty(0, dtype=np.float64)
        hist_online = np.empty(0, dtype=np.float64)
        hist_dev = set()
        hist_addr = set()

    raw = _raw_embed_one(
        ep_amts=ep_amts,
        ep_dt=ep_dt,
        ep_online=ep_online,
        ep_prod=ep_prod,
        ep_dev=ep_dev,
        ep_addr=ep_addr,
        ep_d1=ep_d1,
        hist_amts=hist_amts,
        hist_online=hist_online,
        hist_dev=hist_dev,
        hist_addr=hist_addr,
        asof_shared_dev_cards=asof_shared_dev_cards,
    )
    return standardizer.transform(raw) if standardizer is not None else raw


def _build_tx_index(tx: pd.DataFrame) -> tuple[dict[int, int], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build fast numpy lookup arrays per TransactionID, per card_id, and per device_profile (sorted by t)."""
    tx_sorted = tx.sort_values("t", kind="stable").reset_index(drop=True)
    tid_arr = tx_sorted["TransactionID"].to_numpy(dtype=np.int64)
    tid_to_pos = dict(zip(tid_arr.tolist(), range(len(tid_arr))))

    amt_col = "amount" if "amount" in tx_sorted.columns else "TransactionAmt"
    t_sec = _to_epoch_sec(tx_sorted["t"])

    cols = {
        "tid": tid_arr,
        "t_sec": t_sec,
        "dt": tx_sorted["TransactionDT"].to_numpy(dtype=np.float64),
        "amt": tx_sorted[amt_col].to_numpy(dtype=np.float64),
        "online": (tx_sorted["channel"].to_numpy() == "online").astype(np.float64),
        "prod": tx_sorted["ProductCD"].fillna("").to_numpy(),
        "dev": tx_sorted["device_profile"].fillna("").to_numpy(),
        "addr": tx_sorted["addr1"].fillna(-1.0).to_numpy(dtype=np.float64),
        "d1": tx_sorted["D1"].fillna(0.0).to_numpy(dtype=np.float64),
        "email": tx_sorted["P_emaildomain"].fillna("").to_numpy(),
        "card": tx_sorted["card_id"].to_numpy(),
        "risk": tx_sorted["risk_score"].fillna(0.0).to_numpy(dtype=np.float64),
    }

    card_groups = tx_sorted.groupby("card_id", sort=False).indices
    non_empty_dev = tx_sorted[tx_sorted["device_profile"].fillna("") != ""]
    dev_groups = non_empty_dev.groupby("device_profile", sort=False).indices
    return tid_to_pos, cols, card_groups, dev_groups


def _count_asof_shared_dev_cards(
    devs: Iterable[str],
    card_id: str,
    t_cut_sec: int,
    dev_groups: dict[str, np.ndarray],
    t_sec_all: np.ndarray,
    card_all: np.ndarray,
) -> int:
    """Count distinct OTHER cards that used any device in `devs` on or before `t_cut_sec`."""
    other_cards: set[str] = set()
    for d in devs:
        if not d:
            continue
        d_idx = dev_groups.get(d)
        if d_idx is None or len(d_idx) == 0:
            continue
        cut = int(np.searchsorted(t_sec_all[d_idx], t_cut_sec, side="right"))
        if cut > 0:
            for c in card_all[d_idx[:cut]].tolist():
                if c != card_id:
                    other_cards.add(str(c))
    return len(other_cards)


def _select_asof_window(
    card_indices: np.ndarray,
    t_cut_sec: int,
    t_sec_all: np.ndarray,
    window_hours: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the causal 24-hour episode window ending at the card's latest transaction
    with `t <= opened_at` (`t_cut_sec`), and the prior card history strictly before
    that 24-hour window. Uses ONLY `card_id` and `opened_at` (zero post-hoc case columns).
    """
    if card_indices is None or len(card_indices) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    c_times = t_sec_all[card_indices]
    cut_right = int(np.searchsorted(c_times, t_cut_sec, side="right"))
    asof_indices = card_indices[:cut_right]
    if len(asof_indices) == 0:
        return card_indices[:1], np.empty(0, dtype=np.int64)

    t_latest_sec = int(t_sec_all[asof_indices[-1]])
    win_start_sec = t_latest_sec - int(window_hours * 3600)
    asof_times = t_sec_all[asof_indices]
    win_left = int(np.searchsorted(asof_times, win_start_sec, side="left"))

    ep_idx = asof_indices[win_left:]
    hist_idx = asof_indices[:win_left]
    return ep_idx, hist_idx


def embed_closed_cases(
    cc: pd.DataFrame | None = None,
    tx: pd.DataFrame | None = None,
    return_standardizer: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Standardizer]:
    """Compute standardized 24-dim behaviour embeddings for all closed cases
    strictly as-of each case's `opened_at`.

    IMPORTANT: To prevent any label/post-investigation leakage, only `card_id`
    and `opened_at` from `cc` are used to slice `tx` (`t <= opened_at`).
    """
    if cc is None:
        cc = closed_cases()
    if tx is None:
        tx = tx_all()

    _, cols, card_groups, dev_groups = _build_tx_index(tx)
    t_sec_all = cols["t_sec"]
    dt_all = cols["dt"]
    amt_all = cols["amt"]
    online_all = cols["online"]
    prod_all = cols["prod"]
    dev_all = cols["dev"]
    addr_all = cols["addr"]
    d1_all = cols["d1"]
    card_all = cols["card"]

    opened_ts = pd.to_datetime(cc["opened_at"])
    opened_sec = _to_epoch_sec(opened_ts)
    closed_ts = pd.to_datetime(cc["closed_at"])

    n_cases = len(cc)
    raw_mat = np.zeros((n_cases, EMBED_DIM), dtype=np.float64)
    cards = cc["card_id"].tolist()

    for idx in range(n_cases):
        c_id = str(cards[idx])
        t_cut = int(opened_sec[idx])
        c_indices = card_groups.get(c_id)
        ep_idx, hist_idx = _select_asof_window(c_indices, t_cut, t_sec_all, window_hours=24)

        if len(hist_idx) > 0:
            h_amts = amt_all[hist_idx]
            h_onl = online_all[hist_idx]
            h_devs = {str(d) for d in dev_all[hist_idx].tolist() if d}
            h_addrs = {float(a) for a in addr_all[hist_idx].tolist() if a >= 0}
        else:
            h_amts = np.empty(0, dtype=np.float64)
            h_onl = np.empty(0, dtype=np.float64)
            h_devs = set()
            h_addrs = set()

        if len(ep_idx) > 0:
            ep_a = amt_all[ep_idx]
            ep_d = dt_all[ep_idx]
            ep_o = online_all[ep_idx]
            ep_p = prod_all[ep_idx]
            ep_dv = dev_all[ep_idx]
            ep_ad = np.where(addr_all[ep_idx] < 0, np.nan, addr_all[ep_idx])
            ep_d1_arr = d1_all[ep_idx]
            n_shared_cards = _count_asof_shared_dev_cards(
                devs={str(d) for d in ep_dv.tolist() if d},
                card_id=c_id,
                t_cut_sec=t_cut,
                dev_groups=dev_groups,
                t_sec_all=t_sec_all,
                card_all=card_all,
            )
        else:
            ep_a = np.empty(0, dtype=np.float64)
            ep_d = np.empty(0, dtype=np.float64)
            ep_o = np.empty(0, dtype=np.float64)
            ep_p = np.empty(0, dtype=object)
            ep_dv = np.empty(0, dtype=object)
            ep_ad = np.empty(0, dtype=np.float64)
            ep_d1_arr = np.empty(0, dtype=np.float64)
            n_shared_cards = 0

        raw_mat[idx] = _raw_embed_one(
            ep_amts=ep_a,
            ep_dt=ep_d,
            ep_online=ep_o,
            ep_prod=ep_p,
            ep_dev=ep_dv,
            ep_addr=ep_ad,
            ep_d1=ep_d1_arr,
            hist_amts=h_amts,
            hist_online=h_onl,
            hist_dev=h_devs,
            hist_addr=h_addrs,
            asof_shared_dev_cards=n_shared_cards,
        )

    # Fit standardizer strictly on earlier cases (closed before 2016-10-01) so the Oct-Dec backtest has zero leakage
    cal_mask = (closed_ts < pd.Timestamp("2016-10-01")).to_numpy()
    fit_slice = raw_mat[cal_mask] if np.any(cal_mask) else raw_mat
    std = Standardizer.fit(fit_slice)
    Z = std.transform(raw_mat)

    out = pd.DataFrame({
        "case_id": cc["case_id"].to_numpy(),
        "card_id": cc["card_id"].to_numpy(),
        "customer_id": cc["customer_id"].to_numpy(),
        "opened_at": opened_ts.to_numpy(),
        "closed_at": closed_ts.to_numpy(),
        "outcome": cc["outcome"].to_numpy(),
        "pattern": cc["pattern"].to_numpy(),
        "embedding": list(Z),
    })
    for j in range(EMBED_DIM):
        out[f"f{j}"] = Z[:, j]

    if return_standardizer:
        return out, std
    return out


def embed_exam_cases(
    cp: pd.DataFrame | None = None,
    tx: pd.DataFrame | None = None,
    standardizer: Standardizer | None = None,
) -> pd.DataFrame:
    """Compute standardized 24-dim behaviour embeddings for all 18 exam cases in
    case_pack() as-of each case's opened_at.
    """
    if cp is None:
        cp = case_pack()
    if tx is None:
        tx = tx_all()
    if standardizer is None:
        _, standardizer = embed_closed_cases(closed_cases(), tx, return_standardizer=True)

    _, cols, _, dev_groups = _build_tx_index(tx)
    tx_by_id = tx.set_index("TransactionID", drop=False)

    rows = []
    for r in cp.itertuples():
        opened_at = pd.Timestamp(r.opened_at)
        t_cut_sec = _to_epoch_sec(opened_at)
        flagged_id = int(r.flagged_txn_id)
        if flagged_id in tx_by_id.index:
            actual_card = str(tx_by_id.loc[flagged_id, "card_id"])
        else:
            actual_card = str(r.card_id)

        asof_tx = tx[(tx["card_id"] == actual_card) & (tx["t"] <= opened_at)]
        if flagged_id in tx_by_id.index:
            t0 = pd.Timestamp(tx_by_id.loc[flagged_id, "t"])
            win_start = t0 - pd.Timedelta(hours=24)
            ep_tx = asof_tx[(asof_tx["t"] >= win_start) & (asof_tx["t"] <= opened_at)]
            if ep_tx.empty:
                ep_tx = tx_by_id.loc[[flagged_id]]
        else:
            ep_tx = asof_tx.tail(1)

        hist_tx = asof_tx[~asof_tx["TransactionID"].isin(ep_tx["TransactionID"])]
        n_shared = _count_asof_shared_dev_cards(
            devs={str(d) for d in ep_tx["device_profile"].fillna("").tolist() if d},
            card_id=actual_card,
            t_cut_sec=t_cut_sec,
            dev_groups=dev_groups,
            t_sec_all=cols["t_sec"],
            card_all=cols["card"],
        )

        vec = embed_case(ep_tx, hist_tx, standardizer=standardizer, asof_shared_dev_cards=n_shared)
        row_dict = {
            "case_id": r.case_id,
            "card_id": actual_card,
            "customer_id": r.customer_id,
            "flagged_txn_id": flagged_id,
            "opened_at": opened_at,
            "embedding": vec,
        }
        for j in range(EMBED_DIM):
            row_dict[f"f{j}"] = float(vec[j])
        rows.append(row_dict)

    return pd.DataFrame(rows)


def export_tg_vectors(cc_emb: pd.DataFrame | None = None, out_path: Path | None = None) -> Path:
    """Write data/tg_r2/case_vectors.csv formatted for TigerGraph 4.2 VECTOR loading."""
    if cc_emb is None:
        cc_emb = embed_closed_cases()
    if out_path is None:
        TG_R2_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TG_R2_DIR / "case_vectors.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    vecs = [
        ",".join(f"{float(x):.6f}" for x in v)
        for v in cc_emb["embedding"].tolist()
    ]
    df_out = pd.DataFrame({"case_id": cc_emb["case_id"].to_numpy(), "behaviour": vecs})
    df_out.to_csv(out_path, index=False, sep="|")
    return out_path


class TigerGraphUnavailableError(RuntimeError):
    """Raised when ASSAY_TG=1 and TigerGraph is unreachable or errors out (fail-closed)."""


def _tg_run_query(query_name: str, params: dict) -> list[dict]:
    """Execute an installed GSQL query on TigerGraph when ASSAY_TG=1.
    Fails closed (raises TigerGraphUnavailableError) if TigerGraph is down or unreachable.
    """
    try:
        from assay.tg import TigerGraphStore

        store = TigerGraphStore()
        res = store.conn.runInstalledQuery(query_name, params=params, timeout=30000)
        if not res or not isinstance(res, list) or not isinstance(res[0], dict) or not res[0]:
            # a reply with no result body is garbage, not "no neighbours"
            raise ValueError(f"malformed reply: {str(res)[:200]}")
        first_key = next(iter(res[0]))
        rows = res[0][first_key]
        return rows if isinstance(rows, list) else []
    except Exception as exc:
        raise TigerGraphUnavailableError(
            f"Fail-closed: ASSAY_TG=1 but TigerGraph query '{query_name}' failed: {exc}"
        ) from exc


class LocalVectorIndex:
    """Reference search index over closed-case embeddings + 2-hop entity index
    over shared DeviceProfile, BillingRegion (addr1), and EmailDomain (P_emaildomain).
    When `ASSAY_TG=1`, `similar_cases_vec` and `similar_cases_fused` route through
    TigerGraph and fail closed if TigerGraph is unreachable.
    """

    def __init__(self, cc_emb: pd.DataFrame, cc: pd.DataFrame, tx: pd.DataFrame):
        self.cc_emb = cc_emb.reset_index(drop=True)
        self.cc = cc.reset_index(drop=True)
        self.case_ids = self.cc_emb["case_id"].to_numpy()
        self.case_to_idx: dict[str, int] = {str(cid): i for i, cid in enumerate(self.case_ids)}
        self.outcomes = self.cc_emb["outcome"].to_numpy()
        self.patterns = self.cc_emb["pattern"].to_numpy()
        self.cards = self.cc_emb["card_id"].to_numpy()
        self.opened_sec = _to_epoch_sec(self.cc_emb["opened_at"])
        self.closed_sec = _to_epoch_sec(self.cc_emb["closed_at"])
        self.mat = np.vstack(self.cc_emb["embedding"].to_list()).astype(np.float64)

        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.mat)

        tid_to_pos, cols, card_groups, dev_groups = _build_tx_index(tx)
        self.tid_to_pos = tid_to_pos
        self.cols = cols
        self.card_groups = card_groups
        self.dev_groups = dev_groups

        self.dev_to_cases: dict[str, list[int]] = {}
        self.region_to_cases: dict[int, list[int]] = {}
        self.email_to_cases: dict[str, list[int]] = {}
        self.card_to_cases: dict[str, list[int]] = {}

        dev_all = cols["dev"]
        addr_all = cols["addr"]
        email_all = cols["email"]

        self.case_devs: list[set[str]] = []
        self.case_regions: list[set[int]] = []
        self.case_emails: list[set[str]] = []
        self.case_cards: list[set[str]] = []

        for i, tids in enumerate(self.cc["txn_list"].tolist()):
            pos = [tid_to_pos[t] for t in tids if t in tid_to_pos]
            devs = {str(dev_all[p]) for p in pos if dev_all[p]}
            regs = {int(addr_all[p]) for p in pos if addr_all[p] >= 0}
            ems = {str(email_all[p]) for p in pos if email_all[p]}
            c_id = str(self.cards[i])
            c_set = {c_id} | {str(x) for x in self.cc["connected_list"].iloc[i]}

            self.case_devs.append(devs)
            self.case_regions.append(regs)
            self.case_emails.append(ems)
            self.case_cards.append(c_set)

            for d in devs:
                self.dev_to_cases.setdefault(d, []).append(i)
            for rg in regs:
                self.region_to_cases.setdefault(rg, []).append(i)
            for em in ems:
                self.email_to_cases.setdefault(em, []).append(i)
            for c_str in c_set:
                self.card_to_cases.setdefault(c_str, []).append(i)

    def similar_cases_vec(
        self,
        qv: np.ndarray,
        k: int = 5,
        as_of: pd.Timestamp | None = None,
        exclude_case_id: str | None = None,
        use_tg_if_enabled: bool = True,
    ) -> list[VectorNeighbor]:
        """Pure cosine vector top-k over ClosedCase.behaviour restricted to cases
        closed on or before `as_of`, plus 1-2 hop shared cards/devices on the hits.
        """
        if use_tg_if_enabled and os.environ.get("ASSAY_TG") == "1":
            as_of_str = (
                pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S")
                if as_of is not None
                else "2099-01-01 00:00:00"
            )
            rows = _tg_run_query(
                "similar_cases_vec",
                {
                    "qv": [round(float(x), 6) for x in np.asarray(qv).ravel().tolist()],
                    "k": int(k),
                    "as_of": as_of_str,
                },
            )
            q = np.asarray(qv, dtype=np.float64).ravel()
            q_norm = q / max(float(np.linalg.norm(q)), 1e-9)
            out_tg: list[VectorNeighbor] = []
            for r in rows:
                vid = str(r.get("v_id", r.get("case_id", "")))
                if exclude_case_id and vid == exclude_case_id:
                    continue
                attrs = r.get("attributes", r)
                raw_dist = attrs.get("distance", attrs.get("@dist"))
                if raw_dist is not None:
                    dist_val = float(raw_dist)
                elif vid in self.case_to_idx:
                    dist_val = round(float(max(1.0 - float(self.mat[self.case_to_idx[vid]] @ q_norm), 0.0)), 6)
                else:
                    dist_val = 0.0
                sc = attrs.get("shared_cards", attrs.get("@shared_cards", []))
                sd = attrs.get("shared_devices", attrs.get("@shared_devices", []))
                out_tg.append(
                    VectorNeighbor(
                        case_id=vid,
                        outcome=str(attrs.get("outcome", "")),
                        pattern=str(attrs.get("pattern", "")),
                        distance=dist_val,
                        connecting_path="tigergraph_vectorSearch",
                        shared_cards=tuple(str(x) for x in sc if x),
                        shared_devices=tuple(str(x) for x in sd if x),
                    )
                )
            return out_tg[:k]

        q = np.asarray(qv, dtype=np.float64).reshape(1, -1)
        cut_sec = _to_epoch_sec(as_of) if as_of is not None else None

        if cut_sec is not None:
            elig = np.where(self.closed_sec <= cut_sec)[0]
        else:
            elig = np.arange(len(self.mat), dtype=np.int64)

        if exclude_case_id:
            elig = elig[self.case_ids[elig] != exclude_case_id]
        if len(elig) == 0:
            return []

        q_norm = q / max(float(np.linalg.norm(q)), 1e-9)
        sub = self.mat[elig]
        cos_d = 1.0 - (sub @ q_norm.T).ravel()
        order = np.argsort(cos_d, kind="stable")[:k]

        return [
            VectorNeighbor(
                case_id=str(self.case_ids[elig[j]]),
                outcome=str(self.outcomes[elig[j]]),
                pattern=str(self.patterns[elig[j]]),
                distance=round(float(max(cos_d[j], 0.0)), 6),
                connecting_path="vector_only",
                shared_cards=tuple(sorted(self.case_cards[int(elig[j])])),
                shared_devices=tuple(sorted(self.case_devs[int(elig[j])])),
            )
            for j in order
        ]

    def reachable_candidates(
        self,
        card_id: str,
        as_of: pd.Timestamp | None = None,
        episode_tids: Iterable[int] | None = None,
        exclude_case_id: str | None = None,
    ) -> dict[int, tuple[int, str]]:
        """Traverse from Card (using only card transactions with `t <= as_of`) up to
        2 hops through shared DeviceProfile, BillingRegion, or EmailDomain to ClosedCase
        vertices with `closed_at <= as_of`.
        """
        cut_sec = _to_epoch_sec(as_of) if as_of is not None else None
        c_indices = self.card_groups.get(card_id, np.empty(0, dtype=np.int64))
        if cut_sec is not None and len(c_indices) > 0:
            c_times = self.cols["t_sec"][c_indices]
            c_indices = c_indices[: int(np.searchsorted(c_times, cut_sec, side="right"))]

        if episode_tids is not None:
            ep_pos = [
                self.tid_to_pos[int(t)]
                for t in episode_tids
                if int(t) in self.tid_to_pos
                and (cut_sec is None or int(self.cols["t_sec"][self.tid_to_pos[int(t)]]) <= cut_sec)
            ]
            if ep_pos:
                pos_arr = np.unique(np.concatenate([c_indices, np.array(ep_pos, dtype=np.int64)]))
            else:
                pos_arr = c_indices
        else:
            pos_arr = c_indices

        devs = {str(self.cols["dev"][p]) for p in pos_arr if self.cols["dev"][p]}
        regs = {int(self.cols["addr"][p]) for p in pos_arr if self.cols["addr"][p] >= 0}
        ems = {str(self.cols["email"][p]) for p in pos_arr if self.cols["email"][p]}

        cand: dict[int, tuple[int, str]] = {}

        def _consider(idx: int, tier: int, path: str) -> None:
            if cut_sec is not None and self.closed_sec[idx] > cut_sec:
                return
            if exclude_case_id and str(self.case_ids[idx]) == exclude_case_id:
                return
            prev = cand.get(idx)
            if prev is None or tier < prev[0]:
                cand[idx] = (tier, path)

        for idx in self.card_to_cases.get(card_id, ()):
            _consider(idx, 0, f"Card({card_id})->ClosedCase({self.case_ids[idx]})")

        for d in sorted(devs):
            for idx in self.dev_to_cases.get(d, ()):
                _consider(idx, 1, f"Card({card_id})->DeviceProfile({d[:24]})->ClosedCase({self.case_ids[idx]})")

        for rg in sorted(regs):
            for idx in self.region_to_cases.get(rg, ()):
                if ems and (ems & self.case_emails[idx]):
                    em_hit = sorted(ems & self.case_emails[idx])[0]
                    _consider(
                        idx,
                        2,
                        f"Card({card_id})->BillingRegion({rg})+EmailDomain({em_hit})->ClosedCase({self.case_ids[idx]})",
                    )
                else:
                    _consider(idx, 3, f"Card({card_id})->BillingRegion({rg})->ClosedCase({self.case_ids[idx]})")

        for em in sorted(ems):
            for idx in self.email_to_cases.get(em, ()):
                _consider(idx, 4, f"Card({card_id})->EmailDomain({em})->ClosedCase({self.case_ids[idx]})")

        return cand

    def similar_cases_traversal_only(
        self,
        card_id: str,
        k: int = 5,
        as_of: pd.Timestamp | None = None,
        episode_tids: Iterable[int] | None = None,
        exclude_case_id: str | None = None,
    ) -> list[VectorNeighbor]:
        """Graph traversal candidate selection ordered purely by structural tier
        and recency (no behaviour vector ranking).
        """
        cand = self.reachable_candidates(
            card_id=card_id,
            as_of=as_of,
            episode_tids=episode_tids,
            exclude_case_id=exclude_case_id,
        )
        if not cand:
            return []
        ordered = sorted(
            cand.items(),
            key=lambda kv: (kv[1][0], -int(self.closed_sec[kv[0]]), str(self.case_ids[kv[0]])),
        )[:k]
        return [
            VectorNeighbor(
                case_id=str(self.case_ids[idx]),
                outcome=str(self.outcomes[idx]),
                pattern=str(self.patterns[idx]),
                distance=float(tier),
                connecting_path=path,
                shared_cards=tuple(sorted(self.case_cards[idx])),
                shared_devices=tuple(sorted(self.case_devs[idx])),
            )
            for idx, (tier, path) in ordered
        ]

    def similar_cases_fused(
        self,
        card_id: str,
        qv: np.ndarray,
        k: int = 5,
        hops: int = 2,
        as_of: pd.Timestamp | None = None,
        episode_tids: Iterable[int] | None = None,
        exclude_case_id: str | None = None,
        use_tg_if_enabled: bool = True,
    ) -> list[VectorNeighbor]:
        """Vector top-k restricted to closed cases reachable from `card_id` within
        2 hops through shared DeviceProfile / BillingRegion / EmailDomain and
        closed on or before `as_of`.
        """
        if use_tg_if_enabled and os.environ.get("ASSAY_TG") == "1":
            as_of_str = (
                pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S")
                if as_of is not None
                else "2099-01-01 00:00:00"
            )
            rows = _tg_run_query(
                "similar_cases_fused",
                {
                    "c": (str(card_id),),
                    "qv": [round(float(x), 6) for x in np.asarray(qv).ravel().tolist()],
                    "k": int(k),
                    "hops": int(hops),
                    "as_of": as_of_str,
                },
            )
            q = np.asarray(qv, dtype=np.float64).ravel()
            q_norm = q / max(float(np.linalg.norm(q)), 1e-9)
            out_tg: list[VectorNeighbor] = []
            for r in rows:
                vid = str(r.get("v_id", r.get("case_id", "")))
                if exclude_case_id and vid == exclude_case_id:
                    continue
                attrs = r.get("attributes", r)
                raw_dist = attrs.get("distance", attrs.get("@dist"))
                if raw_dist is not None:
                    dist_val = float(raw_dist)
                elif vid in self.case_to_idx:
                    dist_val = round(float(max(1.0 - float(self.mat[self.case_to_idx[vid]] @ q_norm), 0.0)), 6)
                else:
                    dist_val = 0.0
                sc = attrs.get("shared_cards", attrs.get("@shared_cards", []))
                sd = attrs.get("shared_devices", attrs.get("@shared_devices", []))
                out_tg.append(
                    VectorNeighbor(
                        case_id=vid,
                        outcome=str(attrs.get("outcome", "")),
                        pattern=str(attrs.get("pattern", "")),
                        distance=dist_val,
                        connecting_path=str(attrs.get("connecting_path", attrs.get("@connecting_path", ""))),
                        shared_cards=tuple(str(x) for x in sc if x),
                        shared_devices=tuple(str(x) for x in sd if x),
                    )
                )
            return out_tg[:k]

        cand = self.reachable_candidates(
            card_id=card_id,
            as_of=as_of,
            episode_tids=episode_tids,
            exclude_case_id=exclude_case_id,
        )
        if not cand:
            return self.similar_cases_vec(
                qv, k=k, as_of=as_of, exclude_case_id=exclude_case_id, use_tg_if_enabled=False
            )

        q = np.asarray(qv, dtype=np.float64).reshape(1, -1)
        q_norm = q / max(float(np.linalg.norm(q)), 1e-9)

        idx_arr = np.array(list(cand.keys()), dtype=np.int64)
        sub = self.mat[idx_arr]
        cos_d = 1.0 - (sub @ q_norm.T).ravel()

        scored: list[tuple[float, float, int, str]] = []
        for pos_i, case_idx in enumerate(idx_arr):
            tier, path = cand[int(case_idx)]
            d_val = float(max(cos_d[pos_i], 0.0))
            eff_score = d_val - (0.20 if tier <= 1 else (0.05 if tier == 2 else 0.0))
            scored.append((eff_score, d_val, int(case_idx), path))

        scored.sort(key=lambda x: (x[0], x[1], str(self.case_ids[x[2]])))
        top = scored[:k]
        return [
            VectorNeighbor(
                case_id=str(self.case_ids[c_idx]),
                outcome=str(self.outcomes[c_idx]),
                pattern=str(self.patterns[c_idx]),
                distance=round(d_val, 6),
                connecting_path=path,
                shared_cards=tuple(sorted(self.case_cards[c_idx])),
                shared_devices=tuple(sorted(self.case_devs[c_idx])),
            )
            for _, d_val, c_idx, path in top
        ]


def _knn_fraud_prob(neighbors: list[VectorNeighbor], prior_fraud: float = 0.5) -> float:
    """Distance-weighted k-NN probability of confirmed_fraud with Laplace smoothing."""
    if not neighbors:
        return prior_fraud
    weights = []
    labels = []
    for n in neighbors:
        w = 1.0 / (0.05 + max(float(n.distance), 0.0))
        weights.append(w)
        labels.append(1.0 if n.outcome == "confirmed_fraud" else 0.0)
    w_arr = np.array(weights, dtype=np.float64)
    y_arr = np.array(labels, dtype=np.float64)
    raw_p = float(np.sum(w_arr * y_arr) / np.sum(w_arr))
    return float(0.92 * raw_p + 0.08 * prior_fraud)


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(np.mean(y_pred[pos] == 1)) if np.any(pos) else 0.0
    tnr = float(np.mean(y_pred[neg] == 0)) if np.any(neg) else 0.0
    return 0.5 * (tpr + tnr)


def run_ablation(
    cc: pd.DataFrame | None = None,
    tx: pd.DataFrame | None = None,
    k: int = 5,
    eval_start: str = "2016-10-01",
    eval_end: str = "2017-01-01",
) -> AblationReport:
    """Held-out closed-case ablation over ALL Oct-Dec closed cases (Nov-Dec hold only 45) (as-of each
    case's opened_at, with the index restricted to cases closed before each
    eval case opened):
      - baseline: non-vector risk_score in the causal 24h episode window (`t <= opened_at`)
      - (i) vector-only k-NN over prior closed cases
      - (ii) traversal-only 2-hop graph k-NN over prior closed cases
      - (iii) fused 2-hop traversal + vector k-NN
    Also counts how often vector/fused precedent flips the baseline verdict and
    whether the flip is right, and records `filter_trace` showing the exact case
    count at each step.
    """
    if cc is None:
        cc = closed_cases()
    if tx is None:
        tx = tx_all()

    cc_emb = embed_closed_cases(cc, tx)
    index = LocalVectorIndex(cc_emb, cc, tx)

    # Filter trace using epoch seconds AND string prefix check so nothing can silently drop Nov-Dec rows
    t_start_sec = _to_epoch_sec(eval_start)
    t_end_sec = _to_epoch_sec(eval_end)
    after_start = index.opened_sec >= t_start_sec
    before_end = index.opened_sec < t_end_sec
    eval_mask = after_start & before_end
    eval_indices = np.where(eval_mask)[0]

    filter_trace = {
        "total_closed_cases": int(len(cc)),
        "opened_ge_eval_start": int(np.sum(after_start)),
        "opened_in_eval_window": int(len(eval_indices)),
        "min_asof_prior_cases_available": int(
            min((int(np.sum(index.closed_sec <= index.opened_sec[i])) for i in eval_indices), default=0)
        ),
    }

    t_sec_all = index.cols["t_sec"]
    risk_all = index.cols["risk"]

    rows: list[AblationRow] = []
    for idx in eval_indices:
        cid = str(index.case_ids[idx])
        card_id = str(index.cards[idx])
        as_of = pd.Timestamp(cc["opened_at"].iloc[idx])
        t_cut_sec = int(index.opened_sec[idx])
        actual = 1 if str(index.outcomes[idx]) == "confirmed_fraud" else 0
        qv = index.mat[idx]

        # Non-vector baseline: max bank risk_score in the causal 24h episode window
        c_indices = index.card_groups.get(card_id, np.empty(0, dtype=np.int64))
        ep_idx, _ = _select_asof_window(c_indices, t_cut_sec, t_sec_all, window_hours=24)
        p_base = float(np.max(risk_all[ep_idx])) if len(ep_idx) > 0 else 0.5

        vec_nb = index.similar_cases_vec(
            qv, k=k, as_of=as_of, exclude_case_id=cid, use_tg_if_enabled=False
        )
        trav_nb = index.similar_cases_traversal_only(
            card_id=card_id, k=k, as_of=as_of, episode_tids=None, exclude_case_id=cid
        )
        fused_nb = index.similar_cases_fused(
            card_id=card_id,
            qv=qv,
            k=k,
            hops=2,
            as_of=as_of,
            episode_tids=None,
            exclude_case_id=cid,
            use_tg_if_enabled=False,
        )

        p_vec = _knn_fraud_prob(vec_nb)
        if trav_nb:
            trav_raw = float(np.mean([1.0 if n.outcome == "confirmed_fraud" else 0.0 for n in trav_nb]))
            p_trav = float(0.92 * trav_raw + 0.08 * 0.5)
        else:
            p_trav = 0.5
        p_fused = _knn_fraud_prob(fused_nb)

        vec_ids = [n.case_id for n in vec_nb]
        trav_ids = [n.case_id for n in trav_nb]
        fused_ids = [n.case_id for n in fused_nb]
        changed = set(fused_ids) != set(vec_ids)

        pred_base = int(p_base >= 0.5)
        pred_vec = int(p_vec >= 0.5)
        pred_fused = int(p_fused >= 0.5)

        vec_flipped_base = pred_vec != pred_base
        fused_flipped_base = pred_fused != pred_base
        fused_flipped_vec = pred_fused != pred_vec

        rows.append(
            AblationRow(
                case_id=cid,
                actual_fraud=actual,
                p_base=p_base,
                p_vec=p_vec,
                p_trav=p_trav,
                p_fused=p_fused,
                vec_ids=vec_ids,
                trav_ids=trav_ids,
                fused_ids=fused_ids,
                changed_precedent=changed,
                vec_flipped_base=vec_flipped_base,
                vec_flip_correct=bool(vec_flipped_base and (pred_vec == actual)),
                fused_flipped_base=fused_flipped_base,
                fused_flip_correct=bool(fused_flipped_base and (pred_fused == actual)),
                fused_flipped_vec=fused_flipped_vec,
                fused_vs_vec_flip_correct=bool(fused_flipped_vec and (pred_fused == actual)),
            )
        )

    filter_trace["evaluated_rows"] = len(rows)

    y = np.array([r.actual_fraud for r in rows], dtype=np.float64)
    p_b = np.array([r.p_base for r in rows], dtype=np.float64)
    p_v = np.array([r.p_vec for r in rows], dtype=np.float64)
    p_t = np.array([r.p_trav for r in rows], dtype=np.float64)
    p_f = np.array([r.p_fused for r in rows], dtype=np.float64)

    maj_acc = float(max(np.mean(y == 1), np.mean(y == 0)))
    acc_base = float(np.mean((p_b >= 0.5) == y))
    acc_vec = float(np.mean((p_v >= 0.5) == y))
    acc_trav = float(np.mean((p_t >= 0.5) == y))
    acc_fused = float(np.mean((p_f >= 0.5) == y))

    bal_base = _balanced_acc(y, (p_b >= 0.5).astype(np.float64))
    bal_vec = _balanced_acc(y, (p_v >= 0.5).astype(np.float64))
    bal_trav = _balanced_acc(y, (p_t >= 0.5).astype(np.float64))
    bal_fused = _balanced_acc(y, (p_f >= 0.5).astype(np.float64))

    brier_base = float(np.mean((p_b - y) ** 2))
    brier_vec = float(np.mean((p_v - y) ** 2))
    brier_trav = float(np.mean((p_t - y) ** 2))
    brier_fused = float(np.mean((p_f - y) ** 2))

    changed_cnt = int(sum(1 for r in rows if r.changed_precedent))
    legit_vec = [r.case_id for r in rows if r.actual_fraud == 0 and r.p_vec >= 0.70]
    legit_trav = [r.case_id for r in rows if r.actual_fraud == 0 and r.p_trav >= 0.70]
    legit_fused = [r.case_id for r in rows if r.actual_fraud == 0 and r.p_fused >= 0.70]

    vec_flips = int(sum(1 for r in rows if r.vec_flipped_base))
    vec_flips_ok = int(sum(1 for r in rows if r.vec_flip_correct))
    fused_flips = int(sum(1 for r in rows if r.fused_flipped_base))
    fused_flips_ok = int(sum(1 for r in rows if r.fused_flip_correct))
    fv_flips = int(sum(1 for r in rows if r.fused_flipped_vec))
    fv_flips_ok = int(sum(1 for r in rows if r.fused_vs_vec_flip_correct))

    return AblationReport(
        n_eval=len(rows),
        n_fraud=int(np.sum(y == 1)),
        n_cleared=int(np.sum(y == 0)),
        k=k,
        majority_baseline_acc=maj_acc,
        acc_base=acc_base,
        acc_vec=acc_vec,
        acc_trav=acc_trav,
        acc_fused=acc_fused,
        bal_acc_base=bal_base,
        bal_acc_vec=bal_vec,
        bal_acc_trav=bal_trav,
        bal_acc_fused=bal_fused,
        brier_base=brier_base,
        brier_vec=brier_vec,
        brier_trav=brier_trav,
        brier_fused=brier_fused,
        changed_precedent_count=changed_cnt,
        changed_precedent_frac=float(changed_cnt / max(len(rows), 1)),
        vec_flips_vs_base=vec_flips,
        vec_flips_vs_base_correct=vec_flips_ok,
        vec_flips_vs_base_acc=float(vec_flips_ok / max(vec_flips, 1)),
        fused_flips_vs_base=fused_flips,
        fused_flips_vs_base_correct=fused_flips_ok,
        fused_flips_vs_base_acc=float(fused_flips_ok / max(fused_flips, 1)),
        fused_flips_vs_vec=fv_flips,
        fused_flips_vs_vec_correct=fv_flips_ok,
        fused_flips_vs_vec_acc=float(fv_flips_ok / max(fv_flips, 1)),
        filter_trace=filter_trace,
        legit_acted_vec=legit_vec,
        legit_acted_trav=legit_trav,
        legit_acted_fused=legit_fused,
        rows=rows,
    )


class FusedVectorDetector(Detector):
    """Detector wrapper exposing TigerVector fused search as an attested Finding
    with a measured as-of backtest over held-out closed cases.
    """

    name = "tiger_vector_fused"
    pattern = "vector_fused_precedent"
    FIRE_THRESHOLD = 0.80

    def __init__(self, k: int = 5):
        self.k = k

    def run(self, ctx: CaseCtx) -> list[Finding]:
        cc_emb, std = embed_closed_cases(ctx.cc, ctx.tx, return_standardizer=True)
        idx = LocalVectorIndex(cc_emb, ctx.cc, ctx.tx)

        card_tx = ctx.tx[(ctx.tx["card_id"] == ctx.card_id) & (ctx.tx["t"] <= ctx.opened_at)]
        flagged_rows = ctx.tx[ctx.tx["TransactionID"] == int(ctx.flagged_txn_id)]
        if not flagged_rows.empty:
            t0 = pd.Timestamp(flagged_rows["t"].iloc[0])
            ep_tx = card_tx[(card_tx["t"] >= t0 - pd.Timedelta(hours=24)) & (card_tx["t"] <= ctx.opened_at)]
            if ep_tx.empty:
                ep_tx = flagged_rows
        else:
            ep_tx = card_tx.tail(1)

        hist_tx = card_tx[~card_tx["TransactionID"].isin(ep_tx["TransactionID"])]
        t_cut_sec = _to_epoch_sec(ctx.opened_at)
        n_shared = _count_asof_shared_dev_cards(
            devs={str(d) for d in ep_tx["device_profile"].fillna("").tolist() if d},
            card_id=ctx.card_id,
            t_cut_sec=t_cut_sec,
            dev_groups=idx.dev_groups,
            t_sec_all=idx.cols["t_sec"],
            card_all=idx.cols["card"],
        )
        qv = embed_case(ep_tx, hist_tx, standardizer=std, asof_shared_dev_cards=n_shared)
        ep_tids = ep_tx["TransactionID"].astype(int).tolist()

        neighbors = idx.similar_cases_fused(
            card_id=ctx.card_id,
            qv=qv,
            k=self.k,
            hops=2,
            as_of=ctx.opened_at,
            episode_tids=ep_tids,
            exclude_case_id=ctx.case_id,
            use_tg_if_enabled=True,
        )
        p_fraud = _knn_fraud_prob(neighbors)
        if p_fraud < self.FIRE_THRESHOLD or not neighbors:
            return []

        top_pattern = pd.Series([n.pattern for n in neighbors if n.outcome == "confirmed_fraud"]).mode()
        pat = str(top_pattern.iloc[0]) if not top_pattern.empty else self.pattern
        prec_ids = [n.case_id for n in neighbors]
        shared_cards = sorted({c for n in neighbors for c in n.shared_cards if c != ctx.card_id})

        claim = Attestation(
            what=(
                f"Fused vector+traversal search on card {ctx.card_id} retrieved {len(neighbors)} "
                f"2-hop precedent cases ({', '.join(prec_ids[:3])}) with k-NN fraud probability {p_fraud:.3f}."
            ),
            evidence=Evidence(
                txn_ids=ep_tids,
                card_ids=[ctx.card_id] + shared_cards[:5],
                customer_ids=[ctx.customer_id],
                case_ids=prec_ids,
            ),
            query="similar_cases_fused",
            params={
                "c": ctx.card_id,
                "qv": [round(float(x), 4) for x in qv.tolist()],
                "k": self.k,
                "hops": 2,
                "as_of": pd.Timestamp(ctx.opened_at).strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        return [
            Finding(
                detector=self.name,
                pattern=pat,
                claim=claim,
                lr=1.59,
                affected_txn_ids=ep_tids,
                connected_card_ids=shared_cards[:5],
            )
        ]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        rep = run_ablation(cc=cc, tx=tx, k=self.k)
        caught: list[str] = []
        missed: list[str] = []
        legit_acted: list[str] = []

        for r in rep.rows:
            fired = r.p_fused >= self.FIRE_THRESHOLD
            if r.actual_fraud == 1:
                if fired:
                    caught.append(r.case_id)
                else:
                    missed.append(r.case_id)
            else:
                if fired:
                    legit_acted.append(r.case_id)

        p_fire_fraud = (len(caught) + 1.0) / (rep.n_fraud + 2.0)
        p_fire_legit = (len(legit_acted) + 1.0) / (rep.n_cleared + 2.0)
        lr = float(p_fire_fraud / p_fire_legit)

        return Backtest(
            detector=self.name,
            n_cases=rep.n_eval,
            caught=caught,
            missed=missed,
            legit_acted_against=legit_acted,
            lr_measured=lr,
        )
