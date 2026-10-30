from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Backtest, CaseCtx, Detector, Finding

FREE_MAIL = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'anonymous.com',
    'outlook.com', 'icloud.com', 'mail.com', 'live.com', 'msn.com',
    'ymail.com', 'comcast.net', 'att.net', 'verizon.net', 'sbcglobal.net',
    'bellsouth.net', 'cox.net', 'earthlink.net', 'charter.net', 'me.com', 'mac.com'
}
GENERIC_DEVICES = {'Windows', 'iOS Device', 'MacOS', 'Android', '', 'None', 'nan'}


def hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """Exact one-sided Fisher exact / hypergeometric survival P(X >= k)."""
    if k <= 0:
        return 1.0
    if k > min(n, N) or M <= 0:
        return 0.0
    den = math.comb(M, N)
    if den == 0:
        return 0.0
    prob = 0.0
    for x in range(k, min(n, N) + 1):
        if M - n < N - x:
            continue
        prob += (math.comb(n, x) * math.comb(M - n, N - x)) / den
    return min(max(prob, 0.0), 1.0)


def lr_with_ci(k_pos: int, n_pos: int, k_neg: int, n_neg: int) -> dict[str, float]:
    """Computes Laplace-smoothed LR and 95% log-normal confidence interval (Katz method)."""
    a = k_pos + 1.0
    b = max(n_pos - k_pos, 0) + 1.0
    c = k_neg + 1.0
    d = max(n_neg - k_neg, 0) + 1.0
    p_pos = a / (a + b)
    p_neg = c / (c + d)
    lr = p_pos / p_neg
    se_log_lr = math.sqrt((1.0 / a) - (1.0 / (a + b)) + (1.0 / c) - (1.0 / (c + d)))
    ci_low = math.exp(math.log(lr) - 1.96 * se_log_lr)
    ci_high = math.exp(math.log(lr) + 1.96 * se_log_lr)
    return {
        "k_pos": int(k_pos),
        "n_pos": int(n_pos),
        "k_neg": int(k_neg),
        "n_neg": int(n_neg),
        "p_pos": float(p_pos),
        "p_neg": float(p_neg),
        "lr": float(lr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _size_bucket(n: int) -> str | None:
    if n == 3:
        return "3"
    if 4 <= n <= 5:
        return "4-5"
    if n >= 6:
        return "6+"
    return None


def _is_specific_device(d: str) -> bool:
    if not d or d in GENERIC_DEVICES:
        return False
    return not (d.startswith('Windows | ?') or d.startswith('iOS Device | ?') or d.startswith('MacOS | ?'))


@dataclass
class PrecomputedIndex:
    """Lightweight, precomputed numpy/dict index over tx_all() to avoid repeated DataFrame scans."""
    t_ns: np.ndarray
    tid: np.ndarray
    card_idx: np.ndarray
    cards: np.ndarray
    card_to_idx: dict[str, int]
    prod: np.ndarray
    chan: np.ndarray
    amt: np.ndarray
    dev_specific: np.ndarray
    card_first_5th_t_ns: np.ndarray
    card_first_c250_t_ns: np.ndarray
    card_dev_first_t_ns: dict[tuple[int, str], int]
    tid_to_row: dict[int, int]
    month_card_pools: dict[str, dict[tuple[int, str], np.ndarray]]
    month_card_decile_prod: dict[str, dict[int, tuple[int, str]]]
    cap_99: float
    global_card_counts: dict[str, int]


@lru_cache(maxsize=1)
def get_precomputed_index() -> PrecomputedIndex:
    t0_wall = time.perf_counter()
    from assay.r2.frames import tx_all, closed_cases
    tx = tx_all()
    cc = closed_cases()

    t_ns = tx['t'].values.astype('datetime64[ns]').astype(np.int64)
    tid = tx['TransactionID'].values.astype(np.int64)
    prod = tx['ProductCD'].astype(str).values
    chan = tx['channel'].astype(str).values
    amt = tx['amount'].values.astype(np.float64)

    cards, card_idx = np.unique(tx['card_id'].astype(str).values, return_inverse=True)
    n_cards = len(cards)
    card_to_idx = {c: i for i, c in enumerate(cards)}
    tid_to_row = {int(t_id): i for i, t_id in enumerate(tid)}

    # Specific device strings
    raw_dev = tx['device_profile'].fillna('').astype(str).values
    uniq_devs, dev_inv = np.unique(raw_dev, return_inverse=True)
    uniq_spec = np.array([d if _is_specific_device(d) else '' for d in uniq_devs], dtype=object)
    dev_specific = uniq_spec[dev_inv]

    # Per-card 5th transaction timestamp (meaning card has >=5 prior txns when t > card_first_5th_t_ns[c])
    INF_NS = np.iinfo(np.int64).max
    card_first_5th_t_ns = np.full(n_cards, INF_NS, dtype=np.int64)
    order = np.argsort(card_idx, kind='mergesort')
    sorted_c = card_idx[order]
    sorted_t = t_ns[order]
    _, start_idx, counts = np.unique(sorted_c, return_index=True, return_counts=True)
    valid_5 = counts >= 5
    card_first_5th_t_ns[sorted_c[start_idx[valid_5]]] = sorted_t[start_idx[valid_5] + 4]

    # Per-card first timestamp of ProductCD == 'C' and amount >= 250.0
    card_first_c250_t_ns = np.full(n_cards, INF_NS, dtype=np.int64)
    c250_mask = (prod == 'C') & (amt >= 250.0)
    if c250_mask.any():
        c_sub = card_idx[c250_mask]
        t_sub = t_ns[c250_mask]
        # Since tx is sorted by t, first occurrence per card is minimum t
        _, first_pos = np.unique(c_sub, return_index=True)
        card_first_c250_t_ns[c_sub[first_pos]] = t_sub[first_pos]

    # First seen timestamp for (card_idx, specific_device)
    spec_mask = dev_specific != ''
    card_dev_first_t_ns: dict[tuple[int, str], int] = {}
    for c_i, d_s, t_val in zip(card_idx[spec_mask], dev_specific[spec_mask], t_ns[spec_mask]):
        key = (int(c_i), str(d_s))
        if key not in card_dev_first_t_ns:
            card_dev_first_t_ns[key] = int(t_val)

    # Precompute monthly card activity decile and primary ProductCD for fast control matching
    months = tx['t'].dt.strftime('%Y-%m').values
    month_card_pools: dict[str, dict[tuple[int, str], np.ndarray]] = {}
    month_card_decile_prod: dict[str, dict[int, tuple[int, str]]] = {}

    for m_str in np.unique(months):
        m_mask = months == m_str
        m_df = pd.DataFrame({'c': card_idx[m_mask], 'p': prod[m_mask]})
        cnts = m_df.groupby('c').size().rename('cnt')
        p_cnts = m_df.groupby(['c', 'p']).size().unstack(fill_value=0)
        p_mode = p_cnts.idxmax(axis=1).rename('p_mode')
        stats = pd.concat([cnts, p_mode], axis=1).reset_index()
        try:
            stats['dec'] = pd.qcut(stats['cnt'].rank(method='first'), 10, labels=False)
        except ValueError:
            stats['dec'] = 0

        pools: dict[tuple[int, str], np.ndarray] = {}
        for (dec, pm), grp in stats.groupby(['dec', 'p_mode']):
            pools[(int(dec), str(pm))] = grp['c'].values.astype(np.int64)
        for dec, grp in stats.groupby('dec'):
            pools[(int(dec), '*')] = grp['c'].values.astype(np.int64)

        month_card_pools[str(m_str)] = pools
        month_card_decile_prod[str(m_str)] = {
            int(r.c): (int(r.dec), str(r.p_mode)) for r in stats.itertuples()
        }

    # Global cardinality stats on ProductCD_amount over legit Jul-Oct 2016 (memory-safe)
    fraud_txns = {int(x) for t_list in cc.txn_list for x in t_list}
    jo_cutoff = pd.Timestamp('2016-11-01').value
    jo_mask = (t_ns < jo_cutoff) & (~np.isin(tid, list(fraud_txns)))

    amt_cents = np.round(amt * 100).astype(np.int64)
    f1_df = pd.DataFrame({
        'c': card_idx,
        'k': pd.Series(prod) + '_' + pd.Series(np.char.mod('%.2f', amt))
    })
    jo_counts = f1_df[jo_mask].groupby('k')['c'].nunique()
    cap_99 = float(jo_counts.quantile(0.99)) if not jo_counts.empty else 45.0

    all_f1_counts = f1_df.groupby('k')['c'].nunique().to_dict()
    global_card_counts = {f"ProductCD_amount:{k}": int(v) for k, v in all_f1_counts.items()}

    # Add R_emaildomain and P_emaildomain_ProductCD global counts
    r_col = tx['R_emaildomain'].fillna('').astype(str).values
    r_valid = (r_col != '') & (r_col != 'None') & (r_col != 'nan') & (~np.isin(r_col, list(FREE_MAIL)))
    if r_valid.any():
        r_df = pd.DataFrame({'c': card_idx[r_valid], 'k': r_col[r_valid]})
        for k, v in r_df.groupby('k')['c'].nunique().items():
            global_card_counts[f"R_emaildomain:{k}"] = int(v)

    addr1_vals = tx['addr1'].values
    a_valid = ~np.isnan(addr1_vals)
    if a_valid.any():
        a_df = pd.DataFrame({
            'c': card_idx[a_valid],
            'k': pd.Series(prod[a_valid]) + '_' + pd.Series(addr1_vals[a_valid].astype(int).astype(str))
        })
        for k, v in a_df.groupby('k')['c'].nunique().items():
            global_card_counts[f"ProductCD_addr1:{k}"] = int(v)

    elapsed = time.perf_counter() - t0_wall
    print(f"[CPC] Precomputed index built in {elapsed:.2f}s (cap_99={cap_99:.1f})", flush=True)
    return PrecomputedIndex(
        t_ns=t_ns,
        tid=tid,
        card_idx=card_idx,
        cards=cards,
        card_to_idx=card_to_idx,
        prod=prod,
        chan=chan,
        amt=amt,
        dev_specific=dev_specific,
        card_first_5th_t_ns=card_first_5th_t_ns,
        card_first_c250_t_ns=card_first_c250_t_ns,
        card_dev_first_t_ns=card_dev_first_t_ns,
        tid_to_row=tid_to_row,
        month_card_pools=month_card_pools,
        month_card_decile_prod=month_card_decile_prod,
        cap_99=cap_99,
        global_card_counts=global_card_counts,
    )


def extract_features_sub(tx_window: pd.DataFrame) -> pd.DataFrame:
    """Extracts candidate origin features on a small subset of transactions (cohort + matched controls)."""
    if tx_window.empty:
        return pd.DataFrame(columns=['card_id', 'TransactionID', 'feature'])

    c_id = tx_window['card_id'].values
    t_id = tx_window['TransactionID'].values
    prod = tx_window['ProductCD'].astype(str).values
    amt = np.char.mod('%.2f', tx_window['amount'].values)

    frames = [
        pd.DataFrame({
            'card_id': c_id,
            'TransactionID': t_id,
            'feature': 'ProductCD_amount:' + pd.Series(prod) + '_' + pd.Series(amt)
        })
    ]

    valid_addr1 = tx_window['addr1'].notna().values
    if valid_addr1.any():
        addr1_str = tx_window['addr1'].values[valid_addr1].astype(int).astype(str)
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_addr1],
            'TransactionID': t_id[valid_addr1],
            'feature': 'ProductCD_addr1:' + pd.Series(prod[valid_addr1]) + '_' + pd.Series(addr1_str)
        }))

    r_col = tx_window['R_emaildomain']
    valid_r = (r_col.notna() & (~r_col.isin(FREE_MAIL)) & (r_col != 'None') & (r_col != 'nan') & (r_col != '')).values
    if valid_r.any():
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_r],
            'TransactionID': t_id[valid_r],
            'feature': 'R_emaildomain:' + r_col.values[valid_r].astype(str)
        }))

    p_col = tx_window['P_emaildomain']
    valid_p = (p_col.notna() & (~p_col.isin(FREE_MAIL)) & (p_col != 'None') & (p_col != 'nan') & (p_col != '')).values
    if valid_p.any():
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_p],
            'TransactionID': t_id[valid_p],
            'feature': 'P_emaildomain_ProductCD:' + pd.Series(p_col.values[valid_p].astype(str)) + '_' + pd.Series(prod[valid_p])
        }))

    d_col = tx_window['device_profile']
    valid_d = (d_col.notna() & (~d_col.isin(GENERIC_DEVICES))).values
    if valid_d.any():
        d_vals = d_col.values[valid_d].astype(str)
        non_triv = np.array([_is_specific_device(v) for v in d_vals], dtype=bool)
        if non_triv.any():
            idx = np.where(valid_d)[0][non_triv]
            frames.append(pd.DataFrame({
                'card_id': c_id[idx],
                'TransactionID': t_id[idx],
                'feature': 'device_profile:' + d_vals[non_triv]
            }))

    valid_dist = tx_window['dist1'].notna().values
    if valid_dist.any():
        dist_str = tx_window['dist1'].values[valid_dist].astype(int).astype(str)
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_dist],
            'TransactionID': t_id[valid_dist],
            'feature': 'merchant_like:' + pd.Series(prod[valid_dist]) + '_' + pd.Series(amt[valid_dist]) + '_' + pd.Series(dist_str)
        }))

    return pd.concat(frames, ignore_index=True)


def fast_find_cohort_by_tid(idx: PrecomputedIndex, flagged_txn_id: int, opened_at_ns: int) -> tuple[dict | None, list[str], dict[str, int], dict[str, list[int]]]:
    """O(log N) binary-search loss-shape cohort finder using precomputed arrays."""
    row_i = idx.tid_to_row.get(int(flagged_txn_id))
    if row_i is None:
        return None, [], {}, {}

    t0_ns = int(idx.t_ns[row_i])
    if t0_ns > opened_at_ns:
        return None, [], {}, {}

    flagged_c_idx = int(idx.card_idx[row_i])
    flagged_card = str(idx.cards[flagged_c_idx])
    prod = str(idx.prod[row_i])
    chan = str(idx.chan[row_i])

    DAY_NS = 86_400_000_000_000
    near_lo = int(np.searchsorted(idx.t_ns, t0_ns - 3 * DAY_NS, side='left'))
    near_hi = int(np.searchsorted(idx.t_ns, min(t0_ns + 3 * DAY_NS, opened_at_ns), side='right'))

    c_mask = (idx.card_idx[near_lo:near_hi] == flagged_c_idx) & (idx.prod[near_lo:near_hi] == prod) & (idx.chan[near_lo:near_hi] == chan)
    near_amts = idx.amt[near_lo:near_hi][c_mask]

    if prod == 'C' and chan == 'online' and ((near_amts >= 250.0) & (near_amts <= 2200.0)).sum() >= 2:
        shape = {
            'mode': 'c_online_burst',
            'ProductCD': 'C',
            'channel': 'online',
            'min_amt': 250.0,
            'max_amt': 2200.0,
            'min_count': 2,
            'max_count': 5,
            'prior_cap': 250.0,
            't0_ns': t0_ns,
            'flagged_card': flagged_card,
            'flagged_c_idx': flagged_c_idx,
        }
    else:
        if len(near_amts) < 2:
            return None, [], {}, {}
        min_a = float(near_amts.min()) * 0.65
        max_a = float(near_amts.max()) * 1.5
        if min_a < 250.0 or prod != 'C' or chan != 'online':
            return None, [], {}, {}
        shape = {
            'mode': 'c_online_burst',
            'ProductCD': 'C',
            'channel': 'online',
            'min_amt': max(250.0, min_a),
            'max_amt': min(2200.0, max_a),
            'min_count': 2,
            'max_count': 5,
            'prior_cap': 250.0,
            't0_ns': t0_ns,
            'flagged_card': flagged_card,
            'flagged_c_idx': flagged_c_idx,
        }

    # Search [-15d, min(t0 + 10d, opened_at_ns)]
    w_lo = int(np.searchsorted(idx.t_ns, t0_ns - 15 * DAY_NS, side='left'))
    w_hi = int(np.searchsorted(idx.t_ns, min(t0_ns + 10 * DAY_NS, opened_at_ns), side='right'))
    if w_hi <= w_lo:
        return shape, [], {}, {}

    sub_p = idx.prod[w_lo:w_hi]
    sub_ch = idx.chan[w_lo:w_hi]
    sub_a = idx.amt[w_lo:w_hi]
    m = (sub_p == shape['ProductCD']) & (sub_ch == shape['channel']) & (sub_a >= shape['min_amt']) & (sub_a <= shape['max_amt'])
    if not m.any():
        return shape, [], {}, {}

    hit_rows = np.where(m)[0] + w_lo
    hit_c = idx.card_idx[hit_rows]
    uniq_c, c_counts = np.unique(hit_c, return_counts=True)

    cand_c = uniq_c[(c_counts >= shape['min_count']) & (c_counts <= shape['max_count'])].tolist()
    if flagged_c_idx in uniq_c and flagged_c_idx not in cand_c:
        if c_counts[uniq_c == flagged_c_idx][0] >= 2:
            cand_c.append(flagged_c_idx)

    if len(cand_c) < 3:
        return shape, [], {}, {}

    cohort: list[str] = []
    first_loss_ns: dict[str, int] = {}
    burst_txns: dict[str, list[int]] = {}
    burst_devs: dict[str, set[str]] = {}

    SPAN_MAX_NS = int(4.0 * DAY_NS)

    for c_i in cand_c:
        c_rows = hit_rows[hit_c == c_i]
        c_ts = idx.t_ns[c_rows]
        if c_ts[-1] - c_ts[0] > SPAN_MAX_NS:
            found_sub = False
            for j in range(len(c_rows) - shape['min_count'] + 1):
                sub_rows = c_rows[j:j + shape['min_count']]
                if idx.t_ns[sub_rows[-1]] - idx.t_ns[sub_rows[0]] <= SPAN_MAX_NS:
                    c_rows = sub_rows
                    c_ts = idx.t_ns[c_rows]
                    found_sub = True
                    break
            if not found_sub:
                continue

        t_burst_start = int(c_ts[0])

        # Must have >= 5 prior transactions (so 5th transaction timestamp < t_burst_start)
        if idx.card_first_5th_t_ns[c_i] >= t_burst_start:
            continue

        # Must have NO prior C charges >= $250 before t_burst_start
        if idx.card_first_c250_t_ns[c_i] < t_burst_start:
            continue

        # Check novel device: none of the specific devices in c_rows were seen on c_i before t_burst_start
        devs_in_burst = {str(d) for d in idx.dev_specific[c_rows] if d != ''}
        used_before = False
        for d in devs_in_burst:
            first_seen = idx.card_dev_first_t_ns.get((int(c_i), d))
            if first_seen is not None and first_seen < t_burst_start:
                used_before = True
                break
        if used_before:
            continue

        c_str = str(idx.cards[c_i])
        cohort.append(c_str)
        first_loss_ns[c_str] = t_burst_start
        burst_txns[c_str] = [int(x) for x in idx.tid[c_rows]]
        burst_devs[c_str] = devs_in_burst

    # Prune shared devices across cohort
    dev_to_cards: dict[str, list[str]] = {}
    for c_str in cohort:
        for d in burst_devs.get(c_str, set()):
            dev_to_cards.setdefault(d, []).append(c_str)

    shared_cards = set()
    for d, cs in dev_to_cards.items():
        if len(cs) > 1:
            for c_str in cs:
                if c_str != flagged_card:
                    shared_cards.add(c_str)

    cohort = sorted([c_str for c_str in cohort if c_str not in shared_cards])
    first_loss_ns = {c_str: first_loss_ns[c_str] for c_str in cohort}
    burst_txns = {c_str: burst_txns[c_str] for c_str in cohort}
    return shape, cohort, first_loss_ns, burst_txns


def fast_matched_controls(idx: PrecomputedIndex, affected_cards: list[str], t0_ns: int, rng_seed: int = 42) -> dict[str, list[str]]:
    """Fast lookup of 20 activity-matched control cards per affected card using precomputed monthly decile pools."""
    rng = np.random.default_rng(rng_seed)
    m_str = pd.Timestamp(t0_ns).strftime('%Y-%m')
    if m_str not in idx.month_card_pools:
        m_str = sorted(idx.month_card_pools.keys())[-1]

    pools = idx.month_card_pools[m_str]
    dec_map = idx.month_card_decile_prod[m_str]

    aff_indices = {idx.card_to_idx[c] for c in affected_cards if c in idx.card_to_idx}
    used = set(aff_indices)
    controls: dict[str, list[str]] = {}

    for c_str in affected_cards:
        c_i = idx.card_to_idx.get(c_str, -1)
        dec, pm = dec_map.get(c_i, (5, 'W'))
        cand = pools.get((dec, pm), np.array([], dtype=np.int64))
        avail = [int(x) for x in cand if int(x) not in used]
        if len(avail) < 20:
            cand2 = pools.get((dec, '*'), np.array([], dtype=np.int64))
            avail = [int(x) for x in cand2 if int(x) not in used]
        if len(avail) >= 20:
            chosen = rng.choice(avail, size=20, replace=False).tolist()
        else:
            chosen = avail
        used.update(chosen)
        controls[c_str] = [str(idx.cards[x]) for x in chosen]

    return controls


def compute_origin_lifts(
    idx: PrecomputedIndex,
    affected_cards: list[str],
    first_loss_ns: dict[str, int],
    controls: dict[str, list[str]],
    as_of_tx: pd.DataFrame,
    lookback_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes matched control lift and Fisher exact p-value for candidate origins prior to each card's first loss."""
    all_ctrl_cards = {ctrl_c for c_list in controls.values() for ctrl_c in c_list}
    target_cards = set(affected_cards) | all_ctrl_cards
    sub_tx = as_of_tx[as_of_tx.card_id.isin(target_cards)]
    if sub_tx.empty:
        return pd.DataFrame(), pd.DataFrame(columns=['card_id', 'TransactionID', 'feature'])

    feats_all = extract_features_sub(sub_tx)
    # Attach transaction timestamp for per-card [t_loss - lookback, t_loss) filtering
    tid_to_t = sub_tx.set_index('TransactionID')['t']
    feats_all['t'] = feats_all['TransactionID'].map(tid_to_t)

    aff_frames = []
    ctrl_frames = []

    for c in affected_cards:
        t_loss = pd.Timestamp(first_loss_ns[c])
        w_start = t_loss - pd.Timedelta(days=lookback_days)
        c_f = feats_all[(feats_all.card_id == c) & (feats_all.t >= w_start) & (feats_all.t < t_loss)]
        if not c_f.empty:
            aff_frames.append(c_f[['card_id', 'TransactionID', 'feature']])

        c_ctrls = controls.get(c, [])
        if c_ctrls:
            ctrl_f = feats_all[(feats_all.card_id.isin(c_ctrls)) & (feats_all.t >= w_start) & (feats_all.t < t_loss)]
            if not ctrl_f.empty:
                ctrl_frames.append(ctrl_f[['card_id', 'TransactionID', 'feature']])

    aff_feats = pd.concat(aff_frames, ignore_index=True) if aff_frames else pd.DataFrame(columns=['card_id', 'TransactionID', 'feature'])
    ctrl_feats = pd.concat(ctrl_frames, ignore_index=True) if ctrl_frames else pd.DataFrame(columns=['card_id', 'TransactionID', 'feature'])

    if aff_feats.empty:
        return pd.DataFrame(), aff_feats

    aff_counts = aff_feats.groupby('feature')['card_id'].nunique()
    ctrl_counts = ctrl_feats.groupby('feature')['card_id'].nunique() if not ctrl_feats.empty else pd.Series(dtype=int)

    N_aff = len(affected_cards)
    N_ctrl = max(len(all_ctrl_cards), 1)

    rows = []
    for feat, k_aff in aff_counts.items():
        if k_aff < 2:
            continue
        k_ctrl = int(ctrl_counts.get(feat, 0))
        share_aff = k_aff / N_aff
        share_ctrl = max(k_ctrl, 0.5) / N_ctrl
        lift = share_aff / share_ctrl
        pval = hypergeom_sf(int(k_aff), N_aff + N_ctrl, int(k_aff) + k_ctrl, N_aff)
        card_glob = int(idx.global_card_counts.get(feat, k_aff + k_ctrl))
        rows.append({
            'feature': feat,
            'k_aff': int(k_aff),
            'k_ctrl': k_ctrl,
            'lift': float(lift),
            'pval': float(pval),
            'cardinality': card_glob
        })

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(['pval', 'lift'], ascending=[True, False]).reset_index(drop=True)
    return res_df, aff_feats


class CommonPointOfCompromise(Detector):
    name = "cpc"
    pattern = "undocumented:common_point_of_compromise"
    _cached_bt: Backtest | None = None
    _cached_stratified_lr: dict[str, dict[str, float]] | None = None
    last_diagnostics: dict[str, dict] = {}

    def get_stratified_lr_table(self) -> dict[str, dict[str, float]]:
        if CommonPointOfCompromise._cached_stratified_lr is not None:
            return CommonPointOfCompromise._cached_stratified_lr
        from assay.r2.frames import closed_cases, tx_all
        self.backtest(closed_cases(), tx_all())
        return CommonPointOfCompromise._cached_stratified_lr or {}

    def run(self, ctx: CaseCtx) -> list[Finding]:
        idx = get_precomputed_index()
        opened_ns = int(pd.Timestamp(ctx.opened_at).value)

        # under ASSAY_TG=1 the graph finds the cohort and counts the origins (graph/queries/r2_cpc.gsql, fails
        # closed); the statistics stay here. Parity with the local path: docs/r2/C10_tg_cpc.md
        on_tg = os.environ.get("ASSAY_TG") == "1"
        if on_tg:
            from assay.r2.tg_cpc import compute_origin_lifts_tg, find_cohort_by_tid_tg
            find_cohort = find_cohort_by_tid_tg
            origin_lifts = lambda *a, **k: compute_origin_lifts_tg(*a, opened_at=ctx.opened_at, **k)  # noqa: E731
        else:
            find_cohort, origin_lifts = fast_find_cohort_by_tid, compute_origin_lifts
        backend = "tigergraph" if on_tg else "local"
        shape, affected_cards, first_loss_ns, burst_txns = find_cohort(idx, int(ctx.flagged_txn_id), opened_ns)
        if shape is None:
            CommonPointOfCompromise.last_diagnostics[ctx.case_id] = {
                "status": "no_loss_shape",
                "cohort_size": 0,
                "cohort_cards": [],
            }
            return []

        cohort_size = len(affected_cards)
        bucket = _size_bucket(cohort_size)

        if cohort_size < 3 or ctx.card_id not in affected_cards:
            CommonPointOfCompromise.last_diagnostics[ctx.case_id] = {
                "status": "cohort_too_small",
                "cohort_size": cohort_size,
                "cohort_cards": affected_cards,
                "min_allowed_size": 3,
                "bucket_lr": 1.0,
                "bucket_ci_low": 0.0,
            }
            return []

        strat_table = self.get_stratified_lr_table()
        bucket_stats = strat_table.get(bucket, {}) if bucket else {}
        lr_val = float(bucket_stats.get("lr", 1.0))
        ci_low = float(bucket_stats.get("ci_low", 0.0))

        size_3_allowed = strat_table.get("3", {}).get("ci_low", 0.0) > 2.0
        min_allowed_size = 3 if size_3_allowed else 4

        if cohort_size < min_allowed_size or ctx.card_id not in affected_cards:
            CommonPointOfCompromise.last_diagnostics[ctx.case_id] = {
                "status": "cohort_too_small",
                "cohort_size": cohort_size,
                "cohort_cards": affected_cards,
                "min_allowed_size": min_allowed_size,
                "bucket_lr": lr_val,
                "bucket_ci_low": ci_low,
            }
            return []

        controls = fast_matched_controls(idx, affected_cards, shape['t0_ns'], rng_seed=42)
        res_60, aff_feats_60 = origin_lifts(idx, affected_cards, first_loss_ns, controls, ctx.tx, lookback_days=60)

        cap = idx.cap_99

        min_k_origin = min(4, cohort_size) if cohort_size >= 3 else 4
        valid = pd.DataFrame()
        if not res_60.empty:
            valid = res_60[
                (res_60.k_aff >= min_k_origin) &
                (res_60.lift >= 5.0) &
                (res_60.pval < 0.001) &
                (res_60.cardinality <= cap)
            ]

        flagged_burst_txns = burst_txns.get(ctx.card_id, [ctx.flagged_txn_id])
        all_cohort_burst_txns = sorted({tid for tids in burst_txns.values() for tid in tids})

        if valid.empty:
            best_desc = "none"
            if not res_60.empty:
                b = res_60.iloc[0]
                best_desc = f"{b.feature} (lift={b.lift:.2f}x, p={b.pval:.2e}, cardinality={b.cardinality})"
            CommonPointOfCompromise.last_diagnostics[ctx.case_id] = {
                "status": "coordinated_cohort_no_origin",
                "cohort_size": cohort_size,
                "cohort_cards": affected_cards,
                "bucket": bucket,
                "lr": lr_val,
                "ci_low": ci_low,
                "best_candidate": best_desc,
            }

            claim = Attestation(
                what=(
                    f"Coordinated loss-shape cohort across {cohort_size} cards (bucket '{bucket}', measured LR={lr_val:.2f}, 95% CI low={ci_low:.2f}: "
                    f"2-3 online ProductCD {shape['ProductCD']} charges on novel devices with no shared device); "
                    f"no common origin found above matched-control threshold (best candidate: {best_desc})."
                ),
                evidence=Evidence(
                    txn_ids=all_cohort_burst_txns,
                    card_ids=affected_cards,
                    customer_ids=[ctx.customer_id],
                ),
                query="r2_cpc_loss_cohort" if on_tg else "cpc_loss_shape_cohort",
                params={
                    "card_id": ctx.card_id,
                    "flagged_txn_id": int(ctx.flagged_txn_id),
                    "ProductCD": shape['ProductCD'],
                    "channel": shape['channel'],
                    "min_amt": float(shape['min_amt']),
                    "max_amt": float(shape['max_amt']),
                    "cohort_size_bucket": bucket,
                    "lookback_days": 60,
                    "backend": backend,
                },
            )
            return [Finding(
                detector=self.name,
                pattern="undocumented:coordinated_cnp_burst_no_shared_device",
                claim=claim,
                lr=ci_low,  # weigh by the 95% CI lower bound: the point LR rests on very few positive clusters
                affected_txn_ids=flagged_burst_txns,
                connected_card_ids=[c for c in affected_cards if c != ctx.card_id],
            )]

        best = valid.iloc[0]
        best_feat = str(best.feature)
        origin_txns = sorted(aff_feats_60[aff_feats_60.feature == best_feat]['TransactionID'].astype(int).unique().tolist())
        evidence_txns = sorted(set(flagged_burst_txns) | set(origin_txns))

        CommonPointOfCompromise.last_diagnostics[ctx.case_id] = {
            "status": "common_origin_found",
            "cohort_size": cohort_size,
            "cohort_cards": affected_cards,
            "bucket": bucket,
            "lr": lr_val,
            "ci_low": ci_low,
            "origin": best_feat,
        }

        claim = Attestation(
            what=(
                f"Affected loss-shape cohort ({cohort_size} cards, bucket '{bucket}', measured LR={lr_val:.2f}) shares common origin {best_feat} "
                f"within 60 days before loss (matched lift={best.lift:.1f}x, Fisher p={best.pval:.2e}, cardinality={int(best.cardinality)})."
            ),
            evidence=Evidence(
                txn_ids=evidence_txns,
                card_ids=affected_cards,
                customer_ids=[ctx.customer_id],
            ),
            query="r2_cpc_origin_counts" if on_tg else "cpc_origin_search",
            params={
                "card_id": ctx.card_id,
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "origin_key": best_feat,
                "cohort_size_bucket": bucket,
                "lookback_days": 60,
                "backend": backend,
            },
        )
        return [Finding(
            detector=self.name,
            pattern=self.pattern,
            claim=claim,
            lr=ci_low,  # weigh by the 95% CI lower bound: the point LR rests on very few positive clusters
            affected_txn_ids=flagged_burst_txns,
            connected_card_ids=[c for c in affected_cards if c != ctx.card_id],
        )]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame, max_samples: int = 300) -> Backtest:
        """Fast backtest stratified by cohort size ('3', '4-5', '6+') using precomputed binary-search index."""
        if max_samples == 300 and CommonPointOfCompromise._cached_bt is not None and CommonPointOfCompromise._cached_stratified_lr is not None:
            return CommonPointOfCompromise._cached_bt

        t0_bt = time.perf_counter()
        idx = get_precomputed_index()

        # Positive episodes: confirmed fraud cases where the card has >=2 online C charges >= $250 in the case
        fraud_cc = cc[cc.outcome == 'confirmed_fraud']
        pos_cases = []
        for row in fraud_cc.itertuples():
            if len(row.txn_list) < 2:
                continue
            r_indices = [idx.tid_to_row[tid] for tid in row.txn_list if tid in idx.tid_to_row]
            if len(r_indices) < 2:
                continue
            c_hits = sum(1 for r_i in r_indices if idx.prod[r_i] == 'C' and idx.chan[r_i] == 'online' and idx.amt[r_i] >= 250.0)
            if c_hits >= 2:
                pos_cases.append((str(row.case_id), int(row.txn_list[-1]), int(pd.Timestamp(row.opened_at).value)))

        rng = np.random.default_rng(42)
        if len(pos_cases) > max_samples:
            chosen_idx = rng.choice(len(pos_cases), size=max_samples, replace=False)
            pos_cases = [pos_cases[i] for i in sorted(chosen_idx)]

        pos_bucket_hits: dict[str, list[str]] = {"3": [], "4-5": [], "6+": []}
        pos_cohort_sizes: dict[str, int] = {}

        for case_id, f_tid, opened_ns in pos_cases:
            shape, cohort, _, _ = fast_find_cohort_by_tid(idx, f_tid, opened_ns)
            if shape is None or shape['flagged_card'] not in cohort:
                pos_cohort_sizes[case_id] = 0
                continue
            c_size = len(cohort)
            pos_cohort_sizes[case_id] = c_size
            b = _size_bucket(c_size)
            if b is not None:
                pos_bucket_hits[b].append(case_id)

        # Negative population: fixed-seed sample of legitimate Jul-Oct 2016 card-days (default 5,000, fast via precomputed index)
        fraud_txns = {int(x) for t_list in cc.txn_list for x in t_list}
        jo_cutoff = pd.Timestamp('2016-11-01').value
        legit_rows = np.where((idx.t_ns < jo_cutoff) & (~np.isin(idx.tid, list(fraud_txns))))[0]
        n_neg_target = 5000 if max_samples >= 300 else max_samples
        sampled_neg_rows = rng.choice(legit_rows, size=min(n_neg_target, len(legit_rows)), replace=False)

        neg_bucket_hits: dict[str, list[str]] = {"3": [], "4-5": [], "6+": []}

        for r_i in sampled_neg_rows:
            if idx.prod[r_i] != 'C' or idx.chan[r_i] != 'online' or idx.amt[r_i] < 250.0:
                continue
            f_tid = int(idx.tid[r_i])
            opened_ns = int(idx.t_ns[r_i])
            shape, cohort, _, _ = fast_find_cohort_by_tid(idx, f_tid, opened_ns)
            if shape is None or shape['flagged_card'] not in cohort:
                continue
            b = _size_bucket(len(cohort))
            if b is not None:
                neg_bucket_hits[b].append(str(f_tid))

        n_pos = max(len(pos_cases), 1)
        n_neg = len(sampled_neg_rows)

        strat_table: dict[str, dict[str, float]] = {}
        for b in ("3", "4-5", "6+"):
            strat_table[b] = lr_with_ci(len(pos_bucket_hits[b]), n_pos, len(neg_bucket_hits[b]), n_neg)

        CommonPointOfCompromise._cached_stratified_lr = strat_table

        size_3_active = strat_table["3"]["ci_low"] > 2.0
        active_buckets = ("3", "4-5", "6+") if size_3_active else ("4-5", "6+")

        caught = []
        missed = []
        for case_id, _, _ in pos_cases:
            c_size = pos_cohort_sizes.get(case_id, 0)
            b = _size_bucket(c_size)
            if b in active_buckets:
                caught.append(case_id)
            else:
                missed.append(case_id)

        legit_acted = []
        for b in active_buckets:
            legit_acted.extend(neg_bucket_hits[b])

        overall_stats = lr_with_ci(len(caught), n_pos, len(legit_acted), n_neg)
        bt = Backtest(
            detector=self.name,
            n_cases=n_pos + n_neg,
            caught=caught,
            missed=missed,
            legit_acted_against=legit_acted,
            lr_measured=overall_stats["lr"],
        )
        if max_samples == 300:
            CommonPointOfCompromise._cached_bt = bt
        print(f"[CPC] Backtest completed in {time.perf_counter() - t0_bt:.2f}s (n_pos={n_pos}, n_neg={n_neg}, LR={bt.lr_measured:.2f})", flush=True)
        return bt


CPC_QUERY_NAMES = (
    "cpc_loss_shape_cohort",
    "r2_cpc_loss_cohort",
    "cpc_origin_search",
    "r2_cpc_origin_counts",
)


def rerun(params: dict, query: str | None = None) -> list[int]:
    """Re-runnable attestation query entrypoint for CPC queries (`cpc_loss_shape_cohort` /
    `r2_cpc_loss_cohort` and `cpc_origin_search` / `r2_cpc_origin_counts`).
    """
    q = query or params.get("query")
    if q is not None and str(q) not in CPC_QUERY_NAMES:
        return []
    from assay.r2.frames import case_pack, closed_cases, tx_all

    tx = tx_all()
    cc = closed_cases()
    card_id = str(params["card_id"])
    flagged_txn_id = int(params["flagged_txn_id"])
    if "as_of" in params or "opened_at" in params:
        as_of = pd.Timestamp(params.get("as_of") or params.get("opened_at"))
    else:
        cp = case_pack()
        hit = cp.loc[cp["flagged_txn_id"] == flagged_txn_id, "opened_at"]
        if not hit.empty:
            as_of = pd.Timestamp(hit.iat[0])
        else:
            t_hit = tx.loc[tx["TransactionID"] == flagged_txn_id, "t"]
            as_of = pd.Timestamp(t_hit.iat[0]) if not t_hit.empty else pd.Timestamp.max

    cust_rows = tx.loc[tx["card_id"] == card_id, "customer_id"]
    customer_id = str(cust_rows.iat[0]) if not cust_rows.empty else ""
    ctx = CaseCtx(
        case_id=str(params.get("case_id", "RERUN")),
        flagged_txn_id=flagged_txn_id,
        card_id=card_id,
        customer_id=customer_id,
        opened_at=as_of,
        tx=tx[tx["t"] <= as_of],
        cc=cc[pd.to_datetime(cc["closed_at"]) <= as_of],
    )
    findings = CommonPointOfCompromise().run(ctx)
    if not findings:
        return []
    return [int(x) for x in findings[0].claim.evidence.txn_ids]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPC detector backtest")
    parser.add_argument("--full", action="store_true", help="Run full backtest over all closed cases")
    args = parser.parse_args()

    from assay.r2.frames import closed_cases, tx_all
    det = CommonPointOfCompromise()
    bt_res = det.backtest(closed_cases(), tx_all(), max_samples=10000 if args.full else 300)
    out = {
        "n_cases": bt_res.n_cases,
        "caught": len(bt_res.caught),
        "missed": len(bt_res.missed),
        "legit_acted_against": len(bt_res.legit_acted_against),
        "lr_measured": bt_res.lr_measured,
        "stratified": det.get_stratified_lr_table(),
    }
    print(json.dumps(out, indent=2), flush=True)
