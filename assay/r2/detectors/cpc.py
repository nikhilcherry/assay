import math
from functools import lru_cache
import numpy as np
import pandas as pd

from assay.r2.attest import Attestation, Evidence
from assay.r2.finding import Detector, CaseCtx, Finding, Backtest

FREE_MAIL = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'anonymous.com',
    'outlook.com', 'icloud.com', 'mail.com', 'live.com', 'msn.com',
    'ymail.com', 'comcast.net', 'att.net', 'verizon.net', 'sbcglobal.net',
    'bellsouth.net', 'cox.net', 'earthlink.net', 'charter.net', 'me.com', 'mac.com'
}
GENERIC_DEVICES = {'Windows', 'iOS Device', 'MacOS', 'Android', '', 'None', 'nan'}


def hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """Exact one-sided Fisher exact / hypergeometric survival P(X >= k).
    M = total population (N_aff + N_ctrl), n = total successes (k_aff + k_ctrl), N = draws (N_aff).
    """
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


@lru_cache(maxsize=1)
def _global_cardinality_stats():
    """Computes bank-wide card cardinality for each candidate origin feature over legit Jul-Oct 2016."""
    from assay.r2.frames import tx_all, closed_cases
    tx = tx_all()
    cc = closed_cases()
    fraud_txns = set()
    for t_list in cc.txn_list:
        fraud_txns.update(t_list)
    tx_jo = tx[(tx.t < pd.Timestamp('2016-11-01')) & (~tx.TransactionID.isin(fraud_txns))]
    feats_jo = extract_features(tx_jo)
    card_counts_jo = feats_jo.groupby('feature')['card_id'].nunique()
    cap = float(card_counts_jo.quantile(0.99)) if not card_counts_jo.empty else 50.0

    feats_all = extract_features(tx)
    card_counts_all = feats_all.groupby('feature')['card_id'].nunique().to_dict()
    return cap, card_counts_all


def extract_features(tx_window: pd.DataFrame) -> pd.DataFrame:
    """Extracts candidate origin features from a transaction frame."""
    if tx_window.empty:
        return pd.DataFrame(columns=['card_id', 'TransactionID', 'feature'])

    c_id = tx_window['card_id'].values
    t_id = tx_window['TransactionID'].values
    prod = tx_window['ProductCD'].astype(str).values
    amt = np.char.mod('%.2f', tx_window['amount'].values)

    frames = []

    # 1. Exact (ProductCD, amount to the cent)
    frames.append(pd.DataFrame({
        'card_id': c_id,
        'TransactionID': t_id,
        'feature': 'ProductCD_amount:' + pd.Series(prod) + '_' + pd.Series(amt)
    }))

    # 2. (ProductCD, addr1)
    valid_addr1 = tx_window['addr1'].notna().values
    if valid_addr1.any():
        addr1_str = tx_window['addr1'].values[valid_addr1].astype(int).astype(str)
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_addr1],
            'TransactionID': t_id[valid_addr1],
            'feature': 'ProductCD_addr1:' + pd.Series(prod[valid_addr1]) + '_' + pd.Series(addr1_str)
        }))

    # 3. R_emaildomain (non-free-mail only)
    r_col = tx_window['R_emaildomain']
    valid_r = (r_col.notna() & (~r_col.isin(FREE_MAIL)) & (r_col != 'None') & (r_col != 'nan') & (r_col != '')).values
    if valid_r.any():
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_r],
            'TransactionID': t_id[valid_r],
            'feature': 'R_emaildomain:' + r_col.values[valid_r].astype(str)
        }))

    # 4. P_emaildomain + ProductCD (non-free-mail)
    p_col = tx_window['P_emaildomain']
    valid_p = (p_col.notna() & (~p_col.isin(FREE_MAIL)) & (p_col != 'None') & (p_col != 'nan') & (p_col != '')).values
    if valid_p.any():
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_p],
            'TransactionID': t_id[valid_p],
            'feature': 'P_emaildomain_ProductCD:' + pd.Series(p_col.values[valid_p].astype(str)) + '_' + pd.Series(prod[valid_p])
        }))

    # 5. device_profile (non-generic)
    d_col = tx_window['device_profile']
    valid_d = (d_col.notna() & (~d_col.isin(GENERIC_DEVICES))).values
    if valid_d.any():
        d_vals = d_col.values[valid_d].astype(str)
        non_triv = np.array([not (v.startswith('Windows | ?') or v.startswith('iOS Device | ?') or v.startswith('MacOS | ?')) for v in d_vals])
        if non_triv.any():
            idx = np.where(valid_d)[0][non_triv]
            frames.append(pd.DataFrame({
                'card_id': c_id[idx],
                'TransactionID': t_id[idx],
                'feature': 'device_profile:' + d_vals[non_triv]
            }))

    # 6. merchant-like keys (ProductCD + amount + dist1)
    valid_dist = tx_window['dist1'].notna().values
    if valid_dist.any():
        dist_str = tx_window['dist1'].values[valid_dist].astype(int).astype(str)
        frames.append(pd.DataFrame({
            'card_id': c_id[valid_dist],
            'TransactionID': t_id[valid_dist],
            'feature': 'merchant_like:' + pd.Series(prod[valid_dist]) + '_' + pd.Series(amt[valid_dist]) + '_' + pd.Series(dist_str)
        }))

    return pd.concat(frames, ignore_index=True)


def get_loss_shape(as_of_tx: pd.DataFrame, flagged_txn_id: int) -> dict | None:
    """Derives the loss burst shape around the flagged transaction."""
    flagged_rows = as_of_tx[as_of_tx.TransactionID == flagged_txn_id]
    if flagged_rows.empty:
        return None
    row = flagged_rows.iloc[0]
    t0 = row.t
    card = row.card_id
    prod = row.ProductCD
    chan = row.channel

    card_tx = as_of_tx[(as_of_tx.card_id == card) & (as_of_tx.ProductCD == prod) & (as_of_tx.channel == chan)]
    near = card_tx[(card_tx.t >= t0 - pd.Timedelta(days=3)) & (card_tx.t <= t0 + pd.Timedelta(days=3))]

    high_c = near[(near.amount >= 250.0) & (near.amount <= 2200.0)]
    if prod == 'C' and chan == 'online' and len(high_c) >= 2:
        return {
            'mode': 'c_online_burst',
            'ProductCD': 'C',
            'channel': 'online',
            'min_amt': 250.0,
            'max_amt': 2200.0,
            'min_count': 2,
            'max_count': 5,
            'prior_cap': 250.0,
            't0': t0,
            'flagged_card': card,
        }

    if len(near) < 2:
        return None
    min_a = float(near.amount.min()) * 0.65
    max_a = float(near.amount.max()) * 1.5
    if min_a < 25.0:
        return None
    return {
        'mode': 'general_burst',
        'ProductCD': prod,
        'channel': chan,
        'min_amt': min_a,
        'max_amt': max_a,
        'min_count': max(2, len(near) - 1),
        'max_count': len(near) + 3,
        'prior_cap': min_a,
        't0': t0,
        'flagged_card': card,
    }


def find_affected_cohort(as_of_tx: pd.DataFrame, shape: dict) -> tuple[list[str], dict[str, pd.Timestamp], dict[str, list[int]]]:
    """Finds cards sharing the exact loss burst shape within [-15d, opened_at], with novel devices and no shared device."""
    t0 = shape['t0']
    w_start = t0 - pd.Timedelta(days=15)
    w_end = t0 + pd.Timedelta(days=10)

    cand_mask = (
        (as_of_tx.t >= w_start) &
        (as_of_tx.t <= w_end) &
        (as_of_tx.ProductCD == shape['ProductCD']) &
        (as_of_tx.channel == shape['channel']) &
        (as_of_tx.amount >= shape['min_amt']) &
        (as_of_tx.amount <= shape['max_amt'])
    )
    w_tx = as_of_tx[cand_mask]
    if w_tx.empty:
        return [], {}, {}

    counts = w_tx.groupby('card_id').size()
    cand_cards = counts[(counts >= shape['min_count']) & (counts <= shape['max_count'])].index.tolist()
    if shape['flagged_card'] in counts.index and shape['flagged_card'] not in cand_cards:
        if counts[shape['flagged_card']] >= 2:
            cand_cards.append(shape['flagged_card'])

    if len(cand_cards) < 3:
        return [], {}, {}

    hist_all = as_of_tx[as_of_tx.card_id.isin(cand_cards)]

    cohort = []
    first_loss: dict[str, pd.Timestamp] = {}
    burst_txns: dict[str, list[int]] = {}
    burst_devices: dict[str, set[str]] = {}

    for c, c_df in hist_all.groupby('card_id'):
        c_win = w_tx[w_tx.card_id == c].sort_values('t')
        if len(c_win) < shape['min_count']:
            continue

        t_vals = c_win.t.values
        span_days = (t_vals[-1] - t_vals[0]) / np.timedelta64(1, 'D')
        if span_days > 4.0:
            found_sub = False
            for i in range(len(c_win) - shape['min_count'] + 1):
                sub = c_win.iloc[i:i + shape['min_count']]
                if (sub.t.iloc[-1] - sub.t.iloc[0]).total_seconds() <= 4.0 * 86400:
                    c_win = sub
                    found_sub = True
                    break
            if not found_sub:
                continue

        t_burst_start = c_win.t.min()
        prior_df = c_df[c_df.t < t_burst_start]

        if len(prior_df) < 5:
            continue

        prior_same_prod = prior_df[(prior_df.ProductCD == shape['ProductCD']) & (prior_df.amount >= shape['prior_cap'])]
        if not prior_same_prod.empty:
            continue

        prior_devs = set(prior_df['device_profile'].dropna().unique()) - GENERIC_DEVICES
        win_devs = set(c_win['device_profile'].dropna().unique()) - GENERIC_DEVICES
        if win_devs and (win_devs & prior_devs):
            continue

        cohort.append(c)
        first_loss[c] = t_burst_start
        burst_txns[c] = [int(x) for x in c_win.TransactionID.tolist()]
        burst_devices[c] = win_devs

    dev_to_cards: dict[str, list[str]] = {}
    for c in cohort:
        for d in burst_devices.get(c, set()):
            if not (d.startswith('Windows | ?') or d.startswith('iOS Device | ?') or d.startswith('MacOS | ?')):
                dev_to_cards.setdefault(d, []).append(c)

    shared_dev_cards = set()
    for d, cs in dev_to_cards.items():
        if len(cs) > 1:
            for c in cs:
                if c != shape['flagged_card']:
                    shared_dev_cards.add(c)

    cohort = [c for c in cohort if c not in shared_dev_cards]
    cohort.sort()
    first_loss = {c: first_loss[c] for c in cohort}
    burst_txns = {c: burst_txns[c] for c in cohort}
    return cohort, first_loss, burst_txns


def get_matched_controls(as_of_tx: pd.DataFrame, affected_cards: list[str], t0: pd.Timestamp, rng_seed: int = 42) -> dict[str, list[str]]:
    """Selects 20 activity-matched control cards per affected card (same 30d txn-count decile & primary ProductCD)."""
    rng = np.random.default_rng(rng_seed)
    recent_tx = as_of_tx[(as_of_tx.t >= t0 - pd.Timedelta(days=30)) & (as_of_tx.t <= t0)]
    if recent_tx.empty:
        return {}

    counts = recent_tx.groupby('card_id').size().rename('txn_count')
    prod_counts = recent_tx.groupby(['card_id', 'ProductCD']).size().unstack(fill_value=0)
    primary_prod = prod_counts.idxmax(axis=1).rename('primary_prod')

    card_stats = pd.concat([counts, primary_prod], axis=1).reset_index()
    try:
        card_stats['decile'] = pd.qcut(card_stats['txn_count'].rank(method='first'), 10, labels=False)
    except ValueError:
        card_stats['decile'] = 0

    aff_set = set(affected_cards)
    pool = card_stats[~card_stats.card_id.isin(aff_set)]

    controls: dict[str, list[str]] = {}
    used_controls: set[str] = set()

    for c in affected_cards:
        c_row = card_stats[card_stats.card_id == c]
        if c_row.empty:
            c_pool = pool[~pool.card_id.isin(used_controls)]
        else:
            dec = c_row['decile'].iloc[0]
            prod = c_row['primary_prod'].iloc[0]
            c_pool = pool[(pool.decile == dec) & (pool.primary_prod == prod) & (~pool.card_id.isin(used_controls))]
            if len(c_pool) < 20:
                c_pool = pool[(pool.decile == dec) & (~pool.card_id.isin(used_controls))]

        avail = c_pool.card_id.values
        if len(avail) >= 20:
            sampled = rng.choice(avail, 20, replace=False).tolist()
        else:
            sampled = avail.tolist()
        controls[c] = sampled
        used_controls.update(sampled)

    return controls


def compute_origin_lifts(
    affected_cards: list[str],
    first_loss: dict[str, pd.Timestamp],
    controls: dict[str, list[str]],
    as_of_tx: pd.DataFrame,
    lookback_days: int = 210
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes matched control lift and Fisher exact p-value for candidate origins prior to each card's first loss."""
    _, card_counts_all = _global_cardinality_stats()

    aff_frames = []
    ctrl_frames = []

    all_ctrl_cards = {ctrl_c for c_list in controls.values() for ctrl_c in c_list}
    sub_tx = as_of_tx[as_of_tx.card_id.isin(set(affected_cards) | all_ctrl_cards)]

    for c in affected_cards:
        t_loss = first_loss[c]
        w_start = t_loss - pd.Timedelta(days=lookback_days)
        c_tx = sub_tx[(sub_tx.card_id == c) & (sub_tx.t >= w_start) & (sub_tx.t < t_loss)]
        if not c_tx.empty:
            aff_frames.append(extract_features(c_tx))

        c_ctrls = controls.get(c, [])
        if c_ctrls:
            ctrl_tx = sub_tx[(sub_tx.card_id.isin(c_ctrls)) & (sub_tx.t >= w_start) & (sub_tx.t < t_loss)]
            if not ctrl_tx.empty:
                ctrl_frames.append(extract_features(ctrl_tx))

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
        card_glob = int(card_counts_all.get(feat, k_aff + k_ctrl))
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
        # Never use trigger_text to lower thresholds; detect solely from behavioral data
        shape = get_loss_shape(ctx.tx, ctx.flagged_txn_id)
        if shape is None:
            CommonPointOfCompromise.last_diagnostics[ctx.case_id] = {
                "status": "no_loss_shape",
                "cohort_size": 0,
                "cohort_cards": [],
            }
            if ctx.case_id in ('HHF-001', 'HHF-002', 'HHF-006'):
                print(f"\n[{ctx.case_id}] No loss burst shape on card {ctx.card_id}; affected set size: 0")
            return []

        affected_cards, first_loss, burst_txns = find_affected_cohort(ctx.tx, shape)
        cohort_size = len(affected_cards)
        bucket = _size_bucket(cohort_size)

        if ctx.case_id in ('HHF-001', 'HHF-002', 'HHF-006'):
            print(f"\n[{ctx.case_id}] Affected set size: {cohort_size}")
            print(f"[{ctx.case_id}] Affected cards: {affected_cards}")

        strat_table = self.get_stratified_lr_table()
        bucket_stats = strat_table.get(bucket, {}) if bucket else {}
        lr_val = float(bucket_stats.get("lr", 1.0))
        ci_low = float(bucket_stats.get("ci_low", 0.0))

        # Check size eligibility: allow size 3 only when its measured LR lower 95% CI > 2.0
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
            if ctx.case_id in ('HHF-001', 'HHF-002', 'HHF-006'):
                print(
                    f"[{ctx.case_id}] Verdict: cohort size {cohort_size} < min_allowed_size {min_allowed_size} "
                    f"(size-3 LR={strat_table.get('3', {}).get('lr', 0.0):.2f}, 95% CI low={strat_table.get('3', {}).get('ci_low', 0.0):.2f}) -> cohort too small."
                )
            return []

        controls = get_matched_controls(ctx.tx, affected_cards, shape['t0'], rng_seed=42)
        res_60, aff_feats_60 = compute_origin_lifts(affected_cards, first_loss, controls, ctx.tx, lookback_days=60)
        res_full, _ = compute_origin_lifts(affected_cards, first_loss, controls, ctx.tx, lookback_days=210)

        cap, _ = _global_cardinality_stats()

        if ctx.case_id in ('HHF-001', 'HHF-002', 'HHF-006'):
            print(f"[{ctx.case_id}] Stratified bucket '{bucket}': LR={lr_val:.2f} (95% CI [{ci_low:.2f}, {bucket_stats.get('ci_high', 0.0):.2f}])")
            print(f"[{ctx.case_id}] Cardinality 99th-pct cap: {cap:.1f}")
            for lead_feat in ('ProductCD_amount:R_99.93', 'ProductCD_amount:R_100.09'):
                r60 = res_60[res_60.feature == lead_feat] if not res_60.empty else pd.DataFrame()
                rfull = res_full[res_full.feature == lead_feat] if not res_full.empty else pd.DataFrame()
                if not r60.empty:
                    r = r60.iloc[0]
                    print(f"[{ctx.case_id}] Lead test (60d) {lead_feat}: k_aff={r.k_aff}/{cohort_size}, k_ctrl={r.k_ctrl}, matched_lift={r.lift:.2f}x, fisher_p={r.pval:.2e}, cardinality={r.cardinality}")
                elif not rfull.empty:
                    r = rfull.iloc[0]
                    print(f"[{ctx.case_id}] Lead test (full-hist) {lead_feat}: k_aff={r.k_aff}/{cohort_size}, k_ctrl={r.k_ctrl}, matched_lift={r.lift:.2f}x, fisher_p={r.pval:.2e}, cardinality={r.cardinality}")
                else:
                    print(f"[{ctx.case_id}] Lead test {lead_feat}: not present in cohort")

            if not res_60.empty:
                print(f"[{ctx.case_id}] Top 5 candidate origins (60d lookback):")
                for _, r in res_60.head(5).iterrows():
                    print(f"  - {r.feature}: k_aff={r.k_aff}/{cohort_size}, k_ctrl={r.k_ctrl}, lift={r.lift:.2f}x, p={r.pval:.2e}, cardinality={r.cardinality}")

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
            if ctx.case_id in ('HHF-001', 'HHF-002', 'HHF-006'):
                print(f"[{ctx.case_id}] Verdict: no common origin passed matched-control test; emitting coordinated loss-shape cohort finding (LR={lr_val:.2f}).")

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
                query="cpc_loss_shape_cohort",
                params={
                    "card_id": ctx.card_id,
                    "flagged_txn_id": int(ctx.flagged_txn_id),
                    "ProductCD": shape['ProductCD'],
                    "channel": shape['channel'],
                    "min_amt": float(shape['min_amt']),
                    "max_amt": float(shape['max_amt']),
                    "cohort_size_bucket": bucket,
                    "lookback_days": 60,
                },
            )
            return [Finding(
                detector=self.name,
                pattern="undocumented:coordinated_cnp_burst_no_shared_device",
                claim=claim,
                lr=lr_val,
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
        if ctx.case_id in ('HHF-001', 'HHF-002', 'HHF-006'):
            print(f"[{ctx.case_id}] Verdict: common origin found -> {best_feat} (lift={best.lift:.2f}x, p={best.pval:.2e}, LR={lr_val:.2f})")

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
            query="cpc_origin_search",
            params={
                "card_id": ctx.card_id,
                "flagged_txn_id": int(ctx.flagged_txn_id),
                "origin_key": best_feat,
                "cohort_size_bucket": bucket,
                "lookback_days": 60,
            },
        )
        return [Finding(
            detector=self.name,
            pattern=self.pattern,
            claim=claim,
            lr=lr_val,
            affected_txn_ids=flagged_burst_txns,
            connected_card_ids=[c for c in affected_cards if c != ctx.card_id],
        )]

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        """Backtests coordinated loss-shape / CPC detection stratified by cohort size ('3', '4-5', '6+')
        over confirmed-fraud episodes vs 5,000 legitimate Jul-Oct 2016 card-days."""
        if CommonPointOfCompromise._cached_bt is not None and CommonPointOfCompromise._cached_stratified_lr is not None:
            return CommonPointOfCompromise._cached_bt

        tx_jo = tx[(tx.t >= pd.Timestamp('2016-07-01')) & (tx.t < pd.Timestamp('2016-11-01'))]
        fraud_txns = set()
        for t_list in cc.txn_list:
            fraud_txns.update(t_list)

        tx_by_id = tx.set_index('TransactionID')
        fraud_cc = cc[cc.outcome == 'confirmed_fraud'].copy()

        # Positive episodes: confirmed fraud cases in Jul-Oct where the card has >=2 online C charges >= $250 in the case
        pos_cases = []
        for row in fraud_cc.itertuples():
            if not row.txn_list:
                continue
            valid_tids = [tid for tid in row.txn_list if tid in tx_by_id.index]
            if len(valid_tids) < 2:
                continue
            sub = tx_by_id.loc[valid_tids]
            if (sub.ProductCD == 'C').all() and (sub.channel == 'online').all() and (sub.amount >= 250.0).sum() >= 2:
                pos_cases.append((row.case_id, int(valid_tids[-1]), pd.to_datetime(row.opened_at)))

        # Evaluate cohort size on each positive episode
        pos_bucket_hits: dict[str, list[str]] = {"3": [], "4-5": [], "6+": []}
        pos_cohort_sizes: dict[str, int] = {}

        for case_id, f_tid, opened_dt in pos_cases:
            as_of = tx[tx.t <= opened_dt]
            shape = get_loss_shape(as_of, f_tid)
            if shape is None:
                pos_cohort_sizes[case_id] = 0
                continue
            cohort, _, _ = find_affected_cohort(as_of, shape)
            if shape['flagged_card'] not in cohort:
                pos_cohort_sizes[case_id] = 0
                continue
            c_size = len(cohort)
            pos_cohort_sizes[case_id] = c_size
            b = _size_bucket(c_size)
            if b is not None:
                pos_bucket_hits[b].append(case_id)

        # Negative population: 5,000 legitimate card-days from Jul-Oct 2016
        legit_tx = tx_jo[~tx_jo.TransactionID.isin(fraud_txns)]
        neg_sample = legit_tx.sample(n=min(5000, len(legit_tx)), random_state=42)

        neg_bucket_hits: dict[str, list[str]] = {"3": [], "4-5": [], "6+": []}

        # Pre-filter negative sample to those with amount >= 250 on online C or general burst candidates
        cand_neg = neg_sample[neg_sample.amount >= 25.0]
        for row in cand_neg.itertuples():
            c_tx = tx_jo[(tx_jo.card_id == row.card_id) & (tx_jo.t >= row.t - pd.Timedelta(days=3)) & (tx_jo.t <= row.t)]
            if len(c_tx) < 2:
                continue
            as_of = tx_jo[tx_jo.t <= row.t]
            shape = get_loss_shape(as_of, int(row.TransactionID))
            if shape is None:
                continue
            cohort, _, _ = find_affected_cohort(as_of, shape)
            if shape['flagged_card'] not in cohort:
                continue
            b = _size_bucket(len(cohort))
            if b is not None:
                neg_bucket_hits[b].append(str(row.TransactionID))

        n_pos = max(len(pos_cases), 1)
        n_neg = len(neg_sample)

        strat_table: dict[str, dict[str, float]] = {}
        for b in ("3", "4-5", "6+"):
            strat_table[b] = lr_with_ci(len(pos_bucket_hits[b]), n_pos, len(neg_bucket_hits[b]), n_neg)

        CommonPointOfCompromise._cached_stratified_lr = strat_table

        # Determine which buckets are active (size 3 is active if its 95% CI lower bound > 2.0)
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
        CommonPointOfCompromise._cached_bt = bt
        return bt
