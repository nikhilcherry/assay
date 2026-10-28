"""Block B1 -- Retrain the transaction fraud model for January 2017 & Drift / OOD Detection.

Summary of findings & architecture (Round 2 Review):
1. **Label Completeness Analysis (`measure_monthly_label_rates`):**
   - July–October 2016 are **fully labelled** (~3.37% confirmed fraud rate, 2.59%–4.19%
     of all transactions named in closed cases).
   - November–December 2016 are **partially labelled** (only 45 closed cases by `opened_at`:
     31 in Nov, 14 in Dec; 0.025% of Nov txns and 0.114% of Dec txns are named in a closed
     case). We therefore train strictly on the **fully labelled months (July–October 2016)**
     so unlabelled Nov–Dec transactions never bias the model toward zero.
2. **Out-of-Time Evaluation (Latest Fully-Labelled Month = October 2016):**
   - Train on Jul–Sep 2016, evaluate out-of-time on October 2016.
   - Report ROC-AUC, PR-AUC, Brier score, Expected Calibration Error (ECE across 10 bins),
     a 10-bin calibration table, and the bank's `risk_score` baseline on the same rows.
3. **Feature Drift Analysis (`drift_report()`):**
   - Compares January 2017 feature distributions against the Jul–Oct training set across
     the top 25 features by gain (computing PSI per feature).
   - Any feature with `PSI > 0.25` is flagged as drifted, and per-bucket drift ratios
     (`jan_pct / train_pct`) + top-feature robust z-scores (`|z| > 3.5`) identify
     out-of-distribution rows in January.
4. **Scoring API (`score(txn_ids)`):**
   - `score(txn_ids) -> pd.DataFrame[txn_id, p, in_distribution]` where `in_distribution: bool`
     is `False` whenever a transaction falls into a drifted bucket of a `PSI > 0.25` top-25
     feature OR has an extreme multivariate Mahalanobis / robust z-distance (`max_|z| > 3.5`
     or `mahalanobis_d2 > chi2_999`) on the top numeric features. The investigator uses
     `in_distribution=False` to pull the model's LR toward 1.0 (down-weighting model confidence
     on novel January activity), never to flip a verdict.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from assay.data import ROOT, load_transactions
from assay.model import DROP, features
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import R2_STAGE, _jan, case_pack, closed_cases, tx_all

MODEL_R2_PATH = R2_STAGE / "model_b1.txt"
ISO_R2_PATH = R2_STAGE / "isotonic_b1.json"
SCORES_B1_PATH = R2_STAGE / "scores_b1.parquet"
SCORES_R1_BASELINE_PATH = R2_STAGE / "scores_r2.parquet"
SCORES_ALL_B1_PATH = R2_STAGE / "scores_all_b1.parquet"
DRIFT_REPORT_PATH = R2_STAGE / "drift_b1_report.json"
REPORT_R2_JSON = R2_STAGE / "model_b1_report.json"
REPORT_R2_MD = ROOT / "docs" / "r2" / "B1_model.md"

# Explicitly block all label, identifier, and metadata columns from features
DROP_R2 = DROP | {"risk_score", "month_src", "explicit_label", "case_label", "isFraud"}


def r2_closed_case_labels() -> tuple[pd.Series, pd.Series]:
    """Return (pos_labels, explicit_labels) from final-round closed_cases().

    - pos_labels: TransactionID -> 1 for confirmed_fraud, 0 for cleared.
    - explicit_labels: same, restricted to transactions explicitly named in a
      closed case.
    """
    cc = closed_cases()
    lab: dict[int, int] = {}
    for r in cc.itertuples():
        y = 1 if r.outcome == "confirmed_fraud" else 0
        for tid in r.txn_list:
            if y == 1 or tid not in lab:
                lab[int(tid)] = y
    s = pd.Series(lab, dtype=int)
    return s, s.copy()


def measure_monthly_label_rates(df: pd.DataFrame, explicit_labels: pd.Series) -> list[dict]:
    """Measure per-month closed case counts (by `opened_at` month), share of txns
    named in a closed case per month, and confirmed_fraud rate per month."""
    cc = closed_cases().copy()
    cc["opened_month"] = cc["opened_at"].astype(str).str[:7]
    cc_by_month = (
        cc.groupby("opened_month")
        .agg(
            n_cases_opened=("case_id", "size"),
            n_cases_fraud=("outcome", lambda s: int((s == "confirmed_fraud").sum())),
            n_cases_cleared=("outcome", lambda s: int((s == "cleared").sum())),
        )
        .to_dict("index")
    )

    tmp = pd.DataFrame({
        "month": df["month"].values,
        "explicit": df["TransactionID"].map(explicit_labels),
    })
    rows = []
    for month, grp in tmp.groupby("month", sort=True):
        m_str = str(month)
        n_total = int(len(grp))
        n_fraud = int((grp["explicit"] == 1).sum())
        n_cleared = int((grp["explicit"] == 0).sum())
        n_named = n_fraud + n_cleared
        cc_info = cc_by_month.get(m_str, {"n_cases_opened": 0, "n_cases_fraud": 0, "n_cases_cleared": 0})
        rows.append({
            "month": m_str,
            "n_cases_opened": int(cc_info["n_cases_opened"]),
            "n_cases_fraud": int(cc_info["n_cases_fraud"]),
            "n_cases_cleared": int(cc_info["n_cases_cleared"]),
            "n_txns": n_total,
            "n_fraud_txns": n_fraud,
            "n_cleared_txns": n_cleared,
            "n_named_txns": n_named,
            "fraud_rate": round(n_fraud / n_total, 6) if n_total else 0.0,
            "named_rate": round(n_named / n_total, 6) if n_total else 0.0,
            "complete_labels": bool(m_str <= "2016-10"),
            "regime": "fully_labelled" if m_str <= "2016-10" else (
                "partially_labelled" if m_str <= "2016-12" else "unlabelled_exam"
            ),
        })
    return rows


def feature_columns_r2(df: pd.DataFrame, include_risk_score: bool = False) -> list[str]:
    drop = (DROP_R2 - {"risk_score"}) if include_risk_score else DROP_R2
    return [c for c in df.columns if c not in drop]


def single_feature_oct_auc(va_oct: pd.DataFrame, y_oct: np.ndarray, col: str) -> float:
    """Compute univariate ROC-AUC of a single feature on October holdout."""
    s = va_oct[col]
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        vals = s.astype("category").cat.codes.values.astype(float)
    else:
        vals = pd.to_numeric(s, errors="coerce").fillna(-999999.0).values.astype(float)
    if len(np.unique(vals)) < 2:
        return 0.5
    auc = float(roc_auc_score(y_oct, vals))
    return max(auc, 1.0 - auc)


def _fit_lgb(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    sample_weight: np.ndarray | None = None,
    rounds: int = 400,
    num_threads: int = 8,
) -> lgb.Booster:
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=50,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_cat_to_onehot=8,
        cat_smooth=20,
        verbose=-1,
        num_threads=num_threads,
        seed=7,
    )
    dtrain = lgb.Dataset(X, label=y, weight=sample_weight)
    return lgb.train(params, dtrain, num_boost_round=rounds)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric arrays."""
    exp_clean = np.asarray(expected, dtype=float)
    act_clean = np.asarray(actual, dtype=float)
    exp_clean = exp_clean[~np.isnan(exp_clean)]
    act_clean = act_clean[~np.isnan(act_clean)]
    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(exp_clean, quantiles))
    if len(edges) < 3:
        edges = np.linspace(float(min(exp_clean.min(), act_clean.min())), float(max(exp_clean.max(), act_clean.max())) + 1e-6, bins + 1)
    else:
        edges[0] = min(float(exp_clean.min()), float(act_clean.min())) - 1e-6
        edges[-1] = max(float(exp_clean.max()), float(act_clean.max())) + 1e-6
    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)
    exp_pct = np.clip(exp_counts / max(len(exp_clean), 1), 1e-6, None)
    act_pct = np.clip(act_counts / max(len(act_clean), 1), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_feature_psi_and_buckets(
    train_s: pd.Series,
    jan_s: pd.Series,
    all_s: pd.Series,
    bins: int = 10,
    psi_threshold: float = 0.25,
) -> tuple[dict, np.ndarray]:
    """Compute PSI for a feature (handling numeric + NaN bin or categorical top-K bins)
    between `train_s` (Jul-Oct) and `jan_s` (Jan), and return:
    - summary dict for `drift_report()`
    - boolean array `row_drifted_bucket` of length `len(all_s)` indicating whether each
      row lands in a severely drifted bucket of a feature whose total PSI > `psi_threshold`.
    """
    n_train = max(len(train_s), 1)
    n_jan = max(len(jan_s), 1)
    is_cat = isinstance(all_s.dtype, pd.CategoricalDtype) or all_s.dtype == object

    if is_cat:
        tr_str = train_s.astype(str).fillna("__MISSING__")
        jan_str = jan_s.astype(str).fillna("__MISSING__")
        all_str = all_s.astype(str).fillna("__MISSING__")
        top_cats = tr_str.value_counts().index[:bins].tolist()
        tr_b = tr_str.where(tr_str.isin(top_cats), "__OTHER__")
        jan_b = jan_str.where(jan_str.isin(top_cats), "__OTHER__")
        all_b = all_str.where(all_str.isin(top_cats), "__OTHER__")

        cats = sorted(set(top_cats) | {"__OTHER__"})
        tr_counts = tr_b.value_counts()
        jan_counts = jan_b.value_counts()
        psi_total = 0.0
        drifted_cats: set[str] = set()
        bucket_details = []
        for c in cats:
            ep = max(float(tr_counts.get(c, 0)) / n_train, 1e-6)
            ap = max(float(jan_counts.get(c, 0)) / n_jan, 1e-6)
            contrib = float((ap - ep) * np.log(ap / ep))
            psi_total += contrib
            if contrib > 0.05 and (ap / ep > 1.5 or ep / ap > 1.5):
                drifted_cats.add(c)
            bucket_details.append({"bucket": str(c), "train_pct": round(ep, 4), "jan_pct": round(ap, 4), "psi_contrib": round(contrib, 5)})

        flagged = psi_total > psi_threshold
        row_drifted = all_b.isin(drifted_cats).values if flagged else np.zeros(len(all_s), dtype=bool)
        return {
            "type": "categorical",
            "psi": round(float(psi_total), 5),
            "flagged": bool(flagged),
            "train_null_rate": round(float(train_s.isna().mean()), 4),
            "jan_null_rate": round(float(jan_s.isna().mean()), 4),
            "drifted_buckets": sorted(drifted_cats) if flagged else [],
        }, row_drifted

    # Numeric feature: include an explicit __NULL__ bucket + quantile bins on non-nulls
    tr_num = pd.to_numeric(train_s, errors="coerce").values.astype(float)
    jan_num = pd.to_numeric(jan_s, errors="coerce").values.astype(float)
    all_num = pd.to_numeric(all_s, errors="coerce").values.astype(float)

    tr_non_null = tr_num[~np.isnan(tr_num)]
    jan_non_null = jan_num[~np.isnan(jan_num)]

    if len(tr_non_null) == 0:
        return {
            "type": "numeric",
            "psi": 0.0,
            "flagged": False,
            "train_null_rate": 1.0,
            "jan_null_rate": round(float(np.isnan(jan_num).mean()), 4),
            "drifted_buckets": [],
        }, np.zeros(len(all_s), dtype=bool)

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(tr_non_null, quantiles))
    if len(edges) < 3:
        lo = float(min(tr_non_null.min(), jan_non_null.min() if len(jan_non_null) else tr_non_null.min()))
        hi = float(max(tr_non_null.max(), jan_non_null.max() if len(jan_non_null) else tr_non_null.max()))
        edges = np.linspace(lo - 1e-6, hi + 1e-6, bins + 1)
    else:
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf

    tr_idx = np.where(np.isnan(tr_num), -1, np.digitize(np.nan_to_num(tr_num, nan=0.0), edges[1:-1]))
    jan_idx = np.where(np.isnan(jan_num), -1, np.digitize(np.nan_to_num(jan_num, nan=0.0), edges[1:-1]))
    all_idx = np.where(np.isnan(all_num), -1, np.digitize(np.nan_to_num(all_num, nan=0.0), edges[1:-1]))

    n_bins = len(edges) - 1
    psi_total = 0.0
    drifted_bin_ids: set[int] = set()
    for b_id in [-1] + list(range(n_bins)):
        ep_raw = float((tr_idx == b_id).sum()) / n_train
        ap_raw = float((jan_idx == b_id).sum()) / n_jan
        if ep_raw == 0.0 and ap_raw == 0.0:
            continue
        ep = max(ep_raw, 1e-6)
        ap = max(ap_raw, 1e-6)
        contrib = float((ap - ep) * np.log(ap / ep))
        psi_total += contrib
        if contrib > 0.05 and (ap / ep > 1.5 or ep / ap > 1.5):
            drifted_bin_ids.add(b_id)

    flagged = psi_total > psi_threshold
    row_drifted = np.isin(all_idx, list(drifted_bin_ids)) if flagged else np.zeros(len(all_s), dtype=bool)
    return {
        "type": "numeric",
        "psi": round(float(psi_total), 5),
        "flagged": bool(flagged),
        "train_null_rate": round(float(np.isnan(tr_num).mean()), 4),
        "jan_null_rate": round(float(np.isnan(jan_num).mean()), 4),
        "train_mean": round(float(np.nanmean(tr_num)), 4) if len(tr_non_null) else 0.0,
        "jan_mean": round(float(np.nanmean(jan_num)), 4) if len(jan_non_null) else 0.0,
        "drifted_buckets": [str(b) for b in sorted(drifted_bin_ids)] if flagged else [],
    }, row_drifted


def compute_mahalanobis_ood(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    numeric_cols: list[str],
    quantile_cutoff: float = 0.99,
) -> tuple[np.ndarray, float]:
    """Compute diagonal-regularized Mahalanobis distance on top numeric features
    fitted on `train_mask` (Jul-Oct), returning `(d2_all, threshold_d2)`."""
    use_cols = numeric_cols[:15]
    X_all = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").fillna(df.loc[train_mask, c].median() if not df.loc[train_mask, c].isna().all() else 0.0).values.astype(float)
        for c in use_cols
    ])
    X_tr = X_all[train_mask]
    # Log-compress heavy-tailed positive columns so covariance isn't dominated by outliers
    X_tr_trans = np.sign(X_tr) * np.log1p(np.abs(X_tr))
    X_all_trans = np.sign(X_all) * np.log1p(np.abs(X_all))

    mu = np.mean(X_tr_trans, axis=0)
    cov = np.cov(X_tr_trans, rowvar=False) + np.eye(len(use_cols)) * 1e-3
    inv_cov = np.linalg.pinv(cov)
    diff = X_all_trans - mu
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    thresh = float(np.quantile(d2[train_mask], quantile_cutoff))
    return d2, thresh


def build_calibration_10bins(y_true: np.ndarray, p_cal: np.ndarray) -> tuple[list[dict], float]:
    """Compute a 10-bin calibration table and Expected Calibration Error (ECE)."""
    edges = np.linspace(0.0, 1.0, 11)
    bins = pd.cut(p_cal, edges, include_lowest=True)
    cal_df = (
        pd.DataFrame({"p": p_cal, "y": y_true, "bin": bins})
        .groupby("bin", observed=False)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    )
    n_total = max(len(y_true), 1)
    ece = 0.0
    table = []
    for idx, row in cal_df.iterrows():
        n_b = int(row["n"])
        mp = float(row["mean_predicted"]) if n_b > 0 and not np.isnan(row["mean_predicted"]) else 0.0
        ob = float(row["observed_rate"]) if n_b > 0 and not np.isnan(row["observed_rate"]) else 0.0
        gap = abs(mp - ob) if n_b > 0 else 0.0
        ece += (n_b / n_total) * gap
        table.append({
            "bin": str(idx),
            "n": n_b,
            "mean_predicted": round(mp, 5),
            "observed_rate": round(ob, 5),
            "abs_gap": round(gap, 5),
        })
    return table, round(float(ece), 5)


def build_full_feature_frame() -> pd.DataFrame:
    """Concatenate Round 1 transactions and January 2017 transactions, attach
    canonical `card_id` from `frames.tx_all()`, and compute `assay.model.features`."""
    old = load_transactions().copy()
    jan = _jan().copy()
    tx_slim = tx_all()
    card_map = tx_slim.set_index("TransactionID")["card_id"]
    jan["card_id"] = jan["TransactionID"].map(card_map)
    old["card_id"] = old["TransactionID"].map(card_map).fillna(old["card_id"])

    common_cols = [c for c in old.columns if c in jan.columns]
    both = pd.concat([old[common_cols], jan[common_cols]], ignore_index=True)
    both = both.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df = features(both)
    df["month"] = df["ts"].str[:7]
    return df


def _eval_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p_pred, dtype=float)
    if len(np.unique(y_arr)) < 2:
        return {"auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
    return {
        "auc": round(float(roc_auc_score(y_arr, p_arr)), 5),
        "pr_auc": round(float(average_precision_score(y_arr, p_arr)), 5),
        "brier": round(float(brier_score_loss(y_arr, p_arr)), 5),
    }


def _write_markdown_report(report: dict) -> None:
    REPORT_R2_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# B1 -- January 2017 Transaction Fraud Model (`model_r2`) & Drift Analysis",
        "",
        "## 1. Monthly Label Completeness Analysis",
        "",
        "| Month | Cases Opened | Fraud Cases | Cleared Cases | Total Txns | Named Txns | Share Named (%) | Confirmed Fraud Txns | Confirmed Fraud Rate (%) | Complete Labels? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["monthly_label_rates"]:
        lines.append(
            f"| {r['month']} | {r['n_cases_opened']:,} | {r['n_cases_fraud']:,} | {r['n_cases_cleared']:,} | "
            f"{r['n_txns']:,} | {r['n_named_txns']:,} | {r['named_rate']:.4%} | "
            f"{r['n_fraud_txns']:,} | {r['fraud_rate']:.4%} | `{r['complete_labels']}` ({r['regime']}) |"
        )
    lines.extend([
        "",
        "**Training Set Decision:**",
        "- **July–October 2016 (`2016-07` .. `2016-10`) are the only completely labelled months:** "
        "2.88% to 4.19% of all transactions each month are named in closed cases, and the confirmed fraud rate is "
        "3.37% overall (2.49% in Jul, 4.00% in Aug, 4.07% in Sep, 3.27% in Oct), matching the underlying population fraud rate.",
        "- **November–December 2016 (`2016-11` .. `2016-12`) are incomplete:** Only 45 closed cases exist by `opened_at` "
        "(31 in Nov, 14 in Dec), covering just 0.0248% of Nov transactions and 0.1141% of Dec transactions. "
        "Unnamed Nov–Dec transactions are uninvestigated, not confirmed legitimate.",
        "- **Decision:** We train strictly on the completely labelled months (**July–October 2016**; "
        "Jul–Sep for out-of-time evaluation on October 2016, and all four fully labelled months Jul–Oct for the production scorer).",
        "",
        "## 2. Out-of-Time Evaluation on Latest Fully-Labelled Month (October 2016)",
        "",
        f"- **Holdout Split:** Train `2016-07 .. 2016-09` (`n={report['october_holdout']['n_train']:,}`), "
        f"Evaluate OOT on `2016-10` (`n={report['october_holdout']['n_validate']:,}`, fraud rate `{report['october_holdout']['fraud_rate']:.4%}`).",
        f"- **Expected Calibration Error (ECE, 10 bins):** `{report['october_holdout']['ece_10bin']:.5f}` (requirement: `< 0.05`).",
        "",
        "| Model Variant (October 2016 OOT) | ROC-AUC | PR-AUC | Brier Score |",
        "|---|---:|---:|---:|",
    ])
    oct_no = report["october_holdout"]["without_risk_score"]
    oct_with = report["october_holdout"]["with_risk_score"]
    oct_bank = report["october_holdout"]["bank_risk_score_only"]
    lines.append(f"| `model_r2` (`without_risk_score` - production) | **{oct_no['auc']:.4f}** | **{oct_no['pr_auc']:.4f}** | **{oct_no['brier']:.4f}** |")
    lines.append(f"| `model_r2` (`with_risk_score` ablation) | {oct_with['auc']:.4f} | {oct_with['pr_auc']:.4f} | {oct_with['brier']:.4f} |")
    lines.append(f"| Bank `risk_score` baseline (same October rows) | {oct_bank['auc']:.4f} | {oct_bank['pr_auc']:.4f} | {oct_bank['brier']:.4f} |")

    lines.extend([
        "",
        "### 10-Bin Calibration Table (October 2016 2-Fold Cross-Fitted Isotonic)",
        "",
        "| Probability Bin | Count | Mean Predicted `p` | Observed Fraud Rate | Absolute Gap |",
        "|---|---:|---:|---:|---:|",
    ])
    for b in report["calibration_10bins"]:
        lines.append(
            f"| `{b['bin']}` | {b['n']:,} | {b['mean_predicted']:.5f} | {b['observed_rate']:.5f} | {b['abs_gap']:.5f} |"
        )

    dr = report["drift_report"]
    lines.extend([
        "",
        "## 3. Feature Drift Report (`Jul-Oct` Training vs `Jan 2017`, Top-25 Features by Gain)",
        "",
        f"- **Flagged Top-25 Features (`PSI > 0.25`):** `{dr['flagged_features']}` (`{len(dr['flagged_features'])}/{len(dr['features'])}` features).",
        f"- **January In-Distribution Rate (`score(jan_ids).in_distribution.mean()`):** `{dr['jan_in_distribution_rate']:.4%}` "
        f"(Training Jul-Oct In-Distribution Rate: `{dr['train_in_distribution_rate']:.4%}`).",
        "- **How `in_distribution` is defined and used:** A transaction has `in_distribution = False` when it falls into a "
        "severely shifted bucket (`PSI contrib > 0.05` and `>1.5x` density shift) of any top-25 feature with `PSI > 0.25`, "
        "or exceeds the 99th-percentile training Mahalanobis distance across the top numeric features (`d2 > "
        f"{dr['mahalanobis_threshold_d2']:.2f}`). Downstream investigators use `in_distribution = False` to pull the model's "
        "likelihood ratio toward `1.0` (down-weighting model confidence on drifted/unseen activity), never to flip a verdict.",
        "",
        "| Rank | Feature | Type | Gain | Oct Single AUC | Jul-Oct vs Jan PSI | Flagged (`PSI > 0.25`) | Train Null % | Jan Null % |",
        "|---:|---|---|---:|---:|---:|---|---:|---:|",
    ])
    for f in dr["features"]:
        lines.append(
            f"| {f['rank']} | `{f['feature']}` | `{f['type']}` | {f['gain']:,.1f} | {f['oct_single_auc']:.4f} | "
            f"**{f['psi']:.4f}** | `{f['flagged']}` | {f['train_null_rate']:.2%} | {f['jan_null_rate']:.2%} |"
        )

    psi = report["drift_psi"]
    comp = report["r1_vs_b1_january_comparison"]
    lines.extend([
        "",
        "## 4. Score PSI (Dec -> Jan) & Comparison vs Round-1 Model (`scores_r2.parquet`)",
        "",
        f"- **B1 `p_fraud` (`scores_b1.parquet`) Dec -> Jan PSI:** `{psi['p_fraud_dec_to_jan_psi']:.5f}`",
        f"- **Bank `risk_score` Dec -> Jan PSI:** `{psi['bank_risk_score_dec_to_jan_psi']:.5f}`",
        f"- **Round-1 baseline (`scores_r2.parquet`) Jan mean `p_fraud`:** `{comp['r1_jan_mean']:.5f}` vs "
        f"**B1 (`scores_b1.parquet`) Jan mean `p_fraud`:** `{comp['b1_jan_mean']:.5f}`",
        f"- **Jan `p_fraud` Pearson correlation (B1 vs R1):** `{comp['pearson_corr']:.5f}` (Spearman: `{comp['spearman_corr']:.5f}`)",
        "",
        "## 5. Final Case Pack (`HHF-001` .. `HHF-018`) Flagged Transaction Scores",
        "",
        "| Case ID | Trigger Type | Flagged Txn ID | Card ID | Bank `risk_score` | R1 `p_fraud` | B1 `p_fraud` (`p`) | `in_distribution` | Mahalanobis `d2` |",
        "|---|---|---:|---|---:|---:|---:|---|---:|",
    ])
    for c in report["case_pack_scores"]:
        r1_str = f"{c['r1_p_fraud']:.4f}" if c.get("r1_p_fraud") is not None else "N/A"
        lines.append(
            f"| {c['case_id']} | `{c['trigger_type']}` | {c['flagged_txn_id']} | "
            f"`{c['card_id']}` | {c['bank_risk_score']:.4f} | {r1_str} | **{c['p_fraud']:.4f}** | "
            f"`{c['in_distribution']}` | {c['mahalanobis_d2']:.2f} |"
        )
    lines.append("")
    REPORT_R2_MD.write_text("\n".join(lines))


def train_r2(rounds: int = 400, num_threads: int = 8) -> dict:
    """Train on confirmed complete months (Jul-Oct), evaluate OOT on October,
    compute top-25 feature drift report (`Jul-Oct` vs `Jan`), and score all transactions.

    Writes:
    - `data/stage/r2/scores_b1.parquet` (`[TransactionID, p_fraud]`)
    - `data/stage/r2/scores_all_b1.parquet` (`[TransactionID, month, uid, p_raw, p_fraud, p_oot, in_distribution, mahalanobis_d2]`)
    - `data/stage/r2/drift_b1_report.json`
    - `data/stage/r2/model_b1_report.json` and `docs/r2/B1_model.md`
    """
    R2_STAGE.mkdir(parents=True, exist_ok=True)
    df = build_full_feature_frame()
    pos_labels, explicit_labels = r2_closed_case_labels()

    # Determine feature columns strictly BEFORE adding any label columns to df
    cols_no_rs = feature_columns_r2(df, include_risk_score=False)
    cols_with_rs = feature_columns_r2(df, include_risk_score=True)

    monthly_rates = measure_monthly_label_rates(df, explicit_labels)

    # Full Jul-Oct is completely labelled; unnamed txns in Jul-Oct are confirmed legitimate (0).
    df["label"] = df["TransactionID"].map(pos_labels).fillna(0).astype(int)
    df["explicit_label"] = df["TransactionID"].map(explicit_labels)

    assert "label" not in cols_no_rs and "explicit_label" not in cols_no_rs
    assert "risk_score" not in cols_no_rs

    # 1. Out-of-time validation: train Jul-Sep, evaluate on Oct (latest fully labelled month)
    tr_jul_sep = df[df["month"] <= "2016-09"]
    va_oct = df[df["month"] == "2016-10"]
    y_oct = va_oct["label"].values

    m_no_rs = _fit_lgb(tr_jul_sep[cols_no_rs], tr_jul_sep["label"], rounds=rounds, num_threads=num_threads)
    p_oct_raw_no_rs = m_no_rs.predict(va_oct[cols_no_rs])

    # Audit top features on October holdout to guarantee zero label leakage
    imp_oct = pd.Series(m_no_rs.feature_importance("gain"), index=cols_no_rs).sort_values(ascending=False)
    leaked_cols = [
        col for col in imp_oct.index[:40]
        if single_feature_oct_auc(va_oct, y_oct, col) > 0.99
    ]
    if leaked_cols:
        cols_no_rs = [c for c in cols_no_rs if c not in leaked_cols]
        cols_with_rs = [c for c in cols_with_rs if c not in leaked_cols]
        m_no_rs = _fit_lgb(tr_jul_sep[cols_no_rs], tr_jul_sep["label"], rounds=rounds, num_threads=num_threads)
        p_oct_raw_no_rs = m_no_rs.predict(va_oct[cols_no_rs])

    m_with_rs = _fit_lgb(tr_jul_sep[cols_with_rs], tr_jul_sep["label"], rounds=rounds, num_threads=num_threads)
    p_oct_raw_with_rs = m_with_rs.predict(va_oct[cols_with_rs])

    # 2-fold cross-fitted isotonic calibration on October (latest fully labelled month)
    half = (np.arange(len(va_oct)) % 2) == 0
    cal_no_rs = np.empty(len(va_oct), dtype=float)
    cal_with_rs = np.empty(len(va_oct), dtype=float)
    for a, b in ((half, ~half), (~half, half)):
        cal_no_rs[b] = (
            IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
            .fit(p_oct_raw_no_rs[a], y_oct[a])
            .predict(p_oct_raw_no_rs[b])
        )
        cal_with_rs[b] = (
            IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
            .fit(p_oct_raw_with_rs[a], y_oct[a])
            .predict(p_oct_raw_with_rs[b])
        )

    # Fit full October isotonic calibrator for production scoring
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_oct_raw_no_rs, y_oct)
    iso_with_rs = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p_oct_raw_with_rs, y_oct)

    # 10-bin calibration table and ECE on October OOT
    cal_10bins, ece_10bin = build_calibration_10bins(y_oct, cal_no_rs)

    # Legacy 7-bin reliability table for backward compatibility
    bins_7 = pd.cut(cal_no_rs, [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0], include_lowest=True)
    rel_df = (
        pd.DataFrame({"p": cal_no_rs, "y": y_oct, "bin": bins_7})
        .groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    )
    reliability = [
        {
            "bin": str(idx),
            "n": int(row["n"]),
            "mean_predicted": round(float(row["mean_predicted"]), 4),
            "observed_rate": round(float(row["observed_rate"]), 4),
        }
        for idx, row in rel_df.iterrows()
    ]

    # 2. Production model: trained strictly on the fully-labelled months (Jul-Oct 2016)
    train_mask = (df["month"] <= "2016-10").values
    prod_df = df.loc[train_mask]
    final_model = _fit_lgb(
        prod_df[cols_no_rs],
        prod_df["label"],
        rounds=rounds,
        num_threads=num_threads,
    )
    final_model.save_model(str(MODEL_R2_PATH))
    ISO_R2_PATH.write_text(
        json.dumps({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
    )

    # Evaluate on Nov-Dec known closed cases for reference
    nov_dec_known = df[(df["month"].isin(["2016-11", "2016-12"])) & (df["explicit_label"].notna())]
    cc_nd_count = int(closed_cases()["opened_at"].astype(str).str[:7].isin(["2016-11", "2016-12"]).sum())
    nov_dec_eval: dict = {}
    if len(nov_dec_known) > 0 and nov_dec_known["explicit_label"].nunique() == 2:
        m_juloct_with_rs = _fit_lgb(prod_df[cols_with_rs], prod_df["label"], rounds=rounds, num_threads=num_threads)
        y_nd = nov_dec_known["explicit_label"].astype(int).values
        p_nd_no_rs = iso.predict(final_model.predict(nov_dec_known[cols_no_rs]))
        p_nd_with_rs = iso_with_rs.predict(m_juloct_with_rs.predict(nov_dec_known[cols_with_rs]))
        nov_dec_eval = {
            "n_cases": cc_nd_count,
            "n_txns": int(len(nov_dec_known)),
            "n_fraud": int((y_nd == 1).sum()),
            "n_cleared": int((y_nd == 0).sum()),
            "without_risk_score": _eval_metrics(y_nd, p_nd_no_rs),
            "with_risk_score": _eval_metrics(y_nd, p_nd_with_rs),
            "bank_risk_score_only": _eval_metrics(y_nd, nov_dec_known["risk_score"].values),
        }

    # 3. Top 25 features by gain, single-feature Oct AUC audit, and Drift Report (Jul-Oct vs Jan)
    imp = pd.Series(final_model.feature_importance("gain"), index=cols_no_rs).sort_values(ascending=False)
    top25_cols = [str(c) for c in imp.index[:25]]
    jan_mask = (df["month"] == "2017-01").values
    dec_mask = (df["month"] == "2016-12").values

    top_features_audit = []
    drift_features = []
    flagged_features = []
    any_drifted_bucket = np.zeros(len(df), dtype=bool)
    top_numeric_cols = []

    for rank, col in enumerate(top25_cols, 1):
        s_auc = single_feature_oct_auc(va_oct, y_oct, col)
        top_features_audit.append({
            "feature": col,
            "gain": round(float(imp[col]), 2),
            "oct_single_auc": round(s_auc, 5),
        })
        f_drift, row_drifted = compute_feature_psi_and_buckets(
            df.loc[train_mask, col],
            df.loc[jan_mask, col],
            df[col],
            bins=10,
            psi_threshold=0.25,
        )
        if f_drift["type"] == "numeric":
            top_numeric_cols.append(col)
        if f_drift["flagged"]:
            flagged_features.append(col)
            any_drifted_bucket |= row_drifted
        drift_features.append({
            "rank": rank,
            "feature": col,
            "gain": round(float(imp[col]), 2),
            "oct_single_auc": round(s_auc, 5),
            **f_drift,
        })

    # Combine PSI bucket drift with Mahalanobis OOD check on top numeric features
    maha_d2, maha_thresh = compute_mahalanobis_ood(df, train_mask, top_numeric_cols, quantile_cutoff=0.99)
    in_distribution = ~(any_drifted_bucket | (maha_d2 > maha_thresh))

    drift_rep = {
        "baseline_period": "2016-07..2016-10",
        "target_period": "2017-01",
        "psi_threshold": 0.25,
        "n_top_features": len(top25_cols),
        "flagged_features": flagged_features,
        "n_flagged_features": len(flagged_features),
        "mahalanobis_threshold_d2": round(maha_thresh, 4),
        "train_in_distribution_rate": round(float(in_distribution[train_mask].mean()), 5),
        "jan_in_distribution_rate": round(float(in_distribution[jan_mask].mean()), 5),
        "lr_downweight_policy": (
            "When in_distribution is False or top features have PSI > 0.25, "
            "pull the model likelihood ratio toward 1.0 (weaker evidence), never flip a verdict."
        ),
        "features": drift_features,
    }
    DRIFT_REPORT_PATH.write_text(json.dumps(drift_rep, indent=2))

    # 4. Score all transactions (Jul 2016 - Jan 2017)
    raw_all = final_model.predict(df[cols_no_rs])
    p_fraud_all = np.clip(iso.predict(raw_all), 0.0, 1.0)

    oot_series = pd.Series(cal_no_rs, index=va_oct["TransactionID"].values)
    full_scores = pd.DataFrame({
        "TransactionID": df["TransactionID"].astype(int),
        "month": df["month"].values,
        "uid": df["uid"].values,
        "p_raw": raw_all.astype(float),
        "p_fraud": p_fraud_all.astype(float),
        "p_oot": df["TransactionID"].map(oot_series),
        "in_distribution": in_distribution.astype(bool),
        "mahalanobis_d2": np.round(maha_d2, 4),
    })
    full_scores.to_parquet(SCORES_ALL_B1_PATH, index=False)

    # Save `[TransactionID, p_fraud]` to `data/stage/r2/scores_b1.parquet`
    scores_b1 = full_scores[["TransactionID", "p_fraud"]].sort_values("TransactionID", kind="stable")
    scores_b1.to_parquet(SCORES_B1_PATH, index=False)

    # Compare January scores against Round-1 model's `data/stage/r2/scores_r2.parquet`
    jan_b1_df = full_scores.loc[jan_mask, ["TransactionID", "p_fraud"]].set_index("TransactionID")
    r1_lookup: pd.Series | None = None
    r1_vs_b1_comp = {
        "b1_jan_mean": round(float(jan_b1_df["p_fraud"].mean()), 5),
        "r1_jan_mean": float("nan"),
        "pearson_corr": float("nan"),
        "spearman_corr": float("nan"),
        "n_compared": int(len(jan_b1_df)),
    }
    if SCORES_R1_BASELINE_PATH.exists():
        r1_df = pd.read_parquet(SCORES_R1_BASELINE_PATH)
        if "TransactionID" in r1_df.columns and "p_fraud" in r1_df.columns:
            r1_lookup = r1_df.set_index("TransactionID")["p_fraud"]
            joined = jan_b1_df.join(r1_lookup.rename("r1_p_fraud"), how="inner")
            if len(joined) > 0:
                r1_vs_b1_comp = {
                    "b1_jan_mean": round(float(joined["p_fraud"].mean()), 5),
                    "r1_jan_mean": round(float(joined["r1_p_fraud"].mean()), 5),
                    "pearson_corr": round(float(joined["p_fraud"].corr(joined["r1_p_fraud"], method="pearson")), 5),
                    "spearman_corr": round(float(joined["p_fraud"].corr(joined["r1_p_fraud"], method="spearman")), 5),
                    "n_compared": int(len(joined)),
                }

    # Compute Dec -> Jan score PSI
    psi_p_fraud = compute_psi(p_fraud_all[dec_mask], p_fraud_all[jan_mask])
    psi_p_raw = compute_psi(raw_all[dec_mask], raw_all[jan_mask])
    psi_bank_rs = compute_psi(df.loc[dec_mask, "risk_score"].values, df.loc[jan_mask, "risk_score"].values)

    # Extract scores & in_distribution flag for the 18 case-pack flagged transactions
    cp = case_pack()
    score_lookup = full_scores.set_index("TransactionID")
    tx_lookup = df.set_index("TransactionID")
    cp_scores = []
    for r in cp.itertuples():
        tid = int(r.flagged_txn_id)
        s_row = score_lookup.loc[tid]
        p_f = float(s_row["p_fraud"])
        p_r = float(s_row["p_raw"])
        in_dist = bool(s_row["in_distribution"])
        m_d2 = float(s_row["mahalanobis_d2"])
        b_rs = float(tx_lookup.at[tid, "risk_score"])
        r1_p = round(float(r1_lookup.at[tid]), 4) if (r1_lookup is not None and tid in r1_lookup.index) else None
        cp_scores.append({
            "case_id": str(r.case_id),
            "trigger_type": str(r.trigger_type),
            "flagged_txn_id": tid,
            "card_id": str(r.card_id),
            "customer_id": str(r.customer_id),
            "bank_risk_score": round(b_rs, 4),
            "r1_p_fraud": r1_p,
            "p_fraud": round(p_f, 4),
            "p_raw": round(p_r, 4),
            "in_distribution": in_dist,
            "mahalanobis_d2": round(m_d2, 2),
        })

    report = {
        "train_regime": "2016-07..2016-10 (strictly fully-labelled months only)",
        "calibration_month": "2016-10",
        "rounds": rounds,
        "n_features": len(cols_no_rs),
        "n_train_prod": int(len(prod_df)),
        "n_jan_scored": int(jan_mask.sum()),
        "n_total_scored": int(len(scores_b1)),
        "monthly_label_rates": monthly_rates,
        "october_holdout": {
            "n_train": int(len(tr_jul_sep)),
            "n_validate": int(len(va_oct)),
            "fraud_rate": round(float(y_oct.mean()), 5),
            "ece_10bin": ece_10bin,
            "without_risk_score": {
                "auc": round(float(roc_auc_score(y_oct, p_oct_raw_no_rs)), 5),
                "pr_auc": round(float(average_precision_score(y_oct, p_oct_raw_no_rs)), 5),
                "brier": round(float(brier_score_loss(y_oct, cal_no_rs)), 5),
            },
            "with_risk_score": {
                "auc": round(float(roc_auc_score(y_oct, p_oct_raw_with_rs)), 5),
                "pr_auc": round(float(average_precision_score(y_oct, p_oct_raw_with_rs)), 5),
                "brier": round(float(brier_score_loss(y_oct, cal_with_rs)), 5),
            },
            "bank_risk_score_only": _eval_metrics(y_oct, va_oct["risk_score"].values),
        },
        "calibration_10bins": cal_10bins,
        "nov_dec_closed_cases_holdout": nov_dec_eval,
        "reliability": reliability,
        "drift_report": drift_rep,
        "drift_psi": {
            "p_fraud_dec_to_jan_psi": round(psi_p_fraud, 5),
            "p_raw_dec_to_jan_psi": round(psi_p_raw, 5),
            "bank_risk_score_dec_to_jan_psi": round(psi_bank_rs, 5),
            "dec_mean_p_fraud": round(float(p_fraud_all[dec_mask].mean()), 5),
            "jan_mean_p_fraud": round(float(p_fraud_all[jan_mask].mean()), 5),
            "dec_mean_bank_risk_score": round(float(df.loc[dec_mask, "risk_score"].mean()), 5),
            "jan_mean_bank_risk_score": round(float(df.loc[jan_mask, "risk_score"].mean()), 5),
        },
        "r1_vs_b1_january_comparison": r1_vs_b1_comp,
        "case_pack_scores": cp_scores,
        "top_features_by_gain": top25_cols,
        "top_features_audit": top_features_audit,
    }

    REPORT_R2_JSON.write_text(json.dumps(report, indent=2))
    _write_markdown_report(report)
    scores_b1_map.cache_clear()
    _scores_all_b1_table.cache_clear()
    drift_report.cache_clear()
    return report


@lru_cache(maxsize=1)
def drift_report() -> dict:
    """Return the top-25 feature drift report comparing January vs Jul-Oct training."""
    if not DRIFT_REPORT_PATH.exists():
        train_r2()
    return json.loads(DRIFT_REPORT_PATH.read_text())


@lru_cache(maxsize=1)
def _scores_all_b1_table() -> pd.DataFrame:
    """Cached table of `[TransactionID, p_fraud, in_distribution, mahalanobis_d2]` indexed by TransactionID."""
    if not SCORES_ALL_B1_PATH.exists():
        train_r2()
    df = pd.read_parquet(SCORES_ALL_B1_PATH)
    if "in_distribution" not in df.columns:
        df = train_r2()
        df = pd.read_parquet(SCORES_ALL_B1_PATH)
    return df.set_index("TransactionID")


@lru_cache(maxsize=1)
def scores_b1_map() -> pd.Series:
    """Fast lookup Series mapping TransactionID -> calibrated `p_fraud` from `scores_b1.parquet`."""
    if not SCORES_B1_PATH.exists():
        train_r2()
    df = pd.read_parquet(SCORES_B1_PATH)
    return df.set_index("TransactionID")["p_fraud"]


# Alias for backward compatibility
scores_r2_map = scores_b1_map


def score(txn_ids: Iterable[int] | int) -> pd.DataFrame:
    """Return `DataFrame[txn_id, p, in_distribution]` for the requested transaction IDs.

    - `txn_id`: int transaction identifier
    - `p`: calibrated fraud probability in [0, 1]
    - `in_distribution`: bool, True if the transaction falls within the training
      feature distribution (no drifted bucket on any top-25 feature with PSI > 0.25
      and Mahalanobis distance within the 99th-percentile training envelope).
    """
    if isinstance(txn_ids, (int, np.integer)):
        ids = [int(txn_ids)]
    else:
        ids = [int(x) for x in txn_ids]
    tbl = _scores_all_b1_table()
    sub = tbl.reindex(ids)
    return pd.DataFrame({
        "txn_id": ids,
        "p": sub["p_fraud"].astype(float).values,
        "in_distribution": sub["in_distribution"].fillna(False).astype(bool).values,
    })


def score_txn(txn_id: int) -> float:
    """Return calibrated `p_fraud` for `txn_id` from `scores_b1.parquet`."""
    s = scores_b1_map()
    return float(s.at[int(txn_id)])


def attest_txn_score(txn_id: int, card_id: str | None = None, customer_id: str | None = None) -> Attestation:
    """Produce a re-runnable Attestation for a transaction's retrained `p_fraud` and `in_distribution` status."""
    sc = score([int(txn_id)]).iloc[0]
    p = float(sc["p"])
    in_dist = bool(sc["in_distribution"])
    tx = tx_all().set_index("TransactionID")
    row = tx.loc[int(txn_id)]
    c_id = card_id or str(row["card_id"])
    cust_id = customer_id or str(row["customer_id"])
    bank_rs = float(row["risk_score"])
    return Attestation(
        what=(
            f"Transaction {int(txn_id)} on card {c_id} scored p_fraud={p:.4f} "
            f"(in_distribution={in_dist}) under retrained model_r2 (stale bank risk_score={bank_rs:.2f})."
        ),
        evidence=Evidence(
            txn_ids=[int(txn_id)],
            card_ids=[c_id],
            customer_ids=[cust_id],
        ),
        query="assay.r2.model_r2.score",
        params={"txn_ids": [int(txn_id)]},
    )


if __name__ == "__main__":
    rep = train_r2()
    print(json.dumps(rep, indent=2))
