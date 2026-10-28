"""Tests for B25 (Challenge 13 -- Someone else's schema).

Splits the canonical set into SOURCE_COLS (27 columns that must be mapped from the
variant) and DERIVED_COLS (5 columns recomputed after mapping by one shared function
`recompute_derived_columns()` using the exact `assay.r2.frames.stage()` code path).
Tests both the 4 multi-transform synthetic variants AND dedicated red-team attack
variants from the Red-vs-Blue debate (ATK-08, ATK-09, ATK-10, all-null columns,
extra junk columns, opaque renames/reorders).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assay.r2.frames import tx_all
from assay.r2.schema_map import (
    CANONICAL_COLS,
    DERIVED_COLS,
    IDENTITY_SOURCE_COLS,
    NOT_RECOVERABLE,
    SOURCE_COLS,
    approve,
    apply,
    infer_mapping,
    recompute_derived_columns,
)


@pytest.fixture(scope="module")
def sample_20k() -> pd.DataFrame:
    tx = tx_all()
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(tx), size=20_000, replace=False))
    return tx.iloc[idx].reset_index(drop=True)


def _add_decoys(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    out["decoy_random_uuid"] = [f"UUID-{i:07d}-X" for i in range(n)]
    out["decoy_constant_str"] = "BANK_INTERNAL_V2"
    out["decoy_gaussian_noise"] = rng.normal(loc=-50.0, scale=12.0, size=n)
    out["decoy_small_cat"] = rng.choice(["ALPHA", "BETA", "GAMMA", "DELTA"], size=n)
    return out


def _shuffle_and_rename(
    df: pd.DataFrame,
    rng: np.random.Generator,
    prefix: str = "col",
) -> tuple[pd.DataFrame, dict[str, str]]:
    cols = list(df.columns)
    perm = rng.permutation(len(cols))
    shuffled_cols = [cols[i] for i in perm]
    rename_map = {orig: f"{prefix}_{idx:03d}" for idx, orig in enumerate(shuffled_cols)}
    out = df[shuffled_cols].rename(columns=rename_map)
    return out, rename_map


def _make_variant_1(base: pd.DataFrame) -> pd.DataFrame:
    """Variant 1: Source columns only, renamed + shuffled + cents amounts + epoch seconds + decoys."""
    rng = np.random.default_rng(101)
    df = base[SOURCE_COLS].copy()
    df["amount"] = np.round(df["amount"] * 100).astype("int64")
    epoch_sec = (pd.to_datetime(df["ts"]) - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    df["ts"] = epoch_sec
    df = _add_decoys(df, rng)
    v1, _ = _shuffle_and_rename(df, rng, prefix="v1_f")
    return v1


def _make_variant_2(base: pd.DataFrame) -> pd.DataFrame:
    """Variant 2: Split card_id into (customer_id, card_suffix), date+time split into 2 columns,
    one-hot channel, and packed device_profile string instead of raw device columns."""
    rng = np.random.default_rng(202)
    df = base[SOURCE_COLS + ["device_profile"]].copy()
    parts = df["card_id"].str.split("-", expand=True)
    df["card_suffix_part"] = parts[1]
    df = df.drop(columns=["card_id"])
    # Split ts into date_part and time_part
    dt_parts = df["ts"].astype(str).str.split(" ", expand=True)
    df["tx_date_part"] = dt_parts[0]
    df["tx_time_part"] = dt_parts[1]
    df = df.drop(columns=["ts"])
    df["ch_in_person"] = (df["channel"] == "in_person").astype(int)
    df["ch_online"] = (df["channel"] == "online").astype(int)
    df = df.drop(columns=["channel"])
    # Drop individual device columns; must be recovered by unpacking device_profile
    df = df.drop(columns=["DeviceInfo", "id_30", "id_31", "id_33"])
    df = _add_decoys(df, rng)
    v2, _ = _shuffle_and_rename(df, rng, prefix="v2_attr")
    return v2


def _make_variant_3(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Variant 3: Separate identity table keyed differently + cents amounts + TransactionDT present."""
    rng = np.random.default_rng(303)
    df = base[SOURCE_COLS + ["TransactionDT"]].copy()
    id_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_15", "id_23"]
    has_id = df[id_cols].notna().any(axis=1)
    id_df = df.loc[has_id, ["TransactionID"] + id_cols].copy()
    id_df = id_df.rename(columns={
        "TransactionID": "fk_tx_ref",
        "DeviceType": "id_dev_kind",
        "DeviceInfo": "id_hw_str",
        "id_30": "id_os_ver",
        "id_31": "id_ua_str",
        "id_33": "id_screen_px",
        "id_15": "id_seen_flag",
        "id_23": "id_proxy_type",
    })
    main_df = df.drop(columns=id_cols).copy()
    main_df["amount"] = np.round(main_df["amount"] * 100).astype("int64")
    main_df = _add_decoys(main_df, rng)
    main_shuffled, _ = _shuffle_and_rename(main_df, rng, prefix="v3_tx")
    return {"tx_records": main_shuffled, "id_records": id_df}


def _make_variant_4(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Variant 4: Full kitchen sink -- split tables, split card_id, one-hot channel,
    cents amounts, epoch-ms timestamps, plus decoys."""
    rng = np.random.default_rng(404)
    df = base[SOURCE_COLS].copy()
    id_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_15", "id_23"]
    has_id = df[id_cols].notna().any(axis=1)
    id_df = df.loc[has_id, ["TransactionID"] + id_cols].copy()
    id_shuffled, _ = _shuffle_and_rename(id_df, rng, prefix="v4_id")

    main_df = df.drop(columns=id_cols).copy()
    main_df["amount"] = np.round(main_df["amount"] * 100).astype("int64")
    epoch_ms = ((pd.to_datetime(main_df["ts"]) - pd.Timestamp("1970-01-01")).dt.total_seconds() * 1000).astype("int64")
    main_df["ts"] = epoch_ms
    parts = main_df["card_id"].str.split("-", expand=True)
    main_df["k_num"] = parts[1]
    main_df = main_df.drop(columns=["card_id"])
    main_df["is_pos"] = (main_df["channel"] == "in_person").astype(int)
    main_df["is_ecom"] = (main_df["channel"] == "online").astype(int)
    main_df = main_df.drop(columns=["channel"])
    main_df = _add_decoys(main_df, rng)
    main_shuffled, _ = _shuffle_and_rename(main_df, rng, prefix="v4_main")
    return {"core": main_shuffled, "identity": id_shuffled}


def _evaluate_recovery(canonical_ref: pd.DataFrame, recovered_df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Compare each column in `cols` in recovered_df against ground truth canonical_ref."""
    accuracies: dict[str, float] = {}
    for col in cols:
        if col not in recovered_df.columns:
            accuracies[col] = 0.0
            continue
        s_ref = canonical_ref[col]
        s_rec = recovered_df[col]
        both_null = s_ref.isna() & s_rec.isna()
        if pd.api.types.is_numeric_dtype(s_ref.dtype):
            diff = np.abs(s_ref.fillna(-999999.0).astype(float) - s_rec.fillna(-999999.0).astype(float))
            eq = both_null | (diff <= 1e-4)
        elif pd.api.types.is_datetime64_any_dtype(s_ref.dtype):
            eq = both_null | (pd.to_datetime(s_ref) == pd.to_datetime(s_rec))
        else:
            r_str = s_ref.fillna("<<NULL>>").astype(str).replace({"": "<<NULL>>", "None": "<<NULL>>"})
            c_str = s_rec.fillna("<<NULL>>").astype(str).replace({"": "<<NULL>>", "None": "<<NULL>>"})
            eq = r_str == c_str
        accuracies[col] = float(eq.mean())
    return accuracies


def test_four_synthetic_variants_recovery(sample_20k: pd.DataFrame, tmp_path: Path) -> None:
    variants = {
        "variant_1_renamed_cents_epoch": _make_variant_1(sample_20k),
        "variant_2_split_card_datetime_onehot_packed_dev": _make_variant_2(sample_20k),
        "variant_3_separate_identity_table": _make_variant_3(sample_20k),
        "variant_4_full_combined_stress": _make_variant_4(sample_20k),
    }

    # Compute canonical derived reference via the single shared function `recompute_derived_columns`
    expected_canonical = recompute_derived_columns(sample_20k[SOURCE_COLS], allow_txid_lookup=True)
    ref_by_tid = expected_canonical.set_index("TransactionID")

    for vname, foreign_data in variants.items():
        mapping = infer_mapping(foreign_data, proposal_dir=tmp_path)
        assert Path(mapping.proposal_path).exists()
        approve(mapping.proposal_id, approver_role="L1")
        recovered_df, report = apply(mapping, foreign_data, require_proposal_approval=True)

        # 1. Score recovery strictly on SOURCE_COLS (against sanitized canonical source columns)
        src_acc = _evaluate_recovery(expected_canonical, recovered_df, SOURCE_COLS)
        src_recovered = [c for c, acc in src_acc.items() if acc >= 0.99]
        src_wrong_or_missed = {c: src_acc[c] for c in SOURCE_COLS if src_acc[c] < 0.99}
        src_recovery_rate = len(src_recovered) / len(SOURCE_COLS)

        # 2. Evaluate recomputed DERIVED_COLS against expected_canonical
        der_acc = _evaluate_recovery(expected_canonical, recovered_df, DERIVED_COLS)
        rec_by_tid = recovered_df.set_index("TransactionID")

        print(f"\n=== {vname} (proposal={mapping.proposal_id}) ===")
        print(
            f"SOURCE columns recovered (>=99% exact cell match): "
            f"{len(src_recovered)}/{len(SOURCE_COLS)} ({src_recovery_rate:.1%})"
        )
        if src_wrong_or_missed:
            print(f"SOURCE wrong or missed: {src_wrong_or_missed}")
        print(f"Unmapped decoy columns ignored: {len(report.unmapped_foreign_cols)} -> {report.unmapped_foreign_cols}")

        print("  [SOURCE COLUMNS]")
        for c_col in SOURCE_COLS:
            if c_col in mapping:
                expr, conf, sample_ids = mapping[c_col]
                acc = src_acc[c_col]
                print(f"    {c_col:16s} <- {expr:42s} | conf={conf:.2f} | acc={acc:.3f} | ids={sample_ids[:3]}")
            else:
                print(f"    {c_col:16s} <- MISSING")

        print("  [DERIVED COLUMNS (recomputed via shared recompute_derived_columns)]")
        for d_col in DERIVED_COLS:
            assert d_col in mapping, f"Derived column {d_col} missing from mapping"
            expr, conf, sample_ids = mapping[d_col]
            acc = der_acc[d_col]
            sample_ref_vals = [ref_by_tid.at[tid, d_col] for tid in sample_ids[:3]]
            sample_rec_vals = [rec_by_tid.at[tid, d_col] for tid in sample_ids[:3]]
            print(
                f"    {d_col:16s} := {expr:48s} | acc={acc:.3f} | "
                f"ids={sample_ids[:3]} | vals={sample_rec_vals}"
            )
            assert sample_ref_vals == sample_rec_vals, (
                f"Derived column {d_col} mismatch on sample_ids {sample_ids[:3]}: "
                f"expected {sample_ref_vals}, got {sample_rec_vals}"
            )
            assert acc >= 0.999, f"Derived column {d_col} accuracy {acc:.4f} < 0.999"

        assert src_recovery_rate >= 0.95, (
            f"{vname} recovered only {src_recovery_rate:.1%} of SOURCE columns; "
            f"wrong/missed={src_wrong_or_missed}"
        )
        for c_col, choice in mapping.items():
            att_dict = choice.attestation.to_dict()
            assert len(choice.sample_ids) >= 3
            assert att_dict["how"]["query"] == "assay.r2.schema_map.infer_mapping"
            assert len(att_dict["evidence"]["txn_ids"]) == len(choice.sample_ids)


def test_debate_red_team_attack_variants(sample_20k: pd.DataFrame, tmp_path: Path) -> None:
    """Red-vs-Blue debate attack suite: tests each schema-variant attack from Round 6
    (ATK-08: cents + 0..100 risk_score; ATK-09: missing identity table + split date/time +
    split card_id; ATK-10: timezone-aware ISO timestamps 'Z' and '+05:30'; plus all-null
    columns, extra junk columns, and opaque renames)."""
    rng = np.random.default_rng(606)
    expected_canonical = recompute_derived_columns(sample_20k[SOURCE_COLS], allow_txid_lookup=True)

    # --- Attack A (ATK-08): Amounts in cents + risk_score on 0..100 scale + all-null column + junk decoys ---
    df_atk08 = sample_20k[SOURCE_COLS].copy()
    df_atk08["amount"] = np.round(df_atk08["amount"] * 100).astype("int64")
    df_atk08["risk_score"] = np.round(df_atk08["risk_score"] * 100.0, 4)
    df_atk08["all_null_col_1"] = None
    df_atk08["all_null_col_2"] = np.nan
    df_atk08 = _add_decoys(df_atk08, rng)
    atk08_foreign, _ = _shuffle_and_rename(df_atk08, rng, prefix="atk08_c")

    map_08 = infer_mapping(atk08_foreign, proposal_dir=tmp_path)
    assert Path(map_08.proposal_path).exists()
    assert len(map_08.all_null_foreign) == 2
    assert map_08["amount"].expr.endswith(" / 100.0")
    assert map_08["risk_score"].expr.endswith(" / 100.0")
    approve(map_08.proposal_id, approver_role="L1")
    rec_08, rep_08 = apply(map_08, atk08_foreign, require_proposal_approval=True)
    acc_08 = _evaluate_recovery(expected_canonical, rec_08, SOURCE_COLS + DERIVED_COLS)
    assert all(acc_08[c] >= 0.999 for c in SOURCE_COLS + DERIVED_COLS)
    assert float(rec_08["risk_score"].max()) <= 1.0
    assert len(rep_08.all_null_foreign_cols) == 2

    # --- Attack B (ATK-09): Missing identity table entirely + split card_id + split date & time ---
    tx_only_cols = [c for c in SOURCE_COLS if c not in IDENTITY_SOURCE_COLS]
    df_atk09 = sample_20k[tx_only_cols].copy()
    card_parts = df_atk09["card_id"].str.split("-", expand=True)
    df_atk09["card_seq"] = card_parts[1]
    df_atk09 = df_atk09.drop(columns=["card_id"])
    ts_parts = df_atk09["ts"].astype(str).str.split(" ", expand=True)
    df_atk09["txn_date"] = ts_parts[0]
    df_atk09["txn_time"] = ts_parts[1]
    df_atk09 = df_atk09.drop(columns=["ts"])
    df_atk09["empty_placeholder"] = None
    df_atk09 = _add_decoys(df_atk09, rng)
    atk09_foreign, _ = _shuffle_and_rename(df_atk09, rng, prefix="atk09_x")

    map_09 = infer_mapping(atk09_foreign, proposal_dir=tmp_path)
    assert Path(map_09.proposal_path).exists()
    # All 7 missing identity source columns + device_profile are reported as unrecoverable gaps
    for id_col in IDENTITY_SOURCE_COLS:
        assert map_09.not_recoverable.get(id_col) == NOT_RECOVERABLE
    assert map_09.not_recoverable.get("device_profile") == NOT_RECOVERABLE
    approve(map_09.proposal_id, approver_role="L1")
    rec_09, rep_09 = apply(map_09, atk09_foreign, require_proposal_approval=True)
    # All 20 present transaction SOURCE_COLS are recovered 100%
    acc_09_tx = _evaluate_recovery(expected_canonical, rec_09, tx_only_cols)
    assert all(acc_09_tx[c] >= 0.999 for c in tx_only_cols)
    # Critical ATK-09 check: device_profile is "" (empty string) everywhere, never "None | ? | None | ?"
    assert (rec_09["device_profile"] == "").all()
    assert set(CANONICAL_COLS) <= set(rec_09.columns)

    # --- Attack C (ATK-10): Timezone-aware ISO timestamps ('Z' and '+05:30') ---
    df_atk10 = sample_20k[SOURCE_COLS].copy()
    t_utc = pd.to_datetime(df_atk10["ts"]).dt.tz_localize("UTC")
    # Half formatted with 'Z', half converted to '+05:30' (IST)
    half = len(df_atk10) // 2
    ts_z = t_utc.iloc[:half].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_ist = t_utc.iloc[half:].dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")
    df_atk10["ts"] = pd.concat([ts_z, ts_ist]).reindex(df_atk10.index)
    df_atk10 = _add_decoys(df_atk10, rng)
    atk10_foreign, _ = _shuffle_and_rename(df_atk10, rng, prefix="atk10_z")

    map_10 = infer_mapping(atk10_foreign, proposal_dir=tmp_path)
    assert map_10["ts"].expr.startswith("tz_iso_to_ts(")
    approve(map_10.proposal_id, approver_role="L1")
    rec_10, rep_10 = apply(map_10, atk10_foreign, require_proposal_approval=True)
    # Verify `t` is tz-naive UTC datetime64[ns] and matches canonical `t` 100%
    assert pd.api.types.is_datetime64_any_dtype(rec_10["t"].dtype)
    assert getattr(rec_10["t"].dt, "tz", None) is None
    acc_10 = _evaluate_recovery(expected_canonical, rec_10, SOURCE_COLS + DERIVED_COLS)
    assert all(acc_10[c] >= 0.999 for c in SOURCE_COLS + DERIVED_COLS)

    print("\n=== Debate Red-Team Attack Variants Summary ===")
    print(f"  ATK-08 (cents + 0..100 risk + all-null): 27/27 source, 5/5 derived, all_null={rep_08.all_null_foreign_cols}")
    print(f"  ATK-09 (missing identity + split card/ts): 20/20 tx source, gaps={ sorted(rep_09.not_recoverable_cols.keys()) }")
    print(f"  ATK-10 (tz-aware ISO 'Z' and '+05:30'):  27/27 source, 5/5 derived, t.dtype={rec_10['t'].dtype}")


def test_unrecoverable_transaction_dt_marked_honestly(sample_20k: pd.DataFrame, tmp_path: Path) -> None:
    """When a variant has only a timestamp (no TransactionDT column) and TransactionID
    cannot be joined back to tx_all(), TransactionDT and uid are honestly marked as
    'not recoverable from this variant' in the proposal rather than guessed."""
    v1 = _make_variant_1(sample_20k)
    mapping = infer_mapping(v1, proposal_dir=tmp_path, allow_txid_lookup=False)
    assert mapping.not_recoverable.get("TransactionDT") == NOT_RECOVERABLE
    assert mapping.not_recoverable.get("uid") == NOT_RECOVERABLE
    assert mapping["TransactionDT"].expr == NOT_RECOVERABLE
    assert mapping["uid"].expr == NOT_RECOVERABLE

    approve(mapping.proposal_id, approver_role="L1")
    df_rec, report = apply(mapping, v1, require_proposal_approval=True)
    assert "TransactionDT" in report.not_recoverable_cols
    assert "uid" in report.not_recoverable_cols
    assert "TransactionDT" not in df_rec.columns
    assert "uid" not in df_rec.columns
    assert "t" in df_rec.columns
    assert "device_profile" in df_rec.columns


def test_human_approval_proposal_and_role_enforcement(sample_20k: pd.DataFrame, tmp_path: Path) -> None:
    v1 = _make_variant_1(sample_20k)
    mapping = infer_mapping(v1, approval_threshold=0.95, proposal_dir=tmp_path)
    flagged = mapping.flagged_for_approval()
    assert len(flagged) > 0
    assert not mapping.proposal_approved
    assert Path(mapping.proposal_path).exists()

    with pytest.raises(PermissionError):
        apply(mapping, v1, require_proposal_approval=True)

    with pytest.raises(PermissionError):
        approve(mapping.proposal_id, approver_role="untrusted_guest")

    df_strict, rep_strict = apply(mapping, v1, strict_approval=True, require_proposal_approval=False)
    for fc in flagged:
        assert fc.canonical_col in rep_strict.skipped_unapproved
        assert fc.canonical_col not in df_strict.columns

    approve(mapping.proposal_id, approver_role="L1_lead")
    assert mapping.proposal_approved
    assert mapping.approved_by_role == "L1_lead"
    assert len(mapping.flagged_for_approval()) == 0

    df_approved, rep_approved = apply(mapping, v1, strict_approval=True, require_proposal_approval=True)
    assert len(rep_approved.skipped_unapproved) == 0
    for fc in flagged:
        assert fc.canonical_col in df_approved.columns


def test_confidence_floor_refuses_to_guess(sample_20k: pd.DataFrame, tmp_path: Path) -> None:
    v1 = _make_variant_1(sample_20k)
    mapping = infer_mapping(v1, confidence_floor=0.935, proposal_dir=tmp_path)
    assert len(mapping.refused_below_floor) > 0
    for col, (cand_expr, score) in mapping.refused_below_floor.items():
        assert col not in mapping.choices
        assert score < 0.935
        assert cand_expr in mapping.unmapped_foreign
