"""C6 tests: install, smoke-run, fail-closed checks, and 18-case parity between
TigerGraph GSQL (`r2_micro_drip_cohort`, `r2_recipient_fanin`) and local Python
(`assay.r2.detectors.micro_charge.cohort_members`, `assay.r2.detectors.fan_in.cohort_members`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from assay.r2 import tg_cohorts
from assay.r2.frames import case_pack, tx_all
from assay.r2.tg_r2 import ping_tg
from assay.r2.vector import TigerGraphUnavailableError


@pytest.fixture(scope="module")
def ensure_tg_installed() -> dict:
    if not ping_tg():
        pytest.skip("Skipping C6 TigerGraph cohort tests: TigerGraph REST++ is unreachable on 14240.")
    info = tg_cohorts.install()
    return info


def test_install_and_smoke_run(ensure_tg_installed: dict) -> None:
    assert "installed_queries" in ensure_tg_installed
    assert set(tg_cohorts.COHORT_QUERIES).issubset(set(ensure_tg_installed["installed_queries"]))
    smoke_ms = ensure_tg_installed["smoke_test_ms"]
    for q in tg_cohorts.COHORT_QUERIES:
        assert q in smoke_ms
        assert smoke_ms[q] >= 0.0


def test_as_of_cutoff_respected(ensure_tg_installed: dict) -> None:
    """When `opened_at` is before a flagged transaction's timestamp, both TG queries must
    see 0 matching flagged transactions as-of `opened_at` and return empty cohorts.
    """
    cp = case_pack()
    first = cp.iloc[0]
    early_ts = pd.Timestamp("2016-07-01 00:00:00")

    md_cards, md_txns, md_params = tg_cohorts.micro_drip_cohort_tg(int(first.flagged_txn_id), early_ts)
    assert md_cards == []
    assert md_txns == {}
    assert md_params["flagged_txn_id"] == int(first.flagged_txn_id)

    fi_cards, fi_txns, fi_params = tg_cohorts.recipient_fanin_tg(int(first.flagged_txn_id), early_ts)
    assert fi_cards == []
    assert fi_txns == {}
    assert fi_params["flagged_txn_id"] == int(first.flagged_txn_id)


def test_fail_closed_on_unreachable_or_malformed() -> None:
    """Both TG wrappers and `_extract_row` must fail closed (`TigerGraphUnavailableError`)
    on an unreachable port or malformed reply.
    """
    with pytest.raises(TigerGraphUnavailableError):
        tg_cohorts.micro_drip_cohort_tg(
            3660108,
            "2017-01-15 01:25:19",
            host="http://127.0.0.1",
            port=19999,
            timeout=1.0,
        )

    with pytest.raises(TigerGraphUnavailableError):
        tg_cohorts.recipient_fanin_tg(
            3660260,
            "2017-01-26 21:09:37",
            host="http://127.0.0.1",
            port=19999,
            timeout=1.0,
        )

    with pytest.raises(TigerGraphUnavailableError):
        tg_cohorts._extract_row([], "r2_micro_drip_cohort", ("fired", "member_cards"))

    with pytest.raises(TigerGraphUnavailableError):
        tg_cohorts._extract_row([{"fired": True}], "r2_micro_drip_cohort", ("fired", "member_cards"))


def test_parity_all_18_cases(ensure_tg_installed: dict) -> None:
    """Run parity across all 18 cases in `case_pack()`, print the comparison table,
    and assert identical member sets and per-card txn IDs on every case where either side fires.
    """
    summary = tg_cohorts.parity()
    assert summary["n_cases"] == 18
    assert summary["n_comparisons"] == 36
    assert len(summary["rows"]) == 36

    # Print detailed breakdown for every case where either local or TG fired
    print("\n--- Detailed Fired Cohorts (Local vs TigerGraph) ---")
    for r in summary["rows"]:
        if r.local_members or r.tg_members:
            n_loc_tx = sum(len(v) for v in r.local_txns.values())
            n_tg_tx = sum(len(v) for v in r.tg_txns.values())
            print(
                f"  {r.case_id} ({r.detector}): local_cards={len(r.local_members)} "
                f"(txns={n_loc_tx}), tg_cards={len(r.tg_members)} (txns={n_tg_tx}), "
                f"members_identical={r.members_identical}, txns_identical={r.txns_identical}\n"
                f"    cards={r.tg_members}"
            )
            assert r.local_members == r.tg_members, (
                f"Member mismatch on {r.case_id} ({r.detector}): "
                f"local={r.local_members} vs tg={r.tg_members}"
            )
            assert r.local_txns == r.tg_txns, (
                f"Per-card txn mismatch on {r.case_id} ({r.detector}): "
                f"local={r.local_txns} vs tg={r.tg_txns}"
            )

    assert summary["mismatches"] == []
