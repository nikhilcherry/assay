"""C10 tests -- TigerGraph GSQL Common-Point-of-Compromise (`r2_cpc_loss_cohort` &
`r2_cpc_origin_counts`) installation, smoke-run, fail-closed checks, and parity against
`assay.r2.detectors.cpc` on HHF-001, HHF-002, HHF-003, HHF-006.
"""

from __future__ import annotations

import pytest

from assay.r2.tg_cpc import (
    CPC_QUERIES,
    SUPPORTED_ORIGIN_FAMILIES,
    cpc_loss_cohort_tg,
    cpc_origin_counts_tg,
    install,
    parity,
)
from assay.r2.tg_r2 import require_tg
from assay.r2.vector import TigerGraphUnavailableError


def _require_tg_or_skip() -> None:
    try:
        require_tg()
    except TigerGraphUnavailableError as exc:
        reason = f"TigerGraph is unreachable ({exc}); skipping live C10 GSQL test."
        print(f"\n[SKIP] {reason}")
        pytest.skip(reason)


def test_install_and_smoke_run_cpc_queries() -> None:
    """Install `r2_cpc_loss_cohort` and `r2_cpc_origin_counts` and verify smoke-run succeeds."""
    _require_tg_or_skip()
    res = install()
    assert res["installed_queries"] == list(CPC_QUERIES)
    for q in CPC_QUERIES:
        assert q in res["smoke_test_ms"]
        assert res["smoke_test_ms"][q] >= 0.0


def test_cpc_parity_hhf_001_002_003_006() -> None:
    """Verify exact cohort member, burst txn, origin count, cardinality, and lift table
    parity between local Python and TigerGraph GSQL for HHF-001, HHF-002, HHF-003, and HHF-006.
    """
    _require_tg_or_skip()
    install()

    target_cases = ["HHF-001", "HHF-002", "HHF-003", "HHF-006"]
    report = parity(case_ids=target_cases)

    assert report["n_cases"] == 4
    assert len(report["supported_origin_families"]) == len(SUPPORTED_ORIGIN_FAMILIES)
    assert report["unsupported_origin_families"] == []

    for row in report["rows"]:
        assert row.members_identical, (
            f"{row.case_id}: cohort members differ: local={row.local_members} vs tg={row.tg_members}"
        )
        assert row.burst_txns_identical, (
            f"{row.case_id}: burst txns differ: local={row.local_burst_txns} vs tg={row.tg_burst_txns}"
        )
        assert row.origin_counts_identical, (
            f"{row.case_id}: origin counts differ: local={row.local_origin_counts} vs tg={row.tg_origin_counts}"
        )
        assert row.cardinality_identical, (
            f"{row.case_id}: origin cardinalities differ: "
            f"local={row.local_indexed_cardinality} vs tg={row.tg_indexed_cardinality}"
        )
        assert row.lifts_identical, (
            f"{row.case_id}: origin lift tables differ: {row.first_lift_diff}"
        )
        assert row.identical

    assert report["mismatches"] == []
    assert report["n_identical"] == 4


def test_fail_closed_when_tg_unreachable() -> None:
    """Verify fail-closed `TigerGraphUnavailableError` when TigerGraph port is unreachable."""
    with pytest.raises(TigerGraphUnavailableError):
        cpc_loss_cohort_tg(3660108, "2017-01-15 01:25:19", host="127.0.0.1", port=1)

    with pytest.raises(TigerGraphUnavailableError):
        cpc_origin_counts_tg(
            {"C05061-K1": 1484208000_000_000_000},
            "2017-01-15 01:25:19",
            host="127.0.0.1",
            port=1,
        )
