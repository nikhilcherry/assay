"""Tests for assay.r2.harness (BLOCK B9).

Covers fault injection (down/timeout/garbage at call n), input validation,
canonical double-run byte equality, atomic writing, mid-run failure at call n=2, and fail-closed behavior.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from assay.r2.harness import (
    DependencyDown,
    FaultInjector,
    GarbageResponse,
    RunLedger,
    ToolProxy,
    ToolTimeout,
    idempotency_key,
    run_case,
    write_answer,
)


class FakeStore:
    """Mock store providing sample DataFrame return methods."""

    def __init__(self):
        self.call_log = []

    def get_transaction(self, txn_id: int):
        self.call_log.append(("get_transaction", txn_id))
        return pd.DataFrame({"TransactionID": [txn_id], "amount": [123.45678]})

    def card_window(self, card_id: str, hours: float = 24.0):
        self.call_log.append(("card_window", card_id, hours))
        return pd.DataFrame({"card_id": [card_id], "amount": [50.0]})


def fake_investigate_valid(case_row: dict, store: ToolProxy) -> dict:
    df_tx = store.get_transaction(int(case_row["flagged_txn_id"]))
    df_win = store.card_window(case_row["card_id"])
    return {
        "case_id": case_row["case_id"],
        "summary": "Investigated case successfully",
        "exposure_usd": float(df_tx["amount"].iloc[0]),
        "affected_txn_ids": [int(case_row["flagged_txn_id"])],
        "connected_card_ids": ["C100-K2", "C100-K1"],
    }


def test_proxy_and_fault_injection_down():
    store = FakeStore()
    proxy = ToolProxy(store, timeout_s=1.0)

    # First call succeeds (call count = 1)
    res1 = proxy.get_transaction(101)
    assert not res1.empty

    # Fault injected on second call (call count = 2)
    with FaultInjector("get_transaction:down@2"):
        with pytest.raises(DependencyDown):
            proxy.get_transaction(102)

    assert proxy.calls[-1].ok is False
    assert proxy.calls[-1].error_type == "DependencyDown"


def test_proxy_fault_timeout():
    store = FakeStore()
    proxy = ToolProxy(store, timeout_s=0.2)

    with FaultInjector("get_transaction:timeout@1"):
        with pytest.raises(ToolTimeout):
            proxy.get_transaction(101)

    assert proxy.calls[-1].ok is False
    assert proxy.calls[-1].error_type == "ToolTimeout"


def test_proxy_fault_garbage_and_validator():
    store = FakeStore()

    def df_validator(res):
        return isinstance(res, pd.DataFrame) and "TransactionID" in res.columns

    proxy = ToolProxy(store, timeout_s=1.0, validators={"get_transaction": df_validator})

    with FaultInjector("get_transaction:garbage@1"):
        with pytest.raises(GarbageResponse):
            proxy.get_transaction(101)

    assert proxy.calls[-1].ok is False
    assert proxy.calls[-1].error_type == "GarbageResponse"


def test_invalid_input_rejection():
    store = FakeStore()
    opened_at = "2017-01-15 10:00:00"

    # Hostile SQL injection string
    hostile_case = {
        "case_id": "HHF-001",
        "flagged_txn_id": 1001,
        "card_id": "C001-K1'; DROP TABLE transactions; --",
        "customer_id": "C001",
        "opened_at": opened_at,
    }
    res = run_case(hostile_case, fake_investigate_valid, store)
    assert res.status == "REJECTED_INPUT"
    assert "Hostile or invalid input" in str(res.error)

    # Negative amount
    neg_case = {
        "case_id": "HHF-002",
        "flagged_txn_id": 1002,
        "card_id": "C001-K1",
        "customer_id": "C001",
        "opened_at": opened_at,
        "amount": -50.0,
    }
    res_neg = run_case(neg_case, fake_investigate_valid, store)
    assert res_neg.status == "REJECTED_INPUT"
    assert "Negative transaction amount" in str(res_neg.error)

    # Out of range risk score
    bad_rs_case = {
        "case_id": "HHF-003",
        "flagged_txn_id": 1003,
        "card_id": "C001-K1",
        "customer_id": "C001",
        "opened_at": opened_at,
        "risk_score": 1.5,
    }
    res_rs = run_case(bad_rs_case, fake_investigate_valid, store)
    assert res_rs.status == "REJECTED_INPUT"
    assert "out of range" in str(res_rs.error)


def test_canonical_double_run_byte_equality():
    store1 = FakeStore()
    store2 = FakeStore()
    case_row = {
        "case_id": "HHF-010",
        "flagged_txn_id": 2001,
        "card_id": "C010-K1",
        "customer_id": "C010",
        "opened_at": "2017-01-15 12:00:00",
        "risk_score": 0.5,
    }

    res1 = run_case(case_row, fake_investigate_valid, store1)
    res2 = run_case(case_row, fake_investigate_valid, store2)

    assert res1.status == "COMPLETE"
    assert res2.status == "COMPLETE"

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        sha1 = write_answer(res1, dir_path)
        content1 = (dir_path / "HHF-010.json").read_bytes()

        sha2 = write_answer(res2, dir_path)
        content2 = (dir_path / "HHF-010.json").read_bytes()

        assert sha1 == sha2
        assert content1 == content2


def test_fault_at_call_n2_mid_run():
    store = FakeStore()
    case_row = {
        "case_id": "HHF-099",
        "flagged_txn_id": 9001,
        "card_id": "C099-K1",
        "customer_id": "C099",
        "opened_at": "2017-01-18 10:00:00",
    }

    # Call 1 is get_transaction(101) -> succeeds.
    # Call 2 is card_window(C099-K1) -> fails with down@1 for card_window.
    with FaultInjector("card_window:down@1"):
        res = run_case(case_row, fake_investigate_valid, store)

    assert res.status == "INCOMPLETE"
    assert res.error_type == "DependencyDown"

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        sha = write_answer(res, dir_path, case_id="HHF-099")

        # Must leave NO HHF-099.json on disk
        assert not (dir_path / "HHF-099.json").exists()

        # Must write HHF-099.incomplete.json containing error details
        incomplete_file = dir_path / "HHF-099.incomplete.json"
        assert incomplete_file.exists()

        data = json.loads(incomplete_file.read_text(encoding="utf-8"))
        assert data["status"] == "INCOMPLETE"
        assert data["error_type"] == "DependencyDown"
        assert sha is not None


def test_no_saved_file_written_on_failure():
    store = FakeStore()
    case_row = {
        "case_id": "HHF-020",
        "flagged_txn_id": 3001,
        "card_id": "C020-K1",
        "customer_id": "C020",
        "opened_at": "2017-01-16 08:00:00",
    }

    with FaultInjector("get_transaction:down@1"):
        res = run_case(case_row, fake_investigate_valid, store)

    assert res.status == "INCOMPLETE"

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        sha = write_answer(res, dir_path, case_id="HHF-020")

        # Must write HHF-020.incomplete.json and NOT HHF-020.json
        assert (dir_path / "HHF-020.incomplete.json").exists()
        assert not (dir_path / "HHF-020.json").exists()
        assert sha is not None


def test_idempotency_key_and_run_ledger():
    case_row = {
        "case_id": "HHF-030",
        "flagged_txn_id": 4001,
        "card_id": "C030-K1",
        "customer_id": "C030",
        "opened_at": "2017-01-17 09:00:00",
    }
    key1 = idempotency_key(case_row, "dataset_v1", "commit_abc")
    key2 = idempotency_key(case_row, "dataset_v1", "commit_abc")
    assert key1 == key2

    store = FakeStore()
    res = run_case(case_row, fake_investigate_valid, store)

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        sha = write_answer(res, dir_path)
        content = (dir_path / "HHF-030.json").read_bytes()

        ledger = RunLedger(dir_path / "ledger")
        ledger.record(key1, res, sha, content)

        cached = ledger.get(key1)
        assert cached is not None
        cached_res, cached_sha, cached_content = cached
        assert cached_res.status == "COMPLETE"
        assert cached_sha == sha
        assert cached_content == content
