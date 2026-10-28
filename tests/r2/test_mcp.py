"""Tests for assay.r2.mcp_server driving the MCP server via subprocess stdio pipes."""

import json
import subprocess
import sys
import pytest

from assay.r2.mcp_server import MCPServer, R2Store


@pytest.fixture
def mcp_proc():
    proc = subprocess.Popen(
        [sys.executable, "-m", "assay.r2.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    yield proc
    proc.terminate()
    proc.wait(timeout=2)


def send_recv(proc, req: dict) -> dict:
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "Process produced no output"
    return json.loads(line)


def test_mcp_server_initialize(mcp_proc):
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = send_recv(mcp_proc, req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_server_ping_and_notification(mcp_proc):
    mcp_proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    mcp_proc.stdin.flush()

    ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    resp = send_recv(mcp_proc, ping_req)
    assert resp["id"] == 2
    assert resp["result"] == {}


def test_mcp_server_tools_list(mcp_proc):
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = send_recv(mcp_proc, req)
    assert resp["id"] == 3
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    expected = {"get_transaction", "card_window", "holder_history", "device_neighbors", "prior_cases", "write_case"}
    assert expected == tool_names
    for t in tools:
        assert t["inputSchema"]["additionalProperties"] is False


def test_hhf_007_queries(mcp_proc):
    card_id = "C09582-K1"
    txn_id = 3660350

    # 1. get_transaction
    r1 = send_recv(mcp_proc, {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "get_transaction", "arguments": {"txn_id": txn_id}}
    })
    assert "result" in r1
    t1 = json.loads(r1["result"]["content"][0]["text"])
    assert t1["transaction"]["card_id"] == card_id
    assert int(t1["transaction"]["TransactionID"]) == txn_id

    # 2. card_window
    r2 = send_recv(mcp_proc, {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "card_window",
            "arguments": {
                "card_id": card_id,
                "start": "2017-01-01T00:00:00",
                "end": "2017-01-20T00:00:00"
            }
        }
    })
    assert "result" in r2
    t2 = json.loads(r2["result"]["content"][0]["text"])
    assert t2["card_id"] == card_id
    assert isinstance(t2["transactions"], list)
    assert t2["total_rows"] >= 1

    # 3. holder_history
    r3 = send_recv(mcp_proc, {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {"name": "holder_history", "arguments": {"card_id": card_id}}
    })
    assert "result" in r3
    t3 = json.loads(r3["result"]["content"][0]["text"])
    assert t3["customer_id"] == "C09582"
    assert len(t3["transactions"]) >= 1

    # 4. device_neighbors
    r4 = send_recv(mcp_proc, {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {"name": "device_neighbors", "arguments": {"card_id": card_id}}
    })
    assert "result" in r4
    t4 = json.loads(r4["result"]["content"][0]["text"])
    assert "neighbors" in t4

    # 5. prior_cases
    r5 = send_recv(mcp_proc, {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {"name": "prior_cases", "arguments": {"card_id": card_id}}
    })
    assert "result" in r5
    t5 = json.loads(r5["result"]["content"][0]["text"])
    assert "cases" in t5


def test_injection_attempt(mcp_proc):
    req = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "holder_history",
            "arguments": {"card_id": "C09582-K1; DROP TABLE transactions;"}
        }
    }
    resp = send_recv(mcp_proc, req)
    assert "error" in resp
    assert resp["error"]["code"] == -32602


def test_unknown_tool(mcp_proc):
    req = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "tools/call",
        "params": {"name": "execute_raw_gsql", "arguments": {"query": "RUN QUERY all()"}}
    }
    resp = send_recv(mcp_proc, req)
    assert "error" in resp
    assert resp["error"]["code"] in (-32601, -32602)


def test_oversize_range(mcp_proc):
    req = {
        "jsonrpc": "2.0",
        "id": 22,
        "method": "tools/call",
        "params": {
            "name": "card_window",
            "arguments": {
                "card_id": "C09582-K1",
                "start": "2016-01-01T00:00:00",
                "end": "2016-08-01T00:00:00"
            }
        }
    }
    resp = send_recv(mcp_proc, req)
    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "120 days" in resp["error"]["message"]


def test_write_without_approval(mcp_proc):
    case_no_appr = {
        "case_id": "HHF-007",
        "case": {
            "verdict": "confirmed_fraud",
            "pattern": "card_not_present_fraud"
        },
        "initial_recommendation": ["VERIFY_WITH_CUSTOMER"],
        "final_recommendation": ["BLOCK_CARD", "CREATE_CASE", "FILE_REPORT"]
    }
    req = {
        "jsonrpc": "2.0",
        "id": 23,
        "method": "tools/call",
        "params": {
            "name": "write_case",
            "arguments": {"case_json": case_no_appr}
        }
    }
    resp = send_recv(mcp_proc, req)
    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "approval id" in resp["error"]["message"].lower()


def test_write_with_approval(mcp_proc):
    case_with_appr = {
        "case_id": "HHF-007",
        "case": {
            "verdict": "confirmed_fraud",
            "pattern": "card_not_present_fraud"
        },
        "initial_recommendation": ["VERIFY_WITH_CUSTOMER"],
        "final_recommendation": ["BLOCK_CARD", "CREATE_CASE", "FILE_REPORT"],
        "approval_id": "APPR-L2-9982"
    }
    req = {
        "jsonrpc": "2.0",
        "id": 24,
        "method": "tools/call",
        "params": {
            "name": "write_case",
            "arguments": {"case_json": case_with_appr}
        }
    }
    resp = send_recv(mcp_proc, req)
    assert "result" in resp
    t = json.loads(resp["result"]["content"][0]["text"])
    assert t["status"] == "written"
    assert t["case_id"] == "HHF-007"


def test_r2_store_direct():
    store = R2Store()
    row, ref = store.get_transaction(3660350)
    assert row["card_id"] == "C09582-K1"
    assert ref.startswith("query:get_transaction")

    win, ref2 = store.card_window("C09582-K1", "2017-01-01", "2017-01-20")
    assert len(win) >= 1

    hist, ref3 = store.holder_history("C09582-K1")
    assert len(hist) >= 1
