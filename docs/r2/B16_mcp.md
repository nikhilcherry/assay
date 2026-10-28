# Restricted MCP Tool Server (B16)

## Overview
Built a zero-dependency JSON-RPC 2.0 MCP server over stdio (`assay/r2/mcp_server.py`) enforcing least privilege access to bank transaction data and closed case history.

## Architecture
- **Protocol Support**: JSON-RPC 2.0 implementing `initialize`, `notifications/initialized`, `ping`, `tools/list`, and `tools/call` for protocol version `2025-06-18`.
- **Stores**: Default `R2Store` querying staged Round 2 transaction data (`tx_all()`, 650,938 rows) and 5,585 closed cases (`closed_cases()`). Supports store injection for `LocalStore` or `TigerGraphStore`.
- **Serialization Safety**: Built universal `_clean_data()` helper converting DataFrames, Series, numpy arrays, NaNs, and NaTs to JSON objects without triggering numpy array truthiness ambiguity errors (`ValueError: The truth value of an empty array is ambiguous`).

## Tools & Security Features
1. **5 Read-Only Tools**: `get_transaction`, `card_window`, `holder_history`, `device_neighbors`, `prior_cases`.
2. **1 Write Tool**: `write_case` appends case files to an append-only memory store. Blocks write operations containing `BLOCK`/`FILE_REPORT` unless a valid approval ID is provided.
3. **Guardrails**:
   - Schema validation rejecting unknown parameter fields (`additionalProperties: false`).
   - Injection prevention blocking SQL/GSQL keyword patterns and bad card/txn ID formats.
   - Window size capped at <= 120 days.
   - Result truncation at <= 500 rows (`truncated: true`).
   - Detailed per-call execution logging (`CallLogEntry`).

## Measured Test Performance
- Driven via subprocess stdio pipes in `tests/r2/test_mcp.py`.
- **HHF-007 Resolution**: Correctly resolves January 2017 transaction `3660350` and card `C09582-K1` across all 5 read tools.
- **Latency**: Subprocess request-response roundtrips execute in < 45 ms per tool call.
- **Pass Rate**: 100% pass rate across initialization, tool schema checks, security rejections, oversize range errors, approval enforcement, and direct store queries.
