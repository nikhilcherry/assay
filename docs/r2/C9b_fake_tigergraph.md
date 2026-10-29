# Drill: a TigerGraph that answers with garbage

`docs/r2/C9_chaos.md` covers 25 fault scenarios (21 PASS, 0 FAIL). Its four in-process TigerGraph
faults never fired, because the agent reaches TigerGraph through `pyTigerGraph`/`requests` rather than
`urllib`, so they are marked INVALID rather than counted. This drill replaces them with a real
network-level fault: a fake TigerGraph (`scripts/fake_tg.py`) answers every REST call with a broken reply.
HHF-005 is the case where the graph decides the ring and both cohorts.

```bash
python scripts/fake_tg.py 14999 <mode> &
ASSAY_TG=1 ASSAY_TG_PORT=14999 python -m assay.r2.agent HHF-005 --out /tmp/x
```

| mode | reply | exit | answer file | first line |
|---|---|---|---|---|
| `empty` | 200, empty body | 3 | none | `FAIL CLOSED [EXIT 3] case=HHF-005 error_type=JSONDecodeError` |
| `obj` | 200, `{}` | 3 | none | `FAIL CLOSED [EXIT 3] ... TigerGraph query 'r2_recipient_fanin' returned empty/malformed` |
| `list` | 200, `[]` | 3 | none | `FAIL CLOSED [EXIT 3] case=HHF-005 error_type=AttributeError` |
| `garbage` | 200, `<html>not json` | 3 | none | `FAIL CLOSED [EXIT 3] case=HHF-005 error_type=JSONDecodeError` |
| `e500` | HTTP 500 | 3 | none | `FAIL CLOSED [EXIT 3] ... returned HTTP 500` |

In every mode the run stops, says why, and writes no answer. It never falls back to local data.
Stopping a run (SIGTERM) ends it the same way: exit 3, `Interrupted: SIGTERM`, no answer.
