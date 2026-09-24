"""Live mode: investigate any transaction, and watch it in the replay.

    python -m assay.serve            # http://127.0.0.1:8740  (~70 s to load the store)

Serves site/ and one endpoint, /api/investigate?txn=<TransactionID>, which runs
the same Investigator on the reference store as an analyst request on that
transaction and returns a trace in exactly the format site/data/ holds, so the
page plays it like any other case. Nothing is precomputed and nothing is written.

Binds to localhost only: the store holds the full dataset.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from assay.data import ROOT
from assay.investigate import Investigator
from assay.patterns import PatternModel
from assay.trace import TracingStore, trace

HOST, PORT = "127.0.0.1", 8740
SITE = ROOT / "site"

store = TracingStore()
inv = Investigator(store, PatternModel())
lock = threading.Lock()        # one investigation at a time: the store's call log is shared


def investigate(txn_id: int) -> dict:
    f = store.tx.loc[txn_id]
    case = pd.Series({
        "case_id": f"LIVE-{txn_id}", "trigger_type": "analyst_request",
        "opened_at": str(f.t + pd.Timedelta(hours=1)), "flagged_txn_id": txn_id,
        "card_id": f.card_id, "customer_id": f.customer_id,
        "trigger_text": f"Live request: investigate transaction {txn_id} "
                        f"({f.TransactionAmt:,.2f} USD, {f.channel.replace('_', ' ')}, {str(f.t)[:16]}).",
    })
    with lock:
        store.memory = []
        t = trace(inv, store, case)
    t["kind"] = "live"
    return json.loads(json.dumps(t, default=str))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SITE), **kw)

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/health":
            return self._json(200, {"ok": True, "transactions": int(len(store.tx))})
        if u.path == "/api/investigate":
            raw = parse_qs(u.query).get("txn", [""])[0]
            if not raw.isdigit():
                return self._json(400, {"error": "txn must be a TransactionID, for example 3478561"})
            txn = int(raw)
            if txn not in store.tx.index:
                return self._json(404, {"error": f"No transaction {txn} in the dataset "
                                                 f"(IDs run {int(store.tx.index.min())}-{int(store.tx.index.max())})."})
            try:
                return self._json(200, investigate(txn))
            except Exception as e:                      # surface the reason, don't hang the page
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        return super().do_GET()

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))


def main() -> int:
    print(f"assay live: http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
