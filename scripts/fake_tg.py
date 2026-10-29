"""Fake TigerGraph for drills: answers every REST call with a broken reply.

    python scripts/fake_tg.py 14999 empty|obj|list|garbage|e500
    ASSAY_TG=1 ASSAY_TG_PORT=14999 python -m assay.r2.agent HHF-005 --out /tmp/x   # must exit 3, no answer
"""
import sys, http.server
MODE = sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def _reply(self):
        body = {"empty": b"", "obj": b"{}", "list": b"[]", "garbage": b"<html>not json", "e500": b'{"error":true}'}[MODE]
        self.send_response(500 if MODE == "e500" else 200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    do_GET = do_POST = _reply
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
