"""Tiny helper server on the pod: read setup/comfy logs and receive chunked uploads.

GET  /log?name=setup.log            -> last 200 lines of /workspace/_status/<name>
GET  /ls?path=rel                    -> listing of /workspace/<rel>
PUT  /put?path=rel&offset=N          -> write body at offset N of /workspace/<rel>
Every request needs header X-Token == $POD_TOKEN.
"""
import os, sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = "/workspace"
TOKEN = os.environ.get("POD_TOKEN", "")


def safe(rel):
    p = os.path.realpath(os.path.join(ROOT, rel))
    # without a network volume /workspace/ComfyUI is a symlink to /opt/ComfyUI (the image)
    if not (p.startswith(ROOT + "/") or p.startswith("/opt/ComfyUI/")):
        raise ValueError("outside workspace")
    return p


class H(BaseHTTPRequestHandler):
    def _ok(self):
        if not TOKEN or self.headers.get("X-Token") != TOKEN:
            self.send_error(403)
            return False
        return True

    def _send(self, code, body, ctype="text/plain"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not self._ok():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/log":
                p = safe("_status/" + os.path.basename(q.get("name", ["setup.log"])[0]))
                lines = open(p, errors="replace").read().splitlines()[-200:] if os.path.exists(p) else ["(no log)"]
                return self._send(200, "\n".join(lines))
            if u.path == "/smi":   # quién ocupa la GPU (17-oct: 94 GB tomados sin que ComfyUI tuviera nada)
                import subprocess
                out = subprocess.run("nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; "
                                     "ps -eo pid,rss,args --sort=-rss | head -15", shell=True, capture_output=True, text=True)
                return self._send(200, out.stdout + out.stderr)
            if u.path == "/ls":
                p = safe(q.get("path", [""])[0] or ".") if q.get("path", [""])[0] else ROOT
                out = [{"name": n, "size": os.path.getsize(os.path.join(p, n))} for n in sorted(os.listdir(p))]
                return self._send(200, json.dumps(out), "application/json")
        except Exception as e:
            return self._send(400, str(e))
        self.send_error(404)

    def do_PUT(self):
        if not self._ok():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            p = safe(q["path"][0])
            off = int(q.get("offset", ["0"])[0])
            n = int(self.headers.get("Content-Length", "0"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            mode = "r+b" if os.path.exists(p) and off > 0 else "wb"
            with open(p, mode) as f:
                f.seek(off)
                f.write(self.rfile.read(n))
            return self._send(200, json.dumps({"size": os.path.getsize(p)}), "application/json")
        except Exception as e:
            return self._send(400, str(e))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8000), H).serve_forever()
