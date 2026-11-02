"""The verifier's only way into the product (scripts/verify): a loopback server, started outside the role sandbox, that
runs the `requisite` CLI of a clean copy of the committed product, replaying recorded responses only, and serves the
files the product writes for a user to read (its case stores' outputs and dossiers, under var/). It never serves the
product's code, its tests or its cassettes, so the verifier judges the product by using it, not by reading it.

POST /run   {"args": [...], "stdin": "..."}  ->  {"exit": n, "stdout": "...", "stderr": "..."}
GET  /file?path=var/...                      ->  the file's bytes (only under var/)
GET  /ls?path=var/...                        ->  the directory's entries (only under var/)

Usage: python3 bridge.py <product dir> <port file>   (writes the port it listens on into the port file)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PRODUCT = Path(sys.argv[1]).resolve()
READABLE = PRODUCT / "var"
TIMEOUT_S = 3 * 3600  # a step of the flow replays many recorded calls; none runs longer than this
LIMIT = 4_000_000  # characters of output returned per stream


def _under(raw: str) -> Path | None:
    path = (PRODUCT / raw).resolve()
    return path if path == READABLE or READABLE in path.parents else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet: the run log is the verifier's transcript
        pass

    def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/run":
            return self._send(404, b'{"error": "no such endpoint"}')
        request = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
        args = [str(a) for a in request.get("args", [])]
        env = {**os.environ, "REQUISITE_REPLAY_ONLY": "1"}
        try:
            done = subprocess.run(  # noqa: S603 (the product's own CLI, from its own venv)
                [str(PRODUCT / ".venv" / "bin" / "requisite"), *args], cwd=PRODUCT, env=env, capture_output=True,
                text=True, input=request.get("stdin"), timeout=TIMEOUT_S,
            )
            out = {"exit": done.returncode, "stdout": done.stdout[-LIMIT:], "stderr": done.stderr[-LIMIT:]}
        except subprocess.TimeoutExpired:
            out = {"exit": 124, "stdout": "", "stderr": f"the command ran longer than {TIMEOUT_S} s and was stopped"}
        self._send(200, json.dumps(out).encode())

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        raw = parse_qs(url.query).get("path", [""])[0]
        path = _under(raw)
        if path is None:
            return self._send(403, b'{"error": "only what the product writes under var/ is readable"}')
        if url.path == "/file" and path.is_file():
            return self._send(200, path.read_bytes(), "application/octet-stream")
        if url.path == "/ls" and path.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return self._send(200, json.dumps(entries).encode())
        return self._send(404, b'{"error": "no such file or directory"}')


def main() -> None:
    READABLE.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Path(sys.argv[2]).write_text(str(server.server_address[1]))
    server.serve_forever()


if __name__ == "__main__":
    main()
