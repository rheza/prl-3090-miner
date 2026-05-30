"""In-process mock of the pearl-gateway miner-RPC, for self-test / CI.

Speaks the exact line-delimited JSON-RPC wire of `miner_rpc/server.py` so the real
GatewayClient drives it unmodified. Lets us validate the full mine->submit->stale-cancel
loop with no node, no GPU, no network dependencies.

It deliberately rotates the work header every `rotate_after` getMiningInfo calls to
exercise stale-job detection, and counts submissions it accepts/drops by header.
"""

from __future__ import annotations

import base64
import json
import os
import socketserver
import threading

from .protocol import MINING_PAUSED_CODE


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        gw = self.server.gw  # type: ignore[attr-defined]
        for raw in self.rfile:
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rid = req.get("id")
            method = req.get("method")
            try:
                if method == "getMiningInfo":
                    job = gw._next_job()
                    if job is None:
                        resp = {"jsonrpc": "2.0", "id": rid,
                                "error": {"code": MINING_PAUSED_CODE, "message": "mining_paused"}}
                    else:
                        resp = {"jsonrpc": "2.0", "id": rid, "result": job}
                elif method == "submitPlainProof":
                    gw._record_submit(req.get("params", {}))
                    resp = {"jsonrpc": "2.0", "id": rid, "result": "submitted"}
                else:
                    resp = {"jsonrpc": "2.0", "id": rid,
                            "error": {"code": -32601, "message": "method not found"}}
            except Exception as exc:  # pragma: no cover
                resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(exc)}}
            self.wfile.write((json.dumps(resp) + "\n").encode())
            self.wfile.flush()


class MockGateway:
    """A localhost TCP gateway stand-in. Use `with MockGateway(...) as gw:`."""

    def __init__(self, target: int, rotate_after: int = 3, paused_first: int = 0):
        self.target = target
        self.rotate_after = rotate_after
        self.paused_first = paused_first
        self._calls = 0
        self._epoch = 0
        self.submissions: list[dict] = []
        self.accepted_by_header: dict[bytes, int] = {}
        self.dropped_stale = 0
        self._lock = threading.Lock()
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None
        self.host, self.port = "127.0.0.1", 0

    def _current_header(self) -> bytes:
        # 80-byte header whose identity changes each epoch (stand-in for a real tip change)
        return (b"PEARLHDR" + self._epoch.to_bytes(8, "little")).ljust(80, b"\x00")

    def _next_job(self):
        with self._lock:
            self._calls += 1
            if self._calls <= self.paused_first:
                return None
            if self.rotate_after and (self._calls - self.paused_first) % self.rotate_after == 0:
                self._epoch += 1
            hdr = self._current_header()
            return {"incomplete_header_bytes": base64.b64encode(hdr).decode(), "target": self.target}

    def _record_submit(self, params: dict) -> None:
        with self._lock:
            self.submissions.append(params)
            hdr = base64.b64decode(params["mining_job"]["incomplete_header_bytes"])
            # Gateway drops submissions for a header that is no longer current (server.py:247-252)
            if hdr == self._current_header():
                self.accepted_by_header[hdr] = self.accepted_by_header.get(hdr, 0) + 1
            else:
                self.dropped_stale += 1

    def __enter__(self) -> "MockGateway":
        gw = self

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((self.host, 0), _Handler)
        self._server.gw = gw  # type: ignore[attr-defined]
        self.host, self.port = self._server.server_address
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
