"""Client for the pearl-gateway miner-RPC.

Line-delimited JSON-RPC 2.0 over TCP (127.0.0.1:8337) or UDS (/tmp/pearlgw.sock),
matching `pearl_gateway/miner_rpc/server.py` and `comm/json_rpc_client.py:63-89`.
"""

from __future__ import annotations

import json
import socket

from .protocol import MINING_PAUSED_CODE, MiningJob, PlainProofSubmission


class MiningPaused(Exception):
    """Gateway has no template yet (JSON-RPC -32001). Back off and retry."""


class GatewayError(Exception):
    pass


class GatewayClient:
    def __init__(self, transport: str = "tcp", host: str = "127.0.0.1", port: int = 8337,
                 socket_path: str = "/tmp/pearlgw.sock", timeout: float = 5.0):
        self.transport = transport
        self.host, self.port, self.socket_path, self.timeout = host, port, socket_path, timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        self._id = 0

    # --- connection ---
    def connect(self) -> None:
        if self.transport == "uds":
            if not hasattr(socket, "AF_UNIX"):
                raise GatewayError("UDS transport unavailable on this platform; use tcp")
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect(self.socket_path)
        else:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock = s
        self._buf = b""

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # --- framing ---
    def _call(self, method: str, params: dict) -> object:
        if self._sock is None:
            self.connect()
        self._id += 1
        req = {"jsonrpc": "2.0", "method": method, "params": params, "id": self._id}
        self._sock.sendall((json.dumps(req) + "\n").encode())
        line = self._read_line()
        resp = json.loads(line)
        if resp.get("id") != self._id:
            raise GatewayError(f"id mismatch: sent {self._id}, got {resp.get('id')}")
        if resp.get("error"):
            err = resp["error"]
            if err.get("code") == MINING_PAUSED_CODE:
                raise MiningPaused(err.get("message", "mining_paused"))
            raise GatewayError(f"{err.get('code')}: {err.get('message')}")
        return resp.get("result")

    def _read_line(self) -> bytes:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise GatewayError("gateway closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    # --- methods (server.py:208-220) ---
    def get_mining_info(self) -> MiningJob:
        return MiningJob.from_dict(self._call("getMiningInfo", {}))

    def submit_plain_proof(self, submission: PlainProofSubmission) -> str:
        result = self._call("submitPlainProof", submission.params())
        return str(result)  # "submitted" (fire-and-forget; accept/reject not returned)
