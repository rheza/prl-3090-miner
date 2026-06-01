"""Minimal AlphaPool Stratum-Pearl client helpers.

This module intentionally implements only the public wire shape observed from a
normal AlphaMiner black-box connection. It does not solve AlphaPool's
``pearl.challenge`` because the challenge algorithm is not public in the
sources checked by this repo.
"""

from __future__ import annotations

import base64
import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class AlphaPoolError(RuntimeError):
    pass


class AlphaPoolChallengeRequired(AlphaPoolError):
    def __init__(self, seed: str, difficulty: int):
        super().__init__(
            "AlphaPool requires pearl.challenge_response before authorize "
            f"(seed={seed[:8]}..., difficulty={difficulty})"
        )
        self.seed = seed
        self.difficulty = difficulty


@dataclass(frozen=True)
class AlphaPoolEndpoint:
    host: str
    port: int
    tls: bool = False

    @classmethod
    def parse(cls, pool: str) -> "AlphaPoolEndpoint":
        if "://" not in pool:
            pool = "stratum+tcp://" + pool
        u = urlparse(pool)
        if u.scheme not in ("stratum+tcp", "stratum+ssl", "stratum+tls"):
            raise ValueError(f"unsupported pool scheme: {u.scheme}")
        if not u.hostname or not u.port:
            raise ValueError("pool must include host and port")
        return cls(u.hostname, int(u.port), tls=u.scheme in ("stratum+ssl", "stratum+tls"))


@dataclass(frozen=True)
class AlphaPoolJob:
    job_id: str
    prev_hash: str
    header_blob: str
    height: int
    ntime: str
    share_nbits: str
    clean_jobs: bool

    @classmethod
    def from_notify_params(cls, params: list[Any]) -> "AlphaPoolJob":
        if len(params) != 7:
            raise ValueError(f"expected 7 mining.notify params, got {len(params)}")
        return cls(
            job_id=str(params[0]),
            prev_hash=str(params[1]),
            header_blob=str(params[2]),
            height=int(params[3]),
            ntime=str(params[4]),
            share_nbits=str(params[5]),
            clean_jobs=bool(params[6]),
        )


def alpha_submit_message(req_id: int, worker: str, job_id: str, plain_proof: bytes) -> dict:
    """Build AlphaPool's Pearl submit message.

    Clean-room capture shows AlphaPool v1 submits:
        mining.submit [worker, job_id, base64_plain_proof]

    Current harness proof bytes are not valid for this API; this helper only
    captures the transport shape for the future real PlainProof path.
    """
    return {
        "id": req_id,
        "method": "mining.submit",
        "params": [worker, job_id, base64.b64encode(plain_proof).decode("ascii")],
    }


class AlphaPoolStratumClient:
    def __init__(self, pool: str, timeout: float = 10.0):
        endpoint = AlphaPoolEndpoint.parse(pool)
        if endpoint.tls:
            raise NotImplementedError("TLS stratum is not implemented; use stratum+tcp for probing")
        self.endpoint = endpoint
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._rx = b""
        self._next_id = 1

    def __enter__(self) -> "AlphaPoolStratumClient":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def connect(self) -> None:
        self._sock = socket.create_connection((self.endpoint.host, self.endpoint.port), self.timeout)
        self._sock.settimeout(self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, method: str, params: Any) -> int:
        if self._sock is None:
            raise AlphaPoolError("not connected")
        req_id = self._next_id
        self._next_id += 1
        msg = {"id": req_id, "method": method, "params": params}
        line = json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"
        self._sock.sendall(line)
        return req_id

    def send_raw(self, msg: dict) -> None:
        if self._sock is None:
            raise AlphaPoolError("not connected")
        line = json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"
        self._sock.sendall(line)

    def recv(self) -> dict:
        if self._sock is None:
            raise AlphaPoolError("not connected")
        while b"\n" not in self._rx:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise AlphaPoolError("connection closed")
            self._rx += chunk
        line, self._rx = self._rx.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))
