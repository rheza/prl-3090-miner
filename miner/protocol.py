"""Wire types for the gateway miner-RPC (grounded in pearl-gateway source).

MiningJob matches `pearl_gateway/comm/dataclasses.py:148-200` and the JSON schema in
`pearl_gateway/miner_rpc/schemas.py:35-42`:
    {"incomplete_header_bytes": <base64 str>, "target": <int>}

The transport is line-delimited JSON-RPC 2.0 (one object per '\n'), no auth
(`miner_rpc/server.py:142-164`). Methods: getMiningInfo, submitPlainProof.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

MAX_TARGET = 2**256 - 1            # dataclasses.py:156
INNER_HASH_LIMIT = 42             # dataclasses.py:155
MINING_PAUSED_CODE = -32001       # dataclasses.py:203-211


@dataclass(frozen=True)
class MiningJob:
    incomplete_header_bytes: bytes
    target: int

    def to_dict(self) -> dict:
        return {
            "incomplete_header_bytes": base64.b64encode(self.incomplete_header_bytes).decode(),
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MiningJob":
        hdr = base64.b64decode(d["incomplete_header_bytes"])
        target = int(d["target"])
        if not (0 <= target <= MAX_TARGET):
            raise ValueError(f"target out of range: {target}")
        if not hdr:
            raise ValueError("incomplete_header_bytes is empty")
        return cls(hdr, target)

    @property
    def header_id(self) -> bytes:
        """Identity used for stale detection — the gateway keys staleness on the
        exact incomplete header bytes (server.py:247-252)."""
        return self.incomplete_header_bytes


@dataclass(frozen=True)
class PlainProofSubmission:
    """What submitPlainProof carries (server.py:214-220). `plain_proof` is the opaque
    base64 blob produced by py-pearl-mining in the full stack; here the backend
    supplies its bytes."""
    plain_proof: bytes
    job: MiningJob

    def params(self) -> dict:
        return {
            "plain_proof": base64.b64encode(self.plain_proof).decode(),
            "mining_job": self.job.to_dict(),
        }
