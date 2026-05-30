"""Load and validate miner.toml (PRD §11.2). Uses stdlib tomllib (Python 3.11+).

Secrets are never read from the file: the RPC password comes from the env var named
by node.rpc_password_env. A world-readable config triggers a warning (PRD §26).
"""

from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass, field


@dataclass
class Config:
    mode: str = "simnet"
    mining_address: str = ""
    rpc_url: str = "https://127.0.0.1:44107"
    rpc_user: str = "rpcuser"
    rpc_password: str = ""
    transport: str = "tcp"
    host: str = "127.0.0.1"
    port: int = 8337
    socket_path: str = "/tmp/pearlgw.sock"
    poll_interval_s: float = 1.0
    backend: str = "cpu"
    devices: list[int] = field(default_factory=lambda: [0])
    max_temp_c: float = 78
    max_vram_temp_c: float = 96
    apply_overclock: bool = False
    stale_cancel: bool = True
    status_interval_s: float = 30.0
    exit_after_invalid: int = 10
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        c = cls()
        c.warnings = []

        # config-file permission check (PRD §26)
        try:
            mode = os.stat(path).st_mode
            if mode & (stat.S_IROTH | stat.S_IWOTH):
                c.warnings.append(f"config file {path} is world-readable/writable; chmod 600 it")
        except OSError:
            pass

        m = raw.get("mode", {})
        c.mode = m.get("type", c.mode)
        c.mining_address = raw.get("wallet", {}).get("mining_address", c.mining_address)

        node = raw.get("node", {})
        c.rpc_url = node.get("rpc_url", c.rpc_url)
        c.rpc_user = node.get("rpc_user", c.rpc_user)
        env_name = node.get("rpc_password_env", "PEARLD_RPC_PASSWORD")
        c.rpc_password = os.environ.get(env_name, "")
        if not c.rpc_password and c.mode in ("solo-local", "mainnet"):
            c.warnings.append(f"env {env_name} is empty; pearld RPC auth will fail")

        gw = raw.get("gateway", {})
        c.transport = gw.get("transport", c.transport)
        c.host = gw.get("host", c.host)
        c.port = int(gw.get("port", c.port))
        c.socket_path = gw.get("socket_path", c.socket_path)
        c.poll_interval_s = float(gw.get("poll_interval_ms", c.poll_interval_s * 1000)) / 1000.0

        gpu = raw.get("gpu", {})
        c.backend = gpu.get("backend", c.backend)
        c.devices = list(gpu.get("devices", c.devices))
        c.max_temp_c = float(gpu.get("max_temp_c", c.max_temp_c))
        c.max_vram_temp_c = float(gpu.get("max_vram_temp_c", c.max_vram_temp_c))
        c.apply_overclock = bool(gpu.get("apply_overclock", c.apply_overclock))

        perf = raw.get("performance", {})
        c.stale_cancel = bool(perf.get("stale_cancel", c.stale_cancel))

        log = raw.get("logging", {})
        c.status_interval_s = float(log.get("status_interval_sec", c.status_interval_s))

        safety = raw.get("safety", {})
        c.exit_after_invalid = int(safety.get("exit_after_invalid_proofs", c.exit_after_invalid))

        # validate the mining address is a public Pearl Taproot address, never a key
        if c.mining_address and not c.mining_address.startswith("prl1p"):
            c.warnings.append(
                "mining_address does not look like a Pearl P2TR (prl1p...) address")
        if any(s in c.mining_address.lower() for s in ("xprv", "seed", "mnemonic")):
            raise ValueError("mining_address must be a PUBLIC address, never a key/seed")
        return c
