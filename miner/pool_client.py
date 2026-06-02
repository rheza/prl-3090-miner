"""Standard, open Stratum v1 client + handshake probe for Pearl pools.

This implements ONLY the public Stratum v1 protocol (mining.subscribe / mining.authorize /
mining.set_difficulty / mining.notify / mining.submit). It does not implement or bypass any
proprietary challenge handshake: if a pool demands one (e.g. AlphaPool/Akoya's pearl.challenge),
the probe detects it, reports it, and stops — no access-control circumvention.

Usage (probe a pool's handshake — connect, subscribe, authorize, observe):
    python miner/pool_client.py --host pool.pearlpool.io --port 34334 --tls \
        --address prl1p... [--worker rig0] [--seconds 15]

A clean run that reaches `mining.notify` means the pool speaks standard Stratum and our miner
can drive it. A `pearl.challenge` / `mining.configure`-required / immediate-drop means the pool
gates with a proprietary handshake we will not reverse-engineer.
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import time


class StratumClient:
    def __init__(self, host: str, port: int, tls: bool, timeout: float = 15.0):
        self.host, self.port, self.tls, self.timeout = host, port, tls, timeout
        self.sock: socket.socket | None = None
        self._rx = b""
        self._id = 0

    def connect(self) -> None:
        raw = socket.create_connection((self.host, self.port), self.timeout)
        if self.tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False          # mining stratum endpoints often use non-CA / SNI-only certs
            ctx.verify_mode = ssl.CERT_NONE
            raw = ctx.wrap_socket(raw, server_hostname=self.host)
        raw.settimeout(self.timeout)
        self.sock = raw

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def send(self, method: str, params) -> int:
        self._id += 1
        line = json.dumps({"id": self._id, "method": method, "params": params}) + "\n"
        self.sock.sendall(line.encode())
        return self._id

    def messages(self, deadline: float):
        """Yield decoded JSON messages until `deadline` (wall clock)."""
        while time.time() < deadline:
            if b"\n" in self._rx:
                line, self._rx = self._rx.split(b"\n", 1)
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line.decode())
                    except json.JSONDecodeError:
                        yield {"_raw": line.decode(errors="replace")}
                continue
            try:
                self.sock.settimeout(max(0.5, deadline - time.time()))
                chunk = self.sock.recv(65536)
            except (socket.timeout, ssl.SSLWantReadError):
                continue
            except OSError as e:
                yield {"_error": f"recv failed: {e}"}
                return
            if not chunk:
                yield {"_closed": True}
                return
            self._rx += chunk


CHALLENGE_HINTS = ("challenge", "pearl.challenge", "authorization required", "unauthorized")


def _collect(c, deadline, watch_ids, label_job_methods=("mining.notify", "job")):
    """Read messages until a job appears, a challenge appears, or the deadline.
    Returns dict of observations."""
    obs = {"job": None, "authorized": None, "challenge": False, "closed": False, "msgs": []}
    for msg in c.messages(deadline):
        if "_closed" in msg:
            obs["closed"] = True
            print("[probe] <- server closed the connection")
            break
        if "_error" in msg:
            print(f"[probe] <- {msg['_error']}")
            break
        obs["msgs"].append(msg)
        blob = json.dumps(msg).lower()
        method = msg.get("method", "")
        res = msg.get("result")
        if msg.get("id") in watch_ids:
            print(f"[probe] <- reply id={msg.get('id')}: {json.dumps(msg)[:200]}")
            # Monero-style login reply embeds the first job in result.job
            if isinstance(res, dict) and ("job" in res or "jobs" in res):
                obs["job"] = res.get("job") or res.get("jobs")
            if res is True:
                obs["authorized"] = True
        elif method in label_job_methods:
            obs["job"] = msg.get("params")
            print(f"[probe] <- JOB via {method}: {json.dumps(msg.get('params'))[:160]}")
        elif method == "mining.set_difficulty":
            print(f"[probe] <- mining.set_difficulty {msg.get('params')}")
        elif method:
            print(f"[probe] <- {method} {json.dumps(msg.get('params'))[:140]}")
        if method.startswith("pearl.challenge") or "challenge_response" in blob \
           or any(h in blob for h in CHALLENGE_HINTS):
            obs["challenge"] = True
            print(f"[probe] <- !! challenge/auth-gate signal: {json.dumps(msg)[:180]}")
        if obs["job"] is not None or obs["challenge"]:
            break
    return obs


def probe(host: str, port: int, tls: bool, address: str, worker: str, seconds: float) -> int:
    user = f"{address}.{worker}" if worker else address
    c = StratumClient(host, port, tls)
    print(f"[probe] connecting to {host}:{port} (tls={tls}) as user={user[:18]}...{user[-6:]}")
    try:
        c.connect()
    except OSError as e:
        print(f"[probe] CONNECT FAILED: {e}\n[probe] (no outbound reachability or wrong host/port)")
        return 2

    # Phase 1: classic Stratum v1 (array params).
    print("[probe] phase 1: classic Stratum v1 (mining.subscribe / mining.authorize, array params)")
    sub_id = c.send("mining.subscribe", ["prl-3090-miner/1.0"])
    auth_id = c.send("mining.authorize", [user, "x"])
    deadline = time.time() + seconds / 2
    obs = _collect(c, deadline, {sub_id, auth_id})

    method_unsupported = any(
        isinstance(m.get("error"), dict) and "not supported" in str(m["error"].get("msg", "")).lower()
        for m in obs["msgs"]
    )
    object_required = any(
        isinstance(m.get("error"), dict) and "object" in str(m["error"].get("msg", "")).lower()
        for m in obs["msgs"]
    )

    # Phase 2: open Monero/Cryptonote-style stratum (login, object params) — what xmrig speaks.
    if obs["job"] is None and not obs["challenge"] and (method_unsupported or object_required or obs["closed"] is False):
        if obs["closed"]:
            print("[probe] connection closed in phase 1; cannot try phase 2")
        else:
            print("[probe] phase 2: open Monero-style stratum (login, object params)")
            login_id = c.send("login", {"login": user, "pass": "x", "agent": "prl-3090-miner/1.0"})
            obs2 = _collect(c, time.time() + seconds / 2, {login_id})
            for k in ("job", "challenge", "closed"):
                obs[k] = obs2[k] if (obs2[k] not in (None, False) or obs[k] in (None, False)) else obs[k]
            obs["authorized"] = obs.get("authorized") or obs2.get("authorized")
    c.close()

    print("\n[probe] ==== verdict ====")
    if obs["challenge"]:
        print("  PROPRIETARY HANDSHAKE — pool requires a challenge/auth extension (not an open protocol).")
        print("  Stopping: this project does NOT reverse-engineer or bypass that gate.")
        return 1
    if obs["job"] is not None:
        print("  OPEN PROTOCOL ✓ — pool issued a real JOB over a public stratum dialect (no secret gate).")
        print("  Mineable with our standard client: job -> GPU search -> PlainProof -> submit.")
        print("  Next: implement this pool's job/submit payload (open dialect) + wire the GPU search.")
        return 0
    if obs["closed"]:
        print("  Pool dropped the open handshakes (likely requires its own client). Not bypassed.")
        return 1
    print("  Inconclusive (no job, no explicit challenge). Try other regional host/port or --tls.")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Standard open Stratum v1 probe for Pearl pools")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--address", default="prl1p56vxsrzefhx8m49c2y4v86up845z6aurrxd3wulep9dk38cz8hcssu8cnm")
    ap.add_argument("--worker", default="rig0")
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()
    return probe(a.host, a.port, a.tls, a.address, a.worker, a.seconds)


if __name__ == "__main__":
    sys.exit(main())
