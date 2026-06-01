#!/usr/bin/env python3
"""Probe AlphaPool's Pearl Stratum endpoint without submitting shares."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miner.alphapool_client import (  # noqa: E402
    AlphaPoolChallengeRequired,
    AlphaPoolJob,
    AlphaPoolStratumClient,
)


def _redact_worker(worker: str) -> str:
    if "." not in worker:
        return worker[:10] + "..." if len(worker) > 14 else worker
    addr, suffix = worker.split(".", 1)
    return f"{addr[:10]}...{addr[-6:]}.{suffix}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default="stratum+tcp://us2.alphapool.tech:5566")
    p.add_argument("--address", required=True, help="public prl1p... payout address")
    p.add_argument("--worker", default="prl3090-probe")
    p.add_argument("--password", default="x;d=32768")
    p.add_argument("--challenge-nonce", default=None,
                   help="manual challenge nonce, if a public solver is available")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--redact", action="store_true", help="redact address in output")
    args = p.parse_args()

    worker = f"{args.address}.{args.worker}"
    shown_worker = _redact_worker(worker) if args.redact else worker
    print(json.dumps({"event": "connect", "pool": args.pool, "worker": shown_worker}))

    with AlphaPoolStratumClient(args.pool, timeout=args.timeout) as client:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            msg = client.recv()
            method = msg.get("method")
            if method == "pearl.challenge":
                params = msg.get("params") or {}
                seed = str(params.get("seed", ""))
                difficulty = int(params.get("difficulty", 0))
                print(json.dumps({
                    "event": "challenge",
                    "seed_prefix": seed[:8],
                    "difficulty": difficulty,
                    "solver": "not implemented",
                }))
                if not args.challenge_nonce:
                    raise AlphaPoolChallengeRequired(seed, difficulty)
                client.send("pearl.challenge_response", {
                    "seed": seed,
                    "nonce": args.challenge_nonce,
                })
                continue
            print(json.dumps({"event": "unexpected_before_configure", "message": msg}))
            break

        configure_id = client.send("mining.configure", [["pearl/v1"], {}])
        subscribe_id = client.send("mining.subscribe", ["prl3090-miner/probe"])
        authorize_id = client.send("mining.authorize", [worker, args.password])

        got = {"configure": False, "subscribe": False, "authorize": False,
               "params": False, "difficulty": False, "job": False}
        end = time.monotonic() + args.timeout
        while time.monotonic() < end and not got["job"]:
            msg = client.recv()
            method = msg.get("method")
            if msg.get("id") == configure_id:
                got["configure"] = True
                print(json.dumps({"event": "configured", "result": msg.get("result")}))
            elif msg.get("id") == subscribe_id:
                got["subscribe"] = True
                print(json.dumps({"event": "subscribed", "result": msg.get("result")}))
            elif msg.get("id") == authorize_id:
                got["authorize"] = bool(msg.get("result"))
                print(json.dumps({"event": "authorized", "result": msg.get("result"),
                                  "error": msg.get("error")}))
            elif method == "pearl.set_mining_params":
                got["params"] = True
                params = (msg.get("params") or [{}])[0]
                print(json.dumps({"event": "mining_params", "params": params}))
            elif method == "mining.set_difficulty":
                got["difficulty"] = True
                print(json.dumps({"event": "difficulty", "params": msg.get("params")}))
            elif method == "mining.notify":
                got["job"] = True
                job = AlphaPoolJob.from_notify_params(msg.get("params") or [])
                print(json.dumps({"event": "job", "job": job.__dict__}))
            else:
                print(json.dumps({"event": "message", "message": msg}))

    print(json.dumps({"event": "done", "ready_for_submit": got["job"] and got["authorize"]}))
    return 0 if got["job"] and got["authorize"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AlphaPoolChallengeRequired as exc:
        print(json.dumps({
            "event": "blocked",
            "reason": "alphapool_challenge_solver_missing",
            "detail": str(exc),
        }))
        raise SystemExit(3)
