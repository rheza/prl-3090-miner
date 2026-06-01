#!/usr/bin/env python3
"""Run or parse a pool benchmark and emit protocol TH/s JSON.

This is a clean-room benchmarking helper: it treats any miner as a black box,
counts accepted/rejected/stale pool shares from stdout, and converts accepted
share difficulty into pool-credited TH/s. It does not inspect miner internals.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miner.protocol_hashrate import (  # noqa: E402
    DEFAULT_RTX3090_SHARE_DIFF,
    DEFAULT_STRATUM_SHARE_UNIT,
    ShareStats,
    parse_lines,
)


@dataclass
class TelemetrySampler:
    device: int = 0
    interval_s: float = 5.0
    samples: list[dict] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = query_nvidia_smi(self.device)
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict:
        if not self.samples:
            return {"power_avg_w": None, "gpu_temp_avg_c": None, "sm_clock_avg_mhz": None}
        def avg(key: str) -> float | None:
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None
        return {
            "power_avg_w": avg("power_w"),
            "gpu_temp_avg_c": avg("gpu_temp_c"),
            "sm_clock_avg_mhz": avg("sm_clock_mhz"),
            "samples": len(self.samples),
        }


def query_nvidia_smi(device: int) -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(device),
                "--query-gpu=power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    parts = [p.strip() for p in out.strip().split(",")]
    if len(parts) < 3:
        return None
    return {
        "power_w": _float_or_none(parts[0]),
        "gpu_temp_c": _float_or_none(parts[1]),
        "sm_clock_mhz": _float_or_none(parts[2]),
    }


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def run_command(args) -> tuple[ShareStats, float, dict]:
    if not args.command:
        raise SystemExit("command mode requires arguments after --")
    log_path = pathlib.Path(args.log_file or f"protocol_bench_{int(time.time())}.log")
    sampler = TelemetrySampler(device=args.device, interval_s=args.telemetry_interval)
    stats = ShareStats(last_share_diff=args.share_diff)
    start = time.monotonic()
    sampler.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                args.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            timer = threading.Timer(args.duration, proc.terminate)
            timer.start()
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    print(line, end="")
                    log.write(line)
                    stats.note_line(line, args.share_diff)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            finally:
                timer.cancel()
    finally:
        sampler.stop()
    elapsed = time.monotonic() - start
    telemetry = sampler.summary()
    telemetry["log_file"] = str(log_path)
    return stats, elapsed, telemetry


def parse_log_files(args) -> tuple[ShareStats, float, dict]:
    lines: list[str] = []
    for p in args.parse_log:
        lines.extend(pathlib.Path(p).read_text(encoding="utf-8", errors="replace").splitlines())
    elapsed = args.elapsed_sec or args.duration
    return parse_lines(lines, args.share_diff), elapsed, {"log_file": [str(p) for p in args.parse_log]}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure Pearl protocol TH/s from accepted pool shares")
    p.add_argument("--duration", type=float, default=600.0, help="benchmark duration for command mode")
    p.add_argument("--elapsed-sec", type=float, default=None, help="elapsed seconds for parse-only mode")
    p.add_argument("--share-diff", type=float, default=DEFAULT_RTX3090_SHARE_DIFF,
                   help="static pool share difficulty; AlphaMiner recommends 32768 for RTX 3090")
    p.add_argument("--share-unit", type=float, default=DEFAULT_STRATUM_SHARE_UNIT,
                   help="work units per difficulty-1 share; conventional Stratum default is 2^32")
    p.add_argument("--pool-credited-th-s", type=float, default=None,
                   help="optional pool dashboard value to include beside local share accounting")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--telemetry-interval", type=float, default=5.0)
    p.add_argument("--log-file", default=None)
    p.add_argument("--output", default=None, help="write JSON summary to this path")
    p.add_argument("--parse-log", nargs="*", default=None, help="parse existing log file(s) instead of running")
    p.add_argument("command", nargs=argparse.REMAINDER, help="miner command after --")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.parse_log:
        stats, elapsed, meta = parse_log_files(args)
    else:
        stats, elapsed, meta = run_command(args)

    summary = stats.summary(elapsed, args.share_unit, power_avg_w=meta.get("power_avg_w"))
    summary.update({
        "pool_dashboard_th_s": args.pool_credited_th_s,
        "benchmark_kind": "protocol_share_rate",
        "static_password_hint": f"x;d={int(args.share_diff)}",
    })
    summary.update(meta)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
