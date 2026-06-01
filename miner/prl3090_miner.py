"""prl3090-miner CLI (PRD §11.1).

    prl3090-miner run --config miner.toml
    prl3090-miner benchmark --duration 300 [--backend cpu|cuda-sm86]
    prl3090-miner list-devices
    prl3090-miner validate-job --input job.json
    prl3090-miner self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .backends import BackendNotBuilt, list_gpus, make_backend
from .config import Config
from .gateway_client import GatewayClient
from .mock_gateway import MockGateway
from .protocol import MiningJob
from .runtime import LoopConfig, Metrics, SafetyMonitor, mine_loop

EASIEST = 2**256 - 1


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
def cmd_list_devices(_args) -> int:
    gpus = list_gpus()
    if not gpus:
        print("No NVIDIA GPUs detected (nvidia-smi not found or no device).")
    for g in gpus:
        print(f"  [{g['index']}] {g['name']}  {g['memory_total_mib']} MiB  "
              f"driver {g['driver']}  temp {g.get('temp_c','-')}C  power {g.get('power_w','-')}W")
    # backend availability
    for name in ("cpu", "cuda-naive", "cuda-mine", "cuda-sm86"):
        try:
            make_backend(name)
            print(f"  backend '{name}': available")
        except BackendNotBuilt as exc:
            print(f"  backend '{name}': NOT BUILT — {exc}")
        except Exception as exc:
            print(f"  backend '{name}': error — {exc}")
    return 0


def cmd_validate_job(args) -> int:
    with open(args.input, "rb") as fh:
        data = json.load(fh)
    try:
        job = MiningJob.from_dict(data)
    except Exception as exc:
        print(f"INVALID: {exc}")
        return 1
    print(f"VALID job: header={len(job.incomplete_header_bytes)} bytes, "
          f"target={hex(job.target)} (difficulty {'easiest' if job.target == EASIEST else 'set'})")
    return 0


def cmd_benchmark(args) -> int:
    try:
        backend = make_backend(args.backend, device=args.device)
    except BackendNotBuilt as exc:
        print(exc)
        return 2
    job = MiningJob(b"BENCHMARK_HEADER".ljust(80, b"\x00"), target=0)  # hardest: never "find"
    metrics = Metrics()
    _log(f"benchmark backend={args.backend} duration={args.duration}s")
    end = time.monotonic() + args.duration
    attempt = 0
    while time.monotonic() < end:
        r = backend.search(job, attempt)
        metrics.work_units += r.work_units
        metrics.attempts += 1
        attempt += 1
    secs = metrics.uptime()
    macs = metrics.throughput_macs_per_s()
    print(json.dumps({
        "backend": args.backend,
        "duration_sec": round(secs, 2),
        "attempts": metrics.attempts,
        "macs_per_sec": round(macs, 1),
        "tmac_per_sec": round(macs / 1e12, 5),
        "note": "harness throughput (MAC/s), NOT protocol TH/s — see STATUS.md",
    }, indent=2))
    return 0


def cmd_self_test(_args) -> int:
    _log("self-test: starting in-process mock gateway (EASIEST target so the CPU backend finds)")
    ok = True
    with MockGateway(target=EASIEST, rotate_after=4) as gw:
        backend = make_backend("cpu")
        metrics = Metrics()
        safety = SafetyMonitor(exit_after_invalid=10)
        events: list[tuple[str, dict]] = []
        with GatewayClient(transport="tcp", host=gw.host, port=gw.port) as client:
            cfg = LoopConfig(mode="self-test", duration_s=4.0, status_interval_s=1.0,
                             poll_interval_s=0.05, stale_cancel=True)
            mine_loop(client, backend, metrics, safety, cfg,
                      on_status=lambda s: _log(s),
                      on_event=lambda k, d: events.append((k, d)))

    submits = sum(1 for k, _ in events if k == "submit")
    new_jobs = sum(1 for k, _ in events if k == "new_job")
    _log(f"events: new_jobs={new_jobs} submits={submits} "
         f"accepted={metrics.accepted} stale={metrics.stale} "
         f"gateway_accepted={sum(gw.accepted_by_header.values())} gateway_dropped_stale={gw.dropped_stale}")

    checks = {
        "polled work": metrics.attempts > 0,
        "saw >=1 job": new_jobs >= 1,
        "detected job rotation (stale handling exercised)": metrics.job_switches >= 2,
        "submitted >=1 proof": submits >= 1,
        "gateway accepted >=1 submit": sum(gw.accepted_by_header.values()) >= 1,
    }
    for name, passed in checks.items():
        _log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    _log("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_run(args) -> int:
    cfg = Config.load(args.config)
    for w in cfg.warnings:
        _log(f"WARN: {w}")
    if cfg.apply_overclock:
        _log("apply_overclock=true — power/clock changes WOULD be applied (not implemented; "
             "use nvidia-smi/MSI Afterburner explicitly). Proceeding without changing clocks.")
    try:
        backend = make_backend(cfg.backend, device=cfg.devices[0] if cfg.devices else 0)
    except BackendNotBuilt as exc:
        _log(str(exc))
        return 2
    metrics = Metrics()
    safety = SafetyMonitor(max_temp_c=cfg.max_temp_c, max_vram_temp_c=cfg.max_vram_temp_c,
                           exit_after_invalid=cfg.exit_after_invalid)
    loop = LoopConfig(mode=cfg.mode, duration_s=args.duration, poll_interval_s=cfg.poll_interval_s,
                      status_interval_s=cfg.status_interval_s, stale_cancel=cfg.stale_cancel)
    _log(f"connecting to gateway via {cfg.transport} "
         f"({cfg.socket_path if cfg.transport == 'uds' else f'{cfg.host}:{cfg.port}'})")
    try:
        with GatewayClient(transport=cfg.transport, host=cfg.host, port=cfg.port,
                           socket_path=cfg.socket_path) as client:
            mine_loop(client, backend, metrics, safety, loop,
                      on_status=lambda s: _log(s), on_event=lambda k, d: _log(f"{k}: {d}"))
    except (ConnectionError, OSError) as exc:
        _log(f"could not reach gateway: {exc}. Is pearl-gateway running? See docs/run-simnet.md")
        return 2
    except KeyboardInterrupt:
        _log("shutting down (Ctrl-C)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prl3090-miner",
                                description="Pearl PRL miner for RTX 3090 / Ampere sm_86")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="mine against a configured gateway")
    pr.add_argument("--config", required=True)
    pr.add_argument("--duration", type=float, default=None, help="seconds (default: run forever)")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("benchmark", help="measure backend throughput")
    pb.add_argument("--duration", type=float, default=30.0)
    pb.add_argument("--backend", default="cpu", choices=["cpu", "cuda-naive", "cuda-mine", "cuda-sm86"])
    pb.add_argument("--device", type=int, default=0)
    pb.set_defaults(func=cmd_benchmark)

    pl = sub.add_parser("list-devices", help="list GPUs and backend availability")
    pl.set_defaults(func=cmd_list_devices)

    pv = sub.add_parser("validate-job", help="validate a job.json against the wire schema")
    pv.add_argument("--input", required=True)
    pv.set_defaults(func=cmd_validate_job)

    ps = sub.add_parser("self-test", help="run the mine->submit loop against an in-process gateway")
    ps.set_defaults(func=cmd_self_test)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
