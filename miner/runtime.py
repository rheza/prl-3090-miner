"""Mining runtime: job management, submission, metrics, safety, and the mine loop.

Conceptually maps to PRD §20 modules job_manager / submitter / metrics / safety,
and implements the telemetry of PRD §16 and the safety controls of PRD §17.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backends import BackendResult
from .gateway_client import GatewayClient, MiningPaused
from .protocol import MiningJob, PlainProofSubmission


# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    started: float = field(default_factory=time.monotonic)
    work_units: int = 0          # MACs attempted
    attempts: int = 0
    submitted: int = 0
    accepted: int = 0            # gateway acks "submitted"; true accept is async (§3.3)
    rejected: int = 0
    stale: int = 0               # in-flight results dropped because header changed
    reconnects: int = 0
    job_switches: int = 0
    last_job_time: float = field(default_factory=time.monotonic)
    last_kernel_ms: float = 0.0

    def uptime(self) -> float:
        return time.monotonic() - self.started

    def job_age(self) -> float:
        return time.monotonic() - self.last_job_time

    def throughput_macs_per_s(self) -> float:
        up = self.uptime()
        return self.work_units / up if up > 0 else 0.0

    def status_line(self, dev: dict, mode: str) -> str:
        up = int(self.uptime())
        tmacs = self.throughput_macs_per_s() / 1e12
        power = dev.get("power_w")
        fields = [
            f"GPU: {dev.get('name', '?')}",
            f"Mode: {mode}",
            # NOTE: until the CUDA backend lands this is a harness throughput (MAC/s),
            # NOT the protocol TH/s. STATUS.md is explicit about this.
            f"Throughput: {tmacs:.4f} TMAC/s (harness)",
            f"Power: {power if power not in (None, '') else '-'} W",
        ]
        try:
            if power not in (None, "", "0"):
                fields.append(f"Eff: {tmacs * 1e3 / float(power):.2f} GMAC/W")
        except (TypeError, ValueError):
            pass
        fields += [
            f"GPU temp: {dev.get('temp_c') or '-'} C",
            f"VRAM temp: {dev.get('vram_temp_c') or '-'} C",
            f"Accepted: {self.accepted}",
            f"Rejected: {self.rejected}",
            f"Stale: {self.stale}",
            f"Uptime: {up // 3600:02d}:{(up % 3600) // 60:02d}:{up % 60:02d}",
            f"Job age: {self.job_age():.0f}s",
        ]
        return " | ".join(fields)


# --------------------------------------------------------------------------- #
class SafetyMonitor:
    """Temperature throttle + invalid-proof circuit breaker (PRD §17)."""

    def __init__(self, max_temp_c: float = 78, max_vram_temp_c: float = 96,
                 exit_after_invalid: int = 10):
        self.max_temp_c = max_temp_c
        self.max_vram_temp_c = max_vram_temp_c
        self.exit_after_invalid = exit_after_invalid
        self.invalid_proofs = 0

    def check_thermal(self, dev: dict) -> str | None:
        """Return a reason string if we should throttle, else None."""
        for key, limit in (("temp_c", self.max_temp_c), ("vram_temp_c", self.max_vram_temp_c)):
            v = dev.get(key)
            try:
                if v is not None and float(v) >= limit:
                    return f"{key}={v} >= limit {limit}"
            except (TypeError, ValueError):
                pass
        return None

    def note_invalid_proof(self) -> bool:
        """Returns True if the invalid-proof limit is exceeded (caller should exit)."""
        self.invalid_proofs += 1
        return self.invalid_proofs >= self.exit_after_invalid


# --------------------------------------------------------------------------- #
class JobManager:
    """Tracks the current job and detects new-tip header changes (§3.4)."""

    def __init__(self, client: GatewayClient):
        self.client = client
        self.current: MiningJob | None = None

    def poll(self) -> tuple[MiningJob | None, bool]:
        """Returns (job, changed). job is None while mining is paused."""
        try:
            job = self.client.get_mining_info()
        except MiningPaused:
            return None, False
        changed = self.current is None or job.header_id != self.current.header_id
        self.current = job
        return job, changed


# --------------------------------------------------------------------------- #
@dataclass
class LoopConfig:
    mode: str = "simnet"
    duration_s: float | None = None
    status_interval_s: float = 30.0
    poll_interval_s: float = 1.0
    stale_cancel: bool = True


def mine_loop(client: GatewayClient, backend, metrics: Metrics, safety: SafetyMonitor,
              cfg: LoopConfig, on_status=None, on_event=None) -> Metrics:
    """The core mine->submit loop. Pure stdlib + backend; safe to run under self-test."""
    jm = JobManager(client)
    attempt = 0
    last_status = 0.0
    end = (time.monotonic() + cfg.duration_s) if cfg.duration_s else None

    while end is None or time.monotonic() < end:
        job, changed = jm.poll()
        if job is None:
            time.sleep(cfg.poll_interval_s)
            continue
        if changed:
            metrics.job_switches += 1
            metrics.last_job_time = time.monotonic()
            attempt = 0
            if on_event:
                on_event("new_job", {"target_hex": hex(job.target)})

        # thermal safety (PRD §17): throttle instead of pushing the GPU past limits
        dev = backend.device_info()
        reason = safety.check_thermal(dev)
        if reason:
            if on_event:
                on_event("throttle", {"reason": reason})
            time.sleep(cfg.poll_interval_s)
            continue

        t0 = time.monotonic()
        result: BackendResult = backend.search(job, attempt)
        metrics.last_kernel_ms = (time.monotonic() - t0) * 1e3
        metrics.attempts += 1
        metrics.work_units += result.work_units
        attempt += 1

        if result.found and result.plain_proof is not None:
            # stale cancellation: re-confirm the header is still current before submitting (§3.4).
            # Use a direct client call (not jm.poll) so we don't disturb the job-switch baseline.
            if cfg.stale_cancel:
                try:
                    latest = client.get_mining_info()
                except MiningPaused:
                    latest = None
                if latest is None or latest.header_id != job.header_id:
                    metrics.stale += 1
                    if on_event:
                        on_event("stale_drop", {})
                    continue
            metrics.submitted += 1
            try:
                ack = client.submit_plain_proof(PlainProofSubmission(result.plain_proof, job))
                if ack == "submitted":
                    metrics.accepted += 1   # gateway-level ack; chain accept is async (§3.3)
                else:
                    metrics.rejected += 1
                if on_event:
                    on_event("submit", {"ack": ack})
            except Exception as exc:  # network/submit failure
                metrics.rejected += 1
                if on_event:
                    on_event("submit_error", {"error": str(exc)})

        now = time.monotonic()
        if on_status and (now - last_status) >= cfg.status_interval_s:
            on_status(metrics.status_line(dev, cfg.mode))
            last_status = now

    return metrics
