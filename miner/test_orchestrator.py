"""Tests for the orchestration layer (no GPU, no network beyond localhost)."""

from __future__ import annotations

import base64

import pytest

from miner.backends import BackendNotBuilt, CpuBackend, make_backend
from miner.gateway_client import GatewayClient
from miner.mock_gateway import MockGateway
from miner.protocol import MAX_TARGET, MiningJob, PlainProofSubmission
from miner.runtime import LoopConfig, Metrics, SafetyMonitor, mine_loop

EASIEST = MAX_TARGET


def test_mining_job_roundtrip():
    job = MiningJob(b"\xde\xad\xba\xbe" * 8, target=12345)
    d = job.to_dict()
    assert base64.b64decode(d["incomplete_header_bytes"]) == job.incomplete_header_bytes
    assert MiningJob.from_dict(d) == job


def test_mining_job_rejects_bad_target():
    with pytest.raises(ValueError):
        MiningJob.from_dict({"incomplete_header_bytes": base64.b64encode(b"x").decode(),
                             "target": MAX_TARGET + 1})


def test_cpu_backend_finds_under_easiest_and_misses_under_hardest():
    be = CpuBackend()
    hdr = b"H".ljust(80, b"\x00")
    assert be.search(MiningJob(hdr, EASIEST), 0).found is True
    assert be.search(MiningJob(hdr, 0), 0).found is False


def test_safety_thermal_and_invalid_breaker():
    s = SafetyMonitor(max_temp_c=78, max_vram_temp_c=96, exit_after_invalid=3)
    assert s.check_thermal({"temp_c": 70}) is None
    assert s.check_thermal({"temp_c": 80}) is not None
    assert s.check_thermal({"vram_temp_c": 97}) is not None
    assert not s.note_invalid_proof()
    assert not s.note_invalid_proof()
    assert s.note_invalid_proof()  # 3rd → trip


def test_cuda_backend_reports_not_built():
    with pytest.raises(BackendNotBuilt):
        make_backend("cuda-sm86")


def test_full_mine_loop_submits_and_handles_staleness():
    with MockGateway(target=EASIEST, rotate_after=4) as gw:
        metrics = Metrics()
        events = []
        with GatewayClient(transport="tcp", host=gw.host, port=gw.port) as client:
            mine_loop(client, CpuBackend(), metrics, SafetyMonitor(),
                      LoopConfig(duration_s=3.0, status_interval_s=999,
                                 poll_interval_s=0.02, stale_cancel=True),
                      on_event=lambda k, d: events.append((k, d)))
        assert metrics.attempts > 0
        assert metrics.job_switches >= 2          # header rotated -> stale handling exercised
        assert sum(1 for k, _ in events if k == "submit") >= 1
        assert sum(gw.accepted_by_header.values()) >= 1


def test_throttle_blocks_search_when_overtemp():
    """A backend reporting an over-limit temp must cause the loop to skip GPU work."""
    class HotBackend(CpuBackend):
        def device_info(self):
            return {"name": "hot", "temp_c": 95, "vram_temp_c": None, "power_w": None}

    with MockGateway(target=EASIEST, rotate_after=0) as gw:
        metrics = Metrics()
        with GatewayClient(transport="tcp", host=gw.host, port=gw.port) as client:
            mine_loop(client, HotBackend(), metrics, SafetyMonitor(max_temp_c=78),
                      LoopConfig(duration_s=1.0, status_interval_s=999, poll_interval_s=0.02),
                      on_event=lambda k, d: None)
        assert metrics.attempts == 0   # never searched while throttled
