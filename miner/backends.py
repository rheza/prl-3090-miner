"""Pluggable mining backends.

A backend turns a MiningJob into (optionally) a PlainProof by doing the NoisyGEMM
search. Two backends exist:

  * CpuBackend     — the NumPy reference (reference/pearl_reference.py). This is a
                     HARNESS, not a real miner: it has no vLLM model to source the
                     real A/B activations+weights and no commitment-hash chain
                     (needs py-pearl-mining), so it synthesizes A/B from the header
                     and uses a stand-in pow_key. It exists to validate orchestration,
                     metrics, safety, and the mine->submit loop end to end.

  * CudaSm86Backend— loads the compiled sm_86 extension (cuda/). Raises
                     BackendNotBuilt with a pointer to the port docs until the kernels
                     are built (see docs/cuda-sm86-port.md, scripts/build_sm86.sh).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from blake3 import blake3

# make reference/pearl_reference.py importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reference"))
from pearl_reference import NoiseGenerator, NoisyGemm  # noqa: E402

from .protocol import MiningJob  # noqa: E402


@dataclass
class BackendResult:
    found: bool
    plain_proof: bytes | None
    work_units: int          # MAC count attempted this call (for throughput accounting)
    detail: dict


class BackendNotBuilt(RuntimeError):
    pass


class CpuBackend:
    """Reference-correct harness backend (validates the loop, not production mining)."""

    name = "cpu"
    # Production-like shape (settings.py: tile 128x256, rank 128).
    M, K, N, R = 128, 256, 256, 128

    def __init__(self) -> None:
        self._gemm = NoisyGemm(noise_range=128, noise_rank=self.R,
                               hash_tile_h=16, hash_tile_w=16,
                               matmul_tile_h=128, matmul_tile_w=128)

    def device_info(self) -> dict:
        return {"name": "CPU (NumPy reference)", "backend": self.name,
                "temp_c": None, "vram_temp_c": None, "power_w": None}

    def search(self, job: MiningJob, attempt: int) -> BackendResult:
        hdr = job.incomplete_header_bytes
        seed = int.from_bytes(blake3(hdr + attempt.to_bytes(8, "little")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        A = rng.integers(-64, 64, size=(self.M, self.K), dtype=np.int8)
        B = rng.integers(-64, 64, size=(self.K, self.N), dtype=np.int8)
        key_A = blake3(hdr + b"A").digest()      # stand-in for commitment_A (real: §4.4)
        key_B = blake3(hdr + b"B").digest()
        E_AL, E_AR, E_BL, E_BR = NoiseGenerator(noise_rank=self.R).generate(
            key_A, key_B, self.M, self.K, self.N)
        _, found = self._gemm.run(A, B, E_AL, E_AR, E_BL, E_BR,
                                  pow_key=key_A, pow_target=job.target)
        work = self.M * self.K * self.N
        if not found:
            return BackendResult(False, None, work, {"attempt": attempt})
        ob = self._gemm.opened_block
        proof = json.dumps({
            "_harness": True,  # NOT a real PlainProof; py-pearl-mining builds the real one
            "A_row_indices": ob.A_row_indices,
            "B_column_indices": ob.B_column_indices,
            "transcript_words": [hex(w) for w in ob.transcript_words],
        }).encode()
        return BackendResult(True, proof, work, {"attempt": attempt,
                                                 "rows": ob.A_row_indices[:1],
                                                 "cols": ob.B_column_indices[:1]})


class CudaNaiveBackend:
    """Correct-but-slow GPU backend (plain integer cores) via cuda/build/libprl_naive.so.

    Validated bit-for-bit vs the reference on the GPU (reference/run_cuda_golden.py). This
    is the M4 correctness milestone — it makes the miner actually run on the GPU, but it is
    NOT the speed target (that needs the tensor-core port, docs/cuda-sm86-port.md). Like
    CpuBackend it synthesizes A/B from the header (no vLLM model here), so it is a harness.
    Linux/WSL only.
    """

    name = "cuda-naive"
    M, K, N, R = 128, 256, 256, 128

    def __init__(self, device: int = 0) -> None:
        from .cuda_naive import NaiveCuda  # local import: optional dependency
        self.device = device
        self._cuda = NaiveCuda()
        self._noise = NoiseGenerator(noise_rank=self.R)

    def device_info(self) -> dict:
        gpus = list_gpus()
        g = gpus[self.device] if self.device < len(gpus) else {}
        return {"name": g.get("name", "CUDA GPU"), "backend": self.name,
                "temp_c": g.get("temp_c"), "vram_temp_c": None, "power_w": g.get("power_w")}

    def search(self, job: MiningJob, attempt: int) -> BackendResult:
        hdr = job.incomplete_header_bytes
        seed = int.from_bytes(blake3(hdr + attempt.to_bytes(8, "little")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        A = rng.integers(-64, 64, size=(self.M, self.K), dtype=np.int8)
        B = rng.integers(-64, 64, size=(self.K, self.N), dtype=np.int8)
        key_A = blake3(hdr + b"A").digest()
        key_B = blake3(hdr + b"B").digest()
        E_AL, E_AR, E_BL, E_BR = self._noise.generate(key_A, key_B, self.M, self.K, self.N)
        _, found, a_row, b_col, tr = self._cuda.run(A, B, E_AL, E_AR, E_BL, E_BR,
                                                    pow_key=key_A, pow_target=job.target)
        work = self.M * self.K * self.N
        if not found:
            return BackendResult(False, None, work, {"attempt": attempt})
        proof = json.dumps({"_harness": True, "a_row": a_row, "b_col": b_col,
                            "transcript_words": [hex(int(w)) for w in tr]}).encode()
        return BackendResult(True, proof, work, {"attempt": attempt, "a_row": a_row, "b_col": b_col})


class CudaMineBackend:
    """The FUSED tensor-core miner kernel (cuda/build/libprl_miner.so) — the real hot loop.

    Validated bit-for-bit vs golden on the GPU (cuda/tests/validate_mine.py). Like the
    other backends it synthesizes A/B from the header (no vLLM model here), so it is a
    harness for the kernel; production sources A/B from the model. A/B are preloaded once
    per job and keyed noise varies per attempt, matching the intended device-resident
    tensor path instead of recopying matrices on every attempt. Linux/WSL only;
    specialized for noise_rank=128 (shape 128x256x256).
    """

    name = "cuda-mine"
    M, K, N, R = 128, 256, 256, 128

    def __init__(self, device: int = 0) -> None:
        from .cuda_mine import MineCuda
        self.device = device
        self._cuda = MineCuda()
        self._preloaded_header_id: bytes | None = None
        self._batch_size = max(1, int(os.environ.get("PRL_CUDA_BATCH", "256")))

    def device_info(self) -> dict:
        gpus = list_gpus()
        g = gpus[self.device] if self.device < len(gpus) else {}
        return {"name": g.get("name", "CUDA GPU"), "backend": self.name,
                "temp_c": g.get("temp_c"), "vram_temp_c": None, "power_w": g.get("power_w")}

    def search(self, job: MiningJob, attempt: int) -> BackendResult:
        hdr = job.incomplete_header_bytes
        if self._preloaded_header_id != job.header_id:
            seed = int.from_bytes(blake3(hdr + b"AB").digest()[:8], "little")
            rng = np.random.default_rng(seed)
            A = rng.integers(-64, 64, size=(self.M, self.K), dtype=np.int8)
            B = rng.integers(-64, 64, size=(self.K, self.N), dtype=np.int8)
            self._cuda.preload_inputs(A, B)
            self._preloaded_header_id = job.header_id
        batch = self._batch_size if job.target == 0 else 1
        keys_A = []
        keys_B = []
        for i in range(batch):
            nonce = (attempt + i).to_bytes(8, "little")
            keys_A.append(blake3(hdr + nonce + b"A").digest())
            keys_B.append(blake3(hdr + nonce + b"B").digest())
        # noise is generated ON-GPU from the keys (no Python BLAKE3 loop — the old 68% bottleneck)
        if batch == 1:
            found, a_row, b_col, tr = self._cuda.run_keyed_preloaded((self.M, self.K, self.N),
                                                                     keys_A[0], keys_B[0], job.target)
            found_attempt = 0
        else:
            found_attempt, found, a_row, b_col, tr = self._cuda.run_keyed_preloaded_batch(
                (self.M, self.K, self.N), keys_A, keys_B, job.target)
        work = self.M * self.K * self.N
        if not found:
            return BackendResult(False, None, work * batch, {"attempt": attempt, "attempts": batch})
        proof = json.dumps({"_harness": True, "a_row": a_row, "b_col": b_col,
                            "transcript_words": [hex(int(w)) for w in tr]}).encode()
        return BackendResult(True, proof, work * (found_attempt + 1),
                             {"attempt": attempt + found_attempt, "attempts": found_attempt + 1,
                              "a_row": a_row, "b_col": b_col})


class CudaSm86Backend:
    """Loads the compiled (tensor-core) sm_86 kernels. Until built, fails with guidance."""

    name = "cuda-sm86"

    def __init__(self, device: int = 0) -> None:
        self.device = device
        try:
            import prl_cuda  # noqa: F401  (built by cuda/ -> see scripts/build_sm86.sh)
        except Exception as exc:
            raise BackendNotBuilt(
                "cuda-sm86 backend is not built yet. The Ampere kernel port is the "
                "remaining work — see docs/cuda-sm86-port.md and run scripts/build_sm86.sh "
                f"in WSL/Ubuntu with the CUDA Toolkit installed. (import error: {exc})"
            ) from exc
        self._mod = prl_cuda

    def device_info(self) -> dict:  # pragma: no cover - needs GPU + build
        return self._mod.device_info(self.device)

    def search(self, job: MiningJob, attempt: int) -> BackendResult:  # pragma: no cover
        raise NotImplementedError("wire up to prl_cuda.search once kernels pass golden vectors")


def make_backend(name: str, device: int = 0):
    if name == "cpu":
        return CpuBackend()
    if name == "cuda-naive":
        return CudaNaiveBackend(device=device)
    if name == "cuda-mine":
        return CudaMineBackend(device=device)
    if name == "cuda-sm86":
        return CudaSm86Backend(device=device)
    raise ValueError(f"unknown backend '{name}' (choices: cpu, cuda-naive, cuda-mine, cuda-sm86)")


def list_gpus() -> list[dict]:
    """Enumerate NVIDIA GPUs via nvidia-smi (NVML-backed). Empty list if none."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=index,name,memory.total,driver_version,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": parts[0], "name": parts[1], "memory_total_mib": parts[2],
                         "driver": parts[3],
                         "temp_c": parts[4] if len(parts) > 4 else None,
                         "power_w": parts[5] if len(parts) > 5 else None})
    return gpus
