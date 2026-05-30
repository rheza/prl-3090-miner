"""Measure the NAIVE sm_86 backend's GPU throughput — to quantify, honestly, how far
the correctness-first naive kernel is from the tensor-core target (~100 TH/s class).

This number is expected to be SMALL: plain integer cores + per-call malloc/copy overhead,
no tensor cores. Its only purpose is to motivate the tensor-core port (docs/cuda-sm86-port.md).

    PYTHONPATH=<repo> python cuda/tests/bench_naive.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from miner.cuda_naive import NaiveCuda  # noqa: E402

try:
    from blake3 import blake3
except Exception:
    blake3 = None


def main() -> int:
    cuda = NaiveCuda()
    m, k, n, r = 128, 256, 256, 128         # production-like shape (settings.py)
    rng = np.random.default_rng(0)
    A = rng.integers(-64, 64, size=(m, k), dtype=np.int8)
    B = rng.integers(-64, 64, size=(k, n), dtype=np.int8)
    # structurally-valid noise via the reference generator
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "reference"))
    from pearl_reference import NoiseGenerator
    E_AL, E_AR, E_BL, E_BR = NoiseGenerator(noise_rank=r).generate(b"K" * 32, b"L" * 32, m, k, n)
    key = b"K" * 32
    target = 0   # hardest: full pipeline runs, never short-circuits

    iters = 200
    cuda.run(A, B, E_AL, E_AR, E_BL, E_BR, key, target)  # warm up
    t0 = time.perf_counter()
    for _ in range(iters):
        cuda.run(A, B, E_AL, E_AR, E_BL, E_BR, key, target)
    dt = time.perf_counter() - t0

    per_call = dt / iters
    macs = m * k * n
    mac_s = macs / per_call
    print(f"naive sm_86 backend (plain int cores + per-call alloc):")
    print(f"  shape m={m} k={k} n={n} r={r}, {iters} iters, {per_call*1e3:.3f} ms/call")
    print(f"  ~{mac_s/1e9:.2f} GMAC/s  (=={mac_s/1e12:.5f} TMAC/s)")
    print(f"  NOTE: includes malloc/copy overhead and uses NO tensor cores. The ~100 TH/s")
    print(f"        class target needs the mma.sync port — see docs/cuda-sm86-port.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
