"""Validate the Ampere mma.sync int8 tensor-core GEMM (libprl_mma.so) against the golden
C (== A@B) on the GPU, and measure kernel-only throughput.

    PYTHONPATH=<repo> python cuda/tests/validate_mma.py
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SO = ROOT / "cuda" / "build" / "libprl_mma.so"
GOLDEN = ROOT / "tests" / "golden"

i8p = ctypes.POINTER(ctypes.c_int8)
i32p = ctypes.POINTER(ctypes.c_int32)


def main() -> int:
    if not SO.exists():
        print(f"{SO} not found — build it first"); return 2
    lib = ctypes.CDLL(str(SO))
    lib.prl_mma_last_error.restype = ctypes.c_char_p
    lib.prl_mma_device_count.restype = ctypes.c_int
    lib.prl_mma_gemm.argtypes = [i8p, i8p, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.prl_mma_gemm.restype = ctypes.c_int
    lib.prl_mma_gemm_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double)]
    lib.prl_mma_gemm_bench.restype = ctypes.c_int
    print(f"CUDA devices: {lib.prl_mma_device_count()}")

    ok = True
    for case in json.loads((GOLDEN / "manifest.json").read_text())["cases"]:
        d = np.load(GOLDEN / case["file"])
        A = np.ascontiguousarray(d["A"], dtype=np.int8)
        B = np.ascontiguousarray(d["B"], dtype=np.int8)
        m, k = A.shape; n = B.shape[1]
        C = np.zeros((m, n), dtype=np.int32)
        rc = lib.prl_mma_gemm(A.ctypes.data_as(i8p), B.ctypes.data_as(i8p),
                              C.ctypes.data_as(i32p), m, k, n)
        if rc != 0:
            print(f"  {case['name']}: rc={rc} {lib.prl_mma_last_error().decode()}"); ok = False; continue
        match = np.array_equal(C, d["C"])
        ok = ok and match
        print(f"  [{'PASS' if match else 'FAIL'}] {case['name']:26s} mma C == golden C (=A@B)")

    print("THROUGHPUT (kernel-only, this UNOPTIMIZED one-warp-per-tile kernel):")
    for (m, k, n) in [(128, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        ms = ctypes.c_double(0)
        rc = lib.prl_mma_gemm_bench(m, k, n, 50, ctypes.byref(ms))
        if rc != 0:
            print(f"  bench {m}x{k}x{n}: {lib.prl_mma_last_error().decode()}"); continue
        macs = m * k * n
        gmacs = macs / (ms.value / 1e3) / 1e9
        print(f"  {m:5d}x{k:5d}x{n:5d}: {ms.value:.3f} ms  {gmacs:8.1f} GMAC/s  "
              f"({2*gmacs/1e3:.2f} TOPS)")
    print("  NOTE: no shared-mem tiling / cp.async yet (re-reads global per tile). The")
    print("        smem-tiled + cp.async pipeline is the next step (docs/cuda-sm86-port.md §3,§6).")
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
