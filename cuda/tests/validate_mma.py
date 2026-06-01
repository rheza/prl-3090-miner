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
    lib.prl_mma_gemm_smem.argtypes = [i8p, i8p, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.prl_mma_gemm_smem.restype = ctypes.c_int
    lib.prl_mma_gemm_smem_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_double)]
    lib.prl_mma_gemm_smem_bench.restype = ctypes.c_int
    lib.prl_mma_gemm_cpasync.argtypes = [i8p, i8p, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.prl_mma_gemm_cpasync.restype = ctypes.c_int
    lib.prl_mma_gemm_cpasync_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_double)]
    lib.prl_mma_gemm_cpasync_bench.restype = ctypes.c_int
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
        Cs = np.zeros((m, n), dtype=np.int32)
        rcs = lib.prl_mma_gemm_smem(A.ctypes.data_as(i8p), B.ctypes.data_as(i8p),
                                    Cs.ctypes.data_as(i32p), m, k, n)
        match_s = (rcs == 0 and np.array_equal(Cs, d["C"]))
        cp = "n/a"  # cp.async kernel needs m%128==n%128==0 (skips the 64x64 case)
        if m % 128 == 0 and n % 128 == 0:
            Cc = np.zeros((m, n), dtype=np.int32)
            rcc = lib.prl_mma_gemm_cpasync(A.ctypes.data_as(i8p), B.ctypes.data_as(i8p),
                                           Cc.ctypes.data_as(i32p), m, k, n)
            cp_pass = (rcc == 0 and np.array_equal(Cc, d["C"]))
            cp = "PASS" if cp_pass else "FAIL"
            ok = ok and cp_pass
        ok = ok and match and match_s
        print(f"  [{'PASS' if match else 'FAIL'}/{'PASS' if match_s else 'FAIL'}/{cp}] "
              f"{case['name']:26s} simple/smem/cpasync C == golden C (=A@B)")

    print("THROUGHPUT (kernel-only):  simple=one warp/tile  smem=64x64 tiled  cpasync=128x128 double-buffered")
    for (m, k, n) in [(128, 256, 256), (512, 512, 512), (1024, 1024, 1024), (4096, 4096, 4096)]:
        macs = m * k * n
        out = {}
        for label, fn in (("simple", lib.prl_mma_gemm_bench), ("smem", lib.prl_mma_gemm_smem_bench),
                          ("cpasync", lib.prl_mma_gemm_cpasync_bench)):
            ms = ctypes.c_double(0)
            if fn(m, k, n, 30, ctypes.byref(ms)) == 0:
                out[label] = macs / (ms.value / 1e3) / 1e9
        s = out.get("simple", 0); sm = out.get("smem", 0); cpa = out.get("cpasync", 0)
        print(f"  {m:5d}^3:  simple {2*s/1e3:6.2f}  smem {2*sm/1e3:6.2f}  "
              f"cpasync {2*cpa/1e3:7.2f} TOPS   (cpasync {cpa/sm if sm else 0:.1f}x vs smem)")
    print("  NOTE: GA102 int8 peak ~284 TOPS. Next: fuse the noised+transcript path onto this GEMM.")
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
