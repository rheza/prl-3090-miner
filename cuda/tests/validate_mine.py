"""Validate the FUSED tensor-core mining kernel (libprl_miner.so) against the FULL
golden vectors (found / winning indices / 16 transcript words) for the noise_rank=128
production cases, on the GPU. Also measures the fused GEMM+transcript throughput.

    PYTHONPATH=<repo> python cuda/tests/validate_mine.py
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SO = ROOT / "cuda" / "build" / "libprl_miner.so"
GOLDEN = ROOT / "tests" / "golden"

i8p = ctypes.POINTER(ctypes.c_int8)
u8p = ctypes.POINTER(ctypes.c_uint8)
u32p = ctypes.POINTER(ctypes.c_uint32)
ip = ctypes.POINTER(ctypes.c_int)


def main() -> int:
    if not SO.exists():
        print(f"{SO} not found — build it first"); return 2
    lib = ctypes.CDLL(str(SO))
    lib.prl_mine_last_error.restype = ctypes.c_char_p
    lib.prl_mine_device_count.restype = ctypes.c_int
    lib.prl_mine_run.argtypes = [i8p, i8p, i8p, i8p, i8p, i8p,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 u8p, u8p, ip, ip, ip, u32p]
    lib.prl_mine_run.restype = ctypes.c_int
    lib.prl_mine_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_double)]
    lib.prl_mine_bench.restype = ctypes.c_int
    print(f"CUDA devices: {lib.prl_mine_device_count()}")

    def p8(a):
        a = np.ascontiguousarray(a, dtype=np.int8)
        return a, a.ctypes.data_as(i8p)

    ok = True
    tested = 0
    for case in json.loads((GOLDEN / "manifest.json").read_text())["cases"]:
        if case["noise_rank"] != 128:
            continue  # fused kernel is specialized for the production r=128 config
        tested += 1
        d = np.load(GOLDEN / case["file"])
        m, k, n = case["m"], case["k"], case["n"]
        A, pA = p8(d["A"]); B, pB = p8(d["B"])
        EAL, pEAL = p8(d["E_AL"]); EAR, pEAR = p8(d["E_AR"])
        EBL, pEBL = p8(d["E_BL"]); EBR, pEBR = p8(d["E_BR"])
        key = np.frombuffer(bytes.fromhex(case["key_A"]), dtype=np.uint8).copy()
        tgt = np.frombuffer(int(case["pow_target"], 16).to_bytes(32, "little"), dtype=np.uint8).copy()
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = lib.prl_mine_run(pA, pB, pEAL, pEAR, pEBL, pEBR, m, k, n,
                              key.ctypes.data_as(u8p), tgt.ctypes.data_as(u8p),
                              ctypes.byref(found), ctypes.byref(a_row), ctypes.byref(b_col),
                              tr.ctypes.data_as(u32p))
        if rc != 0:
            print(f"  {case['name']}: rc={rc} {lib.prl_mine_last_error().decode()}"); ok = False; continue
        ok_found = bool(found.value) == case["found_block"]
        ok_loc = ok_tr = True
        if case["found_block"]:
            ok_loc = (a_row.value == case["A_row_indices"][0] and b_col.value == case["B_column_indices"][0])
            ok_tr = [hex(int(w)) for w in tr] == case["transcript_words"]
        good = ok_found and ok_loc and ok_tr
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {case['name']:26s} "
              f"found={'=' if ok_found else 'X'} loc={'=' if ok_loc else 'X'} transcript={'=' if ok_tr else 'X'}")

    print("FUSED MINER throughput (k_mine: noised GEMM + per-chunk transcript, kernel-only):")
    for (m, k, n) in [(128, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048)]:
        ms = ctypes.c_double(0)
        if lib.prl_mine_bench(m, k, n, 30, ctypes.byref(ms)) == 0:
            gmacs = m * k * n / (ms.value / 1e3) / 1e9
            print(f"  {m:5d}x{k:5d}x{n:5d}: {ms.value:.3f} ms  {gmacs:8.1f} GMAC/s  ({2*gmacs/1e3:.2f} TOPS)")
    print("RESULT:", "ALL PASS" if (ok and tested) else ("NO r=128 CASES" if not tested else "FAILURES"))
    return 0 if (ok and tested) else 1


if __name__ == "__main__":
    sys.exit(main())
