"""Validate the PROTOCOL-VALID 2x64 GPU miner kernel (prl_mine2_run) against the
protocol-accurate oracle (tests/golden_protocol/, real 2x64 hash tile) on the GPU.

    PYTHONPATH=<repo> python cuda/tests/validate_mine2.py
"""
from __future__ import annotations

import ctypes
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SO = ROOT / "cuda" / "build" / "libprl_miner.so"
GOLDEN = ROOT / "tests" / "golden_protocol"
i8p = ctypes.POINTER(ctypes.c_int8); u8p = ctypes.POINTER(ctypes.c_uint8)
u32p = ctypes.POINTER(ctypes.c_uint32); ip = ctypes.POINTER(ctypes.c_int)


def main() -> int:
    if not SO.exists():
        print(f"{SO} not found — build first"); return 2
    lib = ctypes.CDLL(str(SO))
    lib.prl_mine_last_error.restype = ctypes.c_char_p
    lib.prl_mine2_run.argtypes = [i8p, i8p, i8p, i8p, i8p, i8p, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, u8p, u8p, ip, ip, ip, u32p]
    lib.prl_mine2_run.restype = ctypes.c_int
    lib.prl_mine2_bench.argtypes = [ctypes.c_int]*4 + [ctypes.POINTER(ctypes.c_double)]
    lib.prl_mine2_bench.restype = ctypes.c_int
    lib.prl_mine2_bench_batched.argtypes = [ctypes.c_int]*5 + [ctypes.POINTER(ctypes.c_double)]
    lib.prl_mine2_bench_batched.restype = ctypes.c_int

    def p8(a):
        a = np.ascontiguousarray(a, dtype=np.int8); return a, a.ctypes.data_as(i8p)

    ok = True
    for case in json.loads((GOLDEN / "manifest.json").read_text())["cases"]:
        d = np.load(GOLDEN / case["file"])
        m, k, n = case["m"], case["k"], case["n"]
        A, pA = p8(d["A"]); B, pB = p8(d["B"])
        EAL, pEAL = p8(d["E_AL"]); EAR, pEAR = p8(d["E_AR"]); EBL, pEBL = p8(d["E_BL"]); EBR, pEBR = p8(d["E_BR"])
        key = np.frombuffer(bytes.fromhex(case["key_A"]), np.uint8).copy()
        tgt = np.frombuffer(int(case["pow_target"], 16).to_bytes(32, "little"), np.uint8).copy()
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0); tr = np.zeros(16, np.uint32)
        rc = lib.prl_mine2_run(pA, pB, pEAL, pEAR, pEBL, pEBR, m, k, n,
                               key.ctypes.data_as(u8p), tgt.ctypes.data_as(u8p),
                               ctypes.byref(found), ctypes.byref(a_row), ctypes.byref(b_col),
                               tr.ctypes.data_as(u32p))
        if rc != 0:
            print(f"  {case['name']}: rc={rc} {lib.prl_mine_last_error().decode()}"); ok = False; continue
        of = bool(found.value) == case["found_block"]
        ol = otr = True
        if case["found_block"]:
            ol = (a_row.value == case["A_row_indices"][0] and b_col.value == case["B_column_indices"][0])
            otr = [hex(int(w)) for w in tr] == case["transcript_words"]
        good = of and ol and otr; ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {case['name']:18s} found={'=' if of else 'X'} "
              f"loc={'=' if ol else 'X'} transcript={'=' if otr else 'X'}")

    print("THROUGHPUT (k_mine2, 2x64 protocol kernel, kernel-only):")
    for (m, k, n) in [(128, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048)]:
        ms = ctypes.c_double(0)
        if lib.prl_mine2_bench(m, k, n, 30, ctypes.byref(ms)) == 0:
            g = m*k*n/(ms.value/1e3)/1e9
            print(f"  {m}x{k}x{n}: {ms.value:.3f} ms  {g:.1f} GMAC/s  ({2*g/1e3:.2f} TOPS)")
        else:
            print(f"  {m}x{k}x{n}: {lib.prl_mine_last_error().decode()}")

    # The mining-relevant metric: a real attempt is 128x256x256 = grid (1,2) = 2 blocks (~2.4%
    # of the GPU), so throughput == batched attempts/sec. Sweep NATT to find where the GPU fills.
    m, k, n = 128, 256, 256
    print(f"BATCHED MINING THROUGHPUT (real per-attempt shape {m}x{k}x{n}, noised GEMM + 2x64 hash):")
    for natt in [1, 16, 64, 128, 256, 512, 1024]:
        ms = ctypes.c_double(0)
        if lib.prl_mine2_bench_batched(m, k, n, natt, 50, ctypes.byref(ms)) == 0:
            aps = natt / (ms.value / 1e3)
            tops = 2.0 * natt * m * k * n / (ms.value / 1e3) / 1e12
            print(f"  natt={natt:5d}: {ms.value:.4f} ms/iter  {aps/1e6:7.3f} M attempts/s  ({tops:6.2f} TOPS)")
        else:
            print(f"  natt={natt}: {lib.prl_mine_last_error().decode()}")
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
