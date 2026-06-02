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

    # ---- batched fast path: cross-check prl_mine2_batch_run vs the verified single path ----
    sys.path.insert(0, str(ROOT / "reference"))
    import pearl_reference as ref  # noqa: E402
    lib.prl_mine2_batch_run.argtypes = [i8p]*6 + [ctypes.c_int]*4 + [u8p, u8p, ip, ip, ip, u32p]
    lib.prl_mine2_batch_run.restype = ctypes.c_int
    pf = next(c for c in json.loads((GOLDEN / "manifest.json").read_text())["cases"] if c["name"] == "p_found")
    d = np.load(GOLDEN / pf["file"]); m, k, n, R = pf["m"], pf["k"], pf["n"], pf["noise_rank"]
    Aa, pA = p8(d["A"]); Bb, pB = p8(d["B"])
    tgt = np.frombuffer(int(pf["pow_target"], 16).to_bytes(32, "little"), np.uint8).copy()
    natt = 8
    kA = [f"batchKeyA_{j:02d}".encode().ljust(32, b"\0")[:32] for j in range(natt)]
    kB = [f"batchKeyB_{j:02d}".encode().ljust(32, b"\0")[:32] for j in range(natt)]
    nz = [ref.NoiseGenerator(noise_rank=R).generate(kA[j], kB[j], m, k, n) for j in range(natt)]

    exp_att, exp_loc, exp_tr, hold = -1, None, None, []          # first winner per the single path
    for j in range(natt):
        ps = []
        for arr in nz[j]:
            a, p = p8(arr); hold.append(a); ps.append(p)
        kj = np.frombuffer(kA[j], np.uint8).copy()
        f0 = ctypes.c_int(0); r0 = ctypes.c_int(0); c0 = ctypes.c_int(0); t0 = np.zeros(16, np.uint32)
        lib.prl_mine2_run(pA, pB, ps[0], ps[1], ps[2], ps[3], m, k, n,
                          kj.ctypes.data_as(u8p), tgt.ctypes.data_as(u8p),
                          ctypes.byref(f0), ctypes.byref(r0), ctypes.byref(c0), t0.ctypes.data_as(u32p))
        if f0.value and exp_att < 0:
            exp_att, exp_loc, exp_tr = j, (r0.value, c0.value), [int(w) for w in t0]

    EAL = np.ascontiguousarray(np.stack([nz[j][0] for j in range(natt)]), np.int8)
    EAR = np.ascontiguousarray(np.stack([nz[j][1] for j in range(natt)]), np.int8)
    EBL = np.ascontiguousarray(np.stack([nz[j][2] for j in range(natt)]), np.int8)
    EBR = np.ascontiguousarray(np.stack([nz[j][3] for j in range(natt)]), np.int8)
    keys = np.frombuffer(b"".join(kA), np.uint8).copy()
    fa = ctypes.c_int(0); ar = ctypes.c_int(0); bc = ctypes.c_int(0); tr = np.zeros(16, np.uint32)
    rc = lib.prl_mine2_batch_run(pA, pB, EAL.ctypes.data_as(i8p), EAR.ctypes.data_as(i8p),
                                 EBL.ctypes.data_as(i8p), EBR.ctypes.data_as(i8p), m, k, n, natt,
                                 keys.ctypes.data_as(u8p), tgt.ctypes.data_as(u8p),
                                 ctypes.byref(fa), ctypes.byref(ar), ctypes.byref(bc), tr.ctypes.data_as(u32p))
    loc_ok = exp_att < 0 or ((ar.value, bc.value) == exp_loc and [int(w) for w in tr] == exp_tr)
    bg = (rc == 0 and fa.value == exp_att and loc_ok)
    ok = ok and bg
    print(f"  [{'PASS' if bg else 'FAIL'}] batch_vs_single   natt={natt} winner_attempt={fa.value} "
          f"(single={exp_att}) loc/transcript={'=' if loc_ok else 'X'}")

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
    for natt in [256, 512, 1024, 2048, 4096]:
        ms = ctypes.c_double(0)
        if lib.prl_mine2_bench_batched(m, k, n, natt, 100, ctypes.byref(ms)) == 0:
            aps = natt / (ms.value / 1e3)
            tops = 2.0 * natt * m * k * n / (ms.value / 1e3) / 1e12
            print(f"  natt={natt:5d}: {ms.value:.4f} ms/iter  {aps/1e6:7.3f} M attempts/s  ({tops:6.2f} TOPS)")
        else:
            print(f"  natt={natt}: {lib.prl_mine_last_error().decode()}")
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
