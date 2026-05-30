"""Validate the naive sm_86 CUDA backend (libprl_naive.so) against the golden vectors
on a real GPU, plus a direct device-BLAKE3 unit test vs the Python blake3 oracle.

Run (in WSL after scripts/build_naive_wsl.sh):
    python reference/run_cuda_golden.py
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys

import numpy as np
from blake3 import blake3

ROOT = pathlib.Path(__file__).resolve().parents[1]
SO = ROOT / "cuda" / "build" / "libprl_naive.so"
GOLDEN = ROOT / "tests" / "golden"

i8p = ctypes.POINTER(ctypes.c_int8)
u8p = ctypes.POINTER(ctypes.c_uint8)
i32p = ctypes.POINTER(ctypes.c_int32)
u32p = ctypes.POINTER(ctypes.c_uint32)
ip = ctypes.POINTER(ctypes.c_int)


def _load():
    lib = ctypes.CDLL(str(SO))
    lib.prl_naive_last_error.restype = ctypes.c_char_p
    lib.prl_naive_device_count.restype = ctypes.c_int
    lib.prl_blake3_64.argtypes = [u8p, u8p, u8p]
    lib.prl_blake3_64.restype = ctypes.c_int
    lib.prl_naive_run.argtypes = [
        i8p, i8p, i8p, i8p, i8p, i8p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        u8p, u8p, i32p, ip, ip, ip, u32p,
    ]
    lib.prl_naive_run.restype = ctypes.c_int
    return lib


def _i8(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a, a.ctypes.data_as(i8p)


def test_blake3(lib) -> bool:
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(256):
        msg = rng.integers(0, 256, size=64, dtype=np.uint8)
        key = rng.integers(0, 256, size=32, dtype=np.uint8)
        out = np.zeros(32, dtype=np.uint8)
        rc = lib.prl_blake3_64(msg.ctypes.data_as(u8p), key.ctypes.data_as(u8p),
                               out.ctypes.data_as(u8p))
        if rc != 0:
            print("  blake3 rc", rc, lib.prl_naive_last_error().decode()); return False
        expect = blake3(msg.tobytes(), key=key.tobytes()).digest()
        if out.tobytes() != expect:
            ok = False
            print("  blake3 MISMATCH\n   dev", out.tobytes().hex(), "\n   exp", expect.hex())
            break
    print(f"  [{'PASS' if ok else 'FAIL'}] device BLAKE3 keyed-64 vs python blake3 (256 vectors)")
    return ok


def test_golden(lib) -> bool:
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    all_ok = True
    for case in manifest["cases"]:
        d = np.load(GOLDEN / case["file"])
        m, k, n, r = case["m"], case["k"], case["n"], case["noise_rank"]
        th, tw = case["hash_tile_h"], case["hash_tile_w"]
        A, pA = _i8(d["A"]); B, pB = _i8(d["B"])
        EAL, pEAL = _i8(d["E_AL"]); EAR, pEAR = _i8(d["E_AR"])
        EBL, pEBL = _i8(d["E_BL"]); EBR, pEBR = _i8(d["E_BR"])
        key = np.frombuffer(bytes.fromhex(case["key_A"]), dtype=np.uint8).copy()
        tgt = np.frombuffer(int(case["pow_target"], 16).to_bytes(32, "little"), dtype=np.uint8).copy()
        C = np.zeros((m, n), dtype=np.int32)
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = lib.prl_naive_run(
            pA, pB, pEAL, pEAR, pEBL, pEBR, m, k, n, r, th, tw,
            key.ctypes.data_as(u8p), tgt.ctypes.data_as(u8p),
            C.ctypes.data_as(i32p), ctypes.byref(found), ctypes.byref(a_row),
            ctypes.byref(b_col), tr.ctypes.data_as(u32p))
        if rc != 0:
            print(f"  {case['name']}: rc={rc} {lib.prl_naive_last_error().decode()}"); all_ok = False; continue

        ok_C = np.array_equal(C, d["C"])
        ok_found = bool(found.value) == case["found_block"]
        ok_loc = ok_tr = True
        if case["found_block"]:
            ok_loc = (a_row.value == case["A_row_indices"][0] and b_col.value == case["B_column_indices"][0])
            ok_tr = [hex(int(w)) for w in tr] == case["transcript_words"]
        ok = ok_C and ok_found and ok_loc and ok_tr
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['name']:26s} "
              f"C={'=' if ok_C else 'X'} found={'=' if ok_found else 'X'} "
              f"loc={'=' if ok_loc else 'X'} transcript={'=' if ok_tr else 'X'}")
    return all_ok


def main() -> int:
    if not SO.exists():
        print(f"{SO} not found — run scripts/build_naive_wsl.sh first"); return 2
    lib = _load()
    print(f"CUDA devices: {lib.prl_naive_device_count()}")
    b = test_blake3(lib)
    g = test_golden(lib)
    print("RESULT:", "ALL PASS" if (b and g) else "FAILURES")
    return 0 if (b and g) else 1


if __name__ == "__main__":
    sys.exit(main())
