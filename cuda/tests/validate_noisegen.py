"""Verify GPU noise generation (prl_noisegen) is bit-exact vs the reference NoiseGenerator.
This is the fix for the 68%-of-attempt Python-BLAKE3 noise bottleneck.

    PYTHONPATH=<repo> python cuda/tests/validate_noisegen.py
"""

from __future__ import annotations

import ctypes
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SO = ROOT / "cuda" / "build" / "libprl_miner.so"
sys.path.insert(0, str(ROOT / "reference"))
from pearl_reference import NoiseGenerator  # noqa: E402

i8p = ctypes.POINTER(ctypes.c_int8)
u8p = ctypes.POINTER(ctypes.c_uint8)
KEY_A = b"test_key_A_123456789012345678901"
KEY_B = b"test_key_B_123456789012345678901"


def main() -> int:
    lib = ctypes.CDLL(str(SO))
    lib.prl_noisegen.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 i8p, i8p, i8p, i8p]
    lib.prl_noisegen.restype = ctypes.c_int
    ok = True
    for (m, k, n) in [(128, 256, 256), (256, 256, 512)]:
        r = 128
        EAL = np.zeros((m, r), np.int8); EAR = np.zeros((r, k), np.int8)
        EBL = np.zeros((k, r), np.int8); EBR = np.zeros((r, n), np.int8)
        ka = np.frombuffer(KEY_A, np.uint8).copy(); kb = np.frombuffer(KEY_B, np.uint8).copy()
        rc = lib.prl_noisegen(ka.ctypes.data_as(u8p), kb.ctypes.data_as(u8p), m, k, n,
                              EAL.ctypes.data_as(i8p), EAR.ctypes.data_as(i8p),
                              EBL.ctypes.data_as(i8p), EBR.ctypes.data_as(i8p))
        if rc != 0:
            print(f"  {m}x{k}x{n}: rc={rc}"); ok = False; continue
        rAL, rAR, rBL, rBR = NoiseGenerator(noise_rank=r).generate(KEY_A, KEY_B, m, k, n)
        checks = {"E_AL": np.array_equal(EAL, rAL), "E_AR": np.array_equal(EAR, rAR),
                  "E_BL": np.array_equal(EBL, rBL), "E_BR": np.array_equal(EBR, rBR)}
        good = all(checks.values())
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {m}x{k}x{n}: "
              + " ".join(f"{nm}{'=' if v else 'X'}" for nm, v in checks.items()))
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
