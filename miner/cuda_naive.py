"""ctypes wrapper for the naive sm_86 CUDA backend (cuda/build/libprl_naive.so).

The naive backend is CORRECT (validated bit-for-bit vs the reference on the GPU, see
reference/run_cuda_golden.py) but SLOW (plain integer cores). It's the M4 correctness
gate, not the speed target. Linux/WSL only — the .so is built by scripts/build_naive_wsl.sh.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_SO = pathlib.Path(__file__).resolve().parents[1] / "cuda" / "build" / "libprl_naive.so"

_i8 = ctypes.POINTER(ctypes.c_int8)
_u8 = ctypes.POINTER(ctypes.c_uint8)
_i32 = ctypes.POINTER(ctypes.c_int32)
_u32 = ctypes.POINTER(ctypes.c_uint32)
_ip = ctypes.POINTER(ctypes.c_int)


class NaiveCudaUnavailable(RuntimeError):
    pass


class NaiveCuda:
    def __init__(self) -> None:
        if not _SO.exists():
            raise NaiveCudaUnavailable(
                f"{_SO} not found — build it in WSL/Linux with scripts/build_naive_wsl.sh")
        try:
            lib = ctypes.CDLL(str(_SO))
        except OSError as exc:  # e.g. loading a Linux .so on Windows
            raise NaiveCudaUnavailable(f"cannot load {_SO}: {exc}") from exc
        lib.prl_naive_last_error.restype = ctypes.c_char_p
        lib.prl_naive_device_count.restype = ctypes.c_int
        lib.prl_naive_run.argtypes = [
            _i8, _i8, _i8, _i8, _i8, _i8,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            _u8, _u8, _i32, _ip, _ip, _ip, _u32]
        lib.prl_naive_run.restype = ctypes.c_int
        self._lib = lib
        if lib.prl_naive_device_count() < 1:
            raise NaiveCudaUnavailable("no CUDA device visible to libprl_naive.so")

    def run(self, A, B, E_AL, E_AR, E_BL, E_BR, pow_key: bytes, pow_target: int,
            hash_tile: int = 16):
        m, k = A.shape
        n = B.shape[1]
        r = E_AL.shape[1]
        def p8(a):
            a = np.ascontiguousarray(a, dtype=np.int8)
            return a, a.ctypes.data_as(_i8)
        A, pA = p8(A); B, pB = p8(B)
        EAL, pEAL = p8(E_AL); EAR, pEAR = p8(E_AR); EBL, pEBL = p8(E_BL); EBR, pEBR = p8(E_BR)
        key = np.frombuffer(pow_key, dtype=np.uint8).copy()
        tgt = np.frombuffer(int(pow_target).to_bytes(32, "little"), dtype=np.uint8).copy()
        C = np.zeros((m, n), dtype=np.int32)
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = self._lib.prl_naive_run(
            pA, pB, pEAL, pEAR, pEBL, pEBR, m, k, n, r, hash_tile, hash_tile,
            key.ctypes.data_as(_u8), tgt.ctypes.data_as(_u8),
            C.ctypes.data_as(_i32), ctypes.byref(found), ctypes.byref(a_row),
            ctypes.byref(b_col), tr.ctypes.data_as(_u32))
        if rc != 0:
            raise RuntimeError(f"prl_naive_run rc={rc}: {self._lib.prl_naive_last_error().decode()}")
        return C, bool(found.value), a_row.value, b_col.value, tr
