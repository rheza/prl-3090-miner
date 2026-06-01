"""ctypes wrapper for the FUSED tensor-core miner kernel (cuda/build/libprl_miner.so).

This is the real mining hot loop on tensor cores (noised GEMM + transcript + on-device
BLAKE3 PoW), validated bit-for-bit vs the golden vectors on the GPU
(cuda/tests/validate_mine.py). Specialized for noise_rank=128; needs m%128==n%128==k%128==0.
Linux/WSL only. Build: scripts/build_miner_wsl.sh.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_SO = pathlib.Path(__file__).resolve().parents[1] / "cuda" / "build" / "libprl_miner.so"
_i8 = ctypes.POINTER(ctypes.c_int8)
_u8 = ctypes.POINTER(ctypes.c_uint8)
_u32 = ctypes.POINTER(ctypes.c_uint32)
_ip = ctypes.POINTER(ctypes.c_int)


class MineCudaUnavailable(RuntimeError):
    pass


class MineCuda:
    def __init__(self) -> None:
        if not _SO.exists():
            raise MineCudaUnavailable(f"{_SO} not found — build with scripts/build_miner_wsl.sh")
        try:
            lib = ctypes.CDLL(str(_SO))
        except OSError as exc:
            raise MineCudaUnavailable(f"cannot load {_SO}: {exc}") from exc
        lib.prl_mine_last_error.restype = ctypes.c_char_p
        lib.prl_mine_device_count.restype = ctypes.c_int
        lib.prl_mine_run.argtypes = [_i8, _i8, _i8, _i8, _i8, _i8,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     _u8, _u8, _ip, _ip, _ip, _u32]
        lib.prl_mine_run.restype = ctypes.c_int
        self._lib = lib
        if lib.prl_mine_device_count() < 1:
            raise MineCudaUnavailable("no CUDA device visible to libprl_miner.so")

    def run(self, A, B, E_AL, E_AR, E_BL, E_BR, pow_key: bytes, pow_target: int):
        m, k = A.shape
        n = B.shape[1]
        def p8(a):
            a = np.ascontiguousarray(a, dtype=np.int8)
            return a, a.ctypes.data_as(_i8)
        A, pA = p8(A); B, pB = p8(B)
        EAL, pEAL = p8(E_AL); EAR, pEAR = p8(E_AR); EBL, pEBL = p8(E_BL); EBR, pEBR = p8(E_BR)
        key = np.frombuffer(pow_key, dtype=np.uint8).copy()
        tgt = np.frombuffer(int(pow_target).to_bytes(32, "little"), dtype=np.uint8).copy()
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = self._lib.prl_mine_run(pA, pB, pEAL, pEAR, pEBL, pEBR, m, k, n,
                                    key.ctypes.data_as(_u8), tgt.ctypes.data_as(_u8),
                                    ctypes.byref(found), ctypes.byref(a_row), ctypes.byref(b_col),
                                    tr.ctypes.data_as(_u32))
        if rc != 0:
            raise RuntimeError(f"prl_mine_run rc={rc}: {self._lib.prl_mine_last_error().decode()}")
        return bool(found.value), a_row.value, b_col.value, tr
