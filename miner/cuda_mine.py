"""ctypes wrapper for the FUSED tensor-core miner kernel (cuda/build/libprl_miner.so).

Uses the PERSISTENT CONTEXT API (prl_mine_ctx_*): device buffers + a CUDA stream are
allocated once per shape and reused across every attempt, so the orchestrator loop runs
near kernel speed instead of paying a cudaMalloc/cudaFree per attempt. The kernel itself
(noised GEMM + transcript + on-device BLAKE3) is validated bit-for-bit vs the golden
vectors on the GPU (cuda/tests/validate_mine.py). Linux/WSL only; noise_rank=128.
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
        lib.prl_mine_ctx_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.prl_mine_ctx_create.restype = ctypes.c_void_p
        lib.prl_mine_ctx_run.argtypes = [ctypes.c_void_p, _i8, _i8, _i8, _i8, _i8, _i8,
                                         _u8, _u8, _ip, _ip, _ip, _u32]
        lib.prl_mine_ctx_run.restype = ctypes.c_int
        lib.prl_mine_ctx_run_keyed.argtypes = [ctypes.c_void_p, _i8, _i8, _u8, _u8, _u8,
                                               _ip, _ip, _ip, _u32]
        lib.prl_mine_ctx_run_keyed.restype = ctypes.c_int
        lib.prl_mine_ctx_set_inputs.argtypes = [ctypes.c_void_p, _i8, _i8]
        lib.prl_mine_ctx_set_inputs.restype = ctypes.c_int
        lib.prl_mine_ctx_run_keyed_preloaded.argtypes = [ctypes.c_void_p, _u8, _u8, _u8,
                                                         _ip, _ip, _ip, _u32]
        lib.prl_mine_ctx_run_keyed_preloaded.restype = ctypes.c_int
        lib.prl_mine_ctx_run_keyed_preloaded_batch.argtypes = [
            ctypes.c_void_p, _u8, _u8, ctypes.c_int, _u8, _ip, _ip, _ip, _ip, _u32]
        lib.prl_mine_ctx_run_keyed_preloaded_batch.restype = ctypes.c_int
        lib.prl_mine_ctx_destroy.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._ctx: dict[tuple[int, int, int], int] = {}
        if lib.prl_mine_device_count() < 1:
            raise MineCudaUnavailable("no CUDA device visible to libprl_miner.so")

    def _ctx_for(self, m: int, k: int, n: int) -> int:
        key = (m, k, n)
        h = self._ctx.get(key)
        if h is None:
            h = self._lib.prl_mine_ctx_create(m, k, n)
            if not h:
                raise RuntimeError(f"ctx_create failed: {self._lib.prl_mine_last_error().decode()}")
            self._ctx[key] = h
        return h

    def run(self, A, B, E_AL, E_AR, E_BL, E_BR, pow_key: bytes, pow_target: int):
        m, k = A.shape
        n = B.shape[1]
        h = self._ctx_for(m, k, n)
        def p8(a):
            a = np.ascontiguousarray(a, dtype=np.int8)
            return a, a.ctypes.data_as(_i8)
        A, pA = p8(A); B, pB = p8(B)
        EAL, pEAL = p8(E_AL); EAR, pEAR = p8(E_AR); EBL, pEBL = p8(E_BL); EBR, pEBR = p8(E_BR)
        key = np.frombuffer(pow_key, dtype=np.uint8).copy()
        tgt = np.frombuffer(int(pow_target).to_bytes(32, "little"), dtype=np.uint8).copy()
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = self._lib.prl_mine_ctx_run(h, pA, pB, pEAL, pEAR, pEBL, pEBR,
                                        key.ctypes.data_as(_u8), tgt.ctypes.data_as(_u8),
                                        ctypes.byref(found), ctypes.byref(a_row), ctypes.byref(b_col),
                                        tr.ctypes.data_as(_u32))
        if rc != 0:
            raise RuntimeError(f"prl_mine_ctx_run rc={rc}: {self._lib.prl_mine_last_error().decode()}")
        return bool(found.value), a_row.value, b_col.value, tr

    def run_keyed(self, A, B, key_A: bytes, key_B: bytes, pow_target: int):
        """Noise is generated ON-GPU from the keys (no Python BLAKE3 loop)."""
        m, k = A.shape
        n = B.shape[1]
        h = self._ctx_for(m, k, n)
        def p8(a):
            a = np.ascontiguousarray(a, dtype=np.int8)
            return a, a.ctypes.data_as(_i8)
        A, pA = p8(A); B, pB = p8(B)
        ka = np.frombuffer(key_A, dtype=np.uint8).copy()
        kb = np.frombuffer(key_B, dtype=np.uint8).copy()
        tgt = np.frombuffer(int(pow_target).to_bytes(32, "little"), dtype=np.uint8).copy()
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = self._lib.prl_mine_ctx_run_keyed(h, pA, pB, ka.ctypes.data_as(_u8), kb.ctypes.data_as(_u8),
                                              tgt.ctypes.data_as(_u8), ctypes.byref(found),
                                              ctypes.byref(a_row), ctypes.byref(b_col),
                                              tr.ctypes.data_as(_u32))
        if rc != 0:
            raise RuntimeError(f"prl_mine_ctx_run_keyed rc={rc}: {self._lib.prl_mine_last_error().decode()}")
        return bool(found.value), a_row.value, b_col.value, tr

    def preload_inputs(self, A, B) -> None:
        m, k = A.shape
        n = B.shape[1]
        h = self._ctx_for(m, k, n)
        def p8(a):
            a = np.ascontiguousarray(a, dtype=np.int8)
            return a, a.ctypes.data_as(_i8)
        A, pA = p8(A); B, pB = p8(B)
        rc = self._lib.prl_mine_ctx_set_inputs(h, pA, pB)
        if rc != 0:
            raise RuntimeError(f"prl_mine_ctx_set_inputs rc={rc}: {self._lib.prl_mine_last_error().decode()}")

    def run_keyed_preloaded(self, shape: tuple[int, int, int], key_A: bytes, key_B: bytes, pow_target: int):
        m, k, n = shape
        h = self._ctx_for(m, k, n)
        ka = np.frombuffer(key_A, dtype=np.uint8).copy()
        kb = np.frombuffer(key_B, dtype=np.uint8).copy()
        tgt = np.frombuffer(int(pow_target).to_bytes(32, "little"), dtype=np.uint8).copy()
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = self._lib.prl_mine_ctx_run_keyed_preloaded(
            h, ka.ctypes.data_as(_u8), kb.ctypes.data_as(_u8), tgt.ctypes.data_as(_u8),
            ctypes.byref(found), ctypes.byref(a_row), ctypes.byref(b_col), tr.ctypes.data_as(_u32))
        if rc != 0:
            raise RuntimeError(
                f"prl_mine_ctx_run_keyed_preloaded rc={rc}: {self._lib.prl_mine_last_error().decode()}")
        return bool(found.value), a_row.value, b_col.value, tr

    def run_keyed_preloaded_batch(self, shape: tuple[int, int, int], keys_A: list[bytes],
                                  keys_B: list[bytes], pow_target: int):
        if len(keys_A) != len(keys_B) or not keys_A:
            raise ValueError("keys_A and keys_B must be non-empty lists of equal length")
        m, k, n = shape
        h = self._ctx_for(m, k, n)
        ka = np.frombuffer(b"".join(keys_A), dtype=np.uint8).copy()
        kb = np.frombuffer(b"".join(keys_B), dtype=np.uint8).copy()
        tgt = np.frombuffer(int(pow_target).to_bytes(32, "little"), dtype=np.uint8).copy()
        found_attempt = ctypes.c_int(0)
        found = ctypes.c_int(0); a_row = ctypes.c_int(0); b_col = ctypes.c_int(0)
        tr = np.zeros(16, dtype=np.uint32)
        rc = self._lib.prl_mine_ctx_run_keyed_preloaded_batch(
            h, ka.ctypes.data_as(_u8), kb.ctypes.data_as(_u8), len(keys_A),
            tgt.ctypes.data_as(_u8), ctypes.byref(found_attempt), ctypes.byref(found),
            ctypes.byref(a_row), ctypes.byref(b_col), tr.ctypes.data_as(_u32))
        if rc != 0:
            raise RuntimeError(
                f"prl_mine_ctx_run_keyed_preloaded_batch rc={rc}: "
                f"{self._lib.prl_mine_last_error().decode()}")
        return found_attempt.value, bool(found.value), a_row.value, b_col.value, tr

    def __del__(self):
        lib = getattr(self, "_lib", None)
        for h in getattr(self, "_ctx", {}).values():
            try:
                lib.prl_mine_ctx_destroy(h)
            except Exception:
                pass
