"""Minimal profiling target: runs the fused k_mine kernel a few times so a profiler
(ncu) can capture it. Usage: ncu ... python cuda/tests/prof_target.py"""
import ctypes
import pathlib

SO = pathlib.Path(__file__).resolve().parents[1] / "build" / "libprl_miner.so"
lib = ctypes.CDLL(str(SO))
lib.prl_mine_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_double)]
lib.prl_mine_bench.restype = ctypes.c_int
ms = ctypes.c_double(0)
lib.prl_mine_bench(1024, 1024, 1024, 5, ctypes.byref(ms))
print("k_mine avg ms:", ms.value)
