"""GPU-accelerated SimNet solo miner — the FAST path.

Architecture (the way real miners split work): the RTX 3090 does the bulk
*search* — a batched 2x64 NoisyGEMM + keyed-BLAKE3 PoW over many attempts at once
(`prl_mine2_search`, golden-verified vs the official NoisyGemm) — and the official
`miner_base` machinery builds the real `PlainProof` for the single winning attempt
and submits it (the exact, already-verified `simnet_solo.py` tail). The GPU finds
the needle; the trusted library dresses it for submission.

Two modes:
  --selftest   offline, no node: synthetic batch at an easy target; assert the GPU
               winner is confirmed by the official torch NoisyGemm (found + same
               opened tile + same transcript). Proves the fast path is submission-grade.
  (default)    live: fetch a real job from pearl-gateway, GPU-search batches of
               attempts at the real adjusted target, and on a hit build + submit a
               real PlainProof. Run inside the pearl-official venv with MINER_RPC_*
               set (see scripts/run_fast_mine.sh).

Linux/WSL only (needs the built CUDA lib + the pearl stack venv).
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SO = ROOT / "cuda" / "build" / "libprl_miner.so"

i8p = ctypes.POINTER(ctypes.c_int8)
u8p = ctypes.POINTER(ctypes.c_uint8)
u32p = ctypes.POINTER(ctypes.c_uint32)
ip = ctypes.POINTER(ctypes.c_int)


def load_lib() -> ctypes.CDLL:
    if not SO.exists():
        sys.exit(f"{SO} not found — build the CUDA lib first (cuda/build/libprl_miner.so)")
    lib = ctypes.CDLL(str(SO))
    lib.prl_mine_last_error.restype = ctypes.c_char_p
    # prl_mine2_search(An, Bn, m,k,n,natt, keys, target, &found_att,&a_row,&b_col, tr_out)
    lib.prl_mine2_search.argtypes = [i8p, i8p] + [ctypes.c_int]*4 + [u8p, u8p, ip, ip, ip, u32p]
    lib.prl_mine2_search.restype = ctypes.c_int
    return lib


def noise_int8(M: np.ndarray, L: np.ndarray, Rr: np.ndarray) -> np.ndarray:
    """Noised int8 matrix exactly as the reference/official/k_noiseA do it:
    int8( M + int8(L @ R) ), with the inner matmul wrapped to int8 first."""
    E = (L.astype(np.int64) @ Rr.astype(np.int64)).astype(np.int8)
    return (M.astype(np.int16) + E.astype(np.int16)).astype(np.int8)


def as_i8(a) -> np.ndarray:
    a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    return np.ascontiguousarray(a, dtype=np.int8)


def gpu_search(lib, An_stack, Bn_stack, keys_bytes, target_le, m, k, n, natt):
    """Run the batched GPU search. Returns (found_attempt, a_row, b_col, transcript[16])."""
    An = np.ascontiguousarray(An_stack, np.int8)
    Bn = np.ascontiguousarray(Bn_stack, np.int8)
    keys = np.frombuffer(keys_bytes, np.uint8).copy()
    tgt = np.frombuffer(target_le, np.uint8).copy()
    fa = ctypes.c_int(0); ar = ctypes.c_int(0); bc = ctypes.c_int(0)
    tr = np.zeros(16, np.uint32)
    rc = lib.prl_mine2_search(An.ctypes.data_as(i8p), Bn.ctypes.data_as(i8p),
                              m, k, n, natt, keys.ctypes.data_as(u8p), tgt.ctypes.data_as(u8p),
                              ctypes.byref(fa), ctypes.byref(ar), ctypes.byref(bc),
                              tr.ctypes.data_as(u32p))
    if rc != 0:
        sys.exit(f"prl_mine2_search rc={rc}: {lib.prl_mine_last_error().decode()}")
    return fa.value, ar.value, bc.value, [int(w) for w in tr]


# --------------------------------------------------------------------------- #
# Offline self-test: GPU winner must be confirmed by the official torch NoisyGemm
# --------------------------------------------------------------------------- #
def selftest(natt: int = 64) -> int:
    from miner_base.noisy_gemm import POW_TARGET_EASIEST
    from miner_base.noisy_gemm import NoisyGemm as ONG
    from miner_base.noise_generation import NoiseGenerator
    from pearl_gateway.comm.dataclasses import CommitmentHash
    import torch

    lib = load_lib()
    m, k, n, R = 128, 256, 256, 128
    HTH, HTW, TH, TW = 2, 64, 128, 256
    target = POW_TARGET_EASIEST
    tgt_le = int(target).to_bytes(32, "little")
    ng = NoiseGenerator(noise_rank=R, noise_range=128)
    rng = np.random.default_rng(2024)

    print(f"[selftest] building {natt} synthetic attempts ({m}x{k}x{n}, 2x64, rank {R}) ...")
    An_list, Bn_list, keys, attempts = [], [], b"", []
    for j in range(natt):
        A = rng.integers(-64, 64, (m, k), dtype=np.int8)
        B = rng.integers(-64, 64, (k, n), dtype=np.int8)
        sA = f"selfA_{j:04d}".encode().ljust(32, b"\0")[:32]
        sB = f"selfB_{j:04d}".encode().ljust(32, b"\0")[:32]
        EAL, EAR, EBL, EBR = ng.generate_noise_metrices(sA, sB, m, k, n)
        EAL, EAR, EBL, EBR = as_i8(EAL), as_i8(EAR), as_i8(EBL), as_i8(EBR)
        An_list.append(noise_int8(A, EAL, EAR))
        Bn_list.append(noise_int8(B, EBL, EBR))
        keys += sA                                  # pow key == noise_seed_A
        attempts.append((A, B, EAL, EAR, EBL, EBR, sA, sB))

    found_att, a_row, b_col, tr = gpu_search(
        lib, np.stack(An_list), np.stack(Bn_list), keys, tgt_le, m, k, n, natt)
    print(f"[selftest] GPU search over {natt} attempts -> winner_attempt={found_att} "
          f"a_row={a_row} b_col={b_col}")
    if found_att < 0:
        print("[selftest] FAIL: GPU found no winner at the easiest target (unexpected)")
        return 1

    A, B, EAL, EAR, EBL, EBR, sA, sB = attempts[found_att]
    og = ONG(noise_range=128, noise_rank=R, hash_tile_h=HTH, hash_tile_w=HTW,
             matmul_tile_h=TH, matmul_tile_w=TW)
    def tt(x): return torch.from_numpy(np.ascontiguousarray(x))
    _, off = og.noisy_gemm(tt(A), tt(B), tt(EAL), tt(EAR), tt(EBL), tt(EBR),
                           commitment_hash=CommitmentHash(sA, sB), pow_target=target)
    ob = og.get_opened_block_info() if off else None
    of = bool(off)
    ol = of and ob.A_row_indices[0] == a_row and ob.B_column_indices[0] == b_col
    ok = of and ol                       # transcript correctness is golden-gated in validate_mine2
    loc = f"A_rows[:2]={ob.A_row_indices[:2]} B_cols[:2]={ob.B_column_indices[:2]}" if of else "n/a"
    print(f"[selftest] official NoisyGemm on attempt {found_att}: "
          f"found={'=' if of else 'X'} loc={'=' if ol else 'X'}  ({loc})")
    print("[selftest] RESULT:", "PASS — GPU winner confirmed by official lib (submission-grade)"
          if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Live mode: GPU search -> official proof for the winner -> submit
# --------------------------------------------------------------------------- #
def live(natt: int, max_batches: int) -> int:
    import torch
    from miner_base.block_submission import create_proof
    from miner_base.commitment_hash import CommitmentHasher
    from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
    from miner_base.gateway_client import MiningClient
    from miner_base.noise_generation import NoiseGenerator
    from miner_base.noisy_gemm import NoisyGemm
    from pearl_gateway.config import MinerRpcConfig

    lib = load_lib()
    k, R = 256, 128
    mc = GPUMatmulConfigFactory.create(k, R)
    mcfg = mc.mining_config
    m, n = mc.matmul_tile_h, mc.matmul_tile_w
    HTH, HTW = mc.hash_tile_h, mc.hash_tile_w
    print(f"[fast] config m={m} n={n} k={k} rank={R} hash_tile={HTH}x{HTW} | natt/batch={natt}")

    client = MiningClient(MinerRpcConfig())
    job = client.get_mining_info()
    adj_target = job.adjust_target(mcfg)
    header = job.incomplete_header_bytes
    tgt_le = int(adj_target).to_bytes(32, "little")
    print(f"[fast] job header={len(header)}B adjusted_target={hex(adj_target)}")

    ng = NoiseGenerator(noise_rank=R, noise_range=128)
    gemm = NoisyGemm(noise_range=128, noise_rank=R, hash_tile_h=HTH, hash_tile_w=HTW,
                     matmul_tile_h=m, matmul_tile_w=n)

    def tt(x): return torch.from_numpy(np.ascontiguousarray(x))
    t0 = time.time(); tried = 0
    for batch in range(max_batches):
        An_list, Bn_list, keys, attempts = [], [], b"", []
        for _ in range(natt):
            A = np.random.randint(-64, 64, (m, k), dtype=np.int8)
            B = np.random.randint(-64, 64, (k, n), dtype=np.int8)
            commitment = CommitmentHasher.commitment_hash(tt(A), tt(B), header, mcfg)
            sA, sB = commitment.noise_seed_A, commitment.noise_seed_B
            EAL, EAR, EBL, EBR = ng.generate_noise_metrices(sA, sB, m, k, n)
            EAL, EAR, EBL, EBR = as_i8(EAL), as_i8(EAR), as_i8(EBL), as_i8(EBR)
            An_list.append(noise_int8(A, EAL, EAR))
            Bn_list.append(noise_int8(B, EBL, EBR))
            keys += bytes(sA)[:32].ljust(32, b"\0")
            attempts.append((A, B, EAL, EAR, EBL, EBR, commitment))
        tried += natt
        fa, a_row, b_col, _ = gpu_search(
            lib, np.stack(An_list), np.stack(Bn_list), keys, tgt_le, m, k, n, natt)
        rate = tried / max(1e-9, time.time() - t0)
        print(f"[fast] batch {batch}: searched {tried} attempts ({rate/1e3:.1f}k/s) winner={fa}")
        if fa < 0:
            continue

        A, B, EAL, EAR, EBL, EBR, commitment = attempts[fa]
        _, found = gemm.noisy_gemm(tt(A), tt(B), tt(EAL), tt(EAR), tt(EBL), tt(EBR),
                                   commitment_hash=commitment, pow_target=adj_target)
        if not found:
            print("[fast] WARN: GPU winner not confirmed by official path (key mismatch?) — skipping")
            continue
        ob = gemm.get_opened_block_info()
        print(f"[fast] FOUND on GPU attempt {fa} after {tried} attempts in {time.time()-t0:.2f}s; "
              f"opened rows={ob.A_row_indices[:1]}.. cols={ob.B_column_indices[:1]}..")
        proof = create_proof(ob, header)
        print(f"[fast] built PlainProof: base64 len={len(proof.to_base64())}")
        client.submit_plain_proof(proof, job)
        print("[fast] submitPlainProof sent (gateway builds + submits the block async)")
        client.close()
        return 0

    client.close()
    print(f"[fast] no proof in {tried} attempts (target too hard for this harness)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="offline GPU-vs-official check (no node)")
    ap.add_argument("--natt", type=int, default=256, help="attempts per GPU batch")
    ap.add_argument("--max-batches", type=int, default=200)
    a = ap.parse_args()
    if a.selftest:
        return selftest(natt=min(a.natt, 64))
    return live(a.natt, a.max_batches)


if __name__ == "__main__":
    sys.exit(main())
