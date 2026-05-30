"""
Pearl PRL Proof-of-Useful-Work — standalone CPU reference implementation.

This is a faithful, dependency-light (numpy + blake3 only) re-implementation of the
official Pearl miner reference, which lives in:

    pearl-official/miner/miner-base/src/miner_base/
        noisy_gemm.py        -> NoisyGemm  (this file: class NoisyGemm)
        noise_generation.py  -> NoiseGenerator (this file: class NoiseGenerator)
        inner_hash.py        -> InnerHasher (this file: inner_hash_tensor)

The official reference uses PyTorch (torch==2.11.0) and, for the commitment-hash /
Merkle-proof path, the compiled Rust extension `py-pearl-mining` (not shipped as
source). This file deliberately reproduces ONLY the parts that are needed to
generate deterministic golden vectors for the CUDA sm_86 backend:

  * noise generation from 32-byte keys (keyed BLAKE3),
  * the additive low-rank noising of A and B,
  * the tiled noisy GEMM with intermediate-tile inner-hash accumulation,
  * the keyed-BLAKE3 PoW transcript check,
  * exact denoising so that C == A @ B.

The `pow_key` and `noise_seed_*` values are taken as raw 32-byte inputs here (in the
full stack they are derived by CommitmentHasher from Merkle roots over A and B; that
derivation needs py-pearl-mining and is documented in docs/protocol-notes.md but is
NOT required to validate the GEMM/hash/PoW kernels).

Every nontrivial step cites the official source as `file:line` so the mapping can be
audited. Do not "improve" the algorithm here — bit-exactness with the official
reference is the whole point.

Cross-check: reference/verify_against_official.py feeds identical inputs into the
official torch NoisyGemm (when installed, e.g. inside WSL) and asserts equality.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
from blake3 import blake3

# --- constants (noisy_gemm.py:17-24) ---------------------------------------
TRANSCRIPT_SIZE_U32 = 16          # 64-byte transcript == blake3 MSG_BLOCK_SIZE_U32
POW_TARGET_HARDEST = 0
POW_TARGET_EASIEST = 2**256 - 1
HASH_ACCUMULATE_ROTATION = 13     # must match HASH_ACCUMULATE_ROTATION in pow_utils.hpp
U32 = 0xFFFFFFFF


def rotl32(x: int, n: int) -> int:
    """Rotate-left a 32-bit value (noisy_gemm.py:27-29)."""
    x &= U32
    return ((x << n) | (x >> (32 - n))) & U32


def mul_hi_u32(a: int, b: int) -> int:
    """High 32 bits of a 32x32->64 unsigned multiply (noise_generation.py:20-22)."""
    return ((a & U32) * (b & U32)) >> 32


# ---------------------------------------------------------------------------
# Noise generation — faithful port of noise_generation.py
# ---------------------------------------------------------------------------
class NoiseGenerator:
    """Deterministic keyed-BLAKE3 noise generator (noise_generation.py:25-187)."""

    DIGEST_SIZE = 32

    def __init__(self, noise_rank: int = 128, noise_range: int = 128) -> None:
        if noise_range & (noise_range - 1) or noise_range == 0:
            raise ValueError("noise_range must be a power of two")
        if noise_rank & (noise_rank - 1) or noise_rank == 0:
            raise ValueError("noise_rank must be a power of two")
        if noise_range > 128:
            raise ValueError("noise_range must fit in uint7")
        if noise_rank % self.DIGEST_SIZE != 0:
            raise ValueError(f"noise_rank must be divisible by {self.DIGEST_SIZE}")

        self.noise_rank = noise_rank
        # noise_generation.py:48-52
        idxs_per_col = 2
        _noise_range = noise_range // idxs_per_col
        self.zero_point_translation = _noise_range // 2
        self.range_mask = _noise_range - 1
        self.rank_mask = noise_rank - 1

    @staticmethod
    def _random_hash(index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
        """noise_generation.py:101-105 — 8x int32 prepend (slot=prepend_index set to
        1+index), then seed, keyed-BLAKE3."""
        prepend = np.zeros(8, dtype=np.int32)
        prepend[prepend_index] = 1 + index
        return blake3(prepend.tobytes() + seed, key=key).digest()

    def _uniform_matrix(self, seed: bytes, key: bytes, rows: int) -> np.ndarray:
        """Dense uniform int8 matrix in [-zpt, range_mask-zpt] (noise_generation.py:107-135)."""
        cols = self.noise_rank
        assert len(seed) == 32 and len(key) == 32
        draws = -(-rows * cols // self.DIGEST_SIZE)  # ceil
        raw = b"".join(self._random_hash(i, seed, key, 0) for i in range(draws))
        flat = np.frombuffer(raw, dtype=np.uint8)[: rows * cols].astype(np.int16)
        out = ((flat & self.range_mask) - self.zero_point_translation).astype(np.int8)
        return out.reshape(rows, cols)

    def _permutation_matrix(
        self, seed: bytes, key: bytes, rows: int, cols: int, assign_columns: bool
    ) -> np.ndarray:
        """Signed permutation matrix: each line has one +1 and one -1
        (noise_generation.py:137-187)."""
        assert (rows == self.noise_rank) or (cols == self.noise_rank)
        assert len(seed) == 32 and len(key) == 32
        mat = np.zeros((rows, cols), dtype=np.int8)
        required_lines = cols if assign_columns else rows
        bytes_per_line = 4
        draws = -(-required_lines * bytes_per_line // self.DIGEST_SIZE)  # ceil
        per_hash = self.DIGEST_SIZE // bytes_per_line  # 8

        for i in range(draws):
            words = np.frombuffer(self._random_hash(i, seed, key, 1), dtype=np.uint32)
            for k in range(per_hash):
                assignment_index = i * per_hash + k
                if assignment_index >= required_lines:
                    break
                u = int(words[k])
                first_idx = u & self.rank_mask
                second_idx = first_idx ^ (1 + mul_hi_u32(self.noise_rank - 1, u))
                second_idx &= self.rank_mask  # keep inside [0, rank)
                perm = np.zeros(self.noise_rank, dtype=np.int8)
                perm[first_idx] = 1
                perm[second_idx] = -1
                if assign_columns:
                    mat[:, assignment_index] = perm
                else:
                    mat[assignment_index, :] = perm
        return mat

    def generate(
        self, key_A: bytes, key_B: bytes, A_rows: int, common_dim: int, B_cols: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (E_AL m x r, E_AR r x k, E_BL k x r, E_BR r x n)
        (noise_generation.py:54-99)."""
        for name, d in (("A_rows", A_rows), ("common_dim", common_dim), ("B_cols", B_cols)):
            if d < self.noise_rank:
                raise ValueError(f"{name} must be >= noise_rank ({self.noise_rank}), got {d}")
        seed_A = b"A_tensor" + b"\x00" * 24
        seed_B = b"B_tensor" + b"\x00" * 24
        E_AL = self._uniform_matrix(seed_A, key_A, A_rows)
        E_AR = self._permutation_matrix(seed_A, key_A, self.noise_rank, common_dim, True)
        E_BL = self._permutation_matrix(seed_B, key_B, common_dim, self.noise_rank, False)
        E_BR = self._uniform_matrix(seed_B, key_B, B_cols).T.copy()  # (n x r) -> (r x n)
        return E_AL, E_AR, E_BL, E_BR


# ---------------------------------------------------------------------------
# Inner hash — faithful port of inner_hash.py
# ---------------------------------------------------------------------------
def inner_hash_tile(tile_i32: np.ndarray) -> int:
    """XOR-reduce all int32 elements viewed as uint32 (inner_hash.py:7-32).

    XOR is commutative+associative so flatten order is irrelevant to the result.
    """
    return int(np.bitwise_xor.reduce(np.ascontiguousarray(tile_i32).reshape(-1).view(np.uint32)))


def inner_hash_tensor(block_i32: np.ndarray, tile_h: int, tile_w: int):
    """Return list of (hi, wi, hash_u32) for each tile_h x tile_w tile
    (inner_hash.py:41-65)."""
    nh = block_i32.shape[0] // tile_h
    nw = block_i32.shape[1] // tile_w
    out = []
    for hi in range(nh):
        for wi in range(nw):
            tile = block_i32[hi * tile_h:(hi + 1) * tile_h, wi * tile_w:(wi + 1) * tile_w]
            out.append((hi, wi, inner_hash_tile(tile)))
    return out


# ---------------------------------------------------------------------------
# Noisy GEMM — faithful port of noisy_gemm.py
# ---------------------------------------------------------------------------
@dataclass
class OpenedBlock:
    A_row_indices: list[int]
    B_column_indices: list[int]
    transcript_words: list[int] = field(default_factory=list)


def _i64(a: np.ndarray) -> np.ndarray:
    return a.astype(np.int64)


class NoisyGemm:
    """Tiled noisy GEMM + inner-hash PoW (noisy_gemm.py:108-696)."""

    def __init__(
        self,
        noise_range: int = 128,
        noise_rank: int = 128,
        hash_tile_h: int = 16,
        hash_tile_w: int = 16,
        matmul_tile_h: int = 128,
        matmul_tile_w: int = 128,
    ) -> None:
        if noise_range > 128:
            raise ValueError("noise_range must fit in uint7")
        if matmul_tile_h < hash_tile_h or matmul_tile_w < hash_tile_w:
            raise ValueError(
                f"matmul_tile_h={matmul_tile_h} and matmul_tile_w={matmul_tile_w} must be "
                f">= hash_tile_h={hash_tile_h} and hash_tile_w={hash_tile_w}"
            )
        self.noise_range = noise_range
        self.noise_rank = noise_rank
        self.hash_tile_h = hash_tile_h
        self.hash_tile_w = hash_tile_w
        self.matmul_tile_h = matmul_tile_h
        self.matmul_tile_w = matmul_tile_w
        self.opened_block: OpenedBlock | None = None

    # --- range validation (noisy_gemm.py:156-173) ---
    def _validate_matrix_range(self, t: np.ndarray) -> None:
        data_range = 256 - self.noise_range
        lo = -data_range // 2
        hi = lo + data_range - 1
        if t.min() < lo or t.max() > hi:
            raise ValueError(f"tensor must be in the range [{lo}, {hi}]")

    # --- noising (noisy_gemm.py:175-283) ---
    def noise_A(self, A, E_AL, E_AR, E_BL):
        self._validate_matrix_range(A)
        E_A = (_i64(E_AL) @ _i64(E_AR)).astype(np.int8)            # m x k
        A_noised = (A.astype(np.int16) + E_A).astype(np.int8)      # int8 wrap matches torch
        A_E_BL = (_i64(A) @ _i64(E_BL)).astype(np.int32)          # m x r
        return A_noised, A_E_BL

    def noise_B(self, B, E_AR, E_BL, E_BR):
        self._validate_matrix_range(B)
        E_B = (_i64(E_BL) @ _i64(E_BR)).astype(np.int8)            # k x n
        B_noised = (B.astype(np.int16) + E_B).astype(np.int8)
        EAR_BpEB = (_i64(E_AR) @ _i64(B_noised)).astype(np.int32)  # r x n
        return B_noised, EAR_BpEB

    # --- PoW transcript check (noisy_gemm.py:309-326) ---
    @staticmethod
    def _check_pow_target(transcript_words: list[int], pow_key: bytes, pow_target: int) -> bool:
        buf = b"".join(struct.pack("<I", w & U32) for w in transcript_words)
        digest = blake3(buf, key=pow_key).digest()
        return int.from_bytes(digest, "little") <= pow_target

    # --- tiled matmul with transcript accumulation (noisy_gemm.py:357-520) ---
    def _tiled_matmul(self, A, B, pow_key, pow_target):
        assert A.shape[1] == B.shape[0]
        m, k = A.shape
        n = B.shape[1]
        hth, htw, r = self.hash_tile_h, self.hash_tile_w, self.noise_rank
        if m < hth or n < htw or r < min(hth, htw):
            raise ValueError("matmul tile should be larger than hash tile and noise rank")

        C = np.zeros((m, n), dtype=np.int32)
        found = False
        for i in range(0, m, r):
            i_max = min(i + r, m)
            for j in range(0, n, r):
                j_max = min(j + r, n)
                bh, bw = i_max - i, j_max - j
                nth, ntw = bh // hth, bw // htw
                transcripts = [[[0] * TRANSCRIPT_SIZE_U32 for _ in range(ntw)] for _ in range(nth)]
                C_block = np.zeros((bh, bw), dtype=np.int32)
                reduction = 0
                has_full = False
                for p in range(0, k, r):
                    p_max = min(p + r, k)
                    C_block = (C_block.astype(np.int64)
                               + _i64(A[i:i_max, p:p_max]) @ _i64(B[p:p_max, j:j_max])).astype(np.int32)
                    if bh >= hth and bw >= htw and (p_max - p == r):
                        for hi, wi, h in inner_hash_tensor(C_block, hth, htw):
                            idx = reduction % TRANSCRIPT_SIZE_U32
                            transcripts[hi][wi][idx] = rotl32(transcripts[hi][wi][idx], HASH_ACCUMULATE_ROTATION) ^ h
                        reduction += 1
                        has_full = True
                if not found and has_full:
                    for hi in range(nth):
                        for wi in range(ntw):
                            if self._check_pow_target(transcripts[hi][wi], pow_key, pow_target):
                                self.opened_block = OpenedBlock(
                                    A_row_indices=list(range(i + hi * hth, i + hi * hth + hth)),
                                    B_column_indices=list(range(j + wi * htw, j + wi * htw + htw)),
                                    transcript_words=list(transcripts[hi][wi]),
                                )
                                found = True
                                break
                        if found:
                            break
                C[i:i_max, j:j_max] = C_block
        return C, found

    # --- full noisy gemm (noisy_gemm.py:526-696) ---
    def gemm(self, A_noised, B_noised, E_AL, E_BR, A_E_BL, EAR_BpEB, pow_key, pow_target):
        A_EB = (A_E_BL.astype(np.int64) @ _i64(E_BR)).astype(np.int32)
        EA_BpEB = (_i64(E_AL) @ EAR_BpEB.astype(np.int64)).astype(np.int32)
        C_noised, found = self._tiled_matmul(A_noised, B_noised, pow_key, pow_target)
        C = (C_noised.astype(np.int64) - A_EB - EA_BpEB).astype(np.int32)
        return C, found

    def run(self, A, B, E_AL, E_AR, E_BL, E_BR, pow_key, pow_target=POW_TARGET_HARDEST):
        """Top-level entry (noisy_gemm.py:598-696). Mirrors NoisyGemm.noisy_gemm.

        A,B int8; E_* int8; pow_key 32 bytes; pow_target uint256.
        Returns (C int32 m x n, found_block bool). C is always exactly A @ B.
        """
        for t, name in ((A, "A"), (B, "B"), (E_AL, "E_AL"), (E_AR, "E_AR"), (E_BL, "E_BL"), (E_BR, "E_BR")):
            if t.dtype != np.int8:
                raise ValueError(f"{name} must be int8, got {t.dtype}")
        self._validate_matrix_range(A)
        self._validate_matrix_range(B)
        if not (POW_TARGET_HARDEST <= pow_target <= POW_TARGET_EASIEST):
            raise ValueError("pow_target out of range")
        if A.shape[0] < self.matmul_tile_h or A.shape[1] < self.noise_rank or B.shape[1] < self.matmul_tile_w:
            raise ValueError(
                "expected A.shape[0] and A.shape[1] and B.shape[1] to be greater than "
                "matmul_tile_h and noise_rank and matmul_tile_w"
            )
        self.opened_block = None
        A_noised, A_E_BL = self.noise_A(A, E_AL, E_AR, E_BL)
        B_noised, EAR_BpEB = self.noise_B(B, E_AR, E_BL, E_BR)
        C, found = self.gemm(A_noised, B_noised, E_AL, E_BR, A_E_BL, EAR_BpEB, pow_key, pow_target)
        return C, found
