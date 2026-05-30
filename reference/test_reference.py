"""
Self-tests for the standalone NumPy Pearl reference (pearl_reference.py).

These reproduce the assertions of the official test suite
(pearl-official/miner/miner-base/tests/test_noisy_gemm.py) WITHOUT requiring
torch or the py-pearl-mining Rust extension, so they run anywhere numpy+blake3
are available.

The single strongest invariant is the *denoise identity*: regardless of the
random data, the noised+denoised result must equal the plain integer matmul
C == A @ B exactly. The official suite asserts the same (test_noisy_gemm.py:88-92).

Run:  pytest -q reference/test_reference.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from pearl_reference import (
    POW_TARGET_EASIEST,
    POW_TARGET_HARDEST,
    NoiseGenerator,
    NoisyGemm,
)

KEY_A = b"test_key_A_123456789012345678901"   # 32 bytes (test_noisy_gemm.py:12)
KEY_B = b"test_key_B_123456789012345678901"   # 32 bytes (test_noisy_gemm.py:13)
GOLDEN_DIR = pathlib.Path(__file__).resolve().parents[1] / "tests" / "golden"


def _rand_int7(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-64, 64, size=shape, dtype=np.int8)


# --- Official truth table (test_noisy_gemm.py:33-93) -----------------------
@pytest.mark.parametrize(
    "m,k,n,expect_block",
    [(64, 128, 64, False), (128, 256, 128, True), (256, 128, 256, False), (64, 256, 128, True)],
)
@pytest.mark.parametrize("matmul_tile_h,matmul_tile_w", [(32, 46), (48, 64), (64, 64)])
def test_official_truth_table(m, k, n, expect_block, matmul_tile_h, matmul_tile_w):
    gen = NoiseGenerator(noise_rank=64, noise_range=128)
    A = _rand_int7((m, k), seed=(m * 31 + k))
    B = _rand_int7((k, n), seed=(k * 17 + n))
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, m, k, n)
    gemm = NoisyGemm(noise_range=128, noise_rank=64,
                     matmul_tile_h=matmul_tile_h, matmul_tile_w=matmul_tile_w)
    target = POW_TARGET_EASIEST if expect_block else POW_TARGET_HARDEST
    C, found = gemm.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=target)
    expected = A.astype(np.int64) @ B.astype(np.int64)
    assert C.shape == (m, n)
    assert C.dtype == np.int32
    assert np.array_equal(C.astype(np.int64), expected)   # denoise identity
    assert found is expect_block


# --- Denoise identity across ranks/dims ------------------------------------
@pytest.mark.parametrize("m,k,n,r", [(64, 64, 64, 32), (128, 256, 128, 32), (128, 256, 256, 128)])
def test_denoise_identity(m, k, n, r):
    gen = NoiseGenerator(noise_rank=r, noise_range=128)
    A = _rand_int7((m, k), seed=1)
    B = _rand_int7((k, n), seed=2)
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, m, k, n)
    gemm = NoisyGemm(noise_range=128, noise_rank=r, matmul_tile_h=min(m, r), matmul_tile_w=min(n, r))
    C, _ = gemm.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=POW_TARGET_HARDEST)
    assert np.array_equal(C.astype(np.int64), A.astype(np.int64) @ B.astype(np.int64))


def test_zero_matrices_give_zero():
    r = 64
    gen = NoiseGenerator(noise_rank=r)
    A = np.zeros((128, 256), dtype=np.int8)
    B = np.zeros((256, 128), dtype=np.int8)
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    gemm = NoisyGemm(noise_rank=r)
    C, found = gemm.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=POW_TARGET_HARDEST)
    assert np.all(C == 0)
    assert found is False


def test_int7_boundary_values():
    r = 64
    gen = NoiseGenerator(noise_rank=r)
    A = np.full((128, 256), -64, dtype=np.int8)
    B = np.full((256, 128), 63, dtype=np.int8)
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    gemm = NoisyGemm(noise_rank=r)
    C, _ = gemm.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=POW_TARGET_HARDEST)
    assert np.array_equal(C.astype(np.int64), A.astype(np.int64) @ B.astype(np.int64))


def test_out_of_range_rejected():
    r = 64
    gen = NoiseGenerator(noise_rank=r)
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    gemm = NoisyGemm(noise_rank=r)
    A_bad = np.full((128, 256), 64, dtype=np.int8)   # 64 > max int7 (63)
    B_ok = _rand_int7((256, 128), seed=3)
    with pytest.raises(ValueError, match="tensor must be in the range"):
        gemm.run(A_bad, B_ok, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A)


# --- Noise generator structure & determinism -------------------------------
def test_noise_generator_determinism():
    gen = NoiseGenerator(noise_rank=128)
    a = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    b = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_noise_generator_structure():
    r = 128
    gen = NoiseGenerator(noise_rank=r, noise_range=128)
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    # dense uniform matrices live in [-32, 31] (noise_range=128 -> range_mask 63, zpt 32)
    assert E_AL.min() >= -32 and E_AL.max() <= 31
    assert E_BR.min() >= -32 and E_BR.max() <= 31
    # E_AR (r x k): every column has exactly one +1 and one -1
    assert np.all((E_AR == 1).sum(axis=0) == 1)
    assert np.all((E_AR == -1).sum(axis=0) == 1)
    # E_BL (k x r): every row has exactly one +1 and one -1
    assert np.all((E_BL == 1).sum(axis=1) == 1)
    assert np.all((E_BL == -1).sum(axis=1) == 1)


def test_found_block_location_first_tile():
    r = 64
    gen = NoiseGenerator(noise_rank=r)
    A = _rand_int7((128, 256), seed=11)
    B = _rand_int7((256, 128), seed=12)
    E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, 128, 256, 128)
    gemm = NoisyGemm(noise_rank=r)
    C, found = gemm.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=POW_TARGET_EASIEST)
    assert found is True
    assert gemm.opened_block.A_row_indices == list(range(0, 16))
    assert gemm.opened_block.B_column_indices == list(range(0, 16))


# --- Golden replay (only if golden vectors have been generated) ------------
def _golden_cases():
    manifest = GOLDEN_DIR / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())["cases"]


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda c: c["name"])
def test_golden_replay(case):
    data = np.load(GOLDEN_DIR / case["file"])
    gemm = NoisyGemm(
        noise_range=case["noise_range"], noise_rank=case["noise_rank"],
        hash_tile_h=case["hash_tile_h"], hash_tile_w=case["hash_tile_w"],
        matmul_tile_h=case["matmul_tile_h"], matmul_tile_w=case["matmul_tile_w"],
    )
    C, found = gemm.run(
        data["A"], data["B"], data["E_AL"], data["E_AR"], data["E_BL"], data["E_BR"],
        pow_key=bytes.fromhex(case["key_A"]), pow_target=int(case["pow_target"], 16),
    )
    assert np.array_equal(C, data["C"])
    assert found is case["found_block"]
    if found:
        assert gemm.opened_block.A_row_indices == case["A_row_indices"]
        assert gemm.opened_block.B_column_indices == case["B_column_indices"]
