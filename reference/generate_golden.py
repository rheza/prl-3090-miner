"""
Generate deterministic golden test vectors for the Pearl PoUW pipeline.

These vectors are the correctness oracle for the CUDA sm_86 backend
(Milestone 2 / Milestone 4). Each case writes:

  tests/golden/<name>.npz   : exact int8 inputs A,B,E_AL,E_AR,E_BL,E_BR and int32 C
  tests/golden/manifest.json: metadata + expected outputs (found_block, winning
                              row/col indices, the 16-word PoW transcript of the
                              winning tile, and a BLAKE3 digest of C)

A CUDA kernel is correct iff, given the .npz inputs, it reproduces C bit-for-bit,
the same found_block, the same winning indices, and the same transcript words.

Inputs A and B are drawn from a fixed-seed numpy Generator so the vectors are
fully reproducible. Noise matrices are derived from the fixed 32-byte keys via the
faithful NoiseGenerator, exactly as the real miner derives them from the
commitment-hash chain (see docs/protocol-notes.md).

Run:  python reference/generate_golden.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from blake3 import blake3

from pearl_reference import POW_TARGET_EASIEST, POW_TARGET_HARDEST, NoiseGenerator, NoisyGemm

KEY_A = b"test_key_A_123456789012345678901"
KEY_B = b"test_key_B_123456789012345678901"
OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "golden"

# name, m, k, n, rank, hash_tile, matmul_tile_h, matmul_tile_w, target, seed
CASES = [
    ("g1_r32_small_nofound", 64, 64, 64, 32, 16, 32, 32, POW_TARGET_HARDEST, 100),
    ("g2_r64_square_nofound", 128, 128, 128, 64, 16, 64, 64, POW_TARGET_HARDEST, 101),
    ("g3_r64_square_found", 128, 128, 128, 64, 16, 64, 64, POW_TARGET_EASIEST, 101),
    ("g4_prod_rank128_nofound", 128, 256, 256, 128, 16, 128, 128, POW_TARGET_HARDEST, 102),
    ("g5_prod_rank128_found", 128, 256, 256, 128, 16, 128, 128, POW_TARGET_EASIEST, 102),
    ("g6_rect_n256_nofound", 128, 128, 256, 64, 16, 64, 64, POW_TARGET_HARDEST, 103),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"keys": {"key_A": KEY_A.hex(), "key_B": KEY_B.hex()}, "cases": []}

    for (name, m, k, n, rank, htile, mth, mtw, target, seed) in CASES:
        rng = np.random.default_rng(seed)
        A = rng.integers(-64, 64, size=(m, k), dtype=np.int8)
        B = rng.integers(-64, 64, size=(k, n), dtype=np.int8)
        gen = NoiseGenerator(noise_rank=rank, noise_range=128)
        E_AL, E_AR, E_BL, E_BR = gen.generate(KEY_A, KEY_B, m, k, n)
        gemm = NoisyGemm(noise_range=128, noise_rank=rank, hash_tile_h=htile, hash_tile_w=htile,
                         matmul_tile_h=mth, matmul_tile_w=mtw)
        C, found = gemm.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=target)

        # Hard invariant: the denoised result is always exactly A @ B.
        assert np.array_equal(C.astype(np.int64), A.astype(np.int64) @ B.astype(np.int64)), name

        np.savez_compressed(OUT / f"{name}.npz", A=A, B=B, E_AL=E_AL, E_AR=E_AR,
                            E_BL=E_BL, E_BR=E_BR, C=C)
        ob = gemm.opened_block
        manifest["cases"].append({
            "name": name, "file": f"{name}.npz",
            "m": m, "k": k, "n": n,
            "noise_rank": rank, "noise_range": 128,
            "hash_tile_h": htile, "hash_tile_w": htile,
            "matmul_tile_h": mth, "matmul_tile_w": mtw,
            "key_A": KEY_A.hex(), "key_B": KEY_B.hex(),
            "pow_target": hex(target),
            "found_block": found,
            "A_row_indices": ob.A_row_indices if found else None,
            "B_column_indices": ob.B_column_indices if found else None,
            "transcript_words": [hex(w) for w in ob.transcript_words] if found else None,
            "C_blake3": blake3(np.ascontiguousarray(C).tobytes()).hexdigest(),
            "C_shape": list(C.shape),
        })
        print(f"[golden] {name:28s} m={m} k={k} n={n} r={rank} found={found}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[golden] wrote {len(CASES)} cases + manifest.json to {OUT}")


if __name__ == "__main__":
    main()
