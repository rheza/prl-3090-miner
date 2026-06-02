"""Generate PROTOCOL-ACCURATE golden vectors at the real production config
(hash_tile 2x64, noise_rank 128, matmul tile 128x256), and cross-check the NumPy
reference against the OFFICIAL torch NoisyGemm so the GPU-kernel rework has a
protocol-valid oracle. Run in the pearl-official venv (torch + miner_base).

    .venv/bin/python reference/generate_protocol_golden.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
import pearl_reference as ref  # noqa: E402

import torch  # noqa: E402
from blake3 import blake3  # noqa: E402
from miner_base.noisy_gemm import POW_TARGET_EASIEST, POW_TARGET_HARDEST  # noqa: E402
from miner_base.noisy_gemm import NoisyGemm as OfficialNoisyGemm  # noqa: E402
from pearl_gateway.comm.dataclasses import CommitmentHash  # noqa: E402

KEY_A = b"test_key_A_123456789012345678901"
KEY_B = b"test_key_B_123456789012345678901"
OUT = ROOT / "tests" / "golden_protocol"
HTH, HTW = 2, 64          # REAL protocol hash tile
M, K, N, R = 128, 256, 256, 128
TILE_H, TILE_W = 128, 256


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    ok = True
    for name, target in [("p_nofound", POW_TARGET_HARDEST), ("p_found", POW_TARGET_EASIEST)]:
        rng = np.random.default_rng(7 if "found" in name else 9)
        A = rng.integers(-64, 64, (M, K), dtype=np.int8)
        B = rng.integers(-64, 64, (K, N), dtype=np.int8)
        E_AL, E_AR, E_BL, E_BR = ref.NoiseGenerator(noise_rank=R).generate(KEY_A, KEY_B, M, K, N)

        # NumPy reference at the REAL 2x64 hash tile
        g = ref.NoisyGemm(noise_range=128, noise_rank=R, hash_tile_h=HTH, hash_tile_w=HTW,
                          matmul_tile_h=TILE_H, matmul_tile_w=TILE_W)
        C, found = g.run(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=KEY_A, pow_target=target)

        # OFFICIAL torch NoisyGemm, same inputs + 2x64 tile
        og = OfficialNoisyGemm(noise_range=128, noise_rank=R, hash_tile_h=HTH, hash_tile_w=HTW,
                               matmul_tile_h=TILE_H, matmul_tile_w=TILE_W)
        Coff, foff = og.noisy_gemm(t(A), t(B), t(E_AL), t(E_AR), t(E_BL), t(E_BR),
                                   commitment_hash=CommitmentHash(KEY_A, KEY_B), pow_target=target)
        Coff = Coff.numpy()

        c_ok = np.array_equal(C, Coff)
        f_ok = bool(found) == bool(foff)
        loc_ok = True
        if found:
            ob_off = og.get_opened_block_info()
            loc_ok = (g.opened_block.A_row_indices[0] == ob_off.A_row_indices[0]
                      and g.opened_block.B_column_indices[0] == ob_off.B_column_indices[0])
        good = c_ok and f_ok and loc_ok
        ok = ok and good
        print(f"[{name}] ref-vs-official @2x64: C={'=' if c_ok else 'X'} "
              f"found={'=' if f_ok else 'X'}({bool(found)}) loc={'=' if loc_ok else 'X'}")

        np.savez_compressed(OUT / f"{name}.npz", A=A, B=B, E_AL=E_AL, E_AR=E_AR,
                            E_BL=E_BL, E_BR=E_BR, C=C)
        ob = g.opened_block
        cases.append({
            "name": name, "file": f"{name}.npz", "m": M, "k": K, "n": N,
            "noise_rank": R, "noise_range": 128, "hash_tile_h": HTH, "hash_tile_w": HTW,
            "matmul_tile_h": TILE_H, "matmul_tile_w": TILE_W,
            "key_A": KEY_A.hex(), "key_B": KEY_B.hex(), "pow_target": hex(target),
            "found_block": bool(found),
            "A_row_indices": ob.A_row_indices if found else None,
            "B_column_indices": ob.B_column_indices if found else None,
            "transcript_words": [hex(w) for w in ob.transcript_words] if found else None,
            "C_blake3": blake3(np.ascontiguousarray(C).tobytes()).hexdigest(),
        })
    (OUT / "manifest.json").write_text(json.dumps({"cases": cases}, indent=2))
    print(f"RESULT: {'REFERENCE MATCHES OFFICIAL @2x64 — protocol oracle ready' if ok else 'MISMATCH'}")
    print(f"wrote {len(cases)} protocol-golden cases to {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
