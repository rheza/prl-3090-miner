"""
Cross-check the standalone NumPy reference against the OFFICIAL torch reference.

Run this where the official Pearl miner-base is importable (e.g. inside WSL after
`task build:miner`, or `uv pip install -e pearl-official/miner/miner-base`). It feeds
the *identical* inputs from each golden vector into the official `NoisyGemm` /
`NoiseGenerator` and asserts bit-for-bit agreement with our golden outputs.

This closes the correctness loop: if this passes, the golden vectors in tests/golden/
are exactly what the official reference produces, so a CUDA backend that matches the
golden vectors matches the official protocol.

    python reference/verify_against_official.py

Exits non-zero on any mismatch. Requires: torch, and the official miner_base +
pearl_gateway packages on PYTHONPATH.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "tests" / "golden"


def main() -> int:
    try:
        import torch
        from miner_base.noisy_gemm import NoisyGemm as OfficialNoisyGemm
        from pearl_gateway.comm.dataclasses import CommitmentHash
    except Exception as exc:  # pragma: no cover - depends on external install
        print(f"[verify] official miner-base/torch not importable: {exc}")
        print("[verify] run inside the Pearl dev env (WSL): see docs/setup-ubuntu.md")
        return 2

    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    failures = 0
    for case in manifest["cases"]:
        d = np.load(GOLDEN / case["file"])
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        gemm = OfficialNoisyGemm(
            noise_range=case["noise_range"], noise_rank=case["noise_rank"],
            hash_tile_h=case["hash_tile_h"], hash_tile_w=case["hash_tile_w"],
            matmul_tile_h=case["matmul_tile_h"], matmul_tile_w=case["matmul_tile_w"],
        )
        commitment = CommitmentHash(
            noise_seed_A=bytes.fromhex(case["key_A"]),
            noise_seed_B=bytes.fromhex(case["key_B"]),
        )
        C, found = gemm.noisy_gemm(
            to_t(d["A"]), to_t(d["B"]), to_t(d["E_AL"]), to_t(d["E_AR"]),
            to_t(d["E_BL"]), to_t(d["E_BR"]),
            commitment_hash=commitment, pow_target=int(case["pow_target"], 16),
        )
        C_np = C.numpy()
        ok_C = np.array_equal(C_np, d["C"])
        ok_found = bool(found) == case["found_block"]
        status = "OK" if (ok_C and ok_found) else "MISMATCH"
        if not (ok_C and ok_found):
            failures += 1
        print(f"[verify] {case['name']:28s} C={'=' if ok_C else 'X'} "
              f"found(official={bool(found)} golden={case['found_block']}) -> {status}")

    print(f"[verify] {'ALL MATCH' if failures == 0 else f'{failures} MISMATCH(es)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
