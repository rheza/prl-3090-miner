"""Real SimNet solo miner — produces a VALID PlainProof and submits it through the live
pearl-gateway, using the official miner-base machinery (NoisyGEMM + commitment + Merkle
proof via py-pearl-mining). Linux/WSL only (needs the built pearl stack venv).

This is the legitimate submit path: pearld --simnet  <-  pearl-gateway (8337)  <-  this.
A/B are synthetic (SimNet does not verify the useful-work semantics — the node runs with
BFNoPoWCheck), but the PlainProof, commitment, and PoW transcript are all real and
internally consistent, so the gateway can assemble a real block and submit it.

Run inside the pearl-official venv with the gateway env vars set (MINER_RPC_TRANSPORT=tcp,
MINER_RPC_PORT=8337). See scripts/run_simnet.sh.
"""
from __future__ import annotations

import sys
import time

import torch
from miner_base.block_submission import create_proof
from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from miner_base.gateway_client import MiningClient
from miner_base.noise_generation import NoiseGenerator
from miner_base.noisy_gemm import NoisyGemm
from pearl_gateway.config import MinerRpcConfig


def main() -> int:
    k, rank = 256, 128
    mc = GPUMatmulConfigFactory.create(k, rank)
    mcfg = mc.mining_config
    m, n = mc.matmul_tile_h, mc.matmul_tile_w           # 128 x 256 (production tile)
    print(f"[miner] config: m={m} n={n} k={k} rank={rank} "
          f"hash_tile={mc.hash_tile_h}x{mc.hash_tile_w}")

    client = MiningClient(MinerRpcConfig())              # reads MINER_RPC_* env (tcp:8337)
    job = client.get_mining_info()
    adj_target = job.adjust_target(mcfg)
    print(f"[miner] job: header={len(job.incomplete_header_bytes)}B raw_target={hex(job.target)}")
    print(f"[miner] adjusted_target={hex(adj_target)}")

    gemm = NoisyGemm(noise_range=128, noise_rank=rank,
                     hash_tile_h=mc.hash_tile_h, hash_tile_w=mc.hash_tile_w,
                     matmul_tile_h=m, matmul_tile_w=n)
    ng = NoiseGenerator(noise_rank=rank, noise_range=128)

    found = False
    t0 = time.time()
    for attempt in range(500):
        A = torch.randint(-64, 64, (m, k), dtype=torch.int8)   # synthetic (SimNet)
        B = torch.randint(-64, 64, (k, n), dtype=torch.int8)
        commitment = CommitmentHasher.commitment_hash(A, B, job.incomplete_header_bytes, mcfg)
        E_AL, E_AR, E_BL, E_BR = ng.generate_noise_metrices(
            commitment.noise_seed_A, commitment.noise_seed_B, m, k, n)
        _, found = gemm.noisy_gemm(A, B, E_AL, E_AR, E_BL, E_BR,
                                   commitment_hash=commitment, pow_target=adj_target)
        if found:
            print(f"[miner] FOUND proof on attempt {attempt} in {time.time()-t0:.2f}s")
            break
    if not found:
        print("[miner] no proof found (target too hard for this harness)"); return 1

    ob = gemm.get_opened_block_info()
    print(f"[miner] opened tile rows={ob.A_row_indices[:1]}.. cols={ob.B_column_indices[:1]}..")
    proof = create_proof(ob, job.incomplete_header_bytes)
    print(f"[miner] built PlainProof: base64 len={len(proof.to_base64())}")
    client.submit_plain_proof(proof, job)
    print("[miner] submitPlainProof sent (gateway acks 'submitted'; block-build is async)")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
