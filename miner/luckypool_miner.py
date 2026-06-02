"""GPU miner for LuckyPool (open Stratum dialect — no challenge gate, nothing reverse-engineered).

Probed protocol (LuckyPool returns this itself over plain TCP):
  ->  mining.authorize {"wallet": <prl1..>, "worker": w, "agent": ...}   => {"result": true}
  <-  mining.notify    {"header": <hex>, "height": h, "job_id": id, "target": <hex>}
  ->  mining.submit    {"wallet","worker","job_id","proof": <base64 PlainProof>}

Reuses our golden-verified GPU search (`prl_mine2_search`) + the official `miner_base` PlainProof
(`create_proof`) + the official `MiningJob.adjust_target` (target * hash_tile_h*hash_tile_w*
rounded_common_dim). We only implement the public dialect the pool itself speaks.

Modes:
  --dry-run   offline: use a captured LuckyPool job header at an EASY target, run one GPU batch,
              confirm the winner via the official NoisyGemm and build a real PlainProof from
              LuckyPool's actual header. Proves the job->search->proof path end to end, no network.
  (default)   live: connect, authorize, mine each job, submit shares. Run from a non-rate-limited
              IP inside the pearl-official venv:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<repo> python miner/luckypool_miner.py \
        --address prl1p... --worker rig0 --natt 256

NOTE: the mining.submit field names ({wallet,worker,job_id,proof}) are our best read of the open
dialect; the first live submit prints the pool's response so any single field rename is obvious.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "miner"))
from pool_client import StratumClient                       # noqa: E402
from fast_mine import load_lib, noise_int8, as_i8, gpu_search  # noqa: E402

# A real job captured from pearl-eu2.luckypool.io:3360 (height 66399) for the offline dry-run.
CAPTURED_HEADER = ("000040202c4673fc968122024ca557455d3c1c4ac7fae10cff86292720c7d5883eb9db5b"
                   "596468669a69aa87dcdf6ff411812626a5a7c6cf378ccf9c3639136b600178dcefcd1e6ae7640118")


def _imports():
    import torch
    from miner_base.block_submission import create_proof
    from miner_base.commitment_hash import CommitmentHasher
    from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
    from miner_base.noise_generation import NoiseGenerator
    from miner_base.noisy_gemm import NoisyGemm
    from pearl_gateway.comm.dataclasses import MiningJob
    return (torch, create_proof, CommitmentHasher, GPUMatmulConfigFactory,
            NoiseGenerator, NoisyGemm, MiningJob)


def build_batch(np_rng, ng, CommitmentHasher, tt, header, mcfg, m, k, n, natt):
    """natt random attempts on `header`: returns (An_stack, Bn_stack, keys_bytes, attempts)."""
    An_l, Bn_l, keys, atts = [], [], b"", []
    for _ in range(natt):
        A = np_rng.integers(-64, 64, (m, k), dtype=np.int8)
        B = np_rng.integers(-64, 64, (k, n), dtype=np.int8)
        commitment = CommitmentHasher.commitment_hash(tt(A), tt(B), header, mcfg)
        sA, sB = commitment.noise_seed_A, commitment.noise_seed_B
        EAL, EAR, EBL, EBR = ng.generate_noise_metrices(sA, sB, m, k, n)
        EAL, EAR, EBL, EBR = as_i8(EAL), as_i8(EAR), as_i8(EBL), as_i8(EBR)
        An_l.append(noise_int8(A, EAL, EAR))
        Bn_l.append(noise_int8(B, EBL, EBR))
        keys += bytes(sA)[:32].ljust(32, b"\0")
        atts.append((A, B, EAL, EAR, EBL, EBR, commitment))
    return np.stack(An_l), np.stack(Bn_l), keys, atts


def dry_run(natt: int) -> int:
    (torch, create_proof, CommitmentHasher, GPUMatmulConfigFactory,
     NoiseGenerator, NoisyGemm, MiningJob) = _imports()
    from miner_base.noisy_gemm import POW_TARGET_EASIEST

    lib = load_lib()
    k, R = 256, 128
    mc = GPUMatmulConfigFactory.create(k, R); mcfg = mc.mining_config
    m, n = mc.matmul_tile_h, mc.matmul_tile_w
    ng = NoiseGenerator(noise_rank=R, noise_range=128)
    gemm = NoisyGemm(noise_range=128, noise_rank=R, hash_tile_h=mc.hash_tile_h,
                     hash_tile_w=mc.hash_tile_w, matmul_tile_h=m, matmul_tile_w=n)
    def tt(x): return torch.from_numpy(np.ascontiguousarray(x))

    header = bytes.fromhex(CAPTURED_HEADER)
    # MiningJob.adjust_target on the REAL captured target (sanity: it scales by the tile factor).
    real_target = 0x000000000000218dcdb37c99ae924f227d028a1dfb9389b52007dd441355475a
    adj_real = MiningJob(incomplete_header_bytes=header, target=real_target).adjust_target(mcfg)
    print(f"[dry] LuckyPool header={len(header)}B  raw_target={hex(real_target)}")
    print(f"[dry] adjust_target(mcfg) factor=hth*htw*rcd -> adj={hex(adj_real)} (this is what the GPU PoW uses)")

    # Use an EASY target so a winner exists in one batch — validates the full path on the real header.
    tle = int(POW_TARGET_EASIEST).to_bytes(32, "little")
    rng = np.random.default_rng(123)
    An, Bn, keys, atts = build_batch(rng, ng, CommitmentHasher, tt, header, mcfg, m, k, n, natt)
    fa, arow, bcol, _ = gpu_search(lib, An, Bn, keys, tle, m, k, n, natt)
    print(f"[dry] GPU search over {natt} attempts on LuckyPool's header -> winner_attempt={fa} loc=({arow},{bcol})")
    if fa < 0:
        print("[dry] FAIL: no winner at easy target"); return 1
    A, B, EAL, EAR, EBL, EBR, commitment = atts[fa]
    _, found = gemm.noisy_gemm(tt(A), tt(B), tt(EAL), tt(EAR), tt(EBL), tt(EBR),
                               commitment_hash=commitment, pow_target=int(POW_TARGET_EASIEST))
    if not found:
        print("[dry] FAIL: official NoisyGemm did not confirm the GPU winner"); return 1
    ob = gemm.get_opened_block_info()
    proof = create_proof(ob, header)
    b64 = proof.to_base64()
    submit = {"wallet": "prl1p...", "worker": "rig0", "job_id": "57c67909_500000", "proof": b64}
    print(f"[dry] official NoisyGemm confirmed winner; create_proof OK on LuckyPool's header -> {len(b64)}B")
    print(f"[dry] mining.submit would be: {{\"method\":\"mining.submit\",\"params\":"
          f"{{wallet,worker,job_id,proof[{len(b64)}B b64]}}}}")
    print("[dry] RESULT: PASS — LuckyPool job -> GPU search -> official PlainProof path works end to end.")
    return 0


def live(host, port, tls, address, worker, natt, minutes) -> int:
    (torch, create_proof, CommitmentHasher, GPUMatmulConfigFactory,
     NoiseGenerator, NoisyGemm, MiningJob) = _imports()

    lib = load_lib()
    k, R = 256, 128
    mc = GPUMatmulConfigFactory.create(k, R); mcfg = mc.mining_config
    m, n = mc.matmul_tile_h, mc.matmul_tile_w
    ng = NoiseGenerator(noise_rank=R, noise_range=128)
    gemm = NoisyGemm(noise_range=128, noise_rank=R, hash_tile_h=mc.hash_tile_h,
                     hash_tile_w=mc.hash_tile_w, matmul_tile_h=m, matmul_tile_w=n)
    def tt(x): return torch.from_numpy(np.ascontiguousarray(x))
    rng = np.random.default_rng()

    c = StratumClient(host, port, tls)
    print(f"[luckypool] connecting {host}:{port} tls={tls}")
    c.connect()
    c.send("mining.authorize", {"wallet": address, "worker": worker, "agent": "prl-3090-miner/1.0"})
    cur = {"header": None, "tle": None, "job_id": None, "adj": None}

    def apply_job(p):
        hdr = bytes.fromhex(p["header"])
        raw = int(p["target"], 16) if isinstance(p["target"], str) else int(p["target"])
        adj = MiningJob(incomplete_header_bytes=hdr, target=raw).adjust_target(mcfg)
        cur.update(header=hdr, tle=int(adj).to_bytes(32, "little"), job_id=p["job_id"], adj=adj)
        print(f"[luckypool] job {p['job_id']} h={p.get('height')} raw={hex(raw)} adj={hex(adj)}")

    def pump(window):
        for msg in c.messages(time.time() + window):
            if "_closed" in msg or "_error" in msg:
                print(f"[luckypool] link: {msg}"); return "closed"
            meth = msg.get("method", "")
            if meth in ("mining.notify", "job"):
                apply_job(msg["params"])
            elif msg.get("id") == 1:
                print(f"[luckypool] authorize -> {json.dumps(msg)[:120]}")
            elif meth:
                print(f"[luckypool] <- {meth} {json.dumps(msg.get('params'))[:100]}")
        return None

    if pump(8.0) == "closed":
        return 1
    t0 = time.time(); deadline = t0 + minutes * 60
    tried = 0; submitted = 0; accepted = 0

    def report(tag):
        el = max(1e-9, time.time() - t0)
        aps = tried / el
        ths = aps * (m * k * n * 2) / 1e12            # "TH/s" ~= achieved int8 TOPS (attempt = m*k*n MAC)
        # Calibrated to a real datapoint: RTX 5090 @ ~300 TH/s ~= $17.19/day (Tom's Hardware, 2026),
        # LuckyPool fee 1%. Earnings track hashrate linearly; PRL price moves, so this is a rough estimate.
        usd_day = (ths / 300.0) * 17.19 * 0.99
        usd = usd_day * (el / 86400.0)
        print(f"[report:{tag}] elapsed={el/60:.1f}min attempts={tried} rate={aps:,.0f}/s (~{ths:.1f} TH/s) "
              f"| shares submitted={submitted} accepted={accepted} "
              f"| est. earnings so far ~${usd:.4f} (~${usd_day:.2f}/day @ this rate)")

    last_report = t0
    while time.time() < deadline:
        if pump(0.02) == "closed":
            report("final"); return 1
        if cur["header"] is None:
            time.sleep(0.2); continue
        job_id, header, tle = cur["job_id"], cur["header"], cur["tle"]
        An, Bn, keys, atts = build_batch(rng, ng, CommitmentHasher, tt, header, mcfg, m, k, n, natt)
        tried += natt
        fa, arow, bcol, _ = gpu_search(lib, An, Bn, keys, tle, m, k, n, natt)
        if time.time() - last_report > 60:
            report("tick"); last_report = time.time()
        if job_id != cur["job_id"]:            # job rolled over mid-batch -> stale, drop it
            continue
        if fa < 0:
            continue
        A, B, EAL, EAR, EBL, EBR, commitment = atts[fa]
        _, found = gemm.noisy_gemm(tt(A), tt(B), tt(EAL), tt(EAR), tt(EBL), tt(EBR),
                                   commitment_hash=commitment, pow_target=cur["adj"])
        if not found:
            print("[luckypool] GPU winner not confirmed (target scaling?) — skipping"); continue
        ob = gemm.get_opened_block_info()
        b64 = create_proof(ob, header).to_base64()
        submitted += 1
        print(f"[luckypool] SHARE job={job_id} after {tried} attempts; proof {len(b64)}B -> mining.submit")
        sid = c.send("mining.submit", {"wallet": address, "worker": worker, "job_id": job_id, "proof": b64})
        for msg in c.messages(time.time() + 6):
            print(f"[luckypool] submit <- {json.dumps(msg)[:220]}")
            blob = json.dumps(msg).lower()
            if msg.get("result") is True or "accept" in blob:
                accepted += 1
            if msg.get("id") == sid or "result" in msg or "error" in msg:
                break
    c.close()
    report("final")
    print(f"[luckypool] === 12h-style summary === attempts={tried} submitted={submitted} accepted={accepted}")
    print("[luckypool] (authoritative shares/earnings: your LuckyPool dashboard for this address)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="pearl-eu2.luckypool.io")
    ap.add_argument("--port", type=int, default=3360)
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--address", default="prl1p56vxsrzefhx8m49c2y4v86up845z6aurrxd3wulep9dk38cz8hcssu8cnm")
    ap.add_argument("--worker", default="rig0")
    ap.add_argument("--natt", type=int, default=256)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true", help="offline path validation (no network)")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run(min(a.natt, 256))
    return live(a.host, a.port, a.tls, a.address, a.worker, a.natt, a.minutes)


if __name__ == "__main__":
    sys.exit(main())
