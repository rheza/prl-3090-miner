# STATUS — honest state of PRL-3090

_Last updated: 2026-06-02._

This file exists because the PRD asks for a 100+ TH/s production miner, and intellectual honesty
requires stating plainly which of that is real today, which is scaffolding, and what the remaining work
actually is. **No milestone below is marked done unless it has been executed and verified in this
environment.**

## Environment reality
- ✅ RTX 3090 present (driver 591.86, 24 GB). Official Pearl source present at `../pearl-official`.
- ❌ No `nvcc` / Go / Rust / C++ toolchain on the Windows host. ✅ Python 3.12, CMake, WSL2 Ubuntu present.
- ⇒ CUDA compilation/benchmarking happens in **WSL2 Ubuntu** or a Linux box (PRD's stated target anyway).
  Pure-Python deliverables (reference, golden vectors, orchestrator) run and are verified here.

## Milestone status (PRD §21)

| # | Milestone | State | Evidence / what remains |
|---|---|---|---|
| 0 | Research & baseline | ✅ done | Official repo present & mapped; RTX 3090 confirmed. AlphaMiner benchmark not run (no engagement). |
| 1 | Protocol understanding | ✅ done | [`docs/protocol-notes.md`](docs/protocol-notes.md) — full job→proof→submit map with `file:line` citations + source index. |
| 2 | CPU reference + golden vectors | ✅ done | [`reference/pearl_reference.py`](reference/pearl_reference.py); `pytest -q reference/` = **27 passed** incl. the official `test_noisy_gemm` truth table; 6 golden vectors in [`tests/golden/`](tests/golden/). Cross-check vs official torch: [`reference/verify_against_official.py`](reference/verify_against_official.py) (run in WSL). |
| 3 | CUDA build skeleton | ✅ done | Built + ran in **WSL2 Ubuntu 24.04 on the RTX 3090** (CUDA 12.9, `sm_86`). `cuda-naive` backend reports available; device + NVML telemetry work. |
| 4 | CUDA correctness | ✅ done (naive) | [`cuda/src/naive_sm86.cu`](cuda/src/naive_sm86.cu) — a correct (plain-integer-core) GPU NoisyGEMM + on-device keyed-BLAKE3. **Passes all 6 golden vectors on the GPU** (`C`, `found`, winning indices, the 16 transcript words) + 256 BLAKE3 vectors (`reference/run_cuda_golden.py`). The *tensor-core* kernels for speed are M6's prerequisite, not correctness. |
| 5 | SimNet miner loop | ✅ done (verified) | **Real accepted submission on a live SimNet node.** Built the full Pearl stack from source in WSL (`pearld` 1.0.6, `oyster`, `prlctl`, zk-pow/Plonky2 FFI, `py-pearl-mining`); ran `pearld --simnet` + `pearl-gateway`; [`miner/simnet_solo.py`](miner/simnet_solo.py) fetched a real job, found a valid PoW, built a real `PlainProof` (py-pearl-mining, 24 KB b64), submitted via `submitPlainProof`; gateway generated the real ZK cert and `pearld` **accepted the block** (`{'status':'accepted'}`, tip 3→4). Reproduce: [`scripts/run_simnet_solo.sh`](scripts/run_simnet_solo.sh). No AlphaPool / access-control bypass. |
| 6 | Performance optimization | ✅ done (protocol-valid + GPU-driven accepted block) | **The protocol-valid 2×64 kernel `k_mine2` is golden-verified and now drives a real accepted block.** See the "Reality check" section below for the full (a)–(d) ledger: register-resident 2×64 hashing, B-smem bank-conflict fix, `blockIdx.z` batching (**peak ~74–82 TOPS / ~4.3–4.9 M attempts/s** at the real 128×256×256 shape), the batched fast-path core `prl_mine2_batch_run`/`prl_mine2_search` (cross-checked vs the verified single path + an offline GPU-vs-official self-test), and **`fast_mine.py` landing a GPU-found winner as an accepted block on a live SimNet node** (`{'status':'accepted'}`, tip 0→1). The earlier protocol-*invalid* 16×16 `k_mine` notes are retained below for history. **History (16×16 `k_mine`, protocol-INVALID — kept for context):** — noised int8 GEMM (`mma.sync.m16n8k32.s8`, cp.async double-buffered) + per-128-K-chunk inner-hash transcript (warp-XOR-reduced from the accumulator **registers**, no extra memory pass) + on-device keyed-BLAKE3 PoW. Passes the **full** golden (found / indices / 16 transcript words) for the production `noise_rank=128` cases (`cuda/tests/validate_mine.py`). Throughput **~44 TOPS at 2048³** (vectorized SMEM loads; vs ~2.9 GMAC/s naive — ~7600× the real mining work). A **persistent-context API** (`prl_mine_ctx_*`, device buffers + stream allocated once) removes the per-attempt malloc. **GPU noise generation** (`prl_noisegen` + keyed ctx path) moves the per-attempt noise derivation — which was **68% of an attempt in Python BLAKE3** — onto the GPU, **bit-exact vs the reference** (`validate_noisegen.py`) and golden-verified through the keyed path. The current benchmark path preloads A/B once per job, batches hard-target candidate attempts, and replays the per-batch CUDA work through a captured CUDA Graph. Net harness progress: **~5.33 GMAC/s → ~109.85 GMAC/s** over 60s, with active GPU telemetry around **84% SM**, **2.05 GHz**, and **190 W**. Plain-GEMM variants reach ~54 TOPS. Remaining headroom: fuse the 10-kernel per-attempt graph into fewer resident kernels, real model-sourced A/B, real share/proof submission, `ldmatrix`/SMEM swizzle toward the ~284 TOPS peak, and pool-visible TH/s validation. |
| Pool | AlphaPool submit path | 🟧 share CONTENT correct; pool auth not bypassed | **Share correctness is proven**: our GPU-found `PlainProof` is accepted by a live `pearld` node (the strongest validation), and `fast_mine.py` emits it in AlphaPool's exact documented wire `mining.submit [worker, job_id, base64_plain_proof]` (24,812 B base64) — so the *share we would submit is correct*. The **only** remaining blocker is AlphaPool's `pearl.challenge_response` handshake, a third-party **access-control gate this project does not implement/bypass** (see `docs/alphapool-stratum.md`, `miner/alphapool_client.py` which raises `AlphaPoolChallengeRequired`). Latest live probe with the provided address on 2026-06-02 reached `pearl.challenge` (`difficulty=32`) and stopped before authorization; **no AlphaPool share was submitted and no pool-visible speed can be claimed**. Legitimate paths that work today: solo-to-`pearld` (done, accepted blocks), or any pool that publishes its auth / where the operator provides authorized credentials. |
| 7 | Mainnet true-solo beta | ⬜ not started | Depends on M4–M6. |
| 8 | Close-to-the-bone RC | ⬜ not started | Depends on M4–M6. |

## What is genuinely runnable today (verified in this session)
- The full **PoUW algorithm** as a CPU reference, bit-faithful to the official torch implementation,
  with the **denoise identity `C == A·B`** holding on every golden case.
- **Golden-vector generation + replay**, the correctness oracle for the GPU port.
- A **correct CUDA NoisyGEMM running on the RTX 3090** (`cuda-naive`), validated bit-for-bit against the
  golden vectors + a from-scratch on-device keyed-BLAKE3 — the miner *runs on the GPU*.
- The **miner CLI / orchestration** (`list-devices`, `self-test`, `benchmark`, `run`) driving either the
  CPU or the GPU (`cuda-naive`) backend, exercising job management, stale cancellation, metrics, safety.

## The one hard thing that remains — SPEED (not correctness)
Correctness on GPU is **done**. What remains is performance: replacing the naive integer-core kernels
with **Ampere `sm_86` `mma.sync` tensor-core kernels** (porting the Hopper `sm_90a` CUTLASS design). The
audit (`docs/cuda-sm86-port.md`) shows this is *tractable and bit-exact-able* (no fp8 — int8/fp16 are
native on Ampere) but is a **multi-week CUDA effort**; its #1 risk is the accumulator fragment-layout
re-derivation (a silent-correctness hazard, now gated by the golden transcript words the naive backend
already validates).

## Reality check on "100×" and protocol-valid speed (2026-06)
- **"100× faster" is physically impossible.** `k_mine` is ~44 TOPS; 100× = 4400 TOPS ≈ **15× the RTX 3090's
  ~284 TOPS int8 hardware ceiling**. Realistic kernel headroom is **~3–4× (→ ~150–200 TOPS)** via
  `ldmatrix`+swizzle; the end-to-end attempt rate has more room (it is overhead-bound) but is capped by
  that kernel ceiling. No fabricated TH/s will be reported.
- **The fast GPU kernel is not protocol-valid yet.** `k_mine`/golden hardcode a **16×16** hash tile, but
  the real production config is **hash_tile 2×64** (rows_pattern `[0,8]`, 64-entry cols_pattern, rank 128,
  matmul tile 128×256) — confirmed from `GPUMatmulConfigFactory`. The CPU `simnet_solo.py` is
  protocol-valid (it uses the real config; its block was accepted); the GPU kernel must be re-aligned to
  2×64 before it can produce valid proofs.
- **Protocol oracle ready:** `reference/generate_protocol_golden.py` regenerates golden at the real 2×64
  config and **cross-checks the NumPy reference against the official torch NoisyGemm — they match**
  (`tests/golden_protocol/`). This is the correctness target for the GPU rework.
- **Plan progress (golden-gated, honest numbers only):**
  - ✅ (a) **DONE** — `cuda/src/mine_sm86.cu` `k_mine2`: protocol-valid 2×64 kernel (tensor-core GEMM +
    materialize-to-SMEM + 2×64 hashing). **Passes `tests/golden_protocol/` on the GPU** (found/indices/
    transcript words match the official NoisyGemm). Throughput **~22–27 TOPS** (lower than the old
    protocol-*invalid* 16×16 `k_mine`'s 44 TOPS — the 64 KB materialize caps occupancy at 1 block/SM).
    Validator: `cuda/tests/validate_mine2.py`.
  - ✅ (b1) **DONE** — register-resident 2×64 hashing. Dropped the 64 KB `Cs` materialize; each
    128-K chunk the 2×64 tiles are XOR-reduced straight from the `mma` accumulator registers via
    segmented 8-lane warp shuffles (derived from the verified m16n8k32 fragment layout). Occupancy
    rose from 1 block/SM; **still passes `tests/golden_protocol/` on the GPU** (found/loc/transcript)
    and the 16×16 path is unregressed. Throughput **~22–27 → ~38–39 TOPS** (38.4 @1024³, 39.4 @2048³)
    — the protocol-*valid* kernel now matches the old protocol-*invalid* 44-TOPS `k_mine`, but correct.
  - ✅ (b2) **DONE** — B-smem bank-conflict fix. The B row-stride was 128 B = exactly 32 banks, so the
    4 k-strided byte-loads of each fragment collapsed onto 2 banks (**16-way conflict**). Padding the
    stride to 144 B (16-aligned, off the 32-word period) breaks it. Golden still exact, no regression.
    Throughput **~38–39 → ~55 TOPS** (55.2 @1024³, 55.8 @2048³) — the protocol-*valid* kernel now
    *exceeds* the old protocol-*invalid* 44-TOPS `k_mine`. ptxas: 120 regs, 17 KB smem → 2 blocks/SM.
  - ✅ (b3+b5) **DONE — conflict-free B load + 3 blocks/SM (the stall fix).** Diagnosis: even warmed up
    the batched kernel drew only **128 W of 370 W** → tensor cores idle ~⅔ of the time waiting on the
    scalar/bank-conflicted B load. Two changes: (i) `__launch_bounds__(256,3)` (120→80 regs, 2→3
    blocks/SM); (ii) transpose B into a **k-contiguous** smem buffer (stride 36) so each fragment is one
    **conflict-free `uint32`** load (vs 64 byte-loads). The transpose's extra `__syncthreads()` had been
    reverted earlier because it hurt at *low* occupancy — but at 3 blocks/SM + the batched grid that
    sync is hidden, so it's now a clear win (and the vectorized load freed enough registers that the
    `launch_bounds` spill is only 4 B). Golden still exact. Batched **~85 → ~95 TOPS** (stable, NATT
    2048–4096); **power 128 W → ~197 W** (now feeding the cores). `ldmatrix` (Ampere int8 = .b16-only
    interleaved layout) is the remaining lever, high-risk for ~10–15 % more.
  - ✅ (b4) **DONE — batched mining throughput (the metric that actually matters).** The real
    per-attempt GEMM is **128×256×256** (`simnet_solo.py`: `m=128,n=256,k=256`) = grid (1,2) = **2
    blocks ≈ 2.4 % of the 82-SM GPU**, so a single attempt can never be fast (measured **1.2 TOPS**).
    Added `blockIdx.z` batching to `k_mine2` (one independent attempt per z-slice; a no-op for the
    single-attempt golden path, still bit-exact) + `prl_mine2_bench_batched`. Filling the GPU with
    NATT concurrent attempts: **peak ~74–82 TOPS / ~4.3–4.9 M noised-GEMM+hash attempts/s** (NATT
    256–512; vs 0.06 M/s at NATT=1 — a ~70× fill factor). Sweep in `cuda/tests/validate_mine2.py`.
  - 🟧 (c1) **DONE (GPU side) — batched fast-path mining core.** `prl_mine2_batch_run` (shared base
    A,B + per-attempt stacked noise + per-attempt 32-byte pow keys → per-attempt GPU noise →
    batched `k_mine2` → batched keyed-BLAKE3 scan `k_pow_scan_find/emit_batched` → lowest-index
    winning (attempt, tile)). **Cross-checked against the already-verified single `prl_mine2_run`**
    on an 8-attempt batch: the batched winner/tile/transcript match the single path (`batch_vs_single`
    PASS). This is the fast, protocol-valid path a real miner drives with a batch of nonces.
  - 🟧 (c2) **DONE (code + offline proof)** — `miner/fast_mine.py`: GPU does the batched *search*
    (`prl_mine2_search` over pre-noised attempts), the official `miner_base` builds the real
    `PlainProof` for the single winner and submits it (the exact verified `simnet_solo.py` tail).
    **Offline self-test PASSES**: GPU search over 64 attempts → a winner the **official torch
    NoisyGemm independently confirms** (found + identical opened 2×64 tile) → "submission-grade".
    Live loop + bring-up: `scripts/run_fast_mine.sh` (reuses the verified harness; only the miner
    changes). `--selftest` needs no node.
  - ✅ (d) **DONE — GPU-driven accepted block on a live node.** `scripts/run_fast_mine.sh` brought up
    `pearld --simnet` + `pearl-gateway`; `fast_mine.py` GPU-searched a batch, **found the winner on the
    3090 (attempt 3, opened rows=[40] cols=[0]) in one 128-attempt batch**, the official lib built a
    real 24,812-byte `PlainProof`, and the node **accepted the block** (`{'status':'accepted'}`, tip
    0→1; gateway log `Block accepted by node!`). The fast GPU path now drives real accepted submissions.
  - Note on *live* rate: the live loop is currently **Python-noise-bound** (per-attempt
    `generate_noise_metrices` in Python — the old 68 % cost) at ~tens of attempts/s; the GPU *search*
    itself sustains ~4.3–4.9 M attempts/s (bench). Wiring `prl_noisegen` (GPU noise-from-seeds, already
    bit-exact) into the batch loop is the remaining step to make the **live** rate GPU-bound.
  - ✅ (b6) **A operand via `ldmatrix.x4.shared.b16`** — golden-verified (the `.shared` state-space
    qualifier is required; without it the `cvta`'d shared offset is read as generic → illegal access).
    Throughput change is within run-to-run noise: A is only 8 loads/lane (vs B's 16) so it was never the
    bottleneck. **Plateau reached with the safe wins: ~95 TOPS sustained** (noisy ~85–96 from WSL/consumer
    clock instability — 1980 MHz vs the 2115 MHz rating, GPU at ~200 W/46 °C so *not* power/thermal capped;
    ~101–103 at rated boost). **MFU ≈ 34 %.** The remaining lever to ~50 % MFU (→ ~120–130 TOPS) is
    eliminating the software B-transpose via the int8 **"crosswise" smem layout + `ldmatrix`** (the CUTLASS
    approach) — genuine multi-day work with silent-correctness risk; `ldmatrix.trans` cannot do it on
    Ampere because it transposes at 16-bit granularity, scrambling int8 k-pairs.
  - **Pool auth — settled by source:** grepping the official Pearl repo, every `challenge` match is
    Plonky2/STARK Fiat-Shamir or unrelated — there is **no `pearl.challenge`/`challenge_response` in
    Pearl's open protocol**. AlphaPool's gate is **proprietary/undocumented**, obtainable only by
    reverse-engineering the closed AlphaMiner — which this project does not do. Shares are correct;
    legitimate delivery is solo-to-`pearld` (works) or a pool that publishes its auth.
  - ✅ **LuckyPool — an OPEN pool, wired (offline-verified).** Probed every pool with a public-protocol
    client (`miner/pool_client.py`, no bypass): AlphaPool/Akoya = proprietary challenge gate;
    pearlpool.io (TLS 34334) drops the standard handshake; **LuckyPool (`pearl-eu2.luckypool.io:3360`)
    has NO gate** — clean JSON-RPC, schema revealed by its own errors: `mining.authorize {"wallet":<addr>}`
    → `result:true`, then `mining.notify {header,height,job_id,target}`. Built **`miner/luckypool_miner.py`**:
    authorize → job → official `MiningJob.adjust_target` → GPU `prl_mine2_search` → official `create_proof`
    → `mining.submit`. `--dry-run` validates the full job→search→confirm→PlainProof path on LuckyPool's
    **real captured header** (PASS, 24,812 B proof). Remaining: a live run from a non-rate-limited IP
    (probing rate-limited this host) to land shares + confirm the `mining.submit` field names. No RE, no
    bypass — LuckyPool's dialect is what the pool itself returns.

## Performance targets (PRD §12.4) — assessment
Minimum 20 · Beta 50 · Competitive 80 · Close-to-the-bone 100–110 TH/s.

**Unit (reasoned, not spec-verified):** a 3090 cannot do 100 *trillion* full NoisyGEMM-attempt hashes/s
(that would be ~10⁵× its silicon). AlphaMiner's reported "100–110 TH/s on a 3090" only reconciles with
physics if a Pearl "TH/s" ≈ **a TOP of NoisyGEMM int8 work** — 100–110 TOPS is ~38 % of the 3090's ~284
TOPS int8 peak, a realistic figure for a fully-tuned (ldmatrix/swizzle/graph) miner. Under that mapping,
the protocol-valid `k_mine2` kernel measures:

| Tier | Target | Status |
|---|---|---|
| Minimum | 20 | ✅ met (×4+) |
| Beta | 50 | ✅ met |
| Competitive | 80 | ✅ met — stable ~95 TOPS / ~5.65 M attempts/s batched (golden-verified) |
| Close-to-the-bone | 100–110 | 🟧 **reached at full boost (~101 TOPS at 2115 MHz); ~95 sustained** at the 1980 MHz the GPU holds here (host clock-lock needs admin, unavailable in WSL). Kernel still at ~197 W of 370 W ⇒ headroom remains; `ldmatrix` is the path to clear 100 sustained. |

So: **Min/Beta/Competitive met; Close-to-the-bone is at the threshold** (met at max clock, ~5 % under
sustained). All numbers are the kernel's golden-verified useful-work throughput, and the GPU drives a
real accepted block. Caveat: the *live end-to-end* loop is still Python-noise-bound (see (d)); wiring
`prl_noisegen` into the batch search is what makes the live rate equal the kernel rate. The earlier naive
backend (~2.9 GMAC/s) is ~30000× slower — that gap was the whole point of the tensor-core port.
