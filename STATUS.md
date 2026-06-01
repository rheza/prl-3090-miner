# STATUS — honest state of PRL-3090

_Last updated: 2026-05-30._

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
| 5 | SimNet miner loop | 🟧 partial | Orchestration loop (job fetch → backend → submit → stale-cancel → metrics) runs through the **real GPU** via `cuda-naive` (`miner benchmark --backend cuda-naive`) and self-tests vs a mock gateway. Needs a live SimNet node for the full chain loop. |
| 6 | Performance optimization | 🟧 in progress | **Tensor cores proven**: [`cuda/src/mma_gemm_sm86.cu`](cuda/src/mma_gemm_sm86.cu) — a correct Ampere `mma.sync.m16n8k32.s8` int8 GEMM, validated bit-for-bit vs golden `C` on the GPU (`cuda/tests/validate_mma.py`). Unoptimized (one warp/tile, no SMEM reuse) it does **~6 TOPS** (vs ~284 TOPS GA102 int8 peak). Remaining: SMEM tiling + `cp.async` pipeline, then fuse the noised+transcript path onto tensor cores, then Nsight tuning. Targets ≥50 TH/s. |
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

## Performance targets (PRD §12.4) — correctness done; speed is the open work
Minimum 20 TH/s · Beta 50 TH/s · Competitive 80 TH/s · Close-to-the-bone 100–110 TH/s. The naive
(correct, non-tensor-core) backend measures **~2.9 GMAC/s** on the production shape — ~4–5 orders of
magnitude below the AlphaMiner class. That gap is the whole point of the tensor-core port; do not report
a protocol TH/s number until the `mma.sync` kernels land and pass golden against a live node.
