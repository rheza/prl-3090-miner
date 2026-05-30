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
| 3 | CUDA build skeleton | 🟧 scaffolding | [`cuda/`](cuda/) has CMake (`sm_86`), device manager, NVML monitor, kernel decls, golden-loading test harness. **Compiles in WSL once CUDA Toolkit is installed; not yet built here (no nvcc on Windows).** |
| 4 | CUDA correctness | ⬜ not started | The real work: port 4 Hopper kernels to Ampere `mma.sync` (see [`docs/cuda-sm86-port.md`](docs/cuda-sm86-port.md)). Validated against `tests/golden/`. |
| 5 | SimNet miner loop | 🟧 partial | Orchestration loop (job fetch → backend → submit → stale-cancel → metrics) implemented in [`miner/`](miner/) and self-tested with the **CPU backend** against a mock gateway. Needs a live SimNet node + the CUDA backend for the real loop. |
| 6 | Performance optimization | ⬜ not started | Depends on M4. Plan + tooling in [`docs/cuda-sm86-port.md`](docs/cuda-sm86-port.md) §6–7. |
| 7 | Mainnet true-solo beta | ⬜ not started | Depends on M4–M6. |
| 8 | Close-to-the-bone RC | ⬜ not started | Depends on M4–M6. |

## What is genuinely runnable today (verified in this session)
- The full **PoUW algorithm** as a CPU reference, bit-faithful to the official torch implementation,
  with the **denoise identity `C == A·B`** holding on every golden case.
- **Golden-vector generation + replay**, the correctness oracle for the GPU port.
- The **miner CLI / orchestration** (`list-devices`, `self-test`, `benchmark`, `run`) with a CPU backend
  and a mock gateway, exercising job management, stale cancellation, metrics, and safety throttling.

## The one hard thing that remains
Porting the **Hopper `sm_90a` CUTLASS kernels** (main GEMM, NoisingA/B, tensor-hash leaf) to **Ampere
`sm_86` `mma.sync`**. The audit (`docs/cuda-sm86-port.md`) shows this is *tractable and bit-exact-able*
(no fp8 — int8/fp16 paths are native on Ampere) but is a **multi-week CUDA effort**, and its #1 risk is
the accumulator fragment-layout re-derivation (a silent-correctness hazard, gated by the golden
transcript words). The first V1 release will be **much slower than AlphaMiner** and that is expected and
acceptable — correctness before speed (PRD §12.3–12.4).

## Performance targets (PRD §12.4) — none achieved yet; correctness gates them
Minimum 20 TH/s · Beta 50 TH/s · Competitive 80 TH/s · Close-to-the-bone 100–110 TH/s. Current measured
GPU hashrate: **0** (no GPU kernel built). Do not report a number until M4 passes golden.
