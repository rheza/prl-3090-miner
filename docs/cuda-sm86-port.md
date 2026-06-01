# CUDA `sm_86` Port Plan — Hopper → Ampere (RTX 3090)

> Grounded in a file-by-file audit of `../pearl-official/miner/pearl-gemm/csrc`. Citations are
> `file:line` within that tree. This is the **core, irreducibly-hard** part of the whole project.
> Everything else (orchestration, config, metrics) is glue around it.

## TL;DR

The official GPU miner is a **CUTLASS 3.x warp-specialized Hopper kernel** built **only** for
`sm_90a` (`setup.py:89` → `arch=compute_90a,code=sm_90a`; hard `static_assert(kMinComputeCapability>=90)`
at `merkle_tree_roots_kernel.hpp:56`; `using ArchTag = cutlass::arch::Sm90` at
`pearl_noisingA_kernel.h:43`). RTX 3090 is Ampere `sm_86`, which **lacks** the four pillars that
kernel relies on: **TMA**, **wgmma**, **warp-specialization with `setmaxnreg`**, and **thread-block
clusters**.

The good news, established by the audit:
- **No fp8 anywhere.** The math is **int8→int32** (main GEMM, `pearl_gemm_host.h:24`) and **fp16→fp32**
  (denoise). Both have **native Ampere tensor-core support**. So there is **no numeric-emulation detour** —
  a correct Ampere port can be **bit-exact** with the CPU reference and the official kernel.
- Roughly half the kernels are **already portable** (pure integer / cp.async). Only **3 tensor-core
  kernels + 1 tensor-hash leaf kernel** must be rewritten.

The single biggest risk is **layout, not math** (see §5).

---

## 1. Kernel inventory and portability verdict

| Kernel / entry point | Source | Uses (Hopper) | Verdict |
|---|---|---|---|
| `hopper_gemm_ws` (main fused GEMM + PoW transcript) | `pearl_gemm_kernel.h:34`, `collective_mainloop.hpp`, `collective_epilogue.hpp` | TMA, wgmma, warp-spec, clusters, named barriers, stmatrix | **REWRITE** |
| `NoisingKernelA` (A+EA, A·EBL) | `pearl_noisingA_kernel.h:27` | `Sm90`, wgmma, TMA incl. `SM90_TMA_REDUCE_ADD`, 3-WG warp-spec | **REWRITE** |
| `NoisingKernelB` (B+EB, EAR·BpEB) | `pearl_noisingB_kernel.h:26` | mirror of A | **REWRITE** |
| `MerkleTreeRootsKernel` (tensor-hash leaf) | `merkle_tree_roots_kernel.hpp:54-56` | `static_assert(cc>=90)`, TMA, PipelineTmaAsync, GMMA swizzles | **REWRITE** |
| `NoiseGenerationKernel` (EAL/EAR/EBL/EBR) | `noise_generation_kernel.h:88` | plain CUDA + blake3, `__syncthreads` | **PORTABLE** |
| `DenoiseConverterKernel` (i32→fp16 + scale) | `denoise_converter_kernel.h:22` | 128-bit vectorized copy | **PORTABLE** |
| `dynamic_scaled_quant_kernel` | `quantize_kernel.cu:59` | cub BlockReduce + inline PTX `cvt.rni.sat.s8.f32` | **PORTABLE** |
| `inner_hash_kernel` | `inner_hash_kernel.cu:11` | single-thread XOR | **PORTABLE** |
| `ComputeBlakeMTKernel`, `ReduceRootsKernel`, `CommitmentHashFromMerkleRootsKernel` | `tensor_hash/*.hpp` | integer blake3 + smem reduce | **PORTABLE (compute)** |
| `blake3.cu` / `blake3.cuh` | `blake3/` | integer ALU only | **PORTABLE** |

So: **4 must-rewrite, ~7 portable**. The portable set still needs the build retargeted (§2) and the
tensor-hash path needs its *leaf* kernel rewritten while its blake3 math is reused.

---

## 2. Build retarget (Milestone 3)

In a **fork of the pearl-gemm build** (we do not modify `../pearl-official`; we vendor or submodule it):
- `setup.py:89`: `COMPUTE_CAPABILITY = "arch=compute_86,code=sm_86"` (optionally also `compute_80` for
  fat-binary portability to other Ampere).
- `setup.py:459`: relax `MIN_CUDA_VERSION` (currently `13.0`). sm_86 + the int8/fp16 mma paths work from
  **CUDA 11.1+**; target **CUDA 12.x** (the toolchain shipped on Ubuntu 22.04/24.04, PRD §7.2).
- `pyproject.toml:16`: `torch==2.11.0` is not required by sm_86; pin to a CUDA-12 torch build.
- Replace the `cutlass::arch::Sm90` tags and the `static_assert(cc>=90)` with `Sm80` and `cc>=80`
  in the kernels being rewritten.
- `libraries=["cuda"]` (`setup.py:448`) was for `cuTensorMapEncodeTiled` (TMA descriptors) — drops out
  once TMA is removed.

**SMEM budget is a hard constraint.** Default GEMM tile is `128×256×128` int8 with 3 stages
(`default_compiled_kernels.py`). Ampere `sm_86` opt-in SMEM is **99 KB/SM** vs Hopper's 227 KB. The
existing heuristics already query `cudaDevAttrMaxSharedMemoryPerBlockOptin`
(`heuristics.hpp:30`, `pearl_gemm_host.h:97`), so stages auto-shrink — but `128×256×128` @ 3 stages will
**not** fit. Plan to **retune tiles smaller** (e.g. `128×128×64`, 2–3 stages), which changes the
`kNumMmaWarpgroups = bM/64` thread layout (`kernel_traits.hpp:52`) and occupancy.

---

## 3. Tensor-core rewrite (Milestone 4) — wgmma → `mma.sync`

Ampere has **no `wgmma`** (warpgroup async MMA) and **no TMA**. Replace with classic Ampere warp-level MMA
and `cp.async` staging:

| Hopper construct | Ampere `sm_86` replacement |
|---|---|
| `GMMA::ss_op_selector` int8→int32 (`kernel_traits.hpp:75-78`) | `SM80_16x8x32_S32S8S8S32_TN` (`mma.sync.aligned.m16n8k32.s8`) — **native int8 TC on Ampere** |
| `GMMA` fp16→fp32 (denoise, `:144-147`) | `SM80_16x8x16_F32F16F16F32_TN`; bf16 out via `SM80_16x8x16_F32BF16BF16F32_TN` |
| TMA load (`SM90_TMA_LOAD`, `make_tma_copy`) | `cp.async.cg.shared.global` (`SM80_CP_ASYNC_CACHEALWAYS`, already used for scales at `kernel_traits.hpp:128`) + `cp_async_fence`/`cp_async_wait<N>` double/triple buffer |
| `SM90_TMA_REDUCE_ADD` (split-K, `pearl_noisingA_kernel.h:202`) | separate `atomicAdd`/epilogue reduction pass (no TMA reduce on Ampere) |
| `SM90_U32x4_STSM_N` (stmatrix) | `ldmatrix` (`SM75_U32x4_LDSM_N`, already present) for loads; manual SMEM store loop for writeback |
| warp-spec producer/consumer + `warpgroup_reg_dealloc/alloc` (`setmaxnreg`) | **delete** — uniform multi-warp CTA: all warps `cp.async`-load, then all compute |
| clusters / multicast / `launch_kernel_on_cluster` | **delete** — plain `<<<grid,block,smem>>>`, `cM=cN=1` (already the default config) |
| `PipelineTmaAsync` + mbarriers | hand-rolled cp.async ring buffer with `__syncthreads` phase barriers |
| named barriers (`named_barrier.hpp`) | plain `bar.sync` works on Ampere; but the **warpgroup-role logic** gating them must be redesigned for a uniform CTA |

CUTLASS 3.x `cute` abstractions (`Tensor`, `TiledMMA`, `cp.async` atoms) are available on sm_80/86, so a
practical route is to keep the `cute` mainloop scaffolding and swap the **atoms + pipeline** rather than
rewrite from raw PTX.

---

## 4. The hashing / quant / convert path (mostly free)

`blake3.cu`, `inner_hash_kernel.cu`, `quantize_kernel.cu`, `denoise_converter_kernel.h`,
`noise_generation_kernel.h`, and the `commitment_hash`/`reduce_roots` blake3 math compile on sm_86 as-is
once the build is retargeted. Only `MerkleTreeRootsKernel` (the **leaf** stage that uses TMA to stream
the tensor into SMEM) needs a cp.async rewrite; the blake3 compression it calls is reused unchanged.

This is the right place to **start** (Milestone 3→4): get blake3, inner-hash, and noise-gen passing
against [`tests/golden/`](../tests/golden/) before touching the tensor cores. Those kernels alone let you
validate the entire **PoW transcript + commitment** path on GPU.

---

## 5. The #1 risk: accumulator fragment layout (correctness, not compilation)

The mining proof hashes **GEMM-intermediate partial sums** whose register→thread arrangement is baked
into the **Hopper wgmma layout**. The transcript extraction (`collective_mainloop.hpp:276-311`), the
`permute_Aregs_fp8` register repack (`utils.h:232-279`, a misnamed 8-bit *int* shuffle — there is no fp8),
and `write_host_signal_header` (which reports per-thread row/col mapping, `pearl_gemm_kernel.h:268`) all
encode that layout.

Ampere `mma.sync` (m16n8k32 / m16n8k16) produces a **different** accumulator fragment layout. A port that
gets the MMA numerically working but reshuffles fragments incorrectly will compute a
**plausible-but-wrong** inner hash and **silently** fail PoW — it fails *at runtime against golden
vectors*, not at compile time. **Mitigation:** re-derive the m16n8k32 thread→(row,col) map from the PTX
ISA, and gate every kernel behind the `tests/golden/` inner-hash and transcript checks before trusting it.
This is why the golden vectors record the **winning tile's 16 transcript words**, not just `found/C`.

---

## 6. Milestone-mapped work breakdown

- **M3 (skeleton):** fork pearl-gemm build, retarget `sm_86`, get it to compile the **portable** kernels;
  `prl3090-miner list-devices` + NVML; run blake3/inner-hash/noise-gen kernels vs golden. *Exit:* portable
  kernels bit-exact on GPU.
- **M4 (tensor-core correctness):** rewrite `NoisingKernelA/B` first (smaller, simpler than the fused
  mainloop), then `hopper_gemm_ws` mainloop+epilogue, then `MerkleTreeRootsKernel`. Validate each against
  golden (`C`, `found_block`, winning indices, transcript words). *Exit:* full pipeline bit-exact, no
  memory errors.
- **M6 (performance):** Nsight Compute/Systems; tune tiles to the 99 KB SMEM budget, cp.async stages,
  occupancy, `streams=2` overlap, stale-job cancel latency. *Target:* ≥50 TH/s, stale <2%, reject <1%.
- **M8 (close-to-the-bone):** Ampere-specific tuning presets; benchmark vs AlphaMiner black-box.

See [`STATUS.md`](../STATUS.md) for the honest current state and effort estimate.

## 6b. Progress — verified on the RTX 3090 (sm_86, CUDA 12.9)

All kernels below were compiled and validated **bit-for-bit against `tests/golden/` on the physical
3090**. Throughput is kernel-only (cudaEvent), int8, MACs/s → TOPS (×2).

| Kernel | What | Throughput | Golden |
|---|---|---|---|
| `naive_sm86.cu` | scalar integer-core full pipeline | ~2.9 GMAC/s (~0.006 TOPS) | C, found, indices, transcript, BLAKE3 ✅ |
| `mma_gemm_sm86.cu` `k_mma_gemm` | mma.sync m16n8k32 GEMM, one warp/tile | 6.6 TOPS @1024³ | C ✅ |
| `mma_gemm_sm86.cu` `k_mma_gemm_smem` | + 64×64 SMEM tiling | 16.7 TOPS @1024³ | C ✅ |
| `mma_gemm_sm86.cu` `k_mma_gemm_cpasync` | + 128×128 tile, cp.async double-buffer | 23.6 @1024³, **53.9 @4096³** | C ✅ |
| **`mine_sm86.cu` `k_mine`** | **FUSED real miner: noised GEMM + per-chunk transcript (register warp-XOR) + on-device BLAKE3** | **33.6 @1024³, 38.6 @2048³ TOPS** | **found, indices, transcript words ✅ (r=128)** |

Peak reference: GA102 dense int8 ≈ **284 TOPS**. The fused miner is at ~13%; the plain GEMM ~19%.

## 6c. The way to maximize the 3090 (prioritized, grounded in the above + Ampere GEMM practice)

1. **Kill per-launch / per-attempt overhead** (biggest *real-miner* win): persistent device buffers, a
   resident/streamed kernel, and batch many candidate tiles per launch. The orchestrator harness today
   does malloc+copy+launch+free per attempt — that, not the kernel, caps the end-to-end loop.
2. **`ldmatrix.x4` fragment loads** from SMEM instead of the current manual byte-`pk()` gathers —
   bank-conflict-free and far fewer instructions; typically the single largest GEMM speedup remaining.
3. **SMEM swizzle / XOR-permuted layout** to remove remaining bank conflicts in A/B staging.
4. **Register + async double-buffering** of the mma pipeline (more cp.async stages, prefetch fragments).
5. **Tile/occupancy tuning** to GA102 (larger block tiles within the 99 KB opt-in SMEM; balance regs).
6. **Power/clock preset** (`docs/rtx3090-tuning.md`) for TOPS/W once correctness+throughput are fixed.

Each step stays gated by `cuda/tests/validate_mine.py` (full golden incl. transcript words), so speed work
can never silently break the PoW. Realistic open-kernel ceiling on GA102 int8 is ~150–200 TOPS
(~50–70% of peak); the remaining gap from 38 TOPS is items 1–5, each well-understood.

## 7. Profiling toolchain (PRD §12.6)
`scripts/profile_nsight.sh` wraps **Nsight Compute** (`ncu`) for per-kernel counters (tensor-core
utilization, SMEM/occupancy, memory throughput) and **Nsight Systems** (`nsys`) for stream overlap and
job-switch latency. NVML (`nvml_monitor.cpp`) runs on a **separate thread** from the hot loop (PRD §18),
sampling power/temp/clocks. Per-kernel timing uses CUDA events.
