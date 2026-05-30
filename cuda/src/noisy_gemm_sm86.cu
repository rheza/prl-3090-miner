// Ampere sm_86 NoisyGEMM + noising kernels.  ***SCAFFOLD — M4 work.***
//
// This file is where the Hopper->Ampere port lands. The official kernels
// (pearl-official/miner/pearl-gemm/csrc/gemm/) use WGMMA + TMA + warp-specialization +
// clusters, none of which exist on sm_86. The replacement strategy is documented in
// docs/cuda-sm86-port.md §3 and summarized inline below. Bodies intentionally return
// "not implemented" so nothing here is mistaken for a working miner.
//
// Correctness is gated by tests/golden/ (C, found_block, winning indices, AND the 16
// PoW transcript words). The #1 hazard (cuda-sm86-port.md §5) is the accumulator
// fragment layout: m16n8k32 mma.sync arranges results across threads DIFFERENTLY than
// WGMMA, so the inner-hash/transcript extraction must be re-derived for the new layout
// or the PoW silently diverges.
#include "prl_cuda.h"
#include <cuda_runtime.h>

namespace prl {

// --- noising kernels (replace NoisingKernelA/B) ----------------------------
// A_noised = A + E_AL@E_AR ; A_E_BL = A@E_BL          (protocol-notes.md §4.3 step 1)
// B_noised = B + E_BL@E_BR ; EAR_BpEB = E_AR@B_noised (step 2)
// Ampere mapping:
//   * int8 GEMM:  cutlass SM80_16x8x32_S32S8S8S32_TN (mma.sync.m16n8k32.s8) — native int8 TC.
//   * load A/B/E tiles to SMEM with cp.async.cg (SM80_CP_ASYNC_CACHEALWAYS), double-buffered;
//     cp_async_fence / cp_async_wait<N>. NO TMA, NO mbarrier.
//   * all warps load then compute (no producer/consumer warp specialization).
//   * SMEM budget <= ~99 KB/SM (device_manager surfaces it) -> tile 128x128x64, 2-3 stages.
__global__ void noisingA_sm86(/* int8 A,E_AL,E_AR,E_BL -> int8 A_noised, int32 A_E_BL */) {
    // TODO(M4): implement per cuda-sm86-port.md §3.
}
__global__ void noisingB_sm86(/* int8 B,E_AR,E_BL,E_BR -> int8 B_noised, int32 EAR_BpEB */) {
    // TODO(M4)
}

// --- fused noisy GEMM + transcript (replace hopper_gemm_ws) -----------------
// Tiled int8 GEMM of (A_noised, B_noised). After each noise_rank-wide k-reduction,
// hash each 16x16 tile of the running accumulator and fold into its transcript
// (rotl13 ^). Then denoise: C = C_noised - A_E_BL@E_BR - E_AL@EAR_BpEB. (steps 3-6)
__global__ void noisy_gemm_sm86(/* ... -> int32 C, transcripts[ntile][16] */) {
    // TODO(M4): mma.sync mainloop + cp.async pipeline + transcript extraction.
    // CRITICAL: derive thread->(row,col) for SM80_16x8x32 accumulator and feed the
    // inner-hash exactly as the reference (inner_hash.py), validated vs golden transcript.
}

// Host wrappers that the job_runner calls. Return NOT_IMPLEMENTED until filled in.
extern "C" PrlStatus prl_launch_noising(const PrlMiningJob*, void* /*workspace*/, cudaStream_t) {
    return PRL_ERR_NOT_IMPLEMENTED;
}
extern "C" PrlStatus prl_launch_noisy_gemm(const PrlMiningJob*, void* /*workspace*/,
                                           PrlMiningResult*, cudaStream_t) {
    return PRL_ERR_NOT_IMPLEMENTED;
}

} // namespace prl
