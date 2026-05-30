// Orchestrates the device pipeline behind the C ABI (prl_run_noisy_gemm / prl_search).
// Today it returns PRL_ERR_NOT_IMPLEMENTED at the kernel call site (honest: the sm_86
// tensor-core kernels are M4). The H2D/D2H plumbing is sketched so M4 is a fill-in.
#include "prl_cuda.h"
#include <cuda_runtime.h>

namespace prl {
PrlStatus prl_launch_noising(const PrlMiningJob*, void*, cudaStream_t);
PrlStatus prl_launch_noisy_gemm(const PrlMiningJob*, void*, PrlMiningResult*, cudaStream_t);
}

extern "C" {

PrlStatus prl_run_noisy_gemm(const PrlMiningJob* job, PrlMiningResult* out) {
    if (!job || !out || !job->A || !job->B) return PRL_ERR_BAD_ARG;
    out->found = 0;
    out->a_row_start = out->b_col_start = -1;

    // M4 plumbing outline:
    //   1. cudaMalloc device copies of A,B,E_* + workspace (A_noised,B_noised,A_E_BL,EAR_BpEB,
    //      transcripts, C). Use pinned host staging + cudaMemcpyAsync on a stream (PRD §18).
    //   2. prl_launch_noising(job, ws, stream)          // noisingA/B
    //   3. prl_launch_noisy_gemm(job, ws, out, stream)  // fused GEMM + transcript + denoise + PoW
    //   4. cudaMemcpyAsync C + result back; record kernel_ms via CUDA events.
    cudaStream_t stream = nullptr;
    PrlStatus s = prl::prl_launch_noising(job, nullptr, stream);
    if (s != PRL_OK) return s;                 // currently PRL_ERR_NOT_IMPLEMENTED
    return prl::prl_launch_noisy_gemm(job, nullptr, out, stream);
}

PrlStatus prl_search(const PrlMiningJob* job, volatile int* should_stop, PrlMiningResult* out) {
    // M5: loop over the work space (vary nonce/A/B per the protocol) calling
    // prl_run_noisy_gemm until found or *should_stop is set by the orchestrator (new tip).
    if (should_stop && *should_stop) return PRL_OK;
    return prl_run_noisy_gemm(job, out);
}

} // extern "C"
