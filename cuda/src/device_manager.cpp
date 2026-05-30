// CUDA device enumeration/selection (implemented; compiles with the CUDA Toolkit).
// Satisfies PRD Milestone 3: `prl3090-miner list-devices` shows the RTX 3090.
#include "prl_cuda.h"

#include <cstring>
#include <cuda_runtime.h>

extern "C" {

int prl_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

PrlStatus prl_get_device_info(int index, PrlDeviceInfo* out) {
    if (!out) return PRL_ERR_BAD_ARG;
    int n = prl_device_count();
    if (index < 0 || index >= n) return PRL_ERR_NO_DEVICE;

    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, index) != cudaSuccess) return PRL_ERR_CUDA;

    std::memset(out, 0, sizeof(*out));
    out->index = index;
    std::strncpy(out->name, p.name, sizeof(out->name) - 1);
    out->cc_major = p.major;
    out->cc_minor = p.minor;
    out->total_mem_bytes = static_cast<uint64_t>(p.totalGlobalMem);
    out->sm_clock_mhz = p.clockRate / 1000;            // kHz -> MHz (base)
    out->mem_clock_mhz = p.memoryClockRate / 1000;

    int optin = 0;
    // The sm_86 opt-in SMEM ceiling (~99 KB) is a hard constraint for the GEMM tile
    // (docs/cuda-sm86-port.md §2). Surface it so the heuristics can size stages.
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, index);
    out->smem_per_block_optin = optin;
    // temp/power/live clocks are filled by the NVML monitor (prl_nvml_sample).
    return PRL_OK;
}

PrlStatus prl_select_device(int index) {
    if (cudaSetDevice(index) != cudaSuccess) return PRL_ERR_CUDA;
    return PRL_OK;
}

const char* prl_status_str(PrlStatus s) {
    switch (s) {
        case PRL_OK: return "ok";
        case PRL_ERR_NO_DEVICE: return "no_device";
        case PRL_ERR_BAD_ARG: return "bad_arg";
        case PRL_ERR_NOT_IMPLEMENTED: return "not_implemented";
        case PRL_ERR_CUDA: return "cuda_error";
        default: return "unknown";
    }
}

} // extern "C"
