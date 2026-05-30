// PRL-3090 CUDA backend — C ABI.
//
// Callable from Python (ctypes/pybind) and Rust (FFI). The correctness entry point
// `prl_run_noisy_gemm` takes explicit inputs and reproduces the golden vector outputs
// bit-for-bit; the production `prl_search` variant wraps it for a real mining loop.
#ifndef PRL_CUDA_H
#define PRL_CUDA_H

#include "prl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum PrlStatus {
    PRL_OK = 0,
    PRL_ERR_NO_DEVICE = 1,
    PRL_ERR_BAD_ARG = 2,
    PRL_ERR_NOT_IMPLEMENTED = 3,   // returned by the unported kernels today
    PRL_ERR_CUDA = 4,
} PrlStatus;

// Device management (implemented, compiles today — src/device_manager.cpp)
int       prl_device_count(void);
PrlStatus prl_get_device_info(int index, PrlDeviceInfo* out);
PrlStatus prl_select_device(int index);

// NVML telemetry on a background thread (implemented — src/nvml_monitor.cpp)
PrlStatus prl_nvml_start(int index);
PrlStatus prl_nvml_sample(PrlDeviceInfo* out);   // latest cached sample, hot-path safe
void      prl_nvml_stop(void);

// Correctness entry point (golden-vector contract). Runs the full NoisyGEMM pipeline:
//   noise A/B -> tiled noisy GEMM -> inner-hash transcript -> denoise -> PoW check.
// Returns PRL_ERR_NOT_IMPLEMENTED until the sm_86 tensor-core kernels land (M4).
PrlStatus prl_run_noisy_gemm(const PrlMiningJob* job, PrlMiningResult* out);

// Production search: iterate the work space for `header` until found or `should_stop`
// (set by the orchestrator on a new tip). Wraps prl_run_noisy_gemm. (M5)
PrlStatus prl_search(const PrlMiningJob* job, volatile int* should_stop,
                     PrlMiningResult* out);

const char* prl_status_str(PrlStatus s);

#ifdef __cplusplus
}
#endif
#endif // PRL_CUDA_H
