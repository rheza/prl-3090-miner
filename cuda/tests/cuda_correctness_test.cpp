// Minimal C++ smoke test: device enumeration + NVML telemetry.
// Build via CMake (target cuda_correctness_test). Exits 0 iff >=1 CUDA device.
#include "prl_cuda.h"

#include <cstdio>

int main() {
    int n = prl_device_count();
    std::printf("CUDA devices: %d\n", n);
    if (n <= 0) {
        std::printf("FAIL: no CUDA device\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        PrlDeviceInfo info{};
        if (prl_get_device_info(i, &info) != PRL_OK) {
            std::printf("  [%d] get_device_info FAILED\n", i);
            return 1;
        }
        std::printf("  [%d] %s  cc %d.%d  mem %.1f GiB  smem_optin %d KiB\n",
                    info.index, info.name, info.cc_major, info.cc_minor,
                    info.total_mem_bytes / (1024.0 * 1024 * 1024),
                    info.smem_per_block_optin / 1024);
        if (info.cc_major == 8 && info.cc_minor == 6)
            std::printf("       -> Ampere sm_86 (RTX 3090 class): target architecture OK\n");
    }
    if (prl_nvml_start(0) == PRL_OK) {
        PrlDeviceInfo s{};
        prl_nvml_sample(&s);
        std::printf("NVML[0]: temp %.0fC  vram %.0fC  power %.1fW  sm %dMHz  mem %dMHz\n",
                    s.temp_c, s.vram_temp_c, s.power_w, s.sm_clock_mhz, s.mem_clock_mhz);
        prl_nvml_stop();
    }
    std::printf("PASS\n");
    return 0;
}
