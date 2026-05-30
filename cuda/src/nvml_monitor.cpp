// NVML telemetry sampled on a BACKGROUND thread, kept off the mining hot path (PRD §18).
// Implemented; compiles and links against NVML (libnvidia-ml). Provides GPU + VRAM temp,
// power, and live clocks for the status line (PRD §16) and thermal safety (PRD §17).
#include "prl_cuda.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include <nvml.h>

namespace {

std::thread g_thread;
std::atomic<bool> g_run{false};
std::mutex g_mutex;
PrlDeviceInfo g_sample{};
int g_index = 0;

void sample_loop() {
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(static_cast<unsigned>(g_index), &dev) != NVML_SUCCESS) return;

    while (g_run.load(std::memory_order_relaxed)) {
        PrlDeviceInfo s{};
        s.index = g_index;

        unsigned int t = 0;
        if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS)
            s.temp_c = static_cast<float>(t);

        // VRAM (memory) junction temperature via the field-value API.
        nvmlFieldValue_t fv{};
        fv.fieldId = NVML_FI_DEV_MEMORY_TEMP;
        if (nvmlDeviceGetFieldValues(dev, 1, &fv) == NVML_SUCCESS &&
            fv.nvmlReturn == NVML_SUCCESS)
            s.vram_temp_c = static_cast<float>(fv.value.siVal);

        unsigned int mw = 0;
        if (nvmlDeviceGetPowerUsage(dev, &mw) == NVML_SUCCESS)
            s.power_w = static_cast<float>(mw) / 1000.0f;

        unsigned int clk = 0;
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clk) == NVML_SUCCESS)
            s.sm_clock_mhz = static_cast<int>(clk);
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clk) == NVML_SUCCESS)
            s.mem_clock_mhz = static_cast<int>(clk);

        {
            std::lock_guard<std::mutex> lk(g_mutex);
            // preserve the static fields filled by prl_get_device_info
            std::strncpy(s.name, g_sample.name, sizeof(s.name) - 1);
            s.cc_major = g_sample.cc_major;
            s.cc_minor = g_sample.cc_minor;
            s.total_mem_bytes = g_sample.total_mem_bytes;
            s.smem_per_block_optin = g_sample.smem_per_block_optin;
            g_sample = s;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

} // namespace

extern "C" {

PrlStatus prl_nvml_start(int index) {
    if (nvmlInit_v2() != NVML_SUCCESS) return PRL_ERR_CUDA;
    g_index = index;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        prl_get_device_info(index, &g_sample);  // seed static fields
    }
    g_run.store(true);
    g_thread = std::thread(sample_loop);
    return PRL_OK;
}

PrlStatus prl_nvml_sample(PrlDeviceInfo* out) {
    if (!out) return PRL_ERR_BAD_ARG;
    std::lock_guard<std::mutex> lk(g_mutex);
    *out = g_sample;
    return PRL_OK;
}

void prl_nvml_stop(void) {
    g_run.store(false);
    if (g_thread.joinable()) g_thread.join();
    nvmlShutdown();
}

} // extern "C"
