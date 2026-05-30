// pybind11 module `prl_cuda` — what miner/backends.py imports for the cuda-sm86 backend.
// Wraps the C ABI in prl_cuda.h. device_info() works today; search()/run_noisy_gemm()
// raise until the M4 kernels land (they return PRL_ERR_NOT_IMPLEMENTED).
#include "prl_cuda.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;

static py::dict device_info(int index) {
    PrlDeviceInfo info{};
    if (prl_get_device_info(index, &info) != PRL_OK)
        throw std::runtime_error("prl_get_device_info failed");
    // one-shot NVML read for live temp/power
    PrlDeviceInfo live = info;
    if (prl_nvml_start(index) == PRL_OK) {
        prl_nvml_sample(&live);
        prl_nvml_stop();
    }
    py::dict d;
    d["index"] = info.index;
    d["name"] = std::string(info.name);
    d["backend"] = "cuda-sm86";
    d["cc"] = std::to_string(info.cc_major) + "." + std::to_string(info.cc_minor);
    d["total_mem_bytes"] = info.total_mem_bytes;
    d["smem_per_block_optin"] = info.smem_per_block_optin;
    d["temp_c"] = live.temp_c ? py::cast(live.temp_c) : py::none();
    d["vram_temp_c"] = live.vram_temp_c ? py::cast(live.vram_temp_c) : py::none();
    d["power_w"] = live.power_w ? py::cast(live.power_w) : py::none();
    d["sm_clock_mhz"] = live.sm_clock_mhz;
    d["mem_clock_mhz"] = live.mem_clock_mhz;
    return d;
}

static py::object run_noisy_gemm(py::dict /*job*/) {
    // M4: marshal numpy int8 arrays + keys into PrlMiningJob, call prl_run_noisy_gemm,
    // return (C, found, a_row_start, b_col_start, transcript). Until then:
    PrlMiningJob job{};
    PrlMiningResult res{};
    PrlStatus s = prl_run_noisy_gemm(&job, &res);
    throw std::runtime_error(std::string("cuda-sm86 kernels not implemented yet (") +
                             prl_status_str(s) + "); see docs/cuda-sm86-port.md");
}

PYBIND11_MODULE(prl_cuda, m) {
    m.doc() = "PRL-3090 Ampere sm_86 mining backend";
    m.def("device_count", []() { return prl_device_count(); });
    m.def("device_info", &device_info, py::arg("index") = 0);
    m.def("run_noisy_gemm", &run_noisy_gemm, py::arg("job"));
    m.def("search", &run_noisy_gemm, py::arg("job"));  // M5 wires the real loop
}
