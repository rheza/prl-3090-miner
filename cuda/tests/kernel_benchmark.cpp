// Per-kernel micro-benchmark skeleton (PRD §12.6). Times prl_run_noisy_gemm over a
// production-shaped job. Reports "not implemented" until the M4 kernels land.
#include "prl_cuda.h"

#include <chrono>
#include <cstdio>
#include <vector>

int main() {
    if (prl_device_count() <= 0) { std::printf("no device\n"); return 1; }
    prl_select_device(0);

    // production-like shape (settings.py: 128 x 256, rank 128)
    PrlMiningJob job{};
    job.m = 128; job.k = 256; job.n = 256; job.noise_rank = 128; job.noise_range = 128;
    job.hash_tile_h = job.hash_tile_w = 16;
    job.matmul_tile_h = 128; job.matmul_tile_w = 128;
    std::vector<int8_t> A(job.m * job.k, 1), B(job.k * job.n, 1);
    std::vector<int8_t> EAL(job.m * job.noise_rank), EAR(job.noise_rank * job.k),
                        EBL(job.k * job.noise_rank), EBR(job.noise_rank * job.n);
    job.A = A.data(); job.B = B.data();
    job.E_AL = EAL.data(); job.E_AR = EAR.data(); job.E_BL = EBL.data(); job.E_BR = EBR.data();
    std::vector<int32_t> C(job.m * job.n);
    PrlMiningResult out{}; out.C = C.data();

    const int iters = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    PrlStatus st = PRL_OK;
    for (int i = 0; i < iters; ++i) {
        st = prl_run_noisy_gemm(&job, &out);
        if (st != PRL_OK) break;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (st != PRL_OK) {
        std::printf("prl_run_noisy_gemm -> %s (kernels are M4 work)\n", prl_status_str(st));
        return 0;
    }
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double macs = double(job.m) * job.k * job.n;
    std::printf("noisy_gemm: %.3f ms/iter, %.2f GMAC/iter, %.1f GMAC/s\n",
                ms, macs / 1e9, macs / (ms / 1e3) / 1e9);
    return 0;
}
