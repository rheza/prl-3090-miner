// Ampere sm_86 INT8 tensor-core GEMM via mma.sync.m16n8k32.s8.
//
// First real tensor-core step toward target speed (docs/cuda-sm86-port.md §3). This is a
// CORRECT but not-yet-fully-optimized kernel (one warp per 16x8 output tile, no shared-mem
// tiling / cp.async pipeline yet) — its job is to prove the int8 tensor-core path is
// bit-exact for this protocol and to measure the speedup vs the naive integer-core path.
// Validated against tests/golden/ C (== A@B) on the GPU by cuda/tests/validate_mma.py.
//
// Fragment layout for mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (PTX ISA):
//   lane: groupID = lane>>2 (0..7), tig = lane&3 (0..3)
//   A(16x32 row-major): a0=(groupID, tig*4+0..3) a1=(groupID+8, tig*4+0..3)
//                       a2=(groupID, tig*4+16..19) a3=(groupID+8, tig*4+16..19)
//   B(32x8  col-major): b0=(tig*4+0..3, groupID)  b1=(tig*4+16..19, groupID)
//   C(16x8  int32):     c0=(groupID,tig*2) c1=(groupID,tig*2+1) c2=(groupID+8,tig*2) c3=(groupID+8,tig*2+1)
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t pk(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return (uint32_t)(uint8_t)b0 | ((uint32_t)(uint8_t)b1 << 8) |
           ((uint32_t)(uint8_t)b2 << 16) | ((uint32_t)(uint8_t)b3 << 24);
}

// One warp computes one 16x8 output tile of C = A(m x k) @ B(k x n), int8 -> int32.
__global__ void k_mma_gemm(const int8_t* A, const int8_t* B, int32_t* C, int m, int k, int n) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2;          // groupID 0..7
    int t = lane & 3;           // threadID_in_group 0..3
    int rm = blockIdx.x * 16;   // tile row base
    int rn = blockIdx.y * 8;    // tile col base

    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int k0 = 0; k0 < k; k0 += 32) {
        // A fragment
        const int8_t* Ar0 = A + (size_t)(rm + g) * k + k0;
        const int8_t* Ar8 = A + (size_t)(rm + g + 8) * k + k0;
        uint32_t a0 = pk(Ar0[t*4+0],  Ar0[t*4+1],  Ar0[t*4+2],  Ar0[t*4+3]);
        uint32_t a1 = pk(Ar8[t*4+0],  Ar8[t*4+1],  Ar8[t*4+2],  Ar8[t*4+3]);
        uint32_t a2 = pk(Ar0[t*4+16], Ar0[t*4+17], Ar0[t*4+18], Ar0[t*4+19]);
        uint32_t a3 = pk(Ar8[t*4+16], Ar8[t*4+17], Ar8[t*4+18], Ar8[t*4+19]);
        // B fragment (B row-major in memory; col = rn+g, rows = k0 + t*4 + {0..3} / {16..19})
        int bc = rn + g;
        uint32_t b0 = pk(B[(size_t)(k0+t*4+0)*n+bc],  B[(size_t)(k0+t*4+1)*n+bc],
                         B[(size_t)(k0+t*4+2)*n+bc],  B[(size_t)(k0+t*4+3)*n+bc]);
        uint32_t b1 = pk(B[(size_t)(k0+t*4+16)*n+bc], B[(size_t)(k0+t*4+17)*n+bc],
                         B[(size_t)(k0+t*4+18)*n+bc], B[(size_t)(k0+t*4+19)*n+bc]);
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    C[(size_t)(rm + g)     * n + (rn + t*2 + 0)] = c0;
    C[(size_t)(rm + g)     * n + (rn + t*2 + 1)] = c1;
    C[(size_t)(rm + g + 8) * n + (rn + t*2 + 0)] = c2;
    C[(size_t)(rm + g + 8) * n + (rn + t*2 + 1)] = c3;
}

static char g_err[256] = {0};
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    snprintf(g_err, sizeof(g_err), "%s @ %d: %s", #call, __LINE__, cudaGetErrorString(e_)); \
    return 4; } } while (0)

extern "C" {

const char* prl_mma_last_error(void) { return g_err; }
int prl_mma_device_count(void) { int n = 0; cudaGetDeviceCount(&n); return n; }

int prl_mma_gemm(const int8_t* A, const int8_t* B, int32_t* C, int m, int k, int n) {
    if (m % 16 || n % 8 || k % 32) { snprintf(g_err, sizeof(g_err), "need m%%16==n%%8==k%%32==0"); return 2; }
    int8_t *dA, *dB; int32_t* dC;
    CK(cudaMalloc(&dA, (size_t)m*k)); CK(cudaMalloc(&dB, (size_t)k*n));
    CK(cudaMalloc(&dC, (size_t)m*n*sizeof(int32_t)));
    CK(cudaMemcpy(dA, A, (size_t)m*k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B, (size_t)k*n, cudaMemcpyHostToDevice));
    dim3 grid(m/16, n/8);
    k_mma_gemm<<<grid, 32>>>(dA, dB, dC, m, k, n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C, dC, (size_t)m*n*sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

// Kernel-only timing (persistent buffers, cudaEvent) -> avg ms/iter in *out_ms.
int prl_mma_gemm_bench(int m, int k, int n, int iters, double* out_ms) {
    if (m % 16 || n % 8 || k % 32) { snprintf(g_err, sizeof(g_err), "need m%%16==n%%8==k%%32==0"); return 2; }
    int8_t *dA, *dB; int32_t* dC;
    CK(cudaMalloc(&dA, (size_t)m*k)); CK(cudaMalloc(&dB, (size_t)k*n));
    CK(cudaMalloc(&dC, (size_t)m*n*sizeof(int32_t)));
    CK(cudaMemset(dA, 1, (size_t)m*k)); CK(cudaMemset(dB, 1, (size_t)k*n));
    dim3 grid(m/16, n/8);
    k_mma_gemm<<<grid, 32>>>(dA, dB, dC, m, k, n);  // warm-up
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) k_mma_gemm<<<grid, 32>>>(dA, dB, dC, m, k, n);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    *out_ms = (double)ms / iters;
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

} // extern "C"
