// Naive-but-CORRECT CUDA NoisyGEMM for sm_86.  *** This actually runs and is
// validated against tests/golden/ on a real RTX 3090. ***
//
// It uses plain integer CUDA cores (NO tensor cores), so it is SLOW — this is the
// Milestone-4 *correctness* gate, not the performance target. It exists to prove the
// whole device pipeline (noising -> noised tiled GEMM -> inner-hash transcript ->
// on-device keyed-BLAKE3 PoW) is bit-exact with the audited NumPy reference
// (reference/pearl_reference.py) before any tensor-core optimization (docs/cuda-sm86-port.md).
//
// Faithfulness notes:
//  * The transcript hashes the NOISED partial sums (A_noised@B_noised), accumulated per
//    noise_rank-wide k-chunk, exactly as noisy_gemm.py:_tiled_matmul.
//  * The C output equals A@B (the denoise identity, proven by the reference); the naive
//    validator computes A@B directly for C, and the full noised path for the PoW.
//  * Golden cases use dims divisible by noise_rank and 16, so there are no partial tiles.
//
// extern "C" entry points are driven from Python via ctypes (reference/run_cuda_golden.py).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// --------------------------- device BLAKE3 (single 64-byte keyed block) -----
// The PoW hashes exactly 64 bytes (16 u32 LE) with a 32-byte key -> one chunk, one
// block, flags = CHUNK_START|CHUNK_END|ROOT|KEYED_HASH. Output = first 8 state words.
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ void g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    s[a] = s[a] + s[b] + mx; s[d] = rotr32(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my; s[d] = rotr32(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 7);
}
__constant__ uint8_t MSG_SCHED[7][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
    {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
    {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
    {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
    {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
    {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13},
};
__constant__ uint32_t BLAKE3_IV[8] = {
    0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,
    0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};

// msg[16] (u32 words), key[8] (u32), -> out[8] digest words. block_len=64, counter=0.
__device__ void blake3_keyed_block(const uint32_t* msg, const uint32_t* key, uint32_t* out) {
    const uint32_t FLAGS = 1u | 2u | 8u | 16u;  // CHUNK_START|CHUNK_END|ROOT|KEYED_HASH
    uint32_t s[16];
    for (int i = 0; i < 8; ++i) s[i] = key[i];
    s[8]=BLAKE3_IV[0]; s[9]=BLAKE3_IV[1]; s[10]=BLAKE3_IV[2]; s[11]=BLAKE3_IV[3];
    s[12]=0u; s[13]=0u; s[14]=64u; s[15]=FLAGS;
    for (int r = 0; r < 7; ++r) {
        const uint8_t* sc = MSG_SCHED[r];
        g(s,0,4, 8,12, msg[sc[0]],  msg[sc[1]]);
        g(s,1,5, 9,13, msg[sc[2]],  msg[sc[3]]);
        g(s,2,6,10,14, msg[sc[4]],  msg[sc[5]]);
        g(s,3,7,11,15, msg[sc[6]],  msg[sc[7]]);
        g(s,0,5,10,15, msg[sc[8]],  msg[sc[9]]);
        g(s,1,6,11,12, msg[sc[10]], msg[sc[11]]);
        g(s,2,7, 8,13, msg[sc[12]], msg[sc[13]]);
        g(s,3,4, 9,14, msg[sc[14]], msg[sc[15]]);
    }
    for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];   // (s[i+8]^=key[i] not needed: only first 8 words used)
}

// --------------------------- kernels ---------------------------------------
#define IDX(r, c, ld) ((r) * (ld) + (c))

// A_noised[m,k] = (int8)(A + E_AL@E_AR) ; one thread per element.
__global__ void k_noiseA(const int8_t* A, const int8_t* E_AL, const int8_t* E_AR,
                         int8_t* A_noised, int m, int k, int r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * k) return;
    int i = idx / k, j = idx % k;
    int32_t e = 0;
    for (int c = 0; c < r; ++c) e += (int32_t)E_AL[IDX(i, c, r)] * (int32_t)E_AR[IDX(c, j, k)];
    A_noised[idx] = (int8_t)((int32_t)A[idx] + e);
}
// B_noised[k,n] = (int8)(B + E_BL@E_BR)
__global__ void k_noiseB(const int8_t* B, const int8_t* E_BL, const int8_t* E_BR,
                         int8_t* B_noised, int k, int n, int r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * n) return;
    int i = idx / n, j = idx % n;
    int32_t e = 0;
    for (int c = 0; c < r; ++c) e += (int32_t)E_BL[IDX(i, c, r)] * (int32_t)E_BR[IDX(c, j, n)];
    B_noised[idx] = (int8_t)((int32_t)B[idx] + e);
}
// Plain C = A@B (int32) -> equals the denoised result.
__global__ void k_gemm_plain(const int8_t* A, const int8_t* B, int32_t* C, int m, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    int i = idx / n, j = idx % n;
    int32_t acc = 0;
    for (int c = 0; c < k; ++c) acc += (int32_t)A[IDX(i, c, k)] * (int32_t)B[IDX(c, j, n)];
    C[idx] = acc;
}

// One block per output tile (ti,tj), tile = r x r. Accumulates the NOISED partial sums
// in Cscr and folds 16x16 inner hashes into per-subtile transcripts after each k-chunk.
// transcripts laid out by absolute subtile: [ (m/16) x (n/16) ][16].
__global__ void k_noised_transcript(const int8_t* An, const int8_t* Bn, int32_t* Cscr,
                                    uint32_t* transcripts, int m, int k, int n, int r,
                                    int th, int tw) {
    int ti = blockIdx.x, tj = blockIdx.y;
    int i0 = ti * r, j0 = tj * r;
    int nsub_h = r / th, nsub_w = r / tw;     // subtiles per output tile
    int ST_cols = n / tw;
    int nchunks = k / r;

    // zero this tile's Cscr region
    for (int e = threadIdx.x; e < r * r; e += blockDim.x) Cscr[IDX(i0 + e / r, j0 + e % r, n)] = 0;
    __syncthreads();

    for (int pc = 0; pc < nchunks; ++pc) {
        int p = pc * r;
        // accumulate noised partial sums for this chunk
        for (int e = threadIdx.x; e < r * r; e += blockDim.x) {
            int a = e / r, b = e % r;
            int32_t acc = 0;
            for (int c = 0; c < r; ++c)
                acc += (int32_t)An[IDX(i0 + a, p + c, k)] * (int32_t)Bn[IDX(p + c, j0 + b, n)];
            Cscr[IDX(i0 + a, j0 + b, n)] += acc;
        }
        __syncthreads();
        // inner-hash each 16x16 subtile of the cumulative tile; fold into transcript
        int nsub = nsub_h * nsub_w;
        for (int st = threadIdx.x; st < nsub; st += blockDim.x) {
            int hi = st / nsub_w, wi = st % nsub_w;
            uint32_t h = 0;
            for (int a = 0; a < th; ++a)
                for (int b = 0; b < tw; ++b)
                    h ^= (uint32_t)Cscr[IDX(i0 + hi * th + a, j0 + wi * tw + b, n)];
            int sr = (i0 + hi * th) / th, sc = (j0 + wi * tw) / tw;
            uint32_t* tr = &transcripts[(sr * ST_cols + sc) * 16];
            int slot = pc & 15;
            tr[slot] = ((tr[slot] << 13) | (tr[slot] >> 19)) ^ h;  // rotl32(.,13) ^ h
        }
        __syncthreads();
    }
}

// Single-thread deterministic scan in the reference order (ti,tj,hi,wi); first transcript
// whose keyed-BLAKE3 <= target wins. target/key are 8-limb LE u256.
__global__ void k_pow_scan(const uint32_t* transcripts, const uint32_t* key, const uint32_t* target,
                           int m, int n, int r, int th, int tw,
                           int* found, int* a_row, int* b_col, uint32_t* tr_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int ST_cols = n / tw;
    int TI = m / r, TJ = n / r, SH = r / th, SW = r / tw;
    *found = 0; *a_row = -1; *b_col = -1;
    for (int ti = 0; ti < TI; ++ti)
    for (int tj = 0; tj < TJ; ++tj)
    for (int hi = 0; hi < SH; ++hi)
    for (int wi = 0; wi < SW; ++wi) {
        int sr = ti * SH + hi, sc = tj * SW + wi;
        const uint32_t* tr = &transcripts[(sr * ST_cols + sc) * 16];
        uint32_t dig[8];
        blake3_keyed_block(tr, key, dig);
        // dig <= target ? (compare 8 LE limbs, most-significant first)
        bool le = true;
        for (int l = 7; l >= 0; --l) { if (dig[l] < target[l]) { le = true; break; }
                                       if (dig[l] > target[l]) { le = false; break; } }
        if (le) {
            *found = 1; *a_row = sr * th; *b_col = sc * tw;
            for (int t = 0; t < 16; ++t) tr_out[t] = tr[t];
            return;
        }
    }
}

// --------------------------- host C ABI ------------------------------------
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::snprintf(g_err, sizeof(g_err), "%s @ %d: %s", #call, __LINE__, cudaGetErrorString(e_)); \
    return 4; } } while (0)

static char g_err[256] = {0};

__global__ void k_blake3_one(const uint32_t* msg, const uint32_t* key, uint32_t* out);

extern "C" {

const char* prl_naive_last_error(void) { return g_err; }
int prl_naive_device_count(void) { int n = 0; cudaGetDeviceCount(&n); return n; }

// out32 = blake3_keyed(msg64, key32). For unit-testing device BLAKE3 vs python blake3.
int prl_blake3_64(const uint8_t* msg64, const uint8_t* key32, uint8_t* out32) {
    uint32_t *dmsg, *dkey, *dout;
    CK(cudaMalloc(&dmsg, 64)); CK(cudaMalloc(&dkey, 32)); CK(cudaMalloc(&dout, 32));
    CK(cudaMemcpy(dmsg, msg64, 64, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkey, key32, 32, cudaMemcpyHostToDevice));
    k_blake3_one<<<1,1>>>(dmsg, dkey, dout);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out32, dout, 32, cudaMemcpyDeviceToHost));
    cudaFree(dmsg); cudaFree(dkey); cudaFree(dout);
    return 0;
}

int prl_naive_run(const int8_t* A, const int8_t* B,
                  const int8_t* E_AL, const int8_t* E_AR, const int8_t* E_BL, const int8_t* E_BR,
                  int m, int k, int n, int r, int th, int tw,
                  const uint8_t* pow_key32, const uint8_t* pow_target32_le,
                  int32_t* C_out, int* found_out, int* a_row_out, int* b_col_out,
                  uint32_t* transcript_out) {
    if (m % r || n % r || k % r || r % th || r % tw)
        { std::snprintf(g_err, sizeof(g_err), "naive backend needs dims divisible by rank/tile"); return 2; }

    int8_t *dA,*dB,*dEAL,*dEAR,*dEBL,*dEBR,*dAn,*dBn;
    int32_t *dC,*dCscr; uint32_t *dTr,*dKey,*dTgt,*dTrOut; int *dFound,*dARow,*dBCol;
    size_t ST = (size_t)(m / th) * (n / tw);
    CK(cudaMalloc(&dA, m*k)); CK(cudaMalloc(&dB, k*n));
    CK(cudaMalloc(&dEAL, m*r)); CK(cudaMalloc(&dEAR, r*k));
    CK(cudaMalloc(&dEBL, k*r)); CK(cudaMalloc(&dEBR, r*n));
    CK(cudaMalloc(&dAn, m*k)); CK(cudaMalloc(&dBn, k*n));
    CK(cudaMalloc(&dC, m*n*sizeof(int32_t))); CK(cudaMalloc(&dCscr, m*n*sizeof(int32_t)));
    CK(cudaMalloc(&dTr, ST*16*sizeof(uint32_t))); CK(cudaMemset(dTr, 0, ST*16*sizeof(uint32_t)));
    CK(cudaMalloc(&dKey, 32)); CK(cudaMalloc(&dTgt, 32)); CK(cudaMalloc(&dTrOut, 16*sizeof(uint32_t)));
    CK(cudaMalloc(&dFound, sizeof(int))); CK(cudaMalloc(&dARow, sizeof(int))); CK(cudaMalloc(&dBCol, sizeof(int)));

    CK(cudaMemcpy(dA, A, m*k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B, k*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEAL, E_AL, m*r, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEAR, E_AR, r*k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEBL, E_BL, k*r, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEBR, E_BR, r*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKey, pow_key32, 32, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dTgt, pow_target32_le, 32, cudaMemcpyHostToDevice));

    int T = 256;
    k_noiseA<<<(m*k+T-1)/T, T>>>(dA, dEAL, dEAR, dAn, m, k, r);
    k_noiseB<<<(k*n+T-1)/T, T>>>(dB, dEBL, dEBR, dBn, k, n, r);
    k_gemm_plain<<<(m*n+T-1)/T, T>>>(dA, dB, dC, m, k, n);
    dim3 grid(m / r, n / r);
    k_noised_transcript<<<grid, 256>>>(dAn, dBn, dCscr, dTr, m, k, n, r, th, tw);
    k_pow_scan<<<1, 1>>>(dTr, dKey, dTgt, m, n, r, th, tw, dFound, dARow, dBCol, dTrOut);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(C_out, dC, m*n*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(found_out, dFound, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(a_row_out, dARow, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b_col_out, dBCol, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(transcript_out, dTrOut, 16*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(dA);cudaFree(dB);cudaFree(dEAL);cudaFree(dEAR);cudaFree(dEBL);cudaFree(dEBR);
    cudaFree(dAn);cudaFree(dBn);cudaFree(dC);cudaFree(dCscr);cudaFree(dTr);cudaFree(dKey);
    cudaFree(dTgt);cudaFree(dTrOut);cudaFree(dFound);cudaFree(dARow);cudaFree(dBCol);
    return 0;
}

} // extern "C"

// one-thread blake3 kernel used by prl_blake3_64
__global__ void k_blake3_one(const uint32_t* msg, const uint32_t* key, uint32_t* out) {
    if (threadIdx.x == 0) blake3_keyed_block(msg, key, out);
}
