// PRL-3090 CUDA backend — data contract.
//
// These structs mirror exactly what a golden vector contains (tests/golden/*.npz +
// manifest.json), so the correctness harness (cuda/tests/test_cuda_golden.py) maps
// 1:1 onto the C ABI in prl_cuda.h. Layouts and semantics are grounded in
// docs/protocol-notes.md §4 and the official miner-base reference.
#ifndef PRL_TYPES_H
#define PRL_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All matrices are row-major. Data values are int7 [-64,63]; noise is int8.
// See protocol-notes.md §4.1 for dtypes/shapes.
typedef struct PrlMiningJob {
    int32_t m, k, n;            // C = A(m x k) @ B(k x n)
    int32_t noise_rank;         // r (multiple of 32)
    int32_t noise_range;        // 128 in production
    int32_t hash_tile_h, hash_tile_w;     // 16 x 16
    int32_t matmul_tile_h, matmul_tile_w; // 128 x 128/256 in production

    const int8_t* A;            // m x k
    const int8_t* B;            // k x n
    const int8_t* E_AL;         // m x r   (dense uniform)
    const int8_t* E_AR;         // r x k   (signed permutation, per column)
    const int8_t* E_BL;         // k x r   (signed permutation, per row)
    const int8_t* E_BR;         // r x n   (dense uniform)

    uint8_t  pow_key[32];       // = commitment_A (noise_seed_A); keyed-BLAKE3 key
    uint8_t  pow_target[32];    // uint256, little-endian; win iff hash <= target
} PrlMiningJob;

typedef struct PrlMiningResult {
    int32_t* C;                 // m x n int32 output (== A@B exactly; caller-allocated)
    int32_t  found;             // 0/1
    int32_t  a_row_start;       // winning 16x16 tile origin (row); -1 if not found
    int32_t  b_col_start;       // winning tile origin (col);     -1 if not found
    uint32_t transcript[16];    // winning tile's 64-byte PoW transcript (16 x u32 LE)
    double   kernel_ms;         // measured device time for this call
} PrlMiningResult;

typedef struct PrlDeviceInfo {
    int32_t index;
    char    name[256];
    int32_t cc_major, cc_minor; // RTX 3090 -> 8.6
    uint64_t total_mem_bytes;
    int32_t  smem_per_block_optin; // Ampere sm_86: ~99 KB (vs Hopper 227 KB)
    float    temp_c, vram_temp_c, power_w;
    int32_t  sm_clock_mhz, mem_clock_mhz;
} PrlDeviceInfo;

#ifdef __cplusplus
}
#endif
#endif // PRL_TYPES_H
