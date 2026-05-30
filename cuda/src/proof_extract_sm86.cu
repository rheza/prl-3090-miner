// Inner-hash / transcript / PoW extraction.  ***SCAFFOLD — M4 (mostly portable).***
//
// This path is portable to sm_86 with little change (cuda-sm86-port.md §4): it is
// integer-only. The keyed-BLAKE3 device implementation should be reused verbatim from
// pearl-official/miner/pearl-gemm/csrc/blake3/blake3.cu (vendored via CMake), and the
// inner-hash is a plain XOR-reduction (inner_hash.py:7-14).
//
// Algorithm (protocol-notes.md §4.3 steps 4-7):
//   inner_hash(tile)  = XOR over the 16x16 int32 tile reinterpreted as u32  (order-independent)
//   transcript[c%16]  = rotl32(transcript[c%16], 13) ^ inner_hash   (c = k-reduction index)
//   pow              : blake3(transcript[16] as 64 LE bytes, key=pow_key) <= pow_target (u256 LE)
#include "prl_cuda.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace prl {

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// XOR-reduce a 16x16 int32 tile (viewed as u32). One value per hash tile.
// (Full warp/block reduction left for M4; this documents the contract.)
__device__ uint32_t inner_hash_tile(const int32_t* tile, int ld) {
    uint32_t acc = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            acc ^= static_cast<uint32_t>(tile[i * ld + j]);
    return acc;
}

// Fold an inner hash into a per-tile transcript at the cycling position.
__device__ void transcript_accumulate(uint32_t* transcript, int reduction_count, uint32_t h) {
    int idx = reduction_count & 15;                 // % TRANSCRIPT_SIZE_U32 (16)
    transcript[idx] = rotl32(transcript[idx], 13) ^ h;   // HASH_ACCUMULATE_ROTATION
}

// PoW check: keyed BLAKE3 over the 64-byte transcript, compared to the u256 target.
// blake3_keyed(...) comes from the vendored blake3.cu (TODO link in CMake).
extern "C" __device__ int prl_pow_meets_target(const uint32_t* /*transcript16*/,
                                               const uint8_t* /*pow_key32*/,
                                               const uint8_t* /*target32_le*/) {
    // TODO(M4): blake3_keyed(transcript_bytes, key) -> digest; return digest <= target (LE).
    return 0;
}

} // namespace prl
