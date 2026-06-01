// FUSED tensor-core mining kernel for sm_86 (production config: noise_rank=128).
//
// This is the ACTUAL miner hot loop on tensor cores, not a proxy GEMM:
//   noise A/B  ->  noised int8 GEMM on mma.sync.m16n8k32.s8 (cp.async double-buffered)
//   ->  per-128-K-chunk inner-hash transcript (warp-XOR-reduced straight from the mma
//       accumulator registers)  ->  on-device keyed-BLAKE3 PoW vs target.
// Validated bit-for-bit against the FULL golden vectors (found / winning indices /
// 16 transcript words) for the noise_rank=128 cases (g4/g5) on the RTX 3090.
//
// Specialized for r=128: block tile = 128x128 = the mining output tile; 8 warps (4x2),
// each warp owns a 32x64 region = 8 of the 64 (16x16) hash sub-tiles. K-chunk = 128 =
// 4 mma k-steps of 32. Requires m%128==n%128==k%128==0, hash_tile=16.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ---------- device BLAKE3 (single 64-byte keyed block) ----------
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ void gmix(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    s[a]=s[a]+s[b]+mx; s[d]=rotr32(s[d]^s[a],16); s[c]=s[c]+s[d]; s[b]=rotr32(s[b]^s[c],12);
    s[a]=s[a]+s[b]+my; s[d]=rotr32(s[d]^s[a], 8); s[c]=s[c]+s[d]; s[b]=rotr32(s[b]^s[c], 7);
}
__constant__ uint8_t MSG_SCHED[7][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
    {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},{10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
    {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},{9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
    {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};
__constant__ uint32_t B3IV[8] = {0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,
                                 0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
__device__ void blake3_keyed_block(const uint32_t* msg, const uint32_t* key, uint32_t* out) {
    uint32_t s[16];
    for (int i=0;i<8;i++) s[i]=key[i];
    s[8]=B3IV[0]; s[9]=B3IV[1]; s[10]=B3IV[2]; s[11]=B3IV[3];
    s[12]=0; s[13]=0; s[14]=64; s[15]=1u|2u|8u|16u;   // CHUNK_START|CHUNK_END|ROOT|KEYED
    for (int r=0;r<7;r++){ const uint8_t* sc=MSG_SCHED[r];
        gmix(s,0,4,8,12,msg[sc[0]],msg[sc[1]]);  gmix(s,1,5,9,13,msg[sc[2]],msg[sc[3]]);
        gmix(s,2,6,10,14,msg[sc[4]],msg[sc[5]]); gmix(s,3,7,11,15,msg[sc[6]],msg[sc[7]]);
        gmix(s,0,5,10,15,msg[sc[8]],msg[sc[9]]); gmix(s,1,6,11,12,msg[sc[10]],msg[sc[11]]);
        gmix(s,2,7,8,13,msg[sc[12]],msg[sc[13]]);gmix(s,3,4,9,14,msg[sc[14]],msg[sc[15]]); }
    for (int i=0;i<8;i++) out[i]=s[i]^s[i+8];
}

__device__ __forceinline__ uint32_t pk(int8_t b0,int8_t b1,int8_t b2,int8_t b3){
    return (uint32_t)(uint8_t)b0|((uint32_t)(uint8_t)b1<<8)|((uint32_t)(uint8_t)b2<<16)|((uint32_t)(uint8_t)b3<<24);
}
__device__ __forceinline__ void cp_async16(void* s,const void* gp){
    unsigned a=(unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(gp)); }
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait1(){ asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void cp_wait0(){ asm volatile("cp.async.wait_group 0;\n"); }
#define IDX(r,c,ld) ((size_t)(r)*(ld)+(c))

// ---------- noising (simple element-wise; r = noise_rank = 128) ----------
__global__ void k_noiseA(const int8_t* A,const int8_t* EAL,const int8_t* EAR,int8_t* An,int m,int k,int r){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)m*k) return;
    int i=idx/k,j=idx%k; int e=0;
    for(int c=0;c<r;c++) e+=(int)EAL[IDX(i,c,r)]*(int)EAR[IDX(c,j,k)];
    An[idx]=(int8_t)((int)A[idx]+e);
}
__global__ void k_noiseB(const int8_t* B,const int8_t* EBL,const int8_t* EBR,int8_t* Bn,int k,int n,int r){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)k*n) return;
    int i=idx/n,j=idx%n; int e=0;
    for(int c=0;c<r;c++) e+=(int)EBL[IDX(i,c,r)]*(int)EBR[IDX(c,j,n)];
    Bn[idx]=(int8_t)((int)B[idx]+e);
}

// ---------- fused noised GEMM + transcript (r=128) ----------
__global__ void k_mine(const int8_t* An,const int8_t* Bn,uint32_t* transcripts,int m,int k,int n){
    __shared__ int8_t As[2][128*32];
    __shared__ int8_t Bs[2][32*128];
    int tid=threadIdx.x, lane=tid&31, wid=tid>>5;
    int wr=wid>>1, wc=wid&1, g=lane>>2, t=lane&3;
    int br=blockIdx.x*128, bc=blockIdx.y*128;
    int nk=k>>5;                 // k/32 mma steps
    int ST_cols=n>>4;            // n/16
    int acc[2][8][4];
    #pragma unroll
    for(int i=0;i<2;i++) for(int j=0;j<8;j++) for(int e=0;e<4;e++) acc[i][j][e]=0;

#define LOAD(s,koff) do{ int ar=tid>>1,ah=tid&1; \
        cp_async16(&As[s][ar*32+ah*16],&An[IDX(br+ar,(koff)+ah*16,k)]); \
        int br_=tid>>3,bp=tid&7; \
        cp_async16(&Bs[s][br_*128+bp*16],&Bn[IDX((koff)+br_,bc+bp*16,n)]); }while(0)

    LOAD(0,0); cp_commit();
    for(int ks=0;ks<nk;ks++){
        int cur=ks&1;
        if(ks+1<nk){ LOAD((ks+1)&1,(ks+1)*32); cp_commit(); cp_wait1(); } else cp_wait0();
        __syncthreads();
        // B fragments are independent of mrow — compute all 8 ncol ONCE and reuse across
        // both mrow (halves the strided B SMEM loads, the bank-conflicted bottleneck).
        const int8_t* Bb=&Bs[cur][0];
        uint32_t bf0[8], bf1[8];
        #pragma unroll
        for(int ncol=0;ncol<8;ncol++){
            int bcol=wc*64+ncol*8+g;
            bf0[ncol]=pk(Bb[(t*4)*128+bcol],Bb[(t*4+1)*128+bcol],Bb[(t*4+2)*128+bcol],Bb[(t*4+3)*128+bcol]);
            bf1[ncol]=pk(Bb[(t*4+16)*128+bcol],Bb[(t*4+17)*128+bcol],Bb[(t*4+18)*128+bcol],Bb[(t*4+19)*128+bcol]);
        }
        #pragma unroll
        for(int mrow=0;mrow<2;mrow++){
            int arow=wr*32+mrow*16;
            const int8_t* Ar0=&As[cur][(arow+g)*32];
            const int8_t* Ar8=&As[cur][(arow+g+8)*32];
            uint32_t a0=*(const uint32_t*)(Ar0+t*4);
            uint32_t a1=*(const uint32_t*)(Ar8+t*4);
            uint32_t a2=*(const uint32_t*)(Ar0+t*4+16);
            uint32_t a3=*(const uint32_t*)(Ar8+t*4+16);
            #pragma unroll
            for(int ncol=0;ncol<8;ncol++){
                int* c=acc[mrow][ncol];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                  :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bf0[ncol]),"r"(bf1[ncol]));
            }
        }
        __syncthreads();
        // End of a 128-wide K-chunk every 4 mma steps: hash the cumulative accumulator.
        if(((ks+1)&3)==0){
            int chunk=((ks+1)>>2)-1, slot=chunk&15;
            #pragma unroll
            for(int mrow=0;mrow<2;mrow++){
                #pragma unroll
                for(int hcol=0;hcol<4;hcol++){
                    int* ca=acc[mrow][2*hcol]; int* cb=acc[mrow][2*hcol+1];
                    uint32_t loc=(uint32_t)(ca[0]^ca[1]^ca[2]^ca[3]^cb[0]^cb[1]^cb[2]^cb[3]);
                    #pragma unroll
                    for(int off=16;off>0;off>>=1) loc^=__shfl_xor_sync(0xffffffffu,loc,off);
                    if(lane==0){
                        int sr=br/16 + wr*2 + mrow;     // absolute 16x16 sub-tile row
                        int sc=bc/16 + wc*4 + hcol;      // absolute sub-tile col
                        uint32_t* tr=&transcripts[((size_t)sr*ST_cols+sc)*16+slot];
                        *tr=((*tr<<13)|(*tr>>19))^loc;   // rotl32(.,13) ^ inner_hash
                    }
                }
            }
        }
    }
#undef LOAD
}

// ---------- PoW scan (reference order, device BLAKE3) ----------
__device__ __forceinline__ void pow_scan_index_to_tile(int idx, int TJ, int SH, int SW,
                                                       int* sr, int* sc) {
    int wi = idx % SW; idx /= SW;
    int hi = idx % SH; idx /= SH;
    int tj = idx % TJ; idx /= TJ;
    int ti = idx;
    *sr = ti * SH + hi;
    *sc = tj * SW + wi;
}

__device__ __forceinline__ bool digest_le_target(const uint32_t* digest, const uint32_t* target) {
    for (int l = 7; l >= 0; --l) {
        if (digest[l] < target[l]) return true;
        if (digest[l] > target[l]) return false;
    }
    return true;
}

__global__ void k_pow_scan_find(const uint32_t* transcripts, const uint32_t* key,
                                const uint32_t* target, int m, int n, int r, int th,
                                int tw, int* best_idx) {
    int ST_cols = n / tw, TI = m / r, TJ = n / r, SH = r / th, SW = r / tw;
    int total = TI * TJ * SH * SW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int sr, sc;
    pow_scan_index_to_tile(idx, TJ, SH, SW, &sr, &sc);
    const uint32_t* tr = &transcripts[((size_t)sr * ST_cols + sc) * 16];
    uint32_t dig[8];
    blake3_keyed_block(tr, key, dig);
    if (digest_le_target(dig, target)) atomicMin(best_idx, idx);
}

__global__ void k_pow_scan_emit(const uint32_t* transcripts, int m, int n, int r, int th,
                                int tw, const int* best_idx, int* found, int* a_row,
                                int* b_col, uint32_t* tr_out) {
    if (threadIdx.x || blockIdx.x) return;
    int ST_cols = n / tw, TI = m / r, TJ = n / r, SH = r / th, SW = r / tw;
    int total = TI * TJ * SH * SW;
    int idx = *best_idx;
    if (idx >= total) {
        *found = 0;
        *a_row = -1;
        *b_col = -1;
        return;
    }
    int sr, sc;
    pow_scan_index_to_tile(idx, TJ, SH, SW, &sr, &sc);
    const uint32_t* tr = &transcripts[((size_t)sr * ST_cols + sc) * 16];
    *found = 1;
    *a_row = sr * th;
    *b_col = sc * tw;
    for (int x = 0; x < 16; x++) tr_out[x] = tr[x];
}

// ---------- GPU noise generation (faithful to noise_generation.py) ----------
// Removes the per-attempt Python BLAKE3 noise loop (the 68% bottleneck) by deriving
// E_AL/E_AR/E_BL/E_BR on-device from the 32-byte keys, exactly as NoiseGenerator does.
__device__ __forceinline__ uint32_t mulhi_u32(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * b) >> 32);
}
// Dense uniform int8 in [-32,31]: byte&63 - 32 (noise_generation.py:107-135). prepend slot 0.
__global__ void k_ng_dense(const uint32_t* key, const uint32_t* seed, int total, int8_t* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // draw index
    if (i * 32 >= total) return;
    uint32_t m[16];
    #pragma unroll
    for (int j = 0; j < 8; j++) m[j] = 0;
    m[0] = (uint32_t)(1 + i);
    #pragma unroll
    for (int j = 0; j < 8; j++) m[8 + j] = seed[j];
    uint32_t dig[8]; blake3_keyed_block(m, key, dig);
    #pragma unroll
    for (int b = 0; b < 32; b++) {
        int idx = i * 32 + b;
        if (idx < total) { uint8_t by = (dig[b >> 2] >> ((b & 3) * 8)) & 0xFF; out[idx] = (int8_t)((by & 63) - 32); }
    }
}
// Signed-permutation: each line one +1 and one -1 (noise_generation.py:137-187). prepend slot 1.
__global__ void k_ng_perm(const uint32_t* key, const uint32_t* seed, int required, int assign_cols,
                          int r, int other, int8_t* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // draw index (8 lines per draw)
    if (i * 8 >= required) return;
    uint32_t m[16];
    #pragma unroll
    for (int j = 0; j < 8; j++) m[j] = 0;
    m[1] = (uint32_t)(1 + i);
    #pragma unroll
    for (int j = 0; j < 8; j++) m[8 + j] = seed[j];
    uint32_t dig[8]; blake3_keyed_block(m, key, dig);
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        int ai = i * 8 + kk;
        if (ai >= required) break;
        uint32_t u = dig[kk];
        int first = u & (r - 1);
        int second = first ^ (int)(1u + mulhi_u32((uint32_t)(r - 1), u));
        if (assign_cols) { out[first * other + ai] = 1; out[second * other + ai] = -1; }   // r x other
        else { out[ai * r + first] = 1; out[ai * r + second] = -1; }                        // other x r
    }
}
__global__ void k_transpose(const int8_t* U, int8_t* T, int rows, int cols) {  // U[rows][cols] -> T[cols][rows]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    T[(idx % cols) * rows + (idx / cols)] = U[idx];
}

// ---------- host C ABI ----------
static char g_err[256]={0};
#define CK(call) do{ cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    snprintf(g_err,sizeof(g_err),"%s @ %d: %s",#call,__LINE__,cudaGetErrorString(e_)); return 4; } }while(0)

extern "C" {
const char* prl_mine_last_error(void){ return g_err; }
int prl_mine_device_count(void){ int n=0; cudaGetDeviceCount(&n); return n; }

// Full mine: noise -> fused noised GEMM+transcript -> PoW. r fixed at 128 (production).
int prl_mine_run(const int8_t* A,const int8_t* B,const int8_t* EAL,const int8_t* EAR,
                 const int8_t* EBL,const int8_t* EBR,int m,int k,int n,
                 const uint8_t* key32,const uint8_t* target32,
                 int* found,int* a_row,int* b_col,uint32_t* tr_out){
    const int r=128;
    if(m%128||n%128||k%128){ snprintf(g_err,sizeof(g_err),"need m%%128==n%%128==k%%128==0 (r=128)"); return 2; }
    int8_t *dA,*dB,*dEAL,*dEAR,*dEBL,*dEBR,*dAn,*dBn; uint32_t *dTr,*dKey,*dTgt,*dTrOut; int *dF,*dAR,*dBC,*dBest;
    size_t ST=(size_t)(m/16)*(n/16);
    CK(cudaMalloc(&dA,(size_t)m*k));CK(cudaMalloc(&dB,(size_t)k*n));
    CK(cudaMalloc(&dEAL,(size_t)m*r));CK(cudaMalloc(&dEAR,(size_t)r*k));
    CK(cudaMalloc(&dEBL,(size_t)k*r));CK(cudaMalloc(&dEBR,(size_t)r*n));
    CK(cudaMalloc(&dAn,(size_t)m*k));CK(cudaMalloc(&dBn,(size_t)k*n));
    CK(cudaMalloc(&dTr,ST*16*4));CK(cudaMemset(dTr,0,ST*16*4));
    CK(cudaMalloc(&dKey,32));CK(cudaMalloc(&dTgt,32));CK(cudaMalloc(&dTrOut,64));
    CK(cudaMalloc(&dF,4));CK(cudaMalloc(&dAR,4));CK(cudaMalloc(&dBC,4));CK(cudaMalloc(&dBest,4));
    CK(cudaMemcpy(dA,A,(size_t)m*k,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,B,(size_t)k*n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEAL,EAL,(size_t)m*r,cudaMemcpyHostToDevice));CK(cudaMemcpy(dEAR,EAR,(size_t)r*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEBL,EBL,(size_t)k*r,cudaMemcpyHostToDevice));CK(cudaMemcpy(dEBR,EBR,(size_t)r*n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKey,key32,32,cudaMemcpyHostToDevice));CK(cudaMemcpy(dTgt,target32,32,cudaMemcpyHostToDevice));
    int T=256;
    k_noiseA<<<((size_t)m*k+T-1)/T,T>>>(dA,dEAL,dEAR,dAn,m,k,r);
    k_noiseB<<<((size_t)k*n+T-1)/T,T>>>(dB,dEBL,dEBR,dBn,k,n,r);
    dim3 grid(m/128,n/128);
    k_mine<<<grid,256>>>(dAn,dBn,dTr,m,k,n);
    CK(cudaMemset(dBest,0x7f,4));
    int total_tiles=(m/128)*(n/128)*8*8;
    k_pow_scan_find<<<(total_tiles+T-1)/T,T>>>(dTr,dKey,dTgt,m,n,128,16,16,dBest);
    k_pow_scan_emit<<<1,1>>>(dTr,m,n,128,16,16,dBest,dF,dAR,dBC,dTrOut);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(found,dF,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(a_row,dAR,4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b_col,dBC,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(tr_out,dTrOut,64,cudaMemcpyDeviceToHost));
    cudaFree(dA);cudaFree(dB);cudaFree(dEAL);cudaFree(dEAR);cudaFree(dEBL);cudaFree(dEBR);
    cudaFree(dAn);cudaFree(dBn);cudaFree(dTr);cudaFree(dKey);cudaFree(dTgt);cudaFree(dTrOut);
    cudaFree(dF);cudaFree(dAR);cudaFree(dBC);cudaFree(dBest);
    return 0;
}

// Bench just the fused k_mine kernel (the GEMM+transcript hot loop) -> avg ms.
int prl_mine_bench(int m,int k,int n,int iters,double* out_ms){
    if(m%128||n%128||k%128){ snprintf(g_err,sizeof(g_err),"need m%%128==n%%128==k%%128==0"); return 2; }
    int8_t *dAn,*dBn; uint32_t* dTr; size_t ST=(size_t)(m/16)*(n/16);
    CK(cudaMalloc(&dAn,(size_t)m*k));CK(cudaMalloc(&dBn,(size_t)k*n));CK(cudaMalloc(&dTr,ST*16*4));
    CK(cudaMemset(dAn,1,(size_t)m*k));CK(cudaMemset(dBn,1,(size_t)k*n));CK(cudaMemset(dTr,0,ST*16*4));
    dim3 grid(m/128,n/128);
    k_mine<<<grid,256>>>(dAn,dBn,dTr,m,k,n); CK(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
    for(int i=0;i<iters;i++) k_mine<<<grid,256>>>(dAn,dBn,dTr,m,k,n);
    cudaEventRecord(e); CK(cudaEventSynchronize(e));
    float ms=0; cudaEventElapsedTime(&ms,s,e); *out_ms=(double)ms/iters;
    cudaEventDestroy(s);cudaEventDestroy(e); cudaFree(dAn);cudaFree(dBn);cudaFree(dTr); return 0;
}

// Generate the four noise matrices on-GPU from the two 32-byte keys (validation entry).
int prl_noisegen(const uint8_t* keyA, const uint8_t* keyB, int m, int k, int n,
                 int8_t* EAL, int8_t* EAR, int8_t* EBL, int8_t* EBR) {
    const int r = 128;
    uint32_t hsa[8] = {0}, hsb[8] = {0};
    { unsigned char s[32] = {'A','_','t','e','n','s','o','r'}; memcpy(hsa, s, 32); }
    { unsigned char s[32] = {'B','_','t','e','n','s','o','r'}; memcpy(hsb, s, 32); }
    uint32_t *dKA,*dKB,*dSA,*dSB; int8_t *dEAL,*dEAR,*dEBL,*dEBR,*dU;
    CK(cudaMalloc(&dKA,32)); CK(cudaMalloc(&dKB,32)); CK(cudaMalloc(&dSA,32)); CK(cudaMalloc(&dSB,32));
    CK(cudaMalloc(&dEAL,(size_t)m*r)); CK(cudaMalloc(&dEAR,(size_t)r*k));
    CK(cudaMalloc(&dEBL,(size_t)k*r)); CK(cudaMalloc(&dEBR,(size_t)r*n)); CK(cudaMalloc(&dU,(size_t)n*r));
    CK(cudaMemcpy(dKA,keyA,32,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dKB,keyB,32,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSA,hsa,32,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dSB,hsb,32,cudaMemcpyHostToDevice));
    CK(cudaMemset(dEAR,0,(size_t)r*k)); CK(cudaMemset(dEBL,0,(size_t)k*r));
    int dA=((m*r+31)/32), dU2=((n*r+31)/32), pk=((k+7)/8), T=256;
    k_ng_dense<<<(dA+T-1)/T,T>>>(dKA,dSA,m*r,dEAL);
    k_ng_perm<<<(pk+T-1)/T,T>>>(dKA,dSA,k,1,r,k,dEAR);
    k_ng_perm<<<(pk+T-1)/T,T>>>(dKB,dSB,k,0,r,k,dEBL);
    k_ng_dense<<<(dU2+T-1)/T,T>>>(dKB,dSB,n*r,dU);
    k_transpose<<<((size_t)n*r+T-1)/T,T>>>(dU,dEBR,n,r);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(EAL,dEAL,(size_t)m*r,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(EAR,dEAR,(size_t)r*k,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(EBL,dEBL,(size_t)k*r,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(EBR,dEBR,(size_t)r*n,cudaMemcpyDeviceToHost));
    cudaFree(dKA);cudaFree(dKB);cudaFree(dSA);cudaFree(dSB);
    cudaFree(dEAL);cudaFree(dEAR);cudaFree(dEBL);cudaFree(dEBR);cudaFree(dU);
    return 0;
}

// ---- Persistent context: malloc device buffers + a stream ONCE, reuse per attempt ----
// This removes the per-attempt cudaMalloc/cudaFree that caps the orchestrator loop (the
// kernel itself is ~38 TOPS). All work runs on the context's stream with async copies.
typedef struct PrlMineCtx {
    int m, k, n; size_t ST;
    int8_t *dA,*dB,*dEAL,*dEAR,*dEBL,*dEBR,*dAn,*dBn,*dU;
    uint32_t *dTr,*dKey,*dKeyB,*dTgt,*dTrOut,*dSA,*dSB; int *dF,*dAR,*dBC,*dBest;
    cudaStream_t stream;
} PrlMineCtx;

void* prl_mine_ctx_create(int m, int k, int n) {
    if (m%128 || n%128 || k%128) { snprintf(g_err,sizeof(g_err),"ctx: need m%%128==n%%128==k%%128==0"); return nullptr; }
    PrlMineCtx* c = (PrlMineCtx*)calloc(1, sizeof(PrlMineCtx));
    const int r = 128; c->m=m; c->k=k; c->n=n; c->ST=(size_t)(m/16)*(n/16);
    bool ok = cudaMalloc(&c->dA,(size_t)m*k)==cudaSuccess
        && cudaMalloc(&c->dB,(size_t)k*n)==cudaSuccess
        && cudaMalloc(&c->dEAL,(size_t)m*r)==cudaSuccess && cudaMalloc(&c->dEAR,(size_t)r*k)==cudaSuccess
        && cudaMalloc(&c->dEBL,(size_t)k*r)==cudaSuccess && cudaMalloc(&c->dEBR,(size_t)r*n)==cudaSuccess
        && cudaMalloc(&c->dAn,(size_t)m*k)==cudaSuccess && cudaMalloc(&c->dBn,(size_t)k*n)==cudaSuccess
        && cudaMalloc(&c->dTr,c->ST*64)==cudaSuccess && cudaMalloc(&c->dKey,32)==cudaSuccess
        && cudaMalloc(&c->dTgt,32)==cudaSuccess && cudaMalloc(&c->dTrOut,64)==cudaSuccess
        && cudaMalloc(&c->dF,4)==cudaSuccess && cudaMalloc(&c->dAR,4)==cudaSuccess
        && cudaMalloc(&c->dBC,4)==cudaSuccess && cudaMalloc(&c->dBest,4)==cudaSuccess
        && cudaMalloc(&c->dKeyB,32)==cudaSuccess && cudaMalloc(&c->dSA,32)==cudaSuccess
        && cudaMalloc(&c->dSB,32)==cudaSuccess && cudaMalloc(&c->dU,(size_t)n*r)==cudaSuccess
        && cudaStreamCreate(&c->stream)==cudaSuccess;
    if (!ok) { snprintf(g_err,sizeof(g_err),"ctx_create: alloc failed"); return nullptr; }
    uint32_t hsa[8]={0}, hsb[8]={0};
    { unsigned char s[32]={'A','_','t','e','n','s','o','r'}; memcpy(hsa,s,32); }
    { unsigned char s[32]={'B','_','t','e','n','s','o','r'}; memcpy(hsb,s,32); }
    cudaMemcpy(c->dSA,hsa,32,cudaMemcpyHostToDevice);
    cudaMemcpy(c->dSB,hsb,32,cudaMemcpyHostToDevice);
    return c;
}

int prl_mine_ctx_run(void* h, const int8_t* A, const int8_t* B, const int8_t* EAL, const int8_t* EAR,
                     const int8_t* EBL, const int8_t* EBR, const uint8_t* key32, const uint8_t* target32,
                     int* found, int* a_row, int* b_col, uint32_t* tr_out) {
    PrlMineCtx* c = (PrlMineCtx*)h; if (!c) return 2;
    int m=c->m, k=c->k, n=c->n; const int r=128; cudaStream_t s=c->stream;
    cudaMemcpyAsync(c->dA,A,(size_t)m*k,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dB,B,(size_t)k*n,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dEAL,EAL,(size_t)m*r,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dEAR,EAR,(size_t)r*k,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dEBL,EBL,(size_t)k*r,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dEBR,EBR,(size_t)r*n,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dKey,key32,32,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dTgt,target32,32,cudaMemcpyHostToDevice,s);
    cudaMemsetAsync(c->dTr,0,c->ST*64,s);
    int T=256;
    k_noiseA<<<((size_t)m*k+T-1)/T,T,0,s>>>(c->dA,c->dEAL,c->dEAR,c->dAn,m,k,r);
    k_noiseB<<<((size_t)k*n+T-1)/T,T,0,s>>>(c->dB,c->dEBL,c->dEBR,c->dBn,k,n,r);
    dim3 grid(m/128,n/128);
    k_mine<<<grid,256,0,s>>>(c->dAn,c->dBn,c->dTr,m,k,n);
    cudaMemsetAsync(c->dBest,0x7f,4,s);
    int total_tiles=(m/128)*(n/128)*8*8;
    k_pow_scan_find<<<(total_tiles+T-1)/T,T,0,s>>>(c->dTr,c->dKey,c->dTgt,m,n,128,16,16,c->dBest);
    k_pow_scan_emit<<<1,1,0,s>>>(c->dTr,m,n,128,16,16,c->dBest,c->dF,c->dAR,c->dBC,c->dTrOut);
    cudaMemcpyAsync(found,c->dF,4,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(a_row,c->dAR,4,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(b_col,c->dBC,4,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(tr_out,c->dTrOut,64,cudaMemcpyDeviceToHost,s);
    cudaStreamSynchronize(s);
    cudaError_t e=cudaGetLastError();
    if (e!=cudaSuccess){ snprintf(g_err,sizeof(g_err),"ctx_run: %s",cudaGetErrorString(e)); return 4; }
    return 0;
}

// Keyed run: generate noise ON-GPU from the two keys (no Python BLAKE3), then mine.
int prl_mine_ctx_run_keyed(void* h, const int8_t* A, const int8_t* B,
                           const uint8_t* keyA32, const uint8_t* keyB32, const uint8_t* target32,
                           int* found, int* a_row, int* b_col, uint32_t* tr_out) {
    PrlMineCtx* c=(PrlMineCtx*)h; if(!c) return 2;
    int m=c->m, k=c->k, n=c->n; const int r=128; cudaStream_t s=c->stream; int T=256;
    cudaMemcpyAsync(c->dA,A,(size_t)m*k,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dB,B,(size_t)k*n,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dKey,keyA32,32,cudaMemcpyHostToDevice,s);   // key_A doubles as the PoW key
    cudaMemcpyAsync(c->dKeyB,keyB32,32,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(c->dTgt,target32,32,cudaMemcpyHostToDevice,s);
    cudaMemsetAsync(c->dTr,0,c->ST*64,s);
    cudaMemsetAsync(c->dEAR,0,(size_t)r*k,s);
    cudaMemsetAsync(c->dEBL,0,(size_t)k*r,s);
    // GPU noise generation from keys (replaces the Python BLAKE3 loop — the 68% bottleneck)
    k_ng_dense<<<((((m*r+31)/32))+T-1)/T,T,0,s>>>(c->dKey,c->dSA,m*r,c->dEAL);
    k_ng_perm<<<((((k+7)/8))+T-1)/T,T,0,s>>>(c->dKey,c->dSA,k,1,r,k,c->dEAR);
    k_ng_perm<<<((((k+7)/8))+T-1)/T,T,0,s>>>(c->dKeyB,c->dSB,k,0,r,k,c->dEBL);
    k_ng_dense<<<((((n*r+31)/32))+T-1)/T,T,0,s>>>(c->dKeyB,c->dSB,n*r,c->dU);
    k_transpose<<<(((size_t)n*r)+T-1)/T,T,0,s>>>(c->dU,c->dEBR,n,r);
    k_noiseA<<<(((size_t)m*k)+T-1)/T,T,0,s>>>(c->dA,c->dEAL,c->dEAR,c->dAn,m,k,r);
    k_noiseB<<<(((size_t)k*n)+T-1)/T,T,0,s>>>(c->dB,c->dEBL,c->dEBR,c->dBn,k,n,r);
    dim3 grid(m/128,n/128);
    k_mine<<<grid,256,0,s>>>(c->dAn,c->dBn,c->dTr,m,k,n);
    cudaMemsetAsync(c->dBest,0x7f,4,s);
    int total_tiles=(m/128)*(n/128)*8*8;
    k_pow_scan_find<<<(total_tiles+T-1)/T,T,0,s>>>(c->dTr,c->dKey,c->dTgt,m,n,128,16,16,c->dBest);
    k_pow_scan_emit<<<1,1,0,s>>>(c->dTr,m,n,128,16,16,c->dBest,c->dF,c->dAR,c->dBC,c->dTrOut);
    cudaMemcpyAsync(found,c->dF,4,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(a_row,c->dAR,4,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(b_col,c->dBC,4,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(tr_out,c->dTrOut,64,cudaMemcpyDeviceToHost,s);
    cudaStreamSynchronize(s);
    cudaError_t e=cudaGetLastError();
    if (e!=cudaSuccess){ snprintf(g_err,sizeof(g_err),"ctx_run_keyed: %s",cudaGetErrorString(e)); return 4; }
    return 0;
}

void prl_mine_ctx_destroy(void* h) {
    PrlMineCtx* c = (PrlMineCtx*)h; if (!c) return;
    cudaFree(c->dA); cudaFree(c->dB); cudaFree(c->dEAL); cudaFree(c->dEAR);
    cudaFree(c->dEBL); cudaFree(c->dEBR); cudaFree(c->dAn); cudaFree(c->dBn); cudaFree(c->dU);
    cudaFree(c->dTr); cudaFree(c->dKey); cudaFree(c->dKeyB); cudaFree(c->dTgt); cudaFree(c->dTrOut);
    cudaFree(c->dSA); cudaFree(c->dSB);
    cudaFree(c->dF); cudaFree(c->dAR); cudaFree(c->dBC); cudaFree(c->dBest); cudaStreamDestroy(c->stream); free(c);
}
} // extern "C"
