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
        #pragma unroll
        for(int mrow=0;mrow<2;mrow++){
            int arow=wr*32+mrow*16;
            const int8_t* Ar0=&As[cur][(arow+g)*32];
            const int8_t* Ar8=&As[cur][(arow+g+8)*32];
            uint32_t a0=pk(Ar0[t*4],Ar0[t*4+1],Ar0[t*4+2],Ar0[t*4+3]);
            uint32_t a1=pk(Ar8[t*4],Ar8[t*4+1],Ar8[t*4+2],Ar8[t*4+3]);
            uint32_t a2=pk(Ar0[t*4+16],Ar0[t*4+17],Ar0[t*4+18],Ar0[t*4+19]);
            uint32_t a3=pk(Ar8[t*4+16],Ar8[t*4+17],Ar8[t*4+18],Ar8[t*4+19]);
            const int8_t* Bb=&Bs[cur][0];
            #pragma unroll
            for(int ncol=0;ncol<8;ncol++){
                int bcol=wc*64+ncol*8+g;
                uint32_t b0=pk(Bb[(t*4)*128+bcol],Bb[(t*4+1)*128+bcol],Bb[(t*4+2)*128+bcol],Bb[(t*4+3)*128+bcol]);
                uint32_t b1=pk(Bb[(t*4+16)*128+bcol],Bb[(t*4+17)*128+bcol],Bb[(t*4+18)*128+bcol],Bb[(t*4+19)*128+bcol]);
                int* c=acc[mrow][ncol];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                  :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
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
__global__ void k_pow_scan(const uint32_t* transcripts,const uint32_t* key,const uint32_t* target,
                           int m,int n,int r,int th,int tw,int* found,int* a_row,int* b_col,uint32_t* tr_out){
    if(threadIdx.x||blockIdx.x) return;
    int ST_cols=n/tw, TI=m/r, TJ=n/r, SH=r/th, SW=r/tw;
    *found=0; *a_row=-1; *b_col=-1;
    for(int ti=0;ti<TI;ti++) for(int tj=0;tj<TJ;tj++) for(int hi=0;hi<SH;hi++) for(int wi=0;wi<SW;wi++){
        int sr=ti*SH+hi, sc=tj*SW+wi;
        const uint32_t* tr=&transcripts[((size_t)sr*ST_cols+sc)*16];
        uint32_t dig[8]; blake3_keyed_block(tr,key,dig);
        bool le=true;
        for(int l=7;l>=0;l--){ if(dig[l]<target[l]){le=true;break;} if(dig[l]>target[l]){le=false;break;} }
        if(le){ *found=1; *a_row=sr*th; *b_col=sc*tw; for(int x=0;x<16;x++) tr_out[x]=tr[x]; return; }
    }
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
    int8_t *dA,*dB,*dEAL,*dEAR,*dEBL,*dEBR,*dAn,*dBn; uint32_t *dTr,*dKey,*dTgt,*dTrOut; int *dF,*dAR,*dBC;
    size_t ST=(size_t)(m/16)*(n/16);
    CK(cudaMalloc(&dA,(size_t)m*k));CK(cudaMalloc(&dB,(size_t)k*n));
    CK(cudaMalloc(&dEAL,(size_t)m*r));CK(cudaMalloc(&dEAR,(size_t)r*k));
    CK(cudaMalloc(&dEBL,(size_t)k*r));CK(cudaMalloc(&dEBR,(size_t)r*n));
    CK(cudaMalloc(&dAn,(size_t)m*k));CK(cudaMalloc(&dBn,(size_t)k*n));
    CK(cudaMalloc(&dTr,ST*16*4));CK(cudaMemset(dTr,0,ST*16*4));
    CK(cudaMalloc(&dKey,32));CK(cudaMalloc(&dTgt,32));CK(cudaMalloc(&dTrOut,64));
    CK(cudaMalloc(&dF,4));CK(cudaMalloc(&dAR,4));CK(cudaMalloc(&dBC,4));
    CK(cudaMemcpy(dA,A,(size_t)m*k,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,B,(size_t)k*n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEAL,EAL,(size_t)m*r,cudaMemcpyHostToDevice));CK(cudaMemcpy(dEAR,EAR,(size_t)r*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dEBL,EBL,(size_t)k*r,cudaMemcpyHostToDevice));CK(cudaMemcpy(dEBR,EBR,(size_t)r*n,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKey,key32,32,cudaMemcpyHostToDevice));CK(cudaMemcpy(dTgt,target32,32,cudaMemcpyHostToDevice));
    int T=256;
    k_noiseA<<<((size_t)m*k+T-1)/T,T>>>(dA,dEAL,dEAR,dAn,m,k,r);
    k_noiseB<<<((size_t)k*n+T-1)/T,T>>>(dB,dEBL,dEBR,dBn,k,n,r);
    dim3 grid(m/128,n/128);
    k_mine<<<grid,256>>>(dAn,dBn,dTr,m,k,n);
    k_pow_scan<<<1,1>>>(dTr,dKey,dTgt,m,n,128,16,16,dF,dAR,dBC,dTrOut);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(found,dF,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(a_row,dAR,4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b_col,dBC,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(tr_out,dTrOut,64,cudaMemcpyDeviceToHost));
    cudaFree(dA);cudaFree(dB);cudaFree(dEAL);cudaFree(dEAR);cudaFree(dEBL);cudaFree(dEBR);
    cudaFree(dAn);cudaFree(dBn);cudaFree(dTr);cudaFree(dKey);cudaFree(dTgt);cudaFree(dTrOut);
    cudaFree(dF);cudaFree(dAR);cudaFree(dBC);
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
} // extern "C"
