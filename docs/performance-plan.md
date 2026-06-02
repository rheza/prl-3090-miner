# Performance Plan (PRD §12.5, §18)

Correctness first (golden vectors), then optimize one kernel at a time with Nsight. Targets and the
big-picture port are in [`cuda-sm86-port.md`](cuda-sm86-port.md); this is the optimization checklist.

## Order of attack
1. **Make it correct** (M4): all `tests/golden/` pass on GPU, including transcript words. No perf work
   before this — a fast wrong miner is worthless (PRD §12.3).
2. **Make the GEMM fast** (the dominant kernel):
   - int8 tensor cores via `mma.sync.m16n8k32.s8`; maximize MMA issue rate.
   - `cp.async` double/triple-buffered SMEM pipeline; hide global latency.
   - Tile to the **sm_86 99 KB opt-in SMEM** ceiling (e.g. 128×128×64); pick stages for occupancy.
   - Coalesced global loads; vectorized 128-bit (`int4`) access; avoid bank conflicts in SMEM.
   - Register pressure: keep occupancy high enough to hide latency without spilling.
3. **Fuse and overlap:**
   - Fuse denoise + transcript extraction into the mainloop epilogue (avoid extra passes).
   - `streams=2` to overlap job *N+1* H2D with job *N* compute; pinned host buffers; preallocate.
   - Keep NVML sampling on its own thread (already so in `nvml_monitor.cpp`).
   - Keep PoW transcript scanning parallel and order-preserving; do not reintroduce a serial
     post-kernel scan over candidate tiles.
   - Keep benchmark hard-target attempts batched behind CUDA Graph replay; this is now the default
     harness path (`PRL_CUDA_BATCH=256`).
4. **Minimize job-switch latency:** the chain retargets every 3m14s but tips can change anytime; a found
   proof on a stale header is wasted. Make `should_stop` checked between k-tiles, cancel fast.
5. **Reduce host overhead:** no Python in the hot loop (it isn't); batch submissions; avoid needless
   `cudaDeviceSynchronize` (PRD §18).

## Profiling loop
```bash
scripts/profile_nsight.sh compute     # ncu: SOL, Occupancy, Memory/Compute workload
scripts/profile_nsight.sh systems     # nsys: stream overlap, transfers, gaps
```
Track per-kernel: tensor-core utilization, achieved occupancy, SMEM/regs per block, DRAM throughput,
and the eligible-warps stall reasons. Optimize the kernel with the worst SOL first.

## Milestone gates (PRD §21)
- M6: ≥50 TH/s, stale <2%, reject <1%, stable 12 h.
- M8: 80–110 TH/s depending on preset, stable 24 h, benchmarked vs AlphaMiner black box.

## Clean-room parity rule
AlphaMiner may be used only as a black-box benchmark unless the rights holder/operator gives explicit
written permission for a narrower interoperability review. Record its reported TH/s, accepted shares,
rejects, stales, watts, driver, clocks, pool region, worker name, and static difficulty. Do not
decompile, disassemble, patch, trace proprietary internals, or copy behavior from private binaries without
that authorization. Use public Pearl source, public AlphaMiner documentation, pool-visible shares, our own
packet captures, operator-provided protocol details, and Nsight aggregate counters.

Permissioned AlphaPool interoperability is a separate milestone from performance parity: first get an
authorized `pearl.challenge_response` implementation or operator-provided auth path, then submit real
`PlainProof` shares, then measure pool-visible TH/s over a sustained run.
