# Benchmarking (PRD §24)

## What to measure
| Source | How | Meaning |
|---|---|---|
| CPU reference | `pytest reference/` | correctness only (no perf claim) |
| CUDA backend (micro) | `cuda/build/kernel_benchmark`, `scripts/profile_nsight.sh` | per-kernel GMAC/s, tensor-core util, occupancy |
| CUDA backend (e2e) | `python -m miner benchmark --backend cuda-sm86 --duration N` | harness throughput (MAC/s) |
| **Protocol hashrate** | accepted-proof rate against a node, over a long run | the real TH/s number |
| AlphaMiner (black box) | `scripts/compare_alphaminer.sh` | external reference number to chase |

> **Honesty rule:** `benchmark` reports **MAC/s**, not protocol **TH/s**. TH/s only means something
> measured as accepted proofs/sec against a live node. `STATUS.md` keeps the current number at 0 until
> the kernels exist. Do not print a TH/s figure the kernels cannot back.

## Durations
Short 5 min · Medium 1 h · Long 24 h (PRD §24). Use long runs for stale/reject rate and thermal stability.

## Output schema (PRD §24)
`scripts/benchmark_3090.sh` emits JSON alongside an `nvidia-smi dmon` log:
```json
{
  "gpu": "RTX 3090", "backend": "cuda-sm86", "duration_sec": 3600,
  "avg_hashrate_ths": 0.0, "power_avg_w": 0, "efficiency_gh_per_w": 0.0,
  "accepted": 0, "rejected": 0, "stale": 0, "gpu_temp_avg_c": 0, "vram_temp_avg_c": 0
}
```
(Zeros until M4. Fill `avg_hashrate_ths` from accepted proofs against a node, not from the harness.)

## Targets (PRD §12.4) — gated by correctness
Minimum 20 · Beta 50 · Competitive 80 · Close-to-the-bone 100–110 · Stretch >110 TH/s. First release will
be far below AlphaMiner and that is acceptable (correctness before speed). Optimization roadmap:
`cuda-sm86-port.md` §6 and `performance-plan.md`.
