# Benchmarking (PRD §24)

## What to measure
| Source | How | Meaning |
|---|---|---|
| CPU reference | `pytest reference/` | correctness only (no perf claim) |
| CUDA backend (micro) | `cuda/build/kernel_benchmark`, `scripts/profile_nsight.sh` | per-kernel GMAC/s, tensor-core util, occupancy |
| CUDA backend (e2e) | `python -m miner benchmark --backend cuda-mine --duration N` | harness throughput (MAC/s) |
| **Protocol hashrate** | `scripts/protocol_benchmark.py` against a pool or accepted-share log | pool-visible TH/s estimate |
| AlphaMiner (black box) | `scripts/compare_alphaminer.sh` | external clean-room reference number to chase |

> **Honesty rule:** `benchmark` reports **MAC/s**, not protocol **TH/s**. TH/s only means something
> measured from accepted shares/proofs against a pool or live node. Do not convert local TOPS or MAC/s
> into TH/s.

## Protocol TH/s from pool shares
For RTX 3090 AlphaPool comparison, use the public AlphaMiner static difficulty guidance:
`--password 'x;d=32768'`.

```bash
# Run a miner process for 10 minutes and compute pool-share TH/s from accepted shares.
python3 scripts/protocol_benchmark.py \
  --duration 600 \
  --share-diff 32768 \
  --output prl_protocol_10m.json \
  -- ./your-miner --pool stratum+tcp://us2.alphapool.tech:5566 \
       --address prl1pYOURPEARLADDRESS --worker prl3090 --password 'x;d=32768'

# Parse an existing log instead; provide the elapsed wall-clock seconds.
python3 scripts/protocol_benchmark.py \
  --parse-log alpha.log \
  --elapsed-sec 3600 \
  --share-diff 32768 \
  --output alpha_protocol_1h.json
```

The output reports:
- `pool_credited_th_s`: accepted shares × share difficulty × share unit / elapsed time.
- `raw_th_s`: average TH/s self-reported by the miner log when present.
- `accepted_shares`, `rejected`, `stale`, `share_diff`, `power_avg_w`, and `efficiency_th_w`.

`share_unit` defaults to the conventional Stratum difficulty-1 work unit, `2^32`. Keep it explicit in
results until AlphaPool documents a Pearl-specific conversion.

## AlphaMiner clean-room baseline
AlphaMiner source is private. Treat it only as a black-box benchmark:

```bash
ALPHA_BIN=/path/to/alpha-miner \
ADDR=prl1pYOURPEARLADDRESS \
POOL=stratum+tcp://us2.alphapool.tech:5566 \
DURATION=600 \
SHARE_DIFF=32768 \
scripts/compare_alphaminer.sh
```

The public target to chase on RTX 3090 is **100-110 TH/s** on AlphaMiner v1.7.x. Keep the comparison fair:
same GPU, driver, power limit, core clock, pool region, worker difficulty, and run duration. Use 10 min
for a smoke baseline, 1 h for tuning decisions, and 24 h before claiming parity.

## Durations
Short 5 min · Medium 1 h · Long 24 h (PRD §24). Use long runs for stale/reject rate and thermal stability.

## Output schemas
`scripts/benchmark_3090.sh` emits harness JSON alongside an `nvidia-smi dmon` log:
```json
{
  "backend": "cuda-mine",
  "duration_sec": 300.0,
  "attempts": 0,
  "macs_per_sec": 0.0,
  "tmac_per_sec": 0.0,
  "note": "harness throughput (MAC/s), NOT protocol TH/s"
}
```

`scripts/protocol_benchmark.py` emits protocol-share JSON:
```json
{
  "benchmark_kind": "protocol_share_rate",
  "elapsed_sec": 3600.0,
  "pool_credited_th_s": 0.0,
  "raw_th_s": null,
  "accepted_shares": 0,
  "rejected": 0,
  "stale": 0,
  "share_diff": 32768,
  "power_avg_w": 0,
  "efficiency_th_w": 0.0
}
```

## Targets (PRD §12.4) — gated by correctness
Minimum 20 · Beta 50 · Competitive 80 · Close-to-the-bone 100–110 · Stretch >110 TH/s. Optimization roadmap:
`cuda-sm86-port.md` §6 and `performance-plan.md`.
