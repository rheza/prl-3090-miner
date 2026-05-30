# Architecture

## The one insight that shapes everything

The official **`pearl-gateway`** already implements all the hard node integration:
block-template polling (`getblocktemplate`), a work cache with new-tip detection, ZK proof generation
(`generate_proof`), block assembly, and `submitblock` — see [`protocol-notes.md`](protocol-notes.md) §3, §5.

So `prl3090-miner` does **not** re-implement node/RPC/ZK/consensus. The entire delta to mine Pearl on an
RTX 3090 is:

1. a **client of the gateway miner-RPC** (`getMiningInfo` / `submitPlainProof`), and
2. an **Ampere `sm_86` build of the GPU kernels** (the official ones are Hopper `sm_90a` only).

Everything in this repo serves those two things. This collapses the PRD's imagined 8-milestone
from-scratch build into "port one CUDA module + a thin orchestrator," which is both more honest and far
more achievable.

## Topology

```
pearld (Go)  ──getblocktemplate/submitblock──▶  pearl-gateway (Python)  ──getMiningInfo/submitPlainProof──▶  prl3090-miner
  consensus,                                       polls node ~1s,                                              ├─ miner/ (Python cold path)
  zkpow.Verify                                     builds+submits block                                        └─ cuda/  (C++/CUDA hot path)
```

## Cold path / hot path split (PRD §18 — "no Python in the hot loop")

- **Cold path (Python, `miner/`):** job fetch, stale detection, submission, metrics, safety. None of
  this is performance-critical — the gateway itself polls only once per second. Keeping it in Python
  mirrors the official miner and keeps the code small and testable.
- **Hot path (C++/CUDA, `cuda/`):** the NoisyGEMM search. This is the only thing that runs per-nonce, so
  this is the only thing that must be native. The Python layer calls it across a C ABI (`prl_cuda.h`).

This satisfies "no Python in the hot loop" precisely: the hot loop *is* the CUDA backend.

## Module map (PRD §20 → this repo)

| PRD §20 dir | Here |
|---|---|
| `miner/node_client`, `miner/stratum_client` | `miner/gateway_client.py` (gateway speaks for the node; Stratum is a V1.1 add-on) |
| `miner/job_manager` | `miner/runtime.py` `JobManager` |
| `miner/submitter` | `miner/runtime.py` `mine_loop` submit path |
| `miner/metrics` | `miner/runtime.py` `Metrics` (PRD §16 status line) |
| `miner/safety` | `miner/runtime.py` `SafetyMonitor` (PRD §17) |
| `miner/config` | `miner/config.py` |
| `cuda/src/noisy_gemm_sm86.cu` | the GEMM + noising kernels (M4 port) |
| `cuda/src/proof_extract_sm86.cu` | inner-hash / transcript / PoW |
| `cuda/src/{device_manager,nvml_monitor}` | device enum + telemetry (implemented) |
| `reference/` | CPU reference + golden vectors (PRD §13) |

## Why Python orchestration instead of the PRD's Rust/Go

The PRD §19 prefers Rust+C++/CUDA. We deviate deliberately for V1:
- The official stack is **Python orchestration + CUDA hot path**, and it is proven against the live
  protocol. Re-using `pearl-gateway` (Python) for node integration is the lowest-risk path.
- The orchestration is not on the hot path, so language choice there is about ergonomics, not speed.
- The C ABI (`cuda/include/prl_cuda.h`) is language-neutral: a Rust orchestrator can replace `miner/`
  later with zero kernel changes. The boundary is clean by design.

If you want the Rust orchestrator, implement it against `prl_cuda.h` + the gateway JSON-RPC in
`protocol-notes.md` §3.3; the Python `miner/` package is the executable reference for that behavior.

## Data flow for one found block

`getMiningInfo` → `MiningJob{incomplete_header_bytes, target}` → backend derives the commitment chain &
noise (protocol-notes §4.4) → GPU NoisyGEMM search → on win: `(winning A rows, Bᵀ cols, transcript)` →
`PlainProof` (py-pearl-mining) → `submitPlainProof` → gateway builds the ZK cert + block → `submitblock`.
