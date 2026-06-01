# PRL-3090 - Pearl Miner for RTX 3090

PRL-3090 is a correctness-first Pearl **PRL** miner for the **NVIDIA RTX 3090 / Ampere `sm_86`**. The
goal is simple: build the best open, auditable PRL miner for a 3090, mine true-solo through the official
Pearl gateway, and keep every performance claim tied to source, tests, or live-node results.

The project is grounded in the official
[`pearl-research-labs/pearl`](https://github.com/pearl-research-labs/pearl) protocol. It does **not**
reverse engineer, decompile, or copy AlphaMiner or any private binary.

> **Current status:** the miner now has a real Ampere tensor-core hot loop, but it is still a miner
> harness, not a proven mainnet release. The active fast backend is `cuda-mine`: a fused `sm_86`
> kernel for the production `noise_rank=128` shape that runs noised int8 GEMM, extracts the mining
> transcript from accumulator registers, and performs the keyed BLAKE3 PoW check on the GPU. It passes
> the production golden cases for found/not-found state, winning indices, transcript words, persistent
> context execution, and GPU-side noise generation. Live SimNet/testnet/mainnet acceptance is still not
> claimed.

## What Pearl Mining Is

Pearl is a Bitcoin-derived L1 whose proof-of-work is **Proof-of-Useful-Work**: an honest int8 matrix
multiply `C = A*B`, with the PoW lottery embedded in the matmul's intermediate partial sums, plus a
Plonky2 ZK certificate proving the matmul was done correctly. The "useful work" is real LLM inference.
The full, source-cited algorithm is in [`docs/protocol-notes.md`](docs/protocol-notes.md).

The crucial architectural fact is that the official **`pearl-gateway`** already handles node
integration: block templates, work cache, ZK proof generation, block assembly, and `submitblock`. A
3090 miner does not reimplement that. It needs exactly two things:

1. a client for the gateway miner JSON-RPC (`getMiningInfo` / `submitPlainProof`), and
2. an **Ampere `sm_86` build of the GPU mining kernels** because the official kernels target Hopper
   `sm_90a`.

The code follows that approach. `miner/` is the gateway-facing cold path; `cuda/` is the GPU hot path.
The current runnable GPU backends still synthesize A/B matrices from the job header for repeatable
harness tests. Production mining still needs the real model-sourced A/B tensors and real
`py-pearl-mining` `PlainProof` assembly before a live gateway submit is meaningful.

## Milestone Snapshot

| Milestone | Status | What it means |
|---|---:|---|
| Research & protocol map | Done | Official Pearl source is mapped, and the miner path is documented with source citations. |
| CPU reference + golden vectors | Done | `reference/` implements the PoUW algorithm with NumPy and produces the GPU correctness oracle. |
| CUDA build + device layer | Done | `sm_86` CUDA build skeleton, device enumeration, and NVML-backed telemetry exist. |
| GPU correctness | Done for `cuda-naive`; production-rank checks done for `cuda-mine` | `cuda-naive` passes all golden vectors. `cuda-mine` passes the full production `noise_rank=128` golden contract, including found flags, winning indices, transcript words, persistent context, keyed GPU noise, and BLAKE3 checks. |
| SimNet miner loop | Partial | Orchestration, stale cancellation, metrics, safety checks, and mock-gateway self-test work. Full live SimNet acceptance still needs a local node/gateway run with real `PlainProof` assembly. |
| Tensor-core speed | Substantially implemented as a harness | `cuda-mine` fuses noised GEMM + transcript + PoW on Ampere `mma.sync` with `cp.async` staging and persistent device buffers. It is fast enough to be the current optimization target, but its numbers are kernel/harness throughput, not protocol TH/s. |
| Mainnet true-solo beta | Not started | Depends on real model A/B input, real `PlainProof` output, and live SimNet/testnet validation. |
| Close-to-the-bone release | Not started | Depends on sustained 3090 tuning, reject/stale-rate evidence, and long-run stability. |

No protocol TH/s number is claimed yet. Harness benchmarks report MAC/s only; TH/s belongs to accepted
proofs against a live node over a real run.

## What Exists Now

The repo currently represents a working **RTX 3090 PoUW kernel prototype plus miner harness**:

- `miner/prl3090_miner.py` provides `list-devices`, `self-test`, `benchmark`, `validate-job`, and `run`.
- `miner/runtime.py` implements the cold path: job polling, new-tip detection, stale-result dropping,
  gateway submission, metrics, and thermal/invalid-proof safety checks.
- `miner/gateway_client.py` speaks the gateway miner RPC: `getMiningInfo` and `submitPlainProof`.
- `CpuBackend` and `CudaNaiveBackend` are correctness/orchestration harnesses.
- `CudaMineBackend` is the current fast path. It loads `cuda/build/libprl_miner.so` through
  `miner/cuda_mine.py`, keeps CUDA buffers and a stream alive through `prl_mine_ctx_*`, derives noise
  on the GPU from `key_A`/`key_B`, and returns the winning tile/transcript when the PoW target is met.
- `cuda-sm86` is still the older generic extension surface. In this checkout its Python `prl_cuda`
  module is not wired as the production search backend; use `cuda-mine` for the fused miner path.

The important limitation: the backend proof bytes are still marked `"_harness": true`. They prove the
orchestrator and GPU search path, not a final live Pearl block submission.

## Performance Meaning

Latest local verification on the visible RTX 3090 (`driver 591.86`, `24 GB`) shows:

| Check | Result | Meaning |
|---|---:|---|
| `reference/run_cuda_golden.py` | All 6 golden cases pass | `cuda-naive` is still the full correctness oracle, including C, found flags, locations, transcript, and device BLAKE3. |
| `cuda/tests/validate_noisegen.py` | All pass | GPU noise derivation is bit-exact for the checked shapes. |
| `cuda/tests/validate_mine.py` | All production `noise_rank=128` cases pass | `cuda-mine` matches the golden found/not-found state, winning indices, transcript words, persistent context path, and keyed GPU-noise path. |
| `validate_mine.py` kernel-only throughput | ~41.5 TOPS at `2048^3` | CUDA-event timing of fused noised GEMM + transcript + BLAKE3. Useful for kernel tuning. |
| `miner benchmark --backend cuda-mine --duration 6` | 4,611 attempts, ~6.44 GMAC/s harness throughput | End-to-end Python harness throughput including synthetic A/B generation, host/device traffic, launches, and keyed GPU noise. Useful for regression checks. |

Do **not** read those as Pearl protocol hashrate. A real TH/s number requires accepted proofs against a
live node/gateway or accepted pool shares over a sustained run, with stale and reject rates included.
For pool-side benchmarking, use `scripts/protocol_benchmark.py`; for a clean-room AlphaMiner baseline,
use `scripts/compare_alphaminer.sh` with static diff `x;d=32768` on RTX 3090.

## Repository Layout

```text
prl-3090-miner/
|-- reference/        done: pure-NumPy PoUW reference + golden-vector generator
|-- tests/golden/     done: generated golden vectors for GPU correctness
|-- miner/            done: Python orchestration, gateway client, metrics, safety
|-- cuda/             in progress: cuda-naive correctness done; cuda-mine fused tensor-core path works as a harness
|-- docs/             done: protocol notes, sm_86 port plan, setup, tuning, safety, benchmarking
|-- config/           done: miner.example.toml mapped to gateway settings
|-- scripts/          done: Ubuntu, SimNet, benchmark, profiling helpers
`-- ci/               done: GitHub Actions reference-test workflow
```

## Quick Start: CPU Reference and Orchestrator

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy blake3 pytest
python reference/generate_golden.py              # writes tests/golden/*.npz + manifest.json
pytest -q reference/                             # 27 tests incl. official truth table + golden replay
python -m miner.prl3090_miner self-test          # exercises the orchestrator with the CPU backend
python -m miner.prl3090_miner benchmark --backend cpu --duration 10
```

The **CPU backend** is a harness for correctness and orchestration. It is not a production miner because
real mining needs the gateway/vLLM/proving path and the GPU hot loop.

## GPU Backends

| Backend | State | Use it for |
|---|---|---|
| `cpu` | Available anywhere with Python dependencies | Reference correctness and orchestration self-test. |
| `cuda-naive` | Correct but slow, Linux/WSL + RTX 3090 build | GPU correctness validation and end-to-end miner harness runs. |
| `cuda-mine` | Current fused tensor-core miner harness, Linux/WSL + RTX 3090 build | Fast `sm_86` noised GEMM + transcript + on-device BLAKE3 validation and regression benchmarking. |
| `cuda-sm86` | Generic extension surface, not the active search backend | Device/NVML/C-ABI experiments and older `prl_cuda` wiring work. |

Useful GPU commands on WSL/Ubuntu with CUDA installed:

```bash
scripts/build_naive_wsl.sh
PYTHONPATH=. python reference/run_cuda_golden.py
python -m miner.prl3090_miner benchmark --backend cuda-naive --duration 10

scripts/build_miner_wsl.sh
PYTHONPATH=. python cuda/tests/validate_noisegen.py
PYTHONPATH=. python cuda/tests/validate_mine.py
python -m miner.prl3090_miner benchmark --backend cuda-mine --duration 6

scripts/build_sm86.sh
PYTHONPATH=. python cuda/tests/validate_mma.py   # exploratory plain-GEMM kernels
```

The `cuda-mine` / `cuda-sm86` work is described in [`docs/cuda-sm86-port.md`](docs/cuda-sm86-port.md) and tracked in
[`STATUS.md`](STATUS.md).

## Target Real Mining Flow

See [`docs/setup-ubuntu.md`](docs/setup-ubuntu.md), [`docs/run-simnet.md`](docs/run-simnet.md), and
[`docs/run-mainnet-solo.md`](docs/run-mainnet-solo.md). The flow is:

```text
build node -> create wallet (Oyster) -> start pearld -> start pearl-gateway -> start prl3090-miner
```

Always validate on **SimNet** first. On SimNet the ZK check is bypassed in Pearl, so you can validate the
accepted-block loop before moving to testnet/mainnet. As of this README update, that live accepted-block
loop is still a remaining integration milestone.

## Safety and Non-goals

- Never asks for or stores a seed phrase or private key: only the **public** Taproot mining address and
  local RPC credentials are needed.
- Does not reverse engineer, decompile, or copy AlphaMiner; it is only a black-box benchmark target.
- Applies no overclock or power change unless explicitly enabled; temperature limits are enforced by the
  miner runtime.
- No hidden process, no remote telemetry by default, visible logs always.

## License

ISC, matching upstream Pearl. See [`LICENSE`](LICENSE).
