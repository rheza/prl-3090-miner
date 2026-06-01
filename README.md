# PRL-3090 - Pearl Miner for RTX 3090

PRL-3090 is a correctness-first Pearl **PRL** miner for the **NVIDIA RTX 3090 / Ampere `sm_86`**. The
goal is simple: build the best open, auditable PRL miner for a 3090, mine true-solo through the official
Pearl gateway, and make every performance claim traceable to source, tests, or live-node results.

The project is grounded in the official
[`pearl-research-labs/pearl`](https://github.com/pearl-research-labs/pearl) protocol. It does **not**
reverse engineer, decompile, or copy AlphaMiner or any private binary.

> **Current status:** GPU correctness is real; production speed is not done yet. The `cuda-naive`
> backend passes golden vectors on an RTX 3090, and Ampere `mma.sync` tensor-core GEMM prototypes are
> passing golden `C == A@B` with measured progress. The remaining milestone is fusing the full
> noised/transcript mining path onto fast tensor-core kernels and proving it against SimNet/testnet.
> Read [`STATUS.md`](STATUS.md) for the exact milestone ledger.

## What Pearl Mining Is

Pearl is a Bitcoin-derived L1 whose proof-of-work is **Proof-of-Useful-Work**: an honest int8 matrix
multiply `C = A*B`, with the PoW lottery embedded in the matmul's intermediate partial sums, plus a
Plonky2 ZK certificate proving the matmul was done correctly. The "useful work" is real LLM inference.
The full, source-cited algorithm is in [`docs/protocol-notes.md`](docs/protocol-notes.md).

The crucial architectural fact is that the official **`pearl-gateway`** already handles node
integration: block templates, work cache, ZK proof generation, block assembly, and `submitblock`. A
3090 miner does not reimplement that. It needs exactly two things:

1. a client for the gateway miner JSON-RPC (`getMiningInfo` / `submitPlainProof`), and
2. an **Ampere `sm_86` build of the GPU kernels** because the official kernels target Hopper `sm_90a`.

## Milestone Snapshot

| Milestone | Status | What it means |
|---|---:|---|
| Research & protocol map | Done | Official Pearl source is mapped, and the miner path is documented with source citations. |
| CPU reference + golden vectors | Done | `reference/` implements the PoUW algorithm with NumPy and produces the GPU correctness oracle. |
| CUDA build + device layer | Done | `sm_86` CUDA build skeleton, device enumeration, and NVML-backed telemetry exist. |
| GPU correctness | Done for `cuda-naive` | Plain-integer-core CUDA NoisyGEMM passes all golden vectors, including found flags, winning indices, transcript words, and BLAKE3 checks. |
| SimNet miner loop | Partial | Orchestration, stale cancellation, metrics, and mock-gateway self-test work. Full live SimNet acceptance still needs a local node/gateway run. |
| Tensor-core speed | In progress | Ampere int8 `mma.sync` GEMM kernels are validated against golden `C`, with simple, SMEM-tiled, and `cp.async` variants under test. Full noised/transcript fusion remains. |
| Mainnet true-solo beta | Not started | Depends on the fast tensor-core path plus SimNet/testnet validation. |
| Close-to-the-bone release | Not started | Depends on sustained 3090 tuning, reject/stale-rate evidence, and long-run stability. |

No protocol TH/s number is claimed yet. Harness benchmarks report MAC/s only; TH/s belongs to accepted
proofs against a live node over a real run.

## Repository Layout

```text
prl-3090-miner/
|-- reference/        done: pure-NumPy PoUW reference + golden-vector generator
|-- tests/golden/     done: generated golden vectors for GPU correctness
|-- miner/            done: Python orchestration, gateway client, metrics, safety
|-- cuda/             in progress: naive GPU correctness done; tensor-core speed path underway
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
| `cuda-sm86` | Tensor-core production target, not wired for full search yet | Fast Ampere `mma.sync` kernels and full transcript fusion. |

Useful GPU commands on WSL/Ubuntu with CUDA installed:

```bash
scripts/build_naive_wsl.sh
PYTHONPATH=. python reference/run_cuda_golden.py
python -m miner.prl3090_miner benchmark --backend cuda-naive --duration 10

scripts/build_sm86.sh
PYTHONPATH=. python cuda/tests/validate_mma.py
```

The `cuda-sm86` work is described in [`docs/cuda-sm86-port.md`](docs/cuda-sm86-port.md) and tracked in
[`STATUS.md`](STATUS.md).

## Real Mining Flow

See [`docs/setup-ubuntu.md`](docs/setup-ubuntu.md), [`docs/run-simnet.md`](docs/run-simnet.md), and
[`docs/run-mainnet-solo.md`](docs/run-mainnet-solo.md). The flow is:

```text
build node -> create wallet (Oyster) -> start pearld -> start pearl-gateway -> start prl3090-miner
```

Always validate on **SimNet** first. On SimNet the ZK check is bypassed in Pearl, so you can validate the
accepted-block loop before moving to testnet/mainnet.

## Safety and Non-goals

- Never asks for or stores a seed phrase or private key: only the **public** Taproot mining address and
  local RPC credentials are needed.
- Does not reverse engineer, decompile, or copy AlphaMiner; it is only a black-box benchmark target.
- Applies no overclock or power change unless explicitly enabled; temperature limits are enforced by the
  miner runtime.
- No hidden process, no remote telemetry by default, visible logs always.

## License

ISC, matching upstream Pearl. See [`LICENSE`](LICENSE).
