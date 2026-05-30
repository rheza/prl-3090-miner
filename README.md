# PRL-3090 — Close-to-the-Bone Pearl Miner

A high-performance Pearl **PRL** miner for the **NVIDIA RTX 3090 / Ampere `sm_86`**, built to mine
true-solo against a local Pearl node, with the entire mining path grounded in the official
[`pearl-research-labs/pearl`](https://github.com/pearl-research-labs/pearl) protocol — never reverse
engineered from any private binary.

> **Read [`STATUS.md`](STATUS.md) first.** It states exactly what works today, what is scaffolding, and
> what remains (and why). This repo is honest about the fact that the hard part — Ampere tensor-core
> kernels — is a multi-week CUDA effort, and it sets you up to do that effort correctly instead of
> faking it.

## What Pearl mining actually is

Pearl is a Bitcoin-derived L1 whose proof-of-work is **Proof-of-Useful-Work**: an honest int8 matrix
multiply `C = A·B`, with the PoW lottery embedded in the matmul's intermediate partial sums, plus a
Plonky2 ZK certificate proving the matmul was done correctly. The "useful work" is real LLM inference.
The full, source-cited algorithm is in [`docs/protocol-notes.md`](docs/protocol-notes.md).

The crucial architectural fact (see [`docs/architecture.md`](docs/architecture.md)): the official
**`pearl-gateway`** already does *all* node integration — block templates, work cache, ZK proof
generation, block assembly, `submitblock`. **A 3090 miner does not reimplement that.** It needs exactly
two things:

1. a client of the gateway's miner-facing JSON-RPC (`getMiningInfo` / `submitPlainProof`), and
2. an **Ampere `sm_86` build of the GPU kernels** (the official ones are Hopper `sm_90a` only).

## Repository layout

```
prl-3090-miner/
├── reference/        ✅ runnable pure-NumPy PoUW reference + golden-vector generator (no torch/CUDA)
├── tests/golden/     ✅ generated golden vectors (correctness oracle for the GPU backend)
├── miner/            ✅ Python orchestration: gateway client, job manager, metrics, safety (runnable)
├── cuda/             🟧 sm_86 CUDA backend skeleton (CMake + headers + stubs; needs the kernel port)
├── docs/             ✅ protocol notes, sm_86 port plan, setup, tuning, safety, benchmarking
├── config/           ✅ miner.example.toml (mapped to the real gateway env vars)
├── scripts/          ✅ Ubuntu/SimNet/benchmark/profiling helpers
└── ci/               ✅ GitHub Actions (reference tests run on every push)
```

## Quick start (what you can run right now, on any OS with Python)

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy blake3 pytest
python reference/generate_golden.py              # writes tests/golden/*.npz + manifest.json
pytest -q reference/                             # 27 tests incl. the official truth table + golden replay
python -m miner.prl3090_miner self-test          # exercises the orchestrator with the CPU backend
python -m miner.prl3090_miner benchmark --backend cpu --duration 10
```

The miner CLI (PRD §11.1) is implemented with a pluggable backend. The **CPU backend** (the NumPy
reference) works today and lets you validate orchestration, metrics, safety, and the mine→submit loop
against a SimNet node end to end. The **`cuda-sm86` backend** is the port described in
[`docs/cuda-sm86-port.md`](docs/cuda-sm86-port.md).

## Real mining (Ubuntu + RTX 3090)

See [`docs/setup-ubuntu.md`](docs/setup-ubuntu.md), [`docs/run-simnet.md`](docs/run-simnet.md), and
[`docs/run-mainnet-solo.md`](docs/run-mainnet-solo.md). The flow:
**build node → create wallet (Oyster) → start `pearld` → start `pearl-gateway` → start `prl3090-miner`.**
Always validate on **SimNet** first — on SimNet the ZK check is bypassed (`validate.go:421-423`), so you
get a real accepted-block loop without the proprietary proving stack.

## Safety & non-goals (PRD §5, §17)

- Never asks for or stores a seed phrase or private key — only the **public** Taproot mining address and
  local RPC credentials.
- Does not reverse engineer, decompile, or copy AlphaMiner; it is only a black-box benchmark target.
- Applies no overclock/power change unless you explicitly enable it; throttles on temperature limits.
- No hidden process, no remote telemetry by default, visible logs always.

## License

ISC, matching upstream Pearl. See [`LICENSE`](LICENSE).
