# Setup — Ubuntu 22.04 / 24.04 (or WSL2 Ubuntu)

Targets the PRD §7.2 minimum. The automated path is `scripts/setup_ubuntu.sh`; this doc explains it.

## Prerequisites (from pearl-official/README.md)
- Go **1.26+** (pearld, prlctl, oyster)
- Rust toolchain (zk-pow, pearl-blake3, py-pearl-mining)
- C compiler (XMSS FFI)
- Python **3.12** + `uv` (vLLM miner packages)
- `task` runner
- **CUDA Toolkit 12.x** + NVIDIA driver 545+ (GPU backend only)
- 16 GB RAM min / 32 GB recommended (PRD §7.3 — the dev box has 8 GB; upgrade before building CUDA).

## WSL2 note (this machine)
The host is Windows with WSL2 Ubuntu. Do GPU/CUDA builds **inside WSL**:
- Install the **CUDA-on-WSL** toolkit (NOT the Linux display driver — the Windows driver provides the GPU
  to WSL). `nvidia-smi` should already work inside WSL via the Windows 591.86 driver.
- 8 GB system RAM will make `task build` and Nsight profiling painful (PRD §7.3). Prefer ≥32 GB.

## Steps
```bash
# 1. one-shot environment + Pearl build + reference tests
PEARL_DIR=/path/to/pearl-official ./scripts/setup_ubuntu.sh

# 2. verify the reference + orchestration (no GPU needed)
. .venv/bin/activate
pytest -q reference/ miner/
python -m miner self-test
python -m miner list-devices        # should show your RTX 3090 once the driver is visible
```

## The CPU-only quick win
Even with no CUDA toolchain, the reference and orchestrator run anywhere:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python reference/generate_golden.py && pytest -q reference/ miner/
```

Next: [`create-wallet.md`](create-wallet.md) → [`run-local-node.md`](run-local-node.md) →
[`run-simnet.md`](run-simnet.md).
