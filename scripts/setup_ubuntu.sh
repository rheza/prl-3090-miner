#!/usr/bin/env bash
# Set up an Ubuntu 22.04/24.04 box (or WSL2 Ubuntu) to build the Pearl stack and
# the prl-3090 miner. Grounded in pearl-official/README.md "Prerequisites".
# Idempotent-ish; re-run safe. Requires sudo for apt + driver.
set -euo pipefail

PEARL_DIR="${PEARL_DIR:-$(cd "$(dirname "$0")/../../pearl-official" && pwd)}"

echo ">> apt prerequisites (build tools, C compiler for XMSS, clang-format)"
sudo apt-get update
sudo apt-get install -y build-essential clang git curl pkg-config libssl-dev python3.12 python3.12-venv

echo ">> Go 1.26+ (pearld, oyster) — README requires Go 1.26"
if ! command -v go >/dev/null || [[ "$(go version)" != *"go1.26"* && "$(go version)" != *"go1.2"[6-9]* ]]; then
  GO_VER=1.26.0
  curl -fsSL "https://go.dev/dl/go${GO_VER}.linux-amd64.tar.gz" -o /tmp/go.tgz
  sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf /tmp/go.tgz
  export PATH="/usr/local/go/bin:$PATH"
  echo 'export PATH=/usr/local/go/bin:$PATH' >> "$HOME/.bashrc"
fi

echo ">> Rust toolchain (ZK + hashing crates: zk-pow, pearl-blake3, py-pearl-mining)"
command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env" || true

echo ">> uv (vLLM miner Python workspace) + Task runner"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v task >/dev/null || sudo snap install task --classic || \
  (sh -c "$(curl -ssL https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin)

echo ">> CUDA toolkit check (needed only for the GPU backend; nvcc must be on PATH)"
if command -v nvcc >/dev/null; then nvcc --version; else
  echo "   nvcc NOT found. Install CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-toolkit"
  echo "   On WSL2: install the CUDA-on-WSL toolkit (do NOT install the Linux display driver)."
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || \
  echo "   nvidia-smi not available — GPU mining will not work until the driver is set up."

echo ">> Build the Pearl blockchain (pearld, prlctl, oyster)"
cd "$PEARL_DIR"
task build:blockchain
echo ">> Install the vLLM miner Python packages (pearl-gateway, miner-base, ...)"
task build:miner || echo "   (miner build may require a GPU/CUDA; gateway alone is enough for solo wiring)"

echo ">> prl-3090 reference env (numpy + blake3 only)"
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reference/generate_golden.py
pytest -q reference/ miner/

echo ">> Done. Next: docs/run-simnet.md"
