#!/usr/bin/env bash
# Build + validate the NAIVE (correct, slow) sm_86 CUDA backend in WSL/Ubuntu.
# This is the Milestone-4 CORRECTNESS gate — NOT the performance target. The
# tensor-core port for speed is docs/cuda-sm86-port.md.
#
# Verified working on: Ubuntu 24.04 (WSL2) + RTX 3090 + CUDA 12.9 (apt cuda-nvcc).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${PRL_VENV:-$HOME/prlref}"

# 1. nvcc (the pip nvcc wheel ships only ptxas, so use the apt CUDA compiler).
NVCC="$(ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)"
if [ -z "${NVCC:-}" ] || [ ! -x "$NVCC" ]; then
  echo ">> installing cuda-nvcc + cuda-cudart-dev from the NVIDIA CUDA apt repo (one-time)"
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
  sudo dpkg -i cuda-keyring.deb >/dev/null 2>&1
  sudo apt-get update -qq
  NVCC_PKG=$(apt-cache search '^cuda-nvcc-12-' | awk '{print $1}' | sort -V | tail -1)
  CRT_PKG=$(apt-cache search '^cuda-cudart-dev-12-' | awk '{print $1}' | sort -V | tail -1)
  sudo apt-get install -y -qq "$NVCC_PKG" "$CRT_PKG"
  NVCC="$(ls -d /usr/local/cuda*/bin/nvcc | sort -V | tail -1)"
fi
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
echo ">> nvcc: $NVCC ($("$NVCC" --version | tail -1))"

# 2. python venv for the validation harness (numpy + blake3 only).
if [ ! -x "$VENV/bin/python" ]; then
  sudo apt-get install -y -qq python3.12-venv
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q numpy blake3
fi

# 3. compile the naive backend for sm_86.
mkdir -p "$ROOT/cuda/build"
SO="$ROOT/cuda/build/libprl_naive.so"
"$NVCC" -O3 -arch=sm_86 -Xcompiler -fPIC -shared \
  -Xlinker -rpath,"$CUDA_HOME/lib64" \
  "$ROOT/cuda/src/naive_sm86.cu" -o "$SO"
echo ">> built $SO"

# 4. validate against golden vectors + device-BLAKE3 on the real GPU.
"$VENV/bin/python" "$ROOT/reference/generate_golden.py"
"$VENV/bin/python" "$ROOT/reference/run_cuda_golden.py"
echo ">> (optional) GPU throughput: python cuda/tests/bench_naive.py"
