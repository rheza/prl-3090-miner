#!/usr/bin/env bash
# Build + validate the FUSED tensor-core miner kernel (the real hot loop) in WSL/Ubuntu.
# Validated bit-for-bit vs the full golden vectors (found/indices/transcript) on the GPU.
# Verified on: Ubuntu 24.04 (WSL2) + RTX 3090 + CUDA 12.9 (apt cuda-nvcc).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${PRL_VENV:-$HOME/prlref}"

NVCC="$(ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)"
[ -x "${NVCC:-}" ] || { echo "nvcc not found — run scripts/build_naive_wsl.sh once to install it"; exit 1; }
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
[ -x "$VENV/bin/python" ] || { echo "venv missing — run scripts/build_naive_wsl.sh first"; exit 1; }

mkdir -p "$ROOT/cuda/build"
echo ">> compiling mine_sm86.cu (fused tensor-core miner) for sm_86"
"$NVCC" -O3 -arch=sm_86 -Xcompiler -fPIC -shared -Xlinker -rpath,"$CUDA_HOME/lib64" \
  "$ROOT/cuda/src/mine_sm86.cu" -o "$ROOT/cuda/build/libprl_miner.so"
echo ">> validating vs full golden on the GPU"
"$VENV/bin/python" "$ROOT/reference/generate_golden.py"
PYTHONPATH="$ROOT" "$VENV/bin/python" "$ROOT/cuda/tests/validate_mine.py"
