#!/usr/bin/env bash
# Build the Ampere sm_86 GPU backend. This is the M3/M4 work — see docs/cuda-sm86-port.md.
#
# Strategy: we do NOT modify ../pearl-official. We build cuda/ (our sm_86 port) and,
# where we reuse upstream portable kernels (blake3, quantize, denoise, noise-gen), we
# vendor them from pearl-official/miner/pearl-gemm/csrc via CMake include paths.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PEARL_GEMM="${PEARL_GEMM:-$HERE/../pearl-official/miner/pearl-gemm}"

command -v nvcc >/dev/null || { echo "nvcc not found — install CUDA Toolkit 12.x"; exit 1; }
nvcc --version | grep -i release || true

# RTX 3090 is sm_86. (compute_80 added for fat-binary portability to other Ampere.)
ARCH="${CUDA_ARCH:-86}"
echo ">> Building cuda/ for sm_${ARCH} (pearl-gemm source: $PEARL_GEMM)"

cmake -S "$HERE/cuda" -B "$HERE/cuda/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DPEARL_GEMM_DIR="$PEARL_GEMM"
cmake --build "$HERE/cuda/build" -j

echo ">> Validate the built backend against golden vectors"
cd "$HERE"
. .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$HERE/cuda/build:${PYTHONPATH:-}"
python reference/generate_golden.py
pytest -q cuda/tests/test_cuda_golden.py
echo ">> If the golden tests pass, wire backend='cuda-sm86' in miner.toml and re-run self-test."
