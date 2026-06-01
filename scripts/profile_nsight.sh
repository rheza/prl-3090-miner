#!/usr/bin/env bash
# Nsight profiling wrappers (PRD §12.6). Run after the kernels build and pass golden.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-compute}"   # compute | systems

cd "$HERE"; . .venv/bin/activate 2>/dev/null || true

case "$MODE" in
  compute)  # per-kernel counters: tensor-core util, SMEM/occupancy, memory throughput
    command -v ncu >/dev/null || { echo "Nsight Compute (ncu) not found"; exit 1; }
    ncu --set full --target-processes all \
        --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis \
        -o "ncu_$(date +%s)" \
        python -m miner benchmark --backend cuda-mine --duration 20
    ;;
  systems)  # stream overlap, job-switch latency, H2D/D2H transfers
    command -v nsys >/dev/null || { echo "Nsight Systems (nsys) not found"; exit 1; }
    nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-device=all \
        -o "nsys_$(date +%s)" \
        python -m miner benchmark --backend cuda-mine --duration 20
    ;;
  *) echo "usage: $0 [compute|systems]"; exit 2 ;;
esac
