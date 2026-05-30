#!/usr/bin/env bash
# Benchmark the miner and capture GPU telemetry alongside (PRD §24).
# Emits a JSON-ish summary plus an nvidia-smi dmon log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DURATION="${1:-300}"          # seconds (PRD: short 300, medium 3600, long 86400)
BACKEND="${BACKEND:-cuda-sm86}"
OUT="${OUT:-bench_$(date +%Y%m%d_%H%M%S)}"

cd "$HERE"; . .venv/bin/activate 2>/dev/null || true

echo ">> nvidia-smi dmon -> ${OUT}.dmon.log (power/temp/clocks/util)"
nvidia-smi dmon -s pucvmet -o DT -d 1 > "${OUT}.dmon.log" &
DMON=$!; trap 'kill $DMON 2>/dev/null || true' EXIT

echo ">> prl3090-miner benchmark backend=$BACKEND duration=${DURATION}s"
python -m miner benchmark --backend "$BACKEND" --duration "$DURATION" | tee "${OUT}.json"

echo ">> dmon log: ${OUT}.dmon.log ; summary: ${OUT}.json"
echo "   (For protocol TH/s, run against a node and read accepted proofs; harness MAC/s is not TH/s.)"
