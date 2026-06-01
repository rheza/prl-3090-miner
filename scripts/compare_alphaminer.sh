#!/usr/bin/env bash
# Black-box benchmark target ONLY (PRD §3, §8 Way A, §24).
#
# AlphaMiner is a third-party private binary. We DO NOT decompile, patch, or reverse
# engineer it. This script merely runs it (if you have a legitimate copy) against a
# pool you are authorized to use, and records the hashrate IT reports, so our open
# sm_86 miner has a number to chase. Nothing here inspects AlphaMiner internals.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ALPHA_BIN="${ALPHA_BIN:?set ALPHA_BIN to your AlphaMiner binary path}"
POOL="${POOL:-stratum+tcp://us2.alphapool.tech:5566}"
ADDR="${ADDR:?set ADDR to your PUBLIC prl1p... mining address}"
WORKER="${WORKER:-rheza3090}"
DURATION="${DURATION:-600}"
SHARE_DIFF="${SHARE_DIFF:-32768}"
DEVICE="${DEVICE:-0}"
OUT="${OUT:-alpha_protocol_$(date +%Y%m%d_%H%M%S).json}"

echo ">> Running AlphaMiner as a black box for ${DURATION}s"
echo ">> Pool=${POOL} worker=${WORKER} static_diff=x;d=${SHARE_DIFF}"
python3 "$HERE/scripts/protocol_benchmark.py" \
  --duration "$DURATION" \
  --share-diff "$SHARE_DIFF" \
  --device "$DEVICE" \
  --output "$OUT" \
  -- "$ALPHA_BIN" \
      --pool "$POOL" \
      --address "$ADDR" \
      --worker "$WORKER" \
      --password "x;d=${SHARE_DIFF}"

echo ">> Summary: ${OUT}"
echo ">> Compare AlphaMiner's pool-credited TH/s + watts against prl3090-miner's accepted-share rate."
echo "   Record both in docs/benchmarking.md. Do not copy AlphaMiner; only chase its number."
