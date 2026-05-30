#!/usr/bin/env bash
# Black-box benchmark target ONLY (PRD §3, §8 Way A, §24).
#
# AlphaMiner is a third-party private binary. We DO NOT decompile, patch, or reverse
# engineer it. This script merely runs it (if you have a legitimate copy) against a
# pool you are authorized to use, and records the hashrate IT reports, so our open
# sm_86 miner has a number to chase. Nothing here inspects AlphaMiner internals.
set -euo pipefail
ALPHA_BIN="${ALPHA_BIN:?set ALPHA_BIN to your AlphaMiner binary path}"
POOL="${POOL:?set POOL, e.g. stratum+tcp://host:port}"
ADDR="${ADDR:?set ADDR to your PUBLIC prl1p... mining address}"
WORKER="${WORKER:-rheza3090}"
DURATION="${DURATION:-600}"

echo ">> Running AlphaMiner as a black box for ${DURATION}s (reading its reported hashrate only)"
timeout "${DURATION}" "$ALPHA_BIN" \
  --pool "$POOL" --wallet "$ADDR" --worker "$WORKER" 2>&1 | tee "alpha_$(date +%s).log" || true

echo ">> Compare the reported TH/s + watts against prl3090-miner's accepted-proof rate."
echo "   Record both in docs/benchmarking.md. Do not copy AlphaMiner; only chase its number."
