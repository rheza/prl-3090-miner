#!/usr/bin/env bash
# Bring up a local SimNet: pearld + Oyster wallet + pearl-gateway, then the miner.
# SimNet bypasses the ZK PoW check (validate.go:421-423), so the full mine->submit->accept
# loop works WITHOUT the proprietary proving stack — the right place to validate first.
# Grounded in pearl-official/README.md "Running a Node and vLLM Miner".
set -euo pipefail

PEARL_DIR="${PEARL_DIR:-$(cd "$(dirname "$0")/../../pearl-official" && pwd)}"
BIN="$PEARL_DIR/bin"
RPCUSER="${RPCUSER:-rpcuser}"
RPCPASS="${RPCPASS:-rpcpass}"
export PEARLD_RPC_PASSWORD="$RPCPASS"

cd "$PEARL_DIR"

echo ">> 1. Oyster wallet (create once if missing), then start it"
[ -f "$HOME/.oyster/simnet/wallet.db" ] || "$BIN/oyster" -u "$RPCUSER" -P "$RPCPASS" --simnet --create
"$BIN/oyster" -u "$RPCUSER" -P "$RPCPASS" --simnet &
OYSTER_PID=$!; sleep 3

echo ">> 2. Generate a Taproot mining address (PUBLIC — no key leaves the wallet)"
MINING_ADDR="$("$BIN/prlctl" -u "$RPCUSER" -P "$RPCPASS" --simnet -s https://localhost:18554 getnewaddress)"
echo "   mining address: $MINING_ADDR"

echo ">> 3. Start pearld (simnet, RPC 18556)"
"$BIN/pearld" --simnet --notls \
  --rpcuser="$RPCUSER" --rpcpass="$RPCPASS" \
  --rpclisten=127.0.0.1:18556 \
  --miningaddr="$MINING_ADDR" --txindex &
PEARLD_PID=$!; sleep 5

echo ">> 4. Start pearl-gateway (bridges node <-> miner; UDS /tmp/pearlgw.sock)"
export PEARLD_RPC_URL="http://127.0.0.1:18556"
export PEARLD_RPC_USER="$RPCUSER"
export PEARLD_MINING_ADDRESS="$MINING_ADDR"
pearl-gateway start &
GW_PID=$!; sleep 3

echo ">> 5. Start prl3090-miner against the gateway"
cd "$(dirname "$0")/.."
. .venv/bin/activate 2>/dev/null || true
# Edit config/miner.example.toml -> miner.toml (set mining_address + transport=uds) first.
python -m miner run --config "${CONFIG:-miner.toml}" || true

cleanup() { kill "$GW_PID" "$PEARLD_PID" "$OYSTER_PID" 2>/dev/null || true; }
trap cleanup EXIT
wait
