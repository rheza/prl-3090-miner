#!/usr/bin/env bash
# VERIFIED end-to-end legitimate solo loop on SimNet:
#   pearld --simnet  <-  pearl-gateway (TCP 8337)  <-  miner/simnet_solo.py
# Produces a real PlainProof (py-pearl-mining) that the node accepts as a block.
# Verified 2026-06: chain tip advanced and gateway logged "Block accepted by node!".
#
# Prereqs (built earlier in WSL): pearl-official/bin/{pearld,oyster,prlctl},
#   pearl-official/.venv with pearl-gateway + miner-base (uv sync --package ...).
set -euo pipefail
P=/mnt/c/Users/damni/Documents/mining-pearl/pearl-official
MINER=/mnt/c/Users/damni/Documents/mining-pearl/prl-3090-miner
B="$P/bin"; V="$P/.venv"; U=rpcuser; PW=rpcpass; DD="$HOME/prlsimnet"
NODE="--simnet --notls -u $U -P $PW --rpcserver=127.0.0.1:18556"

pkill -f 'bin/pearld' 2>/dev/null || true; pkill -f 'bin/oyster' 2>/dev/null || true
pkill -f 'pearl-gateway' 2>/dev/null || true; sleep 1
rm -rf "$DD"; mkdir -p "$DD/node" "$DD/wallet"

echo ">> 1. pearld --simnet (no mining yet)"
nohup "$B/pearld" --simnet --notls -u "$U" -P "$PW" --rpclisten=127.0.0.1:18556 \
  --datadir="$DD/node" --txindex > /tmp/pearld.log 2>&1 &
sleep 7
echo ">> 2. oyster temp wallet -> mining address"
nohup "$B/oyster" --simnet --createtemp --appdata="$DD/wallet" -u "$U" -P "$PW" \
  --noservertls --noclienttls --rpcconnect=127.0.0.1:18556 \
  --pearldusername="$U" --pearldpassword="$PW" > /tmp/oyster.log 2>&1 &
sleep 10
ADDR="$("$B/prlctl" --simnet --wallet --notls --rpcserver=127.0.0.1:18554 -u "$U" -P "$PW" getnewaddress)"
echo "   mining address: $ADDR"; echo "$ADDR" > /tmp/prl_addr.txt

echo ">> 3. restart pearld with --miningaddr"
pkill -f 'bin/pearld'; sleep 2
nohup "$B/pearld" --simnet --notls -u "$U" -P "$PW" --rpclisten=127.0.0.1:18556 \
  --datadir="$DD/node" --txindex --miningaddr="$ADDR" > /tmp/pearld.log 2>&1 &
sleep 6
echo "   height: $("$B/prlctl" $NODE getblockcount)"

echo ">> 4. pearl-gateway (TCP 8337). torch import ~25s; wait for bind"
export PEARLD_RPC_URL=http://127.0.0.1:18556 PEARLD_RPC_USER="$U" PEARLD_RPC_PASSWORD="$PW"
export PEARLD_MINING_ADDRESS="$ADDR" MINER_RPC_TRANSPORT=tcp MINER_RPC_PORT=8337 MINER_RPC_HOST=127.0.0.1
export PEARL_LOG_LEVEL=INFO CUDA_VISIBLE_DEVICES=""
nohup setsid stdbuf -oL -eL "$V/bin/pearl-gateway" start > /tmp/gw.log 2>&1 &
sleep 45
ss -tlnp 2>/dev/null | grep -q ':8337' && echo "   gateway listening on 8337" || { echo "   gateway not up"; tail /tmp/gw.log; exit 1; }

echo ">> 5. run the real miner -> submit a real PlainProof"
H0="$("$B/prlctl" $NODE getblockcount)"
"$V/bin/python" "$MINER/miner/simnet_solo.py"
echo ">> 6. wait for async block build + check acceptance"
for i in 1 2 3 4 5 6; do sleep 10; H="$("$B/prlctl" $NODE getblockcount)"; echo "   t=$((i*10))s height=$H (was $H0)"; done
grep -iE 'accepted|reject|Submitting block' /tmp/gw.log | tail -5
