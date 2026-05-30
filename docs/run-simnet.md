# Run on SimNet first (the right place to validate)

**Why SimNet:** `checkBlockSanity` sets `BFNoPoWCheck` when the net is SimNet
(`node/blockchain/validate.go:421-423`), and `SolveBlock` emits a dummy certificate
(`solve.go:17-19`). So on SimNet the node **accepts blocks without verifying the ZK PoUW proof**. That
lets you exercise the entire **mine → submit → accept** loop end to end — gateway wiring, job/stale
handling, the GPU kernels' `found_block` path — **without** the proprietary `zkpow`/`py-pearl-mining`
proving stack. This is exactly the development-first ordering the PRD §14.2 mandates.

## One command
```bash
./scripts/run_simnet.sh            # starts oyster + pearld(simnet) + pearl-gateway + the miner
```
It will create a wallet on first run, generate a mining address, start the node on RPC 18556, start the
gateway (UDS `/tmp/pearlgw.sock`), and launch `prl3090-miner run`.

## Manual
```bash
export PEARLD_RPC_PASSWORD=rpcpass
./bin/oyster -u rpcuser -P rpcpass --simnet &
ADDR=$(./bin/prlctl -u rpcuser -P rpcpass --simnet -s https://localhost:18554 getnewaddress)
./bin/pearld --simnet --notls --rpcuser=rpcuser --rpcpass=rpcpass \
             --rpclisten=127.0.0.1:18556 --miningaddr="$ADDR" --txindex &

export PEARLD_RPC_URL=http://127.0.0.1:18556 PEARLD_RPC_USER=rpcuser PEARLD_MINING_ADDRESS="$ADDR"
pearl-gateway start &

cp config/miner.example.toml miner.toml      # set mode="simnet", wallet.mining_address=$ADDR,
                                             # gateway.transport="uds", backend="cuda-sm86" (or "cpu")
python -m miner run --config miner.toml
```

## What success looks like
- `getMiningInfo` returns a job (no `-32001 mining_paused`).
- The miner logs `new_job`, then `submit: {'ack': 'submitted'}`.
- The node accepts the block; a new tip appears and the miner switches jobs.
- With `backend="cpu"` you can validate the whole loop today (the CPU backend is a harness — see
  `STATUS.md`); with `backend="cuda-sm86"` you validate the real GPU kernels once they pass golden.

Then graduate to **testnet** (real PoW + ZK) before **mainnet** (`run-mainnet-solo.md`).
