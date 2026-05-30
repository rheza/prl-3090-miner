# Mainnet true-solo mining

> **Do not run mainnet until SimNet and testnet pass** (PRD §14.2). Mainnet requires the real ZK
> certificate path, so the CUDA backend must produce bit-exact proofs (validated against
> `tests/golden/`) and `py-pearl-mining` must be installed for `generate_proof`.

## Preconditions checklist
- [ ] `pytest -q reference/ miner/` green, and `cuda/tests/test_cuda_golden.py` green (CUDA backend
      matches golden — `C`, `found_block`, indices, **transcript words**).
- [ ] SimNet loop accepts blocks; testnet loop accepts blocks.
- [ ] Node fully synced (`getblockchaininfo`: headers == blocks).
- [ ] Wallet address is a public `prl1p...` you control; seed backed up offline.
- [ ] Thermals validated for a 24 h run (`rtx3090-tuning.md`), `vram_temp_c` headroom confirmed.

## Run
```bash
export PEARLD_RPC_PASSWORD=...                 # never in the toml
./bin/pearld --rpcuser=rpcuser --rpcpass=$PEARLD_RPC_PASSWORD \
             --rpclisten=127.0.0.1:44107 --miningaddr=<prl1p...> --txindex &

export PEARLD_RPC_URL=http://127.0.0.1:44107 PEARLD_RPC_USER=rpcuser \
       PEARLD_MINING_ADDRESS=<prl1p...>
pearl-gateway start &

cp config/miner.example.toml miner.toml        # mode="solo-local", backend="cuda-sm86"
chmod 600 miner.toml
python -m miner run --config miner.toml
```

## Operating notes
- Difficulty retargets every block; target spacing is 3m14s (protocol-notes §6). Solo on one 3090 means
  blocks are rare — that is expected for true solo.
- `submitPlainProof` is fire-and-forget: the miner sees `submitted`, not the chain accept (protocol-notes
  §3.3). Confirm real acceptance via `prlctl getbestblockhash` / wallet balance / gateway logs.
- The invalid-proof breaker exits after `exit_after_invalid_proofs` to avoid spamming the node — if it
  trips, your kernel is producing bad proofs; go back to golden validation, do **not** raise the limit.
