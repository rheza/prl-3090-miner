# Run a local Pearl node (pearld)

```bash
./bin/pearld \
  --rpcuser=rpcuser --rpcpass=rpcpass \
  --rpclisten=0.0.0.0:44107 \
  --miningaddr=<your prl1p... address> \
  --txindex
```
Flags: `--testnet` / `--simnet` for non-mainnet, `--notls` to disable TLS, `--debuglevel=debug` for
verbose logs. See `pearl-official/node/sample-pearld.conf` for everything.

Ports (protocol-notes §6): mainnet RPC **44107** / P2P 44108; testnet 44109/44110; testnet2 44111/44112;
simnet 18556/18555.

## Sync before mining
Let the node sync to the tip first. Useful checks:
```bash
./bin/prlctl -u rpcuser -P rpcpass getblockchaininfo     # headers vs blocks
./bin/prlctl -u rpcuser -P rpcpass getmininginfo         # difficulty, current height
```
The gateway calls `getblocktemplate`; an unsynced node returns templates off the wrong tip and your
proofs will be stale. Mining a not-synced node is the #1 cause of 100% reject (`troubleshooting.md`).

## Consensus facts worth knowing (protocol-notes §6)
- Difficulty retargets **every block** (WTEMA), target spacing **3m14s**.
- Mainnet `PowLimit = 2²⁰⁸−1`, bits `0x1b00ffff`. Genesis is 2026-04-27 — this is a young chain.
