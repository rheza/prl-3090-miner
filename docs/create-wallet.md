# Create a wallet & mining address (Oyster)

The **miner never touches keys** — Oyster (the HD wallet daemon) holds the seed and gives you a
**public** Taproot mining address. (pearl-official/README.md §1.)

```bash
# Create the wallet (you set a passphrase and RECORD THE SEED yourself — the miner never sees it)
./bin/oyster -u rpcuser -P rpcpass --create        # add --simnet/--testnet for non-mainnet

# Start the wallet server, then generate a Taproot (P2TR) mining address
./bin/oyster -u rpcuser -P rpcpass &
./bin/prlctl -u rpcuser -P rpcpass -s https://localhost:44207 getnewaddress
# -> prl1p...   (this is what goes in miner.toml `wallet.mining_address` and PEARLD_MINING_ADDRESS)
```

Wallet server ports: mainnet **44207**, testnet 44209, testnet2 44211, simnet 18554 (protocol-notes §6).

Rules enforced downstream (`blockchain_utils.py:124-160`): the mining address must be a valid Pearl
**bech32m P2TR**, witness v1, 32-byte program (`prl1p...`). `miner/config.py` rejects anything that looks
like a key/seed.

**Back up your seed offline.** The miner cannot recover funds; only the wallet seed can.
