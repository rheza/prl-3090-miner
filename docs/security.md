# Security (PRD §17, §26)

## Key handling — the hard rule
- The miner **never** asks for, reads, or stores a seed phrase or private key. It only ever uses the
  **public** Taproot mining address (`prl1p...`) and local RPC credentials.
- The wallet daemon **Oyster** holds the seed and signs. The miner address is produced by
  `prlctl getnewaddress` and is public by construction (`docs/create-wallet.md`).
- `miner/config.py` refuses to load a config whose `mining_address` contains `xprv`/`seed`/`mnemonic`,
  and warns if it does not look like a `prl1p...` P2TR address.

## Credentials
- The pearld RPC password is read from the **environment variable** named by `node.rpc_password_env`
  (default `PEARLD_RPC_PASSWORD`) — never stored in `miner.toml`.
- `miner/config.py` warns if `miner.toml` is world-readable/writable (`stat` check, PRD §26). `chmod 600`.
- `.gitignore` excludes `miner.toml` and `*.local.toml` so secrets never get committed.

## Network / process hygiene
- The gateway miner-RPC is **local only**: UDS `/tmp/pearlgw.sock` (chmod 0600) or TCP bound to
  `127.0.0.1` (`server.py:106`). The miner connects to localhost; do not expose 8337 publicly.
- No remote telemetry by default (PRD §17). No auto-update. The process name is not hidden; logs and the
  status line are always visible.
- Logs never print secrets (no RPC password, no key material — there is none to print).

## GPU safety (PRD §17, §25.4)
- The miner applies **no** power/clock change unless `gpu.apply_overclock = true` is explicitly set.
- Thermal throttling: the loop skips GPU work whenever NVML reports `temp_c ≥ max_temp_c` or
  `vram_temp_c ≥ max_vram_temp_c` (`SafetyMonitor.check_thermal`). RTX 3090 GDDR6X VRAM runs hot — keep
  `max_vram_temp_c` ≤ 96 and ensure backplate cooling.
- Invalid-proof circuit breaker: after `exit_after_invalid_proofs` rejected/invalid proofs, the miner
  exits rather than spamming the node (PRD §17).

## Releases (PRD §26)
- Publish SHA-256 checksums for any binary artifacts; prefer reproducible builds.
- This project uses the official Pearl protocol as a **reference only**. It does not bundle, patch, copy,
  or decompile AlphaMiner or any private binary (PRD §5, §25). `scripts/compare_alphaminer.sh` runs
  AlphaMiner purely as an external black box and inspects only its self-reported hashrate. AlphaPool
  protocol notes are limited to our own line-delimited JSON traffic, public documentation, and
  operator-provided information; no private binary internals are used.

## Interoperability research (PRD §5, §24, §25)
The project may add pool support through legitimate interoperability work:

- allowed: public protocol documentation, official Pearl source, our own line-delimited JSON traffic,
  our own miner's behavior, pool-visible accepted/rejected share results, and written operator-provided
  challenge/auth specifications;
- allowed: black-box compatibility tests that connect using our own public payout address and ordinary
  miner credentials, provided they do not submit malformed traffic at scale or evade pool limits;
- allowed: implementing `pearl.challenge_response` only when the algorithm is public, independently
  specified by the pool/operator, or otherwise authorized for this project;
- not allowed: extracting secrets, credentials, or challenge algorithms from private binaries without
  authorization, copying proprietary code, patching another miner, or bypassing anti-DDoS/access-control
  checks.

Any private-binary inspection requires explicit written permission from the rights holder/operator and a
separate design note documenting scope, artifacts inspected, and why the result is safe to include.

## Authorization
Mining on hardware you own, against your own node/pool, is the intended use. Do not point the miner at
pools or nodes you are not authorized to use, and do not use it to bypass pool/anti-DDoS controls
(PRD §5).
