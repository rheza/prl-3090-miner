# Troubleshooting (PRD §27)

| Symptom | Likely cause | Fix |
|---|---|---|
| **GPU not detected** (`list-devices` empty) | driver/WSL not exposing the GPU | `nvidia-smi` on host; in WSL install CUDA-on-WSL toolkit (not the Linux driver). Driver 545+. |
| **CUDA version mismatch** at build | toolkit ≠ what torch/cutlass expects | use CUDA 12.x; `nvcc --version`; the official build pins CUDA 13/sm_90a — our `cuda/` retargets to 12.x/sm_86 (`cuda-sm86-port.md` §2). |
| **Driver too old** | <545 | update NVIDIA driver. |
| **`cuda-sm86 backend NOT BUILT`** | kernels not compiled (expected today) | `scripts/build_sm86.sh` in WSL; until M4 the backend intentionally refuses (see `STATUS.md`). |
| **`mining_paused` (-32001) forever** | node has no template (not synced, wrong tip, no `--miningaddr`) | sync node; pass `--miningaddr`; check `getmininginfo`. |
| **RPC auth failed** | wrong user/pass or `PEARLD_RPC_PASSWORD` unset | export the env var; match `--rpcuser/--rpcpass`; `node.rpc_password_env` in toml. |
| **Invalid address** | not a `prl1p...` P2TR | regenerate via `prlctl getnewaddress`; `config.py` warns/blocks key-like strings. |
| **100% reject** | node not synced, or mining a stale tip | sync to tip (`getblockchaininfo` headers==blocks); ensure the gateway poll sees new prev-hash. |
| **High stale rate** | slow job switching / long kernel vs 3m14s spacing | poll ~1s; honor `stale_cancel`; shorten kernel job-switch latency (`cuda-sm86-port.md` §6). |
| **Low hashrate** | tile doesn't fit sm_86 SMEM, low TC occupancy | retune tiles to ≤99 KB optin SMEM; profile with `scripts/profile_nsight.sh compute`. |
| **VRAM overheating** | GDDR6X hot | lower `gpu.power_limit`, improve backplate cooling, drop `max_vram_temp_c`; miner auto-throttles. |
| **Miner crashes after a new job** | stale-buffer / race in job switch | ensure old GPU work is cancelled before reuse; check `should_stop` plumbing (`job_runner.cu`). |
| **`submitPlainProof` ok but no block** | it's fire-and-forget; accept is async | confirm via `prlctl getbestblockhash` / wallet balance / gateway logs (protocol-notes §3.3). |
| **`UDS unavailable on this platform`** | running the client on Windows | use `transport="tcp"` (gateway must run with `MINER_RPC_TRANSPORT=tcp`) or run inside WSL. |
| **Golden tests fail after kernel edit** | accumulator fragment layout wrong | the silent-correctness hazard — re-derive the m16n8k32 thread map (`cuda-sm86-port.md` §5). |
