# Pearl PRL Mining Protocol Notes (Milestone 1 / Task 2)

> **Status: grounded in source, not invented.** Every claim below cites the file and
> line in the official checkout at `../pearl-official` (`github.com/pearl-research-labs/pearl`,
> HEAD `9e8b2263`). Where a fact could **not** be verified from source, it is marked
> **[UNVERIFIED]** and the reason is given. Do not promote an UNVERIFIED item to a code
> assumption without reading the named package.

This document is the single source of truth for what `prl3090-miner` must implement and
interoperate with. The companion runnable spec is [`reference/pearl_reference.py`](../reference/pearl_reference.py)
(a faithful NumPy port of the algorithm) plus the golden vectors in [`tests/golden/`](../tests/golden/).

---

## 0. What Pearl is, in one paragraph

Pearl is an L1 blockchain forked from **btcd / btcwallet / neutrino** (`../pearl-official/README.md:160-168`).
It replaces Bitcoin's SHA-256d proof-of-work with **Proof-of-Useful-Work**: the "work" is an
honest **int8 matrix multiply `C = A·B`** (the matmuls of real LLM inference, run through vLLM —
`README.md:12-15`, paper [arXiv:2504.09971](https://arxiv.org/abs/2504.09971)). The PoW lottery is
embedded in the *intermediate partial sums* of that matmul, so a miner cannot win without actually
performing the useful computation. A separate **Plonky2/STARKy ZK certificate** proves to the
network that the claimed matmul was done correctly, without the verifier redoing it.

The "useful work" is genuine: the production miner loads `pearl-ai/Llama-3.3-70B-Instruct-pearl`
and mines as a by-product of serving inference (`README.md:122-124`).

---

## 1. Repository topology and the absent packages

| Component | Path | Language | Role |
|---|---|---|---|
| **pearld** (full node) | `node/` | Go | consensus, RPC, `getblocktemplate`/`submitblock` |
| **Oyster** (wallet) | `wallet/` | Go | HD wallet, generates the Taproot mining address |
| **pearl-gateway** | `miner/pearl-gateway/` | Python | bridges node JSON-RPC ↔ miner; **does all the hard protocol work** |
| **miner-base** | `miner/miner-base/` | Python (torch) | reference NoisyGEMM / noise / hashing |
| **pearl-gemm** | `miner/pearl-gemm/csrc/` | CUDA/C++ | the GPU kernels (**Hopper sm_90a only** today) |
| **py-pearl-mining** | `py-pearl-mining/` | Rust/PyO3 | compiled bindings: `MiningConfiguration`, `MerkleTree`, `PlainProof`, `IncompleteBlockHeader`, `generate_proof`, `verify_proof` |
| **zk-pow** | `zk-pow/` | Rust (Plonky2) | the ZK circuit + verifier |

**Three things are imported but NOT present as source in this checkout** — do not assume their
internals:

1. **`node/wire`** — `BlockHeader`, `MsgBlock`, `MsgCertificate`, `ZKCertificate`, and all wire
   (de)serialization + size/version constants. Imported by 26+ files (e.g. `node/blockchain/solve.go:8-9`)
   but absent on disk and untracked in git. **[UNVERIFIED: exact header byte layout, nonce width,
   `ProofCommitment` size, `PublicDataSize`, `CertificateMaxSize`.]**
2. **`node/zkpow`** — `Mine(header)` and `VerifyCertificate(header, cert)`, the actual PoUW + ZK
   logic. Only two call sites exist: `node/blockchain/solve.go:20` and `node/blockchain/validate.go:333`.
   **[UNVERIFIED: the on-chain verification math and proof system parameters.]**
3. **`py-pearl-mining`** Rust extension — present as a directory name but the proof-object internals
   (`PlainProof` layout, `generate_proof`) are compiled. The Python that *calls* it is readable.

This is not a blocker for the miner (see §7): on **SimNet the ZK check is skipped**, so the full
mine→submit→accept loop is testable end to end without `zkpow`.

---

## 2. System topology

```
                JSON-RPC / HTTP (Basic auth)            line-delimited JSON-RPC
                  port 44107 (mainnet)                  UDS /tmp/pearlgw.sock  OR  TCP 127.0.0.1:8337
   ┌─────────┐  ◀───────────────────────────  ┌───────────────┐  ◀──────────────────────  ┌───────────────┐
   │ pearld  │   getblocktemplate / submitblock │ pearl-gateway │   getMiningInfo / submitPlainProof │  prl3090-miner │
   │ (node)  │  ───────────────────────────▶   │               │  ──────────────────────▶  │  + CUDA sm_86 │
   └─────────┘                                  └───────────────┘                           └───────────────┘
        │                                              │                                            │
   consensus,                                   polls node every                              GPU NoisyGEMM,
   zkpow.Verify                                 PEARLD_REFRESH_INTERVAL_SECONDS (1s),         finds PlainProof
                                                builds PlainProof→block, submits
```

**Design consequence (the most important takeaway):** `pearl-gateway` already implements *all* node
integration — block-template polling, work caching, stale detection, ZK proof generation, block
assembly, and `submitblock`. **A 3090 miner does not reimplement any of that.** It implements exactly
two things: (a) a client of the gateway's miner-facing JSON-RPC, and (b) the sm_86 CUDA backend.
See [`docs/architecture.md`](architecture.md).

---

## 3. Job lifecycle (node ↔ gateway)

### 3.1 Gateway → node: fetch work
- Method **`getblocktemplate`** (BIP22/23), handler `handleGetBlockTemplate` at `node/rpcserver.go:152`
  (impl `:2131`); registered in `rpcHandlersBeforeInit` (`node/rpcserver.go:136`).
  Gateway call site: `pearl_client.py:88-101`, params
  `{"capabilities":["coinbasevalue","workid","coinbase/append"],"rules":["segwit"]}`.
- Response consumed via `GetBlockTemplateResponse` (`rpc_types.py:24-43`): `bits, curtime, height,
  previousblockhash, target, transactions[], version, longpollid, coinbasevalue, …`.
  **No PoUW/ZK field is in the GBT response** (`node/rpcserver.go:1727`) — the certificate is computed
  out of band by the gateway.
- **`getwork` is NOT implemented** (`node/rpcserver.go:244`, in `rpcUnimplemented`). Do not use it.

### 3.2 Gateway work cache & new-tip detection
- `TemplateScheduler` polls every `PEARLD_REFRESH_INTERVAL_SECONDS` (default **1s**) — `scheduler.py:56-69`,
  `config.py:23`. **Polling only; no long-poll/subscription** even though `longpollid` is parsed
  (`rpc_types.py:35`, unused).
- `WorkCache.update_template` treats a template as new **only when `previousblockhash` changes**
  (`work_cache.py:28-38`). Same-tip `curtime` bumps are ignored.

### 3.3 Miner ↔ gateway: the interface `prl3090-miner` must speak
Transport (`miner_rpc/server.py:80-109`, `config.py:27-49`): line-delimited JSON-RPC 2.0, one JSON
object per `\n`, **no auth** (local only), 1 MiB max line. Selected by `MINER_RPC_TRANSPORT`:
- `uds` (default): Unix socket `MINER_RPC_SOCKET_PATH` = **`/tmp/pearlgw.sock`**, chmod `0600`.
- `tcp`: bound hard to **`127.0.0.1`** on `MINER_RPC_PORT` = **8337**.

Two methods:

| Method | Params | Result | Source |
|---|---|---|---|
| **`getMiningInfo`** | `{}` | `{"incomplete_header_bytes": <base64>, "target": <int>}` | `server.py:208-212` |
| **`submitPlainProof`** | `{"plain_proof": <base64>, "mining_job": {incomplete_header_bytes, target}}` | `"submitted"` (immediate, **fire-and-forget**) | `server.py:214-220` |

- The **MiningJob** wire shape is `{incomplete_header_bytes (b64), target (int)}` (`dataclasses.py:148-200`,
  schema `schemas.py:35-42`). `target` is the uint256 difficulty (lower = harder), `target = bits_to_target(bits)`.
- `MiningJob.INNER_HASH_LIMIT = 42`, `MAX_TARGET = 2**256-1` (`dataclasses.py:155-156`).
- When no template is cached, `getMiningInfo` returns JSON-RPC error **-32001 `mining_paused`**
  (`dataclasses.py:203-211`). Treat this as "back off and retry", not a fatal error.
- **`submitPlainProof` never returns accept/reject** — the gateway acks `"submitted"` and resolves the
  block submission asynchronously; the final `accepted`/`rejected:<reason>` is only *logged* on the
  gateway (`server.py:259`). A miner therefore tracks "submitted" locally and observes acceptance
  indirectly (a new tip appears, or gateway logs). Plan metrics accordingly.

### 3.4 Stale / cancel
There is **no explicit cancel message**. Staleness is by header identity:
1. New tip → `WorkCache` swaps template (`work_cache.py:28-38`); miner sees it on its next `getMiningInfo` poll.
2. Gateway drops a submitted proof whose `incomplete_header_bytes` ≠ current template
   (`server.py:247-252`) — silent drop.
3. `SubmissionService` dedups already-submitted headers (`submission_service.py:38-40`).

**Miner obligation (PRD `stale_cancel`):** poll `getMiningInfo` frequently (≈ the 1s cadence), and the
instant `incomplete_header_bytes` changes, abandon in-flight GPU work for the old header and restart on
the new one. Submitting against a stale header is wasted work (gateway drops it).

---

## 4. The PoUW algorithm (NoisyGEMM)

Runnable, audited port: [`reference/pearl_reference.py`](../reference/pearl_reference.py). Official source:
`miner-base/src/miner_base/{noisy_gemm,noise_generation,inner_hash,commitment_hash}.py`.

### 4.1 Datatypes, shapes, constants
- Compute is **int8 × int8 → int32** (`MMAType.Int7xInt7ToInt32`, `gpu_matmul_config.py:27`). Data values
  are constrained to **int7 `[-64, 63]`**; the noise occupies the rest of the int8 range
  (`noisy_gemm.py:156-164`).
- Shapes: `A: m×k`, `B: k×n`, rank `r = noise_rank`; `E_AL: m×r`, `E_AR: r×k`, `E_BL: k×r`, `E_BR: r×n`;
  output `C: m×n` int32.
- Production constants (`settings.py`): `noise_range=128`, `noise_rank=128`, GEMM tile
  `tile_size_m=128, tile_size_n=256, tile_size_k=128`, `hash_tile = 16×16`,
  `rows_pattern=[0,8]`, 64-entry `cols_pattern`.
- PoW/transcript constants (`noisy_gemm.py:17-24`): `TRANSCRIPT_SIZE_U32=16` (64-byte transcript),
  `HASH_ACCUMULATE_ROTATION=13` (matches `pow_utils.hpp:15`), `POW_TARGET_HARDEST=0`,
  `POW_TARGET_EASIEST=2**256-1`. Generator constraints: `noise_range`,`noise_rank` powers of two,
  `noise_range ≤ 128`, `noise_rank % 32 == 0` (`noise_generation.py:31-41`).

### 4.2 Noise generation (deterministic, keyed BLAKE3) — `noise_generation.py:54-187`
Inputs: `key_A`, `key_B` (32 bytes each), dims. Fixed seeds `seed_A=b"A_tensor"+\x00*24`,
`seed_B=b"B_tensor"+\x00*24` (`:71-80`). Per draw: 8×int32 little-endian array with slot
`prepend_index` set to `1+index`, then seed, then `blake3(msg, key=key)` (`:101-105`).
- **Dense uniform** `E_AL` (m×r) and `E_BR` (n×r then transposed → r×n): byte `b` → `(b & 63) - 32`,
  i.e. values in **[-32, 31]** for `noise_range=128` (`:107-135, 48-52`). `prepend_index=0`.
- **Signed-permutation** `E_AR` (r×k, per **column**) and `E_BL` (k×r, per **row**): per line draw u32,
  `first_idx = u & (r-1)`, `second_idx = first_idx ^ (1 + mulhi_u32(r-1, u))`; place `+1` at `first_idx`,
  `-1` at `second_idx` (`:137-187`). `prepend_index=1`. The CUDA `noise_generation_kernel.h:36-57`
  matches this exactly.

The low-rank structure guarantees `E_A = E_AL·E_AR` and `E_B = E_BL·E_BR` land in `[-63, 62] ⊂ [-64,64)`,
so the int8 cast is lossless and the denoising below is exact.

### 4.3 The pipeline — `noisy_gemm.py:598-696`
1. `A_noised = A + E_A` (int8), `A_E_BL = A·E_BL` (int32) — `noise_A`, `:175-228`.
2. `B_noised = B + E_B` (int8), `EAR_BpEB = E_AR·B_noised` (int32) — `noise_B`, `:230-283`.
3. **Tiled noisy GEMM** `_tiled_matmul`, `:461-520`: output tiles step by `noise_rank` in m and n; within
   a tile, accumulate over k in `noise_rank`-wide chunks. **After each full k-chunk, hash the running
   `C_block`** and fold per-16×16-tile inner hashes into transcripts (`:400-419`).
4. **Inner hash** (`inner_hash.py:7-32`): per 16×16 tile, `hash = XOR-reduce(int32 elements viewed as
   uint32)`. (XOR ⇒ order-independent.)
5. **Transcript accumulation** (`:38-45`): `data[c % 16] = rotl32(data[c % 16], 13) ^ inner_hash`, where
   `c` is the k-reduction index.
6. **Denoise** (`:588-594`): `C = C_noised − A_E_BL·E_BR − E_AL·EAR_BpEB`. Algebraically `C ≡ A·B`
   **exactly** (verified by golden vectors and `test_reference.py::test_denoise_identity`).
7. **Win check** (`_check_pow_target`, `:309-326`): per 16×16 hash tile, pack the 16 transcript words
   little-endian → 64 bytes → `blake3(buf, key=pow_key).digest()` → `int.from_bytes(.,"little") ≤ target`.
   **`pow_key = commitment_hash.noise_seed_A`** (`:683`).
8. On a win, record `A_row_indices = [m … m+16)`, `B_column_indices = [n … n+16)` and stash the
   **non-noised** `A`, `Bᵀ`, and `commitment_hash` for proof creation (`:328-350, 687-694`).

### 4.4 Commitment hash chain — `commitment_hash.py`
- `key = blake3(incomplete_header_bytes ‖ mining_config.to_bytes())` (unkeyed, `:16`).
- `A_root = MerkleTree(A, key)`, `B_root = MerkleTree(Bᵀ, key)` (Bᵀ so a column-strip is openable, `:27-30`).
- `commitment_B = blake3(key ‖ B_root)` (`:36`); `commitment_A = blake3(commitment_B ‖ A_root)` (`:40`).
- `CommitmentHash(noise_seed_A = commitment_A, noise_seed_B = commitment_B)`. **`noise_seed_A` is the
  PoW key** *and* the seed for A's noise; `noise_seed_B` seeds B's noise.
- Merkle leaf = 1024-byte chunk (`CHUNK_SIZE`, `merkle_tree_roots_kernel.hpp:99`); root via keyed BLAKE3
  (`matrix_merkle_tree.py:33,47`). **[UNVERIFIED: multi-leaf tree internal node hashing — needs
  `py-pearl-mining` `MerkleTree`.]**

### 4.5 Hash usage summary
| Step | Call | Bytes |
|---|---|---|
| commitment key | `blake3(hdr ‖ cfg.to_bytes())` unkeyed | variable |
| merkle root | `blake3(padded_tensor, key=key)` | 1024·ceil |
| commitment_B / _A | `blake3(key ‖ B_root)` / `blake3(cmtB ‖ A_root)` | 64 |
| noise RNG | `blake3(8×i32 ‖ seed, key=key)` | 40 |
| inner hash | XOR-reduce (NOT a hash) | per 16×16 tile |
| **PoW** | `blake3(16×u32 LE, key=noise_seed_A)` | 64 |

---

## 5. The proof object and submission

### 5.1 PlainProof — `block_submission.py:11-50`
On a win, the miner builds Merkle openings of the winning rows of `A` and columns of `Bᵀ`:
`PlainProof(m, n, k, noise_rank, a_merkle_proof, bt_merkle_proof)`. Submitted base64 via
`submitPlainProof` (`gateway_client.py:30-40`). **[UNVERIFIED: PlainProof binary layout — compiled in
`py-pearl-mining`. From the miner's side it is an opaque base64 blob produced by that library.]**

### 5.2 Gateway → node block assembly — `proof_generator.py:15-43`, `submission_service.py:29-68`
1. `zk_proof = generate_proof(incomplete_header, plain_proof)` (py-pearl-mining).
2. `ZKCertificate.from_pearl_header(header, zk_proof)` sets `header.proof_commitment =
   double_sha256(version ‖ public_data)` (`zk_certificate.py:84-87`).
3. `PearlBlock.serialize()` = `ZK_CERTIFICATE ‖ BLOCK_HEADER ‖ TX_COUNT(varint) ‖ TXS`
   (`pearl_block.py:23-31`). Header on wire = `incomplete_header.to_bytes()` (version, prev_block,
   merkle_root, timestamp, nbits) ‖ 32-byte `proof_commitment` (`pearl_header.py:59-62`).
4. `submitblock [block_hex]` (`pearl_client.py:103-114`): result `null` ⇒ `"accepted"`; any string ⇒
   `"rejected: <reason>"`.
- ZK certificate struct: `{version:u32=1, header_hash:32, public_data:PUBLICDATA_SIZE,
  proof_data_len:u32, proof_data:≤60000}` (`zk_certificate.py:17-52`).

### 5.3 Node-side validation — `node/blockchain/validate.go`
`submitblock` → `handleSubmitBlock` (`rpcserver.go:3688`) → `SyncManager.ProcessBlock`
(`netsync/manager.go:1792`) → `blockchain.ProcessBlock` → `checkBlockSanity` → `checkProofOfWork`
(`validate.go:309`):
1. `target = CompactToBig(header.Bits)`; reject if ≤0 or > `PowLimit`.
2. reject if certificate missing (`ErrCertificateMissing`).
3. `zkpow.VerifyCertificate(header, cert)` (`validate.go:333`) — the single call that does both the
   target/hash check and the PoUW + ZK verification. **[UNVERIFIED internals: `node/zkpow` absent.]**

---

## 6. Consensus facts relevant to mining

- **Difficulty: WTEMA**, retargets **every block** (`difficulty.go:84-132`):
  `new_target = old_target + (t − T)·old_target / half_life`, clamped to `PowLimit`.
- `TargetTimePerBlock = 3m14s` (π minutes), `WTEMAHalfLife = 168h`, `CoinbaseMaturity = 100`
  (e.g. `params.go:287-289`).
- **Ports** (`node/params.go`, `node/chaincfg/params.go`):

  | Net | RPC | P2P | Wallet | PowLimit | Notes |
  |---|---|---|---|---|---|
  | Mainnet | 44107 | 44108 | 44207 | 2²⁰⁸−1 | bits `0x1b00ffff` |
  | Testnet | 44109 | 44110 | 44209 | 2²⁰⁸−1 | ReduceMinDifficulty |
  | Testnet2 | 44111 | 44112 | 44211 | 2²⁰⁸−1 | |
  | Simnet | 18556 | 18555 | 18554 | 2²³³−1 | **PoWNoRetargeting; ZK check bypassed** |
  | Regtest | 18334 | 18444 | 18332 | 2²³³−1 | PoWNoRetargeting |

- **SimNet bypass (critical for dev):** `checkBlockSanity` sets `BFNoPoWCheck` when
  `chainParams.Net == wire.SimNet` (`validate.go:421-423`), and `SolveBlock` returns a dummy 1-byte
  certificate (`solve.go:17-19`). ⇒ **On SimNet you can mine and get blocks accepted without a valid ZK
  proof or even a valid PoW digest** — exactly the right place to validate the GPU pipeline first
  (PRD §14.2).
- Mainnet genesis timestamp `1777280400` = 2026-04-27 (`genesis.go:102`) — Pearl is a brand-new chain.

---

## 7. What this means for `prl3090-miner` (the build contract)

1. **Speak the gateway miner-RPC** (§3.3): `getMiningInfo` → run GPU → `submitPlainProof`. Handle
   `-32001 mining_paused` (back off), poll at ~1s, abandon stale headers immediately.
2. **The GPU backend** must reproduce §4 **bit-for-bit** (matched against [`tests/golden/`](../tests/golden/)).
   The hard part is the sm_86 kernels — see [`docs/cuda-sm86-port.md`](cuda-sm86-port.md).
3. **Proof creation (`PlainProof`) needs `py-pearl-mining`.** Either link it (it's the official path,
   gateway already calls `generate_proof`) or treat the gateway as the proof/block builder and have the
   miner only deliver the winning `(A, Bᵀ, indices, commitment)` — which is exactly what
   `submitPlainProof` carries.
4. **Validate on SimNet first** (§6): no ZK needed, fast acceptance, real end-to-end loop.
5. **Never touch keys.** The wallet (Oyster) holds the seed; the miner only ever sees the public Taproot
   `PEARLD_MINING_ADDRESS` and local RPC creds (PRD §17, §26). The gateway already enforces P2TR bech32m
   validation (`blockchain_utils.py:124-160`).

---

## 8. Exact source index (Milestone 1 exit criterion: "list of exact files/functions used")

**Work generation:** `node/rpcserver.go:2131 handleGetBlockTemplate`, `:1650 blockTemplateResult`;
`mining/mining.go:648 NewBlockTemplate`. Gateway: `pearl_client.py:88 get_block_template`,
`scheduler.py:84 refresh_template`, `work_cache.py:28 update_template`, `:44 get_mining_job`.

**Miner interface:** `miner_rpc/server.py:208 getMiningInfo`, `:214 submitPlainProof`,
`:237 handle_submit_plain_proof`; schemas `miner_rpc/schemas.py:21,27`.

**Proof creation:** `noisy_gemm.py:598 noisy_gemm`, `:461 _tiled_matmul`, `:309 _check_pow_target`;
`inner_hash.py:23 hash_tile`; `noise_generation.py:54 generate_noise_metrices`;
`commitment_hash.py:18 commitment_hash`; `block_submission.py:11 create_proof`.

**Submission:** `proof_generator.py:15 generate_block`, `submission_service.py:29 submit_plain_proof`,
`pearl_client.py:103 submit_block`; node `rpcserver.go:3688 handleSubmitBlock` →
`netsync/manager.go:1792 ProcessBlock` → `blockchain/validate.go:309 checkProofOfWork` →
`:333 zkpow.VerifyCertificate`.

**Difficulty/target:** `blockchain/difficulty.go:84 calcNextRequiredDifficulty`, `:47 CompactToBig`;
`chaincfg/params.go` (per-net `PowLimit`, `TargetTimePerBlock`, `WTEMAHalfLife`).

**GPU kernels (to port):** `pearl-gemm/csrc/gemm/{collective_mainloop,collective_epilogue,
pearl_gemm_kernel,pearl_noisingA_kernel,pearl_noisingB_kernel}.h*`; portable:
`noise_generation_kernel.h`, `denoise_converter_kernel.h`, `quantize_kernel.cu`, `inner_hash_kernel.cu`,
`blake3/blake3.cu`. Build target `setup.py:89` (`arch=compute_90a,code=sm_90a`).
```
```
