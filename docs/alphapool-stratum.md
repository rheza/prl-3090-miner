# AlphaPool Stratum Notes

These notes are from a clean-room black-box network capture of a normal AlphaMiner client connection on
2026-06-01. They document the public JSON messages on our own pool connection; no AlphaMiner binary was
decompiled, disassembled, patched, or copied.

## Current conclusion

The socket protocol is now mapped well enough to implement the submit path, but this miner cannot submit
valid AlphaPool shares yet.

Two blockers remain:

1. AlphaPool requires an undocumented `pearl.challenge_response` before `mining.authorize`.
2. The current `cuda-mine` backend still returns small `"_harness": true` JSON proof bytes. AlphaPool
   expects a base64 Pearl proof payload around 276,288 bytes raw / 368,384 bytes base64 for the
   observed job shape.

Do not point the current harness proof output at AlphaPool and interpret rejects as a speed result. It is
not a valid share payload.

## Observed handshake

AlphaPool sends the challenge before ordinary Stratum messages:

```json
{"id": null, "method": "pearl.challenge", "params": {"seed": "<hex64>", "difficulty": 32}}
```

AlphaMiner answers:

```json
{"id": 1, "method": "pearl.challenge_response", "params": {"seed": "<same seed>", "nonce": "<hex16>"}}
```

When accepted, the server returns:

```json
{"id": 1, "result": true, "error": null}
```

Then the client negotiates Pearl shares and authorizes:

```json
{"id": 2, "method": "mining.configure", "params": [["pearl/v1"], {}]}
{"id": 3, "method": "mining.subscribe", "params": ["alpha-miner/1.6"]}
{"id": 4, "method": "mining.authorize", "params": ["prl1p...worker", "x;d=32768"]}
```

The server confirms Pearl share format:

```json
{"jsonrpc": "2.0", "id": 2, "result": {"pearl/v1": true, "pearl/v1.share_format": "base64"}}
```

It then sends mining parameters:

```json
{
  "method": "pearl.set_mining_params",
  "params": [{
    "m": 131072,
    "n": 131072,
    "k": 4096,
    "rank": 128,
    "rows_pattern": [0, 32],
    "cols_pattern": [0, 1, "...", 63],
    "mma_type": "Int7xInt7ToInt32"
  }]
}
```

`mining.notify` has this observed positional form:

```json
{
  "method": "mining.notify",
  "params": [
    "<job_id>",
    "<prev_hash_hex>",
    "<header_blob_hex>",
    65612,
    "<ntime_hex>",
    "<share_nbits_hex>",
    true
  ]
}
```

Submit shape:

```json
{"id": 5, "method": "mining.submit", "params": ["prl1p...worker", "<job_id>", "<base64_plain_proof>"]}
```

## Probe command

`scripts/probe_alphapool.py` confirms the endpoint and shows the current challenge blocker without
submitting shares:

```bash
python3 scripts/probe_alphapool.py \
  --pool stratum+tcp://us2.alphapool.tech:5566 \
  --address prl1pYOUR_PUBLIC_ADDRESS \
  --worker prl3090 \
  --password 'x;d=32768' \
  --redact
```

Expected current result against AlphaPool is `alphapool_challenge_solver_missing`. That means the pool
connection is reachable, but this open miner cannot pass AlphaPool auth until the challenge algorithm is
publicly specified or independently reimplemented without private-binary reverse engineering.

## Implementation path

To make `prl3090-miner` submit real AlphaPool shares:

1. Implement the `pearl.challenge_response` solver from a public specification or clean independent
   implementation.
2. Replace harness proof bytes with real `PlainProof` bytes:
   - generate or ingest full-size A/B tensors for the pool mining parameters,
   - produce Merkle roots/proofs for the opened rows/columns,
   - serialize through `py-pearl-mining` compatible `PlainProof` bytes/base64.
3. Add an AlphaPool mode that maps `pearl.set_mining_params` / `mining.notify` into GPU work, submits
   `mining.submit`, and tracks accepted/rejected/stale shares for protocol TH/s.
4. Run at least 1 hour on the same pool region/static difficulty before comparing to AlphaMiner.
