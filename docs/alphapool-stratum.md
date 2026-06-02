# AlphaPool Stratum Notes

These notes are from a clean-room black-box network capture of a normal AlphaMiner client connection on
2026-06-01. They document the public JSON messages on our own pool connection; no AlphaMiner binary was
decompiled, disassembled, patched, or copied.

## Current conclusion

The socket protocol is mapped well enough to know the submit shape, and the proof content path is now
proven on the official Pearl side: `fast_mine.py` can hand a GPU-found winner to the official
`PlainProof` builder and get a live `pearld` node to accept the resulting block through
`submitPlainProof`.

AlphaPool submission is still not available because the connection is blocked before
`mining.authorize` by AlphaPool's undocumented `pearl.challenge_response` handshake. This project does
not reverse engineer private binaries or bypass that gate. Once an authorized/public challenge solver is
available, the submit envelope is:

```json
{"id": 5, "method": "mining.submit", "params": ["prl1p...worker", "<job_id>", "<base64_plain_proof>"]}
```

Do not interpret current AlphaPool probe failures as proof rejects. The client has not reached
authorization or submitted a share.

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

Latest verified probe in this repo on 2026-06-02, using the provided public payout address and
`x;d=32768`, reached `pearl.challenge` (`difficulty=32`) and stopped before authorization. No share was
submitted.

## Implementation path

To make `prl3090-miner` submit real AlphaPool shares:

1. Implement the `pearl.challenge_response` solver from a public specification, an operator-provided
   authorization path, or another legitimate source that does not require bypassing AlphaPool access
   control.
2. Add an AlphaPool mode that maps `pearl.set_mining_params` / `mining.notify` into GPU work, submits
   `mining.submit`, and tracks accepted/rejected/stale shares for protocol TH/s.
3. Run at least 1 hour on the same pool region/static difficulty before comparing to AlphaMiner.
