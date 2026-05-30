# reference/ — CPU reference & golden vectors (Milestone 2)

A faithful, dependency-light (**numpy + blake3 only**, no torch/CUDA/Rust) port of the official Pearl
PoUW reference, plus a deterministic golden-vector generator. This is the **correctness oracle** for the
CUDA `sm_86` backend.

| File | Purpose |
|---|---|
| `pearl_reference.py` | Faithful NumPy port of `noisy_gemm.py` + `noise_generation.py` + `inner_hash.py`. Every step cites the official `file:line`. |
| `generate_golden.py` | Emits `tests/golden/*.npz` (exact int8 inputs + int32 `C`) and `manifest.json` (expected `found_block`, winning indices, the winning tile's 16 PoW transcript words, BLAKE3(C)). |
| `test_reference.py` | 27 tests incl. the official `test_noisy_gemm` truth table, the denoise identity `C==A·B`, noise-matrix structure, and golden replay. |
| `verify_against_official.py` | Cross-checks this NumPy port against the **official torch** reference (run inside WSL where `miner_base` is importable). Closes the correctness loop. |

## Run
```bash
pip install numpy blake3 pytest
python reference/generate_golden.py
pytest -q reference/
# In the Pearl dev env (WSL), prove equivalence to the official torch reference:
python reference/verify_against_official.py
```

## Why this matters
The mining proof hashes **GEMM-intermediate partial sums**, so a "fast but wrong" GPU kernel fails
*silently* (plausible-looking but invalid PoW). The golden vectors pin `C`, `found_block`, the winning
row/col indices, **and the 16 transcript words**, so the CUDA port (esp. its accumulator fragment
layout — the #1 risk in `docs/cuda-sm86-port.md` §5) is validated bit-for-bit before it is ever trusted
on mainnet.
