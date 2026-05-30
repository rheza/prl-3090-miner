"""GPU golden-vector harness: prove the sm_86 backend matches the CPU reference.

Skips if `prl_cuda` is not built (PYTHONPATH must include cuda/build). Until the M4
kernels land, run_noisy_gemm raises "not implemented" and each case xfails cleanly;
once implemented, this enforces bit-for-bit agreement with tests/golden/.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

prl_cuda = pytest.importorskip("prl_cuda", reason="build cuda/ first (scripts/build_sm86.sh)")

GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "tests" / "golden"


def _cases():
    mf = GOLDEN / "manifest.json"
    return json.loads(mf.read_text())["cases"] if mf.exists() else []


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_cuda_matches_golden(case):
    d = np.load(GOLDEN / case["file"])
    job = {
        "m": case["m"], "k": case["k"], "n": case["n"],
        "noise_rank": case["noise_rank"], "noise_range": case["noise_range"],
        "hash_tile_h": case["hash_tile_h"], "hash_tile_w": case["hash_tile_w"],
        "matmul_tile_h": case["matmul_tile_h"], "matmul_tile_w": case["matmul_tile_w"],
        "A": d["A"], "B": d["B"], "E_AL": d["E_AL"], "E_AR": d["E_AR"],
        "E_BL": d["E_BL"], "E_BR": d["E_BR"],
        "pow_key": bytes.fromhex(case["key_A"]),
        "pow_target": int(case["pow_target"], 16),
    }
    try:
        C, found, a_row, b_col, transcript = prl_cuda.run_noisy_gemm(job)
    except RuntimeError as exc:
        if "not implemented" in str(exc).lower():
            pytest.xfail("sm_86 kernels not implemented yet (M4) — see docs/cuda-sm86-port.md")
        raise

    assert np.array_equal(np.asarray(C), d["C"])
    assert bool(found) is case["found_block"]
    if case["found_block"]:
        assert a_row == case["A_row_indices"][0]
        assert b_col == case["B_column_indices"][0]
        assert [hex(int(w)) for w in transcript] == case["transcript_words"]
