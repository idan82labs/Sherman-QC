from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from backend import f1_ordered_projective_gauss_transport as opgt
from backend.f1_candidate_prism_geometry import canonical_triangle_mesh


def _projective_measure(phases: np.ndarray, weight: float = 10_000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.full(len(phases), weight, dtype=np.float64)
    real = weights * np.cos(2.0 * phases)
    imag = weights * np.sin(2.0 * phases)
    return weights, real, imag


def test_streaming_dp_prefers_monotone_measure() -> None:
    monotone = np.linspace(-0.55, 0.55, 25)
    reversal = monotone.copy()
    reversal[10:15] = reversal[10:15][::-1]
    mw, mr, mi = _projective_measure(monotone)
    rw, rr, ri = _projective_measure(reversal)

    monotone_cost, monotone_path, _ = opgt._monotone_fit(range(25), mw, mr, mi)
    reversal_cost, _, _ = opgt._monotone_fit(range(25), rw, rr, ri)

    assert math.isfinite(monotone_cost)
    assert len(monotone_path) == 25
    assert reversal_cost > monotone_cost


def test_projective_measure_is_normal_sign_invariant() -> None:
    normal = np.asarray([0.0, 0.6, 0.8])
    for value in (normal, -normal):
        nb, nc = float(value[1]), float(value[2])
        observed = (nb * nb + nc * nc, nb * nb - nc * nc, 2.0 * nb * nc)
        if value is normal:
            reference = observed
        else:
            assert observed == pytest.approx(reference, abs=0.0)


def test_checked_aggregate_overflow_fails_closed() -> None:
    cells = np.zeros((opgt.AXIAL_SLABS, opgt.TRANSVERSE_STRIPS), dtype=np.int64)
    cells[0, 0] = opgt.INT64_LIMIT
    cells[1, 0] = 1
    with pytest.raises(opgt.OPGTContractError, match="V34_OPGT_RESOURCE_STOP"):
        opgt._checked_axis_sum(cells, axis=0)


def test_fixed_cache_record_contract() -> None:
    decision = {
        "candidate_sha256": "00" * 32,
        "passthrough_line_sha256": "11" * 32,
        "phase_path_sha256": "22" * 32,
        "candidate_id": 7,
        "frozen_v16_order": 3,
        "reason": "NONMONOTONE_GAUSS_PATH",
        "action": opgt.ACTION_NONE,
        "nms_reason": "",
        "predicate_passes": [True, True, True, True, True, True, False],
        "metrics_i64": list(range(12)),
    }
    cache = opgt.build_fixed_cache([decision], "33" * 32, "44" * 32)
    assert len(cache) == opgt.CACHE_HEADER_BYTES + opgt.CACHE_RECORD_BYTES
    assert opgt.CACHE_RECORD_BYTES <= 256


def test_triangle_canonicalization_ignores_face_order_and_winding() -> None:
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    first = canonical_triangle_mesh({"positions": positions, "triangles": [[0, 1, 2], [1, 3, 2]]}, 1.0)
    second = canonical_triangle_mesh({"positions": positions, "triangles": [[2, 3, 1], [2, 1, 0]]}, 1.0)
    assert first.source_sha256 == second.source_sha256
    assert len(first.triangles) == len(second.triangles) == 2


def test_engine_has_one_ordinary_nms_callsite_and_no_full_emission_table() -> None:
    source_path = Path(opgt.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ordinary_nms"
    ]
    assert len(calls) == 1
    assert "def _emissions(" not in source
    assert "np.asarray(rows" not in source
