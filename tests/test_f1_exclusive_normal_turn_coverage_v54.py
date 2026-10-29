from pathlib import Path

import numpy as np
import open3d as o3d

from backend.f1_exclusive_normal_turn_coverage_v54 import (
    AXIAL_BIN_COUNT,
    PRECOMMIT_TOKEN,
    CandidateEvidence,
    _canonical_candidate,
    _dynamic_metrics,
    _principal_dyad_vectors,
    compose_v54,
)


def test_analytic_principal_dyad_vector_is_canonical():
    vectors = _principal_dyad_vectors(
        np.asarray([9.0, 0.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 4.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
    )
    assert np.allclose(vectors[0], [1.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [0.0, 1.0, 0.0])


def test_candidate_bytes_canonicalize_direction_and_negative_zero():
    first = {
        "line_start": [5.0, -0.0, 0.0],
        "line_end": [0.0, -0.0, 0.0],
    }
    second = {
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [5.0, 0.0, 0.0],
    }
    first_bytes, first_hash, *_ = _canonical_candidate(first)
    second_bytes, second_hash, *_ = _canonical_candidate(second)
    assert first_bytes == second_bytes
    assert first_hash == second_hash


def _dense_candidate() -> CandidateEvidence:
    bins = np.repeat(np.arange(AXIAL_BIN_COUNT), 4)
    t = (bins + 0.5) * (96.0 / AXIAL_BIN_COUNT)
    h = np.ones(len(bins), dtype=np.float64)
    return CandidateEvidence(
        digest="a" * 64,
        start=np.zeros(3),
        end=np.asarray([96.0, 0.0, 0.0]),
        axis=np.asarray([1.0, 0.0, 0.0]),
        length_mm=96.0,
        support_vertices=np.arange(96),
        edge_ids=np.arange(96),
        h=h,
        w=h.copy(),
        t=t,
        bins=bins,
        bin_h=np.full(AXIAL_BIN_COUNT, 4.0),
        bin_w=np.full(AXIAL_BIN_COUNT, 4.0),
        effective_edge_count=96.0,
        boundary_contamination=0.0,
        coherent_mask=(1 << AXIAL_BIN_COUNT) - 1,
        total_w=96.0,
        static_reason=None,
    )


def test_global_ownership_accepts_novel_continuous_turn_and_suppresses_reuse():
    candidate = _dense_candidate()
    fresh = _dynamic_metrics(candidate, np.zeros(96, dtype=np.float64))
    assert fresh["admissible"] is True
    assert fresh["novel_count"] == 24
    assert fresh["rho"] == 1.0
    assert fresh["span_fraction"] >= 0.75
    consumed = _dynamic_metrics(candidate, np.ones(96, dtype=np.float64))
    assert consumed["admissible"] is False
    assert consumed["reason"] == "NOVEL_BIN_COUNT"


def _grid_mesh(size: int = 10) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(
        [[float(x), float(y), 0.0] for y in range(size) for x in range(size)],
        dtype=np.float64,
    )
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            b = a + 1
            c = a + size
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    return mesh


def test_compose_v54_keeps_exact_v52_prefix_on_zero_turn_mesh(tmp_path: Path):
    scan_path = tmp_path / "grid.ply"
    assert o3d.io.write_triangle_mesh(str(scan_path), _grid_mesh(), write_ascii=False)
    candidate = {
        "candidate_id": "strict_blind_owned::OB1",
        "candidate_source": "strict_blind_owned",
        "line_start": [0.0, 5.0, 0.0],
        "line_end": [9.0, 5.0, 0.0],
        "support_atom_ids": ["A1"],
        "support_atom_ids_truncated": False,
    }
    prefix = {
        "bend_id": "TRV16-test",
        "line_start": [0.0, 5.0, 0.0],
        "line_end": [9.0, 5.0, 0.0],
        "source_hypothesis_id": candidate["candidate_id"],
    }
    v52 = {
        "predictions": [prefix],
        "source_only": True,
        "research_only": True,
    }
    pool = {"candidates": [candidate]}
    research_inputs = {
        "source_point_count": 100,
        "atom_graph": {
            "voxel_size_mm": 1000.0,
            "atoms": [{"atom_id": "A1", "voxel_key": [0, 0, 0]}],
        },
    }
    result = compose_v54(v52, pool, research_inputs, scan_path)
    audit = result["exclusive_normal_turn_coverage_audit"]
    assert result["predictions"] == [prefix]
    assert audit["precommit_token"] == PRECOMMIT_TOKEN
    assert audit["recovered_candidate_count"] == 0
    assert audit["status"] == "COMMITTED"

