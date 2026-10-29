from pathlib import Path

import numpy as np

from backend.f1_adjacency_lattice_bank_consensus_v53 import (
    PRECOMMIT_TOKEN,
    _certificate,
    _pack,
    compose_v53,
    pair_geometry,
    PairGeometry,
)


def _line(candidate_id: str, source: str, start: float, end: float, y: float = 0.0):
    return {
        "bend_id": candidate_id,
        "candidate_id": candidate_id,
        "candidate_source": source,
        "line_start": [start, y, 0.0],
        "line_end": [end, y, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
    }


def test_pair_geometry_enforces_common_bbox_span():
    mesh = _line("scan_mesh_adjacency::SMA1", "scan_mesh_adjacency", 0.0, 40.0)
    plane = _line("plane_lattice::PLI_RP1_RP2", "plane_lattice", 0.0, 100.0)
    cluster = _line("LO1", "cross_source_line_object", 0.0, 40.0)
    assert pair_geometry(mesh, plane, cluster, 100.0) is None
    mesh["line_end"] = [60.0, 0.0, 0.0]
    geometry = pair_geometry(mesh, plane, cluster, 100.0)
    assert geometry is not None
    assert geometry.common_span_to_bbox_ratio == 0.6


def test_two_bank_certificate_requires_longitudinal_coverage():
    x = np.repeat(np.linspace(0.0, 80.0, 8), 8)
    vertices = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    first = np.zeros(len(vertices), dtype=bool)
    second = np.zeros(len(vertices), dtype=bool)
    first[0::2] = True
    second[1::2] = True
    geometry = PairGeometry((0.0, 80.0), 80.0, 0.8, 1.0, 0.0, 0.0)
    certificate = _certificate(
        (_pack(first), _pack(second)),
        (_pack(first), _pack(second)),
        vertices,
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        geometry,
    )
    assert certificate is not None
    assert certificate["matched_bank_occupied_bin_counts"] == [8, 8]


def test_compose_v53_commits_geometry_abstention_without_reading_mesh():
    mesh = _line("scan_mesh_adjacency::SMA1", "scan_mesh_adjacency", 0.0, 40.0)
    plane = _line("plane_lattice::PLI_RP1_RP2", "plane_lattice", 0.0, 100.0)
    cluster = {
        **_line("LO1", "cross_source_line_object", 0.0, 40.0),
        "cluster_sources": ["cylindrical_wide", "plane_lattice", "scan_mesh_adjacency"],
        "cluster_member_ids": [mesh["candidate_id"], plane["candidate_id"]],
    }
    v52 = {"predictions": [], "source_only": True, "research_only": True}
    pool = {
        "scan_bbox_mm": [100.0, 20.0, 10.0],
        "candidates": [mesh, plane],
        "line_objects": [cluster],
    }
    result = compose_v53(v52, pool, Path("/not/read/because/geometry/abstains.ply"))
    audit = result["adjacency_lattice_bank_consensus_audit"]
    assert audit["precommit_token"] == PRECOMMIT_TOKEN
    assert audit["eligible_pair_count"] == 1
    assert audit["geometrically_eligible_pair_count"] == 0
    assert audit["recovered_candidate_count"] == 0
    assert audit["status"] == "COMMITTED"
