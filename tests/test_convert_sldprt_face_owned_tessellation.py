from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from scripts.convert_sldprt_face_owned_tessellation import (
    extract_face_owned_mesh,
    write_binary_ply,
)
from scripts.run_exp_cad_0015_new_intake_registration import exact_point_to_mesh_distances


def _cadir() -> dict:
    return {
        "units": {"length": "millimeter"},
        "model": {
            "bodies": [{"id": "body:folded"}],
            "faces": [{"id": "face:1"}],
            "tessellations": [
                {
                    "id": "resolved",
                    "body": "body:folded",
                    "faces": ["face:1"],
                    "vertices": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                        {"x": 0, "y": 1, "z": 0},
                    ],
                    "triangles": [[0, 1, 2]],
                },
                {
                    "id": "unresolved-flat-display",
                    "body": None,
                    "faces": [],
                    "vertices": [
                        {"x": 100, "y": 0, "z": 0},
                        {"x": 101, "y": 0, "z": 0},
                        {"x": 100, "y": 1, "z": 0},
                    ],
                    "triangles": [[0, 1, 2]],
                },
            ],
        },
    }


def test_extracts_only_face_owned_geometry_and_records_omission() -> None:
    result = extract_face_owned_mesh(_cadir())
    assert result["mesh"]["vertex_count"] == 3
    assert result["mesh"]["triangle_count"] == 1
    assert result["mesh"]["bbox"]["extent_mm"] == [1.0, 1.0, 0.0]
    assert result["selection"]["selected_tessellation_ids"] == ["resolved"]
    assert result["selection"]["omitted_unresolved_tessellation_ids"] == [
        "unresolved-flat-display"
    ]


def test_unknown_body_fails_closed() -> None:
    cadir = _cadir()
    cadir["model"]["tessellations"][0]["body"] = "body:unknown"
    with pytest.raises(ValueError, match="unknown body"):
        extract_face_owned_mesh(cadir)


def test_binary_ply_is_deterministic_and_open3d_readable(tmp_path: Path) -> None:
    result = extract_face_owned_mesh(_cadir())
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"
    write_binary_ply(first, result["vertices"], result["triangles"])
    write_binary_ply(second, result["vertices"], result["triangles"])
    assert first.read_bytes() == second.read_bytes()
    mesh = o3d.io.read_triangle_mesh(str(first))
    assert len(mesh.vertices) == 3
    assert len(mesh.triangles) == 1


def test_exact_point_to_mesh_distance() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    triangles = np.asarray([[0, 1, 2]], dtype=int)
    points = np.asarray([[0.2, 0.2, 2.0], [0.2, 0.2, 0.0]], dtype=float)
    distances = exact_point_to_mesh_distances(points, vertices, triangles, chunk_size=1)
    np.testing.assert_allclose(distances, [2.0, 0.0], atol=1e-12)
