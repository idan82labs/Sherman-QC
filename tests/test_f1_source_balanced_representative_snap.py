import numpy as np

from backend.f1_source_balanced_representative_snap import (
    snap_source_balanced_representatives,
)


def _row(candidate_id, source, y, *, p90=1.0):
    return {
        "candidate_id": candidate_id,
        "representative_source": source,
        "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]],
        "finite_support_fraction": 1.0,
        "finite_support_longest_run_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "p90_line_distance_mm": p90,
        "two_sided_transition_score": 0.9,
    }


def test_dense_source_gets_one_vote_and_count_is_preserved():
    selected = [_row("selected", "plane_lattice", 0.0)]
    rows = [
        _row("plane", "plane_lattice", 0.0),
        *[_row(f"cyl-{index}", "cylindrical_standard", 3.0 + index * 0.1) for index in range(20)],
        _row("mesh", "scan_mesh_adjacency", 3.0),
    ]
    result = snap_source_balanced_representatives(selected, rows)
    assert len(result["selected"]) == 1
    repaired = np.asarray(result["selected"][0]["span_endpoints"])
    np.testing.assert_allclose(repaired[:, 1], 3.0, atol=0.11)
    assert result["changes"][0]["source_family_count"] == 3


def test_fewer_than_three_sources_is_inert():
    selected = [_row("selected", "plane_lattice", 0.0)]
    result = snap_source_balanced_representatives(
        selected,
        [_row("plane", "plane_lattice", 0.0), _row("cyl", "cylindrical_standard", 3.0)],
    )
    assert result == {"selected": selected, "changes": []}


def test_rigid_rotation_preserves_translation():
    selected = [_row("selected", "plane_lattice", 0.0)]
    rows = [
        _row("plane", "plane_lattice", 0.0),
        _row("cyl", "cylindrical_standard", 3.0),
        _row("mesh", "scan_mesh_adjacency", 3.0),
    ]
    baseline = snap_source_balanced_representatives(selected, rows)["selected"][0]
    angle = np.radians(37.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.asarray([11.0, -7.0, 5.0])

    def transform(row):
        result = dict(row)
        line = np.asarray(row["span_endpoints"]) @ rotation.T + translation
        result["span_endpoints"] = line.tolist()
        return result

    rotated = snap_source_balanced_representatives(
        [transform(selected[0])], [transform(row) for row in rows]
    )["selected"][0]
    expected = np.asarray(baseline["span_endpoints"]) @ rotation.T + translation
    np.testing.assert_allclose(rotated["span_endpoints"], expected, atol=1.0e-7)
