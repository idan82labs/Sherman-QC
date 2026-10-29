import numpy as np

from scripts.select_f1_mesh_route_v15 import (
    recover_separated_parallel_plane_pairs,
    repair_tilted_adjacency_representatives,
)


def _row(candidate_id: str, planes: tuple[str, str], y: float, span: float = 50.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "representative_source": "plane_lattice",
        "source_kind": "scan_supported_plane_intersection",
        "incident_plane_ids": list(planes),
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": span,
        "span_endpoints": [[0.0, y, 0.0], [span, y, 0.0]],
        "finite_support_fraction": 1.0,
        "finite_support_longest_run_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "angle_deg": 90.0,
        "incident_plane_support_counts": [500, 500],
        "incident_plane_cross_support_widths_mm": [12.0, 12.0],
        "p90_line_distance_mm": 1.0,
    }


def test_recovers_widely_separated_parallel_pair_with_shared_face() -> None:
    rows = [_row("A", ("P1", "P3"), 0.0), _row("B", ("P2", "P3"), 20.0)]
    selected, changes = recover_separated_parallel_plane_pairs(rows, [], 100.0)
    assert selected == [0, 1]
    assert [change["candidate_id"] for change in changes] == ["A", "B"]


def test_rejects_nearby_alias_pair() -> None:
    rows = [_row("A", ("P1", "P3"), 0.0), _row("B", ("P2", "P3"), 14.9)]
    assert recover_separated_parallel_plane_pairs(rows, [], 100.0) == ([], [])


def test_rejects_pair_without_shared_face_or_continuity() -> None:
    rows = [_row("A", ("P1", "P3"), 0.0), _row("B", ("P2", "P4"), 20.0)]
    assert recover_separated_parallel_plane_pairs(rows, [], 100.0) == ([], [])
    rows[1]["incident_plane_ids"] = ["P2", "P3"]
    rows[1]["finite_support_longest_run_fraction"] = 0.90
    assert recover_separated_parallel_plane_pairs(rows, [], 100.0) == ([], [])


def test_does_not_add_pair_member_already_represented() -> None:
    rows = [_row("A", ("P1", "P3"), 0.0), _row("B", ("P2", "P3"), 20.0)]
    selected, changes = recover_separated_parallel_plane_pairs(rows, [0], 100.0)
    assert selected == [0, 1]
    assert [change["candidate_id"] for change in changes] == ["B"]


def test_repairs_tilted_adjacency_with_stronger_continuous_band() -> None:
    adjacency = _row("mesh", ("P1", "P2"), 0.0)
    adjacency.update(
        representative_source="scan_mesh_adjacency",
        axis_direction=[1.0, 0.0, 0.0],
        two_sided_transition_score=0.75,
    )
    band = _row("band", ("P1", "P2"), 2.0)
    angle = np.radians(7.0)
    band.update(
        representative_source="cylindrical_standard",
        cluster_source_count=2,
        axis_direction=[float(np.cos(angle)), float(np.sin(angle)), 0.0],
        span_endpoints=[
            [25.0 - 25.0 * float(np.cos(angle)), 2.0 - 25.0 * float(np.sin(angle)), 0.0],
            [25.0 + 25.0 * float(np.cos(angle)), 2.0 + 25.0 * float(np.sin(angle)), 0.0],
        ],
        two_sided_transition_score=0.90,
        two_sided_normal_angle_deg=84.0,
    )
    selected, changes = repair_tilted_adjacency_representatives([adjacency, band], [0])
    assert selected == [1]
    assert changes[0]["to_candidate_id"] == "band"


def test_does_not_repair_without_transition_margin() -> None:
    adjacency = _row("mesh", ("P1", "P2"), 0.0)
    adjacency.update(representative_source="scan_mesh_adjacency", two_sided_transition_score=0.82)
    band = _row("band", ("P1", "P2"), 2.0)
    angle = np.radians(7.0)
    band.update(
        representative_source="cylindrical_standard",
        cluster_source_count=2,
        axis_direction=[float(np.cos(angle)), float(np.sin(angle)), 0.0],
        span_endpoints=[
            [25.0 - 25.0 * float(np.cos(angle)), 2.0 - 25.0 * float(np.sin(angle)), 0.0],
            [25.0 + 25.0 * float(np.cos(angle)), 2.0 + 25.0 * float(np.sin(angle)), 0.0],
        ],
        two_sided_transition_score=0.90,
    )
    assert repair_tilted_adjacency_representatives([adjacency, band], [0]) == ([0], [])
