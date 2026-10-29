from backend.f1_source_consensus_router import route_scan_only_source_consensus


def _line(line_id, y, *, angle_axis=(1.0, 0.0, 0.0), length=40.0):
    axis = angle_axis
    half = 0.5 * length
    return {
        "candidate_id": line_id,
        "span_endpoints": [
            [-half * axis[0], y - half * axis[1], -half * axis[2]],
            [half * axis[0], y + half * axis[1], half * axis[2]],
        ],
    }


def test_routes_to_combined_only_for_strict_parallel_consensus_subset():
    mesh = [_line("M1", 0.0), _line("M2", 10.0), _line("M3", 20.0)]
    combined = [_line("C1", 0.5), _line("C2", 10.5)]

    result = route_scan_only_source_consensus(mesh, combined)

    assert result["route"] == "combined_consensus_subset"
    assert [item["candidate_id"] for item in result["selected"]] == ["C1", "C2"]
    assert result["uses_expected_count"] is False
    assert result["uses_part_identifier"] is False
    assert result["uses_manual_truth"] is False


def test_keeps_mesh_when_axis_families_are_not_parallel():
    mesh = [
        _line("M1", 0.0),
        _line("M2", 10.0, angle_axis=(0.0, 1.0, 0.0)),
        _line("M3", 20.0),
    ]
    combined = [_line("C1", 0.0), _line("C2", 20.0)]

    result = route_scan_only_source_consensus(mesh, combined)

    assert result["route"] == "mesh_only"
    assert result["reason"] == "mesh_result_has_multiple_axis_families"


def test_keeps_mesh_when_combined_is_not_a_strict_subset():
    mesh = [_line("M1", 0.0), _line("M2", 10.0)]
    combined = [_line("C1", 0.0), _line("C2", 10.0)]

    result = route_scan_only_source_consensus(mesh, combined)

    assert result["route"] == "mesh_only"
    assert result["reason"] == "combined_consensus_not_established"


def test_keeps_mesh_when_combined_adds_an_unmatched_parallel_axis():
    mesh = [_line("M1", 0.0), _line("M2", 10.0), _line("M3", 20.0)]
    combined = [_line("C1", 0.0), _line("C2", 35.0)]

    result = route_scan_only_source_consensus(mesh, combined)

    assert result["route"] == "mesh_only"
    assert result["reason"] == "combined_result_introduces_unsupported_axis"


def test_correspondence_must_be_one_to_one():
    mesh = [_line("M1", 0.0), _line("M2", 20.0), _line("M3", 40.0)]
    combined = [_line("C1", 0.5), _line("C2", 1.0)]

    result = route_scan_only_source_consensus(mesh, combined)

    assert result["route"] == "mesh_only"
    assert result["reason"] == "combined_result_introduces_unsupported_axis"


def test_invalid_geometry_fails_closed_to_mesh():
    mesh = [_line("M1", 0.0), {"candidate_id": "broken"}]
    combined = [_line("C1", 0.0)]

    result = route_scan_only_source_consensus(mesh, combined)

    assert result["route"] == "mesh_only"
    assert result["reason"] == "invalid_line_geometry"
