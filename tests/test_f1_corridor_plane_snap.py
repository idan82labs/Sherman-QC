from backend.f1_corridor_plane_snap import snap_corridor_axes_by_plane_pareto


def _plane(plane_id, normal, d):
    return {"flange_id": plane_id, "normal": normal, "d": d}


def _fixture(source_anchor=(0.0, 0.0, 0.0), owned_anchor=(2.0, 2.0, 0.0)):
    selected = [
        {
            "bend_id": "OB1",
            "canonical_bend_key": "scan_corridor_completion::ZS1",
            "anchor": list(owned_anchor),
            "axis_direction": [0.0, 0.0, 1.0],
            "span_endpoints": [[2.0, 2.0, -5.0], [2.0, 2.0, 5.0]],
        }
    ]
    bends = [
        {
            "bend_id": "CORRIDOR_ZS1",
            "anchor": list(source_anchor),
            "axis_direction": [0.0, 0.0, 1.0],
            "span_mm": 10.0,
            "candidate_incident_flange_pairs": [["A", "B"], ["A", "C"]],
        }
    ]
    flanges = [
        _plane("A", [1.0, 0.0, 0.0], 0.0),
        _plane("B", [0.0, 1.0, 0.0], 0.0),
        _plane("C", [0.0, 1.0, 0.0], 0.0),
    ]
    return selected, bends, flanges


def test_snaps_when_source_is_pareto_better_for_all_plane_pairs():
    result = snap_corridor_axes_by_plane_pareto(*_fixture())
    assert result["selected"][0]["anchor"] == [0.0, 0.0, 0.0]
    assert result["selected"][0]["selection_reason"] == "corridor_source_plane_pareto_snap"
    assert len(result["replacements"]) == 1
    assert result["uses_manual_truth"] is False


def test_does_not_snap_when_one_plane_pair_is_not_improved():
    selected, bends, flanges = _fixture(source_anchor=(0.0, 2.0, 0.0), owned_anchor=(2.0, 0.0, 0.0))
    result = snap_corridor_axes_by_plane_pareto(selected, bends, flanges)
    assert result["selected"][0]["anchor"] == [2.0, 0.0, 0.0]
    assert result["replacements"] == []


def test_unrelated_owned_region_fails_closed():
    selected, bends, flanges = _fixture()
    selected[0]["canonical_bend_key"] = "ordinary"
    result = snap_corridor_axes_by_plane_pareto(selected, bends, flanges)
    assert result["selected"] == selected
    assert result["replacements"] == []
