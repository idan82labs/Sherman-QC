from backend.f1_interface_birth_stitch import promote_stitched_interface_births


def _birth(identifier, start, end, pair=("F1", "F2"), width=10.0):
    return {
        "bend_id": identifier,
        "source_kind": "interface_birth",
        "canonical_flange_pair": list(pair),
        "span_endpoints": [list(start), list(end)],
        "span_mm": abs(end[0] - start[0]),
        "cross_width_mm": width,
    }


def _dominant_transition_seeds():
    return [
        {"bend_id": "D1", "source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 420.0},
        {"bend_id": "D2", "source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 180.0},
    ]


def test_stitches_adjacent_fragments_from_same_interface():
    result = promote_stitched_interface_births(
        [],
        _dominant_transition_seeds()
        + [_birth("A", (0, 0, 0), (20, 0, 0)), _birth("B", (25, 0, 0), (45, 0, 0))],
    )
    assert len(result["promoted"]) == 1
    assert result["promoted"][0]["span_mm"] == 45.0
    assert result["promoted"][0]["stitch_member_ids"] == ["A", "B"]


def test_rejects_different_interfaces_or_large_gap():
    different = promote_stitched_interface_births(
        [],
        _dominant_transition_seeds()
        + [_birth("A", (0, 0, 0), (20, 0, 0)), _birth("B", (25, 0, 0), (45, 0, 0), pair=("F1", "F3"))],
    )
    far = promote_stitched_interface_births(
        [],
        _dominant_transition_seeds()
        + [_birth("A", (0, 0, 0), (20, 0, 0)), _birth("B", (40, 0, 0), (60, 0, 0))],
    )
    assert different["promoted"] == []
    assert far["promoted"] == []


def test_does_not_duplicate_a_selected_line():
    births = [_birth("A", (0, 0, 0), (20, 0, 0)), _birth("B", (25, 0, 0), (45, 0, 0))]
    selected = [{"bend_id": "OB1", "span_endpoints": [[0, 0, 0], [45, 0, 0]]}]
    result = promote_stitched_interface_births(selected, _dominant_transition_seeds() + births)
    assert result["promoted"] == []
    assert len(result["selected"]) == 1


def test_replaces_materially_shorter_duplicate():
    births = [_birth("A", (0, 0, 0), (20, 0, 0)), _birth("B", (25, 0, 0), (45, 0, 0))]
    selected = [{"bend_id": "OB1", "span_endpoints": [[0, 0, 0], [25, 0, 0]]}]
    result = promote_stitched_interface_births(selected, _dominant_transition_seeds() + births)
    assert len(result["promoted"]) == 1
    assert len(result["selected"]) == 1
    assert result["selected"][0]["replaced_bend_ids"] == ["OB1"]


def test_fails_closed_without_dominant_transition_topology():
    births = [_birth("A", (0, 0, 0), (20, 0, 0)), _birth("B", (25, 0, 0), (45, 0, 0))]
    result = promote_stitched_interface_births([], births)
    assert result["promoted"] == []
