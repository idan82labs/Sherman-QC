from backend.f1_transition_boundary_refit import promote_supported_transition_boundary


def test_promotes_boundary_with_dominant_planar_neighbor_support():
    atoms = []
    adjacency = {}
    source_ids = []
    dominant_ids = []
    for side, y in enumerate((-5.0, 5.0)):
        for index in range(20):
            identifier = f"S{side}-{index}"
            source_ids.append(identifier)
            atoms.append({"atom_id": identifier, "centroid": [float(index * 10), y, 0.0]})
            neighbors = []
            outside_count = 2 if side == 1 else 1
            for outside_index in range(outside_count):
                outside = f"F{side}-{index}-{outside_index}"
                atoms.append({"atom_id": outside, "centroid": [float(index * 10), y + side + 1, 0.0]})
                neighbors.append(outside)
                if side == 1:
                    dominant_ids.append(outside)
            adjacency[identifier] = neighbors
    hypotheses = [
        {
            "bend_id": "D1", "source_kind": "detector_bend_seed", "bend_class": "transition_only",
            "span_mm": 300.0, "cross_width_mm": 10.0, "anchor": [95.0, 3.0, 0.0], "axis_direction": [1.0, 0.0, 0.0],
            "atom_ids": source_ids, "canonical_flange_pair": ["BASE", "OTHER"],
        },
        {"bend_id": "D2", "source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 100.0},
    ]
    flanges = [
        {"flange_id": "BASE", "atom_ids": dominant_ids},
        {"flange_id": "OTHER", "atom_ids": ["unused"]},
    ]
    result = promote_supported_transition_boundary([], hypotheses, flanges, {"atoms": atoms, "adjacency": adjacency})
    assert len(result["promoted"]) == 1
    assert result["promoted"][0]["boundary_support_counts"] == [0, 36]


def test_fails_closed_without_dominant_span():
    hypotheses = [
        {"source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 250.0},
        {"source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 200.0},
    ]
    result = promote_supported_transition_boundary([], hypotheses, [], {})
    assert result["promoted"] == []
