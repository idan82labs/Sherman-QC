from backend.f1_axial_support_split import promote_axial_support_lobes


def test_splits_high_confidence_strip_with_central_support_deficit():
    atoms = []
    source_ids = []
    for index in list(range(0, 61, 2)) + list(range(120, 181, 2)):
        identifier = f"A{index}"
        source_ids.append(identifier)
        atoms.append({"atom_id": identifier, "centroid": [float(index - 90), 0.0, 0.0]})
    hypotheses = [
        {"bend_id": "DOM", "source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 420.0},
        {
            "bend_id": "SPLIT", "source_kind": "detector_bend_seed", "bend_class": "transition_only",
            "span_mm": 180.0, "confidence": 0.9, "anchor": [0.0, 0.0, 0.0],
            "axis_direction": [1.0, 0.0, 0.0], "atom_ids": source_ids,
        },
    ]
    result = promote_axial_support_lobes([], hypotheses, {"atoms": atoms})
    assert len(result["promoted"]) == 2
    assert all(item["central_to_edge_density"] < 0.7 for item in result["promoted"])


def test_fails_closed_for_uniform_support():
    atoms = [{"atom_id": f"A{i}", "centroid": [float(i), 0.0, 0.0]} for i in range(181)]
    hypotheses = [
        {"bend_id": "DOM", "source_kind": "detector_bend_seed", "bend_class": "transition_only", "span_mm": 420.0},
        {
            "bend_id": "SPLIT", "source_kind": "detector_bend_seed", "bend_class": "transition_only",
            "span_mm": 180.0, "confidence": 0.9, "anchor": [90.0, 0.0, 0.0],
            "axis_direction": [1.0, 0.0, 0.0], "atom_ids": [f"A{i}" for i in range(181)],
        },
    ]
    result = promote_axial_support_lobes([], hypotheses, {"atoms": atoms})
    assert result["promoted"] == []
