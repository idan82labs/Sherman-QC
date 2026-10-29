from backend.f1_dominant_topology_replacement import replace_complete_dominant_topology


def _item(identifier, source, pair=None):
    item = {"bend_id": identifier, "candidate_source": source, "span_endpoints": [[0, 0, 0], [1, 0, 0]]}
    if pair is not None:
        item["canonical_flange_pair"] = list(pair)
    return item


def test_replaces_baseline_only_for_complete_derived_topology():
    selected = [
        _item("OB1", ""),
        _item("W1", "interface_birth_stitch", ("F1", "F2")),
        _item("W2", "interface_birth_stitch", ("F1", "F3")),
        _item("LONG", "transition_boundary_refit"),
        _item("END1", "axial_support_split"),
        _item("END2", "axial_support_split"),
    ]
    result = replace_complete_dominant_topology(selected)
    assert result["applied"] is True
    assert [item["bend_id"] for item in result["selected"]] == ["W1", "W2", "LONG", "END1", "END2"]
    assert result["removed_bend_ids"] == ["OB1"]


def test_fails_closed_for_incomplete_topology():
    selected = [_item("OB1", ""), _item("LONG", "transition_boundary_refit")]
    result = replace_complete_dominant_topology(selected)
    assert result["applied"] is False
    assert len(result["selected"]) == 2
