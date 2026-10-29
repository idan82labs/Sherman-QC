from backend.f1_interface_birth_consensus import promote_repeated_interface_births


def _birth(identifier, y, span, pair):
    return {
        "bend_id": identifier,
        "source_kind": "interface_birth",
        "anchor": [0.0, y, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": span,
        "canonical_flange_pair": list(pair),
    }


def test_promotes_longest_member_of_repeated_distinct_pair_birth_cluster():
    result = promote_repeated_interface_births(
        [],
        [
            _birth("short", 0.0, 18.0, ("F1", "F2")),
            _birth("long", 0.5, 30.0, ("F1", "F3")),
        ],
    )
    assert len(result["promoted"]) == 1
    assert result["promoted"][0]["span_mm"] == 30.0
    assert result["promoted"][0]["candidate_source"] == "interface_birth_consensus"
    assert result["uses_manual_truth"] is False


def test_does_not_promote_same_flange_pair_repetition():
    result = promote_repeated_interface_births(
        [],
        [_birth("one", 0.0, 18.0, ("F1", "F2")), _birth("two", 0.5, 30.0, ("F2", "F1"))],
    )
    assert result["promoted"] == []


def test_does_not_add_consensus_duplicate_of_selected_line():
    selected = [{"bend_id": "existing", "span_endpoints": [[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]]}]
    result = promote_repeated_interface_births(
        selected,
        [_birth("one", 0.0, 18.0, ("F1", "F2")), _birth("two", 0.5, 30.0, ("F1", "F3"))],
    )
    assert result["promoted"] == []
    assert result["selected"] == selected
