from backend.f1_surface_graph_dominance_pruning import prune_surface_graph_dominance


def _edge(name, first, second, score, span=100.0):
    return {
        "candidate_id": name,
        "surface1_id": first,
        "surface2_id": second,
        "two_sided_transition_score": score,
        "span_mm": span,
    }


def test_prunes_weak_short_triangle_shortcut():
    result = prune_surface_graph_dominance(
        [_edge("a-b", "a", "b", 0.9), _edge("a-c", "a", "c", 0.8), _edge("b-c", "b", "c", 0.3, 10.0)]
    )
    assert [item["candidate_id"] for item in result["selected"]] == ["a-b", "a-c"]
    assert result["pruned"][0]["reason"] == "weak_short_triangle_shortcut"


def test_prunes_weak_leaf_extension_from_bounded_degree_joint():
    result = prune_surface_graph_dominance(
        [_edge("a-b", "a", "b", 0.9), _edge("b-c", "b", "c", 0.5)]
    )
    assert [item["candidate_id"] for item in result["selected"]] == ["a-b"]
    assert result["pruned"][0]["reason"] == "weak_leaf_extension"


def test_preserves_weak_edge_at_high_degree_hub():
    candidates = [
        _edge("hub-a", "hub", "a", 0.9),
        _edge("hub-b", "hub", "b", 0.8),
        _edge("hub-c", "hub", "c", 0.7),
        _edge("hub-d", "hub", "d", 0.1),
    ]
    result = prune_surface_graph_dominance(candidates)
    assert result["selected"] == candidates


def test_preserves_high_probability_or_short_leaf_bend():
    high_probability = _edge("b-c", "b", "c", 0.5)
    high_probability["selection_probability"] = 0.9
    short_branch = _edge("b-d", "b", "d", 0.4, span=20.0)
    candidates = [_edge("a-b", "a", "b", 0.9), high_probability, short_branch]

    result = prune_surface_graph_dominance(candidates)

    assert result["selected"] == candidates


def test_prunes_weak_cycle_edge_with_stronger_alternate_path():
    result = prune_surface_graph_dominance(
        [
            _edge("a-b", "a", "b", 0.8),
            _edge("b-c", "b", "c", 0.8),
            _edge("c-d", "c", "d", 0.8),
            _edge("d-a", "d", "a", 0.4, span=40.0),
            _edge("a-x", "a", "x", 0.9),
        ]
    )
    assert [item["candidate_id"] for item in result["selected"]] == ["a-b", "a-x"]
    assert {item["reason"] for item in result["pruned"]} == {
        "weak_cycle_edge",
        "weak_cycle_nonhub_alternate_branch",
    }


def test_preserves_long_structural_cycle_edge():
    candidates = [
        _edge("a-b", "a", "b", 0.8, span=100.0),
        _edge("b-c", "b", "c", 0.8, span=100.0),
        _edge("c-d", "c", "d", 0.8, span=100.0),
        _edge("d-a", "d", "a", 0.4, span=100.0),
    ]

    result = prune_surface_graph_dominance(candidates)

    assert result["selected"] == candidates
