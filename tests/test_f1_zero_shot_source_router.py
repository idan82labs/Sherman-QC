from backend.f1_zero_shot_source_router import route_f1_zero_shot_sources


def _result(profile: str, *, union: int, aspects=(1.0, 2.0, 2.0), abstained=False):
    return {
        "profile_family": profile,
        "abstained": abstained,
        "candidate_generator_summary": {
            "union_candidate_count": union,
            "aspect_ratios": list(aspects),
        },
    }


def test_prismatic_plane_lattice_routes_to_zero_shot_without_oracles():
    decision = route_f1_zero_shot_sources([{"bend_id": "P"}], [{"bend_id": "Z"}], _result("prismatic_plane_lattice_completion", union=12, abstained=True))
    assert decision["route"] == "zero_shot"
    assert decision["selected"][0]["bend_id"] == "Z"
    assert decision["uses_expected_count"] is False
    assert decision["uses_part_identifier"] is False
    assert decision["uses_manual_truth"] is False


def test_prismatic_lattice_without_count_consensus_fails_closed():
    decision = route_f1_zero_shot_sources(
        [{"bend_id": "P1"}, {"bend_id": "P2"}],
        [{"bend_id": "Z"}],
        _result("prismatic_plane_lattice_completion", union=12, abstained=True),
    )
    assert decision["route"] == "patch_graph"
    assert decision["reason"] == "prismatic_lattice_without_safe_source_consensus"


def test_non_abstained_prismatic_lattice_fails_closed_even_when_counts_match():
    decision = route_f1_zero_shot_sources(
        [{"bend_id": "P"}],
        [{"bend_id": "Z"}],
        _result("prismatic_plane_lattice_completion", union=12, abstained=False),
    )
    assert decision["route"] == "patch_graph"


def test_sparse_formed_transition_routes_to_zero_shot():
    decision = route_f1_zero_shot_sources(
        [{"bend_id": "P"}],
        [{"bend_id": "Z"}],
        _result("interior_formed_transition_completion", union=4, aspects=(1.1, 4.4, 4.0)),
    )
    assert decision["route"] == "zero_shot"
    assert decision["reason"] == "sparse_pool_formed_transition_recovery"


def test_dense_formed_transition_fails_closed_to_patch_graph():
    decision = route_f1_zero_shot_sources(
        [{"bend_id": "P"}],
        [{"bend_id": "Z"}],
        _result("interior_formed_transition_completion", union=16, aspects=(1.0, 2.0, 2.2)),
    )
    assert decision["route"] == "patch_graph"
    assert decision["selected"][0]["bend_id"] == "P"


def test_empty_patch_graph_uses_available_zero_shot_geometry():
    decision = route_f1_zero_shot_sources([], [{"bend_id": "Z"}], _result("unknown", union=1))
    assert decision["route"] == "zero_shot"
    assert decision["reason"] == "patch_graph_empty"


def test_formed_features_are_preserved_but_partitioned_from_conventional_output():
    zero = [
        {"bend_id": "L1", "strip_family": "prismatic_plane_lattice_long"},
        {"bend_id": "T1", "strip_family": "interior_formed_transition_strip"},
    ]
    decision = route_f1_zero_shot_sources(
        [{"bend_id": "P"}],
        zero,
        _result("interior_formed_transition_completion", union=2),
    )
    assert [item["bend_id"] for item in decision["selected"]] == ["L1"]
    assert [item["bend_id"] for item in decision["formed_feature_selected"]] == ["T1"]
