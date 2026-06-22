from types import SimpleNamespace

from scan_family_router import (
    build_scan_formation_features,
    route_scan_family,
    solver_options_for_route,
)


def _zero_shot_payload(**overrides):
    payload = {
        "profile_family": "conservative_macro",
        "estimated_bend_count": None,
        "estimated_bend_count_range": [8, 10],
        "hypothesis_confidence": 0.6,
        "abstained": False,
        "dominant_angle_families_deg": [90.0],
        "candidate_generator_summary": {
            "point_count": 500000,
            "local_spacing_mm": 1.1,
            "bbox_mm": [100.0, 80.0, 12.0],
            "aspect_ratios": [1.25, 8.33, 6.66],
            "dominant_angle_family_deg": 90.0,
            "dominant_angle_family_mass": 0.86,
            "duplicate_line_family_rate": 0.4,
            "single_scale_count": 8,
            "multiscale_count": 12,
            "union_candidate_count": 20,
            "scan_state_hint": None,
        },
    }
    payload.update(overrides)
    return payload


def _bend(
    bend_id,
    *,
    incident=("F1", "F2"),
    candidate_pairs=(("F1", "F2"),),
):
    return SimpleNamespace(
        bend_id=bend_id,
        incident_flange_ids=tuple(incident),
        seed_incident_flange_ids=tuple(incident),
        canonical_flange_pair=tuple(sorted(incident)),
        candidate_incident_flange_pairs=tuple(tuple(pair) for pair in candidate_pairs),
    )


def test_stable_zero_shot_routes_to_control_observe_only():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(
            profile_family="conservative_macro",
            estimated_bend_count=6,
            estimated_bend_count_range=[6, 6],
            hypothesis_confidence=0.91,
            dominant_angle_families_deg=[45.0],
            candidate_generator_summary={
                **_zero_shot_payload()["candidate_generator_summary"],
                "dominant_angle_family_deg": 45.0,
                "dominant_angle_family_mass": 0.82,
                "duplicate_line_family_rate": 0.05,
            },
        ),
        atom_graph=SimpleNamespace(atoms=tuple(range(50))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(4)),
        bend_hypotheses=tuple(_bend(f"B{i}") for i in range(6)),
    )

    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert route.route_id == "zero_shot_control_stable"
    assert route.solver_profile == "control_exact_tiebreak"
    assert options["enable_control_exact_solution_tiebreak"] is True


def test_near90_exact_anchor_with_low_duplicate_pressure_routes_to_stable_control():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(
            profile_family="conservative_macro",
            estimated_bend_count=6,
            estimated_bend_count_range=[6, 6],
            hypothesis_confidence=0.69,
            candidate_generator_summary={
                **_zero_shot_payload()["candidate_generator_summary"],
                "duplicate_line_family_rate": 0.125,
            },
        ),
        atom_graph=SimpleNamespace(atoms=tuple(range(50))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(18)),
        bend_hypotheses=tuple(
            _bend(f"B{i}", candidate_pairs=(("F1", "F2"), ("F3", "F4")))
            for i in range(6)
        ),
    )

    route = route_scan_family(features)

    assert route.route_id == "zero_shot_control_stable"


def test_near90_exact_anchor_with_narrow_range_routes_to_stable_control_even_with_duplicate_pressure():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(
            profile_family="orthogonal_dense_sheetmetal",
            estimated_bend_count=10,
            estimated_bend_count_range=[9, 10],
            hypothesis_confidence=0.645,
            candidate_generator_summary={
                **_zero_shot_payload()["candidate_generator_summary"],
                "dominant_angle_family_mass": 1.0,
                "duplicate_line_family_rate": 0.48,
            },
        ),
        atom_graph=SimpleNamespace(atoms=tuple(range(120))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(13)),
        bend_hypotheses=tuple(
            _bend(f"B{i}", candidate_pairs=(("F1", "F2"), ("F3", "F4")))
            for i in range(19)
        ),
    )

    route = route_scan_family(features)

    assert route.route_id == "zero_shot_control_stable"


def test_partial_scan_hint_blocks_exact_promotion_route():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(
            profile_family="scan_limited_partial",
            estimated_bend_count=6,
            estimated_bend_count_range=[6, 6],
            candidate_generator_summary={
                **_zero_shot_payload()["candidate_generator_summary"],
                "scan_state_hint": "partial",
            },
        ),
        atom_graph=SimpleNamespace(atoms=tuple(range(50))),
        flange_hypotheses=(),
        bend_hypotheses=(),
    )

    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert route.route_id == "partial_scan_limited"
    assert route.solver_profile == "partial_observed_recovery"
    assert "partial_or_scan_limited" in route.reason_codes
    assert "observed_bend_semantics" in route.enabled_moves
    assert options["enable_interface_birth"] is True
    assert options["partial_observed_semantics"] is True


def test_sparse_partial_seed_pool_routes_to_support_birth_profile():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(
            profile_family="scan_limited_partial",
            estimated_bend_count=None,
            estimated_bend_count_range=[1, 1],
            candidate_generator_summary={
                **_zero_shot_payload()["candidate_generator_summary"],
                "scan_state_hint": "partial",
                "single_scale_count": 1,
                "multiscale_count": 1,
                "union_candidate_count": 2,
                "dominant_angle_family_mass": 0.82,
            },
        ),
        atom_graph=SimpleNamespace(atoms=tuple(range(100))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(7)),
        bend_hypotheses=(_bend("B1"),),
    )

    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert route.route_id == "partial_sparse_seed_support_birth"
    assert route.solver_profile == "partial_sparse_seed_support_birth"
    assert "sparse_seed_candidate_pool" in route.reason_codes
    assert options["enable_interface_subspan_births"] is True
    assert options["allow_no_contact_birth_rescue"] is True
    assert options["partial_sparse_seed_upper_pad"] == 1


def test_repeated_near90_with_duplicate_pressure_routes_to_repeated_family():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(profile_family="repeated_90_sheetmetal"),
        atom_graph=SimpleNamespace(atoms=tuple(range(100))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(10)),
        bend_hypotheses=tuple(_bend(f"B{i}") for i in range(10)),
    )

    route = route_scan_family(features)

    assert route.route_id == "dense_repeated_near90"
    assert route.solver_profile == "full_geometric_attachment"


def test_seed_incidence_risk_routes_to_interface_birth_profile():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(profile_family="repeated_90_sheetmetal"),
        atom_graph=SimpleNamespace(atoms=tuple(range(120))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(18)),
        bend_hypotheses=tuple(
            _bend(
                f"B{i}",
                incident=("F1",) if i % 2 == 0 else ("F1", "F2"),
                candidate_pairs=(("F1", "F2"), ("F3", "F4")),
            )
            for i in range(18)
        ),
    )

    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert route.route_id == "fragmented_repeat90_interface_birth"
    assert route.solver_profile == "interface_birth_enabled"
    assert "interface_activation" in route.enabled_moves
    assert options["enable_interface_subspan_births"] is True
    assert options["limit_birth_rescue_to_control_upper"] is True
    assert options["prioritize_residual_birth_rescue"] is True
    assert options["limit_subspan_births_to_control_low_plus_one"] is False


def test_low_count_fragmented_repeat90_does_not_enable_high_count_residual_subspans():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(
            profile_family="rolled_or_mixed_transition",
            estimated_bend_count_range=[4, 6],
            candidate_generator_summary={
                **_zero_shot_payload()["candidate_generator_summary"],
                "single_scale_count": 5,
                "multiscale_count": 6,
                "union_candidate_count": 11,
                "duplicate_line_family_rate": 0.45,
            },
        ),
        atom_graph=SimpleNamespace(atoms=tuple(range(120))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(18)),
        bend_hypotheses=tuple(
            _bend(
                f"B{i}",
                incident=("F1",) if i % 2 == 0 else ("F1", "F2"),
                candidate_pairs=(("F1", "F2"), ("F3", "F4")),
            )
            for i in range(18)
        ),
    )

    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert route.route_id == "fragmented_repeat90_interface_birth"
    assert options["enable_interface_subspan_births"] is False
    assert options["prioritize_residual_birth_rescue"] is False


def test_small_tight_bend_gap_routes_to_support_recovery():
    payload = _zero_shot_payload(
        profile_family="conservative_macro",
        candidate_generator_summary={
            **_zero_shot_payload()["candidate_generator_summary"],
            "dominant_angle_family_deg": 45.0,
            "dominant_angle_family_mass": 0.5,
            "single_scale_count": 4,
            "multiscale_count": 12,
            "union_candidate_count": 16,
            "duplicate_line_family_rate": 0.1,
        },
    )
    features = build_scan_formation_features(
        current_zero_shot_result=payload,
        atom_graph=SimpleNamespace(atoms=tuple(range(80))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(5)),
        bend_hypotheses=tuple(_bend(f"B{i}") for i in range(7)),
    )

    route = route_scan_family(features)

    assert route.route_id == "small_tight_bend_recovery"
    assert "support_creation" in route.enabled_moves

    options = solver_options_for_route(route, "apply")
    assert options["enable_interface_subspan_births"] is True
    assert options["limit_birth_rescue_to_control_upper"] is True
    assert options["limit_subspan_births_to_control_low_plus_one"] is True


def test_degenerate_flange_orientation_routes_to_control_observe_only():
    payload = _zero_shot_payload(
        profile_family="rolled_or_mixed_transition",
        candidate_generator_summary={
            **_zero_shot_payload()["candidate_generator_summary"],
            "dominant_angle_family_deg": 99.0,
            "dominant_angle_family_mass": 0.55,
            "single_scale_count": 6,
            "multiscale_count": 14,
            "union_candidate_count": 20,
        },
    )
    flanges = tuple(
        SimpleNamespace(flange_id=f"F{i}", normal=(0.1, -0.42, 0.9))
        for i in range(1, 6)
    )
    bends = tuple(
        _bend(f"B{i}", incident=("F1",), candidate_pairs=())
        for i in range(16)
    )

    features = build_scan_formation_features(
        current_zero_shot_result=payload,
        atom_graph=SimpleNamespace(atoms=tuple(range(80))),
        flange_hypotheses=flanges,
        bend_hypotheses=bends,
    )
    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert features.flange_orientation_family_count == 1
    assert route.route_id == "incidence_unavailable_control"
    assert route.solver_profile == "observe_only"
    assert "flange_incidence_unavailable" in route.reason_codes
    assert options["enable_interface_birth"] is False


def test_mixed_transition_routes_to_mixed_recovery_profile():
    payload = _zero_shot_payload(
        profile_family="rolled_or_mixed_transition",
        candidate_generator_summary={
            **_zero_shot_payload()["candidate_generator_summary"],
            "dominant_angle_family_deg": 116.0,
            "dominant_angle_family_mass": 0.55,
            "duplicate_line_family_rate": 0.33,
        },
    )
    features = build_scan_formation_features(
        current_zero_shot_result=payload,
        atom_graph=SimpleNamespace(atoms=tuple(range(80))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(7)),
        bend_hypotheses=tuple(_bend(f"B{i}") for i in range(8)),
    )

    route = route_scan_family(features)
    options = solver_options_for_route(route, "apply")

    assert route.route_id == "ambiguous_mixed_transition"
    assert route.solver_profile == "mixed_transition_recovery"
    assert options["allow_mixed_transition_single_contact"] is True
    assert options["enable_interface_birth"] is True
    assert options["enable_selected_pair_duplicate_prune"] is True
    assert options["selected_pair_duplicate_prune_min_active_bends"] == 0
    assert options["selected_pair_duplicate_prune_alias_atom_max"] == 15
    assert options["enable_final_duplicate_cluster_cleanup"] is True
    assert options["enable_final_duplicate_cluster_scoring"] is True
    assert options["final_duplicate_cluster_alias_atom_max"] == 15


def test_observe_mode_keeps_default_solver_options():
    features = build_scan_formation_features(
        current_zero_shot_result=_zero_shot_payload(profile_family="repeated_90_sheetmetal"),
        atom_graph=SimpleNamespace(atoms=tuple(range(100))),
        flange_hypotheses=tuple(SimpleNamespace(flange_id=f"F{i}") for i in range(10)),
        bend_hypotheses=tuple(_bend(f"B{i}") for i in range(10)),
    )
    route = route_scan_family(features)

    options = solver_options_for_route(route, "observe")

    assert options["solver_profile"] == "default"
    assert options["allow_interface_activation"] is True
