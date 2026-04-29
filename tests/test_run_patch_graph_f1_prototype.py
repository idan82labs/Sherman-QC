import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_patch_graph_f1_prototype.py"
    spec = importlib.util.spec_from_file_location("run_patch_graph_f1_prototype", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_module()


def test_compare_only_counts_tightened_when_candidate_contains_truth():
    result = runner._compare([3, 4], [0, 0], 3)
    assert result["tightened"] is False
    assert result["broadened"] is False


def test_compare_marks_tightened_when_narrower_and_truth_is_kept():
    result = runner._compare([10, 12], [11, 12], 12)
    assert result["tightened"] is True


def test_entry_result_payload_allows_missing_render_info():
    class DummyResult:
        bend_count_range = (5, 7)
        exact_bend_count = 6
        owned_bend_regions = [object(), object()]
        candidate_solution_energies = (1.0, 2.0)
        current_zero_shot_count_range = (8, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {"scan_family_route": {"route_id": "ambiguous_mixed_transition"}}

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 6}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["artifacts"] == {}
    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [5, 7]
    assert payload["raw_f1_candidate"]["map_bend_count"] == 6
    assert payload["raw_f1_candidate"]["exact_bend_count"] is None
    assert payload["scan_family_route"]["route_id"] == "ambiguous_mixed_transition"


def test_entry_result_payload_uses_control_for_applied_observe_only_route():
    class DummyResult:
        bend_count_range = (1, 1)
        exact_bend_count = 1
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (6, 6)
        current_zero_shot_exact_count = 6
        current_zero_shot_result = {"profile_family": "conservative_macro"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "zero_shot_control_stable",
                "solver_profile": "observe_only",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 6}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] == 6
    assert payload["candidate"]["bend_count_range"] == [6, 6]
    assert payload["raw_f1_candidate"]["exact_bend_count"] == 1
    assert payload["raw_f1_candidate"]["map_bend_count"] == 1


def test_entry_result_payload_envelopes_control_and_f1_for_partial_observed_route():
    class DummyResult:
        bend_count_range = (4, 4)
        exact_bend_count = 4
        owned_bend_regions = [object(), object(), object(), object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (2, 2)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "scan_limited_partial"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "partial_scan_limited",
                "solver_profile": "partial_observed_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={
            "part_key": "partial_demo",
            "scan_path": "/tmp/partial_demo.ply",
            "expectation": {
                "expected_bend_count": 4,
                "label_semantics": "visible_observed_conventional_bends",
                "scan_limited": True,
            },
        },
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [2, 4]
    assert payload["candidate"]["candidate_source"] == "partial_observed_control_f1_envelope"
    assert payload["candidate"]["guard_reason"] == "scan_limited_blocks_full_count_promotion"
    assert payload["raw_f1_candidate"]["exact_bend_count"] == 4


def test_entry_result_payload_keeps_partial_control_upper_when_raw_f1_overbirths():
    class DummyResult:
        bend_count_range = (10, 10)
        exact_bend_count = 10
        owned_bend_regions = [object()] * 10
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (6, 8)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "scan_limited_partial"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "partial_scan_limited",
                "scan_family": "partial_scan_limited",
                "solver_profile": "partial_observed_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={
            "part_key": "partial_overbirth",
            "scan_path": "/tmp/partial_overbirth.ply",
            "expectation": {"expected_bend_count": 6, "scan_limited": True},
        },
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [6, 8]
    assert payload["candidate"]["candidate_source"] == "partial_observed_control_f1_envelope"


def test_entry_result_payload_tightens_partial_broad_lower_bound_with_raw_f1_support():
    class DummyResult:
        bend_count_range = (4, 4)
        exact_bend_count = 4
        owned_bend_regions = [object()] * 4
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (1, 6)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "scan_limited_partial"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "partial_scan_limited",
                "scan_family": "partial_scan_limited",
                "solver_profile": "partial_observed_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={
            "part_key": "partial_lower",
            "scan_path": "/tmp/partial_lower.ply",
            "expectation": {"expected_bend_count": 5, "scan_limited": True},
        },
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [4, 6]


def test_entry_result_payload_caps_broad_partial_upper_tail_when_raw_f1_is_stable():
    class DummyResult:
        bend_count_range = (5, 5)
        exact_bend_count = 5
        owned_bend_regions = [object()] * 5
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (4, 14)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "scan_limited_partial"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "partial_scan_limited",
                "scan_family": "partial_scan_limited",
                "solver_profile": "partial_observed_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={
            "part_key": "partial_broad_upper",
            "scan_path": "/tmp/partial_broad_upper.ply",
            "expectation": {"expected_bend_count": 8, "scan_limited": True},
        },
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [5, 8]


def test_entry_result_payload_uses_raw_lower_and_control_lower_when_partial_ranges_are_disjoint_low():
    class DummyResult:
        bend_count_range = (3, 5)
        exact_bend_count = 3
        owned_bend_regions = [object()] * 3
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (7, 22)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "scan_limited_partial"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "partial_scan_limited",
                "scan_family": "partial_scan_limited",
                "solver_profile": "partial_observed_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={
            "part_key": "partial_disjoint_low",
            "scan_path": "/tmp/partial_disjoint_low.ply",
            "expectation": {"expected_bend_count": 5, "scan_limited": True},
        },
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [3, 7]


def test_entry_result_payload_pads_sparse_seed_partial_envelope():
    class DummyResult:
        bend_count_range = (5, 6)
        exact_bend_count = 5
        owned_bend_regions = [object()] * 5
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (1, 1)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "scan_limited_partial"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "partial_sparse_seed_support_birth",
                "scan_family": "partial_scan_limited",
                "solver_profile": "partial_sparse_seed_support_birth",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={
            "part_key": "partial_sparse",
            "scan_path": "/tmp/partial_sparse.ply",
            "expectation": {
                "expected_bend_count": 7,
                "label_semantics": "visible_observed_conventional_bends",
                "scan_limited": True,
            },
        },
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [5, 7]
    assert payload["candidate"]["candidate_source"] == "partial_sparse_seed_f1_envelope"
    assert payload["candidate"]["guard_reason"] == "scan_limited_sparse_seed_blocks_exact_promotion"


def test_entry_result_payload_guards_against_f1_outside_narrow_control_range():
    class DummyResult:
        bend_count_range = (7, 8)
        exact_bend_count = 8
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (11, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [11, 12]
    assert payload["candidate"]["candidate_source"] == "control_guard"
    assert payload["candidate"]["guard_reason"] == "f1_range_disjoint_from_control"


def test_entry_result_payload_intersects_overlapping_f1_with_narrow_control_range():
    class DummyResult:
        bend_count_range = (2, 3)
        exact_bend_count = 3
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (3, 4)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "ambiguous_mixed_transition",
                "solver_profile": "mixed_transition_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 3}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] == 3
    assert payload["candidate"]["bend_count_range"] == [3, 3]
    assert payload["candidate"]["candidate_source"] == "control_intersection_f1"
    assert payload["candidate"]["guard_reason"] == "f1_intersection_with_narrow_control"


def test_entry_result_payload_guards_against_f1_disjoint_from_wide_control_range():
    class DummyResult:
        bend_count_range = (3, 5)
        exact_bend_count = 3
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (8, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "small_tight_bend_recovery",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 10}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [8, 12]
    assert payload["candidate"]["guard_reason"] == "f1_range_disjoint_from_control"


def test_entry_result_payload_promotes_f1_when_it_tightens_within_control_range():
    class DummyResult:
        bend_count_range = (12, 12)
        exact_bend_count = 12
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (11, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] == 12
    assert payload["candidate"]["bend_count_range"] == [12, 12]
    assert payload["candidate"]["candidate_source"] == "raw_f1"


def test_entry_result_payload_uses_control_constrained_range_for_singleton_below_control_upper():
    class DummyResult:
        bend_count_range = (11, 11)
        exact_bend_count = 11
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (10, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "repeated_near90"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [11, 12]
    assert payload["candidate"]["candidate_source"] == "control_constrained_f1"
    assert payload["candidate"]["guard_reason"] == "singleton_below_control_upper"


def test_entry_result_payload_promotes_singleton_with_family_residual_repair_evidence():
    class DummyResult:
        bend_count_range = (11, 11)
        exact_bend_count = 11
        owned_bend_regions = [object()] * 11
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (10, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "repeated_near90"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "ysherman", "scan_path": "/tmp/ysherman.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
        promotion_evidence={
            "family_residual_repair_eligible": True,
            "promoted_exact_count": 12,
            "candidate_bend_id": "OB7",
            "checks": {
                "single_uncovered_flange_family_pair": True,
                "candidate_owns_baseline_residual": True,
                "raw_competitors_family_covered": True,
                "baseline_residual_fraction": 1.0,
            },
        },
    )

    assert payload["candidate"]["exact_bend_count"] == 12
    assert payload["candidate"]["bend_count_range"] == [12, 12]
    assert payload["candidate"]["candidate_source"] == "family_residual_repair_promoted_f1"
    assert payload["candidate"]["guard_reason"] == "single_uncovered_flange_family_residual_candidate"


def test_entry_result_payload_rejects_family_residual_repair_when_not_immediate_upper():
    class DummyResult:
        bend_count_range = (11, 11)
        exact_bend_count = 11
        owned_bend_regions = [object()] * 11
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (10, 13)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "repeated_near90"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "ysherman", "scan_path": "/tmp/ysherman.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
        promotion_evidence={
            "family_residual_repair_eligible": True,
            "promoted_exact_count": 12,
            "checks": {
                "single_uncovered_flange_family_pair": True,
                "candidate_owns_baseline_residual": True,
                "raw_competitors_family_covered": True,
                "baseline_residual_fraction": 1.0,
            },
        },
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [10, 13]
    assert payload["candidate"]["candidate_source"] == "control_guard"


def test_entry_result_payload_rejects_family_residual_repair_without_structural_checks():
    class DummyResult:
        bend_count_range = (11, 11)
        exact_bend_count = 11
        owned_bend_regions = [object()] * 11
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (10, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "repeated_near90"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "ysherman", "scan_path": "/tmp/ysherman.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
        promotion_evidence={
            "family_residual_repair_eligible": True,
            "promoted_exact_count": 12,
            "checks": {
                "single_uncovered_flange_family_pair": True,
                "candidate_owns_baseline_residual": False,
                "raw_competitors_family_covered": True,
                "baseline_residual_fraction": 1.0,
            },
        },
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [11, 12]
    assert payload["candidate"]["candidate_source"] == "control_constrained_f1"


def test_entry_result_payload_promotes_singleton_with_count_sweep_evidence():
    class DummyResult:
        bend_count_range = (4, 4)
        exact_bend_count = 4
        owned_bend_regions = [object(), object(), object(), object()]
        candidate_solution_energies = (1.0, 200.0)
        candidate_solution_counts = (4, 4)
        current_zero_shot_count_range = (3, 16)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "small_tight_bend_recovery",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 4}},
        result=DummyResult(),
        render_info=None,
        promotion_evidence={"exact_promotion_eligible": True, "promoted_exact_count": 4},
    )

    assert payload["candidate"]["exact_bend_count"] == 4
    assert payload["candidate"]["bend_count_range"] == [4, 4]
    assert payload["candidate"]["candidate_source"] == "count_sweep_promoted_f1"
    assert payload["candidate"]["guard_reason"] == "count_sweep_upper_tail_rejected"


def test_entry_result_payload_guards_upper_tail_singleton_on_wide_control_without_evidence():
    class DummyResult:
        bend_count_range = (12, 12)
        exact_bend_count = 12
        owned_bend_regions = [object()] * 12
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (3, 16)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "ambiguous_mixed_transition",
                "solver_profile": "mixed_transition_recovery",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "wide_upper_tail", "scan_path": "/tmp/wide.ply", "expectation": {"expected_bend_count": 4}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [3, 16]
    assert payload["candidate"]["candidate_source"] == "control_guard"
    assert payload["candidate"]["guard_reason"] == "upper_tail_singleton_on_wide_control"


def test_entry_result_payload_guards_upper_tail_subset_on_wide_control_without_evidence():
    class DummyResult:
        bend_count_range = (6, 7)
        exact_bend_count = 6
        owned_bend_regions = [object()] * 6
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (3, 13)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "small_tight_bend_recovery",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "wide_subset", "scan_path": "/tmp/wide_subset.ply", "expectation": {"expected_bend_count": 3}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [3, 13]
    assert payload["candidate"]["candidate_source"] == "control_guard"
    assert payload["candidate"]["guard_reason"] == "upper_tail_f1_on_wide_control"


def test_count_sweep_promotion_evidence_accepts_strong_upper_tail_rejection():
    evidence = runner._count_sweep_promotion_evidence(
        {
            "part_key": "demo",
            "variants": [
                {
                    "variant_id": "baseline",
                    "raw_exact_bend_count": 4,
                    "raw_bend_count_range": [4, 4],
                    "best_energy": 1000.0,
                    "promotion_candidate": True,
                },
                {
                    "variant_id": "cap_5",
                    "raw_map_bend_count": 5,
                    "energy_delta_vs_baseline": 125.0,
                },
                {
                    "variant_id": "uncapped",
                    "raw_map_bend_count": 15,
                    "energy_delta_vs_baseline": 650.0,
                },
            ],
        }
    )

    assert evidence["exact_promotion_eligible"] is True
    assert evidence["promoted_exact_count"] == 4


def test_count_sweep_promotion_evidence_rejects_weak_upper_tail_gap():
    evidence = runner._count_sweep_promotion_evidence(
        {
            "part_key": "demo",
            "variants": [
                {
                    "variant_id": "baseline",
                    "raw_exact_bend_count": 4,
                    "raw_bend_count_range": [4, 4],
                    "best_energy": 1000.0,
                    "promotion_candidate": True,
                },
                {
                    "variant_id": "cap_5",
                    "raw_map_bend_count": 5,
                    "energy_delta_vs_baseline": 25.0,
                },
                {
                    "variant_id": "uncapped",
                    "raw_map_bend_count": 15,
                    "energy_delta_vs_baseline": 650.0,
                },
            ],
        }
    )

    assert evidence["exact_promotion_eligible"] is False
    assert "upper_tail_energy_gap_too_small" in evidence["blockers"]


def test_load_promotion_evidence_accepts_single_explicit_payload(tmp_path):
    path = tmp_path / "promotion.json"
    path.write_text(
        """
        {
          "part_key": "ysherman",
          "family_residual_repair_eligible": true,
          "promoted_exact_count": 12
        }
        """,
        encoding="utf-8",
    )

    evidence = runner._load_promotion_evidence([str(path)])

    assert evidence["ysherman"]["family_residual_repair_eligible"] is True
    assert evidence["ysherman"]["promoted_exact_count"] == 12
    assert evidence["ysherman"]["source_path"] == str(path.resolve())


def test_load_promotion_evidence_accepts_evidence_list(tmp_path):
    path = tmp_path / "promotion-list.json"
    path.write_text(
        """
        {
          "evidence": [
            {"part_key": "a", "promoted_exact_count": 4},
            {"part_key": "b", "promoted_exact_count": 6}
          ]
        }
        """,
        encoding="utf-8",
    )

    evidence = runner._load_promotion_evidence([str(path)])

    assert sorted(evidence) == ["a", "b"]
    assert evidence["a"]["promoted_exact_count"] == 4
    assert evidence["b"]["promoted_exact_count"] == 6


def test_entry_result_payload_does_not_promote_exact_for_non_singleton_f1_range():
    class DummyResult:
        bend_count_range = (11, 12)
        exact_bend_count = 11
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (11, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "rolled_or_mixed_transition"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 12}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [11, 12]
    assert payload["candidate"]["candidate_source"] == "raw_f1"


def test_entry_result_payload_guards_small_tight_upper_tail_on_wide_control():
    class DummyResult:
        bend_count_range = (11, 12)
        exact_bend_count = 11
        owned_bend_regions = [object()]
        candidate_solution_energies = (1.0,)
        current_zero_shot_count_range = (8, 12)
        current_zero_shot_exact_count = None
        current_zero_shot_result = {"profile_family": "conservative_macro"}
        repair_diagnostics = {
            "scan_family_route": {
                "route_id": "small_tight_bend_recovery",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        }

    payload = runner._entry_result_payload(
        entry={"part_key": "demo", "scan_path": "/tmp/demo.ply", "expectation": {"expected_bend_count": 9}},
        result=DummyResult(),
        render_info=None,
    )

    assert payload["candidate"]["exact_bend_count"] is None
    assert payload["candidate"]["bend_count_range"] == [8, 12]
    assert payload["candidate"]["candidate_source"] == "control_guard"
    assert payload["candidate"]["guard_reason"] == "upper_tail_f1_on_wide_control"


def test_raw_f1_promotion_diagnostic_marks_stable_small_tight_singleton_for_review():
    diagnostic = runner._raw_f1_promotion_diagnostic(
        route_payload={"route_id": "small_tight_bend_recovery", "applied": True},
        raw_exact=4,
        raw_range=[4, 4],
        control_range=[3, 16],
        solution_energies=[100.0, 180.0, 220.0],
        solution_counts=[4, 4, 4],
        owned_bend_region_count=4,
    )

    assert diagnostic["candidate"] is True
    assert diagnostic["recommendation"] == "manual_review_exact_promotion_candidate"
    assert "large_energy_gap" in diagnostic["reasons"]
    assert diagnostic["promotion_safety"] == "manual_review_with_large_energy_gap"


def test_raw_f1_promotion_diagnostic_marks_low_gap_stable_singleton_for_manual_review_only():
    diagnostic = runner._raw_f1_promotion_diagnostic(
        route_payload={"route_id": "small_tight_bend_recovery", "applied": True},
        raw_exact=8,
        raw_range=[8, 8],
        control_range=[8, 26],
        solution_energies=[2287.6, 2289.7, 2290.4],
        solution_counts=[8, 8, 8],
        owned_bend_region_count=8,
    )

    assert diagnostic["candidate"] is True
    assert diagnostic["recommendation"] == "manual_review_exact_promotion_candidate"
    assert diagnostic["large_energy_gap"] is False
    assert diagnostic["promotion_safety"] == "manual_review_only"
    assert "stable_singleton_manual_review_candidate" in diagnostic["reasons"]
    assert "energy_gap_too_small" in diagnostic["blockers"]


def test_raw_f1_promotion_diagnostic_blocks_wrong_route_even_with_stable_singleton():
    diagnostic = runner._raw_f1_promotion_diagnostic(
        route_payload={"route_id": "fragmented_repeat90_interface_birth", "applied": True},
        raw_exact=11,
        raw_range=[11, 11],
        control_range=[10, 12],
        solution_energies=[100.0, 180.0, 220.0],
        solution_counts=[11, 11, 11],
        owned_bend_region_count=11,
    )

    assert diagnostic["candidate"] is False
    assert "route_not_allowed_for_exact_upper_tail_promotion" in diagnostic["blockers"]


def test_aggregate_handles_empty_results():
    assert runner._aggregate([])["scans_processed"] == 0


def test_aggregate_groups_results_by_route_id():
    aggregate = runner._aggregate(
        [
            {
                "scan_family_route": {"route_id": "dense_repeated_near90"},
                "metrics": {
                    "candidate_exact_match": True,
                    "control_range_contains_truth": True,
                    "candidate_range_contains_truth": True,
                    "range_delta": {"tightened": True, "broadened": False},
                },
            },
            {
                "scan_family_route": {"route_id": "dense_repeated_near90"},
                "metrics": {
                    "candidate_exact_match": False,
                    "control_range_contains_truth": True,
                    "candidate_range_contains_truth": True,
                    "range_delta": {"tightened": False, "broadened": True},
                },
            },
        ]
    )

    route_bucket = aggregate["by_route_id"]["dense_repeated_near90"]
    assert route_bucket["scans_processed"] == 2
    assert route_bucket["candidate_exact_accuracy"] == 0.5
    assert route_bucket["tightened_case_count"] == 1


def test_aggregate_tracks_raw_f1_and_guard_reasons_separately():
    aggregate = runner._aggregate(
        [
            {
                "part_key": "candidate",
                "expectation": {"expected_bend_count": 12},
                "control": {"bend_count_range": [11, 12]},
                "candidate": {
                    "exact_bend_count": None,
                    "bend_count_range": [11, 12],
                    "candidate_source": "control_guard",
                    "guard_reason": "f1_range_disjoint_from_control",
                },
                "raw_f1_candidate": {"map_bend_count": 8, "exact_bend_count": None, "bend_count_range": [7, 8]},
                "raw_f1_promotion_diagnostic": {"candidate": True},
                "scan_family_route": {"route_id": "fragmented_repeat90_interface_birth"},
                "metrics": {
                    "candidate_exact_match": None,
                    "control_range_contains_truth": True,
                    "candidate_range_contains_truth": True,
                    "range_delta": {"tightened": False, "broadened": False},
                },
            },
            {
                "part_key": "guarded",
                "expectation": {"expected_bend_count": 6},
                "control": {"bend_count_range": [6, 6]},
                "candidate": {
                    "exact_bend_count": 6,
                    "bend_count_range": [6, 6],
                    "candidate_source": "control_guard",
                    "guard_reason": "route_observe_only",
                },
                "raw_f1_candidate": {"exact_bend_count": 1, "bend_count_range": [1, 1]},
                "raw_f1_promotion_diagnostic": {"candidate": False},
                "scan_family_route": {"route_id": "zero_shot_control_stable"},
                "metrics": {
                    "candidate_exact_match": True,
                    "control_range_contains_truth": True,
                    "candidate_range_contains_truth": True,
                    "range_delta": {"tightened": False, "broadened": False},
                },
            },
        ]
    )

    assert aggregate["candidate_exact_accuracy"] == 1.0
    assert aggregate["raw_f1_exact_accuracy"] == 0.0
    assert aggregate["render_admissible_exact_accuracy"] is None
    assert aggregate["candidate_range_contains_truth_rate"] == 1.0
    assert aggregate["raw_f1_range_contains_truth_rate"] == 0.0
    assert aggregate["render_admissible_range_contains_truth_rate"] is None
    assert aggregate["candidate_source_counts"] == {"control_guard": 2}
    assert aggregate["guard_reason_counts"]["f1_range_disjoint_from_control"] == 1
    assert aggregate["raw_f1_manual_promotion_candidate_count"] == 1
    assert aggregate["raw_f1_manual_promotion_candidate_parts"] == ["candidate"]
    assert aggregate["by_route_id"]["fragmented_repeat90_interface_birth"]["guarded_case_count"] == 1
    assert aggregate["by_route_id"]["zero_shot_control_stable"]["raw_f1_exact_accuracy"] == 0.0


def test_attach_candidate_render_semantics_rejects_suppressed_marker_count():
    payload = {
        "candidate": {
            "candidate_source": "raw_f1",
            "exact_bend_count": 12,
            "bend_count_range": [12, 12],
        }
    }

    runner._attach_candidate_render_semantics(
        payload,
        render_info={
            "overview_path": "/tmp/raw.png",
            "scan_context_path": "/tmp/context.png",
            "render_suppressed_owned_region_count": 3,
            "render_suppressed_region_ids": ["OB10", "OB11", "OB12"],
        },
        rendered_owned_region_count=9,
    )

    assert payload["candidate"]["accepted_render_semantics"] == "scan_context_only_raw_f1_debug_not_final"
    assert payload["candidate"]["accepted_render_marker_count"] == 0
    assert payload["candidate"]["accepted_render_overview_path"] == "/tmp/context.png"
    assert payload["candidate"]["visual_acceptance_status"] == "blocked"
    assert payload["candidate"]["visual_acceptance_blockers"] == [
        "owned_region_markers_suppressed",
        "render_marker_count_mismatch",
    ]
    assert payload["candidate"]["render_suppressed_region_ids"] == ["OB10", "OB11", "OB12"]
    assert payload["candidate"]["render_admissible_exact_bend_count"] == 9
    assert payload["candidate"]["render_admissible_bend_count_range"] == [9, 9]
    assert payload["candidate"]["render_admissible_delta_from_candidate_exact"] == -3
    assert payload["candidate"]["marker_acceptance_status"] == "blocked"
    assert payload["candidate"]["marker_acceptance_blockers"] == [
        "marker_count_mismatch",
        "owned_region_markers_suppressed",
    ]
    assert "visual_acceptance:owned_region_markers_suppressed" in payload["raw_f1_promotion_diagnostic"]["blockers"]
    assert payload["raw_f1_promotion_diagnostic"]["candidate"] is False


def test_attach_candidate_render_semantics_uses_marker_admissibility_without_render():
    payload = {
        "candidate": {
            "candidate_source": "raw_f1",
            "exact_bend_count": 12,
            "bend_count_range": [12, 12],
        }
    }

    runner._attach_candidate_render_semantics(
        payload,
        render_info=None,
        rendered_owned_region_count=9,
        marker_admissibility={
            "marker_admissible_owned_region_count": 9,
            "marker_suppressed_owned_region_count": 3,
            "marker_suppressed_region_ids": ["OB10", "OB11", "OB12"],
        },
    )

    assert payload["candidate"]["visual_acceptance_status"] == "blocked"
    assert payload["candidate"]["visual_acceptance_blockers"] == [
        "owned_region_markers_suppressed",
        "render_marker_count_mismatch",
        "render_not_generated",
    ]
    assert payload["candidate"]["render_admissible_exact_bend_count"] == 9
    assert payload["candidate"]["render_suppressed_region_ids"] == ["OB10", "OB11", "OB12"]
    assert payload["candidate"]["marker_acceptance_status"] == "blocked"
    assert payload["candidate"]["marker_acceptance_blockers"] == [
        "marker_count_mismatch",
        "owned_region_markers_suppressed",
    ]


def test_attach_candidate_render_semantics_accepts_exact_render_count():
    payload = {
        "candidate": {
            "candidate_source": "raw_f1",
            "exact_bend_count": 12,
            "bend_count_range": [12, 12],
        }
    }

    runner._attach_candidate_render_semantics(
        payload,
        render_info={"overview_path": "/tmp/raw.png", "scan_context_path": "/tmp/context.png"},
        rendered_owned_region_count=12,
    )

    assert payload["candidate"]["accepted_render_semantics"] == "owned_regions_match_accepted_exact"
    assert payload["candidate"]["accepted_render_marker_count"] == 12
    assert payload["candidate"]["accepted_render_overview_path"] == "/tmp/raw.png"
    assert payload["candidate"]["visual_acceptance_status"] == "render_verified"
    assert payload["candidate"]["visual_acceptance_blockers"] == []
    assert payload["candidate"]["render_admissible_exact_bend_count"] == 12
    assert payload["candidate"]["render_admissible_bend_count_range"] == [12, 12]
    assert payload["candidate"]["render_admissible_delta_from_candidate_exact"] == 0
    assert payload["candidate"]["marker_acceptance_status"] == "marker_verified"
    assert payload["candidate"]["marker_acceptance_blockers"] == []
    assert payload["raw_f1_promotion_diagnostic"]["visual_acceptance_status"] == "render_verified"


def test_aggregate_reports_render_admissible_accuracy_separately():
    aggregate = runner._aggregate(
        [
            {
                "part_key": "visual-blocked",
                "expectation": {"expected_bend_count": 12},
                "control": {"bend_count_range": [10, 12]},
                "candidate": {
                    "exact_bend_count": 12,
                    "bend_count_range": [12, 12],
                    "candidate_source": "raw_f1",
                    "render_admissible_exact_bend_count": 9,
                    "render_admissible_bend_count_range": [9, 9],
                    "visual_acceptance_status": "blocked",
                    "marker_acceptance_status": "blocked",
                },
                "raw_f1_candidate": {"exact_bend_count": 12, "bend_count_range": [12, 12]},
                "raw_f1_promotion_diagnostic": {"candidate": False},
                "scan_family_route": {"route_id": "fragmented_repeat90_interface_birth"},
                "metrics": {
                    "candidate_exact_match": True,
                    "control_range_contains_truth": True,
                    "candidate_range_contains_truth": True,
                    "range_delta": {"tightened": True, "broadened": False},
                },
            }
        ]
    )

    assert aggregate["candidate_exact_accuracy"] == 1.0
    assert aggregate["raw_f1_exact_accuracy"] == 1.0
    assert aggregate["render_admissible_exact_accuracy"] == 0.0
    assert aggregate["render_admissible_range_contains_truth_rate"] == 0.0
    assert aggregate["render_blocked_candidate_parts"] == ["visual-blocked"]
    assert aggregate["marker_blocked_candidate_parts"] == ["visual-blocked"]
