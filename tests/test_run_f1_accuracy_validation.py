import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_f1_accuracy_validation.py"
    spec = importlib.util.spec_from_file_location("run_f1_accuracy_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validator = _load_module()

from patch_graph_latent_decomposition import DecompositionResult, OwnedBendRegion, SurfaceAtomGraph, TypedLabelAssignment


def _owned_region(index: int) -> OwnedBendRegion:
    return OwnedBendRegion(
        bend_id=f"OB{index}",
        bend_class="transition_only",
        incident_flange_ids=("F1", "F2"),
        canonical_bend_key=f"bend::{index}",
        owned_atom_ids=(f"A{index}",),
        anchor=(float(index), 0.0, 0.0),
        axis_direction=(1.0, 0.0, 0.0),
        support_centroid=(float(index), 0.0, 0.0),
        span_endpoints=((float(index), -1.0, 0.0), (float(index), 1.0, 0.0)),
        angle_deg=90.0,
        radius_mm=None,
        visibility_state="internal",
        support_mass=10.0,
        admissible=True,
        admissibility_reasons=tuple(),
        post_bend_class="transition_only",
        debug_confidence=1.0,
    )


def _decomposition_result(region_count: int = 11) -> DecompositionResult:
    regions = tuple(_owned_region(index) for index in range(1, region_count + 1))
    return DecompositionResult(
        scan_path="/tmp/ysherman.ply",
        part_id="ysherman",
        atom_graph=SurfaceAtomGraph(atoms=tuple(), adjacency={}, edge_weights={}, voxel_size_mm=1.0, local_spacing_mm=1.0),
        flange_hypotheses=tuple(),
        bend_hypotheses=tuple(),
        assignment=TypedLabelAssignment(atom_labels={}, label_types={}, energy=0.0),
        owned_bend_regions=regions,
        exact_bend_count=region_count,
        bend_count_range=(region_count, region_count),
        candidate_solution_energies=(1.0,),
        candidate_solution_counts=(region_count,),
        object_level_confidence_summary={},
        repair_diagnostics={
            "scan_family_route": {
                "route_id": "fragmented_repeat90_interface_birth",
                "solver_profile": "interface_birth_enabled",
                "applied": True,
            }
        },
        current_zero_shot_result={"profile_family": "repeated_near90"},
        current_zero_shot_count_range=(10, 12),
        current_zero_shot_exact_count=None,
    )


def test_partial_review_does_not_score_whole_part_total_as_observed_truth():
    entry = {
        "part_key": "partial_demo",
        "scan_path": "/tmp/partial.ply",
        "expectation": {"expected_bend_count": 10},
        "audit_tags": ["partial", "known_total_bends"],
    }
    row = {
        "candidate": {
            "exact_bend_count": 6,
            "bend_count_range": [6, 6],
            "candidate_source": "raw_f1",
            "guard_reason": None,
            "owned_bend_region_count": 6,
        },
        "raw_f1_candidate": {"map_bend_count": 6, "bend_count_range": [6, 6]},
        "scan_family_route": {"route_id": "partial_scan_limited"},
    }

    review = validator._review_payload(
        track="partial",
        entry=entry,
        row=row,
        render_info={"overview_path": "/tmp/render.png"},
        escalation_reasons=validator._partial_escalation_reasons(entry, row),
    )

    assert review["scan_limited"] is True
    assert review["validation_target"] == "visible_observed_bend_count"
    assert review["expected_bend_count_scored"] is None
    assert review["known_total_bend_count_context"] == 10
    assert review["algorithm"]["observed_bend_count_prediction"] == 6
    assert review["manual_review"]["required"] is True
    assert "whole_part_total_available_but_not_scored_as_observed_truth" in review["manual_review"]["reason_codes"]
    assert "deterministic_observed_count_needs_manual_confirmation" in review["manual_review"]["reason_codes"]


def test_partial_review_scores_manual_observed_truth_when_scan_limited():
    entry = {
        "part_key": "partial_manual",
        "scan_path": "/tmp/partial_manual.ply",
        "expectation": {
            "expected_bend_count": 5,
            "label_semantics": "visible_observed_conventional_bends",
            "scan_limited": True,
            "known_whole_part_total_context": 9,
        },
        "audit_tags": ["partial", "observed_visible_bends", "scan_limited"],
    }
    row = {
        "candidate": {
            "exact_bend_count": None,
            "bend_count_range": [7, 22],
            "candidate_source": "raw_f1",
            "guard_reason": None,
            "owned_bend_region_count": 12,
        },
        "raw_f1_candidate": {"map_bend_count": 12, "bend_count_range": [7, 22]},
        "scan_family_route": {"route_id": "partial_scan_limited"},
    }

    review = validator._review_payload(
        track="partial",
        entry=entry,
        row=row,
        render_info={},
        escalation_reasons=validator._partial_escalation_reasons(entry, row),
    )

    assert review["expected_bend_count_scored"] == 5
    assert review["known_total_bend_count_context"] == 9
    assert "observed_count_range_excludes_manual_truth" in review["manual_review"]["reason_codes"]
    assert "whole_part_total_available_but_not_scored_as_observed_truth" not in review["manual_review"]["reason_codes"]


def test_review_payload_reports_feature_count_semantics_for_raised_forms():
    entry = {
        "part_key": "raised_demo",
        "scan_path": "/tmp/raised.ply",
        "expectation": {
            "expected_bend_count": 8,
            "expected_conventional_bend_count": 8,
            "expected_raised_form_feature_count": 2,
            "expected_total_feature_count": 10,
            "label_semantics": "full_scan_conventional_bends",
        },
    }
    row = {
        "candidate": {"exact_bend_count": 10, "bend_count_range": [9, 10]},
        "raw_f1_candidate": {"map_bend_count": 10, "bend_count_range": [9, 10]},
        "scan_family_route": {},
    }

    review = validator._review_payload(
        track="known",
        entry=entry,
        row=row,
        render_info={},
        escalation_reasons=validator._known_escalation_reasons({**row, "expectation": entry["expectation"]}),
    )

    assert review["algorithm"]["accepted_exact_bend_count"] is None
    assert review["algorithm"]["raw_exact_bend_count_field"] == 10
    assert review["feature_count_semantics"]["counted_target"] == "conventional_bends"
    assert review["feature_count_semantics"]["expected_conventional_bend_count"] == 8
    assert review["feature_count_semantics"]["expected_raised_form_feature_count"] == 2
    assert review["feature_count_semantics"]["algorithm_feature_split_status"] == "not_modeled"


def test_known_review_escalates_deterministic_truth_conflict():
    row = {
        "expectation": {"expected_bend_count": 4},
        "candidate": {"exact_bend_count": 6, "bend_count_range": [6, 6]},
    }

    reasons = validator._known_escalation_reasons(row)

    assert "deterministic_count_conflicts_with_known_truth" in reasons


def test_known_review_does_not_call_non_singleton_exact_field_deterministic():
    row = {
        "expectation": {"expected_bend_count": 8},
        "candidate": {"exact_bend_count": 10, "bend_count_range": [9, 10]},
    }

    reasons = validator._known_escalation_reasons(row)

    assert "deterministic_count_conflicts_with_known_truth" not in reasons
    assert "candidate_range_excludes_known_truth" in reasons


def test_known_review_escalates_range_excluding_truth():
    row = {
        "expectation": {"expected_bend_count": 4},
        "candidate": {"exact_bend_count": None, "bend_count_range": [5, 7]},
    }

    reasons = validator._known_escalation_reasons(row)

    assert "candidate_range_excludes_known_truth" in reasons


def test_known_review_flags_non_singleton_range_for_render_review():
    row = {
        "expectation": {"expected_bend_count": 4},
        "candidate": {"exact_bend_count": None, "bend_count_range": [4, 6]},
    }

    reasons = validator._known_escalation_reasons(row)

    assert "non_singleton_known_count_range" in reasons


def test_final_marker_overlay_requires_singleton_range_matching_exact():
    assert validator._has_final_accepted_exact(
        {"accepted_exact_bend_count": 12, "accepted_bend_count_range": [12, 12]}
    )
    assert not validator._has_final_accepted_exact(
        {"accepted_exact_bend_count": 10, "accepted_bend_count_range": [9, 10]}
    )
    assert not validator._has_final_accepted_exact(
        {"accepted_exact_bend_count": None, "accepted_bend_count_range": [9, 10]}
    )


def test_known_track_renders_supplemental_owned_regions_for_direct_promotion_evidence(tmp_path, monkeypatch):
    captured = {}

    def fake_run_patch_graph_latent_decomposition(**kwargs):
        return _decomposition_result(region_count=11)

    def fake_render_decomposition_artifacts(*, result, output_dir):
        captured["render_region_count"] = len(result.owned_bend_regions)
        captured["render_exact"] = result.exact_bend_count
        return {
            "overview_path": str(tmp_path / "overview.png"),
            "scan_context_path": str(tmp_path / "context.png"),
            "overview_view_paths": {},
            "scan_context_view_paths": {},
        }

    monkeypatch.setattr(validator, "run_patch_graph_latent_decomposition", fake_run_patch_graph_latent_decomposition)
    monkeypatch.setattr(validator, "render_decomposition_artifacts", fake_render_decomposition_artifacts)
    monkeypatch.setattr(validator, "_write_review_contact_sheet", lambda **kwargs: str(tmp_path / "contact.png"))

    rows = validator._run_track(
        track="known",
        slice_payload={
            "entries": [
                {
                    "part_key": "ysherman",
                    "scan_path": "/tmp/ysherman.ply",
                    "expectation": {"expected_bend_count": 12},
                }
            ]
        },
        output_dir=tmp_path / "out",
        review_output_dir=tmp_path / "review",
        runtime_config=None,
        route_mode="apply",
        cache_dir=tmp_path / "cache",
        promotion_evidence={
            "ysherman": {
                "family_residual_repair_eligible": True,
                "promoted_exact_count": 12,
                "checks": {
                    "single_uncovered_flange_family_pair": True,
                    "candidate_owns_baseline_residual": True,
                    "raw_competitors_family_covered": True,
                    "baseline_residual_fraction": 1.0,
                },
                "supplemental_owned_bend_region": {
                    "bend_id": "OB12",
                    "bend_class": "transition_only",
                    "incident_flange_ids": ["F3", "F4"],
                    "canonical_bend_key": "supplemental::OB12",
                    "owned_atom_ids": ["A12"],
                    "anchor": [12.0, 0.0, 0.0],
                    "axis_direction": [1.0, 0.0, 0.0],
                    "support_centroid": [12.0, 0.0, 0.0],
                    "span_endpoints": [[12.0, -1.0, 0.0], [12.0, 1.0, 0.0]],
                    "angle_deg": 90.0,
                    "support_mass": 10.0,
                    "admissible": True,
                    "post_bend_class": "transition_only",
                },
            }
        },
        render_all_known=True,
        limit=None,
        part_keys=[],
    )

    assert rows[0]["algorithm"]["accepted_exact_bend_count"] == 12
    assert rows[0]["algorithm"]["accepted_bend_count_range"] == [12, 12]
    assert rows[0]["algorithm"]["candidate_source"] == "family_residual_repair_promoted_f1"
    assert captured["render_region_count"] == 12
    assert captured["render_exact"] == 12
