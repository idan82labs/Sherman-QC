"""Tests for shared bend inspection pipeline utilities."""

import json
import sys
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bend_inspection_pipeline as pipeline
from feature_detection.bend_detector import BendInspectionReport, BendSpecification, BendMatch


def test_build_not_detected_match_returns_bendmatch():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    matcher = pipeline.ProgressiveBendMatcher()

    match = pipeline._build_not_detected_match(
        cad_bend=cad_bend,
        matcher=matcher,
        action_item="Need more scan coverage",
    )

    assert isinstance(match, BendMatch)
    assert match.status == "NOT_DETECTED"
    assert match.action_item == "Need more scan coverage"


def test_build_not_detected_match_supports_observability_fields():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    matcher = pipeline.ProgressiveBendMatcher()
    match = pipeline._build_not_detected_match(
        cad_bend=cad_bend,
        matcher=matcher,
        action_item="Observed but still flat",
        physical_completion_state="NOT_FORMED",
        observability_state="UNFORMED",
        observability_detail_state="OBSERVED_NOT_FORMED",
        observability_confidence=0.72,
        visibility_score=0.58,
        visibility_score_source="heuristic_local_visibility_v0",
        surface_visibility_ratio=1.0,
        local_support_score=0.42,
        side_balance_score=0.35,
        assignment_source="CAD_LOCAL_NEIGHBORHOOD",
        assignment_confidence=0.72,
        observed_surface_count=2,
        local_point_count=44,
        local_evidence_score=0.63,
        measurement_mode="cad_local_neighborhood",
        measurement_method="surface_classification",
        measurement_context={"search_radius_mm": 35.0},
    )

    assert match.observability_state == "UNFORMED"
    assert match.observability_detail_state == "OBSERVED_NOT_FORMED"
    assert match.physical_completion_state == "NOT_FORMED"
    assert match.visibility_score == 0.58
    assert match.surface_visibility_ratio == 1.0
    assert match.assignment_source == "CAD_LOCAL_NEIGHBORHOOD"
    assert match.observed_surface_count == 2
    assert match.measurement_mode == "cad_local_neighborhood"


def test_local_measurement_observability_classifies_flat_state():
    measurement = {
        "success": True,
        "measured_angle": 176.0,
        "measurement_confidence": "high",
        "point_count": 80,
        "side1_points": 40,
        "side2_points": 40,
        "cad_alignment1": 0.99,
        "cad_alignment2": 0.98,
    }
    obs = pipeline._local_measurement_observability(measurement, cad_angle=90.0)
    assert obs["state"] == "UNFORMED"
    assert obs["detail_state"] == "OBSERVED_NOT_FORMED"
    assert obs["observed_surface_count"] == 2
    assert 0.0 <= obs["visibility_score"] <= 1.0
    assert obs["surface_visibility_ratio"] == 1.0


def test_local_measurement_observability_requires_trusted_parent_frame_for_partial_section_formed():
    measurement = {
        "success": True,
        "measured_angle": 92.0,
        "aggregate_angle_deg": 92.0,
        "measurement_method": "parent_frame_profile_section",
        "measurement_confidence": "medium",
        "slice_count": 6,
        "planned_slice_count": 7,
        "effective_slice_count": 5.2,
        "angle_dispersion_deg": 1.8,
        "side1_points": 72,
        "side2_points": 70,
        "expected_zone_visible": True,
        "visible_span_ratio": 0.62,
    }

    unknown_obs = pipeline._local_measurement_observability(
        measurement,
        cad_angle=90.0,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "GLOBAL_ICP",
            "alignment_confidence": 0.95,
            "alignment_abstained": False,
        },
    )
    assert unknown_obs["state"] == "UNKNOWN"

    formed_obs = pipeline._local_measurement_observability(
        measurement,
        cad_angle=90.0,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.9,
            "alignment_abstained": False,
        },
    )
    assert formed_obs["state"] == "FORMED"
    assert formed_obs["detail_state"] == "OBSERVED_FORMED"
    assert formed_obs["visibility_score_source"] == "parent_frame_profile_section_v2"


def test_local_measurement_observability_uses_parent_frame_slices_for_unformed_partial():
    measurement = {
        "success": True,
        "measured_angle": 178.5,
        "aggregate_angle_deg": 178.5,
        "measurement_method": "parent_frame_profile_section",
        "measurement_confidence": "medium",
        "slice_count": 6,
        "planned_slice_count": 7,
        "effective_slice_count": 5.0,
        "angle_dispersion_deg": 1.2,
        "side1_points": 64,
        "side2_points": 63,
        "expected_zone_visible": True,
        "visible_span_ratio": 0.71,
    }

    obs = pipeline._local_measurement_observability(
        measurement,
        cad_angle=90.0,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.88,
            "alignment_abstained": False,
        },
    )

    assert obs["state"] == "UNFORMED"
    assert obs["detail_state"] == "OBSERVED_NOT_FORMED"


def test_local_measurement_observability_keeps_partial_section_unknown_when_visible_span_is_weak():
    measurement = {
        "success": True,
        "measured_angle": 92.0,
        "aggregate_angle_deg": 92.0,
        "measurement_method": "parent_frame_profile_section",
        "measurement_confidence": "medium",
        "slice_count": 6,
        "planned_slice_count": 7,
        "effective_slice_count": 5.1,
        "angle_dispersion_deg": 1.5,
        "side1_points": 68,
        "side2_points": 66,
        "expected_zone_visible": False,
        "visible_span_ratio": 0.22,
    }

    obs = pipeline._local_measurement_observability(
        measurement,
        cad_angle=90.0,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.88,
            "alignment_abstained": False,
        },
    )

    assert obs["state"] == "UNKNOWN"
    assert obs["detail_state"] == "PARTIALLY_OBSERVED"


def test_local_measurement_observability_blocks_weak_profile_section_promotion_on_partial_without_parent_frame():
    measurement = {
        "success": True,
        "measured_angle": 11.8,
        "measurement_method": "profile_section",
        "measurement_confidence": "medium",
        "point_count": 0,
        "side1_points": 4279,
        "side2_points": 292,
        "cad_alignment1": 0.997,
        "cad_alignment2": 1.0,
    }

    obs = pipeline._local_measurement_observability(
        measurement,
        cad_angle=12.0,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "GLOBAL_ICP",
            "alignment_confidence": 0.87,
            "alignment_abstained": False,
        },
    )

    assert obs["state"] == "UNKNOWN"
    assert obs["detail_state"] == "PARTIALLY_OBSERVED"


def test_build_local_assignment_candidates_includes_null_and_preferred_measurement():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
        flange1_normal=np.array([0.0, 1.0, 0.0]),
        flange2_normal=np.array([0.0, 0.0, 1.0]),
    )
    measurement = {
        "success": True,
        "measured_angle": 91.0,
        "measurement_confidence": "high",
        "point_count": 64,
        "side1_points": 30,
        "side2_points": 34,
        "cad_alignment1": 0.98,
        "cad_alignment2": 0.97,
        "measurement_method": "probe_region",
    }
    entry = {
        "center": np.array([5.0, 0.0, 0.0]),
        "cad_angle": 90.0,
        "cad_bend_dict": {
            "bend_center": np.array([5.0, 0.0, 0.0]),
            "bend_direction": np.array([1.0, 0.0, 0.0]),
            "surface1_normal": np.array([0.0, 1.0, 0.0]),
            "surface2_normal": np.array([0.0, 0.0, 1.0]),
        },
        "measurement": measurement,
        "observability": pipeline._local_measurement_observability(measurement, cad_angle=90.0),
    }

    candidates, selected = pipeline._build_local_assignment_candidates(
        cad_bend,
        [entry],
        preferred_idx=0,
    )

    assert any(item["candidate_kind"] == "NULL" for item in candidates)
    assert selected["candidate_kind"] == "MEASUREMENT"
    assert selected["measurement_index"] == 0
    assert selected["assignment_score"] > 0.0


def test_build_local_assignment_candidates_prefers_candidate_measured_angle_over_source_cad_angle():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=120.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([20.0, 0.0, 0.0]),
        flange1_normal=np.array([0.0, 1.0, 0.0]),
        flange2_normal=np.array([0.0, 0.0, 1.0]),
    )
    base_entry = {
        "cad_angle": 120.0,
        "cad_bend_dict": {
            "bend_center": np.array([10.0, 0.0, 0.0]),
            "bend_direction": np.array([1.0, 0.0, 0.0]),
            "surface1_normal": np.array([0.0, 1.0, 0.0]),
            "surface2_normal": np.array([0.0, 0.0, 1.0]),
        },
    }
    near_target = {
        **base_entry,
        "center": np.array([10.0, 1.0, 0.0]),
        "measurement": {
            "success": True,
            "measured_angle": 119.2,
            "measurement_confidence": "high",
            "point_count": 48,
            "side1_points": 24,
            "side2_points": 24,
            "cad_alignment1": 0.98,
            "cad_alignment2": 0.98,
            "measurement_method": "surface_classification",
        },
        "observability": {
            "state": "FORMED",
            "visibility_score": 0.9,
            "evidence_score": 0.9,
        },
    }
    farther_angle = {
        **base_entry,
        "center": np.array([10.0, 0.2, 0.0]),
        "measurement": {
            "success": True,
            "measured_angle": 127.9,
            "measurement_confidence": "high",
            "point_count": 48,
            "side1_points": 24,
            "side2_points": 24,
            "cad_alignment1": 0.98,
            "cad_alignment2": 0.98,
            "measurement_method": "surface_classification",
        },
        "observability": {
            "state": "FORMED",
            "visibility_score": 0.9,
            "evidence_score": 0.9,
        },
    }

    candidates, selected = pipeline._build_local_assignment_candidates(
        cad_bend,
        [farther_angle, near_target],
    )

    measured_candidates = [c for c in candidates if c["candidate_kind"] == "MEASUREMENT"]
    assert len(measured_candidates) == 2
    by_index = {c["measurement_index"]: c for c in measured_candidates}
    assert by_index[0]["cad_angle_gap_deg"] > by_index[1]["cad_angle_gap_deg"]
    assert by_index[0]["assignment_score"] < by_index[1]["assignment_score"]
    assert selected["candidate_kind"] == "NULL"


def test_select_local_assignment_candidate_prefers_highest_assignment_score_measurement():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=120.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([20.0, 0.0, 0.0]),
        flange1_normal=np.array([0.0, 1.0, 0.0]),
        flange2_normal=np.array([0.0, 0.0, 1.0]),
    )
    base_entry = {
        "cad_angle": 120.0,
        "cad_bend_dict": {
            "bend_center": np.array([10.0, 0.0, 0.0]),
            "bend_direction": np.array([1.0, 0.0, 0.0]),
            "surface1_normal": np.array([0.0, 1.0, 0.0]),
            "surface2_normal": np.array([0.0, 0.0, 1.0]),
        },
    }
    stronger = {
        **base_entry,
        "center": np.array([10.0, 1.0, 0.0]),
        "measurement": {
            "success": True,
            "measured_angle": 119.2,
            "measurement_confidence": "high",
            "point_count": 48,
            "side1_points": 24,
            "side2_points": 24,
            "cad_alignment1": 0.98,
            "cad_alignment2": 0.98,
            "measurement_method": "surface_classification",
        },
        "observability": {
            "state": "FORMED",
            "visibility_score": 0.9,
            "evidence_score": 0.9,
        },
    }
    weaker = {
        **base_entry,
        "center": np.array([10.0, 0.2, 0.0]),
        "measurement": {
            "success": True,
            "measured_angle": 127.9,
            "measurement_confidence": "high",
            "point_count": 48,
            "side1_points": 24,
            "side2_points": 24,
            "cad_alignment1": 0.98,
            "cad_alignment2": 0.98,
            "measurement_method": "surface_classification",
        },
        "observability": {
            "state": "FORMED",
            "visibility_score": 0.9,
            "evidence_score": 0.9,
        },
    }

    candidates, selected, null_score, best_idx = pipeline._select_local_assignment_candidate(
        cad_bend,
        [weaker, stronger],
    )

    assert any(item["candidate_kind"] == "NULL" for item in candidates)
    assert selected["candidate_kind"] == "MEASUREMENT"
    assert selected["measurement_index"] == 1
    assert best_idx == 1
    assert selected["assignment_score"] > null_score


def test_local_assignment_confidence_promotes_strong_surface_classification_dominance():
    observability = {
        "confidence": 0.5,
        "visibility_score": 0.902,
        "evidence_score": 0.9,
        "observed_surface_count": 2,
    }
    assignment_candidates = [
        {
            "candidate_id": "local_candidate_5",
            "candidate_kind": "MEASUREMENT",
            "assignment_score": 0.8116,
        },
        {
            "candidate_id": "null_candidate",
            "candidate_kind": "NULL",
            "assignment_score": 0.0989,
        },
    ]
    selected_candidate = assignment_candidates[0]

    confidence = pipeline._local_assignment_confidence(
        observability,
        selected_candidate,
        assignment_candidates,
        null_assignment_score=0.0989,
        measurement_success=True,
    )

    assert confidence >= 0.82


def test_local_assignment_confidence_does_not_promote_close_runner_up_case():
    observability = {
        "confidence": 0.9,
        "visibility_score": 0.92,
        "evidence_score": 0.91,
        "observed_surface_count": 2,
    }
    assignment_candidates = [
        {
            "candidate_id": "local_candidate_1",
            "candidate_kind": "MEASUREMENT",
            "assignment_score": 0.79,
        },
        {
            "candidate_id": "local_candidate_2",
            "candidate_kind": "MEASUREMENT",
            "assignment_score": 0.75,
        },
        {
            "candidate_id": "null_candidate",
            "candidate_kind": "NULL",
            "assignment_score": 0.05,
        },
    ]
    selected_candidate = assignment_candidates[0]

    confidence = pipeline._local_assignment_confidence(
        observability,
        selected_candidate,
        assignment_candidates,
        null_assignment_score=0.05,
        measurement_success=True,
    )

    assert confidence == 0.9


def test_resolve_local_assignment_conflicts_prefers_unique_total_score():
    cad_b3 = BendSpecification(
        bend_id="CAD_B3",
        target_angle=90.37,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    cad_b4 = BendSpecification(
        bend_id="CAD_B4",
        target_angle=90.50,
        target_radius=2.0,
        bend_line_start=np.array([20.0, 0.0, 0.0]),
        bend_line_end=np.array([30.0, 0.0, 0.0]),
    )
    plans = [
        {
            "cad_bend": cad_b3,
            "assignment_candidates": [
                {
                    "candidate_id": "local_candidate_0",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 0,
                    "assignment_score": 0.6269,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "local_candidate_1",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 1,
                    "assignment_score": 0.6108,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "null_candidate",
                    "candidate_kind": "NULL",
                    "measurement_index": None,
                    "assignment_score": 0.05,
                    "used_by_other_bend": False,
                },
            ],
            "selected_candidate": {
                "candidate_id": "local_candidate_0",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 0,
                "assignment_score": 0.6269,
                "used_by_other_bend": False,
            },
            "null_assignment_score": 0.05,
            "best_idx": 0,
        },
        {
            "cad_bend": cad_b4,
            "assignment_candidates": [
                {
                    "candidate_id": "local_candidate_0",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 0,
                    "assignment_score": 0.7365,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "null_candidate",
                    "candidate_kind": "NULL",
                    "measurement_index": None,
                    "assignment_score": 0.05,
                    "used_by_other_bend": False,
                },
            ],
            "selected_candidate": {
                "candidate_id": "local_candidate_0",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 0,
                "assignment_score": 0.7365,
                "used_by_other_bend": False,
            },
            "null_assignment_score": 0.05,
            "best_idx": 0,
        },
    ]

    resolved = pipeline._resolve_local_assignment_conflicts(plans)

    assert resolved[0]["best_idx"] == 1
    assert resolved[0]["selected_candidate"]["candidate_id"] == "local_candidate_1"
    assert resolved[1]["best_idx"] == 0
    assert resolved[1]["selected_candidate"]["candidate_id"] == "local_candidate_0"
    candidate0_for_b3 = next(
        item for item in resolved[0]["assignment_candidates"]
        if item["candidate_id"] == "local_candidate_0"
    )
    assert candidate0_for_b3["used_by_other_bend"] is True


def test_resolve_local_assignment_conflicts_prefers_successful_measurement_over_failed_probe():
    cad_b3 = BendSpecification(
        bend_id="CAD_B3",
        target_angle=90.37,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    cad_b4 = BendSpecification(
        bend_id="CAD_B4",
        target_angle=90.50,
        target_radius=2.0,
        bend_line_start=np.array([20.0, 0.0, 0.0]),
        bend_line_end=np.array([30.0, 0.0, 0.0]),
    )
    cad_b5 = BendSpecification(
        bend_id="CAD_B5",
        target_angle=90.03,
        target_radius=2.0,
        bend_line_start=np.array([40.0, 0.0, 0.0]),
        bend_line_end=np.array([50.0, 0.0, 0.0]),
    )
    plans = [
        {
            "cad_bend": cad_b3,
            "assignment_candidates": [
                {
                    "candidate_id": "local_candidate_0",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 0,
                    "assignment_score": 0.6270,
                    "measurement_success": True,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "local_candidate_1",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 1,
                    "assignment_score": 0.6117,
                    "measurement_success": True,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "null_candidate",
                    "candidate_kind": "NULL",
                    "measurement_index": None,
                    "assignment_score": 0.05,
                    "used_by_other_bend": False,
                },
            ],
            "selected_candidate": {
                "candidate_id": "local_candidate_1",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 1,
                "assignment_score": 0.6117,
                "measurement_success": True,
                "used_by_other_bend": False,
            },
            "null_assignment_score": 0.05,
            "best_idx": 1,
        },
        {
            "cad_bend": cad_b4,
            "assignment_candidates": [
                {
                    "candidate_id": "local_candidate_0",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 0,
                    "assignment_score": 0.7376,
                    "measurement_success": True,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "null_candidate",
                    "candidate_kind": "NULL",
                    "measurement_index": None,
                    "assignment_score": 0.05,
                    "used_by_other_bend": False,
                },
            ],
            "selected_candidate": {
                "candidate_id": "local_candidate_0",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 0,
                "assignment_score": 0.7376,
                "measurement_success": True,
                "used_by_other_bend": False,
            },
            "null_assignment_score": 0.05,
            "best_idx": 0,
        },
        {
            "cad_bend": cad_b5,
            "assignment_candidates": [
                {
                    "candidate_id": "local_candidate_1",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 1,
                    "assignment_score": 0.8234,
                    "measurement_success": True,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "local_candidate_2",
                    "candidate_kind": "MEASUREMENT",
                    "measurement_index": 2,
                    "assignment_score": 0.5955,
                    "measurement_success": False,
                    "used_by_other_bend": False,
                },
                {
                    "candidate_id": "null_candidate",
                    "candidate_kind": "NULL",
                    "measurement_index": None,
                    "assignment_score": 0.05,
                    "used_by_other_bend": False,
                },
            ],
            "selected_candidate": {
                "candidate_id": "local_candidate_2",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 2,
                "assignment_score": 0.5955,
                "measurement_success": False,
                "used_by_other_bend": False,
            },
            "null_assignment_score": 0.05,
            "best_idx": 2,
        },
    ]

    resolved = pipeline._resolve_local_assignment_conflicts(plans)

    assert resolved[0]["best_idx"] is None
    assert resolved[0]["selected_candidate"]["candidate_id"] == "null_candidate"
    assert resolved[1]["best_idx"] == 0
    assert resolved[1]["selected_candidate"]["candidate_id"] == "local_candidate_0"
    assert resolved[2]["best_idx"] == 1
    assert resolved[2]["selected_candidate"]["candidate_id"] == "local_candidate_1"
    failed_probe_for_b5 = next(
        item for item in resolved[2]["assignment_candidates"]
        if item["candidate_id"] == "local_candidate_2"
    )
    assert failed_probe_for_b5["used_by_other_bend"] is False


def test_local_assignment_entry_angle_prefers_measured_angle_then_cad_angle():
    assert pipeline._local_assignment_entry_angle(
        {
            "cad_angle": 120.0,
            "measurement": {
                "aggregate_angle_deg": 119.4,
                "measured_angle": 122.0,
            },
        }
    ) == 119.4
    assert pipeline._local_assignment_entry_angle(
        {
            "cad_angle": 120.0,
            "measurement": {
                "measured_angle": 121.2,
            },
        }
    ) == 121.2
    assert pipeline._local_assignment_entry_angle(
        {
            "cad_angle": 120.0,
            "measurement": {},
        }
    ) == 120.0


def test_allow_probe_region_implausible_completion_accepts_strong_partial_probe_support():
    cad_bend = BendSpecification(
        bend_id="CAD_B12",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    allowed = pipeline._allow_probe_region_implausible_completion(
        measurement={
            "success": True,
            "measured_angle": 179.8,
            "measurement_method": "probe_region",
        },
        cad_bend=cad_bend,
        observability={
            "observed_surface_count": 2,
            "surface_visibility_ratio": 1.0,
            "local_support_score": 1.0,
            "side_balance_score": 0.149,
        },
        selected_candidate={"assignment_score": 0.6791},
        null_assignment_score=0.1916,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "GLOBAL_ICP",
            "alignment_confidence": 0.72,
            "alignment_abstained": False,
        },
    )

    assert allowed is True


def test_allow_probe_region_implausible_completion_rejects_unbalanced_probe_support():
    cad_bend = BendSpecification(
        bend_id="CAD_B4",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    allowed = pipeline._allow_probe_region_implausible_completion(
        measurement={
            "success": True,
            "measured_angle": 179.8,
            "measurement_method": "probe_region",
        },
        cad_bend=cad_bend,
        observability={
            "observed_surface_count": 2,
            "surface_visibility_ratio": 1.0,
            "local_support_score": 1.0,
            "side_balance_score": 0.078,
        },
        selected_candidate={"assignment_score": 0.6797},
        null_assignment_score=0.05,
        scan_state="partial",
        alignment_metadata={
            "alignment_source": "GLOBAL_ICP",
            "alignment_confidence": 0.72,
            "alignment_abstained": False,
        },
    )

    assert allowed is False


def test_build_not_detected_match_supports_assignment_candidate_fields():
    cad_bend = BendSpecification(
        bend_id="B1",
        target_angle=90.0,
        target_radius=2.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
    )
    matcher = pipeline.ProgressiveBendMatcher()
    match = pipeline._build_not_detected_match(
        cad_bend=cad_bend,
        matcher=matcher,
        action_item="No local winner",
        assignment_candidate_id="null_candidate",
        assignment_candidate_kind="NULL",
        assignment_candidate_score=0.77,
        assignment_null_score=0.77,
        assignment_candidate_count=3,
        measurement_context={"assignment_candidates": [{"candidate_id": "null_candidate"}]},
    )

    assert match.assignment_candidate_id == "null_candidate"
    assert match.assignment_candidate_kind == "NULL"
    assert match.assignment_candidate_score == 0.77
    assert match.assignment_null_score == 0.77
    assert match.assignment_candidate_count == 3


def test_cad_spec_to_measurement_dict_uses_selected_bend_geometry():
    cad_bend = BendSpecification(
        bend_id="CAD_B4",
        target_angle=120.0,
        target_radius=3.0,
        bend_line_start=np.array([1.0, 2.0, 3.0]),
        bend_line_end=np.array([5.0, 2.0, 3.0]),
        flange1_normal=np.array([0.0, 1.0, 0.0]),
        flange2_normal=np.array([0.0, 0.0, 1.0]),
    )

    payload = pipeline._cad_spec_to_measurement_dict(cad_bend, ordinal=4)

    assert payload["bend_id"] == "CAD_B4"
    assert payload["bend_name"] == "CAD_B4"
    assert payload["bend_angle"] == 120.0
    assert np.allclose(payload["bend_center"], np.array([3.0, 2.0, 3.0]))
    assert np.allclose(payload["bend_direction"], np.array([1.0, 0.0, 0.0]))
    assert payload["surface1_id"] == -9
    assert payload["surface2_id"] == -10


def test_build_selected_measurement_dicts_preserves_real_surface_ids_when_matched():
    cad_bend = BendSpecification(
        bend_id="CAD_B4",
        target_angle=90.0,
        target_radius=2.5,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([20.0, 0.0, 0.0]),
        flange1_normal=np.array([0.0, 1.0, 0.0]),
        flange2_normal=np.array([0.0, 0.0, 1.0]),
    )

    class _DummyExtractedBend:
        bend_center = np.array([10.5, 0.0, 0.0], dtype=np.float64)
        bend_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        bend_angle = 88.0

        def to_dict(self):
            return {
                "bend_id": 1,
                "bend_name": "B1",
                "bend_angle": 88.0,
                "bend_center": [10.5, 0.0, 0.0],
                "bend_direction": [1.0, 0.0, 0.0],
                "bend_line_start": [0.0, 0.0, 0.0],
                "bend_line_end": [20.0, 0.0, 0.0],
                "surface1_id": 17,
                "surface2_id": 18,
                "surface1_normal": [1.0, 0.0, 0.0],
                "surface2_normal": [0.0, 1.0, 0.0],
            }

    payloads, matched_count = pipeline._build_selected_measurement_dicts(
        [cad_bend],
        [_DummyExtractedBend()],
    )

    assert matched_count == 1
    assert len(payloads) == 1
    assert payloads[0]["bend_id"] == "CAD_B4"
    assert payloads[0]["bend_angle"] == 90.0
    assert payloads[0]["surface1_id"] == 17
    assert payloads[0]["surface2_id"] == 18
    assert np.allclose(np.asarray(payloads[0]["bend_center"], dtype=np.float64), np.array([10.0, 0.0, 0.0]))


def test_load_cad_geometry_step_uses_cad_import(monkeypatch, tmp_path):
    step_path = tmp_path / "part.stp"
    step_path.write_text("dummy-step", encoding="utf-8")

    class _DummyMesh:
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

    class _DummyResult:
        success = True
        mesh = _DummyMesh()
        error = None

    seen = {}

    def _fake_import(path: str, deflection: float):
        seen["path"] = path
        seen["deflection"] = deflection
        return _DummyResult()

    monkeypatch.setattr(pipeline, "import_cad_file", _fake_import)

    verts, faces, source = pipeline.load_cad_geometry(str(step_path), cad_import_deflection=0.07)

    assert seen["path"] == str(step_path)
    assert abs(seen["deflection"] - 0.07) < 1e-9
    assert source == "cad_import"
    assert verts.shape == (3, 3)
    assert faces.shape == (1, 3)


def test_load_bend_runtime_config_supports_wrapped_payload(tmp_path):
    config_path = tmp_path / "config.json"
    payload = {
        "name": "best",
        "runtime_config": {
            "scan_detector_kwargs": {"min_plane_points": 33},
            "matcher_kwargs": {"max_angle_diff": 13.0},
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = pipeline.load_bend_runtime_config(str(config_path))

    assert cfg.scan_detector_kwargs["min_plane_points"] == 33
    assert cfg.matcher_kwargs["max_angle_diff"] == 13.0


def test_runtime_config_deepcopy_isolation():
    raw = {
        "scan_detector_kwargs": {"min_plane_points": 12},
        "matcher_kwargs": {"max_angle_diff": 11.0},
    }
    cfg = pipeline.BendInspectionRuntimeConfig.from_dict(raw)
    raw["scan_detector_kwargs"]["min_plane_points"] = 99
    assert cfg.scan_detector_kwargs["min_plane_points"] == 12

    serialized = cfg.to_dict()
    serialized["scan_detector_kwargs"]["min_plane_points"] = 77
    assert cfg.scan_detector_kwargs["min_plane_points"] == 12


def test_apply_scan_runtime_profile_uses_stronger_partial_dense_settings():
    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        {"min_plane_points": 50, "min_plane_area": 30.0},
        {"voxel_size": 1.0, "outlier_nb_neighbors": 20, "outlier_std_ratio": 2.0},
        800_000,
        scan_state="partial",
        fallback=False,
    )

    assert profile == "partial_scan_dense_primary"
    assert ctor["min_plane_points"] >= 70
    assert ctor["min_plane_area"] >= 40.0
    assert call["voxel_size"] >= 2.6
    assert call["multiscale"] is False


def test_apply_scan_runtime_profile_uses_partial_light_profile_for_mid_size_partials():
    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        {"min_plane_points": 50, "min_plane_area": 30.0},
        {"voxel_size": 1.0, "outlier_nb_neighbors": 20, "outlier_std_ratio": 2.0},
        225_000,
        scan_state="partial",
        fallback=False,
    )

    assert profile == "partial_scan_light_primary"
    assert ctor["min_plane_points"] >= 62
    assert ctor["min_plane_area"] >= 35.0
    assert call["voxel_size"] >= 2.25
    assert call["multiscale"] is False


def test_apply_scan_runtime_profile_preserves_medium_partial_fallback_recall():
    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    fallback_ctor, fallback_call = pipeline._fallback_scan_detection_config(runtime_cfg)

    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        fallback_ctor,
        fallback_call,
        368_352,
        scan_state="partial",
        fallback=True,
    )

    assert profile == "partial_scan_medium_fallback"
    assert ctor["min_plane_points"] <= 36
    assert ctor["min_plane_area"] <= 20.0
    assert call["voxel_size"] <= 2.0
    assert call["multiscale"] is False


def test_apply_scan_runtime_profile_preserves_light_partial_fallback_recall_without_multiscale():
    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    fallback_ctor, fallback_call = pipeline._fallback_scan_detection_config(runtime_cfg)

    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        fallback_ctor,
        fallback_call,
        225_000,
        scan_state="partial",
        fallback=True,
    )

    assert profile == "partial_scan_light_fallback"
    assert ctor["min_plane_points"] <= 35
    assert ctor["min_plane_area"] <= 20.0
    assert call["voxel_size"] <= 1.85
    assert call["multiscale"] is False


def test_run_progressive_bend_inspection_surfaces_warnings(monkeypatch):
    dummy_report = BendInspectionReport(part_id="PART", matches=[])

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), None, "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: ([], "detector_point_cloud"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((30, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_scan_bends",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: dummy_report,
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert report is dummy_report
    assert details.cad_source == "mock"
    assert details.cad_bend_count == 0
    assert details.scan_bend_count == 0
    assert details.overdetected_count == 0
    assert "No bends extracted from CAD" in details.warnings
    assert "No bends detected in scan" in details.warnings


def test_run_progressive_bend_inspection_applies_expected_override(monkeypatch, tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            [
                {
                    "part": "PART",
                    "drawing_expected_bends": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BEND_EXPECTED_OVERRIDES", str(override_path))

    cad_bends = [
        BendSpecification(
            bend_id=f"B{i+1}",
            target_angle=90.0,
            target_radius=2.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0.0, 10.0, 0.0]),
        )
        for i in range(6)
    ]
    matches = []
    for i, b in enumerate(cad_bends):
        status = "PASS" if i < 3 else "NOT_DETECTED"
        matches.append(BendMatch(cad_bend=b, detected_bend=None, status=status))
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), None, "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "detector_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((30, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_scan_bends",
        lambda **kwargs: [object()] * 3,
    )
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: report,
    )

    _, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert details.cad_bend_count == 6
    assert details.expected_bend_count == 3
    assert abs(details.expected_progress_pct - 100.0) < 1e-9
    assert details.overdetected_count == 0
    assert any("Expected bend override applied" in w for w in details.warnings)


def test_run_progressive_bend_inspection_caps_expected_progress(monkeypatch, tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps([{"part": "PART", "drawing_expected_bends": 3}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("BEND_EXPECTED_OVERRIDES", str(override_path))

    cad_bends = [
        BendSpecification(
            bend_id=f"B{i+1}",
            target_angle=90.0,
            target_radius=2.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0.0, 10.0, 0.0]),
        )
        for i in range(6)
    ]
    matches = [BendMatch(cad_bend=b, detected_bend=None, status="PASS") for b in cad_bends]
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), None, "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "detector_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((30, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_scan_bends",
        lambda **kwargs: [object()] * 6,
    )
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: report,
    )

    _, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert details.expected_bend_count == 3
    assert details.overdetected_count == 3
    assert abs(details.expected_progress_pct - 100.0) < 1e-9
    assert any("Over-detected bends vs expected baseline" in w for w in details.warnings)


def test_estimate_scan_quality_flags_good_scan():
    x = np.linspace(0.0, 120.0, 80)
    y = np.linspace(0.0, 80.0, 60)
    xx, yy = np.meshgrid(x, y)
    cad = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    rng = np.random.default_rng(42)
    scan = cad + rng.normal(0.0, 0.15, cad.shape)

    quality = pipeline.estimate_scan_quality(
        scan_points=scan,
        cad_points=cad,
        expected_bends=8,
        completed_bends=6,
        passed_bends=6,
    )

    assert quality["status"] in {"GOOD", "FAIR"}
    assert quality["coverage_pct"] > 55.0
    assert quality["density_pts_per_cm2"] > 10.0


def test_estimate_scan_quality_flags_poor_scan():
    x = np.linspace(0.0, 120.0, 80)
    y = np.linspace(0.0, 80.0, 60)
    xx, yy = np.meshgrid(x, y)
    cad = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    # Sparse partial corner capture with low point count and limited coverage.
    sparse_idx = np.where((cad[:, 0] < 30.0) & (cad[:, 1] < 20.0))[0]
    scan = cad[sparse_idx][::8]

    quality = pipeline.estimate_scan_quality(
        scan_points=scan,
        cad_points=cad,
        expected_bends=8,
        completed_bends=1,
        passed_bends=1,
    )

    assert quality["status"] == "POOR"
    assert (
        quality["coverage_pct"] < 45.0
        or quality["density_pts_per_cm2"] < 8.0
    )


def _make_spec(
    bend_id: str,
    *,
    angle: float = 90.0,
    radius: float = 3.0,
    bend_form: str = "FOLDED",
) -> BendSpecification:
    return BendSpecification(
        bend_id=bend_id,
        target_angle=angle,
        target_radius=radius,
        bend_line_start=np.zeros(3),
        bend_line_end=np.array([10.0, 0.0, 0.0]),
        flange1_area=100.0,
        flange2_area=100.0,
        bend_form=bend_form,
    )


def _make_match(
    cad_bend: BendSpecification,
    status: str,
    *,
    angle_deviation: float = 0.0,
    confidence: float = 0.8,
) -> BendMatch:
    return BendMatch(
        cad_bend=cad_bend,
        detected_bend=None,
        status=status,
        angle_deviation=angle_deviation,
        match_confidence=confidence,
    )


def _mark_confident_global_match(match: BendMatch, candidate_id: str) -> BendMatch:
    match.assignment_source = "GLOBAL_DETECTION"
    match.assignment_confidence = max(float(match.match_confidence or 0.0), 0.88)
    match.assignment_candidate_id = candidate_id
    match.assignment_candidate_kind = "GLOBAL_DETECTION"
    match.assignment_candidate_score = max(float(match.assignment_confidence or 0.0), 0.88)
    match.assignment_null_score = 0.12
    match.assignment_candidate_count = 1
    match.measurement_method = "detected_bend_match"
    match.measurement_mode = "global_detection"
    match.physical_completion_state = "FORMED"
    match.observability_state = "FORMED"
    match.measurement_context = {"trusted_alignment_for_release": True}
    return match


def _mark_confident_local_match(match: BendMatch, candidate_id: str) -> BendMatch:
    match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    match.assignment_confidence = max(float(match.match_confidence or 0.0), 0.9)
    match.assignment_candidate_id = candidate_id
    match.assignment_candidate_kind = "MEASUREMENT"
    match.assignment_candidate_score = max(float(match.assignment_confidence or 0.0), 0.845)
    match.assignment_null_score = 0.05
    match.assignment_candidate_count = 1
    match.measurement_method = "surface_classification"
    match.measurement_mode = "cad_local_neighborhood"
    match.physical_completion_state = "FORMED"
    match.observability_state = "FORMED"
    match.measurement_context = {
        "trusted_alignment_for_release": True,
        "assignment_candidates": [
            {
                "candidate_id": candidate_id,
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 0,
                "assignment_score": max(float(match.assignment_candidate_score or 0.0), 0.845),
                "used_by_other_bend": False,
            },
            {
                "candidate_id": "null_candidate",
                "candidate_kind": "NULL",
                "measurement_index": None,
                "assignment_score": 0.05,
                "used_by_other_bend": False,
            },
        ],
    }
    return match


def test_merge_bend_reports_preserves_primary_detection_when_local_misses():
    b1 = _make_spec("B1")
    b2 = _make_spec("B2")
    b3 = _make_spec("B3")
    primary = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(b1, "PASS", angle_deviation=0.4),
            _make_match(b2, "NOT_DETECTED"),
            _make_match(b3, "PASS", angle_deviation=0.6),
        ],
    )
    primary.compute_summary()
    local = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(b1, "FAIL", angle_deviation=8.0),
            _make_match(b2, "FAIL", angle_deviation=6.0),
            _make_match(b3, "NOT_DETECTED"),
        ],
    )
    local.compute_summary()

    merged = pipeline.merge_bend_reports(primary, local, part_id="PART")

    assert merged.detected_count == 3
    assert [m.status for m in merged.matches] == ["PASS", "FAIL", "PASS"]


def test_merge_bend_reports_prefers_authoritative_local_absence_for_weak_global_match():
    b1 = _make_spec("B1")
    primary = BendInspectionReport(
        part_id="PART",
        matches=[_make_match(b1, "PASS", angle_deviation=0.4, confidence=0.78)],
    )
    primary.compute_summary()
    local_match = _make_match(b1, "NOT_DETECTED")
    local_match.observability_state = "UNFORMED"
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.87,
    )
    assert merged.detected_count == 0
    assert merged.matches[0].status == "NOT_DETECTED"

    merged_low_fit = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.55,
    )
    assert merged_low_fit.detected_count == 1
    assert merged_low_fit.matches[0].status == "PASS"


def test_merge_bend_reports_requires_trusted_parent_frame_for_authoritative_local_absence():
    b1 = _make_spec("B1")
    primary = BendInspectionReport(
        part_id="PART",
        matches=[_make_match(b1, "PASS", angle_deviation=0.4, confidence=0.78)],
    )
    primary.compute_summary()
    local_match = _make_match(b1, "NOT_DETECTED")
    local_match.observability_state = "UNFORMED"
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged_global_only = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.87,
        local_alignment_metadata={
            "alignment_source": "GLOBAL_ICP",
            "alignment_confidence": 0.92,
            "alignment_abstained": False,
        },
    )
    assert merged_global_only.detected_count == 1
    assert merged_global_only.matches[0].status == "PASS"

    merged_plane = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.87,
        local_alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.92,
            "alignment_abstained": False,
        },
    )
    assert merged_plane.detected_count == 0
    assert merged_plane.matches[0].status == "NOT_DETECTED"


def test_merge_bend_reports_does_not_treat_unobserved_local_absence_as_authoritative():
    b1 = _make_spec("B1")
    primary = BendInspectionReport(
        part_id="PART",
        matches=[_make_match(b1, "PASS", angle_deviation=0.4, confidence=0.78)],
    )
    primary.compute_summary()
    local_match = _make_match(b1, "NOT_DETECTED")
    local_match.observability_state = "UNKNOWN"
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        local_alignment_fitness=0.87,
    )

    assert merged.detected_count == 1
    assert merged.matches[0].status == "PASS"


def test_merge_bend_reports_prefers_local_hybrid_verification_over_global_detection():
    b1 = _make_spec("B1")
    b1.countable_in_regression = True

    primary_match = _make_match(b1, "FAIL", angle_deviation=13.5, confidence=0.92)
    primary_match.assignment_source = "GLOBAL_DETECTION"
    primary_match.measurement_mode = "global_detection"
    primary = BendInspectionReport(part_id="PART", matches=[primary_match])
    primary.compute_summary()

    local_match = _make_match(b1, "NOT_DETECTED", confidence=0.2)
    local_match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match.measurement_mode = "cad_local_neighborhood"
    local_match.observability_state = "UNKNOWN"
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        feature_policy={"countable_source": "legacy_folded_hybrid"},
    )

    assert merged.detected_count == 0
    assert merged.matches[0].status == "NOT_DETECTED"


def test_report_short_obtuse_geometry_penalty_prefers_plausible_line_geometry():
    b1 = _make_spec("B6", angle=120.0)
    b1.bend_line_end = np.array([53.0, 0.0, 0.0])

    bad_match = _make_match(b1, "FAIL", angle_deviation=7.88, confidence=0.75)
    bad_match.line_length_deviation_mm = 79.951
    bad_report = BendInspectionReport(part_id="PART", matches=[bad_match])
    bad_report.compute_summary()

    good_match = _make_match(b1, "FAIL", angle_deviation=8.58, confidence=0.9)
    good_match.line_length_deviation_mm = 0.0
    good_report = BendInspectionReport(part_id="PART", matches=[good_match])
    good_report.compute_summary()

    assert pipeline._report_short_obtuse_geometry_penalty(good_report) < pipeline._report_short_obtuse_geometry_penalty(bad_report)


def test_merge_bend_reports_prefers_local_short_obtuse_geometry_over_bad_global_line_match():
    b1 = _make_spec("B6", angle=120.0)
    b1.bend_line_end = np.array([53.0, 0.0, 0.0])

    primary_match = _make_match(b1, "FAIL", angle_deviation=7.88, confidence=0.75)
    primary_match.assignment_source = "GLOBAL_DETECTION"
    primary_match.measurement_mode = "global_detection"
    primary_match.feature_family = "COUNTABLE_DISCRETE_BEND"
    primary_match.line_length_deviation_mm = 79.951

    local_match = _make_match(b1, "FAIL", angle_deviation=8.58, confidence=0.9)
    local_match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match.measurement_mode = "cad_local_neighborhood"
    local_match.measurement_method = "surface_classification"
    local_match.feature_family = "COUNTABLE_DISCRETE_BEND"
    local_match.line_length_deviation_mm = 0.0

    primary = BendInspectionReport(part_id="PART", matches=[primary_match])
    primary.compute_summary()
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged = pipeline.merge_bend_reports(primary, local, part_id="PART")

    assert merged.matches[0].assignment_source == "CAD_LOCAL_NEIGHBORHOOD"
    assert merged.matches[0].line_length_deviation_mm == 0.0


def test_merge_bend_reports_preserves_decision_ready_primary_over_ambiguous_local_pass():
    b1 = _make_spec("B1", angle=90.0)

    primary_match = _make_match(b1, "WARNING", angle_deviation=0.2, confidence=0.88)
    primary_match.assignment_source = "GLOBAL_DETECTION"
    primary_match.assignment_confidence = 0.88
    primary_match.assignment_candidate_id = "B1"
    primary_match.assignment_candidate_kind = "GLOBAL_DETECTION"
    primary_match.assignment_candidate_score = 0.88
    primary_match.assignment_null_score = 0.12
    primary_match.assignment_candidate_count = 1
    primary_match.measurement_method = "detected_bend_match"
    primary_match.physical_completion_state = "FORMED"
    primary_match.observability_state = "FORMED"
    primary_match.measurement_context = {"trusted_alignment_for_release": True}

    local_match = _make_match(b1, "PASS", angle_deviation=0.1, confidence=0.9)
    local_match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match.assignment_confidence = 0.9
    local_match.assignment_candidate_id = "local_candidate_0"
    local_match.assignment_candidate_kind = "MEASUREMENT"
    local_match.assignment_candidate_score = 0.845
    local_match.assignment_null_score = 0.05
    local_match.assignment_candidate_count = 3
    local_match.measurement_method = "surface_classification"
    local_match.physical_completion_state = "FORMED"
    local_match.observability_state = "FORMED"
    local_match.measurement_context = {
        "trusted_alignment_for_release": True,
        "assignment_candidates": [
            {
                "candidate_id": "local_candidate_0",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 0,
                "assignment_score": 0.845,
                "used_by_other_bend": False,
            },
            {
                "candidate_id": "local_candidate_1",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 1,
                "assignment_score": 0.821,
                "used_by_other_bend": False,
            },
            {
                "candidate_id": "null_candidate",
                "candidate_kind": "NULL",
                "measurement_index": None,
                "assignment_score": 0.05,
                "used_by_other_bend": False,
            },
        ],
    }

    primary = BendInspectionReport(part_id="PART", matches=[primary_match])
    primary.compute_summary()
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged = pipeline.merge_bend_reports(primary, local, part_id="PART")

    assert merged.matches[0].assignment_source == "GLOBAL_DETECTION"
    assert merged.matches[0].measurement_method == "detected_bend_match"
    assert merged.matches[0].status == "WARNING"
    assert merged.matches[0].correspondence_state() == "CONFIDENT"


def test_merge_bend_reports_hybrid_policy_does_not_override_with_ambiguous_local_pass():
    b1 = _make_spec("B1", angle=90.0)
    b1.countable_in_regression = True

    primary_match = _make_match(b1, "WARNING", angle_deviation=0.2, confidence=0.88)
    primary_match.assignment_source = "GLOBAL_DETECTION"
    primary_match.assignment_confidence = 0.88
    primary_match.assignment_candidate_id = "B1"
    primary_match.assignment_candidate_kind = "GLOBAL_DETECTION"
    primary_match.assignment_candidate_score = 0.88
    primary_match.assignment_null_score = 0.12
    primary_match.assignment_candidate_count = 1
    primary_match.measurement_method = "detected_bend_match"
    primary_match.physical_completion_state = "FORMED"
    primary_match.observability_state = "FORMED"
    primary_match.measurement_mode = "global_detection"
    primary_match.measurement_context = {"trusted_alignment_for_release": True}

    local_match = _make_match(b1, "PASS", angle_deviation=0.1, confidence=0.9)
    local_match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match.assignment_confidence = 0.9
    local_match.assignment_candidate_id = "local_candidate_0"
    local_match.assignment_candidate_kind = "MEASUREMENT"
    local_match.assignment_candidate_score = 0.845
    local_match.assignment_null_score = 0.05
    local_match.assignment_candidate_count = 3
    local_match.measurement_method = "surface_classification"
    local_match.measurement_mode = "cad_local_neighborhood"
    local_match.physical_completion_state = "FORMED"
    local_match.observability_state = "FORMED"
    local_match.measurement_context = {
        "trusted_alignment_for_release": True,
        "assignment_candidates": [
            {
                "candidate_id": "local_candidate_0",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 0,
                "assignment_score": 0.845,
                "used_by_other_bend": False,
            },
            {
                "candidate_id": "local_candidate_1",
                "candidate_kind": "MEASUREMENT",
                "measurement_index": 1,
                "assignment_score": 0.821,
                "used_by_other_bend": False,
            },
            {
                "candidate_id": "null_candidate",
                "candidate_kind": "NULL",
                "measurement_index": None,
                "assignment_score": 0.05,
                "used_by_other_bend": False,
            },
        ],
    }

    primary = BendInspectionReport(part_id="PART", matches=[primary_match])
    primary.compute_summary()
    local = BendInspectionReport(part_id="PART", matches=[local_match])
    local.compute_summary()

    merged = pipeline.merge_bend_reports(
        primary,
        local,
        part_id="PART",
        feature_policy={"countable_source": "legacy_folded_hybrid"},
    )

    assert merged.matches[0].assignment_source == "GLOBAL_DETECTION"
    assert merged.matches[0].measurement_method == "detected_bend_match"
    assert merged.matches[0].correspondence_state() == "CONFIDENT"


def test_report_decision_grade_rank_ignores_ambiguous_local_status_gains():
    b1 = _make_spec("B1", angle=90.0)
    b2 = _make_spec("B2", angle=90.0)

    primary_match_1 = _make_match(b1, "WARNING", angle_deviation=0.2, confidence=0.88)
    primary_match_1.assignment_source = "GLOBAL_DETECTION"
    primary_match_1.assignment_confidence = 0.88
    primary_match_1.assignment_candidate_id = "B1"
    primary_match_1.assignment_candidate_kind = "GLOBAL_DETECTION"
    primary_match_1.assignment_candidate_score = 0.88
    primary_match_1.assignment_null_score = 0.12
    primary_match_1.assignment_candidate_count = 1
    primary_match_1.physical_completion_state = "FORMED"
    primary_match_1.observability_state = "FORMED"

    primary_match_2 = _make_match(b2, "FAIL", angle_deviation=1.8, confidence=0.84)
    primary_match_2.assignment_source = "GLOBAL_DETECTION"
    primary_match_2.assignment_confidence = 0.84
    primary_match_2.assignment_candidate_id = "B2"
    primary_match_2.assignment_candidate_kind = "GLOBAL_DETECTION"
    primary_match_2.assignment_candidate_score = 0.84
    primary_match_2.assignment_null_score = 0.16
    primary_match_2.assignment_candidate_count = 1
    primary_match_2.physical_completion_state = "FORMED"
    primary_match_2.observability_state = "FORMED"

    ambiguous_candidates = [
        {
            "candidate_id": "local_candidate_0",
            "candidate_kind": "MEASUREMENT",
            "measurement_index": 0,
            "assignment_score": 0.845,
            "used_by_other_bend": False,
        },
        {
            "candidate_id": "local_candidate_1",
            "candidate_kind": "MEASUREMENT",
            "measurement_index": 1,
            "assignment_score": 0.821,
            "used_by_other_bend": False,
        },
        {
            "candidate_id": "null_candidate",
            "candidate_kind": "NULL",
            "measurement_index": None,
            "assignment_score": 0.05,
            "used_by_other_bend": False,
        },
    ]

    local_match_1 = _make_match(b1, "PASS", angle_deviation=0.1, confidence=0.9)
    local_match_1.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match_1.assignment_confidence = 0.9
    local_match_1.assignment_candidate_id = "local_candidate_0"
    local_match_1.assignment_candidate_kind = "MEASUREMENT"
    local_match_1.assignment_candidate_score = 0.845
    local_match_1.assignment_null_score = 0.05
    local_match_1.assignment_candidate_count = 3
    local_match_1.physical_completion_state = "FORMED"
    local_match_1.observability_state = "FORMED"
    local_match_1.measurement_context = {"assignment_candidates": ambiguous_candidates}

    local_match_2 = _make_match(b2, "FAIL", angle_deviation=1.2, confidence=0.9)
    local_match_2.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match_2.assignment_confidence = 0.9
    local_match_2.assignment_candidate_id = "local_candidate_0"
    local_match_2.assignment_candidate_kind = "MEASUREMENT"
    local_match_2.assignment_candidate_score = 0.845
    local_match_2.assignment_null_score = 0.05
    local_match_2.assignment_candidate_count = 3
    local_match_2.physical_completion_state = "FORMED"
    local_match_2.observability_state = "FORMED"
    local_match_2.measurement_context = {"assignment_candidates": ambiguous_candidates}

    primary = BendInspectionReport(part_id="PART", matches=[primary_match_1, primary_match_2])
    primary.compute_summary()
    local = BendInspectionReport(part_id="PART", matches=[local_match_1, local_match_2])
    local.compute_summary()

    assert pipeline._report_rank(local, 2) > pipeline._report_rank(primary, 2)
    assert pipeline._report_decision_grade_rank(local, 2) < pipeline._report_decision_grade_rank(primary, 2)
    assert pipeline._should_promote_dense_local_report(primary, local, 2) is False


def test_feature_policy_hybrid_parts_always_require_local_refinement():
    countable = _make_spec("B1")
    rolled = _make_spec("R1", bend_form="ROLLED")
    rolled.countable_in_regression = False
    report = BendInspectionReport(
        part_id="PART",
        matches=[_make_match(countable, "PASS"), _make_match(rolled, "PASS")],
    )
    report.compute_summary()

    assert pipeline._feature_policy_requires_local_refinement(
        cad_bends=[rolled, countable],
        report=report,
        feature_policy={"countable_source": "legacy_folded_hybrid"},
    ) is True


def test_align_engine_with_retries_keeps_first_good_alignment():
    class _Aligned:
        def __init__(self, pts):
            self.points = pts

    class _Engine:
        def __init__(self):
            self.calls = []
            self.aligned_scan = _Aligned(np.zeros((2, 3), dtype=np.float64))

        def align(self, auto_scale=True, tolerance=1.0, random_seed=None):
            self.calls.append(random_seed)
            self.aligned_scan = _Aligned(np.full((2, 3), float(random_seed), dtype=np.float64))
            return 0.91, 0.25

    phase_events = []
    best_points, fitness, rmse, seed, warnings_local = pipeline._align_engine_with_retries(
        engine=_Engine(),
        part_id="PART",
        tolerance_angle=1.0,
        refinement_cfg={
            "deterministic_seed": 17,
            "retry_seeds": [29, 43],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        },
        phase_cb=phase_events.append,
    )

    assert best_points is not None
    assert fitness == 0.91
    assert rmse == 0.25
    assert seed == 17
    assert warnings_local == []
    assert any("align attempt begin seed=17" in msg for msg in phase_events)


def test_align_engine_with_retries_selects_best_retry_and_discards_untrusted_fit():
    class _Aligned:
        def __init__(self, pts):
            self.points = pts

    class _Engine:
        def __init__(self):
            self.calls = []
            self.aligned_scan = _Aligned(np.zeros((2, 3), dtype=np.float64))
            self.results = {
                17: (0.30, 1.40),
                29: (0.88, 0.95),
                43: (0.51, 1.10),
            }

        def align(self, auto_scale=True, tolerance=1.0, random_seed=None):
            self.calls.append(random_seed)
            self.aligned_scan = _Aligned(np.full((2, 3), float(random_seed), dtype=np.float64))
            return self.results[int(random_seed)]

    engine = _Engine()
    best_points, fitness, rmse, seed, warnings_local = pipeline._align_engine_with_retries(
        engine=engine,
        part_id="PART",
        tolerance_angle=1.0,
        refinement_cfg={
            "deterministic_seed": 17,
            "retry_seeds": [29, 43],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        },
        phase_cb=lambda _: None,
    )

    assert engine.calls == [17, 29, 43]
    assert best_points is not None
    assert np.all(best_points == 29.0)
    assert fitness == 0.88
    assert rmse == 0.95
    assert seed == 29
    assert any("retried alternate deterministic alignment seeds" in w for w in warnings_local)

    low_engine = _Engine()
    low_engine.results = {
        17: (0.30, 1.40),
        29: (0.45, 1.25),
        43: (0.51, 1.10),
    }
    best_points, fitness, rmse, seed, warnings_local = pipeline._align_engine_with_retries(
        engine=low_engine,
        part_id="PART",
        tolerance_angle=1.0,
        refinement_cfg={
            "deterministic_seed": 17,
            "retry_seeds": [29, 43],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        },
        phase_cb=lambda _: None,
    )

    assert best_points is None
    assert fitness == 0.51
    assert seed == 43
    assert any("discarded: alignment fitness" in w for w in warnings_local)


def test_choose_mesh_cad_baseline_prefers_expected_hint():
    detector_specs = [_make_spec(f"D{i}", angle=90.0, radius=12.0, bend_form="ROLLED") for i in range(6)]
    legacy_specs = [_make_spec(f"L{i}") for i in range(10)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=10,
    )

    assert selected == legacy_specs
    assert strategy == "legacy_mesh_expected_hint"


def test_choose_mesh_cad_baseline_prefers_legacy_on_tray_like_rolled_undercount():
    detector_specs = [_make_spec(f"D{i}", angle=90.0, radius=12.0, bend_form="ROLLED") for i in range(6)]
    legacy_specs = [_make_spec(f"L{i}", angle=89.0) for i in range(11)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
    )

    assert selected == legacy_specs
    assert strategy == "legacy_mesh_tray_override"


def test_choose_mesh_cad_baseline_keeps_detector_without_clear_legacy_advantage():
    detector_specs = [_make_spec(f"D{i}", angle=60.0, radius=3.0, bend_form="FOLDED") for i in range(6)]
    legacy_specs = [_make_spec(f"L{i}", angle=45.0) for i in range(7)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
    )

    assert selected == detector_specs
    assert strategy == "detector_mesh"


def test_recover_legacy_mesh_cad_specs_for_expected_only_promotes_material_improvement(monkeypatch):
    baseline_specs = [_make_spec(f"B{i}") for i in range(9)]
    improved_specs = [_make_spec(f"R{i}") for i in range(16)]
    weak_specs = [_make_spec(f"W{i}") for i in range(11)]
    calls = []

    def _fake_build(*, extractor_overrides=None, **kwargs):
        calls.append(extractor_overrides)
        if extractor_overrides == {"min_area_pct": 0.15, "max_bend_gap": 20.0, "dedup_distance": 6.0}:
            return improved_specs
        return weak_specs

    monkeypatch.setattr(pipeline, "_build_legacy_mesh_cad_specs", _fake_build)

    recovered, strategy = pipeline._recover_legacy_mesh_cad_specs_for_expected(
        cad_vertices=np.zeros((10, 3)),
        cad_triangles=np.zeros((3, 3), dtype=np.int32),
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        expected_bend_count_hint=16,
        baseline_specs=baseline_specs,
    )

    assert recovered == improved_specs
    assert strategy == "legacy_mesh_expected_recovery"
    assert calls


def test_extract_cad_bend_specs_with_strategy_uses_expected_recovery(monkeypatch):
    detector_specs = [_make_spec(f"D{i}") for i in range(4)]
    legacy_specs = [_make_spec(f"L{i}") for i in range(9)]
    recovered_specs = [_make_spec(f"R{i}") for i in range(16)]

    monkeypatch.setattr(pipeline, "_build_detector_cad_specs", lambda **kwargs: detector_specs)
    monkeypatch.setattr(pipeline, "_build_legacy_mesh_cad_specs", lambda **kwargs: legacy_specs)
    monkeypatch.setattr(
        pipeline,
        "_recover_legacy_mesh_cad_specs_for_expected",
        lambda **kwargs: (recovered_specs, "legacy_mesh_expected_recovery"),
    )

    selected, strategy = pipeline.extract_cad_bend_specs_with_strategy(
        cad_vertices=np.zeros((10, 3)),
        cad_triangles=np.zeros((3, 3), dtype=np.int32),
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        expected_bend_count_hint=16,
    )

    assert selected == recovered_specs
    assert strategy == "legacy_mesh_expected_recovery"


def test_run_progressive_bend_inspection_passes_expected_hint_to_strategy(monkeypatch, tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps([{"part": "PART", "drawing_expected_bends": 10}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("BEND_EXPECTED_OVERRIDES", str(override_path))

    seen = {}
    cad_bends = [_make_spec("B1")]
    dummy_report = BendInspectionReport(
        part_id="PART",
        matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
    )
    dummy_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )

    def _fake_extract(**kwargs):
        seen["expected_hint"] = kwargs.get("expected_bend_count_hint")
        return cad_bends, "legacy_mesh_expected_hint"

    monkeypatch.setattr(pipeline, "extract_cad_bend_specs_with_strategy", _fake_extract)
    monkeypatch.setattr(pipeline, "load_scan_points", lambda scan_path: np.zeros((30, 3)))
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (dummy_report, 0, False),
    )

    _, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert seen["expected_hint"] == 10
    assert details.cad_strategy == "legacy_mesh_expected_hint"


def test_apply_scan_runtime_profile_adapts_large_scans():
    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        {"min_plane_points": 50, "min_plane_area": 30.0},
        {"preprocess": True},
        715_498,
        fallback=False,
    )

    assert profile == "full_scan_dense_primary"
    assert ctor["min_plane_points"] >= 55
    assert call["voxel_size"] >= 2.0
    assert call["multiscale"] is False


def test_apply_scan_runtime_profile_skips_mesh_vertex_scans():
    ctor, call, profile = pipeline._apply_scan_runtime_profile(
        {"min_plane_points": 50, "min_plane_area": 30.0},
        {"preprocess": True},
        715_498,
        fallback=False,
        scan_geometry_kind="mesh_vertices",
    )

    assert profile is None
    assert ctor == {"min_plane_points": 50, "min_plane_area": 30.0}
    assert call == {"preprocess": True}


def test_detect_and_match_with_fallback_uses_adaptive_profile(monkeypatch):
    seen_calls = []
    cad_bends = [_make_spec("B1")]

    def _fake_detect_scan_bends(**kwargs):
        seen_calls.append(kwargs["scan_detect_call_kwargs"])
        return []

    monkeypatch.setattr(pipeline, "detect_scan_bends", _fake_detect_scan_bends)
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: BendInspectionReport(
            part_id="PART",
            matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
        ),
    )

    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    report, selected_count, fallback_used = pipeline.detect_and_match_with_fallback(
        scan_points=np.zeros((500_000, 3)),
        cad_bends=cad_bends,
        part_id="PART",
        runtime_config=runtime_cfg,
        expected_bend_count=10,
    )

    assert report.part_id == "PART"
    assert selected_count == 0
    assert len(seen_calls) == 2
    assert seen_calls[0]["voxel_size"] >= 2.0
    assert seen_calls[0]["multiscale"] is False
    assert seen_calls[1]["voxel_size"] >= 1.75
    assert seen_calls[1]["multiscale"] is False


def test_detect_and_match_with_fallback_skips_adaptive_profile_for_mesh_vertices(monkeypatch):
    seen_calls = []
    cad_bends = [_make_spec("B1")]

    def _fake_detect_scan_bends(**kwargs):
        seen_calls.append(dict(kwargs["scan_detect_call_kwargs"]))
        return []

    monkeypatch.setattr(pipeline, "detect_scan_bends", _fake_detect_scan_bends)
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: BendInspectionReport(
            part_id="PART",
            matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
        ),
    )

    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    report, selected_count, fallback_used = pipeline.detect_and_match_with_fallback(
        scan_points=np.zeros((500_000, 3)),
        cad_bends=cad_bends,
        part_id="PART",
        runtime_config=runtime_cfg,
        expected_bend_count=10,
        scan_geometry_kind="mesh_vertices",
    )

    assert report.part_id == "PART"
    assert selected_count == 0
    assert fallback_used is False
    assert len(seen_calls) == 2
    assert "voxel_size" not in seen_calls[0]
    assert seen_calls[0]["preprocess"] is True
    assert seen_calls[1]["voxel_size"] == 0.75
    assert seen_calls[1]["multiscale"] is True


def test_detect_and_match_with_fallback_skips_dense_global_fallback(monkeypatch):
    seen_calls = []
    cad_bends = [_make_spec("B1")]

    def _fake_detect_scan_bends(**kwargs):
        seen_calls.append(kwargs["scan_detect_call_kwargs"])
        return []

    monkeypatch.setattr(pipeline, "detect_scan_bends", _fake_detect_scan_bends)
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: BendInspectionReport(
            part_id="PART",
            matches=[BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED")],
        ),
    )

    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    _, _, fallback_used = pipeline.detect_and_match_with_fallback(
        scan_points=np.zeros((715_498, 3)),
        cad_bends=cad_bends,
        part_id="PART",
        runtime_config=runtime_cfg,
        expected_bend_count=10,
    )

    assert fallback_used is False
    assert len(seen_calls) == 1
    assert seen_calls[0]["voxel_size"] >= 2.0


def test_detect_and_match_with_fallback_skips_zero_detection_partial_fallback(monkeypatch):
    seen_calls = []
    cad_bends = [_make_spec("B1"), _make_spec("B2")]

    def _fake_detect_scan_bends(**kwargs):
        seen_calls.append(kwargs["scan_detect_call_kwargs"])
        return []

    monkeypatch.setattr(pipeline, "detect_scan_bends", _fake_detect_scan_bends)
    monkeypatch.setattr(
        pipeline,
        "match_detected_bends",
        lambda **kwargs: BendInspectionReport(
            part_id="PART",
            matches=[
                BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="NOT_DETECTED"),
                BendMatch(cad_bend=cad_bends[1], detected_bend=None, status="NOT_DETECTED"),
            ],
        ),
    )

    runtime_cfg = pipeline.BendInspectionRuntimeConfig()
    _, _, fallback_used = pipeline.detect_and_match_with_fallback(
        scan_points=np.zeros((224_829, 3)),
        cad_bends=cad_bends,
        part_id="PART",
        runtime_config=runtime_cfg,
        expected_bend_count=7,
        scan_state="partial",
    )

    assert fallback_used is False
    assert len(seen_calls) == 1


def test_run_progressive_bend_inspection_uses_dense_local_refinement(monkeypatch):
    cad_bends = [_make_spec("B1"), _make_spec("B2")]
    weak_report = BendInspectionReport(
        part_id="PART",
        matches=[
            _mark_confident_global_match(BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="PASS"), "B1"),
            BendMatch(cad_bend=cad_bends[1], detected_bend=None, status="NOT_DETECTED"),
        ],
    )
    weak_report.compute_summary()
    refined_report = BendInspectionReport(
        part_id="PART",
        matches=[
            _mark_confident_local_match(BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="PASS"), "local_candidate_b1"),
            _mark_confident_local_match(BendMatch(cad_bend=cad_bends[1], detected_bend=None, status="PASS"), "local_candidate_b2"),
        ],
    )
    refined_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "legacy_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((600_000, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (weak_report, 1, False),
    )
    monkeypatch.setattr(
        pipeline,
        "refine_dense_scan_bends_via_alignment",
        lambda **kwargs: (refined_report, ["Dense-scan local refinement used CAD-guided ICP alignment (voxel=2.00mm, fitness=99.00%, rmse=0.100mm)"]),
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert report.detected_count == 2
    assert details.scan_bend_count == 2
    assert any("Dense-scan local refinement used CAD-guided ICP alignment" in w for w in details.warnings)
    assert any("Dense-scan local refinement promoted over primary global detection" in w for w in details.warnings)


def test_run_progressive_bend_inspection_accepts_tied_merge_when_short_obtuse_geometry_improves(monkeypatch):
    cad_bends = [_make_spec("B6", angle=120.0)]
    cad_bends[0].bend_line_end = np.array([53.0, 0.0, 0.0])

    primary_match = _make_match(cad_bends[0], "FAIL", angle_deviation=7.88, confidence=0.75)
    primary_match.assignment_source = "GLOBAL_DETECTION"
    primary_match.measurement_mode = "global_detection"
    primary_match.feature_family = "COUNTABLE_DISCRETE_BEND"
    primary_match.line_length_deviation_mm = 79.951
    primary_report = BendInspectionReport(part_id="PART", matches=[primary_match])
    primary_report.compute_summary()

    local_match = _make_match(cad_bends[0], "FAIL", angle_deviation=8.58, confidence=0.9)
    local_match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    local_match.measurement_mode = "cad_local_neighborhood"
    local_match.measurement_method = "surface_classification"
    local_match.feature_family = "COUNTABLE_DISCRETE_BEND"
    local_match.line_length_deviation_mm = 0.0
    local_report = BendInspectionReport(part_id="PART", matches=[local_match])
    local_report.compute_summary()

    merged_match = _make_match(cad_bends[0], "FAIL", angle_deviation=8.58, confidence=0.9)
    merged_match.assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
    merged_match.measurement_mode = "cad_local_neighborhood"
    merged_match.measurement_method = "surface_classification"
    merged_match.feature_family = "COUNTABLE_DISCRETE_BEND"
    merged_match.line_length_deviation_mm = 0.0
    merged_report = BendInspectionReport(part_id="PART", matches=[merged_match])
    merged_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "legacy_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((600_000, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (primary_report, 1, False),
    )
    monkeypatch.setattr(
        pipeline,
        "_dense_scan_requires_local_refinement",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        pipeline,
        "refine_dense_scan_bends_via_alignment",
        lambda **kwargs: (local_report, ["Dense-scan local refinement used CAD-guided ICP alignment (voxel=1.50mm, fitness=90.00%, rmse=1.000mm)"]),
    )
    monkeypatch.setattr(
        pipeline,
        "merge_bend_reports",
        lambda *args, **kwargs: merged_report,
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        expected_bend_override=1,
        scan_state="full",
    )

    assert report.matches[0].assignment_source == "CAD_LOCAL_NEIGHBORHOOD"
    assert report.matches[0].line_length_deviation_mm == 0.0
    assert any("short-obtuse geometry plausibility" in w for w in details.warnings)


def test_run_progressive_bend_inspection_applies_explicit_expected_bend_override(monkeypatch):
    cad_bends = [_make_spec("B1"), _make_spec("B2"), _make_spec("B3")]
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_expected_bend_overrides",
        lambda: {},
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "detector_mesh"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((50_000, 3)),
    )

    def _fake_detect_and_match_with_fallback(**kwargs):
        captured["expected_bend_count"] = kwargs["expected_bend_count"]
        captured["cad_bends"] = [b.bend_id for b in kwargs["cad_bends"]]
        report = BendInspectionReport(
            part_id="PART",
            matches=[
                BendMatch(cad_bend=kwargs["cad_bends"][0], detected_bend=None, status="PASS"),
                BendMatch(cad_bend=kwargs["cad_bends"][1], detected_bend=None, status="PASS"),
            ],
        )
        report.compute_summary()
        return report, 2, False

    monkeypatch.setattr(pipeline, "detect_and_match_with_fallback", _fake_detect_and_match_with_fallback)

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        expected_bend_override=2,
        scan_state="partial",
    )

    assert report.detected_count == 2
    assert captured["expected_bend_count"] == 2
    assert captured["cad_bends"] == ["CAD_B1", "CAD_B2"]
    assert details.expected_bend_count == 2
    assert any("Expected bend override applied for PART: CAD=3 expected=2" in w for w in details.warnings)


def test_run_progressive_bend_inspection_applies_part_feature_policy(monkeypatch):
    cad_bends = [
        _make_spec("R1", angle=140.0),
        _make_spec("R2", angle=141.0),
    ]
    for bend in cad_bends:
        bend.bend_form = "ROLLED"

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(pipeline, "load_expected_bend_overrides", lambda: {})
    monkeypatch.setattr(
        pipeline,
        "load_part_feature_policies",
        lambda: {
            "PART": {
                "non_counting_feature_forms": ["ROLLED"],
                "rolled_feature_type": "ROLLED_SECTION",
            }
        },
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "detector_mesh"),
    )
    monkeypatch.setattr(pipeline, "load_scan_points", lambda scan_path: np.zeros((5_000, 3)))

    def _fake_detect_and_match_with_fallback(**kwargs):
        report = BendInspectionReport(
            part_id="PART",
            matches=[
                BendMatch(cad_bend=kwargs["cad_bends"][0], detected_bend=None, status="PASS"),
                BendMatch(cad_bend=kwargs["cad_bends"][1], detected_bend=None, status="PASS"),
            ],
        )
        report.compute_summary()
        return report, 2, False

    monkeypatch.setattr(pipeline, "detect_and_match_with_fallback", _fake_detect_and_match_with_fallback)
    monkeypatch.setattr(pipeline, "_dense_scan_requires_local_refinement", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        pipeline,
        "estimate_scan_quality",
        lambda **kwargs: {"status": "GOOD", "coverage_pct": 80.0, "density_pts_per_cm2": 10.0},
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        expected_bend_override=2,
        scan_state="partial",
    )

    assert report.detected_count == 0
    assert report.total_cad_bends == 0
    assert report.to_dict()["summary"]["detected_process_features"] == 2
    assert all(match.cad_bend.countable_in_regression is False for match in report.matches)
    assert all(match.cad_bend.feature_type == "ROLLED_SECTION" for match in report.matches)
    assert any("Applied part feature policy for PART" in w for w in details.warnings)


def test_dense_scan_requires_local_refinement_for_near_complete_gap():
    cad_bends = [_make_spec("B1"), _make_spec("B2"), _make_spec("B3"), _make_spec("B4"), _make_spec("B5"), _make_spec("B6"), _make_spec("B7")]
    report = BendInspectionReport(
        part_id="PART",
        matches=[
            BendMatch(cad_bend=cad_bends[0], detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad_bends[1], detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad_bends[2], detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad_bends[3], detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad_bends[4], detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad_bends[5], detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad_bends[6], detected_bend=None, status="NOT_DETECTED"),
        ],
    )
    report.compute_summary()

    assert pipeline._dense_scan_requires_local_refinement(np.zeros((600_000, 3)), report, 7) is True


def test_dense_scan_skips_local_refinement_for_medium_partial_under_recovery():
    cad_bends = [_make_spec(f"B{i}") for i in range(1, 9)]
    matches = []
    for idx, bend in enumerate(cad_bends):
        status = "PASS" if idx == 0 else "NOT_DETECTED"
        matches.append(BendMatch(cad_bend=bend, detected_bend=None, status=status))
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    assert pipeline._dense_scan_requires_local_refinement(
        np.zeros((350_000, 3)),
        report,
        8,
        scan_state="partial",
    ) is False


def test_dense_scan_requires_local_refinement_for_large_partial_severe_under_recovery():
    cad_bends = [_make_spec(f"B{i}") for i in range(1, 17)]
    matches = []
    for idx, bend in enumerate(cad_bends):
        status = "FAIL" if idx < 3 else "NOT_DETECTED"
        matches.append(BendMatch(cad_bend=bend, detected_bend=None, status=status))
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    assert pipeline._dense_scan_requires_local_refinement(
        np.zeros((391_613, 3)),
        report,
        16,
        scan_state="partial",
    ) is True


def test_dense_scan_requires_local_refinement_for_medium_full_severe_under_recovery():
    cad_bends = [_make_spec(f"B{i}") for i in range(1, 17)]
    matches = []
    for idx, bend in enumerate(cad_bends):
        status = "PASS" if idx < 10 else "NOT_DETECTED"
        matches.append(BendMatch(cad_bend=bend, detected_bend=None, status=status))
    report = BendInspectionReport(part_id="PART", matches=matches)
    report.compute_summary()

    assert pipeline._dense_scan_requires_local_refinement(
        np.zeros((473_764, 3)),
        report,
        16,
        scan_state="full",
    ) is True




def test_choose_mesh_cad_baseline_prefers_spec_hint_when_coverage_equal_but_count_gap_smaller():
    detector_specs = [
        _make_spec("B1", angle=90.0),
        _make_spec("B2", angle=90.0),
        _make_spec("B3", angle=90.0),
        _make_spec("B4", angle=90.0),
    ]
    legacy_specs = [_make_spec("L1", angle=90.0)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
        spec_bend_angles_hint=[90.0],
        spec_schedule_type="complete_or_near_complete",
    )

    assert strategy == "legacy_mesh_spec_hint"
    assert len(selected) == 1


def test_choose_mesh_cad_baseline_prefers_spec_hint_when_matches_improve():
    detector_specs = [
        _make_spec("B1", angle=90.0),
        _make_spec("B2", angle=125.0),
    ]
    legacy_specs = [_make_spec("L1", angle=90.0)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
        spec_bend_angles_hint=[90.0, 125.0],
        spec_schedule_type="complete_or_near_complete",
    )

    assert strategy == "detector_mesh_spec_hint"
    assert len(selected) == 2


def test_choose_mesh_cad_baseline_builds_hybrid_for_rolled_shell_policy():
    rolled_1 = _make_spec("R1", angle=140.0)
    rolled_1.bend_form = "ROLLED"
    rolled_2 = _make_spec("R2", angle=142.0)
    rolled_2.bend_form = "ROLLED"

    folded_a = _make_spec("L1", angle=88.0)
    folded_a.flange1_area = 80.0
    folded_a.flange2_area = 1200.0
    folded_a.bend_line_end = np.array([0.0, 30.0, 0.0])

    folded_b = _make_spec("L2", angle=92.0)
    folded_b.flange1_area = 90.0
    folded_b.flange2_area = 1100.0
    folded_b.bend_line_start = np.array([25.0, 0.0, 0.0])
    folded_b.bend_line_end = np.array([25.0, 30.0, 0.0])

    pseudo = _make_spec("L3", angle=40.0)
    pseudo.flange1_area = 800.0
    pseudo.flange2_area = 900.0
    pseudo.bend_line_end = np.array([0.0, 8.0, 0.0])

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=[rolled_1, rolled_2],
        legacy_specs=[folded_a, folded_b, pseudo],
        expected_bend_count_hint=2,
        feature_policy={
            "non_counting_feature_forms": ["ROLLED"],
            "rolled_feature_type": "ROLLED_SECTION",
            "countable_source": "legacy_folded_hybrid",
        },
    )

    assert strategy == "feature_policy_rolled_hybrid"
    assert len(selected) == 4
    assert sum(1 for spec in selected if spec.countable_in_regression) == 2
    assert sum(1 for spec in selected if not spec.countable_in_regression) == 2
    assert all(spec.feature_type == "ROLLED_SECTION" for spec in selected[:2])
    assert all(spec.feature_type == "DISCRETE_BEND" for spec in selected[2:])


def test_choose_mesh_cad_baseline_hybrid_uses_angle_hints_instead_of_default_90_bias():
    rolled_1 = _make_spec("R1", angle=140.0)
    rolled_1.bend_form = "ROLLED"
    rolled_2 = _make_spec("R2", angle=142.0)
    rolled_2.bend_form = "ROLLED"

    folded_90_a = _make_spec("L1", angle=88.0)
    folded_90_a.flange1_area = 80.0
    folded_90_a.flange2_area = 1200.0
    folded_90_a.bend_line_end = np.array([0.0, 30.0, 0.0])

    folded_90_b = _make_spec("L2", angle=92.0)
    folded_90_b.flange1_area = 90.0
    folded_90_b.flange2_area = 1100.0
    folded_90_b.bend_line_start = np.array([25.0, 0.0, 0.0])
    folded_90_b.bend_line_end = np.array([25.0, 30.0, 0.0])

    hinted_a = _make_spec("L3", angle=155.0)
    hinted_a.flange1_area = 82.0
    hinted_a.flange2_area = 1110.0
    hinted_a.bend_line_start = np.array([0.0, 0.0, 0.0])
    hinted_a.bend_line_end = np.array([0.0, 30.0, 0.0])

    hinted_b = _make_spec("L4", angle=157.55)
    hinted_b.flange1_area = 84.0
    hinted_b.flange2_area = 1090.0
    hinted_b.bend_line_start = np.array([25.0, 0.0, 0.0])
    hinted_b.bend_line_end = np.array([25.0, 30.0, 0.0])

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=[rolled_1, rolled_2],
        legacy_specs=[folded_90_a, folded_90_b, hinted_a, hinted_b],
        expected_bend_count_hint=2,
        spec_bend_angles_hint=[25.0, 22.45],
        feature_policy={
            "non_counting_feature_forms": ["ROLLED"],
            "rolled_feature_type": "ROLLED_SECTION",
            "countable_source": "legacy_folded_hybrid",
        },
    )

    assert strategy == "feature_policy_rolled_hybrid"
    countable = [spec for spec in selected if spec.countable_in_regression]
    assert len(countable) == 2
    assert sorted(round(float(spec.target_angle), 2) for spec in countable) == [155.0, 157.55]


def test_choose_mesh_cad_baseline_hybrid_uses_family_selector_for_rolled_shell_with_two_flange_bends():
    rolled_1 = _make_spec("R1", angle=140.0, bend_form="ROLLED")
    rolled_1.bend_line_start = np.array([0.0, 0.0, -40.0])
    rolled_1.bend_line_end = np.array([0.0, 0.0, 40.0])
    rolled_2 = _make_spec("R2", angle=142.0, bend_form="ROLLED")
    rolled_2.bend_line_start = np.array([10.0, 0.0, -40.0])
    rolled_2.bend_line_end = np.array([10.0, 0.0, 40.0])

    upper_short = _make_spec("L_upper", angle=90.0)
    upper_short.bend_line_start = np.array([-20.0, 54.0, -8.0])
    upper_short.bend_line_end = np.array([-34.0, 16.0, -8.0])
    upper_short.flange1_area = 36.0
    upper_short.flange2_area = 50.0
    upper_short.flange1_normal = np.array([0.0, 0.0, 1.0])
    upper_short.flange2_normal = np.array([1.0, 0.0, 0.0])

    upper_wrong_deep = _make_spec("L_wrong_short", angle=90.0)
    upper_wrong_deep.bend_line_start = np.array([-25.0, 26.0, -97.0])
    upper_wrong_deep.bend_line_end = np.array([-25.0, 26.0, -57.0])
    upper_wrong_deep.flange1_area = 140.0
    upper_wrong_deep.flange2_area = 775.0
    upper_wrong_deep.flange1_normal = np.array([0.0, 0.0, 1.0])
    upper_wrong_deep.flange2_normal = np.array([1.0, 0.0, 0.0])

    upper_wrong_long = _make_spec("L_wrong_long", angle=90.0)
    upper_wrong_long.bend_line_start = np.array([-20.3, 35.3, -102.0])
    upper_wrong_long.bend_line_end = np.array([-20.3, 35.3, -58.0])
    upper_wrong_long.flange1_area = 36.0
    upper_wrong_long.flange2_area = 775.0
    upper_wrong_long.flange1_normal = np.array([1.0, 0.0, 0.0])
    upper_wrong_long.flange2_normal = np.array([0.0, 1.0, 0.0])

    lower_wide = _make_spec("L_lower", angle=161.82)
    lower_wide.bend_line_start = np.array([-25.8, 26.0, -76.0])
    lower_wide.bend_line_end = np.array([-25.8, 26.0, 60.0])
    lower_wide.flange1_area = 775.0
    lower_wide.flange2_area = 1884.0
    lower_wide.flange1_normal = np.array([1.0, 0.0, 0.0])
    lower_wide.flange2_normal = np.array([0.0, 1.0, 0.0])

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=[rolled_1, rolled_2],
        legacy_specs=[upper_wrong_long, upper_wrong_deep, lower_wide, upper_short],
        expected_bend_count_hint=2,
        spec_bend_angles_hint=[25.0, 22.45],
        feature_policy={
            "non_counting_feature_forms": ["ROLLED"],
            "rolled_feature_type": "ROLLED_SECTION",
            "countable_source": "legacy_folded_hybrid",
            "part_family": "rolled_shell_with_two_flange_bends",
        },
    )

    assert strategy == "feature_policy_rolled_hybrid_flange_pair"
    countable = [spec for spec in selected if spec.countable_in_regression]
    assert len(countable) == 2
    mids = [0.5 * (np.asarray(spec.bend_line_start) + np.asarray(spec.bend_line_end)) for spec in countable]
    ys = sorted(round(float(mid[1]), 1) for mid in mids)
    zs = sorted(round(float(mid[2]), 1) for mid in mids)
    assert ys == [26.0, 35.0]
    assert zs == [-8.0, -8.0]
    assert any(round(float(spec.bend_line_length), 1) < 70.0 for spec in countable)
    assert any(round(float(spec.bend_line_length), 1) > 100.0 for spec in countable)


def test_apply_part_feature_policy_uses_metadata_truth_angle_overrides(monkeypatch, tmp_path):
    metadata_root = tmp_path / "data" / "bend_corpus" / "parts" / "28870000"
    metadata_root.mkdir(parents=True)
    (metadata_root / "metadata.json").write_text(
        json.dumps(
            {
                "user_truth": {
                    "countable_bend_target_angles_deg": {
                        "CAD_B4": 25.0,
                        "CAD_B5": 22.45,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    fake_pipeline = tmp_path / "backend" / "bend_inspection_pipeline.py"
    fake_pipeline.parent.mkdir(parents=True)
    fake_pipeline.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(pipeline, "__file__", str(fake_pipeline))

    policy = pipeline._augment_feature_policy_from_part_truth(
        {"countable_source": "legacy_folded_hybrid"},
        "28870000",
    )

    assert policy["countable_bend_angle_overrides"] == {"CAD_B4": 25.0, "CAD_B5": 22.45}


def test_augment_feature_policy_from_part_truth_preserves_part_family(monkeypatch, tmp_path):
    metadata_root = tmp_path / "data" / "bend_corpus" / "parts" / "28870000"
    metadata_root.mkdir(parents=True)
    (metadata_root / "metadata.json").write_text(
        json.dumps(
            {
                "user_truth": {
                    "part_family": "rolled_shell_with_two_flange_bends",
                }
            }
        ),
        encoding="utf-8",
    )

    fake_pipeline = tmp_path / "backend" / "bend_inspection_pipeline.py"
    fake_pipeline.parent.mkdir(parents=True)
    fake_pipeline.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(pipeline, "__file__", str(fake_pipeline))

    policy = pipeline._augment_feature_policy_from_part_truth(
        {"countable_source": "legacy_folded_hybrid"},
        "28870000",
    )

    assert policy["part_family"] == "rolled_shell_with_two_flange_bends"


def test_refine_dense_scan_uses_aligned_raw_cloud_for_hybrid_parts(monkeypatch):
    import cad_bend_extractor
    import qc_engine
    import bend_detector

    cad_bends = [_make_spec("CAD_B4", angle=25.0), _make_spec("CAD_B5", angle=22.45)]
    for idx, bend in enumerate(cad_bends):
        bend.countable_in_regression = True
        bend.bend_line_start = np.array([float(idx), 0.0, 0.0])
        bend.bend_line_end = np.array([float(idx), 30.0, 0.0])
        bend.flange1_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        bend.flange2_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    raw_points = np.zeros((100, 3), dtype=np.float64)
    processed_points = np.zeros((12, 3), dtype=np.float64)
    captured = {}

    class _DummyCloud:
        def __init__(self, pts):
            self.points = pts

    class _DummyMesh:
        def __init__(self):
            self.vertices = np.zeros((4, 3), dtype=np.float64)
            self.triangles = np.array([[0, 1, 2]], dtype=np.int32)

    class _DummyEngine:
        def __init__(self, progress_callback=None):
            self.progress_callback = progress_callback
            self.scan_pcd = _DummyCloud(raw_points)
            self.aligned_scan = _DummyCloud(processed_points)
            self.aligned_scan_raw = _DummyCloud(raw_points)
            self.reference_mesh = _DummyMesh()

        def load_reference(self, cad_path):
            return True

        def load_scan(self, scan_path):
            return True

        def preprocess(self, voxel_size=1.0):
            return True

        def align(self, auto_scale=True, tolerance=1.0, random_seed=None):
            return 0.99, 0.1

    class _DummyExtract:
        def __init__(self):
            self.total_bends = 1
            self.bends = [cad_bends[0]]
            self.surfaces = []

    monkeypatch.setattr(qc_engine, "ScanQCEngine", _DummyEngine)
    monkeypatch.setattr(
        pipeline,
        "_align_engine_with_retries",
        lambda **kwargs: (processed_points.copy(), 0.99, 0.1, 17, []),
    )
    monkeypatch.setattr(
        cad_bend_extractor,
        "CADBendExtractor",
        lambda: type("_Extractor", (), {"extract_from_mesh": lambda self, mesh: _DummyExtract()})(),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_selected_measurement_dicts",
        lambda selected_bends, extracted_bends: (
            [
                {
                    "bend_id": bend.bend_id,
                    "bend_angle": float(bend.target_angle),
                    "bend_center": ((np.asarray(bend.bend_line_start) + np.asarray(bend.bend_line_end)) / 2.0).tolist(),
                    "bend_line_point1": np.asarray(bend.bend_line_start).tolist(),
                    "bend_line_point2": np.asarray(bend.bend_line_end).tolist(),
                    "surface1_normal": np.asarray(bend.flange1_normal).tolist(),
                    "surface2_normal": np.asarray(bend.flange2_normal).tolist(),
                }
                for bend in selected_bends
            ],
            0,
        ),
    )

    def _fake_measure_all_scan_bends(points, cad_bend_dicts, **kwargs):
        captured["points"] = len(points)
        return [
            {
                "success": False,
                "measured_angle": None,
                "measurement_method": "probe_region",
                "measurement_confidence": "low",
                "point_count": 0,
                "side1_points": 0,
                "side2_points": 0,
                "error": "stub",
            }
            for _ in cad_bend_dicts
        ]

    monkeypatch.setattr(bend_detector, "measure_all_scan_bends", _fake_measure_all_scan_bends)

    report, warnings = pipeline.refine_dense_scan_bends_via_alignment(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        cad_bends=cad_bends,
        part_id="28870000",
        tolerance_angle=1.0,
        feature_policy={
            "countable_source": "legacy_folded_hybrid",
        },
    )

    assert report is not None
    assert captured["points"] == len(raw_points)


def test_choose_mesh_cad_baseline_can_supplement_detector_with_spatially_distinct_legacy_specs():
    detector_specs = []
    for idx, x in enumerate([100.0, 160.0, 20.0, 25.0, 162.0], start=1):
        spec = BendSpecification(
            bend_id=f"D{idx}",
            target_angle=90.0,
            target_radius=12.0,
            bend_line_start=np.array([x, 0.0, 0.0]),
            bend_line_end=np.array([x, 50.0, 0.0]),
            flange1_area=1000.0,
            flange2_area=800.0,
            bend_form="ROLLED",
        )
        detector_specs.append(spec)

    legacy_specs = []
    for idx, x in enumerate([-125.0, -85.0, 92.0, 158.0, -122.0, 210.0, -210.0, 250.0, -260.0, 300.0], start=1):
        spec = BendSpecification(
            bend_id=f"L{idx}",
            target_angle=112.0,
            target_radius=3.0,
            bend_line_start=np.array([x, -10.0, 0.0]),
            bend_line_end=np.array([x, 35.0, 0.0]),
            flange1_area=240.0,
            flange2_area=210.0,
            bend_form="FOLDED",
        )
        legacy_specs.append(spec)

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=7,
        spec_bend_angles_hint=[],
        spec_schedule_type=None,
    )

    assert strategy == "detector_mesh_expected_supplement"
    assert len(selected) == 7
    mids = [0.5 * (s.bend_line_start + s.bend_line_end) for s in selected]
    xs = sorted(round(float(m[0]), 1) for m in mids)
    assert sum(1 for x in xs if x < 0.0) >= 2


def test_choose_mesh_cad_baseline_ignores_mixed_section_callouts_for_strong_spec_selection():
    detector_specs = [
        _make_spec("B1", angle=90.0),
        _make_spec("B2", angle=125.0),
    ]
    legacy_specs = [_make_spec("L1", angle=90.0)]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
        spec_bend_angles_hint=[90.0],
        spec_schedule_type="mixed_section_callouts",
    )

    assert strategy == "detector_mesh"
    assert len(selected) == 2


def test_choose_mesh_cad_baseline_uses_partial_schedule_only_as_compactness_hint():
    detector_specs = [
        _make_spec("B1", angle=90.0),
        _make_spec("B2", angle=125.0),
    ]
    legacy_specs = [
        _make_spec("L1", angle=90.0),
        _make_spec("L2", angle=90.0),
        _make_spec("L3", angle=90.0),
        _make_spec("L4", angle=125.0),
    ]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
        spec_bend_angles_hint=[90.0, 125.0],
        spec_schedule_type="partial_angle_schedule",
    )

    assert strategy == "detector_mesh_partial_schedule_hint"
    assert len(selected) == 2


def test_choose_mesh_cad_baseline_prefers_rolled_profile_detector_when_legacy_explodes_profile():
    detector_specs = [_make_spec("B1", angle=96.0, radius=12.0, bend_form="ROLLED")]
    legacy_specs = [
        _make_spec("L1", angle=90.0, bend_form="FOLDED"),
        _make_spec("L2", angle=90.0, bend_form="FOLDED"),
        _make_spec("L3", angle=90.0, bend_form="FOLDED"),
        _make_spec("L4", angle=125.0, bend_form="FOLDED"),
    ]

    selected, strategy = pipeline._choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=None,
        spec_bend_angles_hint=[90.0],
    )

    assert strategy == "detector_mesh_rolled_profile_hint"
    assert len(selected) == 1


def test_run_progressive_bend_inspection_merges_local_refinement_without_erasing_primary(monkeypatch):
    cad_bends = [_make_spec("B1"), _make_spec("B2"), _make_spec("B3")]
    primary_report = BendInspectionReport(
        part_id="PART",
        matches=[
            _mark_confident_global_match(_make_match(cad_bends[0], "PASS", angle_deviation=0.5), "B1"),
            _make_match(cad_bends[1], "NOT_DETECTED"),
            _mark_confident_global_match(_make_match(cad_bends[2], "PASS", angle_deviation=0.7), "B3"),
        ],
    )
    primary_report.compute_summary()
    local_report = BendInspectionReport(
        part_id="PART",
        matches=[
            _make_match(cad_bends[0], "NOT_DETECTED"),
            _mark_confident_local_match(_make_match(cad_bends[1], "FAIL", angle_deviation=7.0), "local_candidate_b2"),
            _mark_confident_local_match(_make_match(cad_bends[2], "FAIL", angle_deviation=12.0), "local_candidate_b3"),
        ],
    )
    local_report.compute_summary()

    monkeypatch.setattr(
        pipeline,
        "load_cad_geometry",
        lambda cad_path, cad_import_deflection=0.05: (np.zeros((20, 3)), np.array([[0, 1, 2]]), "mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_cad_bend_specs_with_strategy",
        lambda **kwargs: (cad_bends, "legacy_mesh_expected_hint"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_scan_points",
        lambda scan_path: np.zeros((600_000, 3)),
    )
    monkeypatch.setattr(
        pipeline,
        "detect_and_match_with_fallback",
        lambda **kwargs: (primary_report, 2, False),
    )
    monkeypatch.setattr(
        pipeline,
        "refine_dense_scan_bends_via_alignment",
        lambda **kwargs: (local_report, ["Dense-scan local refinement used CAD-guided ICP alignment (voxel=1.50mm, fitness=90.00%, rmse=1.000mm)"]),
    )

    report, details = pipeline.run_progressive_bend_inspection(
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.ply",
        part_id="PART",
        tolerance_angle=1.0,
        tolerance_radius=0.5,
    )

    assert report.detected_count == 3
    assert [m.status for m in report.matches] == ["PASS", "FAIL", "PASS"]
    assert details.scan_bend_count == 3
    assert any("Dense-scan local refinement merged with primary global detection" in w for w in details.warnings)
