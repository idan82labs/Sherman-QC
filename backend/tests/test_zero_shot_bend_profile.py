from types import SimpleNamespace

import numpy as np

import zero_shot_bend_profile as zsbp


def _fake_bend(angle: float, direction, start, end, confidence: float = 0.9):
    start_arr = np.asarray(start, dtype=np.float64)
    end_arr = np.asarray(end, dtype=np.float64)
    direction_arr = np.asarray(direction, dtype=np.float64)
    return SimpleNamespace(
        measured_angle=float(angle),
        bend_line_direction=direction_arr,
        bend_line_start=start_arr,
        bend_line_end=end_arr,
        bend_line_length=float(np.linalg.norm(end_arr - start_arr)),
        confidence=float(confidence),
    )


def test_select_zero_shot_result_abstains_on_low_margin():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.72,
            estimated_bend_count=5,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.0],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.7,
            structural_compactness=0.7,
            perturbation_stability=0.7,
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.75,
            estimated_bend_count=6,
            estimated_bend_count_range=(5, 6),
            dominant_angle_families_deg=[90.0],
            exact_count_eligible=True,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.8,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.70,
            estimated_bend_count=4,
            estimated_bend_count_range=(4, 5),
            dominant_angle_families_deg=[120.0],
            exact_count_eligible=False,
            angle_family_coherence=0.7,
            duplicate_family_suppression=0.75,
            structural_compactness=0.75,
            perturbation_stability=0.75,
        ),
    }
    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 5,
            "multiscale_count": 6,
            "union_candidate_count": 8,
            "duplicate_line_family_rate": 0.1,
            "dominant_angle_family_mass": 0.8,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.profile_family == "repeated_90_sheetmetal"
    assert "policy_margin_low" in result.abstain_reasons
    assert result.estimated_bend_count is None


def test_select_zero_shot_result_keeps_exact_count_when_top_policies_agree():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.91,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 1),
            dominant_angle_families_deg=[126.0],
            exact_count_eligible=True,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.95,
            perturbation_stability=1.0,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.905,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 1),
            dominant_angle_families_deg=[129.0],
            exact_count_eligible=True,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.95,
            perturbation_stability=1.0,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 1,
            "multiscale_count": 1,
            "union_candidate_count": 2,
            "duplicate_line_family_rate": 0.0,
            "dominant_angle_family_mass": 1.0,
            "dominant_angle_family_deg": 126.0,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count == 1
    assert result.profile_family == "conservative_macro"


def test_select_zero_shot_result_prefers_orthogonal_dense_on_repeated_90_overcount():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.86,
            estimated_bend_count=12,
            estimated_bend_count_range=(12, 13),
            dominant_angle_families_deg=[90.6],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.57,
            perturbation_stability=0.75,
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.64,
            estimated_bend_count=10,
            estimated_bend_count_range=(9, 10),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.32,
            structural_compactness=0.46,
            perturbation_stability=0.73,
            support_breakdown={"guard_ok": True},
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.44,
            estimated_bend_count=14,
            estimated_bend_count_range=(12, 14),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.64,
            perturbation_stability=0.47,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 12,
            "multiscale_count": 19,
            "union_candidate_count": 31,
            "duplicate_line_family_rate": 0.48,
            "dominant_angle_family_mass": 0.87,
            "dominant_angle_family_deg": 90.6,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.profile_family == "orthogonal_dense_sheetmetal"
    assert result.estimated_bend_count_range == (9, 10)


def test_select_zero_shot_result_abstains_on_mixed_angle_conservative_winner():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.91,
            estimated_bend_count=5,
            estimated_bend_count_range=(5, 5),
            dominant_angle_families_deg=[91.3, 130.6],
            exact_count_eligible=False,
            angle_family_coherence=0.95,
            duplicate_family_suppression=1.0,
            structural_compactness=0.8,
            perturbation_stability=0.9,
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.44,
            estimated_bend_count=6,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.45,
            structural_compactness=0.64,
            perturbation_stability=0.47,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.50,
            estimated_bend_count=8,
            estimated_bend_count_range=(5, 8),
            dominant_angle_families_deg=[91.0, 130.0, 168.0],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.8,
            structural_compactness=0.7,
            perturbation_stability=0.46,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test.stl",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 5,
            "multiscale_count": 8,
            "union_candidate_count": 13,
            "duplicate_line_family_rate": 0.38,
            "dominant_angle_family_mass": 0.77,
            "dominant_angle_family_deg": 91.3,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.profile_family == "conservative_macro"
    assert "mixed_angle_family_ambiguous" in result.abstain_reasons
    assert result.estimated_bend_count_range == (4, 8)


def test_select_zero_shot_result_keeps_consensus_count_range_on_low_margin():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.8641,
            estimated_bend_count=10,
            estimated_bend_count_range=(9, 10),
            dominant_angle_families_deg=[90.188],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.57,
            perturbation_stability=0.75,
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.7377,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 11),
            dominant_angle_families_deg=[90.179],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.4,
            structural_compactness=0.5,
            perturbation_stability=0.72,
            support_breakdown={"guard_ok": True},
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.6177,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 11),
            dominant_angle_families_deg=[90.179],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.7,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.8640,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 11),
            dominant_angle_families_deg=[90.171],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.75,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 10,
            "multiscale_count": 11,
            "union_candidate_count": 21,
            "duplicate_line_family_rate": 0.5,
            "dominant_angle_family_mass": 1.0,
            "dominant_angle_family_deg": 90.18,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (9, 11)
    assert result.support_breakdown["structural_count_consensus"] == [9, 11]


def test_select_zero_shot_result_reports_rolled_family_while_abstaining():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.67,
            estimated_bend_count=10,
            estimated_bend_count_range=(5, 10),
            dominant_angle_families_deg=[148.4, 113.4, 167.2],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.8,
            structural_compactness=0.6,
            perturbation_stability=0.4,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.22,
            estimated_bend_count=13,
            estimated_bend_count_range=(11, 13),
            dominant_angle_families_deg=[156.0, 115.0, 139.5],
            exact_count_eligible=False,
            angle_family_coherence=0.7,
            duplicate_family_suppression=0.6,
            structural_compactness=0.6,
            perturbation_stability=0.4,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 10,
            "multiscale_count": 13,
            "union_candidate_count": 23,
            "duplicate_line_family_rate": 0.6,
            "dominant_angle_family_mass": 0.6,
            "dominant_angle_family_deg": 148.4,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.profile_family == "rolled_or_mixed_transition"


def test_select_zero_shot_result_does_not_force_scan_limited_from_metadata_hint_alone():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.96,
            estimated_bend_count=4,
            estimated_bend_count_range=(4, 4),
            dominant_angle_families_deg=[90.2],
            exact_count_eligible=True,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.9,
            perturbation_stability=1.0,
            support_breakdown={
                "bend_support_strips": [
                    {
                        "strip_id": "S1",
                        "strip_family": "discrete_fold_strip",
                        "axis_direction": [1.0, 0.0, 0.0],
                        "axis_line_hint": {"midpoint": [0.0, 0.0, 0.0]},
                        "adjacent_flange_normals": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        "dominant_angle_deg": 90.2,
                        "radius_mm": 1.0,
                        "support_patch_ids": ["patch_1"],
                        "support_point_count": 12,
                        "cross_fold_width_mm": 3.0,
                        "along_axis_span_mm": 20.0,
                        "support_bbox_mm": [1.0, 20.0, 1.0],
                        "observability_status": "supported",
                        "merge_provenance": {},
                        "stability_score": 1.0,
                        "confidence": 0.95,
                    }
                ]
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test_partial.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 10,
            "multiscale_count": 4,
            "union_candidate_count": 14,
            "duplicate_line_family_rate": 0.1,
            "dominant_angle_family_mass": 1.0,
            "dominant_angle_family_deg": 90.2,
            "scan_state_hint": "partial",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.estimated_bend_count is None
    assert result.profile_family == "scan_limited_partial"
    assert "scan_state_partial" in result.abstain_reasons


def test_select_zero_shot_result_allows_full_scan_rolled_structural_range():
    policy_scores = {
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.6936,
            estimated_bend_count=13,
            estimated_bend_count_range=(7, 14),
            dominant_angle_families_deg=[95.6, 125.4, 164.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.3968,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
        ),
        "conservative_macro": zsbp.PolicyScore(
            score=0.5109,
            estimated_bend_count=8,
            estimated_bend_count_range=(7, 10),
            dominant_angle_families_deg=[91.1, 107.9, 162.6],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.57,
            perturbation_stability=0.4831,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test_full.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 8,
            "multiscale_count": 13,
            "union_candidate_count": 21,
            "duplicate_line_family_rate": 0.47,
            "dominant_angle_family_mass": 0.71,
            "dominant_angle_family_deg": 95.6,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.profile_family == "rolled_or_mixed_transition"
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (7, 14)


def test_repeated_90_policy_collapses_duplicate_line_families():
    original_single = [
        _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
        _fake_bend(90.2, [1, 0, 0], [2, 0, 0], [2, 20, 0]),
    ]
    original_multi = original_single + [
        _fake_bend(91.0, [0, 1, 0], [0, 30, 0], [20, 30, 0]),
        _fake_bend(140.0, [0, 1, 0], [0, 60, 0], [20, 60, 0]),
    ]
    bundles = {
        "original": {"single_scale": original_single, "multiscale": original_multi},
        "voxel_0.8": {"single_scale": original_single, "multiscale": original_multi},
        "voxel_1.2": {"single_scale": original_single, "multiscale": original_multi},
    }
    scan_features = {
        "dominant_angle_family_mass": 0.75,
    }

    score = zsbp._score_repeated_90_sheetmetal(scan_features, bundles)

    assert score.estimated_bend_count == 2
    assert score.dominant_angle_families_deg == [90.28]
    assert score.support_breakdown["raw_candidate_count"] == 6
    assert len(score.support_breakdown["bend_support_strips"]) == 2
    assert score.support_breakdown["strip_duplicate_merge_count"] >= 1


def test_effective_strip_count_can_cap_to_line_groups():
    strip_summary = {
        "strip_count": 14,
        "line_groups": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
    }

    assert zsbp._effective_strip_count(strip_summary) == 14
    assert zsbp._effective_strip_count(strip_summary, cap_to_line_groups=True) == 10


def test_perturbation_count_consensus_range_trims_single_outlier():
    perturbation = {
        "counts": [11, 14, 12],
    }

    assert zsbp._perturbation_count_consensus_range(perturbation) == (11, 12)


def test_identity_exact_count_eligible_requires_strong_identity_and_mode_support():
    perturbation = {
        "count_span": 1,
        "identity_stability": 0.91,
        "modal_count_support": 2.0 / 3.0,
    }

    assert zsbp._identity_exact_count_eligible(perturbation, dominant_mass=0.9) is True
    assert zsbp._identity_exact_count_eligible(perturbation, dominant_mass=0.5) is False


def test_identity_exact_count_recovery_eligible_requires_score_and_stability_floor():
    perturbation = {
        "count_span": 1,
        "identity_stability": 0.91,
        "modal_count_support": 2.0 / 3.0,
    }

    assert zsbp._identity_exact_count_recovery_eligible(
        score=0.64,
        stability=0.76,
        perturbation=perturbation,
        dominant_mass=0.87,
        min_score=0.6,
    ) is True
    assert zsbp._identity_exact_count_recovery_eligible(
        score=0.55,
        stability=0.76,
        perturbation=perturbation,
        dominant_mass=0.87,
        min_score=0.6,
    ) is False


def test_stability_score_includes_identity_stability():
    perturbation = {
        "counts": [10, 10, 9],
        "dominant_angle_families_deg": [91.26, 91.1, 90.55],
        "identity_stability": 0.9,
    }

    assert zsbp._stability_score(perturbation) > 0.7


def test_perturbation_summary_includes_variant_summaries():
    bundles = {
        "original": {
            "single_scale": [
                _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
                _fake_bend(90.1, [1, 0, 0], [40, 0, 0], [40, 20, 0]),
            ],
            "multiscale": [
                _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
                _fake_bend(90.1, [1, 0, 0], [40, 0, 0], [40, 20, 0]),
                _fake_bend(90.2, [1, 0, 0], [80, 0, 0], [80, 20, 0]),
            ],
        },
        "voxel_0.8": {
            "single_scale": [
                _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
            ],
            "multiscale": [
                _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
                _fake_bend(90.2, [1, 0, 0], [80, 0, 0], [80, 20, 0]),
            ],
        },
        "voxel_1.2": {
            "single_scale": [
                _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
                _fake_bend(90.1, [1, 0, 0], [40, 0, 0], [40, 20, 0]),
            ],
            "multiscale": [
                _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
                _fake_bend(90.1, [1, 0, 0], [40, 0, 0], [40, 20, 0]),
            ],
        },
    }

    perturbation = zsbp._perturbation_summary(
        bundles,
        "conservative_macro",
        spacing_mm=2.0,
    )

    assert set(perturbation["variant_summaries"]) == {"original", "voxel_0.8", "voxel_1.2"}
    assert perturbation["variant_summaries"]["original"]["raw_candidate_count"] == 2
    assert perturbation["variant_summaries"]["original"]["line_family_group_count"] == 1
    assert perturbation["variant_summaries"]["original"]["effective_count"] == 1
    assert perturbation["variant_summaries"]["voxel_0.8"]["raw_candidate_count"] == 1
    assert perturbation["variant_summaries"]["voxel_0.8"]["evidence_atom_count"] == 1


def test_orthogonal_dense_policy_strongly_collapses_repeated_90_candidates():
    original_single = [
        _fake_bend(90.0, [1, 0, 0], [0, 0, 0], [0, 20, 0]),
        _fake_bend(90.1, [1, 0, 0], [40, 0, 0], [40, 20, 0]),
        _fake_bend(90.2, [1, 0, 0], [80, 0, 0], [80, 20, 0]),
        _fake_bend(89.9, [1, 0, 0], [120, 0, 0], [120, 20, 0]),
        _fake_bend(90.1, [1, 0, 0], [160, 0, 0], [160, 20, 0]),
        _fake_bend(90.0, [1, 0, 0], [200, 0, 0], [200, 20, 0]),
        _fake_bend(90.0, [1, 0, 0], [240, 0, 0], [240, 20, 0]),
    ]
    original_multi = original_single + [
        _fake_bend(90.0, [1, 0, 0], [10, 0, 0], [10, 20, 0]),
        _fake_bend(90.0, [1, 0, 0], [50, 0, 0], [50, 20, 0]),
        _fake_bend(90.0, [1, 0, 0], [90, 0, 0], [90, 20, 0]),
        _fake_bend(90.0, [1, 0, 0], [130, 0, 0], [130, 20, 0]),
        _fake_bend(90.0, [1, 0, 0], [170, 0, 0], [170, 20, 0]),
    ]
    bundles = {
        "original": {"single_scale": original_single, "multiscale": original_multi},
        "voxel_0.8": {"single_scale": original_single, "multiscale": original_multi},
        "voxel_1.2": {"single_scale": original_single, "multiscale": original_multi},
    }
    scan_features = {
        "dominant_angle_family_mass": 0.9,
        "dominant_angle_family_deg": 90.0,
        "single_scale_count": len(original_single),
    }

    score = zsbp._score_orthogonal_dense_sheetmetal(scan_features, bundles)

    assert score.estimated_bend_count < len(original_multi)
    assert score.support_breakdown["guard_ok"] is True


def test_select_zero_shot_result_partial_consensus_abstains():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.8641,
            estimated_bend_count=10,
            estimated_bend_count_range=(9, 10),
            dominant_angle_families_deg=[90.188],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.57,
            perturbation_stability=0.75,
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.7377,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 11),
            dominant_angle_families_deg=[90.179],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.4,
            structural_compactness=0.5,
            perturbation_stability=0.72,
            support_breakdown={"guard_ok": True},
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.6177,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 11),
            dominant_angle_families_deg=[90.179],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.7,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.8640,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 11),
            dominant_angle_families_deg=[90.171],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.75,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test_partial.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 10,
            "multiscale_count": 11,
            "union_candidate_count": 21,
            "duplicate_line_family_rate": 0.5,
            "dominant_angle_family_mass": 1.0,
            "dominant_angle_family_deg": 90.18,
            "scan_state_hint": "partial",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.profile_family == "scan_limited_partial"
    assert "scan_state_partial" in result.abstain_reasons


def test_select_zero_shot_result_partial_low_margin_abstain_uses_scan_limited_family():
    policy_scores = {
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.71,
            estimated_bend_count=10,
            estimated_bend_count_range=(8, 11),
            dominant_angle_families_deg=[90.0],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.49,
            abstain_reasons=["policy_confidence_low", "perturbation_unstable"],
        ),
        "conservative_macro": zsbp.PolicyScore(
            score=0.7095,
            estimated_bend_count=9,
            estimated_bend_count_range=(8, 10),
            dominant_angle_families_deg=[90.0],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.9,
            structural_compactness=0.8,
            perturbation_stability=0.5,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test_partial_margin.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 9,
            "multiscale_count": 10,
            "union_candidate_count": 19,
            "duplicate_line_family_rate": 0.3,
            "dominant_angle_family_mass": 1.0,
            "dominant_angle_family_deg": 90.0,
            "scan_state_hint": "partial",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.profile_family == "scan_limited_partial"
    assert "scan_state_partial" in result.abstain_reasons


def test_select_zero_shot_result_uses_winner_perturbation_consensus_when_abstaining():
    policy_scores = {
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.7157,
            estimated_bend_count=11,
            estimated_bend_count_range=(11, 14),
            dominant_angle_families_deg=[91.379],
            exact_count_eligible=False,
            angle_family_coherence=0.95,
            duplicate_family_suppression=0.78,
            structural_compactness=0.8,
            perturbation_stability=0.48,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
            support_breakdown={
                "perturbation": {
                    "counts": [11, 14, 12],
                    "dominant_angle_families_deg": [91.379, 92.314, 91.627],
                    "duplicate_merge_counts": [3, 2, 5],
                }
            },
        ),
        "conservative_macro": zsbp.PolicyScore(
            score=0.58,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.0],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.85,
            structural_compactness=0.7,
            perturbation_stability=0.7,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test_det.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 10,
            "multiscale_count": 14,
            "union_candidate_count": 24,
            "duplicate_line_family_rate": 0.42,
            "dominant_angle_family_mass": 0.93,
            "dominant_angle_family_deg": 91.38,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (10, 12)
    assert result.support_breakdown["perturbation_count_consensus"] == [11, 12]


def test_select_zero_shot_result_skips_perturbation_consensus_when_angle_mass_is_not_strong():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.82,
            estimated_bend_count=5,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.2],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.9,
            structural_compactness=0.8,
            perturbation_stability=0.6,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
            support_breakdown={
                "perturbation": {
                    "counts": [5, 5, 8],
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.45,
            estimated_bend_count=7,
            estimated_bend_count_range=(6, 8),
            dominant_angle_families_deg=[120.0],
            exact_count_eligible=False,
            angle_family_coherence=0.7,
            duplicate_family_suppression=0.8,
            structural_compactness=0.7,
            perturbation_stability=0.6,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/test_weak_mass.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [10, 10, 10],
            "aspect_ratios": [1, 1, 1],
            "single_scale_count": 5,
            "multiscale_count": 8,
            "union_candidate_count": 13,
            "duplicate_line_family_rate": 0.2,
            "dominant_angle_family_mass": 0.76,
            "dominant_angle_family_deg": 90.2,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count_range == (4, 6)
    assert result.support_breakdown["perturbation_count_consensus"] is None


def test_select_zero_shot_result_uses_small_bend_rescue_range_on_sparse_conservative_winner():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.6965,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 4),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.4111,
            abstain_reasons=["policy_confidence_low", "dominant_angle_family_weak", "perturbation_count_unstable"],
            support_breakdown={"retained_candidate_count": 3},
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.3273,
            estimated_bend_count=4,
            estimated_bend_count_range=(6, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.4,
            structural_compactness=0.5,
            perturbation_stability=0.4,
            support_breakdown={"guard_ok": False},
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4067,
            estimated_bend_count=6,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "counts": [6, 4, 6],
                    "count_span": 2,
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7279,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.6721,
            estimated_bend_count=6,
            estimated_bend_count_range=(5, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.85,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "counts": [6, 5, 6],
                    "count_span": 1,
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7962,
                }
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/compact_fixture.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [190.0, 129.0, 153.0],
            "aspect_ratios": [1.47, 1.24, 0.84],
            "single_scale_count": 1,
            "multiscale_count": 7,
            "union_candidate_count": 8,
            "duplicate_line_family_rate": 0.125,
            "dominant_angle_family_mass": 0.625,
            "dominant_angle_family_deg": 90.491,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (5, 6)
    assert result.support_breakdown["small_bend_rescue_range"] == [5, 6]
    assert result.support_breakdown["small_bend_rescue_mode"] == "structural_overlap"
    assert result.support_breakdown["small_bend_rescue_policies"] == [
        "repeated_90_sheetmetal",
        "rolled_or_mixed_transition",
    ]


def test_select_zero_shot_result_skips_small_bend_rescue_without_multiscale_gap():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.6965,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 4),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.4111,
            abstain_reasons=["policy_confidence_low", "dominant_angle_family_weak", "perturbation_count_unstable"],
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4067,
            estimated_bend_count=6,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.45,
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.6721,
            estimated_bend_count=6,
            estimated_bend_count_range=(5, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.85,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.45,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/compact_fixture.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [190.0, 129.0, 153.0],
            "aspect_ratios": [1.47, 1.24, 0.84],
            "single_scale_count": 4,
            "multiscale_count": 7,
            "union_candidate_count": 8,
            "duplicate_line_family_rate": 0.125,
            "dominant_angle_family_mass": 0.625,
            "dominant_angle_family_deg": 90.491,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.estimated_bend_count_range == (1, 6)
    assert result.support_breakdown["small_bend_rescue_range"] is None


def test_select_zero_shot_result_small_bend_rescue_falls_back_to_union_without_identity_agreement():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.6965,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 4),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.4111,
            abstain_reasons=["policy_confidence_low", "dominant_angle_family_weak", "perturbation_count_unstable"],
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4067,
            estimated_bend_count=6,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.55,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.6721,
            estimated_bend_count=6,
            estimated_bend_count_range=(5, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.85,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.60,
                }
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/compact_fixture_union.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [190.0, 129.0, 153.0],
            "aspect_ratios": [1.47, 1.24, 0.84],
            "single_scale_count": 1,
            "multiscale_count": 7,
            "union_candidate_count": 8,
            "duplicate_line_family_rate": 0.125,
            "dominant_angle_family_mass": 0.625,
            "dominant_angle_family_deg": 90.491,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.estimated_bend_count_range == (4, 6)
    assert result.support_breakdown["small_bend_rescue_mode"] == "union"


def test_select_zero_shot_result_promotes_exact_count_for_sparse_small_bend_modal_match():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.6965,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 4),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.4111,
            abstain_reasons=["policy_confidence_low", "dominant_angle_family_weak", "perturbation_count_unstable"],
            support_breakdown={"retained_candidate_count": 1},
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4067,
            estimated_bend_count=6,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "counts": [6, 4, 6],
                    "count_span": 2,
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7279,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.6721,
            estimated_bend_count=6,
            estimated_bend_count_range=(5, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.85,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "counts": [6, 5, 6],
                    "count_span": 1,
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7962,
                }
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/compact_fixture_exact.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [190.0, 129.0, 153.0],
            "aspect_ratios": [1.47, 1.24, 0.84],
            "single_scale_count": 1,
            "multiscale_count": 7,
            "union_candidate_count": 8,
            "duplicate_line_family_rate": 0.125,
            "dominant_angle_family_mass": 0.625,
            "dominant_angle_family_deg": 90.491,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count == 6
    assert result.estimated_bend_count_range == (6, 6)
    assert result.support_breakdown["small_bend_exact_recovery"]["mode"] == "structural_overlap_modal_match"


def test_select_zero_shot_result_small_bend_exact_recovery_skips_partial_scans():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.6965,
            estimated_bend_count=1,
            estimated_bend_count_range=(1, 4),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.4111,
            abstain_reasons=["policy_confidence_low", "dominant_angle_family_weak", "perturbation_count_unstable"],
            support_breakdown={"retained_candidate_count": 1},
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4067,
            estimated_bend_count=6,
            estimated_bend_count_range=(4, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.9,
            duplicate_family_suppression=0.45,
            structural_compactness=0.6,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7279,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.6721,
            estimated_bend_count=6,
            estimated_bend_count_range=(5, 6),
            dominant_angle_families_deg=[90.491],
            exact_count_eligible=False,
            angle_family_coherence=0.85,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.45,
            support_breakdown={
                "perturbation": {
                    "modal_count": 6,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7962,
                }
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/compact_fixture_partial.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [190.0, 129.0, 153.0],
            "aspect_ratios": [1.47, 1.24, 0.84],
            "single_scale_count": 1,
            "multiscale_count": 7,
            "union_candidate_count": 8,
            "duplicate_line_family_rate": 0.125,
            "dominant_angle_family_mass": 0.625,
            "dominant_angle_family_deg": 90.491,
            "scan_state_hint": "partial",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.estimated_bend_count is None


def test_select_zero_shot_result_uses_mixed_lower_bound_modal_bridge():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.7709,
            estimated_bend_count=4,
            estimated_bend_count_range=(3, 4),
            dominant_angle_families_deg=[115.1, 94.456, 136.399],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.2829,
            abstain_reasons=["dominant_angle_family_weak"],
            support_breakdown={
                "perturbation": {
                    "counts": [4, 3, 3],
                    "modal_count": 3,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.3486,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.8433,
            estimated_bend_count=4,
            estimated_bend_count_range=(4, 5),
            dominant_angle_families_deg=[116.403, 90.251, 138.825],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.773,
            abstain_reasons=["dominant_angle_family_weak"],
            support_breakdown={
                "perturbation": {
                    "counts": [4, 5, 4],
                    "modal_count": 4,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.7730,
                }
            },
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4674,
            estimated_bend_count=5,
            estimated_bend_count_range=(4, 5),
            dominant_angle_families_deg=[115.882],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.45,
            structural_compactness=0.7,
            perturbation_stability=0.7016,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/mixed_fixture.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [317.0, 284.0, 160.0],
            "aspect_ratios": [1.11, 1.98, 1.77],
            "single_scale_count": 4,
            "multiscale_count": 5,
            "union_candidate_count": 9,
            "duplicate_line_family_rate": 0.3333,
            "dominant_angle_family_mass": 0.5556,
            "dominant_angle_family_deg": 116.403,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (3, 4)
    assert result.support_breakdown["mixed_lower_bound_rescue"]["mode"] == "mixed_lower_bound_modal_bridge"


def test_select_zero_shot_result_skips_mixed_lower_bound_modal_bridge_without_winner_identity():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.7709,
            estimated_bend_count=4,
            estimated_bend_count_range=(3, 4),
            dominant_angle_families_deg=[115.1, 94.456, 136.399],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.2829,
            support_breakdown={
                "perturbation": {
                    "modal_count": 3,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.3486,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.8433,
            estimated_bend_count=4,
            estimated_bend_count_range=(4, 5),
            dominant_angle_families_deg=[116.403, 90.251, 138.825],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.773,
            support_breakdown={
                "perturbation": {
                    "modal_count": 4,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.60,
                }
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/mixed_fixture_skip.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [317.0, 284.0, 160.0],
            "aspect_ratios": [1.11, 1.98, 1.77],
            "single_scale_count": 4,
            "multiscale_count": 5,
            "union_candidate_count": 9,
            "duplicate_line_family_rate": 0.3333,
            "dominant_angle_family_mass": 0.5556,
            "dominant_angle_family_deg": 116.403,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.estimated_bend_count_range == (3, 5)
    assert result.support_breakdown["mixed_lower_bound_rescue"] is None


def test_select_zero_shot_result_uses_near_orthogonal_family_bridge_range():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.7192,
            estimated_bend_count=7,
            estimated_bend_count_range=(5, 11),
            dominant_angle_families_deg=[90.702],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.5244,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.5861,
            estimated_bend_count=10,
            estimated_bend_count_range=(7, 12),
            dominant_angle_families_deg=[91.836],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.35,
            structural_compactness=0.6,
            perturbation_stability=0.5699,
            support_breakdown={
                "perturbation": {
                    "modal_count": 7,
                    "modal_count_support": 1.0 / 3.0,
                    "identity_stability": 0.5699,
                    "matched_reference_fraction_mean": 0.7,
                }
            },
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4697,
            estimated_bend_count=10,
            estimated_bend_count_range=(7, 10),
            dominant_angle_families_deg=[91.836],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.64,
            perturbation_stability=0.6253,
            support_breakdown={
                "perturbation": {
                    "modal_count": 7,
                    "modal_count_support": 1.0 / 3.0,
                    "identity_stability": 0.6253,
                    "matched_reference_fraction_mean": 0.75,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.7056,
            estimated_bend_count=13,
            estimated_bend_count_range=(7, 13),
            dominant_angle_families_deg=[92.558, 163.377],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.8,
            structural_compactness=0.75,
            perturbation_stability=0.515,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/repeated_90_bridge.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [264.0, 251.0, 134.0],
            "aspect_ratios": [1.05, 1.97, 1.88],
            "single_scale_count": 7,
            "multiscale_count": 13,
            "union_candidate_count": 20,
            "duplicate_line_family_rate": 0.35,
            "dominant_angle_family_mass": 0.9,
            "dominant_angle_family_deg": 90.702,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is True
    assert result.estimated_bend_count_range == (10, 12)
    assert result.support_breakdown["near_orthogonal_range_rescue"]["mode"] == "near_orthogonal_family_bridge"


def test_select_zero_shot_result_skips_near_orthogonal_family_bridge_without_match_coverage():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.7192,
            estimated_bend_count=7,
            estimated_bend_count_range=(5, 11),
            dominant_angle_families_deg=[90.702],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.5244,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.5861,
            estimated_bend_count=10,
            estimated_bend_count_range=(7, 12),
            dominant_angle_families_deg=[91.836],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.35,
            structural_compactness=0.6,
            perturbation_stability=0.5699,
            support_breakdown={
                "perturbation": {
                    "modal_count": 7,
                    "modal_count_support": 1.0 / 3.0,
                    "identity_stability": 0.5699,
                    "matched_reference_fraction_mean": 0.4,
                }
            },
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.4697,
            estimated_bend_count=10,
            estimated_bend_count_range=(7, 10),
            dominant_angle_families_deg=[91.836],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.64,
            perturbation_stability=0.6253,
            support_breakdown={
                "perturbation": {
                    "modal_count": 7,
                    "modal_count_support": 1.0 / 3.0,
                    "identity_stability": 0.6253,
                    "matched_reference_fraction_mean": 0.45,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.7056,
            estimated_bend_count=13,
            estimated_bend_count_range=(7, 13),
            dominant_angle_families_deg=[92.558, 163.377],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.8,
            structural_compactness=0.75,
            perturbation_stability=0.515,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/repeated_90_bridge_skip.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [264.0, 251.0, 134.0],
            "aspect_ratios": [1.05, 1.97, 1.88],
            "single_scale_count": 7,
            "multiscale_count": 13,
            "union_candidate_count": 20,
            "duplicate_line_family_rate": 0.35,
            "dominant_angle_family_mass": 0.9,
            "dominant_angle_family_deg": 90.702,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.estimated_bend_count_range == (7, 13)
    assert result.support_breakdown["near_orthogonal_range_rescue"] is None


def test_select_zero_shot_result_promotes_exact_count_for_near_orthogonal_dropout_match():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.71,
            estimated_bend_count=7,
            estimated_bend_count_range=(5, 9),
            dominant_angle_families_deg=[90.7],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.57,
            perturbation_stability=0.6,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.62,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.35,
            structural_compactness=0.62,
            perturbation_stability=0.72,
            support_breakdown={
                "perturbation": {
                    "counts": [10, 9, 10],
                    "count_span": 1,
                    "modal_count": 10,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.84,
                    "matched_reference_fraction_mean": 0.9,
                }
            },
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.51,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.66,
            perturbation_stability=0.71,
            support_breakdown={
                "perturbation": {
                    "counts": [10, 9, 10],
                    "count_span": 1,
                    "modal_count": 10,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.86,
                    "matched_reference_fraction_mean": 0.92,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.60,
            estimated_bend_count=11,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.4, 160.0],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.8,
            structural_compactness=0.72,
            perturbation_stability=0.55,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/near_orthogonal_exact.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [264.0, 251.0, 134.0],
            "aspect_ratios": [1.05, 1.97, 1.88],
            "single_scale_count": 7,
            "multiscale_count": 13,
            "union_candidate_count": 20,
            "duplicate_line_family_rate": 0.35,
            "dominant_angle_family_mass": 0.9,
            "dominant_angle_family_deg": 90.7,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count == 10
    assert result.estimated_bend_count_range == (10, 10)
    assert result.support_breakdown["near_orthogonal_exact_recovery"]["mode"] == "near_orthogonal_dropout_match"


def test_select_zero_shot_result_skips_near_orthogonal_exact_when_dropout_too_large():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.71,
            estimated_bend_count=7,
            estimated_bend_count_range=(5, 9),
            dominant_angle_families_deg=[90.7],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.57,
            perturbation_stability=0.6,
            abstain_reasons=["policy_confidence_low", "perturbation_count_unstable"],
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.62,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.35,
            structural_compactness=0.62,
            perturbation_stability=0.72,
            support_breakdown={
                "perturbation": {
                    "counts": [10, 8, 10],
                    "count_span": 2,
                    "modal_count": 10,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.84,
                    "matched_reference_fraction_mean": 0.9,
                }
            },
        ),
        "repeated_90_sheetmetal": zsbp.PolicyScore(
            score=0.51,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.2],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.45,
            structural_compactness=0.66,
            perturbation_stability=0.71,
            support_breakdown={
                "perturbation": {
                    "counts": [10, 8, 10],
                    "count_span": 2,
                    "modal_count": 10,
                    "modal_count_support": 2.0 / 3.0,
                    "identity_stability": 0.86,
                    "matched_reference_fraction_mean": 0.92,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.60,
            estimated_bend_count=11,
            estimated_bend_count_range=(10, 10),
            dominant_angle_families_deg=[91.4, 160.0],
            exact_count_eligible=False,
            angle_family_coherence=0.8,
            duplicate_family_suppression=0.8,
            structural_compactness=0.72,
            perturbation_stability=0.55,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/near_orthogonal_exact_skip.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [264.0, 251.0, 134.0],
            "aspect_ratios": [1.05, 1.97, 1.88],
            "single_scale_count": 7,
            "multiscale_count": 13,
            "union_candidate_count": 20,
            "duplicate_line_family_rate": 0.35,
            "dominant_angle_family_mass": 0.9,
            "dominant_angle_family_deg": 90.7,
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.estimated_bend_count is None
    assert result.support_breakdown["near_orthogonal_exact_recovery"] is None


def test_select_zero_shot_result_uses_rolled_structural_support_bridge():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.5168,
            estimated_bend_count=8,
            estimated_bend_count_range=(7, 10),
            dominant_angle_families_deg=[91.086, 107.939, 162.642],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.5124,
            support_breakdown={
                "perturbation": {
                    "matched_reference_fraction_mean": 0.6875,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.7117,
            estimated_bend_count=12,
            estimated_bend_count_range=(7, 14),
            dominant_angle_families_deg=[95.632, 125.38, 164.168],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.6259,
            support_breakdown={
                "perturbation": {
                    "identity_stability": 0.6259,
                    "matched_reference_fraction_mean": 0.75,
                    "reference_strip_count": 12,
                },
                "bend_support_strips": [{} for _ in range(12)],
            },
        ),
        "orthogonal_dense_sheetmetal": zsbp.PolicyScore(
            score=0.174,
            estimated_bend_count=6,
            estimated_bend_count_range=(7, 13),
            dominant_angle_families_deg=[93.715],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.35,
            structural_compactness=0.6,
            perturbation_stability=0.6022,
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/rolled_support_bridge.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [148.0, 146.0, 80.0],
            "aspect_ratios": [1.01, 1.85, 1.82],
            "single_scale_count": 8,
            "multiscale_count": 13,
            "union_candidate_count": 21,
            "duplicate_line_family_rate": 0.4762,
            "dominant_angle_family_mass": 0.7143,
            "dominant_angle_family_deg": 95.632,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (8, 12)
    assert result.support_breakdown["rolled_structural_range_rescue"]["mode"] == "rolled_structural_support_bridge"


def test_select_zero_shot_result_skips_rolled_structural_support_bridge_without_match_support():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.5168,
            estimated_bend_count=8,
            estimated_bend_count_range=(7, 10),
            dominant_angle_families_deg=[91.086, 107.939, 162.642],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.5714,
            perturbation_stability=0.5124,
            support_breakdown={
                "perturbation": {
                    "matched_reference_fraction_mean": 0.5,
                }
            },
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.7117,
            estimated_bend_count=12,
            estimated_bend_count_range=(7, 14),
            dominant_angle_families_deg=[95.632, 125.38, 164.168],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.8,
            structural_compactness=0.8,
            perturbation_stability=0.6259,
            support_breakdown={
                "perturbation": {
                    "identity_stability": 0.6259,
                    "matched_reference_fraction_mean": 0.75,
                    "reference_strip_count": 12,
                },
                "bend_support_strips": [{} for _ in range(12)],
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/rolled_support_bridge_skip.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [148.0, 146.0, 80.0],
            "aspect_ratios": [1.01, 1.85, 1.82],
            "single_scale_count": 8,
            "multiscale_count": 13,
            "union_candidate_count": 21,
            "duplicate_line_family_rate": 0.4762,
            "dominant_angle_family_mass": 0.7143,
            "dominant_angle_family_deg": 95.632,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.estimated_bend_count_range == (7, 14)
    assert result.support_breakdown["rolled_structural_range_rescue"] is None


def test_select_zero_shot_result_uses_rolled_structural_consensus_bridge():
    policy_scores = {
        "conservative_macro": zsbp.PolicyScore(
            score=0.4917,
            estimated_bend_count=10,
            estimated_bend_count_range=(10, 12),
            dominant_angle_families_deg=[91.379, 125.0],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=1.0,
            structural_compactness=0.7,
            perturbation_stability=0.55,
            support_breakdown={"perturbation": {"matched_reference_fraction_mean": 0.8}},
        ),
        "rolled_or_mixed_transition": zsbp.PolicyScore(
            score=0.7326,
            estimated_bend_count=11,
            estimated_bend_count_range=(11, 14),
            dominant_angle_families_deg=[91.379, 125.0],
            exact_count_eligible=False,
            angle_family_coherence=1.0,
            duplicate_family_suppression=0.7857,
            structural_compactness=0.8776,
            perturbation_stability=0.5536,
            support_breakdown={
                "perturbation": {
                    "identity_stability": 0.723,
                    "matched_reference_fraction_mean": 0.9545,
                    "reference_strip_count": 11,
                    "counts": [11, 14, 12],
                },
                "bend_support_strips": [{} for _ in range(11)],
            },
        ),
    }

    result = zsbp._select_zero_shot_result(
        scan_path="/tmp/rolled_support_consensus_bridge.ply",
        part_id="P",
        policy_scores=policy_scores,
        scan_features={
            "point_count": 100,
            "bbox_mm": [220.0, 220.0, 90.0],
            "aspect_ratios": [1.0, 2.2, 2.0],
            "single_scale_count": 9,
            "multiscale_count": 14,
            "union_candidate_count": 23,
            "duplicate_line_family_rate": 0.3,
            "dominant_angle_family_mass": 0.82,
            "dominant_angle_family_deg": 91.379,
            "scan_state_hint": "full",
        },
        candidate_bundles={"original": {"single_scale": [], "multiscale": []}},
    )

    assert result.abstained is False
    assert result.estimated_bend_count is None
    assert result.estimated_bend_count_range == (11, 12)
    assert result.support_breakdown["rolled_structural_range_rescue"]["mode"] == "rolled_structural_consensus_bridge"
    assert result.support_breakdown["rolled_structural_range_rescue"]["winner_consensus_range"] == [11, 12]


def test_select_renderable_support_strips_caps_exact_count():
    strips = [
        zsbp.BendSupportStrip(
            strip_id=f"S{index}",
            strip_family="discrete_fold_strip",
            axis_direction=[1.0, 0.0, 0.0],
            axis_line_hint={"midpoint": [float(index * 30), 0.0, 0.0]},
            adjacent_flange_normals=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dominant_angle_deg=90.0,
            radius_mm=None,
            support_patch_ids=[f"P{index}"],
            support_point_count=100 - index,
            cross_fold_width_mm=6.0,
            along_axis_span_mm=40.0,
            support_bbox_mm=[1.0, 1.0, 1.0],
            observability_status="supported",
            merge_provenance={"merged_atom_ids": [f"A{index}"]},
            stability_score=1.0,
            confidence=1.0 - (index * 0.01),
        )
        for index in range(1, 15)
    ]

    selected = zsbp._select_renderable_support_strips(
        strips,
        target_count=10,
        spacing_mm=1.0,
    )

    assert len(selected) == 10
    assert [strip.strip_id for strip in selected] == [f"S{index}" for index in range(1, 11)]


def test_build_evidence_atoms_trace_informed_experiment_fields():
    bends = [
        SimpleNamespace(
            bend_id="B1",
            measured_angle=90.0,
            measured_radius=1.0,
            bend_line_start=np.asarray([0.0, 0.0, 0.0]),
            bend_line_end=np.asarray([20.0, 0.0, 0.0]),
            bend_line_direction=np.asarray([1.0, 0.0, 0.0]),
            bend_line_length=20.0,
            confidence=0.9,
            radius_fit_error=0.0,
            flange1=SimpleNamespace(normal=np.asarray([0.0, 0.0, 1.0]), area=100.0),
            flange2=SimpleNamespace(normal=np.asarray([0.0, 1.0, 0.0]), area=95.0),
        ),
        SimpleNamespace(
            bend_id="B2",
            measured_angle=90.0,
            measured_radius=1.0,
            bend_line_start=np.asarray([22.0, 0.0, 0.0]),
            bend_line_end=np.asarray([42.0, 0.0, 0.0]),
            bend_line_direction=np.asarray([1.0, 0.0, 0.0]),
            bend_line_length=20.0,
            confidence=0.92,
            radius_fit_error=0.0,
            flange1=SimpleNamespace(normal=np.asarray([0.0, 0.0, 1.0]), area=102.0),
            flange2=SimpleNamespace(normal=np.asarray([0.0, 1.0, 0.0]), area=93.0),
        ),
    ]
    trace_graph = zsbp._build_candidate_trace_graph(bends, spacing_mm=1.0)
    atoms = zsbp._build_evidence_atoms(
        bends,
        line_groups=[bends],
        source_mode="original:single_scale",
        spacing_mm=1.0,
        strip_family="discrete_fold_strip",
        trace_graph=trace_graph,
        experiment_config={"flags": {"trace_informed_atoms": True}},
    )

    assert len(atoms) == 2
    assert atoms[0].trace_endpoints
    assert atoms[0].trace_length_mm == 20.0
    assert atoms[0].fold_strength > 0.0
    assert atoms[0].connected_component_size == 2


def test_same_support_strip_trace_aware_rejects_large_gap_without_support():
    lhs = zsbp.BendEvidenceAtom(
        atom_id="A1",
        source_mode="x",
        candidate_bend_id="B1",
        local_axis_direction=[1.0, 0.0, 0.0],
        local_line_family_id="F1",
        local_flange_compatibility={"flange_normal_delta_deg": 90.0},
        local_spacing_mm=1.0,
        local_support_patch_id="P1",
        local_support_span_mm=20.0,
        support_patch_midpoint=[0.0, 0.0, 0.0],
        support_bbox_mm=[1.0, 1.0, 1.0],
        local_normal_summary={},
        local_curvature_summary={},
        null_assignment_eligible=True,
        observability_flags=[],
        ambiguity_flags=["low_confidence_candidate"],
        trace_endpoints=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        trace_length_mm=20.0,
        trace_continuity_score=0.2,
        fold_strength=0.2,
        branch_degree=0,
        connected_component_size=1,
    )
    rhs = zsbp.BendEvidenceAtom(
        atom_id="A2",
        source_mode="x",
        candidate_bend_id="B2",
        local_axis_direction=[1.0, 0.0, 0.0],
        local_line_family_id="F1",
        local_flange_compatibility={"flange_normal_delta_deg": 90.0},
        local_spacing_mm=1.0,
        local_support_patch_id="P2",
        local_support_span_mm=20.0,
        support_patch_midpoint=[35.0, 0.0, 0.0],
        support_bbox_mm=[1.0, 1.0, 1.0],
        local_normal_summary={},
        local_curvature_summary={},
        null_assignment_eligible=True,
        observability_flags=[],
        ambiguity_flags=["low_confidence_candidate"],
        trace_endpoints=[[35.0, 0.0, 0.0], [55.0, 0.0, 0.0]],
        trace_length_mm=20.0,
        trace_continuity_score=0.2,
        fold_strength=0.2,
        branch_degree=0,
        connected_component_size=1,
    )

    assert zsbp._same_support_strip(lhs, rhs, spacing_mm=1.0, trace_aware=False) is True
    assert zsbp._same_support_strip(lhs, rhs, spacing_mm=1.0, trace_aware=True) is False
