from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_zero_shot_repeated_near_90 import analyze_zero_shot_repeated_near_90


def test_analyze_zero_shot_repeated_near_90_filters_by_expected_angle_and_extracts_policy_spans(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"aggregate": {}, "results": []}), encoding="utf-8")

    repeated_payload = {
        "part_key": "Y",
        "scan_name": "ysherman.ply",
        "scan_path": "/tmp/ysherman.ply",
        "expectation": {
            "expected_bend_count": 12,
            "dominant_angle_families_deg": [90.0],
        },
        "result": {
            "profile_family": "conservative_macro",
            "estimated_bend_count": None,
            "estimated_bend_count_range": [10, 12],
            "abstained": True,
            "abstain_reasons": ["perturbation_count_unstable"],
            "dominant_angle_families_deg": [90.7],
            "candidate_generator_summary": {
                "single_scale_count": 7,
                "multiscale_count": 13,
                "union_candidate_count": 20,
                "duplicate_line_family_rate": 0.35,
                "strip_duplicate_merge_count": 7,
                "perturbation_variants": {
                    "original": {"single_scale_count": 7, "multiscale_count": 13},
                    "voxel_0.8": {"single_scale_count": 5, "multiscale_count": 7},
                },
            },
            "policy_scores": {
                "conservative_macro": {
                    "effective_structural_count": 7,
                    "structural_count_range": [10, 12],
                    "dominant_angle_deg": 90.7,
                    "dominant_angle_mass": 1.0,
                    "score": 0.72,
                    "perturbation_summary": {
                        "reference_strip_count": 10,
                        "mean_matched_reference_fraction": 0.65,
                        "identity_stability": 0.61,
                        "modal_support": 0.67,
                        "variant_summaries": {
                            "original": {
                                "effective_count": 10,
                                "raw_candidate_count": 20,
                                "line_family_group_count": 7,
                                "strip_duplicate_merge_count": 7,
                            },
                            "voxel_0.8": {
                                "effective_count": 7,
                                "raw_candidate_count": 12,
                                "line_family_group_count": 5,
                                "strip_duplicate_merge_count": 4,
                            },
                            "voxel_1.2": {
                                "effective_count": 12,
                                "raw_candidate_count": 19,
                                "line_family_group_count": 8,
                                "strip_duplicate_merge_count": 6,
                            },
                        },
                    },
                }
            },
        },
    }
    non_repeated_payload = {
        "part_key": "A",
        "scan_name": "angled.ply",
        "scan_path": "/tmp/angled.ply",
        "expectation": {
            "expected_bend_count": 1,
            "dominant_angle_families_deg": [45.0],
        },
        "result": {
            "profile_family": "rolled_or_mixed_transition",
            "estimated_bend_count_range": [1, 1],
            "abstained": False,
            "policy_scores": {},
        },
    }
    (tmp_path / "ysherman.ply--aaa.json").write_text(json.dumps(repeated_payload), encoding="utf-8")
    (tmp_path / "angled.ply--bbb.json").write_text(json.dumps(non_repeated_payload), encoding="utf-8")

    analysis = analyze_zero_shot_repeated_near_90(summary_path)

    assert analysis["repeated_scan_count"] == 1
    row = analysis["repeated_scans"][0]
    assert row["part_key"] == "Y"
    assert row["selected_count_range"] == [10, 12]
    assert row["candidate_generator_summary"]["union_candidate_count"] == 20
    policy = row["policy_rows"][0]
    assert policy["policy_name"] == "conservative_macro"
    assert policy["effective_counts"] == {"original": 10, "voxel_0.8": 7, "voxel_1.2": 12}
    assert policy["effective_count_span"] == 5
    assert policy["raw_candidate_count_span"] == 8
