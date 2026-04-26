import json

import numpy as np

from backend.feature_detection.bend_detector import BendSpecification
from domains.bend.services.part_profiling import (
    build_cad_part_profile,
    build_scan_part_profile,
    build_scan_structural_hypothesis,
)


def _make_spec(
    bend_id: str,
    *,
    angle: float = 90.0,
    bend_form: str = "FOLDED",
    countable: bool = True,
) -> BendSpecification:
    return BendSpecification(
        bend_id=bend_id,
        target_angle=angle,
        target_radius=1.0,
        bend_line_start=np.array([0.0, 0.0, 0.0]),
        bend_line_end=np.array([0.0, 10.0, 0.0]),
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        bend_form=bend_form,
        countable_in_regression=countable,
    )


def test_build_cad_part_profile_prefers_expected_override_and_metadata_angles():
    profile = build_cad_part_profile(
        cad_bends=[
            _make_spec("CAD_B1", angle=90.0),
            _make_spec("CAD_B2", angle=45.0),
            _make_spec("CAD_B3", angle=90.0, countable=False),
        ],
        cad_path="part.stp",
        expected_bend_override=5,
        spec_bend_angles_hint=[90.0, 45.0],
        spec_schedule_type="complete_or_near_complete",
        metadata_truth={
            "expected_total_bends": 3,
            "countable_bend_target_angles_deg": {"CAD_B1": 90.0, "CAD_B2": 25.0},
        },
    )

    assert profile.profile_mode == "cad_nominal"
    assert profile.nominal_source_class == "semantic_nominal_cad"
    assert profile.profile_confidence == 1.0
    assert profile.cad_target is not None
    assert profile.cad_target.expected_bend_count == 5
    assert profile.cad_target.expected_bend_count_range == (5, 5)
    assert profile.cad_target.dominant_angle_families_deg == (25.0, 90.0)
    assert profile.cad_target.authority_level == "authoritative"
    assert profile.cad_target.provenance["expected_bend_count_source"] == "expected_bend_override"
    assert profile.cad_target.provenance["angle_family_source"] == "metadata_truth.countable_bend_target_angles_deg"


def test_build_cad_part_profile_normalizes_feature_policy_with_metadata_truth():
    profile = build_cad_part_profile(
        cad_bends=[
            _make_spec("CAD_B1", angle=140.0, bend_form="ROLLED"),
            _make_spec("CAD_B2", angle=25.0),
        ],
        cad_path="part.stl",
        feature_policy={
            "countable_source": "legacy_folded_hybrid",
            "non_counting_feature_forms": ["rolled"],
        },
        metadata_truth={
            "part_family": "rolled_shell_with_two_flange_bends",
            "countable_bend_target_angles_deg": {"CAD_B2": 25.0},
        },
    )

    assert profile.nominal_source_class == "mesh_nominal"
    assert profile.profile_confidence == 0.7
    assert profile.cad_target is not None
    assert profile.cad_target.authority_level == "geometric_prior"
    assert profile.cad_target.rolled_allowed is False
    assert profile.cad_target.countable_feature_policy["part_family"] == "rolled_shell_with_two_flange_bends"
    assert profile.cad_target.countable_feature_policy["countable_bend_angle_overrides"] == {"CAD_B2": 25.0}
    assert profile.cad_target.countable_feature_policy["non_counting_feature_forms"] == ["ROLLED"]


def test_build_cad_part_profile_uses_partial_schedule_for_expected_range():
    profile = build_cad_part_profile(
        cad_bends=[
            _make_spec("CAD_B1", angle=90.0),
            _make_spec("CAD_B2", angle=90.0),
            _make_spec("CAD_B3", angle=125.0),
            _make_spec("CAD_B4", angle=125.0),
        ],
        cad_path="part.step",
        spec_bend_angles_hint=[90.0, 125.0],
        spec_schedule_type="partial_angle_schedule",
    )

    assert profile.cad_target is not None
    assert profile.cad_target.expected_bend_count == 4
    assert profile.cad_target.expected_bend_count_range == (2, 4)
    assert profile.cad_target.provenance["expected_bend_count_source"] == "spec_schedule_type.partial_angle_schedule"
    assert profile.cad_target.dominant_angle_families_deg == (90.0, 125.0)


def test_part_profile_to_dict_is_json_safe():
    profile = build_cad_part_profile(
        cad_bends=[_make_spec("CAD_B1", angle=90.0)],
        cad_path="part.stp",
        feature_policy={"countable_bend_ids": ["CAD_B1"]},
    )

    payload = profile.to_dict()
    assert payload["cad_target"]["nominal_bend_set"] == ["CAD_B1"]
    assert payload["cad_target"]["expected_bend_count_range"] == [1, 1]
    assert payload["cad_target"]["countable_feature_policy"]["countable_bend_ids"] == ["CAD_B1"]
    json.dumps(payload)


def test_build_scan_structural_hypothesis_normalizes_zero_shot_payload():
    hypothesis = build_scan_structural_hypothesis(
        zero_shot_result={
            "profile_family": "orthogonal_dense_sheetmetal",
            "estimated_bend_count": 10,
            "estimated_bend_count_range": [9, 10],
            "dominant_angle_families_deg": [90.5],
            "hypothesis_confidence": 0.82,
            "abstained": False,
            "abstain_reasons": [],
            "strip_count_confidence": 0.79,
            "support_breakdown": {
                "line_family_groups": [["LF1", "LF2"], ["LF3"]],
                "strip_duplicate_merge_count": 2,
                "nested": [{"strip_id": "S1"}],
            },
            "bend_support_strips": [
                {"strip_id": "S1", "support_patch_ids": ("P1", "P2")},
            ],
            "renderable_bend_objects": [
                {"bend_id": "S1", "detail_segments": ("theta≈90.5", "supported")},
            ],
        }
    )

    assert hypothesis.profile_family == "orthogonal_dense_sheetmetal"
    assert hypothesis.estimated_bend_count == 10
    assert hypothesis.estimated_bend_count_range == (9, 10)
    assert hypothesis.dominant_angle_families_deg == (90.5,)
    assert hypothesis.line_family_groups == (("LF1", "LF2"), ("LF3",))
    assert hypothesis.strip_count_confidence == 0.79
    assert hypothesis.strip_duplicate_merge_count == 2
    assert hypothesis.bend_support_strips[0]["support_patch_ids"] == ["P1", "P2"]
    assert hypothesis.renderable_bend_objects[0]["detail_segments"] == ["theta≈90.5", "supported"]
    json.dumps(hypothesis.to_dict())


def test_build_scan_part_profile_accepts_to_dict_result_and_preserves_scan_hypothesis():
    class DummyResult:
        def to_dict(self):
            return {
                "scan_path": "/tmp/demo-scan.ply",
                "part_id": "P-123",
                "profile_family": "rolled_or_mixed_transition",
                "estimated_bend_count": None,
                "estimated_bend_count_range": [8, 12],
                "dominant_angle_families_deg": [91.4],
                "hypothesis_confidence": 0.61,
                "abstained": True,
                "abstain_reasons": ["perturbation_instability"],
                "support_breakdown": {
                    "line_family_groups": [["LF_A"]],
                    "strip_duplicate_merge_count": 1,
                },
                "bend_support_strips": [{"strip_id": "S7"}],
                "renderable_bend_objects": [{"bend_id": "S7", "status": "WARNING"}],
            }

    profile = build_scan_part_profile(zero_shot_result=DummyResult())

    assert profile.profile_mode == "scan_zero_shot"
    assert profile.nominal_source_class == "scan_only"
    assert profile.profile_confidence == 0.61
    assert profile.cad_target is None
    assert profile.scan_hypothesis is not None
    assert profile.scan_hypothesis.estimated_bend_count is None
    assert profile.scan_hypothesis.estimated_bend_count_range == (8, 12)
    assert profile.scan_hypothesis.abstained is True
    assert profile.scan_hypothesis.renderable_bend_objects[0]["bend_id"] == "S7"
    assert profile.provenance["scan_path"] == "/tmp/demo-scan.ply"
    assert profile.provenance["part_id"] == "P-123"
    json.dumps(profile.to_dict())
