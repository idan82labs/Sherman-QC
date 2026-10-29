import json
import math

import pytest

from scripts.run_shaked_selector_evaluation import (
    MESH_FEATURE_NAMES,
    _complete_short_mesh,
    _load_shaked_gold,
    _mesh_case_features,
    _prediction_case_payload,
    _select,
    _validate_candidate_payload,
    validate_split,
)


def _split():
    return {
        "development_case_ids": ["01_alpha", "02_beta"],
        "locked_case_ids": ["03_gamma"],
        "selection_rule": {
            "kind": "sha256_rank",
            "salt": "test",
            "locked_count": 1,
        },
    }


def test_validate_split_enforces_declared_sha_ranking():
    split = _split()
    case_ids = split["development_case_ids"] + split["locked_case_ids"]
    # Assign the actual lowest digest to locked while retaining the same set.
    import hashlib

    ranked = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(f"test:{case_id}".encode()).hexdigest(),
    )
    split["locked_case_ids"] = ranked[:1]
    split["development_case_ids"] = ranked[1:]

    validate_split(split, case_ids)

    split["locked_case_ids"], split["development_case_ids"][0] = (
        [split["development_case_ids"][0]],
        split["locked_case_ids"][0],
    )
    with pytest.raises(ValueError, match="SHA-256"):
        validate_split(split, case_ids)


def test_shaked_gold_loader_reads_only_explicit_case_ids(tmp_path):
    requested = tmp_path / "01_requested"
    requested.mkdir()
    (requested / "bend_instance_labels.json").write_text(
        json.dumps(
            {
                "case_id": "01_requested",
                "authoritative_ground_truth": False,
                "review_status": "assistant_multiview_complete_user_review_pending",
                "manual_bends": [
                    {
                        "bend_id": "M1",
                        "review_decision": "accepted_user_review_pending",
                    }
                ],
            }
        )
    )
    locked = tmp_path / "02_locked"
    locked.mkdir()
    (locked / "bend_instance_labels.json").write_text("this must not be opened")

    gold, hashes = _load_shaked_gold(tmp_path, ["01_requested"])

    assert [item["bend_id"] for item in gold["01_requested"]] == ["M1"]
    assert set(hashes) == {"01_requested"}


def test_candidate_contract_fails_closed_on_truth_or_points():
    payload = {
        "coordinate_frame": "scan_raw",
        "source_point_serialization": False,
        "inference_contract": {
            "part_identifier_exposed": False,
            "expected_count_exposed": False,
            "manual_truth_exposed": False,
            "cad_exposed": False,
        },
    }
    _validate_candidate_payload(payload, "case")

    payload["inference_contract"]["expected_count_exposed"] = True
    with pytest.raises(ValueError, match="blind candidate contract"):
        _validate_candidate_payload(payload, "case")


def test_prediction_payload_contains_no_evaluation_metrics_or_truth():
    payload = _prediction_case_payload(
        "08_locked",
        [{"candidate_id": "C1", "span_endpoints": [[0, 0, 0], [10, 0, 0]]}],
        "adapted_holdout",
        {
            "model_kind": "extra_trees",
            "probability_threshold": 0.5,
            "cylindrical_offset_cap_mm": 12.0,
            "duplicate_offset_tolerance_mm": 5.0,
            "two_sided_min_support_gate": 0.3,
        },
        ["01_training"],
    )

    assert payload["evaluation_labels_read"] is False
    assert payload["training_case_ids"] == ["01_training"]
    assert "metrics" not in payload
    assert "expected_count" not in json.dumps(payload)


def test_mesh_features_are_identity_free_and_case_relative():
    candidates = [
        {
            "bend_angle": 90.0,
            "surface1_area": 100.0,
            "surface2_area": 50.0,
            "span_mm": 20.0,
            "two_sided_transition_score": 0.8,
        },
        {
            "bend_angle": 45.0,
            "surface1_area": 25.0,
            "surface2_area": 10.0,
            "span_mm": 10.0,
            "two_sided_transition_score": 0.2,
        },
    ]

    values = _mesh_case_features(candidates, {"scan_bbox_mm": [100, 100, 20]})

    assert len(values) == 2
    assert len(values[0]) == len(MESH_FEATURE_NAMES)
    assert values[0][MESH_FEATURE_NAMES.index("mesh_span_over_diagonal")] > values[1][
        MESH_FEATURE_NAMES.index("mesh_span_over_diagonal")
    ]
    assert not any("case_id" in name or "count" in name for name in MESH_FEATURE_NAMES)
    assert math.isclose(
        values[0][MESH_FEATURE_NAMES.index("mesh_log_minimum_surface_area_span_ratio")],
        math.log1p(50.0 / (20.0 * 20.0)),
    )


def test_mesh_surface_gap_cap_filters_only_mesh_candidates():
    mesh = {
        "candidate_id": "M",
        "candidate_source": "scan_mesh_adjacency",
        "min_surface_gap": 20.0,
        "span_endpoints": [[0, 0, 0], [10, 0, 0]],
    }
    non_mesh = {
        "candidate_id": "N",
        "candidate_source": "strict_blind_owned",
        "min_surface_gap": 20.0,
        "span_endpoints": [[0, 20, 0], [10, 20, 0]],
    }
    config = {
        "probability_threshold": 0.5,
        "cylindrical_offset_cap_mm": 1e9,
        "duplicate_offset_tolerance_mm": 4.0,
        "two_sided_min_support_gate": 0.0,
        "mesh_max_surface_gap_mm": 10.0,
    }

    selected = _select(([mesh, non_mesh], [0.9, 0.9], [0.9, 0.9]), config)

    assert [item["candidate_id"] for item in selected] == ["N"]


def test_completion_grid_config_uses_default_duplicate_tolerance():
    candidate = {
        "bend_id": "LO1",
        "candidate_id": "scan_mesh_adjacency::SMA1",
        "span_endpoints": [[0, 0, 0], [10, 0, 0]],
        "cluster_sources": ["scan_mesh_adjacency"],
        "cluster_max_two_sided_normal_angle_deg": 70.0,
        "cluster_max_finite_support_fraction": 0.95,
    }
    result = _complete_short_mesh(
        [],
        [candidate],
        {
            "short_mesh_max_span_mm": 16.0,
            "short_mesh_min_normal_turn_deg": 60.0,
            "short_mesh_min_finite_support_fraction": 0.8,
        },
    )
    assert result["added_ids"] == ["LO1"]
