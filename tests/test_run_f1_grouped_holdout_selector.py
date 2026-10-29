import json

import numpy as np

from scripts.run_f1_grouped_holdout_selector import (
    FEATURE_NAMES,
    SOURCES,
    _features,
    _endpoints,
    _point_segment_distance,
    _attach_arrangement_features,
    _duplicate,
    _load_additional_training_data,
    _overlay_two_sided_features,
    part_group,
    training_case_ids,
)
from scripts.evaluate_f1_against_manual_gold import pair_metrics


def test_part_group_keeps_all_scans_of_physical_part_together():
    assert part_group("01_10839000") == "10839000"
    assert part_group("03_10839000") == "10839000"
    assert part_group("13_ysherman") == "ysherman"


def test_training_cases_exclude_every_scan_of_held_out_part():
    cases = ["01_10839000", "02_10839000", "04_15651004", "06_38172000", "07_38172000"]

    assert training_case_ids(cases, "02_10839000") == [
        "04_15651004",
        "06_38172000",
        "07_38172000",
    ]


def test_duplicate_offset_tolerance_is_explicit_and_deterministic():
    left = {"span_endpoints": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]}
    right = {"span_endpoints": [[0.0, 5.5, 0.0], [100.0, 5.5, 0.0]]}

    assert not _duplicate(left, right, offset_tolerance_mm=5.0)
    assert _duplicate(left, right, offset_tolerance_mm=6.0)


def test_additional_training_loader_accepts_only_completed_accepted_axes(tmp_path):
    candidates = tmp_path / "candidates"
    labels = tmp_path / "labels" / "15_44211000"
    candidates.mkdir()
    labels.mkdir(parents=True)
    (candidates / "01_44211000.json").write_text(json.dumps({"line_objects": []}))
    (labels / "bend_instance_labels.json").write_text(
        json.dumps(
            {
                "job_id": "15_44211000",
                "authoritative_ground_truth": False,
                "review_status": "visual_review_complete_training_only",
                "manual_bends": [
                    {"bend_id": "P01", "review_decision": "accepted"},
                    {"bend_id": "P02", "review_decision": "rejected"},
                ],
            }
        )
    )

    payloads, gold = _load_additional_training_data(candidates, tmp_path / "labels")

    assert list(payloads) == ["01_44211000"]
    assert [bend["bend_id"] for bend in gold["01_44211000"]] == ["P01"]


def test_arrangement_features_capture_junctions_parallel_families_and_duplicates():
    candidates = [
        {"span_endpoints": [[0, 0, 0], [20, 0, 0]]},
        {"span_endpoints": [[20, 0, 0], [20, 20, 0]]},
        {"span_endpoints": [[0, 8, 0], [20, 8, 0]]},
        {"span_endpoints": [[0, 0.5, 0], [20, 0.5, 0]]},
    ]

    _attach_arrangement_features(candidates, {"scan_bbox_mm": [100, 100, 20]})

    assert candidates[0]["arrangement_orthogonal_junction_neighbor_count"] >= 1
    assert candidates[0]["arrangement_connected_endpoint_count"] >= 1
    assert candidates[0]["arrangement_parallel_family_count"] == 2
    assert candidates[0]["arrangement_near_duplicate_count"] == 1
    assert 0.0 < candidates[0]["case_percentile_arrangement_parallel_family_count"] <= 1.0


def test_two_sided_overlay_copies_only_diagnostic_fields_by_stable_id():
    candidates = [
        {
            "candidate_id": "C1",
            "score": 0.7,
            "span_mm": 20.0,
            "span_endpoints": [[0, 0, 0], [20, 0, 0]],
        }
    ]
    diagnostic = [
        {
            "candidate_id": "C1",
            "score": 0.1,
            "span_mm": 999.0,
            "span_endpoints": [[0, 0, 0], [999, 0, 0]],
            "cluster_max_two_sided_min_support_fraction": 0.4,
        }
    ]

    matched = _overlay_two_sided_features(candidates, diagnostic)

    assert matched == 1
    assert candidates[0]["score"] == 0.7
    assert candidates[0]["span_mm"] == 20.0
    assert candidates[0]["cluster_max_two_sided_min_support_fraction"] == 0.4


def test_two_sided_overlay_falls_back_only_to_identical_geometry():
    candidates = [{"candidate_id": "old", "span_endpoints": [[20, 0, 0], [0, 0, 0]]}]
    diagnostic = [
        {
            "candidate_id": "new",
            "span_endpoints": [[0, 0, 0], [20, 0, 0]],
            "two_sided_transition_score": 0.8,
        }
    ]

    assert _overlay_two_sided_features(candidates, diagnostic) == 1
    assert candidates[0]["two_sided_transition_score"] == 0.8


def test_scan_mesh_adjacency_has_an_explicit_identity_free_source_feature():
    candidate = {
        "span_endpoints": [[0, 0, 0], [20, 0, 0]],
        "candidate_source": "cross_source_line_object",
        "representative_source": "scan_mesh_adjacency",
        "cluster_sources": ["scan_mesh_adjacency"],
    }

    values = _features(candidate, {"scan_bbox_mm": [100, 100, 20]})

    assert len(values) == len(FEATURE_NAMES)
    assert "scan_mesh_adjacency" in SOURCES
    assert "source_scan_mesh_adjacency" in FEATURE_NAMES
    assert values[FEATURE_NAMES.index("source_scan_mesh_adjacency")] == 1.0


def _reference_arrangement(candidates, payload):
    endpoints = [_endpoints(candidate) for candidate in candidates]
    axes = []
    for item in endpoints:
        delta = item[1] - item[0]
        axes.append(delta / max(float(np.linalg.norm(delta)), 1e-12))
    diagonal = max(float(np.linalg.norm(payload["scan_bbox_mm"])), 1.0)
    junction_tolerance = max(4.0, min(15.0, 0.03 * diagonal))
    parallel_tolerance = max(10.0, min(35.0, 0.08 * diagonal))
    values = [dict(junction=0, orthogonal=0, endpoint=0, parallel=0, duplicate=0, connected=set()) for _ in candidates]
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left, right = endpoints[left_index], endpoints[right_index]
            angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(axes[left_index], axes[right_index]))), 0.0, 1.0))))
            endpoint_distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            closest_pair = np.unravel_index(int(np.argmin(endpoint_distances)), endpoint_distances.shape)
            segment_gap = min(
                _point_segment_distance(left[0], right),
                _point_segment_distance(left[1], right),
                _point_segment_distance(right[0], left),
                _point_segment_distance(right[1], left),
            )
            if angle >= 25.0 and segment_gap <= junction_tolerance:
                values[left_index]["junction"] += 1
                values[right_index]["junction"] += 1
                if angle >= 60.0:
                    values[left_index]["orthogonal"] += 1
                    values[right_index]["orthogonal"] += 1
            if angle >= 25.0 and endpoint_distances[closest_pair] <= junction_tolerance:
                values[left_index]["endpoint"] += 1
                values[right_index]["endpoint"] += 1
                values[left_index]["connected"].add(int(closest_pair[0]))
                values[right_index]["connected"].add(int(closest_pair[1]))
            metrics = pair_metrics(left, right)
            if angle <= 7.5 and metrics["transverse_offset_mm"] <= parallel_tolerance:
                values[left_index]["parallel"] += 1
                values[right_index]["parallel"] += 1
            if _duplicate(candidates[left_index], candidates[right_index]):
                values[left_index]["duplicate"] += 1
                values[right_index]["duplicate"] += 1
    return values


def test_vectorized_arrangement_matches_scalar_pair_predicates():
    rng = np.random.default_rng(17)
    candidates = []
    for _ in range(30):
        start = rng.uniform(-80.0, 80.0, size=3)
        delta = rng.uniform(-40.0, 40.0, size=3)
        if np.linalg.norm(delta) < 2.0:
            delta[0] += 5.0
        candidates.append({"span_endpoints": [start.tolist(), (start + delta).tolist()]})
    payload = {"scan_bbox_mm": [200.0, 180.0, 120.0]}
    expected = _reference_arrangement(candidates, payload)

    _attach_arrangement_features(candidates, payload)

    for candidate, item in zip(candidates, expected):
        assert candidate["arrangement_junction_neighbor_count"] == item["junction"]
        assert candidate["arrangement_orthogonal_junction_neighbor_count"] == item["orthogonal"]
        assert candidate["arrangement_endpoint_neighbor_count"] == item["endpoint"]
        assert candidate["arrangement_connected_endpoint_count"] == len(item["connected"])
        assert candidate["arrangement_parallel_family_count"] == item["parallel"]
        assert candidate["arrangement_near_duplicate_count"] == item["duplicate"]
