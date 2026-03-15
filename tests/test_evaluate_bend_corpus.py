import importlib.util
from pathlib import Path


MODULE_PATH = Path("/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_corpus.py")
spec = importlib.util.spec_from_file_location("evaluate_bend_corpus", MODULE_PATH)
evaluate_bend_corpus = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(evaluate_bend_corpus)


def _item(scan_name, state, matches):
    payload = {
        "part_key": "PART",
        "scan_name": scan_name,
        "expectation": {
            "state": state,
            "expected_total_bends": 10,
            "expected_completed_bends": None,
        },
        "summary": evaluate_bend_corpus._recompute_summary(matches),
        "details": {"cad_strategy": "detector_mesh"},
        "matches": matches,
    }
    payload["metrics"] = evaluate_bend_corpus._score_scan(payload, payload["expectation"])
    return payload


def test_ordered_scan_records_prioritizes_partial_sequence_before_full():
    metadata = {
        "scan_files": [
            {"name": "full.ply", "state": "full", "sequence": 95, "sequence_confident": True},
            {"name": "scan_2.ply", "state": "partial", "sequence": 2, "sequence_confident": True},
            {"name": "scan_1.ply", "state": "partial", "sequence": 1, "sequence_confident": True},
        ]
    }

    ordered = evaluate_bend_corpus._ordered_scan_records(metadata)

    assert [item["name"] for item in ordered] == ["scan_1.ply", "scan_2.ply", "full.ply"]


def test_sequence_continuity_prior_carries_forward_strong_previous_bend():
    previous_matches = [
        {
            "bend_id": "CAD_B9",
            "status": "PASS",
            "confidence": 0.82,
            "local_evidence_score": 0.73,
            "physical_completion_state": "FORMED",
            "observability_state": "OBSERVED_FORMED",
            "observability_confidence": 0.82,
            "visibility_score": 0.71,
            "visibility_score_source": "heuristic_local_visibility_v0",
            "surface_visibility_ratio": 1.0,
            "local_support_score": 0.63,
            "side_balance_score": 0.44,
            "local_point_count": 120,
            "measurement_mode": "cad_local_neighborhood",
            "measurement_method": "probe_region",
            "measurement_context": {},
        }
    ]
    current_matches = [
        {
            "bend_id": "CAD_B9",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "local_evidence_score": 0.16,
            "observability_state": "UNOBSERVED",
            "observability_confidence": 0.0,
            "visibility_score": 0.12,
            "visibility_score_source": "heuristic_local_visibility_v0",
            "surface_visibility_ratio": 0.0,
            "local_support_score": 0.0,
            "side_balance_score": 0.0,
            "local_point_count": 0,
            "measurement_mode": "cad_local_neighborhood",
            "measurement_method": None,
            "measurement_context": {},
        }
    ]
    previous_item = _item("scan_1.ply", "partial", previous_matches)
    current_item = _item("scan_2.ply", "partial", current_matches)

    updated, carried = evaluate_bend_corpus._apply_sequence_continuity_prior(previous_item, current_item)

    assert carried == ["CAD_B9"]
    assert updated["summary"]["completed_bends"] == 1
    assert updated["metrics"]["continuity_inferred_bends"] == 1
    assert updated["matches"][0]["status"] == "PASS"
    assert updated["matches"][0]["physical_completion_state"] == "FORMED"
    assert updated["matches"][0]["visibility_score"] == 0.12
    assert updated["matches"][0]["assignment_source"] == "SEQUENCE_CONTINUITY"
    assert updated["matches"][0]["assignment_confidence"] == 0.82
    assert updated["matches"][0]["assignment_candidate_id"] == "null_candidate"
    assert updated["matches"][0]["assignment_candidate_kind"] == "NULL"
    assert updated["matches"][0]["measurement_mode"] == "sequence_continuity"
    assert updated["matches"][0]["continuity_inferred"] is True


def test_sequence_continuity_prior_does_not_override_observed_not_formed():
    previous_matches = [
        {
            "bend_id": "CAD_B9",
            "status": "FAIL",
            "confidence": 0.7,
            "local_evidence_score": 0.64,
            "observability_state": "OBSERVED_FORMED",
            "observability_confidence": 0.7,
            "local_point_count": 110,
            "measurement_mode": "cad_local_neighborhood",
            "measurement_method": "probe_region",
            "measurement_context": {},
        }
    ]
    current_matches = [
        {
            "bend_id": "CAD_B9",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "local_evidence_score": 0.08,
            "observability_state": "OBSERVED_NOT_FORMED",
            "observability_confidence": 0.68,
            "local_point_count": 90,
            "measurement_mode": "cad_local_neighborhood",
            "measurement_method": "probe_region",
            "measurement_context": {},
        }
    ]
    previous_item = _item("scan_1.ply", "partial", previous_matches)
    current_item = _item("scan_2.ply", "partial", current_matches)

    updated, carried = evaluate_bend_corpus._apply_sequence_continuity_prior(previous_item, current_item)

    assert carried == []
    assert updated["summary"]["completed_bends"] == 0
    assert updated["matches"][0]["status"] == "NOT_DETECTED"


def test_should_apply_sequence_continuity_requires_confident_forward_partial_sequence():
    previous_item = _item("scan_1.ply", "partial", [])
    current_item = _item("scan_2.ply", "partial", [])

    assert evaluate_bend_corpus._should_apply_sequence_continuity(
        previous_item,
        current_item,
        {"sequence": 1, "sequence_confident": True},
        {"sequence": 2, "sequence_confident": True},
    ) is True
    assert evaluate_bend_corpus._should_apply_sequence_continuity(
        previous_item,
        current_item,
        {"sequence": 1, "sequence_confident": False},
        {"sequence": 2, "sequence_confident": True},
    ) is False


def test_aggregate_reports_staged_monotonicity_and_retention():
    first = _item(
        "scan_1.ply",
        "partial",
        [
            {"bend_id": "CAD_B1", "status": "PASS", "observability_state": "OBSERVED_FORMED", "local_evidence_score": 0.8},
            {"bend_id": "CAD_B2", "status": "FAIL", "observability_state": "OBSERVED_FORMED", "local_evidence_score": 0.7},
        ],
    )
    first["part_key"] = "P1"
    first["scan_sequence"] = 1
    first["sequence_confident"] = True

    second = _item(
        "scan_2.ply",
        "partial",
        [
            {"bend_id": "CAD_B1", "status": "PASS", "observability_state": "OBSERVED_FORMED", "local_evidence_score": 0.8},
            {"bend_id": "CAD_B2", "status": "PASS", "observability_state": "UNOBSERVED", "local_evidence_score": 0.5, "continuity_inferred": True},
            {"bend_id": "CAD_B3", "status": "PASS", "observability_state": "OBSERVED_FORMED", "local_evidence_score": 0.8},
        ],
    )
    second["part_key"] = "P1"
    second["scan_sequence"] = 2
    second["sequence_confident"] = True
    second["metrics"]["continuity_inferred_bends"] = 1

    aggregate = evaluate_bend_corpus._aggregate([first, second])

    assert aggregate["continuity_inferred_scan_count"] == 1
    assert aggregate["continuity_inferred_bends_total"] == 1
    assert aggregate["staged_monotonic_pair_count"] == 1
    assert aggregate["staged_monotonic_pair_rate"] == 1.0
    assert aggregate["staged_retained_bend_rate"] == 1.0


def test_structured_score_scan_uses_count_posterior_median():
    item = _item(
        "scan_1.ply",
        "full",
        [{"bend_id": "CAD_B1", "status": "PASS", "observability_state": "OBSERVED_FORMED"}],
    )
    item["structured_context"] = {
        "count_posterior": {
            "median_completed_bends": 3,
            "map_completed_bends": 2,
            "mean_completed_bends": 2.6,
        }
    }
    metrics = evaluate_bend_corpus._structured_score_scan(
        item,
        {"expected_total_bends": 4, "expected_completed_bends": 4},
    )
    assert metrics["completed_bends_median"] == 3
    assert metrics["expected_total_error_median"] == -1
    assert metrics["expected_completed_error_median"] == -1


def test_aggregate_includes_structured_count_metrics():
    item = _item(
        "full_scan.ply",
        "full",
        [{"bend_id": "CAD_B1", "status": "PASS", "observability_state": "OBSERVED_FORMED"}],
    )
    item["part_key"] = "P1"
    item["expectation"] = {"state": "full", "expected_total_bends": 2, "expected_completed_bends": 2}
    item["metrics"] = evaluate_bend_corpus._score_scan(item, item["expectation"])
    item["structured_context"] = {
        "count_posterior": {
            "median_completed_bends": 2,
            "map_completed_bends": 2,
            "mean_completed_bends": 1.9,
        }
    }
    item["structured_metrics"] = evaluate_bend_corpus._structured_score_scan(item, item["expectation"])

    aggregate = evaluate_bend_corpus._aggregate([item])

    assert aggregate["structured_scans"] == 1
    assert aggregate["structured_mae_vs_expected_total_completed"] == 0.0
    assert aggregate["structured_mae_vs_expected_completed"] == 0.0
    assert aggregate["partial_mae_vs_expected_total_completed"] is None
    assert aggregate["structured_partial_mae_vs_expected_total_completed"] is None
    assert aggregate["structured_full_scan_exact_completion_rate"] == 1.0


def test_aggregate_includes_partial_mae_metrics():
    item = _item(
        "partial_scan.ply",
        "partial",
        [{"bend_id": "CAD_B1", "status": "PASS", "observability_state": "OBSERVED_FORMED"}],
    )
    item["part_key"] = "P2"
    item["expectation"] = {"state": "partial", "expected_total_bends": 4, "expected_completed_bends": 2}
    item["metrics"] = evaluate_bend_corpus._score_scan(item, item["expectation"])
    item["structured_context"] = {
        "count_posterior": {
            "median_completed_bends": 2,
            "map_completed_bends": 2,
            "mean_completed_bends": 1.8,
        }
    }
    item["structured_metrics"] = evaluate_bend_corpus._structured_score_scan(item, item["expectation"])

    aggregate = evaluate_bend_corpus._aggregate([item])

    assert aggregate["partial_mae_vs_expected_total_completed"] == 3.0
    assert aggregate["partial_mae_vs_expected_completed"] == 1.0
    assert aggregate["structured_partial_mae_vs_expected_total_completed"] == 2.0
    assert aggregate["structured_partial_mae_vs_expected_completed"] == 0.0
