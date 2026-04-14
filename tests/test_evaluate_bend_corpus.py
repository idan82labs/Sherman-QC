import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path("/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/scripts/evaluate_bend_corpus.py")
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


def test_stable_scan_output_name_distinguishes_extension_collisions():
    ply_name = evaluate_bend_corpus._stable_scan_output_name("44211000_A.ply")
    stl_name = evaluate_bend_corpus._stable_scan_output_name("44211000_A.stl")
    assert ply_name != stl_name
    assert ply_name.endswith(".json")
    assert stl_name.endswith(".json")


def test_load_results_from_manifest_preserves_failed_entries(tmp_path):
    success_path = tmp_path / "scan_a.json"
    success_payload = {"part_key": "P1", "scan_name": "a.ply", "metrics": {"completed_bends": 1}}
    success_path.write_text(json.dumps(success_payload), encoding="utf-8")

    manifest = {
        "warnings": ["P1/b.ply: timed out"],
        "entries": [
            {
                "part_key": "P1",
                "scan_name": "a.ply",
                "scan_path": "/tmp/a.ply",
                "scan_state": "full",
                "expected_total_bends": 1,
                "expected_completed_bends": 1,
                "output_json": str(success_path),
                "status": "success",
            },
            {
                "part_key": "P1",
                "scan_name": "b.ply",
                "scan_path": "/tmp/b.ply",
                "scan_state": "full",
                "expected_total_bends": 1,
                "expected_completed_bends": 1,
                "output_json": str(tmp_path / "missing.json"),
                "status": "failed",
                "error": "timed out",
            },
        ],
    }

    results, warnings = evaluate_bend_corpus._load_results_from_manifest(manifest)

    assert len(results) == 2
    assert results[0]["scan_name"] == "a.ply"
    assert results[1]["scan_name"] == "b.ply"
    assert results[1]["error"] == "timed out"
    assert warnings == ["P1/b.ply: timed out"]


def test_run_case_subprocess_appends_skip_heatmap_flag(monkeypatch, tmp_path):
    output_json = tmp_path / "case.json"
    captured = {}

    def fake_run(cmd, capture_output, text, timeout, env, check):
        captured["cmd"] = cmd
        output_json.write_text(json.dumps({"summary": {}, "matches": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(evaluate_bend_corpus.subprocess, "run", fake_run)

    evaluate_bend_corpus._run_case_subprocess(
        part_key="P1",
        cad_path="/tmp/cad.stp",
        scan_path="/tmp/scan.stl",
        output_json=output_json,
        expectation={"state": "full", "expected_total_bends": 1, "expected_completed_bends": 1},
        spec_bend_angles=[90.0],
        spec_schedule_type="complete",
        timeout_sec=120,
        skip_heatmap_consistency=True,
        heatmap_cache_dir=None,
    )

    assert "--skip-heatmap-consistency" in captured["cmd"]


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


def test_should_apply_sequence_continuity_uses_scan_name_fallback_when_previous_partial_is_complete():
    previous_matches = [
        {
            "bend_id": f"CAD_B{i}",
            "status": "PASS",
            "confidence": 0.85,
            "local_evidence_score": 0.8,
            "physical_completion_state": "FORMED",
            "observability_state": "FORMED",
            "measurement_context": {},
        }
        for i in range(1, 9)
    ]
    current_matches = previous_matches[:4] + [
        {
            "bend_id": f"CAD_B{i}",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "local_evidence_score": 0.0,
            "physical_completion_state": "UNKNOWN",
            "observability_state": "UNKNOWN",
            "measurement_context": {},
        }
        for i in range(5, 9)
    ]
    previous_item = _item("40500000_0_SCAN1 1.ply", "partial", previous_matches)
    current_item = _item("40500000_0_SCAN2 1.ply", "partial", current_matches)
    previous_item["expectation"]["expected_total_bends"] = 8
    current_item["expectation"]["expected_total_bends"] = 8
    previous_item["metrics"] = evaluate_bend_corpus._score_scan(previous_item, previous_item["expectation"])
    current_item["metrics"] = evaluate_bend_corpus._score_scan(current_item, current_item["expectation"])

    assert evaluate_bend_corpus._should_apply_sequence_continuity(
        previous_item,
        current_item,
        {"sequence": 10, "sequence_confident": False, "name": "40500000_0_SCAN1 1.ply"},
        {"sequence": 10, "sequence_confident": False, "name": "40500000_0_SCAN2 1.ply"},
    ) is True


def test_should_apply_sequence_continuity_scan_name_fallback_requires_complete_previous_partial():
    previous_matches = [
        {
            "bend_id": f"CAD_B{i}",
            "status": "PASS",
            "confidence": 0.85,
            "local_evidence_score": 0.8,
            "physical_completion_state": "FORMED",
            "observability_state": "FORMED",
            "measurement_context": {},
        }
        for i in range(1, 6)
    ]
    current_matches = previous_matches[:4]
    previous_item = _item("23553000_B_SCAN_1.ply", "partial", previous_matches)
    current_item = _item("23553000_B_SCAN_2.ply", "partial", current_matches)
    previous_item["expectation"]["expected_total_bends"] = 16
    current_item["expectation"]["expected_total_bends"] = 16
    previous_item["metrics"] = evaluate_bend_corpus._score_scan(previous_item, previous_item["expectation"])
    current_item["metrics"] = evaluate_bend_corpus._score_scan(current_item, current_item["expectation"])

    assert evaluate_bend_corpus._should_apply_sequence_continuity(
        previous_item,
        current_item,
        {"sequence": 1, "sequence_confident": False, "name": "23553000_B_SCAN_1.ply"},
        {"sequence": 2, "sequence_confident": False, "name": "23553000_B_SCAN_2.ply"},
    ) is False


def test_partial_to_full_retention_carries_explicit_previous_partial_bend():
    previous_matches = [
        {
            "bend_id": "CAD_B13",
            "status": "FAIL",
            "confidence": 0.5,
            "local_evidence_score": 0.899,
            "physical_completion_state": "FORMED",
            "observability_state": "FORMED",
            "measurement_method": "surface_classification",
            "measurement_context": {},
        }
    ]
    current_matches = [
        {
            "bend_id": f"CAD_B{i}",
            "status": "PASS",
            "confidence": 0.7,
            "local_evidence_score": 0.7,
            "physical_completion_state": "FORMED",
            "observability_state": "FORMED",
            "measurement_context": {},
        }
        for i in range(1, 10)
        if i != 13
    ] + [
        {
            "bend_id": "CAD_B13",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "local_evidence_score": 0.0,
            "physical_completion_state": "UNKNOWN",
            "observability_state": "UNKNOWN",
            "measurement_context": {},
        },
    ]
    previous_item = _item("23553000_B_SCAN_2.ply", "partial", previous_matches)
    current_item = _item("23553000_B_FULL_SCAN 1.ply", "full", current_matches)

    assert evaluate_bend_corpus._should_apply_partial_to_full_retention(previous_item, current_item) is True

    updated, retained = evaluate_bend_corpus._apply_partial_to_full_retention(previous_item, current_item)
    retained_match = next(match for match in updated["matches"] if match["bend_id"] == "CAD_B13")

    assert retained == ["CAD_B13"]
    assert updated["summary"]["completed_bends"] == 10
    assert updated["metrics"]["retained_from_previous_partial_bends"] == 1
    assert retained_match["status"] == "FAIL"
    assert retained_match["physical_completion_state"] == "FORMED"
    assert retained_match["measurement_method"] == "retain_explicit_previous_partial_into_full"
    assert retained_match["retained_from_previous_partial"] is True


def test_partial_to_full_retention_does_not_carry_sequence_inferred_partial_bend():
    previous_matches = [
        {
            "bend_id": "CAD_B13",
            "status": "PASS",
            "confidence": 0.9,
            "local_evidence_score": 0.8,
            "physical_completion_state": "FORMED",
            "observability_state": "FORMED",
            "continuity_inferred": True,
            "measurement_method": "carry_forward_previous_partial",
            "measurement_context": {},
        }
    ]
    current_matches = [
        {
            "bend_id": f"CAD_B{i}",
            "status": "PASS",
            "confidence": 0.7,
            "local_evidence_score": 0.7,
            "physical_completion_state": "FORMED",
            "observability_state": "FORMED",
            "measurement_context": {},
        }
        for i in range(1, 4)
        if i != 13
    ] + [
        {
            "bend_id": "CAD_B13",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "local_evidence_score": 0.0,
            "physical_completion_state": "UNKNOWN",
            "observability_state": "UNKNOWN",
            "measurement_context": {},
        },
    ]
    previous_item = _item("scan_2.ply", "partial", previous_matches)
    current_item = _item("full_scan.ply", "full", current_matches)

    assert evaluate_bend_corpus._should_apply_partial_to_full_retention(previous_item, current_item) is True
    updated, retained = evaluate_bend_corpus._apply_partial_to_full_retention(previous_item, current_item)

    assert retained == []
    assert updated["summary"]["completed_bends"] == 3


def test_evaluate_part_skip_existing_preserves_previous_partial_for_sequence_continuity(tmp_path, monkeypatch):
    metadata = {
        "part_key": "38172000",
        "expected_bends_override": 10,
        "scan_files": [
            {
                "name": "38172000_01_SCAN_1.ply",
                "path": "/tmp/38172000_01_SCAN_1.ply",
                "state": "partial",
                "sequence": 1,
                "sequence_confident": True,
            },
            {
                "name": "38172000_01_SCAN_2 1.ply",
                "path": "/tmp/38172000_01_SCAN_2_1.ply",
                "state": "partial",
                "sequence": 2,
                "sequence_confident": True,
            },
        ],
    }
    labels = {"expected_total_bends": 10, "scans": []}
    metadata_path = tmp_path / "metadata.json"
    labels_path = tmp_path / "labels.template.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    previous_item = _item(
        "38172000_01_SCAN_1.ply",
        "partial",
        [
            {
                "bend_id": "CAD_B2",
                "status": "PASS",
                "confidence": 0.753,
                "local_evidence_score": 0.875,
                "physical_completion_state": "FORMED",
                "observability_state": "FORMED",
                "measurement_method": "detected_bend_match",
                "measurement_context": {},
            }
        ],
    )
    current_item = _item(
        "38172000_01_SCAN_2 1.ply",
        "partial",
        [
            {
                "bend_id": "CAD_B2",
                "status": "NOT_DETECTED",
                "confidence": 0.0,
                "local_evidence_score": 0.0,
                "physical_completion_state": "UNKNOWN",
                "observability_state": "UNKNOWN",
                "measurement_context": {},
            }
        ],
    )

    part_record = {
        "part_key": "38172000",
        "metadata_path": str(metadata_path),
        "labels_template_path": str(labels_path),
    }
    output_dir = tmp_path / "evaluation"
    part_out_dir = output_dir / "38172000"
    part_out_dir.mkdir(parents=True, exist_ok=True)
    existing_path = part_out_dir / evaluate_bend_corpus._stable_scan_output_name("38172000_01_SCAN_1.ply")
    existing_path.write_text(json.dumps(previous_item), encoding="utf-8")

    def fake_run_case_subprocess(**kwargs):
        return json.loads(json.dumps(current_item))

    monkeypatch.setattr(evaluate_bend_corpus, "_run_case_subprocess", fake_run_case_subprocess)
    monkeypatch.setattr(evaluate_bend_corpus, "_annotate_structured_predictions", lambda item, expectation, unary_bundle: item)
    monkeypatch.setattr(evaluate_bend_corpus, "_preferred_cad_file", lambda metadata: "/tmp/38172000_01.stp")
    monkeypatch.setattr(evaluate_bend_corpus, "_spec_bend_angles", lambda metadata: [])
    monkeypatch.setattr(evaluate_bend_corpus, "_spec_schedule_type", lambda metadata: None)

    results, warnings, manifest_entries = evaluate_bend_corpus.evaluate_part(
        part_record=part_record,
        output_dir=output_dir,
        skip_existing=True,
        timeout_sec=1,
        unary_bundle=None,
        skip_heatmap_consistency=True,
        heatmap_cache_dir=None,
    )

    assert warnings == []
    assert [entry["status"] for entry in manifest_entries] == ["skipped_existing", "success"]
    carried = next(match for match in results[1]["matches"] if match["bend_id"] == "CAD_B2")
    assert carried["measurement_method"] == "carry_forward_previous_partial"
    assert carried["assignment_source"] == "SEQUENCE_CONTINUITY"
    assert carried["continuity_inferred"] is True
    assert results[1]["metrics"]["continuity_inferred_bends"] == 1


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
    assert aggregate["scoreboards"]["completion"]["scan_level_mae_expected_completed"] == 1.0
    assert aggregate["scoreboards"]["abstention"]["abstention_rate"] == 1.0


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
    assert aggregate["scoreboards"]["completion"]["false_negative_completion_rate"] == 0.5


def test_heatmap_consistency_metrics_ignore_rolled_features():
    item = _item(
        "full_scan.ply",
        "full",
        [
            {
                "bend_id": "CAD_B1",
                "status": "PASS",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "ROLLED_SECTION",
                "countable_in_regression": False,
                "heatmap_consistency": {"status": "NOT_APPLICABLE"},
            },
            {
                "bend_id": "CAD_B4",
                "status": "FAIL",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "profile_section",
                "heatmap_consistency": {"status": "CONTRADICTED"},
            },
        ],
    )
    metrics = evaluate_bend_corpus._heatmap_consistency_metrics(item, item["expectation"])
    assert metrics["countable_bends"] == 1
    assert metrics["contradicted_bends"] == 1
    assert "profile_section" in metrics["measurement_method_breakdown"]


def test_aggregate_includes_heatmap_consistency_counts():
    item = _item(
        "full_scan.ply",
        "full",
        [
            {
                "bend_id": "CAD_B4",
                "status": "PASS",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "probe_region",
                "heatmap_consistency": {"status": "SUPPORTED"},
            },
            {
                "bend_id": "CAD_B5",
                "status": "FAIL",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "profile_section",
                "heatmap_consistency": {"status": "CONTRADICTED"},
            },
        ],
    )
    item["heatmap_consistency_metrics"] = evaluate_bend_corpus._heatmap_consistency_metrics(item, item["expectation"])

    aggregate = evaluate_bend_corpus._aggregate([item])

    assert aggregate["heatmap_consistency_scans"] == 1
    assert aggregate["heatmap_supported_bends"] == 1
    assert aggregate["heatmap_contradicted_bends"] == 1
    assert aggregate["heatmap_supported_rate"] == 0.5
    assert aggregate["heatmap_pass_contradicted"] == 0
    assert aggregate["heatmap_fail_warning_supported"] == 0
    assert aggregate["heatmap_fail_supported"] == 0
    assert aggregate["heatmap_warning_supported"] == 0


def test_aggregate_includes_heatmap_status_slices():
    item = _item(
        "partial_scan.ply",
        "partial",
        [
            {
                "bend_id": "CAD_B1",
                "status": "PASS",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "probe_region",
                "heatmap_consistency": {"status": "CONTRADICTED", "independence_class": "INDEPENDENT"},
            },
            {
                "bend_id": "CAD_B2",
                "status": "FAIL",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "profile_section",
                "heatmap_consistency": {"status": "SUPPORTED", "independence_class": "PARTIAL"},
            },
            {
                "bend_id": "CAD_B3",
                "status": "WARNING",
                "observability_state": "OBSERVED_FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "surface_classification",
                "heatmap_consistency": {"status": "SUPPORTED", "independence_class": "PARTIAL"},
            },
            {
                "bend_id": "CAD_B4",
                "status": "NOT_DETECTED",
                "observability_state": "UNOBSERVED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "unknown",
                "heatmap_consistency": {"status": "INSUFFICIENT_EVIDENCE", "independence_class": "INDEPENDENT"},
            },
        ],
    )
    item["heatmap_consistency_metrics"] = evaluate_bend_corpus._heatmap_consistency_metrics(item, item["expectation"])

    aggregate = evaluate_bend_corpus._aggregate([item])

    assert aggregate["heatmap_pass_contradicted"] == 1
    assert aggregate["heatmap_fail_supported"] == 1
    assert aggregate["heatmap_warning_supported"] == 1
    assert aggregate["heatmap_fail_warning_supported"] == 2
    assert aggregate["heatmap_partial_pass_contradicted"] == 1
    assert aggregate["scoreboards"]["position"]["heatmap_contradicted_rate"] == 0.25


def test_aggregate_includes_contradiction_case_listing():
    item = _item(
        "full_scan.ply",
        "full",
        [
            {
                "bend_id": "CAD_B2",
                "status": "FAIL",
                "observability_state": "OBSERVED_FORMED",
                "physical_completion_state": "FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measurement_method": "detected_bend_match",
                "assignment_source": "GLOBAL_DETECTION",
                "measured_angle": 135.3,
                "target_angle": 134.7,
                "line_center_deviation_mm": 3.94,
                "assignment_candidate_count": 1,
                "assignment_confidence": 0.9,
                "assignment_candidate_score": 0.8,
                "assignment_null_score": 0.1,
            },
        ],
    )
    item["metrics"] = evaluate_bend_corpus._score_scan(item, item["expectation"])
    aggregate = evaluate_bend_corpus._aggregate([item])
    assert aggregate["contradiction_case_count"] == 1
    case = aggregate["_contradiction_cases"][0]
    assert case["bend_id"] == "CAD_B2"
    assert case["correspondence_state"] == "CONFIDENT"


def test_write_scoreboards_outputs_json(tmp_path):
    item = _item(
        "full_scan.ply",
        "full",
        [
            {
                "bend_id": "CAD_B1",
                "status": "PASS",
                "observability_state": "OBSERVED_FORMED",
                "physical_completion_state": "FORMED",
                "feature_type": "DISCRETE_BEND",
                "countable_in_regression": True,
                "measured_angle": 90.2,
                "target_angle": 90.0,
                "angle_deviation": 0.2,
                "line_center_deviation_mm": 0.4,
                "assignment_candidate_count": 1,
                "assignment_confidence": 0.92,
                "assignment_candidate_score": 0.8,
                "assignment_null_score": 0.1,
            },
        ],
    )
    aggregate = evaluate_bend_corpus._aggregate([item])
    evaluate_bend_corpus._write_scoreboards(tmp_path, aggregate)

    payload = json.loads((tmp_path / "scoreboards.json").read_text(encoding="utf-8"))
    assert payload["scoreboards"]["completion"]["countable_bends"] == 1
    assert payload["scoreboards"]["metrology"]["angle_mae_deg"] == 0.2


def test_write_contradiction_debug_outputs_markdown_and_json(tmp_path):
    aggregate = {
        "_contradiction_cases": [
            {
                "part_key": "49125000",
                "scan_name": "49125000_A00.ply",
                "bend_id": "CAD_B2",
                "status": "FAIL",
                "measured_angle": 135.33,
                "target_angle": 134.72,
                "line_center_deviation_mm": 3.94,
                "correspondence_state": "CONFIDENT",
            }
        ]
    }

    evaluate_bend_corpus._write_contradiction_debug(tmp_path, aggregate)

    json_payload = json.loads((tmp_path / "decision_contradiction_cases.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "decision_contradiction_cases.md").read_text(encoding="utf-8")
    assert json_payload[0]["bend_id"] == "CAD_B2"
    assert "49125000" in markdown
    assert "CAD_B2" in markdown
