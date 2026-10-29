from pathlib import Path
from types import SimpleNamespace

import numpy as np
import bend_analysis_worker as worker
from domains.bend.services import structured_count_reporting_service as structured_service


def test_promote_structured_count_reporting_adds_operator_fields(monkeypatch, tmp_path):
    report = {
        "part_id": "PART",
        "summary": {
            "completed_bends": 4,
            "expected_bends": 10,
            "remaining_bends": 6,
        },
        "operator_brief": {
            "headline": "4/10 completed, 2 in spec, 2 out of spec, 6 remaining",
        },
        "matches": [],
    }

    model_path = tmp_path / "fake.joblib"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr(worker, "_latest_unary_model_path", lambda: model_path)
    monkeypatch.setattr(structured_service, "load_unary_models", lambda _path: {"fake": True})
    monkeypatch.setattr(
        structured_service,
        "score_case_payload",
        lambda payload, _bundle: {
            **payload,
            "structured_context": {
                "count_posterior": {
                    "median_completed_bends": 6,
                    "map_completed_bends": 6,
                    "mean_completed_bends": 5.7,
                }
            },
        },
    )

    promoted = worker._promote_structured_count_reporting(report)

    assert promoted["summary"]["match_evidence_completed_bends"] == 4
    assert promoted["summary"]["structured_completed_bends"] == 6
    assert promoted["summary"]["structured_remaining_bends"] == 4
    assert promoted["summary"]["structured_count_delta_vs_match"] == 2
    assert "Count model: 6/10 complete" in promoted["operator_brief"]["headline"]
    assert "Direct evidence: 4 explicit bends" in promoted["operator_brief"]["headline"]


def test_promote_structured_count_reporting_caps_hybrid_process_family_to_observed_count(monkeypatch, tmp_path):
    report = {
        "part_id": "28870000",
        "summary": {
            "countable_completed_bends": 1,
            "completed_bends": 1,
            "expected_bends": 2,
            "remaining_bends": 1,
        },
        "operator_brief": {
            "headline": "1/2 completed, 0 in spec, 1 out of spec, 1 remaining",
        },
        "matches": [
            {
                "cad_bend": {
                    "bend_id": "CAD_B1",
                    "countable_in_regression": False,
                },
                "observability_state": "OBSERVED_FORMED",
            },
            {
                "cad_bend": {
                    "bend_id": "CAD_B4",
                    "countable_in_regression": True,
                },
                "observability_state": "UNOBSERVED",
            },
            {
                "cad_bend": {
                    "bend_id": "CAD_B5",
                    "countable_in_regression": True,
                },
                "observability_state": "OBSERVED_FORMED",
            },
        ],
    }

    model_path = tmp_path / "fake.joblib"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr(worker, "_latest_unary_model_path", lambda: model_path)
    monkeypatch.setattr(structured_service, "load_unary_models", lambda _path: {"fake": True})
    monkeypatch.setattr(
        structured_service,
        "score_case_payload",
        lambda payload, _bundle: {
            **payload,
            "structured_context": {
                "count_posterior": {
                    "median_completed_bends": 2,
                    "map_completed_bends": 2,
                    "mean_completed_bends": 1.8,
                }
            },
        },
    )
    promoted = worker._promote_structured_count_reporting(report)

    assert promoted["summary"]["structured_completed_bends"] == 1
    assert promoted["summary"]["structured_remaining_bends"] == 1
    assert promoted["summary"]["structured_count_delta_vs_match"] == 0


def test_augment_progressive_bend_report_filters_process_actions_from_operator_brief():
    from domains.bend.services.report_enrichment_service import augment_progressive_bend_report

    class _Details:
        expected_bend_count = 2
        expected_progress_pct = 50.0
        overdetected_count = 0

    report = {
        "summary": {
            "countable_completed_bends": 1,
            "completed_in_spec": 0,
        },
        "matches": [
            {"cad_bend": {"bend_id": "CAD_B1", "countable_in_regression": False}},
            {"cad_bend": {"bend_id": "CAD_B4", "countable_in_regression": True}},
            {"cad_bend": {"bend_id": "CAD_B5", "countable_in_regression": True}},
        ],
        "operator_actions": [
            {"bend_id": "CAD_B1", "status": "FAIL", "location": "angle", "action": "rolled"},
            {"bend_id": "CAD_B4", "status": "NOT_DETECTED", "location": "bend_line", "action": "upper"},
            {"bend_id": "CAD_B5", "status": "FAIL", "location": "bend_angle", "action": "lower"},
        ],
    }

    augmented = augment_progressive_bend_report(report, _Details())

    assert [item["bend_id"] for item in augmented["operator_brief"]["actions"]] == ["CAD_B4", "CAD_B5"]
    assert [item["bend_id"] for item in augmented["process_feature_actions"]] == ["CAD_B1"]
    assert augmented["summary"]["critical_actions"] == 2
    assert augmented["summary"]["process_feature_actions"] == 1


def test_export_aligned_scan_visualization_persists_deviations(monkeypatch, tmp_path):
    def fake_build(*_args, **_kwargs):
        aligned_path = tmp_path / "aligned_scan.ply"
        aligned_path.write_text("ply", encoding="utf-8")
        deviations_path = tmp_path / "deviations.npy"
        np.save(deviations_path, np.array([0.1, -0.2, 0.05], dtype=np.float32))
        return {
            "aligned_scan_path": aligned_path,
            "aligned_points": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
            "deviations_path": deviations_path,
            "deviation_stats": {"count": 3},
        }

    monkeypatch.setattr(worker, "build_bend_heatmap_evidence", fake_build)

    bundle = worker._export_aligned_scan_visualization(
        cad_path="cad.step",
        scan_path="scan.ply",
        output_dir=tmp_path,
        tolerance_angle=1.0,
        runtime_cfg=SimpleNamespace(local_refinement_kwargs={"deterministic_seed": 17}),
    )

    assert bundle["aligned_scan_path"] == tmp_path / "aligned_scan.ply"
    assert bundle["deviations_path"] == tmp_path / "deviations.npy"
    assert np.allclose(np.load(bundle["deviations_path"]), np.array([0.1, -0.2, 0.05], dtype=np.float32))
    assert bundle["deviation_stats"]["count"] == 3


def test_resolve_scan_state_requires_explicit_source_observation_state():
    explicit = worker._resolve_scan_state(
        "scan_01.ply",
        SimpleNamespace(scan_state="full"),
        {"summary": {}, "scan_quality": {}},
    )
    count_inferred = worker._resolve_scan_state(
        "full_final_complete.ply",
        SimpleNamespace(scan_state="unknown"),
        {
            "summary": {"expected_bends": 2, "completed_bends": 2, "remaining_bends": 0},
            "scan_quality": {"coverage_pct": 85.0},
        },
    )

    assert explicit == "full"
    assert count_inferred is None
