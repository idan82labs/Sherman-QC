"""Actual worker orchestration with fully synthetic, non-native dependencies.

This is not native runtime or physical qualification. Only this named worker
source and pure decision source are imported; geometry/model/qualification/
artifact providers are replaced before module execution. IO stays in tmp_path.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


WORKER_SOURCE = Path(__file__).resolve().parents[1] / "backend" / "bend_analysis_worker.py"


def _unexpected(*_args, **_kwargs):
    raise AssertionError("Unexpected synthetic provider call; native or protected IO is forbidden")


def _stub_module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def synthetic_worker(monkeypatch):
    # No NumPy/native geometry library is needed to exercise this boundary.
    fake_numpy = _stub_module(
        monkeypatch, "numpy",
        ndarray=type("SyntheticArray", (), {}),
        integer=type("SyntheticInteger", (int,), {}),
        floating=type("SyntheticFloat", (float,), {}),
        load=_unexpected,
    )
    _stub_module(monkeypatch, "domains.bend.services.heatmap_evidence_service", build_bend_heatmap_evidence=_unexpected)
    _stub_module(
        monkeypatch, "domains.bend.services.heatmap_consistency_service",
        evaluate_bend_heatmap_consistency=_unexpected,
        merge_heatmap_consistency_into_report=_unexpected,
    )
    _stub_module(monkeypatch, "domains.bend.services.report_enrichment_service", augment_progressive_bend_report=_unexpected)
    _stub_module(monkeypatch, "domains.bend.services.structured_count_reporting_service", promote_structured_count_reporting=_unexpected)
    _stub_module(
        monkeypatch, "domains.bend.services.runtime_service",
        export_bend_reference_visualization=_unexpected,
        export_reference_mesh_from_vertices=_unexpected,
        load_bend_runtime_config=_unexpected,
        run_progressive_bend_inspection=_unexpected,
    )
    _stub_module(monkeypatch, "domains.reporting.services.bend_report_service", build_bend_artifact_bundle=_unexpected)
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("_synthetic_worker_decision_boundary", WORKER_SOURCE)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert worker.np is fake_numpy
    newly_imported = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"open3d", "trimesh", "scipy", "OCC", "cadquery"} for name in newly_imported)
    return worker


@pytest.mark.parametrize("artifact_behavior,expected_render", [
    ("manifest", "UNVERIFIED"),
    ("overview_only", "UNAVAILABLE"),
    ("bundle_failure", "UNAVAILABLE"),
    ("reference_failure", "UNAVAILABLE"),
])
def test_actual_worker_projects_json_text_and_artifact_failure(
    synthetic_worker, monkeypatch, tmp_path, capsys, artifact_behavior, expected_render,
):
    worker = synthetic_worker
    calls = []
    source_report = {
        "part_id": "SYNTHETIC_BOUNDARY",
        "alignment": {},
        "summary": {"total_bends": 3, "completed_bends": 999, "passed": 999, "progress_pct": 100},
        "matches": [
            {"bend_id": "A", "status": "PASS", "observability_state": "UNKNOWN",
             "observability_detail_state": "UNOBSERVED", "physical_completion_state": "FORMED", "measured_angle": 90.0},
            {"bend_id": "B", "status": "NOT_DETECTED", "observability_state": "OBSERVED_NOT_FORMED",
             "physical_completion_state": "NOT_FORMED", "measured_angle": 180.0},
            {"bend_id": "C", "status": "WARNING", "observability_state": "PARTIALLY_OBSERVED",
             "physical_completion_state": "PARTIALLY_FORMED", "measured_angle": 120.0},
            {"bend_id": "P", "status": "PASS", "observability_state": "FORMED", "countable_in_regression": False},
        ],
    }
    initial_report = deepcopy(source_report)
    report = SimpleNamespace(to_dict=lambda: deepcopy(source_report), to_table_string=_unexpected, processing_time_ms=0.0)
    details = SimpleNamespace(cad_vertices=object(), cad_triangles=object(), scan_state=None,
                              to_dict=lambda: {"source": "synthetic"})
    runtime_config = SimpleNamespace(cad_import_deflection=0.1, to_dict=lambda: {"synthetic": True})

    def run_pipeline(**kwargs):
        calls.append("pipeline")
        assert kwargs["use_part_truth_hints"] is False
        assert kwargs["expected_bend_override"] is None
        assert kwargs["spec_bend_angles_hint"] is None
        return report, details

    def export_reference(**kwargs):
        calls.append("reference")
        if artifact_behavior == "reference_failure":
            raise RuntimeError("synthetic reference failure")
        # This placeholder is never parsed by a geometry/artifact provider.
        (kwargs["output_dir"] / "reference_mesh.ply").write_text("synthetic placeholder", encoding="utf-8")

    def heatmap(*_args, **_kwargs):
        calls.append("heatmap")
        return {}

    def enrich(payload, _details):
        calls.append("enrich")
        payload["summary"]["expected_bends"] = 999
        return payload

    def count_estimate(payload):
        calls.append("model_stub")
        payload["summary"]["structured_completed_bends"] = 999
        return payload

    def qualification():
        calls.append("qualification_stub")
        # Even an upstream declaration must not override the additive V1 ban.
        return {"production_use_allowed": True, "operating_mode": "PRODUCTION_QC"}

    def build_artifacts(**kwargs):
        calls.append("artifacts")
        projected = kwargs["report_dict"]["decision_contract"]
        assert projected["observed_counts"]["formed"] == 0
        assert projected["bends"][0]["state"] == "UNKNOWN"
        assert projected["production_disposition"] == "NOT_AUTHORIZED"
        assert projected["render_evidence"] == "UNAVAILABLE"
        if artifact_behavior == "bundle_failure":
            raise RuntimeError("synthetic artifact failure")
        artifacts = {"bend_overlay_overview_url": "/synthetic/overview.png"}
        if artifact_behavior == "manifest":
            artifacts["bend_overlay_manifest_url"] = "/synthetic/manifest.json"
        return SimpleNamespace(artifacts=artifacts)

    for name, value in {
        "load_bend_runtime_config": lambda: runtime_config,
        "run_progressive_bend_inspection": run_pipeline,
        "export_reference_mesh_from_vertices": export_reference,
        "build_bend_heatmap_evidence": heatmap,
        "augment_progressive_bend_report": enrich,
        "promote_structured_count_reporting": count_estimate,
        "_load_qualification_context": qualification,
        "build_bend_artifact_bundle": build_artifacts,
    }.items():
        monkeypatch.setattr(worker, name, value)

    output_dir = tmp_path / "synthetic_worker"
    args = SimpleNamespace(output_dir=str(output_dir), job_id="SYNTHETIC", cad_path="synthetic-not-opened.step",
                           scan_path="synthetic-not-opened.ply", part_id="SYNTHETIC_BOUNDARY", part_name="Synthetic",
                           tolerance_angle=1.0, tolerance_radius=0.5)
    assert worker.run_worker(args) == 0
    assert source_report == initial_report
    captured = capsys.readouterr()
    assert captured.err == ""
    events = [(tag, json.loads(payload)) for tag, payload in
              (line.split("\t", 1) for line in captured.out.splitlines())]
    assert not any(tag == "ERROR" for tag, _ in events)
    result = next(payload for tag, payload in events if tag == "RESULT")
    assert Path(result["report_path"]).parent == output_dir
    persisted = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    text = Path(result["table_path"]).read_text(encoding="utf-8")
    decision = persisted["decision_contract"]
    assert decision["observed_counts"] == {"formed": 0, "unformed": 1, "unknown": 2, "total": 3, "partially_observed": 1}
    assert decision["registration"] == "UNVERIFIED"
    assert decision["render_evidence"] == expected_render
    assert decision["production_disposition"] == "NOT_AUTHORIZED"
    assert persisted["summary"]["structured_completed_bends"] == 999  # preserved, not promoted
    assert persisted["matches"][0]["status"] == "PASS"  # raw evidence remains immutable
    assert "Observed formed: 0/3" in text
    assert "Observed unformed: 1; Unknown: 2" in text
    assert "A | UNKNOWN | UNOBSERVED" in text
    assert f"Render evidence: {expected_render}" in text
    assert "999" not in text and "A | PASS" not in text
    assert text.startswith("RESEARCH ONLY - NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT")
    assert calls[:6] == ["pipeline", "reference", "heatmap", "enrich", "model_stub", "qualification_stub"]
    assert calls.count("artifacts") == (0 if artifact_behavior == "reference_failure" else 1)
    assert ("bend_overlay_manifest_url" in result["artifacts"]) is (artifact_behavior == "manifest")
    if artifact_behavior in {"bundle_failure", "reference_failure"}:
        assert any("warning" in payload.get("message", "") for tag, payload in events if tag == "PROGRESS")
