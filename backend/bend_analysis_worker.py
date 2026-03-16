"""
Isolated bend-inspection worker process.

Runs the progressive bend pipeline outside the API process so:
- heavy Open3D work does not block the FastAPI server event loop
- thread limits can be applied consistently on memory-constrained Macs
- bend jobs can emit structured progress/result payloads back to the server
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from bend_unary_model import load_unary_models, score_case_payload
from qc_engine import ScanQCEngine
from bend_inspection_pipeline import (
    export_bend_reference_visualization,
    export_reference_mesh_from_vertices,
    load_bend_runtime_config,
    run_progressive_bend_inspection,
)
from bend_report_generator import build_bend_artifact_bundle


def _sanitize_error_message(message: Any) -> str:
    text = str(message) if message is not None else ""
    text = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", text)
    text = " ".join(text.split())
    return text.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        num = float(value)
        return num if math.isfinite(num) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _latest_unary_model_path() -> Optional[Path]:
    latest = Path("/Users/idant/82Labs/Sherman QC/Full Project/output/unary_models/latest.json")
    if not latest.exists():
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidate = payload.get("model_path") or payload.get("path")
    if candidate:
        model_path = Path(str(candidate))
        if model_path.is_dir():
            model_path = model_path / "bend_unary_models.joblib"
        return model_path if model_path.exists() else None
    latest_dir = payload.get("latest_bootstrap_dir")
    if not latest_dir:
        return None
    model_path = Path(str(latest_dir)) / "bend_unary_models.joblib"
    return model_path if model_path.exists() else None


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"{tag}\t{json.dumps(payload, ensure_ascii=False)}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sherman QC isolated bend worker")
    p.add_argument("--job-id", required=True)
    p.add_argument("--cad-path", required=True)
    p.add_argument("--scan-path", required=True)
    p.add_argument("--part-id", required=True)
    p.add_argument("--part-name", required=True)
    p.add_argument("--tolerance-angle", type=float, required=True)
    p.add_argument("--tolerance-radius", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    return p


def _augment_report_dict(report_dict: Dict[str, Any], details: Any) -> Dict[str, Any]:
    report_dict.setdefault("summary", {})
    report_dict["summary"]["expected_bends"] = details.expected_bend_count
    report_dict["summary"]["expected_progress_pct"] = round(details.expected_progress_pct, 2)
    report_dict["summary"]["overdetected_vs_expected"] = details.overdetected_count
    completed_bends = int(report_dict["summary"].get("completed_bends", report_dict["summary"].get("detected", 0)))
    completed_in_spec = int(report_dict["summary"].get("completed_in_spec", report_dict["summary"].get("passed", 0)))
    completed_out_of_spec = max(0, completed_bends - completed_in_spec)
    remaining_expected = max(0, int(details.expected_bend_count) - min(completed_bends, int(details.expected_bend_count)))
    report_dict["summary"]["completed_bends"] = completed_bends
    report_dict["summary"]["completed_in_spec"] = completed_in_spec
    report_dict["summary"]["completed_out_of_spec"] = completed_out_of_spec
    report_dict["summary"]["remaining_bends"] = remaining_expected
    report_dict["summary"]["is_complete"] = remaining_expected == 0
    rolled_count = sum(1 for m in report_dict.get("matches", []) if m.get("bend_form") == "ROLLED")
    folded_count = sum(1 for m in report_dict.get("matches", []) if m.get("bend_form") == "FOLDED")
    report_dict["summary"]["rolled_bends"] = rolled_count
    report_dict["summary"]["folded_bends"] = folded_count
    if isinstance(report_dict.get("scan_quality"), dict):
        sq = report_dict["scan_quality"]
        report_dict["summary"]["scan_quality_status"] = sq.get("status")
        report_dict["summary"]["scan_coverage_pct"] = sq.get("coverage_pct")
        report_dict["summary"]["scan_density_pts_per_cm2"] = sq.get("density_pts_per_cm2")
    operator_actions = report_dict.get("operator_actions", [])
    critical_actions = [
        a for a in operator_actions
        if a.get("status") in {"FAIL", "NOT_DETECTED"}
    ]
    report_dict["summary"]["critical_actions"] = len(critical_actions)
    scan_quality = report_dict.get("scan_quality", {}) if isinstance(report_dict, dict) else {}
    quality_status = str(scan_quality.get("status", "")).strip().upper()
    quality_suffix = ""
    if quality_status:
        coverage_val = scan_quality.get("coverage_pct", None)
        density_val = scan_quality.get("density_pts_per_cm2", None)
        coverage_txt = f"{coverage_val:.0f}%" if isinstance(coverage_val, (int, float)) else "-"
        density_txt = f"{density_val:.0f}" if isinstance(density_val, (int, float)) else "-"
        quality_suffix = f" | Scan {quality_status} ({coverage_txt} cov, {density_txt} pts/cm^2)"
    report_dict["operator_brief"] = {
        "headline": (
            f"{completed_bends}/{details.expected_bend_count} completed, "
            f"{completed_in_spec} in spec, {completed_out_of_spec} out of spec, "
            f"{remaining_expected} remaining"
            f"{quality_suffix}"
        ),
        "actions": operator_actions[:6],
    }
    return report_dict


def _promote_structured_count_reporting(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    model_path = _latest_unary_model_path()
    if model_path is None:
        return report_dict
    try:
        bundle = load_unary_models(model_path)
        annotated = score_case_payload(report_dict, bundle)
    except Exception:
        return report_dict

    structured = (annotated.get("structured_context") or {}).get("count_posterior") or {}
    median_completed = structured.get("median_completed_bends")
    if median_completed is None:
        return annotated

    summary = annotated.setdefault("summary", {})
    expected_bends = int(summary.get("expected_bends") or summary.get("total_bends") or 0)
    direct_completed = int(summary.get("completed_bends", summary.get("detected", 0)) or 0)
    direct_remaining = int(summary.get("remaining_bends", max(0, expected_bends - direct_completed)) or 0)
    median_completed = int(median_completed)
    summary["match_evidence_completed_bends"] = direct_completed
    summary["match_evidence_remaining_bends"] = direct_remaining
    summary["structured_completed_bends"] = median_completed
    summary["structured_remaining_bends"] = max(0, expected_bends - min(expected_bends, median_completed))
    summary["structured_count_source"] = "posterior_median"
    summary["structured_count_mean"] = structured.get("mean_completed_bends")
    summary["structured_count_map"] = structured.get("map_completed_bends")
    summary["structured_count_delta_vs_match"] = median_completed - direct_completed

    brief = annotated.setdefault("operator_brief", {})
    headline = str(brief.get("headline") or "").strip()
    structured_suffix = (
        f" | Count model: {median_completed}/{expected_bends} complete"
        if expected_bends > 0
        else f" | Count model: {median_completed} complete"
    )
    evidence_suffix = f" | Direct evidence: {direct_completed} explicit bends"
    if structured_suffix not in headline:
        brief["headline"] = f"{headline}{structured_suffix}{evidence_suffix}".strip(" |")
    return annotated


def _export_aligned_scan_visualization(
    cad_path: str,
    scan_path: str,
    output_dir: Path,
    tolerance_angle: float,
    runtime_cfg: Any,
) -> Optional[Path]:
    import trimesh

    engine = ScanQCEngine(progress_callback=None)
    engine.load_reference(cad_path)
    engine.load_scan(scan_path)

    raw_scan_count = int(len(engine.scan_pcd.points)) if engine.scan_pcd is not None else 0
    voxel_size = 2.0 if raw_scan_count >= 650000 else 1.5 if raw_scan_count >= 450000 else 1.0
    engine.preprocess(voxel_size=voxel_size)

    seed = int(getattr(runtime_cfg, "local_refinement_kwargs", {}).get("deterministic_seed", 17))
    engine.align(auto_scale=True, tolerance=float(tolerance_angle), random_seed=seed)
    if engine.aligned_scan is None:
        return None

    aligned_path = output_dir / "aligned_scan.ply"
    points = np.asarray(engine.aligned_scan.points, dtype=np.float32)
    if len(points) == 0:
        return None
    colors = np.tile(np.array([[56, 189, 248, 210]], dtype=np.uint8), (len(points), 1))
    cloud = trimesh.PointCloud(points, colors=colors)
    cloud.export(str(aligned_path), file_type="ply")
    return aligned_path if aligned_path.exists() else None


def run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.time()
        _emit("PROGRESS", {"stage": "init", "progress": 2, "message": "Loading bend runtime configuration..."})
        runtime_cfg = load_bend_runtime_config()

        _emit("PROGRESS", {"stage": "pipeline", "progress": 25, "message": "Running progressive bend pipeline..."})
        report, details = run_progressive_bend_inspection(
            cad_path=args.cad_path,
            scan_path=args.scan_path,
            part_id=args.part_id,
            tolerance_angle=float(args.tolerance_angle),
            tolerance_radius=float(args.tolerance_radius),
            runtime_config=runtime_cfg,
        )
        report.processing_time_ms = (time.time() - started) * 1000.0

        _emit("PROGRESS", {"stage": "artifacts", "progress": 82, "message": "Building bend overlay artifacts..."})
        reference_mesh_path: Optional[Path] = None
        artifacts_payload: Dict[str, Any] = {}
        try:
            # Use the SAME CAD vertices that the pipeline used for bend detection.
            # This guarantees the reference mesh PLY and bend line coordinates
            # are in the exact same coordinate system (no dual-import mismatch).
            if details.cad_vertices is not None:
                export_reference_mesh_from_vertices(
                    vertices=details.cad_vertices,
                    triangles=details.cad_triangles,
                    output_dir=output_dir,
                )
            else:
                # Fallback: re-import CAD file (legacy path)
                export_bend_reference_visualization(
                    cad_path=args.cad_path,
                    output_dir=output_dir,
                    cad_import_deflection=runtime_cfg.cad_import_deflection,
                )
            # Free large arrays now that PLY is written — avoids holding
            # full CAD geometry in memory during artifact/aligned-scan phases.
            details.cad_vertices = None
            details.cad_triangles = None

            candidate = output_dir / "reference_mesh.ply"
            if candidate.exists():
                reference_mesh_path = candidate
        except Exception as exc:
            _emit("PROGRESS", {"stage": "artifacts", "progress": 86, "message": f"Reference mesh export warning: {_sanitize_error_message(exc)}"})

        try:
            aligned_scan_path = _export_aligned_scan_visualization(
                cad_path=args.cad_path,
                scan_path=args.scan_path,
                output_dir=output_dir,
                tolerance_angle=float(args.tolerance_angle),
                runtime_cfg=runtime_cfg,
            )
            if aligned_scan_path is not None:
                artifacts_payload["aligned_scan_url"] = f"/api/aligned-scan/{{job_id}}.ply"
        except Exception as exc:
            _emit("PROGRESS", {"stage": "artifacts", "progress": 87, "message": f"Aligned scan export warning: {_sanitize_error_message(exc)}"})

        report_dict = _json_safe(report.to_dict())
        report_dict = _augment_report_dict(report_dict, details)
        report_dict = _promote_structured_count_reporting(report_dict)

        if reference_mesh_path is not None:
            try:
                artifact_bundle = build_bend_artifact_bundle(
                    report_dict=report_dict,
                    reference_mesh_path=reference_mesh_path,
                    output_dir=output_dir,
                )
                artifacts_payload = _json_safe(artifact_bundle.artifacts)
            except Exception as exc:
                fallback = {"reference_mesh_url": f"/api/reference-mesh/{{job_id}}.ply"}
                artifacts_payload = fallback
                _emit("PROGRESS", {"stage": "artifacts", "progress": 88, "message": f"Overlay artifact warning: {_sanitize_error_message(exc)}"})

        _emit("PROGRESS", {"stage": "persist", "progress": 92, "message": "Writing bend inspection outputs..."})
        report_path = output_dir / "bend_inspection_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, allow_nan=False)

        table_path = output_dir / "bend_inspection_table.txt"
        table_path.write_text(report.to_table_string(), encoding="utf-8")

        payload = {
            "report_path": str(report_path),
            "table_path": str(table_path),
            "artifacts": artifacts_payload,
            "processing_time_ms": report.processing_time_ms,
            "pipeline_details": _json_safe(details.to_dict()),
            "runtime_config": _json_safe(runtime_cfg.to_dict()),
        }
        _emit("RESULT", payload)
        return 0
    except Exception as exc:
        clean_error = _sanitize_error_message(exc)
        _emit("ERROR", {"error": clean_error})
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_worker(args)


if __name__ == "__main__":
    exit_code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)
