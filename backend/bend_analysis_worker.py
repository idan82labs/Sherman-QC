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

from domains.bend.application.finalize_bend_report import finalize_bend_report, format_bend_decision_table
from domains.bend.services.promotion_readiness import load_program_status
from domains.bend.services.heatmap_evidence_service import build_bend_heatmap_evidence
from domains.bend.services.heatmap_consistency_service import (
    evaluate_bend_heatmap_consistency,
    merge_heatmap_consistency_into_report,
)
from domains.bend.services.report_enrichment_service import augment_progressive_bend_report
from domains.bend.services.structured_count_reporting_service import promote_structured_count_reporting
from domains.bend.services.runtime_service import (
    export_bend_reference_visualization,
    export_reference_mesh_from_vertices,
    load_bend_runtime_config,
    run_progressive_bend_inspection,
)
from domains.reporting.services.bend_report_service import build_bend_artifact_bundle


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


def _load_qualification_context() -> Dict[str, Any]:
    """Share the API's non-authorizing status projection inside the worker."""
    default = Path(__file__).resolve().parents[1] / "docs/research/artifacts/f1_program_status_current.json"
    source = Path(os.getenv("SHERMAN_QC_F1_PROGRAM_STATUS_PATH", str(default))).expanduser()
    result = load_program_status(source)
    return {**result, "promotion_status": result["status"]}


def _guard_table_text(table_text: str, qualification: Dict[str, Any]) -> str:
    # No legacy mapping, even one containing a true flag, verifies a release.
    return (
        "RESEARCH ONLY - NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT\n"
        "Per-bend PASS/FAIL values are experimental measurements.\n\n"
        + table_text
    )


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"{tag}\t{json.dumps(payload, ensure_ascii=False)}", flush=True)


def _resolve_scan_state(scan_name: str, details: Any, report_dict: Dict[str, Any]) -> Optional[str]:
    # Filename and expected/completed bend counts are forbidden inference
    # inputs.  Only an explicit source-observation state may open full/partial
    # behavior; otherwise downstream logic must abstain.
    explicit = str(getattr(details, "scan_state", "") or "").strip().lower()
    if explicit in {"full", "partial"}:
        return explicit
    return None




def _latest_unary_model_path() -> Optional[Path]:
    """Compatibility shim during worker-to-domain extraction."""
    from domains.bend.services.structured_count_reporting_service import latest_unary_model_path

    return latest_unary_model_path()


def _promote_structured_count_reporting(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility shim during worker-to-domain extraction."""
    return promote_structured_count_reporting(
        report_dict,
        latest_pointer_path=_latest_unary_model_path(),
    )

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


def _export_aligned_scan_visualization(
    cad_path: str,
    scan_path: str,
    output_dir: Path,
    tolerance_angle: float,
    runtime_cfg: Any,
) -> Dict[str, Any]:
    return build_bend_heatmap_evidence(
        cad_path,
        scan_path,
        tolerance_angle=tolerance_angle,
        runtime_cfg=runtime_cfg,
        output_dir=output_dir,
    )


def run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.time()
        _emit("PROGRESS", {"stage": "init", "progress": 2, "message": "Loading bend runtime configuration..."})
        runtime_cfg = load_bend_runtime_config()

        _emit("PROGRESS", {"stage": "alignment", "progress": 8, "message": "Aligning scan once for bend analysis and heatmap..."})
        scan_bundle = _export_aligned_scan_visualization(
            cad_path=args.cad_path, scan_path=args.scan_path, output_dir=output_dir,
            tolerance_angle=float(args.tolerance_angle), runtime_cfg=runtime_cfg,
        )
        from domains.bend.services.inspection_frame import validate_aligned_points
        aligned_points = validate_aligned_points(scan_bundle.get("aligned_points"), scan_bundle.get("alignment"))

        _emit("PROGRESS", {"stage": "pipeline", "progress": 25, "message": "Running progressive bend pipeline..."})
        report, details = run_progressive_bend_inspection(
            cad_path=args.cad_path,
            scan_path=args.scan_path,
            part_id=args.part_id,
            tolerance_angle=float(args.tolerance_angle),
            tolerance_radius=float(args.tolerance_radius),
            runtime_config=runtime_cfg,
            spec_bend_angles_hint=None,
            spec_schedule_type=None,
            expected_bend_override=None,
            use_part_truth_hints=False,
            aligned_scan_points=aligned_points,
            scan_alignment=scan_bundle["alignment"],
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

        aligned_scan_path: Optional[Path] = None
        aligned_points: Optional[np.ndarray] = None
        deviations_path: Optional[Path] = None
        deviation_stats: Optional[Dict[str, Any]] = None
        try:
            # Reuse the exact registration consumed by bend detection. Never
            # independently align the display cloud after computing markers.
            aligned_scan_path = scan_bundle.get("aligned_scan_path")
            aligned_points = scan_bundle.get("aligned_points")
            deviations_path = scan_bundle.get("deviations_path")
            deviation_stats = scan_bundle.get("deviation_stats")
            if aligned_scan_path is not None:
                artifacts_payload["aligned_scan_url"] = f"/api/aligned-scan/{{job_id}}.ply"
        except Exception as exc:
            _emit("PROGRESS", {"stage": "artifacts", "progress": 87, "message": f"Aligned scan export warning: {_sanitize_error_message(exc)}"})

        report_dict = _json_safe(report.to_dict())
        report_dict["alignment"] = scan_bundle["alignment"]
        report_dict = augment_progressive_bend_report(report_dict, details)
        report_dict = promote_structured_count_reporting(report_dict)
        report_dict["qualification_context"] = _load_qualification_context()
        if aligned_scan_path is not None and deviations_path is not None:
            try:
                deviations = scan_bundle.get("deviations")
                if deviations is None:
                    deviations = np.load(str(deviations_path))
            except Exception:
                deviations = None
            if aligned_points is not None and deviations is not None and len(aligned_points) == len(deviations):
                consistency_bundle = evaluate_bend_heatmap_consistency(
                    report,
                    aligned_points,
                    deviations,
                    scan_state=_resolve_scan_state(Path(args.scan_path).name, details, report_dict),
                    scan_name=Path(args.scan_path).name,
                    global_tolerance_mm=float(args.tolerance_radius),
                )
                report_dict = merge_heatmap_consistency_into_report(report_dict, consistency_bundle)

        # The application projection is additive; raw scientific/legacy fields
        # remain available, but cannot authorize physical completion/disposition.
        report_dict = finalize_bend_report(report_dict)
        if reference_mesh_path is not None:
            try:
                artifact_bundle = build_bend_artifact_bundle(
                    report_dict=report_dict,
                    reference_mesh_path=reference_mesh_path,
                    output_dir=output_dir,
                )
                artifacts_payload = {
                    **artifacts_payload,
                    **_json_safe(artifact_bundle.artifacts),
                }
            except Exception as exc:
                fallback = {"reference_mesh_url": f"/api/reference-mesh/{{job_id}}.ply"}
                artifacts_payload = {
                    **artifacts_payload,
                    **fallback,
                }
                _emit("PROGRESS", {"stage": "artifacts", "progress": 88, "message": f"Overlay artifact warning: {_sanitize_error_message(exc)}"})

        report_dict = finalize_bend_report(
            report_dict,
            render_artifacts_available=bool(artifacts_payload.get("bend_overlay_manifest_url")),
        )
        _emit("PROGRESS", {"stage": "persist", "progress": 92, "message": "Writing bend inspection outputs..."})
        report_path = output_dir / "bend_inspection_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, allow_nan=False)

        table_path = output_dir / "bend_inspection_table.txt"
        table_path.write_text(
            format_bend_decision_table(report_dict),
            encoding="utf-8",
        )

        payload = {
            "report_path": str(report_path),
            "table_path": str(table_path),
            "artifacts": artifacts_payload,
            "processing_time_ms": report.processing_time_ms,
            "pipeline_details": _json_safe(details.to_dict()),
            "runtime_config": _json_safe(runtime_cfg.to_dict()),
            "deviation_stats": _json_safe(deviation_stats),
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
