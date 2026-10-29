#!/usr/bin/env python3
"""
Run one bend corpus regression case in an isolated process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from bend_unary_model import load_unary_models, score_case_payload  # noqa: E402
from bend_inspection_pipeline import load_bend_runtime_config, run_progressive_bend_inspection  # noqa: E402
from domains.bend.services.heatmap_consistency_service import (  # noqa: E402
    evaluate_bend_heatmap_consistency,
    merge_heatmap_consistency_into_report,
)
from domains.bend.services.heatmap_evidence_service import build_bend_heatmap_evidence  # noqa: E402
from evaluate_bend_corpus import _heatmap_consistency_metrics, _score_scan, _structured_score_scan  # noqa: E402


def _latest_unary_model_path() -> Path | None:
    latest = ROOT / "output" / "unary_models" / "latest.json"
    if not latest.exists():
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidate = payload.get("model_path") or payload.get("path")
    if not candidate:
        latest_dir = payload.get("latest_bootstrap_dir")
        if latest_dir:
            model_path = Path(str(latest_dir)) / "bend_unary_models.joblib"
            return model_path if model_path.exists() else None
        return None
    model_path = Path(str(candidate))
    if model_path.is_dir():
        model_path = model_path / "bend_unary_models.joblib"
    return model_path if model_path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one bend corpus scan case.")
    parser.add_argument("--part-key", required=True)
    parser.add_argument(
        "--inference-part-id",
        default="",
        help="Identity exposed to the pipeline; blind runners should provide an anonymized value.",
    )
    parser.add_argument("--cad-path", required=True)
    parser.add_argument("--scan-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--state", default="unknown")
    parser.add_argument("--expected-total")
    parser.add_argument("--expected-completed")
    parser.add_argument("--spec-bend-angles-json", default="[]")
    parser.add_argument("--spec-schedule-type", default="")
    parser.add_argument("--skip-heatmap-consistency", action="store_true")
    parser.add_argument("--heatmap-cache-dir", default="")
    parser.add_argument(
        "--inference-mode",
        choices=("blind_cad_ply", "operator_assisted"),
        default="blind_cad_ply",
        help=(
            "blind_cad_ply keeps expected counts and part-specific truth policies scoring-only; "
            "operator_assisted reproduces the production workflow with drawing/operator hints."
        ),
    )
    args = parser.parse_args()

    expected_total = None if args.expected_total in {None, "", "None"} else int(float(args.expected_total))
    expected_completed = None if args.expected_completed in {None, "", "None"} else int(float(args.expected_completed))
    spec_bend_angles: List[float] = [float(v) for v in json.loads(args.spec_bend_angles_json or "[]")]

    runtime_config = load_bend_runtime_config(None)
    total_start = perf_counter()
    bend_start = perf_counter()
    blind_cad_ply = args.inference_mode == "blind_cad_ply"
    inference_part_id = str(args.inference_part_id or args.part_key)
    report, details = run_progressive_bend_inspection(
        cad_path=args.cad_path,
        scan_path=args.scan_path,
        part_id=inference_part_id,
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        runtime_config=runtime_config,
        spec_bend_angles_hint=None if blind_cad_ply else spec_bend_angles,
        spec_schedule_type=None if blind_cad_ply else (args.spec_schedule_type or None),
        expected_bend_override=None if blind_cad_ply else expected_total,
        scan_state=(args.state or None),
        use_part_truth_hints=not blind_cad_ply,
    )
    bend_elapsed_sec = perf_counter() - bend_start
    report_dict = report.to_dict()
    heatmap_cache_dir = Path(args.heatmap_cache_dir) if str(args.heatmap_cache_dir or "").strip() else None
    heatmap_elapsed_sec = None
    heatmap_error = None
    heatmap_cache_hit = None
    if not args.skip_heatmap_consistency:
        heatmap_start = perf_counter()
        try:
            heatmap_bundle = build_bend_heatmap_evidence(
                args.cad_path,
                args.scan_path,
                tolerance_angle=1.0,
                runtime_cfg=runtime_config,
                output_dir=None,
                cache_dir=heatmap_cache_dir,
            )
            heatmap_cache_hit = bool(heatmap_bundle.get("cache_hit"))
            aligned_points = heatmap_bundle.get("aligned_points")
            deviations = heatmap_bundle.get("deviations")
            if aligned_points is not None and deviations is not None and len(aligned_points) == len(deviations):
                consistency = evaluate_bend_heatmap_consistency(
                    report,
                    aligned_points,
                    deviations,
                    scan_state=(args.state or None),
                    scan_name=Path(args.scan_path).name,
                    global_tolerance_mm=0.5,
                )
                report_dict = merge_heatmap_consistency_into_report(report_dict, consistency)
        except Exception as exc:
            heatmap_error = str(exc)
        finally:
            heatmap_elapsed_sec = perf_counter() - heatmap_start
    payload: Dict[str, Any] = {
        "part_key": args.part_key,
        "cad_path": args.cad_path,
        "scan_path": args.scan_path,
        "scan_name": Path(args.scan_path).name,
        "expectation": {
            "state": args.state,
            "expected_total_bends": expected_total,
            "expected_completed_bends": expected_completed,
        },
        "spec_bend_angles_hint": spec_bend_angles,
        "spec_schedule_type": args.spec_schedule_type or None,
        "inference_mode": args.inference_mode,
        "inference_part_id": inference_part_id,
        "summary": report_dict.get("summary") or {},
        "details": details.to_dict(),
        "timing": {
            "bend_report_sec": round(bend_elapsed_sec, 3),
            "heatmap_consistency_sec": round(heatmap_elapsed_sec, 3) if heatmap_elapsed_sec is not None else None,
            "total_sec": round(perf_counter() - total_start, 3),
            "heatmap_consistency_skipped": bool(args.skip_heatmap_consistency),
            "heatmap_consistency_error": heatmap_error,
            "heatmap_cache_dir": str(heatmap_cache_dir) if heatmap_cache_dir is not None else None,
            "heatmap_cache_hit": heatmap_cache_hit,
        },
        "warnings": details.warnings,
        "matches": report_dict.get("matches") or [],
        "alignment_metadata": dict(getattr(report, "alignment_metadata", {}) or {}),
    }
    payload["metrics"] = _score_scan(payload, payload["expectation"])
    unary_model_path = _latest_unary_model_path()
    if unary_model_path is not None:
        payload = score_case_payload(payload, load_unary_models(unary_model_path))
    payload["structured_metrics"] = _structured_score_scan(payload, payload["expectation"])
    payload["heatmap_consistency_metrics"] = _heatmap_consistency_metrics(payload, payload["expectation"])
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    gc.collect()
    print(json.dumps({"event": "case_done", "part_key": args.part_key, "scan_name": Path(args.scan_path).name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
