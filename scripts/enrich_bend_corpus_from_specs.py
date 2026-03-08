#!/opt/homebrew/bin/python3.11
"""
Enrich bend corpus metadata with parsed XLSX bend-angle information.

This keeps spec-backed parts usable for regression without requiring manual
inspection of each spreadsheet on every run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path("/Users/idant/82Labs/Sherman QC/Full Project")
sys.path.insert(0, str(ROOT / "backend"))

from bend_inspection_pipeline import extract_cad_bend_specs_with_strategy, load_bend_runtime_config, load_cad_geometry  # noqa: E402
from dimension_parser import parse_dimension_file  # noqa: E402


CORPUS_ROOT = ROOT / "data" / "bend_corpus"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _angle_distance(spec_angle: float, cad_angle: float) -> float:
    return min(abs(cad_angle - spec_angle), abs((180.0 - cad_angle) - spec_angle))


def _coverage(spec_angles: List[float], cad_angles: List[float], threshold: float = 10.0) -> Dict[str, Any]:
    remaining = list(cad_angles)
    matched: List[Dict[str, Any]] = []
    unmatched_specs: List[float] = []
    for spec_angle in spec_angles:
        if not remaining:
            unmatched_specs.append(spec_angle)
            continue
        errors = [_angle_distance(spec_angle, cad_angle) for cad_angle in remaining]
        best_idx = min(range(len(errors)), key=errors.__getitem__)
        best_error = errors[best_idx]
        if best_error <= threshold:
            matched.append(
                {
                    "spec_angle": round(spec_angle, 3),
                    "cad_angle": round(remaining[best_idx], 3),
                    "error_deg": round(best_error, 3),
                }
            )
            remaining.pop(best_idx)
        else:
            unmatched_specs.append(spec_angle)
    mae = round(sum(m["error_deg"] for m in matched) / len(matched), 3) if matched else None
    return {
        "matched_count": len(matched),
        "unmatched_spec_count": len(unmatched_specs),
        "unmatched_cad_count": len(remaining),
        "matched": matched,
        "unmatched_spec_angles": [round(v, 3) for v in unmatched_specs],
        "mae_deg": mae,
    }


def _classify_spec_schedule(spec_angles: List[float], cad_bend_count: int, coverage: Dict[str, Any]) -> str:
    if not spec_angles:
        return "none"
    acute_ratio = float(sum(1 for angle in spec_angles if angle < 90.0)) / len(spec_angles)
    matched = int(coverage.get("matched_count") or 0)
    unmatched_spec = int(coverage.get("unmatched_spec_count") or 0)
    unmatched_cad = int(coverage.get("unmatched_cad_count") or 0)

    if len(spec_angles) <= 2 and cad_bend_count >= max(4, len(spec_angles) * 3):
        return "partial_angle_schedule"
    if acute_ratio >= 0.35 and unmatched_spec >= max(2, len(spec_angles) // 4):
        return "mixed_section_callouts"
    if unmatched_cad <= 2 and unmatched_spec <= 2:
        return "complete_or_near_complete"
    if matched >= 1 and unmatched_cad >= max(3, matched * 2):
        return "partial_angle_schedule"
    return "mixed_unknown"


def main() -> int:
    inventory = _load_json(CORPUS_ROOT / "dataset_inventory.json")
    runtime_cfg = load_bend_runtime_config(None)
    summary: List[Dict[str, Any]] = []

    for part in inventory.get("parts", []):
        spec_files = part.get("spec_files") or []
        cad_files = part.get("cad_files") or []
        if not spec_files or not cad_files:
            continue

        metadata_path = CORPUS_ROOT / "parts" / part["part_key"] / "metadata.json"
        metadata = _load_json(metadata_path)
        prior_schedule_type = None
        if isinstance(metadata.get("spec_analysis"), dict):
            prior_schedule_type = metadata["spec_analysis"].get("schedule_type")

        spec_path = spec_files[0]["path"]
        spec_result = parse_dimension_file(spec_path)
        spec_angles = [float(dim.value) for dim in spec_result.bend_angles if dim.value is not None]

        cad_path = cad_files[0]["path"]
        cad_vertices, cad_triangles, _ = load_cad_geometry(cad_path, cad_import_deflection=runtime_cfg.cad_import_deflection)
        cad_specs, strategy = extract_cad_bend_specs_with_strategy(
            cad_vertices=cad_vertices,
            cad_triangles=cad_triangles,
            tolerance_angle=1.0,
            tolerance_radius=0.5,
            cad_extractor_kwargs=runtime_cfg.cad_extractor_kwargs,
            cad_detector_kwargs=runtime_cfg.cad_detector_kwargs,
            cad_detect_call_kwargs=runtime_cfg.cad_detect_call_kwargs,
            expected_bend_count_hint=part.get("expected_bends_override"),
            spec_bend_angles_hint=spec_angles,
            spec_schedule_type=prior_schedule_type,
        )
        cad_angles = [float(spec.target_angle) for spec in cad_specs]
        coverage = _coverage(spec_angles, cad_angles)
        schedule_type = _classify_spec_schedule(spec_angles, len(cad_specs), coverage)

        metadata["spec_analysis"] = {
            "spec_file": spec_path,
            "parse_success": bool(spec_result.success),
            "dimension_count": len(spec_result.dimensions),
            "bend_angle_count": len(spec_result.bend_angles),
            "bend_angles_deg": [round(v, 3) for v in spec_angles],
            "bend_radius": spec_result.bend_radius,
            "warnings": list(spec_result.warnings),
            "cad_strategy_with_spec_hint": strategy,
            "cad_bend_count_with_spec_hint": len(cad_specs),
            "cad_bend_angles_deg": [round(v, 3) for v in cad_angles],
            "cad_spec_coverage": coverage,
            "schedule_type": schedule_type,
        }
        _dump_json(metadata_path, metadata)
        summary.append(
            {
                "part_key": part["part_key"],
                "bend_angle_count": len(spec_angles),
                "cad_bend_count": len(cad_specs),
                "cad_strategy_with_spec_hint": strategy,
                "matched_spec_angles": coverage["matched_count"],
                "mae_deg": coverage["mae_deg"],
                "schedule_type": schedule_type,
            }
        )

    (CORPUS_ROOT / "spec_enrichment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
