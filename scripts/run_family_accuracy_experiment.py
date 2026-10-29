#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / 'Downloads'
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'scripts'))

from analyze_aligned_surface_coverage import (  # noqa: E402
    _cluster_face_patches,
    _sample_points,
    _summarize_bend_zones,
    _summarize_patches,
    _projection_plot,
)
from bend_inspection_pipeline import (  # noqa: E402
    _apply_scan_runtime_profile,
    _fallback_scan_detection_config,
    detect_scan_bends,
    extract_cad_bend_specs_with_strategy,
    load_bend_runtime_config,
    load_cad_geometry,
    load_scan_points,
    match_detected_bends,
    run_progressive_bend_inspection,
)
from domains.bend.services.heatmap_consistency_service import evaluate_bend_heatmap_consistency  # noqa: E402
from domains.bend.services.heatmap_evidence_service import build_bend_heatmap_evidence  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

BASELINE_SUMMARY = ROOT / 'data/bend_corpus/evaluation/benchmark_stability_core_20260330_partial_to_full_retention_v1/summary.json'
TARGET_CASES = [
    {
        'family': '28870000',
        'scan_label': 'full_reference',
        'part_id': '28870000',
        'cad_path': str(DOWNLOADS / '28870000_A.stp'),
        'scan_path': str(DOWNLOADS / '28870000_A_FULL_SCAN.ply'),
        'state': 'full',
        'expected_total': 2,
        'expected_completed': 2,
    },
    {
        'family': '28870000',
        'scan_label': 'part1_candidate',
        'part_id': '28870000',
        'cad_path': str(DOWNLOADS / '28870000_A.stp'),
        'scan_path': str(DOWNLOADS / 'part 1.ply'),
        'state': 'unknown',
        'expected_total': 2,
        'expected_completed': 2,
    },
    {
        'family': '23553000',
        'scan_label': 'partial_scan_1',
        'part_id': '23553000',
        'cad_path': str(DOWNLOADS / '23553000_B.stp'),
        'scan_path': str(DOWNLOADS / '23553000_B_SCAN_1.ply'),
        'state': 'partial',
        'expected_total': 16,
        'expected_completed': 4,
    },
    {
        'family': '23553000',
        'scan_label': 'partial_scan_2',
        'part_id': '23553000',
        'cad_path': str(DOWNLOADS / '23553000_B.stp'),
        'scan_path': str(DOWNLOADS / '23553000_B_SCAN_2.ply'),
        'state': 'partial',
        'expected_total': 16,
        'expected_completed': 9,
    },
    {
        'family': '23553000',
        'scan_label': 'full_scan',
        'part_id': '23553000',
        'cad_path': str(DOWNLOADS / '23553000_B.stp'),
        'scan_path': str(DOWNLOADS / '23553000_B_FULL_SCAN 1.ply'),
        'state': 'full',
        'expected_total': 16,
        'expected_completed': 16,
    },
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _load_baseline_metrics() -> Dict[str, Any]:
    payload = json.loads(BASELINE_SUMMARY.read_text(encoding='utf-8'))
    return payload.get('aggregate') or {}


def _coverage_summary(cad_path: str, scan_path: str, output_dir: Path, runtime_cfg: Any) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_bend_heatmap_evidence(
        cad_path,
        scan_path,
        tolerance_angle=1.0,
        runtime_cfg=runtime_cfg,
        output_dir=output_dir,
        cache_dir=None,
    )
    aligned_points = np.asarray(bundle.get('aligned_points'), dtype=np.float64)
    if aligned_points.size == 0:
        return {'error': 'aligned_points_unavailable'}
    cad_vertices, cad_faces, cad_source = load_cad_geometry(cad_path)
    mesh = trimesh.Trimesh(vertices=np.asarray(cad_vertices, dtype=np.float64), faces=np.asarray(cad_faces, dtype=np.int32), process=False)
    sample_points, sample_face_indices = _sample_points(mesh, 120000, 17)
    tree = cKDTree(aligned_points)
    distances, _ = tree.query(sample_points, k=1, workers=-1)
    thresholds = [0.5, 1.0, 2.0]
    covered_by_threshold = {f'{thr:.2f}mm': (distances <= thr) for thr in thresholds}
    patch_labels = _cluster_face_patches(mesh, 8.0)
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)
    patch_summaries = _summarize_patches(sample_face_indices, patch_labels, covered_by_threshold, face_area)
    significant_patches = [
        patch for patch in patch_summaries
        if int(patch.get('sample_count', 0)) >= 25 and float(patch.get('surface_area_mm2', 0.0)) >= 25.0
    ]
    cad_specs, _ = extract_cad_bend_specs_with_strategy(
        cad_vertices=np.asarray(cad_vertices, dtype=np.float64),
        cad_triangles=np.asarray(cad_faces, dtype=np.int32),
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        cad_extractor_kwargs=getattr(runtime_cfg, 'cad_extractor_kwargs', {}) or {},
        cad_detector_kwargs=getattr(runtime_cfg, 'cad_detector_kwargs', {}) or {},
        cad_detect_call_kwargs=getattr(runtime_cfg, 'cad_detect_call_kwargs', {}) or {},
        feature_policy=getattr(runtime_cfg, 'feature_policy', None),
    )
    bend_zone_summaries = _summarize_bend_zones(sample_points, covered_by_threshold, cad_specs)
    projection_mask = covered_by_threshold['1.00mm']
    projection_path = output_dir / 'aligned_surface_coverage_projection.png'
    _projection_plot(sample_points, projection_mask, projection_path, 'Aligned CAD coverage @ 1.00 mm')
    summary = {
        'cad_source': cad_source,
        'distance_stats_mm': {
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'p95': float(np.percentile(distances, 95)),
            'max': float(np.max(distances)),
        },
        'global_surface_coverage': {key: float(np.mean(mask)) for key, mask in covered_by_threshold.items()},
        'top_low_coverage_patches': sorted(
            significant_patches,
            key=lambda entry: (entry['coverage'].get('1.00mm', 1.0), -entry['surface_area_mm2'])
        )[:12],
        'bend_zone_coverage': sorted(
            bend_zone_summaries,
            key=lambda entry: entry['coverage'].get('1.00mm', 1.0)
        ),
        'artifacts': {
            'projection_path': str(projection_path),
            'aligned_scan_path': str(bundle.get('aligned_scan_path')) if bundle.get('aligned_scan_path') is not None else None,
        },
    }
    (output_dir / 'aligned_surface_coverage_summary.json').write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    return _jsonable(summary)


def _raw_detection_debug(scan_points: np.ndarray, cad_bends: List[Any], runtime_cfg: Any, part_id: str, scan_state: str) -> Dict[str, Any]:
    primary_ctor, primary_call, _ = _apply_scan_runtime_profile(
        runtime_cfg.scan_detector_kwargs,
        runtime_cfg.scan_detect_call_kwargs,
        len(scan_points),
        scan_state=scan_state,
        fallback=False,
    )
    primary_detected = detect_scan_bends(scan_points=scan_points, scan_detector_kwargs=primary_ctor, scan_detect_call_kwargs=primary_call)
    primary_report = match_detected_bends(detected_bends=primary_detected, cad_bends=cad_bends, part_id=part_id, matcher_kwargs=runtime_cfg.matcher_kwargs)
    primary_report.compute_summary()

    fb_ctor, fb_call = _fallback_scan_detection_config(runtime_cfg)
    fb_ctor, fb_call, _ = _apply_scan_runtime_profile(fb_ctor, fb_call, len(scan_points), scan_state=scan_state, fallback=True)
    fallback_detected = detect_scan_bends(scan_points=scan_points, scan_detector_kwargs=fb_ctor, scan_detect_call_kwargs=fb_call)
    fallback_report = match_detected_bends(detected_bends=fallback_detected, cad_bends=cad_bends, part_id=part_id, matcher_kwargs=runtime_cfg.matcher_kwargs)
    fallback_report.compute_summary()

    return {
        'primary_detection_count': int(len(primary_detected)),
        'fallback_detection_count': int(len(fallback_detected)),
        'primary_detected_bends': [_jsonable(b.to_dict()) for b in primary_detected],
        'fallback_detected_bends': [_jsonable(b.to_dict()) for b in fallback_detected],
        'primary_match_summary': _jsonable(primary_report.to_dict().get('summary') or {}),
        'fallback_match_summary': _jsonable(fallback_report.to_dict().get('summary') or {}),
    }


def _target_debug_cards(case: Dict[str, Any], report_dict: Dict[str, Any], aligned_coverage: Dict[str, Any], heatmap_consistency: Dict[str, Any]) -> List[Dict[str, Any]]:
    coverage_by_bend = {
        str(item.get('bend_id')): item
        for item in (aligned_coverage.get('bend_zone_coverage') or [])
    }
    consistency_by_bend = {
        str(item.get('bend_id')): item
        for item in ((heatmap_consistency.get('per_bend') or {}) if isinstance(heatmap_consistency.get('per_bend'), dict) else {})
    }
    cards: List[Dict[str, Any]] = []
    for match in report_dict.get('matches') or []:
        if not bool(match.get('countable_in_regression', True)):
            continue
        bend_id = str(match.get('bend_id'))
        heatmap_result = (heatmap_consistency.get('per_bend') or {}).get(bend_id, {}) if isinstance(heatmap_consistency.get('per_bend'), dict) else {}
        cards.append({
            'target_id': bend_id,
            'scan_name': case['scan_label'],
            'feature_family': match.get('feature_family'),
            'is_countable': bool(match.get('countable_in_regression', True)),
            'nominal_angle_deg': match.get('target_angle'),
            'status': match.get('status'),
            'observability_state': match.get('observability_state'),
            'observability_detail_state': match.get('observability_detail_state'),
            'aligned_zone_coverage_1mm': (coverage_by_bend.get(bend_id, {}).get('coverage') or {}).get('1.00mm'),
            'candidate_count': match.get('assignment_candidate_count'),
            'best_candidate_id': match.get('assignment_candidate_id'),
            'best_candidate_kind': match.get('assignment_candidate_kind'),
            'best_candidate_score': match.get('assignment_candidate_score'),
            'null_candidate_score': match.get('assignment_null_score'),
            'assignment_outcome': match.get('assignment_source'),
            'local_metrology_outcome': {
                'measurement_mode': match.get('measurement_mode'),
                'measurement_method': match.get('measurement_method'),
                'measured_angle_deg': match.get('measured_angle'),
                'measurement_context': match.get('measurement_context'),
            },
            'heatmap_roi_support': {
                'status': heatmap_result.get('status'),
                'consistency_score': heatmap_result.get('consistency_score'),
                'roi_point_count': heatmap_result.get('roi_point_count'),
                'roi_coverage_ratio': heatmap_result.get('roi_coverage_ratio'),
                'local_abs_p95_mm': heatmap_result.get('local_abs_p95_mm'),
            },
            'final_interpretation': {
                'semantically_correct': None,
                'notes': [],
            },
        })
    return cards


def _run_case(case: Dict[str, Any], output_dir: Path, runtime_cfg: Any) -> Dict[str, Any]:
    case_dir = output_dir / case['family'] / case['scan_label']
    case_dir.mkdir(parents=True, exist_ok=True)

    report, details = run_progressive_bend_inspection(
        cad_path=case['cad_path'],
        scan_path=case['scan_path'],
        part_id=case['part_id'],
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        runtime_config=runtime_cfg,
        expected_bend_override=case['expected_total'],
        scan_state=case['state'],
        use_part_truth_hints=True,
    )
    report_dict = _jsonable(report.to_dict())
    details_dict = _jsonable(details.to_dict())

    aligned_coverage = _coverage_summary(case['cad_path'], case['scan_path'], case_dir / 'aligned_coverage', runtime_cfg)

    heatmap_output_dir = case_dir / 'heatmap_bundle'
    heatmap_output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_bundle = build_bend_heatmap_evidence(
        case['cad_path'],
        case['scan_path'],
        tolerance_angle=1.0,
        runtime_cfg=runtime_cfg,
        output_dir=heatmap_output_dir,
        cache_dir=None,
    )
    heatmap_consistency = {}
    if heatmap_bundle.get('aligned_points') is not None and heatmap_bundle.get('deviations') is not None:
        consistency = evaluate_bend_heatmap_consistency(
            report,
            np.asarray(heatmap_bundle['aligned_points']),
            np.asarray(heatmap_bundle['deviations']),
            scan_state=case['state'],
            scan_name=Path(case['scan_path']).name,
            global_tolerance_mm=0.5,
        )
        heatmap_consistency = _jsonable(consistency)

    cad_vertices, cad_triangles, _ = load_cad_geometry(case['cad_path'])
    cad_bends, cad_strategy = extract_cad_bend_specs_with_strategy(
        cad_vertices=np.asarray(cad_vertices, dtype=np.float64),
        cad_triangles=np.asarray(cad_triangles, dtype=np.int32),
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        cad_extractor_kwargs=getattr(runtime_cfg, 'cad_extractor_kwargs', {}) or {},
        cad_detector_kwargs=getattr(runtime_cfg, 'cad_detector_kwargs', {}) or {},
        cad_detect_call_kwargs=getattr(runtime_cfg, 'cad_detect_call_kwargs', {}) or {},
        feature_policy=None,
        part_id=case['part_id'],
    )
    raw_scan_points = load_scan_points(case['scan_path'])
    detection_debug = _raw_detection_debug(raw_scan_points, cad_bends, runtime_cfg, case['part_id'], case['state'])

    target_debug_cards = _target_debug_cards(case, report_dict, aligned_coverage, heatmap_consistency)

    payload = {
        'case': case,
        'report': report_dict,
        'details': details_dict,
        'cad_strategy': cad_strategy,
        'cad_targets': [_jsonable(b.to_dict()) for b in cad_bends],
        'raw_detection_debug': detection_debug,
        'aligned_coverage': aligned_coverage,
        'heatmap_consistency': heatmap_consistency,
        'target_debug_cards': target_debug_cards,
    }
    (case_dir / 'experiment_case.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def _write_iteration_summary(output_dir: Path, iteration_payload: Dict[str, Any]) -> None:
    summary_path = output_dir / 'iteration_summary.json'
    summary_path.write_text(json.dumps(_jsonable(iteration_payload), indent=2, ensure_ascii=False), encoding='utf-8')


def _append_ledger(iteration_name: str, output_dir: Path, iteration_payload: Dict[str, Any]) -> None:
    ledger_path = ROOT / 'docs/research/RECURSIVE_ACCURACY_LOOP_LEDGER_20260330.md'
    lines = []
    if ledger_path.exists():
        lines.append(ledger_path.read_text(encoding='utf-8').rstrip())
    else:
        lines.append('# Recursive Accuracy Loop Ledger\n')
    lines.append(f"\n## {iteration_name}\n")
    lines.append(f"- output: `{output_dir}`")
    lines.append(f"- baseline structured total MAE: `{iteration_payload['baseline_metrics'].get('structured_mae_vs_expected_total_completed')}`")
    lines.append(f"- baseline structured completed MAE: `{iteration_payload['baseline_metrics'].get('structured_mae_vs_expected_completed')}`")
    for case in iteration_payload['cases']:
        report = case['report']
        lines.append(f"- {case['case']['family']} / {case['case']['scan_label']}: detected `{report['summary'].get('detected')}` of `{report['summary'].get('total_bends')}`, countable completed `{report['summary'].get('countable_completed_bends', report['summary'].get('completed_bends'))}`")
    ledger_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run frozen family accuracy experiment pack.')
    parser.add_argument('--iteration-name', default=f"iteration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()

    runtime_cfg = load_bend_runtime_config(None)
    output_dir = ROOT / 'output' / 'family_accuracy_loop' / args.iteration_name
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _load_baseline_metrics()
    cases_payload = []
    for case in TARGET_CASES:
        cases_payload.append(_run_case(copy.deepcopy(case), output_dir, runtime_cfg))
        gc.collect()

    iteration_payload = {
        'iteration_name': args.iteration_name,
        'generated_at': datetime.now().isoformat(),
        'baseline_metrics': baseline_metrics,
        'cases': cases_payload,
    }
    _write_iteration_summary(output_dir, iteration_payload)
    _append_ledger(args.iteration_name, output_dir, iteration_payload)
    print(json.dumps({'iteration_name': args.iteration_name, 'output_dir': str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
