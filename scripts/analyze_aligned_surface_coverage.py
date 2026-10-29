#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'scripts'))

from bend_inspection_pipeline import (  # noqa: E402
    extract_cad_bend_specs_with_strategy,
    load_bend_runtime_config,
    load_cad_geometry,
)
from domains.bend.services.heatmap_evidence_service import build_bend_heatmap_evidence  # noqa: E402


def _sample_points(mesh: trimesh.Trimesh, sample_count: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64), np.asarray(face_indices, dtype=np.int32)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _cluster_face_patches(mesh: trimesh.Trimesh, normal_angle_deg: float) -> np.ndarray:
    face_count = len(mesh.faces)
    if face_count == 0:
        return np.empty((0,), dtype=np.int32)
    uf = _UnionFind(face_count)
    cos_thresh = math.cos(math.radians(normal_angle_deg))
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    for a, b in np.asarray(mesh.face_adjacency, dtype=np.int32):
        if np.dot(normals[a], normals[b]) >= cos_thresh:
            uf.union(int(a), int(b))
    roots = [uf.find(i) for i in range(face_count)]
    label_map: Dict[int, int] = {}
    labels = np.empty((face_count,), dtype=np.int32)
    next_label = 0
    for i, root in enumerate(roots):
        label = label_map.get(root)
        if label is None:
            label = next_label
            label_map[root] = label
            next_label += 1
        labels[i] = label
    return labels


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    seg = end - start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1e-9:
        return np.linalg.norm(points - start[None, :], axis=1)
    rel = points - start[None, :]
    t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
    proj = start[None, :] + t[:, None] * seg[None, :]
    return np.linalg.norm(points - proj, axis=1)


def _projection_plot(points: np.ndarray, covered: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d1117')
    fig.patch.set_facecolor('#0d1117')
    views = [((0, 1), 'XY'), ((0, 2), 'XZ'), ((1, 2), 'YZ')]
    colors = np.where(covered[:, None], np.array([[64, 196, 99]]) / 255.0, np.array([[255, 95, 86]]) / 255.0)
    for ax, (dims, label) in zip(axes, views):
        ax.set_facecolor('#0d1117')
        ax.scatter(points[:, dims[0]], points[:, dims[1]], c=colors, s=1.2, linewidths=0, alpha=0.9)
        ax.set_title(label, color='white', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#444')
    fig.suptitle(title, color='white', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def _summarize_patches(
    sample_face_indices: np.ndarray,
    patch_labels: np.ndarray,
    covered_by_threshold: Dict[str, np.ndarray],
    face_area: np.ndarray,
) -> List[Dict[str, Any]]:
    patch_face_ids: Dict[int, List[int]] = defaultdict(list)
    for face_idx, patch_id in enumerate(patch_labels.tolist()):
        patch_face_ids[patch_id].append(face_idx)

    sample_patch_ids = patch_labels[sample_face_indices]
    patch_sample_counts: Dict[int, int] = defaultdict(int)
    patch_cov_counts: Dict[Tuple[int, str], int] = defaultdict(int)
    for sample_idx, patch_id in enumerate(sample_patch_ids.tolist()):
        patch_sample_counts[patch_id] += 1
        for threshold_key, covered in covered_by_threshold.items():
            if bool(covered[sample_idx]):
                patch_cov_counts[(patch_id, threshold_key)] += 1

    summaries: List[Dict[str, Any]] = []
    for patch_id, faces in patch_face_ids.items():
        face_ids = np.asarray(faces, dtype=np.int32)
        area = float(face_area[face_ids].sum()) if len(face_ids) else 0.0
        item: Dict[str, Any] = {
            'patch_id': int(patch_id),
            'triangle_count': int(len(face_ids)),
            'surface_area_mm2': area,
            'sample_count': int(patch_sample_counts.get(patch_id, 0)),
            'coverage': {},
        }
        denom = max(1, patch_sample_counts.get(patch_id, 0))
        for threshold_key in covered_by_threshold.keys():
            item['coverage'][threshold_key] = float(patch_cov_counts.get((patch_id, threshold_key), 0) / denom)
        summaries.append(item)
    summaries.sort(key=lambda entry: entry['surface_area_mm2'], reverse=True)
    return summaries


def _summarize_bend_zones(sample_points: np.ndarray, covered_by_threshold: Dict[str, np.ndarray], specs: Iterable[Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        if not bool(getattr(spec, 'countable_in_regression', True)):
            continue
        start = np.asarray(spec.bend_line_start, dtype=np.float64)
        end = np.asarray(spec.bend_line_end, dtype=np.float64)
        distances = _distance_to_segment(sample_points, start, end)
        zone_half_width_mm = max(6.0, 0.35 * float(np.linalg.norm(end - start)))
        mask = distances <= zone_half_width_mm
        sample_count = int(np.count_nonzero(mask))
        if sample_count == 0:
            continue
        entry: Dict[str, Any] = {
            'bend_id': str(spec.bend_id),
            'target_angle_deg': float(getattr(spec, 'target_angle', 0.0)),
            'bend_line_length_mm': float(np.linalg.norm(end - start)),
            'zone_half_width_mm': float(zone_half_width_mm),
            'sample_count': sample_count,
            'coverage': {},
        }
        for threshold_key, covered in covered_by_threshold.items():
            entry['coverage'][threshold_key] = float(np.count_nonzero(covered[mask]) / sample_count)
        summaries.append(entry)
    summaries.sort(key=lambda item: item['bend_line_length_mm'], reverse=True)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description='Analyze true aligned CAD-surface coverage for a scan.')
    parser.add_argument('--cad-path', required=True)
    parser.add_argument('--scan-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sample-count', type=int, default=120000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--thresholds-mm', default='0.5,1.0,2.0')
    parser.add_argument('--normal-cluster-deg', type=float, default=8.0)
    parser.add_argument('--tolerance-angle', type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x.strip()) for x in str(args.thresholds_mm).split(',') if x.strip()]
    runtime_cfg = load_bend_runtime_config(None)

    bundle = build_bend_heatmap_evidence(
        args.cad_path,
        args.scan_path,
        tolerance_angle=float(args.tolerance_angle),
        runtime_cfg=runtime_cfg,
        output_dir=output_dir,
        cache_dir=None,
    )
    aligned_points = bundle.get('aligned_points')
    if aligned_points is None or len(aligned_points) == 0:
        raise RuntimeError('Failed to obtain aligned scan points')
    aligned_points = np.asarray(aligned_points, dtype=np.float64)

    cad_vertices, cad_faces, cad_source = load_cad_geometry(args.cad_path)
    if cad_faces is None or len(cad_faces) == 0:
        raise RuntimeError('CAD surface coverage analysis requires triangle faces')
    mesh = trimesh.Trimesh(vertices=np.asarray(cad_vertices, dtype=np.float64), faces=np.asarray(cad_faces, dtype=np.int32), process=False)
    sample_points, sample_face_indices = _sample_points(mesh, int(args.sample_count), int(args.seed))

    tree = cKDTree(aligned_points)
    distances, _nearest_indices = tree.query(sample_points, k=1, workers=-1)

    covered_by_threshold: Dict[str, np.ndarray] = {}
    global_cov: Dict[str, float] = {}
    for threshold in thresholds:
        key = f'{threshold:.2f}mm'
        covered = distances <= threshold
        covered_by_threshold[key] = covered
        global_cov[key] = float(np.count_nonzero(covered) / len(sample_points))

    patch_labels = _cluster_face_patches(mesh, float(args.normal_cluster_deg))
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)
    patch_summaries = _summarize_patches(sample_face_indices, patch_labels, covered_by_threshold, face_area)

    cad_specs, cad_strategy = extract_cad_bend_specs_with_strategy(
        cad_vertices=np.asarray(cad_vertices, dtype=np.float64),
        cad_triangles=np.asarray(cad_faces, dtype=np.int32),
        tolerance_angle=float(args.tolerance_angle),
        tolerance_radius=0.5,
        cad_extractor_kwargs=getattr(runtime_cfg, 'cad_extractor_kwargs', {}) or {},
        cad_detector_kwargs=getattr(runtime_cfg, 'cad_detector_kwargs', {}) or {},
        cad_detect_call_kwargs=getattr(runtime_cfg, 'cad_detect_call_kwargs', {}) or {},
        feature_policy=getattr(runtime_cfg, 'feature_policy', None),
    )
    bend_zone_summaries = _summarize_bend_zones(sample_points, covered_by_threshold, cad_specs)

    projection_threshold = covered_by_threshold[f'{thresholds[min(1, len(thresholds)-1)]:.2f}mm']
    _projection_plot(
        sample_points,
        projection_threshold,
        output_dir / 'aligned_surface_coverage_projection.png',
        f'Aligned CAD coverage @ {thresholds[min(1, len(thresholds)-1)]:.2f} mm',
    )

    significant_patches = [
        patch for patch in patch_summaries
        if int(patch.get('sample_count', 0)) >= 25 and float(patch.get('surface_area_mm2', 0.0)) >= 25.0
    ]
    top_low_patches = sorted(
        significant_patches,
        key=lambda entry: (entry['coverage'].get(f'{thresholds[min(1, len(thresholds)-1)]:.2f}mm', 1.0), -entry['surface_area_mm2'])
    )[:12]
    top_low_bends = sorted(
        bend_zone_summaries,
        key=lambda entry: entry['coverage'].get(f'{thresholds[min(1, len(thresholds)-1)]:.2f}mm', 1.0)
    )

    summary: Dict[str, Any] = {
        'cad_path': str(args.cad_path),
        'scan_path': str(args.scan_path),
        'cad_source': cad_source,
        'sample_count': int(len(sample_points)),
        'aligned_scan_point_count': int(len(aligned_points)),
        'distance_stats_mm': {
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'p95': float(np.percentile(distances, 95)),
            'max': float(np.max(distances)),
        },
        'global_surface_coverage': global_cov,
        'thresholds_mm': thresholds,
        'patch_count': int(len(patch_summaries)),
        'significant_patch_count': int(len(significant_patches)),
        'cad_bend_strategy': cad_strategy,
        'countable_bend_zone_count': int(len(bend_zone_summaries)),
        'top_low_coverage_patches': top_low_patches,
        'bend_zone_coverage': top_low_bends,
        'artifacts': {
            'projection_path': str(output_dir / 'aligned_surface_coverage_projection.png'),
            'aligned_scan_path': str(bundle.get('aligned_scan_path')) if bundle.get('aligned_scan_path') is not None else None,
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'patch_coverage.json').write_text(json.dumps(patch_summaries, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'bend_zone_coverage.json').write_text(json.dumps(bend_zone_summaries, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
