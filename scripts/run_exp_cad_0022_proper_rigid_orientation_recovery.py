#!/usr/bin/env python3
"""Recover reflected legacy registrations with a deterministic SO(3) pose pool.

EXP_CAD_0022 evaluates all 24 orientation-preserving signed PCA-axis
permutations, followed by fixed coarse and fine ICP.  Reflections and fitted
scale are structurally impossible.  N011 is the recovery challenge because its
EXP15 transform has det(R)=-1; N012 is the proper-transform same-family control.

The experiment is source-only.  Native RoundBend carriers are qualification
guards and native-resolution visuals are mandatory.  No gold, label, expected
count, or evaluator input is opened.
"""

from __future__ import annotations

import argparse
from itertools import permutations, product
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP21_PATH = ROOT / "scripts" / "run_exp_cad_0021_protected_registration_selector.py"
SPEC = importlib.util.spec_from_file_location("exp21_selector", EXP21_PATH)
EXP21 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP21)

from research.f1_deterministic_reference_surface_v1 import sample_surface  # noqa: E402


MAX_OVERLAP_RADIUS_MM = 10.0
CLOSE_RADIUS_MM = 5.0
MIN_DIRECTIONAL_SUPPORT = 1000
MIN_HARMONIC_OVERLAP = 0.80


def right_handed_pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(points, dtype=np.float64)
    center = np.mean(source, axis=0)
    centered = source - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(centered), 1))
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    values = eigenvalues[order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    return center, basis, values


def proper_signed_permutation_matrices() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for row, column in enumerate(permutation):
            base[row, column] = 1.0
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.diag(signs) @ base
            if np.linalg.det(matrix) > 0:
                matrices.append(matrix)
    matrices.sort(key=lambda matrix: tuple(int(value) for value in matrix.ravel()))
    if len(matrices) != 24:
        raise AssertionError("expected exactly 24 proper signed permutations")
    return matrices


def pca_initial_transform(
    scan_center: np.ndarray,
    scan_basis: np.ndarray,
    reference_center: np.ndarray,
    reference_basis: np.ndarray,
    signed_permutation: np.ndarray,
) -> np.ndarray:
    rotation = reference_basis @ signed_permutation @ scan_basis.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = reference_center - rotation @ scan_center
    return EXP21.canonicalize_transform(transform)


def _cloud(points: np.ndarray, normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return cloud


def refine_hypothesis(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        20.0,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    loss = o3d.pipelines.registration.TukeyLoss(k=2.0)
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        5.0,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-8,
            relative_rmse=1e-8,
            max_iteration=100,
        ),
    )
    return EXP21.canonicalize_transform(fine.transformation), {
        "coarse_fitness": float(coarse.fitness),
        "coarse_inlier_rmse_mm": float(coarse.inlier_rmse),
        "fine_fitness": float(fine.fitness),
        "fine_inlier_rmse_mm": float(fine.inlier_rmse),
    }


def overlap_distance_arrays(
    transformed_scan: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scan_tree = cKDTree(transformed_scan)
    reference_tree = cKDTree(reference_points)
    cad_to_scan, _ = scan_tree.query(reference_points, k=1, workers=1)
    scan_to_cad, _ = reference_tree.query(transformed_scan, k=1, workers=1)
    return cad_to_scan, scan_to_cad


def fixed_union_overlap_metrics(
    cad_to_scan: np.ndarray,
    scan_to_cad: np.ndarray,
    cad_union_mask: np.ndarray,
    scan_union_mask: np.ndarray,
) -> dict[str, Any]:
    cad = cad_to_scan[cad_union_mask]
    scan = scan_to_cad[scan_union_mask]
    if len(cad) < MIN_DIRECTIONAL_SUPPORT or len(scan) < MIN_DIRECTIONAL_SUPPORT:
        raise ValueError("insufficient fixed-union overlap support")
    cad_fraction = float(np.mean(cad <= CLOSE_RADIUS_MM))
    scan_fraction = float(np.mean(scan <= CLOSE_RADIUS_MM))
    harmonic = 0.0 if cad_fraction + scan_fraction == 0 else 2.0 * cad_fraction * scan_fraction / (cad_fraction + scan_fraction)
    combined = np.concatenate((np.clip(cad, 0.0, MAX_OVERLAP_RADIUS_MM), np.clip(scan, 0.0, MAX_OVERLAP_RADIUS_MM)))
    return {
        "cad_union_point_count": int(len(cad)),
        "scan_union_point_count": int(len(scan)),
        "cad_to_scan_within_5mm_fraction": cad_fraction,
        "scan_to_cad_within_5mm_fraction": scan_fraction,
        "harmonic_overlap_fraction": harmonic,
        "combined_median_distance_mm": float(np.median(combined)),
        "combined_p90_distance_mm": float(np.percentile(combined, 90)),
    }


def native_event_qualification(
    transformed_scan: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    *,
    thickness_mm: float,
) -> tuple[bool, list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    union = np.zeros(len(transformed_scan), dtype=bool)
    passed = True
    for event in events:
        metric, mask = EXP21.EXP18.evaluate_carrier_support(
            transformed_scan,
            event,
            thickness_mm=thickness_mm,
            min_points_per_bin=3,
        )
        minimum_bins = max(4, int(math.ceil(0.5 * int(metric["bin_count"]))))
        event_pass = bool(
            int(metric["supported_bin_count"]) >= minimum_bins
            and int(metric["carrier_point_count"]) >= 50
        )
        metric["minimum_required_supported_bins"] = minimum_bins
        metric["qualification_pass"] = event_pass
        rows.append(metric)
        union |= mask
        passed &= event_pass
    return passed, rows, union


def select_absolute_candidate(rows: Sequence[Mapping[str, Any]], *, legacy_id: Optional[str]) -> str:
    eligible = [
        row
        for row in rows
        if bool(row["proper_rigid_transform"])
        and bool(row["native_event_qualification_pass"])
        and float(row["overlap_metrics"]["harmonic_overlap_fraction"]) >= MIN_HARMONIC_OVERLAP
    ]
    if not eligible:
        raise ValueError("no eligible proper-rigid registration candidate")

    def rank(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
        metric = row["overlap_metrics"]
        return (
            -float(metric["harmonic_overlap_fraction"]),
            float(metric["combined_p90_distance_mm"]),
            float(metric["combined_median_distance_mm"]),
            str(row["candidate_id"]),
        )

    best = sorted(eligible, key=rank)[0]
    if legacy_id is None:
        return str(best["candidate_id"])
    legacy = next(row for row in eligible if row["candidate_id"] == legacy_id)
    if best["candidate_id"] == legacy_id:
        return legacy_id
    base = legacy["overlap_metrics"]
    trial = best["overlap_metrics"]
    material = bool(
        float(trial["harmonic_overlap_fraction"]) >= float(base["harmonic_overlap_fraction"]) + 0.005
        or (
            float(trial["harmonic_overlap_fraction"]) >= float(base["harmonic_overlap_fraction"]) - 0.001
            and float(trial["combined_median_distance_mm"]) <= float(base["combined_median_distance_mm"]) - 0.03
            and float(trial["combined_p90_distance_mm"]) <= float(base["combined_p90_distance_mm"]) - 0.05
        )
    )
    return str(best["candidate_id"]) if material else legacy_id


def execute_case(
    *,
    case_token: str,
    scan_path: Path,
    registration_path: Path,
    mesh_path: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
    output_root: Path,
    reference_sample_count: int,
) -> dict[str, Any]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if EXP21.sha256_path(scan_path) != registration["scan_sha256"]:
        raise ValueError(f"scan binding mismatch for {case_token}")
    if EXP21.sha256_path(mesh_path) != registration["engine_mesh_sha256"]:
        raise ValueError(f"mesh binding mismatch for {case_token}")
    raw_cloud = o3d.io.read_point_cloud(str(scan_path))
    raw_full = np.asarray(raw_cloud.points, dtype=np.float64)
    raw_points, raw_normals = EXP21.density_capped_points_and_normals(raw_full)
    vertices, triangles = EXP21._mesh_arrays(mesh_path)
    reference_points, reference_normals = sample_surface(vertices, triangles, reference_sample_count)
    scan_center, scan_basis, scan_eigenvalues = right_handed_pca_basis(raw_points)
    reference_center, reference_basis, reference_eigenvalues = right_handed_pca_basis(reference_points)
    source_icp = _cloud(raw_points).voxel_down_sample(1.5)
    source_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.5, max_nn=40))
    target_icp = _cloud(reference_points, reference_normals)

    candidates: list[dict[str, Any]] = []
    seen_transforms: set[tuple[float, ...]] = set()
    legacy_transform = np.asarray(registration["raw_scan_to_reference_transform"], dtype=np.float64)
    legacy_stats = EXP21.absolute_similarity_stats(legacy_transform)
    legacy_id: Optional[str] = None
    if bool(legacy_stats["proper_similarity_transform"]):
        legacy_transform = EXP21.canonicalize_transform(legacy_transform)
        legacy_id = "LEGACY_PROPER"
        candidates.append({"candidate_id": legacy_id, "transform": legacy_transform, "icp_diagnostic": None})
        seen_transforms.add(tuple(np.round(legacy_transform, 6).ravel()))

    for index, signed_permutation in enumerate(proper_signed_permutation_matrices(), 1):
        initial = pca_initial_transform(
            scan_center,
            scan_basis,
            reference_center,
            reference_basis,
            signed_permutation,
        )
        transform, diagnostic = refine_hypothesis(source_icp, target_icp, initial)
        key = tuple(np.round(transform, 6).ravel())
        if key in seen_transforms:
            continue
        seen_transforms.add(key)
        candidates.append(
            {
                "candidate_id": f"SO3_PCA_{index:02d}",
                "signed_permutation": signed_permutation.tolist(),
                "transform": transform,
                "icp_diagnostic": diagnostic,
            }
        )

    transformed = [EXP21.apply_transform(raw_points, row["transform"]) for row in candidates]
    distance_pairs = [overlap_distance_arrays(points, reference_points) for points in transformed]
    cad_union = np.any(np.vstack([pair[0] <= MAX_OVERLAP_RADIUS_MM for pair in distance_pairs]), axis=0)
    scan_union = np.any(np.vstack([pair[1] <= MAX_OVERLAP_RADIUS_MM for pair in distance_pairs]), axis=0)
    events = list(summary["extraction"]["events"])
    thickness_mm = float(summary["extraction"]["sheet_thickness_mm"])
    candidate_rows: list[dict[str, Any]] = []
    support_masks: dict[str, np.ndarray] = {}
    transformed_by_id: dict[str, np.ndarray] = {}
    for row, points, distances in zip(candidates, transformed, distance_pairs):
        stats = EXP21.absolute_similarity_stats(row["transform"])
        event_pass, event_rows, support_mask = native_event_qualification(points, events, thickness_mm=thickness_mm)
        candidate_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "signed_permutation": row.get("signed_permutation"),
                "absolute_transform": row["transform"].tolist(),
                "absolute_transform_stats": stats,
                "proper_rigid_transform": bool(
                    stats["proper_similarity_transform"]
                    and abs(float(stats["uniform_scale"]) - 1.0) <= 1e-6
                ),
                "icp_diagnostic": row["icp_diagnostic"],
                "overlap_metrics": fixed_union_overlap_metrics(distances[0], distances[1], cad_union, scan_union),
                "native_event_qualification_pass": event_pass,
                "native_event_metrics": event_rows,
            }
        )
        support_masks[row["candidate_id"]] = support_mask
        transformed_by_id[row["candidate_id"]] = points
    selection_authorized = True
    selection_blockers: list[str] = []
    try:
        selected_id = select_absolute_candidate(candidate_rows, legacy_id=legacy_id)
    except ValueError as error:
        selection_authorized = False
        selection_blockers.append(str(error).upper().replace(" ", "_"))
        proper_rows = [row for row in candidate_rows if bool(row["proper_rigid_transform"])]
        if not proper_rows:
            raise
        selected_id = str(
            sorted(
                proper_rows,
                key=lambda row: (
                    -float(row["overlap_metrics"]["harmonic_overlap_fraction"]),
                    float(row["overlap_metrics"]["combined_p90_distance_mm"]),
                    str(row["candidate_id"]),
                ),
            )[0]["candidate_id"]
        )
    selected_row = next(row for row in candidate_rows if row["candidate_id"] == selected_id)
    selected_points = transformed_by_id[selected_id]
    selected_support = support_masks[selected_id]
    legacy_points = EXP21.apply_transform(raw_points, legacy_transform)
    _, _, legacy_support = native_event_qualification(legacy_points, events, thickness_mm=thickness_mm)
    legacy_metric = (
        next(row["overlap_metrics"] for row in candidate_rows if row["candidate_id"] == legacy_id)
        if legacy_id is not None
        else EXP21.global_visible_reference_metrics(legacy_points, reference_points[cad_union])
    )
    selected_display_metric = {
        "within_2mm_fraction": selected_row["overlap_metrics"]["cad_to_scan_within_5mm_fraction"],
        "p90_distance_mm": selected_row["overlap_metrics"]["combined_p90_distance_mm"],
    }
    legacy_display_metric = {
        "within_2mm_fraction": (
            legacy_metric.get("cad_to_scan_within_5mm_fraction")
            if "cad_to_scan_within_5mm_fraction" in legacy_metric
            else legacy_metric["within_2mm_fraction"]
        ),
        "p90_distance_mm": legacy_metric["combined_p90_distance_mm"] if "combined_p90_distance_mm" in legacy_metric else legacy_metric["p90_distance_mm"],
    }
    case_root = output_root / "cases" / case_token
    before_after_path = case_root / "native_legacy_vs_proper.png"
    EXP21.render_before_after(
        before_after_path,
        case_token=case_token,
        baseline_points=legacy_points,
        selected_points=selected_points,
        reference_points=reference_points[cad_union],
        events=events,
        baseline_support=legacy_support,
        selected_support=selected_support,
        selected_candidate_id=selected_id,
        baseline_global=legacy_display_metric,
        selected_global=selected_display_metric,
    )
    delta_path = case_root / "native_legacy_vs_proper_delta.png"
    EXP21.render_global_delta(
        delta_path,
        case_token=case_token,
        baseline_points=legacy_points,
        selected_points=selected_points,
        reference_points=reference_points[cad_union],
        events=events,
        selected_candidate_id=selected_id,
    )
    selected_full = EXP21.apply_transform(raw_full, np.asarray(selected_row["absolute_transform"], dtype=np.float64))
    selected_scan_path = case_root / "selected_proper_aligned_scan.ply"
    case_root.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(selected_scan_path), _cloud(selected_full), write_ascii=False, compressed=False):
        raise ValueError("failed to write selected proper scan")
    payload = {
        "schema": "EXP_CAD_0022_PROPER_RIGID_ORIENTATION_RECOVERY_CASE_V1",
        "experiment_id": "EXP_CAD_0022",
        "case_token": case_token,
        "scan_sha256": registration["scan_sha256"],
        "engine_mesh_sha256": registration["engine_mesh_sha256"],
        "source_registration_semantic_sha256": EXP21.sha256_path(registration_path),
        "source_candidate_summary_sha256": EXP21.sha256_path(summary_path),
        "legacy_absolute_transform_stats": legacy_stats,
        "legacy_candidate_selectable": legacy_id is not None,
        "raw_density_capped_point_count": int(len(raw_points)),
        "reference_sample_count": int(len(reference_points)),
        "scan_pca_eigenvalues": scan_eigenvalues.tolist(),
        "reference_pca_eigenvalues": reference_eigenvalues.tolist(),
        "proper_orientation_hypothesis_count": 24,
        "deduplicated_candidate_count": len(candidate_rows),
        "fixed_union_cad_point_count": int(np.sum(cad_union)),
        "fixed_union_scan_point_count": int(np.sum(scan_union)),
        "candidate_rows": candidate_rows,
        "selected_candidate_id": selected_id,
        "selection_authorized": selection_authorized,
        "selection_blockers": selection_blockers,
        "selected_absolute_transform": selected_row["absolute_transform"],
        "selected_overlap_metrics": selected_row["overlap_metrics"],
        "selected_native_event_qualification_pass": selected_row["native_event_qualification_pass"],
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "artifacts": {
            "selected_aligned_scan": {"path": str(selected_scan_path.resolve()), "sha256": EXP21.sha256_path(selected_scan_path)},
            "legacy_vs_proper": {"path": str(before_after_path.resolve()), "sha256": EXP21.sha256_path(before_after_path)},
            "legacy_vs_proper_delta": {"path": str(delta_path.resolve()), "sha256": EXP21.sha256_path(delta_path)},
        },
        "labels_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    semantic_path = case_root / "semantic_result.json"
    EXP21.atomic_json(semantic_path, payload)
    return {
        "case_token": case_token,
        "legacy_candidate_selectable": legacy_id is not None,
        "selected_candidate_id": selected_id,
        "selection_authorized": selection_authorized,
        "selection_blockers": selection_blockers,
        "semantic_sha256": EXP21.sha256_path(semantic_path),
        "selected_scan_sha256": EXP21.sha256_path(selected_scan_path),
        "render_sha256": EXP21.sha256_path(before_after_path),
        "delta_render_sha256": EXP21.sha256_path(delta_path),
        "selected_harmonic_overlap": selected_row["overlap_metrics"]["harmonic_overlap_fraction"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-token", action="append", required=True)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference-sample-count", type=int, default=80000)
    args = parser.parse_args(argv)
    summary_path = args.candidate_summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    allowed = {str(row["case_token"]) for row in summary["records"]}
    mesh_hash_index = EXP21._mesh_hash_index(args.mesh_root.resolve())
    records: list[dict[str, Any]] = []
    for case_token in args.case_token:
        if case_token not in allowed:
            raise ValueError(f"case {case_token} not bound to candidate summary")
        registration_path = args.registration_root / "cases" / case_token / "semantic_result.json"
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        mesh_path = mesh_hash_index[registration["engine_mesh_sha256"]]
        scan_path = args.scan_root / f"{case_token}.ply"
        print(case_token, flush=True)
        records.append(
            execute_case(
                case_token=case_token,
                scan_path=scan_path.resolve(),
                registration_path=registration_path.resolve(),
                mesh_path=mesh_path,
                summary_path=summary_path,
                summary=summary,
                output_root=args.output_root.resolve(),
                reference_sample_count=args.reference_sample_count,
            )
        )
    output = {
        "schema": "EXP_CAD_0022_PROPER_RIGID_ORIENTATION_RECOVERY_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0022",
        "case_count": len(records),
        "records": records,
        "candidate_generation": "ALL_24_ORIENTATION_PRESERVING_SIGNED_PCA_AXIS_PERMUTATIONS_COARSE_P2P_FINE_ROBUST_P2PL_V1",
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "labels_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    EXP21.atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", output)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
