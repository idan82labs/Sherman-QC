#!/usr/bin/env python3
"""Adjudicate EXP22 proper poses as symmetry classes, not unique ownership.

EXP22 proved that an arbitrary absolute harmonic-overlap threshold is invalid on
the incomplete 50476000_A tessellation.  EXP23 does not rerun or tune ICP.  It
rescored the frozen EXP22 candidate transforms on full, fixed source identities
using H2 and clipped L5, requires complete native-event support, and compares
the best proper pose directly with the frozen incumbent.  Symmetry-equivalent
poses are retained only as a pose class; absolute feature ownership abstains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP22_PATH = ROOT / "scripts" / "run_exp_cad_0022_proper_rigid_orientation_recovery.py"
SPEC = importlib.util.spec_from_file_location("exp22_recovery", EXP22_PATH)
EXP22 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP22)
EXP21 = EXP22.EXP21

from research.f1_deterministic_reference_surface_v1 import sample_surface  # noqa: E402


def full_bidirectional_metrics(scan: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    scan_tree = cKDTree(scan)
    reference_tree = cKDTree(reference)
    cad_distances, _ = scan_tree.query(reference, k=1, workers=1)
    scan_distances, _ = reference_tree.query(scan, k=1, workers=1)
    cad_coverage = float(np.mean(cad_distances <= 2.0))
    scan_coverage = float(np.mean(scan_distances <= 2.0))
    harmonic = 0.0 if cad_coverage + scan_coverage == 0 else 2.0 * cad_coverage * scan_coverage / (cad_coverage + scan_coverage)
    clipped_loss = 0.5 * (
        float(np.mean(np.minimum(cad_distances, 5.0) / 5.0))
        + float(np.mean(np.minimum(scan_distances, 5.0) / 5.0))
    )
    return {
        "scan_point_count": int(len(scan)),
        "cad_point_count": int(len(reference)),
        "cad_to_scan_within_2mm_fraction": cad_coverage,
        "scan_to_cad_within_2mm_fraction": scan_coverage,
        "harmonic_overlap_h2": harmonic,
        "clipped_bidirectional_loss_l5": clipped_loss,
    }


def strict_event_pass(event_rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for row in event_rows:
        event_id = str(row["event_id"])
        tolerance = float(row["radial_tolerance_mm"])
        if int(row["supported_bin_count"]) != int(row["bin_count"]):
            reasons.append(f"{event_id}:NOT_ALL_BINS_SUPPORTED")
        median = row.get("median_carrier_radial_residual_mm")
        p90 = row.get("p90_carrier_radial_residual_mm")
        if median is None or float(median) > tolerance / 2.0 + 1e-12:
            reasons.append(f"{event_id}:MEDIAN_RESIDUAL_GATE")
        if p90 is None or float(p90) > tolerance + 1e-12:
            reasons.append(f"{event_id}:P90_RESIDUAL_GATE")
    return not reasons, reasons


def rank_candidate(rows: Sequence[Mapping[str, Any]]) -> str:
    eligible = [row for row in rows if bool(row["proper_rigid_transform"]) and bool(row["strict_native_event_pass"])]
    if not eligible:
        raise ValueError("no strict proper candidate")
    best = sorted(
        eligible,
        key=lambda row: (
            -float(row["full_bidirectional_metrics"]["harmonic_overlap_h2"]),
            float(row["full_bidirectional_metrics"]["clipped_bidirectional_loss_l5"]),
            str(row["candidate_id"]),
        ),
    )[0]
    return str(best["candidate_id"])


def relative_motion(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    relative = np.asarray(first, dtype=np.float64) @ np.linalg.inv(np.asarray(second, dtype=np.float64))
    stats = EXP21.rigid_delta_stats(relative)
    return {"rotation_deg": stats["rotation_deg"], "translation_mm": stats["translation_mm"]}


def execute_case(
    *,
    case_token: str,
    exp22_semantic: Path,
    registration_path: Path,
    scan_path: Path,
    mesh_path: Path,
    output_root: Path,
    reference_sample_count: int,
) -> dict[str, Any]:
    parent = json.loads(exp22_semantic.read_text(encoding="utf-8"))
    if EXP21.sha256_path(scan_path) != parent["scan_sha256"] or EXP21.sha256_path(mesh_path) != parent["engine_mesh_sha256"]:
        raise ValueError("EXP22 source binding mismatch")
    raw = np.asarray(o3d.io.read_point_cloud(str(scan_path)).points, dtype=np.float64)
    raw_points, _raw_normals = EXP21.density_capped_points_and_normals(raw)
    vertices, triangles = EXP21._mesh_arrays(mesh_path)
    reference_points, _reference_normals = sample_surface(vertices, triangles, reference_sample_count)
    rows: list[dict[str, Any]] = []
    for parent_row in parent["candidate_rows"]:
        transform = np.asarray(parent_row["absolute_transform"], dtype=np.float64)
        points = EXP21.apply_transform(raw_points, transform)
        event_pass, reasons = strict_event_pass(parent_row["native_event_metrics"])
        rows.append(
            {
                "candidate_id": parent_row["candidate_id"],
                "absolute_transform": parent_row["absolute_transform"],
                "proper_rigid_transform": parent_row["proper_rigid_transform"],
                "strict_native_event_pass": event_pass,
                "strict_native_event_reasons": reasons,
                "full_bidirectional_metrics": full_bidirectional_metrics(points, reference_points),
            }
        )
    best_id = rank_candidate(rows)
    best = next(row for row in rows if row["candidate_id"] == best_id)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    legacy_transform = np.asarray(registration["raw_scan_to_reference_transform"], dtype=np.float64)
    legacy_points = EXP21.apply_transform(raw_points, legacy_transform)
    legacy_metrics = full_bidirectional_metrics(legacy_points, reference_points)
    best_metrics = best["full_bidirectional_metrics"]
    no_geometry_regression = bool(
        float(best_metrics["harmonic_overlap_h2"]) >= float(legacy_metrics["harmonic_overlap_h2"]) - 1e-12
        and float(best_metrics["clipped_bidirectional_loss_l5"]) <= float(legacy_metrics["clipped_bidirectional_loss_l5"]) + 1e-12
    )
    legacy_proper = bool(EXP21.absolute_similarity_stats(legacy_transform)["proper_similarity_transform"])
    motion = relative_motion(np.asarray(best["absolute_transform"], dtype=np.float64), legacy_transform) if legacy_proper else None
    symmetry_nonunique = bool(not legacy_proper or (motion is not None and motion["rotation_deg"] >= 90.0))
    status = (
        "PROPER_POSE_CLASS_RECOVERED_ABSOLUTE_OWNERSHIP_ABSTAIN"
        if no_geometry_regression and symmetry_nonunique
        else "NO_RETAINABLE_PROPER_POSE_CLASS"
    )
    payload = {
        "schema": "EXP_CAD_0023_SYMMETRY_CLASS_PROPER_REGISTRATION_CASE_V1",
        "experiment_id": "EXP_CAD_0023",
        "case_token": case_token,
        "source_exp22_semantic_sha256": EXP21.sha256_path(exp22_semantic),
        "scan_sha256": parent["scan_sha256"],
        "engine_mesh_sha256": parent["engine_mesh_sha256"],
        "candidate_rows": rows,
        "best_proper_candidate_id": best_id,
        "best_proper_metrics": best_metrics,
        "legacy_metrics_under_same_score": legacy_metrics,
        "best_vs_legacy_no_geometry_regression": no_geometry_regression,
        "legacy_transform_proper": legacy_proper,
        "best_vs_legacy_relative_motion": motion,
        "symmetry_equivalence_nonunique": symmetry_nonunique,
        "pose_class_status": status,
        "absolute_feature_ownership_authorized": False,
        "visual_artifacts_reused_from_exp22": parent["artifacts"],
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "labels_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    case_root = output_root / "cases" / case_token
    semantic = case_root / "semantic_result.json"
    EXP21.atomic_json(semantic, payload)
    return {
        "case_token": case_token,
        "best_proper_candidate_id": best_id,
        "pose_class_status": status,
        "semantic_sha256": EXP21.sha256_path(semantic),
        "h2_legacy": legacy_metrics["harmonic_overlap_h2"],
        "h2_best_proper": best_metrics["harmonic_overlap_h2"],
        "l5_legacy": legacy_metrics["clipped_bidirectional_loss_l5"],
        "l5_best_proper": best_metrics["clipped_bidirectional_loss_l5"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-token", action="append", required=True)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--exp22-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference-sample-count", type=int, default=80000)
    args = parser.parse_args(argv)
    mesh_index = EXP21._mesh_hash_index(args.mesh_root.resolve())
    records = []
    for case_token in args.case_token:
        registration = json.loads((args.registration_root / "cases" / case_token / "semantic_result.json").read_text())
        records.append(
            execute_case(
                case_token=case_token,
                exp22_semantic=(args.exp22_root / "cases" / case_token / "semantic_result.json").resolve(),
                registration_path=(args.registration_root / "cases" / case_token / "semantic_result.json").resolve(),
                scan_path=(args.scan_root / f"{case_token}.ply").resolve(),
                mesh_path=mesh_index[registration["engine_mesh_sha256"]],
                output_root=args.output_root.resolve(),
                reference_sample_count=args.reference_sample_count,
            )
        )
    output = {
        "schema": "EXP_CAD_0023_SYMMETRY_CLASS_PROPER_REGISTRATION_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0023",
        "case_count": len(records),
        "records": records,
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
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
