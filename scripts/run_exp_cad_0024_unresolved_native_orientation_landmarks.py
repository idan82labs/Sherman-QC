#!/usr/bin/env python3
"""Use unresolved native display lists only to disambiguate CAD orientation.

The face-owned EXP15 mesh deliberately omitted SLDPRT DisplayLists whose B-rep
face ownership was unresolved.  For 50476000_A those tables include the
asymmetric central tab and planar profile needed to distinguish two otherwise
equivalent proper poses.  EXP24 keeps this lane strictly orientation-only:

* source and decoder hashes are verified;
* the unresolved table inventory must exactly match the frozen mesh report;
* each table is sampled deterministically and weighted equally;
* only proper EXP22 candidates passing strict native-event qualification enter;
* no unresolved table may create, delete, relocate, or classify a bend.

No labels, expected counts, gold, filenames-as-identity, or evaluator data are
used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP23_PATH = ROOT / "scripts" / "run_exp_cad_0023_symmetry_class_proper_registration.py"
SPEC = importlib.util.spec_from_file_location("exp23_pose_class", EXP23_PATH)
EXP23 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP23)
EXP22 = EXP23.EXP22
EXP21 = EXP23.EXP21

from research.f1_deterministic_reference_surface_v1 import sample_surface  # noqa: E402


SUPPORTED_CADMPEG_VERSION = "cadmpeg 0.5.5"
LANDMARK_CLOSE_RADIUS_MM = 2.0
MIN_TABLE_COVERAGE = 0.95
MIN_MEAN_TABLE_COVERAGE = 0.98
MIN_RUNNER_UP_MARGIN = 0.20
CLUSTER_TRANSLATION_MM = 0.01
CLUSTER_ROTATION_DEG = 0.01


def decode_cadir(source: Path, cadmpeg: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    version = subprocess.run([str(cadmpeg), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if version != SUPPORTED_CADMPEG_VERSION:
        raise ValueError(f"unsupported cadmpeg version {version!r}")
    with tempfile.TemporaryDirectory(prefix="exp24-cadir-") as directory:
        root = Path(directory)
        cadir_path = root / "model.cadir.json"
        report_path = root / "decoder.report.json"
        result = subprocess.run(
            [
                str(cadmpeg), "convert", str(source), "--input-format", "sldprt",
                "--format", "cadir", "--output", str(cadir_path), "--report", str(report_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cadmpeg failed: {result.stderr[-1000:]}")
        return json.loads(cadir_path.read_text(encoding="utf-8")), json.loads(report_path.read_text(encoding="utf-8"))


def unresolved_landmark_tables(
    cadir: Mapping[str, Any],
    *,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    expected = sorted(str(value) for value in expected_ids)
    rows: list[dict[str, Any]] = []
    for table in cadir["model"]["tessellations"]:
        if table.get("body") and table.get("faces"):
            continue
        vertices = np.asarray([[point["x"], point["y"], point["z"]] for point in table.get("vertices", [])], dtype=np.float64)
        triangles = np.asarray(table.get("triangles", []), dtype=np.int64)
        if not len(vertices) or not len(triangles):
            raise ValueError(f"empty unresolved table {table.get('id')}")
        sample_count = max(200, min(5000, 20 * len(triangles)))
        points, _normals = sample_surface(vertices, triangles, sample_count)
        rows.append(
            {
                "table_id": str(table["id"]),
                "vertex_count": int(len(vertices)),
                "triangle_count": int(len(triangles)),
                "sample_count": int(len(points)),
                "sample_points": points,
                "bbox_min_mm": np.min(vertices, axis=0).tolist(),
                "bbox_max_mm": np.max(vertices, axis=0).tolist(),
                "sample_sha256": hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
            }
        )
    rows.sort(key=lambda row: row["table_id"])
    actual = [row["table_id"] for row in rows]
    if actual != expected:
        raise ValueError("unresolved display-list inventory differs from frozen mesh report")
    return rows


def table_coverage_metrics(transformed_scan: np.ndarray, tables: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tree = cKDTree(transformed_scan)
    table_rows: list[dict[str, Any]] = []
    all_distances: list[np.ndarray] = []
    for table in tables:
        distances, _ = tree.query(np.asarray(table["sample_points"], dtype=np.float64), k=1, workers=1)
        all_distances.append(distances)
        table_rows.append(
            {
                "table_id": table["table_id"],
                "sample_count": int(len(distances)),
                "within_2mm_fraction": float(np.mean(distances <= LANDMARK_CLOSE_RADIUS_MM)),
                "median_distance_mm": float(np.median(distances)),
                "p90_distance_mm": float(np.percentile(distances, 90)),
            }
        )
    combined = np.concatenate(all_distances)
    coverages = np.asarray([row["within_2mm_fraction"] for row in table_rows], dtype=np.float64)
    return {
        "table_count": len(table_rows),
        "table_metrics": table_rows,
        "minimum_table_coverage": float(np.min(coverages)),
        "mean_equal_table_coverage": float(np.mean(coverages)),
        "combined_point_coverage": float(np.mean(combined <= LANDMARK_CLOSE_RADIUS_MM)),
        "combined_median_distance_mm": float(np.median(combined)),
        "combined_p90_distance_mm": float(np.percentile(combined, 90)),
    }


def transform_near(first: np.ndarray, second: np.ndarray) -> bool:
    motion = EXP23.relative_motion(first, second)
    return bool(
        motion["translation_mm"] <= CLUSTER_TRANSLATION_MM
        and motion["rotation_deg"] <= CLUSTER_ROTATION_DEG
    )


def cluster_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: str(value["candidate_id"])):
        transform = np.asarray(row["absolute_transform"], dtype=np.float64)
        target = None
        for cluster in clusters:
            if transform_near(transform, np.asarray(cluster["representative"]["absolute_transform"], dtype=np.float64)):
                target = cluster
                break
        if target is None:
            clusters.append({"representative": row, "members": [str(row["candidate_id"])]})
        else:
            target["members"].append(str(row["candidate_id"]))
    return clusters


def rank_clusters(clusters: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        clusters,
        key=lambda cluster: (
            -float(cluster["representative"]["orientation_landmark_metrics"]["minimum_table_coverage"]),
            -float(cluster["representative"]["orientation_landmark_metrics"]["mean_equal_table_coverage"]),
            float(cluster["representative"]["orientation_landmark_metrics"]["combined_p90_distance_mm"]),
            str(cluster["representative"]["candidate_id"]),
        ),
    )


def orientation_gate(ranked: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    if len(ranked) < 2:
        return False, ["FEWER_THAN_TWO_DISTINCT_POSE_CLUSTERS"]
    best = ranked[0]["representative"]["orientation_landmark_metrics"]
    runner = ranked[1]["representative"]["orientation_landmark_metrics"]
    reasons: list[str] = []
    if float(best["minimum_table_coverage"]) < MIN_TABLE_COVERAGE:
        reasons.append("BEST_MINIMUM_TABLE_COVERAGE_GATE")
    if float(best["mean_equal_table_coverage"]) < MIN_MEAN_TABLE_COVERAGE:
        reasons.append("BEST_MEAN_TABLE_COVERAGE_GATE")
    margin = float(best["mean_equal_table_coverage"]) - float(runner["mean_equal_table_coverage"])
    if margin < MIN_RUNNER_UP_MARGIN:
        reasons.append("RUNNER_UP_MARGIN_GATE")
    return not reasons, reasons


def render_orientation_comparison(
    path: Path,
    *,
    case_token: str,
    scan_points: np.ndarray,
    landmark_points: np.ndarray,
    best: Mapping[str, Any],
    runner: Mapping[str, Any],
) -> None:
    candidates = (best, runner)
    transformed = [EXP21.apply_transform(scan_points, np.asarray(row["absolute_transform"], dtype=np.float64)) for row in candidates]
    scan_step = max(1, len(scan_points) // 120000)
    landmark_step = max(1, len(landmark_points) // 40000)
    landmark = landmark_points[::landmark_step]
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    figure, axes = plt.subplots(3, 2, figsize=(30, 24), dpi=100, facecolor="#0b1016")
    for column, row in enumerate(candidates):
        points = transformed[column][::scan_step]
        metric = row["orientation_landmark_metrics"]
        for index, (horizontal, vertical, title) in enumerate(projections):
            axis = axes[index, column]
            axis.set_facecolor("#0b1016")
            axis.scatter(points[:, horizontal], points[:, vertical], color="#46d9ff", s=0.75, alpha=0.28, linewidths=0)
            axis.scatter(landmark[:, horizontal], landmark[:, vertical], color="#ffd23f", s=1.8, alpha=0.70, linewidths=0)
            combined = np.vstack((points, landmark))
            x0, x1 = np.percentile(combined[:, horizontal], [0.05, 99.95])
            y0, y1 = np.percentile(combined[:, vertical], [0.05, 99.95])
            xm, ym = max((x1 - x0) * 0.04, 1.0), max((y1 - y0) * 0.04, 1.0)
            axis.set_xlim(x0 - xm, x1 + xm)
            axis.set_ylim(y0 - ym, y1 + ym)
            axis.set_aspect("equal", adjustable="box")
            axis.set_axis_off()
            axis.set_title(
                f"{title} — {row['candidate_id']}\n"
                f"min/mean table coverage={metric['minimum_table_coverage']:.3f}/{metric['mean_equal_table_coverage']:.3f}",
                color="white", fontsize=15, pad=8,
            )
    figure.suptitle(
        f"{case_token} — unresolved native orientation landmarks\n"
        "cyan: transformed raw scan; yellow: source-native unresolved DisplayList samples (orientation-only)",
        color="white", fontsize=21, y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)


def execute_case(
    *,
    case_token: str,
    scan_path: Path,
    exp22_semantic: Path,
    tables: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    parent = json.loads(exp22_semantic.read_text(encoding="utf-8"))
    if EXP21.sha256_path(scan_path) != parent["scan_sha256"]:
        raise ValueError("scan binding mismatch")
    raw = np.asarray(o3d.io.read_point_cloud(str(scan_path)).points, dtype=np.float64)
    raw_points, _normals = EXP21.density_capped_points_and_normals(raw)
    eligible: list[dict[str, Any]] = []
    for row in parent["candidate_rows"]:
        strict_pass, strict_reasons = EXP23.strict_event_pass(row["native_event_metrics"])
        if not bool(row["proper_rigid_transform"]) or not strict_pass:
            continue
        transform = np.asarray(row["absolute_transform"], dtype=np.float64)
        transformed = EXP21.apply_transform(raw_points, transform)
        eligible.append(
            {
                "candidate_id": row["candidate_id"],
                "absolute_transform": row["absolute_transform"],
                "strict_native_event_pass": strict_pass,
                "strict_native_event_reasons": strict_reasons,
                "orientation_landmark_metrics": table_coverage_metrics(transformed, tables),
            }
        )
    clusters = rank_clusters(cluster_candidates(eligible))
    passed, reasons = orientation_gate(clusters)
    best = clusters[0]["representative"]
    runner = clusters[1]["representative"]
    margin = (
        float(best["orientation_landmark_metrics"]["mean_equal_table_coverage"])
        - float(runner["orientation_landmark_metrics"]["mean_equal_table_coverage"])
    )
    landmark_points = np.vstack([np.asarray(table["sample_points"], dtype=np.float64) for table in tables])
    case_root = output_root / "cases" / case_token
    render_path = case_root / "native_orientation_landmark_comparison.png"
    render_orientation_comparison(
        render_path,
        case_token=case_token,
        scan_points=raw_points,
        landmark_points=landmark_points,
        best=best,
        runner=runner,
    )
    payload = {
        "schema": "EXP_CAD_0024_UNRESOLVED_NATIVE_ORIENTATION_LANDMARK_CASE_V1",
        "experiment_id": "EXP_CAD_0024",
        "case_token": case_token,
        "scan_sha256": parent["scan_sha256"],
        "source_exp22_semantic_sha256": EXP21.sha256_path(exp22_semantic),
        "eligible_candidate_count": len(eligible),
        "distinct_pose_cluster_count": len(clusters),
        "pose_clusters": [
            {
                "representative_candidate_id": cluster["representative"]["candidate_id"],
                "member_candidate_ids": cluster["members"],
                "absolute_transform": cluster["representative"]["absolute_transform"],
                "orientation_landmark_metrics": cluster["representative"]["orientation_landmark_metrics"],
            }
            for cluster in clusters
        ],
        "selected_candidate_id": best["candidate_id"] if passed else None,
        "selected_absolute_transform": best["absolute_transform"] if passed else None,
        "runner_up_candidate_id": runner["candidate_id"],
        "mean_equal_table_coverage_margin": margin,
        "orientation_gate_pass": passed,
        "orientation_gate_reasons": reasons,
        "absolute_registration_orientation_authorized": passed,
        "unresolved_lane_scope": "ORIENTATION_ONLY_FORBIDDEN_FOR_BEND_DISCOVERY_CLASSIFICATION_OR_MEASUREMENT",
        "render": {"path": str(render_path.relative_to(output_root)), "sha256": EXP21.sha256_path(render_path)},
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "labels_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    semantic = case_root / "semantic_result.json"
    EXP21.atomic_json(semantic, payload)
    return {
        "case_token": case_token,
        "orientation_gate_pass": passed,
        "selected_candidate_id": payload["selected_candidate_id"],
        "runner_up_candidate_id": payload["runner_up_candidate_id"],
        "coverage_margin": margin,
        "semantic_sha256": EXP21.sha256_path(semantic),
        "render_sha256": EXP21.sha256_path(render_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-token", action="append", required=True)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--source-sldprt", required=True, type=Path)
    parser.add_argument("--cadmpeg", required=True, type=Path)
    parser.add_argument("--mesh-report", required=True, type=Path)
    parser.add_argument("--exp22-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    mesh_report = json.loads(args.mesh_report.read_text(encoding="utf-8"))
    source_sha = EXP21.sha256_path(args.source_sldprt)
    cadmpeg_sha = EXP21.sha256_path(args.cadmpeg)
    if source_sha != mesh_report["source"]["sha256"] or cadmpeg_sha != mesh_report["dependency"]["sha256"]:
        raise ValueError("source or decoder hash differs from frozen mesh report")
    cadir, decoder_report = decode_cadir(args.source_sldprt.resolve(), args.cadmpeg.resolve())
    decode = decoder_report.get("decode_report", {})
    if not decode.get("geometry_transferred") or decode.get("container_only"):
        raise ValueError("decoder did not transfer geometry")
    tables = unresolved_landmark_tables(
        cadir,
        expected_ids=mesh_report["selection"]["omitted_unresolved_tessellation_ids"],
    )
    table_manifest = [
        {key: value for key, value in table.items() if key != "sample_points"}
        for table in tables
    ]
    records = []
    for case_token in args.case_token:
        print(case_token, flush=True)
        records.append(
            execute_case(
                case_token=case_token,
                scan_path=(args.scan_root / f"{case_token}.ply").resolve(),
                exp22_semantic=(args.exp22_root / "cases" / case_token / "semantic_result.json").resolve(),
                tables=tables,
                output_root=args.output_root.resolve(),
            )
        )
    selected_ids = {row["selected_candidate_id"] for row in records if row["orientation_gate_pass"]}
    output = {
        "schema": "EXP_CAD_0024_UNRESOLVED_NATIVE_ORIENTATION_LANDMARK_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0024",
        "source_sldprt_sha256": source_sha,
        "cadmpeg_sha256": cadmpeg_sha,
        "mesh_report_sha256": EXP21.sha256_path(args.mesh_report),
        "unresolved_table_count": len(tables),
        "unresolved_table_manifest": table_manifest,
        "case_count": len(records),
        "orientation_gate_pass_count": int(sum(row["orientation_gate_pass"] for row in records)),
        "cross_scan_selected_candidate_ids": sorted(selected_ids),
        "cross_scan_consistent_orientation_class": len(selected_ids) == 1,
        "records": records,
        "unresolved_lane_scope": "ORIENTATION_ONLY_FORBIDDEN_FOR_BEND_DISCOVERY_CLASSIFICATION_OR_MEASUREMENT",
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
