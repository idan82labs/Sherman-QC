#!/usr/bin/env python3
"""Build provisional line-label review jobs from independent CAD/scan matches.

The output is deliberately not authoritative ground truth.  Every line must be
accepted by visual review before it may enter the training-only expansion set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]

JOBS = (
    (
        "14_28870000",
        "source-assets/02_CAD_QC_3D/28870000_A_FULL_SCAN.ply",
        "data/bend_corpus/evaluation/benchmark_stability_core_20260402_selected_face_recovery_v1/28870000/28870000_A_FULL_SCAN.ply--e91bd366.json",
    ),
    (
        "15_44211000",
        "source-assets/02_CAD_QC_3D/44211000_A.ply",
        "data/bend_corpus/evaluation/benchmark_stability_core_20260402_selected_face_recovery_v1/44211000/44211000_A.ply--a1522cf5.json",
    ),
    (
        "16_47266000",
        "source-assets/02_CAD_QC_3D/47266000_F.stl",
        "data/bend_corpus/evaluation/benchmark_stability_core_20260402_selected_face_recovery_v1/47266000/47266000_F.stl--f9a0dc03.json",
    ),
    (
        "17_48963008",
        "source-assets/02_CAD_QC_3D/48963008_B.ply",
        "data/bend_corpus/evaluation/benchmark_stability_core_20260402_selected_face_recovery_v1/48963008/48963008_B.ply--deb31545.json",
    ),
    (
        "18_48991006",
        "source-assets/02_CAD_QC_3D/48991006_A 4.ply",
        "data/bend_corpus/evaluation/benchmark_stability_core_20260402_selected_face_recovery_v1/48991006/48991006_A_4.ply--8d1970d1.json",
    ),
)

# These are manual, multi-view visual decisions, not detector outputs.  Keeping
# them beside the builder makes regeneration deterministic and prevents rejected
# mixed-frame/face-crossing lines from silently entering a training set.
REVIEW_DECISIONS: dict[str, dict[str, tuple[str, str]]] = {
    "14_28870000": {
        "P01": ("rejected", "mixed coordinate frame; the full line floats hundreds of millimetres from the scan"),
        "P02": ("rejected", "mixed coordinate frame; the full line floats hundreds of millimetres from the scan"),
    },
    "15_44211000": {
        "P01": ("accepted", "multi-view line follows the supported sheet-to-curved-end transition"),
    },
    "16_47266000": {
        "P01": ("rejected", "multi-view line cuts diagonally across a planar face instead of following a bend axis"),
    },
    "17_48963008": {
        "P01": ("accepted", "multi-view line follows the lower longitudinal flange transition"),
        "P02": ("accepted", "multi-view line follows the upper longitudinal flange transition"),
        "P03": ("rejected", "line crosses the part diagonally and is mostly unsupported by the scan"),
    },
    "18_48991006": {
        "P01": ("accepted", "trimmed line follows the long rear flange transition with two-sided surface support"),
        "P02": ("accepted", "trimmed line follows the right outer flange transition with two-sided surface support"),
        "P03": ("accepted", "trimmed line follows the right inner return transition with two-sided surface support"),
        "P04": ("accepted", "trimmed line follows the left inner return transition with two-sided surface support"),
        "P05": ("accepted", "trimmed line follows the left outer flange transition with two-sided surface support"),
        "P06": ("accepted", "trimmed short line follows the left tab transition with two-sided surface support"),
        "P07": ("accepted", "trimmed short line follows the right tab transition with two-sided surface support"),
        "P08": ("accepted", "trimmed line follows the descending front-tab transition; one side is partially occluded"),
        "P09": ("rejected", "scan-supported line lies across one planar face; local normals differ by only about 16 degrees"),
    },
}

SUPPORT_DISTANCE_MM = 3.0
SUPPORT_SAMPLE_COUNT = 201
MIN_SUPPORTED_SPAN_MM = 8.0
MIN_SUPPORTED_FRACTION = 0.10
COMPLETE_REVIEW_CASES = {"15_44211000", "17_48963008", "18_48991006"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_points(scan_path: Path, max_points: int = 200_000) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(scan_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        mesh = o3d.io.read_triangle_mesh(str(scan_path))
        points = np.asarray(mesh.vertices, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError(f"scan contains no finite geometry: {scan_path}")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points


def _longest_true_run(values: np.ndarray) -> tuple[int, int]:
    best_start = best_end = current_start = 0
    in_run = False
    for index, value in enumerate(np.asarray(values, dtype=bool)):
        if value and not in_run:
            current_start = index
            in_run = True
        if in_run and (not value or index == len(values) - 1):
            current_end = index + 1 if value else index
            if current_end - current_start > best_end - best_start:
                best_start, best_end = current_start, current_end
            in_run = False
    return best_start, best_end


def line_scan_support(
    points: np.ndarray,
    line_start: list[float],
    line_end: list[float],
    *,
    distance_mm: float = SUPPORT_DISTANCE_MM,
    sample_count: int = SUPPORT_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Measure and crop a provisional line to its longest scan-supported run."""

    start = np.asarray(line_start, dtype=np.float64)
    end = np.asarray(line_end, dtype=np.float64)
    parameters = np.linspace(0.0, 1.0, max(3, int(sample_count)), dtype=np.float64)
    samples = start + parameters[:, None] * (end - start)
    distances, _ = cKDTree(np.asarray(points, dtype=np.float64)).query(samples, k=1, workers=1)
    supported = np.asarray(distances <= float(distance_mm), dtype=bool)
    run_start, run_end = _longest_true_run(supported)
    span_mm = float(np.linalg.norm(end - start))
    if run_end > run_start:
        trimmed_start = samples[run_start]
        trimmed_end = samples[run_end - 1]
        supported_span_mm = float(np.linalg.norm(trimmed_end - trimmed_start))
    else:
        trimmed_start = start
        trimmed_end = start
        supported_span_mm = 0.0
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    margin = float(distance_mm)
    inside = np.all((samples >= bbox_min - margin) & (samples <= bbox_max + margin), axis=1)
    support_fraction = float(np.mean(supported))
    frame_integrity_pass = bool(
        np.any(inside)
        and support_fraction >= MIN_SUPPORTED_FRACTION
        and supported_span_mm >= MIN_SUPPORTED_SPAN_MM
    )
    return {
        "sample_count": int(len(samples)),
        "distance_threshold_mm": float(distance_mm),
        "source_span_mm": span_mm,
        "bbox_overlap_fraction": float(np.mean(inside)),
        "support_fraction": support_fraction,
        "longest_support_run_fraction": float((run_end - run_start) / len(samples)),
        "supported_span_mm": supported_span_mm,
        "median_distance_mm": float(np.median(distances)),
        "p90_distance_mm": float(np.quantile(distances, 0.90)),
        "trimmed_line_start": trimmed_start.tolist(),
        "trimmed_line_end": trimmed_end.tolist(),
        "frame_integrity_pass": frame_integrity_pass,
    }


def attach_scan_integrity(payload: dict[str, Any], points: np.ndarray) -> None:
    payload["scan_point_count_used_for_integrity"] = int(len(points))
    payload["scan_bbox_min"] = np.min(points, axis=0).tolist()
    payload["scan_bbox_max"] = np.max(points, axis=0).tolist()
    for bend in payload["manual_bends"]:
        bend["scan_support"] = line_scan_support(
            points,
            bend["line_start"],
            bend["line_end"],
        )


def apply_review_decisions(
    payload: dict[str, Any],
    decisions: Mapping[str, tuple[str, str]],
) -> None:
    for bend in payload["manual_bends"]:
        bend_id = str(bend["bend_id"])
        if bend_id not in decisions:
            continue
        decision, reason = decisions[bend_id]
        if decision not in {"accepted", "rejected"}:
            raise ValueError(f"{payload['job_id']} {bend_id}: invalid review decision {decision}")
        support = bend.get("scan_support") or {}
        if decision == "accepted" and not support.get("frame_integrity_pass"):
            raise ValueError(f"{payload['job_id']} {bend_id}: accepted line failed frame integrity")
        bend["review_decision"] = decision
        bend["review_reason"] = reason
        if decision == "accepted":
            bend["source_line_start"] = bend["line_start"]
            bend["source_line_end"] = bend["line_end"]
            bend["line_start"] = support["trimmed_line_start"]
            bend["line_end"] = support["trimmed_line_end"]
            bend["geometry_adjustment"] = "trimmed_to_longest_scan_supported_run"
    counts = {
        decision: sum(bend["review_decision"] == decision for bend in payload["manual_bends"])
        for decision in ("accepted", "rejected", "unreviewed")
    }
    payload["review_counts"] = counts
    payload["review_status"] = (
        "visual_review_complete_training_only"
        if counts["unreviewed"] == 0
        else "provisional_requires_visual_review"
    )


def build_labels(case_id: str, evaluation: Mapping[str, Any], scan_path: Path) -> dict[str, Any]:
    expected = int((evaluation.get("expectation") or {}).get("expected_completed_bends") or 0)
    countable = [
        match
        for match in evaluation.get("matches") or ()
        if match.get("countable_in_regression") is not False
        and match.get("detected_line_start") is not None
        and match.get("detected_line_end") is not None
    ]
    if len(countable) != expected:
        raise ValueError(
            f"{case_id}: expected {expected} countable matches but found {len(countable)}"
        )
    bends = []
    for index, match in enumerate(countable, start=1):
        bends.append(
            {
                "bend_id": f"P{index:02d}",
                "line_start": [float(value) for value in match["detected_line_start"]],
                "line_end": [float(value) for value in match["detected_line_end"]],
                "semantic": "conventional_bend",
                "source": "cad_scan_match_provisional",
                "source_bend_id": str(match.get("bend_id") or ""),
                "review_decision": "unreviewed",
            }
        )
    return {
        "schema_version": 1,
        "kind": "provisional_bend_instance_labels",
        "job_id": case_id,
        "coordinate_frame": "scan_raw",
        "scan_path": str(scan_path.resolve()),
        "review_status": "provisional_requires_visual_review",
        "authoritative_ground_truth": False,
        "expected_count_source": "existing_user_backed_corpus_expectation",
        "manual_bends": bends,
        "marker_reviews": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/f1-training-expansion/cad-guided-provisional-v1",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    manifest = []
    strict_blind_entries = []
    for case_id, scan_relative, evaluation_relative in JOBS:
        scan_path = (ROOT.parent / scan_relative).resolve()
        evaluation_path = (ROOT / evaluation_relative).resolve()
        if not scan_path.exists() or not evaluation_path.exists():
            raise FileNotFoundError(f"{case_id}: missing scan or evaluation input")
        labels = build_labels(case_id, _read(evaluation_path), scan_path)
        attach_scan_integrity(labels, _scan_points(scan_path))
        apply_review_decisions(labels, REVIEW_DECISIONS.get(case_id, {}))
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        labels_path = case_dir / "bend_instance_labels.json"
        labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "case_id": case_id,
                "scan_path": str(scan_path),
                "labels_path": str(labels_path),
                "provisional_bend_count": len(labels["manual_bends"]),
                "accepted_bend_count": labels["review_counts"]["accepted"],
                "rejected_bend_count": labels["review_counts"]["rejected"],
                "unreviewed_bend_count": labels["review_counts"]["unreviewed"],
                "review_status": labels["review_status"],
            }
        )
        if case_id in COMPLETE_REVIEW_CASES:
            if labels["review_counts"]["unreviewed"]:
                raise ValueError(f"{case_id}: complete review case still has unreviewed axes")
            part_key = case_id.split("_", 1)[1]
            strict_blind_entries.append(
                {
                    "part_key": part_key,
                    "scan_path": str(scan_path),
                    "expectation": {
                        "label_semantics": "training_only_visually_adjudicated_bend_axes",
                        "expected_bend_count": labels["review_counts"]["accepted"],
                        "expected_conventional_bend_count": labels["review_counts"]["accepted"],
                        "state": "provisional_training_only",
                        "truth_record_id": f"{case_id}::cad_guided_provisional_v1",
                    },
                    "audit_tags": [
                        "strict_blind_inference",
                        "training_expansion_only",
                        "non_authoritative",
                        "manual_multiview_adjudication",
                    ],
                }
            )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "authoritative_ground_truth": False,
                "case_count": len(manifest),
                "provisional_bend_count": sum(row["provisional_bend_count"] for row in manifest),
                "accepted_bend_count": sum(row["accepted_bend_count"] for row in manifest),
                "rejected_bend_count": sum(row["rejected_bend_count"] for row in manifest),
                "unreviewed_bend_count": sum(row["unreviewed_bend_count"] for row in manifest),
                "cases": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "strict_blind_training_slice.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "cad_guided_provisional_v1_strict_blind_training_expansion",
                "target": "training_only_visually_adjudicated_bend_axes",
                "authoritative_ground_truth": False,
                "entries": strict_blind_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(manifest), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
