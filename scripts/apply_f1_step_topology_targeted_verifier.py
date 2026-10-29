#!/usr/bin/env python3
"""Project native STEP bend axes through the existing overlay and verify in scan."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_additional_line_label_review import _scan_points, line_scan_support
from scripts.evaluate_f1_against_manual_gold import pair_metrics
from scripts.extract_step_topology_bend_axes import extract_step_bend_axes


NORMAL_CORRIDOR_RADIUS_MM = 10.0
NORMAL_LINE_SAMPLE_COUNT = 31
NORMAL_MAX_POINTS = 1500
NORMAL_NEIGHBORS = 20
NORMAL_MIN_POINTS = 40
NORMAL_MIN_TURN_DEG = 15.0
NORMAL_MIN_SIDE_FRACTION = 0.15


def _to_raw(line: list[list[float]], raw_to_cad: list[list[float]]) -> np.ndarray:
    transform = np.asarray(raw_to_cad, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("invalid raw_scan_to_cad_transform")
    inverse = np.linalg.inv(transform)
    endpoints = np.asarray(line, dtype=np.float64)
    homogeneous = np.column_stack([endpoints, np.ones(2, dtype=np.float64)])
    return (homogeneous @ inverse.T)[:, :3]


def _duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = np.asarray([left["detected_line_start"], left["detected_line_end"]])
    b = np.asarray([right["detected_line_start"], right["detected_line_end"]])
    metrics = pair_metrics(a, b)
    return bool(
        metrics["axis_angle_deg"] <= 5.0
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.60
    )


def _canonical_normal(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    vector = vector / norm
    dominant = int(np.argmax(np.abs(vector)))
    return -vector if vector[dominant] < 0.0 else vector


def _principal_unoriented_normal(rows: np.ndarray) -> np.ndarray:
    scatter = np.einsum("ni,nj->ij", rows, rows)
    _, vectors = np.linalg.eigh(scatter)
    return _canonical_normal(vectors[:, -1])


def _two_surface_normal_turn(
    points: np.ndarray,
    tree: cKDTree,
    line_start: list[float],
    line_end: list[float],
) -> dict[str, Any]:
    start = np.asarray(line_start, dtype=np.float64)
    end = np.asarray(line_end, dtype=np.float64)
    direction = end - start
    span = float(np.linalg.norm(direction))
    if span <= 1e-9:
        return {"passed": False, "reason": "degenerate_axis", "point_count": 0}
    unit = direction / span
    parameters = np.linspace(0.0, 1.0, NORMAL_LINE_SAMPLE_COUNT)
    samples = start + parameters[:, None] * direction
    neighborhoods = tree.query_ball_point(
        samples, r=NORMAL_CORRIDOR_RADIUS_MM, workers=1
    )
    indices = sorted({int(index) for rows in neighborhoods for index in rows})
    if not indices:
        return {"passed": False, "reason": "empty_normal_corridor", "point_count": 0}
    local = np.asarray(points[indices], dtype=np.float64)
    axial = (local - start) @ unit
    closest = start + np.clip(axial, 0.0, span)[:, None] * unit
    radial = np.linalg.norm(local - closest, axis=1)
    local = local[(axial >= -1.0) & (axial <= span + 1.0) & (radial <= NORMAL_CORRIDOR_RADIUS_MM)]
    if len(local) > NORMAL_MAX_POINTS:
        order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
        selected = np.linspace(0, len(order) - 1, NORMAL_MAX_POINTS, dtype=int)
        local = local[order[selected]]
    if len(local) < NORMAL_MIN_POINTS:
        return {
            "passed": False,
            "reason": "insufficient_normal_points",
            "point_count": int(len(local)),
        }

    local_tree = cKDTree(local)
    k = min(NORMAL_NEIGHBORS, len(local))
    _, neighbors = local_tree.query(local, k=k, workers=1)
    normals = []
    for neighbor_indices in np.atleast_2d(neighbors):
        neighborhood = local[np.asarray(neighbor_indices, dtype=int)]
        centered = neighborhood - np.mean(neighborhood, axis=0)
        covariance = centered.T @ centered
        _, vectors = np.linalg.eigh(covariance)
        normals.append(_canonical_normal(vectors[:, 0]))
    normals_array = np.asarray(normals, dtype=np.float64)

    center_a = normals_array[0]
    dissimilarity = 1.0 - np.abs(normals_array @ center_a)
    center_b = normals_array[int(np.argmax(dissimilarity))]
    assignments = np.zeros(len(normals_array), dtype=bool)
    for _ in range(12):
        next_assignments = (
            np.abs(normals_array @ center_b) > np.abs(normals_array @ center_a)
        )
        if np.array_equal(next_assignments, assignments) and np.any(assignments):
            break
        assignments = next_assignments
        if np.any(~assignments):
            center_a = _principal_unoriented_normal(normals_array[~assignments])
        if np.any(assignments):
            center_b = _principal_unoriented_normal(normals_array[assignments])

    count_a = int(np.sum(~assignments))
    count_b = int(np.sum(assignments))
    side_fraction = min(count_a, count_b) / len(normals_array)
    turn_deg = float(
        np.degrees(
            np.arccos(np.clip(abs(float(np.dot(center_a, center_b))), -1.0, 1.0))
        )
    )
    passed = bool(
        count_a > 0
        and count_b > 0
        and side_fraction >= NORMAL_MIN_SIDE_FRACTION
        and turn_deg >= NORMAL_MIN_TURN_DEG
    )
    return {
        "passed": passed,
        "reason": "two_surface_turn" if passed else "weak_or_unbalanced_normal_turn",
        "point_count": int(len(local)),
        "surface_a_points": count_a,
        "surface_b_points": count_b,
        "minimum_side_fraction": float(side_fraction),
        "normal_turn_deg": turn_deg,
        "surface_a_normal": center_a.tolist(),
        "surface_b_normal": center_b.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--topology-root", type=Path, required=True)
    parser.add_argument("--require-normal-turn", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.input_summary.read_text(encoding="utf-8"))
    output = copy.deepcopy(source)
    totals = {
        "overlay_abstained_cases": 0,
        "native_step_axes": 0,
        "scan_supported_axes": 0,
        "normal_turn_verified_axes": 0,
        "deduplicated_axes": 0,
        "retained_axes": 0,
    }
    audit_cases = []
    for result in output["results"]:
        alignment = dict(result.get("alignment_metadata") or {})
        raw_to_cad = alignment.get("raw_scan_to_cad_transform")
        if raw_to_cad is None:
            totals["overlay_abstained_cases"] += 1
            result["matches"] = []
            result["summary"] = {
                **dict(result.get("summary") or {}),
                "native_step_topology_bends": None,
                "scan_verified_step_bends": 0,
                "cad_targeted_abstention": "automatic_overlay_unavailable_or_untrusted",
            }
            audit_cases.append(
                {
                    "part_key": result["part_key"],
                    "scan_path": result["scan_path"],
                    "native_step_axis_count": None,
                    "retained_axis_count": 0,
                    "abstention": "automatic_overlay_unavailable_or_untrusted",
                    "records": [],
                }
            )
            continue
        topology = extract_step_bend_axes(Path(result["cad_path"]))
        topology_path = args.topology_root / (str(result["part_key"]) + ".json")
        topology_path.parent.mkdir(parents=True, exist_ok=True)
        topology_path.write_text(
            json.dumps(topology, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        points = _scan_points(Path(result["scan_path"]))
        scan_tree = cKDTree(points)
        accepted = []
        records = []
        for axis in topology["bend_axes"]:
            totals["native_step_axes"] += 1
            raw_line = _to_raw(
                [axis["line_start"], axis["line_end"]], raw_to_cad
            )
            # The neutral axis lies between the paired inner/outer surfaces.
            # A 3 mm support radius covers the observed sheet surface for the
            # thin-gauge corpus while remaining narrower than neighboring bends.
            support = line_scan_support(points, raw_line[0].tolist(), raw_line[1].tolist())
            record = {
                "bend_id": axis["bend_id"],
                "cad_axis_start": axis["line_start"],
                "cad_axis_end": axis["line_end"],
                "raw_scan_axis_start": raw_line[0].tolist(),
                "raw_scan_axis_end": raw_line[1].tolist(),
                "radii_mm": axis["radii_mm"],
                "angular_sweeps_deg": axis["angular_sweeps_deg"],
                "scan_support": support,
                "accepted": bool(support["frame_integrity_pass"]),
            }
            records.append(record)
            if not support["frame_integrity_pass"]:
                continue
            totals["scan_supported_axes"] += 1
            normal_turn = _two_surface_normal_turn(
                points,
                scan_tree,
                support["trimmed_line_start"],
                support["trimmed_line_end"],
            )
            record["normal_turn_verification"] = normal_turn
            if args.require_normal_turn and not normal_turn["passed"]:
                record["accepted"] = False
                record["rejection_reason"] = "failed_two_surface_normal_turn"
                continue
            if normal_turn["passed"]:
                totals["normal_turn_verified_axes"] += 1
            row = {
                "bend_id": axis["bend_id"],
                "cad_line_start": axis["line_start"],
                "cad_line_end": axis["line_end"],
                "detected_line_start": support["trimmed_line_start"],
                "detected_line_end": support["trimmed_line_end"],
                "assignment_source": "STEP_TOPOLOGY_SCAN_VERIFIED",
                "assignment_candidate_kind": "CAD_TARGET",
                "measurement_method": "native_step_paired_cylinder_scan_support",
                "observability_state": "FORMED",
                "cad_targeted_local_verifier": "accepted_native_step_axis",
                "normal_turn_verification": normal_turn,
                "step_topology": {
                    "radii_mm": axis["radii_mm"],
                    "angular_sweeps_deg": axis["angular_sweeps_deg"],
                    "surface_count": axis["surface_count"],
                },
            }
            duplicate = next((existing for existing in accepted if _duplicate(row, existing)), None)
            if duplicate is not None:
                totals["deduplicated_axes"] += 1
                record["accepted"] = False
                record["rejection_reason"] = "duplicate_of_" + duplicate["bend_id"]
                continue
            accepted.append(row)
        totals["retained_axes"] += len(accepted)
        result["matches"] = accepted
        result["summary"] = {
            **dict(result.get("summary") or {}),
            "native_step_topology_bends": len(topology["bend_axes"]),
            "scan_verified_step_bends": len(accepted),
        }
        audit_cases.append(
            {
                "part_key": result["part_key"],
                "scan_path": result["scan_path"],
                "topology_path": str(topology_path.resolve()),
                "native_step_axis_count": len(topology["bend_axes"]),
                "retained_axis_count": len(accepted),
                "records": records,
            }
        )

    audit = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_step_topology_targeted_verifier_audit",
        "uses_existing_automatic_cad_scan_overlay": True,
        "native_step_topology": True,
        "mesh_crease_candidates_allowed": False,
        "global_scan_proposals_allowed": False,
        "gold_or_expected_counts_used": False,
        "fitted_parameters": False,
        "requires_two_surface_normal_turn": bool(args.require_normal_turn),
        "normal_turn_contract": {
            "corridor_radius_mm": NORMAL_CORRIDOR_RADIUS_MM,
            "line_sample_count": NORMAL_LINE_SAMPLE_COUNT,
            "maximum_points": NORMAL_MAX_POINTS,
            "neighbors": NORMAL_NEIGHBORS,
            "minimum_points": NORMAL_MIN_POINTS,
            "minimum_turn_deg": NORMAL_MIN_TURN_DEG,
            "minimum_side_fraction": NORMAL_MIN_SIDE_FRACTION
        },
        "totals": totals,
        "cases": audit_cases,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.audit.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
