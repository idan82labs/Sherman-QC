#!/usr/bin/env python3
"""V14 with wider duplicate suppression and paired plane-fold recovery.

The pair recovery is deliberately count-free.  It recognizes two long,
continuous, scan-supported plane intersections which share a face, run in
parallel, have comparable spans, and are separated by a material fraction of
the part envelope.  This distinguishes a pair of physical return folds from
nearby aliases of one fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import build_feature_matrix, endpoints, select_indices_configurable
from scripts.evaluate_f1_against_manual_gold import pair_metrics
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.select_f1_mesh_route_v13 import RETRY_THRESHOLD
from scripts.select_f1_mesh_route_v14 import recover_continuous_long_seed_families
from scripts.train_f1_bounded_mesh_router_v9 import NMS_OVERLAP, SCORE_THRESHOLD


PROFILE = "F1_BOUNDED_MESH_PAIRED_PLANE_FOLD_V15"
NMS_ANGLE_DEG = 7.0
NMS_DISTANCE_MM = 6.0
PAIR_MIN_SPAN_FRACTION = 0.40
PAIR_MIN_SUPPORT_FRACTION = 0.95
PAIR_MIN_INTERSECTION_ANGLE_DEG = 75.0
PAIR_MIN_PLANE_SUPPORT_COUNT = 400
PAIR_MIN_CROSS_SUPPORT_WIDTH_MM = 10.0
PAIR_MAX_P90_LINE_DISTANCE_MM = 2.0
PAIR_MAX_AXIS_ANGLE_DEG = 2.0
PAIR_MIN_SPAN_RATIO = 0.90
PAIR_MIN_SEPARATION_FRACTION = 0.15
REPRESENTATIVE_MIN_ANGLE_DEG = 6.0
REPRESENTATIVE_MAX_ANGLE_DEG = 8.0
REPRESENTATIVE_MAX_DISTANCE_MM = 4.0
REPRESENTATIVE_MIN_OVERLAP = 0.75
REPRESENTATIVE_MIN_TRANSITION_GAIN = 0.10


def select_with_v15_nms(
    rows: list[dict], scores, *, score_threshold: float = SCORE_THRESHOLD
) -> tuple[list[int], bool]:
    selected = select_indices_configurable(
        rows,
        scores,
        threshold=score_threshold,
        max_angle=NMS_ANGLE_DEG,
        max_distance=NMS_DISTANCE_MM,
        min_overlap=NMS_OVERLAP,
    )
    retried = not selected
    if retried:
        selected = select_indices_configurable(
            rows,
            scores,
            threshold=RETRY_THRESHOLD,
            max_angle=NMS_ANGLE_DEG,
            max_distance=NMS_DISTANCE_MM,
            min_overlap=NMS_OVERLAP,
        )
    return selected, retried


def _eligible_plane_intersection(row: dict, bbox_span: float) -> bool:
    if row.get("representative_source") != "plane_lattice":
        return False
    if row.get("source_kind") != "scan_supported_plane_intersection":
        return False
    if float(row.get("span_mm") or 0.0) < PAIR_MIN_SPAN_FRACTION * bbox_span:
        return False
    support = (
        float(row.get("finite_support_fraction") or 0.0),
        float(row.get("finite_support_longest_run_fraction") or 0.0),
        float(row.get("endpoint_support_fraction") or 0.0),
    )
    if min(support) < PAIR_MIN_SUPPORT_FRACTION:
        return False
    if float(row.get("angle_deg") or 0.0) < PAIR_MIN_INTERSECTION_ANGLE_DEG:
        return False
    plane_counts = [float(value) for value in row.get("incident_plane_support_counts") or ()]
    cross_widths = [float(value) for value in row.get("incident_plane_cross_support_widths_mm") or ()]
    if len(plane_counts) < 2 or min(plane_counts) < PAIR_MIN_PLANE_SUPPORT_COUNT:
        return False
    if len(cross_widths) < 2 or min(cross_widths) < PAIR_MIN_CROSS_SUPPORT_WIDTH_MM:
        return False
    return float(row.get("p90_line_distance_mm") or float("inf")) <= PAIR_MAX_P90_LINE_DISTANCE_MM


def _represented(row: dict, rows: list[dict], selected: list[int]) -> bool:
    for index in selected:
        metrics = pair_metrics(endpoints(row), endpoints(rows[index]))
        if (
            metrics["axis_angle_deg"] <= NMS_ANGLE_DEG
            and metrics["transverse_offset_mm"] <= NMS_DISTANCE_MM
            and metrics["shorter_span_overlap"] >= NMS_OVERLAP
        ):
            return True
    return False


def recover_separated_parallel_plane_pairs(
    rows: list[dict], selected: list[int], bbox_span: float
) -> tuple[list[int], list[dict]]:
    """Recover unrepresented members of strongly supported paired folds."""

    eligible = [index for index, row in enumerate(rows) if _eligible_plane_intersection(row, bbox_span)]
    paired: set[int] = set()
    pair_evidence: dict[int, list[dict]] = {}
    for position, first_index in enumerate(eligible):
        first = rows[first_index]
        first_planes = set(str(value) for value in first.get("incident_plane_ids") or ())
        for second_index in eligible[:position]:
            second = rows[second_index]
            shared = sorted(first_planes.intersection(str(value) for value in second.get("incident_plane_ids") or ()))
            if not shared:
                continue
            first_axis = np.asarray(first["axis_direction"], dtype=np.float64)
            second_axis = np.asarray(second["axis_direction"], dtype=np.float64)
            cosine = abs(float(np.dot(first_axis, second_axis))) / (
                float(np.linalg.norm(first_axis)) * float(np.linalg.norm(second_axis))
            )
            axis_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            span_ratio = min(float(first["span_mm"]), float(second["span_mm"])) / max(
                float(first["span_mm"]), float(second["span_mm"])
            )
            metrics = pair_metrics(endpoints(first), endpoints(second))
            separation_fraction = float(metrics["transverse_offset_mm"]) / bbox_span
            if (
                axis_angle > PAIR_MAX_AXIS_ANGLE_DEG
                or span_ratio < PAIR_MIN_SPAN_RATIO
                or separation_fraction < PAIR_MIN_SEPARATION_FRACTION
            ):
                continue
            evidence = {
                "paired_candidate_ids": sorted((str(first["candidate_id"]), str(second["candidate_id"]))),
                "shared_plane_ids": shared,
                "axis_angle_deg": round(axis_angle, 12),
                "span_ratio": round(span_ratio, 12),
                "separation_fraction": round(separation_fraction, 12),
            }
            paired.update((first_index, second_index))
            pair_evidence.setdefault(first_index, []).append(evidence)
            pair_evidence.setdefault(second_index, []).append(evidence)

    recovered = list(selected)
    changes: list[dict] = []
    for index in sorted(paired, key=lambda value: str(rows[value].get("candidate_id") or "")):
        row = rows[index]
        if _represented(row, rows, recovered):
            continue
        recovered.append(index)
        changes.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "repair_basis": "separated_parallel_continuous_plane_intersection_pair",
                "pair_evidence": sorted(pair_evidence[index], key=lambda item: item["paired_candidate_ids"]),
            }
        )
    return recovered, changes


def repair_tilted_adjacency_representatives(
    rows: list[dict], selected: list[int]
) -> tuple[list[int], list[dict]]:
    """Replace a tilted adjacency lane when stronger continuous band evidence agrees nearby."""

    repaired: list[int] = []
    changes: list[dict] = []
    for index in selected:
        source = rows[index]
        if source.get("representative_source") != "scan_mesh_adjacency":
            repaired.append(index)
            continue
        choices: list[tuple[float, float, int, str, int]] = []
        source_transition = float(source.get("two_sided_transition_score") or 0.0)
        for candidate_index, candidate in enumerate(rows):
            if candidate.get("representative_source") not in ("cylindrical_standard", "cylindrical_wide"):
                continue
            if int(candidate.get("cluster_source_count") or 0) < 2:
                continue
            support = (
                float(candidate.get("finite_support_fraction") or 0.0),
                float(candidate.get("finite_support_longest_run_fraction") or 0.0),
                float(candidate.get("endpoint_support_fraction") or 0.0),
            )
            if min(support) < PAIR_MIN_SUPPORT_FRACTION:
                continue
            metrics = pair_metrics(endpoints(source), endpoints(candidate))
            if not (
                REPRESENTATIVE_MIN_ANGLE_DEG < metrics["axis_angle_deg"] <= REPRESENTATIVE_MAX_ANGLE_DEG
                and metrics["transverse_offset_mm"] <= REPRESENTATIVE_MAX_DISTANCE_MM
                and metrics["shorter_span_overlap"] >= REPRESENTATIVE_MIN_OVERLAP
            ):
                continue
            transition = float(candidate.get("two_sided_transition_score") or 0.0)
            if transition < source_transition + REPRESENTATIVE_MIN_TRANSITION_GAIN:
                continue
            choices.append(
                (
                    -transition,
                    -float(candidate.get("two_sided_normal_angle_deg") or 0.0),
                    -int(candidate.get("cluster_source_count") or 0),
                    str(candidate.get("candidate_id") or ""),
                    candidate_index,
                )
            )
        replacement = sorted(choices)[0][-1] if choices else index
        repaired.append(replacement)
        if replacement != index:
            changes.append(
                {
                    "from_candidate_id": str(source["candidate_id"]),
                    "to_candidate_id": str(rows[replacement]["candidate_id"]),
                    "repair_basis": "nearby_continuous_multisource_band_with_stronger_two_sided_transition",
                }
            )
    return list(dict.fromkeys(repaired)), changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    scores = joblib.load(args.v9_model)["estimator"].predict_proba(build_feature_matrix(rows))[:, 1]
    selected, retried = select_with_v15_nms(rows, scores)
    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    coverage: list[dict] = []
    if retried and selected:
        selected, coverage = recover_continuous_long_seed_families(rows, scores, selected, bbox_span)
    selected, pair_recoveries = recover_separated_parallel_plane_pairs(rows, selected, bbox_span)
    selected, tilted_repairs = repair_tilted_adjacency_representatives(rows, selected)
    prediction_rows, repairs = repair_bounded_cylindrical_representatives(rows, selected)
    predictions = []
    for row in prediction_rows:
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        source_index = int(row.get("_selector_source_index", rows.index(row) if row in rows else 0))
        predictions.append(
            {
                "bend_id": f"TRV15-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": "bounded_mesh_v15_paired_plane_fold",
                "selector_score": round(float(scores[source_index]), 12),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_bounded_mesh_paired_plane_fold_selection",
        "profile": PROFILE,
        "route": "bounded_mesh_v15_paired_plane_fold",
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "no_output_retry_used": retried,
        "initial_threshold": SCORE_THRESHOLD,
        "retry_threshold": RETRY_THRESHOLD,
        "nms_angle_deg": NMS_ANGLE_DEG,
        "nms_distance_mm": NMS_DISTANCE_MM,
        "seed_family_coverage_recoveries": coverage,
        "paired_plane_fold_recoveries": pair_recoveries,
        "tilted_adjacency_repairs": tilted_repairs,
        "representative_repairs": repairs,
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "pair_recovery_count": len(pair_recoveries), "retry_used": retried}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
