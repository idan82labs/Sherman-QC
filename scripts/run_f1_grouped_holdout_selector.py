#!/usr/bin/env python3
"""Evaluate F1 candidate selection with nested leave-one-part-out fitting.

Every outer fold excludes all scans of the evaluated physical part.  The
selection threshold is chosen only from inner folds of the remaining parts.
This is a development cross-validation checkpoint, not a locked blind release
benchmark, but it prevents same-part and same-scan label leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    CASE_JOBS,
    Thresholds,
    _endpoints,
    evaluate,
    is_match,
    pair_metrics,
)


SOURCES = (
    "strict_blind_owned",
    "bend_hypothesis",
    "control_support_strip",
    "evidence_trace",
    "fragment_merge",
    "endpoint_connector",
    "cylindrical_standard",
    "cylindrical_wide",
    "plane_lattice",
    "scan_mesh_adjacency",
)
FEATURE_NAMES = (
    "span_mm",
    "span_over_diagonal",
    "radius_mm",
    "transverse_offset_mm",
    "vote_count",
    "support_count",
    "support_fraction",
    "normal_spread_deg",
    "raw_score",
    "confidence",
    "stability_score",
    "angle_deg",
    "cross_width_mm",
    "consensus_count",
    "radius_consensus_count",
    "support_per_span",
    "finite_support_fraction",
    "finite_support_longest_run_fraction",
    "endpoint_support_fraction",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "transverse_angular_span_deg",
    "transverse_angular_bin_fraction",
    "normal_second_eigenvalue",
    "normal_spread_deg_scan",
    "corridor_points_per_mm",
    "cluster_member_count",
    "cluster_source_count",
    "cluster_max_finite_support_fraction",
    "cluster_max_endpoint_support_fraction",
    "cluster_max_normal_second_eigenvalue",
    "cluster_max_normal_spread_deg_scan",
    "cluster_mean_median_line_distance_mm",
    "cluster_max_two_sided_transition_score",
    "cluster_max_two_sided_normal_angle_deg",
    "cluster_max_two_sided_support_balance",
    "cluster_max_two_sided_min_support_fraction",
    "cluster_max_two_sided_normal_coherence",
    "cluster_max_tight_normal_second_eigenvalue_r2mm",
    "cluster_max_tight_to_wide_normal_second_ratio",
    "case_percentile_support_count",
    "case_percentile_score",
    "case_percentile_consensus_count",
    "case_percentile_finite_support_fraction",
    "case_percentile_finite_support_longest_run_fraction",
    "case_percentile_endpoint_support_fraction",
    "case_percentile_normal_second_eigenvalue",
    "case_percentile_normal_spread_deg_scan",
    "case_percentile_corridor_points_per_mm",
    "case_percentile_transverse_angular_span_deg",
    "case_percentile_cluster_member_count",
    "case_percentile_cluster_source_count",
    "case_percentile_two_sided_transition_score",
    "case_percentile_two_sided_normal_angle_deg",
    "case_percentile_two_sided_support_balance",
    "case_percentile_two_sided_min_support_fraction",
    "case_percentile_two_sided_normal_coherence",
    "case_percentile_tight_normal_second_eigenvalue_r2mm",
    "case_percentile_tight_to_wide_normal_second_ratio",
    *(f"source_{source}" for source in SOURCES),
)
ARRANGEMENT_FEATURE_NAMES = (
    "arrangement_junction_neighbor_count",
    "arrangement_orthogonal_junction_neighbor_count",
    "arrangement_endpoint_neighbor_count",
    "arrangement_connected_endpoint_count",
    "arrangement_parallel_family_count",
    "arrangement_near_duplicate_count",
    "case_percentile_arrangement_junction_neighbor_count",
    "case_percentile_arrangement_orthogonal_junction_neighbor_count",
    "case_percentile_arrangement_endpoint_neighbor_count",
    "case_percentile_arrangement_connected_endpoint_count",
    "case_percentile_arrangement_parallel_family_count",
    "case_percentile_arrangement_near_duplicate_count",
)
THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
CYLINDRICAL_OFFSET_CAP_GRID = (5.0, 6.0, 7.0, 8.0, 12.0)
DUPLICATE_OFFSET_TOLERANCE_GRID = (5.0, 6.0, 8.0, 10.0)
ADDITIONAL_TRAINING_WEIGHT_GRID = (0.0, 0.05, 0.20, 1.0)
TWO_SIDED_MIN_SUPPORT_GATE_GRID = (0.0, 0.30, 0.40)
TOPOLOGY_COMPLETION_DELTA_GRID = (0.0, 0.10, 0.20)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def part_group(case_id: str) -> str:
    return case_id.split("_", 1)[1] if "_" in case_id else case_id


def training_case_ids(case_ids: Sequence[str], held_out_case_id: str) -> list[str]:
    held_out_group = part_group(held_out_case_id)
    return [case_id for case_id in case_ids if part_group(case_id) != held_out_group]


def _point_segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    delta = segment[1] - segment[0]
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - segment[0]))
    parameter = float(np.clip(np.dot(point - segment[0], delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (segment[0] + parameter * delta)))


def _attach_arrangement_features(
    candidates: Sequence[dict[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    """Attach deterministic bounded-memory line-arrangement features.

    The historical implementation called several scalar geometry helpers for
    every candidate pair.  That became prohibitively slow once scan-mesh
    adjacency raised a case to hundreds of line objects.  This version keeps
    the same pair predicates but evaluates one left-candidate row at a time in
    NumPy, bounding temporary memory to O(candidate_count).
    """

    if not candidates:
        return
    endpoints = np.asarray([_endpoints(candidate) for candidate in candidates], dtype=np.float64)
    deltas = endpoints[:, 1] - endpoints[:, 0]
    lengths = np.linalg.norm(deltas, axis=1)
    axes = deltas / np.maximum(lengths[:, None], 1e-12)
    midpoints = np.mean(endpoints, axis=1)
    bbox = np.asarray(payload.get("scan_bbox_mm") or (1.0, 1.0, 1.0), dtype=np.float64)
    diagonal = max(float(np.linalg.norm(bbox)), 1.0)
    junction_tolerance = max(4.0, min(15.0, 0.03 * diagonal))
    parallel_offset_tolerance = max(10.0, min(35.0, 0.08 * diagonal))
    count = len(candidates)
    junction_counts = np.zeros(count, dtype=np.int64)
    orthogonal_counts = np.zeros(count, dtype=np.int64)
    endpoint_counts = np.zeros(count, dtype=np.int64)
    connected_endpoint_masks = np.zeros(count, dtype=np.uint8)
    parallel_counts = np.zeros(count, dtype=np.int64)
    duplicate_counts = np.zeros(count, dtype=np.int64)

    def point_to_segments(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        segment_deltas = ends - starts
        denominator = np.einsum("ij,ij->i", segment_deltas, segment_deltas)
        parameters = np.divide(
            np.einsum("ij,ij->i", points - starts, segment_deltas),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        parameters = np.clip(parameters, 0.0, 1.0)
        projections = starts + parameters[:, None] * segment_deltas
        return np.linalg.norm(points - projections, axis=1)

    for left_index in range(count - 1):
        right_slice = slice(left_index + 1, count)
        right_indices = np.arange(left_index + 1, count)
        right_endpoints = endpoints[right_slice]
        right_axes = axes[right_slice]
        dots = np.abs(right_axes @ axes[left_index])
        angles = np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))

        endpoint_distance_matrix = np.linalg.norm(
            endpoints[left_index][None, :, None, :] - right_endpoints[:, None, :, :],
            axis=3,
        )
        flattened_distances = endpoint_distance_matrix.reshape((-1, 4))
        closest_flat = np.argmin(flattened_distances, axis=1)
        endpoint_gaps = flattened_distances[np.arange(len(right_indices)), closest_flat]
        closest_left_endpoint = closest_flat // 2
        closest_right_endpoint = closest_flat % 2

        repeated_left_start = np.repeat(
            endpoints[left_index, 0][None, :], len(right_indices), axis=0
        )
        repeated_left_end = np.repeat(
            endpoints[left_index, 1][None, :], len(right_indices), axis=0
        )
        segment_gaps = np.minimum.reduce(
            (
                point_to_segments(repeated_left_start, right_endpoints[:, 0], right_endpoints[:, 1]),
                point_to_segments(repeated_left_end, right_endpoints[:, 0], right_endpoints[:, 1]),
                point_to_segments(right_endpoints[:, 0], repeated_left_start, repeated_left_end),
                point_to_segments(right_endpoints[:, 1], repeated_left_start, repeated_left_end),
            )
        )

        junction_mask = (angles >= 25.0) & (segment_gaps <= junction_tolerance)
        orthogonal_mask = junction_mask & (angles >= 60.0)
        endpoint_mask = (angles >= 25.0) & (endpoint_gaps <= junction_tolerance)
        junction_counts[left_index] += int(np.count_nonzero(junction_mask))
        junction_counts[right_indices[junction_mask]] += 1
        orthogonal_counts[left_index] += int(np.count_nonzero(orthogonal_mask))
        orthogonal_counts[right_indices[orthogonal_mask]] += 1
        endpoint_counts[left_index] += int(np.count_nonzero(endpoint_mask))
        endpoint_counts[right_indices[endpoint_mask]] += 1
        for left_endpoint in (0, 1):
            if np.any(endpoint_mask & (closest_left_endpoint == left_endpoint)):
                connected_endpoint_masks[left_index] |= np.uint8(1 << left_endpoint)
        matched_right = right_indices[endpoint_mask]
        matched_right_endpoints = closest_right_endpoint[endpoint_mask]
        for right_index, right_endpoint in zip(matched_right, matched_right_endpoints):
            connected_endpoint_masks[right_index] |= np.uint8(1 << int(right_endpoint))

        midpoint_delta = midpoints[right_slice] - midpoints[left_index]
        left_projection = midpoint_delta @ axes[left_index]
        left_offset = np.linalg.norm(
            midpoint_delta - left_projection[:, None] * axes[left_index], axis=1
        )
        right_projection = np.einsum("ij,ij->i", midpoint_delta, right_axes)
        right_offset = np.linalg.norm(
            midpoint_delta - right_projection[:, None] * right_axes, axis=1
        )
        transverse_offsets = np.maximum(left_offset, right_offset)
        parallel_mask = (angles <= 7.5) & (transverse_offsets <= parallel_offset_tolerance)
        parallel_counts[left_index] += int(np.count_nonzero(parallel_mask))
        parallel_counts[right_indices[parallel_mask]] += 1

        relative_right = right_endpoints - midpoints[left_index][None, None, :]
        right_intervals = np.sort(relative_right @ axes[left_index], axis=1)
        left_half = 0.5 * lengths[left_index]
        overlaps = np.maximum(
            0.0,
            np.minimum(left_half, right_intervals[:, 1])
            - np.maximum(-left_half, right_intervals[:, 0]),
        )
        shorter_overlap = overlaps / np.maximum(
            np.minimum(lengths[left_index], lengths[right_slice]), 1e-9
        )
        duplicate_mask = (
            (angles <= 7.5)
            & (transverse_offsets <= 5.0)
            & (shorter_overlap >= 0.45)
        )
        duplicate_counts[left_index] += int(np.count_nonzero(duplicate_mask))
        duplicate_counts[right_indices[duplicate_mask]] += 1

    values = []
    for index in range(count):
        values.append(
            {
                "arrangement_junction_neighbor_count": float(junction_counts[index]),
                "arrangement_orthogonal_junction_neighbor_count": float(
                    orthogonal_counts[index]
                ),
                "arrangement_endpoint_neighbor_count": float(endpoint_counts[index]),
                "arrangement_connected_endpoint_count": float(
                    int(bool(connected_endpoint_masks[index] & 1))
                    + int(bool(connected_endpoint_masks[index] & 2))
                ),
                "arrangement_parallel_family_count": float(parallel_counts[index]),
                "arrangement_near_duplicate_count": float(duplicate_counts[index]),
            }
        )
    base_names = ARRANGEMENT_FEATURE_NAMES[:6]
    for candidate, item in zip(candidates, values):
        candidate.update(item)
    for name in base_names:
        ranked = rankdata([float(candidate[name]) for candidate in candidates], method="average")
        for candidate, rank in zip(candidates, ranked):
            candidate[f"case_percentile_{name}"] = float(rank / len(candidates))


def _features(
    candidate: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    include_arrangement: bool = False,
) -> list[float]:
    endpoints = _endpoints(dict(candidate))
    span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    bbox = np.asarray(payload.get("scan_bbox_mm") or (1.0, 1.0, 1.0), dtype=np.float64)
    diagonal = max(float(np.linalg.norm(bbox)), 1.0)
    support = float(
        candidate.get("support_count")
        or min(candidate.get("incident_plane_support_counts") or (0,))
        or 0.0
    )
    point_count = max(float(payload.get("transient_sampled_point_count") or 1.0), 1.0)
    source = str(candidate.get("representative_source") or candidate.get("candidate_source") or "")
    cluster_sources = {str(value) for value in candidate.get("cluster_sources") or ()}
    values = [
        span,
        span / diagonal,
        float(candidate.get("radius_mm") or 0.0),
        float(candidate.get("transverse_offset_mm") or 0.0),
        float(candidate.get("vote_count") or 0.0),
        support,
        support / point_count,
        float(candidate.get("normal_spread_deg") or 0.0),
        float(candidate.get("score") or 0.0),
        float(candidate.get("confidence") or 0.0),
        float(candidate.get("stability_score") or 0.0),
        float(candidate.get("angle_deg") or 0.0),
        float(candidate.get("cross_width_mm") or candidate.get("cross_fold_width_mm") or 0.0),
        float(candidate.get("consensus_count") or 0.0),
        float(candidate.get("radius_consensus_count") or 0.0),
        support / max(span, 1.0),
        float(candidate.get("finite_support_fraction") or 0.0),
        float(candidate.get("finite_support_longest_run_fraction") or 0.0),
        float(candidate.get("endpoint_support_fraction") or 0.0),
        float(candidate.get("median_line_distance_mm") or 0.0),
        float(candidate.get("p90_line_distance_mm") or 0.0),
        float(candidate.get("transverse_angular_span_deg") or 0.0),
        float(candidate.get("transverse_angular_bin_fraction") or 0.0),
        float(candidate.get("normal_second_eigenvalue") or 0.0),
        float(candidate.get("normal_spread_deg_scan") or 0.0),
        float(candidate.get("corridor_points_per_mm") or 0.0),
        float(candidate.get("cluster_member_count") or 1.0),
        float(candidate.get("cluster_source_count") or 1.0),
        float(candidate.get("cluster_max_finite_support_fraction") or 0.0),
        float(candidate.get("cluster_max_endpoint_support_fraction") or 0.0),
        float(candidate.get("cluster_max_normal_second_eigenvalue") or 0.0),
        float(candidate.get("cluster_max_normal_spread_deg_scan") or 0.0),
        float(candidate.get("cluster_mean_median_line_distance_mm") or 0.0),
        float(candidate.get("cluster_max_two_sided_transition_score") or 0.0),
        float(candidate.get("cluster_max_two_sided_normal_angle_deg") or 0.0),
        float(candidate.get("cluster_max_two_sided_support_balance") or 0.0),
        float(candidate.get("cluster_max_two_sided_min_support_fraction") or 0.0),
        float(candidate.get("cluster_max_two_sided_normal_coherence") or 0.0),
        float(candidate.get("cluster_max_tight_normal_second_eigenvalue_r2mm") or 0.0),
        float(candidate.get("cluster_max_tight_to_wide_normal_second_ratio") or 0.0),
        float(candidate.get("case_percentile_support_count") or 0.0),
        float(candidate.get("case_percentile_score") or 0.0),
        float(candidate.get("case_percentile_consensus_count") or 0.0),
        float(candidate.get("case_percentile_finite_support_fraction") or 0.0),
        float(candidate.get("case_percentile_finite_support_longest_run_fraction") or 0.0),
        float(candidate.get("case_percentile_endpoint_support_fraction") or 0.0),
        float(candidate.get("case_percentile_normal_second_eigenvalue") or 0.0),
        float(candidate.get("case_percentile_normal_spread_deg_scan") or 0.0),
        float(candidate.get("case_percentile_corridor_points_per_mm") or 0.0),
        float(candidate.get("case_percentile_transverse_angular_span_deg") or 0.0),
        float(candidate.get("case_percentile_cluster_member_count") or 0.0),
        float(candidate.get("case_percentile_cluster_source_count") or 0.0),
        float(candidate.get("case_percentile_two_sided_transition_score") or 0.0),
        float(candidate.get("case_percentile_two_sided_normal_angle_deg") or 0.0),
        float(candidate.get("case_percentile_two_sided_support_balance") or 0.0),
        float(candidate.get("case_percentile_two_sided_min_support_fraction") or 0.0),
        float(candidate.get("case_percentile_two_sided_normal_coherence") or 0.0),
        float(candidate.get("case_percentile_tight_normal_second_eigenvalue_r2mm") or 0.0),
        float(candidate.get("case_percentile_tight_to_wide_normal_second_ratio") or 0.0),
    ]
    values.extend(float(source == name or name in cluster_sources) for name in SOURCES)
    if include_arrangement:
        values.extend(float(candidate.get(name) or 0.0) for name in ARRANGEMENT_FEATURE_NAMES)
    return values


def _match_quality(
    candidate: Mapping[str, Any],
    gold: Sequence[Mapping[str, Any]],
    thresholds: Thresholds,
) -> float:
    qualities = []
    for item in gold:
        metrics = pair_metrics(_endpoints(dict(item)), _endpoints(dict(candidate)))
        if not is_match(metrics, thresholds):
            continue
        cost = (
            metrics["axis_angle_deg"] / thresholds.angle_deg
            + metrics["transverse_offset_mm"] / thresholds.transverse_offset_mm
            + (1.0 - min(metrics["gold_span_overlap"], 1.0))
            + (1.0 - min(metrics["prediction_span_overlap"], 1.0))
        )
        qualities.append(1.0 / (1.0 + cost))
    return max(qualities, default=0.0)


def _duplicate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    offset_tolerance_mm: float = 5.0,
) -> bool:
    metrics = pair_metrics(_endpoints(dict(left)), _endpoints(dict(right)))
    return (
        metrics["axis_angle_deg"] <= 7.5
        and metrics["transverse_offset_mm"] <= float(offset_tolerance_mm)
        and metrics["shorter_span_overlap"] >= 0.45
    )


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    qualities: np.ndarray,
    *,
    model_kind: str,
    sample_weights: np.ndarray | None = None,
):
    if model_kind == "logistic":
        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.20,
                class_weight="balanced",
                max_iter=2000,
                random_state=17,
            ),
        )
        fit_kwargs = (
            {"logisticregression__sample_weight": sample_weights}
            if sample_weights is not None
            else {}
        )
        classifier.fit(features, labels, **fit_kwargs)
        return classifier, None
    classifier = ExtraTreesClassifier(
        n_estimators=160,
        class_weight="balanced",
        random_state=17,
        n_jobs=1,
        max_features=None,
        min_samples_leaf=2,
    )
    regressor = ExtraTreesRegressor(
        n_estimators=160,
        random_state=17,
        n_jobs=1,
        max_features=None,
        min_samples_leaf=2,
    )
    classifier.fit(features, labels, sample_weight=sample_weights)
    regressor.fit(features, qualities, sample_weight=sample_weights)
    return classifier, regressor


def _predict(
    classifier: Any,
    regressor: Any,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = classifier.predict_proba(features)[:, 1]
    predicted_qualities = probabilities if regressor is None else regressor.predict(features)
    return probabilities, predicted_qualities


def _ranked_selection(
    candidates: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    predicted_qualities: Sequence[float],
    *,
    probability_threshold: float,
    cylindrical_offset_cap_mm: float,
    duplicate_offset_tolerance_mm: float = 5.0,
    two_sided_min_support_gate: float = 0.0,
    topology_completion_probability_delta: float = 0.0,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            (float(quality), float(probability), str(candidate.get("candidate_source") or ""), dict(candidate))
            for candidate, probability, quality in zip(candidates, probabilities, predicted_qualities)
            if float(probability) >= probability_threshold
            and (
                float(two_sided_min_support_gate) <= 0.0
                or float(candidate.get("cluster_max_two_sided_min_support_fraction") or 0.0)
                >= float(two_sided_min_support_gate)
            )
            and not (
                str(candidate.get("representative_source") or candidate.get("candidate_source") or "").startswith(
                    "cylindrical"
                )
                and float(candidate.get("transverse_offset_mm") or 0.0) > cylindrical_offset_cap_mm
            )
        ),
        key=lambda item: (-item[0], -item[1], item[2], str(item[3].get("bend_id") or "")),
    )
    selected: list[dict[str, Any]] = []
    for quality, probability, _source, candidate in ranked:
        if any(
            _duplicate(
                candidate,
                retained,
                offset_tolerance_mm=duplicate_offset_tolerance_mm,
            )
            for retained in selected
        ):
            continue
        candidate["selection_probability"] = probability
        candidate["predicted_alignment_quality"] = quality
        selected.append(candidate)
    if float(topology_completion_probability_delta) > 0.0 and selected:
        selected_ids = {str(candidate.get("candidate_id")) for candidate in selected}
        completion_ranked = sorted(
            (
                (float(quality), float(probability), dict(candidate))
                for candidate, probability, quality in zip(
                    candidates,
                    probabilities,
                    predicted_qualities,
                )
                if str(candidate.get("candidate_id")) not in selected_ids
                and float(probability)
                >= probability_threshold - float(topology_completion_probability_delta)
                and float(candidate.get("cluster_max_two_sided_min_support_fraction") or 0.0) >= 0.30
                and float(candidate.get("cluster_max_two_sided_normal_angle_deg") or 0.0) >= 20.0
                and float(candidate.get("cluster_max_finite_support_fraction") or 0.0) >= 0.50
            ),
            key=lambda item: (-item[0], -item[1], str(item[2].get("candidate_id") or "")),
        )
        completion_limit = max(2, min(6, len(selected)))
        completed = 0
        for quality, probability, candidate in completion_ranked:
            if any(_duplicate(candidate, retained) for retained in selected):
                continue
            candidate_endpoints = _endpoints(candidate)
            candidate_axis = candidate_endpoints[1] - candidate_endpoints[0]
            candidate_axis /= max(float(np.linalg.norm(candidate_axis)), 1e-12)
            connected = False
            for retained in selected:
                retained_endpoints = _endpoints(retained)
                retained_axis = retained_endpoints[1] - retained_endpoints[0]
                retained_axis /= max(float(np.linalg.norm(retained_axis)), 1e-12)
                angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip(abs(float(np.dot(candidate_axis, retained_axis))), 0.0, 1.0)
                        )
                    )
                )
                segment_gap = min(
                    _point_segment_distance(candidate_endpoints[0], retained_endpoints),
                    _point_segment_distance(candidate_endpoints[1], retained_endpoints),
                    _point_segment_distance(retained_endpoints[0], candidate_endpoints),
                    _point_segment_distance(retained_endpoints[1], candidate_endpoints),
                )
                if angle >= 25.0 and segment_gap <= 10.0:
                    connected = True
                    break
            if not connected:
                continue
            candidate["selection_probability"] = probability
            candidate["predicted_alignment_quality"] = quality
            candidate["selection_reason"] = "scan_only_sparse_topology_completion"
            selected.append(candidate)
            completed += 1
            if completed >= completion_limit:
                break
    return selected


def _line_metrics(
    case_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    from scripts.evaluate_f1_against_manual_gold import match_lines

    thresholds = Thresholds()
    tp = fp = fn = exact = 0
    for case_id, predictions in case_predictions.items():
        result = match_lines(list(gold_by_case[case_id]), list(predictions), thresholds)
        tp += result["true_positive"]
        fp += result["false_positive"]
        fn += result["false_negative"]
        exact += int(result["exact"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "exact": float(exact), "fp": float(fp), "tp": float(tp), "fn": float(fn)}


def _load_additional_training_data(
    candidate_root: Path,
    label_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    labels_by_group: dict[str, list[dict[str, Any]]] = {}
    for labels_path in sorted(label_root.glob("*/bend_instance_labels.json")):
        payload = _read(labels_path)
        if payload.get("authoritative_ground_truth") is not False:
            raise ValueError(f"additional labels must be explicitly non-authoritative: {labels_path}")
        if payload.get("review_status") != "visual_review_complete_training_only":
            continue
        accepted = [
            dict(bend)
            for bend in payload.get("manual_bends") or ()
            if bend.get("review_decision") == "accepted"
        ]
        if not accepted:
            continue
        labels_by_group[part_group(str(payload["job_id"]))] = accepted

    payloads = {
        path.stem: _read(path)
        for path in sorted(candidate_root.glob("[0-9][0-9]_*.json"))
    }
    gold: dict[str, list[dict[str, Any]]] = {}
    for case_id in payloads:
        group = part_group(case_id)
        if group not in labels_by_group:
            raise ValueError(f"{case_id}: no complete accepted training labels for physical part {group}")
        gold[case_id] = labels_by_group[group]
    if not payloads:
        raise ValueError(f"no additional candidate payloads found under {candidate_root}")
    return payloads, gold


def _overlay_two_sided_features(
    candidates: Sequence[dict[str, Any]],
    diagnostic_candidates: Sequence[Mapping[str, Any]],
) -> int:
    def geometry_key(candidate: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        endpoints = _endpoints(dict(candidate))
        ordered = sorted(tuple(float(value) for value in np.round(point, 6)) for point in endpoints)
        return ordered[0], ordered[1]

    by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in diagnostic_candidates
        if candidate.get("candidate_id") is not None
    }
    by_geometry = {geometry_key(candidate): candidate for candidate in diagnostic_candidates}
    matched = 0
    for candidate in candidates:
        diagnostic = by_id.get(str(candidate.get("candidate_id")))
        if diagnostic is None:
            diagnostic = by_geometry.get(geometry_key(candidate))
        if diagnostic is None:
            continue
        matched += 1
        for key, value in diagnostic.items():
            if "two_sided_" in key:
                candidate[key] = value
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "output/experiments/f1-10of13/strict-blind-uniform-candidate-pool-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/f1-10of13/grouped-holdout-selector-v1",
    )
    parser.add_argument(
        "--candidate-kind",
        choices=("line_objects", "raw_candidates"),
        default="line_objects",
    )
    parser.add_argument(
        "--model-kind",
        choices=("extra_trees", "logistic"),
        default="extra_trees",
    )
    parser.add_argument(
        "--tune-cylindrical-offset-cap",
        action="store_true",
        help="Tune a cylinder seed-offset cap using inner folds only.",
    )
    parser.add_argument(
        "--tune-duplicate-offset-tolerance",
        action="store_true",
        help="Tune parallel-line alias suppression using inner folds only.",
    )
    parser.add_argument(
        "--additional-training-candidate-root",
        type=Path,
        help="Identity-free candidate pools used only as auxiliary training groups.",
    )
    parser.add_argument(
        "--additional-training-label-root",
        type=Path,
        help="Complete, visually reviewed, explicitly non-authoritative auxiliary labels.",
    )
    parser.add_argument(
        "--tune-additional-training-weight",
        action="store_true",
        help="Tune auxiliary influence using inner original-corpus folds only.",
    )
    parser.add_argument(
        "--arrangement-features",
        action="store_true",
        help="Add identity-free candidate-set junction and parallel-family features.",
    )
    parser.add_argument(
        "--tune-two-sided-support-gate",
        action="store_true",
        help="Tune a hard two-planar-region support gate using inner folds only.",
    )
    parser.add_argument(
        "--two-sided-feature-root",
        type=Path,
        help="Diagnostic pool supplying only two-sided fields by stable candidate ID.",
    )
    parser.add_argument(
        "--tune-topology-completion",
        action="store_true",
        help="Tune sparse junction completion probability delta using inner folds only.",
    )
    args = parser.parse_args()
    if bool(args.additional_training_candidate_root) != bool(args.additional_training_label_root):
        parser.error("additional training candidate and label roots must be supplied together")
    if args.tune_additional_training_weight and not args.additional_training_candidate_root:
        parser.error("additional training weight tuning requires auxiliary training roots")

    candidate_root = args.candidate_root.resolve()
    payloads = {
        path.stem: _read(path)
        for path in sorted(candidate_root.glob("[0-9][0-9]_*.json"))
    }
    evaluation_case_ids = sorted(payloads)
    thresholds = Thresholds()
    gold = {
        case_id: _read(ROOT / "output" / CASE_JOBS[case_id] / "bend_instance_labels.json")["manual_bends"]
        for case_id in evaluation_case_ids
    }
    additional_case_ids: list[str] = []
    if args.additional_training_candidate_root:
        additional_payloads, additional_gold = _load_additional_training_data(
            args.additional_training_candidate_root.resolve(),
            args.additional_training_label_root.resolve(),
        )
        overlap = sorted(set(payloads) & set(additional_payloads))
        if overlap:
            raise ValueError(f"additional training case IDs collide with evaluation cases: {overlap}")
        payloads.update(additional_payloads)
        gold.update(additional_gold)
        additional_case_ids = sorted(additional_payloads)
    all_case_ids = sorted(payloads)
    rows: dict[str, list[tuple[dict[str, Any], list[float], int, float, bool]]] = {}
    two_sided_payloads = (
        {
            path.stem: _read(path)
            for path in sorted(args.two_sided_feature_root.resolve().glob("[0-9][0-9]_*.json"))
        }
        if args.two_sided_feature_root
        else {}
    )
    overlay_coverage: dict[str, dict[str, int]] = {}
    for case_id, payload in payloads.items():
        case_rows = []
        candidate_key = "line_objects" if args.candidate_kind == "line_objects" else "candidates"
        candidates = list(payload.get(candidate_key) or ())
        if case_id in two_sided_payloads:
            matched = _overlay_two_sided_features(
                candidates,
                list(two_sided_payloads[case_id].get(candidate_key) or ()),
            )
            overlay_coverage[case_id] = {"matched": matched, "candidate_count": len(candidates)}
            if matched != len(candidates):
                raise ValueError(
                    f"{case_id}: two-sided diagnostic overlay matched {matched}/{len(candidates)} candidates"
                )
        if args.arrangement_features:
            _attach_arrangement_features(candidates, payload)
        for candidate in candidates:
            quality = _match_quality(candidate, gold[case_id], thresholds)
            case_rows.append(
                (
                    candidate,
                    _features(candidate, payload, include_arrangement=args.arrangement_features),
                    int(quality > 0.0),
                    quality,
                    case_id in additional_case_ids,
                )
            )
        rows[case_id] = case_rows

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_map: dict[str, Path] = {}
    fold_rows = []
    offset_caps = CYLINDRICAL_OFFSET_CAP_GRID if args.tune_cylindrical_offset_cap else (1.0e9,)
    duplicate_offsets = (
        DUPLICATE_OFFSET_TOLERANCE_GRID
        if args.tune_duplicate_offset_tolerance
        else (5.0,)
    )
    additional_weights = (
        ADDITIONAL_TRAINING_WEIGHT_GRID
        if args.tune_additional_training_weight
        else (1.0,)
    )
    two_sided_support_gates = (
        TWO_SIDED_MIN_SUPPORT_GATE_GRID
        if args.tune_two_sided_support_gate
        else (0.0,)
    )
    topology_completion_deltas = (
        TOPOLOGY_COMPLETION_DELTA_GRID
        if args.tune_topology_completion
        else (0.0,)
    )
    for outer_group in sorted({part_group(case_id) for case_id in evaluation_case_ids}):
        test_cases = [case_id for case_id in evaluation_case_ids if part_group(case_id) == outer_group]
        train_cases = [case_id for case_id in all_case_ids if part_group(case_id) != outer_group]

        inner_predictions = {
            (
                additional_weight,
                support_gate,
                completion_delta,
                threshold,
                offset_cap,
                duplicate_offset,
            ): {}
            for additional_weight in additional_weights
            for support_gate in two_sided_support_gates
            for completion_delta in topology_completion_deltas
            for threshold in THRESHOLD_GRID
            for offset_cap in offset_caps
            for duplicate_offset in duplicate_offsets
        }
        inner_validation_cases = [
            case_id for case_id in train_cases if case_id in evaluation_case_ids
        ]
        for inner_group in sorted({part_group(case_id) for case_id in inner_validation_cases}):
            inner_validation = [
                case_id
                for case_id in inner_validation_cases
                if part_group(case_id) == inner_group
            ]
            inner_training = [case_id for case_id in train_cases if part_group(case_id) != inner_group]
            train_rows = [row for case_id in inner_training for row in rows[case_id]]
            features = np.asarray([row[1] for row in train_rows], dtype=np.float64)
            labels = np.asarray([row[2] for row in train_rows], dtype=np.int64)
            qualities = np.asarray([row[3] for row in train_rows], dtype=np.float64)
            for additional_weight in additional_weights:
                sample_weights = np.asarray(
                    [additional_weight if row[4] else 1.0 for row in train_rows],
                    dtype=np.float64,
                )
                classifier, regressor = _fit(
                    features,
                    labels,
                    qualities,
                    model_kind=args.model_kind,
                    sample_weights=sample_weights,
                )
                for case_id in inner_validation:
                    case_features = np.asarray([row[1] for row in rows[case_id]], dtype=np.float64)
                    probabilities, predicted_qualities = _predict(classifier, regressor, case_features)
                    candidates = [row[0] for row in rows[case_id]]
                    for threshold in THRESHOLD_GRID:
                        for support_gate in two_sided_support_gates:
                            for completion_delta in topology_completion_deltas:
                                for offset_cap in offset_caps:
                                    for duplicate_offset in duplicate_offsets:
                                        inner_predictions[
                                            (
                                                additional_weight,
                                                support_gate,
                                                completion_delta,
                                                threshold,
                                                offset_cap,
                                                duplicate_offset,
                                            )
                                        ][case_id] = _ranked_selection(
                                            candidates,
                                            probabilities,
                                            predicted_qualities,
                                            probability_threshold=threshold,
                                            cylindrical_offset_cap_mm=offset_cap,
                                            duplicate_offset_tolerance_mm=duplicate_offset,
                                            two_sided_min_support_gate=support_gate,
                                            topology_completion_probability_delta=completion_delta,
                                        )
        scored_thresholds = [
            (
                additional_weight,
                support_gate,
                completion_delta,
                threshold,
                offset_cap,
                duplicate_offset,
                _line_metrics(
                    inner_predictions[
                        (
                            additional_weight,
                            support_gate,
                            completion_delta,
                            threshold,
                            offset_cap,
                            duplicate_offset,
                        )
                    ],
                    gold,
                ),
            )
            for additional_weight in additional_weights
            for support_gate in two_sided_support_gates
            for completion_delta in topology_completion_deltas
            for threshold in THRESHOLD_GRID
            for offset_cap in offset_caps
            for duplicate_offset in duplicate_offsets
        ]
        baseline_choice = max(
            [item for item in scored_thresholds if item[1] == 0.0 and item[2] == 0.0],
            key=lambda item: (
                item[6]["f1"],
                item[6]["exact"],
                -item[6]["fp"],
                -item[0],
                -item[1],
                item[2],
                item[3],
                -item[4],
                -item[5],
            ),
        )
        gate_choices = [
            item
            for item in scored_thresholds
            if item[0] == baseline_choice[0]
            and item[3] == baseline_choice[3]
            and item[4] == baseline_choice[4]
            and item[5] == baseline_choice[5]
        ]
        (
            selected_additional_weight,
            selected_two_sided_support_gate,
            selected_topology_completion_delta,
            selected_threshold,
            selected_offset_cap,
            selected_duplicate_offset,
            selected_inner_metrics,
        ) = max(
            gate_choices,
            key=lambda item: (
                item[6]["f1"],
                item[6]["exact"],
                -item[6]["fp"],
                -item[1],
                -item[2],
            ),
        )

        train_rows = [row for case_id in train_cases for row in rows[case_id]]
        features = np.asarray([row[1] for row in train_rows], dtype=np.float64)
        labels = np.asarray([row[2] for row in train_rows], dtype=np.int64)
        qualities = np.asarray([row[3] for row in train_rows], dtype=np.float64)
        classifier, regressor = _fit(
            features,
            labels,
            qualities,
            model_kind=args.model_kind,
            sample_weights=np.asarray(
                [selected_additional_weight if row[4] else 1.0 for row in train_rows],
                dtype=np.float64,
            ),
        )
        for case_id in test_cases:
            case_features = np.asarray([row[1] for row in rows[case_id]], dtype=np.float64)
            probabilities, predicted_qualities = _predict(classifier, regressor, case_features)
            selected = _ranked_selection(
                [row[0] for row in rows[case_id]],
                probabilities,
                predicted_qualities,
                probability_threshold=selected_threshold,
                cylindrical_offset_cap_mm=selected_offset_cap,
                duplicate_offset_tolerance_mm=selected_duplicate_offset,
                two_sided_min_support_gate=selected_two_sided_support_gate,
                topology_completion_probability_delta=selected_topology_completion_delta,
            )
            for index, candidate in enumerate(selected, start=1):
                candidate["bend_id"] = f"F1GH{index}"
            case_dir = output_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = case_dir / "decomposition_result.json"
            prediction_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "algorithm": "f1_nested_grouped_holdout_selector_v1",
                        "evaluation_mode": "nested_leave_one_physical_part_out",
                        "held_out_part_group": outer_group,
                        "training_case_ids": train_cases,
                        "additional_training_weight": selected_additional_weight,
                        "two_sided_min_support_gate": selected_two_sided_support_gate,
                        "topology_completion_probability_delta": selected_topology_completion_delta,
                        "selection_probability_threshold": selected_threshold,
                        "cylindrical_offset_cap_mm": selected_offset_cap,
                        "duplicate_offset_tolerance_mm": selected_duplicate_offset,
                        "source_point_serialization": False,
                        "owned_bend_regions": selected,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            prediction_map[case_id] = prediction_path
        fold_rows.append(
            {
                "held_out_part_group": outer_group,
                "held_out_case_ids": test_cases,
                "training_case_ids": train_cases,
                "selected_additional_training_weight": selected_additional_weight,
                "selected_two_sided_min_support_gate": selected_two_sided_support_gate,
                "selected_topology_completion_probability_delta": selected_topology_completion_delta,
                "selected_threshold": selected_threshold,
                "selected_cylindrical_offset_cap_mm": selected_offset_cap,
                "selected_duplicate_offset_tolerance_mm": selected_duplicate_offset,
                "inner_training_metrics": selected_inner_metrics,
            }
        )

    report = evaluate(prediction_map, thresholds)
    report["evaluation_mode"] = "nested_leave_one_physical_part_out"
    report["leakage_contract"] = {
        "same_scan_labels_in_training": False,
        "same_part_labels_in_training": False,
        "outer_test_labels_used_for_threshold_selection": False,
        "candidate_generation_uses_manual_truth": False,
        "additional_training_cases_used_as_outer_tests": False,
        "additional_training_cases_used_for_inner_threshold_selection": False,
        "additional_training_labels_authoritative": False,
        "locked_blind_release_claim": False,
    }
    report["additional_training_case_ids"] = additional_case_ids
    report["tuned_additional_training_weight"] = bool(args.tune_additional_training_weight)
    report["tuned_two_sided_support_gate"] = bool(args.tune_two_sided_support_gate)
    report["tuned_topology_completion"] = bool(args.tune_topology_completion)
    report["two_sided_feature_overlay"] = overlay_coverage
    report["feature_names"] = list(FEATURE_NAMES) + (
        list(ARRANGEMENT_FEATURE_NAMES) if args.arrangement_features else []
    )
    report["arrangement_features"] = bool(args.arrangement_features)
    report["candidate_kind"] = args.candidate_kind
    report["model_kind"] = args.model_kind
    report["tuned_cylindrical_offset_cap"] = bool(args.tune_cylindrical_offset_cap)
    report["tuned_duplicate_offset_tolerance"] = bool(
        args.tune_duplicate_offset_tolerance
    )
    report["folds"] = fold_rows
    (output_dir / "grouped_holdout_line_accuracy.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "case_map.json").write_text(
        json.dumps({key: str(value) for key, value in prediction_map.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
