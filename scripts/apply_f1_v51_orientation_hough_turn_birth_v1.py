#!/usr/bin/env python3
"""Orientation-aware Hough births from the frozen local-turn ridge field."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from scripts import apply_f1_v51_local_turn_locus_birth_v1 as ridge
from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics


RESULT_KIND = "sherman_qc_f1_v51_orientation_hough_turn_birth_result_v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _unit(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate direction")
    result = result / norm
    pivot = int(np.argmax(np.abs(result)))
    return -result if result[pivot] < 0.0 else result


def _axis_angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_unit(left), _unit(right)))), 0.0, 1.0))))


def _local_axes(
    points: np.ndarray,
    normals: np.ndarray,
    ridge_indexes: np.ndarray,
    ridge_profile: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    count = min(int(ridge_profile["local_neighbor_count"]), len(points))
    axes = np.zeros((len(ridge_indexes), 3), dtype=np.float64)
    ratios = np.full(len(ridge_indexes), math.inf, dtype=np.float64)
    valid_axis = np.zeros(len(ridge_indexes), dtype=np.bool_)
    for start in range(0, len(ridge_indexes), 20_000):
        stop = min(start + 20_000, len(ridge_indexes))
        query = points[ridge_indexes[start:stop]]
        distances, indexes = tree.query(
            query,
            k=count,
            distance_upper_bound=float(ridge_profile["local_radius_mm"]),
            workers=1,
        )
        if count == 1:
            distances = distances[:, None]
            indexes = indexes[:, None]
        valid = (indexes < len(points)) & np.isfinite(distances)
        safe_indexes = np.where(valid, indexes, 0)
        local_normals = normals[safe_indexes]
        weights = valid.astype(np.float64)
        counts = np.sum(weights, axis=1)
        moment = np.einsum("bki,bkj,bk->bij", local_normals, local_normals, weights)
        moment /= np.maximum(counts[:, None, None], 1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(moment)
        local_axes = eigenvectors[:, :, 0]
        local_ratios = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-12)
        accepted = (
            (counts >= int(ridge_profile["minimum_local_neighbors"]))
            & np.isfinite(local_ratios)
            & (local_ratios <= float(profile["local_axis_maximum_null_to_turn_eigenvalue_ratio"]))
        )
        for local_index, value in enumerate(local_axes):
            if accepted[local_index]:
                axes[start + local_index] = _unit(value)
        ratios[start:stop] = local_ratios
        valid_axis[start:stop] = accepted
    return ridge_indexes[valid_axis], axes[valid_axis], ratios[valid_axis]


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.8:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = _unit(np.cross(axis, reference))
    return first, np.cross(axis, first)


def _orientation_key(axis: np.ndarray, quantum: float) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.rint(_unit(axis) / quantum).astype(np.int64))


def _candidate_rows(
    points: np.ndarray,
    contrast: np.ndarray,
    accepted_indexes: np.ndarray,
    local_axes: np.ndarray,
    profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    orientation_rows: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    quantum = float(profile["orientation_component_quantum"])
    for ordinal, axis in enumerate(local_axes):
        orientation_rows[_orientation_key(axis, quantum)].append(ordinal)
    raw = []
    qualified_orientation_groups = 0
    hough_cells = 0
    supported_runs = 0
    for orientation_key in sorted(orientation_rows):
        ordinals = np.asarray(orientation_rows[orientation_key], dtype=np.int64)
        if len(ordinals) < int(profile["minimum_orientation_group_points"]):
            continue
        qualified_orientation_groups += 1
        axis_moment = local_axes[ordinals].T @ local_axes[ordinals]
        axis = _unit(np.linalg.eigh(axis_moment)[1][:, -1])
        errors = np.asarray([_axis_angle(value, axis) for value in local_axes[ordinals]])
        retained = errors <= float(profile["maximum_local_axis_error_deg"])
        ordinals = ordinals[retained]
        errors = errors[retained]
        if len(ordinals) < int(profile["minimum_orientation_group_points"]):
            continue
        group_indexes = accepted_indexes[ordinals]
        group_points = points[group_indexes]
        first, second = _basis(axis)
        cell_mm = float(profile["transverse_hough_cell_mm"])
        cell_values = np.floor(
            np.column_stack((group_points @ first, group_points @ second)) / cell_mm
        ).astype(np.int64)
        cell_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
        for local_ordinal, cell in enumerate(cell_values.tolist()):
            cell_rows[(int(cell[0]), int(cell[1]))].append(local_ordinal)
        for cell in sorted(cell_rows):
            local_ordinals = np.asarray(cell_rows[cell], dtype=np.int64)
            if len(local_ordinals) < int(profile["minimum_track_points"]):
                continue
            hough_cells += 1
            track_points = group_points[local_ordinals]
            track_indexes = group_indexes[local_ordinals]
            track_errors = errors[local_ordinals]
            projection = track_points @ axis
            bin_mm = float(profile["longitudinal_bin_mm"])
            bins = np.floor(projection / bin_mm).astype(np.int64)
            unique = sorted(set(int(value) for value in bins))
            runs: list[list[int]] = []
            for value in unique:
                if not runs or (value - runs[-1][-1]) * bin_mm > float(profile["maximum_supported_gap_mm"]):
                    runs.append([value])
                else:
                    runs[-1].append(value)
            for run_index, run in enumerate(runs, 1):
                if len(run) < int(profile["minimum_supported_bin_count"]):
                    continue
                run_mask = np.isin(bins, np.asarray(run, dtype=np.int64))
                run_points = track_points[run_mask]
                run_indexes = track_indexes[run_mask]
                run_errors = track_errors[run_mask]
                if len(run_points) < int(profile["minimum_track_points"]):
                    continue
                transverse = run_points - (run_points @ axis)[:, None] * axis
                transverse_center = np.mean(transverse, axis=0)
                radial = transverse - transverse_center
                q80 = float(np.quantile(np.linalg.norm(radial, axis=1), 0.80))
                if q80 > float(profile["maximum_track_transverse_q80_mm"]):
                    continue
                low = float(min(run) * bin_mm)
                high = float((max(run) + 1) * bin_mm)
                span = high - low
                if span < float(profile["minimum_span_mm"]):
                    continue
                supported_runs += 1
                start = transverse_center + low * axis
                end = transverse_center + high * axis
                identity = {
                    "orientation_key": orientation_key,
                    "hough_cell": cell,
                    "run": run_index,
                    "line_start": np.round(start, 9).tolist(),
                    "line_end": np.round(end, 9).tolist(),
                }
                token = hashlib.sha256(canonical_bytes(identity)).hexdigest()[:20]
                median_contrast = float(np.median(contrast[run_indexes]))
                raw.append({
                    "bend_id": f"V51OHTB1-{token}",
                    "line_start": start.tolist(),
                    "line_end": end.tolist(),
                    "source_kind": "orientation_hough_turn_birth_v1",
                    "selection_route": "frozen_ridge_local_axis_orientation_offset_hough",
                    "source_hypothesis_id": f"o{orientation_key}-c{cell}-r{run_index}",
                    "selector_score": float(len(run) * len(run_points) * median_contrast / max(1.0 + q80, 1e-9)),
                    "orientation_hough_evidence": {
                        "orientation_key": list(orientation_key),
                        "hough_cell": list(cell),
                        "track_point_count": int(len(run_points)),
                        "supported_bin_count": int(len(run)),
                        "span_mm": span,
                        "transverse_q80_mm": q80,
                        "local_axis_error_q80_deg": float(np.quantile(run_errors, 0.80)),
                        "median_normal_change_deg": median_contrast,
                    },
                })
    return raw, {
        "orientation_group_count": len(orientation_rows),
        "qualified_orientation_group_count": qualified_orientation_groups,
        "hough_cell_count": hough_cells,
        "supported_run_count": supported_runs,
    }


def _duplicate(left: Mapping[str, Any], right: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    metrics = pair_metrics(_endpoints(left), _endpoints(right))
    return bool(
        metrics["axis_angle_deg"] <= float(profile["duplicate_angle_deg"])
        and metrics["transverse_offset_mm"] <= float(profile["duplicate_offset_mm"])
        and metrics["shorter_span_overlap"] >= float(profile["duplicate_shorter_span_overlap"])
    )


def generate_births(
    scan_path: Path,
    ridge_profile: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points, normals, diagnostics = ridge._geometry_cloud(scan_path, ridge_profile)
    ridge_indexes, _second, contrast = ridge._local_turn_field(points, normals, ridge_profile)
    accepted_indexes, local_axes, ratios = _local_axes(
        points, normals, ridge_indexes, ridge_profile, profile
    )
    raw, hough_diagnostics = _candidate_rows(
        points, contrast, accepted_indexes, local_axes, profile
    )
    raw.sort(key=lambda row: (-float(row["selector_score"]), str(row["bend_id"])))
    kept = []
    for row in raw:
        if any(_duplicate(row, incumbent, profile) for incumbent in kept):
            continue
        kept.append(row)
        if len(kept) >= int(profile["maximum_births_per_scan"]):
            break
    diagnostics.update({
        "turn_ridge_point_count": int(len(ridge_indexes)),
        "local_axis_point_count": int(len(accepted_indexes)),
        "median_local_axis_null_ratio": float(np.median(ratios)) if len(ratios) else None,
        **hough_diagnostics,
        "qualified_birth_count_before_nms_and_cap": int(len(raw)),
        "birth_count_after_nms_and_cap": int(len(kept)),
    })
    return kept, diagnostics


def apply_births(
    v51_result: Mapping[str, Any],
    scan_path: Path,
    ridge_lock: Mapping[str, Any],
    parameter_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if v51_result.get("source_only") is not True or v51_result.get("research_only") is not True:
        raise ValueError("V51 input must be source-only and research-only")
    if any(v51_result.get(name) is not False for name in ("labels_used", "cad_used", "part_identity_used")):
        raise ValueError("V51 input reports forbidden evidence")
    if parameter_lock.get("status") != "LOCKED_BEFORE_CASE_OUTCOME_EVALUATION":
        raise ValueError("orientation-Hough parameter precommit is not locked")
    ridge_file_hash = hashlib.sha256(
        (json.dumps(ridge_lock, indent=2) + "\n").encode()
    ).hexdigest()
    # The on-disk byte binding is validated by the evaluator; this runtime also
    # enforces exact parameter reuse through the supplied frozen profile.
    if ridge_lock.get("status") != "LOCKED_BEFORE_CASE_OUTCOME_EVALUATION":
        raise ValueError("ridge-field precommit is not locked")
    profile = dict(parameter_lock["parameters"])
    births, diagnostics = generate_births(scan_path, ridge_lock["parameters"], profile)
    incumbents = [deepcopy(row) for row in v51_result.get("predictions") or []]
    additions = []
    for row in births:
        if any(_duplicate(row, incumbent, profile) for incumbent in incumbents):
            continue
        if any(_duplicate(row, incumbent, profile) for incumbent in additions):
            continue
        additions.append(row)
    return {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "predictions": incumbents + additions,
        "candidate_count": len(incumbents) + len(additions),
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "orientation_hough_turn_birth_audit": {
            **diagnostics,
            "incumbent_count": len(incumbents),
            "generated_birth_count": len(births),
            "emitted_addition_count": len(additions),
        },
        "parameter_lock_sha256": hashlib.sha256(canonical_bytes(parameter_lock)).hexdigest(),
        "ridge_parameter_lock_canonical_sha256": hashlib.sha256(canonical_bytes(ridge_lock)).hexdigest(),
        "ridge_parameter_lock_pretty_sha256_diagnostic": ridge_file_hash,
    }
