"""Fixed-set bend-local audit for an executed CAD/scan registration refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AuditConfig:
    line_sample_count: int = 201
    support_distance_mm: float = 3.0
    minimum_pre_support_fraction: float = 0.25
    minimum_pre_supported_samples: int = 50
    support_drop_absolute: float = 0.10
    median_increase_mm: float = 0.25
    median_relative_increase: float = 0.20
    p90_increase_mm: float = 0.50
    p90_relative_increase: float = 0.20


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    return source @ matrix[:3, :3].T + matrix[:3, 3]


def line_samples(start: Sequence[float], end: Sequence[float], count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("line sample count must be at least two")
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    if first.shape != (3,) or last.shape != (3,) or not np.all(np.isfinite([first, last])):
        raise ValueError("line endpoints must be finite 3-vectors")
    return first[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (last - first)[None, :]


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool).tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def support_metrics(line: np.ndarray, transformed_scan_points: np.ndarray, config: AuditConfig) -> dict[str, Any]:
    samples = np.asarray(line, dtype=np.float64)
    scan = np.asarray(transformed_scan_points, dtype=np.float64)
    if len(samples) != config.line_sample_count:
        raise ValueError("unexpected fixed line sample count")
    if scan.ndim != 2 or scan.shape[1] != 3 or len(scan) == 0:
        raise ValueError("transformed scan must be nonempty Nx3")
    distances, _ = cKDTree(scan).query(samples, k=1, workers=1)
    supported = distances <= config.support_distance_mm
    supported_count = int(np.sum(supported))
    return {
        "sample_count": int(len(samples)),
        "supported_sample_count": supported_count,
        "support_fraction": float(supported_count / len(samples)),
        "longest_contiguous_support_fraction": float(_longest_true_run(supported) / len(samples)),
        "median_distance_mm": float(np.median(distances)),
        "p90_distance_mm": float(np.percentile(distances, 90)),
        "maximum_distance_mm": float(np.max(distances)),
    }


def global_residual_metrics(
    transformed_scan_points: np.ndarray,
    reference_points: np.ndarray,
) -> dict[str, Any]:
    scan = np.asarray(transformed_scan_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    if len(scan) == 0 or len(reference) == 0:
        raise ValueError("global audit point sets must be nonempty")
    distances, _ = cKDTree(reference).query(scan, k=1, workers=1)
    return {
        "point_count": int(len(scan)),
        "median_distance_mm": float(np.median(distances)),
        "p90_distance_mm": float(np.percentile(distances, 90)),
        "mean_distance_mm": float(np.mean(distances)),
    }


def classify_reversal(before: Mapping[str, Any], after: Mapping[str, Any], config: AuditConfig) -> dict[str, Any]:
    pre_support = float(before["support_fraction"])
    pre_count = int(before["supported_sample_count"])
    eligible = bool(
        pre_support >= config.minimum_pre_support_fraction
        and pre_count >= config.minimum_pre_supported_samples
    )
    support_drop = pre_support - float(after["support_fraction"])
    median_before = float(before["median_distance_mm"])
    median_increase = float(after["median_distance_mm"]) - median_before
    p90_before = float(before["p90_distance_mm"])
    p90_increase = float(after["p90_distance_mm"]) - p90_before
    median_relative = median_increase / max(median_before, 1e-12)
    p90_relative = p90_increase / max(p90_before, 1e-12)
    support_trigger = support_drop >= config.support_drop_absolute
    median_trigger = bool(
        median_increase >= config.median_increase_mm
        and median_relative >= config.median_relative_increase
    )
    p90_trigger = bool(
        p90_increase >= config.p90_increase_mm
        and p90_relative >= config.p90_relative_increase
    )
    reversal = bool(eligible and (support_trigger or median_trigger or p90_trigger))
    return {
        "eligible_pre_step_support": eligible,
        "support_fraction_drop": float(support_drop),
        "median_distance_increase_mm": float(median_increase),
        "median_distance_relative_increase": float(median_relative),
        "p90_distance_increase_mm": float(p90_increase),
        "p90_distance_relative_increase": float(p90_relative),
        "support_trigger": bool(support_trigger),
        "median_trigger": bool(median_trigger),
        "p90_trigger": bool(p90_trigger),
        "material_local_reversal": reversal,
    }


def audit_transform_pair(
    *,
    processed_scan_points: np.ndarray,
    reference_points: np.ndarray,
    reference_center: Sequence[float],
    global_transform: np.ndarray,
    datum_transform: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    config: AuditConfig = AuditConfig(),
) -> dict[str, Any]:
    scan_global = apply_transform(processed_scan_points, global_transform)
    scan_datum = apply_transform(processed_scan_points, datum_transform)
    center = np.asarray(reference_center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("reference center must be a 3-vector")
    records = []
    for event in sorted(events, key=lambda row: str(row["event_id"])):
        samples = line_samples(
            np.asarray(event["cad_line_start_mm"], dtype=np.float64) - center,
            np.asarray(event["cad_line_end_mm"], dtype=np.float64) - center,
            config.line_sample_count,
        )
        before = support_metrics(samples, scan_global, config)
        after = support_metrics(samples, scan_datum, config)
        classification = classify_reversal(before, after, config)
        records.append({
            "event_id": str(event["event_id"]),
            "global_icp": before,
            "datum_refinement": after,
            "classification": classification,
        })
    reversal_ids = [
        row["event_id"] for row in records
        if row["classification"]["material_local_reversal"]
    ]
    return {
        "schema": "FIXED_CAD_EVENT_REGISTRATION_EXECUTION_AUDIT_V1",
        "config": asdict(config),
        "global_metrics": {
            "global_icp": global_residual_metrics(scan_global, reference_points),
            "datum_refinement": global_residual_metrics(scan_datum, reference_points),
        },
        "event_count": int(len(records)),
        "eligible_event_count": int(sum(row["classification"]["eligible_pre_step_support"] for row in records)),
        "material_local_reversal_count": int(len(reversal_ids)),
        "material_local_reversal_event_ids": reversal_ids,
        "records": records,
    }


def contract_payload() -> dict[str, Any]:
    return {
        "schema": "FIXED_CAD_EVENT_REGISTRATION_EXECUTION_AUDIT_V1",
        "config": asdict(AuditConfig()),
        "same_points_before_after": True,
        "changes_predictions": False,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }

