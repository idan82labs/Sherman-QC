"""Fail-closed canonical semantics for CAD-to-scan registration qualification."""

from __future__ import annotations

import json
import math
from typing import Any


DECIMAL_PLACES = 9


def canonicalize(value: Any, decimal_places: int = DECIMAL_PLACES) -> Any:
    """Recursively quantize finite floats and normalize negative zero."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite number in semantic payload")
        rounded = round(value, decimal_places)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, list):
        return [canonicalize(item, decimal_places) for item in value]
    if isinstance(value, tuple):
        return [canonicalize(item, decimal_places) for item in value]
    if isinstance(value, dict):
        return {str(key): canonicalize(item, decimal_places) for key, item in value.items()}
    raise TypeError(f"unsupported semantic value: {type(value).__name__}")


def semantic_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a raw EXP12 diagnostic into the frozen EXP13 semantic contract."""
    selected = bool(raw["datum_selected"])
    payload = {
        "schema": "EXP_CAD_0013_CANONICAL_REGISTRATION_CASE_V1",
        "experiment_id": "EXP_CAD_0013",
        "case_token": raw["case_token"],
        "scan_sha256": raw["scan_sha256"],
        "cad_sha256": raw["cad_sha256"],
        "cad_template_sha256": raw["cad_template_sha256"],
        "deterministic_reference_points_sha256": raw["deterministic_reference_points_sha256"],
        "deterministic_reference_normals_sha256": raw["deterministic_reference_normals_sha256"],
        "processed_scan_point_count": raw["processed_scan_point_count"],
        "reference_point_count": raw["reference_point_count"],
        "voxel_size_mm": raw["voxel_size_mm"],
        "datum_candidate_captured": bool(raw["datum_candidate_captured"]),
        "datum_selected": selected,
        "global_transform": raw["global_transform"],
        "selected_datum_transform": raw["datum_transform"] if selected else None,
        "datum_selection_stats": raw["datum_selection_stats"] if selected else None,
        "material_local_reversal_event_ids": raw["material_local_reversal_event_ids"] if selected else [],
        "global_metrics": raw["global_metrics"] if selected else None,
        "rejected_datum_candidate_fail_closed": not selected,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }
    return canonicalize(payload)


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    canonical = canonicalize(payload)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
