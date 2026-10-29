"""Validation and durable storage for manually adjudicated bend instances."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1
LABEL_KIND = "sherman_qc_bend_instance_labels"
COORDINATE_FRAME = "reference_mesh_raw"
LABELS_FILENAME = "bend_instance_labels.json"
REVIEW_DECISIONS = frozenset({"real_bend", "not_bend_outline", "uncertain"})
REVIEW_STATUSES = frozenset({"in_progress", "complete"})
TOP_LEVEL_KEYS = frozenset({
    "schema_version",
    "kind",
    "job_id",
    "coordinate_frame",
    "review_status",
    "manual_bends",
    "marker_reviews",
    "created_at",
    "updated_at",
})
MANUAL_BEND_KEYS = frozenset({"bend_id", "semantic", "source", "line_start", "line_end"})
MARKER_REVIEW_KEYS = frozenset({"bend_id", "decision", "reason"})


class BendInstanceLabelsValidationError(ValueError):
    """Raised when a bend-label payload violates the persisted contract."""


def empty_bend_instance_labels(job_id: str) -> Dict[str, Any]:
    """Return the deterministic, empty labels document for a job."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LABEL_KIND,
        "job_id": job_id,
        "coordinate_frame": COORDINATE_FRAME,
        "review_status": "in_progress",
        "manual_bends": [],
        "marker_reviews": [],
    }


def _require_nonempty_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BendInstanceLabelsValidationError(f"{path} must be a non-empty string")
    if value != value.strip():
        raise BendInstanceLabelsValidationError(f"{path} must not have leading or trailing whitespace")
    return value


def _require_point(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise BendInstanceLabelsValidationError(f"{path} must be a three-number array")
    coordinates = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise BendInstanceLabelsValidationError(f"{path}[{index}] must be a finite number")
        number = float(coordinate)
        if not math.isfinite(number):
            raise BendInstanceLabelsValidationError(f"{path}[{index}] must be a finite number")
        coordinates.append(number)
    return coordinates[0], coordinates[1], coordinates[2]


def _reject_unexpected_keys(value: Dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unexpected = sorted(str(key) for key in set(value) - allowed)
    if unexpected:
        raise BendInstanceLabelsValidationError(
            f"{path} contains unexpected keys: {', '.join(unexpected)}"
        )


def validate_bend_instance_labels(payload: Any, expected_job_id: str) -> Dict[str, Any]:
    """Validate and normalize a schema-v1 bend-instance label document.

    A point at the coordinate origin is valid. ``nonzero`` refers to the line
    segment itself: its start and end points must not be identical.
    """
    if not isinstance(payload, dict):
        raise BendInstanceLabelsValidationError("payload must be a JSON object")
    _reject_unexpected_keys(payload, TOP_LEVEL_KEYS, "payload")

    if payload.get("schema_version") != SCHEMA_VERSION or isinstance(payload.get("schema_version"), bool):
        raise BendInstanceLabelsValidationError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("kind") != LABEL_KIND:
        raise BendInstanceLabelsValidationError(f"kind must be {LABEL_KIND}")

    job_id = _require_nonempty_id(payload.get("job_id"), "job_id")
    if job_id != expected_job_id:
        raise BendInstanceLabelsValidationError("payload job_id does not match the requested job")
    if payload.get("coordinate_frame") != COORDINATE_FRAME:
        raise BendInstanceLabelsValidationError(f"coordinate_frame must be {COORDINATE_FRAME}")
    review_status = payload.get("review_status")
    if not isinstance(review_status, str) or review_status not in REVIEW_STATUSES:
        allowed = ", ".join(sorted(REVIEW_STATUSES))
        raise BendInstanceLabelsValidationError(f"review_status must be one of: {allowed}")
    for timestamp_key in ("created_at", "updated_at"):
        if timestamp_key in payload and not isinstance(payload[timestamp_key], str):
            raise BendInstanceLabelsValidationError(f"{timestamp_key} must be a string")

    manual_bends = payload.get("manual_bends")
    if not isinstance(manual_bends, list):
        raise BendInstanceLabelsValidationError("manual_bends must be an array")
    manual_ids: set[str] = set()
    for index, bend in enumerate(manual_bends):
        path = f"manual_bends[{index}]"
        if not isinstance(bend, dict):
            raise BendInstanceLabelsValidationError(f"{path} must be an object")
        _reject_unexpected_keys(bend, MANUAL_BEND_KEYS, path)
        bend_id = _require_nonempty_id(bend.get("bend_id"), f"{path}.bend_id")
        if bend_id in manual_ids:
            raise BendInstanceLabelsValidationError(f"duplicate manual bend_id: {bend_id}")
        manual_ids.add(bend_id)
        if bend.get("semantic") != "conventional_bend":
            raise BendInstanceLabelsValidationError(f"{path}.semantic must be conventional_bend")
        if bend.get("source") != "human_two_point":
            raise BendInstanceLabelsValidationError(f"{path}.source must be human_two_point")
        start = _require_point(bend.get("line_start"), f"{path}.line_start")
        end = _require_point(bend.get("line_end"), f"{path}.line_end")
        if start == end:
            raise BendInstanceLabelsValidationError(f"{path} must define a nonzero line segment")

    marker_reviews = payload.get("marker_reviews")
    if not isinstance(marker_reviews, list):
        raise BendInstanceLabelsValidationError("marker_reviews must be an array")
    review_ids: set[str] = set()
    for index, review in enumerate(marker_reviews):
        path = f"marker_reviews[{index}]"
        if not isinstance(review, dict):
            raise BendInstanceLabelsValidationError(f"{path} must be an object")
        _reject_unexpected_keys(review, MARKER_REVIEW_KEYS, path)
        bend_id = _require_nonempty_id(review.get("bend_id"), f"{path}.bend_id")
        if bend_id in review_ids:
            raise BendInstanceLabelsValidationError(f"duplicate marker review bend_id: {bend_id}")
        review_ids.add(bend_id)
        decision = review.get("decision")
        if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
            allowed = ", ".join(sorted(REVIEW_DECISIONS))
            raise BendInstanceLabelsValidationError(f"{path}.decision must be one of: {allowed}")
        if "reason" in review and review["reason"] is not None and not isinstance(review["reason"], str):
            raise BendInstanceLabelsValidationError(f"{path}.reason must be a string or null")

    # Verify that no non-standard value slipped through which Python's encoder
    # would serialize as non-JSON NaN/Infinity. Return a detached JSON value so
    # callers cannot mutate the object while it is being persisted.
    try:
        encoded = json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BendInstanceLabelsValidationError(f"payload is not finite JSON: {exc}") from exc
    return json.loads(encoded)


def load_bend_instance_labels(output_dir: Path, expected_job_id: str) -> Optional[Dict[str, Any]]:
    """Load and validate saved labels, or return ``None`` when none are saved."""
    path = Path(output_dir) / LABELS_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BendInstanceLabelsValidationError(f"could not read saved bend labels: {exc}") from exc
    return validate_bend_instance_labels(payload, expected_job_id)


def save_bend_instance_labels(output_dir: Path, payload: Any, expected_job_id: str) -> Dict[str, Any]:
    """Validate and atomically replace a job's bend-instance labels file."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"bend inspection output directory does not exist: {directory}")
    validated = validate_bend_instance_labels(payload, expected_job_id)
    serialized = json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{LABELS_FILENAME}.",
        suffix=".tmp",
        dir=directory,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, directory / LABELS_FILENAME)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return validated
