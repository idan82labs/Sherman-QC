#!/usr/bin/env python3
"""Materialize a compact source-only submission from a frozen candidate payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _canonical_direction(value: np.ndarray) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("hypothesis has invalid axis direction")
    direction = direction / norm
    first = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(first) and direction[first[0]] < 0:
        direction *= -1.0
    return direction


def _candidate_endpoints(candidate: dict) -> tuple[np.ndarray, np.ndarray, float]:
    raw_endpoints = candidate.get("span_endpoints")
    if isinstance(raw_endpoints, list) and len(raw_endpoints) >= 2:
        endpoints = np.asarray(raw_endpoints[:2], dtype=np.float64)
        if endpoints.shape != (2, 3) or not np.all(np.isfinite(endpoints)):
            raise ValueError("candidate has invalid span_endpoints")
        span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if span <= 0:
            raise ValueError("candidate has zero-length span_endpoints")
        return endpoints[0], endpoints[1], span

    anchor = np.asarray(candidate["anchor"], dtype=np.float64)
    direction = _canonical_direction(np.asarray(candidate["axis_direction"], dtype=np.float64))
    span = float(candidate.get("span_mm") or 0.0)
    if not np.all(np.isfinite(anchor)) or not np.isfinite(span) or span <= 0:
        raise ValueError("candidate has invalid anchor/span")
    return anchor - 0.5 * span * direction, anchor + 0.5 * span * direction, span


def build(payload: dict, *, collection: str | None = None) -> dict:
    if collection is not None:
        source_rows = payload.get(collection)
        if not isinstance(source_rows, list):
            raise ValueError(f"payload has no list collection named {collection!r}")
        source_collection = collection
    else:
        source_rows = payload.get("bend_hypotheses")
        source_collection = "bend_hypotheses"
        if source_rows is None:
            source_rows = payload.get("candidates")
            source_collection = "candidates"
    rows = []
    for index, hypothesis in enumerate(source_rows or []):
        line_start, line_end, span = _candidate_endpoints(hypothesis)
        source_id = str(
            hypothesis.get("candidate_id")
            or hypothesis.get("bend_id")
            or f"hypothesis-{index:04d}"
        )
        candidate_key = hashlib.sha256(
            b"F1_SEMANTIC_ORACLE_CANDIDATE_V1\0"
            + np.ascontiguousarray(np.concatenate((line_start, line_end)), dtype="<f8").tobytes()
        ).hexdigest()[:20]
        rows.append(
            {
                "bend_id": f"HYP-{index + 1:03d}-{candidate_key}",
                "line_start": line_start.tolist(),
                "line_end": line_end.tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": hypothesis.get("candidate_source") or hypothesis.get("source_kind"),
                "bend_class": hypothesis.get("bend_class"),
                "visibility_state": hypothesis.get("visibility_state"),
                "confidence": hypothesis.get("confidence") or hypothesis.get("debug_confidence"),
                "angle_deg": hypothesis.get("angle_deg"),
                "radius_mm": hypothesis.get("radius_mm"),
                "span_mm": span,
                "cross_width_mm": hypothesis.get("cross_width_mm"),
                "incident_flange_ids": hypothesis.get("incident_flange_ids") or [],
                "canonical_flange_pair": hypothesis.get("canonical_flange_pair"),
            }
        )
    rows.sort(key=lambda row: row["bend_id"])
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_source_only_semantic_oracle_candidate_pool",
        "source_collection": source_collection,
        "predictions": rows,
        "candidate_count": len(rows),
        "identity_or_filename_features_used": False,
        "gold_or_oracle_feedback_used": False,
        "c006_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = build(payload, collection=args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": result["candidate_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
