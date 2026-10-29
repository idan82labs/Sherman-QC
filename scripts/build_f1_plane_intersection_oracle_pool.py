#!/usr/bin/env python3
"""Build source-only flange-plane intersection candidates for Family-A adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _canonical_direction(value: np.ndarray) -> np.ndarray:
    direction = value / np.linalg.norm(value)
    first = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(first) and direction[first[0]] < 0:
        direction *= -1.0
    return direction


def build(payload: dict) -> dict:
    atoms = {str(row["atom_id"]): row for row in payload["atom_graph"]["atoms"]}
    flanges = sorted(payload.get("flange_hypotheses") or [], key=lambda row: str(row["flange_id"]))
    rows = []
    for left_index, left in enumerate(flanges):
        for right in flanges[left_index + 1 :]:
            n1 = np.asarray(left["normal"], dtype=np.float64)
            n2 = np.asarray(right["normal"], dtype=np.float64)
            n1 /= np.linalg.norm(n1)
            n2 /= np.linalg.norm(n2)
            direction_raw = np.cross(n1, n2)
            sine = float(np.linalg.norm(direction_raw))
            if sine <= math.sin(math.radians(10.0)):
                continue
            direction = _canonical_direction(direction_raw)
            centroids = np.asarray(
                [
                    atoms[atom_id]["centroid"]
                    for atom_id in sorted(set((left.get("atom_ids") or []) + (right.get("atom_ids") or [])))
                    if atom_id in atoms
                ],
                dtype=np.float64,
            )
            if len(centroids) < 4:
                continue
            mean = np.mean(centroids, axis=0)
            matrix = np.vstack((n1, n2, direction))
            rhs = np.asarray((-float(left["d"]), -float(right["d"]), float(np.dot(direction, mean))))
            try:
                point = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                continue
            axial = (centroids - point) @ direction
            clips = (
                ("q05_q95", float(np.quantile(axial, 0.05)), float(np.quantile(axial, 0.95))),
                ("q00_q100", float(np.min(axial)), float(np.max(axial))),
            )
            angle = math.degrees(math.acos(float(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))))
            pair = sorted((str(left["flange_id"]), str(right["flange_id"])))
            for clip_name, low, high in clips:
                if high - low <= 1e-6:
                    continue
                key = hashlib.sha256(
                    b"F1_PLANE_INTERSECTION_ORACLE_POOL_V1\0"
                    + "\0".join(pair + [clip_name]).encode("ascii")
                ).hexdigest()[:20]
                rows.append(
                    {
                        "bend_id": f"PI-{pair[0]}-{pair[1]}-{clip_name}-{key}",
                        "line_start": (point + low * direction).tolist(),
                        "line_end": (point + high * direction).tolist(),
                        "source_kind": "flange_plane_intersection",
                        "incident_flange_ids": pair,
                        "plane_angle_degrees": angle,
                        "clip_kind": clip_name,
                        "support_atom_count": len(centroids),
                        "span_mm": high - low,
                        "minimum_flange_confidence": min(float(left.get("confidence") or 0.0), float(right.get("confidence") or 0.0)),
                        "identity_or_filename_features_used": False,
                    }
                )
    rows.sort(key=lambda row: row["bend_id"])
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_plane_intersection_oracle_candidate_pool",
        "predictions": rows,
        "candidate_count": len(rows),
        "gold_or_oracle_feedback_used": False,
        "identity_or_filename_features_used": False,
        "c006_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": result["candidate_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
