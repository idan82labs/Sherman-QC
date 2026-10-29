"""V60: per-view token segments followed by calibrated cross-view consensus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import backend.f1_calibrated_multiview_turn_consensus_v58 as v58


PROFILE = "F1_V60_CALIBRATED_PER_VIEW_SEGMENT_CONSENSUS_V1"


def _fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = np.mean(points, axis=0)
    covariance = (points - centre).T @ (points - centre)
    values, vectors = np.linalg.eigh(covariance)
    return centre, v58._canonical_direction(vectors[:, int(np.argmax(values))])


def _per_view_segments(tokens: list[v58.Token], B: float) -> list[dict[str, Any]]:
    remaining = list(tokens)
    segments = []
    while remaining:
        candidates = []
        for seed_index, seed in enumerate(remaining):
            support = [
                index for index, token in enumerate(remaining)
                if v58._angle_deg(seed.tangent, token.tangent) <= 10.0
                and v58._line_distance(token.point, seed.point, seed.tangent) <= 0.008 * B
            ]
            if len(support) < 3:
                continue
            points = np.asarray([remaining[index].point for index in support])
            axial = (points - seed.point) @ seed.tangent
            span = float(np.ptp(axial))
            if span < 0.05 * B:
                continue
            score = sum(remaining[index].score for index in support)
            candidates.append(((-len(support), -span, -score, seed_index), support))
        if not candidates:
            break
        _, support = min(candidates, key=lambda row: row[0])
        chosen = [remaining[index] for index in support]
        points = np.asarray([token.point for token in chosen])
        centre, direction = _fit(points)
        axial = (points - centre) @ direction
        segments.append(
            {
                "view": chosen[0].view,
                "point": centre,
                "direction": direction,
                "start": centre + float(np.min(axial)) * direction,
                "end": centre + float(np.max(axial)) * direction,
                "tokens": chosen,
            }
        )
        removed = set(support)
        remaining = [token for index, token in enumerate(remaining) if index not in removed]
    return segments


def _segment_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return 0.5 * (
        v58._line_distance(left["point"], right["point"], right["direction"])
        + v58._line_distance(right["point"], left["point"], left["direction"])
    )


def _cross_view_lines(segments: list[dict[str, Any]], B: float) -> list[dict[str, Any]]:
    remaining = list(segments)
    lines = []
    while remaining:
        candidates = []
        for seed_index, seed in enumerate(remaining):
            support = [
                index for index, segment in enumerate(remaining)
                if v58._angle_deg(seed["direction"], segment["direction"]) <= 8.0
                and _segment_distance(seed, segment) <= 0.010 * B
            ]
            views = {int(remaining[index]["view"]) for index in support}
            if len(views) < 3:
                continue
            token_count = sum(len(remaining[index]["tokens"]) for index in support)
            candidates.append(((-len(views), -token_count, seed_index), support))
        if not candidates:
            break
        _, support = min(candidates, key=lambda row: row[0])
        chosen_segments = [remaining[index] for index in support]
        tokens = [token for segment in chosen_segments for token in segment["tokens"]]
        points = np.asarray([token.point for token in tokens])
        centre, direction = _fit(points)
        residuals = np.asarray([v58._line_distance(point, centre, direction) for point in points])
        direction_errors = np.asarray([v58._angle_deg(token.tangent, direction) for token in tokens])
        axial = (points - centre) @ direction
        span = float(np.ptp(axial))
        views = sorted({int(segment["view"]) for segment in chosen_segments})
        if span >= 0.10 * B and v58._nearest_rank(residuals.tolist(), 0.95) <= 0.008 * B and float(np.median(direction_errors)) <= 8.0:
            lines.append(
                {
                    "line_start": (centre + float(np.min(axial)) * direction).tolist(),
                    "line_end": (centre + float(np.max(axial)) * direction).tolist(),
                    "supporting_view_count": len(views),
                    "per_view_segment_count": len(chosen_segments),
                    "token_count": len(tokens),
                    "median_turn_deg": float(np.median([token.alpha for token in tokens])),
                    "p95_residual_over_B": v58._nearest_rank(residuals.tolist(), 0.95) / B,
                }
            )
        removed = set(support)
        remaining = [segment for index, segment in enumerate(remaining) if index not in removed]
    return lines


def compose_v60(bundle: Path, v52_result: dict[str, Any]) -> dict[str, Any]:
    source, frames, source_sha, bundle_sha = v58.load_bundle(bundle)
    B = float(np.linalg.norm(np.ptp(source, axis=0)))
    stride = v58._stride(frames)
    view_segments = []
    token_count = 0
    for view, frame in enumerate(frames):
        raw, _, _ = v58._raw_tokens_for_view(frame, view, stride, B)
        thinned = v58._thin(raw, frame.depth.shape, B)
        token_count += len(thinned)
        view_segments.extend(_per_view_segments(thinned, B))
    raw_lines = _cross_view_lines(view_segments, B)
    incumbents = list(v52_result.get("predictions") or [])
    births = []
    for raw in raw_lines:
        if any(
            (lambda metrics: metrics[0] <= 4.0 and metrics[1] <= 0.008 * B and metrics[2] >= 0.60)(v58._line_metrics(raw, incumbent))
            for incumbent in incumbents
        ):
            continue
        payload = np.asarray(raw["line_start"] + raw["line_end"], dtype=">f8").tobytes()
        births.append(
            {
                "bend_id": "TRV60-" + hashlib.sha256(payload).hexdigest()[:20],
                "line_start": raw["line_start"],
                "line_end": raw["line_end"],
                "selection_route": "calibrated_per_view_segment_consensus_v60",
                "source_kind": "organized_calibrated_raw_range",
                **{key: raw[key] for key in ("supporting_view_count", "per_view_segment_count", "token_count", "median_turn_deg", "p95_residual_over_B")},
            }
        )
    predictions = list(incumbents) + births
    audit = {
        "pixel_stride": stride,
        "thinned_token_count": token_count,
        "per_view_segment_count": len(view_segments),
        "cross_view_line_count": len(raw_lines),
        "suppressed_v52_match_count": len(raw_lines) - len(births),
        "raw_new_count": len(births),
        "retained_v52_count": len(incumbents),
        "retained_v52_byte_preservation": True,
    }
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v60_calibrated_per_view_segment_consensus_result",
        "profile": PROFILE,
        "source_sha256": source_sha,
        "capability_sha256": bundle_sha,
        "v52_sha256": v52_result.get("v52_result_sha256"),
        "predictions": predictions,
        "audit": audit,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
    }
    semantic = json.dumps({"profile": PROFILE, "predictions": predictions, "audit": audit}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["v60_result_sha256"] = hashlib.sha256(semantic).hexdigest()
    return result
