"""V62: complementary cumulative-turn consensus for rounded bend facets.

V62 deliberately leaves the retained V61 predictions byte-for-byte intact.
It targets only transitions whose near bank rotates modestly while the wider
bank accumulates a larger turn, then requires four-view line consensus.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import backend.f1_calibrated_multiview_turn_consensus_v58 as v58
import backend.f1_calibrated_per_view_segment_consensus_v60 as v60


PROFILE = "F1_V62_CALIBRATED_DISTRIBUTED_TURN_CONSENSUS_V1"
ROUTE = "calibrated_distributed_turn_consensus_v62"


def _distributed_tokens(frame: v58.Frame, view: int, stride: int, B: float) -> list[v58.Token]:
    normals, usable = v58._normal_field(frame, 1, B)
    raw: list[v58.Token] = []
    height, width = frame.depth.shape
    for direction_ordinal, (dc, dr) in enumerate(v58.DIRECTIONS):
        for row in range(0, height, stride):
            rr0, rr1 = row - dr, row + dr
            if rr0 < 0 or rr1 >= height:
                continue
            for col in range(0, width, stride):
                cc0, cc1 = col - dc, col + dc
                if cc0 < 0 or cc1 >= width or not (usable[row, col] and usable[rr0, cc0] and usable[rr1, cc1]):
                    continue
                if v58._angle_deg(normals[rr0, cc0], normals[rr1, cc1]) < 3.0:
                    continue
                left_s, left_l, left_connected = v58._side_samples(frame, normals, usable, row, col, dr, dc, 1, -1, B)
                right_s, right_l, right_connected = v58._side_samples(frame, normals, usable, row, col, dr, dc, 1, 1, B)
                if not (left_connected and right_connected and len(left_s) >= 4 and len(right_s) >= 4 and len(left_l) >= 6 and len(right_l) >= 6):
                    continue
                Ls, Ll, Rs, Rl = (v58._representative(values) for values in (left_s, left_l, right_s, right_l))
                if any(value is None for value in (Ls, Ll, Rs, Rl)):
                    continue
                assert Ls is not None and Ll is not None and Rs is not None and Rl is not None
                alpha_s = v58._angle_deg(Ls[0], Rs[0])
                alpha_l = v58._angle_deg(Ll[0], Rl[0])
                concentration = alpha_s / max(alpha_l, 1.0)
                if not (
                    5.0 <= alpha_s < 15.0
                    and 16.0 <= alpha_l <= 80.0
                    and 0.18 <= concentration <= 0.75
                    and alpha_l - alpha_s >= 4.0
                ):
                    continue
                cross = np.cross(Ll[0], Rl[0])
                tangent = v58._unit(cross)
                if tangent is None or float(np.linalg.norm(cross)) < math.sin(math.radians(16.0)):
                    continue
                tangent = v58._canonical_direction(tangent)
                roughness = Ll[1] + Rl[1]
                score = alpha_l * (1.0 - 0.5 * concentration) / (1.0 + roughness / 8.0)
                raw.append(
                    v58.Token(
                        frame.points[row, col].copy(), tangent, score, alpha_l,
                        concentration, roughness, view, row, col, direction_ordinal,
                    )
                )
    return raw


def compose_v62(bundle: Path, v61_result: dict[str, Any]) -> dict[str, Any]:
    source, frames, source_sha, bundle_sha = v58.load_bundle(bundle)
    B = float(np.linalg.norm(np.ptp(source, axis=0)))
    stride = v58._stride(frames)
    view_segments: list[dict[str, Any]] = []
    token_count = 0
    for view, frame in enumerate(frames):
        raw = _distributed_tokens(frame, view, stride, B)
        thinned = v58._thin(raw, frame.depth.shape, B)
        token_count += len(thinned)
        view_segments.extend(v60._per_view_segments(thinned, B))
    consensus = [
        row for row in v60._cross_view_lines(view_segments, B)
        if int(row["supporting_view_count"]) >= 4 and 16.0 <= float(row["median_turn_deg"]) <= 80.0
    ]
    incumbents = list(v61_result.get("predictions") or [])
    births = []
    for raw in consensus:
        if any(
            (lambda metrics: metrics[0] <= 4.0 and metrics[1] <= 0.008 * B and metrics[2] >= 0.60)(v58._line_metrics(raw, incumbent))
            for incumbent in incumbents
        ):
            continue
        payload = np.asarray(raw["line_start"] + raw["line_end"], dtype=">f8").tobytes()
        births.append(
            {
                "bend_id": "TRV62-" + hashlib.sha256(payload).hexdigest()[:20],
                "line_start": raw["line_start"],
                "line_end": raw["line_end"],
                "selection_route": ROUTE,
                "source_kind": "organized_calibrated_raw_range",
                **{key: raw[key] for key in ("supporting_view_count", "per_view_segment_count", "token_count", "median_turn_deg", "p95_residual_over_B")},
            }
        )
    predictions = incumbents + births
    audit = {
        "pixel_stride": stride,
        "distributed_token_count": token_count,
        "per_view_segment_count": len(view_segments),
        "four_view_consensus_count": len(consensus),
        "suppressed_incumbent_match_count": len(consensus) - len(births),
        "raw_new_count": len(births),
        "retained_v61_count": len(incumbents),
        "retained_v61_byte_preservation": True,
    }
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v62_calibrated_distributed_turn_consensus_result",
        "profile": PROFILE,
        "source_sha256": source_sha,
        "capability_sha256": bundle_sha,
        "v61_sha256": v61_result.get("v61_result_sha256"),
        "predictions": predictions,
        "audit": audit,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "development_only_after_v3_truth_open": True,
    }
    semantic = json.dumps({"profile": PROFILE, "predictions": predictions, "audit": audit}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["v62_result_sha256"] = hashlib.sha256(semantic).hexdigest()
    return result
