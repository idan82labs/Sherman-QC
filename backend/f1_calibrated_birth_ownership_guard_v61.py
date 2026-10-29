"""V61: source-only ownership guard for calibrated multiview births.

This post-processor does not create, move, or refit lines.  It keeps the frozen
V52 output byte-for-byte and admits a V60 birth only when it is not a
near-coincident representation of any V52 incumbent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

import backend.f1_calibrated_multiview_turn_consensus_v58 as v58


PROFILE = "F1_V61_CALIBRATED_BIRTH_OWNERSHIP_GUARD_V1"
V60_ROUTE = "calibrated_per_view_segment_consensus_v60"
V61_ROUTE = "calibrated_birth_ownership_guard_v61"


def compose_v61(v52_result: dict[str, Any], v60_result: dict[str, Any], B: float) -> dict[str, Any]:
    incumbents = list(v52_result.get("predictions") or [])
    raw_births = [row for row in v60_result.get("predictions") or [] if row.get("selection_route") == V60_ROUTE]
    admitted = []
    suppressed = []
    for raw in raw_births:
        nearest = None
        for incumbent in incumbents:
            angle, distance, overlap = v58._line_metrics(raw, incumbent)
            metrics = {"angle_deg": angle, "distance_over_B": distance / B, "overlap": overlap, "incumbent_bend_id": incumbent["bend_id"]}
            if nearest is None or (metrics["distance_over_B"], metrics["angle_deg"], -metrics["overlap"], metrics["incumbent_bend_id"]) < (nearest["distance_over_B"], nearest["angle_deg"], -nearest["overlap"], nearest["incumbent_bend_id"]):
                nearest = metrics
        if nearest is not None and nearest["angle_deg"] <= 4.0 and nearest["distance_over_B"] <= 0.010 and nearest["overlap"] >= 0.60:
            suppressed.append({"v60_bend_id": raw["bend_id"], **nearest})
            continue
        payload = np.asarray(raw["line_start"] + raw["line_end"], dtype=">f8").tobytes()
        admitted.append({
            **raw,
            "bend_id": "TRV61-" + hashlib.sha256(payload).hexdigest()[:20],
            "selection_route": V61_ROUTE,
            "upstream_v60_bend_id": raw["bend_id"],
        })
    predictions = incumbents + admitted
    audit = {
        "retained_v52_count": len(incumbents),
        "retained_v52_byte_preservation": True,
        "v60_birth_count": len(raw_births),
        "ownership_suppressed_count": len(suppressed),
        "admitted_birth_count": len(admitted),
        "distance_over_B_max": 0.010,
        "angle_deg_max": 4.0,
        "overlap_min": 0.60,
        "suppressed": suppressed,
    }
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v61_calibrated_birth_ownership_guard_result",
        "profile": PROFILE,
        "source_sha256": v60_result.get("source_sha256"),
        "capability_sha256": v60_result.get("capability_sha256"),
        "v52_sha256": v52_result.get("v52_result_sha256"),
        "v60_sha256": v60_result.get("v60_result_sha256"),
        "predictions": predictions,
        "audit": audit,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "development_only_after_v1_truth_open": True,
    }
    semantic = json.dumps({"profile": PROFILE, "predictions": predictions, "audit": audit}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["v61_result_sha256"] = hashlib.sha256(semantic).hexdigest()
    return result
