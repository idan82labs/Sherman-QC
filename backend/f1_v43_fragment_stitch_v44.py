"""Compose retained V43 rim guarding with retained V27 fragment stitching."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments


PROFILE = "F1_V44_V43_RETAINED_FRAGMENT_STITCH_V1"
COMPLEX_ROUTE = "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair"


def _prediction_row(prediction: dict[str, Any]) -> dict[str, Any]:
    row = deepcopy(prediction)
    row["candidate_id"] = str(
        prediction.get("source_hypothesis_id") or prediction.get("bend_id") or ""
    )
    row["span_endpoints"] = [prediction["line_start"], prediction["line_end"]]
    return row


def _stable_bend_id(candidate_id: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]
    return f"TRV44-{digest}"


def _serialize(row: dict[str, Any], route: str) -> dict[str, Any]:
    if str(row.get("candidate_source") or "") != "collinear_fragment_stitch":
        prediction = deepcopy(row)
        prediction.pop("candidate_id", None)
        prediction.pop("span_endpoints", None)
        return prediction
    endpoints = row["span_endpoints"]
    candidate_id = str(row["candidate_id"])
    return {
        "bend_id": _stable_bend_id(candidate_id),
        "line_start": endpoints[0],
        "line_end": endpoints[1],
        "selection_route": route,
        "selector_score": row.get("selector_score"),
        "source_hypothesis_id": candidate_id,
        "source_kind": "collinear_fragment_stitch",
        "stitched_member_candidate_ids": list(row["stitched_member_candidate_ids"]),
    }


def compose_v44(v43_result: dict[str, Any]) -> dict[str, Any]:
    """Apply V27 only after V43 and only on the bounded-complex route."""

    result = deepcopy(v43_result)
    route = str(result.get("route") or "")
    rows = [_prediction_row(value) for value in result.get("predictions", [])]
    stitch = {"selected": rows, "changes": []}
    if route == COMPLEX_ROUTE:
        stitch = stitch_collinear_selected_fragments(rows)
    predictions = [_serialize(value, route) for value in stitch["selected"]]
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v44_v43_fragment_stitch_result",
            "profile": PROFILE,
            "v43_retained": True,
            "v27_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "collinear_fragment_stitch_changes": stitch["changes"],
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    canonical = json.dumps(
        {"profile": PROFILE, "predictions": predictions, "changes": stitch["changes"]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    result["v44_result_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result
