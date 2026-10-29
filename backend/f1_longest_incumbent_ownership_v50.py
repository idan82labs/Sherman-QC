"""Correct V45 incumbent ownership by canonicalizing duplicate lines longest-first."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import numpy as np

from backend.f1_deterministic_paired_return_completion_v45 import (
    _canonical,
    compose_v45,
    prediction_row,
)
from backend.f1_elongated_lane_consolidation_v46 import _row_key
from research.f1_multifamily_selector_v2.model import endpoints


PROFILE = "F1_V50_LONGEST_INCUMBENT_OWNERSHIP_V1"


def _length(prediction: dict[str, Any]) -> float:
    line = endpoints(prediction_row(prediction))
    return float(np.linalg.norm(line[1] - line[0]))


def compose_v50(
    v44_result: dict[str, Any], line_objects: list[dict[str, Any]], bbox_span: float
) -> dict[str, Any]:
    """Run V45 semantics after deterministic longest-first incumbent ordering."""

    source = deepcopy(v44_result)
    source["predictions"] = sorted(
        source.get("predictions", []),
        key=lambda value: (-_length(value), _row_key(prediction_row(value))),
    )
    result = compose_v45(source, line_objects, bbox_span)
    canonical_ids = [
        str(value.get("source_hypothesis_id") or value.get("bend_id") or "")
        for value in source.get("predictions", [])
    ]
    result.update(
        {
            "kind": "sherman_qc_f1_v50_longest_incumbent_ownership_result",
            "profile": PROFILE,
            "v44_retained": True,
            "v45_semantics_retained": True,
            "longest_first_incumbent_ownership": True,
            "canonical_incumbent_order": canonical_ids,
        }
    )
    payload = {
        "profile": PROFILE,
        "predictions": result["predictions"],
        "paired_return_completion_changes": result[
            "paired_return_completion_changes"
        ],
        "post_completion_nms_suppressed_candidate_ids": result[
            "post_completion_nms_suppressed_candidate_ids"
        ],
        "canonical_incumbent_order": canonical_ids,
    }
    result["v50_result_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return result
