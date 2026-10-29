#!/usr/bin/env python3
"""Unified V16 route plus the authorized native-full-mesh V17 rescue.

V16 remains frozen route ownership.  This runner delegates the complete V16
selection first and invokes V17 only when V16 chose the native full-resolution
V11 lane.  If no rescue survives, the emitted JSON is byte-identical to the
frozen V16 runner output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_f1_full_mesh_adjacency_hysteresis_v17 import (
    FROZEN_V11_DUPLICATE_PREDICATE_SHA256,
    PROFILE as V17_PROFILE,
    assert_frozen_duplicate_predicate,
    rescue_rejected_adjacency,
)
from scripts.select_f1_mesh_route_v16 import ROUTE_FULL, build_result, select_unified


PROFILE = "F1_SOURCE_ONLY_UNIFIED_MESH_ROUTE_V17"
RESCUE_ROUTE = "full_resolution_mesh_v11_axial_hysteresis_rescue_v17"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_unified_v17(pool: dict, v9: dict, v4: dict) -> dict:
    """Run frozen V16, then append V17 rescues only on its full-mesh lane."""

    selection = select_unified(pool, v9, v4)
    selection["mesh_adjacency_axial_hysteresis_rescues"] = []
    if selection["route"] != ROUTE_FULL:
        return selection

    assert_frozen_duplicate_predicate()
    rows = selection["rows"]
    row_index = {
        str(row.get("candidate_id") or ""): index for index, row in enumerate(rows)
    }
    incumbent_indices = [
        row_index[str(row.get("candidate_id") or "")]
        for row in selection["prediction_rows"]
    ]
    retained, rescues = rescue_rejected_adjacency(rows, incumbent_indices)
    selection["prediction_rows"] = [dict(rows[index]) for index in retained]
    selection["mesh_adjacency_axial_hysteresis_rescues"] = rescues
    return selection


def build_result_v17(pool: dict, selection: dict) -> dict:
    """Preserve exact V16 bytes for a no-op; identify activated V17 outputs."""

    result = build_result(pool, selection)
    rescues = selection["mesh_adjacency_axial_hysteresis_rescues"]
    if not rescues:
        return result

    rescue_ids = {str(item["candidate_id"]) for item in rescues}
    for prediction in result["predictions"]:
        if str(prediction["source_hypothesis_id"]) in rescue_ids:
            prediction["selection_route"] = RESCUE_ROUTE
    result["kind"] = "sherman_qc_f1_source_only_unified_mesh_route_v17_selection"
    result["profile"] = PROFILE
    result["intervention_profile"] = V17_PROFILE
    result["mesh_adjacency_axial_hysteresis_rescues"] = rescues
    result["v11_duplicate_predicate_sha256"] = FROZEN_V11_DUPLICATE_PREDICATE_SHA256
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    selection = select_unified_v17(pool, joblib.load(args.v9_model), joblib.load(args.v4_model))
    result = build_result_v17(pool, selection)
    result["candidate_pool_sha256"] = _sha(args.input)
    result["v9_model_sha256"] = _sha(args.v9_model)
    result["v4_model_sha256"] = _sha(args.v4_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "rescue_count": len(selection["mesh_adjacency_axial_hysteresis_rescues"]),
                "route": result["route"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
