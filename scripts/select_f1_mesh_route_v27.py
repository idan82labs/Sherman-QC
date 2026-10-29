#!/usr/bin/env python3
"""V16 plus count-free collinear selected-fragment stitching."""

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

from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments
from scripts.select_f1_mesh_route_v16 import ROUTE_COMPLEX, build_result, select_unified


PROFILE = "F1_SOURCE_ONLY_UNIFIED_MESH_ROUTE_V27_COLLINEAR_FRAGMENT_STITCH"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    selection = select_unified(
        pool, joblib.load(args.v9_model), joblib.load(args.v4_model)
    )
    stitch = {"selected": selection["prediction_rows"], "changes": []}
    if selection["route"] == ROUTE_COMPLEX:
        stitch = stitch_collinear_selected_fragments(selection["prediction_rows"])
    selection["prediction_rows"] = stitch["selected"]
    result = build_result(pool, selection)
    result["profile"] = PROFILE
    result["collinear_fragment_stitch_changes"] = stitch["changes"]
    result["v9_model_sha256"] = _sha(args.v9_model)
    result["v4_model_sha256"] = _sha(args.v4_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "prediction_count": len(result["predictions"]),
                "route": result["route"],
                "stitched_component_count": len(stitch["changes"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
