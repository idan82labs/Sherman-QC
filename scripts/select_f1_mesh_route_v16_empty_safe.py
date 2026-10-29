#!/usr/bin/env python3
"""V16 adapter that terminalizes an empty source-only pool deterministically.

Non-empty pools execute the retained V16 implementation unchanged. An empty
pool is a valid detector outcome, not a classifier input, so it emits the same
V16 result schema with zero predictions and no recovery activity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.select_f1_mesh_route_v16 as retained
from research.f1_multifamily_selector_v2.model import FEATURE_NAMES
from scripts.train_f1_bounded_mesh_router_v9 import PROFILE as V9_PROFILE
from scripts.train_f1_topology_recall_v4 import PROFILE as V4_PROFILE


def _validate_bundle(bundle: dict, profile: str, label: str) -> None:
    if bundle.get("profile") != profile or tuple(bundle.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError(f"{label} model bundle does not match the frozen feature schema")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    v9 = joblib.load(args.v9_model)
    v4 = joblib.load(args.v4_model)
    rows = list(pool.get("line_objects") or [])
    if rows:
        selection = retained.select_unified(pool, v9, v4)
        empty_pool_terminalized = False
    else:
        _validate_bundle(v9, V9_PROFILE, "V9")
        _validate_bundle(v4, V4_PROFILE, "V4")
        selection = {
            "rows": [],
            "prediction_rows": [],
            "v9_scores": np.zeros(0, dtype=float),
            "bbox_span": max((float(value) for value in pool.get("scan_bbox_mm") or (1.0,))),
            "route": retained.classify_route(pool, 0),
            "no_output_retry_used": False,
            "representative_repairs": [],
            "short_fragment_prunes": [],
            "seed_family_coverage_recoveries": [],
            "paired_plane_fold_recoveries": [],
            "tilted_adjacency_repairs": [],
        }
        empty_pool_terminalized = True

    result = retained.build_result(pool, selection)
    result["candidate_pool_sha256"] = retained._sha(args.input)
    result["v9_model_sha256"] = retained._sha(args.v9_model)
    result["v4_model_sha256"] = retained._sha(args.v4_model)
    result["empty_pool_terminalized"] = empty_pool_terminalized
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": result["candidate_count"], "empty_pool_terminalized": empty_pool_terminalized, "route": result["route"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
