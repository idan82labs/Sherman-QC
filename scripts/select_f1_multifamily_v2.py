#!/usr/bin/env python3
"""Apply the frozen selector V2 to one source-only candidate pool."""

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

from research.f1_multifamily_selector_v2.model import (
    FEATURE_NAMES,
    PROFILE,
    build_feature_matrix,
    endpoints,
    select_indices,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool.get("line_objects") or []]
    bundle = joblib.load(args.model)
    if bundle.get("profile") != PROFILE or tuple(bundle.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("selector model bundle does not match the frozen feature schema")
    scores = bundle["estimator"].predict_proba(build_feature_matrix(rows))[:, 1]
    selected = select_indices(rows, scores)
    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row.get("candidate_id"))
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode("utf-8")).hexdigest()[:20]
        predictions.append(
            {
                "bend_id": f"MFV2-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selector_score": round(float(scores[index]), 12),
                "span_mm": row.get("span_mm"),
                "radius_mm": row.get("radius_mm"),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_multifamily_source_only_selection",
        "profile": PROFILE,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "candidate_pool_sha256": _sha(args.input),
        "model_sha256": _sha(args.model),
        "inference_contract": {
            "part_identifier_exposed": False,
            "expected_count_exposed": False,
            "manual_truth_exposed": False,
            "cad_exposed": False,
            "current_case_oracle_feedback_exposed": False,
            "filename_features_exposed": False,
        },
        "source_point_serialization": False,
        "c006_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": len(predictions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
