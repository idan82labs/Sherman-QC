#!/usr/bin/env python3
"""Apply the research-only V4 precision core plus adjacency recall route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, build_feature_matrix, endpoints
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE, select


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    bundle = joblib.load(args.model)
    if bundle.get("profile") != PROFILE or tuple(bundle.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")
    core_scores = bundle["core_estimator"].predict_proba(features)[:, 1]
    adjacency_scores = np.zeros(len(rows), dtype=float)
    adjacency_indices = [index for index, row in enumerate(rows) if row.get("representative_source") == ADJACENCY_SOURCE]
    if adjacency_indices:
        adjacency_scores[adjacency_indices] = bundle["adjacency_estimator"].predict_proba(features[adjacency_indices])[:, 1]
    selected = select(rows, core_scores, adjacency_scores)
    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        route = "adjacency_recall" if row.get("representative_source") == ADJACENCY_SOURCE and adjacency_scores[index] >= bundle["adjacency_threshold"] else "v3_precision_core"
        predictions.append({
            "bend_id": f"TRV4-{digest}",
            "line_start": line[0].tolist(),
            "line_end": line[1].tolist(),
            "source_hypothesis_id": source_id,
            "source_kind": row.get("representative_source"),
            "selection_route": route,
            "selector_score": round(float(adjacency_scores[index] if route == "adjacency_recall" else core_scores[index]), 12),
        })
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_topology_adjacency_recall_selection",
        "profile": PROFILE,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "candidate_pool_sha256": sha(args.input),
        "model_sha256": sha(args.model),
        "research_only": True,
        "new_unseen_transfer_evidence": False,
        "c006_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
