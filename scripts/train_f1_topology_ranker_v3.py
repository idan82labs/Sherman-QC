#!/usr/bin/env python3
"""Train selector V3 on one canonical representative per development object."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, build_feature_matrix, select_indices_configurable


PROFILE = "F1_CANONICAL_TOPOLOGY_RANKER_V3"
CASES_V2 = ("C008", "C031", "C028", "C022", "C025", "C035", "C021", "C026")
CASES_V3 = ("C018", "C027", "C005", "C033")
ALL_CASES = (*CASES_V2, *CASES_V3)
THRESHOLD = 0.50
NMS_ANGLE = 12.0
NMS_DISTANCE = 10.0
NMS_OVERLAP = 0.30
RANDOM_STATE = 20260811


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(pool_dir: Path, run_dir: Path, case: str) -> dict[str, Any]:
    pool_path = pool_dir / f"{case}.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    submission = json.loads((run_dir / f"{case}_line_objects_submission.json").read_text(encoding="utf-8"))
    feedback = json.loads((run_dir / f"{case}_line_objects_feedback.json").read_text(encoding="utf-8"))
    prediction_by_source = {str(row["source_hypothesis_id"]): row for row in submission["predictions"]}
    feedback_by_id = {str(row["candidate_id"]): row for row in feedback["candidate_feedback"]}
    rows = [dict(row) for row in pool["line_objects"]]
    object_ids = [
        feedback_by_id[str(prediction_by_source[str(row["candidate_id"])]["bend_id"])]["opaque_object_id"]
        for row in rows
    ]
    by_object: dict[str, list[int]] = collections.defaultdict(list)
    for index, object_id in enumerate(object_ids):
        if object_id:
            by_object[str(object_id)].append(index)
    target = np.zeros(len(rows), dtype=np.uint8)
    for indices in by_object.values():
        representative = max(
            indices,
            key=lambda index: (
                float(rows[index].get("case_percentile_two_sided_transition_score") or 0.0),
                float(rows[index].get("case_percentile_score") or 0.0),
                float(rows[index].get("finite_support_fraction") or 0.0),
                float(rows[index].get("span_mm") or 0.0),
                str(rows[index].get("candidate_id") or ""),
            ),
        )
        target[representative] = 1
    return {
        "rows": rows,
        "features": build_feature_matrix(rows),
        "target": target,
        "object_ids": object_ids,
        "expected": len(by_object) + int(feedback["metrics"]["false_negative"]),
        "pool_sha256": sha(pool_path),
    }


def model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def metrics(items: list[tuple[list[int], list[str | None], int]]) -> dict[str, Any]:
    tp = fp = fn = 0
    for selected, object_ids, expected in items:
        seen: set[str] = set()
        for index in selected:
            object_id = object_ids[index]
            if object_id and object_id not in seen:
                seen.add(object_id)
            else:
                fp += 1
        tp += len(seen)
        fn += expected - len(seen)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def select(rows: list[dict[str, Any]], scores: np.ndarray) -> list[int]:
    return select_indices_configurable(rows, scores, threshold=THRESHOLD, max_angle=NMS_ANGLE, max_distance=NMS_DISTANCE, min_overlap=NMS_OVERLAP)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--v2-development-dir", type=Path, required=True)
    parser.add_argument("--v3-development-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = {
        case: load_case(args.pool_dir, args.v2_development_dir if case in CASES_V2 else args.v3_development_dir, case)
        for case in ALL_CASES
    }
    held_out = []
    by_case = {}
    for case in ALL_CASES:
        train = [value for value in ALL_CASES if value != case]
        estimator = model().fit(np.vstack([data[value]["features"] for value in train]), np.concatenate([data[value]["target"] for value in train]))
        scores = estimator.predict_proba(data[case]["features"])[:, 1]
        selected = select(data[case]["rows"], scores)
        item = (selected, data[case]["object_ids"], data[case]["expected"])
        held_out.append(item)
        by_case[case] = {**metrics([item]), "selected_count": len(selected), "candidate_pool_sha256": data[case]["pool_sha256"]}
    estimator = model().fit(np.vstack([data[value]["features"] for value in ALL_CASES]), np.concatenate([data[value]["target"] for value in ALL_CASES]))
    bundle = {
        "profile": PROFILE,
        "feature_names": FEATURE_NAMES,
        "threshold": THRESHOLD,
        "nms_angle": NMS_ANGLE,
        "nms_distance": NMS_DISTANCE,
        "nms_overlap": NMS_OVERLAP,
        "development_cases": ALL_CASES,
        "lock_sha256": sha(args.lock),
        "estimator": estimator,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "validation": "leave_one_family_out",
        "development_cases": list(ALL_CASES),
        "held_out_micro_metrics": metrics(held_out),
        "held_out_by_case": by_case,
        "model_sha256": sha(args.model),
        "lock_sha256": sha(args.lock),
        "transfer_feedback_used": False,
        "c006_opened": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["held_out_micro_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
