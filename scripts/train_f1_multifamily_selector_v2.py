#!/usr/bin/env python3
"""Train the frozen V2 selector on locked development-only oracle feedback."""

from __future__ import annotations

import argparse
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

from research.f1_multifamily_selector_v2.model import (
    FEATURE_NAMES,
    PROFILE,
    RANDOM_STATE,
    SCORE_THRESHOLD,
    build_feature_matrix,
    select_indices,
)


DEVELOPMENT_CASES = ("C008", "C031", "C028", "C022", "C025", "C035", "C021", "C026")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_case(pool_path: Path, run_dir: Path, case_token: str) -> tuple[list[dict[str, Any]], np.ndarray, list[str | None], int]:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    submission_path = run_dir / f"{case_token}_line_objects_submission.json"
    feedback_path = run_dir / f"{case_token}_line_objects_feedback.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    prediction_by_source = {
        str(row["source_hypothesis_id"]): row for row in submission["predictions"]
    }
    feedback_by_id = {str(row["candidate_id"]): row for row in feedback["candidate_feedback"]}
    rows = [dict(row) for row in pool["line_objects"]]
    object_ids: list[str | None] = []
    for row in rows:
        prediction = prediction_by_source[str(row["candidate_id"])]
        object_ids.append(feedback_by_id[str(prediction["bend_id"])]["opaque_object_id"])
    labels = np.asarray([value is not None for value in object_ids], dtype=np.uint8)
    expected = len({value for value in object_ids if value}) + int(feedback["metrics"]["false_negative"])
    return rows, labels, object_ids, expected


def _model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=10,
        max_features=0.7,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _metrics(selections: list[tuple[list[int], list[str | None], int]]) -> dict[str, Any]:
    tp = fp = fn = 0
    for selected, object_ids, expected in selections:
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
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    loaded = {
        case: _load_case(args.pool_dir / f"{case}.json", args.development_run_dir, case)
        for case in DEVELOPMENT_CASES
    }
    matrices = {case: build_feature_matrix(value[0]) for case, value in loaded.items()}
    held_out: list[tuple[list[int], list[str | None], int]] = []
    per_case: dict[str, Any] = {}
    for case in DEVELOPMENT_CASES:
        train_cases = [value for value in DEVELOPMENT_CASES if value != case]
        estimator = _model()
        estimator.fit(
            np.vstack([matrices[value] for value in train_cases]),
            np.concatenate([loaded[value][1] for value in train_cases]),
        )
        scores = estimator.predict_proba(matrices[case])[:, 1]
        selected = select_indices(loaded[case][0], scores)
        item = (selected, loaded[case][2], loaded[case][3])
        held_out.append(item)
        per_case[case] = {
            **_metrics([item]),
            "selected_count": len(selected),
            "candidate_pool_sha256": _sha(args.pool_dir / f"{case}.json"),
        }

    final_model = _model()
    final_model.fit(
        np.vstack([matrices[case] for case in DEVELOPMENT_CASES]),
        np.concatenate([loaded[case][1] for case in DEVELOPMENT_CASES]),
    )
    bundle = {
        "profile": PROFILE,
        "feature_names": FEATURE_NAMES,
        "score_threshold": SCORE_THRESHOLD,
        "development_cases": DEVELOPMENT_CASES,
        "lock_sha256": _sha(args.lock),
        "estimator": final_model,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "validation": "leave_one_family_out",
        "development_cases": list(DEVELOPMENT_CASES),
        "held_out_micro_metrics": _metrics(held_out),
        "held_out_by_case": per_case,
        "score_threshold": SCORE_THRESHOLD,
        "model_sha256": _sha(args.model),
        "lock_sha256": _sha(args.lock),
        "transfer_feedback_used": False,
        "c006_opened": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["held_out_micro_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
