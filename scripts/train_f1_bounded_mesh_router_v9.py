#!/usr/bin/env python3
"""Train and validate the source-only bounded-mesh precision router V9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import (
    FEATURE_NAMES,
    build_feature_matrix,
    endpoints,
    select_indices_configurable,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, match_lines, pair_metrics
from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh


PROFILE = "F1_BOUNDED_MESH_COMPLEXITY_ROUTER_V9"
RANDOM_STATE = 20260812
SCORE_THRESHOLD = 0.45
SMALL_POOL_MAXIMUM = 400
NMS_ANGLE = 6.0
NMS_DISTANCE = 5.0
NMS_OVERLAP = 0.50
TRAINING_WEIGHTING_ROW_CLASS_BALANCED = "row_class_balanced"
TRAINING_WEIGHTING_EQUAL_FAMILY_EQUAL_CLASS = "equal_family_equal_class_v1"


def _model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=4,
        max_features=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def family_class_balanced_weights(
    groups: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Give each family equal mass and split its mass equally by class."""

    weights = np.zeros(len(target), dtype=np.float64)
    families = sorted(set(groups.tolist()))
    for family in families:
        family_mask = groups == family
        classes = sorted(set(target[family_mask].tolist()))
        for label in classes:
            mask = family_mask & (target == label)
            weights[mask] = 1.0 / (
                len(families) * len(classes) * int(mask.sum())
            )
    return weights


def _fit_estimator(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    training_weighting: str,
) -> ExtraTreesClassifier:
    estimator = _model()
    if training_weighting == TRAINING_WEIGHTING_EQUAL_FAMILY_EQUAL_CLASS:
        estimator.set_params(class_weight=None)
        return estimator.fit(
            features,
            target,
            sample_weight=family_class_balanced_weights(groups, target),
        )
    return estimator.fit(features, target)


def _prediction(rows: list[dict], selected: list[int]) -> list[dict]:
    result = []
    for index in selected:
        line = endpoints(rows[index])
        result.append(
            {
                "bend_id": str(index),
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
            }
        )
    return result


def _metrics(items: list[dict]) -> dict:
    tp = sum(int(row["true_positive"]) for row in items)
    fp = sum(int(row["false_positive"]) for row in items)
    fn = sum(int(row["false_negative"]) for row in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(row["exact"]) for row in items),
        "case_count": len(items),
    }


def _load_cases(pool_dir: Path, gold_manifest: Path) -> list[dict]:
    manifest = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    gold_rows = {
        str(row["case_token"]): row
        for row in json.loads(gold_manifest.read_text(encoding="utf-8"))["cases"]
    }
    thresholds = Thresholds()
    cases = []
    for item in manifest["cases"]:
        token = str(item["case_id"])
        pool = json.loads((pool_dir / f"{token}.json").read_text(encoding="utf-8"))
        rows = [dict(row) for row in pool["line_objects"]]
        gold = list(gold_rows[token]["manual_bends"])
        gold_lines = [
            np.asarray((row["line_start"], row["line_end"]), dtype=float)
            for row in gold
        ]
        target = np.asarray(
            [
                any(is_match(pair_metrics(gold_line, endpoints(row)), thresholds) for gold_line in gold_lines)
                for row in rows
            ],
            dtype=np.uint8,
        )
        cases.append(
            {
                "token": token,
                "family": str(gold_rows[token]["part_key"]),
                "rows": rows,
                "gold": gold,
                "features": build_feature_matrix(rows),
                "target": target,
                "bbox_span": max(float(value) for value in pool["scan_bbox_mm"]),
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--training-weighting",
        choices=(
            TRAINING_WEIGHTING_ROW_CLASS_BALANCED,
            TRAINING_WEIGHTING_EQUAL_FAMILY_EQUAL_CLASS,
        ),
        default=TRAINING_WEIGHTING_ROW_CLASS_BALANCED,
    )
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args()

    cases = _load_cases(args.pool_dir.resolve(), args.gold_manifest.resolve())
    features = np.vstack([case["features"] for case in cases])
    target = np.concatenate([case["target"] for case in cases])
    groups = np.concatenate(
        [[case["family"]] * len(case["rows"]) for case in cases]
    )
    offsets = np.cumsum([0] + [len(case["rows"]) for case in cases])
    held_out_scores = np.zeros(len(target), dtype=float)
    for train, test in GroupKFold(n_splits=5).split(features, target, groups):
        estimator = _fit_estimator(
            features[train], target[train], groups[train], args.training_weighting
        )
        held_out_scores[test] = estimator.predict_proba(features[test])[:, 1]

    by_case = []
    for index, case in enumerate(cases):
        if len(case["rows"]) <= SMALL_POOL_MAXIMUM:
            selected = select_bounded_mesh(case["rows"], case["bbox_span"])
            route = "bounded_mesh_small_pool_recall"
        else:
            scores = held_out_scores[offsets[index] : offsets[index + 1]]
            selected = select_indices_configurable(
                case["rows"],
                scores,
                threshold=args.score_threshold,
                max_angle=NMS_ANGLE,
                max_distance=NMS_DISTANCE,
                min_overlap=NMS_OVERLAP,
            )
            route = "bounded_mesh_complex_pool_precision"
        scored = match_lines(case["gold"], _prediction(case["rows"], selected), Thresholds())
        by_case.append(
            {
                "case_token": case["token"],
                "part_family": case["family"],
                "route": route,
                "line_object_count": len(case["rows"]),
                **{key: scored[key] for key in ("true_positive", "false_positive", "false_negative", "exact")},
            }
        )

    estimator = _fit_estimator(
        features, target, groups, args.training_weighting
    )
    bundle = {
        "profile": PROFILE,
        "feature_names": FEATURE_NAMES,
        "score_threshold": args.score_threshold,
        "small_pool_maximum": SMALL_POOL_MAXIMUM,
        "nms_angle": NMS_ANGLE,
        "nms_distance": NMS_DISTANCE,
        "nms_overlap": NMS_OVERLAP,
        "estimator": estimator,
        "training_case_tokens": tuple(case["token"] for case in cases),
        "training_part_family_count": len({case["family"] for case in cases}),
        "training_weighting_profile": args.training_weighting,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "validation": "five_fold_part_family_grouped_out_of_fold",
        "metrics": _metrics(by_case),
        "cases": by_case,
        "training_line_object_count": len(target),
        "positive_line_object_count": int(target.sum()),
        "part_family_count": bundle["training_part_family_count"],
        "score_threshold": args.score_threshold,
        "small_pool_maximum": SMALL_POOL_MAXIMUM,
        "training_weighting_profile": args.training_weighting,
        "source_only_features": True,
        "part_identity_feature_used": False,
        "expected_count_feature_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
