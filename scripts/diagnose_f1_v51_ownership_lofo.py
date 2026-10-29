#!/usr/bin/env python3
"""Nested-LOFO diagnostic for a V51 source-feature ownership selector.

This is a funding diagnostic, not a model version.  Each outer physical family
is excluded from model fitting and from threshold selection.  The inner
leave-one-family-out predictions choose a threshold by line-level F1, then the
outer family is scored with the frozen geometric matcher.  If neither fixed
model family beats unchanged V51, no V74 ownership selector is funded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases


NUMERIC_FEATURES = (
    "case_percentile_support_count",
    "case_percentile_score",
    "case_percentile_consensus_count",
    "case_percentile_finite_support_fraction",
    "case_percentile_finite_support_longest_run_fraction",
    "case_percentile_endpoint_support_fraction",
    "case_percentile_normal_second_eigenvalue",
    "case_percentile_normal_spread_deg_scan",
    "case_percentile_corridor_points_per_mm",
    "case_percentile_transverse_angular_span_deg",
    "case_percentile_cluster_member_count",
    "case_percentile_cluster_source_count",
    "case_percentile_two_sided_transition_score",
    "case_percentile_two_sided_normal_angle_deg",
    "case_percentile_two_sided_support_balance",
    "case_percentile_two_sided_min_support_fraction",
    "case_percentile_two_sided_normal_coherence",
    "case_percentile_tight_normal_second_eigenvalue_r2mm",
    "case_percentile_tight_to_wide_normal_second_ratio",
    "finite_support_fraction",
    "finite_support_longest_run_fraction",
    "endpoint_support_fraction",
    "two_sided_transition_score",
    "two_sided_min_support_fraction",
    "two_sided_support_balance",
    "corridor_points_per_mm",
    "cluster_member_count",
    "cluster_source_count",
)
THRESHOLDS = tuple(float(value) for value in np.arange(0.10, 0.91, 0.05))


def _gold_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["scan_sha256"]): row for row in catalog["cases"]}


def _feature(candidate: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float]:
    values = {name: float(candidate.get(name) or 0.0) for name in NUMERIC_FEATURES}
    source = str(candidate.get("representative_source") or candidate.get("candidate_source") or "unknown")
    values[f"source={source}"] = 1.0
    values[f"route={prediction.get('selection_route') or 'unknown'}"] = 1.0
    return values


def _load_cases(public: dict[str, Any], catalog: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    indexed_gold = _gold_index(catalog)
    cases = []
    for public_row in public["cases"]:
        gold = indexed_gold.get(str(public_row["source_sha256"]))
        if gold is None or gold.get("truth_tier") != "codex_manual_multiview_count_backed":
            continue
        case_id = str(public_row["opaque_case_id"])
        result = json.loads((root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        pool = json.loads((root / "primary" / case_id / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
        candidate_index: dict[str, dict[str, Any]] = {}
        for candidate in pool["line_objects"]:
            for key in ("bend_id", "candidate_id", "source_candidate_id"):
                if candidate.get(key) is not None:
                    candidate_index[str(candidate[key])] = candidate
        full_match = match_lines(gold["bends"], result["predictions"], Thresholds())
        matched_ids = {str(row["prediction_id"]) for row in full_match["matches"]}
        rows = []
        for prediction in result["predictions"]:
            source_id = str(prediction["source_hypothesis_id"])
            candidate = candidate_index.get(source_id)
            if candidate is None:
                raise ValueError(f"{case_id}: no pool record for {source_id}")
            rows.append(
                {
                    "prediction": prediction,
                    "feature": _feature(candidate, prediction),
                    "label": int(str(prediction["bend_id"]) in matched_ids),
                }
            )
        cases.append({"case_id": case_id, "gold": gold["bends"], "rows": rows})
    if len(cases) != 12:
        raise ValueError(f"expected 12 cases, got {len(cases)}")
    return cases


def _fit(cases: list[dict[str, Any]], kind: str) -> tuple[DictVectorizer, Any]:
    vectorizer = DictVectorizer(sparse=False)
    features = vectorizer.fit_transform([row["feature"] for case in cases for row in case["rows"]])
    labels = np.asarray([row["label"] for case in cases for row in case["rows"]], dtype=np.int64)
    common = dict(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.7,
        class_weight="balanced",
        random_state=17,
        n_jobs=1,
    )
    classifier = ExtraTreesClassifier(**common) if kind == "extra_trees" else RandomForestClassifier(**common)
    classifier.fit(features, labels)
    return vectorizer, classifier


def _probabilities(vectorizer: DictVectorizer, classifier: Any, case: dict[str, Any]) -> np.ndarray:
    matrix = vectorizer.transform([row["feature"] for row in case["rows"]])
    return classifier.predict_proba(matrix)[:, 1]


def _score_case(case: dict[str, Any], probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = [
        row["prediction"]
        for row, probability in zip(case["rows"], probabilities)
        if float(probability) >= threshold
    ]
    result = match_lines(case["gold"], predictions, Thresholds())
    result["case_id"] = case["case_id"]
    result["selected_prediction_ids"] = [str(row["bend_id"]) for row in predictions]
    result["selected_predictions"] = predictions
    return result


def _threshold(inner_predictions: list[tuple[dict[str, Any], np.ndarray]]) -> float:
    best: tuple[float, float] | None = None
    for threshold in THRESHOLDS:
        scored = score_localized_cases(
            [_score_case(case, probabilities, threshold) for case, probabilities in inner_predictions]
        )
        candidate = (float(scored["f1"] or 0.0), -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[1]


def _run_model(cases: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    outer_rows = []
    for outer_index, outer_case in enumerate(cases):
        train = [case for index, case in enumerate(cases) if index != outer_index]
        inner_predictions = []
        for inner_index, inner_case in enumerate(train):
            inner_train = [case for index, case in enumerate(train) if index != inner_index]
            vectorizer, classifier = _fit(inner_train, kind)
            inner_predictions.append((inner_case, _probabilities(vectorizer, classifier, inner_case)))
        threshold = _threshold(inner_predictions)
        vectorizer, classifier = _fit(train, kind)
        scored = _score_case(outer_case, _probabilities(vectorizer, classifier, outer_case), threshold)
        scored["selected_threshold"] = threshold
        outer_rows.append(scored)
    aggregate = score_localized_cases(outer_rows)
    aggregate["minimum_family_recall"] = min(
        row["true_positive"] / row["gold_count"] for row in outer_rows
    )
    return {"model_kind": kind, "metrics": aggregate, "outer_cases": outer_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--v51-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-prediction-root", type=Path)
    args = parser.parse_args()
    public = json.loads(args.public_manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.gold_catalog.read_text(encoding="utf-8"))
    cases = _load_cases(public, catalog, args.v51_root)
    models = [_run_model(cases, kind) for kind in ("extra_trees", "random_forest")]
    baseline_rows = []
    for case in cases:
        baseline_rows.append(_score_case(case, np.ones(len(case["rows"])), 0.5))
    baseline = score_localized_cases(baseline_rows)
    baseline["minimum_family_recall"] = min(
        row["true_positive"] / row["gold_count"] for row in baseline_rows
    )
    best_model = max(models, key=lambda row: float(row["metrics"]["f1"] or 0.0))
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_ownership_nested_lofo_funding_diagnostic",
        "evidence_class": "development_consumed_physical",
        "promotion_eligible": False,
        "outer_family_labels_used_in_fit_or_threshold": False,
        "baseline": baseline,
        "models": models,
        "best_model_kind": best_model["model_kind"],
        "best_f1_delta": float(best_model["metrics"]["f1"] or 0.0) - float(baseline["f1"] or 0.0),
        "fund_successor": bool(float(best_model["metrics"]["f1"] or 0.0) > float(baseline["f1"] or 0.0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.visual_prediction_root is not None:
        for row in best_model["outer_cases"]:
            payload = {
                "schema_version": 1,
                "kind": "sherman_qc_f1_v51_ownership_nested_lofo_visual_diagnostic",
                "source_only": True,
                "research_only": True,
                "promotion_eligible": False,
                "route": f"diagnostic_nested_lofo_{best_model['model_kind']}",
                "predictions": row["selected_predictions"],
            }
            path = args.visual_prediction_root / row["case_id"] / "semantic_result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "baseline_f1": baseline["f1"], "best_model_kind": report["best_model_kind"], "best_f1_delta": report["best_f1_delta"], "fund_successor": report["fund_successor"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
