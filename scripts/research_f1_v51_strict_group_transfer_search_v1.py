#!/usr/bin/env python3
"""Strict case-group transfer search on the consumed Aug-30 development cohort.

This is a truth-aware *research evaluator*, never an inference entry point.  It
fits only on complete training cases, selects the operating point with grouped
inner out-of-fold predictions, and scores complete outer-held-out cases.  The
runtime-shaped policy consumes source candidate features and geometry only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_f1_v51_source_only_recovery_v1 import candidate_endpoints
from scripts.evaluate_f1_against_manual_gold import Thresholds, _endpoints, is_match, match_lines, pair_metrics


CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
CACHE = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2/primary"
ROLE = "2026-08-30_ai_multiview_consensus_development"
MATCH = Thresholds(5.0, 4.0, 0.60, 0.60)
SOURCE_NAMES = (
    "plane_lattice",
    "cylindrical_standard",
    "cylindrical_wide",
    "scan_mesh_adjacency",
    "strict_blind_owned",
    "bend_hypothesis",
    "endpoint_connector",
    "fragment_merge",
    "control_support_strip",
)


@dataclass
class CaseData:
    case_id: str
    gold: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    matrix: np.ndarray
    target: np.ndarray
    v51_ids: set[str]


def _numeric_feature_names(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    excluded = {
        "surface1_id", "surface2_id", "support_atom_ids_truncated", "admissible",
    }
    names = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                names.add(key)
    return tuple(sorted(names))


def _source(row: dict[str, Any]) -> str:
    return str(row.get("representative_source") or row.get("candidate_source") or "")


def _matrix(
    rows: list[dict[str, Any]],
    names: tuple[str, ...],
    v51_ids: set[str],
) -> np.ndarray:
    values = []
    for row in rows:
        source = _source(row)
        record = []
        for name in names:
            value = row.get(name)
            number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
            record.append(number if math.isfinite(number) else math.nan)
        record.extend(float(source == name) for name in SOURCE_NAMES)
        record.append(float(str(row.get("candidate_id")) in v51_ids))
        values.append(record)
    return np.asarray(values, dtype=np.float64)


def load_cases() -> tuple[list[CaseData], tuple[str, ...]]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    records = [dict(row) for row in catalog["cases"] if row.get("scan_role") == ROLE]
    if [row["case_id"] for row in records] != [f"T{i:03d}" for i in range(1, 29)]:
        raise ValueError("expected ordered T001..T028")
    raw: list[tuple[dict[str, Any], list[dict[str, Any]], set[str]]] = []
    all_rows = []
    for case in records:
        case_id = str(case["case_id"])
        pool = json.loads((CACHE / case_id / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
        rows = [dict(row) for row in pool["line_objects"]]
        v51 = json.loads((CACHE / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        v51_ids = {str(row.get("source_hypothesis_id")) for row in v51["predictions"]}
        raw.append((case, rows, v51_ids))
        all_rows.extend(rows)
    names = _numeric_feature_names(all_rows)
    result = []
    for case, rows, v51_ids in raw:
        gold_endpoints = [_endpoints(row) for row in case["bends"]]
        target = np.asarray([
            any(is_match(pair_metrics(truth, candidate_endpoints(row)), MATCH) for truth in gold_endpoints)
            for row in rows
        ], dtype=np.bool_)
        result.append(CaseData(
            case_id=str(case["case_id"]),
            gold=[dict(row) for row in case["bends"]],
            rows=rows,
            matrix=_matrix(rows, names, v51_ids),
            target=target,
            v51_ids=v51_ids,
        ))
    return result, names + tuple(f"source_{name}" for name in SOURCE_NAMES) + ("selected_by_v51",)


def _extra_trees(depth: int | None, leaf: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesClassifier(
            n_estimators=256,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=1701,
            n_jobs=-1,
        ),
    )


def _hist(leaf: int, l2: float) -> Any:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=leaf,
        l2_regularization=l2,
        class_weight="balanced",
        random_state=1701,
    )


MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "extra_d12_l2": lambda: _extra_trees(12, 2),
    "extra_d18_l2": lambda: _extra_trees(18, 2),
    "extra_full_l3": lambda: _extra_trees(None, 3),
    "hist_l20_l2": lambda: _hist(20, 2.0),
    "hist_l40_l5": lambda: _hist(40, 5.0),
}


def _fit(factory: Callable[[], Any], cases: list[CaseData]) -> Any:
    model = factory()
    model.fit(np.concatenate([case.matrix for case in cases]), np.concatenate([case.target for case in cases]))
    return model


def _predict(model: Any, case: CaseData) -> np.ndarray:
    return np.asarray(model.predict_proba(case.matrix)[:, 1], dtype=np.float64)


def _duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = pair_metrics(candidate_endpoints(left), candidate_endpoints(right))
    return bool(metrics["axis_angle_deg"] <= 5.0 and metrics["transverse_offset_mm"] <= 4.0 and metrics["shorter_span_overlap"] >= 0.60)


def select(case: CaseData, scores: np.ndarray, *, threshold: float, cap: int) -> list[dict[str, Any]]:
    ranked = sorted(
        zip(scores.tolist(), case.rows),
        key=lambda value: (-round(float(value[0]), 12), -float(value[1].get("span_mm") or 0.0), str(value[1]["candidate_id"])),
    )
    predictions = []
    for score, row in ranked:
        if round(float(score), 12) < threshold:
            break
        prediction = {
            "bend_id": f"STRICT-{row['candidate_id']}",
            "line_start": candidate_endpoints(row)[0].tolist(),
            "line_end": candidate_endpoints(row)[1].tolist(),
        }
        if any(_duplicate(row, incumbent["_source"]) for incumbent in predictions):
            continue
        prediction["_source"] = row
        predictions.append(prediction)
        if len(predictions) >= cap:
            break
    for prediction in predictions:
        prediction.pop("_source")
    return predictions


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    return {
        "case_count": len(rows), "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp else 0.0,
        "prediction_count": tp + fp, "gold_bend_count": tp + fn,
    }


def _score(cases: list[CaseData], probabilities: dict[str, np.ndarray], threshold: float, cap: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for case in cases:
        prediction = select(case, probabilities[case.case_id], threshold=threshold, cap=cap)
        row = match_lines(case.gold, prediction, MATCH)
        row["case_id"] = case.case_id
        rows.append(row)
    return _aggregate(rows), rows


THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
CAPS = (1, 2, 3, 4, 5, 6)


def _inner_select(train_cases: list[CaseData]) -> tuple[str, float, int, dict[str, Any]]:
    n_splits = min(5, len(train_cases))
    groups = np.arange(len(train_cases))
    splitter = GroupKFold(n_splits=n_splits)
    best = None
    diagnostics = []
    for model_name, factory in MODEL_FACTORIES.items():
        probabilities: dict[str, np.ndarray] = {}
        dummy = np.zeros(len(train_cases))
        for fit_indices, test_indices in splitter.split(dummy, dummy, groups):
            fit_cases = [train_cases[int(index)] for index in fit_indices]
            model = _fit(factory, fit_cases)
            for index in test_indices:
                case = train_cases[int(index)]
                probabilities[case.case_id] = _predict(model, case)
        for threshold in THRESHOLDS:
            for cap in CAPS:
                metrics, _rows = _score(train_cases, probabilities, threshold, cap)
                rank = (metrics["f1"], metrics["true_positive"], -metrics["false_positive"], -cap, threshold)
                diagnostics.append((rank, model_name, threshold, cap, metrics))
                if best is None or rank > best[0]:
                    best = (rank, model_name, threshold, cap, metrics)
    assert best is not None
    _, model_name, threshold, cap, metrics = best
    return model_name, threshold, cap, metrics


def nested_outer(cases: list[CaseData], n_splits: int) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=n_splits)
    group = np.arange(len(cases))
    dummy = np.zeros(len(cases))
    rows = []
    folds = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(dummy, dummy, group), 1):
        train = [cases[int(index)] for index in train_indices]
        test = [cases[int(index)] for index in test_indices]
        model_name, threshold, cap, inner = _inner_select(train)
        model = _fit(MODEL_FACTORIES[model_name], train)
        probabilities = {case.case_id: _predict(model, case) for case in test}
        metrics, fold_rows = _score(test, probabilities, threshold, cap)
        for row in fold_rows:
            row["outer_fold"] = fold
        rows.extend(fold_rows)
        folds.append({
            "fold": fold,
            "train_cases": [case.case_id for case in train],
            "held_out_cases": [case.case_id for case in test],
            "selected_model": model_name,
            "selected_threshold": threshold,
            "selected_cap": cap,
            "inner_group_oof": inner,
            "outer_held_out": metrics,
        })
        print(json.dumps(folds[-1], sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["case_id"])
    return {"metrics": _aggregate(rows), "folds": folds, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-folds", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases, feature_names = load_cases()
    result = {
        "schema_version": 1,
        "kind": "f1_v51_consumed_development_strict_nested_group_transfer_search_v1",
        "candidate_universe": "cached_scan_only_line_objects_raw_geometry",
        "truth_visible_only_to_fitter_and_evaluator": True,
        "part_identity_feature_used": False,
        "case_id_feature_used": False,
        "feature_names": list(feature_names),
        "nested_group_evaluation": nested_outer(cases, args.outer_folds),
    }
    print(json.dumps(result["nested_group_evaluation"]["metrics"], sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
