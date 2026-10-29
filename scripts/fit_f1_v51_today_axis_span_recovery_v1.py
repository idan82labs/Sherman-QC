#!/usr/bin/env python3
"""Fit and diagnose the consumed-development axis/span recovery tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_post_v51_precision_ownership_v52_development import compose_v52_development
from scripts.apply_f1_v51_today_axis_span_recovery_v1 import (
    DEFAULT_FIXED_SPANS_MM,
    DEFAULT_SPAN_SCALES,
    FEATURE_NAMES,
    MODEL_KIND,
    _canonical_bytes,
    _duplicate,
    generate_axis_span_variants,
)
from scripts.apply_f1_v51_source_only_recovery_v1 import candidate_endpoints
from scripts.evaluate_f1_against_manual_gold import (
    Thresholds,
    _endpoints,
    is_match,
    match_lines,
    pair_metrics,
)


DEFAULT_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
DEFAULT_V51_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2/primary"
DEFAULT_RECALL_ROOT = ROOT / "output/research/f1-v51-today-source-recovery-v1-20260831-r3/balanced/primary"
DEFAULT_OUTPUT = ROOT / "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json"
TODAY_SCAN_ROLE = "2026-08-30_ai_multiview_consensus_development"
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)
TREE_PARAMETERS = {
    "criterion": "entropy",
    "max_depth": 24,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 0,
}
OPERATING_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0)
OPERATING_CAPS = (1, 2, 3, 4)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _today_cases(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [dict(row) for row in catalog.get("cases") or [] if row.get("scan_role") == TODAY_SCAN_ROLE]
    if [row.get("case_id") for row in cases] != [f"T{index:03d}" for index in range(1, 29)]:
        raise ValueError("catalog must contain ordered daily cases T001..T028")
    return cases


def _feature_matrix(variants: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in variants],
        dtype=np.float32,
    )


def _variant_labels(
    gold_records: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> np.ndarray:
    gold = [_endpoints(row) for row in gold_records]
    return np.asarray(
        [
            any(is_match(pair_metrics(truth, _endpoints(row)), THRESHOLDS) for truth in gold)
            for row in variants
        ],
        dtype=np.bool_,
    )


def _base_union(v51: dict[str, Any], recall: dict[str, Any], pool: dict[str, Any]) -> list[dict[str, Any]]:
    precision = compose_v52_development(v51, pool)
    incumbent_ids = {str(row.get("bend_id")) for row in v51.get("predictions") or []}
    additions = [row for row in recall.get("predictions") or [] if str(row.get("bend_id")) not in incumbent_ids]
    return [dict(row) for row in precision["predictions"]] + [dict(row) for row in additions]


def _case_inputs(
    case_id: str, v51_root: Path, recall_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_root = v51_root / case_id
    v51 = json.loads((case_root / "semantic_result.json").read_text(encoding="utf-8"))
    pool = json.loads((case_root / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
    recall = json.loads((recall_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
    return v51, recall, pool


def _tree_payload(classifier: DecisionTreeClassifier) -> dict[str, Any]:
    tree = classifier.tree_
    positive_index = list(classifier.classes_).index(True)
    probabilities = []
    for node in range(tree.node_count):
        values = np.asarray(tree.value[node]).reshape(-1)
        probabilities.append(float(values[positive_index] / values.sum()))
    return {
        "children_left": [int(value) for value in tree.children_left],
        "children_right": [int(value) for value in tree.children_right],
        "feature_index": [int(value) for value in tree.feature],
        "threshold": [float(value) for value in tree.threshold],
        "positive_probability": probabilities,
    }


def _select_variants(
    base: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    threshold: float,
    cap: int,
) -> list[dict[str, Any]]:
    ranked = [
        (round(float(probability), 12), row)
        for probability, row in zip(probabilities, variants)
        if round(float(probability), 12) >= threshold
    ]
    ranked.sort(
        key=lambda item: (
            -item[0],
            abs(
                np.log(
                    max(
                        float(item[1]["axis_span_variant"]["variant_span_mm"])
                        / float(item[1]["axis_span_variant"]["base_span_mm"]),
                        1e-12,
                    )
                )
            ),
            str(item[1]["bend_id"]),
        )
    )
    profile = {
        "duplicate_angle_deg": 5.0,
        "duplicate_offset_mm": 4.0,
        "duplicate_span_overlap": 0.60,
    }
    emitted: list[dict[str, Any]] = []
    owned_hypotheses: set[str] = set()
    for probability, row in ranked:
        hypothesis = str(row["source_hypothesis_id"])
        if hypothesis in owned_hypotheses:
            continue
        if any(_duplicate(row, incumbent, profile) for incumbent in base):
            continue
        if any(_duplicate(row, incumbent, profile) for incumbent in emitted):
            continue
        clean = {key: value for key, value in row.items() if key != "features"}
        clean["selector_score"] = probability
        emitted.append(clean)
        owned_hypotheses.add(hypothesis)
        if len(emitted) >= cap:
            break
    return base + emitted


def _aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in scored)
    fp = sum(int(row["false_positive"]) for row in scored)
    fn = sum(int(row["false_negative"]) for row in scored)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if tp else 0.0
    return {
        "case_count": len(scored),
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(row["exact"]) for row in scored),
    }


def _evaluate_classifier(
    classifier: DecisionTreeClassifier,
    cases: list[dict[str, Any]],
    v51_root: Path,
    recall_root: Path,
    *,
    threshold: float,
    cap: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    for case in cases:
        case_id = str(case["case_id"])
        v51, recall, pool = _case_inputs(case_id, v51_root, recall_root)
        base = _base_union(v51, recall, pool)
        variants = generate_axis_span_variants(pool)
        probabilities = classifier.predict_proba(_feature_matrix(variants))[:, list(classifier.classes_).index(True)]
        predictions = _select_variants(base, variants, probabilities, threshold=threshold, cap=cap)
        result = match_lines(case["bends"], predictions, THRESHOLDS)
        result["case_id"] = case_id
        result["base_prediction_count"] = len(base)
        result["variant_emitted_count"] = len(predictions) - len(base)
        scored.append(result)
    return _aggregate(scored), scored


def _evaluate_operating_grid(
    classifier: DecisionTreeClassifier,
    cases: list[dict[str, Any]],
    v51_root: Path,
    recall_root: Path,
) -> dict[tuple[float, int], tuple[dict[str, Any], list[dict[str, Any]]]]:
    grids: dict[tuple[float, int], list[dict[str, Any]]] = {
        (threshold, cap): []
        for threshold in OPERATING_THRESHOLDS
        for cap in OPERATING_CAPS
    }
    positive_index = list(classifier.classes_).index(True)
    for case in cases:
        case_id = str(case["case_id"])
        v51, recall, pool = _case_inputs(case_id, v51_root, recall_root)
        base = _base_union(v51, recall, pool)
        variants = generate_axis_span_variants(pool)
        probabilities = classifier.predict_proba(_feature_matrix(variants))[:, positive_index]
        for (threshold, cap), scored in grids.items():
            predictions = _select_variants(
                base,
                variants,
                probabilities,
                threshold=threshold,
                cap=cap,
            )
            result = match_lines(case["bends"], predictions, THRESHOLDS)
            result["case_id"] = case_id
            result["base_prediction_count"] = len(base)
            result["variant_emitted_count"] = len(predictions) - len(base)
            scored.append(result)
    return {key: (_aggregate(rows), rows) for key, rows in grids.items()}


def fit(catalog_path: Path, v51_root: Path, recall_root: Path) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    v51_root = v51_root.resolve()
    recall_root = recall_root.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cases = _today_cases(catalog)
    matrices = []
    labels = []
    groups = []
    pool_hashes = []
    variant_count_by_case: dict[str, int] = {}
    positive_count_by_case: dict[str, int] = {}
    for case_index, case in enumerate(cases):
        case_id = str(case["case_id"])
        v51, recall, pool = _case_inputs(case_id, v51_root, recall_root)
        pool_path = v51_root / case_id / "pool" / f"{case_id}.json"
        pool_hashes.append(_sha(pool_path))
        variants = generate_axis_span_variants(pool)
        matrix = _feature_matrix(variants)
        base = _base_union(v51, recall, pool)
        base_score = match_lines(case["bends"], base, THRESHOLDS)
        matched_gold_ids = {str(row["gold_id"]) for row in base_score["matches"]}
        uncovered_gold = [
            row for row in case["bends"] if str(row["bend_id"]) not in matched_gold_ids
        ]
        target = _variant_labels(uncovered_gold, variants)
        matrices.append(matrix)
        labels.append(target)
        groups.append(np.full(len(target), case_index, dtype=np.int16))
        variant_count_by_case[case_id] = len(variants)
        positive_count_by_case[case_id] = int(target.sum())
    matrix = np.concatenate(matrices)
    target = np.concatenate(labels)
    group = np.concatenate(groups)
    classifier = DecisionTreeClassifier(**TREE_PARAMETERS)
    classifier.fit(matrix, target)

    grid = _evaluate_operating_grid(classifier, cases, v51_root, recall_root)
    best: tuple[float, int, dict[str, Any], list[dict[str, Any]]] | None = None
    operating_search = []
    for (threshold, cap), (metrics, scored) in grid.items():
        operating_search.append({"threshold": threshold, "cap": cap, "metrics": metrics})
        rank = (
            float(metrics["f1"]),
            int(metrics["true_positive"]),
            -int(metrics["false_positive"]),
            -cap,
            threshold,
        )
        if best is None or rank > (
            float(best[2]["f1"]),
            int(best[2]["true_positive"]),
            -int(best[2]["false_positive"]),
            -best[1],
            best[0],
        ):
            best = (threshold, cap, metrics, scored)
    assert best is not None
    threshold, cap, resub_metrics, resub_cases = best

    # Seven grouped folds hold out four complete scans each.  The final
    # full-cohort operating point is frozen for this diagnostic; therefore it
    # is informative about model transfer across cases but is not a nested or
    # untouched estimate.
    oof_rows = []
    splitter = GroupKFold(n_splits=7)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(matrix, target, group), 1):
        fold_classifier = DecisionTreeClassifier(**TREE_PARAMETERS)
        fold_classifier.fit(matrix[train_indices], target[train_indices])
        held_out_case_indices = sorted({int(value) for value in group[test_indices]})
        held_out_cases = [cases[index] for index in held_out_case_indices]
        _fold_metrics, fold_rows = _evaluate_classifier(
            fold_classifier,
            held_out_cases,
            v51_root,
            recall_root,
            threshold=threshold,
            cap=cap,
        )
        for row in fold_rows:
            row["fold"] = fold
        oof_rows.extend(fold_rows)
    oof_rows.sort(key=lambda row: str(row["case_id"]))
    oof_metrics = _aggregate(oof_rows)

    tree = _tree_payload(classifier)
    return {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "model_name": "V51_TODAY_AXIS_SPAN_INCREMENTAL_RECOVERY_DEVELOPMENT_V1_R2",
        "development_tuned_model": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "training_evidence_class": "CONSUMED_AI_ADJUDICATED_PHYSICAL_DEVELOPMENT",
        "training_catalog_sha256": _sha(catalog_path),
        "training_pool_set_sha256": hashlib.sha256("\n".join(pool_hashes).encode()).hexdigest(),
        "training_variant_row_count": int(len(target)),
        "training_positive_variant_row_count": int(target.sum()),
        "training_target": "variants_matching_gold_uncovered_by_precision_plus_r3_recall_union",
        "feature_names": list(FEATURE_NAMES),
        "tree_parameters": TREE_PARAMETERS,
        "tree": tree,
        "profile": {
            "probability_threshold": threshold,
            "recovery_cap": cap,
            "duplicate_angle_deg": 5.0,
            "duplicate_offset_mm": 4.0,
            "duplicate_span_overlap": 0.60,
            "span_scales": list(DEFAULT_SPAN_SCALES),
            "fixed_spans_mm": list(DEFAULT_FIXED_SPANS_MM),
        },
        "development_diagnostics": {
            "resubstitution_metrics": resub_metrics,
            "resubstitution_cases": [
                {
                    "case_id": row["case_id"],
                    "true_positive": row["true_positive"],
                    "false_positive": row["false_positive"],
                    "false_negative": row["false_negative"],
                    "variant_emitted_count": row["variant_emitted_count"],
                }
                for row in resub_cases
            ],
            "case_group_held_out_metrics": oof_metrics,
            "case_group_held_out_fold_count": 7,
            "case_group_held_out_operating_point_nested": False,
            "case_group_held_out_cases": [
                {
                    "case_id": row["case_id"],
                    "fold": row["fold"],
                    "true_positive": row["true_positive"],
                    "false_positive": row["false_positive"],
                    "false_negative": row["false_negative"],
                    "variant_emitted_count": row["variant_emitted_count"],
                }
                for row in oof_rows
            ],
            "variant_count_by_case": variant_count_by_case,
            "positive_variant_count_by_case": positive_count_by_case,
            "operating_search": operating_search,
            "overfit_caveat": (
                "The tree and operating point consume all 28 AI-adjudicated development cases. "
                "Resubstitution is optimization evidence only. Group-held-out tree predictions reuse "
                "the full-cohort operating point and are diagnostic, not untouched generalization evidence."
            ),
        },
        "inference_contract": {
            "catalog_opened_at_inference": False,
            "labels_opened_at_inference": False,
            "expected_count_used_at_inference": False,
            "cad_used_at_inference": False,
            "part_identity_used_at_inference": False,
            "runtime_inputs": [
                "retained_v51_semantic_result",
                "cached_v51_source_recovery_result",
                "scan_only_candidate_pool",
                "this_model",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--v51-root", type=Path, default=DEFAULT_V51_ROOT)
    parser.add_argument("--recall-root", type=Path, default=DEFAULT_RECALL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = fit(args.catalog, args.v51_root, args.recall_root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen axis/span model")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha(output),
                "tree_nodes": len(payload["tree"]["children_left"]),
                "resubstitution": payload["development_diagnostics"]["resubstitution_metrics"],
                "case_group_held_out": payload["development_diagnostics"]["case_group_held_out_metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
