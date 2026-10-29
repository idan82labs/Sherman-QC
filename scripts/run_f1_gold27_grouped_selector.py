#!/usr/bin/env python3
"""Cross-fit the F1 candidate selector on the frozen 27-part gold corpus.

Each outer fold is a locked set of physical part families. Threshold and
duplicate-suppression choices are selected only by inner cross-validation over
the remaining frozen folds. The evaluated part family is therefore absent
from model fitting and hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402
from scripts.run_f1_grouped_holdout_selector import (  # noqa: E402
    FEATURE_NAMES,
    _attach_arrangement_features,
    _features,
    _fit,
    _match_quality,
    _predict,
    _ranked_selection,
)


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_SPLIT = ROOT / "output/research/f1-gold27-20260803/freeze/grouped_five_fold_split.json"
DEFAULT_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/grouped_selector_v1"
PROBABILITY_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
DUPLICATE_TOLERANCES_MM = (5.0, 8.0)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for token in sorted(predictions):
        result = match_lines(list(gold[token]), list(predictions[token]), Thresholds())
        tp = int(result["true_positive"])
        fp = int(result["false_positive"])
        fn = int(result["false_negative"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result["line_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["case_token"] = token
        rows.append(result)
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    line_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "line_f1": round(line_f1, 6),
        "exact_scans": sum(bool(row["exact"]) for row in rows),
        "scan_count": len(rows),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-kind", choices=("line_objects", "raw_candidates"), default="line_objects")
    parser.add_argument("--model-kind", choices=("extra_trees", "logistic"), default="extra_trees")
    parser.add_argument("--arrangement-features", action="store_true")
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    split = _read(args.split_manifest.resolve())
    if sealed["corpus_freeze_id"] != split["corpus_freeze_id"]:
        raise ValueError("freeze ID mismatch")
    cases = {str(case["case_token"]): case for case in sealed.get("cases") or []}
    gold = {token: list(case["manual_bends"]) for token, case in cases.items()}
    family = {token: str(case["part_key"]) for token, case in cases.items()}
    fold_tokens = {
        int(fold["fold"]): sorted(
            token for token in cases if family[token] in set(fold["locked_part_families"])
        )
        for fold in split.get("folds") or []
    }
    if sorted(token for values in fold_tokens.values() for token in values) != sorted(cases):
        raise ValueError("grouped split does not partition every frozen case exactly once")

    payloads = {token: _read(args.candidate_root.resolve() / f"{token}.json") for token in cases}
    candidate_key = "line_objects" if args.candidate_kind == "line_objects" else "candidates"
    candidate_map: dict[str, list[dict[str, Any]]] = {}
    feature_map: dict[str, np.ndarray] = {}
    label_map: dict[str, np.ndarray] = {}
    quality_map: dict[str, np.ndarray] = {}
    for token, payload in payloads.items():
        candidates = [dict(candidate) for candidate in payload.get(candidate_key) or []]
        if args.arrangement_features:
            _attach_arrangement_features(candidates, payload)
        candidate_map[token] = candidates
        feature_map[token] = np.asarray(
            [_features(candidate, payload, include_arrangement=args.arrangement_features) for candidate in candidates],
            dtype=np.float64,
        )
        qualities = np.asarray([_match_quality(candidate, gold[token], Thresholds()) for candidate in candidates])
        quality_map[token] = qualities
        label_map[token] = (qualities > 0).astype(np.int64)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crossfit_predictions: dict[str, list[dict[str, Any]]] = {}
    fold_reports = []
    all_fold_ids = sorted(fold_tokens)
    for outer_fold in all_fold_ids:
        held_out = fold_tokens[outer_fold]
        training_folds = [fold for fold in all_fold_ids if fold != outer_fold]
        inner_predictions = {
            (threshold, duplicate): {}
            for threshold in PROBABILITY_THRESHOLDS
            for duplicate in DUPLICATE_TOLERANCES_MM
        }
        for inner_fold in training_folds:
            validation = fold_tokens[inner_fold]
            inner_training = [
                token
                for fold in training_folds
                if fold != inner_fold
                for token in fold_tokens[fold]
            ]
            features = np.vstack([feature_map[token] for token in inner_training])
            labels = np.concatenate([label_map[token] for token in inner_training])
            qualities = np.concatenate([quality_map[token] for token in inner_training])
            classifier, regressor = _fit(features, labels, qualities, model_kind=args.model_kind)
            for token in validation:
                probabilities, predicted_qualities = _predict(classifier, regressor, feature_map[token])
                for threshold in PROBABILITY_THRESHOLDS:
                    for duplicate in DUPLICATE_TOLERANCES_MM:
                        inner_predictions[(threshold, duplicate)][token] = _ranked_selection(
                            candidate_map[token],
                            probabilities,
                            predicted_qualities,
                            probability_threshold=threshold,
                            cylindrical_offset_cap_mm=1.0e9,
                            duplicate_offset_tolerance_mm=duplicate,
                        )
        ranked_configs = [
            (threshold, duplicate, _aggregate(predictions, gold))
            for (threshold, duplicate), predictions in inner_predictions.items()
        ]
        selected_threshold, selected_duplicate, inner_metrics = max(
            ranked_configs,
            key=lambda item: (
                item[2]["line_f1"],
                item[2]["exact_scans"],
                item[2]["recall"],
                -item[2]["false_positive"],
                item[0],
                -item[1],
            ),
        )
        training = [token for fold in training_folds for token in fold_tokens[fold]]
        features = np.vstack([feature_map[token] for token in training])
        labels = np.concatenate([label_map[token] for token in training])
        qualities = np.concatenate([quality_map[token] for token in training])
        classifier, regressor = _fit(features, labels, qualities, model_kind=args.model_kind)
        held_out_predictions = {}
        for token in held_out:
            probabilities, predicted_qualities = _predict(classifier, regressor, feature_map[token])
            selected = _ranked_selection(
                candidate_map[token],
                probabilities,
                predicted_qualities,
                probability_threshold=selected_threshold,
                cylindrical_offset_cap_mm=1.0e9,
                duplicate_offset_tolerance_mm=selected_duplicate,
            )
            for index, candidate in enumerate(selected, start=1):
                candidate["bend_id"] = f"F1G{index}"
            crossfit_predictions[token] = selected
            held_out_predictions[token] = selected
            case_dir = output_dir / "cases" / token
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction_owned_regions.json").write_text(
                json.dumps({"owned_bend_regions": selected}, indent=2) + "\n",
                encoding="utf-8",
            )
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "held_out_case_tokens": held_out,
                "held_out_part_families": sorted({family[token] for token in held_out}),
                "training_part_families": sorted({family[token] for token in training}),
                "selected_probability_threshold": selected_threshold,
                "selected_duplicate_tolerance_mm": selected_duplicate,
                "inner_validation_metrics": {key: value for key, value in inner_metrics.items() if key != "cases"},
                "held_out_metrics": {key: value for key, value in _aggregate(held_out_predictions, gold).items() if key != "cases"},
            }
        )
        print(
            f"fold={outer_fold} threshold={selected_threshold:.2f} duplicate={selected_duplicate:.1f} "
            f"heldout_f1={fold_reports[-1]['held_out_metrics']['line_f1']:.3f}",
            flush=True,
        )

    metrics = _aggregate(crossfit_predictions, gold)
    exact_families = sum(
        all(next(row for row in metrics["cases"] if row["case_token"] == token)["exact"] for token in tokens)
        for part in sorted(set(family.values()))
        if (tokens := [token for token in cases if family[token] == part])
    )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_grouped_crossfit_selector",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "evaluation_mode": "five_fold_outer_part_group_crossfit_with_inner_fold_selection",
        "leakage_contract": {
            "same_scan_labels_in_training": False,
            "same_part_labels_in_training": False,
            "outer_fold_labels_used_for_hyperparameter_selection": False,
            "candidate_generation_uses_manual_truth": False,
            "part_identifier_or_expected_count_used_as_feature": False,
        },
        "candidate_kind": args.candidate_kind,
        "model_kind": args.model_kind,
        "arrangement_features": bool(args.arrangement_features),
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
        "exact_part_families": exact_families,
        "total_part_families": len(set(family.values())),
        "folds": fold_reports,
    }
    report_path = output_dir / "grouped_crossfit_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {key: value for key, value in metrics.items() if key != "cases"}, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
