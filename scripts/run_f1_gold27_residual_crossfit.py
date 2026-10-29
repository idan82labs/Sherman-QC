#!/usr/bin/env python3
"""Cross-fit a residual bend selector on top of frozen mesh-lane predictions."""

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

from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    Thresholds,
    _endpoints,
    match_lines,
    pair_metrics,
)
from scripts.run_f1_gold27_grouped_selector import _aggregate  # noqa: E402
from scripts.run_f1_grouped_holdout_selector import (  # noqa: E402
    _duplicate,
    _features,
    _fit,
    _match_quality,
    _predict,
    _ranked_selection,
)


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_SPLIT = ROOT / "output/research/f1-gold27-20260803/freeze/grouped_five_fold_split.json"
DEFAULT_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_MESH = ROOT / "output/research/f1-gold27-20260803/mesh_crossfit_v1"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/residual_crossfit_v1"
THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99)
DUPLICATE_TOLERANCES = (4.0, 6.0, 8.0)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_predictions(mesh_root: Path, tokens: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    return {
        token: list(_read(mesh_root / "cases" / token / "prediction_owned_regions.json")["owned_bend_regions"])
        for token in tokens
    }


def _relation_features(candidate: Mapping[str, Any], base: Sequence[Mapping[str, Any]], diagonal: float) -> list[float]:
    if not base:
        return [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    candidate_line = _endpoints(dict(candidate))
    metrics = [pair_metrics(candidate_line, _endpoints(dict(item))) for item in base]
    endpoint_gaps = [
        min(float(np.linalg.norm(left - right)) for left in candidate_line for right in _endpoints(dict(item)))
        for item in base
    ]
    orthogonal_junctions = sum(
        metric["axis_angle_deg"] >= 25.0 and gap <= max(5.0, 0.02 * diagonal)
        for metric, gap in zip(metrics, endpoint_gaps)
    )
    parallel_siblings = sum(
        metric["axis_angle_deg"] <= 7.5
        and 4.0 < metric["transverse_offset_mm"] <= max(25.0, 0.08 * diagonal)
        and metric["shorter_span_overlap"] >= 0.45
        for metric in metrics
    )
    return [
        float(len(base)),
        min(metric["axis_angle_deg"] for metric in metrics) / 90.0,
        min(metric["transverse_offset_mm"] for metric in metrics) / diagonal,
        min(endpoint_gaps) / diagonal,
        max(metric["shorter_span_overlap"] for metric in metrics),
        float(orthogonal_junctions),
        float(parallel_siblings),
    ]


def _select_union(base, candidates, probabilities, qualities, threshold, duplicate_tolerance):
    residual = _ranked_selection(
        candidates,
        probabilities,
        qualities,
        probability_threshold=threshold,
        cylindrical_offset_cap_mm=1.0e9,
        duplicate_offset_tolerance_mm=duplicate_tolerance,
    )
    selected = [dict(item) for item in base]
    for candidate in residual:
        if any(_duplicate(candidate, retained, offset_tolerance_mm=duplicate_tolerance) for retained in selected):
            continue
        selected.append(dict(candidate))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-kind", choices=("line_objects", "raw_candidates"), default="line_objects")
    parser.add_argument("--model-kind", choices=("extra_trees", "logistic"), default="extra_trees")
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    split = _read(args.split_manifest.resolve())
    mesh_report = _read(args.mesh_root.resolve() / "mesh_crossfit_metrics.json")
    if len({sealed["corpus_freeze_id"], split["corpus_freeze_id"], mesh_report["corpus_freeze_id"]}) != 1:
        raise ValueError("freeze ID mismatch")
    cases = {str(case["case_token"]): case for case in sealed["cases"]}
    tokens = sorted(cases)
    gold = {token: list(cases[token]["manual_bends"]) for token in tokens}
    family = {token: str(cases[token]["part_key"]) for token in tokens}
    fold_tokens = {
        int(fold["fold"]): sorted(token for token in tokens if family[token] in set(fold["locked_part_families"]))
        for fold in split["folds"]
    }
    base = _base_predictions(args.mesh_root.resolve(), tokens)
    payloads = {token: _read(args.candidate_root.resolve() / f"{token}.json") for token in tokens}
    key = "line_objects" if args.candidate_kind == "line_objects" else "candidates"
    candidates = {}
    features = {}
    labels = {}
    qualities = {}
    residual_gold = {}
    for token in tokens:
        base_match = match_lines(gold[token], base[token], Thresholds())
        unmatched = set(base_match["unmatched_gold_ids"])
        residual_gold[token] = [item for item in gold[token] if str(item["bend_id"]) in unmatched]
        diagonal = max(float(np.linalg.norm(payloads[token].get("scan_bbox_mm") or [1, 1, 1])), 1.0)
        case_candidates = [
            dict(candidate)
            for candidate in payloads[token].get(key) or []
            if not any(_duplicate(candidate, retained, offset_tolerance_mm=4.0) for retained in base[token])
        ]
        candidates[token] = case_candidates
        features[token] = np.asarray(
            [
                _features(candidate, payloads[token]) + _relation_features(candidate, base[token], diagonal)
                for candidate in case_candidates
            ],
            dtype=np.float64,
        )
        case_qualities = np.asarray(
            [_match_quality(candidate, residual_gold[token], Thresholds()) for candidate in case_candidates],
            dtype=np.float64,
        )
        qualities[token] = case_qualities
        labels[token] = (case_qualities > 0).astype(np.int64)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    fold_reports = []
    fold_ids = sorted(fold_tokens)
    for outer in fold_ids:
        held_out = fold_tokens[outer]
        training_folds = [fold for fold in fold_ids if fold != outer]
        oof = {}
        for inner in training_folds:
            validation = fold_tokens[inner]
            training = [token for fold in training_folds if fold != inner for token in fold_tokens[fold]]
            classifier, regressor = _fit(
                np.vstack([features[token] for token in training]),
                np.concatenate([labels[token] for token in training]),
                np.concatenate([qualities[token] for token in training]),
                model_kind=args.model_kind,
            )
            for token in validation:
                oof[token] = _predict(classifier, regressor, features[token])
        scored = []
        for threshold in THRESHOLDS:
            for duplicate in DUPLICATE_TOLERANCES:
                trial = {
                    token: _select_union(base[token], candidates[token], *oof[token], threshold, duplicate)
                    for token in oof
                }
                scored.append((threshold, duplicate, _aggregate(trial, gold)))
        threshold, duplicate, inner_metrics = max(
            scored,
            key=lambda item: (
                item[2]["line_f1"], item[2]["exact_scans"], item[2]["recall"], -item[2]["false_positive"], item[0], -item[1]
            ),
        )
        training = [token for fold in training_folds for token in fold_tokens[fold]]
        classifier, regressor = _fit(
            np.vstack([features[token] for token in training]),
            np.concatenate([labels[token] for token in training]),
            np.concatenate([qualities[token] for token in training]),
            model_kind=args.model_kind,
        )
        held_predictions = {}
        for token in held_out:
            probability, quality = _predict(classifier, regressor, features[token])
            selected = _select_union(base[token], candidates[token], probability, quality, threshold, duplicate)
            for index, candidate in enumerate(selected, start=1):
                candidate["bend_id"] = f"F1R{index}"
            predictions[token] = selected
            held_predictions[token] = selected
            case_dir = output_dir / "cases" / token
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction_owned_regions.json").write_text(
                json.dumps({"owned_bend_regions": selected}, indent=2) + "\n", encoding="utf-8"
            )
        held_metrics = _aggregate(held_predictions, gold)
        fold_reports.append(
            {
                "outer_fold": outer,
                "held_out_case_tokens": held_out,
                "selected_probability_threshold": threshold,
                "selected_duplicate_tolerance_mm": duplicate,
                "inner_validation_metrics": {key: value for key, value in inner_metrics.items() if key != "cases"},
                "held_out_metrics": {key: value for key, value in held_metrics.items() if key != "cases"},
            }
        )
        print(f"fold={outer} threshold={threshold:.2f} heldout_f1={held_metrics['line_f1']:.3f}", flush=True)

    metrics = _aggregate(predictions, gold)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_mesh_plus_residual_grouped_crossfit",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "candidate_kind": args.candidate_kind,
        "model_kind": args.model_kind,
        "evaluation_mode": "five_fold_outer_part_group_crossfit_with_inner_fold_selection",
        "leakage_contract": {
            "same_scan_labels_in_training": False,
            "same_part_labels_in_training": False,
            "outer_fold_labels_used_for_hyperparameter_selection": False,
            "candidate_generation_uses_manual_truth": False,
            "part_identifier_or_expected_count_used_as_feature": False,
            "mesh_base_is_grouped_crossfit": True,
        },
        "metrics": metrics,
        "folds": fold_reports,
    }
    report_path = output_dir / "residual_crossfit_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {key: value for key, value in metrics.items() if key != "cases"}, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
