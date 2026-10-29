#!/usr/bin/env python3
"""Evaluate the F1 scan-mesh bend lane with grouped five-fold cross-fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_gold27_grouped_selector import _aggregate
from scripts.run_shaked_selector_evaluation import (
    _complete_parallel_siblings,
    _complete_scan_supported_short_seeds,
    _complete_short_mesh,
    _complete_surface_wedges,
    _fit_mesh_rows,
    _mesh_case_features,
    _predict_mesh_rows,
    _prune_parallel_fragments,
    _prune_surface_graph_outlines,
    _prune_weak_parallel_outlines,
    _select,
    _snap_mesh_consensus,
    complete_dominant_fold_lane,
)
from scripts.run_f1_grouped_holdout_selector import _match_quality
from scripts.run_f1_grouped_holdout_selector import _fit, _predict
from scripts.evaluate_f1_against_manual_gold import Thresholds


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_SPLIT = ROOT / "output/research/f1-gold27-20260803/freeze/grouped_five_fold_split.json"
DEFAULT_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/mesh_crossfit_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_select(prediction, config):
    return _select(prediction, config)


def _fit_model(rows, tokens, model_kind):
    if model_kind == "logistic":
        return _fit_mesh_rows(rows, tokens)
    flattened = [row for token in tokens for row in rows[token]]
    return _fit(
        np.asarray([row[1] for row in flattened], dtype=np.float64),
        np.asarray([row[2] for row in flattened], dtype=np.int64),
        np.asarray([row[3] for row in flattened], dtype=np.float64),
        model_kind="extra_trees",
    )


def _predict_model(model, case_rows, model_kind):
    if model_kind == "logistic":
        return _predict_mesh_rows(model, case_rows)
    candidates = [row[0] for row in case_rows]
    probabilities, qualities = _predict(
        model[0], model[1], np.asarray([row[1] for row in case_rows], dtype=np.float64)
    )
    return candidates, probabilities, qualities


def _apply_stack(selected, payload, config):
    line_objects = payload.get("line_objects") or []
    candidates = payload.get("candidates") or []
    selected = _complete_short_mesh(selected, line_objects, config)["selected"]
    selected = _snap_mesh_consensus(selected, line_objects, config)["selected"]
    selected = _complete_surface_wedges(selected, line_objects, config)["selected"]
    lane = complete_dominant_fold_lane(line_objects)
    if lane["eligible"]:
        selected = lane["selected"]
    selected = _complete_parallel_siblings(selected, candidates)["selected"]
    selected = _prune_parallel_fragments(selected)["selected"]
    selected = _complete_scan_supported_short_seeds(selected, line_objects, candidates)["selected"]
    selected = _prune_weak_parallel_outlines(selected)["selected"]
    selected = _prune_surface_graph_outlines(selected)["selected"]
    return [dict(candidate) for candidate in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-kind", choices=("logistic", "extra_trees"), default="logistic")
    parser.add_argument(
        "--domain-specific",
        action="store_true",
        help="Fit separate models for geometry-observable compact and large surface meshes.",
    )
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    split = _read(args.split_manifest.resolve())
    if sealed["corpus_freeze_id"] != split["corpus_freeze_id"]:
        raise ValueError("freeze ID mismatch")
    cases = {str(case["case_token"]): case for case in sealed["cases"]}
    gold = {token: list(case["manual_bends"]) for token, case in cases.items()}
    family = {token: str(case["part_key"]) for token, case in cases.items()}
    fold_tokens = {
        int(fold["fold"]): sorted(
            token for token in cases if family[token] in set(fold["locked_part_families"])
        )
        for fold in split["folds"]
    }
    payloads = {token: _read(args.candidate_root.resolve() / f"{token}.json") for token in cases}
    domains = {
        token: (
            "compact_surface_mesh"
            if float(payloads[token].get("generator_diagnostics", {}).get("scan_mesh_adjacency", {}).get("mesh_area") or 0.0)
            <= 50_000.0
            else "large_surface_mesh"
        )
        for token in cases
    }
    rows = {}
    for token, payload in payloads.items():
        candidates = [
            dict(item)
            for item in payload.get("candidates") or []
            if item.get("candidate_source") == "scan_mesh_adjacency"
        ]
        features = _mesh_case_features(candidates, payload) if candidates else []
        rows[token] = [
            (candidate, feature, int((quality := _match_quality(candidate, gold[token], Thresholds())) > 0), quality)
            for candidate, feature in zip(candidates, features)
        ]

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
            training = [
                token for fold in training_folds if fold != inner for token in fold_tokens[fold] if rows[token]
            ]
            for domain in sorted(set(domains.values())):
                domain_training = [
                    token for token in training if not args.domain_specific or domains[token] == domain
                ]
                model = _fit_model(rows, domain_training, args.model_kind)
                for token in validation:
                    if args.domain_specific and domains[token] != domain:
                        continue
                    oof[token] = _predict_model(model, rows[token], args.model_kind) if rows[token] else ([], [], [])
                if not args.domain_specific:
                    break

        scored = []
        for threshold in np.arange(0.05, 0.951, 0.05):
            for gap_cap in (6.0, 8.0, 10.0, 12.0, 1.0e9):
                config = {
                    "model_kind": f"{args.model_kind}_mesh_adjacency",
                    "logistic_c": 0.20,
                    "probability_threshold": float(round(threshold, 8)),
                    "cylindrical_offset_cap_mm": 1.0e9,
                    "duplicate_offset_tolerance_mm": 4.0,
                    "two_sided_min_support_gate": 0.0,
                    "mesh_max_surface_gap_mm": gap_cap,
                }
                candidate_predictions = {token: _base_select(oof[token], config) for token in oof}
                scored.append((config, _aggregate(candidate_predictions, gold)))
        config, inner_base = max(
            scored,
            key=lambda item: (
                item[1]["line_f1"], item[1]["exact_scans"], -item[1]["false_positive"], item[0]["probability_threshold"]
            ),
        )

        base_oof = {token: _base_select(oof[token], config) for token in oof}
        completion_grid = [
            {"short_mesh_max_span_mm": 0.0, "short_mesh_min_normal_turn_deg": 999.0, "short_mesh_min_finite_support_fraction": 1.0}
        ] + [
            {
                "short_mesh_max_span_mm": span,
                "short_mesh_min_normal_turn_deg": turn,
                "short_mesh_min_finite_support_fraction": support,
            }
            for span in (12.0, 16.0, 20.0, 25.0)
            for turn in (60.0, 70.0, 80.0)
            for support in (0.8, 0.9)
        ]
        completion_scores = []
        for completion in completion_grid:
            trial = {**config, **completion}
            candidate_predictions = {
                token: _snap_mesh_consensus(
                    _complete_short_mesh(base_oof[token], payloads[token].get("line_objects") or [], trial)["selected"],
                    payloads[token].get("line_objects") or [],
                    trial,
                )["selected"]
                for token in oof
            }
            completion_scores.append((trial, _aggregate(candidate_predictions, gold)))
        config, inner_completed = max(
            completion_scores,
            key=lambda item: (
                item[1]["line_f1"], item[1]["exact_scans"], -item[1]["false_positive"], -item[0]["short_mesh_max_span_mm"]
            ),
        )

        training = [token for fold in training_folds for token in fold_tokens[fold] if rows[token]]
        models = {
            domain: _fit_model(
                rows,
                [token for token in training if not args.domain_specific or domains[token] == domain],
                args.model_kind,
            )
            for domain in sorted(set(domains.values()))
            if args.domain_specific
        }
        if not args.domain_specific:
            models = {"all": _fit_model(rows, training, args.model_kind)}
        held_predictions = {}
        held_base = {}
        for token in held_out:
            model = models[domains[token] if args.domain_specific else "all"]
            prediction = _predict_model(model, rows[token], args.model_kind) if rows[token] else ([], [], [])
            base = _base_select(prediction, config)
            held_base[token] = base
            selected = _apply_stack(base, payloads[token], config)
            for index, candidate in enumerate(selected, start=1):
                candidate["bend_id"] = f"F1M{index}"
            predictions[token] = selected
            held_predictions[token] = selected
            case_dir = output_dir / "cases" / token
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction_owned_regions.json").write_text(
                json.dumps({"owned_bend_regions": selected}, indent=2) + "\n", encoding="utf-8"
            )
        fold_reports.append(
            {
                "outer_fold": outer,
                "held_out_case_tokens": held_out,
                "selected_config": config,
                "inner_base_metrics": {key: value for key, value in inner_base.items() if key != "cases"},
                "inner_completed_metrics": {key: value for key, value in inner_completed.items() if key != "cases"},
                "held_out_base_metrics": {key: value for key, value in _aggregate(held_base, gold).items() if key != "cases"},
                "held_out_full_stack_metrics": {key: value for key, value in _aggregate(held_predictions, gold).items() if key != "cases"},
            }
        )
        print(f"fold={outer} heldout_f1={fold_reports[-1]['held_out_full_stack_metrics']['line_f1']:.3f}", flush=True)

    metrics = _aggregate(predictions, gold)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_mesh_grouped_crossfit",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "evaluation_mode": "five_fold_outer_part_group_crossfit_with_inner_fold_selection",
        "model_kind": args.model_kind,
        "domain_specific": bool(args.domain_specific),
        "domain_contract": (
            {"feature": "scan_mesh_area_mm2", "compact_max": 50000.0}
            if args.domain_specific
            else None
        ),
        "leakage_contract": {
            "same_scan_labels_in_training": False,
            "same_part_labels_in_training": False,
            "outer_fold_labels_used_for_hyperparameter_selection": False,
            "candidate_generation_uses_manual_truth": False,
            "part_identifier_or_expected_count_used_as_feature": False,
            "prior_corpus_development_exposure_of_fixed_postprocessors": True,
        },
        "metrics": metrics,
        "folds": fold_reports,
    }
    report_path = output_dir / "mesh_crossfit_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {key: value for key, value in metrics.items() if key != "cases"}, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
