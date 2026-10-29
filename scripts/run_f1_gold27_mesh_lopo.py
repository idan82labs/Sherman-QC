#!/usr/bin/env python3
"""Run the fixed F1 mesh profile in leave-one-physical-part-out mode."""

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

from scripts.evaluate_f1_against_manual_gold import Thresholds  # noqa: E402
from scripts.run_f1_gold27_grouped_selector import _aggregate  # noqa: E402
from scripts.run_f1_gold27_mesh_crossfit import (  # noqa: E402
    _apply_stack,
    _fit_model,
    _predict_model,
)
from scripts.run_f1_grouped_holdout_selector import _match_quality  # noqa: E402
from scripts.run_shaked_selector_evaluation import (  # noqa: E402
    _complete_short_mesh,
    _mesh_case_features,
    _select,
    _snap_mesh_consensus,
)


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/mesh_lopo_fixed_v1"
FIXED_CONFIG = {
    "model_kind": "logistic_mesh_adjacency",
    "logistic_c": 0.20,
    "probability_threshold": 0.15,
    "cylindrical_offset_cap_mm": 1.0e9,
    "duplicate_offset_tolerance_mm": 4.0,
    "two_sided_min_support_gate": 0.0,
    "mesh_max_surface_gap_mm": 6.0,
    "short_mesh_max_span_mm": 16.0,
    "short_mesh_min_normal_turn_deg": 60.0,
    "short_mesh_min_finite_support_fraction": 0.8,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-kind", choices=("logistic", "extra_trees"), default="logistic")
    parser.add_argument("--nested-tuning", action="store_true")
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    cases = {str(case["case_token"]): case for case in sealed["cases"]}
    tokens = sorted(cases)
    gold = {token: list(cases[token]["manual_bends"]) for token in tokens}
    family = {token: str(cases[token]["part_key"]) for token in tokens}
    payloads = {token: _read(args.candidate_root.resolve() / f"{token}.json") for token in tokens}
    rows = {}
    for token in tokens:
        candidates = [
            dict(item)
            for item in payloads[token].get("candidates") or []
            if item.get("candidate_source") == "scan_mesh_adjacency"
        ]
        features = _mesh_case_features(candidates, payloads[token]) if candidates else []
        rows[token] = [
            (candidate, feature, int((quality := _match_quality(candidate, gold[token], Thresholds())) > 0), quality)
            for candidate, feature in zip(candidates, features)
        ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    fold_reports = []
    all_parts = sorted(set(family.values()))
    for part in all_parts:
        held_out = [token for token in tokens if family[token] == part]
        training = [token for token in tokens if family[token] != part and rows[token]]
        config = dict(FIXED_CONFIG)
        inner_metrics = None
        if args.nested_tuning:
            oof = {}
            for inner_part in [value for value in all_parts if value != part]:
                inner_validation = [token for token in tokens if family[token] == inner_part]
                inner_training = [
                    token
                    for token in tokens
                    if family[token] not in {part, inner_part} and rows[token]
                ]
                inner_model = _fit_model(rows, inner_training, args.model_kind)
                for token in inner_validation:
                    oof[token] = (
                        _predict_model(inner_model, rows[token], args.model_kind)
                        if rows[token]
                        else ([], [], [])
                    )
            scored = []
            for threshold in np.arange(0.05, 0.951, 0.05):
                for gap_cap in (6.0, 8.0, 10.0, 12.0, 1.0e9):
                    trial = {
                        **FIXED_CONFIG,
                        "probability_threshold": float(round(threshold, 8)),
                        "mesh_max_surface_gap_mm": gap_cap,
                    }
                    trial_predictions = {
                        token: _select(oof[token], trial)
                        for token in oof
                    }
                    scored.append((trial, _aggregate(trial_predictions, gold)))
            config, inner_base = max(
                scored,
                key=lambda item: (
                    item[1]["line_f1"],
                    item[1]["exact_scans"],
                    -item[1]["false_positive"],
                    item[0]["probability_threshold"],
                ),
            )
            base_oof = {token: _select(oof[token], config) for token in oof}
            completion_grid = [
                {
                    "short_mesh_max_span_mm": 0.0,
                    "short_mesh_min_normal_turn_deg": 999.0,
                    "short_mesh_min_finite_support_fraction": 1.0,
                }
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
                trial_predictions = {
                    token: _snap_mesh_consensus(
                        _complete_short_mesh(
                            base_oof[token],
                            payloads[token].get("line_objects") or [],
                            trial,
                        )["selected"],
                        payloads[token].get("line_objects") or [],
                        trial,
                    )["selected"]
                    for token in oof
                }
                completion_scores.append((trial, _aggregate(trial_predictions, gold)))
            config, inner_metrics = max(
                completion_scores,
                key=lambda item: (
                    item[1]["line_f1"],
                    item[1]["exact_scans"],
                    -item[1]["false_positive"],
                    -item[0]["short_mesh_max_span_mm"],
                ),
            )
        model = _fit_model(rows, training, args.model_kind)
        held_predictions = {}
        for token in held_out:
            prediction = _predict_model(model, rows[token], args.model_kind) if rows[token] else ([], [], [])
            selected = _select(prediction, config)
            selected = _apply_stack(selected, payloads[token], config)
            for index, candidate in enumerate(selected, start=1):
                candidate["bend_id"] = f"F1L{index}"
            predictions[token] = selected
            held_predictions[token] = selected
            case_dir = output_dir / "cases" / token
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction_owned_regions.json").write_text(
                json.dumps({"owned_bend_regions": selected}, indent=2) + "\n", encoding="utf-8"
            )
        metrics = _aggregate(held_predictions, gold)
        fold_reports.append(
            {
                "held_out_part_family": part,
                "held_out_case_tokens": held_out,
                "training_part_families": sorted({family[token] for token in training}),
                "selected_config": config,
                "inner_validation_metrics": (
                    {key: value for key, value in inner_metrics.items() if key != "cases"}
                    if inner_metrics is not None
                    else None
                ),
                "metrics": {key: value for key, value in metrics.items() if key != "cases"},
            }
        )
        print(f"part={part} heldout_f1={metrics['line_f1']:.3f}", flush=True)

    metrics = _aggregate(predictions, gold)
    exact_families = sum(
        all(next(row for row in metrics["cases"] if row["case_token"] == token)["exact"] for token in tokens if family[token] == part)
        for part in sorted(set(family.values()))
    )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_mesh_leave_one_part_out",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "evaluation_mode": (
            "nested_leave_one_physical_part_family_out"
            if args.nested_tuning
            else "leave_one_physical_part_family_out_fixed_prior_config"
        ),
        "model_kind": args.model_kind,
        "fixed_config": None if args.nested_tuning else FIXED_CONFIG,
        "leakage_contract": {
            "same_scan_labels_in_training": False,
            "same_part_labels_in_training": False,
            "held_out_labels_used_for_hyperparameter_selection": False,
            "candidate_generation_uses_manual_truth": False,
            "part_identifier_or_expected_count_used_as_feature": False,
            "prior_corpus_development_exposure_of_fixed_config_and_postprocessors": not args.nested_tuning,
            "fixed_postprocessor_family_has_prior_corpus_development_exposure": True,
        },
        "metrics": metrics,
        "exact_part_families": exact_families,
        "total_part_families": len(set(family.values())),
        "folds": fold_reports,
    }
    report_path = output_dir / "mesh_lopo_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {key: value for key, value in metrics.items() if key != "cases"}, "exact_part_families": exact_families, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
