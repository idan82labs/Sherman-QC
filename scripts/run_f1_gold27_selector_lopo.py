#!/usr/bin/env python3
"""Run the broad F1 candidate selector leave-one-physical-part-out."""

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
from scripts.run_f1_grouped_holdout_selector import (  # noqa: E402
    _features,
    _fit,
    _match_quality,
    _predict,
    _ranked_selection,
)


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/selector_lopo_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-kind", choices=("line_objects", "raw_candidates"), default="line_objects")
    parser.add_argument("--model-kind", choices=("logistic", "extra_trees"), default="logistic")
    parser.add_argument("--probability-threshold", type=float, default=0.95)
    parser.add_argument("--duplicate-tolerance-mm", type=float, default=8.0)
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    cases = {str(case["case_token"]): case for case in sealed["cases"]}
    tokens = sorted(cases)
    gold = {token: list(cases[token]["manual_bends"]) for token in tokens}
    family = {token: str(cases[token]["part_key"]) for token in tokens}
    payloads = {token: _read(args.candidate_root.resolve() / f"{token}.json") for token in tokens}
    candidate_key = "line_objects" if args.candidate_kind == "line_objects" else "candidates"
    candidates = {token: [dict(item) for item in payloads[token].get(candidate_key) or []] for token in tokens}
    features = {
        token: np.asarray([_features(item, payloads[token]) for item in candidates[token]], dtype=np.float64)
        for token in tokens
    }
    qualities = {
        token: np.asarray([_match_quality(item, gold[token], Thresholds()) for item in candidates[token]], dtype=np.float64)
        for token in tokens
    }
    labels = {token: (qualities[token] > 0).astype(np.int64) for token in tokens}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    fold_reports = []
    for part in sorted(set(family.values())):
        held_out = [token for token in tokens if family[token] == part]
        training = [token for token in tokens if family[token] != part]
        classifier, regressor = _fit(
            np.vstack([features[token] for token in training]),
            np.concatenate([labels[token] for token in training]),
            np.concatenate([qualities[token] for token in training]),
            model_kind=args.model_kind,
        )
        held_predictions = {}
        for token in held_out:
            probabilities, predicted_qualities = _predict(classifier, regressor, features[token])
            selected = _ranked_selection(
                candidates[token],
                probabilities,
                predicted_qualities,
                probability_threshold=args.probability_threshold,
                cylindrical_offset_cap_mm=1.0e9,
                duplicate_offset_tolerance_mm=args.duplicate_tolerance_mm,
            )
            for index, candidate in enumerate(selected, start=1):
                candidate["bend_id"] = f"F1B{index}"
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
        "kind": "sherman_qc_f1_gold27_broad_selector_leave_one_part_out",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "evaluation_mode": "leave_one_physical_part_family_out_fixed_prior_config",
        "candidate_kind": args.candidate_kind,
        "model_kind": args.model_kind,
        "probability_threshold": args.probability_threshold,
        "duplicate_tolerance_mm": args.duplicate_tolerance_mm,
        "leakage_contract": {
            "same_scan_labels_in_training": False,
            "same_part_labels_in_training": False,
            "held_out_labels_used_for_hyperparameter_selection": False,
            "candidate_generation_uses_manual_truth": False,
            "part_identifier_or_expected_count_used_as_feature": False,
            "fixed_config_has_prior_corpus_development_exposure": True,
        },
        "metrics": metrics,
        "exact_part_families": exact_families,
        "total_part_families": len(set(family.values())),
        "folds": fold_reports,
    }
    report_path = output_dir / "selector_lopo_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {key: value for key, value in metrics.items() if key != "cases"}, "exact_part_families": exact_families, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
