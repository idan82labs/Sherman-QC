#!/usr/bin/env python3
"""Fit the F1 selector on all frozen gold27 labels and report training fit.

This artifact is useful for deployment prototyping and candidate-ceiling
diagnosis. It is explicitly not a blind, holdout, or generalization score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds  # noqa: E402
from scripts.run_f1_gold27_grouped_selector import _aggregate  # noqa: E402
from scripts.run_f1_grouped_holdout_selector import (  # noqa: E402
    FEATURE_NAMES,
    _features,
    _match_quality,
    _ranked_selection,
)


DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/train_all_selector_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    tokens = [str(case["case_token"]) for case in sealed["cases"]]
    gold = {str(case["case_token"]): list(case["manual_bends"]) for case in sealed["cases"]}
    family = {str(case["case_token"]): str(case["part_key"]) for case in sealed["cases"]}
    payloads = {token: _read(args.candidate_root.resolve() / f"{token}.json") for token in tokens}
    candidates = {token: [dict(item) for item in payloads[token].get("candidates") or []] for token in tokens}
    features = {
        token: np.asarray([_features(item, payloads[token]) for item in candidates[token]], dtype=np.float64)
        for token in tokens
    }
    qualities = {
        token: np.asarray([_match_quality(item, gold[token], Thresholds()) for item in candidates[token]], dtype=np.float64)
        for token in tokens
    }
    x = np.vstack([features[token] for token in tokens])
    quality = np.concatenate([qualities[token] for token in tokens])
    y = (quality > 0).astype(np.int64)
    classifier = ExtraTreesClassifier(
        n_estimators=256,
        class_weight="balanced",
        random_state=17,
        n_jobs=1,
        max_features=None,
        min_samples_leaf=1,
    ).fit(x, y)
    regressor = ExtraTreesRegressor(
        n_estimators=256,
        random_state=17,
        n_jobs=1,
        max_features=None,
        min_samples_leaf=1,
    ).fit(x, quality)
    raw_predictions = {
        token: (
            classifier.predict_proba(features[token])[:, 1],
            regressor.predict(features[token]),
        )
        for token in tokens
    }
    trials = []
    for duplicate_tolerance in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        predictions = {
            token: _ranked_selection(
                candidates[token],
                raw_predictions[token][0],
                raw_predictions[token][1],
                probability_threshold=0.5,
                cylindrical_offset_cap_mm=1.0e9,
                duplicate_offset_tolerance_mm=duplicate_tolerance,
            )
            for token in tokens
        }
        trials.append((duplicate_tolerance, predictions, _aggregate(predictions, gold)))
    duplicate_tolerance, predictions, metrics = max(
        trials,
        key=lambda item: (item[2]["line_f1"], item[2]["exact_scans"], -item[2]["false_positive"], -item[0]),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for token, selected in predictions.items():
        for index, candidate in enumerate(selected, start=1):
            candidate["bend_id"] = f"F1T{index}"
        case_dir = output_dir / "cases" / token
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "prediction_owned_regions.json").write_text(
            json.dumps({"owned_bend_regions": selected}, indent=2) + "\n", encoding="utf-8"
        )
    model_path = output_dir / "selector.joblib"
    joblib.dump(
        {
            "classifier": classifier,
            "quality_regressor": regressor,
            "feature_names": list(FEATURE_NAMES),
            "candidate_kind": "raw_candidates",
            "probability_threshold": 0.5,
            "duplicate_tolerance_mm": duplicate_tolerance,
            "corpus_freeze_id": sealed["corpus_freeze_id"],
        },
        model_path,
    )
    exact_families = sum(
        all(next(row for row in metrics["cases"] if row["case_token"] == token)["exact"] for token in tokens if family[token] == part)
        for part in sorted(set(family.values()))
    )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_train_all_fit",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "claim": "training_fit_only_not_generalization_evidence",
        "labels_used_for_model_fit": True,
        "labels_used_for_duplicate_tolerance_selection": True,
        "candidate_generation_uses_manual_truth": False,
        "source_point_serialization": False,
        "positive_candidate_count": int(np.count_nonzero(y)),
        "candidate_count": int(len(y)),
        "selected_duplicate_tolerance_mm": duplicate_tolerance,
        "metrics": metrics,
        "exact_part_families": exact_families,
        "total_part_families": len(set(family.values())),
        "model_path": str(model_path),
    }
    report_path = output_dir / "training_fit_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {key: value for key, value in metrics.items() if key != "cases"}, "exact_part_families": exact_families, "duplicate_tolerance_mm": duplicate_tolerance, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
