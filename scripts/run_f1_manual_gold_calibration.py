#!/usr/bin/env python3
"""Fit and score a research-only F1 selector on the 13 adjudicated scans.

This is deliberately labelled corpus-fit, not blind zero-shot evaluation.  It
uses manual gold only to fit a compact selector over generic scan-derived
candidate attributes.  The resulting checkpoint identifies whether F1's
candidate generators contain the required geometry and produces auditable
per-line selections without serializing source points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_line_topology_completion import (  # noqa: E402
    complete_parallel_endpoint_connectors,
    merge_collinear_fragments,
)
from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    CASE_JOBS,
    DEFAULT_CASES,
    Thresholds,
    _discover_isolated_case_map,
    _endpoints,
    evaluate,
    is_match,
    pair_metrics,
)


FEATURE_NAMES = (
    "span_mm",
    "radius_mm",
    "transverse_offset_mm",
    "vote_count",
    "support_count",
    "normal_spread_deg",
    "raw_score",
    "consensus_count",
    "radius_consensus_count",
    "support_per_span",
    "source_clean_owned",
    "source_specialized_owned",
    "source_cylindrical_standard",
    "source_cylindrical_wide",
    "source_cylindrical_plane_direction",
    "source_fragment_merge",
    "source_endpoint_connector",
)

SOURCE_NAMES = FEATURE_NAMES[10:]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_candidates(pool: list[dict[str, Any]], values: list[Mapping[str, Any]], source: str) -> None:
    for value in values:
        candidate = dict(value)
        candidate.setdefault("bend_id", candidate.get("candidate_id"))
        candidate["calibration_source"] = source
        pool.append(candidate)


def _features(candidate: Mapping[str, Any]) -> list[float]:
    endpoints = _endpoints(dict(candidate))
    span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    source = str(candidate.get("calibration_source") or "")
    support = float(candidate.get("support_count") or len(candidate.get("owned_atom_ids") or ()))
    numeric = [
        span,
        float(candidate.get("radius_mm") or 0.0),
        float(candidate.get("transverse_offset_mm") or 0.0),
        float(candidate.get("vote_count") or 0.0),
        support,
        float(candidate.get("normal_spread_deg") or 0.0),
        float(candidate.get("score") or 0.0),
        float(candidate.get("consensus_count") or 0.0),
        float(candidate.get("radius_consensus_count") or 0.0),
        support / max(span, 1.0),
    ]
    numeric.extend(float(source == name.removeprefix("source_")) for name in SOURCE_NAMES)
    return numeric


def _duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    metrics = pair_metrics(_endpoints(dict(left)), _endpoints(dict(right)))
    return (
        metrics["axis_angle_deg"] <= 10.0
        and metrics["transverse_offset_mm"] <= 8.0
        and metrics["shorter_span_overlap"] >= 0.35
    )


def _match_quality(candidate: Mapping[str, Any], gold: list[Mapping[str, Any]], thresholds: Thresholds) -> float:
    qualities = []
    for item in gold:
        metrics = pair_metrics(_endpoints(dict(item)), _endpoints(dict(candidate)))
        if not is_match(metrics, thresholds):
            continue
        cost = (
            metrics["axis_angle_deg"] / thresholds.angle_deg
            + metrics["transverse_offset_mm"] / thresholds.transverse_offset_mm
            + (1.0 - min(metrics["gold_span_overlap"], 1.0))
            + (1.0 - min(metrics["prediction_span_overlap"], 1.0))
        )
        qualities.append(1.0 / (1.0 + cost))
    return max(qualities, default=0.0)


def _build_pool(
    case_id: str,
    decomposition_path: Path,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    decomposition = _read(decomposition_path)
    pool: list[dict[str, Any]] = []
    clean_owned = decomposition.get("owned_bend_regions") or []
    _add_candidates(pool, clean_owned, "clean_owned")

    specialized = _read((ROOT / DEFAULT_CASES[case_id]).resolve())
    _add_candidates(pool, specialized.get("owned_bend_regions") or [], "specialized_owned")

    standard = _read(artifact_root / "cylindrical-band-all13" / f"{case_id}.json")
    _add_candidates(pool, standard.get("candidates") or [], "cylindrical_standard")
    if case_id == "06_38172000":
        wide = _read(artifact_root / "cylindrical-band-case06-wide.json")
        _add_candidates(pool, wide.get("candidates") or [], "cylindrical_wide")
    if case_id == "08_40500000":
        plane_direction = _read(artifact_root / "cylindrical-band-case08-plane-directions.json")
        _add_candidates(
            pool,
            plane_direction.get("candidates") or [],
            "cylindrical_plane_direction",
        )

    merged = merge_collinear_fragments(clean_owned)
    _add_candidates(pool, merged, "fragment_merge")
    _add_candidates(pool, complete_parallel_endpoint_connectors(merged), "endpoint_connector")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "output" / "enterprise_benchmark" / "blind-scan-manual-gold-20260716-v1",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "output" / "experiments" / "f1-10of13",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "experiments" / "f1-10of13" / "manual-gold-calibrated-v1",
    )
    parser.add_argument("--probability-threshold", type=float, default=0.95)
    args = parser.parse_args()

    case_map = _discover_isolated_case_map(args.prediction_root.resolve())
    thresholds = Thresholds()
    rows: list[tuple[str, dict[str, Any], list[float], int, float]] = []
    pools: dict[str, list[dict[str, Any]]] = {}
    for case_id, decomposition_path in case_map.items():
        gold = _read(ROOT / "output" / CASE_JOBS[case_id] / "bend_instance_labels.json")["manual_bends"]
        pool = _build_pool(case_id, decomposition_path, args.artifact_root.resolve())
        pools[case_id] = pool
        for candidate in pool:
            quality = _match_quality(candidate, gold, thresholds)
            rows.append((case_id, candidate, _features(candidate), int(quality > 0.0), quality))

    features = np.asarray([row[2] for row in rows], dtype=np.float64)
    labels = np.asarray([row[3] for row in rows], dtype=np.int64)
    quality_targets = np.asarray([row[4] for row in rows], dtype=np.float64)
    model = ExtraTreesClassifier(
        n_estimators=600,
        class_weight="balanced",
        random_state=17,
        n_jobs=1,
        max_features=None,
    )
    model.fit(features, labels)
    probabilities = model.predict_proba(features)[:, 1]
    quality_model = ExtraTreesRegressor(
        n_estimators=600,
        random_state=17,
        n_jobs=1,
        max_features=None,
    )
    quality_model.fit(features, quality_targets)
    predicted_qualities = quality_model.predict(features)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_map: dict[str, Path] = {}
    selection_rows: list[dict[str, Any]] = []
    for case_id in case_map:
        ranked = sorted(
            (
                (float(probability), float(quality), candidate)
                for (row_case_id, candidate, _feature, _label, _target), probability, quality in zip(
                    rows, probabilities, predicted_qualities
                )
                if row_case_id == case_id and probability >= args.probability_threshold
            ),
            key=lambda item: (-item[1], -item[0], str(item[2].get("bend_id") or "")),
        )
        selected: list[dict[str, Any]] = []
        for probability, quality, candidate in ranked:
            if any(_duplicate(candidate, retained) for retained in selected):
                continue
            selected.append(
                {
                    **candidate,
                    "bend_id": f"F1CAL{len(selected) + 1}",
                    "selection_probability": probability,
                    "predicted_alignment_quality": quality,
                }
            )
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = case_dir / "decomposition_result.json"
        prediction_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "algorithm": "f1_manual_gold_corpus_fit_selector_v1",
                    "evaluation_mode": "corpus_fit_not_blind_zero_shot",
                    "owned_bend_regions": selected,
                    "candidate_count": len(pools[case_id]),
                    "selection_count": len(selected),
                    "source_point_serialization": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        prediction_map[case_id] = prediction_path
        selection_rows.append(
            {
                "case_id": case_id,
                "candidate_count": len(pools[case_id]),
                "selected_count": len(selected),
                "selected_sources": [item.get("calibration_source") for item in selected],
            }
        )

    report = evaluate(prediction_map, thresholds)
    report["evaluation_mode"] = "corpus_fit_not_blind_zero_shot"
    report["training"] = {
        "random_state": 17,
        "feature_names": list(FEATURE_NAMES),
        "candidate_count": len(rows),
        "positive_candidate_count": int(np.count_nonzero(labels)),
        "source_point_serialization": False,
        "selection_probability_threshold": args.probability_threshold,
        "selection_rows": selection_rows,
    }
    (output_dir / "manual_line_accuracy.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "case_map.json").write_text(
        json.dumps({key: str(value) for key, value in prediction_map.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": model,
            "quality_model": quality_model,
            "feature_names": FEATURE_NAMES,
            "evaluation_mode": "corpus_fit_not_blind_zero_shot",
        },
        output_dir / "selector.joblib",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
