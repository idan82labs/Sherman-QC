#!/usr/bin/env python3
"""Score F1 line predictions against the consolidated bend-line gold catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    DEFAULT_CASES as LEGACY_13_DEFAULT_CASES,
    Thresholds,
    match_lines,
)


DEFAULT_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
DEFAULT_OUTPUT = ROOT / "output/experiments/f1-bend-line-gold-catalog-evaluation.json"
SHAKED_12_FROZEN_ROOT = (
    ROOT / "output/enterprise_benchmark/shaked-count-backed-12-blind-baseline-20260717"
)
XV_DEVELOPMENT_MANIFEST = (
    ROOT
    / "output/research/f1-new-shaked-pairs-20260730-v1/mesh_topology_profile_v1/prediction_manifest.json"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("owned_bend_regions", "manual_bends", "bends", "predictions"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    raise ValueError("prediction payload has no supported bend-line array")


def _selected_cases(catalog: dict[str, Any], tier: str) -> list[dict[str, Any]]:
    cases = list(catalog.get("cases") or [])
    if tier == "promotion":
        return [case for case in cases if case.get("promotion_eligible")]
    if tier == "development":
        return [case for case in cases if not case.get("promotion_eligible")]
    return cases


def existing_development_prediction_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for case_root in sorted((SHAKED_12_FROZEN_ROOT / "isolated_cases").glob("[0-9][0-9]_*/")):
        matches = sorted(case_root.glob("result/**/decomposition_result.json"))
        if len(matches) != 1:
            raise ValueError(f"{case_root.name}: expected one frozen prediction, found {len(matches)}")
        mapping[f"shaked12/{case_root.name}"] = str(matches[0].resolve())
    xv = _read(XV_DEVELOPMENT_MANIFEST)
    for token, path in sorted((xv.get("prediction_paths") or {}).items()):
        mapping[f"shaked_xv/{token}"] = str(Path(path).resolve())
    return mapping


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = sum(result["true_positive"] for result in results)
    false_positive = sum(result["false_positive"] for result in results)
    false_negative = sum(result["false_negative"] for result in results)
    gold_count = true_positive + false_negative
    prediction_count = true_positive + false_positive
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "case_count": len(results),
        "unique_part_count": len({result["part_id"] for result in results}),
        "gold_bend_count": gold_count,
        "prediction_count": prediction_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(result["exact"] for result in results),
        "exact_case_rate": round(sum(result["exact"] for result in results) / len(results), 6),
    }


def evaluate(
    catalog: dict[str, Any],
    prediction_map: dict[str, str],
    *,
    tier: str,
    thresholds: Thresholds,
) -> dict[str, Any]:
    cases = _selected_cases(catalog, tier)
    expected_ids = {str(case["case_id"]) for case in cases}
    unexpected = sorted(set(prediction_map) - expected_ids)
    if unexpected:
        raise ValueError(f"prediction map contains cases outside tier {tier}: {unexpected}")
    missing = sorted(expected_ids - set(prediction_map))
    if missing:
        raise ValueError(f"prediction map is missing {len(missing)} cases: {missing}")

    results = []
    for case in cases:
        case_id = str(case["case_id"])
        raw_path = Path(prediction_map[case_id]).expanduser()
        prediction_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        prediction_path = prediction_path.resolve()
        if not prediction_path.is_file():
            raise FileNotFoundError(f"{case_id}: prediction missing: {prediction_path}")
        prediction_payload = _read(prediction_path)
        result = match_lines(case["bends"], _prediction_lines(prediction_payload), thresholds)
        result.update(
            {
                "case_id": case_id,
                "part_id": case["part_id"],
                "truth_tier": case["truth_tier"],
                "gold_path": case["source_labels_path"],
                "prediction_path": str(prediction_path),
            }
        )
        results.append(result)

    truth_tiers = sorted({str(result["truth_tier"]) for result in results})
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_bend_line_gold_catalog_accuracy",
        "catalog_snapshot_sha256": catalog["source_snapshot_sha256"],
        "evaluation_tier": tier,
        "thresholds": {
            "axis_angle_deg": thresholds.angle_deg,
            "transverse_offset_mm": thresholds.transverse_offset_mm,
            "gold_span_overlap": thresholds.gold_span_overlap,
            "prediction_span_overlap": thresholds.prediction_span_overlap,
            "matching": "one_to_one_minimum_cost",
        },
        "metrics": _metrics(results),
        "metrics_by_truth_tier": {
            truth_tier: _metrics(
                [result for result in results if result["truth_tier"] == truth_tier]
            )
            for truth_tier in truth_tiers
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--prediction-map", type=Path)
    parser.add_argument(
        "--legacy-13-defaults",
        action="store_true",
        help="Use the frozen current prediction paths for the original 13-case promotion set",
    )
    parser.add_argument(
        "--existing-development-defaults",
        action="store_true",
        help="Use the frozen Shaked-12 baseline plus the existing X/V development predictions",
    )
    parser.add_argument("--tier", choices=("promotion", "development", "all"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angle-deg", type=float, default=5.0)
    parser.add_argument("--offset-mm", type=float, default=4.0)
    parser.add_argument("--gold-overlap", type=float, default=0.60)
    parser.add_argument("--prediction-overlap", type=float, default=0.60)
    args = parser.parse_args()

    modes = sum(bool(value) for value in (args.prediction_map, args.legacy_13_defaults, args.existing_development_defaults))
    if modes != 1:
        parser.error(
            "choose exactly one of --prediction-map, --legacy-13-defaults, "
            "or --existing-development-defaults"
        )
    prediction_map = (
        {
            f"corpus/{case_id}": str(path)
            for case_id, path in LEGACY_13_DEFAULT_CASES.items()
        }
        if args.legacy_13_defaults
        else existing_development_prediction_map()
        if args.existing_development_defaults
        else _read(args.prediction_map.resolve())
    )
    if args.legacy_13_defaults and args.tier != "promotion":
        parser.error("--legacy-13-defaults requires --tier promotion")
    if args.existing_development_defaults and args.tier != "development":
        parser.error("--existing-development-defaults requires --tier development")
    report = evaluate(
        _read(args.catalog.resolve()),
        prediction_map,
        tier=args.tier,
        thresholds=Thresholds(
            args.angle_deg,
            args.offset_mm,
            args.gold_overlap,
            args.prediction_overlap,
        ),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
