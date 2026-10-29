#!/usr/bin/env python3
"""Matched Gold11 ablation of scan-only/CAD-guided fusion semantics.

No model or threshold is fitted. The fixed geometric matcher defines CAD
corroboration. Intersection retains A0 scan lines corroborated by a gated CAD
line. Union retains all A0 lines and adds only non-duplicate gated CAD lines.
"""

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

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases


def _line(row: dict[str, Any]) -> np.ndarray:
    endpoints = row.get("span_endpoints") or [row["line_start"], row["line_end"]]
    return np.asarray(endpoints, dtype=np.float64)


def _duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = pair_metrics(_line(left), _line(right))
    return (
        metrics["axis_angle_deg"] <= 5.0
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.60
    )


def _normalized_a0(row: dict[str, Any]) -> dict[str, Any]:
    endpoints = _line(row)
    return {
        "bend_id": str(row["bend_id"]),
        "line_start": endpoints[0].tolist(),
        "line_end": endpoints[1].tolist(),
        "fusion_source": "scan_only_a0",
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = score_localized_cases(rows)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family_id"]), []).append(row)
    recalls = {
        family: sum(row["true_positive"] for row in family_rows)
        / sum(row["gold_count"] for row in family_rows)
        for family, family_rows in by_family.items()
    }
    aggregate["minimum_family_recall"] = min(recalls.values())
    aggregate["family_recalls"] = recalls
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--a0-pool-manifest", type=Path, required=True)
    parser.add_argument("--a0-root", type=Path, required=True)
    parser.add_argument("--cad-prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    slice_payload = json.loads(args.slice.read_text(encoding="utf-8"))
    catalog = json.loads(args.gold_catalog.read_text(encoding="utf-8"))
    gold_by_sha = {str(row["scan_sha256"]): row for row in catalog["cases"]}
    pool = json.loads(args.a0_pool_manifest.read_text(encoding="utf-8"))
    token_by_sha = {
        str(row["source_scan_sha256"]): str(row["case_token"])
        for row in pool["cases"]
    }
    variants = {name: [] for name in ("scan_only_a0", "cad_only_gated", "intersection", "union")}
    case_records = []
    for entry in slice_payload["entries"]:
        scan_sha = str(entry["scan_sha256"])
        a0_token = token_by_sha.get(scan_sha)
        if a0_token is None:
            continue
        gold = gold_by_sha[scan_sha]
        a0_payload = json.loads(
            (args.a0_root / a0_token / "prediction_owned_regions.json").read_text(encoding="utf-8")
        )
        scan_rows = [_normalized_a0(row) for row in a0_payload["owned_bend_regions"]]
        cad_payload = json.loads(
            (args.cad_prediction_root / str(entry["case_token"]) / "semantic_result.json").read_text(encoding="utf-8")
        )
        cad_rows = [dict(row) for row in cad_payload["predictions"]]
        intersection = [row for row in scan_rows if any(_duplicate(row, cad) for cad in cad_rows)]
        union = list(scan_rows)
        for cad in cad_rows:
            if not any(_duplicate(cad, existing) for existing in union):
                added = dict(cad)
                added["bend_id"] = "CADF_" + str(added["bend_id"])
                added["fusion_source"] = "cad_guided_gated_rescue"
                union.append(added)
        rows_by_variant = {
            "scan_only_a0": scan_rows,
            "cad_only_gated": cad_rows,
            "intersection": intersection,
            "union": union,
        }
        scored_case = {"case_id": str(entry["case_token"]), "family_id": str(gold["part_id"])}
        for name, predictions in rows_by_variant.items():
            score = match_lines(gold["bends"], predictions, Thresholds())
            score.update({"case_id": str(entry["case_token"]), "family_id": str(gold["part_id"])})
            variants[name].append(score)
            scored_case[name] = {
                "prediction_count": score["prediction_count"],
                "true_positive": score["true_positive"],
                "false_positive": score["false_positive"],
                "false_negative": score["false_negative"],
                "exact": score["exact"],
            }
        case_records.append(scored_case)
    if len(case_records) != 11:
        raise ValueError(f"expected 11 matched A0 cases, got {len(case_records)}")
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_scan_fusion_gold11_diagnostic",
        "label_free_fusion": True,
        "fitted_parameters": False,
        "matching_constants": {
            "axis_angle_deg_max": 5.0,
            "transverse_offset_mm_max": 4.0,
            "shorter_span_overlap_min": 0.60,
        },
        "metrics": {name: _metrics(rows) for name, rows in variants.items()},
        "cases": case_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({name: {"f1": values["f1"], "precision": values["precision"], "recall": values["recall"]} for name, values in report["metrics"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
