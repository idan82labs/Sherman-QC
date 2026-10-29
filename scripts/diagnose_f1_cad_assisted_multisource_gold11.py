#!/usr/bin/env python3
"""Fixed scan + legacy CAD + native STEP topology fusion on matched Gold11."""

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
    return bool(
        metrics["axis_angle_deg"] <= 5.0
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.60
    )


def _a0(row: dict[str, Any]) -> dict[str, Any]:
    line = _line(row)
    return {
        "bend_id": str(row["bend_id"]),
        "line_start": line[0].tolist(),
        "line_end": line[1].tolist(),
        "fusion_source": "scan_only_a0",
    }


def _append_unique(target: list[dict[str, Any]], rows: list[dict[str, Any]], source: str) -> None:
    for row in rows:
        if any(_duplicate(row, existing) for existing in target):
            continue
        added = dict(row)
        added["bend_id"] = source.upper() + "_" + str(row["bend_id"])
        added["fusion_source"] = source
        target.append(added)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = score_localized_cases(rows)
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(str(row["family_id"]), []).append(row)
    recalls = {
        family: sum(row["true_positive"] for row in cases)
        / sum(row["gold_count"] for row in cases)
        for family, cases in families.items()
    }
    result["family_recalls"] = recalls
    result["minimum_family_recall"] = min(recalls.values())
    return result


def _predictions(root: Path, case_id: str) -> list[dict[str, Any]]:
    payload = json.loads(
        (root / case_id / "semantic_result.json").read_text(encoding="utf-8")
    )
    return [dict(row) for row in payload["predictions"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--a0-pool-manifest", type=Path, required=True)
    parser.add_argument("--a0-root", type=Path, required=True)
    parser.add_argument("--legacy-cad-root", type=Path, required=True)
    parser.add_argument("--topology-root", type=Path, required=True)
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
    variants = {name: [] for name in ("scan_only", "incumbent_union", "triple_union")}
    case_records = []
    for entry in slice_payload["entries"]:
        scan_sha = str(entry["scan_sha256"])
        token = token_by_sha.get(scan_sha)
        if token is None:
            continue
        case_id = str(entry["case_token"])
        gold = gold_by_sha[scan_sha]
        a0_payload = json.loads(
            (args.a0_root / token / "prediction_owned_regions.json").read_text(encoding="utf-8")
        )
        scan_rows = [_a0(row) for row in a0_payload["owned_bend_regions"]]
        legacy = _predictions(args.legacy_cad_root, case_id)
        topology = _predictions(args.topology_root, case_id)
        incumbent = list(scan_rows)
        _append_unique(incumbent, legacy, "legacy_cad")
        triple = list(incumbent)
        _append_unique(triple, topology, "step_topology")
        rows_by_variant = {
            "scan_only": scan_rows,
            "incumbent_union": incumbent,
            "triple_union": triple,
        }
        case_record = {"case_id": case_id, "family_id": str(gold["part_id"])}
        for name, predictions in rows_by_variant.items():
            score = match_lines(gold["bends"], predictions, Thresholds())
            score.update({"case_id": case_id, "family_id": str(gold["part_id"])})
            variants[name].append(score)
            case_record[name] = {
                "prediction_count": score["prediction_count"],
                "true_positive": score["true_positive"],
                "false_positive": score["false_positive"],
                "false_negative": score["false_negative"],
            }
        case_records.append(case_record)
    if len(case_records) != 11:
        raise ValueError(f"expected 11 matched cases, got {len(case_records)}")
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_assisted_multisource_gold11_diagnostic",
        "label_free_fusion": True,
        "fitted_parameters": False,
        "metrics": {name: _aggregate(rows) for name, rows in variants.items()},
        "cases": case_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                }
                for name, values in report["metrics"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
