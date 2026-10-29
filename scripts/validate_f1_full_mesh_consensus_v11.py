#!/usr/bin/env python3
"""Regression-check V10 and V11 on full-resolution development meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, build_feature_matrix, endpoints
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.select_f1_full_mesh_consensus_v10 import repair_full_mesh_representatives
from scripts.select_f1_full_mesh_consensus_v11 import prune_short_cylindrical_fragments
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


def _prediction(rows: list[dict], selected: list[int]) -> list[dict]:
    return [
        {
            "bend_id": str(index),
            "line_start": endpoints(rows[index])[0].tolist(),
            "line_end": endpoints(rows[index])[1].tolist(),
        }
        for index in selected
    ]


def _aggregate(cases: list[dict], prefix: str) -> dict:
    tp = sum(int(row[prefix]["true_positive"]) for row in cases)
    fp = sum(int(row[prefix]["false_positive"]) for row in cases)
    fn = sum(int(row[prefix]["false_negative"]) for row in cases)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(row[prefix]["exact"]) for row in cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v4 = joblib.load(args.v4_model)
    if v4.get("profile") != V4_PROFILE or tuple(v4.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")
    gold = {
        str(row["case_token"]): row
        for row in json.loads(args.gold_manifest.read_text(encoding="utf-8"))["cases"]
    }
    manifest = json.loads((args.pool_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for item in manifest["cases"]:
        token = str(item["case_id"])
        pool = json.loads((args.pool_dir / f"{token}.json").read_text(encoding="utf-8"))
        diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
        if not bool(diagnostics.get("available")) or bool(diagnostics.get("bounded_decimation_used")):
            continue
        rows = [dict(row) for row in pool["line_objects"]]
        features = build_feature_matrix(rows)
        core_scores = v4["core_estimator"].predict_proba(features)[:, 1]
        adjacency_scores = np.zeros(len(rows), dtype=float)
        adjacency_indices = [
            index for index, row in enumerate(rows)
            if row.get("representative_source") == ADJACENCY_SOURCE
        ]
        if adjacency_indices:
            adjacency_scores[adjacency_indices] = v4["adjacency_estimator"].predict_proba(
                features[adjacency_indices]
            )[:, 1]
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
        v10_selected, repairs = repair_full_mesh_representatives(rows, selected)
        v11_selected, prunes = prune_short_cylindrical_fragments(rows, v10_selected)
        cases.append(
            {
                "case_token": token,
                "v10": match_lines(gold[token]["manual_bends"], _prediction(rows, v10_selected), Thresholds()),
                "v11": match_lines(gold[token]["manual_bends"], _prediction(rows, v11_selected), Thresholds()),
                "representative_repair_count": len(repairs),
                "short_fragment_prune_count": len(prunes),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "f1_full_resolution_mesh_v11_development_regression",
        "case_count": len(cases),
        "v10_metrics": _aggregate(cases, "v10"),
        "v11_metrics": _aggregate(cases, "v11"),
        "true_positive_loss": _aggregate(cases, "v10")["true_positive"] - _aggregate(cases, "v11")["true_positive"],
        "total_short_fragment_prunes": sum(row["short_fragment_prune_count"] for row in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "v10_metrics", "v11_metrics", "true_positive_loss", "total_short_fragment_prunes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
