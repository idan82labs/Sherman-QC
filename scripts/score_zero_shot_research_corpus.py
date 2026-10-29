#!/usr/bin/env python3
"""Score frozen F1 outputs after inference, with grouped research statistics and renders."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402
from scripts.render_f1_gold_overlay_review import render_case  # noqa: E402


DEFAULT_GOLD = ROOT / "output/research/f1-corpus-evidence-20260729-v1/freeze/sealed_gold_manifest.json"
DEFAULT_INFERENCE = ROOT / "output/research/f1-corpus-evidence-20260729-v1/inference/inference_summary.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-corpus-evidence-20260729-v1/evaluation"


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [round(max(0.0, center - spread), 6), round(min(1.0, center + spread), 6)]


def _micro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    line_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "line_f1": round(line_f1, 6),
    }


def _group_bootstrap(rows: list[dict[str, Any]], repeats: int = 10_000) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["part_key"]), []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(20260729)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sampled_rows = [row for key in sampled for row in groups[str(key)]]
        values.append(_micro(sampled_rows)["line_f1"])
    return {
        "line_f1_95pct_group_bootstrap": [
            round(float(np.quantile(values, 0.025)), 6),
            round(float(np.quantile(values, 0.975)), 6),
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--inference-summary", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-renders", action="store_true")
    args = parser.parse_args()
    gold = json.loads(args.gold_manifest.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_summary.read_text(encoding="utf-8"))
    if gold["corpus_freeze_id"] != inference["corpus_freeze_id"]:
        raise ValueError("freeze ID mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    match_errors = []
    for case in gold.get("cases") or []:
        token = str(case["case_token"])
        paths = inference["result_map"][token]
        manifest = json.loads(Path(paths["overlay_manifest"]).read_text(encoding="utf-8"))
        predictions = []
        for bend in manifest.get("bends") or []:
            line = bend.get("detected_line") or bend.get("cad_line") or {}
            start = line.get("display_start")
            end = line.get("display_end")
            if start is None or end is None:
                continue
            predictions.append(
                {"bend_id": str(bend.get("bend_id") or f"P{len(predictions)+1}"), "span_endpoints": [start, end]}
            )
        result = match_lines(case["manual_bends"], predictions, Thresholds())
        result.update(
            {
                "case_token": token,
                "record_id": case["record_id"],
                "part_key": case["part_key"],
                "scan_name": case["scan_name"],
                "scan_state": case["scan_state"],
            }
        )
        for match in result["matches"]:
            match_errors.append(match)
        case_dir = output_dir / "cases" / token
        case_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = case_dir / "prediction_owned_regions.json"
        prediction_path.write_text(json.dumps({"owned_bend_regions": predictions}, indent=2) + "\n")
        label_path = Path(case["label_path"])
        scan_candidates = sorted((Path(paths["zero_shot_result"]).parents[1]).glob("input.*"))
        if len(scan_candidates) != 1:
            raise ValueError(f"expected one opaque scan alias for {token}")
        render_path = case_dir / "gold_vs_f1_six_view.png"
        if not args.skip_renders:
            render_case(
                case_id=token,
                scan_path=scan_candidates[0],
                gold_path=label_path,
                prediction_path=prediction_path,
                output_path=render_path,
                max_points=260_000,
            )
        result["review_render"] = str(render_path) if render_path.is_file() else None
        (case_dir / "score.json").write_text(json.dumps(result, indent=2) + "\n")
        rows.append(result)

    micro = _micro(rows)
    exact_scans = sum(int(row["exact"]) for row in rows)
    count_exact_scans = sum(int(row["gold_count"] == row["prediction_count"]) for row in rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["part_key"]), []).append(row)
    exact_families = sum(all(row["exact"] for row in family) for family in grouped.values())
    error_summary = {}
    for key in ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap"):
        values = [float(item[key]) for item in match_errors]
        error_summary[key] = {
            "median": round(float(np.median(values)), 6) if values else None,
            "p95": round(float(np.quantile(values, 0.95)), 6) if values else None,
        }
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_zero_shot_f1_research_corpus_evidence",
        "corpus_freeze_id": gold["corpus_freeze_id"],
        "evaluation_contract": {
            **gold["protocol"],
            "matching": "one_to_one_minimum_cost",
            "axis_angle_deg_max": 5.0,
            "transverse_offset_mm_max": 4.0,
            "gold_span_overlap_min": 0.6,
            "prediction_span_overlap_min": 0.6,
            "calibration_result_excluded": True,
        },
        "sample": {
            "scan_count": len(rows),
            "unique_part_family_count": len(grouped),
            "full_scan_count": sum(row["scan_state"] == "full" for row in rows),
            "partial_scan_count": sum(row["scan_state"] == "partial" for row in rows),
        },
        "micro_bend_metrics": micro,
        "exact_scan_metrics": {
            "exact_scans": exact_scans,
            "total_scans": len(rows),
            "rate": round(exact_scans / len(rows), 6),
            "wilson_95pct": _wilson(exact_scans, len(rows)),
        },
        "count_only_metrics": {
            "count_exact_scans": count_exact_scans,
            "total_scans": len(rows),
            "rate": round(count_exact_scans / len(rows), 6),
            "geometrically_false_exact_count_scans": count_exact_scans - exact_scans,
        },
        "exact_part_family_metrics": {
            "all_scans_exact_families": exact_families,
            "total_part_families": len(grouped),
            "rate": round(exact_families / len(grouped), 6),
            "wilson_95pct": _wilson(exact_families, len(grouped)),
        },
        "matched_geometry": error_summary,
        "scan_state_subgroups": {
            state: {
                **_micro([row for row in rows if row["scan_state"] == state]),
                "scan_count": sum(row["scan_state"] == state for row in rows),
            }
            for state in ("full", "partial")
        },
        "uncertainty": _group_bootstrap(rows),
        "cases": rows,
    }
    report_path = output_dir / "research_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "micro_bend_metrics": micro,
        "exact_scan_metrics": report["exact_scan_metrics"],
        "exact_part_family_metrics": report["exact_part_family_metrics"],
        "research_metrics": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
