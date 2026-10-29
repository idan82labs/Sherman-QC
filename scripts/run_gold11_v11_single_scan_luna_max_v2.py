#!/usr/bin/env python3
"""Versioned one-scan V11 result wrapper with explicit score metrics.

The underlying V1 candidate remains the inference implementation.  This
wrapper preserves it byte-for-byte and writes a new artifact directory with
explicit precision/recall fields, a metric formula declaration, and wrapper
provenance.  It never reopens the frozen 11-case attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gold11_v11_single_case_candidate_v1 import (
    CASE_TOKEN,
    DEFAULT_OUTPUT as V1_DEFAULT_OUTPUT,
    run as run_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explicit_metrics(score: dict[str, Any]) -> dict[str, Any]:
    """Derive the standard metrics from the unchanged matcher counts."""
    tp = int(score["true_positive"])
    fp = int(score["false_positive"])
    fn = int(score["false_negative"])
    if min(tp, fp, fn) < 0:
        raise ValueError("NEGATIVE_MATCH_COUNT")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "gold_count": int(score["gold_count"]),
        "prediction_count": int(score["prediction_count"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "formula": "TP/(TP+FP), TP/(TP+FN), 2TP/(2TP+FP+FN)",
    }


def run(output: Path) -> dict[str, Any]:
    result = run_v1(output)
    result["score"].update(explicit_metrics(result["score"]))
    result["score"]["score_schema"] = "UNCHANGED_ONE_TO_ONE_MATCH_COUNTS_EXPLICIT_METRICS_V1"
    result["source_bindings"]["wrapper_script_sha256"] = _sha256(Path(__file__).resolve())
    result["wrapper"] = {
        "implementation": "V11_SINGLE_SCAN_EXPLICIT_SCORE_METRICS_WRAPPER_V2",
        "base_candidate": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V1",
        "base_output_contract": str(V1_DEFAULT_OUTPUT.relative_to(ROOT)),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checksums_path = output / "CHECKSUMS.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["result_sha256"] = _sha256(result_path)
    checksums["source_bindings"] = result["source_bindings"]
    checksums_path.write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/research/gold11_v11_single_scan_luna_max_v2"
        / CASE_TOKEN,
    )
    args = parser.parse_args()
    result = run(args.output.resolve())
    score = result["score"]
    print(json.dumps({
        "status": result["status"],
        "predictions": score["prediction_count"],
        "true_positive": score["true_positive"],
        "false_positive": score["false_positive"],
        "false_negative": score["false_negative"],
        "gold_count": score["gold_count"],
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "bend_count_capture": score["bend_count_capture"],
        "target_80_percent_met": score["target_80_percent_met"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
