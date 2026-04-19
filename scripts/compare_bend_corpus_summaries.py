#!/usr/bin/env python3
"""Compare two bend corpus evaluation summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from typing import Sequence


IGNORED_TOP_LEVEL_KEYS = {"results", "warnings", "manifest_path"}
IGNORED_AGGREGATE_KEYS = {"_contradiction_cases", "_invariant_fail_cases"}


def _load_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in payload and isinstance(payload["aggregate"], dict):
        return payload["aggregate"]
    return payload


def _flatten_metrics(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if prefix == "" and key in IGNORED_TOP_LEVEL_KEYS:
                continue
            if prefix == "" and key == "aggregate" and isinstance(value, dict):
                flat.update(_flatten_metrics(value, ""))
                continue
            if key in IGNORED_AGGREGATE_KEYS:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(_flatten_metrics(value, next_prefix))
        return flat
    if isinstance(obj, list):
        return flat
    flat[prefix] = obj
    return flat


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_summaries(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    baseline_flat = _flatten_metrics(baseline)
    candidate_flat = _flatten_metrics(candidate)
    all_keys = sorted(set(baseline_flat) | set(candidate_flat))
    changed: List[Dict[str, Any]] = []
    only_baseline: List[str] = []
    only_candidate: List[str] = []

    for key in all_keys:
        in_baseline = key in baseline_flat
        in_candidate = key in candidate_flat
        if in_baseline and not in_candidate:
            only_baseline.append(key)
            continue
        if in_candidate and not in_baseline:
            only_candidate.append(key)
            continue
        baseline_value = baseline_flat[key]
        candidate_value = candidate_flat[key]
        if baseline_value == candidate_value:
            continue
        row: Dict[str, Any] = {
            "metric": key,
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
        if _is_numeric(baseline_value) and _is_numeric(candidate_value):
            row["delta"] = round(float(candidate_value) - float(baseline_value), 6)
        changed.append(row)

    return {
        "changed_metric_count": len(changed),
        "only_in_baseline_count": len(only_baseline),
        "only_in_candidate_count": len(only_candidate),
        "changed_metrics": changed,
        "only_in_baseline": only_baseline,
        "only_in_candidate": only_candidate,
    }


def _markdown_lines(report: Dict[str, Any], baseline_path: Path, candidate_path: Path) -> Iterable[str]:
    yield "# Bend Corpus Summary Comparison"
    yield ""
    yield f"- Baseline: `{baseline_path}`"
    yield f"- Candidate: `{candidate_path}`"
    yield f"- Changed metrics: `{report['changed_metric_count']}`"
    yield f"- Only in baseline: `{report['only_in_baseline_count']}`"
    yield f"- Only in candidate: `{report['only_in_candidate_count']}`"
    yield ""
    yield "## Changed Metrics"
    yield ""
    if not report["changed_metrics"]:
        yield "No changed metrics."
        return
    yield "| Metric | Baseline | Candidate | Delta |"
    yield "| --- | ---: | ---: | ---: |"
    for row in report["changed_metrics"]:
        delta = row.get("delta", "")
        yield f"| `{row['metric']}` | `{row['baseline']}` | `{row['candidate']}` | `{delta}` |"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline summary.json path")
    parser.add_argument("candidate", type=Path, help="Candidate summary.json path")
    parser.add_argument("--output-json", type=Path, help="Optional path to write comparison JSON")
    parser.add_argument("--output-markdown", type=Path, help="Optional path to write comparison markdown")
    args = parser.parse_args(argv)

    report = compare_summaries(_load_summary(args.baseline), _load_summary(args.candidate))
    payload = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        **report,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.write_text("\n".join(_markdown_lines(payload, args.baseline, args.candidate)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
