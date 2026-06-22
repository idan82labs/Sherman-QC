#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _range_contains(count_range: Sequence[Any], truth: int | None) -> bool | None:
    if truth is None or len(count_range) < 2:
        return None
    low = _as_int(count_range[0])
    high = _as_int(count_range[1])
    if low is None or high is None:
        return None
    return low <= truth <= high


def _range_width(count_range: Sequence[Any]) -> int | None:
    if len(count_range) < 2:
        return None
    low = _as_int(count_range[0])
    high = _as_int(count_range[1])
    if low is None or high is None:
        return None
    return max(0, high - low)


def _record_truth(record: Mapping[str, Any]) -> int | None:
    return _as_int(record.get("human_observed_bend_count"))


def _record_prediction(record: Mapping[str, Any]) -> int | None:
    return _as_int(record.get("algorithm_exact") if record.get("algorithm_exact") is not None else record.get("algorithm_observed_prediction"))


def _manual_case_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    truth = _record_truth(record)
    prediction = _record_prediction(record)
    count_range = list(record.get("algorithm_range") or ())
    contains = _range_contains(count_range, truth)
    width = _range_width(count_range)
    exact_match = prediction == truth if prediction is not None and truth is not None else None
    if exact_match is True:
        status = "exact_match"
    elif contains is True:
        status = "truth_in_range"
    elif contains is False:
        status = "truth_outside_range"
    elif truth is None:
        status = "missing_truth"
    else:
        status = "not_scored"
    return {
        "part_key": record.get("part_key"),
        "scan_name": record.get("scan_name"),
        "track": record.get("track"),
        "truth_observed": truth,
        "algorithm_prediction": prediction,
        "algorithm_range": count_range,
        "range_width": width,
        "truth_in_range": contains,
        "exact_match": exact_match,
        "exclude_from_accuracy": bool(record.get("exclude_from_accuracy")),
        "status": status,
        "notes": record.get("notes") or "",
    }


def _manual_metrics(manual_validation: Mapping[str, Any]) -> Dict[str, Any]:
    all_rows = [_manual_case_row(record) for record in list(manual_validation.get("records") or ())]
    scored = [row for row in all_rows if not row["exclude_from_accuracy"] and row["truth_observed"] is not None]
    range_scored = [row for row in scored if row["truth_in_range"] is not None]
    exact_scored = [row for row in scored if row["algorithm_prediction"] is not None]
    range_hits = [row for row in range_scored if row["truth_in_range"] is True]
    exact_hits = [row for row in exact_scored if row["exact_match"] is True]
    range_misses = [row for row in range_scored if row["truth_in_range"] is False]
    status_counts = Counter(str(row["status"]) for row in scored)
    widths = [int(row["range_width"]) for row in range_scored if row["range_width"] is not None]
    return {
        "record_count": len(all_rows),
        "scored_case_count": len(scored),
        "excluded_case_count": len([row for row in all_rows if row["exclude_from_accuracy"]]),
        "range_scored_case_count": len(range_scored),
        "range_hit_count": len(range_hits),
        "range_hit_rate": round(len(range_hits) / max(len(range_scored), 1), 4),
        "exact_scored_case_count": len(exact_scored),
        "exact_hit_count": len(exact_hits),
        "exact_hit_rate": round(len(exact_hits) / max(len(exact_scored), 1), 4),
        "average_range_width": round(sum(widths) / max(len(widths), 1), 4) if widths else None,
        "status_counts": dict(sorted(status_counts.items())),
        "range_miss_cases": range_misses,
        "manual_cases": all_rows,
    }


def _diagnostic_gate_cases(raw_review: Mapping[str, Any]) -> Dict[str, Any]:
    cases = list(raw_review.get("cases") or ())
    by_recommendation: Dict[str, list[Mapping[str, Any]]] = {}
    for row in cases:
        key = str(row.get("final_offline_recommendation") or "unknown")
        by_recommendation.setdefault(key, []).append(row)
    return {
        "case_count": len(cases),
        "recommendation_counts": raw_review.get("recommendation_counts") or {},
        "candidate_for_review": [
            {
                "part_id": row.get("part_id"),
                "target_count": row.get("target_count"),
                "selected_family_count": row.get("selected_family_count"),
                "safe_candidate_count": row.get("safe_candidate_count"),
                "reason_codes": row.get("final_recommendation_reasons") or [],
            }
            for row in by_recommendation.get("candidate_for_sparse_arrangement_review", [])
        ],
        "held_or_abstained": [
            {
                "part_id": row.get("part_id"),
                "recommendation": row.get("final_offline_recommendation"),
                "quality_decision": row.get("quality_decision"),
                "sparse_status": row.get("sparse_status"),
                "reason_codes": row.get("final_recommendation_reasons") or [],
            }
            for row in cases
            if row.get("final_offline_recommendation") != "candidate_for_sparse_arrangement_review"
        ],
    }


def _merge_decision(manual: Mapping[str, Any], raw_gates: Mapping[str, Any]) -> Dict[str, Any]:
    blockers: list[str] = []
    allowed: list[str] = []
    if manual.get("range_hit_rate") != 1.0:
        blockers.append("manual_validation_has_truth_outside_range_cases")
    if raw_gates.get("held_or_abstained"):
        blockers.append("raw_crease_lane_has_hold_or_abstain_cases")
    if raw_gates.get("candidate_for_review"):
        allowed.append("sparse_arrangement_candidate_exists_for_offline_review")
    blockers.append("runtime_path_unchanged_and_f1_still_offline_only")
    return {
        "main_branch_runtime_promotion": "not_ready",
        "offline_pr_merge_readiness": "safe_as_diagnostic_artifacts" if allowed else "diagnostic_only_no_runtime_promotion",
        "blockers": blockers,
        "allowed_next_steps": allowed
        + [
            "expand_sparse_arrangement_validation_on_clean_known_cases",
            "keep_scan_quality_gate_for_partial_or_poor_coverage_scans",
            "do_not_replace_existing_exact_f1_results_when_raw_crease_lane_abstains",
        ],
    }


def summarize_f1_merge_readiness(
    *,
    manual_validation: Mapping[str, Any],
    raw_crease_review: Mapping[str, Any],
) -> Dict[str, Any]:
    manual = _manual_metrics(manual_validation)
    raw_gates = _diagnostic_gate_cases(raw_crease_review)
    decision = _merge_decision(manual, raw_gates)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Merge-readiness status for offline F1 research artifacts.",
        "runtime_contract": "No runtime zero-shot or product API mutation is approved by this report.",
        "manual_validation_metrics": manual,
        "raw_crease_review_gates": raw_gates,
        "merge_decision": decision,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    manual = dict(report.get("manual_validation_metrics") or {})
    raw = dict(report.get("raw_crease_review_gates") or {})
    decision = dict(report.get("merge_decision") or {})
    lines = [
        "# F1 Merge Readiness Status",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"Runtime contract: `{report.get('runtime_contract')}`",
        "",
        "## Decision",
        "",
        f"- Main-branch runtime promotion: `{decision.get('main_branch_runtime_promotion')}`.",
        f"- Offline PR merge readiness: `{decision.get('offline_pr_merge_readiness')}`.",
        "",
        "Blockers:",
    ]
    for blocker in decision.get("blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "Allowed next steps:"])
    for step in decision.get("allowed_next_steps") or []:
        lines.append(f"- `{step}`")

    lines.extend(
        [
            "",
            "## Manual Validation Metrics",
            "",
            f"- Scored cases: `{manual.get('scored_case_count')}` from `{manual.get('record_count')}` records.",
            f"- Range hit rate: `{manual.get('range_hit_rate')}` (`{manual.get('range_hit_count')}/{manual.get('range_scored_case_count')}`).",
            f"- Exact hit rate where an exact/prediction exists: `{manual.get('exact_hit_rate')}` (`{manual.get('exact_hit_count')}/{manual.get('exact_scored_case_count')}`).",
            f"- Average accepted range width: `{manual.get('average_range_width')}`.",
            "",
            "Range misses:",
        ]
    )
    misses = list(manual.get("range_miss_cases") or [])
    if not misses:
        lines.append("- None.")
    for row in misses:
        lines.append(
            f"- `{row.get('part_key')}` / `{row.get('scan_name')}`: truth `{row.get('truth_observed')}`, range `{row.get('algorithm_range')}`, status `{row.get('status')}`."
        )

    lines.extend(["", "## Raw Crease Review Gates", ""])
    for key, value in sorted((raw.get("recommendation_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "Sparse-arrangement review candidates:"])
    candidates = list(raw.get("candidate_for_review") or [])
    if not candidates:
        lines.append("- None.")
    for row in candidates:
        lines.append(
            f"- `{row.get('part_id')}`: selected `{row.get('selected_family_count')}` / target `{row.get('target_count')}`, safe candidates `{row.get('safe_candidate_count')}`."
        )
    lines.extend(["", "Held or abstained cases:"])
    for row in raw.get("held_or_abstained") or []:
        lines.append(
            f"- `{row.get('part_id')}`: `{row.get('recommendation')}` (`{row.get('quality_decision')}`, `{row.get('sparse_status')}`)."
        )
    lines.extend(["", "## Practical Merge Path", ""])
    lines.extend(
        [
            "- Merge the current branch only as an offline research/diagnostic artifact lane.",
            "- Do not route production counts through sparse raw creases yet.",
            "- Next promotion candidate is not all F1; it is the narrow sparse-arrangement gate on clean known scans like `37361005` after broader validation.",
            "- Poor/partial scans such as `47959001` and `49001000` should stay behind scan-quality/observed-support gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize F1 offline merge readiness.")
    parser.add_argument("--manual-validation-json", required=True, type=Path)
    parser.add_argument("--raw-crease-review-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    report = summarize_f1_merge_readiness(
        manual_validation=_load_json(args.manual_validation_json),
        raw_crease_review=_load_json(args.raw_crease_review_json),
    )
    _write_json(args.output_json, report)
    _write_text(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "main_branch_runtime_promotion": report["merge_decision"]["main_branch_runtime_promotion"],
                "offline_pr_merge_readiness": report["merge_decision"]["offline_pr_merge_readiness"],
                "range_hit_rate": report["manual_validation_metrics"]["range_hit_rate"],
                "raw_recommendation_counts": report["raw_crease_review_gates"]["recommendation_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
