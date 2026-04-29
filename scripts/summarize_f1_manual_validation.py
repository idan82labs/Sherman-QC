#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _range_contains(value: Optional[int], count_range: Sequence[Any]) -> Optional[bool]:
    if value is None or len(count_range) < 2:
        return None
    low = _as_int(count_range[0])
    high = _as_int(count_range[1])
    if low is None or high is None:
        return None
    return min(low, high) <= value <= max(low, high)


def _truth_count(label: Mapping[str, Any]) -> Optional[int]:
    truth = _as_int(label.get("human_observed_bend_count"))
    if truth is None and str(label.get("track") or "") == "known":
        truth = _as_int(label.get("human_full_part_bend_count"))
    return truth


def _algorithm_exact(label: Mapping[str, Any]) -> Optional[int]:
    value = _as_int(label.get("algorithm_observed_prediction"))
    if value is None:
        value = _as_int(label.get("algorithm_exact"))
    return value


def _is_disputed_or_special(label: Mapping[str, Any]) -> bool:
    notes = str(label.get("notes") or "").lower()
    return bool(
        label.get("legacy_expected_disputed")
        or label.get("feature_breakdown")
        or label.get("part_family")
        or "problematic" in notes
        or "poor" in notes
        or "מעורגל" in notes
        or "rolled" in notes
        or "raised" in notes
    )


def _empty_bucket() -> Dict[str, Any]:
    return {
        "cases": 0,
        "pending": 0,
        "exact_evaluated": 0,
        "exact_correct": 0,
        "range_evaluated": 0,
        "range_contains": 0,
    }


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    bucket["exact_accuracy"] = None if bucket["exact_evaluated"] == 0 else round(bucket["exact_correct"] / bucket["exact_evaluated"], 3)
    bucket["range_contains_truth_rate"] = None if bucket["range_evaluated"] == 0 else round(bucket["range_contains"] / bucket["range_evaluated"], 3)
    return bucket


def _score_label(label: Mapping[str, Any]) -> Dict[str, Any]:
    truth = _truth_count(label)
    algorithm_exact = _algorithm_exact(label)
    algorithm_range = list(label.get("algorithm_range") or ())
    exact_match = None if truth is None or algorithm_exact is None else bool(algorithm_exact == truth)
    range_hit = _range_contains(truth, algorithm_range)
    return {
        "track": str(label.get("track") or "unknown"),
        "part_key": label.get("part_key"),
        "scan_name": label.get("scan_name"),
        "truth_count": truth,
        "algorithm_exact_or_observed_prediction": algorithm_exact,
        "algorithm_range": algorithm_range,
        "exact_match": exact_match,
        "range_contains_truth": range_hit,
        "excluded_from_accuracy": bool(label.get("exclude_from_accuracy")),
        "exclusion_reason": label.get("exclusion_reason"),
        "disputed_or_special": _is_disputed_or_special(label),
        "feature_breakdown": label.get("feature_breakdown"),
        "part_family": label.get("part_family"),
        "notes": label.get("notes") or "",
        "false_positive_bend_ids": list(label.get("false_positive_bend_ids") or ()),
        "missed_bend_notes": list(label.get("missed_bend_notes") or ()),
    }


def _add_to_bucket(bucket: Dict[str, Any], scored: Mapping[str, Any]) -> None:
    bucket["cases"] += 1
    bucket["pending"] += int(scored.get("truth_count") is None)
    exact_match = scored.get("exact_match")
    range_hit = scored.get("range_contains_truth")
    bucket["exact_evaluated"] += int(exact_match is not None)
    bucket["exact_correct"] += int(bool(exact_match))
    bucket["range_evaluated"] += int(range_hit is not None)
    bucket["range_contains"] += int(bool(range_hit))


def summarize_manual_validation(labels_payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    clean_bucket = _empty_bucket()
    disputed_bucket = _empty_bucket()
    excluded_bucket = _empty_bucket()
    all_bucket = _empty_bucket()
    by_track: Dict[str, Dict[str, int]] = {}
    label_rows = list(labels_payload.get("labels") or labels_payload.get("records") or ())
    for label in label_rows:
        scored = _score_label(label)
        rows.append(scored)
        _add_to_bucket(all_bucket, scored)
        if scored["excluded_from_accuracy"]:
            _add_to_bucket(excluded_bucket, scored)
        elif scored["disputed_or_special"]:
            _add_to_bucket(disputed_bucket, scored)
        else:
            _add_to_bucket(clean_bucket, scored)
        bucket = by_track.setdefault(scored["track"], _empty_bucket())
        _add_to_bucket(bucket, scored)
    for bucket in by_track.values():
        _finalize_bucket(bucket)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(rows),
        "pending_label_count": int(all_bucket["pending"]),
        "exact_evaluated": int(all_bucket["exact_evaluated"]),
        "exact_accuracy": _finalize_bucket(dict(all_bucket))["exact_accuracy"],
        "range_evaluated": int(all_bucket["range_evaluated"]),
        "range_contains_truth_rate": _finalize_bucket(dict(all_bucket))["range_contains_truth_rate"],
        "clean_accuracy": _finalize_bucket(clean_bucket),
        "disputed_or_special_accuracy": _finalize_bucket(disputed_bucket),
        "excluded_accuracy": _finalize_bucket(excluded_bucket),
        "by_track": dict(sorted(by_track.items())),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize manual labels for F1 validation review queues.")
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    labels_path = Path(args.labels_json).expanduser().resolve()
    summary = summarize_manual_validation(_load_json(labels_path))
    output_path = Path(args.output_json).expanduser().resolve() if args.output_json else labels_path.with_name("manual_validation_summary.json")
    _write_json(output_path, summary)
    print(json.dumps({"output_json": str(output_path), "case_count": summary["case_count"], "pending_label_count": summary["pending_label_count"], "exact_accuracy": summary["exact_accuracy"], "range_contains_truth_rate": summary["range_contains_truth_rate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
