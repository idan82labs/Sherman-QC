#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _truth_conventional(record: Mapping[str, Any]) -> Optional[int]:
    feature = dict(record.get("feature_breakdown") or {})
    value = _as_int(feature.get("conventional_bends"))
    if value is None:
        value = _as_int(record.get("human_observed_bend_count"))
    return value


def _truth_raised(record: Mapping[str, Any]) -> Optional[int]:
    feature = dict(record.get("feature_breakdown") or {})
    return _as_int(feature.get("raised_form_features"))


def _truth_total_features(record: Mapping[str, Any]) -> Optional[int]:
    feature = dict(record.get("feature_breakdown") or {})
    explicit = _as_int(feature.get("legacy_expected_total_features"))
    if explicit is not None:
        return explicit
    conventional = _truth_conventional(record)
    raised = _truth_raised(record)
    if conventional is not None and raised is not None:
        return conventional + raised
    return _as_int(record.get("human_full_part_bend_count"))


def _algorithm_range(record: Mapping[str, Any]) -> List[int]:
    values = list(record.get("algorithm_range") or ())
    if len(values) < 2:
        return []
    low = _as_int(values[0])
    high = _as_int(values[1])
    if low is None or high is None:
        return []
    return [low, high]


def _algorithm_singleton(record: Mapping[str, Any]) -> Optional[int]:
    count_range = _algorithm_range(record)
    if len(count_range) == 2 and count_range[0] == count_range[1]:
        return count_range[0]
    return _as_int(record.get("algorithm_exact"))


def _algorithm_raw_exact(record: Mapping[str, Any]) -> Optional[int]:
    return _as_int(record.get("algorithm_raw_exact_field"))


def _classify_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    count_range = _algorithm_range(record)
    conventional = _truth_conventional(record)
    raised = _truth_raised(record)
    total = _truth_total_features(record)
    singleton = _algorithm_singleton(record)
    raw_exact = _algorithm_raw_exact(record)
    conventional_hit = _range_contains(conventional, count_range)
    total_hit = _range_contains(total, count_range)
    has_nonconventional = bool((raised or 0) > 0 or record.get("part_family") or record.get("legacy_expected_disputed"))
    excluded = bool(record.get("exclude_from_accuracy"))

    blocker = None
    if excluded:
        blocker = "excluded_bad_or_unscored_scan"
    elif has_nonconventional and conventional_hit is False and total_hit is True:
        blocker = "total_feature_match_but_conventional_miss"
    elif has_nonconventional and conventional_hit is True:
        blocker = "special_feature_range_contains_conventional_truth"
    elif conventional_hit is False:
        blocker = "conventional_range_miss"
    elif singleton is not None and conventional is not None and singleton != conventional:
        blocker = "singleton_conventional_mismatch"
    else:
        blocker = "clean_or_contained"

    promotion_safe = blocker == "clean_or_contained" and not has_nonconventional

    return {
        "track": record.get("track"),
        "part_key": record.get("part_key"),
        "scan_name": record.get("scan_name"),
        "algorithm_range": count_range,
        "algorithm_exact": record.get("algorithm_exact"),
        "algorithm_raw_exact_field": raw_exact,
        "truth_conventional_bends": conventional,
        "truth_raised_form_features": raised,
        "truth_total_features": total,
        "range_contains_conventional": conventional_hit,
        "range_contains_total_features": total_hit,
        "has_nonconventional_features": has_nonconventional,
        "part_family": record.get("part_family"),
        "excluded_from_accuracy": excluded,
        "feature_semantics_status": blocker,
        "promotion_safe_under_conventional_semantics": promotion_safe,
        "algorithm_feature_split_status": (dict(record.get("feature_count_semantics") or {}).get("algorithm_feature_split_status") or "unknown"),
        "notes": record.get("notes") or "",
    }


def audit_feature_semantics(labels_payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [_classify_record(record) for record in list(labels_payload.get("records") or labels_payload.get("labels") or ())]
    status_counts = Counter(str(row["feature_semantics_status"]) for row in rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "promotion_safe_case_count": sum(1 for row in rows if row["promotion_safe_under_conventional_semantics"]),
        "nonconventional_case_count": sum(1 for row in rows if row["has_nonconventional_features"]),
        "feature_split_not_modeled_count": sum(
            1
            for row in rows
            if row["has_nonconventional_features"] and row["algorithm_feature_split_status"] == "not_modeled"
        ),
        "blockers": [
            row
            for row in rows
            if row["feature_semantics_status"]
            in {
                "total_feature_match_but_conventional_miss",
                "conventional_range_miss",
                "singleton_conventional_mismatch",
            }
        ],
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit F1 feature-count semantics against manual labels.")
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    report = audit_feature_semantics(_load_json(args.labels_json))
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "case_count": report["case_count"],
                "blocker_count": len(report["blockers"]),
                "status_counts": report["status_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
