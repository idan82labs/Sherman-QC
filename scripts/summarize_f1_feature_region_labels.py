#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


VALID_LABELS = {
    "conventional_bend",
    "raised_form_feature",
    "fragment_or_duplicate",
    "rolled_transition",
    "unclear",
}


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


def summarize_feature_region_labels(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(payload.get("region_labels") or ())
    invalid: List[Dict[str, Any]] = []
    pending: List[str] = []
    counts: Counter[str] = Counter()
    labeled_regions: List[Dict[str, Any]] = []

    for row in rows:
        bend_id = str(row.get("bend_id") or "")
        label = row.get("human_feature_label")
        if label is None or label == "":
            pending.append(bend_id)
            normalized = None
        else:
            normalized = str(label)
            if normalized not in VALID_LABELS:
                invalid.append({"bend_id": bend_id, "label": normalized})
            else:
                counts[normalized] += 1
        labeled_regions.append(
            {
                "bend_id": bend_id,
                "human_feature_label": normalized,
                "suggested_role": row.get("suggested_role"),
                "diagnostic_tags": list(row.get("diagnostic_tags") or ()),
                "notes": row.get("notes") or "",
            }
        )

    expected_conventional = _as_int(payload.get("truth_conventional_bends"))
    expected_raised = _as_int(payload.get("truth_raised_form_features"))
    expected_total = _as_int(payload.get("truth_total_features"))
    expected_counted_total = None
    if expected_conventional is not None:
        expected_counted_total = expected_conventional + int(expected_raised or 0)
    conventional_count = int(counts["conventional_bend"])
    raised_count = int(counts["raised_form_feature"])
    total_counted_feature_count = conventional_count + raised_count + int(counts["rolled_transition"])
    region_capacity_deficit = (
        max(0, int(expected_counted_total) - len(rows))
        if expected_counted_total is not None
        else 0
    )

    complete = not pending and not invalid
    resolves_conventional = complete and expected_conventional is not None and conventional_count == expected_conventional
    resolves_raised = complete and (
        expected_raised is None
        or raised_count == expected_raised
    )
    resolves_total = complete and (
        expected_total is None
        or total_counted_feature_count == expected_total
    )

    blocker_status = "pending_labels"
    if region_capacity_deficit > 0:
        blocker_status = "owned_region_capacity_deficit"
    if invalid:
        blocker_status = "invalid_labels"
    elif region_capacity_deficit == 0 and complete and resolves_conventional and resolves_raised:
        blocker_status = "resolved_by_region_labels"
    elif region_capacity_deficit == 0 and complete:
        blocker_status = "labels_do_not_resolve_expected_counts"

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_key": payload.get("part_key"),
        "scan_name": payload.get("scan_name"),
        "region_count": len(rows),
        "valid_labels": sorted(VALID_LABELS),
        "pending_region_ids": pending,
        "invalid_labels": invalid,
        "label_counts": dict(sorted(counts.items())),
        "computed_counts": {
            "conventional_bend_count": conventional_count,
            "raised_form_feature_count": raised_count,
            "rolled_transition_count": int(counts["rolled_transition"]),
            "fragment_or_duplicate_count": int(counts["fragment_or_duplicate"]),
            "unclear_count": int(counts["unclear"]),
            "total_counted_feature_count": total_counted_feature_count,
        },
        "expected_counts": {
            "conventional_bend_count": expected_conventional,
            "raised_form_feature_count": expected_raised,
            "total_feature_count": expected_total,
            "counted_conventional_plus_raised": expected_counted_total,
        },
        "label_capacity": {
            "owned_region_count": len(rows),
            "expected_counted_conventional_plus_raised": expected_counted_total,
            "owned_region_capacity_deficit": region_capacity_deficit,
            "requires_split_or_missing_region_recovery": bool(region_capacity_deficit > 0),
        },
        "resolves_expected_counts": {
            "conventional_bends": bool(resolves_conventional),
            "raised_form_features": bool(resolves_raised),
            "total_features": bool(resolves_total),
        },
        "blocker_status": blocker_status,
        "regions": labeled_regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize manually labeled F1 owned-region feature classes.")
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    summary = summarize_feature_region_labels(_load_json(args.labels_json))
    _write_json(args.output_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "region_count": summary["region_count"],
                "pending_count": len(summary["pending_region_ids"]),
                "invalid_count": len(summary["invalid_labels"]),
                "blocker_status": summary["blocker_status"],
                "computed_counts": summary["computed_counts"],
            },
            indent=2,
        )
    )
    return 0 if not summary["invalid_labels"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
