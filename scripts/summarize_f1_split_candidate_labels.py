#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


VALID_SPLIT_LABELS = {
    "keep_single",
    "split_into_two_features",
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


def summarize_split_candidate_labels(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(payload.get("candidate_labels") or ())
    capacity_deficit = _as_int(payload.get("capacity_deficit"))
    invalid: List[Dict[str, Any]] = []
    pending: List[str] = []
    counts: Counter[str] = Counter()
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        bend_id = str(row.get("bend_id") or "")
        label = row.get("human_split_label")
        if label is None or label == "":
            pending.append(bend_id)
            normalized = None
        else:
            normalized = str(label)
            if normalized not in VALID_SPLIT_LABELS:
                invalid.append({"bend_id": bend_id, "label": normalized})
            else:
                counts[normalized] += 1
        candidates.append(
            {
                "bend_id": bend_id,
                "human_split_label": normalized,
                "suggested_split_rank": row.get("suggested_split_rank"),
                "notes": row.get("notes") or "",
            }
        )

    recovered_capacity = int(counts["split_into_two_features"])
    complete = not pending and not invalid
    resolves_capacity = (
        complete
        and capacity_deficit is not None
        and recovered_capacity == int(capacity_deficit)
    )
    over_split_risk = (
        complete
        and capacity_deficit is not None
        and recovered_capacity > int(capacity_deficit)
    )

    blocker_status = "pending_split_labels"
    if invalid:
        blocker_status = "invalid_split_labels"
    elif complete and capacity_deficit is None:
        blocker_status = "split_labels_complete_no_capacity_target"
    elif over_split_risk:
        blocker_status = "split_labels_over_recover_capacity"
    elif resolves_capacity:
        blocker_status = "resolved_by_split_labels"
    elif complete:
        blocker_status = "split_labels_do_not_resolve_capacity_deficit"

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_key": payload.get("part_key"),
        "scan_path": payload.get("scan_path"),
        "candidate_count": len(rows),
        "valid_split_labels": sorted(VALID_SPLIT_LABELS),
        "pending_candidate_ids": pending,
        "invalid_labels": invalid,
        "label_counts": dict(sorted(counts.items())),
        "capacity_resolution": {
            "capacity_deficit": capacity_deficit,
            "recovered_capacity_from_splits": recovered_capacity,
            "remaining_capacity_deficit": (
                max(0, int(capacity_deficit) - recovered_capacity)
                if capacity_deficit is not None
                else None
            ),
            "resolves_capacity_deficit": bool(resolves_capacity),
            "over_split_risk": bool(over_split_risk),
        },
        "blocker_status": blocker_status,
        "candidates": candidates,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize manually labeled F1 owned-region split candidates.")
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    summary = summarize_split_candidate_labels(_load_json(args.labels_json))
    _write_json(args.output_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "candidate_count": summary["candidate_count"],
                "pending_count": len(summary["pending_candidate_ids"]),
                "invalid_count": len(summary["invalid_labels"]),
                "blocker_status": summary["blocker_status"],
                "capacity_resolution": summary["capacity_resolution"],
            },
            indent=2,
        )
    )
    return 0 if not summary["invalid_labels"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
