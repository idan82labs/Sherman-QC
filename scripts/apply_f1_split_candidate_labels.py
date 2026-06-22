#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from summarize_f1_split_candidate_labels import VALID_SPLIT_LABELS, summarize_split_candidate_labels


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_label_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"split assignment must be BEND_ID=LABEL, got {raw!r}")
    bend_id, label = raw.split("=", 1)
    bend_id = bend_id.strip()
    label = label.strip()
    if not bend_id:
        raise ValueError(f"missing bend id in assignment {raw!r}")
    if label not in VALID_SPLIT_LABELS:
        raise ValueError(f"invalid split label {label!r} for {bend_id}; valid labels: {sorted(VALID_SPLIT_LABELS)}")
    return bend_id, label


def _parse_label_assignments(raw_values: Sequence[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for raw_value in raw_values:
        for chunk in str(raw_value).replace("\n", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            bend_id, label = _parse_label_assignment(chunk)
            labels[bend_id] = label
    return labels


def _load_label_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    payload = _load_json(path)
    raw_map = payload.get("labels") if isinstance(payload.get("labels"), Mapping) else payload
    if not isinstance(raw_map, Mapping):
        raise ValueError("split label map JSON must be an object or contain an object at key 'labels'")
    labels: Dict[str, str] = {}
    for bend_id, label in raw_map.items():
        bend = str(bend_id).strip()
        value = str(label).strip()
        if value not in VALID_SPLIT_LABELS:
            raise ValueError(f"invalid split label {value!r} for {bend}; valid labels: {sorted(VALID_SPLIT_LABELS)}")
        labels[bend] = value
    return labels


def apply_split_candidate_labels(
    *,
    template_payload: Mapping[str, Any],
    label_map: Mapping[str, str],
    notes_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload = json.loads(json.dumps(template_payload))
    notes_map = notes_map or {}
    known_candidates = {str(row.get("bend_id") or "") for row in list(payload.get("candidate_labels") or ())}
    unknown = sorted(set(label_map) - known_candidates)
    if unknown:
        raise ValueError(f"unknown split candidate ids in label map: {unknown}")
    for row in list(payload.get("candidate_labels") or ()):
        bend_id = str(row.get("bend_id") or "")
        if bend_id in label_map:
            row["human_split_label"] = str(label_map[bend_id])
        if bend_id in notes_map:
            row["notes"] = str(notes_map[bend_id])
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply simple OB split assignments to an F1 split-candidate template.")
    parser.add_argument("--template-json", required=True, type=Path)
    parser.add_argument("--label-map-json", type=Path, default=None, help="JSON object like {'OB2': 'split_into_two_features'} or {'labels': {...}}.")
    parser.add_argument("--label", action="append", default=[], help="Inline assignment, e.g. OB2=split_into_two_features. May be repeated.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)

    labels = _load_label_map(args.label_map_json)
    labels.update(_parse_label_assignments(args.label))

    filled = apply_split_candidate_labels(template_payload=_load_json(args.template_json), label_map=labels)
    _write_json(args.output_json, filled)
    summary = summarize_split_candidate_labels(filled)
    if args.summary_json:
        _write_json(args.summary_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "summary_json": str(args.summary_json) if args.summary_json else None,
                "applied_label_count": len(labels),
                "pending_count": len(summary["pending_candidate_ids"]),
                "invalid_count": len(summary["invalid_labels"]),
                "blocker_status": summary["blocker_status"],
                "capacity_resolution": summary["capacity_resolution"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
