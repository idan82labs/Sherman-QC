#!/usr/bin/env python3
"""Build deterministic, non-authoritative labels for Shaked's 12 count-backed scans."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from build_additional_line_label_review import _scan_points, line_scan_support


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data/bend_corpus/annotations/shaked_12_assistant_manual_selection_20260717.json"
DEFAULT_REVIEW_ROOT = ROOT / "output/manual_labels/shaked-count-backed-12-20260717"
DEFAULT_SCAN_ROOT = ROOT / "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_path(review_root: Path, case: dict[str, Any]) -> Path:
    stem = str(case["scan_stem"])
    if case["source"] == "mesh_adjacency":
        return review_root / "mesh_adjacency_candidates" / f"{stem}.json"
    if case["source"] == "review_candidates":
        return review_root / "review_candidates_v1" / f"{case['case_id']}.json"
    raise ValueError(f"{case['case_id']}: unsupported source {case['source']}")


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("bend_id"))


def _line(record: dict[str, Any]) -> tuple[list[float], list[float]]:
    start = record.get("line_start")
    end = record.get("line_end")
    if start is None or end is None:
        endpoints = record.get("span_endpoints") or []
        if len(endpoints) >= 2:
            start, end = endpoints[:2]
    if start is None or end is None:
        raise ValueError(f"candidate {_record_id(record)} has no line endpoints")
    return [float(value) for value in start], [float(value) for value in end]


def _candidate_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Retain traceability without copying bulky candidate/cache payloads."""

    keys = (
        "candidate_id",
        "source_kind",
        "representative_source",
        "incident_plane_ids",
        "merged_candidate_ids",
        "bend_name",
        "bend_angle",
        "angle_deg",
        "confidence",
        "finite_support_fraction",
        "finite_support_longest_run_fraction",
        "endpoint_support_fraction",
        "median_line_distance_mm",
        "p90_line_distance_mm",
    )
    return {key: deepcopy(record[key]) for key in keys if record.get(key) is not None}


def build_case(
    case: dict[str, Any],
    *,
    scan_root: Path,
    review_root: Path,
) -> dict[str, Any]:
    source_path = _source_path(review_root, case)
    source = _read(source_path)
    records = source.get("manual_bends") or []
    by_id = {_record_id(record): record for record in records}
    selected_ids = [str(value) for value in case["selected_ids"]]
    expected = int(case["expected_bend_count"])
    if len(selected_ids) != expected or len(set(selected_ids)) != expected:
        raise ValueError(
            f"{case['case_id']}: selected {len(selected_ids)} non-unique IDs for expected {expected}"
        )
    missing = [value for value in selected_ids if value not in by_id]
    if missing:
        raise ValueError(f"{case['case_id']}: missing candidate IDs {missing} in {source_path}")

    scan_path = scan_root / f"{case['scan_stem']}.ply"
    if not scan_path.is_file():
        raise FileNotFoundError(scan_path)
    points = _scan_points(scan_path)
    bends: list[dict[str, Any]] = []
    short_supported_ids = {
        str(value) for value in case.get("allow_short_scan_supported_ids") or []
    }
    short_min_span_mm = float(case.get("short_scan_supported_min_span_mm") or 8.0)
    for index, selected_id in enumerate(selected_ids, start=1):
        candidate = deepcopy(by_id[selected_id])
        line_start, line_end = _line(candidate)
        support = line_scan_support(points, line_start, line_end)
        short_supported_override = bool(
            selected_id in short_supported_ids
            and support["supported_span_mm"] >= short_min_span_mm
            and support["support_fraction"] >= 0.35
            and support["bbox_overlap_fraction"] >= 0.90
        )
        if not support["frame_integrity_pass"] and not short_supported_override:
            raise ValueError(
                f"{case['case_id']} {selected_id}: accepted line failed scan/frame integrity: {support}"
            )
        bends.append(
            {
                "bend_id": f"B{index:02d}",
                "line_start": support["trimmed_line_start"],
                "line_end": support["trimmed_line_end"],
                "source_line_start": line_start,
                "source_line_end": line_end,
                "semantic": "conventional_bend",
                "source": str(case["source"]),
                "source_candidate_id": selected_id,
                "source_candidate_metadata": _candidate_metadata(candidate),
                "review_decision": "accepted_user_review_pending",
                "review_reason": str(case["review_reason"]),
                "confidence": str(case["confidence"]),
                "geometry_adjustment": "trimmed_to_longest_scan_supported_run",
                "frame_integrity_decision": (
                    "pass"
                    if support["frame_integrity_pass"]
                    else "documented_short_bend_override"
                ),
                "short_bend_min_supported_span_mm": (
                    short_min_span_mm if short_supported_override else None
                ),
                "scan_support": support,
            }
        )

    return {
        "schema_version": 1,
        "kind": "assistant_multiview_count_backed_bend_instance_labels",
        "case_id": str(case["case_id"]),
        "part_number": str(case["part_number"]),
        "coordinate_frame": "scan_raw",
        "scan_path": str(scan_path.resolve()),
        "drawing_path": str((ROOT / str(case["drawing"])).resolve()),
        "candidate_source_path": str(source_path.resolve()),
        "review_status": "assistant_multiview_complete_user_review_pending",
        "authoritative_ground_truth": False,
        "expected_count_source": "shaked_email_20260716",
        "sender_bend_count": expected,
        "accepted_bend_count": len(bends),
        "manual_bends": bends,
        "marker_reviews": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REVIEW_ROOT / "adjudicated_v1")
    args = parser.parse_args()

    ledger = _read(args.ledger.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, Any]] = []
    for case in ledger.get("cases") or []:
        labels = build_case(
            case,
            scan_root=args.scan_root.resolve(),
            review_root=args.review_root.resolve(),
        )
        case_root = output_root / str(case["case_id"])
        case_root.mkdir(parents=True, exist_ok=True)
        output_path = case_root / "bend_instance_labels.json"
        output_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        manifest_cases.append(
            {
                "case_id": labels["case_id"],
                "part_number": labels["part_number"],
                "sender_bend_count": labels["sender_bend_count"],
                "accepted_bend_count": labels["accepted_bend_count"],
                "confidence": str(case["confidence"]),
                "labels_path": str(output_path),
            }
        )

    total_expected = sum(int(case["sender_bend_count"]) for case in manifest_cases)
    total_accepted = sum(int(case["accepted_bend_count"]) for case in manifest_cases)
    if total_expected != total_accepted:
        raise ValueError(f"total count mismatch: expected {total_expected}, accepted {total_accepted}")
    output_manifest = {
        "schema_version": 1,
        "kind": "assistant_multiview_count_backed_label_manifest",
        "review_status": "assistant_multiview_complete_user_review_pending",
        "authoritative_ground_truth": False,
        "case_count": len(manifest_cases),
        "expected_bend_count": total_expected,
        "accepted_bend_count": total_accepted,
        "cases": manifest_cases,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
