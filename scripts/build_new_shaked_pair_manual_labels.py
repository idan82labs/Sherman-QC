#!/usr/bin/env python3
"""Materialize non-authoritative bend labels for Shaked's new X/V scan pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from build_additional_line_label_review import _scan_points, line_scan_support


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = (
    ROOT
    / "data/bend_corpus/annotations/shaked_xv_pairs_assistant_manual_selection_20260730.json"
)
DEFAULT_SLICE = (
    ROOT
    / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/inference_slice.json"
)
DEFAULT_CANDIDATE_ROOT = ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/candidate_pool"
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels"
)
PRIOR_X_OVERLAY = (
    ROOT
    / "output/experiments/good-vs-defective-38172000-20260728/"
    "f1_corrected_prismatic_X_v2/zero_shot_bend_overlay_manifest.json"
)
PRIOR_V_MANUAL = (
    ROOT
    / "output/experiments/good-vs-defective-38172000-20260728/"
    "manual_10_bends_V_good_corrected_v2.json"
)
DRAWINGS = {
    "35705010_A": "35705010_A.PDF",
    "38173000_02": "38173000_02.pdf",
    "38172000_01": "38172000_01.pdf",
    "50545000_05": "MEC002864_05.pdf",
    "50546000_05": "MEC002865_05.pdf",
    "50556000_04": "MEC002966_04.pdf",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line(record: dict[str, Any]) -> tuple[list[float], list[float]]:
    start = record.get("line_start")
    end = record.get("line_end")
    if start is None or end is None:
        endpoints = record.get("span_endpoints") or []
        if len(endpoints) == 2:
            start, end = endpoints
    if start is None or end is None:
        detected = record.get("detected_line") or record.get("cad_line") or {}
        start = detected.get("display_start") or detected.get("raw_start")
        end = detected.get("display_end") or detected.get("raw_end")
    if start is None or end is None:
        raise ValueError(f"record {record.get('bend_id')} has no line endpoints")
    return [float(value) for value in start], [float(value) for value in end]


def _candidate_records(candidate_root: Path, token: str, selected_ids: Iterable[str]) -> list[dict[str, Any]]:
    matches = sorted(candidate_root.glob(f"*_{token}.json"))
    if len(matches) != 1:
        raise ValueError(f"{token}: expected one candidate pool, found {len(matches)}")
    payload = _read(matches[0])
    records = {
        str(record.get("source_candidate_id")): record
        for record in payload.get("candidates") or []
        if record.get("candidate_source") == "scan_mesh_adjacency"
    }
    selected = []
    for source_id in selected_ids:
        if source_id not in records:
            raise ValueError(f"{token}: missing scan-mesh candidate {source_id}")
        selected.append(dict(records[source_id]))
    return selected


def _prior_overlay_records(selected_ids: Iterable[str]) -> list[dict[str, Any]]:
    records = {str(row.get("bend_id")): row for row in _read(PRIOR_X_OVERLAY).get("bends") or []}
    return [dict(records[source_id]) for source_id in selected_ids]


def _prior_manual_records() -> list[dict[str, Any]]:
    return [dict(row) for row in _read(PRIOR_V_MANUAL).get("manual_bends") or []]


def _source_records(
    case: dict[str, Any],
    *,
    candidate_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    kind = str(case["source_kind"])
    if kind == "candidate_pool":
        token = str(case["case_token"])
        matches = sorted(candidate_root.glob(f"*_{token}.json"))
        return (
            _candidate_records(candidate_root, token, case.get("selected_ids") or []),
            str(matches[0].resolve()),
        )
    if kind == "prior_corrected_overlay":
        return _prior_overlay_records(case.get("selected_ids") or []), str(PRIOR_X_OVERLAY)
    if kind == "prior_corrected_manual":
        return _prior_manual_records(), str(PRIOR_V_MANUAL)
    raise ValueError(f"unsupported source_kind: {kind}")


def build_case(
    case: dict[str, Any],
    *,
    scan_path: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    records, source_path = _source_records(case, candidate_root=candidate_root)
    expected = int(case["expected_bend_count"])
    if len(records) != expected:
        raise ValueError(
            f"{case['case_token']}: selected {len(records)} records for expected {expected}"
        )
    points = _scan_points(scan_path)
    bends = []
    for index, record in enumerate(records, start=1):
        source_start, source_end = _line(record)
        support = line_scan_support(points, source_start, source_end)
        if not support["frame_integrity_pass"]:
            raise ValueError(
                f"{case['case_token']} record {record.get('bend_id')}: failed scan support: {support}"
            )
        source_id = str(
            record.get("source_candidate_id")
            or record.get("bend_id")
            or f"source-{index}"
        )
        bends.append(
            {
                "bend_id": f"B{index:02d}",
                "line_start": support["trimmed_line_start"],
                "line_end": support["trimmed_line_end"],
                "source_line_start": source_start,
                "source_line_end": source_end,
                "semantic": "conventional_bend",
                "source": str(case["source_kind"]),
                "source_candidate_id": source_id,
                "review_decision": "accepted_user_review_pending",
                "review_reason": str(case["review_reason"]),
                "confidence": str(case["confidence"]),
                "geometry_adjustment": "trimmed_to_longest_scan_supported_run",
                "scan_support": support,
                "source_candidate_metadata": {
                    key: record[key]
                    for key in (
                        "candidate_source",
                        "source_kind",
                        "bend_angle",
                        "confidence",
                        "surface1_id",
                        "surface2_id",
                        "surface1_area",
                        "surface2_area",
                        "min_surface_gap",
                        "two_sided_transition_score",
                        "physical_role",
                    )
                    if record.get(key) is not None
                },
            }
        )

    drawing_path = (
        ROOT
        / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/drawings"
        / DRAWINGS[str(case["part_id"])]
    ).resolve()
    return {
        "schema_version": 1,
        "kind": "assistant_multiview_drawing_backed_bend_instance_labels",
        "case_token": str(case["case_token"]),
        "part_id": str(case["part_id"]),
        "role": str(case["role"]),
        "coordinate_frame": "scan_raw",
        "scan_path": str(scan_path.resolve()),
        "scan_sha256": _sha256(scan_path),
        "drawing_path": str(drawing_path),
        "drawing_sha256": _sha256(drawing_path),
        "candidate_source_path": source_path,
        "review_status": "assistant_multiview_complete_user_review_pending",
        "authoritative_ground_truth": False,
        "expected_count_source": "shaked_gmail_20260728_20260729",
        "sender_bend_count": expected,
        "accepted_bend_count": len(bends),
        "confidence": str(case["confidence"]),
        "manual_bends": bends,
        "marker_reviews": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    ledger = _read(args.ledger.resolve())
    inference_slice = _read(args.slice.resolve())
    scan_paths = {
        str(row["part_key"]): (ROOT / str(row["scan_path"])).resolve()
        for row in inference_slice.get("entries") or []
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in ledger.get("cases") or []:
        token = str(case["case_token"])
        if token not in scan_paths:
            raise ValueError(f"{token}: missing from inference slice")
        labels = build_case(
            case,
            scan_path=scan_paths[token],
            candidate_root=args.candidate_root.resolve(),
        )
        case_root = output_root / token
        case_root.mkdir(parents=True, exist_ok=True)
        output_path = case_root / "bend_instance_labels.json"
        output_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        manifest_cases.append(
            {
                "case_token": token,
                "part_id": labels["part_id"],
                "role": labels["role"],
                "sender_bend_count": labels["sender_bend_count"],
                "accepted_bend_count": labels["accepted_bend_count"],
                "confidence": labels["confidence"],
                "labels_path": str(output_path),
            }
        )

    expected_total = sum(int(row["sender_bend_count"]) for row in manifest_cases)
    accepted_total = sum(int(row["accepted_bend_count"]) for row in manifest_cases)
    if expected_total != accepted_total:
        raise ValueError(f"total mismatch: expected {expected_total}, accepted {accepted_total}")
    manifest = {
        "schema_version": 1,
        "kind": "assistant_multiview_drawing_backed_label_manifest",
        "review_status": "assistant_multiview_complete_user_review_pending",
        "authoritative_ground_truth": False,
        "case_count": len(manifest_cases),
        "expected_bend_count": expected_total,
        "accepted_bend_count": accepted_total,
        "cases": manifest_cases,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
