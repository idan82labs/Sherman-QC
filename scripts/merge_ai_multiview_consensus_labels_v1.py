#!/usr/bin/env python3
"""Merge three prediction-hidden AI bend-line reviews into development labels."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


TODAY = "2026-08-30"
ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = [f"T{index:03d}" for index in range(1, 29)]
TRUTH_TIER = "codex_manual_multiview_scan_only"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_scan(value: str, manifest_base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (manifest_base / path).resolve()


def _bends(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("manual_bends", "bends"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    raise ValueError("review case/label must expose manual_bends or bends")


def _point(row: Any, *, context: str) -> tuple[float, float, float]:
    if not isinstance(row, list) or len(row) != 3:
        raise ValueError(f"{context}: point must have three coordinates")
    values = tuple(float(value) for value in row)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{context}: point coordinates must be finite")
    return values


def _line(row: dict[str, Any], *, context: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    start = _point(row.get("line_start"), context=context)
    end = _point(row.get("line_end"), context=context)
    if start == end:
        raise ValueError(f"{context}: zero-length line")
    return (start, end) if start <= end else (end, start)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _line_distance(
    left: tuple[tuple[float, float, float], tuple[float, float, float]],
    right: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    return max(_distance(left[0], right[0]), _distance(left[1], right[1]))


def _mean_line(lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]]) -> tuple[list[float], list[float]]:
    return (
        [sum(line[0][axis] for line in lines) / len(lines) for axis in range(3)],
        [sum(line[1][axis] for line in lines) / len(lines) for axis in range(3)],
    )


def _case_rows(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if manifest.get("label_date") != TODAY:
        raise ValueError(f"{path}: label_date must be {TODAY}")
    if str(manifest.get("review_status") or manifest.get("consensus_status") or "").upper() != "COMPLETE":
        raise ValueError(f"{path}: review_status must be COMPLETE")
    if manifest.get("prediction_hidden") is not True:
        raise ValueError(f"{path}: prediction_hidden must be true")
    if manifest.get("independent_review") is not True:
        raise ValueError(f"{path}: independent_review must be true")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: cases must be an object array")
    if [str(row.get("case_id") or "") for row in rows] != CASE_IDS:
        raise ValueError(f"{path}: cases must be complete and ordered T001..T028")
    return [dict(row) for row in rows]


def _case_bends(row: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    if str(row.get("review_status") or row.get("consensus_status") or "").upper() != "COMPLETE":
        raise ValueError(f"{row.get('case_id')}: reviewer case is incomplete")
    if isinstance(row.get("manual_bends"), list) or isinstance(row.get("bends"), list):
        return _bends(row)
    label_value = row.get("source_labels_path") or row.get("label_path")
    if not label_value:
        raise ValueError(f"{row.get('case_id')}: missing reviewed bends or label path")
    label_path = _resolve(str(label_value), manifest_path.parent)
    if not label_path.is_file():
        raise ValueError(f"{row.get('case_id')}: reviewer label file missing")
    expected = row.get("source_labels_sha256") or row.get("label_sha256")
    if expected and _sha(label_path) != expected:
        raise ValueError(f"{row.get('case_id')}: reviewer label hash mismatch")
    return _bends(_read(label_path))


def _merge_case(
    case_id: str,
    reviewer_bends: list[list[dict[str, Any]]],
    *,
    tolerance_mm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    lines = [
        [_line(row, context=f"{case_id}/reviewer{reviewer + 1}/line{index + 1}") for index, row in enumerate(rows)]
        for reviewer, rows in enumerate(reviewer_bends)
    ]
    candidates = []
    for indices in itertools.product(*(range(len(rows)) for rows in lines)):
        chosen = [lines[reviewer][index] for reviewer, index in enumerate(indices)]
        distances = [_line_distance(chosen[left], chosen[right]) for left, right in ((0, 1), (0, 2), (1, 2))]
        if max(distances, default=0.0) <= tolerance_mm:
            mean_start, mean_end = _mean_line(chosen)
            candidates.append((sum(distances), tuple(mean_start + mean_end), indices, chosen, max(distances)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used = [set(), set(), set()]
    accepted = []
    for _, _, indices, chosen, maximum in candidates:
        if any(index in used[reviewer] for reviewer, index in enumerate(indices)):
            continue
        for reviewer, index in enumerate(indices):
            used[reviewer].add(index)
        start, end = _mean_line(chosen)
        accepted.append(
            {
                "line_start": start,
                "line_end": end,
                "confidence": "three_reviewer_consensus",
                "reviewer_support_count": 3,
                "maximum_endpoint_disagreement_mm": round(maximum, 9),
                "source_reviewer_line_indices": list(indices),
            }
        )
    unmatched = [sorted(set(range(len(rows))) - used[index]) for index, rows in enumerate(lines)]
    if any(unmatched):
        return [], {
            "case_id": case_id,
            "reason": "NO_COMPLETE_ONE_TO_ONE_THREE_REVIEWER_CONSENSUS",
            "reviewer_line_counts": [len(rows) for rows in lines],
            "matched_consensus_count": len(accepted),
            "unmatched_line_indices": unmatched,
        }
    accepted.sort(key=lambda row: tuple(row["line_start"] + row["line_end"]))
    for index, row in enumerate(accepted, 1):
        row["bend_id"] = f"{case_id}-B{index:03d}"
    return accepted, None


def merge_reviewer_manifests(
    manifest_paths: list[Path], *, tolerance_mm: float = 2.0, verify_scan_files: bool = True
) -> dict[str, Any]:
    if len(manifest_paths) != 3:
        raise ValueError("exactly three independent reviewer manifests are required")
    loaded = [(path.resolve(), _read(path.resolve())) for path in manifest_paths]
    reviewer_ids = [str(payload.get("reviewer_id") or "") for _, payload in loaded]
    if any(not reviewer_id for reviewer_id in reviewer_ids) or len(set(reviewer_ids)) != 3:
        raise ValueError("three distinct non-empty reviewer_id values are required")
    rows_by_reviewer = [_case_rows(payload, path) for path, payload in loaded]
    unresolved = []
    merged_cases = []
    for position, case_id in enumerate(CASE_IDS):
        rows = [reviewer_rows[position] for reviewer_rows in rows_by_reviewer]
        scan_sha = str(rows[0].get("scan_sha256") or "")
        scan_paths = [_resolve_scan(str(row.get("scan_path") or ""), loaded[index][0].parent) for index, row in enumerate(rows)]
        if any(str(row.get("scan_sha256") or "") != scan_sha for row in rows[1:]) or len(set(scan_paths)) != 1:
            raise ValueError(f"{case_id}: reviewer scan path/hash disagreement")
        if len(scan_sha) != 64 or any(char not in "0123456789abcdef" for char in scan_sha):
            raise ValueError(f"{case_id}: invalid lowercase scan SHA-256")
        scan_path = scan_paths[0]
        if verify_scan_files and (not scan_path.is_file() or _sha(scan_path) != scan_sha):
            raise ValueError(f"{case_id}: scan file/hash mismatch")
        reviewer_bends = [_case_bends(row, loaded[index][0]) for index, row in enumerate(rows)]
        bends, failure = _merge_case(case_id, reviewer_bends, tolerance_mm=tolerance_mm)
        if failure:
            unresolved.append(failure)
            continue
        merged_cases.append(
            {
                "case_id": case_id,
                "part_id": scan_path.stem,
                "private_filename": scan_path.name,
                "scan_path": str(scan_path),
                "scan_sha256": scan_sha,
                "coordinate_frame": str(rows[0].get("coordinate_frame") or "scan_raw"),
                "bend_count": len(bends),
                "manual_bends": bends,
                "consensus_status": "COMPLETE",
                "truth_tier": TRUTH_TIER,
                "promotion_eligible": False,
                "authoritative_ground_truth": False,
            }
        )
    return {
        "schema_version": 1,
        "kind": "sherman_qc_three_ai_multiview_consensus_merge",
        "label_date": TODAY,
        "reviewer_ids": reviewer_ids,
        "reviewer_manifest_sha256": [_sha(path) for path, _ in loaded],
        "prediction_hidden": True,
        "tolerance_mm": tolerance_mm,
        "consensus_status": "COMPLETE" if not unresolved else "UNRESOLVED",
        "truth_tier": TRUTH_TIER,
        "promotion_eligible": False,
        "authoritative_ground_truth": False,
        "resolved_case_count": len(merged_cases),
        "unresolved_case_count": len(unresolved),
        "unresolved_cases": unresolved,
        "cases": merged_cases,
    }


def load_adjudicated_manifest(adjudicator_path: Path, reviewer_paths: list[Path]) -> dict[str, Any]:
    """Validate a fourth-agent adjudication of the three blinded first passes."""

    if len(reviewer_paths) != 3:
        raise ValueError("adjudication must bind exactly three first-pass reviewer manifests")
    reviewer_paths = [path.resolve() for path in reviewer_paths]
    payload = _read(adjudicator_path.resolve())
    if payload.get("label_date") != TODAY or str(payload.get("review_status") or "").upper() != "COMPLETE":
        raise ValueError("adjudicator manifest must be complete and dated 2026-08-30")
    if payload.get("prediction_hidden") is not True or payload.get("independent_review") is not False:
        raise ValueError("adjudicator must truthfully bind prediction-hidden non-independent adjudication")
    if payload.get("authoritative_ground_truth") is not False or payload.get("promotion_eligible") is not False:
        raise ValueError("adjudicator cannot grant authority or promotion eligibility")
    expected_hashes = [_sha(path) for path in reviewer_paths]
    declared_hashes = payload.get("reviewer_manifest_sha256")
    if isinstance(declared_hashes, dict):
        declared_hashes = [declared_hashes.get(key) for key in ("A", "B", "C")]
    if declared_hashes != expected_hashes:
        raise ValueError("adjudicator reviewer manifest hash binding mismatch")
    rows = payload.get("cases")
    if not isinstance(rows, list) or [str(row.get("case_id") or "") for row in rows] != CASE_IDS:
        raise ValueError("adjudicator cases must be complete and ordered T001..T028")
    cases = []
    for row in rows:
        case_id = str(row["case_id"])
        if str(row.get("review_status") or "").upper() != "COMPLETE":
            raise ValueError(f"{case_id}: adjudication is incomplete")
        scan_path = _resolve_scan(str(row.get("scan_path") or ""), adjudicator_path.resolve().parent)
        scan_sha = str(row.get("scan_sha256") or "")
        if not scan_path.is_file() or _sha(scan_path) != scan_sha:
            raise ValueError(f"{case_id}: adjudicated scan path/hash mismatch")
        bends = _bends(row)
        seen_ids = set()
        for index, bend in enumerate(bends, 1):
            _line(bend, context=f"{case_id}/adjudicated-line-{index}")
            bend_id = str(bend.get("bend_id") or "")
            if not bend_id or bend_id in seen_ids:
                raise ValueError(f"{case_id}: missing or duplicate adjudicated bend_id")
            seen_ids.add(bend_id)
        if int(row.get("bend_count", -1)) != len(bends):
            raise ValueError(f"{case_id}: adjudicated bend_count mismatch")
        cases.append(
            {
                "case_id": case_id,
                "part_id": scan_path.stem,
                "private_filename": scan_path.name,
                "scan_path": str(scan_path),
                "scan_sha256": scan_sha,
                "coordinate_frame": str(row.get("coordinate_frame") or "source_scan_raw"),
                "bend_count": len(bends),
                "manual_bends": bends,
                "consensus_status": "COMPLETE",
            }
        )
    return {
        "schema_version": 1,
        "kind": "sherman_qc_three_ai_reviews_plus_fourth_adjudication",
        "label_date": TODAY,
        "reviewer_ids": list(payload.get("reviewer_ids_adjudicated") or []),
        "reviewer_manifest_sha256": expected_hashes,
        "adjudicator_id": str(payload.get("reviewer_id") or ""),
        "adjudicator_manifest_sha256": _sha(adjudicator_path.resolve()),
        "prediction_hidden": True,
        "consensus_status": "COMPLETE",
        "truth_tier": TRUTH_TIER,
        "promotion_eligible": False,
        "authoritative_ground_truth": False,
        "resolved_case_count": len(cases),
        "unresolved_case_count": 0,
        "unresolved_cases": [],
        "cases": cases,
    }


def materialize_complete_manifest(merge: dict[str, Any], output_dir: Path) -> Path:
    if merge.get("consensus_status") != "COMPLETE" or merge.get("unresolved_cases"):
        raise ValueError("cannot materialize a complete label manifest from unresolved consensus")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in merge["cases"]:
        label_path = output_dir / "labels" / f"{case['case_id']}.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_payload = {
            "schema_version": 1,
            "kind": "sherman_qc_ai_multiview_consensus_bend_labels",
            "case_id": case["case_id"],
            "scan_sha256": case["scan_sha256"],
            "coordinate_frame": case["coordinate_frame"],
            "consensus_status": "COMPLETE",
            "manual_bends": case["manual_bends"],
            "truth_tier": TRUTH_TIER,
            "promotion_eligible": False,
            "authoritative_ground_truth": False,
        }
        label_path.write_text(json.dumps(label_payload, indent=2) + "\n", encoding="utf-8")
        manifest_cases.append(
            {
                key: case[key]
                for key in ("case_id", "part_id", "private_filename", "scan_path", "scan_sha256", "coordinate_frame", "bend_count", "consensus_status")
            }
            | {"source_labels_path": str(label_path.resolve()), "source_labels_sha256": _sha(label_path)}
        )
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_ai_multiview_consensus_label_manifest",
        "label_date": TODAY,
        "consensus_status": "COMPLETE",
        "prediction_hidden": True,
        "reviewer_ids": merge["reviewer_ids"],
        "reviewer_manifest_sha256": merge.get("reviewer_manifest_sha256"),
        "adjudicator_id": merge.get("adjudicator_id"),
        "adjudicator_manifest_sha256": merge.get("adjudicator_manifest_sha256"),
        "truth_tier": TRUTH_TIER,
        "promotion_eligible": False,
        "authoritative_ground_truth": False,
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    path = output_dir / "complete_label_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-manifest", type=Path, action="append", required=True)
    parser.add_argument("--adjudicator-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance-mm", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merge = (
        load_adjudicated_manifest(args.adjudicator_manifest, args.reviewer_manifest)
        if args.adjudicator_manifest
        else merge_reviewer_manifests(args.reviewer_manifest, tolerance_mm=args.tolerance_mm)
    )
    report_path = output_dir / "consensus_merge_report.json"
    report_path.write_text(json.dumps(merge, indent=2) + "\n", encoding="utf-8")
    if merge["consensus_status"] == "COMPLETE":
        manifest_path = materialize_complete_manifest(merge, output_dir)
        print(str(manifest_path))
    else:
        print(json.dumps({"status": "UNRESOLVED", "unresolved_cases": merge["unresolved_cases"]}, indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
