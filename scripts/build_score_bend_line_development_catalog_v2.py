#!/usr/bin/env python3
"""Build and score the 2026-08-30 non-authoritative development catalog.

This is intentionally a new lineage: it reads but never writes the v1 catalog,
requires a new freeze id, and emits separate legacy, today, and pooled scores.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_bend_line_gold_catalog import evaluate  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds  # noqa: E402


DEFAULT_LEGACY_CATALOG = (
    ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
)
TODAY = "2026-08-30"
TODAY_TRUTH_TIER = "codex_manual_multiview_scan_only"
EXPECTED_TODAY_CASE_COUNT = 28


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _rows(payload: dict[str, Any], *, name: str) -> list[dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"{name}: cases must be an array")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{name}: every case must be an object")
    return [dict(row) for row in rows]


def _bends(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("manual_bends", "bends"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    raise ValueError("label payload must contain manual_bends or bends")


def _validate_bends(case_id: str, bends: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(bends)
    ids: set[str] = set()
    for index, bend in enumerate(result, 1):
        bend_id = str(bend.get("bend_id") or "")
        if not bend_id or bend_id in ids:
            raise ValueError(f"{case_id}: missing or duplicate bend_id at row {index}")
        ids.add(bend_id)
        start, end = bend.get("line_start"), bend.get("line_end")
        if not (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == len(end) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in start + end)
            and start != end
        ):
            raise ValueError(f"{case_id}/{bend_id}: invalid finite nonzero 3D line")
    return result


def _prediction_map(payload: dict[str, Any]) -> dict[str, str]:
    if isinstance(payload.get("rows"), list):
        mapping: dict[str, str] = {}
        for row in payload["rows"]:
            if not isinstance(row, dict) or not row.get("case_id") or not row.get("prediction_path"):
                raise ValueError("prediction-map rows require case_id and prediction_path")
            case_id = str(row["case_id"])
            if case_id in mapping:
                raise ValueError(f"duplicate prediction-map case: {case_id}")
            mapping[case_id] = str(row["prediction_path"])
        return mapping
    if all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()):
        return dict(payload)
    raise ValueError("prediction map must be a case-to-path object or contain rows[]")


def build_catalog(
    legacy_path: Path,
    label_manifest_path: Path,
    *,
    freeze_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy_raw = legacy_path.read_bytes()
    legacy = json.loads(legacy_raw)
    manifest_raw = label_manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    legacy_sha = _sha256_bytes(legacy_raw)
    manifest_sha = _sha256_bytes(manifest_raw)

    if not freeze_id.strip():
        raise ValueError("freeze_id must be non-empty")
    forbidden_ids = {
        str(legacy.get(key))
        for key in ("freeze_id", "catalog_id", "source_snapshot_sha256")
        if legacy.get(key)
    }
    if freeze_id in forbidden_ids:
        raise ValueError("expanded catalog must not reuse a legacy freeze/snapshot id")
    if manifest.get("label_date") != TODAY:
        raise ValueError(f"label manifest label_date must be {TODAY}")
    if str(manifest.get("consensus_status") or "").upper() != "COMPLETE":
        raise ValueError("label manifest consensus_status must be COMPLETE")

    rows = _rows(manifest, name="label manifest")
    if len(rows) != EXPECTED_TODAY_CASE_COUNT:
        raise ValueError(f"label manifest requires exactly {EXPECTED_TODAY_CASE_COUNT} cases")
    filenames = [str(row.get("private_filename") or row.get("filename") or row.get("scan_name") or "") for row in rows]
    if any(not name for name in filenames) or filenames != sorted(filenames):
        raise ValueError("manifest cases must be in lexicographic private-filename order")
    expected_ids = [f"T{index:03d}" for index in range(1, EXPECTED_TODAY_CASE_COUNT + 1)]
    if [str(row.get("case_id") or "") for row in rows] != expected_ids:
        raise ValueError("case_id sequence must be exactly T001..T028")

    legacy_cases = copy.deepcopy(list(legacy.get("cases") or []))
    legacy_ids = {str(case["case_id"]) for case in legacy_cases}
    legacy_scan_hashes = {str(case["scan_sha256"]) for case in legacy_cases}
    today_cases: list[dict[str, Any]] = []
    seen_scan_hashes: set[str] = set()
    manifest_base = label_manifest_path.parent
    for row, filename in zip(rows, filenames):
        case_id = str(row["case_id"])
        for required in ("part_id", "scan_path", "scan_sha256", "source_labels_path"):
            if not row.get(required):
                raise ValueError(f"{case_id}: missing {required}")
        if str(row.get("consensus_status") or "").upper() != "COMPLETE":
            raise ValueError(f"{case_id}: consensus_status must be COMPLETE")
        if case_id in legacy_ids:
            raise ValueError(f"{case_id}: collides with legacy case id")
        scan_sha = str(row["scan_sha256"])
        if len(scan_sha) != 64 or any(char not in "0123456789abcdef" for char in scan_sha):
            raise ValueError(f"{case_id}: scan_sha256 must be lowercase SHA-256")
        if scan_sha in legacy_scan_hashes or scan_sha in seen_scan_hashes:
            raise ValueError(f"{case_id}: duplicate or legacy scan content")
        seen_scan_hashes.add(scan_sha)
        scan_path = _resolve(str(row["scan_path"]), manifest_base)
        if not scan_path.is_file() or _sha256_file(scan_path) != scan_sha:
            raise ValueError(f"{case_id}: scan file/hash mismatch")
        labels_path = _resolve(str(row["source_labels_path"]), manifest_base)
        if not labels_path.is_file():
            raise ValueError(f"{case_id}: label file missing")
        if row.get("source_labels_sha256") and _sha256_file(labels_path) != row["source_labels_sha256"]:
            raise ValueError(f"{case_id}: label file/hash mismatch")
        labels = _read_json(labels_path)
        if labels.get("case_id") is not None and str(labels["case_id"]) != case_id:
            raise ValueError(f"{case_id}: label case_id mismatch")
        if labels.get("scan_sha256") is not None and str(labels["scan_sha256"]) != scan_sha:
            raise ValueError(f"{case_id}: label scan_sha256 mismatch")
        bends = _validate_bends(case_id, _bends(labels))
        declared_count = row.get("bend_count")
        if declared_count is not None and int(declared_count) != len(bends):
            raise ValueError(f"{case_id}: declared bend_count mismatch")
        today_cases.append(
            {
                "case_id": case_id,
                "part_id": str(row["part_id"]),
                "scan_role": "2026-08-30_ai_multiview_consensus_development",
                "scan_path": str(scan_path),
                "scan_name": filename,
                "scan_sha256": scan_sha,
                "coordinate_frame": str(row.get("coordinate_frame") or labels.get("coordinate_frame") or "scan_raw"),
                "bend_count": len(bends),
                "bends": bends,
                "truth_tier": TODAY_TRUTH_TIER,
                "review_status": "complete",
                "consensus_status": "COMPLETE",
                "scoring_eligible": True,
                "promotion_eligible": False,
                "authoritative_ground_truth": False,
                "source_labels_path": str(labels_path),
                "source_manifest_path": str(label_manifest_path.resolve()),
            }
        )

    all_cases = legacy_cases + today_cases
    tier_counts: dict[str, int] = {}
    for case in all_cases:
        tier = str(case["truth_tier"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    snapshot_sha = _sha256_bytes(
        json.dumps(
            {"freeze_id": freeze_id, "legacy_sha256": legacy_sha, "label_manifest_sha256": manifest_sha},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    catalog = {
        "schema_version": 2,
        "kind": "sherman_qc_bend_line_development_catalog",
        "freeze_id": freeze_id,
        "source_snapshot_sha256": snapshot_sha,
        "lineage": {
            "legacy_catalog_path": str(legacy_path.resolve()),
            "legacy_catalog_sha256": legacy_sha,
            "legacy_source_snapshot_sha256": legacy.get("source_snapshot_sha256"),
            "today_label_manifest_path": str(label_manifest_path.resolve()),
            "today_label_manifest_sha256": manifest_sha,
            "legacy_catalog_mutated": False,
        },
        "semantic_contract": copy.deepcopy(legacy.get("semantic_contract")),
        "summary": {
            "case_count": len(all_cases),
            "legacy_case_count": len(legacy_cases),
            "today_case_count": len(today_cases),
            "bend_count": sum(int(case["bend_count"]) for case in all_cases),
            "legacy_bend_count": sum(int(case["bend_count"]) for case in legacy_cases),
            "today_bend_count": sum(int(case["bend_count"]) for case in today_cases),
            "tier_case_counts": dict(sorted(tier_counts.items())),
            "promotion_eligible_today_case_count": 0,
            "authoritative_today_case_count": 0,
        },
        "cohorts": {
            "legacy": [str(case["case_id"]) for case in legacy_cases],
            "today_2026_08_30": [case["case_id"] for case in today_cases],
        },
        "cases": all_cases,
    }
    audit = {
        "legacy_sha256_before": legacy_sha,
        "label_manifest_sha256": manifest_sha,
        "today_case_count": len(today_cases),
    }
    return catalog, audit


def score_catalog(
    catalog: dict[str, Any], prediction_map: dict[str, str], thresholds: Thresholds
) -> dict[str, dict[str, Any]]:
    legacy_ids = set(catalog["cohorts"]["legacy"])
    today_ids = set(catalog["cohorts"]["today_2026_08_30"])
    all_ids = legacy_ids | today_ids
    if set(prediction_map) != all_ids:
        missing = sorted(all_ids - set(prediction_map))
        extra = sorted(set(prediction_map) - all_ids)
        raise ValueError(f"prediction map must exactly cover expanded catalog; missing={missing}, extra={extra}")

    def subset(ids: set[str]) -> dict[str, Any]:
        return {**catalog, "cases": [case for case in catalog["cases"] if case["case_id"] in ids]}

    return {
        "legacy_only": evaluate(subset(legacy_ids), {key: prediction_map[key] for key in legacy_ids}, tier="all", thresholds=thresholds),
        "today_only": evaluate(subset(today_ids), {key: prediction_map[key] for key in today_ids}, tier="all", thresholds=thresholds),
        "pooled": evaluate(catalog, prediction_map, tier="all", thresholds=thresholds),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-catalog", type=Path, default=DEFAULT_LEGACY_CATALOG)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--prediction-map", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angle-deg", type=float, default=5.0)
    parser.add_argument("--offset-mm", type=float, default=4.0)
    parser.add_argument("--gold-overlap", type=float, default=0.60)
    parser.add_argument("--prediction-overlap", type=float, default=0.60)
    args = parser.parse_args()

    legacy = args.legacy_catalog.resolve()
    manifest = args.label_manifest.resolve()
    prediction_path = args.prediction_map.resolve()
    output = args.output_dir.resolve()
    catalog, audit = build_catalog(legacy, manifest, freeze_id=args.freeze_id)
    legacy_sha_after = _sha256_file(legacy)
    if legacy_sha_after != audit["legacy_sha256_before"]:
        raise RuntimeError("legacy v1 catalog changed during orchestration")
    predictions = {
        case_id: str(_resolve(path, prediction_path.parent))
        for case_id, path in _prediction_map(_read_json(prediction_path)).items()
    }
    scores = score_catalog(
        catalog,
        predictions,
        Thresholds(args.angle_deg, args.offset_mm, args.gold_overlap, args.prediction_overlap),
    )
    catalog_path = output / "bend_line_development_catalog_v2.json"
    _write(catalog_path, catalog)
    for cohort, report in scores.items():
        report["cohort"] = cohort
        report["promotion_eligible"] = False
        report["authoritative_ground_truth"] = False
        _write(output / f"baseline_accuracy_{cohort}.json", report)
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_2026_08_30_expanded_development_baseline_result",
        "freeze_id": args.freeze_id,
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256_file(catalog_path),
        "legacy_catalog_sha256_before": audit["legacy_sha256_before"],
        "legacy_catalog_sha256_after": legacy_sha_after,
        "legacy_catalog_byte_identical": legacy_sha_after == audit["legacy_sha256_before"],
        "label_manifest_sha256": audit["label_manifest_sha256"],
        "prediction_map_sha256": _sha256_file(prediction_path),
        "truth_tier_today": TODAY_TRUTH_TIER,
        "promotion_eligible": False,
        "authoritative_ground_truth": False,
        "metrics": {cohort: report["metrics"] for cohort, report in scores.items()},
    }
    _write(output / "baseline_accuracy_summary.json", result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
