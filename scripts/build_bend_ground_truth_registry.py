#!/usr/bin/env python3
"""Build the authoritative, typed bend ground-truth registry.

The registry keeps inference labels separate from scoring truth and makes
semantic conflicts explicit. It never treats mail/import counts as adjudicated
ground truth merely because the related files are available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "data" / "bend_corpus" / "dataset_inventory.json"
DEFAULT_MANUAL = ROOT / "data" / "bend_corpus" / "annotations" / "f1_manual_validation_20260425.json"
DEFAULT_OUTPUT = ROOT / "data" / "bend_corpus" / "annotations" / "ground_truth_registry.json"
DEFAULT_INSTANCE_LABEL_ROOT = ROOT / "output"
DEFAULT_SOURCE_ROOTS = (
    ROOT.parent / "source-assets",
    Path.home() / "Downloads" / "02_CAD_QC_3D",
    Path.home() / "Downloads" / "08_Archives_Transfers",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_key(part_key: Any, scan_name: Any) -> str:
    return f"{str(part_key or '').strip()}::{Path(str(scan_name or '')).name}"


def _source_index(source_roots: Iterable[Path]) -> Dict[str, list[Path]]:
    index: Dict[str, list[Path]] = {}
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path.resolve())
    return index


def _resolve_source_path(raw_path: Any, name: Any, index: Mapping[str, list[Path]]) -> Optional[Path]:
    direct = Path(str(raw_path or "")).expanduser()
    if direct.is_file():
        return direct.resolve()
    candidates = list(index.get(Path(str(name or "")).name) or ())
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0] if candidates else None


def _load_instance_geometry(root: Path) -> Dict[str, Dict[str, Any]]:
    geometry: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return geometry
    for label_path in sorted(root.glob("bend_label_*/bend_instance_labels.json")):
        meta_path = label_path.with_name("bend_inspection_meta.json")
        if not meta_path.exists():
            continue
        try:
            labels = _load_json(label_path)
            metadata = _load_json(meta_path)
        except Exception:
            continue
        seed_case = dict(metadata.get("seed_case") or {})
        source_scan = Path(str(seed_case.get("source_scan") or ""))
        part_key = str(metadata.get("part_id") or "")
        if not part_key or not source_scan.name:
            continue
        manual_bends = list(labels.get("manual_bends") or ())
        key = _record_key(part_key, source_scan.name)
        geometry[key] = {
            "schema_version": int(labels.get("schema_version") or 1),
            "kind": str(labels.get("kind") or "sherman_qc_bend_instance_labels"),
            "coordinate_frame": str(labels.get("coordinate_frame") or ""),
            "coordinate_frame_provenance": seed_case.get("coordinate_frame_provenance"),
            "review_status": str(labels.get("review_status") or ""),
            "labeling_mode": seed_case.get("labeling_mode"),
            "source_file": str(label_path.resolve()),
            "source_scan": str(source_scan.resolve()) if source_scan.is_file() else str(source_scan),
            "source_sha256": seed_case.get("source_sha256"),
            "bend_count": len(manual_bends),
            "bends": manual_bends,
        }
    return geometry


def _manual_record(raw: Mapping[str, Any], legacy_expected: Optional[int]) -> Dict[str, Any]:
    observed = _as_int(raw.get("human_observed_bend_count"))
    full = _as_int(raw.get("human_full_part_bend_count"))
    breakdown = dict(raw.get("feature_breakdown") or {})
    raised = _as_int(breakdown.get("raised_form_features")) or 0
    issues = []
    if observed is not None and observed < 0:
        issues.append("negative_observed_count")
    if full is not None and full < 0:
        issues.append("negative_full_part_count")
    if observed is not None and full is not None and observed > full:
        issues.append("observed_exceeds_full_part")
    if legacy_expected is not None and full is not None and legacy_expected != full:
        issues.append("legacy_total_conflicts_with_manual")
    excluded = bool(raw.get("exclude_from_accuracy"))
    if excluded:
        status = "excluded"
    elif observed is None:
        status = "needs_adjudication"
        issues.append("missing_manual_observed_count")
    elif any(issue in {"negative_observed_count", "negative_full_part_count", "observed_exceeds_full_part"} for issue in issues):
        status = "needs_adjudication"
    else:
        status = "manual_verified"
    return {
        "record_id": _record_key(raw.get("part_key"), raw.get("scan_name")),
        "part_key": str(raw.get("part_key") or ""),
        "scan_name": Path(str(raw.get("scan_name") or "")).name,
        "scan_path": str(raw.get("scan_path") or ""),
        "scan_state": "partial" if str(raw.get("track") or "").lower() == "partial" else "full",
        "part_family": str(raw.get("part_family") or "conventional_unknown"),
        "count_target": "visible_observed_conventional_bends",
        "counts": {
            "observed_conventional_bends": observed,
            "full_part_conventional_bends": full,
            "raised_form_features": raised,
        },
        "truth_status": status,
        "release_metric_eligible": status == "manual_verified",
        "legacy_expected_total": legacy_expected,
        "legacy_expected_disputed": bool(raw.get("legacy_expected_disputed")),
        "validation_issues": sorted(set(issues)),
        "exclusion_reason": raw.get("exclusion_reason"),
        "notes": str(raw.get("notes") or ""),
        "provenance": {
            "source_type": "manual_visual_review",
            "source_file": str(DEFAULT_MANUAL),
            "reviewed_at": "2026-04-25",
        },
    }


def _legacy_records(parts: Iterable[Mapping[str, Any]], manual_keys: set[str]) -> list[Dict[str, Any]]:
    records = []
    for part in parts:
        part_key = str(part.get("part_key") or "")
        legacy_expected = _as_int(part.get("expected_bends_override"))
        for scan in part.get("scan_files") or ():
            scan_name = Path(str(scan.get("name") or "")).name
            key = _record_key(part_key, scan_name)
            if key in manual_keys:
                continue
            records.append(
                {
                    "record_id": key,
                    "part_key": part_key,
                    "scan_name": scan_name,
                    "scan_path": str(scan.get("path") or ""),
                    "scan_state": str(scan.get("state") or "unknown"),
                    "part_family": "unknown",
                    "count_target": "visible_observed_conventional_bends",
                    "counts": {
                        "observed_conventional_bends": None,
                        "full_part_conventional_bends": legacy_expected,
                        "raised_form_features": 0,
                    },
                    "truth_status": "legacy_unadjudicated",
                    "release_metric_eligible": False,
                    "legacy_expected_total": legacy_expected,
                    "legacy_expected_disputed": False,
                    "validation_issues": ["manual_instance_truth_missing"],
                    "exclusion_reason": None,
                    "notes": "Imported corpus/mail truth; requires visual or drawing adjudication.",
                    "provenance": {
                        "source_type": "legacy_corpus_inventory",
                        "source_file": str(DEFAULT_INVENTORY),
                    },
                }
            )
    return records


def build_registry(
    inventory: Mapping[str, Any],
    manual: Mapping[str, Any],
    *,
    source_roots: Iterable[Path] = (),
    instance_geometry_by_record: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    parts = list(inventory.get("parts") or ())
    parts_by_key = {str(part.get("part_key") or ""): part for part in parts}
    legacy_by_part = {
        str(part.get("part_key") or ""): _as_int(part.get("expected_bends_override"))
        for part in parts
    }
    manual_records = [
        _manual_record(raw, legacy_by_part.get(str(raw.get("part_key") or "")))
        for raw in manual.get("records") or ()
    ]
    manual_keys = {str(record["record_id"]) for record in manual_records}
    records = [*manual_records, *_legacy_records(parts, manual_keys)]
    instance_geometry_by_record = instance_geometry_by_record or {}
    for record in records:
        instance_geometry = instance_geometry_by_record.get(str(record["record_id"]))
        if instance_geometry:
            record["instance_geometry"] = dict(instance_geometry)
            observed = (record.get("counts") or {}).get("observed_conventional_bends")
            instance_count = int(instance_geometry.get("bend_count") or 0)
            if observed is not None and int(observed) != instance_count:
                record["validation_issues"] = sorted(
                    set(record.get("validation_issues") or ()) | {"instance_count_conflicts_with_observed"}
                )
                record["truth_status"] = "needs_adjudication"
                record["release_metric_eligible"] = False
    resolved_roots = tuple(Path(path).expanduser().resolve() for path in source_roots)
    index = _source_index(resolved_roots)
    if resolved_roots:
        for record in records:
            part = parts_by_key.get(str(record.get("part_key") or ""), {})
            scan_path = _resolve_source_path(record.get("scan_path"), record.get("scan_name"), index)
            cad_paths = []
            for cad in part.get("cad_files") or ():
                resolved = _resolve_source_path(cad.get("path"), cad.get("name"), index)
                if resolved is not None and resolved not in cad_paths:
                    cad_paths.append(resolved)
            record["source_geometry"] = {
                "scan_available": scan_path is not None,
                "resolved_scan_path": None if scan_path is None else str(scan_path),
                "scan_size_bytes": None if scan_path is None else scan_path.stat().st_size,
                "cad_available": bool(cad_paths),
                "resolved_cad_paths": [str(path) for path in cad_paths],
            }
            if scan_path is not None:
                record["scan_path"] = str(scan_path)
            else:
                record["validation_issues"] = sorted(
                    set(record.get("validation_issues") or ()) | {"scan_source_unavailable"}
                )
                record["release_metric_eligible"] = False
    records.sort(key=lambda record: (str(record["part_key"]), str(record["scan_name"])))
    status_counts: Dict[str, int] = {}
    for record in records:
        status = str(record["truth_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semantic_contract": {
            "primary_count": "visible_observed_conventional_bends",
            "raised_form_features": "reported_separately_not_counted_as_conventional",
            "rolled_or_rounded_features": "typed_family_not_implicitly_counted_as_conventional",
            "partial_scan_rule": "never_substitute_full_part_total_for_visible_observed_count",
        },
        "release_policy": {
            "eligible_truth_statuses": ["manual_verified", "dual_review_verified"],
            "requires_instance_geometry_for_visual_localization_metrics": True,
        },
        "summary": {
            "record_count": len(records),
            "release_metric_eligible_count": sum(bool(record["release_metric_eligible"]) for record in records),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "source_roots": [str(path) for path in resolved_roots],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--manual", default=str(DEFAULT_MANUAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--instance-label-root", default=str(DEFAULT_INSTANCE_LABEL_ROOT))
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Search root for relocated scan/CAD files. May be repeated.",
    )
    args = parser.parse_args()
    source_roots = [Path(value) for value in args.source_root] or list(DEFAULT_SOURCE_ROOTS)
    registry = build_registry(
        _load_json(Path(args.inventory)),
        _load_json(Path(args.manual)),
        source_roots=source_roots,
        instance_geometry_by_record=_load_instance_geometry(Path(args.instance_label_root)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(registry["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
