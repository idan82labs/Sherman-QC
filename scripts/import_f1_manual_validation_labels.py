#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE_DIR = ROOT / "data/bend_corpus/evaluation_slices"
DEFAULT_ANNOTATION_PATH = ROOT / "data/bend_corpus/annotations/f1_manual_validation_20260425.json"


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


def _is_special_or_disputed(label: Mapping[str, Any]) -> bool:
    notes = str(label.get("notes") or "").lower()
    return bool(
        label.get("exclude_from_accuracy")
        or label.get("legacy_expected_disputed")
        or label.get("feature_breakdown")
        or label.get("part_family")
        or "problematic" in notes
        or "poor" in notes
        or "rolled" in notes
        or "מעורגל" in notes
        or "raised" in notes
    )


def _scan_key(part_key: Any, scan_name: Any) -> Tuple[str, str]:
    return (str(part_key or ""), Path(str(scan_name or "")).name)


def _build_scan_path_index(slice_paths: Sequence[Path]) -> Dict[Tuple[str, str], str]:
    index: Dict[Tuple[str, str], str] = {}
    for slice_path in slice_paths:
        if not slice_path.exists():
            continue
        payload = _load_json(slice_path)
        for entry in payload.get("entries") or ():
            scan_path = str(entry.get("scan_path") or "")
            if not scan_path:
                continue
            key = _scan_key(entry.get("part_key"), Path(scan_path).name)
            index.setdefault(key, scan_path)
    return index


def _metadata_scan_paths(parts_dir: Path) -> Dict[Tuple[str, str], str]:
    index: Dict[Tuple[str, str], str] = {}
    for metadata_path in sorted(parts_dir.glob("*/metadata.json")):
        payload = _load_json(metadata_path)
        part_key = str(payload.get("part_key") or metadata_path.parent.name)
        for scan in payload.get("scan_files") or ():
            scan_path = str(scan.get("path") or "")
            if scan_path:
                index.setdefault(_scan_key(part_key, Path(scan_path).name), scan_path)
    return index


def _default_slice_paths() -> List[Path]:
    return [
        DEFAULT_SLICE_DIR / "zero_shot_known_count_plys.json",
        DEFAULT_SLICE_DIR / "zero_shot_partial_total_bends.json",
        DEFAULT_SLICE_DIR / "zero_shot_small_tight_bends.json",
        DEFAULT_SLICE_DIR / "zero_shot_repeated_near_90.json",
    ]


def _feature_breakdown(label: Mapping[str, Any], conventional: Optional[int]) -> Dict[str, Any]:
    raw = dict(label.get("feature_breakdown") or {})
    if conventional is not None and "conventional_bends" not in raw:
        raw["conventional_bends"] = conventional
    if "counted_target" not in raw:
        raw["counted_target"] = "conventional_bends"
    return raw


def _audit_tags(label: Mapping[str, Any], *, special: bool) -> List[str]:
    tags = ["manual_validation", "conventional_bend_count"]
    track = str(label.get("track") or "")
    if track == "partial":
        tags.extend(["partial", "observed_visible_bends", "scan_limited"])
    elif track == "known":
        tags.extend(["known_count", "full_scan"])
    if label.get("exclude_from_accuracy"):
        tags.append("excluded_from_clean_accuracy")
    if label.get("legacy_expected_disputed"):
        tags.append("legacy_expected_disputed")
    if label.get("feature_breakdown"):
        tags.append("feature_breakdown")
    if label.get("part_family"):
        tags.append(str(label["part_family"]))
    if special:
        tags.append("special_or_disputed")
    return sorted(dict.fromkeys(tags))


def _entry_from_label(label: Mapping[str, Any], scan_path_index: Mapping[Tuple[str, str], str]) -> Optional[Dict[str, Any]]:
    conventional = _as_int(label.get("human_observed_bend_count"))
    if conventional is None and not label.get("exclude_from_accuracy"):
        return None
    part_key = str(label.get("part_key") or "")
    scan_name = Path(str(label.get("scan_name") or "")).name
    scan_path = scan_path_index.get(_scan_key(part_key, scan_name))
    if not scan_path:
        scan_path = str(label.get("scan_path") or scan_name)
    feature_breakdown = _feature_breakdown(label, conventional)
    raised_count = _as_int(feature_breakdown.get("raised_form_features"))
    expected_total_features = None
    if conventional is not None and raised_count is not None:
        expected_total_features = conventional + raised_count
    full_context = _as_int(label.get("human_full_part_bend_count"))
    special = _is_special_or_disputed(label)
    label_semantics = "visible_observed_conventional_bends" if label.get("track") == "partial" else "full_scan_conventional_bends"
    expectation: Dict[str, Any] = {
        "label_semantics": label_semantics,
        "manual_validation_source": "f1_manual_validation_20260425",
    }
    if conventional is not None:
        expectation.update({
            "expected_bend_count": conventional,
            "expected_conventional_bend_count": conventional,
        })
    if label.get("track") == "partial":
        expectation["state"] = "partial"
        expectation["scan_limited"] = True
        expectation["known_whole_part_total_context"] = full_context
    if raised_count is not None:
        expectation["expected_raised_form_feature_count"] = raised_count
        if expected_total_features is not None:
            expectation["expected_total_feature_count"] = expected_total_features
    if label.get("part_family"):
        expectation["part_family"] = label.get("part_family")
    if label.get("exclude_from_accuracy"):
        expectation["exclude_from_accuracy"] = True
        expectation["exclusion_reason"] = label.get("exclusion_reason")
    if label.get("legacy_expected_disputed"):
        expectation["legacy_expected_disputed"] = True
    return {
        "part_key": part_key,
        "scan_path": scan_path,
        "expectation": expectation,
        "audit_tags": _audit_tags(label, special=special),
        "manual_validation": {
            "scan_name": scan_name,
            "algorithm_range": list(label.get("algorithm_range") or ()),
            "algorithm_observed_prediction": label.get("algorithm_observed_prediction"),
            "algorithm_exact": label.get("algorithm_exact"),
            "feature_breakdown": feature_breakdown,
            "notes": label.get("notes") or "",
            "contact_sheet": label.get("contact_sheet"),
        },
    }


def _annotation_record(label: Mapping[str, Any], scan_path_index: Mapping[Tuple[str, str], str]) -> Dict[str, Any]:
    part_key = str(label.get("part_key") or "")
    scan_name = Path(str(label.get("scan_name") or "")).name
    return {
        "part_key": part_key,
        "scan_name": scan_name,
        "scan_path": scan_path_index.get(_scan_key(part_key, scan_name), str(label.get("scan_path") or "")),
        "track": label.get("track"),
        "human_observed_bend_count": label.get("human_observed_bend_count"),
        "human_full_part_bend_count": label.get("human_full_part_bend_count"),
        "feature_breakdown": label.get("feature_breakdown"),
        "part_family": label.get("part_family"),
        "legacy_expected_disputed": bool(label.get("legacy_expected_disputed")),
        "exclude_from_accuracy": bool(label.get("exclude_from_accuracy")),
        "exclusion_reason": label.get("exclusion_reason"),
        "algorithm_range": list(label.get("algorithm_range") or ()),
        "algorithm_observed_prediction": label.get("algorithm_observed_prediction"),
        "algorithm_exact": label.get("algorithm_exact"),
        "notes": label.get("notes") or "",
        "contact_sheet": label.get("contact_sheet"),
    }


def build_manual_validation_payload(
    *,
    labels_payload: Mapping[str, Any],
    scan_path_index: Mapping[Tuple[str, str], str],
) -> Dict[str, Any]:
    labels = list(labels_payload.get("labels") or ())
    records = [_annotation_record(label, scan_path_index) for label in labels]
    entries = [
        entry
        for entry in (_entry_from_label(label, scan_path_index) for label in labels)
        if entry is not None
    ]
    clean_known = [
        entry
        for entry in entries
        if "known_count" in entry["audit_tags"]
        and "special_or_disputed" not in entry["audit_tags"]
        and "excluded_from_clean_accuracy" not in entry["audit_tags"]
    ]
    partial_observed = [
        entry
        for entry in entries
        if "partial" in entry["audit_tags"] and "excluded_from_clean_accuracy" not in entry["audit_tags"]
    ]
    special = [
        entry
        for entry in entries
        if "special_or_disputed" in entry["audit_tags"] or "excluded_from_clean_accuracy" in entry["audit_tags"]
    ]
    return {
        "annotation": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": labels_payload.get("generated_at"),
            "schema_version": 1,
            "label_semantics": "conventional bends are the primary count target; raised/form and rolled/rounded features are explicit metadata.",
            "records": records,
        },
        "slices": {
            "f1_manual_clean_known_count_plys": {
                "name": "f1_manual_clean_known_count_plys",
                "description": "Manual-validated clean full-scan conventional bend count anchors. Excludes rolled/special/bad-scan/disputed-label cases.",
                "entries": clean_known,
            },
            "f1_manual_partial_observed_bends": {
                "name": "f1_manual_partial_observed_bends",
                "description": "Manual-validated partial/prelim scan visible conventional bend counts. These are observed scan-limited targets, not whole-part totals.",
                "entries": partial_observed,
            },
            "f1_manual_special_cases": {
                "name": "f1_manual_special_cases",
                "description": "Manual-validated special/disputed/excluded cases for diagnostics, not clean accuracy promotion.",
                "entries": special,
            },
        },
    }


def write_manual_validation_outputs(
    *,
    payload: Mapping[str, Any],
    annotation_path: Path,
    slice_dir: Path,
) -> Dict[str, str]:
    _write_json(annotation_path, payload["annotation"])
    paths = {"annotation_path": str(annotation_path)}
    for name, slice_payload in dict(payload["slices"]).items():
        path = slice_dir / f"{name}.json"
        _write_json(path, slice_payload)
        paths[name] = str(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Import F1 manual validation labels into repo-local annotations and corrected evaluation slices.")
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--annotation-json", default=str(DEFAULT_ANNOTATION_PATH))
    parser.add_argument("--slice-dir", default=str(DEFAULT_SLICE_DIR))
    parser.add_argument("--source-slice-json", action="append", default=[])
    args = parser.parse_args()

    labels_path = Path(args.labels_json).expanduser().resolve()
    slice_dir = Path(args.slice_dir).resolve()
    source_slices = [Path(path).expanduser().resolve() for path in args.source_slice_json] or _default_slice_paths()
    scan_path_index = {}
    scan_path_index.update(_metadata_scan_paths(ROOT / "data/bend_corpus/parts"))
    scan_path_index.update(_build_scan_path_index(source_slices))
    payload = build_manual_validation_payload(labels_payload=_load_json(labels_path), scan_path_index=scan_path_index)
    paths = write_manual_validation_outputs(
        payload=payload,
        annotation_path=Path(args.annotation_json).resolve(),
        slice_dir=slice_dir,
    )
    print(json.dumps({"paths": paths, "counts": {name: len(value["entries"]) for name, value in payload["slices"].items()}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
