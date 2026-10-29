#!/usr/bin/env python3
"""Build resolved blind benchmark manifests from typed ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data" / "bend_corpus" / "annotations" / "ground_truth_registry.json"
DEFAULT_PLY_OUTPUT = ROOT / "data" / "bend_corpus" / "evaluation_slices" / "blind_ply_typed_release.json"
DEFAULT_CAD_OUTPUT = ROOT / "data" / "bend_corpus" / "evaluation_slices" / "blind_cad_ply_typed_release.json"
RELEASE_COUNT_TARGET = "visible_observed_conventional_bends"
ALLOWED_RELEASE_TRUTH_STATUSES = {"manual_verified", "dual_review_verified"}
ALLOWED_RELEASE_PROVENANCE_TYPES = {
    "manual_visual_review",
    "independent_dual_review_consensus",
}


def _validate_registry_contract(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != 2:
        raise ValueError("blind benchmark requires ground-truth registry schema version 2")
    semantic_contract = registry.get("semantic_contract") or {}
    if semantic_contract.get("primary_count") != RELEASE_COUNT_TARGET:
        raise ValueError("ground-truth registry primary-count semantic contract is invalid")


def _validate_release_record(record: Mapping[str, Any]) -> tuple[str, int, list[str]]:
    record_id = str(record.get("record_id") or "<unknown>")
    status = str(record.get("truth_status") or "")
    if status not in ALLOWED_RELEASE_TRUTH_STATUSES:
        raise ValueError(f"release-eligible record has unapproved truth status: {record_id}: {status}")
    if record.get("count_target") != RELEASE_COUNT_TARGET:
        raise ValueError(f"release-eligible record has wrong count target: {record_id}")
    source_type = str((record.get("provenance") or {}).get("source_type") or "")
    if source_type not in ALLOWED_RELEASE_PROVENANCE_TYPES:
        raise ValueError(
            f"release-eligible record has unapproved provenance: {record_id}: {source_type}"
        )
    observed = (record.get("counts") or {}).get("observed_conventional_bends")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise ValueError(f"release-eligible record has invalid observed count: {record_id}")
    source = record.get("source_geometry") or {}
    scan_path = str(source.get("resolved_scan_path") or "")
    if source.get("scan_available") is not True or not scan_path or not Path(scan_path).is_file():
        raise ValueError(f"release-eligible record has no resolved scan source: {record_id}")
    cad_paths = [
        str(path)
        for path in source.get("resolved_cad_paths") or ()
        if Path(str(path)).is_file()
    ]
    return scan_path, observed, cad_paths


def build_blind_slices(registry: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    _validate_registry_contract(registry)
    ply_entries = []
    cad_entries = []
    for record in registry.get("records") or ():
        if not bool(record.get("release_metric_eligible")):
            continue
        scan_path, observed, cad_paths = _validate_release_record(record)
        expectation = {
            "label_semantics": RELEASE_COUNT_TARGET,
            "expected_bend_count": int(observed),
            "expected_conventional_bend_count": int(observed),
            "state": str(record.get("scan_state") or "unknown"),
            "scan_limited": str(record.get("scan_state") or "") == "partial",
            "known_whole_part_total_context": (record.get("counts") or {}).get("full_part_conventional_bends"),
            "expected_raised_form_feature_count": (record.get("counts") or {}).get("raised_form_features", 0),
            "truth_record_id": record.get("record_id"),
        }
        entry = {
            "part_key": str(record.get("part_key") or ""),
            "scan_path": str(scan_path),
            "expectation": expectation,
            "audit_tags": ["blind", "typed_truth", str(record.get("scan_state") or "unknown")],
        }
        ply_entries.append(entry)
        if cad_paths:
            cad_entries.append({**entry, "cad_path": cad_paths[0], "inference_mode": "blind_cad_ply"})
    key = lambda entry: (entry["part_key"], Path(entry["scan_path"]).name)
    ply_entries.sort(key=key)
    cad_entries.sort(key=key)
    base = {
        "schema_version": 1,
        "truth_registry_schema_version": registry.get("schema_version"),
        "target": RELEASE_COUNT_TARGET,
    }
    return (
        {**base, "name": "blind_ply_typed_release", "entries": ply_entries},
        {**base, "name": "blind_cad_ply_typed_release", "inference_mode": "blind_cad_ply", "entries": cad_entries},
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--ply-output", type=Path, default=DEFAULT_PLY_OUTPUT)
    parser.add_argument("--cad-output", type=Path, default=DEFAULT_CAD_OUTPUT)
    args = parser.parse_args(argv)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    ply, cad = build_blind_slices(registry)
    for path, payload in ((args.ply_output, ply), (args.cad_output, cad)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ply_entries": len(ply["entries"]), "cad_ply_entries": len(cad["entries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
