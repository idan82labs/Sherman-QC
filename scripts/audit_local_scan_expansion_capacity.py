#!/usr/bin/env python3
"""Inventory local scan candidates and quantify the evidence gap to 50 scans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_local_capacity(
    *,
    registry: Mapping[str, Any],
    registry_sha256: str,
    scan_root: Path,
    target_scan_count: int = 50,
) -> dict[str, Any]:
    if registry.get("schema_version") != 2:
        raise ValueError("scan-capacity audit requires truth-registry schema version 2")
    if target_scan_count <= 0:
        raise ValueError("target scan count must be positive")
    scan_root = scan_root.resolve()
    if not scan_root.is_dir():
        raise ValueError(f"scan root does not exist: {scan_root}")

    paths: dict[Path, dict[str, Any]] = {}
    missing_registry_sources: list[str] = []
    for record in registry.get("records", []):
        if not isinstance(record, Mapping):
            continue
        source = record.get("source_geometry") or {}
        raw_path = source.get("resolved_scan_path")
        if not raw_path:
            missing_registry_sources.append(str(record.get("record_id") or "<unknown>"))
            continue
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            missing_registry_sources.append(str(record.get("record_id") or "<unknown>"))
            continue
        item = paths.setdefault(path, {"records": []})
        item["records"].append(
            {
                "record_id": str(record.get("record_id") or ""),
                "part_key": str(record.get("part_key") or ""),
                "truth_status": str(record.get("truth_status") or ""),
                "release_metric_eligible": bool(record.get("release_metric_eligible")),
            }
        )

    for path in scan_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".ply":
            paths.setdefault(path.resolve(), {"records": []})

    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=str):
        digest = _sha256(path)
        group = groups.setdefault(
            digest,
            {
                "geometry_sha256": digest,
                "paths": [],
                "records": [],
            },
        )
        try:
            relative = str(path.relative_to(scan_root))
        except ValueError:
            relative = str(path)
        group["paths"].append(
            {
                "path": str(path),
                "relative_to_scan_root": relative,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )
        group["records"].extend(paths[path]["records"])

    entries = []
    for digest in sorted(groups):
        group = groups[digest]
        records = group["records"]
        release_eligible = any(record["release_metric_eligible"] for record in records)
        if release_eligible:
            role = "release_eligible_scan_geometry"
        elif records:
            role = "registered_scan_candidate_requires_adjudication"
        else:
            role = "unregistered_ply_candidate_requires_provenance_and_adjudication"
        entries.append(
            {
                **group,
                "candidate_role": role,
                "release_metric_eligible": release_eligible,
                "part_keys": sorted({record["part_key"] for record in records if record["part_key"]}),
                "duplicate_path_count": len(group["paths"]),
            }
        )

    eligible = [entry for entry in entries if entry["release_metric_eligible"]]
    registered_pending = [
        entry
        for entry in entries
        if entry["candidate_role"] == "registered_scan_candidate_requires_adjudication"
    ]
    unregistered = [
        entry
        for entry in entries
        if entry["candidate_role"]
        == "unregistered_ply_candidate_requires_provenance_and_adjudication"
    ]
    optimistic_capacity = len(entries)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "capacity_audit_not_truth",
        "policy": (
            "Non-release entries are candidates only. They count toward capacity only after "
            "measured-scan provenance and independent typed adjudication are established."
        ),
        "target_release_scan_count": target_scan_count,
        "source": {
            "truth_registry_sha256": registry_sha256,
            "scan_root": str(scan_root),
        },
        "counts": {
            "discovered_file_paths": len(paths),
            "unique_geometry_sha256": len(entries),
            "release_eligible_unique_geometry": len(eligible),
            "registered_pending_unique_geometry": len(registered_pending),
            "unregistered_ply_unique_geometry": len(unregistered),
            "duplicate_file_paths": len(paths) - len(entries),
            "optimistic_local_capacity_if_all_candidates_are_valid": optimistic_capacity,
            "minimum_new_external_scans_even_at_optimistic_local_capacity": max(
                0, target_scan_count - optimistic_capacity
            ),
            "additional_verified_scans_needed_from_current_state": max(
                0, target_scan_count - len(eligible)
            ),
            "missing_registry_source_records": len(missing_registry_sources),
        },
        "missing_registry_source_record_ids": sorted(missing_registry_sources),
        "entries": entries,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--target-scan-count", type=int, default=50)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    payload = audit_local_capacity(
        registry=registry,
        registry_sha256=_sha256(args.registry),
        scan_root=args.scan_root,
        target_scan_count=args.target_scan_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
