#!/usr/bin/env python3
"""Build a prediction-hidden review queue for pending local scan geometries."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


FORBIDDEN_REVIEW_KEY_FRAGMENTS = (
    "f1",
    "prediction",
    "expected",
    "legacy_expected",
    "holdout",
    "split",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _drawing_index(roots: Iterable[Path]) -> list[Path]:
    drawings = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if not resolved.is_dir():
            continue
        for path in resolved.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                canonical = path.resolve()
                if canonical not in seen:
                    seen.add(canonical)
                    drawings.append(canonical)
    return sorted(drawings, key=str)


def _drawings_for_parts(drawings: Sequence[Path], part_keys: Sequence[str]) -> list[dict[str, Any]]:
    matches = []
    for path in drawings:
        searchable = " ".join(path.parts).casefold()
        if part_keys and any(part_key.casefold() in searchable for part_key in part_keys):
            matches.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return matches


def _registry_by_record_id(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(record.get("record_id") or ""): record
        for record in registry.get("records", [])
        if isinstance(record, Mapping) and record.get("record_id")
    }


def _verify_geometry_entry(entry: Mapping[str, Any]) -> None:
    declared = str(entry.get("geometry_sha256") or "")
    paths = entry.get("paths") or []
    if not declared or not paths:
        raise ValueError("capacity entry lacks geometry identity or source paths")
    for item in paths:
        path = Path(str(item.get("path") or "")).resolve()
        if not path.is_file():
            raise ValueError(f"pending review source is missing: {path}")
        actual = _sha256(path)
        if actual != declared:
            raise ValueError(f"pending review geometry hash drift: {path}: {actual} != {declared}")


def build_review_queue(
    *,
    capacity: Mapping[str, Any],
    capacity_sha256: str,
    registry: Mapping[str, Any],
    registry_sha256: str,
    drawing_roots: Iterable[Path] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    if capacity.get("status") != "capacity_audit_not_truth":
        raise ValueError("review queue requires a non-truth scan-capacity audit")
    if registry.get("schema_version") != 2:
        raise ValueError("review queue requires truth-registry schema version 2")
    drawings = _drawing_index(drawing_roots)
    registry_by_id = _registry_by_record_id(registry)
    queue_entries = []

    for source_entry in capacity.get("entries", []):
        if source_entry.get("release_metric_eligible") is True:
            continue
        _verify_geometry_entry(source_entry)
        geometry_sha256 = str(source_entry["geometry_sha256"])
        part_keys = sorted(str(value) for value in source_entry.get("part_keys", []) if value)
        source_records = []
        withheld_cad = []
        for record_summary in source_entry.get("records", []):
            record_id = str(record_summary.get("record_id") or "")
            record = registry_by_id.get(record_id, {})
            source_records.append(
                {
                    "record_id": record_id,
                    "truth_status": str(record_summary.get("truth_status") or ""),
                    "release_metric_eligible": False,
                }
            )
            for raw_path in (record.get("source_geometry") or {}).get("resolved_cad_paths") or []:
                path = Path(str(raw_path)).resolve()
                if path.is_file() and str(path) not in {item["path"] for item in withheld_cad}:
                    withheld_cad.append(
                        {
                            "path": str(path),
                            "sha256": _sha256(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )
        matched_drawings = _drawings_for_parts(drawings, part_keys)
        queue_entries.append(
            {
                "candidate_id": f"scan-candidate-{geometry_sha256[:16]}",
                "geometry_sha256": geometry_sha256,
                "candidate_role": source_entry.get("candidate_role"),
                "part_keys": part_keys,
                "identity_status": "resolved" if len(part_keys) == 1 else "requires_part_identity_review",
                "source_paths": list(source_entry.get("paths") or []),
                "source_records": source_records,
                "drawing_evidence": matched_drawings,
                "drawing_status": "available" if matched_drawings else "missing",
                "cad_evidence_withheld_until_reconciliation": withheld_cad,
                "adjudication_status": "pending_independent_review",
            }
        )
    queue_entries.sort(key=lambda item: item["candidate_id"])

    drawing_statuses = Counter(entry["drawing_status"] for entry in queue_entries)
    identity_statuses = Counter(entry["identity_status"] for entry in queue_entries)
    queue = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "prediction_hidden_scan_adjudication_queue",
        "metric_eligibility": "none_until_independent_dual_review_and_provenance_confirmation",
        "review_boundary": {
            "f1_outputs_included": False,
            "legacy_counts_included": False,
            "holdout_membership_included": False,
            "cad_visible_on_first_pass": False,
            "required_first_pass_sources": "raw_geometry_and_manufacturing_drawing_only",
        },
        "source_identity": {
            "capacity_audit_sha256": capacity_sha256,
            "truth_registry_sha256": registry_sha256,
        },
        "counts": {
            "pending_unique_geometries": len(queue_entries),
            "drawing_statuses": dict(sorted(drawing_statuses.items())),
            "identity_statuses": dict(sorted(identity_statuses.items())),
        },
        "entries": queue_entries,
    }

    template_entries = []
    for entry in queue_entries:
        template_entries.append(
            {
                "candidate_id": entry["candidate_id"],
                "geometry_sha256": entry["geometry_sha256"],
                "part_key": entry["part_keys"][0] if len(entry["part_keys"]) == 1 else None,
                "source_paths": entry["source_paths"],
                "drawing_evidence": entry["drawing_evidence"],
                "review_status": "pending",
                "measured_scan_provenance": "unknown",
                "scan_state": "unknown",
                "formed_stage_state": "unknown",
                "part_family": None,
                "visible_observed_conventional_bend_count": None,
                "bend_instance_annotations_reference_mesh_raw": [],
                "excluded_feature_notes": [],
                "ambiguity_notes": [],
            }
        )
    template = {
        "schema_version": 1,
        "queue_sha256": None,
        "reviewer_id": None,
        "reviewed_at": None,
        "review_contract": {
            "independent_of_other_reviewer": True,
            "f1_outputs_seen": False,
            "legacy_counts_seen": False,
            "holdout_membership_seen": False,
            "cad_evidence_seen": False,
            "source": "raw_geometry_and_manufacturing_drawing_only",
        },
        "allowed_measured_scan_provenance": [
            "measured_scan_confirmed",
            "cad_tessellation_confirmed",
            "unknown",
        ],
        "entries": template_entries,
    }
    return queue, template


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_REVIEW_KEY_FRAGMENTS):
                raise ValueError(f"forbidden review key emitted: {key}")
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-audit", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--drawing-root", action="append", default=[], type=Path)
    parser.add_argument("--queue-output", required=True, type=Path)
    parser.add_argument("--template-output", required=True, type=Path)
    args = parser.parse_args(argv)
    capacity = json.loads(args.capacity_audit.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    queue, template = build_review_queue(
        capacity=capacity,
        capacity_sha256=_sha256(args.capacity_audit),
        registry=registry,
        registry_sha256=_sha256(args.registry),
        drawing_roots=args.drawing_root,
    )
    _assert_no_forbidden_keys(template["entries"])
    args.queue_output.parent.mkdir(parents=True, exist_ok=True)
    args.queue_output.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    template["queue_sha256"] = _sha256(args.queue_output)
    args.template_output.parent.mkdir(parents=True, exist_ok=True)
    args.template_output.write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(queue["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
