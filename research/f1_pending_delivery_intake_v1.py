"""Validate and materialize a delivered V/X cohort into opaque custody.

The evaluator-only source manifest may contain identities and V/X provenance.
The public receipt contains only opaque IDs and geometry-integrity metadata.
This stage never grants qualification credit or treats sender counts as truth.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from research.f1_physical_intake_v2 import IntakeError, _ply_header


SCHEMA = "SHERMAN_QC_PENDING_DELIVERY_MATERIALIZATION_V1"
HEX = frozenset("0123456789abcdef")


def _source_filename(row: Mapping[str, Any]) -> str:
    stem = str(row.get("source_linked_stem") or row.get("part_key") or "").strip()
    role = str(row.get("role") or "")
    suffix = "V" if role == "V_good" else ("X" if role == "X_defective" else "")
    if not stem or not suffix:
        raise IntakeError("every delivery row requires a source stem and V/X role")
    return f"{stem} ({suffix}).ply"


def validate_correspondence(manifest: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    if audit.get("schema") != "SHERMAN_QC_SOURCE_DELIVERY_CORRESPONDENCE_AUDIT_V1":
        raise IntakeError("unexpected source correspondence schema")
    expected_rows = manifest.get("opaque_cases")
    audit_rows = audit.get("cases")
    if not isinstance(expected_rows, list) or not isinstance(audit_rows, list):
        raise IntakeError("correspondence cases must be lists")
    expected = {str(row.get("opaque_id")): _source_filename(row) for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise IntakeError("manifest opaque IDs must be unique")
    observed: dict[str, str] = {}
    link_hashes: set[str] = set()
    for row in audit_rows:
        if not isinstance(row, Mapping):
            raise IntakeError("correspondence row must be an object")
        opaque_id = str(row.get("opaque_id") or "")
        filename = str(row.get("source_filename") or "")
        digest = row.get("sharepoint_link_sha256")
        if opaque_id in observed:
            raise IntakeError("duplicate correspondence opaque_id")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in HEX for char in digest):
            raise IntakeError("sharepoint_link_sha256 must be lowercase hex64")
        if digest in link_hashes:
            raise IntakeError("duplicate SharePoint link hash")
        if any("http" in str(value).lower() for value in row.values()):
            raise IntakeError("raw links may not be persisted in correspondence audit")
        link_hashes.add(digest)
        observed[opaque_id] = filename
    if observed != expected:
        raise IntakeError("source correspondence does not match the frozen delivery manifest")
    return {
        "status": "PASS",
        "scan_count": len(observed),
        "unique_link_hash_count": len(link_hashes),
        "raw_sharepoint_links_persisted": False,
        "qualification_capacity_credit": 0,
    }


def _under(root: Path, relative: str, field: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise IntakeError(f"{field} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise IntakeError(f"{field} escapes its root") from exc
    return resolved


def inspect_delivery(
    manifest: Mapping[str, Any],
    *,
    download_root: Path,
    target_root: Path,
    consumed_scan_hashes: set[str] | frozenset[str],
) -> dict[str, Any]:
    if manifest.get("kind") != "sherman_qc_new_physical_scan_evaluator_only_intake":
        raise IntakeError("unexpected pending delivery manifest kind")
    rows = manifest.get("opaque_cases")
    if not isinstance(rows, list) or not rows:
        raise IntakeError("opaque_cases must be non-empty")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    public_rows = []
    private_rows = []
    copy_plan = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise IntakeError(f"opaque_cases[{index}] must be an object")
        opaque_id = str(row.get("opaque_id") or "").strip()
        if not opaque_id or opaque_id in seen_ids:
            raise IntakeError("opaque_id values must be unique and non-empty")
        seen_ids.add(opaque_id)
        filename = _source_filename(row)
        source = _under(download_root, filename, "source filename")
        target_relative = str(row.get("target_path") or "")
        target = _under(target_root, target_relative, "target_path")
        if not source.is_file():
            raise IntakeError(f"missing delivered scan for {opaque_id}: {filename}")
        if source.suffix.lower() != ".ply" or target.suffix.lower() != ".ply":
            raise IntakeError("source and target must be PLY files")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen_hashes:
            raise IntakeError("duplicate scan content within pending delivery")
        if digest in consumed_scan_hashes:
            raise IntakeError("pending delivery contains development-consumed scan content")
        seen_hashes.add(digest)
        header = _ply_header(source)
        already_materialized = False
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise IntakeError(f"opaque target conflict for {opaque_id}")
            already_materialized = True
        public_rows.append({
            "case_id": opaque_id,
            "scan_sha256": digest,
            "byte_count": source.stat().st_size,
            **header,
            "units": "mm",
            "truth_available_to_model_lane": False,
            "cad_available_to_scan_only_lane": False,
            "already_materialized": already_materialized,
        })
        private_rows.append({
            "case_id": opaque_id,
            "source_filename": filename,
            "target_path": target_relative,
            "scan_sha256": digest,
            "part_key": row.get("part_key"),
            "role": row.get("role"),
            "sender_count_non_authoritative": row.get("sender_count_non_authoritative"),
            "delivery_date": row.get("delivery_date"),
        })
        if not already_materialized:
            copy_plan.append((source, target))
    public_rows.sort(key=lambda item: item["case_id"])
    private_rows.sort(key=lambda item: item["case_id"])
    return {
        "public_receipt": {
            "schema": SCHEMA,
            "status": "VALIDATED_PENDING_OPAQUE_COPY",
            "scan_count": len(public_rows),
            "cases": public_rows,
            "qualification_capacity_credit": 0,
            "required_next_gate": "CUSTODIAN_METADATA_AND_INDEPENDENT_DUAL_REVIEW_LINE_TRUTH",
            "identity_fields_absent": [
                "source_filename", "part_key", "role", "sender_count_non_authoritative",
                "delivery_date", "expected_count", "gold", "prediction",
            ],
        },
        "evaluator_receipt": {
            "schema": SCHEMA,
            "status": "VALIDATED_PENDING_OPAQUE_COPY",
            "scan_count": len(private_rows),
            "cases": private_rows,
            "sender_counts_are_line_truth": False,
            "vx_roles_available_to_inference": False,
            "qualification_capacity_credit": 0,
        },
        "copy_plan": copy_plan,
    }


def materialize_copy_plan(copy_plan: list[tuple[Path, Path]]) -> None:
    """Copy only after the complete cohort has passed validation."""
    staged: list[tuple[str, Path]] = []
    try:
        for source, target in copy_plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            os.close(descriptor)
            shutil.copyfile(source, temporary)
            staged.append((temporary, target))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _target in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)
