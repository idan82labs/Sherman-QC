"""Fail-closed preparation of untouched physical scans under split custody.

The public result contains only opaque case identifiers and geometry-integrity
metadata.  Morphology, manufacturing-instance, session, and source-path
mappings remain in the custodian-only result.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "F1_PHYSICAL_INTAKE_CUSTODIAN_V2"
HEX = frozenset("0123456789abcdef")
CASE_FIELDS = {
    "relative_path",
    "morphology_group_sha256",
    "manufacturing_instance_sha256",
    "scan_session_sha256",
    "calibration_sha256",
    "units",
    "coverage_reviewed",
    "quality_reviewed",
    "relationship_to_prior_design",
}
RELATIONSHIPS = {"NOVEL_MORPHOLOGY", "RELATED_BUT_UNSEEN_INSTANCE"}


class IntakeError(ValueError):
    pass


def _hex64(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise IntakeError(f"{path} must be lowercase hex64")
    return value


def _role(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 64:
        raise IntakeError(f"{path} must be a non-empty role identifier")
    return value.strip()


def _ply_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        lines = []
        total = 0
        while True:
            raw = handle.readline()
            if not raw:
                raise IntakeError(f"{path.name}: truncated PLY header")
            total += len(raw)
            if total > 1_048_576:
                raise IntakeError(f"{path.name}: PLY header exceeds 1 MiB")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise IntakeError(f"{path.name}: PLY header is not ASCII") from exc
            lines.append(line)
            if line == "end_header":
                break
    if not lines or lines[0] != "ply":
        raise IntakeError(f"{path.name}: missing PLY magic")
    formats = [line.split() for line in lines if line.startswith("format ")]
    if len(formats) != 1 or len(formats[0]) < 2 or formats[0][1] not in {"ascii", "binary_little_endian"}:
        raise IntakeError(f"{path.name}: unsupported or ambiguous PLY format")
    vertex_count = None
    in_vertex = False
    vertex_properties: list[str] = []
    for line in lines:
        fields = line.split()
        if fields[:1] == ["element"]:
            in_vertex = len(fields) == 3 and fields[1] == "vertex"
            if in_vertex:
                try:
                    vertex_count = int(fields[2])
                except ValueError as exc:
                    raise IntakeError(f"{path.name}: invalid vertex count") from exc
        elif in_vertex and len(fields) >= 3 and fields[0] == "property":
            vertex_properties.append(fields[-1])
    if vertex_count is None or vertex_count <= 0:
        raise IntakeError(f"{path.name}: positive vertex count required")
    if vertex_count > 50_000_000:
        raise IntakeError(f"{path.name}: vertex count exceeds bounded intake limit")
    if not {"x", "y", "z"}.issubset(vertex_properties):
        raise IntakeError(f"{path.name}: x/y/z vertex properties required")
    return {"ply_format": formats[0][1], "vertex_count": vertex_count}


def prepare(
    payload: Any,
    *,
    intake_root: Path,
    consumed_scan_hashes: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise IntakeError(f"schema must be {SCHEMA}")
    if payload.get("cohort") != "UNTOUCHED_QUALIFICATION":
        raise IntakeError("only UNTOUCHED_QUALIFICATION may enter this intake")
    custodian = _role(payload.get("custodian_role"), "custodian_role")
    implementers_raw = payload.get("model_implementer_roles")
    if not isinstance(implementers_raw, list) or not implementers_raw:
        raise IntakeError("model_implementer_roles must be a non-empty list")
    implementers = {_role(value, "model_implementer_roles") for value in implementers_raw}
    if custodian in implementers:
        raise IntakeError("custodian may not be a model implementer")
    salt = _hex64(payload.get("blinding_salt_sha256"), "blinding_salt_sha256")
    protocol = _hex64(payload.get("acquisition_protocol_sha256"), "acquisition_protocol_sha256")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise IntakeError("cases must be a non-empty list")
    root = intake_root.resolve()
    public_cases = []
    private_cases = []
    seen_scan_hashes: set[str] = set()
    seen_sessions: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != CASE_FIELDS:
            raise IntakeError(f"cases[{index}] contains missing or forbidden fields")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise IntakeError(f"cases[{index}].relative_path must be relative")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise IntakeError(f"cases[{index}].relative_path escapes intake root") from exc
        if source.suffix.lower() != ".ply" or not source.is_file():
            raise IntakeError(f"cases[{index}] must reference an existing PLY file")
        if row.get("units") != "mm":
            raise IntakeError(f"cases[{index}].units must be mm")
        if row.get("coverage_reviewed") is not True or row.get("quality_reviewed") is not True:
            raise IntakeError(f"cases[{index}] requires completed coverage and quality review")
        relationship = row.get("relationship_to_prior_design")
        if relationship not in RELATIONSHIPS:
            raise IntakeError(f"cases[{index}] has invalid relationship declaration")
        morphology = _hex64(row.get("morphology_group_sha256"), f"cases[{index}].morphology_group_sha256")
        instance = _hex64(row.get("manufacturing_instance_sha256"), f"cases[{index}].manufacturing_instance_sha256")
        session = _hex64(row.get("scan_session_sha256"), f"cases[{index}].scan_session_sha256")
        calibration = _hex64(row.get("calibration_sha256"), f"cases[{index}].calibration_sha256")
        if session in seen_sessions:
            raise IntakeError("scan_session_sha256 must be unique per scan")
        seen_sessions.add(session)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen_scan_hashes:
            raise IntakeError("duplicate scan content within intake")
        if digest in consumed_scan_hashes:
            raise IntakeError("scan content already exists in development-consumed custody")
        seen_scan_hashes.add(digest)
        header = _ply_header(source)
        case_id = hmac.new(bytes.fromhex(salt), bytes.fromhex(digest), hashlib.sha256).hexdigest()[:24]
        public_cases.append({
            "case_id": case_id,
            "scan_sha256": digest,
            "byte_count": source.stat().st_size,
            **header,
            "units": "mm",
            "coverage_reviewed": True,
            "quality_reviewed": True,
            "truth_available_to_model_lane": False,
            "cad_available_to_scan_only_lane": False,
        })
        private_cases.append({
            "case_id": case_id,
            "relative_path": relative,
            "scan_sha256": digest,
            "morphology_group_sha256": morphology,
            "manufacturing_instance_sha256": instance,
            "scan_session_sha256": session,
            "calibration_sha256": calibration,
            "relationship_to_prior_design": relationship,
        })
    public_cases.sort(key=lambda row: row["case_id"])
    private_cases.sort(key=lambda row: row["case_id"])
    families: dict[str, set[str]] = {}
    for row in private_cases:
        families.setdefault(row["morphology_group_sha256"], set()).add(row["manufacturing_instance_sha256"])
    qualified_families = {family for family, instances in families.items() if len(instances) >= 2}
    qualifying_instances = {
        row["manufacturing_instance_sha256"]
        for row in private_cases
        if row["morphology_group_sha256"] in qualified_families
    }
    ready = len(qualifying_instances) >= 30 and len(qualified_families) >= 15
    public = {
        "schema": "F1_PHYSICAL_INTAKE_PUBLIC_V2",
        "status": "READY_FOR_CUSTODIAN_SEAL" if ready else "INTAKE_ACCEPTED_CAPACITY_PENDING",
        "cohort": "UNTOUCHED_QUALIFICATION",
        "acquisition_protocol_sha256": protocol,
        "scan_count": len(public_cases),
        "cases": public_cases,
        "identity_fields_absent": [
            "relative_path", "filename", "part_identity", "morphology_group_sha256",
            "manufacturing_instance_sha256", "scan_session_sha256", "calibration_sha256",
            "expected_count", "gold", "prediction", "blinding_salt_sha256",
        ],
    }
    private = {
        "schema": "F1_PHYSICAL_INTAKE_CUSTODIAN_RECEIPT_V2",
        "status": public["status"],
        "custodian_role": custodian,
        "scan_count": len(private_cases),
        "morphology_family_count": len(families),
        "qualified_two_instance_family_count": len(qualified_families),
        "qualifying_independent_instance_count": len(qualifying_instances),
        "minimum_scan_and_instance_count": 30,
        "minimum_two_instance_family_count": 15,
        "cases": private_cases,
    }
    return {"public_manifest": public, "custodian_receipt": private}
