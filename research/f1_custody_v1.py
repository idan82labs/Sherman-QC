"""Deterministic blind custody allocation for untouched physical F1 evidence."""

from __future__ import annotations

from itertools import combinations
import hashlib
import hmac
import json
from typing import Any, Mapping


SCHEMA = "F1_PHYSICAL_CUSTODY_INPUT_V1"
OUTPUT_SCHEMA = "F1_BLIND_ANNOTATION_ASSIGNMENT_V1"
HEX = frozenset("0123456789abcdef")


class CustodyError(ValueError):
    pass


def _hex64(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise CustodyError(f"{path} must be lowercase hex64")
    return value


def _role(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 64:
        raise CustodyError(f"{path} must be a non-empty role identifier")
    return value.strip()


def _stable(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def allocate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise CustodyError(f"schema must be {SCHEMA}")
    cohort = payload.get("cohort")
    if cohort != "UNTOUCHED_QUALIFICATION":
        raise CustodyError("only UNTOUCHED_QUALIFICATION may enter this allocator")
    roles = payload.get("roles")
    if not isinstance(roles, Mapping):
        raise CustodyError("roles must be an object")
    custodian = _role(roles.get("custodian"), "roles.custodian")
    adjudicator = _role(roles.get("adjudicator"), "roles.adjudicator")
    annotators_raw = roles.get("annotators")
    if not isinstance(annotators_raw, list) or len(annotators_raw) < 2:
        raise CustodyError("at least two annotators are required")
    annotators = sorted({_role(value, "roles.annotators") for value in annotators_raw})
    if len(annotators) != len(annotators_raw):
        raise CustodyError("annotator identifiers must be unique")
    implementers_raw = roles.get("model_implementers", [])
    if not isinstance(implementers_raw, list):
        raise CustodyError("roles.model_implementers must be a list")
    implementers = {_role(value, "roles.model_implementers") for value in implementers_raw}
    privileged = {custodian, adjudicator, *annotators}
    if len(privileged) != 2 + len(annotators):
        raise CustodyError("custodian, adjudicator, and annotators must be distinct")
    if privileged & implementers:
        raise CustodyError("model implementers may not hold annotation custody roles")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise CustodyError("cases must be a non-empty list")
    charter_sha = _hex64(payload.get("charter_sha256"), "charter_sha256")
    blinding_salt = _hex64(payload.get("blinding_salt_sha256"), "blinding_salt_sha256")
    normalized = []
    seen_scans = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CustodyError(f"cases[{index}] must be an object")
        if set(row) != {"scan_sha256", "morphology_group_sha256"}:
            raise CustodyError(f"cases[{index}] contains forbidden identity or truth fields")
        scan = _hex64(row.get("scan_sha256"), f"cases[{index}].scan_sha256")
        morphology = _hex64(row.get("morphology_group_sha256"), f"cases[{index}].morphology_group_sha256")
        if scan in seen_scans:
            raise CustodyError("duplicate scan hash")
        seen_scans.add(scan)
        case_id = hmac.new(bytes.fromhex(blinding_salt), bytes.fromhex(scan), hashlib.sha256).hexdigest()[:24]
        normalized.append({"case_id": case_id, "scan_sha256": scan, "morphology_group_sha256": morphology})
    normalized.sort(key=lambda row: row["case_id"])
    family_count = len({row["morphology_group_sha256"] for row in normalized})
    if len(normalized) < 30 or family_count < 15:
        public = {
            "schema": OUTPUT_SCHEMA,
            "status": "CORPUS_CAPACITY_ABSTAIN",
            "cohort": cohort,
            "charter_sha256": charter_sha,
            "inventory_scan_count": len(normalized),
            "inventory_morphology_family_count": family_count,
            "minimum_scan_count": 30,
            "minimum_morphology_family_count": 15,
            "assignments": [],
            "geometry_opened": 0,
        }
        return {
            "public_assignment": public,
            "custodian_only_case_map": {
                "schema": "F1_CUSTODIAN_ONLY_CASE_MAP_V1",
                "status": "CORPUS_CAPACITY_ABSTAIN",
                "custodian": custodian,
                "public_assignment_sha256": hashlib.sha256(_stable(public)).hexdigest(),
                "cases": [],
            },
        }
    loads = {annotator: 0 for annotator in annotators}
    assignments = []
    pairs = list(combinations(annotators, 2))
    for row in normalized:
        pair = min(
            pairs,
            key=lambda values: (
                max(loads[values[0]], loads[values[1]]),
                loads[values[0]] + loads[values[1]],
                hashlib.sha256(f"{row['case_id']}::{values[0]}::{values[1]}".encode()).hexdigest(),
            ),
        )
        for annotator in pair:
            loads[annotator] += 1
        assignments.append({
            "case_id": row["case_id"],
            "annotators": list(pair),
            "adjudicator": adjudicator,
            "prediction_access": False,
            "expected_count_access": False,
            "cad_access": False,
        })
    custody_map = [{
        "case_id": row["case_id"],
        "scan_sha256": row["scan_sha256"],
        "morphology_group_sha256": row["morphology_group_sha256"],
    } for row in normalized]
    public = {
        "schema": OUTPUT_SCHEMA,
        "status": "READY_FOR_BLIND_ANNOTATION",
        "cohort": cohort,
        "charter_sha256": charter_sha,
        "case_count": len(assignments),
        "morphology_family_count": family_count,
        "independent_annotations_per_case": 2,
        "assignments": assignments,
        "annotator_loads": loads,
        "blind_fields_absent": [
            "scan_sha256", "scan_path", "filename", "part_identity", "morphology_group_sha256",
            "expected_count", "prediction", "gold", "blinding_salt_sha256",
        ],
    }
    return {
        "public_assignment": public,
        "custodian_only_case_map": {
            "schema": "F1_CUSTODIAN_ONLY_CASE_MAP_V1",
            "custodian": custodian,
            "public_assignment_sha256": hashlib.sha256(_stable(public)).hexdigest(),
            "cases": custody_map,
        },
    }
