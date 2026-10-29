"""Non-authorizing projection of legacy program-status declarations.

A digest in a JSON document is a reference, not proof that its bytes, candidate,
scope or approval were verified. V2 program status is a progress display only.
The future release verifier must be a separate trusted boundary; neither this
projection nor an eligibility checker may grant floor disposition authority.
This module has no runtime/native dependencies or implicit file access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _base() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "SHERMAN_QC_BEND_INSPECTION_READINESS",
        "operating_mode": "RESEARCH_ONLY",
        "production_use_allowed": False,
        "supervised_shadow_authorized": False,
        "autonomous_disposition_status": "NOT_AUTHORIZED",
        "evidence_verification": "NOT_PERFORMED",
        "requires_independent_verification": True,
        "physical_generalization_progress": False,
        "generalization_progress_indicator": {},
        "lane_status": {},
        "contract_soak": {},
    }


def unavailable_readiness() -> dict[str, Any]:
    return {
        **_base(),
        "status": "UNAVAILABLE_FAIL_CLOSED",
        "active_blockers": ["VERIFIED_PROGRAM_STATUS_UNAVAILABLE"],
        "detail": "Program status is unavailable or malformed; no release authority was verified.",
    }


def project_program_status(payload: Any) -> dict[str, Any]:
    """Project display metadata, never accept self-declared authorization.

    An explicit operator-reviewed shadow release also needs an external trusted
    authorization path. It cannot be enabled by inserting a field in this file.
    """
    if not isinstance(payload, dict) or payload.get("kind") != "SHERMAN_QC_F1_PROGRAM_STATUS_V2":
        return unavailable_readiness()
    blockers = payload.get("active_blockers", [])
    if not isinstance(blockers, list) or any(not isinstance(x, str) or not x.strip() for x in blockers):
        return unavailable_readiness()
    for field in ("generalization_progress_indicator", "lane_status", "contract_soak"):
        if field in payload and not isinstance(payload[field], dict):
            return unavailable_readiness()
    for field in ("production_promotion_authorized", "physical_generalization_progress", "supervised_shadow_authorized"):
        if field in payload and type(payload[field]) is not bool:
            return unavailable_readiness()
    try:
        # Reject non-JSON and non-finite nested display metadata, including when
        # called directly rather than through the JSON loader.
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return unavailable_readiness()
    requested = payload.get("production_promotion_authorized") is True or payload.get("supervised_shadow_authorized") is True
    status = "PROMOTION_EVIDENCE_INVALID_FAIL_CLOSED" if requested else "PROMOTION_BLOCKED"
    active_blockers = list(dict.fromkeys(blockers))
    if requested:
        active_blockers = ["PROMOTION_EVIDENCE_INVALID_FAIL_CLOSED", *active_blockers]
    return {
        **_base(),
        "status": status,
        "active_blockers": active_blockers,
        "physical_generalization_progress": payload.get("physical_generalization_progress") is True,
        "physical_generalization_progress_verified": False,
        "generalization_progress_indicator": payload.get("generalization_progress_indicator", {}),
        "lane_status": payload.get("lane_status", {}),
        "contract_soak": payload.get("contract_soak", {}),
        "detail": "Legacy program status is a declaration, not candidate-bound release verification.",
    }


def load_program_status(path: Path) -> dict[str, Any]:
    """Read only the explicitly supplied status file; never follow evidence references."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(1_048_577)
        if len(raw) > 1_048_576:
            return unavailable_readiness()
        payload = json.loads(raw)
        result = project_program_status(payload)
    except (OSError, ValueError, RecursionError):
        return unavailable_readiness()
    # This is byte identity for diagnostics, explicitly not evidence validation.
    result["program_status_sha256"] = hashlib.sha256(raw).hexdigest()
    return result
