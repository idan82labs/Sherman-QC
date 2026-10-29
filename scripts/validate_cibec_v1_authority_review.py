#!/usr/bin/env python3
"""Validate two-authority acceptance of a CIBEC candidate review payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "output/research/cibec-v1-authority-candidate-v2"


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _line_ok(event: dict[str, Any]) -> bool:
    try:
        start = [float(item) for item in event["line_start_mm"]]
        end = [float(item) for item in event["line_end_mm"]]
    except (KeyError, TypeError, ValueError):
        return False
    if len(start) != 3 or len(end) != 3 or not all(math.isfinite(item) for item in start + end):
        return False
    return math.dist(start, end) >= 8.0


def validate(pack: Path) -> tuple[int, dict[str, Any]]:
    charter = json.loads((pack / "CIBEC_V1_CHARTER_CANDIDATE.json").read_text())
    answers = json.loads((pack / "S01_S14_EXPECTATIONS_CANDIDATE.json").read_text())
    attestation = json.loads((pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").read_text())
    payload_sha = hashlib.sha256(_bytes({"charter": charter, "answer_key": answers})).hexdigest()
    blockers: list[str] = []
    expected_ids = [f"S{index:02d}" for index in range(1, 15)]
    if list((answers.get("fixtures") or {}).keys()) != expected_ids:
        blockers.append("FIXTURE_SET_NOT_EXACT_S01_S14")
    for fixture_id, row in (answers.get("fixtures") or {}).items():
        events = row.get("events") if isinstance(row, dict) else None
        if row.get("case_status") != "RESOLVED" or not isinstance(events, list) or not events:
            blockers.append(f"{fixture_id}_SEMANTICS_NOT_RESOLVED")
            continue
        if any(not isinstance(event, dict) or not _line_ok(event) for event in events):
            blockers.append(f"{fixture_id}_INVALID_MARKER")
    if attestation.get("review_payload_sha256") != payload_sha:
        blockers.append("ATTESTATION_PAYLOAD_HASH_STALE")
    for field in ("manufacturing_authority", "countersigning_authority"):
        signature = attestation.get(field)
        if not isinstance(signature, dict) or not str(signature.get("name") or "").strip():
            blockers.append(f"{field.upper()}_NAME_REQUIRED")
            continue
        if signature.get("decision") != "ACCEPT":
            blockers.append(f"{field.upper()}_ACCEPTANCE_REQUIRED")
        if signature.get("signed_payload_sha256") != payload_sha:
            blockers.append(f"{field.upper()}_PAYLOAD_SIGNATURE_REQUIRED")
    manufacturing = attestation.get("manufacturing_authority")
    countersigning = attestation.get("countersigning_authority")
    if (
        isinstance(manufacturing, dict)
        and isinstance(countersigning, dict)
        and str(manufacturing.get("name") or "").strip()
        and str(manufacturing.get("name") or "").strip().casefold()
        == str(countersigning.get("name") or "").strip().casefold()
    ):
        blockers.append("AUTHORITY_IDENTITIES_MUST_BE_DISTINCT")
    report = {
        "schema": "CIBEC_V1_AUTHORITY_REVIEW_VALIDATION",
        "review_payload_sha256": payload_sha,
        "fixture_count": len(answers.get("fixtures") or {}),
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
        "blockers": blockers,
        "status": "READY_TO_MATERIALIZE_SIGNED_CIBEC" if not blockers else "AWAITING_AUTHORITY_REVIEW",
    }
    return (0 if not blockers else 2), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    code, report = validate(args.pack.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
