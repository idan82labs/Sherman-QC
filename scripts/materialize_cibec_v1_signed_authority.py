#!/usr/bin/env python3
"""Materialize official CIBEC artifacts only after two exact-hash acceptances."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_cibec_v1_authority_review import DEFAULT_PACK, validate
from research.cibec_v1.contract import CHARTER_FIELDS


BASE_FIXTURES = ROOT / "research/cibec_v1/source_free_fixture_manifest.json"
DEFAULT_OUTPUT = ROOT / "output/research/cibec-v1-signed-materialization"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(pack: Path, output: Path) -> tuple[int, dict[str, Any]]:
    review_code, review = validate(pack)
    report: dict[str, Any] = {
        "schema": "CIBEC_V1_SIGNED_AUTHORITY_MATERIALIZATION",
        "review_payload_sha256": review["review_payload_sha256"],
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
        "outputs_written": [],
    }
    if review_code:
        report["status"] = "REFUSED_UNSIGNED_OR_INVALID_REVIEW"
        report["blockers"] = review["blockers"]
        return 2, report
    charter_candidate = json.loads((pack / "CIBEC_V1_CHARTER_CANDIDATE.json").read_text())
    answer_candidate = json.loads((pack / "S01_S14_EXPECTATIONS_CANDIDATE.json").read_text())
    attestation = json.loads((pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").read_text())
    payload_sha = review["review_payload_sha256"]
    charter = {
        "schema": "CIBEC_V1_CHARTER",
        "status": "SIGNED_AND_FROZEN",
        "decisions": {field: charter_candidate["decisions"][field] for field in CHARTER_FIELDS},
        "signatures": [
            {
                "role": "MANUFACTURING_AUTHORITY",
                "name": attestation["manufacturing_authority"]["name"].strip(),
                "signed_payload_sha256": payload_sha,
            },
            {
                "role": "COUNTERSIGNING_AUTHORITY",
                "name": attestation["countersigning_authority"]["name"].strip(),
                "signed_payload_sha256": payload_sha,
            },
        ],
    }
    fixtures = json.loads(BASE_FIXTURES.read_text())
    fixtures["status"] = "SIGNED_AND_FROZEN"
    fixtures["expected_outputs"] = answer_candidate["fixtures"]
    fixtures["authority_signature_sha256"] = payload_sha
    charter_path = output / "CIBEC_V1_CHARTER.yaml"
    fixture_path = output / "source_free_fixture_manifest.json"
    desired = {charter_path: charter, fixture_path: fixtures}
    for path, value in desired.items():
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report.update(status="REFUSED_OUTPUT_COLLISION", blockers=[f"invalid existing output: {path.name}"])
                return 2, report
            if current != value:
                report.update(status="REFUSED_OUTPUT_COLLISION", blockers=[f"existing output differs: {path.name}"])
                return 2, report
    _write(charter_path, charter)
    _write(fixture_path, fixtures)
    report["status"] = "SIGNED_ARTIFACTS_MATERIALIZED"
    report["blockers"] = []
    report["outputs_written"] = [str(charter_path), str(fixture_path)]
    _write(output / "materialization_report.json", report)
    return 0, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    code, report = materialize(args.pack.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
