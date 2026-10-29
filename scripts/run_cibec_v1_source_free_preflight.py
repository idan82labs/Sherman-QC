#!/usr/bin/env python3
"""Run the sealed, source-free entry gate for CIBEC_V1.

The command deliberately stops before fixture semantics or any physical asset
is opened unless the charter and fixture expectations carry the required
authority signatures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cibec_v1.contract import CibecValidationError, validate_charter


DEFAULT_CHARTER = ROOT / "research/cibec_v1/CIBEC_V1_CHARTER.yaml"
DEFAULT_FIXTURES = ROOT / "research/cibec_v1/source_free_fixture_manifest.json"
DEFAULT_OUTPUT = ROOT / "output/research/cibec-v1-source-free-preflight/preflight_report.json"
PINNED_FILES = (
    ROOT / "scripts/render_f1_cad_guided_gold13_audits.py",
    ROOT / "scripts/render_f1_source_only_prediction_audit.py",
    ROOT / "scripts/evaluate_f1_against_manual_gold.py",
    ROOT / "research/cibec_v1/contract.py",
    ROOT / "research/cibec_v1/agreement.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(charter_path: Path, fixture_path: Path, output_path: Path) -> tuple[int, dict[str, Any]]:
    charter = json.loads(charter_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema": "CIBEC_V1_SOURCE_FREE_PREFLIGHT",
        "authorized_scope": "CIBEC_V1_ONLY_NO_PHYSICAL_INFERENCE",
        "charter_sha256": _sha(charter_path),
        "fixture_manifest_sha256": _sha(fixture_path),
        "pinned_files": {str(path.relative_to(ROOT)): _sha(path) for path in PINNED_FILES},
        "fixture_count": len(fixtures.get("fixtures") or []),
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
    }
    blockers: list[str] = []
    try:
        validate_charter(charter)
    except CibecValidationError as exc:
        blockers.append(f"CHARTER_SIGNATURE_REQUIRED: {exc}")
    if report["fixture_count"] != 14:
        blockers.append("FIXTURE_SET_MUST_CONTAIN_EXACTLY_14_CASES")
    if fixtures.get("expected_outputs") is None:
        blockers.append("SIGNED_FIXTURE_EXPECTATIONS_REQUIRED")
    signature = fixtures.get("authority_signature_sha256")
    if not isinstance(signature, str) or len(signature) != 64:
        blockers.append("FIXTURE_AUTHORITY_SIGNATURE_REQUIRED")
    report["blockers"] = blockers
    report["status"] = "BLOCKED_PRE_PHYSICAL" if blockers else "READY_FOR_SOURCE_FREE_EXECUTION"
    _write(output_path, report)
    return (2 if blockers else 0), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charter", type=Path, default=DEFAULT_CHARTER)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    code, report = run(args.charter.resolve(), args.fixtures.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
