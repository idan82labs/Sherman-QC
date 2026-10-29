#!/usr/bin/env python3
"""Build a fail-closed, report-only F1 qualification stage ledger."""

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

from scripts.finalize_f1_contract_qualification_v1 import QualificationError, validate_report
from scripts.validate_cibec_v1_authority_review import validate as validate_authority


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_status(path: Path, *, schema: str | None = None, kind: str | None = None) -> tuple[bool, str]:
    if not path.is_file():
        return False, "REPORT_MISSING"
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return False, "REPORT_INVALID_JSON"
    if not isinstance(value, dict):
        return False, "REPORT_NOT_OBJECT"
    if schema is not None and value.get("schema") != schema:
        return False, "REPORT_SCHEMA_MISMATCH"
    if kind is not None and value.get("kind") != kind:
        return False, "REPORT_KIND_MISMATCH"
    return value.get("status") == "PASS", str(value.get("status") or "STATUS_MISSING")


def build(
    root: Path,
    *,
    soak_report: Path,
    authority_pack: Path,
    source_free_report: Path,
    physical_report: Path,
    cad_grouping_report: Path,
    custody_public_report: Path,
    unified_gate_report: Path,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    try:
        soak = _read(soak_report)
        validate_report(soak, root)
        soak_ok, soak_detail = True, "PASS"
    except (OSError, json.JSONDecodeError, QualificationError, KeyError, TypeError, ValueError) as exc:
        soak_ok, soak_detail = False, str(exc)
    stages.append({"stage": "SOURCE_FREE_CONTRACT_SOAK", "status": "PASS" if soak_ok else "BLOCKED", "detail": soak_detail})

    authority_code, authority = validate_authority(authority_pack)
    authority_ok = authority_code == 0
    stages.append({
        "stage": "TWO_AUTHORITY_CIBEC_ACCEPTANCE",
        "status": "PASS" if authority_ok else "BLOCKED",
        "detail": authority["status"],
        "blockers": authority["blockers"],
        "review_payload_sha256": authority["review_payload_sha256"],
    })

    upstream = soak_ok and authority_ok
    checks = (
        ("SIGNED_SOURCE_FREE_QUALIFICATION", source_free_report, "CIBEC_V1_SOURCE_FREE_QUALIFICATION", None),
        ("CONSUMED_PHYSICAL_CIBEC_QUALIFICATION", physical_report, None, "CIBEC_V1_PHYSICAL_ANNOTATION_QUALIFICATION"),
        ("CAD_GROUPING_QUALIFICATION", cad_grouping_report, None, "CIBEC_V1_CAD_GROUPING_QUALIFICATION"),
    )
    for stage, path, schema, kind in checks:
        if not upstream:
            stages.append({"stage": stage, "status": "NOT_REACHED", "detail": "UPSTREAM_GATE_BLOCKED"})
            continue
        passed, detail = _report_status(path, schema=schema, kind=kind)
        stages.append({"stage": stage, "status": "PASS" if passed else "BLOCKED", "detail": detail})
        upstream = passed

    if not upstream:
        stages.append({"stage": "SEQUESTERED_30_BY_15_CORPUS", "status": "NOT_REACHED", "detail": "UPSTREAM_GATE_BLOCKED"})
    else:
        try:
            custody = _read(custody_public_report)
            custody_ok = (
                custody.get("schema") == "F1_BLIND_ANNOTATION_ASSIGNMENT_V1"
                and custody.get("status") == "READY_FOR_BLIND_ANNOTATION"
                and int(custody.get("case_count", 0)) >= 30
                and int(custody.get("morphology_family_count", 0)) >= 15
            )
            detail = str(custody.get("status") or "STATUS_MISSING")
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            custody_ok, detail = False, "REPORT_MISSING_OR_INVALID"
        stages.append({"stage": "SEQUESTERED_30_BY_15_CORPUS", "status": "PASS" if custody_ok else "BLOCKED", "detail": detail})
        upstream = custody_ok

    if not upstream:
        stages.append({"stage": "UNTOUCHED_PHYSICAL_GENERALIZATION_GATE", "status": "NOT_REACHED", "detail": "UPSTREAM_GATE_BLOCKED"})
    else:
        try:
            gate = _read(unified_gate_report)
            gates = gate.get("gates") if isinstance(gate, dict) else None
            grades = {name: row.get("grade") for name, row in gates.items()} if isinstance(gates, dict) else {}
            gate_ok = bool(grades) and all(value == "A_VERIFIED_PHYSICAL_PROGRESS" for value in grades.values())
            detail = grades or "REPORT_MISSING_OR_INVALID"
        except (OSError, json.JSONDecodeError, AttributeError):
            gate_ok, detail = False, "REPORT_MISSING_OR_INVALID"
        stages.append({"stage": "UNTOUCHED_PHYSICAL_GENERALIZATION_GATE", "status": "PASS" if gate_ok else "BLOCKED", "detail": detail})

    first_blocked = next((row["stage"] for row in stages if row["status"] == "BLOCKED"), None)
    return {
        "schema_version": 1,
        "kind": "SHERMAN_QC_F1_QUALIFICATION_READINESS_V1",
        "status": "PASS" if first_blocked is None and all(row["status"] == "PASS" for row in stages) else "BLOCKED",
        "first_blocked_stage": first_blocked,
        "physical_generalization_progress": any(row["stage"] == "UNTOUCHED_PHYSICAL_GENERALIZATION_GATE" and row["status"] == "PASS" for row in stages),
        "production_promotion_authorized": False,
        "stages": stages,
    }


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--soak-report", type=Path, required=True)
    parser.add_argument("--authority-pack", type=Path, default=ROOT / "output/research/cibec-v1-authority-candidate-v2")
    parser.add_argument("--source-free-report", type=Path, default=ROOT / "output/research/cibec-v1-source-free-qualification/qualification_report.json")
    parser.add_argument("--physical-report", type=Path, default=ROOT / "output/research/cibec-v1-physical-qualification/result.json")
    parser.add_argument("--cad-grouping-report", type=Path, default=ROOT / "output/research/cibec-v1-cad-grouping-qualification/result.json")
    parser.add_argument("--custody-public-report", type=Path, default=ROOT / "output/research/f1-blind-custody/public_assignment.json")
    parser.add_argument("--unified-gate-report", type=Path, default=ROOT / "output/research/f1-untouched-unified-gate/result.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.root.resolve(), soak_report=args.soak_report.resolve(), authority_pack=args.authority_pack.resolve(),
        source_free_report=args.source_free_report.resolve(), physical_report=args.physical_report.resolve(),
        cad_grouping_report=args.cad_grouping_report.resolve(), custody_public_report=args.custody_public_report.resolve(),
        unified_gate_report=args.unified_gate_report.resolve(),
    )
    _atomic(args.output, report)
    print(json.dumps({"status": report["status"], "first_blocked_stage": report["first_blocked_stage"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
