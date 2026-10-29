#!/usr/bin/env python3
"""Build the truth-denying source manifest for EXP_CAD_0011."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
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


def build(parent: Path, case_root: Path) -> dict:
    raw = json.loads(parent.read_text())
    records = []
    for index, row in enumerate(raw.get("results") or [], 1):
        scan = Path(str(row["scan_path"])).resolve()
        cad = Path(str(row["cad_path"])).resolve()
        template = (case_root / f"C{index:03d}" / "cad_bend_template.json").resolve()
        metadata = dict(row.get("alignment_metadata") or {})
        token = "RG" + hashlib.sha256(scan.read_bytes()).hexdigest()[:10].upper()
        records.append({
            "case_token": token,
            "scan_path": str(scan),
            "scan_sha256": sha(scan),
            "cad_path": str(cad),
            "cad_sha256": sha(cad),
            "cad_template_path": str(template),
            "cad_template_sha256": sha(template),
            "parent_alignment_source": metadata.get("alignment_source"),
            "parent_alignment_abstained": bool(metadata.get("alignment_abstained")),
        })
    if len(records) != 13 or len({row["scan_sha256"] for row in records}) != 13:
        raise ValueError("expected thirteen unique source scans")
    return {
        "schema": "EXP_CAD_0011_TRUTH_DENYING_SOURCE_MANIFEST_V1",
        "experiment_id": "EXP_CAD_0011",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
        "generalization_evidence": False,
        "gold_or_expected_counts_present": False,
        "filenames_are_audit_only": True,
        "case_count": len(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.parent.resolve(), args.case_root.resolve())
    atomic(args.output.resolve(), payload)
    print(json.dumps({"case_count": payload["case_count"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

