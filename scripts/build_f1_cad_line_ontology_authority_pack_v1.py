#!/usr/bin/env python3
"""Freeze the physical three-convention CAD bend-line authority review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_stable(payload)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    event_refs = []
    for sidecar_path in sorted(args.visual_root.glob("C*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        image_path = sidecar_path.with_suffix(".png")
        if not image_path.is_file():
            raise ValueError(f"missing image for {sidecar_path.name}")
        case_id = str(sidecar["case_id"])
        events = []
        for visual_index, row in sorted(sidecar["event_index"].items(), key=lambda item: int(item[0])):
            ref = f"{case_id}:{row['event_id']}"
            event_refs.append(ref)
            events.append({"event_ref":ref,"visual_index":int(visual_index),**row})
        cases.append({
            "case_id": case_id,
            "geometry_name": sidecar["geometry_name"],
            "image_relative": str(image_path.relative_to(args.output_root)),
            "image_sha256": _sha(image_path),
            "sidecar_relative": str(sidecar_path.relative_to(args.output_root)),
            "sidecar_sha256": _sha(sidecar_path),
            "events": events,
        })
    if not cases:
        raise ValueError("authority pack contains no cases")
    manifest = {
        "schema":"F1_CAD_LINE_ONTOLOGY_AUTHORITY_PACK_V1",
        "status":"AWAITING_TWO_INDEPENDENT_AUTHORITIES",
        "evidence_class":"DEVELOPMENT_CONSUMED_PHYSICAL_ADVISORY",
        "case_count":len(cases),
        "event_review_count":len(event_refs),
        "truth_opened":False,
        "expected_counts_opened":False,
        "prediction_score_shown":False,
        "conventions":{
            "CURVATURE_CENTER":"Axis through the paired CAD cylinders' center of curvature.",
            "NEUTRAL_HALF_TURN":"Line at the paired-skin neutral radius and half angular sweep.",
            "OBSERVED_SURFACE_INTERSECTION":"Intersection of two spatially fitted physical scan flanks.",
            "OTHER_AUTHORITY_DRAWN":"Authority supplies a replacement line under the signed charter.",
        },
        "event_status_choices":["QC_BEND","NOT_QC_BEND","UNRESOLVED"],
        "cases":cases,
    }
    manifest_sha = hashlib.sha256(_stable(manifest)).hexdigest()
    _atomic(args.output_root / "MANIFEST.json", manifest)
    template_cases = [{
        "event_ref":ref,
        "event_status":"PENDING",
        "line_convention":"PENDING",
        "rationale":"",
        "replacement_line":None,
    } for ref in event_refs]
    for role in ("MANUFACTURING_AUTHORITY", "COUNTERSIGNING_AUTHORITY"):
        _atomic(args.output_root / f"{role}_DECISION_TEMPLATE.json", {
            "schema":"F1_CAD_LINE_ONTOLOGY_AUTHORITY_DECISION_V1",
            "review_pack_sha256":manifest_sha,
            "authority_role":role,
            "authority_name":"REQUIRED_REAL_NAME",
            "decision":"PENDING",
            "events":template_cases,
        })
    print(json.dumps({"status":manifest["status"],"cases":len(cases),"events":len(event_refs),"review_pack_sha256":manifest_sha},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
