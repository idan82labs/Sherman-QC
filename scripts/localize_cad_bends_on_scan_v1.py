#!/usr/bin/env python3
"""Localize a frozen CAD bend template on a later PLY scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_f1_step_topology_targeted_verifier import _duplicate, _to_raw, _two_surface_normal_turn
from scripts.build_additional_line_label_review import _scan_points, line_scan_support
from scripts.render_f1_source_only_prediction_audit import render

TRUSTED_SOURCE = "PLANE_DATUM_REFINEMENT"
MIN_CONFIDENCE = 0.55


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def localize(template: dict, scan: Path, registration: dict) -> dict:
    if template.get("schema") != "SHERMAN_QC_CAD_BEND_TEMPLATE_V1":
        raise ValueError("invalid CAD bend template")
    transform = registration.get("raw_scan_to_cad_transform")
    trusted = bool(
        not registration.get("alignment_abstained")
        and registration.get("alignment_source") == TRUSTED_SOURCE
        and float(registration.get("alignment_confidence") or 0.0) >= MIN_CONFIDENCE
        and transform is not None
    )
    base = {
        "schema": "SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V1",
        "lane": "CAD_ASSISTED",
        "route": "CAD_FIRST_SCAN_LOCALIZED_V1",
        "cad_template_sha256": hashlib.sha256((json.dumps(template, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
        "step_sha256": template["step_sha256"],
        "scan_sha256": _sha(scan),
        "registration_sha256": hashlib.sha256((json.dumps(registration, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
        "coordinate_frame": "RAW_SCAN",
        "units": "mm",
        "cad_guided_zero_shot": True,
        "global_scan_proposals_allowed": False,
        "gold_or_expected_counts_used": False,
    }
    if not trusted:
        return {**base, "status": "ABSTAIN", "reason": "UNTRUSTED_CAD_SCAN_REGISTRATION", "predictions": [], "records": []}
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix[:3, :3])) < 1e-9:
        raise ValueError("invalid raw_scan_to_cad_transform")
    points = _scan_points(scan)
    tree = cKDTree(points)
    accepted: list[dict] = []
    records = []
    for event in template["events"]:
        raw = _to_raw([event["cad_line_start_mm"], event["cad_line_end_mm"]], transform)
        support = line_scan_support(points, raw[0].tolist(), raw[1].tolist())
        record = {"event_id": event["event_id"], "scan_support": support, "accepted": False}
        if support["frame_integrity_pass"]:
            turn = _two_surface_normal_turn(points, tree, support["trimmed_line_start"], support["trimmed_line_end"])
            record["normal_turn_verification"] = turn
            if turn["passed"]:
                row = {
                    "bend_id": event["event_id"],
                    "source_hypothesis_id": event["source_bend_id"],
                    "line_start": support["trimmed_line_start"],
                    "line_end": support["trimmed_line_end"],
                    "detected_line_start": support["trimmed_line_start"],
                    "detected_line_end": support["trimmed_line_end"],
                    "cad_line_start": event["cad_line_start_mm"],
                    "cad_line_end": event["cad_line_end_mm"],
                    "step_face_indices": event["step_face_indices"],
                    "normal_turn_deg": turn["normal_turn_deg"],
                    "assignment_source": "CAD_TEMPLATE_SCAN_LOCALIZED",
                }
                duplicate = next((prior for prior in accepted if _duplicate(row, prior)), None)
                if duplicate is None:
                    accepted.append(row); record["accepted"] = True
                else:
                    record["rejection_reason"] = "DUPLICATE_OF_" + duplicate["bend_id"]
            else:
                record["rejection_reason"] = "NO_TWO_SURFACE_NORMAL_TURN"
        else:
            record["rejection_reason"] = "INSUFFICIENT_SCAN_SUPPORT"
        records.append(record)
    accepted.sort(key=lambda row: row["bend_id"])
    return {**base, "status": "COMPLETE", "cad_event_count": template["event_count"], "localized_bend_count": len(accepted), "predictions": accepted, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-output", type=Path)
    args = parser.parse_args()
    payload = localize(json.loads(args.template.read_text()), args.scan.resolve(), json.loads(args.registration.read_text()))
    _atomic(args.output, payload)
    if args.visual_output and payload["status"] == "COMPLETE":
        render(geometry=args.scan.resolve(), prediction_path=args.output.resolve(), output=args.visual_output.resolve(), max_points=250_000)
    print(json.dumps({"status": payload["status"], "localized_bend_count": len(payload["predictions"])}, sort_keys=True))
    return 0 if payload["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
