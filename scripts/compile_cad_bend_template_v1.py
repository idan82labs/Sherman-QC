#!/usr/bin/env python3
"""Compile a STEP model into a compact, reusable CAD-first bend template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_step_topology_bend_axes import extract_step_bend_axes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_template(step: Path) -> dict:
    extracted = extract_step_bend_axes(step)
    events = [{
        "event_id": f"CAD_EVENT_{index:04d}",
        "source_bend_id": row["bend_id"],
        "step_face_indices": row["face_indices"],
        "cad_line_start_mm": row["line_start"],
        "cad_line_end_mm": row["line_end"],
        "radii_mm": row["radii_mm"],
        "angular_sweeps_deg": row["angular_sweeps_deg"],
        "axial_span_mm": row["axial_span_mm"],
        "relation": "MANY_FACES_TO_ONE_EVENT",
    } for index, row in enumerate(extracted["bend_axes"], 1)]
    return {
        "schema": "SHERMAN_QC_CAD_BEND_TEMPLATE_V1",
        "step_sha256": _sha(step),
        "extractor_sha256": _sha(Path(__file__).with_name("extract_step_topology_bend_axes.py")),
        "coordinate_frame": "CAD",
        "units": "mm",
        "event_count": len(events),
        "events": events,
        "rejected_cylinder_group_count": len(extracted["rejected_groups"]),
        "contract": extracted["contract"],
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compile_template(args.step.resolve())
    _atomic(args.output, payload)
    print(json.dumps({"status": "COMPLETE", "event_count": payload["event_count"], "step_sha256": payload["step_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
