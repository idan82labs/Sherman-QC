#!/usr/bin/env python3
"""Localize neutral-surface CAD bend lines on a registered physical scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compile_cad_bend_template_v1 import _atomic
from scripts.localize_cad_bends_on_scan_v1 import localize as localize_v1
from scripts.render_f1_source_only_prediction_audit import render


def localize(template: dict, scan: Path, registration: dict) -> dict:
    if template.get("schema") != "SHERMAN_QC_CAD_BEND_TEMPLATE_V2":
        raise ValueError("invalid CAD bend template V2")
    v1_compatible = {**template, "schema": "SHERMAN_QC_CAD_BEND_TEMPLATE_V1"}
    result = localize_v1(v1_compatible, scan, registration)
    result.update({
        "schema": "SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V2",
        "route": "CAD_NEUTRAL_SURFACE_SCAN_LOCALIZED_V2",
        "cad_template_sha256": hashlib.sha256(
            (json.dumps(template, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "canonical_line_rule": "NEUTRAL_RADIUS_HALF_ANGULAR_TURN",
    })
    for prediction in result["predictions"]:
        prediction["assignment_source"] = "CAD_NEUTRAL_SURFACE_SCAN_LOCALIZED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-output", type=Path)
    args = parser.parse_args()
    payload = localize(
        json.loads(args.template.read_text()),
        args.scan.resolve(),
        json.loads(args.registration.read_text()),
    )
    _atomic(args.output, payload)
    if args.visual_output and payload["status"] == "COMPLETE":
        render(
            geometry=args.scan.resolve(),
            prediction_path=args.output.resolve(),
            output=args.visual_output.resolve(),
            max_points=250_000,
        )
    print(json.dumps({
        "status": payload["status"],
        "localized_bend_count": len(payload["predictions"]),
    }, sort_keys=True))
    return 0 if payload["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
