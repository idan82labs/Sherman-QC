#!/usr/bin/env python3
"""Render exact EXP_CAD_0009 semantic payloads."""

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
from scripts.render_f1_source_only_prediction_audit import render


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    rows = []
    for record in summary["records"]:
        case_id = str(record["case_id"])
        prediction = Path(record["semantic_result"])
        geometry = Path(record["scan_path"])
        output = args.output_root / f"{case_id}.png"
        render(geometry=geometry, prediction_path=prediction, output=output, max_points=250_000)
        rows.append({
            "case_id": case_id,
            "geometry": str(geometry.resolve()),
            "geometry_sha256": _sha(geometry),
            "semantic_result": str(prediction.resolve()),
            "semantic_result_sha256": _sha(prediction),
            "render": str(output.resolve()),
            "render_sha256": _sha(output),
            "prediction_count": int(record["final_predictions"]),
        })
    _atomic(args.manifest, {"schema": "SHERMAN_QC_EXP_CAD_0009_CANDIDATE_BOUND_VISUAL_MANIFEST_V1", "status": "AWAITING_MANUAL_ADJUDICATION", "candidate_bound": True, "case_count": len(rows), "records": rows})
    print(json.dumps({"status": "COMPLETE", "cases": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
