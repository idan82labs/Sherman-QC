#!/usr/bin/env python3
"""Render native-resolution prediction audits for the CAD-guided Gold13 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_physical15_development_lane import _contact_sheet
from scripts.render_f1_source_only_prediction_audit import render


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    records = []
    images = []
    for row in comparison["visual_records"]:
        case_id = str(row["case_id"])
        output = args.output_root / f"{case_id}.png"
        metadata = render(
            geometry=Path(row["scan_path"]),
            prediction_path=Path(row["prediction_path"]),
            output=output,
            max_points=250_000,
        )
        images.append(output)
        records.append({**metadata, "case_id": case_id, "image_sha256": _sha(output)})
    sheets = []
    for first in range(0, len(images), 3):
        path = args.output_root / f"contact_sheet_{first // 3 + 1:02d}.png"
        _contact_sheet(images[first:first + 3], path)
        sheets.append({"path": str(path.resolve()), "sha256": _sha(path)})
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_guided_gold13_native_visual_manifest",
        "case_count": len(records),
        "records": records,
        "contact_sheets": sheets,
        "visual_ruling": "PENDING_MANUAL_NATIVE_RESOLUTION_REVIEW",
    }
    path = args.output_root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "cases": len(records), "sheets": len(sheets)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
