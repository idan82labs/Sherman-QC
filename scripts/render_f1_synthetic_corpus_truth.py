#!/usr/bin/env python3
"""Render all synthetic scan/truth overlays and compact visual contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_manual_bend_review import render


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.corpus_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    truth_by_case = {str(row["case_id"]): row for row in truth["cases"]}
    label_root = root / "truth_cases"
    render_root = root / "truth_overlays"
    contact_root = root / "contact_sheets"
    for directory in (label_root, render_root, contact_root):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        label_path = label_root / f"{case_id}.json"
        label_path.write_text(json.dumps(truth_by_case[case_id], sort_keys=True, indent=2) + "\n", encoding="utf-8")
        output = render_root / f"{case_id}.png"
        render(Path(case["scan_path"]), label_path, output, 200_000)
        rows.append({"case_id": case_id, "overlay_path": str(output), "overlay_sha256": _sha(output), "scan_sha256": case["scan_sha256"], "truth_count": len(truth_by_case[case_id]["manual_bends"])})

    sheets = []
    for sheet_index in range(5):
        subset = rows[sheet_index * 6 : (sheet_index + 1) * 6]
        canvas = Image.new("RGB", (2400, 1800), "#07101f")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(subset):
            image = Image.open(row["overlay_path"]).convert("RGB")
            image.thumbnail((780, 820), Image.Resampling.LANCZOS)
            column = index % 3
            line = index // 3
            x = column * 800 + (800 - image.width) // 2
            y = line * 900 + 55 + (800 - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((column * 800 + 24, line * 900 + 20), f"{row['case_id']} — {row['truth_count']} bends", fill="#ffffff")
        sheet_path = contact_root / f"cases_{sheet_index * 6 + 1:03d}_{(sheet_index + 1) * 6:03d}.png"
        canvas.save(sheet_path, quality=95)
        sheets.append({"path": str(sheet_path), "sha256": _sha(sheet_path), "case_ids": [row["case_id"] for row in subset]})
    report = {"schema_version": 1, "kind": "synthetic_lidar_truth_visual_audit_artifacts", "confirmatory_eligible": False, "cases": rows, "contact_sheets": sheets}
    (root / "truth_visual_artifacts.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "contact_sheet_count": len(sheets), "root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
