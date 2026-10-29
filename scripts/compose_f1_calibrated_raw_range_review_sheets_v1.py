#!/usr/bin/env python3
"""Compose high-resolution review sheets from frozen truth-free visuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-sheet", type=int, default=3)
    args = parser.parse_args()
    visual_root = args.visual_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    case_dirs = sorted(path for path in visual_root.glob("O???") if path.is_dir())
    rows = []
    for sheet_index, start in enumerate(range(0, len(case_dirs), args.cases_per_sheet), 1):
        group = case_dirs[start : start + args.cases_per_sheet]
        rendered = []
        for case in group:
            depth = Image.open(case / "depth_contact_native_tiles.png").convert("RGB")
            alignment = Image.open(case / "backprojection_alignment_3d.png").convert("RGB")
            depth.thumbnail((1152, 468), Image.Resampling.LANCZOS)
            alignment.thumbnail((1350, 468), Image.Resampling.LANCZOS)
            rendered.append((case.name, depth.copy(), alignment.copy()))
        row_height = 510
        canvas = Image.new("RGB", (2540, row_height * len(rendered)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (case_id, depth, alignment) in enumerate(rendered):
            y = row_index * row_height
            draw.text((12, y + 8), f"{case_id} | all 8 native depth views", fill="black")
            draw.text((1180, y + 8), f"{case_id} | calibrated views on fused source", fill="black")
            canvas.paste(depth, (12, y + 32))
            canvas.paste(alignment, (1180, y + 32))
        path = output / f"review_sheet_{sheet_index:02d}.png"
        canvas.save(path)
        rows.append({"sheet": sheet_index, "case_ids": [item[0] for item in rendered], "path": str(path)})
    report = {
        "schema_version": 1,
        "kind": "calibrated_raw_range_visual_review_sheets_v1",
        "case_count": len(case_dirs),
        "sheet_count": len(rows),
        "truth_opened": False,
        "sheets": rows,
    }
    (output / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
