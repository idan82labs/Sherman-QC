#!/usr/bin/env python3
"""Compose final scored V59 overlay review sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.visual_root / "manifest.json").read_text())["records"]
    outputs = []
    for start in range(0, len(rows), 4):
        canvas = Image.new("RGB", (1800, 1100), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(rows[start : start + 4]):
            image = Image.open(row["image"]).convert("RGB")
            image.thumbnail((880, 500), Image.Resampling.LANCZOS)
            x, y = (offset % 2) * 900 + 10, (offset // 2) * 550 + 35
            canvas.paste(image, (x, y))
            draw.text((x, 8 + (offset // 2) * 550), f"{row['case_id']} | TP={row['true_positive']} FP={row['false_positive']} FN={row['false_negative']}", fill="black")
        output = args.output_root / f"review_sheet_{start // 4 + 1:02d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        outputs.append(str(output.resolve()))
    (args.output_root / "manifest.json").write_text(json.dumps({"schema_version": 1, "sheets": outputs}, indent=2) + "\n")
    print(json.dumps({"sheet_count": len(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
