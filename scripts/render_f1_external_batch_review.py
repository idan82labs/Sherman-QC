#!/usr/bin/env python3
"""Render and count-score an external F1 batch after blind inference completes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_line_audit import render  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--intake-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()

    prediction_root = args.prediction_root.resolve()
    intake_path = args.intake_manifest.resolve()
    intake_root = intake_path.parent
    output_dir = args.output_dir.resolve()
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    rows = []
    render_paths: list[Path] = []
    for part in intake.get("parts") or ():
        expected = int(part["expected_bend_count"])
        for scan in part.get("scans") or ():
            token = str(scan["case_token"])
            prediction_path = prediction_root / token / "decomposition_result.json"
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            prediction["scan_path"] = str((intake_root / str(scan["path"])).resolve())
            review_manifest = output_dir / "manifests" / f"{token}.json"
            _write_json(review_manifest, prediction)
            render_path = output_dir / "renders" / f"{token}_{part['part_id']}_{scan['role']}.png"
            render(review_manifest, render_path, args.max_points, False, None)
            predicted = len(prediction.get("owned_bend_regions") or ())
            rows.append(
                {
                    "case_token": token,
                    "part_id": part["part_id"],
                    "role": scan["role"],
                    "expected_bend_count": expected,
                    "predicted_bend_count": predicted,
                    "count_exact": predicted == expected,
                    "prediction_path": str(prediction_path),
                    "render_path": str(render_path),
                }
            )
            render_paths.append(render_path)

    thumbnails = []
    for path in render_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1280, 430), Image.Resampling.LANCZOS)
        thumbnails.append(image.copy())
    width = 2560
    cell_height = 470
    height = cell_height * ((len(thumbnails) + 1) // 2)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, row) in enumerate(zip(thumbnails, rows)):
        x = (index % 2) * 1280
        y = (index // 2) * cell_height
        sheet.paste(image, (x, y + 34))
        draw.text(
            (x + 12, y + 8),
            f"{row['case_token']}  {row['part_id']} {row['role']}  F1={row['predicted_bend_count']} expected={row['expected_bend_count']}",
            fill="#111827",
        )
    contact_sheet = output_dir / "all_12_v18_contact_sheet.jpg"
    sheet.save(contact_sheet, quality=92)
    summary = {
        "schema_version": 1,
        "inference_truth_used": False,
        "post_inference_count_scoring": True,
        "scan_count": len(rows),
        "count_exact": sum(int(row["count_exact"]) for row in rows),
        "count_accuracy": round(sum(int(row["count_exact"]) for row in rows) / len(rows), 6),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    _write_json(output_dir / "review_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("scan_count", "count_exact", "count_accuracy", "contact_sheet")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
