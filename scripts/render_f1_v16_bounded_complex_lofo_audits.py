#!/usr/bin/env python3
"""Render every V16 bounded-complex LOFO mismatch at native visual scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_gold_overlay_review import render_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=300_000)
    args = parser.parse_args()

    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    gold_rows = {
        str(row["case_token"]): row
        for row in json.loads(args.gold_manifest.resolve().read_text(encoding="utf-8"))["cases"]
    }
    inference_rows = {
        str(row["case_token"]): row
        for row in json.loads(args.inference_manifest.resolve().read_text(encoding="utf-8"))["cases"]
    }
    output_dir = args.output_dir.resolve()
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    reviews = []
    for family in report["families"]:
        for case in family["cases"]:
            if case["exact"]:
                continue
            token = str(case["case_token"])
            gold = gold_rows[token]
            label_path = Path(str(gold["label_path"])).resolve()
            scan_path = Path(str(inference_rows[token]["source_scan_path"])).resolve()
            prediction_path = prediction_dir / f"{token}.json"
            prediction_path.write_text(
                json.dumps({"owned_bend_regions": case["predictions"]}, indent=2) + "\n",
                encoding="utf-8",
            )
            reviews.append(
                render_case(
                    case_id=token,
                    scan_path=scan_path,
                    gold_path=label_path,
                    prediction_path=prediction_path,
                    output_path=output_dir / f"{token}.png",
                    max_points=args.max_points,
                )
            )
    (output_dir / "review.json").write_text(json.dumps(reviews, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rendered_mismatch_count": len(reviews), "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
