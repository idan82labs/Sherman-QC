#!/usr/bin/env python3
"""Score and render a zero-shot overlay manifest against manual bend axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.render_f1_gold_overlay_review import render_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="zero_shot_manual_review")
    args = parser.parse_args()

    gold_payload = json.loads(args.manual_labels.read_text(encoding="utf-8"))
    manifest = json.loads(args.overlay_manifest.read_text(encoding="utf-8"))
    predictions = []
    for bend in manifest.get("bends") or []:
        line = bend.get("detected_line") or bend.get("cad_line") or {}
        start = line.get("display_start")
        end = line.get("display_end")
        if start is None or end is None:
            continue
        predictions.append(
            {
                "bend_id": str(bend.get("bend_id") or f"P{len(predictions) + 1}"),
                "span_endpoints": [start, end],
                "source_kind": bend.get("source_kind"),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction_owned_regions.json"
    prediction_path.write_text(
        json.dumps({"owned_bend_regions": predictions}, indent=2) + "\n",
        encoding="utf-8",
    )
    score = match_lines(gold_payload.get("manual_bends") or [], predictions, Thresholds())
    score.update(
        {
            "case_id": args.case_id,
            "manual_labels_path": str(args.manual_labels.resolve()),
            "overlay_manifest_path": str(args.overlay_manifest.resolve()),
            "inference_truth_exposed": False,
        }
    )
    render = render_case(
        case_id=args.case_id,
        scan_path=args.scan_path.resolve(),
        gold_path=args.manual_labels.resolve(),
        prediction_path=prediction_path,
        output_path=output_dir / "gold_vs_zero_shot_six_view.png",
        max_points=260_000,
    )
    score["render"] = render
    (output_dir / "score.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
