#!/usr/bin/env python3
"""Render post-inference 3-D overlays for the fixed simple-family diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
from scripts.run_gold11_v11_multi_scan_luna_max_v1 import _read_cases, _selected_references


def _line(item: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(item["line_start"], dtype=float), np.asarray(item["line_end"], dtype=float)


def render(result_path: Path, output_dir: Path) -> list[Path]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cases = _read_cases()
    references = _selected_references(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in result["cases"]:
        token = row["case_token"]
        scan = prepare_mesh_scan_v1(scan_bytes=Path(row["input_bindings"]["scan_path"]).read_bytes(),
                                    max_engine_rows=50_000)
        points = np.asarray(scan["raw_points"], dtype=float)
        # Deterministic display thinning; inference already completed and is
        # unaffected by this derived visualization.
        if len(points) > 8000:
            points = points[np.linspace(0, len(points) - 1, 8000, dtype=int)]
        gold = [_line(item) for item in references[token]["bends"]]
        predicted = [_line(item) for item in row["recovery"].get("predictions", [])]
        figure = plt.figure(figsize=(16, 5.6), constrained_layout=True)
        score = row["score"]
        title = (f"{token} / {row['part_key']}  "
                 f"TP {score['true_positive']} FP {score['false_positive']} FN {score['false_negative']}  "
                 f"pose={row['pose_selection'].get('pose_source', 'UNAVAILABLE')}")
        figure.suptitle(title, fontsize=11)
        for index, (elev, azim) in enumerate(((18, -60), (18, 30), (72, -60))):
            ax = figure.add_subplot(1, 3, index + 1, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.3, alpha=0.11,
                       color="#777777", depthshade=False)
            for start, end in gold:
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                        color="#e45756", linewidth=1.4, linestyle="--", alpha=0.8)
            for start, end in predicted:
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                        color="#00a6a6", linewidth=1.6, alpha=0.95)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.set_title(("Gold dashed / scan-supported predicted solid" if index == 0
                          else f"view {index + 1}"), fontsize=9)
            ax.set_box_aspect((1, 1, 1))
        path = output_dir / f"OVERLAY_{token}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path,
                        default=ROOT / "docs/research/gold11_v11_simple_family_diagnostic_v2/RESULT.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/research/gold11_v11_simple_family_diagnostic_v2/overlays")
    args = parser.parse_args()
    print(json.dumps({"status": "RENDERED_POST_INFERENCE_OVERLAYS",
                      "paths": [str(path.resolve()) for path in render(args.result.resolve(), args.output.resolve())]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
