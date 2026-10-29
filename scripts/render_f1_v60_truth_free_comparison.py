#!/usr/bin/env python3
"""Render truth-free V52/V60 three-view source overlays for visual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROUTE = "calibrated_per_view_segment_consensus_v60"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(row: dict) -> np.ndarray:
    return np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--v60-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("visual output root is not empty")
    records = []
    for index in range(1, 31):
        case_id = f"O{index:03d}"
        bundle = args.corpus_root / "public" / case_id
        v52 = json.loads((args.v52_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        v60 = json.loads((args.v60_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        raw = [row for row in v60["predictions"] if row.get("selection_route") == ROUTE]
        if not raw:
            continue
        points = np.asarray(o3d.io.read_point_cloud(str(bundle / "source.ply")).points, dtype=np.float64)
        if len(points) > 100_000:
            points = points[np.linspace(0, len(points) - 1, 100_000, dtype=int)]
        fig = plt.figure(figsize=(16, 9), dpi=256)
        views = ((24, -58, "oblique"), (90, -90, "top"), (8, -90, "side"))
        for panel, (elevation, azimuth, title) in enumerate(views, 1):
            axis = fig.add_subplot(1, 3, panel, projection="3d")
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.08, c="#aab2bd", alpha=0.35, rasterized=True)
            for row in v52["predictions"]:
                line = _line(row)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#2f855a", linewidth=4.0)
            for row in raw:
                line = _line(row)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#00bcd4", linewidth=4.0)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(title)
            axis.set_axis_off()
            axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
        fig.suptitle(f"{case_id} | green retained V52 | cyan raw V60", fontsize=18)
        output = args.output_root / f"{case_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        records.append({"case_id": case_id, "raw_new_count": len(raw), "image": str(output.resolve()), "image_sha256": _sha(output)})
    manifest = {"schema_version": 1, "kind": "sherman_qc_f1_v60_truth_free_visual_comparison", "truth_opened": False, "changed_case_count": len(records), "records": records}
    path = args.output_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_case_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
