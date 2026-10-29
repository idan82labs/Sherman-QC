#!/usr/bin/env python3
"""Render truth-free V61/V62 comparisons on the prospective complex V4 corpus."""

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


ROUTE = "calibrated_distributed_turn_consensus_v62"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(row: dict) -> np.ndarray:
    return np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v61-root", type=Path, required=True)
    parser.add_argument("--v62-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("visual output root is not empty")

    records = []
    for index in range(1, 31):
        case_id = f"O{index:03d}"
        source_path = args.corpus_root / "public" / case_id / "source.ply"
        v61_path = args.v61_root / "primary" / case_id / "semantic_result.json"
        v62_path = args.v62_root / "primary" / case_id / "semantic_result.json"
        v61 = json.loads(v61_path.read_text(encoding="utf-8"))
        v62 = json.loads(v62_path.read_text(encoding="utf-8"))
        births = [row for row in v62["predictions"] if row.get("selection_route") == ROUTE]
        if not births:
            continue

        points = np.asarray(o3d.io.read_point_cloud(str(source_path)).points, dtype=np.float64)
        if len(points) > 120_000:
            points = points[np.linspace(0, len(points) - 1, 120_000, dtype=int)]

        figure = plt.figure(figsize=(18, 10), dpi=240, facecolor="#f8fafc")
        views = (
            (24, -58, "oblique"),
            (90, -90, "top"),
            (8, -90, "side"),
            (20, 32, "reverse oblique"),
            (2, 0, "end"),
            (-16, -58, "underside"),
        )
        for panel, (elevation, azimuth, title) in enumerate(views, 1):
            axis = figure.add_subplot(2, 3, panel, projection="3d")
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2], s=0.09,
                c="#aab2bd", alpha=0.34, rasterized=True,
            )
            for row in v61["predictions"]:
                line = _line(row)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="white", linewidth=6.0)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#16a34a", linewidth=3.1)
            for birth_index, row in enumerate(births, start=1):
                line = _line(row)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="white", linewidth=7.0)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#00bcd4", linewidth=4.0)
                midpoint = line.mean(axis=0)
                axis.text(
                    midpoint[0], midpoint[1], midpoint[2], f"B{birth_index}",
                    fontsize=8, fontweight="bold", color="#075985",
                )
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.set_axis_off()
            axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
        figure.suptitle(
            f"{case_id} TRUTH-FREE V61→V62 REVIEW | green V61 | cyan V62 births ({len(births)})",
            fontsize=17, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        output = args.output_root / f"{case_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", pad_inches=0.05, facecolor=figure.get_facecolor())
        plt.close(figure)
        records.append(
            {
                "case_id": case_id,
                "v61_count": len(v61["predictions"]),
                "v62_count": len(v62["predictions"]),
                "v62_birth_count": len(births),
                "image": str(output.resolve()),
                "image_sha256": _sha(output),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v62_v4_truth_free_visual_comparison",
        "truth_opened": False,
        "changed_case_count": len(records),
        "records": records,
    }
    path = args.output_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_case_count": len(records), "birth_count": sum(row["v62_birth_count"] for row in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
