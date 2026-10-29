#!/usr/bin/env python3
"""Render native-mesh PSRO station/ray diagnostics for a terminal failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_paired_skin_ribbon_ownership_v18 import NativeMeshProbeContext, probe_candidate
from scripts.run_f1_psro_v18_preflight import _source_adjacency_rows
from scripts.select_f1_mesh_route_v16 import select_unified


VIEWS = (
    (22.0, -65.0, "oblique front"),
    (28.0, 25.0, "oblique reverse"),
    (82.0, -90.0, "top"),
    (-28.0, -65.0, "underside"),
)


def _bound_rows(pool: dict, selection: dict) -> list[dict]:
    source_rows = _source_adjacency_rows(pool)
    incumbent_ids = {str(row.get("candidate_id") or "") for row in selection["prediction_rows"]}
    result = []
    for clustered in selection["rows"]:
        if str(clustered.get("representative_source") or "") != "scan_mesh_adjacency":
            continue
        if str(clustered.get("candidate_id") or "") in incumbent_ids:
            continue
        source = source_rows.get(str(clustered.get("source_candidate_id") or ""))
        if source is None:
            continue
        bound = dict(source)
        for key in ("candidate_id", "line_start", "line_end", "span_endpoints", "anchor", "axis_direction", "span_mm"):
            if clustered.get(key) is not None:
                bound[key] = clustered[key]
        result.append(bound)
    return result


def render(pool: dict, v9: dict, v4: dict, output: Path) -> dict:
    selection = select_unified(pool, v9, v4)
    mesh = o3d.io.read_triangle_mesh(str(pool["scan_path"]))
    context = NativeMeshProbeContext(mesh)
    rows = _bound_rows(pool, selection)
    records = [probe_candidate(context, row) for row in rows]

    vertices = context.vertices
    stride = max(1, len(vertices) // 120_000)
    visible = vertices[::stride]
    center = np.mean(vertices, axis=0)
    extent = float(np.max(np.ptp(vertices, axis=0)))
    colors = ("#e53935", "#ff8f00", "#8e24aa")
    fig = plt.figure(figsize=(18, 14), dpi=180)
    for panel, (elev, azim, title) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        ax.scatter(visible[:, 0], visible[:, 1], visible[:, 2], s=0.08, c="#aeb8c2", alpha=0.32, depthshade=False)
        for color, row, record in zip(colors, rows, records):
            endpoints = np.asarray(row["span_endpoints"], dtype=np.float64)
            ax.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color=color, linewidth=3.0)
            ax.text(*np.mean(endpoints, axis=0), str(row["candidate_id"]).split("::")[-1], color=color, fontsize=9, weight="bold")
            for side in record.sides:
                for station, (origin_face, hit_face) in enumerate(zip(side.origin_faces, side.hit_faces)):
                    if origin_face < 0:
                        continue
                    origin = context.face_centers[origin_face]
                    ax.scatter(*origin, s=13, c=color, marker="o", depthshade=False)
                    if hit_face >= 0:
                        hit = context.face_centers[hit_face]
                        ax.plot([origin[0], hit[0]], [origin[1], hit[1]], [origin[2], hit[2]], color=color, linewidth=1.0, alpha=0.9)
                        ax.scatter(*hit, s=16, c=color, marker="x", depthshade=False)
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(center[0] - extent / 2, center[0] + extent / 2)
        ax.set_ylim(center[1] - extent / 2, center[1] + extent / 2)
        ax.set_zlim(center[2] - extent / 2, center[2] + extent / 2)
        ax.set_axis_off()
    fig.suptitle(
        "C017 PSRO_V1 terminal zero-activation audit\n"
        "lines=candidates; circles=valid ray origins; crosses=opposite hits; missing stations emit no marker",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output),
        "candidate_count": len(records),
        "records": [record.to_summary() for record in records],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(
        json.loads(args.input.read_text(encoding="utf-8")),
        joblib.load(args.v9_model),
        joblib.load(args.v4_model),
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

