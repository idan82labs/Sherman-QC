#!/usr/bin/env python3
"""Render V58 raw births projected into native calibrated depth frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    births = [row for row in prediction["predictions"] if row.get("selection_route") == "calibrated_multiview_turn_consensus_v58"]
    if not births:
        raise ValueError("no raw V58 births")
    fig, axes = plt.subplots(len(births), len(manifest["frames"]), figsize=(24, 3.5 * len(births)), squeeze=False)
    for birth_index, birth in enumerate(births):
        start, end = np.asarray(birth["line_start"], dtype=float), np.asarray(birth["line_end"], dtype=float)
        samples = start[None, :] + np.linspace(0.0, 1.0, 256)[:, None] * (end - start)[None, :]
        for frame_index, frame in enumerate(manifest["frames"]):
            depth = np.asarray(np.load(args.bundle / frame["depth_file"], allow_pickle=False), dtype=float)
            valid = np.asarray(np.load(args.bundle / frame["validity_file"], allow_pickle=False), dtype=bool)
            K = np.asarray(frame["K"], dtype=float)
            T = np.asarray(frame["T_world_from_camera"], dtype=float)
            camera = (samples - T[:3, 3]) @ T[:3, :3]
            projected = camera @ K.T
            uv = projected[:, :2] / projected[:, 2:3]
            inside = (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            image = np.full_like(depth, np.nan)
            image[valid] = depth[valid]
            axis = axes[birth_index, frame_index]
            axis.imshow(image, cmap="gray_r", interpolation="nearest")
            if np.any(inside):
                axis.plot(uv[inside, 0], uv[inside, 1], color="#00e5ff", linewidth=2.0)
            axis.set_title(f"view {frame_index} | projected={int(np.count_nonzero(inside))}/256")
            axis.set_axis_off()
    fig.suptitle(f"{manifest['opaque_case_id']} native 384x288 depth | cyan V58 raw birth", fontsize=16)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"birth_count": len(births), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
