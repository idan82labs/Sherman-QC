#!/usr/bin/env python3
"""Native-resolution visual audit of CFAC's mandatory full-cylinder preemption."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_cfac_v1.core import canonicalize_xyz, estimate_jets
from research.f1_cfac_v1.fixtures import concentric_bend


def _panel(ax, source, focal, first, second, title):
    ax.scatter(source[:, first], source[:, second], s=0.45, c="#9ca6b4", alpha=0.18, linewidths=0)
    ax.scatter(focal[:, first], focal[:, second], s=4.0, c="#e63946", alpha=0.78, linewidths=0)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d8dee8", linewidth=0.35, alpha=0.55)


def main() -> int:
    source = canonicalize_xyz(concentric_bend(angle_degrees=360.0, flange_length=0.0))
    jets = estimate_jets(source)
    focal = jets.focal_points[jets.focal_vote]
    cap = min(0.60 * len(source.xyz), 120_000)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=256)
    _panel(axes[0, 0], source.xyz, focal, 0, 2, "Cross-section X–Z: two source cylinders collapse to the same center")
    _panel(axes[0, 1], source.xyz, focal, 0, 1, "X–Y: focal votes form the cylinder axis")
    _panel(axes[1, 0], source.xyz, focal, 2, 1, "Z–Y: focal votes remain visibly line-like")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            (
                "CFAC V1 FULL-CYLINDER DECOY",
                "",
                f"source points: {len(source.xyz)}",
                f"focal votes: {len(focal)}",
                f"focal fraction: {len(focal)/len(source.xyz):.6f}",
                f"frozen nonselective cap: {cap:.1f} ({cap/len(source.xyz):.6f})",
                "",
                "MANDATORY EXPECTATION",
                "Reach twin-radius stage, then fail occupied-arc width.",
                "",
                "OBSERVED FROZEN ORDER",
                "The 60% focal-vote cap stops the fixture before the focal graph.",
                "",
                "VISUAL CRITIQUE",
                "The red focal geometry is already the intended one-dimensional axis.",
                "The upstream cap rejects a visually correct collapse solely because",
                "a pure cylindrical decoy naturally makes nearly every source point vote.",
            )
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=13,
    )
    fig.suptitle(
        "CONCENTRIC_FOCAL_AXIS_COLLAPSE_V1 — mandatory decoy is preempted before its causal gate",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = ROOT / "output/research/cfac-v1-source-free-regression-stop"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "full-cylinder-focal-cap-preemption.png"
    fig.savefig(path, dpi=256, facecolor="white")
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
