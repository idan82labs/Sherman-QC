#!/usr/bin/env python3
"""Render the mandatory synthetic bend regression failure for visual adjudication."""

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

from scripts.run_f1_amhc_v1_source_free_regressions import bent_double_skin, _run


def _scatter(ax, source, active, first, second, title):
    ax.scatter(source[:, first], source[:, second], s=0.4, c="#9aa4b2", alpha=0.16, linewidths=0)
    if len(active):
        ax.scatter(active[:, first], active[:, second], s=5.5, c="#ef233c", alpha=0.85, linewidths=0)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fbfcfe")
    ax.grid(color="#dce2ea", linewidth=0.35, alpha=0.5)


def main() -> int:
    _row, evaluation = _run("double_skin_90deg_bend", bent_double_skin())
    graph = evaluation.graph
    pairs = evaluation.pairs
    assert graph is not None and pairs is not None
    active_edges = graph.core_edges[graph.active_core]
    active = (
        (pairs.midpoints[active_edges[:, 0]] + pairs.midpoints[active_edges[:, 1]]) * 0.5
        if len(active_edges) else np.empty((0, 3))
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=256)
    _scatter(axes[0, 0], evaluation.source.xyz, active, 0, 2, "Side X–Z: active hinge current spans the full cylindrical arc")
    _scatter(axes[0, 1], evaluation.source.xyz, active, 0, 1, "Top X–Y: active field is an area strip, not a line")
    _scatter(axes[1, 0], evaluation.source.xyz, active, 2, 1, "End Z–Y: broad activation persists across sheet width")
    axes[1, 1].axis("off")
    counts = evaluation.semantic_payload["counts"]
    fractions = evaluation.semantic_payload["fractions"]
    component = max(evaluation.components, key=lambda row: row.edge_count)
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            (
                "AMHC V1 SOURCE-FREE BEND REGRESSION",
                "",
                f"decision: {evaluation.decision}",
                f"reciprocal pairs: {counts['reciprocal_pairs']}",
                f"core midpoints: {counts['core_midpoints']}",
                f"active core edges: {counts['active_core_edges']}",
                f"active fraction: {fractions['active_core_edges']:.6f}  (wall ≤ 0.05)",
                f"largest active component edges: {component.edge_count}",
                f"diameter: {component.diameter:.3f}",
                f"line-support fraction: {component.linear_fraction:.6f}  (wall ≥ 0.75)",
                "",
                "VISUAL RULING",
                "The intended radius-12 cylindrical bend produces a coherent",
                "two-dimensional red band covering the curved surface.",
                "This is not isolated boundary noise and not a renderer artifact.",
                "It directly contradicts the frozen one-dimensional line-support gate.",
            )
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=13,
    )
    fig.suptitle(
        "ANTIPODAL_MIDSURFACE_HINGE_CURRENT_V1 — mandatory positive fixture fails",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = ROOT / "output/research/amhc-v1-source-free-regression-stop"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "double-skin-90deg-cylindrical-bend-failure.png"
    fig.savefig(path, dpi=256, facecolor="white")
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
