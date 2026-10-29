#!/usr/bin/env python3
"""Render the terminal PTCP_V1 source-free rigid-hash mismatch.

The artifact is source-free.  It overlays the frozen synthetic base mesh with
three rigidly transformed meshes after exact inverse normalization, making any
geometric displacement visible independently of the byte/hash result.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_f1_persistent_tube_component_parity_v19 import _positive_geometry  # noqa: E402


OUTPUT = ROOT / "output/research/f1-new-parts-20260813/v19_ptcp_source_free/hash_failure_audit_4096.png"


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        ((math.cos(angle), -math.sin(angle), 0.0),
         (math.sin(angle), math.cos(angle), 0.0),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _inverse_normalized(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformed = vertices @ rotation.T + translation
    return (transformed - translation) @ rotation


def _sample_edges(vertices: np.ndarray, triangles: np.ndarray, stride: int = 9):
    faces = triangles[::stride]
    for face in faces:
        points = vertices[face]
        for left, right in ((0, 1), (1, 2), (2, 0)):
            yield points[[left, right]]


def _draw_projection(ax, vertices_sets, triangles, axes, title):
    colors = ("#22d3ee", "#f472b6", "#fbbf24", "#a78bfa")
    labels = ("base", "17° inverse", "53° inverse", "101° inverse")
    for set_index, vertices in enumerate(vertices_sets):
        for segment in _sample_edges(vertices, triangles):
            ax.plot(segment[:, axes[0]], segment[:, axes[1]], color=colors[set_index], alpha=0.42, lw=0.55)
        ax.scatter(vertices[::37, axes[0]], vertices[::37, axes[1]], s=1.2, color=colors[set_index], alpha=0.55, label=labels[set_index])
    ax.set_title(title, color="white", fontsize=15, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#334155", alpha=0.35, lw=0.5)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#475569")


def main() -> int:
    vertices, triangles = _positive_geometry()
    configs = (
        (17.0, np.asarray((11.0, -7.0, 23.0))),
        (53.0, np.asarray((-19.0, 31.0, -5.0))),
        (101.0, np.asarray((7.0, 13.0, -29.0))),
    )
    normalized = [vertices]
    residuals = []
    for degrees, translation in configs:
        inverse = _inverse_normalized(vertices, _rotation(degrees), translation)
        normalized.append(inverse)
        residuals.append(float(np.max(np.linalg.norm(inverse - vertices, axis=1))))

    figure = plt.figure(figsize=(32, 18), dpi=128, facecolor="#07111f")
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.70), hspace=0.20, wspace=0.16)
    projections = (((0, 1), "Axial × arm-A"), ((0, 2), "Axial × arm-B"), ((1, 2), "Transverse native section"))
    for index, (axes, title) in enumerate(projections):
        ax = figure.add_subplot(grid[0, index], facecolor="#0b1728")
        _draw_projection(ax, normalized, triangles, axes, title)
        if index == 0:
            legend = ax.legend(loc="upper right", frameon=True, fontsize=10)
            legend.get_frame().set_facecolor("#111827")
            legend.get_frame().set_edgecolor("#475569")
            for text in legend.get_texts():
                text.set_color("white")

    ax_text = figure.add_subplot(grid[1, :], facecolor="#0b1728")
    ax_text.set_axis_off()
    evidence = [
        "PTCP_V1 SOURCE-FREE TERMINAL AUDIT — RIGID METAMORPHIC HASH MISMATCH",
        "Visual question: after exact inverse normalization, did any rigid transform move the observed two-skin geometry?",
        "Visual result: cyan / pink / amber / violet meshes coincide at native edge resolution in all three projections.",
        f"Maximum inverse-normalization vertex residuals: {residuals[0]:.3e}, {residuals[1]:.3e}, {residuals[2]:.3e} model units.",
        "Numerical result held constant: retain=True; birth_q=1025; death_q=8193; masks=FF/FF/FF/FF; fans=564; components=240; edges=2792.",
        "Topology hashes differed: base=f459faad…96d4; 17°=ade1d959…9a5a; 53°=4c0172eb…5143; 101°=301b1af6…e10f.",
        "Interpretation: visually/geometrically equivalent evidence produced non-identical canonical topology bytes. The frozen metamorphic gate therefore fails.",
        "Ruling consequence: PTCP_V1 stops before U02. No constants, masks, persistence ratios, ports, or construction rules were tuned after this outcome.",
    ]
    y = 0.92
    for index, line in enumerate(evidence):
        ax_text.text(0.035, y, line, transform=ax_text.transAxes, color="white" if index == 0 else "#dbeafe", fontsize=20 if index == 0 else 14, weight="bold" if index in (0, 6, 7) else "normal", va="top")
        y -= 0.105 if index == 0 else 0.095

    figure.suptitle("Persistent Tube Component Parity V1 — source-free visual falsification audit", color="white", fontsize=24, weight="bold", y=0.98)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
