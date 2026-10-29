#!/usr/bin/env python3
"""Render the mandatory 4096-pixel PNCT_V1 source-free native audits."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_projective_normal_chord_transport_v20 import analyze_candidate  # noqa: E402
from research.f1_pnct_v20_fixtures import source_free_cases  # noqa: E402


COLORS = plt.cm.twilight(np.linspace(0.02, .98, 12))


def _row(name, length):
    return {"candidate_id": name, "candidate_key": hashlib.sha256(name.encode()).hexdigest()[:32], "span_endpoints": [[-length / 2, 0, 0], [length / 2, 0, 0]]}


def _record(name, vertices, triangles, angle, length):
    seeds = ((0, 1, 0), (0, math.cos(math.radians(angle)), math.sin(math.radians(angle))))
    return analyze_candidate(vertices=vertices, triangles=triangles, row=_row(name, length), scale=1.0, seed_normals=seeds)


def _segments(ax, station, title):
    for b, bin_ in enumerate(station.bins):
        for segment in station.segments:
            # A segment is colored by the closest bin direction for legibility.
            delta = (segment.angle - (b + .5) * math.pi / 24.0) % math.pi
            if min(delta, math.pi - delta) <= math.pi / 24.0:
                ax.plot([segment.p0[1], segment.p1[1]], [segment.p0[2], segment.p1[2]], color=COLORS[b], lw=.8, alpha=.75)
        for chord in bin_.chords:
            # Draw the short chord through its midpoint in the bin-normal direction.
            normal = np.asarray((math.cos((b + .5) * math.pi / 24), math.sin((b + .5) * math.pi / 24)))
            half = .5 * chord.distance * normal
            midpoint = chord.midpoint[1:]
            ax.plot([midpoint[0] - half[0], midpoint[0] + half[0]], [midpoint[1] - half[1], midpoint[1] + half[1]], color="#ff4d6d" if chord.ambiguous else "#14b8a6", lw=.5, alpha=.55)
    if station.midpoint_path:
        path = np.asarray([point[1:] for point in station.midpoint_path])
        ax.plot(path[:, 0], path[:, 1], color="#f59e0b", lw=1.8, marker=".", ms=2.5)
    ax.axhline(0, color="#475569", lw=.25); ax.axvline(0, color="#475569", lw=.25)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    status = "PASS" if station.final else station.reason.upper()
    ax.set_title(f"{title} · {status}\nowned {sum(b.owned for b in station.bins)}/12 · t={station.thickness or 0:.2f}", fontsize=7, color="#dcfce7" if station.final else "#fecaca")
    ax.set_facecolor("#0b1220")


def render(name, vertices, triangles, angle, length, output):
    record = _record(name, vertices, triangles, angle, length)
    arc = record.selected_arc or (record.arcs[0] if record.arcs else None)
    fig = plt.figure(figsize=(16, 16), dpi=256, facecolor="#070b12")
    grid = fig.add_gridspec(5, 4, height_ratios=(1.1, 1.1, 1, 1, .55), hspace=.36, wspace=.18)
    projections = ((0, 1, "XY / axial-tail"), (0, 2, "XZ / axial-tail"), (1, 2, "YZ / cross-section"))
    for column, (left, right, title) in enumerate(projections):
        ax = fig.add_subplot(grid[0, column])
        sample = vertices[::max(1, len(vertices)//12000)]
        ax.scatter(sample[:, left], sample[:, right], s=.22, color="#8dd3ff", alpha=.55, rasterized=True)
        if left == 0:
            ax.plot([-length/2, length/2], [0, 0], color="#f97316", lw=2)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_title(title, color="white", fontsize=10); ax.set_facecolor("#0b1220"); ax.tick_params(colors="#64748b", labelsize=6)
    info = fig.add_subplot(grid[0, 3]); info.axis("off"); info.set_facecolor("#0b1220")
    arc_failures = [item.failure_mask for item in record.arcs]
    info.text(0, 1, f"PNCT V20 · {name}\n\nretain: {record.retain}\nfailure mask: {record.failure_mask}\narc failures: {arc_failures}\nsegments/station: {record.debug.get('station_segment_counts')}\n\nLegend\n12 hues: projective bins\nteal: valid chords\nred: ambiguous chords\namber: midpoint transport\norange: candidate axis", va="top", color="white", family="monospace", fontsize=8)
    if arc:
        for index, station in enumerate(arc.stations):
            row = 1 + index // 4; column = index % 4
            _segments(fig.add_subplot(grid[row, column]), station, f"S{index:02d} x={station.coordinate:.2f}")
        summary = fig.add_subplot(grid[4, :]); summary.axis("off")
        owned = np.asarray([[int(bin_.owned) for bin_ in station.bins] for station in arc.stations])
        summary.imshow(owned, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        summary.set_yticks(range(11), [f"S{i:02d}" for i in range(11)], fontsize=6, color="white")
        summary.set_xticks(range(12), [f"B{i:02d}" for i in range(12)], fontsize=6, color="white")
        summary.set_title("Owned station/bin support — red cells are the frozen coverage failure", color="white", fontsize=9)
    fig.suptitle(f"Native-resolution PNCT audit · {name} · {'ACTIVATION' if record.retain else 'MISMATCH / REJECTION'}", color="white", fontsize=15, y=.995)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    positives, _ = source_free_cases()
    for name, case in positives.items():
        render(name, *case, args.output / f"{name}_audit_4096.png")


if __name__ == "__main__":
    main()
