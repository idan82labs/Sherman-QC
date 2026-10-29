#!/usr/bin/env python3
"""Render the frozen V51 axis-normal falsification artifact.

For an active V51 append, slices are perpendicular to the incumbent axis. Scan
points close to the incumbent/selected cylindrical hypotheses are colored by
family. Inactive cases still receive an immutable artifact documenting that no
parallel-return append was made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.f1_multifamily_selector_v2.model import endpoints
from scripts.render_manual_bend_review import _points


FAMILY = re.compile(r"^(CB\d+)_")


def _family(candidate_id: str) -> str:
    leaf = candidate_id.rsplit("::", 1)[-1]
    match = FAMILY.match(leaf)
    return match.group(1) if match else "unknown"


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.85:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, seed)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(axis, first)
    return first, second


def _candidate(pool: dict, candidate_id: str) -> dict:
    rows = [row for row in pool.get("line_objects") or [] if str(row.get("candidate_id")) == candidate_id]
    if len(rows) != 1:
        raise ValueError(f"expected one candidate {candidate_id!r}, found {len(rows)}")
    return rows[0]


def render(scan: Path, prediction: Path, pool_path: Path, output: Path, case_id: str) -> dict:
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    audit = dict(payload.get("single_axis_parallel_return_recovery_audit") or {})
    points = _points(scan, 200_000)
    if not len(points):
        raise ValueError("scan has no finite points")

    output.parent.mkdir(parents=True, exist_ok=True)
    if audit.get("gate_active") is not True:
        fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
        ax.set_facecolor("#07101f")
        fig.patch.set_facecolor("#07101f")
        ax.text(0.5, 0.58, "V51 GATE INACTIVE", ha="center", va="center", color="#d9e7ff", fontsize=28, fontweight="bold")
        ax.text(0.5, 0.43, str(audit.get("reason") or "unspecified"), ha="center", va="center", color="#8ba8cf", fontsize=17)
        ax.text(0.5, 0.30, f"{case_id}  |  {len(points):,} source points inspected", ha="center", va="center", color="#6283af", fontsize=13)
        ax.axis("off")
        fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return {"case_id": case_id, "gate_active": False, "slice_count": 0, "source_point_count": int(len(points))}

    incumbent_id = str(audit["incumbent_candidate_id"])
    selected_id = str(audit["selected_candidate_id"])
    incumbent = endpoints(_candidate(pool, incumbent_id))
    selected = endpoints(_candidate(pool, selected_id))
    axis = incumbent[1] - incumbent[0]
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:
        raise ValueError("incumbent axis has zero length")
    unit = axis / length
    u, v = _basis(unit)
    origin = incumbent.mean(axis=0)
    axial = (points - origin) @ unit
    transverse_u = (points - origin) @ u
    transverse_v = (points - origin) @ v
    positions = np.linspace(-0.42 * length, 0.42 * length, 9)
    slab = max(0.0125 * length, 1.0)
    radial_limit = max(float(audit.get("transverse_offset_mm") or 0.0) * 2.75, 12.0)
    inc_center = incumbent.mean(axis=0)
    sel_center = selected.mean(axis=0)
    inc_uv = np.array([(inc_center - origin) @ u, (inc_center - origin) @ v])
    sel_uv = np.array([(sel_center - origin) @ u, (sel_center - origin) @ v])
    inc_family = _family(incumbent_id)
    sel_family = _family(selected_id)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=180)
    fig.patch.set_facecolor("#07101f")
    for ax, location in zip(axes.flat, positions):
        mask = np.abs(axial - location) <= slab
        radius = np.sqrt(transverse_u**2 + transverse_v**2)
        mask &= radius <= radial_limit
        x = transverse_u[mask]
        y = transverse_v[mask]
        ax.set_facecolor("#07101f")
        ax.scatter(x, y, s=1.0, color="#74879e", alpha=0.30, linewidths=0, rasterized=True)
        if len(x):
            inc_dist = np.sqrt((x - inc_uv[0]) ** 2 + (y - inc_uv[1]) ** 2)
            sel_dist = np.sqrt((x - sel_uv[0]) ** 2 + (y - sel_uv[1]) ** 2)
            support_radius = max(2.5, min(radial_limit * 0.18, 8.0))
            inc_support = inc_dist <= support_radius
            sel_support = sel_dist <= support_radius
            ax.scatter(x[inc_support], y[inc_support], s=2.4, color="#42d4ff", alpha=0.70, linewidths=0, rasterized=True)
            ax.scatter(x[sel_support], y[sel_support], s=2.4, color="#ff4fd8", alpha=0.70, linewidths=0, rasterized=True)
        ax.scatter(*inc_uv, s=125, marker="x", color="#42d4ff", linewidths=3.0, label=inc_family)
        ax.scatter(*sel_uv, s=125, marker="+", color="#ff4fd8", linewidths=3.0, label=sel_family)
        ax.set_title(f"axis station {location / length:+.2f} L  |  n={int(mask.sum())}", color="white", fontsize=13)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-radial_limit, radial_limit)
        ax.set_ylim(-radial_limit, radial_limit)
        ax.tick_params(colors="#6f88a9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#234162")
    fig.suptitle(
        f"{case_id} — V51 axis-normal stack\nincumbent {inc_family} (cyan ×) / appended {sel_family} (magenta +)",
        color="white",
        fontsize=22,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {
        "case_id": case_id,
        "gate_active": True,
        "slice_count": len(positions),
        "source_point_count": int(len(points)),
        "incumbent_family": inc_family,
        "selected_family": sel_family,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.scan.resolve(), args.prediction.resolve(), args.pool.resolve(), args.output.resolve(), args.case_id)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
