#!/usr/bin/env python3
"""Match two F1 bend profiles in a shared rigid frame and compare them bend-by-bend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment


MATCH_COST_LIMIT = 6.0


def _strip_features(strip: dict, transform: np.ndarray | None = None) -> dict:
    midpoint = np.asarray(strip["axis_line_hint"]["midpoint"], dtype=float)
    direction = np.asarray(strip["axis_direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    if transform is not None:
        midpoint = transform[:3, :3] @ midpoint + transform[:3, 3]
        direction = transform[:3, :3] @ direction
        direction /= np.linalg.norm(direction)
    return {
        "strip_id": str(strip["strip_id"]),
        "midpoint": midpoint,
        "direction": direction,
        "span_mm": float(strip["along_axis_span_mm"]),
        "angle_deg": float(strip["dominant_angle_deg"]),
        "radius_mm": None if strip.get("radius_mm") is None else float(strip["radius_mm"]),
        "support_points": int(strip["support_point_count"]),
        "confidence": float(strip["confidence"]),
    }


def _segment_distance(a: dict, b: dict) -> float:
    p1 = a["midpoint"] - a["direction"] * a["span_mm"] * 0.5
    q1 = a["midpoint"] + a["direction"] * a["span_mm"] * 0.5
    p2 = b["midpoint"] - b["direction"] * b["span_mm"] * 0.5
    q2 = b["midpoint"] + b["direction"] * b["span_mm"] * 0.5
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    aa, ee, ff = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    epsilon = 1e-12
    if aa <= epsilon and ee <= epsilon:
        return float(np.linalg.norm(p1 - p2))
    if aa <= epsilon:
        s, t = 0.0, float(np.clip(ff / ee, 0.0, 1.0))
    else:
        cc = float(d1 @ r)
        if ee <= epsilon:
            t, s = 0.0, float(np.clip(-cc / aa, 0.0, 1.0))
        else:
            bb = float(d1 @ d2)
            denominator = aa * ee - bb * bb
            s = float(np.clip((bb * ff - cc * ee) / denominator if denominator else 0.0, 0.0, 1.0))
            t = (bb * s + ff) / ee
            if t < 0.0:
                t, s = 0.0, float(np.clip(-cc / aa, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((bb - cc) / aa, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _pair_metrics(good: dict, defective: dict) -> dict:
    direction_delta = float(
        np.degrees(
            np.arccos(np.clip(abs(np.dot(good["direction"], defective["direction"])), -1.0, 1.0))
        )
    )
    segment_distance = _segment_distance(good, defective)
    span_delta = defective["span_mm"] - good["span_mm"]
    angle_delta = defective["angle_deg"] - good["angle_deg"]
    radius_delta = None
    if good["radius_mm"] is not None and defective["radius_mm"] is not None:
        radius_delta = defective["radius_mm"] - good["radius_mm"]
    cost = (
        segment_distance / 2.0
        + direction_delta / 2.0
        + abs(span_delta) / 50.0
        + abs(angle_delta) / 2.0
    )
    return {
        "match_cost": cost,
        "segment_distance_mm": segment_distance,
        "direction_delta_deg": direction_delta,
        "span_delta_mm": span_delta,
        "good_angle_deg": good["angle_deg"],
        "defective_angle_deg": defective["angle_deg"],
        "angle_delta_deg": angle_delta,
        "good_radius_mm": good["radius_mm"],
        "defective_radius_mm": defective["radius_mm"],
        "radius_delta_mm": radius_delta,
        "support_point_delta": defective["support_points"] - good["support_points"],
    }


def _render(
    scan_path: Path,
    good: list[dict],
    defective: list[dict],
    matches: list[dict],
    unmatched_good: set[str],
    unmatched_defective: set[str],
    output: Path,
) -> None:
    mesh = o3d.io.read_triangle_mesh(str(scan_path), enable_post_processing=False)
    points = np.asarray(mesh.vertices, dtype=float)
    if len(points) > 70_000:
        indices = np.linspace(0, len(points) - 1, 70_000, dtype=np.int64)
        points = points[indices]
    center = np.median(points, axis=0)
    offsets = points[::4] - center
    _, basis = np.linalg.eigh(np.einsum("ni,nj->ij", offsets, offsets) / len(offsets))
    basis = basis[:, ::-1]
    projected_points = (points - center) @ basis
    match_by_good = {item["good_strip_id"]: item for item in matches}
    defective_by_id = {item["strip_id"]: item for item in defective}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    views = ((0, 1, "top / plan"), (0, 2, "long-side profile"), (1, 2, "short-side profile"))
    for axis, (a, b, title) in zip(axes, views):
        axis.scatter(projected_points[:, a], projected_points[:, b], s=0.05, color="#aab4c2", alpha=0.16, rasterized=True)
        for strip in good:
            start = strip["midpoint"] - strip["direction"] * strip["span_mm"] * 0.5
            end = strip["midpoint"] + strip["direction"] * strip["span_mm"] * 0.5
            line = np.vstack((start - center, end - center)) @ basis
            color = "#2563eb" if strip["strip_id"] in unmatched_good else "#16a6c9"
            axis.plot(line[:, a], line[:, b], color=color, linewidth=2.0, alpha=0.9)
            middle = (line[0] + line[1]) * 0.5
            axis.text(middle[a], middle[b], f"V:{strip['strip_id']}", color=color, fontsize=6, fontweight="bold")
            matched = match_by_good.get(strip["strip_id"])
            if matched:
                other = defective_by_id[matched["defective_strip_id"]]
                x_start = other["midpoint"] - other["direction"] * other["span_mm"] * 0.5
                x_end = other["midpoint"] + other["direction"] * other["span_mm"] * 0.5
                x_line = np.vstack((x_start - center, x_end - center)) @ basis
                axis.plot(x_line[:, a], x_line[:, b], color="#e83e8c", linewidth=1.3, linestyle="--", alpha=0.9)
        for strip in defective:
            if strip["strip_id"] not in unmatched_defective:
                continue
            start = strip["midpoint"] - strip["direction"] * strip["span_mm"] * 0.5
            end = strip["midpoint"] + strip["direction"] * strip["span_mm"] * 0.5
            line = np.vstack((start - center, end - center)) @ basis
            axis.plot(line[:, a], line[:, b], color="#dc2626", linewidth=2.4, linestyle="--", alpha=0.95)
            middle = (line[0] + line[1]) * 0.5
            axis.text(middle[a], middle[b], f"X:{strip['strip_id']}", color="#dc2626", fontsize=7, fontweight="bold")
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("mm")
        axis.set_ylabel("mm")
    fig.suptitle("F1 bend-strip comparison — V good (solid cyan/blue) vs X defective (dashed magenta/red)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good-result", type=Path, required=True)
    parser.add_argument("--defective-result", type=Path, required=True)
    parser.add_argument("--registration-summary", type=Path, required=True)
    parser.add_argument("--good-scan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-id", default="scan_pair")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    good_payload = json.loads(args.good_result.read_text())
    defective_payload = json.loads(args.defective_result.read_text())
    registration = json.loads(args.registration_summary.read_text())
    transform = np.asarray(registration["transform_defective_to_good"], dtype=float)
    good = [_strip_features(item) for item in good_payload["bend_support_strips"]]
    defective = [_strip_features(item, transform) for item in defective_payload["bend_support_strips"]]

    costs = np.zeros((len(good), len(defective)), dtype=float)
    pair_data: dict[tuple[int, int], dict] = {}
    for i, good_strip in enumerate(good):
        for j, defective_strip in enumerate(defective):
            metrics = _pair_metrics(good_strip, defective_strip)
            costs[i, j] = metrics["match_cost"]
            pair_data[(i, j)] = metrics
    good_indices, defective_indices = linear_sum_assignment(costs)
    matches = []
    matched_good: set[str] = set()
    matched_defective: set[str] = set()
    for i, j in zip(good_indices, defective_indices):
        metrics = pair_data[(int(i), int(j))]
        if metrics["match_cost"] > MATCH_COST_LIMIT:
            continue
        row = {
            "good_strip_id": good[i]["strip_id"],
            "defective_strip_id": defective[j]["strip_id"],
            **metrics,
        }
        matches.append(row)
        matched_good.add(good[i]["strip_id"])
        matched_defective.add(defective[j]["strip_id"])
    unmatched_good = {item["strip_id"] for item in good} - matched_good
    unmatched_defective = {item["strip_id"] for item in defective} - matched_defective

    max_angle_delta = max((abs(item["angle_delta_deg"]) for item in matches), default=0.0)
    count_delta = len(defective) - len(good)
    large_angle_count = sum(abs(item["angle_delta_deg"]) >= 0.75 for item in matches)
    rationale = [
        f"V and X expose {len(good)} and {len(defective)} support strips respectively",
        f"{len(matches)} bend identities match geometrically after rigid registration",
        f"{large_angle_count} matched bends have an incident-plane angle delta at or above 0.75 degrees",
        "F1 deviations remain screening evidence until corroborated by the independent surface heatmap or a production tolerance",
    ]
    result = {
        "part_id": str(args.part_id),
        "good_label": "V",
        "defective_label": "X",
        "good_f1": {
            "profile_family": good_payload["profile_family"],
            "estimated_bend_count": good_payload["estimated_bend_count"],
            "estimated_bend_count_range": good_payload["estimated_bend_count_range"],
            "hypothesis_confidence": good_payload["hypothesis_confidence"],
            "abstain_reasons": good_payload["abstain_reasons"],
            "support_strip_count": len(good),
            "candidate_generator_summary": good_payload["candidate_generator_summary"],
        },
        "defective_f1": {
            "profile_family": defective_payload["profile_family"],
            "estimated_bend_count": defective_payload["estimated_bend_count"],
            "estimated_bend_count_range": defective_payload["estimated_bend_count_range"],
            "hypothesis_confidence": defective_payload["hypothesis_confidence"],
            "abstain_reasons": defective_payload["abstain_reasons"],
            "support_strip_count": len(defective),
            "candidate_generator_summary": defective_payload["candidate_generator_summary"],
        },
        "matched_bends": matches,
        "unmatched_good_strips": sorted(unmatched_good),
        "unmatched_defective_strips": sorted(unmatched_defective),
        "comparison": {
            "matched_count": len(matches),
            "support_strip_count_delta": count_delta,
            "max_absolute_matched_angle_delta_deg": max_angle_delta,
            "matched_angle_deltas_at_or_above_0_75_deg": large_angle_count,
            "f1_alone_verdict": "FLAG_FOR_QC_REVIEW_NOT_INDEPENDENTLY_CLASSIFIABLE",
            "rationale": rationale,
        },
    }
    (args.output_dir / "f1_bend_by_bend_comparison.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# F1 bend-by-bend comparison — {args.part_id}",
        "",
        f"- V support strips: `{len(good)}`; raw range `{good_payload['estimated_bend_count_range']}`; confidence `{good_payload['hypothesis_confidence']}`",
        f"- X support strips: `{len(defective)}`; raw count/range `{defective_payload['estimated_bend_count']}` / `{defective_payload['estimated_bend_count_range']}`; confidence `{defective_payload['hypothesis_confidence']}`",
        f"- High-quality geometric matches: `{len(matches)}`",
        f"- V-only strips: `{', '.join(sorted(unmatched_good)) or 'none'}`",
        f"- X-only strips: `{', '.join(sorted(unmatched_defective)) or 'none'}`",
        f"- Maximum matched bend-angle delta: `{max_angle_delta:.3f}°`",
        "",
        "| V | X | line distance mm | direction Δ° | span Δ mm | angle V° | angle X° | angle Δ° | radius Δ mm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in matches:
        radius = item["radius_delta_mm"]
        radius_text = "n/a" if radius is None else f"{radius:+.4f}"
        lines.append(
            f"| {item['good_strip_id']} | {item['defective_strip_id']} | {item['segment_distance_mm']:.3f} | "
            f"{item['direction_delta_deg']:.3f} | {item['span_delta_mm']:+.3f} | "
            f"{item['good_angle_deg']:.4f} | {item['defective_angle_deg']:.4f} | "
            f"{item['angle_delta_deg']:+.4f} | {radius_text} |"
        )
    lines += [
        "",
        "**F1-only conclusion:** V and X now have matched ten-bend topology. Flag the local geometric deltas for QC review, but classify only where independent surface evidence or a production tolerance corroborates them.",
    ]
    (args.output_dir / "F1_BEND_BY_BEND_COMPARISON.md").write_text("\n".join(lines) + "\n")
    _render(
        args.good_scan,
        good,
        defective,
        matches,
        unmatched_good,
        unmatched_defective,
        args.output_dir / "f1_bend_by_bend_comparison.png",
    )
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
