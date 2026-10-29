#!/usr/bin/env python3
"""Light 4-view overlay PNG per scan: scan points grey, gold green, best_f1 red, runtime blue.

Everything is drawn in the raw-scan frame (gold mapped with the recovered frame record,
exactly as score.py scores it). Views are orthographic projections onto the scan's own
principal axes (top = PC1/PC2, front = PC1/PC3, side = PC2/PC3) plus one isometric view.
Predictions that the frozen matcher counted as TP are solid; FP predictions are dashed;
FN gold lines are drawn thicker. Point clouds are subsampled (deterministic) to keep it
light (matplotlib Agg).

  render_overlays.py --case-list .../case_list_broad_v2.json --runs-root .../runs \
      --scores-dir .../broad_baseline --out-root .../broad_baseline/overlays
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402

LANES = (("best_f1", "predictions_best_f1.json", "#d62728"), ("runtime", "predictions_runtime.json", "#1f5fd6"))
GOLD_COLOR = "#1a9c3a"


def _views(points: np.ndarray):
    center = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points[np.random.default_rng(0).choice(len(points), min(len(points), 20000),
                                                                       replace=False)] - center, full_matrices=False)
    e1, e2, e3 = vt[0], vt[1], vt[2]
    iso_x = (e1 - e2) / np.sqrt(2.0)
    iso_y = (e1 + e2 - 2.0 * e3) / np.sqrt(6.0)
    return center, [("top (PC1-PC2)", e1, e2), ("front (PC1-PC3)", e1, e3), ("side (PC2-PC3)", e2, e3),
                    ("isometric", iso_x, iso_y)]


def _load_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = bb.read_json(path)
    return [r for r in payload.get("owned_bend_regions") or [] if bb.valid_line(r.get("line_start"), r.get("line_end"))
            or r.get("span_endpoints")]


def render_case(entry: dict, *, runs_root: Path, scores: dict[str, dict], frame_records: dict | None,
                out_dir: Path, max_points: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import score as sc

    points = bb.load_scan_vertices(Path(entry["scan_path"]))
    rng = np.random.default_rng(12345)
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    center, views = _views(points)
    try:
        gold = sc.load_gold(entry["gold"], frame_records=frame_records)
        gold_lines, gold_note = gold["lines"], ""
    except sc.Excluded as error:
        gold_lines, gold_note = [], f" gold excluded: {error.reason}"
    lane_lines, lane_score = {}, {}
    for lane, fname, _color in LANES:
        lane_lines[lane] = _load_lines(runs_root / lane / entry["case_id"] / fname)
        lane_score[lane] = scores.get(lane, {}).get(entry["case_id"], {})
    fn_ids = set()
    for lane in ("best_f1",):
        fn_ids |= set(lane_score[lane].get("unmatched_gold_ids") or [])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=90)
    for ax, (name, ax_x, ax_y) in zip(axes.ravel(), views):
        rel = points - center
        x, y, depth = rel @ ax_x, rel @ ax_y, rel @ np.cross(ax_x, ax_y)
        order = np.argsort(depth)
        shade = (depth[order] - depth.min()) / max(1e-9, np.ptp(depth))
        ax.scatter(x[order], y[order], s=0.3, c=0.55 + 0.35 * shade, cmap="Greys_r", vmin=0, vmax=1,
                   linewidths=0, rasterized=True)

        def draw(line: dict, color: str, lw: float, ls: str = "-", z: int = 3) -> None:
            ends = bb.endpoints(line) - center
            ax.plot(ends @ ax_x, ends @ ax_y, color=color, lw=lw, ls=ls,
                    solid_capstyle="round", zorder=z)

        for line in gold_lines:
            draw(line, GOLD_COLOR, 5.0 if line["bend_id"] in fn_ids else 3.5, "-", 3)
        for lane, _fname, color in LANES:
            unmatched = set(lane_score[lane].get("unmatched_prediction_ids") or [])
            for index, line in enumerate(lane_lines[lane]):
                pid = str(line.get("bend_id") or f"P{index + 1:02d}")
                draw(line, color, 1.6, "--" if pid in unmatched else "-", 4)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    legend = [plt.Line2D([], [], color=GOLD_COLOR, lw=3.5, label=f"gold ({len(gold_lines)}; thick = missed by best_f1)")]
    for lane, _fname, color in LANES:
        s = lane_score[lane]
        if s.get("disposition") == "SCORED_COMPLETE_LABEL":
            text = f"{lane}: {s['prediction_count']} lines TP {s['true_positive']} FP {s['false_positive']} " \
                   f"FN {s['false_negative']} F1 {s['f1']:.3f}"
        else:
            text = f"{lane}: {len(lane_lines[lane])} lines ({s.get('disposition') or 'not scored'})"
        status = s.get("lane_status")
        if status and status != "OK":
            text += f" [{status}]"
        legend.append(plt.Line2D([], [], color=color, lw=1.6, label=text + " (dashed = FP)"))
    fig.legend(handles=legend, loc="lower center", ncol=1, fontsize=9, frameon=False)
    stage = "STAGED" if entry.get("staged_or_incomplete") else "full"
    fig.suptitle(f"{entry['case_id']}  family {entry['family_id']}  {entry['tier']}  {stage}  "
                 f"{Path(entry['scan_path']).name}{gold_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "overlay_4views.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--frame-records", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--only", action="append")
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--rss-limit-gib", type=float, default=9.0)
    args = parser.parse_args(argv)
    cases = bb.read_json(args.case_list)["cases"]
    frame_records = bb.read_json(args.frame_records) if args.frame_records else None
    scores = {}
    for lane, _f, _c in LANES:
        path = args.scores_dir / f"scores_{lane}.json"
        scores[lane] = {r["case_id"]: r for r in bb.read_json(path)["cases"]} if path.is_file() else {}
    for entry in cases:
        if args.only and entry["case_id"] not in args.only:
            continue
        bb.wait_for_rss_budget(args.rss_limit_gib)
        target = render_case(entry, runs_root=args.runs_root, scores=scores, frame_records=frame_records,
                             out_dir=args.out_root / entry["case_id"], max_points=args.max_points)
        print(target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
