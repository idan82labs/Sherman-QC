#!/usr/bin/env python3
"""Render six-view gold-versus-F1 bend-axis review sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    CASE_JOBS,
    Thresholds,
    _discover_isolated_case_map,
    _endpoints,
    match_lines,
)
from scripts.render_manual_bend_review import _points, _stable_pca, _view_basis  # noqa: E402


def render_case(
    *,
    case_id: str,
    scan_path: Path,
    gold_path: Path,
    prediction_path: Path,
    output_path: Path,
    max_points: int,
) -> dict[str, object]:
    points = _points(scan_path, max_points)
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    gold = json.loads(gold_path.read_text(encoding="utf-8"))["manual_bends"]
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))["owned_bend_regions"]
    result = match_lines(gold, predictions, Thresholds())
    matched_gold = {item["gold_id"] for item in result["matches"]}
    matched_predictions = {item["prediction_id"] for item in result["matches"]}
    gold_lines = [(str(item["bend_id"]), (_endpoints(item) - center) @ pca) for item in gold]
    prediction_lines = [
        (str(item["bend_id"]), (_endpoints(item) - center) @ pca) for item in predictions
    ]

    views = (
        ("PC1 / PC2", np.eye(3)),
        ("PC1 / PC3", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)),
        ("PC2 / PC3", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)),
        ("Oblique front", _view_basis(np.array([1.0, 1.0, 0.75]), np.array([0.0, 0.0, 1.0]))),
        ("Oblique reverse", _view_basis(np.array([-1.0, 0.8, 0.65]), np.array([0.0, 0.0, 1.0]))),
        ("Oblique underside", _view_basis(np.array([0.8, -1.0, -0.7]), np.array([0.0, 0.0, 1.0]))),
    )
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=180)
    fig.patch.set_facecolor("#07101f")
    for ax, (title, view) in zip(axes.flat, views):
        view_points = projected @ view
        depth = view_points[:, 2]
        order = np.argsort(depth)
        normalized = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        colors = plt.get_cmap("bone")(0.24 + 0.64 * normalized)
        ax.scatter(
            view_points[order, 0],
            view_points[order, 1],
            s=0.26,
            c=colors[order],
            alpha=0.38,
            linewidths=0,
            rasterized=True,
        )
        for bend_id, endpoints in gold_lines:
            line = endpoints @ view
            color = "#39ff88" if bend_id in matched_gold else "#ff9f1c"
            ax.plot(line[:, 0], line[:, 1], color="black", linewidth=9.0, alpha=0.76, solid_capstyle="round")
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=6.0, alpha=0.92, solid_capstyle="round")
        for bend_id, endpoints in prediction_lines:
            line = endpoints @ view
            color = "#42d4ff" if bend_id in matched_predictions else "#ff3b6b"
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=color,
                linewidth=2.5,
                linestyle=(0, (5, 2)),
                solid_capstyle="round",
                zorder=5,
            )
        ax.set_title(title, color="white", fontsize=15, fontweight="bold")
        ax.set_facecolor("#07101f")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    status = (
        f"TP {result['true_positive']}  FP {result['false_positive']}  FN {result['false_negative']}"
        + ("  EXACT" if result["exact"] else "  MISMATCH")
    )
    fig.suptitle(
        f"{case_id} — gold solid / F1 dashed — {status}",
        color="white",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "matched gold: green  |  missed gold: orange  |  matched F1: cyan  |  false F1: red",
        ha="center",
        color="white",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {"case_id": case_id, **result, "render_path": str(output_path)}


def _discover_label_cases(label_root: Path) -> list[tuple[str, Path, Path]]:
    cases = []
    for label_path in sorted(label_root.glob("[0-9][0-9]_*/*bend_instance_labels.json")):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or label_path.parent.name)
        scan_path = Path(str(payload.get("scan_path") or "")).expanduser()
        if not scan_path.is_file():
            raise ValueError(f"{case_id}: label scan_path is missing: {scan_path}")
        cases.append((case_id, label_path, scan_path))
    if not cases:
        raise ValueError(f"no bend-instance labels found under {label_root}")
    if len({case_id for case_id, _label, _scan in cases}) != len(cases):
        raise ValueError("label root contains duplicate case IDs")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--gold-manifest", type=Path)
    parser.add_argument("--inference-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()
    rows = []
    if args.gold_manifest is not None:
        if args.inference_summary is None:
            parser.error("--gold-manifest requires --inference-summary")
        gold = json.loads(args.gold_manifest.resolve().read_text(encoding="utf-8"))
        inference = json.loads(args.inference_summary.resolve().read_text(encoding="utf-8"))
        cases = []
        for item in gold.get("cases") or []:
            case_id = str(item["case_token"])
            input_candidates = sorted(
                Path(inference["result_map"][case_id]["zero_shot_result"]).parents[1].glob("input.*")
            )
            if len(input_candidates) != 1:
                raise ValueError(f"{case_id}: expected exactly one opaque inference input")
            cases.append((case_id, Path(item["label_path"]), input_candidates[0]))
    elif args.label_root is not None:
        cases = _discover_label_cases(args.label_root.resolve())
    else:
        if args.clean_root is None:
            parser.error("one of --clean-root or --label-root is required")
        clean_map = _discover_isolated_case_map(args.clean_root.resolve())
        cases = []
        for case_id, clean_path in clean_map.items():
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            cases.append(
                (
                    case_id,
                    ROOT / "output" / CASE_JOBS[case_id] / "bend_instance_labels.json",
                    Path(clean["scan_path"]),
                )
            )
    for case_id, gold_path, scan_path in cases:
        prediction_candidates = (
            args.prediction_root.resolve() / case_id / "decomposition_result.json",
            args.prediction_root.resolve() / "cases" / case_id / "prediction_owned_regions.json",
            args.prediction_root.resolve() / case_id / "prediction_owned_regions.json",
        )
        prediction_matches = [path for path in prediction_candidates if path.is_file()]
        if len(prediction_matches) != 1:
            raise ValueError(
                f"{case_id}: expected exactly one prediction artifact, found {prediction_matches}"
            )
        rows.append(
            render_case(
                case_id=case_id,
                scan_path=scan_path,
                gold_path=gold_path,
                prediction_path=prediction_matches[0],
                output_path=args.output_dir.resolve() / f"{case_id}.png",
                max_points=args.max_points,
            )
        )
    (args.output_dir.resolve() / "review.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "mismatch_count": sum(not row["exact"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
