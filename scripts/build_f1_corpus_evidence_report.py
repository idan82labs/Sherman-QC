#!/usr/bin/env python3
"""Build the honest frozen-corpus F1 evidence summary and visual index."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"
METRICS = BASE / "evaluation/research_metrics.json"


FAILURE_NOTES = {
    "C01": "Parallel ridge/lane substitution on a framed part; count is correct but geometry is not.",
    "C02": "Two supported folds recovered; two remaining axes are assigned to neighboring boundaries.",
    "C03": "Plane-lattice completion collapses six visible bends to two perimeter junctions.",
    "C04": "Exact: the single visible conventional bend is recovered with finite support.",
    "C05": "Three of four folds recovered; extra endpoint/parallel residuals remain.",
    "C06": "Mixed under-recovery and formed-transition over-generation on a partial scan.",
    "C07": "Severe parallel-surface lane ambiguity; correct count neighborhood, wrong physical axes.",
    "C08": "Perimeter outlines replace several short attachment bends.",
    "C09": "Two long folds recovered; short/local bends remain under-detected.",
    "C10": "Wall outlines and free edges are selected instead of the five attachment bends.",
    "C11": "Curved/rolled geometry triggers unsupported extrapolated line hypotheses.",
    "C12": "Both labeled bends recovered, plus two false parallel lines on curved geometry.",
    "C13": "Complex multi-flange topology is dominated by wrong parallel lanes and outline axes.",
}


def _build_case_chart(rows: list[dict], path: Path) -> None:
    labels = [row["case_token"] for row in rows]
    gold = np.asarray([row["gold_count"] for row in rows])
    pred = np.asarray([row["prediction_count"] for row in rows])
    tp = np.asarray([row["true_positive"] for row in rows])
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=180)
    ax.bar(x - width, gold, width, label="manual bends", color="#142033")
    ax.bar(x, pred, width, label="F1 predictions", color="#E59A13")
    ax.bar(x + width, tp, width, label="geometric true positives", color="#20A66A")
    ax.set_xticks(x, labels)
    ax.set_ylabel("finite bend-line instances")
    ax.set_title("Correct count is not sufficient: finite bend geometry must also match")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _build_contact_sheet(rows: list[dict], path: Path) -> None:
    thumb_w, thumb_h = 720, 540
    columns = 3
    rows_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows_count * thumb_h), "#071729")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        source = Image.open(row["review_render"]).convert("RGB")
        source.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w + (thumb_w - source.width) // 2
        y = (index // columns) * thumb_h + (thumb_h - source.height) // 2
        sheet.paste(source, (x, y))
        draw.rectangle((x, y, x + source.width - 1, y + source.height - 1), outline="#D6DEE8", width=2)
    sheet.save(path, quality=92)


def _build_ablation_chart(ablation: list[dict], path: Path) -> None:
    labels = [row["label"] for row in ablation]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    for offset, key, label, color in (
        (-width, "precision", "precision", "#142033"),
        (0.0, "recall", "recall", "#E59A13"),
        (width, "line_f1", "finite-line F1", "#20A66A"),
    ):
        values = [row[key] for row in ablation]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 0.36)
    ax.set_ylabel("score")
    ax.set_title("Ablation on the identical frozen 13-scan corpus")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = list(data["cases"])
    chart = BASE / "evaluation/case_level_counts_vs_geometry.png"
    contact = BASE / "evaluation/all_13_gold_vs_f1_contact_sheet.jpg"
    ablation_chart = BASE / "evaluation/completion_ablation.png"
    _build_case_chart(rows, chart)
    _build_contact_sheet(rows, contact)
    ablation = []
    for key, label, metrics_path in (
        ("base", "base", BASE / "ablation/base/evaluation/research_metrics.json"),
        ("formed", "formed only", BASE / "ablation/formed/evaluation/research_metrics.json"),
        ("prismatic", "lattice only", BASE / "ablation/prismatic/evaluation/research_metrics.json"),
        ("both", "both", METRICS),
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["micro_bend_metrics"]
        ablation.append({"key": key, "label": label, **metrics})
    _build_ablation_chart(ablation, ablation_chart)
    operations = json.loads((BASE / "inference/inference_summary.json").read_text(encoding="utf-8"))["operations"]
    max_rss = max(int(row["maximum_resident_set_bytes"]) for row in operations)
    max_wall = max(float(row["wall_time_sec"]) for row in operations)
    micro = data["micro_bend_metrics"]
    lines = [
        "# F1 Frozen-Corpus Evidence Report — 2026-07-29",
        "",
        "## Bottom line",
        "",
        "The current uniformly applied zero-shot profile does **not** generalize from the corrected 38172000 result to the full verified corpus.",
        "",
        f"- 13 verified scan records, representing {data['sample']['unique_part_family_count']} independent part families.",
        f"- {micro['true_positive']}/{micro['gold_bend_count']} manual bend axes recovered.",
        f"- Precision {micro['precision']:.3f}, recall {micro['recall']:.3f}, finite-line F1 {micro['line_f1']:.3f}.",
        f"- Geometrically exact scans: {data['exact_scan_metrics']['exact_scans']}/{data['exact_scan_metrics']['total_scans']} ({data['exact_scan_metrics']['rate']:.1%}).",
        f"- Count-only exact scans: {data['count_only_metrics']['count_exact_scans']}/{data['count_only_metrics']['total_scans']}; {data['count_only_metrics']['geometrically_false_exact_count_scans']} of those have the right count but the wrong physical axes.",
        f"- Peak isolated-job RSS {max_rss/(1024**3):.2f} GiB; maximum wall time {max_wall:.1f} s.",
        "",
        "The earlier 12/13 corpus-fit selector is excluded from research accuracy claims because it was explicitly trained and scored on these same manual labels.",
        "",
        f"![Case-level evidence]({chart})",
        "",
        "## Completion ablation",
        "",
        "| Variant | TP | FP | FN | Precision | Recall | Finite-line F1 | Exact scans |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ablation:
        lines.append(
            f"| {row['label']} | {row['true_positive']} | {row['false_positive']} | {row['false_negative']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['line_f1']:.3f} | 1/13 |"
        )
    lines += [
        "",
        f"![Completion ablation]({ablation_chart})",
        "",
        "Formed-transition completion adds two true positives but fifteen false positives relative to base. Plane-lattice completion removes seven false positives without increasing true positives. Their combination is the strongest line-level variant, but the 0.033 absolute F1 gain does not improve exact-scan accuracy.",
        "",
        "## Geometry-only optimization checkpoint",
        "",
        "The clean-cache patch graph improved the frozen-corpus diagnostic to TP 28 / FP 42 / FN 43 (F1 0.397, 1/13 exact). A part-family-grouped split was then frozen before routing work. The first router improved development F1 to 0.454 but reduced the then-unopened holdout from 0.473 to 0.383, so that router was rejected.",
        "",
        "All subsequent corpus-wide numbers are explicitly **post-holdout diagnostics**, not unbiased generalization estimates. The accepted v7 diagnostic uses only geometry and never reads a part identifier, expected bend count, or manual truth at inference time:",
        "",
        "| Checkpoint | TP | FP | FN | Precision | Recall | F1 | Exact scans | Evidence class |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        "| frozen uniform zero-shot | 19 | 54 | 52 | 0.260 | 0.268 | 0.264 | 1/13 | pre-optimization frozen evidence |",
        "| clean-cache patch graph | 28 | 42 | 43 | 0.400 | 0.394 | 0.397 | 1/13 | pre-router diagnostic |",
        "| routed v5 | 37 | 39 | 34 | 0.487 | 0.521 | 0.503 | 2/13 | post-holdout diagnostic |",
        "| routed v7 | 38 | 39 | 33 | 0.494 | 0.535 | 0.514 | 2/13 | post-holdout diagnostic |",
        "| routed v9 finite wedge | 39 | 37 | 32 | 0.513 | 0.549 | 0.531 | 2/13 | post-holdout diagnostic |",
        "| routed v11 supported boundary | 40 | 37 | 31 | 0.519 | 0.563 | 0.541 | 2/13 | post-holdout diagnostic |",
        "| routed v14 complete topology | 42 | 33 | 29 | 0.560 | 0.592 | 0.575 | 3/13 | post-holdout diagnostic |",
        "",
        "V14 retains the earlier typed formed-feature separation, geometry-only source routing, Pareto incident-plane snapping, and interface-birth consensus. It adds finite wedge stitching, supported parallel-skin snapping, neighbor-graph transition-boundary fitting, axial support-lobe splitting, and a fail-closed complete-topology replacement. Together these changes move C10 from 0/5 to exact 5/5 without changing the diagnostic holdout or the prior exact controls.",
        "",
        "The design follows established point-cloud feature extraction principles: recover crease patterns from a neighborhood graph and local wedges; preserve sharp features under robust local fitting; fit finite line segments on planar support rather than infinite ridges; and reject weak boundary support. Primary references: [Gumhold, Wang, and MacLeod (2001)](https://www.researchgate.net/publication/2554207_Feature_Extraction_from_Point_Clouds), [Öztireli, Guennebaud, and Gross (2009)](https://diglib.eg.org/items/38de8cf6-2183-4a47-bbf2-ca179cb64725), [Lu, Liu, and Li (2019)](https://arxiv.org/abs/1901.02532), and [Mineo, Pierce, and Summan (2019)](https://doi.org/10.1016/j.jcde.2018.02.001).",
        "",
        f"Accepted routed results: `{BASE / 'routed_v14_post_holdout_complete_topology'}`.",
        "",
        "## Bend-level evidence scorecard",
        "",
        "V14 bend objects were matched back to a dense scan-only evidence pool and assigned an interpretable 0-100 score from finite support, endpoint support, two-sided normal transition, normal coherence, cross-source agreement, line-to-scan distance, solver stability, and topology provenance. The score is not a probability and does not use truth at inference time.",
        "",
        "A development-only sweep found that threshold 70 raises development precision from 0.644 to 0.706 but lowers recall from 0.630 to 0.522 and reduces exact scans from three to two. On the already-open diagnostic holdout the same fixed threshold produces only 0.480 precision and 0.480 recall. Therefore the score is approved for rating and review prioritization, not as a global automatic rejection gate.",
        "",
        f"Scorecard report: `{BASE / 'scorecard_v1/F1_BEND_EVIDENCE_SCORECARD_REPORT.md'}`.",
        "",
        "## Leakage controls",
        "",
        "Each scan was frozen by SHA-256 and passed to a fresh process through an opaque `input.ply` or `input.stl` alias. The inference process received no part ID, expected bend count, manual axes, or promotion metadata. Scoring began only after all thirteen inference jobs finished.",
        "",
        "Primary one-to-one matching gates were fixed before scoring: ≤5° axis angle, ≤4 mm symmetric transverse offset, ≥60% gold-span overlap, and ≥60% prediction-span overlap.",
        "",
        "## Per-case results",
        "",
        "| Case | Part family | State | Gold | Pred | TP | FP | FN | Exact | Visual diagnosis |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_token']} | {row['part_key']} | {row['scan_state']} | {row['gold_count']} | {row['prediction_count']} | {row['true_positive']} | {row['false_positive']} | {row['false_negative']} | {'yes' if row['exact'] else 'no'} | {FAILURE_NOTES[row['case_token']]} |"
        )
    lines += [
        "",
        "## Visual audit index",
        "",
        "Green solid/cyan dashed indicates a matched manual/F1 axis. Orange is a missed manual bend. Red dashed is a false F1 axis.",
        "",
        f"![All 13 visual audits]({contact})",
        "",
        "## Research interpretation",
        "",
        "The strongest supported result is currently **proof of feasibility on selected geometries**, not corpus-level accuracy. The primary scientific blocker is semantic axis ownership: the detector often finds real planar transition evidence but assigns the bend to a neighboring thickness surface, free outline, or parallel ridge. Short attachment bends and curved/rolled families form distinct secondary failure modes.",
        "",
        "The next development cycle must use part-family-grouped validation. No change should be reported as an improvement unless it raises held-out finite-line F1 and exact geometry without using the held-out family’s labels or count.",
        "",
        "## Ground-truth limitation",
        "",
        "These labels are registry-approved manual visual annotations, but the study should obtain an independent second human review and inter-rater agreement before publication. Repeated scans of 10839000 and 38172000 are correlated observations and are therefore grouped by part family in uncertainty calculations.",
    ]
    report = BASE / "F1_FROZEN_CORPUS_EVIDENCE_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "case_chart": str(chart), "ablation_chart": str(ablation_chart), "contact_sheet": str(contact)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
