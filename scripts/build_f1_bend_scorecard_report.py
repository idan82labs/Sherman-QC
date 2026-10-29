#!/usr/bin/env python3
"""Build visual research documentation for the intrinsic F1 bend scorecard."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"
OUT = BASE / "scorecard_v1"


def main() -> int:
    scorecard = json.loads((OUT / "bend_scorecard.json").read_text(encoding="utf-8"))
    metrics = json.loads((BASE / "routed_v14_post_holdout_complete_topology/score_all.json").read_text(encoding="utf-8"))
    split = json.loads((BASE / "freeze/grouped_development_holdout.json").read_text(encoding="utf-8"))
    development = {item["case_token"] for item in split["cases"] if item["split"] == "development"}
    matched = {row["case_token"]: {item["prediction_id"] for item in row["matches"]} for row in metrics["cases"]}
    true_scores = []
    false_scores = []
    for case in scorecard["cases"]:
        if case["case_token"] not in development:
            continue
        for bend in case["bends"]:
            score = float(bend["evidence"]["evidence_score"])
            (true_scores if bend["bend_id"] in matched[case["case_token"]] else false_scores).append(score)
    sweep = json.loads((OUT / "development_threshold_sweep.json").read_text(encoding="utf-8"))["sweep"]
    thresholds = np.asarray([row["threshold"] for row in sweep])
    precision = np.asarray([row["precision"] for row in sweep])
    recall = np.asarray([row["recall"] for row in sweep])
    f1 = np.asarray([row["line_f1"] for row in sweep])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    bins = np.arange(35, 101, 5)
    axes[0].hist(false_scores, bins=bins, alpha=0.72, color="#ef476f", label="false axes")
    axes[0].hist(true_scores, bins=bins, alpha=0.72, color="#20a66a", label="matched bends")
    axes[0].axvline(70, color="#142033", linestyle="--", linewidth=1.5, label="triage threshold 70")
    axes[0].set_xlabel("intrinsic evidence score")
    axes[0].set_ylabel("development bend objects")
    axes[0].set_title("Score overlap reveals semantic ownership errors")
    axes[0].legend(frameon=False)
    axes[1].plot(thresholds, precision, label="precision", color="#142033")
    axes[1].plot(thresholds, recall, label="recall", color="#e59a13")
    axes[1].plot(thresholds, f1, label="finite-line F1", color="#20a66a")
    axes[1].axvline(70, color="#666", linestyle="--", linewidth=1.2)
    axes[1].set_xlim(35, 95)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("acceptance threshold")
    axes[1].set_ylabel("development metric")
    axes[1].set_title("A hard threshold trades recall before fixing lane identity")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    chart = OUT / "development_score_threshold_diagnostics.png"
    fig.savefig(chart, facecolor="white")
    plt.close(fig)

    report = OUT / "F1_BEND_EVIDENCE_SCORECARD_REPORT.md"
    table = OUT / "bend_scorecard_table.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_token", "bend_id", "evidence_score", "evidence_rating", "review_queue",
                "parallel_lane_state", "parallel_lane_margin", "competitor_line_object_id",
                "finite_support", "endpoint_support", "two_sided_transition", "source_count",
            ),
        )
        writer.writeheader()
        for case in scorecard["cases"]:
            for bend in case["bends"]:
                evidence = bend["evidence"]
                ownership = evidence["parallel_lane_ownership"]
                features = evidence["features"]
                writer.writerow(
                    {
                        "case_token": case["case_token"],
                        "bend_id": bend["bend_id"],
                        "evidence_score": evidence["evidence_score"],
                        "evidence_rating": evidence["evidence_rating"],
                        "review_queue": evidence.get("review_queue"),
                        "parallel_lane_state": ownership.get("state"),
                        "parallel_lane_margin": ownership.get("margin"),
                        "competitor_line_object_id": ownership.get("competitor_line_object_id"),
                        "finite_support": features.get("finite_support_longest_run_fraction"),
                        "endpoint_support": features.get("endpoint_support_fraction"),
                        "two_sided_transition": features.get("two_sided_transition_score"),
                        "source_count": features.get("cluster_source_count"),
                    }
                )
    report.write_text(
        "\n".join(
            [
                "# F1 bend evidence scorecard - research checkpoint",
                "",
                "## Outcome",
                "",
                "Every final v14 bend axis now has a deterministic 0-100 intrinsic evidence score and A-D rating. The score uses scan geometry only: finite-line and endpoint support, two-sided normal transition, normal coherence, cross-source agreement, line-to-scan distance, solver stability, and topology provenance. It does not use part identity, expected bend count, or manual truth, and it is explicitly not a probability.",
                "",
                f"![Development score diagnostics]({chart})",
                "",
                "## Threshold finding",
                "",
                "| Operating point | TP | FP | FN | Precision | Recall | F1 | Exact scans |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                "| retain all / C-or-better | 29 | 16 | 17 | 0.644 | 0.630 | 0.637 | 3 |",
                "| threshold 70 (development triage) | 24 | 10 | 22 | 0.706 | 0.522 | 0.600 | 2 |",
                "| A-only (80) | 12 | 8 | 34 | 0.600 | 0.261 | 0.364 | 1 |",
                "",
                "Threshold 70 is useful only as a review-priority operating point. On the already-open diagnostic holdout it gives precision 0.480, recall 0.480, and F1 0.480. Therefore no single global threshold is approved as an automatic bend accept/reject gate.",
                "",
                "Parallel-lane ownership margin is exported separately. A stronger parallel competitor occurred for 9/29 development true bends and 5/16 false axes - essentially the same rate - so margin alone is also not a rejection rule. It is used to route semantic-lane cases to review while preserving the original evidence score.",
                "",
                "## What the 2001 paper changes for us",
                "",
                "Gumhold, Wang, and MacLeod worked with roughly 7,700 to 83,800 points and local graphs of 10-16 nearest neighbors. Our bounded analysis uses up to 200,000 points and can estimate local support, normal distributions, and finite endpoints more reliably. But their key observation still holds: laser scanning often produces no samples exactly on a physical crease, so more density does not by itself identify the correct semantic bend lane. The useful inheritance is the neighborhood graph, feature-pattern continuity, and two-plane wedge recovery; our advantage is enough density to score those signals at multiple radii and along the full finite axis.",
                "",
                "## Recommended use",
                "",
                "- Show the score and component breakdown beside every F1 marker.",
                "- Use >=70 to prioritize likely bends for human review, not to auto-accept them.",
                "- Route low score, small score-margin, and competing parallel-lane cases to manual adjudication.",
                "- Add a semantic ownership margin comparing the selected axis against its strongest parallel sibling; this is the missing signal exposed by C07.",
                "- Calibrate any production threshold only on a new external, family-grouped labeled corpus.",
                "",
                "## Resource profile",
                "",
                "The 13-scan dense evidence pass completed in 256 seconds with 3.16 GB maximum RSS. Source points were transient and were not serialized.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report), "chart": str(chart), "table": str(table)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
