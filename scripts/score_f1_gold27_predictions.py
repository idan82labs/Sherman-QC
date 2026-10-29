#!/usr/bin/env python3
"""Score a frozen gold27 prediction root with part- and truth-tier summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402
from scripts.run_f1_gold27_grouped_selector import _aggregate  # noqa: E402


BASE = ROOT / "output/research/f1-gold27-20260803"
DEFAULT_GOLD = BASE / "freeze/sealed_gold_manifest.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_part_family_metrics(
    predictions: dict[str, list[dict[str, Any]]],
    gold: dict[str, list[dict[str, Any]]],
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate line and exact-count metrics with equal family weight."""

    tokens_by_family: dict[str, list[str]] = {}
    for token, case in cases.items():
        tokens_by_family.setdefault(str(case["part_key"]), []).append(token)
    rows = []
    for family in sorted(tokens_by_family):
        tokens = sorted(tokens_by_family[family])
        metrics = _aggregate(
            {token: predictions[token] for token in tokens},
            {token: gold[token] for token in tokens},
        )
        rows.append(
            {
                "part_family": family,
                **{key: value for key, value in metrics.items() if key != "cases"},
                "exact_count_accuracy": round(
                    float(metrics["exact_scans"]) / max(int(metrics["scan_count"]), 1),
                    6,
                ),
                "case_tokens": tokens,
            }
        )
    macro_fields = ("precision", "recall", "line_f1", "exact_count_accuracy")
    macro = {
        field: round(sum(float(row[field]) for row in rows) / max(len(rows), 1), 6)
        for field in macro_fields
    }
    return {
        "family_count": len(rows),
        "family_macro": macro,
        "minimum_family_recall": round(
            min((float(row["recall"]) for row in rows), default=0.0), 6
        ),
        "families_below_0_80_recall": [
            str(row["part_family"]) for row in rows if float(row["recall"]) < 0.80
        ],
        "families_with_zero_recall": [
            str(row["part_family"]) for row in rows if float(row["recall"]) == 0.0
        ],
        "families": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sealed = _read(args.gold_manifest.resolve())
    cases = {str(case["case_token"]): case for case in sealed["cases"]}
    predictions = {
        token: list(
            _read(args.prediction_root.resolve() / "cases" / token / "prediction_owned_regions.json")[
                "owned_bend_regions"
            ]
        )
        for token in cases
    }
    gold = {token: list(case["manual_bends"]) for token, case in cases.items()}
    metrics = _aggregate(predictions, gold)
    rows = {str(row["case_token"]): row for row in metrics["cases"]}
    for token, row in rows.items():
        row.update(
            {
                "part_key": cases[token]["part_key"],
                "truth_tier": cases[token]["truth_tier"],
                "promotion_eligible": bool(cases[token]["promotion_eligible"]),
            }
        )
    tiers = {}
    tier_family_metrics = {}
    for tier in sorted({str(case["truth_tier"]) for case in cases.values()}):
        tier_tokens = [token for token, case in cases.items() if case["truth_tier"] == tier]
        tiers[tier] = {
            key: value
            for key, value in _aggregate(
                {token: predictions[token] for token in tier_tokens},
                {token: gold[token] for token in tier_tokens},
            ).items()
            if key != "cases"
        }
        tier_family_metrics[tier] = build_part_family_metrics(
            {token: predictions[token] for token in tier_tokens},
            {token: gold[token] for token in tier_tokens},
            {token: cases[token] for token in tier_tokens},
        )
    exact_families = sum(
        all(rows[token]["exact"] for token in cases if str(cases[token]["part_key"]) == part)
        for part in sorted({str(case["part_key"]) for case in cases.values()})
    )
    family_metrics = build_part_family_metrics(predictions, gold, cases)
    report = {
        "schema_version": 2,
        "kind": "sherman_qc_f1_gold27_prediction_score",
        "corpus_freeze_id": sealed["corpus_freeze_id"],
        "prediction_root": str(args.prediction_root.resolve()),
        "metrics": metrics,
        "truth_tier_metrics": tiers,
        "truth_tier_part_family_metrics": tier_family_metrics,
        "part_family_metrics": family_metrics,
        "exact_part_families": exact_families,
        "total_part_families": len({str(case["part_key"]) for case in cases.values()}),
    }
    output = args.output or (args.prediction_root.resolve() / "gold27_score.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": {key: value for key, value in metrics.items() if key != "cases"},
                "exact_part_families": exact_families,
                "part_family_macro": family_metrics["family_macro"],
                "minimum_family_recall": family_metrics["minimum_family_recall"],
                "families_below_0_80_recall": family_metrics[
                    "families_below_0_80_recall"
                ],
                "truth_tier_metrics": tiers,
                "report": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
