#!/usr/bin/env python3
"""Compare two Gold37 scores with deterministic paired family resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


METRICS = ("precision", "recall", "line_f1", "exact_count_accuracy")
DEFAULT_RESAMPLES = 10_000


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family_rows(score: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload = dict(score.get("part_family_metrics") or {})
    rows = {
        str(row["part_family"]): dict(row)
        for row in payload.get("families") or ()
    }
    if len(rows) != int(payload.get("family_count") or 0):
        raise ValueError("family score rows are missing or duplicate")
    return rows


def _case_set(rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        str(token)
        for row in rows.values()
        for token in row.get("case_tokens") or ()
    }


def compare_scores(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    if int(resamples) < 1 or int(resamples) > 100_000:
        raise ValueError("resamples must be in [1, 100000]")
    baseline_freeze = str(baseline.get("corpus_freeze_id") or "")
    variant_freeze = str(variant.get("corpus_freeze_id") or "")
    if not baseline_freeze or baseline_freeze != variant_freeze:
        raise ValueError("score corpus freeze IDs differ")
    baseline_rows = _family_rows(baseline)
    variant_rows = _family_rows(variant)
    if set(baseline_rows) != set(variant_rows):
        raise ValueError("score part-family sets differ")
    if _case_set(baseline_rows) != _case_set(variant_rows):
        raise ValueError("score case-token sets differ")
    baseline_scan_count = int((baseline.get("metrics") or {}).get("scan_count") or 0)
    variant_scan_count = int((variant.get("metrics") or {}).get("scan_count") or 0)
    if baseline_scan_count != variant_scan_count or baseline_scan_count != len(
        _case_set(baseline_rows)
    ):
        raise ValueError("score scan counts differ from the family case set")

    families = sorted(baseline_rows)
    delta_matrix = np.asarray(
        [
            [
                float(variant_rows[family][metric])
                - float(baseline_rows[family][metric])
                for metric in METRICS
            ]
            for family in families
        ],
        dtype=np.float64,
    )
    seed = int.from_bytes(
        hashlib.sha256(
            f"{baseline_freeze}:PAIRED_FAMILY_BOOTSTRAP_V1".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(families), size=(int(resamples), len(families)), dtype=np.int32
    )
    bootstrap = np.mean(delta_matrix[indices], axis=1)
    summaries = {}
    for metric_index, metric in enumerate(METRICS):
        values = bootstrap[:, metric_index]
        summaries[metric] = {
            "observed_family_macro_delta": round(
                float(np.mean(delta_matrix[:, metric_index])), 6
            ),
            "paired_family_bootstrap_95_ci": [
                round(float(np.quantile(values, 0.025)), 6),
                round(float(np.quantile(values, 0.975)), 6),
            ],
            "bootstrap_probability_delta_gt_zero": round(
                float(np.mean(values > 0.0)), 6
            ),
            "families_improved": int(np.count_nonzero(delta_matrix[:, metric_index] > 0.0)),
            "families_tied": int(np.count_nonzero(delta_matrix[:, metric_index] == 0.0)),
            "families_worse": int(np.count_nonzero(delta_matrix[:, metric_index] < 0.0)),
        }
    family_deltas = [
        {
            "part_family": family,
            **{
                f"{metric}_delta": round(
                    float(variant_rows[family][metric])
                    - float(baseline_rows[family][metric]),
                    6,
                )
                for metric in METRICS
            },
            "case_tokens": sorted(str(v) for v in baseline_rows[family]["case_tokens"]),
        }
        for family in families
    ]
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_paired_family_variant_comparison",
        "evaluation_only": True,
        "inference_metadata_exposure": False,
        "corpus_freeze_id": baseline_freeze,
        "case_count": baseline_scan_count,
        "part_family_count": len(families),
        "bootstrap": {
            "profile": "PAIRED_FAMILY_BOOTSTRAP_V1",
            "resamples": int(resamples),
            "seed_sha256_prefix_u64": seed,
            "unit": "part_family",
        },
        "metric_deltas": summaries,
        "family_deltas": family_deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args()
    report = compare_scores(
        _read(args.baseline.resolve()),
        _read(args.variant.resolve()),
        resamples=args.resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
