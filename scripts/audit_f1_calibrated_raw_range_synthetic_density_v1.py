#!/usr/bin/env python3
"""Compare fused synthetic raw-range sources with physical LiDAR spacing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_f1_synthetic_lidar_density import REFERENCE_PATHS, _row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    corpus = args.corpus_root.resolve()
    manifest = json.loads((corpus / "public_manifest.json").read_text(encoding="utf-8"))
    references = [_row(repo / relative) for relative in REFERENCE_PATHS]
    synthetic = [
        _row(Path(row["bundle_path"]) / "source.ply", row["opaque_case_id"])
        for row in manifest["cases"]
    ]
    reference_medians = [row["nearest_neighbor_mm"]["0.5"] for row in references]
    synthetic_medians = [row["nearest_neighbor_mm"]["0.5"] for row in synthetic]
    lower, upper = min(reference_medians), max(reference_medians)
    report = {
        "schema_version": 1,
        "kind": "calibrated_raw_range_synthetic_density_audit_v1",
        "confirmatory_eligible": False,
        "reference": references,
        "synthetic": synthetic,
        "summary": {
            "reference_scan_count": len(references),
            "synthetic_scan_count": len(synthetic),
            "reference_median_nn_range_mm": [lower, upper],
            "reference_median_of_medians_mm": float(np.median(reference_medians)),
            "synthetic_median_nn_range_mm": [min(synthetic_medians), max(synthetic_medians)],
            "synthetic_median_of_medians_mm": float(np.median(synthetic_medians)),
            "all_synthetic_medians_inside_observed_reference_range": all(
                lower <= value <= upper for value in synthetic_medians
            ),
            "all_synthetic_points_at_or_below_195k": all(row["point_count"] <= 195_000 for row in synthetic),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if all((
        report["summary"]["all_synthetic_medians_inside_observed_reference_range"],
        report["summary"]["all_synthetic_points_at_or_below_195k"],
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
