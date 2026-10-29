#!/usr/bin/env python3
"""Apply geometry-only residual pruning to accepted routed F1 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_boundary_truncated_residual_pruning import prune_boundary_truncated_residuals  # noqa: E402
from backend.f1_rolled_parallel_doublet_pruning import prune_rolled_parallel_doublet  # noqa: E402

BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=BASE / "routed_v15_post_holdout_dense_lattice")
    parser.add_argument("--zero-summary", type=Path, default=BASE / "inference/inference_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "routed_v16_post_holdout_residual_pruning")
    args = parser.parse_args()
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    cases = []
    for case_dir in sorted(args.prediction_root.glob("C[0-9][0-9]")):
        token = case_dir.name
        prediction = json.loads((case_dir / "decomposition_result.json").read_text(encoding="utf-8"))
        zero = json.loads(Path(zero_summary["result_map"][token]["zero_shot_result"]).read_text(encoding="utf-8"))
        boundary = prune_boundary_truncated_residuals(
            prediction.get("owned_bend_regions") or [], prediction.get("source_route") or {}
        )
        rolled = prune_rolled_parallel_doublet(
            boundary["selected"], prediction.get("source_route") or {}, zero.get("bend_support_strips") or []
        )
        prediction["boundary_truncated_residual_pruning"] = {key: value for key, value in boundary.items() if key != "selected"}
        prediction["rolled_parallel_doublet_pruning"] = {key: value for key, value in rolled.items() if key != "selected"}
        prediction["owned_bend_regions"] = rolled["selected"]
        destination = args.output_dir / token / "decomposition_result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        cases.append({
            "case_token": token,
            "prediction_path": str(destination),
            "boundary_pruned_count": len(boundary["pruned"]),
            "rolled_doublet_pruned_count": len(rolled["pruned"]),
        })
    (args.output_dir / "pruning_summary.json").write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "cases": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
