#!/usr/bin/env python3
"""Apply post-v16 finite topology completion without identifiers or expected counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_patch_graph_endpoint_frame_completion import complete_patch_graph_endpoint_frame  # noqa: E402
from backend.f1_rolled_local_bend_completion import complete_rolled_local_bends  # noqa: E402
from backend.f1_rolled_long_strip_pruning import prune_rolled_long_strips  # noqa: E402

BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=BASE / "routed_v16_post_holdout_residual_pruning")
    parser.add_argument("--candidate-pool-root", type=Path, default=BASE / "scorecard_candidate_pool_v1")
    parser.add_argument("--zero-summary", type=Path, default=BASE / "inference/inference_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "routed_v17_post_holdout_structural_completion")
    args = parser.parse_args()
    manifest = json.loads((args.candidate_pool_root / "manifest.json").read_text(encoding="utf-8"))
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    cases = []
    for index, case_dir in enumerate(sorted(args.prediction_root.glob("C[0-9][0-9]"))):
        token = case_dir.name
        prediction = json.loads((case_dir / "decomposition_result.json").read_text(encoding="utf-8"))
        pool = json.loads(Path(manifest["cases"][index]["path"]).read_text(encoding="utf-8"))
        zero = json.loads(Path(zero_summary["result_map"][token]["zero_shot_result"]).read_text(encoding="utf-8"))
        rolled_pruning = prune_rolled_long_strips(
            prediction.get("owned_bend_regions") or [], prediction.get("source_route") or {}, zero.get("bend_support_strips") or []
        )
        frame = complete_patch_graph_endpoint_frame(
            rolled_pruning["selected"], prediction.get("source_route") or {}, pool.get("line_objects") or []
        )
        rolled_completion = complete_rolled_local_bends(
            frame["selected"], prediction.get("source_route") or {}, zero.get("bend_support_strips") or [], pool.get("line_objects") or []
        )
        prediction["rolled_long_strip_pruning"] = {key: value for key, value in rolled_pruning.items() if key != "selected"}
        prediction["patch_graph_endpoint_frame_completion"] = {key: value for key, value in frame.items() if key != "selected"}
        prediction["rolled_local_bend_completion"] = {key: value for key, value in rolled_completion.items() if key != "selected"}
        prediction["owned_bend_regions"] = rolled_completion["selected"]
        destination = args.output_dir / token / "decomposition_result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        cases.append({
            "case_token": token,
            "prediction_path": str(destination),
            "rolled_long_pruned_count": len(rolled_pruning["pruned"]),
            "frame_promoted_count": len(frame["promoted"]),
            "frame_pruned_count": len(frame["pruned"]),
            "rolled_local_promoted_count": len(rolled_completion["promoted"]),
        })
    (args.output_dir / "completion_summary.json").write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "cases": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
