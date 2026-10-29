#!/usr/bin/env python3
"""Apply post-v17 prismatic pruning and topology completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_partial_lattice_completion import complete_partial_lattice  # noqa: E402
from backend.f1_prismatic_junction_completion import complete_prismatic_junction  # noqa: E402
from backend.f1_prismatic_transition_pruning import prune_weak_prismatic_transitions  # noqa: E402

BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=BASE / "routed_v17_post_holdout_structural_completion")
    parser.add_argument("--candidate-pool-root", type=Path, default=BASE / "scorecard_candidate_pool_v1")
    parser.add_argument("--zero-summary", type=Path, default=BASE / "inference/inference_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "routed_v18_post_holdout_prismatic_completion")
    args = parser.parse_args()
    manifest = json.loads((args.candidate_pool_root / "manifest.json").read_text(encoding="utf-8"))
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    cases = []
    for index, case_dir in enumerate(sorted(args.prediction_root.glob("C[0-9][0-9]"))):
        token = case_dir.name
        prediction = json.loads((case_dir / "decomposition_result.json").read_text(encoding="utf-8"))
        pool = json.loads(Path(manifest["cases"][index]["path"]).read_text(encoding="utf-8"))
        zero = json.loads(Path(zero_summary["result_map"][token]["zero_shot_result"]).read_text(encoding="utf-8"))
        pruning = prune_weak_prismatic_transitions(
            prediction.get("owned_bend_regions") or [], prediction.get("source_route") or {}, zero.get("bend_support_strips") or [], pool.get("line_objects") or []
        )
        junction = complete_prismatic_junction(
            pruning["selected"], prediction.get("source_route") or {}, zero.get("bend_support_strips") or [], pool.get("line_objects") or []
        )
        partial = complete_partial_lattice(
            junction["selected"], prediction.get("source_route") or {}, zero.get("bend_support_strips") or [], pool.get("line_objects") or []
        )
        prediction["prismatic_transition_pruning"] = {key: value for key, value in pruning.items() if key != "selected"}
        prediction["prismatic_junction_completion"] = {key: value for key, value in junction.items() if key != "selected"}
        prediction["partial_lattice_completion"] = {key: value for key, value in partial.items() if key != "selected"}
        prediction["owned_bend_regions"] = partial["selected"]
        destination = args.output_dir / token / "decomposition_result.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        cases.append({
            "case_token": token,
            "prediction_path": str(destination),
            "weak_prismatic_pruned_count": len(pruning["pruned"]),
            "junction_promoted_count": len(junction["promoted"]),
            "partial_promoted_count": len(partial["promoted"]),
            "partial_pruned_count": len(partial["pruned"]),
            "partial_refit_count": len(partial["refit"]),
        })
    (args.output_dir / "completion_summary.json").write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "cases": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
