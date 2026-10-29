#!/usr/bin/env python3
"""Apply bounded dense-evidence F1 completions to routed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_parallel_width_lattice_completion import complete_parallel_width_lattice  # noqa: E402

BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=BASE / "routed_v14_post_holdout_complete_topology")
    parser.add_argument("--candidate-pool-root", type=Path, default=BASE / "scorecard_candidate_pool_v1")
    parser.add_argument("--zero-summary", type=Path, default=BASE / "inference/inference_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "routed_v15_post_holdout_dense_lattice")
    args = parser.parse_args()
    manifest = json.loads((args.candidate_pool_root / "manifest.json").read_text(encoding="utf-8"))
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    cases = []
    for index, pool_case in enumerate(manifest["cases"], start=1):
        token = f"C{index:02d}"
        prediction = json.loads((args.prediction_root / token / "decomposition_result.json").read_text(encoding="utf-8"))
        pool = json.loads(Path(pool_case["path"]).read_text(encoding="utf-8"))
        zero = json.loads(Path(zero_summary["result_map"][token]["zero_shot_result"]).read_text(encoding="utf-8"))
        completion = complete_parallel_width_lattice(
            prediction.get("owned_bend_regions") or [], zero.get("bend_support_strips") or [], pool.get("line_objects") or []
        )
        prediction["parallel_width_lattice_completion"] = {key: value for key, value in completion.items() if key not in {"selected", "promoted"}}
        prediction["promoted_parallel_width_companions"] = completion["promoted"]
        prediction["owned_bend_regions"] = completion["selected"]
        case_dir = args.output_dir / token
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / "decomposition_result.json"
        path.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        cases.append({"case_token": token, "prediction_path": str(path), "promoted_count": len(completion["promoted"])})
    (args.output_dir / "completion_summary.json").write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "promotions": {item["case_token"]: item["promoted_count"] for item in cases}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
