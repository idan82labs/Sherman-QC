#!/usr/bin/env python3
"""Build intrinsic bend-level F1 evidence scorecards without opening truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_bend_evidence_score import score_bend_evidence  # noqa: E402

BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=BASE / "routed_v14_post_holdout_complete_topology")
    parser.add_argument("--candidate-pool-root", type=Path, default=BASE / "scorecard_candidate_pool_v1")
    parser.add_argument("--output", type=Path, default=BASE / "scorecard_v1/bend_scorecard.json")
    args = parser.parse_args()
    manifest = json.loads((args.candidate_pool_root / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for index, pool_case in enumerate(manifest["cases"], start=1):
        token = f"C{index:02d}"
        prediction_path = args.prediction_root / token / "decomposition_result.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        pool = json.loads(Path(pool_case["path"]).read_text(encoding="utf-8"))
        bends = []
        for bend in prediction.get("owned_bend_regions") or []:
            scored = dict(bend)
            scored["evidence"] = score_bend_evidence(bend, pool.get("line_objects") or [])
            bends.append(scored)
        cases.append(
            {
                "case_token": token,
                "prediction_path": str(prediction_path),
                "candidate_pool_path": str(pool_case["path"]),
                "bend_count": len(bends),
                "bends": bends,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_intrinsic_f1_bend_evidence_scorecard",
        "score_semantics": "0_to_100_intrinsic_geometry_evidence_not_probability",
        "rating_bands": {"A": ">=80", "B": "65-79.999", "C": "50-64.999", "D": "<50"},
        "inference_contract": {"uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    distribution = {rating: 0 for rating in "ABCD"}
    for case in cases:
        for bend in case["bends"]:
            distribution[bend["evidence"]["evidence_rating"]] += 1
    print(json.dumps({"case_count": len(cases), "bend_count": sum(case["bend_count"] for case in cases), "rating_distribution": distribution, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
