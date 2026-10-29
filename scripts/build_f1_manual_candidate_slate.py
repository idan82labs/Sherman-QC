#!/usr/bin/env python3
"""Build a neutral, source-only candidate slate for native-resolution review.

This utility does not score, rank, or select bends.  It exposes the sparse
mesh/strict-blind hypotheses already present in a candidate pool so a human
can precommit a visual interpretation before the learned selector is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import endpoints


REVIEW_SOURCES = {"bend_hypothesis", "scan_mesh_adjacency", "strict_blind_owned"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-all", action="store_true")
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    predictions = []
    for index, row in enumerate(rows):
        sources = set(str(value) for value in (row.get("cluster_sources") or ()))
        representative = str(row.get("representative_source") or "")
        if not args.include_all and not (sources & REVIEW_SOURCES or representative in REVIEW_SOURCES):
            continue
        line = endpoints(row)
        predictions.append(
            {
                "bend_id": f"M{index + 1:03d}",
                "candidate_id": str(row.get("candidate_id") or f"line-object-{index + 1}"),
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "representative_source": representative,
                "cluster_sources": sorted(sources),
            }
        )

    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_neutral_manual_candidate_slate",
        "profile": "SOURCE_ONLY_NATIVE_VISUAL_PRECOMMIT",
        "predictions": predictions,
        "candidate_count": len(predictions),
        "selector_run": False,
        "gold_opened": False,
        "expected_count_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": len(predictions), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
