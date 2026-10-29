#!/usr/bin/env python3
"""Development-only oracle diagnostic for V58 token coverage at true bends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_calibrated_multiview_turn_consensus_v58 as v58


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output exists")
    private = json.loads((args.corpus_root / "evaluator_only" / "private_manifest.json").read_text())
    family_by_case = {row["opaque_case_id"]: row["family_id"] for row in private["cases"]}
    cases = []
    totals = {"truth_bends": 0, "with_any_token": 0, "three_view": 0, "eight_tokens": 0, "span_gate": 0, "occupancy_gate": 0, "all_consensus_support_gates": 0}
    for index in range(1, 31):
        case_id = f"O{index:03d}"
        bundle = args.corpus_root / "public" / case_id
        source, frames, _, _ = v58.load_bundle(bundle)
        B = float(np.linalg.norm(np.ptp(source, axis=0)))
        stride = v58._stride(frames)
        tokens = []
        for view, frame in enumerate(frames):
            raw, _, _ = v58._raw_tokens_for_view(frame, view, stride, B)
            tokens.extend(v58._thin(raw, frame.depth.shape, B))
        truth = json.loads((args.corpus_root / "evaluator_only" / case_id / "truth.json").read_text())
        bend_rows = []
        for bend in truth["manual_bends"]:
            start, end = np.asarray(bend["line_start"], dtype=float), np.asarray(bend["line_end"], dtype=float)
            direction = _unit(end - start)
            support = [token for token in tokens if v58._angle_deg(token.tangent, direction) <= 6.0 and v58._line_distance(token.point, start, direction) <= 0.006 * B]
            if support:
                points = np.asarray([token.point for token in support])
                views = np.asarray([token.view for token in support], dtype=int)
                occupied, two_view, span, _ = v58._occupancy(points, start, direction, views)
            else:
                views = np.asarray([], dtype=int)
                occupied = two_view = 0
                span = 0.0
            distinct = len(set(int(value) for value in views))
            row = {
                "bend_id": bend["bend_id"],
                "token_count": len(support),
                "distinct_view_count": distinct,
                "span_over_B": span / B,
                "occupied_bin_count": occupied,
                "two_view_bin_count": two_view,
                "passes_three_view": distinct >= 3,
                "passes_eight_token": len(support) >= 8,
                "passes_span": span >= 0.10 * B,
                "passes_occupancy": occupied >= 7 and two_view >= 5,
            }
            row["passes_all_support_gates"] = all(row[key] for key in ("passes_three_view", "passes_eight_token", "passes_span", "passes_occupancy"))
            bend_rows.append(row)
            totals["truth_bends"] += 1
            totals["with_any_token"] += int(len(support) > 0)
            totals["three_view"] += int(row["passes_three_view"])
            totals["eight_tokens"] += int(row["passes_eight_token"])
            totals["span_gate"] += int(row["passes_span"])
            totals["occupancy_gate"] += int(row["passes_occupancy"])
            totals["all_consensus_support_gates"] += int(row["passes_all_support_gates"])
        cases.append({"case_id": case_id, "family_id": family_by_case[case_id], "thinned_token_count": len(tokens), "bends": bend_rows})
        print(json.dumps({"case_id": case_id, "tokens": len(tokens), "truth_bends": len(bend_rows), "all_support": sum(row["passes_all_support_gates"] for row in bend_rows)}, sort_keys=True), flush=True)
    report = {"schema_version": 1, "kind": "sherman_qc_f1_v58_truth_token_coverage_diagnostic", "development_only": True, "confirmatory_eligible": False, "aggregate": totals, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
