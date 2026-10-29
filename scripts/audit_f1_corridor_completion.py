#!/usr/bin/env python3
"""Build a renderable scan-only corridor-completion audit from F1 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from f1_corridor_completion import complete_scan_supported_corridors, corridor_completion_applicability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("decomposition", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.decomposition.read_text(encoding="utf-8"))
    hypotheses = list(payload.get("bend_hypotheses") or [])
    strips = [item for item in hypotheses if item.get("source_kind") == "zero_shot_strip_seed"]
    births = [item for item in hypotheses if item.get("source_kind") == "interface_birth"]
    atom_graph = dict(payload.get("atom_graph") or {})
    result = complete_scan_supported_corridors(
        strip_hypotheses=strips,
        interface_birth_hypotheses=births,
        local_spacing_mm=float(atom_graph.get("local_spacing_mm") or 1.0),
    )
    solver_options = dict((payload.get("repair_diagnostics") or {}).get("solver_options") or {})
    result["applicability"] = corridor_completion_applicability(
        route_payload=solver_options.get("scan_family_route"),
        strip_hypothesis_count=len(strips),
        interface_birth_hypothesis_count=len(births),
    )
    result.update(
        {
            "scan_path": payload.get("scan_path"),
            "part_id": payload.get("part_id"),
            "bend_count_range": [result["selected_count"], result["selected_count"]],
            "exact_bend_count": result["selected_count"],
            "owned_bend_regions": result["selected"],
            "provenance": {
                "source_decomposition": str(args.decomposition.resolve()),
                "truth_used_by_solver": False,
                "target_count_used": False,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_strip_count", "selected_uncovered_birth_count", "selected_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
