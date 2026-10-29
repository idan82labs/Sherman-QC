#!/usr/bin/env python3
"""Aggregate the truth-free BIRTH_V1 stage census over solved graph pools."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_structured_birth_diagnostics import (  # noqa: E402
    DIAGNOSTIC_PROFILE,
    FUNNEL_STAGES,
    diagnose_residual_birth_eligibility,
)


MAX_CASES = 256


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_batch_diagnostic(case_root: Path) -> dict[str, Any]:
    case_dirs = sorted(
        path
        for path in case_root.iterdir()
        if path.is_dir()
        and (path / "typed_union_graph_00.json").is_file()
        and (path / "cpsat_solution_00.json").is_file()
    )
    if not case_dirs:
        raise ValueError(f"no solved structured-union cases found under {case_root}")
    if len(case_dirs) > MAX_CASES:
        raise ValueError(f"diagnostic case count exceeds cap: {len(case_dirs)} > {MAX_CASES}")
    aggregate_funnel: Counter[str] = Counter()
    aggregate_rejections: Counter[str] = Counter()
    cases = []
    for case_dir in case_dirs:
        diagnostic = diagnose_residual_birth_eligibility(
            _read(case_dir / "typed_union_graph_00.json"),
            _read(case_dir / "cpsat_solution_00.json"),
        )
        aggregate_funnel.update(diagnostic["funnel"])
        aggregate_rejections.update(diagnostic["rejections"])
        cases.append(
            {
                "case_token": case_dir.name,
                "input_counts": diagnostic["input_counts"],
                "thresholds": diagnostic["thresholds"],
                "funnel": diagnostic["funnel"],
                "rejections": diagnostic["rejections"],
                "first_empty_transition": diagnostic["first_empty_transition"],
                "proposal_ids": diagnostic["proposal_ids"],
            }
        )
    rejection_ranking = [
        {"reason": reason, "count": int(count)}
        for reason, count in sorted(
            aggregate_rejections.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_birth_v1_batch_stage_census",
        "diagnostic_profile": DIAGNOSTIC_PROFILE,
        "research_only": True,
        "runtime_integration": False,
        "strict_scan_only": True,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "case_count": len(cases),
        "aggregate_funnel": {
            stage: int(aggregate_funnel[stage]) for stage in FUNNEL_STAGES
        },
        "aggregate_rejections": dict(sorted(aggregate_rejections.items())),
        "rejection_ranking": rejection_ranking,
        "cases_with_emitted_columns": sum(
            bool(case["funnel"]["emitted_columns_after_cap"]) for case in cases
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_batch_diagnostic(args.case_root.resolve())
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
