#!/usr/bin/env python3
"""Bind retained scan-only and CAD-assisted research stages to exact files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_candidate_ir_v1 import canonical_json_bytes, validate_stage_graph


DEFAULT_OUTPUT = ROOT / "docs/research/artifacts/2026-08-25_f1_retained_stage_graph_v1.json"


SCAN_STAGES = (
    ("scan_v16_route", "scripts/select_f1_mesh_route_v16.py", (), "PLY_BOUNDED_200K"),
    ("scan_v43_two_sided_support", "backend/f1_two_sided_parent_support_guard_v43.py", ("scan_v16_route",), "F1_CANDIDATE_SET_V1"),
    ("scan_v44_fragment_stitch", "backend/f1_v43_fragment_stitch_v44.py", ("scan_v43_two_sided_support",), "F1_CANDIDATE_SET_V1"),
    ("scan_v50_longest_ownership", "backend/f1_longest_incumbent_ownership_v50.py", ("scan_v44_fragment_stitch",), "F1_CANDIDATE_SET_V1"),
    ("scan_v47_elongated_lane", "backend/f1_route_independent_elongated_lane_v47.py", ("scan_v50_longest_ownership",), "F1_CANDIDATE_SET_V1"),
    ("scan_v48_nested_nms", "backend/f1_nested_partial_duplicate_nms_v48.py", ("scan_v47_elongated_lane",), "F1_CANDIDATE_SET_V1"),
    ("scan_v49_cross_source_nms", "backend/f1_short_span_cross_source_duplicate_nms_v49.py", ("scan_v48_nested_nms",), "F1_CANDIDATE_SET_V1"),
    ("scan_v51_parallel_return", "backend/f1_single_axis_parallel_return_recovery_v51.py", ("scan_v49_cross_source_nms",), "F1_CANDIDATE_SET_V1"),
)

CAD_STAGES = (
    ("cad_native_step_axes", "scripts/extract_step_topology_bend_axes.py", (), "STEP"),
    ("cad_trusted_registration", "scripts/run_blind_cad_ply_benchmark.py", (), "STEP_PLUS_PLY"),
    ("cad_legacy_scan_support", "scripts/apply_f1_cad_guided_scan_frame_gate.py", ("cad_native_step_axes", "cad_trusted_registration"), "F1_CANDIDATE_SET_V1"),
    ("scan_a0_baseline", "scripts/run_blind_cad_ply_benchmark.py", (), "PLY_BOUNDED_200K"),
    ("cad_retained_union", "scripts/diagnose_f1_cad_scan_fusion_gold11.py", ("cad_legacy_scan_support", "scan_a0_baseline"), "F1_CANDIDATE_SET_V1"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph(rows) -> dict[str, object]:
    stages = []
    bindings = {}
    for stage_id, relative, parents, input_schema in rows:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha(path)
        bindings[stage_id] = {"path": relative, "sha256": digest}
        stages.append({"stage_id": stage_id, "implementation_sha256": digest, "parent_stage_ids": list(parents), "input_schema": input_schema, "output_schema": "F1_CANDIDATE_SET_V1"})
    return {"graph": validate_stage_graph({"schema": "F1_STAGE_GRAPH_V1", "stages": stages}), "bindings": bindings}


def build() -> dict[str, object]:
    return {
        "schema": "F1_RETAINED_STAGE_GRAPHS_V1",
        "status": "RESEARCH_ONLY_NOT_PROMOTION",
        "canonical_candidate_schema": "F1_CANDIDATE_SET_V1",
        "scan_only": _graph(SCAN_STAGES),
        "cad_assisted": _graph(CAD_STAGES),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent); temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(result)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, args.output)
    finally:
        if temp.exists(): temp.unlink()
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": _sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
