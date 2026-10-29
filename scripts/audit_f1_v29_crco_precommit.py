#!/usr/bin/env python3
"""Static, nonexecuting CRCO_V1 Gate-0 contract audit.

The detector is parsed as text/AST only.  This audit never imports it, never
builds fixtures, and never opens any real or historical scan artifact.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v29_crco_v1_lock.json"
ENGINE_PATH = ROOT / "backend" / "f1_calibrated_range_continuity_ownership.py"
WORKER_PATH = ROOT / "scripts" / "run_f1_v29_crco_source_free.py"
GENERATOR_PATH = ROOT / "scripts" / "build_f1_v29_crco_source_free_fixtures.py"
RENDERER_PATH = ROOT / "scripts" / "render_f1_v29_crco_source_free.py"
GATES_PATH = ROOT / "scripts" / "run_f1_v29_crco_gates.py"
AUDIT_PATH = Path(__file__).resolve()
OUTPUT_PATH = ROOT / "output" / "research" / "f1-v29-crco-v1" / "precommit_manifest.json"

SOURCE_PATHS = (ENGINE_PATH, WORKER_PATH, GENERATOR_PATH, RENDERER_PATH, GATES_PATH, LOCK_PATH, AUDIT_PATH)
FAILURE_TOKEN = "V29_CRCO_PRECOMMIT_CONTRACT_FAILURE_RETIRE"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment(tree: ast.AST, name: str) -> ast.AST | None:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    return None


def _safe_number(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left, right = _safe_number(node.left), _safe_number(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        return left - right
    raise ValueError(ast.dump(node))


def _literal(tree: ast.AST, name: str) -> Any:
    node = _assignment(tree, name)
    if node is None:
        raise KeyError(name)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _safe_number(node)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def _called_names(function: ast.FunctionDef) -> list[str]:
    calls = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    # AST walk preserves source order for the flat select function.
    return calls


def build_manifest() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    engine_source = ENGINE_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    generator_source = GENERATOR_PATH.read_text(encoding="utf-8")
    renderer_source = RENDERER_PATH.read_text(encoding="utf-8")
    gates_source = GATES_PATH.read_text(encoding="utf-8")
    tree = ast.parse(engine_source, filename=str(ENGINE_PATH))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    constant_names = {
        "station_count": "STATION_COUNT",
        "flank_multipliers": "FLANK_MULTIPLIERS",
        "minimum_matches_per_flank": "MIN_MATCHES_PER_FLANK",
        "decision_station_count": "DECISION_STATION_COUNT",
        "decision_run_length": "DECISION_RUN_LENGTH",
        "minimum_view_separation_deg": "MIN_VIEW_SEPARATION_DEG",
        "match_distance_s": "MATCH_DISTANCE_S",
        "depth_separation_s": "DEPTH_SEPARATION_S",
        "path_gap_s": "PATH_GAP_S",
        "boundary_distance_s": "BOUNDARY_DISTANCE_S",
        "boundary_station_count": "BOUNDARY_STATION_COUNT",
        "coordinate_quantization_divisor": "COORDINATE_QUANTIZATION_DIVISOR",
        "keyframe_count": "KEYFRAME_COUNT",
        "maximum_candidate_records": "MAX_CANDIDATE_RECORDS",
        "maximum_queried_pixels": "MAX_QUERIED_PIXELS",
        "maximum_incremental_bytes": "MAX_INCREMENTAL_BYTES",
        "maximum_cache_bytes": "MAX_CACHE_BYTES",
        "nms_angle_deg": "NMS_ANGLE_DEG",
        "nms_distance_mm": "NMS_DISTANCE_MM",
        "nms_overlap": "NMS_OVERLAP",
    }
    observed_constants = {lock_name: _literal(tree, source_name) for lock_name, source_name in constant_names.items()}
    expected_constants = {name: lock["constants"][name] for name in constant_names}
    observed_constants["flank_multipliers"] = list(observed_constants["flank_multipliers"])
    check("immutable_constants_match_lock", observed_constants == expected_constants, observed_constants)
    check("candidate_schema_exact", list(_literal(tree, "CANDIDATE_SCHEMA_FIELDS")) == lock["candidate_schema_fields"], list(_literal(tree, "CANDIDATE_SCHEMA_FIELDS")))
    check("range_schema_exact", list(_literal(tree, "RANGE_FRAME_SCHEMA_FIELDS")) == lock["range_frame_schema_fields"], list(_literal(tree, "RANGE_FRAME_SCHEMA_FIELDS")))

    select_calls = _called_names(_function(tree, "select_unified_v29"))
    required_order = ["parse_candidates", "parse_range_frames", "select_keyframes", "adjudicate_crco_v1", "deterministic_total_sort", "ordinary_nms", "build_result"]
    positions = [select_calls.index(name) if name in select_calls else -1 for name in required_order]
    check("one_pre_nms_adjudication_then_one_final_nms", positions == sorted(positions) and -1 not in positions and select_calls.count("ordinary_nms") == 1, {"calls": select_calls, "positions": positions})
    check("no_candidate_creation_after_nms", "accepted = ordinary_nms(ordered)\n    return build_result(accepted, decisions, keyframes, ledger, scale)" in engine_source, "ordinary_nms immediately feeds build_result")
    check("legacy_precedes_rescue_for_collision", "return legacy + rescues" in engine_source and "for decision in ordered:" in engine_source, "legacy + rescues feeds ordered NMS")
    check("identity_keys_do_not_use_indices_or_filenames", all(token not in engine_source[engine_source.index("def _canonicalize_line_and_flanks"):engine_source.index("class PixelLedger")] for token in ("filename", "part_id", "candidate_index", "array_index")), "candidate hash region content-only")
    check("endpoint_and_flank_canonicalization", all(token in engine_source for token in ("if end_q < start_q:", "canonical_flanks.sort", "_canonicalize_line_and_flanks")), "canonical endpoint order and unordered flank sort")
    sample_source = ast.get_source_segment(engine_source, _function(tree, "_sample_state")) or ""
    check("invalid_depth_is_unknown_before_classification", sample_source.index("if not bool(frame.valid[v, u])") < sample_source.index("observed = frame.backproject"), "validity checked before observed depth")
    key_source = ast.get_source_segment(engine_source, _function(tree, "select_keyframes")) or ""
    check("keyframe_selection_list_order_independent", "ordered = sorted(frames, key=lambda frame: frame.frame_key)" in key_source and "frame_key" in key_source, "content-key sort before selection")
    check("caps_fail_closed_without_truncation", all(token in engine_source for token in ("CANDIDATE_CAP_EXCEEDED", "QUERY_PIXEL_CAP_EXCEEDED", "INCREMENTAL_MEMORY_CAP_EXCEEDED", "CACHE_CAP_EXCEEDED")) and "[:MAX_" not in engine_source, "all structural caps raise")
    check("worker_has_fixed_source_free_root_only", "--input" not in worker_source and "SOURCE_FREE_ROOT" in worker_source and "choices=_fixture_ids()" in worker_source and "source_free" in worker_source, "no arbitrary input path")
    check("fixture_generator_reads_only_lock_and_literals", all(token not in generator_source for token in ("data/bend_corpus", "historical_35", "recent_26", "V27_outputs", "V28_outputs")), "no empirical input path")
    check("v27_v28_unreachable", all(token not in (engine_source + worker_source + generator_source + renderer_source + gates_source).lower() for token in ("f1_collinear_fragment_stitch", "f1_paired_return_completion", "select_f1_mesh_route_v27", "select_f1_mesh_route_v28")), "no import or reachability token")
    check("renderer_native_long_edge_4096", "W, H = 4096, 2304" in renderer_source and "max(image.size) != 4096" in renderer_source, "four deterministic 4096px artifacts")
    check("gate_order_strictly_bound", 'choices=("gate1", "gate2", "gate3", "gate4")' in gates_source and "_require_precommit()" in gates_source, "post-precommit source-free gates only")

    passed = all(row["passed"] for row in checks)
    return {
        "kind": "v29_crco_static_precommit_manifest",
        "profile": lock["profile"],
        "gate": 0,
        "passed": passed,
        "status": "ELIGIBLE_FOR_GATE_1" if passed else "RETIRED",
        "terminal_token": None if passed else FAILURE_TOKEN,
        "detector_invoked": False,
        "fixture_generator_invoked": False,
        "fixtures_executed": False,
        "real_scans_executed": False,
        "historical_data_opened": False,
        "labels_opened": False,
        "cad_or_drawings_opened": False,
        "checks": checks,
        "source_hashes": [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in SOURCE_PATHS],
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"passed": manifest["passed"], "status": manifest["status"], "terminal_token": manifest["terminal_token"], "check_count": len(manifest["checks"])}, sort_keys=True))
    if not manifest["passed"]:
        for row in manifest["checks"]:
            if not row["passed"]:
                print(json.dumps(row, sort_keys=True), file=sys.stderr)
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
