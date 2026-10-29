#!/usr/bin/env python3
"""Nonexecuting Gate-0 contract audit and lock for V30 AMFC_V1."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v30_amfc_v1_lock.json"
ENGINE_PATH = ROOT / "backend" / "f1_axial_multipath_fold_certificate.py"
WORKER_PATH = ROOT / "scripts" / "run_f1_v30_amfc_source_free.py"
BUILDER_PATH = ROOT / "scripts" / "build_f1_v30_amfc_source_free_fixtures.py"
RENDERER_PATH = ROOT / "scripts" / "render_f1_v30_amfc_source_free.py"
GATES_PATH = ROOT / "scripts" / "run_f1_v30_amfc_gates.py"
AUDIT_PATH = Path(__file__).resolve()
V16_PATH = ROOT / "scripts" / "select_f1_mesh_route_v16.py"
V15_NMS_PATH = ROOT / "scripts" / "select_f1_mesh_route_v15.py"
OUTPUT_PATH = ROOT / "output" / "research" / "f1-v30-amfc-v1" / "precommit_manifest.json"
COUNTER_PATH = ROOT / "output" / "research" / "f1-v30-amfc-v1" / "invocation_events.jsonl"

SOURCE_PATHS = (
    ENGINE_PATH,
    WORKER_PATH,
    BUILDER_PATH,
    RENDERER_PATH,
    GATES_PATH,
    LOCK_PATH,
    AUDIT_PATH,
    V16_PATH,
    V15_NMS_PATH,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment(tree: ast.AST, name: str) -> ast.AST | None:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    return None


def _number(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left, right = _number(node.left), _number(node.right)
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
        return _number(node)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def _called_names(function: ast.FunctionDef) -> list[str]:
    values = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                values.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                values.append(node.func.attr)
    return values


def _invocation_count() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "AMFC_INVOKE")


def build_manifest() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    engine_source = ENGINE_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    builder_source = BUILDER_PATH.read_text(encoding="utf-8")
    renderer_source = RENDERER_PATH.read_text(encoding="utf-8")
    gates_source = GATES_PATH.read_text(encoding="utf-8")
    tree = ast.parse(engine_source, filename=str(ENGINE_PATH))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    constant_names = {
        "normal_neighbors": "NORMAL_NEIGHBORS",
        "normal_radius_s": "NORMAL_RADIUS_S",
        "normal_minimum_neighbors": "NORMAL_MINIMUM_NEIGHBORS",
        "normal_jacobi_sweeps": "NORMAL_JACOBI_SWEEPS",
        "surface_variation_maximum": "SURFACE_VARIATION_MAXIMUM",
        "station_count": "STATION_COUNT",
        "station_half_width_s": "STATION_HALF_WIDTH_S",
        "station_radius_s": "STATION_RADIUS_S",
        "station_end_margin_s": "STATION_END_MARGIN_S",
        "maximum_station_nodes": "MAX_STATION_NODES",
        "station_voxel_s": "STATION_VOXEL_S",
        "reciprocal_neighbors": "RECIPROCAL_NEIGHBORS",
        "maximum_station_edges": "MAX_STATION_EDGES",
        "tail_min_s": "TAIL_MIN_S",
        "tail_max_s": "TAIL_MAX_S",
        "tail_plane_distance_s": "TAIL_PLANE_DISTANCE_S",
        "tail_normal_degrees": "TAIL_NORMAL_DEGREES",
        "tail_minimum_nodes": "TAIL_MINIMUM_NODES",
        "tail_minimum_axial_span_s": "TAIL_MINIMUM_AXIAL_SPAN_S",
        "tail_minimum_outward_span_s": "TAIL_MINIMUM_OUTWARD_SPAN_S",
        "edge_minimum_s": "EDGE_MINIMUM_S",
        "edge_maximum_s": "EDGE_MAXIMUM_S",
        "edge_maximum_axial_s": "EDGE_MAXIMUM_AXIAL_S",
        "edge_tangent_ratio_maximum": "EDGE_TANGENT_RATIO_MAXIMUM",
        "edge_normal_degrees_maximum": "EDGE_NORMAL_DEGREES_MAXIMUM",
        "maximum_paths": "MAX_PATHS",
        "path_minimum_edges": "PATH_MINIMUM_EDGES",
        "path_maximum_edges": "PATH_MAXIMUM_EDGES",
        "path_maximum_length_s": "PATH_MAXIMUM_LENGTH_S",
        "path_maximum_chord_ratio": "PATH_MAXIMUM_CHORD_RATIO",
        "path_endpoint_normal_degrees": "PATH_ENDPOINT_NORMAL_DEGREES",
        "path_minimum_internal_nodes": "PATH_MINIMUM_INTERNAL_NODES",
        "path_maximum_axial_excursion_s": "PATH_MAXIMUM_AXIAL_EXCURSION_S",
        "path_turn_error_degrees": "PATH_TURN_ERROR_DEGREES",
        "path_monotonicity_minimum": "PATH_MONOTONICITY_MINIMUM",
        "minimum_target_turn_degrees": "MINIMUM_TARGET_TURN_DEGREES",
        "maximum_target_turn_degrees": "MAXIMUM_TARGET_TURN_DEGREES",
        "minimum_valid_paths_per_station": "MINIMUM_VALID_PATHS_PER_STATION",
        "minimum_path_midpoint_span_s": "MINIMUM_PATH_MIDPOINT_SPAN_S",
        "minimum_tail_valid_stations": "MINIMUM_TAIL_VALID_STATIONS",
        "minimum_valid_stations": "MINIMUM_VALID_STATIONS",
        "minimum_consecutive_valid_stations": "MINIMUM_CONSECUTIVE_VALID_STATIONS",
        "minimum_valid_paths_total": "MINIMUM_VALID_PATHS_TOTAL",
        "median_turn_error_degrees": "MEDIAN_TURN_ERROR_DEGREES",
        "maximum_eligible_rejected_candidates": "MAX_ELIGIBLE_REJECTED_CANDIDATES",
        "maximum_source_points": "MAX_SOURCE_POINTS",
        "maximum_source_triangles": "MAX_SOURCE_TRIANGLES",
        "maximum_workers": "MAX_WORKERS",
        "maximum_candidate_record_bytes": "MAX_CANDIDATE_RECORD_BYTES",
        "maximum_summary_bytes": "MAX_SUMMARY_BYTES",
        "coordinate_quantization_divisor": "COORDINATE_QUANTIZATION_DIVISOR",
        "turn_quantization_microdegrees": "TURN_QUANTIZATION_MICRODEGREES",
        "nms_angle_degrees": "NMS_ANGLE_DEGREES",
        "nms_distance_mm": "NMS_DISTANCE_MM",
        "nms_minimum_overlap": "NMS_MINIMUM_OVERLAP",
    }
    observed_constants = {name: _literal(tree, source_name) for name, source_name in constant_names.items()}
    check("literal_constants_match_lock", observed_constants == lock["constants"], observed_constants)
    check("source_schema_exact", list(_literal(tree, "SOURCE_SCHEMA_FIELDS")) == lock["source_schema_fields"], list(_literal(tree, "SOURCE_SCHEMA_FIELDS")))
    check("candidate_schema_exact", list(_literal(tree, "CANDIDATE_SCHEMA_FIELDS")) == lock["candidate_schema_fields"], list(_literal(tree, "CANDIDATE_SCHEMA_FIELDS")))

    select_calls = _called_names(_function(tree, "select_unified_v30"))
    required = ["parse_source", "parse_candidates", "compute_point_normals", "deterministic_total_sort", "ordinary_nms", "build_result"]
    positions = [select_calls.index(name) if name in select_calls else -1 for name in required]
    check("one_amfc_stage_then_one_frozen_nms", positions == sorted(positions) and -1 not in positions and select_calls.count("ordinary_nms") == 1, {"calls": select_calls, "positions": positions})
    check("legacy_precedes_rescues", "legacy + rescues" in engine_source and "ACTION_LEGACY" in engine_source, "deterministic_total_sort returns legacy + rescues")
    check("legacy_raw_records_preserved", 'legacy_inputs = [candidate.raw for candidate in candidates if candidate.selected_legacy]' in engine_source, "legacy raw dictionaries copied without mutation")
    check("rescue_only_no_candidate_generator", all(token not in select_calls for token in ("generate_candidate", "create_candidate", "fit_line", "refit_line")), select_calls)
    check("fixed_jacobi_no_blas_eigensolver", "NORMAL_JACOBI_SWEEPS" in engine_source and "np.linalg.eigh" not in engine_source and "np.linalg.eig" not in engine_source, "12-sweep explicit Jacobi")
    check("no_rng_learning_or_adaptation", all(token not in engine_source.lower() for token in ("random.", "np.random", "torch", "tensorflow", "sklearn", "fit_predict", "adaptive_threshold")), "no stochastic or learned mechanism")
    check("all_structures_bounded", all(token in engine_source for token in ("MAX_SOURCE_POINTS", "MAX_SOURCE_TRIANGLES", "MAX_STATION_NODES", "MAX_STATION_EDGES", "MAX_ELIGIBLE_REJECTED_CANDIDATES", "MAX_CANDIDATE_RECORD_BYTES", "MAX_SUMMARY_BYTES", "MAX_WORKERS")), "literal hard caps")
    check("resource_caps_raise_not_truncate", "raise AMFCContractError(REASON_RESOURCE_CAP" in engine_source and "[:MAX_STATION" not in engine_source and "[:MAX_ELIGIBLE" not in engine_source, "caps fail closed")
    canonical_source = ast.get_source_segment(engine_source, _function(tree, "canonical_result_bytes")) or ""
    build_source = ast.get_source_segment(engine_source, _function(tree, "build_result")) or ""
    check("cache_cannot_serialize_source_geometry_or_paths", 'key != "_ephemeral"' in canonical_source and '"coordinates"' not in build_source and '"graph_edges"' not in build_source and '"accepted_paths"' not in build_source, "ephemeral visual geometry stripped")
    check("candidate_id_not_pass_predicate", 'def _candidate_id' in engine_source and 'candidate.candidate_id' not in (ast.get_source_segment(engine_source, _function(tree, "_analyze_station")) or ""), "ID is output/tie metadata only")
    check("allowlisted_schema_rejects_extras", "set(raw) != set(SOURCE_SCHEMA_FIELDS)" in engine_source and "set(raw) != set(CANDIDATE_SCHEMA_FIELDS)" in engine_source, "exact field sets")
    check("worker_has_no_arbitrary_input_path", "--input" not in worker_source and "choices=_fixture_ids()" in worker_source and "choices=VARIANTS" in worker_source and "choices=RUN_TAGS" in worker_source, "fixed source-free roots and choices")
    check("builder_has_no_detector_import", "f1_axial_multipath_fold_certificate" not in builder_source and "select_unified_v30" not in builder_source, "separate harness tree")
    check("builder_uses_direct_3d_surfaces_only", all(token not in builder_source.lower() for token in ("zbuffer", "depth_image", "visibility_ray", "camera_intrinsic")), "analytic coordinate construction")
    check("renderer_native_4096_no_decimation", "W = 4096" in renderer_source and "H = 4096" in renderer_source and all(token not in renderer_source.lower() for token in ("decimate", "downsample", "lod")), "all geometry rendered at 4096x4096")
    check("gate_order_and_one_shot_counter", 'choices=("gate1_verify", "gate2_source_free", "gate3_metamorphic", "gate4_replay", "gate5_resource", "gate6_visual")' in gates_source and "_invocation_count() != 0" in gates_source and '"AMFC_INVOKE"' in worker_source, "append-only invocation counter and strict stages")
    check("research_runtime_isolation", all(not isinstance(node, (ast.Import, ast.ImportFrom)) or not any(alias.name.startswith(("apps", "domains", "infrastructure", "backend.bend_inspection_pipeline", "backend.qc_engine")) for alias in node.names) for node in ast.walk(tree)), "no production/runtime imports")
    check("compiler_fast_math_disabled", sys.flags.optimize == 0 and not hasattr(sys.flags, "fast_math"), {"implementation": platform.python_implementation(), "version": platform.python_version(), "optimize": sys.flags.optimize})
    check("zero_prior_invocations", _invocation_count() == 0, _invocation_count())

    source_hashes = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in SOURCE_PATHS]
    constants_hash = hashlib.sha256(json.dumps(lock["constants"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    schema_hash = hashlib.sha256(json.dumps({"source": lock["source_schema_fields"], "candidate": lock["candidate_schema_fields"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    schedule_hash = hashlib.sha256(json.dumps(lock["execution_schedule"], separators=(",", ":")).encode()).hexdigest()
    interpreter_hash = hashlib.sha256(
        json.dumps(
            {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "flags": {
                    name: getattr(sys.flags, name)
                    for name in dir(sys.flags)
                    if not name.startswith("_")
                    and isinstance(getattr(sys.flags, name), (int, bool))
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    passed = all(row["passed"] for row in checks)
    return {
        "kind": "v30_amfc_static_precommit_manifest",
        "profile": lock["profile"],
        "primitive": lock["primitive"],
        "gate": 0,
        "passed": passed,
        "status": "LOCKED" if passed else "RETIRED",
        "token": lock["tokens"]["mechanism_locked"] if passed else lock["tokens"]["static_retire"],
        "detector_invoked": False,
        "fixture_builder_invoked": False,
        "real_scans_opened": False,
        "historical_data_opened": False,
        "labels_opened": False,
        "cad_opened": False,
        "checks": checks,
        "mechanism_sha256": _sha(ENGINE_PATH),
        "constants_sha256": constants_hash,
        "schema_sha256": schema_hash,
        "interpreter_and_flags_sha256": interpreter_hash,
        "execution_schedule_sha256": schedule_hash,
        "v16_dependency_sha256": _sha(V16_PATH),
        "v15_nms_dependency_sha256": _sha(V15_NMS_PATH),
        "source_hashes": source_hashes,
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"passed": manifest["passed"], "status": manifest["status"], "token": manifest["token"], "check_count": len(manifest["checks"])}, sort_keys=True))
    if not manifest["passed"]:
        for row in manifest["checks"]:
            if not row["passed"]:
                print(json.dumps(row, sort_keys=True), file=sys.stderr)
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
