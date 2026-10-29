#!/usr/bin/env python3
"""Static, non-invoking V32 AVHFM contract and freeze audit."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v32_avhfm_v1_lock.json"
PARENT_LOCK_PATH = ROOT / "research" / "f1_v31_dhff_v1_lock.json"
PARENT_ENGINE_PATH = ROOT / "backend" / "f1_distributed_harmonic_fold_flux.py"
ENGINE_PATH = ROOT / "backend" / "f1_axial_voronoi_harmonic_fold_measure.py"
BUILDER_PATH = ROOT / "scripts" / "build_f1_v32_avhfm_source_free_fixtures.py"
RENDERER_PATH = ROOT / "scripts" / "render_f1_v32_avhfm_source_free.py"
RUNNER_PATH = ROOT / "scripts" / "run_f1_v32_avhfm_source_free.py"
GATES_PATH = ROOT / "scripts" / "run_f1_v32_avhfm_gates.py"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v32-avhfm-v1"
HARNESS_PATH = OUTPUT_ROOT / "preinvoke_harness_manifest.json"
VISUAL_PATH = OUTPUT_ROOT / "preinvoke_visual_manifest.json"
SIGNOFF_PATH = OUTPUT_ROOT / "preinvoke_manual_visual_signoff.json"
PRECOMMIT_PATH = OUTPUT_ROOT / "precommit_manifest.json"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_distributed_harmonic_fold_flux as parent  # noqa: E402
import backend.f1_axial_voronoi_harmonic_fold_measure as engine  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _invocations() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "AVHFM_INVOKE")


def _function_sha(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).strip().encode("utf-8")).hexdigest()


def _dummy_decision(index: int) -> dict[str, Any]:
    return {
        "candidate_id": index,
        "frozen_v16_order": index,
        "candidate_sha256": hashlib.sha256(f"candidate-{index}".encode("ascii")).hexdigest(),
        "action": engine.ACTION_NONE,
        "reason": "V32_ELECTRODE_0_COVERAGE",
        "anchor_xyz": [0.0, 0.0, 0.0],
        "axis_xyz": [0.0, 1.0, 0.0],
        "line_start_xyz": [0.0, -12.0, 0.0],
        "line_end_xyz": [0.0, 12.0, 0.0],
        "scale": 1.0,
        "accepted_length_q21": 0,
        "longest_run_q21": 0,
        "maximum_gap_q21": 24 * engine.Q21,
        "endpoint_left_q21": 0,
        "endpoint_right_q21": 0,
    }


def main() -> int:
    lock = _load(LOCK_PATH)
    parent_lock = _load(PARENT_LOCK_PATH)
    harness = _load(HARNESS_PATH)
    visual = _load(VISUAL_PATH)
    signoff = _load(SIGNOFF_PATH)
    source = ENGINE_PATH.read_text(encoding="utf-8")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("profile_exact", engine.PROFILE == lock["profile"], engine.PROFILE)
    check("primitive_exact", engine.PRIMITIVE == lock["short_name"], engine.PRIMITIVE)
    constant_map = {
        "MAX_SOURCE_SAMPLES": "source_samples_cap", "MAX_CANDIDATES": "candidate_cap", "MAX_LOCAL_VERTICES": "candidate_vertices_cap", "MAX_LOCAL_EDGES": "candidate_edges_cap", "MAX_AXIAL_EVENTS": "axial_events_cap", "AUDIT_HALF_S": "audit_half_s", "SOLVE_HALF_S": "solve_half_s", "WINDOW_HALF_S": "window_half_s", "CELL_HALF_CAP_S": "cell_half_cap_s", "MEASURE_MINIMUM_S": "measure_minimum_s", "PHASE_MEASURE_MINIMUM_S": "phase_measure_minimum_s", "MAX_GAP_S": "maximum_gap_s", "LONGEST_RUN_MINIMUM_S": "longest_run_minimum_s", "ENDPOINT_MEASURE_MINIMUM_S": "endpoint_measure_minimum_s", "ELECTRODE_MEASURE_MINIMUM_S": "electrode_measure_minimum_s", "ELECTRODE_MAX_GAP_S": "electrode_maximum_gap_s", "AXIAL_LEAKAGE_MAXIMUM": "axial_leakage_maximum", "PCG_MAXIMUM_ITERATIONS": "pcg_maximum_iterations", "PCG_RESIDUAL": "pcg_relative_infinity_residual", "CACHE_HEADER_BYTES": "cache_header_bytes", "CACHE_CANDIDATE_BYTES": "cache_candidate_bytes", "MAX_CACHE_BYTES": "cache_max_bytes",
    }
    for source_name, lock_name in constant_map.items():
        observed = getattr(engine, source_name)
        check(f"constant_{source_name.lower()}", observed == lock["constants"][lock_name], {"source": observed, "lock": lock["constants"][lock_name]})
    parent_hash = hashlib.sha256(_canonical(parent_lock)).hexdigest()
    check("canonical_parent_lock_hash", parent_hash == lock["parent_lock_canonical_sha256"], parent_hash)
    imported_hashes = {name: _function_sha(getattr(parent, name)) for name in lock["imported_function_sha256"]}
    check("all_imported_parent_function_hashes", imported_hashes == lock["imported_function_sha256"], imported_hashes)
    inherited_constants = ("TAIL_BINS", "TAIL_MINIMUM_PER_BIN", "TAIL_MINIMUM_TOTAL", "POTENTIAL_BAND_HALF_WIDTH", "POTENTIAL_BAND_MINIMUM_NODES", "POTENTIAL_BAND_MINIMUM_MASS", "ANGULAR_BIN_MINIMUM_NODES", "ANGULAR_BIN_MINIMUM_MASS", "MAXIMUM_BACKTRACK_DEGREES", "MAXIMUM_ANGULAR_VARIATION_DEGREES", "OFF_ARC_MARGIN_DEGREES", "MAXIMUM_OFF_ARC_MASS", "MINIMUM_EFFECTIVE_PARTICIPATION", "MAXIMUM_EDGE_CURRENT_FRACTION")
    check("inherited_local_constants_are_parent_objects", all(getattr(engine, name) == getattr(parent, name) for name in inherited_constants), list(inherited_constants))
    local_source = inspect.getsource(engine._local_stage)
    static_local_proof = all(symbol in local_source for symbol in ("TAIL_BINS", "TAIL_MINIMUM_PER_BIN", "TAIL_MINIMUM_TOTAL", "POTENTIAL_BAND_HALF_WIDTH", "POTENTIAL_BAND_MINIMUM_NODES", "POTENTIAL_BAND_MINIMUM_MASS", "ANGULAR_BIN_MINIMUM_NODES", "ANGULAR_BIN_MINIMUM_MASS", "MAXIMUM_BACKTRACK_DEGREES", "MAXIMUM_ANGULAR_VARIATION_DEGREES", "OFF_ARC_MARGIN_DEGREES", "MAXIMUM_OFF_ARC_MASS", "MINIMUM_EFFECTIVE_PARTICIPATION", "MAXIMUM_EDGE_CURRENT_FRACTION", "theta_degrees"))
    check("static_v31_local_adjudicator_threshold_mapping", static_local_proof, hashlib.sha256(local_source.encode("utf-8")).hexdigest())
    check("local_reason_precedence", source.index('"V32_CURRENT_MEASURE"', source.index("def _local_stage")) < source.index('"V32_AXIAL_LEAKAGE_MEASURE"', source.index("def _local_stage")) < source.index('"V32_TAIL_COVERAGE_MEASURE"', source.index("def _local_stage")) < source.index('"V32_POTENTIAL_COVERAGE_MEASURE"', source.index("def _local_stage")), "current < leakage < tail < potential")
    check("engine_no_file_io", "pathlib" not in imports and not any(name in calls for name in ("open", "read_text", "write_text", "read_bytes", "write_bytes")), sorted(imports))
    check("engine_never_calls_v31_runner", "select_unified_v31" not in source and "run_f1_v31" not in source, "source scan")
    check("ordinary_nms_exactly_one_callsite", calls.count("ordinary_nms") == 1, calls.count("ordinary_nms"))
    check("no_station_mask_or_lattice", all(token not in source for token in ("station_mask", "STATION_OFFSETS", "STATION_COUNT")), "source scan")
    check("single_global_pcg", calls.count("_pcg") == 1 and "for value, cell in zip(event_values, cells)" in source, calls.count("_pcg"))
    check("quantized_sorted_event_windows", all(token in source for token in ("bisect_left", "bisect_right", '"z_q20"', "window_half_q20")), "event window symbols")
    check("runner_requires_lock_before_invocation", runner_source.index("_require_locks()") < runner_source.index("_append_invocation(fixture_id"), "lexical order")
    check("runner_records_before_mechanism", runner_source.rindex("_append_invocation(fixture_id") < runner_source.rindex("select_unified_v32("), "run call order")
    check("runner_registered_variants_only", "VARIANTS =" in runner_source and "unregistered V32 fixture or variant" in runner_source, len(ast.parse(runner_source).body))
    renderer_source = RENDERER_PATH.read_text(encoding="utf-8")
    check("renderer_exact_artifacts", all(name in renderer_source for name in ("raw_multiview.png", "axial_unwrapped_input.png", "harmonic_current_multiview.png", "axial_measure_audit.png", "final_overlay.png")), "artifact scan")
    builder_source = BUILDER_PATH.read_text(encoding="utf-8")
    check("builder_independent_of_v32_engine", "f1_axial_voronoi_harmonic_fold_measure" not in builder_source, "source scan")
    sf27_truth = harness.get("sf27_analytic_truth", {})
    check("sf27_pure_analytic_truth", sf27_truth.get("passed") is True and sf27_truth.get("complete_transition_window_count") == 0 and sf27_truth.get("expected_first_local_coverage_token") == lock["expected_results"]["SF27_STAGGERED_HALF_FOLDS"]["reason"], sf27_truth)
    check("no_prior_mechanism_invocation", _invocations() == 0, _invocations())

    fixture_rows = harness.get("fixture_rows", [])
    check("fixture_manifest_complete", len(fixture_rows) == 28 and [row["fixture_id"] for row in fixture_rows[:27]] == lock["physical_fixture_ids"] and fixture_rows[-1]["fixture_id"] == "STRESS_200K_128", len(fixture_rows))
    fixture_hashes_ok = all(_sha(ROOT / row["path"]) == row["sha256"] for row in fixture_rows)
    check("all_fixture_hashes_current", fixture_hashes_ok, len(fixture_rows))
    check("visual_manifest_complete", len(visual.get("rows", [])) == 62, len(visual.get("rows", [])))
    visual_hashes_ok = all(_sha(ROOT / row["path"]) == row["sha256"] and row["width"] == 4096 and row["height"] == 4096 for row in visual.get("rows", []))
    check("all_visual_hashes_current", visual_hashes_ok, len(visual.get("rows", [])))
    signoff_by_path = {row["path"]: row for row in signoff.get("rows", [])}
    individual_signed = len(signoff_by_path) == 62 and all(signoff_by_path.get(row["path"], {}).get("signed") is True and signoff_by_path[row["path"]]["sha256"] == row["sha256"] and signoff_by_path[row["path"]].get("disposition") for row in visual.get("rows", []))
    aggregate_signed = signoff.get("all_signed") is True and signoff.get("decision") == "ACCEPT_ALL_PREINVOKE_SOURCE_GEOMETRY" and signoff.get("visual_manifest_sha256") == _sha(VISUAL_PATH) and int(signoff.get("artifact_count", -1)) == 62
    check("all_62_individual_native_visual_signatures", individual_signed and aggregate_signed, {"individual": individual_signed, "aggregate": aggregate_signed})

    cache = engine.build_fixed_cache([_dummy_decision(index) for index in range(128)])
    records_ok = len(cache) == 131_328 and cache[28:32] == b"\0" * 4
    for index in range(128):
        record = cache[256 + index * 1024:256 + (index + 1) * 1024]
        records_ok = records_ok and record[784:960] == b"\0" * 176 and record[992:1024] == b"\0" * 32 and hashlib.sha256(record[:960]).digest() == record[960:992]
    check("canonical_fixed_cache_exact_size_and_zero_reserve", records_ok, len(cache))
    forbidden_cache_tokens = tuple(token.encode("ascii") for token in ("fixture", "filename", "source_normals", "positions", "mesh_adjacencies", "potential", "current_edges"))
    check("cache_contains_no_forbidden_geometry_or_identity", not any(token in cache for token in forbidden_cache_tokens), "binary token scan")

    source_paths = (ENGINE_PATH, PARENT_ENGINE_PATH, LOCK_PATH, PARENT_LOCK_PATH, BUILDER_PATH, RENDERER_PATH, RUNNER_PATH, GATES_PATH, Path(__file__))
    source_hashes = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in source_paths]
    fixture_hashes = [{"path": row["path"], "sha256": row["sha256"]} for row in fixture_rows]
    preinvoke_visual_hashes = [{"path": row["path"], "sha256": row["sha256"]} for row in visual["rows"]]
    preinvoke_visual_hashes.extend(({"path": str(VISUAL_PATH.relative_to(ROOT)), "sha256": _sha(VISUAL_PATH)}, {"path": str(SIGNOFF_PATH.relative_to(ROOT)), "sha256": _sha(SIGNOFF_PATH)}, {"path": str(HARNESS_PATH.relative_to(ROOT)), "sha256": _sha(HARNESS_PATH)}))
    passed = all(row["passed"] for row in checks)
    manifest = {"kind": "v32_avhfm_static_precommit_manifest", "passed": passed, "mechanism_invocation_count": _invocations(), "checks": checks, "source_hashes": source_hashes, "fixture_hashes": fixture_hashes, "preinvoke_visual_hashes": preinvoke_visual_hashes, "token": lock["tokens"]["mechanism_locked"] if passed else lock["tokens"]["terminal_retire"]}
    _write(PRECOMMIT_PATH, manifest)
    print(json.dumps({"passed": passed, "checks": len(checks), "failed": [row["name"] for row in checks if not row["passed"]], "mechanism_invocation_count": _invocations(), "token": manifest["token"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
