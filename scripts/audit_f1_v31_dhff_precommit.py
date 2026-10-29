#!/usr/bin/env python3
"""Static, non-invoking V31 DHFF contract and freeze audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v31_dhff_v1_lock.json"
ENGINE_PATH = ROOT / "backend" / "f1_distributed_harmonic_fold_flux.py"
BUILDER_PATH = ROOT / "scripts" / "build_f1_v31_dhff_source_free_fixtures.py"
RENDERER_PATH = ROOT / "scripts" / "render_f1_v31_dhff_source_free.py"
RUNNER_PATH = ROOT / "scripts" / "run_f1_v31_dhff_source_free.py"
GATES_PATH = ROOT / "scripts" / "run_f1_v31_dhff_gates.py"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v31-dhff-v1"
HARNESS_PATH = OUTPUT_ROOT / "gate1_harness_manifest.json"
VISUAL_PATH = OUTPUT_ROOT / "gate1_raw_visual_manifest.json"
VISUAL_SIGNOFF_PATH = OUTPUT_ROOT / "gate1_manual_visual_signoff.json"
PRECOMMIT_PATH = OUTPUT_ROOT / "precommit_manifest.json"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal_constants(tree: ast.AST) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return values


def _invocations() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "DHFF_INVOKE")


def main() -> int:
    lock = _load(LOCK_PATH)
    engine_source = ENGINE_PATH.read_text(encoding="utf-8")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    engine_tree = ast.parse(engine_source)
    runner_tree = ast.parse(runner_source)
    constants = _literal_constants(engine_tree)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("profile_exact", constants.get("PROFILE") == lock["profile"], constants.get("PROFILE"))
    check("primitive_exact", constants.get("PRIMITIVE") == lock["primitive"], constants.get("PRIMITIVE"))
    mappings = {
        "STATION_COUNT": "station_count",
        "STATION_HALF_WIDTH_S": "station_half_width_s",
        "STATION_RADIUS_S": "station_radius_s",
        "STATION_VOXEL_S": "station_voxel_s",
        "MAX_STATION_NODES": "maximum_station_nodes",
        "MAX_STATION_EDGES": "maximum_station_edges",
        "MUTUAL_NEIGHBORS": "mutual_neighbors",
        "EDGE_MAXIMUM_S": "edge_maximum_s",
        "EDGE_NORMAL_DEGREES": "edge_normal_degrees",
        "EDGE_NORMAL_TRAVEL_MAXIMUM": "edge_normal_travel_maximum",
        "PCG_MAXIMUM_ITERATIONS": "pcg_maximum_iterations",
        "MINIMUM_EFFECTIVE_PARTICIPATION": "minimum_effective_participation",
        "MAXIMUM_EDGE_CURRENT_FRACTION": "maximum_edge_current_fraction",
        "MINIMUM_PASSING_STATIONS": "minimum_passing_stations",
        "MINIMUM_CONSECUTIVE_PASSING": "minimum_consecutive_passing",
        "MAX_SOURCE_SAMPLES": "maximum_source_samples",
        "MAX_ELIGIBLE_CANDIDATES": "maximum_eligible_candidates",
        "CACHE_HEADER_BYTES": "cache_header_bytes",
        "CACHE_CANDIDATE_BYTES": "cache_candidate_bytes",
        "MAX_CACHE_BYTES": "maximum_cache_bytes",
        "Q18": "q18",
        "Q24": "q24",
    }
    for source_name, lock_name in mappings.items():
        check(f"constant_{source_name.lower()}", constants.get(source_name) == lock["constants"][lock_name], {"source": constants.get(source_name), "lock": lock["constants"][lock_name]})
    imports = {alias.name for node in ast.walk(engine_tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    check("engine_no_pathlib_or_file_io_import", "pathlib" not in imports, sorted(imports))
    engine_calls = [node.func.id for node in ast.walk(engine_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    check("ordinary_nms_exactly_one_callsite", engine_calls.count("ordinary_nms") == 1, engine_calls.count("ordinary_nms"))
    check("no_shortest_path_primitives", not any(name in engine_source for name in ("heapq", "dijkstra", "shortest_path", "minimum_cost_disjoint", "path_midpoint")), "forbidden path primitive scan")
    check("dirichlet_pcg_present", all(name in engine_source for name in ("_pcg", "PCG_MAXIMUM_ITERATIONS", "phi", "currents")), "electrical solve symbols")
    check("fixed_cache_layout_present", all(name in engine_source for name in ("CACHE_HEADER_BYTES", "CACHE_CANDIDATE_BYTES", "build_fixed_cache", "profile_hash")), "fixed cache symbols")
    check("runner_requires_precommit_before_invocation", runner_source.index("_require_locks()") < runner_source.index("_append_invocation("), "lexical order")
    check("runner_records_before_mechanism", runner_source.rindex("_append_invocation(fixture_id") < runner_source.rindex("select_unified_v31("), "run call order")
    check("runner_fixed_variants", "VARIANTS =" in runner_source and "unregistered V31 fixture or variant" in runner_source, "registered-only CLI")
    check("renderer_has_all_five_artifacts", all(name in RENDERER_PATH.read_text(encoding="utf-8") for name in ("raw_native_multiview_4096.png", "raw_station_sections_4096.png", "station_flux_4096.png", "phi_theta_certificate_4096.png", "final_native_overlay_4096.png")), "artifact names")
    check("builder_does_not_import_engine", "f1_distributed_harmonic_fold_flux" not in BUILDER_PATH.read_text(encoding="utf-8"), "separate analytic harness")
    check("no_prior_invocation", _invocations() == 0, _invocations())

    fixture_hashes: list[dict[str, str]] = []
    raw_visual_hashes: list[dict[str, str]] = []
    visuals_signed = False
    harness_passed = False
    if HARNESS_PATH.exists() and VISUAL_PATH.exists() and VISUAL_SIGNOFF_PATH.exists():
        harness = _load(HARNESS_PATH)
        visual = _load(VISUAL_PATH)
        signoff = _load(VISUAL_SIGNOFF_PATH)
        harness_passed = bool(harness.get("passed_analytic_checks")) and int(harness.get("mechanism_invocation_count", -1)) == 0
        visuals_signed = (
            signoff.get("kind") == "v31_dhff_gate1_manual_visual_signoff"
            and signoff.get("all_signed") is True
            and signoff.get("decision") == "ACCEPT_RAW_GEOMETRY"
            and int(signoff.get("artifact_count", -1)) == 42
            and int(signoff.get("fixture_count", -1)) == 21
            and signoff.get("raw_visual_manifest_sha256") == _sha(VISUAL_PATH)
            and len(visual.get("rows", [])) == 42
        )
        fixture_hashes = [{"path": row["path"], "sha256": row["sha256"]} for row in harness.get("fixture_rows", [])]
        raw_visual_hashes = [{"path": row["path"], "sha256": row["sha256"]} for row in visual.get("rows", [])]
        raw_visual_hashes.extend(
            [
                {"path": str(VISUAL_PATH.relative_to(ROOT)), "sha256": _sha(VISUAL_PATH)},
                {"path": str(VISUAL_SIGNOFF_PATH.relative_to(ROOT)), "sha256": _sha(VISUAL_SIGNOFF_PATH)},
            ]
        )
        check("all_21_physical_fixtures_present", len(lock["fixture_ids"]) == 21 and len(harness.get("fixture_rows", [])) == 22, len(harness.get("fixture_rows", [])))
        check("all_42_raw_visuals_present", len(visual.get("rows", [])) == 42, len(visual.get("rows", [])))
        check("analytic_harness_passed", harness_passed, harness.get("passed_analytic_checks"))
        check("manual_raw_visual_signatures", visuals_signed, {"aggregate_signed": visuals_signed, "artifact_count": signoff.get("artifact_count"), "manifest_sha256": signoff.get("raw_visual_manifest_sha256")})
    else:
        check("all_21_physical_fixtures_present", False, "harness not built")
        check("all_42_raw_visuals_present", False, "harness not built")
        check("analytic_harness_passed", False, "harness not built")
        check("manual_raw_visual_signatures", False, "visuals not signed")

    source_paths = (ENGINE_PATH, LOCK_PATH, BUILDER_PATH, RENDERER_PATH, RUNNER_PATH, GATES_PATH, Path(__file__))
    source_hashes = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in source_paths if path.exists()]
    passed = all(row["passed"] for row in checks)
    manifest = {
        "kind": "v31_dhff_static_precommit_manifest",
        "passed": passed,
        "mechanism_invocation_count": _invocations(),
        "checks": checks,
        "source_hashes": source_hashes,
        "fixture_hashes": fixture_hashes,
        "raw_visual_hashes": raw_visual_hashes,
        "token": lock["tokens"]["mechanism_locked"] if passed else lock["tokens"]["preinvocation_retire"],
    }
    _write(PRECOMMIT_PATH, manifest)
    print(json.dumps({"passed": passed, "checks": len(checks), "failed": [row["name"] for row in checks if not row["passed"]], "token": manifest["token"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
