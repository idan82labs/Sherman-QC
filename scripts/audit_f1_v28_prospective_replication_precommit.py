#!/usr/bin/env python3
"""Static precommit audit for the single authorized V28 replication.

This wrapper never executes V27, V28, a fixture, or a scan.  It binds the
professor-selected action to the frozen sources and determines whether the
existing hashed pipeline is eligible to enter the source-free gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys


EXPECTED_V27_SHA256 = (
    "d67db9e952d8e127cebe4c8117fe3d604830a83a084d73c6aeeb8ec812015527"
)
EXPECTED_V28_SHA256 = (
    "0f0408bb5040d11b3cd6180a4ab46224d720f2a82ff431b6f6b6eff1b06cc9af"
)
ACTION = "V28_FROZEN_PROSPECTIVE_PAIRED_RETURN_REPLICATION_1"
TERMINAL_TOKEN = "V28_PRECOMMIT_PIPELINE_CONTRACT_FAILURE_RETIRE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function not found: {name}")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ast.unparse(node.func)


def _ordered_calls(function: ast.FunctionDef) -> list[dict]:
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    return [
        {
            "line": int(node.lineno),
            "column": int(node.col_offset),
            "name": _call_name(node),
            "expression": ast.unparse(node),
        }
        for node in calls
    ]


def _rank_audit(v28_tree: ast.AST) -> dict:
    function = _function(v28_tree, "complete_paired_return_axis")
    tuple_node: ast.Tuple | None = None
    sorted_key = None
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "choices"
            and node.func.attr == "append"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Tuple)
        ):
            tuple_node = node.args[0]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "choices"
        ):
            for keyword in node.keywords:
                if keyword.arg == "key":
                    sorted_key = ast.unparse(keyword.value)
    if tuple_node is None:
        raise RuntimeError("V28 choices.append rank tuple not found")
    fields = [ast.unparse(value) for value in tuple_node.elts[:-1]]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected V28 rank field count: {len(fields)}")
    uses_round = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "round"
        for value in tuple_node.elts[:-1]
        for node in ast.walk(value)
    )
    return {
        "fields_in_comparison_order": fields,
        "directions": ["ascending", "descending_via_negation", "ascending"],
        "floating_point_canonicalization": "none" if not uses_round else "round",
        "final_stable_tiebreak_field": fields[-1],
        "sorted_key": sorted_key,
        "total_order_for_distinct_candidate_ids": (
            "candidate_id" in fields[-1] and sorted_key == "lambda item: item[:3]"
        ),
    }


def _candidate_group_order_audit(v28_tree: ast.AST, v27_source: str) -> dict:
    function = _function(v28_tree, "complete_paired_return_axis")
    all_rows_order = False
    for node in ast.walk(function):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "row"
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "all_rows"
        ):
            all_rows_order = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "eligible"
                and child.func.attr == "append"
                for child in ast.walk(node)
            )
    fit_uses_input_index_order = (
        "np.vstack([endpoints(rows[index]) for index in indices])" in v27_source
    )
    sign_anchors_first_member = (
        "reference = endpoints(rows[indices[0]])[1] - endpoints(rows[indices[0]])[0]"
        in v27_source
    )
    stable_member_ids = (
        'member_ids = sorted(str(rows[index].get("candidate_id") or "") for index in indices)'
        in v27_source
    )
    return {
        "construction_order": "all_rows input iteration order"
        if all_rows_order
        else "not_extracted",
        "eligible_appended_in_input_order": all_rows_order,
        "fit_point_order_uses_component_input_indices": fit_uses_input_index_order,
        "fitted_axis_sign_anchors_component_first_member": sign_anchors_first_member,
        "stitched_member_ids_are_sorted": stable_member_ids,
        "canonical_fitted_axis_bytes_proven_input_order_independent": False,
        "reason": (
            "Eligible groups inherit all_rows order; the frozen V27 fit consumes "
            "component indices in that order and anchors axis sign to indices[0]."
        ),
    }


def _pipeline_audit(selector_tree: ast.AST, build_tree: ast.AST) -> dict:
    selector_main = _function(selector_tree, "main")
    calls = _ordered_calls(selector_main)
    complete_calls = [row for row in calls if row["name"] == "complete_paired_return_axis"]
    build_calls = [row for row in calls if row["name"] == "build_result"]
    if len(complete_calls) != 1 or len(build_calls) != 1:
        raise RuntimeError("unexpected V28 selector call structure")
    complete_line = complete_calls[0]["line"]
    build_line = build_calls[0]["line"]
    between = [
        row for row in calls if complete_line < row["line"] < build_line
    ]
    post_v28_nms_calls = [
        row for row in between if "nms" in row["name"].lower()
    ]
    build_result = _function(build_tree, "build_result")
    build_result_nms_calls = [
        row for row in _ordered_calls(build_result) if "nms" in row["name"].lower()
    ]
    ordinary_nms_present = bool(post_v28_nms_calls or build_result_nms_calls)
    return {
        "observed_order": [
            "frozen_upstream_select_unified_including_complex_route_nms",
            "V27_selected_fragment_stitching",
            "V28_paired_return_completion",
            "build_result_sort_and_serialization",
        ],
        "required_order": [
            "frozen_upstream_candidate_generation",
            "frozen_selection",
            "V27_selected_fragment_stitching",
            "V28_paired_return_completion",
            "frozen_ordinary_output_nms",
            "research_audit_serialization",
        ],
        "calls_between_v28_and_build_result": between,
        "build_result_nms_calls": build_result_nms_calls,
        "ordinary_output_nms_after_v28_present": ordinary_nms_present,
        "existing_lane_survives_v28_duplicate_collision_enforced": False,
        "v28_proposal_suppressed_on_existing_lane_collision_enforced": False,
        "reason": (
            "The frozen complex-route NMS executes inside select_unified before "
            "V27/V28. After V28, the selector assigns paired['selected'] and calls "
            "build_result, which sorts and serializes without NMS."
        ),
    }


def build_manifest(root: Path) -> dict:
    v27_path = root / "backend/f1_collinear_fragment_stitch.py"
    v28_path = root / "backend/f1_paired_return_completion.py"
    selector_path = root / "scripts/select_f1_mesh_route_v28.py"
    build_path = root / "scripts/select_f1_mesh_route_v16.py"
    ruling_path = root / "docs/research/2026-08-14_gpt56_sol_pro_post_v28_ruling_verbatim.md"
    wrapper_path = Path(__file__).resolve()

    v27_source = v27_path.read_text(encoding="utf-8")
    v28_source = v28_path.read_text(encoding="utf-8")
    selector_source = selector_path.read_text(encoding="utf-8")
    build_source = build_path.read_text(encoding="utf-8")
    v27_sha = _sha256(v27_path)
    v28_sha = _sha256(v28_path)

    rank = _rank_audit(ast.parse(v28_source))
    group_order = _candidate_group_order_audit(ast.parse(v28_source), v27_source)
    pipeline = _pipeline_audit(ast.parse(selector_source), ast.parse(build_source))

    failures = []
    if v27_sha != EXPECTED_V27_SHA256:
        failures.append("V27_HASH_MISMATCH")
    if v28_sha != EXPECTED_V28_SHA256:
        failures.append("V28_HASH_MISMATCH")
    if not rank["total_order_for_distinct_candidate_ids"]:
        failures.append("V28_RANK_NOT_TOTAL")
    if not group_order["canonical_fitted_axis_bytes_proven_input_order_independent"]:
        failures.append("V28_INPUT_ORDER_DEPENDENT_FITTED_AXIS_BYTES")
    if not pipeline["ordinary_output_nms_after_v28_present"]:
        failures.append("POST_V28_ORDINARY_NMS_ABSENT")
    if not pipeline["existing_lane_survives_v28_duplicate_collision_enforced"]:
        failures.append("POST_V28_EXISTING_LANE_PRECEDENCE_ABSENT")

    return {
        "schema_version": 1,
        "action": ACTION,
        "architecture_role": "geometric_postprocessor",
        "structural_hypothesis": "paired_return_counterpart",
        "material_provenance": "unresolved_surface_only",
        "semantic_universality_claim": False,
        "ruling": {
            "path": str(ruling_path.relative_to(root)),
            "sha256": _sha256(ruling_path),
        },
        "frozen_sources": {
            "v27": {
                "path": str(v27_path.relative_to(root)),
                "expected_sha256": EXPECTED_V27_SHA256,
                "actual_sha256": v27_sha,
                "matches": v27_sha == EXPECTED_V27_SHA256,
            },
            "v28": {
                "path": str(v28_path.relative_to(root)),
                "expected_sha256": EXPECTED_V28_SHA256,
                "actual_sha256": v28_sha,
                "matches": v28_sha == EXPECTED_V28_SHA256,
            },
            "selector": {
                "path": str(selector_path.relative_to(root)),
                "sha256": _sha256(selector_path),
            },
            "build_result": {
                "path": str(build_path.relative_to(root)),
                "sha256": _sha256(build_path),
            },
        },
        "audit_wrapper": {
            "path": str(wrapper_path.relative_to(root)),
            "sha256": _sha256(wrapper_path),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "rank_extraction": rank,
        "candidate_group_construction": group_order,
        "pipeline_contract": pipeline,
        "precommit_pass": not failures,
        "terminal_failures": failures,
        "terminal_token": None if not failures else TERMINAL_TOKEN,
        "fixtures_executed": False,
        "real_scans_executed": False,
        "detector_invoked": False,
        "retirement": {
            "paired_return_completion_v28": "RETIRED_CURRENT_ARCHITECTURE"
            if failures
            else "ELIGIBLE_FOR_SOURCE_FREE_GATES",
            "v28b_authorized": False,
            "threshold_relaxation_authorized": False,
            "replacement_fixture_authorized": False,
            "second_positive_authorized": False,
        },
        "prohibited_information": {
            "cad_used": False,
            "drawing_used": False,
            "gold_used": False,
            "expected_count_used": False,
            "part_identity_used": False,
            "filename_identity_used": False,
            "benchmark_feedback_used": False,
            "source_points_serialized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = build_manifest(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "precommit_pass": manifest["precommit_pass"],
                "terminal_failures": manifest["terminal_failures"],
                "terminal_token": manifest["terminal_token"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["precommit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
