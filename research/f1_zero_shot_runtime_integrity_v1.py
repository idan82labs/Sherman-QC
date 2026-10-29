"""Static, fail-closed checks for the Sherman QC zero-shot runtime boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable


def _tree(root: Path, relative: str) -> ast.Module:
    return ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"missing function: {name}")


def _default_value(function: ast.FunctionDef | ast.AsyncFunctionDef, argument: str) -> Any:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults = [None] * (len(positional) - len(function.args.defaults)) + list(function.args.defaults)
    mapping = {arg.arg: default for arg, default in zip(positional, defaults)}
    mapping.update({arg.arg: default for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults)})
    value = mapping.get(argument)
    return value.value if isinstance(value, ast.Constant) else None


def _calls_named(function: ast.AST, call_name: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if name == call_name:
            calls.append(node)
    return calls


def _keyword_constant(call: ast.Call, keyword: str) -> Any:
    for row in call.keywords:
        if row.arg == keyword and isinstance(row.value, ast.Constant):
            return row.value.value
    return object()


def _used_names(function: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}


def _no_calls(function: ast.AST, banned: Iterable[str]) -> bool:
    banned_set = set(banned)
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if name in banned_set:
            return False
    return True


def validate(root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    pipeline = _tree(root, "backend/bend_inspection_pipeline.py")
    progressive = _function(pipeline, "run_progressive_bend_inspection")
    checks["pipeline_truth_hints_default_false"] = _default_value(progressive, "use_part_truth_hints") is False

    for relative, function_name, prefix in (
        ("backend/bend_analysis_worker.py", "run_worker", "worker"),
        ("domains/bend/application/analyze_bend_job.py", "analyze_bend_job", "domain_entry"),
    ):
        function = _function(_tree(root, relative), function_name)
        calls = _calls_named(function, "run_progressive_bend_inspection")
        checks[f"{prefix}_has_single_progressive_call"] = len(calls) == 1
        if calls:
            call = calls[0]
            checks[f"{prefix}_truth_hints_explicit_false"] = _keyword_constant(call, "use_part_truth_hints") is False
            checks[f"{prefix}_expected_override_explicit_none"] = _keyword_constant(call, "expected_bend_override") is None
            checks[f"{prefix}_angle_hint_explicit_none"] = _keyword_constant(call, "spec_bend_angles_hint") is None
            checks[f"{prefix}_schedule_hint_explicit_none"] = _keyword_constant(call, "spec_schedule_type") is None

    worker_state = _function(_tree(root, "backend/bend_analysis_worker.py"), "_resolve_scan_state")
    worker_names = _used_names(worker_state)
    checks["worker_scan_state_ignores_filename"] = "scan_name" not in worker_names
    checks["worker_scan_state_ignores_report_counts"] = "report_dict" not in worker_names

    heatmap_state = _function(
        _tree(root, "domains/bend/services/heatmap_consistency_service.py"),
        "_normalize_scan_state",
    )
    checks["heatmap_scan_state_ignores_filename"] = "scan_name" not in _used_names(heatmap_state)

    profile_state = _function(_tree(root, "backend/zero_shot_bend_profile.py"), "_resolve_scan_state_hint")
    profile_names = _used_names(profile_state)
    checks["zero_shot_profile_ignores_part_identity"] = "part_id" not in profile_names
    checks["zero_shot_profile_ignores_scan_filename"] = "scan_path" not in profile_names

    for relative, prefix in (
        ("apps/api/services/bend_inspection_service.py", "modular_api"),
        ("backend/_server_legacy_impl.py", "legacy_api"),
    ):
        extractor = _function(_tree(root, relative), "extract_cad_bends")
        checks[f"{prefix}_cad_extract_has_no_truth_lookup"] = _no_calls(
            extractor,
            {"load_expected_bend_overrides", "load_part_feature_policies"},
        )

    for name, passed in checks.items():
        if not passed:
            blockers.append(name.upper())
    return {
        "schema": "SHERMAN_QC_F1_ZERO_SHOT_RUNTIME_INTEGRITY_V1",
        "status": "PASS" if not blockers else "FAIL",
        "zero_shot_default": not blockers,
        "checks": checks,
        "blockers": blockers,
    }

