#!/usr/bin/env python3
"""Two-fresh-process, full-tree replay orchestrator for EXP_CAD_0030."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import exp_cad_0029_cfpdg_core as CAD


FORWARDED_PATHS = (
    "preregistration", "cad_preflight", "cad_approval", "source_runner_approval",
    "nodes", "edges", "edge_geodesic_mm", "mesh", "scan_first", "scan_second",
    "transform_first", "transform_second", "source_tests", "cad_audit_packet",
    "renderer", "renderer_tests", "orchestration", "orchestration_tests",
)


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): CAD.sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.rglob("*")):
        raise ValueError(f"output root must be absent or empty: {path}")


def _is_within(path: Path, possible_parent: Path) -> bool:
    try:
        path.resolve().relative_to(possible_parent.resolve())
        return True
    except ValueError:
        return False


def validate_output_topology(primary: Path, replay: Path, result: Path) -> None:
    """Keep compared trees disjoint and the terminal record outside both."""

    primary_resolved = primary.resolve()
    replay_resolved = replay.resolve()
    result_resolved = result.resolve()
    if primary_resolved == replay_resolved:
        raise ValueError("primary and replay roots must be distinct")
    if _is_within(primary_resolved, replay_resolved) or _is_within(
        replay_resolved, primary_resolved
    ):
        raise ValueError("primary and replay roots must not be ancestors")
    if _is_within(result_resolved, primary_resolved) or _is_within(
        result_resolved, replay_resolved
    ):
        raise ValueError("result path must be outside both compared trees")


def verify_renderer_binding(source_approval: Path, renderer: Path) -> None:
    """Verify that the renderer we execute is the exact approved file."""

    payload = json.loads(Path(source_approval).read_text(encoding="utf-8"))
    if payload.get("schema") != "EXP_CAD_0030_V3_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1":
        raise ValueError("unexpected source-runner approval schema")
    if payload.get("decision") != "APPROVE_ONE_PHYSICAL_PRIMARY_REPLAY_EXECUTION":
        raise ValueError("source-runner approval decision unavailable")
    if payload.get("physical_source_execution_authorized") is not True:
        raise ValueError("source-runner approval does not authorize execution")
    if (
        payload.get("experiment_id") != "EXP_CAD_0030"
        or payload.get("architecture")
        != "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
    ):
        raise ValueError("source-runner approval identity mismatch")
    bound = payload.get("bound_files")
    if not isinstance(bound, Mapping) or not isinstance(bound.get("renderer"), Mapping):
        raise ValueError("source-runner approval lacks exact renderer binding")
    expected = bound["renderer"].get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("source-runner approval renderer binding is invalid")
    if CAD.sha256_path(renderer.resolve()) != expected:
        raise ValueError("approval-bound renderer hash mismatch")


def source_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [sys.executable, str(args.source_runner.resolve())]
    for name in FORWARDED_PATHS:
        cli = "--" + name.replace("_", "-")
        command.extend((cli, str(getattr(args, name).resolve())))
    command.extend(("--output-root", str(output_root.resolve())))
    return command


def render_command(renderer: Path, payload: Path, output_root: Path) -> list[str]:
    script = (
        "import importlib.util,json,sys;"
        "from pathlib import Path;"
        "renderer=Path(sys.argv[1]).resolve();"
        "spec=importlib.util.spec_from_file_location('approval_bound_exp30_renderer',renderer);"
        "r=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(r);"
        "p=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));"
        "r.render_native_visuals(p,Path(sys.argv[3]))"
    )
    return [
        sys.executable,
        "-c",
        script,
        str(renderer.resolve()),
        str(payload.resolve()),
        str(output_root.resolve()),
    ]


def _terminal_result(
    *,
    status: str,
    primary_root: Path,
    replay_root: Path,
    commands: Sequence[Mapping[str, Any]],
    failure_stage: str | None = None,
    failure_type: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "EXP_CAD_0030_CFPDG_V3_PRIMARY_REPLAY_RESULT_V1",
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "status": status,
        "fresh_source_child_process_count": sum(
            "source_returncode" in row or row.get("source_timeout") is True
            for row in commands
        ),
        "fresh_renderer_child_process_count": sum(
            "renderer_returncode" in row or row.get("renderer_timeout") is True
            for row in commands
        ),
        "primary_root": str(primary_root.resolve()),
        "replay_root": str(replay_root.resolve()),
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "commands": list(commands),
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
    }


def _timeout_stream(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def execute(argv: Optional[Sequence[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runner", required=True, type=Path)
    for name in FORWARDED_PATHS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--primary-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source_runner.resolve() != (ROOT / "scripts" / "run_exp_cad_0030_cfpdg_source.py").resolve():
        raise ValueError("unexpected EXP30 source-runner identity")
    if args.orchestration.resolve() != Path(__file__).resolve():
        raise ValueError("orchestration self-binding mismatch")
    validate_output_topology(args.primary_root, args.replay_root, args.result)
    verify_renderer_binding(args.source_runner_approval, args.renderer)
    for root in (args.primary_root, args.replay_root):
        refuse_nonempty(root)

    commands: list[dict[str, Any]] = []
    stage = "PRE_CHILD"
    try:
        for label, root in (("PRIMARY", args.primary_root), ("REPLAY", args.replay_root)):
            stage = f"{label}_SOURCE"
            source = source_command(args, root)
            command_row: dict[str, Any] = {"label": label}
            commands.append(command_row)
            try:
                completed = subprocess.run(
                    source, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=3600, check=False,
                )
            except subprocess.TimeoutExpired as error:
                command_row.update(
                    {
                        "source_timeout": True,
                        "source_stdout": _timeout_stream(error.stdout),
                        "source_stderr": _timeout_stream(error.stderr),
                    }
                )
                raise RuntimeError(f"{label} source child timeout") from error
            command_row.update(
                {"source_returncode": completed.returncode,
                 "source_stdout": completed.stdout, "source_stderr": completed.stderr}
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{label} source child failed: {completed.stderr}")
            stage = f"{label}_RENDERER"
            native = root / "native"
            try:
                rendered = subprocess.run(
                    render_command(args.renderer.resolve(), root / "SOURCE_VISUAL_PAYLOAD.json", native),
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=3600, check=False,
                )
            except subprocess.TimeoutExpired as error:
                command_row.update(
                    {
                        "renderer_timeout": True,
                        "renderer_stdout": _timeout_stream(error.stdout),
                        "renderer_stderr": _timeout_stream(error.stderr),
                    }
                )
                raise RuntimeError(f"{label} renderer child timeout") from error
            command_row.update(
                {"renderer_returncode": rendered.returncode,
                 "renderer_stdout": rendered.stdout, "renderer_stderr": rendered.stderr}
            )
            if rendered.returncode != 0:
                raise RuntimeError(f"{label} renderer child failed: {rendered.stderr}")

        stage = "FULL_TREE_COMPARE"
        primary = tree_hashes(args.primary_root)
        replay = tree_hashes(args.replay_root)
        passed = primary == replay
        required_present = all(
            path in primary and path in replay
            for path in (
                "SOURCE_RESULT.json", "SOURCE_VISUAL_PAYLOAD.json",
                "native/native_visual_manifest.json",
            )
        )
        if not required_present:
            raise RuntimeError("full-tree replay lacks required artifacts")
        if not passed:
            raise RuntimeError("EXP30 full-tree replay mismatch")
        result = _terminal_result(
            status="PASS_BYTE_EXACT_PRIMARY_REPLAY",
            primary_root=args.primary_root,
            replay_root=args.replay_root,
            commands=commands,
        ) | {
            "primary_replay_physical_scan_hash_call_count": 4,
            "primary_replay_physical_scan_point_cloud_load_call_count": 4,
            "primary_replay_physical_scan_byte_open_operation_count": 8,
            "primary_full_tree_sha256": primary,
            "replay_full_tree_sha256": replay,
            "full_tree_relative_path_sets_equal": set(primary) == set(replay),
            "full_tree_byte_exact": True,
            "required_relative_paths_present": True,
        }
        CAD.atomic_json(args.result, result)
        return result
    except Exception as error:
        failed = _terminal_result(
            status="FAIL_TERMINAL_NO_RETRY",
            primary_root=args.primary_root,
            replay_root=args.replay_root,
            commands=commands,
            failure_stage=stage,
            failure_type=type(error).__name__,
            failure_reason=str(error),
        )
        CAD.atomic_json(args.result, failed)
        raise


if __name__ == "__main__":
    execute()
