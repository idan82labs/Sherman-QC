#!/usr/bin/env python3
"""Fresh-process orchestration for EXP31 intake + unchanged EXP30 science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import exp_cad_0029_cfpdg_core as CAD


EXPERIMENT_ID = "EXP_CAD_0031"
ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V4_SOURCE_INTERFACE_FILTER"
SOURCE_APPROVAL_SCHEMA = "EXP_CAD_0031_V4_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1"
FORWARDED_PATHS = (
    "preregistration", "scientific_preregistration", "terminal_parent_result",
    "terminal_parent_report", "terminal_parent_orchestration",
    "cad_reuse_audit_packet", "cad_preflight", "cad_reuse_approval",
    "source_runner_audit_packet", "source_runner_approval", "nodes", "edges", "edge_geodesic_mm", "mesh",
    "scan_first", "scan_second", "transform_first", "transform_second",
    "source_tests", "parent_renderer", "parent_renderer_tests", "orchestrator",
    "orchestrator_tests",
)


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): CAD.sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def refuse_nonempty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.rglob("*"))):
        raise ValueError(f"output root must be absent or an empty directory: {path}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_output_topology(primary: Path, replay: Path, result: Path) -> None:
    if primary.resolve() == replay.resolve():
        raise ValueError("primary and replay roots must be distinct")
    if _is_within(primary, replay) or _is_within(replay, primary):
        raise ValueError("primary/replay roots must not be ancestors")
    if _is_within(result, primary) or _is_within(result, replay):
        raise ValueError("terminal result must be outside compared trees")


def verify_parent_renderer_binding(source_approval: Path, renderer: Path) -> None:
    payload = json.loads(Path(source_approval).read_text(encoding="utf-8"))
    if payload.get("schema") != SOURCE_APPROVAL_SCHEMA:
        raise ValueError("unexpected EXP31 source approval schema")
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get(
        "architecture"
    ) != ARCHITECTURE:
        raise ValueError("EXP31 source approval identity mismatch")
    if payload.get("decision") != "APPROVE_ONE_EXP31_PHYSICAL_PRIMARY_REPLAY_EXECUTION":
        raise ValueError("EXP31 source approval decision unavailable")
    if payload.get("physical_source_execution_authorized") is not True:
        raise ValueError("EXP31 source approval lacks physical authority")
    files = payload.get("bound_files")
    if not isinstance(files, Mapping) or not isinstance(
        files.get("parent_renderer"), Mapping
    ):
        raise ValueError("EXP31 source approval lacks exact parent renderer binding")
    expected = files["parent_renderer"].get("sha256")
    if expected != CAD.sha256_path(renderer.resolve()):
        raise ValueError("EXP31 approval-bound parent renderer mismatch")


def source_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [sys.executable, str(args.source_runner.resolve())]
    for name in FORWARDED_PATHS:
        command.extend(
            ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
        )
    command.extend(("--output-root", str(output_root.resolve())))
    return command


def renderer_command(renderer: Path, payload: Path, output_root: Path) -> list[str]:
    script = (
        "import importlib.util,json,sys;"
        "from pathlib import Path;"
        "p=Path(sys.argv[1]).resolve();"
        "s=importlib.util.spec_from_file_location('exp31_bound_parent_renderer',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "x=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));"
        "m.render_native_visuals(x,Path(sys.argv[3]))"
    )
    return [
        sys.executable, "-c", script, str(renderer.resolve()),
        str(payload.resolve()), str(output_root.resolve()),
    ]


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accounting(root: Path) -> dict[str, int]:
    journal = root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    result = root / "SOURCE_RESULT.json"
    failure = root / "SOURCE_CHILD_FAILURE_ACCOUNTING.json"
    if journal.is_file():
        payload = _read_json(journal)
    elif result.is_file():
        payload = _read_json(result)
    elif failure.is_file():
        payload = _read_json(failure)
    else:
        return {
            "physical_scan_hash_call_count": 0,
            "physical_scan_point_cloud_load_call_count": 0,
            "physical_scan_byte_open_operation_count": 0,
        }
    return {
        key: int(payload.get(key, 0))
        for key in (
            "physical_scan_hash_call_count",
            "physical_scan_point_cloud_load_call_count",
            "physical_scan_byte_open_operation_count",
        )
    }


def _terminal(
    *,
    status: str,
    primary: Path,
    replay: Path,
    commands: Sequence[Mapping[str, Any]],
    failure_stage: str | None = None,
    failure_reason: str | None = None,
    full_tree_byte_exact: bool | None = None,
) -> dict[str, Any]:
    primary_accounting = _accounting(primary)
    replay_accounting = _accounting(replay)
    return {
        "schema": "EXP_CAD_0031_CFPDG_V4_PRIMARY_REPLAY_RESULT_V1",
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "scientific_core": "EXP_CAD_0030_CFPDG_V3_UNCHANGED",
        "native_renderer": "EXP_CAD_0030_CFPDG_V3_UNCHANGED",
        "status": status,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "fresh_primary_source_child_count": sum(
            row.get("label") == "PRIMARY"
            and ("source_returncode" in row or row.get("source_timeout") is True)
            for row in commands
        ),
        "fresh_replay_source_child_count": sum(
            row.get("label") == "REPLAY"
            and ("source_returncode" in row or row.get("source_timeout") is True)
            for row in commands
        ),
        "fresh_primary_renderer_child_count": sum(
            row.get("label") == "PRIMARY"
            and ("renderer_returncode" in row or row.get("renderer_timeout") is True)
            for row in commands
        ),
        "fresh_replay_renderer_child_count": sum(
            row.get("label") == "REPLAY"
            and ("renderer_returncode" in row or row.get("renderer_timeout") is True)
            for row in commands
        ),
        "physical_scan_hash_call_count": primary_accounting[
            "physical_scan_hash_call_count"
        ] + replay_accounting["physical_scan_hash_call_count"],
        "physical_scan_point_cloud_load_call_count": primary_accounting[
            "physical_scan_point_cloud_load_call_count"
        ] + replay_accounting["physical_scan_point_cloud_load_call_count"],
        "physical_scan_byte_open_operation_count": primary_accounting[
            "physical_scan_byte_open_operation_count"
        ] + replay_accounting["physical_scan_byte_open_operation_count"],
        "primary_root": str(primary.resolve()),
        "replay_root": str(replay.resolve()),
        "full_tree_byte_exact": full_tree_byte_exact,
        "commands": list(commands),
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
    }


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    CAD.atomic_json(path, payload)


def _run_source(
    args: argparse.Namespace, root: Path, label: str, commands: list[dict[str, Any]]
) -> subprocess.CompletedProcess[str]:
    command = source_command(args, root)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
    except subprocess.TimeoutExpired as error:
        commands.append({"label": label, "source_timeout": True})
        raise RuntimeError(f"{label} source child timed out") from error
    commands.append(
        {
            "label": label,
            "source_returncode": int(completed.returncode),
            "source_stdout": completed.stdout,
            "source_stderr": completed.stderr,
        }
    )
    return completed


def _run_renderer(
    args: argparse.Namespace, root: Path, label: str, commands: list[dict[str, Any]]
) -> subprocess.CompletedProcess[str]:
    command = renderer_command(
        args.parent_renderer, root / "SOURCE_VISUAL_PAYLOAD.json", root / "native"
    )
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
    except subprocess.TimeoutExpired as error:
        commands.append({"label": label, "renderer_timeout": True})
        raise RuntimeError(f"{label} parent renderer timed out") from error
    commands[-1].update(
        {
            "renderer_returncode": int(completed.returncode),
            "renderer_stdout": completed.stdout,
            "renderer_stderr": completed.stderr,
        }
    )
    return completed


def _write_native_context(root: Path) -> None:
    manifest = root / "native/native_visual_manifest.json"
    CAD.atomic_json(
        root / "native/EXP31_NATIVE_CONTEXT.json",
        {
            "schema": "EXP_CAD_0031_NATIVE_CONTEXT_V1",
            "experiment_id": EXPERIMENT_ID,
            "architecture": ARCHITECTURE,
            "context": "EXP31 intake successor / unchanged EXP30 scientific core and renderer",
            "parent_native_manifest": "native_visual_manifest.json",
            "parent_native_manifest_sha256": CAD.sha256_path(manifest),
            "silent_parent_relabeling_performed": False,
            "manual_native_resolution_review_required": True,
        },
    )


def execute(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runner", required=True, type=Path)
    parser.add_argument("--primary-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    for name in FORWARDED_PATHS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source_runner.resolve() != (
        ROOT / "scripts/run_exp_cad_0031_cfpdg_source.py"
    ).resolve():
        raise ValueError("source runner path must be exact EXP31 implementation")
    if args.orchestrator.resolve() != Path(__file__).resolve():
        raise ValueError("orchestrator self-binding mismatch")
    validate_output_topology(args.primary_root, args.replay_root, args.result)
    refuse_nonempty(args.primary_root)
    refuse_nonempty(args.replay_root)
    verify_parent_renderer_binding(args.source_runner_approval, args.parent_renderer)
    commands: list[dict[str, Any]] = []
    failure_stage = "PRIMARY_SOURCE"

    try:
        primary_run = _run_source(args, args.primary_root, "PRIMARY", commands)
        if primary_run.returncode != 0:
            raise RuntimeError(f"PRIMARY source child failed: {primary_run.stderr}")
        primary_result_path = args.primary_root / "SOURCE_RESULT.json"
        if not primary_result_path.is_file():
            raise RuntimeError("PRIMARY source result missing")
        primary_result = _read_json(primary_result_path)
        if primary_result.get("scientific_execute_source_reached") is not True:
            if primary_result.get("status") != "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN":
                raise RuntimeError("PRIMARY pre-science result has unexpected status")
            terminal = _terminal(
                status="ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN",
                primary=args.primary_root,
                replay=args.replay_root,
                commands=commands,
                failure_stage="PRIMARY_SOURCE_INTERFACE",
                failure_reason="pre-frozen source-interface actionability gate",
                full_tree_byte_exact=None,
            )
            _publish(args.result, terminal)
            return terminal
        if not (args.primary_root / "SOURCE_VISUAL_PAYLOAD.json").is_file():
            raise RuntimeError("PRIMARY parent visual payload missing")
        failure_stage = "PRIMARY_RENDERER"
        primary_renderer = _run_renderer(args, args.primary_root, "PRIMARY", commands)
        if primary_renderer.returncode != 0:
            raise RuntimeError(f"PRIMARY parent renderer failed: {primary_renderer.stderr}")
        if not (args.primary_root / "native/native_visual_manifest.json").is_file():
            raise RuntimeError("PRIMARY parent native manifest missing")
        _write_native_context(args.primary_root)

        failure_stage = "REPLAY_SOURCE"
        replay_run = _run_source(args, args.replay_root, "REPLAY", commands)
        if replay_run.returncode != 0:
            raise RuntimeError(f"REPLAY source child failed: {replay_run.stderr}")
        replay_result = _read_json(args.replay_root / "SOURCE_RESULT.json")
        if replay_result.get("scientific_execute_source_reached") is not True:
            raise RuntimeError("REPLAY diverged to source-interface nonactionability")
        failure_stage = "REPLAY_RENDERER"
        replay_renderer = _run_renderer(args, args.replay_root, "REPLAY", commands)
        if replay_renderer.returncode != 0:
            raise RuntimeError(f"REPLAY parent renderer failed: {replay_renderer.stderr}")
        if not (args.replay_root / "native/native_visual_manifest.json").is_file():
            raise RuntimeError("REPLAY parent native manifest missing")
        _write_native_context(args.replay_root)
        failure_stage = "TREE_COMPARE"
        first_tree = tree_hashes(args.primary_root)
        second_tree = tree_hashes(args.replay_root)
        exact = first_tree == second_tree
        if not exact:
            raise RuntimeError("primary/replay full-tree byte mismatch")
        terminal = _terminal(
            status="PASS_PRIMARY_REPLAY_BYTE_EXACT_AWAITING_NATIVE_REVIEW",
            primary=args.primary_root,
            replay=args.replay_root,
            commands=commands,
            full_tree_byte_exact=True,
        )
        terminal["tree_hashes"] = first_tree
        terminal["native_visual_review_status"] = "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW"
        _publish(args.result, terminal)
        return terminal
    except Exception as error:
        terminal = _terminal(
            status="FAIL_TERMINAL_NO_RETRY",
            primary=args.primary_root,
            replay=args.replay_root,
            commands=commands,
            failure_stage=failure_stage,
            failure_reason=str(error),
        )
        _publish(args.result, terminal)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    result = execute(argv)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
