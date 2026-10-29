from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/orchestrate_exp_cad_0031_cfpdg_primary_replay.py"
SPEC = importlib.util.spec_from_file_location("exp31_orchestrator", PATH)
ORCH = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ORCH)


def _write_source_approval(path: Path, renderer: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": ORCH.SOURCE_APPROVAL_SCHEMA,
                "experiment_id": ORCH.EXPERIMENT_ID,
                "architecture": ORCH.ARCHITECTURE,
                "decision": "APPROVE_ONE_EXP31_PHYSICAL_PRIMARY_REPLAY_EXECUTION",
                "physical_source_execution_authorized": True,
                "bound_files": {
                    "parent_renderer": {
                        "sha256": hashlib.sha256(renderer.read_bytes()).hexdigest()
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ORCH.FORWARDED_PATHS:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        paths[name] = path
    paths["parent_renderer"] = ROOT / "scripts/render_exp_cad_0030_cfpdg_native.py"
    paths["orchestrator"] = PATH
    _write_source_approval(paths["source_runner_approval"], paths["parent_renderer"])
    return paths


def _argv(tmp_path: Path, paths: dict[str, Path]) -> list[str]:
    argv = [
        "--source-runner", str(ROOT / "scripts/run_exp_cad_0031_cfpdg_source.py"),
        "--primary-root", str(tmp_path / "primary"),
        "--replay-root", str(tmp_path / "replay"),
        "--result", str(tmp_path / "result.json"),
    ]
    for name in ORCH.FORWARDED_PATHS:
        argv.extend(("--" + name.replace("_", "-"), str(paths[name])))
    return argv


def _source_tree(root: Path, *, reached: bool, marker: bytes = b"same") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SOURCE_RESULT.json").write_text(
        json.dumps(
            {
                "schema": "EXP_CAD_0031_SOURCE_INTERFACE_EXECUTION_ENVELOPE_V1",
                "status": (
                    "PAIRWISE_DEVIATION_ATTRIBUTED"
                    if reached
                    else "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
                ),
                "scientific_execute_source_reached": reached,
                "physical_scan_hash_call_count": 2,
                "physical_scan_point_cloud_load_call_count": 2,
                "physical_scan_byte_open_operation_count": 4,
                "marker": marker.decode(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "EXP31_EXECUTION_CONTEXT.json").write_bytes(marker)
    (root / "FIRST_RETAINED_ORIGINAL_ROW_IDS.npy").write_bytes(marker)
    (root / "SECOND_RETAINED_ORIGINAL_ROW_IDS.npy").write_bytes(marker)
    if reached:
        (root / "SOURCE_VISUAL_PAYLOAD.json").write_bytes(marker)


def test_success_runs_two_sources_two_parent_renderers_and_compares_full_tree(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if "--output-root" in command:
            root = Path(command[command.index("--output-root") + 1])
            _source_tree(root, reached=True)
        else:
            native = Path(command[-1])
            native.mkdir(parents=True)
            (native / "native_visual_manifest.json").write_bytes(b"manifest")
            (native / "global_paired_diagnostic.png").write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    result = ORCH.execute(_argv(tmp_path, paths))
    assert len(calls) == 4
    assert result["status"] == "PASS_PRIMARY_REPLAY_BYTE_EXACT_AWAITING_NATIVE_REVIEW"
    assert result["fresh_primary_source_child_count"] == 1
    assert result["fresh_replay_source_child_count"] == 1
    assert result["fresh_primary_renderer_child_count"] == 1
    assert result["fresh_replay_renderer_child_count"] == 1
    assert result["physical_scan_hash_call_count"] == 4
    assert result["physical_scan_point_cloud_load_call_count"] == 4
    assert result["physical_scan_byte_open_operation_count"] == 8
    assert result["full_tree_byte_exact"] is True
    primary_context = json.loads(
        (tmp_path / "primary/native/EXP31_NATIVE_CONTEXT.json").read_text()
    )
    assert primary_context["context"] == (
        "EXP31 intake successor / unchanged EXP30 scientific core and renderer"
    )
    assert primary_context["silent_parent_relabeling_performed"] is False


def test_primary_interface_nonactionability_stops_without_replay_or_renderer(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        root = Path(command[command.index("--output-root") + 1])
        _source_tree(root, reached=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    result = ORCH.execute(_argv(tmp_path, paths))
    assert len(calls) == 1
    assert result["status"] == "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
    assert result["fresh_primary_source_child_count"] == 1
    assert result["fresh_replay_source_child_count"] == 0
    assert result["fresh_primary_renderer_child_count"] == 0
    assert result["fresh_replay_renderer_child_count"] == 0
    assert result["physical_scan_hash_call_count"] == 2
    assert result["physical_scan_point_cloud_load_call_count"] == 2
    assert result["physical_scan_byte_open_operation_count"] == 4
    assert not (tmp_path / "replay").exists()


def test_approval_rejection_occurs_before_any_source_child_or_output(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    approval = json.loads(paths["source_runner_approval"].read_text())
    approval["physical_source_execution_authorized"] = False
    paths["source_runner_approval"].write_text(json.dumps(approval), encoding="utf-8")

    def forbidden_run(*args, **kwargs):
        raise AssertionError("approval rejection must precede subprocess execution")

    monkeypatch.setattr(ORCH.subprocess, "run", forbidden_run)
    with pytest.raises(ValueError, match="physical authority"):
        ORCH.execute(_argv(tmp_path, paths))
    assert not (tmp_path / "primary").exists()
    assert not (tmp_path / "replay").exists()
    assert not (tmp_path / "result.json").exists()


def test_source_failure_publishes_terminal_with_child_accounting(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)

    def fake_run(command, **kwargs):
        root = Path(command[command.index("--output-root") + 1])
        root.mkdir(parents=True)
        (root / "SOURCE_CHILD_FAILURE_ACCOUNTING.json").write_text(
            json.dumps(
                {
                    "physical_scan_hash_call_count": 2,
                    "physical_scan_point_cloud_load_call_count": 2,
                    "physical_scan_byte_open_operation_count": 4,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="source failed")

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="PRIMARY source child failed"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text())
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["physical_scan_hash_call_count"] == 2
    assert terminal["physical_scan_point_cloud_load_call_count"] == 2
    assert terminal["physical_scan_byte_open_operation_count"] == 4


def test_source_timeout_counts_spawned_child_and_reads_durable_journal(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)

    def fake_run(command, **kwargs):
        root = Path(command[command.index("--output-root") + 1])
        root.mkdir(parents=True)
        (root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json").write_text(
            json.dumps(
                {
                    "physical_scan_hash_call_count": 2,
                    "physical_scan_point_cloud_load_call_count": 2,
                    "physical_scan_byte_open_operation_count": 4,
                    "status": "IN_PROGRESS",
                    "stage": "SOURCE_INTERFACE_AND_PARENT_SCIENCE",
                }
            ),
            encoding="utf-8",
        )
        raise ORCH.subprocess.TimeoutExpired(command, 300)

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="source child timed out"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text())
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_primary_source_child_count"] == 1
    assert terminal["physical_scan_hash_call_count"] == 2
    assert terminal["physical_scan_point_cloud_load_call_count"] == 2
    assert terminal["physical_scan_byte_open_operation_count"] == 4


def test_tree_mismatch_is_terminal_and_output_topology_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    source_count = 0

    def fake_run(command, **kwargs):
        nonlocal source_count
        if "--output-root" in command:
            root = Path(command[command.index("--output-root") + 1])
            source_count += 1
            _source_tree(root, reached=True, marker=b"same" if source_count == 1 else b"diff")
        else:
            native = Path(command[-1])
            native.mkdir(parents=True)
            (native / "native_visual_manifest.json").write_bytes(b"manifest")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="full-tree byte mismatch"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text())
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == "TREE_COMPARE"

    with pytest.raises(ValueError):
        ORCH.validate_output_topology(
            tmp_path / "same", tmp_path / "same", tmp_path / "result2.json"
        )
    with pytest.raises(ValueError):
        ORCH.validate_output_topology(
            tmp_path / "parent", tmp_path / "parent/child", tmp_path / "result3.json"
        )


@pytest.mark.parametrize(
    "failure_target,expected_stage",
    [
        ("PRIMARY_RENDERER", "PRIMARY_RENDERER"),
        ("REPLAY_SOURCE", "REPLAY_SOURCE"),
        ("REPLAY_RENDERER", "REPLAY_RENDERER"),
    ],
)
def test_each_post_primary_failure_persists_precise_stage(
    monkeypatch, tmp_path: Path, failure_target: str, expected_stage: str
) -> None:
    paths = _paths(tmp_path)
    source_count = 0
    renderer_count = 0

    def fake_run(command, **kwargs):
        nonlocal source_count, renderer_count
        if "--output-root" in command:
            source_count += 1
            root = Path(command[command.index("--output-root") + 1])
            if failure_target == "REPLAY_SOURCE" and source_count == 2:
                root.mkdir(parents=True)
                (root / "SOURCE_CHILD_FAILURE_ACCOUNTING.json").write_text(
                    json.dumps(
                        {
                            "physical_scan_hash_call_count": 2,
                            "physical_scan_point_cloud_load_call_count": 2,
                            "physical_scan_byte_open_operation_count": 4,
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=1, stdout="", stderr="replay failed")
            _source_tree(root, reached=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        renderer_count += 1
        native = Path(command[-1])
        native.mkdir(parents=True)
        (native / "native_visual_manifest.json").write_bytes(b"manifest")
        should_fail = (
            failure_target == "PRIMARY_RENDERER" and renderer_count == 1
        ) or (
            failure_target == "REPLAY_RENDERER" and renderer_count == 2
        )
        return SimpleNamespace(
            returncode=1 if should_fail else 0,
            stdout="",
            stderr="renderer failed" if should_fail else "",
        )

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text())
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == expected_stage
