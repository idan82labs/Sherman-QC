from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "orchestrate_exp_cad_0030_cfpdg_primary_replay.py"
SPEC = importlib.util.spec_from_file_location("exp30_orchestration", PATH)
ORCH = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ORCH)


def _write_source_approval(path: Path, renderer: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "EXP_CAD_0030_V3_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1",
                "experiment_id": "EXP_CAD_0030",
                "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
                "decision": "APPROVE_ONE_PHYSICAL_PRIMARY_REPLAY_EXECUTION",
                "physical_source_execution_authorized": True,
                "bound_files": {
                    "renderer": {"sha256": ORCH.CAD.sha256_path(renderer)}
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _argv(tmp_path: Path, paths: dict[str, Path]) -> list[str]:
    argv = [
        "--source-runner", str(ROOT / "scripts" / "run_exp_cad_0030_cfpdg_source.py"),
        "--primary-root", str(tmp_path / "primary"),
        "--replay-root", str(tmp_path / "replay"),
        "--result", str(tmp_path / "result.json"),
    ]
    for name in ORCH.FORWARDED_PATHS:
        argv.extend(("--" + name.replace("_", "-"), str(paths[name])))
    return argv


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ORCH.FORWARDED_PATHS:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    paths["renderer"] = ROOT / "scripts" / "render_exp_cad_0030_cfpdg_native.py"
    paths["orchestration"] = PATH
    _write_source_approval(paths["source_runner_approval"], paths["renderer"])
    return paths


def test_tree_hashes_are_relative_complete_and_detect_extras(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "native").mkdir(parents=True)
        (root / "SOURCE_RESULT.json").write_bytes(b"same\n")
        (root / "native" / "native_visual_manifest.json").write_bytes(b"manifest\n")
    assert ORCH.tree_hashes(first) == ORCH.tree_hashes(second)
    (second / "extra.bin").write_bytes(b"x")
    assert ORCH.tree_hashes(first) != ORCH.tree_hashes(second)


def test_orchestrator_uses_two_fresh_source_and_renderer_children(monkeypatch, tmp_path: Path) -> None:
    dummy_names = [
        name for name in ORCH.FORWARDED_PATHS
        if name not in {"renderer", "orchestration"}
    ]
    paths = {}
    for name in dummy_names:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    paths["renderer"] = ROOT / "scripts" / "render_exp_cad_0030_cfpdg_native.py"
    paths["orchestration"] = PATH
    _write_source_approval(paths["source_runner_approval"], paths["renderer"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if "--output-root" in command:
            root = Path(command[command.index("--output-root") + 1])
            root.mkdir(parents=True)
            (root / "SOURCE_RESULT.json").write_bytes(b"semantic\n")
            (root / "SOURCE_VISUAL_PAYLOAD.json").write_bytes(b"payload\n")
        else:
            native = Path(command[-1])
            native.mkdir(parents=True)
            (native / "native_visual_manifest.json").write_bytes(b"manifest\n")
            (native / "global_paired_diagnostic.png").write_bytes(b"png\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCH.subprocess, "run", fake_run)
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    result_path = tmp_path / "result.json"
    argv = _argv(tmp_path, paths)
    result = ORCH.execute(argv)
    assert len(calls) == 4
    assert sum("--output-root" in call for call in calls) == 2
    assert result["fresh_source_child_process_count"] == 2
    assert result["fresh_renderer_child_process_count"] == 2
    assert result["full_tree_byte_exact"] is True
    assert result["required_relative_paths_present"] is True
    assert result["primary_replay_physical_scan_hash_call_count"] == 4
    assert result["primary_replay_physical_scan_point_cloud_load_call_count"] == 4
    assert result["primary_replay_physical_scan_byte_open_operation_count"] == 8
    assert result_path.is_file()
    renderer_calls = [call for call in calls if "--output-root" not in call]
    assert all(Path(call[3]).resolve() == paths["renderer"].resolve() for call in renderer_calls)


def test_output_topology_rejects_equal_ancestor_and_result_inside(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    result = tmp_path / "result.json"
    ORCH.validate_output_topology(primary, replay, result)
    for first, second, terminal in (
        (primary, primary, result),
        (primary, primary / "nested", result),
        (primary / "nested", primary, result),
        (primary, replay, primary / "result.json"),
        (primary, replay, replay / "result.json"),
    ):
        with pytest.raises(ValueError):
            ORCH.validate_output_topology(first, second, terminal)


def test_renderer_binding_is_exact_not_basename_or_approval_substring(tmp_path: Path) -> None:
    renderer = ROOT / "scripts" / "render_exp_cad_0030_cfpdg_native.py"
    approval = tmp_path / "approval.json"
    _write_source_approval(approval, renderer)
    ORCH.verify_renderer_binding(approval, renderer)
    copy = tmp_path / renderer.name
    copy.write_bytes(renderer.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="renderer hash mismatch"):
        ORCH.verify_renderer_binding(approval, copy)
    payload = json.loads(approval.read_text(encoding="utf-8"))
    payload["decision"] = "NOT_APPROVED"
    approval.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision unavailable"):
        ORCH.verify_renderer_binding(approval, renderer)


def test_source_failure_atomically_persists_terminal_fail(monkeypatch, tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    def failed_run(command, **kwargs):
        return SimpleNamespace(returncode=19, stdout="", stderr="fixture failure")

    monkeypatch.setattr(ORCH.subprocess, "run", failed_run)
    argv = _argv(tmp_path, paths)
    with pytest.raises(RuntimeError, match="source child failed"):
        ORCH.execute(argv)
    terminal = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_source_child_process_count"] == 1
    assert terminal["fresh_renderer_child_process_count"] == 0


def test_source_timeout_atomically_persists_terminal_fail(monkeypatch, tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    def timeout_source(command, **kwargs):
        raise ORCH.subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial-source", stderr=b"source-timeout"
        )

    monkeypatch.setattr(ORCH.subprocess, "run", timeout_source)
    with pytest.raises(RuntimeError, match="source child timeout"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["commands"][0]["source_timeout"] is True
    assert terminal["commands"][0]["source_stderr"] == "source-timeout"


def test_renderer_timeout_atomically_persists_terminal_fail(monkeypatch, tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    def timeout_renderer(command, **kwargs):
        if "--output-root" in command:
            root = Path(command[command.index("--output-root") + 1])
            root.mkdir(parents=True)
            (root / "SOURCE_RESULT.json").write_bytes(b"semantic\n")
            (root / "SOURCE_VISUAL_PAYLOAD.json").write_bytes(b"payload\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise ORCH.subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial", stderr=b"timed out"
        )

    monkeypatch.setattr(ORCH.subprocess, "run", timeout_renderer)
    with pytest.raises(RuntimeError, match="renderer child timeout"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert terminal["commands"][0]["renderer_timeout"] is True
    assert terminal["commands"][0]["renderer_stdout"] == "partial"


def test_renderer_failure_atomically_persists_terminal_fail(monkeypatch, tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)

    def failed_renderer(command, **kwargs):
        if "--output-root" in command:
            root = Path(command[command.index("--output-root") + 1])
            root.mkdir(parents=True)
            (root / "SOURCE_RESULT.json").write_bytes(b"semantic\n")
            (root / "SOURCE_VISUAL_PAYLOAD.json").write_bytes(b"payload\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=23, stdout="", stderr="renderer failure")

    monkeypatch.setattr(ORCH.subprocess, "run", failed_renderer)
    with pytest.raises(RuntimeError, match="renderer child failed"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert terminal["commands"][0]["renderer_returncode"] == 23


def test_tree_mismatch_atomically_persists_terminal_fail(monkeypatch, tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    source_count = 0

    def mismatched_tree(command, **kwargs):
        nonlocal source_count
        if "--output-root" in command:
            root = Path(command[command.index("--output-root") + 1])
            root.mkdir(parents=True)
            source_count += 1
            (root / "SOURCE_RESULT.json").write_bytes(
                b"primary\n" if source_count == 1 else b"replay\n"
            )
            (root / "SOURCE_VISUAL_PAYLOAD.json").write_bytes(b"payload\n")
        else:
            native = Path(command[-1])
            native.mkdir(parents=True)
            (native / "native_visual_manifest.json").write_bytes(b"manifest\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCH.subprocess, "run", mismatched_tree)
    with pytest.raises(RuntimeError, match="full-tree replay mismatch"):
        ORCH.execute(_argv(tmp_path, paths))
    terminal = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["failure_stage"] == "FULL_TREE_COMPARE"
    assert terminal["fresh_source_child_process_count"] == 2
    assert terminal["fresh_renderer_child_process_count"] == 2
