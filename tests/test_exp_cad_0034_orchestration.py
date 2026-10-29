from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/orchestrate_exp_cad_0034_cfpdg_v5_physical.py"
SPEC = importlib.util.spec_from_file_location("exp34_orchestrator", PATH)
ORCH = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ORCH)


def test_module_has_no_top_level_import_of_lock_bound_governance_or_monitor() -> None:
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "scripts.exp_cad_0034_execution_contract" not in imported
    assert "scripts.exp_cad_0034_process_monitor" not in imported


def test_stdlib_bootstrap_rejects_lock_hash_before_bound_import(
    monkeypatch, tmp_path: Path
) -> None:
    fake_root = tmp_path / "repo"
    lock = fake_root / ORCH.LOCK_RELATIVE_PATH
    lock.parent.mkdir(parents=True)
    lock.write_text("{}", encoding="utf-8")
    fake_orchestrator = fake_root / "scripts/orchestrate_exp_cad_0034_cfpdg_v5_physical.py"
    fake_monitor = fake_root / "scripts/exp_cad_0034_process_monitor.py"
    fake_orchestrator.parent.mkdir(parents=True)
    fake_orchestrator.write_bytes(b"orchestrator")
    fake_monitor.write_bytes(b"monitor")
    monkeypatch.setattr(ORCH, "ROOT", fake_root)
    monkeypatch.setattr(ORCH, "__file__", str(fake_orchestrator))
    monkeypatch.setattr(ORCH, "_SOURCE_FREE_TEST_INITIALIZED", False)
    monkeypatch.setattr(ORCH, "_BOUND_RUNTIME_INITIALIZED", False)
    monkeypatch.setattr(
        ORCH.importlib,
        "import_module",
        lambda name: pytest.fail(f"bound import preceded lock verification: {name}"),
    )
    args = SimpleNamespace(
        lock=lock.resolve(),
        lock_sha256="0" * 64,
        orchestrator=fake_orchestrator.resolve(),
        process_monitor=fake_monitor.resolve(),
    )
    with pytest.raises(ValueError, match="bootstrap lock digest mismatch"):
        ORCH._initialize_bound_runtime(args)


def _write_json(path: Path, payload: Any) -> None:
    ORCH.canonical_json_write(path, payload)


def _minimal_png(marker: bytes = b"") -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + ORCH.struct.pack(">II", ORCH.NATIVE_WIDTH, ORCH.NATIVE_HEIGHT)
        + marker
    )


def _write_renderer_tree(root: Path, *, marker: bytes = b"") -> None:
    root.mkdir(parents=True, exist_ok=True)
    asset = root / "global_paired_diagnostic.png"
    asset.write_bytes(_minimal_png(marker))
    _write_json(
        root / "native_visual_manifest.json",
        {
            "schema": ORCH.NATIVE_MANIFEST_SCHEMA,
            "experiment_id": "EXP_CAD_0030",
            "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
            "native_resolution": {
                "width_px": ORCH.NATIVE_WIDTH,
                "height_px": ORCH.NATIVE_HEIGHT,
            },
            "global_asset": {
                "path": asset.name,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "width_px": ORCH.NATIVE_WIDTH,
                "height_px": ORCH.NATIVE_HEIGHT,
            },
            "component_assets": [],
            "node_assets": [],
        },
    )


def _write_journal(root: Path, *, counts: tuple[int, int, int] = (2, 2, 4)) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json",
        {
            "schema": "EXP_CAD_0034_SOURCE_SCAN_ACCOUNTING_JOURNAL_V1",
            "physical_scan_hash_call_count": counts[0],
            "physical_scan_point_cloud_load_call_count": counts[1],
            "physical_scan_byte_open_operation_count": counts[2],
        },
    )


def _write_source_tree(
    root: Path,
    measurement: Path,
    *,
    science: bool,
    marker: str = "same",
    accounting: tuple[int, int, int] = (2, 2, 4),
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    status = "PAIRWISE_DEVIATION_ATTRIBUTED" if science else ORCH.INTERFACE_STATUS
    _write_json(
        root / "SOURCE_RESULT.json",
        {
            "schema": ORCH.SOURCE_SCHEMA,
            "experiment_id": ORCH.EXPERIMENT_ID,
            "architecture": ORCH.ARCHITECTURE,
            "status": status,
            "scientific_execute_source_reached": science,
            "nested_semantic_sha256": "a" * 64 if science else None,
            "nested_visual_sha256": "b" * 64 if science else None,
            "marker": marker,
        },
    )
    for filename in (
        "FIRST_SOURCE_INTERFACE_REPORT.json",
        "SECOND_SOURCE_INTERFACE_REPORT.json",
        "PAIR_ACTIONABILITY.json",
        "RETAINED_IDENTITY_HASHES.json",
    ):
        _write_json(root / filename, {"marker": marker, "file": filename})
    if science:
        _write_json(
            root / "ENGINE_INVOCATION_ACCOUNTING.json",
            {
                "schema": "EXP_CAD_0034_ENGINE_INVOCATION_ACCOUNTING_V1",
                "v5_engine_invocation_count": 1,
                "unique_member_fit_count": 38,
                "kd_node_query_count": 306926,
                "local_fit_call_count": 16154,
                "robustness_row_count": 32,
                "exp32_engine_metrics_payload_present": True,
            },
        )
        _write_json(root / "SOURCE_SEMANTIC_PAYLOAD.json", {"marker": marker})
        _write_json(root / "SOURCE_VISUAL_PAYLOAD.json", {"marker": marker})
        _write_json(
            root / "EXP32_ENGINE_COUNTER_PROJECTION.json", {"marker": marker}
        )
        _write_json(
            measurement / "EXP32_ENGINE_METRICS.json",
            {
                "schema": "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1",
                "experiment_id": "EXP_CAD_0032",
            },
        )
    else:
        _write_json(
            root / "ENGINE_INVOCATION_ACCOUNTING.json",
            {
                "schema": "EXP_CAD_0034_ENGINE_INVOCATION_ACCOUNTING_V1",
                "v5_engine_invocation_count": 0,
                "unique_member_fit_count": 0,
                "kd_node_query_count": 0,
                "local_fit_call_count": 0,
                "robustness_row_count": 0,
                "exp32_engine_metrics_payload_present": False,
            },
        )
    nested_files = {
        "first_source_interface_report_sha256": "FIRST_SOURCE_INTERFACE_REPORT.json",
        "second_source_interface_report_sha256": "SECOND_SOURCE_INTERFACE_REPORT.json",
        "pair_actionability_sha256": "PAIR_ACTIONABILITY.json",
        "retained_identity_hashes_sha256": "RETAINED_IDENTITY_HASHES.json",
        "engine_invocation_accounting_sha256": "ENGINE_INVOCATION_ACCOUNTING.json",
    }
    if science:
        nested_files.update(
            {
                "scientific_semantic_payload_sha256": "SOURCE_SEMANTIC_PAYLOAD.json",
                "scientific_visual_payload_sha256": "SOURCE_VISUAL_PAYLOAD.json",
                "exp32_engine_counter_projection_sha256": "EXP32_ENGINE_COUNTER_PROJECTION.json",
            }
        )
    result_path = root / "SOURCE_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["nested_payload_sha256"] = {
        key: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        for key, filename in nested_files.items()
    }
    _write_json(result_path, result)
    _write_journal(measurement, counts=accounting)


def _pass_measurement(*, wall_limit: float) -> dict[str, Any]:
    return {
        "status": "PASS",
        "returncode": 0,
        "timed_out": False,
        "rss_limit_exceeded": False,
        "sampler_failed": False,
        "resource_kill_reason": None,
        "wall_seconds": min(0.25, wall_limit),
        "wall_limit_seconds": wall_limit,
        "peak_process_tree_rss_bytes": 1024,
        "rss_limit_bytes": ORCH.RSS_LIMIT_BYTES,
        "process_tree_sample_count": 1,
    }


class FakeMonitor:
    def __init__(
        self,
        *,
        source_modes: dict[str, bool] | None = None,
        source_markers: dict[str, str] | None = None,
        renderer_markers: dict[str, bytes] | None = None,
        fail_at: tuple[str, str] | None = None,
        fail_kind: str = "returncode",
        source_accounting: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.source_modes = source_modes or {"PRIMARY": True, "REPLAY": True}
        self.source_markers = source_markers or {}
        self.renderer_markers = renderer_markers or {}
        self.fail_at = fail_at
        self.fail_kind = fail_kind
        self.source_accounting = source_accounting or {}
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        command,
        *,
        measurement_root: Path,
        wall_limit_seconds: float,
        rss_limit_bytes: int,
    ) -> dict[str, Any]:
        command = list(command)
        is_source = "--mode" in command
        if is_source:
            label = command[command.index("--mode") + 1]
            kind = "SOURCE"
        else:
            label = "PRIMARY" if "primary" in str(measurement_root) else "REPLAY"
            kind = "RENDERER"
        self.calls.append(
            {
                "label": label,
                "kind": kind,
                "command": command,
                "measurement_root": Path(measurement_root),
                "wall_limit_seconds": wall_limit_seconds,
                "rss_limit_bytes": rss_limit_bytes,
            }
        )
        if is_source:
            output = Path(command[command.index("--output-root") + 1])
            measurement = Path(command[command.index("--measurement-root") + 1])
            capability_path = Path(
                command[command.index("--source-launch-capability") + 1]
            )
            child_claim_path = Path(
                command[command.index("--source-launch-child-claim") + 1]
            )
            launch_claim_path = Path(command[command.index("--launch-claim") + 1])
            launch_claim_sha = command[command.index("--launch-claim-sha256") + 1]
            capability = json.loads(capability_path.read_text(encoding="utf-8"))
            assert not child_claim_path.exists()
            _write_json(
                child_claim_path,
                {
                    "schema": ORCH.SOURCE_CHILD_CLAIM_SCHEMA,
                    "experiment_id": ORCH.EXPERIMENT_ID,
                    "status": "CLAIMED_BEFORE_STATIC_OR_SCAN_ACCESS",
                    "source_child_ordinal": 1 if label == "PRIMARY" else 2,
                    "mode": label,
                    "source_launch_capability_path": str(capability_path),
                    "source_launch_capability_sha256": hashlib.sha256(
                        capability_path.read_bytes()
                    ).hexdigest(),
                    "command_sha256": capability["command_sha256"],
                    "launch_claim_path": str(launch_claim_path),
                    "launch_claim_sha256": launch_claim_sha,
                },
            )
            _write_source_tree(
                output,
                measurement,
                science=self.source_modes[label],
                marker=self.source_markers.get(label, "same"),
                accounting=self.source_accounting.get(label, (2, 2, 4)),
            )
            assert command[command.index("--scan-accounting-journal") + 1] == str(
                measurement / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
            )
            assert Path(measurement_root) == measurement / "source_process"
        else:
            renderer_root = Path(command[-1])
            _write_renderer_tree(
                renderer_root, marker=self.renderer_markers.get(label, b"")
            )
            expected_parent = renderer_root.parent / "measurement/renderer_process"
            assert Path(measurement_root) == expected_parent
        result = _pass_measurement(wall_limit=wall_limit_seconds)
        if self.fail_at == (label, kind):
            if self.fail_kind == "exception":
                raise RuntimeError("fake monitor exception")
            if self.fail_kind == "timeout":
                result.update(
                    status="FAIL_TERMINAL",
                    returncode=-15,
                    timed_out=True,
                    resource_kill_reason="WALL_LIMIT",
                    wall_seconds=wall_limit_seconds + 0.001,
                )
            elif self.fail_kind == "rss":
                result.update(
                    status="FAIL_TERMINAL",
                    returncode=-15,
                    rss_limit_exceeded=True,
                    resource_kill_reason="RSS_LIMIT",
                    peak_process_tree_rss_bytes=ORCH.RSS_LIMIT_BYTES + 1,
                )
            else:
                result.update(status="FAIL_TERMINAL", returncode=1)
        return result


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths: dict[str, Path] = {}
    for name in ORCH.PATH_ARGUMENTS:
        if name in {
            "attempt_root",
            "primary_source_root",
            "replay_source_root",
            "primary_measurement_root",
            "replay_measurement_root",
            "primary_renderer_root",
            "replay_renderer_root",
            "launch_claim",
            "result",
        }:
            continue
        path = inputs / f"{name}.json"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path.resolve()
    paths["orchestrator"] = PATH.resolve()
    paths["process_monitor"] = (
        ROOT / "scripts/exp_cad_0034_process_monitor.py"
    ).resolve()
    attempt = (tmp_path / "attempt").resolve()
    paths.update(
        {
            "attempt_root": attempt,
            "primary_source_root": attempt / "primary/source",
            "replay_source_root": attempt / "replay/source",
            "primary_measurement_root": attempt / "primary/measurement",
            "replay_measurement_root": attempt / "replay/measurement",
            "primary_renderer_root": attempt / "primary/renderer",
            "replay_renderer_root": attempt / "replay/renderer",
            "launch_claim": attempt / "EXP_CAD_0034_LAUNCH.claim.json",
            "result": attempt / "EXP_CAD_0034_ORCHESTRATION_RESULT.json",
        }
    )
    return paths


def _argv(paths: dict[str, Path]) -> list[str]:
    result: list[str] = []
    for name in ORCH.PATH_ARGUMENTS:
        result.extend(("--" + name.replace("_", "-"), str(paths[name])))
    result.extend(
        ("--lock-sha256", hashlib.sha256(paths["lock"].read_bytes()).hexdigest())
    )
    result.extend(
        (
            "--detached-approval-sha256",
            hashlib.sha256(paths["detached_approval"].read_bytes()).hexdigest(),
        )
    )
    return result


def _authorize(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_verify(**kwargs):
        calls.append(kwargs)
        assert not kwargs["output_paths"]["launch_claim"].exists()
        bound_paths = dict(kwargs["consumed_paths"])
        bound_paths["exp34_contract"] = (
            ROOT / "scripts/exp_cad_0034_execution_contract.py"
        ).resolve()
        return {
            "authorized_outputs": {
                key: f"output/fake/{key}" for key in kwargs["output_paths"]
            },
            "bound_files": {
                key: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for key, path in bound_paths.items()
                if key
                in {
                    "exp34_source_runner",
                    "exp34_monitor",
                    "exp34_orchestrator",
                    "exp34_contract",
                    "renderer",
                }
            },
        }

    ORCH.initialize_source_free_test_dependencies(
        authority_verifier=fake_verify,
        measured_runner=ORCH._uninitialized_dependency,
    )
    return calls


def _run(
    monkeypatch,
    tmp_path: Path,
    monitor: FakeMonitor,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    paths = _fixture_paths(tmp_path)
    authority = _authorize(monkeypatch)
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    result = ORCH.execute(_argv(paths))
    return result, paths, authority


def test_interface_abstention_runs_two_fresh_sources_zero_renderers(
    monkeypatch, tmp_path: Path
) -> None:
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    result, paths, authority = _run(monkeypatch, tmp_path, monitor)
    assert len(authority) == 1
    assert [(row["label"], row["kind"]) for row in monitor.calls] == [
        ("PRIMARY", "SOURCE"),
        ("REPLAY", "SOURCE"),
    ]
    assert result["status"] == ORCH.INTERFACE_TERMINAL_STATUS
    assert result["fresh_primary_source_child_count"] == 1
    assert result["fresh_replay_source_child_count"] == 1
    assert result["fresh_primary_renderer_child_count"] == 0
    assert result["fresh_replay_renderer_child_count"] == 0
    assert result["physical_scan_hash_call_count"] == 4
    assert result["physical_scan_point_cloud_load_call_count"] == 4
    assert result["physical_scan_byte_open_operation_count"] == 8
    assert result["source_tree_byte_exact"] is True
    assert result["renderer_tree_byte_exact"] is None
    assert result["native_visual_review_status"].startswith("NOT_APPLICABLE")
    assert paths["launch_claim"].is_file()
    assert paths["result"].is_file()


def test_science_runs_exact_source_renderer_source_renderer_schedule(
    monkeypatch, tmp_path: Path
) -> None:
    monitor = FakeMonitor()
    result, _, _ = _run(monkeypatch, tmp_path, monitor)
    assert [(row["label"], row["kind"]) for row in monitor.calls] == [
        ("PRIMARY", "SOURCE"),
        ("PRIMARY", "RENDERER"),
        ("REPLAY", "SOURCE"),
        ("REPLAY", "RENDERER"),
    ]
    assert result["status"] == ORCH.SCIENCE_TERMINAL_STATUS
    assert result["source_tree_byte_exact"] is True
    assert result["renderer_tree_byte_exact"] is True
    assert result["fresh_primary_source_child_count"] == 1
    assert result["fresh_replay_source_child_count"] == 1
    assert result["fresh_primary_renderer_child_count"] == 1
    assert result["fresh_replay_renderer_child_count"] == 1
    assert result["native_visual_review_status"] == (
        "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW"
    )


def test_authority_rejection_precedes_claim_and_every_child(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    monitor = FakeMonitor()

    def reject(**kwargs):
        assert not paths["launch_claim"].exists()
        raise ValueError("approval rejected")

    ORCH.initialize_source_free_test_dependencies(
        authority_verifier=reject,
        measured_runner=ORCH._uninitialized_dependency,
    )
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ValueError, match="approval rejected"):
        ORCH.execute(_argv(paths))
    assert monitor.calls == []
    assert not paths["launch_claim"].exists()
    assert not paths["result"].exists()


def test_atomic_claim_cannot_be_overwritten_after_consumption(tmp_path: Path) -> None:
    claim = tmp_path / "EXP_CAD_0034_LAUNCH.claim.json"
    ORCH.create_launch_claim(claim, {"ordinal": 1})
    first = claim.read_bytes()
    with pytest.raises(FileExistsError):
        ORCH.create_launch_claim(claim, {"ordinal": 2})
    assert claim.read_bytes() == first


def test_existing_claim_blocks_before_authority_or_child(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    paths["launch_claim"].parent.mkdir(parents=True)
    paths["launch_claim"].write_text("consumed", encoding="utf-8")
    ORCH.initialize_source_free_test_dependencies(
        authority_verifier=lambda **kwargs: pytest.fail(
            "authority must not be re-entered"
        ),
        measured_runner=lambda *args, **kwargs: pytest.fail("child must not launch"),
    )
    monkeypatch.setattr(
        ORCH,
        "run_measured_command",
        lambda *args, **kwargs: pytest.fail("child must not launch"),
    )
    with pytest.raises(ValueError, match="launch claim already exists"):
        ORCH.execute(_argv(paths))


def test_preexisting_exact_empty_leaf_roots_are_permitted(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    for key in (
        "primary_source_root",
        "replay_source_root",
        "primary_measurement_root",
        "replay_measurement_root",
        "primary_renderer_root",
        "replay_renderer_root",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    authority = _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    result = ORCH.execute(_argv(paths))
    assert len(authority) == 1
    assert result["status"] == ORCH.INTERFACE_TERMINAL_STATUS


@pytest.mark.parametrize(
    "failure_at,expected_calls,expected_stage",
    [
        (("PRIMARY", "SOURCE"), 1, "PRIMARY_SOURCE"),
        (("PRIMARY", "RENDERER"), 2, "PRIMARY_RENDERER"),
        (("REPLAY", "SOURCE"), 3, "REPLAY_SOURCE"),
        (("REPLAY", "RENDERER"), 4, "REPLAY_RENDERER"),
    ],
)
def test_each_child_failure_is_terminal_without_retry_or_third_source(
    monkeypatch,
    tmp_path: Path,
    failure_at: tuple[str, str],
    expected_calls: int,
    expected_stage: str,
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(fail_at=failure_at)
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    assert len(monitor.calls) == expected_calls
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["status"] == ORCH.FAILURE_STATUS
    assert terminal["failure_stage"] == expected_stage
    assert terminal["fresh_primary_source_child_count"] <= 1
    assert terminal["fresh_replay_source_child_count"] <= 1


@pytest.mark.parametrize("failure_kind", ["timeout", "rss", "exception"])
def test_external_monitor_failure_modes_publish_terminal(
    monkeypatch, tmp_path: Path, failure_kind: str
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(
        fail_at=("PRIMARY", "SOURCE"), fail_kind=failure_kind
    )
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["status"] == ORCH.FAILURE_STATUS
    assert terminal["fresh_primary_source_child_count"] == 1
    assert terminal["fresh_replay_source_child_count"] == 0


def test_primary_renderer_failure_launches_no_replay_children(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(fail_at=("PRIMARY", "RENDERER"), fail_kind="timeout")
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["fresh_replay_source_child_count"] == 0
    assert terminal["fresh_replay_renderer_child_count"] == 0


def test_interface_replay_branch_divergence_is_terminal_and_never_renders(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": True})
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    assert [row["kind"] for row in monitor.calls] == ["SOURCE", "SOURCE"]
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "REPLAY_SOURCE"


def test_science_replay_branch_divergence_is_terminal_before_replay_renderer(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": True, "REPLAY": False})
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    assert [(row["label"], row["kind"]) for row in monitor.calls] == [
        ("PRIMARY", "SOURCE"),
        ("PRIMARY", "RENDERER"),
        ("REPLAY", "SOURCE"),
    ]


def test_one_byte_source_mutation_is_detected_as_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(
        source_modes={"PRIMARY": False, "REPLAY": False},
        source_markers={"PRIMARY": "same", "REPLAY": "diff"},
    )
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "SOURCE_TREE_COMPARE"
    assert len(monitor.calls) == 2


def test_one_byte_renderer_mutation_is_detected_as_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(
        renderer_markers={"PRIMARY": b"a", "REPLAY": b"b"}
    )
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "RENDERER_TREE_COMPARE"
    assert len(monitor.calls) == 4


def test_native_manifest_hash_failure_is_terminal_before_replay(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def tampering_monitor(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" not in command and "primary" in str(kwargs["measurement_root"]):
            Path(command[-1], "global_paired_diagnostic.png").write_bytes(
                _minimal_png(b"tampered")
            )
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", tampering_monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert len(monitor.calls) == 2


def test_successful_source_requires_exact_durable_2_2_4_accounting(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_accounting={"PRIMARY": (2, 1, 3)})
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["primary_scan_accounting"] == {
        "physical_scan_hash_call_count": 2,
        "physical_scan_point_cloud_load_call_count": 1,
        "physical_scan_byte_open_operation_count": 3,
    }


def test_durable_journal_precedes_source_result_failure_metadata(
    tmp_path: Path,
) -> None:
    measurement = tmp_path / "measurement"
    _write_journal(measurement, counts=(2, 0, 2))
    source = tmp_path / "source"
    source.mkdir()
    _write_json(
        source / "SOURCE_RESULT.json",
        {
            "physical_scan_hash_call_count": 99,
            "physical_scan_point_cloud_load_call_count": 99,
            "physical_scan_byte_open_operation_count": 99,
        },
    )
    observed, error = ORCH._safe_accounting(measurement)
    assert error is None
    assert observed == {
        "physical_scan_hash_call_count": 2,
        "physical_scan_point_cloud_load_call_count": 0,
        "physical_scan_byte_open_operation_count": 2,
    }


@pytest.mark.parametrize(
    "mutation",
    ["same", "ancestor", "outside", "preexisting_file", "unauthorized_dir"],
)
def test_output_topology_and_namespace_fail_closed(
    monkeypatch, tmp_path: Path, mutation: str
) -> None:
    paths = _fixture_paths(tmp_path)
    if mutation == "same":
        paths["replay_source_root"] = paths["primary_source_root"]
    elif mutation == "ancestor":
        paths["replay_source_root"] = paths["primary_source_root"] / "child"
    elif mutation == "outside":
        paths["primary_source_root"] = (tmp_path / "outside").resolve()
    elif mutation == "preexisting_file":
        paths["primary_source_root"].mkdir(parents=True)
        (paths["primary_source_root"] / "stale").write_bytes(b"x")
    else:
        (paths["attempt_root"] / "rogue").mkdir(parents=True)
    ORCH.initialize_source_free_test_dependencies(
        authority_verifier=lambda **kwargs: pytest.fail(
            "topology must fail before authority"
        ),
        measured_runner=lambda *args, **kwargs: pytest.fail("child must not launch"),
    )
    with pytest.raises(ValueError):
        ORCH.execute(_argv(paths))


def test_future_manual_review_and_global_result_must_be_absent_preclaim(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    args = ORCH._parse_args(_argv(paths))
    outputs = ORCH._output_paths(args)
    outputs["manual_native_review"].parent.mkdir(parents=True, exist_ok=True)
    outputs["manual_native_review"].write_bytes(b"stale")
    with pytest.raises(ValueError, match="manual native review already exists"):
        ORCH.validate_output_topology(outputs)
    outputs["manual_native_review"].unlink()

    fake_root = tmp_path / "fake_repo"
    monkeypatch.setattr(ORCH, "ROOT", fake_root)
    outputs = ORCH._output_paths(args)
    outputs["global_result"].parent.mkdir(parents=True, exist_ok=True)
    outputs["global_result"].write_bytes(b"stale")
    with pytest.raises(ValueError, match="global result already exists"):
        ORCH.validate_output_topology(outputs)


def test_source_commands_forward_exact_modes_authority_and_measurement_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    result, paths, authority = _run(monkeypatch, tmp_path, monitor)
    assert result["status"] == ORCH.INTERFACE_TERMINAL_STATUS
    assert set(authority[0]["consumed_paths"]) == {
        "exp34_source_runner",
        "exp34_monitor",
        "exp34_orchestrator",
        "renderer",
        "parent_preregistration",
        "exp31_preregistration",
        "exp32_preregistration",
        "cad_preflight",
        "cad_nodes",
        "cad_edges",
        "cad_edge_geodesic_mm",
        "accepted_mesh",
        "scan_N009",
        "scan_N010",
        "transform_N009",
        "transform_N010",
    }
    for index, mode in enumerate(("PRIMARY", "REPLAY")):
        call = monitor.calls[index]
        command = call["command"]
        assert command[command.index("--mode") + 1] == mode
        assert command[command.index("--lock-sha256") + 1] == hashlib.sha256(
            paths["lock"].read_bytes()
        ).hexdigest()
        assert command[command.index("--detached-approval-sha256") + 1] == hashlib.sha256(
            paths["detached_approval"].read_bytes()
        ).hexdigest()
        assert command[0] == str(Path(ORCH.sys.executable))
        assert call["wall_limit_seconds"] == ORCH.SOURCE_WALL_LIMIT_SECONDS
        assert call["rss_limit_bytes"] == ORCH.RSS_LIMIT_BYTES
        assert str(paths["scan_first"]) in command
        assert str(paths["scan_second"]) in command


def test_capability_is_exact_and_durable_before_each_source_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    result, paths, _ = _run(monkeypatch, tmp_path, monitor)
    assert result["status"] == ORCH.INTERFACE_TERMINAL_STATUS
    for call, mode, ordinal, measurement_key in (
        (monitor.calls[0], "PRIMARY", 1, "primary_measurement_root"),
        (monitor.calls[1], "REPLAY", 2, "replay_measurement_root"),
    ):
        measurement = paths[measurement_key]
        capability_path = measurement / "SOURCE_LAUNCH_CAPABILITY.json"
        child_claim_path = measurement / "SOURCE_CHILD.claim.json"
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        assert capability["schema"] == ORCH.SOURCE_CAPABILITY_SCHEMA
        assert capability["status"] == "AUTHORIZED_NOT_CLAIMED"
        assert capability["mode"] == mode
        assert capability["source_child_ordinal"] == ordinal
        assert capability["command_argv"] == call["command"]
        assert capability["command_sha256"] == hashlib.sha256(
            ORCH.canonical_bytes(call["command"])
        ).hexdigest()
        assert capability["source_launch_capability_path"] == str(capability_path)
        assert capability["source_launch_child_claim_path"] == str(child_claim_path)
        assert child_claim_path.is_file()


def test_controller_command_and_capability_match_runner_parser_exactly(
    tmp_path: Path,
) -> None:
    runner_path = ROOT / "scripts/run_exp_cad_0034_cfpdg_source.py"
    spec = importlib.util.spec_from_file_location("exp34_source_interface_audit", runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(runner)
    paths = _fixture_paths(tmp_path)
    paths["source_runner"] = runner_path.resolve()
    args = ORCH._parse_args(_argv(paths))
    measurement = paths["primary_measurement_root"]
    capability_path = measurement / "SOURCE_LAUNCH_CAPABILITY.json"
    child_claim_path = measurement / "SOURCE_CHILD.claim.json"
    launch_sha = "a" * 64
    command = ORCH._source_command(
        args,
        mode="PRIMARY",
        output_root=paths["primary_source_root"],
        measurement_root=measurement,
        launch_claim_sha256=launch_sha,
        capability_path=capability_path,
        child_claim_path=child_claim_path,
    )
    raw_argv = command[2:]
    parsed = runner.build_parser().parse_args(raw_argv)
    assert runner._canonical_full_command(raw_argv) == command
    assert parsed.mode == "PRIMARY"
    assert parsed.source_launch_capability == capability_path
    assert parsed.source_launch_child_claim == child_claim_path
    assert parsed.launch_claim == paths["launch_claim"]
    capability = ORCH._source_capability_payload(
        args,
        mode="PRIMARY",
        output_root=paths["primary_source_root"],
        measurement_root=measurement,
        command=command,
        launch_claim_sha256=launch_sha,
        capability_path=capability_path,
        child_claim_path=child_claim_path,
    )
    assert set(capability) == {
        "schema",
        "experiment_id",
        "status",
        "source_child_ordinal",
        "mode",
        "command_argv",
        "command_sha256",
        "preregistration_path",
        "preregistration_sha256",
        "lock_path",
        "lock_sha256",
        "detached_approval_path",
        "detached_approval_sha256",
        "launch_claim_path",
        "launch_claim_sha256",
        "output_root",
        "measurement_root",
        "scan_accounting_journal",
        "source_launch_capability_path",
        "source_launch_child_claim_path",
        "evaluator_input_authorized",
    }


def test_missing_source_child_claim_is_terminal_before_science_or_replay(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def erase_claim(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if command[command.index("--mode") + 1] == "PRIMARY":
            Path(command[command.index("--source-launch-child-claim") + 1]).unlink()
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", erase_claim)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert len(monitor.calls) == 1
    assert not (
        paths["replay_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
    ).exists()


def test_wrong_child_claim_command_binding_is_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def corrupt_claim(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        child = Path(command[command.index("--source-launch-child-claim") + 1])
        row = json.loads(child.read_text(encoding="utf-8"))
        row["command_sha256"] = "0" * 64
        _write_json(child, row)
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", corrupt_claim)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert len(monitor.calls) == 1


def test_replay_capability_is_created_only_after_primary_success(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    original = monitor.__call__

    def check_timing(*args, **kwargs):
        command = list(args[0])
        mode = command[command.index("--mode") + 1]
        replay_capability = (
            paths["replay_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
        )
        if mode == "PRIMARY":
            assert not replay_capability.exists()
        else:
            assert replay_capability.is_file()
            assert (paths["primary_source_root"] / "SOURCE_RESULT.json").is_file()
        return original(*args, **kwargs)

    monkeypatch.setattr(ORCH, "run_measured_command", check_timing)
    result = ORCH.execute(_argv(paths))
    assert result["status"] == ORCH.INTERFACE_TERMINAL_STATUS


def test_primary_failure_never_authorizes_replay_capability(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(fail_at=("PRIMARY", "SOURCE"))
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    assert not (
        paths["replay_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
    ).exists()


def test_renderer_mutation_after_primary_is_rejected_before_renderer_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def mutate_renderer(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" in command and command[command.index("--mode") + 1] == "PRIMARY":
            paths["renderer"].write_bytes(b"mutated-after-primary")
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", mutate_renderer)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert terminal["fresh_primary_renderer_child_count"] == 0
    assert len(monitor.calls) == 1


def test_approval_mutation_after_primary_is_rejected_before_replay_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    original = monitor.__call__

    def mutate_approval(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if command[command.index("--mode") + 1] == "PRIMARY":
            paths["detached_approval"].write_bytes(b"mutated-after-primary")
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", mutate_approval)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "REPLAY_SOURCE"
    assert terminal["fresh_replay_source_child_count"] == 0
    assert len(monitor.calls) == 1


def test_terminal_publication_canonicalizes_nonfinite_failure_reason(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)

    class NonfiniteErrorMonitor(FakeMonitor):
        def __call__(self, *args, **kwargs):
            raise RuntimeError(float("nan"))

    monitor = NonfiniteErrorMonitor()
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    raw = paths["result"].read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)
