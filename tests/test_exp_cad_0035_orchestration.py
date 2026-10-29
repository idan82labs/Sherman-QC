from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/orchestrate_exp_cad_0035_cfpdg_v5_sequential_physical.py"
SPEC = importlib.util.spec_from_file_location("exp35_orchestrator", PATH)
ORCH = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ORCH)


# The frozen EXP34 orchestration tests were authored before EXP34's one-shot
# terminal result existed.  EXP35 must rerun those exact test bytes without
# moving, rewriting, or deleting that now-immutable result.  This bound plugin
# isolates only the one canonical existence check while the EXP34 test module
# runs.  Its own explicit "global result exists" test switches EXP34 to a
# temporary fake ROOT, so that negative gate remains exercised normally.
EXP34_TERMINAL_GLOBAL_RESULT = (
    ROOT / "output/research/autoresearch-loop/EXP_CAD_0034_RESULT.json"
).resolve()
EXP34_ORCHESTRATION_TEST = ROOT / "tests/test_exp_cad_0034_orchestration.py"
SOURCE_FREE_REGRESSION_PATHS = (
    "tests/test_exp_cad_0031_source_interface.py",
    "tests/test_exp_cad_0032_cfpdg_v5_equivalence.py",
    "tests/test_exp_cad_0033_qualification_protocol.py",
    "tests/test_exp_cad_0034_execution_contract.py",
    "tests/test_exp_cad_0034_source_runner.py",
    "tests/test_exp_cad_0034_orchestration.py",
    "tests/test_exp_cad_0034_process_monitor.py",
    "tests/test_exp_cad_0035_execution_contract.py",
    "tests/test_exp_cad_0035_process_monitor.py",
    "tests/test_exp_cad_0035_source_runner.py",
    "tests/test_exp_cad_0035_orchestration.py",
    "tests/test_exp_cad_0035_source_free_memory_qualification.py",
)


class ConsumedExp34GlobalResultIsolation:
    """Pytest plugin that isolates one consumed-parent output existence bit."""

    def pytest_runtest_setup(self, item) -> None:
        if Path(str(item.fspath)).resolve() != EXP34_ORCHESTRATION_TEST:
            return
        original = Path.exists

        def isolated_exists(path: Path) -> bool:
            try:
                if path.resolve(strict=False) == EXP34_TERMINAL_GLOBAL_RESULT:
                    return False
            except (OSError, RuntimeError):
                pass
            return original(path)

        item._exp35_original_path_exists = original
        Path.exists = isolated_exists

    def pytest_runtest_teardown(self, item, nextitem) -> None:
        original = getattr(item, "_exp35_original_path_exists", None)
        if original is not None:
            Path.exists = original


def run_bound_source_free_regression_suite() -> int:
    """Run the exact inherited+EXP35 suite under the bound isolation plugin."""

    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("EXP35 source-free regression must run from repository root")
    return int(
        pytest.main(
            ["-q", "-p", "no:cacheprovider", *SOURCE_FREE_REGRESSION_PATHS],
            plugins=[ConsumedExp34GlobalResultIsolation()],
        )
    )


def test_module_has_no_top_level_import_of_lock_bound_governance_or_monitor() -> None:
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "scripts.exp_cad_0035_execution_contract" not in imported
    assert "scripts.exp_cad_0035_process_monitor" not in imported


def test_stdlib_bootstrap_rejects_lock_hash_before_bound_import(
    monkeypatch, tmp_path: Path
) -> None:
    fake_root = tmp_path / "repo"
    lock = fake_root / ORCH.LOCK_RELATIVE_PATH
    lock.parent.mkdir(parents=True)
    lock.write_text("{}", encoding="utf-8")
    fake_orchestrator = fake_root / "scripts/orchestrate_exp_cad_0035_cfpdg_v5_sequential_physical.py"
    fake_monitor = fake_root / "scripts/exp_cad_0035_process_monitor.py"
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


def _write_journal(
    root: Path,
    *,
    counts: tuple[int, int, int] = (2, 2, 4),
    mode: str = "PRIMARY",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    operation_order = [
        ("N009", "SHA256"),
        ("N010", "SHA256"),
        ("N009", "POINT_CLOUD_LOAD"),
        ("N010", "POINT_CLOUD_LOAD"),
    ]
    attempted = [
        {
            "attempt_ordinal": ordinal,
            "opaque_token": token,
            "operation": operation,
        }
        for ordinal, (token, operation) in enumerate(
            operation_order[: counts[2]], start=1
        )
    ]
    _write_json(
        root / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json",
        {
            "schema": ORCH.SCAN_JOURNAL_SCHEMA,
            "experiment_id": ORCH.EXPERIMENT_ID,
            "mode": mode,
            "status": "COMPLETE",
            "stage": "COMPLETE",
            "physical_scan_hash_call_count": counts[0],
            "physical_scan_point_cloud_load_call_count": counts[1],
            "physical_scan_byte_open_operation_count": counts[2],
            "attempted_operations": attempted,
            "evaluator_role_label_gold_quality_or_expected_count_access_count": 0,
        },
    )


def _write_source_tree(
    root: Path,
    measurement: Path,
    *,
    science: bool,
    marker: str = "same",
    accounting: tuple[int, int, int] = (2, 2, 4),
    mode: str = "PRIMARY",
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
    _write_journal(measurement, counts=accounting, mode=mode)
    _write_json(
        measurement / "SOURCE_LIFECYCLE_MEASUREMENT.json",
        {
            "schema": ORCH.LIFECYCLE_MEASUREMENT_SCHEMA,
            "experiment_id": ORCH.EXPERIMENT_ID,
            "mode": mode,
            "status": "COMPLETE_RELEASE_VERIFIED",
            "explicit_gc_collect_call_count": 2,
            "maximum_live_source_cloud_count": 1,
            "maximum_live_unfiltered_member_bundle_count": 1,
            "N009_release_verified": True,
            "N010_release_verified": True,
            "ordered_events": ORCH.REQUIRED_RELEASE_EVENTS,
        },
    )


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
        "observed_process_tree_pids": [ORCH.os.getpid()],
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
                    "source_process_pid": ORCH.os.getpid(),
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
                mode=label,
            )
            assert command[command.index("--scan-accounting-journal") + 1] == str(
                measurement / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
            )
            assert command[command.index("--lifecycle-measurement") + 1] == str(
                measurement / "SOURCE_LIFECYCLE_MEASUREMENT.json"
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
        ROOT / "scripts/exp_cad_0035_process_monitor.py"
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
            "launch_claim": attempt / "EXP_CAD_0035_LAUNCH.claim.json",
            "result": attempt / "EXP_CAD_0035_ORCHESTRATION_RESULT.json",
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
        bound_paths["exp35_contract"] = (
            ROOT / "scripts/exp_cad_0035_execution_contract.py"
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
                if key not in {"scan_N009", "scan_N010"}
            },
            "environment": {
                "python_executable_invocation_path": str(Path(ORCH.sys.executable)),
                "python_executable_realpath": str(
                    Path(ORCH.sys.executable).resolve(strict=True)
                ),
                "python_executable_sha256": hashlib.sha256(
                    Path(ORCH.sys.executable).resolve(strict=True).read_bytes()
                ).hexdigest(),
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
    claim = tmp_path / "EXP_CAD_0035_LAUNCH.claim.json"
    ORCH.create_launch_claim(claim, {"ordinal": 1})
    first = claim.read_bytes()
    with pytest.raises(FileExistsError):
        ORCH.create_launch_claim(claim, {"ordinal": 2})
    assert claim.read_bytes() == first
    assert stat.S_IMODE(claim.stat().st_mode) == 0o600


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


def test_successful_source_requires_exact_lifecycle_release_proof(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    original = monitor.__call__

    def mutate_lifecycle(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" in command and command[command.index("--mode") + 1] == "PRIMARY":
            lifecycle_path = Path(
                command[command.index("--lifecycle-measurement") + 1]
            )
            lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
            lifecycle["N009_release_verified"] = False
            _write_json(lifecycle_path, lifecycle)
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", mutate_lifecycle)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_primary_source_child_count"] == 1
    assert terminal["fresh_replay_source_child_count"] == 0
    assert "lifecycle proof" in terminal["failure_reason"]


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
    observed, error = ORCH._safe_accounting(measurement, mode="PRIMARY")
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
        "exp35_source_runner",
        "exp35_monitor",
        "exp35_orchestrator",
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
    runner_path = ROOT / "scripts/run_exp_cad_0035_cfpdg_source.py"
    spec = importlib.util.spec_from_file_location("exp35_source_interface_audit", runner_path)
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
        "lifecycle_measurement",
        "source_launch_capability_path",
        "source_launch_child_claim_path",
        "evaluator_input_authorized",
    }
    assert capability["source_child_ordinal"] == 1


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


def test_source_child_pid_must_be_positive_and_monitor_observed(
    monkeypatch, tmp_path: Path
) -> None:
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    result, _paths, _authority = _run(monkeypatch, tmp_path, monitor)
    source_launches = [
        row for row in result["launches"] if row["kind"] == "SOURCE"
    ]
    assert len(source_launches) == 2
    for launch in source_launches:
        assert launch["source_process_pid"] > 0
        assert launch["source_process_pid"] in launch["measurement"][
            "observed_process_tree_pids"
        ]


def test_wrong_unobserved_source_child_pid_is_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor(source_modes={"PRIMARY": False, "REPLAY": False})
    original = monitor.__call__

    def corrupt_pid(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        child = Path(command[command.index("--source-launch-child-claim") + 1])
        row = json.loads(child.read_text(encoding="utf-8"))
        row["source_process_pid"] = max(result["observed_process_tree_pids"]) + 1
        _write_json(child, row)
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", corrupt_pid)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_primary_source_child_count"] == 1
    assert terminal["fresh_replay_source_child_count"] == 0
    assert "child claim binding mismatch" in terminal["failure_reason"]


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


def test_primary_source_symlink_is_terminal_before_renderer_or_replay(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def replace_source_result_with_symlink(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" in command and command[command.index("--mode") + 1] == "PRIMARY":
            source_result = paths["primary_source_root"] / "SOURCE_RESULT.json"
            outside = tmp_path / "same-source-result.json"
            outside.write_bytes(source_result.read_bytes())
            source_result.unlink()
            source_result.symlink_to(outside)
        return result

    monkeypatch.setattr(
        ORCH, "run_measured_command", replace_source_result_with_symlink
    )
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_primary_source_child_count"] == 1
    assert terminal["fresh_primary_renderer_child_count"] == 0
    assert terminal["fresh_replay_source_child_count"] == 0
    assert len(monitor.calls) == 1
    assert not (
        paths["replay_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
    ).exists()
    assert "symlink" in terminal["failure_reason"]


def test_primary_renderer_symlink_is_terminal_before_replay(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def replace_renderer_asset_with_symlink(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" not in command and "primary" in str(kwargs["measurement_root"]):
            asset = paths["primary_renderer_root"] / "global_paired_diagnostic.png"
            outside = tmp_path / "same-global-paired-diagnostic.png"
            outside.write_bytes(asset.read_bytes())
            asset.unlink()
            asset.symlink_to(outside)
        return result

    monkeypatch.setattr(
        ORCH, "run_measured_command", replace_renderer_asset_with_symlink
    )
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert terminal["fresh_primary_source_child_count"] == 1
    assert terminal["fresh_primary_renderer_child_count"] == 1
    assert terminal["fresh_replay_source_child_count"] == 0
    assert len(monitor.calls) == 2
    assert not (
        paths["replay_measurement_root"] / "SOURCE_LAUNCH_CAPABILITY.json"
    ).exists()
    assert "symlink" in terminal["failure_reason"]


def test_native_renderer_tree_rejects_extra_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "native"
    _write_renderer_tree(root)
    (root / "undeclared-empty-directory").mkdir()
    with pytest.raises(ValueError, match="inventory is not exact"):
        ORCH.validate_native_renderer_tree(root)


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


def test_unrelated_bound_file_drift_is_rejected_before_renderer_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def mutate_unrelated_bound_file(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" in command and command[command.index("--mode") + 1] == "PRIMARY":
            paths["parent_preregistration"].write_bytes(b"unrelated-bound-drift")
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", mutate_unrelated_bound_file)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert terminal["fresh_primary_renderer_child_count"] == 0
    assert len(monitor.calls) == 1
    assert "drift: parent_preregistration" in terminal["failure_reason"]


@pytest.mark.parametrize(
    "stale_name",
    ["SOURCE_LAUNCH_CAPABILITY.json", "SOURCE_CHILD.claim.json"],
)
def test_stale_source_capability_or_child_claim_is_terminal_before_spawn(
    monkeypatch, tmp_path: Path, stale_name: str
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = ORCH._reverify_global_authority_state
    injected = False

    def inject_stale_file(*args, **kwargs):
        nonlocal injected
        result = original(*args, **kwargs)
        if not injected:
            injected = True
            stale = paths["primary_measurement_root"] / stale_name
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"stale")
        return result

    monkeypatch.setattr(ORCH, "_reverify_global_authority_state", inject_stale_file)
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_primary_source_child_count"] == 0
    assert monitor.calls == []
    assert "stale" in terminal["failure_reason"]


def test_capability_byte_drift_is_terminal_before_source_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = ORCH.create_launch_claim

    def mutate_capability(path: Path, payload: dict[str, Any]) -> str:
        digest = original(path, payload)
        if path.name == "SOURCE_LAUNCH_CAPABILITY.json":
            path.write_bytes(path.read_bytes() + b" ")
        return digest

    monkeypatch.setattr(ORCH, "create_launch_claim", mutate_capability)
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_SOURCE"
    assert terminal["fresh_primary_source_child_count"] == 0
    assert monitor.calls == []
    assert "capability bytes" in terminal["failure_reason"]


def test_python_executable_drift_is_rejected_before_renderer_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = monitor.__call__

    def mutate_executable(*args, **kwargs):
        result = original(*args, **kwargs)
        command = list(args[0])
        if "--mode" in command and command[command.index("--mode") + 1] == "PRIMARY":
            monkeypatch.setattr(ORCH.sys, "executable", str(paths["renderer"]))
        return result

    monkeypatch.setattr(ORCH, "run_measured_command", mutate_executable)
    with pytest.raises(ORCH.TerminalExecutionError):
        ORCH.execute(_argv(paths))
    terminal = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert terminal["failure_stage"] == "PRIMARY_RENDERER"
    assert terminal["fresh_primary_renderer_child_count"] == 0
    assert len(monitor.calls) == 1
    assert "Python invocation/realpath" in terminal["failure_reason"]


def test_every_child_spawn_revalidates_full_global_authority(
    monkeypatch, tmp_path: Path
) -> None:
    paths = _fixture_paths(tmp_path)
    _authorize(monkeypatch)
    monitor = FakeMonitor()
    original = ORCH._reverify_global_authority_state
    calls = 0

    def count_revalidation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ORCH, "_reverify_global_authority_state", count_revalidation)
    monkeypatch.setattr(ORCH, "run_measured_command", monitor)
    result = ORCH.execute(_argv(paths))
    assert result["status"] == ORCH.SCIENCE_TERMINAL_STATUS
    # Twice per source (before capability publication and immediately before
    # spawn), once per renderer immediately before spawn.
    assert calls == 6


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


def test_consumed_exp34_result_isolation_is_exact_and_restores_path_exists(
    tmp_path: Path,
) -> None:
    plugin = ConsumedExp34GlobalResultIsolation()
    original = Path.exists
    fake_item = SimpleNamespace(fspath=EXP34_ORCHESTRATION_TEST)
    fake_global = tmp_path / "EXP_CAD_0034_RESULT.json"
    fake_global.write_bytes(b"synthetic-terminal")
    try:
        plugin.pytest_runtest_setup(fake_item)
        assert EXP34_TERMINAL_GLOBAL_RESULT.exists() is False
        assert fake_global.exists() is True
    finally:
        plugin.pytest_runtest_teardown(fake_item, None)
    assert Path.exists is original
    assert EXP34_TERMINAL_GLOBAL_RESULT.exists() is True


def test_bound_source_free_regression_inventory_is_exact() -> None:
    assert len(SOURCE_FREE_REGRESSION_PATHS) == 12
    assert len(set(SOURCE_FREE_REGRESSION_PATHS)) == 12
    assert SOURCE_FREE_REGRESSION_PATHS[0].endswith(
        "test_exp_cad_0031_source_interface.py"
    )
    assert SOURCE_FREE_REGRESSION_PATHS[-1].endswith(
        "test_exp_cad_0035_source_free_memory_qualification.py"
    )
    assert all((ROOT / relative).is_file() for relative in SOURCE_FREE_REGRESSION_PATHS)


def test_bound_regression_runner_matches_attested_contract_commands() -> None:
    from scripts import exp_cad_0035_execution_contract as governance

    assert SOURCE_FREE_REGRESSION_PATHS == governance.SOURCE_FREE_REGRESSION_PATHS
    assert list(governance.SOURCE_FREE_TEST_COMMAND) == [
        ".venv/bin/python",
        "-B",
        "tests/test_exp_cad_0035_orchestration.py",
    ]
    assert list(governance.SOURCE_FREE_COLLECT_COMMAND[-12:]) == list(
        SOURCE_FREE_REGRESSION_PATHS
    )


if __name__ == "__main__":
    raise SystemExit(run_bound_source_free_regression_suite())
