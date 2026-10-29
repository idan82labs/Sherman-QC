from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import exp_cad_0033_performance_worker as W
from scripts import exp_cad_0033_qualification_contract as Q
from scripts import exp_cad_0033_qualification_harness as H
from scripts import run_exp_cad_0029_cfpdg_source as FROZEN_PARENT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nonfinite_publication_is_exactly_the_frozen_parent_policy(tmp_path: Path) -> None:
    value = {
        "array": np.asarray([1.0, np.nan, np.inf, -np.inf]),
        "scalar": np.float64(np.nan),
        "nested": (float("inf"), {"x": float("-inf")}),
    }
    assert Q.null_canonicalize(value) == FROZEN_PARENT._jsonable(value)
    output = tmp_path / "canonical.json"
    Q.canonical_json_write(output, value)
    assert output.read_text(encoding="utf-8") == (
        '{"array":[1.0,null,null,null],"nested":[null,{"x":null}],"scalar":null}'
    )


def test_static_cli_rejects_relative_dotdot_symlink_and_same_hash_alias(
    tmp_path: Path,
) -> None:
    expected = Q.ROOT / "scripts/exp_cad_0033_qualification_contract.py"
    lock = {"bound_files": {"target": Q.file_binding(expected)}}
    Q._exact_static_argument(expected, "target", lock)
    with pytest.raises(ValueError, match="static CLI path"):
        Q._exact_static_argument(Path("scripts/exp_cad_0033_qualification_contract.py"), "target", lock)
    dotted = expected.parent / ".." / "scripts" / expected.name
    with pytest.raises(ValueError, match="static CLI path"):
        Q._exact_static_argument(dotted, "target", lock)
    alias = tmp_path / "alias.py"
    alias.symlink_to(expected)
    with pytest.raises(ValueError, match="static CLI path"):
        Q._exact_static_argument(alias, "target", lock)
    copied = tmp_path / "copied.py"
    copied.write_bytes(expected.read_bytes())
    with pytest.raises(ValueError, match="static CLI path"):
        Q._exact_static_argument(copied, "target", lock)


def test_dynamic_result_paths_are_canonical_and_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    inside = root / "full/run_1/source"
    assert Q._beneath(inside, root, label="output") == inside
    with pytest.raises(ValueError, match="strictly beneath"):
        Q._beneath(root, root, label="output")
    with pytest.raises(ValueError, match="strictly beneath"):
        Q._beneath(tmp_path / "outside", root, label="output")
    with pytest.raises(ValueError, match="alias"):
        Q._beneath(root / "full/../escape", root, label="output")


def test_lock_inventory_includes_transitive_parents_arrays_and_attestations() -> None:
    required = {
        "frozen_core",
        "frozen_v4_runner",
        "exp31_preregistration",
        "first_points",
        "first_normals",
        "second_points",
        "second_normals",
        "source_free_test_attestation",
        "environment_attestation",
        "prelock_source_free_readiness_approval",
    }
    assert required <= Q.REQUIRED_BOUND_FILES
    with pytest.raises(ValueError, match="missing="):
        Q.build_lock_payload(
            bindings={},
            envelope_root=Q.ROOT,
            result_root=Q.ROOT / "tmp/never",
            lock_path=Q.ROOT / "tmp/never.lock.json",
        )


def test_preregistered_candidate_and_terminal_sha_drift_is_rejected() -> None:
    prereg = json.loads((Q.ROOT / Q.PREREGISTRATION_PATH).read_text(encoding="utf-8"))
    mapping = {
        "v5_engine": "candidate_engine",
        "source_free_runner": "candidate_source_free_runner",
        "candidate_worker": "candidate_worker",
        "candidate_harness": "candidate_harness",
        "candidate_engine_tests": "candidate_engine_tests",
        "candidate_harness_tests": "candidate_harness_tests",
    }
    bound = {
        lock_key: dict(prereg["candidate_dependencies_at_freeze"][prereg_key])
        for prereg_key, lock_key in mapping.items()
    }
    bound["terminal_exp32_result"] = {
        "path": prereg["terminal_parent"]["result_path"],
        "sha256": prereg["terminal_parent"]["result_sha256"],
    }
    bound["terminal_exp32_report"] = {
        "path": prereg["terminal_parent"]["report_path"],
        "sha256": prereg["terminal_parent"]["report_sha256"],
    }
    Q._enforce_preregistered_bindings(bound)
    bound["candidate_engine"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate binding"):
        Q._enforce_preregistered_bindings(bound)


def test_manifest_verifier_binds_all_four_cad_and_four_array_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    envelope = tmp_path / "envelope"
    envelope.mkdir()
    names = {
        "cad_nodes": "CAD_GRAPH_NODES.npy",
        "cad_edges": "CAD_GRAPH_EDGES.npy",
        "cad_edge_geodesic_mm": "CAD_GRAPH_EDGE_GEODESIC_MM.npy",
        "cad_mesh": "CAD.ply",
        "first_points": "FIRST_POINTS.npy",
        "first_normals": "FIRST_NORMALS.npy",
        "second_points": "SECOND_POINTS.npy",
        "second_normals": "SECOND_NORMALS.npy",
    }
    manifest_files = {}
    bound_files = {}
    for index, (key, name) in enumerate(names.items()):
        path = envelope / name
        path.write_bytes(f"fixture-{index}".encode())
        digest = _sha(path)
        manifest_files[key] = {"relative_path": name, "sha256": digest}
        bound_files[key] = {
            "path": f"envelope/{name}",
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    manifest = envelope / "BENCHMARK_MANIFEST.json"
    manifest.write_text(json.dumps({"files": manifest_files}), encoding="utf-8")
    bound_files["benchmark_manifest"] = {
        "path": "envelope/BENCHMARK_MANIFEST.json",
        "size_bytes": manifest.stat().st_size,
        "sha256": _sha(manifest),
    }
    lock = {"bound_files": bound_files}
    Q._verify_locked_envelope_files(lock)
    (envelope / "SECOND_NORMALS.npy").write_bytes(b"drift")
    with pytest.raises(ValueError, match="second_normals"):
        Q._verify_locked_envelope_files(lock)


def test_lock_copy_is_rejected_before_any_bound_dependency_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    original = tmp_path / "lock.json"
    copied = tmp_path / "copied-lock.json"
    payload = {
        "schema": Q.LOCK_SCHEMA,
        "experiment_id": Q.EXPERIMENT_ID,
        "status": Q.LOCK_STATUS,
        "qualification_invocation_budget": 1,
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "candidate_payload_identity_must_remain_nested": True,
        "worker_static_argument_bindings": Q.WORKER_STATIC_ARGUMENT_BINDINGS,
        "harness_static_argument_bindings": Q.HARNESS_STATIC_ARGUMENT_BINDINGS,
        "authorized_paths": {
            "envelope_root": "envelope",
            "result_root": "result",
            "lock_path": "lock.json",
        },
        "bound_files": {},
    }
    original.write_bytes(Q.canonical_bytes(payload))
    copied.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="locked canonical path"):
        Q.verify_lock(copied, _sha(copied))


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_reference_dijkstra_is_bound_only_during_execute_and_always_restored(
    preexisting: bool, raises: bool
) -> None:
    sentinel = object()
    module = SimpleNamespace()
    if preexisting:
        module.dijkstra = sentinel

    def execute_source() -> str:
        from scipy.sparse.csgraph import dijkstra

        assert module.dijkstra is dijkstra
        if raises:
            raise RuntimeError("fixture")
        return "ok"

    module.execute_source = execute_source
    candidate = SimpleNamespace(
        _load_module=lambda _path, _name: module,
    )
    original_loader = candidate._load_module
    with W._scoped_reference_dijkstra(candidate):
        loaded = candidate._load_module(Path("unused"), "exp32_frozen_parent_bound")
        if raises:
            with pytest.raises(RuntimeError, match="fixture"):
                loaded.execute_source()
        else:
            assert loaded.execute_source() == "ok"
        if preexisting:
            assert loaded.dijkstra is sentinel
        else:
            assert not hasattr(loaded, "dijkstra")
    assert candidate._load_module is original_loader
    assert module.execute_source is execute_source


def test_outer_result_nests_untouched_exp32_payload() -> None:
    candidate = {
        "schema": H.CANDIDATE_SCHEMA,
        "experiment_id": H.CANDIDATE_EXPERIMENT_ID,
        "status": H.CANDIDATE_PASS,
        "nested": {"schema": "INNER", "value": [1, None, 3]},
    }
    before = copy.deepcopy(candidate)
    outer = H.outer_result(
        candidate_payload=candidate,
        qualification_lock_sha256="a" * 64,
        candidate_harness_sha256="b" * 64,
        candidate_worker_sha256="c" * 64,
    )
    assert candidate == before
    assert outer["candidate_payload"] == before
    assert outer["candidate_identity"]["silently_relabelled"] is False
    assert outer["experiment_id"] == "EXP_CAD_0033"
    assert outer["candidate_payload"]["experiment_id"] == "EXP_CAD_0032"


def _command(
    *, python: str, worker: str, mode: str, lock: str, lock_hash: str, output: Path,
    rows: int | None = None,
) -> list[str]:
    command = [python, worker, mode]
    if rows is not None:
        command += ["--rows-per-member", str(rows)]
    command += [
        "--qualification-lock", lock,
        "--qualification-lock-sha256", lock_hash,
        "--output-root", str(output),
    ]
    return command


def test_lock_mutation_stops_before_second_child_and_is_durably_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    python = "/exact/python"
    worker = "/exact/worker.py"
    payload = {
        "schema": H.JOURNAL_SCHEMA,
        "experiment_id": "EXP_CAD_0033",
        "authorized_attempts": 1,
        "attempts_started": 1,
        "attempt_status": "IN_PROGRESS_CONSUMED",
        "qualification_lock_path": str(lock),
        "qualification_lock_sha256": "a" * 64,
        "detached_approval_path": str(tmp_path / "approval.json"),
        "detached_approval_sha256": "d" * 64,
        "environment_verified": True,
        "verified_environment": {"python_executable_invocation_path": python},
        "worker_path": worker,
        "child_launches": [],
        "child_launch_count": 0,
    }
    journal = H.AttemptJournal(result_root / "journal.json", payload)
    journal.persist()
    calls = []

    def measured(command, **_kwargs):
        calls.append(command)
        return {
            "root_pid": 11,
            "returncode": 1,
            "timed_out": False,
            "wall_seconds": 1.0,
            "peak_process_tree_rss_bytes": 1,
        }

    candidate = SimpleNamespace(run_measured_command=measured)
    monkeypatch.setattr(H, "verify_lock", lambda *_args, **_kwargs: {})
    first_output = result_root / "full/run_1/source"
    second_output = result_root / "full/run_1/renderer"
    with H._journal_candidate_launches(candidate, journal):
        candidate.run_measured_command(
            _command(
                python=python, worker=worker, mode="source", lock=str(lock),
                lock_hash="a" * 64, output=first_output, rows=1_000_000,
            ),
            output_root=result_root / "full/run_1/source_measurement",
            hard_timeout_seconds=150,
        )
        with pytest.raises(ValueError, match="recomputed or changed"):
            candidate.run_measured_command(
                _command(
                    python=python, worker=worker, mode="render", lock=str(lock),
                    lock_hash="b" * 64, output=second_output,
                ),
                output_root=result_root / "full/run_1/renderer_measurement",
                hard_timeout_seconds=60,
            )
    assert len(calls) == 1
    assert journal.payload["child_launches"][1]["status"] == "CONTROLLER_EXCEPTION"


def test_candidate_import_occurs_only_after_durable_attempt_claim() -> None:
    source = inspect.getsource(H.main)
    assert source.index("AttemptJournal.consume") < source.index("_load_candidate_harness")
    assert "exp_cad_0032_performance_harness" not in inspect.getsource(H).split(
        "def _load_candidate_harness", 1
    )[0]


def test_resource_fail_stop_includes_wall_rss_timeout_and_returncode() -> None:
    base = {
        "returncode": 0,
        "timed_out": False,
        "wall_seconds": 150.0,
        "peak_process_tree_rss_bytes": 4 * 1024**3,
    }
    assert H._measurement_passes(base, wall_limit=150.0, rss_limit=4 * 1024**3)
    for update in (
        {"returncode": 1},
        {"timed_out": True},
        {"wall_seconds": 150.0001},
        {"peak_process_tree_rss_bytes": 4 * 1024**3 + 1},
    ):
        assert not H._measurement_passes(
            base | update, wall_limit=150.0, rss_limit=4 * 1024**3
        )


def test_strict_measurement_terminates_before_child_return_on_rss_breach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 123
        killed = False

        def poll(self):
            return -15 if self.killed else None

        def wait(self, timeout=None):
            return -15

    process = Process()
    monkeypatch.setattr(H.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def terminate(value):
        assert value is process
        process.killed = True

    monkeypatch.setattr(H, "_terminate_process_group", terminate)
    candidate = SimpleNamespace(
        RSS_LIMIT_BYTES=100,
        MEMORY_SAMPLING_INTERVAL_SECONDS=0.0,
        process_tree_rss_bytes=lambda _pid: 101,
        parse_time_l_max_rss=lambda _text: None,
    )
    result = H._strict_measured_command(
        candidate,
        ["/python", "/worker", "source"],
        output_root=tmp_path / "measurement",
        hard_timeout_seconds=150.0,
    )
    assert process.killed is True
    assert result["rss_limit_exceeded"] is True
    assert result["resource_kill_reason"] == "RSS_LIMIT"


def test_strict_measurement_kills_child_when_sampler_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 456
        killed = False

        def poll(self):
            return -15 if self.killed else None

        def wait(self, timeout=None):
            return -15

    process = Process()
    monkeypatch.setattr(H.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def terminate(value):
        assert value is process
        process.killed = True

    monkeypatch.setattr(H, "_terminate_process_group", terminate)
    candidate = SimpleNamespace(
        RSS_LIMIT_BYTES=100,
        MEMORY_SAMPLING_INTERVAL_SECONDS=0.0,
        process_tree_rss_bytes=lambda _pid: (_ for _ in ()).throw(
            RuntimeError("sampler fixture")
        ),
        parse_time_l_max_rss=lambda _text: None,
    )
    with pytest.raises(RuntimeError, match="sampler fixture"):
        H._strict_measured_command(
            candidate,
            ["/python", "/worker", "source"],
            output_root=tmp_path / "measurement-exception",
            hard_timeout_seconds=150.0,
        )
    assert process.killed is True


def test_strict_measurement_terminates_on_wall_limit_before_child_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 789
        killed = False

        def poll(self):
            return -15 if self.killed else None

        def wait(self, timeout=None):
            return -15

    process = Process()
    monkeypatch.setattr(H.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monotonic = iter((0.0, 151.0, 151.0))
    monkeypatch.setattr(H.time, "monotonic", lambda: next(monotonic))

    def terminate(value):
        assert value is process
        process.killed = True

    monkeypatch.setattr(H, "_terminate_process_group", terminate)
    candidate = SimpleNamespace(
        RSS_LIMIT_BYTES=100,
        MEMORY_SAMPLING_INTERVAL_SECONDS=0.0,
        process_tree_rss_bytes=lambda _pid: 1,
        parse_time_l_max_rss=lambda _text: None,
    )
    result = H._strict_measured_command(
        candidate,
        ["/python", "/worker", "source"],
        output_root=tmp_path / "measurement-wall",
        hard_timeout_seconds=150.0,
    )
    assert process.killed is True
    assert result["timed_out"] is True
    assert result["resource_kill_reason"] == "WALL_LIMIT"


def test_complete_thirteen_child_fsm_enforces_order_rows_outputs_and_pass_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    python = "/exact/python"
    worker = "/exact/worker.py"
    journal = H.AttemptJournal(
        result_root / "journal.json",
        {
            "schema": H.JOURNAL_SCHEMA,
            "experiment_id": "EXP_CAD_0033",
            "authorized_attempts": 1,
            "attempts_started": 1,
            "attempt_status": "IN_PROGRESS_CONSUMED",
            "qualification_lock_path": str(lock),
            "qualification_lock_sha256": "a" * 64,
            "detached_approval_path": str(tmp_path / "approval.json"),
            "detached_approval_sha256": "d" * 64,
            "environment_verified": True,
            "verified_environment": {"python_executable_invocation_path": python},
            "worker_path": worker,
            "child_launches": [],
            "child_launch_count": 0,
        },
    )
    journal.persist()
    pid = iter(range(100, 113))

    def measured(command, **_kwargs):
        launch_index = command.index("--launch-id")
        launch_id = int(command[launch_index + 1])
        disk = json.loads(journal.path.read_text(encoding="utf-8"))
        row = disk["child_launches"][launch_id - 1]
        row["status"] = "WORKER_CLAIMED"
        claim = result_root / f"EXP_CAD_0033_LAUNCH_{launch_id:02d}.claim.json"
        worker_pid = 1000 + launch_id
        claim_payload = {
            "schema": "EXP_CAD_0033_WORKER_LAUNCH_CLAIM_V1",
            "experiment_id": "EXP_CAD_0033",
            "launch_id": launch_id,
            "command_sha256": row["command_sha256"],
            "qualification_lock_sha256": disk["qualification_lock_sha256"],
            "detached_approval_sha256": disk["detached_approval_sha256"],
            "worker_pid": worker_pid,
        }
        block = Q.canonical_bytes(claim_payload)
        claim.write_bytes(block)
        row["claim_path"] = str(claim)
        row["claim_sha256"] = hashlib.sha256(block).hexdigest()
        row["worker_pid"] = worker_pid
        Q.canonical_json_write(journal.path, disk)
        return {
            "root_pid": next(pid),
            "returncode": 0,
            "timed_out": False,
            "wall_seconds": 1.0,
            "peak_process_tree_rss_bytes": 1,
        }

    candidate = SimpleNamespace(run_measured_command=measured)
    monkeypatch.setattr(H, "verify_lock", lambda *_args, **_kwargs: {})
    with H._journal_candidate_launches(candidate, journal):
        for launch_id, (mode, measurement_relative) in enumerate(
            H.EXPECTED_LAUNCH_SCHEDULE, start=1
        ):
            measurement_root = result_root / measurement_relative
            if mode == "source" and measurement_relative.startswith(("full/", "half/")):
                output = measurement_root.parent / "source"
            elif mode == "render":
                output = measurement_root.parent / "renderer"
            else:
                output = measurement_root.parent / "output"
            rows = (
                1_000_000
                if launch_id in {1, 3, 5}
                else 500_000
                if launch_id == 7
                else 100_000
                if mode in {"source", "reference"}
                else None
            )
            candidate.run_measured_command(
                _command(
                    python=python,
                    worker=worker,
                    mode=mode,
                    lock=str(lock),
                    lock_hash="a" * 64,
                    output=output,
                    rows=rows,
                ),
                output_root=measurement_root,
                hard_timeout_seconds=150.0 if mode != "render" else 60.0,
            )
        journal.assert_full_schedule_complete()
        with pytest.raises(ValueError, match="fourteenth"):
            candidate.run_measured_command(
                _command(
                    python=python,
                    worker=worker,
                    mode="source",
                    lock=str(lock),
                    lock_hash="a" * 64,
                    output=result_root / "extra",
                    rows=100_000,
                ),
                output_root=result_root / "extra-measurement",
                hard_timeout_seconds=150.0,
            )
    assert journal.payload["child_launch_count"] == 13
    assert len({row["pid"] for row in journal.payload["child_launches"]}) == 13


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qualification_lock_sha256", "b" * 64),
        ("physical_source_execution_authorized", True),
        ("approval_path", "/wrong/path.json"),
    ],
)
def test_detached_postlock_approval_rejects_authority_or_identity_drift(
    field: str, value, tmp_path: Path
) -> None:
    approval = tmp_path / "approval.json"
    lock = tmp_path / "lock.json"
    payload = {
        "schema": "EXP_CAD_0033_DETACHED_POSTLOCK_APPROVAL_V1",
        "experiment_id": "EXP_CAD_0033",
        "status": "AUTHORIZE_ONE_SOURCE_FREE_QUALIFICATION",
        "approval_path": str(approval),
        "qualification_lock_path": str(lock),
        "qualification_lock_sha256": "a" * 64,
        "authorized_attempts": 1,
        "qualification_executed": False,
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
    }
    payload[field] = value
    approval.write_bytes(Q.canonical_bytes(payload))
    with pytest.raises(ValueError, match="approval"):
        Q.verify_detached_postlock_approval(
            approval,
            _sha(approval),
            qualification_lock=lock,
            qualification_lock_sha256="a" * 64,
        )


def test_worker_candidate_import_is_strictly_post_lock_and_cli_verification() -> None:
    source = inspect.getsource(W.main)
    assert source.index("verify_lock") < source.index("_load_candidate_worker")
    assert source.index("verify_worker_cli") < source.index("_load_candidate_worker")
    bootstrap = Path(W.__file__).read_text(encoding="utf-8").split(
        "def _load_candidate_worker", 1
    )[0]
    assert "exp_cad_0032_performance_worker" not in bootstrap


def test_actual_candidate_publisher_matches_frozen_parent_null_policy(
    tmp_path: Path,
) -> None:
    from scripts import exp_cad_0032_performance_worker as candidate_worker

    value = {"nan": np.nan, "pos": np.inf, "neg": -np.inf, "ok": np.float64(3.0)}
    assert candidate_worker._jsonable(value) == FROZEN_PARENT._jsonable(value)
    output = tmp_path / "candidate.json"
    candidate_worker.canonical_json_write(output, value)
    assert output.read_text(encoding="utf-8") == (
        '{"nan":null,"neg":null,"ok":3.0,"pos":null}'
    )


def _worker_capability_fixture(
    tmp_path: Path,
    *,
    launch_id: int = 1,
    prior_status: str = "RETURNED",
    command_mismatch: bool = False,
):
    result_root = tmp_path / "result"
    result_root.mkdir(exist_ok=True)
    journal_path = result_root / "EXP_CAD_0033_ATTEMPT_JOURNAL.json"
    lock_path = tmp_path / "lock.json"
    approval_path = tmp_path / "approval.json"
    mode, measurement_relative, rows = W.EXPECTED_LAUNCH_SCHEDULE[launch_id - 1]
    output = (
        result_root / measurement_relative
    ).parent / ("source" if mode == "source" and launch_id <= 7 else "renderer" if mode == "render" else "output")
    raw = [mode]
    if rows is not None:
        raw += ["--rows-per-member", str(rows)]
    raw += [
        "--qualification-lock", str(lock_path),
        "--qualification-lock-sha256", "a" * 64,
        "--output-root", str(output),
        "--post-lock-approval", str(approval_path),
        "--post-lock-approval-sha256", "d" * 64,
        "--attempt-journal", str(journal_path),
        "--launch-id", str(launch_id),
    ]
    args = SimpleNamespace(
        mode=mode,
        rows_per_member=rows,
        qualification_lock=lock_path,
        qualification_lock_sha256="a" * 64,
        output_root=output,
        post_lock_approval=approval_path,
        post_lock_approval_sha256="d" * 64,
        attempt_journal=journal_path,
        launch_id=launch_id,
    )
    full = ["/exact/python", str(Path(W.__file__).resolve()), *raw]
    child_rows = []
    for index in range(1, launch_id + 1):
        row_mode, row_measurement, _row_count = W.EXPECTED_LAUNCH_SCHEDULE[index - 1]
        row_command = full if index == launch_id else ["completed", str(index)]
        row = {
            "launch_id": index,
            "mode": row_mode,
            "measurement_root": str(result_root / row_measurement),
            "status": "STARTED" if index == launch_id else prior_status,
            "command": row_command,
            "command_sha256": hashlib.sha256(Q.canonical_bytes(row_command)).hexdigest(),
        }
        child_rows.append(row)
    if command_mismatch:
        child_rows[-1]["command"] = [*full, "--unexpected"]
    journal = {
        "schema": W.JOURNAL_SCHEMA,
        "experiment_id": "EXP_CAD_0033",
        "authorized_attempts": 1,
        "attempts_started": 1,
        "attempt_status": "IN_PROGRESS_CONSUMED",
        "qualification_lock_path": str(lock_path),
        "qualification_lock_sha256": "a" * 64,
        "detached_approval_path": str(approval_path),
        "detached_approval_sha256": "d" * 64,
        "verified_environment": {"python_executable_invocation_path": "/exact/python"},
        "child_launches": child_rows,
    }
    return args, raw, {"authorized_paths": {"result_root": "result"}}, journal_path, journal


def test_direct_worker_without_durable_journal_is_rejected_before_candidate_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(W, "ROOT", tmp_path)
    monkeypatch.setattr(W, "verify_detached_postlock_approval", lambda *_args, **_kwargs: {})
    args, raw, lock, _journal_path, _journal = _worker_capability_fixture(tmp_path)
    with pytest.raises(ValueError, match="journal path"):
        W._claim_launch_capability(args, lock, raw_argv=raw)


def test_worker_main_no_journal_rejects_before_any_bound_envelope_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    monkeypatch.setattr(W, "ROOT", tmp_path)
    result_root = tmp_path / "result"
    result_root.mkdir()
    lock_path = tmp_path / "lock.json"
    approval_path = tmp_path / "approval.json"
    journal_path = result_root / "EXP_CAD_0033_ATTEMPT_JOURNAL.json"
    bound = {
        key: {"path": f"missing/{key}", "size_bytes": 0, "sha256": "0" * 64}
        for key in Q.REQUIRED_BOUND_FILES
    }
    payload = {
        "schema": Q.LOCK_SCHEMA,
        "experiment_id": Q.EXPERIMENT_ID,
        "status": Q.LOCK_STATUS,
        "qualification_invocation_budget": 1,
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "candidate_payload_identity_must_remain_nested": True,
        "worker_static_argument_bindings": Q.WORKER_STATIC_ARGUMENT_BINDINGS,
        "harness_static_argument_bindings": Q.HARNESS_STATIC_ARGUMENT_BINDINGS,
        "authorized_paths": {
            "envelope_root": "envelope",
            "result_root": "result",
            "lock_path": "lock.json",
        },
        "bound_files": bound,
    }
    lock_path.write_bytes(Q.canonical_bytes(payload))
    original_sha = Q.sha256_path
    hashed: list[Path] = []

    def tracked_sha(path: Path) -> str:
        hashed.append(Path(path))
        return original_sha(path)

    monkeypatch.setattr(Q, "sha256_path", tracked_sha)
    monkeypatch.setattr(W, "verify_self_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(W, "verify_detached_postlock_approval", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        W,
        "verify_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full bound-file verification reached before journal claim")
        ),
    )
    monkeypatch.setattr(
        W,
        "_load_candidate_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate import reached before journal claim")
        ),
    )
    raw = [
        "source",
        "--benchmark-manifest", str(tmp_path / "missing/manifest.json"),
        "--rows-per-member", "1000000",
        "--v5-preregistration", str(tmp_path / "missing/v5.json"),
        "--parent-preregistration", str(tmp_path / "missing/parent.json"),
        "--cad-preflight", str(tmp_path / "missing/cad.json"),
        "--source-free-runner", str(tmp_path / "missing/runner.py"),
        "--monitoring-output", str(result_root / "full/run_1/source_measurement/ENGINE_MONITORING_METRICS.json"),
        "--qualification-lock", str(lock_path),
        "--qualification-lock-sha256", original_sha(lock_path),
        "--output-root", str(result_root / "full/run_1/source"),
        "--post-lock-approval", str(approval_path),
        "--post-lock-approval-sha256", "d" * 64,
        "--attempt-journal", str(journal_path),
        "--launch-id", "1",
    ]
    with pytest.raises(ValueError, match="journal path"):
        W.main(raw)
    assert hashed == [lock_path]


def test_valid_started_capability_claims_once_and_replay_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(W, "ROOT", tmp_path)
    monkeypatch.setattr(W, "verify_detached_postlock_approval", lambda *_args, **_kwargs: {})
    args, raw, lock, journal_path, journal = _worker_capability_fixture(tmp_path)
    Q.canonical_json_write(journal_path, journal)
    W._claim_launch_capability(args, lock, raw_argv=raw)
    claimed = json.loads(journal_path.read_text(encoding="utf-8"))
    assert claimed["child_launches"][0]["status"] == "WORKER_CLAIMED"
    assert (tmp_path / "result/EXP_CAD_0033_LAUNCH_01.claim.json").is_file()
    with pytest.raises(ValueError, match="uniquely STARTED"):
        W._claim_launch_capability(args, lock, raw_argv=raw)


def test_launch_claim_writer_handles_partial_os_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(W, "ROOT", tmp_path)
    monkeypatch.setattr(W, "verify_detached_postlock_approval", lambda *_args, **_kwargs: {})
    args, raw, lock, journal_path, journal = _worker_capability_fixture(tmp_path)
    Q.canonical_json_write(journal_path, journal)
    original_write = W.os.write
    writes: list[int] = []

    def short_write(descriptor, value):
        chunk = bytes(value[:5])
        written = original_write(descriptor, chunk)
        writes.append(written)
        return written

    monkeypatch.setattr(W.os, "write", short_write)
    W._claim_launch_capability(args, lock, raw_argv=raw)
    claim_path = tmp_path / "result/EXP_CAD_0033_LAUNCH_01.claim.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["schema"] == "EXP_CAD_0033_WORKER_LAUNCH_CLAIM_V1"
    assert len(writes) > 1


def test_harness_cryptographically_verifies_claim_bytes_and_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(W, "ROOT", tmp_path)
    monkeypatch.setattr(W, "verify_detached_postlock_approval", lambda *_args, **_kwargs: {})
    args, raw, lock, journal_path, journal = _worker_capability_fixture(tmp_path)
    Q.canonical_json_write(journal_path, journal)
    W._claim_launch_capability(args, lock, raw_argv=raw)
    disk = json.loads(journal_path.read_text(encoding="utf-8"))
    attempt = H.AttemptJournal(journal_path, disk)
    row = disk["child_launches"][0]
    attempt._verify_worker_claim(row, 1)
    claim_path = Path(row["claim_path"])
    claim_path.write_bytes(claim_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="claim SHA"):
        attempt._verify_worker_claim(row, 1)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["worker_pid"] += 1
    block = Q.canonical_bytes(claim)
    claim_path.write_bytes(block)
    row["claim_sha256"] = hashlib.sha256(block).hexdigest()
    with pytest.raises(ValueError, match="contents differ"):
        attempt._verify_worker_claim(row, 1)


def test_attempt_claim_fsyncs_result_parent_before_first_journal_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "result"
    events: list[str] = []
    fake_fd = 987654
    original_open = H.os.open
    original_close = H.os.close
    original_fsync = H.os.fsync

    def tracked_open(path, flags, *args):
        if Path(path) == tmp_path:
            events.append("parent_open")
            return fake_fd
        return original_open(path, flags, *args)

    def tracked_fsync(descriptor):
        if descriptor == fake_fd:
            events.append("parent_fsync")
            return None
        return original_fsync(descriptor)

    def tracked_close(descriptor):
        if descriptor == fake_fd:
            events.append("parent_close")
            return None
        return original_close(descriptor)

    def journal_write(path, payload):
        assert events == ["parent_open", "parent_fsync", "parent_close"]
        Path(path).write_bytes(Q.canonical_bytes(payload))
        return _sha(Path(path))

    monkeypatch.setattr(H.os, "open", tracked_open)
    monkeypatch.setattr(H.os, "fsync", tracked_fsync)
    monkeypatch.setattr(H.os, "close", tracked_close)
    monkeypatch.setattr(H, "canonical_json_write", journal_write)
    H.AttemptJournal.consume(
        result_root=result_root,
        lock_path=tmp_path / "lock.json",
        lock_sha256="a" * 64,
        approval_path=tmp_path / "approval.json",
        approval_sha256="d" * 64,
    )
    assert events == ["parent_open", "parent_fsync", "parent_close"]


def test_authorized_result_metadata_is_canonical_disjoint_and_worker_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    valid = {
        "envelope_root": "envelope",
        "result_root": "result",
        "lock_path": "lock.json",
    }
    Q._validate_authorized_paths_metadata(valid, require_result_root_exists=False)
    with pytest.raises(ValueError, match="canonical repo-relative"):
        Q._validate_authorized_paths_metadata(
            valid | {"result_root": "output/../result"},
            require_result_root_exists=False,
        )
    with pytest.raises(ValueError, match="disjoint"):
        Q._validate_authorized_paths_metadata(
            valid | {"result_root": "envelope/result"},
            require_result_root_exists=False,
        )
    with pytest.raises(FileNotFoundError):
        Q._validate_authorized_paths_metadata(valid, require_result_root_exists=True)
    (tmp_path / "result").mkdir()
    Q._validate_authorized_paths_metadata(valid, require_result_root_exists=True)


def test_out_of_order_and_command_mismatch_capabilities_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(W, "ROOT", tmp_path)
    monkeypatch.setattr(W, "verify_detached_postlock_approval", lambda *_args, **_kwargs: {})
    args, raw, lock, journal_path, journal = _worker_capability_fixture(
        tmp_path, launch_id=2, prior_status="STARTED"
    )
    Q.canonical_json_write(journal_path, journal)
    with pytest.raises(ValueError, match="predecessors"):
        W._claim_launch_capability(args, lock, raw_argv=raw)
    args, raw, lock, journal_path, journal = _worker_capability_fixture(
        tmp_path, command_mismatch=True
    )
    Q.canonical_json_write(journal_path, journal)
    with pytest.raises(ValueError, match="canonical child command"):
        W._claim_launch_capability(args, lock, raw_argv=raw)


def test_harness_has_no_prepare_mode_and_keeps_exact_gate_constants() -> None:
    assert "prepare" not in H.build_parser().format_help().lower()
    candidate = SimpleNamespace(
        DEFAULT_ROWS=1_000_000,
        HALF_ROWS=500_000,
        REFERENCE_DIFFERENTIAL_ROWS=100_000,
        SOURCE_WALL_LIMIT_SECONDS=150.0,
        RENDER_WALL_LIMIT_SECONDS=60.0,
        RSS_LIMIT_BYTES=4 * 1024**3,
        BUNDLE_LIMIT_BYTES=2 * 1024**3,
        SCALING_LIMIT=2.5,
    )
    H.assert_frozen_candidate_gates(candidate)
    candidate.DEFAULT_ROWS = 999_999
    with pytest.raises(ValueError, match="drifted"):
        H.assert_frozen_candidate_gates(candidate)


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/build_exp_cad_0033_qualification_lock.py",
        "scripts/exp_cad_0033_performance_worker.py",
        "scripts/exp_cad_0033_qualification_harness.py",
    ],
)
@pytest.mark.parametrize("outside_cwd", [False, True])
def test_exp33_clis_support_absolute_help_from_any_working_directory(
    relative: str, outside_cwd: bool, tmp_path: Path
) -> None:
    script = Q.ROOT / relative
    completed = subprocess.run(
        [sys.executable, "-B", str(script), "--help"],
        cwd=tmp_path if outside_cwd else Q.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
