from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from scripts import run_pwts_representation_preflight_v1 as runner


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(runner.canonical_json_bytes(value))


def _configuration(tokens: tuple[str, ...]) -> dict:
    return {
        "preflight_id": runner.PRELIGHT_ID,
        "case_tokens": list(tokens),
        "execution": {
            "semantic_variants": list(runner.PROCESS_ROLES),
            "workers": 1,
            "cpu_only": True,
            "fresh_process_per_case_variant": True,
            "per_process_peak_rss_bytes": runner.PER_PROCESS_PEAK_RSS_BYTES,
        },
    }


def _source_manifest(tmp_path: Path, tokens: tuple[str, ...]) -> Path:
    rows = []
    for token in tokens:
        raw_path = tmp_path / f"{token}.ply"
        raw_path.write_bytes(f"ply\n{token}\n".encode("ascii"))
        rows.append({
            "case_token": token, "raw_path": str(raw_path),
            "raw_byte_count": raw_path.stat().st_size,
            "raw_ply_sha256": runner.sha256_file(raw_path),
            "ply_header_sha256": "b" * 64, "parsed_vertex_count": 128,
            "finite_coordinate_count": 128, "exact_duplicate_count": 0,
            "ignored_property_name_digest": "c" * 64,
            "canonical_coordinate_multiset_sha256": "d" * 64,
        })
    payload = {
        "preflight_id": runner.PRELIGHT_ID, "kind": "pwts_source_lock_v1",
        "semantic_work_performed": False, "sampling_performed": False,
        "source_points_serialized": False, "cases": rows,
    }
    payload["input_manifest_sha256"] = hashlib.sha256(
        runner.canonical_json_bytes(payload)
    ).hexdigest()
    path = tmp_path / "input_manifest.json"
    _write_json(path, payload)
    return path


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    result = []
    for name in ("schema.json", "semantic.py", "validator.py", "tamper.py"):
        path = tmp_path / name
        path.write_text("{}" if name.endswith("json") else "# locked\n")
        result.append(path)
    return tuple(result)  # type: ignore[return-value]


def _successful_result(spec: runner.ProcessSpec) -> runner.ProcessResult:
    output_dir = Path(spec.output_dir)
    scratch_dir = Path(spec.scratch_dir)
    output_dir.mkdir(parents=True)
    scratch_dir.mkdir(parents=True)
    payload = {
        "preflight_id": runner.PRELIGHT_ID,
        "case_token": spec.case_token,
        "process_role": spec.role,
        "source_points_serialized": False,
        "tables": {key: [] for key in runner.TABLE_ROW_CAPS},
    }
    payload_path = output_dir / "case_payload.json"
    _write_json(payload_path, payload)
    stdout_path = output_dir.parent / "stdout.log"
    stderr_path = output_dir.parent / "stderr.log"
    stdout_path.write_bytes(f"stdout:{spec.case_token}:{spec.role}\n".encode("ascii"))
    stderr_path.write_bytes(b"")
    return runner.ProcessResult(
        case_token=spec.case_token, process_role=spec.role, return_code=0,
        cpu_seconds=1.0, wall_seconds=2.0, process_tree_peak_rss_bytes=100,
        scratch_bytes=0, serialized_bytes=payload_path.stat().st_size,
        limiter_mode=(
            runner.DARWIN_WATCHDOG_LIMITER_MODE
            if runner.sys.platform == "darwin"
            else runner.RLIMIT_AS_LIMITER_MODE
        ),
        rss_sampling_interval_seconds=runner.PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS,
        monitor_available=True, full_lifetime_accounting=True,
        reported_passing=True, rss_cap_exceeded=False,
        scratch_cap_exceeded=False, wall_cap_exceeded=False,
        payload_sha256=runner.sha256_file(payload_path),
        stdout_sha256=runner.sha256_file(stdout_path),
        stderr_sha256=runner.sha256_file(stderr_path),
        success=True, failure_code=None,
    )


def _process_spec(tmp_path: Path) -> runner.ProcessSpec:
    process_root = tmp_path / "process"
    return runner.ProcessSpec(
        case_token="C001",
        role="BASE_A",
        executable=str(tmp_path / "worker.py"),
        expected_executable_sha256="a" * 64,
        expected_runner_sha256="b" * 64,
        raw_path=str(tmp_path / "source.ply"),
        source_manifest_path=str(tmp_path / "manifest.json"),
        expected_source_manifest_file_sha256="c" * 64,
        expected_input_manifest_sha256="d" * 64,
        configuration_path=str(tmp_path / "configuration.json"),
        expected_configuration_file_sha256="e" * 64,
        expected_configuration_sha256=runner.EXPECTED_CONFIGURATION_SHA256,
        schema_path=str(tmp_path / "schema.json"),
        expected_schema_sha256="f" * 64,
        output_dir=str(process_root / "output"),
        scratch_dir=str(process_root / "scratch"),
    )


def _one_case_setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path, Path, Path]:
    tokens = ("C001",)
    monkeypatch.setattr(runner, "EXPECTED_CASE_TOKENS", tokens)
    config_payload = _configuration(tokens)
    config_sha = hashlib.sha256(runner.canonical_json_bytes(config_payload)).hexdigest()
    monkeypatch.setattr(runner, "EXPECTED_CONFIGURATION_SHA256", config_sha)
    config = tmp_path / "configuration.json"
    _write_json(config, config_payload)
    source = _source_manifest(tmp_path, tokens)
    schema, semantic, validator, tamper = _sources(tmp_path)
    return source, config, schema, semantic, validator, tamper


def _run(
    *, tmp_path: Path, sources: tuple[Path, Path, Path, Path, Path, Path], executor,
) -> dict:
    source, config, schema, semantic, validator, tamper = sources
    return runner.run_preflight(
        input_manifest_path=source, configuration_path=config, schema_path=schema,
        semantic_executable=semantic, validator_executable=validator,
        tamper_test_source=tamper, output_root=tmp_path / "out",
        process_executor=executor,
    )


def test_configuration_hash_mismatch_fails_before_process_creation(tmp_path: Path) -> None:
    config = tmp_path / "configuration.json"
    _write_json(config, _configuration(runner.EXPECTED_CASE_TOKENS))
    schema, semantic, validator, tamper = _sources(tmp_path)
    calls = []

    result = runner.run_preflight(
        input_manifest_path=tmp_path / "not-opened.json", configuration_path=config,
        schema_path=schema, semantic_executable=semantic,
        validator_executable=validator, tamper_test_source=tamper,
        output_root=tmp_path / "out", process_executor=lambda spec: calls.append(spec),
    )
    assert result["terminal_status"] == runner.INVALID_STATUS
    assert result["configuration_hash_match"] is False
    assert result["semantic_process_started"] is False
    assert calls == []


def test_orchestrator_runs_exact_fresh_role_matrix_with_stubbed_processes(
    tmp_path: Path, monkeypatch,
) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)
    calls = []

    def execute(spec: runner.ProcessSpec) -> runner.ProcessResult:
        calls.append(spec)
        return _successful_result(spec)

    result = _run(tmp_path=tmp_path, sources=sources, executor=execute)
    assert [(spec.case_token, spec.role) for spec in calls] == [
        ("C001", role) for role in runner.PROCESS_ROLES
    ]
    assert len({spec.output_dir for spec in calls}) == 4
    assert result["failure_code"] == "AWAITING_INDEPENDENT_VALIDATION_AND_TAMPER_SUITE"
    assert result["next_professor_packet_authorized"] is False
    execution_lock = json.loads((tmp_path / "out/execution_lock.json").read_text())
    assert execution_lock["limiter_mode"] in {
        runner.DARWIN_WATCHDOG_LIMITER_MODE,
        runner.RLIMIT_AS_LIMITER_MODE,
    }
    assert execution_lock["rss_sampling_interval_seconds"] == (
        runner.PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS
    )
    assert all(
        row["limiter_mode"] == execution_lock["limiter_mode"]
        and row["rss_sampling_interval_seconds"]
        == execution_lock["rss_sampling_interval_seconds"]
        for row in result["resource_rows"]
    )
    checkpoint = json.loads((tmp_path / "out/progress_checkpoint.json").read_text())
    assert checkpoint["next_spec_index"] == 4
    assert checkpoint["orchestration_result_sha256"] == runner.sha256_file(
        tmp_path / "out/orchestration_result.json"
    )

    rerun_calls = []
    rerun = _run(
        tmp_path=tmp_path, sources=sources,
        executor=lambda spec: rerun_calls.append(spec),
    )
    assert rerun == result
    assert rerun_calls == []


def test_clean_resume_continues_only_next_unstarted_spec(tmp_path: Path, monkeypatch) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)
    first_calls = []

    def interrupt_after_one(spec: runner.ProcessSpec) -> runner.ProcessResult:
        first_calls.append((spec.case_token, spec.role))
        if len(first_calls) == 2:
            raise RuntimeError("simulated host interruption")
        return _successful_result(spec)

    try:
        _run(tmp_path=tmp_path, sources=sources, executor=interrupt_after_one)
    except RuntimeError as exc:
        assert "simulated host interruption" in str(exc)
    else:
        raise AssertionError("simulated interruption did not escape")
    checkpoint = json.loads((tmp_path / "out/progress_checkpoint.json").read_text())
    assert checkpoint["next_spec_index"] == 1
    assert [row["process_role"] for row in checkpoint["completed_process_results"]] == ["BASE_A"]

    resumed_calls = []

    def resume(spec: runner.ProcessSpec) -> runner.ProcessResult:
        resumed_calls.append((spec.case_token, spec.role))
        return _successful_result(spec)

    result = _run(tmp_path=tmp_path, sources=sources, executor=resume)
    assert resumed_calls == [
        ("C001", "BASE_B"),
        ("C001", "ORDER_PERMUTED"),
        ("C001", "INDEPENDENT_VALIDATOR"),
    ]
    assert result["completed_process_count"] == 4


def test_resume_rejects_changed_locked_source_without_starting_process(
    tmp_path: Path, monkeypatch,
) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)

    def interrupt(spec: runner.ProcessSpec) -> runner.ProcessResult:
        if spec.role == "BASE_B":
            raise RuntimeError("stop")
        return _successful_result(spec)

    try:
        _run(tmp_path=tmp_path, sources=sources, executor=interrupt)
    except RuntimeError:
        pass
    sources[3].write_text("# changed semantic source\n")
    calls = []
    result = _run(
        tmp_path=tmp_path, sources=sources,
        executor=lambda spec: calls.append(spec),
    )
    assert result["failure_code"] == "CHECKPOINT_VALIDATION_FAILED"
    assert "execution lock differs" in result["failure_detail"]
    assert calls == []


def test_resume_rejects_tampered_completed_payload_without_rerun(
    tmp_path: Path, monkeypatch,
) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)

    def interrupt(spec: runner.ProcessSpec) -> runner.ProcessResult:
        if spec.role == "BASE_B":
            raise RuntimeError("stop")
        return _successful_result(spec)

    try:
        _run(tmp_path=tmp_path, sources=sources, executor=interrupt)
    except RuntimeError:
        pass
    payload = tmp_path / "out/processes/C001/BASE_A/output/case_payload.json"
    payload.write_bytes(payload.read_bytes() + b"\n")
    calls = []
    result = _run(
        tmp_path=tmp_path, sources=sources,
        executor=lambda spec: calls.append(spec),
    )
    assert result["failure_code"] == "CHECKPOINT_VALIDATION_FAILED"
    assert "evidence hash/size mismatch" in result["failure_detail"]
    assert calls == []


def test_resume_rejects_checkpoint_gap_without_starting_process(
    tmp_path: Path, monkeypatch,
) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)

    def interrupt(spec: runner.ProcessSpec) -> runner.ProcessResult:
        if spec.role == "BASE_B":
            raise RuntimeError("stop")
        return _successful_result(spec)

    try:
        _run(tmp_path=tmp_path, sources=sources, executor=interrupt)
    except RuntimeError:
        pass
    checkpoint_path = tmp_path / "out/progress_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["completed_process_results"] = []
    checkpoint["next_spec_index"] = 1
    checkpoint = runner._content_bound_payload(checkpoint, hash_key="checkpoint_sha256")
    _write_json(checkpoint_path, checkpoint)
    calls = []
    result = _run(
        tmp_path=tmp_path, sources=sources,
        executor=lambda spec: calls.append(spec),
    )
    assert result["failure_code"] == "CHECKPOINT_VALIDATION_FAILED"
    assert "process gap" in result["failure_detail"]
    assert calls == []


def test_terminal_first_role_failure_is_never_retried(tmp_path: Path, monkeypatch) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)
    first_calls = []

    def fail_first(spec: runner.ProcessSpec) -> runner.ProcessResult:
        first_calls.append((spec.case_token, spec.role))
        passing = _successful_result(spec)
        return replace(
            passing, return_code=2, reported_passing=False,
            success=False, failure_code="WORKER_NONZERO_EXIT",
        )

    failed = _run(tmp_path=tmp_path, sources=sources, executor=fail_first)
    assert failed["failure_code"] == "WORKER_NONZERO_EXIT"
    assert first_calls == [("C001", "BASE_A")]
    checkpoint = json.loads((tmp_path / "out/progress_checkpoint.json").read_text())
    assert checkpoint["terminal_failure_code"] == "WORKER_NONZERO_EXIT"
    assert checkpoint["next_spec_index"] == 0

    retry_calls = []
    repeated = _run(
        tmp_path=tmp_path, sources=sources,
        executor=lambda spec: retry_calls.append(spec),
    )
    assert repeated == failed
    assert retry_calls == []


def test_worker_payload_rejects_source_coordinate_serialization(tmp_path: Path) -> None:
    payload = {
        "preflight_id": runner.PRELIGHT_ID, "case_token": "C001",
        "process_role": "BASE_A", "source_points_serialized": False,
        "tables": {key: [] for key in runner.TABLE_ROW_CAPS},
        "raw_points": [[1, 2, 3]],
    }
    path = tmp_path / "case_payload.json"
    _write_json(path, payload)
    try:
        runner.validate_worker_payload(path, case_token="C001", role="BASE_A")
    except ValueError as exc:
        assert "forbidden source-point serialization" in str(exc)
    else:
        raise AssertionError("raw source coordinates were accepted")


def test_darwin_preexec_omits_rlimit_as_but_retains_cpu_and_fsize(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(runner.os, "setsid", lambda: calls.append(("setsid", None)))
    monkeypatch.setattr(
        runner.resource, "setrlimit", lambda kind, value: calls.append((kind, value))
    )

    runner._preexec_limits(limiter_mode=runner.DARWIN_WATCHDOG_LIMITER_MODE)

    kinds = [kind for kind, _value in calls if kind != "setsid"]
    assert runner.resource.RLIMIT_CPU in kinds
    assert runner.resource.RLIMIT_FSIZE in kinds
    assert runner.resource.RLIMIT_AS not in kinds
    contract = runner._limiter_contract(
        platform_name="darwin", monitor_verified=True, rlimit_as_supported=True,
    )
    assert contract.limiter_mode == runner.DARWIN_WATCHDOG_LIMITER_MODE
    assert contract.failure_code is None


def test_non_darwin_preexec_retains_rlimit_as(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(runner.os, "setsid", lambda: None)
    monkeypatch.setattr(
        runner.resource, "setrlimit", lambda kind, value: calls.append((kind, value))
    )

    runner._preexec_limits(limiter_mode=runner.RLIMIT_AS_LIMITER_MODE)

    assert [kind for kind, _value in calls] == [
        runner.resource.RLIMIT_CPU,
        runner.resource.RLIMIT_AS,
        runner.resource.RLIMIT_FSIZE,
    ]
    contract = runner._limiter_contract(
        platform_name="linux", monitor_verified=True, rlimit_as_supported=True,
    )
    assert contract.limiter_mode == runner.RLIMIT_AS_LIMITER_MODE
    assert contract.failure_code is None


def test_execute_fails_closed_when_process_tree_monitor_is_unavailable(
    tmp_path: Path, monkeypatch,
) -> None:
    spec = _process_spec(tmp_path)
    limiter = runner.LimiterContract(
        limiter_mode=runner.DARWIN_WATCHDOG_LIMITER_MODE,
        rss_sampling_interval_seconds=runner.PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS,
        monitor_available=False,
        failure_code="PROCESS_TREE_MONITOR_UNAVAILABLE",
    )
    monkeypatch.setattr(runner, "_limiter_contract", lambda: limiter)
    popen_called = False

    def should_not_launch(*args, **kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("worker launch must remain closed")

    monkeypatch.setattr(runner.subprocess, "Popen", should_not_launch)
    result = runner.execute_supervised(spec)

    assert popen_called is False
    assert result.failure_code == "PROCESS_TREE_MONITOR_UNAVAILABLE"
    assert result.monitor_available is False
    assert result.full_lifetime_accounting is False
    assert result.success is False


def test_preexec_launch_failure_returns_failed_result_and_terminal_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    sources = _one_case_setup(tmp_path, monkeypatch)
    limiter = runner.LimiterContract(
        limiter_mode=runner.DARWIN_WATCHDOG_LIMITER_MODE,
        rss_sampling_interval_seconds=runner.PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS,
        monitor_available=True,
        failure_code=None,
    )
    monkeypatch.setattr(runner, "_limiter_contract", lambda: limiter)

    def launch_failure(*args, **kwargs):
        raise runner.subprocess.SubprocessError("Exception occurred in preexec_fn.")

    monkeypatch.setattr(runner.subprocess, "Popen", launch_failure)
    result = _run(
        tmp_path=tmp_path, sources=sources, executor=runner.execute_supervised,
    )

    assert result["failure_code"] == "PROCESS_LAUNCH_FAILED"
    assert result["completed_process_count"] == 1
    row = result["resource_rows"][0]
    assert row["failure_code"] == "PROCESS_LAUNCH_FAILED"
    assert row["limiter_mode"] == runner.DARWIN_WATCHDOG_LIMITER_MODE
    checkpoint = json.loads((tmp_path / "out/progress_checkpoint.json").read_text())
    assert checkpoint["terminal_failure_code"] == "PROCESS_LAUNCH_FAILED"
    assert checkpoint["next_spec_index"] == 0


def test_process_tree_rss_cap_kills_group_immediately(tmp_path: Path, monkeypatch) -> None:
    spec = _process_spec(tmp_path)
    limiter = runner.LimiterContract(
        limiter_mode=runner.DARWIN_WATCHDOG_LIMITER_MODE,
        rss_sampling_interval_seconds=runner.PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS,
        monitor_available=True,
        failure_code=None,
    )
    monkeypatch.setattr(runner, "_limiter_contract", lambda: limiter)

    class FakeProcess:
        pid = 424242
        return_code = None

        def poll(self):
            return self.return_code

        def wait(self):
            if self.return_code is None:
                self.return_code = -9
            return self.return_code

    process = FakeProcess()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        runner,
        "_process_tree_sample",
        lambda root_pid, pgid: (
            runner.PER_PROCESS_PEAK_RSS_BYTES + 1024,
            (root_pid,),
            False,
        ),
    )
    kill_order = []

    def kill_group(pgid, signum):
        kill_order.append(("group", pgid, signum))
        process.return_code = -9

    monkeypatch.setattr(runner.os, "killpg", kill_group)
    monkeypatch.setattr(
        runner.os, "kill", lambda pid, signum: kill_order.append(("pid", pid, signum))
    )

    result = runner.execute_supervised(spec)

    assert result.failure_code == "PROCESS_TREE_RSS_CAP_EXCEEDED"
    assert result.rss_cap_exceeded is True
    assert result.process_tree_peak_rss_bytes == runner.PER_PROCESS_PEAK_RSS_BYTES + 1024
    assert kill_order[0][:2] == ("group", process.pid)
    assert result.success is False


def test_real_short_lived_worker_has_clean_final_process_group_sample(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        """\
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--role", required=True)
parser.add_argument("--case-token", required=True)
parser.add_argument("--output-dir", required=True)
args, _unknown = parser.parse_known_args()
time.sleep(0.05)
payload = {
    "preflight_id": "PWTS_REPRESENTATION_PREFLIGHT_V1",
    "case_token": args.case_token,
    "process_role": args.role,
    "source_points_serialized": False,
    "tables": {
        "seed_primitive": [],
        "trace_primitive": [],
        "trace_node": [],
        "strip_membership": [],
        "certificate": [],
    },
}
Path(args.output_dir, "case_payload.json").write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":"))
)
""",
        encoding="utf-8",
    )
    spec = replace(
        _process_spec(tmp_path),
        executable=str(worker),
        expected_executable_sha256=runner.sha256_file(worker),
        expected_runner_sha256=runner.sha256_file(Path(runner.__file__)),
    )

    result = runner.execute_supervised(spec)

    assert result.return_code == 0
    assert result.failure_code is None
    assert result.monitor_available is True
    assert result.full_lifetime_accounting is True
    assert result.process_tree_peak_rss_bytes > 0
    assert result.success is True


def test_unsupported_platform_without_rlimit_as_fails_closed() -> None:
    contract = runner._limiter_contract(
        platform_name="unsupported", monitor_verified=True,
        rlimit_as_supported=False,
    )
    assert contract.limiter_mode == runner.UNSUPPORTED_LIMITER_MODE
    assert contract.failure_code == "UNSUPPORTED_PROCESS_TREE_LIMITER"
