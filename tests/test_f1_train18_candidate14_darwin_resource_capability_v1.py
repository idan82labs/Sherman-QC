from __future__ import annotations

import ast
import ctypes
import errno
import json
import os
from pathlib import Path
import signal

import pytest

from backend import f1_train18_candidate14_darwin_resource_capability_v1 as c14


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py"


class _FakeProcPidRusage:
    def __init__(self, samples: dict[int, tuple[int, int, int]]) -> None:
        self.samples = samples
        self.calls: list[tuple[int, int]] = []
        self.error_number: int | None = None
        self.return_value = 0

    def __call__(self, pid: int, flavor: int, pointer: object) -> int:
        self.calls.append((pid, flavor))
        if self.return_value != 0:
            ctypes.set_errno(self.error_number or errno.ESRCH)
            return self.return_value
        resident, footprint, start = self.samples[pid]
        value = ctypes.cast(
            pointer, ctypes.POINTER(c14.RUsageInfoV2)
        ).contents
        value.ri_resident_size = resident
        value.ri_phys_footprint = footprint
        value.ri_proc_start_abstime = start
        return 0


def _sampler(
    samples: dict[int, tuple[int, int, int]],
) -> tuple[c14.DarwinRusageSampler, _FakeProcPidRusage]:
    function = _FakeProcPidRusage(samples)
    return c14.DarwinRusageSampler(function=function), function


def test_exact_source_lineage_and_source_only_surface() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports <= {
        "__future__",
        "ctypes",
        "dataclasses",
        "errno",
        "os",
        "select",
        "signal",
        "subprocess",
        "sys",
        "time",
        "types",
        "typing",
    }
    assert "Path(" not in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open"
        for node in ast.walk(tree)
    )
    assert "RLIMIT_AS" in source
    assert "resource.setrlimit(resource.RLIMIT_AS" not in source
    assert {key: dict(value) for key, value in c14.FIXED_BINDINGS.items()} == {
        "c13_core": {
            "path": "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
            "sha256": "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
            "size_bytes": 65344,
        },
        "c13_core_tests": {
            "path": "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
            "sha256": "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15",
            "size_bytes": 43387,
        },
        "c13_terminal_disposition": {
            "path": "research/f1_aug30_train18_candidate13_terminal_disposition_v1.json",
            "sha256": "a0b2b04c9a3101e2a7228541c1003e34872ccb29c6c1c28318200060cfcf4abd",
            "size_bytes": 3865,
        },
        "c6_hypothesis": {
            "path": "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
            "sha256": "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252",
            "size_bytes": 53837,
        },
        "c9_adapter": {
            "path": "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
            "sha256": "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096",
            "size_bytes": 22354,
        },
        "c9_adapter_tests": {
            "path": "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
            "sha256": "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623",
            "size_bytes": 16598,
        },
    }
    assert c14.SCIENTIFIC_HYPOTHESIS_ID == (
        "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
    )


def test_no_duplicate_literal_result_keys() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RESULT_KEYS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    call = assignments[0].value
    assert isinstance(call, ast.Call) and len(call.args) == 1
    literal = call.args[0]
    assert isinstance(literal, ast.Set)
    keys = [node.value for node in literal.elts if isinstance(node, ast.Constant)]
    assert len(keys) == len(set(keys)) == len(c14.RESULT_KEYS)


def test_rusage_info_v2_layout_is_exact_160_byte_abi() -> None:
    assert ctypes.sizeof(c14.RUsageInfoV2) == 160
    assert c14.RUsageInfoV2.ri_resident_size.offset == 64
    assert c14.RUsageInfoV2.ri_phys_footprint.offset == 72
    assert c14.RUsageInfoV2.ri_proc_start_abstime.offset == 80
    assert c14.RUsageInfoV2.ri_diskio_byteswritten.offset == 152
    assert len(c14.RUsageInfoV2._fields_) == 19


def test_sampler_decodes_exact_v2_fields() -> None:
    sampler, function = _sampler({37: (101, 202, 303)})
    sample = sampler.sample_pid(37)
    assert sample == c14.ProcessUsageSample(
        pid=37,
        resident_size_bytes=101,
        phys_footprint_bytes=202,
        process_start_abstime=303,
    )
    assert function.calls == [(37, c14.RUSAGE_INFO_V2)]


@pytest.mark.parametrize("pid", [0, -1, True, 1.0, "1"])
def test_sampler_rejects_nonpositive_or_nonexact_pid(pid: object) -> None:
    sampler, _ = _sampler({1: (1, 1, 1)})
    with pytest.raises(ValueError, match="positive exact integer"):
        sampler.sample_pid(pid)  # type: ignore[arg-type]


def test_sampler_rejects_libproc_error_without_a_sample() -> None:
    sampler, function = _sampler({37: (101, 202, 303)})
    function.return_value = -1
    function.error_number = errno.EPERM
    with pytest.raises(c14.DarwinRusageError, match="errno 1"):
        sampler.sample_pid(37)


def test_sampler_rejects_missing_process_start_identity() -> None:
    sampler, _ = _sampler({37: (101, 202, 0)})
    with pytest.raises(c14.DarwinRusageError, match="invalid sample"):
        sampler.sample_pid(37)


def test_combined_sample_sums_parent_worker_separately_and_uses_maximum() -> None:
    parent_pid = os.getpid()
    worker_pid = parent_pid + 100000
    sampler, function = _sampler(
        {
            parent_pid: (100, 400, 11),
            worker_pid: (300, 200, 22),
        }
    )
    combined = c14.combined_usage_sample(
        sampler, parent_pid=parent_pid, worker_pid=worker_pid
    )
    assert combined.combined_resident_size_bytes == 400
    assert combined.combined_phys_footprint_bytes == 600
    assert combined.ceiling_comparison_bytes == 600
    assert function.calls == [
        (parent_pid, c14.RUSAGE_INFO_V2),
        (worker_pid, c14.RUSAGE_INFO_V2),
    ]


def test_sampled_ceiling_within_limit_does_not_kill() -> None:
    parent_pid = os.getpid()
    worker_pid = parent_pid + 100000
    sampler, _ = _sampler(
        {parent_pid: (10, 20, 11), worker_pid: (30, 40, 22)}
    )
    kills: list[tuple[int, int]] = []
    ceiling = c14.SampledDarwinCeiling(
        sampler,
        parent_pid=parent_pid,
        worker_pid=worker_pid,
        ceiling_bytes=100,
        kill_function=lambda pid, sig: kills.append((pid, sig)),
    )
    assert ceiling.sample_and_enforce() == "WITHIN_SAMPLED_CEILING"
    assert kills == []


def test_sampled_ceiling_excess_kills_only_exact_worker_once() -> None:
    parent_pid = os.getpid()
    worker_pid = parent_pid + 100000
    sampler, _ = _sampler(
        {parent_pid: (10, 20, 11), worker_pid: (30, 90, 22)}
    )
    kills: list[tuple[int, int]] = []
    ceiling = c14.SampledDarwinCeiling(
        sampler,
        parent_pid=parent_pid,
        worker_pid=worker_pid,
        ceiling_bytes=100,
        kill_function=lambda pid, sig: kills.append((pid, sig)),
    )
    assert ceiling.sample_and_enforce() == "SAMPLED_CEILING_EXCEEDED_KILL_SENT"
    assert kills == [(worker_pid, signal.SIGKILL)]
    with pytest.raises(RuntimeError, match="already sent"):
        ceiling.sample_and_enforce()
    assert kills == [(worker_pid, signal.SIGKILL)]


def test_sampled_ceiling_rejects_process_identity_drift_before_kill() -> None:
    parent_pid = os.getpid()
    worker_pid = parent_pid + 100000
    sampler, function = _sampler(
        {parent_pid: (10, 20, 11), worker_pid: (30, 90, 22)}
    )
    kills: list[tuple[int, int]] = []
    ceiling = c14.SampledDarwinCeiling(
        sampler,
        parent_pid=parent_pid,
        worker_pid=worker_pid,
        ceiling_bytes=1,
        kill_function=lambda pid, sig: kills.append((pid, sig)),
    )
    function.samples[worker_pid] = (30, 90, 23)
    with pytest.raises(c14.DarwinRusageError, match="identity changed"):
        ceiling.sample_and_enforce()
    assert kills == []


@pytest.mark.parametrize("ceiling", [0, -1, True, 1.0])
def test_sampled_ceiling_requires_positive_exact_ceiling(ceiling: object) -> None:
    parent_pid = os.getpid()
    worker_pid = parent_pid + 100000
    sampler, _ = _sampler(
        {parent_pid: (10, 20, 11), worker_pid: (30, 40, 22)}
    )
    with pytest.raises(ValueError, match="positive exact integer"):
        c14.SampledDarwinCeiling(
            sampler,
            parent_pid=parent_pid,
            worker_pid=worker_pid,
            ceiling_bytes=ceiling,  # type: ignore[arg-type]
        )


def test_live_self_libproc_v2_sample_is_available() -> None:
    sample = c14.DarwinRusageSampler().sample_pid(os.getpid())
    assert sample.pid == os.getpid()
    assert sample.resident_size_bytes > 0
    assert sample.phys_footprint_bytes > 0
    assert sample.process_start_abstime > 0


def test_repeatable_source_only_probe_exercises_cpu_handshake_and_kill() -> None:
    result = c14.run_repeatable_source_only_capability_probe()
    assert set(result) == c14.RESULT_KEYS
    assert result["status"] == "PASS_SOURCE_ONLY_NONAUTHORIZING"
    assert result["check_code"] == "C14_DARWIN_RESOURCE_CAPABILITY_PASSED"
    assert result["child_cpu_backup_limit_verified"] is True
    assert result["combined_parent_worker_sampling_verified"] is True
    assert result["kill_path_exercised"] is True
    assert result["memory_enforcement_is_hard_rlimit_as"] is False
    assert result["memory_enforcement_kind"] == (
        "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING"
    )
    assert result["poll_interval_seconds"] == "0.010"
    assert result["sampling_overshoot_between_polls_possible"] is True
    assert result["project_artifact_or_control_filesystem_discovery_or_open_performed"] is False
    assert result["artifact_or_control_input_opened"] is False
    assert result["label_evaluator_metric_fit_holdout_opened_or_started"] is False
    assert json.loads(json.dumps(result, sort_keys=True)) == result


def test_cpu_limit_readback_is_not_an_optimization_removable_assertion() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "assert resource.getrlimit" not in source
    assert "limits=resource.getrlimit(resource.RLIMIT_CPU)" in source
    assert "or sys.exit(91)" in source
    assert '[sys.executable, "-I", "-S", "-c", child_program]' in source
    assert "sys.flags.isolated==1" in source
    assert "sys.flags.no_site==1" in source
    assert "sys.flags.ignore_environment==1" in source
    assert "sys.flags.no_user_site==1" in source


def test_public_result_contains_no_dynamic_pid_measurement_path_or_timing() -> None:
    result = c14._capability_result()
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    lowered = encoded.lower()
    for forbidden in (
        "pid",
        "resident_size",
        "phys_footprint_bytes",
        "process_start",
        "worker_path",
        "elapsed",
        "measured_bytes",
    ):
        assert forbidden not in lowered
    assert all(type(key) is str for key in result)
    assert all(type(value) in {str, int, bool} for value in result.values())


def test_fixed_resource_contract_is_sampled_and_not_a_hard_memory_limit() -> None:
    assert c14.POLL_INTERVAL_SECONDS == 0.010
    assert c14.CHILD_CPU_SOFT_LIMIT_SECONDS == 240
    assert c14.CHILD_CPU_HARD_LIMIT_SECONDS == 241
    assert c14.MEMORY_ENFORCEMENT_IS_HARD_RLIMIT_AS is False
    assert c14.SAMPLING_OVERSHOOT_BETWEEN_POLLS_POSSIBLE is True
    assert "SAMPLED_DARWIN" in c14.MEMORY_ENFORCEMENT_KIND
