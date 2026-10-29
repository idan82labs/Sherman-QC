from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import types

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as c6
from backend import f1_train18_candidate9_lossless_pool_alias_adapter_v1 as c9
from scripts import run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1 as parent_runner
from scripts import run_f1_train18_candidate14_pool_adapter_worker_v1 as worker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_f1_train18_candidate14_pool_adapter_worker_v1.py"
TEST_SOURCE = ROOT / "tests/test_run_f1_train18_candidate14_pool_adapter_worker_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_args(mode: str) -> list[str]:
    return parent_runner._verified_exec_command(
        "worker",
        SOURCE,
        _sha(SOURCE),
        SOURCE.stat().st_size,
        [
        "--worker-sha256",
        _sha(SOURCE),
        "--worker-size-bytes",
        str(SOURCE.stat().st_size),
        "--worker-tests-sha256",
        _sha(TEST_SOURCE),
        "--worker-tests-size-bytes",
        str(TEST_SOURCE.stat().st_size),
        mode,
        ],
    )


def _pool_bytes() -> bytes:
    value = {
        "algorithm": c9.POOL_ALGORITHM,
        "inference_contract": dict(c6.POOL_INFERENCE_CONTRACT),
        "line_objects": [],
        "schema_version": c9.POOL_SCHEMA_VERSION,
        "source_point_serialization": False,
        "working_point_budget": c9.WORKING_POINT_BUDGET,
    }
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _records(blobs: tuple[bytes, ...] | None = None) -> tuple[worker.PoolDescriptor, ...]:
    values = blobs if blobs is not None else (_pool_bytes(),) * worker.DESCRIPTOR_COUNT
    assert len(values) == worker.DESCRIPTOR_COUNT
    return tuple(
        worker.PoolDescriptor(
            ordinal=index,
            pass_code=(
                worker.PRIMARY_PASS_CODE
                if index % 2
                else worker.REPLAY_PASS_CODE
            ),
            size_bytes=len(value),
            sha256=hashlib.sha256(value).hexdigest(),
        )
        for index, value in enumerate(values, 1)
    )


def _stream(blobs: tuple[bytes, ...] | None = None) -> bytes:
    values = blobs if blobs is not None else (_pool_bytes(),) * worker.DESCRIPTOR_COUNT
    return worker.encode_descriptor_prelude(_records(values)) + b"".join(
        worker.encode_blob_frame(value) for value in values
    )


def _valid_aggregate() -> dict[str, object]:
    false_keys = worker.C13_RESULT_KEYS - {
        "check_code",
        "diagnostic_stage_precedence_is_candidate13_contract",
        "exact_c9_success_failure_equivalent_for_all",
        "first_failed_candidate13_diagnostic_stage",
        "kind",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
    value: dict[str, object] = {key: False for key in false_keys}
    value.update(
        {
            "check_code": "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL",
            "diagnostic_stage_precedence_is_candidate13_contract": True,
            "exact_c9_success_failure_equivalent_for_all": True,
            "first_failed_candidate13_diagnostic_stage": "NONE",
            "kind": worker.C13_RESULT_KIND,
            "stage_contract_commitment_sha256": hashlib.sha256(
                worker.canonical_bytes(list(worker.C13_STAGE_CONTRACT))
            ).hexdigest(),
            "stage_passed_all_bitset": "1" * len(worker.C13_STAGE_CONTRACT),
            "stage_reached_all_bitset": "1" * len(worker.C13_STAGE_CONTRACT),
            "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        }
    )
    return value


@pytest.mark.parametrize("mutation", ["initial_oversize", "same_inode_growth"])
def test_source_reader_is_exact_size_bounded_before_and_during_read(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "synthetic-worker-source.py"
    candidate.write_bytes(b"A" if mutation == "same_inode_growth" else b"AA")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    real_read = worker.os.read
    read_sizes: list[int] = []
    grew = False

    def bounded_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        read_sizes.append(size)
        chunk = real_read(descriptor, size)
        if mutation == "same_inode_growth" and chunk and not grew:
            grew = True
            append_fd = os.open(candidate, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(append_fd, b"B")
                os.fsync(append_fd)
            finally:
                os.close(append_fd)
        return chunk

    monkeypatch.setattr(worker.os, "read", bounded_read)
    with pytest.raises(ValueError):
        worker._read_regular_file_once(
            candidate, role="synthetic bounded source", expected_size=1
        )
    if mutation == "initial_oversize":
        assert read_sizes == []
    else:
        assert read_sizes == [1, 1]


def _valid_worker_envelope(
    batch_commitment: str | None = None,
) -> dict[str, object]:
    aggregate = _valid_aggregate()
    value: dict[str, object] = {
        "batch_commitment_sha256": batch_commitment or "7" * 64,
        "blob_inventory_complete": True,
        "c13_aggregate": aggregate,
        "c13_aggregate_commitment_sha256": hashlib.sha256(
            worker.canonical_bytes(aggregate)
        ).hexdigest(),
        "descriptor_inventory_complete": True,
        "input_eof_verified": True,
        "kind": worker.WORKER_RESULT_KIND,
        "schema_version": 1,
    }
    return value


def _read_prelude_from_bytes(value: bytes) -> worker.DescriptorPrelude:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, value)
        os.close(write_fd)
        write_fd = -1
        return worker.read_descriptor_prelude(read_fd)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_worker_source_has_stdlib_top_level_and_exact_fixed_lineage() -> None:
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
        "ast",
        "builtins",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "resource",
        "stat",
        "struct",
        "sys",
        "types",
        "typing",
    }
    assert worker.FIXED_SOURCE_BINDINGS["c14_resource"][1:] == (
        "c82fb204b61f99ce7fb7eab8b5ed15f9003be893c004dac37f347bbdbfe4c079",
        15629,
    )
    assert worker.FIXED_SOURCE_BINDINGS["c14_resource_tests"][1:] == (
        "ac8843ed2fa3d83fc291c5f2ecb0c16fded3320d4484195658551366db8caa51",
        12642,
    )
    assert worker.FIXED_SOURCE_BINDINGS["c13_disposition"][1:] == (
        "a0b2b04c9a3101e2a7228541c1003e34872ccb29c6c1c28318200060cfcf4abd",
        3865,
    )
    assert worker.FIXED_SOURCE_BINDINGS["c13_core"][1:] == (
        "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
        65344,
    )
    assert "._validate_public_result" not in source
    assert "__closure__" not in source
    assert "_execute_lineage_witnesses" in source
    assert '"-I", "-S"' not in source  # isolation is parent launch policy


def test_builder_cluster_are_nonexecuted_exact_lineage_witnesses() -> None:
    names = (
        "_candidate14_worker_verified_builder",
        "_candidate14_worker_verified_clusterer",
    )
    for name in names:
        sys.modules.pop(name, None)
    try:
        builder, clusterer = worker._execute_lineage_witnesses(
            builder_data=worker.FIXED_SOURCE_BINDINGS["builder"][0].read_bytes(),
            clusterer_data=worker.FIXED_SOURCE_BINDINGS["clusterer"][0].read_bytes(),
        )
        assert builder.__candidate14_lineage_only__ is True
        assert clusterer.__candidate14_lineage_only__ is True
        assert set(
            name
            for name, value in vars(builder).items()
            if type(value) is types.FunctionType and value.__module__ == builder.__name__
        ) == {"_bounded_identifiers", "_tag"}
        assert set(
            name
            for name, value in vars(clusterer).items()
            if type(value) is types.FunctionType
            and value.__module__ == clusterer.__name__
        ) == {"cluster_line_candidates"}
        assert builder._tag.__code__.co_filename == str(
            worker.FIXED_SOURCE_BINDINGS["builder"][0]
        )
        assert clusterer.cluster_line_candidates.__code__.co_filename == str(
            worker.FIXED_SOURCE_BINDINGS["clusterer"][0]
        )
        assert "o3d" not in vars(builder)
        assert "cKDTree" not in vars(builder)
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_protocol_constants_are_fixed_and_count_is_exact36() -> None:
    assert len(worker.DESCRIPTOR_MAGIC) == 16
    assert worker.DESCRIPTOR_VERSION == 1
    assert worker.DESCRIPTOR_COUNT == 36
    assert worker.DESCRIPTOR_RECORD.size == 45
    assert worker.BLOB_LENGTH.size == 8
    assert worker.MAX_POOL_LEAF_BYTES == 64 * 1024 * 1024


def test_descriptor_round_trip_seals_all_records_before_blobs() -> None:
    records = _records()
    encoded = worker.encode_descriptor_prelude(records)
    prelude = _read_prelude_from_bytes(encoded)
    assert prelude.records == records
    assert prelude.batch_commitment_sha256 == worker.batch_descriptor_commitment(records)
    assert len(encoded) == 16 + 8 + 36 * 45 + 32


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (0, b"X"),
        (16 + 3, b"\x02"),
        (16 + 7, b"\x23"),
        (16 + 8 + 3, b"\x02"),
        (16 + 8 + 4, b"\x02"),
        (16 + 8 + 36 * 45, b"\xff"),
    ],
)
def test_descriptor_header_order_pass_and_commitment_tamper_reject(
    offset: int, replacement: bytes
) -> None:
    encoded = bytearray(worker.encode_descriptor_prelude(_records()))
    encoded[offset : offset + 1] = replacement
    with pytest.raises((ValueError, EOFError)):
        _read_prelude_from_bytes(bytes(encoded))


def test_descriptor_rejects_wrong_count_duplicate_order_and_oversize() -> None:
    records = list(_records())
    with pytest.raises(ValueError, match="exactly 36"):
        worker.batch_descriptor_commitment(tuple(records[:-1]))
    records[1] = worker.PoolDescriptor(
        ordinal=1,
        pass_code=worker.PRIMARY_PASS_CODE,
        size_bytes=records[1].size_bytes,
        sha256=records[1].sha256,
    )
    with pytest.raises(ValueError, match="order"):
        worker.batch_descriptor_commitment(tuple(records))
    records = list(_records())
    records[0] = worker.PoolDescriptor(
        ordinal=1,
        pass_code=worker.PRIMARY_PASS_CODE,
        size_bytes=worker.MAX_POOL_LEAF_BYTES + 1,
        sha256=records[0].sha256,
    )
    with pytest.raises(ValueError, match="invariant"):
        worker.batch_descriptor_commitment(tuple(records))


def test_blob_frame_is_exact_length_prefix_and_rejects_wrong_types() -> None:
    value = b"abc"
    assert worker.encode_blob_frame(value) == struct.pack(">Q", 3) + value
    with pytest.raises(ValueError):
        worker.encode_blob_frame(bytearray(value))  # type: ignore[arg-type]


def test_cpu_backup_sets_and_reads_exact_240_241(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        worker.resource,
        "setrlimit",
        lambda kind, limits: events.append(("set", kind, limits)),
    )
    monkeypatch.setattr(
        worker.resource,
        "getrlimit",
        lambda kind: (240, 241),
    )
    worker._install_cpu_backup()
    assert events == [("set", worker.resource.RLIMIT_CPU, (240, 241))]


def test_all_prelude_precedes_context_and_exactly36_localizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    blobs = (b"x",) * worker.DESCRIPTOR_COUNT
    prelude = worker.DescriptorPrelude(
        records=_records(blobs),
        batch_commitment_sha256=worker.batch_descriptor_commitment(_records(blobs)),
    )

    class FakeCore:
        FIXED_BINDINGS: dict[str, dict[str, object]] = {}

        @staticmethod
        def make_verified_context(**_kwargs: object) -> object:
            events.append("context")
            return object()

        @staticmethod
        def localize_pool_adapter_contract(
            _context: object, *, pool_bytes: bytes, expected_pool_sha256: str
        ) -> object:
            assert events.index("prelude") < events.index("context")
            assert pool_bytes == b"x"
            assert expected_pool_sha256 == hashlib.sha256(b"x").hexdigest()
            receipt = object()
            events.append(("localize", receipt))
            return receipt

        @staticmethod
        def aggregate_pool_adapter_receipts(
            _context: object, receipts: tuple[object, ...]
        ) -> dict[str, object]:
            assert len(receipts) == 36
            assert len({id(receipt) for receipt in receipts}) == 36
            events.append("aggregate")
            return _valid_aggregate()

    context = worker.BootstrapContext(
        c13_core=FakeCore(),  # type: ignore[arg-type]
        c9_adapter=types.ModuleType("fake_c9"),
        c6_hypothesis=types.ModuleType("fake_c6"),
        builder=types.ModuleType("fake_builder"),
        clusterer=types.ModuleType("fake_cluster"),
        source_bindings={},
        source_commitment_sha256="0" * 64,
        _function_snapshot=(),
        _controlled_callables=(),
        _sentinel=object(),
        _token=object(),
    )
    monkeypatch.setattr(worker, "_validate_bootstrap_context", lambda _value: None)
    monkeypatch.setattr(worker, "_install_cpu_backup", lambda: events.append("cpu"))
    monkeypatch.setattr(
        worker,
        "read_descriptor_prelude",
        lambda _descriptor: (events.append("prelude"), prelude)[1],
    )
    read_fd, write_fd = os.pipe()
    try:
        os.write(
            write_fd,
            b"".join(worker.encode_blob_frame(value) for value in blobs),
        )
        os.close(write_fd)
        write_fd = -1
        result = worker.run_worker(context, input_descriptor=read_fd)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
    assert result == _valid_worker_envelope(prelude.batch_commitment_sha256)
    assert events[:3] == ["cpu", "prelude", "context"]
    assert sum(type(event) is tuple and event[0] == "localize" for event in events) == 36
    assert events[-1] == "aggregate"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kind", "wrong"),
        ("stage_passed_all_bitset", "0111111111"),
        ("stage_reached_all_bitset", "0111111111"),
        ("exact_c9_success_failure_equivalent_for_all", False),
        ("rule_selected", True),
    ],
)
def test_local_public_validator_rejects_malformed_aggregate(
    key: str, value: object
) -> None:
    result = _valid_aggregate()
    result[key] = value
    with pytest.raises(ValueError):
        worker.validate_c13_aggregate(result)


def test_no_mode_and_help_fail_with_empty_stdout() -> None:
    for extra in ([], ["--help"]):
        completed = subprocess.run(
            [sys.executable, str(SOURCE), *extra],
            input=b"",
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 1
        assert completed.stdout == b""
        assert completed.stderr == b"C14_WORKER_FAIL_CLOSED\n"


def test_worker_test_source_size_cap_rejects_before_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_calls = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("oversize worker declaration reached os.read")

    monkeypatch.setattr(worker.os, "read", forbidden_read)
    arguments = _source_args("--source-check-only")[9:]
    size_index = arguments.index("--worker-tests-size-bytes") + 1
    arguments[size_index] = str(worker.MAX_WORKER_TEST_SOURCE_BYTES + 1)
    with pytest.raises(ValueError):
        worker._parse_args(arguments)
    assert read_calls == 0


def test_fresh_source_check_authenticates_exact_sources_only() -> None:
    completed = subprocess.run(
        _source_args("--source-check-only"),
        input=b"",
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b""
    value = json.loads(completed.stdout)
    assert value["status"] == "SOURCE_BYTES_VERIFIED_ONLY"
    assert value["check_code"] == "C14_WORKER_EXACT_SOURCE_BYTES_VERIFIED"
    assert value["fixed_source_and_disposition_lineage_files_opened"] is True
    assert value["input_custody_or_scientific_performance_tested"] is False
    assert value["manifest_pool_semantic_label_evaluator_metric_fit_full28_holdout_opened"] is False


def test_fresh_worker_ignores_pythonpath_sitecustomize(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "sitecustomize-ran"
    (tmp_path / "sitecustomize.py").write_text(
        "import os\nopen(os.environ['C14_SITE_MARKER'], 'w').write('ran')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    environment["C14_SITE_MARKER"] = str(marker)
    completed = subprocess.run(
        _source_args("--source-check-only"),
        input=b"",
        capture_output=True,
        check=False,
        timeout=60,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert marker.exists() is False


def test_fresh_worker_accepts_exact36_synthetic_pools_and_one_aggregate() -> None:
    completed = subprocess.run(
        _source_args("--worker-mode"),
        input=_stream(),
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b""
    value = json.loads(completed.stdout)
    worker.validate_worker_result_envelope(value)
    assert value["c13_aggregate"]["stage_reached_all_bitset"] == "1111111111"
    assert value["c13_aggregate"]["stage_passed_all_bitset"] == "1111111111"
    assert value["batch_commitment_sha256"] == worker.batch_descriptor_commitment(
        _records()
    )


@pytest.mark.parametrize("mutation", ["blob", "trailing", "truncated"])
def test_fresh_worker_failure_stdout_is_empty(mutation: str) -> None:
    value = bytearray(_stream())
    prelude_size = 16 + 8 + worker.DESCRIPTOR_COUNT * 45 + 32
    if mutation == "blob":
        value[prelude_size + 8] ^= 1
    elif mutation == "trailing":
        value.extend(b"X")
    else:
        del value[-1]
    completed = subprocess.run(
        _source_args("--worker-mode"),
        input=bytes(value),
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr == b"C14_WORKER_FAIL_CLOSED\n"


def test_worker_result_contains_no_descriptor_blob_or_per_record_detail() -> None:
    result = _valid_worker_envelope()
    encoded = worker.canonical_bytes(result).decode("utf-8").lower()
    for forbidden in (
        "ordinal",
        "pool_sha",
        "pool_bytes",
        "descriptor_record",
        "record_count",
        "case_id",
        "exception",
        "worker_pid",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("batch_commitment_sha256", "0" * 63),
        ("blob_inventory_complete", False),
        ("descriptor_inventory_complete", False),
        ("input_eof_verified", False),
        ("c13_aggregate_commitment_sha256", "0" * 64),
        ("kind", "wrong"),
    ],
)
def test_worker_result_envelope_tamper_rejects(key: str, value: object) -> None:
    envelope = _valid_worker_envelope()
    envelope[key] = value
    with pytest.raises((ValueError, TypeError)):
        worker.validate_worker_result_envelope(envelope)
