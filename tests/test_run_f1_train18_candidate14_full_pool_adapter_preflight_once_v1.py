from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import types
from types import MappingProxyType

import pytest

from backend import f1_train18_candidate14_darwin_resource_capability_v1 as resource_core
from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as c6
from backend import f1_train18_candidate9_lossless_pool_alias_adapter_v1 as c9
from scripts import run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1 as parent
from scripts import run_f1_train18_candidate14_pool_adapter_worker_v1 as worker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py"
TEST_SOURCE = ROOT / "tests/test_run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_args(mode: str) -> list[str]:
    return parent._verified_exec_command(
        "parent",
        SOURCE,
        _sha(SOURCE),
        SOURCE.stat().st_size,
        [
        "--runner-sha256",
        _sha(SOURCE),
        "--runner-size-bytes",
        str(SOURCE.stat().st_size),
        "--runner-tests-sha256",
        _sha(TEST_SOURCE),
        "--runner-tests-size-bytes",
        str(TEST_SOURCE.stat().st_size),
        mode,
        ],
    )


def _pool_bytes() -> bytes:
    return parent.canonical_bytes(
        {
            "algorithm": c9.POOL_ALGORITHM,
            "inference_contract": dict(c6.POOL_INFERENCE_CONTRACT),
            "line_objects": [],
            "schema_version": c9.POOL_SCHEMA_VERSION,
            "source_point_serialization": False,
            "working_point_budget": c9.WORKING_POINT_BUDGET,
        }
    )


def _vectors() -> tuple[parent.PoolVector, ...]:
    value = _pool_bytes()
    sha = hashlib.sha256(value).hexdigest()
    return tuple(
        parent.PoolVector(
            parent.ManifestRecord(
                ordinal=index,
                pair_ordinal=(index - 1) // 2,
                pass_name="primary" if index % 2 else "replay",
                pool_path=parent.ALLOWED_CACHE_ROOT
                / ("primary" if index % 2 else "replay")
                / f"T{((index - 1) // 2) + 1:03d}"
                / "pool"
                / f"T{((index - 1) // 2) + 1:03d}.json",
                pool_sha256=sha,
                manifest_record_binding_sha256=f"{index:064x}",
            ),
            file_binding_sha256=f"{index + 100:064x}",
            size_bytes=len(value),
            pool_bytes=value,
        )
        for index in range(1, 37)
    )


def _all_pass_aggregate() -> dict[str, object]:
    value: dict[str, object] = {key: False for key in parent.C13_RESULT_KEYS}
    value.update(
        {
            "check_code": "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL",
            "diagnostic_stage_precedence_is_candidate13_contract": True,
            "exact_c9_success_failure_equivalent_for_all": True,
            "first_failed_candidate13_diagnostic_stage": "NONE",
            "kind": "sherman_qc_f1_aug30_candidate13_pool_adapter_stage_result_v1",
            "stage_contract_commitment_sha256": parent.sha256_bytes(
                parent.canonical_bytes(list(parent.C13_STAGE_CONTRACT))
            ),
            "stage_passed_all_bitset": "1" * 10,
            "stage_reached_all_bitset": "1" * 10,
            "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        }
    )
    parent.validate_worker_aggregate(value)
    return value


def _context() -> parent.BootstrapContext:
    token = object()
    return parent.BootstrapContext(
        resource_core=resource_core,
        source_bindings=MappingProxyType({}),
        source_entries=(),
        source_commitment_sha256="a" * 64,
        _resource_functions=(
            resource_core.DarwinRusageSampler,
            resource_core.SampledDarwinCeiling,
            resource_core.run_repeatable_source_only_capability_probe,
        ),
        _sentinel=object(),
        _token=token,
    )


def _qualification(reader: parent.GuardedReader | None = None) -> parent.ControlQualification:
    cases = tuple(f"CASE-{index:02d}" for index in range(18))
    scans = MappingProxyType({case: f"{index + 1:064x}" for index, case in enumerate(cases)})
    train_commitment = parent.sha256_bytes(
        parent.canonical_bytes(
            [{"case_id": case, "scan_sha256": scans[case]} for case in cases]
        )
    )
    return parent.ControlQualification(
        reader=parent.GuardedReader() if reader is None else reader,
        train_ids=cases,
        scan_by_case=scans,
        train_commitment_sha256=train_commitment,
        manifest_path=parent.CACHE_MANIFEST,
        manifest_sha256=parent.CACHE_MANIFEST_SHA256,
        manifest_size_bytes=parent.CACHE_MANIFEST_SIZE_BYTES,
        control_binding_commitment_sha256="b" * 64,
        control_read_inventory_commitment_sha256="c" * 64,
        control_raw_fd_opened_commitment_sha256="d" * 64,
        control_verified_inode_commitment_sha256="e" * 64,
        control_read_started_commitment_sha256="f" * 64,
        resource_capability_result_commitment_sha256="1" * 64,
        worker_source_result_commitment_sha256="2" * 64,
        worker_source_inventory_commitment_sha256="3" * 64,
        _context_token=object(),
        _sentinel=object(),
        _token=object(),
    )


def _issued_qualification(
    context: parent.BootstrapContext,
    reader: parent.GuardedReader | None = None,
) -> parent.ControlQualification:
    original = _qualification(reader)
    token = object()
    qualification = parent.ControlQualification(
        original.reader,
        original.train_ids,
        original.scan_by_case,
        original.train_commitment_sha256,
        original.manifest_path,
        original.manifest_sha256,
        original.manifest_size_bytes,
        original.control_binding_commitment_sha256,
        original.control_read_inventory_commitment_sha256,
        original.control_raw_fd_opened_commitment_sha256,
        original.control_verified_inode_commitment_sha256,
        original.control_read_started_commitment_sha256,
        original.resource_capability_result_commitment_sha256,
        original.worker_source_result_commitment_sha256,
        original.worker_source_inventory_commitment_sha256,
        context._token,
        parent._QUALIFICATION_SEAL,
        token,
    )
    lengths = (
        len(qualification.reader.entries),
        len(qualification.reader.raw_fd_opened_purposes),
        len(qualification.reader.verified_inode_purposes),
        len(qualification.reader.read_started_purposes),
    )
    projection = parent._qualification_issuance_projection(
        qualification,
        entry_length=0,
        raw_length=0,
        verified_length=0,
        started_length=0,
    )
    parent._ISSUED_QUALIFICATIONS[id(qualification)] = (
        qualification,
        token,
        lengths,
        projection,
        parent.sha256_bytes(projection),
    )
    return qualification


def _authorization(root: Path, reader: parent.GuardedReader | None = None) -> parent.AuthorizationSeal:
    qualification = _qualification(reader)
    return parent.AuthorizationSeal(
        reader=qualification.reader,
        qualification=qualification,
        authorization_bytes=b"{}\n",
        authorization_sha256="4" * 64,
        authorization_size_bytes=3,
        qualification_bytes=b"{}\n",
        qualification_sha256="5" * 64,
        qualification_size_bytes=3,
        output_freshness=parent.FreshOutputSeal(root, (1, 1), object()),
        anchor_parent_identity=(1, 1),
        issuance_commitment_sha256="6" * 64,
        _context_token=object(),
        _sentinel=object(),
        _token=object(),
    )


def test_source_is_stdlib_only_and_binds_frozen_worker_resource() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
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
        "builtins",
        "ctypes",
        "dataclasses",
        "errno",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "resource",
        "selectors",
        "signal",
        "stat",
        "struct",
        "subprocess",
        "sys",
        "time",
        "types",
        "typing",
    }
    assert parent.FIXED_SOURCE_BINDINGS["c14_worker"][1:] == (
        "58e68be857aa81651fff10d423a43b4d7fe745f45dafe718a34dd68ec49bc137",
        50687,
    )
    assert parent.FIXED_SOURCE_BINDINGS["c14_worker_tests"][1:] == (
        "6842932c572f448b1f6a087dee44965cdf5a935ba9991787ba808aea951d515a",
        20844,
    )
    assert parent.FIXED_SOURCE_BINDINGS["c14_resource"][1:] == (
        "c82fb204b61f99ce7fb7eab8b5ed15f9003be893c004dac37f347bbdbfe4c079",
        15629,
    )
    assert parent.FIXED_SOURCE_BINDINGS["c14_resource_tests"][1:] == (
        "ac8843ed2fa3d83fc291c5f2ecb0c16fded3320d4484195658551366db8caa51",
        12642,
    )
    assert parent._worker_command()[1:3] == ["-I", "-S"]


def test_candidate13_disposition_contract_binds_present_exact_fields() -> None:
    disposition = {
        "candidate14_engineering_successor_authorized": True,
        "engineering_attempt_id": "CANDIDATE13_POOL_ADAPTER_STAGE_LOCALIZER_V1",
        "kind": "sherman_qc_f1_aug30_train18_candidate13_terminal_disposition_v1",
        "schema_version": 1,
        "scientific_hypothesis_id": parent.SCIENTIFIC_HYPOTHESIS_ID,
    }
    assert "attempt_ordinal" not in disposition
    parent._validate_candidate13_terminal_disposition(disposition)

    mutations = (
        ("candidate14_engineering_successor_authorized", False),
        ("engineering_attempt_id", "CANDIDATE13_OTHER"),
        ("kind", "candidate13_other"),
        ("schema_version", True),
        ("scientific_hypothesis_id", "OTHER_HYPOTHESIS"),
    )
    for key, replacement in mutations:
        changed = dict(disposition)
        changed[key] = replacement
        with pytest.raises(parent.RunnerFailure) as captured:
            parent._validate_candidate13_terminal_disposition(changed)
        assert captured.value.stage == "SOURCE_CONTROL"
        assert captured.value.code == "C13_DISPOSITION_MISMATCH"


def test_no_mode_and_help_fail_closed() -> None:
    for extra in ([], ["--help"]):
        completed = subprocess.run(
            [sys.executable, str(SOURCE), *extra],
            input=b"",
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 1
        value = json.loads(completed.stdout)
        assert value["status"] == "FAIL_CLOSED"
        assert completed.stderr == b""


def test_fresh_source_check_only_passes_without_data_modes() -> None:
    completed = subprocess.run(
        _source_args("--source-check-only"),
        input=b"",
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout.decode("utf-8", "replace")
    value = json.loads(completed.stdout)
    assert value["status"] == "SOURCE_BYTES_VERIFIED_ONLY"
    assert value["check_code"] == "C14_EXACT_SOURCE_BYTES_VERIFIED"
    assert value["manifest_pool_semantic_label_evaluator_metric_fit_full28_holdout_opened"] is False


def test_fresh_parent_ignores_pythonpath_sitecustomize(
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
        timeout=60,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout.decode("utf-8", "replace")
    assert marker.exists() is False


def test_verified_exec_rejects_mismatched_top_level_before_side_effect(
    tmp_path: Path,
) -> None:
    synthetic_root = tmp_path / "verified-root"
    scripts_dir = synthetic_root / "scripts"
    scripts_dir.mkdir(parents=True)
    candidate = scripts_dir / SOURCE.name
    marker = tmp_path / "unverified-top-level-ran"
    candidate_bytes = (
        f"open({str(marker)!r}, 'wb').write(b'unverified')\n".encode("utf-8")
    )
    candidate.write_bytes(candidate_bytes)
    bootstrap = parent.VERIFIED_EXEC_BOOTSTRAP_CODE.replace(
        '_ROOT = "/Users/idantal/ShermanQC/FullProject"',
        f"_ROOT = {json.dumps(str(synthetic_root))}",
        1,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            bootstrap,
            "parent",
            str(candidate),
            "0" * 64,
            str(len(candidate_bytes)),
            "--source-check-only",
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 1
    assert marker.exists() is False
    assert json.loads(completed.stdout)["status"] == "FAIL_CLOSED"


def test_direct_script_path_is_deauthorized_even_with_exact_arguments() -> None:
    safe_command = _source_args("--source-check-only")
    direct = [sys.executable, "-I", "-S", str(SOURCE), *safe_command[9:]]
    completed = subprocess.run(
        direct, capture_output=True, check=False, timeout=30
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "FAIL_CLOSED"


def test_external_metadata_size_caps_reject_before_any_file_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_calls = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("oversize declaration reached os.read")

    monkeypatch.setattr(parent.os, "read", forbidden_read)
    monkeypatch.setattr(parent, "_validate_context", lambda *_args: None)
    with pytest.raises(PermissionError):
        parent.authorize_preflight_once(
            _context(),
            authorization_sha256="0" * 64,
            authorization_size_bytes=parent.MAX_AUTHORIZATION_BYTES + 1,
        )
    with pytest.raises(parent.RunnerFailure):
        parent._binding_from_attestation(
            {
                "kind": parent.QUALIFICATION_KIND,
                "path": str(parent.C14_QUALIFICATION.relative_to(parent.ROOT)),
                "sha256": "0" * 64,
                "size_bytes": parent.MAX_BOUND_SOURCE_OR_CONTROL_BYTES + 1,
            },
            role="CANDIDATE14_QUALIFICATION_RECORD",
            expected_path=parent.C14_QUALIFICATION,
            typed=True,
            expected_kind=parent.QUALIFICATION_KIND,
        )
    oversized_args = _source_args("--source-check-only")[9:]
    test_size_index = oversized_args.index("--runner-tests-size-bytes") + 1
    oversized_args[test_size_index] = str(parent.MAX_RUNNER_TEST_SOURCE_BYTES + 1)
    with pytest.raises(ValueError):
        parent._parse_args(oversized_args)
    assert read_calls == 0


def test_parent_descriptor_bytes_exactly_match_worker_protocol() -> None:
    vectors = _vectors()
    parent_prelude, parent_commitment = parent.build_descriptor_prelude(vectors)
    worker_records = tuple(
        worker.PoolDescriptor(
            ordinal=vector.record.ordinal,
            pass_code=(1 if vector.record.pass_name == "primary" else 2),
            size_bytes=vector.size_bytes,
            sha256=vector.record.pool_sha256,
        )
        for vector in vectors
    )
    assert parent_prelude == worker.encode_descriptor_prelude(worker_records)
    assert parent_commitment == worker.batch_descriptor_commitment(worker_records)


def test_synthetic_manifest_exact36_order_and_pool_only_projection() -> None:
    qualification = _qualification()
    records = []
    for pair_index, case_id in enumerate(qualification.train_ids, 1):
        runtime = f"T{pair_index:03d}"
        pool_sha = f"{pair_index + 100:064x}"
        semantic_sha = f"{pair_index + 200:064x}"
        for pass_name in ("primary", "replay"):
            records.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "pool_path": str(
                        (
                            parent.ALLOWED_CACHE_ROOT
                            / pass_name
                            / runtime
                            / "pool"
                            / f"{runtime}.json"
                        ).relative_to(parent.ROOT)
                    ),
                    "pool_sha256": pool_sha,
                    "runtime_case_id": runtime,
                    "scan_sha256": qualification.scan_by_case[case_id],
                    "semantic_path": str(
                        (
                            parent.ALLOWED_CACHE_ROOT
                            / pass_name
                            / runtime
                            / "semantic_result.json"
                        ).relative_to(parent.ROOT)
                    ),
                    "semantic_sha256": semantic_sha,
                }
            )
    manifest = {
        "case_count": 18,
        "holdout_entry_count": 0,
        "kind": parent.CACHE_MANIFEST_KIND,
        "record_count": 36,
        "records": records,
        "schema_version": 1,
        "source_only": True,
        "split_sha256": parent.FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_allowlist_sha256": parent.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_set_commitment_sha256": qualification.train_commitment_sha256,
        "v51_scan_only_manifest_sha256": parent.V51_SCAN_ONLY_MANIFEST_SHA256,
    }
    parsed = parent._validate_full_manifest(
        manifest, qualification, manifest_binding_sha256="9" * 64
    )
    assert len(parsed) == 36
    assert tuple(record.ordinal for record in parsed) == tuple(range(1, 37))
    assert tuple(record.pass_name for record in parsed[:4]) == (
        "primary",
        "replay",
        "primary",
        "replay",
    )


def test_guarded_reader_denies_semantic_surface_before_stat() -> None:
    reader = parent.GuardedReader()
    with pytest.raises(PermissionError):
        reader.authorize(
            parent.ALLOWED_CACHE_ROOT / "primary/T001/semantic_result.json",
            purpose="CANDIDATE14_PRIMARY_00_SEMANTIC",
            sha256="a" * 64,
            size_bytes=1,
        )
    assert reader.denied_attempts


@pytest.mark.parametrize("completed_stage", ["raw", "inode", "started"])
def test_partial_read_preserves_raw_inode_started_truth(
    completed_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = parent.GuardedReader()
    path = parent.ROOT / "scripts/synthetic-c14-partial.json"
    purpose = "CANDIDATE14_SYNTHETIC_PARTIAL"
    identity = (111, 222)
    reader.allowed[path] = {
        "identity": identity,
        "purpose": purpose,
        "sha256": "0" * 64,
        "size_bytes": 1,
    }

    def failing_read(
        _path: Path,
        *,
        role: str,
        expected_size: int,
        after_raw_fd_open=None,
        after_verified_inode=None,
        before_first_read=None,
        after_exact_bytes_read=None,
    ):
        assert role == purpose
        assert expected_size == 1
        after_raw_fd_open()
        if completed_stage in {"inode", "started"}:
            after_verified_inode(identity)
        if completed_stage == "started":
            before_first_read(identity)
        assert after_exact_bytes_read is None
        raise OSError("synthetic partial read")

    monkeypatch.setattr(parent, "_read_regular_file_once", failing_read)
    with pytest.raises(OSError):
        reader.read_bytes(path, purpose=purpose)
    assert reader.raw_fd_opened_purposes == [purpose]
    assert reader.verified_inode_purposes == (
        [purpose] if completed_stage in {"inode", "started"} else []
    )
    assert reader.read_started_purposes == (
        [purpose] if completed_stage == "started" else []
    )
    assert reader.entries == []


@pytest.mark.parametrize("mutation", ["initial_oversize", "same_inode_growth"])
def test_guarded_file_reader_is_exact_size_bounded_before_and_during_read(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "synthetic-control.json"
    candidate.write_bytes(b"A" if mutation == "same_inode_growth" else b"AA")
    monkeypatch.setattr(parent, "ROOT", tmp_path)
    real_read = parent.os.read
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

    monkeypatch.setattr(parent.os, "read", bounded_read)
    with pytest.raises(ValueError):
        parent._read_regular_file_once(
            candidate, role="synthetic bounded control", expected_size=1
        )
    if mutation == "initial_oversize":
        assert read_sizes == []
    else:
        assert read_sizes == [1, 1]


def test_componentwise_directory_open_rejects_symlinked_intermediate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "redirected").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(parent, "ROOT", root)
    with pytest.raises(OSError):
        parent._open_directory(root / "redirected", role="synthetic redirected parent")


def test_manifest_raw_read_truth_precedes_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    progress = parent.ReadProgress()
    reader = authorization.reader

    monkeypatch.setattr(parent, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(parent, "revalidate_anchor", lambda *_args: None)
    monkeypatch.setattr(reader, "authorize", lambda *_args, **_kwargs: None)

    def read_invalid_manifest(*_args, after_exact_bytes_read=None, **_kwargs):
        after_exact_bytes_read()
        reader.entries.append(
            {
                "device_id": 1,
                "inode": 2,
                "path": str(parent.CACHE_MANIFEST.relative_to(parent.ROOT)),
                "purpose": "TRAIN_CACHE_MANIFEST",
                "sha256": parent.CACHE_MANIFEST_SHA256,
                "size_bytes": parent.CACHE_MANIFEST_SIZE_BYTES,
            }
        )
        return parent.canonical_bytes({})

    monkeypatch.setattr(reader, "read_bytes", read_invalid_manifest)
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.read_full_pool_inventory(
            context, authorization, object(), progress=progress
        )
    assert captured.value.stage == "CACHE_MANIFEST"
    assert progress.manifest_bytes_read_completed is True
    assert progress.manifest_contract_validated is False
    assert progress.all_36_pool_bytes_read_completed is False
    assert progress.pool_inventory_validated is False


def test_manifest_exact_bytes_truth_survives_postread_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    progress = parent.ReadProgress()
    reader = authorization.reader
    monkeypatch.setattr(parent, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(parent, "revalidate_anchor", lambda *_args: None)
    monkeypatch.setattr(reader, "authorize", lambda *_args, **_kwargs: None)

    def close_failed_read(*_args, after_exact_bytes_read=None, **_kwargs):
        after_exact_bytes_read()
        raise OSError("synthetic postread close failure")

    monkeypatch.setattr(reader, "read_bytes", close_failed_read)
    with pytest.raises(OSError, match="postread close failure"):
        parent.read_full_pool_inventory(
            context, authorization, object(), progress=progress
        )
    assert progress.manifest_bytes_read_completed is True
    assert progress.manifest_contract_validated is False
    assert progress.all_36_pool_bytes_read_completed is False
    assert progress.pool_inventory_validated is False


@pytest.mark.parametrize("last_outcome", ["replay_mismatch", "close_failure"])
def test_all_pool_raw_read_truth_precedes_replay_or_close_failure(
    last_outcome: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    progress = parent.ReadProgress()
    reader = authorization.reader
    records = tuple(vector.record for vector in _vectors())
    pool_calls = 0

    monkeypatch.setattr(parent, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(parent, "revalidate_anchor", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_full_manifest", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(reader, "authorize", lambda *_args, **_kwargs: None)

    inventory = types.SimpleNamespace(
        bindings={
            record.pool_path: types.SimpleNamespace(size_bytes=1)
            for record in records
        },
        commitment_sha256="7" * 64,
    )
    monkeypatch.setattr(
        reader, "install_artifact_inventory", lambda *_args, **_kwargs: inventory
    )

    def read_manifest_then_pools(
        path: Path, *, purpose: str, after_exact_bytes_read=None
    ):
        nonlocal pool_calls
        if purpose == "TRAIN_CACHE_MANIFEST":
            data = parent.canonical_bytes({})
            expected_sha = parent.CACHE_MANIFEST_SHA256
            expected_size = parent.CACHE_MANIFEST_SIZE_BYTES
        else:
            pool_calls += 1
            data = (
                b"B"
                if pool_calls == parent.RECORD_COUNT
                and last_outcome == "replay_mismatch"
                else b"A"
            )
            expected_sha = records[pool_calls - 1].pool_sha256
            expected_size = len(data)
        after_exact_bytes_read()
        if (
            purpose != "TRAIN_CACHE_MANIFEST"
            and pool_calls == parent.RECORD_COUNT
            and last_outcome == "close_failure"
        ):
            raise OSError("synthetic 36th-pool postread close failure")
        reader.entries.append(
            {
                "device_id": 1,
                "inode": len(reader.entries) + 1,
                "path": str(path.relative_to(parent.ROOT)),
                "purpose": purpose,
                "sha256": expected_sha,
                "size_bytes": expected_size,
            }
        )
        return data

    monkeypatch.setattr(reader, "read_bytes", read_manifest_then_pools)
    with pytest.raises((parent.RunnerFailure, OSError)) as captured:
        parent.read_full_pool_inventory(
            context, authorization, object(), progress=progress
        )
    if last_outcome == "replay_mismatch":
        assert isinstance(captured.value, parent.RunnerFailure)
        assert captured.value.stage == "POOL_INVENTORY"
    assert progress.manifest_bytes_read_completed is True
    assert progress.manifest_contract_validated is True
    assert progress.all_36_pool_bytes_read_completed is True
    assert progress.pool_inventory_validated is False
    assert progress.pool_inventory_commitment_sha256 == "7" * 64


def _anchor_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[parent.BootstrapContext, parent.AuthorizationSeal, parent.AnchorProgress]:
    root = tmp_path / "repo"
    research = root / "research"
    output = root / "output/research"
    research.mkdir(parents=True)
    output.mkdir(parents=True)
    monkeypatch.setattr(parent, "ROOT", root)
    monkeypatch.setattr(parent, "C14_ANCHOR", research / "anchor.json")
    monkeypatch.setattr(parent, "C14_AUTHORIZATION", research / "auth.json")
    monkeypatch.setattr(parent, "C14_QUALIFICATION", research / "qualification.json")
    monkeypatch.setattr(parent, "OUTPUT_ROOT", output / "candidate14")
    context = _context()
    authorization = _authorization(output)
    parent_fd, parent_identity = parent._open_directory(
        research, role="synthetic anchor parent"
    )
    os.close(parent_fd)
    object.__setattr__(authorization, "anchor_parent_identity", parent_identity)
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    monkeypatch.setattr(parent, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(parent, "_validate_preanchor_reader", lambda *_args: None)
    monkeypatch.setattr(parent, "revalidate_anchor", lambda *_args: None)
    return context, authorization, parent.AnchorProgress()


def test_anchor_o_excl_success_mode_and_retry_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, authorization, progress = _anchor_fixture(tmp_path, monkeypatch)
    gate_anchor_visibility: list[bool] = []
    seal = parent.consume_authorization(
        context,
        authorization,
        resource_check=lambda: gate_anchor_visibility.append(
            parent.C14_ANCHOR.exists()
        ),
        progress=progress,
    )
    assert gate_anchor_visibility == [False]
    assert progress.raw_entry_created_or_partial is True
    assert seal.path.read_bytes() == parent._anchor_payload(context, authorization)
    assert stat.S_IMODE(seal.path.stat().st_mode) == 0o400
    retry_progress = parent.AnchorProgress()
    with pytest.raises(FileExistsError):
        parent.consume_authorization(
            context,
            authorization,
            resource_check=lambda: None,
            progress=retry_progress,
        )
    assert retry_progress.raw_entry_created_or_partial is False


@pytest.mark.parametrize("failure_point", ["pre_open", "mid_write", "post_fsync", "close"])
def test_anchor_failure_truth_and_retry_barrier(
    failure_point: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, authorization, progress = _anchor_fixture(tmp_path, monkeypatch)
    real_open = parent.os.open
    real_write = parent.os.write
    real_fsync = parent.os.fsync
    real_close = parent.os.close
    anchor_fd: int | None = None
    write_calls = 0
    fsync_calls = 0
    close_injected = False

    def patched_open(path, flags, *args, **kwargs):
        nonlocal anchor_fd
        if path == parent.C14_ANCHOR.name and flags & os.O_EXCL:
            if failure_point == "pre_open":
                raise OSError("synthetic pre-open")
            anchor_fd = real_open(path, flags, *args, **kwargs)
            return anchor_fd
        return real_open(path, flags, *args, **kwargs)

    def patched_write(fd, data):
        nonlocal write_calls
        if fd == anchor_fd and failure_point == "mid_write":
            write_calls += 1
            if write_calls == 1:
                return max(1, len(data) // 2)
            raise OSError("synthetic mid-write")
        return real_write(fd, data)

    def patched_fsync(fd):
        nonlocal fsync_calls
        if failure_point == "post_fsync" and fd == anchor_fd:
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("synthetic post-fsync")
        return real_fsync(fd)

    def patched_close(fd):
        nonlocal close_injected
        if failure_point == "close" and fd == anchor_fd and not close_injected:
            close_injected = True
            real_close(fd)
            raise OSError("synthetic close")
        return real_close(fd)

    monkeypatch.setattr(parent.os, "open", patched_open)
    monkeypatch.setattr(parent.os, "write", patched_write)
    monkeypatch.setattr(parent.os, "fsync", patched_fsync)
    monkeypatch.setattr(parent.os, "close", patched_close)
    expected = OSError if failure_point == "pre_open" else parent.AnchorCreationFailure
    with pytest.raises(expected):
        parent.consume_authorization(
            context,
            authorization,
            resource_check=lambda: None,
            progress=progress,
        )
    assert progress.raw_entry_created_or_partial is (failure_point != "pre_open")
    if failure_point != "pre_open":
        assert parent.C14_ANCHOR.exists()


def test_resource_baseline_is_issued_and_rebase_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    parent._validate_resource_baseline(context, baseline)
    object.__setattr__(baseline, "parent_cpu_nanoseconds", 0)
    with pytest.raises(parent.RunnerFailure, match="RESOURCE_BASELINE_CHANGED"):
        parent._validate_resource_baseline(context, baseline)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("control_binding_commitment_sha256", "0" * 64),
        ("train_commitment_sha256", "0" * 64),
        ("resource_capability_result_commitment_sha256", "0" * 64),
        ("worker_source_result_commitment_sha256", "0" * 64),
        ("worker_source_inventory_commitment_sha256", "0" * 64),
        ("manifest_size_bytes", 1),
    ],
)
def test_qualification_independent_issuance_projection_rejects_rebase(
    field: str, replacement: object
) -> None:
    context = _context()
    qualification = _issued_qualification(context)
    parent._validate_qualification(context, qualification)
    object.__setattr__(qualification, field, replacement)
    with pytest.raises(PermissionError):
        parent._validate_qualification(context, qualification)


def test_qualification_reader_prefix_mutation_rejects() -> None:
    context = _context()
    reader = parent.GuardedReader()
    source_path = parent.ROOT / "scripts/fake-original.py"
    reader.entries.append(
        {
            "device_id": 11,
            "inode": 12,
            "path": str(source_path.relative_to(parent.ROOT)),
            "purpose": "CANDIDATE14_SOURCE_FAKE_ORIGINAL",
            "sha256": "1" * 64,
            "size_bytes": 1,
        }
    )
    qualification = _issued_qualification(context, reader)
    qualification.reader.entries.insert(
        0,
        {
            "device_id": 1,
            "inode": 2,
            "path": "scripts/fake.py",
            "purpose": "CANDIDATE14_FAKE",
            "sha256": "0" * 64,
            "size_bytes": 1,
        },
    )
    with pytest.raises(PermissionError):
        parent._validate_qualification(context, qualification)


def test_preanchor_reader_rejects_latent_manifest_capability() -> None:
    context = _context()
    qualification = _qualification()
    reader = qualification.reader
    purposes = [
        "CANDIDATE14_EXTERNAL_AUTHORIZATION",
        "CANDIDATE14_QUALIFICATION_RECORD",
        *parent._expected_control_purposes(),
    ]
    paths = [
        parent.C14_AUTHORIZATION,
        parent.C14_QUALIFICATION,
        *(value[0] for value in parent.FIXED_CONTROLS.values()),
        *(parent.CUSTODY_EVIDENCE_PATHS.values()),
    ]
    assert len(paths) == len(purposes)
    for index, (path, purpose) in enumerate(zip(paths, purposes), 1):
        identity = (1, index)
        reader.entries.append(
            {
                "device_id": identity[0],
                "inode": identity[1],
                "path": str(path.relative_to(parent.ROOT)),
                "purpose": purpose,
                "sha256": f"{index:064x}",
                "size_bytes": 1,
            }
        )
        reader.identities[identity] = path
        reader.consumed_paths.add(path)
        reader.raw_fd_opened_paths.add(path)
        reader.verified_inode_paths.add(path)
        reader.read_started_paths.add(path)
    reader.raw_fd_opened_purposes[:] = purposes
    reader.verified_inode_purposes[:] = purposes
    reader.read_started_purposes[:] = purposes
    authorization = _authorization(parent.ROOT, reader)
    object.__setattr__(authorization, "qualification", qualification)
    parent._validate_preanchor_reader(context, authorization)
    reader.allowed[parent.CACHE_MANIFEST] = {
        "identity": (9, 9),
        "purpose": "TRAIN_CACHE_MANIFEST",
        "sha256": parent.CACHE_MANIFEST_SHA256,
        "size_bytes": parent.CACHE_MANIFEST_SIZE_BYTES,
    }
    with pytest.raises(PermissionError):
        parent._validate_preanchor_reader(context, authorization)


def test_real_synthetic_worker_supervision_returns_one_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    outcome = parent.run_worker_supervised(
        context,
        _vectors(),
        pool_inventory_commitment_sha256="7" * 64,
        resource_baseline=baseline,
    )
    assert outcome.aggregate["stage_reached_all_bitset"] == "1111111111"
    assert outcome.aggregate["stage_passed_all_bitset"] == "1111111111"
    assert outcome.pool_inventory_commitment_sha256 == "7" * 64
    assert type(outcome.child_cpu_seconds) is float
    assert outcome.child_cpu_seconds >= 0.0
    assert type(outcome.child_maxrss_bytes) is int
    assert outcome.child_maxrss_bytes >= 0


def test_supervisor_skips_zero_length_blob_piece_without_stalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    vectors = list(_vectors())
    empty_sha = hashlib.sha256(b"").hexdigest()
    for index in (0, 1):
        original = vectors[index]
        record = parent.ManifestRecord(
            original.record.ordinal,
            original.record.pair_ordinal,
            original.record.pass_name,
            original.record.pool_path,
            empty_sha,
            original.record.manifest_record_binding_sha256,
        )
        vectors[index] = parent.PoolVector(
            record, original.file_binding_sha256, 0, b""
        )
    outcome = parent.run_worker_supervised(
        context,
        tuple(vectors),
        pool_inventory_commitment_sha256="7" * 64,
        resource_baseline=baseline,
    )
    assert outcome.aggregate["stage_passed_all_bitset"] != "1" * 10


@pytest.mark.parametrize("required_child_cpu", [0.0, 239.5])
def test_resource_gate_counts_all_reaped_children_and_post_worker_carry(
    required_child_cpu: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    baseline = parent.ResourceBaseline(
        parent.time.monotonic() - 1.0,
        123,
        0,
        0.0,
        context._token,
        object(),
        object(),
    )

    class FakeCpuSampler:
        def __init__(self, *_args, **_kwargs):
            pass

        def sample(self, _pid: int):
            return 123, 1_000_000_000

    class FakeMemorySampler:
        def sample_pid(self, _pid: int):
            return types.SimpleNamespace(
                resident_size_bytes=1,
                phys_footprint_bytes=1,
            )

    monkeypatch.setattr(parent, "_validate_resource_baseline", lambda *_args: None)
    monkeypatch.setattr(parent, "_DarwinCpuSampler", FakeCpuSampler)
    monkeypatch.setattr(parent, "_reaped_children_usage", lambda: (239.5, 0))
    monkeypatch.setattr(
        context.resource_core, "DarwinRusageSampler", FakeMemorySampler
    )
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.validate_authorized_resource_envelope(
            context,
            baseline,
            required_reaped_child_cpu_seconds=required_child_cpu,
            cumulative_child_maxrss_bytes=0,
        )
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert captured.value.code == "CUMULATIVE_COMBINED_CPU_LIMIT_EXCEEDED"


def test_supervision_setup_failure_reaps_spawned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    real_spawn = parent._spawn_worker_wait4_only
    spawned: list[parent._SpawnedWorker] = []

    def recording_spawn(command, process):
        real_spawn(command, process)
        spawned.append(process)

    monkeypatch.setattr(parent, "_spawn_worker_wait4_only", recording_spawn)
    monkeypatch.setattr(
        resource_core,
        "SampledDarwinCeiling",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic setup")
        ),
    )
    with pytest.raises(parent.RunnerFailure):
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
    )
    assert len(spawned) == 1
    assert spawned[0].returncode is not None


def test_worker_spawn_exec_failure_uses_posix_spawn_without_waitpid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waitpid_calls: list[tuple[object, ...]] = []

    def forbidden_waitpid(*args, **_kwargs):
        waitpid_calls.append(args)
        raise AssertionError("waitpid must not run on a posix_spawn exec failure")

    monkeypatch.setattr(parent.os, "waitpid", forbidden_waitpid)
    missing = tmp_path / "missing-python"
    with pytest.raises(FileNotFoundError):
        parent._spawn_worker_wait4_only(
            [str(missing), "-c", "pass"], parent._SpawnedWorker()
        )
    assert waitpid_calls == []


@pytest.mark.parametrize(
    "close_error",
    [OSError(errno.EIO, "synthetic post-spawn pipe close"), MemoryError()],
)
def test_post_spawn_pipe_close_failure_is_still_reaped_by_wait4(
    close_error: BaseException, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    real_spawn = parent.os.posix_spawn
    real_close = parent.os.close
    real_wait4 = parent.os.wait4
    spawned_pids: list[int] = []
    reaped_pids: list[int] = []
    fail_next_close = False

    def recording_spawn(*args, **kwargs):
        nonlocal fail_next_close
        pid = real_spawn(*args, **kwargs)
        spawned_pids.append(pid)
        fail_next_close = True
        return pid

    def one_failed_close(descriptor: int) -> None:
        nonlocal fail_next_close
        if fail_next_close:
            fail_next_close = False
            raise close_error
        real_close(descriptor)

    def recording_wait4(pid: int, options: int):
        result = real_wait4(pid, options)
        if result[0] != 0:
            reaped_pids.append(result[0])
        return result

    monkeypatch.setattr(parent.os, "posix_spawn", recording_spawn)
    monkeypatch.setattr(parent.os, "close", one_failed_close)
    monkeypatch.setattr(parent.os, "wait4", recording_wait4)
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "WORKER_EXECUTION"
    assert captured.value.code == "WORKER_SPAWN_PIPE_CLOSE_FAILED"
    assert len(spawned_pids) == 1
    assert reaped_pids == spawned_pids


def test_worker_spawn_sanitizes_inheritable_descriptor_before_child_creation(
    tmp_path: Path,
) -> None:
    descriptor = os.open(os.devnull, os.O_RDONLY)
    try:
        os.set_inheritable(descriptor, True)
        missing = tmp_path / "missing-python"
        with pytest.raises(FileNotFoundError):
            parent._spawn_worker_wait4_only(
                [str(missing), "-c", "pass"], parent._SpawnedWorker()
            )
        assert os.get_inheritable(descriptor) is False
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("program", "expected_stage"),
    [
        (
            "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(b'{}\\n')",
            "WORKER_RESULT",
        ),
        (
            "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(b'X'*70000)",
            "WORKER_RESULT",
        ),
        (
            "import sys;sys.stdin.buffer.read();sys.stderr.buffer.write(b'X'*5000)",
            "WORKER_EXECUTION",
        ),
        ("raise SystemExit(2)", "WORKER_EXECUTION"),
    ],
)
def test_supervisor_bounds_invalid_envelope_and_stream_failures(
    program: str, expected_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    monkeypatch.setattr(
        parent,
        "_worker_command",
        lambda: [sys.executable, "-c", program],
    )
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == expected_stage


@pytest.mark.parametrize(
    "program",
    [
        "raise SystemExit(2)",
        "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(b'{}\\n')",
    ],
)
def test_final_wall_gate_precedes_nonzero_or_invalid_worker_classification(
    program: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    monkeypatch.setattr(parent, "_validate_resource_baseline", lambda *_args: None)
    object.__setattr__(
        baseline,
        "wall_started_at",
        parent.time.monotonic() - parent.WALL_TIMEOUT_SECONDS - 1.0,
    )
    monkeypatch.setattr(
        parent, "_worker_command", lambda: [sys.executable, "-c", program]
    )
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert captured.value.code == "WALL_TIMEOUT"


def test_supervisor_rejects_worker_echo_of_wrong_batch_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    aggregate = _all_pass_aggregate()
    envelope = parent.canonical_bytes(
        {
            "batch_commitment_sha256": "0" * 64,
            "blob_inventory_complete": True,
            "c13_aggregate": aggregate,
            "c13_aggregate_commitment_sha256": parent.sha256_bytes(
                parent.canonical_bytes(aggregate)
            ),
            "descriptor_inventory_complete": True,
            "input_eof_verified": True,
            "kind": "sherman_qc_f1_aug30_candidate14_worker_aggregate_envelope_v1",
            "schema_version": 1,
        }
    )
    program = (
        "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write("
        + repr(envelope)
        + ")"
    )
    monkeypatch.setattr(
        parent, "_worker_command", lambda: [sys.executable, "-c", program]
    )
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "WORKER_PROTOCOL"


def test_supervisor_kill_failure_has_bounded_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    real_spawn = parent._spawn_worker_wait4_only
    real_kill = parent.os.kill
    spawned: list[parent._SpawnedWorker] = []

    def recording_spawn(command, process):
        real_spawn(command, process)
        spawned.append(process)

    monkeypatch.setattr(
        parent, "_worker_command", lambda: [sys.executable, "-c", "import time;time.sleep(10)"]
    )
    monkeypatch.setattr(parent, "_spawn_worker_wait4_only", recording_spawn)
    monkeypatch.setattr(
        resource_core,
        "SampledDarwinCeiling",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic setup")
        ),
    )
    monkeypatch.setattr(parent, "WORKER_REAP_GRACE_SECONDS", 0.03)
    monkeypatch.setattr(parent, "WORKER_REAP_POLL_SECONDS", 0.001)
    monkeypatch.setattr(
        parent.os,
        "kill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("synthetic kill denial")
        ),
    )
    try:
        with pytest.raises(parent.RunnerFailure) as captured:
            parent.run_worker_supervised(
                context,
                _vectors(),
                pool_inventory_commitment_sha256="7" * 64,
                resource_baseline=baseline,
            )
        assert captured.value.code in {
            "MEMORY_MONITOR_INITIALIZATION_FAILED",
            "WORKER_KILL_FAILED",
            "WORKER_SUPERVISION_FAILED",
        }
        assert len(spawned) == 1
        assert spawned[0].returncode is None
        assert spawned[0] in parent._UNREAPED_PROCESS_QUARANTINE
    finally:
        monkeypatch.setattr(parent.os, "kill", real_kill)
        for process in spawned:
            real_kill(process.pid, signal.SIGKILL)
            os.waitpid(process.pid, 0)
            process.returncode = -signal.SIGKILL
        parent._UNREAPED_PROCESS_QUARANTINE.clear()


def test_in_loop_failure_with_kill_denial_reaches_bounded_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    real_spawn = parent._spawn_worker_wait4_only
    real_kill = parent.os.kill
    spawned: list[parent._SpawnedWorker] = []

    def recording_spawn(command, process):
        real_spawn(command, process)
        spawned.append(process)

    program = (
        "import sys,time;"
        "sys.stdout.buffer.write(b'X'*70000);sys.stdout.buffer.flush();"
        "time.sleep(10)"
    )
    monkeypatch.setattr(
        parent, "_worker_command", lambda: [sys.executable, "-c", program]
    )
    monkeypatch.setattr(parent, "_spawn_worker_wait4_only", recording_spawn)
    monkeypatch.setattr(parent, "WORKER_REAP_GRACE_SECONDS", 0.03)
    monkeypatch.setattr(parent, "WORKER_REAP_POLL_SECONDS", 0.001)
    monkeypatch.setattr(
        parent.os,
        "kill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("synthetic kill denial")
        ),
    )
    try:
        with pytest.raises(parent.RunnerFailure) as captured:
            parent.run_worker_supervised(
                context,
                _vectors(),
                pool_inventory_commitment_sha256="7" * 64,
                resource_baseline=baseline,
            )
        assert captured.value.stage in {"WORKER_RESULT", "WORKER_EXECUTION"}
        assert len(spawned) == 1
        assert spawned[0].returncode is None
        assert spawned[0] in parent._UNREAPED_PROCESS_QUARANTINE
    finally:
        monkeypatch.setattr(parent.os, "kill", real_kill)
        for process in spawned:
            real_kill(process.pid, signal.SIGKILL)
            os.waitpid(process.pid, 0)
            process.returncode = -signal.SIGKILL
        parent._UNREAPED_PROCESS_QUARANTINE.clear()


def test_process_lookup_plus_wait4_error_still_quarantines_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    real_spawn = parent._spawn_worker_wait4_only
    real_kill = parent.os.kill
    spawned: list[parent._SpawnedWorker] = []

    def recording_spawn(command, process):
        real_spawn(command, process)
        spawned.append(process)

    monkeypatch.setattr(
        parent, "_worker_command", lambda: [sys.executable, "-c", "import time;time.sleep(10)"]
    )
    monkeypatch.setattr(parent, "_spawn_worker_wait4_only", recording_spawn)
    monkeypatch.setattr(
        resource_core,
        "SampledDarwinCeiling",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic setup")
        ),
    )
    monkeypatch.setattr(parent, "WORKER_REAP_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(parent, "WORKER_REAP_POLL_SECONDS", 0.001)
    monkeypatch.setattr(
        parent.os,
        "kill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(
        parent.os,
        "wait4",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ChildProcessError()),
    )
    try:
        with pytest.raises(parent.RunnerFailure):
            parent.run_worker_supervised(
                context,
                _vectors(),
                pool_inventory_commitment_sha256="7" * 64,
                resource_baseline=baseline,
            )
        assert len(spawned) == 1
        assert spawned[0] in parent._UNREAPED_PROCESS_QUARANTINE
        assert spawned[0].stdin is not None and spawned[0].stdin.closed
        assert spawned[0].stdout is not None and spawned[0].stdout.closed
        assert spawned[0].stderr is not None and spawned[0].stderr.closed
    finally:
        monkeypatch.setattr(parent.os, "kill", real_kill)
        for process in spawned:
            real_kill(process.pid, signal.SIGKILL)
            os.waitpid(process.pid, 0)
            process.returncode = -signal.SIGKILL
        parent._UNREAPED_PROCESS_QUARANTINE.clear()


@pytest.mark.parametrize("field_value", [-1.0, float("nan")])
def test_supervisor_rejects_invalid_final_child_rusage(
    field_value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    real_wait4 = parent.os.wait4

    def invalid_wait4(pid: int, options: int):
        waited_pid, status, usage = real_wait4(pid, options)
        if waited_pid == 0:
            return waited_pid, status, usage
        invalid = types.SimpleNamespace(
            ru_utime=field_value,
            ru_stime=usage.ru_stime,
            ru_maxrss=usage.ru_maxrss,
        )
        return waited_pid, status, invalid

    monkeypatch.setattr(parent.os, "wait4", invalid_wait4)
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert captured.value.code == "FINAL_RESOURCE_RUSAGE_INVALID"


def test_final_combined_maxrss_overrides_earlier_worker_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    monkeypatch.setattr(
        parent,
        "_worker_command",
        lambda: [
            sys.executable,
            "-c",
            "import sys;sys.stdout.buffer.write(b'X'*70000)",
        ],
    )
    real_wait4 = parent.os.wait4
    real_getrusage = parent.resource.getrusage
    each_peak = int(parent.MEMORY_CEILING_BYTES * 0.6)

    def high_child_wait4(pid: int, options: int):
        waited_pid, status, usage = real_wait4(pid, options)
        if waited_pid == 0:
            return waited_pid, status, usage
        return waited_pid, status, types.SimpleNamespace(
            ru_utime=usage.ru_utime,
            ru_stime=usage.ru_stime,
            ru_maxrss=each_peak,
        )

    def high_parent_peak(who: int):
        usage = real_getrusage(who)
        if who != parent.resource.RUSAGE_SELF:
            return usage
        return types.SimpleNamespace(
            ru_utime=usage.ru_utime,
            ru_stime=usage.ru_stime,
            ru_maxrss=each_peak,
        )

    monkeypatch.setattr(parent.os, "wait4", high_child_wait4)
    monkeypatch.setattr(parent.resource, "getrusage", high_parent_peak)
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert captured.value.code == "FINAL_COMBINED_MAXRSS_EXCEEDED"


def test_memory_failure_kill_path_does_not_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    calls: list[str] = []

    class FakeCeiling:
        def __init__(self, *_args, **_kwargs):
            pass

        def sample_and_enforce(self):
            calls.append("sample")
            return "SAMPLED_CEILING_EXCEEDED_KILL_SENT"

    monkeypatch.setattr(resource_core, "SampledDarwinCeiling", FakeCeiling)
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert calls == ["sample"]


@pytest.mark.parametrize("failure_point", ["initialize", "sample_or_kill"])
def test_memory_monitor_dependency_failures_remain_resource_terminal(
    failure_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    baseline = parent.begin_authorized_resource_envelope(context)

    class FailedMonitor:
        def __init__(self, *_args, **_kwargs):
            if failure_point == "initialize":
                raise RuntimeError("synthetic monitor initialization")

        def sample_and_enforce(self):
            raise RuntimeError("synthetic libproc sample or kill")

    monkeypatch.setattr(resource_core, "SampledDarwinCeiling", FailedMonitor)
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.run_worker_supervised(
            context,
            _vectors(),
            pool_inventory_commitment_sha256="7" * 64,
            resource_baseline=baseline,
        )
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert captured.value.code == (
        "MEMORY_MONITOR_INITIALIZATION_FAILED"
        if failure_point == "initialize"
        else "MEMORY_SAMPLE_OR_KILL_FAILED"
    )


def test_cpu_and_reaped_child_dependency_failures_are_resource_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parent.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic libproc load")
        ),
    )
    with pytest.raises(parent.RunnerFailure) as initialized:
        parent._DarwinCpuSampler(resource_core)
    assert initialized.value.stage == "RESOURCE_MONITOR"
    assert initialized.value.code == "CPU_SAMPLER_INITIALIZATION_FAILED"

    sampler = object.__new__(parent._DarwinCpuSampler)
    sampler._record_type = resource_core.RUsageInfoV2
    sampler._function = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic ctypes call")
    )
    with pytest.raises(parent.RunnerFailure) as sampled:
        sampler.sample(os.getpid())
    assert sampled.value.stage == "RESOURCE_MONITOR"
    assert sampled.value.code == "CPU_SAMPLE_DEPENDENCY_FAILED"

    monkeypatch.setattr(
        parent.resource,
        "getrusage",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("synthetic children rusage")
        ),
    )
    with pytest.raises(parent.RunnerFailure) as children:
        parent._reaped_children_usage()
    assert children.value.stage == "RESOURCE_MONITOR"
    assert children.value.code == "REAPED_CHILD_USAGE_SAMPLE_FAILED"


def test_cumulative_memory_sampler_exception_is_normalized_to_resource_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(parent, "_validate_context", lambda *_args: None)
    baseline = parent.begin_authorized_resource_envelope(context)
    monkeypatch.setattr(
        context.resource_core,
        "DarwinRusageSampler",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic memory sampler")),
    )
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.validate_authorized_resource_envelope(context, baseline)
    assert captured.value.stage == "RESOURCE_MONITOR"
    assert captured.value.code == "CUMULATIVE_RESOURCE_SAMPLE_FAILED"


def test_supervisor_uses_wait4_only_and_declares_cadence_bound() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    supervised = source[
        source.index("def run_worker_supervised") : source.index("SUCCESS_RESULT_KEYS")
    ]
    assert ".poll(" not in supervised
    assert ".wait(" not in supervised
    assert "process.kill(" not in supervised
    assert "os.wait4(" in supervised
    assert "os.WNOHANG" in supervised
    assert "SAMPLING_GAP_EXCEEDED" in supervised
    assert parent.POLL_INTERVAL_SECONDS == 0.010
    assert parent.MAX_SAMPLING_GAP_SECONDS == 0.050
    assert "sampling_started = time.monotonic()" in supervised
    assert supervised.index("sampling_started = time.monotonic()") < supervised.index(
        "_spawn_worker_wait4_only("
    )
    spawn = source[
        source.index("def _spawn_worker_wait4_only") : source.index(
            "def run_worker_supervised"
        )
    ]
    assert "os.posix_spawn(" in spawn
    assert "subprocess.Popen(" not in spawn
    assert "file_actions=file_actions" in spawn


def test_partial_anchor_terminal_separates_durable_from_receipt() -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="ANCHOR_CREATION",
        anchor_durable=True,
        manifest_bytes_read_completed=False,
        manifest_contract_validated=False,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
    )
    terminal = json.loads(payloads["terminal.json"])
    ledger = json.loads(payloads["read_ledger.json"])
    lock = json.loads(payloads["lock.json"])
    assert terminal["anchor_durable_entry_created_or_partial"] is True
    assert terminal["anchor_receipt_available"] is False
    assert ledger["anchor_durable_entry_created_or_partial"] is True
    assert ledger["anchor_receipt_available"] is False
    assert lock["anchor_durable_entry_created_or_partial"] is True
    assert lock["anchor_receipt_available"] is False


def test_partial_read_terminal_withholds_enumerable_prefix_commitments() -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    authorization.reader.raw_fd_opened_purposes.extend(
        parent._expected_pool_purposes()[:7]
    )
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="POOL_INVENTORY",
        anchor_durable=True,
        manifest_bytes_read_completed=True,
        manifest_contract_validated=True,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
        pool_inventory_commitment_sha256="7" * 64,
    )
    ledger = json.loads(payloads["read_ledger.json"])
    hidden_fields = (
        "raw_fd_opened_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_commitment_sha256",
        "verified_inode_commitment_sha256",
    )
    assert all(ledger[key] is None for key in hidden_fields)
    enumerable = {
        parent._purpose_commitment(
            "CANDIDATE14_FINAL_RAW_FD_OPENED_V1",
            authorization.reader.raw_fd_opened_purposes[:prefix],
        )
        for prefix in range(len(authorization.reader.raw_fd_opened_purposes) + 1)
    }
    assert not any(ledger[key] in enumerable for key in hidden_fields)

    ledger["raw_fd_opened_commitment_sha256"] = next(iter(enumerable))
    base = {
        "terminal.json": payloads["terminal.json"],
        "read_ledger.json": parent.canonical_bytes(ledger),
    }
    lock = parent._lock(context, authorization, None, base)
    lock["anchor_durable_entry_created_or_partial"] = True
    mutated = {**base, "lock.json": parent.canonical_bytes(lock)}
    with pytest.raises(parent.RunnerFailure):
        parent._validate_public_payloads(mutated)


def test_success_payloads_close_all_custody_and_worker_cross_bindings() -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    aggregate = _all_pass_aggregate()
    aggregate_bytes = parent.canonical_bytes(aggregate)
    consumption = parent.ConsumptionSeal(
        parent.C14_ANCHOR,
        "8" * 64,
        1,
        (1, 2),
        (1, 3),
        authorization.authorization_sha256,
        authorization.qualification_sha256,
        context._token,
        object(),
        object(),
    )
    outcome = parent.WorkerOutcome(
        MappingProxyType(aggregate),
        aggregate_bytes,
        "6" * 64,
        "7" * 64,
        0.0,
        0,
    )
    payloads = parent.build_success_payloads(
        context, authorization, consumption, outcome
    )
    result = json.loads(payloads["aggregate_result.json"])
    ledger = json.loads(payloads["read_ledger.json"])
    assert result["descriptor_commitment_sha256"] == "6" * 64
    assert result["pool_inventory_commitment_sha256"] == "7" * 64
    assert ledger["manifest_bytes_read_completed"] is True
    assert ledger["manifest_contract_validated"] is True
    assert ledger["all_36_pool_bytes_read_completed"] is True
    assert ledger["pool_inventory_validated"] is True
    assert ledger["descriptor_completed"] is True
    assert ledger["diagnostic_complete"] is True

    weakened_result = dict(result)
    weakened_ledger = dict(ledger)
    for key in (
        "manifest_bytes_read_completed",
        "manifest_contract_validated",
        "all_36_pool_bytes_read_completed",
        "pool_inventory_validated",
        "descriptor_completed",
    ):
        weakened_ledger[key] = False
    for key in (
        "descriptor_commitment_sha256",
        "pool_inventory_commitment_sha256",
    ):
        weakened_result[key] = None
        weakened_ledger[key] = None
    for key in (
        "raw_fd_opened_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_commitment_sha256",
        "verified_inode_commitment_sha256",
    ):
        weakened_ledger[key] = None
    weakened_base = {
        "aggregate_result.json": parent.canonical_bytes(weakened_result),
        "read_ledger.json": parent.canonical_bytes(weakened_ledger),
    }
    weakened_payloads = {
        **weakened_base,
        "lock.json": parent.canonical_bytes(
            parent._lock(
                context, authorization, consumption, weakened_base
            )
        ),
    }
    with pytest.raises(parent.RunnerFailure) as captured:
        parent._validate_public_payloads(weakened_payloads)
    assert captured.value.code == "SUCCESS_CROSS_BINDING_MISMATCH"

    mutations: tuple[tuple[str, object], ...] = (
        ("check_code", "C14_ARBITRARY_SUCCESS"),
        ("status", "ARBITRARY_SUCCESS"),
        ("candidate13_aggregate_check_code", "C13_ARBITRARY"),
        ("candidate13_aggregate_status", "ARBITRARY"),
        ("stage_contract_commitment_sha256", "0" * 64),
        ("exact_c9_success_failure_equivalent_for_all", "true"),
        ("stage_reached_all_bitset", "0" * 10),
        ("stage_passed_all_bitset", "1" * 9 + "0"),
    )
    for key, replacement in mutations:
        mutated_result = dict(result)
        mutated_result[key] = replacement
        mutated_base = {
            "aggregate_result.json": parent.canonical_bytes(mutated_result),
            "read_ledger.json": payloads["read_ledger.json"],
        }
        mutated_payloads = {
            **mutated_base,
            "lock.json": parent.canonical_bytes(
                parent._lock(context, authorization, consumption, mutated_base)
            ),
        }
        with pytest.raises(parent.RunnerFailure):
            parent._validate_public_payloads(mutated_payloads)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("adapter_or_hypothesis_semantics_modified", True),
        ("holdout_label_evaluator_metric_fit_full28_opened_or_started", True),
        ("engineering_attempt_id", "CANDIDATE14_OTHER"),
        ("scientific_hypothesis_id", "OTHER_HYPOTHESIS"),
    ],
)
def test_terminal_public_contract_is_exact_after_lock_rebuild(
    key: str, replacement: object
) -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="PAYLOAD",
        anchor_durable=False,
        manifest_bytes_read_completed=False,
        manifest_contract_validated=False,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
    )
    terminal = json.loads(payloads["terminal.json"])
    terminal[key] = replacement
    base = {
        "terminal.json": parent.canonical_bytes(terminal),
        "read_ledger.json": payloads["read_ledger.json"],
    }
    mutated = {
        **base,
        "lock.json": parent.canonical_bytes(
            parent._lock(context, authorization, None, base)
        ),
    }
    with pytest.raises(parent.RunnerFailure) as captured:
        parent._validate_public_payloads(mutated)
    assert captured.value.code == "TERMINAL_CROSS_BINDING_MISMATCH"


def test_post_read_worker_failure_retains_pool_commitment_without_descriptor() -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="WORKER_PROTOCOL",
        anchor_durable=True,
        manifest_bytes_read_completed=True,
        manifest_contract_validated=True,
        all_36_pool_bytes_read_completed=True,
        pool_inventory_validated=True,
        pool_inventory_commitment_sha256="7" * 64,
        descriptor_commitment_sha256=None,
    )
    ledger = json.loads(payloads["read_ledger.json"])
    assert ledger["all_36_pool_bytes_read_completed"] is True
    assert ledger["pool_inventory_validated"] is True
    assert ledger["pool_inventory_commitment_sha256"] == "7" * 64
    assert ledger["descriptor_completed"] is False
    assert ledger["descriptor_commitment_sha256"] is None


def test_public_cross_binding_mutation_rejects() -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="ANCHOR_CREATION",
        anchor_durable=True,
        manifest_bytes_read_completed=False,
        manifest_contract_validated=False,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
    )
    ledger = json.loads(payloads["read_ledger.json"])
    ledger["anchor_durable_entry_created_or_partial"] = False
    mutated = dict(payloads)
    mutated["read_ledger.json"] = parent.canonical_bytes(ledger)
    with pytest.raises(parent.RunnerFailure):
        parent._validate_public_payloads(mutated)

    impossible = json.loads(payloads["read_ledger.json"])
    impossible["manifest_bytes_read_completed"] = True
    impossible["manifest_contract_validated"] = True
    impossible["all_36_pool_bytes_read_completed"] = True
    impossible["pool_inventory_commitment_sha256"] = None
    impossible_payloads = dict(payloads)
    impossible_payloads["read_ledger.json"] = parent.canonical_bytes(impossible)
    with pytest.raises(parent.RunnerFailure):
        parent._validate_public_payloads(impossible_payloads)


@pytest.mark.parametrize(
    "stage", ["RESOURCE_MONITOR", "ANCHOR_INTEGRITY", "PAYLOAD"]
)
def test_publication_failure_summary_preserves_bounded_stage(stage: str) -> None:
    summary = parent._publication_failure_summary(
        parent.PublicationProgress(), stage=stage
    )
    assert summary["engineering_failure_stage"] == stage


def test_atomic_no_replace_publication_in_synthetic_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    research = root / "research"
    output_parent = root / "output/research"
    research.mkdir(parents=True)
    output_parent.mkdir(parents=True)
    monkeypatch.setattr(parent, "ROOT", root)
    monkeypatch.setattr(parent, "OUTPUT_ROOT", output_parent / "candidate14")
    monkeypatch.setattr(parent, "C14_AUTHORIZATION", research / "auth.json")
    monkeypatch.setattr(parent, "C14_QUALIFICATION", research / "qualification.json")
    context = _context()
    context = parent.BootstrapContext(
        context.resource_core,
        context.source_bindings,
        context.source_entries,
        context.source_commitment_sha256,
        context._resource_functions,
        context._sentinel,
        context._token,
    )
    authorization = _authorization(output_parent)
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="ANCHOR_CREATION",
        anchor_durable=True,
        manifest_bytes_read_completed=False,
        manifest_contract_validated=False,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
    )
    freshness = parent.prevalidate_fresh_output()
    commitment = parent.publish_atomically(payloads, freshness)
    assert len(commitment) == 64
    assert set(path.name for path in parent.OUTPUT_ROOT.iterdir()) == set(payloads)
    assert stat.S_IMODE(parent.OUTPUT_ROOT.stat().st_mode) == 0o500
    with pytest.raises(FileExistsError):
        parent.prevalidate_fresh_output()


def _publication_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, bytes], parent.FreshOutputSeal]:
    root = tmp_path / "repo"
    research = root / "research"
    output_parent = root / "output/research"
    research.mkdir(parents=True)
    output_parent.mkdir(parents=True)
    monkeypatch.setattr(parent, "ROOT", root)
    monkeypatch.setattr(parent, "OUTPUT_ROOT", output_parent / "candidate14")
    monkeypatch.setattr(parent, "C14_AUTHORIZATION", research / "auth.json")
    monkeypatch.setattr(parent, "C14_QUALIFICATION", research / "qualification.json")
    context = _context()
    authorization = _authorization(output_parent)
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="ANCHOR_CREATION",
        anchor_durable=True,
        manifest_bytes_read_completed=False,
        manifest_contract_validated=False,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
    )
    return payloads, parent.prevalidate_fresh_output()


def test_preanchor_freshness_rejects_staging_and_quarantine_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _payloads, _freshness = _publication_fixture(tmp_path, monkeypatch)
    for suffix in ("staging-foreign", "quarantine-foreign"):
        residue = parent.OUTPUT_ROOT.parent / f".{parent.OUTPUT_ROOT.name}.{suffix}"
        residue.mkdir()
        with pytest.raises(FileExistsError):
            parent.prevalidate_fresh_output()
        residue.rmdir()


@pytest.mark.parametrize("failure_phase", ["write", "fsync", "resource"])
def test_prerename_publication_failure_leaves_blocking_residue_truthfully(
    failure_phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)

    def failed_write(*_args, **_kwargs):
        raise OSError("synthetic staging write failure")

    def failed_fsync(*_args, **_kwargs):
        raise OSError("synthetic staging fsync failure")

    resource_check = None
    if failure_phase == "write":
        monkeypatch.setattr(parent.os, "write", failed_write)
    elif failure_phase == "fsync":
        monkeypatch.setattr(parent.os, "fsync", failed_fsync)
    else:
        def fail_after_staging_exists() -> None:
            if any(
                path.name.startswith(f".{parent.OUTPUT_ROOT.name}.staging-")
                for path in parent.OUTPUT_ROOT.parent.iterdir()
            ):
                raise parent.RunnerFailure(
                    "RESOURCE_MONITOR", "SYNTHETIC_PRERENAME_RESOURCE_FAILURE"
                )

        resource_check = fail_after_staging_exists

    progress = parent.PublicationProgress()
    with pytest.raises((OSError, parent.RunnerFailure)):
        parent.publish_atomically(
            payloads,
            freshness,
            resource_check=resource_check,
            progress=progress,
        )
    residues = tuple(
        path
        for path in parent.OUTPUT_ROOT.parent.iterdir()
        if path.name.startswith(f".{parent.OUTPUT_ROOT.name}.staging-")
    )
    assert len(residues) == 1
    assert progress.output_namespace_entry_created is False
    assert progress.output_verified_complete is False
    assert progress.cleanup_complete is False
    with pytest.raises(FileExistsError):
        parent.prevalidate_fresh_output()


def test_staged_output_inventory_rejects_extra_leaf_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_fchmod = parent.os.fchmod
    injected = False

    def inject_before_sealing(directory_fd: int, mode: int) -> None:
        nonlocal injected
        if mode == 0o700 and not injected:
            injected = True
            extra_fd = os.open(
                "foreign-extra.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
                dir_fd=directory_fd,
            )
            os.close(extra_fd)
        real_fchmod(directory_fd, mode)

    monkeypatch.setattr(parent.os, "fchmod", inject_before_sealing)
    progress = parent.PublicationProgress()
    with pytest.raises(OSError, match="inventory has extras"):
        parent.publish_atomically(payloads, freshness, progress=progress)
    assert progress.output_namespace_entry_created is False
    assert progress.output_verified_complete is False
    assert progress.cleanup_complete is False


def test_last_callback_stage_name_swap_cannot_publish_a_valid_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_fsync = parent.os.fsync
    parent_synced = False
    swapped = False

    def track_parent_sync(descriptor: int) -> None:
        nonlocal parent_synced
        real_fsync(descriptor)
        if parent._identity(os.fstat(descriptor)) == freshness.parent_identity:
            parent_synced = True

    def swap_at_last_callback() -> None:
        nonlocal swapped
        if not parent_synced or swapped:
            return
        stage = next(
            path
            for path in parent.OUTPUT_ROOT.parent.iterdir()
            if path.name.startswith(f".{parent.OUTPUT_ROOT.name}.staging-")
        )
        displaced = stage.with_name(stage.name + ".displaced")
        stage.rename(displaced)
        stage.mkdir(mode=0o700)
        (stage / "foreign.json").write_bytes(b"foreign")
        swapped = True

    monkeypatch.setattr(parent.os, "fsync", track_parent_sync)
    progress = parent.PublicationProgress()
    with pytest.raises(OSError, match="identity mismatch"):
        parent.publish_atomically(
            payloads,
            freshness,
            resource_check=swap_at_last_callback,
            progress=progress,
        )
    assert swapped is True
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is False
    assert progress.cleanup_complete is False
    assert not (parent.OUTPUT_ROOT / "lock.json").exists()


def test_retained_writable_leaf_mutation_is_rejected_before_lock_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_open = parent.os.open
    original_rename = parent._rename_noreplace
    retained_fd: int | None = None
    directory_renamed = False

    def retain_writable_leaf(path, flags, *args, **kwargs):
        nonlocal retained_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if (
            path == "terminal.json"
            and flags & os.O_ACCMODE == os.O_WRONLY
            and retained_fd is None
        ):
            retained_fd = os.dup(descriptor)
        return descriptor

    def mutate_after_directory_rename(
        source_name: str, destination_name: str, directory_fd: int
    ) -> None:
        nonlocal directory_renamed
        original_rename(source_name, destination_name, directory_fd)
        if not directory_renamed:
            directory_renamed = True
            assert retained_fd is not None
            os.lseek(retained_fd, 0, os.SEEK_SET)
            os.write(retained_fd, b"X")
            os.fsync(retained_fd)

    monkeypatch.setattr(parent.os, "open", retain_writable_leaf)
    monkeypatch.setattr(parent, "_rename_noreplace", mutate_after_directory_rename)
    progress = parent.PublicationProgress()
    try:
        with pytest.raises(OSError, match="renamed output leaf mismatch"):
            parent.publish_atomically(payloads, freshness, progress=progress)
    finally:
        if retained_fd is not None:
            os.close(retained_fd)
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is False
    assert progress.cleanup_complete is False
    assert not (parent.OUTPUT_ROOT / "lock.json").exists()


def test_parent_directory_rebind_after_rename_fails_lexical_commit_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    original_rename = parent._rename_noreplace
    moved_parent = parent.OUTPUT_ROOT.parent.with_name("research-moved")
    directory_renamed = False

    def move_parent_after_directory_rename(
        source_name: str, destination_name: str, directory_fd: int
    ) -> None:
        nonlocal directory_renamed
        original_rename(source_name, destination_name, directory_fd)
        if not directory_renamed:
            directory_renamed = True
            parent.OUTPUT_ROOT.parent.rename(moved_parent)
            parent.OUTPUT_ROOT.parent.mkdir(mode=0o700)

    monkeypatch.setattr(
        parent, "_rename_noreplace", move_parent_after_directory_rename
    )
    progress = parent.PublicationProgress()
    with pytest.raises(OSError, match="lexical parent changed"):
        parent.publish_atomically(payloads, freshness, progress=progress)
    assert directory_renamed is True
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is False
    assert progress.cleanup_complete is False
    assert not parent.OUTPUT_ROOT.exists()
    assert (
        moved_parent
        / parent.OUTPUT_ROOT.name
        / ".candidate14-lock.pending"
    ).exists()


def test_pending_lock_swap_at_final_callback_never_marks_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    swapped = False

    def swap_pending_postrename() -> None:
        nonlocal swapped
        pending = parent.OUTPUT_ROOT / ".candidate14-lock.pending"
        if swapped or not pending.exists():
            return
        displaced = pending.with_name(".candidate14-lock.displaced")
        pending.rename(displaced)
        pending.write_bytes(payloads["lock.json"])
        pending.chmod(0o400)
        displaced.unlink()
        swapped = True

    progress = parent.PublicationProgress()
    with pytest.raises(OSError, match="commit-edge leaf changed"):
        parent.publish_atomically(
            payloads,
            freshness,
            resource_check=swap_pending_postrename,
            progress=progress,
        )
    assert swapped is True
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is False
    assert progress.output_inventory_commitment_sha256 is None


def test_final_callback_payload_mutation_fails_full_commit_edge_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_open = parent.os.open
    retained_fd: int | None = None
    mutated = False

    def retain_ledger_writer(path, flags, *args, **kwargs):
        nonlocal retained_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if (
            path == "read_ledger.json"
            and flags & os.O_ACCMODE == os.O_WRONLY
            and retained_fd is None
        ):
            retained_fd = os.dup(descriptor)
        return descriptor

    def mutate_at_final_callback() -> None:
        nonlocal mutated
        if mutated or not parent.OUTPUT_ROOT.exists():
            return
        assert retained_fd is not None
        os.lseek(retained_fd, 0, os.SEEK_SET)
        os.write(retained_fd, b"X")
        os.fsync(retained_fd)
        mutated = True

    monkeypatch.setattr(parent.os, "open", retain_ledger_writer)
    progress = parent.PublicationProgress()
    try:
        with pytest.raises(OSError, match="commit-edge leaf changed"):
            parent.publish_atomically(
                payloads,
                freshness,
                resource_check=mutate_at_final_callback,
                progress=progress,
            )
    finally:
        if retained_fd is not None:
            os.close(retained_fd)
    assert mutated is True
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is False
    assert not (parent.OUTPUT_ROOT / "lock.json").exists()


def test_output_parent_inventory_is_streamed_bounded_and_callback_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _payloads, _freshness = _publication_fixture(tmp_path, monkeypatch)
    callbacks = 0
    yielded = 0

    class Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class Scanner:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal yielded
            if yielded > parent.MAX_OUTPUT_PARENT_ENTRIES:
                raise StopIteration
            yielded += 1
            return Entry(f"unrelated-{yielded}")

    def checked() -> None:
        nonlocal callbacks
        callbacks += 1

    monkeypatch.setattr(parent.os, "scandir", lambda _fd: Scanner())
    with pytest.raises(OSError) as captured:
        parent.prevalidate_fresh_output(integrity_check=checked)
    assert captured.value.errno == errno.E2BIG
    assert yielded == parent.MAX_OUTPUT_PARENT_ENTRIES + 1
    assert callbacks == yielded + 1


def test_postverify_close_failure_returns_truthful_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_open = parent.os.open
    real_close = parent.os.close
    committed_stage_fd: int | None = None
    injected = False

    def patched_open(path, flags, *args, **kwargs):
        nonlocal committed_stage_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if (
            type(path) is str
            and path.startswith(f".{parent.OUTPUT_ROOT.name}.staging-")
            and flags & getattr(os, "O_DIRECTORY", 0)
        ):
            committed_stage_fd = descriptor
        return descriptor

    def patched_close(fd):
        nonlocal injected
        if fd == committed_stage_fd and not injected:
            injected = True
            real_close(fd)
            raise OSError("synthetic final close")
        return real_close(fd)

    monkeypatch.setattr(parent.os, "open", patched_open)
    monkeypatch.setattr(parent.os, "close", patched_close)
    progress = parent.PublicationProgress()
    commitment = parent.publish_atomically(
        payloads, freshness, progress=progress
    )
    assert commitment == progress.output_inventory_commitment_sha256
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is True
    assert progress.cleanup_complete is False


@pytest.mark.parametrize("phase", ["staging", "readback"])
def test_leaf_close_failure_marks_cleanup_incomplete(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_open = parent.os.open
    real_close = parent.os.close
    target_fd: int | None = None
    injected = False

    def patched_open(path, flags, *args, **kwargs):
        nonlocal target_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        access_mode = flags & os.O_ACCMODE
        if (
            path == "terminal.json"
            and (
                (phase == "staging" and access_mode == os.O_WRONLY)
                or (phase == "readback" and access_mode == os.O_RDONLY)
            )
        ):
            target_fd = descriptor
        return descriptor

    def patched_close(fd):
        nonlocal injected
        if fd == target_fd and not injected:
            injected = True
            real_close(fd)
            raise OSError("synthetic leaf close failure")
        return real_close(fd)

    monkeypatch.setattr(parent.os, "open", patched_open)
    monkeypatch.setattr(parent.os, "close", patched_close)
    progress = parent.PublicationProgress()
    with pytest.raises(parent.RunnerFailure) as captured:
        parent.publish_atomically(payloads, freshness, progress=progress)
    assert captured.value.stage == "PAYLOAD"
    assert progress.cleanup_complete is False
    assert progress.output_namespace_entry_created is False
    assert progress.output_verified_complete is False


def test_publication_readback_uses_exact_length_plus_one_byte_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    real_open = parent.os.open
    real_read = parent.os.read
    target_fd: int | None = None
    read_sizes: list[int] = []

    def patched_open(path, flags, *args, **kwargs):
        nonlocal target_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "terminal.json" and flags & os.O_ACCMODE == os.O_RDONLY:
            target_fd = descriptor
        return descriptor

    def patched_read(descriptor: int, size: int) -> bytes:
        if descriptor == target_fd:
            read_sizes.append(size)
            if size == 1:
                return b"X"
        return real_read(descriptor, size)

    monkeypatch.setattr(parent.os, "open", patched_open)
    monkeypatch.setattr(parent.os, "read", patched_read)
    progress = parent.PublicationProgress()
    with pytest.raises(OSError, match="grew during readback"):
        parent.publish_atomically(payloads, freshness, progress=progress)
    assert read_sizes[-1] == 1
    assert len(read_sizes) == 2
    assert read_sizes[0] == len(payloads["terminal.json"])
    assert progress.output_namespace_entry_created is False
    assert progress.output_verified_complete is False


def test_postrename_resource_failure_prevents_lock_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    renamed = False
    original_rename = parent._rename_noreplace

    def rename_then_mark(source_name: str, destination_name: str, parent_fd: int) -> None:
        nonlocal renamed
        original_rename(source_name, destination_name, parent_fd)
        renamed = True

    monkeypatch.setattr(parent, "_rename_noreplace", rename_then_mark)

    def resource_check():
        if renamed:
            raise parent.RunnerFailure("RESOURCE_MONITOR", "SYNTHETIC_POST_RENAME")

    progress = parent.PublicationProgress()
    with pytest.raises(parent.RunnerFailure):
        parent.publish_atomically(
            payloads,
            freshness,
            resource_check=resource_check,
            progress=progress,
        )
    assert progress.output_namespace_entry_created is True
    assert progress.output_verified_complete is False
    assert progress.output_inventory_commitment_sha256 is None
    assert parent.OUTPUT_ROOT.exists()
    assert not (parent.OUTPUT_ROOT / "lock.json").exists()
    assert (parent.OUTPUT_ROOT / ".candidate14-lock.pending").exists()


def test_publication_integrity_callback_spans_write_rename_readback_and_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads, freshness = _publication_fixture(tmp_path, monkeypatch)
    renamed = False
    original_rename = parent._rename_noreplace
    phases: list[bool] = []

    def rename_then_mark(source_name: str, destination_name: str, parent_fd: int) -> None:
        nonlocal renamed
        original_rename(source_name, destination_name, parent_fd)
        renamed = True

    monkeypatch.setattr(parent, "_rename_noreplace", rename_then_mark)
    parent.publish_atomically(
        payloads,
        freshness,
        resource_check=lambda: phases.append(renamed),
    )
    assert phases.count(False) >= 10
    assert phases.count(True) >= 1


def test_terminal_publication_resource_reservation_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    qualification = _issued_qualification(context)
    monkeypatch.setattr(parent, "_validate_context", lambda _value: None)
    record = json.loads(parent.build_qualification_record(context, qualification))
    assert record[
        "terminal_publication_reserved_outside_compute_resource_envelope"
    ] is True
    assert record[
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available"
    ] is True
    assert record["verified_exec_bootstrap_commitment_sha256"] == hashlib.sha256(
        parent.VERIFIED_EXEC_BOOTSTRAP_CODE.encode("utf-8")
    ).hexdigest()
    authorization_contract = parent.build_authorization_contract(
        context,
        qualification,
        qualification_sha256="5" * 64,
        qualification_size_bytes=3,
    )
    authorization = _authorization(parent.ROOT)
    anchor = json.loads(parent._anchor_payload(context, authorization))
    payloads = parent.build_failure_payloads(
        context,
        authorization,
        None,
        stage="ANCHOR_CREATION",
        anchor_durable=True,
        manifest_bytes_read_completed=False,
        manifest_contract_validated=False,
        all_36_pool_bytes_read_completed=False,
        pool_inventory_validated=False,
    )
    ledger = json.loads(payloads["read_ledger.json"])
    assert ledger[
        "terminal_publication_reserved_outside_compute_resource_envelope"
    ] is True
    assert ledger[
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available"
    ] is True
    for value in (record, authorization_contract, anchor):
        assert value["combined_cpu_limit_seconds"] == "240.000"
        assert value["wall_timeout_seconds"] == "300.000"
        assert value["worker_cpu_soft_limit_seconds"] == 240
        assert type(value["worker_cpu_soft_limit_seconds"]) is int
        assert value["worker_cpu_hard_limit_seconds"] == 241
        assert type(value["worker_cpu_hard_limit_seconds"]) is int
    assert record["memory_ceiling_bytes"] == parent.MEMORY_CEILING_BYTES
    assert authorization_contract["memory_ceiling_bytes"] == parent.MEMORY_CEILING_BYTES
    assert ledger["memory_ceiling_bytes"] == parent.MEMORY_CEILING_BYTES
    assert ledger["combined_cpu_limit_seconds"] == "240.000"
    assert ledger["wall_timeout_seconds"] == "300.000"
    assert ledger["worker_cpu_soft_limit_seconds"] == 240
    assert ledger["worker_cpu_hard_limit_seconds"] == 241

    mutations = (
        ("memory_ceiling_bytes", parent.MEMORY_CEILING_BYTES - 1),
        ("combined_cpu_limit_seconds", "239.000"),
        ("wall_timeout_seconds", "299.000"),
        ("worker_cpu_soft_limit_seconds", 239),
        ("worker_cpu_hard_limit_seconds", 240),
    )
    for key, replacement in mutations:
        mutated_ledger = dict(ledger)
        mutated_ledger[key] = replacement
        base = {
            "terminal.json": payloads["terminal.json"],
            "read_ledger.json": parent.canonical_bytes(mutated_ledger),
        }
        mutated = {
            **base,
            "lock.json": parent.canonical_bytes(
                parent._lock(context, authorization, None, base)
            ),
        }
        with pytest.raises(parent.RunnerFailure) as captured:
            parent._validate_public_payloads(mutated)
        assert captured.value.code == "FIXED_PUBLIC_CONTRACT_MISMATCH"


def test_one_shot_full_resource_gate_precedes_anchor_consumption() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    flow = source[source.index("def run_one_shot") : source.index("def _bootstrap_cli_value")]
    authority = flow.index("authorization = authorize_preflight_once(")
    resource_gate = flow.index(
        "validate_authorized_resource_envelope(context, resource_baseline)",
        authority,
    )
    consume = flow.index("consumption = consume_authorization(")
    assert authority < resource_gate < consume
    consume_source = source[
        source.index("def consume_authorization") : source.index(
            "def revalidate_anchor"
        )
    ]
    edge_gate = consume_source.index("resource_check()")
    create = consume_source.index("descriptor = os.open(")
    parent_identity = consume_source.index(
        "parent_identity != authorization.anchor_parent_identity"
    )
    assert parent_identity < edge_gate < create
    callback = flow.index("def publication_integrity_check()")
    assert flow.index("revalidate_anchor(context, consumption)", callback) > callback
    late_window = flow[
        flow.index('is_success = "aggregate_result.json" in payloads') : callback
    ]
    assert "revalidate_anchor(context, consumption)" not in late_window


@pytest.mark.parametrize("failure_point", ["anchor", "pool"])
def test_terminal_transition_resource_gate_overrides_prior_failure_stage(
    failure_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context()
    authorization = _authorization(parent.ROOT)
    freshness = parent.FreshOutputSeal(parent.OUTPUT_ROOT.parent, (1, 1), object())
    consumption = parent.ConsumptionSeal(
        parent.C14_ANCHOR,
        "8" * 64,
        1,
        (1, 2),
        (1, 3),
        authorization.authorization_sha256,
        authorization.qualification_sha256,
        context._token,
        object(),
        object(),
    )
    gate_calls = 0
    failure_call = 2 if failure_point == "anchor" else 3

    def resource_gate(*_args, **_kwargs) -> None:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == failure_call:
            raise parent.RunnerFailure(
                "RESOURCE_MONITOR", "SYNTHETIC_TERMINAL_TRANSITION_LIMIT"
            )

    def consume(*_args, **_kwargs):
        if failure_point == "anchor":
            raise parent.AnchorCreationFailure("synthetic partial anchor")
        return consumption

    def publish(payloads, _freshness, *, progress, **_kwargs):
        progress.output_namespace_entry_created = True
        progress.output_verified_complete = True
        progress.output_inventory_commitment_sha256 = "9" * 64
        return "9" * 64

    monkeypatch.setattr(parent, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(parent, "prevalidate_fresh_output", lambda: freshness)
    monkeypatch.setattr(parent, "begin_authorized_resource_envelope", lambda *_args: object())
    monkeypatch.setattr(parent, "authorize_preflight_once", lambda *_args, **_kwargs: authorization)
    monkeypatch.setattr(parent, "validate_authorized_resource_envelope", resource_gate)
    monkeypatch.setattr(parent, "consume_authorization", consume)
    monkeypatch.setattr(parent, "revalidate_anchor", lambda *_args: None)
    monkeypatch.setattr(
        parent,
        "read_full_pool_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            parent.RunnerFailure("POOL_INVENTORY", "SYNTHETIC_POOL_FAILURE")
        ),
    )
    monkeypatch.setattr(parent, "publish_atomically", publish)
    result = parent.run_one_shot(
        context, authorization_sha256="4" * 64, authorization_size_bytes=3
    )
    assert result["engineering_failure_stage"] == "RESOURCE_MONITOR"
    assert result["diagnostic_complete"] is False


def test_no_duplicate_literal_dict_keys_in_parent_source() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and type(key.value) is str
        ]
        assert len(keys) == len(set(keys))


def test_source_contract_has_no_default_or_semantic_capability() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "required=True" not in source  # manual exact parser; no argparse fallback
    assert "--source-check-only" in source
    assert "--source-control-check-only" in source
    assert "--preflight-once" in source
    assert "semantic_result.json" in source  # explicitly denied/manifest-bound only
    assert "reader.authorize(semantic" not in source
    assert "resource.setrlimit(resource.RLIMIT_AS" not in source
