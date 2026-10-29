from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as hypothesis
from backend import f1_train18_candidate7_conservative_recall_recovery_v1 as runtime
from scripts import preflight_f1_train18_candidate7_semantic_pool_v1 as preflight


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "hypothesis": preflight.sha256_bytes(preflight.HYPOTHESIS_SOURCE.read_bytes()),
        "preflight": preflight.sha256_bytes(preflight.PREFLIGHT_SOURCE.read_bytes()),
        "preflight_tests": preflight.sha256_bytes(
            preflight.PREFLIGHT_TESTS_SOURCE.read_bytes()
        ),
        "runtime": preflight.sha256_bytes(preflight.RUNTIME_SOURCE.read_bytes()),
        "runtime_tests": preflight.sha256_bytes(
            preflight.RUNTIME_TESTS_SOURCE.read_bytes()
        ),
    }


def _semantic() -> bytes:
    return hypothesis.canonical_bytes(
        {
            "cad_used": False,
            "candidate_count": 1,
            "expected_count_used": False,
            "labels_used": False,
            "part_identity_used": False,
            "predictions": [
                {
                    "bend_id": "V51-B0",
                    "line_end": [100.0, 0.0, 0.0],
                    "line_start": [0.0, 0.0, 0.0],
                    "selection_route": "v51-test",
                    "selector_score": None,
                    "source_hypothesis_id": "V51-H0",
                    "source_kind": "plane_lattice",
                }
            ],
            "research_only": True,
            "source_only": True,
        }
    )


def _pool(*, invalid_truth_key: bool = False) -> bytes:
    row: dict[str, object] = {
        "candidate_id": "IGNORED-ID",
        "endpoint_support_fraction": 0.8,
        "finite_support_fraction": 0.8,
        "normal_second_eigenvalue": 0.2,
        "representative_source": "plane_lattice",
        "score": 0.8,
        "span_endpoints": [[0.0, 8.0, 0.0], [100.0, 8.0, 0.0]],
        "two_sided_transition_score": 0.8,
    }
    if invalid_truth_key:
        row["ground_truth"] = True
    return hypothesis.canonical_bytes(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
            "candidate_count": 99,
            "candidates": [{"candidate_id": "RAW-IGNORED"}],
            "inference_contract": {
                "cad_exposed": False,
                "expected_count_exposed": False,
                "manual_truth_exposed": False,
                "part_identifier_exposed": False,
            },
            "line_object_count": 1,
            "line_objects": [row],
            "schema_version": 3,
            "source_point_serialization": False,
            "working_point_budget": 200000,
        }
    )


def _entry(path: Path, purpose: str) -> dict[str, object]:
    data = path.read_bytes()
    metadata = os.lstat(path)
    return {
        "case_id": None,
        "device_id": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "path": str(path.relative_to(preflight.ROOT)),
        "purpose": purpose,
        "sha256": preflight.sha256_bytes(data),
        "size_bytes": len(data),
    }


def _context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> preflight.BootstrapContext:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "research").mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(
        preflight,
        "SOURCE_INVENTORY_CONSUMPTION_ANCHOR",
        root
        / "research/f1_aug30_train18_candidate7_source_inventory_consumption_v1.json",
    )
    monkeypatch.setattr(
        preflight,
        "OUTPUT_ROOT",
        root / "output/research/f1-aug30-train18-candidate7-semantic-pool-preflight-v1",
    )
    specs = (
        ("hypothesis", "CANDIDATE7_SOURCE_HYPOTHESIS"),
        ("preflight", "CANDIDATE7_SOURCE_PREFLIGHT"),
        ("preflight_tests", "CANDIDATE7_SOURCE_PREFLIGHT_TESTS"),
        ("runtime", "CANDIDATE7_SOURCE_RUNTIME"),
        ("runtime_tests", "CANDIDATE7_SOURCE_RUNTIME_TESTS"),
    )
    entries: list[dict[str, object]] = []
    bindings: dict[str, dict[str, object]] = {}
    for name, purpose in specs:
        path = root / f"{name}.py"
        path.write_bytes(f"# {name}\n".encode("ascii"))
        entries.append(_entry(path, purpose))
        bindings[name] = {
            "path": path.name,
            "sha256": (
                runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
                if name == "hypothesis"
                else preflight.sha256_bytes(path.read_bytes())
            ),
            "size_bytes": path.stat().st_size,
        }
    return preflight.BootstrapContext(
        runtime=runtime,
        hypothesis=hypothesis,
        source_bindings=bindings,
        source_bytes={name: b"" for name, _purpose in specs},
        bootstrap_entries=tuple(entries),
    )


def _synthetic_authorization_seal(
    context: preflight.BootstrapContext,
) -> preflight.SourceInventoryAuthorizationSeal:
    reader = preflight.GuardedReader(
        entries=[dict(entry) for entry in context.bootstrap_entries]
    )
    for filename, purpose in (
        (
            "source-inventory-authorization.json",
            "CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
        ),
        (
            "source-control-qualification.json",
            "CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
        ),
    ):
        path = preflight.ROOT / filename
        data = preflight.canonical_bytes({"synthetic": purpose})
        path.write_bytes(data)
        reader.authorize(
            path,
            purpose=purpose,
            sha256=preflight.sha256_bytes(data),
            size_bytes=len(data),
        )
        reader.read_bytes(path, purpose=purpose)
    return preflight.SourceInventoryAuthorizationSeal(
        reader=reader,
        authorization_sha256=str(reader.entries[-2]["sha256"]),
        qualification_record_sha256=str(reader.entries[-1]["sha256"]),
        _sentinel=preflight._AUTHORIZATION_SEAL_SENTINEL,
    )


def _install_actual_shaped_controls(
    context: preflight.BootstrapContext,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generic_binding_has_kind: bool = False,
    staged_binding_has_kind: bool = True,
) -> tuple[tuple[str, ...], dict[str, str], Path]:
    root = preflight.ROOT
    train_ids = tuple(f"TRAIN-X{index:02d}" for index in range(18))
    scan_by_case = {case_id: _sha(f"scan-{case_id}") for case_id in train_ids}
    split = {
        "kind": "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1",
        "label_or_outcome_fields_present": False,
        "train_case_ids": list(train_ids),
    }
    original = {"status": "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN"}
    allowlist = {
        "cases": [
            {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
            for case_id in train_ids
        ],
        "kind": "sherman_qc_f1_aug30_train_only_scan_allowlist_v1",
        "reject_any_case_not_listed": True,
    }
    disposition = (ROOT / runtime.CANDIDATE6_DISPOSITION_PATH).read_bytes()
    evidence_paths: dict[str, Path] = {}
    evidence_bindings: dict[str, dict[str, object]] = {}
    for prefix in ("access_evidence", "builder", "staged_validator_pass", "validator"):
        path = root / f"{prefix}.json"
        data = preflight.canonical_bytes({"value_free": prefix})
        path.write_bytes(data)
        binding: dict[str, object] = {
            "path": path.name,
            "sha256": preflight.sha256_bytes(data),
            "size_bytes": len(data),
        }
        if prefix == "staged_validator_pass" and staged_binding_has_kind:
            binding["kind"] = preflight.CUSTODY_VALIDATION_KIND
        elif prefix != "staged_validator_pass" and generic_binding_has_kind:
            binding["kind"] = f"unexpected-{prefix}"
        evidence_paths[prefix] = path
        evidence_bindings[f"{prefix}_binding"] = binding
    manifest_path = root / "cache-manifest.json"
    manifest_path.write_bytes(preflight.canonical_bytes({"placeholder": True}))
    manifest_sha = preflight.sha256_bytes(manifest_path.read_bytes())
    manifest_size = manifest_path.stat().st_size
    commitment = preflight.sha256_bytes(
        preflight.canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    control_values: dict[str, bytes] = {
        "SEALED_SPLIT": preflight.canonical_bytes(split),
        "ORIGINAL_PREREGISTRATION": preflight.canonical_bytes(original),
        "TRAIN_ALLOWLIST": preflight.canonical_bytes(allowlist),
        "CANDIDATE6_TERMINAL_DISPOSITION": disposition,
    }
    control_paths: dict[str, Path] = {}
    for purpose, data in control_values.items():
        path = root / f"{purpose.lower()}.json"
        path.write_bytes(data)
        control_paths[purpose] = path
    fixed = {
        purpose: (
            control_paths[purpose],
            preflight.sha256_bytes(data),
            len(data),
        )
        for purpose, data in control_values.items()
    }
    attestation = {
        **evidence_bindings,
        "blinded_complement_commitment_sha256": _sha("complement"),
        "canonical_json": True,
        "case_count": 18,
        "holdout_entry_count": 0,
        "holdout_hash_count": 0,
        "holdout_status_count": 0,
        "independent_validation_passed": True,
        "kind": preflight.CACHE_ATTESTATION_KIND,
        "manifest_binding": {
            "kind": preflight.CACHE_MANIFEST_KIND,
            "path": manifest_path.name,
            "sha256": manifest_sha,
            "size_bytes": manifest_size,
        },
        "optimizer_holdout_access_count": 0,
        "record_count": 36,
        "schema_version": 1,
        "split_sha256": fixed["SEALED_SPLIT"][1],
        "train_allowlist_sha256": fixed["TRAIN_ALLOWLIST"][1],
        "train_exact_match": True,
        "train_holdout_intersection_count": 0,
        "train_set_commitment_sha256": commitment,
    }
    attestation_bytes = preflight.canonical_bytes(attestation)
    attestation_path = root / "cache-attestation.json"
    attestation_path.write_bytes(attestation_bytes)
    fixed["TRAIN_CACHE_ATTESTATION"] = (
        attestation_path,
        preflight.sha256_bytes(attestation_bytes),
        len(attestation_bytes),
    )
    monkeypatch.setattr(preflight, "FIXED_CONTROLS", fixed)
    monkeypatch.setattr(preflight, "CUSTODY_EVIDENCE_PATHS", evidence_paths)
    monkeypatch.setattr(
        preflight,
        "CUSTODY_PASS_SHA256",
        str(evidence_bindings["staged_validator_pass_binding"]["sha256"]),
    )
    monkeypatch.setattr(preflight, "CACHE_MANIFEST", manifest_path)
    monkeypatch.setattr(preflight, "CACHE_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(preflight, "CACHE_MANIFEST_SIZE_BYTES", manifest_size)
    return train_ids, scan_by_case, manifest_path


def test_preflight_top_level_is_stdlib_only() -> None:
    tree = ast.parse(preflight.PREFLIGHT_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "scipy", "scripts", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_source_check_only_verifies_five_sources_without_controls() -> None:
    hashes = _source_hashes()
    completed = subprocess.run(
        [
            sys.executable,
            str(preflight.PREFLIGHT_SOURCE),
            "--runtime-sha256",
            hashes["runtime"],
            "--preflight-sha256",
            hashes["preflight"],
            "--runtime-tests-sha256",
            hashes["runtime_tests"],
            "--preflight-tests-sha256",
            hashes["preflight_tests"],
            "--hypothesis-sha256",
            hashes["hypothesis"],
            "--source-check-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "artifact_input_opened": False,
        "label_or_holdout_opened": False,
        "source_binding_count": 5,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_help_does_not_import_runtime_hypothesis_or_numeric_packages() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(preflight.PREFLIGHT_SOURCE), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "f1_train18_candidate6_conservative_recall_recovery_v1",
        "f1_train18_candidate7_conservative_recall_recovery_v1",
        "evaluate_f1_against_manual_gold",
        "numpy",
        "scipy",
        "sklearn",
    ):
        assert forbidden not in completed.stderr


def test_preloaded_symlink_alias_rejects_before_any_source_read(
    tmp_path: Path,
) -> None:
    hashes = _source_hashes()
    alias = tmp_path / "arbitrary_alias.py"
    alias.symlink_to(preflight.HYPOTHESIS_SOURCE)
    sentinel = tmp_path / "source-read"
    code = f"""
import importlib.util, pathlib, sys
alias = pathlib.Path({str(alias)!r})
spec = importlib.util.spec_from_file_location('arbitrary_alias', alias)
loaded = importlib.util.module_from_spec(spec)
sys.modules['arbitrary_alias'] = loaded
spec.loader.exec_module(loaded)
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec2 = importlib.util.spec_from_file_location('isolated_c7_preflight', path)
module = importlib.util.module_from_spec(spec2)
sys.modules['isolated_c7_preflight'] = module
spec2.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_read(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('READ')
    raise AssertionError('source read preceded alias rejection')
module._read_exact_source = forbidden_read
try:
    module.bootstrap_verified_sources(
        runtime_sha256={hashes['runtime']!r},
        preflight_sha256={hashes['preflight']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        preflight_tests_sha256={hashes['preflight_tests']!r},
        hypothesis_sha256={hashes['hypothesis']!r},
    )
except RuntimeError as error:
    assert 'preloaded source alias' in str(error)
else:
    raise AssertionError('preloaded source alias was accepted')
assert not pathlib.Path({str(sentinel)!r}).exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_tampered_source_hash_fails_before_verified_module_execution(
    tmp_path: Path,
) -> None:
    hashes = _source_hashes()
    sentinel = tmp_path / "module-executed"
    code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_c7_tamper', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_c7_tamper'] = module
spec.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_exec(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('EXECUTED')
    raise AssertionError('verified module executed after mismatch')
module._exec_verified_module = forbidden_exec
try:
    module.bootstrap_verified_sources(
        runtime_sha256={'1' * 64!r},
        preflight_sha256={hashes['preflight']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        preflight_tests_sha256={hashes['preflight_tests']!r},
        hypothesis_sha256={hashes['hypothesis']!r},
    )
except ValueError as error:
    assert 'binding mismatch' in str(error)
else:
    raise AssertionError('tampered binding was accepted')
assert not pathlib.Path({str(sentinel)!r}).exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_no_mode_rejects_before_source_control_or_publication() -> None:
    hashes = _source_hashes()
    completed = subprocess.run(
        [
            sys.executable,
            str(preflight.PREFLIGHT_SOURCE),
            "--runtime-sha256",
            hashes["runtime"],
            "--preflight-sha256",
            hashes["preflight"],
            "--runtime-tests-sha256",
            hashes["runtime_tests"],
            "--preflight-tests-sha256",
            hashes["preflight_tests"],
            "--hypothesis-sha256",
            hashes["hypothesis"],
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "one of the arguments" in completed.stderr
    assert "SOURCE_BYTES_VERIFIED_ONLY" not in completed.stdout


def test_source_inventory_mode_requires_external_authorization_sha() -> None:
    hashes = _source_hashes()
    completed = subprocess.run(
        [
            sys.executable,
            str(preflight.PREFLIGHT_SOURCE),
            "--runtime-sha256",
            hashes["runtime"],
            "--preflight-sha256",
            hashes["preflight"],
            "--runtime-tests-sha256",
            hashes["runtime_tests"],
            "--preflight-tests-sha256",
            hashes["preflight_tests"],
            "--hypothesis-sha256",
            hashes["hypothesis"],
            "--source-inventory-once",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "requires --authorization-sha256" in completed.stderr


def test_qualification_only_dispatch_cannot_call_run_or_publish(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hashes = {name: _sha(name) for name in _source_hashes()}
    context = object()
    monkeypatch.setattr(preflight, "bootstrap_verified_sources", lambda **_kwargs: context)
    monkeypatch.setattr(
        preflight,
        "qualification_result",
        lambda _context: {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "check_code": "ALL_SOURCE_CONTROL_CHECKS_PASSED",
            "failure_stage": None,
            "holdout_or_label_opened": False,
            "repeatable": True,
            "scientific_hypothesis_consumed": False,
            "source_control_qualification_complete": True,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
            "verified_control_read_inventory_complete": True,
        },
    )

    def forbidden(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("qualification-only dispatched a consuming capability")

    monkeypatch.setattr(preflight, "run", forbidden)
    monkeypatch.setattr(preflight, "publish_atomically", forbidden)
    monkeypatch.setattr(preflight, "authorize_source_inventory_once", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(preflight.PREFLIGHT_SOURCE),
            "--runtime-sha256",
            hashes["runtime"],
            "--preflight-sha256",
            hashes["preflight"],
            "--runtime-tests-sha256",
            hashes["runtime_tests"],
            "--preflight-tests-sha256",
            hashes["preflight_tests"],
            "--hypothesis-sha256",
            hashes["hypothesis"],
            "--source-control-check-only",
        ],
    )
    assert preflight.main() == 0
    assert "SOURCE_CONTROL_QUALIFIED_REPEATABLE" in capsys.readouterr().out


def test_binding_envelopes_are_conditionally_exact_actual_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(context, monkeypatch)
    qualification = preflight.qualify_source_controls(context)
    assert len(qualification.reader.entries) == 14
    assert all(
        entry["purpose"] != "TRAIN_CACHE_MANIFEST"
        for entry in qualification.reader.entries
    )
    value = preflight.qualification_result(context)
    assert value["status"] == "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
    assert value["artifact_source_inventory_opened"] is False
    assert value["repeatable"] is True


@pytest.mark.parametrize(
    "generic_has_kind,staged_has_kind,expected_code",
    (
        (True, True, "GENERIC_BINDING_ENVELOPE_SCHEMA_MISMATCH"),
        (False, False, "TYPED_BINDING_ENVELOPE_SCHEMA_MISMATCH"),
    ),
)
def test_wrong_generic_or_typed_binding_shape_fails_with_stable_code(
    generic_has_kind: bool,
    staged_has_kind: bool,
    expected_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(
        context,
        monkeypatch,
        generic_binding_has_kind=generic_has_kind,
        staged_binding_has_kind=staged_has_kind,
    )
    value = preflight.qualification_result(context)
    assert value == {
        "artifact_source_inventory_opened": False,
        "check_code": expected_code,
        "failure_stage": "SOURCE_CONTROL_QUALIFICATION",
        "holdout_or_label_opened": False,
        "repeatable": True,
        "status": "SOURCE_CONTROL_QUALIFICATION_FAILED",
    }


def test_one_shot_failure_ledger_retains_all_partial_control_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(
        context, monkeypatch, generic_binding_has_kind=True
    )
    payloads = preflight.build_preflight_payloads(
        context, authorization=_synthetic_authorization_seal(context)
    )
    assert set(payloads) == preflight.FAILURE_FILENAMES
    terminal = json.loads(payloads["preflight_terminal.json"])
    ledger = json.loads(payloads["preflight_read_ledger.json"])
    assert terminal["failure_stage"] == "SOURCE_CONTROL_QUALIFICATION"
    assert terminal["check_code"] == "GENERIC_BINDING_ENVELOPE_SCHEMA_MISMATCH"
    assert len(ledger["reads"]) == 12
    assert [entry["purpose"] for entry in ledger["reads"][-5:]] == [
        "SEALED_SPLIT",
        "ORIGINAL_PREREGISTRATION",
        "TRAIN_ALLOWLIST",
        "CANDIDATE6_TERMINAL_DISPOSITION",
        "TRAIN_CACHE_ATTESTATION",
    ]


def test_source_control_qualification_is_repeatable_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(context, monkeypatch)
    first = preflight.qualification_result(context)
    second = preflight.qualification_result(context)
    assert first == second
    assert first["scientific_hypothesis_consumed"] is False
    assert not tuple(preflight.ROOT.rglob("preflight_report.json"))


def _install_authorization_artifacts(
    context: preflight.BootstrapContext,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qualification_read_count: int = 14,
) -> str:
    root = preflight.ROOT
    output = root / "output" / "candidate7-preflight"
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", output)
    source_bindings = preflight._expected_authorization_source_bindings(context)
    qualification = {
        "artifact_source_inventory_opened": False,
        "attempt_ordinal": 7,
        "cache_manifest_opened": False,
        "check_code": "ALL_SOURCE_CONTROL_CHECKS_PASSED",
        "engineering_attempt_id": runtime.ENGINEERING_ATTEMPT_ID,
        "holdout_or_label_opened": False,
        "kind": preflight.QUALIFICATION_RECORD_KIND,
        "read_count": qualification_read_count,
        "repeatable": True,
        "schema_version": 1,
        "scientific_hypothesis_consumed": False,
        "scientific_hypothesis_id": runtime.SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": source_bindings,
        "source_control_qualification_complete": True,
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
    }
    qualification_bytes = preflight.canonical_bytes(qualification)
    qualification_path = root / "source-control-qualification.json"
    qualification_path.write_bytes(qualification_bytes)
    monkeypatch.setattr(
        preflight, "SOURCE_CONTROL_QUALIFICATION_RECORD", qualification_path
    )
    authorization = {
        "attempt_ordinal": 7,
        "authorization_consumed_on_first_manifest_open": True,
        "engineering_attempt_id": runtime.ENGINEERING_ATTEMPT_ID,
        "holdout_execution_authorized": False,
        "kind": preflight.SOURCE_INVENTORY_AUTHORIZATION_KIND,
        "label_or_metric_access_authorized": False,
        "output_namespace": str(output.relative_to(root)),
        "qualification_record_binding": {
            "kind": preflight.QUALIFICATION_RECORD_KIND,
            "path": qualification_path.name,
            "sha256": preflight.sha256_bytes(qualification_bytes),
            "size_bytes": len(qualification_bytes),
        },
        "retry_after_manifest_open_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": runtime.SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": source_bindings,
        "status": "AUTHORIZED_ONCE",
    }
    authorization_bytes = preflight.canonical_bytes(authorization)
    authorization_path = root / "source-inventory-authorization.json"
    authorization_path.write_bytes(authorization_bytes)
    monkeypatch.setattr(preflight, "SOURCE_INVENTORY_AUTHORIZATION", authorization_path)
    return preflight.sha256_bytes(authorization_bytes)


def test_external_authorization_binds_exact_sources_and_qualification_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(context, monkeypatch)
    authorization_sha = _install_authorization_artifacts(context, monkeypatch)
    reader = preflight.GuardedReader(
        entries=[dict(entry) for entry in context.bootstrap_entries]
    )
    seal = preflight.authorize_source_inventory_once(
            context,
            reader=reader,
            authorization_sha256=authorization_sha,
        )
    assert seal.authorization_sha256 == authorization_sha
    assert len(seal.qualification_record_sha256) == 64
    assert [entry["purpose"] for entry in reader.entries[-2:]] == [
        "CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
        "CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
    ]


def test_authorization_rejects_stale_or_nonpassing_qualification_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(context, monkeypatch)
    authorization_sha = _install_authorization_artifacts(
        context, monkeypatch, qualification_read_count=13
    )
    reader = preflight.GuardedReader(
        entries=[dict(entry) for entry in context.bootstrap_entries]
    )
    with pytest.raises(preflight.CheckFailure) as captured:
        preflight.authorize_source_inventory_once(
            context,
            reader=reader,
            authorization_sha256=authorization_sha,
        )
    assert captured.value.stage == "SOURCE_INVENTORY_AUTHORIZATION"
    assert captured.value.code == "QUALIFICATION_RECORD_CONTRACT_MISMATCH"


def _authorized_and_qualified(
    context: preflight.BootstrapContext,
    authorization_sha: str,
) -> preflight.SourceInventoryAuthorizationSeal:
    reader = preflight.GuardedReader(
        entries=[dict(entry) for entry in context.bootstrap_entries]
    )
    seal = preflight.authorize_source_inventory_once(
        context,
        reader=reader,
        authorization_sha256=authorization_sha,
    )
    preflight.qualify_source_controls(context, reader=reader)
    return seal


def test_durable_consumption_anchor_rejects_any_retry_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(context, monkeypatch)
    authorization_sha = _install_authorization_artifacts(context, monkeypatch)
    first = _authorized_and_qualified(context, authorization_sha)
    consumption = preflight.consume_source_inventory_authorization(context, first)
    assert consumption.path.exists()
    anchor = json.loads(consumption.path.read_bytes())
    assert anchor["status"] == "CONSUMED_BEFORE_MANIFEST_OPEN"
    assert anchor["manifest_opened_at_anchor_creation"] is False
    second = _authorized_and_qualified(context, authorization_sha)
    with pytest.raises(PermissionError, match="already consumed"):
        preflight.consume_source_inventory_authorization(context, second)
    assert all(
        entry["purpose"] != "TRAIN_CACHE_MANIFEST"
        for entry in second.reader.entries
    )


def test_partial_anchor_write_still_permanently_consumes_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    _install_actual_shaped_controls(context, monkeypatch)
    authorization_sha = _install_authorization_artifacts(context, monkeypatch)
    first = _authorized_and_qualified(context, authorization_sha)
    original_write = preflight.os.write
    invoked = False

    def crash_after_partial_write(descriptor: int, data) -> int:  # type: ignore[no-untyped-def]
        nonlocal invoked
        if not invoked:
            invoked = True
            original_write(descriptor, bytes(data[:1]))
            raise OSError("injected crash after partial anchor write")
        return original_write(descriptor, data)

    monkeypatch.setattr(preflight.os, "write", crash_after_partial_write)
    with pytest.raises(OSError, match="injected crash"):
        preflight.consume_source_inventory_authorization(context, first)
    monkeypatch.setattr(preflight.os, "write", original_write)
    assert preflight.SOURCE_INVENTORY_CONSUMPTION_ANCHOR.exists()
    second = _authorized_and_qualified(context, authorization_sha)
    with pytest.raises(PermissionError, match="already consumed"):
        preflight.consume_source_inventory_authorization(context, second)


def test_anchor_parent_swap_is_caught_before_manifest_and_restored_anchor_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    train_ids, scan_by_case, manifest_path = _install_actual_shaped_controls(
        context, monkeypatch
    )
    _install_train_inventory(train_ids, scan_by_case, manifest_path, monkeypatch)
    research = preflight.SOURCE_INVENTORY_CONSUMPTION_ANCHOR.parent
    moved = preflight.ROOT / "moved-research"

    def hide_anchor(_anchor: Path) -> None:
        research.rename(moved)
        research.mkdir()

    payloads = preflight.build_preflight_payloads(
        context,
        authorization=_synthetic_authorization_seal(context),
        after_authorization_consumed=hide_anchor,
    )
    assert set(payloads) == preflight.FAILURE_FILENAMES
    terminal = json.loads(payloads["preflight_terminal.json"])
    ledger = json.loads(payloads["preflight_read_ledger.json"])
    assert terminal["failure_stage"] == "SOURCE_INVENTORY_CONSUMPTION"
    assert all(
        entry["purpose"] != "TRAIN_CACHE_MANIFEST" for entry in ledger["reads"]
    )
    assert list(research.iterdir()) == []
    assert (moved / preflight.SOURCE_INVENTORY_CONSUMPTION_ANCHOR.name).exists()

    research.rmdir()
    moved.rename(research)
    retry = preflight.build_preflight_payloads(
        context,
        authorization=_synthetic_authorization_seal(context),
    )
    retry_terminal = json.loads(retry["preflight_terminal.json"])
    retry_ledger = json.loads(retry["preflight_read_ledger.json"])
    assert retry_terminal["failure_stage"] == "SOURCE_INVENTORY_CONSUMPTION"
    assert all(
        entry["purpose"] != "TRAIN_CACHE_MANIFEST"
        for entry in retry_ledger["reads"]
    )


def _install_train_inventory(
    train_ids: tuple[str, ...],
    scan_by_case: dict[str, str],
    manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_first_pool: bool = False,
    replay_semantic_mismatch: bool = False,
) -> None:
    root = preflight.ROOT
    cache_root = root / "cache"
    monkeypatch.setattr(preflight, "ALLOWED_CACHE_ROOT", cache_root)
    semantic = _semantic()
    pool = _pool()
    records: list[dict[str, object]] = []
    for ordinal, case_id in enumerate(train_ids):
        runtime_id = f"T{900 + ordinal:03d}"
        primary_semantic = semantic
        replay_semantic = semantic
        if replay_semantic_mismatch and ordinal == 0:
            changed = json.loads(semantic)
            changed["candidate_count"] = 999
            replay_semantic = hypothesis.canonical_bytes(changed)
        primary_pool = _pool(invalid_truth_key=True) if invalid_first_pool and ordinal == 0 else pool
        replay_pool = primary_pool
        for pass_name, semantic_bytes, pool_bytes in (
            ("primary", primary_semantic, primary_pool),
            ("replay", replay_semantic, replay_pool),
        ):
            directory = cache_root / pass_name / runtime_id
            pool_directory = directory / "pool"
            pool_directory.mkdir(parents=True)
            semantic_path = directory / "semantic_result.json"
            pool_path = pool_directory / f"{runtime_id}.json"
            semantic_path.write_bytes(semantic_bytes)
            pool_path.write_bytes(pool_bytes)
            records.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "pool_path": str(pool_path.relative_to(root)),
                    "pool_sha256": preflight.sha256_bytes(pool_bytes),
                    "runtime_case_id": runtime_id,
                    "scan_sha256": scan_by_case[case_id],
                    "semantic_path": str(semantic_path.relative_to(root)),
                    "semantic_sha256": preflight.sha256_bytes(semantic_bytes),
                }
            )
    commitment = preflight.sha256_bytes(
        preflight.canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    manifest = {
        "case_count": 18,
        "holdout_entry_count": 0,
        "kind": preflight.CACHE_MANIFEST_KIND,
        "record_count": 36,
        "records": records,
        "schema_version": 1,
        "source_only": True,
        "split_sha256": preflight.FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_allowlist_sha256": preflight.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_set_commitment_sha256": commitment,
        "v51_scan_only_manifest_sha256": preflight.V51_SCAN_ONLY_MANIFEST_SHA256,
    }
    manifest_bytes = preflight.canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = preflight.sha256_bytes(manifest_bytes)
    monkeypatch.setattr(preflight, "CACHE_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(preflight, "CACHE_MANIFEST_SIZE_BYTES", len(manifest_bytes))
    attestation_path = preflight.FIXED_CONTROLS["TRAIN_CACHE_ATTESTATION"][0]
    attestation = json.loads(attestation_path.read_bytes())
    attestation["manifest_binding"] = {
        "kind": preflight.CACHE_MANIFEST_KIND,
        "path": str(manifest_path.relative_to(root)),
        "sha256": manifest_sha,
        "size_bytes": len(manifest_bytes),
    }
    attestation_bytes = preflight.canonical_bytes(attestation)
    attestation_path.write_bytes(attestation_bytes)
    fixed = dict(preflight.FIXED_CONTROLS)
    fixed["TRAIN_CACHE_ATTESTATION"] = (
        attestation_path,
        preflight.sha256_bytes(attestation_bytes),
        len(attestation_bytes),
    )
    monkeypatch.setattr(preflight, "FIXED_CONTROLS", fixed)


def test_synthetic_exact_inventory_builds_pass_with_89_read_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    train_ids, scan_by_case, manifest_path = _install_actual_shaped_controls(
        context, monkeypatch
    )
    _install_train_inventory(train_ids, scan_by_case, manifest_path, monkeypatch)
    payloads = preflight.build_preflight_payloads(
        context, authorization=_synthetic_authorization_seal(context)
    )
    assert set(payloads) == preflight.PASS_FILENAMES, json.loads(
        payloads.get("preflight_terminal.json", b"{}")
    )
    report = json.loads(payloads["preflight_report.json"])
    ledger = json.loads(payloads["preflight_read_ledger.json"])
    lock = json.loads(payloads["preflight_lock.json"])
    assert report["status"] == "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    assert report["scientific_hypothesis_exactly_preserved"] is True
    assert report["check_code"] == "ALL_SOURCE_COMPATIBILITY_CHECKS_PASSED"
    assert report["failure_stage"] is None
    assert len(ledger["reads"]) == 89
    assert ledger["assertions"]["exact_train_inventory_completed"] is True
    assert ledger["assertions"]["label_attestation_or_projection_opened"] is False
    assert lock["preflight_report_sha256"] == preflight.sha256_bytes(
        payloads["preflight_report.json"]
    )
    for value in (report, lock):
        preflight._assert_aggregate_safe(value)


@pytest.mark.parametrize("failure", ("invalid_pool", "replay_mismatch"))
def test_source_failure_terminalizes_with_aggregate_stage_and_code(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    train_ids, scan_by_case, manifest_path = _install_actual_shaped_controls(
        context, monkeypatch
    )
    _install_train_inventory(
        train_ids,
        scan_by_case,
        manifest_path,
        monkeypatch,
        invalid_first_pool=failure == "invalid_pool",
        replay_semantic_mismatch=failure == "replay_mismatch",
    )
    payloads = preflight.build_preflight_payloads(
        context, authorization=_synthetic_authorization_seal(context)
    )
    assert set(payloads) == preflight.FAILURE_FILENAMES
    terminal = json.loads(payloads["preflight_terminal.json"])
    ledger = json.loads(payloads["preflight_read_ledger.json"])
    assert terminal["status"] == "FAILED_ENGINEERING_SOURCE_COMPATIBILITY"
    assert terminal["continuation_authorized"] is False
    assert terminal["failure_stage"] in {
        "CACHE_MANIFEST_VALIDATION",
        "TRAIN_SOURCE_COMPATIBILITY",
    }
    assert isinstance(terminal["check_code"], str) and terminal["check_code"]
    assert ledger["failure_stage"] == terminal["failure_stage"]
    assert ledger["check_code"] == terminal["check_code"]
    assert ledger["assertions"]["label_attestation_or_projection_opened"] is False
    assert "preflight_report.json" not in payloads
    assert "preflight_lock.json" not in payloads


@pytest.mark.parametrize(
    "mutation",
    ("nineteenth", "wrong_order", "wrong_scan", "extra_key", "wrong_root"),
)
def test_manifest_validator_rejects_inventory_expansion_or_rebinding(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, monkeypatch)
    train_ids, scan_by_case, manifest_path = _install_actual_shaped_controls(
        context, monkeypatch
    )
    _install_train_inventory(train_ids, scan_by_case, manifest_path, monkeypatch)
    manifest = json.loads(manifest_path.read_bytes())
    records = manifest["records"]
    if mutation == "nineteenth":
        records.append(deepcopy(records[-1]))
        manifest["record_count"] = 37
    elif mutation == "wrong_order":
        records[0], records[1] = records[1], records[0]
    elif mutation == "wrong_scan":
        records[0]["scan_sha256"] = _sha("wrong")
    elif mutation == "extra_key":
        manifest["unexpected"] = False
    else:
        records[0]["semantic_path"] = "output/research/elsewhere/value.json"
    with pytest.raises(preflight.CheckFailure):
        preflight._validate_cache_manifest(
            manifest,
            train_ids=train_ids,
            scan_by_case=scan_by_case,
            train_commitment=str(manifest["train_set_commitment_sha256"]),
        )


def test_guarded_reader_denies_unbound_label_full28_and_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    reader = preflight.GuardedReader()
    ordinary = root / "ordinary.json"
    ordinary.write_bytes(b"{}\n")
    with pytest.raises(PermissionError, match="NOT_EXACTLY_AUTHORIZED"):
        reader.read_bytes(ordinary, purpose="CACHE")
    for name in (
        "train_label_projection.json",
        "full28-shadow.json",
        "holdout-shadow.json",
        "gold-shadow.json",
    ):
        path = root / name
        path.write_bytes(b"{}\n")
        with pytest.raises(PermissionError):
            reader.authorize(
                path,
                purpose="CACHE",
                sha256=preflight.sha256_bytes(path.read_bytes()),
                size_bytes=path.stat().st_size,
            )


def test_guarded_reader_rejects_hardlink_symlink_and_leaf_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(preflight, "ROOT", root)
    first = root / "first.json"
    first.write_bytes(b"first\n")
    hardlink = root / "hardlink.json"
    os.link(first, hardlink)
    with pytest.raises(ValueError, match="non-hard-linked"):
        preflight.GuardedReader().authorize(
            first,
            purpose="CACHE",
            sha256=preflight.sha256_bytes(first.read_bytes()),
            size_bytes=first.stat().st_size,
        )
    hardlink.unlink()
    alias = root / "alias.json"
    alias.symlink_to(first)
    with pytest.raises(ValueError, match="symlink"):
        preflight.GuardedReader().authorize(
            alias,
            purpose="CACHE",
            sha256=preflight.sha256_bytes(first.read_bytes()),
            size_bytes=first.stat().st_size,
        )
    reader = preflight.GuardedReader()
    reader.authorize(
        first,
        purpose="CACHE",
        sha256=preflight.sha256_bytes(first.read_bytes()),
        size_bytes=first.stat().st_size,
    )
    replacement = root / "replacement.json"
    replacement.write_bytes(b"replacement\n")
    original_open = preflight.os.open
    swapped = False

    def swap_leaf(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if not swapped and Path(path).name == first.name and kwargs.get("dir_fd") is not None:
            swapped = True
            first.unlink()
            replacement.rename(first)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", swap_leaf)
    with pytest.raises(ValueError, match="changed"):
        reader.read_bytes(first, purpose="CACHE")


def test_guarded_reader_rejects_ancestor_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    parent = root / "bound"
    parent.mkdir(parents=True)
    monkeypatch.setattr(preflight, "ROOT", root)
    value = parent / "value.json"
    expected = b"expected\n"
    value.write_bytes(expected)
    reader = preflight.GuardedReader()
    reader.authorize(
        value,
        purpose="CACHE",
        sha256=preflight.sha256_bytes(expected),
        size_bytes=len(expected),
    )
    foreign = root / "foreign"
    foreign.mkdir()
    (foreign / value.name).write_bytes(b"foreign\n")
    moved = root / "moved"
    original_open = preflight.os.open
    swapped = False

    def swap_parent(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if not swapped and Path(path) == parent:
            swapped = True
            parent.rename(moved)
            parent.symlink_to(foreign, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(preflight.os, "open", swap_parent)
    with pytest.raises(OSError):
        reader.read_bytes(value, purpose="CACHE")


def _pass_payloads() -> dict[str, bytes]:
    return {
        "preflight_lock.json": preflight.canonical_bytes({"kind": "lock"}),
        "preflight_read_ledger.json": preflight.canonical_bytes({"kind": "ledger"}),
        "preflight_report.json": preflight.canonical_bytes({"kind": "report"}),
    }


def _publication_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    parent = root / "output" / "research"
    parent.mkdir(parents=True)
    output = parent / "candidate7-preflight"
    monkeypatch.setattr(preflight, "ROOT", root)
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", output)
    return parent, output


def test_atomic_publication_pass_and_terminal_inventories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent, output = _publication_root(tmp_path, monkeypatch)
    payloads = _pass_payloads()
    preflight.publish_atomically(output, payloads)
    assert {path.name for path in output.iterdir()} == set(payloads)
    assert all((output / name).read_bytes() == data for name, data in payloads.items())
    assert output.stat().st_mode & 0o777 == 0o500

    other = output.parent / "candidate7-preflight-failure"
    monkeypatch.setattr(preflight, "OUTPUT_ROOT", other)
    failure = {
        "preflight_read_ledger.json": b"{}\n",
        "preflight_terminal.json": b"{}\n",
    }
    preflight.publish_atomically(other, failure)
    assert {path.name for path in other.iterdir()} == set(failure)


def test_publication_no_clobber_and_exact_owned_cleanup_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    output.mkdir()
    marker = output / "foreign.txt"
    marker.write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError):
        preflight.publish_atomically(output, _pass_payloads())
    assert marker.read_text(encoding="utf-8") == "foreign"
    output.rename(parent / "prior")

    moved = parent / "moved-owned"

    def replace_after_rename(_output: Path) -> None:
        output.rename(moved)
        output.mkdir()
        (output / "foreign.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed"):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_rename=replace_after_rename,
        )
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert moved.exists()
    assert not tuple(parent.glob(f".{output.name}.quarantine-*"))


def test_raced_destination_preserved_while_owned_stage_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)

    def race(_staging: Path) -> None:
        output.mkdir()
        (output / "race.txt").write_text("race", encoding="utf-8")

    with pytest.raises(FileExistsError):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_stage_write=race,
        )
    assert (output / "race.txt").read_text(encoding="utf-8") == "race"
    assert not tuple(parent.glob(f".{output.name}.staging-*"))
    quarantine = tuple(parent.glob(f".{output.name}.quarantine-*"))
    assert len(quarantine) == 1
    assert {path.name for path in quarantine[0].iterdir()} == set(_pass_payloads())


def test_output_parent_swap_writes_nothing_to_foreign_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    root = preflight.ROOT
    foreign = root / "foreign-parent"
    foreign.mkdir()
    moved_parent = root / "moved-parent"

    def swap_parent(_staging: Path) -> None:
        parent.rename(moved_parent)
        parent.symlink_to(foreign, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError), match="parent|symlink"):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_stage_write=swap_parent,
        )
    assert list(foreign.iterdir()) == []


def test_unowned_staging_replacement_is_left_untouched_and_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = _publication_root(tmp_path, monkeypatch)
    moved: list[Path] = []

    def replace_stage(staging: Path) -> None:
        detached = parent / "detached-owned"
        staging.rename(detached)
        staging.mkdir()
        (staging / "foreign.txt").write_text("foreign", encoding="utf-8")
        moved.append(detached)

    with pytest.raises(RuntimeError, match="identity changed"):
        preflight.publish_atomically(
            output,
            _pass_payloads(),
            after_stage_write=replace_stage,
        )
    assert not output.exists()
    assert moved and moved[0].exists()
    staging = tuple(parent.glob(f".{output.name}.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not tuple(parent.glob(f".{output.name}.quarantine-*"))


def test_aggregate_failure_and_pass_artifacts_forbid_source_volume_details() -> None:
    for key in preflight.FORBIDDEN_AGGREGATE_KEYS:
        with pytest.raises(ValueError, match="forbidden key"):
            preflight._assert_aggregate_safe({key: 0})
