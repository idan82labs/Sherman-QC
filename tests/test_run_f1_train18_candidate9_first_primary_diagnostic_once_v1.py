from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from scripts import run_f1_train18_candidate9_first_primary_diagnostic_once_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_f1_train18_candidate9_first_primary_diagnostic_once_v1.py"
TESTS = ROOT / "tests/test_run_f1_train18_candidate9_first_primary_diagnostic_once_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_cli() -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--runner-sha256",
        _sha(RUNNER),
        "--runner-size-bytes",
        str(RUNNER.stat().st_size),
        "--runner-tests-sha256",
        _sha(TESTS),
        "--runner-tests-size-bytes",
        str(TESTS.stat().st_size),
    ]


def _decode_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return json.loads(result.stdout)


def _run_bootstrap_probe(body: str) -> subprocess.CompletedProcess[str]:
    script = f"""
import dataclasses
import importlib.util
import pathlib
import sys
path = pathlib.Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('c9_runner_probe', path)
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)
context = r.bootstrap_verified_sources(
    runner_sha256={_sha(RUNNER)!r}, runner_size_bytes={RUNNER.stat().st_size},
    runner_tests_sha256={_sha(TESTS)!r}, runner_tests_size_bytes={TESTS.stat().st_size})
{body}
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _sha_value(character: str = "a") -> str:
    return character * 64


def _fake_outcome(code: str = "C8_ALL_DIAGNOSTIC_PHASES_PASSED") -> runner.DiagnosticOutcome:
    return runner.DiagnosticOutcome(
        result={"check_code": code, "failure_phase": runner.DIAGNOSTIC_CODE_PHASES[code]},
        original_pool_sha256=_sha_value("1"),
        adapted_pool_sha256=_sha_value("2"),
        projection_byte_equivalent=True,
        first_primary_commitment_sha256=_sha_value("3"),
    )


def _valid_ledger(*, accepted: bool) -> dict[str, object]:
    if accepted:
        progress = runner.DiagnosticProgress(
            diagnostic_call_returned=True,
            first_primary_commitment_sha256=_sha_value("3"),
            original_pool_sha256=_sha_value("1"),
            adapted_pool_sha256=_sha_value("2"),
            projection_byte_equivalent=True,
        )
        stage = None
        code = None
        outcome = _fake_outcome()
    else:
        progress = runner.DiagnosticProgress()
        stage = "FIRST_PAIR_READ"
        code = runner.ENGINEERING_FAILURES[stage]
        outcome = None
    assertions = {
        "authorization_consumption_attempted": True,
        "consumption_anchor_sealed": True,
        "denied_read_attempted": False,
        "diagnostic_call_returned": progress.diagnostic_call_returned,
        "diagnostic_result_accepted": accepted,
        "evaluator_metric_fit_opened_or_started": False,
        "exact_read_inventory_validated": accepted,
        "full28_opened": False,
        "holdout_or_label_opened": False,
        "manifest_inode_verified": accepted,
        "manifest_opened": accepted,
        "manifest_read_started": accepted,
        "only_first_primary_pool_inode_verified": accepted,
        "only_first_primary_pool_opened": accepted,
        "only_first_primary_pool_read_started": accepted,
        "only_first_primary_semantic_inode_verified": accepted,
        "only_first_primary_semantic_opened": accepted,
        "only_first_primary_semantic_read_started": accepted,
        "replay_or_later_pair_opened": False,
    }
    value: dict[str, object] = {
        "adapted_pool_bytes_published": False,
        "adapted_pool_sha256": progress.adapted_pool_sha256,
        "assertions": assertions,
        "attempt_ordinal": 9,
        "authorization_binding_commitment_sha256": _sha_value("4"),
        "authorization_raw_fd_opened_inventory_commitment_sha256": _sha_value("5"),
        "authorization_read_inventory_commitment_sha256": _sha_value("6"),
        "authorization_read_started_inventory_commitment_sha256": _sha_value("7"),
        "authorization_verified_inode_inventory_commitment_sha256": _sha_value("8"),
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "check_code": None if outcome is None else outcome.result["check_code"],
        "consumption_anchor_sha256": _sha_value("9"),
        "control_binding_commitment_sha256": _sha_value("a"),
        "diagnostic_failure_phase": None if outcome is None else outcome.result["failure_phase"],
        "engineering_attempt_id": runner.ENGINEERING_ATTEMPT_ID,
        "engineering_check_code": code,
        "engineering_failure_stage": stage,
        "first_primary_binding_commitment_sha256": progress.first_primary_commitment_sha256,
        "kind": runner.LEDGER_KIND,
        "original_pool_bytes_published": False,
        "original_pool_sha256": progress.original_pool_sha256,
        "projection_byte_equivalent": progress.projection_byte_equivalent,
        "raw_fd_opened_inventory_commitment_sha256": _sha_value("b"),
        "read_inventory_commitment_sha256": _sha_value("c"),
        "read_started_inventory_commitment_sha256": _sha_value("d"),
        "schema_version": 1,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "source_binding_commitment_sha256": _sha_value("e"),
        "status": "DIAGNOSTIC_RECORDED" if accepted else "TERMINAL_ENGINEERING_FAILURE",
        "verified_inode_inventory_commitment_sha256": _sha_value("f"),
    }
    return value


def test_runner_top_level_imports_are_stdlib_only() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "errno",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "stat",
        "sys",
        "types",
        "typing",
    }
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed


def test_fixed_source_bindings_match_cleared_bytes() -> None:
    assert set(runner.FIXED_SOURCE_BINDINGS) == {
        "builder",
        "c6_hypothesis",
        "c7_runtime",
        "c8_adapter",
        "c8_runtime",
        "c8_support",
        "c9_adapter",
        "c9_adapter_tests",
        "cluster",
    }
    for path, expected_sha, expected_size in runner.FIXED_SOURCE_BINDINGS.values():
        data = path.read_bytes()
        assert len(data) == expected_size
        assert hashlib.sha256(data).hexdigest() == expected_sha


def test_source_check_only_passes_in_fresh_process() -> None:
    result = subprocess.run(
        [*_source_cli(), "--source-check-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert _decode_stdout(result) == {
        "artifact_or_control_input_opened": False,
        "holdout_or_label_opened": False,
        "source_inventory_complete": True,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_no_mode_rejects_before_bootstrap() -> None:
    result = subprocess.run(_source_cli(), cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "one of the arguments" in result.stderr


def test_diagnostic_mode_requires_external_authorization_binding() -> None:
    result = subprocess.run(
        [*_source_cli(), "--diagnostic-once"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires exact authorization SHA and size" in result.stderr


def test_authorization_binding_is_rejected_in_safe_modes() -> None:
    result = subprocess.run(
        [
            *_source_cli(),
            "--source-check-only",
            "--authorization-sha256",
            _sha_value(),
            "--authorization-size-bytes",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "valid only with --diagnostic-once" in result.stderr


def test_preloaded_symlink_alias_rejects_before_any_controlled_source_read(tmp_path: Path) -> None:
    alias = tmp_path / "alias.py"
    alias.symlink_to(runner.C9_ADAPTER_SOURCE)
    script = f"""
import importlib.util, pathlib, sys
alias = pathlib.Path({str(alias)!r})
spec = importlib.util.spec_from_file_location('arbitrary_alias', alias)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runner_path = pathlib.Path({str(RUNNER)!r})
rspec = importlib.util.spec_from_file_location('c9_runner_probe', runner_path)
r = importlib.util.module_from_spec(rspec)
sys.modules[rspec.name] = r
rspec.loader.exec_module(r)
r._read_once = lambda *a, **k: (_ for _ in ()).throw(AssertionError('SOURCE_READ_OCCURRED'))
try:
    r.bootstrap_verified_sources(
        runner_sha256={_sha(RUNNER)!r}, runner_size_bytes={RUNNER.stat().st_size},
        runner_tests_sha256={_sha(TESTS)!r}, runner_tests_size_bytes={TESTS.stat().st_size})
except RuntimeError as error:
    assert 'preloaded source alias' in str(error)
else:
    raise AssertionError('preloaded alias was accepted')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout


def test_counterfeit_exact_type_bootstrap_context_is_not_factory_issued() -> None:
    result = _run_bootstrap_probe(
        """
forged = dataclasses.replace(context, _issuance_token=object())
try:
    r._validate_context_integrity(forged)
except PermissionError:
    pass
else:
    raise AssertionError('counterfeit context accepted')
"""
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_controlled_function_code_mutation_is_detected_source_only() -> None:
    result = _run_bootstrap_probe(
        """
context.c9_adapter.adapt_pool_bytes.__code__ = (lambda *args, **kwargs: None).__code__
try:
    r._validate_context_integrity(context)
except PermissionError:
    pass
else:
    raise AssertionError('function-code mutation accepted')
"""
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_control_binding_constant_mutation_is_detected_source_only() -> None:
    result = _run_bootstrap_probe(
        """
context.support.FIXED_CONTROLS['MUTATED'] = (pathlib.Path('x'), 'a' * 64, 1)
try:
    r._validate_context_integrity(context)
except (PermissionError, ValueError):
    pass
else:
    raise AssertionError('control-constant mutation accepted')
"""
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_source_binding_mappings_are_recursively_immutable() -> None:
    result = _run_bootstrap_probe(
        """
try:
    context.source_bindings['c9_adapter']['sha256'] = '0' * 64
except TypeError:
    pass
else:
    raise AssertionError('nested source binding was mutable')
"""
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_forged_authorization_and_control_qualification_seals_are_rejected() -> None:
    result = _run_bootstrap_probe(
        """
fresh = r.FreshOutputSeal(path=r.OUTPUT_ROOT, parent_identity=(1, 2), _sentinel=r._SEAL)
forged_auth = r.AuthorizationSeal(
    reader=object(), authorization_sha256='a' * 64, authorization_size_bytes=1,
    qualification_sha256='b' * 64, qualification_size_bytes=1,
    control_commitment_sha256='c' * 64,
    source_commitment_sha256=context.source_commitment_sha256,
    authority_read_inventory_commitment_sha256='d' * 64,
    authority_raw_fd_opened_commitment_sha256='e' * 64,
    authority_verified_inode_commitment_sha256='f' * 64,
    authority_read_started_commitment_sha256='1' * 64,
    anchor_parent_identity=(1, 2), output_freshness=fresh,
    _context_nonce=context._nonce, _sentinel=r._SEAL, _issuance_token=object())
try:
    r._validate_authorization_seal(context, forged_auth)
except PermissionError:
    pass
else:
    raise AssertionError('forged authorization accepted')
forged_qualification = r.ControlQualification(
    base=object(), reader=object(), c8_disposition_bytes=b'', c8_disposition={},
    c8_first_primary_commitment_sha256='2' * 64,
    control_commitment_sha256='3' * 64,
    base_contract_commitment_sha256='4' * 64,
    control_read_inventory_commitment_sha256='5' * 64,
    control_raw_fd_opened_commitment_sha256='6' * 64,
    control_verified_inode_commitment_sha256='7' * 64,
    control_read_started_commitment_sha256='8' * 64,
    control_entry_prefix_length=1, control_open_prefix_length=1,
    _base_pair_capability_nonce=object(), _context_nonce=context._nonce,
    _sentinel=r._QUALIFICATION_SEAL, _issuance_token=object())
try:
    r._validate_control_qualification(context, forged_qualification)
except PermissionError:
    pass
else:
    raise AssertionError('forged qualification accepted')
"""
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_qualification_record_discloses_nonexecuted_builder_cluster_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace(source_bindings={"x": {"path": "x", "sha256": _sha_value(), "size_bytes": 1}})
    qualification = SimpleNamespace(
        control_commitment_sha256=_sha_value("b"),
        c8_first_primary_commitment_sha256=_sha_value("c"),
        control_raw_fd_opened_commitment_sha256=_sha_value("d"),
        control_read_inventory_commitment_sha256=_sha_value("e"),
        control_read_started_commitment_sha256=_sha_value("f"),
        control_verified_inode_commitment_sha256=_sha_value("1"),
    )
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "_validate_control_qualification", lambda context, qualification: None)
    value = json.loads(runner.build_qualification_record(context, qualification))
    assert value["builder_cluster_lineage_bytes_authenticated"] is True
    assert value["builder_cluster_source_execution_performed"] is False
    assert value["artifact_source_inventory_opened"] is False
    assert value["cache_manifest_opened"] is False


def test_actual_shaped_synthetic_control_qualification_is_factory_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    source_entry = {
        "device_id": 1,
        "inode": 1,
        "path": "synthetic/source.py",
        "purpose": "CANDIDATE9_SOURCE_SYNTHETIC",
        "sha256": _sha_value("1"),
        "size_bytes": 1,
    }
    base_purposes = [
        "BASE_CONTROL",
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
    ]
    entries = [dict(source_entry)]
    for index, purpose in enumerate(base_purposes, start=2):
        entries.append(
            {
                "device_id": 1,
                "inode": index,
                "path": f"synthetic/{purpose.lower()}.json",
                "purpose": purpose,
                "sha256": f"{index:x}" * 64,
                "size_bytes": index,
            }
        )
    reader = SimpleNamespace(
        entries=entries,
        raw_fd_opened_purposes=list(base_purposes),
        verified_inode_purposes=list(base_purposes),
        read_started_purposes=list(base_purposes),
        denied_attempts=[],
    )
    nonce = object()
    reader.first_primary_nonce = nonce
    base = SimpleNamespace(
        reader=reader,
        _pair_capability_nonce=nonce,
        control_binding_commitment_sha256=_sha_value("2"),
        c7_pool_evidence={"device_id": 7, "inode": 8},
        c7_semantic_evidence={"device_id": 5, "inode": 6},
        candidate7_disposition_bytes=b"candidate7",
        manifest_path=ROOT / "tmp/synthetic-c9-qualified-manifest.json",
        manifest_sha256=_sha_value("3"),
        manifest_size_bytes=10,
        scan_by_case={"opaque": _sha_value("4")},
        train_commitment=_sha_value("5"),
        train_ids=("opaque",),
    )
    support = SimpleNamespace(
        FIXED_CONTROLS={"BASE_CONTROL": (ROOT / "tmp/base.json", _sha_value("6"), 1)},
        qualify_source_controls=lambda context: base,
        _assert_public_payload=lambda value: None,
        _assert_value_free_ledger=lambda value: None,
    )
    context = SimpleNamespace(
        support=support,
        source_entries=(source_entry,),
        source_bindings={"synthetic": {"path": source_entry["path"], "sha256": source_entry["sha256"], "size_bytes": 1}},
        _nonce=object(),
        c9_adapter=SimpleNamespace(CANDIDATE8_DISPOSITION_SHA256="unused"),
    )
    ledger = {
        "assertions": {
            "manifest_opened": True,
            "manifest_inode_verified": True,
            "manifest_read_started": True,
            "only_first_primary_semantic_opened": True,
            "only_first_primary_semantic_inode_verified": True,
            "only_first_primary_semantic_read_started": True,
            "only_first_primary_pool_opened": True,
            "only_first_primary_pool_inode_verified": True,
            "only_first_primary_pool_read_started": True,
            "replay_or_later_pair_opened": False,
            "labels_opened": False,
            "evaluator_metric_fit_opened_or_started": False,
        },
        "check_code": "C8_POOL_PARSE_SANITIZE_FAILED",
        "diagnostic_failure_phase": "D02_POOL_PARSE_SANITIZE",
        "first_primary_binding_commitment_sha256": _sha_value("7"),
    }
    ledger_bytes = runner.canonical_bytes(ledger)
    result = {
        "check_code": "C8_POOL_PARSE_SANITIZE_FAILED",
        "diagnostic_failure_phase": "D02_POOL_PARSE_SANITIZE",
        "diagnostic_read_ledger_sha256": runner.sha256_bytes(ledger_bytes),
        "status": "DIAGNOSTIC_COMPLETE_NONAUTHORIZING",
    }
    disposition_bytes = runner.canonical_bytes({"disposition": True})
    monkeypatch.setattr(runner, "C8_DISPOSITION_SHA256", runner.sha256_bytes(disposition_bytes))
    monkeypatch.setattr(runner, "C8_DISPOSITION_SIZE_BYTES", len(disposition_bytes))
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "_support_context", lambda context: object())
    monkeypatch.setattr(runner, "_validate_c8_disposition", lambda value, context: None)
    values = {
        "CANDIDATE8_TERMINAL_DISPOSITION": (disposition_bytes, {"disposition": True}),
        "CANDIDATE8_DIAGNOSTIC_RESULT": (runner.canonical_bytes(result), result),
        "CANDIDATE8_DIAGNOSTIC_LEDGER": (ledger_bytes, ledger),
    }
    next_inode = 20
    def read_bound(reader_value: object, *, path: Path, purpose: str, sha256: str, size_bytes: int) -> tuple[bytes, dict[str, object]]:
        nonlocal next_inode
        data, value = values[purpose]
        reader.entries.append(
            {
                "device_id": 1,
                "inode": next_inode,
                "path": f"synthetic/{purpose.lower()}.json",
                "purpose": purpose,
                "sha256": runner.sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
        next_inode += 1
        reader.raw_fd_opened_purposes.append(purpose)
        reader.verified_inode_purposes.append(purpose)
        reader.read_started_purposes.append(purpose)
        return data, value
    monkeypatch.setattr(runner, "_read_bound", read_bound)
    qualification = runner.qualify_source_controls(context)
    runner._validate_control_qualification(context, qualification)
    record = json.loads(runner.build_qualification_record(context, qualification))
    assert record["candidate8_first_primary_binding_commitment_sha256"] == _sha_value("7")
    assert record["control_read_inventory_commitment_sha256"] == qualification.control_read_inventory_commitment_sha256
    for purpose in (
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        "TRAIN_FIRST_PRIMARY_V51_POOL",
    ):
        reader.entries.append(
            {
                "device_id": 1,
                "inode": next_inode,
                "path": f"synthetic/{purpose.lower()}.json",
                "purpose": purpose,
                "sha256": _sha_value("9"),
                "size_bytes": 1,
            }
        )
        next_inode += 1
        reader.raw_fd_opened_purposes.append(purpose)
        reader.verified_inode_purposes.append(purpose)
        reader.read_started_purposes.append(purpose)
    monkeypatch.setattr(runner, "_validate_authorization_seal", lambda context, authorization: None)
    runner._validate_exact_inventory(context, qualification, object())
    reader.read_started_purposes.append("EXTRA_DIAGNOSTIC_READ")
    with pytest.raises(RuntimeError, match="exact diagnostic read inventory"):
        runner._validate_exact_inventory(context, qualification, object())


def test_authorization_contract_is_exact_and_boundary_denying() -> None:
    context = SimpleNamespace(source_bindings={"x": {"path": "x", "sha256": _sha_value(), "size_bytes": 1}})
    value = runner.build_authorization_contract(
        context,
        qualification_sha256=_sha_value("1"),
        qualification_size_bytes=10,
        control_commitment_sha256=_sha_value("2"),
    )
    assert value["status"] == "AUTHORIZED_ONCE"
    assert value["retry_after_anchor_authorized"] is False
    assert value["replay_or_later_pair_access_authorized"] is False
    assert value["holdout_label_evaluator_metric_fit_access_authorized"] is False
    assert value["builder_cluster_source_execution_performed"] is False


def test_build_payloads_rebinds_exact_qualification_before_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    context = object()
    authorization = SimpleNamespace(
        control_commitment_sha256=_sha_value("1"),
        qualification_sha256=_sha_value("2"),
        qualification_size_bytes=10,
        output_freshness=object(),
    )
    qualification = SimpleNamespace(control_commitment_sha256=_sha_value("1"))
    monkeypatch.setattr(runner, "_validate_authorization_seal", lambda context, authorization: None)
    monkeypatch.setattr(runner, "qualify_source_controls", lambda context: qualification)
    monkeypatch.setattr(runner, "build_qualification_record", lambda context, qualification: b"changed\n")
    monkeypatch.setattr(
        runner,
        "consume_authorization",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor was consumed")),
    )
    with pytest.raises(PermissionError, match="qualification changed"):
        runner.build_payloads(context, authorization)


@pytest.mark.parametrize("code", sorted(runner.DIAGNOSTIC_CODE_PHASES))
def test_public_diagnostic_accepts_only_frozen_code_phase_pairs(code: str) -> None:
    value = runner._public_payload(outcome=_fake_outcome(code), ledger_sha256=_sha_value())
    runner._validate_public_payload(value)
    wrong = dict(value)
    wrong["diagnostic_failure_phase"] = "D05_WRAPPER_RESULT_VALIDATION"
    if wrong["diagnostic_failure_phase"] == runner.DIAGNOSTIC_CODE_PHASES[code]:
        wrong["diagnostic_failure_phase"] = None
    with pytest.raises(ValueError, match="code/phase"):
        runner._validate_public_payload(wrong)


def test_public_terminal_is_detail_free_and_nonretryable() -> None:
    value = runner._public_payload(
        outcome=None,
        ledger_sha256=_sha_value(),
        engineering_stage="REPRESENTATION_ADAPTER",
    )
    assert set(value) == runner.PUBLIC_KEYS
    assert value["check_code"] is None
    assert value["diagnostic_failure_phase"] is None
    assert value["engineering_check_code"] == "LOSSLESS_PROJECTION_REJECTED"
    assert value["retry_authorized"] is False
    assert value["scientific_hypothesis_performance_tested"] is False


@pytest.mark.parametrize("accepted", [False, True])
def test_ledger_exact_schema_and_value_free_contract(accepted: bool) -> None:
    value = _valid_ledger(accepted=accepted)
    runner._validate_ledger(value)
    assert set(value) == runner.LEDGER_KEYS
    assert set(value["assertions"]) == runner.LEDGER_ASSERTION_KEYS
    encoded = json.dumps(value, sort_keys=True)
    for forbidden in ("case_id", "runtime_case_id", "exception_text", "line_start", "feature_value"):
        assert forbidden not in encoded


def test_ledger_rejects_extra_keys_nonbools_and_false_open_ordering() -> None:
    value = _valid_ledger(accepted=True)
    value["case_id"] = "forbidden"
    with pytest.raises(ValueError, match="schema"):
        runner._validate_ledger(value)
    value = _valid_ledger(accepted=True)
    value["assertions"]["manifest_opened"] = 1
    with pytest.raises(ValueError, match="schema"):
        runner._validate_ledger(value)
    value = _valid_ledger(accepted=True)
    value["assertions"]["manifest_opened"] = False
    with pytest.raises(ValueError, match="raw-open"):
        runner._validate_ledger(value)


def test_terminal_ledger_can_truthfully_bind_completed_adapter_and_raw_diagnostic_return() -> None:
    value = _valid_ledger(accepted=False)
    value["engineering_failure_stage"] = "READ_INVENTORY"
    value["engineering_check_code"] = runner.ENGINEERING_FAILURES["READ_INVENTORY"]
    value["adapted_pool_sha256"] = _sha_value("2")
    value["original_pool_sha256"] = _sha_value("1")
    value["projection_byte_equivalent"] = True
    value["first_primary_binding_commitment_sha256"] = _sha_value("3")
    value["assertions"]["diagnostic_call_returned"] = True
    value["assertions"]["manifest_opened"] = True
    value["assertions"]["manifest_inode_verified"] = True
    value["assertions"]["manifest_read_started"] = True
    value["assertions"]["only_first_primary_semantic_opened"] = True
    value["assertions"]["only_first_primary_semantic_inode_verified"] = True
    value["assertions"]["only_first_primary_semantic_read_started"] = True
    value["assertions"]["only_first_primary_pool_opened"] = True
    value["assertions"]["only_first_primary_pool_inode_verified"] = True
    value["assertions"]["only_first_primary_pool_read_started"] = True
    runner._validate_ledger(value)


def test_ledger_rejects_hash_receipt_without_projection_equivalence() -> None:
    value = _valid_ledger(accepted=False)
    value["adapted_pool_sha256"] = _sha_value("2")
    value["original_pool_sha256"] = _sha_value("1")
    with pytest.raises(ValueError, match="projection binding"):
        runner._validate_ledger(value)


def test_terminal_ledger_records_raw_open_before_inode_verification() -> None:
    reader = SimpleNamespace(
        raw_fd_opened_purposes=["TRAIN_CACHE_MANIFEST"],
        verified_inode_purposes=[],
        read_started_purposes=[],
        denied_attempts=[],
        entries=[],
    )
    qualification = SimpleNamespace(reader=reader, control_commitment_sha256=_sha_value("1"))
    authorization = SimpleNamespace(
        authorization_sha256=_sha_value("2"),
        authorization_size_bytes=1,
        authority_raw_fd_opened_commitment_sha256=_sha_value("3"),
        authority_read_inventory_commitment_sha256=_sha_value("4"),
        authority_read_started_commitment_sha256=_sha_value("5"),
        authority_verified_inode_commitment_sha256=_sha_value("6"),
    )
    context = SimpleNamespace(source_commitment_sha256=_sha_value("7"))
    consumption = SimpleNamespace(sha256=_sha_value("8"))
    value = runner._ledger(
        context,
        authorization,
        qualification,
        consumption,
        outcome=None,
        progress=runner.DiagnosticProgress(),
        engineering_failure_stage="FIRST_PAIR_READ",
        engineering_check_code=runner.ENGINEERING_FAILURES["FIRST_PAIR_READ"],
        exact_inventory=False,
    )
    assertions = value["assertions"]
    assert assertions["manifest_opened"] is True
    assert assertions["manifest_inode_verified"] is False
    assert assertions["manifest_read_started"] is False


def test_exact_inventory_revalidates_authorization_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: calls.append("context"))
    monkeypatch.setattr(runner, "_validate_control_qualification", lambda context, qualification: calls.append("qualification"))
    def reject(context: object, authorization: object) -> None:
        calls.append("authorization")
        raise PermissionError("mutated authority reader")
    monkeypatch.setattr(runner, "_validate_authorization_seal", reject)
    with pytest.raises(PermissionError, match="mutated"):
        runner._validate_exact_inventory(object(), object(), object())
    assert calls == ["context", "qualification", "authorization"]


class _FakeReader:
    def __init__(self, manifest_path: Path, semantic_path: Path, pool_path: Path, manifest: bytes, semantic: bytes, pool: bytes) -> None:
        self.payloads = {manifest_path: manifest, semantic_path: semantic, pool_path: pool}
        self.allowed: dict[Path, dict[str, object]] = {}
        self.reads: list[str] = []

    def authorize(self, path: Path, *, purpose: str, sha256: str, size_bytes: int, first_primary_binding: object = None) -> None:
        identity = (11, 12) if "SEMANTIC" in purpose else (13, 14)
        self.allowed[path] = {"identity": identity, "purpose": purpose, "sha256": sha256, "size_bytes": size_bytes}

    def read_bytes(self, path: Path, *, purpose: str) -> bytes:
        self.reads.append(purpose)
        return self.payloads[path]


def _diagnostic_fixture() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, bytes]:
    manifest_path = ROOT / "tmp/synthetic-c9-manifest.json"
    semantic_path = ROOT / "tmp/synthetic-c9-semantic.json"
    pool_path = ROOT / "tmp/synthetic-c9-pool.json"
    semantic = runner.canonical_bytes({"semantic": True})
    pool = runner.canonical_bytes({"pool": True})
    adapted = runner.canonical_bytes({"pool": "adapted"})
    manifest = runner.canonical_bytes({"synthetic": True})
    first = SimpleNamespace(
        semantic_path=semantic_path,
        semantic_sha256=runner.sha256_bytes(semantic),
        semantic_size_bytes=len(semantic),
        pool_path=pool_path,
        pool_sha256=runner.sha256_bytes(pool),
        pool_size_bytes=len(pool),
        commitment_sha256=_sha_value("3"),
    )
    reader = _FakeReader(manifest_path, semantic_path, pool_path, manifest, semantic, pool)
    base = SimpleNamespace(
        manifest_path=manifest_path,
        manifest_sha256=runner.sha256_bytes(manifest),
        manifest_size_bytes=len(manifest),
        c7_semantic_evidence={"device_id": 11, "inode": 12},
        c7_pool_evidence={"device_id": 13, "inode": 14},
        candidate7_disposition_bytes=b"c7",
    )
    qualification = SimpleNamespace(
        reader=reader,
        base=base,
        c8_first_primary_commitment_sha256=first.commitment_sha256,
    )
    captured: dict[str, object] = {}
    support = SimpleNamespace(
        _validate_manifest_first_primary=lambda value, qual: first,
        FIXED_CONTROLS={"CANDIDATE7_TERMINAL_DISPOSITION": (Path("x"), _sha_value("4"), 1)},
    )
    c9_adapter = SimpleNamespace(
        adapt_pool_bytes=lambda context, *, pool_bytes, expected_pool_sha256: SimpleNamespace(
            original_pool_sha256=expected_pool_sha256,
            adapted_pool_sha256=runner.sha256_bytes(adapted),
            adapted_pool_bytes=adapted,
            projection_byte_equivalent=True,
        )
    )
    def diagnose(context: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"check_code": "C8_ALL_DIAGNOSTIC_PHASES_PASSED", "failure_phase": None}
    context = SimpleNamespace(
        support=support,
        c9_adapter=c9_adapter,
        c9_adapter_context=object(),
        c8_adapter=SimpleNamespace(diagnose_bound_pair=diagnose),
        c8_adapter_context=object(),
        c8_runtime=SimpleNamespace(
            _validate_result=lambda result: None,
            PASS_CODE="C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            FAILURE_CODES={
                "D01_SEMANTIC_VALIDATION": "C8_SEMANTIC_VALIDATION_FAILED",
                "D02_POOL_PARSE_SANITIZE": "C8_POOL_PARSE_SANITIZE_FAILED",
                "D03_OWNERSHIP_EXERCISE": "C8_OWNERSHIP_EXERCISE_FAILED",
                "D04_FIXED_RULE_EXERCISE": "C8_FIXED_RULE_EXERCISE_FAILED",
                "D05_WRAPPER_RESULT_VALIDATION": "C8_WRAPPER_RESULT_VALIDATION_FAILED",
            },
            PHASE_CONTRACT=(
                "D01_SEMANTIC_VALIDATION",
                "D02_POOL_PARSE_SANITIZE",
                "D03_OWNERSHIP_EXERCISE",
                "D04_FIXED_RULE_EXERCISE",
                "D05_WRAPPER_RESULT_VALIDATION",
            ),
        ),
    )
    return context, qualification, SimpleNamespace(captured=captured, first=first), adapted


def test_read_adapter_diagnostic_uses_exact_adapted_bytes_and_binds_both_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    context, qualification, state, adapted = _diagnostic_fixture()
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "_validate_control_qualification", lambda context, qualification: None)
    monkeypatch.setattr(runner, "revalidate_anchor", lambda seal: None)
    progress = runner.DiagnosticProgress()
    outcome = runner.read_adapt_and_diagnose(
        context,
        qualification,
        SimpleNamespace(),
        progress=progress,
    )
    assert state.captured["pool_bytes"] == adapted
    assert state.captured["pool_sha256"] == runner.sha256_bytes(adapted)
    assert outcome.original_pool_sha256 == state.first.pool_sha256
    assert outcome.adapted_pool_sha256 == runner.sha256_bytes(adapted)
    assert progress.diagnostic_call_returned is True


def test_read_adapter_rejects_receipt_hash_that_does_not_match_adapted_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    context, qualification, state, _adapted = _diagnostic_fixture()
    context.c9_adapter.adapt_pool_bytes = lambda *args, **kwargs: SimpleNamespace(
        original_pool_sha256=state.first.pool_sha256,
        adapted_pool_sha256=_sha_value("5"),
        adapted_pool_bytes=b"different",
        projection_byte_equivalent=True,
    )
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "_validate_control_qualification", lambda context, qualification: None)
    monkeypatch.setattr(runner, "revalidate_anchor", lambda seal: None)
    with pytest.raises(RuntimeError, match="do not match their receipt"):
        runner.read_adapt_and_diagnose(context, qualification, SimpleNamespace())
    assert state.captured == {}


def test_manifest_pair_must_equal_candidate8_consumed_pair_before_artifact_authorize(monkeypatch: pytest.MonkeyPatch) -> None:
    context, qualification, _state, _adapted = _diagnostic_fixture()
    qualification.c8_first_primary_commitment_sha256 = _sha_value("6")
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "_validate_control_qualification", lambda context, qualification: None)
    monkeypatch.setattr(runner, "revalidate_anchor", lambda seal: None)
    with pytest.raises(RuntimeError, match="differs from Candidate8"):
        runner.read_adapt_and_diagnose(context, qualification, SimpleNamespace())
    assert qualification.reader.reads == ["TRAIN_CACHE_MANIFEST"]


def _within_repo_temp() -> tempfile.TemporaryDirectory[str]:
    (ROOT / "tmp").mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="c9-runner-test-", dir=ROOT / "tmp")


def test_consumption_anchor_is_durable_and_rejects_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        synthetic_root = Path(directory)
        (synthetic_root / "research").mkdir()
        anchor = synthetic_root / "research/f1_aug30_train18_candidate9_first_primary_diagnostic_consumption_v1.json"
        monkeypatch.setattr(runner, "ROOT", synthetic_root)
        monkeypatch.setattr(runner, "C9_ANCHOR", anchor)
        monkeypatch.setattr(runner, "C9_AUTHORIZATION", synthetic_root / "research/authorization.json")
        monkeypatch.setattr(runner, "OUTPUT_ROOT", synthetic_root / "output/diagnostic")
        monkeypatch.setattr(runner, "_validate_authorization_seal", lambda context, seal: None)
        parent_fd, parent_identity = runner._open_stable_directory(anchor.parent, role="test parent")
        os.close(parent_fd)
        context = SimpleNamespace(source_commitment_sha256=_sha_value("a"))
        seal = SimpleNamespace(
            authorization_sha256=_sha_value("b"),
            authorization_size_bytes=10,
            control_commitment_sha256=_sha_value("c"),
            anchor_parent_identity=parent_identity,
        )
        consumption = runner.consume_authorization(context, seal)
        assert anchor.read_bytes() == consumption.data
        assert stat_is_read_only(anchor)
        with pytest.raises(PermissionError, match="already consumed"):
            runner.consume_authorization(context, seal)


def stat_is_read_only(path: Path) -> bool:
    return path.stat().st_mode & 0o777 == 0o400


def test_partial_anchor_write_permanently_blocks_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        synthetic_root = Path(directory)
        (synthetic_root / "research").mkdir()
        anchor = synthetic_root / "research/f1_aug30_train18_candidate9_first_primary_diagnostic_consumption_v1.json"
        monkeypatch.setattr(runner, "ROOT", synthetic_root)
        monkeypatch.setattr(runner, "C9_ANCHOR", anchor)
        monkeypatch.setattr(runner, "C9_AUTHORIZATION", synthetic_root / "research/authorization.json")
        monkeypatch.setattr(runner, "OUTPUT_ROOT", synthetic_root / "output/diagnostic")
        monkeypatch.setattr(runner, "_validate_authorization_seal", lambda context, seal: None)
        parent_fd, parent_identity = runner._open_stable_directory(anchor.parent, role="test parent")
        os.close(parent_fd)
        context = SimpleNamespace(source_commitment_sha256=_sha_value("a"))
        seal = SimpleNamespace(
            authorization_sha256=_sha_value("b"),
            authorization_size_bytes=10,
            control_commitment_sha256=_sha_value("c"),
            anchor_parent_identity=parent_identity,
        )
        original_write = runner.os.write
        calls = 0
        def fail_after_open(descriptor: int, data: object) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("synthetic write failure")
            return original_write(descriptor, data)
        monkeypatch.setattr(runner.os, "write", fail_after_open)
        with pytest.raises(runner.AnchorCreationFailure):
            runner.consume_authorization(context, seal)
        monkeypatch.setattr(runner.os, "write", original_write)
        assert anchor.exists()
        with pytest.raises(PermissionError, match="already consumed"):
            runner.consume_authorization(context, seal)


def test_anchor_parent_replacement_after_fsync_is_detected_and_same_seal_cannot_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        synthetic_root = Path(directory)
        research = synthetic_root / "research"
        research.mkdir()
        anchor = research / "f1_aug30_train18_candidate9_first_primary_diagnostic_consumption_v1.json"
        monkeypatch.setattr(runner, "ROOT", synthetic_root)
        monkeypatch.setattr(runner, "C9_ANCHOR", anchor)
        monkeypatch.setattr(runner, "C9_AUTHORIZATION", research / "authorization.json")
        monkeypatch.setattr(runner, "OUTPUT_ROOT", synthetic_root / "output/diagnostic")
        monkeypatch.setattr(runner, "_validate_authorization_seal", lambda context, seal: None)
        parent_fd, parent_identity = runner._open_stable_directory(research, role="test parent")
        os.close(parent_fd)
        context = SimpleNamespace(source_commitment_sha256=_sha_value("a"))
        seal = SimpleNamespace(
            authorization_sha256=_sha_value("b"),
            authorization_size_bytes=10,
            control_commitment_sha256=_sha_value("c"),
            anchor_parent_identity=parent_identity,
        )
        original_visibility = runner._assert_canonical_anchor_visibility
        moved = synthetic_root / "research-hidden"
        swapped = False
        def swap_then_validate(**kwargs: object) -> None:
            nonlocal swapped
            if not swapped:
                research.rename(moved)
                research.mkdir()
                swapped = True
            original_visibility(**kwargs)
        monkeypatch.setattr(runner, "_assert_canonical_anchor_visibility", swap_then_validate)
        with pytest.raises(runner.AnchorCreationFailure):
            runner.consume_authorization(context, seal)
        assert (moved / anchor.name).exists()
        assert not anchor.exists()
        with pytest.raises(PermissionError, match="parent changed"):
            runner.consume_authorization(context, seal)


@pytest.mark.parametrize("inventory", ["diagnostic_result.json", "diagnostic_terminal.json"])
def test_atomic_publication_writes_exact_two_file_inventory(monkeypatch: pytest.MonkeyPatch, inventory: str) -> None:
    with _within_repo_temp() as directory:
        output = Path(directory) / "output"
        monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
        freshness = runner.prevalidate_fresh_output(output)
        payloads = {"diagnostic_read_ledger.json": b"{\"ledger\":true}\n", inventory: b"{\"public\":true}\n"}
        runner.publish_atomically(output, payloads, freshness=freshness)
        assert sorted(path.name for path in output.iterdir()) == sorted(payloads)
        for name, data in payloads.items():
            assert (output / name).read_bytes() == data


def test_publication_rejects_preexisting_output_and_residue(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        output = Path(directory) / "output"
        monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
        output.mkdir()
        with pytest.raises(FileExistsError):
            runner.prevalidate_fresh_output(output)
    with _within_repo_temp() as directory:
        output = Path(directory) / "output"
        monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
        (output.parent / f".{output.name}.quarantine-blocked").mkdir()
        with pytest.raises(FileExistsError):
            runner.prevalidate_fresh_output(output)


def test_partial_publication_failure_removes_only_owned_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        output = Path(directory) / "output"
        monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
        freshness = runner.prevalidate_fresh_output(output)
        payloads = {"diagnostic_read_ledger.json": b"ledger", "diagnostic_result.json": b"result"}
        original_write = runner.os.write
        calls = 0
        def fail_write(descriptor: int, data: object) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("synthetic staged write failure")
            return original_write(descriptor, data)
        monkeypatch.setattr(runner.os, "write", fail_write)
        with pytest.raises(OSError, match="synthetic"):
            runner.publish_atomically(output, payloads, freshness=freshness)
        assert not output.exists()
        assert not any(path.name.startswith(f".{output.name}.staging-") for path in output.parent.iterdir())


def test_cleanup_and_quarantine_leave_same_name_foreign_leaf_untouched() -> None:
    with _within_repo_temp() as directory:
        parent = Path(directory)
        stage = parent / ".output.staging-test"
        stage.mkdir()
        leaf = stage / "diagnostic_result.json"
        leaf.write_bytes(b"owned")
        directory_identity = runner._identity(stage.stat())
        owned_leaf_identity = runner._identity(leaf.stat())
        leaf.unlink()
        leaf.write_bytes(b"foreign")
        parent_fd, _identity = runner._open_stable_directory(parent, role="synthetic cleanup parent")
        try:
            runner._cleanup_owned_stage(
                parent_fd,
                stage.name,
                directory_identity,
                {leaf.name: owned_leaf_identity},
            )
            assert stage.exists()
            assert leaf.read_bytes() == b"foreign"
            runner._quarantine_owned_publication(
                parent_fd,
                output_name=stage.name,
                output_identity=directory_identity,
                leaf_identities={leaf.name: owned_leaf_identity},
            )
            assert stage.exists()
            assert leaf.read_bytes() == b"foreign"
            assert not any(path.name.startswith(f".{stage.name}.quarantine-") for path in parent.iterdir())
        finally:
            os.close(parent_fd)


def test_postrename_failure_retries_quarantine_collision_and_leaves_no_trusted_output(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        output = Path(directory) / "output"
        monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
        freshness = runner.prevalidate_fresh_output(output)
        payloads = {"diagnostic_read_ledger.json": b"ledger", "diagnostic_result.json": b"result"}
        original_verify = runner._verify_published_payloads
        def fail_final(parent_fd: int, **kwargs: object) -> None:
            if kwargs["directory_name"] == output.name:
                raise RuntimeError("synthetic final verification failure")
            original_verify(parent_fd, **kwargs)
        monkeypatch.setattr(runner, "_verify_published_payloads", fail_final)
        original_rename = runner._renameat_noreplace
        quarantine_attempts = 0
        def collide_once(parent_fd: int, source: str, destination: str) -> None:
            nonlocal quarantine_attempts
            if source == output.name:
                quarantine_attempts += 1
                if quarantine_attempts == 1:
                    raise FileExistsError("synthetic quarantine collision")
            original_rename(parent_fd, source, destination)
        monkeypatch.setattr(runner, "_renameat_noreplace", collide_once)
        with pytest.raises(RuntimeError, match="final verification"):
            runner.publish_atomically(output, payloads, freshness=freshness)
        assert quarantine_attempts == 2
        assert not output.exists()
        quarantines = [path for path in output.parent.iterdir() if path.name.startswith(f".{output.name}.quarantine-")]
        assert len(quarantines) == 1


def test_publication_parent_swap_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with _within_repo_temp() as directory:
        original_parent = Path(directory) / "parent"
        original_parent.mkdir()
        output = original_parent / "output"
        monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
        freshness = runner.prevalidate_fresh_output(output)
        moved = Path(directory) / "moved"
        original_parent.rename(moved)
        original_parent.mkdir()
        with pytest.raises(RuntimeError, match="parent changed"):
            runner.publish_atomically(
                output,
                {"diagnostic_read_ledger.json": b"ledger", "diagnostic_result.json": b"result"},
                freshness=freshness,
            )


def test_main_source_control_failure_exits_nonzero_without_one_shot(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **kwargs: object())
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "qualification_result", lambda context: {"status": "SOURCE_CONTROL_QUALIFICATION_FAILED"})
    monkeypatch.setattr(runner, "run_one_shot", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("one-shot called")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--runner-sha256",
            _sha_value(),
            "--runner-size-bytes",
            "1",
            "--runner-tests-sha256",
            _sha_value("b"),
            "--runner-tests-size-bytes",
            "1",
            "--source-control-check-only",
        ],
    )
    assert runner.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "SOURCE_CONTROL_QUALIFICATION_FAILED"


def test_main_terminal_one_shot_exits_nonzero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **kwargs: object())
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "run_one_shot", lambda *args, **kwargs: {"status": "TERMINAL_ENGINEERING_FAILURE"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--runner-sha256",
            _sha_value(),
            "--runner-size-bytes",
            "1",
            "--runner-tests-sha256",
            _sha_value("b"),
            "--runner-tests-size-bytes",
            "1",
            "--diagnostic-once",
            "--authorization-sha256",
            _sha_value("c"),
            "--authorization-size-bytes",
            "1",
        ],
    )
    assert runner.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "TERMINAL_ENGINEERING_FAILURE"


def test_main_completed_d01_to_d05_or_pass_exits_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **kwargs: object())
    monkeypatch.setattr(runner, "_validate_context_integrity", lambda context: None)
    monkeypatch.setattr(runner, "run_one_shot", lambda *args, **kwargs: {"status": "DIAGNOSTIC_COMPLETE_NONAUTHORIZING"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--runner-sha256",
            _sha_value(),
            "--runner-size-bytes",
            "1",
            "--runner-tests-sha256",
            _sha_value("b"),
            "--runner-tests-size-bytes",
            "1",
            "--diagnostic-once",
            "--authorization-sha256",
            _sha_value("c"),
            "--authorization-size-bytes",
            "1",
        ],
    )
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DIAGNOSTIC_COMPLETE_NONAUTHORIZING"
