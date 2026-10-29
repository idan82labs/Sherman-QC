from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as hypothesis
from backend import f1_train18_candidate7_conservative_recall_recovery_v1 as candidate7
from backend import f1_train18_candidate8_diagnostic_recall_wrapper_v1 as runtime
from scripts import preflight_f1_train18_candidate8_diagnostic_adapter_v1 as adapter


ROOT = Path(__file__).resolve().parents[1]


def _source_hashes() -> dict[str, str]:
    return {
        "adapter": adapter.sha256_bytes(adapter.ADAPTER_SOURCE.read_bytes()),
        "adapter_tests": adapter.sha256_bytes(adapter.ADAPTER_TESTS_SOURCE.read_bytes()),
        "candidate7_runtime": adapter.sha256_bytes(
            adapter.CANDIDATE7_RUNTIME_SOURCE.read_bytes()
        ),
        "hypothesis": adapter.sha256_bytes(adapter.HYPOTHESIS_SOURCE.read_bytes()),
        "runtime": adapter.sha256_bytes(adapter.RUNTIME_SOURCE.read_bytes()),
        "runtime_tests": adapter.sha256_bytes(adapter.RUNTIME_TESTS_SOURCE.read_bytes()),
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
                    "selection_route": "v51-synthetic",
                    "selector_score": None,
                    "source_hypothesis_id": "V51-H0",
                    "source_kind": "plane_lattice",
                }
            ],
            "research_only": True,
            "source_only": True,
        }
    )


def _pool() -> bytes:
    return hypothesis.canonical_bytes(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
            "candidate_count": 99,
            "candidate_source_counts": {"audit-only": 99},
            "candidates": [{"candidate_id": "RAW-IGNORED"}],
            "inference_contract": {
                "cad_exposed": False,
                "expected_count_exposed": False,
                "manual_truth_exposed": False,
                "part_identifier_exposed": False,
            },
            "line_object_count": 1,
            "line_objects": [
                {
                    "candidate_id": "IGNORED-ID",
                    "endpoint_support_fraction": 0.8,
                    "finite_support_fraction": 0.8,
                    "normal_second_eigenvalue": 0.2,
                    "representative_source": "plane_lattice",
                    "score": 0.8,
                    "span_endpoints": [
                        [0.0, 8.0, 0.0],
                        [100.0, 8.0, 0.0],
                    ],
                    "two_sided_transition_score": 0.8,
                }
            ],
            "schema_version": 3,
            "source_point_serialization": False,
            "working_point_budget": 200000,
        }
    )


def _context() -> adapter.AdapterContext:
    hashes = _source_hashes()
    return adapter.AdapterContext(
        runtime=runtime,
        hypothesis=hypothesis,
        candidate7_runtime=candidate7,
        source_bindings={
            name: {"path": name, "sha256": sha, "size_bytes": 1}
            for name, sha in hashes.items()
        },
    )


def _diagnose(
    *, semantic: bytes | None = None, pool: bytes | None = None
) -> dict[str, object]:
    semantic_bytes = _semantic() if semantic is None else semantic
    pool_bytes = _pool() if pool is None else pool
    disposition = (ROOT / runtime.CANDIDATE7_DISPOSITION_PATH).read_bytes()
    return adapter.diagnose_bound_pair(
        _context(),
        semantic_bytes=semantic_bytes,
        pool_bytes=pool_bytes,
        candidate7_disposition_bytes=disposition,
        semantic_sha256=adapter.sha256_bytes(semantic_bytes),
        pool_sha256=adapter.sha256_bytes(pool_bytes),
        candidate7_disposition_sha256=runtime.CANDIDATE7_DISPOSITION_SHA256,
    )


def _source_check_command(hashes: dict[str, str]) -> list[str]:
    return [
        sys.executable,
        str(adapter.ADAPTER_SOURCE),
        "--runtime-sha256",
        hashes["runtime"],
        "--adapter-sha256",
        hashes["adapter"],
        "--runtime-tests-sha256",
        hashes["runtime_tests"],
        "--adapter-tests-sha256",
        hashes["adapter_tests"],
        "--hypothesis-sha256",
        hashes["hypothesis"],
        "--candidate7-runtime-sha256",
        hashes["candidate7_runtime"],
    ]


def test_adapter_top_level_is_stdlib_only_and_has_no_real_input_capability() -> None:
    tree = ast.parse(adapter.ADAPTER_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "scipy", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)
    for forbidden_name in (
        "CACHE_MANIFEST",
        "LABEL_ATTESTATION",
        "OUTPUT_ROOT",
        "PREFLIGHT_AUTHORIZATION",
        "TRAIN_ALLOWLIST",
    ):
        assert not hasattr(adapter, forbidden_name)
    assert not any(
        name.startswith(("publish", "run_source_inventory", "consume_authorization"))
        for name in vars(adapter)
    )


def test_source_check_only_verifies_exact_six_sources() -> None:
    hashes = _source_hashes()
    completed = subprocess.run(
        [*_source_check_command(hashes), "--source-check-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "artifact_discovery_or_open_capability_present": False,
        "artifact_input_opened": False,
        "cli_diagnostic_input_opened": False,
        "holdout_or_label_opened": False,
        "source_binding_count": 6,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_no_mode_rejects_before_source_verification_or_publication() -> None:
    completed = subprocess.run(
        _source_check_command(_source_hashes()),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "--source-check-only" in completed.stderr
    assert completed.stdout == ""


def test_help_does_not_import_controlled_or_numeric_modules() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(adapter.ADAPTER_SOURCE), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "f1_train18_candidate6_conservative_recall_recovery_v1",
        "f1_train18_candidate7_conservative_recall_recovery_v1",
        "f1_train18_candidate8_diagnostic_recall_wrapper_v1",
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
    alias.symlink_to(adapter.HYPOTHESIS_SOURCE)
    sentinel = tmp_path / "source-read"
    code = f"""
import importlib.util, pathlib, sys
alias = pathlib.Path({str(alias)!r})
spec = importlib.util.spec_from_file_location('arbitrary_alias', alias)
loaded = importlib.util.module_from_spec(spec)
sys.modules['arbitrary_alias'] = loaded
spec.loader.exec_module(loaded)
path = pathlib.Path({str(adapter.ADAPTER_SOURCE)!r})
spec2 = importlib.util.spec_from_file_location('isolated_c8_adapter', path)
module = importlib.util.module_from_spec(spec2)
sys.modules['isolated_c8_adapter'] = module
spec2.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_read(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('READ')
    raise AssertionError('source read preceded alias rejection')
module._read_source_once = forbidden_read
try:
    module.bootstrap_verified_sources(
        runtime_sha256={hashes['runtime']!r},
        adapter_sha256={hashes['adapter']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        adapter_tests_sha256={hashes['adapter_tests']!r},
        hypothesis_sha256={hashes['hypothesis']!r},
        candidate7_runtime_sha256={hashes['candidate7_runtime']!r},
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


def test_tampered_source_hash_fails_before_module_execution(tmp_path: Path) -> None:
    hashes = _source_hashes()
    sentinel = tmp_path / "module-executed"
    code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(adapter.ADAPTER_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_c8_tamper', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_c8_tamper'] = module
spec.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_exec(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('EXECUTED')
    raise AssertionError('module execution preceded source verification')
module._exec_module = forbidden_exec
try:
    module.bootstrap_verified_sources(
        runtime_sha256={'1' * 64!r},
        adapter_sha256={hashes['adapter']!r},
        runtime_tests_sha256={hashes['runtime_tests']!r},
        adapter_tests_sha256={hashes['adapter_tests']!r},
        hypothesis_sha256={hashes['hypothesis']!r},
        candidate7_runtime_sha256={hashes['candidate7_runtime']!r},
    )
except ValueError as error:
    assert 'binding mismatch' in str(error)
else:
    raise AssertionError('tampered source binding was accepted')
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


def test_actual_shaped_pair_passes_without_selecting_or_testing_a_rule() -> None:
    result = _diagnose()
    assert result["status"] == "SOURCE_COMPATIBLE"
    assert result["failure_phase"] is None
    assert result["candidate_rule_selected"] is False
    assert result["scientific_hypothesis_performance_tested"] is False
    assert result["candidate_metric_computed"] is False


@pytest.mark.parametrize(
    ("envelope", "forbidden_key", "expected_phase"),
    (
        ("semantic", "target_truth", runtime.PHASE_CONTRACT[0]),
        ("pool", "is_true_positive", runtime.PHASE_CONTRACT[1]),
        ("pool", "outcome", runtime.PHASE_CONTRACT[1]),
    ),
)
def test_actual_envelope_truth_aliases_fail_in_the_bound_phase(
    envelope: str, forbidden_key: str, expected_phase: str
) -> None:
    semantic = json.loads(_semantic())
    pool = json.loads(_pool())
    target = (
        semantic["predictions"][0]
        if envelope == "semantic"
        else pool["line_objects"][0]
    )
    target[forbidden_key] = "SECRET-VALUE"
    result = _diagnose(
        semantic=hypothesis.canonical_bytes(semantic),
        pool=hypothesis.canonical_bytes(pool),
    )
    assert result["failure_phase"] == expected_phase
    assert "SECRET" not in json.dumps(result)


def test_generic_accounting_and_raw_candidate_order_are_ignored_for_selection() -> None:
    baseline = _diagnose()
    pool = json.loads(_pool())
    pool["candidate_count"] = 1_000_000
    pool["candidate_source_counts"] = {"new-audit-key": 1_000_000}
    pool["line_object_count"] = 7_000_000
    pool["candidates"] = list(reversed(pool["candidates"]))
    assert _diagnose(pool=hypothesis.canonical_bytes(pool)) == baseline


def test_candidate7_wrapper_only_reserved_id_failure_remains_d05() -> None:
    semantic = json.loads(_semantic())
    semantic["predictions"][0]["bend_id"] = "C7R-RESERVED-SYNTHETIC"
    result = _diagnose(semantic=hypothesis.canonical_bytes(semantic))
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[4]
    assert result["check_code"] == "C8_WRAPPER_RESULT_VALIDATION_FAILED"


def test_adapter_rejects_nonbytes_and_unbound_hashes() -> None:
    disposition = (ROOT / runtime.CANDIDATE7_DISPOSITION_PATH).read_bytes()
    with pytest.raises(TypeError, match="immutable bytes"):
        adapter.diagnose_bound_pair(
            _context(),
            semantic_bytes="not-bytes",  # type: ignore[arg-type]
            pool_bytes=_pool(),
            candidate7_disposition_bytes=disposition,
            semantic_sha256=hashlib.sha256(b"not-bytes").hexdigest(),
            pool_sha256=adapter.sha256_bytes(_pool()),
            candidate7_disposition_sha256=runtime.CANDIDATE7_DISPOSITION_SHA256,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        adapter.diagnose_bound_pair(
            _context(),
            semantic_bytes=_semantic(),
            pool_bytes=_pool(),
            candidate7_disposition_bytes=disposition,
            semantic_sha256="bad",
            pool_sha256=adapter.sha256_bytes(_pool()),
            candidate7_disposition_sha256=runtime.CANDIDATE7_DISPOSITION_SHA256,
        )


def test_value_free_result_validator_rejects_added_source_detail() -> None:
    result = _diagnose()
    result["candidate_count"] = 1
    with pytest.raises((ValueError, TypeError)):
        adapter._assert_value_free_result(result, runtime)
