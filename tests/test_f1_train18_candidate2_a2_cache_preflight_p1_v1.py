from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import preflight_f1_train18_candidate2_a2_real_cache_schema_p1_v1 as preflight


ROOT = Path(__file__).resolve().parents[1]


def _base(probability: float = 0.5):
    def classifier_probabilities(matrix, classifier):
        medians = np.asarray(classifier["imputer_medians"], dtype=np.float64)
        imputed = np.asarray(matrix, dtype=np.float64).copy()
        rows, columns = np.where(~np.isfinite(imputed))
        if len(rows):
            imputed[rows, columns] = medians[columns]
        assert np.isfinite(imputed).all()
        return np.full(len(imputed), probability, dtype=np.float64)

    return SimpleNamespace(np=np, classifier_probabilities=classifier_probabilities)


def test_synthetic_gate_model_is_complete_for_each_policy_mode() -> None:
    base = SimpleNamespace(feature_names=lambda representation: ("abs_score",))
    runtime = SimpleNamespace(HASH_BINDINGS=("fit_audit_core_sha256",), MODEL_KIND="MODEL")
    for mode in ("EXPLOSION_ONLY", "VETO_ONLY", "EXPLOSION_THEN_VETO"):
        model = preflight._synthetic_gate_model(
            runtime=runtime,
            base=base,
            mode=mode,
            representation=None if mode == "EXPLOSION_ONLY" else "absolute_quality",
            explosion_rule=None if mode == "VETO_ONLY" else "DROP_ALL_SIDECARS_IF_AT_LEAST_2",
        )
        assert model["kind"] == "MODEL"
        assert model["engineering_amendment"] == 2
        assert model["profile"]["policy_mode"] == mode


def test_numeric_core_nan_missing_sentinel_qualifies_after_imputation() -> None:
    matrix = np.asarray([[np.nan, 1.0, 0.0], [0.2, 0.0, 1.0]])
    probabilities = preflight._qualify_veto_matrix(
        base=_base(),
        matrix=matrix,
        names=("abs_selector_score", "missing_selector_score", "source_plane"),
        classifier={"imputer_medians": [0.0, 0.0, 0.0]},
    )
    assert probabilities.tolist() == [0.5, 0.5]


def test_non_core_nan_is_rejected() -> None:
    matrix = np.asarray([[0.1, np.nan]])
    with pytest.raises(ValueError, match="non-core"):
        preflight._qualify_veto_matrix(
            base=_base(),
            matrix=matrix,
            names=("abs_score", "source_plane"),
            classifier={"imputer_medians": [0.0, 0.0]},
        )


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_numeric_core_infinity_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="infinity"):
        preflight._qualify_veto_matrix(
            base=_base(),
            matrix=np.asarray([[value, 0.0]]),
            names=("abs_score", "source_plane"),
            classifier={"imputer_medians": [0.0, 0.0]},
        )


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_post_imputation_probability_must_be_finite_unit_interval(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        preflight._qualify_veto_matrix(
            base=_base(probability),
            matrix=np.asarray([[np.nan, 0.0]]),
            names=("abs_score", "source_plane"),
            classifier={"imputer_medians": [0.0, 0.0]},
        )


def test_p0_source_and_failure_are_exact_canonical_bindings() -> None:
    assert hashlib.sha256(preflight.P0_PREFLIGHT_SOURCE.read_bytes()).hexdigest() == (
        preflight.P0_PREFLIGHT_SOURCE_SHA256
    )
    failure_bytes = preflight.P0_FAILURE.read_bytes()
    assert hashlib.sha256(failure_bytes).hexdigest() == preflight.P0_FAILURE_SHA256
    failure = json.loads(failure_bytes)
    assert preflight.canonical_bytes(failure) == failure_bytes
    assert failure["status"] == "FAIL_CLOSED"
    assert failure["preflight_pass_created"] is False
    assert failure["label_path_authorized_or_opened"] is False
    assert failure["metric_or_score_computed"] is False
    assert failure["holdout_access_count"] == 0


def test_p1_top_level_is_stdlib_only() -> None:
    tree = ast.parse(preflight.PREFLIGHT_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "scripts", "numpy", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_alternate_attestation_rejected_before_verified_import(tmp_path) -> None:
    alternate = tmp_path / preflight.CACHE_ATTESTATION.name
    alternate.write_bytes(preflight.CACHE_ATTESTATION.read_bytes())
    with pytest.raises(PermissionError, match="immutable custody-v2"):
        preflight.run_preflight(
            cache_attestation=alternate,
            cache_attestation_sha256=preflight.CACHE_ATTESTATION_SHA256,
            runtime_code_sha256="0" * 64,
            preflight_code_sha256="0" * 64,
            output_path=preflight.PREFLIGHT_PASS_OUTPUT,
        )


def test_verified_loader_executes_prehashed_bytes_under_path_swap(tmp_path) -> None:
    runtime_path = tmp_path / "runtime.py"
    sentinel = tmp_path / "sentinel.txt"
    safe = b"VALUE = 'SAFE'\n"
    runtime_path.write_bytes(safe)
    code = f"""
import builtins, hashlib, importlib.util, pathlib, sys
path = pathlib.Path({str(preflight.PREFLIGHT_SOURCE)!r})
spec = importlib.util.spec_from_file_location('isolated_p1', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_p1'] = module
spec.loader.exec_module(module)
runtime_path = pathlib.Path({str(runtime_path)!r})
sentinel = pathlib.Path({str(sentinel)!r})
tampered = \"from pathlib import Path\\nPath(\" + repr(str(sentinel)) + \").write_text('EXECUTED')\\n\"
module.RUNTIME_SOURCE = runtime_path
module.A2_RUNTIME_MODULE = 'backend.synthetic_p1_runtime'
original_compile = builtins.compile
def swap(source, filename, mode, *args, **kwargs):
    if filename == str(module.BASE_RUNTIME_SOURCE.resolve()):
        runtime_path.write_text(tampered)
    return original_compile(source, filename, mode, *args, **kwargs)
builtins.compile = swap
_, runtime, _, _ = module._verified_runtime_imports(
    runtime_code_sha256=hashlib.sha256({safe!r}).hexdigest(),
    preflight_code_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
)
assert runtime.VALUE == 'SAFE'
assert not sentinel.exists()
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


def test_help_imports_no_runtime_numpy_or_label_module() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(preflight.PREFLIGHT_SOURCE), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    for forbidden in (
        "numpy",
        "optimize_f1_train18_source_only_v1",
        "evaluate_f1_against_manual_gold",
    ):
        assert forbidden not in completed.stderr
