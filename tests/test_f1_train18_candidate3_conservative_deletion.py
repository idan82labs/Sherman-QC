from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from backend.f1_train18_candidate3_conservative_deletion import (
    ALLOWED_NUMERIC_FEATURES,
    PROFILE,
    apply_candidate3,
    canonical_sha256,
    extract_prediction_features,
    preflight_candidate3_source_compatibility,
    runtime_source_sha256,
)
from scripts.optimize_f1_aug30_train18_candidate3_conservative_deletion import (
    PreparedCase,
    complete_nested_oof,
    fit_rule_for_strategy,
    strategy_space,
)


HEX = {
    "candidate1_terminal_sha256": "0" * 64,
    "candidate2_terminal_sha256": "1" * 64,
    "split_sha256": "2" * 64,
    "train_allowlist_sha256": "3" * 64,
    "original_prereg_sha256": "4" * 64,
    "candidate3_prereg_sha256": "5" * 64,
    "custody_label_attestation_sha256": "6" * 64,
    "custody_cache_attestation_sha256": "7" * 64,
    "runtime_source_sha256": runtime_source_sha256(),
    "optimizer_source_sha256": "a" * 64,
    "tests_source_sha256": "b" * 64,
}


def _prediction(token: str, *, score: float | None = 0.5, y: float = 0.0) -> dict:
    return {
        "bend_id": "P-" + token,
        "line_start": [0.0, y, 0.0],
        "line_end": [100.0, y, 0.0],
        "selection_route": "synthetic_test",
        "selector_score": score,
        "source_hypothesis_id": token,
        "source_kind": "plane_lattice",
    }


def _candidate(token: str, *, transition: float) -> dict:
    return {
        "candidate_id": token,
        "normal_second_eigenvalue": 0.3,
        "p90_line_distance_mm": 1.0,
        "median_line_distance_mm": 0.7,
        "two_sided_transition_score": transition,
        "two_sided_normal_angle_deg": 70.0,
        "two_sided_support_balance": 0.9,
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "transverse_angular_bin_fraction": 0.8,
        "min_surface_gap": 0.1,
    }


def _v51(predictions: list[dict]) -> dict:
    return {
        "predictions": predictions,
        "v51_result_sha256": "untrusted-payload-claim",
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "expected_count_used": False,
    }


def _pass_artifact(rule: dict, *, passed: bool = True) -> dict:
    checks = {
        "nested_oof_f1_increase": passed,
        "nested_oof_true_positive_nondecrease": passed,
        "nested_oof_false_positive_decrease": passed,
        "nested_oof_exact_case_nondecrease": passed,
        "nested_oof_support_tier_tp_nondecrease": passed,
        "primary_replay_exact": passed,
    }
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_train18_candidate3_train_gate_pass_v1",
        "status": "PASS" if passed else "FAIL",
        "attempt_ordinal": 3,
        "holdout_score_opened": False,
        "selected_deletion_rule": rule,
        "bindings": deepcopy(HEX),
        "checks": checks,
        "fit_audit_binding": {
            "kind": "sherman_qc_f1_aug30_train18_candidate3_fit_audit_v1",
            "sha256": "c" * 64,
        },
        "train_read_ledger_binding": {
            "kind": "sherman_qc_f1_aug30_train18_candidate3_read_ledger_v1",
            "sha256": "d" * 64,
        },
        "nested_oof_metrics": {
            "reference": {
                "true_positive": 4,
                "false_positive": 10,
                "false_negative": 6,
                "f1": 0.333333,
                "exact_case_count": 1,
                "support_tier_tp": {"1": 2, "2": 1, "3": 1},
            },
            "candidate": {
                "true_positive": 4,
                "false_positive": 8,
                "false_negative": 6,
                "f1": 0.363636,
                "exact_case_count": 1,
                "support_tier_tp": {"1": 2, "2": 1, "3": 1},
            },
            "delta": {
                "true_positive": 0,
                "false_positive": -2,
                "false_negative": 0,
                "f1": 0.030303,
                "exact_case_count": 0,
                "support_tier_tp": {"1": 0, "2": 0, "3": 0},
            },
        },
    }


def _model(rule: dict, pass_artifact: dict, *, bindings: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_train18_candidate3_conservative_deletion_model_v1",
        "profile": PROFILE,
        "attempt_ordinal": 3,
        "holdout_score_opened": False,
        "deletion_rule": rule,
        "source_only": True,
        "uses_prediction_or_candidate_multiplicity": False,
        "forbidden_inference_inputs": [],
        "bindings": deepcopy(bindings if bindings is not None else HEX),
        "train_gate_pass_binding": {
            "kind": "sherman_qc_f1_train18_candidate3_train_gate_pass_v1",
            "sha256": canonical_sha256(pass_artifact),
        },
    }


def _apply(source: dict, pool: dict, rule: dict, *, passed: bool = True, model_bindings=None):
    evidence = _pass_artifact(rule, passed=passed)
    model = _model(rule, evidence, bindings=model_bindings)
    return apply_candidate3(
        source,
        pool,
        model,
        evidence,
        train_gate_pass_artifact_sha256=canonical_sha256(evidence),
        v51_artifact_sha256="8" * 64,
        pool_artifact_sha256="9" * 64,
    )


def test_nonmatching_rule_preserves_exact_v51_predictions():
    source = _v51([_prediction("one"), _prediction("two", y=10.0)])
    rule = {"feature": "two_sided_transition_score", "operator": "le", "threshold": 0.1}
    result = _apply(source, {"candidates": []}, rule)
    assert result["predictions"] == source["predictions"]
    assert result["deletion_audit"] == []
    assert result["candidate_count"] == 2


def test_deletes_only_a_matching_low_tail_and_keeps_missing_evidence():
    weak = _prediction("weak")
    strong = _prediction("strong", y=10.0)
    missing = _prediction("missing", y=20.0)
    pool = {
        "candidates": [
            _candidate("weak", transition=0.2),
            _candidate("strong", transition=0.8),
        ]
    }
    rule = {"feature": "two_sided_transition_score", "operator": "le", "threshold": 0.3}
    result = _apply(_v51([weak, strong, missing]), pool, rule)
    assert [row["bend_id"] for row in result["predictions"]] == ["P-strong", "P-missing"]
    assert result["deletion_audit"][0]["prediction_id"] == "P-weak"
    assert result["deletion_audit"][0]["reason"] == "frozen_train_nested_rule"


def test_source_family_rule_is_supported_without_using_identity():
    first = _prediction("first")
    second = _prediction("second", y=10.0)
    first["source_kind"] = "strict_blind_owned"
    second["source_kind"] = "plane_lattice"
    rule = {"feature": "source_kind", "operator": "eq", "value": "strict_blind_owned"}
    result = _apply(_v51([first, second]), {"candidates": []}, rule)
    assert [row["bend_id"] for row in result["predictions"]] == ["P-second"]


def test_extracted_features_have_no_identity_or_count_fields():
    features = extract_prediction_features(
        _prediction("opaque"), _candidate("opaque", transition=0.7)
    )
    assert set(features) == set(ALLOWED_NUMERIC_FEATURES) | {"source_kind"}
    forbidden_tokens = ("case", "file", "part", "identity", "count", "expected", "gold", "label")
    assert not any(token in key for key in features for token in forbidden_tokens)


@pytest.mark.parametrize(
    "rule",
    [
        {"feature": "case_id", "operator": "eq", "value": "A30-001"},
        {"feature": "prediction_count", "operator": "ge", "threshold": 3},
        {"feature": "expected_count", "operator": "ge", "threshold": 1},
        {"feature": "selector_score", "operator": "between", "threshold": 0.5},
    ],
)
def test_rejects_prohibited_or_unknown_rule_surfaces(rule):
    evidence = _pass_artifact(rule)
    model = _model(rule, evidence)
    with pytest.raises(ValueError):
        apply_candidate3(
            _v51([_prediction("one")]),
            {"candidates": []},
            model,
            evidence,
            train_gate_pass_artifact_sha256=canonical_sha256(evidence),
            v51_artifact_sha256="8" * 64,
            pool_artifact_sha256="9" * 64,
        )


def test_failed_train_gate_is_non_executable_not_passthrough():
    rule = {"feature": "two_sided_transition_score", "operator": "le", "threshold": 0.3}
    with pytest.raises(ValueError):
        _apply(_v51([_prediction("weak")]), {"candidates": []}, rule, passed=False)


@pytest.mark.parametrize(
    "flag", ["labels_used", "cad_used", "part_identity_used", "expected_count_used"]
)
def test_requires_strict_false_source_only_flags(flag):
    source = _v51([])
    source[flag] = None
    rule = {"feature": "selector_score", "operator": "le", "threshold": 0.1}
    with pytest.raises(ValueError):
        _apply(source, {"candidates": []}, rule)


def test_rejects_gate_hash_or_rule_mismatch():
    rule = {"feature": "selector_score", "operator": "le", "threshold": 0.1}
    evidence = _pass_artifact(rule)
    model = _model(rule, evidence)
    with pytest.raises(ValueError):
        apply_candidate3(
            _v51([]),
            {"candidates": []},
            model,
            evidence,
            train_gate_pass_artifact_sha256="a" * 64,
            v51_artifact_sha256="8" * 64,
            pool_artifact_sha256="9" * 64,
        )
    tampered = deepcopy(evidence)
    tampered["selected_deletion_rule"] = {
        "feature": "selector_score",
        "operator": "ge",
        "threshold": 2.0,
    }
    with pytest.raises(ValueError):
        apply_candidate3(
            _v51([]),
            {"candidates": []},
            model,
            tampered,
            train_gate_pass_artifact_sha256=canonical_sha256(tampered),
            v51_artifact_sha256="8" * 64,
            pool_artifact_sha256="9" * 64,
        )


def test_rejects_mismatched_code_binding():
    bindings = deepcopy(HEX)
    bindings["runtime_source_sha256"] = "b" * 64
    rule = {"feature": "selector_score", "operator": "le", "threshold": 0.1}
    with pytest.raises(ValueError):
        _apply(_v51([]), {"candidates": []}, rule, model_bindings=bindings)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("bend_id"),
        lambda row: row.__setitem__("line_start", [float("nan"), 0.0, 0.0]),
        lambda row: row.__setitem__("selector_score", float("inf")),
        lambda row: row.__setitem__("source_kind", ""),
        lambda row: row.__setitem__("unexpected", 1),
    ],
)
def test_rejects_malformed_prediction_schema_and_nonfinite_values(mutation):
    prediction = _prediction("bad")
    mutation(prediction)
    rule = {"feature": "selector_score", "operator": "le", "threshold": 0.1}
    with pytest.raises(ValueError):
        _apply(_v51([prediction]), {"candidates": []}, rule)


def test_rejects_duplicate_prediction_ids_and_invalid_external_hashes():
    row = _prediction("same")
    rule = {"feature": "selector_score", "operator": "le", "threshold": 0.1}
    with pytest.raises(ValueError):
        _apply(_v51([row, deepcopy(row)]), {"candidates": []}, rule)
    evidence = _pass_artifact(rule)
    model = _model(rule, evidence)
    with pytest.raises(ValueError):
        apply_candidate3(
            _v51([]),
            {"candidates": []},
            model,
            evidence,
            train_gate_pass_artifact_sha256=canonical_sha256(evidence),
            v51_artifact_sha256="not-a-hash",
            pool_artifact_sha256="9" * 64,
        )


def test_pool_permutation_is_exact_and_inputs_are_not_mutated():
    source = _v51([_prediction("weak"), _prediction("strong", y=10.0)])
    candidates = [
        _candidate("weak", transition=0.2),
        _candidate("strong", transition=0.8),
    ]
    pool = {"candidates": candidates}
    rule = {"feature": "two_sided_transition_score", "operator": "le", "threshold": 0.3}
    evidence = _pass_artifact(rule)
    model = _model(rule, evidence)
    before = (deepcopy(source), deepcopy(pool), deepcopy(model), deepcopy(evidence))
    kwargs = {
        "train_gate_pass_artifact_sha256": canonical_sha256(evidence),
        "v51_artifact_sha256": "8" * 64,
        "pool_artifact_sha256": "9" * 64,
    }
    first = apply_candidate3(source, pool, model, evidence, **kwargs)
    second = apply_candidate3(
        source, {"candidates": list(reversed(candidates))}, model, evidence, **kwargs
    )
    assert first == second
    assert (source, pool, model, evidence) == before
    assert first["profile"] == PROFILE
    assert first["labels_used"] is False
    assert first["expected_count_used"] is False
    assert first["case_identity_used"] is False
    assert first["filename_used"] is False
    assert first["sender_count_used"] is False
    assert first["drawing_used"] is False
    assert first["uses_prediction_or_candidate_multiplicity"] is False
    assert first["promotion_eligible"] is False
    assert first["input_v51_artifact_sha256"] == "8" * 64
    assert first["input_pool_artifact_sha256"] == "9" * 64


def _training_case(token: str, *, true_transition: float = 0.8, false_transition: float = 0.2) -> PreparedCase:
    true_prediction = _prediction(token + "-true", y=0.0)
    false_prediction = _prediction(token + "-false", y=10.0)
    return PreparedCase(
        case_id="OPAQUE-" + token,
        gold=(
            {
                "bend_id": "G-" + token,
                "line_start": [0.0, 0.0, 0.0],
                "line_end": [100.0, 0.0, 0.0],
                "reviewer_support_count": 1 + int(token) % 3,
            },
        ),
        v51=_v51([true_prediction, false_prediction]),
        pool={
            "candidates": [
                _candidate(token + "-true", transition=true_transition),
                _candidate(token + "-false", transition=false_transition),
            ]
        },
        v51_artifact_sha256="8" * 64,
        pool_artifact_sha256="9" * 64,
    )


def test_complete_nested_oof_deletes_only_repeated_false_tail_without_harm():
    cases = [_training_case(str(index)) for index in range(6)]
    result = complete_nested_oof(cases)
    assert result["reference"]["true_positive"] == 6
    assert result["reference"]["false_positive"] == 6
    assert result["candidate"]["true_positive"] == 6
    assert result["candidate"]["false_positive"] == 0
    assert result["delta"]["exact_case_count"] == 6
    assert result["delta"]["support_tier_tp"] == {"1": 0, "2": 0, "3": 0}
    assert all(result["checks"].values())
    assert len(result["folds"]) == 6


def test_harmful_tail_is_rejected_by_fit_gate():
    cases = [
        _training_case(str(index), true_transition=0.1, false_transition=0.2)
        for index in range(4)
    ]
    rule, audit = fit_rule_for_strategy(
        cases, {"feature": "two_sided_transition_score", "operator": "le"}
    )
    assert rule is None
    assert audit["eligible_rule_count"] == 0


def test_adaptive_strategy_space_has_no_identity_or_multiplicity_feature():
    strategies = strategy_space()
    assert len(strategies) == 2 * len(ALLOWED_NUMERIC_FEATURES) + 1
    assert not any(
        token in strategy["feature"]
        for strategy in strategies
        for token in ("case", "file", "part", "count", "expected", "gold", "label")
    )


def test_label_free_preflight_accepts_null_selector_score_and_missing_candidate_row():
    prediction = _prediction("sidecar", score=None)
    prediction["source_kind"] = "v41_fragment_tolerant_parent_bridge_sidecar"
    result = preflight_candidate3_source_compatibility(
        _v51([prediction]),
        {"candidates": []},
        v51_artifact_sha256="8" * 64,
        pool_artifact_sha256="9" * 64,
    )
    assert result["status"] == "SOURCE_COMPATIBLE"
    assert result["feature_extraction_complete"] is True
    assert result["all_numeric_le_ge_and_categorical_paths_exercised"] is True
    assert result["semantic_internal_scan_identity_trusted"] is False
    forbidden = {
        "prediction_row_count",
        "feature_row_count",
        "rule_path_check_count",
        "missing_numeric_feature_counts",
        "prediction_count",
        "candidate_count",
    }
    assert forbidden.isdisjoint(result)


def test_label_free_preflight_ignores_internal_scan_identity_claim():
    first = _v51([_prediction("one")])
    second = deepcopy(first)
    first["scan_sha256"] = "untrusted-first"
    second["scan_sha256"] = "untrusted-second"
    kwargs = {"v51_artifact_sha256": "8" * 64, "pool_artifact_sha256": "9" * 64}
    assert preflight_candidate3_source_compatibility(first, {"candidates": []}, **kwargs) == (
        preflight_candidate3_source_compatibility(second, {"candidates": []}, **kwargs)
    )


def test_label_free_preflight_pool_permutation_is_exact():
    source = _v51([_prediction("one"), _prediction("two", y=10.0)])
    candidates = [_candidate("one", transition=0.2), _candidate("two", transition=0.8)]
    kwargs = {"v51_artifact_sha256": "8" * 64, "pool_artifact_sha256": "9" * 64}
    first = preflight_candidate3_source_compatibility(
        source, {"candidates": candidates}, **kwargs
    )
    second = preflight_candidate3_source_compatibility(
        source, {"candidates": list(reversed(candidates))}, **kwargs
    )
    assert first == second


def _source_hashes() -> tuple[Path, str, str, str]:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/preflight_f1_aug30_train18_candidate3_source_compatibility.py"
    runtime = root / "backend/f1_train18_candidate3_conservative_deletion.py"
    return (
        script,
        hashlib.sha256(runtime.read_bytes()).hexdigest(),
        hashlib.sha256(script.read_bytes()).hexdigest(),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def test_preflight_stdlib_bootstrap_verifies_exact_source_bytes_in_subprocess():
    script, runtime_sha, preflight_sha, tests_sha = _source_hashes()
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runtime-sha256",
            runtime_sha,
            "--preflight-sha256",
            preflight_sha,
            "--tests-sha256",
            tests_sha,
            "--source-check-only",
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SOURCE_BYTES_VERIFIED_ONLY" in completed.stdout


def test_preflight_subprocess_rejects_tampered_source_binding_before_cache_access():
    script, _runtime_sha, preflight_sha, tests_sha = _source_hashes()
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runtime-sha256",
            "0" * 64,
            "--preflight-sha256",
            preflight_sha,
            "--tests-sha256",
            tests_sha,
            "--source-check-only",
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "runtime source SHA mismatch" in completed.stderr


def test_preflight_subprocess_rejects_preloaded_canonical_runtime():
    script, runtime_sha, preflight_sha, tests_sha = _source_hashes()
    code = """
import runpy
import sys
import backend.f1_train18_candidate3_conservative_deletion
script, runtime_sha, preflight_sha, tests_sha = sys.argv[1:]
sys.argv = [script, '--runtime-sha256', runtime_sha, '--preflight-sha256', preflight_sha, '--tests-sha256', tests_sha, '--source-check-only']
runpy.run_path(script, run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(script), runtime_sha, preflight_sha, tests_sha],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "preloaded canonical runtime module" in completed.stderr


def test_atomic_publication_subprocess_leaves_no_partial_output(tmp_path):
    script, _runtime_sha, _preflight_sha, _tests_sha = _source_hashes()
    code = """
import importlib.util
from pathlib import Path
import sys
script = Path(sys.argv[1])
parent = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('_candidate3_preflight_test', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target = parent / 'published'
def fail(_staging):
    raise RuntimeError('injected-publication-failure')
try:
    module._publish_atomically(target, {'artifact.json': b'{}'}, after_stage_write=fail)
except RuntimeError as exc:
    assert str(exc) == 'injected-publication-failure'
else:
    raise AssertionError('injected publication failure did not fire')
assert not target.exists()
assert not list(parent.glob('.published.staging-*'))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(script), str(tmp_path)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_persisted_preflight_report_schema_forbids_source_volume_keys():
    script, _runtime_sha, _preflight_sha, _tests_sha = _source_hashes()
    code = """
import importlib.util
from pathlib import Path
import sys
script = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_candidate3_preflight_schema_test', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.FORBIDDEN_SOURCE_VOLUME_KEYS.isdisjoint(module.PERSISTED_REPORT_KEYS)
for key in module.FORBIDDEN_SOURCE_VOLUME_KEYS:
    try:
        module._assert_no_forbidden_source_volume_keys({key: 1})
    except ValueError:
        pass
    else:
        raise AssertionError(f'forbidden key accepted: {key}')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
