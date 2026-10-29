from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backend import f1_train18_candidate2_conservative_deletion_a3_v1 as runtime
from backend.f1_train18_frozen_source_only_candidate_v1 import (
    canonical_bytes,
    feature_names,
)
from scripts import preflight_f1_train18_candidate2_a3_real_cache_schema_v1 as preflight


ROOT = Path(__file__).resolve().parents[1]


def _row(
    bend_id: str,
    *,
    sidecar: bool = False,
    score: object = 0.5,
    y: float = 0.0,
) -> dict:
    return {
        "bend_id": bend_id,
        "line_start": [0.0, y, 0.0],
        "line_end": [10.0, y, 0.0],
        "selector_score": score,
        "source_kind": runtime.SIDECAR_SOURCE_KIND if sidecar else "plane_lattice",
        "selection_route": (
            "v41_fragment_tolerant_parent_bridge"
            if sidecar
            else "v16_full_resolution"
        ),
    }


def _source(rows: list[dict]) -> dict:
    return {
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
        "predictions": rows,
    }


def _pool() -> dict:
    return {"working_point_budget": 200000, "candidates": [], "line_objects": []}


def _hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(f"a3-test|{name}".encode("ascii")).hexdigest()
        for name in runtime.HASH_BINDINGS
    }


def _explosion_model(rule: str = runtime.EXPLOSION_RULES[0]) -> dict:
    return {
        "schema_version": 1,
        "kind": runtime.MODEL_KIND,
        "attempt_ordinal": 2,
        "engineering_amendment": 3,
        "training_gate_passed": True,
        "train_gate_result_kind": (
            "sherman_qc_f1_aug30_train18_candidate2_a3_train_gate_pass_v1"
        ),
        "train_gate_status": "PASS",
        "holdout_score_opened": False,
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "source_only": True,
        "v51_fallback_default": True,
        "deletion_only": True,
        "case_id_feature_used": False,
        "runtime_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "observed_source_multiplicity_used": True,
        "classifier_count_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "profile_name": "SYNTHETIC_A3",
        "profile": {"policy_mode": "EXPLOSION_ONLY", "explosion_rule": rule},
        "feature_names": None,
        "classifier": None,
        **_hashes(),
    }


def _veto_model(probability: float = 1.0, threshold: float = 0.5) -> dict:
    names = feature_names("absolute_quality")
    model = _explosion_model()
    model.update(
        {
            "observed_source_multiplicity_used": False,
            "profile": {
                "policy_mode": "VETO_ONLY",
                "learner": "LABEL_FREE_SYNTHETIC_CONSTANT",
                "representation": "absolute_quality",
                "veto_keep_threshold": threshold,
            },
            "feature_names": list(names),
            "classifier": {
                "imputer_medians": [0.0] * len(names),
                "kind": "constant",
                "positive_probability": probability,
            },
        }
    )
    return model


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


def test_a3_search_space_contains_only_two_drop_all_rules() -> None:
    assert runtime.EXPLOSION_RULES == (
        "DROP_ALL_SIDECARS_IF_AT_LEAST_2",
        "DROP_ALL_SIDECARS_IF_AT_LEAST_3",
    )
    assert runtime.POLICY_MODES == (
        "EXPLOSION_ONLY",
        "VETO_ONLY",
        "EXPLOSION_THEN_VETO",
    )


def test_retired_ranked_sidecar_rule_is_explicitly_rejected() -> None:
    retired = "KEEP" + "_TOP1_SIDECAR_IF_AT_LEAST_2"
    with pytest.raises(ValueError, match="unsupported explosion rule"):
        runtime.apply_explosion_guard([_row("s", sidecar=True)], retired)


@pytest.mark.parametrize("rule,sidecar_count", [(runtime.EXPLOSION_RULES[0], 2), (runtime.EXPLOSION_RULES[1], 3)])
def test_drop_all_rules_preserve_every_non_sidecar_byte(
    rule: str, sidecar_count: int
) -> None:
    base = _row("base")
    rows = [base] + [_row(f"s{index}", sidecar=True) for index in range(sidecar_count)]
    retained, removed = runtime.apply_explosion_guard(rows, rule)
    assert canonical_bytes(retained) == canonical_bytes([base])
    assert removed == [f"s{index}" for index in range(sidecar_count)]


@pytest.mark.parametrize("score_mode", ["missing", "null"])
@pytest.mark.parametrize("sidecar", [False, True])
def test_missing_or_null_selector_score_is_accepted(
    score_mode: str, sidecar: bool
) -> None:
    row = _row("row", sidecar=sidecar)
    if score_mode == "missing":
        row.pop("selector_score")
    else:
        row["selector_score"] = None
    assert runtime.validate_incumbent_predictions([row]) == [row]


@pytest.mark.parametrize(
    "invalid", ["0.5", True, float("nan"), float("inf"), float("-inf")]
)
@pytest.mark.parametrize("sidecar", [False, True])
def test_present_selector_score_must_be_finite_numeric_non_bool(
    invalid: object, sidecar: bool
) -> None:
    with pytest.raises(ValueError, match="selector_score"):
        runtime.validate_incumbent_predictions(
            [_row("row", sidecar=sidecar, score=invalid)]
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[1].update(bend_id=rows[0]["bend_id"]),
        lambda rows: rows[0].pop("bend_id"),
        lambda rows: rows[0].update(line_end=[0.0, 0.0, 0.0]),
        lambda rows: rows[0].update(line_start=[float("nan"), 0.0, 0.0]),
        lambda rows: rows[0].update(source_kind=""),
        lambda rows: rows[0].update(selection_route=""),
    ],
)
def test_malformed_incumbent_stream_is_rejected(mutator) -> None:
    rows = [_row("base"), _row("sidecar", sidecar=True)]
    mutator(rows)
    with pytest.raises(ValueError):
        runtime.validate_incumbent_predictions(rows)


def test_model_requires_a3_gate_and_every_lineage_binding() -> None:
    model = _explosion_model()
    assert runtime.validate_model(model)["policy_mode"] == "EXPLOSION_ONLY"
    for field in runtime.HASH_BINDINGS:
        broken = deepcopy(model)
        broken.pop(field)
        with pytest.raises(ValueError):
            runtime.validate_model(broken)


def test_model_rejects_policy_fields_outside_locked_mode_schema() -> None:
    model = _explosion_model()
    model["profile"]["veto_keep_threshold"] = 0.5
    with pytest.raises(ValueError, match="profile fields"):
        runtime.validate_model(model)
    for field, value in (
        ("engineering_amendment", 2),
        ("training_gate_passed", False),
        ("train_gate_status", "FAIL"),
        ("train_gate_result_kind", "wrong"),
    ):
        broken = deepcopy(model)
        broken[field] = value
        with pytest.raises(ValueError):
            runtime.validate_model(broken)


def test_multiplicity_is_separate_from_forbidden_truth_counts() -> None:
    model = _explosion_model()
    assert model["observed_source_multiplicity_used"] is True
    assert "count_feature_used" not in model
    for field in (
        "classifier_count_feature_used",
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
    ):
        assert model[field] is False
    broken = deepcopy(model)
    broken["observed_source_multiplicity_used"] = False
    with pytest.raises(ValueError, match="multiplicity"):
        runtime.validate_model(broken)


def test_all_policy_modes_are_deletion_only_and_retained_rows_are_identical() -> None:
    rows = [_row("base"), _row("s1", sidecar=True), _row("s2", sidecar=True)]
    explosion = runtime.compose_candidate2(_source(rows), _pool(), _explosion_model())
    assert canonical_bytes(explosion["predictions"]) == canonical_bytes([rows[0]])

    veto = runtime.compose_candidate2(_source(rows), _pool(), _veto_model())
    assert canonical_bytes(veto["predictions"]) == canonical_bytes(rows)

    union = _veto_model()
    union["observed_source_multiplicity_used"] = True
    union["profile"]["policy_mode"] = "EXPLOSION_THEN_VETO"
    union["profile"]["explosion_rule"] = runtime.EXPLOSION_RULES[0]
    composed = runtime.compose_candidate2(_source(rows), _pool(), union)
    assert canonical_bytes(composed["predictions"]) == canonical_bytes([rows[0]])
    for result in (explosion, veto, composed):
        assert result["engineering_amendment"] == 3
        assert result["deletion_only"] is True
        assert result["training_gate_passed"] is True
        assert result["holdout_score_opened"] is False


def test_runtime_and_preflight_sources_have_no_retired_ranked_rule_contract() -> None:
    retired_token = "KEEP" + "_TOP1"
    assert retired_token not in runtime.__file__ and retired_token not in str(
        preflight.PREFLIGHT_SOURCE
    )
    assert retired_token not in Path(runtime.__file__).read_text(encoding="utf-8")
    assert retired_token not in preflight.PREFLIGHT_SOURCE.read_text(encoding="utf-8")


def test_synthetic_gate_model_is_complete_for_every_policy_mode() -> None:
    base = SimpleNamespace(feature_names=lambda representation: ("abs_score",))
    synthetic_runtime = SimpleNamespace(
        HASH_BINDINGS=runtime.HASH_BINDINGS, MODEL_KIND=runtime.MODEL_KIND
    )
    for mode in runtime.POLICY_MODES:
        model = preflight._synthetic_gate_model(
            runtime=synthetic_runtime,
            base=base,
            mode=mode,
            representation=None if mode == "EXPLOSION_ONLY" else "absolute_quality",
            explosion_rule=(
                None if mode == "VETO_ONLY" else runtime.EXPLOSION_RULES[0]
            ),
        )
        assert model["engineering_amendment"] == 3
        assert model["profile"]["policy_mode"] == mode
        assert set(runtime.HASH_BINDINGS).issubset(model)


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
    with pytest.raises(ValueError, match="non-core"):
        preflight._qualify_veto_matrix(
            base=_base(),
            matrix=np.asarray([[0.1, np.nan]]),
            names=("abs_score", "source_plane"),
            classifier={"imputer_medians": [0.0, 0.0]},
        )


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_raw_infinity_is_rejected(value: float) -> None:
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


def test_semantic_scan_hash_must_equal_the_sealed_record_binding() -> None:
    scan_sha = hashlib.sha256(b"scan").hexdigest()
    preflight._validate_semantic_scan_binding(
        {"scan_sha256": scan_sha}, {"scan_sha256": scan_sha}
    )
    with pytest.raises(ValueError, match="differs from sealed binding"):
        preflight._validate_semantic_scan_binding(
            {"scan_sha256": scan_sha},
            {"scan_sha256": hashlib.sha256(b"other").hexdigest()},
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        preflight._validate_semantic_scan_binding(
            {}, {"scan_sha256": scan_sha}
        )


def test_a2_failure_and_disposition_lineage_is_exact_and_canonical() -> None:
    artifacts = (
        (preflight.A2_P0_PREFLIGHT_SOURCE, preflight.A2_P0_PREFLIGHT_SOURCE_SHA256),
        (preflight.A2_P0_FAILURE, preflight.A2_P0_FAILURE_SHA256),
        (preflight.A2_P1_PREFLIGHT_SOURCE, preflight.A2_P1_PREFLIGHT_SOURCE_SHA256),
        (preflight.A2_P1_FAILURE, preflight.A2_P1_FAILURE_SHA256),
        (preflight.A2_PREFIT_DISPOSITION, preflight.A2_PREFIT_DISPOSITION_SHA256),
    )
    for path, expected in artifacts:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for path in (
        preflight.A2_P0_FAILURE,
        preflight.A2_P1_FAILURE,
        preflight.A2_PREFIT_DISPOSITION,
    ):
        data = path.read_bytes()
        payload = json.loads(data)
        assert preflight.canonical_bytes(payload) == data
        assert payload["holdout_access_count"] == 0
    assert json.loads(preflight.A2_P0_FAILURE.read_bytes())["status"] == "FAIL_CLOSED"
    assert json.loads(preflight.A2_P1_FAILURE.read_bytes())["status"] == "FAIL_CLOSED"
    disposition = json.loads(preflight.A2_PREFIT_DISPOSITION.read_bytes())
    assert disposition["continuation_only_as_distinct_a3"] is True
    assert disposition["a2_fit_started"] is False


def test_a3_preflight_top_level_is_stdlib_only() -> None:
    tree = ast.parse(preflight.PREFLIGHT_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "scripts", "numpy", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_alternate_attestation_is_rejected_before_verified_import(tmp_path) -> None:
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
spec = importlib.util.spec_from_file_location('isolated_a3', path)
module = importlib.util.module_from_spec(spec)
sys.modules['isolated_a3'] = module
spec.loader.exec_module(module)
runtime_path = pathlib.Path({str(runtime_path)!r})
sentinel = pathlib.Path({str(sentinel)!r})
tampered = "from pathlib import Path\\nPath(" + repr(str(sentinel)) + ").write_text('EXECUTED')\\n"
module.RUNTIME_SOURCE = runtime_path
module.A3_RUNTIME_MODULE = 'backend.synthetic_a3_runtime'
original_compile = builtins.compile
def swap(source, filename, mode, *args, **kwargs):
    if filename == str(module.BASE_RUNTIME_SOURCE.resolve()):
        runtime_path.write_text(tampered)
    return original_compile(source, filename, mode, *args, **kwargs)
builtins.compile = swap
_, loaded, _, _ = module._verified_runtime_imports(
    runtime_code_sha256=hashlib.sha256({safe!r}).hexdigest(),
    preflight_code_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
)
assert loaded.VALUE == 'SAFE'
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


def test_real_preflight_and_prereg_outputs_are_absent_before_authorization() -> None:
    assert not preflight.PREFLIGHT_PASS_OUTPUT.exists()
    assert not (ROOT / "research/f1_aug30_train18_candidate2_a3_preregistration_v1.json").exists()
