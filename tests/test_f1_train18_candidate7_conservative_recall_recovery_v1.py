from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as hypothesis
from backend import f1_train18_candidate7_conservative_recall_recovery_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prediction(y: float = 0.0, *, token: str = "P0") -> dict[str, object]:
    return {
        "bend_id": f"B-{token}",
        "line_end": [100.0, y, 0.0],
        "line_start": [0.0, y, 0.0],
        "selection_route": "v51-test",
        "selector_score": None,
        "source_hypothesis_id": f"H-{token}",
        "source_kind": "plane_lattice",
    }


def _semantic(predictions: list[dict[str, object]] | None = None) -> bytes:
    return hypothesis.canonical_bytes(
        {
            "cad_used": False,
            "candidate_count": len(predictions or [_prediction()]),
            "expected_count_used": False,
            "labels_used": False,
            "part_identity_used": False,
            "predictions": predictions or [_prediction()],
            "research_only": True,
            "source_only": True,
        }
    )


def _pool(y: float = 8.0) -> bytes:
    return hypothesis.canonical_bytes(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
            "candidate_count": 1,
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
                    "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]],
                    "two_sided_transition_score": 0.8,
                }
            ],
            "schema_version": 3,
            "source_point_serialization": False,
            "working_point_budget": 200000,
        }
    )


def _rule() -> dict[str, object]:
    return {"feature": "score", "operator": "ge", "threshold": 0.5}


def _bindings(runtime_sha: str) -> dict[str, str]:
    values = {key: _sha(key) for key in runtime.BINDING_KEYS}
    values.update(runtime.KNOWN_BINDINGS)
    values["candidate7_runtime_source_sha256"] = runtime_sha
    return values


def _model_and_pass(
    rule: dict[str, object], runtime_sha: str
) -> tuple[dict[str, object], bytes, str]:
    bindings = _bindings(runtime_sha)
    pass_value = {
        "attempt_ordinal": 7,
        "bindings": bindings,
        "checks": {key: True for key in runtime.GATE_CHECK_KEYS},
        "engineering_attempt_id": runtime.ENGINEERING_ATTEMPT_ID,
        "gate_result_value_free": True,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "kind": runtime.PASS_KIND,
        "label_derived_metric_values_present": False,
        "pre_holdout_execution_anchor_created": False,
        "schema_version": 1,
        "scientific_hypothesis_id": runtime.SCIENTIFIC_HYPOTHESIS_ID,
        "selected_rule": rule,
        "status": "PASS",
        "training_gate_passed": True,
    }
    pass_bytes = runtime.canonical_bytes(pass_value)
    pass_sha = runtime.sha256_bytes(pass_bytes)
    model = {
        "additive_only": True,
        "attempt_ordinal": 7,
        "bindings": bindings,
        "deterministic_tie_break_direction": (
            hypothesis.DETERMINISTIC_TIE_BREAK_DIRECTION
        ),
        "deterministic_tie_break_inputs": list(
            hypothesis.DETERMINISTIC_TIE_BREAK_INPUTS
        ),
        "engineering_attempt_id": runtime.ENGINEERING_ATTEMPT_ID,
        "external_launcher_authentication_required": True,
        "fixed_eligibility_inputs": list(hypothesis.FIXED_ELIGIBILITY_INPUTS),
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "holdout_score_opened": False,
        "inference_features": [rule["feature"], "source_kind"],
        "kind": runtime.MODEL_KIND,
        "line_objects_only": True,
        "max_added_false_positives_train18": 2,
        "max_additions_per_case": 1,
        "missing_evidence_abstains": True,
        "ownership_contract": hypothesis._ownership_contract(),
        "post_holdout_model_or_threshold_mutation": (
            "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        ),
        "pre_holdout_execution_anchor_created": False,
        "preserve_v51_prediction_values_and_order": True,
        "profile": runtime.PROFILE,
        "raw_candidates_ignored_for_selection": True,
        "rule": rule,
        "schema_version": 1,
        "scientific_hypothesis_id": runtime.SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_source_sha256": (
            runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
        ),
        "source_only": True,
        "train_gate_pass_binding": {"kind": runtime.PASS_KIND, "sha256": pass_sha},
        "training_gate_passed": True,
        **{field: False for field in runtime.FALSE_INFERENCE_DECLARATIONS},
        **{
            declaration: rule["feature"] == feature or feature == "source_kind"
            for feature, declaration in runtime.FEATURE_DECLARATION_KEYS.items()
        },
    }
    return model, pass_bytes, pass_sha


def test_runtime_top_level_is_stdlib_only_and_filesystem_blind() -> None:
    path = Path(runtime.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_imports = ("backend", "numpy", "pathlib", "scripts", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith(forbidden_imports) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden_imports)
    forbidden_calls = {"open", "Path", "read_bytes", "read_text", "write_bytes"}
    assert not {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }.intersection(forbidden_calls)


def test_engineering_attempt_and_scientific_hypothesis_are_separate() -> None:
    assert runtime.ATTEMPT_ORDINAL == 7
    assert runtime.ENGINEERING_ATTEMPT_ID == "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
    assert runtime.SCIENTIFIC_HYPOTHESIS_ID == (
        "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
    )
    assert "CANDIDATE6" not in runtime.SCIENTIFIC_HYPOTHESIS_ID


def test_exact_frozen_hypothesis_grid_and_ownership_are_verified() -> None:
    runtime.verify_scientific_hypothesis_runtime(
        hypothesis, source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
    )
    rules = runtime.fixed_rule_space(
        hypothesis, source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
    )
    assert len(rules) == 32
    assert runtime.sha256_bytes(runtime.canonical_bytes(list(rules))) == (
        runtime.SCIENTIFIC_RULE_SPACE_SHA256
    )
    with pytest.raises(ValueError, match="source binding"):
        runtime.verify_scientific_hypothesis_runtime(
            hypothesis, source_sha256=_sha("wrong")
        )
    changed = SimpleNamespace(**hypothesis.__dict__)
    changed.MAX_ADDITIONS_PER_CASE = 2
    with pytest.raises(ValueError, match="contract mismatch"):
        runtime.verify_scientific_hypothesis_runtime(
            changed, source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
        )


def test_source_preflight_preserves_hypothesis_and_is_value_free() -> None:
    semantic = _semantic()
    pool = _pool()
    value = runtime.preflight_candidate7_source_bytes(
        semantic,
        pool,
        v51_semantic_sha256=runtime.sha256_bytes(semantic),
        pool_sha256=runtime.sha256_bytes(pool),
        hypothesis_runtime=hypothesis,
        hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
    )
    assert value["attempt_ordinal"] == 7
    assert value["engineering_attempt_id"] == runtime.ENGINEERING_ATTEMPT_ID
    assert value["scientific_hypothesis_exactly_preserved"] is True
    assert value["scientific_hypothesis_id"] == runtime.SCIENTIFIC_HYPOTHESIS_ID
    assert value["candidate_metric_computed"] is False
    assert value["labels_used"] is False
    assert value["holdout_score_opened"] is False


@pytest.mark.parametrize(
    "field,value",
    (
        ("bend_id", "C7R-reserved"),
        ("source_hypothesis_id", "C7G-reserved"),
    ),
)
def test_reserved_candidate7_id_namespace_is_fail_closed(
    field: str, value: str
) -> None:
    prediction = _prediction()
    prediction[field] = value
    semantic = _semantic([prediction])
    with pytest.raises(ValueError, match="reserved Candidate7"):
        runtime.preflight_candidate7_source_bytes(
            semantic,
            _pool(),
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            pool_sha256=runtime.sha256_bytes(_pool()),
            hypothesis_runtime=hypothesis,
            hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
        )


def test_apply_is_additive_and_payload_hash_has_defined_projection() -> None:
    semantic = _semantic()
    pool = _pool()
    runtime_sha = _sha("candidate7-runtime")
    model, pass_bytes, pass_sha = _model_and_pass(_rule(), runtime_sha)
    disposition = (ROOT / runtime.CANDIDATE6_DISPOSITION_PATH).read_bytes()
    result = runtime.apply_candidate7(
        semantic,
        pool,
        model,
        pass_bytes,
        disposition,
        v51_semantic_sha256=runtime.sha256_bytes(semantic),
        pool_sha256=runtime.sha256_bytes(pool),
        train_gate_pass_sha256=pass_sha,
        candidate6_disposition_sha256=runtime.CANDIDATE6_DISPOSITION_SHA256,
        verified_runtime_source_sha256=runtime_sha,
        hypothesis_runtime=hypothesis,
        hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
    )
    assert result["addition_made"] is True
    assert result["predictions"][0] == json.loads(semantic)["predictions"][0]
    assert result["predictions"][1]["bend_id"].startswith("C7R-")
    expected_hash = result.pop("candidate7_result_payload_sha256")
    assert runtime.sha256_bytes(runtime.canonical_bytes(result)) == expected_hash


def test_model_requires_exact_hypothesis_and_external_authentication() -> None:
    runtime_sha = _sha("candidate7-runtime")
    model, pass_bytes, pass_sha = _model_and_pass(_rule(), runtime_sha)
    runtime._validate_model(
        model,
        pass_bytes=pass_bytes,
        pass_sha256=pass_sha,
        verified_runtime_source_sha256=runtime_sha,
        hypothesis_runtime=hypothesis,
        hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
    )
    for key, value in (
        ("external_launcher_authentication_required", False),
        ("scientific_hypothesis_id", "CHANGED"),
        ("max_additions_per_case", 2),
    ):
        changed = deepcopy(model)
        changed[key] = value
        with pytest.raises(ValueError, match="status/source contract"):
            runtime._validate_model(
                changed,
                pass_bytes=pass_bytes,
                pass_sha256=pass_sha,
                verified_runtime_source_sha256=runtime_sha,
                hypothesis_runtime=hypothesis,
                hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
            )


def test_candidate6_disposition_is_exact_and_scientifically_unconsumed() -> None:
    data = (ROOT / runtime.CANDIDATE6_DISPOSITION_PATH).read_bytes()
    runtime._validate_candidate6_disposition(
        data, expected_sha256=runtime.CANDIDATE6_DISPOSITION_SHA256
    )
    value = json.loads(data)
    assert value["scientific_hypothesis_consumed"] is False
    assert value["cache_manifest_opened"] is False
    assert value["semantic_artifact_opened"] is False
    assert value["pool_artifact_opened"] is False
