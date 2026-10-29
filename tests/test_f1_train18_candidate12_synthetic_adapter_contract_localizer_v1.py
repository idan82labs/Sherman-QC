from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
import types

import pytest

from backend import f1_line_object_clustering as clusterer
from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as c6
from backend import f1_train18_candidate9_lossless_pool_alias_adapter_v1 as c9
from backend import f1_train18_candidate12_synthetic_adapter_contract_localizer_v1 as c12
from scripts import build_f1_scan_only_candidate_pool as builder


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py"


@pytest.fixture(scope="module")
def c9_context() -> c9.AdapterContext:
    return c9.make_verified_context(
        hypothesis=c6,
        builder=builder,
        clusterer=clusterer,
        hypothesis_source_sha256=c12.FIXED_BINDINGS["c6_hypothesis"]["sha256"],
        builder_source_sha256=c12.FIXED_BINDINGS["builder"]["sha256"],
        cluster_source_sha256=c12.FIXED_BINDINGS["clusterer"]["sha256"],
    )


def _context(c9_context: c9.AdapterContext) -> c12.LocalizerContext:
    return c12.make_verified_context(
        c9_adapter=c9,
        c9_context=c9_context,
        c6_hypothesis=c6,
        bindings=c12.FIXED_BINDINGS,
    )


def test_module_is_filesystem_blind_and_stdlib_only() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    assert imports <= {"__future__", "dataclasses", "hashlib", "json", "types", "typing"}
    assert not ({"os", "pathlib", "shutil", "subprocess"} & imports)
    assert not ({"open", "exec", "eval", "compile"} & calls)


def test_exact_lineage_and_attempt_contract() -> None:
    assert c12.ATTEMPT_ORDINAL == 12
    assert c12.SCIENTIFIC_HYPOTHESIS_ID == c9.SCIENTIFIC_HYPOTHESIS_ID
    assert c12.FIXED_BINDINGS["c11_disposition"] == {
        "path": "research/f1_aug30_train18_candidate11_terminal_disposition_v1.json",
        "sha256": "58bd4fd18b1f6f4259e051ac1360cf6b204b4c339f1ff6a93e20a0f7d3f05bab",
        "size_bytes": 4150,
    }
    assert c12.FIXED_BINDINGS["c9_adapter"]["sha256"] == (
        "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096"
    )
    assert c12.FIXED_BINDINGS["c6_hypothesis"]["sha256"] == (
        "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252"
    )
    assert c12.FIXED_BINDINGS["c11_core"]["sha256"] == (
        "42c5ded051dd3b57dd48e3039a433afe63a0467edc13d4bd3b65620a6a279359"
    )
    assert c12.FIXED_BINDINGS["c11_runner"]["sha256"] == (
        "d18c583e176778974a9c85ede90ab287f23f5606e1856634c9a6c783c5effdd2"
    )


def test_stage_function_contract_is_exact_one_to_one() -> None:
    assert tuple(stage for stage, _function in c12._STAGE_FUNCTIONS) == c12.STAGE_CONTRACT[:-1]
    assert len(c12._STAGE_FUNCTIONS) + 1 == len(c12.STAGE_CONTRACT)
    assert len(set(c12.STAGE_CONTRACT)) == len(c12.STAGE_CONTRACT)
    assert len(c12._stage_function_snapshot()) == len(c12.STAGE_CONTRACT) - 1


def test_authentic_fixed_matrix_passes_aggregate_only(c9_context: c9.AdapterContext) -> None:
    result = c12.run_synthetic_contract_matrix(_context(c9_context))
    assert set(result) == c12.RESULT_KEYS
    assert result["status"] == "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    assert result["check_code"] == "C12_SYNTHETIC_ADAPTER_CONTRACT_MATRIX_PASSED"
    assert result["first_failed_stage"] == "NONE"
    assert result["stage_reached_all_bitset"] == "1" * len(c12.STAGE_CONTRACT)
    assert result["stage_passed_all_bitset"] == "1" * len(c12.STAGE_CONTRACT)
    assert result["real_input_opened"] is False
    assert result["filesystem_capability_present"] is False
    assert result["rule_selected"] is False
    assert result["metric_computed"] is False
    assert result["fit_performed"] is False
    assert result["holdout_opened"] is False
    assert result["performance_tested"] is False


def test_public_result_has_no_fixture_or_data_bearing_fields(c9_context: c9.AdapterContext) -> None:
    result = c12.run_synthetic_contract_matrix(_context(c9_context))
    forbidden = ("fixture", "index", "count", "source", "value", "identifier", "case", "exception")
    assert all(not any(token in key.lower() for token in forbidden) for key in result)
    assert all(type(value) in (bool, str) for value in result.values())
    encoded = c12.canonical_bytes(result)
    assert b"NEUTRAL" not in encoded
    assert b"line_objects" not in encoded
    assert b"private" not in encoded.lower()


@pytest.mark.parametrize("name", sorted(c12.FIXED_BINDINGS))
def test_context_rejects_every_binding_drift(
    name: str, c9_context: c9.AdapterContext
) -> None:
    bindings = c12._clone(c12.FIXED_BINDINGS)
    bindings[name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding map"):
        c12.make_verified_context(
            c9_adapter=c9,
            c9_context=c9_context,
            c6_hypothesis=c6,
            bindings=bindings,
        )


def test_context_rejects_fake_modules_and_context(c9_context: c9.AdapterContext) -> None:
    with pytest.raises(ValueError, match="exact dependency modules"):
        c12.make_verified_context(
            c9_adapter=SimpleNamespace(),
            c9_context=c9_context,
            c6_hypothesis=c6,
            bindings=c12.FIXED_BINDINGS,
        )
    with pytest.raises(ValueError, match="dependency contract"):
        c12.make_verified_context(
            c9_adapter=c9,
            c9_context=SimpleNamespace(),
            c6_hypothesis=c6,
            bindings=c12.FIXED_BINDINGS,
        )


def test_post_issuance_dependency_function_mutation_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    monkeypatch.setattr(c9, "adapt_pool_bytes", lambda *_args, **_kwargs: None)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_adapted_receipt_class_replacement_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)

    @dataclass(frozen=True)
    class FakeReceipt:
        original_pool_sha256: str
        adapted_pool_sha256: str
        adapted_pool_bytes: bytes
        projection_byte_equivalent: bool

    monkeypatch.setattr(c9, "AdaptedPoolBytes", FakeReceipt)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_adapter_context_class_replacement_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)

    @dataclass(frozen=True)
    class FakeContext:
        hypothesis: object
        builder: object
        clusterer: object
        allowed_source_kinds: frozenset[str]
        _sentinel: object

    monkeypatch.setattr(c9, "AdapterContext", FakeContext)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (c9, "_normalize_line_object"),
        (c6, "_optional_feature"),
        (builder, "_bounded_identifiers"),
        (clusterer, "_canonical"),
    ],
)
def test_post_issuance_transitive_helper_replacement_rejects(
    module: types.ModuleType,
    name: str,
    c9_context: c9.AdapterContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(c9_context)
    original = getattr(module, name)

    def replacement(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, replacement)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


@pytest.mark.parametrize(
    ("module", "name", "replacement"),
    [
        (c9, "TARGET_CANDIDATE_SOURCE", "changed_source"),
        (c9, "SOURCE_FIELDS", frozenset({"candidate_source"})),
        (c6, "MAX_ABS_COORDINATE_MM", 99.0),
        (c6, "FORBIDDEN_KEY_TOKENS", frozenset()),
        (builder, "SUPPORT_ATOM_ID_CAP_PER_CANDIDATE", 1),
    ],
)
def test_post_issuance_dependency_constant_drift_rejects(
    module: types.ModuleType,
    name: str,
    replacement: object,
    c9_context: c9.AdapterContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(c9_context)
    monkeypatch.setattr(module, name, replacement)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_function_default_drift_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    defaults = clusterer.cluster_line_candidates.__kwdefaults__
    assert isinstance(defaults, dict)
    monkeypatch.setitem(defaults, "axis_tolerance_deg", 99.0)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_dependency_module_reference_drift_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    monkeypatch.setattr(clusterer, "math", SimpleNamespace())
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_dependency_sentinel_and_pattern_drift_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    monkeypatch.setattr(c9, "_CONTEXT_SENTINEL", object())
    monkeypatch.setattr(c6, "_HEX64", c6.re.compile(r".+"))
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_local_context_sentinel_cannot_be_rebased_post_issuance(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    changed = object()
    monkeypatch.setattr(c12, "_CONTEXT_SENTINEL", changed)
    object.__setattr__(context, "_sentinel", changed)
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_local_function_default_drift_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    defaults = c12._adapt.__kwdefaults__
    assert isinstance(defaults, dict)
    monkeypatch.setitem(defaults, "raw", b'{"changed":true}\n')
    with pytest.raises(PermissionError, match="not intact"):
        c12.run_synthetic_contract_matrix(context)


def test_post_issuance_stage_function_mutation_rejects(
    c9_context: c9.AdapterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(c9_context)
    changed = list(c12._STAGE_FUNCTIONS)
    changed[0], changed[1] = changed[1], changed[0]
    monkeypatch.setattr(c12, "_STAGE_FUNCTIONS", tuple(changed))
    with pytest.raises((PermissionError, ValueError)):
        c12.run_synthetic_contract_matrix(context)


@pytest.mark.parametrize("stage_index", range(9))
def test_each_closed_stage_localizes_without_exception_detail(
    stage_index: int,
    c9_context: c9.AdapterContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_context_value: c12.LocalizerContext) -> None:
        raise RuntimeError("PRIVATE_FIXTURE_DETAIL_MUST_NOT_ESCAPE")

    changed = list(c12._STAGE_FUNCTIONS)
    stage = changed[stage_index][0]
    changed[stage_index] = (stage, fail)
    monkeypatch.setattr(c12, "_STAGE_FUNCTIONS", tuple(changed))
    result = c12.run_synthetic_contract_matrix(_context(c9_context))
    expected = ["1"] * len(c12.STAGE_CONTRACT)
    expected[stage_index] = "0"
    expected[-1] = "0"
    assert result["stage_reached_all_bitset"] == "1" * len(c12.STAGE_CONTRACT)
    assert result["stage_passed_all_bitset"] == "".join(expected)
    assert result["first_failed_stage"] == stage
    assert result["status"] == "FAILED_ENGINEERING_CONTRACT_LOCALIZATION"
    assert "PRIVATE_FIXTURE_DETAIL" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(extra=True),
        lambda value: value.__setitem__("stage_reached_all_bitset", "1" * 9),
        lambda value: value.__setitem__("stage_passed_all_bitset", "2" * 10),
        lambda value: value.__setitem__("first_failed_stage", "UNKNOWN"),
        lambda value: value.__setitem__(
            "first_failed_stage", c12.STAGE_CONTRACT[1]
        ),
        lambda value: value.__setitem__("real_input_opened", True),
        lambda value: value.__setitem__("performance_tested", True),
        lambda value: value.__setitem__("rule_selected", True),
        lambda value: value.__setitem__("metric_computed", True),
        lambda value: value.__setitem__("fit_performed", True),
        lambda value: value.__setitem__("holdout_opened", True),
    ],
)
def test_public_validator_rejects_schema_and_boundary_drift(
    mutator: object, c9_context: c9.AdapterContext
) -> None:
    value = c12.run_synthetic_contract_matrix(_context(c9_context))
    mutator(value)
    with pytest.raises(ValueError):
        c12._validate_public_result(value)


def test_private_receipt_vector_is_nonserializable() -> None:
    vector = c12._ReceiptVector(
        (c12._StageReceipt(c12.STAGE_CONTRACT[0], True, True),),
        c12._RECEIPT_SENTINEL,
    )
    with pytest.raises(TypeError, match="not serializable"):
        vector.__reduce__()
    with pytest.raises(TypeError):
        json.dumps(vector)


def test_public_validator_rejects_aggregate_bit_inconsistent_with_prior_stage(
    c9_context: c9.AdapterContext,
) -> None:
    value = c12.run_synthetic_contract_matrix(_context(c9_context))
    value["stage_passed_all_bitset"] = "0" + "1" * 9
    value["first_failed_stage"] = c12.STAGE_CONTRACT[0]
    value["check_code"] = "C12_SYNTHETIC_ADAPTER_CONTRACT_MATRIX_FAILED"
    value["status"] = "FAILED_ENGINEERING_CONTRACT_LOCALIZATION"
    with pytest.raises(ValueError):
        c12._validate_public_result(value)


def test_actual_builder_cluster_projection_is_exercised(
    c9_context: c9.AdapterContext
) -> None:
    c12._stage_cluster(_context(c9_context))


def test_all_auxiliary_presence_subsets_and_geometry_domains_are_exercised(
    c9_context: c9.AdapterContext
) -> None:
    c12._stage_auxiliary(_context(c9_context))


def test_missing_and_null_features_remain_distinct(c9_context: c9.AdapterContext) -> None:
    c12._stage_features(_context(c9_context))


def test_final_c6_and_receipt_equivalence_are_exact(c9_context: c9.AdapterContext) -> None:
    context = _context(c9_context)
    c12._stage_c6_validation(context)
    c12._stage_receipt(context)


def test_result_is_deterministic(c9_context: c9.AdapterContext) -> None:
    left = c12.canonical_bytes(c12.run_synthetic_contract_matrix(_context(c9_context)))
    right = c12.canonical_bytes(c12.run_synthetic_contract_matrix(_context(c9_context)))
    assert left == right


def test_contract_declaration_is_source_only_and_non_authorizing() -> None:
    value = c12.contract_declaration()
    assert value["filesystem_capability_present"] is False
    assert value["rule_selected"] is False
    assert value["metric_computed"] is False
    assert value["fit_performed"] is False
    assert value["holdout_opened"] is False
    assert value["outer_verified_dependency_bytes_required"] is True
    assert value["performance_tested"] is False
    assert value["bindings"] == c12.FIXED_BINDINGS
    assert value["stage_contract"] == list(c12.STAGE_CONTRACT)
