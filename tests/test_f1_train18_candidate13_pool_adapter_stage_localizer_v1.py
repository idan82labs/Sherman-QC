from __future__ import annotations

import ast
import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import types

import pytest

from backend import f1_line_object_clustering as clusterer
from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as c6
from backend import f1_train18_candidate9_lossless_pool_alias_adapter_v1 as c9
from backend import f1_train18_candidate13_pool_adapter_stage_localizer_v1 as c13
from scripts import build_f1_scan_only_candidate_pool as builder


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py"


def _context() -> c13.LocalizerContext:
    return c13.make_verified_context(
        c9_adapter=c9,
        c6_hypothesis=c6,
        builder=builder,
        clusterer=clusterer,
        bindings=c13.FIXED_BINDINGS,
    )


def _c9_context() -> c9.AdapterContext:
    return c9.make_verified_context(
        hypothesis=c6,
        builder=builder,
        clusterer=clusterer,
        hypothesis_source_sha256=c13.FIXED_BINDINGS["c6_hypothesis"]["sha256"],
        builder_source_sha256=c13.FIXED_BINDINGS["builder"]["sha256"],
        cluster_source_sha256=c13.FIXED_BINDINGS["clusterer"]["sha256"],
    )


def _clone(value):
    return json.loads(c13.canonical_bytes(value).decode("utf-8"))


def _cross_row() -> dict:
    return {
        "bend_id": "SYNTHETIC-BEND-A",
        "candidate_id": "plane_lattice::SYNTHETIC-A",
        "candidate_source": "cross_source_line_object",
        "cluster_member_count": 2,
        "cluster_member_ids": [
            "evidence_trace::SYNTHETIC-B",
            "plane_lattice::SYNTHETIC-A",
        ],
        "cluster_source_count": 2,
        "cluster_sources": ["evidence_trace", "plane_lattice"],
        "endpoint_support_fraction": 0.75,
        "finite_support_fraction": 0.75,
        "normal_second_eigenvalue": 0.05,
        "representative_source": "plane_lattice",
        "score": 0.5,
        "source_candidate_id": "SYNTHETIC-A",
        "source_kind": "evidence_trace",
        "span_endpoints": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        "two_sided_transition_score": 0.75,
    }


def _noncluster_row() -> dict:
    return {
        "bend_id": "SYNTHETIC-BEND-B",
        "candidate_source": "scan_mesh_adjacency",
        "endpoint_support_fraction": 0.5,
        "finite_support_fraction": 0.5,
        "normal_second_eigenvalue": 0.1,
        "representative_source": "scan_mesh_adjacency",
        "score": 0.25,
        "source_kind": "scan_mesh_adjacency",
        "span_endpoints": [[0.0, 20.0, 0.0], [20.0, 20.0, 0.0]],
        "two_sided_transition_score": 0.5,
    }


def _pool(rows: list[dict] | None = None) -> dict:
    return {
        "algorithm": c9.POOL_ALGORITHM,
        "inference_contract": dict(c6.POOL_INFERENCE_CONTRACT),
        "line_objects": _clone(rows if rows is not None else [_cross_row()]),
        "schema_version": c9.POOL_SCHEMA_VERSION,
        "source_point_serialization": False,
        "working_point_budget": c9.WORKING_POINT_BUDGET,
    }


def _encoded(pool: dict, *, pretty: bool = False) -> bytes:
    if pretty:
        return json.dumps(pool, indent=2, sort_keys=False).encode("utf-8")
    return c13.canonical_bytes(pool)


def _localize(
    context: c13.LocalizerContext,
    pool: dict | None = None,
    *,
    raw: bytes | None = None,
    expected_sha256: str | None = None,
) -> dict:
    data = _encoded(_pool() if pool is None else pool) if raw is None else raw
    expected = c13.sha256_bytes(data) if expected_sha256 is None else expected_sha256
    receipts = _issue_receipts(context, data, expected)
    return c13.aggregate_pool_adapter_receipts(context, receipts)


def _issue_receipts(
    context: c13.LocalizerContext,
    data: bytes,
    expected_sha256: str,
    count: int = 36,
) -> tuple[c13._PoolDiagnosticReceipt, ...]:
    return tuple(
        c13.localize_pool_adapter_contract(
            context,
            pool_bytes=data,
            expected_pool_sha256=expected_sha256,
        )
        for _index in range(count)
    )


def _exact_accepts(
    context: c13.LocalizerContext, data: bytes, expected_sha256: str
) -> bool:
    try:
        c9.adapt_pool_bytes(
            _c9_context(),
            pool_bytes=data,
            expected_pool_sha256=expected_sha256,
        )
        return True
    except Exception:
        return False


def _assert_rejection_stage(result: dict, stage: str) -> None:
    index = c13.STAGE_CONTRACT.index(stage)
    assert result["stage_reached_all_bitset"] == (
        "1" * (index + 1)
        + "0" * (len(c13.STAGE_CONTRACT) - index - 2)
        + "1"
    )
    assert result["stage_passed_all_bitset"] == (
        "1" * index
        + "0" * (len(c13.STAGE_CONTRACT) - index - 1)
        + "1"
    )
    assert result["first_failed_candidate13_diagnostic_stage"] == stage
    assert result["exact_c9_success_failure_equivalent_for_all"] is True
    assert result["status"] == "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING"


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


def test_exact_attempt_lineage_and_c12_disposition_binding() -> None:
    assert c13.ATTEMPT_ORDINAL == 13
    assert c13.SCIENTIFIC_HYPOTHESIS_ID == c9.SCIENTIFIC_HYPOTHESIS_ID
    assert c13.FIXED_BINDINGS["c12_disposition"] == {
        "path": "research/f1_aug30_train18_candidate12_synthetic_contract_pass_disposition_v1.json",
        "sha256": "64c51a9421c39a0b3bdb54e8f30701ae58324127ed4dbb828dde7a83add44f75",
        "size_bytes": 3555,
    }
    assert c13.FIXED_BINDINGS["c12_core"]["sha256"] == (
        "89712f14836e102ba8861b2068d464e98ccfdb670d3447201797373b9a2254f6"
    )
    assert c13.FIXED_BINDINGS["c12_tests"]["sha256"] == (
        "a3cf4013384bb9ef0e3077bb00f3ce7028dc1d3e7d0333bb9ef3382ca0e41340"
    )
    assert c13.FIXED_BINDINGS["c11_disposition"]["sha256"] == (
        "58bd4fd18b1f6f4259e051ac1360cf6b204b4c339f1ff6a93e20a0f7d3f05bab"
    )


def test_stage_function_contract_is_exact_one_to_one() -> None:
    assert tuple(stage for stage, _function in c13._STAGE_FUNCTIONS) == c13.STAGE_CONTRACT
    assert len(c13._STAGE_FUNCTIONS) == len(c13.STAGE_CONTRACT)
    assert len(set(c13.STAGE_CONTRACT)) == len(c13.STAGE_CONTRACT)
    assert len(c13._stage_function_snapshot()) == len(c13.STAGE_CONTRACT)


def test_source_contains_no_duplicate_literal_mapping_or_result_keys() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    result_key_literals: list[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RESULT_KEYS"
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Call)
            assert len(node.value.args) == 1
            assert isinstance(node.value.args[0], (ast.Set, ast.Tuple, ast.List))
            result_key_literals = [
                ast.literal_eval(element) for element in node.value.args[0].elts
            ]
        if isinstance(node, ast.Dict):
            literal_keys = [
                ast.literal_eval(key)
                for key in node.keys
                if key is not None
                and isinstance(key, ast.Constant)
                and type(key.value) is str
            ]
            assert len(literal_keys) == len(set(literal_keys))
    assert result_key_literals is not None
    assert len(result_key_literals) == len(set(result_key_literals))
    assert frozenset(result_key_literals) == c13.RESULT_KEYS


def test_valid_cross_source_pool_passes_exact_c9_chain() -> None:
    context = _context()
    result = _localize(context)
    assert set(result) == c13.RESULT_KEYS
    assert result["status"] == "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    assert result["check_code"] == "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL"
    assert result["first_failed_candidate13_diagnostic_stage"] == "NONE"
    assert result["stage_reached_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
    assert result["stage_passed_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
    assert result["exact_c9_success_failure_equivalent_for_all"] is True


def test_valid_noncanonical_json_is_canonicalized_and_accepted() -> None:
    context = _context()
    data = _encoded(_pool([_cross_row(), _noncluster_row()]), pretty=True)
    result = _localize(context, raw=data)
    assert result["stage_passed_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
    receipt = c9.adapt_pool_bytes(
        _c9_context(),
        pool_bytes=data,
        expected_pool_sha256=c13.sha256_bytes(data),
    )
    parsed = c9._decode_pool(receipt.adapted_pool_bytes, name="synthetic result")
    assert receipt.adapted_pool_bytes == c9.canonical_bytes(parsed)


def test_empty_line_objects_is_valid_exact_c9_and_c6_input() -> None:
    result = _localize(_context(), _pool([]))
    assert result["stage_reached_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
    assert result["stage_passed_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
    assert result["exact_c9_success_failure_equivalent_for_all"] is True


def test_row_identity_order_and_projection_are_preserved() -> None:
    context = _context()
    rows = [_cross_row(), _noncluster_row()]
    for ordered in (rows, list(reversed(rows))):
        data = _encoded(_pool(ordered))
        result = _localize(context, raw=data)
        assert result["stage_passed_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
        receipt = c9.adapt_pool_bytes(
            _c9_context(),
            pool_bytes=data,
            expected_pool_sha256=c13.sha256_bytes(data),
        )
        projected = c9._decode_pool(receipt.adapted_pool_bytes, name="synthetic projection")
        assert [row["bend_id"] for row in projected["line_objects"]] == [
            row["bend_id"] for row in ordered
        ]


def test_input_hash_or_json_failure_localizes_without_detail() -> None:
    context = _context()
    valid = _encoded(_pool())
    wrong_hash = _localize(context, raw=valid, expected_sha256="0" * 64)
    _assert_rejection_stage(wrong_hash, c13.STAGE_CONTRACT[0])
    invalid = _localize(context, raw=b'{"unterminated":')
    _assert_rejection_stage(invalid, c13.STAGE_CONTRACT[0])
    duplicate = _localize(
        context,
        raw=b'{"schema_version":3,"schema_version":3}\n',
    )
    _assert_rejection_stage(duplicate, c13.STAGE_CONTRACT[0])


def test_header_failure_is_separate_from_recursive_forbidden_stage() -> None:
    context = _context()
    value = _pool()
    value["schema_version"] = 2
    _assert_rejection_stage(_localize(context, value), c13.STAGE_CONTRACT[1])
    forbidden = _pool()
    forbidden["safe_envelope"] = {"items": [{"ground_truth": False}]}
    _assert_rejection_stage(_localize(context, forbidden), c13.STAGE_CONTRACT[3])


def test_multi_defect_stage_is_explicit_c13_precedence_not_c9_cause() -> None:
    context = _context()
    value = _pool([_cross_row(), _noncluster_row()])
    value["line_objects"][1]["bend_id"] = value["line_objects"][0]["bend_id"]
    value["safe_envelope"] = {"ground_truth": False}
    result = _localize(context, value)
    _assert_rejection_stage(result, c13.STAGE_CONTRACT[2])
    assert result["diagnostic_stage_precedence_is_candidate13_contract"] is True
    assert result["candidate9_internal_failure_stage_claimed"] is False


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "missing", "nonlist", "nonobject"),
)
def test_row_contract_failures_localize(mutation: str) -> None:
    context = _context()
    value = _pool([_cross_row(), _noncluster_row()])
    if mutation == "duplicate":
        value["line_objects"][1]["bend_id"] = value["line_objects"][0]["bend_id"]
    elif mutation == "missing":
        del value["line_objects"][0]["bend_id"]
    elif mutation == "nonlist":
        value["line_objects"] = {}
    else:
        value["line_objects"][0] = []
    _assert_rejection_stage(_localize(context, value), c13.STAGE_CONTRACT[2])


@pytest.mark.parametrize(
    "mutation",
    ("unknown_spelling", "missing_authority", "unknown_authority", "alias_disagreement"),
)
def test_source_authority_failures_localize(mutation: str) -> None:
    context = _context()
    if mutation == "alias_disagreement":
        value = _pool([_noncluster_row()])
        value["line_objects"][0]["source_kind"] = "evidence_trace"
    else:
        value = _pool()
        row = value["line_objects"][0]
        if mutation == "unknown_spelling":
            row["representativeSource"] = row["representative_source"]
        elif mutation == "missing_authority":
            del row["representative_source"]
        else:
            row["representative_source"] = "foreign_source"
    _assert_rejection_stage(_localize(context, value), c13.STAGE_CONTRACT[4])


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("cluster_member_ids", ["BAD", "plane_lattice::SYNTHETIC-A"]),
        ("cluster_member_ids", ["evidence_trace::", "plane_lattice::SYNTHETIC-A"]),
        ("cluster_sources", ["plane_lattice", "evidence_trace"]),
        ("cluster_source_count", 1),
        ("cluster_member_count", True),
        ("candidate_id", "plane_lattice::OTHER"),
    ),
)
def test_cluster_provenance_failures_localize(field: str, bad) -> None:
    context = _context()
    value = _pool()
    value["line_objects"][0][field] = bad
    _assert_rejection_stage(_localize(context, value), c13.STAGE_CONTRACT[5])


@pytest.mark.parametrize(
    "mutation",
    ("partial_conflict", "incomplete_without_span", "degenerate", "out_of_domain"),
)
def test_geometry_alias_failures_localize(mutation: str) -> None:
    context = _context()
    value = _pool()
    row = value["line_objects"][0]
    if mutation == "partial_conflict":
        row["anchor"] = [999.0, 0.0, 0.0]
    elif mutation == "incomplete_without_span":
        del row["span_endpoints"]
        row["anchor"] = [10.0, 0.0, 0.0]
    elif mutation == "degenerate":
        row["span_endpoints"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    else:
        row["span_endpoints"] = [[0.0, 0.0, 0.0], [2_000_001.0, 0.0, 0.0]]
    _assert_rejection_stage(_localize(context, value), c13.STAGE_CONTRACT[6])


def test_complete_auxiliary_without_span_and_partial_matching_aliases_pass() -> None:
    context = _context()
    complete = _pool()
    row = complete["line_objects"][0]
    del row["span_endpoints"]
    row.update(
        {
            "anchor": [10.0, 0.0, 0.0],
            "axis_direction": [1.0, 0.0, 0.0],
            "span_mm": 20.0,
        }
    )
    assert _localize(context, complete)["stage_passed_all_bitset"] == "1" * 10
    partial = _pool()
    partial["line_objects"][0]["axis_direction"] = [-1.0, 0.0, 0.0]
    assert _localize(context, partial)["stage_passed_all_bitset"] == "1" * 10


def test_every_auxiliary_presence_subset_and_equivalence_tolerance() -> None:
    context = _context()
    fields = ("anchor", "axis_direction", "span_mm")
    matching = {
        "anchor": [10.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 20.0,
    }
    for mask in range(8):
        value = _pool()
        for index, field in enumerate(fields):
            if mask & (1 << index):
                value["line_objects"][0][field] = _clone(matching[field])
        assert _localize(context, value)["exact_c9_success_failure_equivalent_for_all"] is True
    inside = _pool()
    inside["line_objects"][0]["anchor"] = [10.0000005, 0.0, 0.0]
    assert _localize(context, inside)["stage_passed_all_bitset"] == "1" * 10
    outside = _pool()
    outside["line_objects"][0]["anchor"] = [10.000002, 0.0, 0.0]
    _assert_rejection_stage(_localize(context, outside), c13.STAGE_CONTRACT[6])


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("score", "0.5"),
        ("finite_support_fraction", 1.1),
        ("endpoint_support_fraction", -0.1),
        ("two_sided_transition_score", 2.0),
        ("normal_second_eigenvalue", -0.1),
    ),
)
def test_unchanged_c6_feature_validation_localizes(field: str, bad) -> None:
    context = _context()
    value = _pool()
    value["line_objects"][0][field] = bad
    _assert_rejection_stage(_localize(context, value), c13.STAGE_CONTRACT[8])


@pytest.mark.parametrize(
    "field",
    (
        "score",
        "finite_support_fraction",
        "endpoint_support_fraction",
        "two_sided_transition_score",
        "normal_second_eigenvalue",
    ),
)
def test_missing_and_explicit_null_features_are_preserved(field: str) -> None:
    context = _context()
    missing = _pool()
    missing["line_objects"][0].pop(field, None)
    null = _pool()
    null["line_objects"][0][field] = None
    for value, expect_present in ((missing, False), (null, True)):
        data = _encoded(value)
        result = _localize(context, raw=data)
        assert result["stage_passed_all_bitset"] == "1" * 10
        receipt = c9.adapt_pool_bytes(
            _c9_context(),
            pool_bytes=data,
            expected_pool_sha256=c13.sha256_bytes(data),
        )
        projected = c9._decode_pool(receipt.adapted_pool_bytes, name="missingness probe")
        row = projected["line_objects"][0]
        assert (field in row) is expect_present
        if expect_present:
            assert row[field] is None


def test_natural_synthetic_corpus_matches_exact_c9_accept_reject() -> None:
    context = _context()
    corpus: list[bytes] = []
    corpus.append(_encoded(_pool()))
    corpus.append(_encoded(_pool([_noncluster_row()])))
    bad_header = _pool()
    bad_header["algorithm"] = "other"
    corpus.append(_encoded(bad_header))
    bad_forbidden = _pool()
    bad_forbidden["line_objects"][0]["nested"] = [{"label": False}]
    corpus.append(_encoded(bad_forbidden))
    bad_source = _pool()
    bad_source["line_objects"][0]["candidate_source"] = "unknown"
    corpus.append(_encoded(bad_source))
    bad_cluster = _pool()
    bad_cluster["line_objects"][0]["cluster_member_ids"] = ["BAD"]
    corpus.append(_encoded(bad_cluster))
    bad_geometry = _pool()
    bad_geometry["line_objects"][0]["axis_direction"] = [0.0, 1.0, 0.0]
    corpus.append(_encoded(bad_geometry))
    bad_feature = _pool()
    bad_feature["line_objects"][0]["score"] = True
    corpus.append(_encoded(bad_feature))
    for data in corpus:
        expected = c13.sha256_bytes(data)
        exact = _exact_accepts(context, data, expected)
        result = _localize(context, raw=data, expected_sha256=expected)
        local = result["stage_passed_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
        assert local is exact
        assert result["exact_c9_success_failure_equivalent_for_all"] is True


def test_exact_type_input_domain_rejects_subclasses_without_false_parity() -> None:
    context = _context()
    data = _encoded(_pool())

    class BytesSubclass(bytes):
        pass

    class StringSubclass(str):
        pass

    for pool_bytes, expected in (
        (BytesSubclass(data), c13.sha256_bytes(data)),
        (data, StringSubclass(c13.sha256_bytes(data))),
    ):
        result = _localize(
            context,
            raw=pool_bytes,
            expected_sha256=expected,
        )
        assert result["stage_passed_all_bitset"][0] == "0"
        assert result["stage_passed_all_bitset"][-1] == "0"
        assert result["exact_c9_success_failure_equivalent_for_all"] is False
        assert result["status"] == "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE"


def test_deep_json_failure_is_contained_and_exact_outcome_equivalent() -> None:
    context = _context()
    data = b'{"neutral":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n"
    result = _localize(context, raw=data)
    _assert_rejection_stage(result, c13.STAGE_CONTRACT[0])


@pytest.mark.parametrize("stage_index", range(len(c13.STAGE_CONTRACT)))
def test_ordinary_fault_at_every_stage_is_contained_and_value_free(
    monkeypatch: pytest.MonkeyPatch, stage_index: int
) -> None:
    original = c13._STAGE_FUNCTIONS

    def injected(_context, _state):
        raise RuntimeError("PRIVATE_SYNTHETIC_FAULT_DETAIL")

    changed = list(original)
    changed[stage_index] = (changed[stage_index][0], injected)
    monkeypatch.setattr(c13, "_STAGE_FUNCTIONS", tuple(changed))
    context = _context()
    result = _localize(context)
    assert result["first_failed_candidate13_diagnostic_stage"] == c13.STAGE_CONTRACT[stage_index]
    if stage_index == len(c13.STAGE_CONTRACT) - 1:
        assert result["stage_reached_all_bitset"] == "1" * len(c13.STAGE_CONTRACT)
        assert result["stage_passed_all_bitset"] == "1" * stage_index + "0"
    else:
        assert result["stage_reached_all_bitset"] == (
            "1" * (stage_index + 1)
            + "0" * (len(c13.STAGE_CONTRACT) - stage_index - 2)
            + "1"
        )
        assert result["stage_passed_all_bitset"] == (
            "1" * stage_index
            + "0" * (len(c13.STAGE_CONTRACT) - stage_index)
        )
    assert result["exact_c9_success_failure_equivalent_for_all"] is False
    assert result["status"] == "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE"
    assert b"PRIVATE_SYNTHETIC_FAULT_DETAIL" not in c13.canonical_bytes(result)


def test_public_result_is_value_free_and_does_not_claim_c11_cause() -> None:
    result = _localize(_context())
    forbidden_keys = ("case", "ordinal", "count", "identifier", "path", "distribution", "exception", "value")
    assert all(not any(token in key.lower() for token in forbidden_keys) for key in result)
    assert all(type(value) in (bool, str) for value in result.values())
    encoded = c13.canonical_bytes(result)
    assert b"SYNTHETIC-BEND" not in encoded
    assert b"line_objects" not in encoded
    assert b"plane_lattice" not in encoded
    assert result["candidate11_actual_record_cause_resolved"] is False
    assert result["candidate9_internal_failure_stage_claimed"] is False
    assert result["diagnostic_stage_precedence_is_candidate13_contract"] is True
    assert result["performance_tested"] is False
    assert result["rule_selected"] is False
    assert result["metric_computed"] is False
    assert result["fit_performed"] is False
    assert result["holdout_opened"] is False


@pytest.mark.parametrize("binding", sorted(c13.FIXED_BINDINGS))
def test_context_rejects_every_fixed_binding_drift(binding: str) -> None:
    bindings = _clone(c13.FIXED_BINDINGS)
    bindings[binding]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding map"):
        c13.make_verified_context(
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=bindings,
        )


def test_context_rejects_fake_dependency_module() -> None:
    with pytest.raises(ValueError, match="exact dependency modules"):
        c13.make_verified_context(
            c9_adapter=types.SimpleNamespace(**vars(c9)),
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=c13.FIXED_BINDINGS,
        )


def test_hostile_outer_or_nested_binding_mapping_cannot_run_callbacks() -> None:
    triggered: list[str] = []

    class HostileMapping(dict):
        def __iter__(self):
            triggered.append("iter")
            return super().__iter__()

        def __getitem__(self, key):
            triggered.append("getitem")
            return super().__getitem__(key)

    with pytest.raises(ValueError, match="binding map"):
        c13.make_verified_context(
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=HostileMapping(_clone(c13.FIXED_BINDINGS)),
        )
    assert triggered == []
    nested = _clone(c13.FIXED_BINDINGS)
    nested["builder"] = HostileMapping(nested["builder"])
    with pytest.raises(ValueError, match="binding schema"):
        c13.make_verified_context(
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=nested,
        )
    assert triggered == []


def test_nonexact_binding_keys_are_rejected_before_key_callbacks() -> None:
    triggered: list[str] = []

    class HostileKey(str):
        def __hash__(self):
            triggered.append("hash")
            return super().__hash__()

        def __eq__(self, other):
            triggered.append("eq")
            return super().__eq__(other)

    bindings = _clone(c13.FIXED_BINDINGS)
    builder_binding = bindings.pop("builder")
    bindings[HostileKey("builder")] = builder_binding
    triggered.clear()
    with pytest.raises(ValueError, match="binding map"):
        c13.make_verified_context(
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=bindings,
        )
    assert triggered == []

    bindings = _clone(c13.FIXED_BINDINGS)
    sha = bindings["builder"].pop("sha256")
    bindings["builder"][HostileKey("sha256")] = sha
    triggered.clear()
    with pytest.raises(ValueError, match="binding schema"):
        c13.make_verified_context(
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=bindings,
        )
    assert triggered == []


def test_shared_json_member_replacement_is_rejected_before_invalid_input_acceptance() -> None:
    context = _context()
    original = c9.json.loads

    def replacement(text, *args, **kwargs):
        if text == "x":
            return _pool([])
        return original(text, *args, **kwargs)

    c9.json.loads = replacement
    try:
        with pytest.raises(PermissionError, match="context is not intact"):
            c13.localize_pool_adapter_contract(
                context,
                pool_bytes=b"x",
                expected_pool_sha256=c13.sha256_bytes(b"x"),
            )
    finally:
        c9.json.loads = original


def test_live_runtime_member_surface_includes_stdlib_and_numeric_calls() -> None:
    snapshot = c13._live_runtime_member_snapshot(c9, c6, builder, clusterer)
    labels = {entry[0] for entry in snapshot}
    assert labels == {
        "local_hashlib",
        "local_json",
        "c9_hashlib",
        "c9_json",
        "c9_re",
        "c6_hashlib",
        "c6_json",
        "c6_math",
        "c6_re",
        "builder_hashlib",
        "builder_json",
        "builder_math",
        "builder_numpy",
        "cluster_math",
        "cluster_numpy",
    }
    projection = {
        label: {name for name, _contract in members}
        for label, _root, members in snapshot
    }
    assert {"loads", "dumps", "JSONDecodeError"} <= projection["c9_json"]
    assert {"sha256"} <= projection["c9_hashlib"]
    assert {"sub"} <= projection["c9_re"]
    assert {"sqrt", "acos", "degrees", "isfinite"} <= projection["c6_math"]
    assert {"linalg.eigh", "linalg.norm", "asarray"} <= projection["builder_numpy"]


def test_post_run_dependency_revalidation_blocks_midrun_member_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_stages = c13._STAGE_FUNCTIONS
    original_header = original_stages[1][1]
    original_loads = c9.json.loads

    def mutating_header(context, state):
        original_header(context, state)

        def replacement(text, *args, **kwargs):
            return original_loads(text, *args, **kwargs)

        c9.json.loads = replacement

    changed = list(original_stages)
    changed[1] = (changed[1][0], mutating_header)
    monkeypatch.setattr(c13, "_STAGE_FUNCTIONS", tuple(changed))
    context = _context()
    data = _encoded(_pool())
    try:
        with pytest.raises(PermissionError, match="context is not intact"):
            c13.localize_pool_adapter_contract(
                context,
                pool_bytes=data,
                expected_pool_sha256=c13.sha256_bytes(data),
            )
    finally:
        c9.json.loads = original_loads


def test_post_issuance_transitive_helper_replacement_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    original = c9._normalize_line_object

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(c9, "_normalize_line_object", replacement)
    with pytest.raises(PermissionError, match="context is not intact"):
        _localize(context)


def test_dependency_mutation_and_coherent_context_snapshot_rebase_rejects() -> None:
    context = _context()
    original = c9._normalize_line_object

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    c9._normalize_line_object = replacement
    try:
        for field in (
            "c9_adapter",
            "c9_context",
            "_dependency_symbol_snapshot",
            "_issuance_token",
        ):
            with pytest.raises(AttributeError):
                object.__setattr__(context, field, object())
        with pytest.raises(PermissionError, match="context is not intact"):
            _localize(context)
    finally:
        c9._normalize_line_object = original


def test_dependency_snapshots_are_recursively_immutable_and_kwdefault_drift_rejects() -> None:
    context = _context()
    defaults = clusterer.cluster_line_candidates.__kwdefaults__
    assert type(defaults) is dict
    original = defaults["axis_tolerance_deg"]
    defaults["axis_tolerance_deg"] = 99.0
    try:
        with pytest.raises(PermissionError, match="context is not intact"):
            _localize(context)
    finally:
        defaults["axis_tolerance_deg"] = original


def test_context_handle_is_fieldless_opaque_and_nonserializable() -> None:
    context = _context()
    assert type(context) is c13.LocalizerContext
    assert c13.LocalizerContext.__slots__ == ()
    assert repr(context) == "<Candidate13OpaqueContext>"
    assert str(context) == "<Candidate13OpaqueContext>"
    assert not hasattr(context, "__dict__")
    for field in (
        "c9_adapter",
        "c9_context",
        "bindings",
        "reached",
        "passed",
        "_sentinel",
        "_issuance_token",
    ):
        assert not hasattr(context, field)
        with pytest.raises(AttributeError):
            object.__setattr__(context, field, object())
    for operation in (
        lambda: vars(context),
        lambda: asdict(context),
        lambda: pickle.dumps(context),
        lambda: copy.copy(context),
        lambda: copy.deepcopy(context),
    ):
        with pytest.raises(TypeError):
            operation()


def test_forged_context_handle_is_rejected() -> None:
    forged = object.__new__(c13.LocalizerContext)
    data = _encoded(_pool())
    with pytest.raises(PermissionError, match="context handle is not authentic"):
        c13.localize_pool_adapter_contract(
            forged,
            pool_bytes=data,
            expected_pool_sha256=c13.sha256_bytes(data),
        )


def test_module_exposes_no_mutable_issuance_registry_or_factory() -> None:
    forbidden = {
        "_ISSUED_CONTEXTS",
        "_ISSUED_RECEIPTS",
        "context_registry",
        "receipt_registry",
        "_build_opaque_capability_runtime",
        "_activate_opaque_runtime",
    }
    assert forbidden.isdisjoint(vars(c13))


def test_post_issuance_receipt_class_replacement_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()

    class FakeReceipt:
        pass

    monkeypatch.setattr(c9, "AdaptedPoolBytes", FakeReceipt)
    with pytest.raises(PermissionError, match="context is not intact"):
        _localize(context)


@pytest.mark.parametrize(
    ("module", "name", "replacement"),
    (
        (c9, "TARGET_CANDIDATE_SOURCE", "changed"),
        (c9, "AUXILIARY_GEOMETRY_FIELDS", ("anchor",)),
        (c6, "MAX_ABS_COORDINATE_MM", 3.0),
        (c6, "ALLOWED_SOURCE_KINDS", frozenset({"plane_lattice"})),
        (builder, "SUPPORT_ATOM_ID_CAP_PER_CANDIDATE", 1),
    ),
)
def test_post_issuance_transitive_constant_drift_rejects(
    monkeypatch: pytest.MonkeyPatch,
    module,
    name: str,
    replacement,
) -> None:
    context = _context()
    monkeypatch.setattr(module, name, replacement)
    with pytest.raises(PermissionError, match="context is not intact"):
        _localize(context)


def test_public_closure_function_replacement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    original = c13.localize_pool_adapter_contract
    data = _encoded(_pool())

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(c13, "localize_pool_adapter_contract", replacement)
    with pytest.raises(PermissionError, match="opaque runtime integrity"):
        original(
            context,
            pool_bytes=data,
            expected_pool_sha256=c13.sha256_bytes(data),
        )


@pytest.mark.parametrize(
    ("function_name", "cell_name"),
    (
        ("make_verified_context", "context_registry"),
        ("localize_pool_adapter_contract", "receipt_registry"),
    ),
)
def test_closure_registry_cell_replacement_is_rejected(
    function_name: str, cell_name: str
) -> None:
    context = _context()
    data = _encoded(_pool())
    receipts = _issue_receipts(context, data, c13.sha256_bytes(data))
    function = getattr(c13, function_name)
    closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    cell = closure[cell_name]
    original = cell.cell_contents
    cell.cell_contents = {}
    try:
        with pytest.raises(PermissionError, match="opaque runtime integrity"):
            c13.aggregate_pool_adapter_receipts(context, receipts)
    finally:
        cell.cell_contents = original


def test_coherent_context_registry_entry_copy_and_reverse_rebase_rejects() -> None:
    context = _context()
    data = _encoded(_pool())
    function = c13.make_verified_context
    closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    registry = closure["context_registry"].cell_contents
    reverse_registry = closure["context_marker_registry"].cell_contents
    entry = registry[context]
    marker = entry[1]
    reverse = reverse_registry[marker]
    forged = object.__new__(c13.LocalizerContext)
    del registry[context]
    registry[forged] = (forged, *entry[1:])
    reverse_registry[marker] = (forged, entry[4])
    try:
        with pytest.raises(PermissionError, match="not authentic"):
            c13.localize_pool_adapter_contract(
                forged,
                pool_bytes=data,
                expected_pool_sha256=c13.sha256_bytes(data),
            )
    finally:
        del registry[forged]
        registry[context] = entry
        reverse_registry[marker] = reverse


def test_coherent_receipt_registry_entry_copy_and_reverse_rebase_rejects() -> None:
    context = _context()
    data = _encoded(_pool())
    receipts = _issue_receipts(context, data, c13.sha256_bytes(data))
    function = c13.localize_pool_adapter_contract
    closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    registry = closure["receipt_registry"].cell_contents
    reverse_registry = closure["receipt_marker_registry"].cell_contents
    original = receipts[0]
    entry = registry[original]
    marker = entry[1]
    reverse = reverse_registry[marker]
    forged = object.__new__(c13._PoolDiagnosticReceipt)
    del registry[original]
    registry[forged] = (forged, *entry[1:])
    reverse_registry[marker] = (forged, entry[5])
    try:
        with pytest.raises(PermissionError, match="not authentic"):
            c13.aggregate_pool_adapter_receipts(
                context, (forged, *receipts[1:])
            )
    finally:
        del registry[forged]
        registry[original] = entry
        reverse_registry[marker] = reverse


def test_local_class_drift_is_rejected_before_private_state_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()

    class FakeState:
        def __init__(self, **_kwargs):
            raise AssertionError("must not construct unauthenticated state")

    monkeypatch.setattr(c13, "_RecordState", FakeState)
    with pytest.raises(PermissionError, match="context is not intact"):
        _localize(context)


def test_public_validator_rejects_schema_bits_status_and_boundary_drift() -> None:
    result = _localize(_context())
    mutations = []
    extra = dict(result)
    extra["detail"] = "forbidden"
    mutations.append(extra)
    wrong_bit = dict(result)
    wrong_bit["stage_passed_all_bitset"] = "0" + "1" * 9
    mutations.append(wrong_bit)
    wrong_stage = dict(result)
    wrong_stage["first_failed_candidate13_diagnostic_stage"] = c13.STAGE_CONTRACT[0]
    mutations.append(wrong_stage)
    wrong_status = dict(result)
    wrong_status["status"] = "OTHER"
    mutations.append(wrong_status)
    wrong_boundary = dict(result)
    wrong_boundary["metric_computed"] = True
    mutations.append(wrong_boundary)
    for mutated in mutations:
        with pytest.raises(ValueError):
            c13._validate_public_result(mutated)


def test_private_pool_receipt_is_nonserializable_and_not_a_public_mapping() -> None:
    context = _context()
    data = _encoded(_pool())
    receipt = c13.localize_pool_adapter_contract(
        context,
        pool_bytes=data,
        expected_pool_sha256=c13.sha256_bytes(data),
    )
    assert type(receipt) is c13._PoolDiagnosticReceipt
    assert c13._PoolDiagnosticReceipt.__slots__ == ()
    assert not isinstance(receipt, dict)
    assert repr(receipt) == "<Candidate13OpaquePoolReceipt>"
    assert str(receipt) == "<Candidate13OpaquePoolReceipt>"
    assert not hasattr(receipt, "__dict__")
    for field in (
        "reached",
        "passed",
        "exact_c9_success_failure_equivalent",
        "_context_token",
        "_sentinel",
        "_issuance_token",
    ):
        assert not hasattr(receipt, field)
        with pytest.raises(AttributeError):
            object.__setattr__(receipt, field, object())
    for operation in (
        lambda: vars(receipt),
        lambda: asdict(receipt),
        lambda: pickle.dumps(receipt),
        lambda: copy.copy(receipt),
        lambda: copy.deepcopy(receipt),
    ):
        with pytest.raises(TypeError):
            operation()


def test_forged_or_replaced_private_receipt_is_rejected() -> None:
    context = _context()
    data = _encoded(_pool())
    authentic = _issue_receipts(context, data, c13.sha256_bytes(data), 35)
    forged = object.__new__(c13._PoolDiagnosticReceipt)
    with pytest.raises(PermissionError, match="not authentic"):
        c13.aggregate_pool_adapter_receipts(context, authentic + (forged,))

    complete = _issue_receipts(context, data, c13.sha256_bytes(data))
    with pytest.raises(PermissionError, match="duplicate"):
        c13.aggregate_pool_adapter_receipts(
            context, complete[:-1] + (complete[0],)
        )


@pytest.mark.parametrize("count", (1, 35, 37))
def test_aggregate_rejects_any_non36_cardinality(count: int) -> None:
    context = _context()
    data = _encoded(_pool())
    receipts = _issue_receipts(context, data, c13.sha256_bytes(data), count)
    with pytest.raises(ValueError, match="exactly 36"):
        c13.aggregate_pool_adapter_receipts(context, receipts)


def test_module_registry_insertion_cannot_authorize_forged_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authentic_context = _context()
    data = _encoded(_pool())
    authentic = _issue_receipts(
        authentic_context, data, c13.sha256_bytes(data), 35
    )
    forged_context = object.__new__(c13.LocalizerContext)
    forged_receipt = object.__new__(c13._PoolDiagnosticReceipt)
    monkeypatch.setattr(c13, "_ISSUED_CONTEXTS", {forged_context: object()}, raising=False)
    monkeypatch.setattr(c13, "_ISSUED_RECEIPTS", {forged_receipt: object()}, raising=False)
    with pytest.raises(PermissionError, match="context handle is not authentic"):
        c13.localize_pool_adapter_contract(
            forged_context,
            pool_bytes=data,
            expected_pool_sha256=c13.sha256_bytes(data),
        )
    with pytest.raises(PermissionError, match="not authentic"):
        c13.aggregate_pool_adapter_receipts(
            authentic_context, authentic + (forged_receipt,)
        )


def test_receipts_are_bound_to_the_issuing_context() -> None:
    first_context = _context()
    second_context = _context()
    data = _encoded(_pool())
    receipts = _issue_receipts(
        first_context, data, c13.sha256_bytes(data)
    )
    with pytest.raises(PermissionError, match="not authentic"):
        c13.aggregate_pool_adapter_receipts(second_context, receipts)


def test_aggregate_only_result_is_stagewise_and_of_private_receipts() -> None:
    context = _context()
    valid = _encoded(_pool())
    invalid_pool = _pool()
    invalid_pool["schema_version"] = 2
    invalid = _encoded(invalid_pool)
    receipts = tuple(
        c13.localize_pool_adapter_contract(
            context,
            pool_bytes=data,
            expected_pool_sha256=c13.sha256_bytes(data),
        )
        for data in ((valid,) * 35 + (invalid,))
    )
    result = c13.aggregate_pool_adapter_receipts(context, receipts)
    _assert_rejection_stage(result, c13.STAGE_CONTRACT[1])
    assert result["candidate9_internal_failure_stage_claimed"] is False


def test_result_is_deterministic() -> None:
    context = _context()
    first = _localize(context)
    second = _localize(context)
    assert first == second
    assert c13.canonical_bytes(first) == c13.canonical_bytes(second)


def test_contract_declaration_is_non_authorizing_and_cause_neutral() -> None:
    declaration = c13.contract_declaration()
    assert declaration["attempt_ordinal"] == 13
    assert declaration["filesystem_io_performed_by_core"] is False
    assert declaration["input_custody_authenticated_by_core"] is False
    assert declaration["outer_verified_dependency_and_input_bytes_required"] is True
    assert declaration["outer_controlled_import_required"] is True
    assert declaration["resource_bounds_enforced_by_core"] is False
    assert declaration["outer_process_resource_enforcement_required"] is True
    assert declaration["per_pool_receipt_publication_authorized"] is False
    assert declaration["public_result_is_aggregate_only"] is True
    assert declaration["candidate11_actual_record_cause_resolved"] is False
    assert declaration["performance_tested"] is False
    assert declaration["rule_selected"] is False
    assert declaration["stage_contract"] == list(c13.STAGE_CONTRACT)
