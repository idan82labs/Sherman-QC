from __future__ import annotations

import ast
import copy
from itertools import product
import json
import math
from pathlib import Path
import pickle
import types

import pytest

from backend import f1_line_object_clustering as clusterer
from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as c6
from backend import f1_train18_candidate9_lossless_pool_alias_adapter_v1 as c9
from backend import f1_train18_candidate13_pool_adapter_stage_localizer_v1 as c13
from backend import f1_train18_candidate15_geometry_representation_failure_localizer_v1 as c15
from scripts import build_f1_scan_only_candidate_pool as builder


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/f1_train18_candidate15_geometry_representation_failure_localizer_v1.py"


def _context() -> c15.LocalizerContext:
    return c15.make_verified_context(
        c13_localizer=c13,
        c9_adapter=c9,
        c6_hypothesis=c6,
        builder=builder,
        clusterer=clusterer,
        bindings=c15.FIXED_BINDINGS,
    )


def _c13_context() -> c13.LocalizerContext:
    return c13.make_verified_context(
        c9_adapter=c9,
        c6_hypothesis=c6,
        builder=builder,
        clusterer=clusterer,
        bindings=c13.FIXED_BINDINGS,
    )


def _clone(value):
    return json.loads(c15.canonical_bytes(value).decode("utf-8"))


def _row() -> dict:
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


def _pool(row: dict | None = None) -> dict:
    return {
        "algorithm": c9.POOL_ALGORITHM,
        "inference_contract": dict(c6.POOL_INFERENCE_CONTRACT),
        "line_objects": [_clone(_row() if row is None else row)],
        "schema_version": c9.POOL_SCHEMA_VERSION,
        "source_point_serialization": False,
        "working_point_budget": c9.WORKING_POINT_BUDGET,
    }


def _fixed_c14_geometry_failure_pool() -> dict:
    row = _row()
    row["anchor"] = [999.0, 0.0, 0.0]
    return _pool(row)


def _issue(
    context: c15.LocalizerContext,
    pool: dict | None = None,
    *,
    data: bytes | None = None,
    expected_sha256: str | None = None,
    count: int = 36,
) -> tuple[c15._GeometryDiagnosticReceipt, ...]:
    payload = c15.canonical_bytes(_pool() if pool is None else pool) if data is None else data
    expected = c15.sha256_bytes(payload) if expected_sha256 is None else expected_sha256
    return tuple(
        c15.localize_geometry_representation_contract(
            context,
            pool_bytes=payload,
            expected_pool_sha256=expected,
        )
        for _index in range(count)
    )


def _run(pool: dict | None = None) -> dict:
    context = _context()
    return c15.aggregate_geometry_representation_receipts(
        context, _cohort_receipts(context, pool)
    )


def _cohort_receipts(
    context: c15.LocalizerContext, pool: dict | None = None
) -> tuple[c15._GeometryDiagnosticReceipt, ...]:
    return (
        _issue(context, _pool() if pool is None else pool, count=1)
        + _issue(context, _pool(), count=34)
        + _issue(context, _fixed_c14_geometry_failure_pool(), count=1)
    )


def _cohort_receipts_with_first_data(
    context: c15.LocalizerContext, data: bytes
) -> tuple[c15._GeometryDiagnosticReceipt, ...]:
    return (
        _issue(context, data=data, count=1)
        + _issue(context, _pool(), count=34)
        + _issue(context, _fixed_c14_geometry_failure_pool(), count=1)
    )


def _c13_public_cohort_result(first_data: bytes) -> dict:
    context = _c13_context()
    payloads = (
        (first_data,)
        + (c15.canonical_bytes(_pool()),) * 34
        + (c15.canonical_bytes(_fixed_c14_geometry_failure_pool()),)
    )
    receipts = tuple(
        c13.localize_pool_adapter_contract(
            context,
            pool_bytes=payload,
            expected_pool_sha256=c15.sha256_bytes(payload),
        )
        for payload in payloads
    )
    return c13.aggregate_pool_adapter_receipts(context, receipts)


def _prefix_failure_pool(stage_index: int) -> dict:
    pool = _pool()
    if stage_index == 1:
        pool["algorithm"] = "SYNTHETIC-WRONG-ALGORITHM"
    elif stage_index == 2:
        pool["line_objects"].append(_clone(pool["line_objects"][0]))
    elif stage_index == 3:
        pool["line_objects"][0]["manual_truth"] = "synthetic"
    elif stage_index == 4:
        pool["line_objects"][0]["representativeSource"] = "plane_lattice"
    elif stage_index == 5:
        pool["line_objects"][0]["cluster_member_count"] = 3
    else:
        raise AssertionError("unsupported synthetic C13 prefix stage")
    return pool


def _mask(row: dict) -> int:
    return c15._geometry_violation_mask([copy.deepcopy(row)], c6)


def _mask_has(row: dict, name: str) -> bool:
    return bool(_mask(row) & (1 << c15.VIOLATION_CONTRACT.index(name)))


def _closure_values(function) -> dict[str, object]:
    return {
        name: cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ())
    }


def _bit(result: dict, name: str) -> str:
    return result["violation_or_bitset"][c15.VIOLATION_CONTRACT.index(name)]


def test_source_is_filesystem_blind_and_has_no_data_namespace_literals() -> None:
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
    assert imports <= {
        "__future__", "dataclasses", "hashlib", "json", "math", "types", "typing"
    }
    assert not ({"open", "exec", "eval", "compile"} & calls)
    text = SOURCE.read_text(encoding="utf-8").lower()
    assert "eho-" not in text
    assert "gmail" not in text


def test_fixed_lineage_and_vocabularies_are_exact() -> None:
    assert c15.ATTEMPT_ORDINAL == 15
    assert c15.FIXED_BINDINGS["c14_terminal_disposition"] == {
        "path": "research/f1_aug30_train18_candidate14_terminal_disposition_v1.json",
        "sha256": "15235350abdb2cc803bc5fb4abca7b836347a500fa9f24a25fb16ea30945b360",
        "size_bytes": 8008,
    }
    assert c15.STAGE_CONTRACT[0] == "INPUT_HASH_AND_JSON"
    assert len(c15.STAGE_CONTRACT) == 12
    assert len(set(c15.STAGE_CONTRACT)) == 12
    assert c15.sha256_bytes(c15.canonical_bytes(list(c15.STAGE_CONTRACT))) == (
        "d6e2cb81fa08469a07322d8d84bef6b45a7ae0379dad9b89b4ce78ffcc7c8f3c"
    )
    assert len(c15.VIOLATION_CONTRACT) == 8
    assert len(set(c15.VIOLATION_CONTRACT)) == 8
    assert "_PoolDiagnosticReceipt" not in SOURCE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "data",
    (
        b"\xff",
        b'{"schema_version":1,"schema_version":1}',
        b'{"span_mm":NaN}',
    ),
    ids=("invalid-utf8", "duplicate-json-key", "nonfinite-json"),
)
def test_invalid_json_inputs_fail_closed_through_exact36_public_path(data: bytes) -> None:
    c13_result = _c13_public_cohort_result(data)
    assert c13_result["first_failed_candidate13_diagnostic_stage"] == c13.STAGE_CONTRACT[0]
    context = _context()
    receipts = _cohort_receipts_with_first_data(context, data)
    with pytest.raises(
        ValueError, match="did not reproduce the exact C14 C13 aggregate"
    ):
        c15.aggregate_geometry_representation_receipts(context, receipts)
    with pytest.raises(PermissionError):
        c15.aggregate_geometry_representation_receipts(context, receipts)


def test_noncanonical_valid_json_is_accepted_through_exact36_public_path() -> None:
    data = json.dumps(_pool(), ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")
    assert data != c15.canonical_bytes(_pool())
    context = _context()
    result = c15.aggregate_geometry_representation_receipts(
        context, _cohort_receipts_with_first_data(context, data)
    )
    assert result == _run()


@pytest.mark.parametrize(
    ("stage_index", "stage_name"),
    tuple(enumerate(c13.STAGE_CONTRACT[1:6], start=1)),
)
def test_each_c13_prefix_stage_1_to_5_failure_fails_closed_on_public_path(
    stage_index: int, stage_name: str
) -> None:
    data = c15.canonical_bytes(_prefix_failure_pool(stage_index))
    c13_result = _c13_public_cohort_result(data)
    assert c13_result["first_failed_candidate13_diagnostic_stage"] == stage_name
    context = _context()
    receipts = _cohort_receipts_with_first_data(context, data)
    with pytest.raises(
        ValueError, match="did not reproduce the exact C14 C13 aggregate"
    ):
        c15.aggregate_geometry_representation_receipts(context, receipts)


@pytest.mark.parametrize(
    "states",
    tuple(product(("absent", "null", "value"), repeat=3)),
)
def test_all_aux_absent_null_value_combinations_use_exact36_public_path(
    states: tuple[str, str, str]
) -> None:
    fields = ("anchor", "axis_direction", "span_mm")
    matching = {
        "anchor": [10.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 20.0,
    }
    row = _row()
    for field, state in zip(fields, states):
        if state == "null":
            row[field] = None
        elif state == "value":
            row[field] = _clone(matching[field])
    result = _run(_pool(row))
    assert result["c13_aggregate_stage6_parity_for_cohort"] is True
    assert result["exact_c9_full_outcome_parity_for_cohort"] is True
    assert result["aggregate_stage6_violation_coverage_for_cohort"] is True


def test_valid_pool_is_clear_and_non_authorizing() -> None:
    result = _run()
    assert result["stage_reached_all_bitset"] == "1" * 12
    assert result["stage_passed_all_bitset"] == "111111101111"
    assert result["violation_or_bitset"] == "00000100"
    assert result["status"] == "SOURCE_POOL_GEOMETRY_INCOMPATIBLE_NONAUTHORIZING"
    assert result["c13_aggregate_stage6_parity_for_cohort"] is True
    assert result["exact_c9_full_outcome_parity_for_cohort"] is True


def test_no_supported_geometry_is_separate() -> None:
    row = _row()
    del row["span_endpoints"]
    assert _mask_has(row, "NO_SUPPORTED_GEOMETRY")


def test_line_start_end_joint_or_absent() -> None:
    row = _row()
    row["line_start"] = [0.0, 0.0, 0.0]
    assert _mask_has(row, "INCOMPLETE_LINE_ENDPOINT_PAIR")


def test_every_aux_presence_subset_and_explicit_null_uses_c9_key_semantics() -> None:
    fields = ("anchor", "axis_direction", "span_mm")
    matching = {
        "anchor": [10.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 20.0,
    }
    for mask in range(8):
        row = _row()
        for index, field in enumerate(fields):
            if mask & (1 << index):
                row[field] = _clone(matching[field])
        assert not _mask_has(row, "PARTIAL_AUX_WITHOUT_SPAN")
    for field in fields:
        row = _row()
        row[field] = None
        assert not _mask_has(row, "PARTIAL_AUX_WITHOUT_SPAN")
    row = _row()
    del row["span_endpoints"]
    row["anchor"] = None
    assert _mask_has(row, "PARTIAL_AUX_WITHOUT_SPAN")


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("span_endpoints", [[0.0, 0.0], [20.0, 0.0, 0.0]]),
        ("line_start", [0.0, "bad", 0.0]),
        ("anchor", True),
        ("axis_direction", [10**10000, 0.0, 0.0]),
        ("span_mm", "20"),
    ),
)
def test_shape_type_and_finite_failures(field: str, bad) -> None:
    row = _row()
    row[field] = bad
    if field == "line_start":
        row["line_end"] = [20.0, 0.0, 0.0]
    assert _mask_has(row, "MALFORMED_GEOMETRY_VALUE")


@pytest.mark.parametrize("mutation", ("coordinate", "zero_span", "short", "axis", "aux_span"))
def test_conditioned_domain_and_degeneracy(mutation: str) -> None:
    row = _row()
    if mutation == "coordinate":
        row["span_endpoints"] = [[0.0, 0.0, 0.0], [1_000_001.0, 0.0, 0.0]]
    elif mutation == "zero_span":
        row["span_endpoints"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    elif mutation == "short":
        row["span_endpoints"] = [[0.0, 0.0, 0.0], [0.0001, 0.0, 0.0]]
    elif mutation == "axis":
        row.update({"anchor": [10.0, 0.0, 0.0], "axis_direction": [0.0, 0.0, 0.0]})
    else:
        row.update({"anchor": [10.0, 0.0, 0.0], "span_mm": 0.0})
    assert _mask_has(row, "OUT_OF_DOMAIN_OR_DEGENERATE")


def test_partial_aux_reconstruction_conflict_and_exact_tolerance() -> None:
    inside = _row()
    inside["anchor"] = [10.0000005, 0.0, 0.0]
    assert not _mask_has(inside, "PARTIAL_AUX_SPAN_CONFLICT")
    boundary = _row()
    boundary["anchor"] = [10.000001, 0.0, 0.0]
    assert not _mask_has(boundary, "PARTIAL_AUX_SPAN_CONFLICT")
    outside = _row()
    outside["anchor"] = [10.0000011, 0.0, 0.0]
    assert _mask_has(outside, "PARTIAL_AUX_SPAN_CONFLICT")


def test_c6_ordered_reference_tolerance_chain_is_not_all_pairs() -> None:
    row = _row()
    row.update(
        {
            "line_start": [0.00000075, 0.0, 0.0],
            "line_end": [20.00000075, 0.0, 0.0],
            "anchor": [9.99999925, 0.0, 0.0],
            "axis_direction": [1.0, 0.0, 0.0],
            "span_mm": 20.0,
        }
    )
    assert not _mask_has(row, "C6_ORDERED_REFERENCE_REPRESENTATION_CONFLICT")


def test_c6_ordered_reference_conflict() -> None:
    row = _row()
    row.update(
        {
            "line_start": [0.000002, 0.0, 0.0],
            "line_end": [20.000002, 0.0, 0.0],
        }
    )
    assert _mask_has(row, "C6_ORDERED_REFERENCE_REPRESENTATION_CONFLICT")


def test_multibit_diagnosis_is_not_short_circuited() -> None:
    row = _row()
    del row["span_endpoints"]
    row["line_start"] = [0.0, 0.0, 0.0]
    row["anchor"] = ["bad", 0.0, 0.0]
    assert _mask_has(row, "INCOMPLETE_LINE_ENDPOINT_PAIR")
    assert _mask_has(row, "PARTIAL_AUX_WITHOUT_SPAN")
    assert _mask_has(row, "MALFORMED_GEOMETRY_VALUE")


def test_malformed_representation_does_not_suppress_independent_domain_defect() -> None:
    row = _row()
    row["line_start"] = ["bad", 0.0, 0.0]
    row["line_end"] = [20.0, 0.0, 0.0]
    row["axis_direction"] = [0.0, 0.0, 0.0]
    mask = _mask(row)
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("MALFORMED_GEOMETRY_VALUE"))
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("OUT_OF_DOMAIN_OR_DEGENERATE"))


@pytest.mark.parametrize("complete", (False, True))
def test_malformed_only_aux_sets_shape_not_domain(complete: bool) -> None:
    row = _row()
    row["anchor"] = "bad" if not complete else True
    if complete:
        row["axis_direction"] = [1.0, 0.0, 0.0]
        row["span_mm"] = 20.0
    mask = _mask(row)
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("MALFORMED_GEOMETRY_VALUE"))
    assert not mask & (1 << c15.VIOLATION_CONTRACT.index("OUT_OF_DOMAIN_OR_DEGENERATE"))


def test_partial_aux_without_span_still_classifies_axis_degeneracy() -> None:
    row = _row()
    del row["span_endpoints"]
    row["axis_direction"] = [0.0, 0.0, 0.0]
    mask = _mask(row)
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("PARTIAL_AUX_WITHOUT_SPAN"))
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("OUT_OF_DOMAIN_OR_DEGENERATE"))


def test_malformed_span_peer_does_not_suppress_finite_endpoint_domain_defect() -> None:
    row = _row()
    row["span_endpoints"] = [["bad", 0.0, 0.0], [1_000_001.0, 0.0, 0.0]]
    mask = _mask(row)
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("MALFORMED_GEOMETRY_VALUE"))
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("OUT_OF_DOMAIN_OR_DEGENERATE"))


def test_partial_aux_without_span_does_not_invent_nominal_span_max_domain() -> None:
    row = _row()
    del row["span_endpoints"]
    row["span_mm"] = c6.MAX_CONDITIONED_LINE_LENGTH_MM + 1.0
    mask = _mask(row)
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("PARTIAL_AUX_WITHOUT_SPAN"))
    assert not mask & (1 << c15.VIOLATION_CONTRACT.index("OUT_OF_DOMAIN_OR_DEGENERATE"))


@pytest.mark.parametrize(
    ("anchor", "axis", "span_mm"),
    (
        (
            [1000.0, 1000.0, 1000.0],
            [1.0, 1.0, 0.0],
            math.nextafter(c6.MIN_CONDITIONED_LINE_LENGTH_MM, 0.0),
        ),
        (
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            math.nextafter(c6.MAX_CONDITIONED_LINE_LENGTH_MM, math.inf),
        ),
    ),
)
def test_nextafter_nominal_span_boundary_defers_to_c6_reconstructed_geometry(
    anchor: list[float], axis: list[float], span_mm: float
) -> None:
    row = _row()
    del row["span_endpoints"]
    row.update(
        {"anchor": anchor, "axis_direction": axis, "span_mm": span_mm}
    )
    assert c6._endpoints_from_row(row, name="nextafter regression") is not None
    assert not _mask_has(row, "OUT_OF_DOMAIN_OR_DEGENERATE")


def test_huge_json_integer_is_contained_as_malformed_not_raised() -> None:
    row = _row()
    row["axis_direction"] = [10**10000, 0, 0]
    mask = _mask(row)
    assert mask & (1 << c15.VIOLATION_CONTRACT.index("MALFORMED_GEOMETRY_VALUE"))


def test_feature_only_c9_failure_is_not_mislabeled_geometry() -> None:
    row = _row()
    row["score"] = "not-a-number"
    result = _run(_pool(row))
    baseline = _run()
    assert result["violation_or_bitset"] == baseline["violation_or_bitset"]
    assert result["stage_passed_all_bitset"] == baseline["stage_passed_all_bitset"]
    assert result["exact_c9_full_outcome_parity_for_cohort"] is True
    assert result["performance_tested"] is False


def test_source_fields_do_not_change_geometry_classification() -> None:
    base = _run()
    row = _row()
    row["source_kind"] = "plane_lattice"
    changed = _run(_pool(row))
    assert base["violation_or_bitset"] == changed["violation_or_bitset"]


def test_exact36_slots_and_no_37th_issue() -> None:
    context = _context()
    receipts = _issue(context, count=35)
    with pytest.raises(ValueError):
        c15.aggregate_geometry_representation_receipts(context, receipts)
    last = _issue(context, _fixed_c14_geometry_failure_pool(), count=1)[0]
    all_receipts = receipts + (last,)
    with pytest.raises(PermissionError):
        _issue(context, count=1)
    assert c15.aggregate_geometry_representation_receipts(context, all_receipts)[
        "c13_aggregate_called_exactly_once_by_core"
    ] is True


def test_duplicate_reorder_cross_context_and_replay_are_rejected() -> None:
    first = _context()
    first_receipts = _cohort_receipts(first)
    with pytest.raises(PermissionError):
        c15.aggregate_geometry_representation_receipts(
            first, first_receipts[:-1] + (first_receipts[0],)
        )
    with pytest.raises(PermissionError):
        c15.aggregate_geometry_representation_receipts(
            first, tuple(reversed(first_receipts))
        )
    second = _context()
    second_receipts = _cohort_receipts(second)
    with pytest.raises(PermissionError):
        c15.aggregate_geometry_representation_receipts(second, first_receipts)
    result = c15.aggregate_geometry_representation_receipts(first, first_receipts)
    assert result["public_result_is_aggregate_only"] is True
    with pytest.raises(PermissionError):
        c15.aggregate_geometry_representation_receipts(first, first_receipts)
    c15.aggregate_geometry_representation_receipts(second, second_receipts)


def test_handles_are_fieldless_constant_repr_and_nonserializable() -> None:
    context = _context()
    receipt = _issue(context, count=1)[0]
    assert repr(context) == "<Candidate15OpaqueContext>"
    assert repr(receipt) == "<Candidate15OpaqueGeometryReceipt>"
    assert not hasattr(context, "__dict__")
    assert not hasattr(receipt, "__dict__")
    for value in (context, receipt):
        with pytest.raises(TypeError):
            copy.copy(value)
        with pytest.raises(TypeError):
            pickle.dumps(value)


def test_forged_handles_and_callback_subclasses_are_rejected() -> None:
    with pytest.raises(PermissionError):
        c15.localize_geometry_representation_contract(
            c15.LocalizerContext(), pool_bytes=b"{}", expected_pool_sha256="1" * 64
        )
    context = _context()

    class HostileBytes(bytes):
        def __bytes__(self):
            raise AssertionError("callback")

    class HostileString(str):
        def __str__(self):
            raise AssertionError("callback")

    with pytest.raises(TypeError):
        c15.localize_geometry_representation_contract(
            context,
            pool_bytes=HostileBytes(b"{}"),
            expected_pool_sha256="1" * 64,
        )
    with pytest.raises(TypeError):
        c15.localize_geometry_representation_contract(
            context,
            pool_bytes=b"{}",
            expected_pool_sha256=HostileString("1" * 64),
        )


def test_hostile_binding_mapping_is_rejected_without_callbacks() -> None:
    class HostileDict(dict):
        def keys(self):
            raise AssertionError("callback")

        def __iter__(self):
            raise AssertionError("callback")

        def __getitem__(self, _key):
            raise AssertionError("callback")

    with pytest.raises(ValueError):
        c15.make_verified_context(
            c13_localizer=c13,
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=HostileDict(c15.FIXED_BINDINGS),
        )


@pytest.mark.parametrize("binding", sorted(c15.FIXED_BINDINGS))
def test_every_binding_drift_is_rejected(binding: str) -> None:
    bindings = _clone(c15.FIXED_BINDINGS)
    bindings[binding]["size_bytes"] += 1
    with pytest.raises(ValueError):
        c15.make_verified_context(
            c13_localizer=c13,
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=bindings,
        )


def test_fake_dependency_and_c13_export_drift_are_rejected() -> None:
    with pytest.raises(ValueError):
        c15.make_verified_context(
            c13_localizer=types.ModuleType("fake"),
            c9_adapter=c9,
            c6_hypothesis=c6,
            builder=builder,
            clusterer=clusterer,
            bindings=c15.FIXED_BINDINGS,
        )
    context = _context()
    original = c13.localize_pool_adapter_contract
    c13.localize_pool_adapter_contract = lambda *_args, **_kwargs: None
    try:
        with pytest.raises(PermissionError):
            _issue(context, count=1)
    finally:
        c13.localize_pool_adapter_contract = original


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("GEOMETRY_EQUIVALENCE_TOLERANCE_MM", 2.0e-6),
        ("C13_GEOMETRY_STAGE_INDEX", 7),
        ("SCHEMA_VERSION", 1.0),
        ("RESULT_KEYS", set(c15.RESULT_KEYS)),
        ("C13_RUNTIME_CONSTANTS", list(c15.C13_RUNTIME_CONSTANTS)),
        ("C9_RUNTIME_CONSTANTS", list(c15.C9_RUNTIME_CONSTANTS)),
        ("C6_RUNTIME_CONSTANTS", list(c15.C6_RUNTIME_CONSTANTS)),
        ("BUILDER_RUNTIME_CONSTANTS", list(c15.BUILDER_RUNTIME_CONSTANTS)),
    ),
)
def test_post_context_semantic_global_or_exact_type_drift_is_rejected(
    name: str, replacement: object
) -> None:
    context = _context()
    original = getattr(c15, name)
    setattr(c15, name, replacement)
    try:
        with pytest.raises(PermissionError):
            _issue(context, count=1)
    finally:
        setattr(c15, name, original)
    assert len(_issue(context, count=1)) == 1


def test_post_c13_baseexception_terminally_consumes_reserved_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedPostC13Failure(BaseException):
        pass

    original = c15._analyze_pool_record
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InjectedPostC13Failure("synthetic post-C13 failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(c15, "_analyze_pool_record", fail_once)
    context = _context()
    closure = _closure_values(c15.localize_geometry_representation_contract)
    receipt_registry = closure["receipt_registry"]
    paired_registry = closure["paired_c13_registry"]
    receipt_count = len(receipt_registry)
    paired_count = len(paired_registry)
    with pytest.raises(InjectedPostC13Failure):
        _issue(context, count=1)
    assert calls == 1
    assert len(receipt_registry) == receipt_count
    assert len(paired_registry) == paired_count
    with pytest.raises(PermissionError):
        _issue(context, count=1)
    with pytest.raises(PermissionError):
        c15.aggregate_geometry_representation_receipts(context, ())
    assert calls == 1
    assert len(receipt_registry) == receipt_count
    assert len(paired_registry) == paired_count


def test_public_closure_export_drift_is_detected() -> None:
    context = _context()
    original = c15.localize_geometry_representation_contract
    c15.localize_geometry_representation_contract = lambda *_args, **_kwargs: None
    try:
        with pytest.raises(PermissionError):
            original(
                context,
                pool_bytes=c15.canonical_bytes(_pool()),
                expected_pool_sha256=c15.sha256_bytes(c15.canonical_bytes(_pool())),
            )
    finally:
        c15.localize_geometry_representation_contract = original


def test_c13_aggregate_export_drift_is_detected_before_consumption() -> None:
    context = _context()
    receipts = _cohort_receipts(context)
    original = c13.aggregate_pool_adapter_receipts
    c13.aggregate_pool_adapter_receipts = lambda *_args, **_kwargs: {}
    try:
        with pytest.raises(PermissionError):
            c15.aggregate_geometry_representation_receipts(context, receipts)
    finally:
        c13.aggregate_pool_adapter_receipts = original
    assert c15.aggregate_geometry_representation_receipts(context, receipts)[
        "c13_aggregate_called_exactly_once_by_core"
    ] is True


def test_paired_c13_receipt_substitution_and_reverse_deletion_are_detected() -> None:
    context = _context()
    receipts = _issue(context, count=2)
    closure = _closure_values(c15.localize_geometry_representation_contract)
    receipt_registry = closure["receipt_registry"]
    paired_registry = closure["paired_c13_registry"]
    assert type(receipt_registry) is dict
    assert type(paired_registry) is dict
    first_entry = receipt_registry[receipts[0]]
    second_entry = receipt_registry[receipts[1]]
    substituted = first_entry[:9] + (second_entry[9],) + first_entry[10:]
    receipt_registry[receipts[0]] = substituted
    try:
        with pytest.raises(PermissionError):
            _issue(context, count=1)
    finally:
        receipt_registry[receipts[0]] = first_entry
    paired_entry = paired_registry.pop(id(first_entry[9]))
    try:
        with pytest.raises(PermissionError):
            _issue(context, count=1)
    finally:
        paired_registry[id(first_entry[9])] = paired_entry
    assert len(_issue(context, count=1)) == 1


def test_context_registry_entry_rebase_is_detected() -> None:
    first = _context()
    second = _context()
    localize_closure = _closure_values(c15.localize_geometry_representation_contract)
    resolve_context = localize_closure["_resolve_context"]
    resolve_closure = _closure_values(resolve_context)
    context_registry = resolve_closure["context_registry"]
    assert type(context_registry) is dict
    original_second = context_registry[second]
    context_registry[second] = context_registry[first]
    try:
        with pytest.raises(PermissionError):
            _issue(second, count=1)
    finally:
        context_registry[second] = original_second
    assert len(_issue(second, count=1)) == 1


def test_local_exact_c14_c13_validator_rejects_any_projection_drift() -> None:
    exact = _clone(c15.EXPECTED_C14_C13_AGGREGATE)
    c15._validate_exact_c14_c13_aggregate(exact)
    for key in exact:
        drifted = _clone(exact)
        value = drifted[key]
        if type(value) is bool:
            drifted[key] = not value
        else:
            drifted[key] = f"{value}-drift"
        with pytest.raises(ValueError):
            c15._validate_exact_c14_c13_aggregate(drifted)


def test_contract_is_honest_about_security_and_scientific_limits() -> None:
    contract = c15.contract_declaration()
    assert contract["tamper_evident_fresh_worker_runtime"] is True
    assert contract["arbitrary_same_process_python_adversary_resistance_claimed"] is False
    assert contract["isolated_authenticated_worker_required"] is True
    assert contract["outer_no_untrusted_code_or_reflection_worker_required"] is True
    assert contract["per_pool_c13_parity_claimed"] is False
    assert contract["per_pool_violation_exhaustiveness_claimed"] is False
    assert contract["performance_tested"] is False
    assert contract["metric_or_fit_performed"] is False
    assert contract["holdout_opened"] is False
