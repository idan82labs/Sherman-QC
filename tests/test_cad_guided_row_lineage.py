"""Invented integer row declarations only; no scan, CAD or native execution."""

from dataclasses import FrozenInstanceError
import importlib.util
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_row_lineage.py"


@pytest.fixture(scope="module")
def m():
    name = "invented_cad_row_lineage"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.modules.pop(name, None)
    return loaded


def test_smallest_public_plan_is_identity_and_has_disjoint_roles(m):
    plan = m.prepare_row_lineage(raw_row_count=48, datum_raw_row_indices=list(range(36)))
    assert plan.engine_to_raw_row_indices == tuple(range(48))
    assert plan.datum_engine_row_indices == tuple(range(36))
    assert plan.inspection_engine_row_indices == tuple(range(36, 48))
    assert plan.engine_row_count == 48 and plan.dropped_row_count == 0


def test_unsorted_caller_datum_order_survives_mapping_without_reordering_scan(m):
    datum_rows = list(reversed(range(0, 72, 2)))
    plan = m.prepare_row_lineage(
        raw_row_count=120, datum_raw_row_indices=datum_rows, max_engine_rows=48
    )
    assert plan.datum_raw_row_indices == tuple(datum_rows)
    assert tuple(plan.raw_row_for_engine_row(row) for row in plan.datum_engine_row_indices) == tuple(datum_rows)
    assert tuple(sorted(plan.engine_to_raw_row_indices)) == plan.engine_to_raw_row_indices
    assert len(plan.engine_to_raw_row_indices) == len(set(plan.engine_to_raw_row_indices)) == 48
    assert set(plan.datum_engine_row_indices).isdisjoint(plan.inspection_engine_row_indices)
    assert set(plan.datum_engine_row_indices) | set(plan.inspection_engine_row_indices) == set(range(48))
    datum_rows.clear()
    assert len(plan.datum_raw_row_indices) == 36  # Owned immutable snapshot.


def test_every_small_complement_and_rank_count_matches_independent_enumeration(m):
    # All datum subsets for raw counts 1..9; private rank tests intentionally
    # include smaller cases than the public datum/inspection admission limits.
    for raw_count in range(1, 10):
        for datum_count in range(raw_count):
            for datums in combinations(range(raw_count), datum_count):
                complement = [row for row in range(raw_count) if row not in datums]
                for selected_count in range(1, len(complement) + 1):
                    ranks = tuple(m._equally_spaced_ranks(len(complement), selected_count))
                    actual = tuple(m._complement_rows_at_ranks(datums, iter(ranks)))
                    expected = tuple(complement[rank] for rank in ranks)
                    assert actual == expected
                    assert len(actual) == len(set(actual)) == selected_count
                    assert actual == tuple(sorted(actual)) and not set(actual).intersection(datums)
                    assert actual[0] == complement[0]
                    if selected_count > 1:
                        assert actual[-1] == complement[-1]
                        gaps = [right - left for left, right in zip(ranks, ranks[1:])]
                        assert min(gaps) >= 1 and max(gaps) - min(gaps) <= 1


@pytest.mark.parametrize("raw_count", [48, 49, 60, 99, 128, 2_001])
@pytest.mark.parametrize("engine_cap", [48, 96, 200_000])
def test_public_plans_have_exact_counts_bidirectional_lineage_and_endpoint_inclusion(m, raw_count, engine_cap):
    datums = tuple(reversed([row * raw_count // 36 for row in range(36)]))
    plan = m.prepare_row_lineage(
        raw_row_count=raw_count, datum_raw_row_indices=datums, max_engine_rows=engine_cap
    )
    target = min(raw_count, engine_cap)
    complement = [row for row in range(raw_count) if row not in datums]
    k = target - len(datums)
    expected_inspection = tuple(complement[j * (len(complement) - 1) // (k - 1)] for j in range(k))
    actual_inspection = tuple(plan.raw_row_for_engine_row(row) for row in plan.inspection_engine_row_indices)
    assert actual_inspection == expected_inspection
    assert actual_inspection[0] == complement[0] and actual_inspection[-1] == complement[-1]
    assert plan.engine_row_count == target
    assert plan.dropped_row_count == raw_count - target
    assert len(plan.datum_engine_row_indices) == 36 and len(actual_inspection) == k
    assert plan.engine_to_raw_row_indices == tuple(sorted(set(datums) | set(expected_inspection)))
    for engine_row, raw_row in enumerate(plan.engine_to_raw_row_indices):
        assert plan.engine_row_for_raw_row(raw_row) == engine_row
        assert plan.raw_row_for_engine_row(engine_row) == raw_row
    retained = set(plan.engine_to_raw_row_indices)
    assert all(plan.engine_row_for_raw_row(row) is None for row in range(raw_count) if row not in retained)
    assert plan == m.prepare_row_lineage(
        raw_row_count=raw_count, datum_raw_row_indices=datums, max_engine_rows=engine_cap
    )


@pytest.mark.parametrize("value", [None, True, False, 48.0, "48", 0, 47, -1, 1_000_001,
                                  2**100, np.uint64(48), np.int64(48)])
def test_raw_count_refuses_coercion_and_out_of_budget_values(m, value):
    with pytest.raises(m.RowLineageUnavailable, match="RAW_ROW_COUNT_BUDGET"):
        m.prepare_row_lineage(raw_row_count=value, datum_raw_row_indices=list(range(36)))


@pytest.mark.parametrize("value", [None, True, False, 48.0, "48", 0, 47, -1, 200_001,
                                  2**100, np.uint64(48), np.int64(48)])
def test_engine_cap_refuses_coercion_and_out_of_budget_values(m, value):
    with pytest.raises(m.RowLineageUnavailable, match="ENGINE_ROW_COUNT_BUDGET"):
        m.prepare_row_lineage(raw_row_count=100, datum_raw_row_indices=list(range(36)), max_engine_rows=value)


@pytest.mark.parametrize("value", [None, True, False, 0.0, "0", -1, 100, 2**100,
                                  np.uint64(0), np.int64(0), np.uint64(2**64 - 1)])
def test_datum_indices_never_wrap_or_silently_coerce(m, value):
    datums = list(range(36))
    datums[0] = value
    with pytest.raises(m.RowLineageUnavailable, match="DATUM_RAW_ROW_INDEX"):
        m.prepare_row_lineage(raw_row_count=100, datum_raw_row_indices=datums)


@pytest.mark.parametrize("datums", [None, "not rows", set(range(36)), np.arange(36),
                                   [], list(range(35)), list(range(5_001))])
def test_bounded_container_and_datum_count_are_checked_before_iteration(m, datums):
    with pytest.raises(m.RowLineageUnavailable, match="DATUM_ROW_COUNT_OR_CONTAINER"):
        m.prepare_row_lineage(raw_row_count=10_000, datum_raw_row_indices=datums)


def test_generator_is_not_consumed_and_list_subclass_is_not_dispatched(m):
    def forbidden_generator():
        raise AssertionError("must refuse before consuming iterable")
        yield 0

    class ForbiddenList(list):
        def __len__(self):
            raise AssertionError("must refuse list subclass before len")

        def __iter__(self):
            raise AssertionError("must refuse list subclass before iter")

    for datums in [forbidden_generator(), ForbiddenList()]:
        with pytest.raises(m.RowLineageUnavailable, match="DATUM_ROW_COUNT_OR_CONTAINER"):
            m.prepare_row_lineage(raw_row_count=100, datum_raw_row_indices=datums)


def test_duplicate_or_insufficient_inspection_budget_is_rejected_without_dropping_datums(m):
    with pytest.raises(m.RowLineageUnavailable, match="DUPLICATE_DATUM_RAW_ROW_INDEX"):
        m.prepare_row_lineage(raw_row_count=100, datum_raw_row_indices=[0] * 36)
    for raw_count, datum_count, cap in [(48, 37, 200_000), (100, 89, 200_000), (100, 40, 48)]:
        with pytest.raises(m.RowLineageUnavailable, match="INSUFFICIENT_INSPECTION_ROW_BUDGET"):
            m.prepare_row_lineage(
                raw_row_count=raw_count, datum_raw_row_indices=list(range(datum_count)), max_engine_rows=cap
            )


@pytest.mark.parametrize("counts", [(0, 0), (0, 1), (1, 0), (3, 4), (-1, 1),
                                   (True, 1), (5, True), (5.0, 2), (5, 2.0)])
def test_rank_solver_rejects_invalid_counts(m, counts):
    with pytest.raises(m.RowLineageUnavailable, match="INVALID_RANK_SELECTION_COUNTS"):
        tuple(m._equally_spaced_ranks(*counts))


@pytest.mark.parametrize("value", [None, True, False, 0.0, "0", -1, 48, 2**64 - 1, np.uint64(0)])
def test_bidirectional_lookup_refuses_noninteger_and_out_of_bounds_indices(m, value):
    plan = m.prepare_row_lineage(raw_row_count=48, datum_raw_row_indices=list(range(36)))
    with pytest.raises(m.RowLineageUnavailable, match="RAW_ROW_INDEX_OUT_OF_BOUNDS"):
        plan.engine_row_for_raw_row(value)
    with pytest.raises(m.RowLineageUnavailable, match="ENGINE_ROW_INDEX_OUT_OF_BOUNDS"):
        plan.raw_row_for_engine_row(value)


def test_experimental_declared_semantics_cannot_be_confused_with_physical_qualification(m):
    plan = m.prepare_row_lineage(raw_row_count=48, datum_raw_row_indices=list(range(36)))
    assert plan.row_map_status == "DECLARED_ROW_MAP_ONLY"
    assert plan.selection_method == "ENDPOINT_INCLUSIVE_COMPLEMENT_RANK_FLOOR_V1"
    assert plan.source_binding_status == "NOT_BOUND_TO_SOURCE_BYTES"
    assert plan.physical_coverage_status == "UNKNOWN"
    assert plan.datum_selection_status == "CALLER_DECLARED_NOT_PHYSICALLY_VERIFIED"
    assert plan.integration_status == "UNCONNECTED_EXPERIMENTAL"
    assert plan.production_use_allowed is False
    with pytest.raises(FrozenInstanceError):
        plan.raw_row_count = 100
    with pytest.raises(TypeError):
        plan.engine_to_raw_row_indices[0] = 12


def test_largest_declared_plan_is_bounded_without_enumerating_the_raw_population(m, monkeypatch, record_property):
    builtin_range = range

    def output_bounded_range(*args):
        result = builtin_range(*args)
        assert len(result) <= m.MAX_ENGINE_ROWS, "must not enumerate a raw-count-sized range"
        return result

    monkeypatch.setattr(m, "range", output_bounded_range, raising=False)
    # Unsorted datum declarations include both raw endpoints and internal runs.
    datums = tuple(reversed(tuple(range(4_999)) + (999_999,)))
    started = perf_counter()
    plan = m.prepare_row_lineage(raw_row_count=1_000_000, datum_raw_row_indices=datums)
    elapsed = perf_counter() - started
    record_property("row_lineage_1m_to_200k_elapsed_seconds", elapsed)
    assert plan.engine_row_count == 200_000 and plan.dropped_row_count == 800_000
    assert len(plan.datum_engine_row_indices) == 5_000
    assert len(plan.inspection_engine_row_indices) == 195_000
    assert plan.engine_to_raw_row_indices[0] == 0 and plan.engine_to_raw_row_indices[-1] == 999_999
    assert plan.raw_row_for_engine_row(plan.inspection_engine_row_indices[0]) == 4_999
    assert plan.raw_row_for_engine_row(plan.inspection_engine_row_indices[-1]) == 999_998
    assert len(set(plan.engine_to_raw_row_indices)) == 200_000
    assert plan.datum_raw_row_indices == datums
    assert tuple(plan.raw_row_for_engine_row(row) for row in plan.datum_engine_row_indices) == datums
    assert set(plan.datum_engine_row_indices).isdisjoint(plan.inspection_engine_row_indices)
