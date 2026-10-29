"""Unconnected, geometry-blind preparation of an explicit raw-row thinning plan.

This helper sees counts and caller-selected datum row numbers only. It does not
read or transform a scan, select stable datums, bind source bytes, verify units or
frames, or establish physical coverage. It is NOT approved product preprocessing
and does not change the existing CAD-guided file loader's 200,000-point cap.

The returned maps say only ``DECLARED_ROW_MAP_ONLY``. An actual input adapter
would still need independent custody, source binding, datum/frame and sampling
qualification before applying this plan. In particular, equal spacing in file
row rank is not equal spatial spacing or a guarantee that any bend survives.
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from heapq import merge


MAX_RAW_ROWS = 1_000_000
MAX_ENGINE_ROWS = 200_000
MIN_DATUM_ROWS = 36
MAX_DATUM_ROWS = 5_000
MIN_INSPECTION_ROWS = 12


class RowLineageUnavailable(ValueError):
    """A declaration cannot produce the bounded plan; not a part defect."""


@dataclass(frozen=True, slots=True)
class RowLineagePlan:
    """Immutable sparse row map, not an authenticated provenance record.

    ``engine_to_raw_row_indices[e]`` is the original row for engine row ``e``.
    Its strict ascending order preserves file order. The reverse direction is
    available through ``engine_row_for_raw_row`` without a raw-count-sized map.
    Datum tuples preserve the caller's datum order, including when it is not
    sorted; callers must keep any associated datum data in that same order.
    Inspection engine rows are ascending and exclude every declared datum row.
    """

    raw_row_count: int
    max_engine_rows: int
    engine_to_raw_row_indices: tuple[int, ...]
    datum_raw_row_indices: tuple[int, ...]
    datum_engine_row_indices: tuple[int, ...]
    inspection_engine_row_indices: tuple[int, ...]
    row_map_status: str = field(default="DECLARED_ROW_MAP_ONLY", init=False)
    selection_method: str = field(
        default="ENDPOINT_INCLUSIVE_COMPLEMENT_RANK_FLOOR_V1", init=False
    )
    source_binding_status: str = field(default="NOT_BOUND_TO_SOURCE_BYTES", init=False)
    physical_coverage_status: str = field(default="UNKNOWN", init=False)
    datum_selection_status: str = field(
        default="CALLER_DECLARED_NOT_PHYSICALLY_VERIFIED", init=False
    )
    integration_status: str = field(default="UNCONNECTED_EXPERIMENTAL", init=False)
    production_use_allowed: bool = field(default=False, init=False)

    @property
    def engine_row_count(self):
        return len(self.engine_to_raw_row_indices)

    @property
    def dropped_row_count(self):
        return self.raw_row_count - self.engine_row_count

    def engine_row_for_raw_row(self, raw_row_index):
        """Return the retained engine row, or None if that raw row was dropped."""
        if type(raw_row_index) is not int or not 0 <= raw_row_index < self.raw_row_count:
            raise RowLineageUnavailable("RAW_ROW_INDEX_OUT_OF_BOUNDS")
        engine_row = bisect_left(self.engine_to_raw_row_indices, raw_row_index)
        if (engine_row == self.engine_row_count
                or self.engine_to_raw_row_indices[engine_row] != raw_row_index):
            return None
        return engine_row

    def raw_row_for_engine_row(self, engine_row_index):
        """Return the original row, rejecting invalid indices without coercion."""
        if type(engine_row_index) is not int or not 0 <= engine_row_index < self.engine_row_count:
            raise RowLineageUnavailable("ENGINE_ROW_INDEX_OUT_OF_BOUNDS")
        return self.engine_to_raw_row_indices[engine_row_index]


def _equally_spaced_ranks(population_count, selected_count):
    """Yield K integer ranks from N: floor(j*(N-1)/(K-1)), j=0..K-1.

    With 2 <= K <= N, adjacent differences are positive and either the floor
    or ceiling of (N-1)/(K-1). Both endpoints are therefore included exactly.
    The K=1 convention retains rank zero; public plans require K >= 12.
    Integer arithmetic avoids floating-point rounding, duplicate rounding and
    fixed-width overflow. This iterator stores no population-sized collection.
    """
    if (type(population_count) is not int or type(selected_count) is not int
            or not 1 <= selected_count <= population_count):
        raise RowLineageUnavailable("INVALID_RANK_SELECTION_COUNTS")
    if selected_count == 1:
        yield 0
        return
    for selected_rank in range(selected_count):
        yield selected_rank * (population_count - 1) // (selected_count - 1)


def _complement_rows_at_ranks(sorted_datum_rows, ranks):
    """Map increasing non-datum ranks to raw rows with one forward datum cursor.

    Private inputs are already validated by the public planner. For rank r,
    seek raw row x = r + number_of_datums_at_or_before(x). Each crossed datum
    increases x by one, so clusters and endpoint datums are handled exactly.
    Increasing ranks let the datum cursor advance at most D times in total.
    """
    crossed_datums = 0
    for rank in ranks:
        raw_row = rank + crossed_datums
        while (crossed_datums < len(sorted_datum_rows)
               and sorted_datum_rows[crossed_datums] <= raw_row):
            crossed_datums += 1
            raw_row = rank + crossed_datums
        yield raw_row


def prepare_row_lineage(*, raw_row_count, datum_raw_row_indices,
                        max_engine_rows=MAX_ENGINE_ROWS):
    """Build a deterministic bounded plan without reading coordinates or files.

    Counts and row indices must be exact built-in Python ints, not bools,
    floats, NumPy scalars or values coerced through an integer cast. Datums must
    be a bounded list/tuple of unique raw indices. Generators and arbitrary
    iterable objects are refused before iteration. Every datum is retained;
    the remaining engine budget is sampled over the datum *complement* only.

    The output contains min(raw_row_count, max_engine_rows) rows, with at least
    36 and at most 5,000 datum rows plus at least 12 inspection rows. A budget
    that cannot retain all datums and that inspection minimum fails unchanged.
    Preparation uses O(E + D) space, O(E + D log E + D log D) time for E retained
    rows and D datums; it never materializes a raw-count-sized mask/list/map.
    """
    if (type(raw_row_count) is not int
            or not MIN_DATUM_ROWS + MIN_INSPECTION_ROWS <= raw_row_count <= MAX_RAW_ROWS):
        raise RowLineageUnavailable("RAW_ROW_COUNT_BUDGET")
    if (type(max_engine_rows) is not int
            or not MIN_DATUM_ROWS + MIN_INSPECTION_ROWS <= max_engine_rows <= MAX_ENGINE_ROWS):
        raise RowLineageUnavailable("ENGINE_ROW_COUNT_BUDGET")
    if (type(datum_raw_row_indices) not in (list, tuple)
            or not MIN_DATUM_ROWS <= len(datum_raw_row_indices) <= MAX_DATUM_ROWS):
        raise RowLineageUnavailable("DATUM_ROW_COUNT_OR_CONTAINER")

    # Cap the snapshot too: another caller thread mutating a builtin list must
    # not turn the earlier length check into an unbounded copy.
    datum_rows = tuple(datum_raw_row_indices[:MAX_DATUM_ROWS + 1])
    if not MIN_DATUM_ROWS <= len(datum_rows) <= MAX_DATUM_ROWS:
        raise RowLineageUnavailable("DATUM_ROW_COUNT_OR_CONTAINER")
    if any(type(row) is not int or not 0 <= row < raw_row_count for row in datum_rows):
        raise RowLineageUnavailable("DATUM_RAW_ROW_INDEX")
    datum_set = set(datum_rows)
    if len(datum_set) != len(datum_rows):
        raise RowLineageUnavailable("DUPLICATE_DATUM_RAW_ROW_INDEX")
    engine_row_count = min(raw_row_count, max_engine_rows)
    inspection_row_count = engine_row_count - len(datum_rows)
    if inspection_row_count < MIN_INSPECTION_ROWS:
        raise RowLineageUnavailable("INSUFFICIENT_INSPECTION_ROW_BUDGET")

    sorted_datums = tuple(sorted(datum_rows))
    inspection_ranks = _equally_spaced_ranks(
        raw_row_count - len(datum_rows), inspection_row_count
    )
    inspection_raw_rows = _complement_rows_at_ranks(sorted_datums, inspection_ranks)
    engine_to_raw = tuple(merge(sorted_datums, inspection_raw_rows))
    datum_engine = tuple(bisect_left(engine_to_raw, row) for row in datum_rows)
    inspection_engine = tuple(
        engine_row for engine_row, raw_row in enumerate(engine_to_raw)
        if raw_row not in datum_set
    )
    return RowLineagePlan(
        raw_row_count=raw_row_count,
        max_engine_rows=max_engine_rows,
        engine_to_raw_row_indices=engine_to_raw,
        datum_raw_row_indices=datum_rows,
        datum_engine_row_indices=datum_engine,
        inspection_engine_row_indices=inspection_engine,
    )
