"""Prospective invented-only tests; this file has not been executed by its author.

The independent oracle uses individual Python binary64 operations, never the
helper's staged function, matrix multiplication, or a reduction/sum routine.
It tests the declared rounded program, not legacy BLAS equivalence, an exact-real
dot product, physical uncertainty, or general speedup. One explicitly authorized
900,000-row analytic capacity case uses only three axis queries. The next largest
input is one 8,192-row invented plate; all remaining inputs have at most 1,025 rows.
"""
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_raw_extrema_index_v1.py"
AXIS = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)


@pytest.fixture(scope="module")
def extrema():
    """Load only the named pure module, without importing the backend package."""
    name = "_invented_raw_extrema_v1_contract"
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def scalar_projection(row, direction, origin, constant):
    x = float(row[0]) - float(origin[0])
    y = float(row[1]) - float(origin[1])
    z = float(row[2]) - float(origin[2])
    tx = x * float(direction[0])
    ty = y * float(direction[1])
    tz = z * float(direction[2])
    first = tx + ty
    second = first + tz
    answer = second + float(constant)
    return 0.0 if answer == 0.0 else answer


def brute(points, projection, rows=None):
    direction, origin, constant = projection
    identities = list(range(len(points))) if rows is None else [int(x) for x in rows]
    values = [(scalar_projection(points[i], direction, origin, constant), i)
              for i in identities]
    lower, lower_id = min(values, key=lambda pair: (pair[0], pair[1]))
    upper, upper_id = max(values, key=lambda pair: (pair[0], -pair[1]))
    return lower, upper, lower_id, upper_id


def assert_exact(record, expected):
    lower, upper, lower_id, upper_id = expected
    assert struct.pack("!d", record["interval"][0]) == struct.pack("!d", lower)
    assert struct.pack("!d", record["interval"][1]) == struct.pack("!d", upper)
    assert type(record["minimum_raw_row"]) is int
    assert type(record["maximum_raw_row"]) is int
    assert record["minimum_raw_row"] == lower_id
    assert record["maximum_raw_row"] == upper_id


def assert_refused(answer, reason=None):
    assert answer["status"] == "UNAVAILABLE"
    assert answer["projections"] == []
    assert answer["accepted_registration"] is False
    assert answer["observed_bend_count"] is None
    assert answer["production_use_allowed"] is False
    for name in ("point_buffer_sha256", "support_rows_sha256", "support_scope"):
        assert name not in answer
    if reason is not None:
        assert answer["reasons"] == [reason]


def small_points():
    return np.asarray([[-3., 1., 0.], [2., -4., 1.], [1., 2., -3.],
                       [0., 0., 0.], [2., -4., 1.]], dtype=np.float64)


ROUNDING_CASES = (
    ([(0., 0., 0.), (1., 2., 3.), (-2., 4., -1.)], AXIS),
    ([(0., 0., 0.), (1., 2., 3.), (-2., 4., -1.)],
     ((-2., 3., -4.), (1., -2., 3.), 7.)),
    ([(1., 1., 1.), (1., 1., -1.)],
     ((1., float.fromhex("0x1p-53"), -1.), (0., 0., 0.), 0.)),
    ([(1., 1., 1.), (1., 1., -1.)],
     ((1., -1., float.fromhex("0x1p-53")), (0., 0., 0.), 0.)),
    ([(1_000_000., 1_000_000., 1.), (999_999., 999_999., -1.)],
     ((1_000_000., -1_000_000., 1.), (0., 0., 0.), 1.)),
    ([(1_000_000., -1_000_000., 0.), (999_999.9999999999, -1_000_000., 0.)],
     ((1., 1., -0.), (1_000_000., -1_000_000., 0.), -0.)),
    ([(float.fromhex("0x0.0000000000001p-1022"), 0., 0.),
      (float.fromhex("0x0.0000000000002p-1022"), 0., 0.)],
     ((1., 0., 0.), (0., 0., 0.), 0.)),
    ([(float.fromhex("0x1p-1022"), 0., 0.),
      (-float.fromhex("0x1p-1022"), 0., 0.)],
     ((.5, 0., 0.), (0., 0., 0.), 0.)),
    ([(-0., 0., -0.), (0., -0., 0.)],
     ((-0., -1., 0.), (-0., 0., -0.), -0.)),
    ([(1., -1., 0.), (-1., 1., 0.)],
     ((1., 1., 0.), (0., 0., 0.), -0.)),
    ([(1., 2., 3.), (-3., -2., -1.)],
     ((0., 0., 0.), (17., -19., 23.), 1_000_000.)),
    ([(.1, .2, .3), (.3, .2, .1), (-.3, -.2, -.1)],
     ((.7, -.9, .11), (.01, -.02, .03), -.13)),
)


@pytest.mark.parametrize("coordinates,projection", ROUNDING_CASES)
def test_staged_scalar_rounding_exact(extrema, coordinates, projection):
    points = np.asarray(coordinates, dtype=np.float64)
    answer = extrema.project_raw_extrema_v1(points, (projection,))
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], brute(points, projection))
    assert answer["arithmetic"] == "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1"
    assert answer["bound_scope"] == "ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL"
    assert answer["accepted_registration"] is False
    assert answer["observed_bend_count"] is None
    assert answer["production_use_allowed"] is False


@pytest.mark.parametrize("kind", ("flat", "cancellation", "translated", "subnormal"))
def test_every_nonempty_node_encloses_every_original_member(extrema, kind):
    points = np.asarray([(float(i-300), float((i*29) % 31-15), float((i*7) % 11-5))
                         for i in range(601)], dtype=np.float64)
    if kind == "flat":
        points[:, 2] = 0.
    elif kind == "cancellation":
        points[:, 1] = -points[:, 0]
    elif kind == "translated":
        points += np.asarray([999_000., -999_000., 512.])
    else:
        points *= float.fromhex("0x1p-1022")
    index = extrema._Index(points, None, extrema.RawExtremaBudgetsV1())
    projections = (AXIS, ((1., 1., .125), (0., 0., 0.), -.25),
                   ((-1., .3, -.7), (12., -17., 23.), .125),
                   ((-0., 0., -0.), (0., 0., 0.), -0.))
    for node in range(1, index.node_count):
        start, end = int(index.start[node]), int(index.end[node])
        if end <= start:
            continue
        rows = index.rows[start:end]
        assert int(index.minimum_raw[node]) == min(int(i) for i in rows)
        for direction, origin, constant in projections:
            lower, upper = index._bounds(node, np.asarray(direction), np.asarray(origin), constant)
            for raw_id in rows:
                value = scalar_projection(points[int(raw_id)], direction, origin, constant)
                assert lower <= value <= upper


def test_permuted_long_plate_axis_prunes_but_diagonal_only_requires_correctness(extrema):
    points = np.asarray([(float(i), float(i % 7-3), float(i % 5-2))
                         for i in range(8192)], dtype=np.float64)
    permutation = np.asarray([(i*4051) % 8192 for i in range(8192)], dtype=np.int64)
    points = points[permutation]
    diagonal = ((.375, -.625, .875), (17., -11., 3.), .125)
    answer = extrema.project_raw_extrema_v1(points, (AXIS, diagonal))
    assert answer["status"] == "COMPLETE", answer
    for record, projection in zip(answer["projections"], (AXIS, diagonal)):
        assert_exact(record, brute(points, projection))
    assert answer["projections"][0]["work"]["query_rows_evaluated"] < len(points)
    assert answer["projections"][0]["work"]["nodes_pruned"] > 0


def test_900000_row_capacity_preserves_population_hashes_and_analytic_axis_witnesses(extrema, monkeypatch):
    count, multiplier = 900_000, 899_999
    assert math.gcd(count, multiplier) == 1
    raw_rows = np.arange(count, dtype=np.int64)
    coordinate_indices = (raw_rows*multiplier) % count
    points = np.empty((count, 3), dtype=np.float64)
    points[:, 0] = coordinate_indices
    points[:, 1] = coordinate_indices % 149
    points[:, 2] = coordinate_indices % 37
    # The modular bijection proves complete population without an extra large
    # sort or a Python per-point loop. Hash all original XYZ and original IDs.
    assert points.shape == (900_000, 3) and points.nbytes == 21_600_000
    point_hash = hashlib.sha256(memoryview(points).cast("B")).hexdigest()
    support_hash = hashlib.sha256(memoryview(raw_rows).cast("B")).hexdigest()
    del coordinate_indices, raw_rows
    projections = (AXIS, ((0., 1., 0.), (0., 0., 0.), 0.),
                   ((0., 0., 1.), (0., 0., 0.), 0.))
    constructor_snapshots = []
    original_query = extrema._Index.query
    def retain_constructor_counters(index, *args):
        # Observation-only seam: preserve the original object/call and capture
        # actual construction counters before any query, without another build.
        if not constructor_snapshots:
            constructor_snapshots.append(dict(index.stats))
        return original_query(index, *args)
    monkeypatch.setattr(extrema._Index, "query", retain_constructor_counters)
    answer = extrema.project_raw_extrema_v1(points, projections)
    assert answer["status"] == "COMPLETE", answer
    assert len(answer["projections"]) == 3
    assert answer["support_scope"] == "WHOLE_RAW_BUFFER"
    assert answer["support_row_count"] == count
    assert answer["point_buffer_sha256"] == point_hash
    assert answer["support_rows_sha256"] == support_hash
    assert hashlib.sha256(memoryview(points).cast("B")).hexdigest() == point_hash
    evaluated = 0
    for axis, (record, maximum, maximum_id) in enumerate(zip(
            answer["projections"], (899_999., 148., 36.), (1, 41, 13))):
        # Independent equality/first-ID oracle on the analytic coordinate array;
        # no staged helper, matrix product, fit, or scalar 900k-row projection.
        minimum_id = int(np.flatnonzero(points[:, axis] == 0.)[0])
        observed_maximum_id = int(np.flatnonzero(points[:, axis] == maximum)[0])
        assert minimum_id == 0 and observed_maximum_id == maximum_id
        assert_exact(record, (0., maximum, minimum_id, maximum_id))
        assert record["support_row_count"] == count
        # Reverse order puts raw0 in the first x-sorted leaf and raw1/13/41
        # in the last. Only those leaves can need the raw-ID tie check; all
        # others have minimum raw ID above the relevant extremum witness.
        assert record["work"]["query_rows_evaluated"] <= 2*256+7
        evaluated += record["work"]["query_rows_evaluated"]
    assert evaluated <= 1557
    stats = answer["statistics"]
    assert stats["queries"] == 3
    assert stats["query_rows_evaluated"] == evaluated
    assert stats["row_visits"] == 3*count+evaluated
    assert stats["index_visits"] == 5*count+evaluated
    assert stats["coordinate_visits"] == 25*count+9*evaluated
    assert stats["sort_work"] == 18_000_000
    assert stats["hash_bytes"] == 32*count
    assert len(constructor_snapshots) == 1
    constructor = constructor_snapshots[0]
    assert constructor["queries"] == 0
    assert constructor["row_visits"] == 3*count
    assert constructor["index_visits"] == 5*count
    assert constructor["sort_work"] == 18_000_000
    assert constructor["query_rows_evaluated"] == 0
    # These are actual helper counters, not a sublinear-construction or measured
    # end-to-end speedup claim: construction itself visits all rows three times.
    summary = dict(schema_version="raw_extrema_index_v1_capacity_test_v1",
        population_rows=count, point_buffer_sha256=point_hash,
        support_rows_sha256=support_hash, reference_full_query_row_visits=3*count,
        constructor_statistics=constructor, statistics=stats,
        query_work=[record["work"] for record in answer["projections"]],
        reported_extrema=[{name: record[name] for name in
            ("interval", "minimum_raw_row", "maximum_raw_row", "support_row_count")}
            for record in answer["projections"]])
    encoded_summary = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert len(encoded_summary.encode("utf-8")) <= 8192
    print("RAW_EXTREMA_INDEX_V1_CAPACITY_JSON=" + encoded_summary)


def test_loose_anticorrelated_boxes_can_require_all_rows_and_still_refuse_at_cap(extrema):
    points = np.asarray([(float(i), -float(i), 0.) for i in range(1024)], dtype=np.float64)
    projection = ((1., 1., 0.), (0., 0., 0.), 0.)
    answer = extrema.project_raw_extrema_v1(points, (projection,))
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], (0., 0., 0, 0))
    # Each non-singleton leaf box contains artificial nonzero corner extrema,
    # although every actual row sums to zero. No universal pruning gain exists.
    assert answer["projections"][0]["work"]["query_rows_evaluated"] >= len(points)
    required = answer["statistics"]["row_visits"]
    below = extrema.project_raw_extrema_v1(points, (projection,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), max_row_visits=required-1))
    assert_refused(below, "BUDGET_ROW_VISITS")
    assert below["statistics"]["row_visits"] == required


def test_equal_extrema_across_leaves_choose_original_smallest_raw_id(extrema):
    points = np.asarray([(float(i-512)/50., -float(i-512)/50., 0.)
                         for i in range(1025)], dtype=np.float64)
    points[7], points[700] = (3., 4., 0.), (6., 1., 0.)
    points[15], points[900] = (-3., -4., 0.), (-6., -1., 0.)
    projection = ((1., 1., 0.), (0., 0., 0.), 0.)
    expected = brute(points, projection)
    assert expected == (-7., 7., 15, 7)
    answer = extrema.project_raw_extrema_v1(points, (projection,))
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], expected)


@pytest.mark.parametrize("sign", (-1., 1.))
def test_only_one_true_extremum_can_improve_the_axis_seed_set(extrema, sign):
    points = np.zeros((515, 3), dtype=np.float64)
    points[1:5] = [(-2., 0., 0.), (2., 0., 0.), (0., -2., 0.), (0., 2., 0.)]
    points[514] = (1.5, 1.5, 0.)
    projection = ((sign, sign, 0.), (0., 0., 0.), 0.)
    index = extrema._Index(points, None, extrema.RawExtremaBudgetsV1())
    seeded = brute(points, projection, index.seed_rows)
    expected = brute(points, projection)
    assert (seeded[0] == expected[0]) != (seeded[1] == expected[1])
    record = index.query(*projection)
    assert_exact(record, expected)
    assert record["work"]["leaves_evaluated"] > 0


def test_support_is_complete_and_duplicate_positions_keep_distinct_raw_ids(extrema):
    points = small_points()
    support = np.asarray([1, 2, 3, 4], dtype=np.int64)
    projection = ((1., -1., 0.), (0., 0., 0.), 0.)
    answer = extrema.project_raw_extrema_v1(points, (projection,), support_rows=support)
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], brute(points, projection, support))
    assert answer["projections"][0]["maximum_raw_row"] == 1
    assert answer["support_row_count"] == 4
    assert answer["support_scope"] == "COMPLETE_SUPPLIED_RAW_SUPPORT"
    assert answer["point_buffer_sha256"] == hashlib.sha256(points.tobytes(order="C")).hexdigest()
    assert answer["support_rows_sha256"] == hashlib.sha256(support.tobytes(order="C")).hexdigest()


@pytest.mark.parametrize("layout", ("readonly", "fortran", "strided"))
def test_array_layout_does_not_change_owned_coordinate_meaning(extrema, layout):
    points = small_points()
    if layout == "readonly":
        points.flags.writeable = False
    elif layout == "fortran":
        points = np.asfortranarray(points)
    else:
        points = np.repeat(points, 2, axis=0)[::2]
    before = points.tobytes(order="C")
    answer = extrema.project_raw_extrema_v1(points, (AXIS,))
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], brute(points, AXIS))
    assert points.tobytes(order="C") == before


def test_caller_coordinate_and_support_mutation_after_build_cannot_change_query(extrema, monkeypatch):
    points, support = small_points(), np.asarray([0, 2, 4], dtype=np.int64)
    expected = brute(points, AXIS, support)
    point_hash = hashlib.sha256(points.tobytes()).hexdigest()
    support_hash = hashlib.sha256(support.tobytes()).hexdigest()
    original = extrema._Index.query
    def mutate_then_query(index, *args):
        points[:] = 999.
        support[:] = 1
        assert index.points.flags.writeable is False
        assert index.rows.flags.writeable is False
        return original(index, *args)
    monkeypatch.setattr(extrema._Index, "query", mutate_then_query)
    answer = extrema.project_raw_extrema_v1(points, (AXIS,), support_rows=support)
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], expected)
    assert answer["point_buffer_sha256"] == point_hash
    assert answer["support_rows_sha256"] == support_hash


def test_shape_snapshot_does_not_reread_mutated_caller_shape(extrema, monkeypatch):
    points = np.arange(18, dtype=np.float64).reshape(6, 3)
    expected = brute(points.copy(), AXIS)
    original = extrema.np.frombuffer
    calls = []
    def mutate_shape(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            points.shape = (3, 6)
        return original(*args, **kwargs)
    monkeypatch.setattr(extrema.np, "frombuffer", mutate_shape)
    answer = extrema.project_raw_extrema_v1(points, (AXIS,))
    assert answer["status"] == "COMPLETE", answer
    assert_exact(answer["projections"][0], expected)
    assert answer["support_row_count"] == 6


@pytest.mark.parametrize("which", ("raw", "support"))
def test_resize_after_metadata_capture_cannot_bypass_declared_row_cap(extrema, monkeypatch, which):
    points = np.arange(18, dtype=np.float64).reshape(6, 3).copy()
    support = np.asarray([0, 2, 4], dtype=np.int64)
    original = extrema.np.dtype
    touched = []
    def resize_at_dtype_check(value, *args, **kwargs):
        target = "float64" if which == "raw" else "int64"
        if type(value) is str and value == target and not touched:
            touched.append(True)
            if which == "raw":
                points.resize((9, 3), refcheck=False)
            else:
                support.resize((4,), refcheck=False)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(extrema.np, "dtype", resize_at_dtype_check)
    answer = extrema.project_raw_extrema_v1(points, (AXIS,), support_rows=support,
        budgets=replace(extrema.RawExtremaBudgetsV1(), max_raw_rows=6, max_support_rows=3))
    assert touched == [True]
    assert_refused(answer, "CHANGED_RAW_SNAPSHOT_BYTE_COUNT" if which == "raw"
                   else "CHANGED_SUPPORT_SNAPSHOT_BYTE_COUNT")
    assert answer["statistics"]["queries"] == 0


@pytest.mark.parametrize("kind", ("list", "subclass", "float32", "integer", "empty",
                                  "wrong_shape", "nan", "infinity", "out_of_range"))
def test_invalid_points_clear_everything(extrema, kind):
    points = small_points()
    if kind == "list": points = points.tolist()
    elif kind == "subclass": points = points.view(type("DerivedArray", (np.ndarray,), {}))
    elif kind == "float32": points = points.astype(np.float32)
    elif kind == "integer": points = points.astype(np.int64)
    elif kind == "empty": points = np.empty((0, 3), dtype=np.float64)
    elif kind == "wrong_shape": points = points.reshape(3, 5)
    elif kind == "nan": points[0, 0] = float("nan")
    elif kind == "infinity": points[0, 0] = float("inf")
    else: points[0, 0] = 1_000_000.0000000001
    assert_refused(extrema.project_raw_extrema_v1(points, (AXIS,)))


@pytest.mark.parametrize("kind", ("list", "float", "empty", "negative", "past_end",
                                  "duplicate", "decreasing", "matrix"))
def test_invalid_support_identities_are_not_sorted_or_welded(extrema, kind):
    values = {"list": [0, 1], "float": np.asarray([0., 1.]),
              "empty": np.asarray([], dtype=np.int64),
              "negative": np.asarray([-1, 1]), "past_end": np.asarray([0, 5]),
              "duplicate": np.asarray([1, 1]), "decreasing": np.asarray([2, 1]),
              "matrix": np.asarray([[0, 1]])}
    assert_refused(extrema.project_raw_extrema_v1(small_points(), (AXIS,), support_rows=values[kind]))


@pytest.mark.parametrize("kind", ("outer_list", "empty", "record_list", "short_record"))
def test_projection_container_admission(extrema, kind):
    values = {"outer_list": [AXIS], "empty": (), "record_list": (list(AXIS),),
              "short_record": ((AXIS[0], AXIS[1]),)}
    assert_refused(extrema.project_raw_extrema_v1(small_points(), values[kind]))


@pytest.mark.parametrize("bad", ([1., 0., 0.], (True, 0., 0.), (float("nan"), 0., 0.),
                                 (float("inf"), 0., 0.), (1_000_001., 0., 0.), (1., 0.)))
def test_direction_and_origin_strict_scalar_admission(extrema, bad):
    for position in (0, 1):
        projection = list(AXIS)
        projection[position] = bad
        answer = extrema.project_raw_extrema_v1(small_points(), (tuple(projection),))
        assert_refused(answer, "INVALID_PROJECTION_VECTOR")


@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), 1_000_001.))
def test_constant_strict_scalar_admission(extrema, bad):
    assert_refused(extrema.project_raw_extrema_v1(small_points(), ((AXIS[0], AXIS[1], bad),)),
                   "INVALID_PROJECTION_CONSTANT")


def test_typed_budget_and_every_scalar_limit_are_strict(extrema):
    defaults = extrema.RawExtremaBudgetsV1()
    for wrong in (None, {}, object()):
        assert_refused(extrema.project_raw_extrema_v1(small_points(), (AXIS,), budgets=wrong),
                       "INVALID_BUDGET_TYPE")
    for field in fields(defaults):
        for wrong in (True, -1, float(getattr(defaults, field.name)), getattr(defaults, field.name)+1):
            answer = extrema.project_raw_extrema_v1(small_points(), (AXIS,),
                budgets=replace(defaults, **{field.name: wrong}))
            assert_refused(answer)


@pytest.mark.parametrize("field,counter", (
    ("max_index_visits", "index_visits"), ("max_row_visits", "row_visits"),
    ("max_coordinate_visits", "coordinate_visits"), ("max_sort_work", "sort_work"),
    ("max_node_visits", "node_visits"), ("max_hash_bytes", "hash_bytes"),
    ("max_working_bytes", "accounted_working_peak_bytes"),
))
def test_actual_resource_requirement_exact_limit_and_one_below(extrema, field, counter):
    points = small_points()
    baseline = extrema.project_raw_extrema_v1(points, (AXIS,))
    assert baseline["status"] == "COMPLETE", baseline
    required = baseline["statistics"][counter]
    assert required > 0
    exact = extrema.project_raw_extrema_v1(points, (AXIS,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), **{field: required}))
    assert exact["status"] == "COMPLETE", exact
    assert_exact(exact["projections"][0], brute(points, AXIS))
    below = extrema.project_raw_extrema_v1(points, (AXIS,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), **{field: required-1}))
    assert_refused(below)
    assert below["statistics"][counter] >= required


def test_raw_support_query_cardinality_boundaries(extrema):
    points, support = small_points(), np.asarray([1, 3, 4], dtype=np.int64)
    exact = replace(extrema.RawExtremaBudgetsV1(), max_raw_rows=5, max_support_rows=3, max_queries=2)
    answer = extrema.project_raw_extrema_v1(points, (AXIS, AXIS), support_rows=support, budgets=exact)
    assert answer["status"] == "COMPLETE", answer
    for field, value in (("max_raw_rows", 4), ("max_support_rows", 2), ("max_queries", 1)):
        assert_refused(extrema.project_raw_extrema_v1(points, (AXIS, AXIS), support_rows=support,
            budgets=replace(exact, **{field: value})))


def test_constructor_literal_work_counts_are_not_query_only(extrema):
    points = np.asarray([(float(i), 0., 0.) for i in range(5)], dtype=np.float64)
    index = extrema._Index(points, None, extrema.RawExtremaBudgetsV1())
    assert index.stats["queries"] == 0
    assert index.stats["row_visits"] == 15       # raw validation + root + leaves
    assert index.stats["index_visits"] == 25    # support + root + sort2N + leaves
    assert index.stats["coordinate_visits"] == 125  # 6N + 12N + N + 6N
    assert index.stats["sort_work"] == 15       # N * ceil(log2 N)
    assert index.stats["hash_bytes"] == 160     # 5 * (XYZ float64 + raw-ID int64)
    assert index.stats["node_visits"] == 0
    assert index.stats["query_rows_evaluated"] == 0


@pytest.mark.parametrize("field,counter,attempted", (
    ("max_row_visits", "row_visits", 5),
    ("max_coordinate_visits", "coordinate_visits", 30),
    ("max_hash_bytes", "hash_bytes", 120),
    ("max_index_visits", "index_visits", 5),
    ("max_sort_work", "sort_work", 15),
    ("max_working_bytes", "accounted_working_peak_bytes", 65_536),
))
def test_constructor_refusal_preserves_precharged_attempted_work(extrema, field, counter, attempted):
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), **{field: 0}))
    assert_refused(answer)
    assert answer["statistics"][counter] == attempted
    assert answer["statistics"]["queries"] == 0


def test_fixed_memory_preflight_precedes_arithmetic_sentinel_calls(extrema, monkeypatch):
    called = []
    def forbidden():
        called.append(True)
        raise AssertionError("sentinel must not run before fixed reservation")
    monkeypatch.setattr(extrema, "_arithmetic_sentinels", forbidden)
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), max_working_bytes=65_535))
    assert_refused(answer, "WORKING_PAYLOAD_BUDGET")
    assert called == []
    assert answer["statistics"]["accounted_working_peak_bytes"] == 65_536
    assert answer["statistics"]["row_visits"] == 0


@pytest.mark.parametrize("phase", ("seed", "leaf"))
def test_failed_evaluation_is_attempted_work_not_completed_rows_or_leaves(extrema, monkeypatch, phase):
    original = extrema._staged
    evaluations = []
    def fail_selected_batch(points, *args):
        if points.ndim == 2:
            evaluations.append(len(points))
            if len(evaluations) == (1 if phase == "seed" else 2):
                raise ValueError("invented failed numerical batch")
        return original(points, *args)
    monkeypatch.setattr(extrema, "_staged", fail_selected_batch)
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS,))
    assert_refused(answer, "EXTREMA_HELPER_EXCEPTION")
    assert evaluations == ([3] if phase == "seed" else [3, 5])
    assert answer["statistics"]["query_rows_evaluated"] == (0 if phase == "seed" else 3)
    assert answer["statistics"]["leaves_evaluated"] == 0
    assert answer["statistics"]["row_visits"] == (18 if phase == "seed" else 23)
    assert answer["statistics"]["queries"] == 1


def test_report_reservation_exact_and_one_byte_below(extrema):
    baseline = extrema.project_raw_extrema_v1(small_points(), (AXIS,))
    assert baseline["status"] == "COMPLETE", baseline
    required = baseline["statistics"]["report_bytes"] + 8192
    exact = extrema.project_raw_extrema_v1(small_points(), (AXIS,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), max_report_bytes=required))
    assert exact["status"] == "COMPLETE", exact
    below = extrema.project_raw_extrema_v1(small_points(), (AXIS,),
        budgets=replace(extrema.RawExtremaBudgetsV1(), max_report_bytes=required-1))
    assert_refused(below, "BUDGET_REPORT_ENVELOPE_BYTES")
    assert below["statistics"]["report_bytes"] == baseline["statistics"]["report_bytes"]


def test_invalid_late_projection_discards_valid_prefix(extrema):
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS, ((True, 0., 0.), AXIS[1], 0.)))
    assert_refused(answer, "INVALID_PROJECTION_VECTOR")
    assert answer["discarded_projection_count"] == 1
    assert answer["statistics"]["queries"] == 1
    assert answer["statistics"]["report_bytes"] > 0


def test_query_failure_invalidates_reusable_index(extrema):
    index = extrema._Index(small_points(), None, replace(extrema.RawExtremaBudgetsV1(), max_queries=1))
    assert_exact(index.query(*AXIS), brute(small_points(), AXIS))
    with pytest.raises(ValueError, match="BUDGET_QUERIES"):
        index.query(*AXIS)
    assert index.stats["queries"] == 2
    assert index.valid is False
    with pytest.raises(ValueError, match="INDEX_INVALID_OR_PREVIOUSLY_FAILED"):
        index.query(*AXIS)


def test_unexpected_late_exception_clears_all_output_and_retains_work(extrema, monkeypatch):
    original = extrema._Index._bounds
    def late_failure(index, *args):
        if index.stats["queries"] == 2:
            raise RuntimeError("invented late node failure")
        return original(index, *args)
    monkeypatch.setattr(extrema._Index, "_bounds", late_failure)
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS, AXIS))
    assert_refused(answer, "EXTREMA_HELPER_EXCEPTION")
    assert answer["discarded_projection_count"] == 1
    assert answer["statistics"]["queries"] == 2
    assert answer["statistics"]["query_rows_evaluated"] > 0


def test_final_serialization_failure_cannot_leave_complete_shell(extrema, monkeypatch):
    original = extrema.json.JSONEncoder.iterencode
    def fail_final(encoder, value, *args, **kwargs):
        if type(value) is dict and value.get("schema_version") == "raw_extrema_index_v1":
            yield "{"
            raise ValueError("invented final encoder failure")
        yield from original(encoder, value, *args, **kwargs)
    monkeypatch.setattr(extrema.json.JSONEncoder, "iterencode", fail_final)
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS,))
    assert_refused(answer, "EXTREMA_HELPER_EXCEPTION")
    assert answer["discarded_projection_count"] == 1
    assert answer["statistics"]["report_bytes"] > 0


def test_budget_interrupt_is_not_converted_to_ordinary_refusal(extrema, monkeypatch):
    class InventedInterrupt(BaseException):
        pass
    captured = []
    def interrupt(index, *args):
        captured.append(index)
        raise InventedInterrupt()
    monkeypatch.setattr(extrema._Index, "_bounds", interrupt)
    with pytest.raises(InventedInterrupt):
        extrema.project_raw_extrema_v1(small_points(), (AXIS,))
    assert len(captured) == 1
    assert captured[0].valid is False


def test_arithmetic_sentinel_failure_has_an_attempted_constructor_ledger(extrema, monkeypatch):
    def unavailable():
        raise extrema._Unavailable("ARITHMETIC_SENTINEL_REFUSAL")
    monkeypatch.setattr(extrema, "_arithmetic_sentinels", unavailable)
    answer = extrema.project_raw_extrema_v1(small_points(), (AXIS,))
    assert_refused(answer, "ARITHMETIC_SENTINEL_REFUSAL")
    assert answer["statistics"]["row_visits"] == 0
    assert answer["statistics"]["index_visits"] == 0
    assert answer["statistics"]["queries"] == 0
