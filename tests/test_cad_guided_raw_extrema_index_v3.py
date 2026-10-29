"""Invented-only tests for the kd-partition raw-extrema index V3.

V1 is the frozen reference implementation of the same staged program; every
V3 interval/witness must be byte-identical to V1's, and both must equal an
independent scalar/blocked oracle. Efficacy assertions on the invented folded
sheet population are development evidence for this synthetic input only, not
physical query directions, calibrated metrology or a universal speedup.
"""
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
AXIS = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
QUERIES = (
    ((1., 1., -1.), (0., 0., 0.), 0.),
    ((-1., -1., 1.), (0., 0., 0.), 0.),
    ((2., 2., -2.), (0., 0., 0.), 0.),
    ((.5, .5, -.5), (0., 0., 0.), 0.),
    ((1., 0., 0.), (0., 0., 0.), 0.),
    ((0., 1., 0.), (0., 0., 0.), 0.),
    ((0., 0., 1.), (0., 0., 0.), 0.),
    ((1., 1., -1.), (16., -8., 4.), .125),
    ((1., 1., -1.), (-16., 8., -4.), -.125),
    ((1., 1., -0.999755859375), (0., 0., 0.), 0.),
    ((-1., -1., 1.000244140625), (0., 0., 0.), 0.),
    ((.375, -.625, .875), (1., -2., 3.), .125),
)


@pytest.fixture(scope="module")
def modules():
    names = ("cad_guided_raw_extrema_index_v1", "cad_guided_raw_extrema_index_v3")
    package = ModuleType("_invented_extrema_v3_contract")
    package.__path__ = []
    sys.modules[package.__name__] = package
    loaded = []
    try:
        for short in names:
            name = package.__name__ + "." + short
            assert name not in sys.modules
            spec = importlib.util.spec_from_file_location(name, ROOT / "backend" / (short + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        yield tuple(loaded)
    finally:
        for key in tuple(sys.modules):
            if key == package.__name__ or key.startswith(package.__name__ + "."):
                sys.modules.pop(key, None)


@pytest.fixture(scope="module")
def v1(modules):
    return modules[0]


@pytest.fixture(scope="module")
def v3(modules):
    return modules[1]


def bits(value):
    return struct.pack("!d", float(value))


def scalar(row, query):
    direction, origin, constant = query
    x = float(row[0]) - float(origin[0])
    y = float(row[1]) - float(origin[1])
    z = float(row[2]) - float(origin[2])
    tx, ty, tz = x*float(direction[0]), y*float(direction[1]), z*float(direction[2])
    first = tx + ty
    second = first + tz
    answer = second + float(constant)
    return 0.0 if answer == 0.0 else answer


def brute(points, query, rows=None):
    ids = range(len(points)) if rows is None else [int(x) for x in rows]
    values = [(scalar(points[i], query), i) for i in ids]
    low = min(values, key=lambda pair: (pair[0], pair[1]))
    high = max(values, key=lambda pair: (pair[0], -pair[1]))
    return low[0], high[0], low[1], high[1]


def blocked_oracle(points, query):
    direction, origin, constant = query
    low, high, low_id, high_id = math.inf, -math.inf, None, None
    for start in range(0, len(points), 4096):
        delta = np.subtract(points[start:start+4096], np.asarray(origin))
        products = np.multiply(delta, np.asarray(direction))
        first = np.add(products[:, 0], products[:, 1])
        second = np.add(first, products[:, 2])
        values = np.add(second, constant)
        values = np.where(values == 0., 0., values)
        assert np.isfinite(values).all()
        a, b = float(np.min(values)), float(np.max(values))
        ai = start + int(np.flatnonzero(values == a)[0])
        bi = start + int(np.flatnonzero(values == b)[0])
        if a < low or (a == low and (low_id is None or ai < low_id)):
            low, low_id = a, ai
        if b > high or (b == high and (high_id is None or bi < high_id)):
            high, high_id = b, bi
    return low, high, low_id, high_id


def exact(record, expected):
    a, b, ai, bi = expected
    assert bits(record["interval"][0]) == bits(a)
    assert bits(record["interval"][1]) == bits(b)
    assert type(record["minimum_raw_row"]) is int and record["minimum_raw_row"] == ai
    assert type(record["maximum_raw_row"]) is int and record["maximum_raw_row"] == bi


def same_record(a, b):
    assert bits(a["interval"][0]) == bits(b["interval"][0])
    assert bits(a["interval"][1]) == bits(b["interval"][1])
    assert a["minimum_raw_row"] == b["minimum_raw_row"]
    assert a["maximum_raw_row"] == b["maximum_raw_row"]
    assert a["support_row_count"] == b["support_row_count"]


def refused(answer, reason=None):
    assert answer["status"] == "UNAVAILABLE" and answer["projections"] == []
    assert answer["accepted_registration"] is False
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    for name in ("point_buffer_sha256", "support_rows_sha256", "support_scope"):
        assert name not in answer
    if reason is not None:
        assert answer["reasons"] == [reason]


def small_points():
    return np.asarray([[-3., 1., 0.], [2., -4., 1.], [1., 2., -3.],
                       [0., 0., 0.], [2., -4., 1.]], dtype=np.float64)


def population(kind, count=32768):
    """The four consumed V2 discriminator populations, regenerated verbatim."""
    assert count == 32768
    raw = np.arange(count, dtype=np.int64)
    assert math.gcd(4051, count) == 1
    k = (raw * 4051) % count
    points = np.empty((count, 3), dtype=np.float64)
    if kind == "folded_three_planes":
        panel, t = k % 3, k // 3
        u, v = ((t % 128)-63.5)/8., ((t // 128)-42.)/8.
        points[:] = 0.
        points[panel == 0, 0], points[panel == 0, 1] = u[panel == 0], v[panel == 0]
        points[panel == 1, 0], points[panel == 1, 2] = u[panel == 1], v[panel == 1]
        points[panel == 2, 1], points[panel == 2, 2] = u[panel == 2], v[panel == 2]
        cx, sx, cz, sz = 5./13., 12./13., 3./5., 4./5.
        rx_y = np.subtract(np.multiply(cx, points[:, 1]), np.multiply(sx, points[:, 2]))
        rx_z = np.add(np.multiply(sx, points[:, 1]), np.multiply(cx, points[:, 2]))
        rz_x = np.subtract(np.multiply(cz, points[:, 0]), np.multiply(sz, rx_y))
        rz_y = np.add(np.multiply(sz, points[:, 0]), np.multiply(cz, rx_y))
        points[:, 0], points[:, 1], points[:, 2] = rz_x, rz_y, rx_z
    else:
        i, j = k // 128, k % 128
        points[:, 0], points[:, 1] = (i-127.5)/8., (j-63.5)/8.
        points[:, 2] = np.add(points[:, 0], points[:, 1])
        if kind == "resolved_unique_extrema":
            thickness = (((17*i+13*j) % 5)-2)/1024.
            thickness[k == 0], thickness[k == count-1] = -4., 4.
            points[:, 2] = np.add(points[:, 2], thickness)
        elif kind == "repeated_plateau":
            thickness = np.where(i < 64, -1., np.where(i >= 192, 1., 0.))
            points[:, 2] = np.add(points[:, 2], thickness)
        else:
            assert kind == "exact_flat"
    assert points.shape == (32768, 3)
    assert np.isfinite(points).all() and np.all(np.abs(points) <= 1e6)
    return points


def folded_sheet(count):
    """Invented bent-sheet scan: base panel, three bent flanges, arcs, noise, oblique pose.

    Row order is a frozen modular bijection so raw IDs are spatially scrambled
    like an exported scan. Not a physical part, calibration or a Gold case.
    """
    raw = np.arange(count, dtype=np.int64)
    multiplier = 1_000_003
    assert math.gcd(multiplier, count) == 1
    k = (raw*multiplier) % count
    region = k % 7          # 0-2 base panel (3/7 of rows), 3,4,5 flanges, 6 arcs
    t = k // 7
    u = ((t*7919) % 10007)/10007.   # deterministic quasi-random in [0,1)
    v = ((t*104729) % 10009)/10009.
    x = np.empty(count); y = np.empty(count); z = np.empty(count)
    base = region <= 2
    x[base], y[base], z[base] = 300.*u[base]-150., 200.*v[base]-100., 0.
    f = region == 3   # flange bent 90 degrees up along y=+100 edge
    x[f], y[f], z[f] = 300.*u[f]-150., 100.+3., 3.+40.*v[f]
    f = region == 4   # flange bent 90 degrees down along y=-100 edge
    x[f], y[f], z[f] = 300.*u[f]-150., -100.-3., -3.-35.*v[f]
    f = region == 5   # flange bent 45 degrees along x=+150 edge
    s = 50.*v[f]
    x[f], y[f], z[f] = 150.+2.+s*math.cos(math.pi/4.), 200.*u[f]-100., 2.+s*math.sin(math.pi/4.)
    a = region == 6   # bend arcs, radius 3 mm, spread over the three bend lines
    which = (t[a] % 3)
    theta = (math.pi/2.)*u[a]
    ax = np.where(which == 2, 150., 300.*u[a]-150.)
    ay = np.where(which == 0, 100.+3.*np.sin(theta), np.where(which == 1, -100.-3.*np.sin(theta), 200.*v[a]-100.))
    az = np.where(which == 0, 3.-3.*np.cos(theta), np.where(which == 1, -3.+3.*np.cos(theta), 3.-3.*np.cos(theta*.5)))
    ax = np.where(which == 2, 150.+3.*np.sin(theta*.5), ax)
    x[a], y[a], z[a] = ax, ay, az
    noise = (((k*2654435761) % 1000)/1000.-.5)*.04   # +-0.02 mm scanner-like noise
    z = z+noise
    points = np.column_stack((x, y, z))
    # Fixed oblique proper rotation (Rz 3/5,4/5 composed with Rx 5/13,12/13) and offset.
    cx, sx, cz, sz = 5./13., 12./13., 3./5., 4./5.
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    rotation = rz@rx
    points = points@rotation.T+np.array([412.5, -37.25, 88.125])
    points = np.ascontiguousarray(points, dtype=np.float64)
    assert np.isfinite(points).all() and np.all(np.abs(points) <= 1e6)
    return points, rotation


def bend_axis_queries(rotation):
    """Whole-part-context-like queries: bend-line directions with varied constants."""
    axes = (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 1., 0.]))
    queries = []
    for axis in axes[:2]:
        world = rotation@axis
        for sign in (1., -1.):
            direction = tuple(float(sign*x) for x in world)
            for constant in (0., 12.5, -7.25, 101.75):
                queries.append((direction, (0., 0., 0.), constant))
    # A few normal/oblique directions (worst cases for flat regions).
    for direction in ((0., 0., 1.), (.5, .5, .7071067811865476), (-.3, .8, .1)):
        world = rotation@np.array(direction)
        queries.append((tuple(float(x) for x in world), (1., -2., 3.), .5))
    return tuple(queries)


ROUNDING_CASES = (
    ([(0., 0., 0.), (1., 2., 3.), (-2., 4., -1.)], AXIS),
    ([(0., 0., 0.), (1., 2., 3.), (-2., 4., -1.)], ((-2., 3., -4.), (1., -2., 3.), 7.)),
    ([(1., 1., 1.), (1., 1., -1.)], ((1., float.fromhex("0x1p-53"), -1.), (0., 0., 0.), 0.)),
    ([(1., 1., 1.), (1., 1., -1.)], ((1., -1., float.fromhex("0x1p-53")), (0., 0., 0.), 0.)),
    ([(1_000_000., 1_000_000., 1.), (999_999., 999_999., -1.)], ((1_000_000., -1_000_000., 1.), (0., 0., 0.), 1.)),
    ([(1_000_000., -1_000_000., 0.), (999_999.9999999999, -1_000_000., 0.)],
     ((1., 1., -0.), (1_000_000., -1_000_000., 0.), -0.)),
    ([(float.fromhex("0x0.0000000000001p-1022"), 0., 0.), (float.fromhex("0x0.0000000000002p-1022"), 0., 0.)],
     ((1., 0., 0.), (0., 0., 0.), 0.)),
    ([(float.fromhex("0x1p-1022"), 0., 0.), (-float.fromhex("0x1p-1022"), 0., 0.)], ((.5, 0., 0.), (0., 0., 0.), 0.)),
    ([(-0., 0., -0.), (0., -0., 0.)], ((-0., -1., 0.), (-0., 0., -0.), -0.)),
    ([(1., -1., 0.), (-1., 1., 0.)], ((1., 1., 0.), (0., 0., 0.), -0.)),
    ([(1., 2., 3.), (-3., -2., -1.)], ((0., 0., 0.), (17., -19., 23.), 1_000_000.)),
    ([(.1, .2, .3), (.3, .2, .1), (-.3, -.2, -.1)], ((.7, -.9, .11), (.01, -.02, .03), -.13)),
)


@pytest.mark.parametrize("coordinates,projection", ROUNDING_CASES)
def test_staged_scalar_rounding_exact_and_identical_to_v1(v1, v3, coordinates, projection):
    points = np.asarray(coordinates, dtype=np.float64)
    answer = v3.project_raw_extrema_v3(points, (projection,))
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], brute(points, projection))
    reference = v1.project_raw_extrema_v1(points, (projection,))
    same_record(answer["projections"][0], reference["projections"][0])
    assert answer["arithmetic"] == v1.ARITHMETIC == "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1"
    assert answer["partition"] == "RANK_PARTITION_KD_TREE_GLOBAL_HALVED_ROOT_SPAN_AXIS_SCHEDULE_STABLE_SORT_LEAF64_V3"
    assert answer["bound_scope"] == "ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL"
    assert answer["accepted_registration"] is False
    assert answer["observed_bend_count"] is None
    assert answer["production_use_allowed"] is False


def test_staged_function_is_textually_shared_with_v1(v1, v3):
    import inspect
    for name in ("_staged", "_bounds", "_evaluate", "query", "_arithmetic_sentinels"):
        a = getattr(v1._Index, name, None) or getattr(v1, name)
        b = getattr(v3._Index, name, None) or getattr(v3, name)
        assert inspect.getsource(a) == inspect.getsource(b), name


@pytest.mark.parametrize("kind", ("flat", "cancellation", "translated", "subnormal", "oblique"))
def test_every_nonempty_node_encloses_every_original_member(v3, kind):
    points = np.asarray([(float(i-1300), float((i*29) % 31-15), float((i*7) % 11-5))
                         for i in range(2601)], dtype=np.float64)
    if kind == "flat":
        points[:, 2] = 0.
    elif kind == "cancellation":
        points[:, 1] = -points[:, 0]
    elif kind == "translated":
        points += np.asarray([998_000., -998_000., 512.])
    elif kind == "subnormal":
        points *= float.fromhex("0x1p-1022")
    else:
        points[:, 2] = points[:, 0]*.3+points[:, 1]*.7
    index = v3._Index(points, None, v3.RawExtremaBudgetsV3())
    assert index.depth == 6 and index.base == 64 and index.node_count == 128
    projections = (AXIS, ((1., 1., .125), (0., 0., 0.), -.25),
                   ((-1., .3, -.7), (12., -17., 23.), .125),
                   ((-0., 0., -0.), (0., 0., 0.), -0.))
    seen = np.zeros(len(points), dtype=np.int64)
    for node in range(1, index.node_count):
        start, end = int(index.start[node]), int(index.end[node])
        if end <= start:
            continue
        rows = index.rows[start:end]
        assert int(index.minimum_raw[node]) == min(int(i) for i in rows)
        assert (start, end) == index._row_range(node)
        if node >= index.base:
            seen[rows] += 1
            assert end-start <= v3.LEAF_ROWS
            assert np.array_equal(index.low[node], np.min(points[rows], axis=0))
            assert np.array_equal(index.high[node], np.max(points[rows], axis=0))
        for direction, origin, constant in projections:
            lower, upper = index._bounds(node, np.asarray(direction), np.asarray(origin), constant)
            for raw_id in rows:
                value = scalar(points[int(raw_id)], (direction, origin, constant))
                assert lower <= value <= upper
    assert np.all(seen == 1)  # every original row lives in exactly one leaf
    for projection in projections:
        exact(index.query(*projection), brute(points, projection))


def test_permutation_is_a_bijection_and_schedule_is_deterministic(v3):
    points, _ = folded_sheet(40_000)
    a = v3._Index(points, None, v3.RawExtremaBudgetsV3())
    b = v3._Index(points, None, v3.RawExtremaBudgetsV3())
    assert np.array_equal(a.rows, b.rows)
    assert a.axis_schedule == b.axis_schedule and len(a.axis_schedule) == a.depth
    assert np.array_equal(np.sort(a.rows), np.arange(len(points)))
    assert a.rows.flags.writeable is False and a.points.flags.writeable is False
    support = np.arange(0, len(points), 3, dtype=np.int64)
    c = v3._Index(points, support, v3.RawExtremaBudgetsV3())
    assert np.array_equal(np.sort(c.rows), support)
    # A node's first-level split always separates rows by the scheduled axis.
    axis = a.axis_schedule[0]
    left_end = a._row_range(2)[1]
    assert np.max(points[a.rows[:left_end], axis]) <= np.min(points[a.rows[left_end:], axis])


def test_axis_schedule_prefers_largest_remaining_halved_span(v3):
    assert v3._axis_schedule((300., 200., 50.), 12) == (0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1)
    assert v3._axis_schedule((1., 1., 1.), 4) == (0, 1, 2, 0)
    assert v3._axis_schedule((0., 0., 0.), 3) == (0, 0, 0)  # all-zero spans tie to the lowest axis
    assert v3._axis_schedule((0., 5., 0.), 2) == (1, 1)


@pytest.mark.parametrize("kind", ("exact_flat", "resolved_unique_extrema", "repeated_plateau", "folded_three_planes"))
def test_consumed_populations_identical_to_v1_and_oracle(v1, v3, kind):
    points = population(kind)
    reference = v1.project_raw_extrema_v1(points, QUERIES)
    answer = v3.project_raw_extrema_v3(points, QUERIES)
    assert reference["status"] == answer["status"] == "COMPLETE", answer.get("reasons")
    assert answer["point_buffer_sha256"] == reference["point_buffer_sha256"]
    assert answer["support_rows_sha256"] == reference["support_rows_sha256"]
    for a, b, query in zip(answer["projections"], reference["projections"], QUERIES):
        same_record(a, b)
        exact(a, blocked_oracle(points, query))
    work = dict(kind=kind, v1_query_rows=reference["statistics"]["query_rows_evaluated"],
        v3_query_rows=answer["statistics"]["query_rows_evaluated"],
        v1_coordinates=reference["statistics"]["coordinate_visits"],
        v3_coordinates=answer["statistics"]["coordinate_visits"],
        v1_sort_work=reference["statistics"]["sort_work"], v3_sort_work=answer["statistics"]["sort_work"])
    print("RAW_EXTREMA_V3_CONSUMED_POPULATION_WORK_JSON=" + json.dumps(work, sort_keys=True))


def test_folded_sheet_identical_to_v1_with_far_fewer_evaluated_rows(v1, v3):
    count = 200_000
    points, rotation = folded_sheet(count)
    import time
    queries = bend_axis_queries(rotation)
    assert len(queries) == 19
    started = time.process_time()
    reference = v1.project_raw_extrema_v1(points, queries)
    cpu_v1 = time.process_time()-started
    started = time.process_time()
    answer = v3.project_raw_extrema_v3(points, queries)
    cpu_v3 = time.process_time()-started
    assert reference["status"] == answer["status"] == "COMPLETE", answer.get("reasons")
    for a, b, query in zip(answer["projections"], reference["projections"], queries):
        same_record(a, b)
        exact(a, blocked_oracle(points, query))
    v1_rows = sum(record["work"]["query_rows_evaluated"] for record in reference["projections"][:16])
    v3_rows = sum(record["work"]["query_rows_evaluated"] for record in answer["projections"][:16])
    stats_v1, stats_v3 = reference["statistics"], answer["statistics"]
    work = dict(rows=count, queries=len(queries), bend_axis_queries=16,
        v1_bend_axis_rows=v1_rows, v3_bend_axis_rows=v3_rows,
        v1_total_query_rows=stats_v1["query_rows_evaluated"], v3_total_query_rows=stats_v3["query_rows_evaluated"],
        v1_coordinates=stats_v1["coordinate_visits"], v3_coordinates=stats_v3["coordinate_visits"],
        v1_sort_work=stats_v1["sort_work"], v3_sort_work=stats_v3["sort_work"],
        v3_depth=answer["tree_depth"], v3_axis_schedule=answer["axis_schedule"])
    work.update(v1_index_visits=stats_v1["index_visits"], v3_index_visits=stats_v3["index_visits"],
        v1_build_query_cpu=round(cpu_v1, 4), v3_build_query_cpu=round(cpu_v3, 4),
        v1_common_work=stats_v1["coordinate_visits"]+stats_v1["sort_work"]+stats_v1["index_visits"],
        v3_common_work=stats_v3["coordinate_visits"]+stats_v3["sort_work"]+stats_v3["index_visits"])
    print("RAW_EXTREMA_V3_FOLDED_SHEET_WORK_JSON=" + json.dumps(work, sort_keys=True))
    # These are coordinate/row-evaluation observations on this invented input,
    # NOT a net-cost or speedup decision: sort and index work rise with the
    # partition depth (printed above with build+query CPU). Any successor
    # efficacy decision must be frozen separately with a mapped common-work
    # model and build+query CPU, retaining regressions and refusals.
    assert v3_rows*2 < v1_rows
    assert stats_v3["coordinate_visits"] < stats_v1["coordinate_visits"]
    assert stats_v3["sort_work"] <= v3.RawExtremaBudgetsV3().max_sort_work


def test_support_subset_of_folded_sheet_identical_to_v1(v1, v3):
    points, rotation = folded_sheet(30_000)
    support = np.sort(np.unique((np.arange(9_000, dtype=np.int64)*7919) % 30_000)).astype(np.int64)
    queries = bend_axis_queries(rotation)[:6]
    reference = v1.project_raw_extrema_v1(points, queries, support_rows=support)
    answer = v3.project_raw_extrema_v3(points, queries, support_rows=support)
    assert reference["status"] == answer["status"] == "COMPLETE", answer.get("reasons")
    assert answer["support_scope"] == "COMPLETE_SUPPLIED_RAW_SUPPORT"
    assert answer["support_row_count"] == len(support)
    for a, b, query in zip(answer["projections"], reference["projections"], queries):
        same_record(a, b)
        exact(a, brute(points, query, support) if len(support) <= 20_000 else b)


def test_permuted_long_plate_diagonal_query_prunes(v3):
    points = np.asarray([(float(i), float(i % 7-3), float(i % 5-2)) for i in range(8192)], dtype=np.float64)
    permutation = np.asarray([(i*4051) % 8192 for i in range(8192)], dtype=np.int64)
    points = points[permutation]
    diagonal = ((.375, -.625, .875), (17., -11., 3.), .125)
    answer = v3.project_raw_extrema_v3(points, (AXIS, diagonal))
    assert answer["status"] == "COMPLETE", answer
    for record, projection in zip(answer["projections"], (AXIS, diagonal)):
        exact(record, brute(points, projection))
        assert record["work"]["query_rows_evaluated"] < len(points)
        assert record["work"]["nodes_pruned"] > 0


def test_loose_anticorrelated_boxes_can_require_all_rows_and_still_refuse_at_cap(v3):
    points = np.asarray([(float(i), -float(i), 0.) for i in range(1024)], dtype=np.float64)
    projection = ((1., 1., 0.), (0., 0., 0.), 0.)
    answer = v3.project_raw_extrema_v3(points, (projection,))
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], (0., 0., 0, 0))
    assert answer["projections"][0]["work"]["query_rows_evaluated"] >= len(points)
    required = answer["statistics"]["row_visits"]
    below = v3.project_raw_extrema_v3(points, (projection,),
        budgets=replace(v3.RawExtremaBudgetsV3(), max_row_visits=required-1))
    refused(below, "BUDGET_ROW_VISITS")
    assert below["statistics"]["row_visits"] == required


def test_equal_extrema_across_leaves_choose_original_smallest_raw_id(v3):
    points = np.asarray([(float(i-512)/50., -float(i-512)/50., 0.) for i in range(1025)], dtype=np.float64)
    points[7], points[700] = (3., 4., 0.), (6., 1., 0.)
    points[15], points[900] = (-3., -4., 0.), (-6., -1., 0.)
    projection = ((1., 1., 0.), (0., 0., 0.), 0.)
    expected = brute(points, projection)
    assert expected == (-7., 7., 15, 7)
    answer = v3.project_raw_extrema_v3(points, (projection,))
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], expected)


def test_support_is_complete_and_duplicate_positions_keep_distinct_raw_ids(v3):
    points = small_points()
    support = np.asarray([1, 2, 3, 4], dtype=np.int64)
    projection = ((1., -1., 0.), (0., 0., 0.), 0.)
    answer = v3.project_raw_extrema_v3(points, (projection,), support_rows=support)
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], brute(points, projection, support))
    assert answer["projections"][0]["maximum_raw_row"] == 1
    assert answer["support_row_count"] == 4
    assert answer["point_buffer_sha256"] == hashlib.sha256(points.tobytes(order="C")).hexdigest()
    assert answer["support_rows_sha256"] == hashlib.sha256(support.tobytes(order="C")).hexdigest()


def test_caller_mutation_after_build_cannot_change_query(v3, monkeypatch):
    points, support = small_points(), np.asarray([0, 2, 4], dtype=np.int64)
    expected = brute(points, AXIS, support)
    original = v3._Index.query
    def mutate_then_query(index, *args):
        points[:] = 999.
        support[:] = 1
        assert index.points.flags.writeable is False and index.rows.flags.writeable is False
        return original(index, *args)
    monkeypatch.setattr(v3._Index, "query", mutate_then_query)
    answer = v3.project_raw_extrema_v3(points, (AXIS,), support_rows=support)
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], expected)


@pytest.mark.parametrize("kind", ("list", "float32", "empty", "wrong_shape", "nan", "out_of_range"))
def test_invalid_points_clear_everything(v3, kind):
    points = small_points()
    if kind == "list": points = points.tolist()
    elif kind == "float32": points = points.astype(np.float32)
    elif kind == "empty": points = np.empty((0, 3), dtype=np.float64)
    elif kind == "wrong_shape": points = points.reshape(3, 5)
    elif kind == "nan": points[0, 0] = float("nan")
    else: points[0, 0] = 1_000_000.0000000001
    refused(v3.project_raw_extrema_v3(points, (AXIS,)))


@pytest.mark.parametrize("kind", ("list", "float", "empty", "negative", "past_end", "duplicate", "decreasing"))
def test_invalid_support_identities_are_not_sorted_or_welded(v3, kind):
    values = {"list": [0, 1], "float": np.asarray([0., 1.]), "empty": np.asarray([], dtype=np.int64),
              "negative": np.asarray([-1, 1]), "past_end": np.asarray([0, 5]),
              "duplicate": np.asarray([1, 1]), "decreasing": np.asarray([2, 1])}
    refused(v3.project_raw_extrema_v3(small_points(), (AXIS,), support_rows=values[kind]))


def test_typed_budget_and_every_scalar_limit_are_strict(v3):
    defaults = v3.RawExtremaBudgetsV3()
    for wrong in (None, {}, object()):
        refused(v3.project_raw_extrema_v3(small_points(), (AXIS,), budgets=wrong), "INVALID_BUDGET_TYPE")
    for field in fields(defaults):
        for wrong in (True, -1, float(getattr(defaults, field.name)), getattr(defaults, field.name)+1):
            refused(v3.project_raw_extrema_v3(small_points(), (AXIS,), budgets=replace(defaults, **{field.name: wrong})))


def test_v1_budget_type_is_rejected_by_v3(v1, v3):
    refused(v3.project_raw_extrema_v3(small_points(), (AXIS,), budgets=v1.RawExtremaBudgetsV1()), "INVALID_BUDGET_TYPE")


@pytest.mark.parametrize("field,counter", (
    ("max_index_visits", "index_visits"), ("max_row_visits", "row_visits"),
    ("max_coordinate_visits", "coordinate_visits"), ("max_sort_work", "sort_work"),
    ("max_node_visits", "node_visits"), ("max_hash_bytes", "hash_bytes"),
    ("max_working_bytes", "accounted_working_peak_bytes"),
))
def test_actual_resource_requirement_exact_limit_and_one_below(v3, field, counter):
    points, _ = folded_sheet(1_500)  # 24 leaves: partitions exercise sort work
    baseline = v3.project_raw_extrema_v3(points, (AXIS,))
    assert baseline["status"] == "COMPLETE", baseline
    required = baseline["statistics"][counter]
    assert required > 0
    exact_budget = v3.project_raw_extrema_v3(points, (AXIS,), budgets=replace(v3.RawExtremaBudgetsV3(), **{field: required}))
    assert exact_budget["status"] == "COMPLETE", exact_budget
    exact(exact_budget["projections"][0], blocked_oracle(points, AXIS))
    below = v3.project_raw_extrema_v3(points, (AXIS,), budgets=replace(v3.RawExtremaBudgetsV3(), **{field: required-1}))
    refused(below)
    assert below["statistics"][counter] >= required


def test_constructor_literal_work_counts_single_leaf(v3):
    points = np.asarray([(float(i), 0., 0.) for i in range(5)], dtype=np.float64)
    index = v3._Index(points, None, v3.RawExtremaBudgetsV3())
    assert index.depth == 0 and index.base == 1 and index.node_count == 2
    assert index.stats["queries"] == 0
    assert index.stats["row_visits"] == 15        # raw validation + root + leaves
    assert index.stats["index_visits"] == 20      # support + root + permutation copy + leaves; no partition
    assert index.stats["coordinate_visits"] == 120  # 6N + 12N + 6N
    assert index.stats["sort_work"] == 0
    assert index.stats["hash_bytes"] == 160
    assert index.stats["node_visits"] == 0


def test_constructor_literal_work_counts_two_leaves(v3):
    count = 100
    points = np.asarray([(float((i*7) % count), float(i % 3), 0.) for i in range(count)], dtype=np.float64)
    index = v3._Index(points, None, v3.RawExtremaBudgetsV3())
    assert index.depth == 1 and index.base == 2 and index.node_count == 4
    assert index.axis_schedule == (0,)
    assert index.stats["row_visits"] == 3*count
    assert index.stats["index_visits"] == 6*count          # support + root + copy + one partition (gather+write) + leaves
    assert index.stats["coordinate_visits"] == 25*count    # 6N + 12N + N (partition) + 6N
    assert index.stats["sort_work"] == count*(count-1).bit_length()
    assert index.stats["node_visits"] == 1
    assert int(index.end[2]) == 64 and int(index.start[3]) == 64 and int(index.end[3]) == count
    assert np.max(points[index.rows[:64], 0]) <= np.min(points[index.rows[64:], 0])


@pytest.mark.parametrize("field,counter,attempted", (
    ("max_row_visits", "row_visits", 5), ("max_coordinate_visits", "coordinate_visits", 30),
    ("max_hash_bytes", "hash_bytes", 120), ("max_index_visits", "index_visits", 5),
    ("max_working_bytes", "accounted_working_peak_bytes", 65_536),
))
def test_constructor_refusal_preserves_precharged_attempted_work(v3, field, counter, attempted):
    answer = v3.project_raw_extrema_v3(small_points(), (AXIS,), budgets=replace(v3.RawExtremaBudgetsV3(), **{field: 0}))
    refused(answer)
    assert answer["statistics"][counter] == attempted
    assert answer["statistics"]["queries"] == 0


def test_partition_sort_cap_refusal_keeps_attempted_ledger_without_binding(v3):
    points, _ = folded_sheet(4_000)
    answer = v3.project_raw_extrema_v3(points, (AXIS,), budgets=replace(v3.RawExtremaBudgetsV3(), max_sort_work=1))
    refused(answer, "BUDGET_SORT_WORK")
    assert answer["statistics"]["sort_work"] > 1 and answer["statistics"]["queries"] == 0


def test_invalid_late_projection_discards_valid_prefix(v3):
    answer = v3.project_raw_extrema_v3(small_points(), (AXIS, ((True, 0., 0.), AXIS[1], 0.)))
    refused(answer, "INVALID_PROJECTION_VECTOR")
    assert answer["discarded_projection_count"] == 1
    assert answer["statistics"]["queries"] == 1


def test_query_failure_invalidates_reusable_index(v3):
    index = v3._Index(small_points(), None, replace(v3.RawExtremaBudgetsV3(), max_queries=1))
    exact(index.query(*AXIS), brute(small_points(), AXIS))
    with pytest.raises(ValueError, match="BUDGET_QUERIES"):
        index.query(*AXIS)
    assert index.valid is False
    with pytest.raises(ValueError, match="INDEX_INVALID_OR_PREVIOUSLY_FAILED"):
        index.query(*AXIS)


def test_unexpected_late_exception_clears_all_output_and_retains_work(v3, monkeypatch):
    original = v3._Index._bounds
    def late_failure(index, *args):
        if index.stats["queries"] == 2:
            raise RuntimeError("invented late node failure")
        return original(index, *args)
    monkeypatch.setattr(v3._Index, "_bounds", late_failure)
    answer = v3.project_raw_extrema_v3(small_points(), (AXIS, AXIS))
    refused(answer, "EXTREMA_HELPER_EXCEPTION")
    assert answer["discarded_projection_count"] == 1
    assert answer["statistics"]["queries"] == 2


def test_final_serialization_failure_cannot_leave_complete_shell(v3, monkeypatch):
    original = v3.json.JSONEncoder.iterencode
    def fail_final(encoder, value, *args, **kwargs):
        if type(value) is dict and value.get("schema_version") == "raw_extrema_index_v3":
            yield "{"
            raise ValueError("invented final encoder failure")
        yield from original(encoder, value, *args, **kwargs)
    monkeypatch.setattr(v3.json.JSONEncoder, "iterencode", fail_final)
    answer = v3.project_raw_extrema_v3(small_points(), (AXIS,))
    refused(answer, "EXTREMA_HELPER_EXCEPTION")
    assert answer["discarded_projection_count"] == 1


def test_budget_interrupt_is_not_converted_to_ordinary_refusal(v3, monkeypatch):
    class InventedInterrupt(BaseException):
        pass
    captured = []
    def interrupt(index, *args):
        captured.append(index)
        raise InventedInterrupt()
    monkeypatch.setattr(v3._Index, "_bounds", interrupt)
    with pytest.raises(InventedInterrupt):
        v3.project_raw_extrema_v3(small_points(), (AXIS,))
    assert len(captured) == 1 and captured[0].valid is False


def test_capacity_analytic_axis_witnesses_262144_rows(v3):
    count, multiplier = 262_144, 262_147
    assert math.gcd(count, multiplier) == 1
    raw_rows = np.arange(count, dtype=np.int64)
    coordinate_indices = (raw_rows*multiplier) % count
    points = np.empty((count, 3), dtype=np.float64)
    points[:, 0] = coordinate_indices
    points[:, 1] = coordinate_indices % 149
    points[:, 2] = coordinate_indices % 37
    projections = (AXIS, ((0., 1., 0.), (0., 0., 0.), 0.), ((0., 0., 1.), (0., 0., 0.), 0.))
    answer = v3.project_raw_extrema_v3(points, projections)
    assert answer["status"] == "COMPLETE", answer.get("reasons")
    for axis, record in enumerate(answer["projections"]):
        maximum = float(np.max(points[:, axis]))
        minimum_id = int(np.flatnonzero(points[:, axis] == 0.)[0])
        maximum_id = int(np.flatnonzero(points[:, axis] == maximum)[0])
        exact(record, (0., maximum, minimum_id, maximum_id))
        assert record["work"]["query_rows_evaluated"] < count
    assert answer["tree_depth"] == 12 and answer["node_slots"] == 8192
