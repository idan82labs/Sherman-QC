"""Unexecuted synthetic V2 index tests and a fixed four-population cost report.

No physical inputs; costs are reported for the separately frozen root decision,
not asserted as a synthetic correctness requirement. Sixteen prospective items,
four with 32768 rows and 12 fixed queries. Other fixtures have <=1025 rows. The folded
fourth population tests the limitation of one global shear on a whole part:
fixed Rx(5/13,12/13) then Rz(3/5,4/5), separate represented binary64 stages.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys
import time
from types import ModuleType
from dataclasses import replace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("cad_guided_raw_extrema_index_v1", "cad_guided_sheared_bounds_v1",
           "cad_guided_raw_extrema_index_v2")
EXTRA = ("chooser_steps", "build_steps", "merge_steps", "query_bound_steps", "packing_steps")
# Successful V1 _bounds: six sign comparisons + six coordinate selections +
# two times (nine staged arithmetic + zero comparison/selection) + three
# finite/order checks. Add this only where V1's uncharged corner code runs.
# This is a declared source ledger, not all Python/hardware instructions.
WORLD_BOUND_STEPS = 37
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
    # Explicit inert package: never execute backend/__init__.py.
    names = ["backend"] + ["backend." + name for name in MODULES]
    assert all(name not in sys.modules for name in names)
    package = ModuleType("backend")
    package.__path__ = []
    sys.modules["backend"] = package
    loaded = []
    try:
        for short, name in zip(MODULES, names[1:]):
            spec = importlib.util.spec_from_file_location(name, ROOT / "backend" / (short + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            setattr(package, short, module)
            spec.loader.exec_module(module)
            loaded.append(module)
        yield tuple(loaded)
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


def bits(value):
    return struct.pack("!d", float(value))


def scalar(row, query):
    direction, origin, constant = query
    x = float(row[0]) - float(origin[0])
    y = float(row[1]) - float(origin[1])
    z = float(row[2]) - float(origin[2])
    tx, ty, tz = x*direction[0], y*direction[1], z*direction[2]
    first = tx + ty
    second = first + tz
    result = second + constant
    return 0. if result == 0. else result


def scalar_oracle(points, query, rows=None):
    ids = range(len(points)) if rows is None else map(int, rows)
    values = [(scalar(points[i], query), i) for i in ids]
    low = min(values, key=lambda x: (x[0], x[1]))
    high = max(values, key=lambda x: (x[0], -x[1]))
    return low[0], high[0], low[1], high[1]


def blocked_oracle(points, query):
    # Full population in original-ID order; no candidate-provided membership.
    # Independent implementation, validated against scalar stage order below.
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


def refused(answer, reason=None):
    assert answer["status"] == "UNAVAILABLE" and answer["projections"] == []
    assert answer["accepted_registration"] is False
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    for name in ("point_buffer_sha256", "support_rows_sha256", "support_scope"):
        assert name not in answer
    if reason is not None:
        assert answer["reasons"] == [reason]


def population(kind, count=32768):
    assert count == 32768
    raw = np.arange(count, dtype=np.int64)
    assert math.gcd(4051, count) == 1
    k = (raw * 4051) % count  # Frozen bijection, never outcome-selected order.
    points = np.empty((count, 3), dtype=np.float64)
    if kind == "folded_three_planes":
        panel, t = k % 3, k // 3
        u, v = ((t % 128)-63.5)/8., ((t // 128)-42.)/8.
        points[:] = 0.
        points[panel == 0, 0], points[panel == 0, 1] = u[panel == 0], v[panel == 0]
        points[panel == 1, 0], points[panel == 1, 2] = u[panel == 1], v[panel == 1]
        points[panel == 2, 1], points[panel == 2, 2] = u[panel == 2], v[panel == 2]
        assert [int(np.count_nonzero(panel == i)) for i in range(3)] == [10923, 10923, 10922]
        # Fixed ideal proper rotations; represented rows, not ideal geometry,
        # are the numerical oracle inputs. No fit or post-result angle choice.
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
    assert points.shape == (32768, 3) and points.nbytes == 786432
    assert np.isfinite(points).all() and np.all(np.abs(points) <= 1e6)
    return points


@pytest.mark.parametrize("kind", ("exact_flat", "resolved_unique_extrema", "repeated_plateau", "folded_three_planes"))
def test_fixed_four_population_correctness_and_unfiltered_cost_report(modules, monkeypatch, kind):
    import json
    v1, _, v2 = modules
    points = population(kind)
    oracle = [blocked_oracle(points, q) for q in QUERIES]
    point_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
    support_sha = hashlib.sha256(np.arange(len(points), dtype=np.int64).tobytes()).hexdigest()
    reports, answers = {}, {}
    for label, cls, project in (("v1", v1._Index, v1.project_raw_extrema_v1),
                                ("v2", v2._IndexV2, v2.project_raw_extrema_v2)):
        checkpoints, constructors, calls, world_calls = [], [], [0], [0]
        build_cpu, query_cpu = [], []
        original_init = cls.__init__
        original_query = cls.query
        original_bound = cls._bounds if label == "v1" else cls._node_bounds
        def observe_bound(index, *args, _original=original_bound, **kwargs):
            calls[0] += 1
            if label == "v1" or not index.shear_enabled:
                world_calls[0] += 1
            return _original(index, *args, **kwargs)
        def observe_init(index, *args, _original=original_init, **kwargs):
            started = time.process_time()
            try:
                return _original(index, *args, **kwargs)
            finally:
                build_cpu.append(time.process_time()-started)
        def observe_query(index, *args, _original=original_query, **kwargs):
            if not constructors:
                constructors.append(dict(index.stats))
            started = time.process_time()
            try:
                result = _original(index, *args, **kwargs)
            finally:
                query_cpu.append(time.process_time()-started)
            if index.stats["queries"] in (1, 3, 12):
                checkpoints.append(dict(after_query=index.stats["queries"], statistics=dict(index.stats),
                    observed_bound_calls=calls[0], old_world_bound_calls=world_calls[0],
                    common_steps=index.stats["coordinate_visits"]+WORLD_BOUND_STEPS*world_calls[0],
                    build_plus_query_cpu_seconds=sum(build_cpu)+sum(query_cpu)))
            return result
        with monkeypatch.context() as m:
            m.setattr(cls, "query", observe_query)
            m.setattr(cls, "__init__", observe_init)
            m.setattr(cls, "_bounds" if label == "v1" else "_node_bounds", observe_bound)
            answer = project(points, QUERIES)
        answers[label] = answer
        # Both dispositions are emitted below before candidate result assertions.
        reports[label] = dict(status=answer["status"], reasons=answer.get("reasons", []),
            constructor=constructors[0] if constructors else None, checkpoints=checkpoints,
            statistics=answer["statistics"], observed_bound_calls=calls[0],
            old_world_bound_calls=world_calls[0], build_cpu_seconds=sum(build_cpu),
            query_cpu_seconds=sum(query_cpu), build_plus_query_cpu_seconds=sum(build_cpu)+sum(query_cpu),
            cpu_scope="INSTRUMENTED_CONSTRUCTION_AND_QUERY_ONLY_EXCLUDES_ORACLE_AND_REPORT",
            common_steps=answer["statistics"].get("coordinate_visits", 0)+WORLD_BOUND_STEPS*world_calls[0],
            common_scope="LEGACY_COORDINATE_CHARGES_PLUS_ADDED_STEPS_PLUS_37_PER_UNCHARGED_WORLD_BOUND_NOT_HARDWARE_OPS",
            original_corner_arithmetic_ops=18*calls[0] if label == "v1" else None,
            corner_scope="18_ARITHMETIC_OPS_ONLY_EXCLUDES_SIGN_CHECK_ZERO_OVERHEAD_NOT_TOTAL_COMMON_COST",
            projections=answer.get("projections", []))
    summary = dict(schema="SHEARED_INDEX_V2_FIXED_EFFICACY_SOURCE_V1", population=kind,
        rows=len(points), query_count=12, point_sha256=point_sha, support_sha256=support_sha,
        query_records=QUERIES, reports=reports,
        efficacy_decision="EXTERNAL_FROZEN_RULE_RESOLVED_AND_FOLDED_LOWER_COMMON_WORK_NO_FOUR_CASE_AGGREGATE_CPU_REGRESSION",
        whole_folded_part_single_shear_limitation=kind == "folded_three_planes")
    blob = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert len(blob.encode()) <= 32768
    print("SHEARED_INDEX_V2_EFFICACY_JSON=" + blob)
    # Refusal is retained above, never compared as a completed cheaper prefix.
    assert all(report["status"] == "COMPLETE" for report in reports.values()), reports
    assert all(len(report["checkpoints"]) == 3 for report in reports.values())
    for answer in answers.values():
        assert len(answer["projections"]) == 12
        for record, expected in zip(answer["projections"], oracle):
            exact(record, expected)
            assert record["support_row_count"] == len(points)
        assert answer["point_buffer_sha256"] == point_sha
        assert answer["support_rows_sha256"] == support_sha
        assert answer["support_row_count"] == len(points)
        assert answer["support_scope"] == "WHOLE_RAW_BUFFER"
    assert hashlib.sha256(points.tobytes(order="C")).hexdigest() == point_sha
    # No20%/CPU/leaf reduction threshold and no omission of flat/plateau negatives.


@pytest.mark.parametrize("kind", ("oblique", "cancellation", "subnormal", "folded"))
def test_small_actual_node_enclosures_and_independent_oracles(modules, kind):
    v1, _, v2 = modules
    points = np.asarray([(float(i-300)/8., float((i*29)%31-15)/16., float(i%7-3)/32.)
                         for i in range(601)], dtype=np.float64)
    if kind == "oblique": points[:, 2] = .625*points[:, 0]-.375*points[:, 1]+points[:, 2]
    elif kind == "cancellation": points[:, 1] = -points[:, 0]
    elif kind == "subnormal": points *= float.fromhex("0x1p-1022")
    else: points[np.arange(len(points)), np.arange(len(points)) % 3] = 0.
    index = v2._IndexV2(points, None, v1.RawExtremaBudgetsV1())
    for query in (QUERIES[0], QUERIES[9], QUERIES[11]):
        assert scalar_oracle(points, query) == blocked_oracle(points, query)
        direction, origin, constant = query
        for node in range(1, index.node_count):
            start, stop = int(index.start[node]), int(index.end[node])
            if stop <= start:
                continue
            lower, upper, world_lower, world_upper = index._node_bounds(
                node, np.asarray(direction), np.asarray(origin), constant)
            assert world_lower <= lower <= upper <= world_upper
            ids = index.rows[start:stop]
            assert int(index.minimum_raw[node]) == min(map(int, ids))
            for raw_id in ids:
                assert lower <= scalar(points[int(raw_id)], query) <= upper
        exact(index.query(*query), scalar_oracle(points, query))


def test_cross_leaf_original_id_ties_and_complete_subset(modules):
    v1, _, v2 = modules
    points = np.asarray([(float(i-512)/8., -float(i-512)/8., 0.) for i in range(1025)])
    points[7], points[700] = (3., 4., 0.), (6., 1., 0.)
    points[15], points[900] = (-3., -4., 0.), (-6., -1., 0.)
    rows = np.asarray([i for i in range(1025) if i % 3 != 0 or i in (7, 15, 700, 900)], dtype=np.int64)
    query = ((1., 1., 0.), (0., 0., 0.), 0.)
    expected = scalar_oracle(points, query, rows)
    assert expected == (-7., 7., 15, 7)
    answer = v2.project_raw_extrema_v2(points, (query,), support_rows=rows)
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], expected)
    assert answer["support_row_count"] == len(rows)
    assert answer["support_rows_sha256"] == hashlib.sha256(rows.tobytes()).hexdigest()
    assert v2._IndexV2._evaluate is v1._Index._evaluate
    assert v2._IndexV2._build is v1._Index._build


def test_world_priority_not_intersection_priority_controls_traversal(modules, monkeypatch):
    v1, _, v2 = modules
    points = np.asarray([(float(i*i), -float(i*i), 0.) for i in range(1000)], dtype=np.float64)
    index = v2._IndexV2(points, None, v1.RawExtremaBudgetsV1())
    query = ((1., 1., 0.), (0., 0., 0.), 0.)
    assert scalar_oracle(points, query) == (0., 0., 0, 0)
    def sound_double(node):
        world_low = float(index.low[node, 0])+float(index.low[node, 1])
        world_high = float(index.high[node, 0])+float(index.high[node, 1])
        return max(world_low, -1.), min(world_high, 1.), world_low, world_high
    # Inert sound enclosure double isolates ordering; actual primitive bounds
    # are exhaustively checked in the preceding four controls.
    def reference(use_world):
        visits, stack = [], [1]
        while stack:
            node = stack.pop()
            low, high, _, _ = sound_double(node)
            if low == 0. == high and int(index.minimum_raw[node]) >= 0:
                continue
            if node >= index.base:
                visits.append(tuple(map(int, index.rows[int(index.start[node]):int(index.end[node])])))
            else:
                children = []
                for child in (2*node, 2*node+1):
                    if index.end[child] > index.start[child]:
                        bounds = sound_double(child)
                        a, b = bounds[2:] if use_world else bounds[:2]
                        children.append((max(-a, b), -child, child))
                children.sort()
                stack.extend(child for _, _, child in children)
        return visits
    expected = reference(True)
    assert expected != reference(False), "positive premise must distinguish the two priority policies"
    original = v2._IndexV2._evaluate
    observations = []
    def observe(obj, ids, *args):
        observations.append(tuple(map(int, ids)))
        return original(obj, ids, *args)
    monkeypatch.setattr(index, "_node_bounds", lambda node, *args: sound_double(node))
    monkeypatch.setattr(v2._IndexV2, "_evaluate", observe)
    exact(index.query(*query), (0., 0., 0, 0))
    assert observations[1:] == expected  # First call is unchanged seed evaluation.


def test_additional_work_is_cumulative_coordinate_work_not_hidden(modules):
    v1, _, v2 = modules
    points = np.asarray([(float(i)/8., float(i%7)/16., .625*(i/8.)-.375*((i%7)/16.))
                         for i in range(515)])
    index = v2._IndexV2(points, None, v1.RawExtremaBudgetsV1())
    assert index.shear_enabled is True
    before = dict(index.stats)
    assert before["sheared_bound_steps"] == sum(before[key] for key in EXTRA)
    assert before["coordinate_visits"] == 25*len(points)+before["sheared_bound_steps"]
    assert before["build_steps"] == 61*len(points)+78*math.ceil(len(points)/256)
    assert before["merge_steps"] == 75*before["sheared_merges_completed"]
    assert before["sheared_leaves_completed"] == 3
    assert before["sheared_merges_completed"] == 2
    assert before["sheared_single_child_copies"] == 1
    assert before["query_bound_steps"] == 0
    exact(index.query(*QUERIES[0]), scalar_oracle(points, QUERIES[0]))
    after = index.stats
    assert after["sheared_bound_steps"] == sum(after[key] for key in EXTRA)
    assert after["coordinate_visits"] == 25*len(points)+9*after["query_rows_evaluated"]+after["sheared_bound_steps"]
    assert after["query_bound_steps"] > 0 and after["query_bound_steps"] % 390 == 0
    assert index.shear_bounds.shape == (index.node_count, 4)
    assert index.shear_bounds.nbytes == 32*index.node_count
    assert index.shear_bounds.flags.writeable is False


def test_existing_coordinate_cap_refuses_during_added_work(modules):
    v1, _, v2 = modules
    points = np.asarray([(float(i), -float(i), 0.) for i in range(17)])
    baseline = v1.project_raw_extrema_v1(points, (QUERIES[0],))
    assert baseline["status"] == "COMPLETE"
    cap = 25*len(points)  # Exact unchanged V1 construction, no added-work allowance.
    answer = v2.project_raw_extrema_v2(points, (QUERIES[0],),
        budgets=replace(v1.RawExtremaBudgetsV1(), max_coordinate_visits=cap))
    refused(answer, "BUDGET_COORDINATE_VISITS")
    assert answer["statistics"]["queries"] == 0
    assert answer["statistics"]["coordinate_visits"] == cap+16
    assert answer["statistics"]["sheared_bound_steps"] == 16
    assert answer["statistics"]["chooser_steps"] == 16
    assert all(answer["statistics"][name] == 0 for name in EXTRA if name != "chooser_steps")


def test_constructor_primitive_failure_keeps_attempted_ledger_and_no_binding(modules, monkeypatch):
    _, primitive, v2 = modules
    points = np.asarray([(float(i), float(i%3), .625*i-.375*(i%3)) for i in range(17)])
    premise = v2.project_raw_extrema_v2(points, (QUERIES[0],))
    assert premise["status"] == "COMPLETE" and premise["packed_shear_bytes"] > 0
    calls = []
    def failed_leaf(*args, charge, **kwargs):
        calls.append(True)
        charge(64)
        raise primitive.BoundUnavailable("INVENTED_LEAF_BOUND_FAILURE")
    monkeypatch.setattr(primitive, "build_leaf_box", failed_leaf)
    answer = v2.project_raw_extrema_v2(points, (QUERIES[0],))
    refused(answer, "INVENTED_LEAF_BOUND_FAILURE")
    assert calls == [True]
    assert answer["statistics"]["queries"] == 0
    assert answer["statistics"]["build_steps"] >= 64


def test_late_query_failure_clears_prefix_and_invalidates_reusable_index(modules, monkeypatch):
    v1, _, v2 = modules
    points = np.asarray([(float(i), float(i%7), 0.) for i in range(17)])
    original = v2._IndexV2._node_bounds
    touched = []
    def late(index, *args):
        if index.stats["queries"] == 2:
            touched.append(index)
            raise RuntimeError("invented second-query failure")
        return original(index, *args)
    monkeypatch.setattr(v2._IndexV2, "_node_bounds", late)
    answer = v2.project_raw_extrema_v2(points, (QUERIES[0], QUERIES[1]))
    refused(answer, "EXTREMA_V2_EXCEPTION")
    assert answer["discarded_projection_count"] == 1 and len(touched) == 1
    assert touched[0].valid is False
    assert answer["statistics"]["queries"] == 2
    with pytest.raises(ValueError, match="INDEX_INVALID_OR_PREVIOUSLY_FAILED"):
        touched[0].query(*QUERIES[0])


def test_original_snapshot_support_and_tree_remain_unchanged(modules):
    v1, _, v2 = modules
    points = np.asarray([(float(i), -float(i), float(i%5)) for i in range(515)])
    rows = np.arange(0, len(points), 2, dtype=np.int64)
    expected = scalar_oracle(points, QUERIES[0], rows)
    old = v1._Index(points, rows, v1.RawExtremaBudgetsV1())
    new = v2._IndexV2(points, rows, v1.RawExtremaBudgetsV1())
    for field in ("points", "rows", "low", "high", "start", "end", "minimum_raw"):
        assert np.array_equal(getattr(old, field), getattr(new, field))
    assert old.seed_rows == new.seed_rows and old.sort_axis == new.sort_axis
    assert old.point_sha256 == new.point_sha256 and old.support_sha256 == new.support_sha256
    points[:] = 123.
    rows[:] = 1
    exact(new.query(*QUERIES[0]), expected)


def test_zero_direction_rank_one_and_positive_zero_contract(modules):
    _, _, v2 = modules
    points = np.asarray([(float(i), -float(i), 0.) for i in range(515)])
    query = ((-0., 0., -0.), (17., -19., 23.), -0.)
    answer = v2.project_raw_extrema_v2(points, (query,))
    assert answer["status"] == "COMPLETE", answer
    exact(answer["projections"][0], (0., 0., 0, 0))
    assert answer["schema_version"] == "raw_extrema_index_v2"
    assert answer["arithmetic"] == "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1"
    assert answer["accepted_registration"] is False and answer["production_use_allowed"] is False
    assert answer["shear_chooser"]["status"] == "DISABLED_ZERO_SHEAR"
    assert answer["packed_shear_bytes"] == 0 and answer["packed_shear_sha256"] is None
    assert answer["statistics"]["chooser_steps"] > 0
    assert all(answer["statistics"][name] == 0 for name in EXTRA if name != "chooser_steps")
