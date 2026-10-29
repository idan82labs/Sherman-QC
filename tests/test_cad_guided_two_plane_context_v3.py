"""Prospective invented connected V3 context checks, never physical F1.

Source authored only: do not import/collect/run without root's exact frozen
protocol, independent PRE and resource gate. The largest fixture contains
900,000 invented float64 XYZ rows, four scan/CAD triangles, two four-row scan
supports and one caller assignment. No native fitting, application bootstrap,
raw/Gold data, private artifacts or old attempts are accessed. Tiny oracles
round every individual operation using Python float, independently of NumPy's
projection/index implementation. This checks staged binary64 semantics, not
legacy BLAS equivalence or exact-real physical uncertainty.
"""
import ast
from dataclasses import replace
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SLOTS = ("WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
         "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT")
EMPTY = ("plane_bindings", "scheduled_pairs", "families", "candidates", "context_projections", "scan_extrema_indexes")


@pytest.fixture(scope="module")
def modules():
    name = "_invented_connected_context_v3_contract"
    assert not any(key == name or key.startswith(name + ".") for key in sys.modules)
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "backend")]
    sys.modules[name] = package
    try:
        v3 = importlib.import_module(name + ".cad_guided_two_plane_context_v3")
        yield types.SimpleNamespace(v3=v3,
            v1=importlib.import_module(name + ".cad_guided_two_plane_context_v1"),
            v2=importlib.import_module(name + ".cad_guided_two_plane_context_v2"),
            extrema=importlib.import_module(name + ".cad_guided_raw_extrema_index_v1"))
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def _sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _case(m, *, targets=1, cad_height=9.):
    """Two independently specified rectangles plus whole-buffer-only extrema."""
    points = np.array([(0., -6., -2.), (0., 2., -2.), (0., 2., 9.), (0., -6., 9.),
                       (-3., 0., -2.), (7., 0., -2.), (7., 0., 9.), (-3., 0., 9.),
                       (1., 1., -5.), (-1., -1., 13.)], dtype=np.float64)
    triangles = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], dtype=np.int64)
    scans = []
    for index, normal in enumerate(((1., 0., 0.), (0., 1., 0.))):
        ids = np.array((2*index, 2*index+1), dtype=np.int64)
        raw = np.arange(4*index, 4*index+4, dtype=np.int64)
        scans.append(m.v1.ContextPlaneV1(31+index, normal, (0., 0., 0.), 0., 0.,
            True, ids, _sha(ids), raw, _sha(raw)))
    cad, faces, locals_, planes, matches = [], [], [], [], []
    for target in range(targets):
        for index, normal in enumerate(((1., 0., 0.), (0., 1., 0.))):
            vertices = (np.array([(0., -6., -2.), (0., 2., -2.),
                                 (0., 2., cad_height), (0., -6., cad_height)]) if index == 0
                        else np.array([(-3., 0., -2.), (7., 0., -2.),
                                       (7., 0., cad_height), (-3., 0., cad_height)]))
            vertices[:, index] += 2.*target
            rows = np.array((len(cad), len(cad)+1), dtype=np.int64)
            face = 101+2*target+index
            cad.extend((vertices[[0, 1, 2]], vertices[[0, 2, 3]]))
            faces.extend((face, face)); locals_.extend((19, 3))
            anchor = tuple(2.*target if k == index else 0. for k in range(3))
            planes.append(m.v1.ContextPlaneV1(face, normal, anchor, 0., 0., True, rows, _sha(rows)))
        matches.append(m.v1.OrderedPlanePairV1((31, 32), (101+2*target, 102+2*target)))
    return dict(scan_points=points, scan_triangles=triangles,
        cad_triangles=np.asarray(cad, dtype=np.float64), cad_face_ids=np.asarray(faces, dtype=np.int64),
        cad_source_triangle_ids=np.asarray(locals_, dtype=np.int64), scan_planes=tuple(scans),
        cad_planes=tuple(planes), ordered_matches=tuple(matches), scan_origin_mm=(0., 0., 0.),
        cad_origin_mm=(0., 0., 0.), unit_assumption="INVENTED_ONLY_MM_DECLARATION",
        source_frame="invented_scan_v3", target_frame="invented_cad_v3")


def _call(m, args, **caps):
    if "max_signed_families" in caps and type(caps["max_signed_families"]) is int and caps["max_signed_families"] >= 0:
        caps.setdefault("max_extrema_queries", min(3072, 3*caps["max_signed_families"]))
    return m.v3.propose_two_plane_context_v3(**args, budgets=m.v3.TwoPlaneContextBudgetsV3(**caps))


def _scope(result):
    assert result["schema_version"] == "two_plane_context_v3"
    assert result["accepted_registration"] is False
    assert result["finite_support_match_verified"] is False
    assert result["production_use_allowed"] is False
    assert result["observed_bend_count"] is None
    assert result["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert result["axial_context_status"] == "AXIAL_CONTEXT_NOT_CERTIFIED"
    assert result["search_exhaustive"] is False


def _complete(result, assignments=1):
    _scope(result)
    assert result["status"] == "COMPLETE", result.get("reasons")
    assert result["selected_policy_evaluation_complete"] is True
    stats = result["statistics"]
    assert stats["signed_families_planned"] == stats["signed_families_attempted"] == stats["signed_families_completed"] == 4*assignments
    assert stats["candidate_slots_planned"] == stats["slot_records_completed"] == 32*assignments
    assert stats["proposed_slot_count"] + stats["absent_midpoint_slots"] + stats["rejected_slot_count"] == 32*assignments
    assert len(result["families"]) == 4*assignments
    for assignment in range(assignments):
        families = [f for f in result["families"] if f["ordered_assignment_id"] == assignment]
        assert [f["normal_signs"] for f in families] == [list(s) for s in itertools.product((-1, 1), repeat=2)]
        assert all(tuple(s["slot"] for s in f["slots"]) == SLOTS for f in families)
    seen = []
    slots = {(f["signed_family_id"], s["slot"]): s for f in result["families"] for s in f["slots"]}
    for index, candidate in enumerate(result["candidates"]):
        assert candidate["candidate_index"] == index
        assert candidate["accepted_registration"] is False
        for witness in candidate["contributing_slots"]:
            key = (witness["signed_family_id"], witness["slot"])
            assert slots[key]["candidate_index"] == index
            seen.append(key)
    assert len(seen) == len(set(seen)) == stats["proposed_slot_count"]


def _refused(result, reason=None):
    _scope(result)
    assert result["status"] == "UNAVAILABLE"
    assert result["usable_proposals"] is False
    assert result["selected_policy_evaluation_complete"] is False
    assert result["input_digests"] == {}
    assert all(result[name] == [] for name in EMPTY)
    assert result["reasons"]
    if reason is not None:
        assert result["reasons"] == [reason]
    assert len(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 8192


def _scalar_projection(point, direction, origin, constant):
    products = []
    for coordinate in range(3):
        difference = float(point[coordinate]) - float(origin[coordinate])
        products.append(difference * float(direction[coordinate]))
    partial = products[0] + products[1]
    partial = partial + products[2]
    value = partial + float(constant)
    return 0. if value == 0. else value


def _assert_projection_oracle(args, result):
    for record in result["context_projections"]:
        role = record["role"]
        whole = record["scope"] == "WHOLE_SUPPLIED_BUFFER"
        if role == "scan":
            if whole:
                ids = range(len(args["scan_points"]))
            else:
                plane = next(p for p in args["scan_planes"] if p.raw_vertex_rows_sha256 == record["support_rows_sha256"])
                ids = [int(x) for x in plane.raw_vertex_rows]
            entries = [(int(row), args["scan_points"][row]) for row in ids]
        else:
            if whole:
                ids = range(len(args["cad_triangles"]))
            else:
                plane = next(p for p in args["cad_planes"] if p.triangle_rows_sha256 == record["support_rows_sha256"])
                ids = [int(x) for x in plane.triangle_rows]
            entries = [((int(row), slot), args["cad_triangles"][row, slot]) for row in ids for slot in range(3)]
        values = [(identity, _scalar_projection(point, record["projection_direction"],
            record["projected_coordinate_origin_mm"], record["projected_constant_mm"])) for identity, point in entries]
        lower = min(value for _, value in values)
        upper = max(value for _, value in values)
        lo_id = min(identity for identity, value in values if value == lower)
        hi_id = min(identity for identity, value in values if value == upper)
        assert record["interval_mm"] == [lower, upper]
        for key, identity in (("minimum_witness", lo_id), ("maximum_witness", hi_id)):
            if role == "scan":
                assert record[key] == {"raw_vertex_row": identity}
            else:
                row, slot = identity
                assert record[key] == dict(triangle_row=row, vertex_slot=slot,
                    face_id=int(args["cad_face_ids"][row]), source_triangle_id=int(args["cad_source_triangle_ids"][row]))
        for value in record["interval_mm"]:
            if value == 0.:
                assert math.copysign(1., value) == 1.


@pytest.mark.parametrize("targets", (1, 5, 16, 256))
def test_full_caller_schedule_signs_slots_and_independent_projection_oracle(modules, targets):
    args = _case(modules, targets=targets)
    result = _call(modules, args)
    _complete(result, targets)
    _assert_projection_oracle(args, result)


@pytest.mark.parametrize("variant", ("normal", "origins", "reverse_assignment", "negative_normals", "noncontiguous"))
def test_staged_projection_identity_and_scope_variants(modules, variant):
    args = _case(modules)
    if variant == "origins":
        args.update(scan_origin_mm=(1.25, -3.5, 7.125), cad_origin_mm=(-2.75, 5.25, -9.5))
    elif variant == "reverse_assignment":
        args["ordered_matches"] = (modules.v1.OrderedPlanePairV1((32, 31), (102, 101)),)
    elif variant == "negative_normals":
        args["scan_planes"] = tuple(replace(p, normal=tuple(-x for x in p.normal)) for p in args["scan_planes"])
    elif variant == "noncontiguous":
        for name in ("scan_points", "scan_triangles", "cad_triangles"):
            args[name] = np.asfortranarray(args[name])
            assert not args[name].flags.c_contiguous
    result = _call(modules, args)
    _complete(result)
    _assert_projection_oracle(args, result)


@pytest.mark.parametrize("cad_height", (-1., 9., 21.))
def test_absent_midpoints_do_not_reduce_explicit_slot_denominator(modules, cad_height):
    result = _call(modules, _case(modules, cad_height=cad_height))
    _complete(result)
    if cad_height < 9.:
        assert result["statistics"]["absent_midpoint_slots"] > 0


@pytest.mark.parametrize("field,value", (("max_signed_families", 3), ("max_candidate_slots", 31),
    ("max_scan_pairs", 0), ("max_ordered_cad_matches_per_scan_pair", 0)))
def test_schedule_preflight_precedes_array_admission(modules, field, value):
    args = _case(modules)
    args["scan_points"] = object()
    result = _call(modules, args, **{field: value})
    _refused(result, "ORDERED_MATCH_TABLE_BUDGET")
    assert result["statistics"]["scan_row_visits"] == 0


@pytest.mark.parametrize("field,value", (("chunk_rows", 2), ("max_working_bytes", True),
    ("max_scan_rows", 1_000_001), ("max_candidate_slots", 8193),
    ("max_ordered_cad_matches_per_scan_pair", 257)))
def test_invalid_original_budget_values_remain_refusals(modules, field, value):
    _refused(_call(modules, _case(modules), **{field: value}), "INVALID_BUDGET_VALUES")


@pytest.mark.parametrize("defect", ("stale_raw_digest", "incomplete_raw_union", "duplicate_match", "cad_face_incomplete"))
def test_admission_and_identity_failures_clear_the_entire_context(modules, defect):
    args = _case(modules)
    if defect == "stale_raw_digest":
        args["scan_planes"] = (replace(args["scan_planes"][0], raw_vertex_rows_sha256="0"*64), args["scan_planes"][1])
    elif defect == "incomplete_raw_union":
        rows = np.array((0, 1, 2), dtype=np.int64)
        args["scan_planes"] = (replace(args["scan_planes"][0], raw_vertex_rows=rows,
            raw_vertex_rows_sha256=_sha(rows)), args["scan_planes"][1])
    elif defect == "duplicate_match":
        args["ordered_matches"] *= 2
    else:
        rows = np.array((0,), dtype=np.int64)
        args["cad_planes"] = (replace(args["cad_planes"][0], triangle_rows=rows,
            triangle_rows_sha256=_sha(rows)), args["cad_planes"][1])
    _refused(_call(modules, args))


def test_rejected_pair_keeps_four_families_and_thirty_two_rejected_slots(modules):
    args = _case(modules)
    first, second = args["scan_planes"]
    args["scan_planes"] = (first, replace(second, normal=first.normal))
    result = _call(modules, args)
    _complete(result)
    assert result["statistics"]["rejected_slot_count"] == 32
    assert result["context_projections"] == []
    assert result["candidates"] == []


def test_empty_schedule_has_no_usable_seeds(modules):
    args = _case(modules)
    args["ordered_matches"] = ()
    result = _call(modules, args)
    _complete(result, 0)
    assert result["usable_proposals"] is False
    assert result["context_projections"] == []


def test_qualified_standalone_constructor_is_never_called(modules, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("standalone raw snapshot/validation must not run inside context")
    monkeypatch.setattr(modules.extrema._Index, "__init__", forbidden)
    result = _call(modules, _case(modules))
    _complete(result)


def test_all_support_indexes_share_exact_bytes_backed_raw_snapshot(modules, monkeypatch):
    args = _case(modules)
    original = modules.extrema._Index._build
    snapshots, supports = [], []
    def observed(index, rows):
        snapshots.append(index.points)
        supports.append(tuple(int(x) for x in rows))
        assert index.points is not args["scan_points"]
        assert index.points.flags.writeable is False
        owner = index.points
        while isinstance(owner, np.ndarray):
            owner = owner.base
        assert type(owner) is bytes
        with pytest.raises(ValueError):
            index.points.setflags(write=True)
        return original(index, rows)
    monkeypatch.setattr(modules.extrema._Index, "_build", observed)
    result = _call(modules, args)
    _complete(result)
    assert len(snapshots) == 3
    assert all(value is snapshots[0] for value in snapshots)
    assert set(supports) == {tuple(range(10)), (0, 1, 2, 3), (4, 5, 6, 7)}


def test_late_index_query_failure_discards_completed_pose_prefix(modules, monkeypatch):
    original = modules.extrema._Index.query
    original_slots = modules.v3._slots
    count = 0
    slot_calls = 0
    def novel_origin(*args, **kwargs):
        nonlocal slot_calls
        slot_calls += 1
        if slot_calls == 2:
            args = list(args)
            args[3] = (args[3][0] + np.array((0., 0., .125)), args[3][1])
        return original_slots(*args, **kwargs)
    def broken(index, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 4:
            raise modules.extrema._Unavailable("INVENTED_LATE_QUERY_FAILURE")
        return original(index, *args, **kwargs)
    monkeypatch.setattr(modules.extrema._Index, "query", broken)
    monkeypatch.setattr(modules.v3, "_slots", novel_origin)
    result = _call(modules, _case(modules))
    assert count == 4
    _refused(result)
    assert result["discarded_signed_families"] >= 2
    assert result["discarded_candidates"] > 0


def test_small_chunk_does_not_redefine_fixed_index_leaf_semantics(modules, monkeypatch):
    original = modules.extrema._Index._evaluate
    sizes = []
    def observed(index, ids, *args, **kwargs):
        sizes.append(len(ids))
        return original(index, ids, *args, **kwargs)
    monkeypatch.setattr(modules.extrema._Index, "_evaluate", observed)
    result = _call(modules, _case(modules), chunk_rows=3)
    _complete(result)
    assert modules.extrema.LEAF_ROWS == 256
    assert sizes and 3 < max(sizes) <= 256
    _assert_projection_oracle(_case(modules), result)


def test_large_connected_context_preserves_every_invented_raw_row(modules):
    args = _case(modules)
    points = np.empty((900_000, 3), dtype=np.float64)
    points[:, 0] = 0.
    points[:, 1] = 0.
    points[:, 2] = 4.
    points[:10] = args["scan_points"]
    points[456_789] = (0., 0., -27.)
    points[876_543] = (0., 0., 39.)
    args["scan_points"] = points
    result = _call(modules, args)
    _complete(result)
    assert result["input_counts"]["scan_raw_rows"] == 900_000
    assert result["input_digests"]["scan_points_sha256"] == _sha(points)
    whole = [p for p in result["context_projections"] if p["role"] == "scan" and p["scope"] == "WHOLE_SUPPLIED_BUFFER"]
    assert len(whole) == 1
    for record in whole:
        assert record["projection_direction"][:2] == [0., 0.]
        sign = record["projection_direction"][2]
        assert sign in (-1., 1.)
        assert record["interval_mm"] == ([-27., 39.] if sign == 1. else [-39., 27.])
        assert record["minimum_witness"]["raw_vertex_row"] == (456_789 if sign == 1. else 876_543)
        assert record["maximum_witness"]["raw_vertex_row"] == (876_543 if sign == 1. else 456_789)
    assert result["statistics"]["scan_row_visits"] >= 2_700_000
    assert result["statistics"]["scan_row_visits"] < 3_000_000
    stats = result["statistics"]
    index_work = stats["extrema_work"]
    assert stats["extrema_indexed_rows"] == 900_008
    assert stats["extrema_completed_indexes"] == 3
    assert index_work["row_visits"] == 2*900_008 + index_work["query_rows_evaluated"]
    assert index_work["query_rows_evaluated"] < 2048
    assert index_work["coordinate_visits"] == 19*900_008 + 9*index_work["query_rows_evaluated"]
    assert index_work["hash_bytes"] == 8*900_008
    assert stats["accounted_working_peak_bytes"] < 268_435_456
    capacity = dict(record="CONNECTED_CONTEXT_V3_INVENTED_900K_CAPACITY", raw_rows=900_000,
        assigned_support_sizes=[4, 4], ordered_matches=1, signed_families=4, slots=32,
        scope="CONNECTED_FULL_BUFFER_CAPACITY_NOT_DENSE_PLANE_OR_FULL_WORKFLOW_CAPACITY",
        raw_sha256=result["input_digests"]["scan_points_sha256"], statistics=stats,
        whole_projection=whole, indexes=result["scan_extrema_indexes"])
    encoded = json.dumps(capacity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert len(encoded.encode()) <= 8192
    print("TWO_PLANE_CONTEXT_V3_CAPACITY_JSON="+encoded)


def _registry_case(m, args=None, **caps):
    """Private connected seam after original inputs/plane admission, no standalone init."""
    args = _case(m) if args is None else args
    cap = m.v3._configuration(m.v3.TwoPlaneContextBudgetsV3(**caps))
    work, result = m.v3._ContextWork(cap), m.v3._base()
    context = m.v3._inputs(*(args[name] for name in ("scan_points", "scan_triangles",
        "cad_triangles", "cad_face_ids", "cad_source_triangle_ids")), work, result)
    scans = m.v3._planes(args["scan_planes"], "scan", context, work, result)
    m.v3._planes(args["cad_planes"], "cad", context, work, result)
    def project(plane, direction=(0., 0., 1.), origin=(0., 0., 0.), constant=0.):
        return m.v3._projection("scan", plane, np.asarray(direction), np.asarray(origin),
            constant, args["source_frame"], context, work, result)
    return types.SimpleNamespace(args=args, context=context, work=work, result=result,
        scans=scans, project=project)


def _immutable_rows(values):
    return np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


def _support_plane(rows):
    return dict(raw_vertex_rows=rows, raw_vertex_rows_sha256=_sha(rows))


def test_exact_support_reuse_and_distinct_equal_size_memberships(modules):
    h = _registry_case(modules)
    rows_a = _immutable_rows((0, 1, 2, 3))
    rows_b = _immutable_rows((0, 1, 2, 3))
    rows_c = _immutable_rows((4, 5, 6, 7))
    # Different projection directions force registry lookup past projection cache.
    h.project(None)
    h.project(_support_plane(rows_a), (1., 0., 1.))
    h.project(_support_plane(rows_b), (0., 1., 1.))
    h.project(_support_plane(rows_c), (1., 1., 1.))
    stats = h.work.stats
    assert stats["extrema_indexes"] == stats["extrema_completed_indexes"] == 3
    assert stats["extrema_indexed_rows"] == 18
    assert stats["extrema_registry_hits"] == 1
    assert stats["extrema_alias_comparison_rows"] == 4
    ids = [p["scan_extrema_index_id"] for p in h.result["context_projections"]]
    assert ids[1] == ids[2] and ids[0] != ids[1] != ids[3]
    assert len({id(index.points) for index in h.work.registry.indices.values()}) == 1


def test_whole_and_explicit_full_support_preserve_distinct_scope(modules):
    h = _registry_case(modules)
    rows = _immutable_rows(range(10))
    h.project(None)
    h.project(_support_plane(rows))
    assert h.work.stats["extrema_indexes"] == 2
    assert h.work.stats["extrema_indexed_rows"] == 20
    records = h.result["scan_extrema_indexes"]
    assert records[0]["support_rows_sha256"] == records[1]["support_rows_sha256"]
    assert records[0]["support_scope"] != records[1]["support_scope"]
    assert h.result["context_projections"][0]["interval_mm"] == h.result["context_projections"][1]["interval_mm"]


@pytest.mark.parametrize("cache_hit", (False, True))
def test_digest_collision_is_not_accepted_as_support_equality(modules, cache_hit):
    h = _registry_case(modules)
    rows_a = _immutable_rows((0, 1, 2, 3))
    rows_b = _immutable_rows((4, 5, 6, 7))
    h.project(_support_plane(rows_a), (1., 0., 1.))
    forged = dict(raw_vertex_rows=rows_b, raw_vertex_rows_sha256=_sha(rows_a))
    with pytest.raises(modules.v1._Unavailable, match="EXTREMA_SUPPORT_DIGEST_COLLISION"):
        h.project(forged, (1., 0., 1.) if cache_hit else (0., 1., 1.))
    assert h.work.registry.failed is True
    assert h.work.stats["extrema_completed_indexes"] == 1


@pytest.mark.parametrize("direction,origin,constant", (
    ((1., 1., 1.), (0., 0., 0.), 0.),
    ((.25, -.5, .75), (17.125, -8.25, 3.5), -2.75),
    ((0., -0., 0.), (1e6, -1e6, 0.), -0.),
    ((1e-300, -1e-300, 1e-300), (0., 0., 0.), 0.),
    ((1., -1., float.fromhex("0x1p-52")), (0., 0., 0.), float.fromhex("0x1p-53")),
))
def test_connected_bounds_and_raw_ties_match_independent_staged_oracle(modules, direction, origin, constant):
    args = _case(modules)
    points = np.empty((769, 3), dtype=np.float64)
    for row in range(769):
        points[row] = (((row*17) % 83)-41., ((row*23) % 97)-48., ((row*31) % 101)-50.)
    points[:10] = args["scan_points"]
    points[7] = points[700] = (-41., -48., -50.)
    points[9] = points[768] = (41., 48., 50.)
    args["scan_points"] = points
    h = _registry_case(modules, args, chunk_rows=3)
    h.project(None, direction, origin, constant)
    _assert_projection_oracle(args, h.result)
    index, = h.work.registry.indices.values()
    assert index.node_count == 8 and index.base == 4
    assert h.result["scan_extrema_indexes"][0]["leaf_rows"] == 256
    for node in range(1, index.node_count):
        ids = index.rows[int(index.start[node]):int(index.end[node])]
        if not len(ids):
            continue
        lower, upper = index._bounds(node, np.asarray(direction), np.asarray(origin), constant)
        actual = [_scalar_projection(points[row], direction, origin, constant) for row in ids]
        assert lower <= min(actual) <= max(actual) <= upper
    assert h.work.stats["extrema_work"]["query_rows_evaluated"] <= 769+7


def test_cumulative_constructor_query_work_is_not_reset_per_support(modules):
    h = _registry_case(modules)
    h.project(None)
    h.project(h.scans[31])
    h.project(h.scans[32])
    stats, w = h.work.stats, h.work.index_stats
    memberships = 18
    assert stats["extrema_indexes"] == stats["extrema_completed_indexes"] == 3
    assert stats["extrema_indexed_rows"] == memberships
    assert w["queries"] == stats["extrema_queries"] == 3
    assert w["row_visits"] == 2*memberships+w["query_rows_evaluated"]
    assert w["index_visits"] == 5*memberships+w["query_rows_evaluated"]
    assert w["coordinate_visits"] == 19*memberships+9*w["query_rows_evaluated"]
    assert w["hash_bytes"] == 8*memberships
    assert w["sort_work"] == 10*4+2*(4*2)
    assert w["owned_bytes"] == 8*memberships+3*(2*72)
    assert stats["scan_row_visits"] == 10+w["row_visits"]
    assert stats["index_visits"] > w["index_visits"]
    assert stats["hash_bytes"] > w["hash_bytes"]+int(h.args["scan_points"].nbytes)


@pytest.mark.parametrize("kind", ("cad_temporary", "proposal_record"))
def test_later_parent_allocations_include_all_live_registry_bytes(modules, kind):
    h = _registry_case(modules)
    h.project(None)
    h.project(h.scans[31])
    h.project(h.scans[32])
    w = h.work
    parent = sum(w.stats[name] for name in ("owned_array_bytes", "temporary_bytes",
        "fixed_scratch_bytes", "cache_payload_bytes", "proposal_payload_bytes"))
    registry = w.index_stats["owned_bytes"]+w.stats["extrema_registry_metadata_bytes"]+65_536
    assert registry > 0
    if kind == "cad_temporary":
        temporary = 4*320
        w.cap["max_working_bytes"] = parent+registry+temporary-1
        with pytest.raises(modules.v1._Unavailable, match="ACCOUNTED_WORKING_PAYLOAD_BUDGET"):
            modules.v3._projection("cad", None, np.array((1., 0., 0.)), np.zeros(3), 0.,
                "invented_cad_late", h.context, w, h.result)
        assert w.stats["temporary_bytes"] == 0
    else:
        record = dict(invented="later_parent_record"*80)
        size = len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        w.cap["max_working_bytes"] = parent+registry+size-1
        with pytest.raises(modules.v1._Unavailable, match="ACCOUNTED_WORKING_PAYLOAD_BUDGET"):
            w.record(record)
    assert w.stats["accounted_working_peak_bytes"] > w.cap["max_working_bytes"]


@pytest.mark.parametrize("field,value,reason", (
    ("max_extrema_indexes", 2, "BUDGET_EXTREMA_INDEXES"),
    ("max_extrema_indexed_rows", 17, "BUDGET_EXTREMA_INDEXED_ROWS"),
    ("max_extrema_queries", 2, "BUDGET_EXTREMA_QUERIES"),
    ("max_extrema_coordinate_visits", 0, "BUDGET_EXTREMA_COORDINATE_VISITS"),
    ("max_extrema_node_visits", 0, "BUDGET_EXTREMA_NODE_VISITS"),
))
def test_new_cumulative_guards_fail_closed(modules, field, value, reason):
    result = _call(modules, _case(modules), **{field: value})
    _refused(result, reason)
    assert result["statistics"][field[4:]] > value


@pytest.mark.parametrize("field,maximum", (("max_extrema_indexes", 129),
    ("max_extrema_indexed_rows", 4_000_000), ("max_extrema_queries", 3072),
    ("max_extrema_coordinate_visits", 256_000_000), ("max_extrema_node_visits", 16_000_000)))
@pytest.mark.parametrize("bad", ("negative", "overflow", "boolean"))
def test_new_guard_values_are_exact_bounded_integers(modules, field, maximum, bad):
    value = -1 if bad == "negative" else maximum+1 if bad == "overflow" else True
    _refused(_call(modules, _case(modules), **{field: value}), "INVALID_BUDGET_VALUES")


def test_context_inherits_qualified_geometry_and_original_admission_helpers(modules):
    for name in ("_build", "_bounds", "_evaluate", "query"):
        assert getattr(modules.v3._ContextIndex, name) is getattr(modules.extrema._Index, name)
    for name in ("_inputs", "_planes", "_schedule", "_family", "_SLOTS", "_GATES"):
        assert getattr(modules.v3, name) is getattr(modules.v1, name)
    assert modules.v3._MAXIMA[:len(modules.v2._MAXIMA)] == modules.v2._MAXIMA
    assert modules.v3._base()["scan_projection_chunk_scope"] == "FIXED_EXTREMA_LEAF256_NOT_CHUNK_ROWS"
    tree = ast.parse((ROOT/"backend"/"cad_guided_two_plane_context_v3.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "project_raw_extrema_v1" for node in calls)


@pytest.mark.parametrize("field,charge", (("max_scan_row_visits", "row_visits"),
    ("max_index_visits", "index_visits"), ("max_index_sort_work", "sort_work"),
    ("max_hash_bytes", "hash_bytes")))
def test_registry_work_hits_original_parent_caps_cumulatively(modules, field, charge):
    h = _registry_case(modules)
    h.project(None)
    before = h.work.stats[field[4:]]
    earlier_index_charge = h.work.index_stats[charge]
    assert earlier_index_charge > 0
    h.work.cap[field] = before
    with pytest.raises(modules.v1._Unavailable, match="BUDGET_"+field[4:].upper()):
        h.project(h.scans[31], (1., 0., 1.))
    assert h.work.stats[field[4:]] > before
    assert h.work.index_stats[charge] > earlier_index_charge
    assert h.work.registry.failed is True


def test_query_cap_cannot_exceed_three_times_lowered_family_cap(modules):
    result = modules.v3.propose_two_plane_context_v3(**_case(modules),
        budgets=modules.v3.TwoPlaneContextBudgetsV3(max_signed_families=4, max_extrema_queries=13))
    _refused(result, "INVALID_BUDGET_VALUES")


@pytest.mark.parametrize("site", ("late_cad", "late_family_record", "final_report"))
def test_late_parent_failure_clears_indexes_and_all_pose_evidence(modules, monkeypatch, site):
    if site == "late_cad":
        original = modules.v3._projection
        def broken(role, plane, direction, origin, constant, frame, context, work, result):
            if role == "cad" and work.stats["signed_families_completed"] >= 1:
                raise modules.v1._Unavailable("INVENTED_LATE_PARENT_FAILURE")
            return original(role, plane, direction, origin, constant, frame, context, work, result)
        monkeypatch.setattr(modules.v3, "_projection", broken)
    elif site == "late_family_record":
        original = modules.v3._ContextWork.record
        def broken(work, value):
            if isinstance(value, dict) and "signed_family_id" in value and "slots" in value:
                raise modules.v1._Unavailable("INVENTED_LATE_PARENT_FAILURE")
            return original(work, value)
        monkeypatch.setattr(modules.v3._ContextWork, "record", broken)
    else:
        def broken(*args, **kwargs):
            raise modules.v1._Unavailable("INVENTED_LATE_PARENT_FAILURE")
        monkeypatch.setattr(modules.v3, "_final_report", broken)
    result = _call(modules, _case(modules))
    _refused(result, "INVENTED_LATE_PARENT_FAILURE")
    assert result["discarded_candidates"] > 0
    assert result["statistics"]["extrema_completed_indexes"] == 3


@pytest.mark.parametrize("refusal", (False, True))
def test_public_completion_and_refusal_release_registry_owner_cycles(modules, monkeypatch, refusal):
    original = modules.v3._ContextIndex._build
    registries, indexes = [], []
    def observed(index, rows):
        registries.append(index.registry)
        indexes.append(index)
        return original(index, rows)
    monkeypatch.setattr(modules.v3._ContextIndex, "_build", observed)
    if refusal:
        def broken(*args, **kwargs):
            raise modules.v1._Unavailable("INVENTED_RELEASE_REFUSAL")
        monkeypatch.setattr(modules.v3, "_final_report", broken)
    result = _call(modules, _case(modules))
    if refusal:
        _refused(result, "INVENTED_RELEASE_REFUSAL")
    else:
        _complete(result)
    assert len(indexes) == 3
    assert all(index.valid is False for index in indexes)
    for registry in registries:
        assert registry.failed is True
        assert registry.indices == registry.aliases == {}
        assert registry.points is registry.work is registry.result is None


def test_projection_cache_reuses_value_but_authenticates_new_equal_support_alias(modules):
    h = _registry_case(modules)
    rows_a, rows_b = _immutable_rows((0, 1, 2, 3)), _immutable_rows((0, 1, 2, 3))
    first = h.project(_support_plane(rows_a))
    second = h.project(_support_plane(rows_b))
    assert first == second
    assert h.work.stats["extrema_completed_indexes"] == 1
    assert h.work.stats["extrema_alias_comparison_rows"] == 4
    assert h.work.stats["extrema_queries"] == 1
    assert h.work.stats["projection_cache_hits"] == 1
