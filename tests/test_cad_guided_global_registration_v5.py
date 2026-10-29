"""Registration V5: chunked cooperative surface queries (exact merge), bound query reuse, deterministic lane
quotas and the declared CPU guard. Invented finite meshes only; no backend startup, no physical data."""
from dataclasses import replace
import importlib
import json
import itertools
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def m():
    saved = {k: v for k, v in sys.modules.items() if k == "backend" or k.startswith("backend.")}
    for k in saved:
        del sys.modules[k]
    package = types.ModuleType("backend"); package.__path__ = [str(ROOT/"backend")]; sys.modules["backend"] = package
    try:
        yield types.SimpleNamespace(v5=importlib.import_module("backend.cad_guided_global_registration_v5"),
                                    v4=importlib.import_module("backend.cad_guided_global_registration_v4"),
                                    surface=importlib.import_module("backend.cad_guided_triangle_surface_v4"))
    finally:
        for k in tuple(sys.modules):
            if k == "backend" or k.startswith("backend."):
                del sys.modules[k]
        sys.modules.update(saved)


def _mesh(kind, seed=0):
    """Independent finite triangle soups: a box with a hole, a stepped multi-face plate with ties, and an
    awkwardly scaled thin strip. Face ids are strings so identity binding is exercised."""
    rng = np.random.default_rng(seed)
    triangles, faces = [], []
    def quad(a, b, c, d, face):
        triangles.extend([[a, b, c], [a, c, d]]); faces.extend([face, face])
    if kind == "box_with_hole":
        for z, face in ((0., "bottom"), (3., "top")):
            for (x0, x1) in ((0., 2.), (4., 6.)):           # hole between x 2 and 4 on both plates
                quad([x0, 0., z], [x1, 0., z], [x1, 6., z], [x0, 6., z], face)
        for y, face in ((0., "front"), (6., "back")):
            quad([0., y, 0.], [6., y, 0.], [6., y, 3.], [0., y, 3.], face)
    elif kind == "stepped_plate":
        for i in range(6):
            quad([i, 0., .25*i], [i+1., 0., .25*i], [i+1., 4., .25*i], [i, 4., .25*i], "step-%d" % i)
        quad([0., 0., 5.], [6., 0., 5.], [6., 4., 5.], [0., 4., 5.], "cap")        # symmetric competitor for ties
    else:
        scale = 1e-3 if kind == "tiny_strip" else 250.
        for i in range(4):
            quad([i*scale, 0., 0.], [(i+1)*scale, 0., 0.], [(i+1)*scale, .2*scale, 0.], [i*scale, .2*scale, 0.], "strip-%d" % i)
    tri = np.asarray(triangles, float)
    return tri, tuple(faces)


def _index(m, kind):
    tri, faces = _mesh(kind)
    built = m.surface.build_triangle_surface(tri, list(faces))
    assert built["status"] == "COMPLETE", built
    return built["index"], tri


def _rows(kind, count, seed=1):
    rng = np.random.default_rng(seed)
    if kind == "tiny_strip":
        base = rng.uniform([-1e-3, -1e-3, -1e-3], [5e-3, 1.2e-3, 1e-3], (count, 3))
    elif kind == "huge_strip":
        base = rng.uniform([-100., -50., -50.], [1100., 100., 50.], (count, 3))
    else:
        base = rng.uniform([-1., -1., -1.], [7., 7., 6.], (count, 3))
    # threshold-near rows: exactly on faces and exactly half a gate away
    base[:8] = base[:8].round(2)
    return base


@pytest.mark.parametrize("kind", ["box_with_hole", "stepped_plate", "tiny_strip", "huge_strip"])
@pytest.mark.parametrize("chunk", [1024, 1500])
def test_chunked_queries_are_byte_identical_to_whole_queries_and_charge_the_same_rows(m, kind, chunk):
    index, _ = _index(m, kind)
    points = _rows(kind, 4100)
    whole = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4())
    chunked = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4(), chunk_rows=chunk)
    a, b = whole(points), chunked(points)
    for key in m.surface.RESULT_ARRAY_KEYS:
        if key == "source_face_ids":
            assert a[key] == b[key]
        else:   # literal byte identity (signed zeros included), not only value equality
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes(), key
    for key in m.v4._QUERY_METADATA:
        assert a[key] == b[key]
    da, db = a["diagnostics"], b["diagnostics"]
    assert db["passes_completed"] == 2 and db["certificate_failures"] == 0
    assert db["work_units"] == (db["node_visits"]+db["triangle_box_pairs"]+8*db["triangle_pairs"]+db["certificate_pairs"]
                                +3*db["normal_pairs"]+m.surface.VERTEX_LOOKUP_WORK_UNITS*db["vertex_lookup_rows"])
    assert db["certificate_pairs"] == 4*db["triangle_pairs"] and db["triangle_pairs"] >= 2*len(points)
    assert whole.statistics["query_rows"] == chunked.statistics["query_rows"] == len(points)
    assert whole.statistics["query_calls"] == 1 and chunked.statistics["query_calls"] == -(-len(points)//chunk)
    assert chunked.statistics["chunked_query_calls"] == 1 and chunked.statistics["chunk_calls"] == chunked.statistics["query_calls"]
    small = _rows(kind, 700)
    assert chunked.statistics["chunked_query_calls"] == 1 and np.array_equal(chunked(small)["distances"], whole(small)["distances"])
    assert chunked.statistics["chunked_query_calls"] == 1   # a query at or below the chunk size is not chunked


def _fake_clock(cpu_per_call=1., wall_per_call=1., start=0.):
    state = dict(cpu=start, wall=start)
    def clock():
        state["cpu"] += cpu_per_call; state["wall"] += wall_per_call
        return state["cpu"], state["wall"]
    clock.state = state
    return clock


def test_allowance_validation_and_deterministic_quota_semantics(m):
    A = m.v5.RefinementAllowanceV1
    a = A()
    assert a.group_quota("main") == 12 and a.group_quota("completion") == 4 and a.row_quota("completion") == 3_000_000
    assert a.headroom_seconds() == pytest.approx(360.)
    for bad in (dict(enabled=1), dict(main_refined_groups=0), dict(main_refined_groups=49), dict(completion_query_rows=-1),
                dict(cooperative_chunk_rows=100), dict(cpu_guard_seconds=1860.), dict(cpu_budget_seconds=float("nan")),
                dict(cpu_guard_seconds=True)):
        with pytest.raises(ValueError):
            A(**bad)
    meter = m.v5._CpuMeter(A(main_query_rows=5000, completion_query_rows=2000, cpu_guard_seconds=100., cpu_budget_seconds=200.),
                           clock=_fake_clock(cpu_per_call=0.))
    assert meter.exhaustion("main") is None and meter.exhaustion("completion", rows_about_to_query=2000) is None
    assert meter.exhaustion("completion", rows_about_to_query=2001) == "LANE_WORK_QUOTA:completion"
    meter.charge_rows("main", 4000)
    assert meter.exhaustion("main", rows_about_to_query=1000) is None and meter.exhaustion("main", rows_about_to_query=1001) == "LANE_WORK_QUOTA:main"
    assert meter.lane_rows == dict(main=4000, completion=0) and meter.lane_calls == dict(main=1, completion=0)
    disabled = m.v5._CpuMeter(A(enabled=False, main_query_rows=1), clock=_fake_clock(cpu_per_call=1000.))
    assert disabled.exhaustion("main", rows_about_to_query=10**6) is None
    assert m.v5._CpuMeter(None).exhaustion("main") is None


def test_cpu_guard_is_absolute_process_cpu_and_declared_timing_dependent(m):
    clock = _fake_clock(cpu_per_call=0., start=1499.)
    meter = m.v5._CpuMeter(m.v5.RefinementAllowanceV1(), clock=clock)
    assert meter.exhaustion("main") is None
    clock.state["cpu"] = 1500.01                         # imports/preparation count: the guard reads absolute process CPU
    assert meter.exhaustion("main") == "CPU_GUARD" and meter.guard_tripped is True
    summary = meter.summary()
    assert summary["cpu_guard_tripped"] is True and summary["timing_dependence"].startswith("CPU_GUARD_STOPS_ARE_TIMING_DEPENDENT")


def test_meter_sections_accumulate_lanes_but_not_stage_sections_and_never_nest(m):
    clock = _fake_clock(cpu_per_call=2., wall_per_call=3.)
    meter = m.v5._CpuMeter(None, clock=clock)
    meter.begin("main", "REFINE_CANDIDATE_0"); receipt = meter.end(candidate=0)
    assert receipt["cpu_seconds"] == pytest.approx(2.) and receipt["wall_seconds"] == pytest.approx(3.) and receipt["candidate"] == 0
    meter.begin("stage", "PREPARATION"); meter.end()
    assert meter.lane_cpu["main"] == pytest.approx(2.) and meter.lane_cpu["completion"] == 0. and len(meter.receipts) == 2
    meter.begin("completion", "COMPLETION_PREFILTER")
    with pytest.raises(ValueError, match="INVALID_METER_SECTION"):
        meter.begin("main", "nested")
    meter.end()
    with pytest.raises(ValueError, match="NO_ACTIVE_METER_SECTION"):
        meter.end()


def test_typed_allowance_exhaustion_stops_before_and_between_chunks_and_discards_partial_work(m):
    index, _ = _index(m, "box_with_hole")
    points = _rows("box_with_hole", 3000)
    allowance = m.v5.RefinementAllowanceV1(main_query_rows=2500, cooperative_chunk_rows=1024)
    meter = m.v5._CpuMeter(allowance, clock=_fake_clock(cpu_per_call=0.))
    query = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4(),
                            meter=meter, chunk_rows=1024)
    meter.begin("main", "REFINE_CANDIDATE_0")
    with pytest.raises(m.v5._AllowanceExhausted, match="LANE_WORK_QUOTA:main"):
        query(points)                                        # 1024 + 1024 charged, the third chunk (952) exceeds 2500
    assert meter.lane_rows["main"] == 2048 and query.statistics["allowance_stops"] == 1
    assert query.statistics["chunk_calls"] == 2 and query.statistics["query_calls"] == 2
    assert not isinstance(m.v5._AllowanceExhausted("x"), m.v4._SurfaceFailure)   # typed, never a surface failure
    meter.end()
    guard = m.v5._CpuMeter(m.v5.RefinementAllowanceV1(cpu_guard_seconds=10., cpu_budget_seconds=20.),
                           clock=_fake_clock(cpu_per_call=6.))
    query2 = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4(), meter=guard, chunk_rows=1024)
    guard.begin("main", "REFINE_CANDIDATE_1")
    with pytest.raises(m.v5._AllowanceExhausted, match="CPU_GUARD"):
        query2(points)
    guard.end()


def _policy(m, **changes):
    fields = dict(coarse_points=50, refine_points=50, coarse_iterations=1, refine_iterations=1,
                  max_correspondence_distance_mm=.5, huber_delta_mm=.05, max_inlier_plane_p95_mm=.1,
                  min_scan_coverage=.5, min_cad_coverage=.1)
    fields.update(changes)
    return m.v4._v1.WholePartRegistrationPolicyV1(**fields)


def test_diagnosis_query_is_reused_exactly_for_context_support_and_any_changed_binding_requeries(m):
    index, tri = _index(m, "box_with_hole")
    scan = _rows("box_with_hole", 2000)
    cad = tri.reshape(-1, 3)
    refine = scan[:50]
    r, t = np.eye(3), np.zeros(3)
    policy = _policy(m)
    query = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4(), chunk_rows=1024)
    diagnosis, reusable = m.v5._diagnose_with_reusable_query(scan, cad, refine, query, r, t, policy)
    reference = m.v4._diagnose(scan, cad, refine, query, r, t, policy)
    assert {k: v for k, v in diagnosis.items() if not isinstance(v, (list, np.ndarray))} == \
           {k: v for k, v in reference.items() if not isinstance(v, (list, np.ndarray))}
    assert reusable is not None and reusable.rows == len(scan) and json.loads(reusable.payload.metadata_json)["status"] == "COMPLETE"
    calls_before = dict(query.statistics)
    support = m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)
    assert support["query_reuse"] == "REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING"
    assert query.statistics["query_calls"] == calls_before["query_calls"] and query.statistics["reused_query_calls"] == 1
    assert query.statistics["reused_query_rows"] == len(scan)
    performed = m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], None)
    assert performed["query_reuse"] == "PERFORMED_NEW_QUERY"
    for key in ("status", "reasons", "geometric_scan_coverage_lower", "geometric_inlier_count", "row_counts", "finite_distance_metrics"):
        assert support[key] == performed[key], key
    # Any changed binding requeries: pose, rows, query object / index, policy gate, incomplete status.
    other_pose = m.v5._context_support_v5(scan, query, r, t+1e-9, policy, diagnosis["cad_coverage"], reusable)
    assert other_pose["query_reuse"] == "PERFORMED_NEW_QUERY"
    assert m.v5._context_support_v5(scan[:-1], query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    permuted = scan[::-1].copy()
    assert m.v5._context_support_v5(permuted, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    other_query = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4())
    assert m.v5._context_support_v5(scan, other_query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    assert m.v5._context_support_v5(scan, query, r, t, _policy(m, max_correspondence_distance_mm=.4), diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    # Same query object, GENUINELY different valid index (other faces/normals/triangles): an independent key
    # dimension -> a legitimate requery against the new index, never a fatal mismatch of the retained record.
    other_index, _ = _index(m, "stepped_plate")
    original_index = query.index
    query.index = other_index
    replaced = m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)
    assert replaced["query_reuse"] == "PERFORMED_NEW_QUERY" and replaced["reuse_binding"]["rows"] == len(scan)
    rebuilt_same, _ = _index(m, "box_with_hole")            # an identical rebuild is still a different object: requery
    query.index = rebuilt_same
    assert m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    query.index = original_index
    assert m.v5.REGISTRATION_IMPLEMENTATION.endswith("CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1")
    # Retained-evidence INTEGRITY failures are fatal (never silently recomputed, no reuse or fresh query):
    # a status change, a changed array, a changed shape, changed diagnostics, a foreign record type.
    def fatal(record):
        before = dict(query.statistics)
        with pytest.raises(m.v5._SurfaceFailure, match="REUSABLE_QUERY"):
            m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], record)
        assert query.statistics["query_calls"] == before["query_calls"] and query.statistics["reused_query_calls"] == before["reused_query_calls"]
    def record(**fields):
        return m.v5._ReusableQuery(**{**{f: getattr(reusable, f) for f in reusable.__dataclass_fields__}, **fields})
    payload = reusable.payload
    stale = record(payload=replace(payload, metadata_json=payload.metadata_json.replace("COMPLETE", "UNAVAILABLE")))
    fatal(stale)
    arrays = dict(payload.arrays)
    # The owned arrays are read-only views over private bytes: element writes AND re-enabling writes both raise.
    assert all(v.flags.writeable is False for k, v in payload.arrays if k != "source_face_ids")
    with pytest.raises(ValueError):
        arrays["distances"][0] = 123.
    with pytest.raises(ValueError):
        arrays["distances"].setflags(write=True)
    changed = np.array(arrays["distances"], copy=True); changed[0] += 1.
    fatal(record(payload=replace(payload, arrays=tuple((k, changed if k == "distances" else v) for k, v in payload.arrays))))
    same_bytes_other_shape = np.frombuffer(arrays["distances"].tobytes(), dtype=arrays["distances"].dtype).reshape(1, -1)
    fatal(record(payload=replace(payload, arrays=tuple((k, same_bytes_other_shape if k == "distances" else v) for k, v in payload.arrays))))
    fatal(record(payload=replace(payload, diagnostics_json=payload.diagnostics_json[:-1]+",\"injected\":1}")))
    fatal(record(payload=replace(payload, rows=payload.rows-1)))
    with pytest.raises(m.v5._SurfaceFailure, match="REUSABLE_QUERY_RECORD_INVALID"):
        m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], dict(reusable.__dict__))
    # The retained record is unchanged by the fatal probes and still reuses exactly.
    assert m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING"
    # Policy mutation WITHOUT replacing the query object evades an id-only key but not the policy snapshot.
    original_budgets = query.surface_budgets
    query.surface_budgets = replace(original_budgets, max_pair_batch=original_budgets.max_pair_batch-1)   # still a valid query
    assert m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    query.surface_budgets = original_budgets
    original_execution = query.budgets
    query.budgets = replace(original_execution, max_query_calls=original_execution.max_query_calls-1)
    assert m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    query.budgets = original_execution
    query.chunk_rows = 2048
    assert m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"
    query.chunk_rows = 1024
    assert m.v5._context_support_v5(scan, query, r, t, policy, diagnosis["cad_coverage"], reusable)["query_reuse"] == "REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING"
    assert m.v5._context_support_v5(scan, query, r, t, _policy(m, min_scan_coverage=.55), diagnosis["cad_coverage"], reusable)["query_reuse"] == "PERFORMED_NEW_QUERY"


def test_attempted_rows_stay_charged_when_the_primitive_fails_and_completed_rows_do_not(m, monkeypatch):
    index, _ = _index(m, "box_with_hole")
    points = _rows("box_with_hole", 500)
    meter = m.v5._CpuMeter(m.v5.RefinementAllowanceV1(), clock=_fake_clock(cpu_per_call=0.))
    query = m.v5._QueriesV5(index, m.v4.SurfaceRegistrationBudgetsV4(), m.surface.TriangleSurfaceBudgetsV4(), meter=meter)
    meter.begin("main", "REFINE_CANDIDATE_0")
    query(points)
    assert meter.lane_rows["main"] == 500 == meter.lane_rows_completed["main"] and meter.lane_calls_completed["main"] == 1
    real = m.surface.query_closest_surface
    monkeypatch.setattr(m.surface, "query_closest_surface", lambda *a, **k: dict(status="UNAVAILABLE", reason="INJECTED_PRIMITIVE_REFUSAL",
        **{key: None for key in m.surface.RESULT_ARRAY_KEYS}, diagnostics=dict.fromkeys(m.v4._STAT_KEYS, 0)))
    with pytest.raises(m.v4._SurfaceFailure):
        query(points)
    assert meter.lane_rows["main"] == 1000 and meter.lane_rows_completed["main"] == 500     # attempted charged, not completed
    assert meter.lane_calls["main"] == 2 and meter.lane_calls_completed["main"] == 1
    summary = meter.summary()
    assert summary["lane_query_rows_attempted"]["main"] == 1000 and summary["lane_query_rows_completed"]["main"] == 500
