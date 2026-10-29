"""Independent composite-lane controls; invented arrays and explicit seam mocks.

The small square/corner cases use both actual producers and their unchanged
whole-support fits. Counter-injection cases are orchestration tests, not valid
scientific reports or geometry positives. No native, CAD, physical file, case
identity, datum, bend count or general surface-completeness claim is involved.
"""

from dataclasses import asdict, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module():
    # V7 imports V5 through the literal backend namespace. Keep exactly one
    # inert namespace so frozen dataclass identities agree, without __init__.
    def owned(key): return key == "backend" or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved: del sys.modules[key]
    package = types.ModuleType("backend")
    package.__path__ = [str(ROOT/"backend")]
    sys.modules["backend"] = package
    try:
        yield importlib.import_module("backend.cad_guided_scan_plane_preparation_v1")
    finally:
        for key in tuple(sys.modules):
            if owned(key): del sys.modules[key]
        sys.modules.update(saved)


def _square():
    return (np.array([[0., 0., 0.], [16., 0., 0.], [16., 16., 0.], [0., 16., 0.]]),
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))


def _corner():
    # Two real four-row orthogonal faces share an original edge. Neither the
    # unqualified V7 mixed support nor its rejection is removed by the seeder.
    return (np.array([[0., 0., 0.], [16., 0., 0.], [16., 16., 0.], [0., 16., 0.],
                      [0., 0., 16.], [16., 0., 16.]]),
            np.array([[0, 1, 2], [0, 2, 3], [0, 5, 1], [0, 4, 5]], dtype=np.int64))


def _sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _run(module, points, triangles, epsilon=.05, **kwargs):
    return module.prepare_composite_scan_planes_v1(points, triangles,
        position_error_bounds_mm=epsilon, **kwargs)


def _binding(points, triangles, epsilon):
    e = np.full(len(points), epsilon) if np.isscalar(epsilon) else epsilon
    return dict(points_sha256=_sha(points, "<f8"), triangle_raw_rows_sha256=_sha(triangles, "<i8"),
                position_error_bounds_mm_sha256=_sha(e, "<f8"))


def _record(lanes, lane, kind, index):
    table = ("cores" if kind == "CORE" else "closures") if lane == "v7" else "regions"
    return lanes[lane][table][index]


def _complete(module, result, points, triangles, epsilon=.05):
    assert result["status"] == "COMPLETE", result
    assert result["schema_version"] == "composite_scan_planes_v1"
    assert result["accepted_registration"] is result["production_use_allowed"] is False
    assert result["physical_plane_count"] is result["observed_bend_count"] is None
    assert result["search_exhaustive"] is result["raw_coordinate_welding"] is result["residual_trimming"] is False
    assert result["statistics"]["selected_policy_evaluation_complete"] is True
    assert result["lane_execution"] == dict(v7="COMPLETE", seeded_v1="COMPLETE")
    assert result["input_digests"] == _binding(points, triangles, epsilon)
    lanes = result["lanes"]
    assert list(lanes) == ["v7", "seeded_v1"]
    assert lanes["v7"]["schema_version"] == "noisy_mesh_planes_v7"
    assert lanes["seeded_v1"]["schema_version"] == "seeded_plane_proposals_v1"
    origins = []
    for lane in ("v7", "seeded_v1"):
        assert lanes[lane]["status"] == "COMPLETE" and lanes[lane]["input_digests"] == result["input_digests"]
        envelope = module._v7.encode_metadata_v7(lanes[lane], max_bytes=result["budgets"]["max_metadata_bytes"])
        blob = json.dumps(envelope, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        assert result["lane_metadata_sha256"][lane] == hashlib.sha256(blob).hexdigest()
        origins.extend((lane, ref["kind"], ref["record_id"]) for ref in lanes[lane]["proposals"])
    assert [(p["lane"], p["kind"], p["record_id"]) for p in result["planes"]] == origins
    eps = np.full(len(points), epsilon) if np.isscalar(epsilon) else epsilon
    qualified_triangles = qualified_rows = 0
    for plane_id, entry in enumerate(result["planes"]):
        assert entry["plane_id"] == plane_id
        record = _record(lanes, entry["lane"], entry["kind"], entry["record_id"])
        assert record["analysis"]["status"] == "PROPOSED"
        raw, tri = record["raw_vertex_rows"], record["triangle_rows"]
        assert np.array_equal(raw, np.unique(triangles[tri].ravel()))
        assert entry["triangle_rows_sha256"] == record["triangle_rows_sha256"] == _sha(tri, "<i8")
        assert entry["raw_vertex_rows_sha256"] == record["raw_vertex_rows_sha256"] == _sha(raw, "<i8")
        for array in (raw, tri):
            assert not array.flags.writeable
            with pytest.raises(ValueError): array.setflags(write=True)
        analysis = record["analysis"]
        assert analysis["witness_status"] == "FOUND"
        assert analysis["normal_angle_bound_radians"] <= math.pi/18.
        assert analysis["numerical"]["position_error_bound_was_increased"] is False
        residual = np.abs((points[raw]-analysis["witness_anchor_mm"])@analysis["normal"]
                          - analysis["witness_offset_from_anchor_mm"])
        assert np.all(residual <= eps[raw]+1e-12)  # Dot-comparison roundoff only; epsilon is unchanged.
        qualified_triangles += len(tri)
        qualified_rows += len(raw)
    a, b, totals = lanes["v7"]["statistics"], lanes["seeded_v1"]["statistics"], result["statistics"]
    for field in ("triangle_memberships", "raw_memberships", "full_support_row_tests",
                  "fit_evaluations", "retained_array_bytes"):
        assert totals[field] == a[field]+b[field]
    assert totals["raw_predicate_visits"] == a["cone_row_tests"]+b["proposal_raw_row_visits"]
    assert totals["triangle_predicate_visits"] == a["closure_triangle_tests"]+b["proposal_triangle_visits"]+b["seed_selection_triangle_visits"]
    assert totals["edge_predicate_visits"] == a["closure_edge_tests"]+b["proposal_edge_visits"]
    assert totals["qualified_plane_count"] == len(origins)
    assert totals["qualified_triangle_memberships"] == qualified_triangles
    assert totals["qualified_raw_memberships"] == qualified_rows
    assert totals["triangle_memberships"] >= qualified_triangles
    assert totals["raw_memberships"] >= qualified_rows


def _failed(result):
    assert result["status"] == "UNAVAILABLE", result
    assert result["lanes"] == {} and result["planes"] == []
    assert result["statistics"]["selected_policy_evaluation_complete"] is False
    assert result["statistics"]["evidence_scope"] == "ATTEMPTED_WORK_ONLY_COMPOSITE_INVALIDATED_WHOLE"
    assert result["accepted_registration"] is result["production_use_allowed"] is False


@pytest.mark.parametrize("pose,noisy", [(0, False), (1, False), (2, True)])
def test_actual_both_lanes_qualify_tiny_raw_plane_and_keep_all_namespaced_origins(module, pose, noisy):
    points, triangles = _square()
    rotations = [np.eye(3), np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]]),
                 np.array([[1., 8., 4.], [8., 1., -4.], [-4., 4., -7.]])/9.]
    rotation = rotations[pose]
    assert np.allclose(rotation.T@rotation, np.eye(3), atol=1e-15, rtol=0)
    assert abs(np.linalg.det(rotation)-1.) < 1e-15
    if noisy: points[:, 2] = [.001, -.001, .001, -.001]
    points = points@rotation.T+np.array([32., -16., 8.])
    result = _run(module, points, triangles)
    _complete(module, result, points, triangles)
    assert [(p["lane"], p["kind"], p["record_id"]) for p in result["planes"]] == [
        ("v7", "CORE", 0), ("v7", "CORE", 1), ("v7", "CLOSURE", 0), ("seeded_v1", "SEEDED_REGION", 0)]
    for entry in result["planes"]:
        analysis = _record(result["lanes"], entry["lane"], entry["kind"], entry["record_id"])["analysis"]
        target = rotation[:, 2]
        angle = math.atan2(np.linalg.norm(np.cross(analysis["normal"], target)),
                           abs(float(np.dot(analysis["normal"], target))))
        assert angle <= analysis["normal_angle_bound_radians"]+1e-10
    # Identical finite supports in distinct lanes are origins, not duplicate
    # physical planes to be merged or counted as manufacturing entities.
    assert len({p["plane_id"] for p in result["planes"]}) == 4


def test_actual_v7_rejection_tables_survive_successful_seeded_corner_lane(module, monkeypatch):
    points, triangles = _corner()
    returned = {}
    first, second = module._v7.extract_noisy_scan_plane_proposals_v7, module._seeded.extract_seeded_plane_proposals_v1
    def retain_first(*args, **kwargs):
        returned["v7"] = first(*args, **kwargs)
        return returned["v7"]
    def retain_second(*args, **kwargs):
        returned["seeded_v1"] = second(*args, **kwargs)
        return returned["seeded_v1"]
    monkeypatch.setattr(module._v7, "extract_noisy_scan_plane_proposals_v7", retain_first)
    monkeypatch.setattr(module._seeded, "extract_seeded_plane_proposals_v1", retain_second)
    result = _run(module, points, triangles)
    _complete(module, result, points, triangles)
    assert all(result["lanes"][key] is value for key, value in returned.items())
    assert result["lanes"]["v7"]["proposals"] == []
    assert result["lanes"]["v7"]["cores"] and all(
        row["analysis"]["status"] == "REJECTED" and row["analysis"]["reasons"]
        for row in result["lanes"]["v7"]["cores"])
    assert result["planes"] and all(row["lane"] == "seeded_v1" for row in result["planes"])
    assert result["statistics"]["triangle_memberships"] > result["statistics"]["qualified_triangle_memberships"]
    assert result["statistics"]["raw_memberships"] > result["statistics"]["qualified_raw_memberships"]


@pytest.mark.parametrize("corner", [False, True])
def test_explicit_zero_seed_budget_is_complete_with_or_without_v7_proposals(module, corner):
    points, triangles = _corner() if corner else _square()
    result = _run(module, points, triangles, seeded_budgets=module._seeded.SeededPlaneBudgetsV1(max_seeds=0))
    _complete(module, result, points, triangles)
    second = result["lanes"]["seeded_v1"]
    assert second["status"] == "COMPLETE" and second["scheduled_seed_triangle_rows"] == second["seeds"] == []
    assert second["statistics"]["selected_policy_evaluation_complete"] is True
    assert all(row["lane"] == "v7" for row in result["planes"])
    if corner:
        assert result["planes"] == [] and result["reasons"] == ["NO_QUALIFIED_COMPOSITE_SCAN_PLANES"]
    else: assert result["planes"]


def _tight(module):
    # Original square accounting: V7 has two cores and one unique closure;
    # seeder has one region plus one duplicate seed. No unreferenced raw rows.
    return module.CompositeScanPlaneBudgetsV1(max_qualified_planes=4, max_triangle_memberships=12,
        max_raw_memberships=16, max_full_support_row_tests=16, max_raw_predicate_visits=16,
        max_triangle_predicate_visits=12, max_edge_predicate_visits=4, max_fit_evaluations=4,
        max_retained_array_bytes=256)


def test_exact_independently_counted_shared_work_budget_fits_without_geometry_change(module):
    points, triangles = _square()
    result = _run(module, points, triangles, budgets=_tight(module))
    _complete(module, result, points, triangles)
    stats = result["statistics"]
    for key, value in asdict(_tight(module)).items():
        if key != "max_metadata_bytes":
            name = "qualified_plane_count" if key == "max_qualified_planes" else key[4:]
            assert stats[name] == value
    assert result["lanes"]["v7"]["statistics"]["retained_array_bytes"] == 208
    assert result["lanes"]["seeded_v1"]["statistics"]["retained_array_bytes"] == 48


@pytest.mark.parametrize("field", ["max_qualified_planes", "max_triangle_memberships", "max_raw_memberships",
    "max_full_support_row_tests", "max_raw_predicate_visits", "max_triangle_predicate_visits",
    "max_edge_predicate_visits", "max_fit_evaluations", "max_retained_array_bytes", "max_metadata_bytes"])
def test_shared_cap_one_under_or_zero_metadata_never_falls_back_to_one_lane(module, field):
    points, triangles = _square()
    budget = _tight(module)
    value = 0 if field == "max_metadata_bytes" else getattr(budget, field)-1
    result = _run(module, points, triangles, budgets=replace(budget, **{field: value}))
    _failed(result)


def _fake_lane(lane, points, triangles, epsilon, **overrides):
    """Deliberately synthetic counter receipt, NOT a valid geometry producer."""
    common = dict(triangle_memberships=0, raw_memberships=0, full_support_row_tests=0,
        fit_evaluations=0, retained_array_bytes=0, metadata_bytes=0)
    common.update(dict(cone_row_tests=0, closure_triangle_tests=0, closure_edge_tests=0)
                  if lane == "v7" else dict(proposal_raw_row_visits=0, proposal_triangle_visits=0,
                                           proposal_edge_visits=0, seed_selection_triangle_visits=0))
    common.update(overrides)
    return dict(schema_version="MOCK_"+lane, status="COMPLETE", reasons=[], statistics=common,
        input_digests=_binding(points, triangles, epsilon), proposals=[], cores=[], closures=[], regions=[],
        fixture_scope="COUNTER_AND_FAILURE_PLUMBING_ONLY_NOT_GEOMETRY")


def _mock_pair(module, monkeypatch, *, first_values=None, second_values=None, first_hook=None, second_hook=None):
    calls = []
    def make(lane, points, triangles, **kwargs):
        calls.append((lane, points, triangles, kwargs))
        answer = _fake_lane(lane, points, triangles, kwargs["position_error_bounds_mm"],
                            **((first_values if lane == "v7" else second_values) or {}))
        hook = first_hook if lane == "v7" else second_hook
        return hook(answer) if hook else answer
    monkeypatch.setattr(module._v7, "extract_noisy_scan_plane_proposals_v7",
                        lambda p, t, **kwargs: make("v7", p, t, **kwargs))
    monkeypatch.setattr(module._seeded, "extract_seeded_plane_proposals_v1",
                        lambda p, t, **kwargs: make("seeded_v1", p, t, **kwargs))
    return calls


def test_mock_residual_budget_allocation_precedes_dispatch_and_preserves_fixed_seed_quota(module, monkeypatch):
    points, triangles = _square()
    composite = module.CompositeScanPlaneBudgetsV1(max_triangle_memberships=100, max_raw_memberships=110,
        max_full_support_row_tests=120, max_fit_evaluations=20, max_retained_array_bytes=1000,
        max_metadata_bytes=100000, max_raw_predicate_visits=200, max_triangle_predicate_visits=300,
        max_edge_predicate_visits=400)
    first_values = dict(triangle_memberships=7, raw_memberships=11, full_support_row_tests=13,
        fit_evaluations=3, retained_array_bytes=17, metadata_bytes=19, cone_row_tests=23,
        closure_triangle_tests=29, closure_edge_tests=31)
    seed = module._seeded.SeededPlaneBudgetsV1(max_seeds=3, max_seed_selection_triangle_visits=222,
        max_proposal_triangle_visits=111, max_proposal_raw_row_visits=250, max_proposal_edge_visits=500)
    calls = _mock_pair(module, monkeypatch, first_values=first_values)
    result = _run(module, points, triangles, seeded_budgets=seed, budgets=composite)
    assert result["status"] == "COMPLETE", result
    assert [row[0] for row in calls] == ["v7", "seeded_v1"]
    first, second = (asdict(row[3]["budgets"]) for row in calls)
    for key in ("triangle_memberships", "raw_memberships", "full_support_row_tests", "fit_evaluations",
                "retained_array_bytes", "metadata_bytes"):
        assert first["max_"+key] == getattr(composite, "max_"+key)
        assert second["max_"+key] == getattr(composite, "max_"+key)-first_values[key]
    assert second["max_seeds"] == 3
    assert second["max_proposal_raw_row_visits"] == 177
    assert second["max_proposal_edge_visits"] == 369
    assert second["max_seed_selection_triangle_visits"] == 135
    assert second["max_proposal_triangle_visits"] == 111
    assert first["max_cone_row_tests"] == 200
    assert first["max_closure_triangle_tests"] == 300
    assert first["max_closure_edge_tests"] == 400
    assert result["effective_lane_budgets"] == dict(v7=first, seeded_v1=second)
    assert result["requested_lane_budgets"]["seeded_v1"] == asdict(seed)


def test_mock_caller_seed_budget_mutation_during_first_lane_cannot_change_second_schedule(module, monkeypatch):
    points, triangles = _square()
    caller = module._seeded.SeededPlaneBudgetsV1(max_seeds=3, max_proposal_raw_row_visits=111)
    def mutate(answer):
        object.__setattr__(caller, "max_seeds", 1)
        object.__setattr__(caller, "max_proposal_raw_row_visits", 1)
        return answer
    calls = _mock_pair(module, monkeypatch, first_hook=mutate)
    result = _run(module, points, triangles, seeded_budgets=caller)
    assert result["status"] == "COMPLETE", result
    assert calls[1][3]["budgets"].max_seeds == 3
    assert calls[1][3]["budgets"].max_proposal_raw_row_visits == 111
    assert result["requested_lane_budgets"]["seeded_v1"]["max_seeds"] == 3
    assert result["requested_lane_budgets"]["seeded_v1"]["max_proposal_raw_row_visits"] == 111


def test_mock_both_lanes_receive_one_owned_snapshot_even_after_caller_mutation(module, monkeypatch):
    points, triangles = _square()
    error = np.full(len(points), .05)
    expected = points.copy(), triangles.copy(), error.copy()
    def mutate(answer):
        points[:] = 999.
        triangles[:] = -1
        error[:] = 1000.
        return answer
    calls = _mock_pair(module, monkeypatch, first_hook=mutate)
    result = _run(module, points, triangles, error)
    assert result["status"] == "COMPLETE", result
    first, second = calls
    for i, original in ((1, expected[0]), (2, expected[1])):
        assert first[i] is second[i] and np.array_equal(first[i], original)
        assert not first[i].flags.writeable
        with pytest.raises(ValueError): first[i].setflags(write=True)
    a, b = first[3]["position_error_bounds_mm"], second[3]["position_error_bounds_mm"]
    assert a is b and np.array_equal(a, expected[2]) and not a.flags.writeable
    assert result["input_digests"] == _binding(*expected)


def test_mutation_while_first_snapshot_is_copied_fails_before_either_lane(module, monkeypatch):
    points, triangles = _square()
    original, count = module._v5._owned, []
    def mutate(value, dtype):
        answer = original(value, dtype)
        if not count:
            count.append(1)
            triangles[0, 0] = -1
        return answer
    monkeypatch.setattr(module._v5, "_owned", mutate)
    calls = _mock_pair(module, monkeypatch)
    result = _run(module, points, triangles)
    _failed(result)
    assert calls == [] and "OWNED_RAW_INDEX_OUT_OF_RANGE" in result["reasons"][0]


@pytest.mark.parametrize("lane,raises", [("v7", False), ("v7", True), ("seeded_v1", False), ("seeded_v1", True)])
def test_mock_first_or_second_failure_never_publishes_a_single_lane_fallback(module, monkeypatch, lane, raises):
    points, triangles = _square()
    def fail(answer):
        if raises: raise RuntimeError("MOCK_LANE_EXCEPTION")
        answer.update(status="UNAVAILABLE", reasons=["MOCK_LANE_REFUSAL"])
        return answer
    calls = _mock_pair(module, monkeypatch, first_hook=fail if lane == "v7" else None,
                      second_hook=fail if lane == "seeded_v1" else None)
    result = _run(module, points, triangles)
    _failed(result)
    assert [row[0] for row in calls] == (["v7"] if lane == "v7" else ["v7", "seeded_v1"])
    expected = (0 if lane == "v7" else 1) if raises else (1 if lane == "v7" else 2)
    assert result["statistics"]["discarded_lane_records"] == expected
    attempts = result["statistics"]["lane_attempts"]
    if not raises: assert attempts[lane]["reasons"] == ["MOCK_LANE_REFUSAL"]
    expected_execution = dict(v7="COMPLETE", seeded_v1="COMPLETE")
    expected_execution[lane] = "ATTEMPTED" if raises else "UNAVAILABLE"
    if lane == "v7": expected_execution["seeded_v1"] = "NOT_ATTEMPTED"
    assert result["lane_execution"] == expected_execution
    # An exception before return has no fabricated lane receipt/work count.


@pytest.mark.parametrize("fault", ["input_digest", "aggregate", "origin", "lane_digest", "finalize"])
def test_mock_late_binding_aggregate_origin_or_metadata_fault_clears_both_lanes(module, monkeypatch, fault):
    points, triangles = _square()
    def record(answer):
        rows, raw = np.array([0, 1], dtype="<i8"), np.arange(4, dtype="<i8")
        item = dict(analysis=dict(status="PROPOSED"), triangle_rows=rows, raw_vertex_rows=raw,
                    triangle_rows_sha256=_sha(rows, "<i8"), raw_vertex_rows_sha256=_sha(raw, "<i8"))
        answer["cores"] = answer["regions"] = [item]
        answer["proposals"] = [dict(kind="CORE" if answer["schema_version"] == "MOCK_v7" else "SEEDED_REGION", record_id=0)]
        return answer
    def second(answer):
        record(answer)
        if fault == "input_digest": answer["input_digests"]["points_sha256"] = "f"*64
        elif fault == "origin": answer["regions"][0]["analysis"]["status"] = "REJECTED"
        return answer
    calls = _mock_pair(module, monkeypatch, first_values=dict(fit_evaluations=3) if fault == "aggregate" else None,
        second_values=dict(fit_evaluations=2) if fault == "aggregate" else None, first_hook=record, second_hook=second)
    if fault == "lane_digest":
        old, digests = module._lane_digest, []
        def digest(*args, **kwargs):
            digests.append(1)
            if len(digests) == 2: raise RuntimeError("MOCK_LATE_DIGEST_FAILURE")
            return old(*args, **kwargs)
        monkeypatch.setattr(module, "_lane_digest", digest)
    elif fault == "finalize":
        def finalize(*args, **kwargs): raise RuntimeError("MOCK_FINAL_METADATA_FAILURE")
        monkeypatch.setattr(module._v7, "_finalize", finalize)
    kwargs = dict(budgets=module.CompositeScanPlaneBudgetsV1(max_fit_evaluations=4)) if fault == "aggregate" else {}
    result = _run(module, points, triangles, **kwargs)
    _failed(result)
    assert len(calls) == result["statistics"]["discarded_lane_records"] == 2
    if fault == "aggregate":
        assert calls[1][3]["budgets"].max_fit_evaluations == 1
        assert result["statistics"]["lane_attempts"]["seeded_v1"]["statistics"]["fit_evaluations"] == 2
    if fault in ("lane_digest", "finalize"):
        assert result["statistics"]["discarded_plane_records"] == 2
    elif fault == "origin": assert result["statistics"]["discarded_plane_records"] == 1


@pytest.mark.parametrize("fault", ["point_list", "triangle_float", "epsilon_bool", "epsilon_nan",
    "negative_index", "raw_range", "budget_type", "budget_bool", "budget_negative", "budget_above_maximum"])
def test_strict_initial_admission_precedes_both_lane_dispatches(module, monkeypatch, fault):
    points, triangles = _square()
    epsilon, budgets = .05, module.CompositeScanPlaneBudgetsV1()
    if fault == "point_list": points = points.tolist()
    elif fault == "triangle_float": triangles = triangles.astype(float)
    elif fault == "epsilon_bool": epsilon = True
    elif fault == "epsilon_nan": epsilon = np.nan
    elif fault == "negative_index": triangles[0, 0] = -1
    elif fault == "raw_range": points[0, 0] = 1e6+1.
    elif fault == "budget_type": budgets = {}
    elif fault == "budget_bool": budgets = replace(budgets, max_qualified_planes=True)
    elif fault == "budget_negative": budgets = replace(budgets, max_qualified_planes=-1)
    elif fault == "budget_above_maximum": budgets = replace(budgets, max_qualified_planes=65553)
    calls = _mock_pair(module, monkeypatch)
    _failed(_run(module, points, triangles, epsilon, budgets=budgets))
    assert calls == []


def test_mutated_public_composite_default_cannot_raise_literal_cap(module, monkeypatch):
    points, triangles = _square()
    default = module.prepare_composite_scan_planes_v1.__kwdefaults__["budgets"]
    saved = default.max_qualified_planes
    calls = _mock_pair(module, monkeypatch)
    try:
        object.__setattr__(default, "max_qualified_planes", 65553)
        _failed(_run(module, points, triangles))
        assert calls == []
    finally:
        object.__setattr__(default, "max_qualified_planes", saved)
