"""Prospective invented-source checks for explicit V2 context depth admission.

No native, registration, radius fit, physical input or artifact is used.  The
largest fixture has eight raw scan vertices, four scan triangles, 24 CAD planes
and 48 CAD triangles.  Its 256 caller assignments require exactly 1,024 signed
families and 8,192 explicit slots.  This is proposal coverage, never finite
support verification or an accepted pose.  Execute only through a separately
reviewed, frozen, resource-bounded source gate.
"""
import ast
import copy
from dataclasses import replace
import hashlib
import importlib
import inspect
import itertools
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = object()
_DEPTH_FIELD = "max_ordered_cad_matches_per_scan_pair"
_LIMITS = (
    ("max_scan_rows", 1_000_000),
    ("max_scan_triangles", 2_000_000),
    ("max_cad_triangles", 500_000),
    ("max_scan_planes", 128),
    ("max_cad_planes", 2048),
    ("max_scan_pairs", 64),
    (_DEPTH_FIELD, 256),
    ("max_signed_families", 1024),
    ("max_candidate_slots", 8192),
    ("max_scan_row_visits", 64_000_000),
    ("max_cad_vertex_visits", 128_000_000),
    ("max_index_visits", 64_000_000),
    ("max_index_sort_work", 256_000_000),
    ("max_hash_bytes", 1_073_741_824),
    ("max_projection_cache_entries", 6144),
    ("max_working_bytes", 268_435_456),
    ("max_report_bytes", 33_554_432),
    ("chunk_rows", 65_536),
)
_SLOT_NAMES = (
    "WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
    "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT",
)
_EMPTY_OUTPUTS = (
    "plane_bindings", "scheduled_pairs", "families", "candidates", "context_projections",
)


@pytest.fixture(scope="module")
def modules():
    """Load only the two pure helpers beneath one inert, consistent namespace."""
    name = "_invented_two_plane_context_v2_tests"
    assert not any(key == name or key.startswith(name + ".") for key in sys.modules)
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "backend")]
    sys.modules[name] = package
    try:
        v2 = importlib.import_module(name + ".cad_guided_two_plane_context_v2")
        v1 = importlib.import_module(name + ".cad_guided_two_plane_context_v1")
        yield types.SimpleNamespace(v1=v1, v2=v2)
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def _digest(rows):
    return hashlib.sha256(rows.tobytes(order="C")).hexdigest()


def _triple(value):
    return tuple(float(x) for x in value)


def _case(modules, count=1, *, rotation=None, translation=(0., 0., 0.), cad_height=12.):
    """Independent orthogonal rectangles; no production fixture is called."""
    v1 = modules.v1
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    points = np.asarray([
        (0., -3., 0.), (0., 3., 0.), (0., 3., 12.), (0., -3., 12.),
        (-4., 0., 0.), (4., 0., 0.), (4., 0., 12.), (-4., 0., 12.),
    ], dtype=np.float64)
    scan_triangles = np.asarray([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], dtype=np.int64)
    scan_planes = []
    for plane_id, normal, anchor, offset, triangles, raw in (
        (10, (1., 0., 0.), (3., 2., -4.), -3., (0, 1), (0, 1, 2, 3)),
        (20, (0., 1., 0.), (-2., -5., 6.), 5., (2, 3), (4, 5, 6, 7)),
    ):
        triangles = np.asarray(triangles, dtype=np.int64)
        raw = np.asarray(raw, dtype=np.int64)
        scan_planes.append(v1.ContextPlaneV1(
            plane_id, normal, anchor, offset, 0., True,
            triangles, _digest(triangles), raw, _digest(raw),
        ))

    unordered = max(1, (count + 1) // 2)
    nx = min(8, max(1, math.isqrt(unordered)))
    ny = (unordered + nx - 1) // nx
    triangles, faces, local_ids, cad_planes = [], [], [], []
    for direction, size in (("x", nx), ("y", ny)):
        for index in range(size):
            if direction == "x":
                coordinate = 4. * index
                vertices = np.asarray([
                    (coordinate, -3., 0.), (coordinate, 3., 0.),
                    (coordinate, 3., cad_height), (coordinate, -3., cad_height),
                ], dtype=np.float64)
                normal, anchor, offset = (1., 0., 0.), (coordinate + 3., 2., -4.), -3.
                plane_id = 1000 + index
            else:
                coordinate = 6. * index
                vertices = np.asarray([
                    (-4., coordinate, 0.), (4., coordinate, 0.),
                    (4., coordinate, cad_height), (-4., coordinate, cad_height),
                ], dtype=np.float64)
                normal, anchor, offset = (0., 1., 0.), (-2., coordinate - 5., 6.), 5.
                plane_id = 2000 + index
            vertices = vertices @ rotation.T + translation
            rows = np.asarray((len(triangles), len(triangles) + 1), dtype=np.int64)
            triangles.extend((vertices[[0, 1, 2]], vertices[[0, 2, 3]]))
            faces.extend((plane_id, plane_id))
            local_ids.extend((17, 23))
            cad_planes.append(v1.ContextPlaneV1(
                plane_id, _triple(rotation @ np.asarray(normal)),
                _triple(rotation @ np.asarray(anchor) + translation), offset,
                0., True, rows, _digest(rows),
            ))
    targets = []
    for i, j in itertools.product(range(nx), range(ny)):
        targets.extend(((1000 + i, 2000 + j), (2000 + j, 1000 + i)))
    return dict(
        scan_points=points, scan_triangles=scan_triangles,
        cad_triangles=np.asarray(triangles, dtype=np.float64),
        cad_face_ids=np.asarray(faces, dtype=np.int64),
        cad_source_triangle_ids=np.asarray(local_ids, dtype=np.int64),
        scan_planes=tuple(scan_planes), cad_planes=tuple(cad_planes),
        ordered_matches=tuple(v1.OrderedPlanePairV1((10, 20), pair) for pair in targets[:count]),
        scan_origin_mm=(0., 0., 0.), cad_origin_mm=(0., 0., 0.),
        unit_assumption="INVENTED_RECTANGULAR_COORDINATES_DECLARED_MM",
        source_frame="invented_scan", target_frame="invented_cad",
    )


def _call(modules, case, *, version=2, budgets=_DEFAULT):
    arguments = dict(case)
    if budgets is not _DEFAULT:
        arguments["budgets"] = budgets
    function = (modules.v2.propose_two_plane_context_v2 if version == 2
                else modules.v1.propose_two_plane_context_v1)
    return function(**arguments)


def _assert_scope(result):
    assert result["accepted_registration"] is False
    assert result["finite_support_match_verified"] is False
    assert result["production_use_allowed"] is False
    assert result["observed_bend_count"] is None
    assert result["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert result["axial_context_status"] == "AXIAL_CONTEXT_NOT_CERTIFIED"
    assert result["search_exhaustive"] is False
    assert result["scheduler_qualification"] == "NOT_IMPLEMENTED_CALLER_SCHEDULE_NOT_AUTHENTICATED"
    assert result["normal_qualification"] == "CALLER_DECLARED_QUALIFIED_CONES_NOT_RECOMPUTED_OR_AUTHENTICATED"
    assert result["translation_uncertainty"] == "UNAVAILABLE_NO_OFFSET_ERROR_OR_POSE_ENCLOSURE_MODEL"


def _assert_refused(result, reason):
    _assert_scope(result)
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == [reason]
    assert result["selected_policy_evaluation_complete"] is False
    assert result["usable_proposals"] is False
    assert result["input_digests"] == {}
    assert all(result[name] == [] for name in _EMPTY_OUTPUTS)
    assert len(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 4096


def _normalized(result):
    result = copy.deepcopy(result)
    result["schema_version"] = "two_plane_context_v1"
    result.pop("scheduling_capacity", None)
    if "budgets" in result:
        result["budgets"][_DEPTH_FIELD] = 4
    return result


def test_orchestration_ast_has_only_declared_version_and_preflight_delta():
    trees = [ast.parse((ROOT / "backend" / ("cad_guided_two_plane_context_v%d.py" % version)).read_text())
             for version in (1, 2)]
    functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "propose_two_plane_context_v%d" % version)
                 for tree, version in zip(trees, (1, 2))]
    before, after = copy.deepcopy(functions)
    assert isinstance(before.body[0].value, ast.Constant)
    assert isinstance(after.body[0].value, ast.Constant)
    after.name = before.name
    after.body[0] = copy.deepcopy(before.body[0])
    assert ast.dump(after.args.kw_defaults[-1]) == "Call(func=Name(id='TwoPlaneContextBudgetsV2', ctx=Load()), args=[], keywords=[])"
    after.args.kw_defaults[-1] = copy.deepcopy(before.args.kw_defaults[-1])
    block = next(node for node in after.body if isinstance(node, ast.Try))
    admission = block.body[3]
    assert ast.dump(admission) == "Expr(value=Call(func=Name(id='_global_schedule_preflight', ctx=Load()), args=[Name(id='ordered_matches', ctx=Load()), Name(id='cap', ctx=Load())], keywords=[]))"
    assert ast.dump(block.body[2]) == "Expr(value=Call(func=Attribute(value=Name(id='work', ctx=Load()), attr='memory', ctx=Load()), args=[], keywords=[]))"
    del block.body[3]
    assert ast.dump(after, include_attributes=False) == ast.dump(before, include_attributes=False)


def test_helpers_and_gates_are_literal_v1_aliases_with_only_depth_capacity_changed(modules):
    for name in (
        "ContextPlaneV1", "OrderedPlanePairV1", "_Unavailable", "_require", "_vector",
        "_Work", "_inputs", "_planes", "_schedule", "_family", "_slots", "_SLOTS",
        "_GATES", "_final_report", "_json",
    ):
        assert getattr(modules.v2, name) is getattr(modules.v1, name)
    assert modules.v2._MAXIMA == _LIMITS
    assert modules.v1._MAXIMA == tuple((name, 4 if name == _DEPTH_FIELD else maximum)
                                      for name, maximum in _LIMITS)
    assert modules.v2._SLOTS == _SLOT_NAMES
    assert modules.v2._MAX_ORDERED_MATCHES == 256
    base = modules.v2._base()
    assert base.pop("scheduling_capacity") == dict(
        policy="RESIDUAL_DEPTH_WITH_UNCHANGED_GLOBAL_SIGN_AND_SLOT_LIMITS",
        maximum_ordered_matches=256, maximum_distinct_scan_pairs=64,
        maximum_ordered_matches_per_scan_pair=256, signs_per_ordered_match=4, slots_per_sign=8,
    )
    base["schema_version"] = "two_plane_context_v1"
    assert base == modules.v1._base()
    parameters = inspect.signature(modules.v2.propose_two_plane_context_v2).parameters
    assert not any("radius" in name or "thickness" in name or "bend_angle" in name for name in parameters)


@pytest.mark.parametrize("field,maximum", _LIMITS)
@pytest.mark.parametrize("boundary", ("minimum", "maximum"))
def test_all_budget_field_closed_range_endpoints(modules, field, maximum, boundary):
    value = (3 if field == "chunk_rows" else 0) if boundary == "minimum" else maximum
    budget = replace(modules.v2.TwoPlaneContextBudgetsV2(), **{field: value})
    assert modules.v2._configuration(budget)[field] == value


@pytest.mark.parametrize("field,maximum", _LIMITS)
@pytest.mark.parametrize("kind", ("bool", "float", "below", "above"))
def test_every_budget_field_rejects_type_or_range_violation(modules, field, maximum, kind):
    value = {"bool": True, "float": float(maximum), "below": 2 if field == "chunk_rows" else -1,
             "above": maximum + 1}[kind]
    budget = replace(modules.v2.TwoPlaneContextBudgetsV2(), **{field: value})
    with pytest.raises(modules.v1._Unavailable, match="^INVALID_BUDGET_VALUES$"):
        modules.v2._configuration(budget)


@pytest.mark.parametrize("kind", ("v1", "mapping", "none", "bool", "subclass"))
def test_v2_requires_exact_owned_budget_class_before_input(modules, monkeypatch, kind):
    class Derived(modules.v2.TwoPlaneContextBudgetsV2):
        pass
    values = dict(v1=modules.v1.TwoPlaneContextBudgetsV1(), mapping={}, none=None,
                  bool=True, subclass=Derived())
    calls = []
    monkeypatch.setattr(modules.v2, "_inputs", lambda *args: calls.append(args))
    result = _call(modules, _case(modules), budgets=values[kind])
    _assert_refused(result, "INVALID_BUDGET_TYPE")
    assert calls == []
    assert result["statistics"] == {}


@pytest.mark.parametrize("kind", ("numpy_integer", "integer_subclass"))
def test_non_plain_integer_budget_values_are_not_aliases(modules, kind):
    class DerivedInt(int):
        pass
    value = np.int64(6) if kind == "numpy_integer" else DerivedInt(6)
    with pytest.raises(modules.v1._Unavailable, match="^INVALID_BUDGET_VALUES$"):
        modules.v2._configuration(replace(modules.v2.TwoPlaneContextBudgetsV2(),
                                         max_ordered_cad_matches_per_scan_pair=value))


def test_budget_snapshot_is_detached_and_v1_rejects_v2_class(modules):
    budget = modules.v2.TwoPlaneContextBudgetsV2()
    snapshot = modules.v2._configuration(budget)
    object.__setattr__(budget, _DEPTH_FIELD, 1)
    assert snapshot[_DEPTH_FIELD] == 256
    assert modules.v2.TwoPlaneContextBudgetsV2().max_ordered_cad_matches_per_scan_pair == 256
    _assert_refused(_call(modules, _case(modules), version=1,
                         budgets=modules.v2.TwoPlaneContextBudgetsV2()), "INVALID_BUDGET_TYPE")


@pytest.mark.parametrize("count", (5, 42))
def test_v1_still_refuses_a_deeper_single_pair_before_signed_families(modules, count):
    result = _call(modules, _case(modules, count), version=1)
    _assert_refused(result, "PAIR_ASSIGNMENT_QUOTA_EXCEEDED")
    assert result["statistics"]["signed_families_attempted"] == 0
    assert result["statistics"]["slot_records_completed"] == 0


@pytest.mark.parametrize("count", (5, 6, 42, 256))
def test_actual_deep_geometry_has_complete_sign_slot_and_witness_denominator(modules, count):
    case = _case(modules, count)
    result = _call(modules, case)
    _assert_scope(result)
    assert result["status"] == "COMPLETE"
    assert result["selected_policy_evaluation_complete"] is True
    assert result["usable_proposals"] is True
    assert len(result["scheduled_pairs"]) == count
    assert len({tuple(row["scan_plane_ids"]) for row in result["scheduled_pairs"]}) == 1
    assert len(result["families"]) == 4 * count
    stats = result["statistics"]
    assert stats["signed_families_planned"] == stats["signed_families_attempted"] == stats["signed_families_completed"] == 4 * count
    assert stats["candidate_slots_planned"] == stats["slot_records_completed"] == 32 * count
    assert stats["slots_generation_attempted"] == 32 * count
    assert stats["proposed_slot_count"] + stats["absent_midpoint_slots"] + stats["rejected_slot_count"] == 32 * count
    assert stats["rejected_slot_count"] == 0
    proposed = set()
    for assignment in range(count):
        families = result["families"][4 * assignment:4 * assignment + 4]
        assert [family["normal_signs"] for family in families] == [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        for family in families:
            assert family["ordered_assignment_id"] == assignment
            assert family["status"] == "AVAILABLE"
            assert tuple(slot["slot"] for slot in family["slots"]) == _SLOT_NAMES
            for slot in family["slots"]:
                if slot["status"] == "PROPOSED_SEED_ONLY":
                    proposed.add((slot["candidate_index"], family["signed_family_id"], slot["slot"]))
    witnesses = []
    for index, candidate in enumerate(result["candidates"]):
        assert candidate["candidate_index"] == index
        assert candidate["accepted_registration"] is False
        rotation = np.asarray(candidate["rotation"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-10)
        assert abs(np.linalg.det(rotation) - 1.) <= 1e-10
        witnesses.extend((index, row["signed_family_id"], row["slot"])
                         for row in candidate["contributing_slots"])
    assert len(witnesses) == len(set(witnesses)) == stats["proposed_slot_count"]
    assert set(witnesses) == proposed
    assert stats["exact_pose_duplicates"] == stats["proposed_slot_count"] - len(result["candidates"])
    if count == 256:
        assert result["input_counts"] == dict(scan_raw_rows=8, scan_triangles=4, cad_triangles=48)
        assert len(case["cad_planes"]) == 24
        assert stats["projection_cache_entries"] <= 6144
        assert stats["accounted_working_peak_bytes"] <= 268_435_456


@pytest.mark.parametrize("pose", ("identity", "quarter_turn", "cycle", "half_turn"))
def test_independent_rigid_pose_and_every_signed_plane_equation_without_radius(modules, pose):
    rotations = dict(
        identity=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
        quarter_turn=((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)),
        cycle=((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)),
        half_turn=((-1., 0., 0.), (0., 1., 0.), (0., 0., -1.)),
    )
    rotation = np.asarray(rotations[pose])
    translation = np.asarray((7., -11., 19.))
    case = _case(modules, 6, rotation=rotation, translation=translation)
    case.update(scan_origin_mm=(2., -4., 6.), cad_origin_mm=(-3., 5., -7.))
    result = _call(modules, case)
    assert result["status"] == "COMPLETE"
    expected = case["scan_points"] @ rotation.T + translation
    assert any(np.allclose(case["scan_points"] @ np.asarray(row["rotation"]).T
                           + np.asarray(row["translation_mm"]), expected, rtol=0., atol=1e-8)
               for row in result["candidates"])
    scans = {plane.plane_id: plane for plane in case["scan_planes"]}
    cads = {plane.plane_id: plane for plane in case["cad_planes"]}
    for family in result["families"]:
        slot = next(row for row in family["slots"] if row["status"] == "PROPOSED_SEED_ONLY")
        candidate = result["candidates"][slot["candidate_index"]]
        for relation in family["correspondences"]:
            scan, cad = scans[relation["scan_plane_id"]], cads[relation["cad_plane_id"]]
            transformed = (case["scan_points"][scan.raw_vertex_rows]
                           @ np.asarray(candidate["rotation"]).T + candidate["translation_mm"])
            residual = (transformed - cad.anchor_mm) @ np.asarray(cad.normal) - cad.offset_from_anchor_mm
            np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-8)
    _assert_scope(result)


@pytest.mark.parametrize("variant", ("one", "four", "reversed", "origins", "short_axial",
                                     "rank_rejected", "angle_rejected", "malformed_late_plane"))
def test_shared_small_v1_v2_results_are_exact_after_only_declared_normalization(modules, variant):
    case = _case(modules, 4 if variant == "four" else 1,
                 cad_height=6. if variant == "short_axial" else 12.)
    if variant == "reversed":
        row = case["ordered_matches"][0]
        case["ordered_matches"] = (replace(row, scan_plane_ids=row.scan_plane_ids[::-1],
                                           cad_plane_ids=row.cad_plane_ids[::-1]),)
    elif variant == "origins":
        case.update(scan_origin_mm=(2., 3., 4.), cad_origin_mm=(-5., 6., -7.))
    elif variant == "rank_rejected":
        case["scan_planes"] = (case["scan_planes"][0],
                               replace(case["scan_planes"][1], normal=(1., 0., 0.)))
    elif variant == "angle_rejected":
        case["cad_planes"] = (case["cad_planes"][0],
                              replace(case["cad_planes"][1], normal=(.6, .8, 0.)))
    elif variant == "malformed_late_plane":
        case["cad_planes"] = (case["cad_planes"][0],
                              replace(case["cad_planes"][1], triangle_rows_sha256="0" * 64))
    old, new = _call(modules, case, version=1), _call(modules, case)
    assert _normalized(new) == _normalized(old)
    if variant == "malformed_late_plane":
        _assert_refused(new, "STALE_CAD_SUPPORT_TRIANGLE_DIGEST")
    else:
        assert new["status"] == "COMPLETE"
        if variant in ("rank_rejected", "angle_rejected"):
            assert new["usable_proposals"] is False
            assert new["statistics"]["rejected_slot_count"] == 32
        if variant == "short_axial":
            assert new["statistics"]["absent_midpoint_slots"] == 8


@pytest.mark.parametrize("variant", (
    "257", "sign_1023", "slot_8191", "sign_23", "slot_191", "no_scan_pairs",
    "pair_product", "list", "tuple_subclass", "none", "bool", "scratch",
))
def test_global_denominator_and_scratch_refusal_precede_all_buffer_reads(modules, monkeypatch, variant):
    class TupleSubclass(tuple):
        pass
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("input buffers must not be read")
    monkeypatch.setattr(modules.v2, "_inputs", forbidden)
    case = {name: object() for name in (
        "scan_points", "scan_triangles", "cad_triangles", "cad_face_ids", "cad_source_triangle_ids",
        "scan_planes", "cad_planes", "scan_origin_mm", "cad_origin_mm",
        "unit_assumption", "source_frame", "target_frame",
    )}
    count = 257 if variant == "257" else 256 if variant in ("sign_1023", "slot_8191") else 6
    matches, changes = tuple(object() for _ in range(count)), {}
    if variant == "sign_1023": changes["max_signed_families"] = 1023
    if variant == "slot_8191": changes["max_candidate_slots"] = 8191
    if variant == "sign_23": changes["max_signed_families"] = 23
    if variant == "slot_191": changes["max_candidate_slots"] = 191
    if variant == "no_scan_pairs": changes["max_scan_pairs"] = 0
    if variant == "pair_product": changes.update(max_scan_pairs=1, max_ordered_cad_matches_per_scan_pair=5)
    if variant == "list": matches = list(matches)
    if variant == "tuple_subclass": matches = TupleSubclass(matches)
    if variant == "none": matches = None
    if variant == "bool": matches = False
    if variant == "scratch": changes["max_working_bytes"] = 65_535
    case["ordered_matches"] = matches
    result = _call(modules, case, budgets=replace(modules.v2.TwoPlaneContextBudgetsV2(), **changes))
    _assert_refused(result, "ACCOUNTED_WORKING_PAYLOAD_BUDGET" if variant == "scratch" else "ORDERED_MATCH_TABLE_BUDGET")
    assert calls == []
    assert result["statistics"]["accounted_working_peak_bytes"] == 65_536
    for name in ("owned_array_bytes", "hash_bytes", "scan_row_visits", "cad_vertex_visits",
                 "index_visits", "signed_families_attempted", "slot_records_completed"):
        assert result["statistics"][name] == 0


@pytest.mark.parametrize("variant", ("exact_global", "exact_pair", "one_less_pair"))
def test_smaller_operating_caps_are_honored_without_prefix_splitting(modules, variant):
    changes = dict(max_signed_families=24, max_candidate_slots=192)
    if variant in ("exact_pair", "one_less_pair"):
        changes[_DEPTH_FIELD] = 6 if variant == "exact_pair" else 5
    result = _call(modules, _case(modules, 6),
                   budgets=replace(modules.v2.TwoPlaneContextBudgetsV2(), **changes))
    if variant == "one_less_pair":
        _assert_refused(result, "PAIR_ASSIGNMENT_QUOTA_EXCEEDED")
        assert result["statistics"]["signed_families_attempted"] == 0
    else:
        assert result["status"] == "COMPLETE"
        assert len(result["families"]) == 24
        assert result["statistics"]["slot_records_completed"] == 192


def test_empty_schedule_with_zero_denominator_is_complete_without_proposals(modules):
    budget = replace(modules.v2.TwoPlaneContextBudgetsV2(), max_scan_pairs=0,
                     max_ordered_cad_matches_per_scan_pair=0, max_signed_families=0, max_candidate_slots=0)
    result = _call(modules, _case(modules, 0), budgets=budget)
    assert result["status"] == "COMPLETE"
    assert result["selected_policy_evaluation_complete"] is True
    assert result["usable_proposals"] is False
    assert result["scheduled_pairs"] == result["families"] == result["candidates"] == []
    assert result["statistics"]["candidate_slots_planned"] == 0
    assert result["statistics"]["signed_families_completed"] == 0
    _assert_scope(result)


@pytest.mark.parametrize("variant", ("reverse_both", "both_target_orders"))
def test_canonical_reversal_and_distinct_target_orders_preserve_assignment_identity(modules, variant):
    case = _case(modules, 1)
    row = case["ordered_matches"][0]
    if variant == "reverse_both":
        case["ordered_matches"] = (replace(row, scan_plane_ids=row.scan_plane_ids[::-1],
                                           cad_plane_ids=row.cad_plane_ids[::-1]),)
        result = _call(modules, case)
        assert result["scheduled_pairs"] == [dict(scan_plane_ids=[10, 20], cad_plane_ids=[1000, 2000])]
    else:
        case["ordered_matches"] = (row, replace(row, cad_plane_ids=row.cad_plane_ids[::-1]))
        result = _call(modules, case)
        assert result["scheduled_pairs"] == [
            dict(scan_plane_ids=[10, 20], cad_plane_ids=[1000, 2000]),
            dict(scan_plane_ids=[10, 20], cad_plane_ids=[2000, 1000]),
        ]
    assert result["status"] == "COMPLETE"
    assert result["statistics"]["slot_records_completed"] == 32 * len(case["ordered_matches"])


@pytest.mark.parametrize("variant,reason", (
    ("duplicate", "DUPLICATE_ORDERED_PAIR_ASSIGNMENT"),
    ("reverse_duplicate", "DUPLICATE_ORDERED_PAIR_ASSIGNMENT"),
    ("unknown_scan", "UNKNOWN_OR_REPEATED_PLANE_ASSIGNMENT"),
    ("unknown_cad", "UNKNOWN_OR_REPEATED_PLANE_ASSIGNMENT"),
    ("repeated_scan", "UNKNOWN_OR_REPEATED_PLANE_ASSIGNMENT"),
    ("repeated_cad", "UNKNOWN_OR_REPEATED_PLANE_ASSIGNMENT"),
    ("bool_id", "INVALID_IMPORT_LOCAL_PLANE_ID"),
    ("numpy_id", "INVALID_IMPORT_LOCAL_PLANE_ID"),
    ("subclass", "INVALID_ORDERED_PAIR_TYPE"),
    ("mapping", "INVALID_ORDERED_PAIR_TYPE"),
    ("list_source", "EXACTLY_TWO_DISTINCT_ASSIGNMENTS_REQUIRED"),
    ("one_target", "EXACTLY_TWO_DISTINCT_ASSIGNMENTS_REQUIRED"),
))
def test_every_malformed_later_assignment_discards_the_whole_table(modules, variant, reason):
    case = _case(modules, 1)
    row = case["ordered_matches"][0]
    bad = replace(row, cad_plane_ids=row.cad_plane_ids[::-1])
    if variant == "duplicate": bad = row
    if variant == "reverse_duplicate": bad = replace(row, scan_plane_ids=(20, 10), cad_plane_ids=(2000, 1000))
    if variant == "unknown_scan": bad = replace(bad, scan_plane_ids=(10, 999))
    if variant == "unknown_cad": bad = replace(bad, cad_plane_ids=(2000, 999))
    if variant == "repeated_scan": bad = replace(bad, scan_plane_ids=(10, 10))
    if variant == "repeated_cad": bad = replace(bad, cad_plane_ids=(1000, 1000))
    if variant == "bool_id": bad = replace(bad, scan_plane_ids=(True, 20))
    if variant == "numpy_id": bad = replace(bad, scan_plane_ids=(np.int64(10), 20))
    if variant == "subclass":
        class Derived(modules.v1.OrderedPlanePairV1):
            pass
        bad = Derived((10, 20), (2000, 1000))
    if variant == "mapping": bad = dict(scan_plane_ids=(10, 20), cad_plane_ids=(2000, 1000))
    if variant == "list_source": bad = replace(bad, scan_plane_ids=[10, 20])
    if variant == "one_target": bad = replace(bad, cad_plane_ids=(1000,))
    case["ordered_matches"] = (row, bad)
    result = _call(modules, case)
    _assert_refused(result, reason)
    assert result["statistics"]["signed_families_attempted"] == 0


@pytest.mark.parametrize("variant,reason", (
    ("array_subclass", "INVALID_SCAN_POINTS_ARRAY"),
    ("float32", "INVALID_SCAN_POINTS_ARRAY"),
    ("late_nan", "UNSUPPORTED_CONTEXT_COORDINATE_RANGE"),
    ("stale_scan_digest", "STALE_SCAN_SUPPORT_TRIANGLE_DIGEST"),
    ("missing_raw", "INCOMPLETE_SCAN_SUPPORT_RAW_UNION"),
    ("incomplete_cad_face", "INCOMPLETE_ASSIGNED_CAD_FACE"),
    ("unqualified", "DUPLICATE_OR_UNQUALIFIED_PLANE"),
    ("bool_normal", "UNSUPPORTED_VECTOR_RANGE"),
    ("plane_subclass", "INVALID_CONTEXT_PLANE_TYPE"),
    ("duplicate_cad_identity", "DUPLICATE_CAD_SOURCE_TRIANGLE_IDENTITY"),
))
def test_geometry_buffer_and_lineage_validation_is_not_bypassed_by_depth(modules, variant, reason):
    case = _case(modules, 6)
    if variant == "array_subclass":
        class Derived(np.ndarray):
            pass
        case["scan_points"] = case["scan_points"].view(Derived)
    if variant == "float32": case["scan_points"] = case["scan_points"].astype(np.float32)
    if variant == "late_nan": case["scan_points"][-1, -1] = math.nan
    if variant == "stale_scan_digest":
        case["scan_planes"] = (case["scan_planes"][0], replace(case["scan_planes"][1], triangle_rows_sha256="0" * 64))
    if variant == "missing_raw":
        rows = case["scan_planes"][1].raw_vertex_rows[:3].copy()
        case["scan_planes"] = (case["scan_planes"][0], replace(case["scan_planes"][1],
            raw_vertex_rows=rows, raw_vertex_rows_sha256=_digest(rows)))
    if variant == "incomplete_cad_face":
        rows = case["cad_planes"][-1].triangle_rows[:1].copy()
        case["cad_planes"] = case["cad_planes"][:-1] + (replace(case["cad_planes"][-1],
            triangle_rows=rows, triangle_rows_sha256=_digest(rows)),)
    if variant == "unqualified":
        case["cad_planes"] = case["cad_planes"][:-1] + (replace(case["cad_planes"][-1], normal_qualified=False),)
    if variant == "bool_normal":
        case["cad_planes"] = case["cad_planes"][:-1] + (replace(case["cad_planes"][-1], normal=(False, True, False)),)
    if variant == "plane_subclass":
        class Derived(modules.v1.ContextPlaneV1):
            pass
        plane = case["cad_planes"][-1]
        bad = Derived(**{name: getattr(plane, name) for name in plane.__dataclass_fields__})
        case["cad_planes"] = case["cad_planes"][:-1] + (bad,)
    if variant == "duplicate_cad_identity": case["cad_source_triangle_ids"][1] = case["cad_source_triangle_ids"][0]
    result = _call(modules, case)
    _assert_refused(result, reason)
    assert result["statistics"]["signed_families_attempted"] == 0


@pytest.mark.parametrize("kind", ("declared_refusal", "untrusted_exception"))
def test_late_family_failure_discards_every_usable_prefix_and_never_stringifies_unknown_error(modules, monkeypatch, kind):
    original, calls = modules.v2._family, []
    class UntrustedError(Exception):
        def __str__(self):
            raise AssertionError("arbitrary exception stringification is forbidden")
    def fail_late(*args):
        calls.append(1)
        if len(calls) == 5:
            if kind == "declared_refusal":
                raise modules.v1._Unavailable("INVENTED_LATE_FAMILY_FAILURE")
            raise UntrustedError()
        return original(*args)
    monkeypatch.setattr(modules.v2, "_family", fail_late)
    result = _call(modules, _case(modules, 6))
    _assert_refused(result, "INVENTED_LATE_FAMILY_FAILURE" if kind == "declared_refusal" else "CONTEXT_HELPER_EXCEPTION")
    assert len(calls) == 5
    assert result["discarded_signed_families"] == 5
    assert result["discarded_candidates"] > 0
    stats = result["statistics"]
    assert stats["signed_families_attempted"] == 5
    assert stats["signed_families_completed"] == 4
    assert stats["slot_records_completed"] == 32
    assert stats["signed_families_planned"] == 24
    assert stats["candidate_slots_planned"] == 192


def test_actual_late_projection_cache_budget_clears_completed_family_prefix(modules):
    result = _call(modules, _case(modules), budgets=replace(
        modules.v2.TwoPlaneContextBudgetsV2(), max_projection_cache_entries=8))
    _assert_refused(result, "PROJECTION_CACHE_BUDGET")
    assert result["discarded_signed_families"] == 2
    assert result["discarded_candidates"] > 0
    stats = result["statistics"]
    assert stats["projection_cache_entries"] == 8
    assert stats["signed_families_attempted"] == 2
    assert stats["signed_families_completed"] == 1
    assert stats["slot_records_completed"] == 8
