"""Independent invented-plane mathematics and bounded search accounting.

No raw/native/CAD inputs, app imports, physical IDs, or runtime execution during
authorship. Public positive cases require a real proper pose; mocks below are
explicit failure/admission controls, never a substitute for those positives.
"""
from copy import deepcopy
from dataclasses import fields, replace
import itertools
import json
import math

import numpy as np
import pytest

from backend import cad_guided_plane_search_v5 as search


Equation = search.PlaneEquationV5
Observed = search.ObservedPlaneV5
Policy = search.PlaneSearchPolicyV5
Budgets = search.PlaneSearchBudgetsV5
UNIT = "INVENTED_DECLARED_MM_NOT_PHYSICAL_CALIBRATION"
NORMALS = np.array([[1., 0., 0.], [3/5, 4/5, 0.], [2/7, 3/7, 6/7]])
# Independent rational rotation from quaternion (w,x,y,z)=(1,2,3,4)/sqrt(30).
ROTATION = np.array([[-2/3, 2/15, 11/15], [2/3, -1/3, 2/3], [1/3, 14/15, 2/15]])
TRANSLATION = np.array([7., -11., 13.])
ANCHORS = np.array([[4., -2., 1.], [0., 5., -3.], [-2., 1., 6.]])
INTERSECTION = np.array([2., -3., 5.])
SIGNS = tuple(itertools.product((-1, 1), repeat=3))
BUDGET_FIELDS = (
    "max_scan_planes", "max_cad_planes", "max_bank_planes", "max_scan_triplets",
    "max_scan_partner_tests", "max_cad_pair_records", "max_pair_join_tests",
    "max_cad_pair_samples_per_triplet", "max_unsigned_per_scan_triplet",
    "max_compatible_rows_per_triplet", "max_unsigned_assignments",
    "max_signed_assignments", "max_report_bytes", "chunk_pair_rows")


def _tuple(vector):
    return tuple(float(value) for value in vector)


def _equation(identifier, normal, point=INTERSECTION, anchor=(0., 0., 0.), alpha=0.):
    normal, anchor = np.asarray(normal), np.asarray(anchor)
    return Equation(identifier, _tuple(normal), _tuple(anchor),
                    float(normal @ (np.asarray(point)-anchor)), alpha)


def _observation(equation, *, index=0, kind="DECLARED", count=12, rms=0.):
    # These deliberately unmatched partial centroids are not correspondence truth.
    return Observed(equation, count, rms, (100.+index, -50.+3*index, 70.-index), kind)


def _case(*, orthogonal=False, rotation=ROTATION, translation=TRANSLATION, scale=1.):
    normals = np.eye(3) if orthogonal else NORMALS
    observed, target = [], []
    for index, normal in enumerate(normals):
        source = _equation(10+index, normal, scale*INTERSECTION, scale*ANCHORS[index])
        mapped = Equation(110+index, _tuple(rotation @ normal),
            _tuple(rotation @ (scale*ANCHORS[index])+scale*translation),
            source.offset_from_anchor_mm, 0.)
        observed.append(_observation(source, index=index))
        target.append(mapped)
    return tuple(observed), tuple(target)


def _run(observed=None, target=None, **kwargs):
    if observed is None: observed, target = _case()
    return search.propose_plane_search_v5(observed, target, unit_assumption=UNIT, **kwargs)


def _not_admission(result):
    assert result["accepted_registration"] is False and result["production_use_allowed"] is False
    assert result["observed_bend_count"] is None and result["search_exhaustive"] is False
    if "physical_plane_count" in result: assert result["physical_plane_count"] is None
    if "support_provenance" in result:
        assert result["support_provenance"] == "UNVERIFIED_DECLARATIONS"
        assert result["raw_support_provenance_verified"] is False
        assert result["normal_cone_status"] == "STIPULATED_UNCALIBRATED"
        assert result["finite_support_match_verified"] is False
        assert result["translation_uncertainty"] == "UNAVAILABLE_NO_OFFSET_ERROR_MODEL"
        assert result["global_pose_uniqueness"] == "NOT_ESTABLISHED"


def _complete(result):
    assert result["status"] == "COMPLETE", result
    assert result["selected_policy_evaluation_complete"] is True
    stats = result["statistics"]
    assert stats["signed_assignments_planned"] == stats["signed_assignments_attempted"] == stats["signed_assignments_evaluated"]
    assert stats["signed_assignments_evaluated"] == len(result["hypotheses"]) == 8*len(result["unsigned_assignments"])
    assert stats["proposed_hypotheses"]+stats["rejected_hypotheses"] == len(result["hypotheses"])
    assert stats["scan_triplets_evaluated"] == stats["scan_triplets_scheduled"] == len(result["scan_triplets"])
    for assignment in result["unsigned_assignments"]:
        rows = [row for row in result["hypotheses"] if row["unsigned_assignment_id"] == assignment["unsigned_assignment_id"]]
        assert [tuple(row["normal_signs"]) for row in rows] == list(SIGNS)
        assert all(row["status"] in ("PROPOSED", "REJECTED") for row in rows)
    _not_admission(result)


def _refused(result, reason=None):
    assert result["status"] == "UNAVAILABLE" and result["usable_hypotheses"] is False
    assert not result["selected_policy_evaluation_complete"]
    if reason is not None: assert result["reasons"] == [reason]
    for row in result["hypotheses"]:
        assert row["status"] == "INVALIDATED_BY_FATAL_SEARCH_FAILURE"
        assert row["rotation"] is row["translation_mm"] is None
    _not_admission(result)


def _pose_rows(result):
    return [row for row in result["hypotheses"] if row["status"] == "PROPOSED"]


def _pair_angle(first, second):
    # Scalar oracle, independent of the vectorized cross/atan2 index kernel.
    cosine = abs(math.fsum(float(a)*float(b) for a, b in zip(first, second)))
    return math.acos(min(1., max(0., cosine)))


def _interval(first, second, guard):
    angle = _pair_angle(first.normal, second.normal)
    radius = first.normal_angle_bound_radians+second.normal_angle_bound_radians+guard
    return max(0., angle-radius), min(math.pi/2., angle+radius)


def _overlaps(first, second):
    return max(first[0], second[0]) <= min(first[1], second[1])


def _bits(encoded, count):
    raw = bytes.fromhex(encoded)
    assert len(raw) == (count+7)//8
    return [bool(raw[index//8] & (1 << (index % 8))) for index in range(count)]


def _strata(length, quota):
    count = min(length, quota)
    if count == 0: return []
    if count == 1: return [length//2]
    return [(index*(length-1))//(count-1) for index in range(count)]


def _reconstruct_join(result, observed, target):
    """Reconstruct every searched rejection, compatible row and omitted quota."""
    scan = {row.equation.plane_id: row.equation for row in observed}
    cad = {row.plane_id: row for row in target}
    ids = sorted(cad)
    guard = result["policy"]["pair_angle_numeric_guard_radians"]
    pairs = list(itertools.combinations(ids, 2))
    pairs.sort(key=lambda pair: _pair_angle(cad[pair[0]].normal, cad[pair[1]].normal))
    max_radius = max((cad[a].normal_angle_bound_radians+cad[b].normal_angle_bound_radians for a, b in pairs), default=0.)
    assigned = {row["unsigned_assignment_id"]: row for row in result["unsigned_assignments"]}
    totals = dict(windows=0, scheduled=0, incompatible=0, compatible=0, tests=0, omitted=0)
    for row in result["scan_triplets"]:
        if row["analysis"]["status"] != "AVAILABLE": continue
        source = [scan[identifier] for identifier in row["scan_plane_ids"]]
        intervals = [_interval(source[a], source[b], guard) for a, b in ((0, 1), (0, 2), (1, 2))]
        window = [(a, b) for a, b in pairs if intervals[0][0]-max_radius-guard <=
                  _pair_angle(cad[a].normal, cad[b].normal) <= intervals[0][1]+max_radius+guard]
        positions = _strata(len(window), result["budgets"]["max_cad_pair_samples_per_triplet"])
        assert row["indexed_pair_window_count"] == len(window)
        assert row["scheduled_pair_window_positions"] == positions
        assert [tuple(record["cad_pair_ids"]) for record in row["cad_pair_records"]] == [window[i] for i in positions]
        compatible = []
        for record in row["cad_pair_records"]:
            a, b = record["cad_pair_ids"]
            passed = _overlaps(intervals[0], _interval(cad[a], cad[b], guard))
            assert record["status"] == ("PAIR_INTERVAL_COMPATIBLE" if passed else "PAIR_INTERVAL_INCOMPATIBLE")
            totals["tests"] += 1
            if not passed:
                totals["incompatible"] += 1
                assert record["orientations"] == []
                continue
            assert [o["assigned_first_two_cad_ids"] for o in record["orientations"]] == [[a, b], [b, a]]
            for orientation in record["orientations"]:
                first, second = orientation["assigned_first_two_cad_ids"]
                others = [identifier for identifier in ids if identifier not in (first, second)]
                reject_a = [not _overlaps(intervals[1], _interval(cad[first], cad[third], guard)) for third in others]
                reject_b = [not _overlaps(intervals[2], _interval(cad[second], cad[third], guard)) for third in others]
                assert orientation["tested_third_count"] == len(others)
                assert _bits(orientation["rejected_02_packbits_hex"], len(others)) == reject_a
                assert _bits(orientation["rejected_12_packbits_hex"], len(others)) == reject_b
                kept = [third for third, ra, rb in zip(others, reject_a, reject_b) if not (ra or rb)]
                assert orientation["compatible_third_count"] == len(kept)
                compatible.extend((first, second, third) for third in kept)
                totals["tests"] += 2*len(others)
        selected = _strata(len(compatible), result["budgets"]["max_unsigned_per_scan_triplet"])
        assert row["compatible_unsigned_count"] == len(compatible)
        assert row["scheduled_compatible_assignment_positions"] == selected
        assert [tuple(assigned[index]["cad_plane_ids"]) for index in row["scheduled_unsigned_ids"]] == [compatible[index] for index in selected]
        assert row["unscheduled_compatible_unsigned_count"] == len(compatible)-len(selected)
        totals["windows"] += len(window)
        totals["scheduled"] += len(positions)
        totals["compatible"] += len(compatible)
        totals["omitted"] += len(compatible)-len(selected)
    stats = result["statistics"]
    for field, oracle in (("indexed_pair_window_occurrences", "windows"), ("scheduled_pair_occurrences", "scheduled"),
        ("tested_incompatible_pair_occurrences", "incompatible"), ("compatible_unsigned_occurrences", "compatible"),
        ("pair_join_tests", "tests"), ("unscheduled_compatible_unsigned_occurrences", "omitted")):
        assert stats[field] == totals[oracle]
    assert stats["unscheduled_index_window_occurrences"] == totals["windows"]-totals["scheduled"]


@pytest.mark.parametrize("scale", (0.001, 1., 1000.))
def test_asymmetric_independent_real_pose_and_nominal_equation_witnesses(scale):
    observed, target = _case(scale=scale)
    result = _run(observed, target)
    _complete(result)
    _reconstruct_join(result, observed, target)
    poses = _pose_rows(result)
    assert len(poses) == 1, result
    row = poses[0]
    rotation, translation = np.asarray(row["rotation"]), np.asarray(row["translation_mm"])
    np.testing.assert_allclose(rotation, ROTATION, atol=2e-12, rtol=0.)
    np.testing.assert_allclose(translation, scale*TRANSLATION, atol=2e-8, rtol=0.)
    probes = scale*np.array([[2., -3., 5.], [4., 7., -9.], [-6., 3., 1.]])
    np.testing.assert_allclose(probes@rotation.T+translation,
                               probes@ROTATION.T+scale*TRANSLATION, atol=2e-8, rtol=0.)
    for equation in target:
        x = np.asarray(row["cad_intersection_mm"])
        assert abs(np.dot(equation.normal, x-equation.anchor_mm)-equation.offset_from_anchor_mm) <= 1e-8
    assert row["translation_rule"] == "CAD_INTERSECTION_MINUS_R_SCAN_INTERSECTION_NOT_PARTIAL_CENTROIDS"


def test_orthogonal_scheduled_subset_has_independent_signed_permutation_oracle():
    observed, target = _case(orthogonal=True, rotation=np.eye(3), translation=np.zeros(3))
    result = _run(observed, target)
    _complete(result)
    _reconstruct_join(result, observed, target)
    assert len(result["unsigned_assignments"]) == 4 and len(result["hypotheses"]) == 32
    assert result["statistics"]["compatible_unsigned_occurrences"] == 6
    assert result["statistics"]["unscheduled_compatible_unsigned_occurrences"] == 2
    scan = {row.equation.plane_id: np.asarray(row.equation.normal) for row in observed}
    cad = {row.plane_id: np.asarray(row.normal) for row in target}
    for row in result["hypotheses"]:
        rotation = sum((pair["normal_sign"]*np.outer(cad[pair["cad_plane_id"]], scan[pair["scan_plane_id"]])
                        for pair in row["correspondences"]), start=np.zeros((3, 3)))
        proper = np.linalg.det(rotation) > 0
        assert (row["status"] == "PROPOSED") is bool(proper)
        if proper: np.testing.assert_allclose(row["rotation"], rotation, atol=1e-12, rtol=0.)
        else: assert row["rotation"] is row["translation_mm"] is None
    assert len(_pose_rows(result)) == 16  # Not all 24 proper cube alternatives.


def _transport(equation, rotation, shift):
    return replace(equation, normal=_tuple(rotation@equation.normal),
                   anchor_mm=_tuple(rotation@equation.anchor_mm+shift))


@pytest.mark.parametrize("mode", ("anchor", "centroid", "sign", "joint_coefficient", "independent_frames"))
def test_equation_and_frame_changes_do_not_create_centroid_translation_shortcuts(mode):
    observed, target = _case()
    expected_rotation, expected_translation = ROTATION, TRANSLATION
    if mode == "anchor":
        def moved(equation):
            delta = np.array([13., -7., 9.])
            return replace(equation, anchor_mm=_tuple(np.asarray(equation.anchor_mm)+delta),
                offset_from_anchor_mm=equation.offset_from_anchor_mm-float(np.dot(equation.normal, delta)))
        observed = tuple(replace(row, equation=moved(row.equation)) for row in observed)
        target = tuple(moved(row) for row in target)
    elif mode == "centroid":
        observed = tuple(replace(row, support_centroid_mm=(99999.-i, -4567., 6789.)) for i, row in enumerate(observed))
    elif mode == "sign":
        target = tuple(replace(row, normal=_tuple(-np.asarray(row.normal)), offset_from_anchor_mm=-row.offset_from_anchor_mm)
                       if i in (0, 2) else row for i, row in enumerate(target))
    elif mode == "joint_coefficient":
        target = tuple(replace(row, normal=_tuple(np.asarray(row.normal)*(1.+5e-9)),
                               offset_from_anchor_mm=row.offset_from_anchor_mm*(1.+5e-9)) for row in target)
    else:
        qs, qc, cs, cc = ROTATION, ROTATION.T, np.array([10000., -20000., 30000.]), np.array([-7000., 11000., -13000.])
        observed = tuple(replace(row, equation=_transport(row.equation, qs, cs),
                                support_centroid_mm=_tuple(qs@row.support_centroid_mm+cs)) for row in observed)
        target = tuple(_transport(row, qc, cc) for row in target)
        expected_rotation = qc@ROTATION@qs.T
        expected_translation = qc@TRANSLATION+cc-expected_rotation@cs
    result = _run(observed, target)
    _complete(result)
    assert len(_pose_rows(result)) == 1
    np.testing.assert_allclose(_pose_rows(result)[0]["rotation"], expected_rotation, atol=2e-12, rtol=0.)
    np.testing.assert_allclose(_pose_rows(result)[0]["translation_mm"], expected_translation, atol=1e-8, rtol=0.)


def test_imperfect_normal_congruence_still_maps_intrinsic_intersections_covariantly():
    observed, target = _case(rotation=np.eye(3), translation=np.zeros(3))
    c, s = math.cos(.005), math.sin(.005)
    perturbed = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])@target[1].normal
    target = tuple(_equation(row.plane_id, perturbed if i == 1 else row.normal,
                            INTERSECTION, row.anchor_mm, .01) for i, row in enumerate(target))
    first = _run(observed, target)
    _complete(first)
    assert len(_pose_rows(first)) == 1
    row = _pose_rows(first)[0]
    assert max(row["normal_alignment_residuals_radians"]) > 1e-5
    r, t = np.asarray(row["rotation"]), np.asarray(row["translation_mm"])
    np.testing.assert_allclose(r@INTERSECTION+t, INTERSECTION, atol=1e-10, rtol=0.)
    cs, cc = np.array([1000., -2000., 3000.]), np.array([-700., 1100., -1300.])
    shifted_scan = tuple(replace(item, equation=_transport(item.equation, np.eye(3), cs)) for item in observed)
    shifted_cad = tuple(_transport(item, np.eye(3), cc) for item in target)
    second = _run(shifted_scan, shifted_cad)
    _complete(second)
    assert len(_pose_rows(second)) == 1
    np.testing.assert_allclose(_pose_rows(second)[0]["rotation"], r, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(_pose_rows(second)[0]["translation_mm"], t+cc-r@cs, atol=1e-8, rtol=0.)


def test_heterogeneous_cones_admit_identity_even_when_nominal_wahba_witness_rejects():
    # Freeze equal centroids so the orthogonal bank is in increasing-ID order.
    observed = tuple(replace(_observation(_equation(10+i, normal)), support_centroid_mm=(0., 0., 0.))
                     for i, normal in enumerate(np.eye(3)))
    target = (_equation(110, (math.cos(.1), math.sin(.1), 0.), alpha=.1),
              _equation(111, (0., 1., 0.)), _equation(112, (0., 0., 1.)))
    result = _run(observed, target)
    _complete(result)
    matching = [row for row in result["hypotheses"] if all(pair["cad_plane_id"] == pair["scan_plane_id"]+100
                and pair["normal_sign"] == 1 for pair in row["correspondences"])]
    assert len(matching) == 1
    row = matching[0]
    assert row["status"] == "REJECTED" and row["reasons"] == ["NOMINAL_WAHBA_WITNESS_OUTSIDE_DECLARED_GATE"]
    residuals = dict(zip((pair["scan_plane_id"] for pair in row["correspondences"]), row["normal_alignment_residuals_radians"]))
    np.testing.assert_allclose([residuals[10], residuals[11], residuals[12]], [.05, .05, 0.], atol=1e-12, rtol=0.)
    # Independent feasible identity witness: only the first nominal normal is
    # rotated, by exactly its declared .1 cone; the other two are unchanged.
    for source, cad in zip(observed, target):
        assert _pair_angle(source.equation.normal, cad.normal) <= cad.normal_angle_bound_radians+1e-12


def test_reflected_unoriented_triad_has_a_proper_all_negative_alternative():
    reflection = np.diag([-1., 1., 1.])
    observed, target = _case(rotation=reflection, translation=np.zeros(3))
    result = _run(observed, target)
    _complete(result)
    assert len(_pose_rows(result)) == 1
    row = _pose_rows(result)[0]
    assert row["normal_signs"] == [-1, -1, -1]
    np.testing.assert_allclose(row["rotation"], -reflection, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(row["translation_mm"], 2*reflection@INTERSECTION, atol=1e-10, rtol=0.)
    assert np.linalg.det(row["rotation"]) > .999999999


@pytest.mark.parametrize("mode", ("parallel", "near_singular", "cone_unresolved"))
def test_geometrically_unresolved_triplets_are_complete_refusals_not_empty_evidence(mode):
    normals = np.eye(3)
    if mode == "parallel": normals[2] = normals[0]
    if mode == "near_singular": normals[2] = np.array([1., 0., .001])/math.sqrt(1.+1e-6)
    observed = tuple(_observation(_equation(i, normal, alpha=.7 if mode == "cone_unresolved" else 0.), index=i)
                     for i, normal in enumerate(normals))
    target = tuple(_equation(i, normal) for i, normal in enumerate(np.eye(3)))
    result = _run(observed, target)
    _complete(result)
    assert result["hypotheses"] == [] and result["statistics"]["pair_join_tests"] == 0
    analysis = result["scan_triplets"][0]["analysis"]
    assert analysis["reasons"] == (["INDEPENDENCE_UNRESOLVED_WITH_DECLARED_NORMAL_CONES"] if mode == "cone_unresolved"
                                     else ["NOMINAL_TRIPLET_ILL_CONDITIONED"])
    assert result["scan_triplets"][0]["status"] == "SCAN_TRIPLET_UNAVAILABLE_NO_CAD_JOIN"


def test_nominal_intersection_is_rechecked_in_world_equations(monkeypatch):
    original = np.linalg.solve
    monkeypatch.setattr(search.np.linalg, "solve", lambda a, b: original(a, b)+np.array([.1, 0., 0.]))
    result = _run()
    _complete(result)
    analysis = result["scan_triplets"][0]["analysis"]
    assert analysis["reasons"] == ["INTERSECTION_BACKWARD_RESIDUAL_EXCEEDS_GATE"]
    assert analysis["backward_residual_max_mm"] > .05 and result["hypotheses"] == []


def test_thirty_cad_planes_masks_and_nonexhaustive_omissions_are_reconstructable():
    observed = tuple(_observation(_equation(i, normal), index=i) for i, normal in enumerate(np.eye(3)))
    target = tuple(_equation(100+i, np.eye(3)[i % 3], point=(i, -i, i/2)) for i in range(30))
    result = _run(observed, target)
    _complete(result)
    _reconstruct_join(result, observed, target)
    stats = result["statistics"]
    assert stats["input_cad_planes"] == 30 and stats["cad_pair_records"] == 435
    assert stats["indexed_pair_window_occurrences"] == 300 and stats["scheduled_pair_occurrences"] == 12
    assert stats["unscheduled_index_window_occurrences"] == 288
    assert stats["compatible_unsigned_occurrences"] == 240
    assert stats["unscheduled_compatible_unsigned_occurrences"] == 236
    assert len(result["cad_planes"]) == 30 and len(result["hypotheses"]) == 32


def test_unequal_cones_coarse_window_false_positives_are_explicit_rejections():
    observed = tuple(_observation(_equation(i, normal), index=i) for i, normal in enumerate(np.eye(3)))
    target = tuple(_equation(i, normal) for i, normal in enumerate(np.eye(3)))+(
        _equation(3, (math.cos(.1), math.sin(.1), 0.)), _equation(4, (1., 0., 0.), alpha=.7))
    result = _run(observed, target)
    _complete(result)
    _reconstruct_join(result, observed, target)
    assert result["statistics"]["tested_incompatible_pair_occurrences"] > 0


def test_bank_origin_round_robin_omitted_ids_and_triplet_anchor_provenance():
    kinds = ("CORE", "CLOSURE", "DECLARED")
    observed = tuple(_observation(_equation(i, np.eye(3)[i % 3]), index=i, kind=kinds[i % 3]) for i in range(30))
    target = tuple(_equation(i, normal) for i, normal in enumerate(np.eye(3)))
    result = _run(observed, target, budgets=replace(Budgets(), max_bank_planes=4))
    _complete(result)
    bank = result["scan_bank"]
    lookup = {row.equation.plane_id: row for row in observed}
    assert [lookup[i].proposal_kind for i in bank] == ["CLOSURE", "CORE", "DECLARED", "CLOSURE"]
    assert result["statistics"]["unscheduled_eligible_scan_planes"] == 26
    assert {row["equation"]["plane_id"] for row in result["scan_planes"] if row["disposition"] == "NOT_SCHEDULED_IN_SCAN_BANK"} == set(range(30))-set(bank)
    assert len(result["anchor_schedules"]) == 4
    triplets = {row["scan_triplet_id"]: row for row in result["scan_triplets"]}
    for schedule in result["anchor_schedules"]:
        assert schedule["partner_pair_count"] == 3 and schedule["unscheduled_partner_pair_count"] == 1
        assert len(schedule["scheduled_scan_triplet_ids"]) == 2
        for identifier in schedule["scheduled_scan_triplet_ids"]:
            assert schedule["scan_plane_id"] in triplets[identifier]["scan_plane_ids"]
            assert schedule["scan_plane_id"] in triplets[identifier]["origin_anchor_plane_ids"]
    assert result == _run(tuple(reversed(observed)), tuple(reversed(target)), budgets=replace(Budgets(), max_bank_planes=4))


def test_declared_support_exclusions_are_retained_not_repaired_by_cad():
    observed, target = _case()
    observed += (_observation(_equation(20, (1., 0., 0.)), count=2),
                 _observation(_equation(21, (0., 1., 0.)), rms=.100001))
    result = _run(observed, target)
    _complete(result)
    excluded = {row["equation"]["plane_id"]: row["reasons"] for row in result["scan_planes"] if row["disposition"] == "EXCLUDED_BY_DECLARED_SUPPORT_POLICY"}
    assert excluded == {20: ["INSUFFICIENT_DECLARED_SCAN_SUPPORT"], 21: ["DECLARED_SCAN_RMS_EXCEEDS_GATE"]}
    assert len(_pose_rows(result)) == 1


@pytest.mark.parametrize("field,exact,reason", (
    ("max_scan_planes", 3, "PLANE_TABLE_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_cad_planes", 3, "PLANE_TABLE_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_scan_triplets", 1, "SCAN_TRIPLET_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_scan_partner_tests", 3, "SCAN_PARTNER_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_cad_pair_records", 3, "CAD_PAIR_INDEX_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_pair_join_tests", 15, "PAIR_JOIN_WORST_CASE_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_compatible_rows_per_triplet", 6, "CAD_JOIN_BUFFER_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_unsigned_assignments", 4, "ASSIGNMENT_PREFLIGHT_BUDGET_EXCEEDED"),
    ("max_signed_assignments", 32, "ASSIGNMENT_PREFLIGHT_BUDGET_EXCEEDED")))
def test_tiny_actual_preflight_exact_boundary_and_one_below(field, exact, reason):
    observed, target = _case(orthogonal=True)
    allowed = _run(observed, target, budgets=replace(Budgets(), **{field: exact}))
    _complete(allowed)
    assert len(allowed["hypotheses"]) == 32
    denied = _run(observed, target, budgets=replace(Budgets(), **{field: exact-1}))
    _refused(denied, reason)
    assert denied["hypotheses"] == []


@pytest.mark.parametrize("field", BUDGET_FIELDS)
def test_every_fixed_hard_cap_rejects_widening_before_search(field, monkeypatch):
    monkeypatch.setattr(search, "_bank", lambda *a: pytest.fail("Invalid cap reached bank search"))
    _refused(_run(budgets=replace(Budgets(), **{field: getattr(Budgets(), field)+1})), "INVALID_PLANE_SEARCH_BUDGET")


@pytest.mark.parametrize("field", ("max_bank_planes", "max_cad_pair_samples_per_triplet", "max_unsigned_per_scan_triplet"))
def test_zero_schedule_quotas_are_complete_omissions_not_physical_absence(field):
    result = _run(budgets=replace(Budgets(), **{field: 0}))
    _complete(result)
    assert result["hypotheses"] == [] and not result["usable_hypotheses"]
    assert len(result["scan_planes"]) == len(result["cad_planes"]) == 3


@pytest.mark.parametrize("quota", (1, 2, 3, 4))
def test_compatible_assignment_strata_are_middle_or_spread_not_prefix(quota):
    observed, target = _case(orthogonal=True, rotation=np.eye(3), translation=np.zeros(3))
    result = _run(observed, target, budgets=replace(Budgets(), max_unsigned_per_scan_triplet=quota, chunk_pair_rows=1))
    _complete(result)
    _reconstruct_join(result, observed, target)
    assert len(result["hypotheses"]) == 8*quota
    assert result["scan_triplets"][0]["scheduled_compatible_assignment_positions"] == _strata(6, quota)


@pytest.mark.parametrize("bad", (None, "", "   ", 1, "x"*129))
def test_explicit_unit_assumption_is_required(bad):
    observed, target = _case()
    _refused(search.propose_plane_search_v5(observed, target, unit_assumption=bad), "EXPLICIT_MM_UNIT_ASSUMPTION_REQUIRED")


@pytest.mark.parametrize("mode", ("table_list", "duplicate_scan", "duplicate_cad", "bool_id", "numpy_scalar",
    "nonfinite_anchor", "oversized_anchor", "list_vector", "nonunit", "negative_cone", "wide_cone",
    "bool_support", "bad_kind", "subclass_equation", "subclass_budget", "mutated_policy", "mutated_budget"))
def test_invalid_or_bypass_mutated_inputs_never_publish_a_pose(mode):
    observed, target = _case()
    kwargs = {}
    if mode == "table_list": observed = list(observed)
    elif mode == "duplicate_scan": observed = observed+(observed[0],)
    elif mode == "duplicate_cad": target = target+(target[0],)
    elif mode == "bool_support": observed = (replace(observed[0], support_count=True), *observed[1:])
    elif mode == "bad_kind": observed = (replace(observed[0], proposal_kind="TRUTH_FLANGE"), *observed[1:])
    elif mode == "subclass_budget":
        class Extended(Budgets): pass
        kwargs["budgets"] = Extended()
    elif mode == "mutated_policy":
        value = Policy()
        object.__setattr__(value, "min_scan_support_count", False)
        kwargs["policy"] = value
    elif mode == "mutated_budget":
        value = Budgets()
        object.__setattr__(value, "max_cad_planes", 2049)
        kwargs["budgets"] = value
    else:
        equation = target[0]
        if mode == "bool_id": equation = replace(equation, plane_id=True)
        elif mode == "numpy_scalar": equation = replace(equation, offset_from_anchor_mm=np.float64(0.))
        elif mode == "nonfinite_anchor": equation = replace(equation, anchor_mm=(math.nan, 0., 0.))
        elif mode == "oversized_anchor": equation = replace(equation, anchor_mm=(1e6+1, 0., 0.))
        elif mode == "list_vector": equation = replace(equation, normal=list(equation.normal))
        elif mode == "nonunit": equation = replace(equation, normal=(2., 0., 0.))
        elif mode == "negative_cone": equation = replace(equation, normal_angle_bound_radians=-.1)
        elif mode == "wide_cone": equation = replace(equation, normal_angle_bound_radians=math.pi/4.+1e-6)
        else:
            class Extended(Equation): pass
            equation = Extended(equation.plane_id, equation.normal, equation.anchor_mm, equation.offset_from_anchor_mm)
        target = (equation, *target[1:])
    _refused(_run(observed, target, **kwargs))


def test_snapshot_detachment_policy_fields_and_canonical_input_digests():
    observed, target = _case()
    policy, budgets = Policy(), Budgets()
    result = _run(observed, target, policy=policy, budgets=budgets)
    _complete(result)
    snapshot = deepcopy(result)
    assert tuple(field.name for field in fields(Budgets)) == BUDGET_FIELDS
    assert result == _run(tuple(reversed(observed)), tuple(reversed(target)))
    object.__setattr__(observed[0].equation, "normal", (0., 1., 0.))
    object.__setattr__(observed[0], "support_centroid_mm", (0., 0., 0.))
    object.__setattr__(target[0], "offset_from_anchor_mm", -999.)
    object.__setattr__(policy, "min_scan_support_count", 999)
    object.__setattr__(budgets, "max_signed_assignments", 0)
    assert result == snapshot
    assert all(len(value) == 64 for value in result["input_digests"].values())
    json.dumps(result, allow_nan=False)


def test_late_svd_exception_invalidates_prior_proposal_and_current_evaluating_record(monkeypatch):
    observed, target = _case(orthogonal=True, rotation=np.eye(3), translation=np.zeros(3))
    observed = tuple(replace(row, support_centroid_mm=(0., 0., 0.)) for row in observed)
    original, calls = np.linalg.svd, []
    def failing_svd(*args, **kwargs):
        calls.append(None)
        if len(calls) == 5: raise np.linalg.LinAlgError("INVENTED_LATE_SVD_FAILURE")
        return original(*args, **kwargs)
    monkeypatch.setattr(search.np.linalg, "svd", failing_svd)
    result = _run(observed, target)
    _refused(result, "INVENTED_LATE_SVD_FAILURE")
    stats = result["statistics"]
    assert stats["signed_assignments_attempted"] == 3 and stats["signed_assignments_evaluated"] == 2
    assert stats["proposals_invalidated_on_failure"] == 1 and stats["proposed_hypotheses"] == 0
    assert [row["status_before_failure"] for row in result["hypotheses"]] == ["REJECTED", "PROPOSED", "EVALUATING"]
    triplet = result["scan_triplets"][0]
    assert triplet["status"] == "INTERRUPTED_BY_FATAL_SEARCH_FAILURE" and triplet["status_before_failure"] == "EVALUATING"
    # All three unordered pairs and both third-plane orientations were actually
    # searched before the first unsigned refinement. Only one assignment began.
    for row in result["cad_planes"]:
        assert row["scheduled_pair_occurrences"] == 2
        assert row["third_plane_test_occurrences"] == 2
        assert row["scheduled_unsigned_occurrences"] == 1
        assert row["disposition"] == "USED_IN_SCHEDULED_UNSIGNED_ASSIGNMENT"


def test_final_report_overflow_discards_completed_report_not_a_qualified_prefix():
    observed, target = _case(orthogonal=True)
    full = _run(observed, target)
    _complete(full)
    cap = full["statistics"]["metadata_admission_bytes"]+1
    assert cap < len(json.dumps(full, allow_nan=False, separators=(",", ":")).encode())
    result = _run(observed, target, budgets=replace(Budgets(), max_report_bytes=cap))
    _refused(result, "SEARCH_REPORT_UNAVAILABLE_NO_PREFIX")
    assert result["hypotheses"] == [] and result["discarded_hypotheses"] == 32
    assert result["attempted_statistics"]["signed_assignments_evaluated"] == 32
    assert result["evidence_scope"] == "NO_PARTIAL_REPORT_OR_POSE_PUBLISHED"


def test_zero_detailed_report_budget_returns_only_unusable_refusal_envelope():
    result = _run(budgets=replace(Budgets(), max_report_bytes=0))
    _refused(result, "SEARCH_REPORT_UNAVAILABLE_NO_PREFIX")
    assert result["hypotheses"] == [] and "scan_planes" not in result


def test_large_pair_index_preflight_refuses_before_index_allocation(monkeypatch):
    observed, _ = _case()
    target = tuple(_equation(i, np.eye(3)[i % 3]) for i in range(30))
    monkeypatch.setattr(search, "_pair_table", lambda *a: pytest.fail("Pair cap must precede pair-index allocation"))
    _refused(_run(observed, target, budgets=replace(Budgets(), max_cad_pair_records=434)),
             "CAD_PAIR_INDEX_PREFLIGHT_BUDGET_EXCEEDED")
