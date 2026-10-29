"""Independent supplied-plane algebra and refusal controls; no extraction.

Every plane below is invented analytically. No native reader, PLY, scan fixture,
reference pose input to registration, physical part or retained run is used.
These checks qualify only a bounded proposal primitive, never accepted poses.
The parent must review source and tests before the first supervised execution.
"""

from dataclasses import fields, replace
from itertools import permutations, product
import json
import math

import numpy as np
import pytest

from backend import cad_guided_plane_hypotheses_v4 as v4


RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
                       process_tree_rss_bytes=1024**3)
NORMALS = np.array([[1., 0., 0.], [.8, .6, 0.], [0., .6, .8]])
INTERSECTION = np.array([-2., 3., 4.])
ROTATION = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
TRANSLATION = np.array([5., -7., 11.])


def _equation(identifier, normal, offset):
    return v4.PlaneEquationV4(int(identifier), tuple(map(float, normal)), float(offset))


def _observed(equation, **changes):
    return replace(v4.ObservedPlaneV4(equation, 12, 0., (6., 2.)), **changes)


def _fixture(normals=NORMALS, rotation=ROTATION, translation=TRANSLATION):
    scan = tuple(_observed(_equation(identifier, normal, normal @ INTERSECTION))
                 for identifier, normal in zip((7, 13, 29), normals))
    cad = tuple(_equation(identifier, rotation @ normal,
                         normal @ INTERSECTION + (rotation @ normal) @ translation)
                for identifier, normal in zip((107, 113, 129), normals))
    return scan, cad


def _quaternion_rotation(vector):
    angle = math.sqrt(math.fsum(float(value)**2 for value in vector))
    if angle == 0.:
        return np.eye(3)
    w = math.cos(angle / 2.)
    x, y, z = np.asarray(vector) * (math.sin(angle / 2.) / angle)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _move_equation(equation, rotation, shift):
    normal = rotation @ np.asarray(equation.normal)
    return _equation(equation.plane_id, normal, equation.offset_mm + normal @ shift)


def _pose(hypothesis):
    return np.asarray(hypothesis["rotation"]), np.asarray(hypothesis["translation_mm"])


def _complete(result, count):
    assert result["status"] == "COMPLETE", result
    assert result["search_complete"] is True
    assert len(result["hypotheses"]) == count
    statistics = result["statistics"]
    assert statistics["signed_assignments_accounted"] == statistics["planned_signed_assignments"]
    assert statistics["qualified_proposal_assignments"] == count
    assert sum(statistics["rejection_counts"].values()) + count == statistics["planned_signed_assignments"]
    assert statistics["hypotheses_discarded_on_failure"] == 0
    assert result["accepted_registration"] is False
    assert result["registration_status"] == "NOT_EVALUATED"
    assert result["candidates_only"] is True
    assert result["support_provenance"] == "UNVERIFIED_DECLARATIONS"
    assert result["support_raw_row_provenance_verified"] is False
    assert result["finite_support_match_verified"] is False
    assert result["direct_offset_diagnostic_is_origin_invariant"] is False
    assert result["production_use_allowed"] is False
    # Publication must remain ordinary finite JSON, not NumPy objects or NaN.
    json.dumps(result, allow_nan=False)
    for hypothesis in result["hypotheses"]:
        rotation, translation = _pose(hypothesis)
        assert rotation.shape == (3, 3) and translation.shape == (3,)
        assert np.isfinite(rotation).all() and np.isfinite(translation).all()
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-12)
        assert len(hypothesis["correspondences"]) == 3
    return result["hypotheses"]


def _refused(result, reason=None):
    assert result["status"] == "UNSUPPORTED", result
    assert result["search_complete"] is False
    assert result["hypotheses"] == []
    assert result["reason"]
    if reason is not None:
        assert result["reason"] == reason
    assert result["accepted_registration"] is False
    assert result["production_use_allowed"] is False
    json.dumps(result, allow_nan=False)


def test_generic_nonorthogonal_planes_recover_unique_analytic_proposal():
    scan, cad = _fixture()
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    hypothesis, = _complete(result, 1)
    rotation, translation = _pose(hypothesis)
    np.testing.assert_allclose(rotation, ROTATION, rtol=0., atol=1e-12)
    np.testing.assert_allclose(translation, TRANSLATION, rtol=0., atol=1e-12)
    np.testing.assert_allclose(hypothesis["scan_intersection_mm"], INTERSECTION, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(hypothesis["cad_intersection_mm"],
                               ROTATION @ INTERSECTION + TRANSLATION, atol=1e-12, rtol=0.)
    assert hypothesis["correspondences"] == [
        dict(scan_plane_id=s, cad_plane_id=c, normal_sign=1)
        for s, c in zip((7, 13, 29), (107, 113, 129))]
    assert hypothesis["normal_max_residual_radians"] < 1e-12
    assert hypothesis["direct_offset_discrepancy_max_mm"] < 1e-12
    assert result["statistics"]["planned_signed_assignments"] == 48
    assert result["statistics"]["normal_alignments_attempted"] == 48


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_input_order_does_not_choose_a_pose_or_change_enumeration(order):
    scan, cad = _fixture()
    baseline = v4.propose_plane_hypotheses_v4(scan, cad)
    changed = v4.propose_plane_hypotheses_v4(tuple(scan[i] for i in order),
                                           tuple(cad[i] for i in reversed(order)))
    assert changed == baseline


@pytest.mark.parametrize("signs", tuple(product((-1, 1), repeat=3)))
def test_unoriented_plane_signs_preserve_geometry_and_declared_correspondence_sign(signs):
    scan, cad = _fixture()
    cad_signs = (1, -1, 1)
    scan = tuple(replace(row, equation=_equation(row.equation.plane_id,
        sign*np.asarray(row.equation.normal), sign*row.equation.offset_mm))
        for row, sign in zip(scan, signs))
    cad = tuple(_equation(row.plane_id, sign*np.asarray(row.normal), sign*row.offset_mm)
                for row, sign in zip(cad, cad_signs))
    hypothesis, = _complete(v4.propose_plane_hypotheses_v4(scan, cad), 1)
    rotation, translation = _pose(hypothesis)
    np.testing.assert_allclose(rotation, ROTATION, rtol=0., atol=1e-12)
    np.testing.assert_allclose(translation, TRANSLATION, rtol=0., atol=1e-12)
    assert [row["normal_sign"] for row in hypothesis["correspondences"]] == [
        source*target for source, target in zip(signs, cad_signs)]


def test_orthogonal_trihedron_retains_all_24_proper_alternatives_not_a_winner():
    scan, cad = _fixture(np.eye(3), np.eye(3), np.zeros(3))
    hypotheses = _complete(v4.propose_plane_hypotheses_v4(scan, cad), 24)
    expected = []
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3))
            for source, target in enumerate(order):
                matrix[target, source] = signs[source]
            if round(np.linalg.det(matrix)) == 1:
                expected.append(matrix)
    assert len(expected) == 24
    for matrix in expected:
        matching = [h for h in hypotheses if np.allclose(h["rotation"], matrix, rtol=0., atol=1e-12)]
        assert len(matching) == 1
        np.testing.assert_allclose(matching[0]["translation_mm"],
                                   INTERSECTION - matrix @ INTERSECTION, rtol=0., atol=1e-12)


def test_plane_ids_are_only_local_traversal_identity_not_correspondence_features():
    scan, cad = _fixture()
    scan = tuple(replace(row, equation=replace(row.equation, plane_id=identifier))
                 for row, identifier in zip(scan, (91, 3, 41)))
    cad = tuple(replace(row, plane_id=identifier) for row, identifier in zip(cad, (41, 91, 3)))
    hypothesis, = _complete(v4.propose_plane_hypotheses_v4(scan, cad), 1)
    np.testing.assert_allclose(hypothesis["rotation"], ROTATION, rtol=0., atol=1e-12)
    np.testing.assert_allclose(hypothesis["translation_mm"], TRANSLATION, rtol=0., atol=2e-12)
    assert {(row["scan_plane_id"], row["cad_plane_id"]) for row in hypothesis["correspondences"]} == {
        (91, 41), (3, 91), (41, 3)}


@pytest.mark.parametrize("offset_change", (0., 1.))
def test_duplicate_or_parallel_inner_outer_plane_records_preserve_all_assignments(offset_change):
    scan, cad = _fixture()
    cad += (_equation(199, cad[0].normal, cad[0].offset_mm + offset_change),)
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    hypotheses = _complete(result, 2)
    # The dual direction solves [x, .8x+.6y, .6y+.8z] = [1,0,0].
    change = ROTATION @ np.array([1., -4./3., 1.]) * offset_change
    for expected_id, expected_translation in ((107, TRANSLATION), (199, TRANSLATION + change)):
        matches = [h for h in hypotheses if h["correspondences"][0]["cad_plane_id"] == expected_id]
        assert len(matches) == 1
        np.testing.assert_allclose(matches[0]["translation_mm"], expected_translation, rtol=0., atol=2e-12)
        np.testing.assert_allclose(matches[0]["rotation"], ROTATION, rtol=0., atol=1e-12)
    assert result["statistics"]["planned_signed_assignments"] == 192
    assert result["statistics"]["normal_alignments_attempted"] == 96
    assert result["statistics"]["rejection_counts"]["ILL_CONDITIONED_CAD_TRIPLET"] == 96


@pytest.mark.parametrize("scale", (1e-3, 1., 1e3))
def test_consistently_rescaled_dimensional_policy_preserves_proposal(scale):
    scan, cad = _fixture()
    scan = tuple(replace(row, equation=_equation(row.equation.plane_id,
        row.equation.normal, row.equation.offset_mm*scale),
        support_rms_mm=row.support_rms_mm*scale,
        support_extent_mm=tuple(value*scale for value in row.support_extent_mm)) for row in scan)
    cad = tuple(_equation(row.plane_id, row.normal, row.offset_mm*scale) for row in cad)
    policy = replace(v4.PlaneHypothesisPolicyV4(), max_scan_support_rms_mm=.1*scale,
        min_scan_second_extent_mm=.25*scale, max_intersection_backward_residual_mm=1e-8*scale)
    hypothesis, = _complete(v4.propose_plane_hypotheses_v4(scan, cad, policy=policy), 1)
    np.testing.assert_allclose(hypothesis["rotation"], ROTATION, rtol=0., atol=1e-12)
    np.testing.assert_allclose(np.asarray(hypothesis["translation_mm"])/scale,
                               TRANSLATION, rtol=0., atol=2e-12)


@pytest.mark.parametrize("normal_perturbation", (0., .003))
def test_independent_frames_preserve_pose_even_for_imperfect_normal_congruence(normal_perturbation):
    scan, cad = _fixture()
    target_intersection = ROTATION @ INTERSECTION + TRANSLATION
    moved_normal = _quaternion_rotation([normal_perturbation, -.4*normal_perturbation, 0.]) @ np.asarray(cad[1].normal)
    cad = (cad[0], _equation(cad[1].plane_id, moved_normal,
                           moved_normal @ target_intersection), cad[2])
    baseline, = _complete(v4.propose_plane_hypotheses_v4(scan, cad), 1)
    rotation, translation = _pose(baseline)
    if normal_perturbation:
        assert baseline["normal_max_residual_radians"] > 1e-5
    source_frame = _quaternion_rotation([.23, -.17, .31])
    target_frame = _quaternion_rotation([-.11, .37, .19])
    source_shift = np.array([2000., -3000., 4000.])
    target_shift = np.array([-1100., 1700., 2300.])
    changed_scan = tuple(replace(row, equation=_move_equation(row.equation,
                         source_frame, source_shift)) for row in scan)
    changed_cad = tuple(_move_equation(row, target_frame, target_shift) for row in cad)
    changed, = _complete(v4.propose_plane_hypotheses_v4(changed_scan, changed_cad), 1)
    expected_rotation = target_frame @ rotation @ source_frame.T
    expected_translation = target_frame @ translation + target_shift - expected_rotation @ source_shift
    np.testing.assert_allclose(changed["rotation"], expected_rotation, rtol=0., atol=2e-12)
    np.testing.assert_allclose(changed["translation_mm"], expected_translation, rtol=0., atol=2e-8)
    np.testing.assert_allclose(np.asarray(changed["rotation"]) @ np.asarray(changed["scan_intersection_mm"])
        + changed["translation_mm"], changed["cad_intersection_mm"], rtol=0., atol=2e-9)
    if normal_perturbation:
        # Direct-offset discrepancy changes with origin and MUST NOT be a gate.
        assert changed["direct_offset_discrepancy_max_mm"] > .1
        assert changed["normal_max_residual_radians"] < .02


def test_near_unit_normal_rescaling_preserves_both_plane_coefficients():
    scan, cad = _fixture()
    factor = 1. + 4e-9
    # Non-axis normal components remain below one under the explicit admission.
    original = scan[1].equation
    scaled = _equation(original.plane_id, factor*np.asarray(original.normal), factor*original.offset_mm)
    scan = (scan[0], replace(scan[1], equation=scaled), scan[2])
    hypothesis, = _complete(v4.propose_plane_hypotheses_v4(scan, cad), 1)
    np.testing.assert_allclose(hypothesis["translation_mm"], TRANSLATION, rtol=0., atol=2e-12)
    np.testing.assert_allclose(hypothesis["scan_intersection_mm"], INTERSECTION, rtol=0., atol=2e-12)


@pytest.mark.parametrize("role", ("SCAN", "CAD"))
@pytest.mark.parametrize("third", ((.8, .6, 0.), (math.sqrt(1.-1e-8), 0., 1e-4)))
def test_each_frame_independently_requires_a_well_conditioned_normal_triplet(role, third):
    scan, cad = _fixture()
    normals = np.array([[1., 0., 0.], [0., 1., 0.], third])
    if role == "SCAN":
        scan = tuple(_observed(_equation(index, normal, normal @ INTERSECTION))
                     for index, normal in enumerate(normals))
    else:
        cad = tuple(_equation(index, normal, normal @ INTERSECTION)
                    for index, normal in enumerate(normals))
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    _complete(result, 0)
    assert result["statistics"]["normal_alignments_attempted"] == 0
    assert result["statistics"]["rejection_counts"] == {"ILL_CONDITIONED_"+role+"_TRIPLET": 48}


def test_independent_but_noncongruent_normal_triads_refuse_all_signed_permutations():
    scan, _ = _fixture(np.eye(3))
    _, cad = _fixture()
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    _complete(result, 0)
    assert result["reason"] == "NO_ADMISSIBLE_PLANE_HYPOTHESES"
    assert result["statistics"]["normal_alignments_attempted"] == 48
    assert result["statistics"]["rejection_counts"] == {"NORMAL_ALIGNMENT_RESIDUAL_EXCEEDS_GATE": 48}


@pytest.mark.parametrize("role", ("scan", "cad"))
def test_two_planes_leave_translation_unresolved_without_centroid_fill(role):
    scan, cad = _fixture()
    if role == "scan":
        scan = scan[:2]
    else:
        cad = cad[:2]
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    _refused(result, "FEWER_THAN_THREE_ADMISSIBLE_PLANES")
    assert result["statistics"]["normal_alignments_attempted"] == 0


@pytest.mark.parametrize("changes,reason", (
    ({"support_count": 11}, "INSUFFICIENT_DECLARED_PLANE_SUPPORT"),
    ({"support_rms_mm": .100001}, "DECLARED_PLANE_RMS_EXCEEDS_GATE"),
    ({"support_extent_mm": (6., .249999)}, "INSUFFICIENT_DECLARED_SECOND_SUPPORT_EXTENT"),
))
def test_declared_support_gates_do_not_silently_replace_the_third_plane(changes, reason):
    scan, cad = _fixture()
    scan = scan[:2] + (replace(scan[2], **changes),)
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    _refused(result, "FEWER_THAN_THREE_ADMISSIBLE_PLANES")
    assert result["excluded_scan_planes"] == [dict(plane_id=29, reasons=[reason])]
    assert result["statistics"]["eligible_scan_planes"] == 2
    assert len(result["scan_support_summaries"]) == 3


def test_exact_declared_support_boundaries_are_admitted_but_not_verified():
    scan, cad = _fixture()
    scan = tuple(replace(row, support_rms_mm=.1, support_extent_mm=(.25, .25)) for row in scan)
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    _complete(result, 1)
    assert result["excluded_scan_planes"] == []
    assert all(row["support_count"] == 12 for row in result["scan_support_summaries"])


@pytest.mark.parametrize("field,value,reason", (
    ("max_scan_planes", 2, "PLANE_INPUT_BUDGET_EXCEEDED"),
    ("max_cad_planes", 2, "PLANE_INPUT_BUDGET_EXCEEDED"),
    ("max_signed_assignments", 47, "PLANE_ASSIGNMENT_BUDGET_EXCEEDED"),
    ("max_signed_assignments", 0, "PLANE_ASSIGNMENT_BUDGET_EXCEEDED"),
))
def test_preflight_budget_refusal_occurs_before_any_numerical_triplet(monkeypatch, field, value, reason):
    scan, cad = _fixture()
    def forbidden(*args, **kwargs):
        pytest.fail("preflight cap must precede every numerical triplet")
    monkeypatch.setattr(v4.np.linalg, "svd", forbidden)
    result = v4.propose_plane_hypotheses_v4(scan, cad,
        budgets=replace(v4.PlaneHypothesisBudgetsV4(), **{field: value}))
    _refused(result, reason)
    assert result["statistics"]["normal_alignments_attempted"] == 0


@pytest.mark.parametrize("cap", (0, 1, 23))
def test_output_overflow_clears_all_previously_admitted_symmetric_hypotheses(cap):
    scan, cad = _fixture(np.eye(3), np.eye(3), np.zeros(3))
    result = v4.propose_plane_hypotheses_v4(scan, cad,
        budgets=replace(v4.PlaneHypothesisBudgetsV4(), max_hypotheses=cap))
    _refused(result, "PLANE_HYPOTHESIS_BUDGET_EXCEEDED")
    assert result["statistics"]["hypotheses_discarded_on_failure"] == cap
    assert result["statistics"]["qualified_proposal_assignments"] == cap + 1


def test_late_local_svd_failure_clears_an_already_admitted_prefix(monkeypatch):
    scan, cad = _fixture(np.eye(3), np.eye(3), np.zeros(3))
    original = v4.np.linalg.svd
    calls = []
    def fail_late(*args, **kwargs):
        calls.append(1)
        # Two triplet SVDs, then (-,-,-) refusal and (-,-,+) admission.
        if len(calls) == 5:
            raise np.linalg.LinAlgError("invented late arithmetic refusal")
        return original(*args, **kwargs)
    monkeypatch.setattr(v4.np.linalg, "svd", fail_late)
    result = v4.propose_plane_hypotheses_v4(scan, cad)
    _refused(result)
    assert len(calls) == 5
    assert result["statistics"]["hypotheses_discarded_on_failure"] == 1


def _bypass(value, **changes):
    copied = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(copied, field.name, changes.get(field.name, getattr(value, field.name)))
    return copied


@pytest.mark.parametrize("mutation", ("normal_nan", "normal_bool", "offset_inf", "support_bool", "extent_list"))
def test_late_bypass_mutated_record_is_revalidated_before_any_candidate(mutation):
    scan, cad = _fixture()
    last = scan[2]
    if mutation == "normal_nan":
        last = _bypass(last, equation=_bypass(last.equation, normal=(float("nan"), 0., 1.)))
    elif mutation == "normal_bool":
        last = _bypass(last, equation=_bypass(last.equation, normal=(False, .6, .8)))
    elif mutation == "offset_inf":
        last = _bypass(last, equation=_bypass(last.equation, offset_mm=float("inf")))
    elif mutation == "support_bool":
        last = _bypass(last, support_count=True)
    else:
        last = _bypass(last, support_extent_mm=[6., 2.])
    result = v4.propose_plane_hypotheses_v4(scan[:2] + (last,), cad)
    _refused(result)
    assert result["statistics"]["normal_alignments_attempted"] == 0


@pytest.mark.parametrize("role", ("scan", "cad"))
def test_reused_ids_within_one_frame_are_invalid_not_silently_deduplicated(role):
    scan, cad = _fixture()
    if role == "scan":
        scan = scan[:2] + (replace(scan[2], equation=replace(scan[2].equation, plane_id=7)),)
    else:
        cad = cad[:2] + (replace(cad[2], plane_id=107),)
    _refused(v4.propose_plane_hypotheses_v4(scan, cad), "DUPLICATE_PLANE_ID_WITHIN_FRAME")


@pytest.mark.parametrize("mutation", ("scan_list", "cad_list", "policy_dict", "budget_dict", "mutated_policy", "mutated_budget"))
def test_malformed_collections_or_policies_cannot_publish_a_prefix(mutation):
    scan, cad = _fixture()
    kwargs = {}
    if mutation == "scan_list":
        scan = list(scan)
    elif mutation == "cad_list":
        cad = list(cad)
    elif mutation == "policy_dict":
        kwargs["policy"] = {}
    elif mutation == "budget_dict":
        kwargs["budgets"] = {}
    elif mutation == "mutated_policy":
        kwargs["policy"] = _bypass(v4.PlaneHypothesisPolicyV4(), min_normal_condition_ratio=0.)
    else:
        kwargs["budgets"] = _bypass(v4.PlaneHypothesisBudgetsV4(), max_signed_assignments=True)
    _refused(v4.propose_plane_hypotheses_v4(scan, cad, **kwargs))


@pytest.mark.parametrize("kind", ("policy", "budgets"))
def test_extended_dataclass_subclasses_cannot_expand_the_admitted_contract(kind):
    parent = v4.PlaneHypothesisPolicyV4 if kind == "policy" else v4.PlaneHypothesisBudgetsV4
    class Extended(parent):
        extra_authority = True
    # Deliberately bypass the constructor to exercise call-boundary admission.
    value = object.__new__(Extended)
    template = parent()
    for field in fields(template):
        object.__setattr__(value, field.name, getattr(template, field.name))
    scan, cad = _fixture()
    result = v4.propose_plane_hypotheses_v4(scan, cad, **{kind: value})
    _refused(result)
    assert result["statistics"]["normal_alignments_attempted"] == 0


@pytest.mark.parametrize("kwargs", (
    {"plane_id": True}, {"plane_id": -1}, {"plane_id": 2**63},
    {"normal": [1., 0., 0.]}, {"normal": (True, 0., 0.)},
    {"normal": (2., 0., 0.)}, {"normal": (0., 0., 0.)},
    {"normal": (float("nan"), 0., 1.)}, {"offset_mm": float("inf")},
    {"offset_mm": 1_000_001.},
))
def test_equation_constructor_rejects_malformed_or_out_of_range_declarations(kwargs):
    values = dict(plane_id=1, normal=(1., 0., 0.), offset_mm=0.)
    values.update(kwargs)
    with pytest.raises(ValueError):
        v4.PlaneEquationV4(**values)


@pytest.mark.parametrize("field", ("max_scan_planes", "max_cad_planes", "max_signed_assignments", "max_hypotheses"))
def test_hard_work_caps_cannot_be_raised(field):
    budgets = v4.PlaneHypothesisBudgetsV4()
    with pytest.raises(ValueError):
        replace(budgets, **{field: getattr(budgets, field) + 1})


def test_returned_json_mutation_does_not_change_inputs_or_a_fresh_proposal():
    scan, cad = _fixture()
    original = v4.propose_plane_hypotheses_v4(scan, cad)
    mutated = v4.propose_plane_hypotheses_v4(scan, cad)
    mutated["hypotheses"][0]["rotation"][0][0] = 987.
    mutated["scan_support_summaries"][0]["equation"]["normal"][0] = -987.
    assert v4.propose_plane_hypotheses_v4(scan, cad) == original
    assert scan[0].equation.normal == (1., 0., 0.)
