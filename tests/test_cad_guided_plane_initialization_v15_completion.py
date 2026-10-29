"""Completion lane V1 of initializer V15 on an invented three-sheet (plus one parallel sheet) fixture.

Actual noisy-plane extraction, CAD patches, plane search, context V4, the frozen finite prefilter
on surface V4 and the additive completion lane run unmocked. The lane must recover the correct third
face when a two-of-three parent's third support was assigned to a coplanar or parallel wrong face,
while keeping every gate, counter and the main lane unchanged. No files, native CAD or physical data.
"""
import copy
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"


@pytest.fixture(scope="module")
def init():
    prefix = "backend"
    saved = {name: value for name, value in sys.modules.items() if name == prefix or name.startswith(prefix+".")}
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[prefix] = package
    try:
        yield importlib.import_module(prefix+".cad_guided_plane_initialization_v15")
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]
        sys.modules.update(saved)


def _grid(origin, e1, e2, coordinates_1, coordinates_2, alternate):
    points = np.asarray([np.asarray(origin, float)+a*np.asarray(e1, float)+b*np.asarray(e2, float)
                         for a in coordinates_1 for b in coordinates_2], dtype=np.float64)
    triangles, width = [], len(coordinates_2)
    for i in range(len(coordinates_1)-1):
        for j in range(width-1):
            a, b = i*width+j, (i+1)*width+j
            c, d = a+1, b+1
            triangles.extend(((a, b, c), (c, b, d)) if alternate else ((a, b, d), (a, d, c)))
    return points, np.asarray(triangles, dtype=np.int64)


def _arrays():
    """Scan: three orthogonal sheets on x=0, y=0, z=0. CAD: the same three faces (larger) plus a
    fourth face parallel to z=0 at z=10 (face id 41). Returned scan is rigidly moved."""
    lateral = (2., 5., 8., 11., 14.)
    axial = (-6., -3., 0., 3., 6.)
    scans = [_grid((0., 0., 0.), (0., 1., 0.), (0., 0., 1.), lateral, axial, False),      # x = 0
             _grid((0., 0., 0.), (1., 0., 0.), (0., 0., 1.), (3., 6., 9., 12., 15.), axial, True),  # y = 0
             _grid((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (2., 5., 8., 11., 14.), (3., 6., 9., 12., 15.), False)]  # z = 0
    nominal = np.concatenate([value[0] for value in scans])
    scan_triangles = np.concatenate([value[1]+25*i for i, value in enumerate(scans)])
    rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    translation = np.array((9., -5., 7.))
    observed = nominal@rotation.T+translation
    big_lateral = (1., 4.5, 8., 11.5, 15.)
    # The x=0 / y=0 faces span z in [-24, 24] so that +-10 mm shifts along their intersection
    # keep the scan patches inside them; z=0 (37) and z=10 (41) are both valid third-face
    # explanations of the z=0 scan sheet (genuine ambiguity), z=30 (43) is not (the passing
    # supports would slide off their faces).
    big_axial = (-24., -14., -4., 4., 14., 24.)
    cads = [_grid((0., 0., 0.), (0., 1., 0.), (0., 0., 1.), big_lateral, big_axial, True),
            _grid((0., 0., 0.), (1., 0., 0.), (0., 0., 1.), (2., 5.5, 9., 12.5, 16.), big_axial, False),
            _grid((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), big_lateral, (2., 5.5, 9., 12.5, 16.), True),
            _grid((0., 0., 10.), (1., 0., 0.), (0., 1., 0.), big_lateral, (2., 5.5, 9., 12.5, 16.), False),
            _grid((0., 0., 30.), (1., 0., 0.), (0., 1., 0.), big_lateral, (2., 5.5, 9., 12.5, 16.), True)]
    cad = np.concatenate([value[0][value[1]] for value in cads])
    counts = [len(value[1]) for value in cads]
    faces = np.concatenate([np.full(count, face_id, dtype=np.int64) for count, face_id in zip(counts, (17, 29, 37, 41, 43))])
    local = np.concatenate([np.arange(count, dtype=np.int64) for count in counts])
    return (observed, scan_triangles, cad, faces, local), nominal, (rotation, translation)


def _policy(init):
    return init._v1.WholePartRegistrationPolicyV1(coarse_points=50, refine_points=50,
        coarse_iterations=1, refine_iterations=1, max_correspondence_distance_mm=.5,
        huber_delta_mm=.05, max_inlier_plane_p95_mm=.1, min_scan_coverage=.95, min_cad_coverage=.25)


def _configuration(init, **overrides):
    options = dict(two_plane_enabled=True, two_plane_schedule_mode="v1", two_plane_context_mode="v4",
                   two_plane_context_budgets=None)
    options.update(overrides)
    return init._configuration(_policy(init), init._noisy.NoisyMeshPlanePolicyV5(), init._noisy.NoisyMeshPlaneBudgetsV5(),
        init._patch.CoplanarPatchPolicyV4(), init._patch.CoplanarPatchBudgetsV4(), init._search.PlaneSearchPolicyV6(),
        init._search.PlaneSearchBudgetsV6(max_bank_planes=3), init.PlaneInitializationBudgetsV12(), None, **options)


@pytest.fixture(scope="module")
def initialized(init):
    arrays, nominal, pose = _arrays()
    configuration = _configuration(init)
    result, context = init._initialize(*arrays, .025, configuration, UNIT)
    assert result["status"] == "COMPLETE", result["reasons"]
    return result, context, configuration, pose


def _passing_triplet(result):
    rows = [row for row in result["hypotheses"] if row.get("proposal_lane") == "TRIPLET"
            and row["prefilter_status"] == "PASSED" and len(row["assigned_support"]) == 3]
    assert rows, "fixture must yield a passing triplet"
    return rows[0]


def _faces(context):
    return {row["plane_id"]: row for row in context["cad_patches"]["patches"]}


def _support_on_face(row, face_id):
    return [support for support in row["assigned_support"] if support["cad_plane_id"] == face_id][0]


def test_main_lane_passes_and_completion_lane_publishes_bounded_ledger(initialized):
    result, context, configuration, _ = initialized
    lane = result["completion_lane"]
    assert lane["status"] == "COMPLETE" and lane["implementation"] == result["completion_implementation"]
    assert lane["policy"] == dict(enabled=True, max_parents=256, max_completions_per_parent=64,
        max_completions_total=4096, max_shift_mm=1000., min_axis_projection=1e-3, min_pair_sine=1e-3)
    stats = result["statistics"]
    assert stats["main_passing_start_group_count"] >= 1
    assert stats["exact_passing_start_group_count"] == stats["main_passing_start_group_count"]+stats["completion_passing_start_group_count"]
    assert lane["scheduled_completion_count"] == len(lane["hypotheses"])
    assert lane["parent_count"] == len(lane["parents"]) <= lane["policy"]["max_parents"]
    assert result["start_groups"]


def _parent_from(result, context, wrong_face, shift_along_axis):
    """Fabricate a two-of-three parent from the fixture's passing triplet: the z=0 (face 37) support is
    reassigned to `wrong_face` and the pose is shifted along the x=0 / y=0 intersection (the z axis) by
    `shift_along_axis`, exactly the V8 pattern (two supports pass, the third is entirely outside)."""
    row = copy.deepcopy(_passing_triplet(result))
    faces = _faces(context)
    assert {support["cad_plane_id"] for support in row["assigned_support"]} in ({17, 29, 37}, {17, 29, 41})
    if 41 in {support["cad_plane_id"] for support in row["assigned_support"]}:
        # The search already picked the z=10 explanation; treat it as the wrong-face parent directly.
        pass
    current = [support["cad_plane_id"] for support in row["assigned_support"] if support["cad_plane_id"] not in (17, 29)][0]
    third = _support_on_face(row, current)
    third.update(status="REJECTED", cad_plane_id=wrong_face, coverage=None, finite_query_disposition="NOT_REQUIRED_BOUND_EXCLUDED",
                 box_screening_status="COMPLETE", reasons=["ASSIGNED_WHOLE_FACE_BOX_COVERAGE_UPPER_BELOW_MINIMUM"])
    for match in row["correspondences"]:
        if match["cad_plane_id"] == current:
            match["cad_plane_id"] = wrong_face
    m1, m2 = np.asarray(faces[17]["normal"], float), np.asarray(faces[29]["normal"], float)
    axis = np.cross(m1, m2)
    axis /= np.linalg.norm(axis)
    row["translation_mm"] = (np.asarray(row["translation_mm"], float)+shift_along_axis*axis).tolist()
    row.update(prefilter_status="REJECTED", prefilter_reasons=["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
    return row, axis, current


@pytest.mark.parametrize("wrong_face,shift", [(41, 10.), (41, -10.)])
def test_completion_recovers_the_correct_third_face_from_a_parallel_wrong_assignment(init, initialized, wrong_face, shift):
    result, context, configuration, _ = initialized
    parent, axis, origin_face = _parent_from(result, context, wrong_face, shift)
    ledger = init._base()
    groups, lookup = [], {}
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"],
                                _faces(context), groups, lookup)
    lane = ledger["completion_lane"]
    assert lane["status"] == "COMPLETE" and lane["scheduled_parent_count"] == 1
    record = lane["parents"][0]
    assert record["failed_cad_plane_id"] == wrong_face and record["status"] == "SCHEDULED"
    assert abs(abs(float(np.dot(record["axis_direction"], axis)))-1.) < 1e-9
    completions = {row["completed_cad_plane_id"]: row for row in lane["hypotheses"]}
    assert {37, 41, 43} <= set(completions)
    correct = completions[37]
    assert correct["prefilter_status"] == "PASSED"
    assert all(support["status"] == "PASSED" for support in correct["assigned_support"])
    assert abs(correct["shift_residual_mm"]) < 1e-9 and correct["shift_denominator"] != 0.
    # Signed pose consistency from the fixture and the lane's own declared axis: the child pose is
    # parent + shift*u, the parent was displaced by +shift_along_axis*axis from the passing pose, so
    # the completion back onto the origin face needs shift = -shift_along_axis*(axis.u); a face at
    # a different z along the +-z axis needs the extra (z_face - z_origin)*(u.z), independent of the
    # candidate normal's sign (the sign cancels in the quotient).
    u = np.asarray(record["axis_direction"], float)
    uz = float(u[2])
    assert abs(abs(uz)-1.) < 1e-9
    z_of = {37: 0., 41: 10., 43: 30.}
    expected_origin = -shift*float(axis@u)
    assert abs(completions[origin_face]["shift_mm"]-expected_origin) < 1e-6
    for face_id in (37, 41, 43):
        assert abs(completions[face_id]["shift_mm"]-(expected_origin+(z_of[face_id]-z_of[origin_face])*uz)) < 1e-6, face_id
    # z=10 is a genuinely ambiguous alternative on this geometry: retained as its own group.
    assert completions[41]["prefilter_status"] == "PASSED"
    # z=30 would slide the two formerly passing supports off their faces: the old PASSED flags do not exempt them.
    far = completions[43]
    assert far["prefilter_status"] == "REJECTED"
    assert any(support["status"] == "REJECTED" and support["cad_plane_id"] in (17, 29) for support in far["assigned_support"])
    assert lane["passing_completion_count"] == 2 and lane["new_start_group_count"] == 2
    assert {group["lane"] for group in groups} == {"TRIPLET_COMPLETION"}
    assert correct["exact_start_group_index"] in (0, 1) and completions[41]["exact_start_group_index"] in (0, 1)
    assert correct["exact_start_group_index"] != completions[41]["exact_start_group_index"]
    assert lane["ledger_bytes"] > 0 and lane["usable"] is True
    assert all(row["completed_normal_sign"] in (-1, 1) and row["alignment_angle_radians"] <= row["alignment_allowance_radians"]
               for row in lane["hypotheses"])
    stats = lane["statistics"]
    assert stats["hypotheses_evaluated"] == lane["scheduled_completion_count"]
    assert stats["completed_finite_query_support_count"]+stats["bound_excluded_support_count"] == stats["assigned_support_count"]
    assert stats["indexed_face_count"] == len(lane["face_index_evidence"]) >= 2


def test_disabled_policy_and_omission_accounting(init, initialized):
    result, context, base, _ = initialized
    parent, _, _ = _parent_from(result, context, 41, 10.)
    configuration = dict(base, completion_policy=init.CompletionLanePolicyV1(enabled=False))
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], _faces(context), [], {})
    assert ledger["completion_lane"]["status"] == "DISABLED_BY_EXPLICIT_POLICY"
    configuration = dict(base, completion_policy=init.CompletionLanePolicyV1(max_completions_per_parent=1))
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], _faces(context), [], {})
    lane = ledger["completion_lane"]
    assert lane["scheduled_completion_count"] == 1 and lane["omitted_completion_count"] == lane["candidate_count"]-1
    assert lane["parents"][0]["omitted_count"] == lane["omitted_completion_count"]
    configuration = dict(base, completion_policy=init.CompletionLanePolicyV1(max_parents=1))
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent, copy.deepcopy(parent)],
                                context["scan_supports"], _faces(context), [], {})
    assert ledger["completion_lane"]["parent_count"] == 2 and ledger["completion_lane"]["omitted_parent_count"] == 1


def test_parallel_passing_pair_is_unresolved_not_completed(init, initialized):
    result, context, configuration, _ = initialized
    parent, _, _ = _parent_from(result, context, 41, 10.)
    faces = copy.deepcopy(_faces(context))
    faces[29]["normal"] = list(faces[17]["normal"])
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], faces, [], {})
    lane = ledger["completion_lane"]
    assert lane["parallel_passing_pair_count"] == 1 and lane["parents"][0]["status"] == "PASSING_PAIR_PARALLEL_UNRESOLVED"
    assert lane["scheduled_completion_count"] == 0 and lane["status"] == "COMPLETE"


def test_axis_parallel_third_face_is_counted_unresolved(init, initialized):
    result, context, configuration, _ = initialized
    parent, axis, origin_face = _parent_from(result, context, 41, 10.)
    faces = copy.deepcopy(_faces(context))
    # Declare the passing pair's CAD normals so their intersection direction is orthogonal to the
    # candidate faces' normals (the rotated third scan normal): every candidate is then normal-
    # compatible but unresolvable along the axis. No query may run for such candidates.
    third = _support_on_face(parent, 41)
    rotation = np.asarray(parent["rotation"], float)
    n3w = third["normal_sign"]*(rotation@np.asarray(context["scan_supports"][third["scan_plane_id"]]["analysis"]["normal"], float))
    n3w /= np.linalg.norm(n3w)
    other = np.eye(3)[int(np.argmin(np.abs(n3w)))]
    first = np.cross(n3w, other)
    first /= np.linalg.norm(first)
    # Passing pair normals n3w and first: their intersection direction is orthogonal to n3w, so a
    # candidate face with normal n3w is compatible but its distance cannot change along the axis.
    faces[17]["normal"] = n3w.tolist()
    faces[29]["normal"] = first.tolist()
    for face_id in (37, 41):
        faces[face_id]["normal"] = n3w.tolist()
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], faces, [], {})
    lane = ledger["completion_lane"]
    assert lane["axis_parallel_unresolved_count"]+lane["axial_unconstrained_count"] >= 2
    assert lane["scheduled_completion_count"] == 0
    assert lane["parents"][0]["status"] == "ALL_COMPATIBLE_CANDIDATES_UNRESOLVED_OR_OUT_OF_RANGE"


@pytest.mark.parametrize("field,value", [("max_parents", 0), ("max_completions_total", 70000), ("enabled", 1),
                                         ("max_shift_mm", float("nan")), ("min_axis_projection", 0.9)])
def test_invalid_completion_policy_is_rejected(init, field, value):
    with pytest.raises(ValueError):
        init.CompletionLanePolicyV1(**{field: value})



def test_shift_formula_is_invariant_to_axis_and_normal_orientation(init):
    policy = init.CompletionLanePolicyV1()
    witness = np.array([1., 2., 3.]); normal = np.array([0., 0., 1.]); anchor = np.array([5., 5., 10.])
    axis = np.array([0., 0., 1.])
    base = init._completion_shift(witness, normal, normal, anchor, axis, policy)
    assert base["disposition"] == "RESOLVED" and abs(base["shift_mm"]-7.) < 1e-12 and abs(base["residual_mm"]) < 1e-12
    flipped_axis = init._completion_shift(witness, normal, normal, anchor, -axis, policy)
    assert abs(flipped_axis["shift_mm"]+7.) < 1e-12  # lambda flips, t + lambda*u is invariant
    flipped_normal = init._completion_shift(witness, normal, -normal, anchor, axis, policy)
    assert abs(flipped_normal["shift_mm"]-7.) < 1e-12  # sign cancels in the quotient
    null = init._completion_shift(witness, normal, np.array([1., 0., 0.]), anchor, axis, policy)
    assert null["disposition"] == "AXIAL_COMPLETION_UNRESOLVED_DENOMINATOR" and null["shift_mm"] is None
    unconstrained = init._completion_shift(np.array([5., 9., 3.]), normal, np.array([1., 0., 0.]), anchor, axis, policy)
    assert unconstrained["disposition"] == "AXIAL_FAMILY_UNCONSTRAINED_BY_CANDIDATE"
    far = init._completion_shift(witness, normal, normal, anchor+np.array([0., 0., 5000.]), axis, policy)
    assert far["disposition"] == "SHIFT_LIMIT_REJECTED"
    sign, angle = init._candidate_sign(np.array([0., 0., 1.]), np.array([0., 0., -1.]))
    assert sign == -1 and angle < 1e-12


@pytest.mark.parametrize("mutation", ["three_passed", "one_passed", "incomplete_support", "context_lane"])
def test_only_completed_two_of_three_triplet_parents_are_eligible(init, initialized, mutation):
    result, context, configuration, _ = initialized
    parent, _, _ = _parent_from(result, context, 41, 10.)
    if mutation == "three_passed":
        for support in parent["assigned_support"]: support["status"] = "PASSED"
    elif mutation == "one_passed":
        parent["assigned_support"][0]["status"] = "REJECTED"
    elif mutation == "incomplete_support":
        parent["assigned_support"][2]["box_screening_status"] = "INCOMPLETE"
    elif mutation == "context_lane":
        parent["proposal_lane"] = "TWO_PLANE_CONTEXT"
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], _faces(context), [], {})
    assert ledger["completion_lane"]["parent_count"] == 0 and ledger["completion_lane"]["status"] == "COMPLETE"


def test_selected_children_are_published_before_fallible_work_and_preflight_refuses_whole_lane(init, initialized):
    result, context, base, _ = initialized
    parent, _, _ = _parent_from(result, context, 41, 10.)
    tiny = init.PlaneInitializationBudgetsV12(**dict(init.asdict(base["completion_budgets"]), max_prefilter_query_rows=10))
    configuration = dict(base, completion_budgets=tiny)
    ledger = init._base()
    with pytest.raises(Exception, match="COMPLETION_PREFILTER_PREFLIGHT_BUDGET_EXCEEDED"):
        init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], _faces(context), [], {})
    lane = ledger["completion_lane"]
    assert lane["status"] == "INCOMPLETE" and lane["scheduled_completion_count"] == len(lane["hypotheses"]) > 0
    assert all(row["prefilter_status"] == "NOT_ATTEMPTED" and row["assigned_support"] == [] for row in lane["hypotheses"])
    assert lane["parents"][0]["completion_indices"] == [row["completion_index"] for row in lane["hypotheses"]]


def test_near_zero_wrong_candidates_can_omit_the_farther_correct_one_and_the_omission_is_published(init, initialized):
    result, context, base, _ = initialized
    parent, _, _ = _parent_from(result, context, 41, 10.)
    configuration = dict(base, completion_policy=init.CompletionLanePolicyV1(max_completions_per_parent=1))
    ledger = init._base()
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], _faces(context), [], {})
    lane = ledger["completion_lane"]
    scheduled = [row["completed_cad_plane_id"] for row in lane["hypotheses"]]
    assert scheduled == [41] and lane["omitted_completion_count"] >= 2 and lane["parents"][0]["omitted_count"] >= 2
    assert lane["ranking"].startswith("NONEXHAUSTIVE_LOCAL_SCHEDULE")


def test_natural_two_of_three_parents_are_evaluated_inside_the_prefilter(init):
    """Shrink the z=0 CAD face so the search's own third assignment cannot support the scan sheet;
    the lane must then find its parents naturally and evaluate completions through _prefilter."""
    arrays, nominal, pose = _arrays()
    observed, scan_triangles, cad, faces, local = arrays
    keep = faces != 37
    small = _grid((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 2.5, 4.), (2., 3.5, 5.), True)
    cad = np.concatenate((cad[keep], small[0][small[1]]))
    faces = np.concatenate((faces[keep], np.full(len(small[1]), 37, dtype=np.int64)))
    local = np.concatenate((local[keep], np.arange(len(small[1]), dtype=np.int64)))
    configuration = _configuration(init)
    result, context = init._initialize(observed, scan_triangles, cad, faces, local, .025, configuration, UNIT)
    assert result["status"] == "COMPLETE", result["reasons"]
    lane = result["completion_lane"]
    assert lane["status"] == "COMPLETE" and lane["parent_count"] >= 1
    assert lane["scheduled_completion_count"] == len(lane["hypotheses"]) >= 1
    evaluated = [row for row in lane["hypotheses"] if row["prefilter_status"] in ("PASSED", "REJECTED")]
    assert len(evaluated) == len(lane["hypotheses"])
    assert lane["statistics"]["hypotheses_evaluated"] == len(lane["hypotheses"])
    assert result["statistics"]["exact_passing_start_group_count"] == (
        result["statistics"]["main_passing_start_group_count"]+result["statistics"]["completion_passing_start_group_count"])


def _natural_arrays():
    """The natural-parent scene: the z=0 CAD face is shrunk so the search's own third assignment cannot
    support the scan sheet, which makes the lane find two-of-three parents on its own."""
    arrays, nominal, pose = _arrays()
    observed, scan_triangles, cad, faces, local = arrays
    keep = faces != 37
    small = _grid((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 2.5, 4.), (2., 3.5, 5.), True)
    cad = np.concatenate((cad[keep], small[0][small[1]]))
    faces = np.concatenate((faces[keep], np.full(len(small[1]), 37, dtype=np.int64)))
    local = np.concatenate((local[keep], np.arange(len(small[1]), dtype=np.int64)))
    return observed, scan_triangles, cad, faces, local


def test_completion_ledger_cap_is_enforced_after_the_last_mutation_on_every_terminal_path(init, initialized):
    import json
    result, context, base, _ = initialized
    lane = result["completion_lane"]
    encoded = len(json.dumps(dict(lane, ledger_bytes=None), separators=(",", ":"), allow_nan=False).encode())
    assert lane["ledger_bytes"] == encoded and "refinement_schedule" in lane   # post-schedule field included
    assert lane["ledger_bytes_scope"].startswith("ENCODED_LANE_JSON_BYTES_WITH_LEDGER_BYTES_NULL")
    parent, _, _ = _parent_from(result, context, 41, 10.)
    ledger = init._base()
    init._complete_two_of_three(context, ledger, base, [parent], context["scan_supports"], _faces(context), [], {})
    exact = ledger["completion_lane"]["ledger_bytes"]
    assert exact == len(json.dumps(dict(ledger["completion_lane"], ledger_bytes=None), separators=(",", ":")).encode())
    # The lane encodes its own budgets, so the exact boundary is the fixed point where the cap value
    # equals the encoded size under that same cap (same digit count); one byte less must refuse.
    def run(cap):
        budget = init.PlaneInitializationBudgetsV12(**dict(init.asdict(base["completion_budgets"]), max_report_bytes=cap))
        ledger = init._base()
        init._complete_two_of_three(context, ledger, dict(base, completion_budgets=budget), [parent],
                                    context["scan_supports"], _faces(context), [], {})
        return ledger["completion_lane"]
    fixed = run(exact)["ledger_bytes"]
    at_cap = run(fixed)
    assert at_cap["status"] == "COMPLETE" and at_cap["ledger_bytes"] == fixed == at_cap["budgets"]["max_report_bytes"]
    with pytest.raises(Exception, match="COMPLETION_LEDGER_REPORT_BUDGET_EXCEEDED"):
        run(fixed-1)
    budget = init.PlaneInitializationBudgetsV12(**dict(init.asdict(base["completion_budgets"]), max_report_bytes=fixed-1))
    ledger = init._base()
    with pytest.raises(Exception, match="COMPLETION_LEDGER_REPORT_BUDGET_EXCEEDED"):
        init._complete_two_of_three(context, ledger, dict(base, completion_budgets=budget), [parent],
                                    context["scan_supports"], _faces(context), [], {})
    assert ledger["completion_lane"]["ledger_bytes"] is None and ledger["completion_lane"]["status"] == "COMPLETE"
    # Zero selected children with a populated parent ledger (every candidate unresolved along the axis,
    # as in test_axis_parallel_third_face_is_counted_unresolved): the cap still applies on that terminal path.
    tiny = init.PlaneInitializationBudgetsV12(**dict(init.asdict(base["completion_budgets"]), max_report_bytes=64))
    faces = copy.deepcopy(_faces(context))
    third = _support_on_face(parent, 41)
    rotation = np.asarray(parent["rotation"], float)
    n3w = third["normal_sign"]*(rotation@np.asarray(context["scan_supports"][third["scan_plane_id"]]["analysis"]["normal"], float))
    n3w /= np.linalg.norm(n3w)
    first = np.cross(n3w, np.eye(3)[int(np.argmin(np.abs(n3w)))])
    first /= np.linalg.norm(first)
    faces[17]["normal"], faces[29]["normal"] = n3w.tolist(), first.tolist()
    for face_id in (37, 41, 43):
        faces[face_id]["normal"] = n3w.tolist()
    probe = init._base()
    init._complete_two_of_three(context, probe, base, [parent], context["scan_supports"], faces, [], {})
    assert probe["completion_lane"]["scheduled_completion_count"] == 0 and probe["completion_lane"]["parent_count"] == 1
    ledger = init._base()
    with pytest.raises(Exception, match="COMPLETION_LEDGER_REPORT_BUDGET_EXCEEDED"):
        init._complete_two_of_three(context, ledger, dict(base, completion_budgets=tiny), [parent], context["scan_supports"], faces, [], {})
    assert ledger["completion_lane"]["scheduled_completion_count"] == 0 and ledger["completion_lane"]["hypotheses"] == []
    disabled = init._base()
    with pytest.raises(Exception, match="COMPLETION_LEDGER_REPORT_BUDGET_EXCEEDED"):
        init._complete_two_of_three(context, disabled, dict(base, completion_budgets=tiny,
            completion_policy=init.CompletionLanePolicyV1(enabled=False)), [parent], context["scan_supports"], _faces(context), [], {})


def test_completion_only_origins_bind_through_the_live_v16_seam_and_stale_mappings_refuse(init, initialized):
    v16 = importlib.import_module("backend.cad_guided_mesh_offline_v16")
    result, context, configuration, _ = initialized
    parent, _, _ = _parent_from(result, context, 41, 10.)
    parent = dict(parent, hypothesis_index=0)
    ledger = init._base()
    groups, lookup = [], {}
    init._complete_two_of_three(context, ledger, configuration, [parent], context["scan_supports"], _faces(context), groups, lookup)
    lane = ledger["completion_lane"]
    child = [row for row in lane["hypotheses"] if row["prefilter_status"] == "PASSED"][0]
    gi, idx = child["exact_start_group_index"], child["completion_index"]
    group = groups[gi]
    assert group["hypothesis_indices"] == [] and group["lane"] == "TRIPLET_COMPLETION"
    initialization = dict(ledger, hypotheses=[parent], all_passing_start_groups=groups)
    candidate = dict(exact_group_index=gi, hypothesis_indices=[], completion_indices=list(group["completion_indices"]),
                     initialized_rotation=group["initialized_rotation"], initialized_translation_mm=group["initialized_translation_mm"])
    v16._validate_completion_origins(initialization, candidate, gi, candidate["completion_indices"])
    position = lane["parents"][0]["failed_support_position"]
    passing_slot = [slot for slot in range(3) if slot != position][0]
    passing_face = parent["assigned_support"][passing_slot]["cad_plane_id"]

    def refuse(mutate, reason):
        tampered = copy.deepcopy(initialization)
        mutate(tampered)
        with pytest.raises(ValueError, match=reason):
            v16._validate_completion_origins(tampered, candidate, gi, list(candidate["completion_indices"]))
    kid = lambda t: t["completion_lane"]["hypotheses"][idx]
    refuse(lambda t: kid(t).update(replaced_cad_plane_id=passing_face), "COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT")
    refuse(lambda t: kid(t).update(completed_cad_plane_id=999), "COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT")
    refuse(lambda t: kid(t).update(completed_normal_sign=-kid(t)["completed_normal_sign"]), "COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT")
    refuse(lambda t: t["hypotheses"][0]["normal_signs"].__setitem__(passing_slot, -t["hypotheses"][0]["normal_signs"][passing_slot]),
           "COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT")
    refuse(lambda t: t["hypotheses"][0]["correspondences"][passing_slot].update(cad_plane_id=999),
           "COMPLETION_PARENT_NOT_A_COMPLETED_TWO_OF_THREE_TRIPLET|COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT")
    refuse(lambda t: t["completion_lane"]["parents"][0].update(failed_support_position=passing_slot), "COMPLETION_PARENT_LEDGER_NOT_BOUND_TO_CHILD")
    refuse(lambda t: t["completion_lane"]["parents"][0]["completion_indices"].remove(idx), "COMPLETION_PARENT_LEDGER_NOT_BOUND_TO_CHILD")
    refuse(lambda t: t["completion_lane"]["parents"][0].update(axis_direction=[0., 0., 0.]), "COMPLETION_PARENT_LEDGER_NOT_BOUND_TO_CHILD")
    refuse(lambda t: t["hypotheses"][0]["assigned_support"][passing_slot].update(status="REJECTED"), "COMPLETION_PARENT_NOT_A_COMPLETED_TWO_OF_THREE_TRIPLET")
    refuse(lambda t: kid(t).update(shift_mm=kid(t)["shift_mm"]+1e-6), "COMPLETION_POSE_NOT_PARENT_PLUS_SHIFT")
    # Mixed ordinary/completion origins on one group are impossible by construction (a completion-only
    # group has no main origins); a forged main origin on this group is refused by the main-origin ledger.
    assert candidate["hypothesis_indices"] == []


def test_late_failure_after_a_passing_completion_retains_children_but_no_surviving_start(init, monkeypatch):
    observed, scan_triangles, cad, faces, local = _natural_arrays()
    configuration = _configuration(init)
    original = init._cap_completion_lane
    fired = []
    def late(lane, budget):
        if lane.get("passing_completion_count", 0) >= 1 and not fired:
            fired.append(True)
            raise RuntimeError("INJECTED_LATE_SERIALIZATION_FAILURE")
        return original(lane, budget)
    monkeypatch.setattr(init, "_cap_completion_lane", late)
    result, context = init._initialize(observed, scan_triangles, cad, faces, local, .025, configuration, UNIT)
    assert fired == [True] and context is None and result["status"] != "COMPLETE"
    lane = result["completion_lane"]
    assert lane["usable"] is False and lane["status"] == "INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE"
    assert lane["passing_children_invalidated"] is True and lane["ledger_bytes"] is None
    assert any(row["prefilter_status"] == "PASSED" for row in lane["hypotheses"])   # retained selected children
    assert lane["attempted_child_count"] == len(lane["hypotheses"])
    assert result["start_groups"] == [] and result["selected_policy_evaluation_complete"] is False
    assert all(group["initialized_rotation"] is None and group["initialized_translation_mm"] is None
               for group in result["all_passing_start_groups"])
    assert result["statistics"]["exact_passing_start_group_count_scope"] == "ATTEMPTED_PREFILTER_BEFORE_FATAL_NOT_SURVIVING"
