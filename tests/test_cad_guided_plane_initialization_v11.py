"""Prospective V11 pure-array integration controls; no native/physical inputs.

Two required new positives use exactly50 raw rows on two independent rectangles,
the stipulated uniform .05mm bound, a fixed proper permutation pose, and either
zero noise or moment-balanced normal noise <=.002mm. Both faces span>=16mm;
this is NOT the original thin/native fixture or a .05/free-circle qualification.
The two-plane bank is prospectively limited to2 to isolate this lane; extraction,
pair scheduling, axial proposals, full-face boxes and finite distances are real.
Required true start action error<=1e-7mm; every scheduled support keeps25 rows.
No two-plane global-rank/unique-registration or physical recovery claim follows.

The two frozen V10 noisy-registration positive bodies are reused unchanged with
ONLY their registration dispatch redirected to explicitly disabled V11. Other
private controls clearly inject schedules/hypotheses/query outputs; these are
contract oracles, not pose recovery. Root owns first60wall/45CPU/1GiB/1MiB gate.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from backend import cad_guided_plane_initialization_v11 as v11
from backend import cad_guided_plane_initialization_v10 as v10


LEGACY = Path(__file__).with_name("test_cad_guided_plane_initialization_v10.py")
LEGACY_SHA = "4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0"
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
                       process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
SPECS = (((-12., -7., 0.), (24., 0., 0.), (0., 20., 0.)),
         ((19., -7., -6.), (0., 20., 0.), (0., 0., 16.)))
ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
TRANSLATION = np.array([3., -7., 5.])


@pytest.fixture(scope="module")
def legacy():
    blob = LEGACY.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == LEGACY_SHA
    module = ModuleType("_v11_reviewed_v10_array_fixtures")
    module.__file__ = str(LEGACY)
    exec(compile(blob, str(LEGACY), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def owned_seed(legacy):
    # Actual frozen extraction once; later altered hypotheses are private mocks.
    return legacy._private_context()


def _configuration(*, enabled=True, budgets=None, policy=None, search_budgets=None):
    policy = policy or v11._v1.WholePartRegistrationPolicyV1(coarse_points=75, refine_points=75,
        coarse_iterations=2, refine_iterations=2, huber_delta_mm=.01,
        max_inlier_plane_p95_mm=.05, max_correspondence_distance_mm=.5,
        min_scan_coverage=.95, min_cad_coverage=.30)
    return v11._configuration(policy, v11._noisy.NoisyMeshPlanePolicyV5(),
        v11._noisy.NoisyMeshPlaneBudgetsV5(), v11._patch.CoplanarPatchPolicyV4(),
        v11._patch.CoplanarPatchBudgetsV4(), v11._search.PlaneSearchPolicyV6(),
        search_budgets or v11._search.PlaneSearchBudgetsV6(),
        budgets or v11.PlaneInitializationBudgetsV11(), None, enabled)


def _private_case(seed, **kwargs):
    inputs, context, previous, _, ids = deepcopy(seed)
    result = v11._base()
    for key in ("input_digests", "input_counts", "patch_evidence", "hypothesis_evidence"):
        result[key] = previous[key]
    return inputs, context, result, _configuration(**kwargs), ids


def _hypothesis(index, ids, *, lane="TWO_PLANE_CONTEXT", unsigned=0, shift=0., status="PROPOSED"):
    count = 2 if lane == "TWO_PLANE_CONTEXT" else 3
    return dict(hypothesis_index=index, unsigned_assignment_id=unsigned, proposal_lane=lane,
        status=status, reasons=[], rotation=np.eye(3).tolist() if status == "PROPOSED" else None,
        translation_mm=[shift, 0., 0.] if status == "PROPOSED" else None,
        correspondences=[dict(scan_plane_id=a, cad_plane_id=b, normal_sign=1)
            for a, b in zip(ids[:count], (11, 22, 33)[:count])],
        test_scope="PRIVATE_STIPULATED_HYPOTHESIS_NOT_POSE_RECOVERY")


def _run_private(monkeypatch, case):
    inputs, context, template, config, _ = case
    def prepared(*args):
        args[-2].update(deepcopy(template))
        return context
    monkeypatch.setattr(v11, "_prepare", prepared)
    return v11._initialize(*inputs, .005, config, UNIT)[0]


def _pair_schedule_stub(monkeypatch, ids, *, calls=None, matches=2):
    rows = [dict(scan_plane_ids=list(ids[:2]), cad_plane_ids=list(target))
            for target in ((11, 22), (22, 11))[:matches]]
    def scheduled(*args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return dict(status="COMPLETE", selected_policy_evaluation_complete=True,
                    scheduled_matches=deepcopy(rows), test_scope="EXPLICIT_SCHEDULER_STUB")
    monkeypatch.setattr(v11._pair_schedule, "schedule_two_plane_pairs_v1", scheduled)


def _two_rectangles(legacy, noisy):
    patches = [legacy._rectangle(spec, 5) for spec in SPECS]
    nominal = np.vstack([row[0] for row in patches])
    raw = np.vstack([row[1]+25*i for i, row in enumerate(patches)])
    cad_patches = [legacy._rectangle(spec, 2) for spec in SPECS]
    cad = np.vstack([points[triangles] for points, triangles in cad_patches])
    faces = np.repeat(np.array([11, 22], dtype=np.int64), 2)
    local = np.tile(np.arange(2, dtype=np.int64), 2)
    perturbed = nominal.copy()
    if noisy:
        e = np.array([1., -2., 2., -2., 1.])
        assert e.sum() == 0. and e@np.linspace(-1., 1., 5) == 0.
        displacement = .0005*np.outer(e, e).ravel()
        for i, specification in enumerate(SPECS):
            normal = np.cross(specification[1], specification[2])
            normal /= np.linalg.norm(normal)
            perturbed[i*25:(i+1)*25] += displacement[:, None]*normal
    observed = perturbed@ROTATION.T+TRANSLATION
    clean_observed = nominal@ROTATION.T+TRANSLATION
    assert len(observed) == 50 and len(raw) == 64 and len(cad) == 4
    assert np.max(np.linalg.norm(observed-clean_observed, axis=1)) <= .002000000001 < .05
    return (observed, raw, cad, faces, local), nominal, clean_observed


@pytest.mark.parametrize("noisy", (False, True))
def test_actual_fifty_row_two_plane_point_zero_five_lane_required_positive(legacy, noisy):
    inputs, nominal, clean_observed = _two_rectangles(legacy, noisy)
    answer = v11.prefilter_plane_initializations_v11(*inputs,
        position_error_bounds_mm=.05, registration_policy=legacy._policy(50), unit_assumption=UNIT,
        search_budgets=v11._search.PlaneSearchBudgetsV6(max_bank_planes=2))
    legacy._assert_complete_prefilter(answer)
    assert answer["start_groups"], answer["reasons"]
    assert answer["joint_proposal_accounting"]["triplet_records"] == 0
    assert answer["joint_proposal_accounting"]["ordered_pair_matches"] == 2
    assert answer["joint_proposal_accounting"]["pair_slot_reservation"] == 64
    assert answer["input_digests"]["position_error_bounds_mm_sha256"] == v11._digest(np.full(50, .05))
    assert len(answer["hypothesis_evidence"]["scan_bank"]) == 2
    assert answer["two_plane_context"]["statistics"]["slot_records_completed"] == 64
    assert len(answer["hypotheses"]) == 64
    assert all(row["proposal_lane"] == "TWO_PLANE_CONTEXT" for row in answer["hypotheses"])
    assert any(legacy._action_error(clean_observed, nominal, row["initialized_rotation"],
        row["initialized_translation_mm"]) <= 1e-7 for row in answer["start_groups"])
    for row in answer["hypotheses"]:
        if row["prefilter_scheduled"]:
            assert len(row["assigned_support"]) == len(row["correspondences"]) == 2
            legacy._assert_complete_box_supports(row["assigned_support"], 25)
    assert answer["two_plane_context"]["accepted_registration"] is False
    assert answer["global_pose_uniqueness"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("pose_index", range(2))
def test_disabled_lane_runs_the_unchanged_frozen_noisy_positive_body(legacy, monkeypatch, pose_index):
    def redirected(inputs, samples=None, **kwargs):
        return v11.register_whole_part_planes_v11(*inputs,
            position_error_bounds_mm=kwargs.pop("position_error_bounds_mm", legacy.EPSILON_MM),
            cad_initialization_points=inputs[0] if samples is None else samples,
            registration_policy=kwargs.pop("registration_policy", legacy._policy(len(inputs[0]))),
            unit_assumption=kwargs.pop("unit_assumption", legacy.UNIT), two_plane_enabled=False, **kwargs)
    monkeypatch.setattr(legacy, "_register", redirected)
    legacy.test_real_noisy_mesh_search_finite_support_and_registration_required_positive(pose_index)


def test_disabled_lane_preserves_rejected_unsigned_ids_without_new_reservation_gate(legacy, owned_seed, monkeypatch):
    case = _private_case(owned_seed, enabled=False,
        budgets=replace(v11.PlaneInitializationBudgetsV11(), max_prefilter_unsigned_assignments=1))
    _, context, result, configuration, ids = case
    context["hypotheses"] = [_hypothesis(0, ids, lane="TRIPLET", unsigned=0),
        _hypothesis(1, ids, lane="TRIPLET", unsigned=7, status="REJECTED")]
    monkeypatch.setattr(v11._pair_schedule, "schedule_two_plane_pairs_v1",
        lambda *a, **k: pytest.fail("Disabled pair lane reached its scheduler"))
    v11._two_plane_records(context, result, configuration)
    assert result["joint_proposal_accounting"]["triplet_unsigned_reservation"] == 8
    old_context, old_result = deepcopy(context), v10._base()
    old_configuration = legacy._configuration(budgets=replace(v10.PlaneInitializationBudgetsV10(),
        max_prefilter_unsigned_assignments=1))
    monkeypatch.setattr(v10, "_FaceQueries", legacy._AlwaysInsideMockQuery)
    monkeypatch.setattr(v11, "_FaceQueries", legacy._AlwaysInsideMockQuery)
    v10._prefilter(old_context, old_result, old_configuration)
    v11._prefilter(context, result, configuration)
    assert result["status"] == old_result["status"] == "COMPLETE"
    for key in old_result["statistics"]:
        assert result["statistics"][key] == old_result["statistics"][key], key
    assert result["start_groups"] == old_result["start_groups"]


@pytest.mark.parametrize("count,last_unsigned,expected", ((8128, 1015, True),
    (8129, 1016, False), (1, 1022, False), (1, 1021, True)))
def test_joint_record_and_unsigned_reservations_before_context_generation(
        owned_seed, monkeypatch, count, last_unsigned, expected):
    case = _private_case(owned_seed)
    _, context, result, config, ids = case
    context["hypotheses"] = [_hypothesis(i, ids, lane="TRIPLET", unsigned=min(i//8, last_unsigned),
        status="REJECTED") for i in range(count)]
    context["hypotheses"][-1]["unsigned_assignment_id"] = last_unsigned
    result["hypothesis_evidence"]["scan_bank"] = ids[:2]
    calls = []
    _pair_schedule_stub(monkeypatch, ids, calls=calls)
    actual = v11._context.propose_two_plane_context_v1
    helper_calls = []
    def helper(*args, **kwargs):
        helper_calls.append(True)
        return actual(*args, **kwargs)
    monkeypatch.setattr(v11._context, "propose_two_plane_context_v1", helper)
    if expected:
        v11._two_plane_records(context, result, config)
        assert len(context["hypotheses"]) == count+64
        assert len(helper_calls) == 1
        assert result["joint_proposal_accounting"]["pair_slot_reservation"] == 64
    else:
        with pytest.raises(ValueError, match="TWO_PLANE_JOINT_PREFLIGHT_EXCEEDED"):
            v11._two_plane_records(context, result, config)
        assert helper_calls == [] and len(context["hypotheses"]) == count
    assert len(calls) == 1
    assert calls[0][1] == dict(retained_triplet_hypothesis_count=count,
        retained_triplet_unsigned_count=last_unsigned+1,
        max_joint_hypothesis_records=8192, max_joint_unsigned_assignments=1024)


HELPER_FAULTS = ("input_digest", "source_frame", "target_origin", "scheduled_pairs",
    "incomplete", "drop_family", "drop_slot", "duplicate_signs", "wrong_correspondence",
    "boolean_family_index", "boolean_candidate_index", "missing_slot_origin", "wrong_pose",
    "accepted_registration", "completed_slot_count")


@pytest.mark.parametrize("fault", HELPER_FAULTS)
def test_helper_full_slot_pose_frame_and_assignment_binding_refuses_before_finite_work(
        owned_seed, monkeypatch, fault):
    inputs, context, result, config, ids = _private_case(owned_seed)
    context["hypotheses"] = []
    result["hypothesis_evidence"]["scan_bank"] = ids[:2]
    _pair_schedule_stub(monkeypatch, ids)
    original = v11._context.propose_two_plane_context_v1
    def corrupted(*args, **kwargs):
        helper = deepcopy(original(*args, **kwargs))
        assert helper["status"] == "COMPLETE", helper["reasons"]
        if fault == "input_digest": helper["input_digests"]["scan_points_sha256"] = "0"*64
        elif fault == "source_frame": helper["source_frame"] = "DIFFERENT_FRAME"
        elif fault == "target_origin": helper["target_origin_mm"][0] = 1.
        elif fault == "scheduled_pairs": helper["scheduled_pairs"][0]["cad_plane_ids"].reverse()
        elif fault == "incomplete": helper["selected_policy_evaluation_complete"] = False
        elif fault == "drop_family": helper["families"].pop()
        elif fault == "drop_slot": helper["families"][0]["slots"].pop()
        elif fault == "duplicate_signs": helper["families"][1]["normal_signs"] = helper["families"][0]["normal_signs"]
        elif fault == "wrong_correspondence": helper["families"][0]["correspondences"][0]["cad_plane_id"] = 33
        elif fault == "boolean_family_index": helper["families"][0]["signed_family_id"] = False
        elif fault == "boolean_candidate_index": helper["candidates"][0]["candidate_index"] = False
        elif fault == "missing_slot_origin": helper["candidates"][0]["contributing_slots"] = []
        elif fault == "wrong_pose": helper["candidates"][0]["translation_mm"][0] += 1.
        elif fault == "accepted_registration": helper["accepted_registration"] = True
        else: helper["statistics"]["slot_records_completed"] -= 1
        return helper
    monkeypatch.setattr(v11._context, "propose_two_plane_context_v1", corrupted)
    def prepared(*args):
        args[-2].update(deepcopy(result))
        return context
    monkeypatch.setattr(v11, "_prepare_triplets", prepared)
    monkeypatch.setattr(v11, "_FaceQueries", lambda *a: pytest.fail("Malformed helper reached finite work"))
    answer, returned = v11._initialize(*inputs, .005, config, UNIT)
    assert answer["status"] == "UNAVAILABLE" and returned is None and answer["start_groups"] == []
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["box_screening_complete"] is False and answer["numerical_consumer_complete"] is False
    assert answer["statistics"]["query_rows"] == 0
    assert all(row.get("usable_initialization") is False for row in answer["hypotheses"])


@pytest.mark.parametrize("fault", ("third_support", "duplicate_support", "dropped_raw_row", "incomplete_cad_face"))
def test_pair_requires_exactly_two_bound_complete_supports(owned_seed, monkeypatch, fault):
    case = _private_case(owned_seed)
    _, context, _, _, ids = case
    row = _hypothesis(0, ids)
    context["hypotheses"] = [row]
    if fault == "third_support": row["correspondences"].append(dict(scan_plane_id=ids[2], cad_plane_id=33, normal_sign=1))
    elif fault == "duplicate_support": row["correspondences"][1] = deepcopy(row["correspondences"][0])
    elif fault == "dropped_raw_row":
        patch = context["scan_supports"][ids[0]]
        patch["raw_vertex_rows"] = patch["raw_vertex_rows"][1:]
        patch["raw_vertex_rows_sha256"] = v11._digest(patch["raw_vertex_rows"])
        patch["analysis"]["raw_support_count"] -= 1
    else:
        patch = next(p for p in context["cad_patches"]["patches"] if p["plane_id"] == 11)
        for key in ("triangle_rows", "source_triangle_ids"):
            patch[key] = patch[key][1:]
            patch[key+"_sha256"] = v11._digest(patch[key])
    monkeypatch.setattr(v11, "_FaceQueries", lambda *a: pytest.fail("Forged support reached finite work"))
    answer = _run_private(monkeypatch, case)
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["statistics"]["query_rows"] == 0
    assert len(answer["hypotheses"]) == 1


def test_absent_slot_is_retained_without_geometric_rejection_or_refill(legacy, owned_seed, monkeypatch):
    case = _private_case(owned_seed)
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(0, ids, status="ABSENT_CONTEXT_SLOT"),
        _hypothesis(1, ids, status="REJECTED"), _hypothesis(2, ids)]
    monkeypatch.setattr(v11, "_FaceQueries", legacy._AlwaysInsideMockQuery)
    answer = _run_private(monkeypatch, case)
    legacy._assert_complete_prefilter(answer)
    assert answer["statistics"]["absent_context_slot_count"] == 1
    assert answer["statistics"]["search_rejected_hypothesis_count"] == 1
    assert answer["statistics"]["hypotheses_evaluated"] == 1
    assert answer["hypotheses"][0]["prefilter_status"] == "AXIAL_SLOT_ABSENT_NOT_GEOMETRIC_EXCLUSION"
    assert answer["hypotheses"][0]["assigned_support"] == []
    assert [g["hypothesis_indices"] for g in answer["start_groups"]] == [[2]]
    assert answer["statistics"]["planned_query_rows"] == 50


def test_mixed_lanes_share_exactly_one_forty_eight_group_schedule(legacy, owned_seed, monkeypatch):
    case = _private_case(owned_seed)
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(i, ids, lane="TRIPLET" if i%2 == 0 else "TWO_PLANE_CONTEXT",
        unsigned=i, shift=i*1e-9) for i in range(64)]
    monkeypatch.setattr(v11, "_FaceQueries", legacy._AlwaysInsideMockQuery)
    answer = _run_private(monkeypatch, case)
    legacy._assert_complete_prefilter(answer)
    assert len(answer["all_passing_start_groups"]) == 64 and len(answer["start_groups"]) == 48
    assert [g["exact_group_index"] for g in answer["start_groups"]] == [i*63//47 for i in range(48)]
    assert answer["statistics"]["planned_query_rows"] == 32*(75+50)
    assert answer["statistics"]["assigned_support_count"] == 32*(3+2)
    assert all(g["hypothesis_indices"] == [i] for i, g in enumerate(answer["all_passing_start_groups"]))


def test_late_second_pair_failure_invalidates_every_start_and_keeps_both_supports(legacy, owned_seed, monkeypatch):
    case = _private_case(owned_seed)
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(i, ids, unsigned=i) for i in range(2)]
    class Late(legacy._AlwaysInsideMockQuery):
        def __call__(self, face, points):
            if self.result["statistics"]["query_calls"] == 2:
                raise RuntimeError("INVENTED_LATE_PAIR_QUERY_FAILURE")
            return super().__call__(face, points)
    monkeypatch.setattr(v11, "_FaceQueries", Late)
    answer = _run_private(monkeypatch, case)
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["box_screening_complete"] is answer["numerical_consumer_complete"] is False
    assert len(answer["hypotheses"]) == 2 and all(len(r["assigned_support"]) == 2 for r in answer["hypotheses"])
    assert answer["statistics"]["planned_query_rows"] == 100
    assert answer["statistics"]["hypotheses_evaluated"] == 1
    assert answer["all_passing_start_groups"][0]["initialized_rotation"] is None
    assert all(row["usable_initialization"] is False for row in answer["hypotheses"])


def test_pair_real_box_and_query_receive_the_same_complete_chunk_buffer(legacy, owned_seed, monkeypatch):
    case = _private_case(owned_seed, budgets=replace(v11.PlaneInitializationBudgetsV11(), chunk_rows=7))
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(0, ids)]
    actual = v11._FaceQueries
    screened, queried = {}, {}
    class Observed(actual):
        def screen(self, face, points, gate):
            screened.setdefault(face, []).append((points, points.copy()))
            return super().screen(face, points, gate)
        def __call__(self, face, points):
            offset = len(queried.setdefault(face, []))
            original, snapshot = screened[face][offset]
            assert np.shares_memory(original, points)
            assert original.__array_interface__["data"] == points.__array_interface__["data"]
            np.testing.assert_array_equal(points, snapshot)
            queried[face].append(points.copy())
            return super().__call__(face, points)
    monkeypatch.setattr(v11, "_FaceQueries", Observed)
    answer = _run_private(monkeypatch, case)
    legacy._assert_complete_prefilter(answer)
    assert answer["start_groups"] and set(screened) == set(queried) == {11, 22}
    for face, scan_id in zip((11, 22), ids[:2]):
        assert [len(row) for row in queried[face]] == [7, 7, 7, 4]
        np.testing.assert_array_equal(np.vstack(queried[face]),
            context["scan"][context["scan_supports"][scan_id]["raw_vertex_rows"]])
    assert answer["statistics"]["planned_query_rows"] == answer["statistics"]["completed_query_rows"] == 50
    assert answer["statistics"]["completed_box_screen_calls"] == answer["statistics"]["completed_query_calls"] == 8


def test_pair_actual_finite_hole_is_inconclusive_for_boxes_not_admitted_support(legacy, monkeypatch):
    points, raw, cad, faces, local = legacy._mesh()
    omitted = {2*(4*y+x)+side for y in (1, 2) for x in (1, 2) for side in (0, 1)}
    keep = np.array([i not in omitted for i in range(len(cad))])
    seed = legacy._private_context((points, raw, cad[keep], faces[keep], local[keep]))
    case = _private_case(seed, policy=replace(legacy._policy(), min_scan_coverage=1.))
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(0, ids)]
    answer = _run_private(monkeypatch, case)
    legacy._assert_complete_prefilter(answer)
    row, = answer["hypotheses"]
    assert row["prefilter_status"] == "REJECTED" and answer["start_groups"] == []
    assert len(row["assigned_support"]) == 2
    assert [s["query_rows"] for s in row["assigned_support"]] == [25, 25]
    assert [s["finite_unambiguous_inlier_count"] for s in row["assigned_support"]] == [24, 25]
    assert row["assigned_support"][0]["coverage"] < 1.
    assert all(s["box_exclusion_decision"] == "NEEDS_SURFACE_QUERY"
        and s["finite_query_disposition"] == "COMPLETE" for s in row["assigned_support"])
    assert answer["face_index_evidence"][0]["triangle_count"] == 24
    assert answer["statistics"]["bound_excluded_support_count"] == 0


@pytest.mark.parametrize("cap", (49, 50))
def test_pair_exact_full_row_preflight_cap_no_partial_subset(legacy, owned_seed, monkeypatch, cap):
    case = _private_case(owned_seed,
        budgets=replace(v11.PlaneInitializationBudgetsV11(), max_prefilter_query_rows=cap))
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(0, ids)]
    # The admitted branch deliberately mocks finite answers: this isolates the
    # exact row cap, not geometric/numerical work cost or support correctness.
    monkeypatch.setattr(v11, "_FaceQueries", legacy._AlwaysInsideMockQuery if cap == 50
        else lambda *a: pytest.fail("A49-row cap dispatched a50-row pair"))
    answer = _run_private(monkeypatch, case)
    assert answer["statistics"]["planned_query_rows"] == 50
    assert len(answer["hypotheses"][0]["assigned_support"]) == 2
    if cap == 50:
        legacy._assert_complete_prefilter(answer)
        assert answer["start_groups"] and answer["statistics"]["completed_query_rows"] == 50
    else:
        assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
        assert answer["statistics"]["box_screen_rows"] == answer["statistics"]["query_rows"] == 0
        assert answer["box_screening_complete"] is answer["numerical_consumer_complete"] is False


def test_pair_box_helper_failure_preserves_attempt_and_charge_without_completion(owned_seed, monkeypatch):
    case = _private_case(owned_seed)
    _, context, _, _, ids = case
    context["hypotheses"] = [_hypothesis(0, ids)]
    monkeypatch.setattr(v11._box, "screen_face_box_rows", lambda *a, **k:
        dict(status="UNAVAILABLE", reason="INVENTED_BOX_ARITHMETIC_FAILURE"))
    answer = _run_private(monkeypatch, case)
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    stats = answer["statistics"]
    assert stats["box_screen_rows"] == stats["work_units"] == 25
    assert stats["box_screen_calls"] == 1 and stats["completed_box_screen_calls"] == 0
    assert stats["completed_box_screen_rows"] == stats["query_rows"] == 0
    supports = answer["hypotheses"][0]["assigned_support"]
    assert len(supports) == 2 and supports[0]["attempted_box_screen_rows"] == 25
    assert supports[0]["completed_box_screen_rows"] == supports[1]["attempted_box_screen_rows"] == 0
    assert answer["box_screening_complete"] is answer["numerical_consumer_complete"] is False


@pytest.mark.parametrize("enabled", (None, 0, 1, "true", np.bool_(True)))
def test_two_plane_flag_is_exact_boolean_before_extraction(legacy, monkeypatch, enabled):
    monkeypatch.setattr(v11, "_prepare", lambda *a: pytest.fail("Invalid flag reached extraction"))
    answer = v11.prefilter_plane_initializations_v11(*legacy._mesh(), position_error_bounds_mm=.05,
        registration_policy=legacy._policy(), unit_assumption=UNIT, two_plane_enabled=enabled)
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["reasons"] == ["INVALID_TWO_PLANE_LANE_POLICY"]
