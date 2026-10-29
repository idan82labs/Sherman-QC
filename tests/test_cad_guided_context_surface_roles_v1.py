"""Prospective source-only V1 role controls; NOT an extraction/engine trial.

Invented raw arrays and explicitly DECLARED qualified patch analyses exercise
the real V7 codec, full support binding, cone containment and incidence selector.
No extractor/native/CAD/registration/measurement is invoked. These fixtures do
not prove a noisy mesh yields the declarations: the connected caller must prove
that independently. No Gold, physical inputs or truth-based production policy.
Root owns the first 60 wall /45 worker CPU /1 GiB /1 MiB source gate.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest

from backend import cad_guided_context_surface_roles_v1 as roles


RESOURCE_LIMITS = dict(wall_seconds=60, worker_cpu_seconds=45,
                       process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _fixture(*, closure=False):
    # Cap fan center4; only cap vertices0/1 share real raw-ID side triangles.
    # Vertex7 is unreferenced and must not disappear. No coordinate welding.
    p = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.],
                  [0., 0., 0.], [-1., -1., 2.], [1., -1., 2.], [8., 9., 10.]], dtype="<f8")
    t = np.array([[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0], [0, 5, 1], [1, 5, 6]], dtype="<i8")
    epsilon = _digest(np.full(len(p), .05, dtype="<f8"))
    def record(kind, identifier, rows, normal):
        rows = np.array(rows, dtype="<i8")
        raw = np.unique(t[rows].ravel())
        return {"core_id" if kind == "CORE" else "closure_id": identifier,
            "triangle_rows": rows, "raw_vertex_rows": raw,
            "triangle_rows_sha256": _digest(rows), "raw_vertex_rows_sha256": _digest(raw),
            "analysis": dict(status="PROPOSED", witness_status="FOUND", cone_status="AVAILABLE",
                same_physical_plane_verified=False, uncertainty_status="STIPULATED_UNCALIBRATED",
                numerical_bound_status="ENGINEERING_ENCLOSURE_NOT_DIRECTED_ROUNDING_CERTIFICATE",
                normal=list(normal), normal_angle_bound_radians=0., raw_support_count=len(raw))}
    native = dict(schema_version="noisy_mesh_planes_v7", status="COMPLETE", unit="mm",
        coordinate_frame="RAW_SCAN_UNREGISTERED", raw_coordinate_welding=False, residual_trimming=False,
        input_digests=dict(points_sha256=_digest(p), triangle_raw_rows_sha256=_digest(t),
                          position_error_bounds_mm_sha256=epsilon),
        statistics=dict(core_construction_complete=True, core_evaluation_complete=True,
            scheduled_closure_evaluation_complete=True, input_raw_rows=len(p), input_triangles=len(t),
            core_count=2, cores_evaluated=2, unique_closure_count=int(closure), qualified_core_count=2,
            qualified_closure_count=int(closure), closures_scheduled=int(closure), closures_evaluated=int(closure),
            unscheduled_qualified_core_count=2-int(closure)),
        cores=[record("CORE", 0, [0, 1, 2, 3], [0., 0., 1.]), record("CORE", 1, [4, 5], [0., -1., 0.])],
        closures=[record("CLOSURE", 0, [0, 1, 2, 3], [0., 0., 1.])] if closure else [],
        proposals=[dict(kind="CORE", record_id=0), dict(kind="CORE", record_id=1)]
            + ([dict(kind="CLOSURE", record_id=0)] if closure else []))
    return p, t, native, epsilon


def _prepare(fixture, *, encoded=None, budgets=None):
    p, t, native, epsilon = fixture
    return roles.prepare_context_surface_roles_v1(p, t, native,
        roles._codec.encode_metadata_v7(native) if encoded is None else encoded, epsilon,
        budgets=roles.ContextSurfaceRoleBudgetsV1() if budgets is None else budgets)


def _select(prepared, axis=(0., 0., 1.), **kwargs):
    return roles.select_context_measurement_rows_v1(prepared, np.array(axis, dtype="<f8"), **kwargs)


@pytest.mark.parametrize("closure", [False, True])
def test_full_incident_union_keeps_welded_seams_and_unreferenced_rows(closure):
    fixture = _fixture(closure=closure)
    prepared = _prepare(fixture)
    answer = _select(prepared)
    np.testing.assert_array_equal(answer["keep_raw_rows"], [0, 1, 5, 6, 7])
    np.testing.assert_array_equal(answer["excluded_raw_rows"], [2, 3, 4])
    assert prepared.points_sha256 == _digest(fixture[0])
    assert prepared.triangle_raw_rows_sha256 == _digest(fixture[1])
    meta = answer["metadata"]
    assert meta["cap_like_triangle_count"] == 4 and meta["unknown_triangle_count"] == 2
    assert meta["unreferenced_rows_kept"] == 1
    assert meta["completed_triangle_membership_visits"] == 6+4*closure
    assert meta["completed_incidence_corner_visits"] == 18
    assert meta["attempted_incidence_corner_visits"] == 18
    assert meta["whole_selection_preflight_complete"] is True
    assert meta["production_use_allowed"] is False and meta["observed_bend_count"] is None
    assert meta["conformance"] == "INDETERMINATE"
    assert meta["unscheduled_qualified_core_closures"] == 2-int(closure)
    assert meta["extraction_search_exhaustive"] is False
    json.dumps(meta, allow_nan=False)


def test_unqualified_incident_triangle_is_unknown_even_if_its_geometry_is_planar():
    fixture = _fixture()
    native = fixture[2]
    native["cores"][1]["analysis"]["status"] = "REJECTED"
    native["proposals"].pop()
    native["statistics"].update(qualified_core_count=1, unscheduled_qualified_core_count=1)
    answer = _select(_prepare(fixture))
    assert 0 in answer["keep_raw_rows"] and 1 in answer["keep_raw_rows"]
    assert answer["metadata"]["unknown_triangle_count"] == 2


@pytest.mark.parametrize("angle,alpha,cap", [(0., 0., True), (14., .5, True), (14., 2., False),
    (0., 15., False), (16., 0., False), (90., 0., False)])
def test_entire_cone_not_just_nominal_normal_controls_role(angle, alpha, cap):
    fixture = _fixture()
    a = fixture[2]["cores"][0]["analysis"]
    theta = math.radians(angle)
    a.update(normal=[math.sin(theta), 0., math.cos(theta)], normal_angle_bound_radians=math.radians(alpha))
    answer = _select(_prepare(fixture))
    assert (len(answer["excluded_raw_rows"]) == 3) is cap
    assert answer["metadata"]["proposal_dispositions"][0]["role"] == (
        "AXIAL_CAP_LIKE" if cap else "UNKNOWN_NONCAP_OR_UNRESOLVED")


@pytest.mark.parametrize("rotation,translation", [
    (np.eye(3), np.zeros(3)),
    (np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), np.array([13., -9., 7.])),
    (np.diag([-1., -1., 1.]), np.array([-1000., 5., 11.]))])
def test_independent_frame_action_and_unoriented_signs(rotation, translation):
    p, t, native, epsilon = _fixture()
    transformed = p @ rotation.T + translation
    native["input_digests"]["points_sha256"] = _digest(transformed)
    for row in native["cores"]:
        row["analysis"]["normal"] = (-np.array(row["analysis"]["normal"]) @ rotation.T).tolist()
    prepared = _prepare((transformed, t, native, epsilon))
    answer = _select(prepared, -rotation[:, 2])
    np.testing.assert_array_equal(answer["excluded_raw_rows"], [2, 3, 4])


def test_equal_xyz_distinct_raw_id_is_not_welded_into_cap():
    p, t, native, epsilon = _fixture()
    p[7] = p[4]
    native["input_digests"]["points_sha256"] = _digest(p)
    answer = _select(_prepare((p, t, native, epsilon)))
    assert 4 in answer["excluded_raw_rows"] and 7 in answer["keep_raw_rows"]


def test_complete_no_qualified_proposal_is_explicit_unknown_not_missing_metadata():
    fixture = _fixture()
    native = fixture[2]
    for row in native["cores"]:
        row["analysis"]["status"] = "REJECTED"
    native["proposals"] = []
    native["statistics"].update(qualified_core_count=0, unscheduled_qualified_core_count=0)
    answer = _select(_prepare(fixture))
    np.testing.assert_array_equal(answer["keep_raw_rows"], np.arange(8))
    assert not len(answer["excluded_raw_rows"]) and answer["metadata"]["proposal_count"] == 0


@pytest.mark.parametrize("field,value", [("status", "UNAVAILABLE"), ("unit", "inch"),
    ("coordinate_frame", "CAD"), ("raw_coordinate_welding", True), ("residual_trimming", True)])
def test_wrong_source_or_unavailable_metadata_never_defaults_to_empty_cap(field, value):
    fixture = _fixture()
    fixture[2][field] = value
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1):
        _prepare(fixture)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bad_reference", "truncated_cores", "incomplete_eval"])
def test_full_proposal_denominators_not_selected_bank(mutation):
    fixture = _fixture()
    native = fixture[2]
    if mutation == "missing":
        native["proposals"].pop()
    elif mutation == "duplicate":
        native["proposals"].append(deepcopy(native["proposals"][0]))
    elif mutation == "bad_reference":
        native["proposals"][0]["record_id"] = 100
    elif mutation == "truncated_cores":
        native["cores"].pop()
        native["proposals"].pop()
    else:
        native["statistics"]["scheduled_closure_evaluation_complete"] = False
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1):
        _prepare(fixture)


@pytest.mark.parametrize("mutation", ["points", "triangles", "epsilon", "normal", "support", "descriptor"])
def test_stale_bound_inputs_analysis_and_support(mutation):
    fixture = _fixture()
    encoded = roles._codec.encode_metadata_v7(fixture[2])
    if mutation == "points":
        fixture[0][0, 0] += .01
    elif mutation == "triangles":
        fixture[1][0, 0] = 7
    elif mutation == "epsilon":
        fixture = (*fixture[:3], "0"*64)
    elif mutation == "normal":
        fixture[2]["cores"][0]["analysis"]["normal"] = [1., 0., 0.]
    elif mutation == "support":
        fixture[2]["cores"][0]["triangle_rows"][0] = 5
    else:
        fixture[2]["cores"][0]["triangle_rows_sha256"] = "0"*64
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1):
        _prepare(fixture, encoded=encoded)


def test_self_consistent_wrong_raw_union_is_rejected():
    fixture = _fixture()
    record = fixture[2]["cores"][0]
    record["raw_vertex_rows"] = np.array([0, 1, 2, 3, 7], dtype="<i8")
    record["raw_vertex_rows_sha256"] = _digest(record["raw_vertex_rows"])
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1, match="RAW_SUPPORT_UNION"):
        _prepare(fixture)


@pytest.mark.parametrize("axis", [np.zeros(3), np.array([0., 0., 2.]), np.array([0., 0., np.nan]),
    np.array([0., 0., np.inf]), np.array([False, False, True]), np.ones((1, 3)), [0., 0., 1.]])
def test_bad_axis_cannot_produce_row_filter(axis):
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1):
        roles.select_context_measurement_rows_v1(_prepare(_fixture()), axis)


@pytest.mark.parametrize("field,value", [("normal", [0., 0., 2.]), ("normal", [0., 0., float("nan")]),
    ("normal_angle_bound_radians", -.1), ("normal_angle_bound_radians", math.pi/2.),
    ("witness_status", "WITNESS_NOT_FOUND"), ("cone_status", "UNRESOLVED"),
    ("same_physical_plane_verified", True)])
def test_unqualified_or_malformed_cone_invalidates_filter(field, value):
    fixture = _fixture()
    fixture[2]["cores"][0]["analysis"][field] = value
    with pytest.raises((roles.ContextSurfaceRolesUnavailableV1, ValueError)):
        # NaN is refused at the actual upstream encoder boundary, before prepare.
        _prepare(fixture)


@pytest.mark.parametrize("field,limit", [("max_triangle_membership_visits", 5), ("max_incidence_corner_visits", 17)])
def test_whole_selection_preflight_refuses_before_any_union(monkeypatch, field, limit):
    prepared = _prepare(_fixture())
    def unexpected(*args, **kwargs):
        raise AssertionError("selection allocation must not occur after failed preflight")
    monkeypatch.setattr(roles.np, "zeros", unexpected)
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1, match="BEFORE_SELECTION") as caught:
        _select(prepared, **{field: limit})
    work = caught.value.statistics
    assert work["whole_selection_preflight_complete"] is False
    assert work["attempted_triangle_membership_visits"] == work["attempted_incidence_corner_visits"] == 0
    assert work["planned_triangle_membership_visits"] == 6 and work["planned_incidence_corner_visits"] == 18


def test_exact_selection_limits_and_prepared_arrays_detach_from_caller():
    fixture = _fixture()
    prepared = _prepare(fixture)
    fixture[0][:] = 99.
    fixture[1][:] = 0
    fixture[2]["cores"][0]["triangle_rows"][:] = 5
    fixture[2]["cores"][0]["analysis"]["normal"][:] = [1., 0., 0.]
    answer = _select(prepared, max_triangle_membership_visits=6, max_incidence_corner_visits=18)
    np.testing.assert_array_equal(answer["excluded_raw_rows"], [2, 3, 4])
    for array in (prepared._triangles, prepared._referenced, prepared._supports[0].triangle_rows,
                  answer["keep_raw_rows"], answer["excluded_raw_rows"]):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("field,value", [("max_raw_rows", 7), ("max_triangles", 5), ("max_proposals", 1),
    ("max_triangle_memberships", 5), ("max_raw_memberships", 8),
    ("max_incidence_corner_visits", 17), ("max_encoded_metadata_bytes", 0),
    ("max_decoded_metadata_bytes", 0), ("chunk_triangles", 0), ("max_raw_rows", True)])
def test_preparation_budgets_never_return_partial_context(field, value):
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1):
        _prepare(_fixture(), budgets=replace(roles.ContextSurfaceRoleBudgetsV1(), **{field: value}))


def test_post_snapshot_negative_indices_never_alias_last_raw_row(monkeypatch):
    fixture = _fixture()
    original = roles._owned
    def mutate_during_points(value, shape, dtype, name):
        answer = original(value, shape, dtype, name)
        if name == "RAW_POINTS":
            fixture[1][0, 0] = -1
        return answer
    monkeypatch.setattr(roles, "_owned", mutate_during_points)
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1, match="OWNED_RAW_TRIANGLE_IDS"):
        _prepare(fixture)


def test_late_selection_numeric_failure_has_attempted_not_completed_ledger(monkeypatch):
    prepared = _prepare(_fixture())
    original = roles.math.acos
    calls = 0
    def fail_second(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate helper seam")
        return original(value)
    monkeypatch.setattr(roles.math, "acos", fail_second)
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1, match="NUMERIC_FAILURE") as caught:
        _select(prepared)
    assert caught.value.statistics["attempted_triangle_membership_visits"] == 6
    assert caught.value.statistics["completed_triangle_membership_visits"] == 4
    assert caught.value.statistics["attempted_incidence_corner_visits"] == 0


def test_decoded_cap_is_checked_before_decoder_allocation(monkeypatch):
    fixture = _fixture()
    encoded = roles._codec.encode_metadata_v7(fixture[2])
    def unexpected(*args, **kwargs):
        raise AssertionError("decoder must not execute after expanded preflight overflow")
    monkeypatch.setattr(roles._codec, "decode_metadata_v7", unexpected)
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1, match="DECODED_METADATA_BUDGET"):
        _prepare(fixture, encoded=encoded,
                 budgets=replace(roles.ContextSurfaceRoleBudgetsV1(), max_decoded_metadata_bytes=1))


@pytest.mark.parametrize("missing", ["encoded", "native", "descriptors_only"])
def test_missing_arrays_or_evidence_cannot_reconstruct_hashes_as_support(missing):
    p, t, native, epsilon = _fixture()
    encoded = roles._codec.encode_metadata_v7(native)
    if missing == "encoded":
        encoded = None
    elif missing == "native":
        native = None
    else:
        native = roles._codec.decode_metadata_v7(encoded)
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1):
        roles.prepare_context_surface_roles_v1(p, t, native, encoded, epsilon)


@pytest.mark.parametrize("field,value", [("shape", [999]), ("dtype", "<u8"), ("sha256", "0"*64)])
def test_encoded_array_descriptor_cannot_be_replaced_independently(field, value):
    fixture = _fixture()
    encoded = roles._codec.encode_metadata_v7(fixture[2])
    decoded = roles._codec.decode_metadata_v7(encoded)
    decoded["cores"][0]["triangle_rows"][field] = value
    with pytest.raises(roles.ContextSurfaceRolesUnavailableV1, match="DESCRIPTOR"):
        _prepare(fixture, encoded=roles._codec.encode_metadata_v7(decoded))


@pytest.mark.parametrize("competing", ["noncap", "unresolved_cone_containment"])
def test_overlapping_qualified_competitor_vetoes_cap_not_overwritten_by_union(competing):
    fixture = _fixture(closure=True)
    analysis = fixture[2]["closures"][0]["analysis"]
    if competing == "noncap":
        analysis["normal"] = [1., 0., 0.]
    else:
        analysis["normal_angle_bound_radians"] = math.radians(15.)
    answer = _select(_prepare(fixture))
    np.testing.assert_array_equal(answer["keep_raw_rows"], np.arange(8))
    assert not len(answer["excluded_raw_rows"])
    meta = answer["metadata"]
    assert meta["cap_proposed_triangle_count"] == meta["conflicting_triangle_count"] == 4
    assert meta["cap_like_triangle_count"] == 0 and meta["unknown_triangle_count"] == 6
    assert meta["proposal_dispositions"][0]["role"] == "AXIAL_CAP_LIKE"
    assert meta["proposal_dispositions"][2]["role"] == "UNKNOWN_NONCAP_OR_UNRESOLVED"
    assert meta["role_policy"] == "CAP_AND_NO_COMPETING_NONCAP_OR_UNRESOLVED_ALL_INCIDENT_TRIANGLES"
