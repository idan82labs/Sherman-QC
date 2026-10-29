"""Tiny actual composite admission plus explicitly artificial binding faults.

The ordinary plane/corner/skinny fixtures run actual proposal producers and the
actual codec, role admission and selector. Deliberately re-sealed fault fixtures
are declarations for validation/selector tests, not scientific extraction
evidence. Digests prove a structural binding here, never authenticity or fit
accuracy. No native, CAD, physical file, registration or measurement is used.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules():
    def owned(key): return key == "backend" or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved: del sys.modules[key]
    package = types.ModuleType("backend")
    package.__path__ = [str(ROOT/"backend")]
    sys.modules["backend"] = package
    try:
        yield types.SimpleNamespace(
            helper=importlib.import_module("backend.cad_guided_composite_surface_roles_v1"),
            producer=importlib.import_module("backend.cad_guided_scan_plane_preparation_v1"))
    finally:
        for key in tuple(sys.modules):
            if owned(key): del sys.modules[key]
        sys.modules.update(saved)


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _arrays(kind):
    height = .25 if kind == "skinny" else 16.
    p = np.array([[0., 0., 0.], [16., 0., 0.], [16., height, 0.], [0., height, 0.]])
    t = np.array([[0, 1, 2], [0, 2, 3]], dtype="<i8")
    if kind in ("corner", "unreferenced"):
        p = np.vstack((p, [0., 0., 16.], [16., 0., 16.]))
        t = np.vstack((t, [0, 5, 1], [0, 4, 5])).astype("<i8")
    if kind == "unreferenced":
        p = np.vstack((p, p[2]))  # Same XYZ as a cap-only row, a distinct raw ID.
    return p.astype("<f8"), t


def _fixture(modules, kind="square", posed=False):
    p, t = _arrays(kind)
    axis = np.array([0., 0., 1.])
    if posed:
        rotation = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
        assert np.array_equal(rotation.T@rotation, np.eye(3)) and np.linalg.det(rotation) == 1.
        p = p@rotation.T+np.array([32., -16., 8.])
        axis = rotation@axis
    native = modules.producer.prepare_composite_scan_planes_v1(p, t, position_error_bounds_mm=.05)
    assert native["status"] == "COMPLETE", native
    envelope = modules.helper._codec.encode_metadata_v7(native)
    epsilon = _sha(np.full(len(p), .05, dtype="<f8"))
    return types.SimpleNamespace(p=p, t=t, native=native, envelope=envelope, epsilon=epsilon, axis=axis)


def _prepare(modules, fixture, **kwargs):
    return modules.helper.prepare_composite_surface_roles_v1(fixture.p, fixture.t,
        fixture.native, fixture.envelope, fixture.epsilon, **kwargs)


def _select(modules, prepared, axis, **kwargs):
    return modules.helper._roles.select_context_measurement_rows_v1(prepared, axis, **kwargs)


def _reseal_declaration(modules, fixture):
    """Artificially bind a changed declaration; NOT an extraction rerun/proof.

    Necessary to reach deep semantic checks instead of stopping at a stale
    full-tree hash. No geometric fit or qualification is manufactured here.
    """
    codec = modules.helper._codec
    for value in fixture.native["lanes"].values():
        codec._finalize(value, value["budgets"], codec._Work())
    hashes = {}
    for name, value in fixture.native["lanes"].items():
        envelope = codec.encode_metadata_v7(value)
        hashes[name] = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=True,
            allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    fixture.native["lane_metadata_sha256"] = hashes
    codec._finalize(fixture.native, fixture.native["budgets"], codec._Work())
    fixture.envelope = codec.encode_metadata_v7(fixture.native)


def _assert_prepared_binding(modules, fixture, prepared):
    assert type(prepared) is modules.helper._roles.PreparedContextSurfaceRolesV1
    assert prepared.points_sha256 == _sha(fixture.p)
    assert prepared.triangle_raw_rows_sha256 == _sha(fixture.t)
    assert prepared.position_error_bounds_mm_sha256 == fixture.epsilon
    encoded = json.dumps(fixture.envelope, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    assert prepared.metadata_sha256 == hashlib.sha256(encoded).hexdigest()
    assert prepared.encoded_metadata_bytes == len(encoded)
    assert prepared.raw_row_count == len(fixture.p) and prepared.triangle_count == len(fixture.t)
    planes = fixture.native["planes"]
    assert [(row.kind, row.record_id) for row in prepared._supports] == [
        (row["lane"]+":"+row["kind"], row["plane_id"]) for row in planes]
    assert prepared.triangle_membership_count == fixture.native["statistics"]["qualified_triangle_memberships"]
    assert prepared.raw_membership_count == fixture.native["statistics"]["qualified_raw_memberships"]
    assert prepared.unscheduled_qualified_core_count == fixture.native["lanes"]["v7"]["statistics"]["unscheduled_qualified_core_count"]
    assert prepared.triangle_membership_count <= fixture.native["statistics"]["triangle_memberships"]
    for array in (prepared._triangles, prepared._referenced, *(row.triangle_rows for row in prepared._supports)):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.setflags(write=True)


@pytest.mark.parametrize("kind,posed", [("square", False), ("corner", False), ("corner", True),
                                       ("unreferenced", False), ("skinny", False)])
def test_actual_composite_codec_admission_and_full_incidence_selector(modules, kind, posed):
    fixture = _fixture(modules, kind, posed)
    prepared = _prepare(modules, fixture)
    _assert_prepared_binding(modules, fixture, prepared)
    answer = _select(modules, prepared, fixture.axis)
    expected = [] if kind == "skinny" else [0, 1, 2, 3] if kind == "square" else [2, 3]
    assert np.array_equal(answer["excluded_raw_rows"], expected)
    assert np.array_equal(answer["keep_raw_rows"], np.setdiff1d(np.arange(len(fixture.p)), expected))
    meta = answer["metadata"]
    assert meta["selected_proposal_evaluation_complete"] is True and meta["extraction_search_exhaustive"] is False
    assert meta["observed_bend_count"] is None and meta["production_use_allowed"] is False
    assert meta["conflicting_triangle_count"] == 0
    assert meta["completed_triangle_membership_visits"] == prepared.triangle_membership_count
    assert meta["completed_incidence_corner_visits"] == 3*len(fixture.t)
    assert len(meta["proposal_dispositions"]) == len(fixture.native["planes"])
    if kind == "square":
        assert [(s.kind, s.record_id) for s in prepared._supports] == [
            ("v7:CORE", 0), ("v7:CORE", 1), ("v7:CLOSURE", 2), ("seeded_v1:SEEDED_REGION", 3)]
    if kind in ("corner", "unreferenced"):
        assert fixture.native["lanes"]["v7"]["proposals"] == []
        assert all(row["analysis"]["status"] == "REJECTED" for row in fixture.native["lanes"]["v7"]["cores"])
        assert 0 in answer["keep_raw_rows"] and 1 in answer["keep_raw_rows"]
    if kind == "unreferenced":
        assert 2 in answer["excluded_raw_rows"] and 6 in answer["keep_raw_rows"]
        assert meta["unreferenced_rows_kept"] == 1
    if kind == "skinny":
        assert prepared._supports == () and meta["proposal_count"] == 0


@pytest.mark.parametrize("competing_lane", ["v7", "seeded_v1"])
def test_declared_cross_lane_competing_normal_veto_uses_combined_selector(modules, competing_lane):
    fixture = _fixture(modules)
    # An artificial qualified-normal declaration isolates the conflict policy.
    # It is NOT the actual fitted normal or a scientific qualification claim.
    table = "cores" if competing_lane == "v7" else "regions"
    fixture.native["lanes"][competing_lane][table][0]["analysis"]["normal"] = [1., 0., 0.]
    _reseal_declaration(modules, fixture)
    prepared = _prepare(modules, fixture)
    answer = _select(modules, prepared, fixture.axis)
    assert np.array_equal(answer["keep_raw_rows"], np.arange(4))
    assert len(answer["excluded_raw_rows"]) == 0
    assert answer["metadata"]["conflicting_triangle_count"] == 2
    assert answer["metadata"]["cap_proposed_triangle_count"] == 2
    assert answer["metadata"]["cap_like_triangle_count"] == 0
    kinds = {row["kind"] for row in answer["metadata"]["proposal_dispositions"]}
    assert any(k.startswith("v7:") for k in kinds) and any(k.startswith("seeded_v1:") for k in kinds)


@pytest.mark.parametrize("fault", ["points", "triangles", "epsilon", "qualified_normal", "qualified_array",
    "rejected_reason", "rejected_array", "encoded_descriptor", "scalar_alias"])
def test_stale_full_tree_binding_includes_rejected_support_records(modules, fault):
    fixture = _fixture(modules, "corner" if fault.startswith("rejected") else "square")
    if fault == "points": fixture.p[0, 0] += .01
    elif fault == "triangles": fixture.t[0, 0] = 1
    elif fault == "epsilon": fixture.epsilon = "f"*64
    elif fault == "qualified_normal": fixture.native["lanes"]["seeded_v1"]["regions"][0]["analysis"]["normal"] = [1., 0., 0.]
    elif fault == "qualified_array": fixture.native["lanes"]["seeded_v1"]["regions"][0]["triangle_rows"] = np.array([0], dtype="<i8")
    elif fault == "rejected_reason": fixture.native["lanes"]["v7"]["cores"][0]["analysis"]["reasons"].append("STALE_REJECTED_REASON")
    elif fault == "rejected_array": fixture.native["lanes"]["v7"]["cores"][0]["triangle_rows"] = np.array([0], dtype="<i8")
    elif fault == "encoded_descriptor":
        wire = modules.helper._codec.decode_metadata_v7(fixture.envelope)
        wire["lanes"]["seeded_v1"]["regions"][0]["triangle_rows"]["shape"][0] += 1
        fixture.envelope = modules.helper._codec.encode_metadata_v7(wire)
    elif fault == "scalar_alias":
        wire = modules.helper._codec.decode_metadata_v7(fixture.envelope)
        wire["accepted_registration"] = 0
        fixture.envelope = modules.helper._codec.encode_metadata_v7(wire)
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1): _prepare(modules, fixture)


@pytest.mark.parametrize("fault", ["missing_plane", "reordered_planes", "plane_bool", "ref_bool", "lane_order",
    "lane_unavailable", "missing_qualified_ref", "missing_seed", "seed_bool", "region_bool", "seed_duplicate",
    "bad_disposition", "first_seed", "unscheduled_bool", "incomplete_seed", "qualified_count_bool",
    "all_work_count_bool", "all_work_sum", "support_id_range", "wrong_raw_union", "rejected_wrong_raw_union",
    "rejected_v7_wrong_raw_union"])
def test_rebound_malformed_manifest_ids_denominators_and_supports_refuse(modules, fault):
    kind = ("corner" if fault == "rejected_v7_wrong_raw_union" else
            "skinny" if fault == "rejected_wrong_raw_union" else "square")
    fixture = _fixture(modules, kind)
    native = fixture.native
    first, second = native["lanes"]["v7"], native["lanes"]["seeded_v1"]
    if fault == "missing_plane": native["planes"].pop()
    elif fault == "reordered_planes": native["planes"].reverse()
    elif fault == "plane_bool": native["planes"][0]["plane_id"] = False
    elif fault == "ref_bool": first["proposals"][0]["record_id"] = False
    elif fault == "lane_order": native["lanes"] = dict(seeded_v1=second, v7=first)
    elif fault == "lane_unavailable": second["status"] = "UNAVAILABLE"
    elif fault == "missing_qualified_ref":
        second["proposals"] = []
        native["planes"].pop()
        native["statistics"].update(qualified_plane_count=3, qualified_triangle_memberships=6, qualified_raw_memberships=12)
    elif fault == "missing_seed": second["seeds"].pop()
    elif fault == "seed_bool": second["seeds"][0]["seed_triangle_row"] = bool(second["seeds"][0]["seed_triangle_row"])
    elif fault == "region_bool": second["regions"][0]["region_id"] = False
    elif fault == "seed_duplicate": second["scheduled_seed_triangle_rows"][1] = second["scheduled_seed_triangle_rows"][0]
    elif fault == "bad_disposition": second["seeds"][1]["disposition"] = "PROPOSED"
    elif fault == "first_seed": second["regions"][0]["first_seed_triangle_row"] = second["scheduled_seed_triangle_rows"][1]
    elif fault == "unscheduled_bool": second["unscheduled_seed_triangle_count"] = False
    elif fault == "incomplete_seed": second["statistics"]["selected_policy_evaluation_complete"] = False
    elif fault == "qualified_count_bool": second["statistics"]["qualified_region_count"] = True
    elif fault == "all_work_count_bool": second["statistics"]["fit_evaluations"] = True
    elif fault == "all_work_sum": native["statistics"]["triangle_memberships"] += 1
    elif fault == "support_id_range":
        record = second["regions"][0]
        record["triangle_rows"] = np.array([0, 2], dtype="<i8")
        record["triangle_rows_sha256"] = _sha(record["triangle_rows"])
        native["planes"][-1]["triangle_rows_sha256"] = record["triangle_rows_sha256"]
    elif fault in ("wrong_raw_union", "rejected_wrong_raw_union"):
        record = second["regions"][0]
        record["raw_vertex_rows"] = np.array([0, 1, 2], dtype="<i8")
        record["raw_vertex_rows_sha256"] = _sha(record["raw_vertex_rows"])
        record["analysis"]["raw_support_count"] = 3
        second["statistics"].update(raw_memberships=3, full_support_row_tests=3)
        native["statistics"]["raw_memberships"] -= 1
        native["statistics"]["full_support_row_tests"] -= 1
        if fault == "wrong_raw_union":
            native["planes"][-1]["raw_vertex_rows_sha256"] = record["raw_vertex_rows_sha256"]
            native["statistics"]["qualified_raw_memberships"] -= 1
    elif fault == "rejected_v7_wrong_raw_union":
        record = first["cores"][0]
        assert record["analysis"]["status"] == "REJECTED" and first["proposals"] == []
        expected = np.unique(fixture.t[record["triangle_rows"]])
        assert np.array_equal(record["raw_vertex_rows"], expected) and len(expected) > 1
        # Rebind the altered declaration and accounting: only the original
        # triangle-to-raw union proves this rejected record's lineage is wrong.
        record["raw_vertex_rows"] = np.array(expected[:-1], dtype="<i8")
        record["raw_vertex_rows_sha256"] = _sha(record["raw_vertex_rows"])
        record["analysis"]["raw_support_count"] = len(expected)-1
        first["statistics"]["raw_memberships"] -= 1
        first["statistics"]["full_support_row_tests"] -= 1
        native["statistics"]["raw_memberships"] -= 1
        native["statistics"]["full_support_row_tests"] -= 1
    _reseal_declaration(modules, fixture)
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1) as error:
        _prepare(modules, fixture)
    if fault == "rejected_v7_wrong_raw_union":
        assert "INCOMPLETE_COMPOSITE_RAW_UNION" in str(error.value)


def test_self_consistent_lane_hash_corruption_cannot_disable_comparison(modules):
    fixture = _fixture(modules)
    fixture.native["lane_metadata_sha256"]["seeded_v1"] = "f"*64
    modules.helper._codec._finalize(fixture.native, fixture.native["budgets"], modules.helper._codec._Work())
    fixture.envelope = modules.helper._codec.encode_metadata_v7(fixture.native)
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1, match="LANE_DIGEST"):
        _prepare(modules, fixture)


def _exact_budgets(modules, fixture):
    encoded = json.dumps(fixture.envelope, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    wire = modules.helper._codec.decode_metadata_v7(fixture.envelope)
    expanded = json.dumps(wire, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    return modules.helper._roles.ContextSurfaceRoleBudgetsV1(max_raw_rows=4, max_triangles=2,
        max_proposals=4, max_triangle_memberships=8, max_raw_memberships=16,
        max_encoded_metadata_bytes=len(encoded), max_decoded_metadata_bytes=len(expanded),
        max_incidence_corner_visits=6, chunk_triangles=1)


def test_exact_combined_membership_metadata_and_incidence_caps_allow_complete_selection(modules):
    fixture = _fixture(modules)
    cap = _exact_budgets(modules, fixture)
    prepared = _prepare(modules, fixture, budgets=cap)
    _assert_prepared_binding(modules, fixture, prepared)
    assert prepared.encoded_metadata_bytes == cap.max_encoded_metadata_bytes
    assert prepared.decoded_metadata_bytes == cap.max_decoded_metadata_bytes
    assert prepared.triangle_membership_count == 8 and prepared.raw_membership_count == 16
    answer = _select(modules, prepared, fixture.axis, max_triangle_membership_visits=8, max_incidence_corner_visits=6)
    assert np.array_equal(answer["excluded_raw_rows"], np.arange(4))


@pytest.mark.parametrize("field", ["max_raw_rows", "max_triangles", "max_proposals", "max_triangle_memberships",
    "max_raw_memberships", "max_encoded_metadata_bytes", "max_decoded_metadata_bytes", "max_incidence_corner_visits", "chunk_triangles"])
def test_one_under_preparation_budget_never_returns_a_context_prefix(modules, field):
    fixture = _fixture(modules)
    cap = _exact_budgets(modules, fixture)
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1):
        _prepare(modules, fixture, budgets=replace(cap, **{field: getattr(cap, field)-1}))


@pytest.mark.parametrize("field,value", [("max_triangle_membership_visits", 7), ("max_incidence_corner_visits", 5)])
def test_selection_preflight_is_combined_not_one_lane_at_a_time(modules, field, value):
    fixture = _fixture(modules)
    prepared = _prepare(modules, fixture)
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1) as error:
        _select(modules, prepared, fixture.axis, **{field: value})
    work = error.value.statistics
    assert work["planned_triangle_membership_visits"] == 8 and work["planned_incidence_corner_visits"] == 6
    assert work["whole_selection_preflight_complete"] is False
    assert work["completed_triangle_membership_visits"] == work["completed_incidence_corner_visits"] == 0


def test_prepared_context_detaches_caller_state_and_token_is_required(modules):
    fixture = _fixture(modules, "corner")
    before = deepcopy(fixture.envelope)
    p, t = fixture.p.copy(), fixture.t.copy()
    prepared = _prepare(modules, fixture)
    assert fixture.envelope == before and np.array_equal(fixture.p, p) and np.array_equal(fixture.t, t)
    assert modules.helper._codec.encode_metadata_v7(fixture.native) == before
    baseline = _select(modules, prepared, fixture.axis)
    fixture.p[:] = 999.
    fixture.t[:] = -1
    fixture.native["planes"].clear()
    fixture.envelope.clear()
    after = _select(modules, prepared, fixture.axis)
    assert np.array_equal(after["excluded_raw_rows"], baseline["excluded_raw_rows"])
    assert np.array_equal(after["keep_raw_rows"], baseline["keep_raw_rows"])
    with pytest.raises(FrozenInstanceError): prepared.raw_row_count = 1
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1, match="INVALID_PREPARED_ROLE_CONTEXT"):
        _select(modules, replace(prepared, _token=object()), fixture.axis)


@pytest.mark.parametrize("native,wire", [(True, 1), (False, 0), (0, False), (0., -0.),
                                        (np.float32(0.), 0.), (np.int64(0), 0)])
def test_full_metadata_comparison_refuses_scalar_aliases_and_unsupported_wrappers(modules, native, wire):
    with pytest.raises(modules.helper._roles.ContextSurfaceRolesUnavailableV1):
        modules.helper._same_metadata({"a": native}, {"a": wire})


def test_exact_float64_and_tuple_wire_normalization_are_permitted_not_arbitrary_coercion(modules):
    modules.helper._same_metadata({"a": (np.float64(.5), False, None)}, {"a": [.5, False, None]})
