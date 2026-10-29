"""Independent lossless V6 representation gate; invented arrays only.

All maximum-work controls seed a private ledger next to its fixed limit. They
do not allocate a maximum-size report, array or mesh. Geometry comparisons keep
the frozen V5 policy, fixtures and positive/refusal gates unchanged.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import struct

import numpy as np
import pytest

from backend import cad_guided_noisy_mesh_planes_v5 as v5
from backend import cad_guided_noisy_mesh_planes_v6 as v6


FORMAT = "noisy_mesh_metadata_schema_v6"
ENCODED_LIMIT, EXPANDED_LIMIT = 67_108_864, 268_435_456
VISIT_LIMIT, HASH_LIMIT = 64_000_000, 4_294_967_296
RESOURCE_FIELDS = ("metadata_bytes", "metadata_expanded_bytes", "metadata_node_visits", "metadata_array_bytes_hashed")
R_NUMERATOR = np.array([[1, 8, 4], [8, 1, -4], [-4, 4, -7]], dtype=np.int64)
R = R_NUMERATOR.astype(float)/9.
FIXED_FRAMES = ((np.eye(3), np.zeros(3), 1.), (R, np.array([3., -4., 5.]), 1.),
                (R, np.array([37., -21., 9.]), .25), (R.T, np.array([-9., 5., 11.]), 4.))


def _bytes(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _descriptor(array):
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def _logical(value):
    if type(value) is np.ndarray: return _descriptor(value)
    if type(value) is dict: return {key: _logical(item) for key, item in value.items()}
    if type(value) in (tuple, list): return [_logical(item) for item in value]
    return value


def _strict_equal(first, second):
    assert type(first) is type(second), (type(first), type(second))
    if type(first) is dict:
        assert list(first) == list(second)
        for key in first: _strict_equal(first[key], second[key])
    elif type(first) is list:
        assert len(first) == len(second)
        for a, b in zip(first, second): _strict_equal(a, b)
    elif type(first) is float:
        assert struct.pack(">d", first) == struct.pack(">d", second)
    else: assert first == second


def _tree_work(value):
    """Independent declared node/key visit and repeated-array-byte denominator."""
    if type(value) is np.ndarray:
        visits, hashes = _tree_work(_descriptor(value))
        return 1+visits, array_bytes(value)+hashes
    if type(value) is dict:
        child = [_tree_work(item) for item in value.values()]
        return 1+len(value)+sum(row[0] for row in child), sum(row[1] for row in child)
    if type(value) in (list, tuple):
        child = [_tree_work(item) for item in value]
        return 1+sum(row[0] for row in child), sum(row[1] for row in child)
    return 1, 0


def array_bytes(value):
    return int(value.size)*int(value.dtype.itemsize)


def _envelope(root, schemas=None):
    return dict(format=FORMAT, schemas=[] if schemas is None else schemas, root=root)


def _roundtrip(value):
    envelope = v6.encode_metadata_v6(value)
    restored = v6.decode_metadata_v6(envelope)
    expected = _logical(value)
    _strict_equal(restored, expected)
    assert _bytes(restored) == _bytes(expected)
    assert envelope == v6._metadata(value)
    return envelope, restored


@pytest.mark.parametrize("value", (None, False, True, 0, 1, -7, 0., -0., 1., .1, "D", "L", "\u03b1\n\"\\\U0001f642"))
def test_primitives_preserve_exact_types_bits_and_literal_tags(value):
    _roundtrip(value)


def test_ordered_schemas_lists_alias_mutations_and_tag_like_values_are_lossless():
    value = {"second": ["D", 0, ["L", []]], "first": {"b": False, "a": None},
             "reversed": {"a": None, "b": False}, "tuple": (1, 1., -0.)}
    envelope, restored = _roundtrip(value)
    assert ["b", "a"] in envelope["schemas"] and ["a", "b"] in envelope["schemas"]
    saved = deepcopy(restored)
    value["first"]["late"] = [1, 2]
    value["tuple"] = (False,)
    _strict_equal(restored, saved)
    new, decoded = _roundtrip(value)
    assert new != envelope and decoded["first"]["late"] == [1, 2]
    decoded["first"]["late"].append(3)
    assert value["first"]["late"] == [1, 2]


@pytest.mark.parametrize("dtype", ("<i8", ">i4", "<u2", "<f8", "?"))
def test_arrays_preserve_only_explicit_shape_dtype_and_exact_byte_digest(dtype):
    array = np.array([[0, 1], [1, 0]], dtype=dtype)
    envelope, decoded = _roundtrip(dict(array=array))
    assert decoded["array"] == _descriptor(array)
    assert type(decoded["array"]) is dict  # Not authenticated/reconstructed raw rows.
    saved = deepcopy(envelope)
    array[:] = 1
    assert envelope == saved
    assert v6.encode_metadata_v6(dict(array=array)) != envelope


def test_repeated_record_schema_reduces_bytes_without_omitting_any_record_or_value():
    value = [dict(full_support_max_abs_residual_mm=i/1000, normal_angle_bound_radians=.125,
                  source_core_ids=[i, i+1], witness_status="FOUND", candidate_not_physical=True)
             for i in range(100)]
    envelope, decoded = _roundtrip(value)
    assert len(decoded) == 100 and len(envelope["schemas"]) == 1
    assert len(_bytes(envelope)) < len(_bytes(value))


@pytest.mark.parametrize("mode", ("not_dict", "extra", "missing", "wrong_format", "key_subclass", "format_subclass",
    "schemas_tuple", "schema_tuple", "duplicate_key", "duplicate_schema", "unused_schema", "bool_id",
    "negative_id", "large_id", "float_id", "arity", "unknown_tag", "bare_list", "list_arity",
    "dict_arity", "list_values_tuple", "dict_values_tuple", "dict_values_dict", "bare_dict"))
def test_malformed_envelopes_never_return_a_partial_logical_tree(mode):
    envelope = _envelope(["D", 0, [1]], [["a"]])
    class Text(str): pass
    if mode == "not_dict": envelope = list(envelope.items())
    elif mode == "extra": envelope["extra"] = 0
    elif mode == "missing": del envelope["root"]
    elif mode == "wrong_format": envelope["format"] = "other"
    elif mode == "key_subclass": envelope = {Text(key): value for key, value in envelope.items()}
    elif mode == "format_subclass": envelope["format"] = Text(FORMAT)
    elif mode == "schemas_tuple": envelope["schemas"] = (("a",),)
    elif mode == "schema_tuple": envelope["schemas"] = [("a",)]
    elif mode == "duplicate_key": envelope.update(schemas=[["a", "a"]], root=["D", 0, [1, 2]])
    elif mode == "duplicate_schema": envelope["schemas"] = [["a"], ["a"]]
    elif mode == "unused_schema": envelope["schemas"].append(["unused"])
    elif mode in ("bool_id", "negative_id", "large_id", "float_id"):
        envelope["root"][1] = {"bool_id": False, "negative_id": -1, "large_id": 1, "float_id": 0.}[mode]
    elif mode == "arity": envelope["root"][2] = []
    elif mode == "unknown_tag": envelope["root"][0] = "X"
    elif mode == "bare_list": envelope["root"] = [0, 1]
    elif mode == "list_arity": envelope["root"] = ["L", [], None]
    elif mode == "dict_arity": envelope["root"] = ["D", 0]
    elif mode == "list_values_tuple": envelope["root"] = ["L", (1,)]
    elif mode == "dict_values_tuple": envelope["root"][2] = (1,)
    elif mode == "dict_values_dict": envelope["root"][2] = {"a": 1}
    else: envelope["root"] = {"a": 1}
    with pytest.raises(ValueError): v6.decode_metadata_v6(envelope)


def test_outer_arity_refusal_precedes_any_schema_or_root_traversal(monkeypatch):
    monkeypatch.setattr(v6, "_Work", lambda: pytest.fail("Overlarge outer envelope reached traversal"))
    envelope = _envelope(None)
    envelope["fourth"] = object()
    with pytest.raises(ValueError, match="INVALID_METADATA_ENVELOPE"):
        v6.decode_metadata_v6(envelope)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, np.float64(1.), b"bytes", 1j, {1, 2}))
def test_unsupported_or_nonfinite_primitives_refuse_both_directions(value):
    with pytest.raises(ValueError): v6.encode_metadata_v6(value)
    with pytest.raises(ValueError): v6.decode_metadata_v6(_envelope(value))


@pytest.mark.parametrize("kind", ("dict", "list", "tuple", "array", "object_array", "complex_array", "noncontiguous"))
def test_container_subclasses_and_unsupported_arrays_are_not_dispatch_plugins(kind):
    if kind == "dict":
        class Derived(dict): pass
        value = Derived(a=1)
    elif kind == "list":
        class Derived(list): pass
        value = Derived([1])
    elif kind == "tuple":
        class Derived(tuple): pass
        value = Derived((1,))
    elif kind == "array":
        class Derived(np.ndarray): pass
        value = np.arange(3).view(Derived)
    elif kind == "object_array": value = np.array([1], dtype=object)
    elif kind == "complex_array": value = np.array([1j])
    else: value = np.arange(8)[::2]
    with pytest.raises(ValueError): v6.encode_metadata_v6(value)


@pytest.mark.parametrize("maximum", (False, -1, 1., None, ENCODED_LIMIT+1))
def test_encoded_limit_cannot_be_widened_or_coerced(maximum):
    with pytest.raises(ValueError, match="INVALID_METADATA_BYTE_LIMIT"):
        v6.encode_metadata_v6(None, max_bytes=maximum)
    with pytest.raises(ValueError, match="INVALID_METADATA_BYTE_LIMIT"):
        v6.decode_metadata_v6(_envelope(None), max_bytes=maximum)


def test_exact_encoded_size_boundary_and_one_below_for_both_directions():
    value = {"\u03bb": [False, None, -0., 1, 1., "quoted\"\\\n"]}
    envelope, _ = _roundtrip(value)
    size = len(_bytes(envelope))
    assert v6.encode_metadata_v6(value, max_bytes=size) == envelope
    _strict_equal(v6.decode_metadata_v6(envelope, max_bytes=size), value)
    with pytest.raises(ValueError): v6.encode_metadata_v6(value, max_bytes=size-1)
    with pytest.raises(ValueError): v6.decode_metadata_v6(envelope, max_bytes=size-1)
    assert v6._json_size(envelope, size) == size


@pytest.mark.parametrize("kind", ("schema_count", "keys", "key_chars", "string_chars", "integer_bits", "depth"))
def test_exact_scalar_schema_and_depth_boundaries_then_one_over(kind):
    if kind == "schema_count":
        allowed = [{"k"+str(i): i} for i in range(256)]
        denied = allowed+[{"k256": 256}]
    elif kind == "keys":
        allowed = {str(i): i for i in range(128)}
        denied = {str(i): i for i in range(129)}
    elif kind == "key_chars": allowed, denied = {"k"*128: 0}, {"k"*129: 0}
    elif kind == "string_chars": allowed, denied = "\u03bb"*65536, "\u03bb"*65537
    elif kind == "integer_bits": allowed, denied = -(1 << 4095), -(1 << 4096)
    else:
        allowed = 0
        for _ in range(64): allowed = [allowed]
        denied = [allowed]
    envelope, _ = _roundtrip(allowed)
    with pytest.raises(ValueError): v6.encode_metadata_v6(denied)
    if kind == "depth":
        envelope["root"] = ["L", [envelope["root"]]]
        with pytest.raises(ValueError, match="METADATA_DEPTH_LIMIT_EXCEEDED"): v6.decode_metadata_v6(envelope)


@pytest.mark.parametrize("container", ("list", "dict"))
def test_cycles_are_bounded_depth_refusals_not_infinite_encoding(container):
    value = [] if container == "list" else {}
    if container == "list": value.append(value)
    else: value["self"] = value
    with pytest.raises(ValueError, match="METADATA_DEPTH_LIMIT_EXCEEDED"): v6.encode_metadata_v6(value)


def test_exact_seeded_work_and_array_hash_limits_without_large_allocation():
    work = v6._Work()
    work.visits = VISIT_LIMIT-1
    work.visit(64)
    assert work.visits == VISIT_LIMIT
    with pytest.raises(ValueError, match="METADATA_NODE_WORK_LIMIT_EXCEEDED"): work.visit(0)
    with pytest.raises(ValueError, match="METADATA_DEPTH_LIMIT_EXCEEDED"): v6._Work().visit(65)
    array = np.array([1, 2, 3], dtype=np.uint8)
    work.hash_bytes = HASH_LIMIT-3
    work.array(array)
    assert work.hash_bytes == HASH_LIMIT
    with pytest.raises(ValueError, match="METADATA_ARRAY_HASH_WORK_LIMIT_EXCEEDED"): work.array(array)


@pytest.mark.parametrize("direction", ("encode", "decode"))
def test_seeded_expanded_and_visit_denominators_have_exact_small_tree_boundaries(monkeypatch, direction):
    value = {"long_key": [False, -0., "\u03bb"]}
    envelope = v6.encode_metadata_v6(value)
    expanded_size = len(_bytes(value))
    visits, _ = _tree_work(value)
    if direction == "decode": visits += 1  # One schema key admitted before decoding.
    Work, Expanded = v6._Work, v6._Expanded
    for deficit, succeeds in ((0, True), (1, False)):
        def seeded_work():
            result = Work()
            result.visits = VISIT_LIMIT-visits-deficit
            return result
        def seeded_expanded():
            result = Expanded()
            result.size = EXPANDED_LIMIT-expanded_size+deficit
            return result
        with monkeypatch.context() as patch:
            patch.setattr(v6, "_Work", seeded_work)
            patch.setattr(v6, "_Expanded", seeded_expanded)
            function, argument = (v6.encode_metadata_v6, value) if direction == "encode" else (v6.decode_metadata_v6, envelope)
            if succeeds: function(argument)
            else:
                with pytest.raises(ValueError, match="EXPANDED_METADATA_BYTE_LIMIT_EXCEEDED"): function(argument)


def _admission_stats():
    return dict(metadata_bytes=0, metadata_expanded_bytes=0, metadata_node_visits=0, metadata_array_bytes_hashed=0)


def test_one_incremental_schema_table_charges_new_keys_once_before_append():
    work = v6._Work()
    codec, stats, target = v6._Codec(work), _admission_stats(), []
    first, second = dict(a=1, b=False), dict(a=2, b=None)
    expected_first = len(_bytes(["D", 0, [1, False]]))+len(_bytes(["a", "b"]))
    expected_second = len(_bytes(["D", 0, [2, None]]))
    cap = dict(max_metadata_bytes=expected_first+expected_second)
    v6._append(target, first, stats, cap, codec)
    assert stats["metadata_bytes"] == expected_first and codec.schemas == [["a", "b"]]
    v6._append(target, second, stats, cap, codec)
    assert stats["metadata_bytes"] == expected_first+expected_second and target == [first, second]
    assert stats["metadata_expanded_bytes"] == len(_bytes(first))+len(_bytes(second))
    assert stats["metadata_node_visits"] == sum(_tree_work(row)[0] for row in target)
    with pytest.raises(ValueError): v6._append(target, dict(a=3, b=True), stats, cap, codec)
    assert target == [first, second]  # The overflow record was never published.


def test_fresh_final_schema_table_includes_late_fields_without_admission_history():
    stats, target, work = _admission_stats(), [], v6._Work()
    codec = v6._Codec(work)
    record = dict(source_core_ids=[0], closure_disposition="NOT_SCHEDULED_BY_POLICY")
    v6._append(target, record, stats, dict(max_metadata_bytes=ENCODED_LIMIT), codec)
    record["source_core_ids"].append(7)
    record.update(closure_disposition="EXACT_SUPPORT_DUPLICATE", cone_proposal=dict(possible_raw_rows=81, joint_compatibility_verified=False))
    envelope, decoded = _roundtrip(target)
    assert codec.schemas[0] == ["source_core_ids", "closure_disposition"]
    assert envelope["schemas"][0] == ["source_core_ids", "closure_disposition", "cone_proposal"]
    assert decoded[0]["source_core_ids"] == [0, 7]


def _grid(*, hole=False, span=16.):
    coordinates = np.linspace(-span/2, span/2, 9)
    points = np.array([[x, y, 0.] for y in coordinates for x in coordinates])
    triangles = []
    for j in range(8):
        for i in range(8):
            if hole and i in (3, 4) and j in (3, 4): continue
            a, b, c, d = j*9+i, j*9+i+1, (j+1)*9+i, (j+1)*9+i+1
            triangles.extend(((a, b, c), (b, d, c)))
    return points, np.asarray(triangles, dtype=np.int64)


def _run(module, points, triangles, epsilon=.05, **kwargs):
    function = module.extract_noisy_scan_plane_proposals_v5 if module is v5 else module.extract_noisy_scan_plane_proposals_v6
    return function(points, triangles, position_error_bounds_mm=epsilon, **kwargs)


def _boundary(result):
    assert result["uncertainty_status"] == "STIPULATED_UNCALIBRATED"
    assert result["physical_plane_count"] is result["observed_bend_count"] is None
    for key in ("production_use_allowed", "accepted_registration", "physical_flatness_verified",
                "residual_trimming", "raw_coordinate_welding", "search_exhaustive"):
        assert result[key] is False


def _failure(result, discarded=0):
    assert result["status"] == "UNAVAILABLE" and result["reasons"]
    assert all(result[key] == [] for key in ("cores", "closures", "proposals", "partitions"))
    stats = result["statistics"]
    assert stats["discarded_core_records"] >= discarded
    for key in ("core_construction_complete", "core_evaluation_complete", "scheduled_closure_evaluation_complete", "all_qualified_core_closures_evaluated"):
        assert stats[key] is False
    assert stats["evidence_scope"] == "ATTEMPTED_WORK_ONLY_ALL_PROPOSALS_INVALIDATED"
    _boundary(result)


def _without_declared_representation_fields(result):
    comparable = dict(result)
    comparable["statistics"] = dict(result["statistics"])
    comparable.pop("schema_version")
    comparable.pop("metadata_encoding", None)
    for key in RESOURCE_FIELDS: comparable["statistics"].pop(key, None)
    return comparable


def _internal_equal(first, second):
    assert type(first) is type(second)
    if type(first) is np.ndarray:
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes(order="C") == second.tobytes(order="C")
        assert not first.flags.writeable and not second.flags.writeable
    elif type(first) is dict:
        assert list(first) == list(second)
        for key in first: _internal_equal(first[key], second[key])
    elif type(first) in (tuple, list):
        assert len(first) == len(second)
        for a, b in zip(first, second): _internal_equal(a, b)
    else: _strict_equal(first, second)


def _paired(points, triangles, epsilon=.05, **kwargs):
    baseline = _run(v5, points, triangles, epsilon, **kwargs)
    successor = _run(v6, points, triangles, epsilon, **kwargs)
    assert baseline["status"] == successor["status"] == "COMPLETE", (baseline["reasons"], successor["reasons"])
    _internal_equal(_without_declared_representation_fields(baseline), _without_declared_representation_fields(successor))
    _strict_equal(v6.decode_metadata_v6(v6.encode_metadata_v6(successor)), v5._metadata(successor))
    for result in (baseline, successor):
        _boundary(result)
        for record in [*result["cores"], *result["closures"]]:
            rows = record["triangle_rows"]
            np.testing.assert_array_equal(record["raw_vertex_rows"], np.unique(triangles[rows].reshape(-1)))
            for name in ("triangle_rows", "raw_vertex_rows"):
                array = record[name]
                assert array.dtype == np.dtype("<i8") and not array.flags.writeable
                assert hashlib.sha256(array.tobytes()).hexdigest() == record[name+"_sha256"]
                with pytest.raises(ValueError): array.setflags(write=True)
                base = array.base
                while isinstance(base, np.ndarray):
                    with pytest.raises(ValueError): base.setflags(write=True)
                    base = base.base
    return baseline, successor


def _witness(analysis, points, epsilon, true_normal):
    assert analysis["status"] == "PROPOSED" and analysis["witness_status"] == "FOUND" and analysis["cone_status"] == "AVAILABLE"
    normal = np.asarray(analysis["normal"])
    residual = np.abs((points-analysis["witness_anchor_mm"])@normal-analysis["witness_offset_from_anchor_mm"])
    assert np.all(residual <= epsilon)  # Exact original stipulated bounds, no added tolerance.
    assert np.linalg.norm(normal) == pytest.approx(1., abs=1e-14)
    assert analysis["raw_support_count"] == len(points) and analysis["same_physical_plane_verified"] is False
    angle = math.atan2(float(np.linalg.norm(np.cross(normal, true_normal))), abs(float(normal@true_normal)))
    assert angle <= analysis["normal_angle_bound_radians"]
    assert analysis["numerical"]["position_error_bound_was_increased"] is False


@pytest.mark.parametrize("frame", range(4))
def test_actual_frozen_noisy_frames_have_exact_v5_v6_geometry_and_full_evidence(frame):
    truth, triangles = _grid()
    rotation, translation, scale = FIXED_FRAMES[frame]
    local = truth.copy()
    local[:, 2] = [.01*(-1 if (i//9+i%9) % 2 else 1) for i in range(81)]
    observed, clean = (local*scale)@rotation.T+translation, (truth*scale)@rotation.T+translation
    epsilon = np.full(81, .05*scale)
    assert np.all(np.linalg.norm(observed-clean, axis=1) < epsilon)
    _, result = _paired(observed, triangles, epsilon)
    assert len(result["cores"]) == 2 and len(result["closures"]) == 1
    assert result["proposals"] == [dict(kind="CORE", record_id=0), dict(kind="CORE", record_id=1), dict(kind="CLOSURE", record_id=0)]
    assert result["closures"][0]["source_core_ids"] == [0, 1]
    for record in [*result["cores"], *result["closures"]]:
        np.testing.assert_array_equal(record["triangle_rows"], np.arange(128))
        np.testing.assert_array_equal(record["raw_vertex_rows"], np.arange(81))
        _witness(record["analysis"], observed, epsilon, rotation[:, 2])
    expected = dict(edge_records=384, shared_edge_pairs=176, core_count=2, cores_evaluated=2,
        qualified_core_count=2, closures_scheduled=2, closures_evaluated=2, unique_closure_count=1,
        qualified_closure_count=1, fit_evaluations=3, full_support_row_tests=243, cone_row_tests=162,
        closure_triangle_tests=256, closure_edge_tests=352, triangle_memberships=640,
        raw_memberships=243, retained_array_bytes=9112, unscheduled_qualified_core_count=0)
    for key, value in expected.items(): assert result["statistics"][key] == value, key
    assert result["statistics"]["all_qualified_core_closures_evaluated"] is True


@pytest.mark.parametrize("mode", ("hole", "duplicate_xyz", "failed_closure", "rotated_zero", "quota_zero"))
def test_v5_v6_equality_retains_holes_identity_failures_and_declared_omissions(mode):
    points, triangles = _grid(hole=mode == "hole", span=40. if mode == "failed_closure" else 16.)
    epsilon, kwargs = .05, {}
    if mode == "duplicate_xyz":
        points, triangles = np.vstack((points, points)), np.vstack((triangles, triangles+81))
    elif mode == "failed_closure":
        points = np.vstack((points, [[20., 25., 0.], [25., 25., 0.], [0., 0., 1.], [0., 0., -1.]]))
        triangles = np.vstack((triangles, [[79, 80, 81], [80, 81, 82], [81, 82, 83], [82, 83, 84]]))
        epsilon = np.r_[np.ones(83), 0., 0.]
    elif mode == "rotated_zero": points, epsilon = points@R.T+[3., -4., 5.], 0.
    elif mode == "quota_zero": kwargs["budgets"] = replace(v5.NoisyMeshPlaneBudgetsV5(), max_closure_schedules=0)
    _, result = _paired(points, triangles, epsilon, **kwargs)
    if mode == "hole":
        assert len(triangles) == 120 and 40 not in triangles
        np.testing.assert_array_equal(result["raw_input_accounting"]["unreferenced_raw_vertex_rows"], [40])
        np.testing.assert_array_equal(result["closures"][0]["triangle_rows"], np.arange(120))
        centroids = points[triangles].mean(axis=1)
        assert not np.any((np.abs(centroids[:, 0]) < 2) & (np.abs(centroids[:, 1]) < 2))
    elif mode == "duplicate_xyz":
        assert {tuple(row["triangle_rows"]) for row in result["closures"]} == {tuple(range(128)), tuple(range(128, 256))}
    elif mode == "failed_closure":
        retained = [row for row in result["cores"] if row["closure_disposition"] == "WHOLE_CLOSURE_REJECTED_CORE_RETAINED"]
        assert retained
        for core in retained:
            assert core["analysis"]["status"] == "PROPOSED" and dict(kind="CORE", record_id=core["core_id"]) in result["proposals"]
            closure = result["closures"][core["closure_id"]]
            np.testing.assert_array_equal(closure["triangle_rows"], np.arange(132))
            np.testing.assert_array_equal(closure["raw_vertex_rows"], np.arange(85))
            assert closure["analysis"]["witness_status"] == "WITNESS_NOT_FOUND" and closure["analysis"]["status"] == "REJECTED"
            assert dict(kind="CLOSURE", record_id=core["closure_id"]) not in result["proposals"]
    elif mode == "rotated_zero":
        assert result["proposals"] == result["closures"] == []
        assert all(row["analysis"]["witness_status"] == "NUMERICALLY_UNRESOLVED_OFFSET_INTERVAL" for row in result["cores"])
    else:
        assert len(result["proposals"]) == 2 and result["scheduled_core_ids"] == []
        assert result["statistics"]["unscheduled_qualified_core_count"] == 2
        assert result["statistics"]["scheduled_closure_evaluation_complete"] is True
        assert result["statistics"]["all_qualified_core_closures_evaluated"] is False


def test_three_final_passes_share_admission_work_and_reproduce_exact_final_bytes(monkeypatch):
    points, triangles = _grid()
    append, envelope = v6._append, v6._envelope
    appends, passes = [], []
    def checked_append(target, record, stats, cap, codec):
        expected = _tree_work(record)
        before = codec.work.visits, codec.work.hash_bytes
        append(target, record, stats, cap, codec)
        assert (codec.work.visits-before[0], codec.work.hash_bytes-before[1]) == expected
        appends.append((id(codec), id(codec.work), *expected))
    def checked_envelope(value, work, maximum):
        expected = _tree_work(value)
        before = work.visits, work.hash_bytes
        result = envelope(value, work, maximum)
        assert (work.visits-before[0], work.hash_bytes-before[1]) == expected
        assert result[1] == len(_bytes(result[0])) and result[2] == len(_bytes(_logical(value)))
        passes.append((id(work), *expected, value["statistics"]["metadata_bytes"]))
        return result
    monkeypatch.setattr(v6, "_append", checked_append)
    monkeypatch.setattr(v6, "_envelope", checked_envelope)
    result = _run(v6, points, triangles)
    assert result["status"] == "COMPLETE", result["reasons"]
    assert len(passes) == 3 and len({row[0] for row in appends}) == 1
    assert {row[1] for row in appends} == {row[0] for row in passes}
    stats = result["statistics"]
    assert stats["metadata_node_visits"] == sum(row[2] for row in appends)+sum(row[1] for row in passes)
    assert stats["metadata_array_bytes_hashed"] == sum(row[3] for row in appends)+sum(row[2] for row in passes)
    assert len({row[1:3] for row in passes}) == 1
    exported, size, expanded = envelope(result, v6._Work(), ENCODED_LIMIT)
    assert stats["metadata_bytes"] == size == len(_bytes(exported))
    assert stats["metadata_expanded_bytes"] == expanded == len(_bytes(_logical(result)))
    _strict_equal(v6.decode_metadata_v6(exported), _logical(result))


def test_actual_append_and_final_overflow_clear_all_qualified_records(monkeypatch):
    points, triangles = _grid()
    _failure(_run(v6, points, triangles, budgets=replace(v5.NoisyMeshPlaneBudgetsV5(), max_metadata_bytes=1)))
    envelope, admission = v6._envelope, []
    def measured(value, work, maximum):
        if not admission: admission.append(value["statistics"]["metadata_bytes"])
        return envelope(value, work, maximum)
    with monkeypatch.context() as patch:
        patch.setattr(v6, "_envelope", measured)
        complete = _run(v6, points, triangles)
    assert complete["status"] == "COMPLETE"
    # This frozen small fixture leaves a substantial final-tree overhead; the
    # first-pass cap below admits every incremental record, but not final export.
    assert complete["statistics"]["metadata_bytes"]-admission[0] > 256
    failed = _run(v6, points, triangles, budgets=replace(v5.NoisyMeshPlaneBudgetsV5(), max_metadata_bytes=admission[0]))
    _failure(failed, discarded=2)
    assert failed["statistics"]["discarded_closure_records"] == 1 and failed["statistics"]["discarded_proposals"] == 3
    assert failed["reasons"] == ["ENCODED_METADATA_BYTE_LIMIT_EXCEEDED"]
    assert failed["statistics"]["metadata_node_visits"] > 0


@pytest.mark.parametrize("mode", ("size_drift", "structure_drift"))
def test_final_pass_nonconvergence_whole_refuses_without_unbounded_retry(monkeypatch, mode):
    points, triangles = _grid()
    envelope, calls = v6._envelope, []
    def drifting(value, work, maximum):
        result, size, expanded = envelope(value, work, maximum)
        calls.append(None)
        if mode == "structure_drift" and len(calls) == 1: value["late_invented_field"] = [1]
        return result, size+len(calls) if mode == "size_drift" else size, expanded
    monkeypatch.setattr(v6, "_envelope", drifting)
    result = _run(v6, points, triangles)
    _failure(result, discarded=2)
    assert len(calls) == 3 and result["reasons"] == ["METADATA_FINAL_ACCOUNTING_UNRESOLVED"]
