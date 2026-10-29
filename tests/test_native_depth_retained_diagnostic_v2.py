"""Invented retained-wire controls; no native or physical data is loaded.

The imported fixture is only the frozen explicit mocked supplied-object builder,
not an old test or pytest fixture invoked directly. Scientific engine validators
remain mocked; the real retained helper's common/context/association/quantity
predicates and complete orchestration execute. An independent struct encoder
supplies PLY identity buffers, not a geometric accuracy fixture.
"""

import ast
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys
import types

import numpy as np
import pytest

from tests import native_depth_retained_diagnostic_v2 as ledger
from tests.test_native_depth_diagnostic_ledger_v1 import supplied


ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = "tests/native_depth_retained_diagnostic_v2.py"
SOURCE_DEPENDENCIES = (HELPER_SOURCE, "tests/native_depth_diagnostic_ledger_v1.py",
    "tests/test_native_depth_diagnostic_ledger_v1.py", "tests/__init__.py")
HEADER = (b"ply\nformat binary_little_endian 1.0\n"
    b"comment invented partial mesh, units/frame unverified by importer\n"
    b"element vertex 650\nproperty double x\nproperty double y\nproperty double z\n"
    b"element face 1176\nproperty list uchar int vertex_indices\nend_header\n")
SUCCESS = {"COMPLETE_CONDITIONAL_INTEGRITY", "COMPLETE_NO_MEASUREMENTS_INTEGRITY"}


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _ply(raw, triangles):
    """Independent stdlib byte packing, never the decoder under test."""
    return HEADER+b"".join(struct.pack("<ddd", *row) for row in raw)+b"".join(
        struct.pack("<Biii", 3, *row) for row in triangles)


def _attestation(sample):
    # Distinct byte/canonical domains on purpose. This is a MOCK driver receipt,
    # not a claim that any retained original/native file has been read.
    original = json.dumps(sample.record, indent=1, allow_nan=False).encode()+b"\n"
    canonical = _json(sample.record)
    artifact = dict(filename="CASE_RESULT.json", bytes=len(original), sha256=_sha(original))
    inputs = dict(sample.hashes, raw_rows=650, triangle_rows=1176)
    binding = dict(schema="NATIVE_DEPTH_RETAINED_BINDING_V2", original_result_artifact=copy.deepcopy(artifact),
        canonical_record_before=dict(bytes=len(canonical), sha256=_sha(canonical)), input_binding=inputs,
        source_bundle_sha256=ledger.ORIGINAL_SOURCE_MAP_SHA256, original_terminal_output_sha256="8"*64,
        driver_custody_status=ledger.CUSTODY_STATUS, original_accuracy_status="FAIL", original_diagnostic_status="FAIL")
    return original, artifact, binding


def _call(sample, *, artifact=None, binding=None):
    if artifact is None or binding is None:
        _, artifact, binding = _attestation(sample)
    return ledger.evaluate_retained_diagnostic_v2(sample.fixture, sample.record, artifact,
        sample.cad, sample.scan, sample.raw, sample.triangles, binding)


def _criterion(answer, name):
    return next(row for row in answer["criteria"] if row["name"] == name)


@dataclass(frozen=True)
class CompositeBudget:
    max_metadata_bytes: int = 65536


@pytest.fixture
def retained(supplied, monkeypatch):
    """Explicitly adapt the old MOCK schema to strict retained JSON/PLY."""
    sample = supplied
    monkeypatch.setattr(ledger, "_deadline", lambda seconds: nullcontext())
    sample.scan = _ply(sample.raw, sample.triangles)
    sample.hashes["scan_sha256"] = _sha(sample.scan)
    result = sample.result
    result["input_preparation"]["source_bindings"]["scan_sha256"] = sample.hashes["scan_sha256"]
    result["scan_triangle_binding"]["source_sha256"] = sample.hashes["scan_sha256"]
    result["measurements"]["scan_sha256"] = sample.hashes["scan_sha256"]
    sample.record["input_binding"]["scan_sha256"] = sample.hashes["scan_sha256"]
    sample.record["input_artifacts"][1] = dict(filename="SCAN.ply", bytes=len(sample.scan), sha256=sample.hashes["scan_sha256"])
    cad = result["cad_surface_sampling"]
    cad["digests"].update(sample_normals_sha256="f"*64, sample_triangle_rows_sha256="1"*64,
        sampling_policy_sha256="2"*64, mesher_policy_sha256="3"*64,
        array_encoding="C_CONTIGUOUS_LITTLE_ENDIAN_FLOAT64_OR_INT64",
        scope="INPUT_TRIANGLE_AND_SAMPLE_BYTES_NOT_NATIVE_SOURCE_AUTHORIZATION")
    cad["coverage"] = dict(sample_count_by_source_face={"2": 10000, "10": 22768},
        area_by_source_face={"2": 1., "10": 2.})
    sample.kwargs["composite_plane_budgets"] = CompositeBudget()
    expected_config = dict(mode="composite_v1", seeded_budgets={"limit": 7}, composite_budgets=asdict(CompositeBudget()))
    result["policies"]["scan_plane_configuration"] = copy.deepcopy(expected_config)
    init = result["registration"]["evidence"]["initialization"]
    # Literal valid schema-table shape. Decoder itself is an explicit mock here;
    # its full frozen grammar is independently qualified, not reimplemented.
    envelope = {"format": "noisy_mesh_metadata_schema_v6", "root": ["D", 0, [True]], "schemas": [["mock"]]}
    init["patch_evidence"] = {"scan": envelope}
    metadata = {"mock": True}
    def decode(value):
        sample.calls.append("decode")
        assert list(value) == ["format", "schemas", "root"]
        assert value is not envelope and value["schemas"] is envelope["schemas"] and value["root"] is envelope["root"]
        assert type(value["root"][1]) is int
        assert value == {"format": "noisy_mesh_metadata_schema_v6", "schemas": [["mock"]], "root": ["D", 0, [True]]}
        return metadata
    sample.modules["cad_guided_noisy_mesh_planes_v7"].decode_metadata_v7 = decode
    def composite(value, decoded, digest, expected, raw, triangles):
        sample.calls.append("composite")
        assert value is result and decoded is metadata and raw is sample.raw and triangles is sample.triangles
        assert ledger._same(expected, expected_config)
        assert ledger._same(init["mock_composite_configuration"], {"mode": "composite_v1", "complete": True})
        ordered = {"format": envelope["format"], "schemas": envelope["schemas"], "root": envelope["root"]}
        assert digest == _sha(json.dumps(ordered, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode())
    sample.fixture._assert_composite_metadata = composite
    sample.envelope = envelope
    return sample


@pytest.mark.parametrize("measured", [True, False])
def test_all_eight_mock_groups_and_original_failure_domains_preserved(retained, measured):
    if not measured:
        result, value = retained.result, retained.result["measurements"]
        for group in value["support_groups"]:
            group.update(status="UNAVAILABLE", measurements=None, finite_support=None, reasons=["NO_EXISTING_COVER"])
        value.update(status="UNAVAILABLE", predictions=[], reasons=["NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS"])
        result.update(status="UNAVAILABLE", predictions=[], reasons=list(value["reasons"]))
    original, artifact, binding = _attestation(retained)
    before = _json(retained.record)
    answer = _call(retained, artifact=artifact, binding=binding)
    assert answer["schema"] == "NATIVE_DEPTH_RETAINED_DIAGNOSTIC_LEDGER_V2"
    assert answer["status"] == ("COMPLETE_CONDITIONAL_INTEGRITY" if measured else "COMPLETE_NO_MEASUREMENTS_INTEGRITY")
    assert len(answer["criteria"]) == 8 and all(row["status"] == "PASS" for row in answer["criteria"])
    assert answer["denominators"] == dict(native_features=2, contexts=2, windows=4, support_records=2,
        support_groups=2, predictions=2 if measured else 0)
    assert answer["canonical_record_before"] == answer["canonical_record_after"] == dict(bytes=len(before), sha256=_sha(before))
    assert answer["original_result_artifact"] == artifact and artifact["sha256"] != _sha(before)
    assert _sha(original) == artifact["sha256"] and _json(retained.record) == before
    assert answer["driver_attestation_checked"] is True and answer["immutable_decoded_evidence_and_supplied_arrays"] is True
    assert answer["supplied_binding_before"] == answer["supplied_binding_after"]
    assert 0 < answer["resources"]["supplied_binding_preflight_nodes"] <= 512
    assert answer["original_accuracy_status"] == answer["original_diagnostic_status"] == "FAIL"
    assert answer["assurance"] == "SERIALIZATION_RECONSTRUCTED_REPORT_PLUS_SOURCE_ENFORCED_NATIVE_CHECK"
    assert all(answer[key] is False for key in ("accuracy_evaluated", "file_identity_reverified", "truth_accessed",
        "original_runtime_types_recovered", "former_worker_cache_identity_recovered", "native_arrays_reconstructed"))
    assert answer["inference_calls"] == answer["refit_calls"] == 0
    assert retained.calls.count("cover") == retained.calls.count("projection") == 1
    assert len(_json(answer)) <= 32768


def test_exact_decoder_preserves_duplicates_row_order_and_readonly_bytes(retained):
    raw, triangles = retained.raw.copy(), retained.triangles.copy()
    raw[20] = raw[19]  # Distinct raw IDs with the same XYZ are not welded.
    triangles[20] = triangles[19]  # Duplicate triangle tuple is not removed.
    blob = _ply(raw, triangles)
    decoded_raw, decoded_triangles = ledger.decode_retained_ply_v2(blob)
    assert len(blob) == 31131 and decoded_raw.shape == (650, 3) and decoded_triangles.shape == (1176, 3)
    assert decoded_raw.dtype == np.dtype("<f8") and decoded_triangles.dtype == np.dtype("<i8")
    assert np.array_equal(decoded_raw, raw) and np.array_equal(decoded_triangles, triangles)
    assert _sha(decoded_raw.tobytes()) == _sha(raw.astype("<f8").tobytes())
    assert _sha(decoded_triangles.tobytes()) == _sha(triangles.astype("<i8").tobytes())
    for value in (decoded_raw, decoded_triangles):
        assert value.flags.writeable is False
        with pytest.raises(ValueError): value.setflags(write=True)


@pytest.mark.parametrize("fault", ["truncated", "extra", "header", "arity", "negative", "too_large",
    "repeated_index", "nan", "infinite", "coordinate_range", "not_bytes"])
def test_exact_ply_schema_refuses_only_declared_representation_faults(retained, fault):
    value = bytearray(retained.scan)
    face = len(HEADER)+650*24
    if fault == "truncated": value.pop()
    elif fault == "extra": value += b"x"
    elif fault == "header": value[0] = ord("P")
    elif fault == "arity": value[face] = 4
    elif fault == "negative": struct.pack_into("<i", value, face+1, -1)
    elif fault == "too_large": struct.pack_into("<i", value, face+1, 650)
    elif fault == "repeated_index": struct.pack_into("<i", value, face+1, 1)
    elif fault == "nan": struct.pack_into("<d", value, len(HEADER), float("nan"))
    elif fault == "infinite": struct.pack_into("<d", value, len(HEADER), float("inf"))
    elif fault == "coordinate_range": struct.pack_into("<d", value, len(HEADER), 1e6+1.)
    with pytest.raises(ValueError):
        ledger.decode_retained_ply_v2(value if fault == "not_bytes" else bytes(value))


def test_only_expected_tuple_and_exact_float64_are_json_normalized():
    expected = {"steps": (128, 1024), "guard": np.float64(-0.), "enabled": True, "ratio": 1.}
    wire = ledger._wire_expected(expected)
    assert wire == {"steps": [128, 1024], "guard": -0., "enabled": True, "ratio": 1.}
    assert type(wire["guard"]) is float and np.signbit(wire["guard"])
    assert type(wire["enabled"]) is bool and type(wire["ratio"]) is float
    assert type(expected["steps"]) is tuple and type(expected["guard"]) is np.float64
    assert not ledger._same(wire, dict(wire, enabled=1))
    assert not ledger._same(wire, dict(wire, ratio=1))


@pytest.mark.parametrize("value", [np.float32(.5), np.int64(1), complex(1), float("nan"), "x"*4097, 1 << 129])
def test_no_generic_expected_policy_coercion_or_unbounded_scalar(value):
    with pytest.raises(ValueError): ledger._wire_expected({"value": value})


@pytest.mark.parametrize("fault", ["digest_text", "digest_missing", "digest_extra", "digest_bool", "face_integer_key",
    "face_leading_zero", "face_bool_count", "face_total", "area_int", "area_nonfinite"])
def test_typed_sampler_json_schema_not_all_values_hex(retained, fault):
    cad = retained.result["cad_surface_sampling"]
    if fault == "digest_text": cad["digests"]["scope"] = "0"*64
    elif fault == "digest_missing": del cad["digests"]["mesher_policy_sha256"]
    elif fault == "digest_extra": cad["digests"]["future"] = "0"*64
    elif fault == "digest_bool": cad["digests"]["sample_normals_sha256"] = True
    elif fault == "face_integer_key": cad["coverage"]["sample_count_by_source_face"] = {2: 10000, 10: 22768}
    elif fault == "face_leading_zero": cad["coverage"]["sample_count_by_source_face"] = {"02": 10000, "10": 22768}
    elif fault == "face_bool_count": cad["coverage"]["sample_count_by_source_face"]["2"] = True
    elif fault == "face_total": cad["coverage"]["sample_count_by_source_face"]["2"] -= 1
    elif fault == "area_int": cad["coverage"]["area_by_source_face"]["2"] = 1
    elif fault == "area_nonfinite": cad["coverage"]["area_by_source_face"]["2"] = float("nan")
    with pytest.raises(ValueError): ledger._common(retained.result, dict(retained.hashes, raw_rows=650, triangle_rows=1176))


@pytest.mark.parametrize("fault", [None, "extra_key", "wrong_format", "bad_root"])
def test_codec_only_outer_order_restored_without_mutating_report(retained, fault):
    if fault == "extra_key": retained.envelope["extra"] = None
    elif fault == "wrong_format": retained.envelope["format"] = "old"
    elif fault == "bad_root": retained.envelope["root"][1] = False
    before = copy.deepcopy(retained.envelope)
    original_order = list(retained.envelope)
    if fault is None:
        ledger._composite(retained.fixture, retained.result, retained.kwargs, retained.raw, retained.triangles)
        assert retained.calls == ["decode", "composite"]
    else:
        with pytest.raises((ValueError, AssertionError)):
            ledger._composite(retained.fixture, retained.result, retained.kwargs, retained.raw, retained.triangles)
    assert retained.envelope == before and list(retained.envelope) == original_order


@pytest.mark.parametrize("fault", ["original_sha", "canonical_sha", "input_sha", "source_map", "custody",
    "old_accuracy", "old_diagnostic", "artifact_bool"])
def test_driver_attestation_fault_veto_before_scientific_checks(retained, fault):
    _, artifact, binding = _attestation(retained)
    if fault == "original_sha": artifact["sha256"] = "0"*64
    elif fault == "canonical_sha": binding["canonical_record_before"]["sha256"] = "0"*64
    elif fault == "input_sha": binding["input_binding"]["scan_sha256"] = "0"*64
    elif fault == "source_map": binding["source_bundle_sha256"] = "0"*64
    elif fault == "custody": binding["driver_custody_status"] = "COMPLETE"
    elif fault == "old_accuracy": binding["original_accuracy_status"] = "PASS"
    elif fault == "old_diagnostic": binding["original_diagnostic_status"] = "PASS"
    elif fault == "artifact_bool": artifact["bytes"] = True
    answer = _call(retained, artifact=artifact, binding=binding)
    assert answer["status"] not in SUCCESS and answer["driver_attestation_checked"] is False
    assert retained.calls == []


def test_consistently_reattested_raw_still_must_equal_exact_ply_rows(retained):
    # Explicitly forged MOCK attestation: even matching report/hash claims do
    # not let a supplied array replace the independently decoded PLY rows.
    retained.raw[0, 0] += 1.
    retained.hashes["raw_xyz_sha256"] = _sha(retained.raw.astype("<f8").tobytes())
    retained.record["input_binding"]["raw_xyz_sha256"] = retained.hashes["raw_xyz_sha256"]
    answer = _call(retained)
    assert answer["status"] not in SUCCESS and retained.calls == []
    errors = _criterion(answer, "publication_source_input")["errors"]
    assert any(row["error_message"] == "SUPPLIED_ARRAYS_NOT_EXACT_RETAINED_PLY" for row in errors)


@pytest.mark.parametrize("fault", [None, "cache_miss", "backend_reload"])
def test_fresh_preloaded_constructor_state_not_former_worker_identity(retained, monkeypatch, fault):
    retained.native_cache.update(currsize=2, misses=2, hits=4)
    original = retained.fixture._policy_objects
    def policies():
        answer = original()
        retained.native_cache["hits"] += 1
        if fault == "cache_miss": retained.native_cache["misses"] += 1
        elif fault == "backend_reload": monkeypatch.setitem(sys.modules, "backend.cad_guided_mesh_offline_v13", types.ModuleType("changed"))
        return answer
    monkeypatch.setattr(retained.fixture, "_policy_objects", policies)
    answer = _call(retained)
    assert (answer["status"] in SUCCESS) is (fault is None)
    assert answer["former_worker_cache_identity_recovered"] is False
    if fault: assert retained.calls == []


@pytest.mark.parametrize("fault", ["origin_last", "deviation_last", "quantity_last", "projection", "policy_bool", "false_partial"])
def test_real_reused_predicates_reject_late_scientific_corruption(retained, fault):
    result = retained.result
    if fault == "origin_last":
        result["registration"]["evidence"]["initialization"]["hypotheses"][-1]["correspondences"][-1]["normal_sign"] = False
    elif fault == "deviation_last": result["measurements"]["support_groups"][-1]["associations"][-1]["canonical_radius_deviation_mm"] = 0.
    elif fault == "quantity_last": result["measurements"]["support_groups"][-1]["measurements"]["physical_trim_endpoints"] = [[0., 0., 0.]]
    elif fault == "projection": result["predictions"].append({"nested_member_is_not_a_top_prediction": True})
    elif fault == "policy_bool": result["policies"]["measurement"]["limit"] = True
    elif fault == "false_partial":
        value = result["measurements"]
        for group in value["support_groups"]:
            group.update(status="UNAVAILABLE", measurements=None, finite_support=None, reasons=["NO_EXISTING_COVER"])
        value.update(status="UNAVAILABLE", predictions=[], reasons=["NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS"])
        result.update(status="PARTIAL", predictions=[], reasons=list(value["reasons"]))
    answer = _call(retained)
    assert answer["status"] not in SUCCESS
    assert retained.calls.count("cover") == retained.calls.count("projection") == 1
    assert all(row["status"] != "UNREACHED" for row in answer["criteria"])


def test_all_late_rows_and_unknown_bulk_progress_survive_failure(retained, monkeypatch):
    rows = retained.result["measurements"]["support_groups"][-1]["associations"]
    rows[:] = [dict(rows[0], canonical_radius_deviation_mm=99.) for _ in range(12)]
    def broken_cover(*args, **kwargs):
        retained.calls.append("cover")
        raise RuntimeError("MOCK_UNINSTRUMENTED_BULK_FAILURE")
    monkeypatch.setattr(retained.modules["cad_guided_support_cover_measurement_v2"], "validate_support_cover_publication_v2", broken_cover)
    answer = _call(retained)
    nominal = _criterion(answer, "windows_nominals")
    assert nominal["failed"] == 12 and nominal["visited"] == nominal["planned"]
    assert len(nominal["errors"]) == 8 and nominal["errors_truncated"] is True
    proof = _criterion(answer, "raw_support_cover")["bulk_proofs"][0]
    assert proof["validator_returned"] is False and proof["internally_visited_rows"] is None
    assert proof["unreached_rows_upper_bound"] == proof["declared_rows"] == 10
    assert _criterion(answer, "quantity_carrier_projection")["status"] == "PASS"


@pytest.mark.parametrize("target", ["record", "raw", "triangles", "artifact", "attestation"])
def test_late_mutation_of_any_supplied_evidence_domain_vetoes(retained, monkeypatch, target):
    original, artifact, binding = _attestation(retained)
    projection = retained.fixture._assert_top_level_cover_projection
    def mutate(result):
        projection(result)
        if target == "record": retained.record["late_mutation"] = True
        elif target == "raw": retained.raw[0, 0] += 1.
        elif target == "triangles": retained.triangles[0, 0] = 4
        elif target == "artifact": artifact["sha256"] = "0"*64
        else: binding["original_terminal_output_sha256"] = "0"*64
    monkeypatch.setattr(retained.fixture, "_assert_top_level_cover_projection", mutate)
    answer = _call(retained, artifact=artifact, binding=binding)
    assert answer["status"] == "FAIL" and answer["immutable_decoded_evidence_and_supplied_arrays"] is False
    assert _criterion(answer, "publication_source_input")["status"] == "FAIL"
    assert original.endswith(b"\n") and json.loads(original).get("late_mutation") is None


@pytest.mark.parametrize("fault", ["np_float64_actual", "tuple_actual", "nan", "cycle", "nodes", "depth", "bytes"])
def test_decoded_record_preflight_is_strict_and_failed_prefix_not_success(retained, monkeypatch, fault):
    _, artifact, binding = _attestation(retained)
    if fault == "np_float64_actual": retained.record["invalid"] = np.float64(1.)
    elif fault == "tuple_actual": retained.record["invalid"] = (1, 2)
    elif fault == "nan": retained.record["invalid"] = float("nan")
    elif fault == "cycle": retained.record["invalid"] = retained.record
    elif fault == "nodes": monkeypatch.setattr(ledger, "MAX_NODES", 50)
    elif fault == "depth": monkeypatch.setattr(ledger, "MAX_DEPTH", 2)
    elif fault == "bytes": monkeypatch.setattr(ledger, "MAX_RECORD_BYTES", 64)
    answer = _call(retained, artifact=artifact, binding=binding)
    assert answer["status"] not in SUCCESS and retained.calls == []
    assert answer["immutable_decoded_evidence_and_supplied_arrays"] is False


def test_ledger_overflow_raises_and_never_returns_admitted_prefix(retained, monkeypatch):
    monkeypatch.setattr(ledger, "MAX_LEDGER_BYTES", 256)
    with pytest.raises(ValueError, match="LEDGER_BYTE_LIMIT"):
        _call(retained)


def test_bounded_error_detail_preserves_counts_without_rendering_or_catching_interrupts():
    criterion = ledger._Criterion("MOCK", ledger._Work())
    class UnrenderableError(ValueError):
        def __str__(self): raise AssertionError("DO_NOT_RENDER_ARBITRARY_EXCEPTION")
    def failure(): raise UnrenderableError("CODE:"+"x"*400)
    for index in range(10): criterion.check("failure_"+str(index), failure)
    assert criterion.row["planned"] == criterion.row["visited"] == criterion.row["failed"] == 10
    assert len(criterion.row["errors"]) == 8 and criterion.row["errors_truncated"] is True
    assert criterion.row["errors"][0]["error_message"] == "CODE:"+"x"*251
    assert criterion.row["errors"][0]["error_message_truncated"] is True
    def stop(): raise KeyboardInterrupt("DO_NOT_CAPTURE_INTERRUPT")
    with pytest.raises(KeyboardInterrupt): criterion.check("stop", stop)
    assert criterion.row["planned"] == criterion.row["visited"] == 11 and criterion.row["failed"] == 10


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, ledger._v1._BudgetInterrupt])
def test_baseexception_propagates_instead_of_becoming_pass(retained, monkeypatch, interrupt):
    def stop(*args, **kwargs): raise interrupt("MOCK_INTERRUPTION")
    monkeypatch.setattr(retained.fixture, "_assert_composite_metadata", stop)
    with pytest.raises(interrupt): _call(retained)


def test_early_refusal_is_not_complete_native_integrity(retained):
    retained.record["workflow"] = dict(status="UNAVAILABLE", phase="INPUT_PREPARATION", predictions=[], measurements=None)
    answer = _call(retained)
    assert answer["status"] not in SUCCESS
    assert _criterion(answer, "raw_support_cover")["status"] == "UNREACHED"
    assert retained.calls == []


def test_source_zero_file_native_fit_or_truth_calls_and_frozen_predicate_reuse():
    tree = ast.parse((ROOT/HELPER_SOURCE).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_retained_diagnostic_v2")
    assert [argument.arg for argument in function.args.args] == ["fixture_module", "record", "original_result_artifact",
        "cad_blob", "scan_blob", "raw", "triangles", "retained_binding"]
    forbidden = {"open", "read_bytes", "read_text", "_execute_workflow", "_make_fixture", "_evaluate", "_canonical_fit",
        "inspect_cad_guided_mesh_v13", "register_measurement_contexts_v12", "import_module", "exec", "eval", "compile"}
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            assert name not in forbidden
            calls.add(name)
        if isinstance(node, ast.ImportFrom): assert not (node.module or "").startswith(("backend", "OCP"))
        if isinstance(node, ast.Assign):
            assert not any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "_v1" for target in node.targets)
    assert {"_context_header", "_context_row", "_association", "_quantity", "_profile",
        "validate_support_cover_publication_v2", "_assert_schedule_metadata", "_assert_top_level_cover_projection"} <= calls
    assert ledger.CPU_SECONDS == ledger._v1.CPU_SECONDS == 90.
    assert ledger.WALL_SECONDS == ledger._v1.WALL_SECONDS == 120.
