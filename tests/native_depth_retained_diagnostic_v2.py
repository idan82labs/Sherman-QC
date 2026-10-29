"""Retained-byte diagnostic adapter; no native, inference, fixture or fit replay.

Original bytes and canonical decoded JSON are distinct evidence domains. Only
expected policy tuples and exact np.float64 values cross a JSON boundary.
The old accuracy and same-worker diagnostic remain failed. Source/inventory
authentication belongs to the independent driver and its postflight.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import sys

import numpy as np

from tests import native_depth_diagnostic_ledger_v1 as _v1

MAX_LEDGER_BYTES = 32*1024
MAX_RECORD_BYTES = 129*1024**2
MAX_NODES = 8_000_000
MAX_DEPTH = 64
CPU_SECONDS = 90.
WALL_SECONDS = 120.
CRITERIA = _v1.CRITERIA
_deadline = _v1._deadline
_require, _same, _hex, _sha = _v1._require, _v1._same, _v1._hex, _v1._sha
_rows, _integer, _loaded = _v1._rows, _v1._integer, _v1._loaded
_input_digest = _v1._input_digest
_context_header, _context_row = _v1._context_header, _v1._context_row
_profile, _association, _quantity = _v1._profile, _v1._association, _v1._quantity
ORIGINAL_SOURCE_MAP_SHA256 = "5f07c4166b4382fdf4bfe855ee1b6ea76c39803b063b2680cd366cbbd697d022"
CUSTODY_STATUS = "EXACT_TERMINAL_FILES_BOUND_BEFORE_RETAINED_CHECK_PENDING_POSTFLIGHT"
PLY_HEADER = (b"ply\nformat binary_little_endian 1.0\n"
    b"comment invented partial mesh, units/frame unverified by importer\n"
    b"element vertex 650\nproperty double x\nproperty double y\nproperty double z\n"
    b"element face 1176\nproperty list uchar int vertex_indices\nend_header\n")
DIGEST_KEYS = ("oriented_triangle_xyz_sha256", "source_face_ids_sha256",
    "source_triangle_ids_sha256", "sample_points_sha256", "sample_normals_sha256",
    "sample_triangle_rows_sha256", "sampling_policy_sha256", "mesher_policy_sha256")
DIGEST_TEXT = dict(array_encoding="C_CONTIGUOUS_LITTLE_ENDIAN_FLOAT64_OR_INT64",
    scope="INPUT_TRIANGLE_AND_SAMPLE_BYTES_NOT_NATIVE_SOURCE_AUTHORIZATION")


class _Criterion(_v1._Criterion):
    def check(self, label, function):
        """Same counters/interrupt semantics, with bounded first-string detail.

        Do not invoke arbitrary exception __str__ methods or stringify objects.
        A non-string first argument remains explicitly unavailable detail.
        """
        self.work.boundary()
        row = self.row
        row.update(status="PASS" if not row["failed"] else "FAIL", prerequisite="AVAILABLE")
        row["planned"] += 1
        row["visited"] += 1
        try:
            return function()
        except Exception as error:
            row["failed"] += 1
            row["status"] = "FAIL"
            if len(row["errors"]) < _v1.MAX_ERRORS:
                args = BaseException.args.__get__(error)
                message = args[0] if args and type(args[0]) is str else None
                row["errors"].append(dict(check=label[:96], error_type=type(error).__name__[:64],
                    error_message=None if message is None else message[:256],
                    error_message_truncated=message is not None and len(message) > 256))
            else:
                row["errors_truncated"] = True
            return None


def decode_retained_ply_v2(scan_blob):
    """Decode only the frozen byte schema, not a general PLY parser.

    Distinct equal XYZ rows and duplicate face tuples remain untouched. Only
    repeated indices within a triangle are refused, as in the original binding.
    No raw row is selected, synthesized, welded, fitted or geometrically changed.
    """
    expected_size = len(PLY_HEADER)+650*3*8+1176*13
    _require(type(scan_blob) is bytes and len(scan_blob) == expected_size
        and scan_blob[:len(PLY_HEADER)] == PLY_HEADER, "EXACT_RETAINED_PLY_SCHEMA_AND_LENGTH")
    offset = len(PLY_HEADER)
    raw = np.frombuffer(scan_blob[offset:offset+650*24], dtype="<f8").reshape(650, 3)
    packed = np.frombuffer(scan_blob[offset+650*24:],
        dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    _require(np.all(packed["count"] == 3), "RETAINED_PLY_TRIANGLE_ARITY")
    triangles = np.frombuffer(np.asarray(packed["indices"], dtype="<i8").tobytes(),
        dtype="<i8").reshape(1176, 3)
    _require(np.isfinite(raw).all() and np.all(raw >= -1e6) and np.all(raw <= 1e6)
        and np.all(triangles >= 0) and np.all(triangles < 650)
        and np.all(triangles[:, 0] != triangles[:, 1])
        and np.all(triangles[:, 0] != triangles[:, 2])
        and np.all(triangles[:, 1] != triangles[:, 2]), "RETAINED_PLY_COORDINATE_OR_INDEX_RANGE")
    return raw, triangles


class _Work(_v1._Work):
    def preflight(self, value):
        """Strict decoded JSON only; no runtime-type reconstruction."""
        stack = [(iter((value,)), 0)]
        while stack:
            iterator, depth = stack[-1]
            try:
                item = next(iterator)
            except StopIteration:
                stack.pop()
                continue
            self.nodes += 1
            _require(self.nodes <= MAX_NODES and depth <= MAX_DEPTH, "RETAINED_TREE_LIMIT")
            if self.nodes % 4096 == 0:
                self.boundary()
            kind = type(item)
            if kind is dict:
                _require(len(item) <= MAX_NODES-self.nodes, "RETAINED_KEY_LIMIT")
                for key in item:
                    _require(type(key) is str, "DECODED_JSON_KEY_TYPE")
                    self.nodes += 1
                stack.append((iter(item.values()), depth+1))
            elif kind is list:
                _require(len(item) <= MAX_NODES-self.nodes, "RETAINED_LIST_LIMIT")
                stack.append((iter(item), depth+1))
            else:
                _require(kind in (str, int, float, bool, type(None)), "DECODED_JSON_SCALAR_TYPE")
                if kind is float:
                    _require(math.isfinite(item), "DECODED_JSON_NONFINITE")


def _record_digest(record, work):
    work.preflight(record)
    digest, size = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(record):
        blob = fragment.encode()
        size += len(blob)
        _require(size <= MAX_RECORD_BYTES, "CANONICAL_DECODED_RECORD_BYTE_LIMIT")
        digest.update(blob)
        work.boundary()
    return dict(bytes=size, sha256=digest.hexdigest())


def _wire_expected(value):
    """Only the source-declared expected JSON representation, never actual data."""
    visits = 0
    def convert(item, depth=0):
        nonlocal visits
        visits += 1
        _require(visits <= 65536 and depth <= 64, "EXPECTED_POLICY_TREE_LIMIT")
        kind = type(item)
        if kind is dict:
            _require(len(item) <= 128 and all(type(key) is str and len(key) <= 128 for key in item), "EXPECTED_POLICY_KEYS")
            return {key: convert(child, depth+1) for key, child in item.items()}
        if kind in (tuple, list):
            _require(len(item) <= 65536, "EXPECTED_POLICY_SEQUENCE_LIMIT")
            return [convert(child, depth+1) for child in item]
        if kind is np.float64:
            item = float(item)
            kind = float
        _require(kind in (str, int, float, bool, type(None)), "EXPECTED_POLICY_SCALAR")
        if kind is float:
            _require(math.isfinite(item), "EXPECTED_POLICY_NONFINITE")
        elif kind is str:
            _require(len(item) <= 4096, "EXPECTED_POLICY_STRING_LIMIT")
        elif kind is int:
            _require(item.bit_length() <= 128, "EXPECTED_POLICY_INTEGER_LIMIT")
        return item
    answer = convert(value)
    _require(len(json.dumps(answer, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 65536,
        "EXPECTED_POLICY_BYTE_LIMIT")
    return answer


def _binding_digest(artifact, retained):
    """Bound the small supplied attestations separately from the large report.

    This is object stability, not independent confirmation of file custody.
    Its 256-node preflight is separate from the two-report traversal counter.
    """
    stack, visits = [(dict(artifact=artifact, retained=retained), 0)], 0
    while stack:
        item, depth = stack.pop()
        visits += 1
        _require(visits <= 256 and depth <= 8, "SUPPLIED_BINDING_TREE_LIMIT")
        kind = type(item)
        if kind is dict:
            _require(len(item) <= 32 and all(type(key) is str and len(key) <= 128 for key in item),
                "SUPPLIED_BINDING_KEYS")
            visits += len(item)
            _require(visits <= 256, "SUPPLIED_BINDING_TREE_LIMIT")
            stack.extend((child, depth+1) for child in item.values())
        else:
            _require(kind in (str, int, bool, type(None)), "SUPPLIED_BINDING_SCALAR")
            _require(kind is not str or len(item) <= 256, "SUPPLIED_BINDING_STRING_LIMIT")
            _require(kind is not int or item.bit_length() <= 128, "SUPPLIED_BINDING_INTEGER_LIMIT")
    blob = json.dumps(dict(artifact=artifact, retained=retained), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()
    _require(len(blob) <= 16384, "SUPPLIED_BINDING_BYTE_LIMIT")
    return dict(bytes=len(blob), sha256=_sha(blob), preflight_nodes=visits)


def _requested(fixture):
    # These are fresh, source-bound constructor dependencies. No comparison with
    # dead-worker cache identities or cache sizes is made or implied.
    for name in _v1.BACKENDS:
        _loaded(name)
    snapshots = []
    for module in (fixture, sys.modules.get(
            "_two_plane_depth_native_v1_fixture_test_cad_guided_connected_v12_native_bytes")):
        _require(module is not None, "DRIVER_PRELOADED_CONSTRUCTOR_MODULE_REQUIRED")
        cache = module._fixture_module
        info = cache.cache_info()
        _require(type(info.currsize) is int and type(info.misses) is int and info.currsize > 0,
            "DRIVER_CONSTRUCTOR_PRELOAD_REQUIRED")
        snapshots.append((cache, info.currsize, info.misses))
    identities = {name: sys.modules["backend."+name] for name in _v1.BACKENDS}
    kwargs = fixture._policy_objects()
    for cache, size, misses in snapshots:
        info = cache.cache_info()
        _require(info.currsize == size and info.misses == misses, "NEW_CONSTRUCTOR_SOURCE_LOAD_DURING_CHECK")
    _require(all(sys.modules.get("backend."+name) is module for name, module in identities.items()),
        "CONSTRUCTOR_MODULE_IDENTITY_CHANGED")
    _require(type(kwargs) is dict, "EXPECTED_REQUEST_MAP")
    wire = _wire_expected({key: asdict(value) if is_dataclass(value) else value for key, value in kwargs.items()})
    return kwargs, _sha(json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _publication(fixture, record, artifact, binding, digest, cad_blob, scan_blob, retained):
    _require(type(record) is dict and record["full_returned_workflow_retained"] is True
        and record["source_consistency"] is True and record.get("execution_error") is None
        and record.get("publication_error") is None and record["status"] == "RETURNED_PENDING_ORACLE"
        and record["phase"] == "RETURNED_PENDING_ORACLE" and type(record["workflow"]) is dict,
        "ORIGINAL_COMPLETE_SOURCE_BOUND_PUBLICATION")
    _require(type(artifact) is dict and set(artifact) == {"filename", "bytes", "sha256"}
        and artifact["filename"] == "CASE_RESULT.json" and type(artifact["bytes"]) is int
        and 1 <= artifact["bytes"] <= MAX_RECORD_BYTES and _hex(artifact["sha256"]), "ORIGINAL_ARTIFACT_DESCRIPTOR")
    keys = {"schema", "original_result_artifact", "canonical_record_before", "input_binding",
        "source_bundle_sha256", "original_terminal_output_sha256", "driver_custody_status",
        "original_accuracy_status", "original_diagnostic_status"}
    _require(type(retained) is dict and set(retained) == keys
        and retained["schema"] == "NATIVE_DEPTH_RETAINED_BINDING_V2"
        and retained["driver_custody_status"] == CUSTODY_STATUS
        and retained["source_bundle_sha256"] == ORIGINAL_SOURCE_MAP_SHA256
        and _hex(retained["original_terminal_output_sha256"])
        and retained["original_accuracy_status"] == retained["original_diagnostic_status"] == "FAIL"
        and _same(retained["original_result_artifact"], artifact)
        and _same(retained["canonical_record_before"], digest)
        and _same(retained["input_binding"], binding), "RETAINED_DRIVER_ATTESTATION")
    _require(_same(record["source_sha256"], fixture.SOURCE_HASHES), "ORIGINAL_FIXTURE_SOURCE_DECLARATION")
    expected = dict(binding, position_error_bound_mm=.05,
        cad_placement_scope="DECLARED_NOMINAL_CAD_FRAME_NOT_SCAN_POSE")
    _require(_same(record["input_binding"], expected), "ORIGINAL_RECORDER_INPUT_BINDING")
    _require(_same(record["input_artifacts"], [dict(filename="CAD.step", bytes=len(cad_blob), sha256=binding["cad_sha256"]),
        dict(filename="SCAN.ply", bytes=len(scan_blob), sha256=binding["scan_sha256"])]), "ORIGINAL_INPUT_PUBLICATIONS")
    decoded_raw, decoded_triangles = decode_retained_ply_v2(scan_blob)
    _require(_sha(decoded_raw.tobytes()) == binding["raw_xyz_sha256"]
        and _sha(decoded_triangles.tobytes()) == binding["triangle_raw_rows_sha256"], "SUPPLIED_ARRAYS_NOT_EXACT_RETAINED_PLY")


def _common(result, binding):
    cad = result["cad_surface_sampling"]
    digests = cad["digests"]
    _require(type(digests) is dict and set(digests) == set(DIGEST_KEYS)|set(DIGEST_TEXT)
        and all(_hex(digests[key]) for key in DIGEST_KEYS)
        and all(type(digests[key]) is str and digests[key] == value for key, value in DIGEST_TEXT.items()),
        "EXACT_SAMPLER_DIGEST_SCHEMA")
    coverage = cad["coverage"]
    counts, areas = coverage["sample_count_by_source_face"], coverage["area_by_source_face"]
    for mapping in (counts, areas):
        _require(type(mapping) is dict and 1 <= len(mapping) <= 1024, "SERIALIZED_FACE_MAP_SIZE")
        for key in mapping:
            _require(type(key) is str and key.isascii() and key.isdecimal() and len(key) <= 4
                and str(int(key)) == key and 0 <= int(key) < 1024, "CANONICAL_SERIALIZED_FACE_ID")
    _require(set(counts) <= set(areas) and all(type(n) is int and 1 <= n <= 32768 for n in counts.values())
        and sum(counts.values()) == 32768 and all(type(area) is float and math.isfinite(area) and area > 0.
            for area in areas.values()), "SERIALIZED_FACE_COUNTS_AND_AREAS")
    # Explicit typed-schema adapter: the old common predicate mistakenly treats
    # textual declarations as hashes. Validate those declarations above, then
    # apply ALL unchanged old common scientific predicates to the hash submap.
    view = dict(result, cad_surface_sampling=dict(cad, digests={key: digests[key] for key in DIGEST_KEYS}))
    _v1._common(view, binding)


def _policy_binding(result, kwargs):
    mapping = (("workflow", "workflow_policy"), ("registration", "registration_policy"),
        ("measurement", "measurement_policy"), ("support", "instance_policy"), ("noisy", "noisy_policy"),
        ("noisy_budgets", "noisy_budgets"), ("cad_patch", "cad_patch_policy"), ("cad_patch_budgets", "cad_patch_budgets"),
        ("search", "search_policy"), ("search_budgets", "search_budgets"),
        ("initialization_budgets", "initialization_budgets"), ("surface_budgets", "surface_budgets"),
        ("registration_budgets", "registration_budgets"))
    for published, requested in mapping:
        _require(_same(result["policies"][published], _wire_expected(asdict(kwargs[requested]))), "SERIALIZED_FIXED_POLICY")
    _require(kwargs["position_error_bound_mm"] == .05 and kwargs["allow_ocp"] is True
        and kwargs["connected_measurement_contexts"] is True and kwargs["two_plane_enabled"] is True
        and result["policies"]["two_plane_enabled"] is True, "FIXED_CALL_POLICY")
    _require(_same(result["policies"]["stage_policies"], _wire_expected(
        _loaded("cad_guided_stage_policies_v1").ablation_stage_policies_v1(connected=True))), "FIXED_STAGE_POLICY")
    _require(_same(result["policies"]["scan_plane_configuration"], _wire_expected(dict(mode=kwargs["scan_plane_mode"],
        seeded_budgets=asdict(kwargs["seeded_plane_budgets"]), composite_budgets=asdict(kwargs["composite_plane_budgets"])))),
        "SERIALIZED_SCAN_PLANE_CONFIGURATION")
    _require(_same(result["policies"]["two_plane_schedule_configuration"],
        _wire_expected(dict(mode=kwargs["two_plane_schedule_mode"], budgets=asdict(kwargs["two_plane_schedule_budgets"])))),
        "SERIALIZED_SCHEDULE_CONFIGURATION")


def _composite(fixture, result, kwargs, raw, triangles):
    envelope = result["registration"]["evidence"]["initialization"]["patch_evidence"]["scan"]
    _require(type(envelope) is dict and set(envelope) == {"format", "schemas", "root"}
        and envelope["format"] == "noisy_mesh_metadata_schema_v6", "SERIALIZED_COMPOSITE_ENVELOPE")
    # Nested dictionaries are represented by ordered schemas and value arrays.
    # Restore only the declared outer envelope order, never the original report.
    view = {key: envelope[key] for key in ("format", "schemas", "root")}
    decoder = _loaded("cad_guided_noisy_mesh_planes_v7").decode_metadata_v7
    meta = decoder(view)  # unchanged bounded grammar and full decoding
    expected = _wire_expected(dict(mode=kwargs["scan_plane_mode"],
        seeded_budgets=asdict(kwargs["seeded_plane_budgets"]), composite_budgets=asdict(kwargs["composite_plane_budgets"])))
    digest, size = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":")).iterencode(view):
        blob = fragment.encode()
        size += len(blob)
        _require(size <= expected["composite_budgets"]["max_metadata_bytes"], "SERIALIZED_COMPOSITE_BYTE_LIMIT")
        digest.update(blob)
    fixture._assert_composite_metadata(result, meta, digest.hexdigest(), expected, raw, triangles)


def _finish(ledger, criteria, work):
    work.boundary()
    _v1._finish(ledger, criteria, work)
    ledger["resources"]["node_limit_scope"] = "TWO_CANONICAL_DECODED_JSON_PREFLIGHTS_NOT_REUSED_VALIDATOR_INTERNAL_WORK"
    bindings = [ledger.get(key) for key in ("supplied_binding_before", "supplied_binding_after")]
    ledger["resources"]["supplied_binding_preflight_nodes"] = (sum(row["preflight_nodes"] for row in bindings)
        if all(type(row) is dict for row in bindings) else None)
    ledger["resources"]["supplied_binding_node_limit_each"] = 256
    _require(len(json.dumps(ledger, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= MAX_LEDGER_BYTES,
        "RETAINED_DIAGNOSTIC_LEDGER_BYTE_LIMIT")
    work.boundary()
    return ledger


def evaluate_retained_diagnostic_v2(fixture_module, record, original_result_artifact,
                                    cad_blob, scan_blob, raw, triangles, retained_binding):
    """Evaluate decoded retained evidence; no original runtime objects are recovered."""

    work = _Work()
    criteria = [_Criterion(name, work) for name in CRITERIA]
    c1, c2, c3, c4, c5, c6, c7, c8 = criteria
    ledger = dict(schema="NATIVE_DEPTH_RETAINED_DIAGNOSTIC_LEDGER_V2", status="NOT_EVALUABLE",
        helper="tests/native_depth_retained_diagnostic_v2.py", accuracy_evaluated=False,
        assurance="SERIALIZATION_RECONSTRUCTED_REPORT_PLUS_SOURCE_ENFORCED_NATIVE_CHECK",
        original_accuracy_status="FAIL", original_diagnostic_status="FAIL",
        original_result_artifact=None,
        original_runtime_types_recovered=False, former_worker_cache_identity_recovered=False,
        native_arrays_reconstructed=False, driver_attestation_checked=False,
        file_identity_reverified=False, inference_calls=0, refit_calls=0, truth_accessed=False,
        observed_bend_count=None, production_use_allowed=False, canonical_record_before=None, canonical_record_after=None,
        inputs_before=None, inputs_after=None, supplied_binding_before=None, supplied_binding_after=None,
        immutable_decoded_evidence_and_supplied_arrays=False, fixed_request_sha256=None,
        denominators={}, complete_no_measurements=False)
    with _deadline(WALL_SECONDS):
        before = c1.check("bounded_canonical_record_before", lambda: _record_digest(record, work))
        inputs = c1.check("supplied_input_shape_hash", lambda: _input_digest(cad_blob, scan_blob, raw, triangles))
        bound_before = c1.check("bounded_supplied_binding_before", lambda: _binding_digest(original_result_artifact, retained_binding))
        ledger.update(canonical_record_before=before, inputs_before=inputs, supplied_binding_before=bound_before)
        try:
            if before is None or inputs is None or bound_before is None:
                return _finish(ledger, criteria, work)
            c1.check("durable_source_object_binding", lambda: _publication(fixture_module, record,
                original_result_artifact, inputs, before, cad_blob, scan_blob, retained_binding))
            if c1.row["failed"]:
                return _finish(ledger, criteria, work)
            ledger["original_result_artifact"] = dict(original_result_artifact)
            ledger["driver_attestation_checked"] = True
            result = record["workflow"]
            requested = c4.check("already_loaded_fixed_request", lambda: _requested(fixture_module))
            if requested is None:
                c2.unavailable("FIXED_REQUEST_UNAVAILABLE")
                return _finish(ledger, criteria, work)
            kwargs, ledger["fixed_request_sha256"] = requested
            c4.check("all_fixed_policy_publications", lambda: _policy_binding(result, kwargs))
            c2.check("raw_native_sampler_common", lambda: _common(result, inputs))
            def report_lists():
                evidence = result["registration"]["evidence"]
                init = evidence["initialization"]
                measured = result["measurements"]
                original = measured["source_context_result"]
                return (evidence, init, measured, original,
                    _rows(result["native_features"]["features"], 64, "FEATURE_LIMIT"),
                    _rows(evidence["measurement_context_indices"], 48, "CONTEXT_LIMIT"),
                    _rows(original["context_windows"], 4096, "WINDOW_LIMIT"),
                    _rows(original["support_records"], 512, "RECORD_LIMIT"),
                    _rows(measured["support_groups"], 512, "GROUP_LIMIT"),
                    _rows(result["predictions"], 512, "PREDICTION_LIMIT"))
            data = c8.check("complete_report_prerequisite_lists", report_lists)
            if data is None:
                for criterion in (c3, c5, c6, c7):
                    criterion.unavailable("COMPLETE_CONTEXT_MEASUREMENT_REPORT_UNAVAILABLE")
                return _finish(ledger, criteria, work)
            evidence, init, measured, original, features, indices, windows, records, groups, predictions = data
            ledger["denominators"] = dict(native_features=len(features), contexts=len(indices),
                windows=len(windows), support_records=len(records), support_groups=len(groups), predictions=len(predictions))
            feature_by_id = {}
            for i, feature in enumerate(features):
                def one_feature(feature=feature):
                    _require(type(feature) is dict and type(feature["feature_id"]) is str
                        and feature["feature_id"] and feature["feature_id"] not in feature_by_id, "FEATURE_IDENTITY")
                    feature_by_id[feature["feature_id"]] = feature
                    _require(all(type(feature[key]) in (int, float) and math.isfinite(feature[key])
                        for key in ("nominal_radius_mm", "nominal_sweep_deg")), "FEATURE_NOMINALS")
                c2.check("feature_"+str(i), one_feature)
            c2.check("native_feature_denominator", lambda: _require(features
                and type(result["native_features"]["expected_roi_count"]) is int
                and result["native_features"]["expected_roi_count"] == len(features), "FEATURE_DENOMINATOR"))
            c3.check("complete_context_header", lambda: _context_header(result, inputs))
            def context_denominator():
                candidates = _rows(evidence["candidates"], 48, "CANDIDATE_LIMIT")
                _rows(init["hypotheses"], 8192, "HYPOTHESIS_LIMIT")
                _rows(init["all_passing_start_groups"], 8192, "EXACT_GROUP_LIMIT")
                _require(indices and all(_integer(i, len(candidates)-1) for i in indices)
                    and len(set(indices)) == len(indices)
                    and len(_rows(init["start_groups"], 48, "START_GROUP_LIMIT")) == len(candidates)
                    and len(_rows(result["registration_array_binding"], 48, "BINDING_LIMIT")) == len(indices)
                    and indices == [i for i, candidate in enumerate(candidates)
                        if candidate.get("conditional_context_support", {}).get("status") == "SUPPORTED_CONTEXT"],
                    "CONTEXT_DENOMINATOR")
            c3.check("context_denominator", context_denominator)
            # Per-context failures never suppress later contexts or criteria.
            for position, index in enumerate(indices):
                c3.check("context_"+str(position), lambda p=position, i=index: _context_row(result, p, i, inputs))
            c4.bulk("composite_all_retained_binding", len(init["hypotheses"])+len(indices),
                lambda: _composite(fixture_module, result, kwargs, raw, triangles))
            def schedule():
                expected = _wire_expected(dict(mode="v2", budgets=asdict(kwargs["two_plane_schedule_budgets"])))
                bound = _loaded("cad_guided_mesh_offline_v12")._two_plane_schedule_binding(result["registration"], expected)
                fixture_module._assert_schedule_metadata(result, expected, _wire_expected(asdict(kwargs["initialization_budgets"])), bound)
            c4.bulk("schedule_all_retained_binding", len(indices), schedule)
            c4.check("context_profile_joint_slots", lambda: _profile(init))
            def cover():
                answer = _loaded("cad_guided_support_cover_measurement_v2").validate_support_cover_publication_v2(
                    measured, raw_points=raw, scan_sha256=inputs["scan_sha256"])
                _require(_same(answer, result["support_cover_publication_validation"]), "RETAINED_COVER_VALIDATOR_BINDING")
                _require(answer.get("cover_plan_revalidated") is True
                    and not any(answer.get(key) is True for key in ("fatal_refusal", "source_refusal", "plan_refusal")),
                    "COMPLETE_COVER_PROOF_REQUIRED")
            c5.bulk("unchanged_complete_cover_validator_no_refit", len(records)+len(groups)+len(windows)+len(predictions), cover)
            window_keys = []
            for i, window in enumerate(windows):
                def one_window(window=window):
                    _require(type(window["context_index"]) is int and window["context_index"] in indices
                        and type(window["feature_id"]) is str and window["feature_id"] in feature_by_id, "WINDOW_IDENTITY")
                    window_keys.append((window["context_index"], window["feature_id"]))
                c6.check("window_"+str(i), one_window)
            c6.check("exact_window_product", lambda: _require(len(windows) == len(indices)*len(features)
                and len(window_keys) == len(set(window_keys)) and set(window_keys)
                == {(i, feature["feature_id"]) for i in indices for feature in features}, "WINDOW_PRODUCT"))
            for label, rows in (("record", records), ("group", groups)):
                for i, row in enumerate(rows):
                    associations = c6.check(label+"_associations_"+str(i),
                        lambda row=row: _rows(row["associations"], 4096, "ASSOCIATION_LIMIT"))
                    if associations is not None:
                        for j, association in enumerate(associations):
                            c6.check(label+"_"+str(i)+"_nominal_"+str(j), lambda a=association, r=row:
                                _association(a, feature_by_id, indices, r.get("measurements") if label == "group" else None))
            c7.check("top_level_exact_cover_projection", lambda: fixture_module._assert_top_level_cover_projection(result))
            for i, group in enumerate(groups):
                c7.check("quantity_"+str(i), lambda group=group: _quantity(group))
            def branch():
                count = sum(group["status"] == "MEASURED_CONDITIONAL" for group in groups)
                _require(result["phase"] == "COMPLETE" and measured["computation_status"] == "COMPLETE"
                    and original["computation_status"] == "COMPLETE"
                    and measured.get("discarded_measurement_prefix") is not True
                    and original.get("discarded_measurement_prefix") is not True
                    and result["status"] == measured["status"] == ("PARTIAL" if count else "UNAVAILABLE")
                    and _same(result["predictions"], measured["predictions"])
                    and _same(result["reasons"], measured["reasons"])
                    and measured["reasons"] == ([] if count else ["NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS"]),
                    "COMPLETE_BRANCH_SEMANTICS")
                _require(measured["observed_bend_count"] is None and measured["production_use_allowed"] is False
                    and measured["conformance"] == "INDETERMINATE"
                    and measured["joint_configuration_status"] == "NOT_ESTABLISHED"
                    and measured["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
                    and measured["context_quantifier"] == "EXISTS_COMPLETED_EXISTING_COVER_CONTEXT"
                    and measured["scan_sha256"] == inputs["scan_sha256"]
                    and measured["raw_points_sha256"] == inputs["raw_xyz_sha256"], "MEASUREMENT_UNKNOWN_SEMANTICS")
                ledger["complete_no_measurements"] = count == 0
            c8.check("workflow_complete_disposition", branch)
            ledger["status"] = "COMPLETE_NO_MEASUREMENTS_INTEGRITY" if ledger["complete_no_measurements"] else "COMPLETE_CONDITIONAL_INTEGRITY"
        except Exception as error:
            c8.check("unexpected_ordinary_exception", lambda: (_ for _ in ()).throw(error))
            ledger["status"] = "FAIL"
        finally:
            after = c1.check("bounded_canonical_record_after", lambda: _record_digest(record, work))
            after_inputs = c1.check("supplied_inputs_after", lambda: _input_digest(cad_blob, scan_blob, raw, triangles))
            bound_after = c1.check("bounded_supplied_binding_after", lambda: _binding_digest(original_result_artifact, retained_binding))
            ledger.update(canonical_record_after=after, inputs_after=after_inputs, supplied_binding_after=bound_after)
            immutable = (before is not None and inputs is not None and bound_before is not None
                and _same(before, after) and _same(inputs, after_inputs) and _same(bound_before, bound_after))
            c1.check("immutable_decoded_evidence_and_supplied_arrays", lambda: _require(immutable, "SUPPLIED_EVIDENCE_MUTATED"))
            ledger["immutable_decoded_evidence_and_supplied_arrays"] = immutable
            if any(c.row["status"] == "FAIL" for c in criteria):
                ledger["status"] = "FAIL" if ledger["fixed_request_sha256"] is not None or ledger["immutable_decoded_evidence_and_supplied_arrays"] is False else "NOT_EVALUABLE"
            if ledger["status"].startswith("COMPLETE") and any(c.row["status"] != "PASS" for c in criteria):
                ledger["status"] = "NOT_EVALUABLE"
            _finish(ledger, criteria, work)
    return _finish(ledger, criteria, work)
