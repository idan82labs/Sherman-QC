"""Same-worker, supplied-object integrity checks, NOT an accuracy oracle.

No file/loader/native/inference/refit call is made. Source-bound fixture caches
and backend modules must already exist; their identity is attested externally by
the frozen worker, not by this helper. Report hashes do not recover omitted CAD
arrays. Common predicates are isolated from the original native byte oracle;
its PARTIAL/nonempty/truth requirements remain in that unchanged accuracy test.
"""

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import signal
import sys
import time

import numpy as np


MAX_LEDGER_BYTES = 32*1024
MAX_RECORD_BYTES = 129*1024**2
MAX_NODES = 8_000_000  # cumulative across both new preflight traversals
MAX_DEPTH = 64
MAX_ERRORS = 8
CPU_SECONDS = 90.
WALL_SECONDS = 120.
CRITERIA = ("publication_source_input", "raw_native_sampler", "context_origin_pose",
    "composite_schedule_profile", "raw_support_cover", "windows_nominals",
    "quantity_carrier_projection", "branch_denominator")
BACKENDS = ("cad_guided_mesh_offline_v13", "cad_guided_mesh_offline_v12",
    "cad_guided_mesh_offline_v2", "cad_guided_plane_initialization_v12",
    "cad_guided_noisy_mesh_planes_v7", "cad_guided_stage_policies_v1",
    "cad_guided_support_cover_measurement_v2", "cad_guided_two_plane_context_v2",
    "cad_guided_two_plane_schedule_v2", "cad_guided_seeded_plane_proposals_v1",
    "cad_guided_scan_plane_preparation_v1", "cad_guided_plane_initialization_v11",
    "cad_guided_global_registration_v1", "cad_guided_global_registration_v3",
    "cad_guided_triangle_surface_v3", "cad_guided_noisy_mesh_planes_v5",
    "cad_guided_mesh_plane_proposals_v4", "cad_guided_plane_search_v6",
    "cad_guided_instances", "cad_guided_native_features")


class _BudgetInterrupt(BaseException):
    """Never converted to a successful or completed diagnostic receipt."""


def _require(value, code):
    if not value:
        raise ValueError(code)


def _same(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return len(a) == len(b) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) in (str, int, float, bool, type(None), np.float64) and a == b


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value, limit):
    return type(value) is int and 0 <= value <= limit


def _rows(value, limit, code):
    _require(type(value) is list and len(value) <= limit, code)
    return value


@contextmanager
def _deadline(seconds):
    """Clip, never extend, an existing process ITIMER_REAL deadline.

    A shorter enclosing timer and its handler are left untouched. A local expiry
    raises BaseException, preserving worker interruption semantics. These timers
    and boundary CPU checks are not an OS allocation/termination guarantee.
    """
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    if old_delay > 0. and old_delay <= seconds:
        yield
        return
    handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    def expired(signum, frame):
        raise _BudgetInterrupt("NATIVE_DIAGNOSTIC_WALL_LIMIT")
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, handler)
        if old_delay > 0.:
            remaining = old_delay-(time.monotonic()-started)
            # An expired parent timer must fire promptly, not become disabled.
            signal.setitimer(signal.ITIMER_REAL, max(1e-9, remaining), old_interval)


class _Work:
    def __init__(self):
        self.cpu = time.process_time()
        self.wall = time.monotonic()
        self.nodes = 0

    def boundary(self):
        if time.process_time()-self.cpu >= CPU_SECONDS:
            raise _BudgetInterrupt("NATIVE_DIAGNOSTIC_CPU_LIMIT")
        if time.monotonic()-self.wall >= WALL_SECONDS:
            raise _BudgetInterrupt("NATIVE_DIAGNOSTIC_WALL_LIMIT")

    def preflight(self, value):
        stack = [(iter((value,)), 0)]
        while stack:
            iterator, depth = stack[-1]
            try:
                item = next(iterator)
            except StopIteration:
                stack.pop()
                continue
            self.nodes += 1
            _require(self.nodes <= MAX_NODES and depth <= MAX_DEPTH, "RECORD_TREE_LIMIT")
            if self.nodes % 4096 == 0:
                self.boundary()
            kind = type(item)
            if kind is dict:
                _require(len(item) <= MAX_NODES-self.nodes, "RECORD_KEY_LIMIT")
                for key in item:
                    _require(type(key) is str, "RECORD_KEY_TYPE")
                    self.nodes += 1
                stack.append((iter(item.values()), depth+1))
            elif kind in (list, tuple):
                _require(len(item) <= MAX_NODES-self.nodes, "RECORD_CONTAINER_LIMIT")
                stack.append((iter(item), depth+1))
            else:
                _require(kind in (str, int, float, bool, type(None), np.float64), "RECORD_SCALAR_TYPE")
                if kind in (float, np.float64):
                    _require(math.isfinite(item), "RECORD_NONFINITE")


def _record_digest(record, work):
    work.preflight(record)
    size, digest = 0, hashlib.sha256()
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(record):
        # Exactly the original serializer, including exact np.float64 values.
        blob = fragment.encode()
        size += len(blob)
        _require(size <= MAX_RECORD_BYTES, "RECORD_BYTE_LIMIT")
        digest.update(blob)
        work.boundary()
    return dict(bytes=size, sha256=digest.hexdigest())


def _input_digest(cad_blob, scan_blob, raw, triangles):
    _require(type(cad_blob) is bytes and 1 <= len(cad_blob) <= 8*1024**2, "CAD_BYTES")
    _require(type(scan_blob) is bytes and 1 <= len(scan_blob) <= 1024**2, "SCAN_BYTES")
    _require(type(raw) is np.ndarray and raw.shape == (650, 3) and raw.dtype.kind in "fiu"
        and np.isfinite(raw).all() and np.all(raw >= -1e6) and np.all(raw <= 1e6), "RAW_ARRAY")
    _require(type(triangles) is np.ndarray and triangles.shape == (1176, 3)
        and triangles.dtype.kind in "iu" and np.min(triangles) >= 0 and np.max(triangles) < len(raw)
        and np.all(triangles[:, 0] != triangles[:, 1]) and np.all(triangles[:, 0] != triangles[:, 2])
        and np.all(triangles[:, 1] != triangles[:, 2]),
        "TRIANGLE_ARRAY")
    return dict(cad_sha256=_sha(cad_blob), scan_sha256=_sha(scan_blob),
        raw_xyz_sha256=_sha(np.asarray(raw, dtype="<f8").tobytes()),
        triangle_raw_rows_sha256=_sha(np.asarray(triangles, dtype="<i8").tobytes()),
        raw_rows=650, triangle_rows=1176)


class _Criterion:
    def __init__(self, name, work):
        self.work = work
        self.row = dict(name=name, status="UNREACHED", prerequisite="NOT_YET_REACHED",
            planned=0, visited=0, failed=0, errors=[], errors_truncated=False,
            counter_unit="SUPPLEMENTAL_PREDICATE_INVOCATIONS", bulk_proofs=[])

    def check(self, label, function):
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
            if len(row["errors"]) < MAX_ERRORS:
                row["errors"].append(dict(check=label[:96], error_type=type(error).__name__[:64]))
            else:
                row["errors_truncated"] = True
            return None

    def unavailable(self, boundary):
        if self.row["status"] == "UNREACHED":
            self.row["prerequisite"] = boundary

    def bulk(self, label, declared_rows, function):
        before = self.row["failed"]
        self.check(label, function)
        complete = self.row["failed"] == before
        # Frozen validators expose no exception-time internal visit count.
        # Unknown is not zero or full traversal. The full criterion fails.
        self.row["bulk_proofs"].append(dict(check=label, declared_rows=declared_rows,
            validator_returned=complete, internally_visited_rows=None,
            all_declared_rows_validated=complete, unreached_rows_upper_bound=0 if complete else declared_rows,
            partial_progress="NOT_INSTRUMENTED"))


def _loaded(name):
    module = sys.modules.get("backend."+name)
    _require(module is not None, "ALREADY_LOADED_SOURCE_MODULE_REQUIRED")
    return module


def _requested(fixture):
    # The fixed fixture loader was called before inference. Never populate its
    # cache here: require both exact prior cache instances and loaded modules.
    for name in BACKENDS:
        _loaded(name)
    prefix = "_two_plane_depth_native_v1_fixture_"
    legacy_name = prefix+"test_cad_guided_connected_v12_native_bytes"
    legacy = sys.modules.get(legacy_name)
    _require(legacy is not None, "PRELOADED_FIXTURE_CACHE_REQUIRED")
    def state(cache, maximum):
        info = cache.cache_info()
        _require(type(info.currsize) is int and type(info.misses) is int and type(info.hits) is int
            and type(info.maxsize) is int and info.maxsize == maximum and info.hits >= 0,
            "INVALID_FIXTURE_CACHE_TELEMETRY")
        return (info.currsize, info.misses, info.hits)
    outer_before, inner_before = state(fixture._fixture_module, 3), state(legacy._fixture_module, 2)
    # Original accuracy may load ARRAY_SOURCE after the original legacy load.
    # No other outer-key history is part of this source-controlled single case.
    _require(outer_before[0] in (1, 2) and outer_before[1] == outer_before[0]
        and inner_before[:2] == (2, 2), "PRELOADED_FIXTURE_CACHE_REQUIRED")
    names = [legacy_name, "_connected_v12_native_fixture_test_cad_guided_mesh_connected_native_v1",
        "_connected_v12_native_fixture_test_cad_guided_connected_v12"]
    if outer_before[0] == 2:
        names.append(prefix+"test_cad_guided_connected_v12")
    identities = {name: sys.modules.get(name) for name in names}
    _require(all(value is not None for value in identities.values()), "PRELOADED_NAMED_FIXTURES_REQUIRED")
    kwargs = fixture._policy_objects()
    outer_after, inner_after = state(fixture._fixture_module, 3), state(legacy._fixture_module, 2)
    _require(outer_after[:2] == outer_before[:2] and inner_after[:2] == inner_before[:2]
        and outer_after[2] >= outer_before[2] and inner_after[2] >= inner_before[2]
        and all(sys.modules.get(name) is value for name, value in identities.items()),
        "POLICY_CONSTRUCTION_CACHE_MISS_OR_RELOAD")
    _require(type(kwargs) is dict, "REQUEST_CONFIGURATION")
    tree = {key: asdict(value) if is_dataclass(value) else value for key, value in kwargs.items()}
    encoded = json.dumps(tree, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _require(len(encoded) <= 65536, "REQUEST_CONFIGURATION_LIMIT")
    return kwargs, _sha(encoded)


def _publication(fixture, record, artifact, binding, digest, cad_blob, scan_blob):
    _require(type(record) is dict and record["full_returned_workflow_retained"] is True
        and record["source_consistency"] is True and record.get("execution_error") is None
        and record.get("publication_error") is None and record["status"] == "RETURNED_PENDING_ORACLE"
        and record["phase"] == "RETURNED_PENDING_ORACLE" and type(record["workflow"]) is dict,
        "FULL_SOURCE_BOUND_PUBLICATION_REQUIRED")
    _require(type(artifact) is dict and artifact.get("filename") == "CASE_RESULT.json"
        and type(artifact.get("bytes")) is int and _hex(artifact.get("sha256"))
        and _same({key: artifact[key] for key in ("bytes", "sha256")}, digest), "RESULT_OBJECT_BINDING")
    _require(_same(record["source_sha256"], fixture.SOURCE_HASHES), "FIXTURE_SOURCE_DECLARATION")
    expected = dict(binding, position_error_bound_mm=.05,
        cad_placement_scope="DECLARED_NOMINAL_CAD_FRAME_NOT_SCAN_POSE")
    _require(_same(record["input_binding"], expected), "RECORDER_INPUT_BINDING")
    _require(_same(record["input_artifacts"], [dict(filename="CAD.step", bytes=len(cad_blob),
        sha256=binding["cad_sha256"]), dict(filename="SCAN.ply", bytes=len(scan_blob),
        sha256=binding["scan_sha256"])]), "INPUT_PUBLICATION_BINDING")


def _common(result, binding):
    _require(result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V13"
        and result["connected_measurement_contexts"] is True
        and result["observed_bend_count"] is None and result["production_use_allowed"] is False
        and result["conformance"] == "INDETERMINATE"
        and result["registration_mode"] == "FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE"
        and result["prediction_frame"] == result["measurement_geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        "WORKFLOW_COMMON_SEMANTICS")
    _require(result["position_error_model"]["scope"] ==
        "PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY",
        "POSITION_ERROR_SCOPE")
    _require(type(result["position_error_model"]["uniform_euclidean_bound_mm"]) is float
        and result["position_error_model"]["uniform_euclidean_bound_mm"] == .05
        and result["position_error_model"]["status"] == "STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        "POSITION_ERROR_DECLARATION")
    _require(result["cad_sha256"] == binding["cad_sha256"], "CAD_BINDING")
    prep, topology = result["input_preparation"], result["scan_triangle_binding"]
    _require(prep["source_bindings"]["scan_sha256"] == binding["scan_sha256"]
        and prep["source_bindings"]["raw_xyz_sha256"] == binding["raw_xyz_sha256"]
        and type(prep["coverage"]["raw_vertex_count"]) is int
        and prep["coverage"]["raw_vertex_count"] == 650, "SCAN_PREPARATION_BINDING")
    _require(_same(topology, dict(status="CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT",
        scope="SAME_VALIDATED_BYTES_AND_ORIGINAL_ROW_IDENTITIES_NOT_MATERIAL_TOPOLOGY", source_sha256=binding["scan_sha256"],
        raw_points_sha256=binding["raw_xyz_sha256"], triangle_raw_rows_sha256=binding["triangle_raw_rows_sha256"],
        raw_vertex_count=650, triangle_count=1176)), "TRIANGLE_BINDING")
    units = result["cad_units"]
    _require(units["target_unit"] == "mm" and type(units["transferred_roots"]) is int
        and units["transferred_roots"] == 1
        and units["target_unit_action"] == "SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER"
        and units["physical_calibration"] == "NOT_ESTABLISHED", "NATIVE_UNITS")
    names = _rows(units["source_unit_names"]["length"], 64, "UNIT_NAMES")
    _require(names and all(type(name) is str and name.upper() in
        ("MM", "MILLIMETRE", "MILLIMETER") for name in names), "SOURCE_UNITS")
    native = result["native_features"]
    _require(native["adapter_version"] == "NATIVE_TRANSITION_ROI_V1"
        and native["candidate_inventory_complete"] is True and native["exhaustive_bend_inventory"] is False
        and native["physical_observed_bend_count"] is None and native["production_use_allowed"] is False,
        "NATIVE_INVENTORY_SEMANTICS")
    cad, sample = result["cad_surface_sampling"], result["triangle_sample_binding"]
    _require(cad["status"] == "AVAILABLE" and type(cad["sampling_policy"]["sample_count"]) is int
        and cad["sampling_policy"]["sample_count"] == 32768
        and type(sample["initialization_sample_count"]) is int and sample["initialization_sample_count"] == 32768
        and sample["oriented_triangle_xyz_sha256"] == cad["digests"]["oriented_triangle_xyz_sha256"],
        "SAMPLER_BINDING")
    for value in cad["digests"].values():
        _require(_hex(value), "CAD_DIGEST")


def _expected_digests(result, binding):
    cad = result["cad_surface_sampling"]["digests"]
    return dict(scan_points_sha256=binding["raw_xyz_sha256"],
        scan_triangle_raw_rows_sha256=binding["triangle_raw_rows_sha256"],
        position_error_bounds_mm_sha256=_sha(np.full(650, .05, dtype="<f8").tobytes()),
        cad_triangles_sha256=cad["oriented_triangle_xyz_sha256"],
        cad_face_ids_sha256=cad["source_face_ids_sha256"],
        cad_source_triangle_ids_sha256=cad["source_triangle_ids_sha256"])


def _policy_binding(result, kwargs):
    for published, requested in (("workflow", "workflow_policy"), ("registration", "registration_policy"),
        ("measurement", "measurement_policy"), ("support", "instance_policy"), ("noisy", "noisy_policy"),
        ("noisy_budgets", "noisy_budgets"), ("cad_patch", "cad_patch_policy"), ("cad_patch_budgets", "cad_patch_budgets"),
        ("search", "search_policy"), ("search_budgets", "search_budgets"),
        ("initialization_budgets", "initialization_budgets"), ("surface_budgets", "surface_budgets"),
        ("registration_budgets", "registration_budgets")):
        _require(_same(result["policies"][published], asdict(kwargs[requested])), "FIXED_POLICY_PUBLICATION")
    _require(kwargs["position_error_bound_mm"] == .05 and kwargs["allow_ocp"] is True
        and kwargs["connected_measurement_contexts"] is True and kwargs["two_plane_enabled"] is True
        and result["policies"]["two_plane_enabled"] is True, "FIXED_CALL_POLICY")
    _require(_same(result["policies"]["stage_policies"],
        _loaded("cad_guided_stage_policies_v1").ablation_stage_policies_v1(connected=True)), "FIXED_STAGE_POLICY")
    _require(_same(result["policies"]["scan_plane_configuration"], dict(mode=kwargs["scan_plane_mode"],
        seeded_budgets=asdict(kwargs["seeded_plane_budgets"]),
        composite_budgets=asdict(kwargs["composite_plane_budgets"]))), "FIXED_SCAN_PLANE_REQUEST")
    _require(_same(result["policies"]["two_plane_schedule_configuration"],
        dict(mode=kwargs["two_plane_schedule_mode"], budgets=asdict(kwargs["two_plane_schedule_budgets"]))),
        "FIXED_TWO_PLANE_SCHEDULE_REQUEST")


def _context_header(result, binding):
    reg = result["registration"]
    evidence, init = reg["evidence"], reg["evidence"]["initialization"]
    _require(reg["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS" and reg["transform_scan_to_cad"] is None
        and reg["schema_version"] == "noisy_plane_initialized_finite_registration_v12"
        and evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V12"
        and init["schema_version"] == "noisy_plane_initialization_prefilter_v12", "REGISTRATION_SCHEMA")
    _require(init["status"] == "COMPLETE" and evidence["candidate_evaluation_complete"] is True
        and evidence["numerical_consumer_complete"] is True and init["numerical_consumer_complete"] is True
        and init["box_screening_complete"] is True and evidence["accepted_registration"] is False
        and evidence["global_pose_uniqueness"] == "NOT_ESTABLISHED"
        and evidence["connected_measurement_contexts"] is True, "CONTEXT_COMPLETENESS")
    _require(init["box_screening_policy"] == "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
        and init["numerical_support_policy"] == _loaded("cad_guided_stage_policies_v1").PROXIMITY
        and evidence["numerical_support_policy"] == "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
        and evidence["distance_model"] == "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
        and evidence["optimizer_objective"] == "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
        and _same(evidence["stage_policies"], _loaded("cad_guided_stage_policies_v1").ablation_stage_policies_v1(connected=True)),
        "CONSUMER_DECLARATIONS")
    expected = _expected_digests(result, binding)
    _require(_same(evidence["input_digests"], expected) and _same(init["input_digests"], expected), "SIX_INPUT_DIGESTS")


def _pose(rotation, translation):
    _require(type(rotation) is list and len(rotation) == 3 and all(type(row) is list and len(row) == 3
        for row in rotation) and type(translation) is list and len(translation) == 3, "POSE_SHAPE")
    _require(all(type(x) in (int, float) and math.isfinite(x)
        for row in [*rotation, translation] for x in row), "POSE_SCALARS")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    _loaded("cad_guided_mesh_offline_v2")._rigid_transform(matrix)
    return _sha(np.asarray([*np.asarray(rotation).ravel(), *translation], dtype="<f8").tobytes())


def _context_row(result, position, index, binding):
    evidence = result["registration"]["evidence"]
    init, candidates = evidence["initialization"], evidence["candidates"]
    _require(_integer(index, len(candidates)-1), "CONTEXT_INDEX")
    candidate = candidates[index]
    _require(type(candidate["index"]) is int and candidate["index"] == index
        and candidate["execution_status"] == "ATTEMPTED"
        and candidate["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"
        and candidate["conditional_context_support"]["accepted_registration"] is False
        and candidate["numerical_consumer_status"] == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
        and all(candidate[key] == evidence[key] for key in
            ("numerical_support_policy", "distance_model", "optimizer_objective")), "CANDIDATE_CONTRACT")
    pose_sha = _pose(candidate["rotation"], candidate["translation"])
    _require(candidate["conditional_context_support"]["pose_sha256"] == pose_sha, "CONTEXT_POSE_DIGEST")
    row, cad = result["registration_array_binding"][position], result["cad_surface_sampling"]
    expected = _expected_digests(result, binding)
    _require(row["raw_scan_points_sha256"] == expected["scan_points_sha256"]
        and row["raw_triangle_rows_sha256"] == expected["scan_triangle_raw_rows_sha256"]
        and row["position_error_bounds_mm_sha256"] == expected["position_error_bounds_mm_sha256"]
        and row["cad_triangles_sha256"] == expected["cad_triangles_sha256"]
        and row["cad_sampler_points_sha256"] == cad["digests"]["sample_points_sha256"]
        and row["registration_canonical_cad_points_sha256"] == evidence["cad_initialization_points_sha256"]
        and row["registration_selection"] == "ALL_RAW_ROWS", "CONTEXT_ARRAY_REPORT_BINDING")
    group = init["start_groups"][index]
    gi = candidate["exact_group_index"]
    _require(_integer(gi, len(init["all_passing_start_groups"])-1), "EXACT_GROUP_INDEX")
    full = init["all_passing_start_groups"][gi]
    _require(type(group["index"]) is int and group["index"] == index, "START_GROUP_INDEX")
    for value in (group, full):
        _require(type(value["exact_group_index"]) is int and value["exact_group_index"] == gi
            and value["refinement_disposition"] == "SCHEDULED_FOR_V3_REFINEMENT"
            and _same(value["hypothesis_indices"], candidate["hypothesis_indices"])
            and _same(value["initialized_rotation"], candidate["initialized_rotation"])
            and _same(value["initialized_translation_mm"], candidate["initialized_translation_mm"]), "START_GROUP_IDENTITY")
    origins = candidate["hypothesis_indices"]
    _require(type(origins) is list and origins and all(_integer(x, len(init["hypotheses"])-1) for x in origins)
        and len(set(origins)) == len(origins), "ORIGIN_INDICES")
    reciprocal = [i for i, hypothesis in enumerate(init["hypotheses"])
        if hypothesis.get("exact_start_group_index") == gi and hypothesis.get("prefilter_status") == "PASSED"]
    _require(origins == reciprocal, "RECIPROCAL_PASSED_ORIGINS")
    for oi in origins:
        origin = init["hypotheses"][oi]
        _require(type(origin["hypothesis_index"]) is int and origin["hypothesis_index"] == oi
            and origin["prefilter_status"] == "PASSED"
            and _same(origin["rotation"], candidate["initialized_rotation"])
            and _same(origin["translation_mm"], candidate["initialized_translation_mm"]), "ORIGIN_POSE")
        count = {"TRIPLET": 3, "TWO_PLANE_CONTEXT": 2}.get(origin["proposal_lane"])
        _require(count is not None and len(origin["assigned_support"])
            == len(origin["correspondences"]) == count, "ORIGIN_ROLE_COUNT")
        for support, correspondence in zip(origin["assigned_support"], origin["correspondences"]):
            _require(support["status"] == "PASSED" and all(type(support[key]) is int
                and type(correspondence[key]) is int and support[key] == correspondence[key]
                for key in ("scan_plane_id", "cad_plane_id", "normal_sign")), "ORIGIN_ROLE_IDENTITY")


def _profile(init):
    joint, schedule = init["joint_proposal_accounting"], init["two_plane_schedule"]
    _require(joint["pair_lane_enabled"] is True and all(_integer(joint[key], maximum) for key, maximum in
        (("triplet_records", 8192), ("triplet_unsigned_reservation", 1024), ("ordered_pair_matches", 256),
         ("pair_slot_reservation", 8192), ("total_hypothesis_records", 8192))), "JOINT_TYPED_COUNTS")
    _require(joint["total_hypothesis_records"] == len(init["hypotheses"])
        == joint["triplet_records"]+joint["pair_slot_reservation"]
        and joint["pair_slot_reservation"] == 32*joint["ordered_pair_matches"], "JOINT_SLOT_DENOMINATOR")
    if joint["ordered_pair_matches"]:
        helper = init["two_plane_context"]
        _loaded("cad_guided_plane_initialization_v12")._validate_two_plane_context_configuration(helper, "v2")
        _require(helper["status"] == "COMPLETE" and helper["selected_policy_evaluation_complete"] is True
            and helper["accepted_registration"] is False and helper["finite_support_match_verified"] is False
            and type(helper["statistics"]["slot_records_completed"]) is int
            and helper["statistics"]["slot_records_completed"] == joint["pair_slot_reservation"]
            and len(schedule["scheduled_matches"]) == joint["ordered_pair_matches"], "CONTEXT_PROFILE_COMPLETION")
        expected = {key+"_sha256": init["input_digests"][name+"_sha256"] for key, name in (
            ("scan_points", "scan_points"), ("scan_triangles", "scan_triangle_raw_rows"),
            ("cad_triangles", "cad_triangles"), ("cad_face_ids", "cad_face_ids"),
            ("cad_source_triangle_ids", "cad_source_triangle_ids"))}
        _require(_same(helper["input_digests"], expected), "CONTEXT_PROFILE_INPUT_DIGESTS")


def _association(row, features, indices, measured=None):
    _require(type(row) is dict and type(row["context_index"]) is int and row["context_index"] in indices
        and type(row["feature_id"]) is str and row["feature_id"] in features, "ASSOCIATION_IDENTITY")
    feature = features[row["feature_id"]]
    _require(type(row["nominal_source"]) is str and row["nominal_source"].startswith("OCP_IMPORT_LOCAL_V1:")
        and type(row["nominal_radius_mm"]) in (int, float) and type(row["nominal_sweep_deg"]) in (int, float)
        and row["nominal_radius_mm"] == feature["nominal_radius_mm"]
        and row["nominal_sweep_deg"] == feature["nominal_sweep_deg"], "NATIVE_NOMINAL_BINDING")
    if measured is not None:
        _require(row["accepted_registration"] is False, "ASSOCIATION_ACCEPTANCE")
        for key, target in (("canonical_radius_deviation_mm", measured["measured_radius_mm"]-row["nominal_radius_mm"]),
            ("canonical_sweep_magnitude_deviation_deg", measured["measured_sweep_magnitude_deg"]-abs(row["nominal_sweep_deg"]))):
            _require(type(row[key]) in (int, float) and math.isfinite(row[key])
                and abs(row[key]-target) <= 1e-12, "CANONICAL_DEVIATION_ALGEBRA")


def _quantity(group):
    if group["status"] != "MEASURED_CONDITIONAL":
        _require(group["status"] == "UNAVAILABLE" and group["measurements"] is None
            and (group["finite_support"] is None or group["finite_support"].get("instance") is None),
            "REFUSED_GROUP_OUTPUT")
        return
    measured = group["measurements"]
    _require(measured["geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
        and measured["status"] == "CONDITIONAL_SCAN_POINT_ESTIMATE"
        and measured["radius_uncertainty_interval_mm"] is None
        and measured["sweep_uncertainty_interval_deg"] is None
        and measured["physical_trim_endpoints"] is None and measured["datum_relative_deviation"] is None,
        "QUANTITY_UNKNOWN_SEMANTICS")
    _require(all(type(measured[key]) in (int, float) and math.isfinite(measured[key]) for key in
        ("measured_radius_mm", "measured_sweep_magnitude_deg", "measured_sweep_deg")), "FINITE_QUANTITIES")


def _finish(ledger, criteria, work):
    ledger["criteria"] = [criterion.row for criterion in criteria]
    ledger["resources"] = dict(cpu_seconds=time.process_time()-work.cpu,
        wall_seconds=time.monotonic()-work.wall, preflight_nodes=work.nodes,
        node_limit_scope="CUMULATIVE_TWO_NEW_TRAVERSALS_NOT_REUSED_VALIDATOR_INTERNAL_WORK",
        supplemental_cpu_limit=CPU_SECONDS, supplemental_wall_limit=WALL_SECONDS)
    size = len(json.dumps(ledger, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    _require(size <= MAX_LEDGER_BYTES, "DIAGNOSTIC_LEDGER_BYTE_LIMIT")
    return ledger


def evaluate_native_diagnostic_v1(fixture_module, record, result_artifact,
                                  cad_blob, scan_blob, raw, triangles, truth):
    """Evaluate supplied published objects once; `truth` is deliberately unused.

    A raised BaseException remains an interruption, not a diagnostic PASS.
    Ordinary late errors are counted, with post-hash verification in finally.
    A caller must separately veto accuracy attribution if mutation is detected.
    """
    work = _Work()
    criteria = [_Criterion(name, work) for name in CRITERIA]
    c1, c2, c3, c4, c5, c6, c7, c8 = criteria
    ledger = dict(schema="NATIVE_DEPTH_DIAGNOSTIC_LEDGER_V1", status="NOT_EVALUABLE",
        helper="tests/native_depth_diagnostic_ledger_v1.py", accuracy_evaluated=False,
        assurance="REPORT_BINDING_PLUS_SOURCE_ENFORCED_RUNTIME_CHECK",
        file_identity_reverified=False, inference_calls=0, refit_calls=0, truth_accessed=False,
        observed_bend_count=None, production_use_allowed=False, record_before=None, record_after=None,
        inputs_before=None, inputs_after=None, immutable_supplied_evidence=False, fixed_request_sha256=None,
        denominators={}, complete_no_measurements=False)
    with _deadline(WALL_SECONDS):
        before = c1.check("bounded_record_before", lambda: _record_digest(record, work))
        inputs = c1.check("supplied_input_shape_hash", lambda: _input_digest(cad_blob, scan_blob, raw, triangles))
        ledger.update(record_before=before, inputs_before=inputs)
        try:
            if before is None or inputs is None:
                return _finish(ledger, criteria, work)
            c1.check("durable_source_object_binding", lambda: _publication(fixture_module, record,
                result_artifact, inputs, before, cad_blob, scan_blob))
            if c1.row["failed"]:
                return _finish(ledger, criteria, work)
            result = record["workflow"]
            requested = c4.check("already_loaded_fixed_request", lambda: _requested(fixture_module))
            if requested is None:
                c2.unavailable("FIXED_REQUEST_UNAVAILABLE")
                return _finish(ledger, criteria, work)
            kwargs, ledger["fixed_request_sha256"] = requested
            c4.check("all_fixed_policy_publications", lambda: _policy_binding(result, kwargs))
            # An exact input-preparation refusal may establish safe non-output,
            # not that the parser's scientific/format diagnosis is independently
            # correct. Valid fixed inputs/policies cannot justify ADMISSION
            # refusal by copying a reason string; other early phases stay unknown.
            if result.get("phase") == "INPUT_PREPARATION" and result.get("input_preparation", {}).get("status") == "UNAVAILABLE":
                def early():
                    prepared = result["input_preparation"]
                    reasons = _rows(prepared["reasons"], 64, "EARLY_REASON_LIMIT")
                    base = dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status="UNAVAILABLE", phase="INPUT_PREPARATION",
                        predictions=[], measurements=None, observed_bend_count=None, production_use_allowed=False,
                        conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED",
                        units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                        registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE", datum_status="NO_VERIFIED_STABLE_DATUM",
                        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
                        registration_initialization="TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V12",
                        measurement_geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
                        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
                        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE", connected_measurement_contexts=True,
                        position_error_model=dict(uniform_euclidean_bound_mm=.05,
                            status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
                            scope="PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY"))
                    # Exact V13 pre-parser publication shape: later-stage keys
                    # are ABSENT, not null aliases. In particular no stale
                    # registration_array_binding, native or sampling report.
                    _require(set(result) == set(base)|{"policies", "input_preparation", "reasons"}
                        and all(_same(result[key], value) for key, value in base.items()), "EARLY_EXACT_WORKFLOW_FIELDS")
                    expected_prep = dict(status="UNAVAILABLE", reasons=reasons, source_bindings={}, coverage=None,
                        measurement_cloud_contract="FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING",
                        sample_cloud_contract="REGISTRATION_AND_PROPOSALS_ONLY", units_status="UNVERIFIED",
                        coordinate_frame_status="UNVERIFIED")
                    _require(_same(prepared, expected_prep) and reasons
                        and all(type(reason) is str and len(reason) <= 256 for reason in reasons)
                        and _same(result["reasons"], ["MESH_INPUT_UNAVAILABLE", *reasons]), "SAFE_INPUT_REFUSAL_PATH")
                c8.check("input_preparation_refusal_no_output", early)
                if not c8.row["failed"]:
                    ledger["status"] = "SAFE_EARLY_REFUSAL_LIMITED"
                    for criterion in (c2, c3, c5, c6, c7):
                        criterion.unavailable("NOT_APPLICABLE_BY_VALID_REFUSAL:INPUT_PREPARATION")
                    ledger["early_refusal_scope"] = "NO_DOWNSTREAM_OUTPUT_ONLY_PARSER_DIAGNOSIS_NOT_RECOMPUTED"
                return _finish(ledger, criteria, work)
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
                lambda: fixture_module._assert_composite_binding(result, raw, triangles))
            def schedule():
                expected = dict(mode="v2", budgets=asdict(kwargs["two_plane_schedule_budgets"]))
                bound = _loaded("cad_guided_mesh_offline_v12")._two_plane_schedule_binding(result["registration"], expected)
                fixture_module._assert_schedule_metadata(result, expected, asdict(kwargs["initialization_budgets"]), bound)
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
            after = c1.check("bounded_record_after", lambda: _record_digest(record, work))
            after_inputs = c1.check("supplied_inputs_after", lambda: _input_digest(cad_blob, scan_blob, raw, triangles))
            ledger.update(record_after=after, inputs_after=after_inputs)
            c1.check("immutable_supplied_evidence", lambda: _require(before is not None and inputs is not None
                and _same(before, after) and _same(inputs, after_inputs), "SUPPLIED_EVIDENCE_MUTATED"))
            ledger["immutable_supplied_evidence"] = before is not None and inputs is not None and _same(before, after) and _same(inputs, after_inputs)
            if any(c.row["status"] == "FAIL" for c in criteria):
                ledger["status"] = "FAIL" if ledger["fixed_request_sha256"] is not None or ledger["immutable_supplied_evidence"] is False else "NOT_EVALUABLE"
            if ledger["status"].startswith("COMPLETE") and any(c.row["status"] != "PASS" for c in criteria):
                ledger["status"] = "NOT_EVALUABLE"
            _finish(ledger, criteria, work)
    return _finish(ledger, criteria, work)
