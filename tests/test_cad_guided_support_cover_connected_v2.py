"""Prospective paired support-cover packet; NO execution release in this file.

The four curved geometries, CAD declarations, poses, noise, numerical policies
and measurement/carrier oracles are loaded from the exact frozen V12 fixture.
One real registration per curved case supplies identical generated contexts to
two separately executed measurement arms. This isolates aggregation; it is NOT
two independent registration replications. V2 internally recomputes V1, and its
complete nested result must equal the separately evaluated V1 arm exactly.

The two planar cases restore their original PREFILTER-ONLY endpoint. They do
not validate whole-part admission, accepted registration or curved measurement.
Every stage/outcome survives until the final aggregate assertion. Native/byte
adapters, physical inputs, calibration and manufacturing counts remain outside
this constructed qualification.
"""

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = "tests/test_cad_guided_connected_v12.py"
FROZEN_FIXTURE_SHA256 = "fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd"
FROZEN_FIXTURE_MAX_BYTES = 512*1024
OPT_IN = "SHERMAN_RUN_SUPPORT_COVER_CONNECTED_V2"
ARTIFACT_ENV = "SHERMAN_SUPPORT_COVER_CONNECTED_V2_ARTIFACT_DIRECTORY"
PLANAR_CASE_IDS = ("P-clean", "P-noisy")
CURVED_CASE_IDS = ("U-R", "U-A", "L-R", "L-A")
PLANAR_ARMS = ("V11", "V12")
MEASUREMENT_ARMS = ("V1", "V2")
RESOURCE_LIMITS = dict(wall_seconds=1200, work_wall_seconds=1195,
    worker_cpu_seconds=900, sampled_process_group_rss_bytes=4*1024**3,
    stdout_bytes=1024**2, execution_enforcement="EXTERNAL_SUPERVISOR_ONLY_NO_PER_ARM_KILLING")
RESULT_CAPS = dict(PLANAR=8*1024**2, REGISTRATION=16*1024**2, MEASUREMENT=32*1024**2)
START_CAP = 4096
TOTAL_RESULT_CAP = 352*1024**2
TOTAL_START_CAP = 16*START_CAP
ROLE_FINGERPRINT_CAP = 16*1024**2
STAGE_IDS = tuple(("PLANAR", index, arm) for index in range(2) for arm in PLANAR_ARMS) + tuple(
    stage for index in range(4) for stage in
    (("REGISTRATION", index, "SHARED"), ("MEASUREMENT", index, "V1"), ("MEASUREMENT", index, "V2")))
ARTIFACT_SPECS = {
    f"{kind}_{phase}_{index:02d}_{arm}.json": START_CAP if phase == "START" else RESULT_CAPS[kind]
    for kind, index, arm in STAGE_IDS for phase in ("START", "RESULT")}


def _load_frozen_fixture():
    """Hash-bound source-only test module; no tests/backend functions invoked."""
    path = ROOT/FROZEN_FIXTURE_PATH
    with path.open("rb") as stream:
        blob = stream.read(FROZEN_FIXTURE_MAX_BYTES+1)
    assert len(blob) <= FROZEN_FIXTURE_MAX_BYTES
    assert hashlib.sha256(blob).hexdigest() == FROZEN_FIXTURE_SHA256
    name = "_support_cover_v2_exact_frozen_v12_fixture"
    previous = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclass type inspection.
    try:
        # Execute the bytes just authenticated, not a second path read.
        exec(compile(blob, str(path), "exec"), module.__dict__)
        return module
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous


@pytest.fixture(scope="module")
def frozen():
    return _load_frozen_fixture()


def _encode(value, maximum):
    pieces, size = [], 0
    for fragment in json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":")).iterencode(value):
        block = fragment.encode("utf-8")
        size += len(block)
        if size > maximum:
            raise ValueError("PAIRED_PACKET_PUBLICATION_BYTE_LIMIT")
        pieces.append(block)
    return b"".join(pieces)


def _json_sha(value, maximum):
    return hashlib.sha256(_encode(value, maximum)).hexdigest()


def _error(error):
    return dict(type=type(error).__name__, message=str(error)[:1000])


def _immutable_inputs(inputs):
    return {name: np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype).reshape(value.shape)
        if type(value) is np.ndarray else value for name, value in inputs.items()}


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _owned_role_fingerprint(owned):
    if owned is None:
        return None
    arrays = [owned._triangles, owned._referenced, *[row.triangle_rows for row in owned._supports]]
    assert all(type(value) is np.ndarray and not value.flags.writeable for value in arrays)
    evidence = {name: getattr(owned, name) for name in (
        "points_sha256", "triangle_raw_rows_sha256", "position_error_bounds_mm_sha256", "metadata_sha256",
        "raw_row_count", "triangle_count", "triangle_membership_count", "raw_membership_count",
        "encoded_metadata_bytes", "decoded_metadata_bytes", "unscheduled_qualified_core_count")}
    evidence.update(budgets=asdict(owned._budgets), triangles_sha256=_array_sha(owned._triangles),
        referenced_sha256=_array_sha(owned._referenced), supports=[dict(
            kind=row.kind, record_id=row.record_id, triangle_rows_sha256=_array_sha(row.triangle_rows),
            declared_triangle_rows_sha256=row.triangle_rows_sha256,
            raw_vertex_rows_sha256=row.raw_vertex_rows_sha256, raw_support_count=row.raw_support_count,
            normal=list(row.normal), angle_bound_radians=row.angle_bound_radians) for row in owned._supports])
    return dict(sha256=_json_sha(evidence, ROLE_FINGERPRINT_CAP), evidence=evidence)


def _write_record(directory, filename, value, maximum):
    blob = _encode(value, maximum)  # Whole encoding before creating a prefix.
    temporary = directory/(filename+".partial")
    with temporary.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(directory/filename)
    return dict(filename=filename, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


class _Recorder:
    """Complete fixed-stage ledger; CPU/wall intervals are observed, not caps."""

    def __init__(self, directory, *, emit_progress=False):
        self.directory, self.entries = directory, []
        self.result_bytes = self.start_bytes = 0
        self.emit_progress = emit_progress

    def stage(self, kind, index, arm, case_id, binding, call, oracle, *, required_positive=True, safety=None):
        key = (kind, index, arm)
        assert key == STAGE_IDS[len(self.entries)]
        start = dict(kind=kind, case_index=index, case_id=case_id, arm=arm,
            disposition="STARTED", binding=binding,
            monotonic_start_ns=time.monotonic_ns(), process_cpu_start_ns=time.process_time_ns())
        receipt = _write_record(self.directory, f"{kind}_START_{index:02d}_{arm}.json", start, START_CAP)
        self.start_bytes += receipt["bytes"]
        assert self.start_bytes <= TOTAL_START_CAP
        answer = auxiliary = None
        record = dict(start, disposition="COMPLETED_STAGE", answer=None,
            execution_error=None, oracle_error=None, oracle_passed=False,
            safety_error=None, required_positive=required_positive)
        wall, cpu = time.monotonic_ns(), time.process_time_ns()
        try:
            answer, auxiliary = call()
            record["answer"] = answer
        except Exception as error:
            record["execution_error"] = _error(error)
        finally:
            record["timing"] = dict(call_wall_seconds=(time.monotonic_ns()-wall)/1e9,
                call_cpu_seconds=(time.process_time_ns()-cpu)/1e9,
                scope="THIS_STAGE_CALL_ONLY_NOT_SHARED_REGISTRATION_CHARGED_AGAIN")
        if answer is not None:
            wall, cpu = time.monotonic_ns(), time.process_time_ns()
            if safety is not None:
                try:
                    record["safety"] = safety(answer)
                except Exception as error:
                    record["safety_error"] = _error(error)
            try:
                record["oracle"] = oracle(answer)
                record["oracle_passed"] = True
            except Exception as error:
                record["oracle_error"] = _error(error)
            finally:
                record["oracle_timing"] = dict(wall_seconds=(time.monotonic_ns()-wall)/1e9,
                    cpu_seconds=(time.process_time_ns()-cpu)/1e9)
        filename = f"{kind}_RESULT_{index:02d}_{arm}.json"
        try:
            receipt = _write_record(self.directory, filename, record, RESULT_CAPS[kind])
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            record = dict(kind=kind, case_index=index, case_id=case_id, arm=arm,
                disposition="RESULT_RETENTION_UNAVAILABLE", binding=binding,
                answer=None, discarded_answer_whole=True, oracle_passed=False,
                required_positive=required_positive, publication_error=_error(error), timing=record["timing"])
            receipt = _write_record(self.directory, filename, record, START_CAP)
        self.result_bytes += receipt["bytes"]
        assert self.result_bytes <= TOTAL_RESULT_CAP
        self.entries.append(dict(kind=kind, case_index=index, case_id=case_id, arm=arm,
            disposition=record["disposition"], oracle_passed=record["oracle_passed"],
            required_positive=required_positive, safety_error=record.get("safety_error"),
            execution_error=record.get("execution_error"), oracle_error=record.get("oracle_error"),
            publication_error=record.get("publication_error"), timing=record["timing"], artifact=receipt))
        if self.emit_progress:
            print(json.dumps(dict(kind=kind, case_id=case_id, arm=arm,
                disposition=record["disposition"], oracle_passed=record["oracle_passed"],
                execution_error_type=(record.get("execution_error") or {}).get("type"),
                safety_error_type=(record.get("safety_error") or {}).get("type"),
                oracle_error_type=(record.get("oracle_error") or {}).get("type"),
                call_cpu_seconds=record["timing"]["call_cpu_seconds"],
                call_wall_seconds=record["timing"]["call_wall_seconds"],
                retained_artifact_filename=receipt["filename"]), sort_keys=True, allow_nan=False), flush=True)
        return answer, auxiliary, self.entries[-1]


def _planar_call(frozen, inputs, arm):
    module, policies, _ = frozen._policy_objects("C0" if arm == "V11" else "C1", planar=True)
    function = (module.prefilter_plane_initializations_v11 if arm == "V11"
                else module.prefilter_plane_initializations_v12)
    answer = function(inputs["scan_points"], inputs["scan_triangles"], inputs["cad_triangles"],
        inputs["cad_face_ids"], inputs["cad_source_triangle_ids"],
        position_error_bounds_mm=inputs["position_error_bounds_mm"],
        registration_policy=policies["registration_policy"], unit_assumption=policies["unit_assumption"],
        search_budgets=policies["search_budgets"])
    return dict(prefilter=answer, endpoint="ORIGINAL_FIFTY_ROW_PREFILTER_ONLY",
        policy=dict(registration=asdict(policies["registration_policy"]),
                    search_budgets=asdict(policies["search_budgets"])),
        cad_initialization_points_supplied=False, accepted_registration=False), None


def _planar_oracle(answer, truth, *, require_positive):
    value = answer["prefilter"]
    assert answer["accepted_registration"] is False
    assert answer["cad_initialization_points_supplied"] is False
    assert value["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    positive = value.get("status") == "COMPLETE" and bool(value.get("start_groups"))
    if require_positive:
        assert positive, value.get("reasons")
    if not positive:
        return dict(positive=False, retained_baseline_refusal=True)
    assert value["selected_policy_evaluation_complete"] is True
    assert value["numerical_consumer_complete"] is True
    errors = [float(np.linalg.norm(truth["clean_scan_points"]@np.asarray(row["initialized_rotation"]).T
        +np.asarray(row["initialized_translation_mm"])-truth["nominal"], axis=1).max())
        for row in value["start_groups"]]
    assert min(errors) <= 1e-7
    assert value["joint_proposal_accounting"]["triplet_records"] == 0
    assert value["joint_proposal_accounting"]["ordered_pair_matches"] == 2
    assert value["joint_proposal_accounting"]["pair_slot_reservation"] == 64
    assert len(value["hypotheses"]) == 64
    return dict(positive=True, minimum_generated_start_action_error_mm=min(errors),
        curved_measurement_claim=False, whole_part_registration_claim=False)


def _shared_registration(frozen, inputs):
    module, policies, instance_policy = frozen._policy_objects("C2", planar=False)
    cad_points = np.unique(inputs["cad_triangles"].reshape(-1, 3), axis=0)
    cad_points = np.frombuffer(cad_points.tobytes(), dtype=cad_points.dtype).reshape(cad_points.shape)
    assert len(cad_points) >= 36, "CURVED_CAD_SAMPLE_ADMISSION_REGRESSION"
    initial_input_digests = frozen._array_digests(inputs)
    snapshot = {key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
                for key, value in policies.items()}
    registration, owned = module.register_measurement_contexts_v12(
        inputs["scan_points"], inputs["scan_triangles"], inputs["cad_triangles"],
        inputs["cad_face_ids"], inputs["cad_source_triangle_ids"],
        position_error_bounds_mm=inputs["position_error_bounds_mm"], cad_initialization_points=cad_points,
        **policies, connected_measurement_contexts=True)
    answer = dict(registration=registration, registration_sha256=None,
        owned_role_binding=None, policy_snapshot=snapshot, binding_error=None,
        cad_initialization_point_count=len(cad_points), cad_initialization_points_sha256=_array_sha(cad_points),
        cad_initialization_point_contract="UNIQUE_DECLARED_CAD_TRIANGLE_VERTICES_NOT_NATIVE_SAMPLER",
        comparison_contract="ONE_SHARED_REAL_REGISTRATION_SAME_GENERATED_CONTEXTS_NOT_REPLICATIONS")
    auxiliary = dict(owned=owned, instance_policy=instance_policy)
    try:
        assert frozen._array_digests(inputs) == initial_input_digests, "SHARED_REGISTRATION_INPUT_MUTATED"
        answer["registration_sha256"] = _json_sha(registration, RESULT_CAPS["REGISTRATION"])
        answer["owned_role_binding"] = _owned_role_fingerprint(owned)
    except Exception as error:
        # Keep the actual returned registration even if later binding fails;
        # no arm may use unbound shared evidence.
        answer["binding_error"] = _error(error)
        auxiliary = None
    return answer, auxiliary


def _registration_oracle(answer):
    assert answer["binding_error"] is None
    value = answer["registration"]
    assert value["transform_scan_to_cad"] is None
    assert value["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS", value.get("reasons")
    evidence = value["evidence"]
    assert evidence["accepted_registration"] is False
    assert evidence["candidate_evaluation_complete"] is True
    assert evidence["numerical_consumer_complete"] is True
    assert evidence["measurement_context_indices"]
    assert evidence["joint_configuration_status"] == "NOT_ESTABLISHED"
    assert evidence["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
    return dict(completed_generated_contexts=len(evidence["measurement_context_indices"]),
                accepted_registration=False)


def _measurement_call(frozen, inputs, shared, auxiliary, arm, standalone_v1):
    if shared is None or auxiliary is None:
        raise ValueError("NO_COMPLETED_SHARED_REGISTRATION_NO_MEASUREMENT_ATTEMPT")
    from backend.cad_guided_context_measurement_v1 import (
        inspect_generated_context_windows_v1, validate_context_measurement_publication_v1, CADGuidedFeature)
    from backend.cad_guided_support_cover_measurement_v2 import (
        inspect_support_cover_context_windows_v2, validate_support_cover_publication_v2)

    owned, policy = auxiliary["owned"], auxiliary["instance_policy"]
    before = _owned_role_fingerprint(owned)
    assert before == shared["owned_role_binding"], "SHARED_ROLE_CUSTODY_CHANGED"
    registration_blob = _encode(shared["registration"], RESULT_CAPS["REGISTRATION"])
    assert hashlib.sha256(registration_blob).hexdigest() == shared["registration_sha256"]
    registration = json.loads(registration_blob)
    points = inputs["scan_points"]
    scan_sha = _array_sha(points)
    features = (CADGuidedFeature(**frozen.nominal_feature_kwargs()),)
    call = inspect_generated_context_windows_v1 if arm == "V1" else inspect_support_cover_context_windows_v2
    measured = call(points, features, registration=registration, scan_sha256=scan_sha,
                    instance_policy=policy, surface_roles=owned)
    answer = dict(measurements=measured, shared_registration_sha256=shared["registration_sha256"],
        shared_owned_roles_sha256=None if before is None else before["sha256"],
        independent_registration_replication=False, scan_sha256=scan_sha,
        scan_digest_scope="ARRAY_BYTES_ONLY_NOT_SCAN_FILE_AUTHENTICATION", publication_error=None,
        publication_validator_passed=False, input_custody_error=None)
    try:
        assert _encode(registration, RESULT_CAPS["REGISTRATION"]) == registration_blob
        assert _owned_role_fingerprint(owned) == before
        assert _array_sha(points) == scan_sha
    except Exception as error:
        answer["input_custody_error"] = _error(error)
    try:
        if arm == "V1":
            answer["publication_validation"] = validate_context_measurement_publication_v1(measured,
                raw_count=len(points), scan_sha256=scan_sha, raw_points_sha256=scan_sha)
        else:
            answer["publication_validation"] = validate_support_cover_publication_v2(measured,
                raw_points=points, scan_sha256=scan_sha)
        answer["publication_validator_passed"] = True
    except Exception as error:
        answer["publication_error"] = _error(error)
    if arm == "V2":
        nested = measured.get("source_context_result")
        baseline = None if standalone_v1 is None else standalone_v1.get("measurements")
        answer["nested_v1_exact_pair"] = dict(standalone_available=baseline is not None,
            nested_available=nested is not None, exact_equal=False, comparison_error=None)
        try:
            answer["nested_v1_exact_pair"]["exact_equal"] = (baseline is not None and nested is not None
                and _encode(nested, RESULT_CAPS["MEASUREMENT"]) == _encode(baseline, RESULT_CAPS["MEASUREMENT"]))
        except Exception as error:
            # Preserve both actual reports on comparison failure; no prefix or
            # hash-only equality is promoted to a scientific pass.
            answer["nested_v1_exact_pair"]["comparison_error"] = _error(error)
    return answer, None


def _measurement_safety(answer, arm):
    assert answer["publication_validator_passed"] is True, answer["publication_error"]
    assert answer["input_custody_error"] is None
    value = answer["measurements"]
    assert value["observed_bend_count"] is None and value["production_use_allowed"] is False
    assert value["conformance"] == "INDETERMINATE"
    assert value["joint_configuration_status"] == "NOT_ESTABLISHED"
    assert value["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
    if arm == "V2":
        assert answer["nested_v1_exact_pair"]["exact_equal"] is True
        assert value["source_context_result_role"] == "RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS"
        assert value["context_quantifier"] == "EXISTS_COMPLETED_EXISTING_COVER_CONTEXT"
    return dict(publication_validated=True, exact_shared_input_custody=True,
                manufacturing_count_claim=False)


def _measurement_oracle(frozen, answer, truth, arm):
    """Identical positive criteria for both arms; baseline negatives stay evidence."""
    value = answer["measurements"]
    records = value["support_records"] if arm == "V1" else value["support_groups"]
    usable = [row for row in records if row["status"] == "MEASURED_CONDITIONAL"]
    assert usable, value.get("reasons") or "NO_PUBLISHED_CANONICAL_MEASUREMENTS"
    errors = [frozen.assert_scan_derived_measurement(row["measurements"], truth) for row in usable]
    for row in usable:
        measured = row["measurements"]
        assert measured["geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
        assert measured["radius_uncertainty_interval_mm"] is None
        assert measured["sweep_uncertainty_interval_deg"] is None
        assert measured["datum_relative_deviation"] is None
        for association in row["associations"]:
            assert association["accepted_registration"] is False
            assert abs(association["canonical_radius_deviation_mm"]
                       -(measured["measured_radius_mm"]-association["nominal_radius_mm"])) <= 1e-12
            assert abs(association["canonical_sweep_magnitude_deviation_deg"]
                       -(measured["measured_sweep_magnitude_deg"]-abs(association["nominal_sweep_deg"]))) <= 1e-12
    carrier_errors = []
    if truth["case"].localized:
        assert value["predictions"], "MISSING_REQUIRED_LOCALIZED_RAW_CARRIER"
        carrier_errors = [frozen.assert_localized_raw_carrier(row, truth) for row in value["predictions"]]
    return dict(published_measurements=len(usable), measurement_errors=errors, carrier_errors=carrier_errors,
                manufacturing_count_claim=False)


def run_support_cover_packet(frozen, artifact_directory):
    assert os.environ.get(OPT_IN) == "1", "SUPPORT_COVER_PACKET_REQUIRES_EXPLICIT_OPT_IN"
    directory = Path(artifact_directory)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    directory.mkdir(exist_ok=False)  # Supervisor must not pre-create it.
    recorder = _Recorder(directory, emit_progress=True)
    comparisons = []
    for index, case_id in enumerate(PLANAR_CASE_IDS):
        original, truth = frozen._planar_packet(case_id == "P-noisy")
        inputs = _immutable_inputs(original)
        binding = dict(input_digests=frozen._array_digests(inputs),
            fixture_sha256=FROZEN_FIXTURE_SHA256, scope="ORIGINAL_PREFILTER_ONLY_ATTRIBUTION")
        for arm in PLANAR_ARMS:
            require_positive = arm == "V12" or case_id == "P-clean"
            recorder.stage("PLANAR", index, arm, case_id, binding,
                lambda arm=arm: _planar_call(frozen, inputs, arm),
                lambda answer, arm=arm: _planar_oracle(answer, truth,
                    require_positive=arm == "V12" or case_id == "P-clean"), required_positive=require_positive)
    assert tuple(case.case_id for case in frozen.CURVED_CASES) == CURVED_CASE_IDS
    for index, case in enumerate(frozen.CURVED_CASES):
        original, truth = frozen.curved_packet(case)
        inputs = _immutable_inputs(original)
        input_digests = frozen._array_digests(inputs)
        binding = dict(input_digests=input_digests, fixture_sha256=FROZEN_FIXTURE_SHA256,
            scope="UNCHANGED_CURVED_ARRAYS_ONE_PAIRED_REGISTRATION")
        shared, auxiliary, registration_receipt = recorder.stage("REGISTRATION", index, "SHARED",
            case.case_id, binding, lambda: _shared_registration(frozen, inputs), _registration_oracle)
        if registration_receipt["disposition"] != "COMPLETED_STAGE":
            shared = auxiliary = None  # No unretained shared evidence may be used.
        paired_binding = dict(binding, shared_registration_artifact=registration_receipt["artifact"],
            shared_registration_sha256=None if shared is None else shared["registration_sha256"],
            registration_recomputed_for_measurement_arms=False)
        baseline = challenger = None
        arm_entries = {}
        for arm in MEASUREMENT_ARMS:
            answer, _, entry = recorder.stage("MEASUREMENT", index, arm, case.case_id, paired_binding,
                lambda arm=arm: _measurement_call(frozen, inputs, shared, auxiliary, arm, baseline),
                lambda answer, arm=arm: _measurement_oracle(frozen, answer, truth, arm),
                required_positive=arm == "V2", safety=lambda answer, arm=arm: _measurement_safety(answer, arm))
            arm_entries[arm] = entry
            if arm == "V1":
                baseline = answer
            else:
                challenger = answer
        assert frozen._array_digests(inputs) == input_digests
        baseline_value = None if baseline is None else baseline.get("measurements")
        challenger_value = None if challenger is None else challenger.get("measurements")
        baseline_count = (None if baseline_value is None else sum(
            row.get("status") == "MEASURED_CONDITIONAL" for row in baseline_value.get("support_records", [])))
        challenger_count = (None if challenger_value is None else sum(
            row.get("status") == "MEASURED_CONDITIONAL" for row in challenger_value.get("support_groups", [])))
        fully_positive = {arm: entry["disposition"] == "COMPLETED_STAGE"
            and entry["execution_error"] is None and entry["safety_error"] is None and entry["oracle_passed"]
            for arm, entry in arm_entries.items()}
        comparisons.append(dict(case_id=case.case_id, baseline_published_measurements=baseline_count,
            challenger_published_measurements=challenger_count,
            baseline_fully_positive=fully_positive["V1"], challenger_fully_positive=fully_positive["V2"],
            comparison_scope="SAME_GENERATED_CONTEXTS_CONSTRUCTED_DIAGNOSTIC_NOT_GENERALIZATION"))
    assert tuple((row["kind"], row["case_index"], row["arm"]) for row in recorder.entries) == STAGE_IDS
    passed = all(row["disposition"] == "COMPLETED_STAGE" and row["execution_error"] is None
        and row["safety_error"] is None and (row["oracle_passed"] or not row["required_positive"])
        for row in recorder.entries)
    positive_counts = dict(V1=sum(row["baseline_fully_positive"] for row in comparisons),
                           V2=sum(row["challenger_fully_positive"] for row in comparisons))
    improved = positive_counts["V2"] > positive_counts["V1"]
    return dict(schema="SUPPORT_COVER_PAIRED_CONNECTED_ARRAY_PACKET_V2", status="PASS" if passed and improved else "FAIL",
        science_outcome_count=12, shared_registration_count=4, stage_count=16, entries=recorder.entries,
        paired_comparisons=comparisons, fully_positive_case_counts=positive_counts,
        at_least_one_more_fully_positive_case=improved,
        result_bytes=recorder.result_bytes, start_bytes=recorder.start_bytes,
        resource_limits=RESOURCE_LIMITS, result_caps=RESULT_CAPS, whole_result_cap=TOTAL_RESULT_CAP,
        comparison="PAIRED_SAME_GENERATED_CONTEXTS_NOT_INDEPENDENT_REGISTRATION_REPLICATION",
        fixture_sha256=FROZEN_FIXTURE_SHA256, physical_or_native_qualification=False,
        observed_bend_count=None, production_use_allowed=False)


def test_inventory_has_all_science_and_shared_registration_stages():
    assert len(STAGE_IDS) == len(set(STAGE_IDS)) == 16
    assert len(ARTIFACT_SPECS) == 32
    assert sum(cap for name, cap in ARTIFACT_SPECS.items() if "_RESULT_" in name) == TOTAL_RESULT_CAP
    assert sum(cap for name, cap in ARTIFACT_SPECS.items() if "_START_" in name) == TOTAL_START_CAP


@pytest.mark.parametrize("index", range(4), ids=CURVED_CASE_IDS)
def test_frozen_geometry_and_oracles_are_reused_without_mutation(frozen, index):
    case = frozen.CURVED_CASES[index]
    original, truth = frozen.curved_packet(case)
    owned = _immutable_inputs(original)
    assert frozen._array_digests(owned) == frozen._array_digests(original)
    assert all(not value.flags.writeable for value in owned.values() if type(value) is np.ndarray)
    assert len(owned["scan_points"]) == (650 if case.localized else 450)
    assert len(owned["cad_triangles"]) == 564
    assert truth["case"] == case
    assert owned["position_error_bounds_mm"] == .05
    assert frozen.CLEAN_CRITERIA == dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
        generator_endpoint_error_mm=.02, axis_endpoint_error_mm=.02)
    assert frozen.NOISY_CRITERIA == dict(radius_error_mm=.02, absolute_sweep_error_deg=.1,
        generator_endpoint_error_mm=.05, axis_endpoint_error_mm=.05)


def test_recording_encoding_failure_does_not_publish_a_prefix(tmp_path):
    with pytest.raises(ValueError, match="PAIRED_PACKET_PUBLICATION_BYTE_LIMIT"):
        _write_record(tmp_path, "example.json", {"value": "too large"}, 2)
    assert not (tmp_path/"example.json").exists()
    assert not (tmp_path/"example.json.partial").exists()


def test_recorder_retains_failure_then_next_declared_stage(tmp_path):
    recorder = _Recorder(tmp_path)
    def failed():
        raise ValueError("INVENTED_CALL_FAILURE_NO_ENGINE_EXECUTION")
    def unused(answer):
        raise AssertionError("Oracle must not run without an answer")
    recorder.stage("PLANAR", 0, "V11", "P-clean", {}, failed, unused)
    recorder.stage("PLANAR", 0, "V12", "P-clean", {},
        lambda: ({"invented": "retained"}, None), lambda answer: {"valid": True})
    assert len(recorder.entries) == 2
    first = json.loads((tmp_path/"PLANAR_RESULT_00_V11.json").read_bytes())
    assert first["execution_error"]["type"] == "ValueError" and first["answer"] is None
    second = json.loads((tmp_path/"PLANAR_RESULT_00_V12.json").read_bytes())
    assert second["answer"] == {"invented": "retained"} and second["oracle_passed"] is True
    assert all(row["timing"]["call_wall_seconds"] >= 0 and row["timing"]["call_cpu_seconds"] >= 0
               for row in recorder.entries)


def test_recorder_encoding_failure_keeps_whole_discard_disposition(tmp_path):
    recorder = _Recorder(tmp_path)
    recorder.stage("PLANAR", 0, "V11", "P-clean", {},
        lambda: ({"not_json": float("nan")}, None), lambda answer: {})
    value = json.loads((tmp_path/"PLANAR_RESULT_00_V11.json").read_bytes())
    assert value["disposition"] == "RESULT_RETENTION_UNAVAILABLE"
    assert value["answer"] is None and value["discarded_answer_whole"] is True
    assert value["oracle_passed"] is False


def test_packet_opt_in_precedes_artifact_creation(frozen, tmp_path, monkeypatch):
    monkeypatch.delenv(OPT_IN, raising=False)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(AssertionError, match="SUPPORT_COVER_PACKET_REQUIRES_EXPLICIT_OPT_IN"):
        run_support_cover_packet(frozen, path)
    assert not path.exists()


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1", reason="Exact source/connected PRE and bounded supervisor required")
def test_support_cover_connected_packet(frozen):
    directory = os.environ.get(ARTIFACT_ENV)
    assert directory, "Fresh ARTIFACTS path required"
    receipt = run_support_cover_packet(frozen, directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    assert receipt["status"] == "PASS", receipt
