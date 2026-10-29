"""Prospective composite proposal packet, not execution authority.

Reuse the exact independent four curved constructions and every measurement
oracle from the terminal support-cover packet. Only scan-plane preparation
changes. Both measurement algorithms still receive the same new contexts; V1
is diagnostic, not the predecessor of the full changed pipeline. The full
pipeline control is the retained, audited V7+coverV2 result on identical inputs.
No physical/Gold/native fixture is used. No part identifier enters inference.
"""

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = "tests/test_cad_guided_support_cover_connected_v2.py"
HELPER_SHA256 = "c645c7f0ff96e726989eb929176a8b1286f45c1ef996c68b21473ec488707cb5"
CONTROL_POST_SHA256 = "261b12b17d948bdbf37d2d394468debf1ec67dd566ba4bee6672c5ca3a306e05"
OPT_IN = "SHERMAN_RUN_COMPOSITE_PLANE_CONNECTED_V1"
ARTIFACT_ENV = "SHERMAN_COMPOSITE_PLANE_CONNECTED_V1_ARTIFACT_DIRECTORY"
RESOURCE_LIMITS = dict(wall_seconds=4800, work_wall_seconds=4795,
    worker_cpu_seconds=4200, sampled_process_group_rss_bytes=4*1024**3,
    stdout_bytes=1024**2, execution_enforcement="EXTERNAL_SUPERVISOR_ONLY_NO_PER_ARM_KILLING")
COMMON_CAD_DIGESTS = dict(
    cad_face_ids="dca2a38fba5ab6432517f89dace6e40fe48009aec5e8a4773e77e781c0d8a031",
    cad_source_triangle_ids="8408baf40dd517218eb6e665efbd4564c85a91e220363421969d219309d1d41e",
    cad_triangles="835d6e433a70fbd49d9f25b6b72a6a4e5cc5c052186ca7f93d688188b479a663")
CONTROL_SCANS = (
    ("e35413e299bb669522ced7da745d0eab44297047634bf7e21cee270972abf895",
     "02ce0d286d7e0af44b341a50eeaa034e9e713f298bf40c3c17b8ed0cac8ccbc6"),
    ("7ccd0153ee3e9ff38a696994bd04a470eb9948f45aa9779421a115c6a42aefe8",
     "02ce0d286d7e0af44b341a50eeaa034e9e713f298bf40c3c17b8ed0cac8ccbc6"),
    ("994b3d8a4de940c6470393fdb60dc20d9fb368434c135bd33cbc496872cfce3c",
     "8c4f5c76ca41d6fd9d63a7523a4cace886392e307fe07b56140dd8b510c12de6"),
    ("89a9670d884ef4b1d1dc00a0d84b1456ea3118144c5a3c23e2f9ee20e3d998a8",
     "8c4f5c76ca41d6fd9d63a7523a4cace886392e307fe07b56140dd8b510c12de6"))
CONTROL_POSITIVE = (True, True, False, False)


def _load_helper():
    path = ROOT/HELPER_PATH
    with path.open("rb") as stream:
        blob = stream.read(512*1024+1)
    assert len(blob) <= 512*1024 and hashlib.sha256(blob).hexdigest() == HELPER_SHA256
    name = "_composite_plane_v1_frozen_packet_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(blob, str(path), "exec"), module.__dict__)
        return module
    finally:
        if previous is None: del sys.modules[name]
        else: sys.modules[name] = previous


@pytest.fixture(scope="module")
def source():
    helper = _load_helper()
    return helper, helper._load_frozen_fixture()


def _expected_inputs(index):
    point_sha, triangle_sha = CONTROL_SCANS[index]
    return dict(COMMON_CAD_DIGESTS, scan_points=point_sha, scan_triangles=triangle_sha)


def _composite_registration(helper, frozen, inputs):
    module, policies, instance_policy = frozen._policy_objects("C2", planar=False)
    from backend.cad_guided_scan_plane_preparation_v1 import CompositeScanPlaneBudgetsV1
    from backend.cad_guided_seeded_plane_proposals_v1 import SeededPlaneBudgetsV1
    policies.update(scan_plane_mode="composite_v1", seeded_plane_budgets=SeededPlaneBudgetsV1(),
                    composite_plane_budgets=CompositeScanPlaneBudgetsV1())
    cad_points = np.unique(inputs["cad_triangles"].reshape(-1, 3), axis=0)
    cad_points = np.frombuffer(cad_points.tobytes(), dtype=cad_points.dtype).reshape(cad_points.shape)
    assert len(cad_points) >= 36
    before = frozen._array_digests(inputs)
    snapshot = {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v for k, v in policies.items()}
    registration, owned = module.register_measurement_contexts_v12(
        inputs["scan_points"], inputs["scan_triangles"], inputs["cad_triangles"],
        inputs["cad_face_ids"], inputs["cad_source_triangle_ids"],
        position_error_bounds_mm=inputs["position_error_bounds_mm"], cad_initialization_points=cad_points,
        **policies, connected_measurement_contexts=True)
    answer = dict(registration=registration, registration_sha256=None, owned_role_binding=None,
        policy_snapshot=snapshot, binding_error=None, cad_initialization_point_count=len(cad_points),
        cad_initialization_points_sha256=helper._array_sha(cad_points),
        cad_initialization_point_contract="UNIQUE_DECLARED_CAD_TRIANGLE_VERTICES_NOT_NATIVE_SAMPLER",
        comparison_contract="COMPOSITE_PREPARATION_SAME_FROZEN_CURVED_INPUTS")
    auxiliary = dict(owned=owned, instance_policy=instance_policy)
    try:
        assert frozen._array_digests(inputs) == before, "COMPOSITE_REGISTRATION_INPUT_MUTATED"
        answer["registration_sha256"] = helper._json_sha(registration, helper.RESULT_CAPS["REGISTRATION"])
        answer["owned_role_binding"] = helper._owned_role_fingerprint(owned)
        expected = dict(mode="composite_v1", seeded_budgets=asdict(policies["seeded_plane_budgets"]),
                        composite_budgets=asdict(policies["composite_plane_budgets"]))
        evidence = registration["evidence"]
        assert evidence["scan_plane_configuration"] == evidence["initialization"]["scan_plane_configuration"] == expected
    except Exception as error:
        answer["binding_error"], auxiliary = helper._error(error), None
    return answer, auxiliary


def run_composite_packet(helper, frozen, artifact_directory):
    assert os.environ.get(OPT_IN) == "1", "COMPOSITE_PACKET_REQUIRES_EXPLICIT_OPT_IN"
    directory = Path(artifact_directory)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    directory.mkdir(exist_ok=False)
    recorder = helper._Recorder(directory, emit_progress=True)
    comparisons = []
    # These unchanged legacy prefilter controls are not composite measurements.
    for index, case_id in enumerate(helper.PLANAR_CASE_IDS):
        original, truth = frozen._planar_packet(case_id == "P-noisy")
        inputs = helper._immutable_inputs(original)
        binding = dict(input_digests=frozen._array_digests(inputs),
            fixture_sha256=helper.FROZEN_FIXTURE_SHA256, scope="UNCHANGED_LEGACY_PREFILTER_REGRESSION")
        for arm in helper.PLANAR_ARMS:
            positive = arm == "V12" or case_id == "P-clean"
            recorder.stage("PLANAR", index, arm, case_id, binding,
                lambda arm=arm: helper._planar_call(frozen, inputs, arm),
                lambda answer, positive=positive: helper._planar_oracle(answer, truth, require_positive=positive),
                required_positive=positive)
    assert tuple(case.case_id for case in frozen.CURVED_CASES) == helper.CURVED_CASE_IDS
    for index, case in enumerate(frozen.CURVED_CASES):
        original, truth = frozen.curved_packet(case)
        inputs = helper._immutable_inputs(original)
        digests = frozen._array_digests(inputs)
        assert digests == _expected_inputs(index), "RETAINED_CONTROL_INPUTS_NOT_IDENTICAL"
        binding = dict(input_digests=digests, fixture_sha256=helper.FROZEN_FIXTURE_SHA256,
            control_post_sha256=CONTROL_POST_SHA256, scope="IDENTICAL_ARRAYS_COMPOSITE_PROPOSAL_SUCCESSOR")
        shared, auxiliary, registration_entry = recorder.stage("REGISTRATION", index, "SHARED",
            case.case_id, binding, lambda: _composite_registration(helper, frozen, inputs), helper._registration_oracle)
        if registration_entry["disposition"] != "COMPLETED_STAGE": shared = auxiliary = None
        paired_binding = dict(binding, shared_registration_artifact=registration_entry["artifact"],
            shared_registration_sha256=None if shared is None else shared["registration_sha256"],
            registration_recomputed_for_measurement_arms=False)
        diagnostic, entries = None, {}
        for arm in helper.MEASUREMENT_ARMS:
            answer, _, entry = recorder.stage("MEASUREMENT", index, arm, case.case_id, paired_binding,
                lambda arm=arm: helper._measurement_call(frozen, inputs, shared, auxiliary, arm, diagnostic),
                lambda answer, arm=arm: helper._measurement_oracle(frozen, answer, truth, arm),
                required_positive=arm == "V2", safety=lambda answer, arm=arm: helper._measurement_safety(answer, arm))
            entries[arm] = entry
            if arm == "V1": diagnostic = answer
        assert frozen._array_digests(inputs) == digests
        full = {arm: row["disposition"] == "COMPLETED_STAGE" and row["execution_error"] is None
                and row["safety_error"] is None and row["oracle_passed"] for arm, row in entries.items()}
        comparisons.append(dict(case_id=case.case_id, retained_v7_cover_v2_positive=CONTROL_POSITIVE[index],
            composite_cover_v2_positive=full["V2"], same_context_v1_diagnostic_positive=full["V1"],
            same_input_hashes_verified=True))
    assert tuple((r["kind"], r["case_index"], r["arm"]) for r in recorder.entries) == helper.STAGE_IDS
    passed = all(r["disposition"] == "COMPLETED_STAGE" and r["execution_error"] is None
        and r["safety_error"] is None and (r["oracle_passed"] or not r["required_positive"])
        for r in recorder.entries)
    positives = sum(r["composite_cover_v2_positive"] for r in comparisons)
    return dict(schema="COMPOSITE_PLANE_CONNECTED_PACKET_V1",
        status="PASS" if passed and positives == 4 else "FAIL", entries=recorder.entries,
        stage_count=16, result_bytes=recorder.result_bytes, start_bytes=recorder.start_bytes,
        comparisons=comparisons, retained_control_positive_cases=sum(CONTROL_POSITIVE),
        challenger_positive_cases=positives, control_post_sha256=CONTROL_POST_SHA256,
        comparison_scope="RETAINED_AUDITED_CONTROL_SAME_CONSTRUCTED_INPUTS_NOT_PHYSICAL_F1",
        resource_limits=RESOURCE_LIMITS, physical_or_native_qualification=False,
        observed_bend_count=None, production_use_allowed=False)


@pytest.mark.parametrize("index", range(4))
def test_frozen_curved_inputs_match_retained_control_and_keep_original_oracles(source, index):
    helper, frozen = source
    # The complete original fixture check includes clean/noisy error ceilings.
    helper.test_frozen_geometry_and_oracles_are_reused_without_mutation(frozen, index)
    original, _ = frozen.curved_packet(frozen.CURVED_CASES[index])
    assert frozen._array_digests(original) == _expected_inputs(index)


def test_complete_sixteen_stage_inventory_is_unchanged(source):
    helper, _ = source
    helper.test_inventory_has_all_science_and_shared_registration_stages()


def test_opt_in_precedes_artifact_creation(source, tmp_path, monkeypatch):
    helper, frozen = source
    monkeypatch.delenv(OPT_IN, raising=False)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(AssertionError, match="COMPOSITE_PACKET_REQUIRES_EXPLICIT_OPT_IN"):
        run_composite_packet(helper, frozen, path)
    assert not path.exists()


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1", reason="Separate exact PRE and bounded supervisor required")
def test_composite_connected_packet(source):
    directory = os.environ.get(ARTIFACT_ENV)
    assert directory
    receipt = run_composite_packet(*source, directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    assert receipt["status"] == "PASS", receipt
