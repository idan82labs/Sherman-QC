"""Prospective frozen four-case packet with explicit scheduler V2; PRE required.

The retained composite/scheduler-V1 result is a custody-limited 2/4 diagnostic
comparator: its old source freeze omitted tests/__init__.py. The old outcomes
remain negative; this new packet cannot retroactively repair that omission.
The terminal depth V1 packet then failed at the old context-per-pair quota.
This successor selects the explicit context V2 consumer through the same V2
scheduler request. Geometry, raw inputs and all measurement oracles are unchanged.
V1 measurement on the same new contexts remains diagnostic, not that control.
Missing execution limits refuse before any fixture or artifact dispatch.
"""

import ast
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
CONTROL_POST_PATH = "docs/research/2026-09-10_composite_plane_v1_connected_post_run_audit.md"
CONTROL_POST_SHA256 = "1a9ebb6e3a0b04de767b90481336c84a6bc3681c71161daf6a201c568a4c844b"
CONTROL_FIXTURE_PATH = "tests/test_cad_guided_composite_plane_connected_v1.py"
CONTROL_FIXTURE_SHA256 = "938a3672a59794edc2c7e1070a8acf810fb99ee44bf1d345d1d1e03b36c1cc70"
DEPTH_NEGATIVE_POST_PATH = "docs/research/2026-09-11_two_plane_depth_v1_connected_post_run_audit.md"
DEPTH_NEGATIVE_POST_SHA256 = "e9ea38b68e0bee42c74e6e84666998449e20f5a2ec474761d6afb771cae55867"
OPT_IN = "SHERMAN_RUN_TWO_PLANE_DEPTH_CONNECTED_V2"
ARTIFACT_ENV = "SHERMAN_TWO_PLANE_DEPTH_CONNECTED_V2_ARTIFACT_DIRECTORY"
RESOURCE_LIMITS = dict(wall_seconds=6600, work_wall_seconds=6595,
    worker_cpu_seconds=6000, sampled_process_group_rss_bytes=4*1024**3,
    stdout_bytes=1024**2, execution_enforcement="EXTERNAL_SUPERVISOR_ONLY_NO_PER_ARM_KILLING")
# New, explicitly versioned transport allowance, not a change to an old run.
RESULT_CAPS = dict(PLANAR=8*1024**2, REGISTRATION=64*1024**2, MEASUREMENT=32*1024**2)
TOTAL_RESULT_CAP = 544*1024**2
TOTAL_START_CAP = 16*4096
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
# Released POST evidence identifiers only; the packet never opens old artifacts.
CONTROL_REGISTRATION_ARTIFACTS = (
    (4432107, "81802dee98b2b81df5707b6d8bd6dab14b8ba723d8f69120984d2f2ef37ab733"),
    (14245612, "59d5cfd172da18dc8d11e91d45ce3e25ff4fbb12a85bfd54384e46015a23515f"),
    (14113363, "26c92a91f3b52da23d033b1c46601fd8b2f2408808dbb08f623a809c2bf417b0"),
    (8176861, "93e211835b7c029d47d4fd59f034261caeeadcea0f17c0f41198cd5d366845cd"))


def _require_execution_limits():
    wall, work, cpu = (RESOURCE_LIMITS[key] for key in
        ("wall_seconds", "work_wall_seconds", "worker_cpu_seconds"))
    if not (all(type(value) is int and value > 0 for value in (wall, work, cpu))
            and cpu <= work <= wall-5):
        raise ValueError("DEPTH_PACKET_LIMITS_UNSET_EXACT_PRE_REQUIRED")


def _verified_source(relative, expected, maximum=512*1024):
    with (ROOT/relative).open("rb") as stream:
        blob = stream.read(maximum+1)
    assert len(blob) <= maximum and hashlib.sha256(blob).hexdigest() == expected
    return blob


def _load_helper():
    blob = _verified_source(HELPER_PATH, HELPER_SHA256)
    name = "_two_plane_depth_v2_isolated_frozen_packet_helpers"
    spec = importlib.util.spec_from_file_location(name, ROOT/HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(blob, str(ROOT/HELPER_PATH), "exec"), module.__dict__)
        assert module.RESULT_CAPS == dict(PLANAR=8*1024**2, REGISTRATION=16*1024**2, MEASUREMENT=32*1024**2)
        assert module.TOTAL_RESULT_CAP == 352*1024**2
        # Only this fresh test-helper instance changes. Function bytecode,
        # source files, production globals and old packet instances do not.
        module.RESULT_CAPS = dict(RESULT_CAPS)
        module.TOTAL_RESULT_CAP = TOTAL_RESULT_CAP
        module.ARTIFACT_SPECS = {
            f"{kind}_{phase}_{index:02d}_{arm}.json": module.START_CAP if phase == "START" else RESULT_CAPS[kind]
            for kind, index, arm in module.STAGE_IDS for phase in ("START", "RESULT")}
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


def _depth_registration(helper, frozen, inputs):
    module, policies, instance_policy = frozen._policy_objects("C2", planar=False)
    from backend.cad_guided_scan_plane_preparation_v1 import CompositeScanPlaneBudgetsV1
    from backend.cad_guided_seeded_plane_proposals_v1 import SeededPlaneBudgetsV1
    from backend.cad_guided_two_plane_schedule_v2 import TwoPlaneScheduleBudgetsV2
    from backend.cad_guided_mesh_offline_v12 import _two_plane_schedule_binding
    from backend.cad_guided_plane_initialization_v12 import _validate_two_plane_context_configuration
    policies.update(scan_plane_mode="composite_v1", seeded_plane_budgets=SeededPlaneBudgetsV1(),
        composite_plane_budgets=CompositeScanPlaneBudgetsV1(),
        two_plane_schedule_mode="v2", two_plane_schedule_budgets=TwoPlaneScheduleBudgetsV2())
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
        policy_snapshot=snapshot, binding_error=None, two_plane_schedule_binding=None,
        two_plane_context_profile_bound=False,
        cad_initialization_point_count=len(cad_points), cad_initialization_points_sha256=helper._array_sha(cad_points),
        cad_initialization_point_contract="UNIQUE_DECLARED_CAD_TRIANGLE_VERTICES_NOT_NATIVE_SAMPLER",
        comparison_contract="EXPLICIT_DEPTH_SCHEDULER_V2_SAME_COMPOSITE_PREPARATION_AND_FROZEN_INPUTS")
    auxiliary = dict(owned=owned, instance_policy=instance_policy)
    try:
        assert frozen._array_digests(inputs) == before, "DEPTH_REGISTRATION_INPUT_MUTATED"
        answer["registration_sha256"] = helper._json_sha(registration, RESULT_CAPS["REGISTRATION"])
        answer["owned_role_binding"] = helper._owned_role_fingerprint(owned)
        expected = dict(mode="composite_v1", seeded_budgets=asdict(policies["seeded_plane_budgets"]),
                        composite_budgets=asdict(policies["composite_plane_budgets"]))
        evidence = registration["evidence"]
        assert evidence["scan_plane_configuration"] == evidence["initialization"]["scan_plane_configuration"] == expected
        assert helper._encode(evidence["initialization"]["budgets"], 65536) == helper._encode(snapshot["budgets"], 65536)
        requested_schedule = dict(mode="v2", budgets=asdict(policies["two_plane_schedule_budgets"]))
        answer["two_plane_schedule_binding"] = _two_plane_schedule_binding(registration, requested_schedule)
        assert answer["two_plane_schedule_binding"]["two_plane_schedule_mode"] == "v2"
        context_report = evidence["initialization"]["two_plane_context"]
        _validate_two_plane_context_configuration(context_report, "v2")
        assert context_report["status"] == "COMPLETE"
        answer["two_plane_context_profile_bound"] = True
    except Exception as error:
        # Actual reports, including early unavailable schedules, survive.
        # No measurement arm may consume an unbound partial registration.
        answer["binding_error"], auxiliary = helper._error(error), None
    return answer, auxiliary


def run_depth_packet(helper, frozen, artifact_directory):
    assert os.environ.get(OPT_IN) == "1", "DEPTH_PACKET_REQUIRES_EXPLICIT_OPT_IN"
    _require_execution_limits()
    directory = Path(artifact_directory)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    directory.mkdir(exist_ok=False)
    recorder = helper._Recorder(directory, emit_progress=True)
    comparisons = []
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
        assert digests == _expected_inputs(index), "RETAINED_COMPOSITE_CONTROL_INPUTS_NOT_IDENTICAL"
        old_size, old_sha = CONTROL_REGISTRATION_ARTIFACTS[index]
        binding = dict(input_digests=digests, fixture_sha256=helper.FROZEN_FIXTURE_SHA256,
            control_post_sha256=CONTROL_POST_SHA256, retained_control_registration_bytes=old_size,
            retained_control_registration_sha256=old_sha,
            scope="IDENTICAL_ARRAYS_COMPOSITE_SCHEDULER_V2_SUCCESSOR")
        shared, auxiliary, registration_entry = recorder.stage("REGISTRATION", index, "SHARED",
            case.case_id, binding, lambda: _depth_registration(helper, frozen, inputs), helper._registration_oracle)
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
        comparisons.append(dict(case_id=case.case_id, retained_composite_scheduler_v1_cover_v2_positive=CONTROL_POSITIVE[index],
            predecessor_depth_v1_context_v1_positive=False,
            depth_scheduler_v2_cover_v2_positive=full["V2"], same_context_v1_diagnostic_positive=full["V1"],
            same_input_hashes_verified=True))
    assert tuple((r["kind"], r["case_index"], r["arm"]) for r in recorder.entries) == helper.STAGE_IDS
    passed = all(r["disposition"] == "COMPLETED_STAGE" and r["execution_error"] is None
        and r["safety_error"] is None and (r["oracle_passed"] or not r["required_positive"])
        for r in recorder.entries)
    positives = sum(r["depth_scheduler_v2_cover_v2_positive"] for r in comparisons)
    return dict(schema="TWO_PLANE_DEPTH_CONNECTED_PACKET_V2",
        status="PASS" if passed and positives == 4 else "FAIL", entries=recorder.entries,
        stage_count=16, result_bytes=recorder.result_bytes, start_bytes=recorder.start_bytes,
        comparisons=comparisons, retained_control_positive_cases=sum(CONTROL_POSITIVE),
        challenger_positive_cases=positives, control_post_sha256=CONTROL_POST_SHA256,
        predecessor_depth_positive_cases=0, predecessor_depth_post_sha256=DEPTH_NEGATIVE_POST_SHA256,
        comparison_scope="RETAINED_COMPOSITE_DIAGNOSTIC_WITH_KNOWN_IMPORT_CLOSURE_OMISSION_NOT_PHYSICAL_F1",
        retained_comparator_import_closure_complete=False,
        resource_limits=RESOURCE_LIMITS, result_caps=RESULT_CAPS, whole_result_cap=TOTAL_RESULT_CAP,
        physical_or_native_qualification=False, observed_bend_count=None, production_use_allowed=False)


@pytest.mark.parametrize("index", range(4))
def test_frozen_curved_inputs_match_retained_control_and_keep_original_oracles(source, index):
    helper, frozen = source
    helper.test_frozen_geometry_and_oracles_are_reused_without_mutation(frozen, index)
    original, _ = frozen.curved_packet(frozen.CURVED_CASES[index])
    assert frozen._array_digests(original) == _expected_inputs(index)
    # Bind the released comparator without opening its scientific artifacts.
    _verified_source(CONTROL_POST_PATH, CONTROL_POST_SHA256)
    _verified_source(DEPTH_NEGATIVE_POST_PATH, DEPTH_NEGATIVE_POST_SHA256)
    tree = ast.parse(_verified_source(CONTROL_FIXTURE_PATH, CONTROL_FIXTURE_SHA256))
    old = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"CONTROL_SCANS", "CONTROL_POSITIVE"}}
    assert old["CONTROL_SCANS"][index] == CONTROL_SCANS[index]
    assert old["CONTROL_POSITIVE"] == CONTROL_POSITIVE


def test_complete_sixteen_stage_inventory_and_isolated_transport_policy(source):
    helper, _ = source
    helper.test_inventory_has_all_science_and_shared_registration_stages()
    assert helper.RESULT_CAPS == RESULT_CAPS and helper.RESULT_CAPS is not RESULT_CAPS
    assert helper.TOTAL_RESULT_CAP == TOTAL_RESULT_CAP == 4*8*1024**2+4*64*1024**2+8*32*1024**2
    assert helper.TOTAL_START_CAP == TOTAL_START_CAP
    second = _load_helper()
    assert second is not helper and second.RESULT_CAPS is not helper.RESULT_CAPS
    assert second.ARTIFACT_SPECS == helper.ARTIFACT_SPECS
    _verified_source(HELPER_PATH, HELPER_SHA256)
    # Source-level dispatch delta, not a second registration or geometry run.
    old = ast.parse(_verified_source(CONTROL_FIXTURE_PATH, CONTROL_FIXTURE_SHA256))
    current = ast.parse(Path(__file__).read_bytes())
    def policy_updates(tree, name):
        function = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "policies" and node.func.attr == "update"]
        assert len(calls) == 1 and not calls[0].args
        return {row.arg: ast.dump(row.value, include_attributes=False) for row in calls[0].keywords}
    old_updates = policy_updates(old, "_composite_registration")
    new_updates = policy_updates(current, "_depth_registration")
    assert {key: new_updates[key] for key in old_updates} == old_updates
    assert set(new_updates)-set(old_updates) == {"two_plane_schedule_mode", "two_plane_schedule_budgets"}
    assert new_updates["two_plane_schedule_mode"] == ast.dump(ast.Constant(value="v2"))
    assert new_updates["two_plane_schedule_budgets"] == ast.dump(ast.parse("TwoPlaneScheduleBudgetsV2()", mode="eval").body)


def test_opt_in_and_unset_limits_precede_artifact_creation(source, tmp_path, monkeypatch):
    helper, frozen = source
    monkeypatch.delenv(OPT_IN, raising=False)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(AssertionError, match="DEPTH_PACKET_REQUIRES_EXPLICIT_OPT_IN"):
        run_depth_packet(helper, frozen, path)
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setitem(RESOURCE_LIMITS, "worker_cpu_seconds", None)
    with pytest.raises(ValueError, match="DEPTH_PACKET_LIMITS_UNSET_EXACT_PRE_REQUIRED"):
        run_depth_packet(helper, frozen, path)
    assert not path.exists()


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1", reason="Separate exact PRE, frozen limits and bounded supervisor required")
def test_two_plane_depth_connected_packet(source):
    directory = os.environ.get(ARTIFACT_ENV)
    assert directory
    receipt = run_depth_packet(*source, directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    assert receipt["status"] == "PASS", receipt
