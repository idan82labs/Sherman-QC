"""Prospective invented-only V5 runner qualification; never a physical run.

Only selected definitions/constants are AST-loaded. Entry definitions are called
solely behind an explicitly false first-statement guard in detached namespaces;
no CLI, freeze, execute, authorization, real input, process or native work occurs.
Real backend policy constructors use an inert namespace; real pure projection,
matching, publication and complete campaign-tree functions remain connected.
The two prior test sources supply ordinary invented dictionary constructors,
extracted explicitly below, not pytest fixtures or old test execution.

The full campaign has exactly11 invented cases, six invented families,71 invented
straight references,22 arm results and119 scientific paths, plus the complete
runner-declared source/control/46 issued-and-claimed dispatch namespace. Source
file contents/version metadata in that tree are explicitly invented custody
inputs, NOT a runtime/source-freeze attestation. No tiny-schema substitution is
used. Ordinary fixture payload is capped at16MiB; the separate newline boundary
uses one8MiB retained projection (and its one-byte-over negative). Final control
envelopes must meet the production256KiB cap without dropping any ledger.

Prospective source-only gate:180wall/120workerCPU/2GiB/1MiB stdout, no smoke or
native opt-in. Source authored without imports, collection or execution. Any
failure remains a failure until independently reviewed; this is not F1 evidence.
"""

import ast
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_gold11_mesh_development_v5.py"
PREDECESSOR = ROOT / "scripts/run_gold11_mesh_development_v4.py"
HELPER = ROOT / "scripts/gold11_terminal_artifacts_v4.py"
CONTACT_FIXTURE = ROOT / "tests/test_gold11_contact_scoring_v2.py"
SCHEDULE_FIXTURE = ROOT / "tests/test_gold11_schedule_projection_v1.py"
RUNNER_SHA = "71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481"
V4_SHA = "559095a685d6ae5badecdb9704952d5e80304dcd728f8afa0f131a25707ebd85"
HELPER_SHA = "3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838"
MAX_FIXTURE_BYTES = 16 * 1024**2
SELECTED = {
    "digest", "encoded", "unique_object", "decode", "unavailable", "is_hex",
    "require_release", "arm_request", "arm_request_sha256", "all_arm_requests",
    "scan_plane_requests", "projection_request", "compact_result",
    "valid_supervisor_refusal", "reconstruct_scoring_input", "reconstruct_all_terminal_inputs",
    "make_scoring_request", "scorer_arguments", "workflow_entry_and_policies",
    "leaf_sources", "validate_leaf_binding", "workflow_stage_diagnostics",
    "_bounded_read", "bounded_read", "owned_read", "marker_present", "publication_cap",
    "artifact_directory_walk", "artifact_alarm_guard", "publish_bytes", "exclusive_json",
    "terminalize_case_records", "_tree_fingerprint", "_open_tree_directory", "_tree_names",
    "_campaign_schema", "_verify_tree_leaf_bindings", "verify_campaign_inventory",
    "verify_parent_artifacts", "final_source_audit", "score_after_terminal",
    "supervise_case", "retain_worker_result",
}
CLASSES = {"_ArtifactDeadlineExpired", "_CampaignInventoryStop"}
UNSAFE_ENTRIES = {"main", "freeze", "execute", "worker", "scorer_worker",
                  "internal_coordinator", "authorize_leaf", "verify_bundle"}
CONTACT_CONSTRUCTORS = {"_configuration", "_identity", "_instance", "_fit_policy",
    "_workflow", "_configuration_parents", "_wire", "_refusal"}
SCHEDULE_CONSTRUCTORS = {"_sha", "_request", "_schedule", "_payload"}
TIMING_KEYS = ("case_wall_seconds", "case_cpu_seconds", "total_wall_seconds",
    "scorer_wall_seconds", "scorer_cpu_seconds", "parent_read_seconds",
    "terminalization_reserve_seconds")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assignment(tree, name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(v, ast.Name) and v.id == name for v in n.targets))


class FakeSignal:
    """Synchronous declared timer only; no real alarm or wait is started."""
    ITIMER_REAL, SIGALRM = 0, 14

    def __init__(self):
        self.remaining = 120.
        self.handler = lambda *_: None

    def getitimer(self, _):
        return self.remaining, 0.

    def getsignal(self, _):
        return self.handler

    def signal(self, _, handler):
        old, self.handler = self.handler, handler
        return old


@contextmanager
def fake_deadline(_seconds):
    yield


@pytest.fixture
def runner(tmp_path):
    source = SOURCE.read_bytes()
    assert sha(source) == RUNNER_SHA
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
        or isinstance(n, ast.FunctionDef) and n.name in SELECTED
        or isinstance(n, ast.ClassDef) and n.name in CLASSES]
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == SELECTED
    ns = dict(__name__="_invented_v5_definitions", __file__=str(SOURCE))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), ns)
    assert not UNSAFE_ENTRIES.intersection(ns)
    helper_blob = HELPER.read_bytes()
    assert sha(helper_blob) == HELPER_SHA
    helper = ModuleType("_invented_v5_qualified_artifact_helper")
    exec(compile(helper_blob, str(HELPER), "exec"), helper.__dict__)
    run, current = tmp_path / "campaign", tmp_path / "current-invented-sources"
    run.mkdir()
    current.mkdir()
    limits = deepcopy(ns["LIMITS"])
    for key in TIMING_KEYS:
        limits[key] = 20 if key != "total_wall_seconds" else 120
    ns.update(RUN=run.resolve(), ROOT=current.resolve(), LIMITS=limits,
        signal=FakeSignal(), time=SimpleNamespace(monotonic=lambda: 0.),
        deadline=fake_deadline, artifact_module=lambda: helper,
        runtime_metadata=lambda: {"scope": "INVENTED_METADATA_NOT_RUNTIME_ATTESTATION"})
    return SimpleNamespace(ns=ns, helper=helper, run=run, current=current, tmp=tmp_path)


@pytest.fixture(scope="module")
def invented():
    ns = dict(copy=__import__("copy"), json=json, hashlib=hashlib)
    tree = ast.parse(CONTACT_FIXTURE.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in CONTACT_CONSTRUCTORS]
    assert {n.name for n in nodes} == CONTACT_CONSTRUCTORS
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONTACT_FIXTURE), "exec"), ns)
    tree = ast.parse(SCHEDULE_FIXTURE.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in SCHEDULE_CONSTRUCTORS
        or isinstance(n, ast.Assign) and any(isinstance(v, ast.Name)
            and v.id in {"BREADTH", "FULL", "INITIALIZATION"} for v in n.targets)]
    assert {n.name for n in nodes if isinstance(n, ast.FunctionDef)} == SCHEDULE_CONSTRUCTORS
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCHEDULE_FIXTURE), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def projection():
    # Exact pure module graph; not the runner's authority/source loader.
    prefix = "_invented_v5_pure_scripts"
    package = ModuleType(prefix)
    package.__path__ = [str(ROOT / "scripts")]
    assert not any(n == prefix or n.startswith(prefix + ".") for n in sys.modules)
    sys.modules[prefix] = package
    try:
        yield importlib.import_module(prefix + ".gold11_schedule_projection_v1")
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def connect_projection(runner, projection):
    runner.ns["projection_module"] = lambda: projection


def payload_for(invented, arm, *, finite=True):
    return invented["_payload"](schedule="v2" if arm == "challenger" else "v1",
        scan="composite_v1" if arm == "challenger" else "v7", finite=finite)


def clean_receipt(runner, arm, result, wrapper):
    return dict(status="TERMINAL", exit_code=0, budget_stop=None,
        arm_request_sha256=runner.ns["arm_request_sha256"](arm),
        scoring_input_sha256=sha(wrapper), result_sha256=sha(result),
        projection_kind="EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER")


def projected(runner, projection, invented, arm="challenger", *, finite=True):
    connect_projection(runner, projection)
    original = b" \n" + wire(payload_for(invented, arm, finite=finite)) + b"\n "
    outer = runner.ns["compact_result"](original, expected_result_sha256=sha(original), arm=arm)
    retained = wire(outer) + b"\n"
    return original, retained, clean_receipt(runner, arm, original, retained)


def put(root, relative, blob):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        assert stream.write(blob) == len(blob)
    return target


def replace_owned(root, relative, value):
    # Deliberate corruption inside this test's invented directory only.
    blob = value if type(value) is bytes else wire(value) + b"\n"
    (root / relative).write_bytes(blob)


@pytest.mark.parametrize("constant", ["CONFIG", "CERTIFICATE_CAPS", "REVIEW_SHA", "CATALOG_SHA", "SELECTED_SHA"])
def test_frozen_scientific_and_september9_hash_pins_match_v4_ast(constant):
    old_blob = PREDECESSOR.read_bytes()
    assert sha(old_blob) == V4_SHA
    old, new = ast.parse(old_blob), ast.parse(SOURCE.read_bytes())
    assert ast.dump(assignment(new, constant)) == ast.dump(assignment(old, constant))


@pytest.mark.parametrize("entry", sorted(UNSAFE_ENTRIES))
def test_every_entry_is_blocked_before_source_control_stdin_or_dispatch(runner, entry):
    tree = ast.parse(SOURCE.read_bytes())
    node = function(tree, entry)
    assert ast.dump(node.body[0]) == ast.dump(ast.Expr(value=ast.Call(
        func=ast.Name(id="require_release", ctx=ast.Load()), args=[], keywords=[])))
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("getenv", "environ")
                   for n in ast.walk(function(tree, "require_release")))
    # The only invocation of an entry definition is under an explicit false
    # guard in a detached namespace. Its first statement above proves no access
    # can precede that guard; no released/live runner entry is called.
    ns = dict(runner.ns, RELEASE_AUTHORIZED=False)
    exec(compile(ast.Module(body=[function(tree, "require_release"), node],
        type_ignores=[]), str(SOURCE), "exec"), ns)
    required = node.args.posonlyargs + node.args.args
    required = required[:len(required)-len(node.args.defaults)]
    with pytest.raises(ValueError, match="DRAFT_BLOCKED"):
        ns[entry](**{arg.arg: None for arg in required})


def test_declared_source_scope_and_original_case_closure_are_explicit(runner):
    tree = ast.parse(SOURCE.read_bytes())
    old = ast.parse(PREDECESSOR.read_bytes())
    assert ast.dump(function(tree, "approved_cases_from_review")) == ast.dump(function(old, "approved_cases_from_review"))
    assert set(TIMING_KEYS) <= {k.arg for k in assignment(tree, "LIMITS").keywords}
    sources = runner.ns["SOURCES"]
    assert len(sources) == len(set(sources)) == 61
    assert "tests/__init__.py" in sources
    assert "backend/cad_guided_two_plane_context_v2.py" in sources
    assert "scripts/gold11_schedule_projection_v1.py" in sources
    assert "scripts/gold11_contact_scoring_v2.py" in sources
    assert "scripts/run_gold11_mesh_development_v4.py" in sources
    assert "backend/cad_guided_proposal_optimizer_v4.py" not in sources


@pytest.mark.parametrize("key", TIMING_KEYS)
def test_release_rejects_each_missing_boolean_nonfinite_or_nonpositive_time(runner, key):
    runner.ns["RELEASE_AUTHORIZED"] = True
    for bad in (None, False, 0, -1, float("nan"), float("inf"), "20"):
        runner.ns["LIMITS"][key] = bad
        with pytest.raises(ValueError, match="CONCRETE_TIMING_LIMITS_REQUIRED"):
            runner.ns["require_release"]()


@pytest.mark.parametrize("fault", ["case_cpu_float", "scorer_cpu_float", "reserve_equals_total", None])
def test_release_integral_cpu_and_reserved_wall_contract_without_any_access(runner, fault):
    runner.ns["RELEASE_AUTHORIZED"] = True
    if fault == "case_cpu_float": runner.ns["LIMITS"]["case_cpu_seconds"] = 20.
    elif fault == "scorer_cpu_float": runner.ns["LIMITS"]["scorer_cpu_seconds"] = 20.
    elif fault == "reserve_equals_total":
        runner.ns["LIMITS"]["terminalization_reserve_seconds"] = runner.ns["LIMITS"]["total_wall_seconds"]
    if fault:
        with pytest.raises(ValueError, match="INVALID_RELEASE_TIMING_LIMITS"):
            runner.ns["require_release"]()
    else:
        assert runner.ns["require_release"]() is None  # guard only, no entry call


@pytest.fixture
def inert_backend():
    prior = {n: v for n, v in sys.modules.items() if n == "backend" or n.startswith("backend.")}
    for name in prior:
        del sys.modules[name]
    package = ModuleType("backend")
    package.__package__, package.__path__ = "backend", [str(ROOT / "backend")]
    sys.modules["backend"] = package
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "backend" or name.startswith("backend."):
                del sys.modules[name]
        sys.modules.update(prior)


@pytest.mark.parametrize("arm", ["control", "challenger"])
def test_actual_policy_constructors_share_v13_native_v1_free_circle_and_declared_context(runner, inert_backend, arm):
    before = wire(runner.ns["CONFIG"])
    entry, policies = runner.ns["workflow_entry_and_policies"](arm)
    assert entry.__name__ == "inspect_cad_guided_mesh_v13"
    assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV13"
    assert policies["workflow_policy"].native_adapter_version == "v1"
    assert policies["measurement_policy"].fit_model == "free_circle"
    assert policies["position_error_bound_mm"] == .05
    assert policies["two_plane_enabled"] is policies["connected_measurement_contexts"] is True
    assert policies["allow_ocp"] is True  # constructor declaration; entry is NOT called
    assert policies["noisy_policy"].tile_triangles == (128, 1024)
    assert type(policies["initialization_budgets"]).__name__ == "PlaneInitializationBudgetsV12"
    assert policies["initialization_budgets"].max_certificate_pairs == 1_024_000_000
    assert policies["surface_budgets"].max_certificate_pairs == 128_000_000
    assert policies["registration_budgets"].max_certificate_pairs == 8_000_000_000
    request = runner.ns["arm_request"](arm)
    assert request["context_request"]["maximum_ordered_matches"] == 256
    assert request["context_request"]["budgets"]["max_signed_families"] == 1024
    assert request["context_request"]["budgets"]["max_candidate_slots"] == 8192
    if arm == "control":
        assert policies["scan_plane_mode"] == "v7" and policies["two_plane_schedule_mode"] == "v1"
        assert policies["seeded_plane_budgets"] is policies["composite_plane_budgets"] is None
        assert policies["two_plane_schedule_budgets"] is None
        assert request["context_request"]["schema_version"] == "two_plane_context_v1"
        assert request["context_request"]["budgets"]["max_ordered_cad_matches_per_scan_pair"] == 4
    else:
        assert policies["scan_plane_mode"] == "composite_v1" and policies["two_plane_schedule_mode"] == "v2"
        assert type(policies["seeded_plane_budgets"]).__name__ == "SeededPlaneBudgetsV1"
        assert type(policies["composite_plane_budgets"]).__name__ == "CompositeScanPlaneBudgetsV1"
        assert type(policies["two_plane_schedule_budgets"]).__name__ == "TwoPlaneScheduleBudgetsV2"
        assert asdict(policies["two_plane_schedule_budgets"]) == request["schedule_request"]["budgets"]
        assert request["context_request"]["schema_version"] == "two_plane_context_v2"
        assert request["context_request"]["budgets"]["max_ordered_cad_matches_per_scan_pair"] == 256
    assert wire(runner.ns["CONFIG"]) == before
    assert not any(n == "OCP" or n.startswith("OCP.") for n in sys.modules)


@pytest.mark.parametrize("arm", ["control", "challenger"])
def test_full_arm_request_is_detached_and_hash_binds_context(runner, arm):
    first = runner.ns["arm_request"](arm)
    original = runner.ns["arm_request_sha256"](arm)
    first["context_request"]["budgets"]["max_candidate_slots"] = 0
    first["common_config"]["measurement"]["fit_model"] = "not-free-circle"
    assert runner.ns["arm_request_sha256"](arm) == original
    changed = runner.ns["arm_request"](arm)
    changed["context_request"]["schema_version"] = "wrong-context"
    assert sha(wire(changed)) != original


@pytest.mark.parametrize("fault", ["arm", "request_sha", "scan_sha", "extra"])
def test_leaf_declaration_rejects_stale_or_unbound_request(runner, fault):
    binding = dict(arm="challenger", scan_sha256=sha(b"toy-scan"), cad_sha256=sha(b"toy-cad"),
        arm_request_sha256=runner.ns["arm_request_sha256"]("challenger"))
    runner.ns["validate_leaf_binding"]("INFERENCE", binding)
    if fault == "arm": binding["arm"] = "control"
    elif fault == "request_sha": binding["arm_request_sha256"] = sha(b"old")
    elif fault == "scan_sha": binding["scan_sha256"] = "not-a-hash"
    else: binding["family_id"] = "must-not-enter-inference"
    with pytest.raises(ValueError):
        runner.ns["validate_leaf_binding"]("INFERENCE", binding)


def test_scorer_source_and_import_boundary_is_scripts_only(runner, projection):
    names = runner.ns["leaf_sources"]("SCORER")
    assert set(names) == {runner.ns["RUNNER_SOURCE"], runner.ns["ARTIFACT_SOURCE"], *runner.ns["SCORING_SOURCES"]}
    assert not any(n.startswith("backend/") for n in names)
    assert runner.ns["REVIEW"] not in names and runner.ns["PROTOCOL"] not in names
    assert projection._contact.CARRIER == "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
    assert projection._contact.match_lines is projection._contact._legacy.match_lines
    tree = ast.parse(SOURCE.read_bytes())
    scorer = function(tree, "scorer_worker")
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith("backend") for n in ast.walk(scorer))
    assert not any(isinstance(n, ast.Constant) and n.value in ("scan.ply", "cad.step") for n in ast.walk(scorer))


@pytest.mark.parametrize("arm,finite", [("control", True), ("control", False), ("challenger", True), ("challenger", False)])
def test_actual_projection_and_parent_reconstruction_preserve_conditional_or_no_carrier(runner, projection, invented, arm, finite):
    original, retained, receipt = projected(runner, projection, invented, arm, finite=finite)
    answer = runner.ns["reconstruct_scoring_input"](retained, receipt, arm, result_bytes=original)
    assert answer["disposition"] == ("SCORED_CONDITIONAL" if finite else "COMPLETE_NO_LOCALIZED_CARRIERS")
    assert len(answer["predictions"]) == int(finite)
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    assert answer["result_binding"]["supplied_sha256"] == sha(original)
    assert original.startswith(b" \n") and original.endswith(b"\n ")


def test_project_and_revalidate_receive_original_byte_objects_without_reencoding(runner, projection, invented, monkeypatch):
    connect_projection(runner, projection)
    seen = []
    original_project = projection.project_gold11_schedule_workflow_v1
    original_revalidate = projection.revalidate_gold11_schedule_projection_v1
    def project(blob, **kwargs):
        seen.append(("project", blob, kwargs["max_wrapper_bytes"]))
        return original_project(blob, **kwargs)
    def revalidate(blob, **kwargs):
        seen.append(("retained", blob, kwargs["max_wrapper_bytes"]))
        seen.append(("original", kwargs["result_bytes"], kwargs["max_wrapper_bytes"]))
        return original_revalidate(blob, **kwargs)
    monkeypatch.setattr(projection, "project_gold11_schedule_workflow_v1", project)
    monkeypatch.setattr(projection, "revalidate_gold11_schedule_projection_v1", revalidate)
    original, retained, receipt = projected(runner, projection, invented)
    runner.ns["reconstruct_scoring_input"](retained, receipt, "challenger", result_bytes=original)
    assert seen[0][1] is original and seen[1][1] is retained and seen[2][1] is original
    assert all(row[2] == 8 * 1024**2 for row in seen)


@pytest.mark.parametrize("over", [False, True])
def test_actual_retained_wrapper_exact_8mib_including_whitespace_boundary(runner, projection, invented, over):
    original, retained, receipt = projected(runner, projection, invented)
    cap = 8 * 1024**2
    retained += b" " * (cap + int(over) - len(retained))
    receipt["scoring_input_sha256"] = sha(retained)
    if over:
        with pytest.raises(ValueError, match="RETAINED_PROJECTION_BYTE_BUDGET"):
            runner.ns["reconstruct_scoring_input"](retained, receipt, "challenger", result_bytes=original)
    else:
        answer = runner.ns["reconstruct_scoring_input"](retained, receipt, "challenger", result_bytes=original)
        assert answer["disposition"] == "SCORED_CONDITIONAL" and len(retained) == cap


@pytest.mark.parametrize("body_bytes,passes", [(8 * 1024**2 - 1, True), (8 * 1024**2, False)])
def test_prepublication_body_plus_newline_does_not_truncate(body_bytes, passes, runner):
    # Explicit transport mock; not a claim this padding is a valid projector schema.
    result = {"padding": ""}
    result["padding"] = "x" * (body_bytes - len(wire(result)))
    assert len(wire(result)) == body_bytes
    runner.ns["projection_module"] = lambda: SimpleNamespace(project_gold11_schedule_workflow_v1=lambda *_a, **_k: result)
    if passes:
        assert runner.ns["compact_result"](b"{}", expected_result_sha256=sha(b"{}"), arm="control") is result
        assert len(wire(result) + b"\n") == runner.ns["LIMITS"]["compact_case_bytes"]
    else:
        with pytest.raises(ValueError, match="SCORING_PROJECTION_BYTE_BUDGET"):
            runner.ns["compact_result"](b"{}", expected_result_sha256=sha(b"{}"), arm="control")
    assert len(wire(result)) == body_bytes


@pytest.mark.parametrize("fault", ["result_bytes", "result_sha", "wrapper_bytes", "wrapper_sha", "arm", "arm_sha", "kind", "exit_bool"])
def test_unrelated_valid_hashes_or_terminal_mutation_never_reconstruct_positive(runner, projection, invented, fault):
    original, retained, receipt = projected(runner, projection, invented)
    arm = "challenger"
    if fault == "result_bytes": original += b" "
    elif fault == "result_sha": receipt["result_sha256"] = sha(b"another-result")
    elif fault == "wrapper_bytes":
        value = json.loads(retained); value["requested_schedule"]["budgets"]["max_residual_pair_visits"] = 0
        retained = wire(value) + b"\n"; receipt["scoring_input_sha256"] = sha(retained)
    elif fault == "wrapper_sha": receipt["scoring_input_sha256"] = sha(b"another-wrapper")
    elif fault == "arm": arm = "control"
    elif fault == "arm_sha": receipt["arm_request_sha256"] = runner.ns["arm_request_sha256"]("control")
    elif fault == "kind": receipt["projection_kind"] = "SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW"
    else: receipt["exit_code"] = False
    with pytest.raises(ValueError):
        runner.ns["reconstruct_scoring_input"](retained, receipt, arm, result_bytes=original)


@pytest.mark.parametrize("fault", ["last_context", "all_schedule_copies", "initialization", "scan_configuration", "null_v2_on_v1"])
def test_returned_binding_failure_is_whole_arm_refusal_not_a_positive_prefix(runner, projection, invented, fault):
    arm = "control" if fault == "null_v2_on_v1" else "challenger"
    payload = payload_for(invented, arm)
    parents = [payload["policies"], payload["registration"]["evidence"], payload["registration"]["evidence"]["initialization"]]
    if fault == "last_context": payload["registration_array_binding"][-1]["two_plane_schedule_configuration_sha256"] = sha(b"wrong")
    elif fault == "all_schedule_copies":
        for parent in parents: parent["two_plane_schedule_configuration"]["budgets"]["max_residual_pair_visits"] = 0
    elif fault == "initialization": parents[0]["initialization_budgets"]["max_hypothesis_records"] = 1.
    elif fault == "scan_configuration": parents[-1]["scan_plane_configuration"]["seeded_budgets"]["max_seeds"] = 0
    else: parents[0]["two_plane_schedule_configuration"] = None
    connect_projection(runner, projection)
    original = wire(payload)
    outer = runner.ns["compact_result"](original, expected_result_sha256=sha(original), arm=arm)
    assert outer["disposition"] == "REJECTED" and outer["contact_projection"] is None
    retained = wire(outer) + b"\n"
    inner = runner.ns["reconstruct_scoring_input"](retained, clean_receipt(runner, arm, original, retained), arm, result_bytes=original)
    assert runner.ns["valid_supervisor_refusal"](inner) and inner["predictions"] == []
    assert inner["reasons"][0].startswith("RUNNER_SCHEDULE_BINDING_REJECTED:")


@pytest.mark.parametrize("arm", ["control", "challenger"])
def test_early_workflow_refusal_needs_no_fabricated_not_reached_configuration(runner, projection, invented, arm):
    connect_projection(runner, projection)
    original = wire(invented["_refusal"]())
    outer = runner.ns["compact_result"](original, expected_result_sha256=sha(original), arm=arm)
    assert outer["disposition"] == "ADMITTED"
    assert outer["schedule_evidence"]["status"] == "PRESENT_METADATA_CONSISTENT_NOT_COMPLETE_WORKFLOW"
    assert outer["contact_projection"]["disposition"] == "REFUSED"
    assert outer["contact_projection"]["predictions"] == []


@pytest.mark.parametrize("fault", [None, "changed_reason", "positive", "wrong_phase"])
def test_supervisor_failed_case_is_exact_six_key_bound_refusal(runner, fault):
    refused = runner.ns["unavailable"]("INVENTED_KILLED_WORKER")
    receipt = dict(status="TERMINAL", exit_code=-9, budget_stop="INVENTED_WALL_LIMIT",
        arm_request_sha256=runner.ns["arm_request_sha256"]("control"),
        projection_kind="SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW", supervisor_refusal=deepcopy(refused))
    if fault == "changed_reason": refused["reasons"] = ["CHANGED"]
    elif fault == "positive": refused["predictions"] = [{}]
    elif fault == "wrong_phase": refused["phase"] = "COMPLETE"
    blob = wire(refused) + b"\n"; receipt["scoring_input_sha256"] = sha(blob)
    if fault:
        with pytest.raises(ValueError): runner.ns["reconstruct_scoring_input"](blob, receipt, "control")
    else:
        assert runner.ns["reconstruct_scoring_input"](blob, receipt, "control") == refused
        assert set(refused) == {"status", "phase", "reasons", "predictions", "observed_bend_count", "production_use_allowed"}


def _log_receipt(blob):
    return dict(exit_code=0, budget_stop=None, log_created=True,
        forwarded_log_bytes=len(blob), stored_log_bytes=len(blob))


@pytest.fixture
def campaign(runner, projection, invented):
    """Actual full schema and119 science paths; no original cohort metadata."""
    ns = runner.ns
    connect_projection(runner, projection)
    cases = [dict(case_token=f"toy-case-{i:02d}", part_key=f"toy-family-{i % 6}",
        reference_id=f"toy-reference-{i}",
        scan_path=str(runner.tmp/f"NEVER_OPENED_SCAN_{i}.ply"),
        cad_path=str(runner.tmp/f"NEVER_OPENED_CAD_{i % 6}.step"),
        scan_sha256=sha(f"invented-scan-{i}".encode()), cad_sha256=sha(f"invented-cad-{i % 6}".encode())) for i in range(11)]
    source_hashes = {}
    for name in ns["SOURCES"]:
        blob = ("# Invented custody payload, never imported: " + name + "\n").encode()
        source_hashes[name] = sha(blob)
        put(runner.run, "source/" + name, blob)
        put(runner.current, name, blob)
    bundle = dict(schema="GOLD11_MESH_DEVELOPMENT_V5", cases=cases, arms=list(ns["ARMS"]),
        config=deepcopy(ns["CONFIG"]), limits=deepcopy(ns["LIMITS"]),
        catalog_sha256=ns["CATALOG_SHA"], arm_definitions=deepcopy(ns["ARM_DEFINITIONS"]),
        comparator=ns["COMPARATOR"], position_error_declaration=deepcopy(ns["POSITION_ERROR_DECLARATION"]),
        certificate_caps=deepcopy(ns["CERTIFICATE_CAPS"]),
        retained_control_policy=deepcopy(ns["RETAINED_CONTROL_POLICY"]),
        scoring_transport=deepcopy(ns["SCORING_TRANSPORT"]),
        two_plane_declaration=deepcopy(ns["TWO_PLANE_DECLARATION"]),
        artifact_policy=deepcopy(ns["ARTIFACT_POLICY"]),
        historical_comparability="UNESTABLISHED", production_use_allowed=False,
        source_sha256=source_hashes, runtime_metadata=ns["runtime_metadata"](),
        arm_requests=ns["all_arm_requests"]())
    selected = {case["case_token"]: dict(bends=[dict(bend_id=f"toy-ref-{j}",
        line_start=[j * 20, 0, 0], line_end=[j * 20, 0, 10]) for j in range(7 if i < 5 else 6)],
        coordinate_frame="INVENTED_RAW_FRAME") for i, case in enumerate(cases)}
    assert sum(len(v["bends"]) for v in selected.values()) == 71
    # Explicit invented reference binding for semantic parent checks. Source pin
    # preservation is tested independently; no real selected-reference bytes.
    ns["SELECTED_SHA"] = sha(wire(selected))
    bundle["selected_reference_sha256"] = ns["SELECTED_SHA"]
    all_outer, all_inner, receipts, result_hashes = ({arm: {} for arm in ns["ARMS"]} for _ in range(4))
    log = b"invented completed worker\n"
    for case in cases:
        key = case["case_token"]
        for arm in ns["ARMS"]:
            original, retained, receipt = projected(runner, projection, invented, arm)
            outer = json.loads(retained)
            inner = ns["reconstruct_scoring_input"](retained, receipt, arm, result_bytes=original)
            binding = dict(arm=arm, scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"],
                arm_request_sha256=ns["arm_request_sha256"](arm))
            dispatch_id = sha(wire(dict(role="INFERENCE", binding=binding)))
            receipt.update(_log_receipt(log), source_binding_valid=True, inference_started=True,
                worker_dispatch_attempted=True, leaf_dispatch_id=dispatch_id,
                derived_result_retention_complete=True,
                retained_result_artifacts=dict(status="COMPLETE_RESULT_BYTES_RETAINED",
                    artifacts=[dict(path="RESULT.json", sha256=sha(original), bytes=len(original))]),
                workflow_stage_diagnostics=ns["workflow_stage_diagnostics"](json.loads(original), arm))
            prefix = f"cases/{key}/{arm}/"
            start = dict(case_id=key, arm=arm, limits=ns["LIMITS"], no_resume=True,
                arm_request_sha256=ns["arm_request_sha256"](arm))
            for name, blob in (("STARTED.json", wire(start) + b"\n"), ("COMPLETED.json", wire(receipt) + b"\n"),
                    ("RESULT.json", original), ("SCORING_INPUT.json", retained), ("OUTPUT.txt", log)):
                put(runner.run, prefix + name, blob)
            all_outer[arm][key], all_inner[arm][key], receipts[arm][key] = outer, inner, receipt
            result_hashes[arm][key] = sha(original)
    terminal = dict(case_count=11, arm_count=2, terminal_record_count=22,
        outcomes_sha256=sha(wire(all_outer)), result_bindings=result_hashes,
        arm_requests_sha256=sha(wire(ns["all_arm_requests"]())))
    request = ns["make_scoring_request"](bundle, selected, all_inner, receipts)
    contact_score = projection._contact.score_gold11_contact_workflows_v2(**ns["scorer_arguments"](request))
    for arm in ns["ARMS"]:
        assert contact_score[arm]["geometry"]["true_positive"] == 11
        assert contact_score[arm]["geometry"]["false_negative"] == 60
        assert contact_score[arm]["geometry"]["false_positive"] == 0
    original_score = wire(contact_score)
    score = dict(contact_score, experiment_schema="GOLD11_MESH_DEVELOPMENT_V5",
        arm_definitions=ns["ARM_DEFINITIONS"], comparator=ns["COMPARATOR"],
        arm_requests=ns["all_arm_requests"](), scoring_transport=ns["SCORING_TRANSPORT"],
        retained_control_policy=ns["RETAINED_CONTROL_POLICY"], certificate_caps=ns["CERTIFICATE_CAPS"],
        two_plane_declaration=ns["TWO_PLANE_DECLARATION"], artifact_policy=ns["ARTIFACT_POLICY"],
        combined_change_attribution=True, accepted_registration_claim_allowed=False,
        control_identity="FRESH_V13_V7_SCAN_PLANES_SCHEDULE_V1_NATIVE_V1",
        challenger_identity="FRESH_V13_COMPOSITE_SCAN_PLANES_SCHEDULE_V2_NATIVE_V1",
        position_error_bound_mm=ns["CONFIG"]["position_error_bound_mm"],
        position_error_declaration=ns["POSITION_ERROR_DECLARATION"],
        workflow_stage_attribution={arm: {key: receipt["workflow_stage_diagnostics"]
            for key, receipt in rows.items()} for arm, rows in receipts.items()},
        phase_attribution_caveat="MISSING_OR_KILLED_INTERNAL_STAGE_UNKNOWN_USE_RETURNED_NESTED_LEDGERS",
        all_declared_input_bytes_verified=True, source_before_after_consistent=True,
        case_declared_coordinate_frames={key: row["coordinate_frame"] for key, row in selected.items()},
        declared_frame_interpretation="PROVISIONAL_COMMON_RAW_FRAME_NOT_VERIFIED",
        bundle_sha256=sha(wire(bundle)), reference_snapshot_sha256=ns["SELECTED_SHA"],
        scorer_result_sha256=sha(original_score), scoring_request_sha256=sha(wire(request)),
        projection_bindings=request["projection_bindings"])
    root_payloads = {
        "COORDINATOR_COMPLETED.json": wire(dict(status="COMPUTED_PENDING_INDEPENDENT_POST_AUDIT")) + b"\n",
        "COORDINATOR_OUTPUT.txt": log, "INFERENCE_TERMINAL.json": wire(terminal) + b"\n",
        "SCORER_RECEIPT.json": wire(_log_receipt(log)) + b"\n", "SCORER_OUTPUT.txt": log,
        "SCORER_ARTIFACT_RETENTION.json": wire(dict(status="COMPLETE_RESULT_BYTES_RETAINED",
            artifacts=[dict(path="SCORER_RESULT.json", sha256=sha(original_score), bytes=len(original_score))])) + b"\n",
        "SCORER_RESULT.json": original_score, "SCORED.json": wire(score) + b"\n",
        "SELECTED_REFERENCES.json": wire(selected),
    }
    assert len(root_payloads) == 9
    for name, blob in root_payloads.items(): put(runner.run, name, blob)
    coordinator_pid = 4242  # Invented metadata; never queried or signaled.
    audit = dict(decision="INVENTED_NO_AUTHORITY_NOT_RELEASE", bundle_sha256=sha(wire(bundle)),
        reviewer="INVENTED_FIXTURE", blocking_findings=["NOT_EXECUTION_AUTHORITY"])
    start = dict(bundle_sha256=sha(wire(bundle)), audit_sha256=sha(wire(audit)),
        status="STARTED_NO_RETRY", limits=ns["LIMITS"],
        arm_requests_sha256=sha(wire(ns["all_arm_requests"]())),
        timing_scope="SOURCE_PREFLIGHT_THEN_CAMPAIGN_INCLUDING_FROZEN_TERMINAL_RESERVE")
    dispatch = dict(parent_pid=4141, token_sha256=sha(b"invented-no-capability-token"))
    context = dict(schema=ns["LEAF_SCHEMA"], coordinator_pid=coordinator_pid, outer_pid=4141,
        bundle_sha256=sha(wire(bundle)), audit_sha256=sha(wire(audit)),
        started_sha256=sha(wire(start)), coordinator_dispatch_sha256=sha(wire(dispatch)),
        source_sha256=source_hashes, runtime_metadata=bundle["runtime_metadata"])
    controls = {"FROZEN.json": wire(bundle) + b"\n", "PRE_RUN_AUDIT.json": wire(audit) + b"\n",
        "STARTED.json": wire(start) + b"\n", "COORDINATOR_DISPATCH.json": wire(dispatch) + b"\n",
        "COORDINATOR_VERIFIED.json": wire(context) + b"\n"}
    control_bindings = {name: sha(blob) for name, blob in controls.items()}
    for name, blob in controls.items(): put(runner.run, name, blob)
    leaves = [("INFERENCE", dict(arm=arm, scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"],
        arm_request_sha256=ns["arm_request_sha256"](arm))) for case in cases for arm in ns["ARMS"]]
    leaves.append(("SCORER", dict(request_sha256=sha(wire(request)))))
    assert len(leaves) == 23
    for i, (role, binding) in enumerate(leaves):
        identifier = sha(wire(dict(role=role, binding=binding)))
        issued = dict(schema=ns["LEAF_SCHEMA"], role=role, binding=binding,
            cache=str(runner.tmp / f"never-opened-cache-{i}"), cache_identity=dict(device=1, inode=i+1, uid=os.getuid()),
            coordinator_pid=coordinator_pid, context_sha256=sha(wire(context)), token_sha256=sha(f"invented-token-{i}".encode()))
        claimed = dict(worker_pid=5000+i, coordinator_pid=coordinator_pid, role=role,
            dispatch_sha256=sha(wire(issued)), no_retry=True)
        put(runner.run, f"leaf_dispatch/{identifier}.json", wire(issued) + b"\n")
        put(runner.run, f"leaf_dispatch/{identifier}.CLAIMED.json", wire(claimed) + b"\n")
    (runner.run / "coordinator_work").mkdir()
    footprint = sum(p.stat().st_size for p in runner.tmp.rglob("*") if p.is_file())
    assert footprint < MAX_FIXTURE_BYTES
    runner.bundle, runner.controls, runner.supervisor = bundle, control_bindings, _log_receipt(log)
    runner.outcomes, runner.inner, runner.receipts, runner.request = all_outer, all_inner, receipts, request
    runner.selected, runner.score, runner.science_names = selected, score, set(root_payloads) | {
        f"cases/{case['case_token']}/{arm}/{name}" for case in cases for arm in ns["ARMS"]
        for name in ("STARTED.json", "COMPLETED.json", "RESULT.json", "SCORING_INPUT.json", "OUTPUT.txt")}
    assert len(runner.science_names) == 119
    return runner


def parent_evidence(campaign):
    evidence = {}
    campaign.ns["verify_parent_artifacts"](campaign.bundle, campaign.supervisor, evidence)
    assert evidence["complete"] is True and set(evidence["files"]) == campaign.science_names
    return evidence


def inventory(campaign, evidence, **kwargs):
    return campaign.ns["verify_campaign_inventory"](campaign.bundle, evidence, campaign.controls, **kwargs)


def test_full119_campaign_semantics_and_three_real_publication_inventory_phases(campaign):
    ns = campaign.ns
    evidence = parent_evidence(campaign)
    first = inventory(campaign, evidence)
    assert first["complete"] is True and first["errors"] == []
    assert first["scientific_entries"] == 119 and first["schema_directories_not_attempted"] == []
    assert first["scientific_entry_limit"] == 128 and first["schema_read_byte_bound"] <= 8 * 1024**3
    assert sum("neutral_control_binding" in row for row in first["entries"].values()) == 46
    terminal = ns["terminalize_case_records"](campaign.bundle, "NO_REWRITE_VALID_TERMINALS")
    assert len(terminal) == 22 and all(row["durable"] for row in terminal)
    assert all(record["disposition"] == "PREEXISTING_VALID_NO_WRITE"
        for row in terminal for record in row["records"].values())
    source_audit = {}
    ns["final_source_audit"](campaign.bundle, source_audit, control_bindings=campaign.controls)
    assert source_audit["complete"] is source_audit["unchanged"] is True
    evidence["campaign_tree_inventory"] = first
    completed = dict(status="COMPUTED_PENDING_INDEPENDENT_POST_AUDIT", coordinator_receipt=campaign.supervisor,
        failures=[], terminal_case_records=terminal, final_source_audit=source_audit,
        owned_artifact_custody=evidence, parent_wall_seconds=0., parent_peak_rss_bytes=0,
        retained_coordinator_work_directory="coordinator_work",
        release_requires="OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT",
        no_resume=True, production_use_allowed=False)
    # This required full envelope may fail: no small replacement or favorable truncation.
    assert len(wire(completed)) + 1 <= 256 * 1024
    ns["exclusive_json"](campaign.run / "COMPLETED.json", completed)
    published = {"COMPLETED.json": dict(sha256=sha(wire(completed)+b"\n"), bytes=len(wire(completed))+1)}
    second = inventory(campaign, evidence, published=published, previous=first)
    assert second["complete"] is True and second["scientific_entries"] == 119
    gate = dict(status="PENDING_INDEPENDENT_POST_AUDIT", completed_canonical_sha256=sha(wire(completed)),
        source_and_control_audit_sha256=sha(wire(source_audit)), owned_artifact_custody_sha256=sha(wire(evidence)),
        scored_report_sha256=evidence["scored_report_sha256"], scorer_result_sha256=evidence["scorer_result_sha256"],
        reference_snapshot_sha256=ns["SELECTED_SHA"], completed_publication_inventory_sha256=sha(wire(second)),
        final_gate_self_hash_claim=False)
    ns["exclusive_json"](campaign.run / "FINAL_GATE.json", gate)
    published["FINAL_GATE.json"] = dict(sha256=sha(wire(gate)+b"\n"), bytes=len(wire(gate))+1)
    third = inventory(campaign, evidence, published=published, previous=second)
    assert third["complete"] is True and third["scientific_entries"] == 119
    assert third["observed_entries"] == first["observed_entries"] + 2
    assert not any(".partial-" in p.name for p in campaign.run.rglob("*"))


@pytest.mark.parametrize("fault", ["dotfile", "aliased_partial", "separate_same_bytes_partial", "wrong_claim", "wrong_issued_arm", "source_bytes", "symlink_source", "missing_scorer"])
def test_actual_full_campaign_inventory_vetoes_every_unknown_or_unbound_entry(campaign, fault):
    evidence = parent_evidence(campaign)
    prefix = "cases/toy-case-00/control/"
    if fault == "dotfile": put(campaign.run, ".unmanifested", b"not-read")
    elif fault in ("aliased_partial", "separate_same_bytes_partial"):
        target = campaign.run / (prefix + ".SCORING_INPUT.json.partial-0000000000000000")
        source = campaign.run / (prefix + "SCORING_INPUT.json")
        if fault == "aliased_partial": os.link(source, target)
        else: target.write_bytes(source.read_bytes())
    elif fault in ("wrong_claim", "wrong_issued_arm"):
        names = sorted((campaign.run / "leaf_dispatch").glob("*.json"))
        target = next(p for p in names if (p.name.endswith(".CLAIMED.json")) == (fault == "wrong_claim"))
        value = json.loads(target.read_bytes())
        if fault == "wrong_claim": value["dispatch_sha256"] = sha(b"wrong-dispatch")
        else: value["binding"] = {**value["binding"], "arm_request_sha256": sha(b"wrong-arm")}
        target.write_bytes(wire(value)+b"\n")
    elif fault == "source_bytes": replace_owned(campaign.run, "source/"+campaign.ns["SOURCES"][0], b"changed")
    elif fault == "symlink_source":
        target = campaign.run / ("source/"+campaign.ns["SOURCES"][0]); target.unlink()
        target.symlink_to(campaign.current / campaign.ns["SOURCES"][0])
    else: (campaign.run / "SCORER_RESULT.json").unlink()
    result = inventory(campaign, evidence)
    assert result["complete"] is False
    assert result["errors"] or result.get("failure_type")
    assert not (campaign.run / "FINAL_GATE.json").exists()
    if fault in ("dotfile", "aliased_partial", "separate_same_bytes_partial"):
        assert result["scientific_entries"] == 120
        assert any(row["code"] == "UNDECLARED_CAMPAIGN_NAME" for row in result["errors"])


@pytest.mark.parametrize("fault", ["last_wrapper", "last_receipt", "enriched_metric", "raw_result"])
def test_parent_full_reconstruction_vetoes_late_custody_corruption_without_rewriting(campaign, fault):
    prefix = "cases/toy-case-10/challenger/"
    if fault == "last_wrapper":
        p = campaign.run/(prefix+"SCORING_INPUT.json"); value = json.loads(p.read_bytes())
        value["disposition"] = "REJECTED"; replace_owned(campaign.run, prefix+"SCORING_INPUT.json", value)
    elif fault == "last_receipt":
        p = campaign.run/(prefix+"COMPLETED.json"); value = json.loads(p.read_bytes())
        value["arm_request_sha256"] = campaign.ns["arm_request_sha256"]("control")
        replace_owned(campaign.run, prefix+"COMPLETED.json", value)
    elif fault == "enriched_metric":
        value = deepcopy(campaign.score); value["challenger"]["geometry"]["true_positive"] += 1
        replace_owned(campaign.run, "SCORED.json", value)
    else: replace_owned(campaign.run, prefix+"RESULT.json", b'{"changed":"still-valid-json"}')
    before = {str(p.relative_to(campaign.run)): sha(p.read_bytes()) for p in campaign.run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError): campaign.ns["verify_parent_artifacts"](campaign.bundle, campaign.supervisor, {})
    after = {str(p.relative_to(campaign.run)): sha(p.read_bytes()) for p in campaign.run.rglob("*") if p.is_file()}
    assert before == after and len(campaign.science_names) == 119


@pytest.mark.parametrize("terminals", [21, 22])
def test_references_are_unreachable_until_all22_actual_durable_reconstructions(campaign, terminals):
    ns = campaign.ns
    if terminals == 21:
        (campaign.run / "cases/toy-case-10/challenger/COMPLETED.json").unlink()
    ns["verify_bundle"] = lambda: campaign.bundle  # no physical source/control read
    observed = []
    class ReachedReferences(Exception): pass
    def reference_seam(cases):
        observed.append(deepcopy(cases))
        raise ReachedReferences("Stop before any reference/scorer access")
    ns["selected_references"] = reference_seam
    expected = FileNotFoundError if terminals == 21 else ReachedReferences
    with pytest.raises(expected): ns["score_after_terminal"](campaign.bundle, campaign.outcomes, campaign.receipts)
    assert len(observed) == int(terminals == 22)
    assert not (campaign.run / "SCORING_UNAVAILABLE.json").exists()


@pytest.mark.parametrize("stage", ["after_completed", "after_final_gate"])
def test_late_equal_byte_inode_change_vetoes_later_inventory_phase(campaign, stage):
    evidence = parent_evidence(campaign)
    prior = inventory(campaign, evidence)
    assert prior["complete"] is True
    published = {}
    for name in ("COMPLETED.json", "FINAL_GATE.json"):
        value = dict(status="INVENTED_PHASE", phase=name)
        campaign.ns["exclusive_json"](campaign.run/name, value)
        published[name] = dict(sha256=sha(wire(value)+b"\n"), bytes=len(wire(value))+1)
        if (name == "COMPLETED.json") == (stage == "after_completed"):
            target = campaign.run / "cases/toy-case-00/control/SCORING_INPUT.json"
            replacement = campaign.tmp / "invented-replacement"
            replacement.write_bytes(target.read_bytes()); replacement.replace(target)
            failed = inventory(campaign, evidence, published=published, previous=prior)
            assert failed["complete"] is False
            assert any(row["code"] == "PRIOR_INVENTORY_ENTRY_CHANGED_AFTER_PUBLICATION" for row in failed["errors"])
            break
        prior = inventory(campaign, evidence, published=published, previous=prior)
        assert prior["complete"] is True


def test_all_worker_refusals_and_no_carrier_remain_in_scorer_denominator(campaign, projection):
    request = deepcopy(campaign.request)
    refused = campaign.ns["unavailable"]("INVENTED_ALL_ARM_FAILURE")
    for arm in campaign.ns["ARMS"]:
        for key in request[arm+"_results_by_case"]:
            request[arm+"_results_by_case"][key] = deepcopy(refused)
            row = request["projection_bindings"][arm][key]
            row.update(projection_kind="SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW",
                scorer_input_canonical_sha256=sha(wire(refused)))
    score = projection._contact.score_gold11_contact_workflows_v2(**campaign.ns["scorer_arguments"](request))
    for arm in campaign.ns["ARMS"]:
        assert len(score[arm]["case_dispositions"]) == 11
        assert score[arm]["geometry"]["true_positive"] == score[arm]["geometry"]["false_positive"] == 0
        assert score[arm]["geometry"]["false_negative"] == 71
        assert score[arm]["physical_observed_bend_count"]["unknown_case_count"] == 11
    assert score["historical_score_beaten_claim_allowed"] is False


@pytest.mark.parametrize("kind", ["localized", "no_carrier", "killed", "malformed"])
def test_actual_case_orchestration_retains_result_or_whole_refusal_with_fake_process_only(runner, projection, invented, kind):
    ns = runner.ns
    connect_projection(runner, projection)
    reads, calls = [], []
    real_read = ns["_bounded_read"]
    def read(path, cap, expected=None, **kwargs):
        if path in ("invented-scan-token", "invented-cad-token"):
            reads.append(path)
            blob = b"invented-not-parsed-scan" if path == "invented-scan-token" else b"invented-not-parsed-cad"
            assert sha(blob) == expected and len(blob) <= cap
            return blob
        assert Path(path).is_relative_to(runner.tmp)
        return real_read(path, cap, expected, **kwargs)
    ns["_bounded_read"] = read
    ns["coordinator_memory_check"] = lambda: None
    ns["runner_command"] = lambda mode: ["INVENTED_NO_SUBPROCESS", mode]
    ns["tempfile"] = SimpleNamespace(TemporaryDirectory=lambda **kw: tempfile.TemporaryDirectory(dir=runner.tmp, **kw))
    ns["issue_leaf_dispatch"] = lambda role, cache, **binding: dict(dispatch_id=sha(wire(dict(role=role,binding=binding))), token="0"*64)
    original = (b'{"not":"a-workflow"}' if kind == "malformed" else
        b" \n"+wire(payload_for(invented, "challenger", finite=kind != "no_carrier"))+b"\n")
    def supervise(command, request, **kwargs):
        calls.append((command, request, kwargs))
        if kind != "killed": put(kwargs["cache"], "RESULT.json", original)
        put(kwargs["log_path"].parent, kwargs["log_path"].name, b"invented child output\n")
        return dict(_log_receipt(b"invented child output\n"),
            exit_code=-9 if kind == "killed" else 0, budget_stop="INVENTED_WALL" if kind == "killed" else None)
    ns["supervise_process"] = supervise
    case = dict(case_token="toy-one", scan_path="invented-scan-token", cad_path="invented-cad-token",
        scan_sha256=sha(b"invented-not-parsed-scan"), cad_sha256=sha(b"invented-not-parsed-cad"))
    destination = runner.run / "cases/toy-one/challenger"
    outer, receipt = ns["supervise_case"](case, "challenger", destination, 100.)
    assert reads == ["invented-scan-token", "invented-cad-token"] and len(calls) == 1
    retained = (destination / "SCORING_INPUT.json").read_bytes()
    saved_result = (destination / "RESULT.json").read_bytes() if kind != "killed" else None
    inner = ns["reconstruct_scoring_input"](retained, receipt, "challenger", result_bytes=saved_result)
    if kind == "localized": assert inner["disposition"] == "SCORED_CONDITIONAL" and len(inner["predictions"]) == 1
    elif kind == "no_carrier": assert inner["disposition"] == "COMPLETE_NO_LOCALIZED_CARRIERS" and inner["predictions"] == []
    else: assert ns["valid_supervisor_refusal"](inner) and inner["predictions"] == []
    if kind != "killed": assert saved_result == original and sha(saved_result) == receipt["result_sha256"]
    else:
        assert receipt["retained_result_artifacts"]["status"] == "NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN"
        assert receipt["workflow_stage_diagnostics"]["evidence"] == "NO_RETURNED_WORKFLOW_INTERNAL_STAGES_UNKNOWN"
    assert (destination / "COMPLETED.json").is_file()


def test_actual_inventory_functions_not_replaced_and_finalization_order_static():
    tree = ast.parse(SOURCE.read_bytes())
    node = function(tree, "execute")
    phases = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name): continue
        if call.func.id == "verify_campaign_inventory": phases.append((call.lineno, "inventory"))
        elif call.func.id == "exclusive_json":
            names = [n.value for n in ast.walk(call.args[0]) if isinstance(n, ast.Constant)]
            if names in (["COMPLETED.json"], ["FINAL_GATE.json"]): phases.append((call.lineno, names[0]))
    assert [name for _, name in sorted(phases)] == ["inventory", "COMPLETED.json", "inventory", "FINAL_GATE.json", "inventory"]
    assert {"_campaign_schema", "_verify_tree_leaf_bindings", "verify_campaign_inventory"} <= SELECTED
    test_tree = ast.parse(Path(__file__).read_bytes())
    forbidden = {"_campaign_schema", "_verify_tree_leaf_bindings", "verify_campaign_inventory"}
    for assigned in ast.walk(test_tree):
        if isinstance(assigned, ast.Assign):
            for target in assigned.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    assert target.slice.value not in forbidden
