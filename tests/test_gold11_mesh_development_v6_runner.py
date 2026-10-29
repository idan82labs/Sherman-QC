"""Prospective, invented-only V6 runner boundary tests; source authored, not run.

No full runner or backend module is imported. Explicit AST definitions execute
only in detached namespaces whose process, authority, I/O and clock seams are
inert spies. The real release flags stay false/unreleased in those namespaces;
an explicit local guard stub admits only the selected mocked operation. Separate
guard-only tests use local copies to test refusal, never a released entrypoint.
Source reads are a fixed named allowlist. No fixture opens an attempt, raw input,
private cache, native library, real process, signal, timer, or reference catalog.
Fixture payloads are bounded below 100 KiB. This is not maximum-campaign envelope,
native integration, telemetry, physical timing, or accuracy qualification.
"""

import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_gold11_mesh_development_v6.py"
OLD = ROOT / "scripts/run_gold11_mesh_development_v5.py"
RUNNER_SHA = "7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80"
OLD_SHA = "71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481"
MAX_FIXTURE_BYTES = 100 * 1024
SOURCE_READ_ALLOWLIST = {SOURCE, OLD,
    ROOT / "backend/cad_guided_two_plane_context_v1.py",
    ROOT / "backend/cad_guided_two_plane_context_v2.py",
    ROOT / "backend/cad_guided_two_plane_context_v3.py"}
CONTEXT_SOURCE_SHA = {
    "cad_guided_two_plane_context_v1.py": "9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a",
    "cad_guided_two_plane_context_v2.py": "d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663",
    "cad_guided_two_plane_context_v3.py": "9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d",
}
CONSTANTS = {
    "ROOT", "RUN", "PROTOCOL", "REVIEW", "REVIEW_SHA", "CATALOG", "CATALOG_SHA",
    "SELECTED_SHA", "SOURCES", "RELEASE_AUTHORIZED", "TIMING_RELEASE_STATUS",
    "STAGE_TELEMETRY_STATUS", "STAGE_SAMPLER_SOURCE", "STAGE_SAMPLER_SHA",
    "_STAGE_SAMPLER_MODULE", "STAGE_SAMPLE_SOURCE_PATHS", "STAGE_LOG_DECLARATION", "PENDING_THREE_CASE_DIAGNOSTIC", "ARMS",
    "ARM_DEFINITIONS", "COMPARATOR", "POSITION_ERROR_DECLARATION", "LIMITS",
    "CERTIFICATE_CAPS", "CONFIG", "TWO_PLANE_DECLARATION", "CONTEXT_V3_REQUEST",
    "CONTEXT_REQUESTS_BY_ARM", "SEEDED_PLANE_BUDGETS", "COMPOSITE_PLANE_BUDGETS",
    "SCHEDULE_V2_BUDGETS", "SCORING_SOURCES", "SCORING_TRANSPORT", "ARTIFACT_SOURCE",
    "ARTIFACT_SOURCE_SHA", "ARTIFACT_POLICY", "SUPERVISION_SOURCE",
    "SUPERVISION_SOURCE_SHA", "_COORDINATOR_WRITERS_QUIESCENT", "_COORDINATOR_ABORTED",
    "LEAF_SCHEMA", "RUNNER_SOURCE",
}
PURE = {
    "digest", "encoded", "unique_object", "decode", "unavailable", "is_hex",
    "require_release", "arm_request", "arm_request_sha256", "all_arm_requests",
    "scan_plane_requests", "projection_request", "compact_result", "leaf_sources",
    "reconstruct_scoring_input", "valid_supervisor_refusal", "writers_quiescent",
    "contained", "cache_identity", "fresh_private_cache", "owned_leaf_cache",
    "retain_worker_result", "supervise_process", "coordinator_memory_check",
    "workflow_entry_and_policies", "score_after_terminal", "stage_sampler_module",
    "engine_call_with_stage_sampler", "_campaign_schema",
    "_tree_fingerprint", "verify_campaign_inventory", "_verify_tree_leaf_bindings",
}
ENTRIES = (
    "main", "freeze", "execute", "worker", "scorer_worker", "internal_coordinator",
    "authorize_leaf", "verify_bundle", "supervise_process", "supervise_case",
    "workflow_entry_and_policies", "selected_references", "score_after_terminal",
    "fresh_private_cache", "stage_sampler_module", "engine_call_with_stage_sampler",
)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def value_node(tree, name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def source_tree(path, expected=None):
    # Only this module's explicitly named source paths call this helper.
    assert path in SOURCE_READ_ALLOWLIST
    blob = path.read_bytes()
    assert len(blob) <= 256 * 1024
    expected = expected if expected is not None else CONTEXT_SOURCE_SHA[path.name]
    if expected is not None:
        assert sha(blob) == expected
    return ast.parse(blob, filename=str(path))


def inject(ns, tree, names):
    nodes = [fn(tree, name) for name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<V6-INVENTED-AST-ONLY>", "exec"), ns)


def forbidden(*_args, **_kwargs):
    raise AssertionError("UNMOCKED_AUTHORITY_IO_PROCESS_OR_NATIVE_SEAM")


class LexicalPath(type(Path())):
    """Path arithmetic only for fake roots; never query host path resolution."""
    def resolve(self, *_a, **_k):
        return self

    def lstat(self, *_a, **_k):
        return forbidden()

    def mkdir(self, *_a, **_k):
        return forbidden()


@contextmanager
def inert_deadline(_seconds=None):
    yield


@pytest.fixture
def sandbox():
    tree = source_tree(SOURCE, RUNNER_SHA)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in n.targets)]
    actual = {t.id for n in nodes for t in n.targets if isinstance(t, ast.Name)}
    assert actual == CONSTANTS
    ns = dict(__builtins__=dict(vars(builtins)), Path=LexicalPath, json=json,
        hashlib=hashlib, math=math, stat=stat, contextmanager=contextmanager)
    ns["__builtins__"].update(open=forbidden, __import__=forbidden)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<V6-CONSTANTS-ONLY>", "exec"), ns)
    inject(ns, tree, sorted(PURE))
    # These are lexical path values only: no mkdir/stat/open occurs.
    ns.update(ROOT=LexicalPath("/invented/source"), RUN=LexicalPath("/invented/campaign"),
        os=SimpleNamespace(getuid=lambda: 123, getpid=lambda: 22, getppid=lambda: 11,
            getpgrp=lambda: 22, sep="/"),
        sys=SimpleNamespace(platform="darwin"),
        time=SimpleNamespace(monotonic=lambda: 0.), deadline=inert_deadline,
        shutil=SimpleNamespace(rmtree=forbidden), tempfile=SimpleNamespace(mkdtemp=forbidden),
        resource=SimpleNamespace(RUSAGE_SELF=0, getrusage=forbidden),
        supervision_module=forbidden, verify_bundle=forbidden, selected_references=forbidden,
        bounded_read=forbidden, _bounded_read=forbidden, owned_read=forbidden,
        publish_bytes=forbidden, exclusive_json=forbidden, issue_leaf_dispatch=forbidden,
        activate_coordinator=forbidden, initialize_worker=forbidden,
        runner_command=forbidden, projection_module=forbidden, print=lambda *_a, **_k: None)
    return SimpleNamespace(ns=ns, tree=tree, events=[])


def admit_mock(sandbox):
    """Explicit detached-only guard stub: literal release values never changed."""
    ns = sandbox.ns
    assert ns["RELEASE_AUTHORIZED"] is False
    assert ns["TIMING_RELEASE_STATUS"] == "UNRELEASED_V5_CARRYOVER_VALUES_NOT_V6_APPROVAL"
    ns["require_release"] = lambda: sandbox.events.append("detached-mock-guard")


def receipt(**changes):
    result = dict(supervisor_schema="GOLD11_OWNED_PROCESS_SUPERVISION_V3",
        direct_child_reaped=True, owned_group_empty_confirmed=True,
        requires_process_attention=False, observed_foreign_group_escape_detected=False,
        observed_foreign_group_escape_unconfirmed_after_cleanup=False,
        observed_identity_unknown_still_present=False, cleanup_errors=[],
        cleanup_errors_discarded=0, control_close_errors=[], exit_code=0, budget_stop=None)
    result.update(changes)
    assert len(wire(result)) < MAX_FIXTURE_BYTES
    return result


def bundle_for(ns):
    result = dict(arms=["control", "challenger"],
        cases=[dict(case_token=f"invented_{i:02}", scan_sha256=sha(f"scan{i}".encode()),
            cad_sha256=sha(f"cad{i % 6}".encode())) for i in range(11)],
        source_sha256={name: "a" * 64 for name in ns["SOURCES"]}, runtime_metadata={"invented": True})
    assert len(wire(result)) < MAX_FIXTURE_BYTES
    return result


PRESERVED_CONSTANTS = (
    "CONFIG", "LIMITS", "CERTIFICATE_CAPS", "REVIEW_SHA", "CATALOG_SHA", "SELECTED_SHA",
    "POSITION_ERROR_DECLARATION", "SEEDED_PLANE_BUDGETS", "COMPOSITE_PLANE_BUDGETS",
)


@pytest.mark.parametrize("name", PRESERVED_CONSTANTS)
def test_preserved_science_limits_and_reference_pins(name):
    old, new = source_tree(OLD, OLD_SHA), source_tree(SOURCE, RUNNER_SHA)
    assert ast.dump(value_node(old, name)) == ast.dump(value_node(new, name))


@pytest.mark.parametrize("name", ("workflow_stage_diagnostics", "selected_references", "scorer_worker"))
def test_preserved_returned_diagnostics_reference_selector_and_scorer(name):
    old, new = source_tree(OLD, OLD_SHA), source_tree(SOURCE, RUNNER_SHA)
    left, right = deepcopy(fn(old, name)), deepcopy(fn(new, name))
    if name == "selected_references":
        assert isinstance(right.body[0], ast.Expr) and right.body[0].value.func.id == "require_release"
        right.body.pop(0)
    assert ast.dump(left) == ast.dump(right)


@pytest.mark.parametrize("entry", ENTRIES)
def test_release_guard_precedes_every_entry_effect(sandbox, entry):
    node = fn(sandbox.tree, entry)
    statements = [n for n in node.body if not isinstance(n, ast.Global)
        and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                 and isinstance(n.value.value, str))]
    assert ast.dump(statements[0]) == ast.dump(ast.Expr(value=ast.Call(
        func=ast.Name(id="require_release", ctx=ast.Load()), args=[], keywords=[])))
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("getenv", "environ")
        for n in ast.walk(fn(sandbox.tree, "require_release")))
    inject(sandbox.ns, sandbox.tree, [entry])
    args = [None] * (len(node.args.posonlyargs) + len(node.args.args) - len(node.args.defaults))
    kwargs = {arg.arg: None for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
              if default is None}
    with pytest.raises(ValueError, match="V6_DRAFT_BLOCKED"):
        sandbox.ns[entry](*args, **kwargs)
    assert sandbox.events == []


@pytest.mark.parametrize("flag", (False, None, 0, 1, "true", True))
def test_local_guard_copy_has_no_truthiness_or_single_flag_bypass(sandbox, flag):
    ns = dict(sandbox.ns, RELEASE_AUTHORIZED=flag)
    inject(ns, sandbox.tree, ["require_release"])
    expected = "V6_TIMING_DECLARATION_NOT_RELEASED" if flag is True else "V6_DRAFT_BLOCKED"
    with pytest.raises(ValueError, match=expected):
        ns["require_release"]()
    assert sandbox.ns["RELEASE_AUTHORIZED"] is False


@pytest.mark.parametrize("key,bad", (("case_cpu_seconds", True), ("case_wall_seconds", float("nan")),
    ("scorer_cpu_seconds", 1.5), ("terminalization_reserve_seconds", 104400)))
def test_local_guard_only_copy_rejects_invalid_timing(sandbox, key, bad):
    ns = dict(sandbox.ns, RELEASE_AUTHORIZED=True, TIMING_RELEASE_STATUS="REVIEWED_V6_EXACT_TIMING_RELEASE",
              LIMITS=deepcopy(sandbox.ns["LIMITS"]))
    ns["LIMITS"][key] = bad
    inject(ns, sandbox.tree, ["require_release"])
    with pytest.raises(ValueError, match="V6_.*TIMING"):
        ns["require_release"]()
    assert sandbox.ns["RELEASE_AUTHORIZED"] is False


@pytest.mark.parametrize("arm", ("control", "challenger"))
def test_full_context_request_and_original_bytes_reach_v2_projection(sandbox, arm):
    ns = sandbox.ns
    expected = ns["arm_request"](arm)
    assert expected["two_plane_context_mode"] == "v3"
    assert expected["implementation"] == "CAD_GUIDED_MESH_OFFLINE_V14_CONTEXT_V3"
    assert expected["context_request"] == ns["CONTEXT_V3_REQUEST"]
    assert expected["schedule_request"]["mode"] == ("v1" if arm == "control" else "v2")
    expected["context_request"]["budgets"]["max_scan_rows"] = 1
    assert ns["CONTEXT_REQUESTS_BY_ARM"][arm]["budgets"]["max_scan_rows"] == 1000000
    original = b' \n{"invented":"opaque workflow bytes"}\n '
    wrapper, calls = {"invented_wrapper": True}, []
    def project(data, **kwargs):
        calls.append(("project", data, deepcopy(kwargs)))
        return wrapper
    def rebuild(data, **kwargs):
        calls.append(("rebuild", data, deepcopy(kwargs)))
        return {"scorer_input": {"invented_inner": True}}
    ns["projection_module"] = lambda: SimpleNamespace(project_gold11_schedule_workflow_v2=project,
        revalidate_gold11_schedule_projection_v2=rebuild)
    assert ns["compact_result"](original, expected_result_sha256=sha(original), arm=arm) is wrapper
    retained = wire(wrapper) + b"\n"
    row = dict(status="TERMINAL", exit_code=0, budget_stop=None,
        arm_request_sha256=ns["arm_request_sha256"](arm), scoring_input_sha256=sha(retained),
        result_sha256=sha(original), projection_kind="EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER")
    assert ns["reconstruct_scoring_input"](retained, row, arm, result_bytes=original) == {"invented_inner": True}
    assert calls[0][1] is original and calls[1][2]["result_bytes"] is original
    for _, _, arguments in calls:
        assert arguments["expected_context_request"] == ns["CONTEXT_V3_REQUEST"]
        assert arguments["expected_schedule_request"] == ns["arm_request"](arm)["schedule_request"]
    assert ns["SCORING_SOURCES"][-2:] == ("scripts/gold11_schedule_projection_v1.py", "scripts/gold11_schedule_projection_v2.py")


def test_context_maxima_are_source_grounded_without_backend_import(sandbox):
    # Closed scalar AST data only, not context functions/imports or backend init.
    v1 = source_tree(ROOT / "backend/cad_guided_two_plane_context_v1.py")
    v2 = source_tree(ROOT / "backend/cad_guided_two_plane_context_v2.py")
    v3 = source_tree(ROOT / "backend/cad_guided_two_plane_context_v3.py")
    base = dict(ast.literal_eval(value_node(v1, "_MAXIMA")))
    expected_override = ast.parse('tuple((name, 256 if name == "max_ordered_cad_matches_per_scan_pair" else maximum) for name, maximum in _v1._MAXIMA)', mode="eval").body
    assert ast.dump(value_node(v2, "_MAXIMA")) == ast.dump(expected_override)
    extras = dict(ast.literal_eval(value_node(v3, "_EXTRA_MAXIMA")))
    base["max_ordered_cad_matches_per_scan_pair"] = 256
    base.update(extras)
    assert sandbox.ns["CONTEXT_V3_REQUEST"]["budgets"] == base
    assert len(base) == 23 and all(type(v) is int for v in base.values())


def constructor_mocks(sandbox, drift=None):
    ns, calls = sandbox.ns, []
    defaults = {"TwoPlaneScheduleBudgetsV1": ns["TWO_PLANE_DECLARATION"]["schedule_budgets"]}
    def constructor(name):
        def create(**kwargs):
            data = deepcopy(defaults.get(name, {}))
            data.update(kwargs)
            if drift == name:
                key = next(iter(data))
                data[key] += 1
            calls.append((name, data))
            return SimpleNamespace(kind=name, data=data)
        return create
    def context(mode, budget):
        assert mode == "v3" and budget.kind == "TwoPlaneContextBudgetsV3"
        expected = deepcopy(ns["CONTEXT_V3_REQUEST"])
        expected["budgets"] = deepcopy(budget.data)
        if drift == "context_identity":
            expected["implementation"] = "WRONG"
        return budget, expected
    entry = lambda **_k: forbidden()
    permitted = {
        "backend.cad_guided_global_registration_v1": ["WholePartRegistrationPolicyV1"],
        "backend.cad_guided_native_features": ["NativeGuidedPolicy"],
        "backend.cad_guided_instances": ["SupportPolicy"],
        "backend.cad_guided_noisy_mesh_planes_v5": ["NoisyMeshPlanePolicyV5", "NoisyMeshPlaneBudgetsV5"],
        "backend.cad_guided_mesh_plane_proposals_v4": ["CoplanarPatchPolicyV4", "CoplanarPatchBudgetsV4"],
        "backend.cad_guided_plane_search_v6": ["PlaneSearchPolicyV6", "PlaneSearchBudgetsV6"],
        "backend.cad_guided_triangle_surface_v3": ["TriangleSurfaceBudgets"],
        "backend.cad_guided_global_registration_v3": ["SurfaceRegistrationBudgetsV3"],
        "backend.cad_guided_mesh_offline_v14": ["MeshWorkflowPolicyV14", "inspect_cad_guided_mesh_v14"],
        "backend.cad_guided_plane_initialization_v13": ["_context_configuration"],
        "backend.cad_guided_plane_initialization_v12": ["PlaneInitializationBudgetsV12"],
        "backend.cad_guided_two_plane_schedule_v1": ["TwoPlaneScheduleBudgetsV1"],
        "backend.cad_guided_two_plane_context_v3": ["TwoPlaneContextBudgetsV3"],
        "backend.cad_guided_seeded_plane_proposals_v1": ["SeededPlaneBudgetsV1"],
        "backend.cad_guided_scan_plane_preparation_v1": ["CompositeScanPlaneBudgetsV1"],
        "backend.cad_guided_two_plane_schedule_v2": ["TwoPlaneScheduleBudgetsV2"],
    }
    def isolated_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        assert level == 0 and name in permitted and set(fromlist) <= set(permitted[name])
        members = {item: constructor(item) for item in permitted[name]}
        if "_context_configuration" in members:
            members["_context_configuration"] = context
        if "inspect_cad_guided_mesh_v14" in members:
            members["inspect_cad_guided_mesh_v14"] = entry
        return SimpleNamespace(**members)
    ns["__builtins__"]["__import__"] = isolated_import
    ns["asdict"] = lambda row: deepcopy(row.data)
    admit_mock(sandbox)
    return calls, entry


@pytest.mark.parametrize("arm", ("control", "challenger"))
def test_both_successor_constructor_paths_use_explicit_v3(sandbox, arm):
    calls, entry = constructor_mocks(sandbox)
    actual, policies = sandbox.ns["workflow_entry_and_policies"](arm)
    assert actual is entry
    assert policies["two_plane_context_mode"] == "v3"
    assert policies["two_plane_context_budgets"].data == sandbox.ns["CONTEXT_V3_REQUEST"]["budgets"]
    assert policies["workflow_policy"].data["native_adapter_version"] == "v1"
    assert policies["initialization_budgets"].kind == "PlaneInitializationBudgetsV12"
    assert policies["registration_policy"].data == sandbox.ns["CONFIG"]["registration"]
    assert policies["measurement_policy"].data == sandbox.ns["CONFIG"]["measurement"]
    if arm == "control":
        assert policies["scan_plane_mode"] == "v7" and policies["two_plane_schedule_mode"] == "v1"
        assert policies["two_plane_schedule_budgets"] is policies["seeded_plane_budgets"] is None
    else:
        assert policies["scan_plane_mode"] == "composite_v1" and policies["two_plane_schedule_mode"] == "v2"
        assert policies["two_plane_schedule_budgets"].data == sandbox.ns["SCHEDULE_V2_BUDGETS"]
    assert len(wire(calls)) < MAX_FIXTURE_BYTES


@pytest.mark.parametrize("drift", ("context_identity", "TwoPlaneScheduleBudgetsV1", "TwoPlaneScheduleBudgetsV2"))
def test_constructor_effective_declaration_drift_refuses(sandbox, drift):
    constructor_mocks(sandbox, drift)
    with pytest.raises(ValueError, match="BINDING"):
        sandbox.ns["workflow_entry_and_policies"]("challenger")


QUIESCENCE_CASES = (
    ("clean_leaf", None, None, False, True),
    ("clean_outer", None, None, True, True),
    ("leaf_group_not_asserted", "owned_group_empty_confirmed", None, False, True),
    ("outer_group_false", "owned_group_empty_confirmed", False, True, False),
    ("outer_group_int", "owned_group_empty_confirmed", 1, True, False),
    ("schema", "supervisor_schema", "GOLD11_OWNED_PROCESS_SUPERVISION_V1", False, False),
    ("reaped_false", "direct_child_reaped", False, False, False),
    ("reaped_int", "direct_child_reaped", 1, False, False),
    ("attention", "requires_process_attention", True, False, False),
    ("attention_int", "requires_process_attention", 0, False, False),
    ("escape", "observed_foreign_group_escape_detected", True, False, False),
    ("escape_unconfirmed", "observed_foreign_group_escape_unconfirmed_after_cleanup", True, False, False),
    ("unknown_identity", "observed_identity_unknown_still_present", True, False, False),
    ("cleanup_error", "cleanup_errors", ["invented"], False, False),
    ("cleanup_tuple", "cleanup_errors", (), False, False),
    ("discarded_true", "cleanup_errors_discarded", True, False, False),
    ("discarded_float", "cleanup_errors_discarded", 0., False, False),
    ("discarded_negative", "cleanup_errors_discarded", -1, False, False),
    ("close_error", "control_close_errors", ["invented"], False, False),
    ("missing", "requires_process_attention", "DELETE", False, False),
)


@pytest.mark.parametrize("_label,key,value,outer,expected", QUIESCENCE_CASES)
def test_strict_v3_quiescence_table(sandbox, _label, key, value, outer, expected):
    row = receipt()
    if key is not None:
        if value == "DELETE":
            del row[key]
        else:
            row[key] = value
    assert sandbox.ns["writers_quiescent"](row, new_session=outer) is expected


@pytest.mark.parametrize("state", ("never_dispatched", "clean", "unknown", "retention_failed", "identity_changed", "cleanup_failed"))
def test_fresh_leaf_cache_cleanup_is_quiescence_and_retention_gated(sandbox, state):
    ns, events = sandbox.ns, sandbox.events
    cache, identity = Path("/invented/private/new-leaf"), dict(device=1, inode=2, uid=123)
    row = receipt(worker_dispatch_attempted=state != "never_dispatched",
                  derived_result_retention_complete=state != "retention_failed")
    if state == "unknown":
        row["observed_identity_unknown_still_present"] = True
    ns["fresh_private_cache"] = lambda prefix: (events.append(("create", prefix)) or (cache, identity))
    def identity_check(_path):
        events.append("identity")
        return dict(identity, inode=3) if state == "identity_changed" else identity
    def remove(path):
        events.append(("remove", path))
        if state == "cleanup_failed":
            raise OSError("invented cleanup failure")
    ns.update(cache_identity=identity_check, shutil=SimpleNamespace(rmtree=remove))
    if state in ("identity_changed", "cleanup_failed"):
        with pytest.raises((ValueError, OSError)):
            with ns["owned_leaf_cache"]("INFERENCE", row):
                events.append("body")
    else:
        with ns["owned_leaf_cache"]("INFERENCE", row) as actual:
            assert actual == cache and row["private_cache_cleanup_complete"] is None
            events.append("body")
    if state in ("unknown", "retention_failed"):
        assert events == [("create", "gold11-v6-inference-"), "body"]
        assert row["private_cache_custody"]["status"].startswith("OPAQUE_RETAINED_")
    elif state == "identity_changed":
        assert not any(type(event) is tuple and event[0] == "remove" for event in events)
    else:
        assert events[-1] == ("remove", cache)
    assert row["private_cache_cleanup_complete"] is (state in ("clean", "never_dispatched"))


@pytest.mark.parametrize("bad", (None, "relative", "symlink", "mode", "uid", "file"))
def test_cache_identity_rejects_nonprivate_or_rebound_path(sandbox, bad):
    ns = sandbox.ns
    class LexicalCache:
        def lstat(self):
            mode = stat.S_IFREG | 0o600 if bad == "file" else stat.S_IFDIR | (0o755 if bad == "mode" else 0o700)
            return SimpleNamespace(st_dev=1, st_ino=2, st_uid=999 if bad == "uid" else 123, st_mode=mode)
        def is_absolute(self):
            return bad != "relative"
        def resolve(self):
            return object() if bad == "symlink" else self
    fake = LexicalCache()
    ns["Path"] = lambda value: fake
    if bad is None:
        assert ns["cache_identity"](fake) == dict(device=1, inode=2, uid=123)
    else:
        with pytest.raises(ValueError, match="LEAF_CACHE_NOT_PRIVATE"):
            ns["cache_identity"](fake)


@pytest.mark.parametrize("state", ("fresh", "inside_attempt", "wrong_platform"))
def test_fresh_cache_has_exact_external_creation_scope_before_dispatch(sandbox, state):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    cache = LexicalPath("/invented/campaign/private" if state == "inside_attempt" else "/private/tmp/invented-fresh-only")
    identity = dict(device=1, inode=2, uid=123)
    def create(**kwargs):
        assert kwargs == dict(prefix="gold11-v6-inference-", dir="/private/tmp")
        events.append("create")
        return str(cache)
    ns.update(tempfile=SimpleNamespace(mkdtemp=create), cache_identity=lambda path: identity)
    ns["sys"].platform = "linux" if state == "wrong_platform" else "darwin"
    if state == "fresh":
        assert ns["fresh_private_cache"]("gold11-v6-inference-") == (cache, identity)
    else:
        with pytest.raises(ValueError, match="PRIVATE_CACHE_"):
            ns["fresh_private_cache"]("gold11-v6-inference-")
    assert events.count("create") == (state != "wrong_platform")


@pytest.mark.parametrize("bad", ("unknown", "missing_custody", "identity_mismatch"))
def test_retention_refuses_before_private_enumeration(sandbox, bad):
    ns = sandbox.ns
    class Opaque:
        def __str__(self):
            return "/invented/private/opaque"
        def __truediv__(self, _name):
            return forbidden()
        def glob(self, _pattern):
            return forbidden()
    cache, row = Opaque(), receipt()
    if bad == "unknown":
        row["direct_child_reaped"] = False
        ns["cache_identity"] = forbidden
    elif bad == "identity_mismatch":
        row["private_cache_custody"] = dict(path=str(cache), identity={"inode": 1})
        ns["cache_identity"] = lambda _: {"inode": 2}
    with pytest.raises(ValueError, match="QUIESCENCE_REQUIRED|FRESH_PRIVATE_CACHE_IDENTITY"):
        ns["retain_worker_result"](cache, Path("/invented/result"), row)


@pytest.mark.parametrize("complete", (False, True))
def test_retention_positive_reads_only_after_receipt_and_fresh_identity(sandbox, complete):
    ns, events = sandbox.ns, sandbox.events
    payload = b'{"invented_result":true}'
    class ResultPath:
        def lstat(self):
            events.append("result-metadata")
            if not complete:
                raise FileNotFoundError("invented missing result")
            return SimpleNamespace(st_dev=1, st_ino=3)
    result_path = ResultPath()
    class Cache:
        def __str__(self):
            return "/invented/private/new-cache"
        def __truediv__(self, name):
            assert name == "RESULT.json"
            return result_path
        def glob(self, pattern):
            assert pattern == ".RESULT.json.partial-*"
            events.append("private-enumeration")
            return []
    cache, identity = Cache(), dict(device=1, inode=2, uid=123)
    row = receipt(private_cache_custody=dict(path=str(cache), identity=identity))
    real = ns["writers_quiescent"]
    ns["writers_quiescent"] = lambda record: events.append("quiescence") or real(record)
    ns["cache_identity"] = lambda path: events.append("identity") or identity
    def read(path, cap, **kwargs):
        assert path is result_path and cap == ns["LIMITS"]["case_result_bytes"]
        assert kwargs == {"allow_empty": False}
        events.append("private-result-read")
        return payload
    ns.update(_bounded_read=read, publish_bytes=lambda path, blob: events.append(("publish", path.name, blob)))
    result = ns["retain_worker_result"](cache, LexicalPath("/invented/retained"), row)
    assert events[:4] == ["quiescence", "identity", "private-enumeration", "result-metadata"]
    if complete:
        assert events[4:] == ["private-result-read", ("publish", "RESULT.json", payload)]
        assert result["artifacts"] == [dict(path="RESULT.json", sha256=sha(payload), bytes=len(payload))]
    else:
        assert len(events) == 4 and result["artifacts"] == []
        assert result["status"] == "NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN"


@pytest.mark.parametrize("state", ("clean", "unknown", "exception"))
def test_supervisor_wrapper_latches_custody_before_call_and_blocks_next_leaf(sandbox, state):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    def supervise(*_a, **_k):
        assert ns["_COORDINATOR_WRITERS_QUIESCENT"] is False
        events.append("supervise")
        if state == "exception":
            raise RuntimeError("invented supervisor failure")
        return receipt(observed_identity_unknown_still_present=state == "unknown")
    ns["supervision_module"] = lambda: SimpleNamespace(supervise_process=supervise)
    kwargs = dict(cache=Path("/invented/cache"), log_path=Path("/invented/log"),
        wall_seconds=1, rss_limit=1, log_limit=1)
    if state == "exception":
        with pytest.raises(RuntimeError):
            ns["supervise_process"]([], {}, **kwargs)
    else:
        ns["supervise_process"]([], {}, **kwargs)
    assert ns["_COORDINATOR_WRITERS_QUIESCENT"] is (state == "clean")
    if state != "clean":
        with pytest.raises(ValueError, match="NO_LEAF_AFTER"):
            ns["supervise_process"]([], {}, **kwargs)
        assert events.count("supervise") == 1


@pytest.mark.parametrize("peak,platform,valid", ((0, "darwin", True), (4 * 1024**3, "darwin", True),
    (4 * 1024**3 + 1, "darwin", False), (-1, "darwin", False), (True, "darwin", False),
    (float("nan"), "darwin", False), (float("inf"), "darwin", False), (1, "linux", False)))
def test_coordinator_highwater_is_finite_platform_bound_not_process_table(sandbox, peak, platform, valid):
    ns, calls = sandbox.ns, []
    ns["sys"].platform = platform
    ns["resource"] = SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda who: calls.append(who) or SimpleNamespace(ru_maxrss=peak))
    if valid:
        result = ns["coordinator_memory_check"]()
        assert result == dict(bytes=peak, scope="DARWIN_RU_MAXRSS_LIFETIME_HIGHWATER_NOT_CURRENT_RSS")
    else:
        with pytest.raises(RuntimeError, match="COORDINATOR_"):
            ns["coordinator_memory_check"]()
    assert calls == ([] if platform != "darwin" else [0])
    names = {n.name for n in sandbox.tree.body if isinstance(n, ast.FunctionDef)}
    assert not names & {"process_table", "descendants"}


@pytest.mark.parametrize("aborted,quiescent", ((True, True), (False, False)))
def test_score_refuses_before_reference_or_bundle_access(sandbox, aborted, quiescent):
    admit_mock(sandbox)
    sandbox.ns.update(_COORDINATOR_ABORTED=aborted, _COORDINATOR_WRITERS_QUIESCENT=quiescent)
    with pytest.raises(ValueError, match="NO_SCORING_OR_REFERENCE_ACCESS"):
        sandbox.ns["score_after_terminal"]({}, {}, {})
    assert sandbox.events == ["detached-mock-guard"]


def scoring_mocks(sandbox, state):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    bundle = bundle_for(ns)
    outcomes = {arm: {row["case_token"]: {"invented": True} for row in bundle["cases"]} for arm in ns["ARMS"]}
    rows = deepcopy(outcomes)
    terminal = dict(case_count=11, arm_count=2, terminal_record_count=22,
        outcomes_sha256=sha(wire(outcomes)), arm_requests_sha256=sha(wire(ns["all_arm_requests"]())))
    published = {}
    cache, identity = Path("/invented/private/scorer"), dict(device=1, inode=2, uid=123)
    def publish(path, value):
        assert len(wire(value)) < MAX_FIXTURE_BYTES
        published[path.name] = deepcopy(value)
        events.append(("publish", path.name))
    def supervise(*_a, **_k):
        events.append("supervise")
        if state == "supervisor_exception":
            ns["_COORDINATOR_WRITERS_QUIESCENT"] = False
            raise RuntimeError("invented supervisor exception")
        return receipt(observed_identity_unknown_still_present=state == "unknown")
    def retain(*_a, **_k):
        events.append("retain")
        if state == "retention_failed":
            raise ValueError("invented retention failed")
        return dict(status="COMPLETE_RESULT_BYTES_RETAINED", artifacts=[dict(path="SCORER_RESULT.json", sha256="a" * 64, bytes=2)])
    def remove(path):
        events.append(("remove", path))
        if state == "cleanup_failed":
            raise OSError("invented cleanup failed")
    ns.update(verify_bundle=lambda: bundle,
        reconstruct_all_terminal_inputs=lambda _: (outcomes, outcomes, rows),
        owned_read=lambda *_a, **_k: wire(terminal), selected_references=lambda _: {"invented": "selected"},
        publish_bytes=lambda path, blob: events.append(("publish_bytes", path.name)),
        make_scoring_request=lambda *_: {"invented_request": True}, scorer_arguments=lambda _: {},
        fresh_private_cache=lambda _: (cache, identity), cache_identity=lambda _: identity,
        publish_private_scoring_request=lambda *_: events.append("private-request"),
        issue_leaf_dispatch=lambda *_a, **_k: {"dispatch_id": "invented"},
        runner_command=lambda _: [], supervise_process=supervise, retain_worker_result=retain,
        exclusive_json=publish, bounded_read=lambda *_a, **_k: b"{}",
        shutil=SimpleNamespace(rmtree=remove))
    return bundle, outcomes, rows, published


@pytest.mark.parametrize("state", ("unknown", "supervisor_exception", "retention_failed", "cleanup_failed"))
def test_scorer_failure_always_attempts_receipt_and_never_cleans_uncertain_bytes(sandbox, state):
    bundle, outcomes, rows, published = scoring_mocks(sandbox, state)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        sandbox.ns["score_after_terminal"](bundle, outcomes, rows)
    row = published["SCORER_RECEIPT.json"]
    assert row["worker_dispatch_attempted"] is True and row["private_cache_cleanup_complete"] is False
    assert "failure_type" in row and "SCORED.json" not in published
    assert sandbox.events[-1] == ("publish", "SCORER_RECEIPT.json")
    if state in ("unknown", "supervisor_exception"):
        assert "retain" not in sandbox.events
    if state != "cleanup_failed":
        assert not any(type(event) is tuple and event[0] == "remove" for event in sandbox.events)
        assert row["private_cache_custody"]["status"].startswith("OPAQUE_RETAINED_")


def coordinator_mocks(sandbox, state):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    inject(ns, sandbox.tree, ["internal_coordinator"])
    bundle, audit = bundle_for(ns), {"invented_audit": True}
    cache, identity, token = Path("/invented/private/coordinator"), dict(device=1, inode=2, uid=123), "a" * 64
    dispatch = dict(parent_pid=11, token_sha256=sha(token.encode()), cache=str(cache), cache_identity=identity)
    start = dict(bundle_sha256=sha(wire(bundle)), audit_sha256=sha(wire(audit)),
        status="STARTED_NO_RETRY", limits=ns["LIMITS"], arm_requests_sha256=sha(wire(ns["all_arm_requests"]())))
    class LexicalPaths:
        def __new__(cls, value):
            return Path(value)
        @staticmethod
        def cwd():
            return cache
    publications = {}
    def supervise(case, arm, *_a):
        events.append(("case", case["case_token"], arm))
        row = receipt(worker_dispatch_attempted=True, derived_result_retention_complete=state != "retention_failed",
            private_cache_cleanup_complete=state != "cleanup_failed")
        if state == "unknown":
            ns["_COORDINATOR_WRITERS_QUIESCENT"] = False
            row["observed_identity_unknown_still_present"] = True
        return ns["unavailable"]("invented"), row
    def terminalize(_bundle, reason):
        events.append(("terminalize", reason))
        return [dict(case_id=row["case_token"], arm=arm, durable=True) for row in bundle["cases"] for arm in ns["ARMS"]]
    ns.update(Path=LexicalPaths, verify_bundle=lambda: bundle, verify_pre_run_audit=lambda _: audit,
        cache_identity=lambda _: identity, marker_present=lambda _: False,
        bounded_read=lambda path, *_a: wire(dispatch if path.name == "COORDINATOR_DISPATCH.json" else start),
        sys=SimpleNamespace(stdin=SimpleNamespace(buffer=SimpleNamespace(read=lambda _: wire(dict(token=token, parent_pid=11))))),
        activate_coordinator=lambda *_: events.append("activate"), supervise_case=supervise,
        coordinator_memory_check=forbidden, terminalize_case_records=terminalize,
        score_after_terminal=forbidden,
        exclusive_json=lambda path, value: publications.__setitem__(path.name, deepcopy(value)))
    ns["os"].environ = {"HOME": str(cache), "TMPDIR": str(cache)}
    return publications


@pytest.mark.parametrize("state", ("unknown", "retention_failed", "cleanup_failed"))
def test_coordinator_abort_never_dispatches_second_leaf_or_scores(sandbox, state):
    publications = coordinator_mocks(sandbox, state)
    sandbox.ns["internal_coordinator"]()
    assert sandbox.ns["_COORDINATOR_ABORTED"] is True
    assert sum(type(e) is tuple and e[0] == "case" for e in sandbox.events) == 1
    rows = publications["SCORING_UNAVAILABLE.json"]["terminal_case_records"]
    assert len(rows) == 22 and all(row["durable"] is (state != "unknown") for row in rows)
    assert sum(type(e) is tuple and e[0] == "terminalize" for e in sandbox.events) == (state != "unknown")
    assert publications["COORDINATOR_COMPLETED.json"]["status"] == "TERMINAL_SCORING_UNAVAILABLE"


def outer_mocks(sandbox, state):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    inject(ns, sandbox.tree, ["execute"])
    bundle, audit = bundle_for(ns), {"invented_audit": True}
    published, identity = {}, dict(device=1, inode=2, uid=123)
    def publish(path, value):
        assert len(wire(value)) < MAX_FIXTURE_BYTES
        published[path.name] = deepcopy(value)
        events.append(("publish", path.name))
    real_quiescent = ns["writers_quiescent"]
    def quiescent(row, **kwargs):
        events.append("quiescence")
        return real_quiescent(row, **kwargs)
    def audit_sources(_bundle, evidence, **_kwargs):
        events.append("source-postflight")
        evidence["unchanged"] = state != "audit_failure"
        if state == "audit_failure":
            raise ValueError("invented source drift")
    def terminalize(_bundle, _reason):
        events.append("terminalize")
        return [dict(case_id=row["case_token"], arm=arm, durable=True) for row in bundle["cases"] for arm in ns["ARMS"]]
    def artifacts(_bundle, _row, evidence):
        events.append("artifacts")
        evidence.update(scored_report_sha256="a" * 64, scorer_result_sha256="b" * 64)
    def inventory(*_a, **_k):
        events.append("inventory")
        return dict(complete=True, observed_entries=119, scientific_entries=119)
    ns.update(verify_bundle=lambda: bundle, verify_pre_run_audit=lambda _: audit,
        owned_read=lambda name, *_a: wire(bundle if name == "FROZEN.json" else audit) + b"\n",
        marker_present=lambda _: False, exclusive_json=publish,
        fresh_private_cache=lambda _: (LexicalPath("/invented/private/fresh-coordinator"), identity),
        cache_identity=lambda _: identity, secrets=SimpleNamespace(token_hex=lambda _: "c" * 64),
        runner_command=lambda _: [], supervise_process=lambda *_a, **_k: receipt(
            observed_identity_unknown_still_present=state == "unknown", child_pid=22),
        writers_quiescent=quiescent, final_source_audit=audit_sources,
        terminalize_case_records=terminalize, verify_parent_artifacts=artifacts,
        verify_campaign_inventory=inventory,
        resource=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=1)))
    return published


@pytest.mark.parametrize("state", ("clean", "unknown", "audit_failure"))
def test_outer_postflight_and_all22_fill_follow_whole_group_quiescence(sandbox, state):
    published = outer_mocks(sandbox, state)
    result = sandbox.ns["execute"]()
    assert result == (0 if state == "clean" else 1)
    completed = published["COMPLETED.json"]
    assert len(completed["terminal_case_records"]) == 22
    assert completed["coordinator_private_cache_custody"]["contents_enumerated_or_read"] is False
    dispatch = published["COORDINATOR_DISPATCH.json"]
    assert set(dispatch) == {"parent_pid", "token_sha256", "cache", "cache_identity"}
    assert not dispatch["cache"].startswith(str(sandbox.ns["RUN"]))
    if state == "unknown":
        assert not set(sandbox.events).intersection({"source-postflight", "terminalize", "artifacts", "inventory"})
        assert all(row["durable"] is False for row in completed["terminal_case_records"])
    else:
        for operation in ("source-postflight", "terminalize", "artifacts", "inventory"):
            assert sandbox.events.index("quiescence") < sandbox.events.index(operation)
        assert all(row["durable"] is True for row in completed["terminal_case_records"])
    if state == "audit_failure":
        assert "FINAL_GATE.json" not in published
        assert any(row["phase"] == "FINAL_SOURCE_AUDIT_AFTER_QUIESCENCE" for row in completed["failures"])


def test_full_schema_preserves_scientific_cap_and_excludes_private_cache(sandbox):
    ns = sandbox.ns
    bundle = bundle_for(ns)
    files, directories = ns["_campaign_schema"](bundle, {"scoring_request_sha256": "b" * 64}, {}, {})
    assert "coordinator_work" not in directories[""]
    assert all("private" not in path and "cache" not in path for path in directories)
    required = [name for name, spec in files.items() if spec["required"] and name.startswith("cases/")]
    assert len(required) == 44
    assert sum("leaf" in row for row in files.values()) == 46
    assert sum(row["category"] == "SOURCE" for row in files.values()) == len(ns["SOURCES"])
    read_bound = sum(row["max_bytes"] for row in files.values()) + 32768 + sum(
        row["max_bytes"] for row in files.values() if "leaf" in row)
    assert read_bound <= 8 * 1024**3
    assert ns["ARTIFACT_POLICY"]["scientific_manifest_entries"] == 128
    assert ns["ARTIFACT_POLICY"]["custody_envelope_bytes"] == 256 * 1024
    assert len(wire({"files": files, "directories": directories})) < MAX_FIXTURE_BYTES
    # This small schema, not a fabricated maximal custody ledger, is bounded.
    assert 128 * 4096 == ns["LIMITS"]["case_log_bytes"] // 2


@pytest.mark.parametrize("unknown", ("coordinator_work", "unexpected_cache"))
def test_unknown_cache_name_is_metadata_only_veto_never_recursed(sandbox, unknown):
    ns, events = sandbox.ns, sandbox.events
    bundle = bundle_for(ns)
    info = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFDIR | 0o700,
        st_nlink=1, st_size=0, st_mtime_ns=1, st_ctime_ns=1)
    def open_directory(relative):
        events.append(("open-directory", relative))
        if relative:
            raise FileNotFoundError("invented absent declared directory")
        return 9
    ns.update(signal=SimpleNamespace(ITIMER_REAL=0, getitimer=lambda _: (1., 0.)),
        _CampaignInventoryStop=type("_CampaignInventoryStop", (ValueError,), {}),
        artifact_alarm_guard=inert_deadline, artifact_directory_walk=inert_deadline,
        artifact_module=lambda: SimpleNamespace(inventory_directory_v4=forbidden, ensure_terminal_record_v4=forbidden),
        _open_tree_directory=open_directory, _tree_names=lambda _fd, _cap: ([unknown], None),
        os=SimpleNamespace(fstat=lambda _: info, stat=lambda *_a, **_k: info, close=lambda _: None))
    result = ns["verify_campaign_inventory"](bundle, {}, {})
    assert result["complete"] is False and result["read_bytes"] == 0
    assert result["scientific_entries"] == 1
    assert result["entries"][unknown]["status"] == "UNDECLARED_METADATA_ONLY_NO_READ"
    assert dict(code="UNDECLARED_CAMPAIGN_NAME", path=unknown) in result["errors"]
    assert ("open-directory", unknown) not in events
    assert len(wire(result)) < MAX_FIXTURE_BYTES


def test_no_automatic_cache_finalizer_and_no_diagnostic_release(sandbox):
    tree, ns = sandbox.tree, sandbox.ns
    assert not any(isinstance(n, ast.Attribute) and n.attr == "TemporaryDirectory" for n in ast.walk(tree))
    assert ns["STAGE_TELEMETRY_STATUS"].startswith("WIRED_SOURCE_DRAFT_PENDING")
    assert ns["PENDING_THREE_CASE_DIAGNOSTIC"].startswith("UNIMPLEMENTED")
    assert ns["STAGE_SAMPLE_SOURCE_PATHS"] == tuple(name for name in ns["SOURCES"]
        if name.startswith("backend/") and name.endswith(".py")) + (ns["RUNNER_SOURCE"],)
    old = source_tree(OLD, OLD_SHA)
    def loops(t):
        target = fn(t, "internal_coordinator")
        outer = next(n for n in ast.walk(target) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "case")
        inner = next(n for n in outer.body if isinstance(n, ast.For))
        return ast.dump(outer.target), ast.dump(outer.iter), ast.dump(inner.target), ast.dump(inner.iter)
    assert loops(old) == loops(tree)
    assert ns["ARMS"] == ["control", "challenger"]


def test_sampler_preload_closure_and_worker_order_before_backend_hook(sandbox):
    ns, tree = sandbox.ns, sandbox.tree
    assert ns["STAGE_SAMPLER_SOURCE"] in ns["leaf_sources"]("INFERENCE")
    assert ns["STAGE_SAMPLER_SOURCE"] not in ns["leaf_sources"]("SCORER")
    assert ns["STAGE_SAMPLER_SOURCE"] in ns["SOURCES"]
    calls = sorted((n.lineno, n.func.id) for n in ast.walk(fn(tree, "worker"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    ordered = [name for _, name in calls]
    assert ordered.index("authorize_leaf") < ordered.index("stage_sampler_module") < ordered.index("initialize_worker")
    assert ordered.index("initialize_worker") < ordered.index("engine_call_with_stage_sampler")
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("bounded_read", "stage_sampler_module")
        for n in ast.walk(fn(tree, "engine_call_with_stage_sampler")))
    assert ast.dump(fn(tree, "worker_audit_hook")) == ast.dump(fn(source_tree(OLD, OLD_SHA), "worker_audit_hook"))
    cap = ns["STAGE_LOG_DECLARATION"]
    assert cap["sample_stream_bytes"] + cap["receipt_bytes"] + cap["other_output_remaining_bytes"] == ns["LIMITS"]["case_log_bytes"]


@pytest.mark.parametrize("relative,allowed", (("backend/invented.py", True),
    ("scripts/gold11_engine_stage_sampler_v1.py", False), ("scripts/anything_else.py", False)))
def test_unchanged_backend_hook_has_no_broad_scripts_read_exception(sandbox, relative, allowed):
    ns = sandbox.ns
    inject(ns, sandbox.tree, ["worker_audit_hook"])
    ns.update(runtime_roots=lambda: ("/invented/runtime",), _ARTIFACT_DIRECTORY_WALK=False)
    ns["os"] = SimpleNamespace(sep="/", fsdecode=str, O_WRONLY=1, O_RDWR=2, O_CREAT=64,
        O_TRUNC=512, O_DIRECTORY=65536, O_NOFOLLOW=131072,
        path=SimpleNamespace(realpath=lambda p: str(p), isabs=lambda p: str(p).startswith("/")))
    hook = ns["worker_audit_hook"](LexicalPath("/invented/snapshot"), LexicalPath("/invented/cache"))
    args = ("/invented/snapshot/" + relative, "r", 0)
    if allowed:
        hook("open", args)
    else:
        with pytest.raises(PermissionError, match="WORKER_PATH_OUTSIDE_FROZEN_SCOPE"):
            hook("open", args)


def test_sampler_hash_failure_stops_worker_before_hook_input_or_entry(sandbox):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    inject(ns, sandbox.tree, ["worker"])
    snapshot, cache = LexicalPath("/invented/snapshot"), LexicalPath("/invented/cache")
    ns.update(resource=SimpleNamespace(RLIMIT_CPU=0, setrlimit=lambda *_: events.append("cpu-limit")),
        sys=SimpleNamespace(stdin=SimpleNamespace(buffer=SimpleNamespace(read=lambda _: b"{}"))),
        authorize_leaf=lambda *_: (snapshot, cache, {"arm": "control"}))
    def bound(path, cap, expected):
        assert path == snapshot / ns["STAGE_SAMPLER_SOURCE"]
        assert cap == 2 * 1024**2 and expected == ns["STAGE_SAMPLER_SHA"]
        events.append("hash-refusal")
        raise ValueError("DECLARED_HASH_MISMATCH")
    ns["bounded_read"] = bound
    with pytest.raises(ValueError, match="DECLARED_HASH_MISMATCH"):
        ns["worker"]()
    assert events == ["detached-mock-guard", "cpu-limit", "detached-mock-guard", "hash-refusal"]


@pytest.mark.parametrize("preload", (None, "wrong_hash"))
def test_sampler_missing_or_wrong_preload_refuses_before_entry(sandbox, preload):
    admit_mock(sandbox)
    sandbox.ns["_STAGE_SAMPLER_MODULE"] = (None if preload is None else
        SimpleNamespace(__verified_source_sha256="0" * 64))
    with pytest.raises(ValueError, match="VERIFIED_STAGE_SAMPLER_PRELOAD_REQUIRED"):
        sandbox.ns["engine_call_with_stage_sampler"](LexicalPath("/invented/snapshot"), forbidden, b"s", b"c", {})


@pytest.mark.parametrize("state", ("clean", "constructor_failed", "start_failed", "finish_failed", "write_failed", "oversized"))
@pytest.mark.parametrize("engine_fails", (False, True))
def test_sampler_diagnostics_cannot_replace_engine_result_or_exception(sandbox, state, engine_fails):
    ns, events = sandbox.ns, sandbox.events
    admit_mock(sandbox)
    scan, cad = b"INVENTED_SCAN_NOT_GEOMETRY", b"INVENTED_CAD_NOT_GEOMETRY"
    policy, result = object(), {"unchanged_result": [1, 2, 3]}
    original = wire(result)
    engine_error, writes = RuntimeError("invented exact engine exception"), []
    class Sampler:
        def __init__(self, **kwargs):
            assert kwargs == dict(source_root="/invented/snapshot", source_paths=ns["STAGE_SAMPLE_SOURCE_PATHS"], output_fd=2)
            events.append("construct")
            if state == "constructor_failed":
                raise ValueError("invented constructor failure")
        def start(self):
            events.append("start")
            if state == "start_failed":
                raise ValueError("invented start failure")
        def finish(self):
            events.append("finish")
            if state == "finish_failed":
                raise ValueError("invented finish failure")
            return {"invented_diagnostic": "x" * (5000 if state == "oversized" else 1)}
    def entry(**kwargs):
        assert set(kwargs) == {"scan_bytes", "cad_bytes", "invented_policy"}
        assert kwargs["scan_bytes"] is scan and kwargs["cad_bytes"] is cad and kwargs["invented_policy"] is policy
        events.append("entry")
        if engine_fails:
            raise engine_error
        return result
    def write(line):
        events.append("write")
        if state == "write_failed":
            raise OSError("invented stderr failure")
        writes.append(line)
        return len(line)
    ns.update(_STAGE_SAMPLER_MODULE=SimpleNamespace(__verified_source_sha256=ns["STAGE_SAMPLER_SHA"],
        MainThreadStageSamplerV1=Sampler),
        sys=SimpleNamespace(stderr=SimpleNamespace(write=write, flush=lambda: events.append("flush"))))
    if engine_fails:
        with pytest.raises(RuntimeError) as raised:
            ns["engine_call_with_stage_sampler"](LexicalPath("/invented/snapshot"), entry, scan, cad, {"invented_policy": policy})
        assert raised.value is engine_error
    else:
        assert ns["engine_call_with_stage_sampler"](LexicalPath("/invented/snapshot"), entry, scan, cad, {"invented_policy": policy}) is result
    assert wire(result) == original and events.count("entry") == 1
    assert events.index("construct") < events.index("entry") < events.index("write")
    if state != "constructor_failed":
        assert events.index("entry") < events.index("finish") < events.index("write")
    else:
        assert "finish" not in events
    if state == "write_failed":
        assert writes == []
    else:
        assert len(writes) == 1 and len(writes[0].encode()) <= 4096
        assert writes[0].startswith("\n") and writes[0].endswith("\n")
        if state == "oversized":
            assert json.loads(writes[0]) == {"status": "STAGE_SAMPLER_RECEIPT_TOO_LARGE_OMITTED_WHOLE"}
        else:
            diagnostic = json.loads(writes[0])
            assert diagnostic["startup_failure_type"] == ("ValueError" if state in ("constructor_failed", "start_failed") else None)
