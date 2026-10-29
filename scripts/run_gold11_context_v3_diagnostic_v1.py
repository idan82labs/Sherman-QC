"""UNRELEASED, unscored, separately owned three-case/six-arm diagnostic.

Only exact-hash-selected V6 inference definitions are reused in an explicit NEW
globals profile. No V6 runner, backend module, catalog or scorer is imported at
startup. False release/timing gates precede source, control and input access.
The consumed six-slot profile is not V6's all11/22 protocol or a prior retry.
"""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
from importlib import metadata as import_metadata
import json
import math
import os
from pathlib import Path
import resource
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType


ROOT = Path("/Users/idantal/ShermanQC/FullProject")
RUNNER_SOURCE = "scripts/run_gold11_context_v3_diagnostic_v1.py"
RUN = ROOT / "docs/research/gold11_context_v3_diagnostic_v1_attempt"
PROTOCOL = "docs/research/2026-09-12_gold11_context_v3_diagnostic_v1_protocol.md"
DESIGN = "docs/research/2026-09-12_gold11_context_v3_diagnostic_v1_design.md"
V6_SOURCE = "scripts/run_gold11_mesh_development_v6.py"
V6_SHA = "7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80"
REVIEW_SOURCE = "scripts/review_gold11_inputs_v1.py"
REVIEW_SHA = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
RELEASE_AUTHORIZED = True
TIMING_RELEASE_STATUS = "REVIEWED_EXACT_SIX_SLOT_DIAGNOSTIC_V1_TIMING_RELEASE"
SCHEMA = "GOLD11_CONTEXT_V3_DIAGNOSTIC_V1_UNSCORED"
LEAF_SCHEMA = "ONE_USE_PARENT_BOUND_CONTEXT_V3_DIAGNOSTIC_V1_INFERENCE_ONLY"
SELECTED_ORDER = ("CG08506AE97D", "CG4BDD1371B6", "CG4FAF273093")
SELECTION_ROLES = {
    "CG08506AE97D": "PRIOR_CONSUMED_CUMULATIVE_SUPPORT_CAPACITY_PRESSURE",
    "CG4BDD1371B6": "PRIOR_CONSUMED_REGISTRATION_OR_NEAR_TIE_REFUSAL",
    "CG4FAF273093": "PRIOR_CONSUMED_CPU_EXHAUSTION_OR_UNFINISHED_STAGE",
}
ARMS = ("control", "challenger")
DIAGNOSTIC_LIMITS = dict(case_wall_seconds=3540, case_cpu_seconds=1860,
    sampled_worker_tree_rss_bytes=8*1024**3, coordinator_rss_bytes=4*1024**3,
    global_tree_rss_bytes=12*1024**3, case_log_bytes=1024**2,
    case_result_bytes=128*1024**2, compact_case_bytes=8*1024**2,
    max_scan_bytes=64*1024**2, max_cad_bytes=8*1024**2,
    total_wall_seconds=25200, terminalization_reserve_seconds=2400,
    parent_read_seconds=180, coordinator_log_bytes=2*1024**2)
NO_SCORE_POLICY = dict(catalog_access=False, reference_selection=False,
    scorer_dispatch=False, score_computation=False, f1_claim=False,
    denominator="FIXED_THREE_CONSUMED_CASES_TIMES_TWO_ARMS_SIX_TERMINAL_SLOTS",
    purpose="CAPACITY_REGISTRATION_CPU_LOCALIZATION_NOT_ACCURACY_OR_PROMOTION")

# Constants are evaluated only from the pinned source's explicit assignments.
# ROOT/RUN, catalog/reference globals, V6 release flags and V6 source profile are
# deliberately absent. The retained transport/algorithm identifiers stay honest.
V6_CONSTANTS = (
    "CONFIG", "CERTIFICATE_CAPS", "POSITION_ERROR_DECLARATION", "ARM_DEFINITIONS",
    "COMPARATOR", "TWO_PLANE_DECLARATION", "CONTEXT_V3_REQUEST", "CONTEXT_REQUESTS_BY_ARM",
    "SEEDED_PLANE_BUDGETS", "COMPOSITE_PLANE_BUDGETS", "SCHEDULE_V2_BUDGETS",
    "SCORING_SOURCES", "SCORING_TRANSPORT", "ARTIFACT_SOURCE", "ARTIFACT_SOURCE_SHA",
    "ARTIFACT_POLICY", "SUPERVISION_SOURCE", "SUPERVISION_SOURCE_SHA",
    "STAGE_SAMPLER_SOURCE", "STAGE_SAMPLER_SHA", "STAGE_LOG_DECLARATION",
)
V6_FUNCTIONS = (
    "digest", "encoded", "unique_object", "decode", "deadline", "unavailable", "is_hex",
    "arm_request", "arm_request_sha256", "all_arm_requests", "scan_plane_requests",
    "projection_request", "projection_module", "compact_result", "valid_supervisor_refusal",
    "reconstruct_scoring_input", "workflow_stage_diagnostics",
    "_bounded_read", "bounded_read", "owned_read", "marker_present", "artifact_module",
    "artifact_directory_walk", "artifact_alarm_guard", "publish_bytes", "exclusive_json",
    "runtime_metadata", "contained", "runtime_roots", "allowed_native_library",
    "worker_audit_hook", "install_namespace", "initialize_worker", "supervision_module",
    "cache_identity", "writers_quiescent", "owned_leaf_cache", "issue_leaf_dispatch",
    "authorize_leaf", "workflow_entry_and_policies", "stage_sampler_module",
    "engine_call_with_stage_sampler", "worker", "clean_environment", "memory_budget",
    "supervise_process", "coordinator_memory_check", "retain_worker_result", "supervise_case",
    "terminalize_case_records", "_tree_fingerprint", "_open_tree_directory", "_tree_names",
    "_verify_tree_leaf_bindings", "verify_campaign_inventory",
)
V6_CLASSES = ("_ArtifactDeadlineExpired", "_CampaignInventoryStop")
FORBIDDEN_EXTRACTIONS = (
    "main", "freeze", "execute", "internal_coordinator", "selected_references",
    "scorer_worker", "score_after_terminal", "make_scoring_request", "scorer_arguments",
    "reconstruct_all_terminal_inputs", "verify_bundle", "verify_pre_run_audit",
    "verify_parent_artifacts", "_campaign_schema", "final_source_audit",
)
CALLBACK_PROFILE = (
    "require_release", "publication_cap", "leaf_sources", "validate_leaf_binding",
    "fresh_private_cache", "runner_command", "_campaign_schema",
)
_PROFILE = None


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require_release():
    if RELEASE_AUTHORIZED is not True:
        raise ValueError("DIAGNOSTIC_V1_SOURCE_DRAFT_NOT_RELEASED")
    if TIMING_RELEASE_STATUS != "REVIEWED_EXACT_SIX_SLOT_DIAGNOSTIC_V1_TIMING_RELEASE":
        raise ValueError("DIAGNOSTIC_V1_TIMING_NOT_RELEASED")
    if sys.platform != "darwin":
        raise ValueError("DIAGNOSTIC_V1_DARWIN_REQUIRED")
    if (DIAGNOSTIC_LIMITS["total_wall_seconds"] != 25200
            or DIAGNOSTIC_LIMITS["terminalization_reserve_seconds"] != 2400
            or DIAGNOSTIC_LIMITS["case_wall_seconds"] != 3540
            or DIAGNOSTIC_LIMITS["case_cpu_seconds"] != 1860
            or any(type(value) is not int or value <= 0 for value in DIAGNOSTIC_LIMITS.values())):
        raise ValueError("DIAGNOSTIC_V1_FIXED_PROSPECTIVE_LIMITS_CHANGED")


@contextmanager
def bootstrap_deadline(seconds):
    """Bound the one named source load before reusable definitions exist."""
    def expired(_signum, _frame):
        raise TimeoutError("DIAGNOSTIC_BOOTSTRAP_SOURCE_DEADLINE")
    started = time.monotonic()
    previous = signal.getitimer(signal.ITIMER_REAL)
    effective = min(seconds, previous[0]) if previous[0] > 0 else seconds
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, effective)
    try:
        yield
    finally:
        elapsed = time.monotonic()-started
        signal.setitimer(signal.ITIMER_REAL, max(.000001, previous[0]-elapsed) if previous[0] > 0 else 0, previous[1])
        signal.signal(signal.SIGALRM, old)


def bootstrap_read(path, cap, expected):
    """Only a named source file; no import loader or directory discovery."""
    require_release()
    with bootstrap_deadline(DIAGNOSTIC_LIMITS["parent_read_seconds"]):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= cap:
                raise ValueError("DIAGNOSTIC_BOOTSTRAP_SOURCE_TYPE_OR_SIZE")
            chunks, total = [], 0
            while True:
                chunk = os.read(fd, min(65536, cap+1-total))
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise ValueError("DIAGNOSTIC_BOOTSTRAP_SOURCE_BYTE_CAP")
                chunks.append(chunk)
            after = os.fstat(fd)
            if any(getattr(before, key) != getattr(after, key)
                   for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                raise ValueError("DIAGNOSTIC_BOOTSTRAP_SOURCE_CHANGED")
            blob = b"".join(chunks)
            if digest(blob) != expected:
                raise ValueError("DIAGNOSTIC_BOOTSTRAP_SOURCE_HASH")
            return blob
        finally:
            os.close(fd)


def _assignment(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name]
    if len(matches) != 1:
        raise ValueError("DIAGNOSTIC_EXACT_CONSTANT_REQUIRED:"+name)
    return matches[0]


def source_profile_declaration(namespace, sources):
    return dict(schema="EXPLICIT_DIAGNOSTIC_V1_INFERENCE_AST_PROFILE",
        provider_source=V6_SOURCE, provider_sha256=V6_SHA,
        unchanged_function_names=list(V6_FUNCTIONS), unchanged_class_names=list(V6_CLASSES),
        constant_names=list(V6_CONSTANTS), own_callback_names=list(CALLBACK_PROFILE),
        forbidden_function_names=list(FORBIDDEN_EXTRACTIONS), runner_source=RUNNER_SOURCE,
        run_namespace=str(RUN), leaf_schema=LEAF_SCHEMA, limits=DIAGNOSTIC_LIMITS,
        source_paths=list(sources), selected_case_order=list(SELECTED_ORDER), arms=list(ARMS),
        selection_roles=SELECTION_ROLES, no_score_policy=NO_SCORE_POLICY,
        arm_requests=namespace["all_arm_requests"](),
        stage_source_paths=list(namespace["STAGE_SAMPLE_SOURCE_PATHS"]),
        reused_transport_schema=namespace["SCORING_TRANSPORT"]["wrapper_schema"],
        retained_projection_slot="SCORING_INPUT.json_IS_TRANSPORT_NAME_NOT_SCORER_AUTHORITY",
        inventory_algorithm="UNCHANGED_V6_BODY_WITH_OWN_SIX_SLOT_SCHEMA_EXPLICITLY_WRAPPED")


def profile():
    """Closed extraction into a new namespace; never mutate a loaded V6 module."""
    global _PROFILE
    require_release()
    if _PROFILE is not None:
        return _PROFILE
    frozen = Path(__file__).resolve() == (RUN/"source"/RUNNER_SOURCE).resolve()
    base = RUN/"source" if frozen else ROOT
    blob = bootstrap_read(base/V6_SOURCE, 256*1024, V6_SHA)
    tree = ast.parse(blob, filename=str(base/V6_SOURCE))
    # SOURCES' two nonliteral terminal names are explicit source declarations,
    # never paths discovered by evaluation, import or directory enumeration.
    source_expression = _assignment(tree, "SOURCES").value
    if not isinstance(source_expression, ast.Tuple):
        raise ValueError("DIAGNOSTIC_V6_SOURCE_TUPLE_REQUIRED")
    source_literals = []
    for element in source_expression.elts:
        if isinstance(element, ast.Constant) and type(element.value) is str:
            source_literals.append(element.value)
        elif not isinstance(element, ast.Name) or element.id not in ("REVIEW", "PROTOCOL"):
            raise ValueError("DIAGNOSTIC_UNEXPECTED_SOURCE_EXPRESSION")
    backends = tuple(name for name in source_literals if name.startswith("backend/") and name.endswith(".py"))
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("DIAGNOSTIC_BACKEND_SOURCE_PROFILE")
    imports = dict(argparse=argparse, ast=ast, contextmanager=contextmanager, asdict=asdict,
        fcntl=fcntl, hashlib=hashlib, import_metadata=import_metadata, json=json, math=math,
        os=os, Path=Path, resource=resource, secrets=secrets, selectors=selectors,
        shutil=shutil, signal=signal, stat=stat, subprocess=subprocess, sys=sys,
        tempfile=tempfile, time=time, ModuleType=ModuleType)
    namespace = dict(imports, __name__="_gold11_context_v3_diagnostic_v1_profile",
        __file__=__file__, __builtins__=__builtins__, ARMS=list(ARMS))
    constant_nodes = [_assignment(tree, name) for name in V6_CONSTANTS]
    # Keep the provider's source order for dependent constant expressions.
    constant_nodes.sort(key=lambda node: node.lineno)
    exec(compile(ast.Module(body=constant_nodes, type_ignores=[]), str(base/V6_SOURCE), "exec"), namespace)
    sources = tuple(dict.fromkeys((RUNNER_SOURCE, V6_SOURCE, *backends,
        namespace["ARTIFACT_SOURCE"], namespace["SUPERVISION_SOURCE"],
        namespace["STAGE_SAMPLER_SOURCE"], *namespace["SCORING_SOURCES"],
        REVIEW_SOURCE, DESIGN, PROTOCOL)))
    if len(sources) > 96 or any(Path(name).is_absolute() or ".." in Path(name).parts for name in sources):
        raise ValueError("DIAGNOSTIC_SOURCE_PROFILE_BOUND")
    namespace.update(ROOT=ROOT, RUN=RUN, RUNNER_SOURCE=RUNNER_SOURCE, LEAF_SCHEMA=LEAF_SCHEMA,
        SOURCES=sources, LIMITS=dict(DIAGNOSTIC_LIMITS), require_release=require_release,
        publication_cap=publication_cap, leaf_sources=leaf_sources,
        validate_leaf_binding=validate_leaf_binding, fresh_private_cache=fresh_private_cache,
        runner_command=runner_command, _campaign_schema=campaign_schema,
        _VERIFIED_ARTIFACT_SHA=None, _ARTIFACT_MODULE=None, _ARTIFACT_DIRECTORY_WALK=False,
        _VERIFIED_SUPERVISION_SHA=None, _SUPERVISION_MODULE=None,
        _VERIFIED_SCORING_SOURCES=None, _SCORING_MODULES=None, _STAGE_SAMPLER_MODULE=None,
        _COORDINATOR_CONTEXT=None, _COORDINATOR_WRITERS_QUIESCENT=True, _COORDINATOR_ABORTED=False,
        STAGE_SAMPLE_SOURCE_PATHS=(*backends, V6_SOURCE, RUNNER_SOURCE))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in V6_FUNCTIONS
        or isinstance(node, ast.ClassDef) and node.name in V6_CLASSES]
    names = {node.name for node in selected}
    if (names != set(V6_FUNCTIONS) | set(V6_CLASSES) or len(selected) != len(names)
            or names.intersection(FORBIDDEN_EXTRACTIONS)):
        raise ValueError("DIAGNOSTIC_EXACT_UNCHANGED_DEFINITION_DENOMINATOR")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(base/V6_SOURCE), "exec"), namespace)
    if (any(name in namespace for name in ("CATALOG", "CATALOG_SHA", "SELECTED_SHA"))
            or any(name in namespace for name in FORBIDDEN_EXTRACTIONS if name not in CALLBACK_PROFILE)):
        raise ValueError("DIAGNOSTIC_FORBIDDEN_SCORE_OR_REFERENCE_PROFILE")
    namespace["DIAGNOSTIC_PROFILE"] = source_profile_declaration(namespace, sources)
    namespace["DIAGNOSTIC_PROFILE_SHA"] = digest(encoded(namespace["DIAGNOSTIC_PROFILE"]))
    _PROFILE = namespace
    return namespace


def selected_cases_from_review(blob):
    """Read only fixed source CASES; never execute review or follow references."""
    if digest(blob) != REVIEW_SHA:
        raise ValueError("DIAGNOSTIC_SEPTEMBER9_REVIEW_SOURCE_HASH")
    call = _assignment(ast.parse(blob), "CASES").value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name) or call.func.value.id != "json"
            or call.func.attr != "loads" or len(call.args) != 1 or call.keywords):
        raise ValueError("DIAGNOSTIC_FIXED_REVIEW_CASES_SCHEMA")
    n = profile()
    cases = n["decode"](ast.literal_eval(call.args[0]))
    if type(cases) is not list or len(cases) != 11:
        raise ValueError("DIAGNOSTIC_ORIGINAL_ELEVEN_CASE_AUTHORITY")
    by_token, authority = {}, {}
    for case in cases:
        if type(case) is not dict or set(case) != {"case_token", "part_key", "reference_id",
                "scan_path", "scan_sha256", "cad_path", "cad_sha256"}:
            raise ValueError("DIAGNOSTIC_ORIGINAL_CASE_SOURCE_SHAPE")
        token = case["case_token"]
        if type(token) is not str or token in by_token:
            raise ValueError("DIAGNOSTIC_ORIGINAL_CASE_DUPLICATE")
        for kind in ("scan", "cad"):
            path, sha = case[kind+"_path"], case[kind+"_sha256"]
            if type(path) is not str or not path.startswith("/") or not n["is_hex"](sha, 64):
                raise ValueError("DIAGNOSTIC_ORIGINAL_INPUT_BINDING")
            if path in authority and authority[path] != sha:
                raise ValueError("DIAGNOSTIC_ORIGINAL_INPUT_HASH_CONFLICT")
            authority[path] = sha
        # Deliberately discard reference_id; it is never a diagnostic input.
        by_token[token] = {key: case[key] for key in ("case_token", "part_key", "scan_path",
            "scan_sha256", "cad_path", "cad_sha256")}
    if (len(authority) != 17 or len({case["scan_sha256"] for case in cases}) != 11
            or len({case["cad_sha256"] for case in cases}) != 6
            or not set(SELECTED_ORDER) <= set(by_token)):
        raise ValueError("DIAGNOSTIC_EXACT_RAW17_SOURCE_AUTHORITY")
    return [by_token[token] for token in SELECTED_ORDER], authority


def publication_cap(path):
    n, path = profile(), Path(path)
    try:
        source = path.relative_to(RUN/"source").as_posix()
    except ValueError:
        source = None
    if source in n["SOURCES"]:
        return 2*1024**2
    if path.parent == RUN/"leaf_dispatch":
        return 16384
    forbidden = {"SCORER_RESULT.json", "SCORER_RESULT_PARTIAL.bin", "SCORING_REQUEST.json",
        "SCORED.json", "SELECTED_REFERENCES.json", "SCORER_RECEIPT.json"}
    if path.name in forbidden:
        raise ValueError("DIAGNOSTIC_SCORING_PUBLICATION_FORBIDDEN")
    caps = {"scan.ply": DIAGNOSTIC_LIMITS["max_scan_bytes"], "cad.step": DIAGNOSTIC_LIMITS["max_cad_bytes"],
        "RESULT.json": DIAGNOSTIC_LIMITS["case_result_bytes"], "RESULT_PARTIAL.bin": DIAGNOSTIC_LIMITS["case_result_bytes"],
        "SCORING_INPUT.json": DIAGNOSTIC_LIMITS["compact_case_bytes"],
        "COORDINATOR_VERIFIED.json": 32768, "COORDINATOR_DISPATCH.json": 4096}
    if path.name == "COMPLETED.json" and path.parent != RUN:
        return DIAGNOSTIC_LIMITS["compact_case_bytes"]
    return caps.get(path.name, 256*1024)


def leaf_sources(role):
    n = profile()
    if role != "INFERENCE":
        raise ValueError("DIAGNOSTIC_INFERENCE_ONLY_NO_SCORER")
    return (RUNNER_SOURCE, V6_SOURCE, n["ARTIFACT_SOURCE"], n["STAGE_SAMPLER_SOURCE"],
        *(name for name in n["SOURCES"] if name.startswith("backend/")))


def validate_leaf_binding(role, binding):
    n = profile()
    if (role != "INFERENCE" or type(binding) is not dict
            or set(binding) != {"arm", "scan_sha256", "cad_sha256", "arm_request_sha256"}
            or type(binding["arm"]) is not str or binding["arm"] not in ARMS
            or binding["arm_request_sha256"] != n["arm_request_sha256"](binding["arm"])
            or any(not n["is_hex"](value, 64) for key, value in binding.items() if key != "arm")):
        raise ValueError("DIAGNOSTIC_INFERENCE_BINDING_SCHEMA")


def fresh_private_cache(prefix):
    n = profile()
    # Exact caller literals only; own namespace prefixes are declared, not found.
    if prefix not in ("gold11-v6-inference-", "gold11-diagnostic-v1-coordinator-"):
        raise ValueError("DIAGNOSTIC_PRIVATE_CACHE_ROLE")
    owned_prefix = "gold11-diagnostic-v1-inference-" if prefix == "gold11-v6-inference-" else prefix
    cache = Path(tempfile.mkdtemp(prefix=owned_prefix, dir="/private/tmp")).resolve()
    identity = n["cache_identity"](cache)
    if n["contained"](str(cache), (str(RUN),)):
        raise ValueError("DIAGNOSTIC_PRIVATE_CACHE_OUTSIDE_ATTEMPT_REQUIRED")
    return cache, identity


def runner_command(mode):
    require_release()
    if mode not in ("--worker", "--internal-coordinator"):
        raise ValueError("DIAGNOSTIC_NO_SCORER_OR_OTHER_CHILD_MODE")
    return [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
        str(RUN/"source"/RUNNER_SOURCE), mode]


def frozen_bundle(blobs, cases, authority, runtime):
    n = profile()
    return dict(schema=SCHEMA, source_profile=n["DIAGNOSTIC_PROFILE"],
        source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"],
        source_sha256={name: digest(blob) for name, blob in blobs.items()},
        cases=cases, arms=list(ARMS), terminal_slots=6, selection_order=list(SELECTED_ORDER),
        selection_roles=SELECTION_ROLES, original_review_source_sha256=REVIEW_SHA,
        original_raw17_authority=authority, original_raw17_authority_sha256=digest(encoded(authority)),
        limits=DIAGNOSTIC_LIMITS, config=n["CONFIG"], certificate_caps=n["CERTIFICATE_CAPS"],
        arm_requests=n["all_arm_requests"](), runtime_metadata=runtime,
        no_score_policy=NO_SCORE_POLICY, production_use_allowed=False,
        consumed_diagnostic_not_unseen_validation=True, all11_release_allowed=False)


def freeze():
    require_release()
    n = profile()
    with n["deadline"](DIAGNOSTIC_LIMITS["parent_read_seconds"]):
        blobs = {name: n["bounded_read"](ROOT/name, 2*1024**2) for name in n["SOURCES"]}
        if digest(blobs[V6_SOURCE]) != V6_SHA:
            raise ValueError("DIAGNOSTIC_V6_PROVIDER_CHANGED")
        cases, authority = selected_cases_from_review(blobs[REVIEW_SOURCE])
        bundle = frozen_bundle(blobs, cases, authority, n["runtime_metadata"]())
        if len(encoded(bundle))+1 > 256*1024:
            raise ValueError("DIAGNOSTIC_FROZEN_CONTROL_BYTE_CAP")
        RUN.mkdir(exist_ok=False)
        for name, blob in blobs.items():
            target = RUN/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            n["publish_bytes"](target, blob)
        n["exclusive_json"](RUN/"FROZEN.json", bundle)
    print(json.dumps(dict(status="DIAGNOSTIC_FROZEN_NOT_EXECUTED", schema=SCHEMA,
        bundle_sha256=digest(encoded(bundle)), no_score_policy=NO_SCORE_POLICY)), flush=True)


def verify_bundle():
    require_release()
    n = profile()
    with n["deadline"](DIAGNOSTIC_LIMITS["parent_read_seconds"]):
        bundle = n["decode"](n["owned_read"]("FROZEN.json", 256*1024))
        hashes = bundle.get("source_sha256")
        if type(hashes) is not dict or set(hashes) != set(n["SOURCES"]):
            raise ValueError("DIAGNOSTIC_SOURCE_DENOMINATOR_CHANGED")
        frozen = {}
        for name, sha in hashes.items():
            if not n["is_hex"](sha, 64):
                raise ValueError("DIAGNOSTIC_SOURCE_HASH_SCHEMA")
            n["bounded_read"](ROOT/name, 2*1024**2, sha)
            frozen[name] = n["bounded_read"](RUN/"source"/name, 2*1024**2, sha)
        if hashes[V6_SOURCE] != V6_SHA:
            raise ValueError("DIAGNOSTIC_PROVIDER_SOURCE_CHANGED")
        cases, authority = selected_cases_from_review(frozen[REVIEW_SOURCE])
        expected = frozen_bundle(frozen, cases, authority, n["runtime_metadata"]())
        if encoded(bundle) != encoded(expected):
            raise ValueError("DIAGNOSTIC_FROZEN_PROFILE_OR_INPUT_AUTHORITY_CHANGED")
        for source, pin, slot in ((n["SUPERVISION_SOURCE"], n["SUPERVISION_SOURCE_SHA"], "_VERIFIED_SUPERVISION_SHA"),
                (n["ARTIFACT_SOURCE"], n["ARTIFACT_SOURCE_SHA"], "_VERIFIED_ARTIFACT_SHA")):
            if hashes[source] != pin:
                raise ValueError("DIAGNOSTIC_QUALIFIED_HELPER_PIN")
            n[slot] = pin
        if hashes[n["STAGE_SAMPLER_SOURCE"]] != n["STAGE_SAMPLER_SHA"]:
            raise ValueError("DIAGNOSTIC_SAMPLER_PIN_CHANGED")
        n["_VERIFIED_SCORING_SOURCES"] = {name: hashes[name] for name in n["SCORING_SOURCES"]}
        return bundle


def verify_pre(bundle):
    n = profile()
    audit = n["decode"](n["owned_read"]("PRE_RUN_AUDIT.json", 256*1024))
    if (type(audit) is not dict or audit.get("decision") != "GREEN_FOR_EXACT_UNSCORED_DIAGNOSTIC_V1"
            or audit.get("bundle_sha256") != digest(encoded(bundle))
            or audit.get("source_profile_sha256") != n["DIAGNOSTIC_PROFILE_SHA"]
            or audit.get("selected_case_order") != list(SELECTED_ORDER)
            or audit.get("terminal_slots") != 6 or type(audit.get("terminal_slots")) is not int
            or audit.get("limits") != DIAGNOSTIC_LIMITS or audit.get("no_score_policy") != NO_SCORE_POLICY
            or type(audit.get("reviewer")) is not str or not audit["reviewer"].strip()
            or audit.get("blocking_findings") != []):
        raise ValueError("DIAGNOSTIC_EXACT_INDEPENDENT_PRE_REQUIRED")
    return audit


def six_slots(bundle, reason, durable=False):
    return [dict(case_id=case["case_token"], arm=arm, durable=durable, reason=reason)
        for case in bundle["cases"] for arm in ARMS]


def activate_coordinator(bundle, audit, expected, start):
    n = profile()
    context = dict(schema=LEAF_SCHEMA, coordinator_pid=os.getpid(), outer_pid=os.getppid(),
        bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
        started_sha256=digest(encoded(start)), coordinator_dispatch_sha256=digest(encoded(expected)),
        source_sha256=bundle["source_sha256"], runtime_metadata=bundle["runtime_metadata"],
        diagnostic_source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], no_score_policy=NO_SCORE_POLICY)
    n["exclusive_json"](RUN/"COORDINATOR_VERIFIED.json", context)
    (RUN/"leaf_dispatch").mkdir(exist_ok=False)
    n["_COORDINATOR_CONTEXT"] = context


def internal_coordinator():
    require_release()
    n = profile()
    if Path(__file__).resolve() != (RUN/"source"/RUNNER_SOURCE).resolve():
        raise ValueError("DIAGNOSTIC_FROZEN_COORDINATOR_REQUIRED")
    dispatch = n["decode"](sys.stdin.buffer.read(4096))
    bundle, audit = verify_bundle(), None
    audit = verify_pre(bundle)
    expected = n["decode"](n["owned_read"]("COORDINATOR_DISPATCH.json", 4096))
    start = n["decode"](n["owned_read"]("STARTED.json", 4096))
    if (type(expected) is not dict or set(expected) != {"parent_pid", "token_sha256", "cache", "cache_identity"}
            or type(expected["cache"]) is not str):
        raise ValueError("DIAGNOSTIC_COORDINATOR_CACHE_DISPATCH_SCHEMA")
    cache = Path(expected["cache"])
    if (n["cache_identity"](cache) != expected["cache_identity"] or n["contained"](str(cache), (str(RUN),))
            or Path.cwd() != cache or os.environ.get("HOME") != str(cache) or os.environ.get("TMPDIR") != str(cache)):
        raise ValueError("DIAGNOSTIC_COORDINATOR_PRIVATE_CACHE_BINDING")
    if (type(dispatch) is not dict or set(dispatch) != {"token", "parent_pid"}
            or not n["is_hex"](dispatch["token"], 64)
            or digest(dispatch["token"].encode()) != expected["token_sha256"]
            or dispatch["parent_pid"] != os.getppid() or expected["parent_pid"] != os.getppid()
            or os.getpgrp() != os.getpid() or start.get("status") != "STARTED_NO_RETRY"
            or start.get("limits") != DIAGNOSTIC_LIMITS
            or start.get("source_profile_sha256") != n["DIAGNOSTIC_PROFILE_SHA"]
            or start.get("bundle_sha256") != digest(encoded(bundle))
            or start.get("audit_sha256") != digest(encoded(audit))
            or start.get("arm_requests_sha256") != digest(encoded(n["all_arm_requests"]()))):
        raise ValueError("DIAGNOSTIC_EXACT_COORDINATOR_DISPATCH_REQUIRED")
    for name in ("COMPLETED.json", "COORDINATOR_COMPLETED.json", "INFERENCE_TERMINAL.json",
                 "DIAGNOSTIC_UNAVAILABLE.json", "FINALIZATION_FAILED.json", "FINAL_GATE.json"):
        if n["marker_present"](RUN/name):
            raise ValueError("DIAGNOSTIC_ALREADY_TERMINAL_NO_RETRY")
    activate_coordinator(bundle, audit, expected, start)
    started, terminal = time.monotonic(), []
    try:
        for case in bundle["cases"]:
            for arm in ARMS:
                _, row = n["supervise_case"](case, arm, RUN/"cases"/case["case_token"]/arm,
                    DIAGNOSTIC_LIMITS["total_wall_seconds"]-(time.monotonic()-started)-DIAGNOSTIC_LIMITS["terminalization_reserve_seconds"])
                if row.get("source_binding_valid") is not True:
                    raise ValueError("DIAGNOSTIC_INPUT_BINDING_NOT_ESTABLISHED_ABORT")
                if (not n["_COORDINATOR_WRITERS_QUIESCENT"]
                        or row.get("worker_dispatch_attempted") is True and not n["writers_quiescent"](row)
                        or row.get("derived_result_retention_complete") is False
                        or row.get("private_cache_cleanup_complete") is False):
                    raise ValueError("DIAGNOSTIC_LEAF_CUSTODY_OR_RETENTION_ABORT")
                n["coordinator_memory_check"]()
        terminal = n["terminalize_case_records"](bundle, "DIAGNOSTIC_MISSING_TERMINAL_NO_RETRY")
        if len(terminal) != 6 or not all(row["durable"] for row in terminal):
            raise ValueError("DIAGNOSTIC_ALL_SIX_DURABLE_TERMINALS_REQUIRED")
        evidence = reconstruct_six(bundle)
        n["exclusive_json"](RUN/"INFERENCE_TERMINAL.json", dict(schema=SCHEMA, terminal_slots=6,
            case_count=3, arm_count=2, source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"],
            arm_requests_sha256=digest(encoded(n["all_arm_requests"]())),
            evidence_sha256=digest(encoded(evidence)), no_score_policy=NO_SCORE_POLICY))
        status = "DIAGNOSTIC_INFERENCE_TERMINAL_UNSCORED_PENDING_POST"
    except Exception as error:
        n["_COORDINATOR_ABORTED"] = True
        terminal = (n["terminalize_case_records"](bundle, "DIAGNOSTIC_COORDINATOR_ABORT_NO_RETRY")
            if n["_COORDINATOR_WRITERS_QUIESCENT"] else
            six_slots(bundle, "UNCONFIRMED_WRITERS_DEFER_FILL_TO_OUTER"))
        n["exclusive_json"](RUN/"DIAGNOSTIC_UNAVAILABLE.json", dict(schema=SCHEMA,
            failure_type=type(error).__name__, reason=str(error)[:160], terminal_slots=terminal,
            no_retry=True, no_score_policy=NO_SCORE_POLICY))
        status = "DIAGNOSTIC_COORDINATOR_ABORT_UNSCORED"
    n["exclusive_json"](RUN/"COORDINATOR_COMPLETED.json", dict(schema=SCHEMA, status=status,
        parent_wall_seconds=time.monotonic()-started, terminal_slots=6, no_score_policy=NO_SCORE_POLICY))
    print(json.dumps(dict(status=status, terminal_slots=6, score_computed=False)), flush=True)


def bind_closed_log(relative, receipt, cap, *, clean=False):
    """Post-quiescence closed bytes/counts; not a hash captured by the writer."""
    n = profile()
    created = receipt.get("log_created")
    present = n["marker_present"](RUN/relative)
    if clean and created is not True:
        raise ValueError("DIAGNOSTIC_CLEAN_WORKER_LOG_CREATION_REQUIRED")
    if not present:
        if created is True or clean or receipt.get("forwarded_log_bytes", 0) != 0:
            raise ValueError("DIAGNOSTIC_CLAIMED_LOG_MISSING")
        return dict(status="NOT_CREATED_OR_UNOBSERVED_FAILED_ATTEMPT", bytes=None)
    data = n["owned_read"](relative, cap, allow_empty=True)
    if created is True:
        forwarded, stored = receipt.get("forwarded_log_bytes"), receipt.get("stored_log_bytes")
        if (type(forwarded) is not int or forwarded < 0 or type(stored) is not int
                or stored != min(forwarded, cap) or len(data) != stored
                or forwarded > cap and not receipt.get("budget_stop")):
            raise ValueError("DIAGNOSTIC_CLOSED_LOG_BYTE_COUNT_BINDING")
        status = "CLOSED_LOG_BYTES_BOUND" if forwarded <= cap else "EXPLICIT_BUDGET_TRUNCATED_LOG"
    else:
        status = "FAILED_LOG_UNKNOWN_OBSERVED_BYTE_COUNT"
    return dict(status=status, bytes=len(data), sha256=digest(data),
        hash_scope="POST_QUIESCENCE_FILE_BYTES_NOT_SUPERVISOR_CAPTURE_TIME_HASH")


def reconstruct_six(bundle):
    """Six original-result/retained-projection bindings, never a scorer call."""
    n = profile()
    if not n["_COORDINATOR_WRITERS_QUIESCENT"]:
        raise ValueError("DIAGNOSTIC_NO_RECONSTRUCTION_WHILE_LEAF_WRITERS_UNKNOWN")
    if [case["case_token"] for case in bundle["cases"]] != list(SELECTED_ORDER) or bundle["arms"] != list(ARMS):
        raise ValueError("DIAGNOSTIC_EXACT_SIX_SLOT_ORDER")
    evidence = dict(schema="DIAGNOSTIC_V1_SIX_SLOT_EXACT_TRANSPORT_BINDINGS", rows=[], files={}, logs={},
        no_score_policy=NO_SCORE_POLICY, source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"])
    with n["deadline"](DIAGNOSTIC_LIMITS["parent_read_seconds"]):
        for case in bundle["cases"]:
            for arm in ARMS:
                prefix = "cases/"+case["case_token"]+"/"+arm+"/"
                receipt_bytes = n["owned_read"](prefix+"COMPLETED.json", DIAGNOSTIC_LIMITS["compact_case_bytes"])
                receipt = n["decode"](receipt_bytes)
                if (type(receipt) is not dict or receipt.get("status") != "TERMINAL"
                        or receipt.get("arm_request_sha256") != n["arm_request_sha256"](arm)):
                    raise ValueError("DIAGNOSTIC_TERMINAL_RECEIPT_BINDING")
                projection_bytes = n["owned_read"](prefix+"SCORING_INPUT.json",
                    DIAGNOSTIC_LIMITS["compact_case_bytes"], receipt.get("scoring_input_sha256"))
                original = None
                clean = (type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
                    and receipt.get("budget_stop") is None and "failure_type" not in receipt)
                if clean:
                    if (not n["writers_quiescent"](receipt) or receipt.get("source_binding_valid") is not True
                            or receipt.get("derived_result_retention_complete") is not True
                            or receipt.get("private_cache_cleanup_complete") is not True):
                        raise ValueError("DIAGNOSTIC_CLEAN_RESULT_CUSTODY_REQUIRED")
                    original = n["owned_read"](prefix+"RESULT.json", DIAGNOSTIC_LIMITS["case_result_bytes"],
                        receipt.get("result_sha256"))
                inner = n["reconstruct_scoring_input"](projection_bytes, receipt, arm, result_bytes=original)
                # Preserve all previously retained complete/partial bytes even
                # on a failed worker. Their existence does not make them usable.
                retained = receipt.get("retained_result_artifacts", {})
                if type(retained) is not dict or type(retained.get("artifacts", [])) is not list:
                    raise ValueError("DIAGNOSTIC_RETAINED_ARTIFACT_RECORD_SCHEMA")
                artifacts = retained.get("artifacts", [])
                if len(artifacts) > 1:
                    raise ValueError("DIAGNOSTIC_ONE_RETAINED_COMPLETE_OR_PARTIAL_RESULT")
                for item in artifacts:
                    if (type(item) is not dict or set(item) != {"path", "sha256", "bytes"}
                            or item["path"] not in ("RESULT.json", "RESULT_PARTIAL.bin")
                            or not n["is_hex"](item["sha256"], 64)
                            or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= DIAGNOSTIC_LIMITS["case_result_bytes"]):
                        raise ValueError("DIAGNOSTIC_RETAINED_ARTIFACT_SCOPE")
                    data = n["owned_read"](prefix+item["path"], DIAGNOSTIC_LIMITS["case_result_bytes"],
                        item["sha256"], allow_empty=item["path"].endswith(".bin"))
                    if len(data) != item["bytes"]:
                        raise ValueError("DIAGNOSTIC_RETAINED_ARTIFACT_SIZE")
                    evidence["files"][prefix+item["path"]] = dict(sha256=digest(data), bytes=len(data))
                    del data
                for name, data in (("COMPLETED.json", receipt_bytes), ("SCORING_INPUT.json", projection_bytes)):
                    evidence["files"][prefix+name] = dict(sha256=digest(data), bytes=len(data))
                log = bind_closed_log(prefix+"OUTPUT.txt", receipt, DIAGNOSTIC_LIMITS["case_log_bytes"], clean=clean)
                evidence["logs"][prefix+"OUTPUT.txt"] = log
                if "sha256" in log:
                    evidence["files"][prefix+"OUTPUT.txt"] = {key: log[key] for key in ("sha256", "bytes")}
                row = dict(case_id=case["case_token"], arm=arm,
                    arm_request_sha256=n["arm_request_sha256"](arm),
                    original_input_hashes=dict(scan=case["scan_sha256"], cad=case["cad_sha256"]),
                    receipt_sha256=digest(receipt_bytes), retained_projection_sha256=digest(projection_bytes),
                    original_result_sha256=receipt.get("result_sha256"),
                    reconstruction="EXACT_RESULT_AND_V2_WRAPPER" if clean else "EXACT_SUPERVISOR_REFUSAL",
                    reconstructed_inner_sha256=digest(encoded(inner)),
                    exit_code=receipt.get("exit_code"), budget_stop=receipt.get("budget_stop"),
                    failure_type=receipt.get("failure_type"), source_binding_valid=receipt.get("source_binding_valid"),
                    worker_dispatch_attempted=receipt.get("worker_dispatch_attempted"),
                    writer_quiescence_receipt=n["writers_quiescent"](receipt),
                    private_cache_cleanup_complete=receipt.get("private_cache_cleanup_complete"),
                    workflow_stage_diagnostics=receipt.get("workflow_stage_diagnostics"),
                    retained_process_receipt_location=prefix+"COMPLETED.json",
                    retained_sampler_and_native_log_location=prefix+"OUTPUT.txt",
                    last_observed_stack_interpretation="LOG_ONLY_UNKNOWN_WORK_AFTER_LAST_OBSERVATION",
                    cpu_and_memory_interpretation="PRESERVE_RAW_RECEIPT_LOG_SCOPES_NO_CURRENT_RSS_FROM_HWM",
                    score_computed=False)
                evidence["rows"].append(row)
                del original, inner, projection_bytes, receipt_bytes, receipt
                if len(encoded(evidence))+1 > 256*1024:
                    raise ValueError("DIAGNOSTIC_SIX_SLOT_EVIDENCE_ENVELOPE")
    if len(evidence["rows"]) != 6:
        raise ValueError("DIAGNOSTIC_SIX_SLOT_RECONSTRUCTION_DENOMINATOR")
    return evidence


def campaign_schema(bundle, artifact_evidence, control_bindings, published):
    """Own six-slot namespace, consumed by the unchanged V6 inventory algorithm."""
    n = profile()
    files, directories = {}, {"": {}}
    def add(relative, cap, category, *, required=False, sha=None, size=None):
        parts = Path(relative).parts
        if (not parts or Path(relative).is_absolute() or relative != "/".join(parts)
                or any(part in (".", "..") for part in parts) or relative in files):
            raise ValueError("DIAGNOSTIC_SCHEMA_PATH_INVALID_OR_DUPLICATE")
        for index in range(len(parts)):
            parent, name = "/".join(parts[:index]), parts[index]
            kind = "file" if index == len(parts)-1 else "directory"
            if directories.setdefault(parent, {}).setdefault(name, kind) != kind:
                raise ValueError("DIAGNOSTIC_SCHEMA_PATH_KIND_CONFLICT")
            if kind == "directory":
                directories.setdefault("/".join(parts[:index+1]), {})
        files[relative] = dict(max_bytes=cap, category=category, required=required, sha256=sha, bytes=size)
    for name in n["SOURCES"]:
        add("source/"+name, 2*1024**2, "SOURCE", required=True, sha=bundle["source_sha256"][name])
    controls = {"FROZEN.json": 256*1024, "PRE_RUN_AUDIT.json": 256*1024, "STARTED.json": 4096,
        "COORDINATOR_DISPATCH.json": 4096, "COORDINATOR_VERIFIED.json": 32768,
        "DIAGNOSTIC_UNAVAILABLE.json": 256*1024, "FINALIZATION_FAILED.json": 256*1024}
    for name, cap in controls.items():
        add(name, cap, "CONTROL", required=name in control_bindings, sha=control_bindings.get(name))
    for name, cap in (("COORDINATOR_OUTPUT.txt", DIAGNOSTIC_LIMITS["coordinator_log_bytes"]),
            ("COORDINATOR_COMPLETED.json", 4096), ("INFERENCE_TERMINAL.json", 256*1024),
            ("DIAGNOSTIC_EVIDENCE.json", 256*1024)):
        row = artifact_evidence.get("files", {}).get(name, {})
        add(name, cap, "SCIENTIFIC", required=bool(row), sha=row.get("sha256"), size=row.get("bytes"))
    case_caps = {"STARTED.json": 256*1024, "COMPLETED.json": DIAGNOSTIC_LIMITS["compact_case_bytes"],
        "SCORING_INPUT.json": DIAGNOSTIC_LIMITS["compact_case_bytes"] , "OUTPUT.txt": DIAGNOSTIC_LIMITS["case_log_bytes"],
        "RESULT.json": DIAGNOSTIC_LIMITS["case_result_bytes"], "RESULT_PARTIAL.bin": DIAGNOSTIC_LIMITS["case_result_bytes"]}
    if [row["case_token"] for row in bundle["cases"]] != list(SELECTED_ORDER):
        raise ValueError("DIAGNOSTIC_SCHEMA_FIXED_CASE_ORDER")
    for case in bundle["cases"]:
        for arm in ARMS:
            for name, cap in case_caps.items():
                relative = "cases/"+case["case_token"]+"/"+arm+"/"+name
                row = artifact_evidence.get("files", {}).get(relative, {})
                add(relative, cap, "SCIENTIFIC", required=name in ("COMPLETED.json", "SCORING_INPUT.json"),
                    sha=row.get("sha256"), size=row.get("bytes"))
            binding = dict(arm=arm, scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"],
                arm_request_sha256=n["arm_request_sha256"](arm))
            identifier = digest(encoded(dict(role="INFERENCE", binding=binding)))
            for claim in (False, True):
                relative = "leaf_dispatch/"+identifier+(".CLAIMED.json" if claim else ".json")
                add(relative, 16384, "DISPATCH", required=identifier in artifact_evidence.get("required_leaf_dispatch_ids", []))
                files[relative]["leaf"] = dict(role="INFERENCE", binding=binding, identifier=identifier, claim=claim)
    for name, row in published.items():
        if name not in ("COMPLETED.json", "FINAL_GATE.json"):
            raise ValueError("DIAGNOSTIC_UNKNOWN_FINAL_PUBLICATION")
        add(name, 256*1024, "CONTROL", required=True, sha=row["sha256"], size=row["bytes"])
    directories[""]["leaf_dispatch"] = "directory"
    directories.setdefault("leaf_dispatch", {})
    return files, directories


def source_tree_shape():
    """Named frozen source branches only; unknown names veto before dispatch."""
    n = profile()
    expected = {"source": {}}
    for name in n["SOURCES"]:
        parts = ("source", *Path(name).parts)
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            kind = "file" if index == len(parts)-1 else "directory"
            prior = expected.setdefault(parent, {}).setdefault(parts[index], kind)
            if prior != kind:
                raise ValueError("DIAGNOSTIC_SOURCE_TREE_DECLARATION_CONFLICT")
            if kind == "directory":
                expected.setdefault("/".join(parts[:index+1]), {})
    for relative, names in sorted(expected.items()):
        fd = n["_open_tree_directory"](relative)
        try:
            before = os.fstat(fd)
            actual, overflow = n["_tree_names"](fd, len(names))
            if overflow is not None or actual != sorted(names):
                raise ValueError("DIAGNOSTIC_UNDECLARED_OR_MISSING_SOURCE_NAME")
            for name, kind in names.items():
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                valid = stat.S_ISDIR(info.st_mode) if kind == "directory" else stat.S_ISREG(info.st_mode)
                if not valid:
                    raise ValueError("DIAGNOSTIC_FROZEN_SOURCE_TYPE")
            after_names, after_overflow = n["_tree_names"](fd, len(names))
            if after_names != actual or after_overflow is not None or n["_tree_fingerprint"](before) != n["_tree_fingerprint"](os.fstat(fd)):
                raise ValueError("DIAGNOSTIC_FROZEN_SOURCE_DIRECTORY_CHANGED")
        finally:
            os.close(fd)


def source_postflight(bundle, bindings):
    n = profile()
    verified = verify_bundle()
    if encoded(verified) != encoded(bundle):
        raise ValueError("DIAGNOSTIC_POSTFLIGHT_BUNDLE_CHANGED")
    source_tree_shape()
    controls = {}
    for name, sha in bindings.items():
        blob = n["owned_read"](name, 256*1024, sha)
        controls[name] = dict(bytes=len(blob), sha256=digest(blob))
    return dict(schema="DIAGNOSTIC_V1_POST_QUIESCENCE_SOURCE_CONTROL_AUDIT", unchanged=True,
        source_sha256=bundle["source_sha256"], control_bindings=controls,
        source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], no_input_or_reference_reread=True)


def execute():
    require_release()
    n = profile()
    # Pure source/control preflight does not open any physical input. The fixed
    # source profile and exact PRE are bound before STARTED consumes this run.
    start_attempted = False
    try:
        with n["deadline"](DIAGNOSTIC_LIMITS["parent_read_seconds"]):
            bundle = verify_bundle()
            audit = verify_pre(bundle)
            source_tree_shape()
            fd = n["_open_tree_directory"]("")
            try:
                names, overflow = n["_tree_names"](fd, 3)
                if overflow is not None or names != ["FROZEN.json", "PRE_RUN_AUDIT.json", "source"]:
                    raise ValueError("DIAGNOSTIC_STARTED_UNKNOWN_OR_TERMINAL_ATTEMPT_NO_RETRY")
            finally:
                os.close(fd)
            bindings = {}
            for name, expected in (("FROZEN.json", bundle), ("PRE_RUN_AUDIT.json", audit)):
                blob = n["owned_read"](name, 256*1024)
                if n["decode"](blob) != expected:
                    raise ValueError("DIAGNOSTIC_PREFLIGHT_CONTROL_CHANGED_BEFORE_START")
                bindings[name] = digest(blob)
            start = dict(schema=SCHEMA, status="STARTED_NO_RETRY", bundle_sha256=digest(encoded(bundle)),
                audit_sha256=digest(encoded(audit)), limits=DIAGNOSTIC_LIMITS,
                arm_requests_sha256=digest(encoded(n["all_arm_requests"]())),
                source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], terminal_slots=6)
            bindings["STARTED.json"] = digest(encoded(start)+b"\n")
            start_attempted = True
            n["exclusive_json"](RUN/"STARTED.json", start)
    except BaseException as error:
        if not start_attempted or isinstance(error, FileExistsError):
            raise
        try:
            with n["deadline"](1):
                n["exclusive_json"](RUN/"FINALIZATION_FAILED.json", dict(schema=SCHEMA,
                    status="TERMINAL_VETO_NO_RELEASE", phase="STARTED_PUBLICATION",
                    failure_type=type(error).__name__, worker_dispatched=False,
                    terminal_slots=six_slots(bundle, "START_PUBLICATION_FAILED_NO_DISPATCH"), no_score_policy=NO_SCORE_POLICY))
        except BaseException:
            pass
        return 1
    begun = time.monotonic()
    receipt, failures, terminal, postflight, evidence, artifact_evidence = {}, [], [], {}, {}, {}
    cache_custody, inventory, stopped = None, None, False
    status = "DIAGNOSTIC_TERMINAL_UNAVAILABLE_UNSCORED"
    def remaining():
        left = DIAGNOSTIC_LIMITS["total_wall_seconds"]-(time.monotonic()-begun)
        if left <= 0:
            raise TimeoutError("DIAGNOSTIC_CAMPAIGN_ABSOLUTE_DEADLINE_EXPIRED")
        return left
    def failure(phase, error):
        failures.append(dict(phase=phase, failure_type=type(error).__name__,
            reason=str(error)[:160] if isinstance(error, (ValueError, TimeoutError)) else type(error).__name__))
    try:
        with n["deadline"](max(.001, remaining()-DIAGNOSTIC_LIMITS["terminalization_reserve_seconds"])):
            cache, identity = fresh_private_cache("gold11-diagnostic-v1-coordinator-")
            cache_custody = dict(path=str(cache), identity=identity,
                status="OPAQUE_PRIVATE_CACHE_RETAINED_OUTSIDE_ATTEMPT", contents_enumerated_or_read=False)
            token = secrets.token_hex(32)
            dispatch = dict(parent_pid=os.getpid(), token_sha256=digest(token.encode()), cache=str(cache), cache_identity=identity)
            bindings["COORDINATOR_DISPATCH.json"] = digest(encoded(dispatch)+b"\n")
            n["exclusive_json"](RUN/"COORDINATOR_DISPATCH.json", dispatch)
            if n["cache_identity"](cache) != identity:
                raise ValueError("DIAGNOSTIC_COORDINATOR_CACHE_CHANGED_BEFORE_DISPATCH")
            receipt = n["supervise_process"](runner_command("--internal-coordinator"),
                dict(token=token, parent_pid=os.getpid()), cache=cache,
                log_path=RUN/"COORDINATOR_OUTPUT.txt", new_session=True,
                wall_seconds=max(.001, remaining()-DIAGNOSTIC_LIMITS["terminalization_reserve_seconds"]),
                rss_limit=DIAGNOSTIC_LIMITS["global_tree_rss_bytes"], log_limit=DIAGNOSTIC_LIMITS["coordinator_log_bytes"])
    except BaseException as error:
        failure("COORDINATOR_SUPERVISION", error)
    try:
        with n["deadline"](remaining()):
            stopped = n["writers_quiescent"](receipt, new_session=True)
            if not stopped:
                terminal = six_slots(bundle, "WRITER_QUIESCENCE_UNCONFIRMED_NO_CONTENT_POSTFLIGHT_OR_FILL")
                raise ValueError("DIAGNOSTIC_WHOLE_GROUP_QUIESCENCE_REQUIRED")
            # No source/control content postflight, result read, inventory or
            # terminal fill is reached before strict whole-group quiescence.
            if "COORDINATOR_DISPATCH.json" in bindings:
                context = dict(schema=LEAF_SCHEMA, coordinator_pid=receipt.get("child_pid"), outer_pid=os.getpid(),
                    bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
                    started_sha256=digest(encoded(start)), coordinator_dispatch_sha256=digest(encoded(dispatch)),
                    source_sha256=bundle["source_sha256"], runtime_metadata=bundle["runtime_metadata"],
                    diagnostic_source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], no_score_policy=NO_SCORE_POLICY)
                bindings["COORDINATOR_VERIFIED.json"] = digest(encoded(context)+b"\n")
            try:
                postflight = source_postflight(bundle, bindings)
            except BaseException as error:
                failure("SOURCE_CONTROL_POSTFLIGHT_AFTER_QUIESCENCE", error)
            terminal = n["terminalize_case_records"](bundle, "DIAGNOSTIC_OUTER_FALLBACK_NO_RETRY")
            if len(terminal) != 6 or not all(row["durable"] for row in terminal):
                raise ValueError("DIAGNOSTIC_TERMINAL_STORAGE_FAILURE")
            evidence = reconstruct_six(bundle)
            n["exclusive_json"](RUN/"DIAGNOSTIC_EVIDENCE.json", evidence)
            artifact_evidence = dict(files=dict(evidence["files"]))
            artifact_evidence["required_leaf_dispatch_ids"] = [digest(encoded(dict(role="INFERENCE", binding=dict(
                arm=row["arm"], scan_sha256=row["original_input_hashes"]["scan"],
                cad_sha256=row["original_input_hashes"]["cad"], arm_request_sha256=row["arm_request_sha256"]))))
                for row in evidence["rows"] if row["reconstruction"] == "EXACT_RESULT_AND_V2_WRAPPER"]
            evidence_bytes = encoded(evidence)+b"\n"
            artifact_evidence["files"]["DIAGNOSTIC_EVIDENCE.json"] = dict(sha256=digest(evidence_bytes), bytes=len(evidence_bytes))
            outer_log = bind_closed_log("COORDINATOR_OUTPUT.txt", receipt, DIAGNOSTIC_LIMITS["coordinator_log_bytes"],
                clean=receipt.get("exit_code") == 0 and receipt.get("budget_stop") is None)
            if "sha256" in outer_log:
                artifact_evidence["files"]["COORDINATOR_OUTPUT.txt"] = {key: outer_log[key] for key in ("sha256", "bytes")}
            if receipt.get("exit_code") != 0 or receipt.get("budget_stop") is not None:
                raise ValueError("DIAGNOSTIC_COORDINATOR_EXIT_OR_BUDGET_STOP")
            if n["marker_present"](RUN/"DIAGNOSTIC_UNAVAILABLE.json") or n["marker_present"](RUN/"FINALIZATION_FAILED.json"):
                raise ValueError("DIAGNOSTIC_COORDINATOR_OR_FINALIZATION_VETO")
            terminal_blob = n["owned_read"]("INFERENCE_TERMINAL.json", 256*1024)
            declared_terminal = n["decode"](terminal_blob)
            if (type(declared_terminal) is not dict or declared_terminal.get("schema") != SCHEMA
                    or any(type(declared_terminal.get(key)) is not int or declared_terminal[key] != count
                        for key, count in (("terminal_slots", 6), ("case_count", 3), ("arm_count", 2)))
                    or declared_terminal.get("source_profile_sha256") != n["DIAGNOSTIC_PROFILE_SHA"]
                    or declared_terminal.get("arm_requests_sha256") != digest(encoded(n["all_arm_requests"]()))
                    or declared_terminal.get("evidence_sha256") != digest(encoded(evidence))
                    or declared_terminal.get("no_score_policy") != NO_SCORE_POLICY):
                raise ValueError("DIAGNOSTIC_COORDINATOR_SIX_SLOT_EVIDENCE_BINDING")
            completed_blob = n["owned_read"]("COORDINATOR_COMPLETED.json", 4096)
            coordinator_completed = n["decode"](completed_blob)
            if (type(coordinator_completed) is not dict or coordinator_completed.get("schema") != SCHEMA
                    or coordinator_completed.get("status") != "DIAGNOSTIC_INFERENCE_TERMINAL_UNSCORED_PENDING_POST"
                    or type(coordinator_completed.get("terminal_slots")) is not int or coordinator_completed["terminal_slots"] != 6
                    or coordinator_completed.get("no_score_policy") != NO_SCORE_POLICY):
                raise ValueError("DIAGNOSTIC_COORDINATOR_COMPLETION_BINDING")
            for name, data in (("INFERENCE_TERMINAL.json", terminal_blob), ("COORDINATOR_COMPLETED.json", completed_blob)):
                artifact_evidence["files"][name] = dict(sha256=digest(data), bytes=len(data))
            if not failures:
                status = "DIAGNOSTIC_SIX_SLOTS_UNSCORED_PENDING_INDEPENDENT_POST"
    except BaseException as error:
        failure("SIX_SLOT_TERMINAL_EVIDENCE", error)
    try:
        with n["deadline"](remaining()):
            if not stopped:
                raise ValueError("DIAGNOSTIC_NO_INVENTORY_WITH_UNKNOWN_WRITERS")
            inventory = n["verify_campaign_inventory"](bundle, artifact_evidence, bindings)
            if inventory.get("complete") is not True:
                raise ValueError("DIAGNOSTIC_COMPLETE_TREE_CUSTODY_REQUIRED")
    except BaseException as error:
        failure("COMPLETE_DIAGNOSTIC_TREE_INVENTORY", error)
    try:
        with n["deadline"](remaining()):
            peak, elapsed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, time.monotonic()-begun
            if elapsed > DIAGNOSTIC_LIMITS["total_wall_seconds"] or peak > DIAGNOSTIC_LIMITS["global_tree_rss_bytes"]:
                raise ValueError("DIAGNOSTIC_PARENT_FINAL_WALL_OR_HIGHWATER")
            if failures:
                status = "DIAGNOSTIC_TERMINAL_CUSTODY_OR_RESOURCE_FAILURE_UNSCORED"
            completed = dict(schema=SCHEMA, status=status, source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"],
                coordinator_receipt=receipt, failures=failures, terminal_slot_records=terminal,
                post_quiescence_source_audit=postflight,
                reused_v6_inventory_algorithm_receipt=inventory,
                diagnostic_inventory_profile="OWN_SIX_SLOT_SCHEMA_NOT_V6_ALL22",
                evidence_sha256=digest(encoded(evidence)) if evidence else None,
                coordinator_private_cache_custody=cache_custody,
                parent_wall_seconds=elapsed, parent_lifetime_highwater_bytes=peak,
                no_score_policy=NO_SCORE_POLICY, no_resume=True, all11_release_allowed=False)
            if len(encoded(completed))+1 > 256*1024:
                raise ValueError("DIAGNOSTIC_COMPLETED_ENVELOPE_BOUND")
            n["exclusive_json"](RUN/"COMPLETED.json", completed)
            if failures:
                raise ValueError("DIAGNOSTIC_TERMINAL_FAILURE_NO_FINAL_GATE")
            published = {"COMPLETED.json": dict(sha256=digest(encoded(completed)+b"\n"), bytes=len(encoded(completed))+1)}
            after = n["verify_campaign_inventory"](bundle, artifact_evidence, bindings, published=published, previous=inventory)
            if after.get("complete") is not True:
                raise ValueError("DIAGNOSTIC_POST_COMPLETED_INVENTORY_CHANGED")
            gate = dict(schema=SCHEMA, status="UNSCORED_DIAGNOSTIC_PENDING_INDEPENDENT_POST",
                completed_sha256=digest(encoded(completed)), evidence_sha256=digest(encoded(evidence)),
                post_completed_inventory_sha256=digest(encoded(after)), source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"],
                terminal_slots=6, no_score_policy=NO_SCORE_POLICY, all11_release_allowed=False)
            n["exclusive_json"](RUN/"FINAL_GATE.json", gate)
            published["FINAL_GATE.json"] = dict(sha256=digest(encoded(gate)+b"\n"), bytes=len(encoded(gate))+1)
            final = n["verify_campaign_inventory"](bundle, artifact_evidence, bindings, published=published, previous=after)
            if final.get("complete") is not True or time.monotonic()-begun > DIAGNOSTIC_LIMITS["total_wall_seconds"]:
                raise ValueError("DIAGNOSTIC_POST_GATE_CUSTODY_OR_WALL_FAILURE")
            print(json.dumps(dict(schema=SCHEMA, status=status, score_computed=False,
                terminal_slots=6, final_inventory_sha256=digest(encoded(final)), all11_release_allowed=False)), flush=True)
        return 0
    except BaseException as error:
        try:
            with n["deadline"](1):
                n["exclusive_json"](RUN/"FINALIZATION_FAILED.json", dict(schema=SCHEMA,
                    status="TERMINAL_VETO_NO_RELEASE", failure_type=type(error).__name__,
                    failures=failures, no_score_policy=NO_SCORE_POLICY))
        except BaseException:
            pass
        return 1


def main():
    require_release()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--internal-coordinator", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker:
        profile()["worker"]()
    elif args.internal_coordinator:
        internal_coordinator()
    else:
        raise SystemExit(execute())


if __name__ == "__main__":
    main()
