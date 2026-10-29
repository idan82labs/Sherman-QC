"""Prospective 21-outcome V7 delta only; no target import or old132 collection.

Authoring is source-only. Future execution reads just pinned V7/V6 source bytes,
then selects named definitions/constants into an explicit inert namespace. All
data, controls, metadata, identities and publications below are invented doubles.
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
V7 = ROOT / "scripts/run_gold11_mesh_development_v7.py"
V6 = ROOT / "scripts/run_gold11_mesh_development_v6.py"
V7_SHA = "27737eff034891e8f7a6d09ba63f5bd17a3947c67ffd553ddfb73a931fef5cb3"
V6_SHA = "7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80"
DEFINITIONS = (
    "digest", "encoded", "unique_object", "decode", "unavailable", "is_hex",
    "require_release", "arm_request", "arm_request_sha256", "all_arm_requests",
    "scan_plane_requests", "valid_supervisor_refusal", "freeze", "verify_bundle",
    "prestart_tree_admission", "execute", "internal_coordinator", "score_after_terminal",
    "make_scoring_request", "scorer_arguments", "_tree_fingerprint", "contained", "main",
)
CONSTANTS = (
    "ROOT", "RUN", "PROTOCOL", "REVIEW", "REVIEW_SHA", "CATALOG", "CATALOG_SHA", "SELECTED_SHA",
    "SOURCE_BASIS", "SOURCE_BASIS_SHA", "SOURCE_PROFILE", "SOURCES", "RELEASE_AUTHORIZED",
    "TIMING_RELEASE_STATUS", "ARMS", "ARM_DEFINITIONS", "COMPARATOR", "POSITION_ERROR_DECLARATION",
    "LIMITS", "CERTIFICATE_CAPS", "RETAINED_CONTROL_POLICY", "CONFIG", "TWO_PLANE_DECLARATION",
    "CONTEXT_V3_REQUEST", "CONTEXT_REQUESTS_BY_ARM", "SEEDED_PLANE_BUDGETS", "COMPOSITE_PLANE_BUDGETS",
    "SCHEDULE_V2_BUDGETS", "SCORING_SOURCES", "SCORING_TRANSPORT", "ARTIFACT_SOURCE",
    "ARTIFACT_SOURCE_SHA", "ARTIFACT_POLICY", "SUPERVISION_SOURCE", "SUPERVISION_SOURCE_SHA",
    "_COORDINATOR_WRITERS_QUIESCENT", "_COORDINATOR_ABORTED", "LEAF_SCHEMA", "RUNNER_SOURCE",
)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def deny(*_a, **_k):
    raise AssertionError("UNMOCKED_IO_PROCESS_REFERENCE_NATIVE_OR_IMPORT")


@contextmanager
def inert_deadline(*_a):
    yield


class LexicalPath(type(Path())):
    def resolve(self, *_a, **_k):
        return self

    def mkdir(self, *_a, **_k):
        return None

    def lstat(self, *_a, **_k):
        return deny()


@pytest.fixture
def isolated():
    source, basis = V7.read_bytes(), V6.read_bytes()
    assert len(source) == 169028 and sha(source) == V7_SHA
    assert len(basis) == 166049 and sha(basis) == V6_SHA
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in CONSTANTS]
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in DEFINITIONS]
    assert {node.targets[0].id for node in assignments} == set(CONSTANTS)
    assert {node.name for node in definitions} == set(DEFINITIONS)
    events = []
    ns = dict(__name__="_invented_v7_delta", __builtins__=dict(vars(builtins)),
        Path=LexicalPath, math=math, json=json, hashlib=hashlib, stat=stat,
        os=SimpleNamespace(sep="/", getpid=lambda: 22, getppid=lambda: 11,
            getpgrp=lambda: 22, close=deny, environ={}),
        sys=SimpleNamespace(stdin=SimpleNamespace(buffer=SimpleNamespace(read=deny))),
        time=SimpleNamespace(monotonic=lambda: 0.), print=lambda *_a, **_k: None,
        deadline=inert_deadline, argparse=SimpleNamespace(ArgumentParser=deny),
        resource=SimpleNamespace(getrusage=deny), secrets=SimpleNamespace(token_hex=deny))
    ns["__builtins__"].update(open=deny, __import__=deny)
    exec(compile(ast.Module(body=assignments+definitions, type_ignores=[]), "<INERT-V7-DELTA-DEFINITIONS>", "exec"), ns)
    ns.update(ROOT=LexicalPath("/invented/source"), RUN=LexicalPath("/invented/run"),
        bounded_read=deny, owned_read=deny, publish_bytes=deny, exclusive_json=deny,
        runtime_metadata=deny, approved_cases_from_review=deny, marker_present=deny,
        cache_identity=deny, activate_coordinator=deny, supervise_case=deny,
        coordinator_memory_check=deny, terminalize_case_records=deny,
        fresh_private_cache=deny, supervise_process=deny, selected_references=deny,
        reconstruct_all_terminal_inputs=deny, owned_leaf_cache=deny,
        issue_leaf_dispatch=deny, publish_private_scoring_request=deny,
        _open_tree_directory=deny, _tree_names=deny, verify_pre_run_audit=deny)
    return SimpleNamespace(ns=ns, source=source, basis=basis, events=events)


def admit(isolated):
    ns = isolated.ns
    assert ns["RELEASE_AUTHORIZED"] is False
    assert ns["TIMING_RELEASE_STATUS"] == "UNRELEASED_V6_CARRYOVER_VALUES_NOT_V7_APPROVAL"
    ns["require_release"] = lambda: None  # detached pure-boundary admission, never source release
    return ns


def invented_cases():
    return [dict(case_token=f"INVENTED_{i:02}", part_key=f"family_{i % 6}",
        scan_path=f"/invented/never_open/scan_{i}.ply", cad_path=f"/invented/never_open/cad_{i % 6}.step",
        scan_sha256=sha(f"scan{i}".encode()), cad_sha256=sha(f"cad{i % 6}".encode())) for i in range(11)]


def frozen_store(isolated):
    ns, store, events = isolated.ns, {}, isolated.events
    blobs = {name: ("INVENTED_SOURCE:"+name).encode() for name in ns["SOURCES"]}
    blobs[ns["SOURCE_BASIS"]] = isolated.basis
    # Fixture-only helper-store pins: no helper source is read/imported/qualified.
    ns["ARTIFACT_SOURCE_SHA"] = sha(blobs[ns["ARTIFACT_SOURCE"]])
    ns["SUPERVISION_SOURCE_SHA"] = sha(blobs[ns["SUPERVISION_SOURCE"]])
    def read(path, cap, expected=None):
        base = ns["RUN"] / "source" if str(path).startswith(str(ns["RUN"] / "source")) else ns["ROOT"]
        if path == ns["RUN"] / "FROZEN.json":
            value = store["FROZEN.json"]
        else:
            name = path.relative_to(base).as_posix()
            value = store["source/"+name] if base != ns["ROOT"] else blobs[name]
        assert len(value) <= cap
        if expected is not None and sha(value) != expected:
            raise ValueError("INVENTED_SOURCE_STORE_HASH_MISMATCH")
        events.append(("source-read", str(path)))
        return value
    def publish(path, value):
        name = path.relative_to(ns["RUN"]).as_posix()
        assert name not in store
        store[name] = value
    ns.update(bounded_read=read, publish_bytes=publish,
        exclusive_json=lambda path, value: publish(path, wire(value)+b"\n"),
        runtime_metadata=lambda: {"scope": "INVENTED_RUNTIME"},
        approved_cases_from_review=lambda _blob: invented_cases())
    ns["freeze"]()
    bundle = ns["verify_bundle"]()
    assert len(bundle["cases"]) == 11 and bundle["source_profile"] == ns["SOURCE_PROFILE"]
    assert len(wire(bundle)) < 128 * 1024
    return bundle, store, blobs


@pytest.mark.parametrize("state", ("release", "basis", "profile"))
def test_release_and_actual_freeze_verify_basis_profile_contracts(isolated, state):
    ns = isolated.ns
    if state == "release":
        for name, args in (("freeze", ()), ("verify_bundle", ()), ("prestart_tree_admission", ()),
                ("execute", ()), ("internal_coordinator", ()), ("score_after_terminal", ({}, {}, {})), ("main", ())):
            with pytest.raises(ValueError, match="V7_DRAFT_BLOCKED_NOT_FROZEN_OR_AUTHORIZED"):
                ns[name](*args)
        ns["RELEASE_AUTHORIZED"] = True  # only guard test; no source/profile invocation
        with pytest.raises(ValueError, match="V7_TIMING_DECLARATION_NOT_RELEASED"):
            ns["require_release"]()
        assert isolated.events == []
        return
    admit(isolated)
    bundle, store, blobs = frozen_store(isolated)  # actual positive freeze/verify first
    if state == "basis":
        blobs[ns["SOURCE_BASIS"]] += b"changed"
        with pytest.raises(ValueError, match="V7_FROZEN_V6_SOURCE_BASIS_CHANGED"):
            ns["freeze"]()
        bundle["source_sha256"][ns["SOURCE_BASIS"]] = "0" * 64
        reason = "V7_FROZEN_V6_SOURCE_BASIS_CHANGED"
    else:
        bundle["source_profile"] = dict(bundle["source_profile"], implementation="INVENTED_WRONG_PROFILE")
        reason = "FROZEN_PROTOCOL_CONSTANTS_CHANGED"
    store["FROZEN.json"] = wire(bundle)
    with pytest.raises(ValueError, match=reason):
        ns["verify_bundle"]()


def topology(isolated, state):
    ns, events = isolated.ns, isolated.events
    files = {"FROZEN.json", "PRE_RUN_AUDIT.json"} | {"source/"+name for name in ns["SOURCES"]}
    directories = {"", "source"}
    for path in files:
        parts = path.split("/")
        directories.update("/".join(parts[:i]) for i in range(1, len(parts)))
    assert len(ns["SOURCES"]) == 70 and len(directories) == 7
    first = "source/"+ns["RUNNER_SOURCE"]
    symlink = None
    if state == "root_extra": files.add("UNKNOWN.json")
    if state == "spent_root": files.add("STARTED.json")
    if state == "missing_source": files.remove(first)
    if state == "file_symlink": symlink = first
    if state == "directory_symlink": symlink = "source/backend"
    if state == "declaration_conflict": ns["SOURCES"] += (ns["RUNNER_SOURCE"]+"/child.py",)
    paths, descriptors, enumerations, stats = files | directories, {}, {}, {}
    def names(relative):
        prefix = relative+"/" if relative else ""
        return sorted(path[len(prefix):] for path in paths if path != relative
            and path.startswith(prefix) and "/" not in path[len(prefix):])
    def opened(relative):
        assert relative in directories
        fd = len(descriptors)+1
        descriptors[fd] = relative
        events.append(("open", fd))
        return fd
    def info(path):
        mode = stat.S_IFDIR if path in directories else stat.S_IFREG
        if path == symlink: mode = stat.S_IFLNK
        return SimpleNamespace(st_dev=1, st_ino=1, st_mode=mode, st_nlink=1,
            st_size=0, st_mtime_ns=1, st_ctime_ns=1)
    def fstat(fd):
        stats[fd] = stats.get(fd, 0)+1
        row = info(descriptors[fd])
        if state == "changed_fingerprint" and stats[fd] > 1:
            row.st_mtime_ns = 2
        return row
    def enumerate_names(fd, cap):
        enumerations[fd] = enumerations.get(fd, 0)+1
        result = names(descriptors[fd])
        if state == "changed_second_names" and enumerations[fd] > 1:
            result = sorted(result+["INVENTED_LATE_NAME"])
        return result[:cap], result[cap] if len(result) > cap else None
    ns.update(_open_tree_directory=opened, _tree_names=enumerate_names)
    ns["os"] = SimpleNamespace(fstat=fstat,
        stat=lambda name, dir_fd, follow_symlinks: info((descriptors[dir_fd]+"/" if descriptors[dir_fd] else "")+name),
        close=lambda fd: events.append(("close", fd)))


TREE_STATES = ("valid", "root_extra", "spent_root", "missing_source", "file_symlink",
    "directory_symlink", "changed_second_names", "changed_fingerprint", "declaration_conflict")


@pytest.mark.parametrize("state", TREE_STATES)
def test_actual_tree_admission_in_execute_precedes_started_and_dispatch(isolated, state):
    ns = admit(isolated)
    topology(isolated, "valid")
    ns["prestart_tree_admission"]()  # exact same source declaration positive premise
    assert len([event for event in isolated.events if event[0] == "open"]) == 7
    isolated.events.clear()
    topology(isolated, state)
    bundle, audit = {"invented_bundle": True}, {"invented_pre": True}
    ns.update(verify_bundle=lambda: bundle, verify_pre_run_audit=lambda _: audit,
        owned_read=lambda name, *_a: wire(bundle if name == "FROZEN.json" else audit),
        marker_present=lambda _: False)
    publications = []
    def publication(path, value):
        publications.append(path.name)
        if path.name == "STARTED.json":
            raise RuntimeError("INVENTED_STOP_AFTER_ADMISSION_BEFORE_ANY_DISPATCH")
    ns["exclusive_json"] = publication
    if state == "valid":
        assert ns["execute"]() == 1  # deliberate inert STARTED publisher stop
        assert publications == ["STARTED.json", "FINALIZATION_FAILED.json"]
    else:
        reason = ("V7_PRESTART_SOURCE_TREE_DECLARATION_CONFLICT" if state == "declaration_conflict" else
            "V7_PRESTART_TREE_DIRECTORY_CHANGED" if state.startswith("changed_") else
            "V7_PRESTART_TREE_TYPE" if state.endswith("symlink") else
            "V7_PRESTART_UNDECLARED_MISSING_OR_SPENT_TREE_NAME")
        with pytest.raises(ValueError, match=reason):
            ns["execute"]()
        assert publications == []
    assert [fd for kind, fd in isolated.events if kind == "open"] == [fd for kind, fd in isolated.events if kind == "close"]


def coordinator(isolated, binding):
    ns, events = isolated.ns, isolated.events
    bundle, audit = dict(cases=invented_cases(), arms=ns["ARMS"]), {"invented_pre": True}
    cache, identity, token = LexicalPath("/invented/private/coordinator"), dict(device=1, inode=2, uid=3), "a"*64
    expected = dict(parent_pid=11, token_sha256=sha(token.encode()), cache=str(cache), cache_identity=identity)
    start = dict(status="STARTED_NO_RETRY", limits=ns["LIMITS"], bundle_sha256=sha(wire(bundle)),
        audit_sha256=sha(wire(audit)), arm_requests_sha256=sha(wire(ns["all_arm_requests"]())))
    class Paths:
        def __new__(cls, value): return LexicalPath(value)
        @staticmethod
        def cwd(): return cache
    ns.update(Path=Paths, verify_bundle=lambda: bundle, verify_pre_run_audit=lambda _: audit,
        cache_identity=lambda _: identity, bounded_read=lambda path, *_a: wire(expected if path.name == "COORDINATOR_DISPATCH.json" else start),
        marker_present=lambda _: False, activate_coordinator=lambda *_a: events.append("activate"),
        coordinator_memory_check=lambda: events.append("memory"),
        score_after_terminal=lambda *_a: events.append("scorer"))
    ns["os"].environ = {"HOME": str(cache), "TMPDIR": str(cache)}
    ns["sys"].stdin.buffer.read = lambda _: wire(dict(token=token, parent_pid=11))
    retained, publications = {}, {}
    outcome = ns["unavailable"]("INVENTED_NEGATIVE_ORIGINAL_REASON")
    original = dict(status="TERMINAL", worker_dispatch_attempted=False,
        derived_result_retention_complete=True, private_cache_cleanup_complete=True,
        supervisor_refusal=outcome)
    if binding != "missing": original["source_binding_valid"] = binding
    original_bytes = wire(original)
    def leaf(case, arm, destination, _remaining):
        events.append(("leaf", case["case_token"], arm))
        retained[str(destination)] = original_bytes
        return deepcopy(outcome), deepcopy(original)
    def fill(b, reason):
        return [dict(case_id=case["case_token"], arm=arm, durable=True, reason=reason)
            for case in b["cases"] for arm in b["arms"]]
    ns.update(supervise_case=leaf, terminalize_case_records=fill,
        exclusive_json=lambda path, value: publications.__setitem__(path.name, deepcopy(value)))
    return publications, retained, original_bytes


@pytest.mark.parametrize("binding", (True, False, "missing", None, 1, "true"))
def test_binding_boolean_identity_stops_later_work_and_preserves_original_negative(isolated, binding):
    ns = admit(isolated)
    actual_score = ns["score_after_terminal"]
    # Independently prove the complete authenticated coordinator premise first.
    publications, retained, original = coordinator(isolated, True)
    ns["internal_coordinator"]()
    assert len(retained) == 22 and isolated.events.count("scorer") == 1
    assert publications["INFERENCE_TERMINAL.json"]["terminal_record_count"] == 22
    if binding is True:
        assert all(value == original for value in retained.values())
        return
    isolated.events.clear()
    publications, retained, original = coordinator(isolated, binding)
    ns["internal_coordinator"]()
    assert len(retained) == 1 and list(retained.values()) == [original]
    assert "memory" not in isolated.events and "scorer" not in isolated.events
    refusal = publications["SCORING_UNAVAILABLE.json"]
    assert refusal["reason"] == "INPUT_BINDING_NOT_ESTABLISHED_ABORT_NO_FURTHER_DISPATCH_OR_SCORING"
    assert len(refusal["terminal_case_records"]) == 22
    assert "INFERENCE_TERMINAL.json" not in publications and ns["_COORDINATOR_ABORTED"] is True
    assert json.loads(original)["supervisor_refusal"]["reasons"] == ["INVENTED_NEGATIVE_ORIGINAL_REASON"]
    with pytest.raises(ValueError, match="NO_SCORING_OR_REFERENCE_ACCESS_AFTER_ABORT_OR_CUSTODY_LOSS"):
        actual_score({}, {}, {})  # actual scorer entry refuses before any reference seam


def request_fixture(isolated):
    ns = isolated.ns
    bundle = dict(cases=invented_cases(), arms=ns["ARMS"])
    selected = {case["case_token"]: dict(bends=[{"invented": i} for i in range(7 if index < 5 else 6)])
        for index, case in enumerate(bundle["cases"])}
    assert sum(len(row["bends"]) for row in selected.values()) == 71
    inner, receipts = ({arm: {} for arm in ns["ARMS"]} for _ in range(2))
    for arm in ns["ARMS"]:
        for case in bundle["cases"]:
            key = case["case_token"]
            inner[arm][key] = ns["unavailable"]("INVENTED_BOUND_NEGATIVE")
            receipts[arm][key] = dict(scoring_input_sha256=sha(wire(inner[arm][key])),
                projection_kind="SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW", source_binding_valid=True)
    request = ns["make_scoring_request"](bundle, selected, inner, receipts)
    arguments = ns["scorer_arguments"](request)
    assert len(wire(request)) < 128*1024 and sum(map(len, request["projection_bindings"].values())) == 22
    assert arguments["control_results_by_case"] == inner["control"]
    assert arguments["challenger_results_by_case"] == inner["challenger"]
    return bundle, selected, inner, receipts, request


class LengthOnlyEncodedRequest:
    """Comparison/order double only: neither bytes nor an allocated large payload."""
    def __init__(self, length): self.length = length
    def __len__(self): return self.length


class ScorerBoundaryReached(RuntimeError):
    pass


@pytest.mark.parametrize("state", ("tiny_all22", "exact_cap", "cap_plus_one"))
def test_actual_all22_request_consumer_and_cap_precedes_scorer_cache(isolated, state):
    ns = admit(isolated)
    bundle, selected, inner, receipts, request = request_fixture(isolated)
    if state == "tiny_all22":
        assert request["schema"] == "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6"
        return
    terminal = dict(case_count=11, arm_count=2, terminal_record_count=22,
        outcomes_sha256=sha(wire(inner)), arm_requests_sha256=sha(wire(ns["all_arm_requests"]())))
    ns.update(verify_bundle=lambda: bundle, reconstruct_all_terminal_inputs=lambda _: (inner, inner, receipts),
        owned_read=lambda *_a: wire(terminal), selected_references=lambda _: selected,
        publish_bytes=lambda path, blob: isolated.events.append(("publish", path.name)),
        exclusive_json=lambda path, row: isolated.events.append(("publish", path.name)))
    cap = ns["LIMITS"]["scorer_request_bytes"]
    assert cap == 192*1024**2 and 22*ns["LIMITS"]["compact_case_bytes"] == 176*1024**2
    actual_encoder = ns["encoded"]
    declared_length = [cap]
    def length_encoder(value):
        if type(value) is dict and value.get("schema") == request["schema"]:
            assert ns["scorer_arguments"](value) == ns["scorer_arguments"](request)
            isolated.events.append("encoded-full-request-length-double")
            return LengthOnlyEncodedRequest(declared_length[0])
        return actual_encoder(value)
    ns["encoded"] = length_encoder
    @contextmanager
    def cache(role, receipt):
        assert role == "SCORER"
        isolated.events.append("scorer-cache-boundary")
        raise ScorerBoundaryReached("INVENTED_STOP_BEFORE_CACHE_CREATION_OR_DISPATCH")
        yield  # generator shape only; no cache/path exists
    ns["owned_leaf_cache"] = cache
    if state == "cap_plus_one":
        with pytest.raises(ScorerBoundaryReached, match="INVENTED_STOP_BEFORE_CACHE_CREATION_OR_DISPATCH"):
            ns["score_after_terminal"](bundle, inner, receipts)  # exact-cap positive premise
        assert "scorer-cache-boundary" in isolated.events
        isolated.events.clear()
        declared_length[0] = cap+1
    expected = ScorerBoundaryReached if state == "exact_cap" else ValueError
    reason = "INVENTED_STOP_BEFORE_CACHE_CREATION_OR_DISPATCH" if state == "exact_cap" else "SCORER_REQUEST_BYTE_BUDGET"
    with pytest.raises(expected, match=reason):
        ns["score_after_terminal"](bundle, inner, receipts)
    assert "encoded-full-request-length-double" in isolated.events
    if state == "exact_cap":
        assert isolated.events.index("encoded-full-request-length-double") < isolated.events.index("scorer-cache-boundary")
        assert ("publish", "SCORER_RECEIPT.json") in isolated.events
    else:
        assert "scorer-cache-boundary" not in isolated.events
        assert ("publish", "SCORER_RECEIPT.json") not in isolated.events
