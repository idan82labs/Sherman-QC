"""Prospective new-delta diagnostic gate: no previous132 suite is collected.

Only the two named Python source buffers are read. The real diagnostic profile
constructs its exact V6 AST allowlist inside an inert namespace; no provider
module, backend, helper, sampler, process or native library is imported. All
authority/data files below are invented in-memory dictionaries. This file was
authored source-only, without imports, compilation, collection or execution.
"""
import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "scripts/run_gold11_context_v3_diagnostic_v1.py"
PROVIDER = ROOT / "scripts/run_gold11_mesh_development_v6.py"
DIAGNOSTIC_SHA = "08f9e50b52f0225ca54e1920ef21fed2d0840a19941fb20b338b43e6b141a44f"
PROVIDER_SHA = "7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80"
FUNCTIONS = (
    "encoded", "digest", "require_release", "bootstrap_deadline", "bootstrap_read", "_assignment",
    "source_profile_declaration", "profile", "selected_cases_from_review", "publication_cap",
    "leaf_sources", "validate_leaf_binding", "fresh_private_cache", "runner_command",
    "frozen_bundle", "freeze", "verify_bundle", "verify_pre", "six_slots", "activate_coordinator",
    "internal_coordinator", "bind_closed_log", "reconstruct_six", "campaign_schema",
    "source_tree_shape", "source_postflight", "execute", "main",
)
CONSTANTS = (
    "ROOT", "RUNNER_SOURCE", "RUN", "PROTOCOL", "DESIGN", "V6_SOURCE", "V6_SHA",
    "REVIEW_SOURCE", "REVIEW_SHA", "RELEASE_AUTHORIZED", "TIMING_RELEASE_STATUS",
    "SCHEMA", "LEAF_SCHEMA", "SELECTED_ORDER", "SELECTION_ROLES", "ARMS",
    "DIAGNOSTIC_LIMITS", "NO_SCORE_POLICY", "V6_CONSTANTS", "V6_FUNCTIONS", "V6_CLASSES",
    "FORBIDDEN_EXTRACTIONS", "CALLBACK_PROFILE", "_PROFILE",
)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def deny(*_args, **_kwargs):
    raise AssertionError("UNMOCKED_IO_AUTHORITY_PROCESS_NATIVE_OR_IMPORT")


@contextmanager
def inert_deadline(_seconds=None):
    yield


class LexicalPath(type(Path())):
    """No host path resolution, stat, mkdir or other filesystem operation."""
    mkdir_events = []

    def resolve(self, *_a, **_k):
        return self

    def mkdir(self, *_a, **_k):
        self.mkdir_events.append(str(self))

    def lstat(self, *_a, **_k):
        return deny()


def read_source(path, expected):
    assert path in (DIAGNOSTIC, PROVIDER)
    data = path.read_bytes()
    assert len(data) < 256 * 1024 and sha(data) == expected
    return data


@pytest.fixture
def sandbox():
    candidate = read_source(DIAGNOSTIC, DIAGNOSTIC_SHA)
    provider = read_source(PROVIDER, PROVIDER_SHA)
    tree = ast.parse(candidate)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in CONSTANTS]
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert {node.targets[0].id for node in assignments} == set(CONSTANTS)
    assert {node.name for node in definitions} == set(FUNCTIONS)
    empty = SimpleNamespace()
    ns = dict(__builtins__=dict(vars(builtins)), __name__="_invented_diagnostic_definitions",
        __file__="/invented/live/scripts/run_gold11_context_v3_diagnostic_v1.py",
        ast=ast, Path=LexicalPath, contextmanager=contextmanager, asdict=asdict,
        hashlib=hashlib, json=json, math=math, stat=stat,
        argparse=empty, fcntl=empty, import_metadata=empty, selectors=empty,
        subprocess=SimpleNamespace(Popen=deny), shutil=SimpleNamespace(rmtree=deny),
        tempfile=SimpleNamespace(mkdtemp=deny), ModuleType=deny,
        resource=SimpleNamespace(getrusage=deny, setrlimit=deny),
        signal=SimpleNamespace(ITIMER_REAL=0, SIGALRM=14, getitimer=lambda _: (30., 0.), signal=deny, setitimer=deny),
        secrets=SimpleNamespace(token_hex=deny), time=SimpleNamespace(monotonic=lambda: 0.),
        os=SimpleNamespace(sep="/", getpid=lambda: 22, getppid=lambda: 11, getpgrp=lambda: 22,
            getuid=lambda: 123, close=lambda _: None, open=deny, read=deny),
        sys=SimpleNamespace(platform="darwin", modules={}, stdin=SimpleNamespace(buffer=SimpleNamespace(read=deny))),
        print=lambda *_a, **_k: None)
    ns["__builtins__"].update(open=deny, __import__=deny)
    exec(compile(ast.Module(body=assignments + definitions, type_ignores=[]), "<INVENTED-DIAGNOSTIC-DEFINITIONS>", "exec"), ns)
    ns.update(ROOT=LexicalPath("/invented/source"), RUN=LexicalPath("/invented/diagnostic"))
    LexicalPath.mkdir_events = []
    return SimpleNamespace(ns=ns, tree=tree, provider=provider, candidate=candidate, events=[])


def admit_profile(sandbox):
    ns, events = sandbox.ns, sandbox.events
    assert ns["RELEASE_AUTHORIZED"] is False
    assert ns["TIMING_RELEASE_STATUS"] == "UNRELEASED_PROSPECTIVE_SIX_SLOT_LIMITS"
    ns["require_release"] = lambda: events.append("detached-guard")
    def source(path, cap, expected):
        assert path == ns["ROOT"] / ns["V6_SOURCE"] and cap == 256 * 1024
        assert expected == PROVIDER_SHA and sha(sandbox.provider) == expected
        events.append("provider-buffer")
        return sandbox.provider
    ns["bootstrap_read"] = source
    n = ns["profile"]()
    # Every operation below is inert until a test explicitly supplies its seam.
    n.update(deadline=inert_deadline, artifact_directory_walk=inert_deadline,
        artifact_alarm_guard=inert_deadline, bounded_read=deny, _bounded_read=deny,
        owned_read=deny, marker_present=deny, publish_bytes=deny, exclusive_json=deny,
        artifact_module=deny, supervision_module=deny, projection_module=deny,
        runtime_metadata=deny, _open_tree_directory=deny, _tree_names=deny,
        cache_identity=deny, coordinator_memory_check=deny)
    return n


def invented_review(sandbox):
    order = sandbox.ns["SELECTED_ORDER"]
    rows = []
    for index in range(11):
        rows.append(dict(case_token=order[index] if index < 3 else f"INVENTED{index:02}",
            part_key=f"invented_part_{index % 6}", reference_id=f"UNREAD_REFERENCE_{index}",
            scan_path=f"/invented/authority/scan_{index}.ply", scan_sha256=sha(f"scan-{index}".encode()),
            cad_path=f"/invented/authority/cad_{index % 6}.step", cad_sha256=sha(f"cad-{index % 6}".encode())))
    return rows


def review_bytes(rows):
    result = b"CASES = json.loads(r'''" + wire(rows) + b"''')\n"
    assert len(result) < 16 * 1024
    return result


def authorize_invented_review(sandbox, rows=None):
    ns = sandbox.ns
    rows = invented_review(sandbox) if rows is None else rows
    blob = review_bytes(rows)
    # Explicitly local authority double. The production literal is checked once
    # before replacement; no physical review source or input path is opened.
    if ns["REVIEW_SHA"] == "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955":
        sandbox.events.append("invented-authority-installed")
    ns["REVIEW_SHA"] = sha(blob)
    cases, authority = ns["selected_cases_from_review"](blob)
    return blob, cases, authority


def bundle_for(sandbox, n):
    review, cases, authority = authorize_invented_review(sandbox)
    blobs = {name: b"INVENTED_SOURCE:" + name.encode() for name in n["SOURCES"]}
    blobs[sandbox.ns["V6_SOURCE"]] = sandbox.provider
    blobs[sandbox.ns["REVIEW_SOURCE"]] = review
    result = sandbox.ns["frozen_bundle"](blobs, cases, authority, {"scope": "INVENTED_RUNTIME_METADATA"})
    assert len(wire(result)) < 64 * 1024
    return result, blobs


def clean_receipt(**extra):
    result = dict(supervisor_schema="GOLD11_OWNED_PROCESS_SUPERVISION_V3", direct_child_reaped=True,
        owned_group_empty_confirmed=True, requires_process_attention=False,
        observed_foreign_group_escape_detected=False,
        observed_foreign_group_escape_unconfirmed_after_cleanup=False,
        observed_identity_unknown_still_present=False, cleanup_errors=[], cleanup_errors_discarded=0,
        control_close_errors=[], exit_code=0, budget_stop=None, child_pid=22,
        log_created=True, forwarded_log_bytes=0, stored_log_bytes=0)
    result.update(extra)
    return result


def terminal_fixture(sandbox, n, *, one_clean=False):
    bundle, _ = bundle_for(sandbox, n)
    storage, projections = {}, []
    for case in bundle["cases"]:
        for arm in sandbox.ns["ARMS"]:
            prefix = "cases/" + case["case_token"] + "/" + arm + "/"
            refusal = n["unavailable"]("INVENTED_REFUSAL", "GLOBAL_TERMINALIZATION")
            projection = wire(refusal) + b"\n"
            row = dict(status="TERMINAL", arm_request_sha256=n["arm_request_sha256"](arm),
                scoring_input_sha256=sha(projection), supervisor_refusal=refusal,
                projection_kind="SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW")
            if one_clean and not storage:
                original = b' \n{"INVENTED_RESULT":true}\n '
                projection = b'{"INVENTED_V2_WRAPPER":true}\n'
                row.update(clean_receipt(), source_binding_valid=True, derived_result_retention_complete=True,
                    private_cache_cleanup_complete=True, result_sha256=sha(original),
                    scoring_input_sha256=sha(projection), projection_kind="EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER",
                    retained_result_artifacts=dict(status="COMPLETE_RESULT_BYTES_RETAINED",
                        artifacts=[dict(path="RESULT.json", sha256=sha(original), bytes=len(original))]))
                row.pop("supervisor_refusal")
                storage[prefix + "RESULT.json"] = original
                storage[prefix + "OUTPUT.txt"] = b""
            storage[prefix + "COMPLETED.json"] = wire(row) + b"\n"
            storage[prefix + "SCORING_INPUT.json"] = projection
    def read(relative, cap, expected=None, **_kwargs):
        data = storage[relative]
        assert len(data) <= cap
        if expected is not None and sha(data) != expected:
            raise ValueError("INVENTED_OWNED_HASH_MISMATCH")
        sandbox.events.append(("owned-read", relative))
        return data
    def revalidate(data, **kwargs):
        assert data == b'{"INVENTED_V2_WRAPPER":true}\n'
        assert kwargs["result_bytes"] == b' \n{"INVENTED_RESULT":true}\n '
        assert kwargs["expected_context_request"] == n["CONTEXT_V3_REQUEST"]
        assert kwargs["expected_projection_sha256"] == sha(data)
        projections.append(kwargs)
        return {"scorer_input": {"INVENTED_RECONSTRUCTED_CONTACT": True}}
    n.update(owned_read=read, marker_present=lambda path: path.relative_to(sandbox.ns["RUN"]).as_posix() in storage,
        projection_module=lambda: SimpleNamespace(revalidate_gold11_schedule_projection_v2=revalidate))
    return bundle, storage, projections


@pytest.mark.parametrize("entry", ("profile", "freeze", "execute", "internal_coordinator", "main", "bootstrap_read"))
def test_false_guard_precedes_diagnostic_entry_effects(sandbox, entry):
    args = (None, None, None) if entry == "bootstrap_read" else ()
    with pytest.raises(ValueError, match="DIAGNOSTIC_V1_SOURCE_DRAFT_NOT_RELEASED"):
        sandbox.ns[entry](*args)
    assert sandbox.events == [] and LexicalPath.mkdir_events == []


def test_local_boolean_alone_cannot_release_prospective_timing(sandbox):
    ns = sandbox.ns
    ns["RELEASE_AUTHORIZED"] = True  # local guard-only invocation, no profile call
    with pytest.raises(ValueError, match="DIAGNOSTIC_V1_TIMING_NOT_RELEASED"):
        ns["require_release"]()
    assert sandbox.events == []


def test_actual_closed_profile_resolves_55_functions_2_classes_and_callbacks(sandbox):
    n = admit_profile(sandbox)
    ns = sandbox.ns
    assert len(ns["V6_FUNCTIONS"]) == 55 and len(ns["V6_CLASSES"]) == 2
    assert len(ns["V6_CONSTANTS"]) == 21 and len(ns["CALLBACK_PROFILE"]) == 7
    assert len(n["SOURCES"]) == len(set(n["SOURCES"])) == 64
    assert sum(path.startswith("backend/") for path in n["SOURCES"]) == 50
    for name in ns["V6_FUNCTIONS"]:
        # A few deliberately replaced I/O functions are saved by this check's
        # independent second real profile, before operation doubles are applied.
        assert name in n
    ns["_PROFILE"] = None
    original = ns["profile"]()
    for name in ns["V6_FUNCTIONS"]:
        function = original[name]
        while hasattr(function, "__wrapped__"):
            function = function.__wrapped__
        assert function.__globals__ is original
        assert function.__code__.co_filename == str(ns["ROOT"] / ns["V6_SOURCE"])
    assert all(isinstance(original[name], type) for name in ns["V6_CLASSES"])
    for callback in ns["CALLBACK_PROFILE"]:
        own_name = "campaign_schema" if callback == "_campaign_schema" else callback
        assert original[callback] is ns[own_name]
    assert not any(name in original for name in ("CATALOG", "CATALOG_SHA", "SELECTED_SHA", "scorer_worker", "score_after_terminal"))
    assert original["DIAGNOSTIC_PROFILE_SHA"] == sha(wire(original["DIAGNOSTIC_PROFILE"]))
    assert original["DIAGNOSTIC_PROFILE"]["selected_case_order"] == list(ns["SELECTED_ORDER"])
    assert original["STAGE_SAMPLE_SOURCE_PATHS"][-2:] == (ns["V6_SOURCE"], ns["RUNNER_SOURCE"])
    assert ns["sys"].modules == {} and ns["RELEASE_AUTHORIZED"] is False


@pytest.mark.parametrize("change", ("missing_function", "forbidden_function"))
def test_profile_closed_extraction_rejects_delta_after_positive_premise(sandbox, change):
    admit_profile(sandbox)
    sandbox.ns["_PROFILE"] = None
    additional = "invented_missing_function" if change == "missing_function" else "scorer_worker"
    sandbox.ns["V6_FUNCTIONS"] += (additional,)
    with pytest.raises(ValueError, match="DIAGNOSTIC_EXACT_UNCHANGED_DEFINITION_DENOMINATOR"):
        sandbox.ns["profile"]()
    assert sandbox.ns["_PROFILE"] is None


@pytest.mark.parametrize("change", ("valid", "hash", "count", "duplicate", "source_form", "hash_conflict", "missing_selected"))
def test_source_cases_parser_uses_only_invented_11_rows_17_path_authority(sandbox, change):
    admit_profile(sandbox)
    blob, cases, authority = authorize_invented_review(sandbox)
    assert len(authority) == 17 and [row["case_token"] for row in cases] == list(sandbox.ns["SELECTED_ORDER"])
    assert all("reference_id" not in row for row in cases)
    if change == "valid":
        assert all(path.startswith("/invented/") for path in authority)
        return
    rows = invented_review(sandbox)
    if change == "hash":
        blob += b" "
        expected = "DIAGNOSTIC_SEPTEMBER9_REVIEW_SOURCE_HASH"
    else:
        if change == "count":
            rows.pop()
            expected = "DIAGNOSTIC_ORIGINAL_ELEVEN_CASE_AUTHORITY"
        elif change == "duplicate":
            rows[1]["case_token"] = rows[0]["case_token"]
            expected = "DIAGNOSTIC_ORIGINAL_CASE_DUPLICATE"
        elif change == "hash_conflict":
            rows[6]["cad_sha256"] = "0" * 64
            expected = "DIAGNOSTIC_ORIGINAL_INPUT_HASH_CONFLICT"
        elif change == "missing_selected":
            rows[0]["case_token"] = "INVENTED_MISSING_SELECTED"
            expected = "DIAGNOSTIC_EXACT_RAW17_SOURCE_AUTHORITY"
        else:
            expected = "DIAGNOSTIC_FIXED_REVIEW_CASES_SCHEMA"
        blob = review_bytes(rows)
        if change == "source_form":
            blob = blob.replace(b"json.loads", b"json.dumps")
        sandbox.ns["REVIEW_SHA"] = sha(blob)  # intended parser branch, invented bytes only
    with pytest.raises(ValueError, match=expected):
        sandbox.ns["selected_cases_from_review"](blob)
    assert not any(type(event) is tuple and event[0] == "owned-read" for event in sandbox.events)


def freeze_fixture(sandbox, n):
    bundle, blobs = bundle_for(sandbox, n)
    # These three local helper pins belong only to the invented source-store
    # double. Production literals remain pinned in the real provider source;
    # no helper code is imported, compiled, invoked or represented as qualified.
    for source_key, pin_key in (("ARTIFACT_SOURCE", "ARTIFACT_SOURCE_SHA"),
            ("SUPERVISION_SOURCE", "SUPERVISION_SOURCE_SHA"), ("STAGE_SAMPLER_SOURCE", "STAGE_SAMPLER_SHA")):
        n[pin_key] = sha(blobs[n[source_key]])
    store, reads = {}, []
    def read(path, cap, expected=None):
        base = sandbox.ns["RUN"] / "source" if str(path).startswith(str(sandbox.ns["RUN"])) else sandbox.ns["ROOT"]
        relative = path.relative_to(base).as_posix()
        value = store["source/" + relative] if base != sandbox.ns["ROOT"] else blobs[relative]
        if len(value) > cap or expected is not None and sha(value) != expected:
            raise ValueError("INVENTED_DECLARED_SOURCE_HASH_MISMATCH")
        reads.append(str(path))
        return value
    def publish(path, value):
        relative = path.relative_to(sandbox.ns["RUN"]).as_posix()
        if relative in store:
            raise FileExistsError("invented exclusive publication")
        store[relative] = value
    n.update(bounded_read=read, owned_read=lambda name, *_a, **_k: store[name],
        publish_bytes=publish, exclusive_json=lambda path, value: publish(path, wire(value) + b"\n"),
        runtime_metadata=lambda: {"scope": "INVENTED_RUNTIME_METADATA"})
    return store, reads


@pytest.mark.parametrize("change", ("valid", "profile", "source_hash"))
def test_real_freeze_verify_round_trip_in_exact_64_slot_invented_store(sandbox, change):
    n = admit_profile(sandbox)
    store, reads = freeze_fixture(sandbox, n)
    sandbox.ns["freeze"]()
    actual = sandbox.ns["verify_bundle"]()
    assert actual["terminal_slots"] == 6 and len(actual["original_raw17_authority"]) == 17
    assert set(store) == {"source/" + name for name in n["SOURCES"]} | {"FROZEN.json"}
    assert all(path.startswith((str(sandbox.ns["ROOT"]), str(sandbox.ns["RUN"]))) for path in reads)
    assert not any("/invented/authority/" in path for path in reads)
    if change == "valid":
        assert actual["no_score_policy"] == sandbox.ns["NO_SCORE_POLICY"]
        return
    if change == "profile":
        frozen = json.loads(store["FROZEN.json"])
        frozen["terminal_slots"] = 22
        store["FROZEN.json"] = wire(frozen) + b"\n"
        expected = "DIAGNOSTIC_FROZEN_PROFILE_OR_INPUT_AUTHORITY_CHANGED"
    else:
        store["source/" + n["SOURCES"][0]] += b"changed"
        expected = "INVENTED_DECLARED_SOURCE_HASH_MISMATCH"
    with pytest.raises(ValueError, match=expected):
        sandbox.ns["verify_bundle"]()


@pytest.mark.parametrize("change", ("valid", "profile", "slots", "score"))
def test_exact_six_slot_pre_contract_with_positive_premise(sandbox, change):
    n = admit_profile(sandbox)
    bundle, _ = bundle_for(sandbox, n)
    audit = dict(decision="GREEN_FOR_EXACT_UNSCORED_DIAGNOSTIC_V1", bundle_sha256=sha(wire(bundle)),
        source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], selected_case_order=list(sandbox.ns["SELECTED_ORDER"]),
        terminal_slots=6, limits=sandbox.ns["DIAGNOSTIC_LIMITS"], no_score_policy=sandbox.ns["NO_SCORE_POLICY"],
        reviewer="invented independent reviewer", blocking_findings=[])
    n["owned_read"] = lambda *_a, **_k: wire(audit)
    assert sandbox.ns["verify_pre"](bundle) == audit
    if change == "valid":
        return
    if change == "profile":
        audit["source_profile_sha256"] = "0" * 64
    elif change == "slots":
        audit["terminal_slots"] = 22
    else:
        audit["no_score_policy"] = dict(audit["no_score_policy"], scorer_dispatch=True)
    with pytest.raises(ValueError, match="DIAGNOSTIC_EXACT_INDEPENDENT_PRE_REQUIRED"):
        sandbox.ns["verify_pre"](bundle)


@pytest.mark.parametrize("attempt", ("leaf", "command", "publication"))
def test_actual_callback_profile_excludes_every_scorer_route(sandbox, attempt):
    n = admit_profile(sandbox)
    assert n["leaf_sources"]("INFERENCE")[:2] == (sandbox.ns["RUNNER_SOURCE"], sandbox.ns["V6_SOURCE"])
    assert n["runner_command"]("--worker")[-1] == "--worker"
    assert n["publication_cap"](sandbox.ns["RUN"] / "cases/invented/control/RESULT.json") == 128 * 1024**2
    reason = dict(leaf="DIAGNOSTIC_INFERENCE_ONLY_NO_SCORER",
        command="DIAGNOSTIC_NO_SCORER_OR_OTHER_CHILD_MODE",
        publication="DIAGNOSTIC_SCORING_PUBLICATION_FORBIDDEN")[attempt]
    with pytest.raises(ValueError, match=reason):
        if attempt == "leaf":
            n["leaf_sources"]("SCORER")
        elif attempt == "command":
            n["runner_command"]("--score-worker")
        else:
            n["publication_cap"](sandbox.ns["RUN"] / "SCORED.json")


def memory_tree(sandbox, n, files, directories, data, *, extra=None, symlink=None):
    paths = set(files) | set(directories)
    if extra is not None:
        paths.add(extra)
        data[extra] = b"UNKNOWN_CONTENT_MUST_NOT_BE_READ"
    identities = {path: index + 1 for index, path in enumerate(sorted(paths))}
    descriptors, counters = {}, dict(reads=0, opened=[])
    def names(relative):
        prefix = relative + "/" if relative else ""
        return sorted(path[len(prefix):] for path in paths if path.startswith(prefix)
            and path != relative and "/" not in path[len(prefix):])
    def info(path):
        directory = path in directories
        mode = stat.S_IFDIR | 0o700 if directory else stat.S_IFREG | 0o600
        if path == symlink:
            mode = stat.S_IFLNK | 0o777
        return SimpleNamespace(st_dev=1, st_ino=identities[path], st_mode=mode, st_nlink=1,
            st_size=0 if directory else len(data[path]), st_mtime_ns=1, st_ctime_ns=1)
    def opened(relative):
        assert relative in directories
        counters["opened"].append(relative)
        fd = len(descriptors) + 1
        descriptors[fd] = relative
        return fd
    n.update(_open_tree_directory=opened,
        _tree_names=lambda fd, cap: (names(descriptors[fd])[:cap], names(descriptors[fd])[cap] if len(names(descriptors[fd])) > cap else None))
    fake_os = SimpleNamespace(fstat=lambda fd: info(descriptors[fd]),
        stat=lambda name, dir_fd, follow_symlinks: info((descriptors[dir_fd] + "/" if descriptors[dir_fd] else "") + name),
        close=lambda _: None)
    n["os"] = fake_os
    sandbox.ns["os"] = fake_os
    def bound(relative, cap, expected=None, **_kwargs):
        assert relative != extra
        blob = data[relative]
        assert len(blob) <= cap and (expected is None or sha(blob) == expected)
        counters["reads"] += len(blob)
        return blob
    n["owned_read"] = bound
    def single(directory, name, _factory, validator, max_bytes):
        relative = directory.relative_to(sandbox.ns["RUN"]).as_posix()
        relative = "" if relative == "." else relative
        path = (relative + "/" if relative else "") + name
        blob = bound(path, max_bytes)
        assert validator(blob)
        return dict(read_bytes=len(blob), sha256=sha(blob), bytes=len(blob))
    def inventory(directory, admitted, max_entries, max_total_bytes):
        relative = directory.relative_to(sandbox.ns["RUN"]).as_posix()
        found = names(relative)
        assert len(found) <= max_entries
        rows, used = {}, 0
        for name, spec in admitted.items():
            path = relative + "/" + name
            blob = bound(path, spec["max_bytes"], spec["sha256"])
            assert spec["bytes"] is None or len(blob) == spec["bytes"]
            used += len(blob)
            rows[name] = dict(status="REGULAR_FILE_BYTES_BOUND", sha256=sha(blob), bytes=len(blob))
        assert used <= max_total_bytes
        return dict(complete=True, read_bytes=used, initial_names=found, final_names=found,
            entries=rows, errors=[], first_unrecorded_name=None)
    n["artifact_module"] = lambda: SimpleNamespace(ensure_terminal_record_v4=single, inventory_directory_v4=inventory)
    return counters


@pytest.mark.parametrize("state", ("exact", "unknown", "symlink"))
def test_actual_named_source_tree_admission_never_reads_unknown_contents(sandbox, state):
    n = admit_profile(sandbox)
    bundle, _ = bundle_for(sandbox, n)
    files, directories = sandbox.ns["campaign_schema"](bundle, {}, {}, {})
    files = {path: spec for path, spec in files.items() if path.startswith("source/")}
    directories = {path: row for path, row in directories.items() if path == "source" or path.startswith("source/")}
    data = {path: b"invented source metadata only" for path in files}
    premise = memory_tree(sandbox, n, files, directories, dict(data))
    sandbox.ns["source_tree_shape"]()
    assert premise["reads"] == 0
    bad = "source/backend/UNDECLARED.py" if state == "unknown" else None
    symlink = next(iter(files)) if state == "symlink" else None
    counters = memory_tree(sandbox, n, files, directories, data, extra=bad, symlink=symlink)
    if state == "exact":
        sandbox.ns["source_tree_shape"]()
    else:
        reason = "DIAGNOSTIC_UNDECLARED_OR_MISSING_SOURCE_NAME" if state == "unknown" else "DIAGNOSTIC_FROZEN_SOURCE_TYPE"
        with pytest.raises(ValueError, match=reason):
            sandbox.ns["source_tree_shape"]()
    assert counters["reads"] == 0


@pytest.mark.parametrize("extra", (None, "SCORED.json"))
def test_actual_six_slot_schema_is_consumed_by_real_v6_inventory(sandbox, extra):
    n = admit_profile(sandbox)
    bundle, _ = bundle_for(sandbox, n)
    for source in n["SOURCES"]:
        bundle["source_sha256"][source] = sha(("INVENTED:" + source).encode())
    files, directories = sandbox.ns["campaign_schema"](bundle, {}, {}, {})
    assert len([path for path, row in files.items() if path.startswith("cases/") and row["required"]]) == 12
    assert len([row for row in files.values() if "leaf" in row]) == 12
    assert all(row["leaf"]["role"] == "INFERENCE" for row in files.values() if "leaf" in row)
    assert "coordinator_work" not in directories[""] and "SCORED.json" not in files
    data = {path: ("INVENTED:" + path.removeprefix("source/")).encode() if path.startswith("source/") else b"{}"
        for path, row in files.items() if row["required"]}
    present_files = {path: row for path, row in files.items() if path in data}
    counters = memory_tree(sandbox, n, present_files, directories, data, extra=extra)
    result = n["verify_campaign_inventory"](bundle, {}, {})
    assert result["complete"] is (extra is None)
    assert result["scientific_entries"] == (12 if extra is None else 13)
    if extra is not None:
        assert dict(code="UNDECLARED_CAMPAIGN_NAME", path=extra) in result["errors"]
        assert result["entries"][extra]["status"] == "UNDECLARED_METADATA_ONLY_NO_READ"
    assert result["read_bytes"] == counters["reads"]
    assert len(wire(result)) < 100 * 1024


@pytest.mark.parametrize("state", ("all_negative", "one_clean", "projection_changed", "log_count_changed"))
def test_six_slot_reconstruction_retains_negatives_and_binds_clean_result_bytes(sandbox, state):
    n = admit_profile(sandbox)
    bundle, storage, calls = terminal_fixture(sandbox, n, one_clean=state != "all_negative")
    positive = sandbox.ns["reconstruct_six"](bundle)
    assert len(positive["rows"]) == 6 and positive["no_score_policy"]["score_computation"] is False
    assert len(calls) == (state != "all_negative")
    if state in ("all_negative", "one_clean"):
        assert sum(row["reconstruction"] == "EXACT_RESULT_AND_V2_WRAPPER" for row in positive["rows"]) == (state == "one_clean")
        return
    prefix = "cases/" + sandbox.ns["SELECTED_ORDER"][0] + "/control/"
    if state == "projection_changed":
        storage[prefix + "SCORING_INPUT.json"] += b" "
        reason = "INVENTED_OWNED_HASH_MISMATCH"
    else:
        row = json.loads(storage[prefix + "COMPLETED.json"])
        row["stored_log_bytes"] = 1
        storage[prefix + "COMPLETED.json"] = wire(row) + b"\n"
        reason = "DIAGNOSTIC_CLOSED_LOG_BYTE_COUNT_BINDING"
    with pytest.raises(ValueError, match=reason):
        sandbox.ns["reconstruct_six"](bundle)


@pytest.mark.parametrize("state", ("closed", "missing_claimed", "failed_unknown_count"))
def test_closed_log_scope_is_not_capture_time_hash(sandbox, state):
    n = admit_profile(sandbox)
    data = b"invented log\n"
    cap = sandbox.ns["DIAGNOSTIC_LIMITS"]["case_log_bytes"]
    assert cap == 1024**2
    n.update(marker_present=lambda _: True, owned_read=lambda relative, actual_cap, **kwargs: data if actual_cap == cap else deny())
    row = clean_receipt(forwarded_log_bytes=len(data), stored_log_bytes=len(data))
    positive = sandbox.ns["bind_closed_log"]("cases/invented/control/OUTPUT.txt", row, cap, clean=True)
    assert positive["sha256"] == sha(data) and positive["hash_scope"] == "POST_QUIESCENCE_FILE_BYTES_NOT_SUPERVISOR_CAPTURE_TIME_HASH"
    if state == "closed":
        return
    if state == "missing_claimed":
        n["marker_present"] = lambda _: False
        with pytest.raises(ValueError, match="DIAGNOSTIC_CLAIMED_LOG_MISSING"):
            sandbox.ns["bind_closed_log"]("cases/invented/control/OUTPUT.txt", row, cap)
    else:
        result = sandbox.ns["bind_closed_log"]("cases/invented/control/OUTPUT.txt", {}, cap)
        assert result["status"] == "FAILED_LOG_UNKNOWN_OBSERVED_BYTE_COUNT"


def coordinator_fixture(sandbox, n):
    ns, events = sandbox.ns, sandbox.events
    bundle, _ = bundle_for(sandbox, n)
    audit, token = {"invented_pre": True}, "a" * 64
    cache = LexicalPath("/invented/private/coordinator")
    identity = dict(device=1, inode=2, uid=123)
    expected = dict(parent_pid=11, token_sha256=sha(token.encode()), cache=str(cache), cache_identity=identity)
    start = dict(status="STARTED_NO_RETRY", limits=ns["DIAGNOSTIC_LIMITS"],
        source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], bundle_sha256=sha(wire(bundle)),
        audit_sha256=sha(wire(audit)), arm_requests_sha256=sha(wire(n["all_arm_requests"]())))
    controls = {"COORDINATOR_DISPATCH.json": wire(expected), "STARTED.json": wire(start)}
    class Paths:
        def __new__(cls, value):
            return LexicalPath(value)
        @staticmethod
        def cwd():
            return cache
    ns.update(Path=Paths, __file__=str(ns["RUN"] / "source" / ns["RUNNER_SOURCE"]),
        verify_bundle=lambda: bundle, verify_pre=lambda _: audit,
        sys=SimpleNamespace(stdin=SimpleNamespace(buffer=SimpleNamespace(read=lambda _: wire(dict(token=token, parent_pid=11))))),
        reconstruct_six=lambda _: {"invented_evidence": True})
    ns["os"].environ = {"HOME": str(cache), "TMPDIR": str(cache)}
    publications = {}
    n.update(owned_read=lambda name, *_a, **_k: controls[name], cache_identity=lambda _: identity,
        marker_present=lambda _: False,
        exclusive_json=lambda path, value: publications.__setitem__(path.name, deepcopy(value)),
        supervise_case=lambda case, arm, *_a: events.append(("leaf", case["case_token"], arm)) or ({},
            dict(source_binding_valid=True, worker_dispatch_attempted=False, derived_result_retention_complete=True,
                private_cache_cleanup_complete=True)), coordinator_memory_check=lambda: None,
        terminalize_case_records=lambda b, reason: ns["six_slots"](b, reason, True))
    return bundle, controls, start, publications


@pytest.mark.parametrize("state", ("valid", "wrong_profile"))
def test_six_slot_coordinator_context_authenticates_new_profile_before_dispatch(sandbox, state):
    n = admit_profile(sandbox)
    bundle, controls, start, publications = coordinator_fixture(sandbox, n)
    sandbox.ns["internal_coordinator"]()  # same valid in-memory premise for both controls
    assert [event[1:] for event in sandbox.events if type(event) is tuple and event[0] == "leaf"] == [
        (case["case_token"], arm) for case in bundle["cases"] for arm in sandbox.ns["ARMS"]]
    if state == "wrong_profile":
        publications.clear()
        sandbox.events.clear()
        n["_COORDINATOR_CONTEXT"] = None  # discard only the invented first-premise state
        start["source_profile_sha256"] = "0" * 64
        controls["STARTED.json"] = wire(start)
        with pytest.raises(ValueError, match="DIAGNOSTIC_EXACT_COORDINATOR_DISPATCH_REQUIRED"):
            sandbox.ns["internal_coordinator"]()
        assert publications == {} and not any(type(event) is tuple and event[0] == "leaf" for event in sandbox.events)
    else:
        context = publications["COORDINATOR_VERIFIED.json"]
        assert context["schema"] == sandbox.ns["LEAF_SCHEMA"]
        assert context["diagnostic_source_profile_sha256"] == n["DIAGNOSTIC_PROFILE_SHA"]
        assert n["_COORDINATOR_CONTEXT"] == context
        assert publications["INFERENCE_TERMINAL.json"]["terminal_slots"] == 6


def test_unestablished_input_binding_preserves_pre_read_reason_and_stops_next_leaf(sandbox):
    n = admit_profile(sandbox)
    _, _, _, publications = coordinator_fixture(sandbox, n)
    original = dict(status="TERMINAL", source_binding_valid=False, failure_type="ValueError",
        supervisor_refusal=n["unavailable"]("TOTAL_WALL_RESERVE_UNAVAILABLE"), worker_dispatch_attempted=False)
    retained = wire(original)
    calls, case_store = [], {}
    def pre_read_refusal(case, arm, destination, _remaining):
        calls.append((case["case_token"], arm))
        case_store[str(destination / "COMPLETED.json")] = retained
        return deepcopy(original["supervisor_refusal"]), deepcopy(original)
    n.update(supervise_case=pre_read_refusal, coordinator_memory_check=deny)
    sandbox.ns["reconstruct_six"] = deny
    sandbox.ns["internal_coordinator"]()
    assert calls == [(sandbox.ns["SELECTED_ORDER"][0], "control")]
    assert list(case_store.values()) == [retained]
    failure = publications["DIAGNOSTIC_UNAVAILABLE.json"]
    assert failure["reason"] == "DIAGNOSTIC_INPUT_BINDING_NOT_ESTABLISHED_ABORT"
    assert "HASH" not in failure["reason"] and len(failure["terminal_slots"]) == 6
    assert original["supervisor_refusal"]["reasons"] == ["TOTAL_WALL_RESERVE_UNAVAILABLE"]
    assert n["_COORDINATOR_ABORTED"] is True


def outer_fixture(sandbox, n, state):
    ns, events = sandbox.ns, sandbox.events
    bundle, storage, _ = terminal_fixture(sandbox, n)
    evidence = ns["reconstruct_six"](bundle)  # actual six-slot positive premise
    events.clear()
    audit = {"invented_pre": True}
    controls = {"FROZEN.json": wire(bundle) + b"\n", "PRE_RUN_AUDIT.json": wire(audit) + b"\n"}
    declared = dict(schema=ns["SCHEMA"], terminal_slots=6, case_count=3, arm_count=2,
        source_profile_sha256=n["DIAGNOSTIC_PROFILE_SHA"], arm_requests_sha256=sha(wire(n["all_arm_requests"]())),
        evidence_sha256=sha(wire(evidence)), no_score_policy=ns["NO_SCORE_POLICY"])
    if state == "evidence_mismatch":
        declared["evidence_sha256"] = "0" * 64
    controls["INFERENCE_TERMINAL.json"] = wire(declared)
    controls["COORDINATOR_COMPLETED.json"] = wire(dict(schema=ns["SCHEMA"],
        status="DIAGNOSTIC_INFERENCE_TERMINAL_UNSCORED_PENDING_POST", terminal_slots=6, no_score_policy=ns["NO_SCORE_POLICY"]))
    controls["COORDINATOR_OUTPUT.txt"] = b""
    if state == "abort":
        controls["DIAGNOSTIC_UNAVAILABLE.json"] = b"{}"
    published = {}
    def publish(path, value):
        assert len(wire(value)) < 256 * 1024
        published[path.name] = deepcopy(value)
        controls[path.name] = wire(value) + b"\n"
        events.append(("publish", path.name))
    def source_postflight(*_):
        events.append("source-postflight")
        return {"unchanged": True}
    def reconstruct(_):
        events.append("reconstruct-six")
        return deepcopy(evidence)
    real_quiescent = n["writers_quiescent"]
    outer_receipt = clean_receipt()
    assert real_quiescent(outer_receipt, new_session=True) is True
    if state == "unknown":
        outer_receipt["observed_identity_unknown_still_present"] = True
    def quiescent(row, **kwargs):
        events.append("quiescence")
        return real_quiescent(row, **kwargs)
    def fill(b, reason):
        events.append("terminal-six")
        return ns["six_slots"](b, reason, True)
    def inventory(*_a, **_k):
        events.append("inventory")
        return dict(complete=True, observed_entries=30, scientific_entries=12)
    identity = dict(device=1, inode=2, uid=123)
    ns.update(verify_bundle=lambda: bundle, verify_pre=lambda _: audit,
        source_tree_shape=lambda: events.append("source-shape"), source_postflight=source_postflight,
        reconstruct_six=reconstruct, fresh_private_cache=lambda _: (LexicalPath("/invented/private/new-coordinator"), identity),
        secrets=SimpleNamespace(token_hex=lambda _: "c" * 64),
        resource=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=1)))
    n.update(_open_tree_directory=lambda _: 1, _tree_names=lambda *_: (["FROZEN.json", "PRE_RUN_AUDIT.json", "source"], None),
        owned_read=lambda name, *_a, **_k: controls[name], exclusive_json=publish,
        cache_identity=lambda _: identity, writers_quiescent=quiescent,
        supervise_process=lambda *_a, **_k: deepcopy(outer_receipt),
        terminalize_case_records=fill, verify_campaign_inventory=inventory,
        marker_present=lambda path: path.relative_to(ns["RUN"]).as_posix() in controls)
    return published


@pytest.mark.parametrize("state", ("clean", "unknown", "abort", "evidence_mismatch"))
def test_outer_six_slot_terminal_and_evidence_digest_follow_whole_group_quiescence(sandbox, state):
    n = admit_profile(sandbox)
    published = outer_fixture(sandbox, n, state)
    code = sandbox.ns["execute"]()
    assert code == (0 if state == "clean" else 1)
    completed = published["COMPLETED.json"]
    assert completed["schema"] == sandbox.ns["SCHEMA"] and len(completed["terminal_slot_records"]) == 6
    assert completed["no_score_policy"]["score_computation"] is False
    assert completed["coordinator_private_cache_custody"]["contents_enumerated_or_read"] is False
    if state == "unknown":
        assert not set(sandbox.events).intersection({"source-postflight", "terminal-six", "reconstruct-six", "inventory"})
        assert all(row["durable"] is False for row in completed["terminal_slot_records"])
    else:
        for operation in ("source-postflight", "terminal-six", "reconstruct-six", "inventory"):
            assert sandbox.events.index("quiescence") < sandbox.events.index(operation)
        assert all(row["durable"] is True for row in completed["terminal_slot_records"])
    if state == "clean":
        assert published["FINAL_GATE.json"]["terminal_slots"] == 6
    else:
        assert "FINAL_GATE.json" not in published
        if state == "evidence_mismatch":
            assert any(row["reason"] == "DIAGNOSTIC_COORDINATOR_SIX_SLOT_EVIDENCE_BINDING" for row in completed["failures"])
        elif state == "abort":
            assert any(row["reason"] == "DIAGNOSTIC_COORDINATOR_OR_FINALIZATION_VETO" for row in completed["failures"])
