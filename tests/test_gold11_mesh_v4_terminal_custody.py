"""Prospective focused V4 custody controls; source-authored, not executed.

Only exact AST-extracted runner custody functions and the qualified stdlib-only
artifact helper are evaluated by this future suite. All files are tiny invented
pytest-owned temporary files. No runner import/CLI, source freeze, process,
native/backend module, physical input, Gold or cached prediction is accessed.

The schema and neutral-dispatch validator are explicit small-tree seams, NOT
qualification of the full real11-case runner or its V11 candidate. Real helper
and filesystem behavior remain connected for inventory/publication tests.
Signal delivery is an explicit synchronous fake; no timer/process is started.
Prospective first gate:60s wall/45s workerCPU/1GiB/1MiB, root-controlled only.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"scripts/run_gold11_mesh_development_v4.py"
HELPER = ROOT/"scripts/gold11_terminal_artifacts_v4.py"
PREDECESSOR = ROOT/"scripts/run_gold11_mesh_development_v3.py"
HELPER_SHA = "3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838"
PREDECESSOR_SHA = "77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7"
SELECTED_FUNCTIONS = {
    "digest", "encoded", "decode", "unique_object", "is_hex", "artifact_directory_walk",
    "artifact_alarm_guard", "publish_bytes", "terminalize_case_records", "_tree_fingerprint",
    "_open_tree_directory", "_tree_names", "verify_campaign_inventory",
}
SELECTED_CLASSES = {"_ArtifactDeadlineExpired", "_CampaignInventoryStop"}


class FakeSignal:
    """No actual signal/timer changes; alarm delivery is a caller-injected seam."""
    ITIMER_REAL, SIGALRM = 0, 14

    def __init__(self):
        self.remaining = 100.
        self.original = lambda *_: None
        self.handler = self.original
        self.handler_changes = []

    def getitimer(self, _):
        return self.remaining, 0.

    def getsignal(self, _):
        return self.handler

    def signal(self, _, handler):
        old, self.handler = self.handler, handler
        self.handler_changes.append(handler)
        return old

    def expire(self):
        self.remaining = 0.
        self.handler(self.SIGALRM, None)


class OSProxy:
    def __init__(self, **overrides):
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides.get(name, getattr(os, name))


@contextmanager
def declared_deadline(_seconds):
    # Only a seam for callers whose deadline wrapper would install a real timer.
    # FakeSignal's already-active timer is left untouched throughout the tests.
    yield


def small_schema(file_specs, directory_names=()):
    files, directories = deepcopy(file_specs), {"": {}}
    for relative in [*directory_names, *files]:
        parts = Path(relative).parts
        for i, name in enumerate(parts):
            parent = "/".join(parts[:i])
            is_file = i == len(parts)-1 and relative in files
            directories.setdefault(parent, {})[name] = "file" if is_file else "directory"
            if not is_file:
                directories.setdefault("/".join(parts[:i+1]), {})
    return files, directories


def file_spec(blob, category="SCIENTIFIC", *, authenticate=True):
    return dict(max_bytes=max(1, len(blob)), bytes=len(blob), required=True, category=category,
                sha256=hashlib.sha256(blob).hexdigest() if authenticate else None)


@pytest.fixture
def custody(tmp_path):
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in SELECTED_FUNCTIONS
        or isinstance(node, ast.ClassDef) and node.name in SELECTED_CLASSES]
    assert {node.name for node in selected} == SELECTED_FUNCTIONS | SELECTED_CLASSES
    helper_blob = HELPER.read_bytes()
    assert hashlib.sha256(helper_blob).hexdigest() == HELPER_SHA
    helper = ModuleType("_invented_v4_artifact_helper")
    exec(compile(helper_blob, str(HELPER), "exec"), helper.__dict__)
    signal = FakeSignal()
    state = dict(schema=small_schema({}), opened=[], helper_calls=[], neutral_calls=0)

    def neutral(*_):
        state["neutral_calls"] += 1

    namespace = dict(os=os, stat=stat, Path=Path, hashlib=hashlib, json=json,
        contextmanager=contextmanager, signal=signal, time=SimpleNamespace(monotonic=lambda: 0.),
        RUN=tmp_path.resolve(), _ARTIFACT_DIRECTORY_WALK=False,
        _campaign_schema=lambda *_args, **_kwargs: deepcopy(state["schema"]),
        _verify_tree_leaf_bindings=neutral, artifact_module=lambda: helper,
        deadline=declared_deadline, publication_cap=lambda _: 1024,
        LIMITS=dict(parent_read_seconds=20, compact_case_bytes=1024))
    namespace["unavailable"] = lambda reason, phase="SUPERVISOR": dict(status="UNAVAILABLE", reasons=[reason], phase=phase)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    real_open = namespace["_open_tree_directory"]

    def opened(relative):
        state["opened"].append(relative)
        return real_open(relative)

    namespace["_open_tree_directory"] = opened
    real_inventory = helper.inventory_directory_v4

    def inventory(directory, *args, **kwargs):
        state["helper_calls"].append(Path(directory).relative_to(namespace["RUN"]).as_posix())
        return real_inventory(directory, *args, **kwargs)

    helper.inventory_directory_v4 = inventory
    return SimpleNamespace(ns=namespace, state=state, helper=helper, signal=signal, root=namespace["RUN"])


def inspect(custody, **kwargs):
    result = custody.ns["verify_campaign_inventory"]({}, {}, {}, **kwargs)
    json.dumps(result, allow_nan=False)
    return result


def write(custody, relative, blob=b"x"):
    path = custody.root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def test_complete_small_mixed_tree_is_real_helper_bound_without_new_files(custody):
    payloads = {"ROOT_CONTROL.json": b"{}", "source/minimal.py": b"# invented\n",
                "cases/invented/RESULT.json": b"{\"status\":\"UNAVAILABLE\"}"}
    custody.state["schema"] = small_schema({name: file_spec(blob,
        "CONTROL" if name.startswith("ROOT") else "SOURCE" if name.startswith("source/") else "SCIENTIFIC")
        for name, blob in payloads.items()}, ("empty",))
    (custody.root/"empty").mkdir()
    for name, blob in payloads.items():
        write(custody, name, blob)
    before = sorted(path.relative_to(custody.root).as_posix() for path in custody.root.rglob("*"))
    result = inspect(custody)
    assert result["complete"] is True and result["errors"] == []
    assert result["scientific_entries"] == 1
    assert result["schema_directories_not_attempted"] == []
    assert result["first_unrecorded_path"] is None and result["read_bytes_exact"] is True
    assert result["observed_entries"] == len(before) == len(result["entries"])
    assert result["read_bytes"] == sum(map(len, payloads.values()))
    for name, blob in payloads.items():
        assert result["entries"][name]["sha256"] == hashlib.sha256(blob).hexdigest()
    assert before == sorted(path.relative_to(custody.root).as_posix() for path in custody.root.rglob("*"))


@pytest.mark.parametrize("counts,stop_branch", [((130, 2), "alpha"), ((70, 70, 1), "beta")])
def test_overflow_is_one_global_witness_and_stops_later_branches(custody, counts, stop_branch):
    branches = ("alpha", "beta", "gamma")[:len(counts)]
    custody.state["schema"] = small_schema({}, branches)
    for branch, count in zip(branches, counts):
        for i in range(count):
            write(custody, f"{branch}/unlisted-{i:03d}")
    result = inspect(custody)
    assert result["complete"] is False and result["phase"] == "STOPPED_GLOBAL_METADATA_BUDGET"
    assert result["observed_entries"] == result["metadata_entry_limit"] == len(branches)+128
    assert len(result["entries"]) == result["observed_entries"]
    assert result["scientific_entries"] == 128
    witness = result["first_unrecorded_path"]
    assert witness.startswith(stop_branch+"/") and witness not in result["entries"]
    assert sum(row["code"] == "CAMPAIGN_METADATA_ENTRY_LIMIT" for row in result["errors"]) == 1
    stopped_at = branches.index(stop_branch)
    assert result["schema_directories_not_attempted"] == list(branches[stopped_at+1:])
    assert not set(branches[stopped_at+1:]).intersection(custody.state["opened"])
    assert stop_branch not in custody.state["helper_calls"]
    assert custody.state["neutral_calls"] == 0
    assert result["directories"][stop_branch]["name_enumeration_complete"] is False
    assert not any(row["code"] == "REQUIRED_CAMPAIGN_NAME_MISSING" for row in result["errors"])


def test_final_pass_replacement_charges_union_not_equal_per_pass_length(custody):
    custody.state["schema"] = small_schema({}, ("alpha", "omega"))
    (custody.root/"omega").mkdir()
    for i in range(128):
        write(custody, f"alpha/old-{i:03d}")
    real_names, calls = custody.ns["_tree_names"], {}
    alpha_inode = (custody.root/"alpha").stat().st_ino

    def namespace_change(fd, maximum):
        names, overflow = real_names(fd, maximum)
        if os.fstat(fd).st_ino == alpha_inode:
            calls["alpha"] = calls.get("alpha", 0)+1
            if calls["alpha"] == 2:
                assert len(names) == 128 and overflow is None
                return names[1:]+["late-new-name"], None
        return names, overflow

    # Explicit final-enumerator change seam; not actual concurrent writers.
    custody.ns["_tree_names"] = namespace_change
    result = inspect(custody)
    assert result["complete"] is False and result["phase"] == "STOPPED_GLOBAL_METADATA_BUDGET"
    assert result["observed_entries"] == 130
    assert result["first_unrecorded_path"] == "alpha/late-new-name"
    assert result["schema_directories_not_attempted"] == ["omega"]
    assert "omega" not in custody.state["opened"]


def test_late_name_below_global_limit_is_retained_and_vetoes_completion(custody):
    custody.state["schema"] = small_schema({"alpha/file": file_spec(b"x")}, ("alpha",))
    write(custody, "alpha/file")
    real_names, calls = custody.ns["_tree_names"], []
    alpha_inode = (custody.root/"alpha").stat().st_ino

    def late(fd, maximum):
        names, overflow = real_names(fd, maximum)
        if os.fstat(fd).st_ino == alpha_inode:
            calls.append(1)
            if len(calls) == 2:
                return names+["late"], None
        return names, overflow

    custody.ns["_tree_names"] = late
    result = inspect(custody)
    assert not result["complete"] and "alpha/late" in result["entries"]
    assert result["observed_entries"] == 3 and result["scientific_entries"] == 2
    assert result["first_unrecorded_path"] is None
    assert result["entries"]["alpha/late"]["status"] == "NAME_OBSERVED_NOT_YET_VALIDATED"
    assert any(row["code"] == "CAMPAIGN_DIRECTORY_NAMES_CHANGED" for row in result["errors"])


@pytest.mark.parametrize("category,count,complete", [("CONTROL", 130, True), ("SCIENTIFIC", 128, True),
                                                   ("SCIENTIFIC", 129, False)])
def test_scientific128_is_separate_from_bounded_control_namespace(custody, category, count, complete):
    files = {f"f-{i:03d}": file_spec(b"x", category) for i in range(count)}
    custody.state["schema"] = small_schema(files)
    for name in files:
        write(custody, name)
    result = inspect(custody)
    assert result["complete"] is complete and result["observed_entries"] == count
    assert result["scientific_entries"] == (count if category == "SCIENTIFIC" else 0)
    assert result["first_unrecorded_path"] is None
    if not complete:
        assert any(row["code"] == "SCIENTIFIC_MANIFEST_ENTRY_BUDGET" for row in result["errors"])


def test_root_unknown_name_cannot_report_locally_complete(custody):
    custody.state["schema"] = small_schema({}, ("empty",))
    (custody.root/"empty").mkdir()
    write(custody, ".same-byte-extra", b"{}")
    result = inspect(custody)
    assert not result["complete"] and not result["directories"][""]["complete"]
    assert result["entries"][".same-byte-extra"]["status"] == "UNDECLARED_METADATA_ONLY_NO_READ"
    assert result["scientific_entries"] == 1 and result["read_bytes"] == 0


@pytest.mark.parametrize("exception", [None, ValueError("ordinary")])
def test_alarm_guard_restores_handler_without_resetting_timer(custody, exception):
    before = custody.signal.getitimer(0)
    try:
        with custody.ns["artifact_alarm_guard"]():
            assert custody.signal.handler is not custody.signal.original
            if exception is not None:
                raise exception
    except ValueError as caught:
        assert caught is exception
    assert custody.signal.handler is custody.signal.original
    assert custody.signal.getitimer(0) == before
    assert len(custody.signal.handler_changes) == 2


def test_already_expired_alarm_guard_and_inventory_do_no_io(custody):
    custody.signal.remaining = 0.
    touched = []
    with pytest.raises(TimeoutError, match="NOT_ACTIVE"):
        with custody.ns["artifact_alarm_guard"]():
            touched.append(True)
    result = inspect(custody)
    assert touched == [] and custody.signal.handler_changes == []
    assert custody.state["opened"] == custody.state["helper_calls"] == []
    assert not result["complete"] and result["deadline_expired"] is True
    assert result["read_bytes_exact"] is False and result["read_bytes"] == 0


def test_alarm_during_handler_installation_still_restores_and_converts(custody):
    install = custody.signal.signal
    touched = []

    def immediate_alarm(number, handler):
        prior = install(number, handler)
        if handler is not custody.signal.original:
            custody.signal.expire()
        return prior

    custody.signal.signal = immediate_alarm
    with pytest.raises(TimeoutError, match="EXPIRED"):
        with custody.ns["artifact_alarm_guard"]():
            touched.append(True)
    assert touched == [] and custody.signal.handler is custody.signal.original
    assert len(custody.signal.handler_changes) == 2


@pytest.mark.parametrize("interrupt", ["deadline", "keyboard"])
def test_real_helper_closes_descriptors_before_alarm_conversion_or_base_propagation(custody, interrupt):
    opened, closed = [], []

    def tracked_open(*args, **kwargs):
        fd = os.open(*args, **kwargs)
        opened.append(fd)
        return fd

    def tracked_close(fd):
        closed.append(fd)
        return os.close(fd)

    def interrupted_fsync(_):
        if interrupt == "deadline":
            custody.signal.expire()
        raise KeyboardInterrupt("invented control")

    custody.helper.os = OSProxy(open=tracked_open, close=tracked_close, fsync=interrupted_fsync)
    expected = TimeoutError if interrupt == "deadline" else KeyboardInterrupt
    with pytest.raises(expected):
        with custody.ns["artifact_alarm_guard"]():
            custody.helper.publish_bytes_exclusive_v4(custody.root, "terminal.json", b"{}", max_bytes=16)
    assert custody.signal.handler is custody.signal.original
    assert opened and len(opened) == len(closed)
    for fd in set(opened):
        with pytest.raises(OSError):
            os.fstat(fd)
    retained = list(custody.root.iterdir())
    assert len(retained) == 1 and retained[0].name.startswith(".terminal.json.partial-")
    assert retained[0].read_bytes() == b"{}" and not (custody.root/"terminal.json").exists()


@pytest.mark.parametrize("seam", ["open", "stat", "helper"])
def test_inventory_timeout_never_continues_to_next_branch_or_final_audit(custody, seam):
    custody.state["schema"] = small_schema({"alpha/file": file_spec(b"x")}, ("alpha", "beta"))
    write(custody, "alpha/file")
    (custody.root/"beta").mkdir()
    if seam == "open":
        real_open = custody.ns["_open_tree_directory"]

        def fail_open(relative):
            if relative == "alpha":
                custody.state["opened"].append(relative)
                raise TimeoutError("invented open deadline")
            return real_open(relative)

        custody.ns["_open_tree_directory"] = fail_open
    elif seam == "stat":
        def fail_stat(name, *args, **kwargs):
            if name == "file":
                raise TimeoutError("invented stat deadline")
            return os.stat(name, *args, **kwargs)

        custody.ns["os"] = OSProxy(stat=fail_stat)
    else:
        def swallowed_ordinary_failure(*_args, **_kwargs):
            custody.state["helper_calls"].append("alpha")
            try:
                custody.signal.expire()
            except Exception:
                pytest.fail("Artifact alarm was incorrectly swallowed as an ordinary helper failure")

        custody.helper.inventory_directory_v4 = swallowed_ordinary_failure
    result = inspect(custody)
    assert not result["complete"] and result["phase"] == "INTERRUPTED_BY_ABSOLUTE_DEADLINE"
    assert result["deadline_expired"] is True and result["read_bytes_exact"] is False
    assert result["schema_directories_not_attempted"] == ["beta"]
    assert "beta" not in custody.state["opened"] and custody.state["neutral_calls"] == 0
    assert custody.signal.handler is custody.signal.original
    assert result["read_bytes"] == 0


def test_terminal_fill_timeout_retains_all22_dispositions_without_later_io(custody):
    calls = []

    def interrupted(*args, **_kwargs):
        calls.append(args[1])
        custody.signal.expire()

    custody.helper.ensure_terminal_record_v4 = interrupted
    bundle = dict(cases=[dict(case_token=f"invented-{i}") for i in range(11)], arms=["control", "challenger"])
    rows = custody.ns["terminalize_case_records"](bundle, "INVENTED_FAILURE")
    assert len(rows) == 22 and all(row["durable"] is False for row in rows)
    assert calls == ["SCORING_INPUT.json"]
    assert rows[0]["disposition"] == "INTERRUPTED_BY_TERMINAL_FILL_DEADLINE"
    assert all(row["disposition"] == "NOT_ATTEMPTED_AFTER_TERMINAL_FILL_DEADLINE" for row in rows[1:])
    assert sorted(path.relative_to(custody.root).as_posix() for path in custody.root.rglob("*")) == [
        "cases", "cases/invented-0", "cases/invented-0/control"]
    assert custody.signal.handler is custody.signal.original


@pytest.mark.parametrize("replace_inode", [False, True])
def test_later_publication_phase_rejects_changed_prior_file_or_identity(custody, replace_inode):
    custody.state["schema"] = small_schema({"bound": file_spec(b"abc", "CONTROL", authenticate=False)})
    path = write(custody, "bound", b"abc")
    prior = inspect(custody)
    assert prior["complete"] is True
    if replace_inode:
        replacement = write(custody, "temporary", b"abc")
        replacement.replace(path)
    else:
        path.write_bytes(b"xyz")
    final = inspect(custody, previous=prior)
    assert not final["complete"]
    assert any(row["code"] == "PRIOR_INVENTORY_ENTRY_CHANGED_AFTER_PUBLICATION" for row in final["errors"])


def test_frozen_scientific_constants_and_three_publication_phases_are_preserved():
    old_blob = PREDECESSOR.read_bytes()
    assert hashlib.sha256(old_blob).hexdigest() == PREDECESSOR_SHA
    old, new = ast.parse(old_blob), ast.parse(SOURCE.read_bytes())

    def value(tree, name):
        nodes = [node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        assert len(nodes) == 1
        return ast.dump(nodes[0], include_attributes=False)

    for name in ("CONFIG", "LIMITS", "CERTIFICATE_CAPS", "REVIEW_SHA", "CATALOG_SHA", "SELECTED_SHA"):
        assert value(new, name) == value(old, name)
    execute = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    phases = []
    for node in ast.walk(execute):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "verify_campaign_inventory":
            phases.append((node.lineno, "inventory"))
        elif node.func.id == "exclusive_json" and node.args:
            literals = [part.value for part in ast.walk(node.args[0]) if isinstance(part, ast.Constant)]
            if literals in (["COMPLETED.json"], ["FINAL_GATE.json"]):
                phases.append((node.lineno, literals[0]))
    assert [phase for _, phase in sorted(phases)] == [
        "inventory", "COMPLETED.json", "inventory", "FINAL_GATE.json", "inventory"]
    assert "UNQUALIFIED" in SOURCE.read_text().upper()
    assert SELECTED_CLASSES == {"_ArtifactDeadlineExpired", "_CampaignInventoryStop"}
