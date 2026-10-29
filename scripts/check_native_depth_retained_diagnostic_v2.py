"""Frozen candidate for one retained-native integrity check; exact PRE required.

The original eight terminal files and 111-source bundle are immutable inputs.
Only new pure tests, fresh policy constructors and report validators may run.
Pins and denominator are fixed; independent PRE/archive are required to launch.
"""

import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
RELEASE_AUTHORIZED = True
EXPECTED_OUTCOMES = 72
CPU_SECONDS, WORK_SECONDS, WALL_SECONDS = 120, 175, 180
RSS_BYTES, OUTPUT_BYTES = 2*1024**3, 1024**2
SOURCE_BYTES, CONTROL_BYTES, CHECK_BYTES = 2*1024**2, 256*1024, 64*1024
RECORD_BYTES = 129*1024**2
SELF = "scripts/check_native_depth_retained_diagnostic_v2.py"
HELPER = "tests/native_depth_retained_diagnostic_v2.py"
TEST = "tests/test_native_depth_retained_diagnostic_v2.py"
PROTOCOL = "docs/research/2026-09-11_native_depth_retained_diagnostic_v2_protocol.md"
NATIVE_POST = "docs/research/2026-09-11_two_plane_depth_native_diagnostic_post_run_audit.md"
PREDECESSOR = "scripts/check_cad_guided_two_plane_depth_native_v1.py"
PREDECESSOR_SHA256 = "cb36e30d9cb7bcb39dd2afeb45d7e02421361eeb4edb554c0540442650b7ddf5"
ORIGINAL_SOURCE_BUNDLE_SHA256 = "5f07c4166b4382fdf4bfe855ee1b6ea76c39803b063b2680cd366cbbd697d022"
ADDITIONAL_SOURCE_FILES = (SELF, HELPER, TEST, PROTOCOL, NATIVE_POST)
# SELF is bound externally by exact PRE/archive and internally before/after and
# parent-to-worker; a literal hash of this entire file cannot include itself.
NEW_SOURCE_PINS = {
    HELPER: "469c08fd0626666627aaa781865082c6af22521d8872f4d3d08055f54f25300f",
    TEST: "9e3510d81e0a6b0269268dc62a30eb6d465118fcb6fbed51f81c412122ecb7bf",
    PROTOCOL: "d15319f039e47499202744282d19c72e06a941dab5b29d513c8895006ac26041",
    NATIVE_POST: "fdfbe4aa5e2dc46eb0f972e7ad1c52b99a2be4c8834774fe1c0474bddd54bc87",
}
FIXTURE = "tests/test_cad_guided_two_plane_depth_native_v1.py"
ORIGINAL_DIRECTORY = "docs/research/cad_engine_v1_synthetic_checks/two_plane_depth_v1_native-d4dw1v2p"
ORIGINAL_PGID = 89392
ORIGINAL_INPUTS = {
    "STARTED.json": (14125, "f908c624b324dcddcf0c46b181f1a55ae43faca5441f15fe7edf22e4d9d9f10e"),
    "OUTPUT.txt": (7140, "0868bfc465282e5549370682916aeb696264cbccdf5c2167f4016abaf810fbd1"),
    "ARTIFACT_INVENTORY.json": (574, "e566c9d984f93a888a45f37011bf7e5686121722db9022e1ad0136a78cc02646"),
    "COMPLETED.json": (14948, "f5a8629505ff4e4735eabb95b25321c208e06812c7bec8c071c1edad05ab7796"),
    "ARTIFACTS/CASE_START.json": (1000, "ba7d2a4e3c05ba8e47b4b65e9bf44e55bd13487fdce80e2e1ba55acc34524a55"),
    "ARTIFACTS/CAD.step": (30521, "63e7ec51b132ed4bf514e3b753f204e84abbeed84ef5025abf9799d3d2b70ad6"),
    "ARTIFACTS/SCAN.ply": (31131, "05ae6538c3b0c7262241193dc75fe4ccf8c70cf937911e6809c63bfce92efac1"),
    "ARTIFACTS/CASE_RESULT.json": (27980423, "4b6705fdfea4b8c3dd0b307c1b5e14677be6419b60a695828a55fc385b7d59ca"),
}
CRITERIA = ("publication_source_input", "raw_native_sampler", "context_origin_pose",
    "composite_schedule_profile", "raw_support_cover", "windows_nominals",
    "quantity_carrier_projection", "branch_denominator")
SCOPE = "RETAINED_SERIALIZED_NATIVE_REPORT_INTEGRITY_NOT_SAME_WORKER_ACCURACY_OR_PHYSICAL_F1"


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_frozen():
    # First entry operation: no environment flag or CLI value bypasses this.
    if RELEASE_AUTHORIZED is not True:
        raise ValueError("RETAINED_NATIVE_V2_NOT_FROZEN")
    if type(EXPECTED_OUTCOMES) is not int or not 1 <= EXPECTED_OUTCOMES <= 512:
        raise ValueError("EXACT_NEW_PURE_TEST_DENOMINATOR_REQUIRED")
    if set(NEW_SOURCE_PINS) != {HELPER, TEST, PROTOCOL, NATIVE_POST} or not all(
            _hex(value) for value in NEW_SOURCE_PINS.values()):
        raise ValueError("FINAL_RETAINED_HELPER_TEST_PROTOCOL_POST_PINS_REQUIRED")
    if sys.platform != "darwin":
        raise ValueError("REVIEWED_DARWIN_RUSAGE_BYTE_UNITS_REQUIRED")
    values = (CPU_SECONDS, WORK_SECONDS, WALL_SECONDS, RSS_BYTES, OUTPUT_BYTES, CHECK_BYTES)
    if any(type(value) is not int for value in values) or values != (120, 175, 180, 2*1024**3, 1024**2, 64*1024):
        raise ValueError("RETAINED_NATIVE_V2_FROZEN_LIMITS_CHANGED")


def _remaining(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("RETAINED_NATIVE_V2_ABSOLUTE_DEADLINE")


def _no_initializer():
    if os.path.lexists(ROOT/"__init__.py"):
        raise ValueError("UNDECLARED_ENCLOSING_REPOSITORY_INITIALIZER")


def _identity(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _directory(path, deadline):
    path = Path(path)
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("FIXED_ABSOLUTE_PATH_REQUIRED")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            _remaining(deadline)
            following = os.open(part, flags, dir_fd=fd)
            old, fd = fd, following
            os.close(old)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(path, cap, deadline, *, keep=True, allow_empty=False):
    """Stable bounded regular-file read; all directory components no-follow.

    These are quiescent identity checks, not an atomic adversarial snapshot or
    a hard timeout on an uninterruptible operating-system call.
    """
    path = Path(path)
    directory, fd = _directory(path.parent, deadline), None
    try:
        directory_identity = _identity(os.fstat(directory))[:2]
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not int(not allow_empty) <= before.st_size <= cap:
            raise ValueError("REGULAR_FILE_TYPE_OR_SIZE_REQUIRED")
        size, digest, chunks = 0, hashlib.sha256(), []
        while True:
            _remaining(deadline)
            block = os.read(fd, min(65536, cap-size+1))
            if not block:
                break
            size += len(block)
            if size > cap:
                raise ValueError("FILE_BYTE_BUDGET")
            digest.update(block)
            if keep:
                chunks.append(block)
        after = os.fstat(fd)
        following = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(following) != _identity(after) or size != before.st_size:
            raise ValueError("FILE_CHANGED_DURING_READ")
        reopened = _directory(path.parent, deadline)
        try:
            if _identity(os.fstat(reopened))[:2] != directory_identity:
                raise ValueError("FILE_PARENT_REBOUND_DURING_READ")
        finally:
            os.close(reopened)
        return dict(bytes=size, sha256=digest.hexdigest(), identity=_identity(after)), b"".join(chunks) if keep else None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _canonical(value, cap):
    digest, size = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False,
                                     ensure_ascii=True).iterencode(value):
        block = fragment.encode("utf-8")
        size += len(block)
        if size > cap:
            raise ValueError("CANONICAL_ENCODING_BYTE_BUDGET")
        digest.update(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def _strict_json(blob):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("DUPLICATE_JSON_KEY")
            value[key] = item
        return value
    def reject(value):
        raise ValueError("NONFINITE_JSON_CONSTANT")
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("NONFINITE_JSON_NUMBER")
        return result
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=reject, parse_float=number)


def _definitions(name, relative, blob):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT/relative)
    sys.modules[name] = module
    try:
        exec(compile(blob, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _supervisor(deadline):
    binding, blob = _read(ROOT/PREDECESSOR, SOURCE_BYTES, deadline)
    if binding["sha256"] != PREDECESSOR_SHA256:
        raise ValueError("EXACT_NATIVE_SUPERVISOR_SOURCE_REQUIRED")
    module = _definitions("_retained_native_v2_supervisor_definitions", PREDECESSOR, blob)
    supervisor = module._supervisor()  # Source declarations only; never its main.
    original = tuple(supervisor.SOURCE_FILES)
    if len(original) != 111 or len(set(original)) != 111 or set(original) & set(ADDITIONAL_SOURCE_FILES):
        raise ValueError("EXACT_ORIGINAL_111_SOURCE_DECLARATION_REQUIRED")
    supervisor.SOURCE_FILES = [*original, *ADDITIONAL_SOURCE_FILES]
    supervisor.WALL_SECONDS, supervisor.WORK_SECONDS, supervisor.CPU_SECONDS = WALL_SECONDS, WORK_SECONDS, CPU_SECONDS
    supervisor.RSS_BYTES, supervisor.OUTPUT_BYTES = RSS_BYTES, OUTPUT_BYTES
    return supervisor, original


def _sources(supervisor, original, deadline, *, keep=False):
    names = tuple(supervisor.SOURCE_FILES)
    if len(names) != 116 or len(set(names)) != 116 or names != (*original, *ADDITIONAL_SOURCE_FILES):
        raise ValueError("EXACT_116_SOURCE_CLOSURE_REQUIRED")
    hashes, identities, buffers = {}, {}, {}
    for name in names:
        binding, blob = _read(ROOT/name, SOURCE_BYTES, deadline, keep=keep)
        hashes[name], identities[name] = binding["sha256"], binding
        if keep:
            buffers[name] = blob
    base = {name: hashes[name] for name in original}
    if _canonical(base, CONTROL_BYTES)["sha256"] != ORIGINAL_SOURCE_BUNDLE_SHA256:
        raise ValueError("ORIGINAL_111_SOURCE_BUNDLE_DRIFT")
    if any(hashes[name] != expected for name, expected in NEW_SOURCE_PINS.items()):
        raise ValueError("NEW_RETAINED_SOURCE_PIN_DRIFT")
    if hashes[PREDECESSOR] != PREDECESSOR_SHA256:
        raise ValueError("NATIVE_PREDECESSOR_PIN_DRIFT")
    return dict(hashes=hashes, files=identities, mapping=_canonical(hashes, CONTROL_BYTES)), buffers


def _original_empty(supervisor, deadline):
    if supervisor._group_sample(ORIGINAL_PGID, deadline):
        raise ValueError("ORIGINAL_TERMINAL_GROUP_NOT_EMPTY_NO_INPUT_READ")


def _original_inputs(supervisor, deadline, *, keep=False):
    _original_empty(supervisor, deadline)
    result, buffers = {}, {}
    for name, (size, sha256) in ORIGINAL_INPUTS.items():
        binding, blob = _read(ROOT/ORIGINAL_DIRECTORY/name, size, deadline, keep=keep)
        if (binding["bytes"], binding["sha256"]) != (size, sha256):
            raise ValueError("ORIGINAL_TERMINAL_INPUT_PIN_DRIFT:"+name)
        result[name] = binding
        if keep:
            buffers[name] = blob
    _original_empty(supervisor, deadline)
    return dict(files=result, mapping=_canonical(result, CONTROL_BYTES)), buffers


def _original_receipts(buffers, original_hashes):
    start = _strict_json(buffers["STARTED.json"])
    completed = _strict_json(buffers["COMPLETED.json"])
    inventory = _strict_json(buffers["ARTIFACT_INVENTORY.json"])
    if (start.get("source_sha256_before") != original_hashes
            or completed.get("source_sha256_after") != original_hashes
            or completed.get("source_before_after_consistent") is not True
            or type(completed.get("exit_code")) is not int or completed["exit_code"] != 1
            or completed.get("process_group_confirmed_empty") is not True
            or completed.get("requires_group_attention") is not False
            or any(completed.get(key) is not None for key in ("budget_stop", "primary_error", "cleanup_error"))
            or completed.get("artifact_inventory_complete") is not True
            or completed.get("repository_initializer_absent_after") is not True
            or completed.get("output_sha256") != ORIGINAL_INPUTS["OUTPUT.txt"][1]
            or inventory.get("complete") is not True or inventory.get("errors") != []):
        raise ValueError("EXACT_ORIGINAL_RED_TERMINAL_CUSTODY_REQUIRED")
    expected = {name.split("/", 1)[1]: dict(bytes=size, sha256=sha256)
                for name, (size, sha256) in ORIGINAL_INPUTS.items() if name.startswith("ARTIFACTS/")}
    if inventory.get("entries") != expected:
        raise ValueError("ORIGINAL_FOUR_SCIENCE_ARTIFACT_BINDINGS_REQUIRED")


class _ForbiddenExecution(BaseException):
    """Do not force retained continuation after a native/inference boundary hit."""


def _veto(state, reason):
    if len(state["violations"]) < 8:
        state["violations"].append(reason)
    raise _ForbiddenExecution(reason)


class _BoundRepositoryImports:
    """Execute the source-preflight buffers; never cached repository bytecode."""
    def __init__(self, buffers, guard):
        self.buffers, self.guard = buffers, guard

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "OCP" or fullname.startswith("OCP.") or fullname == "OCC" or fullname.startswith("OCC."):
            _veto(self.guard, "NATIVE_IMPORT_FORBIDDEN")
        if fullname == "backend" or fullname.startswith("backend.") or fullname == "tests" or fullname.startswith("tests."):
            relative = fullname.replace(".", "/")+".py"
            if fullname in ("backend", "tests"):
                relative = fullname+"/__init__.py"
            if relative not in self.buffers:
                _veto(self.guard, "UNDECLARED_REPOSITORY_IMPORT:"+fullname)
            return importlib.util.spec_from_loader(fullname, self, origin=str(ROOT/relative),
                                                  is_package=fullname in ("backend", "tests"))
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__.replace(".", "/")+".py"
        if module.__name__ in ("backend", "tests"):
            relative = module.__name__+"/__init__.py"
            module.__path__ = [str(ROOT/module.__name__)]
        module.__file__ = str(ROOT/relative)
        exec(compile(self.buffers[relative], module.__file__, "exec"), module.__dict__)


def _install_guards():
    state = dict(file_free=False, violations=[])
    def audit(event, args):
        if event == "import" and args and (args[0] == "OCP" or args[0].startswith("OCP.")
                                           or args[0] == "OCC" or args[0].startswith("OCC.")):
            _veto(state, "NATIVE_IMPORT_FORBIDDEN")
        if state["file_free"] and (event == "open" or event.startswith("subprocess.")
                or event.startswith("socket.") or event in ("os.listdir", "os.scandir", "os.system")):
            _veto(state, "RETAINED_HELPER_FILE_OR_PROCESS_ACCESS_FORBIDDEN")
    fixture_paths = {str(ROOT/name) for name in (FIXTURE,
        "tests/test_cad_guided_connected_v12_native_bytes.py", "tests/test_cad_guided_mesh_connected_native_v1.py",
        "tests/test_cad_guided_connected_v12.py")}
    forbidden_fixture = {"_make_fixture", "_execute_workflow", "_evaluate", "curved_packet", "_scan_mesh",
        "_cad_arrays", "_asymmetric_native_strip", "_owned_step_bytes", "_ply_bytes", "run_single_native_byte_case"}
    backend_prefix = str(ROOT/"backend")+"/"
    def profile(frame, event, arg):
        if event != "call":
            return
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        if filename in fixture_paths and name in forbidden_fixture:
            _veto(state, "FIXTURE_OR_ORACLE_EXECUTION_FORBIDDEN:"+name)
        if filename.startswith(backend_prefix) and name != "_fit_policy" and (name.startswith(("inspect_cad_guided", "initialize_",
                "register_", "fit_", "_fit_", "sample_", "tessellate_")) or "_refit" in name):
            _veto(state, "ENGINE_INFERENCE_OR_REFIT_FORBIDDEN:"+name)
    sys.addaudithook(audit)
    sys.setprofile(profile)
    state["profile"] = profile
    return state


class _Outcomes:
    def __init__(self):
        self.collected = self.passed = self.failed = self.skipped = self.errors = self.deselected = 0
        self.xfailed = self.xpassed = 0
        self.interruption_seen = False

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collectreport(self, report):
        self.errors += int(report.failed)

    def pytest_runtest_makereport(self, item, call):
        # pytest can absorb a BaseException into a failing test report. Preserve
        # that safety boundary rather than treating it as ordinary test failure.
        if call.excinfo is not None and not isinstance(call.excinfo.value, Exception):
            self.interruption_seen = True

    def pytest_runtest_logreport(self, report):
        if getattr(report, "wasxfail", None) is not None:
            self.xfailed += int(report.skipped)
            self.xpassed += int(report.when == "call" and report.passed)
        if report.skipped:
            self.skipped += 1
        elif report.when == "call":
            self.passed += int(report.passed)
            self.failed += int(report.failed)
        elif report.failed:
            self.errors += 1


def _pure_pass(outcomes, code):
    return (type(code) is int and code == 0 and type(outcomes) is dict
        and set(outcomes) == {"collected", "passed", "failed", "skipped", "errors", "deselected", "xfailed", "xpassed"}
        and all(type(value) is int for value in outcomes.values())
        and outcomes["collected"] == outcomes["passed"] == EXPECTED_OUTCOMES
        and all(outcomes[key] == 0 for key in ("failed", "skipped", "errors", "deselected", "xfailed", "xpassed")))


def _ledger_pass(ledger):
    # A limited early refusal is not complete retained native-route integrity.
    if type(ledger) is not dict or ledger.get("schema") != "NATIVE_DEPTH_RETAINED_DIAGNOSTIC_LEDGER_V2" or ledger.get("status") not in (
            "COMPLETE_NO_MEASUREMENTS_INTEGRITY", "COMPLETE_CONDITIONAL_INTEGRITY"):
        return False
    rows = ledger.get("criteria")
    return (type(rows) is list and len(rows) == 8 and all(type(row) is dict for row in rows)
        and tuple(row.get("name") for row in rows) == CRITERIA
        and all(row.get("status") == "PASS" for row in rows)
        and ledger.get("driver_attestation_checked") is True
        and ledger.get("immutable_decoded_evidence_and_supplied_arrays") is True
        and type(ledger.get("canonical_record_before")) is dict
        and ledger["canonical_record_before"] == ledger.get("canonical_record_after")
        and ledger.get("original_accuracy_status") == ledger.get("original_diagnostic_status") == "FAIL")


def _publish(path, value, cap=CONTROL_BYTES):
    chunks, size = [], 1  # Include the final newline before accepting fragments.
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        block = fragment.encode()
        size += len(block)
        if size > cap:
            raise ValueError("PUBLICATION_BYTE_BUDGET")
        chunks.append(block)
    blob = b"".join(chunks)+b"\n"
    with path.open("xb") as stream:
        if stream.write(blob) != len(blob):
            raise OSError("PUBLICATION_SHORT_WRITE")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def _worker():
    _require_frozen()
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    started = time.monotonic()
    deadline = started+WORK_SECONDS
    _no_initializer()
    supervisor, original = _supervisor(deadline)
    source, buffers = _sources(supervisor, original, deadline, keep=True)
    inputs, original_buffers = _original_inputs(supervisor, deadline, keep=True)
    if (source["mapping"]["sha256"] != os.environ.get("SHERMAN_RETAINED_V2_SOURCE_SHA256")
            or inputs["mapping"]["sha256"] != os.environ.get("SHERMAN_RETAINED_V2_INPUT_SHA256")):
        raise ValueError("PARENT_TO_WORKER_EXACT_SOURCE_INPUT_IDENTITY_REQUIRED")
    _original_receipts(original_buffers, {name: source["hashes"][name] for name in original})
    retained = Path(os.environ["SHERMAN_RETAINED_V2_ATTEMPT"])
    base = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    if not retained.is_absolute() or retained.parent != base or not retained.name.startswith("native_depth_retained_v2-"):
        raise ValueError("PARENT_OWNED_RETAINED_ATTEMPT_REQUIRED")
    fd = _directory(retained, deadline)
    os.close(fd)
    cache = Path(os.environ["TMPDIR"])
    fd = _directory(cache, deadline)
    os.close(fd)
    sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
    # Neither enclosing repository nor backend initializer is executed.
    package = types.ModuleType("backend")
    package.__path__ = [str(ROOT/"backend")]
    sys.modules["backend"] = package
    guard = _install_guards()
    finder = _BoundRepositoryImports(buffers, guard)
    sys.meta_path.insert(0, finder)
    import pytest
    outcomes = _Outcomes()
    code = int(pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider", "--assert=plain",
        "-q", "-s", "--basetemp", str(cache/"pytest-invented"), str(ROOT/TEST)], plugins=[outcomes]))
    pure = {key: value for key, value in vars(outcomes).items() if key != "interruption_seen"}
    if code not in (0, 1) or outcomes.interruption_seen or guard["violations"] or sys.getprofile() is not guard["profile"]:
        raise _ForbiddenExecution("PURE_STAGE_INTERRUPTION_OR_GUARD_VIOLATION_NO_RETAINED_CONTINUATION")
    # An ordinary control failure does not hide the retained diagnostic. Source,
    # input and safety failures below, or BaseException above, prohibit it.
    _no_initializer()
    repeated, _ = _sources(supervisor, original, deadline)
    repeated_inputs, _ = _original_inputs(supervisor, deadline)
    if repeated != source or repeated_inputs != inputs:
        raise ValueError("PRE_RETAINED_SOURCE_OR_INPUT_IDENTITY_CHANGED")
    fixture = _definitions("_retained_native_v2_original_fixture", FIXTURE, buffers[FIXTURE])
    helper = importlib.import_module("tests.native_depth_retained_diagnostic_v2")
    backends = helper._v1.BACKENDS
    if (type(backends) is not tuple or len(backends) != 20 or len(backends) != len(set(backends))
            or any(type(name) is not str or "backend/"+name+".py" not in source["hashes"] for name in backends)):
        raise ValueError("HELPER_PRELOAD_DEPENDENCIES_OUTSIDE_FROZEN_SOURCE_CLOSURE")
    for name in backends:
        importlib.import_module("backend."+name)
    fixture._policy_objects()  # Fresh constructors/cache setup only, never fixture geometry.
    if any(name == "OCP" or name.startswith("OCP.") for name in sys.modules):
        raise _ForbiddenExecution("NATIVE_MODULE_PRESENT_BEFORE_RETAINED_CHECK")
    cad_blob, scan_blob = original_buffers["ARTIFACTS/CAD.step"], original_buffers["ARTIFACTS/SCAN.ply"]
    raw, triangles = helper.decode_retained_ply_v2(scan_blob)
    import numpy as np
    input_binding = dict(cad_sha256=hashlib.sha256(cad_blob).hexdigest(), scan_sha256=hashlib.sha256(scan_blob).hexdigest(),
        raw_xyz_sha256=hashlib.sha256(np.asarray(raw, dtype="<f8").tobytes()).hexdigest(),
        triangle_raw_rows_sha256=hashlib.sha256(np.asarray(triangles, dtype="<i8").tobytes()).hexdigest(),
        raw_rows=len(raw), triangle_rows=len(triangles))
    if input_binding != dict(cad_sha256=ORIGINAL_INPUTS["ARTIFACTS/CAD.step"][1],
            scan_sha256=ORIGINAL_INPUTS["ARTIFACTS/SCAN.ply"][1], raw_rows=650, triangle_rows=1176,
            raw_xyz_sha256="994b3d8a4de940c6470393fdb60dc20d9fb368434c135bd33cbc496872cfce3c",
            triangle_raw_rows_sha256="8c4f5c76ca41d6fd9d63a7523a4cace886392e307fe07b56140dd8b510c12de6"):
        raise ValueError("INDEPENDENT_NATIVE_POST_PLY_BINDINGS_REQUIRED")
    record = _strict_json(original_buffers["ARTIFACTS/CASE_RESULT.json"])
    canonical_before = _canonical(record, RECORD_BYTES)
    artifact = dict(filename="CASE_RESULT.json", bytes=ORIGINAL_INPUTS["ARTIFACTS/CASE_RESULT.json"][0],
                    sha256=ORIGINAL_INPUTS["ARTIFACTS/CASE_RESULT.json"][1])
    retained_binding = dict(schema="NATIVE_DEPTH_RETAINED_BINDING_V2", original_result_artifact=artifact,
        canonical_record_before=canonical_before, input_binding=input_binding,
        source_bundle_sha256=ORIGINAL_SOURCE_BUNDLE_SHA256,
        original_terminal_output_sha256=ORIGINAL_INPUTS["OUTPUT.txt"][1],
        driver_custody_status="EXACT_TERMINAL_FILES_BOUND_BEFORE_RETAINED_CHECK_PENDING_POSTFLIGHT",
        original_accuracy_status="FAIL", original_diagnostic_status="FAIL")
    ledger, error = None, None
    helper_cpu, helper_wall = time.process_time(), time.monotonic()
    try:
        guard["file_free"] = True
        ledger = helper.evaluate_retained_diagnostic_v2(fixture, record, artifact, cad_blob, scan_blob,
                                                       raw, triangles, retained_binding)
    except Exception as caught:
        error = type(caught).__name__+":"+str(caught)[:256]
    finally:
        guard["file_free"] = False
    helper_cpu, helper_wall = time.process_time()-helper_cpu, time.monotonic()-helper_wall
    canonical_after = _canonical(record, RECORD_BYTES)
    _no_initializer()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    native = any(name == "OCP" or name.startswith("OCP.") for name in sys.modules)
    retained_pass = (error is None and _ledger_pass(ledger) and canonical_after == canonical_before
                     and helper_cpu <= 90 and helper_wall <= 120 and not native and not guard["violations"]
                     and sys.getprofile() is guard["profile"])
    check = dict(schema="NATIVE_DEPTH_RETAINED_CHECK_V2", scope=SCOPE,
        status="PASS_PENDING_PARENT_POSTFLIGHT" if _pure_pass(pure, code) and retained_pass else "FAIL",
        source_mapping=source["mapping"], input_mapping=inputs["mapping"],
        original_source_bundle_sha256=ORIGINAL_SOURCE_BUNDLE_SHA256, retained_binding=retained_binding,
        original_accuracy_status="FAIL", original_diagnostic_status="FAIL",
        pure_test_outcomes=pure, pytest_exit_code=code, expected_pure_test_outcomes=EXPECTED_OUTCOMES,
        retained_check_denominator=1, retained_check_attempted=1, retained_check_passed=retained_pass,
        helper_ledger=ledger, helper_error=error, helper_cpu_seconds=helper_cpu, helper_wall_seconds=helper_wall,
        canonical_record_after=canonical_after, decoded_record_before_after_consistent=canonical_after == canonical_before,
        worker_cpu_seconds=usage.ru_utime+usage.ru_stime, worker_peak_rss_bytes=usage.ru_maxrss,
        native_ocp_imported=native, repository_initializer_absent=True,
        manufacturing_accuracy_qualified=False, production_use_allowed=False)
    published = _publish(retained/"CHECK_RESULT.json", check, CHECK_BYTES)
    print(json.dumps(dict(kind="RETAINED_NATIVE_V2_WORKER_RECEIPT", check_result=published,
        status=check["status"], pure_test_outcomes=pure, pytest_exit_code=code,
        retained_check_passed=retained_pass, worker_cpu_seconds=check["worker_cpu_seconds"],
        worker_peak_rss_bytes=usage.ru_maxrss), sort_keys=True, allow_nan=False), flush=True)
    return 0 if check["status"] == "PASS_PENDING_PARENT_POSTFLIGHT" else 1


def _error(error):
    return type(error).__name__+":"+str(error)[:256]


def _exact_attempt_names(retained, deadline):
    """Inspect only this new owned directory, never discover original inputs."""
    expected = {"STARTED.json", "OUTPUT.txt", "INPUT_INVENTORY.json", "COMPLETED.json", "CHECK_RESULT.json"}
    directory = _directory(retained, deadline)
    try:
        names = []
        with os.scandir(directory) as entries:
            for entry in entries:
                _remaining(deadline)
                names.append(entry.name)
                if len(names) > 5:
                    raise ValueError("UNDECLARED_SIXTH_RETAINED_ATTEMPT_FILE")
        if len(names) != 5 or set(names) != expected:
            raise ValueError("EXACT_FIVE_RETAINED_ATTEMPT_NAMES_REQUIRED")
    finally:
        os.close(directory)


def main():
    _require_frozen()  # Before CLI dispatch, source/data/import access or attempt creation.
    if sys.argv[1:] == ["--worker"]:
        return _worker()
    if sys.argv[1:]:
        raise ValueError("NO_INPUT_TEST_POLICY_OR_BUDGET_OVERRIDES")
    started = time.monotonic()
    deadline = started+WALL_SECONDS
    _no_initializer()
    supervisor, original = _supervisor(deadline)
    source_before, _ = _sources(supervisor, original, deadline)
    inputs_before, _ = _original_inputs(supervisor, deadline)
    base = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    fd = _directory(base, deadline)
    os.close(fd)
    retained = Path(tempfile.mkdtemp(prefix="native_depth_retained_v2-", dir=base))
    limits = dict(worker_cpu_seconds=CPU_SECONDS, work_wall_seconds=WORK_SECONDS, parent_wall_seconds=WALL_SECONDS,
        sampled_group_rss_bytes=RSS_BYTES, output_bytes=OUTPUT_BYTES, check_result_bytes=CHECK_BYTES,
        helper_cpu_seconds=90, helper_wall_seconds=120)
    start_binding = _publish(retained/"STARTED.json", dict(schema="NATIVE_DEPTH_RETAINED_START_V2",
        status="STARTED_ONE_RETAINED_CHECK_NO_RETRY", scope=SCOPE, source_sha256_before=source_before["hashes"],
        source_mapping=source_before["mapping"], original_input_files_before=inputs_before["files"],
        original_input_mapping=inputs_before["mapping"], repository_initializer_absent_before=True,
        expected_pure_test_outcomes=EXPECTED_OUTCOMES, retained_check_denominator=1,
        source_file_count=116, original_source_file_count=111, limits=limits,
        original_accuracy_status="FAIL", original_diagnostic_status="FAIL"))
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", SHERMAN_RETAINED_V2_SOURCE_SHA256=source_before["mapping"]["sha256"],
        SHERMAN_RETAINED_V2_INPUT_SHA256=inputs_before["mapping"]["sha256"], SHERMAN_RETAINED_V2_ATTEMPT=str(retained))
    print(json.dumps(dict(status="RUNNING_NOT_EVIDENCE_COMPLETE", attempt=str(retained))), flush=True)
    # Debit preflight elapsed time from the unchanged inherited relative-clock
    # supervisor. These are remaining allowances, never increased frozen caps.
    elapsed = time.monotonic()-started
    supervisor.WALL_SECONDS, supervisor.WORK_SECONDS = WALL_SECONDS-elapsed, WORK_SECONDS-elapsed
    if supervisor.WORK_SECONDS <= 0:
        raise TimeoutError("PREFLIGHT_EXHAUSTED_WORK_ALLOWANCE")
    try:
        receipt = supervisor.supervise_native_context_test_v3(
            [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u", str(ROOT/SELF), "--worker"],
            cwd=ROOT, env=env, retained=retained)
    except Exception as caught:
        receipt = dict(exit_code=None, budget_stop="SUPERVISION_RECEIPT_UNAVAILABLE", primary_error=_error(caught),
            cleanup_error="GROUP_CUSTODY_UNAVAILABLE", process_group_confirmed_empty=False, requires_group_attention=True)
    receipt.update(scope=SCOPE, limits=limits, retained_attempt_directory=str(retained),
        original_accuracy_status="FAIL", original_diagnostic_status="FAIL")
    after_source = after_inputs = check = None
    postflight_errors = []
    inventory = dict(schema="NATIVE_DEPTH_RETAINED_INPUT_INVENTORY_V2", complete=False,
        before=inputs_before, after=None, before_after_consistent=False, errors=[])
    if receipt.get("process_group_confirmed_empty") is True:
        for label in ("inputs", "sources", "check_and_output"):
            try:
                if label == "inputs":
                    after_inputs, _ = _original_inputs(supervisor, deadline)
                    inventory.update(after=after_inputs, before_after_consistent=after_inputs == inputs_before,
                        complete=after_inputs == inputs_before)
                    if after_inputs != inputs_before:
                        raise ValueError("ORIGINAL_INPUT_BEFORE_AFTER_IDENTITY_CHANGED")
                elif label == "sources":
                    _no_initializer()
                    after_source, _ = _sources(supervisor, original, deadline)
                    if after_source != source_before:
                        raise ValueError("SOURCE_BEFORE_AFTER_IDENTITY_CHANGED")
                else:
                    check_binding, check_blob = _read(retained/"CHECK_RESULT.json", CHECK_BYTES, deadline)
                    check = _strict_json(check_blob)
                    output_binding, output_blob = _read(retained/"OUTPUT.txt", OUTPUT_BYTES, deadline, allow_empty=True)
                    rows = []
                    for line in output_blob.splitlines():
                        if line.startswith(b"{"):
                            try:
                                row = _strict_json(line)
                            except (ValueError, UnicodeError):
                                continue
                            if type(row) is dict and row.get("kind") == "RETAINED_NATIVE_V2_WORKER_RECEIPT":
                                rows.append(row)
                    if len(rows) != 1 or rows[0].get("check_result") != {key: check_binding[key] for key in ("bytes", "sha256")}:
                        raise ValueError("EXACT_ONE_WORKER_RESULT_PUBLICATION_BINDING_REQUIRED")
                    if (output_binding["sha256"] != receipt.get("output_sha256")
                            or output_binding["bytes"] != receipt.get("output_bytes_observed")):
                        raise ValueError("COMPLETE_OUTPUT_RETENTION_REQUIRED")
                    receipt.update(check_result_artifact={key: check_binding[key] for key in ("bytes", "sha256")},
                        worker_receipt=rows[0], output_retained_bytes=output_binding["bytes"],
                        output_artifact={key: output_binding[key] for key in ("bytes", "sha256")})
            except Exception as caught:
                postflight_errors.append(label+":"+_error(caught))
    else:
        postflight_errors.append("WRITER_QUIESCENCE_UNCONFIRMED_NO_POSTFLIGHT_READS")
    inventory["errors"] = list(postflight_errors)
    inventory["complete"] = inventory["complete"] and not postflight_errors
    inventory_binding = _publish(retained/"INPUT_INVENTORY.json", inventory)
    receipt.update(source_sha256_after=after_source["hashes"] if after_source else None,
        source_before_after_consistent=after_source == source_before, input_inventory_complete=inventory["complete"],
        input_inventory_artifact=inventory_binding, postflight_errors=postflight_errors,
        repository_initializer_absent_after=not os.path.lexists(ROOT/"__init__.py"),
        parent_wall_seconds=time.monotonic()-started, parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    check_ok = (type(check) is dict and check.get("schema") == "NATIVE_DEPTH_RETAINED_CHECK_V2"
        and check.get("status") == "PASS_PENDING_PARENT_POSTFLIGHT"
        and check.get("source_mapping") == source_before["mapping"] and check.get("input_mapping") == inputs_before["mapping"]
        and _pure_pass(check.get("pure_test_outcomes"), check.get("pytest_exit_code"))
        and type(check.get("retained_check_denominator")) is int and check["retained_check_denominator"] == 1
        and type(check.get("retained_check_attempted")) is int and check["retained_check_attempted"] == 1
        and check.get("retained_check_passed") is True and _ledger_pass(check.get("helper_ledger"))
        and check.get("decoded_record_before_after_consistent") is True
        and check.get("native_ocp_imported") is False and check.get("repository_initializer_absent") is True
        and check.get("original_accuracy_status") == check.get("original_diagnostic_status") == "FAIL"
        and check.get("manufacturing_accuracy_qualified") is False and check.get("production_use_allowed") is False
        and type(check.get("worker_cpu_seconds")) in (int, float) and math.isfinite(check["worker_cpu_seconds"])
        and 0 <= check["worker_cpu_seconds"] <= CPU_SECONDS
        and type(check.get("worker_peak_rss_bytes")) is int and 0 < check["worker_peak_rss_bytes"] <= RSS_BYTES)
    passed = (check_ok and receipt.get("exit_code") == 0 and not receipt.get("budget_stop")
        and not receipt.get("primary_error") and not receipt.get("cleanup_error")
        and receipt.get("process_group_confirmed_empty") is True and not postflight_errors
        and receipt["source_before_after_consistent"] and inventory["complete"]
        and receipt["repository_initializer_absent_after"] and receipt["parent_wall_seconds"] < WALL_SECONDS)
    receipt["status"] = "PASS_PENDING_INDEPENDENT_POST" if passed else "TERMINAL_RETAINED_CHECK_FAILURE"
    completed_binding = _publish(retained/"COMPLETED.json", receipt)
    final_error = None
    try:
        if receipt.get("process_group_confirmed_empty") is not True:
            raise ValueError("NO_FINAL_PUBLICATION_READS_BEFORE_GROUP_EMPTY")
        _exact_attempt_names(retained, deadline)
        expected = {"STARTED.json": start_binding, "INPUT_INVENTORY.json": inventory_binding,
                    "COMPLETED.json": completed_binding, "CHECK_RESULT.json": receipt.get("check_result_artifact"),
                    "OUTPUT.txt": receipt.get("output_artifact")}
        for name, binding in expected.items():
            cap = OUTPUT_BYTES if name == "OUTPUT.txt" else CHECK_BYTES if name == "CHECK_RESULT.json" else CONTROL_BYTES
            actual, _ = _read(retained/name, cap, deadline, keep=False, allow_empty=name == "OUTPUT.txt")
            if {key: actual[key] for key in ("bytes", "sha256")} != binding:
                raise ValueError("FINAL_PUBLICATION_BINDING_CHANGED:"+name)
        _exact_attempt_names(retained, deadline)
    except Exception as caught:
        final_error = _error(caught)
    final_within_wall = time.monotonic()-started < WALL_SECONDS
    print(json.dumps(dict(status=receipt["status"], attempt=str(retained), outcome_gate=passed,
        final_publication_error=final_error, final_publication_within_wall=final_within_wall), sort_keys=True), flush=True)
    return 0 if passed and not final_error and final_within_wall else 1


if __name__ == "__main__":
    raise SystemExit(main())
