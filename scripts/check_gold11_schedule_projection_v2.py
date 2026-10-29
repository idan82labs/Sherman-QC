"""Fixed invented-only schedule/context projection suite; exact independent PRE required.

No source/test import or execution was used to author this harness. The parent
must independently ensure V5 and other owned heavy writers are terminal before
the single approved invocation. No retry or suite/resource override is exposed.
The 30-second supervisor clock excludes parent pre/postflight and publication
I/O. Runtime evidence binds entry points/metadata, not all transitive binaries.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types


ROOT = Path(__file__).absolute().parents[1]
SITE = ROOT / ".venv/lib/python3.11/site-packages"
PYTHON = ROOT / ".venv/bin/python"
TEST = "tests/test_gold11_schedule_projection_v2.py"
PRE = "docs/research/2026-09-12_schedule_projection_v2_source_pre_run_audit.md"
PRE_MARKER = "GOLD11_SCHEDULE_PROJECTION_V2_EXACT_PRE_JSON="
WORKER_MARKER = "GOLD11_SCHEDULE_PROJECTION_V2_WORKER_JSON="
SOURCE_FILES = (
    "scripts/check_gold11_schedule_projection_v2.py",
    "scripts/check_cad_guided_connected_v12.py",
    "scripts/gold11_schedule_projection_v2.py",
    "scripts/gold11_schedule_projection_v1.py",
    "scripts/gold11_contact_scoring_v2.py",
    "scripts/gold11_mesh_scoring_v1.py",
    "scripts/evaluate_f1_against_manual_gold.py",
    "scripts/f1_physical_evaluation_contract_v2.py",
    "tests/test_gold11_schedule_projection_v2.py",
    "tests/test_gold11_schedule_projection_v1.py",
    "tests/test_gold11_contact_scoring_v2.py",
    "tests/__init__.py",
    "backend/cad_guided_two_plane_context_v1.py",
    "backend/cad_guided_two_plane_context_v2.py",
    "backend/cad_guided_two_plane_context_v3.py",
    "docs/research/2026-09-12_schedule_projection_v2_source_review.md",
    "docs/research/2026-09-12_schedule_projection_v2_source_protocol.md",
)
PINNED = {
    "scripts/check_cad_guided_connected_v12.py": "0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694",
    "scripts/gold11_schedule_projection_v2.py": "728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db",
    "scripts/gold11_schedule_projection_v1.py": "a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245",
    "scripts/gold11_contact_scoring_v2.py": "9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff",
    "scripts/gold11_mesh_scoring_v1.py": "aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234",
    "scripts/evaluate_f1_against_manual_gold.py": "1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62",
    "scripts/f1_physical_evaluation_contract_v2.py": "28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7",
    "tests/test_gold11_schedule_projection_v2.py": "61014e1500742ed40d16d7494bbeccae93e4c2ce722317e676d7123881653a76",
    "tests/test_gold11_schedule_projection_v1.py": "8d3113d8017d3e50e096696563daf38c4f62deacb87ca552263a6adcac8faf72",
    "tests/test_gold11_contact_scoring_v2.py": "f9576b9d4b7267565657ee18a4b344e150a48da9c86f9bf5da2259db3d9fe2cb",
    "tests/__init__.py": "5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b",
    "backend/cad_guided_two_plane_context_v1.py": "9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a",
    "backend/cad_guided_two_plane_context_v2.py": "d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663",
    "backend/cad_guided_two_plane_context_v3.py": "9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d",
    "docs/research/2026-09-12_schedule_projection_v2_source_review.md": "96b795eea31fa0a2bf272be134ace3676afa4610058e4295b34d9a47450b617d",
}
EXPECTED_OUTCOMES = 237
CPU_SECONDS, WORK_SECONDS, WALL_SECONDS = 20, 25, 30
RSS_BYTES, OUTPUT_BYTES = 512*1024**2, 1024**2
SOURCE_BYTES, ADMIN_BYTES = 2*1024**2, 128*1024
LIMITS = dict(worker_cpu_seconds=CPU_SECONDS, work_wall_seconds=WORK_SECONDS,
    supervision_wall_seconds=WALL_SECONDS, sampled_group_rss_bytes=RSS_BYTES,
    output_bytes=OUTPUT_BYTES)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path, cap):
    """Bounded regular final-component no-follow read on the trusted checkout.

    Stable descriptor/path checks are custody evidence, not a hostile-filesystem
    sandbox or an atomic snapshot across several independently read files.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap,
                 "INVALID_REGULAR_FILE_OR_SIZE")
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(65536, cap-size+1))
            if not block:
                break
            size += len(block)
            _require(size <= cap, "FILE_READ_BYTE_CAP")
            chunks.append(block)
        after = os.fstat(fd)
        _require(_identity(before) == _identity(after) == _identity(Path(path).lstat())
            and size == before.st_size, "FILE_CHANGED_DURING_READ")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _binding(path, cap):
    blob = _read(path, cap)
    return dict(bytes=len(blob), sha256=_sha(blob))


def _root_initializer_absent():
    try:
        (ROOT / "__init__.py").lstat()
    except FileNotFoundError:
        return
    raise ValueError("ROOT_PACKAGE_INITIALIZER_MUST_BE_ABSENT")


def _sources():
    _root_initializer_absent()
    answer = {name: _binding(ROOT/name, SOURCE_BYTES) for name in SOURCE_FILES}
    _require(len(answer) == 17, "SOURCE_CLOSURE_NOT_SEVENTEEN")
    _require(all(answer[name]["sha256"] == digest for name, digest in PINNED.items()),
             "FROZEN_PROJECTION_CLOSURE_CHANGED")
    return answer


def _source_map_sha(bindings):
    return _sha(_json({name: row["sha256"] for name, row in bindings.items()}))


def _runtime():
    _require(sys.version_info[:2] == (3, 11), "REQUIRES_APPLICATION_PYTHON_311")
    _require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
             "REQUIRES_I_S_B_FLAGS")
    executable = Path(sys.executable).resolve(strict=True)
    _require(executable == PYTHON.resolve(strict=True), "WRONG_APPLICATION_INTERPRETER")
    info = executable.stat()
    versions = {}
    for name in ("numpy", "scipy", "pytest"):
        matches = list(importlib.metadata.Distribution.discover(name=name, path=[str(SITE)]))
        _require(len(matches) == 1, "AMBIGUOUS_PACKAGE_DISTRIBUTION")
        versions[name] = matches[0].version
        _require(type(versions[name]) is str and 0 < len(versions[name]) <= 100,
                 "INVALID_DISTRIBUTION_VERSION")
    return dict(python_version=sys.version, implementation=sys.implementation.name,
        executable=str(executable), executable_stat=list(_identity(info)),
        platform=sys.platform, machine=os.uname().machine, distribution_versions=versions,
        package_initializers={name: _binding(SITE/name/"__init__.py", SOURCE_BYTES)
                              for name in ("numpy", "scipy", "pytest")},
        scope="ORDINARY_LOADER_ENTRYPOINTS_METADATA_NOT_FULL_RUNTIME_BINARY_CLOSURE")


def _pre(digest, sources, runtime):
    _require(type(digest) is str and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest), "EXACT_PRE_SHA_REQUIRED")
    blob = _read(ROOT/PRE, SOURCE_BYTES)
    _require(_sha(blob) == digest, "EXACT_PRE_BYTES_MISMATCH")
    lines = [line[len(PRE_MARKER):] for line in blob.decode("utf-8").splitlines()
             if line.startswith(PRE_MARKER)]
    _require(len(lines) == 1 and len(lines[0].encode("utf-8")) <= 4096,
             "UNIQUE_BOUNDED_EXACT_PRE_MARKER_REQUIRED")
    marker = json.loads(lines[0])
    expected = dict(schema="GOLD11_SCHEDULE_PROJECTION_V2_EXACT_PRE", decision="GREEN",
        scope="INVENTED_ONLY", expected_outcomes=EXPECTED_OUTCOMES,
        source_map_sha256=_source_map_sha(sources), runtime_binding_sha256=_sha(_json(runtime)))
    _require(type(marker) is dict and set(marker) == set(expected)
        and type(marker["expected_outcomes"]) is int and marker == expected,
        "EXACT_PRE_SCOPE_SOURCE_OR_DENOMINATOR_MISMATCH")
    return dict(path=PRE, bytes=len(blob), sha256=digest, declaration=marker)


def _write(path, value):
    blob = _json(value)
    _require(0 < len(blob) <= ADMIN_BYTES, "ADMIN_PUBLICATION_BYTE_CAP")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(blob)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)
    return dict(bytes=len(blob), sha256=_sha(blob))


def _supervisor(sources):
    name = "scripts/check_cad_guided_connected_v12.py"
    blob = _read(ROOT/name, SOURCE_BYTES)
    _require(_sha(blob) == sources[name]["sha256"], "SUPERVISOR_SOURCE_CHANGED")
    tree = ast.parse(blob, filename=str(ROOT/name))
    wanted = {"_remaining", "_group_sample", "supervise_native_context_test_v3"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in wanted]
    _require(len(functions) == 3 and {node.name for node in functions} == wanted,
             "SUPERVISOR_FUNCTION_DEFINITION_SET_CHANGED")
    namespace = dict(Path=Path, tempfile=tempfile, time=time, os=os, subprocess=subprocess,
        selectors=selectors, signal=signal, hashlib=hashlib, CPU_SECONDS=CPU_SECONDS,
        WORK_SECONDS=WORK_SECONDS, WALL_SECONDS=WALL_SECONDS, RSS_BYTES=RSS_BYTES,
        OUTPUT_BYTES=OUTPUT_BYTES)
    # Original AST bodies only; no historical imports, SOURCE_FILES, main or
    # suite selection is evaluated. The complete original file is still bound.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT/name), "exec"), namespace)
    return namespace["supervise_native_context_test_v3"]


class _Outcomes:
    def __init__(self):
        self.nodes, self.reports = [], {}
        self.deselected = self.collection_errors = 0
        self.invalid_report = False

    def pytest_collection_finish(self, session):
        self.nodes = [item.nodeid for item in session.items]
        _require(len(self.nodes) == EXPECTED_OUTCOMES and len(set(self.nodes)) == EXPECTED_OUTCOMES,
                 "EXACT_FROZEN_UNIQUE_COLLECTED_OUTCOMES_REQUIRED")
        _require(all(Path(str(item.path)).absolute() == ROOT/TEST for item in session.items),
                 "UNDECLARED_TEST_FILE_COLLECTED")

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collectreport(self, report):
        self.collection_errors += int(report.failed)

    def pytest_runtest_logreport(self, report):
        if report.nodeid not in self.nodes or report.when not in ("setup", "call", "teardown"):
            self.invalid_report = True
            return
        phases = self.reports.setdefault(report.nodeid, {})
        if report.when in phases:
            self.invalid_report = True
        # Presence is the marker: an empty xfail reason must not turn XPASS
        # into an ordinary pass or escape the strict no-xfail qualification.
        phases[report.when] = (report.outcome, hasattr(report, "wasxfail"))

    def result(self):
        result = dict(collected=len(self.nodes), passed=0, failed=0, skipped=0,
            xfailed=0, xpassed=0, errors=0, phase_triplets_passed=0,
            deselected=self.deselected, collection_errors=self.collection_errors,
            invalid_report=self.invalid_report, node_ids_sha256=_sha(_json(self.nodes)))
        for phases in self.reports.values():
            result["phase_triplets_passed"] += int(set(phases) == {"setup", "call", "teardown"}
                and all(value == ("passed", False) for value in phases.values()))
            for phase, (outcome, xfail) in phases.items():
                if xfail:
                    result["xpassed" if outcome == "passed" else "xfailed"] += 1
                elif outcome == "failed":
                    result["failed" if phase == "call" else "errors"] += 1
                elif outcome == "skipped":
                    result["skipped"] += 1
                elif phase == "call" and outcome == "passed":
                    result["passed"] += 1
        return result


def _outcomes_ok(row):
    return (type(row) is dict and all(type(row.get(key)) is int and row[key] == EXPECTED_OUTCOMES
        for key in ("collected", "passed", "phase_triplets_passed"))
        and all(type(row.get(key)) is int and row[key] == 0 for key in
            ("failed", "skipped", "xfailed", "xpassed", "errors", "deselected", "collection_errors"))
        and row.get("invalid_report") is False)


def _loaded_environment(numpy, scipy, pytest, runtime):
    for name, module in (("numpy", numpy), ("scipy", scipy), ("pytest", pytest)):
        _require(Path(module.__file__).resolve(strict=True) == (SITE/name/"__init__.py").resolve(strict=True),
                 "ACTUAL_PACKAGE_INITIALIZER_PATH_MISMATCH")
        _require(module.__version__ == runtime["distribution_versions"][name]
            and _binding(Path(module.__file__), SOURCE_BYTES) == runtime["package_initializers"][name],
            "ACTUAL_PACKAGE_ENTRYPOINT_OR_VERSION_MISMATCH")


def _worker(pre_sha):
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    request_path = Path(os.environ["GOLD11_SCHEDULE_PROJECTION_REQUEST"])
    request_blob = _read(request_path, ADMIN_BYTES)
    _require(_sha(request_blob) == os.environ["GOLD11_SCHEDULE_PROJECTION_REQUEST_SHA"], "REQUEST_BYTES_CHANGED")
    request = json.loads(request_blob)
    sources, runtime = _sources(), _runtime()
    pre = _pre(pre_sha, sources, runtime)
    _require(request["source_bindings"] == sources and request["runtime"] == runtime
        and request["pre"] == pre and request["limits"] == LIMITS
        and request["test"] == TEST and request["expected_outcomes"] == EXPECTED_OUTCOMES,
        "WORKER_PREFLIGHT_REQUEST_MISMATCH")
    # Canonicalize ONLY the new supervisor-owned cache. In particular, macOS
    # /var aliases must not be mistaken for a source/input custody relaxation.
    cache = Path(os.environ["HOME"]).resolve(strict=True)
    _require(cache.is_dir(), "OWNED_CACHE_NOT_DIRECTORY")
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME"):
        os.environ[key] = str(cache)
    sys.pycache_prefix = str(cache/"bytecode")
    sys.path.insert(0, str(SITE))  # -S: no site initialization or .pth execution.
    tests = types.ModuleType("tests")
    tests.__package__, tests.__path__ = "tests", [str(ROOT/"tests")]
    _require("tests" not in sys.modules and "backend" not in sys.modules, "UNEXPECTED_PROJECT_PRELOAD")
    sys.modules["tests"] = tests
    forbidden_initializers = {str(ROOT/"backend/__init__.py"), str(ROOT/"__init__.py")}

    def forbid_native_or_server(event, arguments):
        if event == "import" and arguments[0].split(".")[0] in {"OCP", "open3d", "backend"}:
            raise RuntimeError("UNDECLARED_NATIVE_OR_BACKEND_IMPORT")
        if event == "exec" and getattr(arguments[0], "co_filename", None) in forbidden_initializers:
            raise RuntimeError("PROJECT_SERVER_OR_ROOT_INITIALIZER_EXECUTION")

    sys.addaudithook(forbid_native_or_server)
    import numpy
    import scipy
    import pytest
    _loaded_environment(numpy, scipy, pytest, runtime)
    outcomes = _Outcomes()
    code = int(pytest.main(["-c", "/dev/null", "--noconftest", "--import-mode=importlib",
        "--rootdir", str(ROOT), "-p", "no:cacheprovider", "-q", "-s",
        "--basetemp", str(cache/"pytest-work"), str(ROOT/TEST)], plugins=[outcomes]))
    after_sources, after_runtime = _sources(), _runtime()
    _loaded_environment(numpy, scipy, pytest, after_runtime)
    _require(_pre(pre_sha, after_sources, after_runtime) == pre, "PRE_CHANGED_DURING_WORKER")
    counts = outcomes.result()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu = usage.ru_utime+usage.ru_stime
    forbidden_loaded = sorted(name for name in sys.modules
        if name.split(".")[0] in {"OCP", "open3d", "backend"})
    gate = (code == 0 and _outcomes_ok(counts) and sources == after_sources
        and runtime == after_runtime and not forbidden_loaded and cpu <= CPU_SECONDS)
    ledger = dict(schema="GOLD11_SCHEDULE_PROJECTION_V2_WORKER", outcome_gate=gate,
        pytest_exit_code=code, outcomes=counts, source_bindings_after=after_sources,
        runtime_after=after_runtime, pre_after=pre, forbidden_loaded_modules=forbidden_loaded,
        worker_cpu_seconds=cpu, worker_peak_rss_bytes=usage.ru_maxrss*(1 if sys.platform == "darwin" else 1024))
    _require(len(_json(ledger)) <= ADMIN_BYTES, "WORKER_RECEIPT_BYTE_CAP")
    print(WORKER_MARKER+_json(ledger).decode("utf-8"), flush=True)
    return 0 if gate else 1


def _inventory(directory, expected):
    _require(stat.S_ISDIR(directory.lstat().st_mode), "RETAINED_DIRECTORY_CHANGED")
    names = []
    with os.scandir(directory) as iterator:
        for item in iterator:
            _require(len(names) < 4, "RETAINED_NAME_CAP")
            names.append(item.name)
    _require(set(names) == set(expected), "EXACT_ADMIN_NAMESPACE_REQUIRED")
    actual = {name: _binding(directory/name, OUTPUT_BYTES if name == "OUTPUT.txt" else ADMIN_BYTES)
              for name in names}
    _require(actual == expected, "FINAL_PUBLICATION_IDENTITY_MISMATCH")
    return actual


def _parent(pre_sha):
    parent_started = time.monotonic()
    sources, runtime = _sources(), _runtime()
    pre = _pre(pre_sha, sources, runtime)  # No attempt or worker exists before exact PRE admission.
    supervisor = _supervisor(sources)
    base = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    _require(stat.S_ISDIR(base.lstat().st_mode), "RETAINED_PARENT_DIRECTORY_REQUIRED")
    retained = Path(tempfile.mkdtemp(prefix="gold11_schedule_projection_v2-", dir=base))
    request = dict(schema="GOLD11_SCHEDULE_PROJECTION_V2_STARTED", test=TEST, expected_outcomes=EXPECTED_OUTCOMES,
        limits=LIMITS, source_bindings=sources, source_map_sha256=_source_map_sha(sources),
        runtime=runtime, pre=pre, scope="INVENTED_BYTES_CONTEXT_METADATA_PROJECTION_ONLY",
        lifecycle_scope="ONE_WORKER_PER_APPROVED_INVOCATION_EXTERNAL_NO_RETRY_CUSTODY")
    started = _write(retained/"STARTED.json", request)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        GOLD11_SCHEDULE_PROJECTION_REQUEST=str(retained/"STARTED.json"), GOLD11_SCHEDULE_PROJECTION_REQUEST_SHA=started["sha256"])
    try:
        receipt = supervisor([str(PYTHON), "-I", "-S", "-B", "-u", str(ROOT/SOURCE_FILES[0]),
            "--worker", "--pre-sha256", pre_sha], cwd=ROOT, env=env, retained=retained)
    except Exception as error:
        receipt = dict(exit_code=None, budget_stop="SUPERVISION_RECEIPT_UNAVAILABLE",
            primary_error=type(error).__name__+":"+str(error)[:256], cleanup_error="GROUP_CUSTODY_UNKNOWN",
            process_group_confirmed_empty=False, requires_group_attention=True)
    ledger = output_binding = after_sources = after_runtime = after_pre = None
    postflight_errors = []
    if receipt.get("process_group_confirmed_empty") is True:
        # Independently attempt source/runtime custody even if the worker died
        # before its receipt, or pytest produced an ordinary failing outcome.
        try:
            after_sources = _sources()
        except Exception as error:
            postflight_errors.append("SOURCES:"+type(error).__name__+":"+str(error)[:256])
        try:
            after_runtime = _runtime()
        except Exception as error:
            postflight_errors.append("RUNTIME:"+type(error).__name__+":"+str(error)[:256])
        try:
            after_pre = _pre(pre_sha, after_sources if after_sources is not None else sources,
                after_runtime if after_runtime is not None else runtime)
        except Exception as error:
            postflight_errors.append("PRE:"+type(error).__name__+":"+str(error)[:256])
        try:
            output = _read(retained/"OUTPUT.txt", OUTPUT_BYTES)
            output_binding = dict(bytes=len(output), sha256=_sha(output))
            _require(output_binding["sha256"] == receipt["output_sha256"], "SUPERVISOR_OUTPUT_HASH_MISMATCH")
            lines = [line[len(WORKER_MARKER):] for line in output.decode("utf-8").splitlines()
                     if line.startswith(WORKER_MARKER)]
            _require(len(lines) == 1 and len(lines[0].encode("utf-8")) <= ADMIN_BYTES,
                     "EXACT_WORKER_RECEIPT_REQUIRED")
            ledger = json.loads(lines[0])
        except Exception as error:
            postflight_errors.append("OUTPUT:"+type(error).__name__+":"+str(error)[:256])
    else:
        postflight_errors.append("WRITERS_NOT_QUIESCENT_NO_OUTPUT_OR_POSTFLIGHT_READ")
    outcome_gate = (receipt.get("exit_code") == 0 and not receipt.get("budget_stop")
        and not receipt.get("primary_error") and not receipt.get("cleanup_error")
        and receipt.get("process_group_confirmed_empty") is True and not postflight_errors
        and type(ledger) is dict and ledger.get("schema") == "GOLD11_SCHEDULE_PROJECTION_V2_WORKER"
        and ledger.get("outcome_gate") is True and ledger.get("pytest_exit_code") == 0
        and _outcomes_ok(ledger.get("outcomes")) and ledger.get("forbidden_loaded_modules") == []
        and after_sources == sources == ledger.get("source_bindings_after")
        and after_runtime == runtime == ledger.get("runtime_after")
        and after_pre == pre == ledger.get("pre_after"))
    completed = dict(schema="GOLD11_SCHEDULE_PROJECTION_V2_COMPLETED",
        status="PASS_PENDING_FINAL_PUBLICATION_AND_INDEPENDENT_POST" if outcome_gate else "FAIL",
        outcome_gate=outcome_gate, supervision=receipt, worker=ledger, postflight_errors=postflight_errors,
        source_bindings_after=after_sources, runtime_after=after_runtime, pre_after=after_pre,
        parent_prepublication_wall_seconds=time.monotonic()-parent_started,
        parent_io_scope="NOT_INSIDE_30_SECOND_SUPERVISOR_CLOCK", retained_attempt_directory=str(retained))
    completed_binding = _write(retained/"COMPLETED.json", completed)
    final_inventory, publication_error = None, None
    if receipt.get("process_group_confirmed_empty") is True and output_binding is not None:
        try:
            final_inventory = _inventory(retained, {"STARTED.json": started,
                "OUTPUT.txt": output_binding, "COMPLETED.json": completed_binding})
        except Exception as error:
            publication_error = type(error).__name__+":"+str(error)[:256]
    else:
        publication_error = "FINAL_PUBLICATION_CUSTODY_UNAVAILABLE"
    final_gate = outcome_gate and publication_error is None and final_inventory is not None
    print(_json(dict(status="PASS_PENDING_INDEPENDENT_POST" if final_gate else "FAIL",
        outcome_gate=outcome_gate, final_publication_gate=publication_error is None and final_inventory is not None,
        publication_error=publication_error, final_inventory=final_inventory,
        parent_through_inventory_wall_seconds=time.monotonic()-parent_started,
        retained_attempt_directory=str(retained))).decode("utf-8"), flush=True)
    return 0 if final_gate else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pre-sha256", required=True)
    args = parser.parse_args()
    _require(len(args.pre_sha256) == 64 and all(c in "0123456789abcdef" for c in args.pre_sha256),
             "EXACT_PRE_SHA_REQUIRED")
    return _worker(args.pre_sha256) if args.worker else _parent(args.pre_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
