"""Pure mocked cache-path regression; no cache, native or original data reads.

Only named source files are read. Selected function definitions execute with
invented filesystem seams, never the full driver or either real entrypoint.
These18 controls accompany, not replace, the unchanged72 V2 helper controls.
"""

import ast
from pathlib import Path, PurePosixPath
import stat
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = "scripts/check_native_depth_retained_diagnostic_v3.py"
PREVIOUS = "scripts/check_native_depth_retained_diagnostic_v2.py"
PREFIX = "native-context-v3-tests-"
SUPPLIED = "/var/folders/invented/T/"+PREFIX+"example"
CANONICAL = "/private"+SUPPLIED


def _tree(relative=DRIVER):
    return ast.parse((ROOT/relative).read_text())


def _definitions(names, namespace):
    tree = _tree()
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), DRIVER, "exec"), namespace)
    return namespace


def _mock_cache(*, fault=None, canonical=CANONICAL):
    events = []
    info = types.SimpleNamespace(st_dev=7, st_ino=19, st_mode=stat.S_IFDIR | 0o700,
        st_size=64, st_mtime_ns=101, st_ctime_ns=102)
    class MockPath:
        def __init__(self, value): self.path = PurePosixPath(str(value))
        @property
        def name(self): return self.path.name
        def is_absolute(self): return self.path.is_absolute()
        def __str__(self): return str(self.path)
        def resolve(self, *, strict):
            events.append(("resolve", str(self), strict))
            assert strict is True
            if fault == "resolve": raise OSError("MOCK_ALIAS_RESOLUTION_FAILURE")
            return MockPath(canonical)
    def remaining(deadline):
        events.append(("deadline", deadline))
        if fault == "deadline": raise TimeoutError("MOCK_DEADLINE")
    def lstat(path):
        events.append(("lstat", str(path)))
        if fault == "leaf_symlink":
            return types.SimpleNamespace(**dict(vars(info), st_mode=stat.S_IFLNK | 0o777))
        return info
    def directory(path, deadline):
        events.append(("directory", str(path), deadline))
        if fault == "open": raise NotADirectoryError("MOCK_CANONICAL_NOFOLLOW_FAILURE")
        return 23
    def fstat(fd):
        events.append(("fstat", fd))
        return types.SimpleNamespace(**dict(vars(info), st_ino=20)) if fault == "identity" else info
    def close(fd): events.append(("close", fd))
    namespace = dict(Path=MockPath, stat=stat, _remaining=remaining, _directory=directory,
        os=types.SimpleNamespace(lstat=lstat, fstat=fstat, close=close))
    _definitions(("_identity", "_owned_cache_root"), namespace)
    return namespace["_owned_cache_root"], events


@pytest.mark.parametrize("alias", [True, False])
def test_only_owned_cache_ancestor_alias_is_canonicalized_then_nofollow_admitted(alias):
    value = SUPPLIED if alias else CANONICAL
    function, events = _mock_cache()
    path, receipt = function(value, 17.)
    assert str(path) == CANONICAL
    assert receipt == dict(supplied_path=value, canonical_path=CANONICAL,
        identity=[7, 19, stat.S_IFDIR | 0o700, 64, 101, 102],
        scope="ONLY_SUPERVISOR_OWNED_CACHE_ANCESTOR_ALIAS_CANONICALIZED")
    assert [row[0] for row in events] == ["deadline", "lstat", "deadline", "resolve", "deadline", "directory", "fstat", "close"]
    assert next(row for row in events if row[0] == "directory")[1] == CANONICAL
    assert events[-1] == ("close", 23)


@pytest.mark.parametrize("value", [None, "", "relative/"+PREFIX+"x", "/", "/var/wrong",
    "/var/../"+PREFIX+"x", "/var/./"+PREFIX+"x", "/var/"+PREFIX+"x"*4096])
def test_invalid_scalar_or_path_refuses_before_any_filesystem_seam(value):
    function, events = _mock_cache()
    with pytest.raises(ValueError): function(value, 17.)
    assert events == [("deadline", 17.)]


@pytest.mark.parametrize("fault", ["leaf_symlink", "resolve", "identity", "open"])
def test_leaf_nofollow_resolution_open_and_rebinding_failures_remain_fatal(fault):
    function, events = _mock_cache(fault=fault)
    with pytest.raises((ValueError, OSError)): function(SUPPLIED, 17.)
    names = [row[0] for row in events]
    if fault == "leaf_symlink": assert names == ["deadline", "lstat"]
    if fault == "resolve": assert "directory" not in names
    if fault == "identity": assert events[-1] == ("close", 23)
    if fault == "open": assert "fstat" not in names and "close" not in names


def test_expired_deadline_does_not_resolve_or_open_any_path():
    function, events = _mock_cache(fault="deadline")
    with pytest.raises(TimeoutError): function(SUPPLIED, 17.)
    assert events == [("deadline", 17.)]


def test_all_source_original_input_and_scientific_custody_predicates_preserved():
    old, new = _tree(PREVIOUS), _tree()
    preserved = ("_directory", "_read", "_canonical", "_strict_json", "_original_empty", "_original_inputs",
        "_original_receipts", "_install_guards", "_ledger_pass", "_publish", "_exact_attempt_names")
    for name in preserved:
        left = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
        right = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False), name
    def assignment(tree, name):
        return next(node for node in tree.body if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets))
    for name in ("ORIGINAL_INPUTS", "ORIGINAL_SOURCE_BUNDLE_SHA256", "CRITERIA", "HELPER", "TEST", "RECORD_BYTES"):
        assert ast.dump(assignment(old, name), include_attributes=False) == ast.dump(assignment(new, name), include_attributes=False)
    for names in (("CPU_SECONDS", "WORK_SECONDS", "WALL_SECONDS"), ("RSS_BYTES", "OUTPUT_BYTES")):
        def grouped(tree):
            return next(node for node in tree.body if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Tuple) and [part.id for part in target.elts] == list(names) for target in node.targets))
        assert ast.dump(grouped(old), include_attributes=False) == ast.dump(grouped(new), include_attributes=False)


def test_worker_has_exact_two_test_files_and_single_cache_only_adapter_call():
    tree = _tree()
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_worker")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_owned_cache_root"]
    assert len(calls) == 1 and calls[0] in list(ast.walk(worker))
    assert ast.unparse(calls[0].args[0]) == "os.environ['TMPDIR']"
    pytest_call = next(node for node in ast.walk(worker) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "pytest" and node.func.attr == "main")
    assert pytest_call.lineno > calls[0].lineno
    assert "*[str(ROOT / name) for name in TESTS]" in ast.unparse(pytest_call.args[0])
    assert "str(cache / 'pytest-invented')" in ast.unparse(pytest_call.args[0])
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)}
    assert ast.literal_eval(assignments["TEST"]) == "tests/test_native_depth_retained_diagnostic_v2.py"
    assert ast.literal_eval(assignments["CACHE_TEST"]) == "tests/test_native_depth_retained_cache_v3.py"
    assert ast.unparse(assignments["TESTS"]) == "(TEST, CACHE_TEST)"
    assert ast.literal_eval(assignments["ORIGINAL_PURE_OUTCOMES"]) == 72
    assert ast.literal_eval(assignments["CACHE_OUTCOMES"]) == 18
    assert len(assignments["ADDITIONAL_SOURCE_FILES"].elts) == 9


def test_false_release_guard_precedes_every_entry_access_not_a_permanent_flag_requirement():
    namespace = _definitions(("_require_frozen", "_worker", "main"), dict(RELEASE_AUTHORIZED=False))
    for name in ("_require_frozen", "_worker", "main"):
        with pytest.raises(ValueError, match="RETAINED_NATIVE_V3_NOT_FROZEN"):
            namespace[name]()
    tree = _tree()
    for name in ("main", "_worker"):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert isinstance(function.body[0], ast.Expr) and ast.unparse(function.body[0].value) == "_require_frozen()"
