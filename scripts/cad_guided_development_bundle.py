"""Freeze/verify one named source-only R0 control, never run physical inputs.

The JSON contains actual local source bytes and exact dependency metadata.
Dependency RECORD hashes bind metadata, not every installed binary or OS loader.
This is a development source/config snapshot, NOT full runtime attestation,
scientific authorization, an executable historical incumbent or a score claim.
"""

import argparse
import ast
import base64
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / ".venv/lib/python3.11/site-packages"
BUNDLE = ROOT / "docs/research/baselines/cad_datum_registered_r0_20260909.json"
SOURCES = tuple("backend/" + name + ".py" for name in (
    "cad_guided_offline", "cad_guided_case_input", "cad_guided_registration",
    "cad_guided_native_features", "cad_guided_measurement", "cad_guided_instances",
    "cad_guided_evaluation", "cad_guided_paired_evaluation", "cad_import",
    "cad_guided_development_control",
)) + ("scripts/cad_guided_development_bundle.py",)
DISTRIBUTIONS = ("numpy", "scipy", "cadquery-ocp", "cadquery-ocp-proxy", "vtk")
MAX_SOURCE_BYTES = 2 * 1024**2
MAX_BUNDLE_BYTES = 8 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_regular(path, limit):
    """Bound each explicit regular file; never follow a final symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("UNSUPPORTED_SNAPSHOT_FILE")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            blob = stream.read(limit + 1)
        after = os.fstat(fd)
        fields = ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(blob) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError("SNAPSHOT_FILE_CHANGED")
        return blob
    finally:
        os.close(fd)


def _distribution(name):
    normalized = name.lower().replace("-", "_")
    matches = [item for item in importlib.metadata.distributions(path=[str(SITE)])
               if item.metadata["Name"].lower().replace("-", "_") == normalized]
    if len(matches) != 1:
        raise ValueError("DEPENDENCY_IDENTITY_NOT_UNIQUE: " + name)
    item = matches[0]
    files = {}
    for suffix in ("METADATA", "WHEEL", "RECORD"):
        paths = [path for path in item.files or () if path.name == suffix and ".dist-info/" in str(path)]
        if len(paths) != 1:
            raise ValueError("DEPENDENCY_METADATA_MISSING: " + name)
        files[suffix] = digest(read_regular(item.locate_file(paths[0]), 8 * 1024**2))
    return {"name": item.metadata["Name"], "version": item.version, "metadata_sha256": files}


def runtime_metadata():
    # Explicit module search without executing package initializers or .pth files.
    origins = {}
    for name in ("numpy", "scipy", "OCP", "OCC"):
        spec = importlib.machinery.PathFinder.find_spec(name, [str(SITE)])
        origins[name] = None if spec is None else str(Path(spec.origin).relative_to(SITE))
    if origins["OCC"] is not None or any(origins[name] is None for name in ("numpy", "scipy", "OCP")):
        raise ValueError("REQUIRED_OCP_ONLY_IMPORT_ENVIRONMENT_UNAVAILABLE")
    return {
        "python_version": platform.python_version(), "implementation": platform.python_implementation(),
        "architecture": platform.machine(), "platform": platform.system(),
        "interpreter_sha256": digest(read_regular(Path(sys.executable).resolve(), 128 * 1024**2)),
        "import_origins_relative_to_site": origins,
        "distributions": [_distribution(name) for name in DISTRIBUTIONS],
        "assurance": "DEPENDENCY_METADATA_ONLY_NOT_INSTALLED_BINARY_ATTESTATION",
        "unbound": ["OS_AND_STDLIB_BYTES", "NATIVE_DYNAMIC_LIBRARY_RESOLUTION",
                    "INSTALLED_FILES_VS_RECORD_CONTENT", "OCCT_RESOURCES_AND_TBB_THREAD_COUNT"],
    }


def specification_from_source(blob):
    # Never execute code to inspect configuration: even decorators, defaults,
    # comprehensions or expression statements in a substituted source are inert.
    tree = ast.parse(blob)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "CONTROL_ID":
            continue
        if (isinstance(node, ast.FunctionDef) and node.name in ("control_specification", "construct_policies")
                and not node.decorator_list and not node.args.defaults and not node.args.kw_defaults):
            continue
        raise ValueError("CONTROL_MODULE_HAS_STARTUP_IMPORT_OR_ACTION")
    identifiers = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "CONTROL_ID"]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "control_specification"]
    if len(identifiers) != 1 or len(functions) != 1:
        raise ValueError("CONTROL_LITERAL_SPECIFICATION_REQUIRED")
    control_id = ast.literal_eval(identifiers[0].value)
    if type(control_id) is not str:
        raise ValueError("CONTROL_LITERAL_ID_REQUIRED")
    function = functions[0]
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if (function.decorator_list or function.args.args or function.args.posonlyargs
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg
            or len(body) != 1 or not isinstance(body[0], ast.Return)):
        raise ValueError("CONTROL_LITERAL_SPECIFICATION_REQUIRED")

    class ResolveControlId(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id != "CONTROL_ID":
                raise ValueError("CONTROL_NONLITERAL_SPECIFICATION")
            return ast.Constant(value=control_id)

    result = ast.literal_eval(ResolveControlId().visit(body[0].value))
    if type(result) is not dict or result.get("control_id") != control_id:
        raise ValueError("CONTROL_LITERAL_SPECIFICATION_REQUIRED")
    return result


def capture_bundle():
    records = {}
    for name in SOURCES:
        blob = read_regular(ROOT / name, MAX_SOURCE_BYTES)
        ast.parse(blob, filename=name)
        records[name] = {"sha256": digest(blob), "bytes": len(blob),
                         "source_base64": base64.b64encode(blob).decode("ascii")}
    payload = {
        "schema": "CAD_DEVELOPMENT_SOURCE_BUNDLE_V1", "sources": records,
        "specification": specification_from_source(base64.b64decode(
            records["backend/cad_guided_development_control.py"]["source_base64"])),
        "runtime_metadata": runtime_metadata(),
        "physical_access_authorized": False, "production_use_allowed": False,
    }
    # Detect persistent edits during capture. Captured bytes, not this check,
    # are what a later frozen importer executes.
    if any(digest(read_regular(ROOT / name, MAX_SOURCE_BYTES)) != row["sha256"]
           for name, row in records.items()):
        raise ValueError("SOURCE_CHANGED_DURING_FREEZE")
    return {"bundle_sha256": digest(canonical(payload)), "payload": payload}


def validate_bundle(bundle):
    if len(canonical(bundle)) > MAX_BUNDLE_BYTES:
        raise ValueError("BUNDLE_BYTE_BUDGET")
    if type(bundle) is not dict or set(bundle) != {"bundle_sha256", "payload"}:
        raise ValueError("INVALID_BUNDLE_SCHEMA")
    payload = bundle["payload"]
    if (type(payload) is not dict or set(payload) != {"schema", "sources", "specification",
            "runtime_metadata", "physical_access_authorized", "production_use_allowed"}
            or payload["schema"] != "CAD_DEVELOPMENT_SOURCE_BUNDLE_V1"
            or payload["physical_access_authorized"] is not False
            or payload["production_use_allowed"] is not False
            or digest(canonical(payload)) != bundle["bundle_sha256"]):
        raise ValueError("INVALID_BUNDLE_BINDING")
    if type(payload["sources"]) is not dict or set(payload["sources"]) != set(SOURCES):
        raise ValueError("INVALID_SOURCE_CLOSURE")
    captured = {}
    for name, row in payload["sources"].items():
        if type(row) is not dict or set(row) != {"sha256", "bytes", "source_base64"}:
            raise ValueError("INVALID_SOURCE_RECORD")
        if type(row["source_base64"]) is not str or len(row["source_base64"]) > 4 * MAX_SOURCE_BYTES // 3 + 4:
            raise ValueError("SOURCE_BYTE_BUDGET")
        try:
            blob = base64.b64decode(row["source_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("INVALID_SOURCE_ENCODING") from exc
        if (type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_SOURCE_BYTES
                or len(blob) != row["bytes"] or digest(blob) != row["sha256"]):
            raise ValueError("INVALID_SOURCE_DIGEST")
        ast.parse(blob, filename=name)
        captured[name] = blob
    if specification_from_source(captured["backend/cad_guided_development_control.py"]) != payload["specification"]:
        raise ValueError("CONFIGURATION_SOURCE_MISMATCH")
    return captured


def install_captured_importer(bundle):
    """Use only already-validated captured backend bytes in a fresh process.

    This executes trusted project code, not a sandbox for arbitrary bundle input.
    Self hashes prove consistency, not third-party authenticity or authority.
    """
    captured = validate_bundle(bundle)
    bundle_digest = bundle["bundle_sha256"]
    if any(name == "backend" or name.startswith("backend.") for name in sys.modules):
        raise ValueError("FROZEN_IMPORT_REQUIRES_FRESH_BACKEND_NAMESPACE")
    sources = {"backend." + Path(name).stem: blob for name, blob in captured.items()
               if name.startswith("backend/")}

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__file__ = "<frozen:" + bundle_digest + ":" + module.__name__ + ">"
            exec(compile(sources[module.__name__], module.__file__, "exec", dont_inherit=True), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "backend":
                return importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            if fullname.startswith("backend."):
                if fullname not in sources:
                    raise ModuleNotFoundError("UNFROZEN_BACKEND_IMPORT_REFUSED", name=fullname)
                return importlib.util.spec_from_loader(fullname, Loader())
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    return finder


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_BUNDLE_KEY")
        result[key] = value
    return result


def read_bundle():
    blob = read_regular(BUNDLE, MAX_BUNDLE_BYTES)
    result = json.loads(blob, object_pairs_hook=_unique_object)
    validate_bundle(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "verify"))
    args = parser.parse_args()
    if args.mode == "freeze":
        bundle = capture_bundle()
        encoded = canonical(bundle) + b"\n"
        if len(encoded) > MAX_BUNDLE_BYTES:
            raise ValueError("BUNDLE_BYTE_BUDGET")
        validate_bundle(bundle)
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite a control, even after a failed experiment.
        with BUNDLE.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        bundle = read_bundle()
        if bundle["payload"]["runtime_metadata"] != runtime_metadata():
            raise ValueError("RUNTIME_METADATA_CHANGED")
    captured = validate_bundle(bundle)
    drift = [name for name, blob in captured.items()
             if digest(read_regular(ROOT / name, MAX_SOURCE_BYTES)) != digest(blob)]
    print(json.dumps({"mode": args.mode, "bundle_sha256": bundle["bundle_sha256"],
                      "source_count": len(captured), "working_tree_drift": drift,
                      "assurance": "SOURCE_CONFIG_SNAPSHOT_RUNTIME_METADATA_ONLY",
                      "physical_score": None, "production_use_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
