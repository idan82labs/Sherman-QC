"""Static freeze checks only; these do not run readers or native simulations."""
import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "cad_native_file_roundtrip_checks.py"


def source_tree():
    return ast.parse(SOURCE.read_text())


def assignment(tree, name):
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def test_frozen_thirteen_case_reader_matrix():
    matrix = ast.literal_eval(assignment(source_tree(), "ROUNDTRIP_MATRIX"))
    expected = tuple((kind, format_, "solid_preserved")
                     for kind in ("box", "quarter_cylinder", "located_compound")
                     for format_ in ("step", "brep"))
    expected += (("box", "iges", "surface_mode_solid_ownership_lost"),)
    expected += tuple((kind, format_, "import_failure")
                      for kind in ("malformed", "absent") for format_ in ("step", "brep", "iges"))
    assert matrix == expected and len(matrix) == 13
    assert ast.literal_eval(assignment(source_tree(), "GEOMETRY_ABSOLUTE_TOLERANCE")) == 2e-6


def test_native_imports_are_not_module_startup_actions():
    tree = source_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith(("OCP", "OCC", "backend", "numpy")) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.module.startswith(("OCP", "OCC", "backend", "numpy"))
        else:
            assert isinstance(node, (ast.FunctionDef, ast.Assign, ast.Expr))
            if isinstance(node, ast.Expr):
                assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def test_run_has_no_input_arguments_and_checks_isolation_before_native_import():
    run = next(node for node in source_tree().body if isinstance(node, ast.FunctionDef) and node.name == "run_checks")
    assert not run.args.args and not run.args.kwonlyargs and run.args.vararg is None and run.args.kwarg is None
    first = run.body[0]
    assert isinstance(first, ast.Assign) and isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name) and first.value.func.id == "_isolated_cache"


def test_real_convenience_reader_call_requires_explicit_ocp_opt_in():
    calls = [node for node in ast.walk(source_tree()) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "namespace" and isinstance(node.func.slice, ast.Constant)
             and node.func.slice.value == "import_cad_file"]
    assert len(calls) == 1
    options = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
    assert options == {"deflection": .02, "allow_ocp": True}


def test_export_and_application_entrypoints_are_excluded_from_compiled_importer():
    function = next(node for node in source_tree().body if isinstance(node, ast.FunctionDef)
                    and node.name == "_importer_namespace")
    methods = ast.literal_eval(assignment(function, "methods"))
    assert {"import_file", "_read_step", "_read_brep", "_read_iges"} <= methods
    assert not any(name.startswith(("export", "_write")) for name in methods)
    assert "_load_ocp" in methods and "_capture_native_topology" in methods
