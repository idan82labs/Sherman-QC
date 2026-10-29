"""Pure rational/source guards for the native worker; never import Open3D."""

import ast
from fractions import Fraction
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cad_native_open3d_checks.py"


@pytest.fixture
def capsule():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    names = {"rational_rotations", "exact_rotation_check", "exact_float_floor"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    env = {"Fraction": Fraction}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SCRIPT), "exec"), env)
    return env, tree


def test_all_declared_rotations_are_exact_proper_rotations(capsule):
    env, _ = capsule
    rotations = env["rational_rotations"]()
    assert set(rotations) == {"identity", "axis_cycle", "rational_z", "rational_quaternion"}
    assert all(env["exact_rotation_check"](matrix) for matrix in rotations.values())


def test_reflection_is_not_mislabeled_proper_rotation(capsule):
    env, _ = capsule
    reflection = ((Fraction(-1), Fraction(0), Fraction(0)),
                  (Fraction(0), Fraction(1), Fraction(0)),
                  (Fraction(0), Fraction(0), Fraction(1)))
    assert not env["exact_rotation_check"](reflection)


def test_slightly_nonorthogonal_rational_matrix_rejected(capsule):
    env, _ = capsule
    matrix = [list(row) for row in env["rational_rotations"]()["rational_z"]]
    matrix[0][0] += Fraction(1, 10**20)
    assert not env["exact_rotation_check"](matrix)


@pytest.mark.parametrize("value, expected", [
    (math.nextafter(.125, 0.), 0), (.125, 1), (math.nextafter(.125, 1.), 1),
    (math.nextafter(-.125, -1.), -1), (-.125, 0), (math.nextafter(-.125, 0.), 0),
])
def test_fraction_floor_resolves_actual_binary64_boundary_neighbors(capsule, value, expected):
    env, _ = capsule
    assert env["exact_float_floor"](value, -.125, .25) == expected


def test_exact_reference_does_not_assume_rounded_subtraction_is_truth(capsule):
    env, _ = capsule
    value = math.nextafter(.125, 0.)
    # The real arithmetic of these binary64 operands lies below the boundary;
    # binary64 subtraction rounds to .25 and changes the subsequent floor.
    assert env["exact_float_floor"](value, -.125, .25) == 0
    assert math.floor((value + .125) / .25) == 1


def test_fixed_matrix_declaration_and_caps_are_unchanged(capsule):
    _, tree = capsule
    names = {"MATRIX_STRATA", "MATRIX_ROTATIONS", "MATRIX_SCALES", "MATRIX_DENSITIES",
             "MATRIX_CASE_COUNT", "RAW_VERTEX_CAP"}
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in names}
    assert len(values["MATRIX_STRATA"]) * len(values["MATRIX_ROTATIONS"]) * len(values["MATRIX_SCALES"]) * len(values["MATRIX_DENSITIES"]) == 96
    assert values["MATRIX_CASE_COUNT"] == 96
    assert values["RAW_VERTEX_CAP"] == 200_000
    assert values["MATRIX_SCALES"] == (.125, 1., 8.)
    assert values["MATRIX_DENSITIES"] == (1, 4)


def test_native_import_only_occurs_inside_run_checks(capsule):
    _, tree = capsule
    native = []
    for node in tree.body:
        imports = [item for item in ast.walk(node) if isinstance(item, (ast.Import, ast.ImportFrom))]
        for item in imports:
            names = [alias.name for alias in item.names] if isinstance(item, ast.Import) else [item.module]
            if "open3d" in names:
                native.append(node)
    assert len(native) == 1
    assert isinstance(native[0], ast.FunctionDef) and native[0].name == "run_checks"
    assert not any(isinstance(node, ast.If) for node in tree.body)


def test_script_has_no_file_writes_or_subprocess_network_entrypoints(capsule):
    _, tree = capsule
    banned_names = {"open", "eval", "socket", "subprocess", "requests", "urlopen"}
    banned_attributes = {"write_text", "write_bytes", "write", "mkdir", "unlink", "rmdir",
                         "read_triangle_mesh", "read_point_cloud", "read_image", "download"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_names
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in banned_attributes
