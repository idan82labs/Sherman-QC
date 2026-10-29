"""Exact-source binding selection/adapter checks with native imports guarded out.

Only backend/cad_import.py is read. Invented Python doubles replace discovery,
every OCC/OCP/gmsh import, file readers and native geometry calls.
"""
import ast
import builtins
import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend" / "cad_import.py"


class Shape:
    # Deliberately no copy constructor, as in the OCP adapter's supported seam.
    def __init__(self):
        self.token, self.location, self.orientation = "token", 17, "REVERSED"

    def Orientation(self):
        return self.orientation

    def Oriented(self, value):
        assert value == self.orientation
        result = Shape()
        result.token, result.location, result.orientation = self.token, self.location, value
        return result


class Face:
    def __init__(self, shape):
        self.shape = shape


class Edge:
    def __init__(self, shape):
        self.shape = shape


def harness(packages=(), *, broken=None, gmsh=False):
    tree = ast.parse(SOURCE.read_text())
    declarations = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id.startswith("MAX_") for t in n.targets)]
    for name in ("_select_native_cad_binding", "_tessellation_provenance", "CADFormat", "BoundingBox", "CADMesh", "CADImportResult"):
        nodes.append(copy.deepcopy(declarations[name]))
    importer = copy.deepcopy(declarations["CADImporter"])
    names = {"__init__", "_check_occ_available", "_load_occ", "_load_ocp", "_tessellate",
             "_capture_native_topology", "_calculate_face_normals", "_read_brep", "_get_bounding_box"}
    importer.body = [n for n in importer.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in importer.body} == names
    nodes.append(importer)
    for name in ("import_cad_file", "is_cad_import_available"):
        nodes.append(copy.deepcopy(declarations[name]))
    calls, imports, probes = [], [], []

    def find_spec(name):
        assert name in ("OCC", "OCP")
        probes.append(name)
        return object() if name in packages else None

    def triangulation(face, location):
        assert isinstance(face, Face)
        calls.append(("Triangulation_s", face.shape, location))
        return "triangles"

    def degenerated(edge):
        assert isinstance(edge, Edge)
        calls.append(("Degenerated_s", edge.shape))
        return False

    def read(shape, path, builder):
        calls.append(("Read_s", shape, path, builder))
        return True

    def add(shape, bounds):
        calls.append(("Add_s", shape, bounds))

    reader = object()
    tool = SimpleNamespace(Triangulation_s=triangulation, Degenerated_s=degenerated)
    topo = SimpleNamespace(Face_s=Face, Edge_s=Edge)
    common = {
        "STEPControl": dict(STEPControl_Reader=reader),
        "IGESControl": dict(IGESControl_Reader=object()),
        "BRepTools": dict(BRepTools=SimpleNamespace(Read_s=read)),
        "BRep": dict(BRep_Builder=Shape, BRep_Tool=tool),
        "TopoDS": dict(TopoDS=topo, TopoDS_Shape=Shape, TopoDS_Compound=Shape,
                       TopoDS_Iterator=object()),
        "TopExp": dict(TopExp_Explorer=object()),
        "TopAbs": {"TopAbs_" + value: value for value in
                   ("SOLID", "SHELL", "FACE", "EDGE", "FORWARD", "REVERSED", "INTERNAL", "EXTERNAL")},
        "BRepMesh": dict(BRepMesh_IncrementalMesh=object()),
        "Bnd": dict(Bnd_Box=lambda: SimpleNamespace(Get=lambda: (0, 0, 0, 10, 20, 30))),
        "BRepBndLib": dict(BRepBndLib=SimpleNamespace(Add_s=add)),
        "TopLoc": dict(TopLoc_Location=Shape),
        "Poly": dict(Poly_Triangulation=object()),
        "IFSelect": dict(IFSelect_RetDone=1),
    }
    modules = {"OCP." + name: SimpleNamespace(**attrs) for name, attrs in common.items()}
    occ_common = copy.deepcopy(common)
    occ_common["BRepTools"] = dict(breptools=SimpleNamespace(Read=read))
    occ_common["BRepBndLib"] = dict(brepbndlib=SimpleNamespace(Add=add))
    occ_common["BRep"] = dict(BRep_Builder=Shape, BRep_Tool=SimpleNamespace(
        Triangulation=triangulation, Degenerated=degenerated))
    occ_common["TopoDS"] = dict(TopoDS_Shape=Shape, TopoDS_Compound=Shape,
                               TopoDS_Iterator=object(), topods=SimpleNamespace(Edge=Edge))
    modules.update({"OCC.Core." + name: SimpleNamespace(**attrs) for name, attrs in occ_common.items()})

    def guarded_import(name, *args, **kwargs):
        imports.append(name)
        if name == "importlib.util":
            return SimpleNamespace(find_spec=find_spec)
        if name == "types":
            return SimpleNamespace(SimpleNamespace=SimpleNamespace)
        if name == "gmsh":
            if gmsh:
                return SimpleNamespace()
            raise ImportError("absent test double")
        if name in modules and name.split(".")[0] in packages:
            if broken is not None and name == broken[0]:
                raise broken[1]("injected binding unavailable")
            return modules[name]
        raise AssertionError("Unexpected native/package import: " + name)

    namespace = {
        "__name__": __name__, "__builtins__": dict(vars(builtins), __import__=guarded_import),
        "np": np, "dataclass": dataclass, "field": field, "Enum": Enum,
        "Optional": Optional, "Dict": Dict, "List": List, "Tuple": Tuple, "Any": Any,
        "logger": SimpleNamespace(warning=lambda *args: None),
        "_import_with_gmsh": lambda *args: calls.append(("gmsh-file", args)) or "gmsh-result",
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    assert imports == []
    return namespace, calls, imports, probes, modules


@pytest.mark.parametrize("packages,allow_ocp,expected,probes", [
    ((), False, None, ["OCC"]),
    ((), True, None, ["OCC", "OCP"]),
    (("OCP",), False, None, ["OCC"]),
    (("OCP",), True, "ocp", ["OCC", "OCP"]),
    (("OCC",), False, "pythonocc", ["OCC"]),
    (("OCC",), True, "pythonocc", ["OCC"]),
    (("OCC", "OCP"), False, "pythonocc", ["OCC"]),
    (("OCC", "OCP"), True, "pythonocc", ["OCC"]),
])
def test_discovery_is_shared_deterministic_and_never_imports_native(packages, allow_ocp, expected, probes):
    namespace, _, imports, actual_probes, _ = harness(packages)
    assert namespace["_select_native_cad_binding"](allow_ocp=allow_ocp) == expected
    assert actual_probes == probes
    assert imports == ["importlib.util"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True), []])
@pytest.mark.parametrize("entrypoint", ["selector", "constructor", "import", "available"])
def test_opt_in_requires_python_bool_before_discovery_or_import(value, entrypoint):
    namespace, calls, imports, probes, _ = harness(("OCP",), gmsh=True)
    with pytest.raises(TypeError, match="allow_ocp must be a bool"):
        if entrypoint == "selector":
            namespace["_select_native_cad_binding"](allow_ocp=value)
        elif entrypoint == "constructor":
            namespace["CADImporter"](allow_ocp=value)
        elif entrypoint == "import":
            namespace["import_cad_file"]("never-opened.step", allow_ocp=value)
        else:
            namespace["is_cad_import_available"](allow_ocp=value)
    assert calls == [] and imports == [] and probes == []


def test_default_constructor_and_selector_never_admit_installed_ocp():
    namespace, calls, imports, probes, _ = harness(("OCP",))
    assert namespace["_select_native_cad_binding"]() is None
    value = namespace["CADImporter"]()
    assert value.allow_ocp is False
    with pytest.raises(ImportError, match="Selected native CAD binding is unavailable"):
        value._load_occ()
    assert value.native_binding is None
    assert probes == ["OCC", "OCC"]
    assert imports == ["importlib.util", "importlib.util"] and calls == []


@pytest.mark.parametrize("gmsh", [False, True])
def test_default_availability_ignores_ocp_and_keeps_gmsh_policy(gmsh):
    namespace, calls, imports, probes, _ = harness(("OCP",), gmsh=gmsh)
    assert namespace["is_cad_import_available"]() is gmsh
    assert probes == ["OCC"] and calls == []
    assert imports == ["importlib.util", "gmsh"]


def test_default_file_import_ignores_ocp_and_preserves_gmsh_path():
    namespace, calls, imports, probes, _ = harness(("OCP",), gmsh=True)
    assert namespace["import_cad_file"]("never-opened.step", 0.25) == "gmsh-result"
    assert calls == [("gmsh-file", ("never-opened.step", 0.25))]
    assert probes == ["OCC"] and imports == ["importlib.util"]


def test_ocp_static_method_and_value_copy_adapters():
    namespace, calls, imports, _, modules = harness(("OCP",))
    value = namespace["CADImporter"](allow_ocp=True)
    value._load_occ()
    assert value.native_binding == "ocp" and value._occ_loaded
    assert value._native_topology_api_available
    assert not any(name.startswith("OCC") for name in imports)
    shape, location = Shape(), Shape()
    clone = value._TopoDS_Shape(shape)
    assert clone is not shape and vars(clone) == vars(shape)
    assert isinstance(value._TopoDS_Shape(), Shape)
    assert value._BRep_Tool.Triangulation(shape, location) == "triangles"
    assert calls[-1] == ("Triangulation_s", shape, location)
    assert value._BRep_Tool.Degenerated(value._native_edge_cast(shape)) is False
    assert calls[-1] == ("Degenerated_s", shape)
    assert value._breptools.Read(shape, "invented.brep", "builder") is True
    assert calls[-1] == ("Read_s", shape, "invented.brep", "builder")
    value._brepbndlib.Add(shape, "bounds")
    assert calls[-1] == ("Add_s", shape, "bounds")
    assert value._STEPControl_Reader is modules["OCP.STEPControl"].STEPControl_Reader
    before = list(imports)
    value._load_occ()
    assert imports == before


def test_ocp_shape_copy_does_not_call_unsupported_constructor():
    namespace, _, _, _, _ = harness(("OCP",))
    value = namespace["CADImporter"](allow_ocp=True)
    value._load_occ()
    with pytest.raises(TypeError):
        Shape(Shape())
    original = Shape()
    copied = value._TopoDS_Shape(original)
    assert copied is not original and copied.token == original.token
    assert copied.location == original.location and copied.Orientation() == original.Orientation()


@pytest.mark.parametrize("allow_ocp", [False, True])
def test_pythonocc_is_preferred_and_its_existing_adapters_stay_direct(allow_ocp):
    namespace, _, imports, probes, modules = harness(("OCC", "OCP"))
    value = namespace["CADImporter"](allow_ocp=allow_ocp)
    value._load_occ()
    assert value.native_binding == "pythonocc"
    assert probes == ["OCC"]
    assert not any(name.startswith("OCP") for name in imports)
    assert value._BRep_Tool is modules["OCC.Core.BRep"].BRep_Tool
    assert value._TopoDS_Shape is Shape
    assert value._breptools is modules["OCC.Core.BRepTools"].breptools


@pytest.mark.parametrize("package", ["OCC", "OCP"])
@pytest.mark.parametrize("error", [ImportError, OSError])
def test_broken_selected_binding_is_not_absence_or_fallback(package, error):
    module = "OCC.Core.STEPControl" if package == "OCC" else "OCP.STEPControl"
    packages = ("OCC", "OCP") if package == "OCC" else ("OCP",)
    namespace, calls, imports, _, _ = harness(packages, broken=(module, error), gmsh=True)
    assert namespace["is_cad_import_available"](allow_ocp=True) is False
    result = namespace["import_cad_file"]("never-opened.step", allow_ocp=True)
    assert result.success is False
    assert result.error == "Selected native CAD binding is unavailable: " + ("pythonocc" if package == "OCC" else "ocp")
    assert "gmsh" not in imports and calls == []
    if package == "OCC":
        assert not any(name.startswith("OCP") for name in imports)


@pytest.mark.parametrize("package", ["OCC", "OCP"])
def test_selected_file_failure_is_returned_without_another_backend_attempt(package):
    namespace, calls, imports, _, _ = harness((package,), gmsh=True)
    result = object()
    attempted = []
    namespace["CADImporter"].import_file = lambda self, path: attempted.append((self.native_binding, path)) or result
    assert namespace["import_cad_file"]("never-opened.step", 0.25, allow_ocp=True) is result
    assert attempted == [("pythonocc" if package == "OCC" else "ocp", "never-opened.step")]
    assert calls == [] and "gmsh" not in imports


def test_unexpected_selected_file_exception_is_not_retried():
    namespace, calls, imports, _, _ = harness(("OCP",), gmsh=True)
    def fail(self, path):
        raise RuntimeError("invented file failure")
    namespace["CADImporter"].import_file = fail
    with pytest.raises(RuntimeError, match="invented file failure"):
        namespace["import_cad_file"]("never-opened.step", allow_ocp=True)
    assert calls == [] and "gmsh" not in imports


@pytest.mark.parametrize("gmsh", [False, True])
def test_gmsh_availability_only_when_native_packages_are_absent(gmsh):
    namespace, _, imports, _, _ = harness(gmsh=gmsh)
    assert namespace["is_cad_import_available"]() is gmsh
    assert imports == ["importlib.util", "gmsh"]


def test_no_native_packages_selects_existing_gmsh_file_path_once():
    namespace, calls, imports, _, _ = harness()
    assert namespace["import_cad_file"]("never-opened.step", 0.25) == "gmsh-result"
    assert calls == [("gmsh-file", ("never-opened.step", 0.25))]
    assert imports == ["importlib.util"]


def test_ocp_brep_and_bbox_wrappers_preserve_call_arguments():
    namespace, calls, _, _, _ = harness(("OCP",))
    value = namespace["CADImporter"](allow_ocp=True)
    value._load_occ()
    shape = value._read_brep("invented-not-opened.brep")
    assert isinstance(shape, Shape)
    assert calls[-1][0] == "Read_s" and calls[-1][1] is shape
    assert calls[-1][2] == "invented-not-opened.brep"
    bounds = value._get_bounding_box(shape)
    assert bounds.min_point == (0, 0, 0) and bounds.max_point == (10, 20, 30)
    assert calls[-1][0] == "Add_s" and calls[-1][1] is shape


def test_ocp_tessellation_records_binding_without_promoting_native_metadata():
    namespace, _, _, _, _ = harness(("OCP",))
    value = namespace["CADImporter"](allow_ocp=True)
    value._load_occ()
    value._BRepMesh_IncrementalMesh = lambda *args: SimpleNamespace(Perform=lambda: None)
    value._TopExp_Explorer = lambda *args: SimpleNamespace(More=lambda: False)
    result = value._tessellate(Shape())
    assert result.tessellation_provenance["backend"] == "ocp"
    assert result.tessellation_provenance["native_topology_status"] == "UNKNOWN"
    assert result.native_topology["status"] == "UNAVAILABLE"
    assert result.native_topology["reasons"] == ["NO_NATIVE_FACES"]
