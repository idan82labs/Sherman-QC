"""Bounded native STEP inventory for a separately approved compatibility review.

The only input is an already hash-verified STEP snapshot in the supervising
worker's owned temporary directory. This module does not discover or authenticate
raw sources, mesh geometry, select datums, register scans, detect bends, qualify
features, score references, or change any file. Native geometry is import-local
CAD evidence, not a physical measurement or manufacturing-event inventory.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import stat


MAX_TRANSFER_ROOTS = 32
MAX_TREE_NODES = 8192
MAX_TREE_DEPTH = 32
MAX_FACE_OCCURRENCES = 2048
MAX_EDGE_OCCURRENCES = 8192
MAX_VERTEX_OCCURRENCES = 8192
MAX_UNIT_NAMES = 64
MAX_UNIT_NAME_LENGTH = 128


class _AdmissionStop(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _AdmissionStop(reason)


def _number(value):
    result = float(value)
    _require(math.isfinite(result), "NONFINITE_NATIVE_GEOMETRY")
    return result


def _xyz(value):
    return [_number(value.X()), _number(value.Y()), _number(value.Z())]


def _unit_xyz(value):
    xyz = _xyz(value)
    length = math.hypot(*xyz)
    _require(math.isfinite(length) and length > 0, "INVALID_NATIVE_DIRECTION")
    return [component / length for component in xyz]


def _owned_snapshot(path):
    """Require the caller's isolated cache and a regular, non-symlink snapshot.

    Source-byte authentication and read budgets belong to the supervising reader;
    a location/type check here does not claim that authentication was performed.
    """
    _require(isinstance(path, (str, Path)), "EXPLICIT_SNAPSHOT_PATH_REQUIRED")
    candidate = Path(path)
    _require(candidate.is_absolute(), "EXPLICIT_SNAPSHOT_PATH_REQUIRED")
    _require(candidate.suffix.lower() in (".step", ".stp"), "STEP_SNAPSHOT_REQUIRED")
    lexical_cache = Path(os.path.abspath(os.environ["TMPDIR"]))
    cache = lexical_cache.resolve(strict=True)
    _require(cache == Path.cwd().resolve() == Path(os.environ["HOME"]).resolve(),
             "ISOLATED_OWNED_CACHE_REQUIRED")
    lexical = Path(os.path.abspath(candidate))
    # macOS /var and /private/var can name the same supervisor-owned directory.
    # Admit either known spelling, then require resolved containment as well.
    _require(any(lexical != root and root in lexical.parents
                 for root in (lexical_cache, cache)), "SNAPSHOT_OUTSIDE_OWNED_CACHE")
    info = os.lstat(lexical)
    _require(stat.S_ISREG(info.st_mode) and info.st_size > 0, "REGULAR_STEP_SNAPSHOT_REQUIRED")
    resolved = lexical.resolve(strict=True)
    _require(resolved != cache and cache in resolved.parents, "SNAPSHOT_OUTSIDE_OWNED_CACHE")
    return resolved


def inspect_step_snapshot(path):
    """Return JSON-ready CAD admission facts without touching a live application.

    FileUnits reports source declarations. The reader's system length unit is
    explicitly set to 1.0 mm after ReadFile and before TransferRoots, and queried
    again. This records the conversion request/reader state, NOT scanner units or
    physical calibration. Every topology counter counts occurrences, not unique
    native entities or physical bends. Partial failure evidence stays explicit.
    """
    result = {
        "schema": "GOLD11_STEP_ADMISSION_REVIEW_V1", "status": "UNAVAILABLE",
        "reasons": [], "phase": "SNAPSHOT_ADMISSION",
        "input_binding_scope": "CALLER_HASH_VERIFIED_OWNED_TEMP_SNAPSHOT_REQUIRED",
        "source_bytes_verified_by_this_function": False,
        "geometry_identity_scope": "IMPORT_LOCAL_OCCURRENCES_NOT_PERSISTENT_IDS",
        "geometry_length_frame": "NATIVE_IMPORTED_CAD_FRAME",
        "physical_calibration_status": "NOT_VERIFIED",
        "scan_unit_status": "NOT_INSPECTED",
        "scan_registration_status": "NOT_RUN",
        "datum_selection_status": "NOT_RUN",
        "feature_qualification_status": "NOT_RUN",
        "native_validity_status": "NOT_CHECKED",
        "meshing_performed": False, "production_use_allowed": False,
        "topology_inventory_complete": False, "geometry_inventory_complete": False,
        "limits": {"transfer_roots": MAX_TRANSFER_ROOTS, "topology_nodes": MAX_TREE_NODES,
                   "topology_depth": MAX_TREE_DEPTH, "face_occurrences": MAX_FACE_OCCURRENCES,
                   "edge_occurrences": MAX_EDGE_OCCURRENCES,
                   "vertex_occurrences": MAX_VERTEX_OCCURRENCES},
        "topology_occurrence_counts": {}, "surface_type_histogram": {},
        "planar_faces": [], "cylindrical_faces": [],
    }
    try:
        snapshot = _owned_snapshot(path)
        result["phase"] = "IMPORT_NATIVE_READER"
        import numpy as np
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.Bnd import Bnd_Box
        from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TColStd import TColStd_SequenceOfAsciiString
        from OCP.TopAbs import (
            TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL,
            TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SHAPE,
            TopAbs_FORWARD, TopAbs_REVERSED,
        )
        from OCP.TopoDS import TopoDS, TopoDS_Iterator

        result["phase"] = "READ_STEP"
        reader = STEPControl_Reader()
        read_status = reader.ReadFile(str(snapshot))
        result["step_read_status"] = str(read_status)
        _require(read_status == IFSelect_RetDone, "STEP_READ_FAILED")

        result["phase"] = "SOURCE_UNITS_AND_TRANSFER_ADMISSION"
        lengths, angles, solid_angles = (TColStd_SequenceOfAsciiString() for _ in range(3))
        reader.FileUnits(lengths, angles, solid_angles)

        def unit_names(sequence):
            count = int(sequence.Length())
            _require(0 <= count <= MAX_UNIT_NAMES, "SOURCE_UNIT_NAME_BUDGET")
            names = []
            for index in range(1, count + 1):
                name = str(sequence.Value(index).ToCString())
                _require(len(name) <= MAX_UNIT_NAME_LENGTH, "SOURCE_UNIT_NAME_BUDGET")
                names.append(name)
            return names

        source_units = {"length": unit_names(lengths), "plane_angle": unit_names(angles),
                        "solid_angle": unit_names(solid_angles)}
        default_system_unit = _number(reader.SystemLengthUnit())
        _require(default_system_unit > 0, "INVALID_NATIVE_SYSTEM_LENGTH_UNIT")
        root_count = int(reader.NbRootsForTransfer())
        result.update(source_unit_names=source_units,
                      source_unit_evidence="STEP_SOURCE_DECLARATIONS_NOT_PHYSICAL_CALIBRATION",
                      source_length_unit_declaration_present=bool(source_units["length"]),
                      system_length_unit_before_override=default_system_unit,
                      roots_for_transfer=root_count)
        _require(1 <= root_count <= MAX_TRANSFER_ROOTS, "STEP_TRANSFER_ROOT_BUDGET")
        reader.SetSystemLengthUnit(1.0)
        system_unit = _number(reader.SystemLengthUnit())
        _require(abs(system_unit - 1.) <= 1e-12, "SYSTEM_LENGTH_UNIT_OVERRIDE_NOT_CONFIRMED")
        result.update(system_length_unit_for_transfer=system_unit, target_length_unit="mm",
                      conversion_action="SET_SYSTEM_LENGTH_UNIT_1_MM_AFTER_READ_BEFORE_TRANSFER",
                      conversion_validation="READER_STATE_CONFIRMED_NOT_PHYSICALLY_CALIBRATED")

        result["phase"] = "TRANSFER_STEP_ROOTS"
        transferred = int(reader.TransferRoots())
        result["transferred_root_count"] = transferred
        _require(1 <= transferred <= MAX_TRANSFER_ROOTS, "STEP_ROOT_TRANSFER_FAILED")
        _require(transferred == root_count, "STEP_PARTIAL_ROOT_TRANSFER")
        shape = reader.OneShape()
        _require(not shape.IsNull(), "NULL_TRANSFERRED_SHAPE")

        result["phase"] = "BOUNDED_TOPOLOGY_INVENTORY"
        types = {
            TopAbs_COMPOUND: "COMPOUND", TopAbs_COMPSOLID: "COMPSOLID",
            TopAbs_SOLID: "SOLID", TopAbs_SHELL: "SHELL", TopAbs_FACE: "FACE",
            TopAbs_WIRE: "WIRE", TopAbs_EDGE: "EDGE", TopAbs_VERTEX: "VERTEX",
            TopAbs_SHAPE: "SHAPE",
        }
        result["top_level_type"] = types.get(shape.ShapeType(), "UNKNOWN")
        counts = result["topology_occurrence_counts"]
        stack, faces, nodes, maximum_depth = [(shape, 0)], [], 0, 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            maximum_depth = max(maximum_depth, depth)
            result.update(topology_nodes_visited=nodes, maximum_topology_depth=maximum_depth)
            _require(nodes <= MAX_TREE_NODES, "NATIVE_TOPOLOGY_NODE_BUDGET")
            _require(depth <= MAX_TREE_DEPTH, "NATIVE_TOPOLOGY_DEPTH_BUDGET")
            kind = current.ShapeType()
            name = types.get(kind, "UNKNOWN")
            counts[name] = counts.get(name, 0) + 1
            if kind == TopAbs_FACE:
                _require(counts[name] <= MAX_FACE_OCCURRENCES, "NATIVE_FACE_OCCURRENCE_BUDGET")
                faces.append(TopoDS.Face_s(current))
            elif kind == TopAbs_EDGE:
                _require(counts[name] <= MAX_EDGE_OCCURRENCES, "NATIVE_EDGE_OCCURRENCE_BUDGET")
            elif kind == TopAbs_VERTEX:
                _require(counts[name] <= MAX_VERTEX_OCCURRENCES, "NATIVE_VERTEX_OCCURRENCE_BUDGET")
            iterator = TopoDS_Iterator(current, True, True)
            while iterator.More():
                _require(nodes + len(stack) < MAX_TREE_NODES, "NATIVE_TOPOLOGY_NODE_BUDGET")
                _require(depth + 1 <= MAX_TREE_DEPTH, "NATIVE_TOPOLOGY_DEPTH_BUDGET")
                child = iterator.Value()
                stack.append((child.Oriented(child.Orientation()), depth + 1))
                iterator.Next()
        result.update(topology_inventory_complete=True,
                      solid_occurrence_count=counts.get("SOLID", 0),
                      topology_count_semantics="OCCURRENCES_NOT_UNIQUE_ENTITIES")

        # No whole-shape geometry operation occurs before topology admission.
        result["phase"] = "EXACT_GEOMETRY_BOUNDING_BOX"
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        _require(not box.IsVoid(), "VOID_NATIVE_BOUNDING_BOX")
        extrema = [_number(value) for value in box.Get()]
        _require(len(extrema) == 6 and all(extrema[index] <= extrema[index + 3] for index in range(3)),
                 "INVALID_NATIVE_BOUNDING_BOX")
        result["bounding_box"] = {
            "minimum": extrema[:3], "maximum": extrema[3:], "unit": "mm",
            "method": "BRepBndLib.AddOptimal_NO_TRIANGULATION_NO_SHAPE_TOLERANCE_PADDING",
            "numerical_precision_scope": "NATIVE_FLOATING_POINT_NOT_SYMBOLIC_EXACTNESS",
        }

        result["phase"] = "BOUNDED_ANALYTIC_SURFACE_INVENTORY"
        planar_normals = []
        for index, face in enumerate(faces):
            surface = BRepAdaptor_Surface(face, True)
            kind = surface.GetType()
            name = str(kind).rsplit(".", 1)[-1]
            histogram = result["surface_type_histogram"]
            histogram[name] = histogram.get(name, 0) + 1
            orientation = face.Orientation()
            orientation_name = str(orientation).rsplit(".", 1)[-1]
            if kind == GeomAbs_Plane:
                properties = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, properties, False, False)
                area = _number(properties.Mass())
                _require(area >= 0, "NEGATIVE_NATIVE_SURFACE_AREA")
                normal = _unit_xyz(surface.Plane().Axis().Direction())
                if orientation == TopAbs_REVERSED:
                    normal = [-value for value in normal]
                signed_normal = orientation in (TopAbs_FORWARD, TopAbs_REVERSED)
                result["planar_faces"].append({
                    "face_occurrence_index": index, "orientation": orientation_name,
                    "area_mm2": area, "centroid_mm": _xyz(properties.CentreOfMass()),
                    "normal": normal,
                    "normal_orientation_status": "FACE_ORIENTED" if signed_normal else "UNORIENTED_PLANE_ONLY",
                    "area_method": "NATIVE_SURFACE_PROPERTIES_NO_TRIANGULATION",
                })
                if area > 0:
                    planar_normals.append(normal)
            elif kind == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                radius = _number(cylinder.Radius())
                _require(radius > 0, "NONPOSITIVE_NATIVE_CYLINDER_RADIUS")
                result["cylindrical_faces"].append({
                    "face_occurrence_index": index, "orientation": orientation_name,
                    "radius_mm": radius, "axis_direction": _unit_xyz(cylinder.Axis().Direction()),
                    "axis_location_mm": _xyz(cylinder.Location()),
                    "parameter_bounds": {
                        "u_min": _number(surface.FirstUParameter()),
                        "u_max": _number(surface.LastUParameter()),
                        "v_min": _number(surface.FirstVParameter()),
                        "v_max": _number(surface.LastVParameter()),
                    },
                    "trim_semantics": "NATIVE_PARAMETER_BOUNDS_NOT_RECTANGULAR_TRIM_CERTIFICATION",
                    "inner_outer_or_bend_classification": "NOT_PERFORMED",
                })
        singular = None
        if planar_normals:
            normal_array = np.asarray(planar_normals, dtype=np.float64)
            gram = normal_array.T @ normal_array / len(normal_array)
            singular = [_number(value) for value in np.linalg.svd(gram, compute_uv=False)]
        result["cad_plane_normal_diagnostic"] = {
            "positive_area_plane_occurrence_count": len(planar_normals),
            "normalized_normal_gram_singular_values": singular,
            "normalization": "MEAN_OF_UNIT_NORMAL_OUTER_PRODUCTS_UNWEIGHTED",
            "scope": "CAD_NORMAL_DIVERSITY_ONLY_NOT_SCAN_OR_SIX_DOF_POSE_OBSERVABILITY",
            "datum_suitability": "NOT_ESTABLISHED",
        }
        result.update(status="AVAILABLE_REVIEW_INVENTORY", phase="COMPLETE",
                      geometry_inventory_complete=True)
    except _AdmissionStop as error:
        result["reasons"] = [str(error)]
    except Exception as error:
        result["reasons"] = ["NATIVE_ADMISSION_ERROR:" + type(error).__name__]
    return result
