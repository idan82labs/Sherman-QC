"""One exact consumed-CAD validity observation; no inference or repair.

Public input is bytes, never a raw path or scan. An external audited supervisor
must enforce the fixed wall/CPU/RSS/output limits and one-shot custody. The CLI
flag is native opt-in, not authority. Native import is lazy; importing this file
does not open data, load the application or import OCP.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile


INTENDED_CAD_SHA256 = "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"
INTENDED_CAD_BYTES = 260402
MAX_CAD_BYTES = 1024**2
MAX_TREE_NODES = 4096
MAX_TREE_DEPTH = 32
MAX_UNIQUE_SHAPES = 2048
MAX_DETAILS = 1024
MAX_STATUSES_PER_LIST = 32
MAX_CONTEXTS_PER_ENTITY = 64
MAX_CONTEXT_RECORDS = 4096
MAX_REPORT_BYTES = 768 * 1024
TYPE_CAPS = dict(COMPOUND=32, COMPSOLID=16, SOLID=8, SHELL=64,
                 FACE=256, WIRE=512, EDGE=1024, VERTEX=1024)
EXTERNAL_LIMITS = dict(wall_seconds=120, cpu_seconds=90,
                       sampled_process_tree_rss_bytes=1024**3,
                       combined_output_bytes=1024**2)


class DiagnosticStop(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise DiagnosticStop(reason)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _base_report():
    return dict(schema="V2_NATIVE_VALIDITY_DIAGNOSTIC", status="UNAVAILABLE",
                scope="ONE_CONSUMED_CAD_NATIVE_VALIDITY_NOT_INFERENCE",
                reasons=[], evidence_complete=False, native_valid=None,
                diagnostics_truncated=False, production_use_allowed=False,
                conformance="INDETERMINATE", physical_generalization_claim=False,
                observed_bend_count=None, predictions_not_computed=True,
                native_identity_scope="THIS_IMPORT_ONLY_NOT_PERSISTENT_OR_V1_ID_EQUIVALENCE",
                native_coordinate_scope="IMPORTED_CAD_FRAME_TARGET_MM_NOT_SCAN_REGISTRATION",
                expected_cad_sha256=INTENDED_CAD_SHA256,
                expected_cad_bytes=INTENDED_CAD_BYTES,
                topology_caps=dict(tree_nodes=MAX_TREE_NODES, depth=MAX_TREE_DEPTH,
                                   unique_shapes=MAX_UNIQUE_SHAPES, by_type=TYPE_CAPS),
                detail_caps=dict(entities=MAX_DETAILS, statuses_per_list=MAX_STATUSES_PER_LIST,
                                 contexts_per_entity=MAX_CONTEXTS_PER_ENTITY,
                                 total_contexts=MAX_CONTEXT_RECORDS, report_bytes=MAX_REPORT_BYTES),
                external_supervision_required=EXTERNAL_LIMITS,
                analyzer_options=dict(geometric_controls=True, parallel=False, exact_method=True),
                native_analyzer_count=0, step_import_count=0)


def _bounded_report(report):
    if len(_encoded(report)) <= MAX_REPORT_BYTES:
        return report
    minimal = _base_report()
    minimal.update(status="UNAVAILABLE", reasons=["DIAGNOSTIC_REPORT_BYTE_BUDGET"],
                   diagnostics_truncated=True, native_valid=report.get("native_valid"),
                   native_analyzer_count=report.get("native_analyzer_count", 0),
                   step_import_count=report.get("step_import_count", 0),
                   omitted_detail_bytes_at_least=MAX_REPORT_BYTES + 1)
    return minimal


def _status_codes(values, names):
    """Preserve native enum codes; a bounded prefix is never complete evidence."""
    codes = []
    for value in values:
        if len(codes) == MAX_STATUSES_PER_LIST:
            return codes, False
        code, name = int(value), str(value).split(".")[-1]
        _require(len(name) <= 128, "NATIVE_STATUS_NAME_BUDGET")
        previous = names.setdefault(str(code), name)
        _require(previous == name, "CONFLICTING_NATIVE_STATUS_ENUM")
        codes.append(code)
    return codes, bool(codes)


def _read_step_owned_snapshot(cad_bytes):
    """Same unit-bound single reader/transfer sequence as frozen mesh V1."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TColStd import TColStd_SequenceOfAsciiString

    reader = STEPControl_Reader()
    with tempfile.TemporaryDirectory(prefix="v2-native-validity-") as directory:
        path = Path(directory) / "snapshot.step"
        with path.open("xb") as stream:
            stream.write(cad_bytes)
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise ValueError("STEP_READ_FAILED")
        lengths, angles, solid_angles = (TColStd_SequenceOfAsciiString() for _ in range(3))
        reader.FileUnits(lengths, angles, solid_angles)
        declarations = {}
        for key, seq in (("length", lengths), ("plane_angle", angles), ("solid_angle", solid_angles)):
            if not 0 <= seq.Length() <= 64:
                raise ValueError("STEP_UNIT_DECLARATION_BUDGET")
            declarations[key] = [str(seq.Value(i).ToCString()) for i in range(1, seq.Length()+1)]
            if any(len(v) > 128 for v in declarations[key]):
                raise ValueError("STEP_UNIT_DECLARATION_BUDGET")
        if not declarations["length"]:
            raise ValueError("STEP_LENGTH_DECLARATION_ABSENT")
        roots = int(reader.NbRootsForTransfer())
        if not 1 <= roots <= 32:
            raise ValueError("STEP_TRANSFER_ROOT_BUDGET")
        reader.SetSystemLengthUnit(1.)
        if abs(float(reader.SystemLengthUnit())-1.) > 1e-12:
            raise ValueError("STEP_TARGET_UNIT_NOT_CONFIRMED")
        if int(reader.TransferRoots()) != roots:
            raise ValueError("STEP_PARTIAL_ROOT_TRANSFER")
        shape = reader.OneShape()
        if shape.IsNull():
            raise ValueError("NULL_NATIVE_SHAPE")
    return shape, dict(source_unit_names=declarations, target_unit="mm",
                       target_unit_action="SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER",
                       transferred_roots=roots, physical_calibration="NOT_ESTABLISHED")


def _topology(shape):
    """Bound the unmodified imported occurrence tree before native validation."""
    from OCP.TopAbs import (TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID,
                           TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX)
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS_Iterator
    type_names = dict(zip((TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL,
                           TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX), TYPE_CAPS))
    mapping, shapes, entities, occurrences = TopTools_IndexedMapOfShape(), [], [], []
    counts = {name: 0 for name in TYPE_CAPS}
    stack = [(shape, -1, 0)]
    while stack:
        current, parent_occurrence, depth = stack.pop()
        _require(len(occurrences) < MAX_TREE_NODES and depth <= MAX_TREE_DEPTH,
                 "DIAGNOSTIC_TOPOLOGY_BUDGET")
        _require(not current.IsNull() and current.ShapeType() in type_names,
                 "UNSUPPORTED_NATIVE_TOPOLOGY_TYPE")
        kind = type_names[current.ShapeType()]
        index = mapping.FindIndex(current)
        if index == 0:
            _require(len(shapes) < MAX_UNIQUE_SHAPES and counts[kind] < TYPE_CAPS[kind],
                     "DIAGNOSTIC_UNIQUE_ENTITY_BUDGET")
            index = mapping.Add(current)
            _require(index == len(shapes) + 1, "NATIVE_IDENTITY_MAP_MISMATCH")
            shapes.append(current)
            entities.append(dict(native_id=index - 1, type=kind, type_index=counts[kind],
                                 occurrence_count=0, location_is_identity=bool(current.Location().IsIdentity())))
            counts[kind] += 1
        entities[index - 1]["occurrence_count"] += 1
        occurrence_index = len(occurrences)
        occurrences.append([index - 1, parent_occurrence, int(current.Orientation())])
        children, iterator = [], TopoDS_Iterator(current, True, True)
        while iterator.More():
            _require(len(occurrences) + len(stack) + len(children) < MAX_TREE_NODES,
                     "DIAGNOSTIC_TOPOLOGY_BUDGET")
            child = iterator.Value()
            children.append(child.Oriented(child.Orientation()))
            iterator.Next()
        stack.extend((child, occurrence_index, depth + 1) for child in reversed(children))
    return mapping, shapes, entities, occurrences, counts


def _owner_solid(shape):
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_COMPSOLID
    from OCP.TopoDS import TopoDS_Iterator
    current, path = shape, []
    while current.ShapeType() in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
        _require(len(path) < MAX_TREE_DEPTH, "DIAGNOSTIC_OWNER_DEPTH_BUDGET")
        iterator = TopoDS_Iterator(current, True, True)
        _require(iterator.More(), "SINGLE_SOLID_OWNER_REQUIRED")
        child = iterator.Value()
        current = child.Oriented(child.Orientation())
        iterator.Next()
        _require(not iterator.More(), "SINGLE_SOLID_OWNER_REQUIRED")
        path.append(0)
    _require(current.ShapeType() == TopAbs_SOLID, "SINGLE_SOLID_OWNER_REQUIRED")
    return current, path


def _properties(shape, kind):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCP.TopoDS import TopoDS
    properties = {}
    if kind in ("VERTEX", "EDGE", "FACE"):
        native = {"VERTEX": TopoDS.Vertex_s, "EDGE": TopoDS.Edge_s, "FACE": TopoDS.Face_s}[kind](shape)
        tolerance = float(BRep_Tool.Tolerance_s(native))
        valid = math.isfinite(tolerance) and tolerance >= 0
        properties.update(native_tolerance_mm=tolerance if valid else None,
                          native_tolerance_numeric_valid=valid)
    if kind == "FACE":
        properties["geometry_type"] = str(BRepAdaptor_Surface(native, True).GetType()).split(".")[-1]
    elif kind == "EDGE":
        properties.update(degenerated=bool(BRep_Tool.Degenerated_s(native)),
                          same_parameter=bool(BRep_Tool.SameParameter_s(native)),
                          same_range=bool(BRep_Tool.SameRange_s(native)))
        properties["geometry_type"] = ("DEGENERATE" if properties["degenerated"]
                                       else str(BRepAdaptor_Curve(native).GetType()).split(".")[-1])
    if "geometry_type" in properties:
        _require(len(properties["geometry_type"]) <= 128, "NATIVE_GEOMETRY_NAME_BUDGET")
    return properties


def _inspect_imported_shape(shape, report):
    """Private constructed-test seam; one analyzer on this same imported solid."""
    from OCP.BRepCheck import BRepCheck_Analyzer, BRepCheck_NoError
    mapping, shapes, entities, occurrences, counts = _topology(shape)
    solid, owner_path = _owner_solid(shape)
    report.update(owner_path=owner_path, topology_unique_counts=counts,
                  topology_occurrence_count=len(occurrences),
                  occurrence_columns=["native_id", "parent_occurrence_index", "orientation_enum"],
                  occurrences=occurrences, entities=entities,
                  status_names={}, status_context_count_observed=0)
    report["native_analyzer_count"] = 1
    analyzer = BRepCheck_Analyzer(solid, True, False, True)
    report["native_valid"] = bool(analyzer.IsValid())
    complete, details = True, []
    status_counts = dict(self_error_entities=0, context_error_records=0,
                         invalid_subtree_entities=0, unavailable_entity_details=0)
    for shape, entity in zip(shapes, entities):
        if entity["type"] in ("COMPOUND", "COMPSOLID"):
            continue  # Owner containers have no BRepCheck_Result of their own.
        if len(details) == MAX_DETAILS:
            complete = False
            report["diagnostics_truncated"] = True
            break
        row = dict(native_id=entity["native_id"], self_status_codes=[], contexts=[],
                   detail_complete=False, errors=[])
        details.append(row)
        try:
            row.update(_properties(shape, entity["type"]))
        except Exception as error:
            row["errors"].append("PROPERTY_READ:" + type(error).__name__)
        try:
            row["subtree_valid"] = bool(analyzer.IsValid(shape))
            status_counts["invalid_subtree_entities"] += not row["subtree_valid"]
            native_result = analyzer.Result(shape)
            _require(native_result is not None, "NATIVE_RESULT_UNAVAILABLE")
            codes, self_complete = _status_codes(native_result.Status(), report["status_names"])
            row["self_status_codes"] = codes
            status_counts["self_error_entities"] += any(code != int(BRepCheck_NoError) for code in codes)
            context_complete = True
            native_result.InitContextIterator()
            while native_result.MoreShapeInContext():
                if (len(row["contexts"]) == MAX_CONTEXTS_PER_ENTITY
                        or report["status_context_count_observed"] == MAX_CONTEXT_RECORDS):
                    context_complete = False
                    break
                context = native_result.ContextualShape()
                owner = mapping.FindIndex(context)
                _require(owner > 0, "NATIVE_CONTEXT_OUTSIDE_IMPORTED_TOPOLOGY")
                codes, codes_complete = _status_codes(native_result.StatusOnShape(), report["status_names"])
                row["contexts"].append([owner - 1, codes])
                report["status_context_count_observed"] += 1
                context_complete = context_complete and codes_complete
                status_counts["context_error_records"] += any(code != int(BRepCheck_NoError) for code in codes)
                native_result.NextShapeInContext()
            row["detail_complete"] = self_complete and context_complete and not row["errors"]
            if not self_complete or not context_complete:
                report["diagnostics_truncated"] = True
        except Exception as error:
            row["errors"].append("STATUS_READ:" + type(error).__name__)
        if not row["detail_complete"]:
            complete = False
            status_counts["unavailable_entity_details"] += 1
    report.update(entity_details=details, status_summary=status_counts,
                  status_summary_scope="OBSERVED_DETAILS_ONLY_IF_TRUNCATED_OR_UNAVAILABLE",
                  evidence_complete=complete,
                  status="DIAGNOSTIC_COMPLETE" if complete else "PARTIAL_DIAGNOSTIC")
    if not complete:
        report["reasons"].append("NATIVE_VALIDITY_DETAIL_INCOMPLETE")
    return report


def diagnose_v2_native_validity(cad_bytes, *, allow_ocp=False):
    """Hard-bound physical entry: exactly the admitted consumed CAD bytes only."""
    report = _base_report()
    if type(allow_ocp) is not bool or not allow_ocp:
        report["reasons"] = ["EXPLICIT_OCP_OPT_IN_REQUIRED"]
        return report
    if type(cad_bytes) is not bytes or not 0 < len(cad_bytes) <= MAX_CAD_BYTES:
        report["reasons"] = ["EXACT_CAD_BYTES_REQUIRED"]
        return report
    report["observed_cad_sha256"] = hashlib.sha256(cad_bytes).hexdigest()
    report["observed_cad_bytes"] = len(cad_bytes)
    if len(cad_bytes) != INTENDED_CAD_BYTES or report["observed_cad_sha256"] != INTENDED_CAD_SHA256:
        report["reasons"] = ["CONSUMED_CAD_BINDING_MISMATCH"]
        return report
    try:
        report["step_import_count"] = 1
        shape, units = _read_step_owned_snapshot(cad_bytes)
        report["cad_units"] = units
        # The same native handle passes directly to diagnostics. No healing,
        # meshing, serialization, copy, second transfer or inference intervenes.
        _inspect_imported_shape(shape, report)
    except Exception as error:
        reason = str(error) if isinstance(error, (DiagnosticStop, ValueError)) else type(error).__name__
        report.update(status="UNAVAILABLE", evidence_complete=False)
        report["reasons"].append("DIAGNOSTIC_STOP:" + reason[:160])
    return _bounded_report(report)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["--exact-cad-bytes-stdin"]:
        raise SystemExit("Use only --exact-cad-bytes-stdin under the independently audited one-shot supervisor.")
    blob = sys.stdin.buffer.read(MAX_CAD_BYTES + 1)
    report = diagnose_v2_native_validity(blob, allow_ocp=True)
    sys.stdout.buffer.write(_encoded(report) + b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
