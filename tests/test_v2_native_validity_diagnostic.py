"""Source/double checks plus explicitly gated invented native validity cases.

Default tests neither import OCP nor read physical CAD/scan/Gold. The source
equivalence check reads only the exact frozen V1 Python reader source. Native
construction and execution remain separately root-supervised.
"""
import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/v2_native_validity_diagnostic.py"
FROZEN_READER = ROOT / "docs/research/gold11_mesh_development_v1_attempt/source/backend/cad_guided_mesh_offline_v1.py"
FROZEN_READER_SHA = "55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba"
spec = importlib.util.spec_from_file_location("_v2_native_validity_diagnostic_tests", SOURCE)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
NATIVE_OPT_IN = "SHERMAN_RUN_V2_NATIVE_VALIDITY_CONSTRUCTED"
NATIVE_CASES = ("valid_box", "open_shell_invalid_solid", "two_material_owners", "unit_bound_step_box")


@pytest.fixture
def forbid_native(monkeypatch):
    calls, original = [], builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise AssertionError("NATIVE_IMPORT_FORBIDDEN")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return calls


def test_whole_module_import_is_standard_library_only(forbid_native):
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    assert forbid_native == []
    tree = ast.parse(SOURCE.read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert {alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names} == {
        "hashlib", "json", "math", "sys", "tempfile"}
    assert {node.module for node in imports if isinstance(node, ast.ImportFrom)} == {"pathlib"}


@pytest.mark.parametrize("allow", (False, None, 1, "true"))
def test_opt_in_precedes_byte_or_native_work(allow, forbid_native):
    report = diagnostic.diagnose_v2_native_validity(object(), allow_ocp=allow)
    assert report["reasons"] == ["EXPLICIT_OCP_OPT_IN_REQUIRED"]
    assert report["step_import_count"] == report["native_analyzer_count"] == 0
    assert forbid_native == []


@pytest.mark.parametrize("blob", (None, "step", bytearray(b"step"), b"", b"x" * (1024**2 + 1)))
def test_only_bounded_exact_bytes_are_accepted(blob, forbid_native):
    report = diagnostic.diagnose_v2_native_validity(blob, allow_ocp=True)
    assert report["reasons"] == ["EXACT_CAD_BYTES_REQUIRED"]
    assert forbid_native == []


@pytest.mark.parametrize("blob", (b"invented", b"x" * 260402))
def test_intended_consumed_hash_cannot_be_overridden(blob, forbid_native):
    report = diagnostic.diagnose_v2_native_validity(blob, allow_ocp=True)
    assert diagnostic.INTENDED_CAD_SHA256 == "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"
    assert diagnostic.INTENDED_CAD_BYTES == 260402
    assert report["reasons"] == ["CONSUMED_CAD_BINDING_MISMATCH"]
    assert report["observed_cad_sha256"] == hashlib.sha256(blob).hexdigest()
    assert report["step_import_count"] == report["native_analyzer_count"] == 0
    assert forbid_native == []


def _bind_invented_double(monkeypatch):
    blob = b"invented reader double only"
    monkeypatch.setattr(diagnostic, "INTENDED_CAD_SHA256", hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(diagnostic, "INTENDED_CAD_BYTES", len(blob))
    return blob


def test_one_import_passes_same_native_handle_without_copy_or_retry(monkeypatch, forbid_native):
    blob, handle, calls = _bind_invented_double(monkeypatch), object(), []

    def read(value):
        assert value is blob
        calls.append("read")
        return handle, {"target_unit": "mm"}

    def inspect(value, report):
        assert value is handle
        calls.append("inspect")
        report.update(native_valid=False, evidence_complete=True, status="DIAGNOSTIC_COMPLETE")

    monkeypatch.setattr(diagnostic, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(diagnostic, "_inspect_imported_shape", inspect)
    report = diagnostic.diagnose_v2_native_validity(blob, allow_ocp=True)
    assert calls == ["read", "inspect"]
    assert report["step_import_count"] == 1
    assert report["native_valid"] is False
    assert report["evidence_complete"] is True  # diagnostic completed, not valid solid
    assert report["production_use_allowed"] is False
    assert report["conformance"] == "INDETERMINATE"
    assert report["observed_bend_count"] is None
    assert forbid_native == []


def test_failed_read_has_no_diagnostic_or_retry(monkeypatch, forbid_native):
    blob, calls = _bind_invented_double(monkeypatch), []

    def fail(_):
        calls.append("read")
        raise ValueError("STEP_PARTIAL_ROOT_TRANSFER")

    monkeypatch.setattr(diagnostic, "_read_step_owned_snapshot", fail)
    monkeypatch.setattr(diagnostic, "_inspect_imported_shape", lambda *args: pytest.fail("unexpected inspection"))
    report = diagnostic.diagnose_v2_native_validity(blob, allow_ocp=True)
    assert calls == ["read"]
    assert report["native_valid"] is None and report["native_analyzer_count"] == 0
    assert report["reasons"] == ["DIAGNOSTIC_STOP:STEP_PARTIAL_ROOT_TRANSFER"]
    assert report["evidence_complete"] is False
    assert forbid_native == []


def test_unit_bound_reader_body_matches_frozen_v1_except_scratch_prefix():
    frozen = FROZEN_READER.read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == FROZEN_READER_SHA

    def reader_body(blob):
        tree = ast.parse(blob)
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_read_step_owned_snapshot")
        node.body = node.body[1:]  # descriptive docstring differs, not behavior
        for item in ast.walk(node):
            if isinstance(item, ast.Constant) and item.value == "v2-native-validity-":
                item.value = "cad-mesh-v1-"
        return ast.dump(node, include_attributes=False)

    assert reader_body(SOURCE.read_bytes()) == reader_body(frozen)


class Status(int):
    def __str__(self):
        return "BRepCheck_Status.TEST_" + str(int(self))


def test_status_names_codes_and_errors_are_preserved_without_whitelisting():
    names = {}
    codes, complete = diagnostic._status_codes([Status(0), Status(8), Status(36)], names)
    assert codes == [0, 8, 36] and complete is True
    assert names == {"0": "TEST_0", "8": "TEST_8", "36": "TEST_36"}


def test_status_overflow_is_explicit_and_never_claims_complete():
    codes, complete = diagnostic._status_codes((Status(0) for _ in range(33)), {})
    assert codes == [0] * 32 and complete is False
    assert diagnostic._status_codes([], {}) == ([], False)


def test_report_overflow_cannot_publish_a_favorable_prefix():
    report = diagnostic._base_report()
    report.update(native_valid=True, evidence_complete=True, status="DIAGNOSTIC_COMPLETE",
                  oversized="x" * diagnostic.MAX_REPORT_BYTES)
    bounded = diagnostic._bounded_report(report)
    assert bounded["status"] == "UNAVAILABLE"
    assert bounded["evidence_complete"] is False
    assert bounded["diagnostics_truncated"] is True
    assert bounded["native_valid"] is True  # retained observation, not a green gate
    assert bounded["production_use_allowed"] is False
    assert len(diagnostic._encoded(bounded)) < diagnostic.MAX_REPORT_BYTES


def test_source_has_one_exact_analyzer_and_no_alternate_geometry_pipeline():
    tree = ast.parse(SOURCE.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    analyzers = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "BRepCheck_Analyzer"]
    assert len(analyzers) == 1
    assert [item.value for item in analyzers[0].args[1:]] == [True, False, True]
    native_imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                      and node.module and node.module.startswith("OCP.")}
    assert native_imports == {"OCP.STEPControl", "OCP.IFSelect", "OCP.TColStd", "OCP.TopAbs",
                              "OCP.TopTools", "OCP.TopoDS", "OCP.BRep", "OCP.BRepAdaptor", "OCP.BRepCheck"}
    attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert not attributes.intersection({"InContext", "Minimum", "Blind", "Perform", "SetExactMethod",
                                        "SetStatus", "UpdateEdge", "UpdateVertex", "UpdateFace"})
    assert diagnostic.MAX_TREE_NODES == 4096 and diagnostic.MAX_DETAILS == 1024
    assert diagnostic.MAX_CONTEXT_RECORDS == 4096 and diagnostic.MAX_STATUSES_PER_LIST == 32


def _cached_result_doubles(monkeypatch, context_lists):
    """Exercise aggregation/truncation without importing a native extension."""
    handles, calls = [object() for _ in context_lists], []

    class Result:
        def __init__(self, contexts):
            self.contexts, self.index = contexts, 0

        def Status(self):
            return [Status(0)]

        def InitContextIterator(self):
            self.index = 0

        def MoreShapeInContext(self):
            return self.index < len(self.contexts)

        def ContextualShape(self):
            return handles[0]

        def StatusOnShape(self):
            return self.contexts[self.index]

        def NextShapeInContext(self):
            self.index += 1

    results = [Result(contexts) for contexts in context_lists]

    class Analyzer:
        def __init__(self, *args):
            calls.append(args)

        def IsValid(self, shape=None):
            return False if shape is None else shape is handles[0]

        def Result(self, shape):
            return results[handles.index(shape)]

    parent, child = types.ModuleType("OCP"), types.ModuleType("OCP.BRepCheck")
    parent.__path__ = []
    child.BRepCheck_Analyzer, child.BRepCheck_NoError = Analyzer, Status(0)
    monkeypatch.setitem(sys.modules, "OCP", parent)
    monkeypatch.setitem(sys.modules, "OCP.BRepCheck", child)
    mapping = types.SimpleNamespace(FindIndex=lambda shape: handles.index(shape) + 1)
    entities = [dict(native_id=i, type="FACE") for i in range(len(handles))]
    monkeypatch.setattr(diagnostic, "_topology", lambda shape: (mapping, handles, entities, [], {"FACE": len(handles)}))
    monkeypatch.setattr(diagnostic, "_owner_solid", lambda shape: (shape, []))
    monkeypatch.setattr(diagnostic, "_properties", lambda *args: {})
    return handles[0], calls


def test_cached_context_errors_survive_favorable_self_status(monkeypatch):
    shape, calls = _cached_result_doubles(monkeypatch, [[[Status(8)]]])
    report = diagnostic._inspect_imported_shape(shape, diagnostic._base_report())
    assert calls == [(shape, True, False, True)]
    assert report["status"] == "DIAGNOSTIC_COMPLETE" and report["native_valid"] is False
    assert report["entity_details"][0]["self_status_codes"] == [0]
    assert report["entity_details"][0]["contexts"] == [[0, [8]]]
    assert report["status_summary"]["context_error_records"] == 1


def test_context_count_truncation_cannot_certify_favorable_prefix(monkeypatch):
    shape, calls = _cached_result_doubles(monkeypatch, [[[Status(0)]] * 64 + [[Status(8)]]])
    report = diagnostic._inspect_imported_shape(shape, diagnostic._base_report())
    assert len(calls) == 1 and report["status_context_count_observed"] == 64
    assert report["status"] == "PARTIAL_DIAGNOSTIC" and report["evidence_complete"] is False
    assert report["diagnostics_truncated"] is True and report["native_valid"] is False


def test_total_context_budget_is_global_not_reset_per_entity(monkeypatch):
    shape, calls = _cached_result_doubles(monkeypatch, [[[Status(0)], [Status(0)]], [[Status(8)]]])
    monkeypatch.setattr(diagnostic, "MAX_CONTEXT_RECORDS", 2)
    report = diagnostic._inspect_imported_shape(shape, diagnostic._base_report())
    assert len(calls) == 1 and report["status_context_count_observed"] == 2
    assert report["entity_details"][1]["detail_complete"] is False
    assert report["status"] == "PARTIAL_DIAGNOSTIC" and report["diagnostics_truncated"] is True


def test_entity_detail_cap_does_not_hide_an_unvisited_invalid_entity(monkeypatch):
    shape, calls = _cached_result_doubles(monkeypatch, [[[Status(0)]], [[Status(8)]]])
    monkeypatch.setattr(diagnostic, "MAX_DETAILS", 1)
    report = diagnostic._inspect_imported_shape(shape, diagnostic._base_report())
    assert len(calls) == 1 and len(report["entity_details"]) == 1
    assert report["status"] == "PARTIAL_DIAGNOSTIC" and report["native_valid"] is False
    assert report["evidence_complete"] is False and report["diagnostics_truncated"] is True


def test_topology_budget_stops_before_analyzer_construction(monkeypatch):
    shape, calls = _cached_result_doubles(monkeypatch, [[]])

    def refuse(_):
        raise diagnostic.DiagnosticStop("DIAGNOSTIC_TOPOLOGY_BUDGET")

    monkeypatch.setattr(diagnostic, "_topology", refuse)
    with pytest.raises(diagnostic.DiagnosticStop, match="DIAGNOSTIC_TOPOLOGY_BUDGET"):
        diagnostic._inspect_imported_shape(shape, diagnostic._base_report())
    assert calls == []


def _assert_native_matrix_report(report, case):
    """Judge the exported bounded report, never an unexportable favorable object."""
    assert report["production_use_allowed"] is False
    if case == "two_material_owners":
        assert report["native_analyzer_count"] == 0
        assert report["reasons"] == ["SINGLE_SOLID_OWNER_REQUIRED"]
        assert report["evidence_complete"] is False
    else:
        assert report["status"] == "DIAGNOSTIC_COMPLETE", report
        assert report["native_analyzer_count"] == 1 and report["evidence_complete"] is True
        assert report["native_valid"] is (case != "open_shell_invalid_solid")
        if case == "open_shell_invalid_solid":
            assert report["status_summary"]["self_error_entities"] + report["status_summary"]["context_error_records"] > 0
        else:
            assert report["status_summary"]["self_error_entities"] == 0
            assert report["status_summary"]["context_error_records"] == 0
        assert report["topology_unique_counts"]["FACE"] == (5 if case == "open_shell_invalid_solid" else 6)


def test_report_bounding_refusal_cannot_pass_the_native_matrix_oracle():
    report = diagnostic._base_report()
    report.update(native_valid=True, evidence_complete=True, status="DIAGNOSTIC_COMPLETE",
                  native_analyzer_count=1, topology_unique_counts={"FACE": 6},
                  status_summary={"self_error_entities": 0, "context_error_records": 0},
                  oversized="x" * diagnostic.MAX_REPORT_BYTES)
    _assert_native_matrix_report(report, "valid_box")
    bounded = diagnostic._bounded_report(report)
    with pytest.raises(AssertionError):
        _assert_native_matrix_report(bounded, "valid_box")


def run_constructed_v2_native_validity_checks():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        raise RuntimeError("EXPLICIT_ROOT_CONSTRUCTED_NATIVE_OPT_IN_REQUIRED")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Shell, TopoDS_Solid
    records = []
    for case in NATIVE_CASES:
        row = dict(case=case, passed=False, phase="fixture")
        try:
            shape = BRepPrimAPI_MakeBox(10., 12., 14.).Shape()
            if case == "open_shell_invalid_solid":
                # The original attempt tried Remove on a locked builder-owned
                # shell and failed before inspection. Keep the intended five
                # faces, but assemble them in new owned containers without
                # unlocking/mutating the original box or repairing any geometry.
                shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
                shell_value = shell_explorer.Current()
                source_shell = shell_value.Oriented(shell_value.Orientation())
                builder, shell, solid = BRep_Builder(), TopoDS_Shell(), TopoDS_Solid()
                builder.MakeShell(shell)
                face_explorer = TopExp_Explorer(source_shell, TopAbs_FACE)
                assert face_explorer.More()
                face_explorer.Next()  # Same intended omission: first box face.
                retained_faces = 0
                while face_explorer.More():
                    face_value = face_explorer.Current()
                    builder.Add(shell, face_value.Oriented(face_value.Orientation()))
                    retained_faces += 1
                    face_explorer.Next()
                assert retained_faces == 5
                builder.MakeSolid(solid)
                builder.Add(solid, shell)
                shape = solid
            if case == "two_material_owners":
                builder, compound = BRep_Builder(), TopoDS_Compound()
                builder.MakeCompound(compound)
                builder.Add(compound, shape)
                builder.Add(compound, BRepPrimAPI_MakeBox(2., 3., 4.).Shape())
                shape = compound
            report = diagnostic._base_report()
            report.update(scope="INVENTED_NATIVE_VALIDITY_DIAGNOSTIC_ONLY",
                          expected_cad_sha256=None, expected_cad_bytes=None)
            if case == "unit_bound_step_box":
                with tempfile.TemporaryDirectory(prefix="invented-validity-v2-") as folder:
                    path = Path(folder) / "invented.step"
                    writer = STEPControl_Writer()
                    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
                    assert writer.Write(str(path)) == IFSelect_RetDone
                    blob = path.read_bytes()
                    assert 0 < len(blob) < diagnostic.MAX_CAD_BYTES
                shape, units = diagnostic._read_step_owned_snapshot(blob)
                report.update(step_import_count=1, cad_units=units)
                assert units["target_unit"] == "mm" and units["transferred_roots"] == 1
            row["phase"] = "native_diagnostic"
            try:
                diagnostic._inspect_imported_shape(shape, report)
            except diagnostic.DiagnosticStop as error:
                report["reasons"].append(str(error))
            report = diagnostic._bounded_report(report)
            row["report"] = report
            json.dumps(report, allow_nan=False)
            _assert_native_matrix_report(report, case)
            row.update(passed=True, phase="complete")
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error))
        records.append(row)
    return dict(scope="CONSTRUCTED_V2_NATIVE_VALIDITY_ONLY", expected_case_count=4,
                scored_case_count=len(records), passed=sum(row["passed"] for row in records), records=records)


def test_constructed_entry_opt_in_before_native(forbid_native, monkeypatch):
    monkeypatch.delenv(NATIVE_OPT_IN, raising=False)
    with pytest.raises(RuntimeError, match="EXPLICIT_ROOT_CONSTRUCTED_NATIVE_OPT_IN_REQUIRED"):
        run_constructed_v2_native_validity_checks()
    assert forbid_native == []


@pytest.mark.skipif(os.environ.get(NATIVE_OPT_IN) != "1", reason="Separate root-supervised native opt-in required")
def test_fixed_constructed_native_validity_matrix():
    report = run_constructed_v2_native_validity_checks()
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    assert report["expected_case_count"] == report["scored_case_count"] == report["passed"] == 4
