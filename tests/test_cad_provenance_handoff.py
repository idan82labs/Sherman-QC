"""Invented-input checks of native-context transport, never root verification.

Load only the pure cad_provenance module by its exact file path. Pipeline and
detector checks compile explicit AST definitions or inspect syntax; no backend
package, native library, application, CAD file, scan or research input is loaded.
The few geometry coordinates below are invented and deliberately coincident.
"""
from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass, field
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PURE_SOURCE = ROOT / "backend" / "cad_provenance.py"
PIPELINE_SOURCE = ROOT / "backend" / "bend_inspection_pipeline.py"
DETECTOR_SOURCE = ROOT / "backend" / "feature_detection" / "bend_detector.py"


@pytest.fixture(scope="module")
def api():
    tree = ast.parse(PURE_SOURCE.read_text(encoding="utf-8"), filename=str(PURE_SOURCE))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= {"math", "re", "numpy"}
        elif isinstance(node, ast.ImportFrom):
            assert node.module == "__future__"
    spec = importlib.util.spec_from_file_location("isolated_cad_provenance_contract", PURE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    assert len(selected) == len(names)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + selected,
                        type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def native_and_ranges():
    """Two owner occurrences; face 1 has no emitted vertices or triangles."""
    ranges = {
        "schema_version": 1, "kind": "CAD_TESSELLATION_PROVENANCE_V1",
        "backend": "INVENTED_NATIVE_DOUBLE", "mapping_status": "FACE_OCCURRENCE_RANGES",
        "source_face_count": 3, "vertex_count": 6, "triangle_count": 2,
        "face_records": [
            dict(source_face_index=0, vertex_range=[0, 3], triangle_range=[0, 1],
                 tessellation_status="TESSELLATED"),
            dict(source_face_index=1, vertex_range=[3, 3], triangle_range=[1, 1],
                 tessellation_status="NO_TRIANGULATION"),
            dict(source_face_index=2, vertex_range=[3, 6], triangle_range=[1, 2],
                 tessellation_status="TESSELLATED"),
        ],
        "reasons": [], "limitations": [],
    }
    def edge(identity, orientation="FORWARD", degenerate=False):
        return dict(edge_id=identity, orientation=orientation, degenerate=degenerate)
    native = {
        "schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1",
        "status": "CAPTURED_IMPORT_LOCAL_INCIDENCE",
        "owner_records": [
            dict(owner_id="owner:A", kind="SOLID", occurrence_path=[0]),
            dict(owner_id="owner:B", kind="SHELL", occurrence_path=[1, 0]),
        ],
        "face_records": [
            dict(source_face_index=0, face_id="face:0", owner_id="owner:A",
                 edge_uses=[edge("edge:shared-A"), edge("edge:seam", "FORWARD"),
                            edge("edge:seam", "REVERSED")]),
            dict(source_face_index=1, face_id="face:1", owner_id="owner:A",
                 edge_uses=[edge("edge:shared-A", "REVERSED"), edge("edge:zero", degenerate=True)]),
            dict(source_face_index=2, face_id="face:2", owner_id="owner:B",
                 edge_uses=[edge("edge:B")]),
        ],
    }
    return native, ranges


def candidate(identity="candidate:0", *, pass_index=0, rows1=(0, 1), rows2=(2,), start=(0., 0., 0.), end=(5., 0., 0.)):
    return {
        "candidate_id": identity, "detection_pass": pass_index,
        "bend_line_start": list(start), "bend_line_end": list(end),
        "flange1_source_vertex_indices": list(rows1), "flange2_source_vertex_indices": list(rows2),
        "flange1_segments": [dict(segment_id=f"pass:{pass_index}:plane:0", source_vertex_indices=list(rows1))],
        "flange2_segments": [dict(segment_id=f"pass:{pass_index}:plane:1", source_vertex_indices=list(rows2))],
    }


def transport_ledger(records):
    passes = []
    for index in range(max(record["detection_pass"] for record in records) + 1):
        segments = {}
        for record in records:
            if record["detection_pass"] == index:
                for key in ("flange1_segments", "flange2_segments"):
                    for segment in record[key]:
                        segments[segment["segment_id"]] = copy.deepcopy(segment)
        assigned = {row for segment in segments.values() for row in segment["source_vertex_indices"]}
        passes.append(dict(detection_pass=index, segments=list(segments.values()),
                           rejected_source_vertex_indices=[],
                           unassigned_source_vertex_indices=sorted(set(range(6)) - assigned)))
    return {
        "schema_version": 1, "kind": "CAD_SUPPORT_LINEAGE_V1", "status": "COMPLETE",
        "raw_vertex_count": 6, "candidate_records": records, "candidate_count": len(records),
        "processed_source_vertex_indices": [[index] for index in range(6)],
        "preprocessing_rejected_source_vertex_indices": [], "segmentation_passes": passes,
    }


def provenance_and_selected():
    native, provenance = native_and_ranges()
    records = [candidate(), candidate("candidate:1", pass_index=1, rows1=(3, 4), rows2=(5,))]
    provenance["native_topology"] = native
    provenance["support_lineage"] = transport_ledger(records)
    selected = [SimpleNamespace(cad_provenance=copy.deepcopy(provenance["support_lineage"]),
                                bend_line_start=np.array([-99., 0., 0.]),
                                bend_line_end=np.array([99., 0., 0.]))]
    return provenance, selected


def assert_no_native_authority(value):
    assert value["identity_scope"] == "IMPORT_LOCAL_NATIVE_OCCURRENCES_NOT_PERSISTENT_IDS"
    assert value["native_geometry_is_verified"] is False
    assert value["event_identity_is_verified"] is False
    assert value["source_identity_is_verified"] is False
    assert value["observed_state"] == "UNKNOWN"


def assert_no_root_authority(value):
    assert value["root_role_status"] == "NOT_NOMINATED"
    assert value["grouping_eligible"] is False and value["grouping_inputs"] is None
    assert value["event_identity_is_verified"] is False
    assert value["source_identity_is_verified"] is False
    assert value["physical_geometry_is_verified"] is False
    assert value["observed_state"] == "UNKNOWN"
    for record in value["candidate_records"]:
        assert record["root_face_ids"] == [] and record["root_role_status"] == "NOT_NOMINATED"


def test_native_capture_preserves_occurrences_orientations_seams_and_empty_face_snapshot(api):
    native, ranges = native_and_ranges()
    native.update(identity_scope="PERSISTENT_VERIFIED", native_geometry_is_verified=True,
                  event_identity_is_verified=True, observed_state="FORMED", grouping_eligible=True)
    captured = api.sanitize_native_topology(native, ranges)
    assert captured["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    assert captured["owner_records"] == native["owner_records"]
    assert captured["face_records"] == native["face_records"]
    assert "grouping_eligible" not in captured
    assert_no_native_authority(captured)
    native["face_records"][0]["edge_uses"][0]["edge_id"] = "mutated"
    native["owner_records"][0]["occurrence_path"].append(100)
    assert captured["face_records"][0]["edge_uses"][0]["edge_id"] == "edge:shared-A"
    assert captured["owner_records"][0]["occurrence_path"] == [0]


@pytest.mark.parametrize("bad", ["schema_bool", "kind", "status", "face_index_bool", "mesh_index_bool",
                                "missing_owner", "duplicate_owner", "duplicate_path", "unused_owner",
                                "wrong_face_id", "bad_orientation", "degenerate_not_bool",
                                "cross_owner_edge", "inconsistent_degeneracy", "late_face_none"])
def test_malformed_native_incidence_clears_every_record(api, bad):
    native, ranges = native_and_ranges()
    if bad == "schema_bool": native["schema_version"] = True
    elif bad == "kind": native["kind"] = "INVENTED_OTHER_KIND"
    elif bad == "status": native["status"] = "COMPLETE"
    elif bad == "face_index_bool": native["face_records"][0]["source_face_index"] = False
    elif bad == "mesh_index_bool": ranges["face_records"][0]["source_face_index"] = False
    elif bad == "missing_owner": native["face_records"][2]["owner_id"] = "owner:missing"
    elif bad == "duplicate_owner": native["owner_records"].append(copy.deepcopy(native["owner_records"][0]))
    elif bad == "duplicate_path": native["owner_records"][1]["occurrence_path"] = [0]
    elif bad == "unused_owner": native["owner_records"].append(dict(owner_id="owner:unused", kind="FACE", occurrence_path=[9]))
    elif bad == "wrong_face_id": native["face_records"][2]["face_id"] = "face:0"
    elif bad == "bad_orientation": native["face_records"][2]["edge_uses"][0]["orientation"] = "UNKNOWN"
    elif bad == "degenerate_not_bool": native["face_records"][2]["edge_uses"][0]["degenerate"] = 0
    elif bad == "cross_owner_edge": native["face_records"][2]["edge_uses"][0]["edge_id"] = "edge:shared-A"
    elif bad == "inconsistent_degeneracy": native["face_records"][1]["edge_uses"][0]["degenerate"] = True
    else: native["face_records"][-1] = None
    captured = api.sanitize_native_topology(native, ranges)
    assert captured["status"] == "UNAVAILABLE"
    assert captured["owner_records"] == captured["face_records"] == []
    assert_no_native_authority(captured)


@pytest.mark.parametrize("field", ["MAX_FACES", "MAX_EDGE_USES"])
def test_native_mapping_cap_abstains_wholly_without_rescuing_prefix(api, monkeypatch, field):
    native, ranges = native_and_ranges()
    monkeypatch.setattr(api, field, 2)
    captured = api.sanitize_native_topology(native, ranges)
    assert captured["status"] == "UNAVAILABLE" and captured["face_records"] == captured["owner_records"] == []


def test_array_valued_native_metadata_cannot_interrupt_geometry_fallback(api):
    native, ranges = native_and_ranges()
    native["owner_records"][-1]["kind"] = np.array(["SHELL", "SOLID"])
    captured = api.sanitize_native_topology(native, ranges)
    assert captured["status"] == "UNAVAILABLE" and captured["face_records"] == []
    assert_no_native_authority(captured)


def test_array_valued_lineage_metadata_clears_handoff_without_raising(api):
    provenance, selected = provenance_and_selected()
    provenance["support_lineage"]["kind"] = np.array(["CAD_SUPPORT_LINEAGE_V1", "OTHER"])
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []
    assert_no_root_authority(handoff)


def test_original_candidates_survive_selected_merge_and_map_exact_context_across_empty_face(api):
    provenance, selected = provenance_and_selected()
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "CAPTURED_SUPPORT_CONTEXT"
    assert handoff["selected_spec_lineage_complete"] is True
    assert handoff["selected_spec_memberships"] == [dict(selected_spec_index=0, status="CAPTURED",
                                                        candidate_ids=["candidate:0", "candidate:1"])]
    assert len(handoff["candidate_records"]) == 2
    # The two original finite candidates have identical coordinates, but their
    # distinct source support and owner occurrences must never be welded.
    for record, owner, face in zip(handoff["candidate_records"], ("owner:A", "owner:B"), ("face:0", "face:2")):
        assert record["bend_line_start"] == [0., 0., 0.] and record["bend_line_end"] == [5., 0., 0.]
        for flange in record["flanges"]:
            assert flange["owner_ids"] == [owner] and flange["context_face_ids"] == [face]
            assert flange["segments"][0]["context_face_ids"] == [face]
    assert handoff["candidate_records"][1]["flanges"][0]["source_vertex_indices"] == [3, 4]
    assert_no_root_authority(handoff)
    selected[0].cad_provenance["candidate_records"][0]["bend_line_end"][0] = 900.
    provenance["support_lineage"]["candidate_records"][1]["flange1_source_vertex_indices"].append(0)
    assert handoff["candidate_records"][0]["bend_line_end"] == [5., 0., 0.]
    assert handoff["candidate_records"][1]["flanges"][0]["source_vertex_indices"] == [3, 4]


def test_precoplanar_segments_retain_separate_faces_without_nominating_generic_flange_roots(api):
    provenance, _ = provenance_and_selected()
    record = candidate(rows1=(0, 1, 3, 4), rows2=(2, 5))
    record["flange1_segments"] = [dict(segment_id="pass:0:plane:0", source_vertex_indices=[0, 1]),
                                   dict(segment_id="pass:0:plane:1", source_vertex_indices=[3, 4])]
    record["flange2_segments"][0]["segment_id"] = "pass:0:plane:2"
    provenance["support_lineage"] = transport_ledger([record])
    selected = [SimpleNamespace(cad_provenance=copy.deepcopy(provenance["support_lineage"]))]
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    flange = handoff["candidate_records"][0]["flanges"][0]
    assert flange["owner_ids"] == ["owner:A", "owner:B"]
    assert flange["context_face_ids"] == ["face:0", "face:2"]
    assert [piece["context_face_ids"] for piece in flange["segments"]] == [["face:0"], ["face:2"]]
    assert_no_root_authority(handoff)


@pytest.mark.parametrize("bad", ["missing", "modified", "duplicate", "late_invalid", "ndarray", "schema_bool", "kind", "raw_count_bool"])
def test_missing_or_malformed_selected_strategy_retains_primary_ledger_without_selected_prefix(api, bad):
    provenance, selected = provenance_and_selected()
    if bad == "missing": selected = [SimpleNamespace()]
    elif bad == "modified": selected[0].cad_provenance["candidate_records"][0]["bend_line_end"][0] = 5.1
    elif bad == "duplicate": selected.append(copy.deepcopy(selected[0]))
    elif bad == "late_invalid": selected.append(SimpleNamespace())
    elif bad == "ndarray": selected[0].cad_provenance["candidate_records"][0]["bend_line_end"] = np.array([5., 0., 0.])
    elif bad == "schema_bool": selected[0].cad_provenance["schema_version"] = True
    elif bad == "kind": selected[0].cad_provenance["kind"] = "OTHER_LEDGER"
    else: selected[0].cad_provenance["raw_vertex_count"] = True
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "CAPTURED_SUPPORT_CONTEXT"
    assert len(handoff["candidate_records"]) == 2
    assert handoff["selected_spec_lineage_complete"] is False
    assert handoff["selected_spec_memberships"] == []
    assert "SELECTED_STRATEGY_LINEAGE_UNAVAILABLE" in handoff["reasons"]
    assert_no_root_authority(handoff)


@pytest.mark.parametrize("bad", ["duplicate_id", "late_nan", "late_zero_segment", "late_bool_row", "late_float_row",
                                "late_out_of_range", "late_unsorted", "late_duplicate_row", "segment_union",
                                "duplicate_segment", "missing_segments", "detection_pass_bool", "raw_count_bool"])
def test_malformed_primary_ledger_never_leaves_a_successful_candidate_prefix(api, bad):
    provenance, selected = provenance_and_selected()
    record = provenance["support_lineage"]["candidate_records"][-1]
    if bad == "duplicate_id": record["candidate_id"] = "candidate:0"
    elif bad == "late_nan": record["bend_line_end"][0] = float("nan")
    elif bad == "late_zero_segment": record["bend_line_end"] = record["bend_line_start"][:]
    elif bad == "late_bool_row": record["flange1_source_vertex_indices"] = [True]
    elif bad == "late_float_row": record["flange1_source_vertex_indices"] = [3.]
    elif bad == "late_out_of_range": record["flange1_source_vertex_indices"] = [6]
    elif bad == "late_unsorted": record["flange1_source_vertex_indices"] = [4, 3]
    elif bad == "late_duplicate_row": record["flange1_source_vertex_indices"] = [3, 3]
    elif bad == "segment_union": record["flange1_segments"][0]["source_vertex_indices"] = [3]
    elif bad == "duplicate_segment": record["flange1_segments"] *= 2
    elif bad == "missing_segments": record["flange1_segments"] = []
    elif bad == "detection_pass_bool": record["detection_pass"] = True
    else: provenance["support_lineage"]["raw_vertex_count"] = True
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []
    assert handoff["selected_spec_memberships"] == [] and handoff["selected_spec_lineage_complete"] is False
    assert_no_root_authority(handoff)


@pytest.mark.parametrize("field,value", [("MAX_RAW_VERTICES", 5), ("MAX_CANDIDATES", 1), ("MAX_SUPPORT_REFERENCES", 9)])
def test_handoff_budget_failure_discards_complete_earlier_candidate(api, monkeypatch, field, value):
    provenance, selected = provenance_and_selected()
    monkeypatch.setattr(api, field, value)
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []
    assert handoff["selected_spec_memberships"] == []


@pytest.mark.parametrize("bad", ["missing_preprocessing", "missing_rejected", "overlap_processed",
                                "overlap_rejected", "lost_raw_row", "missing_passes", "overcap_passes",
                                "late_pass_none", "late_pass_bool", "late_missing_row", "late_overlap",
                                "late_preprocess_rejected", "candidate_on_rejected", "candidate_wrong_pass",
                                "candidate_wrong_original_segment", "candidate_unknown_segment"])
def test_full_transport_partition_and_candidate_source_binding_fail_closed(api, bad):
    provenance, selected = provenance_and_selected()
    ledger = provenance["support_lineage"]
    final_pass, final_candidate = ledger["segmentation_passes"][-1], ledger["candidate_records"][-1]
    if bad == "missing_preprocessing": del ledger["processed_source_vertex_indices"]
    elif bad == "missing_rejected": del ledger["preprocessing_rejected_source_vertex_indices"]
    elif bad == "overlap_processed": ledger["processed_source_vertex_indices"].append([0])
    elif bad == "overlap_rejected": ledger["preprocessing_rejected_source_vertex_indices"] = [0]
    elif bad == "lost_raw_row": ledger["processed_source_vertex_indices"].pop()
    elif bad == "missing_passes": del ledger["segmentation_passes"]
    elif bad == "overcap_passes": ledger["segmentation_passes"] *= 2
    elif bad == "late_pass_none": ledger["segmentation_passes"][-1] = None
    elif bad == "late_pass_bool": final_pass["detection_pass"] = True
    elif bad == "late_missing_row": final_pass["unassigned_source_vertex_indices"] = [0, 1]
    elif bad == "late_overlap": final_pass["rejected_source_vertex_indices"] = [3]
    elif bad == "late_preprocess_rejected":
        ledger["processed_source_vertex_indices"].pop()
        ledger["preprocessing_rejected_source_vertex_indices"] = [5]
    elif bad == "candidate_on_rejected":
        final_pass["segments"].pop()
        final_pass["rejected_source_vertex_indices"] = [5]
    elif bad == "candidate_wrong_pass": final_candidate["detection_pass"] = 0
    elif bad == "candidate_wrong_original_segment":
        final_candidate["flange1_source_vertex_indices"] = [3]
        final_candidate["flange1_segments"][0]["source_vertex_indices"] = [3]
    else: final_candidate["flange1_segments"][0]["segment_id"] = "pass:1:plane:99"
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []
    assert handoff["selected_spec_memberships"] == []


def test_valid_voxel_group_membership_survives_as_all_original_rows(api):
    provenance, selected = provenance_and_selected()
    provenance["support_lineage"]["processed_source_vertex_indices"] = [[0, 1], [2], [3, 4], [5]]
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "CAPTURED_SUPPORT_CONTEXT"
    assert handoff["candidate_records"][0]["flanges"][0]["source_vertex_indices"] == [0, 1]
    assert handoff["candidate_records"][1]["flanges"][0]["source_vertex_indices"] == [3, 4]


def test_one_processed_voxel_cannot_be_split_between_two_segmentation_supports(api):
    provenance, _ = provenance_and_selected()
    record = candidate(rows1=(0,), rows2=(1, 2))
    ledger = transport_ledger([record])
    ledger["processed_source_vertex_indices"] = [[0, 1], [2], [3], [4], [5]]
    provenance["support_lineage"] = ledger
    selected = [SimpleNamespace(cad_provenance=copy.deepcopy(ledger))]
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []


@pytest.mark.parametrize("span", [[2, 6], [3, 7], [3, 5], [True, 6], [3., 6]])
def test_invalid_or_incomplete_raw_vertex_face_ranges_do_not_guess_by_geometry(api, span):
    provenance, selected = provenance_and_selected()
    provenance["face_records"][-1]["vertex_range"] = span
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []


def test_exact_detector_serialized_candidate_schema_is_accepted_without_importing_detector(api):
    classes = definitions(DETECTOR_SOURCE, {"SourcePlaneSupport", "SourceBendCandidate"}, {"dataclass": dataclass})
    segment, record_type = classes["SourcePlaneSupport"], classes["SourceBendCandidate"]
    original = record_type(candidate_id="candidate:0", detection_pass=0,
                           bend_line_start=(0., 0., 0.), bend_line_end=(5., 0., 0.),
                           flange1_source_vertex_indices=(0, 1), flange2_source_vertex_indices=(2,),
                           flange1_segments=(segment("pass:0:plane:0", (0, 1)),),
                           flange2_segments=(segment("pass:0:plane:1", (2,)),)).to_dict()
    assert original == candidate()
    provenance, _ = provenance_and_selected()
    provenance["support_lineage"] = transport_ledger([original])
    selected = [SimpleNamespace(cad_provenance=copy.deepcopy(provenance["support_lineage"]))]
    handoff = api.build_cad_provenance_handoff(provenance, selected)
    assert handoff["selected_spec_lineage_complete"] is True
    assert handoff["candidate_records"][0]["flanges"][0]["source_vertex_indices"] == [0, 1]


@pytest.fixture
def loader(api):
    native, ranges = native_and_ranges()
    # Two coincident triangles remain six source vertices in separate face blocks.
    vertices = np.tile([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], (2, 1))
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mesh = SimpleNamespace(vertices=vertices, faces=triangles,
                           tessellation_provenance=ranges, native_topology=native)
    calls = []
    def import_cad(path, *, deflection):
        calls.append((path, deflection))
        return SimpleNamespace(success=True, mesh=mesh, error=None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No actual native module or filesystem reader is allowed")
    stub = SimpleNamespace(io=SimpleNamespace(read_triangle_mesh=forbidden, read_point_cloud=forbidden))
    def guarded_import(name, *args, **kwargs):
        if name == "__future__": return builtins.__import__(name, *args, **kwargs)
        if name == "open3d": return stub
        raise AssertionError(f"Unexpected source import {name}")
    namespace = dict(np=np, copy=copy, Path=Path, CAD_BREP_EXTS={".step"}, import_cad_file=import_cad,
                     sanitize_native_topology=api.sanitize_native_topology,
                     __builtins__={**vars(builtins), "__import__": guarded_import})
    functions = definitions(PIPELINE_SOURCE, {"load_cad_geometry"}, namespace)
    return functions["load_cad_geometry"], mesh, calls


def test_exact_loader_only_transports_native_incidence_with_both_explicit_flags(loader):
    load, mesh, calls = loader
    old = load("invented.step")
    assert len(old) == 3 and old[2] == "cad_import"
    face_only = load("invented.step", include_provenance=True)
    assert len(face_only) == 4 and "native_topology" not in face_only[3]
    captured = load("invented.step", include_provenance=True, include_native_topology=True)
    np.testing.assert_array_equal(captured[0], mesh.vertices)
    np.testing.assert_array_equal(captured[1], mesh.faces)
    assert captured[3]["native_topology"]["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    assert captured[3]["native_topology"]["face_records"][2]["owner_id"] == "owner:B"
    assert_no_native_authority(captured[3]["native_topology"])
    mesh.native_topology["face_records"][2]["edge_uses"].clear()
    assert captured[3]["native_topology"]["face_records"][2]["edge_uses"]
    assert calls == [("invented.step", .05)] * 3


@pytest.mark.parametrize("flags", [dict(include_native_topology=True),
                                   dict(include_provenance=True, include_native_topology=1),
                                   dict(include_provenance=1, include_native_topology=True)])
def test_native_loader_invalid_flag_combinations_fail_before_import(loader, flags):
    load, _, calls = loader
    with pytest.raises(ValueError): load("invented.step", **flags)
    assert calls == []


@pytest.mark.parametrize("missing", ["attribute", "none", "malformed", "tessellation"])
def test_exact_loader_missing_or_malformed_native_capture_remains_unknown(loader, missing):
    load, mesh, _ = loader
    if missing == "attribute": del mesh.native_topology
    elif missing == "none": mesh.native_topology = None
    elif missing == "malformed": mesh.native_topology["face_records"][-1]["owner_id"] = "owner:missing"
    else: mesh.tessellation_provenance["face_records"][-1]["vertex_range"] = [3, 5]
    native = load("invented.step", include_provenance=True, include_native_topology=True)[3]["native_topology"]
    assert native["status"] == "UNAVAILABLE" and native["face_records"] == native["owner_records"] == []
    assert_no_native_authority(native)


@pytest.mark.parametrize("with_provenance", [False, True])
def test_exact_build_specs_propagates_primary_extractor_ledger_without_changing_default_call(api, with_provenance):
    provenance, expected = provenance_and_selected()
    primary_ledger = provenance.pop("support_lineage")
    calls = []
    class Extractor:
        def __init__(self, **kwargs): calls.append(("constructor", kwargs))
        def extract_from_mesh(self, vertices, triangles, **kwargs):
            calls.append(("extract", vertices, triangles, kwargs))
            self.last_source_lineage = primary_ledger
            return expected
    namespace = definitions(PIPELINE_SOURCE, {"_filter_kwargs", "_build_detector_cad_specs"},
                            dict(inspect=inspect, CADBendExtractor=Extractor))
    vertices, triangles = np.zeros((6, 3)), np.array([[0, 1, 2], [3, 4, 5]])
    kwargs = dict(cad_vertices=vertices, cad_triangles=triangles, tolerance_angle=1., tolerance_radius=.5,
                  cad_extractor_kwargs=None, cad_detector_kwargs=None, cad_detect_call_kwargs=None)
    if with_provenance: kwargs["cad_provenance"] = provenance
    result = namespace["_build_detector_cad_specs"](**kwargs)
    assert result is expected and calls[1][1] is vertices and calls[1][2] is triangles
    if with_provenance:
        assert calls[1][3] == {"provenance": provenance}
        assert provenance["support_lineage"] is primary_ledger
        handoff = api.build_cad_provenance_handoff(provenance, result)
        assert handoff["selected_spec_lineage_complete"] is True
    else:
        assert calls[1][3] == {} and "support_lineage" not in provenance


def test_active_run_ast_requests_native_handoff_passes_it_and_retains_it_in_details():
    tree = ast.parse(PIPELINE_SOURCE.read_text(encoding="utf-8"), filename=str(PIPELINE_SOURCE))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "run_progressive_bend_inspection")
    def calls(name):
        return [node for node in ast.walk(run) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == name]
    load, = calls("load_cad_geometry")
    keywords = {item.arg: item.value for item in load.keywords}
    assert isinstance(keywords["include_provenance"], ast.Constant) and keywords["include_provenance"].value is True
    assert isinstance(keywords["include_native_topology"], ast.Constant) and keywords["include_native_topology"].value is True
    for name in ("extract_cad_bend_specs_with_strategy", "BendInspectionRunDetails"):
        call, = calls(name)
        value = next(item.value for item in call.keywords if item.arg == "cad_provenance")
        assert isinstance(value, ast.Name) and value.id == "cad_provenance"
    handoff, = calls("build_cad_provenance_handoff")
    assert [node.id for node in handoff.args] == ["cad_provenance", "cad_bends"]
    assignment = next(node for node in ast.walk(run) if isinstance(node, ast.Assign) and node.value is handoff)
    target, = assignment.targets
    assert isinstance(target, ast.Subscript) and target.value.id == "cad_provenance" and target.slice.value == "handoff"
    assert handoff.lineno > calls("select_cad_bends_for_expected")[0].lineno
    assert handoff.lineno < calls("load_scan_points")[0].lineno


def test_run_details_retains_ledger_but_only_exports_compact_non_authoritative_summary(api):
    provenance, selected = provenance_and_selected()
    provenance["handoff"] = api.build_cad_provenance_handoff(provenance, selected)
    namespace = definitions(PIPELINE_SOURCE, {"BendInspectionRunDetails"},
                            dict(dataclass=dataclass, field=field, cad_provenance_summary=api.cad_provenance_summary))
    details = namespace["BendInspectionRunDetails"](
        cad_source="invented", cad_strategy="detector", cad_vertex_count=6, cad_triangle_count=2,
        scan_point_count=0, cad_bend_count=1, scan_bend_count=0, expected_bend_count=1,
        expected_progress_pct=0., overdetected_count=0, cad_provenance=provenance)
    assert details.cad_provenance is provenance
    summary = details.to_dict()["cad_provenance"]
    assert summary["precontraction_candidate_count"] == 2
    assert summary["selected_spec_lineage_complete"] is True
    assert summary["root_role_status"] == "NOT_NOMINATED" and summary["grouping_eligible"] is False
    assert summary["observed_state"] == "UNKNOWN"
    assert "candidate_records" not in summary and "support_lineage" not in summary and "face_records" not in summary
