"""Connected exact-source CAD transport with invented native/geometry doubles.

No backend package or native library is imported. The chain executes the actual
tessellator/capture, loader, extraction adapter, detector transport stages,
deduplication, rolled contraction, handoff and compact diagnostics. Geometry
proposals are stipulated, not detected physical bends. Existing legacy selected
contraction remains unchanged; original finite/support records must survive it.
"""
from __future__ import annotations

import builtins
import copy
from dataclasses import dataclass, field
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent


def harness(filename, name):
    """Load only a specifically reviewed synthetic test harness by exact path."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers():
    return SimpleNamespace(
        native=harness("test_cad_native_topology_capture.py", "cad_e2e_native_harness"),
        support=harness("test_cad_support_lineage.py", "cad_e2e_support_harness"),
        handoff=harness("test_cad_provenance_handoff.py", "cad_e2e_handoff_harness"),
    )


def invented_mesh(helpers, *, coincident=False, native_fault=False):
    native = helpers.native
    coordinates = {}
    owners = []
    for owner_index in range(2):
        edge = native.Shape("EDGE", f"owner-{owner_index}-shared")
        reverse = copy.copy(edge)
        reverse.orientation = "REVERSED"
        faces = []
        for local_index in range(2):
            index = owner_index * 2 + local_index
            token = f"face-{index}"
            x = 0. if coincident else index * 20.
            coordinates[token] = [(x, 0., 0.), (x + 2., 0., 0.), (x, 2., 0.)]
            faces.append(native.Shape("FACE", token, [edge if local_index == 0 else reverse]))
        owners.append(native.Shape("SOLID", f"owner-{owner_index}", [
            native.Shape("SHELL", f"shell-{owner_index}", faces)]))
    shape = native.Shape("COMPOUND", "assembly", owners)
    importer = native.tessellation_importer(native.source_namespace())

    def triangulation(face, location):
        xyz = coordinates[face.token]
        return SimpleNamespace(NbNodes=lambda: 3, NbTriangles=lambda: 1,
                               Node=lambda index: native.Point(xyz[index - 1]),
                               Triangle=lambda index: SimpleNamespace(Get=lambda: (1, 2, 3)))

    importer._BRep_Tool.Triangulation = triangulation
    if native_fault:
        def broken_handle(original):
            copied = copy.copy(original)
            copied.IsSame = lambda other: 1  # Nonboolean kernel result cannot establish identity.
            return copied
        importer._TopoDS_Shape = broken_handle
    mesh = importer._tessellate(shape)
    expected_vertices = np.array([point for index in range(4) for point in coordinates[f"face-{index}"]])
    expected_faces = np.arange(12).reshape(4, 3)
    np.testing.assert_array_equal(mesh.vertices, expected_vertices)
    np.testing.assert_array_equal(mesh.faces, expected_faces)
    return mesh


def identity_preprocessing(ns, *, late_bad_trace=False):
    calls = []
    class Cloud:
        def __init__(self, points=None, normals=None):
            self.points = np.empty((0, 3)) if points is None else np.array(points, copy=True)
            self.normals = np.empty((0, 3)) if normals is None else np.array(normals, copy=True)

        def has_normals(self):
            return len(self.points) > 0 and len(self.normals) == len(self.points)

        def voxel_down_sample(self, **kwargs):
            calls.append("legacy_voxel")
            return Cloud(self.points, self.normals)

        def voxel_down_sample_and_trace(self, **kwargs):
            calls.append("trace")
            assert kwargs["approximate_class"] is False
            np.testing.assert_array_equal(kwargs["min_bound"], self.points.min(axis=0) - .15)
            np.testing.assert_array_equal(kwargs["max_bound"], self.points.max(axis=0) + .15)
            memberships = [[index] for index in range(len(self.points))]
            if late_bad_trace:
                memberships[-1] = [0]  # First eleven rows are valid, final raw row is lost/reused.
            return Cloud(self.points, self.normals), np.empty((0, 8)), memberships

        def remove_statistical_outlier(self, **kwargs):
            calls.append("outliers")
            return Cloud(self.points, self.normals), list(range(len(self.points)))

        def estimate_normals(self, **kwargs):
            calls.append("normals")
            self.normals = np.tile([0., 0., 1.], (len(self.points), 1))

    ns["fake_open3d"] = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=Cloud, KDTreeSearchParamHybrid=lambda **kwargs: kwargs),
        utility=SimpleNamespace(Vector3dVector=lambda values: np.array(values, copy=True)),
    )
    return calls


def connected_chain(helpers, *, offsets=(0., 1.), coincident=False, native_fault=False,
                    late_bad_trace=False, retain_provenance=True):
    api = helpers.handoff.api.__wrapped__()
    mesh = invented_mesh(helpers, coincident=coincident, native_fault=native_fault)
    ns = helpers.support.ns.__wrapped__()
    preprocessing = identity_preprocessing(ns, late_bad_trace=late_bad_trace)
    detectors, dedup_sizes, rolled_sizes, coplanar_sizes = [], [], [], []
    real_detector = ns["BendDetector"]

    def configured_detector(**kwargs):
        kwargs.update(min_plane_points=3, min_plane_area=0, adaptive_threshold=False,
                      merge_coplanar_planes=True)
        detector = real_detector(**kwargs)
        detector._compute_ransac_iterations = lambda count: 1
        detector._ransac_plane_fit = lambda rows, **kw: (
            np.array([0., 0., 1.]), 0., np.arange(len(rows)) < 3)
        detector._estimate_area = lambda rows: 10.
        detector._extract_boundary = lambda rows: rows.copy()
        detector._planes_coplanar_and_adjacent = lambda first, second: False
        detector._find_adjacent_planes = lambda planes: [(planes[0], planes[1]), (planes[2], planes[3])]
        detector._analyze_bend = lambda first, second, points, bend_id: helpers.support.bend(
            ns, first, second, offset=offsets[int(bend_id[1:]) - 1],
            confidence=.9 - .1 * (int(bend_id[1:]) - 1))
        actual_dedup, actual_coplanar = detector._merge_multiscale_bends, detector._merge_coplanar_planes
        def dedup(bends, **kw):
            result = actual_dedup(bends, **kw)
            dedup_sizes.append((len(bends), len(result)))
            return result
        def coplanar(planes):
            result = actual_coplanar(planes)
            coplanar_sizes.append((len(planes), len(result)))
            return result
        detector._merge_multiscale_bends, detector._merge_coplanar_planes = dedup, coplanar
        detectors.append(detector)
        return detector

    ns["BendDetector"] = configured_detector
    actual_rolled = ns["CADBendExtractor"]._merge_rolled_specs
    def rolled(extractor, specs):
        result = actual_rolled(extractor, specs)
        rolled_sizes.append((len(specs), len(result)))
        return result
    ns["CADBendExtractor"]._merge_rolled_specs = rolled
    imported = []
    def import_cad(path, *, deflection):
        assert path == "invented-connected.step"
        imported.append((path, deflection))
        return SimpleNamespace(success=True, mesh=mesh, error=None)
    def guarded_import(name, *args, **kwargs):
        if name == "__future__": return builtins.__import__(name, *args, **kwargs)
        if name == "open3d": return ns["fake_open3d"]
        raise AssertionError("Native/package import forbidden: " + name)
    pipeline = helpers.handoff.definitions(
        helpers.handoff.PIPELINE_SOURCE,
        {"load_cad_geometry", "_filter_kwargs", "_build_detector_cad_specs", "BendInspectionRunDetails"},
        dict(__name__=__name__, np=np, copy=copy, inspect=inspect, Path=Path, dataclass=dataclass, field=field,
             CAD_BREP_EXTS={".step"}, import_cad_file=import_cad, CADBendExtractor=ns["CADBendExtractor"],
             sanitize_native_topology=api.sanitize_native_topology, cad_provenance_summary=api.cad_provenance_summary,
             __builtins__={**vars(builtins), "__import__": guarded_import}),
    )
    flags = {"include_provenance": True, "include_native_topology": True} if retain_provenance else {}
    loaded = pipeline["load_cad_geometry"]("invented-connected.step", **flags)
    vertices, triangles, source = loaded[:3]
    provenance = loaded[3] if retain_provenance else None
    specs = pipeline["_build_detector_cad_specs"](
        cad_vertices=vertices, cad_triangles=triangles, tolerance_angle=1., tolerance_radius=.5,
        cad_extractor_kwargs={"detect_call_kwargs": {"preprocess": not coincident, "multiscale": False}},
        cad_detector_kwargs=None, cad_detect_call_kwargs=None, cad_provenance=provenance,
    )
    if retain_provenance:
        provenance["handoff"] = api.build_cad_provenance_handoff(provenance, specs)
    details = pipeline["BendInspectionRunDetails"](
        cad_source=source, cad_strategy="invented_detector_transport", cad_vertex_count=len(vertices),
        cad_triangle_count=len(triangles), scan_point_count=0, cad_bend_count=len(specs),
        scan_bend_count=0, expected_bend_count=len(specs), expected_progress_pct=0., overdetected_count=0,
        cad_vertices=vertices, cad_triangles=triangles, cad_provenance=provenance,
    )
    assert imported == [("invented-connected.step", .05)]
    return SimpleNamespace(mesh=mesh, vertices=vertices, triangles=triangles, specs=specs,
                           provenance=provenance, details=details, detector=detectors[0],
                           preprocessing=preprocessing, dedup_sizes=dedup_sizes,
                           rolled_sizes=rolled_sizes, coplanar_sizes=coplanar_sizes, imports=ns["imports"])


def assert_geometry_unchanged(actual, baseline):
    for name in ("vertices", "triangles"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(baseline, name))
    assert [spec.to_dict() for spec in actual.specs] == [spec.to_dict() for spec in baseline.specs]
    for first, second in zip(actual.specs, baseline.specs):
        np.testing.assert_array_equal(first.bend_line_start, second.bend_line_start)
        np.testing.assert_array_equal(first.bend_line_end, second.bend_line_end)


@pytest.mark.parametrize("offsets", [(0., 1.), (0., 11.)])
@pytest.mark.parametrize("coincident", [False, True])
def test_exact_connected_chain_preserves_two_native_contexts_despite_legacy_contraction(helpers, offsets, coincident):
    actual = connected_chain(helpers, offsets=offsets, coincident=coincident)
    baseline = connected_chain(helpers, offsets=offsets, coincident=coincident, retain_provenance=False)
    assert_geometry_unchanged(actual, baseline)
    assert len(actual.specs) == 1
    assert actual.coplanar_sizes == [(4, 4)]  # Actual coplanar traversal; explicit no-merge geometry control.
    assert actual.dedup_sizes == [(2, 1 if offsets[1] == 1. else 2)]
    assert actual.rolled_sizes == ([(1, 1)] if offsets[1] == 1. else [(2, 1)])
    assert len(actual.vertices) == 12 and len(actual.triangles) == 4
    if coincident:
        assert actual.preprocessing == [] and actual.imports == []  # No false identity voxel trace for coincident cells.
        np.testing.assert_array_equal(actual.vertices[:3], actual.vertices[9:])
    else:
        assert actual.preprocessing == ["legacy_voxel", "trace", "outliers", "normals"]
        assert actual.imports == ["open3d"]
    provenance, handoff = actual.provenance, actual.provenance["handoff"]
    assert provenance["native_topology"]["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    native_faces = provenance["native_topology"]["face_records"]
    assert [face["owner_id"] for face in native_faces] == ["owner:0", "owner:0", "owner:1", "owner:1"]
    assert native_faces[0]["edge_uses"][0]["edge_id"] == native_faces[1]["edge_uses"][0]["edge_id"]
    assert native_faces[0]["edge_uses"][0]["edge_id"] != native_faces[2]["edge_uses"][0]["edge_id"]
    ledger = provenance["support_lineage"]
    assert ledger["status"] == "COMPLETE"
    assert ledger["processed_source_vertex_indices"] == tuple((index,) for index in range(12))
    assert ledger["preprocessing_rejected_source_vertex_indices"] == ()
    assert len(ledger["segmentation_passes"]) == 1
    assert handoff["status"] == "CAPTURED_SUPPORT_CONTEXT" and handoff["selected_spec_lineage_complete"] is True
    assert handoff["selected_spec_memberships"][0]["candidate_ids"] == ["candidate:0", "candidate:1"]
    assert len(handoff["candidate_records"]) == 2
    for index, record in enumerate(handoff["candidate_records"]):
        assert record["bend_line_start"] == [offsets[index], 0., 0.]
        assert record["bend_line_end"] == [offsets[index] + 4., 0., 0.]
        for flange_index, flange in enumerate(record["flanges"]):
            face_index = 2 * index + flange_index
            assert flange["owner_ids"] == [f"owner:{index}"]
            assert flange["context_face_ids"] == [f"face:{face_index}"]
            assert flange["source_vertex_indices"] == list(range(3 * face_index, 3 * face_index + 3))
        assert record["root_face_ids"] == []
    helpers.handoff.assert_no_root_authority(handoff)
    summary = actual.details.to_dict()["cad_provenance"]
    assert actual.details.cad_provenance is provenance
    assert summary["precontraction_candidate_count"] == 2 and summary["grouping_eligible"] is False
    assert "candidate_records" not in summary and "support_lineage" not in summary


@pytest.mark.parametrize("fault", ["native_identity", "late_trace"])
def test_connected_late_failure_clears_handoff_without_changing_legacy_geometry(helpers, fault):
    kwargs = dict(offsets=(0., 11.), native_fault=fault == "native_identity", late_bad_trace=fault == "late_trace")
    actual = connected_chain(helpers, **kwargs)
    baseline = connected_chain(helpers, retain_provenance=False, **kwargs)
    assert_geometry_unchanged(actual, baseline)
    assert len(actual.specs) == 1
    handoff = actual.provenance["handoff"]
    assert handoff["status"] == "UNAVAILABLE" and handoff["candidate_records"] == []
    assert handoff["selected_spec_memberships"] == [] and handoff["selected_spec_lineage_complete"] is False
    if fault == "native_identity":
        assert actual.provenance["native_topology"]["status"] == "UNAVAILABLE"
        assert actual.provenance["support_lineage"]["status"] == "COMPLETE"
    else:
        assert actual.provenance["support_lineage"]["status"] == "UNAVAILABLE"
        assert actual.provenance["support_lineage"]["candidate_records"] == []
    assert actual.details.to_dict()["cad_provenance"]["precontraction_candidate_count"] == 0
    helpers.handoff.assert_no_root_authority(handoff)
