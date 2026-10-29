#!/usr/bin/env python3
"""Generate the fresh corpus-matched prospective lane for PSROC V64.

This lane is authored after the V5R3B corpus-similarity rejection.  It uses a
small near-neighbour vocabulary derived from public scan morphology only:
offset window brackets, shallow mounting trays, broad sparse panels, localized
lips, short finite tabs, small creases, and shallow rounded stamps.  It does
not reuse V5R3B geometry or outcomes.

Truth and design meshes are evaluator-only.  Public inference receives a
dense triangle-connected organized-range fusion from four deterministic views;
all twelve calibrated depth views remain available for truth-free visual QA.
Detector execution is forbidden until every native visual and the mesh-spacing
gate are manually accepted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v3 as base
import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v5r1_pilot as geom
import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v5r3 as old
from scripts.generate_f1_lidar_like_synthetic_corpus import _rotation


TOKEN = "SYNTHETIC_PSROC_V64_CORPUS_MATCHED_PROSPECTIVE_V1_TRUTH_UNOPENED"
CORPUS_KIND = "sherman_qc_f1_psroc_v64_corpus_matched_prospective_v1"
CAPTURE_INDICES = (17,)
THICKNESS = 1.25
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 576
DEFAULT_VIEW_COUNT = 12
TSDF_VOXEL_LENGTH_MM = 0.55
TSDF_TRUNCATION_MM = 2.20
TSDF_DEPTH_TRUNCATION_MM = 2_000.0

FAMILIES: tuple[dict[str, Any], ...] = (
    {"id": "CM01", "name": "offset_window_bracket", "radius": 1.5, "reference_class": "irregular multi-arm bracket", "challenge": "asymmetric long folds with one short fold and one small crease"},
    {"id": "CM02", "name": "shallow_mounting_ear_tray", "radius": 1.6, "reference_class": "four-wall mounting tray", "challenge": "ordinary finite-radius tray walls with relief corners and sparse ears"},
    {"id": "CM03", "name": "broad_sparse_access_panel", "radius": 1.25, "reference_class": "large sparse panel with one local flange", "challenge": "one long edge fold and one small isolated crease on a large sparse sheet"},
    {"id": "CM04", "name": "broad_panel_with_local_lips", "radius": 1.1, "reference_class": "large panel with asymmetric edge lips", "challenge": "two separated finite lips plus one rounded fold without a return box"},
    {"id": "CM05", "name": "narrow_tabbed_gusset", "radius": 1.4, "reference_class": "narrow irregular bracket with short tabs", "challenge": "one long fold, one short fold and one small crease around an irregular window"},
    {"id": "CM06", "name": "shallow_local_rib_cover", "radius": 1.4, "reference_class": "sparse cover with shallow rounded local stamp", "challenge": "two shallow rounded stamp shoulders plus one ordinary edge fold"},
)


def _poly(rows: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def _truth(rows: list[dict[str, Any]], line: tuple[np.ndarray, np.ndarray]) -> None:
    geom.truth_line(rows, *line)


def _sheet(builder: geom.Builder, outer: np.ndarray, holes: list[np.ndarray], spacing: float) -> None:
    geom.add_sheet(builder, outer, holes, thickness=THICKNESS, spacing=spacing)


def _flange_y(
    builder: geom.Builder,
    x0: float,
    x1: float,
    y0: float,
    *,
    radius: float,
    length: float,
    angle_deg: float,
    up: bool,
    direction: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx = max(12, int(math.ceil(abs(x1 - x0) / 2.0)))
    na = max(7, int(math.ceil(math.radians(angle_deg) * radius / 0.45)))
    return geom.add_rounded_flange_y(
        builder, x0, x1, y0,
        radius=radius, length=length, angle_deg=angle_deg,
        up=up, direction=direction, nx=nx, na=na,
    )


def _flange_x(
    builder: geom.Builder,
    y0: float,
    y1: float,
    x0: float,
    *,
    radius: float,
    length: float,
    angle_deg: float,
    up: bool,
    direction: float,
) -> tuple[np.ndarray, np.ndarray]:
    temp = geom.Builder()
    start, end = _flange_y(
        temp, y0, y1, x0,
        radius=radius, length=length, angle_deg=angle_deg,
        up=up, direction=direction,
    )
    vertices = np.asarray(temp.vertices, dtype=np.float64)[:, [1, 0, 2]]
    offset = len(builder.vertices)
    builder.vertices.extend(vertices)
    builder.faces.extend(tuple(offset + value for value in face) for face in temp.faces)
    return start[[1, 0, 2]], end[[1, 0, 2]]


def case1() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    """Irregular offset bracket: one large window, sparse mounting features."""
    b, truth = geom.Builder(), []
    outer = _poly([
        (-121, -43), (-93, -43), (-88, -52), (-48, -52), (-43, -44),
        (61, -44), (66, -35), (108, -35), (108, -9), (118, -4),
        (118, 37), (88, 37), (82, 49), (21, 49), (15, 58),
        (-65, 58), (-71, 49), (-112, 49), (-112, 21), (-121, 15),
    ])
    holes = [
        geom._rounded_rect(-62, 42, -27, 30, 8, n=9),
        geom._slot(91, 14, 22, 7, 90, n=10),
        geom._circle(-91, 31, 5.2, n=28),
    ]
    _sheet(b, outer, holes, spacing=3.8)
    _truth(truth, _flange_y(b, -105, 59, -43, radius=1.5, length=18, angle_deg=76, up=True, direction=-1))
    _truth(truth, _flange_y(b, -62, 79, 49, radius=1.8, length=20, angle_deg=83, up=True, direction=1))
    _truth(truth, _flange_x(b, -25, 27, -121, radius=1.2, length=13, angle_deg=61, up=False, direction=-1))
    _truth(truth, _flange_x(b, -17, 22, 118, radius=0.9, length=8, angle_deg=42, up=True, direction=1))
    return b.mesh(), truth, {"morphology": "irregular_offset_window_bracket", "truth_feature_classes": ["long_fold", "long_fold", "short_fold", "small_crease"]}


def case2() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    """Shallow tray with relief corners and three non-symmetric mounting ears."""
    b, truth = geom.Builder(), []
    outer = _poly([
        (-86, -60), (-66, -60), (-66, -72), (-35, -72), (-35, -60),
        (68, -60), (68, -70), (96, -70), (96, -57), (104, -52),
        (104, 52), (80, 52), (80, 65), (49, 65), (49, 52),
        (-86, 52),
    ])
    holes = [geom._circle(5, -2, 15, n=42), geom._circle(-50, -66, 5, n=26), geom._circle(82, -64, 5, n=26), geom._circle(64, 59, 5, n=26)]
    _sheet(b, outer, holes, spacing=3.5)
    _truth(truth, _flange_y(b, -83, 67, -60, radius=1.6, length=23, angle_deg=84, up=True, direction=-1))
    _truth(truth, _flange_y(b, -82, 77, 52, radius=1.6, length=21, angle_deg=80, up=True, direction=1))
    _truth(truth, _flange_x(b, -52, 49, -86, radius=1.5, length=19, angle_deg=82, up=True, direction=-1))
    _truth(truth, _flange_x(b, -47, 47, 104, radius=1.5, length=18, angle_deg=78, up=True, direction=1))
    return b.mesh(), truth, {"morphology": "shallow_mounting_ear_tray", "truth_feature_classes": ["tray_wall"] * 4}


def case3() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    """Broad sparse panel with one long flange and one short local crease."""
    b, truth = geom.Builder(), []
    outer = _poly([(-132, -79), (126, -79), (126, -57), (137, -52), (137, 72), (88, 72), (83, 80), (-132, 80)])
    holes = [
        geom._circle(91, 40, 9, n=34),
        geom._slot(-101, 44, 26, 5.5, 90, n=10),
        geom._slot(-82, 44, 26, 5.5, 90, n=10),
        geom._circle(-30, -8, 3.2, n=22),
        geom._circle(33, 19, 3.2, n=22),
    ]
    _sheet(b, outer, holes, spacing=4.6)
    _truth(truth, _flange_y(b, -121, 112, -79, radius=1.25, length=15, angle_deg=69, up=True, direction=-1))
    _truth(truth, _flange_y(b, -119, -86, 80, radius=0.75, length=6, angle_deg=31, up=False, direction=1))
    return b.mesh(), truth, {"morphology": "broad_sparse_access_panel", "truth_feature_classes": ["long_edge_fold", "small_crease"]}


def case4() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    """Broad panel with two asymmetric localized lips, not a full return box."""
    b, truth = geom.Builder(), []
    outer = _poly([(-128, -76), (113, -76), (113, -66), (130, -61), (130, 69), (58, 69), (52, 77), (-128, 77)])
    holes = [geom._circle(-93, -48, 5, n=26), geom._circle(97, 42, 5, n=26), geom._slot(-40, 38, 24, 6, 0, n=10)]
    _sheet(b, outer, holes, spacing=4.5)
    _truth(truth, _flange_y(b, -118, -16, -76, radius=1.1, length=11, angle_deg=57, up=True, direction=-1))
    _truth(truth, _flange_y(b, 18, 108, -76, radius=1.1, length=10, angle_deg=54, up=False, direction=-1))
    _truth(truth, _flange_y(b, -54, 47, 77, radius=2.8, length=13, angle_deg=66, up=True, direction=1))
    return b.mesh(), truth, {"morphology": "broad_panel_asymmetric_local_lips", "truth_feature_classes": ["finite_lip", "finite_lip", "rounded_fold"]}


def case5() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    """Narrow gusset with one window, a long fold, and two short tabs."""
    b, truth = geom.Builder(), []
    outer = _poly([(-118, -37), (-61, -37), (-55, -45), (28, -45), (34, -37), (111, -37), (111, -11), (121, -6), (121, 33), (73, 33), (67, 41), (-118, 41)])
    holes = [geom._rounded_rect(-45, 46, -22, 22, 7, n=8), geom._circle(92, 12, 5.5, n=28)]
    _sheet(b, outer, holes, spacing=3.9)
    _truth(truth, _flange_y(b, -105, 70, 41, radius=1.4, length=16, angle_deg=71, up=True, direction=1))
    _truth(truth, _flange_y(b, -108, -72, -37, radius=0.8, length=7, angle_deg=38, up=False, direction=-1))
    _truth(truth, _flange_y(b, 44, 99, -37, radius=0.9, length=9, angle_deg=46, up=True, direction=-1))
    return b.mesh(), truth, {"morphology": "narrow_tabbed_gusset", "truth_feature_classes": ["long_fold", "small_crease", "short_fold"]}


def case6() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    """Sparse cover with a shallow finite rounded stamp and one edge fold."""
    b, truth = geom.Builder(), []
    outer = _poly([(-112, -66), (96, -66), (96, -55), (112, -49), (112, 55), (76, 55), (69, 65), (-112, 65)])
    holes = [geom._circle(-82, -38, 5, n=26), geom._circle(82, 34, 6, n=28), geom._slot(66, -31, 22, 6, 90, n=10)]
    _sheet(b, outer, holes, spacing=4.0)
    # A single, shallow local stamped channel.  The two shoulders are the only
    # truth lines; the rounded crown is deliberately not a separate bend.
    profile = [(-17, 0), (-13, 0), (-10, 0.8), (-7, 3.1), (-4, 5.2), (4, 5.2), (7, 3.1), (10, 0.8), (13, 0), (17, 0)]
    for line in old._extruded_profile(b, -37, 42, profile, nx=42, truth_indices=(2, 7)):
        _truth(truth, line)
    _truth(truth, _flange_y(b, -101, 63, -66, radius=1.4, length=14, angle_deg=63, up=True, direction=-1))
    return b.mesh(), truth, {"morphology": "sparse_cover_shallow_local_rib", "truth_feature_classes": ["rounded_facet_shoulder", "rounded_facet_shoulder", "edge_fold"]}


BUILDERS = (case1, case2, case3, case4, case5, case6)


def build_case(family: dict[str, Any], capture_index: int):
    number = int(family["id"][-2:])
    mesh, truth, metadata = BUILDERS[number - 1]()
    rng = np.random.default_rng(910_000 + 97 * number + capture_index)
    rotation = _rotation(920_000 + 53 * number + capture_index)
    translation = rng.uniform([-10, -8, -4], [10, 8, 4])
    vertices = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    for row in truth:
        line = np.asarray([row["line_start"], row["line_end"]], dtype=np.float64) @ rotation.T + translation
        row["line_start"] = line[0].round(9).tolist()
        row["line_end"] = line[1].round(9).tolist()
    metadata.update({
        "family_id": family["id"],
        "family_name": family["name"],
        "reference_class": family["reference_class"],
        "capture": capture_index,
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "bend_count": len(truth),
        "finite_sheet_topology": True,
        "finite_bend_radius": True,
        "dense_repeated_vent_pattern": False,
        "broad_roll_negative": False,
        "v5r3b_geometry_reused": False,
        "rotation": rotation.round(12).tolist(),
        "translation_mm": translation.round(9).tolist(),
    })
    return mesh, truth, metadata


def _frame_range_mesh(bundle: Path, manifest: dict[str, Any]) -> tuple[o3d.geometry.TriangleMesh, list[int]]:
    """Fuse every calibrated view into one bounded TSDF surface.

    Stacking independent view meshes creates nearly coincident ghost skins and
    artificial parallel ridges.  TSDF integration yields one reconstructed
    surface, matching the corpus' fused-scan topology more closely.  No truth,
    family ID, or detector output enters the fusion or decimation.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL_LENGTH_MM,
        sdf_trunc=TSDF_TRUNCATION_MM,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    used_ordinals: list[int] = []
    for row in manifest["frames"]:
        depth = np.load(bundle / row["depth_file"], allow_pickle=False).astype(np.float32)
        validity = np.load(bundle / row["validity_file"], allow_pickle=False).astype(bool)
        depth[~validity] = 0.0
        color = np.zeros((*depth.shape, 3), dtype=np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=TSDF_DEPTH_TRUNCATION_MM,
            convert_rgb_to_intensity=False,
        )
        K = np.asarray(row["K"], dtype=np.float64)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth.shape[1], depth.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        T_world_from_camera = np.asarray(row["T_world_from_camera"], dtype=np.float64)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_world_from_camera))
        used_ordinals.append(int(row["ordinal"]))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError("TSDF fusion produced no triangles")
    while len(mesh.vertices) > base.capture.MAX_SOURCE_POINTS:
        target = max(
            1_000,
            int(len(mesh.triangles) * base.capture.MAX_SOURCE_POINTS / len(mesh.vertices) * 0.92),
        )
        reduced = mesh.simplify_quadric_decimation(target)
        if len(reduced.vertices) >= len(mesh.vertices):
            raise RuntimeError("TSDF decimation failed to reduce the source mesh")
        mesh = reduced
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(mesh.triangles) or len(mesh.vertices) > base.capture.MAX_SOURCE_POINTS:
        raise RuntimeError("invalid bounded public TSDF mesh")
    return mesh, used_ordinals


def _edge_quantiles(mesh: o3d.geometry.TriangleMesh) -> list[float]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) > 300_000:
        triangles = triangles[np.linspace(0, len(triangles) - 1, 300_000, dtype=np.int64)]
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 1e-8)]
    return np.percentile(lengths, (10, 50, 90)).round(6).tolist()


def _replace_public_sources(output_root: Path) -> None:
    public_path = output_root / "public_manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private_path = output_root / "evaluator_only" / "private_manifest.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    all_meta_path = output_root / "evaluator_only" / "all_generation_metadata.json"
    all_meta = json.loads(all_meta_path.read_text(encoding="utf-8"))
    meta_by_id = {row["opaque_case_id"]: row for row in all_meta["cases"]}
    private_by_id = {row["opaque_case_id"]: row for row in private["cases"]}
    for public_row in public["cases"]:
        case_id = public_row["opaque_case_id"]
        bundle = output_root / "public" / case_id
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mesh, ordinals = _frame_range_mesh(bundle, manifest)
        quantiles = _edge_quantiles(mesh)
        source_path = bundle / "source.ply"
        if not o3d.io.write_triangle_mesh(str(source_path), mesh, write_ascii=False, compressed=False, write_vertex_normals=True):
            raise RuntimeError(f"failed to write {source_path}")
        source_sha = base.capture._sha256(source_path)
        # Preserve the frozen capture-bundle v1 schema exactly.  Mesh-specific
        # audit fields live only in the corpus-level public row and private
        # generation metadata, never inside the strict per-bundle manifest.
        manifest["source_geometry"] = {"file": "source.ply", "sha256": source_sha}
        base.capture._atomic_json(manifest_path, manifest)
        public_row.update({
            "manifest_sha256": base.capture._sha256(manifest_path),
            "source_sha256": source_sha,
            "source_point_count": len(mesh.vertices),
            "source_triangle_count": len(mesh.triangles),
            "source_edge_length_mm_p10_p50_p90": quantiles,
            "source_view_ordinals": ordinals,
        })
        metadata = meta_by_id[case_id]
        metadata.update({
            "source_sha256": source_sha,
            "source_point_count": len(mesh.vertices),
            "source_triangle_count": len(mesh.triangles),
            "source_geometry_kind": "calibrated_tsdf_triangle_fusion_v1",
            "tsdf_voxel_length_mm": TSDF_VOXEL_LENGTH_MM,
            "tsdf_truncation_mm": TSDF_TRUNCATION_MM,
            "source_view_ordinals": ordinals,
            "source_edge_length_mm_p10_p50_p90": quantiles,
        })
        metadata_path = output_root / "evaluator_only" / case_id / "generation_metadata.json"
        base.capture._atomic_json(metadata_path, metadata)
        private_by_id[case_id]["generation_metadata_sha256"] = base.capture._sha256(metadata_path)
    public.update({
        "source_geometry_kind": "calibrated_tsdf_triangle_fusion_v1",
        "source_triangle_connectivity_required": True,
        "detector_execution_authorized": False,
        "visual_preacceptance_required": True,
        "truth_opened": False,
        "v5r3b_reused": False,
    })
    base.capture._atomic_json(public_path, public)
    base.capture._atomic_json(private_path, private)
    base.capture._atomic_json(all_meta_path, all_meta)


def generate(output_root: Path, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, view_count: int = DEFAULT_VIEW_COUNT, family_limit: int | None = None):
    previous = (base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case)
    base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case = TOKEN, CORPUS_KIND, CAPTURE_INDICES, FAMILIES, build_case
    try:
        base.generate(output_root, width=width, height=height, view_count=view_count, family_limit=family_limit)
    finally:
        base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case = previous
    _replace_public_sources(output_root)
    renderer_path = output_root / "evaluator_only" / "renderer_manifest.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer.update({
        "generator_sha256": base.capture._sha256(Path(__file__).resolve()),
        "tsdf_voxel_length_mm": TSDF_VOXEL_LENGTH_MM,
        "tsdf_truncation_mm": TSDF_TRUNCATION_MM,
        "tsdf_depth_truncation_mm": TSDF_DEPTH_TRUNCATION_MM,
        "source_view_ordinals": list(range(view_count)),
        "source_geometry_kind": "calibrated_tsdf_triangle_fusion_v1",
        "corpus_similarity_rejection_reference": "docs/research/2026-08-15_f1_v5r3b_posthoc_corpus_similarity_rejection.md",
        "visual_acceptance_required_before_inference": True,
        "detector_execution_authorized": False,
        "truth_opened_during_generation": False,
        "v5r3b_geometry_reused": False,
    })
    base.capture._atomic_json(renderer_path, renderer)
    return json.loads((output_root / "public_manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=DEFAULT_VIEW_COUNT)
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    report = generate(args.output_root.resolve(), width=args.width, height=args.height, view_count=args.view_count, family_limit=args.family_limit)
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "case_count": report["case_count"],
        "family_count": report["family_count"],
        "detector_execution_authorized": False,
        "truth_opened": False,
        "token": TOKEN,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
