#!/usr/bin/env python3
"""Generate the corpus-conditioned V5R3 LTCO prospective pilot.

V5R3 is authored only after the V63 implementation lock.  It deliberately
replaces the rejected rounded-carrier V5/V5R2 vocabulary with stamped-sheet
assemblies patterned on the real corpus: open trays, mounting ears, large
apertures, finite returns, local channels, close parallel bends, small creases,
and broad smooth rolls.  The public source is a triangle-connected fusion of
the organized range frames, not the hidden design mesh and not a point-only
PLY.  Detector execution remains forbidden until native visual preacceptance.
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
from scripts.generate_f1_lidar_like_synthetic_corpus import _rotation


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V5R3_CORPUS_CONDITIONED_LTCO_PROSPECTIVE"
CORPUS_KIND = "sherman_qc_f1_calibrated_raw_range_corpus_conditioned_ltco_v5r3"
CAPTURE_INDICES = (13,)
THICKNESS = 1.35
RANGE_MESH_PIXEL_STRIDE = 3

FAMILIES: tuple[dict[str, Any], ...] = (
    {"id":"RC01","name":"four_side_service_tray","radius":1.8,"challenge":"open four-sided tray with corner reliefs, four mounting ears, center aperture and unequal finite returns"},
    {"id":"RC02","name":"windowed_u_guard","radius":2.2,"challenge":"large rectangular access window, close parallel walls, short ear bends and one broad rolled negative"},
    {"id":"RC03","name":"stamped_local_hat_panel","radius":1.4,"challenge":"finite local hat channel, terminated close folds, small mounting creases and irregular carrier outline"},
    {"id":"RC04","name":"vented_roll_and_tab_guard","radius":2.0,"challenge":"dense vent slots, compound tabs, short returns and a broad smooth guard roll"},
    {"id":"RC05","name":"deep_box_with_hemmed_returns","radius":1.6,"challenge":"deep box section with two close return bends per side, corner relief gaps and mounting lugs"},
    {"id":"RC06","name":"asymmetric_instrument_bracket","radius":1.1,"challenge":"stepped silhouette, large and small apertures, mixed short creases, rounded facets and partial edge folds"},
)


def _poly(values: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _truth(rows: list[dict[str, Any]], line: tuple[np.ndarray, np.ndarray]) -> None:
    geom.truth_line(rows, *line)


def _sheet(builder: geom.Builder, outer: np.ndarray, holes: list[np.ndarray], *, spacing: float = 4.5) -> None:
    geom.add_sheet(builder, outer, holes, thickness=THICKNESS, spacing=spacing)


def _extruded_profile(
    builder: geom.Builder,
    x0: float,
    x1: float,
    yz: list[tuple[float, float]],
    *,
    nx: int = 30,
    truth_indices: tuple[int, ...] = (),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Add one continuous finite sheet feature from an authored YZ profile."""
    top = [
        [np.asarray((x, y, z), dtype=np.float64) for x in np.linspace(x0, x1, nx + 1)]
        for y, z in yz
    ]
    bottom = [
        [np.asarray((x, y, z - THICKNESS), dtype=np.float64) for x in np.linspace(x0, x1, nx + 1)]
        for y, z in yz
    ]
    builder._surface(top)
    builder._surface(bottom, reverse=True)
    builder.quad(top[0][0], top[0][-1], bottom[0][-1], bottom[0][0], nu=nx, nv=1)
    builder.quad(top[-1][-1], top[-1][0], bottom[-1][0], bottom[-1][-1], nu=nx, nv=1)
    for row in range(len(yz) - 1):
        builder.quad(top[row][0], top[row + 1][0], bottom[row + 1][0], bottom[row][0], nu=1, nv=1)
        builder.quad(top[row + 1][-1], top[row][-1], bottom[row][-1], bottom[row + 1][-1], nu=1, nv=1)
    lines = []
    for index in truth_indices:
        y, z = yz[index]
        lines.append((np.asarray((x0, y, z)), np.asarray((x1, y, z))))
    return lines


def case1():
    b, truth = geom.Builder(), []
    outer = _poly([
        (-108,-55),(-91,-55),(-91,-69),(-59,-69),(-59,-55),(54,-55),(54,-68),(88,-68),(88,-55),(108,-55),
        (108,55),(90,55),(90,68),(58,68),(58,55),(-55,55),(-55,67),(-89,67),(-89,55),(-108,55),
    ])
    holes = [geom._rounded_rect(-47,47,-31,31,8,n=7)]
    holes += [geom._circle(x,y,5.5,n=24) for x,y in ((-74,-61),(72,-61),(-72,60),(74,61))]
    holes += [geom._slot(-77,0,22,7,90), geom._slot(79,4,25,7,90)]
    _sheet(b, outer, holes, spacing=4.2)
    _truth(truth, geom.add_rounded_flange_y(b,-91,88,-55,radius=1.8,length=24,angle_deg=82,up=True,direction=-1,nx=34,na=9))
    _truth(truth, geom.add_rounded_flange_y(b,-89,90,55,radius=1.8,length=19,angle_deg=76,up=True,direction=1,nx=34,na=9))
    _truth(truth, geom.add_rounded_flange_x(b,-47,46,-108,radius=1.6,length=15,angle_deg=68,up=False,direction=-1))
    _truth(truth, geom.add_rounded_flange_x(b,-44,48,108,radius=1.6,length=17,angle_deg=72,up=True,direction=1))
    # Small finite mounting-ear creases, intentionally much shorter than the tray walls.
    _truth(truth, geom.add_rounded_flange_y(b,-88,-61,67,radius=0.9,length=7,angle_deg=42,up=True,direction=1,nx=8,na=6))
    _truth(truth, geom.add_rounded_flange_y(b,59,86,-68,radius=0.9,length=7,angle_deg=38,up=False,direction=-1,nx=8,na=6))
    return b.mesh(), truth, {"morphology":"open_tray_with_reliefs_and_ears","broad_roll_negative_count":0,"sheet_thickness_mm":THICKNESS}


def case2():
    b, truth = geom.Builder(), []
    outer = _poly([(-104,-66),(76,-66),(76,-55),(104,-55),(104,55),(78,55),(78,68),(-104,68)])
    holes = [geom._rounded_rect(-62,59,-39,39,9,n=8), geom._circle(88,-43,5), geom._circle(90,43,5)]
    holes += [geom._slot(-84,y,18,5,90,n=9) for y in (-34,0,34)]
    _sheet(b, outer, holes, spacing=4.6)
    _truth(truth, geom.add_rounded_flange_y(b,-96,68,-66,radius=2.2,length=29,angle_deg=84,up=True,direction=-1,nx=32,na=10))
    _truth(truth, geom.add_rounded_flange_y(b,-96,69,68,radius=2.2,length=27,angle_deg=80,up=True,direction=1,nx=32,na=10))
    _truth(truth, geom.add_rounded_flange_x(b,-53,52,-104,radius=1.2,length=11,angle_deg=55,up=False,direction=-1))
    # Deliberate broad-roll negative beside the true close parallel wall folds.
    geom.add_rounded_flange_x(b,-48,48,104,radius=17.0,length=5,angle_deg=86,up=True,direction=1)
    _truth(truth, geom.add_rounded_flange_y(b,80,101,55,radius=0.8,length=6,angle_deg=36,up=False,direction=1,nx=7,na=6))
    return b.mesh(), truth, {"morphology":"windowed_u_guard","broad_roll_negative_count":2,"sheet_thickness_mm":THICKNESS}


def case3():
    b, truth = geom.Builder(), []
    outer = _poly([(-112,-61),(-74,-61),(-69,-68),(-28,-68),(-23,-61),(109,-61),(109,45),(96,45),(96,62),(-112,62)])
    holes = [geom._circle(-88,-35,6), geom._circle(-87,37,6), geom._slot(78,-35,28,8,0), geom._slot(77,26,23,7,90)]
    # Leave the local channel footprint unperforated; adjacent apertures create realistic clutter.
    holes += [geom._circle(50,5,4.5), geom._slot(-48,-38,20,6,0)]
    _sheet(b, outer, holes, spacing=4.3)
    # One continuous finite stamped hat.  The rounded/faceted shoulders, top,
    # thickness skin and both termination caps are part of the same surface.
    hat = [(-24,0),(-21,0),(-19,0.8),(-17,3.2),(-14,10.5),(-11,14.5),(11,14.5),(14,10.5),(17,3.2),(19,0.8),(21,0),(24,0)]
    for line in _extruded_profile(b,-45,45,hat,nx=26,truth_indices=(2,4,6,8)):
        _truth(truth, line)
    _truth(truth, geom.add_rounded_flange_y(b,-104,-72,62,radius=0.8,length=7,angle_deg=34,up=False,direction=1,nx=8,na=5))
    _truth(truth, geom.add_rounded_flange_x(b,-53,37,109,radius=1.6,length=15,angle_deg=67,up=False,direction=1))
    return b.mesh(), truth, {"morphology":"finite_local_hat_and_mounting_creases","broad_roll_negative_count":0,"sheet_thickness_mm":THICKNESS}


def case4():
    b, truth = geom.Builder(), []
    outer = _poly([(-110,-58),(92,-58),(92,-45),(108,-45),(108,48),(82,48),(82,61),(-110,61)])
    holes: list[np.ndarray] = [geom._circle(-81,-35,6), geom._circle(-82,37,6), geom._slot(87,1,24,7,90)]
    for row in range(4):
        for col in range(7):
            holes.append(geom._slot(-43+17*col,-30+19*row,11,3.6,0,n=8))
    _sheet(b, outer, holes, spacing=4.0)
    # Large guard roll is explicitly not truth; the short tabs and bottom wall are.
    geom.add_rounded_flange_y(b,-101,76,61,radius=19.0,length=5,angle_deg=92,up=True,direction=1,nx=38,na=20)
    _truth(truth, geom.add_rounded_flange_y(b,-103,72,-58,radius=2.0,length=18,angle_deg=70,up=False,direction=-1,nx=33,na=9))
    _truth(truth, geom.add_rounded_flange_x(b,-39,-10,108,radius=1.0,length=9,angle_deg=44,up=True,direction=1))
    _truth(truth, geom.add_rounded_flange_x(b,13,43,108,radius=1.0,length=9,angle_deg=44,up=False,direction=1))
    _truth(truth, geom.add_rounded_flange_y(b,-104,-72,-58,radius=0.75,length=6,angle_deg=31,up=True,direction=-1,nx=8,na=5))
    return b.mesh(), truth, {"morphology":"vented_guard_with_broad_roll","broad_roll_negative_count":2,"sheet_thickness_mm":THICKNESS}


def case5():
    b, truth = geom.Builder(), []
    outer = _poly([(-101,-63),(101,-63),(101,52),(88,52),(88,66),(51,66),(51,57),(-51,57),(-51,66),(-89,66),(-89,52),(-101,52)])
    holes = [geom._circle(0,-4,19,n=36), geom._circle(-70,-38,5), geom._circle(70,-38,5)]
    holes += [geom._slot(-66,29,24,7,0), geom._slot(64,29,24,7,0)]
    _sheet(b, outer, holes, spacing=4.4)
    # Continuous deep walls with integral inward hem returns.  Earlier pilot
    # geometry placed these returns beside the wall as detached rails; this
    # profile makes the full material path explicit and connected.
    lower = [(-60,0),(-63,0),(-65,1),(-67,4),(-68,20),(-67,24),(-64,27),(-55,27),(-52,25)]
    upper = [(49,0),(52,0),(54,1),(56,4),(57,20),(56,24),(53,27),(44,27),(41,25)]
    for line in _extruded_profile(b,-91,91,lower,nx=38,truth_indices=(2,4,6,7)):
        _truth(truth, line)
    for line in _extruded_profile(b,-91,91,upper,nx=38,truth_indices=(2,4,6,7)):
        _truth(truth, line)
    _truth(truth, geom.add_rounded_flange_x(b,-47,45,-101,radius=1.4,length=18,angle_deg=73,up=True,direction=-1))
    _truth(truth, geom.add_rounded_flange_x(b,-45,45,101,radius=1.4,length=18,angle_deg=73,up=True,direction=1))
    return b.mesh(), truth, {"morphology":"deep_box_double_returns","broad_roll_negative_count":0,"sheet_thickness_mm":THICKNESS}


def case6():
    b, truth = geom.Builder(), []
    outer = _poly([(-116,-60),(-58,-60),(-51,-69),(17,-69),(24,-60),(104,-60),(104,-28),(114,-20),(114,43),(95,43),(88,58),(19,58),(14,66),(-67,66),(-73,57),(-116,57)])
    holes = [geom._rounded_rect(-48,35,-33,34,10,n=8), geom._circle(75,-31,7), geom._slot(80,22,31,8,90)]
    holes += [geom._circle(x,50,4.5,n=22) for x in (-48,-22,5,32)]
    _sheet(b, outer, holes, spacing=4.1)
    _truth(truth, geom.add_rounded_flange_y(b,-106,-58,-60,radius=0.8,length=8,angle_deg=39,up=True,direction=-1,nx=12,na=5))
    _truth(truth, geom.add_rounded_flange_y(b,28,96,-60,radius=1.1,length=14,angle_deg=61,up=False,direction=-1,nx=18,na=7))
    _truth(truth, geom.add_rounded_flange_y(b,-68,83,58,radius=1.5,length=17,angle_deg=69,up=True,direction=1,nx=32,na=8))
    _truth(truth, geom.add_rounded_flange_x(b,-51,49,-116,radius=1.2,length=13,angle_deg=57,up=False,direction=-1))
    # A shallow rounded facet is a negative, unlike the compact creases above.
    geom.add_rounded_flange_x(b,-18,38,114,radius=13.0,length=4,angle_deg=64,up=True,direction=1)
    return b.mesh(), truth, {"morphology":"asymmetric_stepped_instrument_bracket","broad_roll_negative_count":2,"sheet_thickness_mm":THICKNESS}


BUILDERS = (case1, case2, case3, case4, case5, case6)


def build_case(family: dict[str, Any], capture_index: int):
    number = int(family["id"][-2:])
    mesh, truth, metadata = BUILDERS[number - 1]()
    rng = np.random.default_rng(730_000 + 97 * number + capture_index)
    rotation = _rotation(740_000 + 53 * number + capture_index)
    translation = rng.uniform([-12,-9,-5], [12,9,5])
    vertices = np.asarray(mesh.vertices) @ rotation.T + translation
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    for row in truth:
        line = np.asarray([row["line_start"], row["line_end"]]) @ rotation.T + translation
        row["line_start"] = line[0].round(9).tolist()
        row["line_end"] = line[1].round(9).tolist()
    metadata.update({
        "family_id": family["id"], "family_name": family["name"], "capture": capture_index,
        "vertex_count": len(mesh.vertices), "triangle_count": len(mesh.triangles),
        "bend_count": len(truth), "finite_sheet_topology": True,
        "constant_cross_section": False, "finite_bend_radius": True,
        "feature_vocabulary": ["mounting_ears","corner_reliefs","large_apertures","short_returns","small_creases","rounded_facets"],
        "rotation": rotation.round(12).tolist(), "translation_mm": translation.round(9).tolist(),
    })
    return mesh, truth, metadata


def _frame_range_mesh(bundle: Path, manifest: dict[str, Any]) -> o3d.geometry.TriangleMesh:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    stride = RANGE_MESH_PIXEL_STRIDE
    for row in manifest["frames"]:
        depth = np.load(bundle / row["depth_file"], allow_pickle=False).astype(np.float64)
        validity = np.load(bundle / row["validity_file"], allow_pickle=False).astype(bool)
        K = np.asarray(row["K"], dtype=np.float64)
        T = np.asarray(row["T_world_from_camera"], dtype=np.float64)
        ys = np.arange(0, depth.shape[0], stride, dtype=np.int64)
        xs = np.arange(0, depth.shape[1], stride, dtype=np.int64)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        grid_valid = validity[grid_y, grid_x]
        grid_depth = depth[grid_y, grid_x]
        pixels = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)))
        camera = (np.linalg.inv(K) @ pixels.T).T * grid_depth.ravel()[:, None]
        world = camera @ T[:3, :3].T + T[:3, 3]
        world = world.reshape((*grid_x.shape, 3))
        index_grid = np.full(grid_valid.shape, -1, dtype=np.int64)
        valid_points = world[grid_valid]
        offset = len(vertices)
        index_grid[grid_valid] = offset + np.arange(len(valid_points), dtype=np.int64)
        vertices.extend(valid_points)

        neighbor_lengths = []
        for left, right, mask in (
            (world[:,:-1], world[:,1:], grid_valid[:,:-1] & grid_valid[:,1:]),
            (world[:-1], world[1:], grid_valid[:-1] & grid_valid[1:]),
        ):
            if np.any(mask):
                neighbor_lengths.append(np.linalg.norm(right-left, axis=2)[mask])
        typical = float(np.median(np.concatenate(neighbor_lengths))) if neighbor_lengths else 1.0
        edge_limit = max(4.0 * typical, 2.0)
        for iy in range(len(ys)-1):
            for ix in range(len(xs)-1):
                quad = (index_grid[iy,ix], index_grid[iy,ix+1], index_grid[iy+1,ix+1], index_grid[iy+1,ix])
                positions = (world[iy,ix], world[iy,ix+1], world[iy+1,ix+1], world[iy+1,ix])
                for local in ((0,1,2),(0,2,3)):
                    ids = tuple(int(quad[k]) for k in local)
                    if min(ids) < 0:
                        continue
                    tri = np.asarray([positions[k] for k in local])
                    lengths = np.linalg.norm(tri[[1,2,0]] - tri[[0,1,2]], axis=1)
                    if bool(np.all(lengths <= edge_limit)):
                        faces.append(ids)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.vertices) > base.capture.MAX_SOURCE_POINTS:
        raise RuntimeError(f"range mesh exceeds source-point cap: {len(mesh.vertices)}")
    if len(mesh.triangles) == 0:
        raise RuntimeError("range mesh has no connected triangles")
    return mesh


def _replace_public_sources_with_range_meshes(output_root: Path) -> None:
    public_manifest_path = output_root / "public_manifest.json"
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    private_manifest_path = output_root / "evaluator_only" / "private_manifest.json"
    private_manifest = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    all_meta_path = output_root / "evaluator_only" / "all_generation_metadata.json"
    all_meta = json.loads(all_meta_path.read_text(encoding="utf-8"))
    metadata_by_id = {row["opaque_case_id"]: row for row in all_meta["cases"]}
    private_by_id = {row["opaque_case_id"]: row for row in private_manifest["cases"]}
    for public_row in public_manifest["cases"]:
        case_id = public_row["opaque_case_id"]
        bundle = output_root / "public" / case_id
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mesh = _frame_range_mesh(bundle, manifest)
        source_path = bundle / "source.ply"
        if not o3d.io.write_triangle_mesh(str(source_path), mesh, write_ascii=False, compressed=False, write_vertex_normals=True):
            raise RuntimeError(f"failed to write connected public range mesh {source_path}")
        source_sha = base.capture._sha256(source_path)
        manifest["source_geometry"] = {
            "file": "source.ply", "sha256": source_sha,
            "kind": "organized_range_triangle_fusion_v1",
            "vertex_count": len(mesh.vertices), "triangle_count": len(mesh.triangles),
            "pixel_stride": RANGE_MESH_PIXEL_STRIDE,
        }
        base.capture._atomic_json(manifest_path, manifest)
        public_row.update({
            "manifest_sha256": base.capture._sha256(manifest_path), "source_sha256": source_sha,
            "source_point_count": len(mesh.vertices), "source_triangle_count": len(mesh.triangles),
        })
        meta = metadata_by_id[case_id]
        meta.update({
            "source_sha256": source_sha, "source_point_count": len(mesh.vertices),
            "source_triangle_count": len(mesh.triangles),
            "source_geometry_kind": "organized_range_triangle_fusion_v1",
            "source_mesh_pixel_stride": RANGE_MESH_PIXEL_STRIDE,
        })
        metadata_path = output_root / "evaluator_only" / case_id / "generation_metadata.json"
        base.capture._atomic_json(metadata_path, meta)
        private_by_id[case_id]["generation_metadata_sha256"] = base.capture._sha256(metadata_path)
    public_manifest.update({
        "source_geometry_kind": "organized_range_triangle_fusion_v1",
        "source_triangle_connectivity_required": True,
        "detector_execution_authorized": False,
        "visual_preacceptance_required": True,
    })
    base.capture._atomic_json(public_manifest_path, public_manifest)
    base.capture._atomic_json(private_manifest_path, private_manifest)
    base.capture._atomic_json(all_meta_path, all_meta)


def generate(output_root: Path, *, width: int, height: int, view_count: int, family_limit: int | None = None) -> dict[str, Any]:
    previous = (base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case)
    base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case = TOKEN, CORPUS_KIND, CAPTURE_INDICES, FAMILIES, build_case
    try:
        base.generate(output_root, width=width, height=height, view_count=view_count, family_limit=family_limit)
    finally:
        base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case = previous
    _replace_public_sources_with_range_meshes(output_root)
    renderer_path = output_root / "evaluator_only" / "renderer_manifest.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer.update({
        "generator_sha256": base.capture._sha256(Path(__file__).resolve()),
        "v63_configuration_sha256": "778bda993566612fbe671390106962c97b58bbb92c5a8f49c0c6e07cdd1501c4",
        "range_mesh_pixel_stride": RANGE_MESH_PIXEL_STRIDE,
        "source_geometry_kind": "organized_range_triangle_fusion_v1",
        "visual_acceptance_required_before_inference": True,
        "detector_execution_authorized": False,
        "v5_v5r2_reuse": False,
        "truth_opened_during_generation": False,
    })
    base.capture._atomic_json(renderer_path, renderer)
    return json.loads((output_root / "public_manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=base.capture.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=base.capture.DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=base.capture.DEFAULT_VIEW_COUNT)
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    report = generate(args.output_root.resolve(), width=args.width, height=args.height, view_count=args.view_count, family_limit=args.family_limit)
    print(json.dumps({"output_root":str(args.output_root.resolve()),"case_count":report["case_count"],"family_count":report["family_count"],"detector_execution_authorized":False,"token":TOKEN}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
