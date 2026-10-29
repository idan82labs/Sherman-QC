#!/usr/bin/env python3
"""Generate a visually gated finite-sheet-metal calibrated capture pilot.

V5 deliberately abandons the constant-cross-section extrusion used by V1--V4.
Every case is a finite assembly of sheet patches with local fold endpoints,
unequal flanges, free edges, and irregular negative geometry.  Smooth rolled
regions are present but are *not* labelled as manufacturing bend lines.

This pilot is morphology-validation evidence only.  It must be visually
accepted before it is enlarged or passed to a detector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v3 as base
from scripts.generate_f1_lidar_like_synthetic_corpus import _rotation


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V5_FINITE_PART_PILOT_NON_CONFIRMATORY"
CORPUS_KIND = "sherman_qc_f1_calibrated_raw_range_finite_part_pilot_v5"
CAPTURE_INDICES = (10,)

FAMILIES: tuple[dict[str, Any], ...] = (
    {"id": "FP01", "name": "asymmetric_mounting_bracket", "radius": None, "challenge": "long and short boundary flanges, unequal end tab, two irregular cutouts"},
    {"id": "FP02", "name": "three_sided_shallow_tray", "radius": None, "challenge": "finite tray walls, interrupted front ears, corner reliefs and perforations"},
    {"id": "FP03", "name": "offset_service_cover", "radius": None, "challenge": "unequal stepped cover with short side returns and non-repeating apertures"},
    {"id": "FP04", "name": "local_doghouse_panel", "radius": None, "challenge": "local raised enclosure in a finite carrier plate with terminated bend lines"},
    {"id": "FP05", "name": "rolled_shield_with_sharp_tabs", "radius": None, "challenge": "broad rounded blend as negative control beside two true local tab bends"},
    {"id": "FP06", "name": "perforated_return_bracket", "radius": None, "challenge": "small local crease, partial return, slots, notch and unequal free edges"},
)


class MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        self.faces: list[tuple[int, int, int]] = []

    def quad(
        self,
        p00: tuple[float, float, float],
        p10: tuple[float, float, float],
        p11: tuple[float, float, float],
        p01: tuple[float, float, float],
        *,
        nu: int = 12,
        nv: int = 8,
        keep: Callable[[float, float, np.ndarray], bool] | None = None,
    ) -> None:
        """Add a bilinear finite patch; `keep` can remove complete cells."""
        corners = [np.asarray(p, dtype=np.float64) for p in (p00, p10, p11, p01)]
        offset = len(self.vertices)
        for j in range(nv + 1):
            v = j / nv
            for i in range(nu + 1):
                u = i / nu
                point = (
                    (1 - u) * (1 - v) * corners[0]
                    + u * (1 - v) * corners[1]
                    + u * v * corners[2]
                    + (1 - u) * v * corners[3]
                )
                self.vertices.append(point)
        for j in range(nv):
            for i in range(nu):
                u = (i + 0.5) / nu
                v = (j + 0.5) / nv
                ids = (
                    offset + j * (nu + 1) + i,
                    offset + j * (nu + 1) + i + 1,
                    offset + (j + 1) * (nu + 1) + i + 1,
                    offset + (j + 1) * (nu + 1) + i,
                )
                center = np.mean([self.vertices[k] for k in ids], axis=0)
                if keep is not None and not keep(u, v, center):
                    continue
                self.faces.append((ids[0], ids[1], ids[2]))
                self.faces.append((ids[0], ids[2], ids[3]))

    def rolled_y_edge(
        self,
        x0: float,
        x1: float,
        y0: float,
        radius: float,
        *,
        upward: bool = True,
        nx: int = 28,
        na: int = 14,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quarter cylinder tangent to z=0; its ends are negative controls."""
        sign = 1.0 if upward else -1.0
        offset = len(self.vertices)
        for j in range(na + 1):
            angle = 0.5 * math.pi * j / na
            y = y0 + radius * math.sin(angle)
            z = sign * radius * (1.0 - math.cos(angle))
            for i in range(nx + 1):
                x = x0 + (x1 - x0) * i / nx
                self.vertices.append(np.asarray((x, y, z), dtype=np.float64))
        for j in range(na):
            for i in range(nx):
                a = offset + j * (nx + 1) + i
                b = a + 1
                d = offset + (j + 1) * (nx + 1) + i
                c = d + 1
                self.faces.extend(((a, b, c), (a, c, d)))
        return np.asarray((x0, y0 + radius, sign * radius)), np.asarray((x1, y0 + radius, sign * radius))

    def mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(self.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(self.faces, dtype=np.int32)),
        )
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh


def _outside_features(features: list[tuple[str, float, float, float, float]]) -> Callable[[float, float, np.ndarray], bool]:
    """Return a cell predicate for circular/rectangular through-cut negatives."""
    def keep(_u: float, _v: float, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        for kind, cx, cy, a, b in features:
            if kind == "circle" and ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 < 1.0:
                return False
            if kind == "rect" and abs(x - cx) < a and abs(y - cy) < b:
                return False
        return True
    return keep


def _truth_line(rows: list[dict[str, Any]], start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
    rows.append({"bend_id": f"G{len(rows)+1:03d}", "line_start": list(start), "line_end": list(end)})


def _case_1() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    b, truth = MeshBuilder(), []
    holes = [("circle", -47, 10, 10, 10), ("rect", 31, -15, 17, 7)]
    b.quad((-88,-48,0),(88,-48,0),(88,48,0),(-88,48,0), nu=36, nv=20, keep=_outside_features(holes))
    b.quad((-72,-48,0),(61,-48,0),(61,-67,17),(-72,-67,17), nu=28, nv=5)
    _truth_line(truth, (-72,-48,0),(61,-48,0))
    b.quad((-88,1,0),(-88,42,0),(-105,42,-13),(-105,1,-13), nu=8, nv=5)
    _truth_line(truth, (-88,1,0),(-88,42,0))
    b.quad((59,48,0),(86,48,0),(86,63,10),(59,63,10), nu=7, nv=4)
    _truth_line(truth, (59,48,0),(86,48,0))
    return b.mesh(), truth, {"features": holes, "rounded_negative_count": 0}


def _case_2() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    b, truth = MeshBuilder(), []
    holes = [("circle", -48,-21,7,7),("circle", 36,18,5,5),("rect", 0,0,12,4)]
    b.quad((-82,-54,0),(82,-54,0),(82,54,0),(-82,54,0), nu=34, nv=22, keep=_outside_features(holes))
    b.quad((-82,-47,0),(-82,47,0),(-82,61,23),(-82,-61,23), nu=20, nv=6)
    _truth_line(truth, (-82,-47,0),(-82,47,0))
    b.quad((82,-47,0),(82,47,0),(82,61,18),(82,-61,18), nu=20, nv=5)
    _truth_line(truth, (82,-47,0),(82,47,0))
    b.quad((-67,54,0),(55,54,0),(55,68,28),(-67,68,28), nu=26, nv=7)
    _truth_line(truth, (-67,54,0),(55,54,0))
    for x0, x1 in ((-78,-55),(57,78)):
        b.quad((x0,-54,0),(x1,-54,0),(x1,-68,10),(x0,-68,10),nu=6,nv=4)
        _truth_line(truth,(x0,-54,0),(x1,-54,0))
    return b.mesh(), truth, {"features": holes, "rounded_negative_count": 0}


def _case_3() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    b, truth = MeshBuilder(), []
    # Five unequal finite panels form an offset cover; aperture patterns do not repeat.
    aperture = [("circle", 4,16,6,6),("rect", 56,-13,11,5)]
    b.quad((-102,-49,0),(-42,-49,0),(-42,49,0),(-102,49,0),nu=13,nv=20)
    b.quad((-42,-49,0),(-29,-49,14),(-29,49,14),(-42,49,0),nu=5,nv=20)
    _truth_line(truth,(-42,-49,0),(-42,49,0))
    _truth_line(truth,(-29,-49,14),(-29,49,14))
    b.quad((-29,-49,14),(38,-49,14),(38,49,14),(-29,49,14),nu=15,nv=20,keep=_outside_features(aperture))
    b.quad((38,-49,14),(52,-49,3),(52,49,3),(38,49,14),nu=5,nv=20)
    _truth_line(truth,(38,-49,14),(38,49,14))
    _truth_line(truth,(52,-49,3),(52,49,3))
    b.quad((52,-49,3),(101,-49,3),(101,49,3),(52,49,3),nu=11,nv=20)
    b.quad((-84,49,0),(-48,49,0),(-48,63,-11),(-84,63,-11),nu=8,nv=4)
    _truth_line(truth,(-84,49,0),(-48,49,0))
    return b.mesh(), truth, {"features": aperture, "rounded_negative_count": 0}


def _case_4() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    b, truth = MeshBuilder(), []
    opening = [("rect", 13,3,31,20),("circle",-63,-28,6,6),("circle",69,27,8,8)]
    b.quad((-105,-62,0),(105,-62,0),(105,62,0),(-105,62,0),nu=42,nv=25,keep=_outside_features(opening))
    # Local doghouse: four terminated side folds and a top, deliberately off-centre.
    x0,x1,y0,y1,h = -18,44,-17,23,24
    b.quad((x0,y0,0),(x1,y0,0),(x1,y0,h),(x0,y0,h),nu=14,nv=6); _truth_line(truth,(x0,y0,0),(x1,y0,0))
    b.quad((x1,y0,0),(x1,y1,0),(x1,y1,h),(x1,y0,h),nu=9,nv=6); _truth_line(truth,(x1,y0,0),(x1,y1,0))
    b.quad((x1,y1,0),(x0,y1,0),(x0,y1,h),(x1,y1,h),nu=14,nv=6); _truth_line(truth,(x1,y1,0),(x0,y1,0))
    b.quad((x0,y1,0),(x0,y0,0),(x0,y0,h),(x0,y1,h),nu=9,nv=6); _truth_line(truth,(x0,y1,0),(x0,y0,0))
    b.quad((x0,y0,h),(x1,y0,h),(x1,y1,h),(x0,y1,h),nu=14,nv=9)
    b.quad((-98,-62,0),(-59,-62,0),(-59,-75,8),(-98,-75,8),nu=9,nv=4); _truth_line(truth,(-98,-62,0),(-59,-62,0))
    return b.mesh(), truth, {"features": opening, "rounded_negative_count": 0}


def _case_5() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    b, truth = MeshBuilder(), []
    holes = [("circle",-53,-14,8,8),("rect",19,8,15,5)]
    b.quad((-92,-51,0),(92,-51,0),(92,38,0),(-92,38,0),nu=38,nv=18,keep=_outside_features(holes))
    # This broad quarter-round is intentionally unlabeled: it is a smooth form,
    # not a discrete manufacturing bend centre.
    b.rolled_y_edge(-92,92,38,18,nx=38,na=16)
    b.quad((-78,-51,0),(-43,-51,0),(-43,-66,13),(-78,-66,13),nu=8,nv=4); _truth_line(truth,(-78,-51,0),(-43,-51,0))
    b.quad((38,-51,0),(78,-51,0),(78,-64,-10),(38,-64,-10),nu=9,nv=4); _truth_line(truth,(38,-51,0),(78,-51,0))
    return b.mesh(), truth, {"features": holes, "rounded_negative_count": 2}


def _case_6() -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    b, truth = MeshBuilder(), []
    holes = [("circle",-67,25,5,5),("circle",-48,25,5,5),("rect",12,-19,22,4),("rect",69,29,9,8)]
    b.quad((-98,-55,0),(98,-55,0),(98,55,0),(-98,55,0),nu=40,nv=22,keep=_outside_features(holes))
    # Small shallow local crease represented by two finite facets in a side ear.
    b.quad((-91,-55,0),(-24,-55,0),(-24,-66,4),(-91,-66,4),nu=15,nv=4); _truth_line(truth,(-91,-55,0),(-24,-55,0))
    b.quad((-91,-66,4),(-24,-66,4),(-24,-73,4),(-91,-73,4),nu=15,nv=3); _truth_line(truth,(-91,-66,4),(-24,-66,4))
    b.quad((39,55,0),(92,55,0),(92,68,17),(39,68,17),nu=12,nv=5); _truth_line(truth,(39,55,0),(92,55,0))
    b.quad((39,68,17),(92,68,17),(92,75,12),(39,75,12),nu=12,nv=3); _truth_line(truth,(39,68,17),(92,68,17))
    return b.mesh(), truth, {"features": holes, "rounded_negative_count": 0}


BUILDERS = (_case_1, _case_2, _case_3, _case_4, _case_5, _case_6)


def build_case(family: dict[str, Any], capture_index: int) -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    family_number = int(family["id"][-2:])
    mesh, truth, metadata = BUILDERS[family_number - 1]()
    rng = np.random.default_rng(510_000 + 97 * family_number + capture_index)
    rotation = _rotation(520_000 + 53 * family_number + capture_index)
    translation = rng.uniform([-16,-12,-7],[16,12,7])
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices @ rotation.T + translation
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    for row in truth:
        line = np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)
        line = line @ rotation.T + translation
        row["line_start"] = line[0].round(9).tolist()
        row["line_end"] = line[1].round(9).tolist()
    metadata.update({
        "family_id": family["id"],
        "family_name": family["name"],
        "capture": capture_index,
        "spacing_mm": None,
        "normal_noise_sigma_mm": base.capture.DEPTH_NOISE_SIGMA_MM,
        "tangential_noise_sigma_mm": None,
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "bend_count": len(truth),
        "finite_sheet_topology": True,
        "constant_cross_section": False,
        "rotation": rotation.round(12).tolist(),
        "translation_mm": translation.round(9).tolist(),
    })
    return mesh, truth, metadata


def generate(output_root: Path, *, width: int = base.capture.DEFAULT_WIDTH, height: int = base.capture.DEFAULT_HEIGHT, view_count: int = base.capture.DEFAULT_VIEW_COUNT, family_limit: int | None = None) -> dict[str, Any]:
    previous = (base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case)
    base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case = TOKEN, CORPUS_KIND, CAPTURE_INDICES, FAMILIES, build_case
    try:
        report = base.generate(output_root, width=width, height=height, view_count=view_count, family_limit=family_limit)
    finally:
        base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES, base.build_case = previous
    renderer_path = output_root / "evaluator_only" / "renderer_manifest.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer.update({
        "generator_sha256": base.capture._sha256(Path(__file__).resolve()),
        "parent_capture_generator_sha256": base.capture._sha256(Path(base.__file__).resolve()),
        "visual_acceptance_required_before_inference": True,
        "finite_sheet_topology": True,
        "constant_cross_section": False,
    })
    base.capture._atomic_json(renderer_path, renderer)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=base.capture.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=base.capture.DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=base.capture.DEFAULT_VIEW_COUNT)
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    report = generate(args.output_root.resolve(), width=args.width, height=args.height, view_count=args.view_count, family_limit=args.family_limit)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "case_count": report["case_count"], "family_count": report["family_count"], "detector_execution_authorized": False, "token": TOKEN}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
