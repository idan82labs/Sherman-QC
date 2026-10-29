#!/usr/bin/env python3
"""Generate a quarantined LiDAR-like sheet-metal bend stress corpus.

The corpus is engineering-only and may never be counted as untouched or
confirmatory evidence. Geometry is formed by extruding filleted 2D sheet
profiles, then applying deterministic sampling, pose, noise, and dropout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
TARGET_MAX_VERTICES = 195_000


FAMILIES: tuple[dict[str, Any], ...] = (
    {"id": "SF01", "name": "right_angle_l", "profile": [(0, 0), (0, 38), (52, 38)], "length": 150, "radius": 3.0},
    {"id": "SF02", "name": "asymmetric_u", "profile": [(0, 48), (0, 0), (70, 0), (70, 32)], "length": 180, "radius": 4.0},
    {"id": "SF03", "name": "offset_z", "profile": [(0, 0), (32, 0), (52, 27), (88, 27)], "length": 165, "radius": 3.5},
    {"id": "SF04", "name": "wide_hat", "profile": [(0, 0), (22, 0), (34, 34), (78, 34), (90, 0), (114, 0)], "length": 220, "radius": 4.0},
    {"id": "SF05", "name": "square_step", "profile": [(0, 0), (30, 0), (30, 24), (72, 24), (72, 48), (116, 48)], "length": 200, "radius": 3.0},
    {"id": "SF06", "name": "return_c", "profile": [(12, 54), (0, 54), (0, 0), (64, 0), (64, 54), (52, 54)], "length": 190, "radius": 3.5},
    {"id": "SF07", "name": "deep_channel", "profile": [(0, 62), (0, 0), (92, 0), (92, 38)], "length": 245, "radius": 5.0},
    {"id": "SF08", "name": "double_step", "profile": [(0, 0), (26, 0), (26, 20), (54, 20), (54, 42), (98, 42)], "length": 175, "radius": 2.5},
    {"id": "SF09", "name": "double_z", "profile": [(0, 0), (28, 0), (44, 21), (72, 21), (88, 0), (120, 0)], "length": 210, "radius": 3.0},
    {"id": "SF10", "name": "corrugated_w", "profile": [(0, 0), (24, 0), (35, 18), (60, 18), (71, 0), (96, 0), (107, 18), (134, 18)], "length": 230, "radius": 2.8},
    {"id": "SF11", "name": "narrow_hat", "profile": [(0, 0), (16, 0), (27, 46), (58, 46), (69, 0), (86, 0)], "length": 160, "radius": 3.2},
    {"id": "SF12", "name": "omega_return", "profile": [(0, 0), (16, 0), (27, 16), (27, 42), (68, 42), (68, 16), (79, 0), (96, 0)], "length": 205, "radius": 2.8},
    {"id": "SF13", "name": "acute_v", "profile": [(0, 0), (48, 0), (69, 48)], "length": 185, "radius": 4.5},
    {"id": "SF14", "name": "obtuse_v", "profile": [(0, 0), (50, 0), (88, 17)], "length": 155, "radius": 5.0},
    {"id": "SF15", "name": "asymmetric_multistep", "profile": [(0, 0), (22, 0), (34, 21), (61, 21), (73, 47), (105, 47), (120, 22)], "length": 235, "radius": 3.0},
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _line_samples(start: np.ndarray, end: np.ndarray, spacing: float, include_end: bool = False) -> list[np.ndarray]:
    length = float(np.linalg.norm(end - start))
    count = max(1, int(math.ceil(length / spacing)))
    values = np.linspace(0.0, 1.0, count + int(include_end), endpoint=include_end)
    return [start * (1.0 - value) + end * value for value in values]


def _fillet(profile: list[tuple[float, float]], radius: float, spacing: float) -> tuple[np.ndarray, list[np.ndarray]]:
    points = [np.asarray(value, dtype=np.float64) for value in profile]
    corners: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for index in range(1, len(points) - 1):
        previous, vertex, following = points[index - 1 : index + 2]
        incoming = _unit(vertex - previous)
        outgoing = _unit(following - vertex)
        turn = math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
        distance = min(radius * math.tan(turn / 2.0), 0.35 * np.linalg.norm(vertex - previous), 0.35 * np.linalg.norm(following - vertex))
        tangent_a = vertex - incoming * distance
        tangent_b = vertex + outgoing * distance
        normal_a = np.array([-incoming[1], incoming[0]])
        normal_b = np.array([-outgoing[1], outgoing[0]])
        matrix = np.column_stack((normal_a, -normal_b))
        coefficients = np.linalg.solve(matrix, tangent_b - tangent_a)
        center = tangent_a + coefficients[0] * normal_a
        angle_a = math.atan2(tangent_a[1] - center[1], tangent_a[0] - center[0])
        angle_b = math.atan2(tangent_b[1] - center[1], tangent_b[0] - center[0])
        signed_cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        direction = 1.0 if signed_cross > 0 else -1.0
        delta = (angle_b - angle_a) % (2.0 * math.pi)
        if direction < 0:
            delta = -((angle_a - angle_b) % (2.0 * math.pi))
        corners.append((tangent_a, tangent_b, center, delta))

    samples: list[np.ndarray] = []
    bend_midpoints: list[np.ndarray] = []
    cursor = points[0]
    for tangent_a, tangent_b, center, delta in corners:
        samples.extend(_line_samples(cursor, tangent_a, spacing))
        radius_actual = float(np.linalg.norm(tangent_a - center))
        angle_a = math.atan2(tangent_a[1] - center[1], tangent_a[0] - center[0])
        arc_length = abs(delta) * radius_actual
        count = max(3, int(math.ceil(arc_length / spacing)))
        for value in np.linspace(0.0, 1.0, count, endpoint=False):
            angle = angle_a + value * delta
            samples.append(center + radius_actual * np.array([math.cos(angle), math.sin(angle)]))
        middle_angle = angle_a + 0.5 * delta
        bend_midpoints.append(center + radius_actual * np.array([math.cos(middle_angle), math.sin(middle_angle)]))
        cursor = tangent_b
    samples.extend(_line_samples(cursor, points[-1], spacing, include_end=True))
    clean = [samples[0]]
    for value in samples[1:]:
        if np.linalg.norm(value - clean[-1]) > 1e-7:
            clean.append(value)
    return np.vstack(clean), bend_midpoints


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.deg2rad(rng.uniform([-18.0, -12.0, -28.0], [18.0, 12.0, 28.0]))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _deviated_profile(family: dict[str, Any], capture: int) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    profile = np.asarray(family["profile"], dtype=np.float64)
    if capture == 0:
        return [tuple(value) for value in profile], {"kind": "nominal", "vertex_index": None, "offset_mm": [0.0, 0.0]}
    index = 1 + (int(family["id"][-2:]) * 3) % (len(profile) - 2)
    magnitude = 2.0 + (int(family["id"][-2:]) % 5) * 0.8
    before = _unit(profile[index] - profile[index - 1])
    after = _unit(profile[index + 1] - profile[index])
    normal = _unit(np.array([-(before + after)[1], (before + after)[0]]))
    offset = normal * magnitude
    profile[index] += offset
    return [tuple(value) for value in profile], {"kind": "controlled_vertex_offset", "vertex_index": int(index), "offset_mm": offset.round(6).tolist()}


def build_case(family: dict[str, Any], capture: int) -> tuple[o3d.geometry.TriangleMesh, list[dict[str, Any]], dict[str, Any]]:
    family_number = int(family["id"][-2:])
    rng = np.random.default_rng(10_000 + 97 * family_number + capture)
    base_spacing = 0.38 + 0.025 * ((family_number + 2 * capture) % 9)
    if capture:
        base_spacing *= 1.22
    profile_spec, deviation = _deviated_profile(family, capture)
    profile, bend_midpoints = _fillet(profile_spec, float(family["radius"]), base_spacing)
    length = float(family["length"]) * (1.0 + (0.012 if capture else 0.0))
    x_count = max(3, int(math.floor(length / base_spacing)) + 1)
    estimated = x_count * len(profile)
    if estimated > TARGET_MAX_VERTICES:
        scale = math.sqrt(estimated / TARGET_MAX_VERTICES)
        base_spacing *= scale
        profile, bend_midpoints = _fillet(profile_spec, float(family["radius"]), base_spacing)
        x_count = max(3, int(math.floor(length / base_spacing)) + 1)
    x_values = np.linspace(-length / 2.0, length / 2.0, x_count)
    vertices = np.empty((x_count * len(profile), 3), dtype=np.float64)
    normals = np.empty_like(vertices)
    for profile_index, yz in enumerate(profile):
        previous = profile[max(0, profile_index - 1)]
        following = profile[min(len(profile) - 1, profile_index + 1)]
        tangent = _unit(following - previous)
        normal = np.array([0.0, -tangent[1], tangent[0]])
        rows = np.column_stack((x_values, np.full(x_count, yz[0]), np.full(x_count, yz[1])))
        start = profile_index * x_count
        vertices[start : start + x_count] = rows
        normals[start : start + x_count] = normal

    normal_sigma = 0.055 + 0.022 * ((family_number + capture) % 6)
    tangential_sigma = 0.025 + 0.01 * ((family_number + 2 * capture) % 4)
    vertices += normals * rng.normal(0.0, normal_sigma, size=(len(vertices), 1))
    vertices[:, 0] += rng.normal(0.0, tangential_sigma, size=len(vertices))
    vertices += np.column_stack((np.zeros(len(vertices)), normals[:, 2], -normals[:, 1])) * rng.normal(0.0, tangential_sigma, size=(len(vertices), 1))
    drift = (0.025 + 0.012 * capture) * np.sin(vertices[:, 0] / max(6.0, 0.17 * length) * 2.0 * math.pi)
    vertices += normals * drift[:, None]

    faces = []
    for profile_index in range(len(profile) - 1):
        row_a = profile_index * x_count
        row_b = (profile_index + 1) * x_count
        for x_index in range(x_count - 1):
            a, b = row_a + x_index, row_a + x_index + 1
            c, d = row_b + x_index, row_b + x_index + 1
            faces.append((a, b, c))
            faces.append((b, d, c))
    faces_array = np.asarray(faces, dtype=np.int32)

    profile_coordinate = np.repeat(np.linspace(0.0, 1.0, len(profile)), x_count)
    x_coordinate = np.tile(np.linspace(0.0, 1.0, x_count), len(profile))
    keep = rng.random(len(vertices)) > (0.0015 + 0.0025 * capture)
    hole_count = 2 + (family_number + capture) % 4
    holes = []
    for hole_index in range(hole_count):
        center_x = rng.uniform(0.10, 0.90)
        center_s = rng.uniform(0.08, 0.92)
        width_x = rng.uniform(0.025, 0.075) * (1.35 if capture else 1.0)
        width_s = rng.uniform(0.018, 0.055) * (1.35 if capture else 1.0)
        normalized_x = (x_coordinate - center_x) / width_x
        normalized_s = (profile_coordinate - center_s) / width_s
        phase = rng.uniform(0.0, 2.0 * math.pi)
        ragged_boundary = (
            0.16 * np.sin(11.0 * normalized_x + phase)
            + 0.11 * np.sin(17.0 * normalized_s - 0.7 * phase)
        )
        hole = normalized_x**2 + normalized_s**2 + ragged_boundary < 1.0
        keep &= ~hole
        holes.append({"x": center_x, "profile": center_s, "half_width_x": width_x, "half_width_profile": width_s})
    if capture:
        truncation = 0.018 + 0.006 * (family_number % 3)
        keep &= ~((x_coordinate > 1.0 - truncation) & (profile_coordinate > 0.45))
        wedge_center = rng.uniform(0.52, 0.76)
        wedge_slope = rng.uniform(-0.22, 0.22)
        occluded = (
            profile_coordinate > wedge_center
        ) & (
            x_coordinate > 0.86 + wedge_slope * (profile_coordinate - wedge_center)
        )
        keep &= ~occluded

    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    face_keep = keep[faces_array].all(axis=1)
    faces_array = remap[faces_array[face_keep]].astype(np.int32)
    vertices = vertices[keep]
    normals = normals[keep]

    rotation = _rotation(20_000 + 53 * family_number + capture)
    translation = rng.uniform([-18.0, -12.0, -8.0], [18.0, 12.0, 8.0])
    vertices = vertices @ rotation.T + translation
    normals = normals @ rotation.T

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces_array),
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    truth = []
    for bend_index, yz in enumerate(bend_midpoints, 1):
        line = np.asarray([[-length / 2.0, yz[0], yz[1]], [length / 2.0, yz[0], yz[1]]])
        line = line @ rotation.T + translation
        truth.append({"bend_id": f"G{bend_index:03d}", "line_start": line[0].round(9).tolist(), "line_end": line[1].round(9).tolist()})
    metadata = {
        "family_id": family["id"],
        "family_name": family["name"],
        "capture": capture + 1,
        "spacing_mm": round(base_spacing, 6),
        "normal_noise_sigma_mm": round(normal_sigma, 6),
        "tangential_noise_sigma_mm": round(tangential_sigma, 6),
        "vertex_count": len(vertices),
        "triangle_count": len(faces_array),
        "bend_count": len(truth),
        "structured_holes": holes,
        "deviation": deviation,
        "rotation": rotation.round(12).tolist(),
        "translation_mm": translation.round(9).tolist(),
    }
    return mesh, truth, metadata


def generate(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"synthetic corpus output already exists: {output_root}")
    scans = output_root / "scans"
    scans.mkdir(parents=True)
    cases = []
    truth_cases = []
    family_map: dict[str, list[str]] = {}
    metadata_rows = []
    for family_index, family in enumerate(FAMILIES):
        family_cases = []
        for capture in range(2):
            case_id = f"S{family_index * 2 + capture + 1:03d}"
            mesh, truth, metadata = build_case(family, capture)
            scan_path = scans / f"{case_id}.ply"
            ok = o3d.io.write_triangle_mesh(str(scan_path), mesh, write_ascii=False, compressed=False, write_vertex_normals=True)
            if not ok:
                raise RuntimeError(f"failed writing {scan_path}")
            family_cases.append(case_id)
            cases.append({"case_id": case_id, "scan_path": str(scan_path.resolve()), "scan_sha256": _sha(scan_path)})
            truth_cases.append({"case_id": case_id, "manual_bends": truth})
            metadata_rows.append({"case_id": case_id, **metadata, "scan_sha256": _sha(scan_path)})
        family_map[str(family["id"])] = family_cases
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_lidar_like_synthetic_stress_corpus",
        "token": TOKEN,
        "confirmatory_eligible": False,
        "case_count": 30,
        "family_count": 15,
        "cases": cases,
    }
    truth = {"schema_version": 1, "token": TOKEN, "confirmatory_eligible": False, "cases": truth_cases}
    families = {"schema_version": 1, "token": TOKEN, "confirmatory_eligible": False, "families": family_map}
    metadata = {"schema_version": 1, "token": TOKEN, "measured_reference_spacing_mm": {"median_of_scan_medians": 0.4631, "observed_scan_median_range": [0.1538, 1.0391]}, "cases": metadata_rows}
    for name, payload in (("manifest.json", manifest), ("truth.json", truth), ("family_map.json", families), ("generation_metadata.json", metadata)):
        (output_root / name).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.output_root.resolve())
    print(json.dumps({"output_root": str(args.output_root.resolve()), "case_count": report["case_count"], "family_count": report["family_count"], "confirmatory_eligible": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
