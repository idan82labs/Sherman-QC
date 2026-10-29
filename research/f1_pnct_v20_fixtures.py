"""Deterministic, source-free PNCT_V1 falsification fixtures."""

from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np


def _extrude(profiles: list[np.ndarray], *, length: float, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    stations = np.asarray([(index - 5) * length / 12.0 for index in range(11)])
    x_values = np.unique(np.concatenate(([-length / 2.0], stations, [length / 2.0])))
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    for profile in profiles:
        base = len(vertices)
        for x in x_values:
            vertices.extend([[x, y, z] for y, z in profile])
        width = len(profile)
        for xi in range(len(x_values) - 1):
            for pi in range(width - 1):
                a = base + xi * width + pi
                b = a + width
                triangles.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    return scale * np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def circular_fold(*, angle_deg: float, mid_radius: float, thickness: float, length: float, tails: float = 20.0, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(angle_deg)
    profiles = []
    for radius in (mid_radius - thickness / 2.0, mid_radius + thickness / 2.0):
        values = np.linspace(0.0, angle, 13)
        arc = np.asarray([(radius * math.cos(t), radius * math.sin(t)) for t in values])
        tangent0 = np.asarray((0.0, 1.0))
        tangent1 = np.asarray((-math.sin(angle), math.cos(angle)))
        profiles.append(np.vstack((arc[0] - tails * tangent0, arc, arc[-1] + tails * tangent1)))
    return _extrude(profiles, length=length, scale=scale)


def noncircular_fold(*, length: float = 40.0, thickness: float = 6.0, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    total = math.radians(75.0)
    phases = np.linspace(0.0, total, 13)
    # Integrate a C2 tangent field whose radius of curvature grows 18s -> 36s.
    center = [np.asarray((0.0, 0.0))]
    for left, right in zip(phases[:-1], phases[1:]):
        mid = 0.5 * (left + right)
        radius = 18.0 + 18.0 * (mid / total)
        center.append(center[-1] + radius * (right - left) * np.asarray((-math.sin(mid), math.cos(mid))))
    midline = np.asarray(center)
    normals = np.asarray([(math.cos(value), math.sin(value)) for value in phases])
    profiles = [midline + sign * 0.5 * thickness * normals for sign in (-1.0, 1.0)]
    tangent0 = np.asarray((0.0, 1.0))
    tangent1 = np.asarray((-math.sin(total), math.cos(total)))
    profiles = [np.vstack((profile[0] - 20.0 * tangent0, profile, profile[-1] + 20.0 * tangent1)) for profile in profiles]
    return _extrude(profiles, length=length, scale=scale)


def deterministic_delete(vertices: np.ndarray, triangles: np.ndarray, fraction: float, *, salt: str, contiguous: bool = False) -> np.ndarray:
    if fraction <= 0.0:
        return triangles.copy()
    keys = []
    for face in triangles:
        points = sorted(tuple(int(np.rint(value * 4096.0)) for value in row) for row in vertices[face])
        keys.append(int.from_bytes(hashlib.sha256((salt + repr(points)).encode("ascii")).digest()[:8], "little"))
    keys = np.asarray(keys, dtype=np.uint64)
    if contiguous:
        centers = vertices[triangles].mean(axis=1)
        phase = np.arctan2(centers[:, 2], centers[:, 1]) % math.pi
        order = np.argsort(phase, kind="stable")
        remove = set(int(value) for value in order[: int(np.rint(fraction * len(order)))])
        return np.asarray([face for index, face in enumerate(triangles) if index not in remove], dtype=np.int32)
    keep = keys % 1_000_000 >= int(fraction * 1_000_000)
    return triangles[keep]


def winding_reverse(triangles: np.ndarray, fraction: float, *, salt: str) -> np.ndarray:
    result = triangles.copy()
    for index in range(len(result)):
        key = int.from_bytes(hashlib.sha256(f"{salt}:{index}".encode()).digest()[:8], "little")
        if key % 1_000_000 < int(fraction * 1_000_000):
            result[index] = result[index, ::-1]
    return result


def combine(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles, offset = [], [], 0
    for points, faces in parts:
        vertices.append(points)
        triangles.append(faces + offset)
        offset += len(points)
    return np.vstack(vertices), np.vstack(triangles)


def flat_strip(*, y: float, z: float, length: float = 40.0, width: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    profile = np.asarray(((y, z - width / 2.0), (y, z + width / 2.0)))
    return _extrude([profile], length=length)


def source_free_cases() -> tuple[dict[str, tuple[np.ndarray, np.ndarray, float, float]], dict[str, tuple[np.ndarray, np.ndarray, float, float]]]:
    positives = {}
    v, t = circular_fold(angle_deg=90, mid_radius=12, thickness=8, length=40)
    t = deterministic_delete(v, t, .05, salt="sf01")
    positives["SF01_standard_fold"] = (v, winding_reverse(t, .20, salt="sf01"), 90.0, 40.0)
    v, t = circular_fold(angle_deg=90, mid_radius=12, thickness=8, length=16)
    t = deterministic_delete(v, t, .05, salt="sf02")
    positives["SF02_axial_crop"] = (v, winding_reverse(t, .20, salt="sf02"), 90.0, 16.0)
    v, t = circular_fold(angle_deg=60, mid_radius=24, thickness=6, length=40)
    positives["SF03_scale_angle"] = (v, deterministic_delete(v, t, .15, salt="sf03", contiguous=True), 60.0, 40.0)
    v, t = noncircular_fold()
    positives["SF04_noncircular"] = (v, deterministic_delete(v, t, .20, salt="sf04"), 75.0, 40.0)

    base_v, base_t = circular_fold(angle_deg=90, mid_radius=12, thickness=8, length=40)
    half = len(base_v) // 2
    single_t = base_t[np.all(base_t < half, axis=1)]
    negatives = {
        "N01_flat_double_skin": (*combine([flat_strip(y=8, z=0), flat_strip(y=16, z=0)]), 90.0, 40.0),
        "N02_single_skin_bend": (base_v[:half], single_t, 90.0, 40.0),
        "N03_silhouette_free_edge": (*flat_strip(y=12, z=0), 90.0, 40.0),
        "N04_straight_opening_rim": (*combine([flat_strip(y=8, z=-4), flat_strip(y=8, z=4)]), 90.0, 40.0),
        "N05_rounded_terminal_cap": (*_extrude([np.asarray([(12 * math.cos(t), 12 * math.sin(t)) for t in np.linspace(0, math.pi, 25)])], length=40), 90.0, 40.0),
        "N06_welded_miter_bead": (*combine([flat_strip(y=12, z=0), _extrude([np.asarray([(12 + 3 * math.cos(t), 3 * math.sin(t)) for t in np.linspace(0, math.pi, 13)])], length=40)]), 90.0, 40.0),
        "N07_overlap_cycle": (*combine([flat_strip(y=8, z=0), flat_strip(y=14, z=0), flat_strip(y=20, z=0)]), 90.0, 40.0),
        "N08_nearby_panel": (*combine([flat_strip(y=8, z=0), flat_strip(y=14, z=0)]), 90.0, 40.0),
        "N09_full_double_wall_cylinder": (*_extrude([np.asarray([(r * math.cos(t), r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 49)]) for r in (8, 16)], length=40), 90.0, 40.0),
    }
    return positives, negatives
