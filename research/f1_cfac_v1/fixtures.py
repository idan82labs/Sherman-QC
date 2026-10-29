"""Frozen analytic fixtures for CFAC-V1."""

from __future__ import annotations

import hashlib
import math

import numpy as np


def _perturb(points: np.ndarray, pitch: float) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row in np.asarray(points, dtype=np.float64):
        raw = np.ascontiguousarray(row, dtype="<f8").tobytes()
        if int.from_bytes(hashlib.sha256(b"CFAC_V1_DELETE\0" + raw).digest()[:8], "little") % 20 == 0:
            continue
        digest = hashlib.sha256(b"CFAC_V1_JITTER\0" + raw).digest()
        jitter = (
            np.asarray(
                [int.from_bytes(digest[index : index + 2], "little") for index in (0, 2, 4)],
                dtype=np.float64,
            )
            / 65535.0
            - 0.5
        ) * (0.06 * pitch)
        rows.append(row + jitter)
    return np.asarray(rows, dtype=np.float64)


def concentric_bend(
    *,
    radii: tuple[float, ...] = (8.0, 16.0),
    angle_degrees: float = 90.0,
    axis_low: float = -48.0,
    axis_high: float = 48.0,
    flange_length: float = 40.0,
    pitch: float = 1.0,
    perturb: bool = True,
) -> np.ndarray:
    angle = math.radians(angle_degrees)
    axial = np.arange(axis_low, axis_high + 0.5 * pitch, pitch)
    rows: list[np.ndarray] = []
    for radius in radii:
        angle_count = max(2, int(math.ceil(radius * angle / pitch)) + 1)
        angles = np.linspace(0.0, angle, angle_count)
        for phi in angles:
            radial = np.asarray((math.cos(phi), 0.0, math.sin(phi)))
            for axis in axial:
                rows.append(np.asarray((0.0, axis, 0.0)) + radius * radial)
        distances = np.arange(pitch, flange_length + 0.5 * pitch, pitch)
        start = np.asarray((radius, 0.0, 0.0))
        start_outward = np.asarray((0.0, 0.0, -1.0))
        end = np.asarray((radius * math.cos(angle), 0.0, radius * math.sin(angle)))
        end_outward = np.asarray((-math.sin(angle), 0.0, math.cos(angle)))
        for distance in distances:
            for axis in axial:
                axis_vector = np.asarray((0.0, axis, 0.0))
                rows.append(axis_vector + start + distance * start_outward)
                rows.append(axis_vector + end + distance * end_outward)
    points = np.asarray(rows, dtype=np.float64)
    return _perturb(points, pitch) if perturb else points


def flat_double_skin(
    *,
    extent: float = 48.0,
    separation: float = 8.0,
    pitch: float = 1.0,
    hole_radius: float | None = None,
    include_hole_wall: bool = False,
) -> np.ndarray:
    axis = np.arange(-extent, extent + 0.5 * pitch, pitch)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    base = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    if hole_radius is not None:
        base = base[np.sum(base[:, :2] ** 2, axis=1) >= hole_radius**2]
    rows = [
        base + np.asarray((0.0, 0.0, 0.5 * separation)),
        base - np.asarray((0.0, 0.0, 0.5 * separation)),
    ]
    if include_hole_wall and hole_radius is not None:
        angle_count = max(24, int(math.ceil(2 * math.pi * hole_radius / pitch)))
        angles = np.linspace(0.0, 2 * math.pi, angle_count, endpoint=False)
        z_values = np.arange(-0.5 * separation, 0.5 * separation + 0.5 * pitch, pitch)
        wall = [
            (hole_radius * math.cos(phi), hole_radius * math.sin(phi), z)
            for phi in angles
            for z in z_values
        ]
        rows.append(np.asarray(wall, dtype=np.float64))
    return _perturb(np.vstack(rows), pitch)


def terminal_geometry(*, pitch: float = 1.0) -> np.ndarray:
    # Two horizontal skins joined at x=0 by a terminal wall.  Radius-4
    # quarter-rounds connect each skin to the wall.
    y_values = np.arange(-48.0, 48.0 + 0.5 * pitch, pitch)
    x_values = np.arange(0.0, 48.0 + 0.5 * pitch, pitch)
    z_values = np.arange(-4.0, 4.0 + 0.5 * pitch, pitch)
    rows: list[np.ndarray] = []
    for y in y_values:
        for x in x_values:
            rows.append(np.asarray((x + 4.0, y, 4.0)))
            rows.append(np.asarray((x + 4.0, y, -4.0)))
        for z in z_values:
            rows.append(np.asarray((0.0, y, z)))
        for phi in np.linspace(0.0, math.pi / 2.0, max(7, int(math.ceil(4 * math.pi / 2 / pitch)) + 1)):
            # Top and bottom quarter-rounds, both with a single radius only.
            rows.append(np.asarray((4.0 - 4.0 * math.cos(phi), y, 4.0 * math.sin(phi))))
            rows.append(np.asarray((4.0 - 4.0 * math.cos(phi), y, -4.0 * math.sin(phi))))
    return _perturb(np.asarray(rows), pitch)


def positive_plus_hole_stress() -> np.ndarray:
    positive = concentric_bend(pitch=0.45, flange_length=40.0)
    plate = flat_double_skin(
        extent=36.0,
        separation=8.0,
        pitch=0.45,
        hole_radius=12.0,
        include_hole_wall=True,
    )
    plate = plate + np.asarray((90.0, 0.0, 0.0))
    combined = np.vstack((positive, plate))
    # Deterministically add a second translated flat plate if needed, then
    # take a hash-ranked 200k subset when above the exact stress cardinality.
    if len(combined) < 200_000:
        extra = flat_double_skin(extent=40.0, separation=8.0, pitch=0.45)
        combined = np.vstack((combined, extra + np.asarray((-100.0, 0.0, 0.0))))
    if len(combined) > 200_000:
        keys = np.fromiter(
            (
                int.from_bytes(
                    hashlib.sha256(
                        b"CFAC_V1_STRESS_SAMPLE\0"
                        + np.ascontiguousarray(row, dtype="<f8").tobytes()
                    ).digest()[:8],
                    "little",
                )
                for row in combined
            ),
            dtype=np.uint64,
            count=len(combined),
        )
        combined = combined[np.lexsort((np.arange(len(combined)), keys))[:200_000]]
    return np.asarray(combined, dtype=np.float64)
