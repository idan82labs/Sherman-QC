"""Immutable analytic source-free fixtures for LSSC-V1."""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np


def _fixture_points(token: str, records: list[tuple[tuple[int, ...], np.ndarray]], s: float = 1.0) -> np.ndarray:
    rows: list[np.ndarray] = []
    prefix = token.encode("ascii") + b"\0"
    for identity, point in records:
        encoded = prefix + b"".join(struct.pack("<q", int(value)) for value in identity)
        delete_value = int.from_bytes(hashlib.sha256(encoded + b"\0DELETE").digest()[:8], "little")
        if delete_value < int(0.05 * (1 << 64)):
            continue
        jitter = np.empty(3, dtype=np.float64)
        for coordinate in range(3):
            digest = hashlib.sha256(encoded + b"\0JITTER\0" + bytes((coordinate,))).digest()
            unit = int.from_bytes(digest[:8], "little") / float((1 << 64) - 1)
            jitter[coordinate] = (2.0 * unit - 1.0) * 0.03 * s
        rows.append(np.asarray(point, dtype=np.float64) + jitter)
    return np.asarray(rows, dtype=np.float64)


def bend(
    *,
    token: str,
    radii: tuple[float, ...],
    angle_degrees: float,
    axis_low: float = -48.0,
    axis_high: float = 48.0,
    flange_length: float = 40.0,
    pitch: float = 1.0,
) -> np.ndarray:
    angle = math.radians(angle_degrees)
    axial = np.arange(axis_low, axis_high + 0.5 * pitch, pitch)
    records: list[tuple[tuple[int, ...], np.ndarray]] = []
    for skin_id, radius in enumerate(radii):
        arc_count = max(2, int(math.ceil(abs(radius * angle) / pitch)) + 1)
        phis = np.linspace(0.0, angle, arc_count)
        for arc_id, phi in enumerate(phis):
            for axial_id, y in enumerate(axial):
                records.append(
                    (
                        (0, skin_id, arc_id, axial_id),
                        np.asarray((radius * math.cos(phi), y, radius * math.sin(phi))),
                    )
                )
        if flange_length > 0:
            distances = np.arange(pitch, flange_length + 0.5 * pitch, pitch)
            start = np.asarray((radius, 0.0, 0.0))
            start_out = np.asarray((0.0, 0.0, -1.0))
            end = np.asarray((radius * math.cos(angle), 0.0, radius * math.sin(angle)))
            end_out = np.asarray((-math.sin(angle), 0.0, math.cos(angle)))
            for side, (origin, outward) in enumerate(((start, start_out), (end, end_out))):
                for distance_id, distance in enumerate(distances):
                    for axial_id, y in enumerate(axial):
                        records.append(
                            (
                                (1, skin_id, side, distance_id, axial_id),
                                origin + outward * distance + np.asarray((0.0, y, 0.0)),
                            )
                        )
    return _fixture_points(token, records, pitch)


def sf01_double_skin() -> np.ndarray:
    return bend(token="LSSC_SF01", radii=(8.0, 16.0), angle_degrees=90.0)


def sf02_axial_crop() -> np.ndarray:
    points = sf01_double_skin()
    return np.ascontiguousarray(points[(points[:, 1] >= -24.0) & (points[:, 1] <= 24.0)])


def sf03_scale_angle() -> np.ndarray:
    return bend(
        token="LSSC_SF03",
        radii=(21.0, 27.0),
        angle_degrees=60.0,
        flange_length=54.0,
    )


def sf04_single_skin() -> np.ndarray:
    return bend(token="LSSC_SF04", radii=(12.0,), angle_degrees=90.0)


def flat_double_skin(
    *,
    token: str,
    hole_radius: float | None = None,
    include_hole_wall: bool = False,
    terminal_wall: bool = False,
    pitch: float = 1.0,
) -> np.ndarray:
    xs = np.arange(-60.0, 60.0 + 0.5 * pitch, pitch)
    ys = np.arange(-48.0, 48.0 + 0.5 * pitch, pitch)
    records: list[tuple[tuple[int, ...], np.ndarray]] = []
    for skin, z in enumerate((-4.0, 4.0)):
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                if hole_radius is not None and x * x + y * y < hole_radius * hole_radius:
                    continue
                records.append(((0, skin, xi, yi), np.asarray((x, y, z))))
    if include_hole_wall and hole_radius is not None:
        angle_count = max(24, int(math.ceil(2.0 * math.pi * hole_radius / pitch)))
        for angle_id, phi in enumerate(np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)):
            for z_id, z in enumerate(np.arange(-4.0, 4.0 + 0.5 * pitch, pitch)):
                records.append(
                    ((1, angle_id, z_id), np.asarray((hole_radius * math.cos(phi), hole_radius * math.sin(phi), z)))
                )
    if terminal_wall:
        for yi, y in enumerate(ys):
            for z_id, z in enumerate(np.arange(-4.0, 4.0 + 0.5 * pitch, pitch)):
                records.append(((2, yi, z_id), np.asarray((60.0, y, z))))
    return _fixture_points(token, records, pitch)


def complete_cylinders(*, token: str, radii: tuple[float, ...]) -> np.ndarray:
    return bend(
        token=token,
        radii=radii,
        angle_degrees=360.0,
        flange_length=0.0,
    )


def sphere(*, token: str = "LSSC_SF10", radius: float = 16.0) -> np.ndarray:
    count = int(round(4.0 * math.pi * radius * radius))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    records: list[tuple[tuple[int, ...], np.ndarray]] = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * index
        records.append(
            ((index,), radius * np.asarray((radial * math.cos(phi), radial * math.sin(phi), z)))
        )
    return _fixture_points(token, records, 1.0)
