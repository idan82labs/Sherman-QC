"""Immutable LEPG-V1 analytic fixtures."""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np


def _jitter_delete(token: str, records: list[tuple[tuple[int, ...], np.ndarray]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    prefix = token.encode("ascii") + b"\0"
    for identity, point in records:
        key = prefix + b"".join(struct.pack("<q", int(value)) for value in identity)
        digest = hashlib.sha256(key).digest()
        if int.from_bytes(digest[:8], "little") % 20 == 0:
            continue
        jitter = []
        for coordinate in range(3):
            value = int.from_bytes(hashlib.sha256(key + bytes((coordinate,))).digest()[:8], "little")
            jitter.append((2.0 * value / float((1 << 64) - 1) - 1.0) * 0.03)
        rows.append(np.asarray(point, dtype=np.float64) + np.asarray(jitter))
    return np.asarray(rows, dtype=np.float64)


def extruded_profile(
    *,
    token: str,
    radii: tuple[float, ...],
    angle_degrees: float,
    continuation_length: float,
    axial_low: float,
    axial_high: float,
) -> np.ndarray:
    theta = math.radians(angle_degrees)
    axial = np.arange(axial_low, axial_high + 0.5, 1.0)
    records: list[tuple[tuple[int, ...], np.ndarray]] = []
    for shell, radius in enumerate(radii):
        start_phi, end_phi = -0.5 * theta, 0.5 * theta
        arc_count = max(2, int(math.ceil(radius * abs(theta))) + 1)
        arc = np.linspace(start_phi, end_phi, arc_count)
        start = np.asarray((radius * math.cos(start_phi), radius * math.sin(start_phi)))
        end = np.asarray((radius * math.cos(end_phi), radius * math.sin(end_phi)))
        start_tangent = np.asarray((-math.sin(start_phi), math.cos(start_phi)))
        end_tangent = np.asarray((-math.sin(end_phi), math.cos(end_phi)))
        profile: list[tuple[int, int, np.ndarray]] = []
        if continuation_length > 0:
            for index, distance in enumerate(np.linspace(continuation_length, 0.0, int(math.ceil(continuation_length)) + 1)[:-1]):
                profile.append((0, index, start - distance * start_tangent))
        for index, phi in enumerate(arc):
            profile.append((1, index, np.asarray((radius * math.cos(phi), radius * math.sin(phi)))))
        if continuation_length > 0:
            for index, distance in enumerate(np.linspace(0.0, continuation_length, int(math.ceil(continuation_length)) + 1)[1:]):
                profile.append((2, index, end + distance * end_tangent))
        for kind, profile_index, point_2d in profile:
            for axial_index, y in enumerate(axial):
                records.append(((shell, kind, profile_index, axial_index), np.asarray((point_2d[0], y, point_2d[1]))))
    return _jitter_delete(token, records)


def sf01() -> np.ndarray:
    return extruded_profile(token="LEPG_SF01", radii=(8.0, 16.0), angle_degrees=90.0, continuation_length=24.0, axial_low=-32.0, axial_high=32.0)


def sf02() -> np.ndarray:
    return extruded_profile(token="LEPG_SF02", radii=(8.0, 16.0), angle_degrees=90.0, continuation_length=24.0, axial_low=-14.0, axial_high=14.0)


def sf03() -> np.ndarray:
    return extruded_profile(token="LEPG_SF03", radii=(21.0, 27.0), angle_degrees=60.0, continuation_length=30.0, axial_low=-36.0, axial_high=36.0)


def sf05_single_skin() -> np.ndarray:
    return extruded_profile(token="LEPG_SF05", radii=(16.0,), angle_degrees=90.0, continuation_length=24.0, axial_low=-32.0, axial_high=32.0)


def sf07_open_sectors() -> np.ndarray:
    return extruded_profile(token="LEPG_SF07", radii=(8.0, 16.0), angle_degrees=90.0, continuation_length=0.0, axial_low=-32.0, axial_high=32.0)


def sf08_short_bend() -> np.ndarray:
    return extruded_profile(token="LEPG_SF08", radii=(8.0, 16.0), angle_degrees=90.0, continuation_length=24.0, axial_low=-6.0, axial_high=6.0)


def flat_double_skin(*, token: str, hole_radius: float | None = None, hole_wall: bool = False, terminal_wall: bool = False) -> np.ndarray:
    records: list[tuple[tuple[int, ...], np.ndarray]] = []
    ys = np.arange(-32.0 if not hole_radius else -40.0, 32.0 if not hole_radius else 40.0 + 0.5, 1.0)
    xs = np.arange(-40.0, 40.5, 1.0)
    for shell, z in enumerate((-4.0, 4.0)):
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                if hole_radius is not None and x * x + y * y < hole_radius * hole_radius:
                    continue
                records.append(((0, shell, xi, yi), np.asarray((x, y, z))))
    if hole_wall and hole_radius is not None:
        count = int(math.ceil(2.0 * math.pi * hole_radius))
        for ai, phi in enumerate(np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)):
            for zi, z in enumerate(np.arange(-4.0, 4.5, 1.0)):
                records.append(((1, ai, zi), np.asarray((hole_radius * math.cos(phi), hole_radius * math.sin(phi), z))))
    if terminal_wall:
        for xi, x in enumerate(xs):
            for zi, z in enumerate(np.arange(-4.0, 4.5, 1.0)):
                records.append(((2, xi, zi), np.asarray((x, 32.0, z))))
    return _jitter_delete(token, records)


def full_cylinders() -> np.ndarray:
    return extruded_profile(token="LEPG_SF06", radii=(8.0, 16.0), angle_degrees=360.0, continuation_length=0.0, axial_low=-32.0, axial_high=32.0)
