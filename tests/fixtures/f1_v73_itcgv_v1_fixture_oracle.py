"""Independent V73 source-free fixture generator and arithmetic oracle.

This module intentionally imports no production V73 code and duplicates no
production predicate implementation.  It provides literal geometric fixtures
and exact expected outcomes for the frozen 32-case lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FixtureCase:
    identifier: str
    source: np.ndarray | None
    candidates: tuple[dict[str, Any], ...]
    expected: str
    reference: str | None = None


DEFAULT_X = np.arange(-100.0, 101.0, 1.0, dtype=np.float64)
DEFAULT_Y = np.arange(-160.0, 161.0, 1.0, dtype=np.float64)


def _surface(z: np.ndarray, x: np.ndarray = DEFAULT_X, y: np.ndarray = DEFAULT_Y) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.broadcast_to(np.asarray(z, dtype=np.float64), yy.shape)
    return np.ascontiguousarray(np.column_stack((xx.ravel(), yy.ravel(), values.ravel())))


def _surface_function(function, x: np.ndarray = DEFAULT_X, y: np.ndarray = DEFAULT_Y) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.asarray(function(yy), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))


def _profile_extrusion(profile: np.ndarray, x: np.ndarray = DEFAULT_X) -> np.ndarray:
    repeated = np.tile(np.asarray(profile, dtype=np.float64), (len(x), 1))
    axial = np.repeat(x, len(profile))
    return np.ascontiguousarray(np.column_stack((axial, repeated)))


def _candidate(
    start=(-96.0, 0.0, 0.0),
    end=(96.0, 0.0, 0.0),
    *,
    spacing=1.0,
    anchor=None,
    axis=None,
    extra=False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "line_start": [float(value) for value in start],
        "line_end": [float(value) for value in end],
        "spacing": float(spacing),
    }
    if anchor is not None:
        row["line_anchor"] = [float(value) for value in anchor]
    if axis is not None:
        row["line_axis"] = [float(value) for value in axis]
    if extra:
        row.update(
            {
                "candidate_opaque_hash": "forbidden-and-ignored",
                "filename": "must-not-serialize.ply",
                "score": 999999.0,
                "expected_count": 123,
                "selection_route": "must-not-be-read",
            }
        )
    return row


def _sharp(slope: float, sign: float = 1.0) -> np.ndarray:
    return _surface_function(lambda y: sign * slope * np.abs(y))


def _rounded(radius: float) -> np.ndarray:
    return _surface_function(lambda y: np.sqrt(y * y + radius * radius) - radius)


def _u_return() -> np.ndarray:
    lower_y = np.arange(-160.0, 1.0, 1.0)
    lower = np.column_stack((lower_y, np.zeros_like(lower_y)))
    bridge_z = np.arange(1.0, 16.0, 1.0)
    bridge = np.column_stack((np.zeros_like(bridge_z), bridge_z))
    upper_y = np.arange(0.0, 161.0, 1.0)
    upper = np.column_stack((upper_y, np.full_like(upper_y, 16.0)))
    return _profile_extrusion(np.vstack((lower, bridge, upper)))


def _two_fold_source() -> np.ndarray:
    pieces = []
    for center in (-12.0, 12.0):
        pieces.append(
            _surface_function(
                lambda q, center=center: (7.0 / 128.0) * np.abs(q - center)
            )
        )
    return np.ascontiguousarray(np.vstack(pieces))


def _two_fold_candidates() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _candidate(start=(-96.0, -12.0, 0.0), end=(96.0, -12.0, 0.0)),
        _candidate(start=(-96.0, 12.0, 0.0), end=(96.0, 12.0, 0.0)),
    )


def _default_roll() -> np.ndarray:
    return _surface_function(lambda y: y * y / 4096.0)


def _small_roll() -> tuple[np.ndarray, dict[str, Any]]:
    x = np.arange(-40.0, 41.0, 1.0)
    source = _surface_function(lambda y: y * y / 4096.0, x=x)
    return source, _candidate(start=(-36.0, 0.0, 0.0), end=(36.0, 0.0, 0.0))


def _permutation(size: int) -> np.ndarray:
    values = np.asarray([(65537 * index + 17) % size for index in range(size)], dtype=np.int64)
    if len(np.unique(values)) != size:
        raise AssertionError("fixed mapping is not a permutation for this fixture")
    return values


def _signed_transform(points: np.ndarray) -> np.ndarray:
    clean = np.asarray(points, dtype=np.float64)
    return np.ascontiguousarray(
        np.column_stack(
            (
                clean[:, 2] + 1024.0,
                -clean[:, 0] - 2048.0,
                clean[:, 1] + 4096.0,
            )
        )
    )


def _transform_point(point) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in point)
    return (z + 1024.0, -x - 2048.0, y + 4096.0)


def build_case(identifier: str) -> FixtureCase:
    if identifier == "G01":
        return FixtureCase(identifier, _surface_function(lambda y: 0.0 * y), (_candidate(),), "VETO_FLAT")
    if identifier == "G02":
        y = np.arange(0.0, 161.0, 1.0)
        return FixtureCase(identifier, _surface_function(lambda value: 0.0 * value, y=y), (_candidate(),), "VETO_FLAT")
    if identifier == "G03":
        return FixtureCase(identifier, _default_roll(), (_candidate(),), "VETO_GROWTH")
    if identifier == "G04":
        source, row = _small_roll()
        return FixtureCase(identifier, np.repeat(source, 7, axis=0), (row,), "MATCH_SMALL_ROLL", "G03_SMALL")
    if identifier == "G05":
        return FixtureCase(identifier, _sharp(7.0 / 128.0), (_candidate(),), "KEEP")
    if identifier == "G06":
        return FixtureCase(identifier, _sharp(1.0), (_candidate(),), "KEEP")
    if identifier == "G07":
        return FixtureCase(identifier, _rounded(64.0), (_candidate(),), "KEEP")
    if identifier == "G08":
        return FixtureCase(identifier, _rounded(16.0), (_candidate(),), "KEEP")
    if identifier == "G09":
        return FixtureCase(identifier, _u_return(), (_candidate(anchor=(0.0, 0.0, 8.0)),), "KEEP")
    if identifier == "G10":
        opposing = np.vstack((_sharp(1.0), _sharp(1.0, -1.0)))
        return FixtureCase(identifier, np.ascontiguousarray(opposing), (_candidate(),), "KEEP")
    if identifier == "G11":
        first = _surface_function(lambda y: (7.0 / 128.0) * y)
        second = _surface_function(lambda y: -(7.0 / 128.0) * y)
        return FixtureCase(identifier, np.ascontiguousarray(np.vstack((first, second))), (_candidate(),), "KEEP")
    if identifier == "G12":
        x = np.arange(-100.0, 5.0, 1.0)
        source = _surface_function(lambda y: np.abs(y), x=x)
        row = _candidate(start=(-96.0, 0.0, 0.0), end=(0.0, 0.0, 0.0))
        return FixtureCase(identifier, source, (row,), "KEEP")
    if identifier == "G13":
        return FixtureCase(identifier, _sharp(1.0), (_candidate(start=(-16.0, 0.0, 0.0), end=(16.0, 0.0, 0.0)),), "KEEP_SHORT")
    if identifier in {"G14", "G15"}:
        return FixtureCase(identifier, _two_fold_source(), _two_fold_candidates(), "CANDIDATE_INDEPENDENCE")
    if identifier == "G16":
        row = _two_fold_candidates()[0]
        return FixtureCase(identifier, _two_fold_source(), (row, dict(row)), "DUPLICATE_CANDIDATE")
    if identifier == "G17":
        return FixtureCase(identifier, _surface_function(lambda y: -y / 64.0), (_candidate(),), "VETO_FLAT")
    if identifier == "G18":
        return FixtureCase(identifier, _sharp(7.0 / 128.0, -1.0), (_candidate(),), "KEEP")
    if identifier == "G19":
        base = _sharp(7.0 / 128.0)
        duplicate_mask = np.equal(np.mod(base[:, 1], 4.0), 0.0)
        duplicated = np.repeat(base[duplicate_mask], 4, axis=0)
        return FixtureCase(identifier, np.ascontiguousarray(np.vstack((base, duplicated))), (_candidate(),), "MATCH_G05", "G05")
    if identifier == "G20":
        return FixtureCase(identifier, np.ascontiguousarray(_default_roll()[::-1]), (_candidate(),), "MATCH_G03", "G03")
    if identifier == "G21":
        base = _rounded(64.0)
        return FixtureCase(identifier, np.ascontiguousarray(base[_permutation(len(base))]), (_candidate(),), "MATCH_G07", "G07")
    if identifier == "G22":
        source = _signed_transform(_default_roll())
        start = _transform_point((-96.0, 0.0, 0.0))
        end = _transform_point((96.0, 0.0, 0.0))
        return FixtureCase(identifier, source, (_candidate(start=start, end=end),), "MATCH_G03", "G03")
    if identifier == "G23":
        source = np.ascontiguousarray(_rounded(64.0) * 8.0)
        row = _candidate(start=(-768.0, 0.0, 0.0), end=(768.0, 0.0, 0.0), spacing=8.0)
        return FixtureCase(identifier, source, (row,), "MATCH_G07", "G07")
    if identifier == "G24":
        shifted = _candidate(anchor=(37.0, 0.0, 0.0), axis=(-1.0, 0.0, 0.0))
        return FixtureCase(identifier, _rounded(64.0), (shifted,), "MATCH_GEOMETRY_G07", "G07")
    if identifier in {"G25", "G26", "G27"}:
        return FixtureCase(identifier, None, tuple(), "BOUNDARY_ORACLE")
    if identifier == "G28":
        index = np.arange(200_000, dtype=np.float64)
        source = np.column_stack((index, np.mod(index, 17.0), np.mod(index, 31.0)))
        row = _candidate(start=(0.0, 0.0, 0.0), end=(32.0, 0.0, 0.0))
        return FixtureCase(identifier, source, (row,), "SOURCE_CAP_ACCEPT")
    if identifier == "G29":
        index = np.arange(200_001, dtype=np.float64)
        source = np.column_stack((index, np.mod(index, 17.0), np.mod(index, 31.0)))
        row = _candidate(start=(0.0, 0.0, 0.0), end=(32.0, 0.0, 0.0))
        return FixtureCase(identifier, source, (row,), "SOURCE_CAP_ABORT")
    if identifier == "G30":
        source = _surface_function(lambda y: 0.0 * y, x=np.arange(-20.0, 21.0), y=np.arange(-8.0, 9.0))
        row = _candidate(start=(-16.0, 0.0, 0.0), end=(16.0, 0.0, 0.0))
        return FixtureCase(identifier, source, tuple(dict(row) for _ in range(64)), "CANDIDATE_CAP_ACCEPT")
    if identifier == "G31":
        source = _surface_function(lambda y: 0.0 * y, x=np.arange(-20.0, 21.0), y=np.arange(-8.0, 9.0))
        row = _candidate(start=(-16.0, 0.0, 0.0), end=(16.0, 0.0, 0.0))
        return FixtureCase(identifier, source, tuple(dict(row) for _ in range(65)), "CANDIDATE_CAP_ABORT")
    if identifier == "G32":
        source = _surface_function(lambda y: 0.0 * y, x=np.arange(-20.0, 21.0), y=np.arange(-8.0, 9.0))
        row = _candidate(start=(-16.0, 0.0, 0.0), end=(16.0, 0.0, 0.0), extra=True)
        return FixtureCase(identifier, source, (row,), "SERIALIZATION")
    raise KeyError(identifier)


def small_roll_reference() -> FixtureCase:
    source, row = _small_roll()
    return FixtureCase("G03_SMALL", source, (row,), "VETO_GROWTH")


def literal_radial_scale0_oracle() -> tuple[tuple[float, bool], ...]:
    return (
        (math.nextafter(8.0, -math.inf), True),
        (8.0, True),
        (math.nextafter(8.0, math.inf), False),
    )


def literal_axial_oracle() -> tuple[tuple[float, bool], ...]:
    return (
        (math.nextafter(4.0, -math.inf), True),
        (4.0, True),
        (math.nextafter(4.0, math.inf), False),
    )


def literal_comparator_oracle() -> dict[str, tuple[tuple[float, ...], bool]]:
    unit = 1.0
    flat = 2.0**-13
    inner = 2.0**-24
    outer = 0.5
    psd = 2.0**-40
    return {
        "flat_below": ((math.nextafter(flat, -math.inf), unit), True),
        "flat_at": ((flat, unit), True),
        "flat_above": ((math.nextafter(flat, math.inf), unit), False),
        "inner_below": ((math.nextafter(inner, -math.inf), unit), False),
        "inner_at": ((inner, unit), True),
        "inner_above": ((math.nextafter(inner, math.inf), unit), True),
        "outer_below": ((math.nextafter(outer, -math.inf), unit), True),
        "outer_at": ((outer, unit), True),
        "outer_above": ((math.nextafter(outer, math.inf), unit), False),
        "psd_lower_out": ((math.nextafter(-psd, -math.inf), unit), False),
        "psd_lower_at": ((-psd, unit), True),
        "psd_lower_in": ((math.nextafter(-psd, math.inf), unit), True),
        "psd_upper_in": ((math.nextafter((1.0 + psd) / 4.0, -math.inf), unit), True),
        "psd_upper_at": (((1.0 + psd) / 4.0, unit), True),
        "psd_upper_out": ((math.nextafter((1.0 + psd) / 4.0, math.inf), unit), False),
        "growth_below": ((inner, unit, math.nextafter(9.0 * inner / 8.0, -math.inf), unit), False),
        "growth_at": ((inner, unit, 9.0 * inner / 8.0, unit), True),
        "growth_above": ((inner, unit, math.nextafter(9.0 * inner / 8.0, math.inf), unit), True),
        "spread_at": ((2.0, unit, 64.0, 1.0, unit, 64.0), True),
        "spread_out": ((math.nextafter(2.0, math.inf), unit, 64.0, 1.0, unit, 64.0), False),
    }


CASE_IDS = tuple(f"G{index:02d}" for index in range(1, 33))


__all__ = [
    "CASE_IDS",
    "FixtureCase",
    "build_case",
    "literal_axial_oracle",
    "literal_comparator_oracle",
    "literal_radial_scale0_oracle",
    "small_roll_reference",
]
