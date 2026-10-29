#!/usr/bin/env python3
"""Build V31 DHFF analytic source-free fixtures and raw visual evidence.

The builder never imports or invokes the DHFF mechanism.  Every fixture is an
explicit sampled surface with normals, parent membership, stable keys, and
optional mesh adjacency.  No labels, scan identities, historical outputs, or
expected counts are supplied to inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v31_dhff_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v31-dhff-v1"
FIXTURE_ROOT = OUTPUT_ROOT / "source_free_fixtures"
RAW_VISUAL_ROOT = OUTPUT_ROOT / "gate1_raw_visuals"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
HARNESS_MANIFEST_PATH = OUTPUT_ROOT / "gate1_harness_manifest.json"
VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "gate1_raw_visual_manifest.json"
STRESS_PATH = FIXTURE_ROOT / "STRESS_200K_128.json"
HARNESS_VERSION = 1
S = 1.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_v31_dhff_source_free import render_raw_fixture  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _values(lower: float, upper: float, step: float) -> np.ndarray:
    return np.linspace(lower, upper, int(round(abs(upper - lower) / step)) + 1, dtype=np.float64)


@dataclass
class Surface:
    positions: list[list[float]]
    normals: list[list[float]]
    parents: list[int]
    stable_keys: list[int]
    edges: list[list[int]]

    @classmethod
    def empty(cls) -> "Surface":
        return cls([], [], [], [], [])

    def append(self, position: np.ndarray, normal: np.ndarray, parent: int, stable_key: int) -> int:
        index = len(self.positions)
        self.positions.append(np.asarray(position, dtype=np.float64).tolist())
        unit = np.asarray(normal, dtype=np.float64)
        unit /= np.linalg.norm(unit)
        self.normals.append(unit.tolist())
        self.parents.append(int(parent))
        self.stable_keys.append(int(stable_key))
        return index

    def extend(self, other: "Surface", translation: np.ndarray | None = None, key_offset: int = 0) -> None:
        base = len(self.positions)
        shift = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        for position, normal, parent, key in zip(other.positions, other.normals, other.parents, other.stable_keys):
            self.append(np.asarray(position) + shift, np.asarray(normal), parent, key_offset + key)
        self.edges.extend([[base + int(a), base + int(b)] for a, b in other.edges])

    def payload(self) -> dict[str, Any]:
        return {
            "positions": self.positions,
            "source_normals": self.normals,
            "parent_fragment_ids": self.parents,
            "stable_source_keys": self.stable_keys,
            "mesh_adjacencies": self.edges,
        }


def _smooth_angle(alpha_deg: float, t: float) -> float:
    z = min(1.0, max(0.0, t))
    return math.radians(alpha_deg) * (10.0 * z**3 - 15.0 * z**4 + 6.0 * z**5)


def _nonmonotone_angle(t: float) -> float:
    keys = ((0.0, 0.0), (0.35, 70.0), (0.65, 35.0), (1.0, 90.0))
    for (x0, y0), (x1, y1) in zip(keys[:-1], keys[1:]):
        if t <= x1:
            f = (t - x0) / (x1 - x0)
            return math.radians((1.0 - f) * y0 + f * y1)
    return math.radians(90.0)


def _cross_section(alpha_deg: float, angle_fn: Callable[[float], float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angle_fn = angle_fn or (lambda t: _smooth_angle(alpha_deg, t))
    cross: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    parents: list[int] = []
    stages: list[int] = []
    for distance in _values(14.0, 0.0, 0.5)[:-1]:
        cross.append(np.asarray([-distance, 0.0]))
        normals.append(np.asarray([0.0, 1.0]))
        parents.append(1)
        stages.append(0)
    position = np.asarray([0.0, 0.0])
    transition_s = _values(0.0, 8.0, 0.5)
    previous_theta = angle_fn(0.0)
    cross.append(position.copy())
    normals.append(np.asarray([math.sin(previous_theta), math.cos(previous_theta)]))
    parents.append(1)
    stages.append(1)
    for left, right in zip(transition_s[:-1], transition_s[1:]):
        theta0 = angle_fn(left / 8.0)
        theta1 = angle_fn(right / 8.0)
        theta_mid = 0.5 * (theta0 + theta1)
        position = position + (right - left) * np.asarray([math.cos(theta_mid), -math.sin(theta_mid)])
        cross.append(position.copy())
        normals.append(np.asarray([math.sin(theta1), math.cos(theta1)]))
        parents.append(1 if right <= 4.0 else 2)
        stages.append(1)
    end = position.copy()
    final_theta = angle_fn(1.0)
    tangent = np.asarray([math.cos(final_theta), -math.sin(final_theta)])
    final_normal = np.asarray([math.sin(final_theta), math.cos(final_theta)])
    for distance in _values(0.5, 14.0, 0.5):
        cross.append(end + distance * tangent)
        normals.append(final_normal.copy())
        parents.append(2)
        stages.append(2)
    return np.asarray(cross), np.asarray(normals), np.asarray(parents), np.asarray(stages)


def _extrude(
    cross: np.ndarray,
    normals_2d: np.ndarray,
    parents: np.ndarray,
    stages: np.ndarray,
    *,
    mesh: bool,
    fragmented_stations: set[int] | None = None,
    active_transition_stations: set[int] | None = None,
    include_opposite_skin: bool = True,
    deterministic_delete_ten_percent: bool = False,
) -> Surface:
    fragmented_stations = fragmented_stations or set()
    y_values = _values(-16.0, 16.0, 0.5)
    station_centers = [-12.0 + 2.4 * j for j in range(11)]
    surface = Surface.empty()
    skins = (0, 1) if include_opposite_skin else (0,)
    grids: dict[tuple[int, int, int], int] = {}
    for skin in skins:
        for yi, y in enumerate(y_values):
            station_hits = {j for j, center in enumerate(station_centers) if abs(float(y) - center) <= 0.75}
            for ci, point_2d in enumerate(cross):
                if stages[ci] == 1 and station_hits.intersection(fragmented_stations):
                    continue
                if stages[ci] == 1 and active_transition_stations is not None and station_hits and not station_hits.intersection(active_transition_stations):
                    continue
                if deterministic_delete_ten_percent and stages[ci] == 1 and (yi * 131 + ci * 17 + skin * 7) % 10 == 0:
                    continue
                normal = np.asarray([normals_2d[ci, 0], 0.0, normals_2d[ci, 1]])
                position = np.asarray([point_2d[0], y, point_2d[1]])
                displacement = 0.04 * math.sin(0.71 * yi + 1.13 * ci)
                position = position + displacement * normal
                if skin == 1:
                    position = position - 1.5 * normal
                    normal = -normal
                key = 1 + skin * 10_000_000 + yi * 10_000 + ci
                grids[(skin, yi, ci)] = surface.append(position, normal, int(parents[ci]), key)
    if mesh:
        for skin in skins:
            for yi in range(len(y_values)):
                for ci in range(len(cross)):
                    current = grids.get((skin, yi, ci))
                    if current is None:
                        continue
                    for neighbor_key in ((skin, yi + 1, ci), (skin, yi, ci + 1)):
                        neighbor = grids.get(neighbor_key)
                        if neighbor is not None:
                            surface.edges.append([current, neighbor])
    return surface


def _candidate(candidate_id: int, alpha_deg: float, cross: np.ndarray, normals: np.ndarray, *, axis: np.ndarray | None = None, center: np.ndarray | None = None) -> dict[str, Any]:
    axis_value = np.asarray([0.0, 1.0, 0.0]) if axis is None else np.asarray(axis, dtype=np.float64)
    axis_value /= np.linalg.norm(axis_value)
    transition_mid = cross[28 + 8]
    center_value = np.asarray([transition_mid[0], 0.0, transition_mid[1]]) if center is None else np.asarray(center, dtype=np.float64)
    end_index = int(np.flatnonzero(np.arange(len(cross)) >= 28 + 16)[0])
    end = cross[end_index]
    theta = math.radians(alpha_deg)
    n_a = np.asarray([0.0, 0.0, 1.0])
    n_b = np.asarray([math.sin(theta), 0.0, math.cos(theta)])
    tangent_b = np.asarray([math.cos(theta), 0.0, -math.sin(theta)])
    return {
        "candidate_id": int(candidate_id),
        "frozen_v16_order": int(candidate_id % 1_000_000),
        "selected_legacy": False,
        "rejected_before_final_nms": True,
        "render_admissible": True,
        "center_xyz": center_value.tolist(),
        "unit_axis_xyz": axis_value.tolist(),
        "segment_length_L": 30.0,
        "local_scale_s": S,
        "signed_turn_alpha": float(alpha_deg),
        "parent_A": {"parent_id": 1, "unit_normal_xyz": n_a.tolist(), "offset": 0.0, "centroid_xyz": [-8.0, 0.0, 0.0]},
        "parent_B": {"parent_id": 2, "unit_normal_xyz": n_b.tolist(), "offset": float(np.dot(n_b, np.asarray([end[0], 0.0, end[1]]))), "centroid_xyz": (np.asarray([end[0], 0.0, end[1]]) + 8.0 * tangent_b).tolist()},
        "render_line_start": (center_value - 15.0 * axis_value).tolist(),
        "render_line_end": (center_value + 15.0 * axis_value).tolist(),
    }


def _payload(fixture_id: str, surface: Surface, candidates: list[dict[str, Any]], construction: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "v31_dhff_source_free_fixture",
        "fixture_id": fixture_id,
        "source_free": True,
        "harness_version": HARNESS_VERSION,
        "construction": construction,
        "source": surface.payload(),
        "candidate_records": candidates,
    }


def _fold_payload(fixture_id: str, alpha: float, *, mesh: bool, fragmented: bool = False, active_transition_stations: set[int] | None = None, nonmonotone: bool = False, aliases: int = 1) -> dict[str, Any]:
    angle_fn = _nonmonotone_angle if nonmonotone else None
    cross, normals, parents, stages = _cross_section(alpha, angle_fn)
    surface = _extrude(
        cross,
        normals,
        parents,
        stages,
        mesh=mesh,
        fragmented_stations={1, 2, 3} if fragmented else set(),
        active_transition_stations=active_transition_stations,
        deterministic_delete_ten_percent=fragmented,
    )
    base = int(hashlib.sha256(fixture_id.encode("ascii")).hexdigest()[:12], 16)
    candidates = [_candidate(base + index, alpha, cross, normals) for index in range(aliases)]
    for order, candidate in enumerate(candidates):
        candidate["frozen_v16_order"] = 100 + order
    return _payload(fixture_id, surface, candidates, f"analytic {alpha:g} degree noncircular C2 fold; 14s tails; 8s quintic transition; 32s axial extent; 1.5s opposite skin" + ("; transition absent in stations 1-3 plus deterministic 10% deletion" if fragmented else ""))


def _plane_grid(origin: np.ndarray, transverse: np.ndarray, normal: np.ndarray, parent: int, y_values: np.ndarray, distances: np.ndarray, key_base: int) -> Surface:
    surface = Surface.empty()
    grid: dict[tuple[int, int], int] = {}
    for yi, y in enumerate(y_values):
        for di, distance in enumerate(distances):
            position = origin + distance * transverse + y * np.asarray([0.0, 1.0, 0.0])
            grid[(yi, di)] = surface.append(position, normal, parent, key_base + yi * 10_000 + di)
    for yi in range(len(y_values)):
        for di in range(len(distances)):
            for neighbor_key in ((yi + 1, di), (yi, di + 1)):
                if neighbor_key in grid:
                    surface.edges.append([grid[(yi, di)], grid[neighbor_key]])
    return surface


def _single_tail(fixture_id: str, missing: str) -> dict[str, Any]:
    cross, normals, _parents, _stages = _cross_section(90.0)
    y = _values(-16.0, 16.0, 0.5)
    surface = Surface.empty()
    if missing != "A":
        surface.extend(_plane_grid(np.zeros(3), np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), 1, y, _values(0.0, 14.0, 0.5), 1))
    if missing != "B":
        theta = math.pi / 2.0
        end = cross[44]
        surface.extend(_plane_grid(np.asarray([end[0], 0.0, end[1]]), np.asarray([math.cos(theta), 0.0, -math.sin(theta)]), np.asarray([math.sin(theta), 0.0, math.cos(theta)]), 2, y, _values(0.0, 14.0, 0.5), 2_000_000))
    candidate = _candidate(int(hashlib.sha256(fixture_id.encode()).hexdigest()[:12], 16), 90.0, cross, normals)
    return _payload(fixture_id, surface, [candidate], f"single broad material tail; tail {missing} is scan-truncated or absent")


def _opening_rim() -> dict[str, Any]:
    """Planar sheet with a rounded edge-opening, but no independent B tail.

    The previous SN02 construction reused the silhouette fixture verbatim.  A
    manual native-resolution review caught that it therefore did not visibly
    instantiate the named nuisance.  This construction cuts a circular notch
    into the observed edge while retaining ample parent-A material behind the
    rim; the candidate still has no parent-B tail by construction.
    """
    fixture_id = "SN02_OPENING_RIM"
    cross, normals, _parents, _stages = _cross_section(90.0)
    x_values = _values(-14.0, 0.0, 0.5)
    y_values = _values(-16.0, 16.0, 0.5)
    surface = Surface.empty()
    grid: dict[tuple[int, int], int] = {}
    notch_center = np.asarray([-1.0, 0.0])
    notch_radius = 4.0
    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            # A circle intersecting x=0 creates a plainly visible rounded
            # opening/rim rather than another straight silhouette edge.
            if float(np.linalg.norm(np.asarray([x, y]) - notch_center)) < notch_radius:
                continue
            grid[(yi, xi)] = surface.append(
                np.asarray([x, y, 0.0]),
                np.asarray([0.0, 0.0, 1.0]),
                1,
                1 + yi * 10_000 + xi,
            )
    for yi in range(len(y_values)):
        for xi in range(len(x_values)):
            current = grid.get((yi, xi))
            if current is None:
                continue
            for neighbor_key in ((yi + 1, xi), (yi, xi + 1)):
                neighbor = grid.get(neighbor_key)
                if neighbor is not None:
                    surface.edges.append([current, neighbor])
    candidate = _candidate(31002, 90.0, cross, normals, center=np.zeros(3))
    candidate["parent_A"]["centroid_xyz"] = [-9.0, 0.0, 0.0]
    return _payload(
        fixture_id,
        surface,
        [candidate],
        "broad planar parent-A sheet with a radius-4s circular notch intersecting its observed edge; visible rounded opening rim; no independent parent-B tail",
    )


def _thickness_return() -> dict[str, Any]:
    fixture_id = "SN03_THICKNESS_RETURN_1P5S"
    cross, normals, _parents, _stages = _cross_section(90.0)
    y = _values(-16.0, 16.0, 0.5)
    surface = _plane_grid(np.zeros(3), np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), 1, y, _values(0.0, 14.0, 0.5), 1)
    short = _plane_grid(np.zeros(3), np.asarray([0.0, 0.0, -1.0]), np.asarray([1.0, 0.0, 0.0]), 2, y, _values(0.0, 1.5, 0.5), 2_000_000)
    surface.extend(short)
    return _payload(fixture_id, surface, [_candidate(31003, 90.0, cross, normals)], "broad parent A with only a 1.5s thickness return for parent B")


def _disconnected(fixture_id: str, gap: float, construction: str) -> dict[str, Any]:
    cross, normals, _parents, _stages = _cross_section(90.0)
    y = _values(-16.0, 16.0, 0.5)
    surface = Surface.empty()
    surface.extend(_plane_grid(np.asarray([-gap / 2.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), 1, y, _values(0.0, 14.0, 0.5), 1))
    surface.extend(_plane_grid(np.asarray([gap / 2.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -1.0]), np.asarray([1.0, 0.0, 0.0]), 2, y, _values(0.0, 14.0, 0.5), 2_000_000))
    candidate = _candidate(int(hashlib.sha256(fixture_id.encode()).hexdigest()[:12], 16), 90.0, cross, normals, center=np.zeros(3))
    candidate["parent_A"]["offset"] = 0.0
    candidate["parent_B"]["offset"] = gap / 2.0
    candidate["parent_A"]["centroid_xyz"] = [-8.0 - gap / 2.0, 0.0, 0.0]
    candidate["parent_B"]["centroid_xyz"] = [gap / 2.0, 0.0, -8.0]
    return _payload(fixture_id, surface, [candidate], construction)


def _cycle_fixture() -> dict[str, Any]:
    fixture_id = "SN08_CORNER_CYCLE_TWO_ROUTE"
    primary = _fold_payload("_PRIMARY", 90.0, mesh=True)
    surface = Surface.empty()
    first = Surface(
        primary["source"]["positions"], primary["source"]["source_normals"], primary["source"]["parent_fragment_ids"], primary["source"]["stable_source_keys"], primary["source"]["mesh_adjacencies"]
    )
    surface.extend(first)
    # A second conducting strip is displaced 1.8s in the local depth direction.
    # Its normals follow a reverse/overshoot route (-35, 125, 90 degrees), so
    # appreciable parallel current is deliberately outside the target arc.
    cross, normals, parents, stages = _cross_section(90.0, lambda t: math.radians(-35.0 * math.sin(math.pi * t) + 90.0 * t + 70.0 * math.sin(math.pi * t) ** 2))
    secondary = _extrude(cross, normals, parents, stages, mesh=True, include_opposite_skin=False)
    secondary_positions = np.asarray(secondary.positions, dtype=np.float64)
    secondary_positions[:, 0] += 1.8
    secondary.positions = secondary_positions.tolist()
    surface.extend(secondary, key_offset=30_000_000)
    base_cross, base_normals, _p, _s = _cross_section(90.0)
    candidate = _candidate(31008, 90.0, base_cross, base_normals)
    return _payload(fixture_id, surface, [candidate], "two parallel observed-surface routes between the same broad tails; secondary route reverses and overshoots normal space")


def _flat_false_crease() -> dict[str, Any]:
    fixture_id = "SN07_FLAT_FALSE_CREASE12"
    cross, normals, _p, _s = _cross_section(12.0)
    surface = _plane_grid(np.zeros(3), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), 1, _values(-16.0, 16.0, 0.5), _values(-14.0, 14.0, 0.5), 1)
    return _payload(fixture_id, surface, [_candidate(31007, 12.0, cross, normals)], "single flat panel with a frozen 12 degree false-crease candidate")


def _fixture_builders() -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "SF01_FOLD60_MESH_C2": lambda: _fold_payload("SF01_FOLD60_MESH_C2", 60.0, mesh=True),
        "SF02_FOLD90_MESH_C2": lambda: _fold_payload("SF02_FOLD90_MESH_C2", 90.0, mesh=True),
        "SF03_FOLD120_MESH_C2": lambda: _fold_payload("SF03_FOLD120_MESH_C2", 120.0, mesh=True),
        "SF04_FOLD60_POINT_C2": lambda: _fold_payload("SF04_FOLD60_POINT_C2", 60.0, mesh=False),
        "SF05_FOLD90_POINT_C2": lambda: _fold_payload("SF05_FOLD90_POINT_C2", 90.0, mesh=False),
        "SF06_FOLD120_POINT_C2": lambda: _fold_payload("SF06_FOLD120_POINT_C2", 120.0, mesh=False),
        "SF07_FOLD60_FRAGMENTED": lambda: _fold_payload("SF07_FOLD60_FRAGMENTED", 60.0, mesh=True, fragmented=True),
        "SF08_FOLD90_FRAGMENTED": lambda: _fold_payload("SF08_FOLD90_FRAGMENTED", 90.0, mesh=True, fragmented=True),
        "SF09_FOLD120_FRAGMENTED": lambda: _fold_payload("SF09_FOLD120_FRAGMENTED", 120.0, mesh=True, fragmented=True),
        "SN01_SILHOUETTE_FREE_EDGE": lambda: _single_tail("SN01_SILHOUETTE_FREE_EDGE", "B"),
        "SN02_OPENING_RIM": _opening_rim,
        "SN03_THICKNESS_RETURN_1P5S": _thickness_return,
        "SN04_DISCONNECTED_MITER90": lambda: _disconnected("SN04_DISCONNECTED_MITER90", 3.0, "two broad orthogonal tails separated by a 3s open miter seam"),
        "SN05_LOCAL_ROUNDED_BRIDGE_4P5S": lambda: _fold_payload("SN05_LOCAL_ROUNDED_BRIDGE_4P5S", 90.0, mesh=True, active_transition_stations={3, 4, 5, 6, 7}),
        "SN06_NONMONOTONE_BEAD_0_70_35_90": lambda: _fold_payload("SN06_NONMONOTONE_BEAD_0_70_35_90", 90.0, mesh=True, nonmonotone=True),
        "SN07_FLAT_FALSE_CREASE12": _flat_false_crease,
        "SN08_CORNER_CYCLE_TWO_ROUTE": _cycle_fixture,
        "SN09_NEARBY_NONTOUCHING_PANELS_1P25S": lambda: _disconnected("SN09_NEARBY_NONTOUCHING_PANELS_1P25S", 1.25, "nearby non-touching panels separated through the sheet-normal direction"),
        "SN10_UNSUPPORTED_PLANE_INTERSECTION": lambda: _disconnected("SN10_UNSUPPORTED_PLANE_INTERSECTION", 4.0, "infinite parent planes intersect analytically but observed material stops short"),
        "SN11_SCAN_BOUNDARY_TRUNCATION": lambda: _single_tail("SN11_SCAN_BOUNDARY_TRUNCATION", "A"),
        "SN12_FIVE_DUPLICATE_AXIS_ALIASES": lambda: _fold_payload("SN12_FIVE_DUPLICATE_AXIS_ALIASES", 90.0, mesh=True, aliases=5),
    }


def _stress_fixture() -> dict[str, Any]:
    surface = Surface.empty()
    candidates: list[dict[str, Any]] = []
    key_offset = 0
    for index in range(8):
        fixture = _fold_payload(f"_STRESS_POS_{index}", 90.0, mesh=False)
        local = Surface(
            fixture["source"]["positions"], fixture["source"]["source_normals"], fixture["source"]["parent_fragment_ids"], fixture["source"]["stable_source_keys"], fixture["source"]["mesh_adjacencies"]
        )
        shift = np.asarray([index * 80.0, 0.0, 0.0])
        surface.extend(local, shift, key_offset)
        candidate = json.loads(json.dumps(fixture["candidate_records"][0]))
        for field in ("center_xyz", "render_line_start", "render_line_end"):
            candidate[field] = (np.asarray(candidate[field]) + shift).tolist()
        for parent in ("parent_A", "parent_B"):
            candidate[parent]["centroid_xyz"] = (np.asarray(candidate[parent]["centroid_xyz"]) + shift).tolist()
            candidate[parent]["offset"] = float(candidate[parent]["offset"] + np.dot(np.asarray(candidate[parent]["unit_normal_xyz"]), shift))
        candidate["candidate_id"] = 90_000 + index
        candidate["frozen_v16_order"] = index
        candidates.append(candidate)
        key_offset += 20_000_000
    cross, normals, _p, _s = _cross_section(90.0)
    for index in range(120):
        candidate = _candidate(91_000 + index, 90.0, cross, normals, center=np.asarray([2000.0 + 50.0 * index, 0.0, 0.0]))
        candidate["frozen_v16_order"] = 8 + index
        candidates.append(candidate)
    remaining = 200_000 - len(surface.positions)
    if remaining < 0:
        raise RuntimeError("stress positive sample count exceeds 200000")
    for index in range(remaining):
        x = 20_000.0 + (index % 400) * 3.0
        y = (index // 400) * 3.0
        surface.append(np.asarray([x, y, 5_000.0]), np.asarray([0.0, 0.0, 1.0]), 999, 800_000_000 + index)
    if len(surface.positions) != 200_000 or len(candidates) != 128:
        raise RuntimeError("stress cardinality drift")
    return _payload("STRESS_200K_128", surface, candidates, "exact 200000-sample, 128-candidate stress: eight folds and 120 empty-location negatives")


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload["source"]
    positions = np.asarray(source["positions"], dtype=np.float64)
    normals = np.asarray(source["source_normals"], dtype=np.float64)
    parents = np.asarray(source["parent_fragment_ids"], dtype=np.uint32)
    stable = np.asarray(source["stable_source_keys"], dtype=np.uint64)
    edges = np.asarray(source["mesh_adjacencies"], dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    checks = {
        "finite": bool(np.isfinite(positions).all() and np.isfinite(normals).all()),
        "schema_lengths": bool(len(positions) == len(normals) == len(parents) == len(stable)),
        "unit_normals": bool(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-12)),
        "stable_keys_unique": bool(len(np.unique(stable)) == len(stable)),
        "edges_valid": bool(edges.shape[1:] == (2,) and (not len(edges) or (edges.min() >= 0 and edges.max() < len(positions)))),
        "source_cap": bool(len(positions) <= 200_000),
        "candidate_cap": bool(len(payload["candidate_records"]) <= 128),
        "candidate_lengths": bool(all(float(value["segment_length_L"]) == 30.0 for value in payload["candidate_records"])),
    }
    return {"passed": all(checks.values()), "checks": checks, "samples": len(positions), "edges": len(edges), "candidates": len(payload["candidate_records"])}


def _invocation_count() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "DHFF_INVOKE")


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if _invocation_count() != 0:
        raise SystemExit("post-invocation fixture rebuild is forbidden")
    builders = _fixture_builders()
    if tuple(builders) != tuple(lock["fixture_ids"]):
        raise SystemExit("fixture order differs from immutable lock")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    COUNTER_PATH.touch(exist_ok=True)
    fixture_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for fixture_id, builder in builders.items():
        payload = builder()
        validation = _validate(payload)
        if not validation["passed"]:
            raise SystemExit(f"fixture validation failed: {fixture_id}: {validation}")
        path = FIXTURE_ROOT / f"{fixture_id}.json"
        _write_json(path, payload)
        visual_dir = RAW_VISUAL_ROOT / fixture_id
        multiview, sections = render_raw_fixture(payload, visual_dir)
        fixture_rows.append({"fixture_id": fixture_id, "path": str(path.relative_to(ROOT)), "sha256": _sha(path), "validation": validation})
        for kind, visual_path in (("raw_native_multiview_4096", multiview), ("raw_station_sections_4096", sections)):
            visual_rows.append({"fixture_id": fixture_id, "kind": kind, "path": str(visual_path.relative_to(ROOT)), "sha256": _sha(visual_path), "width": 4096, "height": 4096, "signed": False, "decision": None})
    stress = _stress_fixture()
    stress_validation = _validate(stress)
    if not stress_validation["passed"] or stress_validation["samples"] != 200_000 or stress_validation["candidates"] != 128:
        raise SystemExit(f"stress validation failed: {stress_validation}")
    _write_json(STRESS_PATH, stress)
    fixture_rows.append({"fixture_id": "STRESS_200K_128", "path": str(STRESS_PATH.relative_to(ROOT)), "sha256": _sha(STRESS_PATH), "validation": stress_validation})
    harness = {"kind": "v31_dhff_gate1_harness_manifest", "harness_version": HARNESS_VERSION, "mechanism_invocation_count": 0, "passed_analytic_checks": True, "raw_visuals_signed": False, "fixture_rows": fixture_rows, "token": lock["tokens"]["preinvocation_retire"]}
    visual_manifest = {"kind": "v31_dhff_gate1_raw_visual_manifest", "harness_version": HARNESS_VERSION, "all_signed": False, "rows": visual_rows}
    _write_json(HARNESS_MANIFEST_PATH, harness)
    _write_json(VISUAL_MANIFEST_PATH, visual_manifest)
    print(json.dumps({"physical_fixtures": len(builders), "stress_samples": 200_000, "raw_visuals": len(visual_rows), "invocation_count": 0, "status": "AWAITING_MANUAL_RAW_VISUAL_SIGNOFF"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
