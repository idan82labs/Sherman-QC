"""Deterministic calibrated multiview turn/change-point consensus (V58).

This research-only implementation consumes organized depth, validity,
intrinsics and calibrated camera poses.  It does not consume labels, CAD,
expected counts, family or part identity.  The constants are fixed before the
prospective synthetic truth is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROFILE = "F1_V58_CALIBRATED_MULTIVIEW_TURN_CONSENSUS_V1"
DIRECTIONS = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1))
MAX_TOKENS_PER_VIEW = 512
MAX_TOKENS_PER_SCAN = 8192
MAX_RAW_LINES = 256
BANK_MEDIAN_RESIDUAL_MAX_DEG = 12.0
BANK_P90_RESIDUAL_MAX_DEG = 20.0
MIN_CONCENTRATION = 0.50


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    for value in result:
        if abs(float(value)) > 1e-12:
            if value < 0.0:
                result *= -1.0
            break
    return result


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 180.0
    cosine = float(np.clip(abs(float(np.dot(a, b))) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _oriented_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 180.0
    cosine = float(np.clip(float(np.dot(a, b)) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.inf
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


@dataclass(frozen=True)
class Frame:
    semantic_hash: str
    depth: np.ndarray
    validity: np.ndarray
    K: np.ndarray
    T: np.ndarray
    points: np.ndarray
    camera: np.ndarray


@dataclass(frozen=True)
class Token:
    point: np.ndarray
    tangent: np.ndarray
    score: float
    alpha: float
    concentration: float
    roughness: float
    view: int
    row: int
    col: int
    direction_ordinal: int


def _semantic_hash(depth: np.ndarray, validity: np.ndarray, K: np.ndarray, T: np.ndarray) -> str:
    digest = hashlib.sha256()
    chunks = (
        np.asarray(depth.shape, dtype=">u8").tobytes(),
        np.ascontiguousarray(depth, dtype=">f8").tobytes(),
        np.ascontiguousarray(validity, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(K, dtype=">f8").tobytes(),
        np.ascontiguousarray(T, dtype=">f8").tobytes(),
    )
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def load_bundle(bundle: Path) -> tuple[np.ndarray, list[Frame], str, str]:
    root = bundle.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "kind", "opaque_case_id", "source_geometry", "frames"}
    if set(manifest) != allowed or manifest.get("schema_version") != 1:
        raise ValueError("unsupported calibrated bundle manifest")
    source_row = manifest["source_geometry"]
    source_path = (root / source_row["file"]).resolve()
    if _sha(source_path) != source_row["sha256"]:
        raise ValueError("source hash mismatch")
    source = np.asarray(o3d.io.read_point_cloud(str(source_path)).points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or not len(source) or not np.isfinite(source).all():
        raise ValueError("invalid source geometry")

    frames: list[Frame] = []
    seen: set[str] = set()
    for ordinal, row in enumerate(manifest["frames"]):
        if row.get("ordinal") != ordinal:
            raise ValueError("noncanonical frame ordinals")
        depth_path = (root / row["depth_file"]).resolve()
        validity_path = (root / row["validity_file"]).resolve()
        if _sha(depth_path) != row["depth_sha256"] or _sha(validity_path) != row["validity_sha256"]:
            raise ValueError("frame hash mismatch")
        depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float64)
        validity = np.asarray(np.load(validity_path, allow_pickle=False), dtype=np.bool_)
        K = np.asarray(row["K"], dtype=np.float64)
        T = np.asarray(row["T_world_from_camera"], dtype=np.float64)
        if depth.ndim != 2 or depth.shape != validity.shape or K.shape != (3, 3) or T.shape != (4, 4):
            raise ValueError("malformed calibrated frame")
        if not np.isfinite(depth[validity]).all() or not np.all(depth[validity] > 0.0) or not np.all(depth[~validity] == 0.0):
            raise ValueError("invalid depth/validity")
        semantic = _semantic_hash(depth, validity, K, T)
        if semantic in seen:
            raise ValueError("duplicate semantic frame")
        seen.add(semantic)
        rows, cols = np.indices(depth.shape)
        camera_xyz = np.stack(
            ((cols - K[0, 2]) * depth / K[0, 0], (rows - K[1, 2]) * depth / K[1, 1], depth),
            axis=-1,
        )
        points = camera_xyz @ T[:3, :3].T + T[:3, 3]
        points[~validity] = np.nan
        frames.append(Frame(semantic, depth, validity, K, T, points, T[:3, 3].copy()))
    frames.sort(key=lambda frame: frame.semantic_hash)
    return source, frames, source_row["sha256"], _sha(manifest_path)


def _stride(frames: list[Frame]) -> int:
    for stride in range(1, 8193):
        # Invalid pixels contain no range observation.  Budget the actual
        # valid organized lattice rather than empty image background so sparse
        # scanner views do not discard two thirds of their geometric samples.
        total = sum(int(np.count_nonzero(frame.validity[::stride, ::stride])) for frame in frames)
        if total <= 200_000:
            return stride
    raise ValueError("pixel budget cannot be satisfied")


def _normal_field(frame: Frame, stride: int, B: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.depth.shape
    normals = np.full((height, width, 3), np.nan, dtype=np.float64)
    usable = np.zeros((height, width), dtype=np.bool_)
    for row, col in np.argwhere(frame.validity):
        row, col = int(row), int(col)
        if row % stride or col % stride:
            continue
        r0, r1 = max(0, row - 3), min(height, row + 4)
        c0, c1 = max(0, col - 3), min(width, col + 4)
        neighbours = frame.points[r0:r1, c0:c1].reshape(-1, 3)
        finite = np.isfinite(neighbours).all(axis=1)
        neighbours = neighbours[finite]
        if len(neighbours) < 12:
            continue
        centre = frame.points[row, col]
        neighbours = neighbours[np.linalg.norm(neighbours - centre, axis=1) <= 0.020 * B]
        if len(neighbours) < 12:
            continue
        centered = neighbours - np.mean(neighbours, axis=0)
        covariance = centered.T @ centered / len(neighbours)
        values, vectors = np.linalg.eigh(covariance)
        total = float(np.sum(values))
        if total <= 0.0 or float(values[0] / total) > 0.05:
            continue
        normal = vectors[:, 0]
        if float(np.dot(normal, frame.camera - centre)) < 0.0:
            normal = -normal
        normals[row, col] = normal
        usable[row, col] = True
    return normals, usable


def _representative(normals: list[np.ndarray]) -> tuple[np.ndarray, float, float] | None:
    if not normals:
        return None
    first = _unit(np.sum(normals, axis=0))
    if first is None:
        return None
    residuals = [_oriented_angle_deg(normal, first) for normal in normals]
    weights = [min(1.0, 5.0 / max(value, 1e-12)) for value in residuals]
    representative = _unit(np.sum([weight * normal for weight, normal in zip(weights, normals)], axis=0))
    if representative is None:
        return None
    residuals = [_oriented_angle_deg(normal, representative) for normal in normals]
    median = float(np.median(residuals))
    p90 = _nearest_rank(residuals, 0.90)
    if median > BANK_MEDIAN_RESIDUAL_MAX_DEG or p90 > BANK_P90_RESIDUAL_MAX_DEG:
        return None
    return representative, median, p90


def _side_samples(
    frame: Frame,
    normals: np.ndarray,
    usable: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    stride: int,
    sign: int,
    B: float,
) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    small: list[np.ndarray] = []
    large: list[np.ndarray] = []
    previous = frame.points[row, col]
    path = 0.0
    connected = True
    for step in range(1, 65):
        rr = row + sign * dr * stride * step
        cc = col + sign * dc * stride * step
        if rr < 0 or cc < 0 or rr >= frame.depth.shape[0] or cc >= frame.depth.shape[1] or not usable[rr, cc]:
            connected = False
            break
        point = frame.points[rr, cc]
        jump = float(np.linalg.norm(point - previous))
        if jump > 0.006 * B:
            connected = False
            break
        path += jump
        previous = point
        if 0.003 * B <= path <= 0.015 * B:
            small.append(normals[rr, cc])
        if 0.006 * B <= path <= 0.030 * B:
            large.append(normals[rr, cc])
        if path > 0.030 * B:
            break
    return small, large, connected


def _raw_tokens_for_view(frame: Frame, view: int, stride: int, B: float) -> tuple[list[Token], np.ndarray, np.ndarray]:
    # Normals and physical banks stay at native organized resolution.  The
    # frozen stride controls eligible profile centres and the global pixel
    # budget; applying it inside a physical bank can jump farther than the
    # 0.006B connectivity radius on scanner-density inputs.
    normals, usable = _normal_field(frame, 1, B)
    raw: list[Token] = []
    height, width = frame.depth.shape
    for direction_ordinal, (dc, dr) in enumerate(DIRECTIONS):
        # A source-only computational prefilter.  It can only remove centres
        # whose immediate two-sided normal change is below the 15-degree gate.
        for row in range(0, height, stride):
            rr0, rr1 = row - dr, row + dr
            if rr0 < 0 or rr1 >= height:
                continue
            for col in range(0, width, stride):
                cc0, cc1 = col - dc, col + dc
                if cc0 < 0 or cc1 >= width or not (usable[row, col] and usable[rr0, cc0] and usable[rr1, cc1]):
                    continue
                if _angle_deg(normals[rr0, cc0], normals[rr1, cc1]) < 10.0:
                    continue
                left_s, left_l, left_connected = _side_samples(frame, normals, usable, row, col, dr, dc, 1, -1, B)
                right_s, right_l, right_connected = _side_samples(frame, normals, usable, row, col, dr, dc, 1, 1, B)
                if not (left_connected and right_connected and len(left_s) >= 4 and len(right_s) >= 4 and len(left_l) >= 6 and len(right_l) >= 6):
                    continue
                Ls, Ll, Rs, Rl = (_representative(values) for values in (left_s, left_l, right_s, right_l))
                if any(value is None for value in (Ls, Ll, Rs, Rl)):
                    continue
                assert Ls is not None and Ll is not None and Rs is not None and Rl is not None
                alpha_s = _angle_deg(Ls[0], Rs[0])
                alpha_l = _angle_deg(Ll[0], Rl[0])
                concentration = alpha_s / max(alpha_l, 1.0)
                if not (15.0 <= alpha_s <= 165.0 and MIN_CONCENTRATION <= concentration <= 1.35):
                    continue
                tangent = _unit(np.cross(Ls[0], Rs[0]))
                if tangent is None or float(np.linalg.norm(np.cross(Ls[0], Rs[0]))) < math.sin(math.radians(15.0)):
                    continue
                tangent = _canonical_direction(tangent)
                roughness = Ls[1] + Rs[1]
                score = alpha_s * min(concentration, 1.0 / concentration) / (1.0 + roughness / 8.0)
                raw.append(Token(frame.points[row, col].copy(), tangent, score, alpha_s, concentration, roughness, view, row, col, direction_ordinal))
    return raw, normals, usable


def _line_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(point - origin, direction)))


def _thin(tokens: list[Token], shape: tuple[int, int], B: float) -> list[Token]:
    height, width = shape
    tiles: dict[tuple[int, int], list[Token]] = {}
    for token in tokens:
        key = (min(15, token.row * 16 // height), min(15, token.col * 16 // width))
        tiles.setdefault(key, []).append(token)
    retained: list[Token] = []
    for key in sorted(tiles):
        ordered = sorted(
            tiles[key],
            key=lambda token: (-token.score, -token.alpha, abs(token.concentration - 1.0), token.roughness, token.row, token.col, token.direction_ordinal),
        )
        first = ordered[0]
        retained.append(first)
        for token in ordered[1:]:
            if _line_distance(token.point, first.point, first.tangent) > 0.012 * B or _angle_deg(token.tangent, first.tangent) > 10.0:
                retained.append(token)
                break
    if len(retained) > MAX_TOKENS_PER_VIEW:
        raise ValueError("per-view token cap exceeded")
    return retained


def _occupancy(points: np.ndarray, origin: np.ndarray, direction: np.ndarray, views: np.ndarray) -> tuple[int, int, float, np.ndarray]:
    axial = (points - origin) @ direction
    low, high = float(np.min(axial)), float(np.max(axial))
    span = high - low
    if span <= 0.0:
        return 0, 0, 0.0, axial
    bins = np.minimum(9, np.floor((axial - low) / span * 10.0).astype(int))
    occupied = len(set(int(value) for value in bins))
    two_view = sum(len(set(int(views[index]) for index in np.flatnonzero(bins == bin_id))) >= 2 for bin_id in range(10))
    return occupied, two_view, span, axial


def _consensus(tokens: list[Token], frames: list[Frame], B: float) -> list[dict[str, Any]]:
    remaining = list(tokens)
    lines: list[dict[str, Any]] = []
    while remaining:
        points = np.asarray([token.point for token in remaining])
        tangents = np.asarray([token.tangent for token in remaining])
        views = np.asarray([token.view for token in remaining], dtype=int)
        candidates = []
        for seed_index, seed in enumerate(remaining):
            angles = np.asarray([_angle_deg(seed.tangent, tangent) for tangent in tangents])
            distances = np.linalg.norm(np.cross(points - seed.point, seed.tangent), axis=1)
            inliers = np.flatnonzero((angles <= 6.0) & (distances <= 0.006 * B))
            if len(inliers) < 8:
                continue
            occupied, two_view, span, _ = _occupancy(points[inliers], seed.point, seed.tangent, views[inliers])
            distinct = len(set(int(value) for value in views[inliers]))
            total_score = sum(remaining[index].score for index in inliers)
            median_residual = float(np.median(distances[inliers]))
            seed_bytes = np.ascontiguousarray(np.r_[seed.point, seed.tangent], dtype=">f8").tobytes()
            candidates.append(((-distinct, -two_view, -occupied, -round(total_score, 9), round(median_residual / B, 9), seed_bytes), seed_index, inliers))
        if not candidates:
            break
        _, seed_index, inliers = min(candidates, key=lambda row: row[0])
        origin = remaining[seed_index].point.copy()
        direction = remaining[seed_index].tangent.copy()
        for _ in range(2):
            chosen = [remaining[index] for index in inliers]
            weights = np.asarray([1.0 / (1.0 + (token.roughness / 5.0) ** 2) for token in chosen])
            chosen_points = np.asarray([token.point for token in chosen])
            origin = np.average(chosen_points, axis=0, weights=weights)
            covariance = ((chosen_points - origin) * weights[:, None]).T @ (chosen_points - origin)
            values, vectors = np.linalg.eigh(covariance)
            direction = _canonical_direction(vectors[:, int(np.argmax(values))])
            angles = np.asarray([_angle_deg(direction, token.tangent) for token in remaining])
            distances = np.asarray([_line_distance(token.point, origin, direction) for token in remaining])
            inliers = np.flatnonzero((angles <= 6.0) & (distances <= 0.006 * B))
        chosen = [remaining[index] for index in inliers]
        chosen_points = np.asarray([token.point for token in chosen])
        chosen_views = np.asarray([token.view for token in chosen], dtype=int)
        occupied, two_view, span, axial = _occupancy(chosen_points, origin, direction, chosen_views)
        distinct_views = sorted(set(int(value) for value in chosen_views))
        diverse = False
        for i, left in enumerate(distinct_views):
            for right in distinct_views[i + 1 :]:
                separation = float(np.linalg.norm(frames[left].camera - frames[right].camera))
                ray_left = origin - frames[left].camera
                ray_right = origin - frames[right].camera
                if separation >= 0.020 * B and _angle_deg(ray_left, ray_right) >= 15.0:
                    diverse = True
        residuals = [_line_distance(token.point, origin, direction) for token in chosen]
        direction_errors = [_angle_deg(token.tangent, direction) for token in chosen]
        accepted = (
            len(distinct_views) >= 3 and diverse and len(chosen) >= 8 and span >= 0.10 * B
            and occupied >= 7 and two_view >= 5 and _nearest_rank(residuals, 0.95) <= 0.004 * B
            and float(np.median(direction_errors)) <= 5.0
        )
        if accepted:
            start = origin + float(np.min(axial)) * direction
            end = origin + float(np.max(axial)) * direction
            lines.append(
                {
                    "line_start": start.tolist(),
                    "line_end": end.tolist(),
                    "supporting_view_count": len(distinct_views),
                    "token_count": len(chosen),
                    "occupied_bin_count": occupied,
                    "two_view_bin_count": two_view,
                    "median_turn_deg": float(np.median([token.alpha for token in chosen])),
                    "median_concentration": float(np.median([token.concentration for token in chosen])),
                    "p95_residual_over_B": _nearest_rank(residuals, 0.95) / B,
                }
            )
            if len(lines) > MAX_RAW_LINES:
                raise ValueError("raw line cap exceeded")
        remove = set(int(index) for index in inliers)
        remaining = [token for index, token in enumerate(remaining) if index not in remove]
    return lines


def _endpoints(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(row["line_start"], dtype=np.float64), np.asarray(row["line_end"], dtype=np.float64)


def _line_metrics(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float, float]:
    a0, a1 = _endpoints(left)
    b0, b1 = _endpoints(right)
    da, db = _unit(a1 - a0), _unit(b1 - b0)
    if da is None or db is None:
        return 180.0, math.inf, 0.0
    angle = _angle_deg(da, db)
    ma, mb = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    distance = 0.5 * (_line_distance(ma, mb, db) + _line_distance(mb, ma, da))
    axis = _unit(da + (db if np.dot(da, db) >= 0.0 else -db))
    if axis is None:
        return angle, distance, 0.0
    ia = sorted((float(np.dot(a0, axis)), float(np.dot(a1, axis))))
    ib = sorted((float(np.dot(b0, axis)), float(np.dot(b1, axis))))
    overlap = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0])) / max(min(ia[1] - ia[0], ib[1] - ib[0]), 1e-12)
    return angle, distance, overlap


def compose_v58(bundle: Path, v52: dict[str, Any]) -> dict[str, Any]:
    if v52.get("source_only") is not True or v52.get("research_only") is not True:
        raise ValueError("V58 requires source-only research V52")
    source, frames, source_sha, bundle_sha = load_bundle(bundle)
    B = float(np.linalg.norm(np.ptp(source, axis=0)))
    if not np.isfinite(B) or B <= 0.0:
        raise ValueError("invalid source scale")
    stride = _stride(frames)
    thinned: list[Token] = []
    raw_by_view: list[list[Token]] = []
    normal_maps: list[np.ndarray] = []
    usable_maps: list[np.ndarray] = []
    for view, frame in enumerate(frames):
        raw, normals, usable = _raw_tokens_for_view(frame, view, stride, B)
        raw_by_view.append(raw)
        normal_maps.append(normals)
        usable_maps.append(usable)
        thinned.extend(_thin(raw, frame.depth.shape, B))
    if len(thinned) > MAX_TOKENS_PER_SCAN:
        raise ValueError("scan token cap exceeded")
    raw_lines = _consensus(thinned, frames, B)

    incumbents = list(v52.get("predictions") or [])
    feasible: list[tuple[float, float, float, int, int]] = []
    for raw_index, raw in enumerate(raw_lines):
        for incumbent_index, incumbent in enumerate(incumbents):
            angle, distance, overlap = _line_metrics(raw, incumbent)
            if angle <= 4.0 and distance <= 0.008 * B and overlap >= 0.60:
                feasible.append((-overlap, distance, angle, raw_index, incumbent_index))
    used_raw: set[int] = set()
    used_incumbents: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, _, raw_index, incumbent_index in sorted(feasible):
        if raw_index not in used_raw and incumbent_index not in used_incumbents:
            used_raw.add(raw_index)
            used_incumbents.add(incumbent_index)
            matches.append((raw_index, incumbent_index))

    retained: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    for incumbent_index, incumbent in enumerate(incumbents):
        if incumbent_index in used_incumbents:
            retained.append(incumbent)
            continue
        start, end = _endpoints(incumbent)
        direction = _unit(end - start)
        if direction is None:
            retained.append(incumbent)
            continue
        negative_stations = 0
        station_minima: list[int] = []
        for fraction in (0.05, 0.14, 0.23, 0.32, 0.41, 0.50, 0.59, 0.68, 0.77, 0.86, 0.95):
            station = start + fraction * (end - start)
            eligible: list[int] = []
            positive: set[int] = set()
            for view, frame in enumerate(frames):
                valid_points = frame.points[usable_maps[view]]
                if not len(valid_points):
                    continue
                nearest_distance, _ = cKDTree(valid_points).query(station, k=1)
                if nearest_distance > 0.008 * B:
                    continue
                eligible.append(view)
                for token in raw_by_view[view]:
                    if np.linalg.norm(token.point - station) <= 0.010 * B and _angle_deg(token.tangent, direction) <= 10.0:
                        positive.add(view)
                        break
            negatives = [view for view in eligible if view not in positive]
            diverse = any(
                np.linalg.norm(frames[left].camera - frames[right].camera) >= 0.020 * B
                and _angle_deg(station - frames[left].camera, station - frames[right].camera) >= 15.0
                for offset, left in enumerate(negatives) for right in negatives[offset + 1 :]
            )
            if len(negatives) >= 3 and diverse:
                negative_stations += 1
                station_minima.append(len(negatives))
        if negative_stations >= 9:
            pruned.append({"v52_ordinal": incumbent_index, "reason": "RAW_VIEW_OBSERVABLE_NEGATIVE", "negative_station_count": negative_stations, "minimum_negative_view_count": min(station_minima)})
        else:
            retained.append(incumbent)

    births: list[dict[str, Any]] = []
    for raw_index, raw in enumerate(raw_lines):
        if raw_index in used_raw:
            continue
        payload = np.asarray(raw["line_start"] + raw["line_end"], dtype=">f8").tobytes()
        births.append(
            {
                "bend_id": "TRV58-" + hashlib.sha256(payload).hexdigest()[:20],
                "line_start": raw["line_start"],
                "line_end": raw["line_end"],
                "selection_route": "calibrated_multiview_turn_consensus_v58",
                "source_kind": "organized_calibrated_raw_range",
                **{key: raw[key] for key in ("supporting_view_count", "token_count", "occupied_bin_count", "two_view_bin_count", "median_turn_deg", "median_concentration", "p95_residual_over_B")},
            }
        )
    births.sort(key=lambda row: (-row["supporting_view_count"], -row["occupied_bin_count"], -np.linalg.norm(np.asarray(row["line_end"]) - np.asarray(row["line_start"])), -row["median_turn_deg"], _canonical_json([row["line_start"], row["line_end"]])))
    predictions = retained + births
    audit = {
        "selected_view_count": len(frames),
        "pixel_stride": stride,
        "thinned_token_count": len(thinned),
        "raw_line_count": len(raw_lines),
        "matched_v52_count": len(matches),
        "retained_v52_count": len(retained),
        "pruned_v52_count": len(pruned),
        "raw_new_count": len(births),
        "bank_median_residual_max_deg": BANK_MEDIAN_RESIDUAL_MAX_DEG,
        "bank_p90_residual_max_deg": BANK_P90_RESIDUAL_MAX_DEG,
        "minimum_concentration": MIN_CONCENTRATION,
    }
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v58_calibrated_multiview_turn_consensus_result",
        "profile": PROFILE,
        "source_sha256": source_sha,
        "capability_sha256": bundle_sha,
        "v52_sha256": v52.get("v52_result_sha256"),
        "scale_B_mm": B,
        "predictions": predictions,
        "pruned_v52": pruned,
        "audit": audit,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
    }
    result["v58_result_sha256"] = hashlib.sha256(_canonical_json({"profile": PROFILE, "predictions": predictions, "pruned_v52": pruned, "audit": audit}).encode()).hexdigest()
    return result
