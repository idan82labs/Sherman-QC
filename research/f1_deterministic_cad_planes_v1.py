"""Deterministic planar datum extraction from exact CAD surface normals."""

from __future__ import annotations

from typing import Any

import numpy as np


def _plane_extent(points: np.ndarray) -> float:
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    basis = eigenvectors[:, order[:2]]
    projected = centered @ basis
    return float(np.ptp(projected[:, 0]) * np.ptp(projected[:, 1]))


def canonical_plane_keys(
    points: np.ndarray,
    normals: np.ndarray,
    normal_decimals: int = 6,
    offset_decimals: int = 6,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape != nrm.shape or len(pts) == 0:
        raise ValueError("points and normals must be matching nonempty Nx3 arrays")
    lengths = np.linalg.norm(nrm, axis=1)
    if np.any(lengths <= 1e-15):
        raise ValueError("zero-length CAD surface normal")
    unit = nrm / lengths[:, None]
    pivot = np.argmax(np.abs(unit), axis=1)
    pivot_values = unit[np.arange(len(unit)), pivot]
    sign = np.where(pivot_values < 0.0, -1.0, 1.0)
    unit = unit * sign[:, None]
    offsets = -np.einsum("ij,ij->i", unit, pts)
    normal_key = np.round(unit, normal_decimals)
    offset_key = np.round(offsets, offset_decimals)[:, None]
    keys = np.column_stack((normal_key, offset_key))
    keys[keys == 0.0] = 0.0
    return keys


def extract_planes(
    points: np.ndarray,
    normals: np.ndarray,
    max_planes: int = 6,
    minimum_fraction: float = 0.02,
    minimum_count: int = 500,
) -> list[dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float64)
    keys = canonical_plane_keys(pts, normals)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    threshold = max(int(minimum_count), int(minimum_fraction * len(pts)))
    planes: list[dict[str, Any]] = []
    for group_index in np.flatnonzero(counts >= threshold).tolist():
        inlier_indices = np.flatnonzero(inverse == group_index)
        inlier_points = np.ascontiguousarray(pts[inlier_indices], dtype=np.float64)
        key = unique[group_index]
        normal = key[:3].astype(np.float64)
        normal /= np.linalg.norm(normal)
        offset = float(key[3])
        planes.append({
            "normal": normal,
            "d": offset,
            "inlier_pts": inlier_points,
            "n_points": int(len(inlier_points)),
            "bbox_area": _plane_extent(inlier_points),
            "centroid": inlier_points.mean(axis=0),
            "canonical_key": key.tolist(),
        })
    planes.sort(key=lambda row: (-row["bbox_area"], tuple(row["canonical_key"])))
    return planes[: int(max_planes)]
