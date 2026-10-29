#!/usr/bin/env python3
"""Build a bounded, identity-free F1 bend-line candidate pool.

The pool is geometry only.  It accepts strict-blind decomposition results and
their recorded scan paths, applies every generator uniformly to every scan,
and never reads part identifiers, expected counts, CAD, or manual labels.
Source points are used transiently and are not serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.f1_cylindrical_band_completion import complete_cylindrical_bands  # noqa: E402
from backend.f1_line_topology_completion import (  # noqa: E402
    complete_parallel_endpoint_connectors,
    merge_collinear_fragments,
)
from backend.f1_line_object_clustering import cluster_line_candidates  # noqa: E402
from backend.cad_bend_extractor import CADBendExtractor  # noqa: E402
from backend.plane_lattice_bend_recovery import (  # noqa: E402
    build_supported_plane_intersections,
    extract_deterministic_planes,
)
from backend.patch_graph_latent_decomposition import (  # noqa: E402
    build_transition_atom_graph_v2,
    compact_transition_graph_v2,
)
from backend.zero_shot_bend_profile import _estimate_local_spacing  # noqa: E402


GLOBAL_POINT_BUDGET = 200_000


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_isolated_case_map(prediction_root: Path) -> dict[str, Path]:
    """Discover any isolated scan-only run without importing a truth registry."""

    isolated_root = prediction_root / "isolated_cases"
    discovered: dict[str, Path] = {}
    for case_dir in sorted(path for path in isolated_root.iterdir() if path.is_dir()):
        matches = sorted(case_dir.glob("result/**/decomposition_result.json"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one decomposition result for {case_dir.name}, found {len(matches)}"
            )
        discovered[case_dir.name] = matches[0].resolve()
    if not discovered:
        raise ValueError(f"no isolated decomposition results found under {isolated_root}")
    return discovered


def deterministic_geometry_sample(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    max_points: int = GLOBAL_POINT_BUDGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one canonical finite working cloud bounded for every generator.

    Ordering and sampling depend only on quantized geometry.  Source order,
    filenames, part identity, and generator-specific state cannot affect the
    selected working set.
    """

    if int(max_points) < 1:
        raise ValueError("max_points must be positive")
    point_array = np.asarray(points, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1:] != (3,):
        raise ValueError("points must have shape (n, 3)")
    if normal_array.shape != point_array.shape:
        raise ValueError("normals must match points")
    finite = np.isfinite(point_array).all(axis=1) & np.isfinite(normal_array).all(axis=1)
    point_array = point_array[finite]
    normal_array = normal_array[finite]
    if not len(point_array):
        return point_array, normal_array

    # Micron quantization stabilizes ordering against harmless parser-level
    # floating noise while keeping sub-scanner-resolution distinctions out of
    # the identity.  Normals break exact-coordinate ties deterministically.
    quantized_points = np.rint(point_array * 1_000.0).astype(np.int64)
    quantized_normals = np.rint(normal_array * 1_000_000.0).astype(np.int64)
    order = np.lexsort(
        (
            quantized_normals[:, 2],
            quantized_normals[:, 1],
            quantized_normals[:, 0],
            quantized_points[:, 2],
            quantized_points[:, 1],
            quantized_points[:, 0],
        )
    )
    point_array = point_array[order]
    normal_array = normal_array[order]
    if len(point_array) > int(max_points):
        indices = np.linspace(
            0,
            len(point_array) - 1,
            num=int(max_points),
            dtype=np.int64,
        )
        point_array = point_array[indices]
        normal_array = normal_array[indices]
    return point_array, normal_array


def _sorted_geometry(cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    return deterministic_geometry_sample(points, normals)


def _load_geometry_views(
    scan_path: Path,
    *,
    plane_voxel_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(scan_path))
    if not len(cloud.points):
        mesh = o3d.io.read_triangle_mesh(str(scan_path))
        if not len(mesh.vertices):
            raise ValueError(f"scan contains no geometry: {scan_path}")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
        cloud.normals = mesh.vertex_normals
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(3.0 * float(plane_voxel_mm), 2.0),
                max_nn=30,
            )
        )
    points, normals = _sorted_geometry(cloud)
    bounded_cloud = o3d.geometry.PointCloud()
    bounded_cloud.points = o3d.utility.Vector3dVector(points)
    bounded_cloud.normals = o3d.utility.Vector3dVector(normals)
    plane_cloud = bounded_cloud.voxel_down_sample(float(plane_voxel_mm))
    if not plane_cloud.has_normals():
        plane_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(3.0 * float(plane_voxel_mm), 2.0),
                max_nn=30,
            )
        )
    plane_points, plane_normals = _sorted_geometry(plane_cloud)
    return points, normals, plane_points, plane_normals


def _declared_ply_vertex_count(path: Path) -> int | None:
    """Read a bounded ASCII PLY header without materializing mesh vertices."""

    if path.suffix.lower() != ".ply":
        return None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            return None
        vertex_count: int | None = None
        for _line_index in range(4096):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path}: truncated PLY header")
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[-1])
                if vertex_count < 0:
                    raise ValueError(f"{path}: negative PLY vertex count")
            if line == "end_header":
                return vertex_count
        raise ValueError(f"{path}: PLY header exceeds deterministic read cap")


SUPPORT_ATOM_ID_CAP_PER_CANDIDATE = 4096
SUPPORT_ATOM_TABLE_CAP = 20_000
SUPPORT_ATOM_EDGE_CAP = 200_000

INFERENCE_FORBIDDEN_KEYS = frozenset(
    {
        "point_indices",
        "source_points",
        "points",
        "bbox_points",
        "filename",
        "part_id",
        "expected_count",
        "manual_truth",
        "cad_metadata",
        "split_group_id",
    }
)


def _reject_recursive_forbidden_keys(
    value: Any,
    *,
    forbidden: frozenset[str] = INFERENCE_FORBIDDEN_KEYS,
    path: str = "$",
) -> None:
    """Fail closed when inference payloads contain source or truth-bearing keys."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key in forbidden:
                raise ValueError(f"forbidden inference key at {child_path}: {key}")
            _reject_recursive_forbidden_keys(
                child,
                forbidden=forbidden,
                path=child_path,
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_recursive_forbidden_keys(
                child,
                forbidden=forbidden,
                path=f"{path}[{index}]",
            )


def _bounded_identifiers(values: Sequence[Any], *, cap: int) -> tuple[list[str], bool]:
    identifiers = sorted({str(value) for value in values if str(value)})
    return identifiers[:cap], len(identifiers) > cap


def _tag(values: list[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    tagged = []
    for index, value in enumerate(values, start=1):
        candidate = dict(value)
        if (
            candidate.get("span_endpoints") is None
            and candidate.get("line_start") is not None
            and candidate.get("line_end") is not None
        ):
            endpoints = np.asarray(
                (candidate["line_start"], candidate["line_end"]), dtype=np.float64
            )
            delta = endpoints[1] - endpoints[0]
            span_mm = float(np.linalg.norm(delta))
            if span_mm > 1e-9:
                candidate["span_endpoints"] = endpoints.tolist()
                candidate.setdefault("anchor", np.mean(endpoints, axis=0).tolist())
                candidate.setdefault("axis_direction", (delta / span_mm).tolist())
                candidate.setdefault("span_mm", span_mm)
        support_ids = [
            value
            for key in ("owned_atom_ids", "atom_ids", "source_atom_ids")
            for value in (candidate.get(key) or ())
        ]
        bounded_support_ids, support_ids_truncated = _bounded_identifiers(
            support_ids,
            cap=SUPPORT_ATOM_ID_CAP_PER_CANDIDATE,
        )
        candidate.setdefault("support_count", len(set(str(value) for value in support_ids)))
        if bounded_support_ids:
            candidate["support_atom_ids"] = bounded_support_ids
            candidate["support_atom_ids_truncated"] = support_ids_truncated
        support_patch_ids, support_patch_ids_truncated = _bounded_identifiers(
            candidate.get("support_patch_ids") or (),
            cap=SUPPORT_ATOM_ID_CAP_PER_CANDIDATE,
        )
        if support_patch_ids:
            candidate["support_patch_ids"] = support_patch_ids
            candidate["support_patch_ids_truncated"] = support_patch_ids_truncated
        for key in (
            "owned_atom_ids",
            "atom_ids",
            "source_atom_ids",
            "merge_provenance",
        ):
            candidate.pop(key, None)
        source_candidate_id = str(
            candidate.get("candidate_id")
            or candidate.get("bend_id")
            or f"candidate_{index}"
        )
        # Historical cylindrical generators reuse identifiers across radius
        # configurations (for example CB2_301 in both standard and wide
        # pools).  Namespace every identity before clustering so diagnostics,
        # cache overlays, and deduplication never alias distinct geometry.
        candidate["source_candidate_id"] = source_candidate_id
        candidate["candidate_id"] = f"{source}::{source_candidate_id}"
        candidate["bend_id"] = candidate["candidate_id"]
        candidate["candidate_source"] = source
        tagged.append(candidate)
    return tagged


def _compact_support_graph(
    decomposition: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize bounded atom statistics and sparse edges, never point indices."""

    graph = dict(decomposition.get("atom_graph") or {})
    atoms = sorted(
        (dict(atom) for atom in graph.get("atoms") or ()),
        key=lambda atom: str(atom.get("atom_id") or ""),
    )
    if len(atoms) > SUPPORT_ATOM_TABLE_CAP:
        raise ValueError(
            f"support atom table exceeds cap: {len(atoms)} > {SUPPORT_ATOM_TABLE_CAP}"
        )
    compact_atoms = [
        {
            "atom_id": str(atom["atom_id"]),
            "centroid": [float(value) for value in atom["centroid"]],
            "normal": [float(value) for value in atom["normal"]],
            "local_spacing_mm": float(atom.get("local_spacing_mm") or 0.0),
            "curvature_proxy": float(atom.get("curvature_proxy") or 0.0),
            "axis_direction": [
                float(value) for value in atom.get("axis_direction") or (0.0, 0.0, 0.0)
            ],
            "weight": float(atom.get("weight") or 0.0),
            "area_proxy_mm2": float(atom.get("area_proxy_mm2") or 0.0),
        }
        for atom in atoms
    ]
    atom_ids = {atom["atom_id"] for atom in compact_atoms}
    adjacency = dict(graph.get("adjacency") or {})
    edge_weights = dict(graph.get("edge_weights") or {})
    edge_pairs = sorted(
        {
            tuple(sorted((str(left), str(right))))
            for left, neighbors in adjacency.items()
            if str(left) in atom_ids
            for right in neighbors or ()
            if str(right) in atom_ids and str(left) != str(right)
        }
    )
    if len(edge_pairs) > SUPPORT_ATOM_EDGE_CAP:
        raise ValueError(
            f"support atom edge table exceeds cap: {len(edge_pairs)} > {SUPPORT_ATOM_EDGE_CAP}"
        )
    compact_edges = []
    for left, right in edge_pairs:
        weight = edge_weights.get(f"{left}::{right}")
        if weight is None:
            weight = edge_weights.get(f"{right}::{left}")
        compact_edges.append(
            {
                "atom_ids": [left, right],
                "weight": float(weight or 0.0),
            }
        )
    return compact_atoms, compact_edges


def _canonical_unit_vector(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("expected a finite non-zero 3-vector")
    vector = vector / norm
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    vector[np.abs(vector) < 1e-15] = 0.0
    return vector


def _deterministic_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    first = _canonical_unit_vector(first)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    second[np.abs(second) < 1e-15] = 0.0
    return first, second


def _repair_zero_normals(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    local_spacing_mm: float,
) -> np.ndarray:
    """Deterministically fill only missing normals on the bounded cloud."""

    point_array = np.asarray(points, dtype=np.float64)
    result = np.asarray(normals, dtype=np.float64).copy()
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > 1e-12
    result[valid] /= lengths[valid, None]
    missing = np.flatnonzero(~valid)
    if not len(missing):
        return result
    tree = cKDTree(point_array)
    neighbor_count = min(30, len(point_array))
    distances, neighbor_ids = tree.query(
        point_array[missing],
        k=neighbor_count,
        workers=1,
    )
    if neighbor_count == 1:
        neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64).reshape(-1, 1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1, 1)
    radius = max(float(local_spacing_mm) * 4.0, 3.0)
    for row_index, point_index in enumerate(missing):
        local_ids = np.asarray(neighbor_ids[row_index], dtype=np.int64)
        local_distances = np.asarray(distances[row_index], dtype=np.float64)
        local_ids = local_ids[np.isfinite(local_distances) & (local_distances <= radius)]
        if len(local_ids) < 3:
            local_ids = np.asarray(neighbor_ids[row_index], dtype=np.int64)[: min(6, neighbor_count)]
        local_points = point_array[local_ids]
        center = np.mean(local_points, axis=0)
        covariance = (local_points - center).T @ (local_points - center)
        _values, vectors = np.linalg.eigh(covariance)
        result[point_index] = _canonical_unit_vector(vectors[:, 0])
    return result


def _enrich_transition_plane_table(
    plane_records: Sequence[Mapping[str, Any]],
    plane_points: np.ndarray,
    transition_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach bounded scan-derived plane extents and V2 mode memberships."""

    support_points = np.asarray(plane_points, dtype=np.float64)
    modes = [dict(item) for item in transition_payload.get("support_mode_table") or ()]
    spacing = float(transition_payload.get("local_spacing_mm") or 0.0)
    if spacing <= 0.0:
        raise ValueError("TRANSITION_ATOM_V2 payload requires positive local spacing")
    distance_scale = min(3.0, max(1.25, 1.5 * spacing))
    memberships_by_mode: dict[str, list[dict[str, Any]]] = {
        str(mode["mode_atom_id"]): [] for mode in modes
    }
    enriched: list[dict[str, Any]] = []
    for raw_plane in sorted(plane_records, key=lambda item: str(item.get("plane_id") or "")):
        plane_id = str(raw_plane.get("plane_id") or "")
        if not plane_id:
            raise ValueError("plane record is missing plane_id")
        normal = _canonical_unit_vector(raw_plane.get("normal") or ())
        offset = float(raw_plane.get("offset", raw_plane.get("d", 0.0)))
        tangent_u, tangent_v = _deterministic_tangent_basis(normal)
        residuals = np.abs(support_points @ normal + offset) if len(support_points) else np.zeros(0)
        inlier_mask = residuals <= distance_scale
        inlier_points = support_points[inlier_mask]
        if len(inlier_points):
            uv = np.column_stack((inlier_points @ tangent_u, inlier_points @ tangent_v))
            low = np.quantile(uv, 0.01, axis=0)
            high = np.quantile(uv, 0.99, axis=0)
            extent_uv = [
                [round(float(low[0]), 6), round(float(high[0]), 6)],
                [round(float(low[1]), 6), round(float(high[1]), 6)],
            ]
            fit_p90 = float(np.quantile(residuals[inlier_mask], 0.90))
        else:
            extent_uv = [[0.0, 0.0], [0.0, 0.0]]
            fit_p90 = math.inf
        component_hash = hashlib.sha256(
            json.dumps(
                [plane_id, extent_uv],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        component_ids = [f"PSC_{component_hash}"] if len(inlier_points) else []
        eligible_mode_ids: list[str] = []
        eligible_parent_ids: set[str] = set()
        for mode in modes:
            mode_id = str(mode["mode_atom_id"])
            mode_normal = _canonical_unit_vector(mode.get("normal_projective") or ())
            cosine = float(np.clip(abs(np.dot(mode_normal, normal)), 0.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cosine)))
            centroid = np.asarray(mode.get("centroid") or (), dtype=np.float64)
            if centroid.shape != (3,):
                raise ValueError(f"mode {mode_id} has invalid centroid")
            distance_mm = abs(float(np.dot(normal, centroid) + offset))
            normal_residual = angle_deg / 15.0
            distance_residual = distance_mm / distance_scale
            eligible = bool(
                normal_residual <= 1.0
                and distance_residual <= 1.0
                and component_ids
            )
            if not eligible:
                continue
            memberships_by_mode[mode_id].append(
                {
                    "canonical_plane_id": plane_id,
                    "normal_residual": round(normal_residual, 8),
                    "plane_distance_residual": round(distance_residual, 8),
                    "support_component_id": component_ids[0],
                    "eligible": True,
                }
            )
            eligible_mode_ids.append(mode_id)
            eligible_parent_ids.add(str(mode["parent_atom_id"]))
        enriched.append(
            {
                "plane_id": plane_id,
                "canonical_plane_id": plane_id,
                "normal": [round(float(value), 12) for value in normal],
                "offset": offset,
                "support_count": int(raw_plane.get("support_count") or len(inlier_points)),
                "merged_plane_ids": sorted(
                    str(value) for value in raw_plane.get("merged_plane_ids") or ()
                ),
                "tangent_basis": [
                    [round(float(value), 12) for value in tangent_u],
                    [round(float(value), 12) for value in tangent_v],
                ],
                "extent_uv": extent_uv,
                "fit_p90_mm": None if not math.isfinite(fit_p90) else round(fit_p90, 8),
                "support_component_ids": component_ids,
                "support_parent_ids": sorted(eligible_parent_ids),
                "support_mode_ids": sorted(eligible_mode_ids),
            }
        )
    membership_records = []
    for mode_id in sorted(memberships_by_mode):
        ordered = sorted(
            memberships_by_mode[mode_id],
            key=lambda item: (
                float(item["normal_residual"]) + float(item["plane_distance_residual"]),
                str(item["canonical_plane_id"]),
            ),
        )
        if len(ordered) > 3:
            ordered = ordered[:3]
        membership_records.append(
            {
                "mode_atom_id": mode_id,
                "plane_side_memberships": ordered,
            }
        )
    return enriched, membership_records


def augment_case_pool_with_transition_v2(
    candidate_pool: Mapping[str, Any],
    *,
    points: np.ndarray,
    normals: np.ndarray,
    plane_points: np.ndarray | None = None,
    plane_normals: np.ndarray | None = None,
) -> dict[str, Any]:
    """Add V2 support to a frozen V1/V2 pool without regenerating paths."""

    forbidden_envelope = {"case_id", "scan_path", "coordinate_frame"}
    source = {
        str(key): value
        for key, value in candidate_pool.items()
        if str(key) not in forbidden_envelope
    }
    _reject_recursive_forbidden_keys(source)
    bounded_points, bounded_normals = deterministic_geometry_sample(
        points,
        normals,
        max_points=GLOBAL_POINT_BUDGET,
    )
    if plane_points is None:
        bounded_plane_points = bounded_points
    else:
        if plane_normals is None:
            raise ValueError("plane_normals are required when plane_points are supplied")
        bounded_plane_points, _bounded_plane_normals = deterministic_geometry_sample(
            plane_points,
            plane_normals,
            max_points=GLOBAL_POINT_BUDGET,
        )
    spacing = max(float(_estimate_local_spacing(bounded_points)), 1e-3)
    transition_normals = _repair_zero_normals(
        bounded_points,
        bounded_normals,
        local_spacing_mm=spacing,
    )
    transition_payload = compact_transition_graph_v2(
        build_transition_atom_graph_v2(
            bounded_points,
            transition_normals,
            local_spacing_mm=spacing,
        )
    )
    plane_table, memberships = _enrich_transition_plane_table(
        source.get("plane_table") or (),
        bounded_plane_points,
        transition_payload,
    )
    result = dict(source)
    result.update(
        {
            "schema_version": 3,
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3_transition_augmented",
            "support_representation_profile": transition_payload[
                "support_representation_profile"
            ],
            "support_parent_table": transition_payload["support_parent_table"],
            "support_mode_table": transition_payload["support_mode_table"],
            "support_mode_edges": transition_payload["support_mode_edges"],
            "plane_table": plane_table,
            "plane_mode_memberships": memberships,
            "transition_atom_caps": transition_payload["transition_atom_caps"],
            "transition_parent_table_sha256": transition_payload[
                "parent_table_sha256"
            ],
            "transition_mode_table_sha256": transition_payload["mode_table_sha256"],
            "transition_mode_edge_table_sha256": transition_payload[
                "mode_edge_table_sha256"
            ],
            "transition_local_spacing_mm": transition_payload["local_spacing_mm"],
            "transition_parent_voxel_size_mm": transition_payload[
                "parent_voxel_size_mm"
            ],
            "transition_fine_cell_size_mm": transition_payload["fine_cell_size_mm"],
            "transition_atom_diagnostics": transition_payload["diagnostics"],
            "transient_sampled_point_count": int(len(bounded_points)),
            "transient_plane_point_count": int(len(bounded_plane_points)),
        }
    )
    _reject_recursive_forbidden_keys(result)
    return result


def _scan_mesh_adjacency_candidates(scan_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract bend candidates from connectivity embedded in the scan PLY.

    This is scan-only geometry.  The extractor name is historical; no CAD,
    part identifier, expected count, or manual truth is provided here.
    """

    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    source_vertex_count = int(len(mesh.vertices))
    source_triangle_count = int(len(mesh.triangles))
    decimated = False
    if source_vertex_count > GLOBAL_POINT_BUDGET and source_triangle_count >= 10:
        # Preserve topology while bounding the adjacency extractor.  A 3:2
        # triangle-to-vertex target lands below the same global point budget
        # on dense scanner meshes without using identity or expected counts.
        target_triangles = min(source_triangle_count, int(GLOBAL_POINT_BUDGET * 1.5))
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        decimated = True
    triangle_count = int(len(mesh.triangles))
    if not len(mesh.vertices) or triangle_count < 10:
        return [], {
            "available": False,
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": triangle_count,
            "source_vertex_count": source_vertex_count,
            "source_triangle_count": source_triangle_count,
            "bounded_decimation_used": decimated,
            "candidate_count": 0,
        }
    if len(mesh.vertices) > GLOBAL_POINT_BUDGET:
        raise ValueError(
            f"bounded scan-mesh adjacency still exceeds point budget: {len(mesh.vertices)}"
        )
    extraction = CADBendExtractor().extract_from_mesh(mesh)
    candidates = []
    for bend in extraction.bends:
        payload = bend.to_dict()
        line_start = np.asarray(bend.bend_line_start, dtype=np.float64)
        line_end = np.asarray(bend.bend_line_end, dtype=np.float64)
        delta = line_end - line_start
        span_mm = float(np.linalg.norm(delta))
        if span_mm <= 1e-9:
            continue
        axis_direction = delta / span_mm
        payload.update(
            {
                "bend_id": f"SMA{int(bend.bend_id)}",
                "candidate_id": f"SMA{int(bend.bend_id)}",
                "line_start": line_start.tolist(),
                "line_end": line_end.tolist(),
                "span_endpoints": [line_start.tolist(), line_end.tolist()],
                "anchor": ((line_start + line_end) * 0.5).tolist(),
                "axis_direction": axis_direction.tolist(),
                "span_mm": span_mm,
                "source_kind": "scan_mesh_surface_adjacency",
            }
        )
        candidates.append(payload)
    return candidates, {
        "available": True,
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": triangle_count,
        "source_vertex_count": source_vertex_count,
        "source_triangle_count": source_triangle_count,
        "bounded_decimation_used": decimated,
        "candidate_count": len(candidates),
        "major_surface_count": int(extraction.total_major_surfaces),
        "mesh_area": float(extraction.mesh_area),
        "warnings": list(extraction.warnings),
    }


def _support_strip_lines(decomposition: Mapping[str, Any]) -> list[dict[str, Any]]:
    control = dict(decomposition.get("current_zero_shot_result") or {})
    lines = []
    for strip in control.get("bend_support_strips") or ():
        hint = dict(strip.get("axis_line_hint") or {})
        midpoint = hint.get("midpoint")
        axis = strip.get("axis_direction")
        span = strip.get("along_axis_span_mm")
        if midpoint is None or axis is None or span is None:
            continue
        lines.append(
            {
                "bend_id": str(strip.get("strip_id") or f"STRIP{len(lines) + 1}"),
                "anchor": midpoint,
                "axis_direction": axis,
                "span_mm": span,
                "radius_mm": strip.get("radius_mm"),
                "support_count": strip.get("support_point_count"),
                "score": strip.get("confidence"),
                "stability_score": strip.get("stability_score"),
                "source_kind": "zero_shot_support_strip",
            }
        )
    return lines


def _evidence_trace_lines(decomposition: Mapping[str, Any]) -> list[dict[str, Any]]:
    control = dict(decomposition.get("current_zero_shot_result") or {})
    lines = []
    for atom in control.get("bend_evidence_atoms") or ():
        endpoints = atom.get("trace_endpoints")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            continue
        array = np.asarray(endpoints, dtype=np.float64)
        if array.shape != (2, 3) or not np.isfinite(array).all():
            continue
        span = float(np.linalg.norm(array[1] - array[0]))
        if span < 8.0:
            continue
        lines.append(
            {
                "bend_id": str(atom.get("atom_id") or f"TRACE{len(lines) + 1}"),
                "span_endpoints": endpoints,
                "span_mm": span,
                "support_count": atom.get("connected_component_size"),
                "score": atom.get("fold_strength"),
                "stability_score": atom.get("trace_continuity_score"),
                "source_kind": "zero_shot_evidence_trace",
            }
        )
    return lines


def _candidate_endpoints(candidate: Mapping[str, Any]) -> np.ndarray:
    if candidate.get("span_endpoints") is not None:
        return np.asarray(candidate["span_endpoints"], dtype=np.float64)
    if candidate.get("line_start") is not None and candidate.get("line_end") is not None:
        return np.asarray((candidate["line_start"], candidate["line_end"]), dtype=np.float64)
    anchor = np.asarray(candidate["anchor"], dtype=np.float64)
    axis = np.asarray(candidate["axis_direction"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * float(candidate["span_mm"]) * axis
    return np.asarray((anchor - half, anchor + half))


def _attach_finite_scan_support(
    candidates: list[dict[str, Any]],
    support_points: np.ndarray,
    *,
    distance_mm: float = 3.0,
    sample_count: int = 25,
) -> None:
    if not candidates or not len(support_points):
        return
    tree = cKDTree(np.asarray(support_points, dtype=np.float64))
    parameters = np.linspace(0.0, 1.0, max(3, int(sample_count)), dtype=np.float64)
    for candidate in candidates:
        endpoints = _candidate_endpoints(candidate)
        samples = endpoints[0] + parameters[:, None] * (endpoints[1] - endpoints[0])
        distances, _indices = tree.query(samples, k=1, workers=1)
        supported = np.asarray(distances <= float(distance_mm), dtype=bool)
        longest = current = 0
        for value in supported:
            current = current + 1 if value else 0
            longest = max(longest, current)
        edge_width = max(1, len(supported) // 8)
        candidate["finite_support_fraction"] = float(np.mean(supported))
        candidate["finite_support_longest_run_fraction"] = float(longest / len(supported))
        candidate["endpoint_support_fraction"] = float(
            0.5 * (np.mean(supported[:edge_width]) + np.mean(supported[-edge_width:]))
        )
        candidate["median_line_distance_mm"] = float(np.median(distances))
        candidate["p90_line_distance_mm"] = float(np.quantile(distances, 0.90))


def _trim_to_longest_scan_supported_run(
    candidates: list[dict[str, Any]],
    support_points: np.ndarray,
    *,
    distance_mm: float = 3.0,
    sample_count: int = 201,
    min_supported_span_mm: float = 7.0,
    point_budget: int = 200_000,
) -> list[dict[str, Any]]:
    """Crop finite candidates to supported scan spans and discard floating lines."""

    points = np.asarray(support_points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > int(point_budget):
        indices = np.linspace(0, len(points) - 1, int(point_budget), dtype=np.int64)
        points = points[indices]
    if not len(points):
        return []
    tree = cKDTree(points)
    parameters = np.linspace(0.0, 1.0, max(3, int(sample_count)), dtype=np.float64)
    trimmed: list[dict[str, Any]] = []
    for original in candidates:
        candidate = dict(original)
        endpoints = _candidate_endpoints(candidate)
        samples = endpoints[0] + parameters[:, None] * (endpoints[1] - endpoints[0])
        distances, _ = tree.query(samples, k=1, workers=1)
        supported = np.asarray(distances <= float(distance_mm), dtype=bool)
        best_start = best_end = current_start = 0
        in_run = False
        for index, value in enumerate(supported):
            if value and not in_run:
                current_start = index
                in_run = True
            if in_run and (not value or index == len(supported) - 1):
                current_end = index + 1 if value else index
                if current_end - current_start > best_end - best_start:
                    best_start, best_end = current_start, current_end
                in_run = False
        if best_end <= best_start:
            continue
        new_endpoints = np.asarray(
            (samples[best_start], samples[best_end - 1]), dtype=np.float64
        )
        span_mm = float(np.linalg.norm(new_endpoints[1] - new_endpoints[0]))
        if span_mm < float(min_supported_span_mm):
            continue
        axis = (new_endpoints[1] - new_endpoints[0]) / span_mm
        candidate["source_span_endpoints"] = endpoints.tolist()
        candidate["span_endpoints"] = new_endpoints.tolist()
        candidate["line_start"] = new_endpoints[0].tolist()
        candidate["line_end"] = new_endpoints[1].tolist()
        candidate["anchor"] = np.mean(new_endpoints, axis=0).tolist()
        candidate["axis_direction"] = axis.tolist()
        candidate["span_mm"] = span_mm
        candidate["scan_support_crop"] = {
            "distance_threshold_mm": float(distance_mm),
            "sample_count": int(len(samples)),
            "source_span_mm": float(np.linalg.norm(endpoints[1] - endpoints[0])),
            "supported_span_mm": span_mm,
            "support_fraction": float(np.mean(supported)),
            "longest_support_run_fraction": float((best_end - best_start) / len(samples)),
        }
        trimmed.append(candidate)
    return trimmed


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.8:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    return first, np.cross(axis, first)


def _circular_span_deg(angles: np.ndarray) -> float:
    values = np.mod(np.asarray(angles, dtype=np.float64), 2.0 * np.pi)
    if len(values) < 2:
        return 0.0
    ordered = np.sort(values)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    return float(np.degrees(2.0 * np.pi - np.max(gaps)))


def _dominant_unoriented_normal(normals: np.ndarray) -> tuple[np.ndarray, float]:
    moment = np.einsum("ni,nj->ij", normals, normals) / max(len(normals), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    dominant = eigenvectors[:, int(np.argmax(eigenvalues))]
    coherence = float(np.max(eigenvalues) / max(float(np.sum(eigenvalues)), 1e-12))
    return dominant, coherence


def _two_sided_transition_features(
    radial: np.ndarray,
    normals: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    split_margin_mm: float = 0.75,
    minimum_side_points: int = 6,
    split_direction_count: int = 16,
) -> dict[str, float]:
    """Measure balanced, planar, two-normal support around a candidate axis."""

    if len(radial) != len(normals) or len(radial) < 2 * minimum_side_points:
        return {
            "two_sided_transition_score": 0.0,
            "two_sided_normal_angle_deg": 0.0,
            "two_sided_support_balance": 0.0,
            "two_sided_min_support_fraction": 0.0,
            "two_sided_normal_coherence": 0.0,
        }
    best: tuple[float, float, float, float, float] | None = None
    for angle in np.linspace(0.0, np.pi, max(4, int(split_direction_count)), endpoint=False):
        direction = np.cos(angle) * first + np.sin(angle) * second
        signed = radial @ direction
        negative = signed <= -float(split_margin_mm)
        positive = signed >= float(split_margin_mm)
        negative_count = int(np.count_nonzero(negative))
        positive_count = int(np.count_nonzero(positive))
        if min(negative_count, positive_count) < int(minimum_side_points):
            continue
        negative_normal, negative_coherence = _dominant_unoriented_normal(normals[negative])
        positive_normal, positive_coherence = _dominant_unoriented_normal(normals[positive])
        normal_angle = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(np.dot(negative_normal, positive_normal))), 0.0, 1.0)
                )
            )
        )
        balance = float(2.0 * min(negative_count, positive_count) / (negative_count + positive_count))
        min_fraction = float(min(negative_count, positive_count) / len(radial))
        coherence = float(np.sqrt(negative_coherence * positive_coherence))
        score = float(balance * coherence * np.sin(np.radians(normal_angle)))
        key = (score, normal_angle, balance, min_fraction, coherence)
        if best is None or key > best:
            best = key
    score, normal_angle, balance, min_fraction, coherence = best or (0.0,) * 5
    return {
        "two_sided_transition_score": score,
        "two_sided_normal_angle_deg": normal_angle,
        "two_sided_support_balance": balance,
        "two_sided_min_support_fraction": min_fraction,
        "two_sided_normal_coherence": coherence,
    }


def _attach_local_bend_support(
    candidates: list[dict[str, Any]],
    support_points: np.ndarray,
    support_normals: np.ndarray,
    *,
    corridor_radius_mm: float = 12.0,
    sample_count: int = 13,
) -> None:
    defaults = {
        "corridor_point_count": 0,
        "corridor_points_per_mm": 0.0,
        "transverse_angular_span_deg": 0.0,
        "transverse_angular_bin_fraction": 0.0,
        "median_corridor_radius_mm": 0.0,
        "normal_second_eigenvalue": 0.0,
        "normal_spread_deg_scan": 0.0,
        "tight_normal_second_eigenvalue_r2mm": 0.0,
        "tight_normal_second_eigenvalue_r6mm": 0.0,
        "tight_to_wide_normal_second_ratio": 0.0,
        "tight_normal_sample_count_r2mm": 0,
        "tight_normal_sample_count_r6mm": 0,
        "two_sided_transition_score": 0.0,
        "two_sided_normal_angle_deg": 0.0,
        "two_sided_support_balance": 0.0,
        "two_sided_min_support_fraction": 0.0,
        "two_sided_normal_coherence": 0.0,
    }
    for candidate in candidates:
        for name, value in defaults.items():
            candidate[name] = value
    if not candidates or not len(support_points):
        return
    points = np.asarray(support_points, dtype=np.float64)
    normals = np.asarray(support_normals, dtype=np.float64)
    tree = cKDTree(points)
    parameters = np.linspace(0.0, 1.0, max(3, int(sample_count)), dtype=np.float64)
    for candidate in candidates:
        endpoints = _candidate_endpoints(candidate)
        delta = endpoints[1] - endpoints[0]
        span = float(np.linalg.norm(delta))
        if span <= 1e-9:
            continue
        axis = delta / span
        samples = endpoints[0] + parameters[:, None] * delta
        neighborhoods = tree.query_ball_point(samples, r=float(corridor_radius_mm), workers=1)
        indices = np.asarray(sorted({index for group in neighborhoods for index in group}), dtype=np.int64)
        if not len(indices):
            continue
        local_points = points[indices]
        local_normals = normals[indices]
        relative = local_points - endpoints[0]
        longitudinal = relative @ axis
        radial = relative - longitudinal[:, None] * axis
        radial_length = np.linalg.norm(radial, axis=1)
        all_normal_lengths = np.linalg.norm(local_normals, axis=1)
        all_valid_normals = all_normal_lengths > 1e-8
        central_span = (longitudinal >= 0.1 * span) & (longitudinal <= 0.9 * span)
        tight_values: dict[float, float] = {}
        for radius_mm in (2.0, 6.0):
            tight_mask = (
                central_span
                & (radial_length <= radius_mm)
                & all_valid_normals
            )
            tight_count = int(np.count_nonzero(tight_mask))
            candidate[f"tight_normal_sample_count_r{int(radius_mm)}mm"] = tight_count
            value = 0.0
            if tight_count >= 20:
                tight_normals = (
                    local_normals[tight_mask]
                    / all_normal_lengths[tight_mask, None]
                )
                moment = (
                    np.einsum("ni,nj->ij", tight_normals, tight_normals)
                    / tight_count
                )
                eigenvalues = np.sort(np.linalg.eigvalsh(moment))[::-1]
                value = float(eigenvalues[1])
            tight_values[radius_mm] = value
            candidate[
                f"tight_normal_second_eigenvalue_r{int(radius_mm)}mm"
            ] = value
        candidate["tight_to_wide_normal_second_ratio"] = float(
            tight_values[2.0] / max(tight_values[6.0], 1e-12)
            if tight_values[6.0] > 0.0
            else 0.0
        )
        mask = (
            (longitudinal >= -2.0)
            & (longitudinal <= span + 2.0)
            & (radial_length >= 0.5)
            & (radial_length <= corridor_radius_mm)
        )
        radial = radial[mask]
        local_normals = local_normals[mask]
        radial_length = radial_length[mask]
        if len(radial) < 6:
            continue
        first, second = _basis(axis)
        angles = np.arctan2(radial @ second, radial @ first)
        occupied = np.unique(np.floor(np.mod(angles, 2.0 * np.pi) / (2.0 * np.pi) * 16).astype(int))
        normal_lengths = np.linalg.norm(local_normals, axis=1)
        valid_normals = normal_lengths > 1e-8
        radial_with_normals = radial[valid_normals]
        local_normals = local_normals[valid_normals] / normal_lengths[valid_normals, None]
        if len(local_normals):
            normal_moment = np.einsum("ni,nj->ij", local_normals, local_normals) / len(local_normals)
            eigenvalues = np.sort(np.linalg.eigvalsh(normal_moment))[::-1]
            dominant = np.linalg.eigh(normal_moment)[1][:, -1]
            normal_angles = np.degrees(
                np.arccos(np.clip(np.abs(local_normals @ dominant), 0.0, 1.0))
            )
            second_eigenvalue = float(eigenvalues[1])
            normal_spread = float(np.quantile(normal_angles, 0.90))
        else:
            second_eigenvalue = 0.0
            normal_spread = 0.0
        candidate["corridor_point_count"] = int(len(radial))
        candidate["corridor_points_per_mm"] = float(len(radial) / max(span, 1.0))
        candidate["transverse_angular_span_deg"] = _circular_span_deg(angles)
        candidate["transverse_angular_bin_fraction"] = float(len(occupied) / 16.0)
        candidate["median_corridor_radius_mm"] = float(np.median(radial_length))
        candidate["normal_second_eigenvalue"] = second_eigenvalue
        candidate["normal_spread_deg_scan"] = normal_spread
        candidate.update(
            _two_sided_transition_features(
                radial_with_normals,
                local_normals,
                first,
                second,
            )
        )


def _attach_case_relative_features(line_objects: list[dict[str, Any]]) -> None:
    feature_names = (
        "support_count",
        "score",
        "consensus_count",
        "finite_support_fraction",
        "finite_support_longest_run_fraction",
        "endpoint_support_fraction",
        "normal_second_eigenvalue",
        "normal_spread_deg_scan",
        "tight_normal_second_eigenvalue_r2mm",
        "tight_to_wide_normal_second_ratio",
        "corridor_points_per_mm",
        "transverse_angular_span_deg",
        "cluster_member_count",
        "cluster_source_count",
        "two_sided_transition_score",
        "two_sided_normal_angle_deg",
        "two_sided_support_balance",
        "two_sided_min_support_fraction",
        "two_sided_normal_coherence",
    )
    if not line_objects:
        return
    for name in feature_names:
        values = np.asarray([float(item.get(name) or 0.0) for item in line_objects], dtype=np.float64)
        percentiles = rankdata(values, method="average") / len(values)
        for item, percentile in zip(line_objects, percentiles):
            item[f"case_percentile_{name}"] = float(percentile)


def build_case_pool(
    *,
    decomposition: Mapping[str, Any],
    points: np.ndarray,
    normals: np.ndarray,
    plane_points: np.ndarray | None = None,
    plane_normals: np.ndarray | None = None,
    scan_mesh_adjacency: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    points, normals = deterministic_geometry_sample(
        points,
        normals,
        max_points=GLOBAL_POINT_BUDGET,
    )
    if plane_points is None:
        plane_points, plane_normals = points, normals
    else:
        if plane_normals is None:
            raise ValueError("plane_normals are required when plane_points are supplied")
        plane_points, plane_normals = deterministic_geometry_sample(
            plane_points,
            plane_normals,
            max_points=GLOBAL_POINT_BUDGET,
        )
    clean = list(decomposition.get("owned_bend_regions") or ())
    hypotheses = list(decomposition.get("bend_hypotheses") or ())
    support_strips = _support_strip_lines(decomposition)
    evidence_traces = _evidence_trace_lines(decomposition)
    merged = merge_collinear_fragments(clean)
    connectors = complete_parallel_endpoint_connectors(merged)
    cylindrical_standard = complete_cylindrical_bands(
        points,
        normals,
        hypotheses,
        point_budget=GLOBAL_POINT_BUDGET,
        corridor_radius_mm=12.0,
    )
    cylindrical_wide = complete_cylindrical_bands(
        points,
        normals,
        hypotheses,
        point_budget=GLOBAL_POINT_BUDGET,
        corridor_radius_mm=30.0,
    )
    planes = extract_deterministic_planes(plane_points)
    plane_lattice = build_supported_plane_intersections(plane_points, planes)
    support_atom_table, support_atom_edges = _compact_support_graph(decomposition)
    transition_spacing_mm = max(float(_estimate_local_spacing(points)), 1e-3)
    transition_normals = _repair_zero_normals(
        points,
        normals,
        local_spacing_mm=transition_spacing_mm,
    )
    transition_graph = build_transition_atom_graph_v2(
        points,
        transition_normals,
        local_spacing_mm=transition_spacing_mm,
    )
    transition_payload = compact_transition_graph_v2(transition_graph)
    plane_table, plane_mode_memberships = _enrich_transition_plane_table(
        plane_lattice.get("merged_planes") or (),
        plane_points,
        transition_payload,
    )
    mesh_candidates = _trim_to_longest_scan_supported_run(
        _tag(list(scan_mesh_adjacency or ()), "scan_mesh_adjacency"),
        points,
    )
    candidates = [
        *_tag(clean, "strict_blind_owned"),
        *_tag(hypotheses, "bend_hypothesis"),
        *_tag(support_strips, "control_support_strip"),
        *_tag(evidence_traces, "evidence_trace"),
        *_tag(merged, "fragment_merge"),
        *_tag(connectors, "endpoint_connector"),
        *_tag(cylindrical_standard.get("candidates") or [], "cylindrical_standard"),
        *_tag(cylindrical_wide.get("candidates") or [], "cylindrical_wide"),
        *_tag(plane_lattice.get("candidates") or [], "plane_lattice"),
        *mesh_candidates,
    ]
    _attach_finite_scan_support(candidates, plane_points)
    _attach_local_bend_support(candidates, plane_points, plane_normals)
    line_objects = cluster_line_candidates(candidates)
    _attach_case_relative_features(line_objects)
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate["candidate_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    payload = {
        "schema_version": 3,
        "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
        "inference_contract": {
            "part_identifier_exposed": False,
            "expected_count_exposed": False,
            "manual_truth_exposed": False,
            "cad_exposed": False,
        },
        "source_point_serialization": False,
        "working_point_contract_version": "GLOBAL_POINT_BUDGET_V1",
        "working_point_budget": GLOBAL_POINT_BUDGET,
        "working_point_budget_applies_to": [
            "finite_scan_support",
            "local_bend_support",
            "cylindrical_standard",
            "cylindrical_wide",
            "plane_extraction",
            "plane_lattice",
            "mesh_candidate_support_trim",
        ],
        "support_atom_serialization": "bounded_identifiers_only",
        "support_atom_id_cap_per_candidate": SUPPORT_ATOM_ID_CAP_PER_CANDIDATE,
        "support_atom_table_cap": SUPPORT_ATOM_TABLE_CAP,
        "support_atom_edge_cap": SUPPORT_ATOM_EDGE_CAP,
        "support_atom_table": support_atom_table,
        "support_atom_edges": support_atom_edges,
        "support_representation_profile": transition_payload[
            "support_representation_profile"
        ],
        "support_parent_table": transition_payload["support_parent_table"],
        "support_mode_table": transition_payload["support_mode_table"],
        "support_mode_edges": transition_payload["support_mode_edges"],
        "plane_mode_memberships": plane_mode_memberships,
        "transition_atom_caps": transition_payload["transition_atom_caps"],
        "transition_parent_table_sha256": transition_payload["parent_table_sha256"],
        "transition_mode_table_sha256": transition_payload["mode_table_sha256"],
        "transition_mode_edge_table_sha256": transition_payload[
            "mode_edge_table_sha256"
        ],
        "transition_local_spacing_mm": transition_payload["local_spacing_mm"],
        "transition_parent_voxel_size_mm": transition_payload[
            "parent_voxel_size_mm"
        ],
        "transition_fine_cell_size_mm": transition_payload["fine_cell_size_mm"],
        "transition_atom_diagnostics": transition_payload["diagnostics"],
        "transient_sampled_point_count": int(len(points)),
        "transient_plane_point_count": int(len(plane_points)),
        "scan_bbox_mm": np.ptp(points, axis=0).astype(np.float64).tolist() if len(points) else [0.0, 0.0, 0.0],
        "candidate_count": len(candidates),
        "line_object_count": len(line_objects),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "generator_diagnostics": {
            "cylindrical_standard": {
                key: value for key, value in cylindrical_standard.items() if key != "candidates"
            },
            "cylindrical_wide": {
                key: value for key, value in cylindrical_wide.items() if key != "candidates"
            },
            "plane_lattice": {
                key: value
                for key, value in plane_lattice.items()
                if key not in {"candidates", "merged_planes"}
            },
            "extracted_plane_count": len(planes),
            "scan_mesh_adjacency_candidate_count": len(scan_mesh_adjacency or ()),
        },
        "plane_table": plane_table,
        "candidates": candidates,
        "line_objects": line_objects,
    }
    _reject_recursive_forbidden_keys(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT / "output/enterprise_benchmark/blind-ply-strict-scan-only-20260716-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/f1-10of13/strict-blind-uniform-candidate-pool-v1",
    )
    parser.add_argument("--voxel-mm", type=float, default=1.8)
    parser.add_argument(
        "--case-id",
        help="Optionally process one opaque case in a fresh bounded process",
    )
    args = parser.parse_args()

    case_map = _discover_isolated_case_map(args.prediction_root.resolve())
    if args.case_id:
        if args.case_id not in case_map:
            raise ValueError(f"unknown opaque case id: {args.case_id}")
        case_map = {args.case_id: case_map[args.case_id]}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, (case_id, decomposition_path) in enumerate(case_map.items(), start=1):
        decomposition = _read(decomposition_path)
        provenance = dict(decomposition.get("provenance") or {})
        if provenance.get("coordinate_frame") != "scan_raw":
            raise ValueError(f"{case_id}: candidate generation requires scan_raw coordinates")
        if decomposition.get("part_id") is not None:
            raise ValueError(f"{case_id}: strict blind decomposition exposed part identity")
        scan_path = Path(str(decomposition["scan_path"])).resolve()
        print(f"[{index}/{len(case_map)}] {case_id}: {scan_path.name}", flush=True)
        points, normals, plane_points, plane_normals = _load_geometry_views(
            scan_path,
            plane_voxel_mm=args.voxel_mm,
        )
        scan_mesh_adjacency, scan_mesh_diagnostics = _scan_mesh_adjacency_candidates(scan_path)
        if int(scan_mesh_diagnostics.get("vertex_count") or 0) > GLOBAL_POINT_BUDGET:
            raise ValueError(
                f"{case_id}: scan-mesh adjacency exceeded global point budget"
            )
        payload = build_case_pool(
            decomposition=decomposition,
            points=points,
            normals=normals,
            plane_points=plane_points,
            plane_normals=plane_normals,
            scan_mesh_adjacency=scan_mesh_adjacency,
        )
        payload["case_id"] = case_id
        payload["scan_path"] = str(scan_path)
        payload["coordinate_frame"] = "scan_raw"
        payload["generator_diagnostics"]["scan_mesh_adjacency"] = scan_mesh_diagnostics
        case_path = output_dir / f"{case_id}.json"
        case_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "case_id": case_id,
                "path": str(case_path),
                "candidate_count": payload["candidate_count"],
                "line_object_count": payload["line_object_count"],
                "candidate_source_counts": payload["candidate_source_counts"],
                "transition_parent_count": len(payload["support_parent_table"]),
                "transition_mode_count": len(payload["support_mode_table"]),
                "transition_mode_edge_count": len(payload["support_mode_edges"]),
            }
        )
    manifest = {
        "schema_version": 3,
        "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
        "case_count": len(manifest_rows),
        "source_point_serialization": False,
        "support_atom_serialization": "bounded_identifiers_only",
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "support_atom_id_cap_per_candidate": SUPPORT_ATOM_ID_CAP_PER_CANDIDATE,
        "working_point_contract_version": "GLOBAL_POINT_BUDGET_V1",
        "working_point_budget": GLOBAL_POINT_BUDGET,
        "voxel_mm": float(args.voxel_mm),
        "cases": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(manifest_rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
