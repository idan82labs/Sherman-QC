"""Frozen V53 adjacency/lattice two-bank consensus selector.

The selector is deliberately append-only.  It consumes a frozen V52 result,
the corresponding frozen V51 candidate pool, and the current source mesh.
No truth, identity, expected-count, or CAD input is accepted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d

from backend.cad_bend_extractor import CADBendExtractor
from backend.f1_cylindrical_consensus_recovery_v52 import _represented
from scripts.evaluate_f1_against_manual_gold import _endpoints


PROFILE = "F1_V53_ADJACENCY_LATTICE_BANK_CONSENSUS_V1"
PRECOMMIT_TOKEN = "V53_ALBC_MECHANISM_LOCKED"
MESH_ROUTE = "scan_mesh_adjacency"
PLANE_ROUTE = "plane_lattice"
MINIMUM_CLUSTER_SOURCE_COUNT = 3
MAXIMUM_PAIR_COUNT = 4096
MAXIMUM_AXIS_ANGLE_DEG = 7.0
MAXIMUM_INFINITE_AXIS_DISTANCE_MM = 6.0
MINIMUM_PAIR_SHORTER_SPAN_OVERLAP = 0.80
MINIMUM_COMMON_SPAN_TO_BBOX_RATIO = 0.50
PLANE_MEMBERSHIP_TOLERANCE_MM = 1.0
MINIMUM_BANK_OVERLAP = 0.60
AXIAL_BIN_COUNT = 8
MINIMUM_VERTICES_PER_BIN = 4
MINIMUM_OCCUPIED_BINS = 7


@dataclass(frozen=True)
class PackedBank:
    data: np.ndarray
    vertex_count: int


@dataclass(frozen=True)
class PairGeometry:
    interval: tuple[float, float]
    common_span_mm: float
    common_span_to_bbox_ratio: float
    pair_shorter_span_overlap: float
    axis_angle_deg: float
    infinite_axis_distance_mm: float


def _canonical_axis(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    line = _endpoints(dict(record))
    if line.shape != (2, 3) or not np.isfinite(line).all():
        raise ValueError("line endpoints must be finite")
    delta = line[1] - line[0]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("line span must be nonzero")
    axis = delta / length
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
    return np.mean(line, axis=0), axis


def _line_geometry(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    line = _endpoints(dict(record))
    if line.shape != (2, 3) or not np.isfinite(line).all():
        raise ValueError("line endpoints must be finite")
    delta = line[1] - line[0]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("line span must be nonzero")
    return np.mean(line, axis=0), delta / length, length


def _infinite_axis_distance(
    first_anchor: np.ndarray,
    first_axis: np.ndarray,
    second_anchor: np.ndarray,
    second_axis: np.ndarray,
) -> float:
    normal = np.cross(first_axis, second_axis)
    norm = float(np.linalg.norm(normal))
    delta = second_anchor - first_anchor
    if norm <= 1e-12:
        return float(np.linalg.norm(delta - first_axis * np.dot(delta, first_axis)))
    return abs(float(np.dot(delta, normal))) / norm


def _projected_interval(
    record: Mapping[str, Any], origin: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    values = sorted(float(np.dot(point - origin, axis)) for point in _endpoints(dict(record)))
    return values[0], values[1]


def pair_geometry(
    mesh_candidate: Mapping[str, Any],
    plane_candidate: Mapping[str, Any],
    cluster: Mapping[str, Any],
    bbox_scale_mm: float,
) -> PairGeometry | None:
    origin, canonical_axis = _canonical_axis(cluster)
    mesh_anchor, mesh_axis, mesh_span = _line_geometry(mesh_candidate)
    plane_anchor, plane_axis, plane_span = _line_geometry(plane_candidate)
    dot = float(np.clip(abs(np.dot(mesh_axis, plane_axis)), 0.0, 1.0))
    angle = math.degrees(math.acos(dot))
    distance = _infinite_axis_distance(mesh_anchor, mesh_axis, plane_anchor, plane_axis)
    mesh_interval = _projected_interval(mesh_candidate, origin, canonical_axis)
    plane_interval = _projected_interval(plane_candidate, origin, canonical_axis)
    low = max(mesh_interval[0], plane_interval[0])
    high = min(mesh_interval[1], plane_interval[1])
    common = max(0.0, high - low)
    overlap = common / max(min(mesh_span, plane_span), 1e-12)
    ratio = common / max(float(bbox_scale_mm), 1e-12)
    if (
        angle > MAXIMUM_AXIS_ANGLE_DEG
        or distance > MAXIMUM_INFINITE_AXIS_DISTANCE_MM
        or overlap < MINIMUM_PAIR_SHORTER_SPAN_OVERLAP
        or ratio < MINIMUM_COMMON_SPAN_TO_BBOX_RATIO
    ):
        return None
    return PairGeometry((low, high), common, ratio, overlap, angle, distance)


def _pack(mask: np.ndarray) -> PackedBank:
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    return PackedBank(np.packbits(values, bitorder="little"), int(len(values)))


def _unpack(bank: PackedBank) -> np.ndarray:
    return np.unpackbits(bank.data, count=bank.vertex_count, bitorder="little").astype(bool)


def _restrict(bank: PackedBank, restriction: PackedBank) -> PackedBank:
    if bank.vertex_count != restriction.vertex_count:
        raise ValueError("packed-bank ordinal mismatch")
    return PackedBank(np.bitwise_and(bank.data, restriction.data), bank.vertex_count)


def _subtract(bank: PackedBank, other: PackedBank) -> PackedBank:
    if bank.vertex_count != other.vertex_count:
        raise ValueError("packed-bank ordinal mismatch")
    return _pack(_unpack(bank) & ~_unpack(other))


def _intersection(first: PackedBank, second: PackedBank) -> PackedBank:
    if first.vertex_count != second.vertex_count:
        raise ValueError("packed-bank ordinal mismatch")
    return PackedBank(np.bitwise_and(first.data, second.data), first.vertex_count)


def _count(bank: PackedBank) -> int:
    return int(np.count_nonzero(_unpack(bank)))


def _overlap_coefficient(first: PackedBank, second: PackedBank) -> float:
    denominator = min(_count(first), _count(second))
    if denominator <= 0:
        return 0.0
    return float(_count(_intersection(first, second)) / denominator)


def _source_vertices_and_mesh_banks(
    scan_path: Path,
    pool: Mapping[str, Any],
    mesh_candidate_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, tuple[PackedBank, PackedBank]]]:
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("malformed or non-finite source geometry")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ValueError("source mesh requires triangle connectivity")
    diagnostics = dict((pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {})
    if diagnostics.get("bounded_decimation_used") is not False:
        raise ValueError("source-vertex ordinal mismatch after bounded decimation")
    if int(diagnostics.get("source_vertex_count") or -1) != len(vertices):
        raise ValueError("source-vertex ordinal mismatch")
    extraction = CADBendExtractor().extract_from_mesh(mesh)
    surfaces = {int(surface.surface_id): surface for surface in extraction.surfaces}
    bends = {f"{MESH_ROUTE}::SMA{int(bend.bend_id)}": bend for bend in extraction.bends}
    frozen = {str(row.get("candidate_id")): row for row in pool.get("candidates") or ()}
    result: dict[str, tuple[PackedBank, PackedBank]] = {}
    for candidate_id in sorted(set(mesh_candidate_ids)):
        row = frozen.get(candidate_id)
        bend = bends.get(candidate_id)
        if row is None or bend is None:
            raise ValueError("unable to reproduce frozen mesh-adjacency hypothesis")
        owners = (int(row["surface1_id"]), int(row["surface2_id"]))
        if owners != (int(bend.surface1_id), int(bend.surface2_id)):
            raise ValueError("frozen mesh-adjacency ownership changed")
        masks = []
        for owner in owners:
            surface = surfaces.get(owner)
            if surface is None:
                raise ValueError("frozen mesh-adjacency surface is unavailable")
            face_ids = np.asarray(surface.face_indices, dtype=np.int64)
            mask = np.zeros(len(vertices), dtype=bool)
            mask[np.unique(triangles[face_ids].reshape(-1))] = True
            masks.append(_pack(mask))
        result[candidate_id] = (masks[0], masks[1])
    return vertices, result


def _plane_banks(
    vertices: np.ndarray,
    pool: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> dict[str, tuple[PackedBank, PackedBank]]:
    candidates = {str(row.get("candidate_id")): row for row in pool.get("candidates") or ()}
    planes = {str(row.get("plane_id")): row for row in pool.get("plane_table") or ()}
    result: dict[str, tuple[PackedBank, PackedBank]] = {}
    for candidate_id in sorted(set(candidate_ids)):
        row = candidates.get(candidate_id)
        if row is None:
            raise ValueError("frozen plane-lattice hypothesis is unavailable")
        plane_ids = tuple(str(value) for value in row.get("incident_plane_ids") or ())
        if len(plane_ids) != 2 or plane_ids[0] == plane_ids[1]:
            raise ValueError("plane-lattice hypothesis lacks two parents")
        banks = []
        for plane_id in plane_ids:
            plane = planes.get(plane_id)
            if plane is None:
                raise ValueError("frozen parent plane is unavailable")
            normal = np.asarray(plane.get("normal"), dtype=np.float64)
            offset = float(plane.get("offset"))
            if normal.shape != (3,) or not np.isfinite(normal).all() or not math.isfinite(offset):
                raise ValueError("malformed frozen parent plane")
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                raise ValueError("zero frozen parent-plane normal")
            normal = normal / norm
            offset /= norm
            banks.append(_pack(np.abs(vertices @ normal + offset) <= PLANE_MEMBERSHIP_TOLERANCE_MM))
        result[candidate_id] = (banks[0], banks[1])
    return result


def _bin_mask(
    bank: PackedBank,
    projections: np.ndarray,
    interval: tuple[float, float],
) -> int:
    members = np.flatnonzero(_unpack(bank))
    if not len(members):
        return 0
    low, high = interval
    width = (high - low) / AXIAL_BIN_COUNT
    values = projections[members]
    indices = np.floor((values - low) / width).astype(np.int64)
    indices[values == high] = AXIAL_BIN_COUNT - 1
    mask = 0
    for index in range(AXIAL_BIN_COUNT):
        if int(np.count_nonzero(indices == index)) >= MINIMUM_VERTICES_PER_BIN:
            mask |= 1 << index
    return mask


def _certificate(
    mesh_banks: tuple[PackedBank, PackedBank],
    plane_banks: tuple[PackedBank, PackedBank],
    vertices: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    geometry: PairGeometry,
) -> dict[str, Any] | None:
    projections = (vertices - origin) @ axis
    low, high = geometry.interval
    axial = _pack((projections >= low) & (projections <= high))
    adjacency = tuple(_restrict(bank, axial) for bank in mesh_banks)
    lattice = tuple(_restrict(bank, axial) for bank in plane_banks)
    adjacency = (_subtract(adjacency[0], adjacency[1]), _subtract(adjacency[1], adjacency[0]))
    lattice = (_subtract(lattice[0], lattice[1]), _subtract(lattice[1], lattice[0]))
    if min(*(_count(bank) for bank in adjacency), *(_count(bank) for bank in lattice)) <= 0:
        return None
    unswapped = (
        _overlap_coefficient(adjacency[0], lattice[0]),
        _overlap_coefficient(adjacency[1], lattice[1]),
    )
    swapped = (
        _overlap_coefficient(adjacency[0], lattice[1]),
        _overlap_coefficient(adjacency[1], lattice[0]),
    )
    use_swapped = min(swapped) > min(unswapped)
    matched_lattice = (lattice[1], lattice[0]) if use_swapped else lattice
    overlaps = swapped if use_swapped else unswapped
    if min(overlaps) < MINIMUM_BANK_OVERLAP:
        return None
    intersections = (
        _intersection(adjacency[0], matched_lattice[0]),
        _intersection(adjacency[1], matched_lattice[1]),
    )
    bin_masks = tuple(_bin_mask(bank, projections, geometry.interval) for bank in intersections)
    bin_counts = tuple(mask.bit_count() for mask in bin_masks)
    if min(bin_counts) < MINIMUM_OCCUPIED_BINS:
        return None
    return {
        "bank_overlap_coefficients": [round(float(value), 12) for value in overlaps],
        "matched_bank_intersection_counts": [_count(bank) for bank in intersections],
        "matched_bank_bin_masks": list(bin_masks),
        "matched_bank_occupied_bin_counts": list(bin_counts),
        "plane_bank_swap": bool(use_swapped),
    }


def _eligible_pairs(
    v52_predictions: list[dict[str, Any]], pool: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows = list(pool.get("candidates") or ())
    by_id = {str(row.get("candidate_id")): row for row in candidate_rows}
    ordinal = {str(row.get("candidate_id")): index for index, row in enumerate(candidate_rows)}
    clusters = []
    pairs = []
    bbox_scale = max((float(value) for value in pool.get("scan_bbox_mm") or ()), default=0.0)
    for cluster_ordinal, raw_cluster in enumerate(pool.get("line_objects") or ()):
        cluster = dict(raw_cluster)
        sources = {str(value) for value in cluster.get("cluster_sources") or ()}
        if (
            len(sources) < MINIMUM_CLUSTER_SOURCE_COUNT
            or MESH_ROUTE not in sources
            or PLANE_ROUTE not in sources
            or _represented(cluster, v52_predictions)
        ):
            continue
        member_ids = [str(value) for value in cluster.get("cluster_member_ids") or ()]
        meshes = [value for value in member_ids if value.startswith(MESH_ROUTE + "::") and value in by_id]
        planes = [value for value in member_ids if value.startswith(PLANE_ROUTE + "::") and value in by_id]
        cluster_record = {
            "cluster": cluster,
            "cluster_ordinal": cluster_ordinal,
            "mesh_ids": meshes,
            "plane_ids": planes,
        }
        clusters.append(cluster_record)
        for mesh_id in meshes:
            for plane_id in planes:
                pairs.append(
                    {
                        **cluster_record,
                        "mesh_id": mesh_id,
                        "plane_id": plane_id,
                        "mesh": by_id[mesh_id],
                        "plane": by_id[plane_id],
                        "mesh_ordinal": ordinal[mesh_id],
                        "plane_ordinal": ordinal[plane_id],
                        "geometry": pair_geometry(by_id[mesh_id], by_id[plane_id], cluster, bbox_scale),
                    }
                )
    return clusters, pairs


def _audit(status: str, **values: Any) -> dict[str, Any]:
    return {
        "status": status,
        "precommit_token": PRECOMMIT_TOKEN,
        "minimum_cluster_source_count": MINIMUM_CLUSTER_SOURCE_COUNT,
        "maximum_pair_count": MAXIMUM_PAIR_COUNT,
        "maximum_axis_angle_deg": MAXIMUM_AXIS_ANGLE_DEG,
        "maximum_infinite_axis_distance_mm": MAXIMUM_INFINITE_AXIS_DISTANCE_MM,
        "minimum_pair_shorter_span_overlap": MINIMUM_PAIR_SHORTER_SPAN_OVERLAP,
        "minimum_common_span_to_bbox_ratio": MINIMUM_COMMON_SPAN_TO_BBOX_RATIO,
        "plane_membership_tolerance_mm": PLANE_MEMBERSHIP_TOLERANCE_MM,
        "minimum_bank_overlap": MINIMUM_BANK_OVERLAP,
        "axial_bin_count": AXIAL_BIN_COUNT,
        "minimum_vertices_per_bin": MINIMUM_VERTICES_PER_BIN,
        "minimum_occupied_bins": MINIMUM_OCCUPIED_BINS,
        **values,
    }


def _finalize(v52_result: Mapping[str, Any], predictions: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(v52_result))
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v53_adjacency_lattice_bank_consensus_result",
            "profile": PROFILE,
            "v52_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "adjacency_lattice_bank_consensus_audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    compact = json.dumps(
        {"profile": PROFILE, "predictions": predictions, "audit": audit},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result["v53_result_sha256"] = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    return result


def compose_v53(
    v52_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    scan_path: Path,
) -> dict[str, Any]:
    """Append certified ALBC lines or deterministically abstain for the scan."""

    if v52_result.get("source_only") is not True or v52_result.get("research_only") is not True:
        raise ValueError("V53 requires a source-only research V52 input")
    prefix = [deepcopy(row) for row in v52_result.get("predictions") or ()]
    try:
        clusters, pairs = _eligible_pairs(prefix, pool)
        if len(pairs) > MAXIMUM_PAIR_COUNT:
            return _finalize(
                v52_result,
                prefix,
                _audit(
                    "PAIR_CAP_ABSTAIN",
                    eligible_cluster_count=len(clusters),
                    eligible_pair_count=len(pairs),
                    geometrically_eligible_pair_count=0,
                    certified_pair_count=0,
                    recovered_candidate_count=0,
                    recovered_candidates=[],
                ),
            )
        geometric = [pair for pair in pairs if pair["geometry"] is not None]
        if not geometric:
            return _finalize(
                v52_result,
                prefix,
                _audit(
                    "COMMITTED",
                    eligible_cluster_count=len(clusters),
                    eligible_pair_count=len(pairs),
                    geometrically_eligible_pair_count=0,
                    certified_pair_count=0,
                    recovered_candidate_count=0,
                    recovered_candidates=[],
                ),
            )
        mesh_ids = [str(pair["mesh_id"]) for pair in geometric]
        plane_ids = [str(pair["plane_id"]) for pair in geometric]
        vertices, mesh_banks = _source_vertices_and_mesh_banks(scan_path, pool, mesh_ids)
        plane_banks = _plane_banks(vertices, pool, plane_ids)
        certified = []
        for pair in geometric:
            origin, axis = _canonical_axis(pair["cluster"])
            certificate = _certificate(
                mesh_banks[pair["mesh_id"]],
                plane_banks[pair["plane_id"]],
                vertices,
                origin,
                axis,
                pair["geometry"],
            )
            if certificate is None:
                continue
            overlaps = certificate["bank_overlap_coefficients"]
            bins = certificate["matched_bank_occupied_bin_counts"]
            pair["certificate"] = certificate
            pair["rank"] = (
                float(min(overlaps)),
                float(sum(overlaps)),
                int(min(bins)),
                float(pair["geometry"].common_span_to_bbox_ratio),
            )
            certified.append(pair)
        best_by_cluster: dict[int, dict[str, Any]] = {}
        for pair in certified:
            key = int(pair["cluster_ordinal"])
            candidate_key = (
                -pair["rank"][0],
                -pair["rank"][1],
                -pair["rank"][2],
                -pair["rank"][3],
                int(pair["mesh_ordinal"]),
                int(pair["plane_ordinal"]),
            )
            current = best_by_cluster.get(key)
            if current is None or candidate_key < current["sort_key"]:
                pair["sort_key"] = candidate_key
                best_by_cluster[key] = pair
        ordered = sorted(
            best_by_cluster.values(),
            key=lambda pair: (
                -pair["rank"][0],
                -pair["rank"][1],
                -pair["rank"][2],
                -pair["rank"][3],
                int(pair["cluster_ordinal"]),
            ),
        )
        predictions = [deepcopy(row) for row in prefix]
        additions = []
        for pair in ordered:
            cluster = pair["cluster"]
            origin, axis = _canonical_axis(cluster)
            low, high = pair["geometry"].interval
            proposed = {
                "line_start": (origin + low * axis).tolist(),
                "line_end": (origin + high * axis).tolist(),
            }
            if _represented(proposed, predictions):
                continue
            cluster_id = str(cluster.get("bend_id") or cluster.get("candidate_id") or "")
            prediction = {
                "bend_id": "TRV53-" + hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:20],
                **proposed,
                "selection_route": "adjacency_lattice_bank_consensus_v53",
                "source_hypothesis_id": cluster_id,
                "source_kind": "adjacency_lattice_bank_consensus",
            }
            predictions.append(prediction)
            certificate = pair["certificate"]
            additions.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_ordinal": int(pair["cluster_ordinal"]),
                    "mesh_route_ordinal": int(pair["mesh_ordinal"]),
                    "plane_route_ordinal": int(pair["plane_ordinal"]),
                    "common_span_mm": round(pair["geometry"].common_span_mm, 12),
                    "common_span_to_bbox_ratio": round(
                        pair["geometry"].common_span_to_bbox_ratio, 12
                    ),
                    "bank_overlap_coefficients": certificate["bank_overlap_coefficients"],
                    "matched_bank_intersection_counts": certificate[
                        "matched_bank_intersection_counts"
                    ],
                    "matched_bank_bin_masks": certificate["matched_bank_bin_masks"],
                    "matched_bank_occupied_bin_counts": certificate[
                        "matched_bank_occupied_bin_counts"
                    ],
                    "line_geometry": proposed,
                }
            )
        return _finalize(
            v52_result,
            predictions,
            _audit(
                "COMMITTED",
                eligible_cluster_count=len(clusters),
                eligible_pair_count=len(pairs),
                geometrically_eligible_pair_count=len(geometric),
                certified_pair_count=len(certified),
                recovered_candidate_count=len(additions),
                recovered_candidates=additions,
            ),
        )
    except Exception as error:  # Frozen rule: any internal error abstains scan-wide.
        return _finalize(
            v52_result,
            prefix,
            _audit(
                "SCAN_LEVEL_ABSTAIN",
                abstention_reason=type(error).__name__,
                eligible_cluster_count=0,
                eligible_pair_count=0,
                geometrically_eligible_pair_count=0,
                certified_pair_count=0,
                recovered_candidate_count=0,
                recovered_candidates=[],
            ),
        )
