#!/usr/bin/env python3
"""Corroborate proper-pose ownership with analytic B-rep cutout carriers.

EXP25 is deliberately independent of EXP24's unresolved DisplayList samples.
It decodes the active analytic B-rep, finds the unique shell containing every
frozen EXP17 native bend face, and extracts cylindrical faces whose axial span
equals the source sheet thickness.  Those 15 through-thickness hole/slot
carriers rank already-frozen EXP22 proper poses; they may not generate or
refine a transform and may not enter bend discovery, classification, or
measurement.

No labels, expected counts, gold, filenames-as-identity, evaluator data,
scanner-frame cue, EXP24 artifact, or DisplayList sample is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP23_PATH = ROOT / "scripts" / "run_exp_cad_0023_symmetry_class_proper_registration.py"
SPEC = importlib.util.spec_from_file_location("exp23_pose_class", EXP23_PATH)
EXP23 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP23)
EXP21 = EXP23.EXP21


SUPPORTED_CADMPEG_VERSION = "cadmpeg 0.5.5"
EXPECTED_CARRIER_FACE_IDS = [
    "sldprt:brep:face#1019",
    "sldprt:brep:face#1026",
    "sldprt:brep:face#1034",
    "sldprt:brep:face#1042",
    "sldprt:brep:face#1050",
    "sldprt:brep:face#1066",
    "sldprt:brep:face#1082",
    "sldprt:brep:face#1098",
    "sldprt:brep:face#1114",
    "sldprt:brep:face#1129",
    "sldprt:brep:face#38",
    "sldprt:brep:face#796",
    "sldprt:brep:face#921",
    "sldprt:brep:face#922",
    "sldprt:brep:face#925",
]
ASYMMETRIC_PROFILE_FACE_IDS = {
    "sldprt:brep:face#1050",
    "sldprt:brep:face#1066",
    "sldprt:brep:face#1082",
    "sldprt:brep:face#1098",
    "sldprt:brep:face#1114",
    "sldprt:brep:face#1129",
}
GEOMETRY_TOLERANCE_MM = 1e-6
CLUSTER_TRANSLATION_MM = 0.01
CLUSTER_ROTATION_DEG = 0.01
CARRIER_TAU_MM = 0.35
FULL_SUPPORT_FLOOR = 10
HALF_SUPPORT_FLOOR = 5
MIN_SUPPORTED_CARRIER_MARGIN = 4
MIN_ASYMMETRIC_CARRIER_MARGIN = 4
MIN_TOTAL_SUPPORT_RATIO = 2.5
TAU_GRID_MM = (0.30, 0.35, 0.40)


def decode_cadir(source: Path, cadmpeg: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    version = subprocess.run(
        [str(cadmpeg), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if version != SUPPORTED_CADMPEG_VERSION:
        raise ValueError(f"unsupported cadmpeg version {version!r}")
    with tempfile.TemporaryDirectory(prefix="exp25-cadir-") as directory:
        root = Path(directory)
        cadir_path = root / "model.cadir.json"
        report_path = root / "decoder.report.json"
        result = subprocess.run(
            [
                str(cadmpeg),
                "convert",
                str(source),
                "--input-format",
                "sldprt",
                "--format",
                "cadir",
                "--output",
                str(cadir_path),
                "--report",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cadmpeg failed: {result.stderr[-1000:]}")
        return (
            json.loads(cadir_path.read_text(encoding="utf-8")),
            json.loads(report_path.read_text(encoding="utf-8")),
        )


def xyz(value: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([value[axis] for axis in "xyz"], dtype=np.float64)


def canonical_unoriented_axis(axis: np.ndarray) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid cylinder axis")
    result = result / norm
    for value in result:
        if abs(float(value)) <= 1e-12:
            continue
        if value < 0.0:
            result = -result
        break
    return result


def index_model(model: Mapping[str, Any], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    indices: dict[str, dict[str, Any]] = {}
    for name in names:
        rows = model.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"missing CADIR collection {name}")
        index = {str(row["id"]): row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f"duplicate CADIR id in {name}")
        indices[name] = index
    return indices


def analytic_through_thickness_carriers(
    cadir: Mapping[str, Any],
    *,
    native_event_face_ids: Sequence[str],
    sheet_thickness_mm: float,
) -> tuple[str, str, list[dict[str, Any]]]:
    model = cadir["model"]
    indices = index_model(
        model,
        (
            "bodies", "regions", "shells", "faces", "loops", "coedges",
            "edges", "vertices", "points", "surfaces", "curves",
        ),
    )
    active = [row for row in model["configurations"] if row.get("active") is True]
    if len(active) != 1 or len(active[0].get("bodies", [])) != 1:
        raise ValueError("active configuration/body is not unique")
    body_id = str(active[0]["bodies"][0])
    body = indices["bodies"].get(body_id)
    if body is None:
        raise ValueError("active body reference is missing")

    shell_ids: list[str] = []
    for region_id in body.get("regions", []):
        region = indices["regions"].get(str(region_id))
        if region is None or region.get("body") != body_id:
            raise ValueError("incomplete active body/region topology")
        shell_ids.extend(str(value) for value in region.get("shells", []))
    required_faces = set(str(value) for value in native_event_face_ids)
    matching_shells = []
    for shell_id in shell_ids:
        shell = indices["shells"].get(shell_id)
        if shell is None or shell.get("region") not in body.get("regions", []):
            raise ValueError("incomplete region/shell topology")
        if required_faces.issubset(set(str(value) for value in shell.get("faces", []))):
            matching_shells.append(shell_id)
    if len(matching_shells) != 1:
        raise ValueError("native event faces do not identify exactly one active shell")
    shell_id = matching_shells[0]
    shell_face_ids = set(str(value) for value in indices["shells"][shell_id]["faces"])

    carriers: list[dict[str, Any]] = []
    for face in model["faces"]:
        face_id = str(face["id"])
        if face_id not in shell_face_ids:
            continue
        surface = indices["surfaces"].get(str(face.get("surface")))
        if surface is None:
            raise ValueError(f"missing surface for {face_id}")
        geometry = surface.get("geometry", {})
        if geometry.get("kind") != "cylinder":
            continue
        surface_axis = canonical_unoriented_axis(xyz(geometry["axis"]))
        surface_radius = float(geometry["radius"])
        circle_rows: dict[str, dict[str, Any]] = {}
        for loop_id in face.get("loops", []):
            loop = indices["loops"].get(str(loop_id))
            if loop is None or loop.get("face") != face_id:
                raise ValueError(f"incomplete face/loop topology for {face_id}")
            for coedge_id in loop.get("coedges", []):
                coedge = indices["coedges"].get(str(coedge_id))
                if coedge is None or coedge.get("owner_loop") != loop_id:
                    raise ValueError(f"incomplete loop/coedge topology for {face_id}")
                edge = indices["edges"].get(str(coedge.get("edge")))
                if edge is None:
                    raise ValueError(f"missing edge for {face_id}")
                curve = indices["curves"].get(str(edge.get("curve")))
                if curve is None:
                    raise ValueError(f"missing curve for {face_id}")
                if curve.get("geometry", {}).get("kind") == "circle":
                    circle_rows[str(curve["id"])] = {
                        "geometry": curve["geometry"],
                        "edge": edge,
                    }
        if len(circle_rows) != 2:
            raise ValueError(f"cylindrical face {face_id} does not have two circle boundaries")
        circles = list(circle_rows.values())
        centers = np.asarray(
            [xyz(row["geometry"]["center"]) for row in circles], dtype=np.float64
        )
        radial_endpoint_directions: list[np.ndarray] = []
        for row in circles:
            circle = row["geometry"]
            if abs(float(circle["radius"]) - surface_radius) > GEOMETRY_TOLERANCE_MM:
                raise ValueError(f"circle/surface radius mismatch for {face_id}")
            circle_axis = canonical_unoriented_axis(xyz(circle["axis"]))
            if float(np.linalg.norm(np.cross(circle_axis, surface_axis))) > GEOMETRY_TOLERANCE_MM:
                raise ValueError(f"circle/surface axis mismatch for {face_id}")
            center = xyz(circle["center"])
            edge = row["edge"]
            for endpoint_name in ("start", "end"):
                vertex_id = str(edge.get(endpoint_name))
                vertex = indices["vertices"].get(vertex_id)
                if vertex is None:
                    raise ValueError(f"missing circle endpoint vertex for {face_id}")
                point = indices["points"].get(str(vertex.get("point")))
                if point is None:
                    raise ValueError(f"missing circle endpoint point for {face_id}")
                radial = xyz(point["position"]) - center
                radial = radial - float(np.dot(radial, surface_axis)) * surface_axis
                radial_norm = float(np.linalg.norm(radial))
                if abs(radial_norm - surface_radius) > GEOMETRY_TOLERANCE_MM:
                    raise ValueError(f"circle endpoint radius mismatch for {face_id}")
                direction = radial / radial_norm
                if not any(
                    float(np.linalg.norm(direction - prior)) <= GEOMETRY_TOLERANCE_MM
                    for prior in radial_endpoint_directions
                ):
                    radial_endpoint_directions.append(direction)
        offset = centers[1] - centers[0]
        axial_offset = float(np.dot(offset, surface_axis))
        perpendicular = offset - axial_offset * surface_axis
        if float(np.linalg.norm(perpendicular)) > GEOMETRY_TOLERANCE_MM:
            raise ValueError(f"circle centers are not coaxial for {face_id}")
        axial_span = abs(axial_offset)
        if abs(axial_span - sheet_thickness_mm) > GEOMETRY_TOLERANCE_MM:
            continue
        if len(radial_endpoint_directions) not in (1, 2):
            raise ValueError(f"ambiguous analytic angular trim for {face_id}")
        reference = xyz(geometry["ref_direction"])
        reference = reference - float(np.dot(reference, surface_axis)) * surface_axis
        reference = reference / float(np.linalg.norm(reference))
        tangent = np.cross(surface_axis, reference)
        tangent = tangent / float(np.linalg.norm(tangent))
        full_circle = len(radial_endpoint_directions) == 1
        angular_start = 0.0
        angular_span = 2.0 * math.pi
        if not full_circle:
            angles = [
                math.atan2(float(np.dot(direction, tangent)), float(np.dot(direction, reference)))
                % (2.0 * math.pi)
                for direction in radial_endpoint_directions
            ]
            forward = (angles[1] - angles[0]) % (2.0 * math.pi)
            if forward <= math.pi:
                angular_start, angular_span = angles[0], forward
            else:
                angular_start, angular_span = angles[1], 2.0 * math.pi - forward
        carriers.append(
            {
                "face_id": face_id,
                "surface_id": str(surface["id"]),
                "midpoint_mm": np.mean(centers, axis=0),
                "axis": surface_axis,
                "radius_mm": surface_radius,
                "axial_span_mm": axial_span,
                "circle_curve_ids": sorted(circle_rows),
                "full_circle": full_circle,
                "angular_start_rad": angular_start,
                "angular_span_rad": angular_span,
                "angular_reference": reference,
                "angular_tangent": tangent,
                "asymmetric_profile_carrier": face_id in ASYMMETRIC_PROFILE_FACE_IDS,
            }
        )

    actual_ids = [row["face_id"] for row in carriers]
    if actual_ids != EXPECTED_CARRIER_FACE_IDS:
        raise ValueError(f"analytic carrier inventory mismatch: {actual_ids}")
    quantized: set[tuple[int, ...]] = set()
    for row in carriers:
        vector = np.concatenate(
            (
                np.asarray(row["midpoint_mm"]),
                np.asarray(row["axis"]),
                np.asarray([row["radius_mm"], row["axial_span_mm"]]),
            )
        )
        key = tuple(int(round(float(value) / GEOMETRY_TOLERANCE_MM)) for value in vector)
        if key in quantized:
            raise ValueError("duplicate canonical analytic carrier")
        quantized.add(key)
    return body_id, shell_id, carriers


def carrier_masks(
    points: np.ndarray,
    carriers: Sequence[Mapping[str, Any]],
    *,
    sheet_thickness_mm: float,
    tau_mm: float,
) -> list[np.ndarray]:
    masks = []
    for carrier in carriers:
        midpoint = np.asarray(carrier["midpoint_mm"], dtype=np.float64)
        axis = np.asarray(carrier["axis"], dtype=np.float64)
        delta = points - midpoint
        axial = delta @ axis
        radial = delta - axial[:, None] * axis[None, :]
        radial_norm = np.linalg.norm(radial, axis=1)
        radial_residual = np.abs(radial_norm - float(carrier["radius_mm"]))
        mask = (
            (np.abs(axial) <= sheet_thickness_mm / 2.0 + tau_mm)
            & (radial_residual <= tau_mm)
        )
        if not bool(carrier["full_circle"]):
            safe = np.maximum(radial_norm, 1e-12)
            unit = radial / safe[:, None]
            reference = np.asarray(carrier["angular_reference"], dtype=np.float64)
            tangent = np.asarray(carrier["angular_tangent"], dtype=np.float64)
            angles = np.mod(
                np.arctan2(unit @ tangent, unit @ reference), 2.0 * math.pi
            )
            relative = np.mod(angles - float(carrier["angular_start_rad"]), 2.0 * math.pi)
            padding = tau_mm / float(carrier["radius_mm"])
            angular = (
                (relative <= float(carrier["angular_span_rad"]) + padding)
                | (relative >= 2.0 * math.pi - padding)
            )
            mask &= angular
        masks.append(mask)
    return masks


def carrier_metrics(
    points: np.ndarray,
    carriers: Sequence[Mapping[str, Any]],
    *,
    sheet_thickness_mm: float,
    tau_mm: float,
    support_floor: int,
    carrier_indices: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    masks = carrier_masks(
        points, carriers, sheet_thickness_mm=sheet_thickness_mm, tau_mm=tau_mm
    )
    all_counts = [int(np.count_nonzero(mask)) for mask in masks]
    indices = list(range(len(carriers))) if carrier_indices is None else list(carrier_indices)
    counts = [all_counts[index] for index in indices]
    asymmetric_positions = [
        position
        for position, index in enumerate(indices)
        if bool(carriers[index]["asymmetric_profile_carrier"])
    ]
    asymmetric_counts = [counts[position] for position in asymmetric_positions]
    return {
        "tau_mm": float(tau_mm),
        "support_floor": int(support_floor),
        "carrier_indices": indices,
        "counts": counts,
        "supported_carrier_count": int(sum(value >= support_floor for value in counts)),
        "carrier_count": len(indices),
        "asymmetric_supported_carrier_count": int(
            sum(value >= support_floor for value in asymmetric_counts)
        ),
        "asymmetric_carrier_count": len(asymmetric_counts),
        "asymmetric_counts": asymmetric_counts,
        "minimum_count": int(min(counts)),
        "total_count": int(sum(counts)),
        "sum_log1p_count": float(sum(math.log1p(value) for value in counts)),
    }


def transform_near(first: np.ndarray, second: np.ndarray) -> bool:
    motion = EXP23.relative_motion(first, second)
    return bool(
        motion["translation_mm"] <= CLUSTER_TRANSLATION_MM
        and motion["rotation_deg"] <= CLUSTER_ROTATION_DEG
    )


def cluster_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: str(value["candidate_id"])):
        transform = np.asarray(row["absolute_transform"], dtype=np.float64)
        target = None
        for cluster in clusters:
            representative = np.asarray(cluster["representative"]["absolute_transform"], dtype=np.float64)
            if transform_near(transform, representative):
                target = cluster
                break
        if target is None:
            clusters.append({"representative": row, "members": [str(row["candidate_id"])]})
        else:
            target["members"].append(str(row["candidate_id"]))
    return clusters


def rank_clusters(
    clusters: Sequence[Mapping[str, Any]],
    metrics_by_signature: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        clusters,
        key=lambda cluster: (
            -int(metrics_by_signature[tuple(cluster["members"])]["asymmetric_supported_carrier_count"]),
            -int(metrics_by_signature[tuple(cluster["members"])]["supported_carrier_count"]),
            -float(metrics_by_signature[tuple(cluster["members"])]["sum_log1p_count"]),
            -int(metrics_by_signature[tuple(cluster["members"])]["total_count"]),
            str(cluster["representative"]["candidate_id"]),
        ),
    )


def evaluate_variant(
    clusters: Sequence[Mapping[str, Any]],
    carriers: Sequence[Mapping[str, Any]],
    *,
    sheet_thickness_mm: float,
    tau_mm: float,
    support_floor: int,
    point_slice: slice = slice(None),
    carrier_indices: Optional[Sequence[int]] = None,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    metrics: dict[tuple[str, ...], dict[str, Any]] = {}
    for cluster in clusters:
        signature = tuple(cluster["members"])
        points = np.asarray(cluster["representative"]["_transformed_points"])[point_slice]
        metrics[signature] = carrier_metrics(
            points,
            carriers,
            sheet_thickness_mm=sheet_thickness_mm,
            tau_mm=tau_mm,
            support_floor=support_floor,
            carrier_indices=carrier_indices,
        )
    ranked = rank_clusters(clusters, metrics)
    rows = []
    for cluster in ranked:
        signature = tuple(cluster["members"])
        rows.append(
            {
                "representative_candidate_id": cluster["representative"]["candidate_id"],
                "member_candidate_ids": list(signature),
                "metrics": metrics[signature],
            }
        )
    return tuple(ranked[0]["members"]), rows


def validate_source_only_parent(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
    require_generalization_flag: bool,
) -> None:
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unexpected source-only parent schema {payload.get('schema')!r}")
    if payload.get("labels_or_expected_counts_used") is not False:
        raise ValueError("source-only parent used labels or expected counts")
    if payload.get("evaluator_data_opened") is not False:
        raise ValueError("source-only parent opened evaluator data")
    if payload.get("is_metric_eligible") is not False:
        raise ValueError("source-only parent is unexpectedly metric eligible")
    if require_generalization_flag and payload.get("generalization_evidence") is not False:
        raise ValueError("source-only parent contains a generalization claim")


def render_carrier_comparison(
    path: Path,
    *,
    case_token: str,
    carriers: Sequence[Mapping[str, Any]],
    sheet_thickness_mm: float,
    best: Mapping[str, Any],
    runner: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    runner_metrics: Mapping[str, Any],
) -> None:
    candidates = (best, runner)
    metrics = (best_metrics, runner_metrics)
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    analytic_curves = []
    for carrier in carriers:
        sample_count = 128 if carrier["full_circle"] else 64
        angles = np.linspace(
            float(carrier["angular_start_rad"]),
            float(carrier["angular_start_rad"]) + float(carrier["angular_span_rad"]),
            sample_count,
            endpoint=True,
        )
        midpoint = np.asarray(carrier["midpoint_mm"], dtype=np.float64)
        reference = np.asarray(carrier["angular_reference"], dtype=np.float64)
        tangent = np.asarray(carrier["angular_tangent"], dtype=np.float64)
        radius = float(carrier["radius_mm"])
        analytic_curves.append(
            midpoint[None, :]
            + radius * (
                np.cos(angles)[:, None] * reference[None, :]
                + np.sin(angles)[:, None] * tangent[None, :]
            )
        )
    analytic_curve_points = np.vstack(analytic_curves)
    figure, axes = plt.subplots(3, 2, figsize=(30, 24), dpi=100, facecolor="#0b1016")
    for column, (candidate, metric) in enumerate(zip(candidates, metrics)):
        points = np.asarray(candidate["_transformed_points"], dtype=np.float64)
        masks = carrier_masks(
            points,
            carriers,
            sheet_thickness_mm=sheet_thickness_mm,
            tau_mm=CARRIER_TAU_MM,
        )
        supported = np.logical_or.reduce(masks)
        step = max(1, len(points) // 120000)
        background = points[::step]
        foreground = points[supported]
        centers = np.asarray([row["midpoint_mm"] for row in carriers], dtype=np.float64)
        for index, (horizontal, vertical, title) in enumerate(projections):
            axis = axes[index, column]
            axis.set_facecolor("#0b1016")
            axis.scatter(
                background[:, horizontal], background[:, vertical],
                color="#46d9ff", s=0.65, alpha=0.20, linewidths=0,
            )
            axis.scatter(
                analytic_curve_points[:, horizontal], analytic_curve_points[:, vertical],
                color="#ff4fd8", s=2.0, alpha=0.82, linewidths=0,
            )
            axis.scatter(
                foreground[:, horizontal], foreground[:, vertical],
                color="#ffd23f", s=2.0, alpha=0.82, linewidths=0,
            )
            axis.scatter(
                centers[:, horizontal], centers[:, vertical],
                facecolors="none", edgecolors="#ff4fd8", s=80, linewidths=1.5,
            )
            combined = np.vstack((background, centers))
            x0, x1 = np.percentile(combined[:, horizontal], [0.05, 99.95])
            y0, y1 = np.percentile(combined[:, vertical], [0.05, 99.95])
            xm, ym = max((x1 - x0) * 0.04, 1.0), max((y1 - y0) * 0.04, 1.0)
            axis.set_xlim(x0 - xm, x1 + xm)
            axis.set_ylim(y0 - ym, y1 + ym)
            axis.set_aspect("equal", adjustable="box")
            axis.set_axis_off()
            axis.set_title(
                f"{title} — {candidate['candidate_id']}\n"
                f"supported={metric['supported_carrier_count']}/{metric['carrier_count']}; "
                f"asym={metric['asymmetric_supported_carrier_count']}/{metric['asymmetric_carrier_count']}; "
                f"total={metric['total_count']}",
                color="white", fontsize=15, pad=8,
            )
    figure.suptitle(
        f"{case_token} — analytic B-rep through-thickness carrier support\n"
        "cyan: transformed deterministic 0.75 mm scan; yellow: carrier-tube scan points; "
        "magenta: analytic face-trim curves/centers",
        color="white", fontsize=21, y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)


def execute_case(
    *,
    case_token: str,
    scan_path: Path,
    exp22_semantic: Path,
    carriers: Sequence[Mapping[str, Any]],
    sheet_thickness_mm: float,
    output_root: Path,
) -> dict[str, Any]:
    parent = json.loads(exp22_semantic.read_text(encoding="utf-8"))
    validate_source_only_parent(
        parent,
        expected_schema="EXP_CAD_0022_PROPER_RIGID_ORIENTATION_RECOVERY_CASE_V1",
        require_generalization_flag=True,
    )
    if parent.get("case_token") != case_token:
        raise ValueError("EXP22 parent case token mismatch")
    if EXP21.sha256_path(scan_path) != parent["scan_sha256"]:
        raise ValueError("scan binding mismatch")
    raw = np.asarray(o3d.io.read_point_cloud(str(scan_path)).points, dtype=np.float64)
    raw_points, _normals = EXP21.density_capped_points_and_normals(raw)
    eligible: list[dict[str, Any]] = []
    for row in parent["candidate_rows"]:
        strict_pass, strict_reasons = EXP23.strict_event_pass(row["native_event_metrics"])
        stats = row["absolute_transform_stats"]
        transform = np.asarray(row["absolute_transform"], dtype=np.float64)
        recomputed_stats = EXP21.absolute_similarity_stats(transform)
        for key in (
            "proper_similarity_transform",
            "reflection_detected",
            "normalized_rotation_determinant",
            "normalized_rotation_orthogonality_error",
            "uniform_scale",
            "uniform_scale_error",
        ):
            parent_value = stats[key]
            computed_value = recomputed_stats[key]
            if isinstance(computed_value, bool):
                if bool(parent_value) != computed_value:
                    raise ValueError(f"parent transform stat mismatch for {key}")
            elif not math.isclose(float(parent_value), float(computed_value), abs_tol=1e-9):
                raise ValueError(f"parent transform stat mismatch for {key}")
        proper = bool(
            row["proper_rigid_transform"]
            and recomputed_stats["proper_similarity_transform"]
            and not recomputed_stats["reflection_detected"]
            and abs(float(recomputed_stats["uniform_scale"]) - 1.0) <= 1e-6
            and abs(float(recomputed_stats["normalized_rotation_determinant"]) - 1.0) <= 1e-6
        )
        if not proper or not strict_pass:
            continue
        eligible.append(
            {
                "candidate_id": row["candidate_id"],
                "absolute_transform": row["absolute_transform"],
                "strict_native_event_pass": strict_pass,
                "strict_native_event_reasons": strict_reasons,
                "_transformed_points": EXP21.apply_transform(raw_points, transform),
            }
        )
    clusters = cluster_candidates(eligible)
    if len(clusters) < 2:
        raise ValueError("fewer than two distinct proper pose clusters")

    central_tau = CARRIER_TAU_MM
    central_signature, central_rows = evaluate_variant(
        clusters,
        carriers,
        sheet_thickness_mm=sheet_thickness_mm,
        tau_mm=central_tau,
        support_floor=FULL_SUPPORT_FLOOR,
    )
    best_row, runner_row = central_rows[:2]
    best_metrics, runner_metrics = best_row["metrics"], runner_row["metrics"]

    tolerance_checks = []
    for tau in TAU_GRID_MM:
        signature, rows = evaluate_variant(
            clusters,
            carriers,
            sheet_thickness_mm=sheet_thickness_mm,
            tau_mm=tau,
            support_floor=FULL_SUPPORT_FLOOR,
        )
        tolerance_checks.append(
            {
                "tau_mm": tau,
                "selected_member_candidate_ids": list(signature),
                "agrees_with_central": signature == central_signature,
                "selected_metrics": rows[0]["metrics"],
                "runner_up_metrics": rows[1]["metrics"],
                "robustness_contract": "WINNER_IDENTITY_STABILITY_ONLY_PRIMARY_MARGIN_GATES_APPLY_AT_CENTRAL_TAU",
            }
        )
    partition_checks = []
    for name, point_slice in (("EVEN", slice(0, None, 2)), ("ODD", slice(1, None, 2))):
        signature, rows = evaluate_variant(
            clusters,
            carriers,
            sheet_thickness_mm=sheet_thickness_mm,
            tau_mm=central_tau,
            support_floor=HALF_SUPPORT_FLOOR,
            point_slice=point_slice,
        )
        partition_checks.append(
            {
                "partition": name,
                "selected_member_candidate_ids": list(signature),
                "agrees_with_central": signature == central_signature,
                "selected_metrics": rows[0]["metrics"],
                "runner_up_metrics": rows[1]["metrics"],
                "robustness_contract": "WINNER_IDENTITY_STABILITY_ONLY_PRIMARY_MARGIN_GATES_APPLY_TO_FULL_POINT_SET",
            }
        )
    leave_one_out_checks = []
    for omitted in range(len(carriers)):
        indices = [index for index in range(len(carriers)) if index != omitted]
        signature, rows = evaluate_variant(
            clusters,
            carriers,
            sheet_thickness_mm=sheet_thickness_mm,
            tau_mm=central_tau,
            support_floor=FULL_SUPPORT_FLOOR,
            carrier_indices=indices,
        )
        leave_one_out_checks.append(
            {
                "omitted_carrier_index": omitted,
                "omitted_face_id": carriers[omitted]["face_id"],
                "selected_member_candidate_ids": list(signature),
                "agrees_with_central": signature == central_signature,
                "selected_metrics": rows[0]["metrics"],
                "runner_up_metrics": rows[1]["metrics"],
                "robustness_contract": "WINNER_IDENTITY_STABILITY_ONLY_PRIMARY_RATIO_NOT_COMPARABLE_AFTER_CARRIER_REMOVAL",
            }
        )

    reasons = []
    if int(best_metrics["supported_carrier_count"]) != len(carriers):
        reasons.append("BEST_DOES_NOT_SUPPORT_ALL_CARRIERS")
    if int(best_metrics["asymmetric_supported_carrier_count"]) != len(ASYMMETRIC_PROFILE_FACE_IDS):
        reasons.append("BEST_DOES_NOT_SUPPORT_ALL_ASYMMETRIC_CARRIERS")
    support_margin = int(best_metrics["supported_carrier_count"]) - int(
        runner_metrics["supported_carrier_count"]
    )
    if support_margin < MIN_SUPPORTED_CARRIER_MARGIN:
        reasons.append("SUPPORTED_CARRIER_MARGIN_GATE")
    asymmetric_margin = int(best_metrics["asymmetric_supported_carrier_count"]) - int(
        runner_metrics["asymmetric_supported_carrier_count"]
    )
    if asymmetric_margin < MIN_ASYMMETRIC_CARRIER_MARGIN:
        reasons.append("ASYMMETRIC_CARRIER_MARGIN_GATE")
    total_support_ratio = float(best_metrics["total_count"]) / max(
        float(runner_metrics["total_count"]), 1.0
    )
    if total_support_ratio < MIN_TOTAL_SUPPORT_RATIO:
        reasons.append("TOTAL_SUPPORT_RATIO_GATE")
    if not all(row["agrees_with_central"] for row in tolerance_checks):
        reasons.append("TOLERANCE_STABILITY_GATE")
    if not all(row["agrees_with_central"] for row in partition_checks):
        reasons.append("POINT_PARTITION_STABILITY_GATE")
    if not all(row["agrees_with_central"] for row in leave_one_out_checks):
        reasons.append("LEAVE_ONE_CARRIER_OUT_STABILITY_GATE")
    passed = not reasons

    cluster_lookup = {tuple(cluster["members"]): cluster for cluster in clusters}
    best = cluster_lookup[tuple(best_row["member_candidate_ids"])]["representative"]
    runner = cluster_lookup[tuple(runner_row["member_candidate_ids"])]["representative"]
    case_root = output_root / "cases" / case_token
    render_path = case_root / "analytic_carrier_orientation_comparison.png"
    render_carrier_comparison(
        render_path,
        case_token=case_token,
        carriers=carriers,
        sheet_thickness_mm=sheet_thickness_mm,
        best=best,
        runner=runner,
        best_metrics=best_metrics,
        runner_metrics=runner_metrics,
    )
    pose_clusters = []
    for row in central_rows:
        cluster = cluster_lookup[tuple(row["member_candidate_ids"])]
        pose_clusters.append(
            {
                "representative_candidate_id": row["representative_candidate_id"],
                "member_candidate_ids": row["member_candidate_ids"],
                "absolute_transform": cluster["representative"]["absolute_transform"],
                "strict_native_event_pass": cluster["representative"]["strict_native_event_pass"],
                "central_carrier_metrics": row["metrics"],
            }
        )
    payload = {
        "schema": "EXP_CAD_0025_ANALYTIC_BREP_CARRIER_ORIENTATION_CASE_V1",
        "experiment_id": "EXP_CAD_0025",
        "case_token": case_token,
        "scan_sha256": parent["scan_sha256"],
        "source_exp22_semantic_sha256": EXP21.sha256_path(exp22_semantic),
        "eligible_candidate_count": len(eligible),
        "distinct_pose_cluster_count": len(clusters),
        "pose_clusters": pose_clusters,
        "selected_candidate_id": best["candidate_id"] if passed else None,
        "selected_member_candidate_ids": list(central_signature) if passed else [],
        "selected_absolute_transform": best["absolute_transform"] if passed else None,
        "runner_up_candidate_id": runner["candidate_id"],
        "supported_carrier_margin": support_margin,
        "asymmetric_supported_carrier_margin": asymmetric_margin,
        "total_support_ratio": total_support_ratio,
        "tolerance_checks": tolerance_checks,
        "point_partition_checks": partition_checks,
        "leave_one_carrier_out_checks": leave_one_out_checks,
        "orientation_gate_pass": passed,
        "orientation_gate_reasons": reasons,
        "absolute_registration_orientation_authorized": False,
        "authorization_blocker": "SEPARATE_MANUAL_NATIVE_RESOLUTION_REVIEW_REQUIRED",
        "transform_refinement_performed": False,
        "analytic_lane_scope": "FROZEN_POSE_RANKING_ONLY_FORBIDDEN_FOR_BEND_DISCOVERY_CLASSIFICATION_OR_MEASUREMENT",
        "exp24_or_displaylist_artifact_read": False,
        "render": {"path": str(render_path.relative_to(output_root)), "sha256": EXP21.sha256_path(render_path)},
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "labels_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    semantic_path = case_root / "semantic_result.json"
    EXP21.atomic_json(semantic_path, payload)
    return {
        "case_token": case_token,
        "orientation_gate_pass": passed,
        "selected_candidate_id": payload["selected_candidate_id"],
        "selected_member_candidate_ids": payload["selected_member_candidate_ids"],
        "runner_up_candidate_id": payload["runner_up_candidate_id"],
        "supported_carrier_count": best_metrics["supported_carrier_count"],
        "runner_up_supported_carrier_count": runner_metrics["supported_carrier_count"],
        "asymmetric_supported_carrier_count": best_metrics["asymmetric_supported_carrier_count"],
        "runner_up_asymmetric_supported_carrier_count": runner_metrics["asymmetric_supported_carrier_count"],
        "minimum_count": best_metrics["minimum_count"],
        "total_count": best_metrics["total_count"],
        "runner_up_total_count": runner_metrics["total_count"],
        "total_support_ratio": total_support_ratio,
        "sum_log1p_count": best_metrics["sum_log1p_count"],
        "semantic_sha256": EXP21.sha256_path(semantic_path),
        "render_sha256": EXP21.sha256_path(render_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-token", action="append", required=True)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--source-sldprt", required=True, type=Path)
    parser.add_argument("--cadmpeg", required=True, type=Path)
    parser.add_argument("--exp17-summary", required=True, type=Path)
    parser.add_argument("--exp22-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    exp17 = json.loads(args.exp17_summary.read_text(encoding="utf-8"))
    validate_source_only_parent(
        exp17,
        expected_schema="EXP_CAD_0017_NATIVE_ROUNDBEND_PATCH_AXIS_SUMMARY_V1",
        require_generalization_flag=False,
    )
    source_sha = EXP21.sha256_path(args.source_sldprt)
    cadmpeg_sha = EXP21.sha256_path(args.cadmpeg)
    if source_sha != exp17["source_sldprt_sha256"]:
        raise ValueError("source SLDPRT differs from frozen EXP17 source")
    if cadmpeg_sha != exp17["cadmpeg_sha256"]:
        raise ValueError("cadmpeg differs from frozen EXP17 dependency")
    sheet_thickness_mm = float(exp17["extraction"]["sheet_thickness_mm"])
    native_faces = [
        face_id
        for event in exp17["extraction"]["events"]
        for face_id in event["face_ids"]
    ]
    if len(native_faces) != 8 or len(set(native_faces)) != 8:
        raise ValueError("frozen EXP17 native event face inventory is not exactly eight")
    cadir, decoder_report = decode_cadir(args.source_sldprt.resolve(), args.cadmpeg.resolve())
    decode = decoder_report.get("decode_report", {})
    if not decode.get("geometry_transferred") or decode.get("container_only"):
        raise ValueError("decoder did not transfer geometry")
    body_id, shell_id, carriers = analytic_through_thickness_carriers(
        cadir,
        native_event_face_ids=native_faces,
        sheet_thickness_mm=sheet_thickness_mm,
    )
    carrier_manifest = [
        {
            "face_id": row["face_id"],
            "surface_id": row["surface_id"],
            "midpoint_mm": np.asarray(row["midpoint_mm"]).tolist(),
            "axis": np.asarray(row["axis"]).tolist(),
            "radius_mm": row["radius_mm"],
            "axial_span_mm": row["axial_span_mm"],
            "circle_curve_ids": row["circle_curve_ids"],
            "full_circle": row["full_circle"],
            "angular_start_rad": row["angular_start_rad"],
            "angular_span_rad": row["angular_span_rad"],
            "angular_reference": np.asarray(row["angular_reference"]).tolist(),
            "angular_tangent": np.asarray(row["angular_tangent"]).tolist(),
            "asymmetric_profile_carrier": row["asymmetric_profile_carrier"],
        }
        for row in carriers
    ]
    records = []
    for case_token in args.case_token:
        print(case_token, flush=True)
        records.append(
            execute_case(
                case_token=case_token,
                scan_path=(args.scan_root / f"{case_token}.ply").resolve(),
                exp22_semantic=(args.exp22_root / "cases" / case_token / "semantic_result.json").resolve(),
                carriers=carriers,
                sheet_thickness_mm=sheet_thickness_mm,
                output_root=args.output_root.resolve(),
            )
        )
    selected_ids = {
        row["selected_candidate_id"] for row in records if row["orientation_gate_pass"]
    }
    all_cases_numeric_gate_pass = bool(records) and all(
        row["orientation_gate_pass"] for row in records
    )
    summary = {
        "schema": "EXP_CAD_0025_ANALYTIC_BREP_CARRIER_ORIENTATION_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0025",
        "source_sldprt_sha256": source_sha,
        "cadmpeg_sha256": cadmpeg_sha,
        "source_exp17_summary_sha256": EXP21.sha256_path(args.exp17_summary),
        "active_body_id": body_id,
        "native_event_shell_id": shell_id,
        "sheet_thickness_mm": sheet_thickness_mm,
        "carrier_count": len(carriers),
        "carrier_manifest": carrier_manifest,
        "gate_config": {
            "carrier_tau_mm": CARRIER_TAU_MM,
            "full_support_floor": FULL_SUPPORT_FLOOR,
            "half_support_floor": HALF_SUPPORT_FLOOR,
            "minimum_supported_carrier_margin": MIN_SUPPORTED_CARRIER_MARGIN,
            "minimum_asymmetric_carrier_margin": MIN_ASYMMETRIC_CARRIER_MARGIN,
            "minimum_total_support_ratio": MIN_TOTAL_SUPPORT_RATIO,
            "tau_grid_mm": list(TAU_GRID_MM),
            "cluster_translation_mm": CLUSTER_TRANSLATION_MM,
            "cluster_rotation_deg": CLUSTER_ROTATION_DEG,
            "geometry_tolerance_mm": GEOMETRY_TOLERANCE_MM,
            "expected_carrier_face_ids": EXPECTED_CARRIER_FACE_IDS,
            "asymmetric_profile_face_ids": sorted(ASYMMETRIC_PROFILE_FACE_IDS),
            "robustness_variant_contract": "WINNER_IDENTITY_STABILITY_ONLY_CENTRAL_FULL_SET_RETAINS_ABSOLUTE_MARGIN_GATES",
        },
        "case_count": len(records),
        "orientation_gate_pass_count": int(sum(row["orientation_gate_pass"] for row in records)),
        "all_cases_numeric_gate_pass": all_cases_numeric_gate_pass,
        "cross_scan_selected_candidate_ids": sorted(selected_ids),
        "cross_scan_selected_candidate_id_consistency": bool(
            all_cases_numeric_gate_pass and len(selected_ids) == 1
        ),
        "cross_scan_consistency_interpretation": "EACH_SCAN_INDEPENDENTLY_SELECTS_ONE_SOURCE_CAD_ORIENTATION; PCA_CANDIDATE_ID_EQUALITY_IS_RECORDED_BUT_NOT_ASSUMED_TO_BE_A_PHYSICAL_CLASS_INVARIANT",
        "records": records,
        "analytic_lane_scope": "FROZEN_POSE_RANKING_ONLY_FORBIDDEN_FOR_BEND_DISCOVERY_CLASSIFICATION_OR_MEASUREMENT",
        "exp24_or_displaylist_artifact_read": False,
        "transform_refinement_performed": False,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_INDEPENDENT_ANALYTIC_VISUAL_GATE",
        "labels_or_expected_counts_used": False,
        "evaluator_data_opened": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    EXP21.atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
