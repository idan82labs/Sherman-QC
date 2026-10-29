#!/usr/bin/env python3
"""Patch-aware source-native SolidWorks RoundBend axis diagnostic.

V1 required one decoded face at each inner/outer cylindrical radius.  Native
sheet-metal B-reps can split one coaxial carrier into multiple disconnected face
patches; each disconnected, interval-matched patch is an independently formed
inspection event.  This successor clusters coaxial carriers, pairs radius levels
by the source sheet thickness, and then pairs inner/outer patches only when their
finite axial intervals agree.  The final event count must still equal both the
active RoundBend inventory and the native ``Bends`` property.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "run_exp_cad_0016_native_roundbend_axes.py"
SPEC = importlib.util.spec_from_file_location("exp16_native_roundbend", V1_PATH)
V1 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(V1)


def _radius_groups(items: list[dict[str, Any]], tolerance_mm: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda value: (value["radius_mm"], value["surface_id"])):
        for group in groups:
            if abs(item["radius_mm"] - group[0]["radius_mm"]) <= tolerance_mm:
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def extract_native_roundbend_patch_axes(
    cadir: Mapping[str, Any],
    *,
    coaxial_tolerance_mm: float = 0.02,
    thickness_tolerance_mm: float = 0.02,
    radius_level_tolerance_mm: float = 0.002,
    interval_endpoint_tolerance_mm: float = 0.05,
) -> dict[str, Any]:
    model = cadir.get("model", {})
    arenas = cadir.get("native", {}).get("sldprt", {}).get("arenas", {})
    features = [
        item for item in arenas.get("features", [])
        if isinstance(item, Mapping) and item.get("kind") == "RoundBend" and item.get("suppressed") is False
    ]
    feature_names = [str(item.get("name")) for item in features]
    if not features or len(set(feature_names)) != len(features):
        raise ValueError("native active RoundBend inventory is empty or non-unique")
    source_bend_count = int(V1._custom_property(cadir, "Bends"))
    thickness_mm = float(V1._custom_property(cadir, "Sheet Metal Thickness"))
    if source_bend_count != len(features):
        raise ValueError("native Bends property disagrees with active RoundBend count")
    if not math.isfinite(thickness_mm) or thickness_mm <= 0:
        raise ValueError("invalid source sheet thickness")

    face_by_surface = {
        str(face["surface"]): str(face["id"])
        for face in model.get("faces", [])
        if isinstance(face, Mapping) and face.get("surface") and face.get("id")
    }
    face_vertices: dict[str, list[np.ndarray]] = {}
    for tessellation in model.get("tessellations", []):
        if not isinstance(tessellation, Mapping) or not tessellation.get("body") or not tessellation.get("faces"):
            continue
        vertices = [V1._xyz(value) for value in tessellation.get("vertices", [])]
        for face_id in tessellation["faces"]:
            face_vertices.setdefault(str(face_id), []).extend(vertices)

    patches: list[dict[str, Any]] = []
    for surface in model.get("surfaces", []):
        if not isinstance(surface, Mapping):
            continue
        geometry = surface.get("geometry", {})
        surface_id = str(surface.get("id"))
        face_id = face_by_surface.get(surface_id)
        if geometry.get("kind") != "cylinder" or not face_id or face_id not in face_vertices:
            continue
        axis = V1._canonical_axis(V1._xyz(geometry["axis"]))
        origin = V1._xyz(geometry["origin"])
        foot = origin - axis * float(np.dot(origin, axis))
        vertices = np.vstack(face_vertices[face_id])
        parameters = vertices @ axis
        patches.append(
            {
                "surface_id": surface_id,
                "face_id": face_id,
                "axis": axis,
                "foot": foot,
                "radius_mm": float(geometry["radius"]),
                "t_min_mm": float(parameters.min()),
                "t_max_mm": float(parameters.max()),
            }
        )

    axis_clusters: list[list[dict[str, Any]]] = []
    for patch in sorted(patches, key=lambda value: value["surface_id"]):
        for cluster in axis_clusters:
            parallel_error = 1.0 - abs(float(np.dot(patch["axis"], cluster[0]["axis"])))
            coaxial_error = float(np.linalg.norm(patch["foot"] - cluster[0]["foot"]))
            if parallel_error <= 1e-8 and coaxial_error <= coaxial_tolerance_mm:
                cluster.append(patch)
                break
        else:
            axis_clusters.append([patch])

    raw_events: list[dict[str, Any]] = []
    used_surfaces: set[str] = set()
    qualifying_cluster_count = 0
    for cluster in axis_clusters:
        radius_groups = _radius_groups(cluster, radius_level_tolerance_mm)
        level_pairs = []
        for index, first in enumerate(radius_groups):
            for second in radius_groups[index + 1 :]:
                error = abs(abs(first[0]["radius_mm"] - second[0]["radius_mm"]) - thickness_mm)
                if error <= thickness_tolerance_mm:
                    level_pairs.append((first, second, error))
        if not level_pairs:
            continue
        qualifying_cluster_count += 1
        for first, second, thickness_error in level_pairs:
            if len(first) != len(second):
                raise ValueError("coaxial radius levels have unequal disconnected patch counts")
            first_sorted = sorted(first, key=lambda value: (value["t_min_mm"], value["t_max_mm"], value["surface_id"]))
            second_sorted = sorted(second, key=lambda value: (value["t_min_mm"], value["t_max_mm"], value["surface_id"]))
            for inner, outer in zip(first_sorted, second_sorted):
                endpoint_error = max(
                    abs(inner["t_min_mm"] - outer["t_min_mm"]),
                    abs(inner["t_max_mm"] - outer["t_max_mm"]),
                )
                if endpoint_error > interval_endpoint_tolerance_mm:
                    raise ValueError("inner/outer cylindrical patches lack matching finite support")
                if inner["surface_id"] in used_surfaces or outer["surface_id"] in used_surfaces:
                    raise ValueError("a source cylinder patch participates in multiple events")
                used_surfaces.update((inner["surface_id"], outer["surface_id"]))
                axis = inner["axis"]
                if float(np.dot(axis, outer["axis"])) < 0:
                    axis = -axis
                foot = (inner["foot"] + outer["foot"]) / 2.0
                t_min = (inner["t_min_mm"] + outer["t_min_mm"]) / 2.0
                t_max = (inner["t_max_mm"] + outer["t_max_mm"]) / 2.0
                start = foot + axis * t_min
                end = foot + axis * t_max
                raw_events.append(
                    {
                        "axis": axis,
                        "foot": foot,
                        "line_start_mm": start,
                        "line_end_mm": end,
                        "span_mm": float(np.linalg.norm(end - start)),
                        "surface_ids": sorted((inner["surface_id"], outer["surface_id"])),
                        "face_ids": sorted((inner["face_id"], outer["face_id"])),
                        "radii_mm": sorted((float(inner["radius_mm"]), float(outer["radius_mm"]))),
                        "interval_endpoint_error_mm": endpoint_error,
                        "thickness_error_mm": thickness_error,
                    }
                )

    raw_events.sort(
        key=lambda item: (
            tuple(float(value) for value in np.round(item["foot"], 8)),
            tuple(float(value) for value in np.round(item["line_start_mm"], 8)),
        )
    )
    if len(raw_events) != source_bend_count:
        raise ValueError(f"source declares {source_bend_count} bends but patch geometry yielded {len(raw_events)} events")
    events = []
    for ordinal, event in enumerate(raw_events, 1):
        events.append(
            {
                "event_id": f"NATIVE_ROUNDBEND_PATCH_{ordinal:02d}",
                "line_start_mm": event["line_start_mm"].tolist(),
                "line_end_mm": event["line_end_mm"].tolist(),
                "span_mm": event["span_mm"],
                "surface_ids": event["surface_ids"],
                "face_ids": event["face_ids"],
                "radii_mm": event["radii_mm"],
                "interval_endpoint_error_mm": event["interval_endpoint_error_mm"],
                "thickness_error_mm": event["thickness_error_mm"],
            }
        )
    return {
        "selector": "ACTIVE_ROUNDBEND_COUNT_SOURCE_THICKNESS_COAXIAL_INTERVAL_PATCH_PAIR_V1",
        "source_bend_count": source_bend_count,
        "active_roundbend_feature_count": len(features),
        "active_roundbend_feature_names": feature_names,
        "sheet_thickness_mm": thickness_mm,
        "qualifying_axis_cluster_count": qualifying_cluster_count,
        "qualifying_event_count": len(events),
        "events": events,
        "feature_to_patch_name_binding": "UNAVAILABLE_IN_DECODER_DO_NOT_INFER",
    }


def render_projection_closeup(
    path: Path,
    case_token: str,
    mesh,
    scan_points: np.ndarray,
    events: list[dict[str, Any]],
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    scan = scan_points[:: max(1, len(scan_points) // 120000)]
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    figure, axes = plt.subplots(1, 3, figsize=(30, 18), dpi=100, facecolor="#0b1016")
    for axis, (horizontal, vertical, title) in zip(axes, projections):
        axis.set_facecolor("#0b1016")
        axis.scatter(vertices[:, horizontal], vertices[:, vertical], color="#aeb8c4", s=1.8, alpha=0.32, linewidths=0)
        axis.scatter(scan[:, horizontal], scan[:, vertical], color="#46d9ff", s=0.8, alpha=0.30, linewidths=0)
        for event in events:
            start = np.asarray(event["line_start_mm"], dtype=np.float64)
            end = np.asarray(event["line_end_mm"], dtype=np.float64)
            axis.plot(
                [start[horizontal], end[horizontal]],
                [start[vertical], end[vertical]],
                color="#ff2da8",
                linewidth=4.5,
                solid_capstyle="round",
            )
            midpoint = (start + end) / 2.0
            axis.text(
                midpoint[horizontal], midpoint[vertical], event["event_id"].split("_")[-1],
                color="#ffd8ef", fontsize=9,
            )
        combined_h = np.concatenate((vertices[:, horizontal], scan[:, horizontal]))
        combined_v = np.concatenate((vertices[:, vertical], scan[:, vertical]))
        h_min, h_max = np.percentile(combined_h, [0.05, 99.95])
        v_min, v_max = np.percentile(combined_v, [0.05, 99.95])
        h_margin = max((h_max - h_min) * 0.04, 1.0)
        v_margin = max((v_max - v_min) * 0.04, 1.0)
        axis.set_xlim(h_min - h_margin, h_max + h_margin)
        axis.set_ylim(v_min - v_margin, v_max + v_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, color="white", fontsize=19, pad=14)
        axis.set_axis_off()
    figure.suptitle(
        f"{case_token} — native RoundBend patch axes, candidate-bound orthographic close-up\n"
        "magenta: source-native axes; cyan: authenticated scan; grey: folded CAD",
        color="white", fontsize=23, y=0.96,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sldprt", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--cadmpeg", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--case-token", action="append", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    source_hash = V1.sha256_path(args.source_sldprt)
    if source_hash != args.expected_source_sha256:
        raise ValueError("source SLDPRT hash does not match authenticated manifest")
    extraction = extract_native_roundbend_patch_axes(V1.decode_cadir(args.source_sldprt.resolve(), args.cadmpeg.resolve()))
    mesh_hash = V1.sha256_path(args.mesh)
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError("engine mesh is empty")
    records = []
    for case_token in args.case_token:
        semantic_path = args.registration_root / "cases" / case_token / "semantic_result.json"
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        if semantic["source_cad_sha256"] != source_hash or semantic["engine_mesh_sha256"] != mesh_hash:
            raise ValueError(f"registration source binding mismatch for {case_token}")
        aligned_scan = Path(semantic["artifacts"]["aligned_scan"]["path"])
        if V1.sha256_path(aligned_scan) != semantic["artifacts"]["aligned_scan"]["sha256"]:
            raise ValueError(f"aligned scan hash mismatch for {case_token}")
        scan_points = np.asarray(o3d.io.read_point_cloud(str(aligned_scan)).points, dtype=np.float64)
        case_root = args.output_root / "cases" / case_token
        render_path = case_root / "native_three_view_roundbend_patch_overlay.png"
        closeup_path = case_root / "native_projection_closeup.png"
        V1.render(render_path, case_token, mesh, scan_points, extraction["events"])
        render_projection_closeup(closeup_path, case_token, mesh, scan_points, extraction["events"])
        case_payload = {
            "schema": "EXP_CAD_0017_NATIVE_ROUNDBEND_PATCH_AXIS_CASE_V1",
            "case_token": case_token,
            "registration_semantic_sha256": V1.sha256_path(semantic_path),
            "source_sldprt_sha256": source_hash,
            "engine_mesh_sha256": mesh_hash,
            "extraction": extraction,
            "render": {"path": str(render_path.resolve()), "sha256": V1.sha256_path(render_path)},
            "projection_closeup": {"path": str(closeup_path.resolve()), "sha256": V1.sha256_path(closeup_path)},
            "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
            "evaluator_data_opened": False,
            "labels_or_expected_counts_used": False,
            "is_metric_eligible": False,
        }
        V1.atomic_json(case_root / "semantic_result.json", case_payload)
        records.append({"case_token": case_token, "semantic_sha256": V1.sha256_path(case_root / "semantic_result.json"), "render_sha256": V1.sha256_path(render_path), "projection_closeup_sha256": V1.sha256_path(closeup_path)})
    summary = {
        "schema": "EXP_CAD_0017_NATIVE_ROUNDBEND_PATCH_AXIS_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0017",
        "source_sldprt_sha256": source_hash,
        "cadmpeg_sha256": V1.sha256_path(args.cadmpeg),
        "engine_mesh_sha256": mesh_hash,
        "case_count": len(records),
        "extraction": extraction,
        "records": records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
    }
    V1.atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({"case_count": len(records), "event_count": len(extraction["events"]), "axis_clusters": extraction["qualifying_axis_cluster_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
