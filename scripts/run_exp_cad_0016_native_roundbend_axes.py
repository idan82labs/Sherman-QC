#!/usr/bin/env python3
"""Render source-native SolidWorks RoundBend axes on aligned physical scans.

This diagnostic does not choose bend events from the scan or from a generic mesh
edge detector.  It decodes the native SLDPRT history, requires agreement between
the active ``RoundBend`` feature count and the source ``Bends`` property, reads
the source sheet thickness, and finds folded-body cylindrical surface pairs that
are coaxial and separated by exactly that thickness.  A finite neutral axis is
then bounded by the two source face tessellations.

The decoder currently does not expose a complete RoundBend-name-to-face mapping,
so event names are assigned in deterministic geometric order.  Count agreement
is therefore an integrity constraint, not a fabricated per-feature provenance.
No labels, expected evaluator counts, scan identity, or score are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


SUPPORTED_CADMPEG_VERSION = "cadmpeg 0.5.5"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _xyz(value: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray([value[axis] for axis in ("x", "y", "z")], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("invalid finite xyz coordinate")
    return result


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid cylinder axis")
    result = axis / norm
    rounded = tuple(float(value) for value in np.round(result, 12))
    if rounded < tuple(-value for value in rounded):
        result = -result
    return result


def _custom_property(cadir: Mapping[str, Any], name: str) -> str:
    properties = cadir.get("native", {}).get("sldprt", {}).get("custom_properties", [])
    matches = []
    for item in properties:
        if not isinstance(item, Mapping) or item.get("name") != name:
            continue
        for value in item.get("values", []):
            if isinstance(value, Mapping) and value.get("value") is not None:
                matches.append(str(value["value"]))
    if not matches:
        # cadmpeg versions may keep the same records in the history arena only.
        history = cadir.get("native", {}).get("sldprt", {}).get("arenas", {}).get("features", [])
        matches = [str(item.get("text")) for item in history if item.get("name") == name and item.get("text") is not None]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one native {name!r} property, got {unique}")
    return unique[0]


def extract_native_roundbend_axes(
    cadir: Mapping[str, Any],
    *,
    coaxial_tolerance_mm: float = 0.02,
    thickness_tolerance_mm: float = 0.02,
) -> dict[str, Any]:
    """Return one finite neutral axis per source-native RoundBend event."""
    model = cadir.get("model", {})
    arenas = cadir.get("native", {}).get("sldprt", {}).get("arenas", {})
    features = [
        item for item in arenas.get("features", [])
        if isinstance(item, Mapping)
        and item.get("kind") == "RoundBend"
        and item.get("suppressed") is False
    ]
    feature_names = [str(item.get("name")) for item in features]
    if not features or len(set(feature_names)) != len(features):
        raise ValueError("native active RoundBend inventory is empty or non-unique")
    source_bend_count = int(_custom_property(cadir, "Bends"))
    thickness_mm = float(_custom_property(cadir, "Sheet Metal Thickness"))
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
        vertices = [_xyz(value) for value in tessellation.get("vertices", [])]
        for face_id in tessellation["faces"]:
            face_vertices.setdefault(str(face_id), []).extend(vertices)

    cylinders = []
    for surface in model.get("surfaces", []):
        if not isinstance(surface, Mapping):
            continue
        geometry = surface.get("geometry", {})
        surface_id = str(surface.get("id"))
        face_id = face_by_surface.get(surface_id)
        if geometry.get("kind") != "cylinder" or not face_id or face_id not in face_vertices:
            continue
        axis = _canonical_axis(_xyz(geometry["axis"]))
        origin = _xyz(geometry["origin"])
        foot = origin - axis * float(np.dot(origin, axis))
        radius = float(geometry["radius"])
        cylinders.append(
            {
                "surface_id": surface_id,
                "face_id": face_id,
                "axis": axis,
                "foot": foot,
                "radius_mm": radius,
                "vertices": face_vertices[face_id],
            }
        )

    pairs = []
    used: set[str] = set()
    for index, first in enumerate(cylinders):
        for second in cylinders[index + 1 :]:
            parallel_error = 1.0 - abs(float(np.dot(first["axis"], second["axis"])))
            coaxial_error = float(np.linalg.norm(first["foot"] - second["foot"]))
            thickness_error = abs(abs(first["radius_mm"] - second["radius_mm"]) - thickness_mm)
            if parallel_error > 1e-8 or coaxial_error > coaxial_tolerance_mm or thickness_error > thickness_tolerance_mm:
                continue
            if first["surface_id"] in used or second["surface_id"] in used:
                raise ValueError("a source cylinder participates in multiple qualifying bend pairs")
            axis = first["axis"]
            if float(np.dot(axis, second["axis"])) < 0:
                axis = -axis
            foot = (first["foot"] + second["foot"]) / 2.0
            vertices = np.vstack([*first["vertices"], *second["vertices"]])
            parameters = vertices @ axis
            start = foot + axis * float(parameters.min())
            end = foot + axis * float(parameters.max())
            if float(np.linalg.norm(end - start)) <= 2.0 * thickness_mm:
                raise ValueError("qualifying bend pair has implausibly short finite support")
            used.update((first["surface_id"], second["surface_id"]))
            pairs.append(
                {
                    "axis": axis,
                    "foot": foot,
                    "line_start_mm": start,
                    "line_end_mm": end,
                    "span_mm": float(np.linalg.norm(end - start)),
                    "surface_ids": sorted((first["surface_id"], second["surface_id"])),
                    "face_ids": sorted((first["face_id"], second["face_id"])),
                    "radii_mm": sorted((float(first["radius_mm"]), float(second["radius_mm"]))),
                    "coaxial_error_mm": coaxial_error,
                    "thickness_error_mm": thickness_error,
                }
            )

    pairs.sort(key=lambda item: tuple(float(value) for value in np.round(item["foot"], 9)))
    if len(pairs) != source_bend_count:
        raise ValueError(f"source declares {source_bend_count} bends but geometry yielded {len(pairs)} pairs")
    events = []
    for ordinal, pair in enumerate(pairs, 1):
        events.append(
            {
                "event_id": f"NATIVE_ROUNDBEND_{ordinal:02d}",
                "line_start_mm": pair["line_start_mm"].tolist(),
                "line_end_mm": pair["line_end_mm"].tolist(),
                "span_mm": pair["span_mm"],
                "surface_ids": pair["surface_ids"],
                "face_ids": pair["face_ids"],
                "radii_mm": pair["radii_mm"],
                "coaxial_error_mm": pair["coaxial_error_mm"],
                "thickness_error_mm": pair["thickness_error_mm"],
            }
        )
    return {
        "selector": "ACTIVE_ROUNDBEND_COUNT_AND_SOURCE_THICKNESS_COAXIAL_CYLINDER_PAIR_V1",
        "source_bend_count": source_bend_count,
        "active_roundbend_feature_count": len(features),
        "active_roundbend_feature_names": feature_names,
        "sheet_thickness_mm": thickness_mm,
        "qualifying_pair_count": len(events),
        "events": events,
        "feature_to_pair_name_binding": "UNAVAILABLE_IN_DECODER_DO_NOT_INFER",
    }


def _equal_axes(axis, points: np.ndarray) -> None:
    mins, maxs = points.min(0), points.max(0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render(path: Path, case_token: str, mesh, scan_points: np.ndarray, events: list[dict[str, Any]]) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    scan = scan_points[:: max(1, len(scan_points) // 80000)]
    combined = np.vstack([vertices, scan])
    views = ((20, -65, "oblique"), (90, -90, "top"), (5, 0, "side"))
    figure = plt.figure(figsize=(30, 18), dpi=100, facecolor="#0b1016")
    for index, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, index, projection="3d", facecolor="#0b1016")
        axis.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles,
            color="#bec6cf", edgecolor="#7f8b98", linewidth=0.12, alpha=0.28, shade=True,
        )
        axis.scatter(scan[:, 0], scan[:, 1], scan[:, 2], color="#46d9ff", s=0.7, alpha=0.32, linewidths=0)
        for event in events:
            start = np.asarray(event["line_start_mm"], dtype=float)
            end = np.asarray(event["line_end_mm"], dtype=float)
            axis.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color="#ff2da8", linewidth=5.0, solid_capstyle="round",
            )
            midpoint = (start + end) / 2.0
            axis.text(midpoint[0], midpoint[1], midpoint[2], event["event_id"].split("_")[-1], color="#ffd8ef", fontsize=9)
        axis.view_init(elev=elevation, azim=azimuth)
        _equal_axes(axis, combined)
        axis.set_title(label, color="white", fontsize=18, pad=14)
        axis.set_axis_off()
    figure.suptitle(
        f"{case_token} — source-native SolidWorks RoundBend axes (magenta)\n"
        "cyan: aligned authenticated scan; grey: folded CAD",
        color="white", fontsize=23, y=0.96,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def decode_cadir(source: Path, cadmpeg: Path) -> dict[str, Any]:
    version = subprocess.run([str(cadmpeg), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if version != SUPPORTED_CADMPEG_VERSION:
        raise ValueError(f"unsupported cadmpeg version {version!r}")
    with tempfile.TemporaryDirectory(prefix="exp16-native-roundbend-") as temporary:
        output = Path(temporary) / "source.cadir.json"
        result = subprocess.run(
            [str(cadmpeg), "convert", str(source), "--input-format", "sldprt", "--format", "cadir", "--output", str(output)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cadmpeg failed: {result.stderr[-1200:]}")
        return json.loads(output.read_text(encoding="utf-8"))


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

    source_hash = sha256_path(args.source_sldprt)
    if source_hash != args.expected_source_sha256:
        raise ValueError("source SLDPRT hash does not match authenticated manifest")
    cadir = decode_cadir(args.source_sldprt.resolve(), args.cadmpeg.resolve())
    extraction = extract_native_roundbend_axes(cadir)
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError("engine mesh is empty")

    records = []
    for case_token in args.case_token:
        semantic_path = args.registration_root / "cases" / case_token / "semantic_result.json"
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        if semantic["source_cad_sha256"] != source_hash:
            raise ValueError(f"registration source CAD mismatch for {case_token}")
        if semantic["engine_mesh_sha256"] != sha256_path(args.mesh):
            raise ValueError(f"registration engine mesh mismatch for {case_token}")
        aligned_scan_path = Path(semantic["artifacts"]["aligned_scan"]["path"])
        if sha256_path(aligned_scan_path) != semantic["artifacts"]["aligned_scan"]["sha256"]:
            raise ValueError(f"aligned scan hash mismatch for {case_token}")
        scan_points = np.asarray(o3d.io.read_point_cloud(str(aligned_scan_path)).points, dtype=np.float64)
        case_root = args.output_root / "cases" / case_token
        render_path = case_root / "native_three_view_roundbend_overlay.png"
        render(render_path, case_token, mesh, scan_points, extraction["events"])
        case_payload = {
            "schema": "EXP_CAD_0016_NATIVE_ROUNDBEND_AXIS_CASE_V1",
            "case_token": case_token,
            "registration_semantic_sha256": sha256_path(semantic_path),
            "source_sldprt_sha256": source_hash,
            "engine_mesh_sha256": sha256_path(args.mesh),
            "extraction": extraction,
            "render": {"path": str(render_path.resolve()), "sha256": sha256_path(render_path)},
            "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
            "evaluator_data_opened": False,
            "labels_or_expected_counts_used": False,
            "is_metric_eligible": False,
        }
        atomic_json(case_root / "semantic_result.json", case_payload)
        records.append(
            {
                "case_token": case_token,
                "semantic_sha256": sha256_path(case_root / "semantic_result.json"),
                "render_sha256": sha256_path(render_path),
            }
        )
    summary = {
        "schema": "EXP_CAD_0016_NATIVE_ROUNDBEND_AXIS_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0016",
        "source_sldprt_sha256": source_hash,
        "cadmpeg_sha256": sha256_path(args.cadmpeg),
        "engine_mesh_sha256": sha256_path(args.mesh),
        "case_count": len(records),
        "extraction": extraction,
        "records": records,
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
    }
    atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({"case_count": len(records), "event_count": len(extraction["events"]), "events": extraction["events"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
