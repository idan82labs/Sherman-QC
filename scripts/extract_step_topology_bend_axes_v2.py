#!/usr/bin/env python3
"""V2 native bend-axis extraction with representation-consistent eligibility.

Unlike the legacy mesh-crease route, this preserves STEP B-Rep surface types.
A sheet-metal bend is represented by coaxial inner/outer cylindrical faces with
distinct radii, a partial angular sweep, and a useful common axial span. Full
cylinders (holes) and thickness-edge fillets are excluded geometrically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


MIN_ANGULAR_SWEEP_DEG = 15.0
MAX_ANGULAR_SWEEP_DEG = 200.0
MIN_AXIAL_SPAN_MM = 8.0
MAX_COAXIAL_OFFSET_MM = 0.5
MAX_AXIS_ANGLE_DEG = 0.5
MIN_COMMON_SPAN_FRACTION = 0.80
MIN_RADIUS_SEPARATION_MM = 0.2
MAX_RADIUS_SEPARATION_MM = 10.0


ANGULAR_PUBLICATION_DECIMALS = 6
MINIMUM_SWEEP_REPRESENTATION_POLICY = "SIX_DECIMAL_DEGREES_AT_EXISTING_MINIMUM"


def _eligible_native_cylinder(sweep_deg: float, span_mm: float) -> bool:
    """Use the published degree representation only at the lower boundary.

    This resolves eligibility/output disagreement caused by native binary
    parameter arithmetic. It is not a manufacturing tolerance. The raw 200
    degree upper cutoff and raw 8 mm span cutoff remain unchanged.
    """
    return bool(math.isfinite(sweep_deg) and math.isfinite(span_mm)
        and round(sweep_deg, ANGULAR_PUBLICATION_DECIMALS) >= MIN_ANGULAR_SWEEP_DEG
        and sweep_deg <= MAX_ANGULAR_SWEEP_DEG and span_mm >= MIN_AXIAL_SPAN_MM)


def _normalized_direction(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    dominant = int(np.argmax(np.abs(vector)))
    if vector[dominant] < 0.0:
        vector = -vector
    return vector


def _cylinder_axis_endpoints(
    location: np.ndarray,
    native_direction: np.ndarray,
    v0: float,
    v1: float,
) -> list[np.ndarray]:
    """Evaluate STEP V parameters in the cylinder's native axis direction.

    Canonical direction is only an ordering convention. Flipping the direction
    before evaluating signed V parameters mirrors axes on cylinders whose STEP
    axis points in the negative canonical direction.
    """
    direction = np.asarray(native_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    endpoints = [
        np.asarray(location, dtype=np.float64) + direction * float(v0),
        np.asarray(location, dtype=np.float64) + direction * float(v1),
    ]
    canonical = _normalized_direction(direction)
    endpoints.sort(key=lambda point: float(np.dot(point, canonical)))
    return endpoints


def _cylinder_turn_midpoint_center(
    location: np.ndarray,
    native_direction: np.ndarray,
    x_direction: np.ndarray,
    y_direction: np.ndarray,
    radius: float,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
) -> np.ndarray:
    """Return the physical midpoint of a cylindrical skin's angular turn.

    A cylinder's axis is the center of curvature, not the bend marker on the
    sheet.  STEP U parameterizes the angular turn and V parameterizes the
    forming axis, so this point is source-only and invariant to face winding.
    """
    direction = np.asarray(native_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x_axis = np.asarray(x_direction, dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.asarray(y_direction, dtype=np.float64)
    y_axis /= np.linalg.norm(y_axis)
    u_mid = 0.5 * (float(u0) + float(u1))
    v_mid = 0.5 * (float(v0) + float(v1))
    radial = math.cos(u_mid) * x_axis + math.sin(u_mid) * y_axis
    return np.asarray(location, dtype=np.float64) + direction * v_mid + float(radius) * radial


def _line_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    a = np.asarray([left["line_start"], left["line_end"]], dtype=np.float64)
    b = np.asarray([right["line_start"], right["line_end"]], dtype=np.float64)
    da = _normalized_direction(a[1] - a[0])
    db = _normalized_direction(b[1] - b[0])
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(da, db)), -1.0, 1.0))))
    center_delta = np.mean(b, axis=0) - np.mean(a, axis=0)
    offset = float(np.linalg.norm(center_delta - np.dot(center_delta, da) * da))
    origin = np.mean(a, axis=0)
    ai = sorted(float(np.dot(point - origin, da)) for point in a)
    bi = sorted(float(np.dot(point - origin, da)) for point in b)
    overlap = max(0.0, min(ai[1], bi[1]) - max(ai[0], bi[0]))
    shorter = min(ai[1] - ai[0], bi[1] - bi[0])
    return {
        "axis_angle_deg": angle,
        "axis_offset_mm": offset,
        "common_span_fraction": overlap / shorter if shorter > 1e-9 else 0.0,
    }


def _coaxial(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = _line_metrics(left, right)
    return bool(
        metrics["axis_angle_deg"] <= MAX_AXIS_ANGLE_DEG
        and metrics["axis_offset_mm"] <= MAX_COAXIAL_OFFSET_MM
        and metrics["common_span_fraction"] >= MIN_COMMON_SPAN_FRACTION
    )


def _group_cylinder_rows(faces: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Unchanged coaxial grouping and canonical marker construction, pure rows."""
    groups: list[list[dict[str, Any]]] = []
    for face in faces:
        group = next((rows for rows in groups if _coaxial(face, rows[0])), None)
        if group is None:
            groups.append([face])
        else:
            group.append(face)

    axes = []
    rejected_groups = []
    for group_index, group in enumerate(groups, start=1):
        radii = sorted({round(float(row["radius_mm"]), 6) for row in group})
        radius_separation = max(radii) - min(radii) if len(radii) >= 2 else 0.0
        accepted = bool(
            len(radii) >= 2
            and MIN_RADIUS_SEPARATION_MM <= radius_separation <= MAX_RADIUS_SEPARATION_MM
        )
        record = {
            "group_id": f"STEP_CYL_{group_index:03d}",
            "face_indices": sorted(int(row["face_index"]) for row in group),
            "radii_mm": radii,
            "radius_separation_mm": radius_separation,
            "surface_count": len(group),
            "angular_sweeps_unrounded_deg": sorted(float(row["angular_sweep_deg"]) for row in group),
            "native_sweep_rows": [{"face_index": int(row["face_index"]),
                "unrounded_deg": float(row["angular_sweep_deg"]),
                "published_deg": round(float(row["angular_sweep_deg"]), ANGULAR_PUBLICATION_DECIMALS)}
                for row in sorted(group, key=lambda row: int(row["face_index"]))],
        }
        if not accepted:
            record["rejection_reason"] = "missing_plausible_inner_outer_cylinder_pair"
            rejected_groups.append(record)
            continue
        representative = max(group, key=lambda row: float(row["axial_span_mm"]))
        canonical_direction = _normalized_direction(
            np.asarray(representative["line_end"], dtype=np.float64)
            - np.asarray(representative["line_start"], dtype=np.float64)
        )
        canonical_center = np.mean(
            np.asarray([row["turn_midpoint_center"] for row in group], dtype=np.float64),
            axis=0,
        )
        canonical_half_span = 0.5 * float(representative["axial_span_mm"])
        canonical_endpoints = [
            canonical_center - canonical_direction * canonical_half_span,
            canonical_center + canonical_direction * canonical_half_span,
        ]
        canonical_endpoints.sort(
            key=lambda point: float(np.dot(point, canonical_direction))
        )
        axes.append(
            {
                **record,
                "bend_id": f"STEP_BEND_{len(axes) + 1:03d}",
                "line_start": representative["line_start"],
                "line_end": representative["line_end"],
                "canonical_line_start": canonical_endpoints[0].tolist(),
                "canonical_line_end": canonical_endpoints[1].tolist(),
                "canonical_line_rule": "NEUTRAL_RADIUS_HALF_ANGULAR_TURN",
                "axial_span_mm": representative["axial_span_mm"],
                "angular_sweeps_deg": sorted(
                    round(float(row["angular_sweep_deg"]), 6) for row in group
                ),
            }
        )
    return axes, rejected_groups


def extract_step_bend_axes(path: Path) -> dict[str, Any]:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"failed to read STEP: {path}")
    if int(reader.TransferRoots()) <= 0:
        raise ValueError(f"STEP has no transferable roots: {path}")
    shape = reader.OneShape()
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face_index += 1
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            sweep_deg = abs(
                math.degrees(surface.LastUParameter() - surface.FirstUParameter())
            )
            v0 = float(surface.FirstVParameter())
            v1 = float(surface.LastVParameter())
            span = abs(v1 - v0)
            if _eligible_native_cylinder(sweep_deg, span):
                axis = cylinder.Axis()
                location = np.array(
                    [axis.Location().X(), axis.Location().Y(), axis.Location().Z()],
                    dtype=np.float64,
                )
                native_direction = np.array(
                    [
                        axis.Direction().X(),
                        axis.Direction().Y(),
                        axis.Direction().Z(),
                    ],
                    dtype=np.float64,
                )
                position = cylinder.Position()
                x_direction = np.array(
                    [
                        position.XDirection().X(),
                        position.XDirection().Y(),
                        position.XDirection().Z(),
                    ],
                    dtype=np.float64,
                )
                y_direction = np.array(
                    [
                        position.YDirection().X(),
                        position.YDirection().Y(),
                        position.YDirection().Z(),
                    ],
                    dtype=np.float64,
                )
                endpoints = _cylinder_axis_endpoints(
                    location, native_direction, v0, v1
                )
                turn_midpoint_center = _cylinder_turn_midpoint_center(
                    location,
                    native_direction,
                    x_direction,
                    y_direction,
                    float(cylinder.Radius()),
                    float(surface.FirstUParameter()),
                    float(surface.LastUParameter()),
                    v0,
                    v1,
                )
                faces.append(
                    {
                        "face_index": face_index,
                        "radius_mm": float(cylinder.Radius()),
                        "angular_sweep_deg": sweep_deg,
                        "axial_span_mm": span,
                        "line_start": endpoints[0].tolist(),
                        "line_end": endpoints[1].tolist(),
                        "turn_midpoint_center": turn_midpoint_center.tolist(),
                    }
                )
        explorer.Next()

    axes, rejected_groups = _group_cylinder_rows(faces)
    return {
        "schema_version": 2,
        "kind": "sherman_qc_step_topology_bend_axes",
        "step_path": str(path.resolve()),
        "contract": {
            "minimum_angular_sweep_deg": MIN_ANGULAR_SWEEP_DEG,
            "minimum_sweep_representation_policy": MINIMUM_SWEEP_REPRESENTATION_POLICY,
            "angular_publication_decimals": ANGULAR_PUBLICATION_DECIMALS,
            "maximum_sweep_uses_unrounded_degrees": True,
            "representation_policy_is_manufacturing_tolerance": False,
            "maximum_angular_sweep_deg": MAX_ANGULAR_SWEEP_DEG,
            "minimum_axial_span_mm": MIN_AXIAL_SPAN_MM,
            "maximum_coaxial_offset_mm": MAX_COAXIAL_OFFSET_MM,
            "maximum_axis_angle_deg": MAX_AXIS_ANGLE_DEG,
            "minimum_common_span_fraction": MIN_COMMON_SPAN_FRACTION,
            "minimum_radius_separation_mm": MIN_RADIUS_SEPARATION_MM,
            "maximum_radius_separation_mm": MAX_RADIUS_SEPARATION_MM,
        },
        "partial_cylindrical_faces": faces,
        "bend_axes": axes,
        "rejected_groups": rejected_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = extract_step_bend_axes(args.step)
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
