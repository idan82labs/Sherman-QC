#!/usr/bin/env python3
"""Convert a native SLDPRT to a deterministic face-owned PLY mesh.

The SolidWorks decoder can expose both B-rep-owned folded-body tessellation and
unresolved display-list geometry (for example a flat-pattern presentation).  A
QC reference mesh must not silently mix those lanes.  This tool therefore:

1. decodes the source with a pinned ``cadmpeg`` binary;
2. accepts only tessellations bound to a decoded body and at least one face;
3. verifies every body, face, vertex, and triangle reference; and
4. writes a byte-deterministic binary little-endian PLY plus a provenance report.

It uses no scan, label, expected count, filename heuristic, or evaluator data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence


SUPPORTED_CADMPEG_VERSION = "cadmpeg 0.5.5"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(value: Mapping[str, Any], *, label: str) -> tuple[float, float, float]:
    try:
        result = (float(value["x"]), float(value["y"]), float(value["z"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} coordinate") from exc
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"non-finite {label} coordinate")
    return result


def extract_face_owned_mesh(cadir: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the source-ordered, face-owned tessellation lane from CADIR."""
    units = cadir.get("units", {})
    if not isinstance(units, Mapping) or units.get("length") != "millimeter":
        raise ValueError("CADIR length unit must be millimeter")
    model = cadir.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("CADIR model is missing")

    bodies = model.get("bodies", [])
    faces = model.get("faces", [])
    tessellations = model.get("tessellations", [])
    if not all(isinstance(items, list) for items in (bodies, faces, tessellations)):
        raise ValueError("CADIR bodies, faces, and tessellations must be arrays")

    body_ids = {str(item.get("id")) for item in bodies if isinstance(item, Mapping)}
    face_ids = {str(item.get("id")) for item in faces if isinstance(item, Mapping)}
    if not body_ids or not face_ids or not tessellations:
        raise ValueError("CADIR contains no usable B-rep/tessellation inventory")

    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    selected_ids: list[str] = []
    selected_body_ids: set[str] = set()
    omitted_ids: list[str] = []
    omitted_vertices: list[tuple[float, float, float]] = []

    for position, item in enumerate(tessellations):
        if not isinstance(item, Mapping):
            raise ValueError(f"tessellation {position} is not an object")
        tessellation_id = str(item.get("id", f"position:{position}"))
        body_id = item.get("body")
        owned_faces = item.get("faces", [])
        local_vertices = item.get("vertices", [])
        local_triangles = item.get("triangles", [])
        if not isinstance(owned_faces, list) or not isinstance(local_vertices, list) or not isinstance(local_triangles, list):
            raise ValueError(f"invalid arrays in tessellation {tessellation_id}")

        is_face_owned = bool(body_id) and bool(owned_faces)
        if not is_face_owned:
            omitted_ids.append(tessellation_id)
            omitted_vertices.extend(
                _xyz(value, label=f"omitted tessellation {tessellation_id}")
                for value in local_vertices
                if isinstance(value, Mapping)
            )
            continue

        body_token = str(body_id)
        if body_token not in body_ids:
            raise ValueError(f"unknown body {body_token} in tessellation {tessellation_id}")
        unknown_faces = sorted(str(value) for value in owned_faces if str(value) not in face_ids)
        if unknown_faces:
            raise ValueError(f"unknown faces in tessellation {tessellation_id}: {unknown_faces}")
        if not local_vertices or not local_triangles:
            raise ValueError(f"empty face-owned tessellation {tessellation_id}")

        local_xyz = [
            _xyz(value, label=f"tessellation {tessellation_id}")
            for value in local_vertices
            if isinstance(value, Mapping)
        ]
        if len(local_xyz) != len(local_vertices):
            raise ValueError(f"non-object vertex in tessellation {tessellation_id}")
        offset = len(vertices)
        vertices.extend(local_xyz)
        for triangle in local_triangles:
            if not isinstance(triangle, list) or len(triangle) != 3:
                raise ValueError(f"non-triangle face in tessellation {tessellation_id}")
            try:
                a, b, c = (int(value) for value in triangle)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid triangle index in tessellation {tessellation_id}") from exc
            if min(a, b, c) < 0 or max(a, b, c) >= len(local_xyz):
                raise ValueError(f"out-of-range triangle index in tessellation {tessellation_id}")
            if len({a, b, c}) != 3:
                raise ValueError(f"degenerate triangle indices in tessellation {tessellation_id}")
            triangles.append((offset + a, offset + b, offset + c))
        selected_ids.append(tessellation_id)
        selected_body_ids.add(body_token)

    if not vertices or not triangles:
        raise ValueError("no face-owned folded-body triangles were decoded")

    mins = tuple(min(point[axis] for point in vertices) for axis in range(3))
    maxs = tuple(max(point[axis] for point in vertices) for axis in range(3))
    omitted_bbox = None
    if omitted_vertices:
        omitted_mins = tuple(min(point[axis] for point in omitted_vertices) for axis in range(3))
        omitted_maxs = tuple(max(point[axis] for point in omitted_vertices) for axis in range(3))
        omitted_bbox = {
            "min_mm": list(omitted_mins),
            "max_mm": list(omitted_maxs),
            "extent_mm": [omitted_maxs[i] - omitted_mins[i] for i in range(3)],
        }

    return {
        "vertices": vertices,
        "triangles": triangles,
        "selection": {
            "rule": "BODY_BOUND_AND_AT_LEAST_ONE_RESOLVED_FACE_V1",
            "selected_tessellation_count": len(selected_ids),
            "selected_tessellation_ids": selected_ids,
            "selected_body_ids": sorted(selected_body_ids),
            "omitted_unresolved_tessellation_count": len(omitted_ids),
            "omitted_unresolved_tessellation_ids": omitted_ids,
            "omitted_unresolved_bbox": omitted_bbox,
        },
        "mesh": {
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "bbox": {
                "min_mm": list(mins),
                "max_mm": list(maxs),
                "extent_mm": [maxs[i] - mins[i] for i in range(3)],
            },
        },
    }


def write_binary_ply(
    path: Path,
    vertices: Sequence[tuple[float, float, float]],
    triangles: Sequence[tuple[int, int, int]],
) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment source face-owned SLDPRT tessellation; units millimeter\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for point in vertices:
            handle.write(struct.pack("<fff", *point))
        for triangle in triangles:
            handle.write(struct.pack("<Biii", 3, *triangle))


def convert(
    source: Path,
    cadmpeg: Path,
    output_ply: Path,
    output_report: Path,
    *,
    expected_source_sha256: Optional[str] = None,
    expected_cadmpeg_sha256: Optional[str] = None,
) -> dict[str, Any]:
    source = source.resolve()
    cadmpeg = cadmpeg.resolve()
    if not source.is_file() or source.suffix.lower() != ".sldprt":
        raise ValueError("source must be an existing .SLDPRT file")
    if not cadmpeg.is_file():
        raise ValueError("cadmpeg binary is missing")
    source_sha256 = sha256_path(source)
    cadmpeg_sha256 = sha256_path(cadmpeg)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError("source SHA-256 does not match the frozen manifest")
    if expected_cadmpeg_sha256 and cadmpeg_sha256 != expected_cadmpeg_sha256:
        raise ValueError("cadmpeg SHA-256 does not match the frozen dependency")

    version_result = subprocess.run([str(cadmpeg), "--version"], capture_output=True, text=True, check=True)
    version = version_result.stdout.strip()
    if version != SUPPORTED_CADMPEG_VERSION:
        raise ValueError(f"unsupported cadmpeg version {version!r}")

    with tempfile.TemporaryDirectory(prefix="sldprt-face-owned-") as temp_dir:
        temp_root = Path(temp_dir)
        cadir_path = temp_root / "model.cadir.json"
        decoder_report_path = temp_root / "decoder.report.json"
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
                str(decoder_report_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cadmpeg failed with {result.returncode}: {result.stderr[-1200:]}")
        cadir = json.loads(cadir_path.read_text(encoding="utf-8"))
        decoder_report = json.loads(decoder_report_path.read_text(encoding="utf-8"))
        decode = decoder_report.get("decode_report", {})
        if not isinstance(decode, Mapping) or not decode.get("geometry_transferred") or decode.get("container_only"):
            raise ValueError("cadmpeg did not transfer source geometry")
        extraction = extract_face_owned_mesh(cadir)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(output_ply, extraction["vertices"], extraction["triangles"])
    report = {
        "schema": "SHERMAN_QC_SLDPRT_FACE_OWNED_TESSELLATION_V1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_ONLY_ENGINE_MESH_READY",
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "bytes": source.stat().st_size,
        },
        "dependency": {
            "path": str(cadmpeg),
            "version": version,
            "sha256": cadmpeg_sha256,
        },
        "decoder": {
            "geometry_transferred": True,
            "container_only": False,
            "losses": decode.get("losses", []),
            "notes": decode.get("notes", []),
        },
        "selection": extraction["selection"],
        "mesh": {
            **extraction["mesh"],
            "path": str(output_ply.resolve()),
            "sha256": sha256_path(output_ply),
            "bytes": output_ply.stat().st_size,
            "format": "binary_little_endian_ply",
            "units": "millimeter",
        },
        "is_metric_eligible": False,
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
    }
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cadmpeg", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-cadmpeg-sha256")
    args = parser.parse_args(argv)
    report = convert(
        args.source,
        args.cadmpeg,
        args.output_ply,
        args.output_report,
        expected_source_sha256=args.expected_source_sha256,
        expected_cadmpeg_sha256=args.expected_cadmpeg_sha256,
    )
    print(json.dumps({"status": report["status"], "selection": report["selection"], "mesh": report["mesh"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
