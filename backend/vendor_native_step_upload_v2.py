"""Bounded, source-bound native STEP candidate and folded-preview producer.

This is a new upload preprocessing transport, not a revision of the qualified
V4 unfolding worker. It shares that worker's admission slot and persistent
process-group journal. The accepted output is a CAD *nominal* bend catalogue and
a visual folded mesh; neither is scan observation nor a manufacturing blank.
No CAD/native parsing happens at module import.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

from backend import vendor_unfold_worker_v4 as shared
from backend.vendor_unfold_contract_v2 import (
    MAX_CANDIDATES, MAX_COORDINATE, PreviewContractError, canonical_candidates,
    strict_json,
)


ROOT = shared.ROOT
PROFILE = "VENDOR_NATIVE_STEP_UPLOAD_V2"
NONCE_ENV = "SHERMAN_NATIVE_STEP_UPLOAD_SUPERVISED_V2"
MAX_SECONDS = shared.MAX_SECONDS
MAX_RSS_KIB = shared.MAX_RSS_KIB
MAX_STEP_BYTES = shared.MAX_CAD_BYTES
MAX_REQUEST_BYTES = shared.MAX_REQUEST_BYTES
MAX_RESULT_BYTES = shared.MAX_RESULT_BYTES
MAX_LOG_BYTES = shared.MAX_LOG_BYTES
MAX_PIN_BYTES = shared.MAX_PIN_BYTES
LIMITS = dict(shared.LIMITS)
_hash, _read, _json = shared._hash, shared._read, shared._json


@dataclass(frozen=True)
class NativeBendSpec:
    """The exact subset accepted by PartCatalog._bend_records."""

    bend_id: str
    target_angle: float
    target_radius: float
    tolerance_angle: float
    tolerance_radius: None
    bend_line_start: tuple[float, float, float]
    bend_line_end: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {"bend_id": self.bend_id, "target_angle": self.target_angle,
            "target_radius": self.target_radius, "tolerance_angle": self.tolerance_angle,
            "tolerance_radius": self.tolerance_radius,
            "bend_line_start": list(self.bend_line_start),
            "bend_line_end": list(self.bend_line_end)}


def _deadline(started: float) -> None:
    if time.monotonic() - started >= MAX_SECONDS:
        raise PreviewContractError("NATIVE_STEP_TIME_BUDGET_EXCEEDED")


def _source_pins() -> dict[str, str]:
    # V3 checks its own declared engine/helper hashes and the OCP/NumPy runtime.
    # This candidate additionally binds the new transport's exact bytes.
    pins = shared._source_pins()
    path = Path(__file__).resolve()
    pins[str(path)] = _hash(path.read_bytes())
    return pins


def _cad_source(status: str, reason: str | None, source_hash: str | None) -> dict[str, Any]:
    return {"status": status, "reason": reason, "source_sha256": source_hash,
        "format": "STEP", "candidate_rule": "PAIRED_NATIVE_PARTIAL_CYLINDERS",
        "line_rule": "NATIVE_MID_RADIUS_HALF_ANGULAR_TURN",
        "line_limitation": "GEOMETRIC_MID_RADIUS_NOT_PHYSICAL_NEUTRAL_AXIS_OR_K_FACTOR_BLANK",
        "angle_rule": "CYLINDER_SWEEP_ROTATION_FROM_FLAT",
        "angle_limitation": "BEND_DIRECTION_AND_INCLUDED_ANGLE_UNVERIFIED",
        "radius_rule": "INNER_NATIVE_CYLINDER_RADIUS",
        "units": "assumed_mm", "units_verified": False,
        "manufacturing_ready": False, "visual_only": True,
        "winding_oriented": False, "folded_normals_authoritative": False,
        "flat_preview_status": "NOT_RUN_SEPARATE_BOUNDED_V5_GATE",
        "transport_profile": PROFILE, "hostile_upload_sandbox_verified": False}


def _unavailable(reason: str, source_hash: str | None = None):
    return [], None, _cad_source("UNAVAILABLE", reason, source_hash)


def _axis_rows(axes: list[dict[str, Any]], tolerance_angle: float) -> list[dict[str, Any]]:
    """Convert native paired-cylinder evidence to exact V4 matching lines.

    The native extractor may report split/multiply owned cylindrical groups.
    This first version deliberately refuses those rather than fabricating an
    angle or silently using just one partial skin. The downstream flat engine
    can be extended under a separately versioned patch-graph contract.
    """
    if type(tolerance_angle) not in (int, float) or not math.isfinite(tolerance_angle) or tolerance_angle <= 0:
        raise PreviewContractError("INVALID_ANGLE_TOLERANCE")
    if type(axes) is not list or not 0 < len(axes) <= MAX_CANDIDATES:
        raise PreviewContractError("NATIVE_STEP_BEND_COUNT_UNSUPPORTED")
    rows = []
    for index, axis in enumerate(axes, start=1):
        if type(axis) is not dict:
            raise PreviewContractError("INVALID_NATIVE_BEND_AXIS")
        radii = axis.get("radii_mm")
        sweeps = axis.get("angular_sweeps_deg")
        if (type(radii) is not list or len(radii) != 2
                or type(sweeps) is not list or len(sweeps) != 2
                or axis.get("surface_count") != 2 or type(axis.get("surface_count")) is not int):
            raise PreviewContractError("SPLIT_OR_AMBIGUOUS_BEND_SKINS")
        if not all(type(value) in (int, float) and math.isfinite(value) for value in radii + sweeps):
            raise PreviewContractError("NONFINITE_NATIVE_BEND_METADATA")
        # V4 accepts a maximum 175° sweep and paired surfaces matching within
        # 1e-5 rad. The extractor reports six decimal places in degrees, so
        # reserve margin rather than accepting a rounded borderline value.
        if not (0 < radii[0] < radii[1] <= MAX_COORDINATE and
                15 <= sweeps[0] <= 175 and 15 <= sweeps[1] <= 175 and
                abs(sweeps[0] - sweeps[1]) <= 0.0005):
            raise PreviewContractError("INCONSISTENT_NATIVE_BEND_METADATA")
        rows.append({"feature_id": f"B{index}",
            "line_start": axis.get("canonical_line_start"),
            "line_end": axis.get("canonical_line_end"),
            "target_angle": float(sum(sweeps) / 2),
            "target_radius": float(radii[0])})
    canonical = canonical_candidates(rows)
    # Ensure the stronger numeric fields survive only on exactly bound lines.
    lookup = {row["feature_id"]: row for row in rows}
    return [{**line, "target_angle": lookup[line["feature_id"]]["target_angle"],
             "target_radius": lookup[line["feature_id"]]["target_radius"]}
            for line in canonical]


def _folded_ply(faces: list[Any]) -> tuple[bytes, int, int]:
    """Serialize the complete native face triangulation with no remeshing.

    Per-face duplicated boundary vertices retain the source coordinate frame.
    The unindexed surfaces are a display mesh, not a watertightness assertion.
    """
    import numpy as np

    vertices: list[Any] = []
    triangles: list[Any] = []
    vertex_offset = 0
    for face in faces:
        points = np.asarray(face.vertices, dtype=np.float64)
        indices = np.asarray(face.triangles)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
                or indices.ndim != 2 or indices.shape[1] != 3 or len(indices) < 1
                or not np.isfinite(points).all() or np.any(np.abs(points) > MAX_COORDINATE)
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(points))):
            raise PreviewContractError("INVALID_NATIVE_FOLDED_MESH")
        vertices.append(points.astype("<f4"))
        triangles.append(indices.astype("<i4") + vertex_offset)
        vertex_offset += len(points)
    if not vertices or vertex_offset > 600_000 or sum(len(t) for t in triangles) > 200_000:
        raise PreviewContractError("NATIVE_FOLDED_MESH_BUDGET_EXCEEDED")
    all_vertices = np.concatenate(vertices)
    all_faces = np.concatenate(triangles)
    if not np.isfinite(all_vertices).all():
        raise PreviewContractError("FLOAT32_FOLDED_MESH_OVERFLOW")
    # The viewer consumes Float32. Refuse triangles that collapse *after* that
    # cast even if OCC's Double tessellation was nominally nondegenerate.
    display_triangles = all_vertices[all_faces].astype(np.float64)
    edges_a = display_triangles[:, 1] - display_triangles[:, 0]
    edges_b = display_triangles[:, 2] - display_triangles[:, 0]
    areas = np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 1e-12):
        raise PreviewContractError("FLOAT32_DEGENERATE_FOLDED_TRIANGLE")
    header = ("ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(all_vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(all_faces)}\n"
        "property list uchar int vertex_indices\nend_header\n").encode("ascii")
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    binary_faces = np.empty(len(all_faces), dtype=face_dtype)
    binary_faces["count"] = 3
    binary_faces["indices"] = all_faces
    payload = header + all_vertices.tobytes() + binary_faces.tobytes()
    if len(payload) > MAX_RESULT_BYTES:
        raise PreviewContractError("NATIVE_FOLDED_MESH_BUDGET_EXCEEDED")
    return payload, len(all_vertices), len(all_faces)


def _assert_native_line_mesh_frame(rows: list[dict[str, Any]], faces: list[Any]) -> None:
    """Ensure source-frame native marks overlap the same native tessellation.

    The neutral-radius line sits inside the inner/outer cylinder envelope; it
    need not be a mesh edge. This is a frame-integrity check, not a claim about
    bend line placement or the flat chart's dimensional accuracy.
    """
    import numpy as np

    all_points = np.concatenate([np.asarray(face.vertices, dtype=np.float64) for face in faces])
    lower, upper = all_points.min(axis=0) - 1e-3, all_points.max(axis=0) + 1e-3
    for row in rows:
        line = np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)
        if not np.isfinite(line).all() or np.any(line < lower) or np.any(line > upper):
            raise PreviewContractError("NATIVE_BEND_LINE_FOLDED_FRAME_MISMATCH")


def _validate_result(result: Any, expected_hash: str, tolerance_angle: float) -> list[NativeBendSpec]:
    if (type(result) is not dict or set(result) != {"profile", "source_sha256", "status",
            "reason", "bends", "folded_mesh_sha256", "folded_mesh_bytes",
            "folded_vertex_count", "folded_triangle_count", "source_pins_sha256"}
            or result["profile"] != PROFILE or result["source_sha256"] != expected_hash
            or result["status"] not in {"AVAILABLE", "UNAVAILABLE"}
            or type(result["status"]) is not str):
        raise PreviewContractError("INVALID_NATIVE_STEP_RESULT")
    if result["status"] == "UNAVAILABLE":
        if (type(result["reason"]) is not str or not 0 < len(result["reason"]) <= 200
                or result["bends"] != [] or result["folded_mesh_sha256"] is not None
                or result["folded_mesh_bytes"] is not None):
            raise PreviewContractError("INVALID_NATIVE_STEP_ABSTENTION")
        return []
    if (result["reason"] is not None or type(result["bends"]) is not list
            or not 0 < len(result["bends"]) <= MAX_CANDIDATES
            or type(result["folded_mesh_sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", result["folded_mesh_sha256"])
            or type(result["folded_mesh_bytes"]) is not int
            or not 0 < result["folded_mesh_bytes"] <= MAX_RESULT_BYTES
            or type(result["folded_vertex_count"]) is not int
            or not 3 <= result["folded_vertex_count"] <= 600_000
            or type(result["folded_triangle_count"]) is not int
            or not 1 <= result["folded_triangle_count"] <= 200_000):
        raise PreviewContractError("INVALID_NATIVE_STEP_RESULT")
    candidates = canonical_candidates(result["bends"])
    if len(candidates) != len(result["bends"]):
        raise PreviewContractError("INVALID_NATIVE_STEP_RESULT")
    bends = []
    by_id = {row["feature_id"]: row for row in result["bends"]}
    for candidate in candidates:
        row = by_id[candidate["feature_id"]]
        if (set(row) != {"feature_id", "line_start", "line_end", "target_angle", "target_radius"}
                or type(row["target_angle"]) not in (int, float)
                or not math.isfinite(row["target_angle"]) or not 15 <= row["target_angle"] <= 200
                or type(row["target_radius"]) not in (int, float)
                or not math.isfinite(row["target_radius"]) or row["target_radius"] <= 0):
            raise PreviewContractError("INVALID_NATIVE_STEP_BEND_METADATA")
        bends.append(NativeBendSpec(candidate["feature_id"], float(row["target_angle"]),
            float(row["target_radius"]), float(tolerance_angle), None,
            tuple(candidate["line_start"]), tuple(candidate["line_end"])))
    return bends


def _run(folder: Path, request: dict[str, Any], pins: dict[str, str], started: float,
         on_spawn, on_quiescent):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(folder), "TMPDIR": str(folder),
        "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(folder / "pycache"), "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        NONCE_ENV: request["nonce"]}
    process = None
    peak = 0
    try:
        with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            _deadline(started)
            if _source_pins() != pins:
                raise PreviewContractError("NATIVE_STEP_SOURCE_CHANGED_BEFORE_SPAWN")
            # Monitor availability is verified before creating native work.
            if not shared._group_rows(os.getpgrp()):
                raise PreviewContractError("NATIVE_STEP_RSS_MONITOR_UNAVAILABLE")
            _deadline(started)
            process = subprocess.Popen([sys.executable, "-X", "faulthandler", "-m",
                "backend.vendor_native_step_upload_v2", "--worker", str(folder / "request.json")],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True)
            on_spawn(process.pid)
            while process.poll() is None:
                _deadline(started)
                rows = shared._group_rows(process.pid)
                if not rows and process.poll() is None:
                    raise PreviewContractError("NATIVE_STEP_RSS_MONITOR_UNAVAILABLE")
                peak = max(peak, sum(rss for _, rss in rows))
                if peak > MAX_RSS_KIB:
                    raise PreviewContractError("NATIVE_STEP_RSS_BUDGET_EXCEEDED")
                if out.tell() + err.tell() > MAX_LOG_BYTES:
                    raise PreviewContractError("NATIVE_STEP_LOG_BUDGET_EXCEEDED")
                time.sleep(.1)
            if any(pid != process.pid for pid, _ in shared._group_rows(process.pid)):
                raise PreviewContractError("NATIVE_STEP_LIVE_DESCENDANTS_AFTER_EXIT")
            if out.tell() + err.tell() > MAX_LOG_BYTES or process.returncode != 0:
                raise PreviewContractError("NATIVE_STEP_WORKER_FAILED")
    finally:
        if process is not None:
            shared._stop_group(process)
        on_quiescent()
    _deadline(started)
    result = strict_json(_read(folder / "result.json", MAX_RESULT_BYTES))
    bends = _validate_result(result, request["expected_hash"], request["tolerance_angle"])
    if result["source_pins_sha256"] != _hash(_json(pins)) or _source_pins() != pins:
        raise PreviewContractError("NATIVE_STEP_SOURCE_CHANGED_DURING_JOB")
    if result["status"] == "UNAVAILABLE":
        return bends, None, result, peak
    mesh = _read(folder / "folded.ply", MAX_RESULT_BYTES)
    if len(mesh) != result["folded_mesh_bytes"] or _hash(mesh) != result["folded_mesh_sha256"]:
        raise PreviewContractError("NATIVE_STEP_FOLDED_MESH_BINDING_MISMATCH")
    _deadline(started)
    return bends, mesh, result, peak


def preprocess_vendor_step(cad_path: str | Path, expected_sha256: str,
                           tolerance_angle: float) -> tuple[list[NativeBendSpec], bytes | None, dict[str, Any]]:
    """Return native nominal bends + source-framed folded PLY, or abstain.

    The return is deliberately not the V4 flat preview. Vendor setup invokes
    its separately controlled V4 path with these *exact* candidate lines.
    """
    started = time.monotonic()
    source_hash = None
    try:
        if (type(expected_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
            raise PreviewContractError("INVALID_EXPECTED_CAD_HASH")
        if type(tolerance_angle) not in (int, float) or not math.isfinite(tolerance_angle) or tolerance_angle <= 0:
            raise PreviewContractError("INVALID_ANGLE_TOLERANCE")
        if Path(cad_path).suffix.lower() not in {".stp", ".step"}:
            raise PreviewContractError("STEP_SOURCE_REQUIRED")
        _deadline(started)
        pins = _source_pins()
        raw = _read(cad_path, MAX_STEP_BYTES)
        source_hash = _hash(raw)
        if source_hash != expected_sha256:
            raise PreviewContractError("CAD_HASH_MISMATCH")
        _deadline(started)
        if not shared._slot.acquire(blocking=False):
            return _unavailable("NATIVE_STEP_WORKER_BUSY", source_hash)
        try:
            fd = os.open(shared.ADMISSION_LOCK_PATH,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "r+b") as lock:
                info = os.fstat(lock.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise PreviewContractError("NATIVE_STEP_LOCK_UNAVAILABLE")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return _unavailable("NATIVE_STEP_WORKER_BUSY", source_hash)
                shared._check_admission_record(lock)
                _deadline(started)
                with tempfile.TemporaryDirectory(prefix="sherman-native-step-v2-") as directory:
                    folder = Path(directory)
                    (folder / "source.step").write_bytes(raw)
                    request = {"profile": PROFILE, "cad_path": str(folder / "source.step"),
                        "expected_hash": expected_sha256, "tolerance_angle": float(tolerance_angle),
                        "nonce": secrets.token_hex(24)}
                    encoded, pin_bytes = _json(request), _json(pins)
                    if len(encoded) > MAX_REQUEST_BYTES or len(pin_bytes) > MAX_PIN_BYTES:
                        raise PreviewContractError("NATIVE_STEP_REQUEST_BUDGET_EXCEEDED")
                    (folder / "request.json").write_bytes(encoded)
                    (folder / "pins.json").write_bytes(pin_bytes)
                    _deadline(started)
                    shared._record(lock, {"state": "ADMITTING", "pgid": None})
                    bends, mesh, result, peak = _run(folder, request, pins, started,
                        lambda pgid: shared._record(lock, {"state": "RUNNING", "pgid": pgid}),
                        lambda: shared._record(lock, {"state": "QUIESCENT", "pgid": None}))
        finally:
            shared._slot.release()
        _deadline(started)
        if result["status"] == "UNAVAILABLE":
            return _unavailable(result["reason"], source_hash)
        meta = _cad_source("AVAILABLE", None, source_hash)
        meta.update({"bend_count": len(bends), "folded_vertex_count": result["folded_vertex_count"],
            "folded_triangle_count": result["folded_triangle_count"],
            "folded_mesh_sha256": result["folded_mesh_sha256"],
            "source_bundle_sha256": _hash(_json(pins)),
            "peak_process_group_rss_kib": peak, "elapsed_seconds": time.monotonic() - started})
        return bends, mesh, meta
    except (PreviewContractError, OSError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError, ImportError) as exc:
        reason = (str(exc) if isinstance(exc, PreviewContractError)
                  else f"NATIVE_STEP_WORKER_UNAVAILABLE_{type(exc).__name__.upper()}")
        return _unavailable(reason, source_hash if source_hash == expected_sha256 else None)


def _main(path: str) -> None:
    folder = Path(path).parent
    request = strict_json(_read(path, MAX_REQUEST_BYTES))
    if (type(request) is not dict or set(request) != {"profile", "cad_path", "expected_hash",
            "tolerance_angle", "nonce"} or request["profile"] != PROFILE
            or type(request["nonce"]) is not str or not request["nonce"]
            or request["nonce"] != os.environ.get(NONCE_ENV)
            or request["cad_path"] != str(folder / "source.step")):
        raise PreviewContractError("SUPERVISED_ENTRYPOINT_REQUIRED")
    pins = strict_json(_read(folder / "pins.json", MAX_PIN_BYTES))
    if _source_pins() != pins:
        raise PreviewContractError("NATIVE_STEP_SOURCE_BUNDLE_MISMATCH")
    raw = _read(folder / "source.step", MAX_STEP_BYTES)
    if _hash(raw) != request["expected_hash"]:
        raise PreviewContractError("CAD_HASH_MISMATCH")
    result = {"profile": PROFILE, "source_sha256": request["expected_hash"],
        "status": "UNAVAILABLE", "reason": "NATIVE_STEP_DEVELOPMENT_UNAVAILABLE", "bends": [],
        "folded_mesh_sha256": None, "folded_mesh_bytes": None,
        "folded_vertex_count": None, "folded_triangle_count": None,
        "source_pins_sha256": _hash(_json(pins))}
    try:
        from backend.vendor_unfold_preview import UnfoldUnavailable
        from backend.vendor_unfold_native_v5 import read_native_v5 as _read_native
        faces, axes = _read_native(folder / "source.step")
        rows = _axis_rows(axes, request["tolerance_angle"])
        _assert_native_line_mesh_frame(rows, faces)
        folded, vertex_count, triangle_count = _folded_ply(faces)
        result.update({"status": "AVAILABLE", "reason": None, "bends": rows,
            "folded_mesh_sha256": _hash(folded), "folded_mesh_bytes": len(folded),
            "folded_vertex_count": vertex_count, "folded_triangle_count": triangle_count})
        (folder / "folded.ply").write_bytes(folded)
    except (UnfoldUnavailable, PreviewContractError) as exc:
        result["reason"] = str(exc)
    _validate_result(result, request["expected_hash"], request["tolerance_angle"])
    encoded = _json(result)
    if len(encoded) > MAX_RESULT_BYTES:
        raise PreviewContractError("NATIVE_STEP_RESULT_BUDGET_EXCEEDED")
    (folder / "result.json").write_bytes(encoded)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Private supervised entrypoint only")
    _main(sys.argv[2])
