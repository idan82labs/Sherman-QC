"""Bounded, explicit CAD/PLY/datum snapshots for the offline candidate.

No discovery, corpus defaults, truth, approval records or application imports.
Calling the file loader is NOT authorization to access physical inputs. The
caller must first pass the independent custody/input/reference/baseline gates.
This restricted point-cloud PLY reader preserves raw row order (datum IDs depend
on it); unsupported encodings/elements fail rather than being silently rewritten.
"""

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

import numpy as np


MAX_POINTS = 200_000
MAX_PLY_BYTES = 64 * 1024**2
MAX_CAD_BYTES = 8 * 1024**2
MAX_DATUM_BYTES = 2 * 1024**2
MAX_HEADER_BYTES = 64 * 1024
MAX_PROPERTIES = 32
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
DATUM_KEYS = {"schema_version", "scan_sha256", "cad_sha256", "reference_frame", "unit",
              "selection_method", "cad_datum_points", "cad_datum_normals", "scan_datum_row_indices"}
INTEGER_TOKEN = re.compile(rb"[+-]?[0-9]+")
FLOAT_TOKEN = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class InputUnavailable(ValueError):
    """Fixed-code admission failure, not a part defect."""


@dataclass(frozen=True)
class RegisteredScanInput:
    scan_points: np.ndarray
    cad_datum_points: np.ndarray
    cad_datum_normals: np.ndarray
    scan_datum_row_indices: np.ndarray
    scan_sha256: str
    cad_sha256: str
    datum_sha256: str

    def inference_arguments(self):
        return {"scan_points": self.scan_points, "cad_datum_points": self.cad_datum_points,
                "cad_datum_normals": self.cad_datum_normals,
                "scan_datum_row_indices": self.scan_datum_row_indices,
                "unit_status": "DECLARED_MM"}

    def binding_summary(self):
        return {"scan_sha256": self.scan_sha256, "cad_sha256": self.cad_sha256,
                "datum_sha256": self.datum_sha256, "scan_rows": len(self.scan_points),
                "datum_rows": len(self.scan_datum_row_indices),
                "reference_frame": "CAD_REFERENCE", "unit": "mm",
                "binding_status": "VERIFIED_SNAPSHOT_BYTES_ONLY",
                "datum_rigidity_and_frame_status": "DECLARED_NOT_PHYSICALLY_VERIFIED"}


def _bounded_bytes(value, limit, code):
    if type(value) is not bytes or not 0 < len(value) <= limit:
        raise InputUnavailable(code)
    return value


def parse_point_cloud_ply(blob):
    """Read one vertex-only scalar-property PLY, without resampling or reordering."""
    _bounded_bytes(blob, MAX_PLY_BYTES, "PLY_BYTE_BUDGET")
    lines, offset = [], 0
    while offset < min(len(blob), MAX_HEADER_BYTES):
        end = blob.find(b"\n", offset, MAX_HEADER_BYTES)
        if end < 0:
            raise InputUnavailable("PLY_HEADER_BUDGET_OR_TERMINATOR")
        try:
            line = blob[offset:end].rstrip(b"\r").decode("ascii")
        except UnicodeDecodeError as exc:
            raise InputUnavailable("PLY_NONASCII_HEADER") from exc
        offset = end + 1
        lines.append(line)
        if line == "end_header":
            break
    if not lines or lines[-1] != "end_header" or lines[0] != "ply":
        raise InputUnavailable("INVALID_PLY_HEADER")
    encoding, count, properties = None, None, []
    for line in lines[1:-1]:
        words = line.split()
        if not words:
            raise InputUnavailable("INVALID_PLY_HEADER")
        if words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "format":
            if (encoding is not None or count is not None or len(words) != 3 or words[2] != "1.0"
                    or words[1] not in ("ascii", "binary_little_endian", "binary_big_endian")):
                raise InputUnavailable("UNSUPPORTED_PLY_FORMAT")
            encoding = words[1]
        elif words[0] == "element":
            if (encoding is None or count is not None or len(words) != 3 or words[1] != "vertex"
                    or not words[2].isascii() or not words[2].isdecimal() or len(words[2]) > 6):
                raise InputUnavailable("UNSUPPORTED_PLY_ELEMENTS")
            count = int(words[2])
            if not 1 <= count <= MAX_POINTS:
                raise InputUnavailable("PLY_POINT_BUDGET")
        elif words[0] == "property":
            if (count is None or len(words) != 3 or words[1] not in PLY_TYPES
                    or not words[2].isidentifier() or len(words[2]) > 64
                    or any(name == words[2] for name, _ in properties)
                    or len(properties) >= MAX_PROPERTIES):
                raise InputUnavailable("UNSUPPORTED_PLY_PROPERTIES")
            properties.append((words[2], PLY_TYPES[words[1]]))
        else:
            raise InputUnavailable("UNSUPPORTED_PLY_HEADER_FIELD")
    names = [name for name, _ in properties]
    if encoding is None or count is None or not all(axis in names for axis in ("x", "y", "z")):
        raise InputUnavailable("PLY_XYZ_REQUIRED")
    body = memoryview(blob)[offset:]
    if encoding == "ascii":
        # Exact row count and scalar count: no np.fromstring partial-parse success.
        stream = io.BytesIO(blob)
        stream.seek(offset)
        points = np.empty((count, 3), dtype=np.float64)
        xyz = [names.index(axis) for axis in ("x", "y", "z")]
        for index in range(count):
            row = stream.readline()
            if not row:
                raise InputUnavailable("PLY_ROW_COUNT_MISMATCH")
            tokens = row.split(None, len(properties))
            if len(tokens) != len(properties):
                raise InputUnavailable("PLY_SCALAR_COUNT_MISMATCH")
            values = []
            for token, (_, dtype) in zip(tokens, properties):
                grammar = FLOAT_TOKEN if dtype[0] == "f" else INTEGER_TOKEN
                if grammar.fullmatch(token) is None:
                    raise InputUnavailable("INVALID_PLY_SCALAR")
                try:
                    value = float(token) if dtype[0] == "f" else int(token)
                except ValueError as exc:
                    raise InputUnavailable("INVALID_PLY_SCALAR") from exc
                if dtype[0] in "iu":
                    bounds = np.iinfo(np.dtype(dtype))
                    if not bounds.min <= value <= bounds.max:
                        raise InputUnavailable("PLY_INTEGER_RANGE")
                elif not math.isfinite(value) or abs(value) > np.finfo(np.dtype(dtype)).max:
                    raise InputUnavailable("PLY_FLOAT_RANGE")
                # ASCII property types have the same numeric meaning as their
                # binary equivalents; do not silently promote f4 input to f8.
                values.append(np.dtype(dtype).type(value).item())
            points[index] = [values[column] for column in xyz]
        if stream.read(1):
            raise InputUnavailable("PLY_ROW_COUNT_MISMATCH")
    else:
        endian = "<" if encoding == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + kind) for name, kind in properties])
        if len(body) != count * dtype.itemsize:
            raise InputUnavailable("PLY_BINARY_SIZE_MISMATCH")
        values = np.frombuffer(body, dtype=dtype, count=count)
        if any(not np.isfinite(values[name]).all() for name, _ in properties):
            raise InputUnavailable("NONFINITE_PLY_SCALAR")
        points = np.column_stack([values[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    if not np.isfinite(points).all() or float(np.max(np.abs(points))) > 1e6:
        raise InputUnavailable("UNSUPPORTED_POINT_NUMERICS")
    points.setflags(write=False)
    return points


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputUnavailable("DUPLICATE_DATUM_JSON_KEY")
        result[key] = value
    return result


def _no_constant(value):
    raise InputUnavailable("NONFINITE_DATUM_JSON")


def _datum_array(value):
    if not isinstance(value, list) or not 36 <= len(value) <= 5000:
        raise InputUnavailable("DATUM_ARRAY_BUDGET")
    if any(not isinstance(row, list) or len(row) != 3
           or any(type(x) not in (float, int) for x in row) for row in value):
        raise InputUnavailable("INVALID_DATUM_ARRAY")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or float(np.max(np.abs(array))) > 1e6:
        raise InputUnavailable("INVALID_DATUM_NUMERICS")
    array.setflags(write=False)
    return array


def bind_registered_scan(*, scan_bytes, cad_bytes, datum_bytes):
    """Validate content-bound datum declarations before any native CAD import."""
    _bounded_bytes(cad_bytes, MAX_CAD_BYTES, "CAD_BYTE_BUDGET")
    _bounded_bytes(scan_bytes, MAX_PLY_BYTES, "PLY_BYTE_BUDGET")
    _bounded_bytes(datum_bytes, MAX_DATUM_BYTES, "DATUM_BYTE_BUDGET")
    try:
        datum = json.loads(datum_bytes, object_pairs_hook=_unique_json_object, parse_constant=_no_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InputUnavailable("INVALID_DATUM_JSON") from exc
    if not isinstance(datum, dict) or set(datum) != DATUM_KEYS or type(datum["schema_version"]) is not int or datum["schema_version"] != 1:
        raise InputUnavailable("INVALID_DATUM_SCHEMA")
    if (datum["unit"] != "mm" or datum["reference_frame"] != "CAD_REFERENCE"
            or datum["selection_method"] != "EXPLICIT_STABLE_REGION_SELECTION"):
        raise InputUnavailable("UNSUPPORTED_DATUM_DECLARATIONS")
    scan_digest, cad_digest = hashlib.sha256(scan_bytes).hexdigest(), hashlib.sha256(cad_bytes).hexdigest()
    if datum["scan_sha256"] != scan_digest or datum["cad_sha256"] != cad_digest:
        raise InputUnavailable("DATUM_SOURCE_BINDING_MISMATCH")
    points = parse_point_cloud_ply(scan_bytes)
    cad_points = _datum_array(datum["cad_datum_points"])
    normals = _datum_array(datum["cad_datum_normals"])
    if cad_points.shape != normals.shape or not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0, atol=1e-6):
        raise InputUnavailable("INVALID_DATUM_NORMALS")
    indices = datum["scan_datum_row_indices"]
    if (not isinstance(indices, list) or not 36 <= len(indices) <= 5000
            or any(type(index) is not int or not 0 <= index < len(points) for index in indices)
            or len(set(indices)) != len(indices) or len(points) - len(indices) < 12):
        raise InputUnavailable("INVALID_DATUM_ROW_SELECTION")
    indices = np.asarray(indices, dtype=np.int64)
    indices.setflags(write=False)
    return RegisteredScanInput(points, cad_points, normals, indices, scan_digest, cad_digest,
                               hashlib.sha256(datum_bytes).hexdigest())


def _read_regular_snapshot(path, limit):
    """Bound a regular file before read; never block on FIFO or follow final symlink."""
    if not isinstance(path, (str, Path)) or not Path(path).is_absolute():
        raise InputUnavailable("EXPLICIT_ABSOLUTE_INPUT_PATH_REQUIRED")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise InputUnavailable("INPUT_FILE_TYPE_OR_BYTE_BUDGET")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            blob = stream.read(limit + 1)
        after = os.fstat(fd)
        if (len(blob) != before.st_size or len(blob) > limit
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise InputUnavailable("INPUT_CHANGED_DURING_SNAPSHOT")
        return blob
    finally:
        os.close(fd)


def load_registered_case(*, cad_path, scan_path, datum_path, allow_ocp=False):
    """Read explicit approved inputs; normal callers must enforce custody first.

    CAD is imported from an immutable-byte copy in an owned temporary directory,
    so native parsing cannot race the bound source file. No gmsh fallback, input
    directory discovery, truth read, unit inference, resampling or pose fallback.
    External supervision must enforce the fixed CPU/wall/RSS limits.
    """
    if type(allow_ocp) is not bool or not allow_ocp:
        return {"status": "UNAVAILABLE", "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"],
                "production_use_allowed": False}
    try:
        suffix = Path(cad_path).suffix.lower()
        if suffix not in (".step", ".stp", ".brep"):
            raise InputUnavailable("UNSUPPORTED_CAD_FORMAT")
        cad = _read_regular_snapshot(cad_path, MAX_CAD_BYTES)
        scan = _read_regular_snapshot(scan_path, MAX_PLY_BYTES)
        datum = _read_regular_snapshot(datum_path, MAX_DATUM_BYTES)
        inputs = bind_registered_scan(scan_bytes=scan, cad_bytes=cad, datum_bytes=datum)
        if __package__:
            from .cad_import import CADImporter
        else:
            from cad_import import CADImporter
        importer = CADImporter(linear_deflection=.25, angular_deflection=.5, allow_ocp=True)
        if importer._check_occ_available() is not True or importer.native_binding != "ocp":
            raise InputUnavailable("EXPLICIT_OCP_BINDING_REQUIRED")
        with tempfile.TemporaryDirectory(prefix="cad-guided-input-") as directory:
            snapshot = Path(directory) / ("snapshot" + suffix)
            with snapshot.open("xb") as stream:
                stream.write(cad)
            result = importer.import_file(str(snapshot), retain_native_shape=True)
        if result.success is not True or result.native_shape is None or result.native_binding != "ocp":
            raise InputUnavailable("NATIVE_CAD_IMPORT_UNAVAILABLE")
        return {"status": "AVAILABLE", "reasons": [], "cad_result": result,
                "scan_input": inputs, "source_bindings": inputs.binding_summary(),
                "production_use_allowed": False}
    except InputUnavailable as exc:
        return {"status": "UNAVAILABLE", "reasons": [str(exc)], "production_use_allowed": False}
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return {"status": "UNAVAILABLE", "reasons": ["INPUT_LOADING_FAILED"], "production_use_allowed": False}
