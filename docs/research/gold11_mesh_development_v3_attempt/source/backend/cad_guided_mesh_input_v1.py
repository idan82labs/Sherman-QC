"""Versioned, pure-byte mesh PLY and coverage-aware scan preparation.

This maintained offline adapter imports no application/native/research runtime.
It validates every supported vertex scalar and face index; faces are disclosed,
not used to invent points or silently repair the mesh. Full raw XYZ is retained
for local measurement; sampled XYZ is for registration/proposals only.

Calling these functions does not authorize physical-data access. Units, frame,
datum stability, continuous surface coverage and metrology remain unverified.
The frozen R0 input adapter and geometry-blind lineage planner are unchanged.
"""

import hashlib
import json
import io
import math
import re

import numpy as np


MAX_PLY_BYTES = 64 * 1024**2
MAX_VERTICES = 1_000_000
MAX_FACES = 2_000_000
MAX_HEADER_BYTES = 64 * 1024
MAX_ELEMENTS = 32
MAX_PROPERTIES = 32
MAX_FACE_VERTICES = 16
MAX_FACE_INDICES = 8_000_000
MAX_ASCII_ROW_BYTES = 196608
MAX_SCALAR_TOKEN_BYTES = 128
MAX_ABS_POSITION = 1e6
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
INTEGER_TOKEN = re.compile(rb"[+-]?[0-9]+")
FLOAT_TOKEN = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class MeshInputUnavailable(ValueError):
    """Fixed-code input/preparation refusal, never a part-defect decision."""


def _require(condition, code):
    if not condition:
        raise MeshInputUnavailable(code)


def _header(blob, facts):
    offset, terminated = 0, False
    while offset < min(len(blob), MAX_HEADER_BYTES):
        end = blob.find(b"\n", offset, MAX_HEADER_BYTES)
        _require(end >= 0, "PLY_HEADER_BUDGET_OR_TERMINATOR")
        try:
            line = blob[offset:end].rstrip(b"\r").decode("ascii")
        except UnicodeDecodeError as exc:
            raise MeshInputUnavailable("PLY_NONASCII_HEADER") from exc
        first = offset == 0
        offset = end + 1
        if first:
            _require(line == "ply", "INVALID_PLY_MAGIC")
            continue
        if line == "end_header":
            terminated = True
            break
        words = line.split()
        _require(bool(words), "INVALID_PLY_HEADER")
        kind = words[0]
        if kind in ("comment", "obj_info"):
            facts["comments" if kind == "comment" else "obj_info"].append(line[len(kind):].lstrip())
        elif kind == "format":
            _require(facts["format"] is None and not facts["elements"] and len(words) == 3,
                     "INVALID_PLY_FORMAT_DECLARATION")
            facts["format"], facts["version"] = words[1:]
        elif kind == "element":
            _require(facts["format"] is not None and len(words) == 3
                     and words[1].isidentifier() and len(words[1]) <= 64
                     and words[2].isascii() and words[2].isdecimal() and len(words[2]) <= 8
                     and len(facts["elements"]) < MAX_ELEMENTS
                     and not any(row["name"] == words[1] for row in facts["elements"]),
                     "INVALID_PLY_ELEMENT_DECLARATION")
            facts["elements"].append({"name": words[1], "count": int(words[2]), "properties": []})
        elif kind == "property":
            _require(bool(facts["elements"]), "PLY_PROPERTY_WITHOUT_ELEMENT")
            properties = facts["elements"][-1]["properties"]
            _require(len(properties) < MAX_PROPERTIES, "PLY_PROPERTY_BUDGET")
            if len(words) == 3:
                record = {"kind": "scalar", "type": words[1], "name": words[2]}
            elif len(words) == 5 and words[1] == "list":
                record = {"kind": "list", "count_type": words[2], "item_type": words[3], "name": words[4]}
            else:
                raise MeshInputUnavailable("INVALID_PLY_PROPERTY_DECLARATION")
            _require(record["name"].isidentifier() and len(record["name"]) <= 64
                     and not any(row["name"] == record["name"] for row in properties),
                     "INVALID_PLY_PROPERTY_NAME")
            properties.append(record)
        else:
            facts["unhandled_header_fields"].append(line)
    facts["header_bytes"] = offset
    facts["header_terminated"] = terminated
    _require(terminated, "PLY_HEADER_BUDGET_OR_TERMINATOR")
    return offset


def _admit_layout(facts):
    _require(facts["format"] in ("ascii", "binary_little_endian", "binary_big_endian")
             and facts["version"] == "1.0", "UNSUPPORTED_PLY_FORMAT")
    _require(not facts["unhandled_header_fields"], "UNHANDLED_PLY_HEADER_FIELDS")
    elements = facts["elements"]
    _require(bool(elements) and elements[0]["name"] == "vertex", "PLY_VERTEX_FIRST_REQUIRED")
    vertex = elements[0]
    _require(1 <= vertex["count"] <= MAX_VERTICES, "PLY_VERTEX_BUDGET")
    properties = vertex["properties"]
    _require(properties and all(row["kind"] == "scalar" and row["type"] in PLY_TYPES for row in properties),
             "UNSUPPORTED_PLY_VERTEX_PROPERTIES")
    _require(all(any(row["name"] == axis for row in properties) for axis in ("x", "y", "z")),
             "PLY_XYZ_REQUIRED")
    if len(elements) == 1:
        return vertex, None
    _require(len(elements) == 2 and elements[1]["name"] == "face", "UNHANDLED_TRAILING_PLY_GEOMETRY")
    face = elements[1]
    _require(0 <= face["count"] <= MAX_FACES, "PLY_FACE_BUDGET")
    fields = face["properties"]
    _require(len(fields) == 1 and fields[0]["kind"] == "list"
             and fields[0]["name"] in ("vertex_indices", "vertex_index")
             and fields[0]["count_type"] in PLY_TYPES and fields[0]["item_type"] in PLY_TYPES
             and PLY_TYPES[fields[0]["count_type"]][0] in "iu"
             and PLY_TYPES[fields[0]["item_type"]][0] in "iu", "UNSUPPORTED_PLY_FACE_PROPERTIES")
    return vertex, face


def _scalar(token, kind):
    _require(len(token) <= MAX_SCALAR_TOKEN_BYTES, "PLY_SCALAR_TOKEN_BUDGET")
    _require((FLOAT_TOKEN if kind[0] == "f" else INTEGER_TOKEN).fullmatch(token) is not None,
             "INVALID_PLY_SCALAR")
    try:
        value = float(token) if kind[0] == "f" else int(token)
    except (ValueError, OverflowError) as exc:
        raise MeshInputUnavailable("INVALID_PLY_SCALAR") from exc
    dtype = np.dtype(kind)
    if kind[0] in "iu":
        bounds = np.iinfo(dtype)
        _require(bounds.min <= value <= bounds.max, "PLY_INTEGER_RANGE")
    else:
        _require(math.isfinite(value) and abs(value) <= np.finfo(dtype).max, "PLY_FLOAT_RANGE")
    return dtype.type(value).item()


def _ascii_row(stream, max_tokens):
    row = stream.readline(MAX_ASCII_ROW_BYTES + 1)
    _require(bool(row), "PLY_ROW_COUNT_MISMATCH")
    _require(len(row) <= MAX_ASCII_ROW_BYTES, "PLY_ASCII_ROW_BUDGET")
    tokens = row.split(None, max_tokens)
    _require(len(tokens) <= max_tokens, "PLY_SCALAR_COUNT_MISMATCH")
    return tokens


def _ascii_body(blob, offset, vertex, face):
    stream = io.BytesIO(blob)
    stream.seek(offset)
    properties = vertex["properties"]
    names = [row["name"] for row in properties]
    xyz = [names.index(axis) for axis in ("x", "y", "z")]
    points = np.empty((vertex["count"], 3), dtype=np.float64)
    for index in range(vertex["count"]):
        tokens = _ascii_row(stream, len(properties))
        _require(len(tokens) == len(properties), "PLY_SCALAR_COUNT_MISMATCH")
        values = [_scalar(token, PLY_TYPES[row["type"]]) for token, row in zip(tokens, properties)]
        points[index] = [values[column] for column in xyz]
    total_indices = 0
    if face is not None:
        prop = face["properties"][0]
        for _ in range(face["count"]):
            tokens = _ascii_row(stream, MAX_FACE_VERTICES + 1)
            _require(bool(tokens), "PLY_FACE_ROW_COUNT_MISMATCH")
            count = _scalar(tokens[0], PLY_TYPES[prop["count_type"]])
            _require(3 <= count <= MAX_FACE_VERTICES, "PLY_FACE_VERTEX_COUNT")
            _require(len(tokens) == count + 1, "PLY_FACE_SCALAR_COUNT_MISMATCH")
            total_indices += count
            _require(total_indices <= MAX_FACE_INDICES, "PLY_FACE_INDEX_BUDGET")
            indices = [_scalar(token, PLY_TYPES[prop["item_type"]]) for token in tokens[1:]]
            _require(all(0 <= index < vertex["count"] for index in indices), "PLY_FACE_VERTEX_INDEX")
            _require(len(set(indices)) == count, "PLY_FACE_DUPLICATE_VERTEX_INDEX")
    _require(not stream.read(1), "PLY_TRAILING_BYTES")
    return points, total_indices


def _binary_body(blob, offset, vertex, face, encoding):
    endian = "<" if encoding == "binary_little_endian" else ">"
    dtype = np.dtype([(row["name"], endian + PLY_TYPES[row["type"]]) for row in vertex["properties"]])
    size = vertex["count"] * dtype.itemsize
    _require(offset + size <= len(blob), "PLY_BINARY_SIZE_MISMATCH")
    records = np.frombuffer(blob, dtype=dtype, offset=offset, count=vertex["count"])
    for name in dtype.names:
        _require(np.isfinite(records[name]).all(), "NONFINITE_PLY_SCALAR")
    points = np.column_stack([records[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    offset += size
    total_indices = 0
    if face is not None:
        prop = face["properties"][0]
        count_dtype = np.dtype(endian + PLY_TYPES[prop["count_type"]])
        index_dtype = np.dtype(endian + PLY_TYPES[prop["item_type"]])
        # Common triangle layout is validated in one vectorized bounded view.
        # Fall back only when the byte layout is genuinely variable; never
        # assume triangle counts from file naming, case IDs or an expected score.
        triangle_dtype = np.dtype([("count", count_dtype), ("indices", index_dtype, (3,))])
        triangle_bytes = face["count"] * triangle_dtype.itemsize
        if len(blob) - offset == triangle_bytes:
            triangles = np.frombuffer(blob, dtype=triangle_dtype, count=face["count"], offset=offset)
            if np.all(triangles["count"] == 3):
                _require(face["count"] == 0 or 3 <= MAX_FACE_VERTICES, "PLY_FACE_VERTEX_COUNT")
                _require(3 * face["count"] <= MAX_FACE_INDICES, "PLY_FACE_INDEX_BUDGET")
                indices = triangles["indices"]
                _require(np.all(indices >= 0) and np.all(indices < vertex["count"]), "PLY_FACE_VERTEX_INDEX")
                _require(np.all(indices[:, 0] != indices[:, 1])
                         and np.all(indices[:, 0] != indices[:, 2])
                         and np.all(indices[:, 1] != indices[:, 2]), "PLY_FACE_DUPLICATE_VERTEX_INDEX")
                return points, 3 * face["count"]
        for _ in range(face["count"]):
            _require(offset + count_dtype.itemsize <= len(blob), "PLY_BINARY_FACE_COUNT_MISSING")
            count = int(np.frombuffer(blob, dtype=count_dtype, offset=offset, count=1)[0])
            offset += count_dtype.itemsize
            _require(3 <= count <= MAX_FACE_VERTICES, "PLY_FACE_VERTEX_COUNT")
            total_indices += count
            _require(total_indices <= MAX_FACE_INDICES, "PLY_FACE_INDEX_BUDGET")
            size = count * index_dtype.itemsize
            _require(offset + size <= len(blob), "PLY_BINARY_FACE_SIZE_MISMATCH")
            indices = np.frombuffer(blob, dtype=index_dtype, offset=offset, count=count)
            _require(np.all(indices >= 0) and np.all(indices < vertex["count"]), "PLY_FACE_VERTEX_INDEX")
            _require(len(np.unique(indices)) == count, "PLY_FACE_DUPLICATE_VERTEX_INDEX")
            offset += size
    _require(offset == len(blob), "PLY_TRAILING_BYTES")
    return points, total_indices


def _immutable_array(value, dtype):
    """Use immutable byte backing, not merely a resettable writeable flag."""
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def parse_mesh_ply_v1(blob):
    """Validate bytes and return ordered XYZ, or no coordinates plus header facts.

    Only vertex-first scalar point rows followed by an optional standard face
    list are supported. Every declared face list is consumed and index-checked;
    it is disclosed but is not used to alter, sample, deduplicate or repair XYZ.
    Unsupported trailing elements decline the whole coordinate result.
    """
    facts = {"format": None, "version": None, "comments": [], "obj_info": [], "elements": [],
             "header_bytes": None, "header_terminated": False, "unhandled_header_fields": [],
             "header_declarations_verified": False}
    base = {"header": facts, "points": None, "production_use_allowed": False,
            "coordinate_frame_status": "UNVERIFIED", "units_status": "UNVERIFIED",
            "row_order": "RAW_VERTEX_ORDER_NO_TRANSFORMATION", "face_index_count": None,
            "source_sha256": None, "vertex_count": None, "face_count": None,
            "mesh_validity_status": "NOT_QUALIFIED"}
    try:
        _require(type(blob) is bytes and 0 < len(blob) <= MAX_PLY_BYTES, "PLY_BYTE_BUDGET")
        base["source_sha256"] = hashlib.sha256(blob).hexdigest()
        offset = _header(blob, facts)
        vertex, face = _admit_layout(facts)
        points, total_indices = (_ascii_body(blob, offset, vertex, face) if facts["format"] == "ascii"
                                 else _binary_body(blob, offset, vertex, face, facts["format"]))
        _require(np.isfinite(points).all() and np.max(np.abs(points)) <= MAX_ABS_POSITION,
                 "UNSUPPORTED_POINT_NUMERICS")
        points = _immutable_array(points, "<f8")
        return {**base, "status": "AVAILABLE", "reasons": [], "points": points,
                "face_index_count": total_indices, "vertex_count": len(points),
                "face_count": 0 if face is None else face["count"],
                "validated_unused_vertex_properties": [p["name"] for p in vertex["properties"]
                                                       if p["name"] not in ("x", "y", "z")],
                "trailing_geometry_status": ("NO_TRAILING_ELEMENTS" if face is None
                    else "FACE_LIST_VALIDATED_NOT_USED_FOR_COORDINATES")}
    except MeshInputUnavailable as exc:
        return {**base, "status": "UNAVAILABLE", "reasons": [str(exc)],
                "trailing_geometry_status": "NOT_VALIDATED_COORDINATES_WITHHELD"}



MAX_ENGINE_ROWS = 200_000
MIN_INSPECTION_ROWS = 12
MIN_DATUM_ROWS = 36
MAX_DATUM_ROWS = 5_000
MAX_VOXEL_LEVEL = 20
COVERAGE_CHUNK_ROWS = 65_536
SAMPLING_METHOD = "DYADIC_AABB_VOXEL_CENTER_REPRESENTATIVE_V1"


def _selection_declarations(datum_raw_row_indices, max_engine_rows, displacement_limit):
    _require(type(max_engine_rows) is int
             and MIN_INSPECTION_ROWS <= max_engine_rows <= MAX_ENGINE_ROWS, "ENGINE_ROW_BUDGET")
    _require(type(datum_raw_row_indices) in (list, tuple)
             and len(datum_raw_row_indices) <= MAX_DATUM_ROWS, "DATUM_ROW_CONTAINER_OR_BUDGET")
    datums = tuple(datum_raw_row_indices[:MAX_DATUM_ROWS + 1])
    _require(len(datums) == 0 or MIN_DATUM_ROWS <= len(datums) <= MAX_DATUM_ROWS,
             "DATUM_ROW_COUNT")
    _require(all(type(row) is int and 0 <= row < MAX_VERTICES for row in datums),
             "DATUM_RAW_ROW_INDEX")
    _require(len(set(datums)) == len(datums), "DUPLICATE_DATUM_RAW_ROW_INDEX")
    _require(max_engine_rows - len(datums) >= MIN_INSPECTION_ROWS, "INSPECTION_ROW_BUDGET")
    if displacement_limit is not None:
        _require(type(displacement_limit) in (int, float), "DISPLACEMENT_LIMIT")
        try:
            displacement_limit = float(displacement_limit)
        except OverflowError as exc:
            raise MeshInputUnavailable("DISPLACEMENT_LIMIT") from exc
        _require(math.isfinite(displacement_limit) and displacement_limit >= 0, "DISPLACEMENT_LIMIT")
    return datums, displacement_limit


def _voxel_codes(fine_bins, level):
    # level <= 20: three packed nonnegative coordinates occupy <= 60 bits.
    # Dyadic right shifts produce nested partitions, so occupied-cell count is
    # monotone in level. This is not true for arbitrary changing cell widths.
    bins = fine_bins >> (MAX_VOXEL_LEVEL - level)
    return (bins[:, 0] << (2 * level)) | (bins[:, 1] << level) | bins[:, 2]


def _voxel_representatives(points, inspection_raw_rows, inspection_budget):
    inspection_points = points[inspection_raw_rows]
    minimum = np.min(inspection_points, axis=0)
    maximum = np.max(inspection_points, axis=0)
    extent = float(np.max(maximum - minimum))
    if extent == 0:
        fine_bins = np.zeros(inspection_points.shape, dtype=np.int64)
    else:
        scaled = (inspection_points - minimum) / extent * (1 << MAX_VOXEL_LEVEL)
        fine_bins = np.floor(scaled).astype(np.int64)
        np.clip(fine_bins, 0, (1 << MAX_VOXEL_LEVEL) - 1, out=fine_bins)

    # Find finest admitted nested grid using <= 5 count probes plus final grouping.
    # Level zero has exactly one occupied inspection cell and is always feasible.
    lower, upper = 0, MAX_VOXEL_LEVEL + 1
    while lower + 1 < upper:
        level = (lower + upper) // 2
        if len(np.unique(_voxel_codes(fine_bins, level))) <= inspection_budget:
            lower = level
        else:
            upper = level
    level = lower
    _, inverse = np.unique(_voxel_codes(fine_bins, level), return_inverse=True)
    voxel_count = int(np.max(inverse)) + 1
    width = extent / (1 << level)
    coarse_bins = fine_bins >> (MAX_VOXEL_LEVEL - level)
    centers = minimum + (coarse_bins.astype(np.float64) + .5) * width
    delta = inspection_points - centers
    # hypot is scale-safe: squaring 1e-200 would underflow to zero and could
    # falsely certify zero displacement or pick a non-nearest representative.
    distance = np.hypot.reduce(delta, axis=1)
    minimum_distance = np.full(voxel_count, np.inf)
    np.minimum.at(minimum_distance, inverse, distance)
    # Exact floating-point ties are broken by the lower ORIGINAL row number.
    eligible = distance == minimum_distance[inverse]
    representatives = np.full(voxel_count, len(points), dtype=np.int64)
    np.minimum.at(representatives, inverse[eligible], inspection_raw_rows[eligible])
    _require(np.all(representatives < len(points)), "VOXEL_REPRESENTATIVE_UNAVAILABLE")
    primary_representatives = representatives.copy()
    extras = np.empty(0, dtype=np.int64)
    if voxel_count < MIN_INSPECTION_ROWS:
        primary_set = set(representatives.tolist())
        extras = np.array([int(row) for row in inspection_raw_rows[:MIN_INSPECTION_ROWS]
                           if int(row) not in primary_set], dtype=np.int64)
        extras = extras[:MIN_INSPECTION_ROWS - voxel_count]
        # There are at most voxel_count primary rows among the first 12 rows,
        # hence enough extras exist without scanning/collecting the full cloud.
        _require(len(extras) + voxel_count == MIN_INSPECTION_ROWS, "INSPECTION_ROW_BUDGET")
        representatives = np.concatenate((representatives, extras))
    return representatives, primary_representatives[inverse], {
        "voxel_level": level, "voxel_width_raw_units": width,
        "occupied_inspection_voxels": voxel_count, "minimum_row_supplement_count": len(extras),
        "inspection_aabb_minimum": minimum.tolist(), "inspection_aabb_maximum": maximum.tolist(),
        "aabb_cube_extent_raw_units": extent,
        "axis_alignment": "RAW_COORDINATE_AXES_NOT_ROTATION_INVARIANT",
    }


def _assigned_coverage(raw_points, representative_raw_rows):
    maximum, worst_raw_row = 0.0, 0
    for start in range(0, len(raw_points), COVERAGE_CHUNK_ROWS):
        stop = min(start + COVERAGE_CHUNK_ROWS, len(raw_points))
        delta = raw_points[start:stop] - raw_points[representative_raw_rows[start:stop]]
        distances = np.hypot.reduce(delta, axis=1)
        local = int(np.argmax(distances))
        if float(distances[local]) > maximum:
            maximum = float(distances[local])
            worst_raw_row = start + local
    return maximum, worst_raw_row


def _array_sha256(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def prepare_mesh_scan_v1(*, scan_bytes, datum_raw_row_indices=(),
                         max_engine_rows=MAX_ENGINE_ROWS,
                         max_point_displacement_raw_units=None):
    """Validate the whole PLY, retain full raw XYZ, and prepare bounded samples.

    No file access, scale inference, alignment, stable-datum selection, model,
    mesh interpolation, reference geometry or case-specific identifiers occur.
    Empty datums explicitly mean provisional whole-part preparation, not datum-
    based metrology. Nonempty declarations preserve every datum and its ordering.

    The inspection partition uses actual raw points nearest dyadic cell centers.
    This is a geometry-aware coverage heuristic, not label-selected resolution.
    Sampling depends on raw axes/AABB and can change with pose; it is not claimed
    rotation invariant. The finest admitted level is <= 20, and output need not
    fill the point budget. Full raw points remain available for final fitting.

    The receipt measures every raw vertex to its assigned retained representative
    in float64. It is not the exact nearest representative distance, continuous
    surface Hausdorff distance, an interval-certified bound, scan calibration or
    feature preservation. An optional caller-frozen displacement ceiling refuses
    the entire prepared output if exceeded; no threshold is automatically tuned.
    """
    base = {
        "schema": "CAD_GUIDED_MESH_INPUT_V1", "status": "UNAVAILABLE", "reasons": [],
        "raw_points": None, "points": None, "engine_to_raw_row_indices": None,
        "inspection_raw_row_indices": None,
        "datum_raw_row_indices": None, "datum_engine_row_indices": None,
        "inspection_engine_row_indices": None, "raw_to_engine_row_indices": None,
        "raw_to_representative_engine_row_indices": None,
        "source_bindings": {}, "coverage": None,
        "production_use_allowed": False, "coordinate_frame_status": "UNVERIFIED",
        "units_status": "UNVERIFIED", "mesh_validity_status": "NOT_QUALIFIED",
        "measurement_cloud_contract": "FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING",
        "measurement_role_contract": "LOCAL_FITTING_EXCLUDES_ALL_DECLARED_RAW_DATUM_ROWS",
        "sample_cloud_contract": "REGISTRATION_AND_PROPOSALS_ONLY",
    }
    try:
        datums, displacement_limit = _selection_declarations(
            datum_raw_row_indices, max_engine_rows, max_point_displacement_raw_units
        )
        parsed = parse_mesh_ply_v1(scan_bytes)
        if parsed["status"] != "AVAILABLE":
            return {**base, "reasons": parsed["reasons"], "input_admission": parsed}
        raw_points = parsed["points"]
        raw_count = len(raw_points)
        _require(all(row < raw_count for row in datums), "DATUM_RAW_ROW_INDEX")
        _require(raw_count - len(datums) >= MIN_INSPECTION_ROWS, "INSPECTION_ROW_BUDGET")
        datum_array = np.array(datums, dtype=np.int64)
        datum_mask = np.zeros(raw_count, dtype=bool)
        datum_mask[datum_array] = True
        inspection_raw_rows = np.flatnonzero(~datum_mask)
        if raw_count <= max_engine_rows:
            engine_to_raw = np.arange(raw_count, dtype=np.int64)
            representative_raw_rows = engine_to_raw.copy()
            voxel_facts = {
                "voxel_level": None, "voxel_width_raw_units": 0.0,
                "occupied_inspection_voxels": None, "minimum_row_supplement_count": 0,
                "sampling_status": "IDENTITY_ALL_RAW_ROWS_RETAINED",
            }
        else:
            representatives, assigned_raw, voxel_facts = _voxel_representatives(
                raw_points, inspection_raw_rows, max_engine_rows - len(datums)
            )
            engine_to_raw = np.sort(np.concatenate((datum_array, representatives)))
            representative_raw_rows = np.arange(raw_count, dtype=np.int64)
            representative_raw_rows[inspection_raw_rows] = assigned_raw
            # All retained rows, including deterministic supplements, map to self.
            representative_raw_rows[engine_to_raw] = engine_to_raw
            voxel_facts["sampling_status"] = "VOXEL_REPRESENTATIVES_PLUS_ALL_DATUMS"
        _require(len(engine_to_raw) <= max_engine_rows
                 and len(np.unique(engine_to_raw)) == len(engine_to_raw), "ENGINE_ROW_BUDGET")
        raw_to_engine = np.full(raw_count, -1, dtype=np.int64)
        raw_to_engine[engine_to_raw] = np.arange(len(engine_to_raw), dtype=np.int64)
        raw_to_representative = raw_to_engine[representative_raw_rows]
        _require(np.all(raw_to_representative >= 0), "INCOMPLETE_RAW_VERTEX_COVERAGE")
        maximum_displacement, worst_row = _assigned_coverage(raw_points, representative_raw_rows)
        coverage = {
            **voxel_facts, "coverage_domain": "ALL_VALIDATED_RAW_VERTICES",
            "assignment": "DATUM_SELF_OR_INSPECTION_SAME_VOXEL_REPRESENTATIVE",
            "distance_measure": "EUCLIDEAN_RAW_VERTEX_TO_ASSIGNED_RETAINED_RAW_VERTEX",
            "numeric_precision": "FLOAT64_NOT_INTERVAL_CERTIFIED",
            "raw_vertex_count": raw_count, "retained_vertex_count": len(engine_to_raw),
            "datum_row_count": len(datums), "inspection_raw_row_count": len(inspection_raw_rows),
            "assigned_vertex_count": raw_count, "dropped_sample_vertex_count": raw_count - len(engine_to_raw),
            "maximum_point_displacement_raw_units": maximum_displacement,
            "maximum_displacement_raw_row_index": worst_row,
            "declared_displacement_limit_raw_units": displacement_limit,
            "displacement_limit_status": "NOT_SPECIFIED" if displacement_limit is None else
                ("WITHIN_DECLARED_LIMIT" if maximum_displacement <= displacement_limit else "EXCEEDED"),
            "datum_displacement_raw_units": 0.0 if datums else None,
            "continuous_surface_coverage": "NOT_ESTABLISHED",
            "bend_feature_preservation": "NOT_ESTABLISHED",
            "physical_metrology_coverage": "NOT_ESTABLISHED",
        }
        if displacement_limit is not None and maximum_displacement > displacement_limit:
            return {**base, "reasons": ["POINT_DISPLACEMENT_LIMIT_EXCEEDED"], "coverage": coverage,
                    "source_bindings": {"scan_sha256": parsed["source_sha256"]}}

        engine_to_raw = _immutable_array(engine_to_raw, "<i8")
        datum_array = _immutable_array(datum_array, "<i8")
        raw_to_engine = _immutable_array(raw_to_engine, "<i8")
        raw_to_representative = _immutable_array(raw_to_representative, "<i8")
        datum_engine = _immutable_array(raw_to_engine[datum_array], "<i8")
        inspection_engine = _immutable_array(np.flatnonzero(~datum_mask[engine_to_raw]), "<i8")
        inspection_raw_rows = _immutable_array(inspection_raw_rows, "<i8")
        points = _immutable_array(raw_points[engine_to_raw], "<f8")
        policy = {
            "method": SAMPLING_METHOD, "max_engine_rows": max_engine_rows,
            "max_voxel_level": MAX_VOXEL_LEVEL, "max_point_displacement_raw_units": displacement_limit,
            "minimum_inspection_rows": MIN_INSPECTION_ROWS, "datum_raw_row_indices": list(datums),
            "tie_break": "MINIMUM_ORIGINAL_RAW_ROW", "full_raw_support_retained": True,
        }
        policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        bindings = {
            "scan_sha256": parsed["source_sha256"], "raw_xyz_sha256": _array_sha256(raw_points),
            "sample_xyz_sha256": _array_sha256(points),
            "engine_to_raw_sha256": _array_sha256(engine_to_raw),
            "raw_to_representative_sha256": _array_sha256(raw_to_representative),
            "sampling_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "array_digest_encoding": "C_CONTIGUOUS_LITTLE_ENDIAN_FLOAT64_OR_INT64",
            "binding_status": "IN_PROCESS_BYTE_AND_ROW_MAP_DIGESTS_NOT_AUTHORIZATION",
        }
        return {
            **base, "status": "AVAILABLE", "raw_points": raw_points, "points": points,
            "engine_to_raw_row_indices": engine_to_raw, "datum_raw_row_indices": datum_array,
            "inspection_raw_row_indices": inspection_raw_rows,
            "datum_engine_row_indices": datum_engine, "inspection_engine_row_indices": inspection_engine,
            "raw_to_engine_row_indices": raw_to_engine,
            "raw_to_representative_engine_row_indices": raw_to_representative,
            "source_bindings": bindings, "coverage": coverage, "sampling_policy": policy,
            "input_admission": {key: value for key, value in parsed.items() if key != "points"},
            "registration_preparation_mode": ("PROVISIONAL_WHOLE_PART_NO_STABLE_DATUMS" if not datums
                                               else "CALLER_DECLARED_DATUM_ROWS_NOT_PHYSICALLY_VERIFIED"),
            "datum_stability_status": "NOT_ESTABLISHED",
        }
    except MeshInputUnavailable as exc:
        return {**base, "reasons": [str(exc)]}
