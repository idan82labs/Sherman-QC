"""Pure diagnostic geometry admission; no files, discovery, fitting or scoring.

This does not change the product PLY loader, its 200k cap, Gold labels or any
coordinate frame. A validated PLY face index list is not a qualified surface;
header comments and unit/frame-like strings remain unverified declarations.
"""

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
REFERENCE_CONSISTENCY_ATOL = 1e-8
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
INTEGER_TOKEN = re.compile(rb"[+-]?[0-9]+")
FLOAT_TOKEN = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class GeometryAdmissionUnavailable(ValueError):
    """Fixed-code diagnostic refusal, never a defect or relabeling decision."""


def _require(condition, code):
    if not condition:
        raise GeometryAdmissionUnavailable(code)


def _header(blob, facts):
    offset, terminated = 0, False
    while offset < min(len(blob), MAX_HEADER_BYTES):
        end = blob.find(b"\n", offset, MAX_HEADER_BYTES)
        _require(end >= 0, "PLY_HEADER_BUDGET_OR_TERMINATOR")
        try:
            line = blob[offset:end].rstrip(b"\r").decode("ascii")
        except UnicodeDecodeError as exc:
            raise GeometryAdmissionUnavailable("PLY_NONASCII_HEADER") from exc
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
                raise GeometryAdmissionUnavailable("INVALID_PLY_PROPERTY_DECLARATION")
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
        raise GeometryAdmissionUnavailable("INVALID_PLY_SCALAR") from exc
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
            for token in tokens[1:]:
                index = _scalar(token, PLY_TYPES[prop["item_type"]])
                _require(0 <= index < vertex["count"], "PLY_FACE_VERTEX_INDEX")
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
            offset += size
    _require(offset == len(blob), "PLY_TRAILING_BYTES")
    return points, total_indices


def parse_diagnostic_ply(blob):
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
            "row_order": "RAW_VERTEX_ORDER_NO_TRANSFORMATION", "face_index_count": None}
    try:
        _require(type(blob) is bytes and 0 < len(blob) <= MAX_PLY_BYTES, "PLY_BYTE_BUDGET")
        offset = _header(blob, facts)
        vertex, face = _admit_layout(facts)
        points, total_indices = (_ascii_body(blob, offset, vertex, face) if facts["format"] == "ascii"
                                 else _binary_body(blob, offset, vertex, face, facts["format"]))
        _require(np.isfinite(points).all() and np.max(np.abs(points)) <= MAX_ABS_POSITION,
                 "UNSUPPORTED_POINT_NUMERICS")
        points.setflags(write=False)
        return {**base, "status": "AVAILABLE", "reasons": [], "points": points,
                "face_index_count": total_indices,
                "trailing_geometry_status": ("NO_TRAILING_ELEMENTS" if face is None
                    else "FACE_LIST_VALIDATED_NOT_USED_FOR_COORDINATES")}
    except GeometryAdmissionUnavailable as exc:
        return {**base, "status": "UNAVAILABLE", "reasons": [str(exc)],
                "trailing_geometry_status": "NOT_VALIDATED_COORDINATES_WITHHELD"}


def _geometry_array(value, shape):
    try:
        raw = np.asarray(value)
        _require(raw.shape == shape and raw.dtype.kind in "fiu", "INVALID_REFERENCE_ARRAY")
        result = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise GeometryAdmissionUnavailable("INVALID_REFERENCE_ARRAY") from exc
    _require(np.isfinite(result).all() and np.max(np.abs(result)) <= MAX_ABS_POSITION,
             "INVALID_REFERENCE_NUMERICS")
    return result


def _line(value):
    result = _geometry_array(value, (2, 3))
    _require(float(np.linalg.norm(result[1] - result[0])) > 1e-9, "DEGENERATE_REFERENCE_LINE")
    return result


def parse_reference_endpoints(record):
    """Historical representation precedence, with explicit contradiction refusal.

    Priority is span_endpoints, line_start/line_end, then anchor/axis/span_mm.
    Existing coordinates and order are returned unchanged from the winning
    representation. Coexisting representations must describe the same unoriented
    finite segment within a disclosed 1e-8 coordinate-unit numeric check. This
    check is not a scorer tolerance or evidence of physical carrier compatibility.
    """
    _require(type(record) is dict, "INVALID_REFERENCE_RECORD")
    choices = []
    endpoints = record.get("span_endpoints")
    # JSON empty lists are false in the historical scorer and permit fallback.
    if endpoints is not None and not (type(endpoints) in (list, tuple) and len(endpoints) == 0):
        choices.append(("span_endpoints", _line(endpoints)))
    direct = [record.get(name) for name in ("line_start", "line_end")]
    if any(value is not None for value in direct):
        _require(all(value is not None for value in direct), "INCOMPLETE_REFERENCE_ENDPOINTS")
        choices.append(("line_start_line_end", _line(direct)))
    anchored = [record.get(name) for name in ("anchor", "axis_direction", "span_mm")]
    if any(value is not None for value in anchored):
        _require(all(value is not None for value in anchored), "INCOMPLETE_REFERENCE_ANCHOR_AXIS_SPAN")
        anchor, axis = (_geometry_array(value, (3,)) for value in anchored[:2])
        span = float(_geometry_array(anchored[2], ()))
        length = float(np.linalg.norm(axis))
        _require(length > 1e-12 and span > 1e-9, "INVALID_REFERENCE_AXIS_OR_SPAN")
        half = axis / length * span / 2.
        choices.append(("anchor_axis_span", _line([anchor - half, anchor + half])))
    _require(bool(choices), "REFERENCE_ENDPOINTS_UNAVAILABLE")
    name, selected = choices[0]
    for _, other in choices[1:]:
        _require(np.allclose(selected, other, atol=REFERENCE_CONSISTENCY_ATOL, rtol=0)
                 or np.allclose(selected, other[::-1], atol=REFERENCE_CONSISTENCY_ATOL, rtol=0),
                 "CONTRADICTORY_REFERENCE_REPRESENTATIONS")
    selected.setflags(write=False)
    return {"endpoints": selected, "selected_representation": name,
            "representations_present": [key for key, _ in choices],
            "numeric_consistency_atol_coordinate_units": REFERENCE_CONSISTENCY_ATOL,
            "physical_carrier_convention_status": "UNVERIFIED", "production_use_allowed": False}
