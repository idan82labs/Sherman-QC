"""Bounded analytic ASCII DXF transport, not manufacture authorization.

Only pure 2D LINE/ARC/CIRCLE rows are accepted. No file access, FreeCAD imports,
preferences, curve approximation, contour repair, or compensation inference.
The caller must separately prove native coverage, contour topology, source units
and approved material/profile provenance. A local roundtrip is not shop approval.
"""
from __future__ import annotations

import math


LAYERS = ("CUT_OUTER", "CUT_INNER", "BEND_UP", "BEND_DOWN")
MAX_ENTITIES = 8192
MAX_ENDPOINT_COMPARISONS = 1_000_000
MAX_DXF_BYTES = 5_000_000
MAX_COORDINATE_MM = 1_000_000.
LINEAR_TOL_MM = 1e-7
_TAU = 2 * math.pi
_EPS = 2.220446049250313e-16


class AnalyticDxfUnavailable(ValueError):
    """Fail-closed transport refusal; no partial DXF is returned."""


def _require(condition, reason):
    if not condition:
        raise AnalyticDxfUnavailable(reason)


def _number(value, limit=MAX_COORDINATE_MM):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), "FINITE_NUMERIC_VALUE_REQUIRED")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise AnalyticDxfUnavailable("FINITE_NUMERIC_VALUE_REQUIRED") from error
    _require(math.isfinite(result) and abs(result) <= limit, "NONFINITE_OR_EXCESSIVE_NUMERIC_VALUE")
    return result


def _point(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 2, "TWO_DIMENSIONAL_POINT_REQUIRED")
    return [_number(v) for v in value]


def _normalize(rows):
    _require(isinstance(rows, (list, tuple)) and 0 < len(rows) <= MAX_ENTITIES, "ENTITY_BUDGET_OR_POPULATION")
    normalized = []
    for row in rows:
        _require(isinstance(row, dict) and row.get("layer") in LAYERS, "UNKNOWN_ENTITY_LAYER")
        kind = row.get("type")
        common = {"type", "layer"}
        if kind == "LINE":
            _require(set(row) == common | {"start", "end"}, "UNEXPECTED_LINE_FIELDS")
            start, end = _point(row["start"]), _point(row["end"])
            _require(math.dist(start, end) > LINEAR_TOL_MM, "DEGENERATE_LINE")
            normalized.append({"type": kind, "layer": row["layer"], "start": start, "end": end})
            continue
        _require(kind in ("ARC", "CIRCLE") and row["layer"] in LAYERS[:2], "UNSUPPORTED_ENTITY_OR_BEND_CURVE")
        fields = {"center", "radius_mm", "sweep_rad"}
        fields |= {"start_angle_rad"} if kind == "ARC" else {"full_circle_verified"}
        _require(set(row) == common | fields, "UNEXPECTED_CIRCULAR_FIELDS")
        center, radius = _point(row["center"]), _number(row["radius_mm"])
        _require(radius > LINEAR_TOL_MM, "DEGENERATE_RADIUS")
        sweep = _number(row["sweep_rad"], _TAU)
        result = {"type": kind, "layer": row["layer"], "center": center, "radius_mm": radius}
        if kind == "CIRCLE":
            # This is an explicit caller certificate, not inference from close
            # endpoints. Near-full and multi-turn arcs can never enter this path.
            _require(row["full_circle_verified"] is True and sweep == _TAU,
                     "CIRCLE_REQUIRES_EXPLICIT_EXACT_FULL_TURN")
            result.update(sweep_rad=_TAU, full_circle_verified=True)
        else:
            start = _number(row["start_angle_rad"], 4 * _TAU)
            _require(0 < abs(sweep) < _TAU, "PARTIAL_ARC_SWEEP_REQUIRED")
            rounding_bound = 128 * _EPS * (1 + max(abs(v) for v in center)
                + radius * (1 + abs(start) + _TAU))
            _require(rounding_bound < LINEAR_TOL_MM
                     and radius * min(abs(sweep), _TAU - abs(sweep)) > rounding_bound,
                     "ILL_CONDITIONED_ARC_SWEEP")
            # DXF ARC is counterclockwise about +Z. Reversing traversal changes
            # endpoint order, never the represented locus or cut/hole layer.
            result.update(start_angle_rad=((start + sweep) if sweep < 0 else start) % _TAU,
                          sweep_rad=abs(sweep))
        normalized.append(result)
    return normalized


def _header():
    return [(999, "DRAFT ONLY - NOT RELEASED FOR CUTTING"), (9, "$ACADVER"), (1, "AC1015"),
            (9, "$HANDSEED"), (5, "10000"),
            (9, "$INSUNITS"), (70, "4"), (9, "$MEASUREMENT"), (70, "1"),
            (9, "$INSBASE"), (10, "0"), (20, "0"), (30, "0"),
            (9, "$ANGBASE"), (50, "0"), (9, "$ANGDIR"), (70, "0")]


def _validate_cut_loops(rows):
    """Endpoint incidence only; no snapping, joining, or hole/overlap inference.

    The arithmetic-sized bound is stricter than the native numerical tolerance;
    it is not permission to repair an intentional physical contour gap. Native
    topological closure and area/overlap proof remain required by the caller.
    """
    coordinate_scale, radius_scale, phase_scale = 1., 0., 1.
    endpoints, circles = {layer: [] for layer in LAYERS[:2]}, {layer: [] for layer in LAYERS[:2]}
    for index, row in enumerate(rows):
        if row["layer"] not in endpoints:
            continue
        if row["type"] == "CIRCLE":
            circles[row["layer"]].append((math.pi * row["radius_mm"] ** 2, _TAU * row["radius_mm"]))
            continue
        if row["type"] == "LINE":
            points = (row["start"], row["end"])
        else:
            radius, center, phase = row["radius_mm"], row["center"], row["start_angle_rad"]
            points = tuple([center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)]
                           for angle in (phase, phase + row["sweep_rad"]))
            radius_scale, phase_scale = max(radius_scale, radius), max(phase_scale, abs(phase))
        coordinate_scale = max(coordinate_scale, *(abs(v) for point in points for v in point))
        endpoints[row["layer"]].extend((index, point) for point in points)
    bound = 128 * _EPS * (coordinate_scale + radius_scale * (1 + phase_scale + _TAU))
    _require(bound < LINEAR_TOL_MM, "ILL_CONDITIONED_CUT_ENDPOINTS")
    comparisons, loop_counts, layer_areas, total_perimeter = 0, {}, {}, 0.
    for layer, entries in endpoints.items():
        mates = {}
        for index, (row_index, point) in enumerate(entries):
            matches = []
            for other_index, (other_row, other_point) in enumerate(entries):
                if index == other_index or row_index == other_row:
                    continue
                comparisons += 1
                _require(comparisons <= MAX_ENDPOINT_COMPARISONS, "CUT_ENDPOINT_PROOF_BUDGET_EXCEEDED")
                if math.dist(point, other_point) <= bound:
                    matches.append(other_index)
            _require(len(matches) == 1, "OPEN_OR_AMBIGUOUS_CUT_ENDPOINT")
            mates[index] = matches[0]
        pending, components = set(range(len(entries))), list(circles[layer])
        while pending:
            first, contributions, perimeter = min(pending), [], 0.
            current, origin = first, entries[first][1]
            while True:
                _require(current in pending and (current ^ 1) in pending, "INCONSISTENT_CUT_LOOP_INCIDENCE")
                row = rows[entries[current][0]]
                a = [v - origin[i] for i, v in enumerate(entries[current][1])]
                b = [v - origin[i] for i, v in enumerate(entries[current ^ 1][1])]
                if row["type"] == "LINE":
                    contributions.append(a[0] * b[1] - b[0] * a[1])
                    perimeter += math.dist(a, b)
                else:
                    start, sweep = row["start_angle_rad"], row["sweep_rad"]
                    if current % 2:
                        start, sweep = start + sweep, -sweep
                    end, radius = start + sweep, row["radius_mm"]
                    cx, cy = [v - origin[i] for i, v in enumerate(row["center"])]
                    contributions.append(radius * cx * (math.sin(end) - math.sin(start))
                        + radius * cy * (math.cos(start) - math.cos(end)) + radius ** 2 * sweep)
                    perimeter += radius * abs(sweep)
                pending.remove(current)
                pending.remove(current ^ 1)
                current = mates[current ^ 1]
                if current == first:
                    break
            components.append((abs(math.fsum(contributions)) / 2., perimeter))
        _require(all(math.isfinite(area) and area > LINEAR_TOL_MM * perimeter
                     for area, perimeter in components), "DEGENERATE_CUT_LOOP_AREA")
        loop_counts[layer] = len(components)
        layer_areas[layer] = math.fsum(area for area, _ in components)
        total_perimeter += math.fsum(perimeter for _, perimeter in components)
    _require(loop_counts["CUT_OUTER"] == 1, "ONE_CLOSED_OUTER_CUT_REQUIRED")
    area = layer_areas["CUT_OUTER"] - layer_areas["CUT_INNER"]
    _require(math.isfinite(area) and area > LINEAR_TOL_MM * total_perimeter, "NONPOSITIVE_NET_CUT_AREA")
    return {"cut_loop_counts": loop_counts, "analytic_cut_area_mm2": area,
            "analytic_cut_area_numerical_tolerance_mm2": LINEAR_TOL_MM * total_perimeter}


def _tables():
    result = [(0, "TABLE"), (2, "LTYPE"), (5, "1"), (330, "0"), (100, "AcDbSymbolTable"),
              (70, "1"), (0, "LTYPE"), (5, "2"), (330, "1"),
              (100, "AcDbSymbolTableRecord"), (100, "AcDbLinetypeTableRecord"),
              (2, "CONTINUOUS"), (70, "0"), (3, "Solid line"), (72, "65"), (73, "0"), (40, "0"),
              (0, "ENDTAB"), (0, "TABLE"), (2, "LAYER"), (5, "3"), (330, "0"),
              (100, "AcDbSymbolTable"), (70, str(len(LAYERS) + 1))]
    for index, layer in enumerate(("0",) + LAYERS, 4):
        result += [(0, "LAYER"), (5, format(index, "X")), (330, "3"),
                   (100, "AcDbSymbolTableRecord"), (100, "AcDbLayerTableRecord"),
                   (2, layer), (70, "0"), (62, "7" if layer == "0" else str(index - 4)), (6, "CONTINUOUS")]
    result += [(0, "ENDTAB"), (0, "TABLE"), (2, "BLOCK_RECORD"), (5, "9"), (330, "0"),
               (100, "AcDbSymbolTable"), (70, "3")]
    for name, handle, layout in (("*Model_Space", "A", "20"), ("*Paper_Space", "B", "21"),
                                  ("*Paper_Space0", "C", "22")):
        result += [(0, "BLOCK_RECORD"), (5, handle), (330, "9"), (100, "AcDbSymbolTableRecord"),
                   (100, "AcDbBlockTableRecord"), (2, name), (340, layout), (70, "4"), (280, "1"), (281, "0")]
    return result + [(0, "ENDTAB")]


def _blocks():
    result = []
    for name, owner, begin, end, paper in (("*Model_Space", "A", "D", "E", False),
        ("*Paper_Space", "B", "F", "10", True), ("*Paper_Space0", "C", "11", "12", True)):
        space = [(67, "1")] if paper else []
        result += [(0, "BLOCK"), (5, begin), (330, owner), (100, "AcDbEntity"), (8, "0")] + space
        result += [(100, "AcDbBlockBegin"), (2, name), (70, "0"), (10, "0"), (20, "0"),
                   (30, "0"), (3, name), (1, ""), (0, "ENDBLK"), (5, end), (330, owner),
                   (100, "AcDbEntity"), (8, "0")] + space + [(100, "AcDbBlockEnd")]
    return result


def _objects():
    # Explicit ownership for the minimal R2000 model/paper-space records.
    result = [(0, "DICTIONARY"), (5, "13"), (330, "0"), (100, "AcDbDictionary"),
              (281, "1"), (3, "ACAD_LAYOUT"), (350, "14"),
              (0, "DICTIONARY"), (5, "14"), (102, "{ACAD_REACTORS"), (330, "13"), (102, "}"),
              (330, "13"), (100, "AcDbDictionary"), (281, "1")]
    for name, handle in (("Model", "20"), ("Layout1", "21"), ("Layout2", "22")):
        result += [(3, name), (350, handle)]
    for index, (name, handle, record) in enumerate((("Model", "20", "A"), ("Layout1", "21", "B"),
                                                  ("Layout2", "22", "C"))):
        result += [(0, "LAYOUT"), (5, handle), (102, "{ACAD_REACTORS"), (330, "14"), (102, "}"),
                   (330, "14"), (100, "AcDbPlotSettings"), (100, "AcDbLayout"), (1, name),
                   (70, "1"), (71, str(index)), (10, "0"), (20, "0"), (11, "420"), (21, "297"),
                   (12, "0"), (22, "0"), (32, "0"), (146, "0"), (13, "0"), (23, "0"), (33, "0"),
                   (16, "1"), (26, "0"), (36, "0"), (17, "0"), (27, "1"), (37, "0"),
                   (76, "0"), (330, record)]
    return result


def _format(value):
    return format(value, ".17g")


def _entity_pairs(row, handle):
    kind = row["type"]
    result = [(0, kind), (5, format(handle, "X")), (330, "A"), (100, "AcDbEntity"), (8, row["layer"]),
              (100, "AcDbLine" if kind == "LINE" else "AcDbCircle")]
    if kind == "LINE":
        for codes, point in (((10, 20, 30), row["start"]), ((11, 21, 31), row["end"])):
            result += list(zip(codes, [_format(point[0]), _format(point[1]), "0"]))
    else:
        result += [(10, _format(row["center"][0])), (20, _format(row["center"][1])), (30, "0"),
                   (40, _format(row["radius_mm"]))]
        if kind == "ARC":
            start = math.degrees(row["start_angle_rad"]) % 360.
            end = math.degrees(row["start_angle_rad"] + row["sweep_rad"]) % 360.
            # A value immediately below360 can round onto360 at the declared
            # decimal grid. Canonicalize AFTER that rounding, not just before.
            start_text = format(float(format(start, ".12f")) % 360., ".12f")
            end_text = format(float(format(end, ".12f")) % 360., ".12f")
            _require(start_text != end_text, "UNREPRESENTABLE_PARTIAL_ARC")
            result += [(100, "AcDbArc"), (50, start_text), (51, end_text)]
    return result + [(210, "0"), (220, "0"), (230, "1")]


def _parse_entity(pairs, handles):
    kind = pairs[0][1]
    _require(kind in {"LINE", "ARC", "CIRCLE"} and len(pairs) >= 6, "UNSUPPORTED_DXF_ENTITY")
    _require(pairs[1][0] == 5 and pairs[2] == (330, "A") and pairs[3] == (100, "AcDbEntity") and pairs[4][0] == 8
             and pairs[4][1] in LAYERS and pairs[5] == (100, "AcDbLine" if kind == "LINE" else "AcDbCircle"),
             "INVALID_DXF_ENTITY_HEADER")
    try:
        handle = int(pairs[1][1], 16)
    except ValueError as error:
        raise AnalyticDxfUnavailable("INVALID_DXF_HANDLE") from error
    _require(256 <= handle < 65536 and handle not in handles, "DUPLICATE_OR_INVALID_DXF_HANDLE")
    handles.add(handle)
    tail = pairs[6:]
    codes = [10, 20, 30, 11, 21, 31] if kind == "LINE" else [10, 20, 30, 40]
    if kind == "ARC":
        codes += [100, 50, 51]
    codes += [210, 220, 230]
    _require([code for code, _ in tail] == codes, "UNEXPECTED_OR_DUPLICATE_DXF_GROUP")
    fields = dict(tail)
    _require(fields[30] == "0" and (kind != "LINE" or fields[31] == "0")
             and [fields[v] for v in (210, 220, 230)] == ["0", "0", "1"], "NONPLANAR_DXF_ENTITY")
    def number(code):
        try:
            return _number(float(fields[code]))
        except (TypeError, ValueError, OverflowError) as error:
            raise AnalyticDxfUnavailable("INVALID_DXF_NUMBER") from error
    row = {"type": kind, "layer": pairs[4][1]}
    if kind == "LINE":
        row.update(start=[number(10), number(20)], end=[number(11), number(21)])
    else:
        row.update(center=[number(10), number(20)], radius_mm=number(40))
        if kind == "CIRCLE":
            row.update(sweep_rad=_TAU, full_circle_verified=True)
        else:
            _require(fields[100] == "AcDbArc", "INVALID_DXF_ARC_SUBCLASS")
            start, end = number(50), number(51)
            _require(0 <= start < 360 and 0 <= end < 360 and start != end, "INVALID_DXF_ARC_ANGLES")
            row.update(start_angle_rad=math.radians(start), sweep_rad=math.radians((end - start) % 360.))
    return row


def read_analytic_dxf(text):
    """Strict parser for this minimal schema, not a general DXF importer."""
    _require(isinstance(text, str) and text.isascii() and len(text) <= MAX_DXF_BYTES, "DXF_TEXT_BUDGET_OR_ENCODING")
    lines = text.splitlines()
    _require(len(lines) % 2 == 0 and all(len(line) <= 128 for line in lines), "INVALID_DXF_PAIR_STREAM")
    try:
        pairs = [(int(lines[i]), lines[i + 1]) for i in range(0, len(lines), 2)]
    except ValueError as error:
        raise AnalyticDxfUnavailable("INVALID_DXF_GROUP_CODE") from error
    sections, index = {}, 0
    for name in ("HEADER", "TABLES", "BLOCKS", "ENTITIES", "OBJECTS"):
        _require(pairs[index:index + 2] == [(0, "SECTION"), (2, name)], "UNEXPECTED_DXF_SECTION")
        index += 2
        start = index
        while index < len(pairs) and pairs[index] != (0, "ENDSEC"):
            index += 1
        _require(index < len(pairs), "UNTERMINATED_DXF_SECTION")
        sections[name] = pairs[start:index]
        index += 1
    _require(pairs[index:] == [(0, "EOF")], "TRAILING_OR_MISSING_DXF_EOF")
    _require(sections["HEADER"] == _header(), "DXF_UNITS_OR_HEADER_MISMATCH")
    _require(sections["TABLES"] == _tables(), "DXF_LAYER_TABLE_MISMATCH")
    _require(sections["BLOCKS"] == _blocks() and sections["OBJECTS"] == _objects(),
             "DXF_MODELSPACE_OWNERSHIP_MISMATCH")
    entities, handles, body, index = [], set(), sections["ENTITIES"], 0
    while index < len(body):
        _require(body[index][0] == 0 and len(entities) < MAX_ENTITIES, "DXF_ENTITY_POPULATION_OR_BUDGET")
        start = index
        index += 1
        while index < len(body) and body[index][0] != 0:
            index += 1
        entities.append(_parse_entity(body[start:index], handles))
    entities = _normalize(entities)
    loops = _validate_cut_loops(entities)
    return {"units": "mm", "rows": entities, "layers": list(LAYERS),
            **loops, "release_allowed": False, "manufacturing_ready": False}


def _encode(rows):
    pairs = [(0, "SECTION"), (2, "HEADER")] + _header() + [(0, "ENDSEC"),
             (0, "SECTION"), (2, "TABLES")] + _tables() + [(0, "ENDSEC"), (0, "SECTION"), (2, "BLOCKS")]
    pairs += _blocks() + [(0, "ENDSEC"), (0, "SECTION"), (2, "ENTITIES")]
    for handle, row in enumerate(rows, 256):
        pairs += _entity_pairs(row, handle)
    pairs += [(0, "ENDSEC"), (0, "SECTION"), (2, "OBJECTS")] + _objects() + [(0, "ENDSEC"), (0, "EOF")]
    text = "".join(f"{code}\n{value}\n" for code, value in pairs)
    _require(len(text) <= MAX_DXF_BYTES, "DXF_TEXT_BUDGET_EXCEEDED")
    return text


def build_analytic_dxf(rows, *, units):
    """Serialize analytic loci only, with explicit declared millimetre units."""
    _require(units == "mm", "EXPLICIT_MM_DXF_UNITS_REQUIRED")
    expected = _normalize(rows)
    loops = _validate_cut_loops(expected)
    text = _encode(expected)
    parsed = read_analytic_dxf(text)
    actual = parsed["rows"]
    _require(_encode(actual) == text, "NONCANONICAL_DXF_ROUNDTRIP")
    _require(len(actual) == len(expected), "DXF_ROUNDTRIP_ENTITY_COUNT")
    errors = []
    for left, right in zip(expected, actual):
        _require(left["type"] == right["type"] and left["layer"] == right["layer"], "DXF_ROUNDTRIP_ENTITY_OR_LAYER")
        if left["type"] == "LINE":
            error = max(math.dist(left[key], right[key]) for key in ("start", "end"))
        else:
            error = math.dist(left["center"], right["center"]) + abs(left["radius_mm"] - right["radius_mm"])
            if left["type"] == "ARC":
                phase_error = abs(math.remainder(left["start_angle_rad"] - right["start_angle_rad"], _TAU))
                error += max(left["radius_mm"], right["radius_mm"]) * (
                    phase_error + abs(left["sweep_rad"] - right["sweep_rad"]))
        _require(math.isfinite(error) and error <= LINEAR_TOL_MM, "DXF_ROUNDTRIP_GEOMETRY_MISMATCH")
        errors.append(error)
    _require(abs(loops["analytic_cut_area_mm2"] - parsed["analytic_cut_area_mm2"])
             <= loops["analytic_cut_area_numerical_tolerance_mm2"], "DXF_ROUNDTRIP_CUT_AREA_MISMATCH")
    counts = {layer: {kind: sum(row["layer"] == layer and row["type"] == kind for row in expected)
                      for kind in ("LINE", "ARC", "CIRCLE")} for layer in LAYERS}
    return {"status": "DRAFT_DXF_BUILT", "text": text, "entity_counts": counts,
            "entity_count": len(expected), "layers": list(LAYERS), "units": "mm", "units_verified": False,
            "roundtrip_checked": True, "maximum_roundtrip_error_mm": max(errors),
            "canonical_parse_reencode_checked": True, "r2000_modelspace_ownership_checked": True,
            "external_importer_compatibility_verified": False,
            "cut_endpoint_incidence_checked": True, **loops,
            "contour_topology_requires_native_proof": True, "source_and_profile_requires_caller_binding": True,
            "release_allowed": False, "manufacturing_ready": False}
