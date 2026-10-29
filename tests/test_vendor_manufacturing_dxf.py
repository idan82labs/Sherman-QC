"""Pure analytic/synthetic DXF checks; no native imports, files or preferences."""
import copy
import math

import pytest

from backend.vendor_manufacturing_dxf import (
    AnalyticDxfUnavailable, build_analytic_dxf, read_analytic_dxf,
)


def line(layer="CUT_OUTER"):
    return {"type": "LINE", "layer": layer, "start": [-2., 3.], "end": [7., 11.]}


def arc(start=.3, sweep=1.2, radius=7., layer="CUT_OUTER"):
    return {"type": "ARC", "layer": layer, "center": [4., -3.], "radius_mm": radius,
            "start_angle_rad": start, "sweep_rad": sweep}


def circle():
    return {"type": "CIRCLE", "layer": "CUT_INNER", "center": [8., 9.],
            "radius_mm": 1.7, "sweep_rad": 2 * math.pi, "full_circle_verified": True}


def outer():
    corners = [[-20., -20.], [20., -20.], [20., 20.], [-20., 20.]]
    return [{"type": "LINE", "layer": "CUT_OUTER", "start": a, "end": b}
            for a, b in zip(corners, corners[1:] + corners[:1])]


def complete_cut(rows):
    """Fresh complete synthetic contours, never shrink or repair production input."""
    result = copy.deepcopy(rows)
    cuts = [row for row in result if row.get("layer") == "CUT_OUTER"]
    if not cuts:
        result += outer()
    elif len(cuts) == 1 and cuts[0].get("type") == "LINE":
        row = cuts[0]
        tip = [-8., 12.]
        result += [{"type": "LINE", "layer": "CUT_OUTER", "start": row["end"], "end": tip},
                   {"type": "LINE", "layer": "CUT_OUTER", "start": tip, "end": row["start"]}]
    elif len(cuts) == 1 and cuts[0].get("type") == "ARC":
        row = cuts[0]
        points = [[row["center"][0] + row["radius_mm"] * math.cos(angle),
                   row["center"][1] + row["radius_mm"] * math.sin(angle)]
                  for angle in (row["start_angle_rad"], row["start_angle_rad"] + row["sweep_rad"])]
        result += [{"type": "LINE", "layer": "CUT_OUTER", "start": points[1], "end": points[0]}]
    return result


def build(rows, units="mm"):
    return build_analytic_dxf(complete_cut(rows), units=units)


def reference_entities(text):
    """Independent test-only group-pair reader; no production parser calls."""
    values = text.splitlines()
    pairs = [(int(values[i]), values[i + 1]) for i in range(0, len(values), 2)]
    start = next(i for i in range(len(pairs) - 1) if pairs[i:i + 2] == [(0, "SECTION"), (2, "ENTITIES")]) + 2
    result, current = [], None
    for code, value in pairs[start:]:
        if code == 0:
            if current is not None:
                result.append(current)
            if value == "ENDSEC":
                break
            current = {0: value}
        elif code != 100:
            assert code not in current
            current[code] = value
    return result


def test_layers_units_analytic_entities_determinism_and_input_immutable():
    rows = outer() + [circle(), line("BEND_UP"), line("BEND_DOWN")]
    before = copy.deepcopy(rows)
    result = build_analytic_dxf(rows, units="mm")
    assert rows == before and result == build_analytic_dxf(rows, units="mm")
    assert result["status"] == "DRAFT_DXF_BUILT" and result["roundtrip_checked"]
    assert result["canonical_parse_reencode_checked"] and result["r2000_modelspace_ownership_checked"]
    assert result["external_importer_compatibility_verified"] is False
    assert result["release_allowed"] is False and result["manufacturing_ready"] is False
    assert result["units_verified"] is False and result["entity_count"] == len(rows)
    assert result["cut_loop_counts"] == {"CUT_OUTER": 1, "CUT_INNER": 1}
    assert abs(result["analytic_cut_area_mm2"] - (1600. - math.pi * 1.7 ** 2)) < 1e-11
    assert "9\n$INSUNITS\n70\n4\n" in result["text"]
    entities = reference_entities(result["text"])
    assert [(e[0], e[8]) for e in entities] == [(row["type"], row["layer"]) for row in rows]
    assert all([e[k] for k in (210, 220, 230)] == ["0", "0", "1"] for e in entities)


@pytest.mark.parametrize("start", [0., .3, 3., -5., 2 * math.pi - .1])
@pytest.mark.parametrize("sweep", [.4, -.4, 4.5, -4.5])
def test_signed_and_wrapped_arc_locus_and_independent_sector_area(start, sweep):
    source = arc(start, sweep)
    result = build([source])
    assert build_analytic_dxf(read_analytic_dxf(result["text"])["rows"], units="mm")["text"] == result["text"]
    dxf = reference_entities(result["text"])[0]
    first, last = math.radians(float(dxf[50])), math.radians(float(dxf[51]))
    actual_sweep = (last - first) % (2 * math.pi)
    expected_first = start if sweep > 0 else start + sweep
    assert abs(actual_sweep - abs(sweep)) < 2e-14
    # Native negative traversal is reversed, but the complete analytic locus is
    # unchanged. Endpoints and midpoint are checked independently of serializer.
    for fraction in (0., .5, 1.):
        actual = first + actual_sweep * fraction
        expected = expected_first + abs(sweep) * fraction
        assert math.dist([math.cos(actual), math.sin(actual)], [math.cos(expected), math.sin(expected)]) * 7 < 1e-11
    assert abs(.5 * float(dxf[40]) ** 2 * actual_sweep - .5 * 7 ** 2 * abs(sweep)) < 1e-10


def test_nearly_closed_arc_is_not_promoted_to_circle():
    source = arc(.2, 2 * math.pi - .02, 1.)
    result = build([source])
    entity = reference_entities(result["text"])[0]
    assert entity[0] == "ARC" and result["entity_counts"]["CUT_OUTER"]["CIRCLE"] == 0
    gap = 2 * math.pi - math.radians((float(entity[51]) - float(entity[50])) % 360.)
    # Two12-decimal-degree values contribute at most one degree quantum plus
    # floating-point radian conversion; this is far below the physical budget.
    assert abs(gap - .02) < 3e-14


@pytest.mark.parametrize("start", [2 * math.pi - 1e-15, 2 * math.pi - 3e-15,
                                  2 * math.pi, -1e-15, -3e-15])
@pytest.mark.parametrize("sweep", [.7, -.7])
def test_decimal_angle_rounding_near_full_phase_wrap_is_canonical(start, sweep):
    result = build([arc(start, sweep)])
    entity = reference_entities(result["text"])[0]
    assert 0 <= float(entity[50]) < 360 and 0 <= float(entity[51]) < 360
    parsed = read_analytic_dxf(result["text"])
    assert build_analytic_dxf(parsed["rows"], units="mm")["text"] == result["text"]
    assert abs(parsed["analytic_cut_area_mm2"] - 7 ** 2 / 2 * (.7 - math.sin(.7))) < 1e-11


@pytest.mark.parametrize("delta", [-1e-12, -1e-6, 1e-12])
def test_circle_requires_exact_full_turn_not_endpoint_closeness(delta):
    row = circle(); row["sweep_rad"] += delta
    with pytest.raises(AnalyticDxfUnavailable):
        build_analytic_dxf([row], units="mm")


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_circle_requires_explicit_boolean_certificate(value):
    row = circle(); row["full_circle_verified"] = value
    with pytest.raises(AnalyticDxfUnavailable):
        build_analytic_dxf([row], units="mm")


@pytest.mark.parametrize("change", [{"sweep_rad": 0.}, {"sweep_rad": 2 * math.pi},
    {"sweep_rad": 2 * math.pi - 1e-15}, {"sweep_rad": 1e-16}, {"start_angle_rad": 1e12},
    {"radius_mm": math.inf}, {"radius_mm": True}, {"radius_mm": 10 ** 400},
    {"center": [0., math.nan]}, {"layer": "BEND_UP"}, {"type": "SPLINE"}, {"extra": 1}])
def test_unsupported_nonfinite_degenerate_and_ill_conditioned_rows(change):
    row = arc(); row.update(change)
    with pytest.raises(AnalyticDxfUnavailable):
        build_analytic_dxf([row], units="mm")


@pytest.mark.parametrize("units", [None, "inch", "assumed_mm", 4])
def test_units_are_explicit_mm_only(units):
    with pytest.raises(AnalyticDxfUnavailable):
        build_analytic_dxf([line()], units=units)


def test_population_and_duplicate_or_unknown_fields_are_refused():
    for rows in ([], [line()] * 8193, [{**line(), "source_id": "unbound"}],
                 [{**line(), "end": [-2., 3.]}], [{**line(), "layer": "CUT_OUTER\n0\nTEXT"}]):
        with pytest.raises(AnalyticDxfUnavailable):
            build_analytic_dxf(rows, units="mm")


def test_open_duplicate_degenerate_and_disconnected_cut_contours_refuse():
    segment = line()
    opposite = {**segment, "start": segment["end"], "end": segment["start"]}
    shifted = [{**row, "start": [v + 100. for v in row["start"]],
                "end": [v + 100. for v in row["end"]]} for row in outer()]
    for rows in ([segment], [segment, opposite], outer() + [outer()[0]], outer() + shifted):
        with pytest.raises(AnalyticDxfUnavailable):
            build_analytic_dxf(rows, units="mm")


def test_line_loop_signed_area_is_independent_of_traversal_and_order():
    rows = outer()
    for reversed_rows in (list(reversed(rows)), [{**row, "start": row["end"], "end": row["start"]} for row in rows]):
        result = build_analytic_dxf(reversed_rows, units="mm")
        assert result["analytic_cut_area_mm2"] == 1600.


def test_arc_chord_segment_area_matches_independent_formula():
    theta, radius = 4.5, 7.
    result = build([arc(.3, -theta, radius)])
    assert abs(result["analytic_cut_area_mm2"] - radius ** 2 / 2 * (theta - math.sin(theta))) < 1e-11


def test_independent_r2000_handles_owners_tables_blocks_and_layout_structure():
    text = build_analytic_dxf(outer(), units="mm")["text"]
    values = text.splitlines()
    pairs = [(int(values[i]), values[i + 1]) for i in range(0, len(values), 2)]
    # Ignore the HEADER allocator ($HANDSEED), which is not an object handle.
    begin = next(i for i in range(len(pairs) - 1) if pairs[i:i + 2] == [(0, "SECTION"), (2, "TABLES")])
    handles = [value for code, value in pairs[begin:] if code == 5]
    assert len(handles) == len(set(handles))
    assert all(value == "0" or value in handles for code, value in pairs[begin:] if code in (330, 340, 350))
    assert sum(pair == (0, "BLOCK_RECORD") for pair in pairs) == 3
    assert sum(pair == (0, "BLOCK") for pair in pairs) == 3
    assert sum(pair == (0, "ENDBLK") for pair in pairs) == 3
    assert sum(pair == (0, "LAYOUT") for pair in pairs) == 3
    for entity in reference_entities(text):
        assert entity[330] == "A"


@pytest.mark.parametrize("change", [
    lambda text: text.replace("330\nA\n100\nAcDbEntity", "330\nB\n100\nAcDbEntity", 1),
    lambda text: text.replace("5\n1\n330\n0\n100\nAcDbSymbolTable", "5\n1\n330\n0", 1),
    lambda text: text.replace("340\n20", "340\nFF", 1),
    lambda text: text.replace("2\nBLOCKS", "2\nBADBLOCKS", 1),
])
def test_ownership_or_skeleton_tampering_refuses(change):
    text = build_analytic_dxf(outer(), units="mm")["text"]
    with pytest.raises(AnalyticDxfUnavailable):
        read_analytic_dxf(change(text))


@pytest.mark.parametrize("mutation", [
    lambda text: text.replace("$INSUNITS\n70\n4", "$INSUNITS\n70\n1"),
    lambda text: text.replace("8\nCUT_OUTER", "8\nBEND_UNKNOWN"),
    lambda text: text.replace("10\n-2", "10\nNaN"),
    lambda text: text.replace("230\n1", "230\n-1"),
    lambda text: text.replace("0\nEOF\n", "0\nEOF\n0\nTEXT\n"),
    lambda text: text.replace("11\n7", "11\n7\n11\n7"),
])
def test_strict_reader_rejects_unit_entity_geometry_and_pair_tampering(mutation):
    text = build([line()])["text"]
    with pytest.raises(AnalyticDxfUnavailable):
        read_analytic_dxf(mutation(text))
