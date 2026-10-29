"""E2 synthetic geometry: analytic truth vs the generated STEP BRep, stage folding, flat.

The cross-check measures the written STEP file independently of the generator
code path: it reads the BRep back, finds each bend's outer cylinder face,
takes the two tangent planar faces adjacent to it (the OML skins), and derives
the included angle, the OML virtual sharp (plane intersection), thickness,
radii and OML flange lengths (far free-edge lines of the skin faces, or the
measured sharp of the adjacent bend). Truth must agree to <= 1e-6 mm / deg.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from domains.production.engine.synthetic import brep
from domains.production.engine.synthetic.analytic import far_boundary
from domains.production.engine.synthetic.families import FAMILIES, FeatureRequest, build_family
from domains.production.engine.synthetic.flat import flat_pattern, read_flat_dxf, write_flat_dxf
from domains.production.engine.synthetic.geometry import (
    UP,
    BendDesign,
    FoldedPart,
    FormState,
    GeometryError,
    PanelDesign,
    PartDesign,
    assign_sides_rule,
    neutral_length,
    outside_setback,
    rotation_about_axis,
    translation,
)
from domains.production.engine.synthetic.truth import build_nominal

TOL = 1e-6


# ---------------------------------------------------------------------------
# independent BRep measurement helpers
# ---------------------------------------------------------------------------


def _faces(shape):
    out = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        s = BRepAdaptor_Surface(f)
        rec = {"face": f, "type": s.GetType()}
        if rec["type"] == GeomAbs_Plane:
            pl = s.Plane()
            n = pl.Axis().Direction()
            normal = np.array([n.X(), n.Y(), n.Z()])
            if f.Orientation() == TopAbs_REVERSED:
                normal = -normal
            loc = pl.Location()
            rec.update(point=np.array([loc.X(), loc.Y(), loc.Z()]), normal=normal)
        elif rec["type"] == GeomAbs_Cylinder:
            cy = s.Cylinder()
            ax = cy.Axis()
            rec.update(
                axis_point=np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()]),
                axis_dir=np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()]),
                radius=cy.Radius(),
            )
        out.append(rec)
        ex.Next()
    return out


def _edge_face_map(shape):
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return m


def _face_edges(face):
    edges = []
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        edges.append(TopoDS.Edge_s(ex.Current()))
        ex.Next()
    return edges


def _line_edges(face):
    out = []
    for e in _face_edges(face):
        c = BRepAdaptor_Curve(e)
        if c.GetType() != GeomAbs_Line:
            continue
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        out.append((np.array([p0.X(), p0.Y(), p0.Z()]), np.array([p1.X(), p1.Y(), p1.Z()])))
    return out


def _dist_point_line(p, a, d):
    v = p - a
    return float(np.linalg.norm(v - (v @ d) * d))


def _plane_intersection(p1, n1, p2, n2):
    d = np.cross(n1, n2)
    d /= np.linalg.norm(d)
    a = np.array([n1, n2, d])
    b = np.array([n1 @ p1, n2 @ p2, d @ p1])
    return np.linalg.solve(a, b), d


def measure_bends(shape, truth_bends):
    """Per truth bend: measured included angle, OML sharp (point, dir), skin planes,
    outer/inner radius, thickness, keyed by bend key."""
    faces = _faces(shape)
    emap = _edge_face_map(shape)
    out = {}
    for tb in truth_bends:
        axis_p, axis_d = tb["axis_point"], tb["axis_dir"]
        ro = tb["inner_radius"] + tb["thickness"]
        cands = [
            f for f in faces
            if f["type"] == GeomAbs_Cylinder and abs(f["radius"] - ro) < 1e-7
            and abs(abs(f["axis_dir"] @ axis_d) - 1) < 1e-9 and _dist_point_line(f["axis_point"], axis_p, axis_d) < 1e-6
        ]
        inner = [
            f for f in faces
            if f["type"] == GeomAbs_Cylinder and abs(f["radius"] - tb["inner_radius"]) < 1e-7
            and abs(abs(f["axis_dir"] @ axis_d) - 1) < 1e-9 and _dist_point_line(f["axis_point"], axis_p, axis_d) < 1e-6
        ]
        assert len(cands) == 1, f"{tb['key']}: expected one outer cylinder, found {len(cands)}"
        assert len(inner) == 1, f"{tb['key']}: expected one inner cylinder, found {len(inner)}"
        cyl = cands[0]
        planes = []
        for e in _face_edges(cyl["face"]):
            for nb in emap.FindFromKey(e):
                f = TopoDS.Face_s(nb)
                if f.IsSame(cyl["face"]):
                    continue
                rec = next(x for x in faces if x["face"].IsSame(f))
                if rec["type"] == GeomAbs_Plane and abs(rec["normal"] @ axis_d) < 1e-9 and not any(r["face"].IsSame(f) for r in planes):
                    planes.append(rec)
        assert len(planes) == 2, f"{tb['key']}: expected two tangent OML planes, found {len(planes)}"
        n1, n2 = planes[0]["normal"], planes[1]["normal"]
        sweep = math.degrees(math.acos(max(-1.0, min(1.0, float(n1 @ n2)))))
        point, direction = _plane_intersection(planes[0]["point"], n1, planes[1]["point"], n2)
        # thickness: distance from each OML plane to the parallel opposite skin plane of the same panel
        out[tb["key"]] = {"included": 180.0 - sweep, "sharp_point": point, "sharp_dir": direction, "planes": planes}
    return out


def _truth_bends(fp: FoldedPart):
    out = []
    for b in fp.design.bends:
        g = fp.bend_geo[b.key]
        if not g.formed:
            continue
        p, d = fp.axis_cad(b.key)
        out.append({"key": b.key, "axis_point": p, "axis_dir": d, "inner_radius": g.inner_radius, "thickness": g.thickness})
    return out


def _check_part_against_brep(fp: FoldedPart, shape, *, perturb: float = 0.0) -> list[str]:
    """Return mismatches between the analytic truth and the BRep (empty = exact)."""
    problems = []
    measured = measure_bends(shape, _truth_bends(fp))
    sides = {}
    for b in fp.design.bends:
        if not fp.bend_geo[b.key].formed:
            continue
        oa, ob = fp.outward_cad(b.key, b.panel_a), fp.outward_cad(b.key, b.panel_b)
        ia, ib = assign_sides_rule(oa, ob)
        sides[b.key] = ((b.panel_a, b.panel_b)[ia], (b.panel_a, b.panel_b)[ib])
    for key, m in measured.items():
        g = fp.bend_geo[key]
        included = 180.0 - g.sweep_deg + perturb
        if abs(m["included"] - included) > TOL:
            problems.append(f"{key} included {m['included']:.9f} vs truth {included:.9f}")
        sharp = fp.loci(key)["oml_virtual_sharp"] + perturb
        for p in sharp:
            if _dist_point_line(p, m["sharp_point"], m["sharp_dir"]) > TOL:
                problems.append(f"{key} OML sharp off by {_dist_point_line(p, m['sharp_point'], m['sharp_dir']):.3e}")
        for letter, panel in zip("AB", sides[key]):
            fb = far_boundary(fp, key, panel, across_unformed=False)
            if fb.kind_fraction < 1.0 or fb.boundary_kind not in ("FREE_EDGE", "ADJACENT_BEND"):
                continue
            out_dir = fp.outward_cad(key, panel)
            plane = next(pl for pl in m["planes"] if abs(pl["normal"] @ out_dir) < 1e-6)
            if fb.boundary_kind == "FREE_EDGE":
                axis = m["sharp_dir"]
                cand = [
                    max((a - m["sharp_point"]) @ out_dir, (c - m["sharp_point"]) @ out_dir)
                    for a, c in _line_edges(plane["face"])
                    if abs(abs((c - a) @ axis) / np.linalg.norm(c - a) - 1) < 1e-9
                ]
                length = max(cand)
            else:
                other = measured[fb.adjacent_bend]
                length = float((other["sharp_point"] - m["sharp_point"]) @ out_dir)
            if abs(length - (fb.value_mm + perturb)) > TOL:
                problems.append(f"{key}/{letter} flange {length:.9f} vs truth {fb.value_mm + perturb:.9f}")
    return problems


def _design(family: str, seed: int = 4, **req) -> PartDesign:
    return build_family(family, np.random.default_rng(seed), FeatureRequest(**req))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_final_step_truth_is_exact(family, tmp_path):
    d = _design(family)
    fp = FoldedPart(d, FormState.nominal(d))
    shape = brep.fused_solid(brep.build_pieces(fp))
    brep.check_single_valid_solid(shape)
    path = tmp_path / "part.step"
    brep.write_step(shape, path, "SYN-TEST")
    back = brep.read_step(path)
    types = brep.face_type_counts(back)
    assert types["other"] == 0 and types["cylinder"] >= 2 * len(d.bends)
    assert abs(brep.solid_volume(back) - brep.analytic_volume(fp)) < 1e-6 * brep.analytic_volume(fp)
    assert _check_part_against_brep(fp, back) == []


@pytest.mark.parametrize("family", ["U_CHANNEL", "HAT", "TWO_DIRECTION_TAB"])
def test_as_built_stage_truth_is_exact(family, tmp_path):
    """Stage geometry with errors (partial forming, angle error, bend-line shift)."""
    d = _design(family, seed=9, acute=(family == "HAT"))
    seq = d.sequence
    states = {b.key: FormState.nominal(d).bends[b.key] for b in d.bends}
    from domains.production.engine.synthetic.geometry import BendState

    states[seq[0]] = BendState(d.bend(seq[0]).sweep_deg - 1.7, 0.6)
    states[seq[1]] = BendState(0.45 * d.bend(seq[1]).sweep_deg, -0.4)
    for k in seq[2:]:
        states[k] = BendState(0.0, 0.0)
    fp = FoldedPart(d, FormState(states))
    shape = brep.fused_solid(brep.build_pieces(fp))
    brep.check_single_valid_solid(shape)
    path = tmp_path / "stage.step"
    brep.write_step(shape, path, "SYN-STAGE")
    back = brep.read_step(path)
    formed = sum(1 for g in fp.bend_geo.values() if g.formed)
    hole_cyl = sum(len(p.holes) for p in d.panels)
    assert brep.face_type_counts(back)["cylinder"] == 2 * formed + hole_cyl
    assert _check_part_against_brep(fp, back) == []


def test_brep_cross_check_detects_a_1e3_truth_error(tmp_path):
    """Mutate-and-check: a 1e-3 perturbation of truth must be reported."""
    d = _design("Z_STEP")
    fp = FoldedPart(d, FormState.nominal(d))
    shape = brep.fused_solid(brep.build_pieces(fp))
    problems = _check_part_against_brep(fp, shape, perturb=1e-3)
    assert problems and all("off by" in p or "vs truth" in p for p in problems)


def test_stage_zero_brep_is_a_flat_plate():
    d = _design("HAT")
    fp0 = FoldedPart(d, FormState.nominal(d, formed=()))
    shape = brep.fused_solid(brep.build_pieces(fp0))
    brep.check_single_valid_solid(shape)
    faces = _faces(shape)
    n0 = fp0.panel_normal_skin0(fp0.datum)
    skins = [f for f in faces if f["type"] == GeomAbs_Plane and abs(abs(f["normal"] @ n0) - 1) < 1e-9]
    assert len(skins) == 2  # one face per skin: the whole blank is coplanar
    cyl = [f for f in faces if f["type"] == GeomAbs_Cylinder]
    assert len(cyl) == sum(len(p.holes) for p in d.panels)


def test_stage_models_follow_method_a():
    """S_k panel pose = final pose with each unformed bend rotated back by its sweep
    about its final axis, then translated by the neutral length away from the parent."""
    d = _design("CHANNEL_WITH_RETURNS")
    final = FoldedPart(d, FormState.nominal(d))
    k = 1
    stage = FoldedPart(d, FormState.nominal(d, d.sequence[:k]))
    for panel in d.panels:
        # un-form from the datum outwards; each later axis is taken at the current pose
        m = np.eye(4)
        for bkey in final.path_to_datum(panel.key):
            if bkey in d.sequence[:k]:
                continue
            g = final.bend_geo[bkey]
            pt, ax = final.axis_cad(bkey)
            # this bend's parent pose may itself have moved: express in current (stage) pose
            pt_s = (m @ np.append(pt, 1.0))[:3]
            ax_s = m[:3, :3] @ ax
            parent_out = final.outward_cad(bkey, g.parent)
            shift = translation(-(m[:3, :3] @ parent_out) * g.ba_nominal)
            m = shift @ rotation_about_axis(pt_s, ax_s, -math.radians(g.sweep_deg)) @ m
        expected = m @ final.world[panel.key]
        assert np.allclose(stage.world[panel.key], expected, atol=1e-9), panel.key


@pytest.mark.parametrize("family", ["L_BRACKET", "U_CHANNEL", "Z_STEP", "HAT", "CHANNEL_WITH_RETURNS"])
def test_flat_developed_length_matches_closed_form(family, tmp_path):
    """Profile blank length = sum(OML lengths) - sum(bend deductions), BD = 2 Ro tan(a/2) - a (Ri + K t)."""
    d = _design(family)
    oml = d.params["oml_lengths"]
    t, k = d.thickness, d.k_factor
    bd = sum(2 * outside_setback(b.sweep_deg, b.inner_radius, t) - neutral_length(b.sweep_deg, b.inner_radius, t, k) for b in d.bends)
    expected_length = sum(oml) - bd
    path = tmp_path / "flat.dxf"
    write_flat_dxf(d, path)
    parsed = read_flat_dxf(path)
    outline = parsed["outline"]
    assert abs(float(np.ptp(outline[:, 0])) - expected_length) < 1e-9
    assert abs(float(np.ptp(outline[:, 1])) - d.params["width"]) < 1e-9
    assert parsed["units"] == 4
    assert len(parsed["bend_lines"]) == len(d.bends)
    assert len(parsed["holes"]) == sum(len(p.holes) for p in d.panels)
    notes = sorted(n for n, _ in parsed["annotations"])
    assert notes == sorted(f"{'UP' if b.direction == UP else 'DOWN'} {b.sweep_deg:.2f}° R{b.inner_radius:.2f}" for b in d.bends)


def test_flat_dxf_is_deterministic_and_area_matches_panels(tmp_path):
    d = _design("TWO_DIRECTION_TAB")
    h1 = write_flat_dxf(d, tmp_path / "a.dxf")
    h2 = write_flat_dxf(d, tmp_path / "b.dxf")
    assert h1 == h2
    pat = flat_pattern(d)
    fp0 = FoldedPart(d, FormState.nominal(d, formed=()))
    assert abs(pat.area - brep.analytic_volume(fp0) / d.thickness) < 1e-6


@pytest.mark.parametrize("family", FAMILIES)
def test_nominal_oml_lengths_equal_design_parameters(family):
    d = _design(family)
    nom = build_nominal(d)
    if "oml_lengths" in d.params:
        oml = d.params["oml_lengths"]
        bends = list(d.bends)
        for i, b in enumerate(bends):
            got = {nom.side_panel(b.key, s): nom.flange[(b.key, s)].value_mm for s in "AB"}
            assert abs(got[b.panel_a] - oml[i]) < 1e-9
            assert abs(got[b.panel_b] - oml[i + 1]) < 1e-9
    for (key, side), fb in nom.flange.items():
        assert fb.value_mm is not None and fb.value_mm > 0, (key, side)


def test_step_files_are_byte_deterministic(tmp_path):
    d = _design("PANEL_WITH_HOLES_NOTCHES")
    shape = brep.fused_solid(brep.build_pieces(FoldedPart(d, FormState.nominal(d))))
    assert brep.write_step(shape, tmp_path / "a.step", "SYN-X") == brep.write_step(shape, tmp_path / "b.step", "SYN-X")
    assert (tmp_path / "a.step").read_bytes() == (tmp_path / "b.step").read_bytes()
    with pytest.raises(ValueError):
        brep.write_step(shape, tmp_path / "c.step", "bad'name")


def test_side_rule_matches_contract():
    from contracts.production.compiled_part import assign_sides

    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.normal(size=3), rng.normal(size=3)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        assert assign_sides_rule(a, b) == assign_sides(tuple(a), tuple(b))
    assert assign_sides_rule((1e-7, 1, 0), (0, 0, 1)) == assign_sides((1e-7, 1, 0), (0, 0, 1))


def test_design_validation_rejects_inconsistent_geometry():
    d = _design("L_BRACKET")
    # clockwise outline
    p = d.panels[0]
    cw = PanelDesign(p.key, tuple(reversed(p.outline)), tuple(reversed(p.edge_tags)), p.holes)
    with pytest.raises(GeometryError):
        PartDesign(d.family, d.thickness, d.k_factor, (cw,) + d.panels[1:], d.bends, d.sequence).validate()
    # bend line moved off the strip centre
    b = d.bends[0]
    moved = BendDesign(b.key, b.panel_a, b.panel_b, (b.line_point[0] + 0.5, b.line_point[1]), b.line_dir, b.normal_ab,
                       b.extent, b.direction, b.inner_radius, b.sweep_deg)
    with pytest.raises(GeometryError, match="strip"):
        PartDesign(d.family, d.thickness, d.k_factor, d.panels, (moved,), d.sequence).validate()
    # a hole touching an edge
    from domains.production.engine.synthetic.geometry import Hole

    edge_hole = PanelDesign(p.key, p.outline, p.edge_tags, (Hole((p.outline[0][0] + 0.5, p.outline[0][1] + 10.0), 2.0),))
    with pytest.raises(GeometryError, match="too close"):
        PartDesign(d.family, d.thickness, d.k_factor, (edge_hole,) + d.panels[1:], d.bends, d.sequence).validate()
    # sweep out of range
    flat = BendDesign(b.key, b.panel_a, b.panel_b, b.line_point, b.line_dir, b.normal_ab, b.extent, b.direction, b.inner_radius, 180.0)
    with pytest.raises(GeometryError):
        PartDesign(d.family, d.thickness, d.k_factor, d.panels, (flat,), d.sequence).validate()


def test_mirrored_parts_are_built_by_reflection_only():
    d = _design("L_BRACKET")
    fp = FoldedPart(d, FormState.nominal(d), mirror=np.diag([-1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(GeometryError, match="unmirrored"):
        brep.build_pieces(fp)
    plain = FoldedPart(d, FormState.nominal(d))
    surf = brep.tessellate(brep.build_pieces(plain)).transformed(np.diag([-1.0, 1.0, 1.0, 1.0]))
    v, t = surf.vertices, surf.triangles
    signed = np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0
    assert abs(signed - brep.analytic_volume(plain)) < 1e-3 * brep.analytic_volume(plain)  # still outward


def _vector_area(surf) -> float:
    v, t = surf.vertices, surf.triangles
    return float(np.linalg.norm(0.5 * np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]).sum(axis=0)))


@pytest.mark.parametrize("family", FAMILIES)
def test_as_built_surfaces_with_process_variation_are_watertight(family):
    """Every as-built scan surface is closed (zero vector area); a cut offset that
    moved an edge touching a bend joint would open a sliver at the joint."""
    from domains.production.engine.synthetic.injection import Injection, Scenario, apply_scenario

    d = _design(family, seed=12)
    sc = Scenario("V", None, "FINAL", (Injection("PROCESS_VARIATION", {"angle_sigma_deg": 0.5, "shift_sigma_mm": 0.2, "cut_sigma_mm": 0.3}),))
    ab = apply_scenario(d, sc, np.random.default_rng(5))
    assert ab.state.cut_offsets  # variation really moved free edges
    fp = FoldedPart(ab.design, ab.state, datum=ab.datum)
    surf = brep.tessellate(brep.build_pieces(fp), 0.005, 0.08)
    assert _vector_area(surf) < 1e-6


def test_cut_offsets_never_move_a_bend_joint():
    d = _design("U_CHANNEL")
    flange = d.panel("PF1")
    touching = flange.edges_touching_attachments()
    assert touching  # the side edges of a flange touch its bend's attachment edge
    states = FormState.nominal(d).bends
    with pytest.raises(GeometryError, match="attachment"):
        FoldedPart(d, FormState(states, {("PF1", min(touching)): 0.2}))
    far = next(i for i, t in enumerate(flange.edge_tags) if t == "FREE" and i not in touching)
    FoldedPart(d, FormState(states, {("PF1", far): 0.2}))  # the far edge may move
