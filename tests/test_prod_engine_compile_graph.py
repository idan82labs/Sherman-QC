"""E1 compile: B-rep reading, flange graph, part model (datum, skins, axes) and loci.

Fixtures are real OCP B-reps built in-test (domains/production/engine/compile/testparts.py):
no E2, no real CAD, no scans, no server imports. Expected values are analytic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.brep import shape_to_model
from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.flange_graph import GraphError, build_flange_graph
from domains.production.engine.compile.geometry import rotation_matrix
from domains.production.engine.compile.part_model import build_part_model

CFG = CompileConfig()


def graph_of(shape):
    return build_flange_graph(shape_to_model(shape), CFG)


def model_of(shape, rigid=None):
    return build_part_model(graph_of(shape), rigid)


def _xz(p) -> np.ndarray:
    return np.array([p[0], p[2]])


@pytest.mark.parametrize(
    "name, make, panels, bends, t",
    [
        ("L", lambda: tp.l_part(width=30.0), 2, 1, 2.0),
        ("U", lambda: tp.u_part(width=30.0), 3, 2, 2.0),
        ("Z", lambda: tp.z_part(width=30.0), 3, 2, 1.5),
        ("hat", lambda: tp.hat_part(width=30.0), 5, 4, 2.0),
    ],
)
def test_profile_families_give_the_expected_flange_graph(name, make, panels, bends, t):
    shape, info = make()
    g = graph_of(shape)
    assert len(g.panels) == panels and len(g.bends) == bends, name
    assert g.thickness == pytest.approx(t, abs=1e-9)
    assert g.thickness_spread < 1e-6
    assert not g.features, [f.reason for f in g.features]
    for b in g.bends.values():
        assert b.sweep_deg == pytest.approx(90.0, abs=1e-6)
        assert b.ri == pytest.approx(info.bends[0].ri, abs=1e-9)
        assert b.ro - b.ri == pytest.approx(t, abs=1e-9)
        assert b.length == pytest.approx(info.width, abs=1e-6)
        assert b.unsupported is None
    for p in g.panels.values():
        assert len(p.sides) == 2 and p.thickness == pytest.approx(t, abs=1e-9)


def test_loci_match_the_analytic_profile_and_the_axis_is_never_the_bend_line():
    shape, info = tp.hat_part(width=30.0)
    m = model_of(shape)
    assert len(m.bends) == len(info.bends)
    for b in m.bends.values():
        loci = b.loci()
        sharp = _xz(0.5 * (loci["oml_virtual_sharp"][0] + loci["oml_virtual_sharp"][1]))
        ref = min(info.bends, key=lambda r: float(np.linalg.norm(_xz(r.oml_sharp) - sharp)))
        assert np.linalg.norm(_xz(ref.oml_sharp) - sharp) < 1e-6
        assert np.linalg.norm(_xz(ref.iml_sharp) - _xz(loci["iml_virtual_sharp"][0])) < 1e-6
        assert np.linalg.norm(_xz(ref.center) - _xz(loci["cylinder_axis"][0])) < 1e-6
        center = loci["cylinder_axis"][0]
        assert np.linalg.norm(loci["outer_apex"][0] - center) == pytest.approx(b.ro, abs=1e-9)
        assert np.linalg.norm(loci["inner_apex"][0] - center) == pytest.approx(b.ri, abs=1e-9)
        assert np.linalg.norm(loci["outer_tangent_parent"][0] - center) == pytest.approx(b.ro, abs=1e-9)
        # the published bend line (OML sharp) is Ro / cos(sweep/2) away from the cylinder axis
        gap = np.linalg.norm(loci["oml_virtual_sharp"][0] - center)
        assert gap == pytest.approx(b.ro / math.cos(math.radians(b.sweep_deg) / 2), abs=1e-9)
        assert gap > b.ro
        # every locus spans the bend length along the axis
        for start, end in loci.values():
            assert np.linalg.norm(end - start) == pytest.approx(b.length, abs=1e-9)


def test_datum_skin0_oml_skin_and_direction_follow_the_contract():
    shape, info = tp.hat_part(width=30.0)
    m = model_of(shape)
    datum = m.panels[m.datum]
    # the top of the hat is the largest panel
    assert datum.graph.area == max(p.graph.area for p in m.panels.values())
    n0 = datum.side(0).normal
    assert n0 == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)  # larger +Z wins
    by_x = sorted(m.bends.values(), key=lambda b: float(b.oml_sharp()[0][0] + b.oml_sharp()[0][2] * 1e-3))
    # walls go down from the top (OML on skin 0 => DOWN); flanges turn outward (OML on skin 1 => UP)
    top_bends = [b for b in m.bends.values() if b.parent == m.datum]
    assert len(top_bends) == 2
    for b in top_bends:
        assert b.oml_skin == 0 and b.direction == "DOWN"
    for b in m.bends.values():
        if b not in top_bends:
            assert b.oml_skin == 1 and b.direction == "UP"
    assert len(by_x) == 4


def test_axis_orientation_takes_the_child_from_flat_to_formed():
    shape, _ = tp.z_part(width=30.0)
    m = model_of(shape)
    for b in m.bends.values():
        r = rotation_matrix(b.axis_dir, b.sweep_rad)
        assert r @ (-b.p_out) == pytest.approx(b.c_out, abs=1e-9)
        assert r @ b.r_parent == pytest.approx(b.r_child, abs=1e-9)
        # outward directions lie in their panel planes and point away from the bend
        assert abs(float(b.p_out @ b.r_parent)) < 1e-9 and abs(float(b.c_out @ b.r_child)) < 1e-9


def test_two_direction_part_has_perpendicular_bends_on_the_base_datum():
    m = model_of(tp.two_direction_part(a=60.0, b=40.0, h1=15.0, h2=12.0))
    assert len(m.panels) == 3 and len(m.bends) == 2
    a, b = list(m.bends.values())
    assert abs(float(a.axis_dir @ b.axis_dir)) < 1e-9
    assert a.parent == b.parent == m.datum


def test_slot_end_rounds_tangent_to_walls_are_walls_not_features():
    shape, _ = tp.l_part(width=40.0)
    # a slot from the base free edge (x = 0) with a round end: the slot's straight walls
    # are tangent to the round end wall along generators, like an outline corner round
    shape = tp.cut_box(shape, (-1.0, 16.0, -1.0), (12.0, 24.0, 5.0))
    shape = tp.cut_hole(shape, (12.0, 20.0, 1.0), (0, 0, 1), 8.0)
    g = graph_of(shape)
    assert len(g.panels) == 2 and len(g.bends) == 1
    assert not g.features, [f.reason for f in g.features]


def test_hem_and_rolled_radius_are_unsupported_by_the_frozen_rule():
    g = graph_of(tp.hem_part()[0])
    kinds = sorted((round(b.sweep_deg), b.unsupported) for b in g.bends.values())
    assert kinds == [(90, None), (172, "HEM")]
    g = graph_of(tp.rolled_part()[0])
    rolled = [b for b in g.bends.values() if b.unsupported]
    assert len(rolled) == 1 and rolled[0].unsupported == "ROLLED" and "10 x t" in rolled[0].unsupported_reason


def test_rule_thresholds_are_strict_mutations_below_them_stay_supported():
    g = graph_of(tp.hem_part(sweep=165.0)[0])
    assert all(b.unsupported is None for b in g.bends.values())
    g = graph_of(tp.rolled_part(ri=18.0)[0])  # 9 t < 10 t: a large but press-brake radius
    assert all(b.unsupported is None for b in g.bends.values())


def test_window_cycle_is_resolved_into_one_windowed_panel():
    plain = graph_of(tp.u_part(width=80.0)[0])
    windowed = graph_of(tp.window_u_part()[0])
    assert len(windowed.panels) == len(plain.panels) == 3
    assert len(windowed.bends) == len(plain.bends) == 2
    assert any("cycle broken" in n for n in windowed.notes)
    base = max(windowed.panels.values(), key=lambda p: len(p.sides[0].faces))
    assert len(base.sides[0].faces) == 2  # the two base pieces are one rigid panel
    for b in windowed.bends.values():
        assert b.length == pytest.approx(80.0, abs=1e-6)  # the bend spans both pieces
    build_part_model(windowed)  # a tree


def test_bend_interrupted_by_a_relief_stays_one_bend():
    shape, _ = tp.l_part(width=50.0)
    shape = tp.cut_box(shape, (30.0, 20.0, -1.0), (50.0, 30.0, 6.0))  # cut through the bend region only
    g = graph_of(shape)
    assert len(g.bends) == 1 and len(g.panels) == 2
    (b,) = g.bends.values()
    assert b.length == pytest.approx(50.0, abs=1e-6)
    assert len(b.inner_faces) == 2 and len(b.outer_faces) == 2


def test_a_flat_plate_has_no_datum_and_is_refused():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    g = graph_of(BRepPrimAPI_MakeBox(50.0, 30.0, 2.0).Shape())
    assert not g.bends
    with pytest.raises(GraphError) as exc:
        build_part_model(g)
    assert exc.value.code == "DATUM_PANEL_UNDETERMINED"


def test_rigid_bends_restrict_the_datum_and_refuse_split_recipe_bends():
    shape, _ = tp.rolled_part()
    g = graph_of(shape)
    rolled = next(k for k, b in g.bends.items() if b.unsupported)
    free_model = build_part_model(g)
    rigid_model = build_part_model(g, {rolled})
    normal = next(b for b in rigid_model.bends.values() if b.unsupported is None)
    assert rigid_model.datum in (normal.parent, normal.child)
    assert free_model.datum != rigid_model.datum  # the largest panel is behind the rigid bend
    # a rolled bend between two press-brake bends cannot be left unmarked
    shape2, _ = tp.profile_part([(0.0, 0.0), (40.0, 0.0), (40.0, 60.0), (90.0, 60.0), (90.0, 30.0)], 2.0,
                                [2.0, 25.0, 2.0], 30.0)
    g2 = graph_of(shape2)
    middle = next(k for k, b in g2.bends.items() if b.unsupported == "ROLLED")
    with pytest.raises(GraphError) as exc:
        build_part_model(g2, {middle})
    assert exc.value.code == "FLANGE_GRAPH_NOT_A_TREE"


def test_ids_are_deterministic_for_the_same_cad():
    shape, _ = tp.hat_part(width=30.0)
    a, b = graph_of(shape), graph_of(shape)
    assert sorted((p.panel_id, round(p.area, 6)) for p in a.panels.values()) == sorted(
        (p.panel_id, round(p.area, 6)) for p in b.panels.values())
    assert sorted(x.graph_bend_id for x in a.bends.values()) == ["G1", "G2", "G3", "G4"]


def test_panel_boundaries_label_free_and_bend_segments():
    shape, _ = tp.l_part(width=30.0)
    g = graph_of(shape)
    for p in g.panels.values():
        for side in p.sides:
            outer = [lp for lp in side.loops if lp.is_outer]
            assert len(outer) == 1
            labels = set(outer[0].labels.tolist())
            assert labels == {0, 1}  # FREE and BEND segments only


@pytest.mark.parametrize("make", [lambda: tp.l_part(width=30.0), lambda: tp.hat_part(width=30.0)])
def test_bspline_encoded_bends_are_recognised_like_analytic_ones(tmp_path, make):
    # some exporters write every plane, bend cylinder and straight edge as a B-spline
    shape, info = make()
    nurbs = tp.nurbs_convert(shape)
    analytic = graph_of(shape)
    brep = shape_to_model(nurbs)
    assert {f.surface for f in brep.faces} == {"bspline"}
    assert any("recognised as cylinder" in n for n in brep.notes) and any("edges recognised as lines" in n
                                                                          for n in brep.notes)
    g = build_flange_graph(brep, CFG)
    assert len(g.panels) == len(analytic.panels) and len(g.bends) == len(analytic.bends)
    assert not g.features
    for b in g.bends.values():
        assert b.sweep_deg == pytest.approx(90.0, abs=1e-6) and b.unsupported is None
        assert b.ri == pytest.approx(info.bends[0].ri, abs=1e-6) and b.ro - b.ri == pytest.approx(info.thickness, abs=1e-6)
    assert g.thickness == pytest.approx(info.thickness, abs=1e-9)


def test_bspline_part_compiles_to_the_analytic_nominals(tmp_path):
    from domains.production.engine.compile import compile_part
    from domains.production.engine.compile.dev_recipe import dev_recipe

    shape, info = tp.hat_part(width=30.0)
    values = []
    for tag, sh in (("A", shape), ("B", tp.nurbs_convert(shape))):
        step = tp.write_step(sh, tmp_path / f"hat_{tag}.step")
        c = compile_part(dev_recipe(step, tp.candidates_from_info(info), sku=f"SYN-BSPL-{tag}"), step,
                         tmp_path / f"out_{tag}").compiled
        assert c.status == "COMPILED", c.blocker_codes
        values.append({x.characteristic_id.split("/", 1)[1] + str(i): x.nominal
                       for i, b in enumerate(c.bends) for x in c.characteristics if x.bend_uid == b.bend_uid})
    assert values[0].keys() == values[1].keys()
    for k in values[0]:
        assert values[1][k] == pytest.approx(values[0][k], abs=1e-6), k


def test_a_multi_body_step_is_refused_with_a_reason(tmp_path):
    from domains.production.engine.compile import compile_part
    from domains.production.engine.compile.brep import CadReadError, load_step
    from domains.production.engine.compile.dev_recipe import dev_recipe
    from domains.production.engine.compile.testparts import _transform

    shape, info = tp.l_part(width=30.0)
    other = _transform(tp.l_part(width=30.0)[0], translate=(0.0, 100.0, 0.0))
    step = tp.write_step(tp.compound_of(shape, other), tmp_path / "two.step")
    with pytest.raises(CadReadError) as exc:
        load_step(step)
    assert exc.value.reason == "MULTIPLE_SOLID_BODIES"
    c = compile_part(dev_recipe(step, tp.candidates_from_info(info), sku="SYN-MULTI"), step, tmp_path / "out").compiled
    assert c.status == "BLOCKED" and c.blocker_codes == ("CAD_READ_FAILED",)
    assert "MULTIPLE_SOLID_BODIES" in c.issues[0].detail
    # a negligible second body (a stray sliver under 1 % of the volume) is dropped, not refused
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    sliver = BRepPrimAPI_MakeBox(1.0, 1.0, 0.5).Shape()
    model = load_step(tp.write_step(tp.compound_of(shape, _transform(sliver, translate=(0.0, 200.0, 0.0))),
                                    tmp_path / "sliver.step"))
    assert model.solid_count == 2


def test_lone_cylinders_join_the_formed_feature_around_them():
    # a rounded boss on the base: its foot fillets are lone cylinders tangent to the panel. Rule 2
    # classifies the whole smooth cluster once (one STEPPED_BOSS); rule 1 split each foot fillet
    # off as its own ROLLED record
    shape, _ = tp.l_part(a=60.0, width=50.0)
    bossed = tp.add_rounded_boss(shape, (15.0, 15.0), (35.0, 35.0), z=2.0, height=1.0, corner_radius=3.0,
                                 fillet_radius=0.8)
    brep = shape_to_model(bossed)
    g = build_flange_graph(brep, CFG)
    assert sorted(f.kind for f in g.features) == ["STEPPED_BOSS"], [(f.kind, f.reason) for f in g.features]
    (feat,) = g.features
    assert {brep.faces[f].surface for f in feat.face_indices} == {"cylinder", "torus", "plane"}
    # the one-sided footprint is the fillet ring around the rounded 20 x 20 foot, not the area of
    # every feature face (walls and both sides): perimeter * 0.8 + pi 0.8^2 = 61.9 mm2
    perimeter = 4 * (20.0 - 6.0) + 2 * math.pi * 3.0
    assert feat.footprint_mm2 == pytest.approx(perimeter * 0.8 + math.pi * 0.8 ** 2, rel=0.02)
    assert feat.area_mm2 > 1.5 * feat.footprint_mm2
    assert g.panels[feat.panel_key].dominant_feature is None


def test_a_dominant_feature_makes_the_panel_unsupported():
    from domains.production.engine.compile.unsupported import panel_feature_dominates

    assert not panel_feature_dominates(49.0, 51.0) and panel_feature_dominates(51.0, 49.0)
    assert not panel_feature_dominates(0.0, 0.0)
