"""E1 compile: flat parsing (DXF / flat STEP), binding, K from the flat and developed-length QA.

The flats are generated analytically from the fixture parameters (testparts.profile_flat):
setback Ro tan(sweep/2), strip sweep (Ri + K t), independent of the compile's development.
K tolerances are fixed (never the reported uncertainty), and every placement of a drawing
(rotation, translation, mirror) must give the same binding.
"""
from __future__ import annotations

import numpy as np
import pytest

from contracts.production.compiled_part import check_compiled_against_recipe
from domains.production.engine.compile import FlatParseError, compile_part, parse_flat, propose_flat_binding
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.reconcile import Mark

K_TRUE = 0.42


@pytest.fixture(scope="module")
def hat(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("flat_hat")
    shape, info = tp.hat_part(width=40.0)
    step = tp.write_step(shape, tmp / "hat.step")
    outer, lines, _ = tp.profile_flat(info, K_TRUE)
    return tmp, step, info, outer, lines


def _cands(info):
    return tp.candidates_from_info(info)


def _bound_compile(tmp, step, info, dxf, mapping, sku, mirrored=None):
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": mapping, "method": "GRAPH_MATCH_RIGID_FIT"}
    if mirrored is not None:
        binding["mirrored"] = mirrored
    recipe = dev_recipe(step, _cands(info), sku=sku, flat_path=dxf, flat_binding=binding)
    return recipe, compile_part(recipe, step, tmp / f"out_{sku}", flat_path=dxf).compiled


def test_dxf_bend_lines_by_layer_linetype_and_annotation(hat):
    tmp, _, _, outer, lines = hat
    by_layer = parse_flat(tp.write_flat_dxf(tmp / "layer.dxf", outer, lines, layer="BEND", linetype=None))
    assert [b.source for b in by_layer.bend_lines] == ["LAYER"] * 4
    by_lt = parse_flat(tp.write_flat_dxf(tmp / "lt.dxf", outer, lines, layer="0", linetype="DASHED"))
    assert [b.source for b in by_lt.bend_lines] == ["LINETYPE"] * 4
    by_text = parse_flat(tp.write_flat_dxf(tmp / "txt.dxf", outer, lines, layer="0", linetype=None,
                                           annotations=["UP 90°", "DOWN 90°", "90° DN", "UP 90"]))
    assert [b.source for b in by_text.bend_lines] == ["ANNOTATION"] * 4
    assert [(b.direction, b.angle_deg) for b in by_text.bend_lines] == [
        ("UP", 90.0), ("DOWN", 90.0), ("DOWN", 90.0), ("UP", 90.0)]
    assert [b.line_id for b in by_layer.bend_lines] == ["L1", "L2", "L3", "L4"]
    assert len(by_layer.outer) == 4 and by_layer.inner == []
    plain = parse_flat(tp.write_flat_dxf(tmp / "plain.dxf", outer, lines, layer="0", linetype=None))
    assert plain.bend_lines == []  # solid lines on the outline layer are outline, never bend lines


def test_dxf_units_and_holes(hat):
    tmp, _, _, outer, lines = hat
    import ezdxf

    path = tp.write_flat_dxf(tmp / "inch.dxf", outer / 25.4, [(a / 25.4, b / 25.4) for a, b in lines],
                             holes=[((np.array([20.0, 20.0]) / 25.4), 3.0 / 25.4)], units_mm=False)
    doc = ezdxf.readfile(str(path))
    doc.units = ezdxf.units.IN
    doc.saveas(str(path))
    flat = parse_flat(path)
    assert flat.scale_to_mm == 25.4 and flat.units == "inch"
    assert np.ptp(flat.outer[:, 0]) == pytest.approx(np.ptp(outer[:, 0]), abs=1e-6)
    assert len(flat.inner) == 1
    assert np.ptp(flat.inner[0][:, 0]) == pytest.approx(6.0, abs=0.05)


def test_unreadable_dxf_is_a_flat_parse_blocker(hat):
    tmp, step, info, _, _ = hat
    bad = tmp / "bad.dxf"
    bad.write_text("this is not a dxf file\n")
    with pytest.raises(FlatParseError):
        parse_flat(bad)
    recipe = dev_recipe(step, _cands(info), sku="SYN-FBAD", flat_path=bad)
    c = compile_part(recipe, step, tmp / "out_bad", flat_path=bad).compiled
    assert c.status == "BLOCKED" and c.blocker_codes == ("FLAT_PARSE_FAILED",)
    assert c.flat.parse_status == "PARSE_FAILED" and c.flat_sha256 == recipe.flat.sha256


def _marks(recipe):
    return [Mark(b.bend_uid, b.line_start_mm, b.line_end_mm) for b in recipe.bends]


def _truth(placed_lines, lines, rotate_deg, offset, mirror):
    """Drawn line id -> profile bend number, by undoing the placement (independent of the binding)."""
    a = np.radians(rotate_deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = {}
    for ln in placed_lines:
        q = (ln.midpoint - np.asarray(offset)) @ r  # inverse rotation
        if mirror:
            q[1] = -q[1]
        out[ln.line_id] = int(np.argmin([abs(q[0] - a0[0]) for a0, _ in lines])) + 1
    return out


def test_proposal_binds_every_line_and_recovers_k(hat):
    tmp, step, info, outer, lines = hat
    dxf = tp.write_flat_dxf(tmp / "prop.dxf", outer, lines, rotate_deg=27.0, offset=(150.0, -60.0))
    recipe = dev_recipe(step, _cands(info), sku="SYN-FPROP")
    out = propose_flat_binding(step, _marks(recipe), dxf)
    assert out["status"] == "PROPOSED"
    prop = out["proposal"]
    assert prop["k_factor"] == pytest.approx(K_TRUE, abs=0.003)
    # line L<i> sits at the i-th strip from the first free edge: bend i of the profile
    # (a 27 deg rotation keeps the drawing's left-to-right order of the parallel strips)
    expected = {f"L{i + 1}": recipe.bends[i].bend_uid for i in range(4)}
    assert prop["bend_line_map"] == expected
    assert prop["developed_max_abs_delta_mm"] < 0.05
    assert all(v < 0.05 for v in prop["line_offsets_mm"].values())
    assert prop["confidence"] == pytest.approx(1.0) and not prop["needs_operator_review"]
    assert prop["fit_choice"]["equivalent_fits"] >= 2 and not prop["fit_choice"]["ambiguous"]


@pytest.mark.parametrize("rotate_deg", [0.0, 27.0, 180.0, 270.0])
@pytest.mark.parametrize("mirror", [False, True])
def test_binding_is_invariant_to_drawing_placement(hat, rotate_deg, mirror):
    # the hat blank is a rectangle: four fits tie on the outline, and only the drawn lines may
    # choose among them; the old tie-break by sort order flipped with a mere translation
    tmp, step, info, outer, lines = hat
    recipe = dev_recipe(step, _cands(info), sku="SYN-FPLACE")
    n_of = {b.bend_uid: i + 1 for i, b in enumerate(recipe.bends)}
    for offset in ((100.0, 50.0), (-300.0, 20.0)):
        dxf = tp.write_flat_dxf(tmp / f"place_{int(rotate_deg)}_{int(mirror)}_{int(offset[0])}.dxf", outer, lines,
                                rotate_deg=rotate_deg, offset=offset, mirror=mirror)
        prop = propose_flat_binding(step, _marks(recipe), dxf)["proposal"]
        truth = _truth(parse_flat(dxf).bend_lines, lines, rotate_deg, offset, mirror)
        assert {k: n_of[v] for k, v in prop["bend_line_map"].items()} == truth, (rotate_deg, mirror, offset)
        assert prop["k_factor"] == pytest.approx(K_TRUE, abs=0.003)
        assert not prop["fit_choice"]["ambiguous"]


def test_annotations_break_a_tie_the_outline_and_lines_cannot(tmp_path):
    # a point-symmetric Z (OML flanges 31.5 / 31.5): blank and strip pattern are symmetric under
    # a half turn, which swaps the UP bend with the DOWN one, so only the annotations can tell the
    # drawing's two readings apart; the part itself is C2-symmetric, so the drawing's mirror
    # image stays an equally valid reading and the binding is still reported as ambiguous
    shape, info = tp.z_part(a=30.0, h=25.0, b=31.5, width=30.0)
    assert info.segment_oml_lengths[0] == pytest.approx(info.segment_oml_lengths[2])
    step = tp.write_step(shape, tmp_path / "zsym.step")
    recipe = dev_recipe(step, _cands(info), sku="SYN-FZSYM")
    compiled = compile_part(recipe, step, tmp_path / "zsym_out").compiled
    directions = [b.direction for b in compiled.bends]
    assert sorted(directions) == ["DOWN", "UP"]
    outer, lines, _ = tp.profile_flat(info, 0.33)
    n_of = {b.bend_uid: i + 1 for i, b in enumerate(recipe.bends)}
    for rotate_deg in (0.0, 180.0):
        dxf = tp.write_flat_dxf(tmp_path / f"zsym_{int(rotate_deg)}.dxf", outer, lines, rotate_deg=rotate_deg,
                                offset=(40.0, -10.0), annotations=[f"{d} 90°" for d in directions])
        prop = propose_flat_binding(step, _marks(recipe), dxf)["proposal"]
        truth = _truth(parse_flat(dxf).bend_lines, lines, rotate_deg, (40.0, -10.0), False)
        assert {k: n_of[v] for k, v in prop["bend_line_map"].items()} == truth, rotate_deg
        assert "annotations" in prop["fit_choice"]["narrowed_by"]
        assert prop["fit_choice"]["ambiguous"] and not prop["mirrored"]
    # without annotations nothing tells the half-turned reading apart: reported, never hidden
    plain = tp.write_flat_dxf(tmp_path / "zsym_plain.dxf", outer, lines, offset=(40.0, -10.0))
    prop = propose_flat_binding(step, _marks(recipe), plain)["proposal"]
    assert "annotations" not in prop["fit_choice"]["narrowed_by"]
    assert prop["fit_choice"]["ambiguous"] and prop["needs_operator_review"] and prop["confidence"] <= 0.5


def test_bound_flat_gives_k_and_the_stages_fold_its_strips(hat):
    tmp, step, info, outer, lines = hat
    dxf = tp.write_flat_dxf(tmp / "bound.dxf", outer, lines, rotate_deg=-12.0, offset=(10.0, 5.0))
    recipe, c = _bound_compile(tmp, step, info, dxf, {f"L{i + 1}": i + 1 for i in range(4)}, "SYN-FB")
    assert c.status == "COMPILED", [(i.code, i.detail) for i in c.issues]
    assert c.k_factor.source == "BOUND_FLAT" and c.k_factor.value == pytest.approx(K_TRUE, abs=0.003)
    assert 0 < c.k_factor.uncertainty < 0.05
    # Method B: the flat (drawn with one K) determines every strip, and the stages fold it
    assert {s.method for s in c.stages} == {"METHOD_B_FOLD_FLAT"}
    assert {s.k_factor_used for s in c.stages} == {c.k_factor.value}
    for b in c.bends:
        assert b.neutral_length_mm == pytest.approx(
            np.radians(b.sweep_deg) * (b.inner_radius_mm + K_TRUE * b.thickness_mm), abs=1e-3)
    fold = c.diagnostics["flat"]["method_b"]
    assert fold["applied"] and fold["explained"] and max(abs(v["delta_mm"]) for v in fold["per_bend"].values()) < 0.01
    assert c.flat.developed_length_check.status == "OK"
    assert "Method B" in c.flat.developed_length_check.detail
    assert c.flat.binding_state == "AUTO_PROPOSED" and c.flat.bend_lines_found == 4
    codes = {i.code for i in c.issues}
    assert "K_FACTOR_DEFAULTED" not in codes and "FLAT_NOT_BOUND" not in codes
    assert check_compiled_against_recipe(c, recipe) == []


def test_unconfirmed_proposal_in_an_operator_recipe_gives_no_k(hat):
    # AUTO_PROPOSED is a proposal: it supplies K only in a DEV_AUTO_CONFIRMED recipe
    tmp, step, info, outer, lines = hat
    dxf = tp.write_flat_dxf(tmp / "unconf.dxf", outer, lines)
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": {f"L{i + 1}": i + 1 for i in range(4)},
               "method": "GRAPH_MATCH_RIGID_FIT"}
    recipe = dev_recipe(step, _cands(info), sku="SYN-FUNCONF", flat_path=dxf, flat_binding=binding,
                        confirmation="OPERATOR_CONFIRMED")
    c = compile_part(recipe, step, tmp / "out_unconf", flat_path=dxf).compiled
    assert c.status == "COMPILED" and c.k_factor.source == "SHOP_DEFAULT"
    assert c.flat.developed_length_check.status == "OK"  # consistent, just not confirmed
    issue = next(i for i in c.issues if i.code == "FLAT_NOT_BOUND")
    assert "not operator-confirmed" in issue.detail


def test_mirrored_flat_drawing_is_bound(hat):
    tmp, step, info, outer, lines = hat
    # mirror across the profile direction (x -> -x): the strip pattern is reversed in the drawing
    mirrored_outer = outer * [-1.0, 1.0]
    mirrored_lines = [(a * [-1.0, 1.0], b * [-1.0, 1.0]) for a, b in lines]
    dxf = tp.write_flat_dxf(tmp / "mirror.dxf", mirrored_outer, mirrored_lines)
    recipe = dev_recipe(step, _cands(info), sku="SYN-FMIR")
    marks = [Mark(b.bend_uid, b.line_start_mm, b.line_end_mm) for b in recipe.bends]
    prop = propose_flat_binding(step, marks, dxf)["proposal"]
    flat = parse_flat(dxf)
    by_id = {ln.line_id: float(ln.midpoint[0]) for ln in flat.bend_lines}
    # the drawn line nearest the first free edge (largest x after the mirror) is bend 1
    first = max(by_id, key=by_id.get)
    assert prop["bend_line_map"][first] == recipe.bends[0].bend_uid
    assert len(prop["bend_line_map"]) == 4
    assert prop["k_factor"] == pytest.approx(K_TRUE, abs=0.003)


def test_wrong_released_map_is_flat_binding_inconsistent(hat):
    tmp, step, info, outer, lines = hat
    dxf = tp.write_flat_dxf(tmp / "swap.dxf", outer, lines)
    swapped = {"L1": 2, "L2": 1, "L3": 3, "L4": 4}
    _, c = _bound_compile(tmp, step, info, dxf, swapped, "SYN-FSWAP")
    assert c.status == "BLOCKED" and "FLAT_BINDING_INCONSISTENT" in c.blocker_codes
    assert c.k_factor.source == "SHOP_DEFAULT"  # an inconsistent flat never sets K


@pytest.mark.parametrize("fmt, extra, shift_line, blocked", [
    ("DXF", 1.0, 0.0, True),    # 1 mm too long: K* = 0.65 would absorb it, the plausible band does not
    ("DXF", 1.0, 0.5, True),    # the same with the drawn line moved with the error
    ("STEP", 1.0, 0.0, True),
    ("STEP", 0.6, 0.0, False),  # 0.47 mm is absorbed up to the band edge: K clamped, residual 0.13 mm
])
def test_a_uniform_length_error_is_not_absorbed_into_k(tmp_path, fmt, extra, shift_line, blocked):
    shape, info = tp.l_part(width=40.0)  # one bend: the weakest case, every mm moves K by 0.32
    step = tp.write_step(shape, tmp_path / "l.step")
    outer, lines, _ = tp.profile_flat(info, 0.33)
    long_outer = outer.copy()
    long_outer[1:3, 0] += extra
    if fmt == "DXF":
        flat = tp.write_flat_dxf(tmp_path / "l.dxf", long_outer, [(a + [shift_line, 0], b + [shift_line, 0])
                                                                  for a, b in lines])
        mapping = {"L1": 1}
    else:
        flat = tp.write_flat_step(tmp_path / "l_flat.step", long_outer, info.thickness)
        mapping = {"I1": 1}
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": mapping, "method": "GRAPH_MATCH_RIGID_FIT"}
    recipe = dev_recipe(step, _cands(info), sku=f"SYN-FU{fmt}{int(10 * extra)}{int(10 * shift_line)}",
                        flat_path=flat, flat_binding=binding)
    c = compile_part(recipe, step, tmp_path / "out", flat_path=flat).compiled
    check = c.flat.developed_length_check
    proposal = c.diagnostics["flat"]["proposal"]
    lo, hi = proposal["k_plausible_band"]
    assert (lo, hi) == pytest.approx((0.18, 0.48), abs=1e-9)
    assert proposal["k_flat_fit"] > hi  # the flat alone would have absorbed the error into K
    assert proposal["k_clamped"] and proposal["needs_operator_review"]
    if blocked:
        assert c.status == "BLOCKED" and c.blocker_codes == ("FLAT_DEVELOPED_LENGTH_MISMATCH",)
        assert check.status == "MISMATCH" and check.max_abs_delta_mm > check.threshold_mm
        assert c.k_factor.source == "SHOP_DEFAULT"
    else:
        assert c.status == "COMPILED" and check.status == "OK"
        assert c.k_factor.source == "BOUND_FLAT" and c.k_factor.value == pytest.approx(hi, abs=1e-6)
        assert c.k_factor.uncertainty >= proposal["k_flat_fit"] - hi - 1e-6  # the clamp is in the uncertainty
        assert "outside the plausible band" in check.detail


@pytest.mark.parametrize("k_drawn, expect_k, status", [(0.45, 0.45, "OK"), (0.20, 0.20, "OK"),
                                                        (0.60, 0.48, "MISMATCH")])
def test_k_from_a_uniformly_drawn_hat(hat, k_drawn, expect_k, status):
    # every strip drawn with one K: inside the plausible band it is recovered; beyond it the four
    # strips' 1.5 mm can no longer pass for a plausible K
    tmp, step, info, _, _ = hat
    outer, lines, _ = tp.profile_flat(info, k_drawn)
    dxf = tp.write_flat_dxf(tmp / f"uniform_{k_drawn}.dxf", outer, lines, rotate_deg=15.0, offset=(20.0, 30.0))
    recipe, c = _bound_compile(tmp, step, info, dxf, {f"L{i + 1}": i + 1 for i in range(4)},
                               f"SYN-FK{int(100 * k_drawn)}")
    proposal = c.diagnostics["flat"]["proposal"]
    assert proposal["k_flat_fit"] == pytest.approx(k_drawn, abs=0.003)
    assert proposal["k_factor"] == pytest.approx(expect_k, abs=0.003)
    assert c.flat.developed_length_check.status == status
    if status == "OK":
        assert c.k_factor.source == "BOUND_FLAT" and c.k_factor.value == pytest.approx(k_drawn, abs=0.003)


def test_hole_centre_marks_are_not_bend_lines(hat):
    tmp, _, _, outer, lines = hat
    path = tp.write_flat_dxf(tmp / "cm.dxf", outer, lines, holes=[((np.array([10.0, 20.0])), 2.5)])
    import ezdxf

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    for dx, dy in ((3.0, 0.0), (0.0, 3.0)):  # a centre-mark cross on the bend linetype
        msp.add_line((10.0 - dx, 20.0 - dy), (10.0 + dx, 20.0 + dy), dxfattribs={"layer": "BEND", "linetype": "DASHED"})
    doc.saveas(str(path))
    flat = parse_flat(path)
    assert len(flat.bend_lines) == 4 and len(flat.inner) == 1
    assert any("centre marks" in n for n in flat.notes)


def test_flat_with_a_wrong_developed_length_is_a_qa_mismatch(hat):
    tmp, step, info, outer, lines = hat
    long_outer = outer.copy()
    long_outer[1:3, 0] += 1.2  # the last flange drawn 1.2 mm too long
    dxf = tp.write_flat_dxf(tmp / "long.dxf", long_outer, lines)
    _, c = _bound_compile(tmp, step, info, dxf, {f"L{i + 1}": i + 1 for i in range(4)}, "SYN-FLONG")
    assert c.status == "BLOCKED" and "FLAT_DEVELOPED_LENGTH_MISMATCH" in c.blocker_codes
    check = c.flat.developed_length_check
    assert check.status == "MISMATCH" and check.max_abs_delta_mm > check.threshold_mm


def test_unbound_flat_is_parsed_with_a_proposal_and_default_k(hat):
    tmp, step, info, outer, lines = hat
    dxf = tp.write_flat_dxf(tmp / "unbound.dxf", outer, lines)
    recipe = dev_recipe(step, _cands(info), sku="SYN-FUNB", flat_path=dxf)
    c = compile_part(recipe, step, tmp / "out_unb", flat_path=dxf).compiled
    assert c.status == "COMPILED"
    assert c.k_factor.source == "SHOP_DEFAULT" and c.flat.binding_state == "UNBOUND"
    assert c.flat.developed_length_check.status == "NOT_EVALUATED"
    assert {"FLAT_NOT_BOUND", "K_FACTOR_DEFAULTED"} <= {i.code for i in c.issues}
    assert len(c.diagnostics["flat"]["proposal"]["bend_line_map"]) == 4


def test_flat_step_lines_are_inferred(tmp_path):
    shape, info = tp.z_part(width=30.0)
    step = tp.write_step(shape, tmp_path / "z.step")
    outer, _, _ = tp.profile_flat(info, 0.38)
    flat_step = tp.write_flat_step(tmp_path / "z_flat.step", outer, info.thickness)
    parsed = parse_flat(flat_step)
    assert parsed.format == "STEP" and parsed.bend_lines == []
    assert parsed.thickness_mm == pytest.approx(info.thickness, abs=1e-9)
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-ZFS")
    marks = [Mark(b.bend_uid, b.line_start_mm, b.line_end_mm) for b in recipe.bends]
    prop = propose_flat_binding(step, marks, flat_step)["proposal"]
    assert sorted(prop["bend_line_map"]) == ["I1", "I2"]
    assert prop["needs_operator_review"]  # inferred lines always need a look
    assert prop["k_factor"] == pytest.approx(0.38, abs=0.005)  # a fixed tolerance, not the reported one
    assert prop["k_uncertainty"] < 0.05
    # a rectangle without drawn lines cannot tell its ends apart: the binding says so
    assert prop["fit_choice"]["ambiguous"] and prop["confidence"] <= 0.5
