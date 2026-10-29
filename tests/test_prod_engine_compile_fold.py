"""E1 compile, Method B: a bound flat folded with its own per-bend strips (``flat_binding.fold_flat``).

The flats are analytic (``testparts.profile_flat`` with one K per bend, as a shop's bend table
draws them): every expected strip, backgauge and S_0 extent is computed from the fixture
parameters, independently of the compile's development and of the solver. Each positive check
is paired with the value the uniform-K model (Method A) would give, so a test that passes under
Method A does not pass here. Drawing errors (a panel or a line off the CAD) must never be
absorbed into the strips, and an implausible strip must never become a nominal.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from contracts.production.compiled_part import check_compiled_against_recipe
from domains.production.engine.compile import compile_part, parse_flat, propose_flat_binding
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.reconcile import Mark

#: a shop bend table: one K per bend of the hat, all inside the plausible band [0.18, 0.48]
K_TABLE = [0.30, 0.38, 0.42, 0.35]


@pytest.fixture(scope="module")
def hat(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fold_hat")
    shape, info = tp.hat_part(width=40.0)
    step = tp.write_step(shape, tmp / "hat.step")
    return tmp, step, info


def _strip(info, i: int, k: float) -> float:
    b = info.bends[i]
    return math.radians(b.sweep_deg) * (b.ri + k * info.thickness)


def _truth_map(dxf, lines, rotate_deg, offset, mirror) -> dict[str, int]:
    """Drawn line id -> profile bend number, by undoing the placement (independent of the binding)."""
    a = np.radians(rotate_deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = {}
    for ln in parse_flat(dxf).bend_lines:
        q = (ln.midpoint - np.asarray(offset)) @ r
        if mirror:
            q[1] = -q[1]
        out[ln.line_id] = int(np.argmin([abs(q[0] - a0[0]) for a0, _ in lines])) + 1
    return out


def _compile(tmp, step, info, ks, sku, *, placement=(0.0, (0.0, 0.0), False), outer_shift=None, line_shift=None,
             mapping=None, confirmation="DEV_AUTO_CONFIRMED", flat_step=False):
    outer, lines, _ = tp.profile_flat(info, ks)
    if outer_shift is not None:
        outer = outer.copy()
        outer[1:3, 0] += outer_shift  # the last flange drawn longer (a panel off its CAD size)
    if line_shift is not None:
        lines = [(a + [line_shift[i], 0.0], b + [line_shift[i], 0.0]) for i, (a, b) in enumerate(lines)]
    rotate_deg, offset, mirror = placement
    if flat_step:
        flat = tp.write_flat_step(tmp / f"{sku}.step", outer, info.thickness)
        mapping = mapping or {f"I{i + 1}": i + 1 for i in range(len(info.bends))}
    else:
        flat = tp.write_flat_dxf(tmp / f"{sku}.dxf", outer, lines, rotate_deg=rotate_deg, offset=offset, mirror=mirror)
        mapping = mapping or _truth_map(flat, tp.profile_flat(info, ks)[1], rotate_deg, offset, mirror)
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": mapping, "method": "GRAPH_MATCH_RIGID_FIT"}
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku=sku, flat_path=flat, flat_binding=binding,
                        confirmation=confirmation)
    compiled = compile_part(recipe, step, tmp / f"out_{sku}", flat_path=flat).compiled
    return recipe, compiled, flat


@pytest.mark.parametrize("placement", [(0.0, (0.0, 0.0), False), (180.0, (-50.0, 10.0), True),
                                       (27.0, (150.0, -60.0), False)])
def test_a_bend_table_flat_is_folded_with_its_own_strips(hat, placement):
    tmp, step, info = hat
    recipe, c, _ = _compile(tmp, step, info, K_TABLE, f"SYN-FOLD{int(placement[0])}{int(placement[2])}",
                            placement=placement)
    assert c.status == "COMPILED", [(i.code, i.detail) for i in c.issues]
    assert check_compiled_against_recipe(c, recipe) == []
    assert {s.method for s in c.stages} == {"METHOD_B_FOLD_FLAT"}
    assert c.k_factor.source == "BOUND_FLAT"
    uniform_errors = []
    for i, b in enumerate(c.bends):  # candidates are in profile order: compiled bend i is profile bend i
        assert b.neutral_length_mm == pytest.approx(_strip(info, i, K_TABLE[i]), abs=1e-3), i
        uniform_errors.append(abs(_strip(info, i, c.k_factor.value) - _strip(info, i, K_TABLE[i])))
    # the uniform K (Method A) misses this flat's strips by up to 0.2 mm: the fold is what matches it
    assert max(uniform_errors) > 0.15
    fold = c.diagnostics["flat"]["method_b"]
    assert fold["applied"] and fold["explained"] and fold["used"]
    assert [round(fold["per_bend"][b.bend_uid]["k_equivalent"], 3) for b in c.bends] == K_TABLE
    assert all(v["excess_mm"] == 0.0 for v in fold["per_bend"].values())
    assert fold["after"]["max_line_offset_mm"] < 1e-3 and fold["after"]["max_extent_delta_mm"] < 1e-3
    # nothing is left unexplained, so the strips' K-equivalent uncertainty is the floor
    assert c.k_factor.uncertainty == pytest.approx(0.005)


def test_folded_stages_and_backgauges_follow_the_flat(hat):
    tmp, step, info = hat
    _, c, _ = _compile(tmp, step, info, K_TABLE, "SYN-FOLDBG")
    seg, n = info.segment_oml_lengths, len(info.bends)

    def bd(i: int, k: float) -> float:  # bend deduction of profile bend i (90 deg) drawn with K = k
        b = info.bends[i]
        return 2.0 * b.ro * math.tan(math.radians(b.sweep_deg) / 2.0) - _strip(info, i, k)

    bgs = sorted((x.stage_index, x) for x in c.characteristics if x.kind == "BACKGAUGE_LENGTH")
    assert [k for k, _ in bgs] == [1, 2, 3]
    gaps = []
    for k, x in bgs:
        # after bend k the flat run (strips k+1..n, drawn with the table) reaches the far free edge
        expected = sum(seg[k:]) - sum(bd(i, K_TABLE[i]) for i in range(k, n))
        uniform = sum(seg[k:]) - sum(bd(i, c.k_factor.value) for i in range(k, n))
        assert x.nominal == pytest.approx(expected, abs=1e-3), k
        gaps.append(abs(uniform - expected))
    assert max(gaps) > 0.1  # Method A's backgauge nominals would be off by that much
    # S_0 is the flat itself: its developed length along the profile is the drawn blank's
    z = np.load(tmp / "out_SYN-FOLDBG" / c.stages[0].sample_cloud.rel_path, allow_pickle=False)
    pts = z["points"].astype(np.float64)
    flat_length = tp.profile_flat(info, K_TABLE)[2]
    assert float(np.ptp(pts[:, 0])) == pytest.approx(flat_length, abs=0.01)
    assert c.diagnostics["s0_flatness_max_mm"] < 1e-4
    # the K sensitivity of a folded strip is still sweep t per unit K
    d = c.diagnostics["backgauge_nominals"]["S1"]
    assert d["d_nominal_d_k_mm"] == pytest.approx((n - 1) * math.pi / 2 * info.thickness, abs=1e-4)


@pytest.mark.parametrize("label, outer_shift, line_shift", [
    ("end010", 0.10, None),            # the last flange drawn 0.1 mm long
    ("end005", 0.05, None),            # ... 0.05 mm long
    ("line010", None, [0, 0.1, 0, 0]),  # one bend line drawn 0.1 mm off its strip centre
])
def test_a_mis_drawn_panel_or_line_is_never_absorbed_into_the_strips(hat, label, outer_shift, line_shift):
    tmp, step, info = hat
    _, c, _ = _compile(tmp, step, info, 0.33, f"SYN-FOLDE{label.upper()}", outer_shift=outer_shift,
                       line_shift=line_shift)
    # the uniform QA passes (well inside 0.5 mm), but the per-bend fold does not explain the flat
    assert c.status == "COMPILED" and c.flat.developed_length_check.status == "OK"
    fold = c.diagnostics["flat"]["method_b"]
    assert not fold["explained"] and not fold["applied"] and "do not explain" in fold["reason"]
    assert {s.method for s in c.stages} == {"METHOD_A_ROTATE_BACK"}
    assert "Method A" in c.flat.developed_length_check.detail
    for i, b in enumerate(c.bends):
        # uniform strips at the bound K stay within 0.03 mm of the true ones; a fold would have moved
        # a strip by about the drawing error itself
        assert b.neutral_length_mm == pytest.approx(_strip(info, i, c.k_factor.value), abs=1e-6)
        assert abs(b.neutral_length_mm - _strip(info, i, 0.33)) < 0.03


def test_an_implausible_strip_is_clamped_and_beyond_the_threshold_it_blocks(hat):
    tmp, step, info = hat
    ks = [0.33, 0.33, 0.58, 0.33]  # bend 3 needs K = 0.58, 0.1 above the band (0.314 mm of strip)
    _, c, _ = _compile(tmp, step, info, ks, "SYN-FOLDCLAMP")
    assert c.status == "COMPILED" and {s.method for s in c.stages} == {"METHOD_B_FOLD_FLAT"}
    fold = c.diagnostics["flat"]["method_b"]
    assert "clamped" in fold["reason"]
    for i, b in enumerate(c.bends):
        k_used = min(ks[i], 0.48)  # the band edge for the implausible one, the flat's own for the rest
        assert b.neutral_length_mm == pytest.approx(_strip(info, i, k_used), abs=1e-3), i
        assert fold["per_bend"][b.bend_uid]["excess_mm"] == pytest.approx(
            _strip(info, i, ks[i]) - _strip(info, i, k_used), abs=2e-3)
    # 0.565 mm beyond the band: the flat disagrees with the CAD at that bend beyond the threshold
    ks = [0.33, 0.33, 0.66, 0.33]
    _, blocked, _ = _compile(tmp, step, info, ks, "SYN-FOLDBLOCK")
    assert blocked.status == "BLOCKED" and blocked.blocker_codes == ("FLAT_DEVELOPED_LENGTH_MISMATCH",)
    check = blocked.flat.developed_length_check
    # (0.66 - 0.48) of K over a 90 deg, t = 2 strip: 0.18 * pi / 2 * 2 mm
    assert check.status == "MISMATCH" and check.max_abs_delta_mm == pytest.approx(0.18 * math.pi, abs=3e-3)
    assert "beyond every plausible K" in check.detail
    # the uniform check alone would have passed it: the per-bend term is what blocks
    assert blocked.diagnostics["flat"]["proposal"]["developed_max_abs_delta_mm"] < check.threshold_mm
    assert blocked.k_factor.source == "SHOP_DEFAULT"


def test_a_wide_bend_table_is_not_a_developed_length_mismatch(tmp_path):
    # thick sheet (t = 4) and a bend table alternating K = 0.20 / 0.47, every K inside the plausible
    # band: no single K fits it within 0.5 mm, but the flat is exactly a plausible per-bend blank
    shape, info = tp.hat_part(width=40.0, t=4.0, ri=4.0)
    step = tp.write_step(shape, tmp_path / "thick.step")
    ks = [0.20, 0.47, 0.20, 0.47]
    outer, lines, _ = tp.profile_flat(info, ks)
    dxf = tp.write_flat_dxf(tmp_path / "thick.dxf", outer, lines)
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": {f"L{i + 1}": i + 1 for i in range(4)},
               "method": "GRAPH_MATCH_RIGID_FIT"}
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-FOLDWIDE", flat_path=dxf, flat_binding=binding)
    c = compile_part(recipe, step, tmp_path / "out", flat_path=dxf).compiled
    uniform = c.diagnostics["flat"]["proposal"]["developed_max_abs_delta_mm"]
    assert uniform > c.flat.developed_length_check.threshold_mm  # the single-K check alone would block it
    assert c.status == "COMPILED", c.blocker_codes
    check = c.flat.developed_length_check
    assert check.status == "OK" and check.max_abs_delta_mm < 1e-3 and "per-bend QA" in check.detail
    assert {s.method for s in c.stages} == {"METHOD_B_FOLD_FLAT"}
    for i, b in enumerate(c.bends):
        assert b.neutral_length_mm == pytest.approx(_strip(info, i, ks[i]), abs=1e-3)


def test_a_two_direction_flat_is_folded_per_bend(tmp_path):
    # base a x b with up flanges on the +X edge (bend along Y) and the +Y edge (bend along X), drawn
    # with a different K per bend; the blank is analytic: strips sweep (Ri + K t), flanges h - Ri
    a, b, t, ri, h1, h2, gap = 60.0, 40.0, 2.0, 2.0, 15.0, 12.0, 4.0
    k_along_y, k_along_x = 0.28, 0.44
    shape = tp.two_direction_part(a=a, b=b, t=t, ri=ri, h1=h1, h2=h2, gap=gap)
    step = tp.write_step(shape, tmp_path / "two.step")
    l1, l2 = math.pi / 2 * (ri + k_along_y * t), math.pi / 2 * (ri + k_along_x * t)
    w1, w2 = h1 - ri, h2 - ri
    outer = np.array([(0.0, 0.0), (a + l1 + w1, 0.0), (a + l1 + w1, b - gap), (a, b - gap), (a, b), (a - gap, b),
                      (a - gap, b + l2 + w2), (0.0, b + l2 + w2)])
    lines = [(np.array([a + l1 / 2, 0.0]), np.array([a + l1 / 2, b - gap])),
             (np.array([0.0, b + l2 / 2]), np.array([a - gap, b + l2 / 2]))]
    dxf = tp.write_flat_dxf(tmp_path / "two.dxf", outer, lines, rotate_deg=12.0, offset=(7.0, -3.0))
    from domains.production.engine.compile.api import load_part_model

    model = load_part_model(step)
    cands = []
    for g in sorted(model.bends.values(), key=lambda g: g.graph_bend_id):
        mid = (g.r_parent + g.r_child) / np.linalg.norm(g.r_parent + g.r_child)
        p = g.axis_point + (g.ri + g.thickness / 2) * mid
        cands.append({"line_start": (p + g.z0 * g.axis_dir).tolist(), "line_end": (p + g.z1 * g.axis_dir).tolist(),
                      "target_angle": g.sweep_deg, "target_radius": g.ri})
    along_y = [i + 1 for i, cnd in enumerate(cands)
               if abs(cnd["line_end"][1] - cnd["line_start"][1]) > abs(cnd["line_end"][0] - cnd["line_start"][0])]
    assert len(along_y) == 1
    # the drawn line b - gap long is the bend along Y, the a - gap long one the bend along X
    mapping = {ln.line_id: (along_y[0] if abs(ln.length - (b - gap)) < 1e-6 else 3 - along_y[0])
               for ln in parse_flat(dxf).bend_lines}
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": mapping, "method": "GRAPH_MATCH_RIGID_FIT"}
    recipe = dev_recipe(step, cands, sku="SYN-FOLD2D", flat_path=dxf, flat_binding=binding)
    c = compile_part(recipe, step, tmp_path / "out", flat_path=dxf).compiled
    assert c.status == "COMPILED", [(i.code, i.detail) for i in c.issues]
    assert {s.method for s in c.stages} == {"METHOD_B_FOLD_FLAT"}
    for bend in c.bends:
        along = abs(bend.axis.direction[1]) > 0.9
        assert bend.neutral_length_mm == pytest.approx(l1 if along else l2, abs=1e-3)
    # one K for both would be off by (0.44 - 0.28) / 2 of K on each strip: 0.25 mm
    assert abs(math.pi / 2 * (ri + c.k_factor.value * t) - l1) > 0.1


def test_strips_the_flat_does_not_determine_stay_method_a(hat):
    tmp, step, info = hat
    # only the two outer bend lines bound: the outline alone cannot split the strips of a chain
    _, partial, _ = _compile(tmp, step, info, K_TABLE, "SYN-FOLDPART", mapping={"L1": 1, "L4": 4})
    fold = partial.diagnostics["flat"]["method_b"]
    assert partial.status == "COMPILED" and not fold["applied"] and fold["min_singular"] < 1e-6
    assert "do not determine every strip" in fold["reason"]
    assert {s.method for s in partial.stages} == {"METHOD_A_ROTATE_BACK"}
    # a flat STEP has no drawn lines at all (they are inferred from the development itself)
    _, fstep, _ = _compile(tmp, step, info, K_TABLE, "SYN-FOLDSTEP", flat_step=True)
    fold = fstep.diagnostics["flat"]["method_b"]
    assert fstep.status == "COMPILED" and not fold["applied"] and "no drawn bend lines" in fold["reason"]
    assert {s.method for s in fstep.stages} == {"METHOD_A_ROTATE_BACK"}


def test_an_unconfirmed_binding_reports_the_fold_but_never_applies_it(hat):
    tmp, step, info = hat
    _, c, _ = _compile(tmp, step, info, K_TABLE, "SYN-FOLDUNC", confirmation="OPERATOR_CONFIRMED")
    assert c.status == "COMPILED" and c.k_factor.source == "SHOP_DEFAULT"
    fold = c.diagnostics["flat"]["method_b"]
    assert fold["used"] and not fold["applied"]
    assert {s.method for s in c.stages} == {"METHOD_A_ROTATE_BACK"}
    for i, b in enumerate(c.bends):
        assert b.neutral_length_mm == pytest.approx(_strip(info, i, 0.33), abs=1e-6)


def test_the_wizard_proposal_previews_the_fold(hat):
    tmp, step, info = hat
    outer, lines, _ = tp.profile_flat(info, K_TABLE)
    dxf = tp.write_flat_dxf(tmp / "wizard.dxf", outer, lines, rotate_deg=33.0, offset=(5.0, 5.0))
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-FOLDWIZ")
    marks = [Mark(b.bend_uid, b.line_start_mm, b.line_end_mm) for b in recipe.bends]
    out = propose_flat_binding(step, marks, dxf)
    assert out["status"] == "PROPOSED" and out["fold"]["used"]
    assert [round(out["fold"]["per_bend"][b.bend_uid]["k_equivalent"], 3) for b in recipe.bends] == K_TABLE


def test_a_fold_failure_falls_back_to_method_a_with_its_reason(hat, monkeypatch):
    # Method B only refines a bound flat's strips: a solver failure must never block the compile
    import domains.production.engine.compile.flat_binding as fb

    def boom(*_a, **_k):
        raise FloatingPointError("synthetic solver failure")

    monkeypatch.setattr(fb, "fold_flat", boom)
    tmp, step, info = hat
    _, c, _ = _compile(tmp, step, info, K_TABLE, "SYN-FOLDBOOM")
    assert c.status == "COMPILED" and c.k_factor.source == "BOUND_FLAT"
    assert {s.method for s in c.stages} == {"METHOD_A_ROTATE_BACK"}
    fold = c.diagnostics["flat"]["method_b"]
    assert not fold["applied"] and "synthetic solver failure" in fold["reason"]
    assert "synthetic solver failure" in c.flat.developed_length_check.detail


def test_the_folded_compile_is_deterministic(hat):
    tmp, step, info = hat
    outer, lines, _ = tp.profile_flat(info, K_TABLE)
    dxf = tp.write_flat_dxf(tmp / "det.dxf", outer, lines, rotate_deg=-7.0, offset=(3.0, 4.0))
    mapping = _truth_map(dxf, lines, -7.0, (3.0, 4.0), False)
    binding = {"state": "AUTO_PROPOSED", "bend_line_map": mapping, "method": "GRAPH_MATCH_RIGID_FIT"}
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-FOLDDET", flat_path=dxf, flat_binding=binding)
    a = compile_part(recipe, step, tmp / "det_a", flat_path=dxf, compiled_at="2026-10-17T00:00:00Z").compiled
    b = compile_part(recipe, step, tmp / "det_b", flat_path=dxf, compiled_at="2026-10-17T00:00:00Z").compiled
    assert {s.method for s in a.stages} == {"METHOD_B_FOLD_FLAT"}
    assert a.compiled_part_sha256 == b.compiled_part_sha256


def test_the_wizard_proposal_uses_the_datum_the_compile_uses_with_rigid_bends(tmp_path):
    # an L with a rolled radius (unsupported, left unmarked: rigid) and one press-brake bend: the
    # compile restricts the datum to the marked bend's component; the wizard must fit the same
    shape, info = tp.rolled_part()
    step = tp.write_step(shape, tmp_path / "rolled.step")
    cands = tp.candidates_from_info(info)[1:]  # mark the press-brake bend only
    outer, lines, _ = tp.profile_flat(info, 0.33)
    dxf = tp.write_flat_dxf(tmp_path / "rolled.dxf", outer, lines[1:])
    recipe = dev_recipe(step, cands, sku="SYN-FOLDRIG", flat_path=dxf)  # UNBOUND: proposal in diagnostics
    c = compile_part(recipe, step, tmp_path / "out", flat_path=dxf).compiled
    assert c.status == "COMPILED", c.blocker_codes
    assert any(f.note.startswith("RIGID_FOLLOWER_OF:") for f in c.unsupported_features)
    compiled_prop = c.diagnostics["flat"]["proposal"]
    marks = [Mark(b.bend_uid, b.line_start_mm, b.line_end_mm) for b in recipe.bends]
    wizard = propose_flat_binding(step, marks, dxf)["proposal"]
    for key in ("bend_line_map", "mirrored", "k_flat_fit", "outline_rms_mm", "flat_to_development"):
        assert wizard[key] == compiled_prop[key], key
