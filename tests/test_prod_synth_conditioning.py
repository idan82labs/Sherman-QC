"""E2 far-boundary conditioning: REQUIRED nominals and scored truths must not depend
on how a consumer places its section stations.

E1 samples a flange corridor at 21 stations with a ``min(0.5 mm, 2 %)`` trim and
reduces them with a 60 % class majority and a 0.05 mm dominant level; E2 uses 41
stations with a ``min(2 mm, 5 %)`` trim and the station median. For every flange
side the recipe makes REQUIRED, and for every backgauge side truth calls
``well_defined``, both rules must give the same number. The reduction below
re-implements E1's documented rule on E2's own station values (no E1 import).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from domains.production.engine.synthetic.analytic import E1_STATION_RULE, FarBoundary, _line_intervals, corridor_stations, far_boundary
from domains.production.engine.synthetic.families import FAMILIES, FeatureRequest, ProfileSegment, build_family, build_profile, random_placement
from domains.production.engine.synthetic.geometry import UP, FoldedPart, FormState, PanelDesign
from domains.production.engine.synthetic.injection import Injection, Scenario, apply_scenario, cut_lines, far_parallel_edges
from domains.production.engine.synthetic.recipe import build_case_recipe, flange_inspection_overrides
from domains.production.engine.synthetic.truth import build_nominal, build_truth


def e1_like_value(fb: FarBoundary) -> float | None:
    """E1's documented reduction (ENGINE_COMPILE nominals): >= 60 % class majority; a
    FREE edge takes its dominant level (>= 60 % of the class, levels chained within
    0.05 mm) or the median of a continuously varying edge; ADJACENT takes the median."""
    if not fb.groups or fb.groups[0][2] < 0.6:
        return None
    kind, adj, frac, median = fb.groups[0][:4]
    if kind == "ADJACENT_BEND":
        return median
    if kind != "FREE_EDGE":
        return None
    class_levels = [lv for lv in fb.levels if lv[0] == kind and lv[1] == adj]
    top = class_levels[0]
    if top[2] / frac >= 0.6:
        return top[3]
    if len(class_levels) >= 0.5 * frac * fb.valid_stations:
        return median
    return None


def _designs(n_seeds: int):
    for fam in FAMILIES:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            req = FeatureRequest(acute=seed % 5 == 1, large_radius=seed % 5 == 2, short_flange=seed % 5 == 3, long_flange=seed % 5 == 4)
            design = build_family(fam, rng, req)
            if seed % 2:
                design = replace(design, placement=tuple(tuple(r) for r in random_placement(rng).tolist()))
            yield fam, seed, design


def _same(fp, key, panel, across, fb_e2):
    fb_e1 = far_boundary(fp, key, panel, across_unformed=across, station_rule=E1_STATION_RULE)
    assert fb_e1.well_defined, (key, panel, fb_e1.levels[:3])
    assert fb_e1.boundary_kind == fb_e2.boundary_kind and fb_e1.adjacent_bend == fb_e2.adjacent_bend
    value = e1_like_value(fb_e1)
    assert value is not None and abs(value - fb_e2.value_mm) < 1e-9, (key, panel, value, fb_e2.value_mm)


def test_required_flange_and_well_defined_backgauge_nominals_are_station_rule_independent():
    required = backgauges = ill = 0
    for fam, seed, design in _designs(16):
        nominal = build_nominal(design)
        overrides = flange_inspection_overrides(nominal)
        for b in design.bends:
            excluded = overrides.get(b.key, {}).get("flange_inspection", {})
            for letter in "AB":
                fb = nominal.flange[(b.key, letter)]
                if excluded.get("side_a" if letter == "A" else "side_b") == "NOT_INSPECTED":
                    continue
                required += 1
                assert fb.well_defined, (fam, seed, b.key, letter)
                _same(nominal.final, b.key, nominal.side_panel(b.key, letter), False, fb)
        for k, per_side in nominal.backgauge.items():
            key = design.sequence[k - 1]
            for letter, fb in per_side.items():
                if not fb.well_defined:
                    ill += 1
                    continue
                backgauges += 1
                _same(nominal.stages[k], key, nominal.side_panel(key, letter), True, fb)
    assert required > 500 and backgauges > 150
    assert ill == 0  # the families are laid out to avoid ill-defined runs (see families.py)


def test_well_defined_as_built_truths_are_station_rule_independent():
    """With process variation, injected cut/shift/angle errors and partial stages, a
    truth that is ``well_defined`` gives the same value under both station rules."""
    checked = 0
    for fam, seed, design in _designs(6):
        nominal = build_nominal(design)
        n = len(design.sequence)
        injections = [Injection("PROCESS_VARIATION", {"angle_sigma_deg": 0.3, "shift_sigma_mm": 0.1, "cut_sigma_mm": 0.1}),
                      Injection("BEND_LINE_SHIFT", {"sequence_index": 1, "delta_mm": 0.8}),
                      Injection("ANGLE", {"sequence_index": n, "delta_included_deg": 1.5})]
        for stage in sorted({None, max(1, n - 1)}, key=lambda x: (x is None, x)):
            ab = apply_scenario(design, Scenario("T", stage, "IN_PROCESS", tuple(injections)), np.random.default_rng(seed))
            actual = FoldedPart(ab.design, ab.state, datum=ab.datum)
            for b in design.bends:
                if not actual.bend_geo[b.key].formed:
                    continue
                for letter in "AB":
                    panel = nominal.side_panel(b.key, letter)
                    fb = far_boundary(actual, b.key, panel, across_unformed=False)
                    if fb.well_defined and nominal.flange[(b.key, letter)].well_defined:
                        _same(actual, b.key, panel, False, fb)
                        checked += 1
    assert checked > 150


def _u_channel(**web):
    segs = [ProfileSegment("PF1", 40.0, "flange"), ProfileSegment("PWEB", 100.0, "web", **web), ProfileSegment("PF2", 60.0, "flange")]
    return build_profile(family="U_CHANNEL", thickness=2.0, width=80.0, segments=segs, sweeps=[90.0, 90.0], directions=[UP, UP],
                         radii=[2.0, 2.0], sequence=[0, 1], params={"oml_lengths": [40.0, 100.0, 60.0], "width": 80.0})


def test_far_boundary_is_the_farthest_crossing():
    """A notch the station line passes over does not end the flange (contract: per-bin
    maximum extent). The web of this U channel has a 6 mm notch in its side edge, so
    the corridor stations at y in [74, 78] leave the material at the notch wall and
    re-enter it; the far boundary is still BND2's OML sharp, 100 mm, everywhere."""
    d = _u_channel(notches=(("y1", 0.3, 0.5, 6.0),))
    fp = FoldedPart(d, FormState.nominal(d))
    fb = far_boundary(fp, "BND1", "PWEB", across_unformed=False)
    assert fb.boundary_kind == "ADJACENT_BEND" and fb.adjacent_bend == "BND2" and fb.kind_fraction == 1.0
    assert fb.min_mm == pytest.approx(100.0, abs=1e-9) and fb.max_mm == pytest.approx(100.0, abs=1e-9)
    assert fb.well_defined and fb.dominant_level_fraction == 1.0
    # mutate-and-check: those stations really leave the material at the notch wall, so a
    # first-exit march would have reported a FREE_EDGE about 70 mm long there
    g = fp.bend_geo["BND1"]
    u = g.outward("PWEB")
    notch_stations = [v for v in corridor_stations(g.extent) if v > 80.0 - 6.0]
    assert notch_stations
    for v in notch_stations:
        intervals = _line_intervals(fp.outline["PWEB"], fp.tags["PWEB"], g.line_point + v * g.d, u)
        assert len(intervals) == 2 and intervals[0][3] == "FREE" and intervals[1][3] == "BND2"
        assert intervals[0][1] - g.sigma < 75.0


def _notch_over_f3(design, cover: float):
    """The pre-fix PANEL layout: the base's top-edge notch over ``cover`` of F3's span."""
    base = design.panel("PBASE")
    xa3, xb3 = design.bend("BNDF3").extent
    pts = list(base.outline)
    y1 = pts[6][1]
    depth = y1 - pts[8][1]
    span = xb3 - xa3
    nx0, nx1 = xa3 + (0.5 - cover / 2) * span, xa3 + (0.5 + cover / 2) * span
    pts[7], pts[8], pts[9], pts[10] = (nx1, y1), (nx1, y1 - depth), (nx0, y1 - depth), (nx0, y1)
    new_base = PanelDesign(base.key, tuple(pts), base.edge_tags, base.holes, base.role)
    out = replace(design, panels=tuple(new_base if p.key == "PBASE" else p for p in design.panels))
    out.validate()
    return out


def test_a_near_50_percent_stepped_edge_is_flagged_and_never_required():
    """Mutate-and-check (review finding): the old PANEL layout put the base notch over
    ~50 % of F3's corridor. Its median flips with station placement, so it must be
    flagged ill-defined, NOT_INSPECTED by the recipe rule, and kept out of the
    expectation lists even when it is truly out of tolerance."""
    good = build_family("PANEL_WITH_HOLES_NOTCHES", np.random.default_rng(4), FeatureRequest())
    bad = _notch_over_f3(good, cover=0.5)
    for design, expect_ok in ((good, True), (bad, False)):
        nominal = build_nominal(design)
        side = next(letter for letter in "AB" if nominal.side_panel("BNDF3", letter) == "PBASE")
        fb = nominal.flange[("BNDF3", side)]
        assert fb.boundary_kind == "FREE_EDGE" and fb.kind_fraction == 1.0
        assert fb.well_defined is expect_ok, fb.levels[:3]
        excluded = flange_inspection_overrides(nominal).get("BNDF3", {}).get("flange_inspection", {})
        assert (excluded.get("side_a" if side == "A" else "side_b") == "NOT_INSPECTED") is (not expect_ok)
        if not expect_ok:
            assert fb.ill_defined_reason == "STEPPED_FAR_BOUNDARY" and 0.35 < fb.dominant_level_fraction < 0.65
            e1 = far_boundary(nominal.final, "BNDF3", "PBASE", across_unformed=False, station_rule=E1_STATION_RULE)
            assert e1_like_value(e1) is None or abs(e1_like_value(e1) - fb.value_mm) > 1.0  # the rules disagree
    # a backgauge across that stepped edge cannot be excluded by the recipe: truth flags it
    staged = replace(bad, sequence=("BNDF3",) + tuple(k for k in bad.sequence if k != "BNDF3"), groups=())
    nominal = build_nominal(staged)
    cr = build_case_recipe(design=staged, nominal=nominal, sku="SYN-TEST-2", revision="A", part_name="t", cad_sha256="a" * 64,
                           drawing_sha256="b" * 64, flat_sha256="c" * 64, inspection_stage="IN_PROCESS",
                           bend_overrides=flange_inspection_overrides(nominal))
    sc = Scenario("S1", 1, "IN_PROCESS", (Injection("BEND_LINE_SHIFT", {"sequence_index": 1, "delta_mm": 1.4}),))
    ab = apply_scenario(staged, sc, np.random.default_rng(0))
    truth = build_truth(case_id="T", nominal=nominal, recipe=cr.recipe, uid_by_key=cr.uid_by_key, as_built=ab, scenario=sc,
                        cad_to_raw=np.eye(4), labels=None, ideal_cad=None, spacing_mm=0.46, symmetry=None, scan_meta={})
    (cid,) = [c for c, v in truth["characteristics"].items() if v["kind"] == "BACKGAUGE_LENGTH"]
    sides = truth["characteristics"][cid]["per_side"]
    base_side = next(s for s, v in sides.items() if v["panel"] == "PBASE")
    assert sides[base_side]["well_defined"] is False and sides[base_side]["truth_in_tolerance"] is False
    assert f"{cid}#{base_side}" in truth["expected"]["characteristics_ill_defined"]
    assert f"{cid}#{base_side}" not in truth["expected"]["characteristics_never_pass"]


def test_collinear_cut_segments_move_together():
    """A far edge split by a notch is one straight cut: process variation gives its
    segments one offset, and a flange-length error moves the whole far end (both
    segments and the notch bottom), so the far boundary keeps its dominant level."""
    d = build_family("PANEL_WITH_HOLES_NOTCHES", np.random.default_rng(7), FeatureRequest())
    f3 = d.panel("PF3")
    lines = cut_lines(f3)
    far = far_parallel_edges(d, "PF3", "BNDF3")
    assert len(far) == 3  # two collinear far-edge segments + the notch bottom
    split = [line for line in lines if len(line) == 2]
    assert len(split) == 1 and set(split[0]) < set(far)
    sc = Scenario("V", None, "FINAL", (Injection("PROCESS_VARIATION", {"angle_sigma_deg": 0.0, "shift_sigma_mm": 0.0, "cut_sigma_mm": 0.3}),))
    ab = apply_scenario(d, sc, np.random.default_rng(3))
    a, b = (ab.state.cut_offsets[("PF3", i)] for i in split[0])
    assert a == b != 0.0
    actual = FoldedPart(ab.design, ab.state, datum=ab.datum)
    assert far_boundary(actual, "BNDF3", "PF3", across_unformed=False).well_defined
    # mutate-and-check: independent offsets on the two segments break the dominant level
    cuts = dict(ab.state.cut_offsets)
    cuts[("PF3", split[0][0])] += 0.2
    broken = FoldedPart(ab.design, FormState(ab.state.bends, cuts), datum=ab.datum)
    assert not far_boundary(broken, "BNDF3", "PF3", across_unformed=False).well_defined
    # a flange-length error shifts every level of the far end by exactly delta
    nominal = build_nominal(d)
    seq = d.sequence.index("BNDF3") + 1
    err = apply_scenario(d, Scenario("E", None, "FINAL", (Injection("FLANGE_LENGTH", {"sequence_index": seq, "panel": "PF3", "delta_mm": 0.9}),)),
                         np.random.default_rng(0))
    fb_err = far_boundary(FoldedPart(err.design, err.state, datum=err.datum), "BNDF3", "PF3", across_unformed=False)
    side = next(letter for letter in "AB" if nominal.side_panel("BNDF3", letter) == "PF3")
    fb_nom = nominal.flange[("BNDF3", side)]
    assert fb_err.well_defined and fb_err.value_mm - fb_nom.value_mm == pytest.approx(0.9, abs=1e-9)
    assert [lv[3] for lv in fb_err.levels] == pytest.approx([lv[3] + 0.9 for lv in fb_nom.levels], abs=1e-9)


def test_partial_span_layouts_keep_one_dominant_boundary():
    """PANEL F1 covers >= 86 % of the base edge and the TWO_DIRECTION_TAB tab <= 15 % of
    F2's edge, so the marches across them stay well defined at every stage."""
    for fam in ("PANEL_WITH_HOLES_NOTCHES", "TWO_DIRECTION_TAB"):
        for seed in range(20):
            design = build_family(fam, np.random.default_rng(seed), FeatureRequest())
            nominal = build_nominal(design)
            for k, per_side in nominal.backgauge.items():
                for letter, fb in per_side.items():
                    assert fb.well_defined, (fam, seed, k, letter, fb.levels[:3])
            if fam == "PANEL_WITH_HOLES_NOTCHES":
                base = design.panel("PBASE")
                xa3, xb3 = design.bend("BNDF3").extent
                notch_x = sorted({p[0] for p in base.outline[7:11]})
                assert notch_x[0] > xb3 or notch_x[-1] < xa3  # the base notch is outside F3's span
                ya, yb = design.bend("BNDF1").extent
                assert (yb - ya) / base.outline[6][1] >= 0.86
            else:
                ta, tb = design.bend("BNDT").extent
                assert (tb - ta) <= 0.15 * design.bend("BNDF2").extent[1] + 1e-9
