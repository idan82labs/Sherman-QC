"""E2 truth semantics: injected errors, stages, order, backgauge (no scanning).

Parts here are built from explicit parameters so expectations are closed-form:
U channel, flanges 40 / web 100 / flange 60 mm (OML), 90 deg, Ri = 2, t = 2,
K = 0.33, width 80, datum = web.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from contracts.production.characteristics import make_characteristic_id
from contracts.production.part_recipe import PartRecipeV1
from domains.production.engine.synthetic.families import ProfileSegment, build_profile, build_family, FeatureRequest
from domains.production.engine.synthetic.geometry import UP, GeometryError, neutral_length, outside_setback
from domains.production.engine.synthetic.injection import Injection, Scenario, apply_scenario, order_consistent
from domains.production.engine.synthetic.recipe import build_case_recipe
from domains.production.engine.synthetic.truth import build_nominal, build_truth

T, RI, K = 2.0, 2.0, 0.33
BD90 = 2 * outside_setback(90.0, RI, T) - neutral_length(90.0, RI, T, K)


def _u_channel(groups=()):
    segs = [ProfileSegment("PF1", 40.0, "flange"), ProfileSegment("PWEB", 100.0, "web"), ProfileSegment("PF2", 60.0, "flange")]
    return build_profile(
        family="U_CHANNEL", thickness=T, width=80.0, segments=segs, sweeps=[90.0, 90.0], directions=[UP, UP],
        radii=[RI, RI], sequence=[0, 1], groups=groups, params={"oml_lengths": [40.0, 100.0, 60.0], "width": 80.0},
    )


def _truth(design, scenario, seed=0, wrong=None):
    nominal = build_nominal(design)
    cr = build_case_recipe(
        design=design, nominal=nominal, sku="SYN-TEST-1", revision="A", part_name="t", cad_sha256="a" * 64,
        drawing_sha256="b" * 64, flat_sha256="c" * 64, inspection_stage=scenario.inspection_stage,
    )
    ab = apply_scenario(design, scenario, np.random.default_rng(seed), wrong_part_design=wrong)
    truth = build_truth(
        case_id="T", nominal=nominal, recipe=cr.recipe, uid_by_key=cr.uid_by_key, as_built=ab, scenario=scenario,
        cad_to_raw=np.eye(4), labels=None, ideal_cad=None, spacing_mm=0.5, symmetry=None, scan_meta={},
    )
    return truth, cr, ab


def _char(truth, cr, key, kind, side=None):
    return truth["characteristics"][make_characteristic_id(kind, bend_uid=cr.uid_by_key[key], side=side)]


def test_nominal_truth_has_zero_deviation_and_contract_ids():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("COMPLETE_NOMINAL", None, "FINAL"))
    PartRecipeV1.model_validate(cr.recipe.model_dump(mode="json"))  # hashes recompute
    assert [b.bend_uid for b in cr.recipe.bends] == [b["bend_uid"] for b in truth["bends"]]
    for cid, c in truth["characteristics"].items():
        assert cid.startswith("bu_") and c["true_deviation"] == 0.0 and c["truth_in_tolerance"] is True
    # OML lengths: B1 sides are the 40 flange and the 100 web
    lengths = sorted(_char(truth, cr, "BND1", "FLANGE_LENGTH", s)["truth"] for s in "AB")
    assert lengths == pytest.approx([40.0, 100.0], abs=1e-9)
    assert truth["expected"]["summary_never"] == []
    assert truth["stage"]["expected_stage_model_id"] == "S2"


def test_angle_error_is_exact():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("E", None, "FINAL", (Injection("ANGLE", {"sequence_index": 2, "delta_included_deg": 1.75}),)))
    c = _char(truth, cr, "BND2", "INCLUDED_ANGLE")
    assert c["truth"] == pytest.approx(91.75, abs=1e-12) and c["true_deviation"] == pytest.approx(1.75, abs=1e-12)
    assert c["truth_in_tolerance"] is False
    assert any(x.endswith("INCLUDED_ANGLE") for x in truth["expected"]["characteristics_never_pass"])
    assert "ALL_MEASURED_IN_TOLERANCE" in truth["expected"]["summary_never"]


def test_flange_cut_error_changes_only_that_side():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("E", None, "FINAL", (Injection("FLANGE_LENGTH", {"sequence_index": 2, "panel": "PF2", "delta_mm": -0.8}),)))
    a, b = (_char(truth, cr, "BND2", "FLANGE_LENGTH", s) for s in "AB")
    devs = {round(a["nominal"]): a["true_deviation"], round(b["nominal"]): b["true_deviation"]}
    assert devs[60] == pytest.approx(-0.8, abs=1e-9) and devs[100] == pytest.approx(0.0, abs=1e-9)


def test_flange_error_on_datum_is_refused():
    d = _u_channel()
    with pytest.raises(GeometryError, match="datum"):
        apply_scenario(d, Scenario("E", None, "FINAL", (Injection("FLANGE_LENGTH", {"sequence_index": 1, "panel": "PWEB", "delta_mm": 1.0}),)),
                       np.random.default_rng(0))


def test_bend_line_shift_moves_material_from_b_to_a():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("E", None, "FINAL", (Injection("BEND_LINE_SHIFT", {"sequence_index": 1, "delta_mm": 0.7}),)))
    a, b = (_char(truth, cr, "BND1", "FLANGE_LENGTH", s) for s in "AB")
    assert a["true_deviation"] == pytest.approx(0.7, abs=1e-9)
    assert b["true_deviation"] == pytest.approx(-0.7, abs=1e-9)
    pos = next(x for x in truth["bends"] if x["key"] == "BND1")["actual"]["bend_line_position"]
    assert np.linalg.norm(pos["offset_vector_mm"]) == pytest.approx(0.7, abs=1e-9)
    # the contract position is the side-A component; the rest is reported, not hidden
    assert math.hypot(pos["value_mm"], pos["normal_component_mm"]) == pytest.approx(0.7, abs=1e-9)


def test_springback_is_shared_by_the_group():
    d = _u_channel(groups=[[0, 1]])
    truth, cr, _ = _truth(d, Scenario("S", None, "FINAL", (Injection("SPRINGBACK_GROUP", {"group_index": 1, "delta_included_deg": 1.2}),)))
    devs = [_char(truth, cr, k, "INCLUDED_ANGLE")["true_deviation"] for k in ("BND1", "BND2")]
    assert devs == pytest.approx([1.2, 1.2], abs=1e-12)


def test_stage_one_backgauge_matches_developed_length():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("S1", 1, "IN_PROCESS"))
    st = truth["stage"]
    assert st["formed_count"] == 1 and st["is_sequence_prefix"] and st["expected_stage_model_id"] == "S1"
    bg = truth["backgauge"]
    assert bg["stage_index"] == 1 and bg["bend_uid"] == cr.uid_by_key["BND1"]
    (side,) = bg["sides"].values()  # only the web side leads to the unformed bend
    assert side["nominal_mm"] == pytest.approx(100.0 + 60.0 - BD90, abs=1e-9)
    assert side["truth_mm"] == pytest.approx(side["nominal_mm"], abs=1e-9)
    cid = make_characteristic_id("BACKGAUGE_LENGTH", bend_uid=cr.uid_by_key["BND1"], stage_index=1)
    assert cid in truth["characteristics"]
    # the unformed bend is never a defect and has no actual
    for kind, s in (("FLANGE_LENGTH", "A"), ("FLANGE_LENGTH", "B"), ("INCLUDED_ANGLE", None), ("BEND_LINE_POSITION", None)):
        c = _char(truth, cr, "BND2", kind, s)
        assert c["truth"] is None and c["applicable"] is False and c["not_applicable_reason"] == "BEND_NOT_YET_FORMED"
    assert len(truth["expected"]["characteristics_never_fail"]) == 4
    assert truth["expected"]["bend_states"][cr.uid_by_key["BND2"]] == "NOT_YET_FORMED"
    # the formed flange whose far boundary is the unformed bend: undefined, not nominal
    web_side = next(_char(truth, cr, "BND1", "FLANGE_LENGTH", s) for s in "AB" if _char(truth, cr, "BND1", "FLANGE_LENGTH", s)["nominal"] == pytest.approx(100.0))
    assert web_side["truth"] is None and web_side["truth_undefined_reason"] == "ADJACENT_BEND_NOT_FORMED"
    # ... and it is listed explicitly: the engine must not decide it (NOT_MEASURED only)
    web_cid = make_characteristic_id("FLANGE_LENGTH", bend_uid=cr.uid_by_key["BND1"], side=web_side["side"])
    exp = truth["expected"]
    assert exp["characteristics_must_be_unknown"] == [web_cid]
    assert web_cid not in exp["characteristics_never_pass"] and web_cid not in exp["characteristics_never_fail"]
    assert exp["characteristics_ill_defined"] == []
    # a defined, well-defined side carries the flag; the backgauge sides too
    other = next(_char(truth, cr, "BND1", "FLANGE_LENGTH", s) for s in "AB" if s != web_side["side"])
    assert other["well_defined"] is True and other["truth_stations"]["dominant_level_fraction"] == 1.0
    assert all(v["well_defined"] for v in bg["sides"].values())


def test_backgauge_error_from_bend_line_shift():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("S1", 1, "IN_PROCESS", (Injection("BEND_LINE_SHIFT", {"sequence_index": 1, "delta_mm": 0.6}),)))
    (side,) = truth["backgauge"]["sides"].values()
    assert abs(side["true_deviation_mm"]) == pytest.approx(0.6, abs=1e-9)
    assert side["truth_in_tolerance"] is False  # tolerance +-0.3


def test_partial_bend_and_fraction_bounds():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("P", 0, "IN_PROCESS", (Injection("PARTIAL_BEND", {"sequence_index": 1, "fraction": 0.5}),)))
    b1 = next(x for x in truth["bends"] if x["key"] == "BND1")["actual"]
    assert b1["state"] == "PARTIALLY_FORMED" and b1["formed_fraction"] == pytest.approx(0.5, abs=1e-12)
    assert b1["included_deg"] == pytest.approx(135.0, abs=1e-12)
    assert "ALL_MEASURED_IN_TOLERANCE" in truth["expected"]["summary_never"]
    for bad in (0.2, 0.75):
        with pytest.raises(GeometryError, match="fraction"):
            apply_scenario(d, Scenario("P", 0, "IN_PROCESS", (Injection("PARTIAL_BEND", {"sequence_index": 1, "fraction": bad}),)),
                           np.random.default_rng(0))


def test_order_swap_is_out_of_order_unless_grouped():
    d = _u_channel()
    sc = Scenario("O", 1, "IN_PROCESS", (Injection("ORDER_SWAP", {"formed_out": 1, "formed_in": 2}),))
    truth, cr, _ = _truth(d, sc)
    st = truth["stage"]
    assert st["formed_bend_uids"] == [cr.uid_by_key["BND2"]]
    assert st["is_sequence_prefix"] is False and st["order_consistent"] is False
    assert st["out_of_order_formed_bend_uids"] == [cr.uid_by_key["BND2"]] and st["expected_stage_model_id"] is None
    assert truth["expected"]["bend_states_contract"][cr.uid_by_key["BND2"]] == "UNEXPECTED_FORMED"
    grouped = _u_channel(groups=[[0, 1]])
    assert order_consistent(grouped, {"BND2"}) is True
    assert order_consistent(d, {"BND2"}) is False
    with pytest.raises(GeometryError):
        apply_scenario(d, Scenario("O", 1, "IN_PROCESS", (Injection("ORDER_SWAP", {"formed_out": 2, "formed_in": 2}),)), np.random.default_rng(0))


def test_missing_bend_in_final_is_never_green_or_fail():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("M", None, "FINAL", (Injection("MISSING_BEND", {"sequence_index": 2}),)))
    assert truth["stage"]["missing_bend_uids_final"] == [cr.uid_by_key["BND2"]]
    assert "ALL_MEASURED_IN_TOLERANCE" in truth["expected"]["summary_never"]
    assert all(cr.uid_by_key["BND2"] in x for x in truth["expected"]["characteristics_never_fail"])


def test_extra_bend_changes_the_split_flange():
    d = _u_channel()
    truth, cr, ab = _truth(d, Scenario("X", None, "FINAL", (Injection("EXTRA_BEND", {"fraction": 0.5, "sweep_deg": 30.0, "direction": "UP"}),)))
    assert len(ab.extra_bends) == 1 and len(truth["extra_bends"]) == 1
    changed = [c for c in truth["characteristics"].values() if c.get("boundary_changed")]
    assert changed and all(c["truth_in_tolerance"] is False for c in changed)
    assert "ALL_MEASURED_IN_TOLERANCE" in truth["expected"]["summary_never"]


def test_mirrored_and_wrong_parts_have_no_truth_values():
    d = _u_channel()
    for sc, wrong in (
        (Scenario("MIR", None, "FINAL", (Injection("MIRRORED", {}),)), None),
        (Scenario("W", None, "FINAL", (Injection("WRONG_SKU", {}),)), build_family("U_CHANNEL", np.random.default_rng(3), FeatureRequest())),
    ):
        truth, cr, _ = _truth(d, sc, wrong=wrong)
        bend_chars = {cid: c for cid, c in truth["characteristics"].items() if c["kind"] != "BACKGAUGE_LENGTH"}
        assert all(c["truth"] is None and c["truth_undefined_reason"] in ("MIRRORED_PART", "WRONG_PART_SCANNED") for c in bend_chars.values())
        assert set(truth["expected"]["characteristics_never_pass"]) == set(bend_chars)
        assert {"ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE"} <= set(truth["expected"]["summary_never"])
    assert truth["stage"]["formed_count"] is None  # wrong part: the stage is undefined


def test_wrong_sku_needs_a_part():
    with pytest.raises(GeometryError):
        apply_scenario(_u_channel(), Scenario("W", None, "FINAL", (Injection("WRONG_SKU", {}),)), np.random.default_rng(0))
    with pytest.raises(ValueError):
        Injection("NOT_A_KIND", {})


def test_stage_models_and_transforms():
    d = _u_channel()
    truth, cr, _ = _truth(d, Scenario("S1", 1, "IN_PROCESS"))
    models = {m["stage_model_id"]: m for m in truth["stage_models_nominal"]}
    assert list(models) == ["S0", "S1", "S2"]
    assert models["S1"]["formed_bend_uids"] == [cr.uid_by_key["BND1"]]
    assert np.allclose(models["S0"]["panel_transforms_final_to_stage"]["PWEB"], np.eye(4))
    assert not np.allclose(models["S1"]["panel_transforms_final_to_stage"]["PF2"], np.eye(4))
    assert np.allclose(models["S1"]["panel_transforms_final_to_stage"]["PF1"], np.eye(4))
