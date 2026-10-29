"""Post-E3 zero-shot overlay rules (``domains.production.zeroshot.overlay``), synthetic only.

Built on the C01 SYN-HAT-1 example result + CompiledPart (``contracts/production/examples``); the zs lines are
synthetic segments in the raw scan frame derived from the example's own registration. No scan, no CAD, no model.

Covers: the overlay never produces or upgrades a PASS; withheld bends get REVIEW lines; a refused scan gets
unverified lines and stays refused (wrong part, mirrored part, untrusted registration); an extra bend on the
right part lowers a pass-like verdict to REVIEW; the PDF lists the overlay in its own section.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "contracts" / "production" / "examples"
SUMMARY_STATES = ("ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE", "OUT_OF_TOLERANCE_FOUND", "REVIEW_REQUIRED",
                  "INCOMPLETE", "INVALID_INSPECTION")
PASS_LIKE = {"ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE"}


@pytest.fixture(scope="module")
def docs():
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.inspection_result import ShopfloorInspectionResultV1

    compiled = CompiledPartV1.model_validate_json((EXAMPLES / "compiled_part_v1.example.json").read_bytes())
    result = ShopfloorInspectionResultV1.model_validate_json((EXAMPLES / "shopfloor_result_v1.example.json").read_bytes())
    invalid = ShopfloorInspectionResultV1.model_validate_json(
        (EXAMPLES / "shopfloor_result_v1.invalid.example.json").read_bytes())
    return compiled, result, invalid


def raw_locus(result, compiled, uid: str, shift=(0.0, 0.0, 0.0)) -> list[list[float]]:
    from domains.production.engine.eval.engine_output import decoded_stage, frame_to_raw, stage_locus

    p = frame_to_raw(result, stage_locus(compiled, decoded_stage(result, compiled), uid, "outer_apex"))
    return (p + np.asarray(shift, float)).tolist()


def zs_doc(result, lines: list[list[list[float]]], score: float = 0.9) -> dict:
    from domains.production.zeroshot.pin import selected_lane

    # the lane stamp of the selected pin (default zs_lane_v1; SHERMAN_ZS_LANE_PIN=zs_lane_v3 runs every rule with v3)
    return {
        "schema": "sherman_zeroshot_lines_v1", "status": "OK", "lane": selected_lane(), "variant": "zs",
        "runtime_version": "zs-prod-1.0.0", "code_fingerprint": "a" * 64, "model_sha256": "b" * 64,
        "scan_sha256": result.scan.sha256, "priors": {"n_bends": 4, "t_mm": 2.0, "K": 1, "max_families": 3},
        "lines": [{"line_id": f"ZS{i + 1:02d}", "start_mm": a, "end_mm": b, "incl_deg": 90.0, "score": score}
                  for i, (a, b) in enumerate(lines)],
    }


def with_state(result, state: str):
    return result.model_copy(update={"summary": result.summary.model_copy(update={"state": state})})


def withhold(result, uid: str, reason: str = "REGISTRATION_UNTRUSTED"):
    rows = []
    for row in result.bends:
        if row.bend_uid == uid:
            chars = tuple(
                c.model_copy(update={"state": "NOT_MEASURED", "decision": "UNKNOWN", "reason": reason, "actual": None,
                                     "deviation": None, "uncertainty": None})
                if c.kind == "BEND_LINE_POSITION" else c for c in row.characteristics)
            row = row.model_copy(update={"characteristics": chars})
        rows.append(row)
    return result.model_copy(update={"bends": tuple(rows)})


def refuse(invalid, validity: str):
    iv = invalid.inspection_validity.model_copy(update={"state": validity})
    return invalid.model_copy(update={"inspection_validity": iv})


def build(result, compiled, lines):
    from domains.production.zeroshot.overlay import build_overlay

    return build_overlay(result, compiled, zs_doc(result, lines), engine_result_sha256="c" * 64)


def extra_line(result) -> list[list[float]]:
    """A bend the recipe does not have: parallel to the bends, 15 mm away from every CAD locus."""
    from domains.production.engine.eval.engine_output import frame_to_raw

    return frame_to_raw(result, np.asarray([[20.0, 0.0, 40.0], [20.0, 100.0, 40.0]])).tolist()


# ------------------------------------------------------------------ never a PASS


def test_effective_state_never_creates_or_upgrades_a_pass():
    from domains.production.zeroshot.overlay import effective_state

    for state, flagged in itertools.product(SUMMARY_STATES, (False, True)):
        eff = effective_state(state, flagged)
        assert eff in (state, "REVIEW_REQUIRED")
        if eff in PASS_LIKE:
            assert eff == state and not flagged
        if eff != state:
            assert state in PASS_LIKE and flagged and eff == "REVIEW_REQUIRED"


@pytest.mark.parametrize("state", SUMMARY_STATES)
def test_overlay_on_every_e3_state_only_lowers(docs, state):
    compiled, result, _ = docs
    r = with_state(result, state)
    lines = [raw_locus(r, compiled, b.bend_uid) for b in compiled.bends] + [extra_line(r)]
    ov = build(r, compiled, lines)
    assert ov.e3_summary_state == state
    assert ov.effective_summary_state == ("REVIEW_REQUIRED" if state in PASS_LIKE else state)
    assert ov.verdict_lowered is (state in PASS_LIKE)
    assert all(ln.decision in ("REVIEW", "UNVERIFIED") for ln in ov.lines)
    assert r.summary.state == state  # the E3 result object is never touched


def test_overlay_model_refuses_any_upgrade_or_pass_line(docs):
    from domains.production.zeroshot.overlay import ZeroshotOverlayV1

    compiled, result, _ = docs
    good = build(with_state(result, "ALL_MEASURED_IN_TOLERANCE"), compiled, [extra_line(result)])
    doc = json.loads(json.dumps(good.model_dump(mode="json", by_alias=True)))
    ZeroshotOverlayV1.model_validate(doc)
    for e3, eff in (("REVIEW_REQUIRED", "ALL_MEASURED_IN_TOLERANCE"), ("OUT_OF_TOLERANCE_FOUND", "STAGE_IN_TOLERANCE"),
                    ("INVALID_INSPECTION", "ALL_MEASURED_IN_TOLERANCE"), ("INCOMPLETE", "REVIEW_REQUIRED"),
                    ("OUT_OF_TOLERANCE_FOUND", "REVIEW_REQUIRED")):
        bad = {**doc, "e3_summary_state": e3, "effective_summary_state": eff, "verdict_lowered": True}
        with pytest.raises(ValueError):
            ZeroshotOverlayV1.model_validate(bad)
    unlowered = {**doc, "effective_summary_state": "ALL_MEASURED_IN_TOLERANCE", "verdict_lowered": False}
    with pytest.raises(ValueError):  # a flag on a pass-like state must lower it
        ZeroshotOverlayV1.model_validate(unlowered)
    for decision in ("PASS", "FAIL"):
        lines = [{**doc["lines"][0], "decision": decision}]
        with pytest.raises(ValueError):
            ZeroshotOverlayV1.model_validate({**doc, "lines": lines})


# ------------------------------------------------------------------ option 2: withheld bends


def test_withheld_bend_gets_a_review_line(docs):
    compiled, result, _ = docs
    uid = compiled.bends[1].bend_uid
    r = withhold(result, uid)
    lines = [raw_locus(r, compiled, b.bend_uid, shift=(0.0, 0.0, 1.0) if b.bend_uid == uid else (0, 0, 0))
             for b in compiled.bends]
    ov = build(r, compiled, lines)
    gap = [ln for ln in ov.lines if ln.kind == "WITHHELD_BEND"]
    assert len(gap) == 1 and gap[0].bend_uid == uid and gap[0].decision == "REVIEW" and gap[0].source == "ZS_LANE_V1"
    assert gap[0].assoc is not None and gap[0].assoc.offset_mm <= 6.0
    assert gap[0].start_frame_mm is not None  # the station / PDF can place it in E3's frame
    assert ov.withheld_bends == 1 and ov.withheld_bends_located == 1
    assert not ov.flags and ov.effective_summary_state == r.summary.state  # the published lines are explained


def test_withheld_bend_without_an_associated_line_stays_withheld(docs):
    compiled, result, _ = docs
    uid = compiled.bends[2].bend_uid
    r = withhold(result, uid, reason="OCCLUDED")
    lines = [raw_locus(r, compiled, b.bend_uid) for b in compiled.bends if b.bend_uid != uid]
    ov = build(r, compiled, lines)
    assert ov.withheld_bends == 1 and ov.withheld_bends_located == 0
    assert not any(ln.kind == "WITHHELD_BEND" for ln in ov.lines)


def test_a_published_line_is_never_reused_as_a_gap_fill(docs):
    compiled, result, _ = docs
    uid = compiled.bends[0].bend_uid
    r = withhold(result, uid)
    # only the other bends' published lines are offered: none may be attached to the withheld bend
    lines = [raw_locus(r, compiled, b.bend_uid) for b in compiled.bends if b.bend_uid != uid]
    ov = build(r, compiled, lines)
    assert ov.withheld_bends_located == 0


# ------------------------------------------------------------------ refused scans


@pytest.mark.parametrize("validity", ["WRONG_PART_SUSPECTED", "MIRRORED_PART_SUSPECTED", "UNTRUSTED_REGISTRATION"])
def test_refused_scan_gets_unverified_lines_and_stays_refused(docs, validity):
    compiled, result, invalid = docs
    r = refuse(invalid, validity)
    lines = [raw_locus(result, compiled, b.bend_uid) for b in compiled.bends] + [extra_line(result)]
    ov = build(r, compiled, lines)
    assert ov.inspection_validity == validity
    assert ov.e3_summary_state == ov.effective_summary_state == "INVALID_INSPECTION" and not ov.verdict_lowered
    assert len(ov.lines) == 5 and all(ln.kind == "REFUSED_SCAN_DETECTION" and ln.decision == "UNVERIFIED" for ln in ov.lines)
    assert all(ln.bend_uid is None and ln.assoc is None and ln.start_frame_mm is None for ln in ov.lines)
    assert not ov.flags and not ov.review_reasons
    assert r.inspection_validity.state == validity and r.summary.state == "INVALID_INSPECTION"


def test_wrong_part_and_mirrored_part_never_become_pass_or_review(docs):
    compiled, _, invalid = docs
    for validity in ("WRONG_PART_SUSPECTED", "MIRRORED_PART_SUSPECTED"):
        r = refuse(invalid, validity)
        many = [[[float(i), 0.0, 0.0], [float(i), 80.0, 0.0]] for i in range(0, 120, 10)]
        ov = build(r, compiled, many)
        assert ov.effective_summary_state == "INVALID_INSPECTION"
        assert {ln.decision for ln in ov.lines} == {"UNVERIFIED"}


# ------------------------------------------------------------------ option 1: unexpected bends


def test_extra_bend_on_the_right_part_raises_review(docs):
    compiled, result, _ = docs
    r = with_state(result, "ALL_MEASURED_IN_TOLERANCE")
    lines = [raw_locus(r, compiled, b.bend_uid) for b in compiled.bends] + [extra_line(r)]
    ov = build(r, compiled, lines)
    unexpected = [ln for ln in ov.lines if ln.kind == "UNEXPECTED_BEND"]
    assert len(unexpected) == 1 and unexpected[0].decision == "REVIEW"
    assert [f.code for f in ov.flags] == ["UNEXPECTED_FEATURE_REVIEW"] and ov.flags[0].line_ids == (unexpected[0].line_id,)
    assert ov.review_reasons == ("UNEXPECTED_FEATURE_REVIEW",)
    assert ov.effective_summary_state == "REVIEW_REQUIRED" and ov.verdict_lowered


def test_right_part_with_only_recipe_bends_is_not_flagged(docs):
    compiled, result, _ = docs
    r = with_state(result, "ALL_MEASURED_IN_TOLERANCE")
    # small detection noise (1.5 mm, 2 deg-ish) stays explained
    lines = [raw_locus(r, compiled, b.bend_uid, shift=(0.0, 0.0, 1.5)) for b in compiled.bends]
    ov = build(r, compiled, lines)
    assert not ov.lines and not ov.flags and ov.effective_summary_state == "ALL_MEASURED_IN_TOLERANCE"


def test_zs_lines_of_another_scan_are_refused(docs):
    from domains.production.zeroshot.overlay import build_overlay

    compiled, result, _ = docs
    doc = zs_doc(result, [extra_line(result)])
    doc["scan_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        build_overlay(result, compiled, doc, engine_result_sha256="c" * 64)
    doc = zs_doc(result, [extra_line(result)])
    doc["status"] = "FAILED"
    with pytest.raises(ValueError):
        build_overlay(result, compiled, doc, engine_result_sha256="c" * 64)


# ------------------------------------------------------------------ PDF section


def test_pdf_lists_overlay_in_its_own_unverified_section(docs):
    from html import escape

    from domains.production.reporting.labels import TEXT
    from domains.production.reporting.shopfloor_pdf import build_report_html

    compiled, result, _ = docs
    r = with_state(result, "ALL_MEASURED_IN_TOLERANCE")
    ov = build(r, compiled, [raw_locus(r, compiled, b.bend_uid) for b in compiled.bends] + [extra_line(r)])
    for lang in ("en", "he"):
        html = build_report_html(result, lang=lang, overlay=ov)
        title = escape(TEXT["sec_zs"][0 if lang == "en" else 1])
        assert title in html and escape(TEXT["zs_lowered"][0 if lang == "en" else 1]) in html
        assert html.count(">Z01<") >= 1 or "Z01" in html
        assert "#15803D" not in html.split(title, 1)[1].split("</table>", 1)[0]  # no pass green in the section
        plain = build_report_html(result, lang=lang)
        assert title not in plain  # no overlay -> no section


# ------------------------------------------------------------------ angles (angle_check/ANGLE_CHECK.md)


def _angle_char(result, uid):
    row = next(r for r in result.bends if r.bend_uid == uid)
    return next(c for c in row.characteristics if c.kind == "INCLUDED_ANGLE")


def _fail_bend(result):
    return next(r.bend_uid for r in result.bends
                if any(c.kind == "INCLUDED_ANGLE" and c.decision == "FAIL" and c.state == "MEASURED" for c in r.characteristics))


def _with_angle(zs, i, deg):
    zs["lines"][i]["incl_deg"] = deg
    return zs


def _joggled(compiled, a_uid, b_uid):
    """The CompiledPart with bends a/b turned into a joggle pair (shared panel, opposite, 30 deg, axes close)."""
    bends = {b.bend_uid: b for b in compiled.bends}
    a, b = bends[a_uid], bends[b_uid]
    near = tuple(float(x) for x in np_add(a.axis.point_mm, (3.0, 0.0, 3.0)))
    new_a = a.model_copy(update={"sweep_deg": 30.0, "direction": "UP"})
    new_b = b.model_copy(update={"sweep_deg": 30.0, "direction": "DOWN",
                                 "axis": b.axis.model_copy(update={"point_mm": near})})
    return compiled.model_copy(update={"bends": tuple(new_a if x.bend_uid == a_uid else new_b if x.bend_uid == b_uid else x
                                                      for x in compiled.bends)})


def np_add(p, d):
    return [float(x) + float(y) for x, y in zip(p, d)]


def test_joggle_detection_from_the_compiled_part(docs):
    from domains.production.zeroshot.overlay import joggle_bends

    compiled, _, _ = docs
    assert joggle_bends(compiled) == set()  # 90 deg hat bends are not joggles
    a, b = compiled.bends[0].bend_uid, compiled.bends[1].bend_uid
    assert joggle_bends(_joggled(compiled, a, b)) == {a, b}


def test_joggle_angle_is_never_shown_as_a_number(docs):
    from domains.production.zeroshot.overlay import ZeroshotOverlayV1, build_overlay

    compiled, result, _ = docs
    a, b = compiled.bends[0].bend_uid, compiled.bends[1].bend_uid
    jc = _joggled(compiled, a, b)
    r = withhold(result, a)
    lines = [raw_locus(r, compiled, x.bend_uid) for x in compiled.bends] + [extra_line(r)]
    ov = build_overlay(r, jc, zs_doc(r, lines), engine_result_sha256="c" * 64)
    gap = [ln for ln in ov.lines if ln.kind == "WITHHELD_BEND"]
    assert gap and gap[0].bend_uid == a and gap[0].incl_deg is None and gap[0].angle_status == "NOT_MEASURED_JOGGLE"
    # on a part with joggles a line of unknown bend is not shown as a number either
    assert all(ln.incl_deg is None for ln in ov.lines)
    doc = ov.model_dump(mode="json", by_alias=True)
    doc["lines"][0]["incl_deg"] = 150.0
    with pytest.raises(ValueError):
        ZeroshotOverlayV1.model_validate(doc)
    # no joggle: the zs angle is shown
    ov2 = build(r, compiled, lines)
    assert all(ln.angle_status == "SHOWN" and ln.incl_deg == 90.0 for ln in ov2.lines)


def test_independent_angle_disagreement_is_a_review_reason_only(docs):
    from domains.production.zeroshot.overlay import build_overlay

    compiled, result, _ = docs
    uid = _fail_bend(result)
    i = [b.bend_uid for b in compiled.bends].index(uid)
    e3 = float(_angle_char(result, uid).actual)
    lines = [raw_locus(result, compiled, b.bend_uid) for b in compiled.bends]
    ov = build_overlay(result, compiled, _with_angle(zs_doc(result, lines), i, e3 + 8.0), engine_result_sha256="c" * 64)
    assert [(a.code, a.bend_uid) for a in ov.angle_checks] == [("INDEPENDENT_ANGLE_DISAGREES", uid)]
    assert "INDEPENDENT_ANGLE_DISAGREES" in ov.review_reasons
    assert ov.effective_summary_state == result.summary.state == "OUT_OF_TOLERANCE_FOUND"  # the FAIL stays
    assert not ov.verdict_lowered and _angle_char(result, uid).decision == "FAIL"
    # within 5 deg: no reason
    ov = build_overlay(result, compiled, _with_angle(zs_doc(result, lines), i, e3 + 3.0), engine_result_sha256="c" * 64)
    assert not ov.angle_checks
    # on a joggle bend: never
    other = next(b.bend_uid for b in compiled.bends if b.bend_uid != uid
                 and ({b.parent_panel_id, b.child_panel_id} & {compiled.bends[i].parent_panel_id, compiled.bends[i].child_panel_id}))
    ov = build_overlay(result, _joggled(compiled, other, uid), _with_angle(zs_doc(result, lines), i, e3 + 8.0),
                       engine_result_sha256="c" * 64)
    assert not ov.angle_checks


@pytest.mark.parametrize("state", SUMMARY_STATES)
def test_angle_reasons_never_create_a_pass(docs, state):
    from domains.production.zeroshot.overlay import build_overlay

    compiled, result, _ = docs
    r = with_state(result, state)
    uid = _fail_bend(r)
    i = [b.bend_uid for b in compiled.bends].index(uid)
    lines = [raw_locus(r, compiled, b.bend_uid) for b in compiled.bends]
    ov = build_overlay(r, compiled, _with_angle(zs_doc(r, lines), i, float(_angle_char(r, uid).actual) - 9.0),
                       engine_result_sha256="c" * 64)
    assert ov.angle_checks and ov.effective_summary_state == state and not ov.verdict_lowered
